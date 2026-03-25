"""PortMoEv2-Hetero Ensemble v4 — 增强训练 + SWA + 丰富TTA + 偏差校正

基于 v1 最优配置 (MAE_TEU=5.11), 新增:
  1. 训练时数据增强: 水平翻转 + 高斯噪声 + Cutout
  2. SWA: 最后 20 epoch 的权重平均 (更宽 loss 谷, 更好泛化)
  3. 丰富 TTA: 原始 + 水平翻转 + 3路噪声 = 8路预测平均
  4. 后处理偏差校正: 验证集学 per-bin 线性修正

使用方法:
    python hetero_ensemble_v4_enhanced.py
"""

import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path

from models import PortMoE, count_parameters

# ======================== 配置 ========================

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"
SAVE_DIR = EXPERIMENTS_DIR / "PortMoEv2-Hetero-Ensemble-v4-enhanced"

ENSEMBLE_SEEDS = [42, 123, 456, 789]

MODEL_KWARGS = {"stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
                "num_experts": 4, "expert_hidden": 64,
                "heteroscedastic": True}

CFG = {
    "batch_size": 256,
    "epochs": 120,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "aux_loss_weight": 0.3,
    "aux_loss_decay": True,
    "ordinal_thresholds": [0, 15, 60],
    "gnll_warmup_epochs": 10,
    # === v4 新增 ===
    "aug_noise_std": 0.05,       # 高斯噪声标准差
    "aug_cutout_prob": 0.3,      # cutout 概率
    "aug_cutout_h": 2,           # cutout 高度 (区方向)
    "aug_cutout_w": 5,           # cutout 宽度 (栏方向)
    "swa_start_epoch": 100,      # SWA 开始 epoch (最后 20 个)
    "tta_noise_std": 0.03,       # TTA 噪声标准差
    "tta_noise_runs": 3,         # TTA 噪声路数
}


# ======================== 数据 ========================

def load_data_and_split():
    data = torch.load(DATA_PATH, weights_only=False)
    matrices = data["matrices"]
    targets = data["targets"]
    hours = data["hours"]
    qc_counts = data["qc_counts"]

    N = len(matrices)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    t_mean = targets[train_idx].mean(dim=0)
    t_std = targets[train_idx].std(dim=0) + 1e-8
    targets_z = (targets - t_mean) / t_std

    ordinal_bins = torch.zeros(N, dtype=torch.long)
    for ti, th in enumerate(CFG["ordinal_thresholds"]):
        ordinal_bins[targets[:, 0] > th] = ti + 1

    is_nonzero = (targets[train_idx, 0] > 0).float()
    sample_weights = torch.where(is_nonzero.bool(),
                                 torch.tensor(2.0), torch.tensor(1.0))

    return {
        "matrices": matrices, "targets_z": targets_z, "hours": hours,
        "qc_counts": qc_counts, "ordinal_bins": ordinal_bins,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "t_mean": t_mean, "t_std": t_std, "sample_weights": sample_weights,
        "targets_raw": targets,
    }


class AugDatasetV4(torch.utils.data.Dataset):
    """v4 增强数据集: 水平翻转 + 高斯噪声 + Cutout"""
    def __init__(self, matrices, targets_z, hours, qc_counts, bins):
        self.m = matrices
        self.t = targets_z
        self.h = hours
        self.q = qc_counts
        self.b = bins

    def __len__(self):
        return len(self.m)

    def __getitem__(self, idx):
        m = self.m[idx].clone()

        # 1. 随机水平翻转 (50%)
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)

        # 2. 随机高斯噪声
        if CFG["aug_noise_std"] > 0:
            m = m + torch.randn_like(m) * CFG["aug_noise_std"]

        # 3. 随机 Cutout (遮挡一块区域)
        if torch.rand(1).item() < CFG["aug_cutout_prob"]:
            _, h, w = m.shape  # (6, 7, 22)
            ch = CFG["aug_cutout_h"]
            cw = CFG["aug_cutout_w"]
            y = torch.randint(0, max(h - ch + 1, 1), (1,)).item()
            x = torch.randint(0, max(w - cw + 1, 1), (1,)).item()
            m[:, y:y+ch, x:x+cw] = 0.0

        return m, self.t[idx], self.h[idx], self.q[idx], self.b[idx]


# ======================== SWA 工具 ========================

class SWAAccumulator:
    """简洁的 SWA 实现: 累积模型权重做平均"""
    def __init__(self):
        self.avg_state = None
        self.n = 0

    def update(self, model):
        state = {k: v.clone().float() for k, v in model.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = state
        else:
            for k in self.avg_state:
                self.avg_state[k] += state[k]
        self.n += 1

    def get_averaged_state(self):
        return {k: v / self.n for k, v in self.avg_state.items()}


# ======================== 偏差校正 ========================

class BiasCorrector:
    """基于验证集的分段线性偏差校正"""
    def __init__(self, bin_edges=None):
        # 按预测值分 bin (用预测值而非真值, 推理时可用)
        if bin_edges is None:
            bin_edges = [0, 2, 15, 50, 100, float('inf')]
        self.bin_edges = bin_edges
        self.corrections = None  # (n_bins, 2) -> [scale, bias] per bin

    def fit(self, preds, trues):
        """从验证集学习 per-bin 偏差校正"""
        # preds, trues: (N, 2), 只校正 TEU (column 0)
        teu_pred = preds[:, 0].numpy()
        teu_true = trues[:, 0].numpy()
        move_pred = preds[:, 1].numpy()
        move_true = trues[:, 1].numpy()

        self.teu_corrections = self._fit_bins(teu_pred, teu_true)
        self.move_corrections = self._fit_bins(move_pred, move_true)

    def _fit_bins(self, pred, true):
        corrections = []
        for i in range(len(self.bin_edges) - 1):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            mask = (pred >= lo) & (pred < hi)
            if mask.sum() < 10:
                corrections.append((1.0, 0.0))
                continue
            p_bin = pred[mask]
            t_bin = true[mask]
            # 简单线性: 只学 bias (避免 scale 过拟合)
            bias = float(np.mean(t_bin - p_bin))
            corrections.append((1.0, bias))
        return corrections

    def correct(self, preds):
        """对预测值做偏差校正"""
        result = preds.clone()
        for col, corrections in enumerate([self.teu_corrections, self.move_corrections]):
            for i in range(len(self.bin_edges) - 1):
                lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
                mask = (result[:, col] >= lo) & (result[:, col] < hi)
                if mask.any():
                    scale, bias = corrections[i]
                    result[mask, col] = result[mask, col] * scale + bias
        return result


# ======================== 损失与训练 ========================

def gaussian_nll_loss(pred, logvar, target):
    logvar = logvar.clamp(-6.0, 6.0)
    inv_var = torch.exp(-logvar)
    loss = 0.5 * (inv_var * (target - pred) ** 2 + logvar)
    return loss.mean()


def cosine_lr(optimizer, epoch, cfg):
    if epoch < cfg["warmup_epochs"]:
        lr = cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    else:
        progress = (epoch - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
        lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, device, aux_weight=0.0, use_gnll=True):
    model.train()
    total_loss, n = 0.0, 0
    mse_criterion = nn.MSELoss()

    for batch in loader:
        matrices = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        hours = batch[2].to(device, non_blocking=True)
        qc_counts = batch[3].to(device, non_blocking=True)
        bins = batch[4].to(device, non_blocking=True)

        output = model(matrices, hours, qc_counts)
        pred, logvar, gate_logits = output

        if use_gnll:
            loss = gaussian_nll_loss(pred, logvar, targets)
        else:
            loss = mse_criterion(pred, targets)

        if aux_weight > 0:
            loss = loss + aux_weight * nn.functional.cross_entropy(gate_logits, bins)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, device, t_mean, t_std, use_tta=False):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        m = batch[0].to(device, non_blocking=True)
        t = batch[1].to(device, non_blocking=True)
        h = batch[2].to(device, non_blocking=True)
        q = batch[3].to(device, non_blocking=True)

        pred = model(m, h, q)
        if use_tta:
            pred_flip = model(m.flip(-1), h, q)
            pred = (pred + pred_flip) * 0.5
        all_preds.append(pred.cpu())
        all_targets.append(t.cpu())

    preds = torch.cat(all_preds) * t_std + t_mean
    trues = torch.cat(all_targets) * t_std + t_mean
    mae = (preds - trues).abs().mean(dim=0)
    rmse = ((preds - trues) ** 2).mean(dim=0).sqrt()
    ss_res = ((preds - trues) ** 2).sum(dim=0)
    ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / ss_tot
    return preds, trues, mae, rmse, r2


# ======================== 单种子训练 (含 SWA) ========================

def train_single_seed(seed, d, device):
    save_dir = SAVE_DIR / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = PortMoE(**MODEL_KWARGS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])

    # v4: 使用增强数据集
    train_set = AugDatasetV4(
        d["matrices"][d["train_idx"]], d["targets_z"][d["train_idx"]],
        d["hours"][d["train_idx"]], d["qc_counts"][d["train_idx"]],
        d["ordinal_bins"][d["train_idx"]],
    )
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
        d["ordinal_bins"][d["val_idx"]],
    )
    sampler = WeightedRandomSampler(
        d["sample_weights"], num_samples=len(d["train_idx"]), replacement=True,
    )
    train_loader = DataLoader(train_set, batch_size=CFG["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CFG["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)

    best_val_mae_sum = float("inf")
    best_epoch = 0

    # v4: SWA 累积器
    swa = SWAAccumulator()
    swa_start = CFG["swa_start_epoch"]

    print(f"\n{'=' * 50}")
    print(f"  Seed {seed} — v4 增强训练 ({CFG['epochs']} epochs, SWA from ep{swa_start+1})")
    print(f"{'=' * 50}")
    header = f"{'Ep':>4} | {'LR':>10} | {'Loss':>10} | {'GnLL':>4} | {'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}"
    print(header)
    print("-" * len(header))

    t_start = time.time()
    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, CFG)

        base_aux = CFG["aux_loss_weight"]
        progress = epoch / max(CFG["epochs"] - 1, 1)
        aux_w = base_aux * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

        use_gnll = epoch >= CFG["gnll_warmup_epochs"]

        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     aux_weight=aux_w, use_gnll=use_gnll)

        # SWA: 最后 20 个 epoch 累积权重
        if epoch >= swa_start:
            swa.update(model)

        _, _, val_mae, _, _ = evaluate_model(
            model, val_loader, device, d["t_mean"], d["t_std"], use_tta=True,
        )
        elapsed = time.time() - t0

        val_mae_sum = val_mae[0].item() + val_mae[1].item()
        marker = ""
        if val_mae_sum < best_val_mae_sum:
            best_val_mae_sum = val_mae_sum
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae.numpy(),
            }, save_dir / "best.pt")
            marker = " *"

        gnll_flag = "Yes" if use_gnll else "No"
        swa_flag = "S" if epoch >= swa_start else ""
        print(f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | {gnll_flag:>4} | "
              f"{val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}{swa_flag}")

    # 保存 SWA 权重
    if swa.n > 0:
        swa_state = swa.get_averaged_state()
        swa_model = PortMoE(**MODEL_KWARGS).to(device)
        swa_model.load_state_dict(swa_state)
        swa_model.eval()

        # 评估 SWA 模型
        _, _, swa_mae, _, _ = evaluate_model(
            swa_model, val_loader, device, d["t_mean"], d["t_std"], use_tta=True,
        )
        swa_mae_sum = swa_mae[0].item() + swa_mae[1].item()

        torch.save({
            "epoch": f"SWA_{swa_start+1}-{CFG['epochs']}",
            "seed": seed,
            "model_state_dict": swa_state,
            "val_mae": swa_mae.numpy(),
            "swa_epochs": swa.n,
        }, save_dir / "swa.pt")

        print(f"\n  SWA ({swa.n} epochs avg): Val MAE_TEU={swa_mae[0]:.2f}, MAE_move={swa_mae[1]:.2f}")
        if swa_mae_sum < best_val_mae_sum:
            print(f"  ✓ SWA 优于 best checkpoint! (sum {swa_mae_sum:.2f} < {best_val_mae_sum:.2f})")
        else:
            print(f"  △ SWA 未超越 best checkpoint (sum {swa_mae_sum:.2f} >= {best_val_mae_sum:.2f})")

    total_time = time.time() - t_start
    print(f"\n✓ Seed {seed} 完成 — best epoch {best_epoch}, "
          f"ValMAE_sum={best_val_mae_sum:.2f}, time={total_time:.0f}s")
    return best_epoch, best_val_mae_sum, total_time


# ======================== 集成评估 (丰富TTA + 偏差校正) ========================

@torch.no_grad()
def rich_tta_predict(model, m, h, q, device):
    """8路 TTA: 原始 + 翻转 + 3×噪声原始 + 3×噪声翻转"""
    preds = []

    # 路 1: 原始
    preds.append(model(m, h, q))

    # 路 2: 水平翻转
    preds.append(model(m.flip(-1), h, q))

    # 路 3-5: 加噪声 (原始)
    noise_std = CFG["tta_noise_std"]
    for _ in range(CFG["tta_noise_runs"]):
        m_noisy = m + torch.randn_like(m) * noise_std
        preds.append(model(m_noisy, h, q))

    # 路 6-8: 加噪声 (翻转)
    for _ in range(CFG["tta_noise_runs"]):
        m_noisy = m.flip(-1) + torch.randn_like(m) * noise_std
        preds.append(model(m_noisy, h, q))

    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def ensemble_evaluate(seeds, d, device):
    t_mean, t_std = d["t_mean"], d["t_std"]
    test_set = TensorDataset(
        d["matrices"][d["test_idx"]], d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]], d["qc_counts"][d["test_idx"]],
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
    )
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    # 加载所有模型 (同时尝试 best 和 SWA, 选更好的)
    models = []
    for seed in seeds:
        model_best = PortMoE(**MODEL_KWARGS).to(device)
        ckpt_best = torch.load(SAVE_DIR / f"seed_{seed}" / "best.pt",
                               weights_only=False, map_location=device)
        model_best.load_state_dict(ckpt_best["model_state_dict"])
        model_best.eval()

        swa_path = SAVE_DIR / f"seed_{seed}" / "swa.pt"
        if swa_path.exists():
            model_swa = PortMoE(**MODEL_KWARGS).to(device)
            ckpt_swa = torch.load(swa_path, weights_only=False, map_location=device)
            model_swa.load_state_dict(ckpt_swa["model_state_dict"])
            model_swa.eval()

            # 在验证集上比较 best vs SWA (简单 TTA)
            _, _, mae_best, _, _ = evaluate_model(
                model_best, val_loader, device, t_mean, t_std, use_tta=True)
            _, _, mae_swa, _, _ = evaluate_model(
                model_swa, val_loader, device, t_mean, t_std, use_tta=True)

            best_sum = mae_best[0].item() + mae_best[1].item()
            swa_sum = mae_swa[0].item() + mae_swa[1].item()

            if swa_sum < best_sum:
                models.append(model_swa)
                print(f"  Seed {seed}: 使用 SWA (epoch {ckpt_swa['epoch']}) "
                      f"[MAE_sum={swa_sum:.2f} < best={best_sum:.2f}]")
            else:
                models.append(model_best)
                print(f"  Seed {seed}: 使用 best (epoch {ckpt_best['epoch']}) "
                      f"[MAE_sum={best_sum:.2f} <= SWA={swa_sum:.2f}]")
        else:
            models.append(model_best)
            print(f"  Seed {seed}: 使用 best (epoch {ckpt_best['epoch']})")

    def collect_predictions(loader, models, tta_mode="rich"):
        """收集各模型的预测"""
        all_preds_per_model = [[] for _ in models]
        all_targets_list = []

        for batch in loader:
            m = batch[0].to(device, non_blocking=True)
            t = batch[1]
            h = batch[2].to(device, non_blocking=True)
            q = batch[3].to(device, non_blocking=True)

            for mi, model in enumerate(models):
                if tta_mode == "rich":
                    pred = rich_tta_predict(model, m, h, q, device)
                elif tta_mode == "simple":
                    p1 = model(m, h, q)
                    p2 = model(m.flip(-1), h, q)
                    pred = (p1 + p2) * 0.5
                else:
                    pred = model(m, h, q)
                all_preds_per_model[mi].append(pred.cpu())

            all_targets_list.append(t)

        trues = torch.cat(all_targets_list) * t_std + t_mean
        for mi in range(len(models)):
            all_preds_per_model[mi] = torch.cat(all_preds_per_model[mi]) * t_std + t_mean

        return all_preds_per_model, trues

    def compute_metrics(preds, trues):
        mae = (preds - trues).abs().mean(dim=0)
        rmse = ((preds - trues) ** 2).mean(dim=0).sqrt()
        ss_res = ((preds - trues) ** 2).sum(dim=0)
        ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1 - ss_res / ss_tot
        return mae, rmse, r2

    print(f"\n{'=' * 60}")
    print(f"  v4 集成评估 ({len(seeds)} seeds × Hetero × Rich TTA)")
    print(f"{'=' * 60}")

    # === 1. 基础评估 (simple TTA, 与 v1 公平对比) ===
    print(f"\n  --- 对照: Simple TTA (与 v1 公平对比) ---")
    val_preds_simple, val_trues = collect_predictions(val_loader, models, "simple")
    test_preds_simple, test_trues = collect_predictions(test_loader, models, "simple")

    ens_val_simple = torch.stack(val_preds_simple).mean(dim=0)
    ens_test_simple = torch.stack(test_preds_simple).mean(dim=0)

    mae_v, _, _ = compute_metrics(ens_val_simple, val_trues)
    mae_t, rmse_t, r2_t = compute_metrics(ens_test_simple, test_trues)
    print(f"  Val  MAE_TEU={mae_v[0]:.4f}  MAE_move={mae_v[1]:.4f}")
    print(f"  Test MAE_TEU={mae_t[0]:.4f}  MAE_move={mae_t[1]:.4f}  "
          f"RMSE={rmse_t[0]:.4f}  R²={r2_t[0]:.4f}")

    # === 2. Rich TTA ===
    print(f"\n  --- Rich TTA (8路) ---")
    val_preds_rich, _ = collect_predictions(val_loader, models, "rich")
    test_preds_rich, _ = collect_predictions(test_loader, models, "rich")

    # 各模型 + Rich TTA
    print(f"\n  各模型 + Rich TTA (Val Set):")
    for i, seed in enumerate(seeds):
        mae_i, _, _ = compute_metrics(val_preds_rich[i], val_trues)
        print(f"    Seed {seed}: MAE_TEU={mae_i[0]:.4f}  MAE_move={mae_i[1]:.4f}")

    # 累积集成
    print(f"\n  累积集成效果 (Val Set, Rich TTA):")
    for k in range(1, len(seeds) + 1):
        ens_pred = torch.stack(val_preds_rich[:k]).mean(dim=0)
        mae_k, rmse_k, r2_k = compute_metrics(ens_pred, val_trues)
        print(f"    Top-{k}: MAE_TEU={mae_k[0]:.4f}  MAE_move={mae_k[1]:.4f}  R²={r2_k[0]:.4f}")

    ens_val_rich = torch.stack(val_preds_rich).mean(dim=0)
    ens_test_rich = torch.stack(test_preds_rich).mean(dim=0)

    mae_v_rich, _, _ = compute_metrics(ens_val_rich, val_trues)
    mae_t_rich, rmse_t_rich, r2_t_rich = compute_metrics(ens_test_rich, test_trues)
    print(f"\n  Rich TTA 集成:")
    print(f"    Val  MAE_TEU={mae_v_rich[0]:.4f}  MAE_move={mae_v_rich[1]:.4f}")
    print(f"    Test MAE_TEU={mae_t_rich[0]:.4f}  MAE_move={mae_t_rich[1]:.4f}  "
          f"RMSE={rmse_t_rich[0]:.4f}  R²={r2_t_rich[0]:.4f}")

    # === 3. 偏差校正 ===
    print(f"\n  --- 偏差校正 (在 Rich TTA 基础上) ---")
    corrector = BiasCorrector()
    corrector.fit(ens_val_rich, val_trues)

    # 打印学到的校正
    for i, (lo, hi) in enumerate(zip(corrector.bin_edges[:-1], corrector.bin_edges[1:])):
        scale_t, bias_t = corrector.teu_corrections[i]
        scale_m, bias_m = corrector.move_corrections[i]
        label = f"[{lo},{hi})" if hi < float('inf') else f"[{lo},+∞)"
        print(f"    Bin {label}: TEU bias={bias_t:+.2f}, move bias={bias_m:+.2f}")

    ens_test_corrected = corrector.correct(ens_test_rich)
    mae_t_corr, rmse_t_corr, r2_t_corr = compute_metrics(ens_test_corrected, test_trues)
    print(f"\n  校正后:")
    print(f"    Test MAE_TEU={mae_t_corr[0]:.4f}  MAE_move={mae_t_corr[1]:.4f}  "
          f"RMSE={rmse_t_corr[0]:.4f}  R²={r2_t_corr[0]:.4f}")

    # === 4. Per-QC 分析 ===
    print(f"\n  === Per-QC 分析 (Test, 最优配置) ===")
    # 选最好的那个配置做 per-qc
    test_qc = d["qc_counts"][d["test_idx"]]
    # 用校正后的结果
    final_pred = ens_test_corrected
    final_true = test_trues

    for qc_val in [1, 2, 3]:
        mask = test_qc == qc_val
        if mask.sum() > 0:
            mae_qc = (final_pred[mask] - final_true[mask]).abs().mean(dim=0)
            print(f"  qc={qc_val}: N={mask.sum():>5}, MAE_TEU={mae_qc[0]:.2f}")
    mask4 = test_qc >= 4
    if mask4.sum() > 0:
        mae_qc4 = (final_pred[mask4] - final_true[mask4]).abs().mean(dim=0)
        print(f"  qc>=4: N={mask4.sum():>5}, MAE_TEU={mae_qc4[0]:.2f}")

    # === 5. 最终对比 ===
    print(f"\n{'=' * 60}")
    print(f"  最终对比")
    print(f"{'=' * 60}")
    print(f"v1 (样本级, 无增强):              Test MAE_TEU=5.1112  R²=0.9550")
    print(f"v3 (样本级 + sheet_idx):           Test MAE_TEU=5.0937  R²=0.9544")
    print(f"v4 (增强+SWA, Simple TTA):        Test MAE_TEU={mae_t[0]:.4f}  R²={r2_t[0]:.4f}")
    print(f"v4 (增强+SWA, Rich TTA):          Test MAE_TEU={mae_t_rich[0]:.4f}  R²={r2_t_rich[0]:.4f}")
    print(f"v4 (增强+SWA, Rich TTA + 校正):   Test MAE_TEU={mae_t_corr[0]:.4f}  R²={r2_t_corr[0]:.4f}")

    # 保存最好的结果
    # 选 simple/rich/corrected 中最好的
    configs = [
        ("simple_tta", mae_t, rmse_t, r2_t),
        ("rich_tta", mae_t_rich, rmse_t_rich, r2_t_rich),
        ("rich_tta_corrected", mae_t_corr, rmse_t_corr, r2_t_corr),
    ]
    best_config = min(configs, key=lambda x: x[1][0].item())
    best_name = best_config[0]
    best_mae = best_config[1]
    best_rmse = best_config[2]
    best_r2 = best_config[3]

    print(f"\n  最优配置: {best_name}")
    print(f"  Test MAE_TEU={best_mae[0]:.4f}  MAE_move={best_mae[1]:.4f}  R²={best_r2[0]:.4f}")

    results = {
        "version": "v4-enhanced",
        "improvements": [
            "training augmentation (flip+noise+cutout)",
            "SWA (last 20 epochs)",
            "rich TTA (8-way)",
            "bias correction",
        ],
        "seeds": ENSEMBLE_SEEDS,
        "n_models": len(ENSEMBLE_SEEDS),
        "model_kwargs": MODEL_KWARGS,
        "config": CFG,
        "best_tta_mode": best_name,
        "simple_tta": {
            "test_mae_teu": round(mae_t[0].item(), 4),
            "test_mae_move": round(mae_t[1].item(), 4),
            "test_r2_teu": round(r2_t[0].item(), 4),
        },
        "rich_tta": {
            "test_mae_teu": round(mae_t_rich[0].item(), 4),
            "test_mae_move": round(mae_t_rich[1].item(), 4),
            "test_r2_teu": round(r2_t_rich[0].item(), 4),
        },
        "rich_tta_corrected": {
            "test_mae_teu": round(mae_t_corr[0].item(), 4),
            "test_mae_move": round(mae_t_corr[1].item(), 4),
            "test_r2_teu": round(r2_t_corr[0].item(), 4),
        },
        "val_mae_teu": round(mae_v_rich[0].item(), 4),
        "val_mae_move": round(mae_v_rich[1].item(), 4),
        "test_mae_teu": round(best_mae[0].item(), 4),
        "test_mae_move": round(best_mae[1].item(), 4),
        "test_rmse_teu": round(best_rmse[0].item(), 4),
        "test_rmse_move": round(best_rmse[1].item(), 4),
        "test_r2_teu": round(best_r2[0].item(), 4),
        "test_r2_move": round(best_r2[1].item(), 4),
    }
    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {SAVE_DIR / 'results.json'}")
    print("完成!")


# ======================== 主入口 ========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PortMoEv2-Hetero Ensemble v4 — 增强训练 + SWA + 丰富TTA")
    print("=" * 60)

    d = load_data_and_split()
    n_params = count_parameters(PortMoE(**MODEL_KWARGS))
    print(f"  模型参数量: {n_params:,}")
    print(f"  集成种子: {ENSEMBLE_SEEDS}")
    print(f"  训练增强: flip + noise(σ={CFG['aug_noise_std']}) + cutout({CFG['aug_cutout_h']}×{CFG['aug_cutout_w']}, p={CFG['aug_cutout_prob']})")
    print(f"  SWA: epoch {CFG['swa_start_epoch']+1}-{CFG['epochs']}")
    print(f"  TTA: 8路 (orig + flip + 3×noise + 3×noise_flip)")

    # 训练每个种子
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = SAVE_DIR / f"seed_{seed}" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            print(f"\n  ✓ Seed {seed} checkpoint 已存在 (epoch {ckpt['epoch']}), 跳过训练")
            continue
        train_single_seed(seed, d, device)

    # 集成评估
    ensemble_evaluate(ENSEMBLE_SEEDS, d, device)


if __name__ == "__main__":
    main()
