"""消融实验: 样本级划分 + sheet_index

与 v1 相同的样本级划分(有泄露)，但加了 sheet_index 特征。
用于对比 v1(5.11) 判断 sheet_index 的纯粹增益。

使用方法:
    python hetero_ensemble_v3_ablation.py
"""

import json
import time
from collections import defaultdict
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
SAVE_DIR = EXPERIMENTS_DIR / "PortMoEv2-Hetero-Ensemble-v3-ablation"

ENSEMBLE_SEEDS = [42, 123, 456, 789]

MODEL_KWARGS = {"stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
                "num_experts": 4, "expert_hidden": 64,
                "heteroscedastic": True,
                "use_sheet_cond": True, "max_sheet_idx": 256}

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
}


# ======================== 数据 ========================

def compute_sheet_indices(file_ids):
    """从 file_ids 计算每个样本在其文件内的快照序号 (0-based)"""
    file_to_indices = defaultdict(list)
    for i, fid in enumerate(file_ids.numpy()):
        file_to_indices[fid].append(i)

    sheet_indices = torch.zeros(len(file_ids), dtype=torch.long)
    for fid, indices in file_to_indices.items():
        for pos, idx in enumerate(indices):
            sheet_indices[idx] = pos
    return sheet_indices


def load_data_and_split():
    data = torch.load(DATA_PATH, weights_only=False)
    matrices = data["matrices"]
    targets = data["targets"]
    hours = data["hours"]
    qc_counts = data["qc_counts"]
    file_ids = data["file_ids"]

    # 计算 sheet_index
    sheet_indices = compute_sheet_indices(file_ids)
    max_si = sheet_indices.max().item()
    print(f"  sheet_index 范围: 0 ~ {max_si}")

    # ---- 样本级划分 (与 v1 完全相同，用于消融对比) ----
    N = len(matrices)
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    print(f"  样本级划分: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

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
        "sheet_indices": sheet_indices,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "t_mean": t_mean, "t_std": t_std, "sample_weights": sample_weights,
        "targets_raw": targets,
    }


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, matrices, targets_z, hours, qc_counts, bins, sheet_indices):
        self.m = matrices
        self.t = targets_z
        self.h = hours
        self.q = qc_counts
        self.b = bins
        self.s = sheet_indices

    def __len__(self):
        return len(self.m)

    def __getitem__(self, idx):
        m = self.m[idx]
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)
        return m, self.t[idx], self.h[idx], self.q[idx], self.b[idx], self.s[idx]


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
        sheet_idx = batch[5].to(device, non_blocking=True)

        output = model(matrices, hours, qc_counts, sheet_idx)
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
        s = batch[4].to(device, non_blocking=True)

        pred = model(m, h, q, s)
        if use_tta:
            pred_flip = model(m.flip(-1), h, q, s)
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


# ======================== 单种子训练 ========================

def train_single_seed(seed, d, device):
    save_dir = SAVE_DIR / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = PortMoE(**MODEL_KWARGS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])

    train_set = AugDataset(
        d["matrices"][d["train_idx"]], d["targets_z"][d["train_idx"]],
        d["hours"][d["train_idx"]], d["qc_counts"][d["train_idx"]],
        d["ordinal_bins"][d["train_idx"]], d["sheet_indices"][d["train_idx"]],
    )
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
        d["sheet_indices"][d["val_idx"]],
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

    print(f"\n{'=' * 50}")
    print(f"  Seed {seed} — 消融: 样本级+sheet_idx ({CFG['epochs']} epochs)")
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
        print(f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | {gnll_flag:>4} | "
              f"{val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}")

    total_time = time.time() - t_start
    print(f"\n✓ Seed {seed} 完成 — best epoch {best_epoch}, "
          f"ValMAE_sum={best_val_mae_sum:.2f}, time={total_time:.0f}s")
    return best_epoch, best_val_mae_sum, total_time


# ======================== 集成评估 ========================

@torch.no_grad()
def ensemble_evaluate(seeds, d, device):
    t_mean, t_std = d["t_mean"], d["t_std"]
    test_set = TensorDataset(
        d["matrices"][d["test_idx"]], d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]], d["qc_counts"][d["test_idx"]],
        d["sheet_indices"][d["test_idx"]],
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
        d["sheet_indices"][d["val_idx"]],
    )
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    # 加载所有模型
    models = []
    for seed in seeds:
        model = PortMoE(**MODEL_KWARGS).to(device)
        ckpt_path = SAVE_DIR / f"seed_{seed}" / "best.pt"
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded seed={seed} (epoch {ckpt['epoch']})")

    def eval_ensemble(loader, label):
        all_preds_per_model = []
        all_targets_list = []
        targets_collected = False

        for mi, model in enumerate(models):
            preds_list = []
            for batch in loader:
                m = batch[0].to(device, non_blocking=True)
                t = batch[1]
                h = batch[2].to(device, non_blocking=True)
                q = batch[3].to(device, non_blocking=True)
                s = batch[4].to(device, non_blocking=True)

                # TTA
                p1 = model(m, h, q, s)
                p2 = model(m.flip(-1), h, q, s)
                pred = (p1 + p2) * 0.5
                preds_list.append(pred.cpu())

                if not targets_collected:
                    all_targets_list.append(t)

            targets_collected = True
            all_preds_per_model.append(torch.cat(preds_list))

        trues = torch.cat(all_targets_list) * t_std + t_mean

        # 逐个与累积
        print(f"\n  === {label}: 各模型 + TTA ===")
        for i, seed in enumerate(seeds):
            p = all_preds_per_model[i] * t_std + t_mean
            mae = (p - trues).abs().mean(dim=0)
            print(f"  Seed {seed:>4} + TTA: MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}")

        # 累积集成
        print(f"\n  === {label}: 累积集成效果 ===")
        for k in range(1, len(seeds) + 1):
            ens_pred = torch.stack(all_preds_per_model[:k]).mean(dim=0) * t_std + t_mean
            mae = (ens_pred - trues).abs().mean(dim=0)
            rmse = ((ens_pred - trues) ** 2).mean(dim=0).sqrt()
            ss_res = ((ens_pred - trues) ** 2).sum(dim=0)
            ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
            r2 = 1 - ss_res / ss_tot
            label_k = f"Top-{k} seeds"
            print(f"  {label_k}: MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}  "
                  f"RMSE_TEU={rmse[0]:.4f}  R²_TEU={r2[0]:.4f}")

        # 全集成结果
        ens_pred = torch.stack(all_preds_per_model).mean(dim=0) * t_std + t_mean
        mae = (ens_pred - trues).abs().mean(dim=0)
        rmse = ((ens_pred - trues) ** 2).mean(dim=0).sqrt()
        ss_res = ((ens_pred - trues) ** 2).sum(dim=0)
        ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1 - ss_res / ss_tot
        return mae, rmse, r2

    print(f"\n{'=' * 60}")
    print(f"  集成评估 ({len(seeds)} seeds × Hetero × TTA)")
    print(f"{'=' * 60}")

    val_mae, val_rmse, val_r2 = eval_ensemble(val_loader, "Val Set")
    test_mae, test_rmse, test_r2 = eval_ensemble(test_loader, "Test Set")

    # Per-QC 分析
    qc_test = d["qc_counts"][d["test_idx"]].numpy()
    t_test_raw = d["targets_raw"][d["test_idx"]]

    # 重新获取 ensemble 预测
    all_preds = []
    for model in models:
        preds_list = []
        for batch in test_loader:
            m = batch[0].to(device, non_blocking=True)
            h = batch[2].to(device, non_blocking=True)
            q = batch[3].to(device, non_blocking=True)
            s = batch[4].to(device, non_blocking=True)
            p1 = model(m, h, q, s)
            p2 = model(m.flip(-1), h, q, s)
            preds_list.append(((p1 + p2) * 0.5).cpu())
        all_preds.append(torch.cat(preds_list))
    ens_pred = torch.stack(all_preds).mean(dim=0) * t_std + t_mean

    print(f"\n  === Per-QC 分析 (Test) ===")
    for qc_val in [1, 2, 3]:
        mask = qc_test == qc_val
        if mask.sum() > 0:
            mae_q = (ens_pred[mask] - t_test_raw[mask]).abs().mean(dim=0)
            print(f"  qc={qc_val}: N={mask.sum():5d}, MAE_TEU={mae_q[0]:.2f}")
    mask_ge4 = qc_test >= 4
    if mask_ge4.sum() > 0:
        mae_q = (ens_pred[mask_ge4] - t_test_raw[mask_ge4]).abs().mean(dim=0)
        print(f"  qc>=4: N={mask_ge4.sum():5d}, MAE_TEU={mae_q[0]:.2f}")

    # 最终对比
    print(f"\n{'=' * 60}")
    print(f"  最终对比")
    print(f"{'=' * 60}")
    print(f"v1 (样本级, 无sheet_idx):        Test MAE_TEU=5.1112  R²=0.9550")
    print(f"v2 (文件级 + sheet_idx):          Test MAE_TEU=6.4616  R²=0.9334")
    print(f"v3 (样本级 + sheet_idx, 消融):    Test MAE_TEU={test_mae[0]:.4f}  "
          f"MAE_move={test_mae[1]:.4f}  R²={test_r2[0]:.4f}")

    results = {
        "version": "v2",
        "improvements": ["file-level split", "sheet_index conditioning"],
        "seeds": ENSEMBLE_SEEDS,
        "n_models": len(ENSEMBLE_SEEDS),
        "model_kwargs": {k: v for k, v in MODEL_KWARGS.items()},
        "config": CFG,
        "val_mae_teu": round(val_mae[0].item(), 4),
        "val_mae_move": round(val_mae[1].item(), 4),
        "test_mae_teu": round(test_mae[0].item(), 4),
        "test_mae_move": round(test_mae[1].item(), 4),
        "test_rmse_teu": round(test_rmse[0].item(), 4),
        "test_rmse_move": round(test_rmse[1].item(), 4),
        "test_r2_teu": round(test_r2[0].item(), 4),
        "test_r2_move": round(test_r2[1].item(), 4),
    }
    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {SAVE_DIR / 'results.json'}")
    print("完成!")


# ======================== 主入口 ========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  消融实验: 样本级划分 + sheet_index")
    print("  对比 v1(5.11) 验证 sheet_index 的纯粹增益")
    print("=" * 60)

    d = load_data_and_split()
    n_params = count_parameters(PortMoE(**MODEL_KWARGS))
    print(f"  模型参数量: {n_params:,}")
    print(f"  集成种子: {ENSEMBLE_SEEDS}")

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
