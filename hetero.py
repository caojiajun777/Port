"""PortMoEv2-Hetero: 异方差高斯损失实验

核心思路：让模型同时预测均值 μ 和 log 方差 log σ²。
- 损失: L = 0.5 * (exp(-log σ²) * (y - μ)² + log σ²) + aux_loss
- qc=1 高噪声样本 → 模型学到大 σ² → 自动降低梯度贡献
- qc≥2 可预测样本 → 模型学到小 σ² → 聚焦拟合
- 推理时只用 μ，与 TTA/集成完全兼容

使用方法:
    python hetero.py
"""

import json
import time
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
SAVE_DIR = EXPERIMENTS_DIR / "PortMoEv2-Hetero"

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
    "gnll_warmup_epochs": 10,  # 前 N epochs 用 MSE，之后切换到 GaussianNLL
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


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, matrices, targets_z, hours, qc_counts, bins):
        self.m, self.t, self.h, self.q, self.b = matrices, targets_z, hours, qc_counts, bins

    def __len__(self):
        return len(self.m)

    def __getitem__(self, idx):
        m = self.m[idx]
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)
        return m, self.t[idx], self.h[idx], self.q[idx], self.b[idx]


# ======================== 异方差高斯 NLL 损失 ========================

def gaussian_nll_loss(pred, logvar, target):
    """
    L = 0.5 * (exp(-logvar) * (target - pred)² + logvar)
    logvar 夹紧到 [-6, 6] 保证数值稳定
    """
    logvar = logvar.clamp(-6.0, 6.0)
    inv_var = torch.exp(-logvar)
    loss = 0.5 * (inv_var * (target - pred) ** 2 + logvar)
    return loss.mean()


# ======================== 训练 ========================

def cosine_lr(optimizer, epoch, cfg):
    if epoch < cfg["warmup_epochs"]:
        lr = cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    else:
        progress = (epoch - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
        lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, device, aux_weight=0.0,
                    use_gnll=True):
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

        pred = model(m, h, q)  # eval mode → only returns mean
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


@torch.no_grad()
def analyze_uncertainty(model, loader, device, t_mean, t_std, qc_counts_raw):
    """分析模型学到的不确定性与 qc_count 的关系"""
    model.train()  # 需要 train mode 以获取 logvar
    all_logvar, all_ae, all_qc = [], [], []

    for batch in loader:
        m = batch[0].to(device, non_blocking=True)
        t = batch[1].to(device, non_blocking=True)
        h = batch[2].to(device, non_blocking=True)
        q = batch[3].to(device, non_blocking=True)

        pred, logvar, _ = model(m, h, q)
        pred_real = pred.cpu() * t_std + t_mean
        t_real = t.cpu() * t_std + t_mean
        ae = (pred_real - t_real).abs()

        all_logvar.append(logvar.cpu())
        all_ae.append(ae)
        all_qc.append(q.cpu())

    logvar = torch.cat(all_logvar)
    ae = torch.cat(all_ae)
    qc = torch.cat(all_qc)

    # 用 TEU 通道分析
    sigma = torch.exp(0.5 * logvar[:, 0])  # 标准差（z-score 空间）
    sigma_real = sigma * t_std[0]  # 换算到原始空间

    print("\n  === 学到的不确定性 vs qc_count ===")
    print(f"  {'qc':>4} | {'n':>6} | {'mean σ':>8} | {'MAE_TEU':>8}")
    print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
    for qc_val in [1, 2, 3]:
        mask = qc == qc_val
        if mask.sum() > 0:
            print(f"  {qc_val:>4} | {mask.sum():>6} | {sigma_real[mask].mean():>8.2f} | "
                  f"{ae[mask, 0].mean():>8.2f}")
    mask_ge4 = qc >= 4
    if mask_ge4.sum() > 0:
        print(f"  {'≥4':>4} | {mask_ge4.sum():>6} | {sigma_real[mask_ge4].mean():>8.2f} | "
              f"{ae[mask_ge4, 0].mean():>8.2f}")

    model.eval()


# ======================== 主函数 ========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PortMoEv2-Hetero: 异方差高斯损失实验")
    print("=" * 60)

    d = load_data_and_split()

    model = PortMoE(**MODEL_KWARGS).to(device)
    n_params = count_parameters(model)
    print(f"  参数量: {n_params:,}")
    print(f"  GaussianNLL warmup: {CFG['gnll_warmup_epochs']} epochs (先用 MSE)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])

    train_set = AugDataset(
        d["matrices"][d["train_idx"]], d["targets_z"][d["train_idx"]],
        d["hours"][d["train_idx"]], d["qc_counts"][d["train_idx"]],
        d["ordinal_bins"][d["train_idx"]],
    )
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
        d["ordinal_bins"][d["val_idx"]],
    )
    test_set = TensorDataset(
        d["matrices"][d["test_idx"]], d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]], d["qc_counts"][d["test_idx"]],
        d["ordinal_bins"][d["test_idx"]],
    )

    sampler = WeightedRandomSampler(
        d["sample_weights"], num_samples=len(d["train_idx"]), replacement=True,
    )
    train_loader = DataLoader(train_set, batch_size=CFG["batch_size"],
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CFG["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=CFG["batch_size"],
                             shuffle=False, num_workers=0, pin_memory=True)

    best_val_mae_sum = float("inf")
    best_epoch = 0

    header = f"{'Ep':>4} | {'LR':>10} | {'Loss':>10} | {'GnLL':>4} | {'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}"
    print(header)
    print("-" * len(header))

    t_start = time.time()
    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, CFG)

        # 辅助损失权重余弦衰减
        base_aux = CFG["aux_loss_weight"]
        progress = epoch / max(CFG["epochs"] - 1, 1)
        aux_w = base_aux * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))

        # GaussianNLL warmup: 前 N epochs 用 MSE 让模型先学到基本预测
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
                "model_state_dict": model.state_dict(),
                "model_kwargs": MODEL_KWARGS,
                "val_mae": val_mae.numpy(),
            }, SAVE_DIR / "best.pt")
            marker = " *"

        gnll_flag = "Yes" if use_gnll else "No"
        print(f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | {gnll_flag:>4} | "
              f"{val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}")

    total_time = time.time() - t_start
    print(f"\n✓ 训练完成 — best epoch {best_epoch}, time={total_time:.0f}s")

    # ======================== 评估 ========================
    print(f"\n{'=' * 60}")
    print("  加载 best checkpoint 评估")
    print(f"{'=' * 60}")

    ckpt = torch.load(SAVE_DIR / "best.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 不确定性分析
    analyze_uncertainty(model, test_loader, device, d["t_mean"], d["t_std"],
                        d["qc_counts"][d["test_idx"]])

    # 测试集评估
    _, _, test_mae, test_rmse, test_r2 = evaluate_model(
        model, test_loader, device, d["t_mean"], d["t_std"], use_tta=False,
    )
    _, _, test_mae_tta, test_rmse_tta, test_r2_tta = evaluate_model(
        model, test_loader, device, d["t_mean"], d["t_std"], use_tta=True,
    )

    print(f"\n  === 测试集结果 ===")
    print(f"  无 TTA:  MAE_TEU={test_mae[0]:.4f}  MAE_move={test_mae[1]:.4f}  "
          f"RMSE_TEU={test_rmse[0]:.4f}  R²={test_r2[0]:.4f}")
    print(f"  有 TTA:  MAE_TEU={test_mae_tta[0]:.4f}  MAE_move={test_mae_tta[1]:.4f}  "
          f"RMSE_TEU={test_rmse_tta[0]:.4f}  R²={test_r2_tta[0]:.4f}")

    # 对比基线
    print(f"\n  === 对比 ===")
    baseline_no_tta = 6.8049
    baseline_tta = 6.2752
    print(f"  PortMoEv2 基线:      MAE_TEU={baseline_no_tta:.4f}")
    print(f"  PortMoEv2+TTA 基线:  MAE_TEU={baseline_tta:.4f}")
    print(f"  Hetero 无 TTA:       MAE_TEU={test_mae[0]:.4f}  "
          f"({'%.1f' % ((1 - test_mae[0]/baseline_no_tta)*100)}%)")
    print(f"  Hetero + TTA:        MAE_TEU={test_mae_tta[0]:.4f}  "
          f"({'%.1f' % ((1 - test_mae_tta[0]/baseline_tta)*100)}%)")

    # 保存结果
    results = {
        "model_kwargs": MODEL_KWARGS,
        "config": CFG,
        "best_epoch": best_epoch,
        "params": n_params,
        "test_no_tta": {
            "mae_teu": round(test_mae[0].item(), 4),
            "mae_move": round(test_mae[1].item(), 4),
            "rmse_teu": round(test_rmse[0].item(), 4),
            "r2_teu": round(test_r2[0].item(), 4),
        },
        "test_tta": {
            "mae_teu": round(test_mae_tta[0].item(), 4),
            "mae_move": round(test_mae_tta[1].item(), 4),
            "rmse_teu": round(test_rmse_tta[0].item(), 4),
            "r2_teu": round(test_r2_tta[0].item(), 4),
        },
    }
    with open(SAVE_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {SAVE_DIR / 'results.json'}")
    print("完成!")


if __name__ == "__main__":
    main()
