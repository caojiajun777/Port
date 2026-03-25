"""时序窗口消融实验 — 文件级划分

对 k=1,2,3,4,5 分别训练 4-seed 集成，评估时序窗口长度的边际效用。
k=3 的文件级结果已由 v6 提供，此脚本会自动跳过已完成的 k。

使用方法:
    python ablation_temporal_k.py          # 依次运行所有 k
    python ablation_temporal_k.py --k 1    # 只运行 k=1
    python ablation_temporal_k.py --k 2 4 5  # 运行指定 k
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Dataset
from pathlib import Path

from models import PortMoE, count_parameters

# ======================== 配置 ========================

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"

ENSEMBLE_SEEDS = [42, 123, 456, 789]

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


def get_model_kwargs(k):
    return {
        "in_channels": 6 * k,
        "stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
        "num_experts": 4, "expert_hidden": 64,
        "heteroscedastic": True,
    }


def get_save_dir(k):
    return EXPERIMENTS_DIR / f"ablation-temporal-k{k}"


# ======================== 时序数据构建 ========================

def build_temporal_matrices(all_matrices, file_ids, k):
    N, C, H, W = all_matrices.shape
    if k == 1:
        return all_matrices.clone(), 0, len(torch.unique(file_ids))

    temporal = torch.zeros(N, C * k, H, W)
    file_to_indices = {}
    for i in range(N):
        fid = file_ids[i].item()
        if fid not in file_to_indices:
            file_to_indices[fid] = []
        file_to_indices[fid].append(i)

    n_padded = 0
    for fid, indices in file_to_indices.items():
        for pos, idx in enumerate(indices):
            if pos < k - 1:
                n_padded += 1
            for frame_i in range(k):
                t = pos - (k - 1 - frame_i)
                ch_start = frame_i * C
                if t >= 0:
                    temporal[idx, ch_start:ch_start + C] = all_matrices[indices[t]]
    return temporal, n_padded, len(file_to_indices)


# ======================== 数据加载 (文件级划分) ========================

def load_data_and_split(k):
    data = torch.load(DATA_PATH, weights_only=False)
    matrices_raw = data["matrices"]
    targets = data["targets"]
    hours = data["hours"]
    qc_counts = data["qc_counts"]
    file_ids = data["file_ids"]

    N = len(matrices_raw)
    print(f"  原始样本: {N}, 构建时序窗口 k={k} ...")
    t0 = time.time()
    matrices, n_padded, n_files = build_temporal_matrices(matrices_raw, file_ids, k)
    elapsed = time.time() - t0
    print(f"  时序矩阵: {list(matrices.shape)}, 耗时 {elapsed:.1f}s")
    print(f"  文件数: {n_files}, 零填充样本: {n_padded} ({100*n_padded/N:.1f}%)")
    del matrices_raw

    # 文件级划分
    unique_files = np.unique(file_ids.numpy())
    n_total_files = len(unique_files)
    rng = np.random.RandomState(42)
    perm_files = rng.permutation(n_total_files)

    n_train_f = int(n_total_files * 0.8)
    n_val_f = int(n_total_files * 0.1)

    train_files = set(unique_files[perm_files[:n_train_f]])
    val_files = set(unique_files[perm_files[n_train_f:n_train_f + n_val_f]])
    test_files = set(unique_files[perm_files[n_train_f + n_val_f:]])

    fid_np = file_ids.numpy()
    train_idx = np.where(np.isin(fid_np, np.array(list(train_files))))[0]
    val_idx = np.where(np.isin(fid_np, np.array(list(val_files))))[0]
    test_idx = np.where(np.isin(fid_np, np.array(list(test_files))))[0]

    print(f"  文件级划分: {n_total_files} files → "
          f"train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"  样本数: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

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
        "matrices": matrices,
        "targets_z": targets_z, "hours": hours,
        "qc_counts": qc_counts, "ordinal_bins": ordinal_bins,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "t_mean": t_mean, "t_std": t_std, "sample_weights": sample_weights,
        "targets_raw": targets,
    }


# ======================== 数据集 ========================

class AugDataset(Dataset):
    def __init__(self, matrices, targets_z, hours, qc_counts, bins):
        self.m = matrices
        self.t = targets_z
        self.h = hours
        self.q = qc_counts
        self.b = bins

    def __len__(self):
        return len(self.m)

    def __getitem__(self, idx):
        m = self.m[idx]
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)
        return m, self.t[idx], self.h[idx], self.q[idx], self.b[idx]


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
        progress = (epoch - cfg["warmup_epochs"]) / max(
            cfg["epochs"] - cfg["warmup_epochs"], 1
        )
        lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (
            1 + np.cos(np.pi * progress)
        )
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


# ======================== 单种子训练 ========================

def train_single_seed(seed, d, device, k, save_dir):
    seed_dir = save_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model_kwargs = get_model_kwargs(k)
    model = PortMoE(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )

    train_set = AugDataset(
        d["matrices"][d["train_idx"]],
        d["targets_z"][d["train_idx"]],
        d["hours"][d["train_idx"]],
        d["qc_counts"][d["train_idx"]],
        d["ordinal_bins"][d["train_idx"]],
    )
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]],
        d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]],
        d["qc_counts"][d["val_idx"]],
        d["ordinal_bins"][d["val_idx"]],
    )
    sampler = WeightedRandomSampler(
        d["sample_weights"], num_samples=len(d["train_idx"]), replacement=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=CFG["batch_size"],
        sampler=sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    best_val_mae_sum = float("inf")
    best_epoch = 0

    print(f"\n{'=' * 50}")
    print(f"  Seed {seed} — k={k} 消融 ({CFG['epochs']} epochs)")
    print(f"{'=' * 50}")
    header = (f"{'Ep':>4} | {'LR':>10} | {'Loss':>10} | {'GnLL':>4} | "
              f"{'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}")
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

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            aux_weight=aux_w, use_gnll=use_gnll,
        )

        _, _, val_mae, _, _ = evaluate_model(
            model, val_loader, device, d["t_mean"], d["t_std"], use_tta=True,
        )
        elapsed = time.time() - t0

        val_mae_sum = val_mae[0].item() + val_mae[1].item()
        marker = ""
        if val_mae_sum < best_val_mae_sum:
            best_val_mae_sum = val_mae_sum
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "seed": seed,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_mae.numpy(),
                },
                seed_dir / "best.pt",
            )
            marker = " *"

        gnll_flag = "Yes" if use_gnll else "No"
        print(
            f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | {gnll_flag:>4} | "
            f"{val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}"
        )

    total_time = time.time() - t_start
    print(
        f"\n✓ Seed {seed} 完成 — best epoch {best_epoch}, "
        f"ValMAE_sum={best_val_mae_sum:.2f}, time={total_time:.0f}s"
    )
    return best_epoch, best_val_mae_sum, total_time


# ======================== 集成评估 ========================

@torch.no_grad()
def ensemble_evaluate(seeds, d, device, k, save_dir):
    t_mean, t_std = d["t_mean"], d["t_std"]
    model_kwargs = get_model_kwargs(k)

    test_set = TensorDataset(
        d["matrices"][d["test_idx"]],
        d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]],
        d["qc_counts"][d["test_idx"]],
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    val_set = TensorDataset(
        d["matrices"][d["val_idx"]],
        d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]],
        d["qc_counts"][d["val_idx"]],
    )
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    models = []
    for seed in seeds:
        model = PortMoE(**model_kwargs).to(device)
        ckpt_path = save_dir / f"seed_{seed}" / "best.pt"
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded seed={seed} (epoch {ckpt['epoch']})")

    def eval_ensemble(loader, label):
        all_preds_per_model = []
        all_targets_list = []
        targets_collected = False

        for mi, model_i in enumerate(models):
            preds_list = []
            for batch in loader:
                m = batch[0].to(device, non_blocking=True)
                t = batch[1]
                h = batch[2].to(device, non_blocking=True)
                q = batch[3].to(device, non_blocking=True)

                p1 = model_i(m, h, q)
                p2 = model_i(m.flip(-1), h, q)
                pred = (p1 + p2) * 0.5
                preds_list.append(pred.cpu())

                if not targets_collected:
                    all_targets_list.append(t)
            targets_collected = True
            all_preds_per_model.append(torch.cat(preds_list))

        trues = torch.cat(all_targets_list) * t_std + t_mean

        print(f"\n  === {label}: 各模型 + TTA ===")
        for i, seed in enumerate(seeds):
            p = all_preds_per_model[i] * t_std + t_mean
            mae = (p - trues).abs().mean(dim=0)
            print(f"  Seed {seed:>4}: MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}")

        print(f"\n  === {label}: 累积集成效果 ===")
        for ki in range(1, len(seeds) + 1):
            ens_pred = torch.stack(all_preds_per_model[:ki]).mean(dim=0) * t_std + t_mean
            mae = (ens_pred - trues).abs().mean(dim=0)
            rmse = ((ens_pred - trues) ** 2).mean(dim=0).sqrt()
            ss_res = ((ens_pred - trues) ** 2).sum(dim=0)
            ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
            r2 = 1 - ss_res / ss_tot
            print(
                f"  Top-{ki}: MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}  "
                f"RMSE={rmse[0]:.4f}  R²={r2[0]:.4f}"
            )

        ens_pred = torch.stack(all_preds_per_model).mean(dim=0) * t_std + t_mean
        mae = (ens_pred - trues).abs().mean(dim=0)
        rmse = ((ens_pred - trues) ** 2).mean(dim=0).sqrt()
        ss_res = ((ens_pred - trues) ** 2).sum(dim=0)
        ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
        r2 = 1 - ss_res / ss_tot
        return mae, rmse, r2, ens_pred, trues

    print(f"\n{'=' * 60}")
    print(f"  k={k} 消融集成评估 ({len(seeds)} seeds × Hetero × TTA)")
    print(f"{'=' * 60}")

    val_mae, val_rmse, val_r2, _, _ = eval_ensemble(val_loader, "Val Set")
    test_mae, test_rmse, test_r2, test_pred, test_true = eval_ensemble(
        test_loader, "Test Set"
    )

    # Per-QC
    test_qc = d["qc_counts"][d["test_idx"]]
    print(f"\n  === Per-QC 分析 (Test, Ensemble+TTA) ===")
    per_qc = {}
    for qc_val, label in [(1, "qc=1"), (2, "qc=2"), (3, "qc=3")]:
        mask = test_qc == qc_val
        n = mask.sum().item()
        if n > 0:
            mae_qc = (test_pred[mask] - test_true[mask]).abs().mean(dim=0)
            print(f"  {label}: N={n:5d}, MAE_TEU={mae_qc[0]:.2f}")
            per_qc[label] = {"n": n, "mae_teu": round(mae_qc[0].item(), 2)}
    mask = test_qc >= 4
    n = mask.sum().item()
    if n > 0:
        mae_qc = (test_pred[mask] - test_true[mask]).abs().mean(dim=0)
        print(f"  qc>=4: N={n:5d}, MAE_TEU={mae_qc[0]:.2f}")
        per_qc["qc>=4"] = {"n": n, "mae_teu": round(mae_qc[0].item(), 2)}

    results = {
        "version": f"ablation-temporal-k{k}",
        "temporal_k": k,
        "split": "file-level",
        "seeds": ENSEMBLE_SEEDS,
        "val_mae_teu": round(val_mae[0].item(), 4),
        "val_mae_move": round(val_mae[1].item(), 4),
        "test_mae_teu": round(test_mae[0].item(), 4),
        "test_mae_move": round(test_mae[1].item(), 4),
        "test_rmse_teu": round(test_rmse[0].item(), 4),
        "test_rmse_move": round(test_rmse[1].item(), 4),
        "test_r2_teu": round(test_r2[0].item(), 4),
        "test_r2_move": round(test_r2[1].item(), 4),
        "per_qc": per_qc,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {save_dir / 'results.json'}")
    return results


# ======================== 主入口 ========================

def run_single_k(k, device):
    save_dir = get_save_dir(k)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  时序窗口消融: k={k} (文件级划分, {6*k}ch)")
    print("=" * 60)

    d = load_data_and_split(k)
    model_kwargs = get_model_kwargs(k)
    n_params = count_parameters(PortMoE(**model_kwargs))
    print(f"  模型参数量: {n_params:,}")

    for seed in ENSEMBLE_SEEDS:
        ckpt_path = save_dir / f"seed_{seed}" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            print(f"\n  ✓ Seed {seed} 已存在 (epoch {ckpt['epoch']}), 跳过")
            continue
        train_single_seed(seed, d, device, k, save_dir)

    return ensemble_evaluate(ENSEMBLE_SEEDS, d, device, k, save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4, 5],
                        help="要运行的 k 值列表 (默认: 1 2 4 5, 跳过 k=3 因为 v6 已有)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    for k in args.k:
        results = run_single_k(k, device)
        all_results[k] = results

    # 汇总
    print("\n" + "=" * 70)
    print("  时序窗口消融汇总 (文件级划分)")
    print("=" * 70)
    print(f"{'k':>3} | {'Channels':>8} | {'MAE_TEU':>9} | {'MAE_move':>9} | {'RMSE':>8} | {'R²':>8}")
    print("-" * 60)
    for k in sorted(all_results.keys()):
        r = all_results[k]
        ch = 6 * k
        print(f"{k:3d} | {ch:>8} | {r['test_mae_teu']:9.4f} | {r['test_mae_move']:9.4f} | "
              f"{r['test_rmse_teu']:8.4f} | {r['test_r2_teu']:8.4f}")

    # 保存汇总
    summary_path = EXPERIMENTS_DIR / "ablation_temporal_k_summary.json"
    summary = {str(k): v for k, v in all_results.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
