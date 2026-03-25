"""PortMoEv2 多种子集成训练与评估

策略：训练 N 个不同随机种子的 PortMoEv2，推理时 TTA × Ensemble 平均。
已有 seed=42 的 checkpoint 会直接复用，不重复训练。

使用方法:
    python ensemble.py
"""

import json
import time
import shutil
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
ENSEMBLE_DIR = EXPERIMENTS_DIR / "PortMoEv2-Ensemble"

# 集成种子列表 — seed=42 已有训练好的 checkpoint
ENSEMBLE_SEEDS = [42, 123, 456, 789]

# PortMoEv2 超参（与 run_experiments.py 完全一致）
MODEL_KWARGS = {"stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
                "num_experts": 4, "expert_hidden": 64}
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
}


# ======================== 数据划分（与 run_experiments 完全一致） ========================

def load_data_and_split():
    """加载数据并使用与 run_experiments.py 完全相同的划分方式"""
    data = torch.load(DATA_PATH, weights_only=False)
    matrices = data["matrices"]
    targets = data["targets"]
    hours = data["hours"]
    qc_counts = data["qc_counts"]

    N = len(matrices)
    rng = np.random.RandomState(42)  # 数据划分种子固定为 42
    perm = rng.permutation(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    # raw z-score（仅用训练集统计）
    t_mean = targets[train_idx].mean(dim=0)
    t_std = targets[train_idx].std(dim=0) + 1e-8
    targets_z = (targets - t_mean) / t_std

    # 序数分类 bins
    ordinal_bins = torch.zeros(N, dtype=torch.long)
    for ti, th in enumerate(CFG["ordinal_thresholds"]):
        ordinal_bins[targets[:, 0] > th] = ti + 1

    # 训练集非零样本加权
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


# ======================== 带增强的数据集 ========================

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


# ======================== 训练函数 ========================

def cosine_lr(optimizer, epoch, cfg):
    if epoch < cfg["warmup_epochs"]:
        lr = cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    else:
        progress = (epoch - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
        lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device, aux_weight=0.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        matrices = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        hours = batch[2].to(device, non_blocking=True)
        qc_counts = batch[3].to(device, non_blocking=True)
        bins = batch[4].to(device, non_blocking=True)

        output = model(matrices, hours, qc_counts)
        if isinstance(output, tuple):
            pred, gate_logits = output
            loss = criterion(pred, targets)
            if aux_weight > 0:
                loss = loss + aux_weight * nn.functional.cross_entropy(gate_logits, bins)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(model, loader, device, t_mean, t_std, use_tta=False):
    """评估单模型，返回未还原的 z-score 预测和还原后的 MAE"""
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


def train_single_seed(seed, d, device):
    """训练一个种子的 PortMoEv2，返回 best_epoch 和 val_mae"""
    save_dir = ENSEMBLE_DIR / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设置随机种子（影响模型初始化和训练随机性，不影响数据划分）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    model = PortMoE(**MODEL_KWARGS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=CFG["weight_decay"])
    criterion = nn.MSELoss()

    # DataLoader（每次创建以确保 sampler 状态干净）
    train_set = AugDataset(
        d["matrices"][d["train_idx"]], d["targets_z"][d["train_idx"]],
        d["hours"][d["train_idx"]], d["qc_counts"][d["train_idx"]],
        d["ordinal_bins"][d["train_idx"]],
    )
    val_set = torch.utils.data.TensorDataset(
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

    print(f"\n{'=' * 50}")
    print(f"  Seed {seed} — 开始训练 ({CFG['epochs']} epochs)")
    print(f"{'=' * 50}")
    header = f"{'Ep':>4} | {'LR':>10} | {'Train':>10} | {'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}"
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

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, aux_weight=aux_w)

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

        print(f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | "
              f"{val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}")

    total_time = time.time() - t_start
    print(f"\n✓ Seed {seed} 完成 — best epoch {best_epoch}, "
          f"ValMAE={best_val_mae_sum:.2f}, time={total_time:.0f}s")
    return best_epoch, best_val_mae_sum, total_time


# ======================== 集成评估 ========================

@torch.no_grad()
def ensemble_evaluate(seeds, d, device):
    """加载多个种子的 checkpoint，集成 + TTA 评估"""
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

    # 加载所有模型
    models = []
    for seed in seeds:
        model = PortMoE(**MODEL_KWARGS).to(device)
        ckpt_path = ENSEMBLE_DIR / f"seed_{seed}" / "best.pt"
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded seed={seed} (epoch {ckpt['epoch']})")

    def eval_ensemble(loader, label):
        all_preds_ens = []  # per-model predictions
        all_targets = []

        # 先收集所有 targets
        for batch in loader:
            all_targets.append(batch[1])
        trues = torch.cat(all_targets) * t_std + t_mean

        # 每个模型的 TTA 预测
        individual_results = []
        for i, model in enumerate(models):
            preds_list = []
            for batch in loader:
                m = batch[0].to(device)
                h = batch[2].to(device)
                q = batch[3].to(device)
                pred = model(m, h, q)
                pred_flip = model(m.flip(-1), h, q)
                pred_avg = (pred + pred_flip) * 0.5
                preds_list.append(pred_avg.cpu())
            preds_i = torch.cat(preds_list) * t_std + t_mean
            mae_i = (preds_i - trues).abs().mean(dim=0)
            individual_results.append((seeds[i], preds_i, mae_i))
            all_preds_ens.append(preds_i)

        # 集成平均
        ens_preds = torch.stack(all_preds_ens).mean(dim=0)
        ens_mae = (ens_preds - trues).abs().mean(dim=0)
        ens_rmse = ((ens_preds - trues) ** 2).mean(dim=0).sqrt()
        ss_res = ((ens_preds - trues) ** 2).sum(dim=0)
        ss_tot = ((trues - trues.mean(dim=0)) ** 2).sum(dim=0)
        ens_r2 = 1 - ss_res / ss_tot

        print(f"\n  === {label} ===")
        for seed, _, mae_i in individual_results:
            print(f"  Seed {seed:>4d} + TTA: MAE_TEU={mae_i[0]:.4f}  MAE_move={mae_i[1]:.4f}")

        print(f"  {'Ensemble':>10} + TTA: MAE_TEU={ens_mae[0]:.4f}  MAE_move={ens_mae[1]:.4f}  "
              f"RMSE_TEU={ens_rmse[0]:.4f}  R2_TEU={ens_r2[0]:.4f}  R2_move={ens_r2[1]:.4f}")

        # 逐步增加集成数量
        print(f"\n  === {label}: 累积集成效果 ===")
        for k in range(1, len(models) + 1):
            ens_k = torch.stack(all_preds_ens[:k]).mean(dim=0)
            mae_k = (ens_k - trues).abs().mean(dim=0)
            print(f"  Top-{k} seeds: MAE_TEU={mae_k[0]:.4f}  MAE_move={mae_k[1]:.4f}")

        return ens_mae, ens_rmse, ens_r2, individual_results

    test_results = eval_ensemble(test_loader, "Test Set")
    val_results = eval_ensemble(val_loader, "Val Set")
    return test_results, val_results


# ======================== 主流程 ========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("加载数据...")
    d = load_data_and_split()
    print(f"训练集: {len(d['train_idx']):,}  验证集: {len(d['val_idx']):,}  "
          f"测试集: {len(d['test_idx']):,}")

    # 检查已有 checkpoint
    existing = []
    need_train = []
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = ENSEMBLE_DIR / f"seed_{seed}" / "best.pt"
        # 特殊处理：seed=42 可以从已有 PortMoEv2 复用
        if seed == 42 and not ckpt_path.exists():
            legacy_path = EXPERIMENTS_DIR / "PortMoEv2" / "best.pt"
            if legacy_path.exists():
                (ENSEMBLE_DIR / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_path, ckpt_path)
                print(f"✓ Seed {seed}: 复用已有 PortMoEv2 checkpoint")
                existing.append(seed)
                continue
        if ckpt_path.exists():
            existing.append(seed)
            print(f"✓ Seed {seed}: checkpoint 已存在，跳过训练")
        else:
            need_train.append(seed)

    print(f"\n已有 {len(existing)} 个 checkpoint, 需要训练 {len(need_train)} 个")

    # 训练缺失的种子
    for seed in need_train:
        train_single_seed(seed, d, device)

    # 集成评估
    print(f"\n{'=' * 60}")
    print(f"  集成评估: {len(ENSEMBLE_SEEDS)} 模型 × TTA")
    print(f"{'=' * 60}")
    test_results, val_results = ensemble_evaluate(ENSEMBLE_SEEDS, d, device)

    # 保存结果
    ens_test_mae, ens_test_rmse, ens_test_r2, _ = test_results
    ens_val_mae, _, _, _ = val_results

    results = {
        "seeds": ENSEMBLE_SEEDS,
        "n_models": len(ENSEMBLE_SEEDS),
        "model_kwargs": MODEL_KWARGS,
        "config": CFG,
        "val_mae_teu": round(ens_val_mae[0].item(), 4),
        "val_mae_move": round(ens_val_mae[1].item(), 4),
        "test_mae_teu": round(ens_test_mae[0].item(), 4),
        "test_mae_move": round(ens_test_mae[1].item(), 4),
        "test_rmse_teu": round(ens_test_rmse[0].item(), 4),
        "test_rmse_move": round(ens_test_rmse[1].item(), 4),
        "test_r2_teu": round(ens_test_r2[0].item(), 4),
        "test_r2_move": round(ens_test_r2[1].item(), 4),
    }
    with open(ENSEMBLE_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  最终对比")
    print(f"{'=' * 60}")
    print(f"PortMoEv2 (单模型):        Test MAE_TEU=6.8049  MAE_move=5.8469  R²=0.9297")
    print(f"PortMoEv2 + TTA:           Test MAE_TEU=6.2752  MAE_move=5.3612  R²=0.9380")
    print(f"PortMoEv2 Ensemble + TTA:  Test MAE_TEU={ens_test_mae[0]:.4f}  "
          f"MAE_move={ens_test_mae[1]:.4f}  R²={ens_test_r2[0]:.4f}")
    delta_pct = (1 - ens_test_mae[0].item() / 6.2752) * 100
    print(f"相对 TTA 基线提升: {delta_pct:+.2f}%")

    print(f"\n结果已保存: {ENSEMBLE_DIR / 'results.json'}")
    print("完成!")


if __name__ == "__main__":
    main()
