"""组件消融实验 (Component Ablation)

基于已有的 k=2 文件级划分 checkpoint，评估各设计组件的贡献：
  - 完整模型 (4-model Ensemble + TTA + GaussNLL)        → 直接读取结果
  - 去掉 TTA        → 从 k=2 checkpoint 推理
  - 去掉 Ensemble   → 单模型 seed=42 推理
  - 去掉 TTA + Ensemble → 单模型无 TTA
  - MSE loss        → 重新训练 (GaussNLL → MSE，其余不变)
  - 去掉时序        → 直接读取 k=1 结果
  - 去掉 MoE        → 直接读取 PlainCNN 结果

使用方法：
    python component_ablation.py              # 只推理（利用已有 checkpoint）
    python component_ablation.py --train_mse  # 额外训练 MSE loss 对照
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

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"

ENSEMBLE_SEEDS = [42, 123, 456, 789]
TEMPORAL_K = 2
K2_EXP_DIR = EXPERIMENTS_DIR / "ablation-temporal-k2"

CFG = {
    "batch_size": 256,
    "epochs": 120,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "aux_loss_weight": 0.3,
    "ordinal_thresholds": [0, 15, 60],
    "gnll_warmup_epochs": 10,
}

MODEL_KWARGS = {
    "in_channels": 6 * TEMPORAL_K,
    "stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
    "num_experts": 4, "expert_hidden": 64,
    "heteroscedastic": True,
}


# ======================== 数据加载 ========================

def build_temporal_matrices(all_matrices, file_ids, k):
    N, C, H, W = all_matrices.shape
    if k == 1:
        return all_matrices.clone(), 0
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
    return temporal, n_padded


def load_data():
    print("加载数据 ...")
    data = torch.load(DATA_PATH, weights_only=False)
    matrices_raw = data["matrices"]
    targets = data["targets"]
    hours = data["hours"]
    qc_counts = data["qc_counts"]
    file_ids = data["file_ids"]

    N = len(matrices_raw)
    matrices, _ = build_temporal_matrices(matrices_raw, file_ids, TEMPORAL_K)
    del matrices_raw

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

    t_mean = targets[train_idx].mean(dim=0)
    t_std = targets[train_idx].std(dim=0) + 1e-8
    targets_z = (targets - t_mean) / t_std

    ordinal_bins = torch.zeros(N, dtype=torch.long)
    for ti, th in enumerate(CFG["ordinal_thresholds"]):
        ordinal_bins[targets[:, 0] > th] = ti + 1

    is_nonzero = (targets[train_idx, 0] > 0).float()
    sample_weights = torch.where(is_nonzero.bool(), torch.tensor(2.0), torch.tensor(1.0))

    print(f"  数据加载完成: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return {
        "matrices": matrices, "targets_z": targets_z, "hours": hours,
        "qc_counts": qc_counts, "ordinal_bins": ordinal_bins,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "t_mean": t_mean, "t_std": t_std, "sample_weights": sample_weights,
        "targets_raw": targets,
    }


# ======================== 推理工具 ========================

def make_test_loader(d):
    test_set = TensorDataset(
        d["matrices"][d["test_idx"]], d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]], d["qc_counts"][d["test_idx"]],
        d["ordinal_bins"][d["test_idx"]],
    )
    return DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)


def load_model(ckpt_path, device, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = MODEL_KWARGS
    model = PortMoE(**model_kwargs).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(models_list, loader, device, use_tta, t_mean, t_std):
    """models_list: list of nn.Module"""
    all_preds_per_model = []
    all_true = []
    collected = False
    for model in models_list:
        preds_list = []
        for batch in loader:
            m = batch[0].to(device, non_blocking=True)
            t = batch[1]; h = batch[2].to(device, non_blocking=True)
            q = batch[3].to(device, non_blocking=True)
            p = model(m, h, q)
            if use_tta:
                p = (p + model(m.flip(-1), h, q)) * 0.5
            preds_list.append(p.cpu())
            if not collected:
                all_true.append(t)
        all_preds_per_model.append(torch.cat(preds_list))
        collected = True

    trues = torch.cat(all_true) * t_std + t_mean
    ens_pred = torch.stack(all_preds_per_model).mean(0) * t_std + t_mean
    mae = (ens_pred - trues).abs().mean(0)
    rmse = ((ens_pred - trues) ** 2).mean(0).sqrt()
    ss_res = ((ens_pred - trues) ** 2).sum(0)
    ss_tot = ((trues - trues.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    return mae, rmse, r2


def fmt(mae, rmse, r2):
    return f"MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}  RMSE={rmse[0]:.4f}  R²={r2[0]:.4f}"


# ======================== 推理消融 ========================

def run_inference_ablation(d, device):
    """利用已有 k=2 checkpoint，评估 TTA / Ensemble 各组合"""
    test_loader = make_test_loader(d)

    # 加载 4 个 k=2 模型
    models = []
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = K2_EXP_DIR / f"seed_{seed}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}\n请先运行 ablation_temporal_k.py --k 2")
        models.append(load_model(ckpt_path, device))
        print(f"  ✓ 加载 k=2 seed={seed}")

    t_mean, t_std = d["t_mean"], d["t_std"]

    print("\n" + "=" * 65)
    print("  推理消融：TTA / Ensemble 组合评估")
    print("=" * 65)
    configs = [
        ("4-Ens + TTA  (完整模型)", models, True),
        ("4-Ens, 无TTA            ", models, False),
        ("单模型(seed=42) + TTA   ", [models[0]], True),
        ("单模型(seed=42), 无TTA  ", [models[0]], False),
    ]
    results = {}
    for label, ms, tta in configs:
        mae, rmse, r2 = predict(ms, test_loader, device, tta, t_mean, t_std)
        print(f"  {label}: {fmt(mae, rmse, r2)}")
        results[label.strip()] = {
            "mae_teu": round(mae[0].item(), 4), "mae_move": round(mae[1].item(), 4),
            "rmse_teu": round(rmse[0].item(), 4), "r2_teu": round(r2[0].item(), 4),
        }
    return results


# ======================== MSE Loss 训练 ========================

class AugDataset(Dataset):
    def __init__(self, matrices, targets_z, hours, qc_counts, bins):
        self.m = matrices; self.t = targets_z; self.h = hours
        self.q = qc_counts; self.b = bins
    def __len__(self): return len(self.m)
    def __getitem__(self, idx):
        m = self.m[idx]
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)
        return m, self.t[idx], self.h[idx], self.q[idx], self.b[idx]


def gaussian_nll_loss(pred, logvar, target):
    logvar = logvar.clamp(-6.0, 6.0)
    inv_var = torch.exp(-logvar)
    return (0.5 * (inv_var * (target - pred) ** 2 + logvar)).mean()


def cosine_lr(optimizer, epoch):
    if epoch < CFG["warmup_epochs"]:
        lr = CFG["lr"] * (epoch + 1) / CFG["warmup_epochs"]
    else:
        progress = (epoch - CFG["warmup_epochs"]) / max(CFG["epochs"] - CFG["warmup_epochs"], 1)
        lr = CFG["min_lr"] + 0.5 * (CFG["lr"] - CFG["min_lr"]) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch_mse(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    mse = nn.MSELoss()
    for batch in loader:
        m = batch[0].to(device, non_blocking=True)
        t = batch[1].to(device, non_blocking=True)
        h = batch[2].to(device, non_blocking=True)
        q = batch[3].to(device, non_blocking=True)
        bins = batch[4].to(device, non_blocking=True)
        pred, logvar, gate_logits = model(m, h, q)
        loss = mse(pred, t)
        # 保留辅助分类 loss
        aux_w = CFG["aux_loss_weight"]
        if aux_w > 0:
            loss = loss + aux_w * nn.functional.cross_entropy(gate_logits, bins)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device, t_mean, t_std, use_tta=True):
    model.eval()
    all_preds, all_trues = [], []
    for batch in loader:
        m = batch[0].to(device, non_blocking=True)
        t = batch[1]; h = batch[2].to(device, non_blocking=True)
        q = batch[3].to(device, non_blocking=True)
        p = model(m, h, q)
        if use_tta:
            p = (p + model(m.flip(-1), h, q)) * 0.5
        all_preds.append(p.cpu()); all_trues.append(t)
    preds = torch.cat(all_preds) * t_std + t_mean
    trues = torch.cat(all_trues) * t_std + t_mean
    mae = (preds - trues).abs().mean(0)
    return preds, trues, mae


def train_mse_variant(d, device):
    """用 MSE loss 重训 PortMoE k=2，4 seeds"""
    save_dir = EXPERIMENTS_DIR / "ablation-mse-loss"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 检查已有 checkpoint（断点继续）
    all_done = all((save_dir / f"seed_{s}" / "best.pt").exists() for s in ENSEMBLE_SEEDS)
    if all_done:
        print("\n  ✓ MSE loss 所有 seed checkpoint 已存在，跳过训练")
        return

    train_set_raw = d["matrices"][d["train_idx"]], d["targets_z"][d["train_idx"]], \
                    d["hours"][d["train_idx"]], d["qc_counts"][d["train_idx"]], \
                    d["ordinal_bins"][d["train_idx"]]
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
        d["ordinal_bins"][d["val_idx"]],
    )
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    sampler = WeightedRandomSampler(d["sample_weights"], num_samples=len(d["train_idx"]), replacement=True)

    for seed in ENSEMBLE_SEEDS:
        ckpt_path = save_dir / f"seed_{seed}" / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            print(f"\n  ✓ MSE seed={seed} checkpoint 已存在 (epoch {ckpt['epoch']}), 跳过")
            continue

        (save_dir / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)
        model = PortMoE(**MODEL_KWARGS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

        train_dataset = AugDataset(*train_set_raw)
        train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler, num_workers=0, pin_memory=True)

        best_val_mae_sum = float("inf"); best_epoch = 0
        print(f"\n{'='*50}")
        print(f"  [MSE Loss] Seed {seed} ({CFG['epochs']} epochs)")
        print(f"{'='*50}")
        header = f"{'Ep':>4} | {'LR':>10} | {'Loss':>10} | {'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}"
        print(header); print("-" * len(header))

        t_start = time.time()
        for epoch in range(CFG["epochs"]):
            t0 = time.time()
            lr = cosine_lr(optimizer, epoch)
            train_loss = train_one_epoch_mse(model, train_loader, optimizer, device)
            _, _, val_mae = eval_model(model, val_loader, device, d["t_mean"], d["t_std"], use_tta=True)
            val_mae_sum = val_mae[0].item() + val_mae[1].item()
            elapsed = time.time() - t0
            marker = ""
            if val_mae_sum < best_val_mae_sum:
                best_val_mae_sum = val_mae_sum; best_epoch = epoch + 1
                torch.save({"epoch": epoch+1, "seed": seed, "model_state_dict": model.state_dict()}, ckpt_path)
                marker = " *"
            print(f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | {val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}")

        total_time = time.time() - t_start
        print(f"\n✓ [MSE] Seed {seed} 完成 — best epoch {best_epoch}, 耗时 {total_time:.0f}s")


@torch.no_grad()
def evaluate_mse_ensemble(d, device):
    """加载 MSE loss checkpoint，集成推理"""
    save_dir = EXPERIMENTS_DIR / "ablation-mse-loss"
    test_loader = make_test_loader(d)
    t_mean, t_std = d["t_mean"], d["t_std"]

    models = []
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = save_dir / f"seed_{seed}" / "best.pt"
        models.append(load_model(ckpt_path, device))
        print(f"  ✓ 加载 MSE seed={seed}")

    mae, rmse, r2 = predict(models, test_loader, device, True, t_mean, t_std)
    print(f"\n  MSE Loss 4-Ens+TTA: {fmt(mae, rmse, r2)}")
    return {
        "mae_teu": round(mae[0].item(), 4), "mae_move": round(mae[1].item(), 4),
        "rmse_teu": round(rmse[0].item(), 4), "r2_teu": round(r2[0].item(), 4),
    }


# ======================== 汇总全部消融结果 ========================

def collect_existing_results():
    """从已有 results.json 读取 k=1 和 PlainCNN 的结果"""
    results = {}

    # k=1（无时序）
    k1_path = EXPERIMENTS_DIR / "ablation-temporal-k1" / "results.json"
    if k1_path.exists():
        r = json.loads(k1_path.read_text())
        results["w/o Temporal (k=1)"] = {
            "mae_teu": r["test_mae_teu"], "mae_move": r["test_mae_move"],
            "rmse_teu": r["test_rmse_teu"], "r2_teu": r["test_r2_teu"],
        }

    # PlainCNN（无 MoE）
    plaincnn_path = EXPERIMENTS_DIR / "baseline-plain_cnn" / "results.json"
    if plaincnn_path.exists():
        r = json.loads(plaincnn_path.read_text())
        results["w/o MoE (PlainCNN)"] = {
            "mae_teu": r["test_mae_teu"], "mae_move": r["test_mae_move"],
            "rmse_teu": r["test_rmse_teu"], "r2_teu": r["test_r2_teu"],
        }

    return results


def print_ablation_table(all_results):
    full_mae = all_results.get("4-Ens + TTA  (完整模型)", {}).get("mae_teu", None)
    # 也尝试去掉多余空格的键
    for k, v in all_results.items():
        if "完整模型" in k:
            full_mae = v["mae_teu"]
            break

    print("\n" + "=" * 80)
    print("  组件消融汇总表")
    print("=" * 80)
    print(f"  {'配置':<32} | {'MAE_TEU':>8} | {'MAE_move':>8} | {'RMSE':>8} | {'R²':>6} | {'vs 完整':>8}")
    print("  " + "-" * 76)

    rows = [
        ("PortMoE 完整 (4-Ens+TTA+GaussNLL)", all_results),
        ("  去掉 Ensemble → 单模型+TTA", all_results),
        ("  去掉 TTA → 4-Ens 无TTA", all_results),
        ("  去掉 Ensemble + TTA", all_results),
        ("  去掉 GaussNLL → MSE loss", all_results),
        ("  去掉 MoE → PlainCNN", all_results),
        ("  去掉 Temporal → k=1", all_results),
    ]

    # 用实际 key 名称输出
    display_order = [
        ("PortMoE 完整 (4-Ens+TTA+GaussNLL)", ["4-Ens + TTA  (完整模型)"]),
        ("去掉 Ensemble → 单模型+TTA", ["单模型(seed=42) + TTA"]),
        ("去掉 TTA → 4-Ens 无TTA", ["4-Ens, 无TTA"]),
        ("去掉 Ensemble + TTA → 单模型无TTA", ["单模型(seed=42), 无TTA"]),
        ("去掉 GaussNLL → MSE loss", ["MSE Loss 4-Ens+TTA"]),
        ("去掉 MoE → PlainCNN", ["w/o MoE (PlainCNN)"]),
        ("去掉 Temporal → k=1", ["w/o Temporal (k=1)"]),
    ]

    for label, key_candidates in display_order:
        r = None
        for kc in key_candidates:
            # 模糊匹配（去掉空格对比）
            for ak, av in all_results.items():
                if ak.strip() == kc.strip():
                    r = av; break
            if r: break
        if r is None:
            print(f"  {label:<32} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>6} | {'N/A':>8}")
            continue
        delta = f"+{(r['mae_teu'] - full_mae)/full_mae*100:.1f}%" if full_mae else "—"
        if r["mae_teu"] == full_mae:
            delta = "—"
        print(f"  {label:<32} | {r['mae_teu']:>8.4f} | {r['mae_move']:>8.4f} | {r['rmse_teu']:>8.4f} | {r['r2_teu']:>6.4f} | {delta:>8}")


# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_mse", action="store_true", help="训练 MSE loss 对照实验")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n" + "=" * 65)
    print("  Step 1: 加载数据")
    print("=" * 65)
    d = load_data()

    print("\n" + "=" * 65)
    print("  Step 2: 推理消融 (TTA / Ensemble)")
    print("=" * 65)
    inference_results = run_inference_ablation(d, device)

    mse_result = None
    if args.train_mse:
        print("\n" + "=" * 65)
        print("  Step 3: MSE Loss 重训 (4 seeds × 120 epochs)")
        print("=" * 65)
        train_mse_variant(d, device)
        mse_result = evaluate_mse_ensemble(d, device)

    # 汇总
    all_results = {}
    all_results.update(inference_results)
    if mse_result:
        all_results["MSE Loss 4-Ens+TTA"] = mse_result
    all_results.update(collect_existing_results())

    print_ablation_table(all_results)

    # 保存
    out_path = EXPERIMENTS_DIR / "component_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
