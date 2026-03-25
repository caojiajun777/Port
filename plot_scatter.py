"""Step 5: 可视化 + 统计检验

生成以下图表与统计结果 (保存到 experiments/figures/):
  1. scatter_comparison.png  — 4模型散点图 (pred vs true, TEU)
  2. error_hist.png          — 误差分布直方图 (4 模型对比)
  3. per_qc_bar.png          — 各 qc 数量下 MAE 柱状图
  4. ablation_bar.png        — 组件消融 MAE 柱状图

统计检验:
  - PortMoE vs 各 Baseline: Wilcoxon signed-rank test (绝对误差)
  - 4 seeds 均值±标准差汇总表(per-seed MAE)

使用方法:
    python plot_scatter.py
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"
FIGURES_DIR = EXPERIMENTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_SEEDS = [42, 123, 456, 789]
K = 2

sys.path.insert(0, str(ROOT))
from models import PortMoE
from baseline_comparison import (
    ConvLSTMBaseline, CNNGRUBaseline, PlainCNNBaseline,
    build_temporal_matrices, BASELINE_REGISTRY, TEMPORAL_K,
)



# ════════════════════════════════════════════════════════════
#  数据加载 & 测试集准备
# ════════════════════════════════════════════════════════════

def load_data():
    print("[ 1/5 ] 加载数据 ...")
    t0 = time.time()
    raw = torch.load(DATA_PATH, weights_only=False)
    matrices_raw = raw["matrices"]
    targets      = raw["targets"]
    hours        = raw["hours"]
    qc_counts    = raw["qc_counts"]
    file_ids     = raw["file_ids"]

    matrices, _, _ = build_temporal_matrices(matrices_raw, file_ids, k=K)

    unique_files = np.unique(file_ids.numpy())
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(unique_files))
    n_tr = int(len(unique_files) * 0.8)
    n_v  = int(len(unique_files) * 0.1)
    train_f = set(unique_files[perm[:n_tr]])
    test_f  = set(unique_files[perm[n_tr + n_v:]])

    fid_np    = file_ids.numpy()
    train_idx = np.where(np.isin(fid_np, list(train_f)))[0]
    test_idx  = np.where(np.isin(fid_np, list(test_f)))[0]

    t_mean = targets[train_idx].mean(0)
    t_std  = targets[train_idx].std(0) + 1e-8
    targets_z = (targets - t_mean) / t_std

    print(f"       数据加载完毕, 耗时 {time.time()-t0:.1f}s, 测试集: {len(test_idx)} 样本")
    return dict(
        matrices=matrices, targets_z=targets_z, targets_raw=targets,
        hours=hours, qc_counts=qc_counts,
        train_idx=train_idx, test_idx=test_idx,
        t_mean=t_mean, t_std=t_std,
    )


# ════════════════════════════════════════════════════════════
#  推理工具
# ════════════════════════════════════════════════════════════

def build_loader(d, idx):
    ds = TensorDataset(
        d["matrices"][idx], d["targets_z"][idx],
        d["hours"][idx], d["qc_counts"][idx],
    )
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)


@torch.no_grad()
def get_predictions(models_list, loader, device, t_mean, t_std, tta=True):
    t_mean = t_mean.to(device)
    t_std  = t_std.to(device)
    per_seed_preds = []
    all_trues = None
    for mi, model in enumerate(models_list):
        preds, trues_list = [], []
        for batch in loader:
            x = batch[0].to(device)
            t = batch[1].to(device)
            h = batch[2].to(device)
            q = batch[3].to(device)
            p = model(x, h, q)
            if tta:
                p = (p + model(x.flip(-1), h, q)) * 0.5
            preds.append(p)
            if mi == 0:
                trues_list.append(t)
        per_seed_preds.append(torch.cat(preds))
        if mi == 0:
            all_trues = torch.cat(trues_list)
    all_trues_raw = all_trues * t_std + t_mean
    ens_pred_raw  = torch.stack(per_seed_preds).mean(0) * t_std + t_mean
    per_seed_raw  = [p * t_std + t_mean for p in per_seed_preds]
    return ens_pred_raw.cpu(), all_trues_raw.cpu(), [p.cpu() for p in per_seed_raw]


# ════════════════════════════════════════════════════════════
#  加载 PortMoE
# ════════════════════════════════════════════════════════════

def load_portmoe(device):
    print("[ 2/5 ] 加载 PortMoE (k=2, 4 seeds) ...")
    MODEL_KWARGS = dict(in_channels=12, stem_ch=48, stage_ch=64, cond_dim=32,
                        num_experts=4, expert_hidden=64, heteroscedastic=True)
    models = []
    for s in ENSEMBLE_SEEDS:
        m = PortMoE(**MODEL_KWARGS).to(device)
        ckpt_path = EXPERIMENTS_DIR / f"ablation-temporal-k2/seed_{s}/best.pt"
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    print("       PortMoE 加载完成")
    return models


# ════════════════════════════════════════════════════════════
#  加载 Baselines
# ════════════════════════════════════════════════════════════

def load_baselines(device):
    print("[ 3/5 ] 加载 Baseline 模型 ...")
    baselines = {}
    for key, info in BASELINE_REGISTRY.items():
        models_b = []
        for s in ENSEMBLE_SEEDS:
            ckpt_path = EXPERIMENTS_DIR / f"baseline-{key}/seed_{s}/best.pt"
            if not ckpt_path.exists():
                print(f"       警告: {ckpt_path} 不存在，跳过")
                break
            model = info["cls"](**info["kwargs"]).to(device)
            ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            models_b.append(model)
        if len(models_b) == len(ENSEMBLE_SEEDS):
            baselines[key] = {"models": models_b, "name": info["name"]}
            print(f"       {info['name']} 加载完成")
    return baselines


# ════════════════════════════════════════════════════════════
#  统计检验
# ════════════════════════════════════════════════════════════

def statistical_tests(portmoe_pred, baselines_pred, trues):
    print("\n" + "="*60)
    print("  统计检验 (Wilcoxon signed-rank test, TEU 绝对误差)")
    print("="*60)
    portmoe_ae = (portmoe_pred[:, 0] - trues[:, 0]).abs().numpy()
    results = {}
    for name, pred in baselines_pred.items():
        baseline_ae = (pred[:, 0] - trues[:, 0]).abs().numpy()
        stat, p = stats.wilcoxon(portmoe_ae, baseline_ae, alternative="less")
        effect_size = stat / (len(portmoe_ae) * (len(portmoe_ae) + 1) / 2)
        results[name] = {"stat": float(stat), "p_value": float(p), "effect_size": float(effect_size)}
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  PortMoE vs {name:<12}: W={stat:.1f}, p={p:.2e} {sig}, r={effect_size:.3f}")
    return results


def per_seed_stats(portmoe_per_seed, baselines_per_seed, trues):
    print("\n" + "="*60)
    print("  各模型 Per-Seed MAE_TEU (均值 ± 标准差)")
    print("="*60)
    all_stats = {}
    models_info = [("PortMoE (k=2)", portmoe_per_seed)]
    for name, per_seed in baselines_per_seed.items():
        models_info.append((name, per_seed))
    for model_name, per_seed in models_info:
        maes = [(p[:, 0] - trues[:, 0]).abs().mean().item() for p in per_seed]
        mean_mae = np.mean(maes); std_mae = np.std(maes)
        all_stats[model_name] = {"mean": mean_mae, "std": std_mae, "per_seed": maes}
        seed_str = "  ".join(f"{m:.4f}" for m in maes)
        print(f"  {model_name:<20}: {mean_mae:.4f} ± {std_mae:.4f}   [{seed_str}]")
    return all_stats


# ════════════════════════════════════════════════════════════
#  图 1: Scatter Comparison
# ════════════════════════════════════════════════════════════

def plot_scatter_comparison(portmoe_pred, baselines_pred, trues, qc_counts):
    print("  Plotting scatter comparison ...")
    model_items = [("PortMoE (Ours)", portmoe_pred)] + \
                  [(info_name, pred) for info_name, pred in baselines_pred.items()]
    n_models = len(model_items)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    qc = qc_counts.numpy()
    colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71", 4: "#9b59b6"}
    for ax, (name, pred) in zip(axes, model_items):
        true_teu = trues[:, 0].numpy(); pred_teu = pred[:, 0].numpy()
        mae_val = np.abs(pred_teu - true_teu).mean()
        r2_val  = 1 - np.sum((pred_teu - true_teu)**2) / np.sum((true_teu - true_teu.mean())**2)
        for qc_v in [1, 2, 3, 4]:
            mask = qc == qc_v if qc_v < 4 else qc >= 4
            label = f"qc={qc_v}" if qc_v < 4 else "qc≥4"
            ax.scatter(true_teu[mask], pred_teu[mask], s=6, alpha=0.35,
                       c=colors.get(qc_v, "#95a5a6"), label=label, rasterized=True)
        vmin = min(true_teu.min(), pred_teu.min())
        vmax = max(true_teu.max(), pred_teu.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.2, alpha=0.7)
        ax.set_title(f"{name}\nMAE={mae_val:.2f}  R²={r2_val:.4f}", fontsize=11)
        ax.set_xlabel("True TEU (avg eff)", fontsize=10); ax.set_ylabel("Predicted TEU", fontsize=10)
        ax.legend(fontsize=8, markerscale=2); ax.tick_params(labelsize=9)
    plt.suptitle("Predicted vs. True TEU — Test Set (color = qc count)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "scatter_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"       保存: {out}")


# ════════════════════════════════════════════════════════════
#  图 2: Error Distribution Histogram
# ════════════════════════════════════════════════════════════

def plot_error_hist(portmoe_pred, baselines_pred, trues):
    print("  Plotting error histogram ...")
    model_items = [("PortMoE (Ours)", portmoe_pred)] + list(baselines_pred.items())
    fig, axes = plt.subplots(1, len(model_items), figsize=(5 * len(model_items), 4), sharey=True)
    if len(model_items) == 1:
        axes = [axes]
    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    true_teu = trues[:, 0].numpy()
    for i, (ax, (name, pred)) in enumerate(zip(axes, model_items)):
        errors = pred[:, 0].numpy() - true_teu
        mae_v  = np.abs(errors).mean()
        ax.hist(errors, bins=60, color=palette[i % len(palette)], alpha=0.75,
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linestyle="--", lw=1)
        ax.axvline( mae_v, color="red", linestyle="-.", lw=1, label=f"MAE={mae_v:.2f}")
        ax.axvline(-mae_v, color="red", linestyle="-.", lw=1)
        ax.set_title(name, fontsize=11); ax.set_xlabel("Prediction Error (TEU)", fontsize=9)
        if i == 0: ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8); ax.tick_params(labelsize=8)
    plt.suptitle("Prediction Error Distribution (TEU) — Test Set", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "error_hist.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"       保存: {out}")


# ════════════════════════════════════════════════════════════
#  图 3: Per-QC MAE Bar Chart
# ════════════════════════════════════════════════════════════

def plot_per_qc_bar(portmoe_pred, baselines_pred, trues, qc_counts):
    print("  Plotting per-QC bar chart ...")
    model_items = [("PortMoE", portmoe_pred)] + list(baselines_pred.items())
    qc = qc_counts.numpy(); true_teu = trues[:, 0].numpy()
    qc_groups = [(1, "qc=1"), (2, "qc=2"), (3, "qc=3"), (4, "qc≥4")]
    group_masks = {1: qc==1, 2: qc==2, 3: qc==3, 4: qc>=4}
    group_ns = {k: int(m.sum()) for k, m in group_masks.items()}
    data = {}
    for name, pred in model_items:
        pred_teu = pred[:, 0].numpy()
        data[name] = [np.abs(pred_teu[group_masks[qv]] - true_teu[group_masks[qv]]).mean()
                      if group_masks[qv].sum() > 0 else 0.0
                      for qv, _ in qc_groups]
    x = np.arange(len(qc_groups)); n_m = len(model_items); width = 0.7 / n_m
    palette = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, maes) in enumerate(data.items()):
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width, label=name,
                      color=palette[i % len(palette)], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, maes):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    tick_labels = [f"{lbl}\n(N={group_ns[qv]:,})" for qv, lbl in qc_groups]
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("MAE (TEU)", fontsize=11)
    ax.set_title("Per-QC MAE Comparison (Test Set, Ensemble+TTA)", fontsize=12)
    ax.legend(fontsize=10); ax.tick_params(labelsize=9); ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = FIGURES_DIR / "per_qc_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"       保存: {out}")


# ════════════════════════════════════════════════════════════
#  图 4: 组件消融 Bar Chart
# ════════════════════════════════════════════════════════════

def plot_ablation_bar():
    print("  Plotting ablation bar chart ...")
    ablation_data = [
        ("4-Ens+TTA\n(Full Model)",   1.7233, "#2196F3"),
        ("4-Ens\nw/o TTA",            1.8192, "#42A5F5"),
        ("Single+TTA",                2.1036, "#FF9800"),
        ("Single\nw/o TTA",           2.3830, "#FF5722"),
        ("MSE Loss\n(vs GaussNLL)",   2.4563, "#9C27B0"),
        ("PlainCNN\n(w/o MoE)",       2.5911, "#F44336"),
        ("k=1\n(w/o Temporal)",       6.6533, "#B71C1C"),
    ]
    labels = [d[0] for d in ablation_data]
    maes   = [d[1] for d in ablation_data]
    cols   = [d[2] for d in ablation_data]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), maes, color=cols, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("MAE (TEU)", fontsize=11)
    ax.set_title("Component Ablation — MAE_TEU on Test Set", fontsize=12)
    ax.axhline(1.7233, color="#1565C0", linestyle="--", lw=1.2, alpha=0.7, label="Full Model (1.72)")
    ax.legend(fontsize=10); ax.set_ylim(bottom=0, top=max(maes) * 1.12); ax.tick_params(labelsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "ablation_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"       保存: {out}")


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("="*60)

    d = load_data()
    test_loader = build_loader(d, d["test_idx"])
    qc_counts_test = d["qc_counts"][d["test_idx"]]

    portmoe_models = load_portmoe(device)
    baselines = load_baselines(device)

    print("[ 4/5 ] 推理 ...")
    portmoe_pred, trues, portmoe_per_seed = get_predictions(
        portmoe_models, test_loader, device, d["t_mean"], d["t_std"], tta=True)

    baselines_pred     = {}
    baselines_per_seed = {}
    for key, info in baselines.items():
        pred, _, per_seed = get_predictions(
            info["models"], test_loader, device, d["t_mean"], d["t_std"], tta=True)
        baselines_pred[info["name"]]     = pred
        baselines_per_seed[info["name"]] = per_seed
        mae = (pred[:, 0] - trues[:, 0]).abs().mean().item()
        print(f"       {info['name']:<12}: MAE_TEU={mae:.4f}")

    portmoe_mae = (portmoe_pred[:, 0] - trues[:, 0]).abs().mean().item()
    print(f"       {'PortMoE':<12}: MAE_TEU={portmoe_mae:.4f}")

    print("\n[ 5/5 ] 统计检验 & 绘图 ...")
    test_results = statistical_tests(portmoe_pred, baselines_pred, trues)
    seed_stats   = per_seed_stats(portmoe_per_seed, baselines_per_seed, trues)

    plot_scatter_comparison(portmoe_pred, baselines_pred, trues, qc_counts_test)
    plot_error_hist(portmoe_pred, baselines_pred, trues)
    plot_per_qc_bar(portmoe_pred, baselines_pred, trues, qc_counts_test)
    plot_ablation_bar()

    stats_output = {
        "wilcoxon_tests": test_results,
        "per_seed_mae": {k: {"mean": v["mean"], "std": v["std"], "per_seed": v["per_seed"]}
                         for k, v in seed_stats.items()},
    }
    out_json = EXPERIMENTS_DIR / "stats_results.json"
    out_json.write_text(json.dumps(stats_output, indent=2, ensure_ascii=False))
    print(f"\n统计结果已保存: {out_json}")
    print("\n图表已保存到 experiments/figures/")
    print("  scatter_comparison.png  error_hist.png  per_qc_bar.png  ablation_bar.png")


if __name__ == "__main__":
    main()

