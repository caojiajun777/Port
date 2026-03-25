"""PortMoEv2-Hetero-Ensemble 深度诊断分析

分析当前最优模型(MAE_TEU=5.11)的误差结构，寻找进一步改进空间。

诊断维度:
1. Per-QC 分组误差 (qc=0,1,2,3,4+)
2. Per-target-range 分组误差 (Zero, Low, Mid, High, VHigh)
3. Per-hour 误差
4. 误差分布统计 (median, percentiles, skewness)
5. Worst 30 样本分析 (qc、target、hour 特征)
6. 模型间一致性分析 (各模型预测相关性、分歧度)
7. 残差 vs 真值 模式
8. 不可约噪声估计
"""
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from models import PortMoE

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "processed_data.pt"
HETERO_ENS_DIR = ROOT / "experiments" / "PortMoEv2-Hetero-Ensemble"

ENSEMBLE_SEEDS = [42, 123, 456, 789]
MODEL_KWARGS = {"stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
                "num_experts": 4, "expert_hidden": 64,
                "heteroscedastic": True}


def load_data():
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

    return {
        "matrices": matrices, "targets": targets,
        "targets_z": targets_z, "hours": hours,
        "qc_counts": qc_counts,
        "train_idx": train_idx, "val_idx": val_idx,
        "test_idx": test_idx,
        "t_mean": t_mean, "t_std": t_std,
    }


@torch.no_grad()
def get_ensemble_predictions(d, device):
    """获取各模型单独预测 + TTA 和集成预测"""
    t_mean, t_std = d["t_mean"], d["t_std"]

    test_set = TensorDataset(
        d["matrices"][d["test_idx"]], d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]], d["qc_counts"][d["test_idx"]],
    )
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    # 加载模型
    models = []
    for seed in ENSEMBLE_SEEDS:
        model = PortMoE(**MODEL_KWARGS).to(device)
        ckpt = torch.load(HETERO_ENS_DIR / f"seed_{seed}" / "best.pt",
                          weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    # 获取各模型 TTA 预测
    per_model_preds = []
    all_targets = []
    targets_collected = False

    for model in models:
        preds_list = []
        for batch in test_loader:
            m = batch[0].to(device, non_blocking=True)
            t = batch[1]
            h = batch[2].to(device, non_blocking=True)
            q = batch[3].to(device, non_blocking=True)

            p1 = model(m, h, q)
            p2 = model(m.flip(-1), h, q)
            pred = (p1 + p2) * 0.5
            preds_list.append(pred.cpu())
            if not targets_collected:
                all_targets.append(t)
        targets_collected = True
        per_model_preds.append(torch.cat(preds_list) * t_std + t_mean)

    trues = torch.cat(all_targets) * t_std + t_mean
    ensemble_pred = torch.stack(per_model_preds).mean(dim=0)

    # 获取异方差 logvar (用第一个模型 train mode)
    models[0].train()
    logvar_list = []
    for batch in test_loader:
        m = batch[0].to(device, non_blocking=True)
        h = batch[2].to(device, non_blocking=True)
        q = batch[3].to(device, non_blocking=True)
        _, logvar, _ = models[0](m, h, q)
        logvar_list.append(logvar.cpu())
    models[0].eval()
    logvar = torch.cat(logvar_list)  # (N_test, 2)
    sigma = torch.exp(0.5 * logvar)  # std

    return per_model_preds, ensemble_pred, trues, sigma


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_data()

    test_hours = d["hours"][d["test_idx"]].numpy()
    test_qc = d["qc_counts"][d["test_idx"]].numpy()
    test_targets_raw = d["targets"][d["test_idx"]].numpy()

    per_model_preds, ens_pred, trues, sigma = get_ensemble_predictions(d, device)
    ens_pred_np = ens_pred.numpy()
    trues_np = trues.numpy()
    sigma_np = sigma.numpy()

    errors_teu = np.abs(ens_pred_np[:, 0] - trues_np[:, 0])
    errors_move = np.abs(ens_pred_np[:, 1] - trues_np[:, 1])
    signed_errors_teu = ens_pred_np[:, 0] - trues_np[:, 0]

    N_test = len(trues_np)
    print(f"\n{'=' * 70}")
    print(f"  PortMoEv2-Hetero-Ensemble 深度诊断 (Test Set, N={N_test})")
    print(f"{'=' * 70}")

    # ============================================================
    # 1. 总体统计
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  1. 总体误差统计")
    print(f"{'=' * 70}")
    print(f"  MAE_TEU:     {errors_teu.mean():.4f}")
    print(f"  MAE_move:    {errors_move.mean():.4f}")
    print(f"  Median AE:   {np.median(errors_teu):.4f}")
    print(f"  P75 AE:      {np.percentile(errors_teu, 75):.4f}")
    print(f"  P90 AE:      {np.percentile(errors_teu, 90):.4f}")
    print(f"  P95 AE:      {np.percentile(errors_teu, 95):.4f}")
    print(f"  P99 AE:      {np.percentile(errors_teu, 99):.4f}")
    print(f"  Max AE:      {errors_teu.max():.4f}")
    print(f"  Mean Signed:  {signed_errors_teu.mean():+.4f} (bias)")
    print(f"  RMSE_TEU:    {np.sqrt((signed_errors_teu ** 2).mean()):.4f}")

    ss_res = ((ens_pred_np[:, 0] - trues_np[:, 0]) ** 2).sum()
    ss_tot = ((trues_np[:, 0] - trues_np[:, 0].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R²_TEU:      {r2:.4f}")

    # 误差比例分解
    low_err = (errors_teu < 5).sum() / N_test * 100
    mid_err = ((errors_teu >= 5) & (errors_teu < 20)).sum() / N_test * 100
    high_err = (errors_teu >= 20).sum() / N_test * 100
    print(f"\n  误差分布: <5={low_err:.1f}%, 5~20={mid_err:.1f}%, >=20={high_err:.1f}%")

    # ============================================================
    # 2. Per-QC 分组
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  2. Per-QC Count 误差分组")
    print(f"{'=' * 70}")
    qc_groups = [(0, "qc=0"), (1, "qc=1"), (2, "qc=2"), (3, "qc=3")]
    qc_groups += [(None, "qc>=4")]

    header = f"{'QC':>8} | {'N':>6} | {'%':>5} | {'MAE_TEU':>8} | {'MAE_mv':>8} | {'Med_AE':>8} | {'P90':>8} | {'σ_TEU':>8} | {'误差贡献':>8}"
    print(header)
    print("-" * len(header))

    total_error = errors_teu.sum()
    for qc_val, label in qc_groups:
        if qc_val is not None:
            mask = test_qc == qc_val
        else:
            mask = test_qc >= 4
        n = mask.sum()
        if n == 0:
            continue
        mae_t = errors_teu[mask].mean()
        mae_m = errors_move[mask].mean()
        med_ae = np.median(errors_teu[mask])
        p90 = np.percentile(errors_teu[mask], 90)
        avg_sigma = sigma_np[mask, 0].mean()
        contrib = errors_teu[mask].sum() / total_error * 100
        pct = n / N_test * 100
        print(f"{label:>8} | {n:6d} | {pct:4.1f}% | {mae_t:8.2f} | {mae_m:8.2f} | {med_ae:8.2f} | {p90:8.2f} | {avg_sigma:8.2f} | {contrib:7.1f}%")

    # ============================================================
    # 3. Per-target-range 分组
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  3. Per-Target-Range 误差分组 (TEU)")
    print(f"{'=' * 70}")
    ranges = [
        ("Zero (=0)", lambda t: t == 0),
        ("Low (0,15]", lambda t: (t > 0) & (t <= 15)),
        ("Mid (15,50]", lambda t: (t > 15) & (t <= 50)),
        ("High (50,100]", lambda t: (t > 50) & (t <= 100)),
        ("VHigh (100,200]", lambda t: (t > 100) & (t <= 200)),
        ("Extreme (>200)", lambda t: t > 200),
    ]
    teu_true = trues_np[:, 0]
    header = f"{'Range':>16} | {'N':>6} | {'%':>5} | {'MAE_TEU':>8} | {'Med_AE':>8} | {'P90':>8} | {'Mean_true':>10} | {'误差贡献':>8}"
    print(header)
    print("-" * len(header))
    for name, cond in ranges:
        mask = cond(teu_true)
        n = mask.sum()
        if n == 0:
            continue
        mae_t = errors_teu[mask].mean()
        med_ae = np.median(errors_teu[mask])
        p90 = np.percentile(errors_teu[mask], 90)
        mean_true = teu_true[mask].mean()
        contrib = errors_teu[mask].sum() / total_error * 100
        pct = n / N_test * 100
        print(f"{name:>16} | {n:6d} | {pct:4.1f}% | {mae_t:8.2f} | {med_ae:8.2f} | {p90:8.2f} | {mean_true:10.2f} | {contrib:7.1f}%")

    # ============================================================
    # 4. Per-Hour 误差
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  4. Per-Hour 误差分析 (Top 5 最差小时)")
    print(f"{'=' * 70}")
    hour_stats = []
    for h in range(24):
        mask = test_hours == h
        n = mask.sum()
        if n == 0:
            continue
        mae_t = errors_teu[mask].mean()
        hour_stats.append((h, n, mae_t))
    hour_stats.sort(key=lambda x: -x[2])
    header = f"{'Hour':>6} | {'N':>6} | {'MAE_TEU':>8}"
    print(header)
    print("-" * len(header))
    for h, n, mae_t in hour_stats[:5]:
        print(f"{h:6d} | {n:6d} | {mae_t:8.2f}")
    print("  ...")
    for h, n, mae_t in hour_stats[-3:]:
        print(f"{h:6d} | {n:6d} | {mae_t:8.2f}")

    # ============================================================
    # 5. QC × Target Range 交叉分析
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  5. QC × Target Range 交叉分析 (MAE_TEU)")
    print(f"{'=' * 70}")
    qc_bins = [(0, "qc=0"), (1, "qc=1"), (2, "qc=2"), (3, "qc=3"), (None, "qc≥4")]
    tgt_bins = [
        ("=0", lambda t: t == 0),
        ("(0,30]", lambda t: (t > 0) & (t <= 30)),
        ("(30,100]", lambda t: (t > 30) & (t <= 100)),
        (">100", lambda t: t > 100),
    ]
    header = f"{'':>8} | " + " | ".join(f"{tb[0]:>10}" for tb in tgt_bins) + " | {'Total':>8}"
    # Manual header
    print(f"{'':>8} | {'=0':>10} | {'(0,30]':>10} | {'(30,100]':>10} | {'>100':>10} | {'Total':>8}")
    print("-" * 78)
    for qc_val, qlabel in qc_bins:
        if qc_val is not None:
            qc_mask = test_qc == qc_val
        else:
            qc_mask = test_qc >= 4
        row = f"{qlabel:>8} |"
        for _, tcond in tgt_bins:
            tmask = tcond(teu_true)
            mask = qc_mask & tmask
            n = mask.sum()
            if n < 5:
                row += f" {'n<5':>10} |"
            else:
                mae_t = errors_teu[mask].mean()
                row += f" {mae_t:7.2f}({n:>3}) |"
        # total
        n_total = qc_mask.sum()
        mae_total = errors_teu[qc_mask].mean() if n_total > 0 else 0
        row += f" {mae_total:8.2f}"
        print(row)

    # ============================================================
    # 6. Worst 30 样本分析
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  6. Worst 30 预测 (按 AE_TEU)")
    print(f"{'=' * 70}")
    worst_idx = np.argsort(-errors_teu)[:30]
    header = f"{'Rank':>4} | {'AE_TEU':>8} | {'True':>8} | {'Pred':>8} | {'QC':>4} | {'Hour':>4} | {'σ_TEU':>6}"
    print(header)
    print("-" * len(header))
    for rank, idx in enumerate(worst_idx):
        ae = errors_teu[idx]
        true_val = teu_true[idx]
        pred_val = ens_pred_np[idx, 0]
        qc = test_qc[idx]
        hour = test_hours[idx]
        sig = sigma_np[idx, 0]
        print(f"{rank+1:4d} | {ae:8.2f} | {true_val:8.1f} | {pred_val:8.1f} | {qc:4d} | {hour:4d} | {sig:6.2f}")

    # QC 分布 of worst 30
    from collections import Counter
    worst_qc = Counter(test_qc[worst_idx])
    print(f"\n  Worst 30 的 QC 分布: {dict(sorted(worst_qc.items()))}")

    # ============================================================
    # 7. 模型间分歧度分析
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  7. 模型间分歧度分析")
    print(f"{'=' * 70}")

    per_model_np = [p.numpy()[:, 0] for p in per_model_preds]

    # 各模型间 MAE 相关
    print("  各模型对 Test set 的 MAE_TEU:")
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        mae_i = np.abs(per_model_np[i] - teu_true).mean()
        print(f"    Seed {seed}: {mae_i:.4f}")

    # 模型间预测标准差（分歧度）
    stack = np.stack(per_model_np, axis=0)  # (4, N)
    model_std = stack.std(axis=0)  # (N,)
    print(f"\n  模型间预测 std (分歧度):")
    print(f"    Mean:   {model_std.mean():.4f}")
    print(f"    Median: {np.median(model_std):.4f}")
    print(f"    P90:    {np.percentile(model_std, 90):.4f}")
    print(f"    P99:    {np.percentile(model_std, 99):.4f}")

    # 分歧度 vs 误差的关系
    high_disagree = model_std > np.percentile(model_std, 90)
    low_disagree = model_std < np.percentile(model_std, 50)
    print(f"\n  高分歧 (top 10%) 样本: MAE_TEU={errors_teu[high_disagree].mean():.4f}")
    print(f"  低分歧 (bot 50%) 样本: MAE_TEU={errors_teu[low_disagree].mean():.4f}")

    # Per-QC 分歧
    print(f"\n  Per-QC 模型分歧度:")
    for qc_val, label in qc_groups:
        if qc_val is not None:
            mask = test_qc == qc_val
        else:
            mask = test_qc >= 4
        if mask.sum() == 0:
            continue
        print(f"    {label}: mean_std={model_std[mask].mean():.4f}")

    # ============================================================
    # 8. 学到的 σ vs 实际误差 关系
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  8. 学到的不确定性 σ vs 实际误差")
    print(f"{'=' * 70}")
    # 按 σ 分位数分组
    sigma_teu = sigma_np[:, 0]
    q_bins = [0, 25, 50, 75, 90, 100]
    prev_p = 0
    header = f"{'σ Range':>16} | {'N':>6} | {'Avg_σ':>8} | {'MAE_TEU':>8} | {'校准比':>8}"
    print(header)
    print("-" * len(header))
    for qi in range(len(q_bins) - 1):
        lo = np.percentile(sigma_teu, q_bins[qi])
        hi = np.percentile(sigma_teu, q_bins[qi + 1])
        if qi == len(q_bins) - 2:
            mask = sigma_teu >= lo
        else:
            mask = (sigma_teu >= lo) & (sigma_teu < hi)
        n = mask.sum()
        if n == 0:
            continue
        avg_sig = sigma_teu[mask].mean()
        mae = errors_teu[mask].mean()
        cal_ratio = mae / (avg_sig + 1e-8)
        label = f"P{q_bins[qi]}-P{q_bins[qi+1]}"
        print(f"{label:>16} | {n:6d} | {avg_sig:8.3f} | {mae:8.3f} | {cal_ratio:8.3f}")

    # ============================================================
    # 9. 残差模式分析
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  9. 残差模式分析 (signed error)")
    print(f"{'=' * 70}")

    # Target 越大越容易低估？
    target_bins = [
        ("=0", teu_true == 0),
        ("(0,15]", (teu_true > 0) & (teu_true <= 15)),
        ("(15,50]", (teu_true > 15) & (teu_true <= 50)),
        ("(50,100]", (teu_true > 50) & (teu_true <= 100)),
        (">100", teu_true > 100),
    ]
    header = f"{'Range':>12} | {'N':>6} | {'MeanBias':>10} | {'MAE':>8} | {'MAPE%':>8}"
    print(header)
    print("-" * len(header))
    for name, mask in target_bins:
        n = mask.sum()
        if n == 0:
            continue
        bias = signed_errors_teu[mask].mean()
        mae = errors_teu[mask].mean()
        # MAPE with protection against zero
        nz = teu_true[mask] > 0
        if nz.sum() > 0:
            mape = (errors_teu[mask][nz] / teu_true[mask][nz]).mean() * 100
        else:
            mape = float("nan")
        print(f"{name:>12} | {n:6d} | {bias:+10.2f} | {mae:8.2f} | {mape:7.1f}%")

    # ============================================================
    # 10. 不可约噪声估计 + 改进空间
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  10. 不可约噪声估计 + 改进空间")
    print(f"{'=' * 70}")

    # 训练集 per-qc 真值方差
    train_idx = d["train_idx"]
    train_targets = d["targets"][train_idx].numpy()
    train_qc = d["qc_counts"][train_idx].numpy()
    train_hours = d["hours"][train_idx].numpy()

    # K-NN 残差估计不可约噪声
    # 简化：相同 (qc, hour) 组合内部的真值方差作为噪声下界
    from collections import defaultdict
    group_residuals = defaultdict(list)
    for i in range(len(train_targets)):
        key = (int(train_qc[i]), int(train_hours[i]))
        group_residuals[key].append(train_targets[i, 0])

    # 组内方差加权平均
    total_var, total_n = 0, 0
    qc_noise = defaultdict(lambda: [0.0, 0])
    for key, vals in group_residuals.items():
        vals = np.array(vals)
        qc = key[0]
        if len(vals) >= 10:
            group_mean = vals.mean()
            group_var = ((vals - group_mean) ** 2).mean()
            total_var += group_var * len(vals)
            total_n += len(vals)
            qc_noise[qc][0] += group_var * len(vals)
            qc_noise[qc][1] += len(vals)

    noise_std = np.sqrt(total_var / total_n) if total_n else 0
    noise_mae = noise_std * np.sqrt(2 / np.pi)  # MAE = σ * √(2/π) for Gaussian

    print(f"  组内噪声 std (per qc,hour): {noise_std:.2f}")
    print(f"  对应不可约 MAE 下界:        {noise_mae:.2f}")
    print(f"  当前 MAE:                    {errors_teu.mean():.2f}")
    print(f"  改进空间 (当前 - 下界):       {errors_teu.mean() - noise_mae:.2f}")
    print(f"  已利用信息比例:               {(1 - (errors_teu.mean() - noise_mae) / errors_teu.mean()) * 100:.1f}%")

    print(f"\n  Per-QC 不可约噪声估计:")
    for qc in sorted(qc_noise.keys()):
        var_sum, n = qc_noise[qc]
        std_qc = np.sqrt(var_sum / n)
        mae_lb = std_qc * np.sqrt(2 / np.pi)
        qc_label = f"qc={qc}" if qc < 4 else f"qc≥4"
        # 对应 qc 的实际 MAE
        if qc < 4:
            mask = test_qc == qc
        else:
            mask = test_qc >= 4
        actual_mae = errors_teu[mask].mean() if mask.sum() > 0 else 0
        print(f"    {qc_label}: 噪声σ={std_qc:.2f}, 不可约MAE={mae_lb:.2f}, 实际MAE={actual_mae:.2f}, 差距={actual_mae - mae_lb:.2f}")

    # ============================================================
    # 11. 零值预测分析
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  11. 零值样本专项分析")
    print(f"{'=' * 70}")
    zero_mask = teu_true == 0
    nonzero_mask = teu_true > 0
    print(f"  零值样本: N={zero_mask.sum()}, MAE={errors_teu[zero_mask].mean():.4f}, "
          f"MeanPred={ens_pred_np[zero_mask, 0].mean():.4f}")
    print(f"  非零样本: N={nonzero_mask.sum()}, MAE={errors_teu[nonzero_mask].mean():.4f}")

    # 零值样本中预测 > 5 的比例
    zero_high_pred = (zero_mask) & (ens_pred_np[:, 0] > 5)
    print(f"  零值中预测>5: {zero_high_pred.sum()} ({zero_high_pred.sum()/zero_mask.sum()*100:.1f}%)")

    # ============================================================
    # 12. 总结与建议
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  12. 分析总结")
    print(f"{'=' * 70}")

    # 按 qc 计算可改进空间
    improvable = 0
    for qc in sorted(qc_noise.keys()):
        var_sum, n = qc_noise[qc]
        std_qc = np.sqrt(var_sum / n)
        mae_lb = std_qc * np.sqrt(2 / np.pi)
        if qc < 4:
            mask = test_qc == qc
        else:
            mask = test_qc >= 4
        actual_mae = errors_teu[mask].mean() if mask.sum() > 0 else 0
        gap = actual_mae - mae_lb
        improvable += gap * mask.sum()

    theoretic_best_mae = errors_teu.mean() - improvable / N_test
    print(f"  当前总 MAE_TEU:  {errors_teu.mean():.4f}")
    print(f"  理论最优 MAE:     {theoretic_best_mae:.4f} (若每个 qc 组都达到噪声下界)")
    print(f"  最大可改进量:     {improvable / N_test:.4f}")

    # 哪个 qc 贡献最多可改进空间
    print(f"\n  各 QC 组的改进贡献:")
    for qc in sorted(qc_noise.keys()):
        var_sum, n = qc_noise[qc]
        std_qc = np.sqrt(var_sum / n)
        mae_lb = std_qc * np.sqrt(2 / np.pi)
        if qc < 4:
            mask = test_qc == qc
        else:
            mask = test_qc >= 4
        actual_mae = errors_teu[mask].mean() if mask.sum() > 0 else 0
        gap = actual_mae - mae_lb
        gap_contrib = gap * mask.sum() / N_test
        print(f"    qc={qc}: gap={gap:.2f}, weighted_contrib={gap_contrib:.4f}")


if __name__ == "__main__":
    main()
