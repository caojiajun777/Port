"""标准 Baseline 对比实验 — 文件级划分, k=2

对比 3 种标准时空模型与 PortMoE (channel-stacking) 的性能差异:
  1. ConvLSTM: 逐帧输入 ConvLSTM 单元，最终隐藏态 → 预测
  2. CNN+GRU: CNN 编码每帧 → GRU 时序建模 → 预测
  3. Plain CNN: 与 PortMoE 同样的通道堆叠，但去掉 MoE (消融 MoE 贡献)

所有 baseline 均采用:
  - 文件级划分 (零数据泄露)
  - k=2 时序窗口 (消融实验最优)
  - 4 seeds × 120 epochs × TTA × 集成
  - 参数量控制在与 PortMoE (~309K) 相近的量级

使用方法:
    python baseline_comparison.py                    # 运行所有 baseline
    python baseline_comparison.py --model convlstm   # 只运行 ConvLSTM
    python baseline_comparison.py --model cnn_gru plain_cnn
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Dataset
from pathlib import Path

from models import count_parameters

# ======================== 配置 ========================

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"

ENSEMBLE_SEEDS = [42, 123, 456, 789]
TEMPORAL_K = 2  # 消融实验最优

CFG = {
    "batch_size": 256,
    "epochs": 120,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "ordinal_thresholds": [0, 15, 60],
    "gnll_warmup_epochs": 10,
}


# ======================== Baseline 模型定义 ========================

class ConvLSTMCell(nn.Module):
    """单个 ConvLSTM 单元"""
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(self, x, hc):
        h, c = hc
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMBaseline(nn.Module):
    """Baseline 1: ConvLSTM 时空模型

    逐帧输入 ConvLSTM，最终隐藏态经 pooling → MLP → 预测
    """
    def __init__(self, in_channels=6, grid_h=7, grid_w=22,
                 hidden_ch=64, num_hours=24, max_qc=20, cond_dim=32):
        super().__init__()
        # 每帧先用 CNN stem 提取空间特征
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        # ConvLSTM
        self.convlstm = ConvLSTMCell(48, hidden_ch, kernel_size=3)
        self.hidden_ch = hidden_ch

        # 条件嵌入
        self.hour_embed = nn.Embedding(num_hours, cond_dim)
        self.qc_embed = nn.Embedding(max_qc + 1, cond_dim)

        # 预测头
        feat_dim = hidden_ch + 2 * cond_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, matrix, hour, qc_cnt):
        """matrix: (B, 12, 7, 22) — k=2 帧拼接"""
        B = matrix.shape[0]
        H, W = matrix.shape[2], matrix.shape[3]
        C_per_frame = 6
        k = matrix.shape[1] // C_per_frame

        # 初始化隐藏态
        device = matrix.device
        h = torch.zeros(B, self.hidden_ch, H, W, device=device)
        c = torch.zeros(B, self.hidden_ch, H, W, device=device)

        # 逐帧输入
        for t in range(k):
            frame = matrix[:, t * C_per_frame:(t + 1) * C_per_frame]  # (B, 6, H, W)
            x = self.stem(frame)  # (B, 48, H, W)
            h, c = self.convlstm(x, (h, c))

        # pooling
        feat = h.mean(dim=[2, 3])  # (B, hidden_ch)

        # 条件拼接
        h_cond = self.hour_embed(hour.clamp(0, 23))
        q_cond = self.qc_embed(qc_cnt.clamp(0, 20))
        feat = torch.cat([feat, h_cond, q_cond], dim=-1)

        return self.head(feat)


class CNNGRUBaseline(nn.Module):
    """Baseline 2: CNN + GRU

    CNN 编码每帧为特征向量 → GRU 处理时序 → 预测
    """
    def __init__(self, in_channels=6, grid_h=7, grid_w=22,
                 cnn_ch=64, gru_hidden=128, num_hours=24, max_qc=20, cond_dim=32):
        super().__init__()
        # 每帧 CNN 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, cnn_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_ch),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # (B, cnn_ch, 1, 1)
        )
        # GRU 时序建模
        self.gru = nn.GRU(cnn_ch, gru_hidden, batch_first=True)

        # 条件嵌入
        self.hour_embed = nn.Embedding(num_hours, cond_dim)
        self.qc_embed = nn.Embedding(max_qc + 1, cond_dim)

        # 预测头
        feat_dim = gru_hidden + 2 * cond_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, matrix, hour, qc_cnt):
        B = matrix.shape[0]
        C_per_frame = 6
        k = matrix.shape[1] // C_per_frame

        # 编码每帧
        frame_feats = []
        for t in range(k):
            frame = matrix[:, t * C_per_frame:(t + 1) * C_per_frame]
            feat = self.encoder(frame).squeeze(-1).squeeze(-1)  # (B, cnn_ch)
            frame_feats.append(feat)

        seq = torch.stack(frame_feats, dim=1)  # (B, k, cnn_ch)
        _, h_n = self.gru(seq)  # h_n: (1, B, gru_hidden)
        gru_out = h_n.squeeze(0)  # (B, gru_hidden)

        # 条件拼接
        h_cond = self.hour_embed(hour.clamp(0, 23))
        q_cond = self.qc_embed(qc_cnt.clamp(0, 20))
        feat = torch.cat([gru_out, h_cond, q_cond], dim=-1)

        return self.head(feat)


class PlainCNNBaseline(nn.Module):
    """Baseline 3: Plain CNN (通道堆叠, 无 MoE)

    与 PortMoE 使用相同的通道堆叠输入和 backbone，
    但用单一回归头替代 MoE，消融 MoE 的贡献。
    """
    def __init__(self, in_channels=12, grid_h=7, grid_w=22,
                 stem_ch=48, stage_ch=64, cond_dim=32,
                 num_hours=24, max_qc=20, dropout=0.1):
        super().__init__()
        from models import PortCNNPlus
        self.backbone = PortCNNPlus(
            in_channels=in_channels, grid_h=grid_h, grid_w=grid_w,
            stem_ch=stem_ch, stage_ch=stage_ch, cond_dim=cond_dim,
            num_hours=num_hours, max_qc=max_qc, dropout=dropout,
            headless=True,
        )
        feat_dim = self.backbone.feat_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, matrix, hour, qc_cnt):
        features = self.backbone.encode(matrix, hour, qc_cnt)
        return self.head(features)


# ======================== 模型注册 ========================

BASELINE_REGISTRY = {
    "convlstm": {
        "cls": ConvLSTMBaseline,
        "kwargs": {"hidden_ch": 64, "cond_dim": 32},
        "name": "ConvLSTM",
    },
    "cnn_gru": {
        "cls": CNNGRUBaseline,
        "kwargs": {"cnn_ch": 64, "gru_hidden": 128, "cond_dim": 32},
        "name": "CNN+GRU",
    },
    "plain_cnn": {
        "cls": PlainCNNBaseline,
        "kwargs": {"in_channels": 6 * TEMPORAL_K, "stem_ch": 48, "stage_ch": 64, "cond_dim": 32},
        "name": "PlainCNN",
    },
}


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

def load_data_and_split():
    data = torch.load(DATA_PATH, weights_only=False)
    matrices_raw = data["matrices"]
    targets = data["targets"]
    hours = data["hours"]
    qc_counts = data["qc_counts"]
    file_ids = data["file_ids"]

    N = len(matrices_raw)
    print(f"  原始样本: {N}, 构建时序窗口 k={TEMPORAL_K} ...")
    t0 = time.time()
    matrices, n_padded, n_files = build_temporal_matrices(matrices_raw, file_ids, k=TEMPORAL_K)
    print(f"  时序矩阵: {list(matrices.shape)}, 耗时 {time.time()-t0:.1f}s")
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
    print(f"  文件级划分: {n_total_files} files → train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"  样本数: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    t_mean = targets[train_idx].mean(dim=0)
    t_std = targets[train_idx].std(dim=0) + 1e-8
    targets_z = (targets - t_mean) / t_std

    is_nonzero = (targets[train_idx, 0] > 0).float()
    sample_weights = torch.where(is_nonzero.bool(), torch.tensor(2.0), torch.tensor(1.0))

    return {
        "matrices": matrices, "targets_z": targets_z, "hours": hours,
        "qc_counts": qc_counts, "train_idx": train_idx, "val_idx": val_idx,
        "test_idx": test_idx, "t_mean": t_mean, "t_std": t_std,
        "sample_weights": sample_weights, "targets_raw": targets,
    }


# ======================== 数据集 ========================

class AugDataset(Dataset):
    def __init__(self, matrices, targets_z, hours, qc_counts):
        self.m = matrices; self.t = targets_z; self.h = hours; self.q = qc_counts
    def __len__(self): return len(self.m)
    def __getitem__(self, idx):
        m = self.m[idx]
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)
        return m, self.t[idx], self.h[idx], self.q[idx]


# ======================== 训练工具 ========================

def cosine_lr(optimizer, epoch, cfg):
    if epoch < cfg["warmup_epochs"]:
        lr = cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    else:
        progress = (epoch - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
        lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, device):
    """Baseline 用 MSE loss (无异方差)"""
    model.train()
    total_loss, n = 0.0, 0
    mse = nn.MSELoss()
    for batch in loader:
        matrices = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        hours = batch[2].to(device, non_blocking=True)
        qc_counts = batch[3].to(device, non_blocking=True)

        pred = model(matrices, hours, qc_counts)
        loss = mse(pred, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
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

def train_single_seed(model_key, seed, d, device):
    info = BASELINE_REGISTRY[model_key]
    save_dir = EXPERIMENTS_DIR / f"baseline-{model_key}" / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)

    model = info["cls"](**info["kwargs"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    train_set = AugDataset(
        d["matrices"][d["train_idx"]], d["targets_z"][d["train_idx"]],
        d["hours"][d["train_idx"]], d["qc_counts"][d["train_idx"]],
    )
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
    )
    sampler = WeightedRandomSampler(d["sample_weights"], num_samples=len(d["train_idx"]), replacement=True)
    train_loader = DataLoader(train_set, batch_size=CFG["batch_size"], sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    best_val_mae_sum = float("inf"); best_epoch = 0

    print(f"\n{'='*50}")
    print(f"  [{info['name']}] Seed {seed} ({CFG['epochs']} epochs, k={TEMPORAL_K})")
    print(f"{'='*50}")
    header = f"{'Ep':>4} | {'LR':>10} | {'Loss':>10} | {'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}"
    print(header); print("-" * len(header))

    t_start = time.time()
    for epoch in range(CFG["epochs"]):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, CFG)
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        _, _, val_mae, _, _ = evaluate_model(model, val_loader, device, d["t_mean"], d["t_std"], use_tta=True)
        elapsed = time.time() - t0
        val_mae_sum = val_mae[0].item() + val_mae[1].item()
        marker = ""
        if val_mae_sum < best_val_mae_sum:
            best_val_mae_sum = val_mae_sum; best_epoch = epoch + 1
            torch.save({"epoch": epoch+1, "seed": seed, "model_state_dict": model.state_dict(), "val_mae": val_mae.numpy()}, save_dir / "best.pt")
            marker = " *"
        print(f"{epoch+1:4d} | {lr:10.6f} | {train_loss:10.6f} | {val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}")

    total_time = time.time() - t_start
    print(f"\n✓ [{info['name']}] Seed {seed} 完成 — best epoch {best_epoch}, ValMAE_sum={best_val_mae_sum:.2f}, time={total_time:.0f}s")
    return best_epoch, best_val_mae_sum, total_time


# ======================== 集成评估 ========================

@torch.no_grad()
def ensemble_evaluate(model_key, seeds, d, device):
    info = BASELINE_REGISTRY[model_key]
    t_mean, t_std = d["t_mean"], d["t_std"]

    test_set = TensorDataset(
        d["matrices"][d["test_idx"]], d["targets_z"][d["test_idx"]],
        d["hours"][d["test_idx"]], d["qc_counts"][d["test_idx"]],
    )
    val_set = TensorDataset(
        d["matrices"][d["val_idx"]], d["targets_z"][d["val_idx"]],
        d["hours"][d["val_idx"]], d["qc_counts"][d["val_idx"]],
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    models = []
    for seed in seeds:
        model = info["cls"](**info["kwargs"]).to(device)
        ckpt = torch.load(EXPERIMENTS_DIR / f"baseline-{model_key}" / f"seed_{seed}" / "best.pt", weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded {info['name']} seed={seed} (epoch {ckpt['epoch']})")

    def eval_ensemble(loader, label):
        all_preds_per_model = []; all_targets_list = []; targets_collected = False
        for mi, model_i in enumerate(models):
            preds_list = []
            for batch in loader:
                m = batch[0].to(device, non_blocking=True)
                t = batch[1]; h = batch[2].to(device, non_blocking=True)
                q = batch[3].to(device, non_blocking=True)
                p1 = model_i(m, h, q); p2 = model_i(m.flip(-1), h, q)
                preds_list.append(((p1 + p2) * 0.5).cpu())
                if not targets_collected: all_targets_list.append(t)
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
            rmse = ((ens_pred - trues)**2).mean(dim=0).sqrt()
            ss_res = ((ens_pred - trues)**2).sum(dim=0)
            ss_tot = ((trues - trues.mean(dim=0))**2).sum(dim=0)
            r2 = 1 - ss_res / ss_tot
            print(f"  Top-{ki}: MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}  RMSE={rmse[0]:.4f}  R²={r2[0]:.4f}")

        ens_pred = torch.stack(all_preds_per_model).mean(dim=0) * t_std + t_mean
        mae = (ens_pred - trues).abs().mean(dim=0)
        rmse = ((ens_pred - trues)**2).mean(dim=0).sqrt()
        ss_res = ((ens_pred - trues)**2).sum(dim=0)
        ss_tot = ((trues - trues.mean(dim=0))**2).sum(dim=0)
        r2 = 1 - ss_res / ss_tot
        return mae, rmse, r2, ens_pred, trues

    print(f"\n{'='*60}")
    print(f"  [{info['name']}] 集成评估 ({len(seeds)} seeds × TTA × k={TEMPORAL_K})")
    print(f"{'='*60}")

    val_mae, val_rmse, val_r2, _, _ = eval_ensemble(val_loader, "Val Set")
    test_mae, test_rmse, test_r2, test_pred, test_true = eval_ensemble(test_loader, "Test Set")

    # Per-QC
    test_qc = d["qc_counts"][d["test_idx"]]
    print(f"\n  === Per-QC 分析 (Test, Ensemble+TTA) ===")
    for qc_val, label in [(1, "qc=1"), (2, "qc=2"), (3, "qc=3")]:
        mask = test_qc == qc_val; n = mask.sum().item()
        if n > 0:
            mae_qc = (test_pred[mask] - test_true[mask]).abs().mean(dim=0)
            print(f"  {label}: N={n:5d}, MAE_TEU={mae_qc[0]:.2f}")
    mask = test_qc >= 4; n = mask.sum().item()
    if n > 0:
        mae_qc = (test_pred[mask] - test_true[mask]).abs().mean(dim=0)
        print(f"  qc>=4: N={n:5d}, MAE_TEU={mae_qc[0]:.2f}")

    # 保存结果
    results = {
        "model": info["name"], "temporal_k": TEMPORAL_K, "split": "file-level",
        "seeds": ENSEMBLE_SEEDS,
        "val_mae_teu": round(val_mae[0].item(), 4), "val_mae_move": round(val_mae[1].item(), 4),
        "test_mae_teu": round(test_mae[0].item(), 4), "test_mae_move": round(test_mae[1].item(), 4),
        "test_rmse_teu": round(test_rmse[0].item(), 4), "test_rmse_move": round(test_rmse[1].item(), 4),
        "test_r2_teu": round(test_r2[0].item(), 4), "test_r2_move": round(test_r2[1].item(), 4),
    }
    results_path = EXPERIMENTS_DIR / f"baseline-{model_key}" / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {results_path}")
    return results


# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="*", default=list(BASELINE_REGISTRY.keys()),
                        choices=list(BASELINE_REGISTRY.keys()))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Baseline 对比实验 — 文件级划分, k={TEMPORAL_K}")
    print("=" * 60)

    d = load_data_and_split()

    all_results = {}
    for model_key in args.model:
        info = BASELINE_REGISTRY[model_key]
        n_params = count_parameters(info["cls"](**info["kwargs"]))
        print(f"\n{'#'*60}")
        print(f"  模型: {info['name']} ({n_params:,} params)")
        print(f"{'#'*60}")

        for seed in ENSEMBLE_SEEDS:
            ckpt_path = EXPERIMENTS_DIR / f"baseline-{model_key}" / f"seed_{seed}" / "best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, weights_only=False)
                print(f"\n  ✓ {info['name']} Seed {seed} checkpoint 已存在 (epoch {ckpt['epoch']}), 跳过训练")
                continue
            train_single_seed(model_key, seed, d, device)

        results = ensemble_evaluate(model_key, ENSEMBLE_SEEDS, d, device)
        all_results[model_key] = results

    # 汇总
    print(f"\n\n{'='*70}")
    print(f"  最终对比汇总 (文件级划分, k={TEMPORAL_K})")
    print(f"{'='*70}")
    print(f"{'Model':<18} | {'MAE_TEU':>8} | {'MAE_move':>8} | {'RMSE':>8} | {'R²':>6}")
    print("-" * 60)
    print(f"{'PortMoE (k=2)':<18} | {'1.7233':>8} | {'1.6902':>8} | {'4.4809':>8} | {'0.9955':>6}")
    for key, res in all_results.items():
        print(f"{res['model']:<18} | {res['test_mae_teu']:>8.4f} | {res['test_mae_move']:>8.4f} | {res['test_rmse_teu']:>8.4f} | {res['test_r2_teu']:>6.4f}")
    print("完成!")


if __name__ == "__main__":
    main()
