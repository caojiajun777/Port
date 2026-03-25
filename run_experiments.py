"""四套方案统一公平对比实验

统一配置:
  - 样本随机划分 train/val/test (80/10/10)
  - MSE 损失 + raw targets（统一基准）
  - 仅用训练集统计量做 z-score 标准化
  - MAE-based 最优 epoch 选择
  - 独立测试集评估

使用方法:
    python run_experiments.py
"""

import json
import time
import datetime
import platform
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

from models import PortCNN, PortViT, PortFormer, PortCNNPlus, PortMoE, count_parameters

# ======================== 目录 & 路径 ========================

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"

# ======================== 超参配置 ========================

SEED = 42
SPLIT_MODE = "random"   # "random" = 样本随机划分, "file_group" = 按文件分组
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# 选择要运行的模型
MODELS_TO_RUN = [
    # "PortCNN",
    # "PortViT",
    # "PortFormer",
    # "PortCNNPlus",
    # "PortMoE",
    # "PortMoEv2",
    "PortMoEv2-EMA",
]

MODEL_CONFIGS = {
    "PortCNN": {
        "model_cls": PortCNN,
        "model_kwargs": {},
        "batch_size": 256,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
    },
    "PortViT": {
        "model_cls": PortViT,
        "model_kwargs": {
            "d_model": 96, "num_layers": 4, "num_heads": 8, "d_ff": 384,
        },
        "batch_size": 256,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
    },
    "PortFormer": {
        "model_cls": PortFormer,
        "model_kwargs": {
            "conv_dim": 32, "d_model": 64, "num_layers": 3,
            "num_heads": 8, "d_ff": 256,
        },
        "batch_size": 256,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
    },
    "PortCNNPlus": {
        "model_cls": PortCNNPlus,
        "model_kwargs": {
            "stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
        },
        "batch_size": 256,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
    },
    "PortMoE": {
        "model_cls": PortMoE,
        "model_kwargs": {
            "stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
            "num_experts": 3, "expert_hidden": 48,
        },
        "batch_size": 256,
        "epochs": 80,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
        "aux_loss_weight": 0.3,
    },
    "PortMoEv2": {
        "model_cls": PortMoE,
        "model_kwargs": {
            "stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
            "num_experts": 4, "expert_hidden": 64,
        },
        "batch_size": 256,
        "epochs": 120,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
        "aux_loss_weight": 0.3,
        "aux_loss_decay": True,
        "ordinal_thresholds": [0, 15, 60],
    },
    "PortMoEv2-EMA": {
        "model_cls": PortMoE,
        "model_kwargs": {
            "stem_ch": 48, "stage_ch": 64, "cond_dim": 32,
            "num_experts": 4, "expert_hidden": 64,
        },
        "batch_size": 256,
        "epochs": 120,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "min_lr": 1e-6,
        "loss": "mse",
        "target_transform": "raw",
        "aux_loss_weight": 0.3,
        "aux_loss_decay": True,
        "ordinal_thresholds": [0, 15, 60],
        "use_ema": True,
        "ema_decay": 0.999,
        "use_tta": True,
        "use_mixup": True,
        "mixup_alpha": 0.2,
    },
}


# ======================== 数据划分 ========================

def random_split_indices(n, seed=42, train_ratio=0.80, val_ratio=0.10):
    """样本级随机划分 train/val/test"""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (perm[:n_train].tolist(),
            perm[n_train:n_train + n_val].tolist(),
            perm[n_train + n_val:].tolist())


def group_split_by_file(file_ids_np, seed=42,
                        train_ratio=0.70, val_ratio=0.15):
    """按 xlsx 文件分组，保证同一船次样本不跨集"""
    unique_files = np.unique(file_ids_np)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_files))
    unique_files = unique_files[perm]

    n = len(unique_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = set(unique_files[:n_train].tolist())
    val_files = set(unique_files[n_train:n_train + n_val].tolist())

    train_idx, val_idx, test_idx = [], [], []
    for i, fid in enumerate(file_ids_np):
        if fid in train_files:
            train_idx.append(i)
        elif fid in val_files:
            val_idx.append(i)
        else:
            test_idx.append(i)

    return train_idx, val_idx, test_idx


# ======================== 数据集 ========================

class PortDataset(Dataset):
    def __init__(self, matrices, targets_norm, hours, qc_counts, bins=None):
        self.matrices = matrices
        self.targets_norm = targets_norm
        self.hours = hours
        self.qc_counts = qc_counts
        self.bins = bins if bins is not None else torch.zeros(len(matrices), dtype=torch.long)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return (self.matrices[idx], self.targets_norm[idx],
                self.hours[idx], self.qc_counts[idx], self.bins[idx])


class AugPortDataset(Dataset):
    """带水平翻转增强的训练数据集"""
    def __init__(self, matrices, targets_norm, hours, qc_counts, bins=None):
        self.matrices = matrices
        self.targets_norm = targets_norm
        self.hours = hours
        self.qc_counts = qc_counts
        self.bins = bins if bins is not None else torch.zeros(len(matrices), dtype=torch.long)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        m = self.matrices[idx]
        if torch.rand(1).item() < 0.5:
            m = m.flip(-1)  # 水平翻转（栏方向 A→X ↔ X→A）
        return (m, self.targets_norm[idx], self.hours[idx],
                self.qc_counts[idx], self.bins[idx])


# ======================== EMA ========================

class ModelEMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        for name, buf in model.named_buffers():
            if name in self.shadow:
                self.shadow[name].copy_(buf.data)

    def apply(self, model):
        """Copy shadow weights into model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.shadow:
                buf.data.copy_(self.shadow[name])


# ======================== 学习率调度 ========================

def cosine_lr(optimizer, epoch, cfg):
    """带线性 warmup 的余弦退火"""
    if epoch < cfg["warmup_epochs"]:
        lr = cfg["lr"] * (epoch + 1) / cfg["warmup_epochs"]
    else:
        progress = (epoch - cfg["warmup_epochs"]) / max(cfg["epochs"] - cfg["warmup_epochs"], 1)
        lr = cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ======================== 训练 / 评估 ========================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    aux_weight=0.0, mixup_alpha=0.0, ema=None):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        matrices = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        hours = batch[2].to(device, non_blocking=True)
        qc_counts = batch[3].to(device, non_blocking=True)
        bins = batch[4].to(device, non_blocking=True)

        # Mixup
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            lam = max(lam, 1 - lam)  # ensure lam >= 0.5
            idx_perm = torch.randperm(matrices.size(0), device=device)
            matrices = lam * matrices + (1 - lam) * matrices[idx_perm]
            targets = lam * targets + (1 - lam) * targets[idx_perm]
            bins_mixed = bins  # keep original bins for aux loss
        else:
            bins_mixed = bins

        output = model(matrices, hours, qc_counts)
        if isinstance(output, tuple):
            pred, gate_logits = output
            loss = criterion(pred, targets)
            if aux_weight > 0:
                loss = loss + aux_weight * nn.functional.cross_entropy(
                    gate_logits, bins_mixed)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_mean, target_std,
             use_log1p=False, use_tta=False):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    for batch in loader:
        matrices = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        hours = batch[2].to(device, non_blocking=True)
        qc_counts = batch[3].to(device, non_blocking=True)

        pred = model(matrices, hours, qc_counts)
        if use_tta:
            pred_flip = model(matrices.flip(-1), hours, qc_counts)
            pred = (pred + pred_flip) * 0.5
        loss = criterion(pred, targets)

        total_loss += loss.item()
        n += 1
        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(n, 1)
    # 反标准化
    preds = torch.cat(all_preds) * target_std + target_mean
    trues = torch.cat(all_targets) * target_std + target_mean
    if use_log1p:
        preds = torch.expm1(preds).clamp(min=0)
        trues = torch.expm1(trues).clamp(min=0)
    mae = (preds - trues).abs().mean(dim=0)
    rmse = ((preds - trues) ** 2).mean(dim=0).sqrt()
    return avg_loss, mae, rmse


# ======================== 单模型训练流程 ========================

def make_criterion(loss_name):
    if loss_name == "huber":
        return nn.SmoothL1Loss()
    return nn.MSELoss()


def train_model(model_name, model, train_loader, val_loader,
                cfg, device, target_mean, target_std, use_log1p=False):
    save_dir = EXPERIMENTS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = make_criterion(cfg.get("loss", "mse"))

    use_ema = cfg.get("use_ema", False)
    use_tta = cfg.get("use_tta", False)
    mixup_alpha = cfg.get("mixup_alpha", 0.0) if cfg.get("use_mixup", False) else 0.0
    ema = ModelEMA(model, decay=cfg.get("ema_decay", 0.999)) if use_ema else None

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    history = []

    header = (f"{'Epoch':>5} | {'LR':>10} | {'Train':>10} | {'Val':>10} | "
              f"{'MAE_TEU':>8} | {'MAE_mv':>8} | {'Time':>6}")
    print(header)
    print("-" * len(header))

    t_start = time.time()

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, cfg)

        # 辅助损失权重（可选余弦衰减）
        base_aux = cfg.get("aux_loss_weight", 0.0)
        if cfg.get("aux_loss_decay", False) and base_aux > 0:
            progress = epoch / max(cfg["epochs"] - 1, 1)
            cur_aux = base_aux * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))
        else:
            cur_aux = base_aux

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            aux_weight=cur_aux, mixup_alpha=mixup_alpha, ema=ema,
        )

        # For validation, use EMA weights if available
        if ema is not None:
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.apply(model)

        val_loss, val_mae, val_rmse = evaluate(
            model, val_loader, criterion, device, target_mean, target_std,
            use_log1p=use_log1p, use_tta=use_tta,
        )

        # Restore original weights for continued training
        if ema is not None:
            model.load_state_dict(orig_state)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch + 1,
            "lr": round(lr, 8),
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "mae_teu": round(val_mae[0].item(), 4),
            "mae_move": round(val_mae[1].item(), 4),
            "rmse_teu": round(val_rmse[0].item(), 4),
            "rmse_move": round(val_rmse[1].item(), 4),
            "time": round(elapsed, 2),
        }
        history.append(record)

        marker = ""
        val_mae_sum = val_mae[0].item() + val_mae[1].item()
        if val_mae_sum < best_val_mae:
            best_val_mae = val_mae_sum
            best_val_loss = val_loss
            # Save EMA weights if available, otherwise save model weights
            save_state = {}
            if ema is not None:
                ema.apply(model)
                save_state = model.state_dict()
                model.load_state_dict(orig_state)
            else:
                save_state = model.state_dict()
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": save_state,
                "val_loss": val_loss,
                "val_mae": val_mae.numpy(),
                "val_rmse": val_rmse.numpy(),
                "target_mean": target_mean.numpy(),
                "target_std": target_std.numpy(),
                "use_log1p": use_log1p,
            }, save_dir / "best.pt")
            marker = " *"

        print(f"{epoch+1:5d} | {lr:10.6f} | {train_loss:10.6f} | {val_loss:10.6f} | "
              f"{val_mae[0]:8.2f} | {val_mae[1]:8.2f} | {elapsed:5.1f}s{marker}")

    total_time = time.time() - t_start
    best_record = min(history, key=lambda r: r["mae_teu"] + r["mae_move"])

    return {
        "history": history,
        "best_epoch": best_record["epoch"],
        "best_val_loss": best_record["val_loss"],
        "best_mae_teu": best_record["mae_teu"],
        "best_mae_move": best_record["mae_move"],
        "best_rmse_teu": best_record["rmse_teu"],
        "best_rmse_move": best_record["rmse_move"],
        "total_time_sec": round(total_time, 1),
    }


# ======================== 工作日志 ========================

def write_work_log(all_results, env_info, split_info=None):
    log_path = EXPERIMENTS_DIR / "work_log.md"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# 港口六通道图像建模 — 实验工作日志\n",
        f"**实验时间**: {now}\n",
        "## 1. 实验环境\n",
    ]
    for k, v in env_info.items():
        lines.append(f"- **{k}**: {v}")

    # ---- 数据划分信息 ----
    if split_info:
        lines.append("\n## 1.5 数据划分\n")
        mode = split_info.get("split_mode", "unknown")
        mode_desc = "样本随机划分" if mode == "random" else "按文件分组划分"
        lines.append(f"- **划分方式**: {mode_desc}")
        lines.append(f"- **总样本数**: {split_info['n_total']}")
        lines.append(f"- **训练集**: {split_info['n_train']} 样本")
        lines.append(f"- **验证集**: {split_info['n_val']} 样本")
        lines.append(f"- **测试集**: {split_info['n_test']} 样本")

    # ---- 配置表 ----
    lines.append("\n## 2. 训练配置\n")
    lines.append("| 方案 | batch | epochs | lr | weight_decay | warmup | loss | target |")
    lines.append("|------|-------|--------|----|-------------|--------|------|--------|")
    for name in all_results:
        cfg = MODEL_CONFIGS[name]
        lines.append(
            f"| {name} | {cfg['batch_size']} | {cfg['epochs']} | "
            f"{cfg['lr']} | {cfg['weight_decay']} | {cfg['warmup_epochs']} | "
            f"{cfg.get('loss', 'mse')} | {cfg.get('target_transform', 'raw')} |"
        )

    # ---- 模型参数量 ----
    lines.append("\n## 3. 模型参数量\n")
    lines.append("| 方案 | 参数量 |")
    lines.append("|------|--------|")
    for name, res in all_results.items():
        lines.append(f"| {name} | {res['params']:,} |")

    # ---- 主结果表 ----
    lines.append("\n## 4. 实验结果对比\n")
    lines.append(
        "| 方案 | Best Epoch | Val Loss | MAE_TEU | MAE_move | "
        "RMSE_TEU | RMSE_move | 训练时间 |"
    )
    lines.append(
        "|------|-----------|----------|---------|----------|"
        "----------|-----------|----------|"
    )
    for name, res in all_results.items():
        r = res["summary"]
        lines.append(
            f"| {name} | {r['best_epoch']} | {r['best_val_loss']:.6f} | "
            f"{r['best_mae_teu']:.2f} | {r['best_mae_move']:.2f} | "
            f"{r['best_rmse_teu']:.2f} | {r['best_rmse_move']:.2f} | "
            f"{r['total_time_sec']:.0f}s |"
        )

    # ---- 测试集结果 ----
    has_test = any("test_mae_teu" in res.get("summary", {})
                   for res in all_results.values())
    if has_test:
        lines.append("\n## 4.5 测试集结果\n")
        lines.append(
            "| 方案 | MAE_TEU | MAE_move | RMSE_TEU | RMSE_move |"
        )
        lines.append(
            "|------|---------|----------|----------|-----------|"
        )
        for name, res in all_results.items():
            r = res["summary"]
            if "test_mae_teu" in r:
                lines.append(
                    f"| {name} | {r['test_mae_teu']:.2f} | "
                    f"{r['test_mae_move']:.2f} | "
                    f"{r['test_rmse_teu']:.2f} | "
                    f"{r['test_rmse_move']:.2f} |"
                )

    # ---- 自动对比分析 ----
    lines.append("\n## 5. 对比分析\n")
    names = list(all_results.keys())
    metrics = {
        "MAE_TEU": {n: all_results[n]["summary"]["best_mae_teu"] for n in names},
        "MAE_move": {n: all_results[n]["summary"]["best_mae_move"] for n in names},
        "Val Loss": {n: all_results[n]["summary"]["best_val_loss"] for n in names},
    }
    for metric, vals in metrics.items():
        best = min(vals, key=vals.get)
        lines.append(f"- **{metric} 最优方案**: {best} ({vals[best]:.4f})")

    # ---- 各模型 Top5 epoch ----
    lines.append("\n## 6. 各模型 Top-5 Epoch 记录\n")
    for name, res in all_results.items():
        h = res["summary"]["history"]
        top5 = sorted(h, key=lambda r: r["val_loss"])[:5]
        lines.append(f"### {name}\n")
        lines.append("| Epoch | Val Loss | MAE_TEU | MAE_move | RMSE_TEU | RMSE_move |")
        lines.append("|-------|----------|---------|----------|----------|-----------|")
        for r in top5:
            lines.append(
                f"| {r['epoch']} | {r['val_loss']:.6f} | "
                f"{r['mae_teu']:.2f} | {r['mae_move']:.2f} | "
                f"{r['rmse_teu']:.2f} | {r['rmse_move']:.2f} |"
            )
        lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n工作日志已保存: {log_path}")


# ======================== main ========================

def main():
    # ---- 固定种子 ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_info = {
        "操作系统": platform.platform(),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA 可用": str(torch.cuda.is_available()),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "设备": str(device),
    }

    if not torch.cuda.is_available():
        print("=" * 60)
        print("  ⚠ CUDA 不可用，将使用 CPU（速度很慢）")
        print("  安装 GPU 版: pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("=" * 60)
    else:
        torch.backends.cudnn.benchmark = True

    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  港口六通道图像建模 — 统一公平对比实验")
    print(f"  划分: {SPLIT_MODE} {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO} | 损失: MSE | 目标: raw")
    print("=" * 60)

    # ---- 加载数据 ----
    print(f"\n加载数据: {DATA_PATH}")
    data = torch.load(DATA_PATH, weights_only=False)
    matrices = data["matrices"]          # (N, 6, 7, 22)
    targets_raw = data["targets"]        # (N, 2) 原始真值
    hours = data["hours"]                # (N,) long
    qc_counts = data["qc_counts"]        # (N,) long
    file_ids = data["file_ids"].numpy()  # (N,) int64

    N = len(matrices)
    unique_files = np.unique(file_ids)
    print(f"总样本数: {N:,},  文件数: {len(unique_files)}")

    # ---- 划分 train/val/test ----
    if SPLIT_MODE == "random":
        train_idx, val_idx, test_idx = random_split_indices(
            N, seed=SEED, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        )
    else:
        train_idx, val_idx, test_idx = group_split_by_file(
            file_ids, seed=SEED,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        )
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    split_info = {
        "split_mode": SPLIT_MODE,
        "n_total": N,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }
    print(f"划分方式: {SPLIT_MODE}  比例: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"训练集: {len(train_idx):,} 样本")
    print(f"验证集: {len(val_idx):,} 样本")
    print(f"测试集: {len(test_idx):,} 样本")

    # ---- 预计算两种目标标准化（仅用训练集统计）----
    # log1p z-score
    targets_log = torch.log1p(targets_raw)
    train_log = targets_log[train_idx]
    log_mean = train_log.mean(dim=0)
    log_std = train_log.std(dim=0) + 1e-8
    targets_log_norm = (targets_log - log_mean) / log_std

    # raw z-score（修正旧版全局统计的泄漏）
    train_raw = targets_raw[train_idx]
    raw_mean = train_raw.mean(dim=0)
    raw_std = train_raw.std(dim=0) + 1e-8
    targets_raw_norm = (targets_raw - raw_mean) / raw_std

    print(f"\nlog1p 目标: mean=[{log_mean[0]:.3f}, {log_mean[1]:.3f}], "
          f"std=[{log_std[0]:.3f}, {log_std[1]:.3f}]")
    print(f"raw 目标:   mean=[{raw_mean[0]:.2f}, {raw_mean[1]:.2f}], "
          f"std=[{raw_std[0]:.2f}, {raw_std[1]:.2f}]")

    # ---- 训练集非零样本加权 ----
    train_targets_raw = targets_raw[train_idx]
    is_nonzero = (train_targets_raw[:, 0] > 0).float()
    sample_weights = torch.where(is_nonzero.bool(),
                                 torch.tensor(2.0), torch.tensor(1.0))

    # ---- 序数 bins 将在每个模型循环内按 config 计算 ----

    # ---- 依次训练选中的模型 ----
    all_results = {}

    for model_name in MODELS_TO_RUN:
        cfg = MODEL_CONFIGS[model_name]
        use_log1p = cfg.get("target_transform") == "log1p"

        if use_log1p:
            t_norm = targets_log_norm
            t_mean, t_std = log_mean, log_std
        else:
            t_norm = targets_raw_norm
            t_mean, t_std = raw_mean, raw_std

        print(f"\n{'=' * 60}")
        print(f"  开始训练: {model_name} (target={cfg.get('target_transform', 'raw')})")
        print(f"{'=' * 60}")

        # ---- 按模型配置计算序数 bins ----
        thresholds = cfg.get("ordinal_thresholds", [0, 30])
        if cfg.get("aux_loss_weight", 0) > 0:
            ordinal_bins = torch.zeros(N, dtype=torch.long)
            for ti, th in enumerate(thresholds):
                ordinal_bins[targets_raw[:, 0] > th] = ti + 1
            n_bins = len(thresholds) + 1
            bin_counts = [(ordinal_bins == i).sum().item() for i in range(n_bins)]
            print(f"序数 bins ({n_bins} 类, 阈值={thresholds}): {bin_counts}")
        else:
            ordinal_bins = torch.zeros(N, dtype=torch.long)

        # 构建数据集
        train_set = AugPortDataset(
            matrices[train_idx], t_norm[train_idx],
            hours[train_idx], qc_counts[train_idx],
            bins=ordinal_bins[train_idx],
        )
        val_set = PortDataset(
            matrices[val_idx], t_norm[val_idx],
            hours[val_idx], qc_counts[val_idx],
            bins=ordinal_bins[val_idx],
        )
        test_set = PortDataset(
            matrices[test_idx], t_norm[test_idx],
            hours[test_idx], qc_counts[test_idx],
            bins=ordinal_bins[test_idx],
        )

        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_idx), replacement=True,
        )

        train_loader = DataLoader(
            train_set, batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_set, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )

        model = cfg["model_cls"](**cfg["model_kwargs"]).to(device)
        n_params = count_parameters(model)
        print(f"参数量: {n_params:,}\n")

        summary = train_model(
            model_name, model, train_loader, val_loader,
            cfg, device, t_mean, t_std, use_log1p=use_log1p,
        )

        # ---- 测试集评估（加载最优 checkpoint）----
        ckpt = torch.load(EXPERIMENTS_DIR / model_name / "best.pt",
                          weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        criterion = make_criterion(cfg.get("loss", "mse"))
        test_loss, test_mae, test_rmse = evaluate(
            model, test_loader, criterion, device, t_mean, t_std,
            use_log1p=use_log1p, use_tta=cfg.get("use_tta", False),
        )
        summary["test_loss"] = round(test_loss, 6)
        summary["test_mae_teu"] = round(test_mae[0].item(), 4)
        summary["test_mae_move"] = round(test_mae[1].item(), 4)
        summary["test_rmse_teu"] = round(test_rmse[0].item(), 4)
        summary["test_rmse_move"] = round(test_rmse[1].item(), 4)

        all_results[model_name] = {
            "params": n_params,
            "config": {k: v for k, v in cfg.items()
                       if k not in ("model_cls", "model_kwargs")},
            "model_kwargs": cfg["model_kwargs"],
            "summary": summary,
        }

        print(f"\n✓ {model_name} 完成 — "
              f"最优 epoch {summary['best_epoch']}, "
              f"MAE_TEU={summary['best_mae_teu']:.2f}, "
              f"MAE_move={summary['best_mae_move']:.2f}")
        print(f"  测试集: MAE_TEU={summary['test_mae_teu']:.2f}, "
              f"MAE_move={summary['test_mae_move']:.2f}")

    # ---- 合并旧结果 ----
    json_path = EXPERIMENTS_DIR / "results.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            old_results = json.load(f)
        # 合并：新结果覆盖同名旧结果
        old_results.update(all_results)
        all_results = old_results

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结构化结果已保存: {json_path}")

    # ---- 保存工作日志 ----
    write_work_log(all_results, env_info, split_info)

    # ---- 终端最终对比 ----
    print(f"\n{'=' * 72}")
    print("  最终结果对比")
    print(f"{'=' * 72}")
    header = (f"{'方案':<16} | {'参数量':>8} | {'Val Loss':>10} | "
              f"{'MAE_TEU':>8} | {'MAE_move':>9} | {'训练时间':>8}")
    print(header)
    print("-" * len(header))
    for name, res in all_results.items():
        r = res["summary"]
        print(f"{name:<16} | {res['params']:>8,} | {r['best_val_loss']:>10.6f} | "
              f"{r['best_mae_teu']:>8.2f} | {r['best_mae_move']:>9.2f} | "
              f"{r['total_time_sec']:>7.0f}s")

    print("\n实验全部完成!")


if __name__ == "__main__":
    main()
