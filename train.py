"""
训练 Port-ViT 模型
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time
from pathlib import Path
from model import PortViT, count_parameters


# ======================== 配置 ========================
class Config:
    # 数据
    data_path = "processed_data.pt"
    train_ratio = 0.8
    batch_size = 128
    num_workers = 0  # Windows 下建议 0

    # 模型
    d_model = 64
    num_layers = 4
    num_heads = 8
    d_ff = 256
    dropout = 0.1

    # 训练
    epochs = 80
    lr = 3e-4
    weight_decay = 1e-4
    warmup_epochs = 5
    min_lr = 1e-6

    # 保存
    save_dir = Path("checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ======================== 数据集 ========================
class PortDataset(Dataset):
    def __init__(self, matrices, targets_norm, hours, qc_counts):
        self.matrices = matrices
        self.targets_norm = targets_norm
        self.hours = hours
        self.qc_counts = qc_counts

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return (
            self.matrices[idx],
            self.targets_norm[idx],
            self.hours[idx],
            self.qc_counts[idx],
        )


# ======================== 学习率调度 ========================
def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    """带 warmup 的余弦退火"""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ======================== 训练 ========================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for matrices, targets, hours, qc_counts in dataloader:
        matrices = matrices.to(device)
        targets = targets.to(device)
        hours = hours.to(device)
        qc_counts = qc_counts.to(device)

        pred = model(matrices, hours, qc_counts)
        loss = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, target_mean, target_std):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    n_batches = 0

    for matrices, targets, hours, qc_counts in dataloader:
        matrices = matrices.to(device)
        targets = targets.to(device)
        hours = hours.to(device)
        qc_counts = qc_counts.to(device)

        pred = model(matrices, hours, qc_counts)
        loss = criterion(pred, targets)

        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # 反标准化计算真实 MAE
    preds = torch.cat(all_preds) * target_std + target_mean
    trues = torch.cat(all_targets) * target_std + target_mean
    mae = (preds - trues).abs().mean(dim=0)

    return avg_loss, mae


def main():
    cfg = Config()
    cfg.save_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Port-ViT 训练（MHA Transformer）")
    print("=" * 60)

    # 加载数据
    print(f"\n加载数据: {cfg.data_path}")
    data = torch.load(cfg.data_path, weights_only=False)
    matrices = data["matrices"]       # (N, 6, 7, 22)
    targets_norm = data["targets_norm"]  # (N, 2)
    hours = data["hours"]             # (N,)
    qc_counts = data["qc_counts"]     # (N,)
    target_mean = data["target_mean"]
    target_std = data["target_std"]

    N = len(matrices)
    print(f"总样本: {N}")
    print(f"目标均值: {target_mean.numpy()}")
    print(f"目标标准差: {target_std.numpy()}")

    # 构建数据集 & 划分
    dataset = PortDataset(matrices, targets_norm, hours, qc_counts)
    train_size = int(N * cfg.train_ratio)
    val_size = N - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"训练集: {train_size}, 验证集: {val_size}")

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

    # 模型
    device = torch.device(cfg.device)
    model = PortViT(
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    print(f"\n模型参数: {count_parameters(model):,}")
    print(f"注意力: 全量 MHA (num_heads={cfg.num_heads})")
    print(f"设备: {device}")

    # 优化器 & 损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    # 训练循环
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>5} | {'LR':>10} | {'Train Loss':>11} | {'Val Loss':>10} | {'MAE_TEU':>8} | {'MAE_move':>9} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch, cfg.epochs, cfg.warmup_epochs, cfg.lr, cfg.min_lr)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device, target_mean, target_std)

        elapsed = time.time() - t0

        # 保存最优
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae.numpy(),
                "target_mean": target_mean,
                "target_std": target_std,
                "config": {
                    "d_model": cfg.d_model,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "d_ff": cfg.d_ff,
                },
            }, cfg.save_dir / "best_model.pt")
            marker = " *"
        else:
            marker = ""

        print(f"{epoch + 1:5d} | {lr:10.6f} | {train_loss:11.6f} | {val_loss:10.6f} | "
              f"{val_mae[0]:8.4f} | {val_mae[1]:9.4f} | {elapsed:5.1f}s{marker}")

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存至 {cfg.save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
