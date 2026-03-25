"""
港口集装箱区栏数据预处理
将xlsx文件转为 (C, H, W) 张量 + 目标值，保存为 .pt 文件
"""
import os
import re
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ======================== 常量 ========================
# 栏位映射 (跳过 I 和 O，避免与 1、0 混淆)
LAN_LIST = list("ABCDEFGHJKLMNPQRSTUVWX")  # 22个
LAN_MAP = {c: i for i, c in enumerate(LAN_LIST)}

QU_RANGE = 7   # 区 0~6
LAN_RANGE = 22  # 栏 A~X

# 特征通道定义（6个有意义特征）
FEATURE_NAMES = [
    "龙门吊数量",
    "待完成指令数",
    "待完成指令得分",
    "饱和时间（秒）",
    "作业优先级",      # 编码: 高=3, 中=2, 低=1, 无=0
    "合理性得分",
]
NUM_CHANNELS = len(FEATURE_NAMES)

# 优先级编码
PRIORITY_MAP = {"高优先级": 3, "中优先级": 2, "低优先级": 1, "无作业": 0}

DATA_DIR = Path("提取的所有船区栏动态变化（含饱和时间）")
OUTPUT_FILE = Path("processed_data.pt")


def extract_hour(sheet_name: str) -> int:
    """从sheet名中提取小时数，如 '113590767_切片_20230820_1234_2' → 12"""
    m = re.search(r"_(\d{4})_\d+$", sheet_name)
    if m:
        return int(m.group(1)[:2])
    return 12  # fallback


def get_qc_count(xl: pd.ExcelFile) -> int:
    """从文件最后一个sheet的QC子表中获取桥吊数量"""
    for sh in reversed(xl.sheet_names):
        df_raw = xl.parse(sh, header=None)
        for i, row in df_raw.iterrows():
            val = row.iloc[0]
            if pd.notna(val) and str(val).strip() == "桥吊QC编号":
                qc_rows = df_raw.iloc[i + 1 :].dropna(how="all")
                return max(len(qc_rows), 1)
    return 1  # fallback


def parse_sheet(xl: pd.ExcelFile, sheet_name: str, qc_count: int):
    """
    解析一个sheet，返回:
    - matrix: (6, 7, 22) numpy array
    - target: (avg_teu, avg_move)
    - hour: int
    """
    df = xl.parse(sheet_name, header=0)

    # 过滤有效区栏行（排除QC子表和NaN行）
    valid_mask = df["区"].apply(
        lambda x: str(x).replace(".", "").replace("-", "").isdigit()
        if pd.notna(x)
        else False
    )
    df_valid = df[valid_mask].copy()

    if len(df_valid) == 0:
        return None

    # 构建 (6, 7, 22) 矩阵，默认填0
    matrix = np.zeros((NUM_CHANNELS, QU_RANGE, LAN_RANGE), dtype=np.float32)

    for _, row in df_valid.iterrows():
        qu = int(float(row["区"]))
        lan_str = str(row["栏"]).strip().upper()

        if qu < 0 or qu >= QU_RANGE:
            continue
        if lan_str not in LAN_MAP:
            continue
        lan = LAN_MAP[lan_str]

        # 通道0: 龙门吊数量
        matrix[0, qu, lan] = float(row.get("龙门吊数量", 0) or 0)
        # 通道1: 待完成指令数
        matrix[1, qu, lan] = float(row.get("待完成指令数", 0) or 0)
        # 通道2: 待完成指令得分
        matrix[2, qu, lan] = float(row.get("待完成指令得分", 0) or 0)
        # 通道3: 饱和时间（秒）
        matrix[3, qu, lan] = float(row.get("饱和时间（秒）", 0) or 0)
        # 通道4: 作业优先级（编码）
        priority_str = str(row.get("作业优先级", "无作业")).strip()
        matrix[4, qu, lan] = PRIORITY_MAP.get(priority_str, 0)
        # 通道5: 合理性得分
        matrix[5, qu, lan] = float(row.get("合理性得分", 0) or 0)

    # 目标值：平均QC效率
    teu_col = [c for c in df.columns if "TEU" in str(c)]
    move_col = [c for c in df.columns if "move" in str(c)]

    total_teu = 0.0
    total_move = 0.0
    if teu_col:
        vals = pd.to_numeric(df_valid[teu_col[0]], errors="coerce")
        total_teu = float(vals.max()) if len(vals) > 0 else 0.0
    if move_col:
        vals = pd.to_numeric(df_valid[move_col[0]], errors="coerce")
        total_move = float(vals.max()) if len(vals) > 0 else 0.0

    if np.isnan(total_teu):
        total_teu = 0.0
    if np.isnan(total_move):
        total_move = 0.0

    avg_teu = total_teu / qc_count
    avg_move = total_move / qc_count

    hour = extract_hour(sheet_name)

    return matrix, (avg_teu, avg_move), hour


def process_one_file(filepath: str, file_id: int):
    """处理一个xlsx文件，返回所有样本"""
    samples = []
    try:
        xl = pd.ExcelFile(filepath)
        qc_count = get_qc_count(xl)

        for sh in xl.sheet_names:
            result = parse_sheet(xl, sh, qc_count)
            if result is not None:
                matrix, target, hour = result
                samples.append({
                    "matrix": matrix,
                    "target": np.array(target, dtype=np.float32),
                    "hour": hour,
                    "qc_count": qc_count,
                    "file_id": file_id,
                })
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(filepath)}: {e}")
    return samples


def main():
    files = sorted(DATA_DIR.glob("*.xlsx"))
    print(f"共 {len(files)} 个文件待处理")

    all_samples = []
    total_files = len(files)

    # 顺序处理（xlsx读写不太适合多进程共享）
    for idx, fp in enumerate(files):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  处理中: {idx + 1}/{total_files} ...")
        samples = process_one_file(str(fp), idx)
        all_samples.extend(samples)

    print(f"\n总样本数: {len(all_samples)}")

    # 拆分为张量
    matrices = np.stack([s["matrix"] for s in all_samples])       # (N, 6, 7, 22)
    targets = np.stack([s["target"] for s in all_samples])         # (N, 2)
    hours = np.array([s["hour"] for s in all_samples], dtype=np.int64)  # (N,)
    qc_counts = np.array([s["qc_count"] for s in all_samples], dtype=np.int64)
    file_ids = np.array([s["file_id"] for s in all_samples], dtype=np.int64)

    # 逐通道标准化（z-score），保存均值和标准差用于推理
    channel_means = []
    channel_stds = []
    for c in range(NUM_CHANNELS):
        ch_data = matrices[:, c, :, :]
        mu = ch_data.mean()
        sigma = ch_data.std() + 1e-8
        matrices[:, c, :, :] = (ch_data - mu) / sigma
        channel_means.append(mu)
        channel_stds.append(sigma)

    # 目标标准化
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0) + 1e-8
    targets_norm = (targets - target_mean) / target_std

    # 统计信息
    print(f"矩阵形状: {matrices.shape}")
    print(f"目标均值: TEU_avg={target_mean[0]:.4f}, move_avg={target_mean[1]:.4f}")
    print(f"目标标准差: TEU_std={target_std[0]:.4f}, move_std={target_std[1]:.4f}")
    print(f"非零目标样本: {(targets[:, 0] > 0).sum()} / {len(targets)}")

    # 保存
    data = {
        "matrices": torch.from_numpy(matrices),
        "targets": torch.from_numpy(targets),
        "targets_norm": torch.from_numpy(targets_norm),
        "hours": torch.from_numpy(hours),
        "qc_counts": torch.from_numpy(qc_counts),
        "file_ids": torch.from_numpy(file_ids),
        "channel_means": torch.tensor(channel_means),
        "channel_stds": torch.tensor(channel_stds),
        "target_mean": torch.from_numpy(target_mean),
        "target_std": torch.from_numpy(target_std),
        "feature_names": FEATURE_NAMES,
        "lan_list": LAN_LIST,
    }
    torch.save(data, OUTPUT_FILE)
    print(f"\n已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
