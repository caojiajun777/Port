"""快速推理消融脚本——复用 k=2 checkpoint，评估 TTA/Ensemble"""
import torch, numpy as np, json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from models import PortMoE

ROOT = Path(".")
EXPERIMENTS_DIR = ROOT / "experiments"
DATA_PATH = ROOT / "processed_data.pt"
ENSEMBLE_SEEDS = [42, 123, 456, 789]
K = 2
MODEL_KWARGS = dict(in_channels=12, stem_ch=48, stage_ch=64, cond_dim=32,
                    num_experts=4, expert_hidden=64, heteroscedastic=True)

data = torch.load(DATA_PATH, weights_only=False)
matrices_raw, targets, hours, qc_counts, file_ids = (
    data["matrices"], data["targets"], data["hours"], data["qc_counts"], data["file_ids"])
N, C, H, W = matrices_raw.shape
temporal = torch.zeros(N, C * K, H, W)
file_to_idx = {}
for i in range(N):
    fid = file_ids[i].item()
    file_to_idx.setdefault(fid, []).append(i)
for fid, idxs in file_to_idx.items():
    for pos, idx in enumerate(idxs):
        for fi in range(K):
            t = pos - (K - 1 - fi)
            if t >= 0:
                temporal[idx, fi * C:(fi + 1) * C] = matrices_raw[idxs[t]]

unique_files = np.unique(file_ids.numpy())
rng = np.random.RandomState(42)
perm = rng.permutation(len(unique_files))
n_tr = int(len(unique_files) * 0.8)
n_v = int(len(unique_files) * 0.1)
train_f = set(unique_files[perm[:n_tr]])
test_f = set(unique_files[perm[n_tr + n_v:]])
fid_np = file_ids.numpy()
train_idx = np.where(np.isin(fid_np, list(train_f)))[0]
test_idx = np.where(np.isin(fid_np, list(test_f)))[0]
t_mean = targets[train_idx].mean(0)
t_std = targets[train_idx].std(0) + 1e-8
targets_z = (targets - t_mean) / t_std

test_set = TensorDataset(temporal[test_idx], targets_z[test_idx],
                         hours[test_idx], qc_counts[test_idx])
loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = []
for s in ENSEMBLE_SEEDS:
    m = PortMoE(**MODEL_KWARGS).to(device)
    ckpt_path = EXPERIMENTS_DIR / f"ablation-temporal-k2/seed_{s}/best.pt"
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    models.append(m)
    epoch = ckpt["epoch"]
    print(f"  loaded seed={s} epoch={epoch}")


@torch.no_grad()
def run(ms, tta):
    all_p, all_t = [], []
    first = True
    for mi in ms:
        ps = []
        for b in loader:
            x, t, h, q = b[0].to(device), b[1], b[2].to(device), b[3].to(device)
            p = mi(x, h, q)
            if tta:
                p = (p + mi(x.flip(-1), h, q)) * 0.5
            ps.append(p.cpu())
            if first:
                all_t.append(t)
        all_p.append(torch.cat(ps))
        first = False
    tr = torch.cat(all_t) * t_std + t_mean
    pr = torch.stack(all_p).mean(0) * t_std + t_mean
    mae = (pr - tr).abs().mean(0)
    rmse = ((pr - tr) ** 2).mean(0).sqrt()
    r2 = 1 - ((pr - tr) ** 2).sum(0) / ((tr - tr.mean(0)) ** 2).sum(0)
    return mae, rmse, r2


configs = [
    ("4-Ens+TTA (完整模型)", models, True),
    ("4-Ens, 无TTA", models, False),
    ("单模型+TTA (seed=42)", [models[0]], True),
    ("单模型, 无TTA", [models[0]], False),
]
print()
results = {}
for label, ms, tta in configs:
    mae, rmse, r2 = run(ms, tta)
    print(f"  {label:<28}: MAE_TEU={mae[0]:.4f}  MAE_move={mae[1]:.4f}  "
          f"RMSE={rmse[0]:.4f}  R2={r2[0]:.4f}")
    results[label] = dict(mae_teu=round(mae[0].item(), 4),
                          mae_move=round(mae[1].item(), 4),
                          rmse_teu=round(rmse[0].item(), 4),
                          r2_teu=round(r2[0].item(), 4))

out_path = EXPERIMENTS_DIR / "inference_ablation_quick.json"
out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n结果已保存: {out_path}")
