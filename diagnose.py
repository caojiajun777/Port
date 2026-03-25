"""PortMoEv2 vs PortMoE vs PortCNNPlus 分区间诊断分析"""
import torch
import numpy as np
from models import PortMoE, PortCNNPlus
from torch.utils.data import TensorDataset, DataLoader

# Load data
data = torch.load('processed_data.pt', weights_only=False)
matrices = data['matrices']
targets = data['targets']
hours = data['hours']
qc_counts = data['qc_counts']

# Same split as training
rng = np.random.RandomState(42)
perm = rng.permutation(len(matrices))
n_train = int(len(matrices) * 0.80)
n_val = int(len(matrices) * 0.10)
test_idx = perm[n_train + n_val:].tolist()
train_idx = perm[:n_train].tolist()

t_mean = targets[train_idx].mean(dim=0)
t_std = targets[train_idx].std(dim=0)

test_matrices = matrices[test_idx]
test_targets = targets[test_idx]
test_hours = hours[test_idx]
test_qc = qc_counts[test_idx]

device = torch.device('cuda')

# Load models
moev2 = PortMoE(stem_ch=48, stage_ch=64, cond_dim=32, num_experts=4, expert_hidden=64)
ckpt = torch.load('experiments/PortMoEv2/best.pt', weights_only=False)
moev2.load_state_dict(ckpt['model_state_dict'])
moev2 = moev2.to(device).eval()

moe = PortMoE(stem_ch=48, stage_ch=64, cond_dim=32, num_experts=3, expert_hidden=48)
ckpt = torch.load('experiments/PortMoE/best.pt', weights_only=False)
moe.load_state_dict(ckpt['model_state_dict'])
moe = moe.to(device).eval()

cnnplus = PortCNNPlus(stem_ch=48, stage_ch=64, cond_dim=32)
ckpt_cpp = torch.load('experiments/PortCNNPlus/best.pt', weights_only=False)
cnnplus.load_state_dict(ckpt_cpp['model_state_dict'])
cnnplus = cnnplus.to(device).eval()

ds = TensorDataset(test_matrices, test_targets, test_hours, test_qc)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

def predict_all(model, loader, t_mean, t_std, device):
    all_preds, all_gates = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            m, t, h, q = [b.to(device) for b in batch]
            if hasattr(model, 'backbone'):
                features = model.backbone.encode(m, h, q)
                gate_logits = model.gate(features)
                gate_weights = torch.softmax(gate_logits, dim=-1)
                expert_preds = torch.stack([e(features) for e in model.experts], dim=1)
                pred = (gate_weights.unsqueeze(-1) * expert_preds).sum(dim=1)
                all_gates.append(gate_weights.cpu())
            else:
                pred = model(m, h, q)
            pred_real = pred.cpu() * t_std + t_mean
            all_preds.append(pred_real)
    preds = torch.cat(all_preds).numpy()
    gates = torch.cat(all_gates).numpy() if all_gates else None
    return preds, gates

moe_preds, moe_gates = predict_all(moe, loader, t_mean, t_std, device)
moev2_preds, moev2_gates = predict_all(moev2, loader, t_mean, t_std, device)
cnnplus_preds, _ = predict_all(cnnplus, loader, t_mean, t_std, device)
true_vals = test_targets.numpy()

intervals = [
    ('Zero (=0)',     lambda t: t == 0),
    ('Low (0,15]',    lambda t: (t > 0) & (t <= 15)),
    ('Mid (15,50]',   lambda t: (t > 15) & (t <= 50)),
    ('High (50,100]', lambda t: (t > 50) & (t <= 100)),
    ('VHigh (>100)',  lambda t: t > 100),
]

teu_true = true_vals[:, 0]

print('=' * 105)
print('  分区间 MAE_TEU 对比 (测试集)')
print('=' * 105)
header = f"{'区间':>16s} | {'样本数':>6s} | {'CnnPlus':>8s} | {'MoE':>8s} | {'MoEv2':>8s} | {'v2 vs C+':>8s} | {'v2 vs MoE':>9s}"
print(header)
print('-' * 105)

for name, cond in intervals:
    mask = cond(teu_true)
    n = mask.sum()
    mae_cpp = np.abs(cnnplus_preds[mask, 0] - teu_true[mask]).mean()
    mae_moe = np.abs(moe_preds[mask, 0] - teu_true[mask]).mean()
    mae_v2 = np.abs(moev2_preds[mask, 0] - teu_true[mask]).mean()
    chg_cpp = (mae_v2 - mae_cpp) / mae_cpp * 100
    chg_moe = (mae_v2 - mae_moe) / mae_moe * 100
    print(f'{name:>16s} | {n:6d} | {mae_cpp:8.2f} | {mae_moe:8.2f} | {mae_v2:8.2f} | {chg_cpp:+7.1f}% | {chg_moe:+8.1f}%')

mae_cpp_all = np.abs(cnnplus_preds[:, 0] - teu_true).mean()
mae_moe_all = np.abs(moe_preds[:, 0] - teu_true).mean()
mae_v2_all = np.abs(moev2_preds[:, 0] - teu_true).mean()
print('-' * 105)
print(f'{"Overall":>16s} | {len(teu_true):6d} | {mae_cpp_all:8.2f} | {mae_moe_all:8.2f} | {mae_v2_all:8.2f} | {(mae_v2_all-mae_cpp_all)/mae_cpp_all*100:+7.1f}% | {(mae_v2_all-mae_moe_all)/mae_moe_all*100:+8.1f}%')

nz = teu_true > 0
mae_cpp_nz = np.abs(cnnplus_preds[nz, 0] - teu_true[nz]).mean()
mae_moe_nz = np.abs(moe_preds[nz, 0] - teu_true[nz]).mean()
mae_v2_nz = np.abs(moev2_preds[nz, 0] - teu_true[nz]).mean()
print(f'{"NonZero":>16s} | {nz.sum():6d} | {mae_cpp_nz:8.2f} | {mae_moe_nz:8.2f} | {mae_v2_nz:8.2f} | {(mae_v2_nz-mae_cpp_nz)/mae_cpp_nz*100:+7.1f}% | {(mae_v2_nz-mae_moe_nz)/mae_moe_nz*100:+8.1f}%')

# R²
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

print()
r2_cpp_teu = r2(teu_true, cnnplus_preds[:, 0])
r2_moe_teu = r2(teu_true, moe_preds[:, 0])
r2_v2_teu = r2(teu_true, moev2_preds[:, 0])
r2_cpp_move = r2(true_vals[:, 1], cnnplus_preds[:, 1])
r2_moe_move = r2(true_vals[:, 1], moe_preds[:, 1])
r2_v2_move = r2(true_vals[:, 1], moev2_preds[:, 1])
print(f'R² (TEU):  CnnPlus={r2_cpp_teu:.4f}  MoE={r2_moe_teu:.4f}  MoEv2={r2_v2_teu:.4f}')
print(f'R² (move): CnnPlus={r2_cpp_move:.4f}  MoE={r2_moe_move:.4f}  MoEv2={r2_v2_move:.4f}')

# Gate routing for MoEv2 (4 experts)
print()
print('=' * 80)
print('  PortMoEv2 门控路由分布 (测试集, hard assignment, K=4)')
print('=' * 80)
hard_v2 = moev2_gates.argmax(axis=1)
for name, cond in intervals:
    mask = cond(teu_true)
    if mask.sum() == 0:
        continue
    g = hard_v2[mask]
    dist = [np.mean(g == i) * 100 for i in range(4)]
    print(f'{name:>16s} (n={mask.sum():5d}): E0={dist[0]:5.1f}%  E1={dist[1]:5.1f}%  E2={dist[2]:5.1f}%  E3={dist[3]:5.1f}%')

print()
print('门控权重均值 (soft routing, K=4):')
for name, cond in intervals:
    mask = cond(teu_true)
    if mask.sum() == 0:
        continue
    g = moev2_gates[mask].mean(axis=0)
    print(f'{name:>16s}: E0={g[0]:.3f}  E1={g[1]:.3f}  E2={g[2]:.3f}  E3={g[3]:.3f}')

# Bias analysis
print()
print('=' * 95)
print('  预测偏差分析 (mean true vs mean pred, TEU)')
print('=' * 95)
header2 = f"{'区间':>16s} | {'True Mean':>10s} | {'CnnPlus':>10s} | {'MoE':>10s} | {'MoEv2':>10s} | {'C+ Bias':>8s} | {'v2 Bias':>9s}"
print(header2)
print('-' * 95)
for name, cond in intervals:
    mask = cond(teu_true)
    tm = teu_true[mask].mean()
    cp = cnnplus_preds[mask, 0].mean()
    mp = moe_preds[mask, 0].mean()
    v2p = moev2_preds[mask, 0].mean()
    print(f'{name:>16s} | {tm:10.2f} | {cp:10.2f} | {mp:10.2f} | {v2p:10.2f} | {cp-tm:+8.2f} | {v2p-tm:+9.2f}')
