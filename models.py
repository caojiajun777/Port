"""
港口区栏六通道图像建模 — 三套实验方案

方案 1: PortCNN      轻量卷积回归（工程基线）
方案 2: PortViT      全量 MHA 注意力（研究对照）
方案 3: PortFormer   CNN + Transformer 混合（主力候选）

输入统一接口:
    forward(matrix, hour, qc_cnt)
    matrix:  (B, 6, 7, 22)
    hour:    (B,)  int  0~23
    qc_cnt:  (B,)  int  0~20
输出:
    (B, 2)  → [avg_TEU_eff, avg_move_eff]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  通用构建块
# ================================================================

class ResBlock(nn.Module):
    """保持空间尺寸不变的残差卷积块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        s = x.mean(dim=[2, 3])              # (B, C)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(s.size(0), -1, 1, 1)


class MultiHeadAttention(nn.Module):
    """全量多头自注意力"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        H, d = self.num_heads, self.head_dim
        q = self.W_q(x).view(B, N, H, d).transpose(1, 2)
        k = self.W_k(x).view(B, N, H, d).transpose(1, 2)
        v = self.W_v(x).view(B, N, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    """从 token 序列中学习加权汇总"""
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x):
        # x: (B, N, D)
        scores = (x @ self.query) / math.sqrt(x.size(-1))  # (B, N)
        weights = F.softmax(scores, dim=-1)                 # (B, N)
        return (x * weights.unsqueeze(-1)).sum(dim=1)       # (B, D)


# ================================================================
#  方案 1: Port-CNN 轻量卷积回归
#
#  (B,6,7,22) → Conv Stem → ResBlock×2 → Conv升维 → ResBlock×2
#  → SE通道注意力 → GAP → 拼接辅助特征 → MLP → (B,2)
# ================================================================

class PortCNN(nn.Module):
    def __init__(self, in_channels=6, num_hours=24, max_qc=20, dropout=0.1):
        super().__init__()
        # Stem: 6 → 32, 保持 7×22
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Stage 1: 32 → 32
        self.stage1 = nn.Sequential(ResBlock(32), ResBlock(32))
        # Stage 2: 32 → 64
        self.up_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(ResBlock(64), ResBlock(64))
        # 通道注意力
        self.se = SEBlock(64)
        # 辅助特征
        self.hour_embed = nn.Embedding(num_hours, 16)
        self.qc_embed = nn.Embedding(max_qc + 1, 16)
        # 回归头: 64+16+16=96
        self.head = nn.Sequential(
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, matrix, hour, qc_cnt):
        x = self.stem(matrix)        # (B, 32, 7, 22)
        x = self.stage1(x)           # (B, 32, 7, 22)
        x = self.up_conv(x)          # (B, 64, 7, 22)
        x = self.stage2(x)           # (B, 64, 7, 22)
        x = self.se(x)               # (B, 64, 7, 22)
        x = x.mean(dim=[2, 3])       # (B, 64) GAP

        h = self.hour_embed(hour.clamp(0, 23))    # (B, 16)
        q = self.qc_embed(qc_cnt.clamp(0, 20))    # (B, 16)
        feat = torch.cat([x, h, q], dim=-1)        # (B, 96)
        return self.head(feat)                      # (B, 2)


# ================================================================
#  方案 2: Port-ViT 全量 MHA
#
#  (B,6,7,22) → reshape (B,154,6) → Linear → +2D pos → +CLS
#  → Transformer×4 → CLS out → 拼接辅助特征 → MLP → (B,2)
# ================================================================

class PortViT(nn.Module):
    def __init__(self, in_channels=6, grid_h=7, grid_w=22,
                 d_model=96, num_layers=4, num_heads=8, d_ff=384,
                 dropout=0.1, num_hours=24, max_qc=20):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_model = d_model

        # Token 投影
        self.token_proj = nn.Linear(in_channels, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # 2D 分离式位置编码
        self.row_embed = nn.Embedding(grid_h, d_model)
        self.col_embed = nn.Embedding(grid_w, d_model)
        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # 辅助特征
        self.hour_embed = nn.Embedding(num_hours, 32)
        self.qc_embed = nn.Embedding(max_qc + 1, 32)
        # 回归头: 96+32+32=160
        self.head = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, matrix, hour, qc_cnt):
        B = matrix.shape[0]
        # (B,6,7,22) → (B,154,6) → (B,154,d)
        x = matrix.permute(0, 2, 3, 1).reshape(B, self.grid_h * self.grid_w, -1)
        x = self.token_proj(x)

        # 2D 位置编码: E_row(i) + E_col(j)
        row_ids = torch.arange(self.grid_h, device=x.device)
        col_ids = torch.arange(self.grid_w, device=x.device)
        pos = (self.row_embed(row_ids).unsqueeze(1)
               + self.col_embed(col_ids).unsqueeze(0))
        pos = pos.reshape(-1, self.d_model)    # (154, d)
        x = x + pos.unsqueeze(0)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls, x], dim=1)         # (B, 155, d)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0, :]                   # (B, d)

        h = self.hour_embed(hour.clamp(0, 23))  # (B, 32)
        q = self.qc_embed(qc_cnt.clamp(0, 20))  # (B, 32)
        aux = torch.cat([h, q], dim=-1)          # (B, 64)
        feat = torch.cat([cls_out, aux], dim=-1) # (B, d+64)
        return self.head(feat)                    # (B, 2)


# ================================================================
#  方案 3: Local-Global PortFormer 混合架构
#
#  (B,6,7,22) → Conv Stem → ResBlock×2 → 1×1 Conv(→64)
#  → flatten (B,154,64) → +2D pos → Transformer×3
#  → Attention Pooling → 拼接辅助特征 → MLP → (B,2)
# ================================================================

class PortFormer(nn.Module):
    def __init__(self, in_channels=6, grid_h=7, grid_w=22,
                 conv_dim=32, d_model=64, num_layers=3, num_heads=8, d_ff=256,
                 dropout=0.1, num_hours=24, max_qc=20):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_model = d_model

        # ---- CNN 局部编码 ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.GELU(),
        )
        self.local_blocks = nn.Sequential(ResBlock(conv_dim), ResBlock(conv_dim))
        self.proj = nn.Sequential(
            nn.Conv2d(conv_dim, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # ---- 位置编码 ----
        self.row_embed = nn.Embedding(grid_h, d_model)
        self.col_embed = nn.Embedding(grid_w, d_model)

        # ---- Transformer 全局建模 ----
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ---- Attention Pooling ----
        self.pool = AttentionPooling(d_model)

        # ---- 辅助特征 ----
        self.hour_embed = nn.Embedding(num_hours, 16)
        self.qc_embed = nn.Embedding(max_qc + 1, 16)

        # ---- 回归头: 64+16+16=96 ----
        self.head = nn.Sequential(
            nn.Linear(d_model + 32, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, matrix, hour, qc_cnt):
        B = matrix.shape[0]
        # CNN 局部编码
        x = self.stem(matrix)          # (B, 32, 7, 22)
        x = self.local_blocks(x)       # (B, 32, 7, 22)
        x = self.proj(x)               # (B, 64, 7, 22)

        # 展平为 token 序列
        x = x.flatten(2).transpose(1, 2)  # (B, 154, 64)

        # 位置编码
        row_ids = torch.arange(self.grid_h, device=x.device)
        col_ids = torch.arange(self.grid_w, device=x.device)
        pos = (self.row_embed(row_ids).unsqueeze(1)
               + self.col_embed(col_ids).unsqueeze(0))
        pos = pos.reshape(-1, self.d_model)
        x = x + pos.unsqueeze(0)

        # Transformer 全局建模
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Attention Pooling
        x = self.pool(x)               # (B, 64)

        # 辅助特征
        h = self.hour_embed(hour.clamp(0, 23))    # (B, 16)
        q = self.qc_embed(qc_cnt.clamp(0, 20))    # (B, 16)
        feat = torch.cat([x, h, q], dim=-1)        # (B, 96)
        return self.head(feat)                      # (B, 2)


# ================================================================
#  PortCNN-Plus 新增构建块
# ================================================================

class MultiScaleStem(nn.Module):
    """多尺度条带卷积 Stem：1×3 + 3×1 + 3×3 并联"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        br = out_ch // 3
        self.h_conv = nn.Conv2d(in_ch, br, (1, 3), padding=(0, 1), bias=False)
        self.v_conv = nn.Conv2d(in_ch, br, (3, 1), padding=(1, 0), bias=False)
        self.s_conv = nn.Conv2d(in_ch, br, 3, padding=1, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(br * 3),
            nn.GELU(),
            nn.Conv2d(br * 3, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.h_conv(x), self.v_conv(x), self.s_conv(x)], dim=1))


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: x' = x * (1 + γ) + β"""
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, channels * 2)
        # 初始化为零，使初始行为接近 identity
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        params = self.fc(cond)                               # (B, C*2)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(gamma.size(0), -1, 1, 1)
        beta = beta.view(beta.size(0), -1, 1, 1)
        return x * (1 + gamma) + beta


class CoordinateAttention(nn.Module):
    """轴向坐标注意力：行和列方向分别建模"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.attn_h = nn.Conv2d(mid, channels, 1)
        self.attn_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        pool_h = x.mean(dim=3, keepdim=True)               # (B, C, H, 1)
        pool_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, W, 1)
        cat = torch.cat([pool_h, pool_w], dim=2)            # (B, C, H+W, 1)
        cat = self.squeeze(cat)
        h_feat, w_feat = cat.split([H, W], dim=2)
        h_att = self.attn_h(h_feat).sigmoid()               # (B, C, H, 1)
        w_att = self.attn_w(w_feat).permute(0, 1, 3, 2).sigmoid()  # (B, C, 1, W)
        return x * h_att * w_att


# ================================================================
#  方案 4: PortCNN-Plus 针对 7×22 小图优化的增强卷积
#
#  改进:
#    1. 行列位置编码 → 8 通道输入
#    2. 多尺度条带卷积 Stem (1×3 + 3×1 + 3×3)
#    3. FiLM 条件调制 hour/qc_count
#    4. Coordinate Attention 轴向全局感知
#    5. 双头回归 (TEU head + move head)
#    6. 配合 Huber 损失使用
# ================================================================

class PortCNNPlus(nn.Module):
    def __init__(self, in_channels=6, grid_h=7, grid_w=22,
                 stem_ch=48, stage_ch=64, cond_dim=32,
                 num_hours=24, max_qc=20, dropout=0.1,
                 use_multiscale_stem=True, use_pos_enc=True,
                 use_film=True, use_coord_attn=True, use_dual_head=True,
                 headless=False, use_sheet_cond=False, max_sheet_idx=256):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.use_film = use_film
        self.use_coord_attn = use_coord_attn
        self.use_dual_head = use_dual_head
        self.headless = headless
        self.use_sheet_cond = use_sheet_cond
        self.max_sheet_idx = max_sheet_idx
        self.feat_dim = stage_ch + (0 if use_film else 32)

        # ---- 行列位置编码 ----
        if use_pos_enc:
            self.row_embed = nn.Parameter(torch.randn(1, 1, grid_h, 1) * 0.02)
            self.col_embed = nn.Parameter(torch.randn(1, 1, 1, grid_w) * 0.02)
            actual_in = in_channels + 2  # 8
        else:
            actual_in = in_channels  # 6

        # ---- Stem ----
        if use_multiscale_stem:
            self.stem = MultiScaleStem(actual_in, stem_ch)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(actual_in, stem_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(stem_ch),
                nn.GELU(),
            )

        # ---- 条件向量 (FiLM 用) ----
        n_cond = 3 if use_sheet_cond else 2
        if use_film:
            self.hour_cond = nn.Embedding(num_hours, cond_dim)
            self.qc_cond = nn.Embedding(max_qc + 1, cond_dim)
            if use_sheet_cond:
                self.sheet_cond = nn.Embedding(max_sheet_idx, cond_dim)
        else:
            self.hour_embed = nn.Embedding(num_hours, 16)
            self.qc_embed = nn.Embedding(max_qc + 1, 16)

        # ---- Stage 1 + FiLM ----
        self.stage1 = nn.Sequential(ResBlock(stem_ch), ResBlock(stem_ch))
        if use_film:
            self.film1 = FiLM(cond_dim * n_cond, stem_ch)

        # ---- Stage 2 + FiLM ----
        self.up_conv = nn.Sequential(
            nn.Conv2d(stem_ch, stage_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(stage_ch),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(ResBlock(stage_ch), ResBlock(stage_ch))
        if use_film:
            self.film2 = FiLM(cond_dim * n_cond, stage_ch)

        # ---- Coordinate Attention + SE ----
        if use_coord_attn:
            self.coord_attn = CoordinateAttention(stage_ch)
        self.se = SEBlock(stage_ch)

        # ---- 回归头（headless 模式不创建）----
        if not headless:
            head_in = self.feat_dim
            if use_dual_head:
                self.teu_head = nn.Sequential(
                    nn.Linear(head_in, 48), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(48, 1),
                )
                self.move_head = nn.Sequential(
                    nn.Linear(head_in, 48), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(48, 1),
                )
            else:
                self.head = nn.Sequential(
                    nn.Linear(head_in, 64), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(64, 2),
                )
        self._init_weights()

    def _init_weights(self):
        film_linears = set()
        if self.use_film:
            film_linears.add(id(self.film1.fc))
            film_linears.add(id(self.film2.fc))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear) and id(m) not in film_linears:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, matrix, hour, qc_cnt, sheet_idx=None):
        """提取特征向量 (B, feat_dim)，不经过回归头"""
        B = matrix.shape[0]
        if self.use_pos_enc:
            row_map = self.row_embed.expand(B, -1, -1, matrix.size(3))
            col_map = self.col_embed.expand(B, -1, matrix.size(2), -1)
            x = torch.cat([matrix, row_map, col_map], dim=1)
        else:
            x = matrix

        x = self.stem(x)
        x = self.stage1(x)
        if self.use_film:
            h_c = self.hour_cond(hour.clamp(0, 23))
            q_c = self.qc_cond(qc_cnt.clamp(0, 20))
            parts = [h_c, q_c]
            if self.use_sheet_cond and sheet_idx is not None:
                s_c = self.sheet_cond(sheet_idx.clamp(0, self.max_sheet_idx - 1))
                parts.append(s_c)
            cond = torch.cat(parts, dim=-1)
            x = self.film1(x, cond)

        x = self.up_conv(x)
        x = self.stage2(x)
        if self.use_film:
            x = self.film2(x, cond)

        if self.use_coord_attn:
            x = self.coord_attn(x)
        x = self.se(x)
        x = x.mean(dim=[2, 3])

        if not self.use_film:
            h = self.hour_embed(hour.clamp(0, 23))
            q = self.qc_embed(qc_cnt.clamp(0, 20))
            x = torch.cat([x, h, q], dim=-1)
        return x

    def forward(self, matrix, hour, qc_cnt, sheet_idx=None):
        x = self.encode(matrix, hour, qc_cnt, sheet_idx=sheet_idx)
        if self.headless:
            return x
        if self.use_dual_head:
            teu = self.teu_head(x)
            move = self.move_head(x)
            return torch.cat([teu, move], dim=-1)
        else:
            return self.head(x)


# ================================================================
#  方案 5: PortMoE — 门控混合专家回归网络
#
#  创新点:
#    1. 共享 PortCNNPlus 编码器提取空间特征
#    2. 门控网络将样本软路由到 K 个专家头
#    3. 辅助序数分类损失引导门控学习
#  解决问题:
#    - 目标分布跨越 [0, 600): 40% 零值 + 重尾
#    - 单回归头被迫"回归均值"，低值高估、高值低估
#    - K 个专家各自负责不同值域的局部回归
# ================================================================

class PortMoE(nn.Module):
    def __init__(self, in_channels=6, grid_h=7, grid_w=22,
                 stem_ch=48, stage_ch=64, cond_dim=32,
                 num_hours=24, max_qc=20, dropout=0.1,
                 num_experts=3, expert_hidden=48,
                 heteroscedastic=False,
                 use_sheet_cond=False, max_sheet_idx=256):
        super().__init__()
        self.num_experts = num_experts
        self.heteroscedastic = heteroscedastic

        # 共享编码器 (PortCNNPlus backbone, headless)
        self.backbone = PortCNNPlus(
            in_channels=in_channels, grid_h=grid_h, grid_w=grid_w,
            stem_ch=stem_ch, stage_ch=stage_ch, cond_dim=cond_dim,
            num_hours=num_hours, max_qc=max_qc, dropout=dropout,
            headless=True,
            use_sheet_cond=use_sheet_cond, max_sheet_idx=max_sheet_idx,
        )
        feat_dim = self.backbone.feat_dim  # 64

        # 门控网络 (兼做辅助序数分类器)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, num_experts),
        )

        # K 个专家回归头
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, 2),
            )
            for _ in range(num_experts)
        ])

        # 异方差: 学习每个样本的 log-variance
        if heteroscedastic:
            self.logvar_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 2),
                nn.GELU(),
                nn.Linear(feat_dim // 2, 2),
            )

        self._init_moe_weights()

    def _init_moe_weights(self):
        modules = [self.gate, *self.experts]
        if self.heteroscedastic:
            modules.append(self.logvar_head)
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, matrix, hour, qc_cnt, sheet_idx=None):
        features = self.backbone.encode(matrix, hour, qc_cnt, sheet_idx=sheet_idx)  # (B, 64)

        gate_logits = self.gate(features)                       # (B, K)
        gate_weights = F.softmax(gate_logits, dim=-1)           # (B, K)

        expert_preds = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )                                                        # (B, K, 2)

        preds = (gate_weights.unsqueeze(-1) * expert_preds).sum(dim=1)  # (B, 2)

        if self.training:
            if self.heteroscedastic:
                logvar = self.logvar_head(features)              # (B, 2)
                return preds, logvar, gate_logits
            return preds, gate_logits
        return preds


# ================================================================
#  工具函数
# ================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_REGISTRY = {
    "PortCNN": PortCNN,
    "PortViT": PortViT,
    "PortFormer": PortFormer,
    "PortCNNPlus": PortCNNPlus,
    "PortMoE": PortMoE,
}


# ================================================================
#  自测
# ================================================================

if __name__ == "__main__":
    B = 4
    matrix = torch.randn(B, 6, 7, 22)
    hour = torch.randint(0, 24, (B,))
    qc_cnt = torch.randint(1, 16, (B,))

    for name, cls in MODEL_REGISTRY.items():
        m = cls()
        m.eval()  # eval mode: all models return plain tensor
        out = m(matrix, hour, qc_cnt)
        print(f"{name:16s}  params={count_parameters(m):>8,}  output={out.shape}")
