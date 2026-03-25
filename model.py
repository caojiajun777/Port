"""
Port-ViT: 基于 Vision Transformer + 全量 MHA 的港口QC效率预测模型

架构：
  - 每个 (区, 栏) 格子视为一个 token（共 7×22=154 个）
  - 6 维特征 → 线性投影至 d_model
  - 可学习 2D 位置编码（行编码 + 列编码）
  - Multi-Head Attention (MHA) Transformer Encoder（全量注意力）
  - CLS token 聚合全局信息
  - 辅助标量特征（小时、QC数量）拼接后回归
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  Multi-Head Attention (MHA) — 全量注意力
#  每个 head 独立学习 Q/K/V，所有 token 对之间都计算注意力得分
#  在 154 tokens 规模下完全可承受，信息无损失
# ================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q、K、V、O 各自独立，每个 head 有完整的参数
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)   N=seq_len, D=d_model
        return: (B, N, D)
        """
        B, N, _ = x.shape
        H, d = self.num_heads, self.head_dim

        # 投影并拆分多头
        q = self.W_q(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = self.W_k(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        v = self.W_v(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)

        # 全量 scaled dot-product attention: 每对 token 都计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, N, d)
        out = out.transpose(1, 2).contiguous().view(B, N, H * d)

        return self.W_o(out)


# ================================================================
#  Transformer Encoder Block (Pre-Norm style)
# ================================================================
class TransformerBlock(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ================================================================
#  Port-ViT
# ================================================================
class PortViT(nn.Module):
    """
    输入:
      matrix: (B, C, H, W) = (B, 6, 7, 22) 特征矩阵
      hour:   (B,) 小时 0~23
      qc_cnt: (B,) QC数量

    输出:
      (B, 2) → (avg_TEU_efficiency, avg_move_efficiency)
    """

    def __init__(
        self,
        in_channels: int = 6,
        grid_h: int = 7,
        grid_w: int = 22,
        d_model: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_hours: int = 24,
        max_qc: int = 20,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.d_model = d_model
        num_tokens = grid_h * grid_w  # 154

        # ---- Token Embedding ----
        # 每个格子的 6 维特征 → d_model
        self.token_proj = nn.Linear(in_channels, d_model)

        # ---- CLS Token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- 2D Positional Encoding (可学习) ----
        # 行位置 + 列位置，分别学习后相加
        self.row_embed = nn.Embedding(grid_h, d_model)
        self.col_embed = nn.Embedding(grid_w, d_model)

        # CLS token 位置编码
        self.cls_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- Transformer Encoder ----
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ---- 辅助特征编码 ----
        self.hour_embed = nn.Embedding(num_hours, d_model // 2)
        self.qc_embed = nn.Embedding(max_qc + 1, d_model // 2)

        # ---- 回归头 ----
        self.head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
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

    def forward(self, matrix: torch.Tensor, hour: torch.Tensor, qc_cnt: torch.Tensor) -> torch.Tensor:
        B = matrix.shape[0]

        # (B, C, H, W) → (B, H, W, C) → (B, H*W, C)
        x = matrix.permute(0, 2, 3, 1).reshape(B, self.grid_h * self.grid_w, -1)

        # Token projection
        x = self.token_proj(x)  # (B, 154, d_model)

        # 2D positional encoding
        row_ids = torch.arange(self.grid_h, device=x.device)
        col_ids = torch.arange(self.grid_w, device=x.device)
        # (H, d) + (W, d) → (H, W, d) → (H*W, d)
        pos = self.row_embed(row_ids).unsqueeze(1) + self.col_embed(col_ids).unsqueeze(0)
        pos = pos.reshape(self.grid_h * self.grid_w, self.d_model)  # (154, d)
        x = x + pos.unsqueeze(0)  # broadcast over batch

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls, x], dim=1)  # (B, 155, d_model)

        # Transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # CLS output
        cls_out = x[:, 0, :]  # (B, d_model)

        # 辅助特征
        h_emb = self.hour_embed(hour.clamp(0, 23))         # (B, d_model//2)
        q_emb = self.qc_embed(qc_cnt.clamp(0, 20))         # (B, d_model//2)
        aux = torch.cat([h_emb, q_emb], dim=-1)             # (B, d_model)

        # 拼接 → 回归
        feat = torch.cat([cls_out, aux], dim=-1)  # (B, d_model + d_model)
        return self.head(feat)  # (B, 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 快速测试
    model = PortViT()
    print(f"模型参数量: {count_parameters(model):,}")
    print(f"注意力机制: 全量 MHA (num_heads=8, 每个 head 独立 Q/K/V)")
    # 模拟输入
    B = 4
    matrix = torch.randn(B, 6, 7, 22)
    hour = torch.randint(0, 24, (B,))
    qc_cnt = torch.randint(1, 16, (B,))

    out = model(matrix, hour, qc_cnt)
    print(f"输入: matrix {matrix.shape}, hour {hour.shape}, qc_cnt {qc_cnt.shape}")
    print(f"输出: {out.shape}")  # (B, 2)
    print(f"输出值: {out}")
