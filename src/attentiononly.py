from __future__ import annotations
import torch
import torch.nn as nn
import transformer_dick_fixed_embed as tr

class AttentionOnlyBlock(nn.Module):
    """Multi-Head Self-Attention + 残差 + LayerNorm（MLPなし）"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim) で扱えるようにする
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        # 残差 + LayerNorm
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class AttentionOnlyNet(nn.Module):
    """
    Multi-Head Attention層のみが連続するネットワーク
    入力はすでに埋め込み済み（形状: (batch, seq_len, d_model)）を想定
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([AttentionOnlyBlock(d_model, n_heads, dropout)for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return x

class AttentionOnlyRegressor(AttentionOnlyNet):
    def __init__(self, d_model: int, n_heads: int, num_layers: int,  dropout: float = 0.1):
        super().__init__(d_model, n_heads, num_layers, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        cls = super().forward(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[:, 0, :]  # (B, d_model)
        yhat = self.head(cls)  # (B,1)
        return yhat

# 動作チェック用のサンプル
if __name__ == "__main__":
    batch_size = 4
    seq_len = 16
    d_model = 64
    n_heads = 8
    num_layers = 6

    net = AttentionOnlyNet(d_model=d_model, n_heads=n_heads, num_layers=num_layers)

    x = torch.randn(batch_size, seq_len, d_model)  # すでに埋め込み済みの入力
    out = net(x)
    print(out.shape)  # -> torch.Size([4, 16, 64])
