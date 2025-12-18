# shared_transformer.py
from __future__ import annotations
import torch
import torch.nn as nn

class SharedTransformerEncoder(nn.Module):
    """
    1つの TransformerEncoderLayer を steps 回“再帰的”に適用する重み共有エンコーダ。
    """
    def __init__(
        self,
        params:dict,
        debug=False  ):

        super().__init__()
        vocab_size=params["vocab_size"]
        d_model=params["d_model"]
        nhead=params["nhead"]
        steps = params["num_layer"]# 反復回数（＝層数に相当）
        use_step_embed= params["use_step_embed"]    # Universal Transformer 的な step embedding
        norm_first = params["norm_first"]        # Pre-LN の方が安定しやすい

        dim_ff= params["dim_ff"]
        max_len= params["max_len"]
        pad_id=params["pad_id"]
        dropout=params["dropout"]

        self.steps = steps
        self.use_step_embed = use_step_embed

        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )
        self.pos = nn.Embedding(max_len, d_model)
        self.step_embed = nn.Embedding(steps, d_model) if use_step_embed else None
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x_tok: torch.Tensor,                 # (B, L, d_model) すでにトークン側で埋め込み済み
        key_padding_mask: torch.Tensor = None,  # (B, L) 1=keep/0=pad なら (==0) を渡す
        attn_mask: torch.Tensor = None,         # (L, L) など（必要な場合）
    ) -> torch.Tensor:
        B, L, D = x_tok.shape
        pos_ids = torch.arange(L, device=x_tok.device).unsqueeze(0).expand(B, L)
        h = x_tok + self.pos(pos_ids)

        for t in range(self.steps):
            if self.step_embed is not None:
                step_ids = torch.full((B, L), t, dtype=torch.long, device=x_tok.device)
                h = h + self.step_embed(step_ids)
            h = self.block(
                h,
                src_key_padding_mask=(key_padding_mask == 0) if key_padding_mask is not None else None,
                attn_mask=attn_mask,
            )
        return self.final_norm(h)


class SharedTransformerRegressor(nn.Module):
    """
    入力埋め込み（学習 or 固定）→ 共有トランスフォーマ → <CLS> 回帰
    """
    def __init__(
        self,
        *,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        steps: int = 6,
        max_len: int = 4096,
        embed_mode: str = "fixed",   # "fixed" = 事前関数でエンコード済み, "learned" = nn.Embedding 使用
        vocab_size: int | None = None,
    ):
        super().__init__()
        assert embed_mode in ("fixed", "learned")
        self.embed_mode = embed_mode

        if embed_mode == "learned":
            assert vocab_size is not None
            self.tok = nn.Embedding(vocab_size, d_model)
        else:
            self.tok = None  # 使わない（外で固定ベクトル化して渡す）

        self.num_proj = nn.Sequential(      # 数値（L:NUM）のスカラーを d_model に持ち上げる
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

        self.encoder = SharedTransformerEncoder(
            d_model=d_model, nhead=nhead, dim_ff=dim_ff,
            dropout=dropout, steps=steps, max_len=max_len, use_step_embed=True, norm_first=True
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,      # (B, L)  学習埋め込みで使う
        input_feats: torch.Tensor | None = None,    # (B, L, d_model) 固定写像で使う
        num_vals: torch.Tensor,                     # (B, L)  数値スカラー（標準化済み）
        num_mask: torch.Tensor,                     # (B, L)  1=数値位置, 0=非数値
        attn_mask: torch.Tensor,                    # (B, L)  1=トークン, 0=PAD
    ):
        if self.embed_mode == "learned":
            x = self.tok(input_ids)    # (B, L, d_model)
        else:
            x = input_feats            # (B, L, d_model) 既に固定写像で作られている前提

        # 数値埋め込みを加算（NUM 位置のみ）
        num_emb = self.num_proj(num_vals.unsqueeze(-1)) * num_mask.unsqueeze(-1).float()
        x = x + num_emb

        h = self.encoder(x_tok=x, key_padding_mask=attn_mask)
        cls = h[:, 0, :]                      # 先頭 <CLS>
        return self.head(cls)                 # (B, 1)
    
