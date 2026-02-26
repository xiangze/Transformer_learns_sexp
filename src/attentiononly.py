from __future__ import annotations
import torch
import torch.nn as nn
import transformer_dick_fixed_embed as tr

class AttentionOnlyBlock(nn.Module):
    """Multi-Head Self-Attention + 残差 + LayerNorm（MLPなし）"""
    def __init__(self, params:dict,debug=False):
        super().__init__()
        d_model=params["d_model"]
        n_heads=params["nhead"]
        #dim_ff= params["dim_ff"]
        #max_len= params["max_len"]
        #pad_id=params["pad_id"]
        dropout=params["dropout"]

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

class SharedAttentionOnly(nn.Module):
    """
    RNN風Attention Only Layer
    """
    def __init__(self, params:dict,debug=False, weightvisible=False):#可視化したいときはTrue
        super().__init__()
        self.steps = params["num_layers"]# 反復回数（＝層数に相当）
        d_model=params["d_model"]
        dropout=params["dropout"]        
        n_heads=params["nhead"]
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim) で扱えるようにする
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        #self.pos = nn.Embedding(max_len, d_model)
        self.step_embed = None
        self.weightvisible=weightvisible

    def forward(
        self,
        x_tok: torch.Tensor,                 # (B, L, d_model) すでにトークン側で埋め込み済み
        key_padding_mask: torch.Tensor = None,  # (B, L) 1=keep/0=pad なら (==0) を渡す
        attn_mask: torch.Tensor = None,         # (L, L) など（必要な場合）
    ) -> torch.Tensor:
        B, L, D = x_tok.shape
        #pos_ids = torch.arange(L, device=x_tok.device).unsqueeze(0).expand(B, L)
        h = x_tok #+ self.pos(pos_ids)

        for t in range(self.steps):
            if self.step_embed is not None:
                step_ids = torch.full((B, L), t, dtype=torch.long, device=x_tok.device)
                h = h + self.step_embed(step_ids)
            attn_out ,_ = self.attn(
                h,h,h,
                key_padding_mask=(key_padding_mask == 0) if key_padding_mask is not None else None,
                attn_mask=attn_mask,
                need_weights=self.weightvisible)
            #可視化したいときはTrue
            h=attn_out
        return h #self.norm(h)+ self.dropout(attn_out)

class AttentionOnlyNet(nn.Module):
    """
    Multi-Head Attention層のみが連続するネットワーク
    入力はすでに埋め込み済み（形状: (batch, seq_len, d_model)）を想定
    """
    def __init__(self, params:dict,debug=False,recursive=False,weightvisible=False):
        super().__init__()
        num_layers=params["num_layers"]
        if(recursive):
            self.layers = nn.ModuleList([SharedAttentionOnly(params,weightvisible=weightvisible)])
        else:
            self.layers = nn.ModuleList([AttentionOnlyBlock(params)for _ in range(num_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #attn_mask=(attn_mask == 0) # True=padding
        for layer in self.layers:
            #print(x)
            x = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        return x

class AttentionOnlyRegressor(AttentionOnlyNet):
    def __init__(self, params:dict,debug=False,recursive=False,weightvisible=False):
        super().__init__(params,debug,recursive,weightvisible)
        d_model=params["d_model"]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        #key_padding_mask = (attn_mask == 0)if key_padding_mask is not None else None, # True=padding
        #key_padding_mask[:, 0] = 0 #False
        cls = super().forward(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[:, 0, :]  # (B, d_model)
        yhat = self.head(cls)  # (B,1)
        return yhat

class AttentionOnlyFRecursiveRegressor(AttentionOnlyRegressor):
    def __init__(self, params:dict,debug=False,weightvisible=False):
        super().__init__(params,debug,True,weightvisible)

# 動作チェック用のサンプル
if __name__ == "__main__":
    params={
        "batch_size" : 4,
        "seq_len" : 16,
        "d_model" : 64,
        "nhead" : 4,
        "num_layers" :3,
        "dropout":0.1,
    }
    print("Attention Only")
    net = AttentionOnlyNet(params)
    x = torch.randn(params["batch_size"], params["seq_len"], params["d_model"])  # すでに埋め込み済みの入力
    out = net(x)
    print(out)
    #print(out.shape)  # -> torch.Size([4, 16, 64])

    print("Attention Only regressor")
    net = AttentionOnlyRegressor(params)
    out = net(x)
    print(out)    
    print("Attention Only recursive")
    net = AttentionOnlyRegressor(params,recursive=True)
    out = net(x)
    print(out)    
