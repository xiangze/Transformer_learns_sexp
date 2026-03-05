from __future__ import annotations
import torch
import torch.nn as nn
import transformer_dick_fixed_embed as tr

class AttentionOnlyBlock(nn.Module):
    """Multi-Head Self-Attention + 残差 + LayerNorm（MLPなし）"""
    def __init__(self, params:dict,debug=False, weightvisible=False):
        super().__init__()
        d_model=params["d_model"]
        n_heads=params["nhead"]
        #max_len= params["max_len"]
        #pad_id=params["pad_id"]
        dropout=params["dropout"]

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (batch, seq, dim) で扱えるようにする
        )
        self.weightvisible=weightvisible
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
            need_weights=self.weightvisible,
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
        self.steps = params["num_layer"]# 反復回数（＝層数に相当）
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
        self.step_embed = None
        self.weightvisible=weightvisible
        self.debug=debug

    def forward(self,
        x_tok: torch.Tensor,                 # (B, L, d_model) すでにトークン側で埋め込み済み
        attn_mask: torch.Tensor = None,         # (L, L) など（必要な場合）
        key_padding_mask: torch.Tensor = None,  # (B, L) 1=keep/0=pad なら (==0) を渡す
    ) -> torch.Tensor:
        B, L,d_model  = x_tok.shape
        pos_ids = torch.arange(L, device=x_tok.device, dtype=x_tok.dtype).unsqueeze(0).expand(B, L)
        h = x_tok + pos_ids.unsqueeze(-1)  # broadcast over d_model

        if(self.debug):
            print("key_padding_mask",key_padding_mask)
            print("attn_mask",attn_mask)
        for t in range(self.steps):
            if self.step_embed is not None:
                step_ids = torch.full((B, L), t, dtype=torch.long, device=x_tok.device)
                h = h + self.step_embed(step_ids)
            attn_out ,_ = self.attn(
                h,h,h,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                need_weights=self.weightvisible)   #可視化したいときはTrue
            h=attn_out
        return h #self.norm(h)+ self.dropout(attn_out)

class AttentionOnlyNet(nn.Module):
    """
    Multi-Head Attention層のみが連続するネットワーク
    入力はすでに埋め込み済みids（形状: (batch, seq_len, d_model)）を想定
    """
    def __init__(self, params:dict,debug=False,recursive=False,weightvisible=False):
        super().__init__()
        num_layers=params["num_layer"]
        self.pad_id=params["pad_id"]
        self.debug=debug
        if(recursive):
            self.layers = nn.ModuleList([SharedAttentionOnly(params,weightvisible=weightvisible)])
        else:
            self.layers = nn.ModuleList([AttentionOnlyBlock(params)for _ in range(num_layers)])

    def forward(self,
        ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        src_ids: torch.Tensor | None = None,   # ← token IDを別引数で受け取る
    ) -> torch.Tensor:
        ids=ids.to(torch.float32)
        if key_padding_mask is None:
            if src_ids is not None:
                key_padding_mask = (src_ids == self.pad_id)  # 2D: (batch, seq_len)
                #key_padding_mask = (ids == self.pad_id)
            elif attn_mask is not None:
                key_padding_mask = (attn_mask == 0) # True=padding
            else:
                raise ValueError("src_ids か attn_mask のどちらかが必要です")
        
        key_padding_mask[:, 0] = False

        if(self.debug):
            print("ids",ids)
            print("attn_mask",attn_mask)
            print("key_padding_mask",key_padding_mask,key_padding_mask.dtype)
            
        valid = (~key_padding_mask).sum(dim=1)
        assert torch.all(valid > 0), "Some sequences fully masked!"

        for layer in self.layers:
            ids = layer(
                ids,
                attn_mask,          # (L,L) causal mask
                key_padding_mask=key_padding_mask,
            )
        return ids

class AttentionOnlyRegressor(AttentionOnlyNet):
    def __init__(self, params:dict,debug=False,recursive=False,weightvisible=False):
        super().__init__(params,debug,recursive,weightvisible)
        d_model=params["d_model"]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        cls = super().forward(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[:, 0, :]  # (B, d_model)
        yhat = self.head(cls)  # (B,1)
        return yhat

class AttentionOnlyRecursiveRegressor(AttentionOnlyRegressor):
    def __init__(self, params:dict,debug=False,weightvisible=False):
        super().__init__(params,debug,True,weightvisible)

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser(description="masked attention, recursive")
    parser.add_argument("--debug", action="store_true")    
    parser.add_argument("--key_padding_none", action="store_true")    
    args = parser.parse_args()

    params={
        "batch_size" : 4,
        "seq_len" : 16,
        "d_model" : 64,
        "nhead" : 4,
        "num_layer" :3,
        "dropout":0.1,
        "pad_id":0,
    }
    params["debug"]=args.debug

    #no emmbedding
    ids=torch.randint(0,10,(params["batch_size"], params["seq_len"], params["d_model"]) ) # すでに埋め込み済みの入力　内部でfloatにする
    mask= torch.tensor([ [1]*(params["seq_len"]-5) +[0]*5]* params["seq_len"],dtype=torch.bool)
    if(args.key_padding_none):
        key_padding_mask=None
        src_ids=torch.tensor([ [0]*(params["seq_len"]-5) +[1]*5]* params["batch_size"],dtype=torch.bool)
    else:
        key_padding_mask=torch.tensor([ [0]*(params["seq_len"]-5) +[1]*5]* params["batch_size"],dtype=torch.bool)
        src_ids=None
    d={"Attention Only":AttentionOnlyNet,"Attention Only regressor":AttentionOnlyRegressor,"Attention Only Recursive Regressor":AttentionOnlyRecursiveRegressor}
    for k,v in d.items():
        print(k)
        net = v(params,debug=params["debug"])        
        out = net(ids,mask, key_padding_mask,src_ids)
        print(out)

