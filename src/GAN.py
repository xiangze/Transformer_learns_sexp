import math
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import randhof_with_weight as hof

# =============================
# 1. S式ペア生成（ダミー版）
# =============================

def generate_sexpr_pair(dummy=False) -> Tuple[str, str]:
    if(dummy):
        examples = [
            ("(+ 1 2)", "3"),
            ("(+ 1 2 3)", "6"),
            ("(* 2 (+ 1 2))", "6"),
            ("(if True 1 0)", "1"),
            ("(if False 1 0)", "0"),
            ("(len [1 2 3])", "3"),
            ("(first [1 2 3])", "1"),
            ("(rest [1 2 3])", "[2 3]"),
        ]
        return random.choice(examples)
    else:
        expr, kind = hof.random_typed_sexp(max_depth=5, want_kind="int")
        reduced, steps = hof.totaleval(hof.sexpr_to_str(expr))[]
        return hof.sexpr_to_str(expr), hof.sexpr_to_str(reduced)

# =============================
# 2. トークナイザ & vocab
# =============================
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<sep>": 3,
}

def tokenize_sexpr(s: str) -> List[str]:
    # 超シンプルなトークナイザ: 空白と括弧・角括弧を分離
    tokens = []
    current = ""
    for ch in s:
        if ch in "[]()":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
        elif ch.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens


class Vocab:
    def __init__(self):
        self.stoi: Dict[str, int] = dict(SPECIAL_TOKENS)
        self.itos: List[str] = [None] * len(SPECIAL_TOKENS)
        for k, v in SPECIAL_TOKENS.items():
            self.itos[v] = k

    def add_sentence(self, tokens: List[str]):
        for t in tokens:
            if t not in self.stoi:
                idx = len(self.stoi)
                self.stoi[t] = idx
                self.itos.append(t)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi[t] for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    @property
    def pad_idx(self):
        return self.stoi["<pad>"]

    @property
    def bos_idx(self):
        return self.stoi["<bos>"]

    @property
    def eos_idx(self):
        return self.stoi["<eos>"]

    @property
    def sep_idx(self):
        return self.stoi["<sep>"]

    @property
    def size(self):
        return len(self.stoi)


# =============================
# 3. Dataset
# =============================

class SExprDataset(Dataset):
    def __init__(self, n_samples: int, max_len: int = 64):
        super().__init__()
        self.pairs: List[Tuple[List[str], List[str]]] = []
        self.vocab = Vocab()
        self.max_len = max_len

        for _ in range(n_samples):
            before, after = generate_sexpr_pair()
            src_tokens = tokenize_sexpr(before)
            tgt_tokens = tokenize_sexpr(after)
            self.pairs.append((src_tokens, tgt_tokens))
            self.vocab.add_sentence(src_tokens)
            self.vocab.add_sentence(tgt_tokens)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, vocab: Vocab, max_len: int = 64):
    src_batch, tgt_batch = zip(*batch)

    def pad_and_encode(tokens_list, add_bos=False, add_eos=True):
        res = []
        for toks in tokens_list:
            seq = []
            if add_bos:
                seq.append(vocab.bos_idx)
            seq.extend(vocab.encode(toks))
            if add_eos:
                seq.append(vocab.eos_idx)
            seq = seq[:max_len]
            pad_len = max_len - len(seq)
            seq += [vocab.pad_idx] * pad_len
            res.append(seq)
        return torch.tensor(res, dtype=torch.long)

    src = pad_and_encode(src_batch, add_bos=True, add_eos=True)
    tgt = pad_and_encode(tgt_batch, add_bos=True, add_eos=True)

    return src, tgt


# =============================
# 4. Transformer Generator
# =============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TransformerGenerator(nn.Module):
    """
    seq2seq Transformer:
      入力: before S式
      出力: after S式
    """

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        """
        src: (B, Ls)
        tgt: (B, Lt)  (teacher forcing)
        """
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (tgt == 0)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        # causal mask for decoder
        Lt = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(Lt).to(tgt.device)

        h = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.out_proj(h)  # (B, Lt, V)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len: int, bos_idx: int, eos_idx: int):
        """
        Greedy decode
        src: (B, Ls)
        return: (B, L_generated)
        """
        self.eval()
        B = src.size(0)
        src_key_padding_mask = (src == 0)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

        ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=src.device)

        for _ in range(max_len - 1):
            tgt_emb = self.pos_enc(self.tgt_embed(ys) * math.sqrt(self.d_model))
            Lt = ys.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(Lt).to(src.device)

            h = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=(ys == 0),
                memory_key_padding_mask=src_key_padding_mask,
            )
            logits = self.out_proj(h[:, -1, :])  # (B, V)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B,1)
            ys = torch.cat([ys, next_token], dim=1)
            if torch.all(next_token == eos_idx):
                break
        return ys


# =============================
# 5. Transformer Discriminator
# =============================

class TransformerDiscriminator(nn.Module):
    """
    入力: [before] <sep> [after] を1列にしたシーケンス
    出力: 本物ペアかどうかのスコア (B,1)
    """

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 512, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, seq):
        """
        seq: (B, L)
        """
        mask = (seq == 0)  # pad mask
        x = self.pos_enc(self.embed(seq) * math.sqrt(self.d_model))
        h = self.encoder(x, src_key_padding_mask=mask)  # (B, L, D)
        # [CLS] 的に最初トークンの表現を使う
        cls = h[:, 0, :]
        logit = self.out(cls)  # (B,1)
        return logit


# =============================
# 6. GAN 風学習ループ
# =============================

def build_real_fake_pairs(src, tgt_real, gen_ids, vocab: Vocab, max_len: int):
    """
    Discriminator に渡す [before] <sep> [after] シーケンスを作る。
    src:      (B, Ls)
    tgt_real: (B, Lt)
    gen_ids:  (B, Lg)  (Generatorが生成した系列)
    """
    B, Ls = src.size()
    _, Lt = tgt_real.size()
    _, Lg = gen_ids.size()

    def cat_and_pad(after_seq):
        B, L_after = after_seq.size()
        sep_col = torch.full((B, 1), vocab.sep_idx, dtype=torch.long, device=src.device)
        concat = torch.cat([src, sep_col, after_seq], dim=1)  # (B, Ls+1+L_after)
        L = concat.size(1)
        if L > max_len:
            concat = concat[:, :max_len]
        else:
            pad_len = max_len - L
            pad = torch.full((B, pad_len), vocab.pad_idx, dtype=torch.long, device=src.device)
            concat = torch.cat([concat, pad], dim=1)
        return concat

    real_pair = cat_and_pad(tgt_real)
    fake_pair = cat_and_pad(gen_ids)
    return real_pair, fake_pair


def train_gan_like(
    n_samples: int = 1000,
    batch_size: int = 32,
    max_len: int = 64,
    n_epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    dataset = SExprDataset(n_samples=n_samples, max_len=max_len)
    vocab = dataset.vocab
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab, max_len),
    )

    gen = TransformerGenerator(vocab_size=vocab.size, max_len=max_len).to(device)
    disc = TransformerDiscriminator(vocab_size=vocab.size, max_len=max_len).to(device)

    opt_G = torch.optim.Adam(gen.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(disc.parameters(), lr=1e-4)

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    for epoch in range(n_epochs):
        gen.train()
        disc.train()
        total_G_loss = 0.0
        total_D_loss = 0.0
        total_steps = 0

        for src, tgt in loader:
            src = src.to(device)  # (B, Ls)
            tgt = tgt.to(device)  # (B, Lt)

            # ============================
            # 1) Discriminator step
            # ============================
            opt_D.zero_grad()

            # real pairs
            with torch.no_grad():
                tgt_real = tgt  # ここでは教師データ通り
                gen_ids = gen.generate(
                    src,
                    max_len=tgt.size(1),
                    bos_idx=vocab.bos_idx,
                    eos_idx=vocab.eos_idx,
                )

            real_pair, fake_pair = build_real_fake_pairs(src, tgt_real, gen_ids, vocab, max_len)

            real_logits = disc(real_pair)
            fake_logits = disc(fake_pair)

            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)

            D_loss_real = bce(real_logits, real_labels)
            D_loss_fake = bce(fake_logits, fake_labels)
            D_loss = (D_loss_real + D_loss_fake) * 0.5
            D_loss.backward()
            opt_D.step()

            # ============================
            # 2) Generator step
            # ============================
            opt_G.zero_grad()

            # teacher forcing 用の入力・出力
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = gen(src, tgt_in)  # (B, Lt-1, V)
            G_ce_loss = ce(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )

            # GAN loss: Discriminator を騙す方向
            gen_ids = gen.generate(
                src,
                max_len=tgt.size(1),
                bos_idx=vocab.bos_idx,
                eos_idx=vocab.eos_idx,
            )
            _, fake_pair = build_real_fake_pairs(
                src, tgt_real, gen_ids, vocab, max_len
            )
            fake_logits_for_G = disc(fake_pair)
            # Generator は「本物」と判定されたい
            G_gan_loss = bce(fake_logits_for_G, torch.ones_like(fake_logits_for_G))

            G_loss = G_ce_loss + 0.1 * G_gan_loss
            G_loss.backward()
            opt_G.step()

            total_G_loss += G_loss.item()
            total_D_loss += D_loss.item()
            total_steps += 1

        print(
            f"Epoch {epoch+1}/{n_epochs} "
            f"G_loss={total_G_loss/total_steps:.4f} "
            f"D_loss={total_D_loss/total_steps:.4f}"
        )

    return gen, disc, vocab


if __name__ == "__main__":
    gen, disc, vocab = train_gan_like(
        n_samples=2000,
        batch_size=32,
        max_len=64,
        n_epochs=5,
    )

    # 簡単なテスト
    test_src = "(+ 1 2 3)"
    tokens = tokenize_sexpr(test_src)
    ids = [vocab.bos_idx] + vocab.encode(tokens) + [vocab.eos_idx]
    if len(ids) < 64:
        ids += [vocab.pad_idx] * (64 - len(ids))
    src_tensor = torch.tensor([ids], dtype=torch.long)
    src_tensor = src_tensor.to(next(gen.parameters()).device)

    with torch.no_grad():
        gen_ids = gen.generate(src_tensor, max_len=32,
                               bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx)
    print("input :", test_src)
    print("output:", " ".join(vocab.decode(gen_ids[0].tolist())))
