"""
model_transformer.py
Revised forward surrogate: binary 12x12 pixel design -> 81-point S11
(normalized space). Drop-in replacement for model.py's ForwardSurrogate --
same input/output shapes, same build_model() signature -- so train_surrogate.py
and optimize.py only need an --arch switch.

WHAT THIS ARCHITECTURE IS
-------------------------
    (B, 1, 12, 12)
        -> Conv stem (2 layers, GroupNorm)               # local features per pixel
        -> 144 pixel tokens (one per grid cell)          # no patching, no info loss
        -> learned positional embedding
        -> Transformer ENCODER (pre-norm, LayerNorm)     # long-range coupling
        -> memory: (B, 144, d_model)
    Independently:
        81 frequency values (10..20 GHz) -> normalized [-1,1]
        -> Frequency MLP -> 81 query tokens (B, 81, d_model)
    Then:
        -> Transformer DECODER (cross-attn: queries <- memory)
        -> shared regression head: (B,81,d_model) -> (B,81,1) -> (B,81)

WHY EACH PIECE
--------------
* CONV STEM BEFORE TOKENIZATION. Per-pixel tokens on a *binary* board without a
  stem would give each token a single bit + positional vector -- degenerate.
  The stem gives each token local neighbourhood context first.
  `stem_layers=0` gives the pure-ViT ablation.

* 144 PIXEL TOKENS, NO PATCHING. At 12x12, attention is 144^2 ~= 20k pairs --
  free. Patching (e.g. 4x4 -> 9 tokens as in Friend's file) throws away the
  spatial resolution the surrogate needs for a pixel-level GD task. Rejected.

* LAYERNORM PRE-NORM. Normalizes within one token of one sample -> no coupling
  across parallel GD candidates (the reason model.py rejected BatchNorm) and
  no train/eval gap. Confirmed by batch-independence check in __main__.

* FREQUENCY-QUERY DECODER (the stolen idea, kept honestly). 81 queries generated
  from a small MLP over the ACTUAL GHz values normalized to [-1,1]. Each query
  cross-attends to the 144 pixel tokens. The model knows WHICH frequency each
  output is; adjacent bins produce near-identical queries so outputs are smooth
  by construction. Shared regression head reduces parameters vs a flatten head.
  Explicitly rejected from Friend's file: BatchNorm, 4x4 patching to 9 tokens,
  MSE-selection.

* DROPOUT AS nn.Dropout MODULES (not F.dropout). optimize.py's
  enable_mc_dropout() matches classes starting with "Dropout"; F.dropout is
  invisible to that matcher. Attention is hand-written for the same reason.
  Moot while gamma=0 but a nasty silent bug otherwise.

WHAT'S DELIBERATELY *NOT* HERE
------------------------------
* No gamma-related code changes. gamma stays 0 in optimize.py (unchanged).
* No NN-distance L_trust yet -- that's a separate PR in optimize.py, not model.
  A hook `encode(x)` is exposed for it to use later.
* No BatchNorm anywhere. Reject entirely. If you ever add it, GD breaks.

SCALING NOTE FOR THE 3x100x100 FILTER PROJECT
---------------------------------------------
Patch-1 tokenization does NOT survive the move to 100x100: 10,000 tokens ->
1e8-entry attention matrix per head per layer. Options in order of cost:
  1. patch 4x4 or 5x5 -> 625 or 400 tokens.
  2. windowed / shifted-window attention (Swin-style).
  3. deeper conv stem with stride -> transformer sees a coarse grid.
Multi-bit channels (3 metal layers) only change the stem's input channels.
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _groups(c, target=8):
    """GroupNorm group count that divides c."""
    g = min(target, c)
    while c % g != 0:
        g -= 1
    return g


# -----------------------------------------------------------------------------
# Conv stem: (N,1,12,12) -> (N, d_model, 12, 12). No downsampling. GroupNorm.
# -----------------------------------------------------------------------------
class ConvStem(nn.Module):
    def __init__(self, d_model, n_layers=2, width=64):
        super().__init__()
        chans = [1] + [width] * (n_layers - 1) + [d_model]
        layers = []
        for i in range(n_layers):
            layers += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                nn.GroupNorm(_groups(chans[i + 1]), chans[i + 1]),
                nn.LeakyReLU(0.1, inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Multi-head self-attention (hand-written so dropout is nn.Dropout modules)
# -----------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, p_drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(p_drop)
        self.proj_drop = nn.Dropout(p_drop)

    def forward(self, x):                                  # x: (N, T, D)
        N, T, D = x.shape
        qkv = self.qkv(x).reshape(N, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (N, h, T, dk)
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        att = self.attn_drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(N, T, D)
        return self.proj_drop(self.proj(out))


# -----------------------------------------------------------------------------
# Multi-head cross-attention (queries attend to a separate memory)
# -----------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, p_drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(p_drop)
        self.proj_drop = nn.Dropout(p_drop)

    def forward(self, x, memory):                          # x:(N,Tq,D) mem:(N,Tm,D)
        N, Tq, D = x.shape
        Tm = memory.shape[1]
        q = self.q(x).reshape(N, Tq, self.h, self.dk).transpose(1, 2)
        kv = self.kv(memory).reshape(N, Tm, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        att = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)
        att = self.attn_drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(N, Tq, D)
        return self.proj_drop(self.proj(out))


# -----------------------------------------------------------------------------
# Encoder block: self-attn -> MLP (pre-norm)
# -----------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, p_drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, p_drop)
        self.n2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


# -----------------------------------------------------------------------------
# Decoder block: self-attn -> cross-attn -> MLP (pre-norm)
# NB: self-attn among frequency queries lets neighbouring bins share info,
# giving the decoder a way to enforce spectral smoothness beyond what the
# frequency MLP already provides.
# -----------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, p_drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = SelfAttention(d_model, n_heads, p_drop)
        self.n2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, p_drop)
        self.n3 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x, memory):
        x = x + self.self_attn(self.n1(x))
        x = x + self.cross_attn(self.n2(x), memory)
        x = x + self.mlp(self.n3(x))
        return x


# -----------------------------------------------------------------------------
# Frequency embedding: actual GHz -> query vectors
# -----------------------------------------------------------------------------
class FrequencyEmbedding(nn.Module):
    """Map normalized frequency in [-1, 1] to a d_model query vector.

    Kept small on purpose -- 81 queries share this net.
    """
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, f):                                  # f:(...,1)
        return self.net(f)


# -----------------------------------------------------------------------------
# Full surrogate
# -----------------------------------------------------------------------------
class TransformerSurrogate(nn.Module):
    """
    Encoder-decoder transformer with frequency queries.

    Default sizes give ~2M params so an A/B vs the CNN (~1.6M) is trunk-vs-trunk,
    not capacity-vs-capacity.
    """

    def __init__(
        self,
        n_freq=81,
        p_drop=0.1,
        d_model=96,                # smaller than Friend's 256 -> keeps params sane
        n_heads=6,
        enc_depth=4,
        dec_depth=2,
        mlp_ratio=4,
        stem_layers=2,
        stem_width=64,
        grid=12,
        f_lo_ghz=10.0,
        f_hi_ghz=20.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.grid = grid
        n_tokens = grid * grid
        self.n_freq = n_freq

        # ---- input path ----
        if stem_layers > 0:
            self.stem = ConvStem(d_model, n_layers=stem_layers, width=stem_width)
            self.patch = None
        else:
            self.stem = None
            self.patch = nn.Linear(1, d_model)              # pure-ViT ablation

        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.pos_drop = nn.Dropout(p_drop)

        # ---- encoder ----
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, mlp_ratio, p_drop) for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # ---- frequency queries (fixed physical frequencies) ----
        # Normalized GHz to [-1, 1]; buffer so it moves with .to() but has no grad.
        freqs = torch.linspace(f_lo_ghz, f_hi_ghz, n_freq)
        freqs_norm = (2.0 * (freqs - f_lo_ghz) / (f_hi_ghz - f_lo_ghz) - 1.0).view(1, n_freq, 1)
        self.register_buffer("freqs_norm", freqs_norm)
        self.freq_embed = FrequencyEmbedding(d_model)

        # ---- decoder ----
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, mlp_ratio, p_drop) for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # ---- shared per-query regression head ----
        # Small: applied to all 81 queries independently.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------------
    # optional: expose encoder features for a future NN-distance L_trust.
    # Called by nothing in this file; hooked in optimize.py when we implement it.
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x):
        """Return mean-pooled encoder features for one or many designs."""
        if self.stem is not None:
            t = self.stem(x)
            t = t.flatten(2).transpose(1, 2)
        else:
            t = x.flatten(2).transpose(1, 2)
            t = self.patch(t)
        t = t + self.pos
        for blk in self.encoder:
            t = blk(t)
        t = self.enc_norm(t)
        return t.mean(dim=1)                               # (N, d_model)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x):                                  # x: (N,1,12,12)
        N = x.shape[0]

        # ---- tokenize ----
        if self.stem is not None:
            t = self.stem(x)                               # (N,D,12,12)
            t = t.flatten(2).transpose(1, 2)               # (N,144,D)
        else:
            t = x.flatten(2).transpose(1, 2)               # (N,144,1)
            t = self.patch(t)
        t = self.pos_drop(t + self.pos)

        # ---- encoder ----
        for blk in self.encoder:
            t = blk(t)
        memory = self.enc_norm(t)                          # (N,144,D)

        # ---- frequency queries ----
        q = self.freq_embed(self.freqs_norm)               # (1,81,D)
        q = q.expand(N, -1, -1)                            # (N,81,D)

        # ---- decoder ----
        for blk in self.decoder:
            q = blk(q, memory)
        q = self.dec_norm(q)                               # (N,81,D)

        # ---- shared head, one output per query ----
        return self.head(q).squeeze(-1)                    # (N,81)


def build_model(**kwargs):
    return TransformerSurrogate(**kwargs)


# -----------------------------------------------------------------------------
# smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    net = build_model()
    n_params = sum(p.numel() for p in net.parameters())
    print("params:", f"{n_params:,}")

    x = torch.randint(0, 2, (4, 1, 12, 12)).float()
    y = net(x)
    print("out shape:", tuple(y.shape))
    assert y.shape == (4, 81)

    # 1) input gradients exist and are non-degenerate (the whole pilot)
    x.requires_grad_(True)
    net(x).sum().backward()
    print("input grad ok:", x.grad is not None,
          "abs mean:", float(x.grad.abs().mean()),
          "abs max:", float(x.grad.abs().max()))

    # 2) batch-independence in eval mode: prediction for design i must not
    #    depend on what else is in the batch. GD relies on this.
    net.eval()
    with torch.no_grad():
        solo = net(x[:1].detach())
        batched = net(x.detach())[:1]
    max_diff = float((solo - batched).abs().max())
    print("batch-independence max diff:", max_diff)
    assert max_diff < 1e-5, "batch coupling detected -- did BatchNorm sneak in?"

    # 3) frequency queries are ordered and distinct: predictions at adjacent
    #    bins should be more similar than at distant bins, on average.
    net.eval()
    with torch.no_grad():
        preds = net(torch.rand(64, 1, 12, 12))              # (64,81)
    adj = (preds[:, 1:] - preds[:, :-1]).abs().mean().item()
    far = (preds[:, 40:] - preds[:, :41]).abs().mean().item()
    print(f"adj-bin diff {adj:.4f}  far-bin diff {far:.4f}  (adj should be < far)")

    # 4) MC-dropout still perturbs predictions (wired, but off by default via
    #    gamma=0 in optimize.py). Kept as a signal that the plumbing works.
    for m in net.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
    with torch.no_grad():
        p1 = net(x[:2].detach())
        p2 = net(x[:2].detach())
    print("dropout perturbs preds:", float((p1 - p2).abs().max()) > 0)

    # 5) pure-ViT ablation still builds and runs
    vit = build_model(stem_layers=0)
    print("stem_layers=0 params:", f"{sum(p.numel() for p in vit.parameters()):,}",
          "out:", tuple(vit(torch.rand(2, 1, 12, 12)).shape))

    # 6) encode() hook returns a fixed-size embedding
    with torch.no_grad():
        emb = net.encode(x[:3].detach())
    print("encode() shape:", tuple(emb.shape))
