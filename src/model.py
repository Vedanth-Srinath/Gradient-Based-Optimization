"""
model.py
Forward surrogate: binary 12x12 pixel design -> 81-point S11 (normalized space).

Design choices, and why:
  * 16 conv layers, NO pooling. 12x12 is tiny; pooling throws away resolution we
    can't spare. With 3x3 convs the receptive field after 16 layers already
    exceeds 12x12, so every output sees the whole board.
  * GroupNorm, NOT BatchNorm. During gradient-based inverse design we run many
    parallel candidate inputs through the frozen net; BatchNorm couples them via
    batch statistics and behaves differently in train vs eval. GroupNorm has no
    such coupling and no train/eval divergence -> stable gradients w.r.t. input.
  * Head is a 1x1 conv bottleneck + a SMALL flatten, NOT global average pooling.
    (Correcting earlier advice: GAP is wrong here -- averaging over space discards
     *where* the metal is, which is exactly what determines S11. We instead cut
     channels with a 1x1 conv, then flatten a small tensor. Keeps spatial info,
     ~200k head params instead of the repo's 37M dense head -> better-conditioned
     input gradients.)
  * Dropout in several places so MC-dropout can later estimate surrogate
    uncertainty (the L_trust term).
  * Standard layers only -> works under torch.autocast (AMP) on the RTX 2070.
"""

import torch
import torch.nn as nn


def _groups(c, target=8):
    """Pick a GroupNorm group count that divides c."""
    g = min(target, c)
    while c % g != 0:
        g -= 1
    return g


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, p_drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class ForwardSurrogate(nn.Module):
    def __init__(self, n_freq=81, p_drop=0.1, bottleneck=16):
        super().__init__()
        # channel schedule over 16 conv layers: 32 -> 64 -> 128
        chans = [1] + [32] * 5 + [64] * 5 + [128] * 6  # 16 blocks
        # dropout on a few interior blocks (indices chosen to spread it out)
        drop_at = {4, 9, 14}
        blocks = []
        for i in range(16):
            p = p_drop if i in drop_at else 0.0
            blocks.append(ConvBlock(chans[i], chans[i + 1], p_drop=p))
        self.features = nn.Sequential(*blocks)

        # 1x1 bottleneck keeps spatial layout, cuts channels before flatten
        self.bottleneck = nn.Sequential(
            nn.Conv2d(chans[-1], bottleneck, kernel_size=1, bias=False),
            nn.GroupNorm(_groups(bottleneck), bottleneck),
            nn.LeakyReLU(0.1, inplace=True),
        )
        flat = bottleneck * 12 * 12
        self.head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_freq),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.bottleneck(x)
        x = torch.flatten(x, 1)
        return self.head(x)          # normalized S11, shape (N, 81)


def build_model(**kwargs):
    return ForwardSurrogate(**kwargs)


if __name__ == "__main__":
    net = build_model()
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randint(0, 2, (4, 1, 12, 12)).float()
    y = net(x)
    print("params:", f"{n_params:,}")
    print("out shape:", tuple(y.shape))
    # gradient-to-input works (the whole point of the pilot)
    x.requires_grad_(True)
    net(x).sum().backward()
    print("input grad ok:", x.grad is not None, "grad abs mean:",
          float(x.grad.abs().mean()))
