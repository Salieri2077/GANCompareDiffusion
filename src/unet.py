import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int):
    """
    timesteps: (B,) int64
    return: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb):
        return self.net(t_emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_ch, out_ch)

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        # (b, heads, head_dim, hw)
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        # attn: (b, heads, hw, hw)
        attn = torch.einsum("bhdm,bhdn->bhmn", q, k) / math.sqrt(head_dim)
        attn = attn.softmax(dim=-1)

        # out: (b, heads, head_dim, hw)
        out = torch.einsum("bhmn,bhdn->bhdm", attn, v)
        out = out.reshape(b, c, h, w)

        out = self.proj(out)
        return x_in + out


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        base_ch=128,
        ch_mult=(1, 2, 2, 2),
        attn_resolutions=(16, 8),
        time_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = TimeMLP(time_dim, time_dim)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down
        self.downs = nn.ModuleList()
        chs = []                      # 只记录每个 ResBlock 的输出通道
        curr_ch = base_ch
        resolution = 32
        
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
        
            self.downs.append(ResBlock(curr_ch, out_ch, time_dim, dropout))
            curr_ch = out_ch
            chs.append(curr_ch)       # 仅 ResBlock 后记录 skip 通道
        
            if resolution in attn_resolutions:
                self.downs.append(AttentionBlock(curr_ch))
        
            if i != len(ch_mult) - 1:
                self.downs.append(Downsample(curr_ch))
                resolution //= 2


        # Mid
        self.mid1 = ResBlock(curr_ch, curr_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(curr_ch)
        self.mid2 = ResBlock(curr_ch, curr_ch, time_dim, dropout)

        # Up
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            self.ups.append(ResBlock(curr_ch + chs.pop(), out_ch, time_dim, dropout))
            curr_ch = out_ch

            if resolution in attn_resolutions:
                self.ups.append(AttentionBlock(curr_ch))

            if i != 0:
                self.ups.append(Upsample(curr_ch))
                resolution *= 2

        self.out_norm = nn.GroupNorm(32, curr_ch)
        self.out_conv = nn.Conv2d(curr_ch, 3, 3, padding=1)

    def forward(self, x, t):
        # t: (B,) int64
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        hs = []
        h = self.in_conv(x)

        for m in self.downs:
            if isinstance(m, ResBlock):
                h = m(h, t_emb)
                hs.append(h)   # 只保存 ResBlock 输出作为 skip
            else:
                h = m(h)       # Attention/Downsample 不入栈


        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        for m in self.ups:
            if isinstance(m, ResBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = m(h, t_emb)
            else:
                h = m(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
