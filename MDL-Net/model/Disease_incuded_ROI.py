import torch
import torch.nn as nn


class Disease_Guide_ROI(nn.Module):
    def __init__(self, dim, heads, i=3, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.iter = i
        self.x1 = 0.
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cls = nn.Linear(dim, dim, bias=qkv_bias)
        mean = nn.Parameter(torch.randn(1, 1, dim))
        std = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
        self.weight = nn.Parameter(torch.normal(mean, std))
        # nn.init.uniform_(self.weight)
        self.gru = nn.GRU(90, 90)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, cls_out):
        # x (B, seq, dim)
        B, N, C = x.shape
        weight = self.weight.expand(B, -1, -1).contiguous()
        for _ in range(self.iter):
            weight_prev = weight
            weight1 = weight.reshape(B, self.num_heads, 1, C // self.num_heads)
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            q = self.cls(cls_out).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q * self.scale * weight1
            k = k * weight1
            attn = (q @ k.transpose(-2, -1))
            attn_s = self.softmax(attn)  # (B heads seq seq)
            attn = self.attn_drop(attn_s)
            x1 = attn @ (v * weight1)
            x1 = x1.transpose(1, 2).reshape(B, N, C)
            x1 = x1.transpose(0, 1)
            self.gru.flatten_parameters()
            weight, _ = self.gru(x1, weight_prev.transpose(0, 1))
            weight = weight.transpose(0, 1)

        x = self.proj(x1.transpose(0, 1))
        x = self.proj_drop(x)
        x = x.squeeze(dim=1)

        return x