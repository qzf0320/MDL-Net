import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch import einsum

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, hidden_size, img_size, patch_size, types=0):
        super().__init__()
        img_size = img_size
        patch_size = patch_size

        if types == 0:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]) * (
                    img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), hidden_size))
        elif types == 1:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2]), hidden_size))
        elif types == 2:
            self.position_embeddings = nn.Parameter(torch.zeros(1, (img_size[0] // patch_size[0]), hidden_size))

    def forward(self, x):
        return x + self.position_embeddings


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            min_indice = self.relative_indices.min()
            self.relative_indices += (-min_indice)
            max_indice = self.relative_indices.max().item()
            self.pos_embedding = nn.Parameter(torch.randn(max_indice + 1, max_indice + 1, max_indice + 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):

        b, n_h, n_w, n_d, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size[0]
        nw_w = n_w // self.window_size[1]
        nw_d = n_d // self.window_size[2]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d) -> b h (nw_h nw_w nw_d) (w_h w_w w_d) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d=self.window_size[2]), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1], self.relative_indices[:, :, 2]]
        else:
            dots += self.pos_embedding

        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w nw_d) (w_h w_w w_d) d -> b (nw_h w_h) (nw_w w_w) (nw_d w_d) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], w_d=self.window_size[2], nw_h=nw_h,
                        nw_w=nw_w, nw_d=nw_d)
        out = self.to_out(out)

        return out


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if attention:
            return attention_output, attention_probs
        return attention_output


class SelfAdaptiveAttention(nn.Module):
    def __init__(self, hidden_size, img_size, patch_size, num_heads, attention_dropout_rate, window_size,
                 pos_embedding=True):
        super().__init__()
        self.attention_x = Attention(hidden_size=hidden_size, num_heads=num_heads,
                                     attention_dropout_rate=attention_dropout_rate)
        self.attention_y = Attention(hidden_size=hidden_size, num_heads=num_heads,
                                     attention_dropout_rate=attention_dropout_rate)
        self.attention_z = Attention(hidden_size=hidden_size, num_heads=num_heads,
                                     attention_dropout_rate=attention_dropout_rate)
        self.window_attention = WindowAttention(dim=hidden_size, heads=num_heads, head_dim=hidden_size // num_heads,
                                                window_size=window_size, relative_pos_embedding=True)
        self.norm = nn.Softmax(dim=-1)
        self.is_position = pos_embedding
        if pos_embedding is True:
            self.pos_embedding1 = PositionEmbedding(hidden_size, img_size=img_size, patch_size=patch_size, types=0)
            self.pos_embedding2 = PositionEmbedding(hidden_size, img_size=img_size, patch_size=patch_size, types=0)
            self.pos_embedding3 = PositionEmbedding(hidden_size, img_size=img_size, patch_size=patch_size, types=0)

    def forward(self, x, y, z):
        B, C, D, H, W = x.shape
        w = x + y + z
        x1 = rearrange(x, "b c d h w -> b (w d h) c")
        y1 = rearrange(y, "b c d h w -> b (w d h) c")
        z1 = rearrange(z, "b c d h w -> b (w d h) c")
        w = w.permute(0, 2, 3, 4, 1)

        if self.is_position:
            x1 = self.pos_embedding1(x1)
            y1 = self.pos_embedding2(y1)
            z1 = self.pos_embedding2(z1)

        x1 = self.attention_x(x1)
        y1 = self.attention_y(y1)
        z1 = self.attention_z(z1)
        w = self.window_attention(w)

        w = rearrange(w, "b d h w c -> b (h w d) c", d=D, h=H, w=W)
        # x3 = rearrange(x3, "(b d) (h w) c -> b (h w d) c", d=D, h=H, w=W)
        # x2 = rearrange(x2, "(b h) (d w) c -> b (h w d) c", d=D, h=H, w=W)
        # x1 = rearrange(x1, "(b d) (h w) c -> b (h w d) c", d=D, h=H, w=W)

        h = self.norm(x1 * y1) * w
        h = self.norm(h)
        out = (h * z1) * w

        return out


class SelfAdaptiveTransformer(nn.Module):
    def __init__(self, hidden_size=128, img_size=(128, 128, 128), patch_size=(8, 8, 8), num_heads=4,
                 attention_dropout_rate=0.2, window_size=(4, 4, 4)):
        super().__init__()
        self.mda = SelfAdaptiveAttention(hidden_size=hidden_size, img_size=img_size, patch_size=patch_size,
                                        num_heads=num_heads,
                                        attention_dropout_rate=attention_dropout_rate, window_size=window_size,
                                        pos_embedding=True)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, hidden_size * 2, dropout_rate=0.2)

    def forward(self, x, y, z):
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b (h w d) c")
        x = self.norm(x)
        y = rearrange(y, "b c d h w -> b (h w d) c")
        y = self.norm(y)
        z = rearrange(z, "b c d h w -> b (h w d) c")
        z = self.norm(z)
        h = x + y + z
        x = rearrange(x, "b (h w d) c -> b c d h w", d=D, h=H, w=W)
        y = rearrange(y, "b (h w d) c -> b c d h w", d=D, h=H, w=W)
        z = rearrange(z, "b (h w d) c -> b c d h w", d=D, h=H, w=W)
        x1 = self.mda(x, y, z)
        x2 = h + x1
        out = self.norm(x2)
        out = self.mlp(out)
        out = x2 + out

        return out


if __name__ == '__main__':
    x = torch.randn(2, 4096, 512)
    x1 = torch.randn(1, 128, 4, 4, 4)
    pos = PositionEmbedding(hidden_size=128, img_size=[128, 128, 128], patch_size=[8, 8, 8])
    mda = SelfAdaptiveAttention(hidden_size=512, img_size=(128, 128, 128), patch_size=(8, 8, 8), num_heads=4,
                               attention_dropout_rate=0.2, window_size=(4, 4, 4))
    mdt = SelfAdaptiveTransformer(hidden_size=128, img_size=(128, 128, 128), patch_size=(32, 32, 32), num_heads=4,
                                 attention_dropout_rate=0.2, window_size=(4, 4, 4))
    # out = pos(x)
    out = mdt(x1, x1, x1)
    print(out.shape)
