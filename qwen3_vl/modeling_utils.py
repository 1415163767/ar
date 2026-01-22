import math
import torch
import numpy as np
import torch.nn as nn


def get_3d_sincos_pos_embed(embed_dim, grid_size_t, grid_size_h, grid_size_w):
    """
    embed_dim: total embedding dim
    returns:
        pos_embed_t: [T, Dt]
        pos_embed_h: [H, Dh]
        pos_embed_w: [W, Dw]
    """
    dim_t = embed_dim - 4 * (embed_dim // 6)
    dim_h = 2 * (embed_dim // 6)
    dim_w = 2 * (embed_dim // 6)

    assert dim_t + dim_h + dim_w == embed_dim

    grid_t = np.arange(grid_size_t, dtype=np.float32)
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)

    pos_embed_t = get_1d_sincos_pos_embed_from_grid(dim_t, grid_t)
    pos_embed_h = get_1d_sincos_pos_embed_from_grid(dim_h, grid_h)
    pos_embed_w = get_1d_sincos_pos_embed_from_grid(dim_w, grid_w)

    return pos_embed_t, pos_embed_h, pos_embed_w


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class VideoPositionEmbedding(nn.Module):
    def __init__(
        self,
        max_t,
        max_h,
        max_w,
        hidden_size,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        dim_t = hidden_size - 4 * (hidden_size // 6)
        dim_h = 2 * (hidden_size // 6)
        dim_w = 2 * (hidden_size // 6)
        
        # === register frozen tables ===
        self.pos_embed_t = nn.Parameter(
            torch.zeros(max_t, dim_t), requires_grad=False
        )
        self.pos_embed_h = nn.Parameter(
            torch.zeros(max_h, dim_h), requires_grad=False
        )
        self.pos_embed_w = nn.Parameter(
            torch.zeros(max_w, dim_w), requires_grad=False
        )

    def _init_weights(self, max_t, max_h, max_w):
        pos_t, pos_h, pos_w = get_3d_sincos_pos_embed(
            self.hidden_size, max_t, max_h, max_w
        )

        self.pos_embed_t.data.copy_(torch.from_numpy(pos_t).float())
        self.pos_embed_h.data.copy_(torch.from_numpy(pos_h).float())
        self.pos_embed_w.data.copy_(torch.from_numpy(pos_w).float())

    def forward(self, position_ids):
        """
        position_ids: Tensor[3, 1, N]
        return: Tensor[N, hidden_size]
        """
        # [3, 1, N] -> [3, N]
        position_ids = position_ids[:, 0]

        t_ids = position_ids[0]  # [N]
        h_ids = position_ids[1]  # [N]
        w_ids = position_ids[2]  # [N]

        emb_t = self.pos_embed_t[t_ids]  # [N, Dt]
        emb_h = self.pos_embed_h[h_ids]  # [N, Dh]
        emb_w = self.pos_embed_w[w_ids]  # [N, Dw]

        return torch.cat([emb_t, emb_h, emb_w], dim=-1)



# --------------------------------------------------------
# TimestepEmbedder
# Reference:
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
# --------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
