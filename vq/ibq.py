import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


class IBQ(nn.Module):
    def __init__(self, n_e, e_dim, quantization_temp=2.0, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.quantization_temp = quantization_temp
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.init = False
        self.codebook_loss_weight = 0.25
        self.is_train = False
    
    def forward(self, z):
        # z = F.normalize(z, p=2, dim=-1)
        # embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        embedding = self.embedding.weight

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('b d,d n->b n', z, torch.einsum('n d -> d n', embedding))
        
        if self.is_train:
            logits = -d / self.quantization_temp
            soft_one_hot = F.softmax(logits, dim=1)
            min_encoding_indices = soft_one_hot.max(1, keepdim=True)[1]
            hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(1, min_encoding_indices, 1.0)
            one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

            z_q = torch.einsum('b n, n d -> b d', one_hot, self.embedding.weight)
            z_q_2 = torch.einsum('b n, n d -> b d', hard_one_hot, self.embedding.weight)

            # compute loss for embedding
            commit_loss = torch.mean((z_q - z) ** 2) + torch.mean((z_q_2.detach() - z) ** 2) + self.beta * torch.mean((z_q_2 - z.detach()) ** 2)
            commit_loss = self.codebook_loss_weight * commit_loss
        else:
            min_encoding_indices = torch.argmin(d, dim=1)
            z_q = embedding[min_encoding_indices].view(z.shape)
            commit_loss = torch.tensor(0.0)

        num_codes = min_encoding_indices[:, 0].unique().numel() if min_encoding_indices.ndim > 1 else min_encoding_indices.unique().numel()
        if dist.get_rank() == 0:
            print(f"[IBQ Infer log] | Sequence length={z.shape[0]:4d} | Unique codes={num_codes:4d} | "f"Commit loss={commit_loss.item():.6f}")

        return z_q, min_encoding_indices

    # def forward(self, z, temp=None, rescale_logits=False, return_logits=False, **kwargs):
    #     assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
    #     assert not rescale_logits, "Only for interface compatible with Gumbel"
    #     assert not return_logits, "Only for interface compatible with Gumbel"

    #     # L2 归一化
    #     if self.l2_norm:
    #         z = F.normalize(z, p=2, dim=-1)
    #         embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
    #     else:
    #         embedding = self.embedding.weight

    #     # 计算距离 d
    #     d = torch.sum(z ** 2, dim=1, keepdim=True) + \
    #         torch.sum(embedding**2, dim=1) - 2 * torch.matmul(z, embedding.t())

    #     if self.training:
    #         # 避免 softmax 数值爆炸
    #         logits = -d / max(self.quantization_temp, 1e-6)
    #         soft_one_hot = F.softmax(logits, dim=1)
    #         min_encoding_indices = soft_one_hot.argmax(dim=1, keepdim=True)

    #         # Hard + straight-through
    #         hard_one_hot = torch.zeros_like(logits).scatter_(1, min_encoding_indices, 1.0)
    #         one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

    #         # quantized vectors
    #         z_q = torch.matmul(one_hot, self.embedding.weight)
    #         z_q_2 = torch.matmul(hard_one_hot, self.embedding.weight)

    #         # commit + codebook loss
    #         commit_loss = (
    #             torch.mean((z_q - z) ** 2) +
    #             torch.mean((z_q_2.detach() - z) ** 2) +
    #             self.beta * torch.mean((z_q_2 - z.detach()) ** 2)
    #         )
    #         commit_loss = self.codebook_loss_weight * commit_loss
    #     else:
    #         # 推理用
    #         min_encoding_indices = torch.argmin(d, dim=1)
    #         z_q = embedding[min_encoding_indices].view(z.shape)
    #         commit_loss = None

    #     # skip_quantization_prob 可选
    #     if self.training and self.skip_quantization_prob > 0.0:
    #         mask = torch.rand_like(z_q[:, 0:1]) <= self.skip_quantization_prob
    #         z_q = torch.where(mask.expand_as(z_q), z, z_q)

    #     print(min_encoding_indices[:10, 0])

    #     return z_q, min_encoding_indices, dict(loss=commit_loss)

    # def decode(self, indices):
    #     z_q = self.embedding(indices)
    #     return z_q


class ResidualBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding='same')
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.activate = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, 3, padding='same')
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x + res


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Process_in(nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim),
            nn.GELU(),
            nn.Linear(codebook_dim, codebook_dim),
        )
        self.ln_q = Qwen2RMSNorm(codebook_dim, eps=1e-6)

    def forward(self, x):
        x = self.ln_q(x)
        x = self.mlp(x)
        return x


class Recon_Head(nn.Module):
    def __init__(self, codebook_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim),
            nn.SiLU(),
            nn.Linear(codebook_dim, codebook_dim),
        )
        self.ln_q = Qwen2RMSNorm(codebook_dim, eps=1e-6)

    def forward(self, x):
        x = self.ln_q(x)
        x_in = x
        x = self.mlp(x)
        x = x + x_in
        return x


class VQ(nn.Module):
    def __init__(
        self, 
        z_channels=2048, 
        codebook_size=16384, 
        codebook_dim=2048, 
        transformer_in_layers=0,
        transformer_out_layers=2,
        use_transformer=False,
        config=None,
        has_mlp_pre=False
    ):
        super().__init__()
        self.quantize = IBQ(codebook_size, codebook_dim)
        # self.quantize = new_IBQ(codebook_size, codebook_dim, norm=False, is_train=False, is_resume=True)
        if use_transformer:
            assert config is not None, "Config must be provided for transformer-based quantization"
            from qwen3_vl import Qwen3_VLVisionBlock
            config.hidden_size = z_channels
            self.transformer_in = nn.ModuleList(
                [Qwen3_VLVisionBlock(config, config._attn_implementation) for _ in range(transformer_in_layers)]
            )
            if has_mlp_pre:
                self.process_in = Process_in(codebook_dim)
            self.transformer_out = nn.ModuleList(
                [Qwen3_VLVisionBlock(config, config._attn_implementation) for _ in range(transformer_out_layers)]
            )

        self.use_transformer = use_transformer
        self.config = config
        self.has_mlp_pre = has_mlp_pre
    
    
    def forward(self, x, cu_seqlens=None, position_embeddings=None, is_image=None, info="vq_final"):
        if self.use_transformer and len(self.transformer_in) > 0:
            for blk in self.transformer_in:
                x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        
        if self.has_mlp_pre:
            x = self.process_in(x)
        
        # print("x min:", x.min().item(), "x max:", x.max().item())
        assert torch.isfinite(x).all(), f"x has NaN/Inf: {x}"

        # quantize
        codebook_loss = None
        if self.quantize.is_train:
            x, code_idx, codebook_loss = self.quantize(x)
        else:
            x, code_idx = self.quantize(x)

        if self.use_transformer and len(self.transformer_out) > 0:
            for blk in self.transformer_out:
                x = blk(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        return x, code_idx, codebook_loss

