from typing import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.get_attention = False

        self.q_weights = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_heads)])
        self.k_weights = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_heads)])
        self.v_weights = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_heads)])
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(num_heads*hidden_dim, hidden_dim)

    def forward(self, X):
        #B, N, D = X.shape
        result = []
        for x in X:
            x_result = [] # H, N, D
            x_attn_mask = [] # H, N, N
            for head in range(self.num_heads):
                q = self.q_weights[head](x)
                k = self.k_weights[head](x)
                v = self.v_weights[head](x)
                attn_weights = self.softmax(q @ k.T / self.hidden_dim**2)
                h = self.softmax(q @ k.T / self.hidden_dim**2) @ v # N, D
                x_result.append(h)
                x_attn_mask.append(attn_weights)
            result.append(torch.hstack(x_result)) # B, H, N, D
        H = torch.cat([torch.unsqueeze(r, dim=0) for r in result])
        out = self.linear(H)
        if self.get_attention:
            return out, x_attn_mask
        else:
            return out # N, D

    def retAtt(self, off=False):
        self.get_attention = True
        if off:
            self.get_attention = False


class VisionTransformer(nn.Module):
    def __init__(self, img_shape, patch_size, hidden_dim, num_heads, out_dim, num_encoder_blocks=6):
        super().__init__()

        self.img_shape = img_shape
        self.patch_size = img_shape[0]*patch_size[0]*patch_size[1]
        self.num_patches = int(img_shape[0]*img_shape[1]/patch_size[0]) ** 2
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.num_encoder_blocks = num_encoder_blocks

        # Linear patching
        self.linear_patching = nn.Linear(self.patch_size, self.hidden_dim)

        # CLS embedding
        self.cls_embedding = nn.Parameter(torch.rand(1, self.hidden_dim))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.rand(1+self.num_patches, self.hidden_dim))

        # Transformer
        self.transformer_1 = nn.Sequential(
                                nn.LayerNorm((1+self.num_patches, self.hidden_dim)),
                                MultiHeadSelfAttention(self.hidden_dim, self.num_heads)
                            )
        self.transformer_2 = nn.Sequential(
                                nn.LayerNorm((1+self.num_patches, self.hidden_dim)),
                                nn.Linear(self.hidden_dim, self.hidden_dim),
                            )

        # MLP head
        self.mlp_head = nn.Sequential(
                            nn.Linear(self.hidden_dim, self.out_dim),
                            nn.Tanh(),
                        )

    def forward(self, X, getAttention=False):
        if getAttention:
            self.transformer_1[1].retAtt()
        else:
            self.transformer_1[1].retAtt(off=True)
        try:
            N, C, H, W = X.shape
        except ValueError:
            N, C, H = X.shape
        patches = X.reshape(N, self.num_patches, self.patch_size)
        E = self.linear_patching(patches)
        cls_embedding = nn.Parameter(self.cls_embedding.repeat(N, 1, 1))
        E = torch.cat([cls_embedding, E], dim=1)
        Epos = nn.Parameter(self.pos_embedding.repeat(N, 1, 1))
        Z = E + Epos

        if getAttention:
            atts = []
            for _ in range(self.num_encoder_blocks):
                res1, attention_masks = self.transformer_1(Z)
                Z = self.transformer_2(res1 + Z)
                atts.append(attention_masks)
            C = self.mlp_head(Z[:,0])
            return C, atts

        for _ in range(self.num_encoder_blocks):
            res1 = self.transformer_1(Z)
            Z = self.transformer_2(res1 + Z)
        C = self.mlp_head(Z[:, 0])
        return C

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    TODO: use this with adam
    """
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        print(lr_factor * self.base_lrs[0])
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
            return lr_factor
        return lr_factor
