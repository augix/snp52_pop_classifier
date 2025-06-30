import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import math
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(self.weight) * self.weight

# class MHA(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.attn = nn.MultiheadAttention( 
#             embed_dim=args.d_model,
#             num_heads=args.n_heads,
#             dropout=args.dropout,
#             batch_first=True,
#         )
#     def forward(self, x: torch.Tensor):
#         output, _ = self.attn(x, x, x)
#         return output

from longformer.longformer import LongformerSelfAttention
class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        # same as LongformerSelfAttention parameters
        # config.attention_window = config.attention_window
        # config.attention_dilation = config.attention_dilation
        # adjust on LongformerSelfAttention parameters
        config.num_hiddens = config.d_model
        config.num_heads = config.n_heads
        config.attention_probs_dropout_prob = config.dropout
        config.attention_mode = 'sliding_chunks'
        config.autoregressive = False
        self.attn = LongformerSelfAttention(config)
    def forward(self, x: torch.Tensor):
        attn_out = self.attn(x, output_attentions=False)[0]
        return attn_out

class MLP(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class mlp(nn.Module):
    def __init__(self, d1, d2, d3, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d1, d2)
        self.fc2 = nn.Linear(d2, d3)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attn = MHA(args)
        # self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.ffn = MLP(args.d_model, args.d_ffn)
        self.attn_norm = RMSNorm(args.d_model)
        self.ffn_norm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: norm, attn, add, norm, ffn, add
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class embedding_int(nn.Module):
    def __init__(self, n_int, d_embed, dtype=torch.bfloat16):
        super().__init__()
        self.d_embed = d_embed
        self.embed = nn.Embedding(n_int, d_embed, dtype=dtype)

    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_embed)
        embeded = self.embed(x) * math.sqrt(self.d_embed)
        return embeded
    
class embedding_dec2bin(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim  # number of bits to encode the integer
        self.dtype = dtype

    def forward(self, x):
        x = decimal2binary(x, bits=self.dim, dtype=self.dtype)
        # shape: batch, seq_len, dim
        return x

def decimal2binary(integer, bits, dtype=torch.bfloat16):
    """Turn integer tensor to binary representation.
    https://github.com/KarenUllrich/pytorch-binary-converter/blob/master/binary_converter.py

    dtype: torch.float32 or torch.float64
    # !!! use torch.float64 for higher precision for very large number above 1e9 !!!

    bits: number of bits to encode the integer
    # biggest number that 32 bits can encode
    # np.power(2, 16) / 1e9, np.power(2, 32) / 1e9
    # 32 bits can encode 4.294967296e+09
    # 16 bits can encode 65536
    # 32 bits is enough for a chromosome

    """
    exponent_bits = -torch.arange(-(bits - 1), 1).type(torch.float64) 
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    exponent_bits = exponent_bits.to(integer.device) # move to the same device as input
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    out = (out - (out % 1)) % 2
    out = out.type(dtype)
    return out

class theModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.dtype = config.dtype
        super().__init__()
        self.embed = embedding_int(config.n_input_values, config.d_value)
        self.id_embed = embedding_dec2bin(config.d_id, dtype=self.dtype) # position/id embedding
        self.correct_dim = nn.Linear(config.d_value + config.d_id, config.d_model)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_output_values)

    def forward(self, x, id):
        x = self.embed(x)
        id = self.id_embed(id)
        if self.config.add_id:
            x = x + id
        else:
            x = self.correct_dim(torch.cat([x, id], dim=-1))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x) # [batch, seqlen, vocab_size]
        return logits
