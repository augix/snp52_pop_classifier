import torch
from torch import nn
import torch.nn.functional as F
# from typing import Optional
import math

# required config attributes:
# config.seq_len
# config.d_model
# config.d_emb
# config.d_ffn
# config.n_heads
# config.n_layers
# config.n_id
# config.d_id
# config.n_output_values
# config.dtype
# config.dropout

# class Linear(nn.Module):
#     dtype = torch.bfloat16
#     def __init__(self, 
#                  in_features: int, 
#                  out_features: int, 
#                  bias: bool = False, 
#                  dtype = None):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(self.out_features))
#         else:
#             self.register_parameter("bias", None)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return F.linear(x, self.weight, self.bias)
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MHA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attn = nn.MultiheadAttention( 
            embed_dim=args.d_model,
            num_heads=args.n_heads,
            dropout=args.dropout,
            batch_first=True,
        )
    def forward(self, x: torch.Tensor):
        output, _ = self.attn(x, x, x)
        return output

class MLP(nn.Module):
    def __init__(self, d1, d2, d3):
        super().__init__()
        self.w1 = nn.Linear(d1, d2)
        self.w2 = nn.Linear(d1, d2)
        self.w3 = nn.Linear(d2, d3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

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
        self.ffn = MLP(args.d_model, args.d_ffn, args.d_model)
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
        d_emb = config.d_emb
        n_output_values = config.n_output_values
        cs = config.cs
        cd = config.cd
        seqlen = config.seqlen
        self.compress1 = mlp(d_emb, cd, cd, dropout=config.dropout)
        if config.add_cls:
            self.compress2 = mlp(seqlen+1, cs, cs, dropout=config.dropout)
        else:
            self.compress2 = mlp(seqlen, cs, cs, dropout=config.dropout)
        self.norm_emb = RMSNorm(cs*cd)
        self.head = mlp(cs*cd, n_output_values, 1, dropout=config.dropout)

    def compress_d_model(self, x):
        # x.shape: [batch, seqlen, d_model]
        # compress d_model to cd
        x = self.compress1(x) # [batch, seqlen, cd]
        return x

    def compress_seq_len(self, x):
        # x.shape: [batch, seqlen, d_model]
        # compress seq_len to cs
        x = x.permute(0, 2, 1) # [batch, d_model, seqlen]
        x = self.compress2(x)  # [batch, d_model, cs]
        x = x.permute(0, 2, 1) # [batch, cs, d_model]
        return x
    
    def create_emb(self, x):
        # x.shape: [batch, seqlen, d_model]
        x = self.compress_d_model(x) # [batch, seqlen, cd]
        x = self.compress_seq_len(x) # [batch, cs, cd]
        # reshape to [batch, cs*cd]
        x = x.reshape(x.shape[0], -1) # [batch, cs*cd]
        x = self.norm_emb(x)
        return x

    def forward(self, contig1, contig2, emb1, emb2):
        # contig1.shape: [batch, seqlen]
        # contig2.shape: [batch, seqlen]
        # emb1.shape: [batch, seqlen, d_emb]
        # emb2.shape: [batch, seqlen, d_emb]
        x = emb1 - emb2 # [batch, seqlen, d_emb]
        # x = torch.cat([emb1, emb2], dim=-1) # [batch, seqlen, d_emb*2]
        x = self.create_emb(x) # [batch, cs*cd]
        logits = self.head(x) # [batch, 1]
        # squeeze
        logits = logits.squeeze(-1)
        result = {
            'logits': logits,
        }
        return result

