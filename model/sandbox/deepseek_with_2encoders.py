import torch
from torch import nn
import torch.nn.functional as F
# from typing import Optional
import math

# required config attributes:
# config.seq_len
# config.d_model
# config.d_ffn
# config.n_heads
# config.n_layers
# config.d_value
# config.d_id
# config.n_input_values
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
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(self.weight) * self.weight

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
        x = F.silu(x)
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

class encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class theModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.dtype = config.dtype
        super().__init__()
        d_model = config.d_model
        seqlen = config.contig_len
        cd = config.cd
        cs = config.cs
        n_output_values = config.n_output_values
        if self.config.add_id:
            d_id = d_model
            d_value = d_model
        else:
            d_id = config.d_id
            d_value = config.d_value
            self.correct_dim = nn.Linear(d_value + d_id, d_model)
        self.embed = embedding_int(config.n_input_values, d_value)
        self.id_embed = embedding_dec2bin(d_id, dtype=self.dtype) # position/id embedding
        # self.layers = torch.nn.ModuleList()
        # for layer_id in range(config.n_layers):
        #     self.layers.append(Block(layer_id, config))
        self.encoder1 = encoder(config)
        self.encoder2 = encoder(config)
        self.norm = RMSNorm(d_model)
        # self.mlp = MLP(d_model, d_model, d_model)
        self.head = nn.Linear(d_model, n_output_values)
        # create emb
        self.compress1 = nn.Linear(d_model, cd)
        self.compress2 = nn.Linear(seqlen+1, cs) # +1 for the cls token
        # predict from emb
        self.mlp2 = MLP(cs*cd+d_id, d_model, d_model)
        self.norm2 = RMSNorm(d_model)
        self.head2 = nn.Linear(d_model, n_output_values)

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
        return x

    def predict_from_emb(self, emb, id):
        # emb.shape: [batch, cs*cd]
        # id.shape: [batch, seqlen, d_id]
        # emb reshape to [batch, 1, cs*cd]
        emb = emb.reshape(emb.shape[0], 1, -1) # [batch, 1, cs*cd]
        # emb repeat to [batch, seqlen, cs*cd]
        emb = emb.repeat(1, id.shape[1], 1) # [batch, seqlen, cs*cd]

        # concat emb and id
        x = torch.cat([emb, id], dim=-1) # [batch, seqlen, cs*cd+d_id]
        x = self.mlp2(x)
        x = self.norm2(x)
        x = self.head2(x) # [batch, seqlen, n_output_values]
        return x

    def forward(self, x, id):
        x = self.embed(x)
        id = self.id_embed(id)
        if self.config.add_id:
            x = x + id
        else:
            x = self.correct_dim(torch.cat([x, id], dim=-1))
        x = self.encoder1(x)
        # encoder output: [batch, seqlen, d_model]
        # x = self.mlp(x)
        x = self.norm(x)
        logits0 = self.head(x) # [batch, seqlen, n_output_values]
        x = self.encoder2(x)
        emb = self.create_emb(x) # [batch, cs*cd]
        logits = self.predict_from_emb(emb, id) # [batch, seqlen, n_output_values]
        result = {
            'logits0': logits0,
            'logits': logits,
        }
        return result

#TODO: 
# add norm and mlp before head
