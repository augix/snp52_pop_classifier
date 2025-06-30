import torch
from torch import nn
import torch.nn.functional as F
# from typing import Optional
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
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

from longformer.longformer import LongformerSelfAttention
class longformer_MHA(nn.Module):
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

class longformer_Block(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attn = longformer_MHA(args)
        # self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.ffn = MLP(args.d_model, args.d_ffn)
        self.attn_norm = RMSNorm(args.d_model)
        self.ffn_norm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: norm, attn, add, norm, ffn, add
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

Block = longformer_Block

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

from model.mha import MultiHeadAttention as MHA

class transformer_Block(nn.Module):
    def __init__(self, input_dim, embed_dim, ffn_dim, num_heads, dropout):
        super().__init__()
        # self.attn = MHA(args)
        self.attn = MHA(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.ffn = MLP(embed_dim, ffn_dim)
        self.attn_norm = RMSNorm(embed_dim)
        self.ffn_norm = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: norm, attn, add, norm, ffn, add
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

# Block = transformer_Block

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
        d_model = config.d_model
        if self.config.add_cls:
            seqlen = config.contig_len + config.n_cls
        else:
            seqlen = config.contig_len
        # cd = config.cd
        # cs = config.cs
        # d_emb = cd * cs
        n_output_values = config.n_output_values
        if self.config.add_id:
            d_id = d_model
            d_value = d_model
        else:
            d_id = config.d_id
            d_value = config.d_value
            self.correct_dim = nn.Linear(d_value + d_id, d_model)
        # embedding
        self.embed = embedding_int(config.n_input_values, d_value)
        self.id_embed = embedding_dec2bin(d_id, dtype=self.dtype) # position/id embedding
        # self.id_embed = embedding_int(seqlen, d_id)
        # attention-based encoder
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))
        # decode from encoder output
        self.mlp_head0 = mlp(d_model, d_model, d_model)
        self.norm_head0 = RMSNorm(d_model)
        self.head0 = nn.Linear(d_model, n_output_values)
        # create emb by transformer
        self.correct_dim_in = nn.Linear(config.d_model, config.d_model2)
        self.emb_layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers2):
            self.emb_layers.append(transformer_Block(config.d_model2, config.d_model2, config.d_ffn2, config.n_heads2, config.dropout))
        d_emb = config.d_emb
        self.correct_dim_out = nn.Linear(config.d_model2, d_emb)
        # decode from emb
        self.norm_head1 = RMSNorm(d_emb)
        self.head1 = nn.Linear(d_emb, seqlen*n_output_values)

    def decode_from_encoder_output(self, x):
        x = self.mlp_head0(x)
        x = self.norm_head0(x)
        logits0 = self.head0(x) # [batch, seqlen, n_output_values]
        return logits0

    def create_emb(self, x):
        # x.shape: [batch, seqlen, d_model]
        x = self.correct_dim_in(x)
        for layer in self.emb_layers:
            x = layer(x)
        x = x[:,0,:]
        x = self.correct_dim_out(x)
        return x

    def decode_from_emb(self, emb):
        emb = self.norm_head1(emb)
        logits = self.head1(emb)
        logits = logits.reshape(logits.shape[0], -1, self.config.n_output_values)
        return logits

    def forward(self, x, id):
        # local_id = self.create_local_id(id)

        x = self.embed(x)
        id = self.id_embed(id)

        if self.config.add_id:
            x = x + id
        else:
            x = self.correct_dim(torch.cat([x, id], dim=-1))

        for layer in self.layers:
            x = layer(x)
        # encoder output: [batch, seqlen, d_model]

        logits0 = self.decode_from_encoder_output(x) # [batch, seqlen, n_output_values]
        emb = self.create_emb(x) # [batch, cs*cd]
        # logits = self.decode_from_emb(emb, local_id) # [batch, seqlen, n_output_values]
        logits = self.decode_from_emb(emb) # [batch, seqlen, n_output_values]
        result = {
            'logits0': logits0,
            'logits': logits,
        }
        return result

#TODO: 
# add norm and mlp before head
