import torch
from torch import nn
import torch.nn.functional as F
from model.units import mlp, RMSNorm, MLP3
import math

def rounded_square_root(x):
    return int(round((x ** 0.5)))
    
class theModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = config.dtype
        d_emb = config.d_emb
        n_output_values = config.n_output_values
        cs = config.cs
        cd = config.cd
        self.bin_sizes = config.bin_sizes
        
        # seqlen = config.seqlen+1 if config.add_cls else config.seqlen
        # self.compress1 = nn.Linear(d_emb, cd)
        self.compress1 = mlp(d_emb*2, rounded_square_root(d_emb*2*cd), cd, config.dropout)
        self.norm1 = RMSNorm(cd)
        # self.compress2 = nn.Linear(seqlen, cs)
        # self.compress2 = mlp(seqlen, rounded_square_root(seqlen*cs), cs, config.dropout)
        self.norm2 = RMSNorm(sum(self.bin_sizes))
        self.correct_dim = nn.Linear(sum(self.bin_sizes), cs)
        # self.correct_dim = mlp(sum(self.bin_sizes), rounded_square_root(sum(self.bin_sizes)*cs), cs, config.dropout)
        # self.dropout = nn.Dropout(config.dropout)
        self.norm_emb = RMSNorm(cs*cd)
        self.head = mlp(cs*cd, rounded_square_root(cs*cd*n_output_values), n_output_values, config.dropout)
        # self.head = nn.Linear(cs*cd, n_output_values)

    def compress_d_model(self, x):
        # x.shape: [batch, seqlen, d_model]
        # compress d_model to cd
        x = self.compress1(x) # [batch, seqlen, cd]
        x = self.norm1(x)
        return x

    def spatial_pyramid_pool(self, x):
        # x.shape: [batch, d_model, seqlen]
        features = []
        for bin_size in self.bin_sizes:
            features.append(nn.AdaptiveAvgPool1d(bin_size)(x))
        return torch.cat(features, dim=-1)

    def compress_seq_len(self, x):
        # x.shape: [batch, seqlen, d_model]
        # compress seq_len to cs 
        x = x.permute(0, 2, 1) # [batch, d_emb, seqlen]
        x = self.spatial_pyramid_pool(x) # [batch, d_emb, sum(bin_sizes)]
        x = self.norm2(x)
        x = F.silu(x)
        x = self.correct_dim(x) # [batch, d_emb, cs]
        x = x.permute(0, 2, 1) # [batch, cs, d_emb]
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
        # x = torch.cat([emb1, emb2], dim=-1) # [batch, seqlen, d_emb*2]
        x = emb1 - emb2 # [batch, seqlen, d_emb]
        # x2 = emb2 - emb2 # [batch, seqlen, d_emb]
        x = torch.cat([x, emb1], dim=-1) # [batch, seqlen, d_emb*2]
        x = self.create_emb(x) # [batch, cs*cd]
        # x = self.dropout(x)
        logits = self.head(x) # [batch, n_output_values]
        result = {
            'logits': logits,
        }
        return result