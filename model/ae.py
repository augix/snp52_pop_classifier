import torch
from torch import nn

from model.units import mlp, RMSNorm, MLP3, rounded_square_root

class theModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.dtype = config.dtype
        super().__init__()
        d_emb = config.d_emb
        n_output_values = config.n_output_values
        cs = config.cs
        cd = config.cd
        seqlen = config.seqlen+1 if config.add_cls else config.seqlen
        self.compress1 = nn.Linear(d_emb, cd)
        self.compress2 = nn.Linear(seqlen, cs)
        # self.compress1 = mlp(d_emb, rounded_square_root(d_emb*cd), cd, config.dropout)
        # self.compress2 = mlp(seqlen, rounded_square_root(seqlen*cs), cs, config.dropout)
        self.norm_emb = RMSNorm(cs*cd)
        # self.dropout = nn.Dropout(config.dropout)
        self.head = mlp(cs*cd, rounded_square_root(cs*cd*n_output_values), n_output_values, config.dropout)
        # self.head = nn.Linear(cs*cd, n_output_values)
        self.decompress1 = nn.Linear(cd, d_emb)
        self.decompress2 = nn.Linear(cs, seqlen)
        # self.decompress1 = mlp(cd, rounded_square_root(d_emb*cd), d_emb, config.dropout)
        # self.decompress2 = mlp(cs, rounded_square_root(seqlen*cs), seqlen, config.dropout)
        
    def decompress_seq_len(self, x):
        # x.shape: [batch, cs, cd]
        x = x.permute(0, 2, 1) # [batch, cd, cs]
        x = self.decompress2(x) # [batch, cd, seqlen]
        x = x.permute(0, 2, 1) # [batch, seqlen, cd]
        return x

    def decompress_d_emb(self, x):
        # x.shape: [batch, seqlen, cd]
        x = self.decompress1(x) # [batch, seqlen, d_emb]
        return x

    def compress_d_emb(self, x):
        # x.shape: [batch, seqlen, d_emb]
        # compress d_emb to cd
        x = self.compress1(x) # [batch, seqlen, cd]
        return x

    def compress_seq_len(self, x):
        # x.shape: [batch, seqlen, d_emb]
        # compress seq_len to cs
        x = x.permute(0, 2, 1) # [batch, d_emb, seqlen]
        x = self.compress2(x)  # [batch, d_emb, cs]
        x = x.permute(0, 2, 1) # [batch, cs, d_emb]
        return x
    
    def compress(self, x):
        # x.shape: [batch, seqlen, d_emb]
        x = self.compress_d_emb(x) # [batch, seqlen, cd]
        x = self.compress_seq_len(x) # [batch, cs, cd]
        # reshape to [batch, cs*cd]
        x = x.reshape(x.shape[0], -1) # [batch, cs*cd]
        x = self.norm_emb(x)
        return x

    def decompress(self, x):
        # x.shape: [batch, cs*cd]
        x = x.reshape(x.shape[0], self.config.cs, self.config.cd) # [batch, cs, cd]
        x = self.decompress_seq_len(x) # [batch, seqlen, cd]
        x = self.decompress_d_emb(x) # [batch, seqlen, d_emb]
        return x

    def forward(self, contig, emb):
        # contig.shape: [batch, seqlen]
        # emb.shape: [batch, seqlen, d_emb]
        
        compressed = self.compress(emb) # [batch, cs*cd]
        logits = self.head(compressed) # [batch, n_output_values]
        decompressed = self.decompress(compressed) # [batch, seqlen, d_emb]

        result = {
            'logits': logits,
            'decompressed': decompressed,
        }
        return result

