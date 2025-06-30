import torch
from torch import nn

from model.units import mlp, RMSNorm, MLP3

def rounded_square_root(x):
    return int(round((x ** 0.5)))
    
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
        # self.compress1 = nn.Linear(d_emb, cd)
        # self.compress2 = nn.Linear(seqlen, cs)
        self.compress1 = mlp(d_emb*2, rounded_square_root(d_emb*2*cd), cd, config.dropout)
        self.compress2 = mlp(seqlen, rounded_square_root(seqlen*cs), cs, config.dropout)
        self.norm_emb = RMSNorm(cs*cd)
        # self.dropout = nn.Dropout(config.dropout)
        self.head = mlp(cs*cd, rounded_square_root(cs*cd*n_output_values), n_output_values, config.dropout)
        # self.head = nn.Linear(cs*cd, n_output_values)

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
        # x = emb1 - emb2 # [batch, seqlen, d_emb]
        x = torch.cat([emb1, emb2], dim=-1) # [batch, seqlen, d_emb*2]
        x = self.create_emb(x) # [batch, cs*cd]
        # x = self.dropout(x)
        logits = self.head(x) # [batch, n_output_values]
        result = {
            'logits': logits,
        }
        return result
