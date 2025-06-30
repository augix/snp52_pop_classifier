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
        seqlen = config.seqlen+1 if config.add_cls else config.seqlen
        # self.compress1 = nn.Linear(d_emb, cd)
        self.compress1 = mlp(d_emb, rounded_square_root(d_emb*cd), cd, config.dropout)
        
        # self.compress2 = nn.Linear(seqlen, cs)
        # self.compress2 = mlp(seqlen, rounded_square_root(seqlen*cs), cs, config.dropout)

        kernel_size = config.kernel_size
        stride = config.stride
        if stride == 1:
            padding = (kernel_size - 1) // 2
        else:
            out_len1 = math.ceil(seqlen / stride)
            pad_needed1 = max(0, (out_len1 - 1) * stride + kernel_size - seqlen)
            padding1 = pad_needed1 // 2
            out_len2 = math.ceil(out_len1 / stride)
            pad_needed2 = max(0, (out_len2 - 1) * stride + kernel_size - out_len1)
            padding2 = pad_needed2 // 2            
        print(f'Conv1d: kernel_size: {kernel_size}, stride: {stride}, padding1: {padding1}, padding2: {padding2}, out_len1: {out_len1}, out_len2: {out_len2}')
        self.conv1 = nn.Conv1d(d_emb, d_emb, kernel_size, stride=stride, padding=padding1)
        self.conv2 = nn.Conv1d(d_emb, d_emb, kernel_size, stride=stride, padding=padding2)
        self.pool = nn.AdaptiveAvgPool1d(cs)
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
        # compress seq_len to cs using 1D convolution
        x = x.permute(0, 2, 1) # [batch, d_emb, seqlen]
        x = self.conv1(x) # [batch, d_emb, conv_output_size]
        x = F.relu(x) # [batch, d_emb, conv_output_size]
        x = self.conv2(x) # [batch, d_emb, conv_output_size]
        x = F.relu(x) # [batch, d_emb, conv_output_size]
        x = self.pool(x) # [batch, d_emb, cs]
        x = x.permute(0, 2, 1) # [batch, cs, d_emb]
        return x
    
    def create_emb(self, x):
        # x.shape: [batch, seqlen, d_model]
        x = self.compress_seq_len(x) # [batch, cs, d_emb]
        x = self.compress_d_model(x) # [batch, cs, cd]
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
        # x2 = emb1 + emb2 # [batch, seqlen, d_emb]
        # x = torch.cat([x1, x2], dim=-1) # [batch, seqlen, d_emb*2]
        x = self.create_emb(x) # [batch, cs*cd]
        # x = self.dropout(x)
        logits = self.head(x) # [batch, n_output_values]
        result = {
            'logits': logits,
        }
        return result
