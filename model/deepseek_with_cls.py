import torch
from torch import nn

from model.units import RMSNorm, MLP, MLP3, mlp, embedding_int, rounded_square_root

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

class Block(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attn = MHA(args)
        self.ffn = MLP3(args.d_model, args.d_ffn, args.d_model)
        self.attn_norm = RMSNorm(args.d_model)
        self.ffn_norm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm: norm, attn, add, norm, ffn, add
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class theModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.dtype = config.dtype
        super().__init__()
        d_model = config.d_model
        d_id = config.d_id
        d_emb = config.d_emb
        n_output_values = config.n_output_values
        self.id_embed = embedding_int(config.n_id, d_id)
        self.correct_dim = mlp(d_emb*2 + d_id, d_model, d_model)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(Block(layer_id, config))
        self.norm = RMSNorm(d_model)
        self.head = mlp(d_model, rounded_square_root(d_model*n_output_values), n_output_values, dropout=config.dropout)

    def forward(self, contig1, contig2, emb1, emb2):
        # contig1.shape: [batch, seqlen]
        # contig2.shape: [batch, seqlen]
        # emb1.shape: [batch, seqlen, d_emb]
        # emb2.shape: [batch, seqlen, d_emb]
        
        id = self.id_embed(contig1) # [batch, seqlen, d_id]
        x = torch.cat([id, emb1, emb2], dim=-1) # [batch, seqlen, d_emb*2+d_id]
        x = self.correct_dim(x) # [batch, seqlen, d_model]
        for layer in self.layers:
            x = layer(x)
        # encoder output: [batch, seqlen, d_model]
        x = self.norm(x)
        x = x[:, 0, :] # [batch, d_model]
        logits = self.head(x) # [batch, n_output_values]
        result = {
            'logits': logits,
        }
        return result

