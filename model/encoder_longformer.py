import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from longformer.longformer import LongformerSelfAttention

class EncoderBlock(nn.Module):
    
    def __init__(self, n_heads, d_model, dim_feedforward, dropout, args):
        """
        Inputs:
            d_model - Dimensionality of the input
            n_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        
        # Attention layer
        config = args
        config.num_hiddens = d_model
        config.num_heads = n_heads
        config.attention_probs_dropout_prob = dropout
        # config.attention_window = args.attention_window
        # config.attention_dilation = args.attention_dilation
        config.attention_mode = 'sliding_chunks'
        config.autoregressive = False
        self.self_attn = LongformerSelfAttention(config)
        
        # Feedforward
        self.linear_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # order: attn, dropout, add, norm, ffn, dropout, add, norm
        # Attention 
        attn_out = self.self_attn(x, output_attentions=False)[0]

        # Dropout, ADD, Normalization
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feedforward
        linear_out = self.linear_net(x)

        # Dropout, ADD, Normalization
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        
        return x
    
class TransformerEncoder(nn.Module):
    
    def __init__(self, n_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
    
    def forward_w_attn(self, x, mask=None):
        map_list = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, output_attentions=True)
            # attn_map is the attention weights for each head. shape: (batchs, heads, seq_len, win*2+1)
            map_list.append(attn_map)
            x = l(x, mask=mask)  # x went through the iterations of layers
        attention_maps = torch.stack(map_list)
        # attention_maps is the attention weights for each layer. shape: (layers, batchs, heads, seq_len, win*2+1)
        return x, attention_maps

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, output_attentions=True)
            attention_maps.append(attn_map)
            x = l(x, mask=mask)  # x went through the iterations of layers
        return attention_maps

def get_encoder(args):
    return TransformerEncoder(n_layers=args.n_layers,
                              n_heads=args.n_heads,
                              d_model=args.d_model,
                              dim_feedforward=args.d_ffn,
                              dropout=args.dropout,
                              args=args)