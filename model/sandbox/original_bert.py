import torch, torch.nn as nn, torch.nn.functional as F
import math

# --------------------------------
#  Define BERT model
# --------------------------------
# Required config attributes for this script:
# config.encoder
# config.n_layers 
# config.n_heads
# config.d_model
# config.d_id
# config.d_value
# config.d_ffn
# config.dropout

# config.n_input_values
# config.n_output_values
# config.attention_window
# config.attention_dilation


class embedding_int(nn.Module):
    def __init__(self, n_int, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.embed = nn.Embedding(n_int, d_embed)
    def forward(self, x):
        # (batch, seq_len) -> (batch, seq_len, d_embed)
        embeded = self.embed(x) * math.sqrt(self.d_embed)
        return embeded

class embedding_dec2bin(nn.Module):
    def __init__(self, dim, dtype=torch.bfloat16):
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

class mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.link = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.link(x)

class input_embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        dropout = config.dropout
        if self.config.add_id:
            d_id = d_model
            d_value = d_model
        else:
            d_id = config.d_id
            d_value = config.d_value
            self.correct_dim = mlp(d_id+d_value, d_model, d_model, dropout)

        self.id_embedding = embedding_dec2bin(d_id, config.dtype)
        self.value_embedding = embedding_int(config.n_input_values, d_value)

    def forward(self, id, x):
        id = self.id_embedding(id)
        x = self.value_embedding(x)
        if self.config.add_id:
            x = x + id
        else:
            x = self.correct_dim(torch.cat([x, id], dim=-1))
        return x

def create_encoder(config):
    if config.encoder == 'longformer':
        from model.encoder_longformer import get_encoder 
    elif config.encoder == 'transformer':
        from model.encoder_transformer import get_encoder 
    else:
        raise ValueError(f"Encoder {config.encoder} not found")
    encoder = get_encoder(config)
    return encoder

class theModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_embedding = input_embedding(config)
        self.encoder = create_encoder(config)
        d_model = config.d_model
        n_output_values = config.n_output_values
        dropout = config.dropout
        self.decoder = mlp(d_model, d_model, n_output_values, dropout)
        self.validation_outputs = []
                    
    def forward(self, x, id): 
        x = self.input_embedding(id, x)
        encoder_output = self.encoder(x)
        logits = self.decoder(encoder_output)
        # logits.shape: (batch, seq_len, n_values)
        return logits
    
