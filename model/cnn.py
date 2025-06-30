import torch
from torch import nn
import torch.nn.functional as F

class theModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dtype = config.dtype
        d_emb = config.d_emb # 128
        n_output_values = config.n_output_values
        seqlen = config.seqlen # 2934
        
        self.conv1 = nn.Conv1d(d_emb*2, 64, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv1d(64, 16, kernel_size=8, stride=4, padding=1)
        
        # Calculate the output size after convolutions and pooling
        # After conv1: (seqlen + 2*1 - 8) // 4 + 1 = (2934 + 2 - 8) // 4 + 1 = 2928 // 4 + 1 = 732 + 1 = 733
        # After first max_pool1d(4): 733 // 4 = 183
        # After conv2: (183 + 2*1 - 8) // 4 + 1 = (183 + 2 - 8) // 4 + 1 = 177 // 4 + 1 = 44 + 1 = 45
        # After second max_pool1d(4): 45 // 4 = 11
        final_length = 11
        
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(16 * final_length, n_output_values)

    def forward(self, contig1, contig2, emb1, emb2):
        # emb1.shape: [batch, seqlen, d_emb]
        # emb2.shape: [batch, seqlen, d_emb]
        
        x = torch.cat([emb1, emb2], dim=-1) # [batch, seqlen, d_emb*2]
        # permute to [batch, d_emb*2, seqlen]
        x = x.permute(0, 2, 1) # [batch, d_emb*2, seqlen]
        
        # Apply convolutions and pooling
        x = F.relu(self.conv1(x)) # [batch, 64, 733]
        x = F.max_pool1d(x, 4) # [batch, 64, 183]
        x = F.relu(self.conv2(x)) # [batch, 16, 45]
        x = F.max_pool1d(x, 4) # [batch, 16, 11]
        
        # flatten
        x = x.view(x.size(0), -1) # [batch, 16*11] = [batch, 176]
        x = self.dropout(x)
        logits = self.head(x) # [batch, n_output_values]
        result = {
            'logits': logits,
        }
        return result
