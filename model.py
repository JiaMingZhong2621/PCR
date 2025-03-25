# -*- coding: gbk -*-
import torch
import torch.nn as nn
class AdaptiveDecision(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.5, num_heads=4, rank=16):
        super(AdaptiveDecision, self).__init__()
        self.norm = nn.LayerNorm(c_in)
        self.down_proj = nn.Linear(c_in, c_in // reduction, bias=False)
        self.gate_proj = nn.Linear(c_in, c_in // reduction, bias=False)

        self.DWConvB = nn.Sequential(
            nn.Conv1d(c_in // reduction, c_in // reduction, kernel_size=3, padding=1, groups=c_in // reduction,
                      bias=False),
            nn.BatchNorm1d(c_in // reduction),
            nn.Conv1d(c_in // reduction, c_in // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_in // reduction)
        )

        self.LightFNN = nn.Sequential(
            nn.Linear(c_in // reduction, c_in // reduction * 2),
            nn.GELU(),
            nn.Linear(c_in // reduction * 2, c_in // reduction)
        )
        self.MHSA = nn.MultiheadAttention(embed_dim=c_in // reduction, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.up_proj = nn.Linear(c_in // reduction, c_in, bias=False)
        self.residual_ratio = 0.6
        self.dropout = nn.Dropout(0.3)
        self.lora_down = nn.Linear(c_in, rank, bias=False)
        self.lora_up = nn.Linear(rank, c_in, bias=False)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(x) * torch.sigmoid(self.gate_proj(x))  # GLU
        x = self.DWConvB(x.unsqueeze(-1)).squeeze(-1)  # 1D Conv
        x = self.LightFNN(x)
        x, _ = self.MHSA(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))  # Self-Attention
        x = self.up_proj(x.squeeze(1))
        x = x + self.lora_up(self.lora_down(x))
        x = self.dropout(x)
        return self.residual_ratio * x + (1 - self.residual_ratio) * residual