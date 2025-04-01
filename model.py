import torch
import torch.nn as nn
class AdaptiveDecision(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.5, num_heads=4, rank=16):
        super(AdaptiveDecision, self).__init__()
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
