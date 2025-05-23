import torch
import torch.nn as nn

AVAI_HEADS = ['linear', 'adapter']

class LoRA(nn.Module):
    def __init__(self, c_in, rank=8):
        super().__init__()
        self.lora_down = nn.Linear(c_in, rank, bias=False)
        self.lora_up = nn.Linear(rank, c_in, bias=False)

    def forward(self, x):
        return self.lora_up(self.lora_down(x))

class AdaptiveDecision(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.5, num_heads=4, rank=16):
        super(AdaptiveDecision, self).__init__()
        self.norm = nn.LayerNorm(c_in)
        self.down_proj = nn.Linear(c_in, c_in // reduction, bias=False)
        self.gate_proj = nn.Linear(c_in, c_in // reduction, bias=False)
        self.residual_ratio = nn.Parameter(torch.tensor(0.6))

        self.conv = nn.Sequential(
            nn.Conv1d(c_in // reduction, c_in // reduction, kernel_size=3, padding=1, groups=c_in // reduction,
                      bias=False),
            nn.BatchNorm1d(c_in // reduction),
            nn.Conv1d(c_in // reduction, c_in // reduction, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_in // reduction)
        )

        self.ffn = nn.Sequential(
            nn.Linear(c_in // reduction, c_in // reduction * 2),
            nn.GELU(),
            nn.Linear(c_in // reduction * 2, c_in // reduction)
        )
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, batch_first=True,
                                          dropout=0.1)
        self.up_proj = nn.Linear(c_in // reduction, c_in, bias=False)
        self.lora = LoRA(c_in, rank=rank)
        self.dropout = nn.Dropout(0.28)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down_proj(x) * torch.sigmoid(self.gate_proj(x))  # GLU
        x = self.conv(x.unsqueeze(-1)).squeeze(-1)  # 1D Conv
        x = self.ffn(x)  # FFN

        num_tokens = 4
        token_dim = 256 // num_tokens
        x = x.view(x.size(0), num_tokens, token_dim)  # [B, 16, 64]
        x, _ = self.attn(x, x, x)  # Now attention is meaningful
        x = x.reshape(residual.size(0), 256)


        x = self.up_proj(x.squeeze(1))
        x = x + self.lora(x)  # LoRA
        x = self.dropout(x)
        return self.residual_ratio * x + (1 - self.residual_ratio) * residual


def make_classifier_head(classifier_head,
                         clip_encoder,
                         classifier_init,
                         bias=False):
    assert classifier_head in AVAI_HEADS
    if clip_encoder == 'ViT-B/16':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024

    num_classes = 5

    linear_head = nn.Linear(1024, num_classes, bias=bias)
    
    if classifier_head == 'linear':
        head = linear_head
    elif classifier_head == 'adapter':
        adapter = AdaptiveDecision(in_features, residual_ratio=0.2)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, num_classes, in_features