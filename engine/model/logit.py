import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x


class LogitScale(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable
        self.logit_scale = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, x,y):
        logit = x * self.logit_scale + y * (1-self.logit_scale)
        return logit