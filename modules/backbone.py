"""backbone from wj"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNet50(nn.Module):
    """Residual Network"""
    
    def __init__(self, out_features, device, is_normalize=True):
        super().__init__()
        self.model = resnet50().to(device)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_features).to(device)
        self.is_normalize = is_normalize
        
        
    def forward(self, x):
        out = self.model(x)
        if self.is_normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class MobileNetV2(nn.Module):
    """MobileNet-v2"""
    pass

