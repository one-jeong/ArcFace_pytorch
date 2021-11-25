import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from modules.backbone import ResNet50
import math


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            num_classes: size of each output sample
            m: margin
    """
    
    def __init__(self, device, num_classes, in_features=512, scale_factor=128.0, margin=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(num_classes, in_features)).to(device)
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, device, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), requires_grad=True).to(device)

        one_hot = one_hot.scatter(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale_factor

        return output


class ArcFace(nn.Module):

    def __init__(self, device, num_classes, scale_factor=64, margin=0.5, backbone_type='resnet', training=False, name='ArcFace'):
        super(ArcFace, self).__init__()
        
        self.device = device
        self.scale_factor = scale_factor
        self.margin = margin
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.training = training
        self.backbone = ResNet50(512, device)
        self.arcmarginProduct = ArcMarginProduct(device, in_features=512, num_classes=num_classes).to(device)


    def forward(self,device, x, labels):
        device = self.device
        x = self.backbone(x)
        if self.training:
            x = self.arcmarginProduct(device, x, labels)
        return x
