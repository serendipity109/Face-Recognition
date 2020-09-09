import torch  
import torch.nn as nn  
from torch.nn import Parameter
import torch.nn.functional as F
import math  

class ArcMarginModel(nn.Module):
    def __init__(self, classes):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(classes, 512)) # 2 classes
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = False
        self.m = 0.5
        self.s = 64.0

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)     # state train:0 test:1
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        values, indices = torch.max(phi, 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return (output, W)