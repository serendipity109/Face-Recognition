import torch  
import torch.nn as nn 

class FocalLoss(nn.Module):
    def __init__(self, classes, gamma = 0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()
        
    def forward(self, input, target, W):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        
        # Regular mechanism
        a = torch.mm(W, torch.t(W))
        a = a - torch.eye(classes).cuda()
        a = torch.max(a, 1).values.sum()
        print((loss.mean(), a*12/classes))
        return (loss.mean()+ a*12/classes)