import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class Semantic_Enhancement_Module(nn.Module):
    def __init__(self,num_class=7, groups = 64):
        super(Semantic_Enhancement_Module, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.label_w = nn.Linear(num_class,2048)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x,label): # (b, c, h, w)
        b, c, h, w = x.size()
        label_w=F.softmax(self.label_w(label),dim=1).view(b,c,1,1).repeat(1,1,w,h)

        x = x.view(b * self.groups, -1, h, w)
        label_w=label_w.view(b*self.groups,-1,h,w)
        x_in=x+label_w
        xn = x * self.avg_pool(x_in)
      #  print(xn.size())
        xn = xn.sum(dim=1, keepdim=True)
        
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x
