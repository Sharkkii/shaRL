#### Utility ####

import math
import torch

# element-wise softplus
# NOTE: beta is the inverse!
def softplus(x, beta=1.0, thr=10.):
    y = torch.where(beta * x >= thr, x, (1./beta) * torch.log(1. + torch.exp(beta * x)))
    return y

# element-wise gauss
def gauss(x, mean, std):
    return (torch.exp(- (x - mean)**2 / (2.0 * std**2))) / torch.sqrt(2.0 * math.pi * std**2)