from .imports import *
from .torch_imports import *

# class AdaptiveConcatPool2d(nn.Module):
#     def __init__(self, sz=None):
#         super().__init__()
#         sz = sz or (1,1)
#         self.ap = nn.AdaptiveAvgPool2d(sz)
#         self.mp = nn.AdaptiveMaxPool2d(sz)
#     def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
       

    def forward(self, x): 
        inp_size = x.size()
        mp = torch.nn.functional.max_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))
        ap = torch.nn.functional.avg_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))
        
        return torch.cat([mp,ap], 1)

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)


class BatchNormExpand(nn.Module): 
    def __init__(self): super().__init__()
    def forward(self, x): 
        inp_size = x.size()
        return x.view(*inp_size,1,1)

class BatchNormContract(nn.Module): 
    def __init__(self): super().__init__()
    def forward(self, x): 
        inp_size = x.size()
        return x.view(*inp_size[:-2])
    
