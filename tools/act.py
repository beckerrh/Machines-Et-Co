import torch
#-------------------------------------------------------------
class ReLU_smooth(torch.nn.Module):
    def __init__(self, device=None, dtype=None, **kwargs):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.lam = kwargs.pop("lambda", 3.0)
    def forward(self, x):
        return torch.log(1+torch.exp(self.lam*x))/self.lam
