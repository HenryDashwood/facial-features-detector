from torch import nn
from torch.nn.functional import mse_loss
from fastai.vision import Tensor

class MSELossFlat(nn.MSELoss):
    def forward(self, input:Tensor, target:Tensor):
        return super().forward(input.view(-1), target.view(-1))