import torch
from torch import nn


class Sparsity:
    """向模型的BN层添加稀疏惩罚
    """
    def __init__(self, model, sparsity_scale=1e-4, penalty_type='L1'):
        self.model = model
        self.sparsity_scale = sparsity_scale
        self.penalty_type = penalty_type

        print('penality_type: %s, sparsity_scale: %.5f' % (self.penalty_type, self.sparsity_scale))

    def updateBN(self):
        if self.penalty_type == 'L1':
            self.updateBN_L1()

    def updateBN_L1(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(self.sparsity_scale * torch.sign(m.weight.data))  # L1