import torch
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable


class MultiFocalLoss(nn.Module):
    """多分类focal loss
    """
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """
        Args:
            input: 模型的输入，取softmax后，表示对应样本属于各类的概率
            target: 真实类标
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        # 沿给定轴dim，将输入索引张量index指定位置的值进行聚合。即取出真实类标对应的预测概率，logpt维度为[batch]
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # exp()对数据取指数，因为上面使用了log_softmax函数
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            # 沿给定轴dim，将输入索引张量index指定位置的值进行聚合。即取出真实类标对应的alpha，at维度为[batch]
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()