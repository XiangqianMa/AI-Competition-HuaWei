import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: q_i = (1 - epsilon) * a_i + epsilon / N.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        '''
        scatter_第一个参数为1表示分别对每行填充；targets.unsqueeze(1)得到的维度为[num_classes, 1]；
        填充方法为：取出targets的第i行中的第一个元素（每行只有一个元素），记该值为j；则前面tensor中的(i,j)元素填充1；
        最终targets的维度为[batch_size, num_classes]，每一行代表一个样本，若该样本类别为j，则只有第j元素为1，其余元素为0
        '''
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # mean(0)表示缩减第0维，也就是按列求均值，得到维度为[num_classes]，得到该batch内每一个类别的损失，再求和
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CrossEntropyLabelSmoothHardMining(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: q_i = (1 - epsilon) * a_i + epsilon / N.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, ratio=0.6, use_gpu=True):
        super(CrossEntropyLabelSmoothHardMining, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.ratio = ratio

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        '''
        scatter_第一个参数为1表示分别对每行填充；targets.unsqueeze(1)得到的维度为[num_classes, 1]；
        填充方法为：取出targets的第i行中的第一个元素（每行只有一个元素），记该值为j；则前面tensor中的(i,j)元素填充1；
        最终targets的维度为[batch_size, num_classes]，每一行代表一个样本，若该样本类别为j，则只有第j元素为1，其余元素为0
        '''
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # mean(0)表示缩减第0维，也就是按列求均值，得到维度为[num_classes]，得到该batch内每一个类别的损失，再求和
        loss = (- targets * log_probs).mean(1)
        selected_number = int(inputs.size(0) * self.ratio)
        loss = torch.sort(loss, descending=True)[0][:selected_number].mean()
        return loss


if __name__ == '__main__':
    criterion = CrossEntropyLabelSmoothHardMining(54)
    input = torch.Tensor(5, 54).cuda()
    target = torch.ones(5).long()
    loss = criterion(input, target)
