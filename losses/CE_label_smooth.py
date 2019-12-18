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


class MultiLabelCrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_parents_classes, num_children_classes, children_predicts_index, epsilon=0.1, use_gpu=True):
        super(MultiLabelCrossEntropyLabelSmooth, self).__init__()
        self.num_parents_classes = num_parents_classes
        self.num_children_classes = num_children_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        # 父类的损失函数
        self.parent_cels = CrossEntropyLabelSmooth(self.num_parents_classes, self.epsilon, self.use_gpu)
        self.children_cels = []
        self.children_predicts_index = children_predicts_index
        # 各个子类的损失函数
        for num in self.num_children_classes:
            self.children_cels.append(CrossEntropyLabelSmooth(num, self.epsilon, self.use_gpu))

    def forward(self, predicts, parent_targets, child_targets):
        # 父类的损失
        parent_predicts = predicts[:, 0:self.num_parents_classes]
        parent_loss = self.parent_cels(parent_predicts, parent_targets)
        # 依据真实父类标计算子类标的损失
        children_loss = 0
        for batch_index, parent_target in enumerate(parent_targets):
            start_index = self.children_predicts_index[parent_target][0] + self.num_parents_classes
            end_index = self.children_predicts_index[parent_target][1] + self.num_parents_classes
            child_predict = predicts[batch_index, start_index:end_index].unsqueeze(0)
            child_target = child_targets[batch_index].unsqueeze(0)
            children_loss += self.children_cels[parent_target](child_predict, child_target)
        children_loss /= parent_targets.size(0)
        loss = parent_loss + children_loss
        return loss
