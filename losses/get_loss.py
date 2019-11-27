import torch
import torch.nn as nn
from losses.CE_label_smooth import CrossEntropyLabelSmooth
from losses.focal_loss import MultiFocalLoss


class Loss(nn.Module):
    def __init__(self, model_name, loss_name, num_classes):
        """

        :param model_name: 模型的名称；类型为str
        :param loss_name: 损失的名称；类型为str
        :param num_classes: 网络的参数
        """
        super(Loss, self).__init__()
        self.model_name = model_name
        self.loss_name = loss_name
        self.loss_struct = []

        for loss in self.loss_name.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'SmoothCrossEntropy':
                loss_function = CrossEntropyLabelSmooth(num_classes=num_classes)
            elif loss_type == 'FocalLoss':
                loss_function = MultiFocalLoss(gamma=2)
            else:
                assert "loss: {} not support yet".format(self.loss_name)

            self.loss_struct.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        # 如果有多个损失函数，在加上一个求和操作
        if len(self.loss_struct) > 1:
            self.loss_struct.append({'type': 'Total', 'weight': 0, 'function': None})

        self.loss_module = nn.ModuleList([l['function'] for l in self.loss_struct if l['function'] is not None])

        # self.log的维度为[1, len(self.loss)]，前面几个分别存放某次迭代各个损失函数的损失值，最后一个存放某次迭代损失值之和
        self.log, self.log_sum = torch.zeros(len(self.loss_struct)), torch.zeros(len(self.loss_struct))

        if torch.cuda.is_available():
            self.loss_module = torch.nn.DataParallel(self.loss_module)
            self.loss_module.cuda()

    def forward(self, outputs, labels):
        """

        :param outputs: 网络的输出，具体维度和网络有关
        :param labels: 数据的真实类标，具体维度和网络有关
        :return loss_sum: 损失函数之和，未经过item()函数，可用于反向传播
        """
        losses = []
        # 计算每一个损失函数的损失值
        for i, l in enumerate(self.loss_struct):
            if l['type'] in ['CrossEntropy', 'SmoothCrossEntropy', 'FocalLoss']:
                loss = l['function'](outputs, labels)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[i] = effective_loss.item()
                self.log_sum[i] += self.log[i]

            # 保留接口
            else:
                raise ValueError('%s is not implemented yet.' % l['type'])

        loss_sum = sum(losses)
        if len(self.loss_struct) > 1:
            self.log[-1] = loss_sum.item()
            self.log_sum[-1] += loss_sum.item()

        return loss_sum

    def record_loss_iteration(self, writer_function=None, global_step=None, type=''):
        """ 用于记录每一次迭代的结果

        :param writer_function: tensorboard的写入函数；类型为callable
        :param global_step: 当前的步数；类型为int
        :param type: 训练模型或者验证模式；类型为str
        :return: [损失名称: 损失值][损失名称: 损失值][损失名称: 损失值]；类型为str
        """
        descript = []
        for l, each_loss in zip(self.loss_struct, self.log):
            if writer_function:
                writer_function(l['type'] + type + 'Iteration', each_loss, global_step)
            descript.append('[{}: {:.4f}]'.format(l['type'], each_loss))
        return ''.join(descript)

    def record_loss_epoch(self, num_iterations, writer_function=None, global_step=None, type=''):
        """ 用于记录每一个epoch的结果

        :param num_iterations：该epoch包含多少个迭代；类型为int
        :param writer_function: tensorboard的写入函数；类型为callable
        :param global_step: 当前的步数；类型为int
        :param type: 训练模型或者验证模式；类型为str
        :return: [Average 损失名称: 平均损失值][Average 损失名称: 平均损失值][Average 损失名称: 平均损失值]；类型为str
        """
        descript = []
        for l, each_loss in zip(self.loss_struct, self.log_sum):
            if writer_function:
                writer_function(l['type'] + type + 'Epoch', each_loss/num_iterations, global_step)
            descript.append('[Average {}: {:.4f}]'.format(l['type'], each_loss/num_iterations))

        # 注意要把 self.log_sum清零
        self.log_sum = torch.zeros(len(self.loss_struct))
        return ''.join(descript)

