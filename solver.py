'''
该文件的功能：实现模型的前向传播，反向传播，损失函数计算，保存模型，加载模型功能
'''
import numpy as np
import torch
import shutil
import os


class Solver:
    def __init__(self, model, device):
        ''' 完成solver类的初始化
        Args:
            model: 网络模型
            device: 设备
        '''
        self.model = model
        self.device = device

    def forward(self, images):
        ''' 实现网络的前向传播功能
        
        Args:
            images: [batch_size, channel, height, width]
            
        Return:
            output: 网络的输出，具体维度和含义与self.model有关，对我们任务而言：
                若self.model为分割模型，则维度为[batch_size, class_num, height, width]，One-hot数据
                若self.model为分类模型，则维度为[batch_size, class_num]，One-hot数据
        '''
        images = images.to(self.device)
        outputs = self.model(images)
        return outputs

    def cal_loss(self, predicts, parent_targets, children_targets, criterion):
        ''' 根据真实类标和预测出的类标计算损失
        
        Args:
            predicts: 网络的预测输出，具体维度和self.model有关，对我们任务而言：
                若为分类模型，则维度为[batch_size, class_num, 1, 1]，预测出的数据
            targets: 真实类标，具体维度和self.model有关，对我们任务而言：
                若为分类模型，则维度为[batch_size, class_num]，真实类标，One-hot数据

            criterion: 使用的损失函数
        Return:
            loss: 计算出的损失值
        '''
        parent_targets = parent_targets.to(self.device)
        children_targets = children_targets.to(self.device)
        return criterion(predicts, parent_targets, children_targets)

    def cal_loss_cutmix(self, predicts, parent_targets_a, parent_targets_b, children_target_a, children_target_b, lam, criterion):
        """计算使用cutmix时的损失

        Args:
            predicts: 网络的预测输出
            targets_a: 类标a
            targets_b: 类标b
            lam: lambda参数
            criterion: 损失函数
        Return:
            loss: 计算出的损失值        
        """
        parent_targets_a = parent_targets_a.to(self.device)
        parent_targets_b = parent_targets_b.to(self.device)
        children_targets_a = children_target_a.to(self.device)
        children_targets_b = children_target_b.to(self.device)
        return criterion(predicts, parent_targets_a, children_targets_a) * lam + criterion(predicts, parent_targets_b, children_targets_b) * (1. - lam)

    def backword(self, optimizer, loss, sparsity=None):
        ''' 实现网络的反向传播
        
        Args:
            optimizer: 模型使用的优化器
            loss: 模型计算出的loss值
        Return:
            None
        '''
        loss.backward()
        # 稀疏度训练
        if sparsity:
            sparsity.updateBN()
        optimizer.step()
        optimizer.zero_grad()

    def save_checkpoint(self, save_path, state, is_best):
        ''' 保存模型参数
        Args:
            save_path: 要保存的权重路径
            state: 存有模型参数、最大dice等信息的字典
            is_best: 是否为最优模型
        Return:
            None
        '''
        torch.save(state, save_path)
        if is_best:
            print('Saving Best Model.')
            save_best_path = '/'.join(save_path.split('/')[:-1] + ['model_best.pth'])
            shutil.copyfile(save_path, save_best_path)
    
    def load_checkpoint(self, load_path):
        ''' 保存模型参数
        Args:
            load_path: 要加载的权重路径
        
        Return:
            加载过权重的模型
        '''
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            self.model.module.load_state_dict(checkpoint['state_dict'])
            print('Successfully Loaded from %s' % (load_path))
            return self.model
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(load_path))