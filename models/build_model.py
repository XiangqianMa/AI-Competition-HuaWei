import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torchvision.models import resnet50

from models.resnext import resnext50, resnext152, resnext18
from models.resnet import ClassificationResnet


class PrepareModel:
    """准备模型和优化器
    """
    def __init__(self):
        pass
    
    def create_model(self, model_type, classes_num, pretrained_path):
        """创建模型
        Args:
            model_type: 模型类型
            classes_num: 类别数目
            droprate: dropout层的概率
            stride: 用于ft_net的stride参数
            model_path: 当模型类型为strong系列时，需要指定默认权重的路径
            neck: 当模型类型为strong系列时，需要指定该参数
            neck_feat: 当模型类型为strong系列时，需要指定该参数
            pretrain_choice: 当模型类型为strong系列时，需要指定该参数
        """
        print('*'*10 + 'Making Model.' + '*'*10)
        if model_type == 'resnet50':
            model = ClassificationResnet(model_type, classes_num, 2, pretrained_path)

        return model
    
    def create_optimizer(self, model_type, model, config):
        """返回优化器

        Args:
            model_type: 模型类型
            model: 待优化的模型
            config: 配置
        Return:
            optimizer: 优化器
        """
        print('*'*10 + 'Making Optimizer.' + '*'*10)
        if config.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), config.lr, weight_decay=config.weight_decay)
            
        return optimizer

    def create_criterion(self, config, classes_num):
        """创建损失函数
        Args:
            criterion_type: 损失函数类型
        Return:
            criterion: 损失函数
        """
        print('*'*10 + 'Making Criterion.' + '*'*10)
        if config.criterion_type == 'CrossEntropy':
            print('Loss Type: %s.' % config.criterion_type)
            criterion = nn.CrossEntropyLoss()

        return criterion

    def create_lr_scheduler(
        self, 
        lr_scheduler_type, 
        optimizer, 
        step_size=None,
        epoch=None,
        ):
        """创建学习率衰减器
        Args:
            lr_scheduler_type: 衰减器类型
            optimizer: 优化器
            step_size: 使用StepLR时，必须指定该参数
        Return:
            my_lr_scheduler: 学习率衰减器
        """
        print('*'*10 + 'Making LR Scheduler.' + '*'*10)
        if lr_scheduler_type == 'StepLR':
            if not step_size:
                raise ValueError('You must specified step_size when you are using StepLR.')
            my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        elif lr_scheduler_type == 'CosineLR':
            if not epoch:
                raise ValueError('You must specified epoch when you are using CosineLR.')
            my_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epoch + 5)

        return my_lr_scheduler