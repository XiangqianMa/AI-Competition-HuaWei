import torch.optim as optim
from torch.optim import lr_scheduler
from models.custom_model import CustomModel


class PrepareModel:
    """准备模型和优化器
    """

    def __init__(self):
        pass

    def create_model(self, model_type, classes_num, last_stride, droprate, pretrained=True):
        """创建模型
        Args:
            model_type: 模型类型
            last_stride: resnet最后一个下采样层的步长；类型为int
            droprate: float, drop rate
            classes_num: 类别数目
        """
        print('Creating model: {}'.format(model_type))
        model = CustomModel(model_type, classes_num, last_stride, droprate, pretrained=pretrained)
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
        print('Creating optimizer: %s' % config.optimizer)
        if config.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), config.lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), config.lr, weight_decay=config.weight_decay, momentum=0.9)

        return optimizer

    def create_lr_scheduler(
            self,
            lr_scheduler_type,
            optimizer,
            step_size=None,
            restart_step=None,
            multi_step=None
    ):
        """创建学习率衰减器
        Args:
            lr_scheduler_type: 衰减器类型
            optimizer: 优化器
            step_size: 使用StepLR时，必须指定该参数
        Return:
            my_lr_scheduler: 学习率衰减器
        """
        print('Creating lr scheduler: %s' % lr_scheduler_type)
        if lr_scheduler_type == 'StepLR':
            if not step_size:
                raise ValueError('You must specified step_size when you are using StepLR.')
            my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        elif lr_scheduler_type == 'CosineLR':
            if not restart_step:
                raise ValueError('You must specified restart_step when you are using CosineLR.')
            my_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, restart_step)
        elif lr_scheduler_type == 'MultiStepLR':
            if not multi_step:
                raise ValueError('You must specified multi step when you are using MultiStepLR.')
            my_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, multi_step)            
        elif lr_scheduler_type == 'ReduceLR':
            my_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        return my_lr_scheduler
