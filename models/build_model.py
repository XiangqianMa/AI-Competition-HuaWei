import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from models.custom_model import CustomModel
from models.custom_attention_model import CustomLocalAttentionModel
from utils.radam import RAdam
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.torchtools.optim import RangerLars, Ranger
from utils.torchtools.lr_scheduler import DelayerScheduler, DelayedCosineAnnealingLR


def convert_layers(model, layer_type_old, layer_type_new, convert_weights=False, num_groups=None):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new, convert_weights)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_type_new(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 

            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias

            model._modules[name] = layer_new

    return model


class PrepareModel:
    """准备模型和优化器
    """

    def __init__(self):
        pass

    def create_model(self, model_type, classes_num, drop_rate=0, pretrained=True, bn_to_gn=False):
        """创建模型
        Args:
            model_type: str, 模型类型
            classes_num: int, 类别数目
            drop_rate: float, 分类层中的drop out系数
            pretrained: bool, 是否使用预训练模型
        """
        print('Creating model: {}'.format(model_type))
        model = CustomModel(model_type, classes_num, drop_rate=drop_rate, pretrained=pretrained)
        if bn_to_gn:
            convert_layers(model, torch.nn.BatchNorm2d, torch.nn.GroupNorm, True, num_groups=16)
        return model

    def create_local_attention_model(self, model_type, classes_num, last_stride=2, drop_rate=0,
                                     pretrained=True, use_local_attention=True):
        """创建模型
        Args:
            model_type: str, 模型类型
            classes_num: int, 类别数目
            last_stride: int, resnet最后一个下采样层的步长
            drop_rate: float, drop rate
            pretrained: bool, 是否使用预训练模型
            use_local_attention: bool, 是否使用局部attention机制
        """
        print('Creating model: {}'.format(model_type))
        model = CustomLocalAttentionModel(model_type, classes_num, last_stride, drop_rate, pretrained, use_local_attention)
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
        ignored_params = list(map(id, model.module.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params and p.requires_grad, model.module.parameters())
        print('Creating optimizer: %s' % config.optimizer)
        if config.optimizer == 'Adam':
            optimizer = optim.Adam(
                [
                    {'params': base_params, 'lr': 0.1 * config.lr},
                    {'params': model.module.classifier.parameters(), 'lr': config.lr}
                ], weight_decay=config.weight_decay)
        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(
                [
                    {'params': base_params, 'lr': 0.1 * config.lr},
                    {'params': model.module.classifier.parameters(), 'lr': config.lr}
                ], weight_decay=config.weight_decay, momentum=0.9)
        elif config.optimizer == 'RAdam':
            optimizer = RAdam(
                [
                    {'params': base_params, 'lr': 0.1 * config.lr},
                    {'params': model.module.classifier.parameters(), 'lr': config.lr}                    
                ], weight_decay=config.weight_decay
            )
        elif config.optimizer == 'RangerLars':
            optimizer = RangerLars(
                [
                    {'params': base_params, 'lr': 0.1 * config.lr},
                    {'params': model.module.classifier.parameters(), 'lr': config.lr}                    
                ], weight_decay=config.weight_decay                
            )
        elif config.optimizer == 'Ranger':
            optimizer = Ranger(
                [
                    {'params': base_params, 'lr': 0.1 * config.lr},
                    {'params': model.module.classifier.parameters(), 'lr': config.lr}                    
                ], weight_decay=config.weight_decay                 
            )

        return optimizer

    def create_lr_scheduler(
            self,
            lr_scheduler_type,
            optimizer,
            step_size=None,
            restart_step=None,
            multi_step=None,
            warmup=False,
            multiplier=None,
            warmup_epoch=None,
            delay_epoch=None
    ):
        """创建学习率衰减器
        Args:
            lr_scheduler_type: 衰减器类型
            optimizer: 优化器
            step_size: 使用StepLR时，必须指定该参数
            warmup: 是否使用warmup
            multiplier: 在warmup轮数结束后，学习率变为初始学习率的multiplier倍
            warmup_epoch: warmup的轮数
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
            # my_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
            my_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
        
        if warmup:
            if not warmup_epoch or not multiplier:
                raise ValueError('warup_epoch and multiplier must be specified when warmup is true.')
            my_lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmup_epoch, after_scheduler=my_lr_scheduler)
        elif delay_epoch:
            print('@ Lr delay epoch: %d' % delay_epoch)
            my_lr_scheduler = DelayerScheduler(optimizer, delay_epoch, my_lr_scheduler)

        return my_lr_scheduler

    def load_chekpoint(self, model, weight_path):
        print('Loading weight from %s.' % weight_path)
        weight = torch.load(weight_path)
        model.load_state_dict(weight['state_dict'])
        return model
