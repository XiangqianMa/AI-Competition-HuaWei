import json
import argparse
from argparse import Namespace


def get_classify_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=tuple, default=[224, 224], help='image size')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')

    parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method in train set')
    parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')

    # model set 
    parser.add_argument('--model_type', type=str, default='resnet50', help='resnet50')
    parser.add_argument('--pretrained_path', type=str, default='/home/mxq/.cache/torch/checkpoints/resnet50-19c8e357.pth', 
        help='the path of model pretrained weight path.')
    
    # model hyper-parameters
    parser.add_argument('--classes_num', type=int, default=54)
    parser.add_argument('--lr', type=float, default=5e-5, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
    # 学习率衰减策略
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='lr scheduler')
    parser.add_argument('--lr_step_size', type=str, default=40, help='lr scheduler step')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
    # 损失函数
    parser.add_argument('--criterion_type', type=str, default='CrossEntropy', help='optimizer type')

    # 路径
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset_root', type=str, default='./datasets/Steel_data')

    config = parser.parse_args()

    return config
