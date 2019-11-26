import argparse


def get_classify_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=tuple, default=[256, 256], help='image size')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size')
    parser.add_argument('--epoch', type=int, default=60, help='epoch')

    parser.add_argument('--augmentation_flag', type=bool, default=True,
                        help='if true, use augmentation method in train set')
    parser.add_argument('--erase_prob', type=float, default=0.5,
                        help='probability of random erase when augmentation_flag is True')
    parser.add_argument('--gray_prob', type=float, default=0.2,
                        help='probability of gray when augmentation_flag is True')                                                 
    parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')
    parser.add_argument('--val_size', type=float, default=0.2, help='the ratio of val data when n_splits=1.')
    # model set 
    parser.add_argument('--model_type', type=str, default='se_resnext101_32x4d', help='resnet50/se_resnext101_32x4d')
    parser.add_argument('--last_stride', type=int, default=2, help='last stride in the resnet model')
    parser.add_argument('--droprate', type=float, default=0, help='dropout rate in classify module')

    # model hyper-parameters
    parser.add_argument('--num_classes', type=int, default=54)
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay in optimizer')
    # 学习率衰减策略
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='lr scheduler')
    parser.add_argument('--lr_step_size', type=int, default=20, help='step_size for StepLR scheduler')
    parser.add_argument('--restart_step', type=int, default=80, help='T_max for CosineAnnealingLR scheduler')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type')
    # 损失函数
    parser.add_argument('--loss_name', type=str, default='1.0*CrossEntropy',
                        help='Select the loss function, CrossEntropy/SmoothCrossEntropy')

    # 路径
    parser.add_argument('--save_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset_root', type=str, default='data/huawei_data/combine')

    config = parser.parse_args()

    return config
