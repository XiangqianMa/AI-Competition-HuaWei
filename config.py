import json
import argparse
from argparse import Namespace


def get_classify_config():
    use_paras = False
    if use_paras:
        with open('./checkpoints/unet_resnet34/' + "params.json", 'r', encoding='utf-8') as json_file:
            config = json.load(json_file)
        # dict to namespace
        config = Namespace(**config)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_size', type=tuple, default=[224, 224], help='image size')
        parser.add_argument('--batch_size', type=int, default=24, help='batch size')
        parser.add_argument('--epoch', type=int, default=50, help='epoch')

        parser.add_argument('--augmentation_flag', type=bool, default=True, help='if true, use augmentation method in train set')
        parser.add_argument('--n_splits', type=int, default=5, help='n_splits_fold')
        parser.add_argument('--crop', type=bool, default=False, help='if true, crop image to [height, width].')
        parser.add_argument('--height', type=int, default=None, help='the height of cropped image')
        parser.add_argument('--width', type=int, default=None, help='the width of cropped image')

        # model set 
        parser.add_argument('--model_type', type=str, default='resnet50', \
            help='unet_resnet34/unet_se_resnext50_32x4d/unet_efficientnet_b4/unet_resnet50/unet_efficientnet_b4')
        parser.add_argument('--pretrained_path', type=str, default='/home/mxq/.cache/torch/checkpoints/resnet50-19c8e357.pth', 
            help='the path of model pretrained weight path.')
        # model hyper-parameters
        parser.add_argument('--classes_num', type=int, default=54)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=5e-5, help='init lr')
        parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay in optimizer')
        parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='lr scheduler')
        parser.add_argument('--lr_step_size', type=str, default=40, help='lr scheduler step')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type')
        parser.add_argument('--criterion_type', type=str, default='CrossEntropy', help='optimizer type')

        # dataset 
        parser.add_argument('--save_path', type=str, default='./checkpoints')
        parser.add_argument('--dataset_root', type=str, default='./datasets/Steel_data')

        config = parser.parse_args()
        # config = {k: v for k, v in args._get_kwargs()}

    return config
