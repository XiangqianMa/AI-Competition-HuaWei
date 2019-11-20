import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.backbones.resnet import ResNet, Bottleneck


class ClassificationResnet(nn.Module):
    def __init__(self, model_type, classes_num, last_stride, pretrained_path):
        super(ClassificationResnet, self).__init__()
        self.resnet = None
        in_channel = 0
        if model_type == 'resnet50':
            self.resnet = ResNet(
                last_stride=last_stride,
                block=Bottleneck,
                layers=[3, 4, 6, 3]
            )
            in_channel = 2048

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, classes_num)
        )
        self.resnet.load_param(pretrained_path)

    def forward(self, x):
        output = self.resnet(x)
        output = self.gap(output)
        global_feature = output.view(output.shape[0], -1)
        output = self.fc(global_feature)
        return output

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
