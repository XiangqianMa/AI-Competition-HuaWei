import torch
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import models.resnext as resnext


class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes, drop_rate=0, pretrained=True):
        """
        Args:
            model_name: model_name: resnet模型的名称；类型为str
            num_classes: num_classes: 类别数目；类型为int
            drop_rate: float, 分类层中的drop out系数
            pretrained: bool, 是否使用预训练权重
        """
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if self.model_name.startswith('efficientnet'):
            if pretrained:
                model = EfficientNet.from_pretrained(self.model_name, num_classes=self.num_classes)
            else:
                model = EfficientNet.from_name(self.model_name, override_params={'num_classes': self.num_classes})
            # 声明特征提取层，池化层与全连接层
            self.feature_layer = model
            in_features = model._conv_head.out_channels

        elif self.model_name in ['resnext101_32x8d_wsl', 'resnext101_32x16d_wsl', 'resnext101_32x32d_wsl', 'resnext101_32x48d_wsl']:
            model = getattr(resnext, self.model_name)(self.num_classes, pretrained=pretrained)
            self.feature_layer = nn.Sequential(*list(model.children())[:-2])
            in_features = model.fc.in_features

        else:
            if pretrained:
                pretrained_type = 'imagenet'
            else:
                pretrained_type = None
            model = getattr(pretrainedmodels, self.model_name)(pretrained=pretrained_type)
            if hasattr(model, 'avgpool') or hasattr(model, 'avg_pool'):
                self.feature_layer = nn.Sequential(*list(model.children())[:-2])
            else:
                self.feature_layer = nn.Sequential(*list(model.children())[:-1])
            if hasattr(model.last_linear, 'in_features'):
                in_features = model.last_linear.in_features
            elif hasattr(model.last_linear, 'in_channels'):
                in_features = model.last_linear.in_channels
            else:
                assert NotImplementedError

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        add_block = [nn.Linear(in_features, 1024), nn.ReLU()]
        if drop_rate > 0:
            add_block += [nn.Dropout(p=drop_rate)]
        add_block += [nn.Linear(1024, self.num_classes)]
        self.classifier = nn.Sequential(*add_block)

    def forward(self, x):
        """

        Args:
            x: 网络的输入；类型为tensor；维度为[batch_size, channel, height, width]

        Returns: 网络预测的类别；类型为tensor；维度为[batch_size, num_classes]
        """
        # 特征提取部分
        if self.model_name.startswith('efficientnet'):
            global_features = self.feature_layer.extract_features(x)
        else:
            global_features = self.feature_layer(x)
        # 经过全局平均池化与线性分类层
        global_features = self.pool(global_features)
        global_features = global_features.view(global_features.shape[0], -1)
        scores = self.classifier(global_features)
        return scores

    def get_classify_result(self, outputs, labels, device):
        """

        Args:
            outputs: 网络的预测标签，维度为[batch_size, num_classes]
            labels: 真实标签，维度为[batch_size, num_classes]
            device: 当前设备

        Returns: 预测对了多少个样本

        """
        outputs = F.softmax(outputs, dim=1)
        return (outputs.max(1)[1] == labels.to(device)).float()


if __name__ == '__main__':
    inputs = torch.rand((12, 3, 256, 256))
    custom_model = CustomModel('efficientnet-b5', num_classes=54, pretrained=True)
    print(custom_model)
    if torch.cuda.is_available():
        custom_model = torch.nn.DataParallel(custom_model)
        custom_model = custom_model.cuda()
    scores = custom_model(inputs.cuda())
    print(scores.size())
