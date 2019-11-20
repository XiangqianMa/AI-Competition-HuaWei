import torch
import torch.nn as nn
import pretrainedmodels
from torchvision import models
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes):
        """

        Args:
            model_name: model_name: resnet模型的名称；类型为str
            num_classes: num_classes: 类别数目；类型为int
        """
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if self.model_name.startswith('resnet'):
            model = getattr(models, self.model_name)(pretrained=True)
            in_features = model.fc.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])

        elif self.model_name.startswith('dpn'):
            model = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            in_features = model.last_linear.in_channels
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_layer.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        elif self.model_name.startswith('densenet'):
            model = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            in_features = model.last_linear.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])
            self.feature_layer.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        else:
            model = getattr(pretrainedmodels, self.model_name)(pretrained='imagenet')
            model.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            in_features = model.last_linear.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):
        """

        Args:
            x: 网络的输入；类型为tensor；维度为[batch_size, channel, height, width]

        Returns: 网络预测的类别；类型为tensor；维度为[batch_size, num_classes]
        """
        global_features = self.feature_layer(x)
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
    inputs = torch.rand((64, 3, 224, 224))
    custom_model = CustomModel('densenet201', num_classes=40)
    scores = custom_model(inputs)
    print(scores.size())
