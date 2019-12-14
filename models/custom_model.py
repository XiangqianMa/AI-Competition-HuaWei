import torch
import torch.nn as nn
import pretrainedmodels
from torchvision import models
import torch.nn.functional as F


class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes, drop_rate=0, pretrained=True):
        """
        Args:
            model_name: model_name: resnet模型的名称；类型为str
            droprate: float, drop rate
            num_classes: num_classes: 类别数目；类型为int
        """
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if pretrained:
            pretrained_type = 'imagenet'
        else:
            pretrained_type = None
        model = getattr(pretrainedmodels, self.model_name)(pretrained=pretrained_type)
        self.feature_layer1 = torch.nn.Sequential(*list(model.children())[:-3])
        self.feature_layer2 = torch.nn.Sequential(*list(model.children())[-3:-2])
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        in_features = 1024*4 + 2048
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
        x = self.feature_layer1(x)
        feature2 = self.feature_layer2(x)
        feature1 = x.view(feature2.size(0), -1, feature2.size(2), feature2.size(3))
        feature = torch.cat([feature1, feature2], dim=1)
        feature = self.avg_pool(feature)
        global_features = feature.view(feature.shape[0], -1)
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
    inputs = torch.rand((2, 3, 224, 224)).cuda()
    custom_model = CustomModel('se_resnext101_32x4d', num_classes=54).cuda()
    scores = custom_model(inputs)
    print(scores.size())
