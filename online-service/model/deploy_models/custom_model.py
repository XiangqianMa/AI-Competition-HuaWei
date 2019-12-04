import torch
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name: model_name: resnet模型的名称；类型为str
            num_classes: num_classes: 类别数目；类型为int
        """
        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if self.model_name.startswith('efficientnet'):
            if pretrained:
                self.model = EfficientNet.from_pretrained(self.model_name, num_classes=self.num_classes)
            else:
                self.model = EfficientNet.from_name(self.model_name, override_params={'num_classes': self.num_classes})
        else:
            if pretrained:
                pretrained_type = 'imagenet'
            else:
                pretrained_type = None
            self.model = getattr(pretrainedmodels, self.model_name)(pretrained=pretrained_type)
            dim_feats = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(dim_feats, self.num_classes)

    def forward(self, x):
        """

        Args:
            x: 网络的输入；类型为tensor；维度为[batch_size, channel, height, width]

        Returns: 网络预测的类别；类型为tensor；维度为[batch_size, num_classes]
        """
        scores = self.model(x)
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
    custom_model = CustomModel('efficientnet-b5', num_classes=54, pretrained=False)
    scores = custom_model(inputs)
    print(scores.size())
