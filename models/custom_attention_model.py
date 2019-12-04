import torch
import torch.nn as nn
import pretrainedmodels
from torchvision import models
import torch.nn.functional as F


class SpatialAttention2d(nn.Module):
    """
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    """

    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 512, 1, 1)  # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 1, 1, 1)  # 1x1 conv
        self.softplus = nn.Softplus(beta=1, threshold=20)  # use default setting.

    def forward(self, x):
        """
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()

    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3), \
            'err: h, w of tensors x({}) and weights({}) must be the same.' \
                .format(x.size, weights.size)
        y = x * weights  # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))  # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1

    def __repr__(self):
        return self.__class__.__name__


class CustomLocalAttentionModel(nn.Module):
    def __init__(self, model_name, num_classes, last_stride=2, droprate=0, pretrained=True, use_local_attention=False):
        """
        Args:
            model_name: model_name: resnet模型的名称；类型为str
            last_stride: resnet模型最后一个下采样层的步长；类型为int
            droprate: float, drop rate
            num_classes: num_classes: 类别数目；类型为int
            use_local_attention, bool, 是否使用局部注意力机制
        """
        super(CustomLocalAttentionModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.last_stride = last_stride
        self.use_local_attention = use_local_attention

        if self.model_name.startswith('resnet'):
            model = getattr(models, self.model_name)(pretrained=pretrained)
            if self.model_name == 'resnet18' or self.model_name == 'resnet34':
                model.layer4[0].conv1.stride = (self.last_stride, self.last_stride)
            else:
                model.layer4[0].conv2.stride = (self.last_stride, self.last_stride)
            model.layer4[0].downsample[0].stride = (self.last_stride, self.last_stride)
            in_features = model.fc.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-2])

        elif self.model_name.startswith('dpn'):
            if pretrained:
                pretrained_type = 'imagenet'
            else:
                pretrained_type = None
            model = getattr(pretrainedmodels, self.model_name)(pretrained=pretrained_type)
            in_features = model.last_linear.in_channels
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])

        elif self.model_name.startswith('densenet'):
            if pretrained:
                pretrained_type = 'imagenet'
            else:
                pretrained_type = None
            model = getattr(pretrainedmodels, self.model_name)(pretrained=pretrained_type)
            in_features = model.last_linear.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-1])

        else:
            if pretrained:
                pretrained_type = 'imagenet'
            else:
                pretrained_type = None
            model = getattr(pretrainedmodels, self.model_name)(pretrained=pretrained_type)
            # # 替换前面的7x7卷积层+MaxPool2d层为两层3x3的卷积层
            # model.layer0 = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            #                                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
            #                                               track_running_stats=True),
            #                                nn.ReLU(inplace=True),
            #                                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2),
            #                                          padding=(1, 1)),
            #                                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
            #                                               track_running_stats=True),
            #                                nn.ReLU(inplace=True),
            #                                ])
            in_features = model.last_linear.in_features
            self.feature_layer = torch.nn.Sequential(*list(model.children())[:-2])

        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # 若使用局部注意力机制
        if self.use_local_attention:
            self.attention = SpatialAttention2d(in_c=in_features, act_fn='relu')
            self.weights_sum = WeightedSum2d()

        add_block = [nn.Linear(in_features, 1024), nn.ReLU()]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block += [nn.Linear(1024, self.num_classes)]
        self.classifier = nn.Sequential(*add_block)

    def forward(self, x):
        """

        Args:
            x: 网络的输入；类型为tensor；维度为[batch_size, channel, height, width]

        Returns: 网络预测的类别；类型为tensor；维度为[batch_size, num_classes]
        """
        # 特征提取部分
        global_features = self.feature_layer(x)

        # 若使用局部注意力机制
        if self.use_local_attention:
            attn_x = F.normalize(global_features, p=2, dim=1)
            attn_score = self.attention(global_features)
            global_features = self.weights_sum([attn_x, attn_score])

        # 经过全局平均池化与线性分类层
        global_features = self.avg_pool(global_features)
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
    custom_model = CustomLocalAttentionModel('resnet50', num_classes=54, pretrained=False, use_local_attention=True)
    scores = custom_model(inputs)
    print(scores.size())
