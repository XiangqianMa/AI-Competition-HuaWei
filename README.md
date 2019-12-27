# 华为云人工智能创新应用大赛

![](README-Template/image)

## 简介

本仓库存放参加华为云人工智能创新应用大赛时编写的代码，该比赛的赛题如下：

本赛题任务是对西安的热门景点、美食、特产、民俗、工艺品等图片进行分类，即首先识别出图片中物品的类别（比如大雁塔、肉夹馍等），然后根据图片分类的规则，输出该图片中物品属于景点、美食、特产、民俗和工艺品中的哪一种。

比赛期间尝试了很多策略，最终分数为0.979，名次为32 / 732名。虽然最终未取得很高的名次，但对分类问题有了更深的理解，感谢华为云组织的比赛。

本仓库适用于解决各种分类问题，支持多种训练技巧，具备较高的可扩展性。

## 开始

该部分主要介绍如何将本项目拷贝并运行在本地机器上。

### 准备工作

在使用本项目之前，需要满足以下基础运行环境要求：

* Python3.7
* Pytorch 1.3.0
* torchvision 0.4.0
* pretrainedmodels
* efficientnet-pytorch
* albumentations

如未安装以上Python package，请直接在GitHub首页搜索对应名称，按照相应仓库的安装说明进行安装即可。


### 安装

#### 下载工程

首先，从GitHub上下载本仓库：

```shell
git clone https://github.com/XiangqianMa/AI-Competition-HuaWei.git
```

然后，切换到工程主目录下：

```shell
cd AI-Competition-HuaWei
```

#### 准备数据集

请从华为云比赛[官网](https://competition.huaweicloud.com/information/1000021526/introduction)下载参赛用训练集，或使用任意自定义的分类数据集。

* 将官方数据集解压到任意目录后，文件夹中包含的文件如下所示：

  ```python
  train_data
  label_id_name.json
  ```

* 接着，在工程目录下创建数据集的软链接

  ```shell
  cd AI-Competition-HuaWei
  mkdir data
  cd data
  ln -s your_data_path huawei_data
  ```

* 要注意的是，数据集的组织形式应满足以下条件：

  1. 图片和标注文件存放在同一文件夹下，如下图所示:

     ![](README-Template/Screenshot from 2019-12-27 10-33-22.png)

  2. 标注文件为`txt`格式，其内容如下：
  
     ```python
     img_1.jpg, 0
     ```

如果是自己的数据集，请处理为和上述格式一致，否则可能导致程序运行出错（与数据集解析程序有关）。
## 运行

在以上准备工作完成后，请首先切换到工程主目录下。

### 训练

首先，创建权重保存文件夹。

```shell
mkdir checkpoints
```

在不修改任何配置的情况下，可直接运行训练程序。

```shell
python train_classifier.py
```

默认训练的配置请参考`config.py`文件。

### 测试

首先，请手动创建如下文件夹。

```shell
mkdir data/demo_data
```

训练完成后，请执行下述语句得到可视化的分类结果，测试将在验证集上进行。

```shell
python demo.py
```

测试完成后，**预测错误的样本**将被存放到`data/demo_data/results`文件夹下。测试示例如下：

<img src="README-Template/img_2625.jpg" style="zoom:67%;" />

#### 线上部署

如需线上部署，请参考`online-service`文件夹。请将你的依赖库放在该文件夹下的`model`文件夹中，`model`的组织形式如下：

![](README-Template/model结构.png)

注意，如果需要在`customize_service.py`导入自定义的模块，请使用如下导入模式：

```python
from model import xxx
```

同时，如果需要在华为云的ModelArts中在线安装依赖库，请修改`config.json`文件如下：

```python
"dependencies": [
    {
        "installer": "pip",
        "packages": [
            {
                "package_name": "Pillow",
                "package_version": "5.0.0",
                "restraint": "EXACT"
            },
            {
                "package_name": "torchvision",
                "package_version": "0.2.1",
                "restraint": "EXACT"
            },
            {
                "package_name": "tqdm"
            }                
        ]
    }
]
```

然后，将上述文件夹上传到个人的obs桶中，按照官网指引运行测试用例即可。

## 支持的功能

### 多网络结构

目前，本工程支持多种类型的卷积神经网络结构，主要有以下几种：

* efficientnet系列
* resnext*_wsl系列
* resnet系列
* se_resnet系列

具体支持的模型结构，请参考`models/custom_model.py`文件，后续如有其他网络结构需要扩展，将进一步更新。

### 丰富的训练策略

在本次比赛中，尝试了很多种训练策略，罗列如下（以下策略都可以通过修改`config.py`文件完成）：

#### 数据增强

* AutoAugment（谷歌出品）
* CutMix（一个很不错的数据增强方法，优于cut_out、random_erase等）
* 灰度化
* 左右、垂直翻转
* 旋转、平移等

#### 多尺度训练

本项目支持多尺度训练，可以自定义训练时使用的一组图像尺度大小，具体请参考`config.py`中的多尺度训练部分。

#### 正则化策略

* 稀疏训练
* `l_1`正则化
* 权重衰减
* DropOut

#### 学习率衰减

* StepLR
* CosineLR
* MultiStepLR
* ReduceLR
* warmup
* 衰减前保持学习率一定epoch不变

#### 优化器

* Adam
* SGD
* RAdam
* RangerLars
* Ranger

#### 损失函数

* SmoothCrossEntropy
* CrossEntropy
* FocalLoss
* SmoothCrossEntropyHardMining

可以使用表达式对上述损失函数进行自由加权组合，例如`0.7\*SmoothCrossEntropy+0.3\*CrossEntropy`。

> **注意：**SmoothCrossEntropyHardMining为个人依据自身理解编写，如有错误欢迎指出。

#### 数据集划分

本项目支持交叉验证和随机数据集划分，具体参数设置可参考`config.py`中的相应设置。

### 数据集扩充

本项目支持从网络上下载图片，并完成自动扩充功能。完成这一内容的代码主要存放在expand_images文件夹下。

首先，使用`bing.py`或`baidu.py`从网络上爬去原始图片；接着，使用`clean_download_image.py`对原始图片中的损坏文件进行清洗；然后，使用`predict_download_image.py`对清洗过后的图片进行类别预测，预测时可以依据验证集上各类的准确率线性设置阈值；最后，使用`combine_dataset_dynamic.py`将伪标签图片动态拷贝（依据各类的验证准确率动态计算）至指定目录下。

在实验过程中，上述策略没有造成很大的性能提升，如有改进方法，欢迎提修改意见。

## 参考

* [AutoAugment](https://github.com/DeepVoltaire/AutoAugment)
* [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch)
* [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr)

## 贡献者

* [XiangqianMa](https://github.com/XiangqianMa)
* [Zdaiot](https://github.com/zdaiot)

> 后续如有改进，将持续更新。