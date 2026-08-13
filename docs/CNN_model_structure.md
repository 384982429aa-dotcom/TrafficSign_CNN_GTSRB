# CNN 模型结构设计：残差跳转连接版

本项目使用带跳转连接的 Residual CNN 对 GTSRB 交通标志数据集进行 43 类分类。模型输入为 48x48x3 的 RGB 彩色交通标志图片，输出为 43 个类别的概率分布。

## 为什么使用跳转连接

普通 CNN 是一层接一层顺序传递特征，网络加深后容易出现梯度衰减、训练困难的问题。跳转连接会把前面层的特征直接传到后面，与卷积后的特征相加，使模型既能学习新的特征变化，也能保留原始特征信息。

残差结构可以表示为：

```text
输出 = 卷积变换 F(x) + 原始输入 x
```

这样网络学习的是残差 F(x)，训练会更稳定，也更适合设计较深的 CNN。

## 网络结构

```text
Input: 48 x 48 x 3
    ↓
Stem Conv2D(32, 3x3) + BatchNormalization + ReLU
    ↓
Residual Block 1: 32 channels
Conv2D(32, 3x3) + BN + ReLU
Conv2D(32, 3x3) + BN
Shortcut Add + ReLU
    ↓
Residual Block 2: 64 channels, downsample
Conv2D(64, 3x3, stride=2) + BN + ReLU
Conv2D(64, 3x3) + BN
1x1 Shortcut Conv
Shortcut Add + ReLU
    ↓
Residual Block 3: 64 channels
    ↓
Residual Block 4: 128 channels, downsample
    ↓
Residual Block 5: 128 channels
    ↓
Residual Block 6: 256 channels, downsample
    ↓
GlobalAveragePooling2D
Dense(256, ReLU) + BatchNormalization + Dropout(0.50)
Dense(43, Softmax)
```

## 层数统计

- Stem 卷积层：1 个
- 残差块：6 个
- 每个残差块主分支包含 2 个 3x3 卷积层
- 主分支卷积层：12 个
- Shortcut 1x1 卷积层：3 个，用于尺寸或通道数变化时对齐
- 总卷积层：16 个
- 输出层：1 个 Softmax 分类层

## 设计说明

- 前端 Stem 卷积先提取基础边缘和颜色特征。
- 残差块通过跳转连接保留浅层信息，缓解深层网络训练困难。
- stride=2 的残差块用于降低特征图尺寸，同时增加通道数。
- 1x1 Shortcut 卷积用于在通道数变化时保证 Add 操作维度一致。
- GlobalAveragePooling2D 减少参数量，比直接 Flatten 更不容易过拟合。
- Dropout 和 L2 正则化用于提高泛化能力。
- Dense(43, Softmax) 对应 GTSRB 的 43 类交通标志。

## 答辩表述

我最终采用的是带跳转连接的残差 CNN。普通 CNN 随着层数加深可能出现梯度衰减和训练困难，所以我在卷积模块中加入 Shortcut 结构，把输入特征直接与卷积后的特征相加。这样模型学习的是残差映射，既能保留浅层边缘、颜色等信息，又能在深层学习更抽象的交通标志语义特征。该网络共有 6 个残差块，总计 16 个卷积层，最后通过全局平均池化和 Softmax 完成 43 类交通标志分类。
