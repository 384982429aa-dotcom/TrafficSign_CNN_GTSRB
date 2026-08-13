# 基于 ResNet18-FPN 的交通标志识别系统

本项目是一个基于卷积神经网络的交通标志识别课设项目，使用 GTSRB 交通标志数据集完成多类别图像分类任务。项目针对交通标志图片尺寸较小、原始图片大小不统一的问题，设计了 **Small-Image ResNet18 + FPN 多尺度特征融合模型**，用于识别不同类别的交通标志。

## 1. 项目目标

设计并训练一个 CNN 模型，对交通标志数据集进行分类识别。模型需要能够识别限速、停车、转弯、警告等不同类型的交通标志，并通过数据增强、多尺度特征融合等方法提升模型鲁棒性。

## 2. 数据集

本项目使用的数据集路径为：

```text
E:\ML\GTSRB
```

当前数据集结构：

```text
GTSRB/
├─ data/
│  ├─ Train.csv
│  ├─ Test.csv
│  └─ Meta.csv
├─ Train/
│  ├─ 0/
│  ├─ 1/
│  └─ ...
├─ Test/
│  ├─ 00000.png
│  ├─ 00001.png
│  └─ ...
└─ Meta/
```

其中：

- `Train.csv`：训练集标注文件，包含图片路径和类别编号。
- `Test.csv`：测试集标注文件，用于最终模型评估。
- `ClassId`：交通标志类别编号。
- `Path`：图片相对路径，例如 `Train/20/00020_00000_00000.png`。

## 3. 项目目录

```text
TrafficSign_CNN_GTSRB/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ docs/
│  └─ CNN_model_structure.md
├─ models/
│  ├─ resnet18_fpn_best.pth
│  ├─ resnet18_fpn_final.pth
│  └─ label_mapping.json
├─ outputs/
│  ├─ figures/
│  ├─ logs/
│  └─ reports/
├─ scripts/
│  └─ train_resnet18_fpn.bat
├─ src/
│  ├─ data/
│  ├─ evaluation/
│  ├─ models/
│  │  └─ cnn_model.py
│  ├─ prediction/
│  ├─ training/
│  │  └─ train.py
│  └─ utils/
├─ requirements.txt
└─ README.md
```

## 4. 模型结构

本项目模型采用：

```text
Small-Image ResNet18 + FPN 多尺度特征融合
```

整体流程：

```text
原始交通标志图片
        ↓
Resize 到 64×64×3
        ↓
归一化到 0~1
        ↓
数据增强
        ↓
Small-Image ResNet18 主干网络
        ↓
C1, C2, C3, C4 多尺度特征
        ↓
FPN / YOLO 风格上采样拼接融合
        ↓
GlobalAveragePooling
        ↓
Dense + Dropout
        ↓
Softmax 分类输出
```

主干网络提取的特征尺度：

```text
C1: 64×64×64
C2: 32×32×128
C3: 16×16×256
C4: 8×8×512
```

FPN 特征融合：

```text
C4 上采样 + C3 拼接 → F3
F3 上采样 + C2 拼接 → F2
F2 上采样 + C1 拼接 → F1
```

这样设计的原因是：交通标志图片可能比较小，如果网络不断下采样，浅层的边缘、颜色、数字和箭头细节容易丢失。通过 FPN 式多尺度拼接，可以同时利用浅层细节特征和深层语义特征。

## 5. 环境依赖

安装依赖：

```bat
cd /d E:\class_work\TrafficSign_CNN_GTSRB
venv\Scripts\pip install -r requirements.txt
```

如果不使用项目中的虚拟环境，也可以使用：

```bat
pip install -r requirements.txt
```

主要依赖：

```text
torch
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## 6. 训练模型

方式一：命令行运行

```bat
cd /d E:\class_work\TrafficSign_CNN_GTSRB
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 64 --epochs 30
```

方式二：运行脚本

```bat
scripts\train_resnet18_fpn.bat
```

训练脚本会自动完成：

- 读取 `Train.csv`
- 根据 `Path` 加载图片
- 将图片统一 resize 到 `64×64`
- 像素归一化到 `0~1`
- 训练集数据增强
- 按类别计算 `class_weight`
- 训练 ResNet18-FPN 模型
- 保存最优模型和最终模型
- 输出训练曲线、混淆矩阵和分类报告
- 使用 `Test.csv` 进行测试集评估

## 7. 训练输出

训练完成后会生成：

```text
models/
├─ resnet18_fpn_best.pth
├─ resnet18_fpn_final.pth
└─ label_mapping.json
```

```text
outputs/
├─ figures/
│  ├─ training_curves.png
│  ├─ validation_confusion_matrix.png
│  └─ test_confusion_matrix.png
├─ logs/
│  └─ training_log.csv
└─ reports/
   ├─ validation_classification_report.txt
   └─ test_classification_report.txt
```

## 8. 模型特点

- 使用 ResNet18 残差结构，缓解深层网络训练困难。
- 针对小尺寸交通标志图片，使用 `3×3 stride=1` 的 Stem 卷积，不使用原版 ResNet 的大步长开头。
- 使用 FPN 式上采样拼接，融合浅层细节和深层语义。
- 支持不同类别数量，输出层通过 `num_classes` 控制。
- 使用数据增强提升模型对旋转、平移、缩放和对比度变化的鲁棒性。
- 使用类别权重缓解类别样本不平衡问题。

## 9. 答辩说明

本项目采用 Small-Image ResNet18-FPN 模型进行交通标志识别。由于交通标志原始图片尺寸不统一，且小尺寸图片较多，因此首先将图片统一 resize 到 64×64。模型主干采用适合小图像的 ResNet18，去掉原版 ResNet 开头的大卷积和 MaxPool，避免过早丢失空间细节。同时，为了进一步保留小目标的边缘、颜色和图案信息，在 ResNet 主干后加入 FPN 多尺度特征融合结构，将深层语义特征逐级上采样，并与浅层高分辨率特征拼接。最后通过全局平均池化和 Softmax 完成交通标志分类。

## 10. 运行命令与参数说明

如果 PyCharm 已经配置好解释器，并且终端当前就在项目根目录：

```text
E:\class_work\TrafficSign_CNN_GTSRB
```

可以直接运行：

```bat
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 64 --epochs 30
```

如果只是想先测试程序能不能正常训练，可以运行：

```bat
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 8 --epochs 1 --no-official-test
```

各参数含义如下：

| 参数 | 示例 | 含义 |
|---|---|---|
| `python` | `python` | 使用当前 PyCharm 解释器运行脚本。如果 PyCharm 环境配置正确，直接写 `python` 即可。 |
| `src\training\train.py` | `src\training\train.py` | 训练脚本路径。 |
| `--data-dir` | `E:\ML\GTSRB` | 数据集根目录，里面应包含 `data\Train.csv`、`data\Test.csv`、`Train`、`Test` 等文件夹。 |
| `--img-size` | `64` | 输入模型前统一缩放后的图片尺寸，即 `64×64`。 |
| `--batch-size` | `64` | 每次送入模型训练的图片数量。显存不足时可以改成 `32`、`16` 或 `8`。 |
| `--epochs` | `30` | 训练轮数。`30` 表示完整训练 30 轮。 |
| `--no-official-test` | 可选 | 跳过官方测试集 `Test.csv` 评估，适合快速测试训练流程。正式训练时可以不加。 |

快速测试命令的作用：

```bat
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 8 --epochs 1 --no-official-test
```

表示只训练 1 轮，batch size 设置为 8，并跳过官方测试集评估。这个命令主要用于检查环境、数据路径和模型训练流程是否正常。

正式训练命令的作用：

```bat
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 64 --epochs 30
```

表示使用 GTSRB 数据集，将图片统一缩放到 `64×64`，每批训练 64 张图片，完整训练 30 轮，并在训练后输出模型、训练曲线、混淆矩阵和分类报告。

如果运行时提示显存不足，可以减小 batch size：

```bat
python src\training\train.py --data-dir E:\ML\GTSRB --img-size 64 --batch-size 32 --epochs 30
```

如果想确认当前 `python` 是否就是 PyCharm 配置好的解释器，可以运行：

```bat
python -c "import sys; print(sys.executable)"
```

正常情况下应输出你在 PyCharm 中配置的解释器路径。
## 11. 单张图片预测

训练完成后，可以使用下面的脚本识别自己提供的交通标志图片：

```bat
python src\prediction\predict_image.py
```

运行后控制台会提示：

```text
请输入图片地址：
```

此时输入你自己的图片路径，例如：

```text
E:\ML\GTSRB\Test\00000.png
```

脚本会自动完成：

- 加载 `models\resnet18_fpn_best.pth`，如果不存在则尝试加载 `models\resnet18_fpn_final.pth`
- 读取你输入的图片
- 将图片 resize 到 `64×64`
- 归一化并输入模型
- 输出预测类别 ID、类别名称、置信度和 Top-5 预测结果
- 弹出窗口显示输入图片和预测结果

示例输出：

```text
当前设备：cuda
已加载模型：E:\class_work\TrafficSign_CNN_GTSRB\models\resnet18_fpn_best.pth
请输入图片地址：E:\ML\GTSRB\Test\00000.png

预测结果：
图片地址：E:\ML\GTSRB\Test\00000.png
预测类别ID：16
预测类别名称：Vehicles over 3.5 tons prohibited / 禁止大型车辆通行
置信度：0.9234
```