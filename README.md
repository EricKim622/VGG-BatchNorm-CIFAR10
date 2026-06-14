 
# VGG-BatchNorm-CIFAR10

VGG-A with/without Batch Normalization on CIFAR-10

---

## 项目介绍
本仓库为课程实验代码，用于验证 Batch Normalization（批归一化）对深度卷积神经网络训练的优化效果。我们在 CIFAR-10 图像分类任务上，对比了无 BN 的原始 VGG-A 模型与加入 BN 的 VGG-A 模型的训练过程与性能差异。

核心目标：
1. 验证 BN 层能显著加快模型收敛速度
2. 对比两组模型的最终分类准确率
3. 通过损失景观（Loss Landscape）可视化，说明 BN 层如何平滑优化曲面、提升训练稳定性

---

## 仓库结构
VGG_BatchNorm
├── data
├── models
├── reports
│   └── figures
├── VGG_Loss_Landscape.py
├── thelast.ipynb
└── README.md
---

## 运行环境配置

# 1. 创建虚拟环境（可选）
conda create -n vgg-bn python=3.8
conda activate vgg-bn

# 2. 安装依赖库
pip install torch torchvision matplotlib tqdm

---

## 运行方式

方式1：直接运行 Python 脚本
python VGG_Loss_Landscape.py

方式2：使用 Jupyter Notebook 运行
jupyter notebook thelast.ipynb

运行后将自动完成：
1. 加载/下载 CIFAR-10 数据集
2. 训练无 BN 与带 BN 的两组 VGG-A 模型（各20轮）
3. 生成训练损失、验证准确率、损失景观等所有图表
4. 所有结果自动保存到 reports/figures/ 目录

---

## 实验结果说明

1. 训练与验证曲线
- 损失曲线：带 BN 模型的损失下降速度更快，全程更平滑
- 准确率曲线：带 BN 模型在第4轮就达到了无 BN 模型20轮的精度，最终准确率高出约10%

2. 损失景观（Loss Landscape）
- 无 BN 模型：损失波动区间全程较宽，训练过程震荡明显
- 带 BN 模型：损失区间随训练持续收窄，优化曲面更平滑，梯度更新更稳定

---

## 参考文献
1. Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015.
2. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
