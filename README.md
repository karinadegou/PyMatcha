# PyMatcha
A lightweight and elegant deep learning library written in Python.

---

## 项目结构

```
PyMatcha/
├── pymatcha/ # 📦 核心库代码
│ ├── tensor.py # 核心 Tensor 类实现（类似 torch.Tensor）
│ ├── nn/ # 神经网络模块（layers、loss、activation 等）
│ ├── optim/ # 优化器模块（如 SGD、Adam 等）
│ ├── autograd/ # 自动微分系统，实现梯度追踪与反向传播
│ ├── utils/ # 工具函数（如张量操作、数学函数等）
│ └── cuda/ # GPU 支持模块（后续可扩展 CUDA 运算）
│
├── tests/ # 🧪 单元测试目录，用于验证各模块功能
├── examples/ # 💡 示例脚本，展示 PyMatcha 的使用方法
├── setup.py # 安装配置脚本（可用 pip install -e . 安装开发版）
├── requirements.txt # 项目依赖列表
├── README.md # 项目说明文档（你正在编辑的文件）
├── LICENSE # 授权协议文件（如 MIT）
└── .gitignore # Git 忽略配置文件
```

---

## 模块说明

- **pymatcha/tensor.py**  
  实现核心 Tensor 类，支持基本的算术运算、形状信息、与 NumPy 兼容的接口。

- **pymatcha/nn/**  
  神经网络模块，包括层（Layer）、损失函数（Loss）、激活函数（Activation）等，用于搭建和训练神经网络。

- **pymatcha/optim/**  
  优化器模块，实现常用优化算法（如 SGD、Adam），用于更新模型参数。

- **pymatcha/autograd/**  
  自动微分系统，实现计算图和梯度追踪，用于反向传播。

- **pymatcha/utils/**  
  工具函数模块，提供张量操作、数学函数、数据处理等辅助功能。

- **pymatcha/cuda/**  
  GPU 支持模块（后续扩展），用于在 CUDA 上执行张量运算以加速计算。

- **tests/**  
  单元测试目录，包含各模块的测试脚本，确保功能正确性。

- **examples/**  
  示例脚本目录，展示 PyMatcha 的实际使用方法，如线性回归、MNIST 训练示例等。

- **setup.py**  
  安装脚本，支持开发安装和依赖管理。

- **requirements.txt**  
  列出项目依赖库，方便环境配置。

- **LICENSE**  
  授权协议文件，PyMatcha采用 MIT 开源协议。

- **.gitignore**  
  Git 忽略配置文件，防止临时文件、虚拟环境被提交到仓库。

---

## 开始使用

```bash
# 安装开发版本
pip install -e .

```
