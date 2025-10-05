# -*- coding: utf-8 -*-
"""
🍵 PyMatcha — Lightweight Deep Learning Library
Inspired by PyTorch.
"""

# 版本信息
__version__ = "0.1.0"

# 对外暴露的核心模块（示例）
from pymatcha import tensor
from pymatcha import nn
from pymatcha import optim
from pymatcha import autograd
from pymatcha import utils

# 如果需要，可以在这里添加快捷导入
# 例如：
# from pymatcha.tensor import Tensor
# from pymatcha.nn.layers import Linear, ReLU
# from pymatcha.optim.sgd import SGD
# from pymatcha.autograd.engine import backward

# 当直接执行 pymatcha 时，调用 CLI
if __name__ == "__main__":
    from pymatcha.__main__ import main
    main()
