# pymatcha/autograd/grad_fn.py
"""
梯度函数，实现基本算子
"""

from pymatcha.autograd.engine import Function
import numpy as np

class Add(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a + b

    def backward(self, grad_output):
        # 对加法，梯度直接传递
        return [grad_output, grad_output]


class Mul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a * b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        return [grad_output * b, grad_output * a]
