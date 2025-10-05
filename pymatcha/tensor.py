# -*- coding: utf-8 -*-
"""
PyMatcha Tensor Module
轻量级 Tensor 类，支持基本运算和自动求导占位
"""

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        """
        初始化 Tensor
        Args:
            data (list, tuple, np.ndarray, int, float): 数据
            requires_grad (bool): 是否需要梯度
        """
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=float)
        elif isinstance(data, (int, float, np.ndarray)):
            self.data = np.array(data, dtype=float)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        self.requires_grad = requires_grad
        self.grad = None  # 梯度占位
        self.shape = self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    # -------------------------------
    # 基本算术运算
    # -------------------------------
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        else:
            return Tensor(other - self.data)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def __rtruediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        else:
            return Tensor(other / self.data)

    # -------------------------------
    # 常用属性和方法
    # -------------------------------
    def numpy(self):
        """返回 numpy 数组"""
        return self.data

    def zero_(self):
        """就地清零"""
        self.data.fill(0)

    def fill_(self, value):
        """就地填充常数"""
        self.data.fill(value)

    def shape(self):
        return self.data.shape

    # 后续可以扩展更多方法，例如 matmul, reshape, sum, mean 等
