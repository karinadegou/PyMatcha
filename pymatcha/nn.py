import numpy as np
from pymatcha.tensor import Tensor

class Module:
    def __call__(self, *inputs):
        # forward 方法由子类实现
        out = self.forward(*inputs)
        
        # 如果需要可以在这里挂 backward
        # 对于简单层，子类可以在 forward 内部处理 _backward
        return out

    def parameters(self):
        # 遍历所有属性，如果是 Tensor 并且 requires_grad=True，就返回
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor) and attr.requires_grad:
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
        return params

# -------------------------
# Linear 层
# -------------------------

# 继承 Module 的 Linear
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.w = Tensor(np.random.randn(in_features, out_features) * 0.1, requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x):
        out = x @ self.w + self.b
        '''
        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += out.grad @ self.w.data.T
            if self.w.requires_grad:
                if self.w.grad is None:
                    self.w.grad = np.zeros_like(self.w.data)
                self.w.grad += x.data.T @ out.grad
            if self.b.requires_grad:
                if self.b.grad is None:
                    self.b.grad = np.zeros_like(self.b.data)
                self.b.grad += np.sum(out.grad, axis=0, keepdims=True)
        '''
        # out._backward = _backward
        # out._prev = {x, self.w, self.b}
        return out
    

class ReLU(Module):
    def __init__(self):
        pass

    def forward(self, x):
        out = x.relu()
        return out
    
class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        # 减去最大值防止溢出
        out = x.softmax(self.axis)
        return out
    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01,
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)
    
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.max_pool2d(kernel_size=self.kernel_size, stride=self.stride)
    
class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.avg_pool2d(kernel_size=self.kernel_size, stride=self.stride)
