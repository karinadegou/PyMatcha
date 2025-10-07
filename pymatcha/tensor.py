import numpy as np

# -------------------------
# Tensor 类
# -------------------------
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    # -----------------
    # 基本算子
    # -----------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad = out.grad
                while grad.ndim > self.grad.ndim:
                    grad = grad.sum(axis=0, keepdims=True)
                for i, dim in enumerate(self.grad.shape):
                    if dim == 1 and grad.shape[i] != 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad = out.grad
                while grad.ndim > other.grad.ndim:
                    grad = grad.sum(axis=0, keepdims=True)
                for i, dim in enumerate(other.grad.shape):
                    if dim == 1 and grad.shape[i] != 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                grad = other.data * out.grad
                for i, dim in enumerate(self.grad.shape):
                    if dim == 1 and grad.shape[i] != 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad

            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                grad = self.data * out.grad
                for i, dim in enumerate(other.grad.shape):
                    if dim == 1 and grad.shape[i] != 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out
    
    def relu(self):
        # 前向传播：ReLU(x) = max(0, x)
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                # 反向传播：ReLU'(x) = 1 if x > 0 else 0
                grad = np.where(self.data > 0, 1, 0) * out.grad
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        out._prev = {self}
        return out
    
    def softmax(self, axis=-1):
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            grad_output = out.grad
            s = out.data
            # 向量化梯度计算：dx = s * (grad_output - sum(grad_output * s))
            dx = s * (grad_output - np.sum(grad_output * s, axis=axis, keepdims=True))

            if self.grad is None:
                self.grad = dx
            else:
                self.grad += dx

        out._backward = _backward
        out._prev = {self}
        return out

    def conv2d(self, weight, bias=None, stride=1, padding=0):
        """
        2D 卷积操作: out = conv2d(self, weight) + bias
        self: (N, C_in, H, W)
        weight: (C_out, C_in, KH, KW)
        bias: (C_out,)
        """
        N, C_in, H, W = self.data.shape
        C_out, _, KH, KW = weight.data.shape
        S = stride

        # 填充
        if padding > 0:
            x_padded = np.pad(self.data, ((0, 0), (0, 0),
                                        (padding, padding),
                                        (padding, padding)), mode='constant')
        else:
            x_padded = self.data

        H_out = (H + 2 * padding - KH) // S + 1
        W_out = (W + 2 * padding - KW) // S + 1

        out_data = np.zeros((N, C_out, H_out, W_out))

        # 卷积计算
        for n in range(N):
            for co in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i * S, j * S
                        region = x_padded[n, :, h_start:h_start+KH, w_start:w_start+KW]
                        out_data[n, co, i, j] = np.sum(region * weight.data[co])
                        if bias is not None:
                            out_data[n, co, i, j] += bias.data[co]

        out = Tensor(out_data, requires_grad=self.requires_grad or weight.requires_grad or (bias and bias.requires_grad))

        def _backward():
            if not out.requires_grad:
                return
            grad_output = out.grad

            dX = np.zeros_like(x_padded)
            dW = np.zeros_like(weight.data)
            dB = np.zeros_like(bias.data) if bias is not None else None

            for n in range(N):
                for co in range(C_out):
                    for i in range(H_out):
                        for j in range(W_out):
                            h_start, w_start = i * S, j * S
                            region = x_padded[n, :, h_start:h_start+KH, w_start:w_start+KW]

                            dW[co] += grad_output[n, co, i, j] * region
                            dX[n, :, h_start:h_start+KH, w_start:w_start+KW] += grad_output[n, co, i, j] * weight.data[co]
                            if bias is not None:
                                dB[co] += grad_output[n, co, i, j]

            # 去掉padding
            if padding > 0:
                dX = dX[:, :, padding:-padding, padding:-padding]

            # 反向传播回传
            if self.requires_grad:
                self.grad = (self.grad + dX) if self.grad is not None else dX
            if weight.requires_grad:
                weight.grad = (weight.grad + dW) if weight.grad is not None else dW
            if bias is not None and bias.requires_grad:
                bias.grad = (bias.grad + dB) if bias.grad is not None else dB

        out._backward = _backward
        out._prev = {self, weight} if bias is None else {self, weight, bias}

        return out
    
    def max_pool2d(self, kernel_size, stride=None):
        if stride is None:
            stride = kernel_size
        x = self.data
        b, c, h, w = x.shape
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        out_data = np.zeros((b, c, out_h, out_w))
        max_index = np.zeros_like(x, dtype=bool)

        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                max_val = np.max(region, axis=(2, 3))
                out_data[:, :, i, j] = max_val
                for bi in range(b):
                    for ci in range(c):
                        mask = region[bi, ci] == max_val[bi, ci]
                        max_index[bi, ci, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size] |= mask

        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            grad_out = out.grad
            dx = np.zeros_like(x)
            for i in range(out_h):
                for j in range(out_w):
                    grad_slice = grad_out[:, :, i, j][:, :, None, None]
                    dx[:, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size] += grad_slice * max_index[:, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
            self.grad = (self.grad + dx) if self.grad is not None else dx

        out._backward = _backward
        out._prev = {self}
        return out

    def avg_pool2d(self, kernel_size, stride=None):
        if stride is None:
            stride = kernel_size
        x = self.data
        b, c, h, w = x.shape
        out_h = (h - kernel_size) // stride + 1
        out_w = (w - kernel_size) // stride + 1
        out_data = np.zeros((b, c, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                region = x[:, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size]
                out_data[:, :, i, j] = np.mean(region, axis=(2, 3))

        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if not self.requires_grad:
                return
            grad_out = out.grad
            dx = np.zeros_like(x)
            for i in range(out_h):
                for j in range(out_w):
                    grad_slice = grad_out[:, :, i, j][:, :, None, None] / (kernel_size * kernel_size)
                    dx[:, :, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size] += grad_slice
            self.grad = (self.grad + dx) if self.grad is not None else dx

        out._backward = _backward
        out._prev = {self}
        return out

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad * np.ones_like(self.data) / self.data.size

        out._backward = _backward
        out._prev = {self}
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                grad = np.where(self.data > 0, 1, 0) * out.grad
                if self.grad is None:
                    self.grad = grad
                else:
                    self.grad += grad

        out._backward = _backward
        out._prev = {self}  # ✅必须加
        return out

    def softmax(self, axis=-1):
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(out_data, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                # 简化处理，假设与交叉熵结合使用
                if self.grad is None:
                    self.grad = out.grad
                else:
                    self.grad += out.grad

        out._backward = _backward
        out._prev = {self}  # ✅必须加
        return out

    # -----------------
    # 自动求导
    # -----------------
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)