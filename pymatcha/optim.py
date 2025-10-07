import numpy as np
from pymatcha.tensor import Tensor

# -------------------------
# 损失函数
# -------------------------
class MSELoss:
    def __call__(self, y_pred, y_true):
        diff = y_pred - y_true
        return (diff * diff).mean()

class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        # y_pred 是 softmax 概率
        batch_size, num_classes = y_pred.data.shape

        # 构造 one-hot
        y_true_one_hot = np.zeros_like(y_pred.data)
        y_true_one_hot[np.arange(batch_size), y_true.data.astype(int)] = 1.0

        # 前向
        eps = 1e-9
        loss_value = -np.sum(y_true_one_hot * np.log(y_pred.data + eps)) / batch_size
        loss = Tensor(loss_value, requires_grad=True)

        def _backward():
            if y_pred.requires_grad:
                y_pred.grad = (y_pred.data - y_true_one_hot) / batch_size

        loss._backward = _backward
        loss._prev = {y_pred}
        return loss

# -------------------------
# 优化器
# -------------------------
class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad