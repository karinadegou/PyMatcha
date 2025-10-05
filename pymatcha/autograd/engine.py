# pymatcha/autograd/engine.py
"""
自动微分核心引擎
"""

class Function:
    """
    计算图节点基类
    """
    def __init__(self, *args):
        self.inputs = args
        self.outputs = None
        self.saved_tensors = []

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)


class Engine:
    """
    自动微分计算图引擎
    """
    @staticmethod
    def backward(tensor, grad=None):
        if grad is None:
            grad = 1.0  # 标量 Tensor 默认为 1
        if tensor.grad is None:
            tensor.grad = grad
        else:
            tensor.grad += grad

        if hasattr(tensor, "grad_fn") and tensor.grad_fn is not None:
            grads = tensor.grad_fn.backward(grad)
            for inp, g in zip(tensor.grad_fn.inputs, grads):
                Engine.backward(inp, g)
