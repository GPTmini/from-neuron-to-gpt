import numpy as np

from src.core import Tensor, Loss


class GPTLoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        flat_softmax = softmax.reshape(-1, softmax.shape[-1])
        flat_y = y.data.reshape(-1).astype(np.int64)
        n = len(flat_y)
        rows = np.arange(n)

        log = np.log(np.clip(flat_softmax[rows, flat_y], 1e-10, 1))
        ce = Tensor(0 - np.sum(log) / n)

        def gradient_fn():
            flat_grad = flat_softmax.copy()
            flat_grad[rows, flat_y] -= 1
            p.grad += ce.grad * flat_grad.reshape(softmax.shape) / n

        return ce.attach(gradient_fn, parents=[p])
