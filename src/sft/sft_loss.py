import numpy as np

from src.core import Loss, Tensor


class SFTLoss(Loss):

    def loss(self, p: Tensor, y: Tensor, mask):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        flat_softmax = softmax.reshape(-1, softmax.shape[-1])
        flat_y = y.data.reshape(-1).astype(np.int64)
        flat_mask = mask.data.reshape(-1)
        n = max(np.sum(flat_mask), 1.0)
        rows = np.arange(len(flat_y))

        log = np.log(np.clip(flat_softmax[rows, flat_y], 1e-10, 1))
        ce = Tensor(0 - np.sum(log * flat_mask) / n)

        def gradient_fn():
            flat_grad = flat_softmax.copy()
            flat_grad[rows, flat_y] -= 1
            flat_grad *= flat_mask[:, None]
            p.grad += ce.grad * flat_grad.reshape(softmax.shape) / n

        return ce.attach(gradient_fn, parents={p})
