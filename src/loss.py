from abc import abstractmethod, ABC

import numpy as np

from src.core.tensor import Tensor


class Loss(ABC):

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass


class MSELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        mse = Tensor(np.mean(np.square(y.data - p.data)))

        def gradient_fn():
            p.grad += mse.grad * (-2 * (y.data - p.data) / y.data.size)

        return mse.attach(gradient_fn, parents=[p])


class CELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        if y.data.ndim == softmax.ndim:
            labels = y.data
        else:
            labels = np.zeros_like(softmax)
            rows = np.arange(y.data.size)
            labels[rows, y.data.astype(np.int64)] = 1

        log = np.log(np.clip(softmax, 1e-10, 1.0))
        ce = Tensor(-np.mean(np.sum(labels * log, axis=-1)))

        def gradient_fn():
            grad = (softmax - labels) / labels.shape[0]
            p.grad += ce.grad * grad

        return ce.attach(gradient_fn, parents=[p])


class BCELoss(Loss):

    def loss(self, p: Tensor, y: Tensor):
        clipped = np.clip(p.data, 1e-7, 1 - 1e-7)
        bce = Tensor(-np.mean(y.data * np.log(clipped) + (1 - y.data) * np.log(1 - clipped)))

        def gradient_fn():
            grad = (clipped - y.data) / (clipped * (1 - clipped)) / y.data.size
            p.grad += bce.grad * grad

        return bce.attach(gradient_fn, parents=[p])
