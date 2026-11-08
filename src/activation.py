import numpy as np

from src.core.layer import Layer
from src.core.tensor import Tensor


class ReLU(Layer):

    def forward(self, x: Tensor):
        a = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad += a.grad * (a.data > 0)

        return a.attach(gradient_fn, parents=[x])


class Tanh(Layer):

    def forward(self, x: Tensor):
        a = Tensor(np.tanh(x.data))

        def gradient_fn():
            x.grad += a.grad * (1 - a.data ** 2)

        return a.attach(gradient_fn, parents=[x])


class Sigmoid(Layer):

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        a = Tensor(1 / (1 + np.exp(-z)))

        def gradient_fn():
            x.grad += a.grad * a.data * (1 - a.data)

        return a.attach(gradient_fn, parents=[x])


class Softmax(Layer):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        a = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            grad = np.sum(a.data * a.grad, axis=self.axis, keepdims=True)
            x.grad += a.data * (a.grad - grad)

        return a.attach(gradient_fn, parents=[x])
