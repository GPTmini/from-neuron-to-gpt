from abc import abstractmethod, ABC

import numpy as np

from src.core.tensor import Tensor, DTYPE


class Layer(ABC):

    def __init__(self):
        self.training = True

    def __call__(self, *args):
        return self.forward(*args)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @abstractmethod
    def forward(self, *args):
        pass

    @property
    def parameters(self):
        return []

    def __repr__(self):
        return f"{type(self).__name__}[]"


class Linear(Layer):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = Tensor(np.random.randn(out_size, in_size).astype(DTYPE) * np.sqrt(2 / in_size))
        self.bias = Tensor(np.zeros(out_size).astype(DTYPE))

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            grad = p.grad.reshape(-1, p.grad.shape[-1])
            self.weight.grad += grad.T @ x.data.reshape(-1, x.shape[-1])
            self.bias.grad += np.sum(grad, axis=0)
            x.grad += p.grad @ self.weight.data

        return p.attach(gradient_fn, {x})

    @property
    def parameters(self):
        return [self.weight, self.bias]


class Composite(Layer, ABC):

    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)

    def train(self):
        super().train()
        for l in self.layers:
            l.train()

    def eval(self):
        super().eval()
        for l in self.layers:
            l.eval()

    @property
    def parameters(self):
        return [p for l in self.layers for p in l.parameters]


class Sequential(Composite):

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x


class Embedding(Layer):

    def __init__(self, vocab_size, embedding_size, std=0.02):
        super().__init__()
        self.weight = Tensor(np.random.randn(vocab_size, embedding_size).astype(DTYPE) * std)

    def forward(self, x: Tensor):
        p = Tensor(self.weight.data[x.data.astype(np.int64)])

        def gradient_fn():
            np.add.at(self.weight.grad, x.data.astype(np.int64), p.grad)

        return p.attach(gradient_fn, parents={self.weight})

    @property
    def parameters(self):
        return [self.weight]


class MeanPool(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.mean(x.data, axis=1))

        def gradient_fn():
            x.grad += p.grad[:, None, :] / x.data.shape[1]

        return p.attach(gradient_fn, parents={x})
