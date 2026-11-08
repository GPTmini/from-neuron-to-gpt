import contextlib

import numpy as np

DTYPE = np.float32


class Tensor:
    grad_enabled = True

    @classmethod
    @contextlib.contextmanager
    def no_grad(cls):
        prev = cls.grad_enabled
        cls.grad_enabled = False
        try:
            yield
        finally:
            cls.grad_enabled = prev

    def __init__(self, data):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = np.zeros_like(self.data)
        self.gradient_fn = None
        self.parents = []

    def backward(self):
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if node in visited:
                continue

            if expanded:
                visited.add(node)
                topo.append(node)
            else:
                stack.append((node, True))
                for p in node.parents:
                    if p not in visited:
                        stack.append((p, False))

        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            if t.gradient_fn is not None:
                t.gradient_fn()

        for t in topo:
            t.gradient_fn = None
            t.parents = []

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __add__(self, other):
        p = Tensor(self.data + other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.shape)
            other.grad += self._unbroadcast(p.grad, other.shape)

        return p.attach(gradient_fn, parents=[self, other])

    def __sub__(self, other):
        p = Tensor(self.data - other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad, self.shape)
            other.grad += self._unbroadcast(-p.grad, other.shape)

        return p.attach(gradient_fn, parents=[self, other])

    def __mul__(self, other):
        p = Tensor(self.data * other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad * other.data, self.shape)
            other.grad += self._unbroadcast(p.grad * self.data, other.shape)

        return p.attach(gradient_fn, parents=[self, other])

    def __truediv__(self, other):
        p = Tensor(self.data / other.data)

        def gradient_fn():
            self.grad += self._unbroadcast(p.grad / other.data, self.shape)
            other.grad += self._unbroadcast(-p.grad * self.data / (other.data ** 2), other.shape)

        return p.attach(gradient_fn, parents=[self, other])

    def __matmul__(self, other):
        p = Tensor(np.matmul(self.data, other.data))

        def gradient_fn():
            self.grad += self._unbroadcast(np.matmul(p.grad, other.data.swapaxes(-1, -2)), self.shape)
            other.grad += self._unbroadcast(np.matmul(self.data.swapaxes(-1, -2), p.grad), other.shape)

        return p.attach(gradient_fn, parents=[self, other])

    def transpose(self, axes=None):
        p = Tensor(np.transpose(self.data, axes))

        def gradient_fn():
            if axes is None:
                self.grad += np.transpose(p.grad)
            else:
                idx = np.argsort(axes)
                self.grad += np.transpose(p.grad, idx)

        return p.attach(gradient_fn, parents=[self])

    @property
    def T(self):
        return self.transpose()

    def reshape(self, shape):
        p = Tensor(np.reshape(self.data, shape))

        def gradient_fn():
            self.grad += np.reshape(p.grad, self.shape)

        return p.attach(gradient_fn, parents=[self])

    def attach(self, gradient_fn, parents):
        if Tensor.grad_enabled:
            self.gradient_fn = gradient_fn
            self.parents = parents
        return self

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"

    @staticmethod
    def _unbroadcast(grad, shape):
        if grad.ndim > len(shape):
            grad = grad.sum(axis=tuple(range(grad.ndim - len(shape))))

        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad.reshape(shape)
