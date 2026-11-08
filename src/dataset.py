from abc import ABC, abstractmethod

from src.core.tensor import Tensor


class Dataset(ABC):

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.load()
        self.train()

    @abstractmethod
    def load(self):
        pass

    def train(self):
        self.data = self.train_data

    def eval(self):
        self.data = self.test_data

    def all(self):
        x, y = self.data
        return Tensor(x), Tensor(y)

    def __len__(self):
        x, *_ = self.data
        return len(x) // self.batch_size

    def __getitem__(self, index):
        s = slice(index * self.batch_size, (index + 1) * self.batch_size)
        x, y = self.data
        return Tensor(x[s]), Tensor(y[s])
