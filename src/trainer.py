from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @abstractmethod
    def train(self, dataset, epochs, scheduler=None):
        pass

    @abstractmethod
    def test(self, dataset):
        pass
