import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):

    @abstractmethod
    def step(self, current_step):
        pass


class WarmupCosineScheduler(LRScheduler):

    def __init__(self, lr, total_steps, warmup_steps, min_lr=0.0):
        self.lr = lr
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(min(warmup_steps, self.total_steps), 0)
        self.min_lr = min_lr

    def step(self, current_step):
        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            return self.lr * (current_step + 1) / self.warmup_steps

        if current_step >= self.total_steps:
            return self.min_lr

        progress = (current_step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.lr - self.min_lr) * cosine
