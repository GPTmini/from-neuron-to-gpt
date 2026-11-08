import os

import numpy as np

from src.core import Tensor
from src.gpt import GPTTrainer


class DPOTrainer(GPTTrainer):

    def __init__(self, model, loss_fn, optimizer, reference):
        super().__init__(model, loss_fn, optimizer)
        self.reference = reference
        self.reference.eval()

    def train(self, dataset, epochs, scheduler=None, filename=None):
        self.model.train()

        steps = 0
        for epoch in range(epochs):
            order = list(range(len(dataset)))
            np.random.shuffle(order)

            total_loss = 0.0
            for step, i in enumerate(order):
                if scheduler is not None:
                    self.optimizer.lr = scheduler.step(steps)

                cx, cy, cm, rx, ry, rm = dataset[i]
                policy_chosen = self._sequence_log_prob(self.model(cx), cy, cm)
                policy_rejected = self._sequence_log_prob(self.model(rx), ry, rm)
                ref_chosen = self._sequence_log_prob(self.reference(cx), cy, cm).data
                ref_rejected = self._sequence_log_prob(self.reference(rx), ry, rm).data
                loss = self.loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

                self.optimizer.zero_grad()
                loss.backward()
                total_loss += float(loss.data)
                self.optimizer.clip_grad_norm()
                self.optimizer.step()
                steps += 1

                if (step + 1) % 100 == 0:
                    lr = f" lr {self.optimizer.lr:.6f}" if scheduler is not None else ""
                    print(f"epoch {epoch + 1} step {step + 1}/{len(dataset)} loss {(total_loss / 100):.4f}{lr}")
                    total_loss = 0.0

            if filename is not None:
                self.save(filename)
                print(f"epoch {epoch + 1} saved DPO model to {filename}")

    def test(self, dataset):
        return None

    @staticmethod
    def _sequence_log_prob(x, y, mask):
        exp = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        y = np.asarray(y.data, dtype=np.int64)
        mask = np.asarray(mask.data, dtype=softmax.dtype)
        picked = np.clip(np.take_along_axis(softmax, y[..., None], axis=-1).squeeze(-1), 1e-10, 1)
        p = Tensor(np.sum(np.log(picked) * mask, axis=-1))

        def gradient_fn():
            grad = -softmax.copy()
            target_grad = np.take_along_axis(grad, y[..., None], axis=-1) + 1.0
            np.put_along_axis(grad, y[..., None], target_grad, axis=-1)
            grad *= mask[..., None]
            x.grad += p.grad[:, None, None] * grad

        return p.attach(gradient_fn, [x])

    def load_reference(self, filename):
        if os.path.isfile(filename):
            data = np.load(filename, allow_pickle=False)
            for i, p in enumerate(self.reference.parameters):
                p.data = data[f"param_{i}"]
                p.grad = np.zeros_like(p.data)
