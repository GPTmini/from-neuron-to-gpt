import numpy as np

from src.core import Loss, Tensor


class DPOLoss(Loss):

    def __init__(self, beta=0.1):
        self.beta = beta

    def loss(self, pc: Tensor, pr: Tensor, rc, rr):
        p_diff = pc.data - pr.data
        r_diff = np.asarray(rc) - np.asarray(rr)
        margin = self.beta * (p_diff - r_diff)
        sig = 1.0 / (1.0 + np.exp(-margin))
        e = Tensor(-np.sum(np.log(np.clip(sig, 1e-10, 1.0))) / pc.shape[0])

        def gradient_fn():
            common = (sig - 1.0) / pc.shape[0] * self.beta * e.grad
            pc.grad += common
            pr.grad += -common

        return e.attach(gradient_fn, parents=[pc, pr])
