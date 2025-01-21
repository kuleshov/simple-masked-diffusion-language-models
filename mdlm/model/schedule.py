import abc

import torch
import torch.nn as nn

class Schedule(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    In MDLM paper notation, returns \alpha_t, and \alpha_t'
    """
    def forward(self, t):
        return self.cumulative(t), self.rate(t)

    @abc.abstractmethod
    def rate(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def cumulative(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass

class LogLinearSchedule(Schedule):
    """Log Linear noise schedule from SEDD/MDLM codebase.

    Built such that 1 - 1/e^(n(t)) interpolates between 0 and
    ~1 when t varies from 0 to 1. Total noise is
    -log(1 - (1 - eps) * t), so the sigma will be
    (1 - eps) * t.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate(self, t):
        return None

    def cumulative(self, t):
        # s = -torch.log1p(-(1 - self.eps) * t)
        # return(1 - torch.exp(-s))
        return t