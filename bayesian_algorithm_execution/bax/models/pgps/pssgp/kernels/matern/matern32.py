import math

import gpflow
import tensorflow as tf

from ..base import ContinuousDiscreteModel, SDEKernelMixin, get_lssm_spec
from .common import get_matern_sde


class Matern32(SDEKernelMixin, gpflow.kernels.Matern32):
    __doc__ = gpflow.kernels.Matern32.__doc__

    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        gpflow.kernels.Matern32.__init__(self, variance, lengthscales, **kwargs)
        SDEKernelMixin.__init__(self, **kwargs)

    def get_spec(self, T):
        return get_lssm_spec(2, T)

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = get_matern_sde(self.variance, self.lengthscales, 2)

        lengthscales = tf.reduce_sum(self.lengthscales)
        lamda = math.sqrt(3) / lengthscales
        variance = tf.reduce_sum(self.variance)

        P_infty = tf.linalg.diag(tf.stack([variance, lamda ** 2 * variance], axis=0))
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
