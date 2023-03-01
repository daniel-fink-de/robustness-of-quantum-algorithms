from numpy import sqrt, divide, zeros, square

"""
This module contains the ADAM optimizer.
"""


class ADAM:
    """
    The ADAM optimizer.
    """

    def __init__(self,
                 n_parameter: int,
                 alpha: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        """
        Initializes the ADAM optimizer.
        """

        self.n_parameter = n_parameter
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = zeros(self.n_parameter, dtype=float)
        self.v = zeros(self.n_parameter, dtype=float)
        self.t = 0

        return

    def step(self, x_t, grad_cost_x_t):
        """
        Performs one step of the ADAM optimizer given the old value
        and the gradient for the old value.

        :param x_t: The old value.
        :param grad_cost_x_t: The gradient of the old value
        :return: The new value x_t+1.
        """

        self.t = self.t + 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad_cost_x_t
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * square(grad_cost_x_t)

        mh = 1.0 / (1.0 - self.beta1**self.t) * self.m
        vh = 1.0 / (1.0 - self.beta2**self.t) * self.v

        vh_sqrt_eps = sqrt(vh) + self.eps

        update = x_t - self.alpha * divide(mh, vh_sqrt_eps)

        return update
