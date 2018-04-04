"""
Tikhonov regularization
"""
import scipy.sparse


class Damping():
    """
    Damping regularization.
    """

    def __init__(self, regul_param, nparams):
        self.regul_param = regul_param
        self.nparams = nparams

    def hessian(self, params=None):  # pylint: disable=unused-argument
        """
        The Hessian matrix
        """
        return self.regul_param*2*scipy.sparse.identity(self.nparams,
                                                        format='csr')

    def gradient(self, params):
        """
        The gradient vector
        """
        return self.regul_param*2*params

    def gradient_at_null(self):  # pylint: disable=no-self-use
        """
        The gradient vector evaluated at the null vector
        """
        return 0
