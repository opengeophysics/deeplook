"""
Tikhonov regularization
"""
import numpy as np
import scipy.sparse


class Damping():

    def __init__(self, regul_param, nparams):
        self.regul_param = regul_param
        self.nparams = nparams

    def hessian(self, params=None):
        return self.regul_param*2*scipy.sparse.identity(self.nparams,
                                                        format='csr')

    def gradient(self, params):
        return self.regul_param*2*params

    def gradient_at_null(self):
        return 0
