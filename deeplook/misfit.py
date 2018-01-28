"""
Define the data misfit classes
"""
import numpy as np
import scipy.sparse

from . import backend as bknd


def linear_solver(goal):
    """
    Find the minimum of a linear goal function.
    """
    hessian = goal.hessian()
    gradient = goal.gradient_at_null()
    estimate = bknd.solve(hessian, -gradient, sym_pos=True)
    return estimate


def normalize_jacobian(jacobian):
    # Normalize the Jacobian to the range [-1, 1] using a variable change
    scale = 1/bknd.abs(jacobian).max(axis=0)
    # Element-wise multiplication with the diagonal of the scale matrix is the
    # same as A.dot(S)
    jacobian = bknd.multiply(jacobian, scale)
    return jacobian, scale


class LinearMisfit():
    """
    The linear least-squares data misfit function.
    """

    def __init__(self, data, jacobian, weights=None, normalize=False,
                 regularization=None):
        self.data = data
        self.normalize = normalize
        if normalize:
            jacobian, self.scale_factor = normalize_jacobian(jacobian)
        self.jacobian = jacobian
        if weights is None:
            self.weights = scipy.sparse.identity(data.size, format='csr')
        else:
            self.weights = scipy.sparse.diags(weights, 0, format='csr')
        if regularization is None:
            regularization = []
        self.regularization = regularization

    def minimize(self, method='linear'):
        """
        Minimize the data-misfit function.
        """
        if method == 'linear':
            method = linear_solver
        estimate = method(self)
        if self.normalize:
            estimate *= self.scale_factor
        return estimate

    def hessian(self, params=None):
        hessian = (self.misfit_hessian(params) +
                   sum(regul.hessian(params) for regul in self.regularization))
        return hessian

    def gradient(self, params):
        gradient = (self.misfit_gradient(params) +
                    sum(regul.gradient(params)
                        for regul in self.regularization))
        return gradient

    def gradient_at_null(self):
        gradient = (self.misfit_gradient_at_null() +
                    sum(regul.gradient_at_null()
                        for regul in self.regularization))
        return gradient

    def misfit_hessian(self, params=None):
        """
        The Hessian matrix.
        """
        hessian = 2*bknd.dot(bknd.dot(self.jacobian.T, self.weights),
                             self.jacobian)
        return hessian

    def misfit_gradient(self, params):
        """
        The gradient vector.
        """
        residuals = self.data - bknd.dot(self.jacobian, params)
        gradient = -2*bknd.dot(bknd.dot(self.jacobian.T, self.weights),
                               residuals)
        return gradient

    def misfit_gradient_at_null(self):
        """
        The gradient vector evaluated at the null vector.
        """
        gradient = -2*bknd.dot(bknd.dot(self.jacobian.T, self.weights),
                               self.data)
        return gradient
