# pylint: disable=redefined-outer-name,no-self-use
"""
Test the gradient descent optimization.
"""
from __future__ import division
from future.builtins import object

import pytest
import numpy as np
import numpy.testing as npt

from ..optimization.gradient_descent import Newton, apply_preconditioning


class Paraboloid(object):
    "An N-dimensional paraboloid function for minimization."

    def __call__(self, params):
        "Evaluate the function at params"
        return params.T.dot(params)

    def derivatives(self, params, include_hessian=False):
        """
        Calculate the derivatives (gradient) of this function.
        If include_hessian==True, will also return the Hessian matrix.
        """
        gradient = 2*params
        if include_hessian:
            hessian = 2*np.identity(params.size)
            return gradient, hessian
        return gradient


@pytest.fixture
def paraboloid():
    "Return an ND paraboloid function."
    return Paraboloid()


def test_newton(paraboloid):
    "Test Newton's method on a paraboloid function with default parameters."
    optimizer = Newton(initial=np.array([1, 0.5, -2, 244, -1e10]))
    estimate = optimizer(paraboloid)
    npt.assert_allclose(estimate, np.zeros_like(estimate), atol=1e-10)


def test_preconditioning():
    "Check that preconditioning is being done correctly on simple matrices"
    hessian = np.array([[2, 4, 5],
                        [-5, -10, 7],
                        [8, 9, 1e-15]])
    gradient = np.array([-1, -2, -3])
    pre_gradient, pre_hessian = apply_preconditioning(gradient, hessian)
    true_hessian = np.array([[1, 2, 2.5],
                             [-0.5, -1, 0.7],
                             [8e10, 9e10, 1e-5]])
    true_gradient = np.array([-0.5, -0.2, -3e10])
    npt.assert_allclose(pre_hessian, true_hessian)
    npt.assert_allclose(pre_gradient, true_gradient)
