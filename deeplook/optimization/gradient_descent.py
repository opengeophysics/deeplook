# pylint: disable=too-few-public-methods
"""
Gradient Descent optimization not found in in scipy.optimize.
"""
import math

from .. import backend as bknd


class StopConvergence():
    r"""
    Stop if the value of the goal function didn't change significantly.

    Checks the following inequality (equation 9.49 in [Aster2012]_):

    .. math::

        |f(\mathbf{p}^{k+1}) - f(\mathbf{p}^k)|
        < \epsilon (1 + |f(\mathbf{p}^k)|)

    in which :math:`f` is the goal function, :math:`\mathbf{p}` is the
    parameter vector, and :math:`\epsilon` is a tolerance parameter. A good
    choice of tolerance is the accuracy of gradient computations.

    Parameters
    ----------
    * tol : float
        The tolerance parameter :math:`\epsilon`. Must be a positive scalar.

    Examples
    --------

    >>> conv = StopConvergence(tol=1e-2)
    >>> conv(new_value=1, old_value=0.99, old_gradient=None)
    True
    >>> conv(new_value=1, old_value=0.95, old_gradient=None)
    False

    """

    def __init__(self, tol):
        self.tol = tol

    def __call__(self, new_value, old_value, old_gradient):
        r"""
        Check if the goal function value changed significantly.

        Parameters
        ----------
        * new_value : int
            The new value of the goal function :math:`f(\mathbf{p}^{k+1})`.
        * old_value : int
            The previous value of the goal function :math:`f(\mathbf{p}^k)`.
        * old_gradient : None
            Required by the API but ignored on this method. Use ``None``.

        Returns
        -------
        * stop : bool
            Whether or not the optimization should stop based on this
            criterium.

        """
        return abs(new_value - old_value) < self.tol*(1 + abs(old_value))


class StopSmallGradient():
    r"""
    Stop if the gradient norm is small.

    Checks the following inequality (equation 9.47 in [Aster2012]_):

    .. math::

        \|\nabla f(\mathbf{p}^k)\|_2
        < \sqrt\epsilon (1 + |f(\mathbf{p}^k)|)

    in which :math:`f` is the goal function, :math:`\nabla` is the gradient
    operator, :math:`\mathbf{p}` is the parameter vector, and :math:`\epsilon`
    is a tolerance parameter. A good choice of tolerance is the accuracy of
    gradient computations.

    Parameters
    ----------
    * tol : float
        The tolerance parameter :math:`\epsilon`. Must be a positive scalar.

    Examples
    --------

    >>> import numpy as np
    >>> conv = StopSmallGradient(tol=1e-2)
    >>> conv(new_value=None, old_value=1, old_gradient=np.array([0.1, 0.1]))
    True
    >>> conv(new_value=None, old_value=1, old_gradient=np.array([1, 1]))
    False
    """

    def __init__(self, tol):
        self.tol = tol

    def __call__(self, new_value, old_value, old_gradient):
        r"""
        Check if the gradient of the goal function is small.

        Parameters
        ----------
        * new_value : int
            The new value of the goal function :math:`f(\mathbf{p}^{k+1})`. Not
            required.
        * old_value : int
            The previous value of the goal function :math:`f(\mathbf{p}^k)`.
        * old_gradient : None
            The gradient vector of the goal function :math:`\nabla
            f(\mathbf{p}^k)`.

        Returns
        -------
        * stop : bool
            Whether or not the optimization should stop based on this
            criterium.

        """
        grad_norm = bknd.norm(old_gradient, ord=2)
        return grad_norm < math.sqrt(self.tol)*(1 + abs(old_value))


def apply_preconditioning(gradient, hessian):
    r"""
    Apply Jacobi (diagonal) preconditioning to the Hessian and gradient.

    The preconditioner :math:`\mathbf{P}` is the diagonal of the Hessian
    matrix. This transformation pre-multiplies (dot) the Hessian and gradient
    by :math:`\mathbf{P}^{-1}`. A fudge factor is applied to the diagonal to
    avoid very small values.

    Parameters
    ----------
    gradient : 1d array
        The gradient vector.
    hessian  : 2d array
        The Hessian matrix.

    Returns
    -------
    gradient, hessian : arrays
        The gradient and Hessian after preconditioning.

    """
    diag = bknd.abs(bknd.diagonal(hessian))
    bknd.fudge(diag, factor=1e-10)
    # This is the diagonal of the preconditioning matrix
    preconditioner = 1/diag
    # Doing element-wise multiplication with the diagonal like this is
    # equivalent to the dot product with the full matrix
    hessian = bknd.multiply(preconditioner, hessian.T).T
    gradient = bknd.multiply(preconditioner, gradient)
    return gradient, hessian


class Newton():
    r"""
    Minimize a scalar function using Newton's method.

    Given a goal function :math:`\Gamma(\mathbf{p})`, minimize it iteratively
    by successively adding steps :math:`\Delta\mathbf{p}^k` to a starting
    estimate :math:`\mathbf{p}^0`. A step is calculated by solving the linear
    system:

    .. math::

        \mathbf{H}(\mathbf{p}^k)\Delta\mathbf{p}^k =
        -\nabla \Gamma(\mathbf{p}^k)

    in which :math:`\mathbf{H}(\mathbf{p}^k)` is the Hessian matrix and
    :math:`\nabla\Gamma(\mathbf{p}^k)` is the gradient vector of the goal
    function, both evaluated at iteration k.

    The iterations stop when the maximum number of iterations is reached or
    when the stopping criteria are fulfilled. The default stopping criteria are
    :class:`~deeplook.optimization.StopConvergence` and
    :class:`~deeplook.optimization.StopSmallGradient`. You can replace the
    defaults by assigning a list of stopping criteria to the
    ``stopping_criteria`` attribute.

    Parameters
    ----------
    * initial : 1d array
        Initial estimate of the parameter vector.
    * precondition : bool
        If ``True``, will apply Jacobi preconditioning (recommended).
    * maxit : int
        The maximum number of iterations.
    * tol : float
        The tolerance parameter for the stopping criteria. Use the precision of
        the gradient computations (see equation 9.47 in [Aster2012]_).

    """

    def __init__(self, initial, precondition=True, maxit=30, tol=1e-5):
        self.initial = initial
        self.maxit = maxit
        self.precondition = precondition
        self.stopping_criteria = [StopSmallGradient(tol), StopConvergence(tol)]
        self.status = None

    def step(self, goal, params):
        """
        Update the parameter vector by taking a step in Newton's method.

        Parameters
        ----------
        * goal : goal function object
            The goal function that will be minimized.
        * params : 1d array
            The parameter vector at the last iteration of the algorithm.

        Returns
        -------
        * new_params : 1d array
            The updated parameter vector.
        * gradient : 1d array
            The gradient vector of the goal function evaluated at the given
            *params*. Useful for convergence analysis.

        """
        gradient, hessian = goal.derivatives(params, include_hessian=True)
        if self.precondition:
            gradient, hessian = apply_preconditioning(gradient, hessian)
        new_params = params + bknd.solve(hessian, -gradient)
        return new_params, gradient

    def _stop(self, new_value, old_value, old_gradient):
        """
        Evaluate the stopping criteria.

        Will be ``True`` is all stopping criteria are ``True``.

        Parameters
        ----------
        * new_value, old_value : float
            The values of the goal function after and before the last step.
        * old_gradient : 1d array
            The value of the gradient vector of the goal function before the
            last step.

        Returns
        -------
        * stop : bool
            If ``True``, stop the iterations.

        """
        return all(criterium(new_value, old_value, old_gradient)
                   for criterium in self.stopping_criteria)

    def __call__(self, goal):
        """
        Minimize the given goal function.

        Parameters
        ----------
        * goal : goal function object
            The object must provide a gradient and a Hessian matrix through the
            ``derivatives`` method. See the documentation for more information.

        Returns
        -------
        * params : 1d array
            The parameter vector that minimized the function.

        """
        values = [goal(self.initial)]
        params = self.initial
        for _ in range(self.maxit):
            params, gradient = self.step(goal, params)
            values.append(goal(params))
            if self._stop(values[-1], values[-2], gradient):
                break
        return params
