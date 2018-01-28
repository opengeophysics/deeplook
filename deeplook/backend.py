# pylint: disable=redefined-builtin
"""
Linear algebra backend for numpy and scipy.sparse.

Delegates to the correct library depending on data type and flags.
Allows you to mix numpy arrays and scipy sparse matrices without extra effort.
"""
from __future__ import division

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla


def multiply(left, right):
    """
    Element-wise multiplication of two arrays.

    Numpy broadcasting rules apply.

    Parameters
    ----------
    * left, right : nd arrays
        The arrays which will be multiplied.

    Returns
    -------
    * prod : nd array
        The result of the element-wise multiplication.

    """
    if sp.sparse.issparse(left):
        result = left.multiply(right)
    elif sp.sparse.issparse(right):
        result = right.multiply(left)
    else:
        result = left*right
    return result


def dot(left, right):
    """
    Take the dot product between two arrays.

    Can handle numpy arrays and scipy.sparse matrices.

    Parameters
    ----------
    * left, right : 1 or 2d arrays
        The vectors/matrices of which to take the dot product.

    Returns
    -------
    * prod : 1 or 2d array
        The dot product of *left* and *right*

    Examples
    --------

    >>> import numpy as np
    >>> left = np.array([[2, 0], [0, 2]])
    >>> right = np.array([[3, 0], [0, 3]])
    >>> dot(left, right)
    array([[6, 0],
           [0, 6]])
    >>> # Mix in a sparse matrix
    >>> from scipy.sparse import diags
    >>> spmatrix = diags([4, 5], 0)
    >>> spmatrix.toarray()
    array([[4., 0.],
           [0., 5.]])
    >>> spdot = dot(spmatrix, right)
    >>> spdot
    array([[12.,  0.],
           [ 0., 15.]])
    >>> spdot2 = dot(left, spmatrix)
    >>> spdot2
    array([[ 8.,  0.],
           [ 0., 10.]])

    """
    if sp.sparse.issparse(right) or sp.sparse.issparse(left):
        result = left*right
    else:
        result = left.dot(right)
    return result


def diagonal(matrix):
    """
    Get the main diagonal of a matrix.

    Parameters
    ----------
    * matrix : 2d-array
        The matrix.

    Returns
    -------
    * diag : 1d-array
        A numpy array with the diagonal of the matrix.

    Examples
    --------

    >>> import numpy as np
    >>> matrix = np.array([[2, 0], [0, 5]])
    >>> diagonal(matrix)
    array([2, 5])
    >>> # With a sparse matrix
    >>> from scipy.sparse import diags
    >>> spmatrix = diags([4, 10], 0)
    >>> diagonal(spmatrix)
    array([ 4., 10.])

    """
    if sp.sparse.issparse(matrix):
        diag = np.array(matrix.diagonal())
    else:
        diag = np.diagonal(matrix).copy()
    return diag


def abs(array):
    """
    Calculate the absolute value of an array.

    Parameters
    ----------
    * array : nd array
        The array with integer or floating point values.

    Returns
    -------
    * abs_array : nd array
        The element-wise absolute value of the array.

    Examples
    --------

    >>> import numpy as np
    >>> abs(np.array([1, -2, 3, -4]))
    array([1, 2, 3, 4])
    """
    return np.abs(array)


def norm(array, ord=2):
    """
    Calculate the Nth order norm of the array.

    Parameters
    ----------
    array : nd array
        The array.
    ord : int
        The order of the norm.

    Returns
    -------
    norm : float
        The Nth order norm of the array.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.array([2, 2, 2, 2])
    >>> norm(x, ord=2)
    4.0

    """
    return np.linalg.norm(array, ord=ord)


def fudge(array, factor):
    """
    Replace small values in an array with a fudge factor.

    Any element whose absolute value is smaller than *factor* will be replaced
    by factor.

    Operates **in memory**, so the original values will be lost.

    Parameters
    ----------
    array : nd array
        Any array that supports numpy-style fancy indexing.

    Returns
    -------
    fudged_array : nd array
        The array with fudge factor applied.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 1e-15, -1e-12])
    >>> fudge(x, factor=1e-10)
    array([1.e+00, 2.e+00, 3.e+00, 1.e-10, 1.e-10])

    """
    array[abs(array) < factor] = factor
    return array


def inv(matrix):
    """
    Calculate the inverse of a matrix.

    Uses the standard :func:`scipy.linalg.inv` if *matrix* is dense.
    If it is sparse (from :mod:`scipy.sparse`) then will use
    :func:`scipy.sparse.linalg.inv`.

    Parameters
    ----------
    * matrix : 2d-array
        A dense numpy array or scipy.sparse matrix.

    Returns
    -------
    * inverse : 2d-array
        The inverse of *matrix*

    """
    if sp.sparse.issparse(matrix):
        result = spla.inv(matrix)
    else:
        result = sp.linalg.inv(matrix)
    return result


def solve(matrix, vector, sym_pos=False):
    """
    Solve a linear system.

    Uses the standard :func:`scipy.linalg.solve` if both *matrix* and *vector*
    are dense.

    If any of the two is sparse (from :mod:`scipy.sparse`) then will use the
    Conjugate Gradient Method (:func:`scipy.sparse.linalg.cgs`).

    Parameters
    ----------
    * matrix : 2d-array
        The matrix defining the linear system.
    * vector : 1d or 2d-array
        The right-side vector of the system.
    * sym_pos : bool
        If the matrix is not sparse and is symmetric positive definite, use
        ``sym_pos=True``.

    Returns
    -------
    * solution : 1d or 2d-array
        The solution of the linear system.

    """
    if sp.sparse.issparse(matrix) or sp.sparse.issparse(vector):
        solution, _ = spla.cgs(matrix, vector)
    else:
        solution = sp.linalg.solve(matrix, vector, sym_pos=sym_pos)
    return solution
