# pylint: disable=redefined-outer-name
"""
Test the numpy and scipy.sparse backend.
"""
import pytest
import numpy as np
import numpy.testing as npt
import scipy as sp

from .. import backend as bknd


@pytest.fixture
def matrix():
    "A sample matrix"
    return np.array([[0, 3, 5],
                     [2, 5, 0],
                     [2, 0, 8]])


def test_inv_numpy(matrix):
    "Calculate the inverse of a numpy array."
    size = matrix.shape[0]
    inv = bknd.inv(matrix)
    npt.assert_allclose(matrix.dot(inv), np.identity(size), atol=1e-10)


def test_inv_sparse(matrix):
    "Calculate the inverse of a sparse matrix."
    size = matrix.shape[0]
    spmatrix = sp.sparse.csc_matrix(matrix)
    inv = bknd.inv(spmatrix)
    npt.assert_allclose(spmatrix.dot(inv).todense(), np.identity(size),
                        atol=1e-10)


def test_solve_numpy(matrix):
    "Solve a linear system with a numpy array matrix"
    true_solution = np.array([5, 6, 7])
    rh_side = matrix.dot(true_solution)
    solution = bknd.solve(matrix, rh_side)
    npt.assert_allclose(matrix.dot(solution), rh_side)
    npt.assert_allclose(true_solution, solution)


def test_solve_sparse(matrix):
    "Solve a linear system with a sparse matrix"
    spmatrix = sp.sparse.csc_matrix(matrix)
    true_solution = np.array([5, 6, 7])
    rh_side = spmatrix.dot(true_solution)
    solution = bknd.solve(spmatrix, rh_side)
    npt.assert_allclose(spmatrix.dot(solution), rh_side)
    npt.assert_allclose(true_solution, solution)


def test_multiply(matrix):
    "Element-wise multiplication using numpy arrays and scipy.sparse matrices"
    true_solution = np.array([[0, 6, 15],
                              [-2, 10, 0],
                              [-2, 0, 24]])
    vector = np.array([-1, 2, 3])

    solution = bknd.multiply(matrix, vector)
    npt.assert_allclose(solution, true_solution)

    spmatrix = sp.sparse.csc_matrix(matrix)
    # Left multiply
    sp_solution = bknd.multiply(spmatrix, vector)
    npt.assert_allclose(sp_solution.todense(), true_solution)
    # Right multiply
    rsp_solution = bknd.multiply(vector, spmatrix)
    npt.assert_allclose(rsp_solution.todense(), true_solution)
