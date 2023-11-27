"""Eigendecomposition functions."""

import numpy as np


def get_matrix_from_eigdec(e: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Restore the original square symmetric matrix from eigenvalues and eigenvectors after eigenvalue decomposition.

    Args:
        e: The vector of eigenvalues with shape (N).
        V: The matrix with eigenvectors as columns with shape (N, N).

    Returns:
        The original matrix used for eigenvalue decomposition with shape (N, N)
    """
    # START TODO #################
    # Transform the eigenvalues into a diagonal matrix of shape (N, N)
    # uppercase_lamba = np.diag(e)
    # Transpose V
    # transposed_v = np.transpose(V)
    # Do the matrix multiplication
    # return np.matmul(np.matmul(V, uppercase_lamba), transposed_v)
    # Do it in 1 line
    return np.matmul(np.matmul(V, np.diag(e)), np.transpose(V))
    # END TODO ###################


def get_euclidean_norm(v: np.ndarray) -> float:
    """Compute the euclidean norm of a vector.

    Args:
        v: The input vector with shape (N).

    Returns:
        The euclidean norm of the vector.
    """
    # START TODO #################
    # Do not use np.linalg.norm, otherwise you will get no points.
    return np.sqrt(np.square(v).sum())
    # END TODO ###################


def get_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute dot product of two vectors.

    Args:
        v1: First input vector with shape (N)
        v2: Second input vector with shape (N)

    Returns:
        Dot product result.
    """
    assert (
        len(v1.shape) == len(v2.shape) == 1 and v1.shape == v2.shape
    ), f"Input vectors must be 1-dimensional and have the same shape, but have shapes {v1.shape} and {v2.shape}"
    # START TODO #################
    return np.multiply(v1, v2).sum()
    # END TODO ###################


def get_inverse(e: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute the inverse of a square symmetric matrix A given its eigenvectors and eigenvalues.

    Args:
        e: The vector of eigenvalues with shape (N).
        V: The matrix with eigenvectors as columns with shape (N, N).

    Returns:
        The inverse of A (i.e. the matrix with given eigenvalues/vectors) with shape (N, N).
    """
    # START TODO #################
    # Do not use np.linalg.inv, otherwise you will get no points.
    # create the diagonal matrix with the inverse
    inverse_lambda = np.diag(1 / e)
    # compute the inverse A given then formula
    return np.matmul(np.matmul(V, inverse_lambda), V.T)
    # END TODO ###################
