import numpy as np


def hadamard(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiply each element with the same index from both 
    arrays and return an array of the same size.

    Args:
        (numpy.ndarray) u: First operand.
        (numpy.ndarray) v: Second operand.

    Returns:
        (np.ndarray): Binary operation results.
    """

    return u * v


def l1(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Subtract each element with the same index from both 
    arrays and return an array of the same size and absolute values.

    Args:
        (numpy.ndarray) u: First operand.
        (numpy.ndarray) v: Second operand.

    Returns:
        (np.ndarray): Binary operation results.
    """

    return np.abs(u - v)


def l2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Subtract each element with the same index from both 
    arrays and return an array of the same size and squared values.

    Args:
        (numpy.ndarray) u: First operand.
        (numpy.ndarray) v: Second operand.

    Returns:
        (np.ndarray): Binary operation results.
    """

    return (u - v) ** 2


def avg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Add each element with the same index from both 
    arrays and return an array of the same size and halved values.

    Args:
        (numpy.ndarray) u: First operand.
        (numpy.ndarray) v: Second operand.

    Returns:
        (np.ndarray): Binary operation results.
    """
    
    return (u + v) / 2.0
