"""Example linear algebra file."""

import numpy as np


def matmul(A: np.ndarray, B: np.ndarray):
    """Main entrypoint. Fill this function in."""

    # Uncomment to pass the test
    return A @ B

    # Comment the following to suppress the error
    # raise NotImplementedError(
    #     "`matmul` function in `linalg.py` needs to be implemented"
    # )

if __name__ == "__main__":
    A = 2 * np.eye(2)
    B = 4 * np.eye(2)

    output = matmul(A, B)
