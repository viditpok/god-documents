"""Example of unit tests."""
import numpy as np
import torch
import torch.nn as nn

from vision.linalg import matmul


def test_matmul() -> None:
    """Delete the assert statement and the uncomment the correct code."""
    A = 2 * np.eye(2)
    B = 4 * np.eye(2)

    output = matmul(A, B)

    expected_output = np.array([[8.0, 0.0], [0.0, 8.0]])
    assert np.allclose(output, expected_output)

def test_pytorch() -> None:
    """Ensure that this test passes on your local machine using `pytest tests`"""
    
    input_example = torch.zeros(1,1,3,3)

    # set the middle pixel to be a one (the rest are zeros)
    input_example[:,:,1,1] = 1.

    conv = nn.Conv2d(1, 5, kernel_size=3, bias=False)
    
    # set the conv weights to be five 3x3 identity matrices
    conv.weight.data = torch.eye(3,3).unsqueeze(0).repeat(5,1,1,1)
    result = conv(input_example).detach()
    
    expected = torch.ones((1, 5, 1, 1))
    
    assert expected.size() == result.size()
    assert np.allclose(expected, result, atol=1e-2)
