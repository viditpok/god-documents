import random

import numpy as np

from macrotones.utils.seed import set_global_seed


def test_set_global_seed_reproducible() -> None:
    set_global_seed(1234)
    python_first = random.random()
    numpy_first = np.random.rand()

    set_global_seed(1234)
    python_second = random.random()
    numpy_second = np.random.rand()

    assert python_first == python_second
    assert numpy_first == numpy_second
