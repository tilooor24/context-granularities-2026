import numpy as np
from typing import List, cast


def calculate_pass_at_k(k: int, scores: List[int]):
    n = len(scores)
    c = sum(1 for score in scores if score == 1)

    if c == 0:
        pass_at_k = 0.0
    elif n - c < k:
        pass_at_k = 1.0
    else:
        pass_at_k = 1.0 - cast(
            float,
            np.prod(1.0 - k / np.arange(n - c + 1, n + 1)).item(),
        )

    return pass_at_k
