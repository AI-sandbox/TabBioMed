import numpy as np


def pareto_mask_min_x_max_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                keep[i] = False
                break
    return keep
