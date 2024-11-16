"""List of distributions for the toolbox"""
import random

import numpy as np

def crude_sampling_zero_one(n_samples: int, seed: int) -> list:
    """
    This function generates a uniform sampling between 0 and 1.

    Args:
        n_samples (int): Number of samples
        seed (int): Seed for random number generation

    Returns:
        u (list): list of random samples
    """
    np.random.seed(seed)
    
    return np.random.uniform(0, 1, n_samples).tolist()


def lhs_sampling_zero_one(n_samples: int, dimension: int, seed: int) -> np.ndarray:
    """
    This function generates a uniform sampling between 0 and 1 using Latin Hypercube Sampling Algorithm.

    Args:
        n_samples (int): Number of samples
        dimension (int): Number of dimensions
        seed (int): Seed for random number generation

    Returns:
        u (np.array): Array of random samples
    """
    np.random.seed(seed)
    r = np.zeros((n_samples, dimension))
    p = np.zeros((n_samples, dimension))
    original_ids = [i for i in range(1, n_samples+1)]
    for i in range(dimension):
        r[:, i] = np.random.uniform(0, 1, n_samples) * (1 / n_samples)
        permuts = original_ids.copy()
        random.shuffle(permuts)
        if i == 0:
            p[:, i] = [i for i in range(1, n_samples + 1)]
        else:
            p[:, i] = permuts.copy()

    p = p * (1 / n_samples)
    u = p - r

    return u