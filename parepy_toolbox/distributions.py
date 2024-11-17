"""List of distributions for the toolbox"""
import numpy as np

def crude_sampling_zero_one(n_samples: int, seed: int=None) -> list:
    """
    This function generates a uniform sampling between 0 and 1.

    Args:
        n_samples (int): Number of samples
        seed (int): Seed for random number generation

    Returns:
        u (list): Random samples
    """
    rng = np.random.default_rng(seed=seed)
    
    return rng.random(n_samples).tolist()


def lhs_sampling_zero_one(n_samples: int, dimension: int, seed: int=None) -> np.ndarray:
    """
    This function generates a uniform sampling between 0 and 1 using Latin Hypercube Sampling Algorithm.

    Args:
        n_samples (int): Number of samples
        dimension (int): Number of dimensions
        seed (int): Seed for random number generation

    Returns:
        u (np.array): Random samples
    """
    r = np.zeros((n_samples, dimension))
    p = np.zeros((n_samples, dimension))
    original_ids = [i for i in range(1, n_samples+1)]
    if seed is not None:
        x = crude_sampling_zero_one(n_samples * dimension, seed)
    else:
        x = crude_sampling_zero_one(n_samples * dimension)
    for i in range(dimension):
        perms = original_ids.copy()
        r[:, i] = x[:n_samples]
        del x[:n_samples]
        if i == 0:
            p[:, i] = perms.copy()
        else:
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(perms)
            p[:, i] = perms.copy()
    u = (p - r) * (1 / n_samples)

    return u


def uniform_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a uniform sampling between a and b.

    Args:
        parameters (dict): Dictionary of parameters. Keys 'a' (min. value [float]), 'b' (max. value [float])
        method (str): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (int): Number of samples
        seed (int): Seed for random number generation
    
    Returns:
        u (list): Random samples
    """

    # Random uniform sampling between 0 and 1
    if method.lower() == 'mcs':
        if seed is not None:
            u_aux = crude_sampling_zero_one(n_samples, seed)
        elif seed is None:
            u_aux = crude_sampling_zero_one(n_samples)
    elif method.lower() == 'lhs':
        if seed is not None:
            u_aux = lhs_sampling_zero_one(n_samples, 1, seed).flatten()
        elif seed is None:
            u_aux = lhs_sampling_zero_one(n_samples, 1).flatten()
    
    # PDF parameters and generation of samples    
    a = parameters['a']
    b = parameters['b']
    u = [float(a + (b - a) * i) for i in u_aux]

    return u


def normal_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a normal sampling with mean mu and standard deviation sigma.

    Args:
        parameters (dict): Dictionary of parameters. Keys 'mu' (mean [float]), 'sigma' (standard deviation [float])
        method (str): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (int): Number of samples
        seed (int): Seed for random number generation
    
    Returns:
        u (list): Random samples
    """

    # Random uniform sampling between 0 and 1
    if method.lower() == 'mcs':
        if seed is not None:
            u_aux1 = crude_sampling_zero_one(n_samples, seed)
            u_aux2 = crude_sampling_zero_one(n_samples, seed+1)
        elif seed is None:
            u_aux1 = crude_sampling_zero_one(n_samples)
            u_aux2 = crude_sampling_zero_one(n_samples)
    elif method.lower() == 'lhs':
        if seed is not None:
            u_aux1 = lhs_sampling_zero_one(n_samples, 2, seed)
        elif seed is None:
            u_aux1 = lhs_sampling_zero_one(n_samples, 2)

    # PDF parameters and generation of samples  
    mean = parameters['mean']
    std = parameters['sigma']
    u = []
    for i in range(n_samples):
        if method.lower() == 'lhs':
            z = float(np.sqrt(-2 * np.log(u_aux1[i, 0])) * np.cos(2 * np.pi * u_aux1[i, 1]))
        elif method.lower() == 'mcs':
            z = float(np.sqrt(-2 * np.log(u_aux1[i])) * np.cos(2 * np.pi * u_aux2[i]))
        u.append(mean + std * z)

    return u


def gumbel_right_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a normal sampling with mean mu and standard deviation sigma.

    Args:
        parameters (dict): Dictionary of parameters. Keys 'mu' (mean [float]), 'sigma' (standard deviation [float])
        method (str): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (int): Number of samples
        seed (int): Seed for random number generation
    
    Returns:
        u (list): Random samples
    """

    # Random uniform sampling between 0 and 1
    if method.lower() == 'mcs':
        if seed is not None:
            u_aux = crude_sampling_zero_one(n_samples, seed)
        elif seed is None:
            u_aux = crude_sampling_zero_one(n_samples)
    elif method.lower() == 'lhs':
        if seed is not None:
            u_aux = lhs_sampling_zero_one(n_samples, 1, seed)
        elif seed is None:
            u_aux = lhs_sampling_zero_one(n_samples, 1)

    # PDF parameters and generation of samples  
    mean = parameters['mean']
    std = parameters['sigma']
    gamma = 0.577216
    beta = np.sqrt(6) * std / np.pi
    mu_n = mean - beta * gamma
    u = []
    for i in range(n_samples):
        u.append(mu_n - beta * np.log(-np.log(u_aux[i])))


    return u