"""Function of probability distributions"""
import numpy as np
from scipy.stats import norm

def crude_sampling_zero_one(n_samples: int, seed: int=None) -> list:
    """
    This function generates a uniform sampling between 0 and 1.

    Args:
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation

    Returns:
        u (List): Random samples
    """
    rng = np.random.default_rng(seed=seed)

    return rng.random(n_samples).tolist()


def lhs_sampling_zero_one(n_samples: int, dimension: int, seed: int=None) -> np.ndarray:
    """
    This function generates a uniform sampling between 0 and 1 using the Latin Hypercube Sampling Algorithm.

    Args:
        n_samples (Integer): Number of samples
        dimension (Integer): Number of dimensions
        seed (Integer): Seed for random number generation

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
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(perms)
        p[:, i] = perms.copy()
    u = (p - r) * (1 / n_samples)

    return u


def uniform_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a Uniform sampling between a (minimum) and b (maximum).

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys:  'min' (Minimum value of the uniform distribution [float]), 'max' (Maximum value of the uniform distribution [float])
        method (String): Sampling method. Supports the following values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation. Use None for a random seed
    
    Returns:
        u (List): Random samples
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
    a = parameters['min']
    b = parameters['max']
    u = [float(a + (b - a) * i) for i in u_aux]

    return u


def normal_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a Normal or Gaussian sampling with mean (mu) and standard deviation (sigma).

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys 'mu' (Mean [float]), 'sigma' (Standard deviation [float])
        method (String): Sampling method. Supports the following values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation. Use None for a random seed
    
    Returns:
        u (List): Random samples
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


def corr_normal_sampling(parameters_b: dict, parameters_g: dict, pho_gb: float, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a Normal or Gaussian sampling with mean (mu) and standard deviation (sigma). Variable g have a correlation rho_gb with b.

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys 'mu' (Mean [float]), 'sigma' (Standard deviation [float])
        method (String): Sampling method. Supports the following values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation. Use None for a random seed
    
    Returns:
        u (List): Random samples
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
    mean_b = parameters_b['mean']
    std_b = parameters_b['sigma']
    mean_g = parameters_g['mean']
    std_g = parameters_g['sigma']
    b = []
    g = []
    for i in range(n_samples):
        if method.lower() == 'lhs':
            z_1 = float(np.sqrt(-2 * np.log(u_aux1[i, 0])) * np.cos(2 * np.pi * u_aux1[i, 1]))
            z_2 = float(np.sqrt(-2 * np.log(u_aux1[i, 0])) * np.sin(2 * np.pi * u_aux1[i, 1]))
        elif method.lower() == 'mcs':
            z_1 = float(np.sqrt(-2 * np.log(u_aux1[i])) * np.cos(2 * np.pi * u_aux2[i]))
            z_2 = float(np.sqrt(-2 * np.log(u_aux1[i])) * np.sin(2 * np.pi * u_aux2[i]))
        b.append(mean_b + std_b * z_1)
        g.append(mean_g + std_g * (pho_gb * z_1 + z_2 * np.sqrt(1 - pho_gb ** 2)))

    return b, g


def lognormal_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a log-normal sampling with mean and standard deviation.

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys 'mu' (mean [float]), 'sigma' (standard deviation [float])
        method (String): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation
    
    Returns:
        u (List): Random samples
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
    epsilon = np.sqrt(np.log(1 + (std/mean)**2))
    lambdaa = np.log(mean) - 0.5 * epsilon**2
    u = []
    for i in range(n_samples):
        if method.lower() == 'lhs':
            z = float(np.sqrt(-2 * np.log(u_aux1[i, 0])) * np.cos(2 * np.pi * u_aux1[i, 1]))
        elif method.lower() == 'mcs':
            z = float(np.sqrt(-2 * np.log(u_aux1[i])) * np.cos(2 * np.pi * u_aux2[i]))
        u.append(np.exp(lambdaa + epsilon * z))

    return u


def gumbel_max_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a Gumbel maximum distribution with a specified mean and standard deviation.

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys 'mu' (mean [float]), 'sigma' (standard deviation [float])
        method (String): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation
    
    Returns:
        u (List): Random samples
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
    mean = parameters['mean']
    std = parameters['sigma']
    gamma = 0.577215665
    beta = np.pi / (np.sqrt(6) * std)
    alpha = mean - gamma / beta
    u = []
    for i in range(n_samples):
        u.append(alpha - (1 / beta) * np.log(-np.log(u_aux[i])))

    return u


def gumbel_min_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a Gumbel Minimum sampling with mean and standard deviation.

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys 'mu' (mean [float]), 'sigma' (standard deviation [float])
        method (String): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation
    
    Returns:
        u (List): Random samples
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
    mean = parameters['mean']
    std = parameters['sigma']
    gamma = 0.577215665
    beta = np.pi / (np.sqrt(6) * std) 
    alpha = mean + gamma / beta
    u = []
    for i in range(n_samples):
        u.append(alpha + (1 / beta) * np.log(-np.log(1 - u_aux[i])))

    return u


def triangular_sampling(parameters: dict, method: str, n_samples: int, seed: int=None) -> list:
    """
    This function generates a triangular sampling with minimun a, mode c, and maximum b.

    Args:
        parameters (Dictionary): Dictionary of parameters. Keys 'a' (minimum [float]), 'c' (mode [float]), and 'b' (maximum [float])
        method (String): Sampling method. Can use 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling)
        n_samples (Integer): Number of samples
        seed (Integer): Seed for random number generation
    
    Returns:
        u (List): Random samples
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
    a = parameters['min']
    c = parameters['mode']
    b = parameters['max']
    u = []
    for i in range(n_samples):
        criteria = (c - a) / (b - a)
        if u_aux[i] < criteria:
            u.append(a + np.sqrt(u_aux[i] * (b - a) * (c - a)))
        else:
            u.append(b - np.sqrt((1 - u_aux[i]) * (b - a) * (b - c)))

    return u


def cdf_gumbel_max(x, u, beta):
    """
    This function generates the CDF of Gumbel maximum distribution.

    Args:
        x (float): Random value.
        u (float): Mode.
        beta (float): Scale.

    Retorns:
        fx (float): CDF in point x.
    """
    fx = np.exp(-np.exp((- beta * (x - u))))
    return fx


def pdf_gumbel_max(x, u, beta):
    """
    Calcula a função densidade de probabilidade (PDF) da distribuição Gumbel Máxima.

    Parâmetros:
        x (float): Valor de entrada para o qual a PDF será calculada.
        u (float): Parâmetro de locação (moda) da distribuição Gumbel Máxima.
        beta (float): Parâmetro de escala da distribuição Gumbel Máxima.

    Retorna:
        float: Valor da PDF no ponto x.
    """
    fx = beta * np.exp((- beta * (x - u))) - np.exp((- beta * (x - u)))
    return fx


def cdf_gumbel_min(x, u, beta):
    """
    Calcula a função de distribuição acumulada (CDF) da distribuição Gumbel Mínima.

    Parâmetros:
        x (float): Valor de entrada para o qual a CDF será calculada.
        u (float): Parâmetro de locação (moda) da distribuição Gumbel Mínima.
        beta (float): Parâmetro de escala da distribuição Gumbel Mínima.

    Retorna:
        float: Valor da CDF no ponto x.
    """
    fx = 1 - np.exp(- np.exp((beta * (x - u))))
    return fx


def cdf_normal(x, u, sigma):
    """
    Calcula a função de distribuição acumulada (CDF) da distribuição Normal.

    Parâmetros:
        x (float): Valor de entrada para o qual a CDF será calculada.
        u (float): Média (locação) da distribuição Normal.
        sigma (float): Desvio padrão (escala) da distribuição Normal.

    Retorna:
        float: Valor da CDF no ponto x.
    """
    fx = norm.cdf(x, loc=u, scale=sigma)
    return fx


def pdf_gumbel_min(x, u, beta):
    """
    Calcula a função densidade de probabilidade (PDF) da distribuição Gumbel Mínima.

    Parâmetros:
        x (float): Valor de entrada para o qual a PDF será calculada.
        u (float): Parâmetro de locação (moda) da distribuição Gumbel Mínima.
        beta (float): Parâmetro de escala da distribuição Gumbel Mínima.

    Retorna:
        float: Valor da PDF no ponto x.
    """
    fx = beta * np.exp((beta * (x - u))) - np.exp(beta * (x - u))
    return fx


def pdf_normal(x, u, sigma):
    """
    Calcula a função densidade de probabilidade (PDF) da distribuição Normal.

    Parâmetros:
        x (float): Valor de entrada para o qual a PDF será calculada.
        u (float): Média (locação) da distribuição Normal.
        sigma (float): Desvio padrão (escala) da distribuição Normal.

    Retorna:
        float: Valor da PDF no ponto x.
    """
    fx = norm.pdf(x, loc=u, scale=sigma)
    return fx


def log_normal(x, lambdaa, epsilon):
    """
    Calcula os parâmetros de locação (u) e escala (sigma) para uma distribuição Log-Normal.

    Parâmetros:
        x (float): Valor de entrada.
        lamb (float): Parâmetro de forma da distribuição Log-Normal.
        epsilon (float): Parâmetro de escala da distribuição Log-Normal.

    Retorna:
        tuple: (u, sigma), onde:
            - u (float): Parâmetro de locação.
            - sigma (float): Parâmetro de escala.
    """
    loc = x * (1 - np.log(x) + lambdaa)
    sigma = x * epsilon
    return loc, sigma


def non_normal_approach_normal(x, dist, params):
    """
    This function convert non normal distribution to normal distribution.

    Parâmetros:
        x (Float): Random variable
        dist (String): Type of distribution: 'gumbel max', 'gumbel min', 'lognormal')
        params (Dictionary): Parameters of distribution

    Retorna:
        mu_t (Float): mean normal model
        sigma_t (Float): standard deviation normal model
    """

    if dist == 'gumbel max':
        u = params.get('u')
        beta = params.get('beta')
        cdf_x = cdf_gumbel_max(x, u, beta)
        pdf_temp = pdf_gumbel_max(x, u, beta)
    elif dist == 'gumbel min':
        u = params.get('u')
        beta = params.get('beta')
        cdf_x = cdf_gumbel_min(x, u, beta)
        pdf_temp = pdf_gumbel_min(x, u, beta)
    
    if dist == 'lognormal':
        epsilon = params.get('epsilon')
        lambdaa = params.get('lambda')
        loc_eq, sigma_eq = log_normal(x, lambdaa, epsilon)
    else:
        icdf = norm.ppf(cdf_x, loc=0, scale=1)
        sigma_eq = norm.pdf(icdf, loc=0, scale=1) / pdf_temp
        loc_eq = x - sigma_eq * icdf

    return float(loc_eq), float(sigma_eq)

