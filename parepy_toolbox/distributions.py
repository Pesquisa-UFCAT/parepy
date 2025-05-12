"""Function of probability distributions"""
import numpy as np
from scipy.stats import norm


def crude_sampling_zero_one(n_samples: int, seed: int=None) -> list:
    """
    Generates a uniform sampling between 0 and 1.

    :param n_samples: Number of samples.
    :param seed: Seed for random number generation.

    :return: List of random samples.
    """
    rng = np.random.default_rng(seed=seed)

    return rng.random(n_samples).tolist()


def lhs_sampling_zero_one(n_samples: int, dimension: int, seed: int=None) -> np.ndarray:
    """
    Generates a uniform sampling between 0 and 1 using the Latin Hypercube Sampling algorithm.

    :param n_samples: Number of samples.
    :param dimension: Number of dimensions.
    :param seed: Seed for random number generation.

    :return: Array of random samples.
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
    Generates a uniform sampling between a minimum (a) and maximum (b) value.

    :param parameters: Dictionary of parameters, including:

        - 'min': Minimum value of the uniform distribution.
        - 'max': Maximum value of the uniform distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation. Use None for a random seed.

    :return: List of random samples.
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
    Generates a normal (Gaussian) sampling with specified mean (mu) and standard deviation (sigma).

    :param parameters: Dictionary of parameters, including:
    
        - 'mu': Mean of the normal distribution.
        - 'sigma': Standard deviation of the normal distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation. Use None for a random seed.

    :return: List of random samples.
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
    Generates a normal (Gaussian) sampling with specified mean (mu) and standard deviation (sigma).

    Variable g has a correlation `rho_gb` with b.

    :param parameters: Dictionary of parameters, including:

        - 'mu': Mean of the normal distribution.
        - 'sigma': Standard deviation of the normal distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation. Use None for a random seed.

    :return: List of random samples.
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
    Generates a log-normal sampling with specified mean and standard deviation.

    :param parameters: Dictionary of parameters, including:

        - 'mu': Mean of the underlying normal distribution.
        - 'sigma': Standard deviation of the underlying normal distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation.

    :return: List of random samples.
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
    Generates a Gumbel maximum distribution with specified mean and standard deviation.

    :param parameters: Dictionary of parameters, including:

        - 'mu': Mean of the Gumbel distribution.
        - 'sigma': Standard deviation of the Gumbel distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation.

    :return: List of random samples.
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
    Generates a Gumbel minimum distribution with specified mean and standard deviation.

    :param parameters: Dictionary of parameters, including:

        - 'mu': Mean of the Gumbel distribution.
        - 'sigma': Standard deviation of the Gumbel distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation.

    :return: List of random samples.
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
    Generates a triangular sampling with minimum a, mode c, and maximum b.

    :param parameters: Dictionary of parameters, including:

        - 'a': Minimum value of the distribution.
        - 'c': Mode (most likely value) of the distribution.
        - 'b': Maximum value of the distribution.

    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling) or 'mcs' (Crude Monte Carlo Sampling).
    :param n_samples: Number of samples.
    :param seed: Seed for random number generation.

    :return: List of random samples.
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


def cdf_gumbel_max(x: float, u: float, beta: float) -> float:
    """
    Calculates the cumulative distribution function (CDF) of the Maximum Gumbel distribution.

    :param x: Input value for which the CDF will be calculated.
    :param u: Location parameter (mode) of the Maximum Gumbel distribution.
    :param beta: Scale parameter of the Maximum Gumbel distribution.

    :return: Value of the CDF at point x.
    """
    fx = np.exp(-np.exp((- beta * (x - u))))
    return fx


def pdf_gumbel_max(x: float, u: float, beta: float) -> float:
    """
    Calculates the probability density function (PDF) of the Maximum Gumbel distribution.

    :param x: Input value for which the PDF will be calculated.
    :param u: Location parameter (mode) of the Maximum Gumbel distribution.
    :param beta: Scale parameter of the Maximum Gumbel distribution.

    :return: Value of the PDF at point x.
    """
    fx = beta * np.exp((- beta * (x - u))) - np.exp((- beta * (x - u)))
    return fx


def cdf_gumbel_min(x: float, u: float, beta: float) -> float:
    """
    Calculates the cumulative distribution function (CDF) of the Minimum Gumbel distribution.

    :param x: Input value for which the CDF will be calculated.
    :param u: Location parameter (mode) of the Minimum Gumbel distribution.
    :param beta: Scale parameter of the Minimum Gumbel distribution.

    :return: Value of the CDF at point x.
    """
    fx = 1 - np.exp(- np.exp((beta * (x - u))))
    return fx


def pdf_gumbel_min(x: float, u: float, beta: float) -> float:
    """
    Calculates the probability density function (PDF) of the Minimum Gumbel distribution.

    :param x: Input value for which the PDF will be calculated.
    :param u: Location parameter (mode) of the Minimum Gumbel distribution.
    :param beta: Scale parameter of the Minimum Gumbel distribution.

    :return: Value of the PDF at point x.
    """
    fx = beta * np.exp((beta * (x - u))) - np.exp(beta * (x - u))
    return fx


def cdf_normal(x: float, u: float, sigma: float) -> float:
    """
    Calculates the cumulative distribution function (CDF) of the Normal distribution.

    :param x: Input value for which the CDF will be calculated.
    :param u: Mean (location) of the Normal distribution.
    :param sigma: Standard deviation (scale) of the Normal distribution.

    :return: Value of the CDF at point x.
    """
    fx = norm.cdf(x, loc=u, scale=sigma)
    return fx


def pdf_normal(x: float, u: float, sigma: float) -> float:
    """
    Calculates the probability density function (PDF) of the Normal distribution.

    :param x: Input value for which the PDF will be calculated.
    :param u: Mean (location) of the Normal distribution.
    :param sigma: Standard deviation (scale) of the Normal distribution.

    :return: Value of the PDF at point x.
    """
    fx = norm.pdf(x, loc=u, scale=sigma)
    return fx


def log_normal(x: float, lambdaa: float, epsilon: float) -> tuple[float, float]:
    """
    Calculates the location (u) and scale (sigma) parameters for a Log-Normal distribution.

    :param x: Input value.
    :param lambdaa: Shape parameter of the Log-Normal distribution.
    :param epsilon: Scale parameter of the Log-Normal distribution.

    :return: Tuple containing:
        - u: Location parameter.
        - sigma: Scale parameter.
    """
    loc = x * (1 - np.log(x) + lambdaa)
    sigma = x * epsilon
    return loc, sigma


def non_normal_approach_normal(x, dist, params):
    """
    Converts a non-normal distribution to an equivalent normal distribution.

    :param x: Random variable.
    :param dist: Type of distribution. Supported values: 'gumbel max', 'gumbel min', 'lognormal'.
    :param params: Dictionary of distribution parameters, depending on the selected distribution type.

        - For 'gumbel max' or 'gumbel min': {'mu': location, 'sigma': scale}
        - For 'lognormal': {'lambda': shape, 'epsilon': scale}

    :return: Tuple containing:
        - mu_t: Mean of the equivalent normal distribution.
        - sigma_t: Standard deviation of the equivalent normal distribution.
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
