import numpy as np
import seaborn as sns

from scipy.stats import norm

def importance_sampling(num_simulations, num_samples, verbose=True, plot=False):
    
    if verbose:
        print("===================================================")
        print("{} Importance Sampling Simulations of size {}".format(num_simulations, num_samples))
        print("===================================================\n")
    
    num_simulations = int(num_simulations)
    num_samples = int(num_samples)
    
    probas = []
    for i in range(num_simulations):
        mu_1, sigma_1 = 0, 0.01
        mu_2, sigma_2 = 0, 0.03
        mu_1_n, sigma_1_n = 0, 0.02
        mu_2_n, sigma_2_n = 0, 0.06
        
        # setup pdfs
        old_pdf_1 = norm(mu_1, sigma_1)
        new_pdf_1 = norm(mu_1_n, sigma_1_n)
        old_pdf_2 = norm(mu_2, sigma_2)
        new_pdf_2 = norm(mu_2_n, sigma_2_n)

        length = np.random.normal(mu_1_n, sigma_1_n, num_samples)
        voltage = np.random.normal(mu_2_n, sigma_2_n, num_samples)
        
        print("length", length)
        print("voltage", voltage)

        # calculate current
        num = 50 * np.square((0.6 - voltage))
        denum = 0.1 + length
        I = num / denum

        print("current", I)

        # calculate f
        true_condition = np.where(I >= 275)

        # calculate weight
        num = old_pdf_1.pdf(length) * old_pdf_2.pdf(voltage)
        denum = new_pdf_1.pdf(length) * new_pdf_2.pdf(voltage)
        weights = num / denum

        print("num", num)
        print("denum", denum)
        print("weights", weights)

        # select weights for nonzero f
        weights = weights[true_condition]

        # compute unbiased proba
        proba = np.sum(weights) / num_samples
        probas.append(proba)
        
        false_condition = np.where(I < 275)
    
    
    mean_proba = np.mean(probas)
    std_proba = np.std(probas)
    
    if verbose:
        print("Probability Mean: {}".format(mean_proba))
        print("Probability Std: {}".format(std_proba))
    
    return probas


probas = importance_sampling(1, 10000, plot=False)

