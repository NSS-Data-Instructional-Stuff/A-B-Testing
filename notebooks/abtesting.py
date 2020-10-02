from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np

def power(baseline, min_effect, sample_size, kind = 'absolute'):
    '''
    Compute the power for a two-sided hypothesis test.
      
    Parameters
    ----------
    baseline : float
        The baseline rate, as a decimal.
    min_effect : float
        The minimum detectable effect, as a decimal.
    sample_size: int
        The number of observations per variant.
    kind : str in ['absolute', 'relative']
        Whether the minimum effect is relative or absolute.
        
    Returns
    -------
    power: float
        The power for the hypothesis test.
    
    '''
    p1 = baseline
    if kind == 'absolute':
        p2 = baseline + min_effect
    elif kind == 'relative':
        p2 = baseline + min_effect*baseline
    
    mu_null = 0
    sigma_null = np.sqrt(p1*(1-p1) / sample_size + p1*(1-p1) / sample_size)

    mu_alt = p2 - p1
    sigma_alt = np.sqrt(p1*(1-p1) / sample_size + p2*(1-p2) / sample_size)
    
    min_reject = norm.ppf(loc = mu_null, scale = sigma_null, q = 0.975)
    
    power = 1 - norm.cdf(min_reject, loc = mu_alt, scale = sigma_alt) - norm.cdf(-min_reject, loc = mu_alt, scale = sigma_alt)
    
    return power

def min_sample_size(baseline, min_effect, kind = 'absolute', desired_power = 0.8):
    '''
    Compute the minimum sample size needed for a desired level of power.
      
    Parameters
    ----------
    baseline : float
        The baseline rate, as a decimal.
    min_effect : float
        The minimum detectable effect, as a decimal.
    kind : str in ['absolute', 'relative']
        Whether the minimum effect is relative or absolute.
    desired_power: float
        The desired power level, as a decimal between 0 and 1.
        
    Returns
    -------
    Sample size needed, as an int.
    
    '''
    return int(fsolve(func = lambda x: (power(baseline = baseline, min_effect = min_effect, sample_size = x, kind = kind) - desired_power )**2, x0 = 1000)[0])
