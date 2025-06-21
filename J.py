import numpy as np

def J(theta):
    """
    Approximated J-function for AWGN-PEXIT curves.
    
    Parameters
    ----------
    theta : scalar or array-like
        σ (or √(variance)) values for which to compute J(θ).
    
    Returns
    -------
    j : ndarray
        Mutual-information approximation, same shape as `theta`.
    """
    theta   = np.asarray(theta, dtype=float)
    j       = np.zeros_like(theta)

    theta_a = 1.6363
    # coefficients (identical to your MATLAB code)
    aj1, bj1, cj1 = -0.0421061, 0.209252,  -0.00640081
    aj2, bj2, cj2 =  0.00181491, -0.142675, -0.0822054
    dj2           =  0.0549608

    # region 1 : 0 ≤ θ < θ_a
    mask1 = (theta >= 0.0) & (theta < theta_a)
    j[mask1] = (
        aj1 * theta[mask1]**3
        + bj1 * theta[mask1]**2
        + cj1 * theta[mask1]
    )

    # region 2 : θ_a < θ < 10
    mask2 = (theta > theta_a) & (theta < 10.0)
    j[mask2] = 1.0 - np.exp(
        aj2 * theta[mask2]**3
        + bj2 * theta[mask2]**2
        + cj2 * theta[mask2]
        + dj2
    )

    # region 3 : θ ≥ 10
    j[theta >= 10.0] = 1.0

    return j
