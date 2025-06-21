import numpy as np

def J_inv(I):
    """
    Approximate inverse J-function (θ as a function of mutual information I).

    Parameters
    ----------
    I : scalar or array-like
        Mutual–information values in the range [0, 1].

    Returns
    -------
    theta : ndarray
        Approximated σ-values (same shape as I) such that  J(theta) ≈ I.
    """
    I = np.asarray(I, dtype=float)
    theta = np.zeros_like(I)

    # breakpoint and polynomial / logarithmic coefficients
    I_a  = 0.3646
    at1, bt1, ct1 = 1.09542, 0.214217,  2.33727   # region 1
    at2, bt2, ct2 = 0.706692, 0.386013, -1.75017  # region 2

    # region-1 : 0 ≤ I ≤ I_a   (quadratic + sqrt fit)
    mask1 = (I >= 0.0) & (I <= I_a)
    theta[mask1] = (
        at1 * I[mask1]**2
        + bt1 * I[mask1]
        + ct1 * np.sqrt(I[mask1])
    )

    # region-2 : I_a < I < 1   (logarithmic fit)
    mask2 = (I > I_a) & (I < 1.0)
    theta[mask2] = -at2 * np.log(bt2 * (1.0 - I[mask2])) - ct2 * I[mask2]

    # region-3 : I == 1  → saturate at a large value (1000 in MATLAB)
    theta[I >= 1.0] = 1000.0

    return theta
