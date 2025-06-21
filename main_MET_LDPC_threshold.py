# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 00:18:51 2025

@author: elahe
"""

import numpy as np
from pexit import pexit  # assumes you already implemented and imported this

def MET_LDPC_threshold(S=None, R=None, pun=None, iterations=250, samples=25):
    """
    Estimates the iterative decoding threshold for a protograph LDPC code using PEXIT.

    Parameters
    ----------
    S : ndarray (optional)
        Base matrix of the protograph. Default: MET example from paper.
    R : float (optional)
        Code rate. If not given, computed from S and puncturing.
    pun : list of int (optional)
        List of punctured variable node indices (0-based). Default: [5]
    iterations : int
        Number of PEXIT iterations per test.
    samples : int
        Number of binary search samples for smoothing the threshold estimate.

    Returns
    -------
    midEN : float
        Estimated Eb/N0 (dB) decoding threshold.
    SNR : float
        Equivalent SNR for QPSK modulation.
    """
    # Default protograph if not provided
    if S is None:
        S = np.array([
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1]
        ])
    else:
        S = np.array(S)

    if pun is None:
        pun = [] # [5]  # 0-based index in Python

    m, n = S.shape
    if R is None:
        R = (n - m) / (n - len(pun))

    leEN = 0.0
    riEN = 2.0

    # Phase 1: find upper bound
    while not pexit(S, riEN, R, pun, iterations):
        leEN = riEN
        riEN += 0.5

    # Phase 2: binary search for threshold
    for _ in range(samples):
        midEN = (riEN + leEN) / 2.0
        if pexit(S, midEN, R, pun, iterations):
            riEN = midEN
        else:
            leEN = midEN

    midEN = (riEN + leEN) / 2.0
    SNR = midEN + 10 * np.log10(np.log2(4) * R)  # for QPSK

    return midEN, SNR


# if __name__ == "__main__":
#     threshold_dB, snr_qpsk = MET_LDPC_threshold()
#     print(f"PEXIT Threshold (Eb/N0): {threshold_dB:.4f} dB")
#     print(f"Equivalent QPSK SNR:     {snr_qpsk:.4f} dB")



if __name__ == "__main__":
    from ar3a_utils import ar3a_base  # or paste the function above

    R_target = 7 / 8
    B = ar3a_base(R_target)
    threshold_dB, snr_qpsk = MET_LDPC_threshold(S=B, R=R_target, pun=[])

    print(f"PEXIT Threshold (Eb/N0) for AR3A R={R_target:.3f}: {threshold_dB:.4f} dB")
    print(f"Equivalent QPSK SNR:                       {snr_qpsk:.4f} dB")
    R_target = 7 / 8
    B = ar3a_base(R_target)          # shape should be (3, 24)
    print("B shape:", B.shape)

    threshold_dB, snr_qpsk = MET_LDPC_threshold(S=B, R=R_target, pun=[])
    print(f"PEXIT Threshold (Eb/N0) for AR3A R={R_target:.3f}: {threshold_dB:.4f} dB")




# stem = [1 1 1 1 1 1;
#         0 1 1 1 1 1;
#         0 0 1 1 1 1]; # for AR3A

# # For rates >= 0.5
# k = round(6 / R - 6)
# B = np.hstack([stem, np.tile(stem[:, -1:], (1, k))])

# # for rates <0.5 
# stem2 = np.repeat(stem, 2, axis=1)  # parallel-edge factor 2
# cols_needed = round((stem2.shape[1] - 3) / (1 - R))
# B = stem2[:, :cols_needed]