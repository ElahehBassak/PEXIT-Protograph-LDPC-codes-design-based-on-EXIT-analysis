import numpy as np
from J import J
from J_1 import J_inv  # J_1 is called J_inv in your Python code

def pexit(B, ENdb, R, pun, iterations=100):
    """
    PEXIT analysis function for AWGN channels.
    
    Parameters
    ----------
    B : np.ndarray
        Base matrix of the protograph (m x n).
    ENdb : float
        Eb/N0 in dB.
    R : float
        Code rate.
    pun : list[int]
        List of punctured node indices (0-based).
    iterations : int
        Number of decoding iterations to simulate.

    Returns
    -------
    a : bool
        True if the mutual information converges to 1, otherwise False.
    """
    B = np.array(B)
    m, n = B.shape
    L = iterations

    EN = 10 ** (ENdb / 10)
    cch2 = np.full(n, 8 * R * EN)
    cch2[pun] = 0.0

    IEv = np.zeros((m, n))
    IEc = np.zeros((m, n))
    IEc[:, pun] = 1.0

    Iapp = np.zeros(n)
    Iapp[pun] = 1.0
    store_Iapp = np.zeros(n)

    for _ in range(L):
        # Variable node update
        for j in range(n):
            temp = J_inv(IEc[:, j]) ** 2
            for i in range(m):
                if B[i, j] == 0:
                    IEv[i, j] = 0.0
                else:
                    val = np.dot(B[:, j], temp) - temp[i] + cch2[j]
                    IEv[i, j] = J(np.sqrt(val))

        # Check node update
        for i in range(m):
            temp = J_inv(1 - IEv[i, :]) ** 2
            for j in range(n):
                if B[i, j] == 0:
                    IEc[i, j] = 0.0
                else:
                    val = np.dot(B[i, :], temp) - temp[j]
                    IEc[i, j] = 1.0 - J(np.sqrt(val))

        # APP computation
        Iapp = J(np.sqrt(np.sum(B * (J_inv(IEc) ** 2), axis=0) + cch2))
        Iapp[pun] = 1.0

        if np.all(Iapp >= 1 - 1e-5):
            return True

        delta = store_Iapp - Iapp
        if np.all(delta < 1e-5) and np.all(delta > -1e-5):
            return False
        else:
            store_Iapp = Iapp.copy()

    return False
