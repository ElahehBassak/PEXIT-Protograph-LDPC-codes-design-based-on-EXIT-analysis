# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 01:22:38 2025

@author: elahe
"""

import numpy as np

def ar3a_base(R):
    stem = np.array([
        [1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1]
    ])

    if R >= 0.5:
        k = int(round(3 / (1 - R) - 6))  # ← fixed formula
        return np.hstack([stem, np.tile(stem[:, -1:], (1, k))])

    # low-rate branch unchanged
    stem2 = np.repeat(stem, 2, axis=1)
    cols_needed = int(round((stem2.shape[1] - 3) / (1 - R)))
    return stem2[:, :cols_needed]
