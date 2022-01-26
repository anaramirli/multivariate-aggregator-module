"""Utilities for multivariate aggregation."""

import numpy as np


def calculate_threshold(df_t):
    """Compute threshold"""
    threshold = np.mean(df_t['T2_train']) + (2 * np.std(df_t['T2_train']))
    return threshold
