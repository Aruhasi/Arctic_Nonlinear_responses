#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:01:00 2023
This script is used to calculate the statistics of the data
The two variables correlation coefficient, significant test of the distribution

"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# import modules
import os
import sys
import glob

import xarray as xr
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ploting function of the confidence ellipse of the two-dimensional dataset
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
# ===========================
# filtering function butterworth
from scipy.signal import butter
def butter_lowpass_sos(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if not 0 < normal_cutoff < 1:
        raise ValueError(f"Normalized cutoff {normal_cutoff:.3f} not in (0,1). Check 'cutoff' and 'fs'.")
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# website: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
# ===========================
# function to calculate the correlation coefficient
"""
Adapted from statsmodels.stats.multitest.multipletests
This function is used to correct the p-values of multiple hypothesis tests.
control the False Discovery Rate: FDR=E(V/R|R>0)P(R>0) after Benjamini and Hochberg (1995) 
when multiple hypothesis testing is performed, e.g. a statistical test is applied on every grid box 
of a model grid. The procedure can be applied to every collection of tests that are performed to 
local grid points, e.g. correlation, t-test, and so on, as long as p-values are provided.
"""
# False discovery rate (FDR) correction
def xr_multipletest(p, alpha=0.05, method='fdr_bh', **multipletests_kwargs):
    """Apply statsmodels.stats.multitest.multipletests for multi-dimensional xr.objects."""
    from statsmodels.stats.multitest import multipletests
    # stack all to 1d array
    p_stacked = p.stack(s=p.dims)
    # mask only where not nan: https://github.com/statsmodels/statsmodels/issues/2899
    mask = np.isfinite(p_stacked)
    pvals_corrected = np.full(p_stacked.shape, np.nan)
    reject = np.full(p_stacked.shape, np.nan)
    # apply test where mask
    reject[mask] = multipletests(
        p_stacked[mask], alpha=alpha, method=method, **multipletests_kwargs)[0]
    pvals_corrected[mask] = multipletests(
        p_stacked[mask], alpha=alpha, method=method, **multipletests_kwargs)[1]
 
    def unstack(reject, p_stacked):
        """Exchange values from p_stacked with reject (1d array) and unstack."""
        xreject = p_stacked.copy()
        xreject.values = reject
        xreject = xreject.unstack()
        return xreject
 
    reject = unstack(reject, p_stacked)
    pvals_corrected = unstack(pvals_corrected, p_stacked)
    return reject, pvals_corrected
 
# reject, xpvals_corrected = xr_multipletest(p)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# function to calculate the correlation coefficient
def calculate_correlation_coefficient(x, y):
    """
    Calculate the correlation coefficient between two variables.
    
    Parameters:
    - x: Array-like variable 1
    - y: Array-like variable 2
    - pearson also gives the p-value of the correlation coefficient which is two-sided t-test 
    Returns:
    - correlation_coefficient: Correlation coefficient between x and y
    """
    correlation_coefficient, _ = stats.pearsonr(x, y)
    return correlation_coefficient
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# function to check the significance of the correlation coefficient
def check_significance(correlation_coefficient, n):
    """
    Check the significance of the correlation coefficient.
    
    Parameters:
    - correlation_coefficient: Correlation coefficient
    - n: Sample size
    
    Returns:
    - p_value: P-value for the correlation coefficient
    """
    t_statistic = correlation_coefficient * np.sqrt((n - 2) / (1 - correlation_coefficient**2))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n - 2))
    return p_value
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pearson_ci(r, n, alpha=0.05):
    z = np.arctanh(r)  # Fisher transformation
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    return r_lower, r_upper
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Kolmogorov-Smirnov test of the distribution
def kolmogorov_smirnov_test(data1, data2):
    """
    Perform the Kolmogorov-Smirnov test to compare two distributions.
    
    Parameters:
    - data1: Array-like variable 1
    - data2: Array-like variable 2
    
    Returns:
    - ks_statistic: KS statistic
    - p_value: P-value for the KS test
    """
    ks_statistic, p_value = stats.ks_2samp(data1, data2)
    return ks_statistic, p_value
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
compiled on 27 Aug 2025: for exponential non-linear fit
"""
import numpy as np
from scipy.optimize import curve_fit

# Models
def exp_model(x, A, B, C):        # A * exp(B x) + C  (offset handles negatives/zeros)
    return A*np.exp(B*x) + C

def power_model(x, a, b, c):      # a * x^b + c
    return a*np.power(x, b) + c

def logistic_model(x, L, k, x0, C):
    return C + L / (1 + np.exp(-k*(x - x0)))

def nls_fit(x, y, model="exp"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Initial guesses (crucial for NLS)
    if model == "exp":
        C0 = np.percentile(y, 5)  # small offset guess
        y0 = np.clip(y - C0, np.finfo(float).eps, None)
        # linearize ln(y0) ≈ ln A + B x to seed A,B
        B0, lnA0 = np.polyfit(x, np.log(y0), 1)
        A0 = np.exp(lnA0)
        p0 = [A0, B0, C0]
        f  = exp_model
        bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])  # A>0
    elif model == "power":
        C0 = np.percentile(y, 5)
        y0 = np.clip(y - C0, np.finfo(float).eps, None)
        b0, lna0 = np.polyfit(np.log(np.clip(x, np.finfo(float).eps, None)), np.log(y0), 1)
        a0 = np.exp(lna0)
        p0 = [a0, b0, C0]
        f  = power_model
        bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])  # a>0
    elif model == "logistic":
        L0  = (np.nanmax(y) - np.nanmin(y))
        k0  = 1.0 / (np.std(x) + 1e-9)
        x00 = np.median(x)
        C0  = np.nanmin(y)
        p0 = [L0, k0, x00, C0]
        f  = logistic_model
        bounds = (-np.inf, np.inf)
    else:
        raise ValueError("Unknown model")

    popt, pcov = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=20000)
    yhat = f(x, *popt)
    # R² (coefficient of determination)
    r2 = 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)
    return popt, pcov, yhat, r2, f
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++