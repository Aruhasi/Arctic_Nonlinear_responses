#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept 15, 2023
__author__ = "Dr. Josie Aruhasi"
This script contains the functions used in the NHLSAT analysis.
The functions are:
    - calc_anom_1981_2010
    - calc_weighted_mean
    - running_mean
    - quadratic_detrend
    - low_pass_filter
    - detrend_and_filter

"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import modules
import os
import sys

import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import savgol_filter
from scipy.stats import linregress
import pymannkendall as mk
# =============================================================================
# Calculate the monthly anomalies
def calc_anom_1961_1990(data):
    """
    Calculate the monthly anomalies of a dataset relative to the 1985-2014 climatology.
    """
    climatology = data.sel(time=slice('1961-01-01', '1990-12-31')).groupby('time.month').mean(dim='time')
    data_anom = data.groupby('time.month') - climatology
    return data_anom

def calc_anom_1850_1900(data):
    """
    Calculate the monthly anomalies of a dataset relative to the 1985-2014 climatology.
    """
    climatology = data.sel(time=slice('1850-01-01', '1900-12-31')).groupby('time.month').mean(dim='time')
    data_anom = data.groupby('time.month') - climatology
    return data_anom

def calc_anom_1995_2014(data):
    """
    Calculate the monthly anomalies of a dataset relative to the 1985-2014 climatology.
    """
    climatology = data.sel(time=slice('1995-01-01', '2014-12-31')).groupby('time.month').mean(dim='time')
    data_anom = data.groupby('time.month') - climatology
    return data_anom
# =============================================================================
"""
deseasonalize the data
"""
def calculate_monthly_climatology(ds, ref_start, ref_end):
    """Compute the long-term monthly mean for each calendar month."""
    ref = ds.sel(time=slice(ref_start, ref_end))
    # Group by month and average over all years in the reference period
    return ref.groupby('time.month').mean('time')

def calculate_deseasonalized_anomalies(ds, monthly_clim):
    """Subtract the monthly climatology from each time step."""
    # Align each timestamp with its month’s climatology before subtracting
    return ds.groupby('time.month') - monthly_clim

def calculate_annual_mean_anomalies(deseasonalized):
    """Average the deseasonalized anomalies into annual means."""
    return deseasonalized.groupby('time.year').mean('time')
# =============================================================================
# calculate the weighted mean of the global land surface temperature
# def calc_weighted_mean(data, lat, lon):
#     """
#     Calculate the weighted mean of a dataset.
#     """
#     data_ma = ma.masked_invalid(data)
#     # weights matrix
#     # weights = np.cos(np.deg2rad(data.lat))
#     weights = np.cos(np.deg2rad(lat))[:, None]
#     # weights = np.cos(np.tile(abs(lat[:,None])*np.pi/180,(1,len(lon))))[np.newaxis,...] #(time,lat,lon)
#     weighted_data = data_ma*weights
    
#     data_weighted_mean = np.nanmean(weighted_data, axis=(1,2)) 
#     return data_weighted_mean
def selreg(var, lat, lon, lat1, lat2, lon1, lon2):
    """
    Select a region
    Parameters
    ----------
    var : 3-D numpy array 
    lat, lon : 1-D arrays
    """
    ind_start_lat=int(np.abs(lat-(lat1)).argmin())
    ind_end_lat=int(np.abs(lat-(lat2)).argmin())+1
    ind_start_lon=int(np.abs(lon-(lon1)).argmin())
    ind_end_lon=int(np.abs(lon-(lon2)).argmin())+1

    #lonlat
    lons = lon[ind_start_lon:ind_end_lon]
    lats = lat[ind_start_lat:ind_end_lat]
    if var.ndim == 2:
        box = var[ind_start_lat:ind_end_lat,ind_start_lon:ind_end_lon]
    elif var.ndim == 3:
        box = var[:,ind_start_lat:ind_end_lat,ind_start_lon:ind_end_lon]
    return box, lons, lats

# =============================================================================
def create_region_mask(ds, lats, lons, lat_name="lat", lon_name="lon"):
    """Creates a mask for the subpolar gyre region in a given dataset."""
    mask = np.zeros((ds.dims[lat_name], ds.dims[lon_name]))
    
    # Find nearest grid points in dataset for each subpolar gyre coordinate
    for lat, lon in zip(lats, lons):
        lat_idx = np.abs(ds[lat_name] - lat).argmin()
        lon_idx = np.abs(ds[lon_name] - lon).argmin()
        mask[lat_idx, lon_idx] = 1  # Mark as part of the subpolar gyre region
    
    return xr.DataArray(mask, coords={lat_name: ds[lat_name], lon_name: ds[lon_name]}, dims=[lat_name, lon_name])
# ==============================================================================
def calc_weighted_mean(data):
    """
    Calculate the weighted mean of a dataset.
    """
    weights = np.cos(np.deg2rad(data.lat))

    data_weighted_mean = (data.weighted(weights)).mean(('lat', 'lon'))
    return data_weighted_mean

# =============================================================================
def lag1_acf(x, nlags=1):
    """
    Lag 1 autocorrelation
    Parameters
    ----------
    x : 1D numpy.ndarray
    nlags : Number of lag
    Returns
    -------
    acf : Lag-1 autocorrelation coefficient
    """
    # x = x[~np.isnan(x)] # added by Josie on 2024-06-20 to remove NaN values
    n = len(x)
    d = n * np.ones(2 * n - 1)

    acov = (np.correlate(x, x, 'full') / d)[n - 1:]
    acf = acov[:nlags]/acov[0]
    return acf

def mk_test(x, a=0.05):
    """
    Mann-Kendall test for trend
    Parameters
    ----------
    x : 1D numpy.ndarray
    a : p-value threshold
    Returns
    -------
    trend : tells the trend (increasing, decreasing or no trend)
    h : True (if trend is present or Z-score statistic is greater than p-value) or False (if trend is absent)
    p : p-value of the significance test
    z : normalized test statistics
    Tau : Kendall Tau (s/D)
    s : Mann-Kendal's score
    var_s : Variance of s
    slope : Sen's slope
    """
    #Calculate lag1 acf
    # x = x[~np.isnan(x)] # added by Josie on 2024-06-20 to remove NaN values
    acf = lag1_acf(x)

    r1 = (-1 + 1.96*np.sqrt(len(x)-2))/len(x)-1
    r2 = (-1 - 1.96*np.sqrt(len(x)-2))/len(x)-1
    if (acf > 0) and (acf > r1):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x, alpha=a)
    elif (acf < 0) and (acf < r2):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x, alpha=a)
    else:
        #Apply original MK test
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(x, alpha=a)
    return slope, p

# =============================================================================
def interpolate_nan_1D(data):
    """
    Interpolate NaN values for a 1D array.
    """
    valid_mask = ~np.isnan(data)
    times = np.arange(data.shape[0])
    
    # If the entire series is NaN, return it as is
    if not valid_mask.any():
        return data
    
    interpolated_data = np.interp(times, times[valid_mask], data[valid_mask])
    return interpolated_data
# =============================================================================
# define one dimension mann-kendall test
def apply_mannkendall(data):
    # Remove NaN values and check for sufficient data points
    valid_data = data[~np.isnan(data)]
    if len(valid_data) <= 1 or np.all(valid_data == valid_data[0]):
        # Skip the test if there are not enough data points or no variation
        return np.nan, np.nan  # Assign default values or handle as appropriate

    # Perform the Mann-Kendall test
    trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(valid_data, alpha=0.05)
    return slope, p
# =============================================================================
def mann_kendall_test_with_nan_handling(data, nan_strategy='remove'):
    """
    Conduct a Mann-Kendall test on data with NaN values.
    
    Parameters:
        data (array-like): Input data with potential NaN values.
        nan_strategy (str): Strategy for handling NaN values ('remove', 'mean', or 'interpolate').
        
    Returns:
        result: Result of the Mann-Kendall test.
    """
    data_series = pd.Series(data)
    
    # Handle NaN values
    if nan_strategy == 'remove':
        cleaned_data = data_series.dropna()
    elif nan_strategy == 'mean':
        cleaned_data = data_series.fillna(data_series.mean())
    elif nan_strategy == 'interpolate':
        cleaned_data = data_series.interpolate()
    else:
        raise ValueError("Invalid nan_strategy. Choose from 'remove', 'mean', or 'interpolate'.")
    
    # Perform Mann-Kendall test
    result = mk.original_test(cleaned_data)
    slope = result.slope
    p = result.p
    
    return slope, p
# =============================================================================
"""
Created on Fri Sept 15, 2023
calculate the OLS.RMSE, OLS.R2, OLS.p-value, OLS.slope, OLS.intercept
"""
import statsmodels.api as sm

def calculate_trend_ols(y):
    """
    Compute the trend using OLS regression.
    
    Parameters:
    - y: Time series data
    
    Returns:
    - tuple: (trend/slope, standard error)
    """
    if np.isnan(y).all():  # if all values are NaN
        return (np.nan, np.nan)
    
    X = np.arange(len(y))
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    model = sm.OLS(y, X, missing='drop')  # handles missing values by dropping them
    results = model.fit()
    
    trend = results.params[1]
    std_err = results.bse[1]
    
    return (trend, std_err)

def grid_trend_ols(data):
    """
    Compute the trend and standard error for each grid cell.
    
    Parameters:
    - data: 3D array with dimensions time x lat x lon
    
    Returns:
    - 2D array of trends (slopes)
    - 2D array of standard errors
    """
    trend_grid = np.empty(data.shape[1:])
    std_err_grid = np.empty(data.shape[1:])
    
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            trend, std_err = calculate_trend_ols(data[:, i, j])
            trend_grid[i, j] = trend
            std_err_grid[i, j] = std_err
    
    return trend_grid, std_err_grid

# =============================================================================
"""
Created on Sun OCt 22, 2023
calculate the linear regression of GMST onto the SAT spatial pattern
Return the regression slopes and p-values
"""
import numpy as np
from scipy.stats import linregress

def linear_regression_gmst(gmst, sat):
    """
    Perform linear regression of a GMST time series onto the Observational SAT spatial pattern.

    Parameters:
    - gmst: 1D numpy array or xarray DataArray of GMST time series
    - sat: 3D numpy array or xarray DataArray of Observational SAT spatial pattern with dimensions (time, lat, lon)

    Returns:
    - slope: The regression slope
    - intercept: The regression intercept
    """

    # Ensure time dimensions match
    time, lat, lon = sat.shape
    assert len(gmst) == time, "GMST and SAT time dimensions must match!"

    # Reshape SAT data to 2D (time, spatial points)
    sat_reshaped = sat.reshape(time, -1)

    # Identify NaN values in gmst or any NaN across sat_reshaped for the same time point
    valid_mask = ~np.isnan(gmst) & ~np.isnan(sat_reshaped).any(axis=1)
    
    # filter both gmst and sat_reshaped to exclude NaN values
    gmst_filtered = gmst[valid_mask]
    sat_reshaped_filtered = sat_reshaped[valid_mask, :]
    
    # Prepare design matrix for linear regression
    A = np.vstack([gmst_filtered, np.ones(time)]).T

    # Perform linear regression across all grid points
    regression_results = np.linalg.lstsq(A, sat_reshaped_filtered, rcond=None)[0]
    slopes = regression_results[0].reshape(lat, lon)
    intercepts = regression_results[1].reshape(lat, lon)

    return slopes, intercepts
# =============================================================================
"""
Created on Mar 04, 2025
verification the spatial pattern differences between two datasets: constrained and unconstrained
"""
def compute_RMSE(data, data_com):
    """
    Compute the RMSE (Root Mean Square Error) for each constrained model output compared to unconstrained.

    Parameters:
    - data: xarray DataArray, shape (member, lat, lon) or (lat, lon) for ensemble-averaged values.
    - data_com: xarray DataArray, shape (lat, lon), representing observed values.

    Returns:
    - RMSE: xarray DataArray of RMSE values, shape (lat, lon).
    - RMSE_weighted: xarray DataArray of weighted RMSE values, weighted by number of ensemble members.
    """
    RMSE = {}

    for key in data.keys():  # key is (model, scenario)
        constrained_data = data[key]  # This is an xarray Dataset
        unconsrtaied_data = data_com[key]
        # Extract the 'tas' variable (Beta pattern)
        if "tas" in constrained_data:
            constrained_data = constrained_data["tas"]
            unconsrtaied_data = unconsrtaied_data["tas"]
        else:
            print(f"Warning: 'tas' not found in {key}. Skipping...")
            continue

    
        # Compute RMSE for each ensemble member, then average over members
        if "member" in constrained_data.dims:
            diff = constrained_data - unconsrtaied_data
            squared_diff = diff ** 2
            rmse_memberwise = np.sqrt(squared_diff.mean(dim="member"))  # RMSE across members
        else:
            rmse_memberwise = np.sqrt(((constrained_data - unconsrtaied_data) ** 2).mean())

        RMSE[key] = rmse_memberwise

    return RMSE

# =============================================================================
"""
Calculate the internal climate variability (ICV) trend pattern's standard deviation for each member.
"""
def calculate_icv_trend_std(trend_data, ensmean_data, axis=0):
    """
    Calculate the internal climate variability (ICV) trend pattern's standard deviation for each member.

    Parameters:
    trend_data (np.ndarray): Array of trend data from individual members.
    ensmean_data (np.ndarray): Array of ensemble mean trend data.
    axis (int): Axis along which to calculate the standard deviation.

    Returns:
    np.ndarray: The standard deviation of the ICV trend pattern.
    """
    # Calculate the deviations of the trend pattern among members
    trend_deviations = (trend_data - ensmean_data) ** 2
    
    # Calculate the variance (average of the squared deviations)
    trend_variance = np.mean(trend_deviations, axis=axis)
    
    # Calculate the standard deviation (square root of the variance)
    trend_std = np.sqrt(trend_variance)
    
    return trend_std


