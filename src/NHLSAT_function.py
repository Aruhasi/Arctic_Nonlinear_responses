#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sept 15, 2023
__author__ = "Josie Aruhasi, PhD"
This script contains the functions used in the NHLSAT analysis.
The functions are:
    - calc_anom_1961_1990
    - calc_weighted_mean
    - calc_seasonal_mean
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
# def calc_anom_1961_1990(data, dates):
#     """
#     Calculate the monthly anomalies of a dataset relative to the 1961-1990 climatology.
#     data: 3D NumPy array with dimensions (time, latitude, longitude)
#     dates: 1D array of datetime objects corresponding to the time dimension of data
#     """
#     # Convert dates to pandas datetime for easier handling
#     dates = pd.to_datetime(dates)

#     # Find indices corresponding to the 1961-1990 period
#     indices = (dates >= '1961-01-01') & (dates <= '1990-12-31')

#     # Filter data for the 1961-1990 period
#     climatology_data = data[indices]

#     # Calculate monthly means for 1961-1990
#     monthly_climatology = np.array([climatology_data[dates[indices].month == month].mean(axis=0) for month in range(1, 13)])

#     # Subtract climatology from each corresponding month in the entire dataset
#     data_anom = np.array([data[dates.month == month] - monthly_climatology[month-1] for month in range(1, 13)])

#     # Concatenate the monthly anomalies back into a single array
#     data_anom = np.concatenate(data_anom, axis=0)

#     # Reorder the data_anom array to match the original time order
#     reorder_indices = np.argsort(dates)
#     data_anom = data_anom[reorder_indices]

#     return data_anom

# Calculate the monthly anomalies
def calc_anom_1961_1990(data):
    """
    Calculate the monthly anomalies of a dataset relative to the 1961-1990 climatology.
    """
    climatology = data.sel(time=slice('1961-01-01', '1990-12-31')).groupby('time.month').mean(dim='time')
    data_anom = data.groupby('time.month') - climatology
    return data_anom

def calc_anom_1981_2010(data):
    """
    Calculate the monthly anomalies of a dataset relative to the 1985-2014 climatology.
    """
    climatology = data.sel(time=slice('1981-01-01', '2010-12-31')).groupby('time.month').mean(dim='time')
    data_anom = data.groupby('time.month') - climatology
    return data_anom

# calculate the weighted mean of the global land surface temperature
def calc_weighted_mean(data):
    """
    Calculate the weighted mean of a dataset.
    """
    weights = np.cos(np.deg2rad(data.lat))

    data_weighted_mean = (data.weighted(weights)).mean(('lat', 'lon'))
    return data_weighted_mean
# =============================================================================
# Calculate the seasonal mean of the monthly anomalies
# def calc_seasonal_mean(data):
#     """
#     Calculate the seasonal means for DJF, MAM, JJA, SON.

#     Parameters:
#     data (xarray.Dataset): The dataset containing the time series data.

#     Returns:
#     xarray.Dataset: A dataset with the seasonal means.
#     """
#     # Create a new dataset to store the seasonal means
#     seasonal_means = xr.Dataset()

#     # Calculate the mean for each season and add it to the dataset
#     # For DJF, we need to handle December separately
#     djf = ((data.sel(time=data['time.month'] == 12).groupby('time.year').mean('time', skipna=True) +
#             data.sel(time=data['time.month'] <= 2).groupby('time.year').mean('time', skipna=True)) / 3)
#     seasonal_means['DJF'] = djf.rename({'year': 'year'})

#     # For other seasons, we can use a more straightforward approach
#     seasonal_means['MAM'] = data.where(data['time.season'] == 'MAM', drop=True).groupby('time.year').mean('time', skipna=True)
#     seasonal_means['JJA'] = data.where(data['time.season'] == 'JJA', drop=True).groupby('time.year').mean('time', skipna=True)
#     seasonal_means['SON'] = data.where(data['time.season'] == 'SON', drop=True).groupby('time.year').mean('time', skipna=True)

#     return seasonal_means

def calc_seasonal_mean(data):
    # Ensure data is sorted by time
    data = data.sortby('time')
    
    # Group data by year
    data_grouped_by_year = data.groupby('time.year')
    
    seasonal_means = {}
    
    for year, data_year in data_grouped_by_year:
        
        # DJF
        if year + 1 in data_grouped_by_year.groups.keys():  # Check if we have data for the next year
            december = data.sel(time=f"{year}-12").mean('time')
            january = data.sel(time=f"{year+1}-01").mean('time')
            february = data.sel(time=f"{year+1}-02").mean('time')
            djf_mean = (december + january + february) / 3
            seasonal_means.setdefault(year, {})['DJF'] = djf_mean
        
        # MAM
        mam_data = data.where((data['time.season'] == 'MAM') & (data['time.year'] == year), drop=True)
        if mam_data.size>0:
            mam_mean = mam_data.mean('time')
            seasonal_means.setdefault(year, {})['MAM'] = mam_mean
        
        # JJA
        jja_data = data.where((data['time.season'] == 'JJA') & (data['time.year'] == year), drop=True)
        if jja_data.size>0:
            jja_mean = jja_data.mean('time')
            seasonal_means.setdefault(year, {})['JJA'] = jja_mean
        
        # SON
        son_data = data.where((data['time.season'] == 'SON') & (data['time.year'] == year), drop=True)
        if son_data.size>0:
            son_mean = son_data.mean('time')
            seasonal_means.setdefault(year, {})['SON'] = son_mean
            
    return seasonal_means
    # """
    # Calculate the seasonal mean of a dataset.
    # """
    # seasons = ['JJA', 'DJF', 'MAM', 'SON']
    # data_seasonal = {}

    # for season in seasons:
    #     if season == 'JJA':
    #         months = [6,7,8]
    #     elif season == 'DJF':
    #         months =[12,1,2]
    #     elif season == 'MAM':
    #         months = [3,4,5]
    #     elif season == 'SON':
    #         months = [9,10,11]

    #     season_months = data.sel(time=data.time.dt.month.isin(months))
    
    # # Calculate the seasonal mean SAT anomalies
    #     season_mean_anomalies = season_months.groupby('time.year').mean('time')
    
    # # Store the seasonal mean in to the dataset
    #     data_seasonal[season] = season_mean_anomalies
    # return data_seasonal
# =============================================================================
# calculate the 9 year running mean without losing the data
def running_mean(x, window=9):
    """
    Compute the running mean of a time series without losing data at the ends.
    """
    n = len(x)
    smoothed_x = np.zeros(n)
    
    half_window = window // 2
    
    for i in range(n):
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        smoothed_x[i] = np.mean(x[start_idx:end_idx])
        
    return smoothed_x
# =============================================================================
# define functions to detrend the time series then calculate the 9-year low-pass filter
def quadratic_trend(data, x):
    """
    Detrend the data by subtracting the quadratic fit.
    
    Parameters:
    - data: Array-like data values to be detrended
    - x: Corresponding x-values (e.g., time or years)
    
    Returns:
    - detrended_data: Data after subtracting the quadratic fit
    """
    coeff = np.polyfit(x, data, 2)  # Fit a quadratic polynomial
    quadratic_fit = np.polyval(coeff, x)
    return quadratic_fit

def low_pass_filter(data, window_size=9):
    """
    Apply a low-pass filter using a running mean.
    
    Parameters:
    - data: Array-like data values to be filtered
    - window_size: Size of the moving window (default is 9 for a 9-year running mean)
    
    Returns:
    - filtered_data: Data after applying the low-pass filter
    """
    filtered_data = savgol_filter(data, window_size, 0)  # 0 is the polynomial order for a simple moving average
    return filtered_data

# 3. Multi-dimensional Quadratic Detrending
def quadratic_detrend_nd(data, x):
    detrended_data = np.apply_along_axis(quadratic_trend, 1, data, x)
    return detrended_data

# 4. Multi-dimensional Low-pass Filtering
def low_pass_filter_nd(data, window_size=9):
    filtered_raw = np.apply_along_axis(low_pass_filter, 0, data, window_size)
    filtered_data = xr.DataArray(filtered_raw, coords=data.coords, dims=data.dims)
    return filtered_data

def filter_quadratic_detrend_Model(data):
    """
    Detrend and apply a low-pass filter to the data.
    
    Parameters:
    - data: Array-like data values to be detrended and filtered
    - x: Corresponding x-values (e.g., time or years)
    - window_size: Size of the moving window (default is 9 for a 9-year running mean)
    
    Returns:
    - filtered_data: Data after applying the low-pass filter
    """
    data_filtered = low_pass_filter_nd(data)
    filtered_array = xr.DataArray(data_filtered,
                                    coords=[np.arange(0,30,1),np.arange(1900,2100,1)],
                                    dims=['run', 'year'])
    # Calculate the multi-model mean
    mme_mean = filtered_array.mean(dim='run') # single time series

    # Compute the quadratic fit for this multi-model mean
    years = np.arange(mme_mean.sizes['year'])
    mme_quadratic_fit = quadratic_trend(mme_mean, years) # single time series, same length as mme_mean,but with the quadratic fitted values
    mme_quadratic_fit

    # Subtract the multi-model mean quadratic fit from each individual ensemble member
    detrended_data = filtered_array - mme_quadratic_fit

    return detrended_data

# 5. Combined Detrending and Filtering Function
def filter_and_detrend(data, x, window_size=9):
    filtered_data = low_pass_filter(data, window_size)
    detrended_data = quadratic_trend(filtered_data, x)
    data_filtered_and_detrended = data - detrended_data
    return data_filtered_and_detrended

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
    y = x - np.nanmean(x)
    n = len(x)
    d = n * np.ones(2 * n - 1)

    acov = (np.correlate(y, y, 'full') / d)[n - 1:]
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
    acf = lag1_acf(x)

    r1 = (-1 + 1.96*np.sqrt(len(x)-2))/len(x)-1
    r2 = (-1 - 1.96*np.sqrt(len(x)-2))/len(x)-1
    if (acf > 0) and (acf > r1):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x)
    elif (acf < 0) and (acf < r2):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x)
    else:
        #Apply original MK test
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(x)
    return h, p, z, Tau, s, var_s, slope, intercept


# def apply_mannkendall_3D(data):
#     """
#     Apply the Mann-Kendall test to a 3D dataset.
#     """
#     time_len, lat_len, lon_len = data.shape
#     p_values = np.empty((lat_len, lon_len))
#     trends = np.empty((lat_len, lon_len))
    
#     for lat_idx in range(lat_len):
#         for lon_idx in range(lon_len):
#             h, p, z, Tau, s, var_s, slope, intercept = mk_test(data[:, lat_idx, lon_idx])
#             p_values[lat_idx, lon_idx] = p
#             trends[lat_idx, lon_idx] = slope
            
#     return p_values, trends

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

def apply_mannkendall_3D_interpolated(data):
    """
    Apply the Mann-Kendall test to a 3D dataset after interpolating NaN values.
    """
    # Derive the lat_length and lon_length from the data's shape
    _, lat_length, lon_length = data.shape
    
    # Initialize the output arrays
    p_values = np.empty((lat_length, lon_length))
    slope_values = np.empty((lat_length, lon_length))
    
    # Loop over the latitudes and longitudes
    for lat_idx in range(lat_length):
        for lon_idx in range(lon_length):
            # Select the data at the current latitude and longitude
            data_lat_lon = data[:, lat_idx, lon_idx]
            
            # Interpolate NaN values
            data_lat_lon_interpolated = interpolate_nan_1D(data_lat_lon)
            
            # Apply the Mann-Kendall test
            result = mk.original_test(data_lat_lon_interpolated, alpha=0.05)
            p_values[lat_idx, lon_idx] = result[2]
            slope_values[lat_idx, lon_idx] = result[7]
            
    return slope_values, p_values
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
    
def apply_mannkendall_3D(data):
    """
    Apply the Mann-Kendall test to a 3D dataset.
    """
    # Initialize the output arrays
    _, lat_length, lon_length = data.shape
    p_values = np.empty((lat_length, lon_length))
    slope_values = np.empty((lat_length, lon_length))
    # data = preprocess_nan(data)
    # Loop over the latitudes and longitudes
    for lat_idx in range(lat_length):
        for lon_idx in range(lon_length):
            # Select the data at the current latitude and longitude
            data_lat_lon = data[:, lat_idx, lon_idx]
            
            # Apply the Mann-Kendall test
            p_values[lat_idx, lon_idx] = mk.original_test(data_lat_lon, alpha=0.05)[2]
           
            slope_values[lat_idx, lon_idx] = mk.original_test(data_lat_lon, alpha=0.05)[7]
           
    return slope_values, p_values
# =============================================================================
def mann_kendall_test_1d(time_series):
    """
    Conduct a Mann-Kendall test on 1D data array.
    
    Parameters:
        time_series (array-like): Input 1D data array.
        
    Returns:
        slope (float): The slope of the trend.
        p (float): The p-value of the test.
    """
    # Handle NaN values by removing them
    cleaned_data = time_series[~np.isnan(time_series)]
    
    # Check if there are enough valid data points
    if len(cleaned_data) < 3:  # or whatever threshold you deem appropriate
        return np.nan, np.nan
    
    # Perform Mann-Kendall test
    try:
        result = mk.original_test(cleaned_data)
        # Extract and return the slope and p-value
        slope = result.slope
        p = result.p
        return slope, p
    except ZeroDivisionError:
        return np.nan, np.nan

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
# import numpy as np
# from scipy.stats import t

# def linear_regression_gmst(gmst, sat):
#     """
#     Perform linear regression of a GMST time series onto the Observed SAT spatial pattern.

#     Parameters:
#     - gmst: 1D numpy array of GMST time series
#     - sat: 3D numpy array of Observed SAT spatial pattern with dimensions (time, lat, lon)

#     Returns:
#     - slopes: 2D numpy array with dimensions (lat, lon) containing regression slopes
#     - p_values: 2D numpy array with dimensions (lat, lon) containing p-values for significance
#     """
#     time, lat, lon = sat.shape
#     assert len(gmst) == time, "GMST and SAT time dimensions must match!"

#     # Initialize arrays to store slopes and p-values
#     slopes = np.empty((lat, lon))
#     p_values = np.empty((lat, lon))

#     # Calculate degrees of freedom for t-test
#     df = time - 2

#     # Loop through each grid point
#     for i in range(lat):
#         for j in range(lon):
#             # Get the SAT time series for the current grid point
#             y = sat[:, i, j]
            
#             # Add a constant to gmst for intercept in regression
#             A = np.vstack([gmst, np.ones(time)]).T
            
#             # Calculate regression coefficients (slope, intercept)
#             slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

#             # Calculate residuals
#             residuals = y - (slope * gmst)

#             # Calculate standard error of the regression slope
#             stderr = np.sqrt(sum(residuals**2) / (time - 2)) / np.sqrt(sum((gmst - gmst.mean())**2))
            
#             # Calculate t-value
#             t_value = slope / stderr
            
#             # Calculate two-tailed p-value
#             p_value = 2 * (1 - t.cdf(abs(t_value), df))

#             # Store results
#             slopes[i, j] = slope
#             p_values[i, j] = p_value

#     return slopes, p_values
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

    # Prepare design matrix for linear regression
    A = np.vstack([gmst, np.ones(time)]).T

    # Perform linear regression across all grid points
    regression_results = np.linalg.lstsq(A, sat_reshaped, rcond=None)[0]
    slopes = regression_results[0].reshape(lat, lon)
    intercepts = regression_results[1].reshape(lat, lon)

    return slopes, intercepts
# def linear_regression_gmst(gmst, sat):
#     """
#     Perform linear regression of a GMST time series onto the Observational SAT spatial pattern.

#     Parameters:
#     - gmst: 1D numpy array or xarray DataArray of GMST time series
#     - sat: 3D numpy array or xarray DataArray of Observational SAT spatial pattern with dimensions (time, lat, lon)

#     Returns:
#     - slope: The regression slope
#     - intercept: The regression intercept
#     """
#     # Convert xarray DataArray to numpy arrays if necessary
#     # Check if inputs are xarray DataArrays and convert to numpy arrays if necessary
#     # if isinstance(gmst, xr.DataArray):
#     #     gmst = gmst.values
#     # if isinstance(sat, xr.DataArray):
#     #     sat = sat.values

#     # Ensure time dimensions match
#     time, lat, lon = sat.shape
#     assert len(gmst) == time, "GMST and SAT time dimensions must match!"

#     # Reshape SAT data to 2D (time, spatial points)
#     sat_reshaped = sat.reshape(time, -1)

#     # Prepare design matrix for linear regression
#     A = np.vstack([gmst, np.ones(time)]).T

#     # Perform linear regression across all grid points
#     regression_results = np.linalg.lstsq(A, sat_reshaped, rcond=None)[0]
#     slopes = regression_results[0].reshape(lat, lon)
#     intercepts = regression_results[1].reshape(lat, lon)

#     return slopes, intercepts
# =============================================================================
"""
Based on the function from linear_regression_gmst_vectorized
Reconstruct the observed SAT anomalous pattern using linear regression coefficients.
Gaining the reconstructed SAT pattern, residuals, and standard error of the regression slope

"""
def reconstruct_observed_sat_vectorized(gmst,var):
    """
    Reconstruct the observed SAT anomalous pattern using linear regression coefficients.

    Parameters:
    - var: The variable to be reconstructed (e.g., SAT, precipitation, etc.)
    - slope: The regression slope obtained from linear regression.
    - intercept: The regression intercept obtained from linear regression.
    - gmst: 1D numpy array of GMST time series.

    Returns:
    - reconstructed_sat: 3D numpy array of the reconstructed SAT anomalous pattern.
    - residuals: 3D numpy array of the residuals.
    """
    slope, intercept = linear_regression_gmst(gmst, var)
    gmst_expanded = gmst[:, np.newaxis, np.newaxis]
    reconstructed_sat = slope * gmst_expanded + intercept
    residuals = var - reconstructed_sat
    
    # # Calculate standard error of the regression slope
    # gmst_centered = gmst - gmst.mean()
    # stderr = np.sqrt((residuals**2).sum() / (len(gmst) - 2)) / np.sqrt((gmst_centered**2).sum())
    
    return reconstructed_sat, residuals
# =============================================================================
from scipy.stats import t

def linear_regression_gmst_vectorized(gmst, sat):
    """
    Perform linear regression of a GMST time series onto the Observed SAT spatial pattern.

    Parameters:
    - gmst: 1D numpy array of GMST time series
    - sat: 3D numpy array of Observed SAT spatial pattern with dimensions (time, lat, lon)

    Returns:
    - slopes: 2D numpy array with dimensions (lat, lon) containing regression slopes
    - p_values: 2D numpy array with dimensions (lat, lon) containing p-values for significance
    """
    time, lat, lon = sat.shape
    assert len(gmst) == time, "GMST and SAT time dimensions must match!"

    # Add a constant to gmst for intercept in regression
    A = np.vstack([gmst, np.ones(time)]).T
    
    # Calculate regression coefficients (slope, intercept)
    # This will give us an array of shape (2, lat, lon) where the first row is the slopes and the second row is the intercepts
    coefficients = np.linalg.lstsq(A, sat, rcond=None)[0]
    
    # Extract the slopes (0th row of coefficients)
    slopes = coefficients[0]
    
    # Calculate residuals
    residuals = sat - slopes[None, :, :] * gmst[:, None, None]
    
    # Calculate standard error of the regression slope
    gmst_centered = gmst - gmst.mean()
    stderr = np.sqrt((residuals**2).sum(axis=0) / (time - 2)) / np.sqrt((gmst_centered**2).sum())
    
    # Calculate t-value
    t_values = slopes / stderr
    
    # Calculate two-tailed p-value
    p_values = 2 * (1 - t.cdf(np.abs(t_values), time - 2))
    
    return slopes, p_values

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


