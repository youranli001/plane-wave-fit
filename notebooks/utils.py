#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains routines for plane-wave fitting of internal tides (or waves with a specific wavenumber and wavefrequency)

Date: First version: 02-01-2026

Dependencies:
    numpy
    scipy
    xarray
    pandas
    matplotlib
    tables
"""

import numpy as np
# import scipy.signal as signal
# import pylab as plt
# import xarray as xr
# import json
# import scipy.interpolate
# import sys
import os
# import ftplib
# import tables
# import pandas as pd
import contextlib

def distance(lon1, lat1, lon2, lat2):
    """
    Compute the Haversine distance between two sets of points.
    
    Parameters:
    - lon1, lat1: Arrays of longitudes and latitudes of the origin points.
    - lon2, lat2: Arrays of longitudes and latitudes of the destination points.
    
    Returns:
    - distances: Array of distances between origin and destination points.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Radius of the Earth in kilometers
    radius = 6371
    
    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distances = radius * c
    '''
    # Example usage with vector inputs
    lon1 = np.array([10, 20, 30])
    lat1 = np.array([40, 50, 60])
    lon2 = np.array([-10, -20, -30])
    lat2 = np.array([-40, -50, -60])

    distances = distance(lon1, lat1, lon2, lat2)
    print(distances)
    '''
    return distances
    

def wrap_lon_diff(lon, lon0):
    """
    Signed minimal longitude difference (degrees),
    wrapped to [-180, 180).
    """
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def lonlat2xy(SWOT_lon, SWOT_lat, original_lon, original_lat):
    """
    Convert lon/lat to local Cartesian distances (km) relative to a reference point.
    Correctly handles longitude wrap-around near 0/360.
    """

    # flatten SWOT grids
    SWOT_lon_vec = SWOT_lon.flatten()
    SWOT_lat_vec = SWOT_lat.flatten()

    # reference point vectors
    original_lon_vec = np.full_like(SWOT_lon_vec, original_lon)
    original_lat_vec = np.full_like(SWOT_lat_vec, original_lat)

    # --- longitude: wrapped signed difference ---
    dlon = wrap_lon_diff(SWOT_lon_vec, original_lon_vec)

    # distance magnitude in X (use absolute lon difference)
    distX_vec = distance(original_lon_vec, original_lat_vec,
                         original_lon_vec + np.abs(dlon),
                         original_lat_vec)

    # assign sign from wrapped difference
    distX_vec[dlon < 0] *= -1

    # --- latitude: no wrap needed ---
    distY_vec = distance(original_lon_vec, original_lat_vec,
                         original_lon_vec, SWOT_lat_vec)
    distY_vec[SWOT_lat_vec < original_lat_vec] *= -1

    # reshape back to swath grid
    distX = distX_vec.reshape(SWOT_lon.shape)
    distY = distY_vec.reshape(SWOT_lat.shape)

    return distX, distY

def datetime64_to_matlab_datenum(dt64_array):
    """
    Converts a numpy datetime64 array to MATLAB datenum format.
    
    Parameters:
    - dt64_array: numpy.datetime64 array
    
    Returns:
    - matlab_datenum: numpy array of MATLAB datenum values
    """
    
    timestamps = (dt64_array - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')  # seconds since '1970-01-01T00:00:00'   
    days_since_epoch = timestamps / (24 * 3600)  # Convert seconds to days    
    matlab_datenum = days_since_epoch + 719529   # Calculate MATLAB datenum
    
    return matlab_datenum

def fit_wave(data_3D, k, omega, X_3D, Y_3D, T_3D, display_process = False):
    """
    Fit a single wave model to data given spatial (X, Y) and temporal (T) coordinates.
    """

    from scipy.stats import t
    from numpy.linalg import lstsq, LinAlgError
    
    radians_to_degrees = 180 / np.pi
    degrees_to_radians = np.pi / 180
    
    # Reshape data and coordinates to vectors
    B_vec = data_3D.flatten()
    X_vec = X_3D.flatten()
    Y_vec = Y_3D.flatten()
    T_vec = T_3D.flatten()
    
    if np.all(np.isnan(B_vec)):
        print('all NaNs')
        return None, None, None, None, None, None, None, None    
    
    # Remove NaN values
    nonnan_mask = ~np.isnan(B_vec)
    B_vec_nonnan = B_vec[nonnan_mask]
    X_vec_nonnan = X_vec[nonnan_mask]
    Y_vec_nonnan = Y_vec[nonnan_mask]
    T_vec_nonnan = T_vec[nonnan_mask]
    
#     if len(B_vec_nonnan) < data_3D.size * 0.2: # len(B_vec_nonnan) < data_3D.shape[0] * 500:
#         print('Availabel spatial points in window is less 500')
#         return None, None, None, None, None, None, None, None        
       
    
    amplitudes = np.zeros(360)
    phases = np.zeros(360)
    residual_variances = np.zeros(360)

    for angle in range(1, 361):
        theta = degrees_to_radians * angle

        # Model wave components
        cos_term = np.cos(k * X_vec_nonnan * np.cos(theta) + k * Y_vec_nonnan * np.sin(theta) - omega * T_vec_nonnan)
        sin_term = np.sin(k * X_vec_nonnan * np.cos(theta) + k * Y_vec_nonnan * np.sin(theta) - omega * T_vec_nonnan)
        A_vec_nonnan = np.vstack([cos_term, sin_term]).T

        try:
            beta, _, _, _ = lstsq(A_vec_nonnan, B_vec_nonnan, rcond=None)
        except np.linalg.LinAlgError as e:
            print(f"SVD did not converge")
            # continue  # Skip this iteration or handle differently
            return None, None, None, None, None, None, None, None

        # Predicted wave signal
        predicted_wave = A_vec_nonnan.dot(beta)

        # Calculate residual variance
        residual_variances[angle - 1] = np.var(B_vec_nonnan - predicted_wave)

        # Amplitude and phase
        amplitudes[angle - 1] = np.sqrt(beta[0]**2 + beta[1]**2)
        phases[angle - 1] = np.arctan2(beta[1], beta[0])

    # Find the direction with maximum amplitude
    max_amplitude = amplitudes.max()
    max_angle = amplitudes.argmax() + 1

    # Recalculate for the direction with maximum amplitude
    theta1 = degrees_to_radians * max_angle
    cos_term = np.cos(k * X_vec_nonnan * np.cos(theta1) + k * Y_vec_nonnan * np.sin(theta1) - omega * T_vec_nonnan)
    sin_term = np.sin(k * X_vec_nonnan * np.cos(theta1) + k * Y_vec_nonnan * np.sin(theta1) - omega * T_vec_nonnan)
    A_vec_nonnan = np.vstack([cos_term, sin_term]).T

    # Perform regression for best direction
    # beta,_, _, _ = lstsq(A_vec_nonnan, B_vec_nonnan, rcond=None)
    try:
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            beta, _, _, _ = lstsq(A_vec_nonnan, B_vec_nonnan, rcond=None)
    except LinAlgError:
        print("SVD did not converge at angle", angle)
        return None, None, None, None, None, None, None, None
            
    cos_term = np.cos(k * X_vec * np.cos(theta1) + k * Y_vec * np.sin(theta1) - omega * T_vec)
    sin_term = np.sin(k * X_vec * np.cos(theta1) + k * Y_vec * np.sin(theta1) - omega * T_vec)
    A_vec = np.vstack([cos_term, sin_term]).T        
    best_predicted_wave = A_vec.dot(beta)
    best_predicted_wave = best_predicted_wave.reshape(data_3D.shape)

    phase = np.arctan2(beta[1], beta[0])

    
    # Calculate standard error and 95% confidence interval for amplitude
    residuals = B_vec_nonnan - A_vec_nonnan.dot(beta)
    df = len(B_vec_nonnan) - 2  # degrees of freedom
    residual_variance = np.sum(residuals**2) / df
    
    # Compute the inverse of A^T * A
    A_transpose_A_inv = np.linalg.inv(A_vec_nonnan.T.dot(A_vec_nonnan))

    # Covariance matrix of the coefficients
    cov_matrix_beta = residual_variance * A_transpose_A_inv

    # Standard errors of beta coefficients
    standard_errors_beta = np.sqrt(np.diag(cov_matrix_beta))

    # Confidence level (e.g., 95%)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    # t-distribution critical value
    t_critical = t.ppf(1 - alpha/2, df)

    # Confidence intervals for beta coefficients
#     beta_lower_bound = beta - t_critical * standard_errors_beta
#     beta_upper_bound = beta + t_critical * standard_errors_beta
    
    # Calculate standard error for amplitude and phase
    standard_error_amplitude = np.sqrt(
        (beta[0] * standard_errors_beta[0])**2 + (beta[1] * standard_errors_beta[1])**2
    ) / max_amplitude
    
    standard_error_phase = np.sqrt(
        (beta[1] * standard_errors_beta[0])**2 + (beta[0] * standard_errors_beta[1])**2
    ) / (beta[0]**2 + beta[1]**2)

    
    # Output the results
    if display_process:
        print("Covariance Matrix of Coefficients:\n", cov_matrix_beta)
        print("Beta coefficients:", beta)
        print("Standard errors:", standard_errors_beta)
        print("standard_error_amplitude:", standard_error_amplitude)
        print('standard_error_phase:', standard_error_phase)
        print('cov_matrix_beta:',cov_matrix_beta)
              
    # Output the results
    uncertainty_estimates = {
        'dof': df,
        't_critical': t_critical,
        'beta': beta,
        'standard_errors_beta': standard_errors_beta,
        'standard_error_amplitude': standard_error_amplitude,
        'standard_error_phase': standard_error_phase,
        'cov_matrix_beta':cov_matrix_beta
    }

    return max_amplitude, max_angle, phase, best_predicted_wave, residual_variances, amplitudes, phases, uncertainty_estimates





