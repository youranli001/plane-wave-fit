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



def fit_wave_frequency_domain(data_3D, k, omega, X_2D, Y_2D, T_1D, f=None, display_process=False):
    """
    Fit a single wave model using frequency-domain approach for faster computation.
    
    Parameters:
    -----------
    data_3D : ndarray, shape (nx, ny, nt)
        Sea surface height anomaly data
    k : float
        Horizontal wavenumber [rad/km]
    omega : float
        Angular frequency [rad/day]
    X_2D : ndarray, shape (nx, ny)
        X coordinates [km]
    Y_2D : ndarray, shape (nx, ny)
        Y coordinates [km]
    T_1D : ndarray, shape (nt,)
        Time coordinates [days]
    f : ndarray, optional
        Frequency vector [cycles/day]. If None, will be computed.
    display_process : bool
        Whether to print progress
    
    Returns:
    --------
    Same as fit_wave() in time domain
    """
    
    from scipy.stats import t
    from numpy.linalg import lstsq, LinAlgError
    import numpy as np
    
    radians_to_degrees = 180 / np.pi
    degrees_to_radians = np.pi / 180
    
    nx, ny, nt = data_3D.shape
    
    # === STAGE 1: Frequency Domain Direction Finding ===
    
    # Perform FFT along time dimension
    B_freq = np.fft.fft(data_3D, axis=2)
    
    # Compute frequency vector if not provided
    if f is None:
        fs = 1 / (T_1D[1] - T_1D[0])  # Sampling frequency [1/day]
        f = np.fft.fftfreq(nt, d=1/fs)  # [cycles/day]
    
    # Extract M2 frequency component
    M2_freq = omega / (2 * np.pi)  # Convert rad/day to cycles/day
    M2_index = np.argmin(np.abs(f - M2_freq))
    B_freq_M2 = B_freq[:, :, M2_index]  # Shape: (nx, ny)
    B_freq_M2_vec = B_freq_M2.flatten()
    
    # Flatten spatial coordinates
    XX = X_2D.flatten()
    YY = Y_2D.flatten()
    
    # Remove NaN values
    nonnan_mask = ~np.isnan(B_freq_M2_vec)
    B_freq_M2_nonnan = B_freq_M2_vec[nonnan_mask]
    XX_nonnan = XX[nonnan_mask]
    YY_nonnan = YY[nonnan_mask]
    
    if len(B_freq_M2_nonnan) == 0:
        print('All NaNs in frequency domain')
        return None, None, None, None, None, None, None, None
    
    # Initialize arrays for directional scan
    amplitudes_freq = np.zeros(360)
    residual_variances_freq = np.zeros(360)
    
    # Directional scan in frequency domain (spatial only)
    for angle in range(1, 361):
        if display_process and angle % 30 == 0:
            print(f'Frequency domain scan - angle: {angle}')
        
        theta = degrees_to_radians * angle
        
        # Construct spatial plane wave (no time dependence)
        cos_term = np.cos(k * XX_nonnan * np.cos(theta) + k * YY_nonnan * np.sin(theta))
        sin_term = np.sin(k * XX_nonnan * np.cos(theta) + k * YY_nonnan * np.sin(theta))
        A_freq = np.vstack([cos_term, sin_term]).T
        
        # Least-squares fit (can handle complex data)
        try:
            beta, _, _, _ = lstsq(A_freq, B_freq_M2_nonnan, rcond=None)
        except LinAlgError:
            print(f"SVD did not converge at angle {angle}")
            continue
        
        # Extract amplitude (use real part of beta for amplitude calculation)
        amplitude = np.sqrt(np.real(beta[0])**2 + np.real(beta[1])**2)
        amplitudes_freq[angle - 1] = amplitude
        
        # Calculate residual variance
        predicted_wave_freq = A_freq.dot(beta)
        residuals_freq = B_freq_M2_nonnan - predicted_wave_freq
        residual_variances_freq[angle - 1] = np.var(residuals_freq)
    
    # Find direction with maximum amplitude
    max_amplitude_freq = amplitudes_freq.max()
    max_angle_freq = amplitudes_freq.argmax() + 1
    
    if display_process:
        print(f'Frequency domain best angle: {max_angle_freq}°, amplitude: {max_amplitude_freq:.4f}')
    
    # === STAGE 2: Time Domain Refinement ===
    
    # Create 3D coordinates
    X_3D = np.repeat(X_2D[:, :, np.newaxis], nt, axis=2)
    Y_3D = np.repeat(Y_2D[:, :, np.newaxis], nt, axis=2)
    T_3D = np.repeat(T_1D[np.newaxis, np.newaxis, :], nx, axis=0)
    T_3D = np.repeat(T_3D, ny, axis=1)
    
    # Flatten for regression
    B_vec = data_3D.flatten()
    X_vec = X_3D.flatten()
    Y_vec = Y_3D.flatten()
    T_vec = T_3D.flatten()
    
    # Remove NaN values
    nonnan_mask_3D = ~np.isnan(B_vec)
    B_vec_nonnan = B_vec[nonnan_mask_3D]
    X_vec_nonnan = X_vec[nonnan_mask_3D]
    Y_vec_nonnan = Y_vec[nonnan_mask_3D]
    T_vec_nonnan = T_vec[nonnan_mask_3D]
    
    # Test both the best direction and its opposite (180° away)
    theta1 = degrees_to_radians * max_angle_freq
    theta2 = (theta1 + np.pi) % (2 * np.pi)
    
    results = []
    for i, theta in enumerate([theta1, theta2]):
        # Construct time-domain plane wave
        cos_term = np.cos(k * X_vec_nonnan * np.cos(theta) + 
                          k * Y_vec_nonnan * np.sin(theta) - 
                          omega * T_vec_nonnan)
        sin_term = np.sin(k * X_vec_nonnan * np.cos(theta) + 
                          k * Y_vec_nonnan * np.sin(theta) - 
                          omega * T_vec_nonnan)
        A_vec_nonnan = np.vstack([cos_term, sin_term]).T
        
        # Perform regression
        try:
            beta, _, _, _ = lstsq(A_vec_nonnan, B_vec_nonnan, rcond=None)
        except LinAlgError:
            print(f"SVD did not converge for direction {i+1}")
            continue
        
        # Calculate amplitude and phase
        amplitude = np.sqrt(beta[0]**2 + beta[1]**2)
        phase = np.arctan2(beta[1], beta[0])
        
        results.append({
            'amplitude': amplitude,
            'phase': phase,
            'beta': beta,
            'theta': theta,
            'angle_deg': max_angle_freq if i == 0 else (max_angle_freq + 180) % 360,
            'A_vec_nonnan': A_vec_nonnan
        })
    
    # Select the direction with larger amplitude
    if results[0]['amplitude'] > results[1]['amplitude']:
        best_result = results[0]
    else:
        best_result = results[1]
    
    max_amplitude = best_result['amplitude']
    max_angle = best_result['angle_deg']
    phase = best_result['phase']
    beta = best_result['beta']
    A_vec_nonnan = best_result['A_vec_nonnan']
    
    if display_process:
        print(f'Time domain refinement - angle: {max_angle}°, amplitude: {max_amplitude:.4f}')
    
    # Reconstruct full predicted wave (including NaN positions)
    theta_best = best_result['theta']
    cos_term = np.cos(k * X_vec * np.cos(theta_best) + 
                      k * Y_vec * np.sin(theta_best) - 
                      omega * T_vec)
    sin_term = np.sin(k * X_vec * np.cos(theta_best) + 
                      k * Y_vec * np.sin(theta_best) - 
                      omega * T_vec)
    A_vec = np.vstack([cos_term, sin_term]).T
    best_predicted_wave = A_vec.dot(beta)
    best_predicted_wave = best_predicted_wave.reshape(data_3D.shape)
    
    # === Calculate Uncertainty Estimates ===
    
    residuals = B_vec_nonnan - A_vec_nonnan.dot(beta)
    df = len(B_vec_nonnan) - 2
    residual_variance = np.sum(residuals**2) / df
    
    # Covariance matrix
    A_transpose_A_inv = np.linalg.inv(A_vec_nonnan.T.dot(A_vec_nonnan))
    cov_matrix_beta = residual_variance * A_transpose_A_inv
    standard_errors_beta = np.sqrt(np.diag(cov_matrix_beta))
    
    # t-distribution critical value
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_critical = t.ppf(1 - alpha/2, df)
    
    # Standard errors for amplitude and phase
    standard_error_amplitude = np.sqrt(
        (beta[0] * standard_errors_beta[0])**2 + 
        (beta[1] * standard_errors_beta[1])**2
    ) / max_amplitude
    
    standard_error_phase = np.sqrt(
        (beta[1] * standard_errors_beta[0])**2 + 
        (beta[0] * standard_errors_beta[1])**2
    ) / (beta[0]**2 + beta[1]**2)
    
    uncertainty_estimates = {
        'dof': df,
        't_critical': t_critical,
        'beta': beta,
        'standard_errors_beta': standard_errors_beta,
        'standard_error_amplitude': standard_error_amplitude,
        'standard_error_phase': standard_error_phase,
        'cov_matrix_beta': cov_matrix_beta
    }
    
    # Return in same format as time-domain version
    # Note: returning frequency-domain amplitudes and residual variances
    return (max_amplitude, max_angle, phase, best_predicted_wave, 
            residual_variances_freq, amplitudes_freq, None, uncertainty_estimates)

