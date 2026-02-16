---
title: "PlaneWaveFit: A Python Package for Two-Dimensional Plane-Wave Fitting of Internal Gravity Waves"
tags:
  - python
  - physical-oceanography
  - internal-tides 
authors:
  - name: Youran Li
    orcid: 0000-0002-4576-5213
    affiliation: 1
  - name: Sarah T. Gille
    orcid: 0000-0001-9144-4368
    affiliation: 1 
  - name: Matthew R. Mazloff
    orcid: 0000-0002-1650-5850
    affiliation: 1 
affiliations:
  - name: Scripps Institution of Oceanography, University of California San Diego, USA
    index: 1
date: 20 February 2026
bibliography: paper.bib
---

# Summary

PlaneWaveFit extracts amplitude, phase, and propagation direction of internal tides from sea surface height (SSH) observations, with quantified uncertainties. The package implements two complementary plane-wave fitting approaches: (1) a time-domain method applicable to any spatiotemporal dataset, including irregularly-sampled observations, and (2) a frequency-domain method for evenly-sampled data (e.g., hourly model output). Both resolve multiple wave components and provide uncertainty estimates. The package also interfaces with a companion Zenodo dataset of precomputed mode-1 M2 internal tide parameters on a global 0.25° grid, enabling direct conversion from fitted surface amplitudes to depth-integrated energy and flux without requiring local stratification data. A detailed mathematical derivation of the fitting methods and uncertainty quantification is provided in [@li2026].

# Statement of need

Extracting internal tides from SSH observations requires separating a baroclinic (depth-varying) tidal signal of O(1 cm) from mesoscale variability of O(1 m). Widely-used packages like `UTide` [@codiga2011unified] and `T_TIDE` [@pawlowicz2002classical] perform temporal harmonic analysis at fixed locations but do not solve for spatial propagation. PlaneWaveFit implements two-dimensional fitting that extracts internal tides that are coherent in both space and time, minimizing sensitivity to mesoscale contamination and enabling robust detection of tidal beams in energetic regions. In synthetic tests, fitted amplitudes, phases, and propagation directions match the prescribed 1 cm values even with correlated eddy noise up to 20 cm root mean square (RMS).

# State of the field

Two-dimensional plane-wave fitting has proven effective for extracting internal tides from SSH data [@zhao2011internal; @zhao2016global; @zhao2017global; @zaron2019baroclinic; @zhao2024internal], but documented, open-source implementations with uncertainty quantification have not been available. PlaneWaveFit addresses this gap.

Internal tides can be represented as a superposition of vertical modes, each with a fixed vertical structure and characteristic horizontal wavelength. Low-mode internal tides propagate thousands of kilometers from generation sites, while higher modes dissipate locally. This package complements vertical mode solvers like dynmodes [@klinck_dynmodes], which compute vertical mode structures from a stratification profile. PlaneWaveFit uses those outputs to compute additional internal tide parameters (following Appendix A of @li2026), including wavelengths, phase and group speeds, and conversion ratios linking SSH amplitude to depth-integrated energy and flux. These conversion ratios are precomputed globally from World Ocean Atlas 2023 (WOA23) climatology and archived in a Zenodo dataset [@li2025internal].

# Software design

## 1. Internal Tide Parameter Database

The plane-wave fitting method requires a tidal angular frequency ($\omega$) and horizontal wavenumber ($k$), which users can specify for any tidal constituent. For convenience, the package provides optional access to a Zenodo-archived dataset [@li2025internal] containing precomputed M2 internal tide parameters. M2 is the principal lunar semidiurnal tide with period of 12.42 hours. The dataset spans modes 1-10 on a global 0.25° grid with the following features:

1. Vertical mode structures of pressure, horizontal velocity, vertical velocity, and vertical displacement
2. Wave properties: phase speeds, group speeds, and wavelengths
3. Conversion ratios relating SSH amplitude to:
   a. Maximum vertical isopycnal displacement
   b. Depth-integrated potential, kinetic, and total energy density
   c. Depth-integrated horizontal energy flux
   
The physical derivation of these conversion ratios is detailed in Appendix A of @li2026. As an example, a 4.12 mm fitted surface amplitude at 35°W, 35.5°S corresponds to a depth-integrated horizontal energy flux of 0.35 kW/m using the precomputed conversion ratio. Users with local stratification profiles can compute these parameters directly following the same methodology; those working with SSH observations alone can use the precomputed database.

## 2. Plane-Wave Fitting Algorithm

Internal tide parameters (amplitude $A_m$, phase $\phi_m$, direction $\theta_m$) are estimated by fitting the SSH model:

$$
\mathrm{SSHA}(x, y, t)
=
\sum_{m=1}^{N}
A_m
\cos\!\left(
k\,x\cos\theta_m
+
k\,y\sin\theta_m
-
\omega t
-
\phi_m
\right)
$$

where $\omega$ can be any tidal angular frequency of interest, and $k$ is the horizontal wavenumber (provided by the user or obtained from the precomputed database). $N$ is the number of fitted wave components.

### Time-Domain Method

For each compass direction (angular increment 1°), least-squares fitting determines the amplitude and phase of a plane wave. When plotted in polar coordinates, a wave component appears as a lobe; the largest lobe gives the amplitude and direction of the dominant wave (\autoref{fig:example_timedomain}). That wave is then predicted and subtracted from the SSH anomaly (SSHA) data, and the procedure repeated to extract additional components. Amplitude and phase uncertainties derive from the least-squares covariance matrix. Details of the fitting procedure and uncertainty derivation are provided in @li2026.

The following example demonstrates the method applied to Surface Water and Ocean Topography (SWOT) satellite observations at 35°W, 35.5°S during the calibration/validation phase. The SSH anomaly (SSHA) data have been preprocessed to remove mesoscale variability by subtracting gridded sea level anomalies from multi-mission satellite altimetry.

![Plane wave fit applied to SWOT SSHA data. (a) SWOT SSHA near 35°W, 35.5°S on April 2, 2023. (b-d) Polar plots showing fitted amplitude (mm) versus propagation direction for the three most energetic wave components; arrows indicate the selected propagation direction.](figures/figure1.png){#fig:example_timedomain}

**Example: SWOT data**
```python
import numpy as np
import xarray as xr
from pathlib import Path
import utils

# Load SWOT data (included in repository)
data_dir = Path("../data")
ds = xr.open_dataset(data_dir / "SWOT_CalVal_SSHA_35W_35p5S.nc")

# Convert coordinates to Cartesian (km) and reshape to (time, y, x)
lon_ref, lat_ref = -35.0, -35.5
distX, distY = utils.lonlat2xy(ds.longitude.values, ds.latitude.values, lon_ref, lat_ref)
X_3D = np.repeat(distX[np.newaxis, :, :], ds.sizes['num_cycles'], axis=0)
Y_3D = np.repeat(distY[np.newaxis, :, :], ds.sizes['num_cycles'], axis=0)
T_3D_dt64 = np.repeat(ds['time'].values[:, :, np.newaxis], ds.sizes['num_pixels'], axis=2)
T_3D = utils.datetime64_to_matlab_datenum(T_3D_dt64)
ssha = ds['ssha'].values

# M2 parameters
omega = 2 * np.pi / (12.42 * 3600) * 86400  # rad/day
k = 0.04124732611475162  # rad/km (from internal tide parameter database)

# Iterative fitting
amp1, theta1, phi1, model1, var1, _, _, uncert1 = utils.fit_wave(
    ssha, k, omega, X_3D, Y_3D, T_3D
)
amp2, theta2, phi2, model2, var2, _, _, uncert2 = utils.fit_wave(
    ssha - model1, k, omega, X_3D, Y_3D, T_3D
)
```


### Frequency-Domain Method

For evenly-sampled data, the frequency-domain method uses a two-step hybrid approach. Step 1 applies temporal Fast Fourier Transform (FFT) to extract the M2 component, reducing the 3D spatiotemporal problem to 2D spatial fitting. This also filters out signals at other frequencies that would otherwise act as noise in the time-domain fitting. The complex spatial field is:

$$B_{M_2}(x, y) = \sum_{m=1}^{N} A_m \cos(k x \cos\theta_m + k y \sin\theta_m - \phi_m)$$

We perform 360 directional scans (1°–360°), fitting this spatial pattern at each angle. Because waves at angles $\theta$ and $\theta + 180°$ produce the same complex spatial field $B_{M_2}(x, y)$, the fitting yields identical amplitudes for both directions, creating symmetric two-lobe polar plots (\autoref{fig:example_frequency_domain}, panels b-d). Note that due to this symmetry, scanning only 180 directions would suffice; however, since the computational cost of these 2D spatial fits is negligible, we retain the full 360° scan for implementation simplicity. Step 2 resolves this 180° ambiguity by performing full time-domain fits at both candidate directions and selecting the one with larger amplitude.

The frequency-domain method achieves substantial speedup by collapsing the time dimension via FFT. Step 1 performs 360 directional scans on 2D spatial data ($n_x \times n_y$ points), which is computationally inexpensive compared to fitting 3D spatiotemporal data ($n_t \times n_x \times n_y$ points). Step 2 then performs only 2 time-domain fits for directional disambiguation, compared to 360 time-domain fits in the time-domain-only method. This yields approximately 180× speedup. The method requires evenly-spaced time samples but handles irregular spatial sampling.

MITgcm LLC4320 is a high-resolution global ocean simulation with 1/48° horizontal resolution and realistic tidal forcing. The following example demonstrates the frequency-domain method applied to this model output at 155°E, 45°S in the Tasman Sea, a region of energetic internal tides radiating northwest from the Macquarie Ridge. The SSHA data (hourly output spanning 40 days, May 29 - July 8, 2012) have been preprocessed with a 24-hour temporal high-pass filter and a 500 km spatial high-pass filter. The example dataset is included in the repository.

![Frequency-domain plane wave fit applied to MITgcm LLC4320 SSHA data in the Tasman Sea. (a) Preprocessed SSHA snapshot at 155°E, 45°S on June 2, 2012. (b-d) Polar plots showing fitted amplitude (mm) versus propagation direction for the three most energetic wave components; arrows indicate the selected propagation direction. Note the symmetric two-lobe patterns characteristic of frequency-domain directional scans.](figures/figure2.png){#fig:example_frequency_domain}

**Example: LLC4320 model output**
```python
# Requires evenly-spaced time: X_2D (nx, ny), Y_2D (nx, ny), t (nt)
X_2D = ds['distX'].values
Y_2D = ds['distY'].values
t = ds['time'].values - ds['time'].values[0]  # days since start

amp1, theta1, phi1, model1, var1, amps1, _, uncert1 = utils.fit_wave_frequency_domain(
    ssha, k, omega, X_2D, Y_2D, t
)
```

## 3. Example Notebooks

The fitting algorithms are implemented in `utils.py`, which provides the core functions `fit_wave()` (time-domain method) and `fit_wave_frequency_domain()` (frequency-domain method), along with coordinate conversion.

The repository includes four Jupyter notebooks demonstrating usage:
1. `01_synthetic_data_example.ipynb`: Demonstrates the time-domain method on synthetic data containing three internal tides, white noise, and correlated eddy fields with varying amplitudes.
2. `02_SWOT_data_example.ipynb`: Demonstrates the time-domain method on irregularly-sampled SWOT satellite observations in the South Atlantic, producing the results shown in \autoref{fig:example_timedomain}.
3. `03_access_internal_tide_parameters.ipynb`: Demonstrates how to access and visualize the Zenodo-archived internal tide parameter database.
4. `04_llc4320_plane_wave_fit_frequency_domain.ipynb`: Demonstrates the frequency-domain method and compares it to the time-domain method using MITgcm LLC4320 model SSH outputs in the Tasman Sea, producing the results shown in \autoref{fig:example_frequency_domain}.

# Research impact

PlaneWaveFit provides a modular, open-source implementation for extracting internal tides from SSH data. Applied to SWOT observations in the Southern Ocean [@li2026], the software successfully extracts coherent tidal beams and refraction patterns even in the presence of strong mesoscale variability. The technique has been used to map internal tides globally using multi-satellite altimetry [@zhao2016global; @zaron2019baroclinic; @zhao2024internal]. By combining fitted amplitudes with the precomputed database, researchers can estimate depth-integrated energy and flux directly from satellite observations, advancing understanding of tidal dissipation and mixing.


# Future Directions

The current implementation provides amplitude and phase uncertainties from the least-squares covariance matrix, but propagation direction — determined by a discrete 1° scan — does not have a corresponding formal uncertainty estimate. The angular width of the amplitude peak in the polar scan could be used to characterize directional uncertainty, where a narrow peak indicates a well-constrained propagation direction and a broad peak indicates a larger uncertainty in propagation direction. Bootstrap 
resampling over observational cycles could also be used to obtain a more rigorous confidence interval that accounts for all sources of variability, including mesoscale contamination.

The current Zenodo dataset uses annual-mean stratification from WOA23 to compute internal tide parameters. Stratification exhibits seasonal variability, particularly in mid-latitudes. Future versions could provide seasonal parameter sets (e.g., winter, spring, summer, fall) to improve accuracy when analyzing data from specific seasons.

The current approach treats all observations equally. A weighted least-squares implementation would allow users to downweight noisy observations or incorporate prior knowledge about expected wave characteristics. This could improve fits when observation error varies spatially (e.g., near satellite ground track gaps or energetic boundary currents) or when prior information about expected propagation directions is available (e.g., from known generation sites).

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements

YL acknowledges support from NASA FINESST award 80NSSC22K1529. STG and MRM were supported by NASA SWOT science team (awards 80NSSC20K1136, 80NSSC24K1657). MRM was supported by NOAA awards NA23NOS4000334, NA20OAR4320278, and NASA Modeling, Analysis, and Prediction Program (JPL Subcontract 1716197). STG acknowledges support from NASA Ocean Surface Topography Science Team (80NSSC21K1822).

# References
