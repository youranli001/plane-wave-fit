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
  - name: Sarah Gille
    orcid: 0000-0001-9144-4368
    affiliation: 1    
  - name: Matthew Mazloff
    orcid: 0000-0002-1650-5850
    affiliation: 1 
affiliations:
  - name: Scripps Institution of Oceanography, University of California San Diego, USA
    index: 1
date: 14 February 2026
bibliography: paper.bib
---

# Summary

PlaneWaveFit extracts amplitude, phase, and propagation direction of internal tides from sea surface height (SSH) observations, with quantified uncertainties. The package implements two complementary plane-wave fitting approaches: (1) a time-domain method applicable to any spatiotemporal dataset, including irregularly-sampled observations, and (2) a frequency-domain method for evenly-sampled data (e.g., hourly model output). Both resolve multiple wave components and provide uncertainty estimates. The package also provides programmatic access to a companion Zenodo dataset of precomputed mode-1 M$_2$​ internal tide parameters on a global 0.25° grid, enabling direct conversion from fitted surface amplitudes to depth-integrated energy and flux without requiring local stratification data.

# Statement of need

Extracting internal tides from SSH observations requires separating a tidal signal of O(1 cm) from mesoscale variability of O(1 m). Widely-used packages like `UTide` [@codiga2011unified] and `T_TIDE` [@pawlowicz2002classical] perform temporal harmonic analysis at fixed locations but cannot resolve spatial propagation. PlaneWaveFit implements two-dimensional fitting that extracts M$_2$ internal tides coherent in both space and time. This dual coherence constraint reduces sensitivity to mesoscale contamination, enabling robust detection of tidal beams in energetic regions. In synthetic tests, fitted amplitudes, phases, and propagation directions match the prescribed 1 cm values with correlated eddy noise up to 20 cm RMS.

# State of the field

Two-dimensional plane-wave fitting has proven effective for extracting internal tides from SSH data [@zhao2011internal; @zhao2016global; @zhao2017global; @zaron2019baroclinic; @zhao2024internal], but documented, open-source implementations with uncertainty quantification have been lacking. PlaneWaveFit addresses this gap.

The package complements vertical mode solvers like dynmodes [@klinck_dynmodes], which compute vertical mode structures from a stratification profile. PlaneWaveFit, following Appendix A of @li2026, uses those outputs to compute additional internal tide parameters, including wavelengths, phase and group speeds, and conversion ratios linking SSH amplitude to depth-integrated energy and flux from WOA23 climatology and archived in a Zenodo dataset.

# Software design

## 1. Internal Tide Parameter Database

Programmatic access to a Zenodo-archived dataset [@li2025internal] containing M$_2$ internal tide parameters (modes 1-10, global 0.25° grid):
- Vertical mode structures of pressure, horizontal velocity, vertical velocity, and vertical displacement
- Wave properties: phase speeds, group speeds, and wavelengths
- Conversion ratios relating SSH amplitude to:
  - Maximum vertical isopycnal displacement
  - Depth-integrated potential, kinetic, and total energy density
  - Depth-integrated horizontal energy flux

The physical derivation of these conversion ratios is detailed in Appendix A of @li2026. Example: a 4.12 mm fitted surface amplitude at 35°W, 35.5°S corresponds to a depth-integrated horizontal energy flux of 0.35 kW/m using the precomputed conversion ratio at that location. Users with local stratification profiles can compute these directly following Appendix A of @li2026; those working with SSH observations alone can use the precomputed database.

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

where $\omega$ is the M$_2$ angular frequency and $k$ is the horizontal wavenumber (provided by the user or obtained from the precomputed internal tide parameter database).

### Time-Domain Method

For each compass direction (angular increment 1°), least-squares fitting determines the amplitude and phase of a plane wave. When plotted in polar coordinates, a wave component appears as a lobe; the largest lobe gives the amplitude and direction of the dominant wave (\autoref{fig:example_timedomain}). That wave is then predicted and subtracted, and the procedure repeated to extract additional components. Amplitude and phase uncertainties derive from the least-squares covariance matrix. Details of the fitting procedure and uncertainty derivation are given in @li2026.

![Plane wave fit applied to SWOT SSHA data. Top left: SWOT SSHA near 35°W, 35.5°S on April 2, 2023. Remaining panels: polar plots showing fitted amplitude (mm) versus propagation direction for the three most energetic wave components; arrows indicate the selected propagation direction.](figure1.png){#fig:example_timedomain}

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

For evenly-sampled data, the frequency-domain method uses a two-step hybrid approach. Step 1 applies temporal FFT to extract the M$_2$ component, reducing the 3D spatiotemporal problem to 2D spatial fitting. The complex spatial field is:

$$B_{M_2}(x, y) = \sum_{m=1}^{N} A_m \cos(k x \cos\theta_m + k y \sin\theta_m + \phi_m)$$

We perform 360 directional scans (1°–360°), fitting this spatial pattern at each angle. Because waves at angles $\theta$ and $\theta + 180°$ create identical spatial patterns, the FFT magnitude cannot distinguish propagation direction, producing two-lobe polar plots. Step 2 resolves this 180° ambiguity by testing both candidate directions in the time domain and selecting the direction with larger amplitude.

This two-step approach achieves ~180× speedup compared to time-domain alone. The speedup comes from collapsing the time dimension via FFT: instead of 360 time-domain fits (each fitting x, y, t data), we perform 360 fast spatial fits (x, y data only) plus 2 time-domain fits. Tested on hourly LLC4320 model output spanning 90 days, runtime reduced from ~6 hours to ~2 minutes on a standard workstation. This method requires evenly-spaced time samples but can handle irregular spatial sampling.


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

# Research impact

PlaneWaveFit provides a modular, open-source implementation for extracting internal tides from SSH data. Applied to SWOT observations in the Southern Ocean [@li2026], the software successfully extracted coherent tidal beams and refraction patterns despite strong mesoscale variability. The technique has been used to map internal tides globally using multi-satellite altimetry [@zhao2016global; @zaron2019baroclinic; @zhao2024internal]. By combining fitted amplitudes with the precomputed database, researchers can estimate depth-integrated energy and flux directly from satellite observations, advancing understanding of tidal dissipation and mixing.

<!--
# Future directions

The current implementation quantifies amplitude and phase uncertainties from the least-squares covariance matrix, but does not provide a formal uncertainty estimate for propagation direction. Direction is determined by a discrete 1° scan and selected at the angle of maximum amplitude — a procedure that does not naturally yield a covariance-based standard error.

A practical proxy for directional sharpness is the angular half-power width of the amplitude peak in the polar scan: a narrow peak indicates a well-constrained direction, while a broad peak indicates that many directions fit the data nearly equally well. This quantity requires no additional computation and could be added as a default output alongside the existing amplitude and phase uncertainties.

A statistically rigorous confidence interval for propagation direction could be obtained through bootstrap resampling over observational cycles. By repeatedly subsampling the available observations and refitting, the spread of direction estimates across replicates provides a formal uncertainty measure that accounts for all sources of variability, including mesoscale contamination. This approach is computationally expensive for global applications but feasible at individual locations.

-->


# Future Directions

The current implementation provides amplitude and phase uncertainties from the least-squares covariance matrix, but propagation direction — determined by a discrete 1° scan — does not have a corresponding formal uncertainty estimate. The angular width of the amplitude peak in the polar scan could be used to characterize directional uncertainty, where a narrow peak indicates a well-constrained propagation direction and a broad peak indicates a larger uncertainty in propagation direction. Bootstrap 
resampling over observational cycles could also be used to obtain a more rigorous confidence interval that accounts for all sources of variability, including mesoscale contamination.

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements

YL acknowledges support from NASA FINESST award 80NSSC22K1529. STG and MRM were supported by NASA SWOT science team (awards 80NSSC20K1136, 80GSFC24CA067). MRM was supported by NOAA awards NA23NOS4000334, NA20OAR4320278, and NASA Modeling, Analysis, and Prediction Program (JPL Subcontract 1716197). STG acknowledges support from NASA Ocean Surface Topography Science Team (80NSSC21K1822).

# References
