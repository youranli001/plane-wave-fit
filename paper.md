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
date: 4 February 2026
bibliography: paper.bib
---

# Summary

Internal tides are internal gravity waves generated when tidal flows interact with seafloor topography. These waves transport energy across ocean basins and drive mixing that affects global ocean circulation and climate. In this paper, we introduce PlaneWaveFit, an open-source software package with a Python interface that extracts the amplitude, phase, and propagation direction of internal tides from Sea Surface Height (SSH) observations, with uncertainty estimates for both amplitude and phase provided \autoref{fig:example}. 

The software implements a two-dimensional plane-wave fitting approach to resolve multiple wave components. It is broadly applicable to data from satellite missions, such as the Surface Water and Ocean Topography (SWOT) mission, numerical ocean model outputs, and synthetic datasets. It also provides built-in access to precomputed internal tide parameters derived from the climatology in the World Ocean Atlas dataset. These parameters specifically target the dominant mode-1 $M_2$ internal tide and include the necessary conversion ratios to calculate depth-integrated energy and energy flux directly from the fitted surface amplitudes.

![Example of plane wave fit performed. Top left: A 2D snapshot of sea surface height anomalies (SSHA) from the SWOT satellite on April 2 2023. Remaining panels: Results of the iterative plane-wave fitting algorithm. The polar plots show the fitted amplitude (mm) as a function of propagation direction for the three most energetic wave components identified. Each panel includes a summary of the extracted wave parameters—maximum amplitude, propagation angle, and phase.](figure.png){#fig:example}



# Statement of need

High-resolution ocean observations and numerical models provide two-dimensional sea surface height (SSH) snapshots that resolve small-scale features. However, identifying internal tides in these datasets is challenging because the tidal signal is often obscured by vigorous mesoscale eddies and ocean fronts.

Commonly used packages like `UTide`[@codiga2011unified] or `T_TIDE` [@pawlowicz2002classical] are designed for temporal harmonic analysis at fixed locations. While effective for stationary records like tide gauges, they do not resolve the spatial propagation of internal tides. PlaneWaveFit addresses this limitation by implementing a two-dimensional fitting approach that extracts $M_2$ internal tides that are both spatially and temporally coherent. This dual coherence constraint makes the method significantly less sensitive to mesoscale contamination, allowing for the robust detection of tidal beams even in energetic ocean regions.

# State of the field
Two-dimensional plane-wave fitting has been widely applied to extract low-mode internal tidal signals from both multi-satellite altimetry data [@zhao2011internal; @zhao2016global; @zhao2017global; @zaron2019baroclinic; @zhao2024internal]. PlaneWaveFit provides a open-source, documented implementation with uncertainty quantification.

The package complements vertical mode solvers like `dynmodes` [@klinck_dynmodes]. While `dynmodes` computes theoretical structures and eigenspeeds from stratification, PlaneWaveFit extends this in two ways. First, it provides global precomputed parameters from World Ocean Atlas 2023 for users without local stratification data. Second, it includes conversion factors needed to translate SSH amplitudes into full-depth energy estimates. This allows users to either perform standalone 2D fits or combine them with the physical parameters to estimate energetics directly.

# Software design

PlaneWaveFit consists of two main components:

## 1. Internal Tide Parameter Database

The package provides programmatic access to a Zenodo-archived dataset [@li2025internal] containing $M_2$ internal tide parameters for vertical modes 1-10 on a global 0.25° grid. The dataset includes:

- Vertical mode structures (pressure, horizontal velocity, vertical velocity)
- Wave properties (phase speeds, group speeds and horizontal wavenumbers)
- Conversion ratios relating SSH amplitude to:
  - Maximum vertical isopycnal displacement
  - Depth-integrated potential, kinetic, and total energy density
  - Depth-integrated horizontal energy flux

For example, a 4.21 mm fitted surface amplitude (\autoref{fig:example}) can be converted into a depth-integrated horizontal energy flux of 0.35 kW/m using the provided conversion factor ($F \approx 1.99 \times 10^7$ W/m), without requiring the user to perform manual vertical integration of stratification profiles. The physical derivation of these factors is detailed in (DOI to be inserted after preprint is released).

## 2. Plane-Wave Fitting Algorithm
Internal tide parameters (amplitude $A$, phase $\phi$, and direction $\theta$) are estimated by fitting plane waves to sea surface height anomalies (SSHA). For a specified tidal frequency $\omega$ and horizontal wavenumber $k$ (provided by the user or obtained from the precomputed lookup table), the wave field is written as:

$$\eta(x, y, t) = \sum_{m=1}^{N} A_m \cos(k_{x,m} x + k_{y,m} y - \omega t - \phi_m)$$

The optimal propagation direction $\theta$ is determined using a directional scanning approach. The algorithm evaluates all compass directions from $1^\circ$ to $360^\circ$; at each angle, a linear least-squares problem is solved to estimate the sine and cosine coefficients $(\beta_1, \beta_2)$. The direction associated with the maximum fitted amplitude is selected as the dominant propagation direction for the wave component under consideration.

Following the selection of the optimal propagation direction, parameter uncertainties are estimated from the covariance matrix of the least-squares solution. The covariance is derived from the residual variance of the fitted model.

### Usage Examples


The following example illustrates the use of fit_wave functionality to extract wave amplitude, phase, and direction from SWOT data. Note that the ssh data is achived in zenodo, and will be downloaded to a `data/` directory within your current working folder.

```python
from zenodo_get import download
import utils  

# 1. Download the SWOT SSHA example dataset
doi = "10.5281/zenodo.18408783"
ssha_fn = "SWOT_CalVal_SSHA_35W_35p5S.nc"
data_dir = Path("data/zenodo")
download(record_or_doi=doi, output_dir=data_dir, file_glob=ssha_fn)
ds = xr.open_dataset(data_dir / ssha_fn) # Load the dataset

# 2. Plot a snapshot of SSHA (Optional)
cycle = 0
fig, ax = plt.subplots(figsize=(6, 4))
pcm = ax.pcolormesh(
    ds.longitude,
    ds.latitude,
    ds.ssha.isel(num_cycles=cycle),
    shading="auto",
    cmap="RdBu_r"
)
cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label("SSHA (m)")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_title(f"SWOT SSHA (Cycle {cycle})")
ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

# 3. Prepare Input Coordinates
# Convert Lat/Lon to Cartesian coordinates (km)
lon_example, lat_example = -35.0, -35.5
distX, distY = utils.lonlat2xy(
ds.longitude.values, ds.latitude.values, lon_example, lat_example)

# Reshape data into (Time, Y, X)
# dimensions: (num_cycles, num_lines, num_pixels)
ssha = ds['ssha'].values
X_3D = np.repeat(
    distX[np.newaxis, :, :], 
    ds.sizes['num_cycles'], 
    axis=0
)
Y_3D = np.repeat(
    distY[np.newaxis, :, :], 
    ds.sizes['num_cycles'], 
    axis=0
)

#  Convert time to decimal days
T_3D_dt64 = np.repeat(
    ds['time'].values[:, :, np.newaxis], 
    ds.sizes['num_pixels'], 
    axis=2
)
T_3D = utils.datetime64_to_matlab_datenum(T_3D_dt64)

# 4. Define Wave Parameters (M2 Internal Tide)
# Frequency [rad/day] and Wavenumber [rad/km]
omega = 2 * np.pi / (12.42 * 3600) * 86400  
k = 0.04124732611475162

# 5. Apply the Plane Wave Fit (Iterative approach)
# Iteration 1: Fit dominant wave
amp1, theta1, phi1, model1, var1, _, _, uncert1 = utils.fit_wave(
    ssha, k, omega, X_3D, Y_3D, T_3D
)

# Iteration 2: Subtract first wave and fit residuals
amp2, theta2, phi2, model2, var2, _, _, uncert2 = utils.fit_wave(
    ssha - model1, k, omega, X_3D, Y_3D, T_3D
)

# Iteration 3: Subtract second wave and fit residuals
amp3, theta3, phi3, model3, var3, _, _, uncert3 = utils.fit_wave(
    ssha - model1 - model2, k, omega, X_3D, Y_3D, T_3D
)

```

# Research impact statement

PlaneWaveFit provides an open-source, modular implementation of 2D plane-wave fitting for internal tides. Applied to SWOT data in the Southern Ocean [cite my paper], the software successfully extracted coherent tidal beams and refraction patterns in regions with strong mesoscale eddies. By combining fitted surface amplitudes with the precomputed conversion database, researchers can estimate depth-integrated energy and flux directly from satellite observations. 

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements
YL acknowledges support from a NASA FINESST award 80NSSC22K1529. STG and MRM were supported by the NASA SWOT science team (NASA awards 80NSSC20K1136 and  80GSFC24CA067). MRM was supported by NOAA awards NA23NOS4000334 and NA20OAR4320278, and the NASA Modeling, Analysis, and Prediction Program (Jet Propulsion Laboratory Subcontract 1716197). STG also acknowledges support from the Ocean Surface Topography Science Team (NASA 80NSSC21K1822).

# References

