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

Internal tides are internal gravity waves generated when tidal flows interact with seafloor topography. These waves transport energy across ocean basins and drive mixing that affects global ocean circulation and climate. In this paper, we introduce PlaneWaveFit, an open-source software package with a Python interface that extracts the amplitude, phase, and propagation direction of internal tides from Sea Surface Height (SSH) observations, with uncertainty estimates for both amplitude and phase provided \autoref{fig:figure1}. 


The software implements two complementary plane-wave fitting approaches: a time-domain method applicable to any spatiotemporal dataset (including irregularly-sampled satellite missions such as SWOT), and a frequency-domain method optimized for evenly-sampled, high-temporal-resolution datasets (e.g., hourly numerical model outputs). Both methods resolve multiple wave components. It also provides built-in access to precomputed internal tide parameters derived from the climatology in the World Ocean Atlas dataset. These parameters specifically target the dominant mode-1 $M_2$ internal tide and include the necessary conversion ratios to calculate depth-integrated energy and energy flux directly from the fitted surface amplitudes.

<!--
The software implements a two-dimensional plane-wave fitting approach to resolve multiple wave components. It is broadly applicable to data from satellite missions, such as the Surface Water and Ocean Topography (SWOT) mission, numerical ocean model outputs, and synthetic datasets. It also provides built-in access to precomputed internal tide parameters derived from the climatology in the World Ocean Atlas dataset. These parameters specifically target the dominant mode-1 $M_2$ internal tide and include the necessary conversion ratios to calculate depth-integrated energy and energy flux directly from the fitted surface amplitudes.
-->
![Example of plane wave fit performed. Top left: Sea surface height anomalies (SSHA) from the SWOT satellite on April 2 2023. Remaining panels: Results of the iterative plane-wave fitting algorithm. The polar plots show the fitted amplitude (mm) as a function of propagation direction for the three most energetic wave components identified. Each panel includes a summary of the extracted wave parameters—maximum amplitude, propagation angle, and phase.](figures/figure1.png){#fig:figure1}



# Statement of need

High-resolution ocean observations and numerical models provide two-dimensional sea surface height (SSH) snapshots that resolve small-scale features. However, identifying internal tides in these datasets is challenging because the tidal signal is often obscured by vigorous mesoscale eddies and ocean fronts.

Commonly used packages like `UTide`[@codiga2011unified] or `T_TIDE` [@pawlowicz2002classical] are designed for temporal harmonic analysis at fixed locations. While effective for stationary records like tide gauges, they do not resolve the spatial propagation of internal tides. PlaneWaveFit addresses this limitation by implementing a two-dimensional fitting approach that extracts $M_2$ internal tides that are both spatially and temporally coherent. This dual coherence constraint makes the method significantly less sensitive to mesoscale contamination, allowing for the robust detection of tidal beams even in energetic ocean regions.

# State of the field
While two-dimensional plane-wave fitting methods have proven effective for extracting internal tidal signals from SSH data [@zhao2011internal; @zhao2016global; @zhao2017global; @zaron2019baroclinic; @zhao2024internal], open-source implementations with documentation and uncertainty quantification have been lacking. PlaneWaveFit addresses this gap.

The package complements vertical mode solvers like `dynmodes` [@klinck_dynmodes]. While `dynmodes` computes theoretical structures from stratification, PlaneWaveFit bridges the gap to observations. It integrates a global database of precomputed conversion factors (derived from World Ocean Atlas 2023), allowing users to translate fitted surface amplitudes directly into full-depth energetics without requiring local stratification data.



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

### Time-Domain Method
The optimal propagation direction $\theta$ is determined using a directional scanning approach. The algorithm evaluates all compass directions from $1^\circ$ to $360^\circ$; at each angle, a linear least-squares problem is solved. The direction associated with the maximum fitted amplitude is selected as the dominant propagation direction for the wave component under consideration. Parameter uncertainties are estimated from the covariance matrix of the least-squares solution, derived from the residual variance of the fitted model. This method is applicable to any spatiotemporal dataset, including irregularly sampled observations.

<!--
Following the selection of the optimal propagation direction, parameter uncertainties are estimated from the covariance matrix of the least-squares solution. The covariance is derived from the residual variance of the fitted model.
-->

#### Usage Examples


The following example illustrates the use of fit_wave functionality to extract wave amplitude, phase, and direction from SWOT data. Note that the ssh data is archived in zenodo, and will be downloaded to a `data/` directory within your current working folder.

```python
from zenodo_get import download
import utils  

# Download the SWOT SSHA example dataset
doi = "10.5281/zenodo.18408783"
ssha_fn = "SWOT_CalVal_SSHA_35W_35p5S.nc"
data_dir = Path("data/zenodo")
download(record_or_doi=doi, output_dir=data_dir, file_glob=ssha_fn)
ds = xr.open_dataset(data_dir / ssha_fn) # Load the dataset

# Prepare Input Coordinates
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

# Define Wave Parameters (M2 Internal Tide)
omega = 2 * np.pi / (12.42 * 3600) * 86400 # Frequency [rad/day]
k = 0.04124732611475162 # Wavenumber [rad/km]

# Apply the Plane Wave Fit (Iterative approach)
# Iteration 1: Fit dominant wave
amp1, theta1, phi1, model1, var1, _, _, uncert1 = utils.fit_wave(
    ssha, k, omega, X_3D, Y_3D, T_3D
)

# Iteration 2: Subtract first wave and fit residuals
amp2, theta2, phi2, model2, var2, _, _, uncert2 = utils.fit_wave(
    ssha - model1, k, omega, X_3D, Y_3D, T_3D
)

```
### Frequency-Domain Method

For evenly-sampled datasets (e.g., hourly model output), the frequency-domain method achieves ~180× computational speedup using a two-stage hybrid approach. First a temporal Fast Fourier Transform (FFT) is applied to extract the M$_2$ tidal component, reducing the 3D spatiotemporal problem to a 2D spatial problem. The complex-valued spatial field at the M$_2$ frequency is then modeled as:

$$
B_{M_2}(x, y)
=
\sum_{m=1}^{N}
A_m \cos\!\left(
k\,x\cos\theta_m
+
k\,y\sin\theta_m
+ \phi_m
\right)
$$

where $A_m$ is the amplitude, $\theta_m$ is the propagation direction, and $\phi_m$ is the spatial phase. The method performs 360 spatial fits (one per degree) to identify propagation directions. However, because the FFT has collapsed the time dimension, this produces two-lobe polar plots (insert figure) because waves at angles $\theta$ and $\theta + 180^\circ$ create identical spatial patterns—the FFT magnitude cannot distinguish propagation direction. Then this 180$^\circ$ ambiguity is resolved with plane wave fit in the time domain (testing both θ and θ+180°) and selecting the direction with larger amplitude. This approach is ~180× faster than time-domain only but requires evenly spaced time samples. \autoref{fig:figure2}. 

![Frequency-domain plane wave fit applied to LLC4320 model SSHA in the Tasman Sea. Top left: SSHA. Remaining panels: Directional amplitude scans for the three dominant wave components. The polar plots show fitted amplitude (m) as a function of propagation direction. Two-lobe patterns result from 180° directional ambiguity in the spatial FFT. Arrows indicate final propagation directions after time-domain disambiguation.](figures/figure2.png){#fig:figure2}


```python
# Frequency-domain requires 2D spatial grids + 1D time
X_2D = ds['distX'].values  # Shape: (nx, ny)
Y_2D = ds['distY'].values  # Shape: (nx, ny)
t = time - time[0]  # shape : (nt) # days

# plane wave fit
amp1, theta1, phase1, model1, var1, amps1, _, uncert1 = utils.fit_wave_frequency_domain(
    ssha, k, omega, X_2D, Y_2D, t
)
```


# Research impact statement

PlaneWaveFit provides an open-source, modular implementation of two-dimensional plane-wave fitting for internal tides. Applied to SWOT data in the Southern Ocean [@li2026], the software successfully extracted coherent tidal beams and refraction patterns in regions with strong mesoscale eddies. The technique has been applied to map internal tides globally using multi-satellite altimetry [@zhao2016global; @zaron2019baroclinic; @zhao2024internal]. By combining fitted surface amplitudes with the precomputed conversion database, researchers can estimate depth-integrated energy and flux directly from satellite observations, facilitating a deeper understanding of tidal dissipation and mixing processes.



# Future Directions

When handling datasets with high temporal resolution (e.g., hourly model output) over long time windows, fitting plane waves directly in the time domain can become computationally expensive. Future versions may implement a frequency-domain alternative. In this approach, the time series is first transformed using a Fourier transform to extract the complex field at the target tidal frequency (e.g., $M_2$). 


# AI usage disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements
YL acknowledges support from a NASA FINESST award 80NSSC22K1529. STG and MRM were supported by the NASA SWOT science team (NASA awards 80NSSC20K1136 and  80GSFC24CA067). MRM was supported by NOAA awards NA23NOS4000334 and NA20OAR4320278, and the NASA Modeling, Analysis, and Prediction Program (Jet Propulsion Laboratory Subcontract 1716197). STG also acknowledges support from the Ocean Surface Topography Science Team (NASA 80NSSC21K1822).

# References

