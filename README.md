# PlaneWaveFit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source Python package for extracting internal tides from sea surface height (SSH) data using two-dimensional plane-wave fitting.

## Overview

The two-dimensional plane-wave fitting technique was introduced by [Zhao et al. (2011)](https://doi.org/10.1175/2010JPO4547.1) and further developed in [Zhao et al. (2016)](https://doi.org/10.1002/2015JC011572). It has been widely used to map internal tides globally from multi-satellite altimetry, but no documented, open-source implementation with uncertainty quantification has been available. PlaneWaveFit provides such an implementation, adding a frequency-domain method for evenly sampled data and a companion Zenodo parameter database.

The core fitting functions are implemented in `utils.py`:

- **`fit_wave()`** — Time-domain method following Zhao et al. (2011). Scans 360 compass directions, performs least-squares fitting at each, and iteratively extracts multiple wave components. Handles irregularly sampled data (e.g., SWOT satellite swaths). Uncertainty estimates are derived from the least-squares covariance matrix.
- **`fit_wave_frequency_domain()`** — Frequency-domain method introduced in this package. Applies FFT to isolate the M2 component, reduces the problem from 3D (x, y, t) to 2D (x, y) spatial fitting, then disambiguates the 180° directional ambiguity with two time-domain fits. ~180× faster than the time-domain method. Requires evenly spaced time samples.

Helper functions (`lonlat2xy`, `datetime64_to_matlab_datenum`) handle coordinate conversion.

The package also provides access to a Zenodo-archived database of precomputed M2 internal tide parameters (modes 1–10, global 0.25° grid from WOA23), enabling conversion from fitted SSH amplitudes to depth-integrated energy and flux.

## Example Data

Two datasets are included in `data/`:

- **`SWOT_CalVal_SSHA_35W_35p5S.nc`** — Sea surface height anomaly from the SWOT satellite (near-daily repeat, 102 cycles) near 35°W, 35.5°S in the South Atlantic. SWOT is a wide-swath satellite altimeter launched in 2022 that resolves ocean variability down to 15–30 km scales.
- **`llc4320_ssha_tasman_155E_45S_subsample.nc`** — Sea surface height anomaly from the MITgcm LLC4320 simulation (hourly output, 40 days) at 155°E, 45°S in the Tasman Sea. LLC4320 is a 1/48° global ocean simulation with realistic tidal forcing.

The M2 internal tide parameter database (modes 1–10, global 0.25° grid) is hosted on Zenodo and accessed via notebook 03:

> Li, Y., Gille, S. T., Mazloff, M. R., & NASA (2026). Global M2 internal tide parameters for modes 1–10 derived from the World Ocean Atlas 2023 (WOA23) [Dataset]. Zenodo. [doi:10.5281/zenodo.18423546](https://doi.org/10.5281/zenodo.18423546)

## Notebooks

Four notebooks demonstrate the package from method validation to real-data application:

1. **`01_synthetic_data_example.ipynb`** — Validates the time-domain method on synthetic data containing three prescribed internal tides, white noise, and correlated eddy fields with varying amplitudes, showing that the method recovers the correct amplitude, phase, and direction.
2. **`02_SWOT_data_example.ipynb`** — Applies the time-domain method to irregularly sampled SWOT satellite observations in the South Atlantic, demonstrating the full workflow from data loading to iterative wave extraction with uncertainty estimates.
3. **`03_access_internal_tide_parameters.ipynb`** — Downloads and visualizes the companion Zenodo M2 parameter database, showing how to convert fitted SSH amplitudes to depth-integrated energy and flux.
4. **`04_LLC4320_data_example.ipynb`** — Applies the frequency-domain method to hourly MITgcm LLC4320 model output in the Tasman Sea, and compares with the time-domain method on the same data, demonstrating the significant speedup.

## Paper

The accompanying JOSS manuscript is included in the repository ([`paper.md`](paper.md)). Submission pending.

> Li, Y., Gille, S. T., & Mazloff, M. R. PlaneWaveFit: A Python package for two-dimensional plane-wave fitting of internal gravity waves. *Journal of Open Source Software*. Pending.

## Getting Started

```bash
git clone https://github.com/youranli001/plane-wave-fit.git
cd plane-wave-fit
pip install -r requirements.txt
```

Then open the notebooks in `notebooks/` to see usage examples, or import `utils.py` directly:

```python
import sys
sys.path.append("/path/to/plane-wave-fit")
import utils

# Time-domain fitting (irregularly sampled data)
amp, direction, phase, model, _, _, _, uncertainty = utils.fit_wave(
    ssha, k, omega, X_3D, Y_3D, T_3D
)

# Frequency-domain fitting (evenly sampled data, much faster)
amp, direction, phase, model, _, _, _, uncertainty = utils.fit_wave_frequency_domain(
    ssha, k, omega, X_2D, Y_2D, T_1D
)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, scipy | Core numerical computation and least-squares fitting |
| xarray, netCDF4 | Reading and handling NetCDF datasets |
| pandas | Time series handling |
| matplotlib, cartopy | Plotting and map projections |
| dask | Lazy loading of large datasets |
| tables (PyTables) | HDF5 file access for the parameter database |
| zenodo-get | Downloading datasets from Zenodo |

## License

[MIT](LICENSE)
