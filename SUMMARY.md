# Tidal Debris Forecast for Dwarf Galaxies

Sensitivity study for next-generation spectroscopic surveys to detect tidal debris around MW and M31 satellite galaxies.

## Science Overview

**Goal:** Determine how many stars from tidal debris are observable (z-band magnitude < threshold) in the 5-20 r_half annulus around dwarf satellite galaxies, for a range of debris mass fractions.

**Method:**
1. Get satellite properties (stellar mass, r_half, distance, metallicity) from LVDB
2. Stellar mass follows a Plummer profile; M_half = M_total/2 within r_half
3. A fraction `f` of M_half exists as tidal debris in the 5-20 r_half annulus
4. Sample stars from Kroupa IMF via `imf.make_cluster()`, remove >1 Msun (dead after 12 Gyr)
5. Use `minimint` for DECam photometry (g, r, i, z, V bands) at the satellite's distance
6. Count observable stars above z-band magnitude limits (19, 21, 23)
7. Debris area = pi * ((20*r_half)^2 - (5*r_half)^2), converted to deg^2

**Key parameters:**
- `f_values = [0.001, 0.01, 0.1, 1.0]` (debris mass fraction)
- `Zmag_values = [19, 21, 23]` (z-band magnitude limits)
- Age = 12 Gyr, metallicity from LVDB (default -2.0)
- Reference cluster = 50,000 Msun (Kroupa IMF)
- M/L = 1.6 for M_V conversion

## Completed Tasks

### Task 1: Individual satellite plots
- 65 MW satellites: `satellite_plots/mw/*.png`
- 40 M31 satellites: `satellite_plots/m31/*.png`
- Each plot has 3 panels: N_remnant vs Zmag, density vs Zmag, CMD
- Covers f = [0.001, 0.01, 0.1, 1.0]

### Task 2: Summary plots
- `mw_satellites_debris.png` - MW 4-panel summary (f=0.01, Zmag=23)
- `m31_satellites_debris.png` - M31 4-panel summary (f=0.01, Zmag=23)
- 4 panels: bar chart, N vs distance, N vs M_V, density vs M_V

### Task 3: Simulation comparison
- Loaded N-body simulation data (`sat_arrs_ting.npy`, 922 satellites)
- Computed debris fraction f_{5-20} = M_star(5-20 r_h) / M_star(<r_h) from simulations
- Compared with MW/M31 satellite properties

**Key simulation results (r_sub < 200 kpc, N=548):**
- Median f_{5-20} = 0.022 (slightly above fiducial f=0.01)
- ~75% have f > 0.001
- ~45% have f > 0.01
- ~20% have f > 0.1
- Heavily stripped (f_bound < 0.5): median f_{5-20} = 0.43

### Stellar mass-size relation
- Derived from M_V-size relation with M/L=1.6:
  `log(r_half/pc) = 0.31 * log10(M_star/Msun) + 0.4875`
- Validated against MW and M31 LVDB data

## IMF Discussion (Resolved)
- User's approach: `imf.chabrier2005.distr.rvs(N)` draws N stars
- Pipeline approach: `imf.make_cluster(M)` targets M Msun total mass
- Both agree when calibrated to same **surviving** stellar mass
- Key insight: scale by Msun (not star count). For Draco at f=0.01:
  - 3013 Msun surviving -> ~52 stars (Chabrier) or ~42 stars (Kroupa)
- Pipeline uses Kroupa IMF

## File Inventory

### Main scripts
| File | Description |
|------|-------------|
| `tidal_debris_forecast.py` | Main pipeline: loads LVDB data, generates individual + summary plots for MW and M31 |
| `sim_debris_fraction.py` | 2x3 panel: f_{5-20} vs galaxy properties, color-coded by f_bound |
| `sim_debris_fraction_ticks.py` | Same 2x3 but with MW/M31 tick marks on x-axes; M31 distances from M31 center |
| `sim_f520_cumulative.py` | Cumulative distribution of f_{5-20}, with r_sub < 200 kpc cut |
| `sim_infall_mass_hist.py` | Histogram of infall stellar mass (M_bound / f_bound) |
| `mstar_rhalf_check.py` | Sanity check: M_star vs r_half relation against LVDB data |

### IMF comparison scripts (from debugging phase)
| File | Description |
|------|-------------|
| `compare_imf.py` | First IMF comparison |
| `compare_imf_v2.py` | Detailed 4-approach comparison |
| `verify_pipeline.py` | Verified pipeline gives ~42 for Draco |
| `compare_kroupa_chabrier.py` | Kroupa vs Chabrier, surviving mass scaling |
| `trace_user_code.py` | Reproduced user's exact rvs() code |

### Data files
| File | Description |
|------|-------------|
| `mw_satellites.csv` | 65 MW dwarf satellites from LVDB |
| `m31_satellites.csv` | 40 M31 satellites from LVDB (dwarf_m31 table) |
| `sat_arrs_ting.npy` | N-body simulation data (922 satellites, 16 columns) |

### Simulation data columns (`sat_arrs_ting.npy`)
```
[0]  f_star_bound        - fraction of stars still bound
[1]  f_bound_dm          - fraction of DM still bound
[2]  m_star_bound        - bound stellar mass
[3]  r_half_sm_2D        - 2D stellar half-mass radius [kpc]
[4]  M_half_2D           - mass within r_half
[5]  slope               - density slope
[6]  v_disp_in           - inner velocity dispersion
[7]  v_disp_out          - outer velocity dispersion
[8]  SM_bins[0]          - cumulative stellar mass within 1 r_half
[9]  SM_bins[1]          - cumulative stellar mass within ~3 r_half
[10] SM_bins[2]          - cumulative stellar mass within 5 r_half
[11] SM_bins[3]          - cumulative stellar mass within 10 r_half
[12] SM_bins[4]          - cumulative stellar mass within 20 r_half
[13] r_sub               - distance from host center [kpc]
[14] i                   - snapshot index
[15] Halo                - halo ID
```

### Output plots
| File | Description |
|------|-------------|
| `mw_satellites_debris.png` | MW summary (f=0.01, z<23) |
| `m31_satellites_debris.png` | M31 summary (f=0.01, z<23) |
| `sim_debris_fraction.png` | 2x3: f_{5-20} vs properties, color by f_bound |
| `sim_debris_fraction_MV.png` | f_{5-20} vs M_V standalone |
| `sim_debris_fraction_ticks.png` | 2x3 with MW/M31 tick marks |
| `sim_f520_cumulative.png` | Cumulative f_{5-20} distribution |
| `sim_infall_mass_hist.png` | Infall mass histogram |
| `mstar_rhalf_check.png` | M_star vs r_half validation |
| `satellite_plots/mw/*.png` | 65 individual MW satellite plots |
| `satellite_plots/m31/*.png` | 40 individual M31 satellite plots |

## Dependencies
```
pip install minimint imf astropy matplotlib numpy
```
- `minimint`: MIST isochrone interpolator for photometry
- `imf`: IMF sampling (`make_cluster` for Kroupa)
- LVDB: accessed via `astropy.table.Table.read()` from vizier/local CSV

## Possible Next Steps
- Switch pipeline from Kroupa to Chabrier IMF (user uses Chabrier)
- Use simulation f_{5-20} distribution as realistic prior instead of fixed f values
- Combine simulation predictions with observational forecasts (assign each observed satellite a predicted f based on its properties)
- Add spectroscopic survey footprint / fiber allocation constraints
- Include background contamination estimates
