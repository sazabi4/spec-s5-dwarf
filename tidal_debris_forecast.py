#!/usr/bin/env python3
"""
Sensitivity study: detecting tidal debris around MW and M31 satellite galaxies
with next-generation spectroscopic surveys.

Uses: LVDB (satellite properties), minimint (isochrone photometry), imf (stellar mass sampling)
"""

import os
import numpy as np
import imf
import minimint
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import warnings
warnings.filterwarnings('ignore')

# ---- Constants ----
LOGAGE = np.log10(12e9)  # log10(12 Gyr in years) ~ 10.079
REFERENCE_MASS = 50000   # Msun for reference cluster sampling
OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5'
PLOTDIR_MW = os.path.join(OUTDIR, 'satellite_plots', 'mw')
PLOTDIR_M31 = os.path.join(OUTDIR, 'satellite_plots', 'm31')
os.makedirs(PLOTDIR_MW, exist_ok=True)
os.makedirs(PLOTDIR_M31, exist_ok=True)

# ---- Load data ----
mw = Table.read(os.path.join(OUTDIR, 'mw_satellites.csv'))
m31 = Table.read(os.path.join(OUTDIR, 'm31_satellites.csv'))

# ---- Initialize minimint ----
print("Initializing minimint interpolator...")
ii_global = minimint.Interpolator(['DECam_r', 'DECam_z'])
print("Done.")


# ===========================================================
# Core functions
# ===========================================================

def prepare_satellite(satellite_row, logage=LOGAGE, reference_mass=REFERENCE_MASS):
    """
    Generate a reference stellar population for a satellite and compute photometry.
    """
    feh = float(satellite_row['metallicity'])
    if np.isnan(feh):
        feh = -2.0
    dm = float(satellite_row['distance_modulus'])

    masses_all = imf.make_cluster(reference_mass, massfunc='kroupa')
    mask_survive = masses_all <= 1.0
    masses = masses_all[mask_survive]
    mass_surviving_total = float(masses.sum())

    n = len(masses)
    res = ii_global(masses, np.full(n, logage), np.full(n, feh))
    app_r = np.array(res['DECam_r']) + dm
    app_z = np.array(res['DECam_z']) + dm
    valid = np.isfinite(app_z) & np.isfinite(app_r)

    return {
        'masses': masses,
        'mass_surviving_total': mass_surviving_total,
        'mass_total_sampled': float(masses_all.sum()),
        'app_r': app_r,
        'app_z': app_z,
        'valid': valid,
        'dm': dm,
        'feh': feh,
        'n_surviving': len(masses),
        'n_valid_phot': int(np.sum(valid)),
    }


def count_debris_stars(satellite_row, prep, f, Zmag):
    """
    Count observable stars in tidal debris (5-20 rh) for given f and Zmag.
    Returns (N_remnant, total_app_mag_z).
    """
    M_total = 10 ** float(satellite_row['mass_stellar'])
    Nstar_half = M_total / 2.0
    debris_mass = f * Nstar_half
    scale = debris_mass / prep['mass_surviving_total']

    observable = prep['valid'] & (prep['app_z'] < Zmag)
    N_ref = np.sum(observable)
    N_remnant = N_ref * scale

    if N_ref > 0:
        flux_ref = np.sum(10 ** (-0.4 * prep['app_z'][observable]))
        flux_total = flux_ref * scale
        total_app_mag_z = -2.5 * np.log10(flux_total) if flux_total > 0 else np.inf
    else:
        total_app_mag_z = np.inf

    return N_remnant, total_app_mag_z


def compute_area_deg2(satellite_row):
    """Compute area between 5rh and 20rh in square degrees."""
    rhalf_arcmin = float(satellite_row['rhalf'])
    area_arcmin2 = np.pi * ((20 * rhalf_arcmin) ** 2 - (5 * rhalf_arcmin) ** 2)
    return area_arcmin2 / 3600.0


# ===========================================================
# Per-satellite 3-panel plot
# ===========================================================

def plot_satellite(satellite_row, save_dir=PLOTDIR_MW, n_real=10):
    """
    Generate a 3-panel plot for one satellite:
      Panel 1: N_remnant vs Zmag for f = [0.001, 0.01, 0.1]
      Panel 2: Number density (N/deg^2) vs Zmag for same f values
      Panel 3: Synthetic CMD for f=0.01

    Averages over n_real reference cluster realizations to reduce IMF noise.
    """
    key = str(satellite_row['key'])
    name_pretty = key.replace('_', ' ').title()
    mv_val = float(satellite_row['M_V'])

    area_deg2 = compute_area_deg2(satellite_row)

    # Run multiple realizations and average
    preps = []
    for _ in range(n_real):
        preps.append(prepare_satellite(satellite_row))
    # Use first prep for CMD plotting (feh, dm are the same across all)
    prep = preps[0]

    f_values = [0.001, 0.01, 0.1, 1.0]
    Zmag_values = [19, 21, 23]

    # Compute results grid, averaged over realizations
    results = np.zeros((len(f_values), len(Zmag_values)))
    density = np.zeros((len(f_values), len(Zmag_values)))

    for i, fv in enumerate(f_values):
        for j, zm in enumerate(Zmag_values):
            Ns = [count_debris_stars(satellite_row, p, fv, zm)[0]
                  for p in preps]
            N_mean = np.mean(Ns)
            results[i, j] = N_mean
            density[i, j] = N_mean / area_deg2 if area_deg2 > 0 else 0.0

    # Parameters annotation
    param_text = (f"Age = 12 Gyr (log age = {LOGAGE:.2f})\n"
                  f"[Fe/H] = {prep['feh']:.2f}\n"
                  f"Distance = {satellite_row['distance']:.1f} kpc "
                  f"(DM = {satellite_row['distance_modulus']:.2f})\n"
                  f"r_half = {satellite_row['rhalf']:.2f}' "
                  f"({satellite_row['rhalf_physical']:.0f} pc)")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50']
    markers = ['o', 's', '^', 'D']

    # ---- Panel 1: N_remnant vs Zmag ----
    ax = axes[0]
    for i, fv in enumerate(f_values):
        yvals = results[i, :]
        # Avoid log(0) by setting floor
        yvals_plot = np.where(yvals > 0, yvals, 1e-3)
        ax.semilogy(Zmag_values, yvals_plot,
                     color=colors[i], marker=markers[i], markersize=10,
                     linewidth=2, label=f'f = {fv}')
        for j, zm in enumerate(Zmag_values):
            val = results[i, j]
            if val >= 1:
                label = f'{val:.0f}'
            elif val >= 0.01:
                label = f'{val:.2f}'
            else:
                label = f'{val:.1e}'
            ax.annotate(label, (zm, yvals_plot[j]),
                       textcoords="offset points", xytext=(8, 5),
                       fontsize=9, color=colors[i])

    ax.set_xlabel('DECam z magnitude limit', fontsize=13)
    ax.set_ylabel('N_remnant (observable stars in 5-20 rh)', fontsize=13)
    ax.set_title(f'{name_pretty} (M$_V$={mv_val:.1f}): N_remnant', fontsize=13)
    ax.legend(fontsize=11, title='Debris fraction f', title_fontsize=10)
    ax.set_xticks(Zmag_values)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(18, 24)

    # ---- Panel 2: Number density (N/deg^2) vs Zmag ----
    ax = axes[1]
    for i, fv in enumerate(f_values):
        yvals = density[i, :]
        yvals_plot = np.where(yvals > 0, yvals, 1e-5)
        ax.semilogy(Zmag_values, yvals_plot,
                     color=colors[i], marker=markers[i], markersize=10,
                     linewidth=2, label=f'f = {fv}')
        for j, zm in enumerate(Zmag_values):
            val = density[i, j]
            if val >= 1:
                label = f'{val:.1f}'
            elif val >= 0.01:
                label = f'{val:.3f}'
            else:
                label = f'{val:.1e}'
            ax.annotate(label, (zm, yvals_plot[j]),
                       textcoords="offset points", xytext=(8, 5),
                       fontsize=9, color=colors[i])

    ax.set_xlabel('DECam z magnitude limit', fontsize=13)
    ax.set_ylabel('Number density (stars / deg²)', fontsize=13)
    ax.set_title(f'{name_pretty} (M$_V$={mv_val:.1f}): Density (5-20 rh)', fontsize=13)
    ax.legend(fontsize=11, title='Debris fraction f', title_fontsize=10,
              loc='lower right')
    ax.set_xticks(Zmag_values)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(18, 24)
    ax.text(0.05, 0.95, f"Area (5-20 rh) = {area_deg2:.2f} deg²",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # ---- Panel 3: Synthetic CMD for f=0.01 ----
    # Uses the SAME reference cluster as panels 1&2, subsampled to match f=0.01
    ax = axes[2]
    f_cmd = 0.01
    M_total = 10 ** float(satellite_row['mass_stellar'])
    debris_mass_cmd = f_cmd * M_total / 2.0
    scale_cmd = debris_mass_cmd / prep['mass_surviving_total']

    # Get scaled star counts per band (consistent with panel 1)
    valid = prep['valid']
    ref_color = prep['app_r'][valid] - prep['app_z'][valid]
    ref_mag_z = prep['app_z'][valid]

    n_z23 = results[1, 2]  # f=0.01, Zmag=23 from panel 1
    n_z21 = results[1, 1]  # f=0.01, Zmag=21
    n_z19 = results[1, 0]  # f=0.01, Zmag=19
    n_band1 = int(round(n_z23 - n_z21))   # 21 < z < 23
    n_band2 = int(round(n_z21 - n_z19))   # 19 < z < 21
    n_bright = int(round(n_z19))           # z < 19

    # Subsample reference cluster: each star included with probability = scale_cmd
    rng = np.random.default_rng(42)  # reproducible
    keep = rng.random(len(ref_mag_z)) < scale_cmd
    sub_color = ref_color[keep]
    sub_mag_z = ref_mag_z[keep]

    # Plot subsampled stars by brightness band
    faint = sub_mag_z >= 23
    band1 = (sub_mag_z < 23) & (sub_mag_z >= 21)
    band2 = (sub_mag_z < 21) & (sub_mag_z >= 19)
    bright = sub_mag_z < 19

    ax.scatter(sub_color[faint], sub_mag_z[faint], s=2, c='lightgrey',
               alpha=0.3, rasterized=True)
    ax.scatter(sub_color[band1], sub_mag_z[band1], s=8, c='#2196F3',
               alpha=0.7,
               label=f'21 < z < 23 ({n_band1} stars)', zorder=3)
    ax.scatter(sub_color[band2], sub_mag_z[band2], s=15, c='#FF9800',
               alpha=0.8,
               label=f'19 < z < 21 ({n_band2} stars)', zorder=4)
    ax.scatter(sub_color[bright], sub_mag_z[bright], s=30, c='#E91E63',
               alpha=0.9,
               label=f'z < 19 ({n_bright} stars)', zorder=5)

    # Overlay isochrone line with adaptive mass grid for better RGB resolution
    max_mass = ii_global.getMaxMass(LOGAGE, prep['feh'])
    ms_grid = np.linspace(0.1, max_mass * 0.85, 200)
    to_grid = np.linspace(max_mass * 0.85, max_mass * 0.9999, 500)
    mass_grid = np.concatenate([ms_grid, to_grid])
    iso_res = ii_global(mass_grid, np.full(len(mass_grid), LOGAGE),
                        np.full(len(mass_grid), prep['feh']))
    iso_r = np.array(iso_res['DECam_r']) + prep['dm']
    iso_z = np.array(iso_res['DECam_z']) + prep['dm']
    iso_ok = np.isfinite(iso_r) & np.isfinite(iso_z)
    ax.plot(iso_r[iso_ok] - iso_z[iso_ok], iso_z[iso_ok],
            'k-', linewidth=1.5, alpha=0.5, label='12 Gyr isochrone', zorder=2)

    # Zmag limit lines
    for zm, zc in zip([19, 21, 23], ['#E91E63', '#FF9800', '#2196F3']):
        ax.axhline(zm, color=zc, linestyle='--', alpha=0.6, linewidth=1)

    ax.set_xlabel('DECam (r - z)', fontsize=13)
    ax.set_ylabel('DECam z (apparent mag)', fontsize=13)
    mv_str = f'{satellite_row["M_V"]:.1f}'
    ax.set_title(f'{name_pretty} (M_V={mv_str}): CMD (f={f_cmd})', fontsize=13)
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc='lower right', markerscale=2)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits based on distance modulus
    ax.set_ylim(prep['dm'] + 12, min(prep['dm'] - 3, 15))

    # Annotate parameters including M_V
    param_text_cmd = (f"M_V = {satellite_row['M_V']:.1f}\n" + param_text)
    ax.text(0.05, 0.05, param_text_cmd, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    outpath = os.path.join(save_dir, f'{key}_debris.png')
    plt.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close()

    return results, density, area_deg2


# ===========================================================
# All-satellites summary plot (Task 2)
# ===========================================================

def run_satellites_summary(table, host_label, f=0.01, Zmag=23, n_real=10):
    """
    Task 2: Run analysis for satellites in a given table at fixed f and Zmag.
    Averages over n_real reference cluster realizations.
    """
    n_sat = len(table)
    print(f"\n{'='*60}")
    print(f"{host_label} SATELLITES SUMMARY: f={f}, Zmag={Zmag}, {n_real} realizations")
    print(f"  {n_sat} satellites")
    print(f"{'='*60}")

    names, N_remnants, areas, densities = [], [], [], []
    distances, M_Vs, log_Mstars, fehs = [], [], [], []

    for i, row in enumerate(table):
        key = str(row['key'])
        print(f"  [{i+1:3d}/{n_sat}] {key:30s}", end="", flush=True)

        Ns = []
        feh_used = None
        for _ in range(n_real):
            p = prepare_satellite(row)
            N_i, _ = count_debris_stars(row, p, f, Zmag)
            Ns.append(N_i)
            if feh_used is None:
                feh_used = p['feh']
        N = np.mean(Ns)
        area = compute_area_deg2(row)
        dens = N / area if area > 0 else 0.0

        names.append(key)
        N_remnants.append(N)
        areas.append(area)
        densities.append(dens)
        distances.append(float(row['distance']))
        M_Vs.append(float(row['M_V']))
        log_Mstars.append(float(row['mass_stellar']))
        fehs.append(feh_used)

        print(f"  N={N:10.1f}  area={area:8.2f} deg²  "
              f"density={dens:8.3f} /deg²")

    # Convert to arrays
    names = np.array(names)
    N_remnants = np.array(N_remnants)
    areas = np.array(areas)
    densities = np.array(densities)
    distances = np.array(distances)
    M_Vs = np.array(M_Vs)
    log_Mstars = np.array(log_Mstars)

    order = np.argsort(N_remnants)[::-1]

    # ---- Summary table ----
    print(f"\n{'='*105}")
    print(f"{'Name':30s} {'M_V':>6s} {'log M*':>7s} {'dist':>7s} "
          f"{'N_rem':>10s} {'area':>8s} {'density':>10s}")
    print(f"{'':30s} {'':>6s} {'':>7s} {'(kpc)':>7s} "
          f"{'':>10s} {'(deg2)':>8s} {'(/deg2)':>10s}")
    print("-" * 105)
    for j in order:
        print(f"{names[j]:30s} {M_Vs[j]:6.1f} {log_Mstars[j]:7.2f} "
              f"{distances[j]:7.1f} {N_remnants[j]:10.1f} "
              f"{areas[j]:8.2f} {densities[j]:10.3f}")

    # ---- PLOT (2x2) ----
    fig = plt.figure(figsize=(22, 16))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # Panel 1: Bar chart of N_remnant
    y_pos = np.arange(len(names))
    bar_colors = plt.cm.viridis(
        (distances[order] - distances.min()) /
        max(distances.max() - distances.min(), 1))
    ax1.barh(y_pos, np.maximum(N_remnants[order], 0.005), color=bar_colors,
             edgecolor='none', height=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([n.replace('_', ' ') for n in names[order]],
                         fontsize=6 if n_sat > 50 else 7)
    ax1.set_xlabel(f'N_remnant (f={f}, z<{Zmag})', fontsize=12)
    ax1.set_title(f'{host_label}: Detectable Tidal Debris Stars', fontsize=13)
    ax1.set_xscale('log')
    ax1.set_xlim(left=0.005)
    ax1.invert_yaxis()
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.axvline(1, color='red', linestyle=':', alpha=0.7, label='N=1')
    ax1.axvline(10, color='orange', linestyle=':', alpha=0.7, label='N=10')
    ax1.axvline(100, color='green', linestyle=':', alpha=0.5, label='N=100')
    ax1.legend(fontsize=9, loc='lower right')
    sm = plt.cm.ScalarMappable(cmap='viridis',
        norm=plt.Normalize(distances.min(), distances.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, pad=0.01, aspect=40)
    cbar.set_label('Distance (kpc)', fontsize=10)

    # Panel 2: N_remnant vs distance
    sc = ax2.scatter(distances, N_remnants, c=log_Mstars, cmap='plasma',
                     s=60, edgecolors='k', linewidth=0.5, zorder=3)
    # Label notable satellites
    for j in range(len(names)):
        if N_remnants[j] > max(np.median(N_remnants) * 5, 1) or j < 5:
            ax2.annotate(names[j].replace('_', ' '),
                        (distances[j], max(N_remnants[j], 0.02)),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.8)
    ax2.set_xlabel('Distance (kpc)', fontsize=12)
    ax2.set_ylabel(f'N_remnant (f={f}, z<{Zmag})', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title(f'{host_label}: N_remnant vs Distance', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(10, color='orange', linestyle=':', alpha=0.5)
    cbar2 = plt.colorbar(sc, ax=ax2, pad=0.01)
    cbar2.set_label('log(M*/M$_\\odot$)', fontsize=10)

    # Panel 3: N_remnant vs M_V
    sc3 = ax3.scatter(M_Vs, N_remnants, c=distances, cmap='viridis',
                      s=60, edgecolors='k', linewidth=0.5, zorder=3)
    for j in range(len(names)):
        if N_remnants[j] > max(np.median(N_remnants) * 5, 1) or j < 5:
            ax3.annotate(names[j].replace('_', ' '),
                        (M_Vs[j], max(N_remnants[j], 0.02)),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.8)
    ax3.set_xlabel('M_V (absolute magnitude)', fontsize=12)
    ax3.set_ylabel(f'N_remnant (f={f}, z<{Zmag})', fontsize=12)
    ax3.set_yscale('log')
    ax3.set_title(f'{host_label}: N_remnant vs Luminosity', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(1, color='red', linestyle=':', alpha=0.5)
    ax3.axhline(10, color='orange', linestyle=':', alpha=0.5)
    ax3.invert_xaxis()
    cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.01)
    cbar3.set_label('Distance (kpc)', fontsize=10)

    # Panel 4: Number density vs M_V
    sc4 = ax4.scatter(M_Vs, densities, c=distances, cmap='viridis',
                      s=np.clip(areas * 5 + 10, 10, 300),
                      edgecolors='k', linewidth=0.5, zorder=3)
    for j in range(len(names)):
        if densities[j] > max(np.median(densities) * 5, 0.01):
            ax4.annotate(names[j].replace('_', ' '),
                        (M_Vs[j], max(densities[j], 1e-4)),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, alpha=0.8)
    ax4.set_xlabel('M_V (absolute magnitude)', fontsize=12)
    ax4.set_ylabel('Number density (stars / deg²)', fontsize=12)
    ax4.set_yscale('log')
    ax4.set_title(f'{host_label}: Debris Star Density (5-20 rh, f={f})', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.invert_xaxis()
    cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.01)
    cbar4.set_label('Distance (kpc)', fontsize=10)
    for area_val in [1, 10, 100]:
        ax4.scatter([], [], s=area_val * 5 + 10, c='grey', edgecolors='k',
                    linewidth=0.5, label=f'{area_val} deg²')
    ax4.legend(title='Area (5-20 rh)', fontsize=8, title_fontsize=9,
               loc='upper left')

    fig.suptitle(f'{host_label} Satellite Tidal Debris Forecast  |  f = {f}, '
                 f'Zmag = {Zmag}, Age = 12 Gyr',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    tag = host_label.lower().replace(' ', '_')
    outpath = os.path.join(OUTDIR, f'{tag}_satellites_debris.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSummary plot saved to {outpath}")


# ===========================================================
# Main
# ===========================================================

if __name__ == '__main__':
    # Task 1: Individual plots for MW satellites
    n_mw = len(mw)
    print(f"Generating MW satellite plots ({n_mw} satellites)...")
    for i, row in enumerate(mw):
        key = str(row['key'])
        print(f"[MW {i+1:2d}/{n_mw}] {key}...", end="", flush=True)
        plot_satellite(row, save_dir=PLOTDIR_MW)
        print(" done")

    # Task 1: Individual plots for M31 satellites
    n_m31 = len(m31)
    print(f"\nGenerating M31 satellite plots ({n_m31} satellites)...")
    for i, row in enumerate(m31):
        key = str(row['key'])
        print(f"[M31 {i+1:2d}/{n_m31}] {key}...", end="", flush=True)
        plot_satellite(row, save_dir=PLOTDIR_M31)
        print(" done")

    # Task 2: Summary plots (separate for MW and M31)
    run_satellites_summary(mw, 'MW', f=0.01, Zmag=23)
    run_satellites_summary(m31, 'M31', f=0.01, Zmag=23)
