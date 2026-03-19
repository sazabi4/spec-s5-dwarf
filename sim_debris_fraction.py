#!/usr/bin/env python3
"""
Compare simulation debris fractions (5-20 rh) with MW and M31 satellite properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import os

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5'

# ---- Load simulation data ----
sats = np.load(os.path.join(OUTDIR, 'sat_arrs_ting.npy'))

f_star_bound = sats[:, 0]
m_star_bound = sats[:, 2]
r_half_kpc = sats[:, 3]
r_sub_kpc = sats[:, 13]  # distance from host center

# Cumulative stellar mass within radial bins (in units of r_half)
m_star_lt1 = sats[:, 8]    # SM_bins[0]: <1 rh
m_star_lt5 = sats[:, 10]   # SM_bins[2]: <5 rh
m_star_lt10 = sats[:, 11]  # SM_bins[3]: <10 rh
m_star_lt20 = sats[:, 12]  # SM_bins[4]: <20 rh

# Debris fraction: mass in 5-20 rh annulus relative to mass within 1 rh
f_5_20 = (m_star_lt20 - m_star_lt5) / m_star_lt1

log_m_star = np.log10(m_star_bound)

# Infall stellar mass = bound mass / f_star_bound
m_star_infall = m_star_bound / f_star_bound
log_m_star_infall = np.log10(m_star_infall)

# Convert m_star_bound to M_V using M/L=1.6 and M_sun_V=4.83
M_V_sim = 4.83 - 2.5 * np.log10(m_star_bound / 1.6)

# ---- Load MW and M31 data ----
mw = Table.read(os.path.join(OUTDIR, 'mw_satellites.csv'))
m31 = Table.read(os.path.join(OUTDIR, 'm31_satellites.csv'))

mw_log_mstar = mw['mass_stellar']
mw_MV = mw['M_V']
mw_rhalf = mw['rhalf_physical'] / 1000.0  # pc -> kpc
mw_dist = mw['distance']  # already in kpc

m31_log_mstar = m31['mass_stellar']
m31_MV = m31['M_V']
m31_rhalf = m31['rhalf_physical'] / 1000.0  # pc -> kpc
m31_dist = m31['distance']  # already in kpc

# Check distance units - LVDB distances should be in kpc
print(f"MW distance range: {np.min(mw['distance']):.1f} - {np.max(mw['distance']):.1f}")
print(f"M31 distance range: {np.min(m31['distance']):.1f} - {np.max(m31['distance']):.1f}")
print(f"MW rhalf_physical range: {np.min(mw['rhalf_physical']):.1f} - {np.max(mw['rhalf_physical']):.1f}")

# ---- Figure 1: f_5_20 vs galaxy properties (sim) with MW/M31 overlaid ----
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# Common plot settings - color-coded by f_star_bound
cmap = plt.cm.coolwarm_r  # blue=intact (f_bound~1), red=stripped (f_bound~0)
scatter_kw = dict(s=12, alpha=0.6, c=f_star_bound, cmap=cmap, vmin=0, vmax=1, rasterized=True)
mw_kw = dict(s=40, alpha=0.8, marker='o', edgecolors='C0', facecolors='none',
             linewidths=1.2, label='MW satellites')
m31_kw = dict(s=40, alpha=0.8, marker='s', edgecolors='C1', facecolors='none',
              linewidths=1.2, label='M31 satellites')

# Reference f lines
f_ref = [0.001, 0.01, 0.1, 1.0]
f_colors = ['C2', 'C3', 'C4', 'C5']

def add_flines(ax):
    for fv, fc in zip(f_ref, f_colors):
        ax.axhline(fv, color=fc, ls='--', alpha=0.5, lw=0.8)

def add_mw_m31_vlines(ax, col):
    for row in mw:
        ax.axvline(row[col], color='C0', alpha=0.08, lw=0.5)
    for row in m31:
        ax.axvline(row[col], color='C1', alpha=0.08, lw=0.5)
    ax.scatter([], [], **mw_kw)
    ax.scatter([], [], **m31_kw)

# Panel 1: f_5_20 vs f_star_bound
ax = axes[0, 0]
sc = ax.scatter(f_star_bound, f_5_20, **scatter_kw)
ax.set_xlabel(r'$f_{\star,\mathrm{bound}}$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.set_xlim(0, 1.05)
add_flines(ax)

# Panel 2: f_5_20 vs log(M_star_bound)
ax = axes[0, 1]
ax.scatter(log_m_star, f_5_20, **scatter_kw)
add_mw_m31_vlines(ax, 'mass_stellar')
ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{bound}} / M_\odot)$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.legend(fontsize=8, loc='upper left')
add_flines(ax)

# Panel 3: f_5_20 vs log(M_star_infall)
ax = axes[0, 2]
ax.scatter(log_m_star_infall, f_5_20, **scatter_kw)
ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{infall}} / M_\odot)$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
add_flines(ax)

# Panel 4: f_5_20 vs M_V
ax = axes[1, 0]
ax.scatter(M_V_sim, f_5_20, **scatter_kw)
add_mw_m31_vlines(ax, 'M_V')
ax.set_xlabel(r'$M_V$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.invert_xaxis()
ax.legend(fontsize=8, loc='upper left')
add_flines(ax)

# Panel 5: f_5_20 vs r_half
ax = axes[1, 1]
ax.scatter(r_half_kpc, f_5_20, **scatter_kw)
ax.set_xlabel(r'$r_{1/2}$ [kpc]', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.set_xlim(0, 1.6)
add_flines(ax)

# Panel 6: f_5_20 vs distance from host
ax = axes[1, 2]
ax.scatter(r_sub_kpc, f_5_20, **scatter_kw)
ax.set_xlabel(r'$r_{\rm sub}$ [kpc]', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
add_flines(ax)

# Colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(r'$f_{\star,\mathrm{bound}}$', fontsize=12)

fig.suptitle(r'Simulation debris fraction: $M_\star(5$-$20\,r_h) / M_\star(<r_h)$',
             fontsize=14, y=0.98)
fig.savefig(os.path.join(OUTDIR, 'sim_debris_fraction.png'), dpi=150, bbox_inches='tight')
print(f"Saved sim_debris_fraction.png")

# ---- Figure 2: f_5_20 vs M_V with MW/M31 comparison ----
fig2, ax2 = plt.subplots(figsize=(10, 6))
sc2 = ax2.scatter(M_V_sim, f_5_20, s=15, alpha=0.6, c=f_star_bound,
                  cmap=cmap, vmin=0, vmax=1, rasterized=True)
ax2.scatter(mw_MV, np.full(len(mw), np.nan), **mw_kw)   # just for legend
ax2.scatter(m31_MV, np.full(len(m31), np.nan), **m31_kw)

# Show MW/M31 M_V as vertical lines
for row in mw:
    ax2.axvline(row['M_V'], color='C0', alpha=0.15, lw=0.5)
for row in m31:
    ax2.axvline(row['M_V'], color='C1', alpha=0.15, lw=0.5)

for fv, fc in zip(f_ref, f_colors):
    ax2.axhline(fv, color=fc, ls='--', alpha=0.6, lw=1,
                label=f'f = {fv}')

ax2.set_xlabel(r'$M_V$', fontsize=13)
ax2.set_ylabel(r'$f_{5-20} = M_\star(5$-$20\,r_h) / M_\star(<r_h)$', fontsize=12)
ax2.set_yscale('log')
ax2.set_ylim(1e-5, 10)
ax2.invert_xaxis()
ax2.legend(fontsize=10, loc='upper left', ncol=2)
ax2.set_title(r'Debris fraction from simulations vs. $M_V$', fontsize=13)
cbar2 = fig2.colorbar(sc2, ax=ax2, pad=0.02)
cbar2.set_label(r'$f_{\star,\mathrm{bound}}$', fontsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTDIR, 'sim_debris_fraction_MV.png'), dpi=150, bbox_inches='tight')
print(f"Saved sim_debris_fraction_MV.png")

# ---- Print some statistics ----
print("\n" + "=" * 60)
print("Debris fraction statistics by f_star_bound bins")
print("=" * 60)
bins = [(0.95, 1.01, 'Intact (f_bound > 0.95)'),
        (0.8, 0.95, 'Mildly stripped (0.8 < f_bound < 0.95)'),
        (0.5, 0.8, 'Moderately stripped (0.5 < f_bound < 0.8)'),
        (0.0, 0.5, 'Heavily stripped (f_bound < 0.5)')]

for lo, hi, label in bins:
    mask = (f_star_bound >= lo) & (f_star_bound < hi)
    n = np.sum(mask)
    if n > 0:
        f_vals = f_5_20[mask]
        print(f"\n{label}: N={n}")
        print(f"  f_5_20: median={np.median(f_vals):.4f}, "
              f"mean={np.mean(f_vals):.4f}, "
              f"16-84%=[{np.percentile(f_vals, 16):.5f}, {np.percentile(f_vals, 84):.4f}]")

print("\n" + "=" * 60)
print("Debris fraction statistics by log(M_star) bins")
print("=" * 60)
mbins = [(2, 4, 'UFDs (log M* = 2-4)'),
         (4, 6, 'Classical dwarfs (log M* = 4-6)'),
         (6, 8, 'Bright dwarfs (log M* = 6-8)')]

for lo, hi, label in mbins:
    mask = (log_m_star >= lo) & (log_m_star < hi)
    n = np.sum(mask)
    if n > 0:
        f_vals = f_5_20[mask]
        print(f"\n{label}: N={n}")
        print(f"  f_5_20: median={np.median(f_vals):.4f}, "
              f"mean={np.mean(f_vals):.4f}, "
              f"16-84%=[{np.percentile(f_vals, 16):.5f}, {np.percentile(f_vals, 84):.4f}]")
