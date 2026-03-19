#!/usr/bin/env python3
"""
Simulation debris fractions (5-20 rh) with MW satellite tick marks on x-axes.
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
r_sub_kpc = sats[:, 13]

m_star_lt1 = sats[:, 8]
m_star_lt5 = sats[:, 10]
m_star_lt20 = sats[:, 12]

f_5_20 = (m_star_lt20 - m_star_lt5) / m_star_lt1

log_m_star = np.log10(m_star_bound)
m_star_infall = m_star_bound / f_star_bound
log_m_star_infall = np.log10(m_star_infall)
M_V_sim = 4.83 - 2.5 * np.log10(m_star_bound / 1.6)

# ---- Load MW and M31 data ----
mw = Table.read(os.path.join(OUTDIR, 'mw_satellites.csv'))
m31 = Table.read(os.path.join(OUTDIR, 'm31_satellites.csv'))

mw_log_mstar = mw['mass_stellar']
mw_MV = mw['M_V']
mw_rhalf = mw['rhalf_physical'] / 1000.0  # pc -> kpc
mw_dist = mw['distance']  # kpc

m31_log_mstar = m31['mass_stellar']
m31_MV = m31['M_V']
m31_rhalf = m31['rhalf_physical'] / 1000.0

# M31 satellite distances from M31 center (not from us)
# M31 center: RA=10.6847929, Dec=41.2690650, d=785 kpc
m31_ra_center = 10.6847929   # deg
m31_dec_center = 41.2690650  # deg
m31_d_center = 785.0         # kpc

def radec_to_xyz(ra_deg, dec_deg, d_kpc):
    """Convert (RA, Dec, distance) to Cartesian (x, y, z)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = d_kpc * np.cos(dec) * np.cos(ra)
    y = d_kpc * np.cos(dec) * np.sin(ra)
    z = d_kpc * np.sin(dec)
    return x, y, z

# M31 center in Cartesian
x0, y0, z0 = radec_to_xyz(m31_ra_center, m31_dec_center, m31_d_center)

# Each M31 satellite in Cartesian, then 3D distance from M31 center
x_sat, y_sat, z_sat = radec_to_xyz(
    np.array(m31['ra']), np.array(m31['dec']), np.array(m31['distance'])
)
m31_dist_from_host = np.sqrt((x_sat - x0)**2 + (y_sat - y0)**2 + (z_sat - z0)**2)

print(f"M31 satellite distance from M31 center: "
      f"min={np.min(m31_dist_from_host):.1f}, max={np.max(m31_dist_from_host):.1f}, "
      f"median={np.median(m31_dist_from_host):.1f} kpc")

# ---- Figure: 2x3 with MW/M31 tick marks on x-axes ----
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

cmap = plt.cm.coolwarm_r
scatter_kw = dict(s=12, alpha=0.6, c=f_star_bound, cmap=cmap, vmin=0, vmax=1, rasterized=True)

f_ref = [0.001, 0.01, 0.1, 1.0]
f_colors = ['C2', 'C3', 'C4', 'C5']

def add_flines(ax):
    for fv, fc in zip(f_ref, f_colors):
        ax.axhline(fv, color=fc, ls='--', alpha=0.5, lw=0.8)

def add_ticks(ax, mw_vals, m31_vals, tick_length=0.08):
    """Add tick marks at top of panel for MW (blue) and M31 (orange) satellites."""
    ylo, yhi = ax.get_ylim()
    for v in mw_vals:
        ax.plot([v, v], [yhi * 10**(-tick_length*5), yhi],
                color='C0', alpha=0.5, lw=0.8, clip_on=True)
    for v in m31_vals:
        ax.plot([v, v], [yhi * 10**(-tick_length*3), yhi * 10**(-tick_length*1.5)],
                color='C1', alpha=0.5, lw=0.8, clip_on=True)

# Panel 1: f_5_20 vs f_star_bound (no MW/M31 — they don't have f_bound)
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
ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{bound}} / M_\odot)$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
add_flines(ax)
add_ticks(ax, mw_log_mstar, m31_log_mstar)
# Legend
ax.plot([], [], color='C0', lw=2, alpha=0.7, label='MW satellites')
ax.plot([], [], color='C1', lw=2, alpha=0.7, label='M31 satellites')
ax.legend(fontsize=8, loc='lower left')

# Panel 3: f_5_20 vs log(M_star_infall) — no MW/M31 (infall mass unknown)
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
ax.set_xlabel(r'$M_V$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.invert_xaxis()
add_flines(ax)
add_ticks(ax, mw_MV, m31_MV)
ax.plot([], [], color='C0', lw=2, alpha=0.7, label='MW')
ax.plot([], [], color='C1', lw=2, alpha=0.7, label='M31')
ax.legend(fontsize=8, loc='lower right')

# Panel 5: f_5_20 vs r_half
ax = axes[1, 1]
ax.scatter(r_half_kpc, f_5_20, **scatter_kw)
ax.set_xlabel(r'$r_{1/2}$ [kpc]', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-5, 10)
ax.set_xlim(0.01, 5.0)
add_flines(ax)
add_ticks(ax, mw_rhalf, m31_rhalf)

# Panel 6: f_5_20 vs distance from host
ax = axes[1, 2]
ax.scatter(r_sub_kpc, f_5_20, **scatter_kw)
ax.set_xlabel(r'$r_{\rm sub}$ [kpc]', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
add_flines(ax)
add_ticks(ax, mw_dist, m31_dist_from_host)

# Colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(r'$f_{\star,\mathrm{bound}}$', fontsize=12)

fig.suptitle(r'Simulation debris fraction: $M_\star(5$-$20\,r_h) / M_\star(<r_h)$'
             '  |  tick marks = MW (blue) / M31 (orange)',
             fontsize=13, y=0.98)
fig.savefig(os.path.join(OUTDIR, 'sim_debris_fraction_ticks.png'), dpi=150, bbox_inches='tight')
print("Saved sim_debris_fraction_ticks.png")
