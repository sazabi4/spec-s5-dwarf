#!/usr/bin/env python3
"""Cumulative distribution of debris fraction f_5_20 from simulations."""

import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5'

sats = np.load(f'{OUTDIR}/sat_arrs_ting.npy')
f_star_bound = sats[:, 0]
m_star_bound = sats[:, 2]
r_sub_kpc = sats[:, 13]
m_star_lt1 = sats[:, 8]
m_star_lt5 = sats[:, 10]
m_star_lt20 = sats[:, 12]
f_5_20 = (m_star_lt20 - m_star_lt5) / m_star_lt1

# Cut: r_sub < 200 kpc
mask = r_sub_kpc < 200
print(f'Total: {len(sats)}, r_sub < 200 kpc: {mask.sum()}')
print(f'f_5_20 (r_sub<200): median={np.median(f_5_20[mask]):.4f}, '
      f'mean={np.mean(f_5_20[mask]):.4f}')

f_cut = f_5_20[mask]
f_all = f_5_20

fig, ax = plt.subplots(figsize=(8, 5))

# Cumulative fraction (fraction of satellites with f_5_20 > x)
f_sorted_all = np.sort(f_all)
f_sorted_cut = np.sort(f_cut)
cdf_all = 1.0 - np.arange(1, len(f_sorted_all)+1) / len(f_sorted_all)
cdf_cut = 1.0 - np.arange(1, len(f_sorted_cut)+1) / len(f_sorted_cut)

ax.step(f_sorted_all, cdf_all, where='post', color='grey', lw=1.5, alpha=0.6,
        label=f'All ({len(f_all)})')
ax.step(f_sorted_cut, cdf_cut, where='post', color='C0', lw=2,
        label=r'$r_{\rm sub} < 200$ kpc (' + str(len(f_cut)) + ')')

# Reference f lines
for fv, fc, ls in zip([0.001, 0.01, 0.1], ['C2', 'C3', 'C4'], [':', '--', '-.']):
    ax.axvline(fv, color=fc, ls=ls, alpha=0.7, lw=1, label=f'f = {fv}')

ax.set_xscale('log')
ax.set_xlabel(r'$f_{5-20} = M_\star(5$-$20\,r_h) / M_\star(<r_h)$', fontsize=13)
ax.set_ylabel('Cumulative fraction (> f)', fontsize=13)
ax.set_xlim(1e-5, 5)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.set_title(r'Cumulative distribution of debris fraction $f_{5-20}$', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/sim_f520_cumulative.png', dpi=150, bbox_inches='tight')
print('Saved sim_f520_cumulative.png')
