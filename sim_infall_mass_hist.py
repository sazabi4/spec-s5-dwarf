#!/usr/bin/env python3
"""Histogram of infall stellar mass from simulations."""

import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5'

sats = np.load(f'{OUTDIR}/sat_arrs_ting.npy')
f_star_bound = sats[:, 0]
m_star_bound = sats[:, 2]
m_star_infall = m_star_bound / f_star_bound
log_m_infall = np.log10(m_star_infall)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(log_m_infall, bins=40, edgecolor='k', alpha=0.7, color='C0')
ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{infall}} / M_\odot)$', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Infall stellar mass distribution (922 simulated satellites)', fontsize=13)

print(f'log M_infall: min={log_m_infall.min():.2f}, max={log_m_infall.max():.2f}')
print(f'  median={np.median(log_m_infall):.2f}, mean={np.mean(log_m_infall):.2f}')
print(f'  16-84%: [{np.percentile(log_m_infall, 16):.2f}, {np.percentile(log_m_infall, 84):.2f}]')

fig.tight_layout()
fig.savefig(f'{OUTDIR}/sim_infall_mass_hist.png', dpi=150, bbox_inches='tight')
print('Saved sim_infall_mass_hist.png')
