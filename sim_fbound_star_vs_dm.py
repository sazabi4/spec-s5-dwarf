#!/usr/bin/env python3
"""Plot stellar vs dark matter bound fraction from simulations."""

import numpy as np
import matplotlib.pyplot as plt

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5-dwarf'

sats = np.load(f'{OUTDIR}/sat_arrs_ting.npy')
f_star = sats[:, 0]
f_dm = sats[:, 1]

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(f_dm, f_star, s=12, alpha=0.5, c=np.log10(sats[:, 2]),
                cmap='viridis', rasterized=True)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='1:1')
ax.set_xlabel(r'$f_{\mathrm{bound,DM}}$', fontsize=13)
ax.set_ylabel(r'$f_{\mathrm{bound,\star}}$', fontsize=13)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-0.02, 1.05)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label(r'$\log_{10}(M_{\star,\mathrm{bound}} / M_\odot)$', fontsize=12)
ax.legend(fontsize=11)
ax.set_title('Stellar vs. dark matter bound fraction', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/sim_fbound_star_vs_dm.png', dpi=150, bbox_inches='tight')
print('Saved sim_fbound_star_vs_dm.png')

# Stats
print(f'f_star > f_dm: {np.sum(f_star > f_dm)}/{len(f_star)} '
      f'({np.sum(f_star > f_dm)/len(f_star):.1%})')
print(f'Median f_star: {np.median(f_star):.3f}, Median f_dm: {np.median(f_dm):.3f}')
print(f'Mean f_star - f_dm: {np.mean(f_star - f_dm):.3f}')
