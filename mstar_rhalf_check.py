#!/usr/bin/env python3
"""Sanity check: stellar mass vs half-light radius for MW and M31 satellites."""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5'
mw = Table.read(f'{OUTDIR}/mw_satellites.csv')
m31 = Table.read(f'{OUTDIR}/m31_satellites.csv')

# Stellar mass (log10) and rhalf (pc) from LVDB
mw_logmstar = mw['mass_stellar']
mw_rhalf = mw['rhalf_physical']  # pc
m31_logmstar = m31['mass_stellar']
m31_rhalf = m31['rhalf_physical']  # pc

# Relation: log(r_half/pc) = 0.31 * log10(M_star) + 0.4875
def compute_rhalf_from_mstar(mstar):
    """
    Compute the half-light radius (rhalf) from stellar mass
    using the Mv-size relation with M/L = 1.6.

    Parameters
    ----------
    mstar : float or array-like
        Stellar mass in solar masses.

    Returns
    -------
    rhalf : float or ndarray
        Estimated half-light radius in parsecs.
    """
    log_rhalf = 0.31 * np.log10(mstar) + 0.4875
    return 10**log_rhalf

mstar_grid = np.logspace(2, 9, 200)
rhalf_fit = compute_rhalf_from_mstar(mstar_grid)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(10**mw_logmstar, mw_rhalf, s=30, alpha=0.7, c='C0', label='MW satellites (LVDB)')
ax.scatter(10**m31_logmstar, m31_rhalf, s=30, alpha=0.7, c='C1', marker='s', label='M31 satellites (LVDB)')
ax.plot(mstar_grid, rhalf_fit, 'k-', lw=2, label=r'Fit: $\log r_h = 0.31 \log M_\star + 0.49$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$M_\star$ [$M_\odot$]', fontsize=13)
ax.set_ylabel(r'$r_{1/2}$ [pc]', fontsize=13)
ax.set_xlim(1e2, 1e9)
ax.set_ylim(5, 5000)
ax.legend(fontsize=10)
ax.set_title('Stellar mass vs. half-light radius', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/mstar_rhalf_check.png', dpi=150, bbox_inches='tight')
print('Saved mstar_rhalf_check.png')
