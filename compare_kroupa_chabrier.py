#!/usr/bin/env python3
"""
Direct comparison: Kroupa vs Chabrier pipeline for Draco.
Also reproduce user's exact approach for apples-to-apples comparison.
"""
import numpy as np
import imf
import minimint
import warnings
warnings.filterwarnings('ignore')

# Draco parameters
mass_stellar_log = 5.78
distance_modulus = 19.557
feh = -2.0
logage = np.log10(12e9)
f = 0.01

M_total = 10**mass_stellar_log
Nstar_half = M_total / 2.0
debris_mass = f * Nstar_half  # 3013 Msun surviving

mi = minimint.Interpolator(['DECam_r', 'DECam_z'])

print(f"Draco: debris_mass = {debris_mass:.0f} Msun (surviving)")
print()

# ============================================================
# 1) Pipeline with Kroupa (current)
# ============================================================
print("=" * 70)
print("PIPELINE WITH KROUPA (make_cluster, current approach)")
print("=" * 70)
results_k = []
for trial in range(20):
    masses_all = imf.make_cluster(50000, massfunc='kroupa')
    mask = masses_all <= 1.0
    masses = masses_all[mask]
    mass_surv = masses.sum()

    res = mi(masses, np.full(len(masses), logage), np.full(len(masses), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)

    scale = debris_mass / mass_surv
    N_ref_z23 = np.sum(valid & (app_z < 23))
    N_remnant = N_ref_z23 * scale

    results_k.append({
        'mass_surv': mass_surv, 'n_surv': len(masses),
        'N_ref_z23': N_ref_z23, 'scale': scale, 'N_remnant': N_remnant
    })

mean_k = np.mean([r['N_remnant'] for r in results_k])
print(f"  Mean surviving mass in ref: {np.mean([r['mass_surv'] for r in results_k]):.0f} Msun")
print(f"  Mean scale: {np.mean([r['scale'] for r in results_k]):.5f}")
print(f"  N_remnant(z<23): {mean_k:.1f} ± {np.std([r['N_remnant'] for r in results_k]):.1f}")
print()

# ============================================================
# 2) Pipeline with Chabrier (using rvs to generate ref cluster)
# ============================================================
print("=" * 70)
print("PIPELINE WITH CHABRIER (rvs for reference cluster)")
print("=" * 70)

# First, figure out how many stars to draw for a large ref cluster
# We want a big enough reference. Draw 200k stars.
N_ref_draw = 200000
results_c = []
for trial in range(20):
    masses_all = imf.chabrier2005.distr.rvs(N_ref_draw)
    mask = masses_all <= 1.0
    masses = masses_all[mask]
    mass_surv = masses.sum()

    res = mi(masses, np.full(len(masses), logage), np.full(len(masses), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)

    scale = debris_mass / mass_surv
    N_ref_z23 = np.sum(valid & (app_z < 23))
    N_remnant = N_ref_z23 * scale

    results_c.append({
        'mass_all': masses_all.sum(), 'mass_surv': mass_surv,
        'n_all': len(masses_all), 'n_surv': len(masses),
        'N_ref_z23': N_ref_z23, 'scale': scale, 'N_remnant': N_remnant
    })

mean_c = np.mean([r['N_remnant'] for r in results_c])
print(f"  Ref cluster: {N_ref_draw} stars drawn")
print(f"  Mean total mass: {np.mean([r['mass_all'] for r in results_c]):.0f} Msun")
print(f"  Mean surviving mass: {np.mean([r['mass_surv'] for r in results_c]):.0f} Msun")
print(f"  Mean scale: {np.mean([r['scale'] for r in results_c]):.6f}")
print(f"  N_remnant(z<23): {mean_c:.1f} ± {np.std([r['N_remnant'] for r in results_c]):.1f}")
print()

# ============================================================
# 3) User's exact approach: rvs(70000) and ratio scaling
# ============================================================
print("=" * 70)
print("USER'S APPROACH: rvs(70000) with ratio scaling")
print("=" * 70)

results_u = []
for trial in range(20):
    masses_all = imf.chabrier2005.distr.rvs(70000)
    mass_total = masses_all.sum()
    mask = masses_all <= 1.0
    masses = masses_all[mask]
    mass_surv = masses.sum()

    res = mi(masses, np.full(len(masses), logage), np.full(len(masses), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23 = np.sum(valid & (app_z < 23))

    # User's ratio: n_z23 / mass_total (using total mass)
    ratio_total = n_z23 / mass_total
    n_scaled_total = debris_mass * ratio_total

    # Correct ratio: n_z23 / mass_surv (using surviving mass)
    ratio_surv = n_z23 / mass_surv
    n_scaled_surv = debris_mass * ratio_surv

    results_u.append({
        'mass_total': mass_total, 'mass_surv': mass_surv,
        'n_all': len(masses_all), 'n_surv': len(masses),
        'n_z23': n_z23,
        'ratio_total': ratio_total, 'n_scaled_total': n_scaled_total,
        'ratio_surv': ratio_surv, 'n_scaled_surv': n_scaled_surv,
    })

print(f"  70000 stars drawn from Chabrier:")
print(f"  Mean total mass (sum all):     {np.mean([r['mass_total'] for r in results_u]):.0f} Msun")
print(f"  Mean surviving mass (sum <1):  {np.mean([r['mass_surv'] for r in results_u]):.0f} Msun")
print(f"  Mean N surviving stars:        {np.mean([r['n_surv'] for r in results_u]):.0f}")
print(f"  Mean N(z<23):                  {np.mean([r['n_z23'] for r in results_u]):.0f}")
print()
print(f"  Scaling to debris_mass = {debris_mass:.0f} Msun:")
mean_ratio_t = np.mean([r['ratio_total'] for r in results_u])
mean_ratio_s = np.mean([r['ratio_surv'] for r in results_u])
mean_n_t = np.mean([r['n_scaled_total'] for r in results_u])
mean_n_s = np.mean([r['n_scaled_surv'] for r in results_u])
print(f"    Using total mass:    N = {debris_mass:.0f} × ({np.mean([r['n_z23'] for r in results_u]):.0f}/{np.mean([r['mass_total'] for r in results_u]):.0f}) = {mean_n_t:.1f}")
print(f"    Using surviving mass: N = {debris_mass:.0f} × ({np.mean([r['n_z23'] for r in results_u]):.0f}/{np.mean([r['mass_surv'] for r in results_u]):.0f}) = {mean_n_s:.1f}")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 70)
print("SUMMARY: Draco f=0.01, z<23, debris = 3013 Msun surviving")
print("=" * 70)
print(f"\n  {'Method':<55s} {'N(z<23)':>8s}")
print(f"  {'-'*55} {'-'*8}")
print(f"  {'Kroupa pipeline (ref cluster + scale by surv mass)':<55s} {mean_k:8.1f}")
print(f"  {'Chabrier pipeline (ref cluster + scale by surv mass)':<55s} {mean_c:8.1f}")
print(f"  {'User rvs(70k): scale by TOTAL mass (sum all)':<55s} {mean_n_t:8.1f}")
print(f"  {'User rvs(70k): scale by SURVIVING mass (sum <1 Msun)':<55s} {mean_n_s:8.1f}")
print()
mean_mt = np.mean([r['mass_total'] for r in results_u])
mean_ms = np.mean([r['mass_surv'] for r in results_u])
print(f"  KEY: For 70k Chabrier stars:")
print(f"    Total mass (sum all):    {mean_mt:.0f} Msun")
print(f"    Surviving mass (sum <1): {mean_ms:.0f} Msun")
print(f"    Ratio total/surviving:   {mean_mt/mean_ms:.2f}x")
print()
print(f"  If you scale by total mass:    {debris_mass:.0f} × (N_z23/{mean_mt:.0f}) → {mean_n_t:.0f} stars")
print(f"  If you scale by surviving mass: {debris_mass:.0f} × (N_z23/{mean_ms:.0f}) → {mean_n_s:.0f} stars")
print(f"  Since debris_mass = surviving mass, you should divide by surviving mass!")
