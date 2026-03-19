#!/usr/bin/env python3
"""
Compare two IMF sampling approaches for Draco (f=0.01, Zmag=23):
  1) imf.make_cluster(debris_mass_Msun) - mass-based
  2) imf.chabrier2005.distr.rvs(N_stars) - count-based
"""
import numpy as np
import imf
import minimint
import warnings
warnings.filterwarnings('ignore')

# Draco parameters (from LVDB)
mass_stellar_log = 5.78   # log10(M*/Msun)
distance_modulus = 19.557
feh = -2.0
logage = np.log10(12e9)
f = 0.01

M_total = 10**mass_stellar_log
Nstar_half = M_total / 2.0
debris_mass = f * Nstar_half  # Msun

print(f"Draco: log M* = {mass_stellar_log}")
print(f"  M_total     = {M_total:.0f} Msun")
print(f"  Nstar_half  = {Nstar_half:.0f} Msun")
print(f"  debris_mass = {debris_mass:.0f} Msun (f={f})")
print(f"  DM = {distance_modulus}, [Fe/H] = {feh}, log(age) = {logage:.3f}")
print()

mi = minimint.Interpolator(['DECam_r', 'DECam_z'])

# -----------------------------------------------------------
# Approach 1: make_cluster(debris_mass) — targets TOTAL MASS
# -----------------------------------------------------------
print("=" * 60)
print("APPROACH 1: imf.make_cluster(debris_mass_Msun)")
print("  Argument = total mass to generate = {:.0f} Msun".format(debris_mass))
print("=" * 60)

results1 = []
for trial in range(10):
    masses = imf.make_cluster(debris_mass, massfunc='kroupa')
    n_total = len(masses)
    total_mass = masses.sum()
    mask = masses <= 1.0
    masses_surv = masses[mask]
    n_surv = len(masses_surv)
    mass_surv = masses_surv.sum()

    res = mi(masses_surv, np.full(n_surv, logage), np.full(n_surv, feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23 = np.sum(valid & (app_z < 23))
    n_z21 = np.sum(valid & (app_z < 21))
    n_z19 = np.sum(valid & (app_z < 19))

    results1.append({
        'n_total': n_total, 'total_mass': total_mass,
        'n_surv': n_surv, 'mass_surv': mass_surv,
        'n_z23': n_z23, 'n_z21': n_z21, 'n_z19': n_z19
    })

print(f"  {'Trial':>5s} {'N_draw':>8s} {'M_tot':>10s} {'N_surv':>8s} "
      f"{'M_surv':>10s} {'z<23':>6s} {'z<21':>6s} {'z<19':>6s}")
for i, r in enumerate(results1):
    print(f"  {i+1:5d} {r['n_total']:8d} {r['total_mass']:10.0f} "
          f"{r['n_surv']:8d} {r['mass_surv']:10.0f} "
          f"{r['n_z23']:6d} {r['n_z21']:6d} {r['n_z19']:6d}")
means1 = {k: np.mean([r[k] for r in results1]) for k in results1[0]}
print(f"  {'MEAN':>5s} {means1['n_total']:8.0f} {means1['total_mass']:10.0f} "
      f"{means1['n_surv']:8.0f} {means1['mass_surv']:10.0f} "
      f"{means1['n_z23']:6.1f} {means1['n_z21']:6.1f} {means1['n_z19']:6.1f}")
print()

# -----------------------------------------------------------
# Approach 2: rvs(N_stars) — draws N INDIVIDUAL STARS
# Using debris_mass as the number of stars (user's interpretation)
# -----------------------------------------------------------
print("=" * 60)
print("APPROACH 2: imf.chabrier2005.distr.rvs(N)")
print("  Argument = number of stars to draw")
print("=" * 60)

# Try with N = debris_mass (= 3013, treating the same number as star count)
N_stars = int(debris_mass)
print(f"\n  Case A: N = {N_stars} (same numerical value as debris_mass)")
results2a = []
for trial in range(10):
    masses = imf.chabrier2005.distr.rvs(N_stars)
    total_mass = masses.sum()
    mask = masses <= 1.0
    masses_surv = masses[mask]
    n_surv = len(masses_surv)
    mass_surv = masses_surv.sum()

    res = mi(masses_surv, np.full(n_surv, logage), np.full(n_surv, feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23 = np.sum(valid & (app_z < 23))
    n_z21 = np.sum(valid & (app_z < 21))
    n_z19 = np.sum(valid & (app_z < 19))

    results2a.append({
        'n_total': N_stars, 'total_mass': total_mass,
        'n_surv': n_surv, 'mass_surv': mass_surv,
        'n_z23': n_z23, 'n_z21': n_z21, 'n_z19': n_z19
    })

print(f"  {'Trial':>5s} {'N_draw':>8s} {'M_tot':>10s} {'N_surv':>8s} "
      f"{'M_surv':>10s} {'z<23':>6s} {'z<21':>6s} {'z<19':>6s}")
for i, r in enumerate(results2a):
    print(f"  {i+1:5d} {r['n_total']:8d} {r['total_mass']:10.0f} "
          f"{r['n_surv']:8d} {r['mass_surv']:10.0f} "
          f"{r['n_z23']:6d} {r['n_z21']:6d} {r['n_z19']:6d}")
means2a = {k: np.mean([r[k] for r in results2a]) for k in results2a[0]}
print(f"  {'MEAN':>5s} {means2a['n_total']:8.0f} {means2a['total_mass']:10.0f} "
      f"{means2a['n_surv']:8.0f} {means2a['mass_surv']:10.0f} "
      f"{means2a['n_z23']:6.1f} {means2a['n_z21']:6.1f} {means2a['n_z19']:6.1f}")

# Try N = 62700 (user's stated number)
N_stars2 = 62700
print(f"\n  Case B: N = {N_stars2} (user's stated value)")
results2b = []
for trial in range(10):
    masses = imf.chabrier2005.distr.rvs(N_stars2)
    total_mass = masses.sum()
    mask = masses <= 1.0
    masses_surv = masses[mask]
    n_surv = len(masses_surv)
    mass_surv = masses_surv.sum()

    res = mi(masses_surv, np.full(n_surv, logage), np.full(n_surv, feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23 = np.sum(valid & (app_z < 23))
    n_z21 = np.sum(valid & (app_z < 21))
    n_z19 = np.sum(valid & (app_z < 19))

    results2b.append({
        'n_total': N_stars2, 'total_mass': total_mass,
        'n_surv': n_surv, 'mass_surv': mass_surv,
        'n_z23': n_z23, 'n_z21': n_z21, 'n_z19': n_z19
    })

print(f"  {'Trial':>5s} {'N_draw':>8s} {'M_tot':>10s} {'N_surv':>8s} "
      f"{'M_surv':>10s} {'z<23':>6s} {'z<21':>6s} {'z<19':>6s}")
for i, r in enumerate(results2b):
    print(f"  {i+1:5d} {r['n_total']:8d} {r['total_mass']:10.0f} "
          f"{r['n_surv']:8d} {r['mass_surv']:10.0f} "
          f"{r['n_z23']:6d} {r['n_z21']:6d} {r['n_z19']:6d}")
means2b = {k: np.mean([r[k] for r in results2b]) for k in results2b[0]}
print(f"  {'MEAN':>5s} {means2b['n_total']:8.0f} {means2b['total_mass']:10.0f} "
      f"{means2b['n_surv']:8.0f} {means2b['mass_surv']:10.0f} "
      f"{means2b['n_z23']:6.1f} {means2b['n_z21']:6.1f} {means2b['n_z19']:6.1f}")

# -----------------------------------------------------------
# Summary comparison
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY COMPARISON FOR DRACO (f=0.01)")
print("=" * 60)
print(f"\n  debris_mass = f * Nstar_half = {debris_mass:.0f}")
print(f"\n  Approach 1: make_cluster({debris_mass:.0f} Msun)")
print(f"    → {means1['n_total']:.0f} stars drawn, "
      f"{means1['mass_surv']:.0f} Msun surviving, "
      f"{means1['n_z23']:.1f} stars with z<23")
print(f"\n  Approach 2a: rvs({int(debris_mass)} stars)")
print(f"    → {means2a['n_total']:.0f} stars drawn, "
      f"{means2a['mass_surv']:.0f} Msun surviving, "
      f"{means2a['n_z23']:.1f} stars with z<23")
print(f"\n  Approach 2b: rvs(62700 stars)")
print(f"    → {means2b['n_total']:.0f} stars drawn, "
      f"{means2b['mass_surv']:.0f} Msun surviving, "
      f"{means2b['n_z23']:.1f} stars with z<23")

print(f"\n  KEY INSIGHT:")
print(f"    make_cluster({debris_mass:.0f} Msun) generates {means1['n_total']:.0f} stars "
      f"totaling {means1['total_mass']:.0f} Msun")
print(f"    rvs({int(debris_mass)}) generates {int(debris_mass)} stars "
      f"totaling {means2a['total_mass']:.0f} Msun")
print(f"    The same number {int(debris_mass)} means MASS in approach 1, "
      f"COUNT in approach 2")

# Compute mean stellar mass for Chabrier
masses_test = imf.chabrier2005.distr.rvs(100000)
mean_mass_all = masses_test.mean()
mean_mass_surv = masses_test[masses_test <= 1.0].mean()
print(f"\n  Mean mass (Chabrier2005): all = {mean_mass_all:.3f} Msun, "
      f"surviving (<1 Msun) = {mean_mass_surv:.3f} Msun")
print(f"  Fraction surviving (<1 Msun) = {np.sum(masses_test<=1)/len(masses_test):.3f}")

# What N would give debris_mass total surviving mass?
frac_surv = np.sum(masses_test <= 1) / len(masses_test)
mass_surv_per_star = mean_mass_surv * frac_surv + 0  # weighted by fraction surviving
N_needed = debris_mass / (mean_mass_all)  # to get debris_mass total
print(f"\n  To get {debris_mass:.0f} Msun total with rvs(), "
      f"need N = {debris_mass/mean_mass_all:.0f} stars")
N_needed2 = debris_mass / mean_mass_surv / frac_surv
print(f"  Alternatively, need N = {int(N_needed2)} for {debris_mass:.0f} Msun "
      f"in surviving stars only")
