#!/usr/bin/env python3
"""Reproduce user's exact code and trace all quantities."""
import numpy as np
import minimint
import imf
import warnings
warnings.filterwarnings('ignore')

mi_iso = minimint.Interpolator(['DECam_g','DECam_r','DECam_i','DECam_z','Bessell_V'])

distance = 81  # kpc (user used C-19, but let's also do Draco)
dm = 5.0 * np.log10(distance) + 10.0
print(f"distance = {distance} kpc")
print(f"dm = {dm:.2f}")
print()

# User's exact code
stellar_mass = 70000
results = []
for trial in range(10):
    mass = imf.chabrier2005.distr.rvs(stellar_mass)
    mass = mass[mass < 1]
    n_surviving = len(mass)
    mass_surviving = sum(mass)

    qq = mi_iso(mass, np.log10(1.2e10), -2)
    app_z = qq['DECam_z'] + dm
    valid = np.isfinite(app_z)

    mag_thresh = 23
    Nstar = sum(app_z[valid] < mag_thresh)

    results.append({
        'n_surv': n_surviving,
        'mass_surv': mass_surviving,
        'Nstar': Nstar,
    })

print(f"User's code: stellar_mass = {stellar_mass} (number of stars drawn)")
print(f"  After mass<1 cut:")
print(f"    N surviving stars: {np.mean([r['n_surv'] for r in results]):.0f}")
print(f"    Surviving mass:    {np.mean([r['mass_surv'] for r in results]):.0f} Msun")
print(f"    Nstar (z<23):      {np.mean([r['Nstar'] for r in results]):.0f}")
print()
print(f"  So 70000 drawn → {np.mean([r['n_surv'] for r in results]):.0f} surviving stars")
print(f"  with total surviving mass = {np.mean([r['mass_surv'] for r in results]):.0f} Msun")
print()

# Now: for Draco with f=0.01, debris = 3013 Msun surviving
# How many stars should user draw to get 3013 Msun surviving?
mean_mass_surv_per_star = np.mean([r['mass_surv'] for r in results]) / np.mean([r['n_surv'] for r in results])
print(f"  Mean surviving mass per surviving star: {mean_mass_surv_per_star:.4f} Msun")
surv_mass_per_draw = np.mean([r['mass_surv'] for r in results]) / stellar_mass
print(f"  Surviving mass per drawn star: {surv_mass_per_draw:.4f} Msun")
print()

debris_mass = 3013  # Msun surviving
N_draw_needed = int(debris_mass / surv_mass_per_draw)
N_surv_expected = int(debris_mass / mean_mass_surv_per_star)
print(f"  For debris_mass = {debris_mass} Msun surviving:")
print(f"    Need to draw: {N_draw_needed} stars from rvs()")
print(f"    Expected surviving stars: {N_surv_expected}")
print()

# Scale user's result to 3013 Msun
mean_Nstar = np.mean([r['Nstar'] for r in results])
mean_mass = np.mean([r['mass_surv'] for r in results])
scaled = mean_Nstar * (debris_mass / mean_mass)
print(f"  Scaling user's result: {mean_Nstar:.0f} × ({debris_mass}/{mean_mass:.0f}) = {scaled:.1f}")
print()

# Verify by actually drawing the right number
print("=" * 60)
print(f"VERIFICATION: rvs({N_draw_needed}) to target {debris_mass} Msun surviving")
print("=" * 60)
verify = []
for trial in range(20):
    mass = imf.chabrier2005.distr.rvs(N_draw_needed)
    mass = mass[mass < 1]
    qq = mi_iso(mass, np.log10(1.2e10), -2)
    app_z = qq['DECam_z'] + dm
    valid = np.isfinite(app_z)
    Nstar = sum(app_z[valid] < 23)
    verify.append({'mass_surv': sum(mass), 'n_surv': len(mass), 'Nstar': Nstar})

print(f"  Surviving mass: {np.mean([r['mass_surv'] for r in verify]):.0f} Msun (target: {debris_mass})")
print(f"  N surviving:    {np.mean([r['n_surv'] for r in verify]):.0f}")
print(f"  Nstar (z<23):   {np.mean([r['Nstar'] for r in verify]):.1f}")
print()

# Now do Draco distance instead of C-19
print("=" * 60)
print("SAME BUT AT DRACO DISTANCE (81.55 kpc, dm=19.557)")
print("=" * 60)
dm_draco = 19.557
verify_draco = []
for trial in range(20):
    mass = imf.chabrier2005.distr.rvs(N_draw_needed)
    mass = mass[mass < 1]
    qq = mi_iso(mass, np.log10(1.2e10), -2)
    app_z = qq['DECam_z'] + dm_draco
    valid = np.isfinite(app_z)
    Nstar = sum(app_z[valid] < 23)
    verify_draco.append({'mass_surv': sum(mass), 'Nstar': Nstar})

print(f"  Surviving mass: {np.mean([r['mass_surv'] for r in verify_draco]):.0f} Msun")
print(f"  Nstar (z<23):   {np.mean([r['Nstar'] for r in verify_draco]):.1f}")
print()

# Also user's 70k draw at Draco distance
print("=" * 60)
print("USER'S rvs(70000) AT DRACO DISTANCE")
print("=" * 60)
verify70k = []
for trial in range(10):
    mass = imf.chabrier2005.distr.rvs(70000)
    mass = mass[mass < 1]
    qq = mi_iso(mass, np.log10(1.2e10), -2)
    app_z = qq['DECam_z'] + dm_draco
    valid = np.isfinite(app_z)
    Nstar = sum(app_z[valid] < 23)
    verify70k.append({'mass_surv': sum(mass), 'n_surv': len(mass), 'Nstar': Nstar})

mean_70k_mass = np.mean([r['mass_surv'] for r in verify70k])
mean_70k_Nstar = np.mean([r['Nstar'] for r in verify70k])
print(f"  Surviving mass: {mean_70k_mass:.0f} Msun")
print(f"  N surviving:    {np.mean([r['n_surv'] for r in verify70k]):.0f}")
print(f"  Nstar (z<23):   {mean_70k_Nstar:.0f}")
print(f"  Scaled to 3013 Msun: {mean_70k_Nstar} × (3013/{mean_70k_mass:.0f}) = {mean_70k_Nstar * 3013/mean_70k_mass:.1f}")
