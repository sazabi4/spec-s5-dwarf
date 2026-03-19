#!/usr/bin/env python3
"""
Detailed comparison of two IMF approaches for Draco (f=0.01, Zmag=23).
Shows that when properly calibrated, both give consistent observable fractions.
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
print(f"  Nstar_half  = {Nstar_half:.0f} Msun  (= M_total / 2)")
print(f"  debris_mass = f * Nstar_half = {debris_mass:.0f} Msun")
print()

mi = minimint.Interpolator(['DECam_r', 'DECam_z'])

# First: characterize the Chabrier IMF
print("=" * 70)
print("CHARACTERIZING CHABRIER 2005 IMF")
print("=" * 70)
test_masses = imf.chabrier2005.distr.rvs(500000)
mean_all = test_masses.mean()
surv_mask = test_masses <= 1.0
frac_surv = surv_mask.sum() / len(test_masses)
mean_surv = test_masses[surv_mask].mean()
mass_surv_per_drawn = mean_all * frac_surv  # not quite right
print(f"  Mean mass (all):        {mean_all:.4f} Msun")
print(f"  Mean mass (surviving):  {mean_surv:.4f} Msun")
print(f"  Fraction surviving:     {frac_surv:.4f}")
print(f"  Total surviving mass per star drawn: {test_masses[surv_mask].sum()/len(test_masses):.4f} Msun")
surv_mass_per_draw = test_masses[surv_mask].sum() / len(test_masses)
print()

# How many rvs() stars needed to get debris_mass Msun of surviving stars?
N_needed = int(debris_mass / surv_mass_per_draw)
print(f"  To get {debris_mass:.0f} Msun surviving mass from rvs():")
print(f"    N = debris_mass / (surv_mass_per_draw) = {debris_mass:.0f} / {surv_mass_per_draw:.4f} = {N_needed}")
print()

# -----------------------------------------------------------
# Approach 1: make_cluster(debris_mass_Msun)
# -----------------------------------------------------------
print("=" * 70)
print(f"APPROACH 1: imf.make_cluster({debris_mass:.0f} Msun)  [Kroupa]")
print("  → targets TOTAL MASS = debris_mass")
print("=" * 70)

n_z23_list, n_surv_list, m_surv_list, m_tot_list = [], [], [], []
for trial in range(20):
    masses = imf.make_cluster(debris_mass, massfunc='kroupa')
    mask = masses <= 1.0
    ms = masses[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23_list.append(np.sum(valid & (app_z < 23)))
    n_surv_list.append(len(ms))
    m_surv_list.append(ms.sum())
    m_tot_list.append(masses.sum())

print(f"  Stars drawn:    {np.mean([len(imf.make_cluster(debris_mass, massfunc='kroupa')) for _ in range(5)]):.0f}")
print(f"  Total mass:     {np.mean(m_tot_list):.0f} Msun")
print(f"  Surviving mass: {np.mean(m_surv_list):.0f} Msun")
print(f"  N surviving:    {np.mean(n_surv_list):.0f}")
print(f"  N(z<23):        {np.mean(n_z23_list):.1f} ± {np.std(n_z23_list):.1f}")
frac_obs_1 = np.mean(n_z23_list) / np.mean(n_surv_list)
print(f"  Observable fraction (N_z23/N_surv): {frac_obs_1:.5f}")
print()

# -----------------------------------------------------------
# Approach 2a: rvs(debris_mass) — same number, but as star count
# -----------------------------------------------------------
print("=" * 70)
print(f"APPROACH 2a: imf.chabrier2005.distr.rvs({int(debris_mass)})  [Chabrier]")
print(f"  → draws {int(debris_mass)} STARS (not Msun)")
print("=" * 70)

n_z23_list2a, n_surv_list2a, m_surv_list2a, m_tot_list2a = [], [], [], []
for trial in range(20):
    masses = imf.chabrier2005.distr.rvs(int(debris_mass))
    mask = masses <= 1.0
    ms = masses[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23_list2a.append(np.sum(valid & (app_z < 23)))
    n_surv_list2a.append(len(ms))
    m_surv_list2a.append(ms.sum())
    m_tot_list2a.append(masses.sum())

print(f"  Stars drawn:    {int(debris_mass)}")
print(f"  Total mass:     {np.mean(m_tot_list2a):.0f} Msun")
print(f"  Surviving mass: {np.mean(m_surv_list2a):.0f} Msun")
print(f"  N surviving:    {np.mean(n_surv_list2a):.0f}")
print(f"  N(z<23):        {np.mean(n_z23_list2a):.1f} ± {np.std(n_z23_list2a):.1f}")
frac_obs_2a = np.mean(n_z23_list2a) / np.mean(n_surv_list2a)
print(f"  Observable fraction (N_z23/N_surv): {frac_obs_2a:.5f}")
print()

# -----------------------------------------------------------
# Approach 2b: rvs(N_needed) — calibrated to match debris_mass
# -----------------------------------------------------------
print("=" * 70)
print(f"APPROACH 2b: imf.chabrier2005.distr.rvs({N_needed})")
print(f"  → N calibrated so surviving mass ≈ {debris_mass:.0f} Msun")
print("=" * 70)

n_z23_list2b, n_surv_list2b, m_surv_list2b, m_tot_list2b = [], [], [], []
for trial in range(20):
    masses = imf.chabrier2005.distr.rvs(N_needed)
    mask = masses <= 1.0
    ms = masses[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23_list2b.append(np.sum(valid & (app_z < 23)))
    n_surv_list2b.append(len(ms))
    m_surv_list2b.append(ms.sum())
    m_tot_list2b.append(masses.sum())

print(f"  Stars drawn:    {N_needed}")
print(f"  Total mass:     {np.mean(m_tot_list2b):.0f} Msun")
print(f"  Surviving mass: {np.mean(m_surv_list2b):.0f} Msun")
print(f"  N surviving:    {np.mean(n_surv_list2b):.0f}")
print(f"  N(z<23):        {np.mean(n_z23_list2b):.1f} ± {np.std(n_z23_list2b):.1f}")
frac_obs_2b = np.mean(n_z23_list2b) / np.mean(n_surv_list2b)
print(f"  Observable fraction (N_z23/N_surv): {frac_obs_2b:.5f}")
print()

# -----------------------------------------------------------
# Also test: what if we use Kroupa for rvs? (make_cluster uses Kroupa)
# -----------------------------------------------------------
print("=" * 70)
print(f"APPROACH 3: imf.kroupa.distr.rvs() calibrated to {debris_mass:.0f} Msun surviving")
print("=" * 70)
test_k = imf.kroupa.distr.rvs(500000)
surv_mass_per_draw_k = test_k[test_k <= 1.0].sum() / len(test_k)
N_needed_k = int(debris_mass / surv_mass_per_draw_k)
print(f"  Kroupa surviving mass per draw: {surv_mass_per_draw_k:.4f} Msun")
print(f"  N needed: {N_needed_k}")

n_z23_list3, n_surv_list3, m_surv_list3 = [], [], []
for trial in range(20):
    masses = imf.kroupa.distr.rvs(N_needed_k)
    mask = masses <= 1.0
    ms = masses[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    n_z23_list3.append(np.sum(valid & (app_z < 23)))
    n_surv_list3.append(len(ms))
    m_surv_list3.append(ms.sum())

print(f"  Surviving mass: {np.mean(m_surv_list3):.0f} Msun")
print(f"  N surviving:    {np.mean(n_surv_list3):.0f}")
print(f"  N(z<23):        {np.mean(n_z23_list3):.1f} ± {np.std(n_z23_list3):.1f}")
frac_obs_3 = np.mean(n_z23_list3) / np.mean(n_surv_list3)
print(f"  Observable fraction (N_z23/N_surv): {frac_obs_3:.5f}")
print()

# -----------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------
print("=" * 70)
print("SUMMARY: All approaches for Draco, f=0.01, z<23")
print("=" * 70)
print(f"  {'Approach':<45s} {'M_surv':>8s} {'N_surv':>8s} {'N(z<23)':>8s} {'frac':>8s}")
print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'1: make_cluster(3013 Msun, kroupa)':<45s} "
      f"{np.mean(m_surv_list):8.0f} {np.mean(n_surv_list):8.0f} "
      f"{np.mean(n_z23_list):8.1f} {frac_obs_1:8.5f}")
print(f"  {'2a: rvs(3013 stars, chabrier)':<45s} "
      f"{np.mean(m_surv_list2a):8.0f} {np.mean(n_surv_list2a):8.0f} "
      f"{np.mean(n_z23_list2a):8.1f} {frac_obs_2a:8.5f}")
print(f"  {'2b: rvs(N_cal stars, chabrier) → 3013 Msun':<45s} "
      f"{np.mean(m_surv_list2b):8.0f} {np.mean(n_surv_list2b):8.0f} "
      f"{np.mean(n_z23_list2b):8.1f} {frac_obs_2b:8.5f}")
print(f"  {'3: rvs(N_cal stars, kroupa) → 3013 Msun':<45s} "
      f"{np.mean(m_surv_list3):8.0f} {np.mean(n_surv_list3):8.0f} "
      f"{np.mean(n_z23_list3):8.1f} {frac_obs_3:8.5f}")
print()
print("  KEY: When calibrated to the SAME surviving mass (~3013 Msun),")
print("  all approaches give consistent N(z<23) counts.")
print(f"  The observable fraction is ~{frac_obs_1:.4f} = "
      f"~{frac_obs_1*100:.2f}% of surviving stars have z<23.")
print()
print("  The difference arises ONLY from what the input number means:")
print(f"    make_cluster(3013) → 3013 = total MASS in Msun → N(z<23) ≈ {np.mean(n_z23_list):.0f}")
print(f"    rvs(3013)          → 3013 = number of STARS    → N(z<23) ≈ {np.mean(n_z23_list2a):.0f}")
print(f"    rvs({N_needed})        → calibrated to same mass → N(z<23) ≈ {np.mean(n_z23_list2b):.0f}")
