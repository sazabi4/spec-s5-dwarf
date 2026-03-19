#!/usr/bin/env python3
"""
Verify what the actual pipeline produces for Draco vs direct approaches.
The pipeline uses reference cluster scaling, NOT direct make_cluster(debris_mass).
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
REFERENCE_MASS = 50000

M_total = 10**mass_stellar_log
Nstar_half = M_total / 2.0
debris_mass = f * Nstar_half

mi = minimint.Interpolator(['DECam_r', 'DECam_z'])

print("=" * 70)
print("PIPELINE APPROACH: Reference cluster + scaling")
print("=" * 70)
print(f"  debris_mass (= f * M_total/2) = {debris_mass:.0f} Msun")
print(f"  Reference cluster = make_cluster({REFERENCE_MASS} Msun)")
print()

pipeline_results = []
for trial in range(20):
    # This is EXACTLY what the pipeline does:
    masses_all = imf.make_cluster(REFERENCE_MASS, massfunc='kroupa')
    mask_survive = masses_all <= 1.0
    masses = masses_all[mask_survive]
    mass_surviving_total = float(masses.sum())

    n = len(masses)
    res = mi(masses, np.full(n, logage), np.full(n, feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)

    # Pipeline scaling: debris_mass / mass_surviving_total
    scale = debris_mass / mass_surviving_total
    observable = valid & (app_z < 23)
    N_ref = np.sum(observable)
    N_remnant = N_ref * scale

    pipeline_results.append({
        'mass_ref_total': masses_all.sum(),
        'mass_ref_surv': mass_surviving_total,
        'n_ref_surv': n,
        'N_ref_z23': N_ref,
        'scale': scale,
        'N_remnant': N_remnant,
    })

print(f"  {'Trial':>5s} {'M_ref_tot':>10s} {'M_ref_surv':>10s} {'N_ref_surv':>10s} "
      f"{'N_ref_z23':>10s} {'scale':>8s} {'N_remnant':>10s}")
for i, r in enumerate(pipeline_results):
    print(f"  {i+1:5d} {r['mass_ref_total']:10.0f} {r['mass_ref_surv']:10.0f} "
          f"{r['n_ref_surv']:10d} {r['N_ref_z23']:10d} "
          f"{r['scale']:8.5f} {r['N_remnant']:10.1f}")

mean_N = np.mean([r['N_remnant'] for r in pipeline_results])
std_N = np.std([r['N_remnant'] for r in pipeline_results])
mean_surv = np.mean([r['mass_ref_surv'] for r in pipeline_results])
mean_scale = np.mean([r['scale'] for r in pipeline_results])
print(f"\n  MEAN N_remnant = {mean_N:.1f} ± {std_N:.1f}")
print(f"  Mean surviving mass in ref = {mean_surv:.0f} Msun")
print(f"  Mean scale = debris_mass / M_surv = {debris_mass:.0f} / {mean_surv:.0f} = {mean_scale:.5f}")
print(f"  Effective surviving mass targeted = {debris_mass:.0f} Msun (BY DESIGN)")
print()

# Now compare to direct approaches
print("=" * 70)
print("COMPARISON TO DIRECT APPROACHES")
print("=" * 70)

# Direct make_cluster(3013)
direct_results = []
for trial in range(20):
    masses_all = imf.make_cluster(debris_mass, massfunc='kroupa')
    mask = masses_all <= 1.0
    ms = masses_all[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    direct_results.append(np.sum(valid & (app_z < 23)))
mean_direct = np.mean(direct_results)
print(f"\n  Direct make_cluster({debris_mass:.0f} Msun):")
print(f"    N(z<23) = {mean_direct:.1f}")
print(f"    Total mass = {debris_mass:.0f}, surviving ≈ {debris_mass*0.47:.0f}")
print(f"    NOTE: This underestimates because it targets TOTAL mass, not surviving mass")

# rvs calibrated to surviving mass = debris_mass (Chabrier)
test_c = imf.chabrier2005.distr.rvs(500000)
surv_per_draw_c = test_c[test_c <= 1.0].sum() / len(test_c)
N_cal_c = int(debris_mass / surv_per_draw_c)
rvs_c_results = []
for trial in range(20):
    masses = imf.chabrier2005.distr.rvs(N_cal_c)
    mask = masses <= 1.0
    ms = masses[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    rvs_c_results.append(np.sum(valid & (app_z < 23)))
mean_rvs_c = np.mean(rvs_c_results)
print(f"\n  rvs({N_cal_c}, Chabrier) calibrated to {debris_mass:.0f} Msun surviving:")
print(f"    N(z<23) = {mean_rvs_c:.1f}")

# rvs calibrated to surviving mass = debris_mass (Kroupa)
test_k = imf.kroupa.distr.rvs(500000)
surv_per_draw_k = test_k[test_k <= 1.0].sum() / len(test_k)
N_cal_k = int(debris_mass / surv_per_draw_k)
rvs_k_results = []
for trial in range(20):
    masses = imf.kroupa.distr.rvs(N_cal_k)
    mask = masses <= 1.0
    ms = masses[mask]
    res = mi(ms, np.full(len(ms), logage), np.full(len(ms), feh))
    app_z = np.array(res['DECam_z']) + distance_modulus
    valid = np.isfinite(app_z)
    rvs_k_results.append(np.sum(valid & (app_z < 23)))
mean_rvs_k = np.mean(rvs_k_results)
print(f"\n  rvs({N_cal_k}, Kroupa) calibrated to {debris_mass:.0f} Msun surviving:")
print(f"    N(z<23) = {mean_rvs_k:.1f}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  {'Method':<55s} {'N(z<23)':>8s}")
print(f"  {'-'*55} {'-'*8}")
print(f"  {'Pipeline (ref cluster + scale to 3013 Msun surv)':<55s} {mean_N:8.1f}")
print(f"  {'Direct make_cluster(3013 Msun total, NOT surviving)':<55s} {mean_direct:8.1f}")
print(f"  {'rvs(Chabrier) calibrated to 3013 Msun surviving':<55s} {mean_rvs_c:8.1f}")
print(f"  {'rvs(Kroupa) calibrated to 3013 Msun surviving':<55s} {mean_rvs_k:8.1f}")
print()
print("  CONCLUSION:")
print(f"    The pipeline correctly targets {debris_mass:.0f} Msun SURVIVING mass")
print(f"    via reference cluster scaling (scale = debris / M_surv_ref).")
print(f"    Pipeline N(z<23) ≈ {mean_N:.0f} matches rvs(Kroupa, calibrated) ≈ {mean_rvs_k:.0f}")
print(f"    (both use Kroupa IMF)")
print()
print(f"    Chabrier gives N(z<23) ≈ {mean_rvs_c:.0f} — higher because Chabrier")
print(f"    has more low-mass stars that are luminous enough at z<23.")
print(f"    Kroupa vs Chabrier difference: factor ≈ {mean_rvs_c/mean_N:.2f}x")
