"""Validation + benchmark des Grecques : python bench_greeks.py"""
import sys
from pathlib import Path
# pricer.py et greeks.py sont dans le même dossier que ce script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
from functools import partial

from greeks import (GREEKS, bs_greeks, crr_american, fd_greeks_crn,
                    mc_asian_greeks, mc_european_greeks)
from pricer import BlackScholes, lsm_american_put, mc_asian_arithmetic


def table(title, est, ref=None, secs=None):
    print(f"\n== {title}" + (f"  [{secs:.2f}s]" if secs is not None else "") + " ==")
    print(f"{'':<7}{'estimation':>14}{'± IC95':>11}{'référence':>13}")
    for g in GREEKS:
        if g not in est:
            continue
        e = est[g]
        val, ci = (e.price, 1.96 * e.std_error) if hasattr(e, "std_error") else (e, float("nan"))
        r = "" if ref is None or g not in ref else f"{float(ref[g]):13.5f}"
        print(f"{g:<7}{val:14.5f}{ci:11.5f}{r}")


def run(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


if __name__ == "__main__":
    m = BlackScholes(s0=100.0, r=0.05, sigma=0.2)
    K, T = 100.0, 1.0

    # 1. Européenne : MC (une simulation) vs formules fermées
    for kind in ("call", "put"):
        est, s = run(mc_european_greeks, m, K, T, kind)
        table(f"Européenne {kind} — MC pathwise / LR (2M)", est, bs_greeks(m, K, T, kind), s)

    # 2. Asiatique : estimateurs pathwise + contrôle vs différences finies CRN
    mc_asian_greeks(m, K, T, n_paths=5_000)                       # compilation JIT
    est, s = run(mc_asian_greeks, m, K, T, control_variate=False)
    table("Asiatique call — pathwise/LR SANS contrôle (500k)", est, secs=s)
    est, s = run(mc_asian_greeks, m, K, T)
    pricer = lambda mm, TT: mc_asian_arithmetic(mm, K, TT, n_paths=500_000, seed=9).price
    ref, s_fd = run(fd_greeks_crn, pricer, m, T)
    table("Asiatique call — pathwise/LR AVEC contrôle géométrique (500k)", est, ref, s)
    print(f"(référence = différences finies CRN, 9 pricings, {s_fd:.2f}s)")

    # 3. Américaine : LSM + différences finies CRN vs arbre binomial
    print("\n[Put américain] arbre binomial de référence (2000 pas)...", flush=True)
    ref, s_tree = run(crr_american, m, K, T, "put")
    print("[Put américain] compilation Numba du noyau LSM (1re fois seulement)...", flush=True)
    lsm_american_put(m, K, T, n_paths=1_000)

    def lsm(mm, TT, _count=[0]):
        _count[0] += 1
        print(f"  LSM pricing {_count[0]}/9 ...", end="\r", flush=True)
        return lsm_american_put(mm, K, TT, n_paths=200_000, seed=3).price

    est, s = run(fd_greeks_crn, lsm, m, T, rel_ds=0.02)
    print(" " * 30, end="\r")
    table("Put américain — LSM + différences finies CRN", est, ref, s)
    print(f"(référence = arbre CRR 2000 pas, {s_tree:.2f}s)")
