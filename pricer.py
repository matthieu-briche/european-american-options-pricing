"""
pricer.py — Pricing d'options financières optimisé CPU.

Méthodes inspirées de G. Pagès, *Numerical Probability* (Springer, 2018) :
  - Ch. 2  : Monte Carlo + intervalle de confiance, Greeks (méthode pathwise / tangente)
  - Ch. 3  : réduction de variance (antithétique, variable de contrôle à beta optimal)
  - Ch. 4  : quasi-Monte Carlo randomisé (Sobol brouillé)
  - Ch. 7-8: options path-dependent (asiatiques) par schéma exact de Black–Scholes
  - Ch. 12 : options américaines par régression (Longstaff–Schwartz)

Principes d'optimisation CPU appliqués :
  1. Vectorisation NumPy (aucune boucle Python sur les trajectoires).
  2. Noyaux Numba @njit(parallel=True, fastmath=True) pour les boucles en temps
     (path-dependent) -> code machine, multi-cœur, sans matrice (n_paths, n_steps).
  3. Traitement par blocs (chunks) : mémoire bornée, données dans le cache CPU.
  4. float64 contigu, opérations in-place, pas d'allocations inutiles.
  5. RNG moderne (PCG64) avec SeedSequence -> reproductible et parallélisable.
  6. Réduction de variance = le meilleur "optimiseur CPU" : diviser la variance
     par 10 équivaut à diviser le nombre de simulations (donc le temps) par 10.
"""
from __future__ import annotations

if __name__ == "__main__":
    # Fichier lancé directement (python pricer.py / bouton ▶ de VS Code) : on exécute la
    # démo via le module importé "pricer" et non via "__main__", car le cache Numba
    # est lié au nom du module.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    import pricer as _m
    _m._demo()
    raise SystemExit(0)

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numba import njit, prange
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

OptionType = Literal["call", "put"]
Z95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Modèle et résultat
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BlackScholes:
    """Modèle de Black–Scholes : dS = S((r - q)dt + sigma dW)."""
    s0: float
    r: float
    sigma: float
    q: float = 0.0

    def __post_init__(self) -> None:
        if self.s0 <= 0 or self.sigma <= 0:
            raise ValueError("s0 et sigma doivent être > 0")


def check_proba_crr(p: float, m: BlackScholes, dt: float) -> None:
    """
    Arbre CRR : la probabilité risque-neutre doit être dans ]0, 1[, sinon l'arbre
    produit des prix sans signification (arbitrage dans l'arbre). Avec u = e^{σ√dt},
    c'est équivalent (exactement) à  σ > |r - q| · √dt.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"Arbre CRR : probabilité p={p:.4f} hors de ]0, 1[ "
            f"(sigma={m.sigma}, r-q={m.r - m.q:+.4f}, dt={dt:.2e}). "
            f"Il faut sigma > |r-q|·√dt = {abs(m.r - m.q) * math.sqrt(dt):.4g} : "
            "augmenter n ou vérifier les paramètres.")


def sigma_min_crr(r: float, q: float, T: float, n: int) -> float:
    """Plus petite volatilité acceptée par un arbre CRR à n pas (voir check_proba_crr)."""
    return abs(r - q) * math.sqrt(T / n)


@dataclass(frozen=True, slots=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        h = Z95 * self.std_error
        return self.price - h, self.price + h

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"{self.price:.6f} ± {Z95*self.std_error:.6f}  (IC95 [{lo:.6f}, {hi:.6f}], N={self.n_paths:,})"


class _RunningStats:
    """Moyenne/variance en ligne par blocs (Chan et al.) : mémoire O(1)."""
    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x: np.ndarray) -> None:
        nb = x.size
        mb = float(x.mean())
        m2b = float(((x - mb) ** 2).sum())
        n = self.n + nb
        d = mb - self.mean
        self.mean += d * nb / n
        self.m2 += m2b + d * d * self.n * nb / n
        self.n = n

    def result(self, scale: float = 1.0) -> MCResult:
        var = self.m2 / (self.n - 1)
        return MCResult(scale * self.mean, scale * math.sqrt(var / self.n), self.n)


# --------------------------------------------------------------------------- #
# 1. Formules fermées (référence + variables de contrôle) — vectorisées
# --------------------------------------------------------------------------- #
def bs_price(m: BlackScholes, K, T, kind: OptionType = "call"):
    """Prix Black–Scholes, vectorisé sur K et T (broadcast NumPy)."""
    K, T = np.asarray(K, float), np.asarray(T, float)
    sq = m.sigma * np.sqrt(T)
    d1 = (np.log(m.s0 / K) + (m.r - m.q + 0.5 * m.sigma**2) * T) / sq
    d2 = d1 - sq
    df_r, df_q = np.exp(-m.r * T), np.exp(-m.q * T)
    if kind == "call":
        return m.s0 * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    return K * df_r * ndtr(-d2) - m.s0 * df_q * ndtr(-d1)


def bs_delta_vega(m: BlackScholes, K, T, kind: OptionType = "call"):
    K, T = np.asarray(K, float), np.asarray(T, float)
    sq = m.sigma * np.sqrt(T)
    d1 = (np.log(m.s0 / K) + (m.r - m.q + 0.5 * m.sigma**2) * T) / sq
    df_q = np.exp(-m.q * T)
    delta = df_q * (ndtr(d1) if kind == "call" else ndtr(d1) - 1.0)
    vega = m.s0 * df_q * np.sqrt(T) * np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return delta, vega


def geometric_asian_price(m: BlackScholes, K: float, T: float, n: int,
                          kind: OptionType = "call") -> float:
    """Asiatique géométrique discrète (dates iT/n, i=1..n) : log G gaussien."""
    mu = math.log(m.s0) + (m.r - m.q - 0.5 * m.sigma**2) * T * (n + 1) / (2 * n)
    s = m.sigma * math.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n * n))
    d2 = (mu - math.log(K)) / s
    d1 = d2 + s
    fwd = math.exp(mu + 0.5 * s * s)
    if kind == "call":
        return math.exp(-m.r * T) * (fwd * ndtr(d1) - K * ndtr(d2))
    return math.exp(-m.r * T) * (K * ndtr(-d2) - fwd * ndtr(-d1))


# --------------------------------------------------------------------------- #
# 2. Européenne Monte Carlo : antithétique + variable de contrôle (Ch. 2-3)
# --------------------------------------------------------------------------- #
def mc_european(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                n_paths: int = 2_000_000, chunk: int = 2**18,
                antithetic: bool = True, control_variate: bool = True,
                seed: int = 42) -> MCResult:
    """
    Simulation exacte de S_T (pas de discrétisation).
    Variable de contrôle : S_T, d'espérance connue S0 e^{(r-q)T}.
    Beta optimal estimé sur un bloc pilote indépendant (évite le biais).
    """
    rng = np.random.default_rng(seed)
    drift = (m.r - m.q - 0.5 * m.sigma**2) * T
    vol = m.sigma * math.sqrt(T)
    fwd = m.s0 * math.exp((m.r - m.q) * T)
    sign = 1.0 if kind == "call" else -1.0

    def payoff_and_cv(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = m.s0 * np.exp(drift + vol * z)
        pay = np.maximum(sign * (st - K), 0.0)
        if antithetic:
            st2 = m.s0 * np.exp(drift - vol * z)
            pay = 0.5 * (pay + np.maximum(sign * (st2 - K), 0.0))
            st = 0.5 * (st + st2)
        return pay, st

    beta = 0.0
    if control_variate:
        p, c = payoff_and_cv(rng.standard_normal(min(chunk, 50_000)))
        cov = np.cov(p, c)
        beta = cov[0, 1] / cov[1, 1]

    stats = _RunningStats()
    n_draws = n_paths // 2 if antithetic else n_paths
    done = 0
    while done < n_draws:
        b = min(chunk, n_draws - done)
        p, c = payoff_and_cv(rng.standard_normal(b))
        if control_variate:
            p -= beta * (c - fwd)
        stats.update(p)
        done += b
    res = stats.result(math.exp(-m.r * T))
    return MCResult(res.price, res.std_error, n_paths)


# --------------------------------------------------------------------------- #
# 3. Quasi-Monte Carlo randomisé (Ch. 4) : Sobol brouillé + réplications
# --------------------------------------------------------------------------- #
def rqmc_european(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                  log2_n: int = 16, n_replicas: int = 32, seed: int = 0) -> MCResult:
    """
    Chaque réplique = un Sobol brouillé indépendant -> estimateur sans biais.
    L'erreur se mesure sur la dispersion entre répliques (IC valide).
    """
    drift = (m.r - m.q - 0.5 * m.sigma**2) * T
    vol = m.sigma * math.sqrt(T)
    sign = 1.0 if kind == "call" else -1.0
    ss = np.random.SeedSequence(seed)
    est = np.empty(n_replicas)
    for i, child in enumerate(ss.spawn(n_replicas)):
        u = qmc.Sobol(d=1, scramble=True, seed=np.random.default_rng(child)).random_base2(log2_n).ravel()
        z = ndtri(u)
        st = m.s0 * np.exp(drift + vol * z)
        est[i] = np.maximum(sign * (st - K), 0.0).mean()
    df = math.exp(-m.r * T)
    return MCResult(df * est.mean(), df * est.std(ddof=1) / math.sqrt(n_replicas),
                    n_replicas * 2**log2_n)


# --------------------------------------------------------------------------- #
# 4. Greeks Monte Carlo : méthode pathwise / processus tangent (Ch. 2.2.4)
# --------------------------------------------------------------------------- #
def mc_greeks_pathwise(m: BlackScholes, K: float, T: float, n_paths: int = 1_000_000,
                       seed: int = 7) -> dict[str, MCResult]:
    """Delta et Vega d'un call : dérivée du payoff le long de la trajectoire."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    sq = math.sqrt(T)
    st = m.s0 * np.exp((m.r - m.q - 0.5 * m.sigma**2) * T + m.sigma * sq * z)
    itm = st > K
    df = math.exp(-m.r * T)
    d_delta = df * itm * st / m.s0                              # dS_T/dS0 = S_T/S0
    d_vega = df * itm * st * (sq * z - m.sigma * T)             # dS_T/dsigma
    out = {}
    for name, x in (("delta", d_delta), ("vega", d_vega)):
        out[name] = MCResult(float(x.mean()), float(x.std(ddof=1) / math.sqrt(n_paths)), n_paths)
    return out


# --------------------------------------------------------------------------- #
# 5. Asiatique arithmétique : noyau Numba parallèle + contrôle géométrique
# --------------------------------------------------------------------------- #
@njit(parallel=True, fastmath=True, cache=True)
def _asian_kernel(z, log_s0, drift_dt, vol_dt, K, sign):
    """
    z : (n_paths, n_steps) gaussiennes. Une trajectoire par itération prange :
    boucle temporelle en registres, aucune matrice de prix stockée.
    Renvoie les payoffs arithmétique et géométrique (variable de contrôle).
    """
    n_paths, n_steps = z.shape
    arith = np.empty(n_paths)
    geo = np.empty(n_paths)
    for i in prange(n_paths):
        log_s = log_s0
        acc = 0.0
        acc_log = 0.0
        for j in range(n_steps):
            log_s += drift_dt + vol_dt * z[i, j]
            acc += math.exp(log_s)
            acc_log += log_s
        a = acc / n_steps
        g = math.exp(acc_log / n_steps)
        arith[i] = max(sign * (a - K), 0.0)
        geo[i] = max(sign * (g - K), 0.0)
    return arith, geo


def mc_asian_arithmetic(m: BlackScholes, K: float, T: float, n_steps: int = 52,
                        kind: OptionType = "call", n_paths: int = 500_000,
                        chunk: int = 2**16, control_variate: bool = True,
                        seed: int = 1) -> MCResult:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift_dt = (m.r - m.q - 0.5 * m.sigma**2) * dt
    vol_dt = m.sigma * math.sqrt(dt)
    sign = 1.0 if kind == "call" else -1.0
    geo_exact = geometric_asian_price(m, K, T, n_steps, kind) * math.exp(m.r * T)

    def block(b):
        z = rng.standard_normal((b, n_steps))
        return _asian_kernel(z, math.log(m.s0), drift_dt, vol_dt, K, sign)

    beta = 0.0
    if control_variate:
        a, g = block(20_000)
        cov = np.cov(a, g)
        beta = cov[0, 1] / cov[1, 1]

    stats = _RunningStats()
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        a, g = block(b)
        if control_variate:
            a -= beta * (g - geo_exact)
        stats.update(a)
        done += b
    return stats.result(math.exp(-m.r * T))


# --------------------------------------------------------------------------- #
# 6. Américaine : Longstaff–Schwartz (Ch. 12)
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def _lsm_kernel(z, log_s0, drift_dt, vol_dt, K, df, degree, n_blocks):
    """
    z : (n_steps, n_paths) gaussiennes, transformées EN PLACE en prix S (pas de
    copie : mémoire divisée par 2). Ligne t = date (t+1)·dt, accès contigu.

    Régression par équations normales : G c = b avec G_kl = Σ x^(k+l),
    b_k = Σ x^k · cash sur les trajectoires dans la monnaie. Les sommes sont
    calculées en parallèle par blocs (réductions sans conflit d'écriture),
    puis on résout un système (degree+1)×(degree+1) : coût O(n) par date,
    au lieu d'une SVD de la matrice n × (degree+1) avec lstsq.
    """
    n_steps, n = z.shape
    p = degree + 1
    # 1. Trajectoires : log-prix cumulés puis exponentielle, en place
    for t in range(n_steps):
        for i in prange(n):
            prev = log_s0 if t == 0 else z[t - 1, i]
            z[t, i] = prev + drift_dt + vol_dt * z[t, i]
    for t in range(n_steps):
        for i in prange(n):
            z[t, i] = math.exp(z[t, i])

    cash = np.empty(n)
    for i in prange(n):
        cash[i] = max(K - z[n_steps - 1, i], 0.0)

    block = (n + n_blocks - 1) // n_blocks
    mom = np.empty((n_blocks, 2 * p - 1))
    rhs = np.empty((n_blocks, p))
    cnt = np.empty(n_blocks, np.int64)
    G = np.empty((p, p))
    coef = np.zeros(p)

    # 2. Programmation dynamique rétrograde
    for t in range(n_steps - 2, -1, -1):
        for i in prange(n):
            cash[i] *= df
        for b in prange(n_blocks):                       # sommes partielles par bloc
            mom_b = np.zeros(2 * p - 1)
            rhs_b = np.zeros(p)
            c = 0
            for i in range(b * block, min(n, (b + 1) * block)):
                s = z[t, i]
                if s < K:
                    x = s / K                            # normalisation : bon conditionnement
                    xk = 1.0
                    for k in range(2 * p - 1):
                        mom_b[k] += xk
                        if k < p:
                            rhs_b[k] += xk * cash[i]
                        xk *= x
                    c += 1
            mom[b] = mom_b
            rhs[b] = rhs_b
            cnt[b] = c
        if cnt.sum() <= p:
            continue
        m_tot = mom.sum(axis=0)
        r_tot = rhs.sum(axis=0)
        for k in range(p):
            for l in range(p):
                G[k, l] = m_tot[k + l]
        coef[:] = np.linalg.solve(G, r_tot)
        for i in prange(n):                              # décision d'exercice
            s = z[t, i]
            if s < K:
                x = s / K
                cont = 0.0
                for k in range(p - 1, -1, -1):           # schéma de Horner
                    cont = cont * x + coef[k]
                if K - s > cont:
                    cash[i] = K - s
    for i in prange(n):
        cash[i] *= df
    return cash


def lsm_american_put(m: BlackScholes, K: float, T: float, n_steps: int = 50,
                     n_paths: int = 200_000, degree: int = 3, seed: int = 3) -> MCResult:
    """
    Programmation dynamique rétrograde ; espérance conditionnelle approchée par
    régression polynomiale sur les trajectoires dans la monnaie.
    (Estimateur légèrement biaisé bas — borne inférieure du vrai prix.)
    Boucle complète dans un noyau Numba parallèle (voir _lsm_kernel).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_steps, n_paths))
    cash = _lsm_kernel(z, math.log(m.s0), (m.r - m.q - 0.5 * m.sigma**2) * dt,
                       m.sigma * math.sqrt(dt), float(K), math.exp(-m.r * dt),
                       degree, 64)
    price = max(float(cash.mean()), K - m.s0)           # exercice immédiat possible
    return MCResult(price, float(cash.std(ddof=1) / math.sqrt(n_paths)), n_paths)


# --------------------------------------------------------------------------- #
# Démonstration rapide
# --------------------------------------------------------------------------- #
def _demo() -> None:
    m = BlackScholes(s0=100.0, r=0.05, sigma=0.2)
    print("S=K=100, r=5 %, sigma=20 %, T=1\n")
    print(f"Call européen exact      : {float(bs_price(m, 100, 1)):.6f}")
    print(f"Call européen MC         : {mc_european(m, 100, 1)}")
    print(f"Call européen QMC        : {rqmc_european(m, 100, 1)}")
    print(f"Asiatique arithm. (52 d.): {mc_asian_arithmetic(m, 100, 1)}")
    print(f"Put américain (LSM)      : {lsm_american_put(m, 100, 1)}")
    print("\nPour la démonstration complète : python bench.py")

