"""
greeks.py — Sensibilités (Grecques) optimisées CPU.

Méthodes (G. Pagès, *Numerical Probability*, Ch. 2.2 et Ch. 10) :
  - Formules fermées Black–Scholes (référence, vectorisées).
  - Méthode PATHWISE (processus tangent) : on dérive le payoff le long de la
    trajectoire. Faible variance, valable pour Delta, Vega, Rho, Theta
    (payoff lipschitzien).
  - Méthode du RAPPORT DE VRAISEMBLANCE (log-likelihood) : on dérive la densité,
    pas le payoff -> marche pour les payoffs discontinus. Utilisée ici en
    estimateur MIXTE pathwise-LR pour le Gamma (le Delta pathwise d'un call
    est discontinu, il ne se redérive pas trajectoire par trajectoire).
  - Différences finies à NOMBRES ALÉATOIRES COMMUNS (CRN) : méthode
    universelle (options américaines / LSM), même seed pour prix de base et bumpés.
  - Arbre binomial CRR vectorisé : référence déterministe pour l'américaine.

Optimisation CPU :
  - UNE seule simulation donne le prix ET toutes les Grecques (pas de re-pricing).
  - Noyau Numba parallèle pour l'asiatique : 10 estimateurs par trajectoire en
    une passe, en registres.
  - Variables de contrôle aussi sur les Grecques (asiatique géométrique, dont
    les Grecques exactes s'obtiennent par dérivation de la formule fermée).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import numpy as np
from numba import njit, prange
from scipy.special import ndtr

from pricer import (BlackScholes, MCResult, OptionType, _RunningStats,
                    geometric_asian_price)

GREEKS = ("price", "delta", "gamma", "vega", "rho", "theta")
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# 1. Formules fermées Black–Scholes (vectorisées sur K, T)
# --------------------------------------------------------------------------- #
def bs_greeks(m: BlackScholes, K, T, kind: OptionType = "call") -> dict[str, np.ndarray]:
    K, T = np.asarray(K, float), np.asarray(T, float)
    sqT = np.sqrt(T)
    d1 = (np.log(m.s0 / K) + (m.r - m.q + 0.5 * m.sigma**2) * T) / (m.sigma * sqT)
    d2 = d1 - m.sigma * sqT
    dfr, dfq = np.exp(-m.r * T), np.exp(-m.q * T)
    pdf1 = _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)
    gamma = dfq * pdf1 / (m.s0 * m.sigma * sqT)
    vega = m.s0 * dfq * pdf1 * sqT
    decay = -m.s0 * dfq * pdf1 * m.sigma / (2.0 * sqT)
    if kind == "call":
        price = m.s0 * dfq * ndtr(d1) - K * dfr * ndtr(d2)
        delta = dfq * ndtr(d1)
        rho = K * T * dfr * ndtr(d2)
        theta = decay - m.r * K * dfr * ndtr(d2) + m.q * m.s0 * dfq * ndtr(d1)
    else:
        price = K * dfr * ndtr(-d2) - m.s0 * dfq * ndtr(-d1)
        delta = dfq * (ndtr(d1) - 1.0)
        rho = -K * T * dfr * ndtr(-d2)
        theta = decay + m.r * K * dfr * ndtr(-d2) - m.q * m.s0 * dfq * ndtr(-d1)
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)


# --------------------------------------------------------------------------- #
# 2. Européenne Monte Carlo : toutes les Grecques en une simulation
# --------------------------------------------------------------------------- #
def mc_european_greeks(m: BlackScholes, K: float, T: float, kind: OptionType = "call",
                       n_paths: int = 2_000_000, chunk: int = 2**18,
                       seed: int = 11) -> dict[str, MCResult]:
    """
    Estimateurs par trajectoire (f = payoff, ind = 1{dans la monnaie}, s = ±1) :
      delta : e^{-rT} s ind S_T / S0                         (pathwise)
      gamma : e^{-rT} s ind S_T / S0² (Z/(σ√T) - 1)          (mixte pathwise-LR)
      vega  : e^{-rT} s ind S_T (√T Z - σT)                  (pathwise)
      rho   : -T e^{-rT} f + e^{-rT} s ind S_T T             (pathwise)
      theta : r e^{-rT} f - e^{-rT} s ind S_T (r-q-σ²/2 + σZ/(2√T))
    Variables antithétiques (Z, -Z) sur tous les estimateurs.
    """
    rng = np.random.default_rng(seed)
    sqT = math.sqrt(T)
    mu = m.r - m.q - 0.5 * m.sigma**2
    df = math.exp(-m.r * T)
    sgn = 1.0 if kind == "call" else -1.0
    stats = {g: _RunningStats() for g in GREEKS}

    def estimators(z: np.ndarray) -> dict[str, np.ndarray]:
        st = m.s0 * np.exp(mu * T + m.sigma * sqT * z)
        f = np.maximum(sgn * (st - K), 0.0)
        h = df * sgn * (f > 0) * st                    # e^{-rT} f'(S_T) S_T
        return dict(
            price=df * f,
            delta=h / m.s0,
            gamma=h / m.s0**2 * (z / (m.sigma * sqT) - 1.0),
            vega=h * (sqT * z - m.sigma * T),
            rho=-T * df * f + h * T,
            theta=m.r * df * f - h * (mu + m.sigma * z / (2.0 * sqT)),
        )

    done, n_draws = 0, n_paths // 2
    while done < n_draws:
        b = min(chunk, n_draws - done)
        z = rng.standard_normal(b)
        e1, e2 = estimators(z), estimators(-z)
        for g in GREEKS:
            stats[g].update(0.5 * (e1[g] + e2[g]))
        done += b
    return {g: _with_n(stats[g].result(), n_paths) for g in GREEKS}


def _with_n(r: MCResult, n: int) -> MCResult:
    return MCResult(r.price, r.std_error, n)


# --------------------------------------------------------------------------- #
# 3. Asiatique arithmétique : noyau Numba, 5 Grecques + contrôle géométrique
# --------------------------------------------------------------------------- #
@njit(parallel=True, fastmath=True, cache=True)
def _asian_greeks_kernel(z, s0, r, mu, sigma, dt, K, sgn, df):
    """
    Renvoie out[10, n] : lignes 0-4 = (prix, delta, gamma, vega, rho) arithmétique,
    lignes 5-9 = idem pour l'asiatique géométrique (variable de contrôle).
    Gamma mixte : le poids LR porte sur le 1er incrément Z_0 (seul lien avec S0).
    """
    n_paths, n = z.shape
    T = n * dt
    sqdt = math.sqrt(dt)
    out = np.empty((10, n_paths))
    for i in prange(n_paths):
        log_s = math.log(s0)
        w = 0.0
        a = 0.0      # somme S_tj
        a_sig = 0.0  # somme dS_tj/dsigma = S_tj (W_tj - sigma t_j)
        a_r = 0.0    # somme dS_tj/dr     = S_tj t_j
        lg = 0.0     # somme log S_tj
        g_sig = 0.0  # somme (W_tj - sigma t_j)
        g_r = 0.0    # somme t_j
        for j in range(n):
            t = (j + 1) * dt
            w += sqdt * z[i, j]
            log_s += mu * dt + sigma * sqdt * z[i, j]
            s = math.exp(log_s)
            a += s
            a_sig += s * (w - sigma * t)
            a_r += s * t
            lg += log_s
            g_sig += w - sigma * t
            g_r += t
        a /= n; a_sig /= n; a_r /= n
        g = math.exp(lg / n); gs = g * g_sig / n; gr = g * g_r / n
        lr = z[i, 0] / (sigma * sqdt) - 1.0
        # arithmétique
        fa = max(sgn * (a - K), 0.0)
        ia = df * sgn if fa > 0.0 else 0.0
        out[0, i] = df * fa
        out[1, i] = ia * a / s0
        out[2, i] = ia * a / (s0 * s0) * lr
        out[3, i] = ia * a_sig
        out[4, i] = -T * df * fa + ia * a_r
        # géométrique
        fg = max(sgn * (g - K), 0.0)
        ig = df * sgn if fg > 0.0 else 0.0
        out[5, i] = df * fg
        out[6, i] = ig * g / s0
        out[7, i] = ig * g / (s0 * s0) * lr
        out[8, i] = ig * gs
        out[9, i] = -T * df * fg + ig * gr
    return out


def geometric_asian_greeks(m: BlackScholes, K: float, T: float, n: int,
                           kind: OptionType = "call") -> np.ndarray:
    """Grecques EXACTES de l'asiatique géométrique (dérivées de la formule fermée)."""
    p = lambda mm: geometric_asian_price(mm, K, T, n, kind)
    hs, hv, hr = 1e-3 * m.s0, 1e-5, 1e-6
    up, dn = replace(m, s0=m.s0 + hs), replace(m, s0=m.s0 - hs)
    return np.array([
        p(m),
        (p(up) - p(dn)) / (2 * hs),
        (p(up) - 2 * p(m) + p(dn)) / hs**2,
        (p(replace(m, sigma=m.sigma + hv)) - p(replace(m, sigma=m.sigma - hv))) / (2 * hv),
        (p(replace(m, r=m.r + hr)) - p(replace(m, r=m.r - hr))) / (2 * hr),
    ])


def mc_asian_greeks(m: BlackScholes, K: float, T: float, n_steps: int = 52,
                    kind: OptionType = "call", n_paths: int = 500_000,
                    chunk: int = 2**16, control_variate: bool = True,
                    seed: int = 5) -> dict[str, MCResult]:
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    args = (m.s0, m.r, m.r - m.q - 0.5 * m.sigma**2, m.sigma, dt, K,
            1.0 if kind == "call" else -1.0, math.exp(-m.r * T))
    names = GREEKS[:5]
    exact_geo = geometric_asian_greeks(m, K, T, n_steps, kind)

    beta = np.zeros(5)
    if control_variate:                                # bloc pilote indépendant
        o = _asian_greeks_kernel(rng.standard_normal((20_000, n_steps)), *args)
        for k in range(5):
            c = np.cov(o[k], o[k + 5])
            beta[k] = c[0, 1] / c[1, 1]

    stats = [_RunningStats() for _ in names]
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        o = _asian_greeks_kernel(rng.standard_normal((b, n_steps)), *args)
        for k in range(5):
            stats[k].update(o[k] - beta[k] * (o[k + 5] - exact_geo[k]))
        done += b
    return {g: s.result() for g, s in zip(names, stats)}


# --------------------------------------------------------------------------- #
# 4. Méthode universelle : différences finies à nombres aléatoires communs
# --------------------------------------------------------------------------- #
PriceFn = Callable[[BlackScholes, float], float]


def fd_greeks_crn(price_fn: PriceFn, m: BlackScholes, T: float,
                  rel_ds: float = 0.01, d_sigma: float = 0.01,
                  d_r: float = 1e-3, d_t: float = 1 / 52) -> dict[str, float]:
    """
    price_fn(model, T) DOIT utiliser une seed fixe (CRN) : le bruit Monte Carlo
    s'annule alors en grande partie dans les différences. Différences centrées.
    Coût : 7 évaluations du pricer -> à réserver aux produits sans estimateur
    pathwise/LR simple (américaines, barrières, ...).
    """
    ds = rel_ds * m.s0
    p0 = price_fn(m, T)
    pu, pd = price_fn(replace(m, s0=m.s0 + ds), T), price_fn(replace(m, s0=m.s0 - ds), T)
    vega = (price_fn(replace(m, sigma=m.sigma + d_sigma), T)
            - price_fn(replace(m, sigma=m.sigma - d_sigma), T)) / (2 * d_sigma)
    rho = (price_fn(replace(m, r=m.r + d_r), T) - price_fn(replace(m, r=m.r - d_r), T)) / (2 * d_r)
    theta = -(price_fn(m, T + d_t) - price_fn(m, T - d_t)) / (2 * d_t)
    return dict(price=p0, delta=(pu - pd) / (2 * ds),
                gamma=(pu - 2 * p0 + pd) / ds**2, vega=vega, rho=rho, theta=theta)


# --------------------------------------------------------------------------- #
# 5. Référence américaine : arbre binomial CRR vectorisé
# --------------------------------------------------------------------------- #
def crr_american(m: BlackScholes, K: float, T: float, kind: OptionType = "put",
                 n: int = 2000) -> dict[str, float]:
    """Delta, Gamma, Theta lus sur les premiers nœuds ; Vega, Rho par re-pricing."""
    def tree(mm: BlackScholes):
        dt = T / n
        u = math.exp(mm.sigma * math.sqrt(dt)); d = 1.0 / u
        p = (math.exp((mm.r - mm.q) * dt) - d) / (u - d)
        disc = math.exp(-mm.r * dt)
        sgn = 1.0 if kind == "call" else -1.0
        j = np.arange(n + 1)
        v = np.maximum(sgn * (mm.s0 * u**j * d**(n - j) - K), 0.0)
        keep = {}
        for step in range(n - 1, -1, -1):
            j = np.arange(step + 1)
            s = mm.s0 * u**j * d**(step - j)
            v = np.maximum(disc * (p * v[1:] + (1 - p) * v[:-1]), sgn * (s - K))
            if step <= 2:
                keep[step] = (s, v.copy())
        return keep, dt

    k, dt = tree(m)
    (s1, v1), (s2, v2), (_, v0) = k[1], k[2], k[0]
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    gamma = ((v2[2] - v2[1]) / (s2[2] - s2[1]) - (v2[1] - v2[0]) / (s2[1] - s2[0])) \
        / (0.5 * (s2[2] - s2[0]))
    theta = (v2[1] - v0[0]) / (2 * dt)
    price = lambda mm: tree(mm)[0][0][1][0]
    vega = (price(replace(m, sigma=m.sigma + 1e-3)) - price(replace(m, sigma=m.sigma - 1e-3))) / 2e-3
    rho = (price(replace(m, r=m.r + 1e-4)) - price(replace(m, r=m.r - 1e-4))) / 2e-4
    return dict(price=v0[0], delta=delta, gamma=gamma, vega=vega, rho=rho, theta=theta)
