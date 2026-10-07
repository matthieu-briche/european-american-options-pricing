"""Fixtures et helpers partagés.

Philosophie des tests Monte Carlo :
  - seeds FIXES -> tests déterministes (jamais de test « flaky » en CI) ;
  - un estimateur est jugé correct si la référence exacte est à moins de
    N_SIGMA écarts-types (4σ : faux échec ~ 6e-5 si on changeait la seed) ;
  - on teste aussi la réduction de variance (std_error) et pas seulement le prix.
"""
import math
import sys
from pathlib import Path

# pricer.py / greeks.py importables quelle que soit la façon de lancer les tests
_SRC = str(Path(__file__).resolve().parents[1])  # dossier racine du projet
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)  # en 1re position : priorité sur tout autre module « pricer »

import pytest

from pricer import BlackScholes

N_SIGMA = 4.0


@pytest.fixture
def model() -> BlackScholes:
    return BlackScholes(s0=100.0, r=0.05, sigma=0.2)


@pytest.fixture
def model_div() -> BlackScholes:
    return BlackScholes(s0=100.0, r=0.03, sigma=0.3, q=0.02)


def assert_mc_close(res, ref: float, n_sigma: float = N_SIGMA, atol: float = 0.0) -> None:
    """|estimation - référence| <= n_sigma * std_error + atol."""
    err = abs(res.price - float(ref))
    tol = n_sigma * res.std_error + atol
    assert err <= tol, f"estimation={res.price:.6f} ref={float(ref):.6f} " \
                       f"écart={err:.2e} > tolérance={tol:.2e} (se={res.std_error:.2e})"


def fd(f, x: float, h: float) -> float:
    """Dérivée centrée."""
    return (f(x + h) - f(x - h)) / (2 * h)


def fd2(f, x: float, h: float) -> float:
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


def isclose(a, b, rel=1e-6, abs_=1e-8) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
