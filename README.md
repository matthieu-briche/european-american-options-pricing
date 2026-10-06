# options_pricing

Pricing et Grecques d'options optimisés CPU (NumPy + Numba), d'après
G. Pagès, *Numerical Probability* (Springer, 2018).

## Fichiers

| Fichier | Rôle | Bouton ▶ de VS Code |
|---|---|---|
| `pricer.py` | Prix : Black–Scholes, Monte Carlo, QMC, asiatique, américaine (LSM) | démo des prix |
| `greeks.py` | Grecques : formules fermées, pathwise, LR, différences finies, arbre CRR | démo des Grecques |
| `bench.py` | Benchmark complet des prix | |
| `bench_greeks.py` | Benchmark complet des Grecques | |
| `tests/` | 116 tests pytest | lance les tests du fichier |

## Installation

```bash
python -m pip install numpy scipy numba pytest pytest-cov hypothesis
```

## Utilisation

```python
from pricer import BlackScholes, mc_european
from greeks import mc_european_greeks

m = BlackScholes(s0=100, r=0.05, sigma=0.2)
print(mc_european(m, K=100, T=1))
print(mc_european_greeks(m, K=100, T=1)["delta"])
```

## Tests

```bash
python -m pytest              # 116 tests
python -m pytest -m "not slow"
```

## Validation (ajout)

| Fichier | Rôle |
|---|---|
| `dividendes.py` | Arbre CRR américain avec dividendes discrets (modèles « spot » et « escrowed ») |
| `valider_pricer.py` + `adaptateur_crr.py` | Prix et grecques de `crr_american` vs QuantLib, propriétés théoriques |
| `valider_lsm.py` | Put Longstaff–Schwartz vs QuantLib (biais, convergence, seeds) |
| `valider_dividendes.py` | `crr_american_div` vs QuantLib, prix et grecques, deux modèles |
| `valider_marche.py` | Test sur une vraie chaîne d'options (Yahoo Finance ou CSV) |
| `generer_chaine_test.py` | Chaîne synthétique pour tester `valider_marche.py` hors ligne |

```bash
python -m pip install QuantLib pandas matplotlib yfinance
python valider_pricer.py --module adaptateur_crr
python valider_dividendes.py
python valider_marche.py --ticker AAPL --taux 0.04      # taux du jour à renseigner
```
