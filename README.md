J'ai écrit un module de pricing, pricer.py, et un script de démo, bench.py. Les deux tournent et les résultats ont été comparés aux formules exactes. 
L'organisation suit les chapitres du livre de Pagès.

L'idée principale

Pour aller vite en CPU, il y a deux leviers, et le premier compte plus que le second :

Réduire la variance. L'erreur Monte Carlo vaut σ/√N. Diviser σ par 10 permet donc de diviser N, et le temps de calcul, par 100. Aucune optimisation de code ne fait gagner autant.

Réduire le coût de chaque trajectoire :

Vectorisation NumPy : aucune boucle Python sur les trajectoires.
Numba (@njit(parallel=True)) pour les boucles en temps des options path-dependent : le code est compilé et réparti sur tous les cœurs.
Calcul par blocs, avec une moyenne et une variance mises à jour au fil des blocs : la mémoire reste bornée et les données restent dans le cache du processeur.
Simulation exacte de Black–Scholes quand c'est possible, ce qui évite le biais du schéma d'Euler (chapitre 7).
Générateur aléatoire PCG64 avec SeedSequence : les résultats sont reproductibles et le générateur se découpe proprement entre processus.

Ce que contient le module

Fonction	Méthode	Chapitre

bs_price, bs_delta_vega	Formules fermées, calculées sur des tableaux de strikes et maturités	–
mc_european	Monte Carlo + antithétique + variable de contrôle S_T avec β optimal	2–3
rqmc_european	Quasi-Monte Carlo (suites de Sobol brouillées), avec un intervalle de confiance valide	4
mc_greeks_pathwise	Delta et Vega par la méthode pathwise (processus tangent)	2.2.4
mc_asian_arithmetic	Calcul Numba parallèle, avec l'asiatique géométrique comme variable de contrôle	3, 8
lsm_american_put	Longstaff–Schwartz (régression polynomiale)	12
Résultats mesurés (2 cœurs, S=K=100, r=5 %, σ=20 %, T=1)
Cas	Temps	Prix ± demi-largeur de l'IC à 95 %
Call : Python pur, 200 000 trajectoires	0,118 s	10,487
Call : NumPy brut, 2 M de trajectoires	0,064 s	10,4556 ± 0,020
Call : antithétique + variable de contrôle	0,047 s	10,4480 ± 0,0038
Call : quasi-Monte Carlo Sobol, environ 2 M de points	0,091 s	10,45055 ± 0,00006 (exact : 10,45058)
Asiatique : sans variable de contrôle	0,49 s	5,859 ± 0,022
Asiatique : avec variable de contrôle géométrique	0,48 s	5,8535 ± 0,0006
Put américain (LSM)	0,96 s	6,080 ± 0,032 (valeur de référence de la littérature ≈ 6,09)

À temps égal, la variable de contrôle de l'asiatique rend l'erreur environ 36 fois plus petite. Pour obtenir la même précision sans elle, il faudrait environ 1 300 fois plus de trajectoires. Le quasi-Monte Carlo est particulièrement efficace sur les problèmes de petite dimension.

Pour une utilisation « pro »
Valider systématiquement : chaque estimateur renvoie un MCResult avec son intervalle de confiance, et chaque méthode doit être comparée à une formule fermée avant d'être appliquée à un produit exotique.
Tests unitaires (pytest) : vérifier que le prix exact tombe dans l'IC à 95 %, et vérifier la parité call-put.
Profiler avant d'optimiser : line_profiler ou py-spy.
Au-delà d'un seul processus : concurrent.futures.ProcessPoolExecutor avec SeedSequence.spawn() pour plusieurs cœurs ou machines. Pour la volatilité locale avec schéma d'Euler, le multilevel Monte Carlo du chapitre 9 est l'étape suivante.

Pour lancer la démo : pip install numpy scipy numba, puis python bench.py. 
Le premier appel Numba compile le code ; cette compilation est ensuite mise en cache.

Pricer
PY 
Bench
PY 

Je vais ajouter un module greeks.py qui calcule Delta, Gamma, Vega, Rho et Theta pour les trois familles d'options (européenne, asiatique, américaine), avec les méthodes du chapitre 10 du livre, puis comparer les résultats à des références exactes.

Le nouveau module greeks.py calcule Delta, Gamma, Vega, Rho et Theta pour l'européenne, l'asiatique et l'américaine. Je l'ai fait tourner et chaque résultat est comparé à une valeur de référence.

Le principe qui fait gagner du CPU : une seule simulation donne le prix et toutes les Grecques en même temps. La méthode naïve, qui consiste à décaler un paramètre (« bump ») et à recalculer le prix, demande au moins 7 calculs de prix complets.

Les méthodes, selon l'option (chapitres 2.2 et 10 du livre)
Grecque	Méthode utilisée	Pourquoi
Delta, Vega, Rho, Theta	Pathwise : on dérive le payoff le long de chaque trajectoire	Variance très faible
Gamma	Mixte pathwise + rapport de vraisemblance : un poids Z/(σ√T) − 1 est appliqué au Delta pathwise	Le Delta d'un call est discontinu, on ne peut pas le dériver une seconde fois
Asiatique	Les mêmes estimateurs, calculés dans un noyau Numba parallèle (10 estimateurs par trajectoire, en un seul passage), plus une variable de contrôle géométrique sur chaque Grecque	Les Grecques exactes de l'asiatique géométrique se déduisent de sa formule fermée
Américaine (LSM)	Différences finies avec nombres aléatoires communs : la même graine (seed) pour le prix de base et pour les prix décalés	Méthode universelle, car la frontière d'exercice empêche un estimateur pathwise simple
Référence pour l'américaine	Arbre binomial CRR vectorisé, 2000 pas	Valeur déterministe pour valider
Résultats obtenus (S=K=100, r=5 %, σ=20 %, T=1)

Call européen, 2 M de trajectoires, 0,14 s pour tout le calcul :

	Monte Carlo ± IC95	Exact
Delta	0,63697 ± 0,00028	0,63683
Gamma	0,01874 ± 0,00005	0,01876
Vega	37,486 ± 0,091	37,524
Rho	53,251 ± 0,030	53,232
Theta	−6,411 ± 0,009	−6,414

Call asiatique, 52 dates de constatation, 0,52 s :

La variable de contrôle réduit l'intervalle de confiance d'un facteur 7 à 36 selon la Grecque. Pour le Vega, par exemple, on passe de ± 0,112 à ± 0,007.
Les résultats concordent avec les différences finies, qui prennent 4,6 s.

Put américain :

LSM avec différences finies : Delta −0,4105, Vega 37,49, Theta −2,23.
Arbre binomial : Delta −0,4111, Vega 37,49, Theta −2,24.
Le Gamma est le point faible : 0,0217 contre 0,0230. C'est un problème connu, car la décision d'exercice change d'une trajectoire à 
l'autre quand on décale S0. Dans ce cas, l'arbre ou une EDP restent plus précis.

Points à retenir pour une utilisation pro : 
Le Theta est annualisé. Divisez-le par 365 (ou 252) pour l'avoir par jour. Le Vega et le Rho sont donnés pour une variation de 1,00 : 
divisez par 100 pour avoir la sensibilité par point de volatilité ou de taux.
La graine fixe est obligatoire dans fd_greeks_crn. Sans elle, le bruit Monte Carlo ne se compense plus d'un prix à l'autre 
et les différences finies sont inutilisables.
L'estimateur LR du Gamma de l'asiatique place son poids sur le premier pas de temps. 
Sa variance augmente donc avec le nombre de dates de constatation ; la variable de contrôle compense largement cet effet ici.

Greeks
PY 
Bench greeks
PY 

ok peux tu me faire tous le test unitaires necessaires en python?

Je vais écrire une suite pytest complète pour pricer.py et greeks.py, puis la faire tourner jusqu'à ce que tous les tests passent.

La suite de tests est prête : 116 tests, tous verts, en environ 11 s, avec 100 % des lignes de pricer.py et greeks.py couvertes. 
Elle est dans options_pricing.zip, avec le reste du projet.

bash
pip install numpy scipy numba pytest pytest-cov hypothesis
pytest                                   # tous les tests (~11 s)
pytest -m "not slow"                     # sans les 2 tests lourds (~5 s)
NUMBA_DISABLE_JIT=1 pytest --cov=pricer --cov=greeks --cov-report=term-missing

Pour mesurer la couverture, il faut lancer les tests avec NUMBA_DISABLE_JIT=1. 
Sinon, l'outil ne voit pas l'intérieur des noyaux compilés par Numba et annonce environ 84 %.

Ce qui est testé

test_closed_form.py, les formules exactes :

valeurs de référence (Hull) et refus des paramètres invalides ;
parité call-put ;
calcul sur un tableau de strikes identique au calcul strike par strike ;
cas limites (strike nul, maturité nulle) ;
chaque Grecque fermée comparée à une dérivée numérique, pour le call et le put, sur 4 couples (K, T) ;
relations de parité entre les Grecques ;
avec Hypothesis, qui génère des centaines de paramètres au hasard : bornes de non-arbitrage, 
équation aux dérivées partielles de Black–Scholes (Θ + ½σ²S²Γ + (r−q)SΔ − rV = 0), signes et monotonie ;
l'asiatique géométrique avec une seule date redonne Black–Scholes, et sa parité est vérifiée.

test_monte_carlo.py, les prix Monte Carlo :

le résultat est sans biais pour les 4 combinaisons « antithétique / variable de contrôle » ;
chaque technique réduit bien la variance ;
l'erreur standard décroît en 1/√N ;
l'intervalle de confiance à 95 % contient le vrai prix environ 95 fois sur 100 tirages ;
même graine, même résultat ; graine différente, résultat différent ;
le noyau Numba de l'asiatique est comparé à une version NumPy simple, et l'inégalité arithmético-géométrique 
(moyenne arithmétique ≥ moyenne géométrique) est vérifiée sur chaque trajectoire ;
LSM est comparé à l'arbre binomial, et un put très dans la monnaie vaut sa valeur intrinsèque.

test_greeks_mc.py, les Grecques :

les 6 Grecques Monte Carlo européennes sont comparées aux formules fermées, sur 4 cas ;
sur chaque trajectoire, le Delta et le Vega pathwise du noyau sont comparés à une dérivée numérique ;
les estimateurs géométriques servant de variable de contrôle sont sans biais ;
les différences finies sur un pricer exact retrouvent les formules fermées ;
l'utilisation de la même graine dans les différences finies est indispensable : elle rend l'erreur sur le Gamma au moins 10 fois plus petite ;
l'arbre binomial converge, et un call américain sans dividende vaut l'européen ;
les Grecques LSM sont comparées à celles de l'arbre.
Choix de conception
Toutes les graines sont fixes, donc les tests donnent toujours le même résultat, y compris en intégration continue. 
Un estimateur Monte Carlo est accepté si l'écart à la référence est inférieur à 4 erreurs standard.
Deux défauts trouvés en écrivant les tests, tous deux dans les tests eux-mêmes et pas dans le pricer :
Avec une seule date, la variable de contrôle de l'asiatique est parfaite : l'erreur standard vaut 0 et l'estimateur donne le prix exact. Le test vérifie maintenant cette propriété.
Hypothesis a trouvé un cas très dans la monnaie où deux prix ne diffèrent que d'un arrondi machine (1e-14). Le test de monotonie accepte maintenant cet arrondi.
Les deux tests lourds (les Grecques par différences finies sur LSM et sur l'asiatique) sont marqués slow, pour pouvoir les sauter pendant le développement.
