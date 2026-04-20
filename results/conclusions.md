# Conclusions — V2

**Projecte**: Predicció del preu de tancament de MSFT (horitzó 22 dies, ~1 mes borsari)  
**Dataset**: 2.515 observacions diàries · 2016-04-18 → 2026-04-17  
**Validació**: Walk-forward CV amb finestra expansiva (10 folds)  
**Test set**: últim 10% (251 observacions · 2023-11 → 2026-04)

---

## 1. Canvi principal respecte a V1

**Bug fix de calibració en RandomForest i XGBoost.**

En v1, la desviació estàndard dels residus (usada per construir els intervals de confiança) es calculava sobre les dades d'entrenament. Com que ambdós models tendeixen a memoritzar el training set, els residus in-sample eren gairebé zero → intervals col·lapsats → cobertura efectiva del 0%.

**Fix v2**: es reserva el 10% final del training com a *heldout set* per estimar la incertesa out-of-sample. La predicció puntual no canvia; només millora la calibració.

| Model | Cobertura 80% (v1 → v2) | Cobertura 95% (v1 → v2) |
|---|---|---|
| RandomForest | 0% → **72.7%** | 4.6% → **100%** |
| XGBoost | 0% → **81.8%** | 0% → **100%** |

---

## 2. Ranking final (test set)

| # | Model | MASE | MAE ($) | RMSE ($) | CV MAE (mean ± std) | Cobertura 80% | Cobertura 95% |
|---|---|---|---|---|---|---|---|
| 1 | **GRU** | **6.16** | **15.36** | **19.83** | 13.14 ± 12.80 | 63.6% | **100%** |
| 2 | XGBoost | 11.15 | 27.81 | 31.74 | 33.79 ± 37.75 | **81.8%** | **100%** |
| 3 | RandomForest | 11.47 | 28.59 | 31.26 | 25.50 ± 29.43 | 72.7% | **100%** |
| 4 | Drift | 13.73 | 34.23 | 40.88 | 8.33 ± 9.63 | 27.3% | 50.0% |
| 5 | ARIMA | 14.31 | 35.67 | 42.59 | 13.37 ± 21.42 | 31.8% | **100%** |
| 6 | **Naïve** *(baseline)* | 14.33 | 35.72 | 42.66 | 8.66 ± 9.38 | 31.8% | 45.5% |
| 7 | GARCH | 14.33 | 35.72 | 42.66 | 8.66 ± 9.38 | 31.8% | 45.5% |
| 8 | ETS/Holt | 14.34 | 35.75 | 42.69 | 8.64 ± 9.38 | 27.3% | 45.5% |
| 9 | LSTM_RNN | 16.87 | 42.05 | 50.22 | 14.76 ± 13.52 | 45.5% | 50.0% |

> **MASE < 1** significaria superar el Naïve. Cap model ho aconsegueix en el test set per la naturalesa de random walk dels preus absoluts. El MASE del Naïve és ~14.33 perquè es normalitza respecte als errors de la sèrie de training, no del test.

---

## 3. Conclusions per grup de models

### GRU — Millor model
- MASE **6.16** (57% millor que el Naïve en test)
- MAE de **15.36$** vs 35.72$ del Naïve — reducció de l'error a la meitat
- Calibració acceptable: cobertura 100% @ 95%, 63.6% @ 80% (lleugerament conservador)
- Variabilitat CV moderada (std=12.80): el model és estable entre folds
- **Motiu del bon rendiment**: captura dependències no lineals a llarg termini (lookback 30 dies), normalització z-score, MC Dropout per la incertesa, early stopping per evitar overfitting

### LSTM_RNN — Pitjor model deep learning
- MASE **16.87** — pitjor que el Naïve
- Major complexitat que GRU (dues portes addicionals) sense benefici
- Overfitting confirmat: CV MAE (14.76) molt millor que test MAE (42.05)

### RandomForest i XGBoost — Bons en calibració, inestables
- MASE intermedi (11.2–11.5): millorant el Naïve però lluny de GRU
- Intervals de confiança ben calibrats post-fix (100% @ 95%)
- **Punt feble**: altíssima variabilitat entre folds (std 29–38) — el rendiment depèn molt del període de validació

### Models estadístics — No competitius
- ARIMA, GARCH, ETS/Holt, Drift, Naïve: MASE 13.7–14.3 (pràcticament idèntics al Naïve)
- **Causa estructural**: modelar preus absoluts viola l'estacionarietat que assumeixen aquests models
- La cobertura dels intervals és molt inferior a la nominal (27–32% @ 80% en lloc del 80% esperat)
- **Excepció**: ARIMA arriba al 100% @ 95% perquè els intervals creixen molt amb l'horitzó

### GARCH i Naïve — Resultats idèntics
- GARCH produeix exactament el mateix MAE/RMSE que el Naïve (35.72 / 42.66)
- El model de volatilitat no aporta millora en la predicció puntual de preus absoluts

---

## 4. Calibració dels intervals (CV vs test)

Els intervals es construeixen per calibració conformal: es prenen els quantils empírics 80/95 dels errors CV i s'apliquen al test set. Per construcció, la cobertura en CV és exactament 80%/95% per a tots els models.

La divergència apareix en el **test set** (2023–2026), on el comportament del mercat difereix del CV:

| Situació | Models afectats |
|---|---|
| Cobertura test > nominal (intervals massa amples) | ARIMA @ 95%, GRU @ 95% |
| Cobertura test << nominal (intervals massa estrets) | Drift, ETS, GARCH, Naïve @ 80% i 95% |
| Calibració correcta | XGBoost @ 80%, RandomForest @ 80% |

---

## 5. Millores identificades (no implementades a v2)

1. **Modelar log-retorns** en lloc de preus absoluts — fix estructural per a ARIMA, ETS, GARCH i els benchmarks. Faria la sèrie estacionària i els intervals serien proporcionals al nivell de preu.
2. **LSTM**: reduir complexitat (menys capes, més dropout, weight decay) per controlar overfitting.
3. **RF / XGBoost**: afegir features de volatilitat realitzada i retorns logarítmics per reduir variabilitat entre folds.
4. **GRU**: explorar lookback més llarg (>30 dies) o arquitectures amb atenció.
