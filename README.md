# TimeSeriesExercice

Exercici de predicció de sèries temporals per a l'assignatura *Advanced Machine Learning Techniques* del Màster en Ciència de Dades de la Universitat de Girona.

---

## Descripció de l'experiment

**Objectiu:** predir el preu de tancament diari de Microsoft (MSFT) a 22 passos vista (horitzó d'un mes borsari) i avaluar comparativament nou models de forecasting tant en precisió com en qualitat de la incertesa.

**Dades:** sèrie de preus diaris de MSFT del 2016-04-18 al 2026-04-16 obtinguda via `yfinance` (2.514 observacions). No s'aplica cap transformació a la sèrie base — es modela el preu de tancament directament.

**Partició temporal:**
- **CV data (90%):** 2.263 observacions per a entrenament i validació creuada
- **Test set (10%):** 251 observacions, reservades fins a l'avaluació final

**Validació:** walk-forward cross-validation amb finestra expansiva (10 folds). En cada fold s'entrena sobre tot l'historial disponible i es valida sobre el bloc de dades immediatament posterior. Això dóna 10 estimacions independents del MAE, molt més robustes que un únic tall train/val.

**Mètriques d'avaluació:**

| Mètrica | Interpretació |
|---|---|
| `test_mase` | Error normalitzat per la dificultat de la sèrie. MASE < 1 significa que el model bat el Naïve. |
| `test_mae` | Error absolut mitjà en unitats de preu ($). |
| `test_rmse` | Penalitza errors grans (sensible a outliers). |
| `cv_mae_mean ± std` | Estimació robusta del rendiment a partir dels 10 folds. |
| `test_coverage_80/95` | Proporció de valors reals que cauen dins l'interval de confiança. Ideal: 0.80 i 0.95. |

**Models implementats:**

| Model | Tipus | Justificació |
|---|---|---|
| Naïve | Benchmark | Referència mínima: prediu que demà = avui |
| Drift | Benchmark | Naïve amb tendència lineal constant |
| ETS / Holt | Estadístic | Suavitzat exponencial amb nivell + tendència |
| ARIMA | Estadístic | Model lineal per a sèries univariants; pmdarima tria l'ordre automàticament |
| GARCH | Estadístic | Capta l'agrupació de volatilitat, especialment rellevant per a finances |
| Random Forest | ML | Regressió sobre lags + features de finestra mòbil |
| XGBoost | ML | Mateixa estructura que RF, gradient boosting |
| LSTM | Deep Learning | Xarxa recurrent per a dependències a llarg termini |
| GRU | Deep Learning | Versió alleugerida del LSTM |

---

## Estructura del projecte

```
TimeSeriesExercice/
├── experiments/
│   ├── exploratory_data_analysis.ipynb   # Anàlisi exploratòria de la sèrie
│   ├── forecasting_pipeline.ipynb        # Pipeline reutilitzable (un model per execució)
│   ├── run_all_models.ipynb              # Executa la pipeline per a tots els models via papermill
│   ├── model_comparison.ipynb           # Comparació de resultats i ranking final
│   └── outputs/                         # Notebooks executats per papermill
│       ├── v1/                          # Outputs versió 1
│       └── v2/                          # Outputs versió 2
├── models/
│   ├── models_v1/                       # Models versió 1 (fix cobertura RF/XGBoost)
│   │   ├── naive.py
│   │   ├── drift.py
│   │   ├── ets.py
│   │   ├── arima.py
│   │   ├── garch.py
│   │   ├── random_forest.py             # resid_std out-of-sample (fix cobertura)
│   │   ├── xgboost_model.py             # resid_std out-of-sample + early stopping sempre actiu
│   │   ├── rnn.py
│   │   └── gru.py
│   ├── dummy.py                         # Plantilla d'interfície (no s'evalua)
│   ├── naive.py
│   ├── drift.py
│   ├── ets.py
│   ├── arima.py
│   ├── garch.py
│   ├── random_forest.py                 # Versió 2 (millores en curs)
│   ├── xgboost_model.py
│   ├── rnn.py
│   └── gru.py
├── results/                             # Generat per run_all_models.ipynb
│   ├── v1/                              # Resultats versió 1
│   │   └── <MODEL_NAME>/
│   │       ├── metrics.json
│   │       └── model.pkl
│   └── v2/                              # Resultats versió 2
│       └── <MODEL_NAME>/
│           ├── metrics.json
│           └── model.pkl
├── notes/
│   └── seleccio_models.md              # Justificació de la selecció de models
└── subject_content/                    # Materials del curs
```

**Contracte d'interfície dels models:** cada fitxer `.py` ha d'exposar `MODEL_NAME: str`, una funció `train(y_train, y_val=None) -> model` i un mètode `model.predict(h) -> dict` amb claus `mean`, `lower_80`, `upper_80`, `lower_95`, `upper_95`.

**Selecció de versió:** canvia `MODEL_VERSION = "v1"` → `"v2"` a `run_all_models.ipynb` i `model_comparison.ipynb` per alternar entre versions. La versió `"v1"` usa `models/models_v1/` i guarda a `results/v1/`; la `"v2"` usa `models/` i guarda a `results/v2/`.

---

## Versió 1 — Resultats originals

**Ranking final (ordenat per `rank_mitjà`, menor = millor):**

| Posició | Model | MASE test | MAE test | CV MAE | Cobertura 80% | Cobertura 95% |
|---|---|---|---|---|---|---|
| 1 | **Drift** | 13.73 | 34.23 | 8.33 ± 9.63 | 22.7% | 27.3% |
| 2 | GRU | 14.76 | 36.79 | 8.10 ± 5.74 | 36.4% | **72.7%** |
| 3 | GARCH | 14.33 | 35.72 | 8.66 ± 9.38 | 27.3% | 40.9% |
| 4 | ARIMA | 14.31 | 35.67 | 13.37 ± 21.42 | 22.7% | 27.3% |
| 5 | Naïve | 14.33 | 35.72 | 8.66 ± 9.38 | 22.7% | 27.3% |
| 6 | ETS/Holt | 14.34 | 35.75 | 8.64 ± 9.38 | 22.7% | 27.3% |
| 7 | LSTM | 24.74 | 61.69 | 15.66 ± 21.64 | 45.5% | 45.5% |
| 8 | RandomForest | 15.01 | 37.41 | **7.89 ± 9.07** | 0% | 4.6% |
| 9 | XGBoost | 19.57 | 48.79 | 12.21 ± 12.48 | 0% | 0% |

**Observacions clau:**
- Cap model supera MASE = 1: cap bat el Naïve en el test set. Això és esperat en sèries de preus (hipòtesi del mercat eficient).
- **GRU** és el millor en calibració d'incertesa (cobertura 95% del 72.7%), malgrat tenir un MASE lleugerament pitjor que Drift.
- **RandomForest i XGBoost** tenien cobertura 0% — un bug de calibració (vegeu Versió 2).
- La cobertura de tots els models és molt per sota de l'ideal en el test set, perquè el periode de test (2023–2026) conté una recuperació brusca del preu que cap model no havia vist als folds de CV.

---

## Versió 2 — Millores aplicades

### Bug corregit: col·lapse dels intervals de confiança a RF i XGBoost

**Causa arrel:** tant `random_forest.py` com `xgboost_model.py` calculaven `resid_std` sobre els **residus d'entrenament** (in-sample):

```python
# Codi bugat (v1)
resid_std = float(np.std(Y - model.predict(X)))
```

RandomForest i XGBoost tendeixen a memoritzar les dades d'entrenament, de manera que els residus in-sample són gairebé zero (`resid_std ≈ 0`). Amb una desviació estàndard nul·la, els intervals `mean ± z·std` col·lapsen a la pròpia predicció i cap valor real hi cau dins → cobertura 0%.

**Fix aplicat:** reservar internament el 10% final de les dades de train com a heldout per estimar `resid_std` de forma out-of-sample, de manera equivalent a com Drift i els models estadístics estimen la volatilitat:

```python
# Codi corregit (v2)
n_val   = max(N_LAGS + 1, int(len(values) * 0.10))
tr      = values[:-n_val]
heldout = values[-n_val:]

# ... entrena el model sobre `tr` ...

# Residus sobre heldout (el model no ha vist aquestes dades)
resid_std = float(np.std(Y_heldout - model.predict(X_heldout)))
```

A XGBoost, aquest heldout intern s'usa també per a l'early stopping (que en v1 quedava inactiu si no es passava `y_val` explícitament).

**Fitxers modificats:** [`models/models_v1/random_forest.py`](models/models_v1/random_forest.py), [`models/models_v1/xgboost_model.py`](models/models_v1/xgboost_model.py)

### Millores pendents (no implementades)

Les següents millores podrien incrementar tant la precisió com la calibració de tots els models:

- **Modelar log-retorns en lloc de preus absoluts** — fa la sèrie estacionària, elimina el biaix de nivell i millora els intervals de tots els models. És la transformació estàndard en finances.
- **ARIMA / ETS:** els intervals gaussians simètrics no escalen amb el nivell del preu. Modelar en retorns logarítmics i recompondre al final resoldria la infra-cobertura observada.
- **GARCH:** verificar la convergència dels paràmetres (omega, alpha, beta); provar EGARCH per capturar asimetria de la volatilitat.
- **LSTM:** presenta un CV MAE alt i alta desviació entre folds, indicant overfitting. Reduir la complexitat de la xarxa o augmentar la regularització (dropout, weight decay).
- **RF / XGBoost:** afegir features de retorn logarítmic i volatilitat realitzada en lloc de lags de preu absolut.

---

## Com reproduir els resultats

```bash
# 1. Crear l'entorn virtual i instal·lar dependències
python -m venv venv
source venv/bin/activate
pip install papermill yfinance statsmodels pmdarima arch scikit-learn xgboost torch jupyter

# 2. (Opcional) Explorar les dades
jupyter notebook experiments/exploratory_data_analysis.ipynb

# 3. Executar tots els models
jupyter notebook experiments/run_all_models.ipynb

# 4. Comparar resultats
jupyter notebook experiments/model_comparison.ipynb
```

Els resultats es generen a `results/<versió>/<MODEL_NAME>/metrics.json` (p.ex. `results/v1/Drift/metrics.json`).
