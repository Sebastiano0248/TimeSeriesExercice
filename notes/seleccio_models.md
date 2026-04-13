# Selecció de models per a l'exercici

**Problema:** predicció del preu de tancament d'una sola acció tecnològica.  
Sèrie univariant, dades diàries, molt sorollosa, comportament proper a un **random walk**.

---

## Models que val la pena aplicar

### Naïve — *imprescindible*
El benchmark de referència per a cotitzacions de borsa. Prediu que el preu de demà serà igual al d'avui (`ŷ = y_t`).

Les cotitzacions s'assemblen molt a un random walk i en finances el Naïve és notòriament difícil de superar. Si el model no el bat clarament, no aporta res.

### ETS / Holt — *bon baseline fort*
Holt (nivell + tendència) pot funcionar bé si la sèrie té creixement sostingut. Simple, ràpid i ofereix intervals de predicció naturals.

### ARIMA — *model estadístic principal*
Model estadístic natural per a una sèrie univariant amb tendència. Sobre els **retorns** (primera diferència del preu) la sèrie sol ser estacionària i modelable. Justificable, interpretable i exigit pel temari.

### GARCH — *especialment rellevant per a finances*
Les cotitzacions presenten **agrupació de volatilitat** (períodes de calma i períodes turbulents). El temari el destaca explícitament per a finances.

Si l'objectiu inclou modelar la incertesa de la predicció, GARCH és imprescindible. Es pot combinar amb ARIMA en un model **ARIMA-GARCH**.

### ML amb features (lags + arbre) — *senzill i competitiu*
Lags del preu/retorn com a regressors + XGBoost o Random Forest. Sovint sorprenentment competitiu, fàcil d'implementar i interpretar. Bon representant dels mètodes no estadístics sense la complexitat d'una xarxa recurrent.

### LSTM / GRU — *no estadístic natural*
Capturen no-linealitats i dependències temporals que ARIMA no pot. GRU és més lleuger. Amb dades diàries de diversos anys (yfinance pot donar 5-10 anys fàcilment) hi ha prou dades per entrenar.

> Cal validació temporal estricta i comparar sempre amb els benchmarks estadístics.

---

## Models que no val la pena aplicar

### SARIMA — *estacionalitat no justificada*
Requereix estacionalitat periòdica clara. Els preus diaris no tenen un patró setmanal o mensual estable. Si hi ha algun efecte dia-de-la-setmana, és feble i inconsistent. No cal forçar-ho.

### Filtre de Kalman — *cost alt, guany baix*
Útil per a dades faltants o components latents complexos. Per a cotitzacions diàries sense buits significatius, afegeix complexitat innecessària sense benefici clar.

### VAR / VARX / VECM — *fora d'abast*
Requereixen diverses sèries endògenes. L'enunciat és **univariant** (un sol ticker). Caldria incorporar altres sèries (índex de mercat, competidors) i justificar que estan disponibles al moment de predir. Sobredimensionat per a l'exercici.

### Transformers temporals — *overkill*
Necessiten molta dada i infraestructura per brillar. Per a una sèrie univariant diària, el guany sobre LSTM/GRU és empíricament incert i el cost de disseny és alt. El temari mateix avisa: *"no sempre superen TCN o LSTM"*.

### TCN / CNN temporal — *poc natural per a borsa*
Capturen patrons locals repetitius. Les cotitzacions no tenen motius locals estables que es repeteixin (no és una sèrie de sensors industrials). Menys natural que LSTM per a aquest domini.

### Models híbrids — *opcional i arriscat*
Interessant en teoria (ARIMA + LSTM sobre residuals), però afegeix complexitat de pipeline, risc de leakage entre etapes i dificultat de validació. Només val la pena si els models simples ja funcionen bé i es vol exprimir una mica més de rendiment.

---

## Resum

| | Model | Veredicte |
|---|---|---|
| ✅ | Naïve | Obligatori com a benchmark |
| ✅ | ETS / Holt | Bon segon benchmark |
| ✅ | ARIMA | Model estadístic principal |
| ✅ | GARCH | Molt rellevant per a finances |
| ✅ | ML + lags | Senzill i competitiu |
| ✅ | LSTM / GRU | No estadístic natural |
| ⚠️ | Híbrid | Opcional, si el temps ho permet |
| ❌ | SARIMA | Estacionalitat no justificada |
| ❌ | Filtre de Kalman | Complex sense benefici clar |
| ❌ | VAR / VECM | Univariant, fora d'abast |
| ❌ | Transformer | Overkill per a una sola sèrie |
| ❌ | TCN / CNN | Poc natural per a borsa |
