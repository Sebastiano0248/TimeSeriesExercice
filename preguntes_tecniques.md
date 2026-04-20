# Preguntes Tècniques — TimeSeriesExercice

Preguntes de revisió sobre decisions de disseny, teoria i resultats del projecte de predicció de sèries temporals (MSFT, horitzó h=22).

---

## Dades i Preprocessament

1. **Per què s'han modelat els preus absoluts en lloc dels log-retorns?** Quines conseqüències té aquesta decisió sobre els models estadístics com ARIMA i ETS?

2. **Quins problemes introdueix la no-estacionarietat en una sèrie de preus?** Com detectaries formalment si una sèrie és estacionària?

3. **Per què s'ha triat un horitzó h=22 i no h=1 o h=5?** Quins avantatges i desavantatges té un horitzó llarg en termes d'acumulació d'error?

4. **Quina és la diferència entre una finestra expansiva i una finestra lliscant en validació creuada temporal?** Per quina raó es va triar la finestra expansiva en aquest projecte?

5. **El conjunt de test cobreix el període 2023–2026, que inclou una recuperació des de mínims fins a màxims històrics. Com afecta això a la validesa de les mètriques de test?** Seria representatiu si els preus haguessin baixat durant aquest període?

### Respostes

**1.** Es van modelar preus absoluts per simplicitat i interpretabilitat directa en dòlars. La conseqüència principal és que la sèrie és **no-estacionària**: la mitjana i la variància creixen amb el temps (MSFT ha passat de ~$42 a ~$539). ARIMA requereix diferenciar (d=1) per eliminar la tendència, però fins i tot amb d=1 el model prediu essencialment el valor anterior. ETS/Holt assumeix tendència lineal local, però la tendència a 10 anys no és transferible als 22 dies següents. Amb log-retorns, la sèrie seria aproximadament estacionària i de variància estable, cosa que beneficiaria tots els models estadístics.

**2.** La no-estacionarietat implica que la mitjana i la variància canvien en el temps, invalidant les assumpcions dels models estadístics clàssics (autocorrelació constant, distribució estable). Un ARIMA(p,0,q) donaria prediccions clarament errònies. Per detectar-la formalment s'usen el test **ADF (Augmented Dickey-Fuller)** — hipòtesi nul·la: té arrel unitària (no és estacionària); si no es rebutja, s'ha de diferenciar — i el test **KPSS** — hipòtesi nul·la: és estacionària. La combinació dels dos dona major robustesa. En el cas de MSFT, l'ADF no es rebutja a cap nivell raonable, confirmant la no-estacionarietat.

**3.** h=22 correspon aproximadament a **un mes borsari** (~22 dies de mercat obert), que és el mínim horitzó rellevant per a decisions de cartera o gestió de risc. Horitzons curts (h=1, h=5) no permeten planificació financera significativa; horitzons llargs (h=60+) acumulen massa error. Els **desavantatges** d'un horitzó llarg inclouen: en predicció recursiva (autoregressive), l'error de cada pas es propaga al següent, fent que la incertesa creixi proporcional a √h en models Naïve/Drift i de forma no-lineal en models de ML/DL; a h=22 la predicció puntual s'acosta a la predicció incondicional (la mitjana de la distribució estacionària dels retorns).

**4.** Una **finestra expansiva** incrementa el conjunt d'entrenament a cada fold (fold k entrena sobre els primers k·chunk punts). Una **finestra lliscant** manté la mida d'entrenament fixa, descartant les dades més antigues. Es va triar la finestra expansiva perquè simula el comportament real en producció: un model desplegat sempre entrena amb tot l'historial disponible, i en sèries financeres el volum de dades sòl ser beneficiós. La finestra lliscant seria preferible si es creu que els règims antics no són informatius (p.ex., si el mercat ha canviat estructuralment).

**5.** El test (2023–2026) inclou una tendència alcista sostinguda des de mínims (~$220) fins a màxims (~$539), un règim **no representatiu de la majoria del període CV** (que és molt més variable). Això afavoreix els models amb biaix alcista (com Drift) i perjudica models que prediquen reversió. Si els preus haguessin baixat durant el test, els resultats s'invertirien: el GRU (que segueix la tendència) empitjoraria i el Naïve (que no extrapola) podria quedar millor posicionat. Les mètriques de test, per tant, capturen el rendiment en un règim alcista específic, no la qualitat general del model.

---

## Models Estadístics

6. **ARIMA assoleix un MASE de 14.31, pitjor que el Naïve. Explica per quin motiu un model tan establert no supera un simple random walk en preus borsaris.**

7. **GARCH(1,1) modela la variància condicional però dóna prediccions puntuals idèntiques al Naïve. Per què? Quin seria el cas d'ús correcte del GARCH?**

8. **ETS/Holt assumeix un component de tendència. Donat que MSFT ha passat de $42 a $539, per què el model no aprofita aquesta tendència per millorar les prediccions?**

9. **El model Drift extrapola la tendència total de la sèrie. Quins riscos té extrapolar la tendència de 10 anys per predir els pròxims 22 dies?**

10. **Per quina raó es van descartar SARIMA, VAR i Kalman Filter? Quin criteri es va fer servir per justificar l'exclusió de cada un?**

### Respostes

**6.** En preus borsaris, els increments diaris (retorns) presenten **autocorrelació pràcticament nul·la** (consistent amb la hipòtesi del mercat eficient en forma dèbil). ARIMA, en diferenciar la sèrie per fer-la estacionaria, prediu que l'increment esperat és la seva mitjana recent (~zero). El resultat és equivalent al Naïve però afegint complexitat (selecció de p, q). Si hi hagués autocorrelació real en la sèrie diferenciada, ARIMA podria aprofitar-la; com que no n'hi ha, el soroll del model empitjora respecte al Naïve pur. MASE=14.31 vs Naïve=14.33 confirma que ARIMA no aporta res rellevant però tampoc no és molt pitjor.

**7.** GARCH(1,1) modela la **variància condicional** σ²_t, no la mitjana. L'equació de la mitjana sol ser un procés simple (random walk o constant), de manera que la predicció puntual és idèntica al Naïve. Les prediccions de GARCH divergirien del Naïve si la component de la mitjana fos un AR/MA no-trivial. El cas d'ús correcte de GARCH és la **predicció de volatilitat**: quantificació del risc (VaR, CVaR), pricing d'opcions, dimensionament de posicions, i calibratge d'intervals de confiança proporcionals a la volatilitat esperada.

**8.** ETS/Holt estima la tendència com una **exponential smoothing de les diferències recents** (no la tendència global de 10 anys). Si el mercat ha estat relativament pla els últims mesos del conjunt d'entrenament, la tendència local estimada s'aproxima a zero. A més, una tendència de +$0.18/dia extrapolada 22 passos afegeix ~$4 a la predicció, que és marginal respecte a la MAE (~$35). El model és eficient per sèries amb tendències locals estables, però la tendència de MSFT és sorollosa i canvia de règim sovint.

**9.** El model Drift extrapola la taxa de canvi mitjana sobre **tot l'historial** (~10 anys). El risc principal és que aquest ritme de creixement (+$0.18/dia) no és sostenible indefinidament: no captura correccions, canvis de règim, ni el fet que a preus alts la mateixa pujada en dòlars representa un percentatge menor. A curt termini (22 dies), la tendència historica de 10 anys és massa sorollosa per ser predictiva: la variabilitat diària (~$5-$10) domina completament sobre el component de tendència extrapolat (~$4 en 22 dies).

**10.** **SARIMA**: Els preus diaris de MSFT no mostren patrons estacionals clars a escala diària/setmanal/mensual; els efectes de calendari (ex. "effect of month-end") existeixen però son febles i difícilment modelables amb una SARIMA estàndard, sense justificació empírica clara. **VAR**: Requereix variables addicionals (índexs, sector, etc.) que introduirien decisions de selecció de features i possibles problemes de cointegració; complicaria el pipeline sense garantia de millora. **Kalman Filter**: Requereix especificació explícita del model d'espai d'estats (matrius de transició i emissió), cosa que implica assumpcions fortes sobre la dinàmica del sistema; la seva implementació és significativament més complexa que els models ja inclosos.

---

## Machine Learning (RF i XGBoost)

11. **En la versió 1, RF i XGBoost tenien una cobertura del 0% als intervals de confiança. Explica exactament per quin motiu succeïa això i com es va solucionar a la versió 2.**

12. **RF i XGBoost utilitzen 28 lags com a features. Per quin motiu els lags en preus absoluts poden ser problemàtics? Quina alternativa hauria estat més robusta?**

13. **El Random Forest té una variança molt alta entre folds (σ = 29.43 en MAE). Què indica això sobre la fiabilitat del model per a ús real?**

14. **En la predicció recursiva (autoregressive), l'error s'acumula en cada pas. Com afecta aquest fenomen diferentment a RF vs GRU per a h=22 passos?**

15. **XGBoost utilitza early stopping sobre un conjunt de validació. Si aquest conjunt és el 10% final de les dades d'entrenament, quin risc de data leakage existeix en un context de sèrie temporal?**

### Respostes

**11.** A la versió 1, els thresholds de calibratge conformal es calculaven sobre els **errors del propi conjunt d'entrenament** (in-sample), no sobre errors out-of-fold. Això fa que els thresholds siguin molt petits (el model ajusta bé les dades amb les quals va ser entrenat), resultant en intervals extremadament estrets que mai cobreixen el test. A la versió 2 es va corregir: els thresholds es calculen exclusivament a partir dels **errors de predicció out-of-fold** (la porció de validació de cada fold que el model no havia vist), que reflecteixen l'error real de generalització i donen intervals calibrats correctament.

**12.** Els preus absoluts en lags consecutius (lag_1, lag_2, ..., lag_28) estan **gairebé perfectament correlacionats** entre si (correlació >0.99), ja que el preu es mou poc d'un dia a l'altre. Això causa multicolinealitat severa: els models aprenen efectivament "prediu el valor anterior" (Naïve), sense extreure cap senyal addicional. Una alternativa més robusta seria usar **log-retorns en lags** (diferències logarítmiques), que són aproximadament estacionaris i independents. Altra opció: incloure features derivades com volatilitat mòbil, volum, o indicadors tècnics.

**13.** Una desviació estàndard del MAE de 29.43 (sobre una mitjana de 25.50) indica que el rendiment del model **fluctua enormement entre folds**: alguns folds donen MAE baix (el model generalitza), altres molt alt (el model sobreajusta). En ús real, això significa que no podem confiar en el model: en algunes finestres temporals funcionarà raonablement però en d'altres fallarà greument. Un model útil en producció hauria de tenir una variança molt menor (σ/μ < 0.3 és una regla d'or habitual).

**14.** En predicció recursiva a h=22, cada pas usa la predicció anterior com a input. En **RF**: cada predicció és independent donat l'input; l'error es propaga perquè el preu erroni s'usa com a lag, però el forest no té memòria interna; els arbres tendeixen a la mitjana de la distribució d'entrenament, fent que les prediccions llunyanes convergeixin a la mitjana i perdin variabilitat. En **GRU**: l'error es propaga tant per l'input com per l'**estat ocult** (hidden state), que "recorda" la seqüència; les portes (reset/update) ajuden a amortigar errors excessius, però si l'error s'acumula en l'estat ocult l'efecte pot ser major. En la pràctica, GRU mostra millor comportament a h=22 perquè la regularització via dropout i el gradient clipping limiten la propagació d'errors.

**15.** Si el conjunt de validació per a l'early stopping és el 10% temporal final del fold d'entrenament, aquest conjunt és **immediament anterior al fold de test dins del CV**. El model pot sobreajustar-se implícitament a les condicions d'aquest subperiode, que és justament el que precedeix el que es vol predir. Resultat: el model para d'entrenar en el punt òptim per a aquella distribució temporal específica, no per a la distribució general. Solució: fer servir un **fold CV intern** independent (nested CV) o seleccionar les dades de validació d'early stopping de manera aleatòria dins del conjunt d'entrenament (acceptable en aquest context perquè XGBoost no usa lags seqüencials en el sentit d'un RNN).

---

## Deep Learning (LSTM i GRU)

16. **GRU supera LSTM amb un MASE de 6.16 vs 16.87. Explica les diferències arquitecturals entre LSTM i GRU. Per quina raó la simplicitat del GRU pot ser un avantatge en sèries financeres?**

17. **LSTM té una CV MAE de 14.76 però una test MAE de 42.05. Quin fenomen explica aquest gap tan gran? Com podries detectar-ho durant el desenvolupament abans d'arribar al test?**

18. **Explica el funcionament de MC Dropout com a mètode per quantificar incertesa. Quines hipòtesis fa implícitament? Quin tipus d'incertesa captura (aleatòria vs epistèmica)?**

19. **Per què es normalitzen les entrades dels models neurals amb z-score? Quines conseqüències tindria no normalitzar-les en una sèrie amb tendència forta com MSFT?**

20. **El context window dels RNNs és de 30 dies. Com podries determinar la longitud òptima del context? Quins experiments faries?**

### Respostes

**16.** **LSTM** té 3 portes (input, forget, output) i 2 estats interns (hidden state h_t i cell state c_t). La cell state actua com a "memòria de llarg termini" explícita. **GRU** té 2 portes (reset, update) i només 1 estat (h_t), fusionant la cell i hidden state. Conseqüència: GRU té ~2/3 dels paràmetres d'LSTM de la mateixa mida. En sèries financeres on el senyal és feble i la relació entrada-sortida és gairebé un random walk, la complexitat addicional d'LSTM no aporta expressivitat útil però sí augmenta el risc d'overfitting. GRU, amb menys paràmetres i regularització equivalent (dropout=0.2), generalitza millor al test (MASE 6.16 vs 16.87).

**17.** Amb CV MAE=14.76 i test MAE=42.05 hi ha un **overfitting al règim temporal del CV**. El model memoritza patrons de 2016–2023 (incloent la pandèmia, la pujada post-COVID) que no es repeteixen igual a 2023–2026 (tendència ATH continuada). Per detectar-ho durant el desenvolupament: (1) monitorar les corbes train/val loss — si la val loss es dispara mentre la train continua baixant, el model sobreajusta; (2) comparar les distribucions d'error entre folds cronològicament ordenats — si els folds recents són molt pitjors que els antics, hi ha distributional shift; (3) fer servir més dropout o weight decay i veure si la divergència CV/val es redueix.

**18.** **MC Dropout** deixa el dropout actiu durant la inferència i executa 300 forward passes amb la mateixa entrada, obtenint una distribució de prediccions. La variança d'aquesta distribució aproxima la incertesa del model. Hipòtesis implícites: (1) el dropout com a aproximació variacional a una distribució posterior (Gal & Ghahramani, 2016); (2) les sortides segueixen una distribució aproximadament gaussiana. **Tipus d'incertesa**: captura principalment **incertesa epistèmica** (incertesa sobre els pesos del model, que es reduiria amb més dades), no **aleatòria** (soroll irreductible de la sèrie). En pràctica per preus borsaris, la major part de la incertesa real és aleatòria (volatilitat del mercat), que MC Dropout no captura.

**19.** Sense z-score (mean=0, std=1), les entrades serien valors com $200–$540. Els gradients durant backpropagation serien enormes, causant **gradient explosion** o aprenentatge molt lent. La funció de pèrdua (MSE) estaria dominada per l'escala dels preus en lloc de per els patrons de la sèrie. A més, els pesos inicials (normalment ~N(0,0.1)) serien completament inadequats per als inputs d'aquesta escala. La normalització fa que tots els inputs estiguin en el rang [-3, 3] aproximadament, estabilitzant l'optimització i permetent que el GRU detecti patrons relatius de la sèrie.

**20.** Per determinar la longitud òptima del context faria un **grid search de SEQ_LEN ∈ {5, 10, 15, 20, 30, 60, 90, 120}** avaluant la CV MAE en el walk-forward complet. Complementàriament: (1) calcular el **ACF/PACF** dels log-retorns per identificar lags amb autocorrelació significativa; (2) fer servir **mutual information** entre el preu en t i el preu en t-k per detectar dependències no lineals; (3) analitzar si el guany marginal de MAE és significant a partir d'un cert SEQ_LEN (si SEQ_LEN=30 i SEQ_LEN=60 donen el mateix MAE, 30 és suficient). En sèries financeres, la informació predicitva sol ser poca més enllà de 5–10 dies, però dependències de volatilitat poden arribar a 20–30 dies.

---

## Quantificació d'Incertesa i Intervals de Confiança

21. **Explica el mètode de calibratge conformal que s'ha fet servir per construir els intervals de confiança. Per quin motiu la cobertura en CV és exactament la nominal però divergeix en el test?**

22. **El model ARIMA assoleix una cobertura del 100% al nivell 95% en el test. Sembla un bon resultat, però podria ser un problema. Explica per quina raó intervals massa amples no indiquen un bon model d'incertesa.**

23. **GRU té una cobertura del 63.6% al nivell 80% (per sota de l'objectiu). Naïve té una cobertura del 31.8%. Quin dels dos té els intervals més informatius? Com mesuraries la qualitat d'un interval més enllà de la cobertura?**

24. **Naïve i GARCH tenen exactament la mateixa cobertura. Si GARCH modela explícitament la volatilitat, per quina raó no millora la cobertura?**

### Respostes

**21.** En calibratge conformal, s'acumulen els **errors absoluts out-of-fold** de tots els folds del CV (en total ~2.263 errors). Es calcula el quantil empíric al nivell desitjat (80è o 95è percentil) i s'aplica com a semipanxell simètric a les prediccions del test. La cobertura al CV és exactament nominal **per construcció**: si agafem el 80è percentil dels errors, exactament el 80% d'aquells errors estan per sota del llindar. Al test divergeix perquè el llindar es va calibrar amb la distribució d'errors del CV (2016–2023), però el test (2023–2026, recuperació a ATH) té una volatilitat diferent: els errors reals del test superen el llindar calibrat més sovint del previst.

**22.** Una cobertura del 100% al 95% significa que els intervals **sempre contenen el valor real**, cosa que s'aconsegueix fàcilment amb intervals molt amples (p. ex., ±$500). Un bon interval ha de ser el **més estret possible** mantenint la cobertura nominal. La cobertura excessiva d'ARIMA indica intervals sobredimensionats, que no aporten informació útil sobre la incertesa real. Mètriques de qualitat d'intervals més adequades: **Winkler Score** (amplada + penalització per miss), **interval sharpness** (amplada mitjana), o **CRPS (Continuous Ranked Probability Score)**, que mesura la qualitat conjunta de cobertura i precisió.

**23.** GRU (63.6%) té intervals **més informatius** que Naïve (31.8%), tot i no assolir l'objectiu del 80%. Intervals amb cobertura del 31.8% quan s'espera el 80% significa que els intervals de Naïve són massa estrets (molts misses). GRU té menys misses però intervals potser massa estrets també. Per mesurar qualitat més enllà de la cobertura: (1) **Winkler Score**: penalitza misses i intervals amples alhora; (2) **Interval width** (intervalles GRU haurien de ser més amples per assolir el 80%, però si no calen tant d'amples com Naïve per cobrir els valors, és informatiu); (3) **CRPS**: integra tota la funció de distribució predita vs el valor real; (4) **Calibration curve**: cobertura observada vs nominal a múltiples nivells.

**24.** La raó és que en aquest projecte els intervals de GARCH s'han construit amb el **mateix mètode de calibratge conformal** que la resta de models: s'aplica el quantil dels errors absoluts out-of-fold, sense usar la σ_t estimada per GARCH. El mètode conformal fa que la cobertura i l'amplada dels intervals siguin idèntiques a Naïve (que té els mateixos errors, ja que les prediccions puntuals són iguals). Per aprofitar realment el GARCH s'hauria d'usar la σ_t estimada per construir intervals adaptatius: en períodes d'alta volatilitat, l'interval s'eixamplaria automàticament; en períodes de baixa volatilitat, es tornaria més estret.

---

## Teoria de Sèries Temporals i Mercats Financers

25. **Ningún model aconsegueix un MASE < 1, és a dir, cap supera el Naïve de forma consistent. Com es relaciona aquest resultat amb la Hipòtesi del Mercat Eficient (EMH)? El GRU la refuta parcialment?**

26. **Quin és el "volatility clustering" i com s'observa en una sèrie de preus? Per quina raó és rellevant per al GARCH però no per als models de ML?**

27. **Explica la diferència entre predicció puntual i predicció probabilística. Per a quins casos d'ús reals (gestió de cartera, risc, trading) és més important cada una?**

28. **La mètrica MASE escala l'error per l'error del Naïve. Per quina raó és preferible al MAPE en sèries financeres? Quines limitacions té el MASE?**

29. **Si reentrenessis tots els models amb log-retorns en lloc de preus absoluts, quins models esperaries que millorin més? Raona la resposta model per model.**

30. **Un inversor utilitza el teu millor model (GRU) per prendre decisions de compra/venda. La predicció té un MAE de $15.36 sobre un preu mitjà de ~$400. Això és útil en la pràctica? Com caldria interpretar aquest error en context financer?**

### Respostes

**25.** Cap model amb MASE < 1 és consistent amb la **hipòtesi del mercat eficient en forma dèbil (weak-form EMH)**: si els preus incorporen tota la informació histórica, cap model basat en preus passats pot superar el random walk de forma sistemàtica. El GRU obté MASE=6.16 al test (millor que Naïve en termes absoluts de MAE), però això no refuta l'EMH: el test coincideix amb un període de tendència alcista molt pronunciada que el GRU pot seguir parcialment, però en un mercat lateral o bajista el GRU probablement empitjoraria. Per refutar parcialment l'EMH caldria mostrar retorns ajustats al risc estadísticament significatius en múltiples règims de mercat.

**26.** El **volatility clustering** és el fenomen per el qual períodes d'alta volatilitat tendeixen a ser seguits per períodes d'alta volatilitat, i viceversa ("la volatilitat és persistent"). S'observa a MSFT com a ràfegues de grans moviments diaris (p.ex., setmanes de ±3–5% per dia) seguides de períodes tranquils (±0.5–1% per dia). GARCH modela exactament això: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}, on σ²_t depèn dels xocs anteriors. En models de ML (RF, XGBoost), el target és el preu (o el retorn), no la volatilitat; el clustering de volatilitat no és una feature explícita, de manera que no s'aprofita directament. Podria incorporar-se afegint la volatilitat rolling com a input feature.

**27.** La **predicció puntual** dona un únic valor esperit (best guess). La **predicció probabilística** dona una distribució o interval sobre els possibles valors futurs. Per **gestió de risc** (VaR, stop-loss): es necesiten quantils de la distribució → predicció probabilística imprescindible. Per **pricing d'opcions**: es necessita la distribució de preus futurs (o volatilitat implicada) → probabilística. Per **trading direccional** (compra/venda): interessa la predicció de la direcció (positiu/negatiu del retorn) → puntual pot ser suficient. Per **gestió de cartera (optimització mean-variance)**: es necessiten retorns esperats (puntual) i matriu de covariàncies → combinació.

**28.** El **MAPE** (Mean Absolute Percentage Error) és problemàtic en sèries financeres perquè: els preus mai arriben a zero (divisió per zero), però en preus baixos els percentatges es disparen de manera no informativa; a més, és asimètric (errors cap amunt pesen diferent que cap avall). El **MASE** normalitza per l'error del Naïve (el millor benchmark trivial), fent-lo agnòstic a l'escala. **Limitacions del MASE**: si el Naïve és molt bo (MASE ~1 per a qualsevol model), discrimina poc entre models; si el Naïve és molt dolent, infla artificialment la qualitat percebuda; per sèries intermitents (zeros), el Naïve pot tenir MAE=0 fent MASE indefinit.

**29.** Amb log-retorns: **ARIMA** milloraria més (la sèrie de retorns és aproximadament estacionaria, eliminant la necessitat de diferenciar i permetent que ARIMA capturi autocorrelació real si n'hi hagués). **ETS/Holt** milloraria moderadament (la tendència en retorns és propera a zero, però la sèrie seria estacionaria). **RF/XGBoost** millorarien (menys multicolinealitat entre lags, features més informatives), però el guany seria menor que en models estadístics ja que ja funcionen raonablement. **GRU/LSTM** millorarien moderadament (inputs estacionaris faciliten l'aprenentatge, intervals de confiança serien proporcionals al risc en lloc d'absoluts). **Drift** empitjoraria (en retorns, la tendència és quasi-zero, resultant en una predicció de retorn nul equivalent al Naïve de retorns).

**30.** Un MAE de $15.36 sobre ~$400 representa un **error relatiu del ~3.8%** en un horitzó de 22 dies. Per context: la volatilitat diària de MSFT sol ser ~1.5%, que acumulada a 22 dies dona ~7% de volatilitat esperada. El MAE del GRU (~3.8%) és millor que el soroll de mercat (~7%), però té una trampa: el MAE mesura l'error en el **nivell de preu**, no en la **direcció**. Un inversor necessita saber "puja o baixa?", no "a quin preu exactament?". Si el model prediu $420 quan el preu real és $405, l'error és $15 però la **direcció** pot ser correcta (o no). En pràctica financera, el que importa és la **precisió direccional** (accuracy de la predicció del signe del retorn) i el **Sharpe ratio** de la estratègia resultant, no el MAE.

---

## Decisions d'Enginyeria i Pipeline

31. **El pipeline es basa en `papermill` per executar notebooks parametritzats. Quins avantatges té respecte a scripts Python en termes de reproducibilitat i auditoria de resultats?**

32. **Cada model exposa la mateixa interfície: `train(y_train, y_val)` i `predict(h)`. Quin principi de disseny de software aplica? Quins avantatges té per afegir nous models en el futur?**

33. **Els resultats es desen com a `metrics.json` i fitxers `.pkl`. Quines limitacions té desar models com a pickles? Quines alternatives existeixen per a producció?**

34. **La partició és 90% CV / 10% test. Donat que el test conté 251 observacions, és suficient per obtenir estimacions robustes de les mètriques? Com afecta la mida del test a la variança de les estimacions?**

### Respostes

**31.** Papermill parametritza un notebook Jupyter (injectant variables com MODEL_PATH) i l'executa generant un **output notebook amb totes les cel·les evaluades, gràfics inclosos**. Avantatges sobre scripts: (1) **Auditoria completa**: l'output notebook és un artefacte executable que mostra exactament com es van obtenir els resultats (no caldrà "tornar a córrer per verificar"); (2) **Reproducibilitat**: els paràmetres injectats queden registrats a la cel·la d'entrada del notebook; (3) **Visualització integrada**: gràfics i taules queden guardat dins el propi output, no en fitxers separats; (4) **Detecció d'errors**: si el notebook falla en qualsevol cel·la, l'output mostra exactament on i per quina raó.

**32.** Aplica el **principi d'interfície / Strategy Pattern**: cada model implementa el mateix contracte (`train(y_train, y_val)` → retorna objecte; objecte té `predict(h)` → retorna dict estàndard). El pipeline (`forecasting_pipeline.ipynb`) no necessita conèixer cap detall intern del model. Avantatges: (1) afegir un model nou és copiar un fitxer i implementar dos mètodes; (2) el pipeline no canvia mai quan s'afegeix un nou model; (3) els models poden testarse de forma aïllada; (4) permet execució en paral·lel sense conflictes (cada model és independent).

**33.** Limitacions dels pickles: (1) **Seguretat**: deserialitzar un pickle d'origen desconegut permet execució de codi arbitrari; (2) **Compatibilitat**: incompatible entre versions de Python i de les biblioteques (un pickle de sklearn 1.3 pot fallar amb sklearn 1.4); (3) **Difícil versionat**: no és humanament llegible, git mostra un diff binari no útil. Alternatives per a producció: **ONNX** per a models neurals (GRU, LSTM); **joblib** per a models scikit-learn (RF); **PMML** o serialització JSON custom per a models estadístics (ARIMA, ETS); **MLflow** o **BentoML** com a capa d'abstracció que gestiona formats i versions automàticament.

**34.** 251 observacions cobreix ~11 finestres no-solapades de h=22 passos (251/22 ≈ 11.4), és a dir, efectivament **~11 prediccions independents**. Amb n=11 punts la variança de les estimacions de MAE és elevada: l'interval de confiança d'un estimador construït sobre ~11 observacions pot ser de ±30–50% del valor central. Formalment, si els errors seguissin una distribució normal, l'error estàndard de la mitjana seria σ/√11 ≈ 0.3σ. Per obtenir estimacions robustes de forma general s'haurien de tenir com a mínim 30–50 finestres no-solapades (>660 observacions de test). En aquest projecte, les mètriques del test s'han d'interpretar amb precaució: reflecteixen el rendiment en un únic règim de mercat amb poca potència estadística per discriminar models propers.
