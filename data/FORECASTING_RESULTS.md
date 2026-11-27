# Historical expansion and first forecasting baselines

The 35 processed Sydney-local months from January 2023 through November 2025 contain **306,720** continuous five-minute intervals: 210,528 train and 96,192 validation. Monthly quality reports show zero missing AEMO/weather intervals, duplicates or invalid records. UTC keys are unique and continuous across month and daylight saving boundaries. April and October month lengths and the 36 repeated-hour `fold=1` rows match the Sydney clock. Prices retain **27,998** negative intervals and **782** intervals at or above $1/kWh; the range is **−$1.00 to $20.30/kWh**. Raw sources, URLs, retrieval metadata and checksums are in `raw/source_manifest.json`, with derived CSV checksums in each monthly summary. The original seven-day replay and financial benchmarks are unchanged.

The one-day Dec 31 2022 AEMO warm-up supplies 24-hour price and demand lags for Jan 1. The prior-week seasonal feature requires more history, so the first 1,716 training origins are excluded. A forecast target must remain within its origin's train/validation partition; twelve boundary-crossing train origins are excluded. `features/build_report.json` records counts and output hashes: **208,800** training and **96,180** validation feature rows. Both origin and target are checked against Jan 13–19 2025, leaving **94,152** scored validation forecasts. Targets are the wholesale price at the five-minute interval beginning exactly one hour after each decision. The full 41-feature inventory and timing assumptions are in [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md).

`python -m solaralpha.price_forecasting.run` rebuilds the deterministic compressed feature files, fits a **training-only** median, scores persistence and previous-day seasonal forecasts, and writes `features/baseline_metrics.json`, `features/validation_baseline_predictions.csv.gz` and four PNGs under `../plots/forecasting/`. It needs `matplotlib==3.11.2` from optional `requirements-forecast.txt`; ingestion, feature construction and baseline metrics use the standard library. This Matplotlib release supports Python 3.14; the workflow was run with Python 3.14.7.

| Forecast | Validation MAE | RMSE | Median absolute error | Negative-price recall | ≥$1/kWh spike recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Latest completed price (persistence) | $0.05719/kWh | $0.56104/kWh | $0.01655/kWh | 71.27% | 37.97% |
| Same target time previous day | $0.07669/kWh | $0.65276/kWh | $0.02766/kWh | 57.72% | 16.88% |
| Training median ($0.08353/kWh) | $0.07253/kWh | $0.48742/kWh | $0.03736/kWh | 0% | 0% |

There are **12,742** negative and **237** ≥$1/kWh actual targets among scored validation origins. Persistence best tracks ordinary price changes by MAE and median error, but can project a transient spike into the next hour, increasing RMSE. A constant median suppresses those forecast spikes and wins RMSE while missing every negative-price and spike event. Event precision, false positives, conditional MAE and bias are in the metrics JSON. The ordinary-day plot uses March 10 2025; the extreme-event day (October 10 2025) was chosen for illustration from validation as the day containing its largest absolute price. It is an exploratory plot choice, not model fitting. These are wholesale-price forecast metrics, not simulated household savings or tariff predictions.

The existing modeled household noise uses a seed reset at each monthly processing origin; the frozen replay seeded it at its seven-day origin. Therefore a given timestamp may receive different noise if the requested run starts elsewhere, and month boundaries restart the random sequence. A future range-independent option would derive a deterministic noise draw from a fixed seed and UTC timestamp (for example a cryptographic hash mapped to a normal variate), then apply the unchanged load equation. That would require a **versioned new dataset and separately labeled replay comparison**. This milestone leaves the existing noise and all prior fixtures unchanged.

The forecast features assume the last completed price and demand are published by the next decision, with zero further delay. Monthly AEMO files contain revised hindsight data and do not provide publication timestamps. This makes the validation scores optimistic relative to a live agent. No ERA5 ex-post weather, current solar or future realised data enters the predictors; the held-out December 2025–February 2026 period remains untouched.

## XGBoost V1: fixed one-hour price experiment

The first trained model reuses the **same 41 predictors and target** above. It fits on 208,800 January 2023–December 2024 rows. January–June 2025 contributes 50,100 eligible rows for early stopping; July–November 2025 contributes 44,040 evaluation rows. The 2,028 development-week origin/target rows remain excluded, as do twelve June origins whose one-hour target starts in July. The original train/validation boundary already excluded twelve December 2024 origins. No feature, target, weather or file from December 2025–February 2026 is opened.

XGBoost builds an additive prediction, `prediction = initial_value + tree_1(features) + ... + tree_B(features)`. Each new tree is fitted to reduce the remaining **squared-error** loss. Here the learning rate scales every tree's correction; depth and minimum child weight limit how finely trees partition historical conditions; row and feature subsampling reduce dependence on any one sample or predictor; L1/L2 penalties discourage large leaf values. The fixed configuration is `n_estimators=1000`, `max_depth=5`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `reg_alpha=0.1`, `reg_lambda=1.0`, `objective=reg:squarederror`, `eval_metric=mae`, `early_stopping_rounds=50`, `tree_method=hist`, `random_state=42`, and four CPU threads. Negative prices and extreme spikes remain in training targets without clipping or sample weighting.

Early stopping watches **January–June MAE only** while each tree is added. The best was iteration **60** (zero-based; 61 trees used for prediction), with tuning MAE **$0.06817/kWh**. Training ended after 111 rounds when the tuning MAE had failed to improve for 50 rounds. The July–November scores did not select the tree count or alter any hyperparameter. The training curve in `../plots/forecasting/xgb_v1/learning_curves.png` shows training MAE continuing downward while tuning MAE rises after the selected round.

| Forecast | Jan–Jun MAE | Jul–Nov MAE | Jul–Nov RMSE | Jul–Nov median absolute error | Jul–Nov mean bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost V1 | $0.06817 | $0.04324 | $0.28968 | $0.02780 | +$0.01310 |
| Persistence | $0.07226 | **$0.04005** | $0.38550 | **$0.01579** | +$0.00003 |
| Previous-day seasonal | $0.09762 | $0.05289 | $0.38633 | $0.02920 | +$0.00125 |
| Training constant median | $0.08304 | $0.06057 | $0.27744 | $0.04441 | −$0.00127 |
| Training target-hour-of-week median | $0.07337 | $0.04874 | **$0.27408** | $0.03370 | −$0.00147 |

All errors and bias are in **AUD/kWh** and use identical eligible rows within each period. The new hour-of-week median is fitted solely from 2023–2024 training targets, in 168 bins for Sydney target weekday and hour. This benchmark captures a weekly schedule but cannot react to current conditions. It has the lowest evaluation RMSE because constant-like forecasts avoid large false spikes. XGBoost beats persistence on tuning MAE, but **does not beat it on July–November MAE**. XGBoost's lower evaluation RMSE than persistence comes with larger median error and positive bias; it is not a uniform improvement.

| Jul–Nov event | Actual count | XGBoost true / false predicted | Persistence true / false predicted | XGBoost precision / recall / F1 | Persistence precision / recall / F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negative price (`< 0`) | 8,261 | 0 / 0 | 6,193 / 2,068 | undefined / 0 / 0 | 75.0% / 75.0% / 75.0% |
| Spike (`>= $1/kWh`) | 14 | 0 / 51 | 0 / 14 | 0 / 0 / 0 | 0 / 0 / 0 |

Precision is undefined when a model makes no positive predictions; the JSON records `null`, and F1 is reported as zero. On January–June tuning, XGBoost caught 56 of 223 spikes with 176 false positives, but still caught none of 4,481 negative prices. July–November has only 14 spikes. These event counts are small enough that a general spike-performance claim would be fragile. The fixed $1/kWh threshold has not been adjusted to improve V1.

Monthly evaluation MAE shows the same result in each month: XGBoost/persistence is **0.03374/0.03187** in July, **0.03689/0.03551** in August, **0.03711/0.02904** in September, **0.04313/0.03942** in October, and **0.06584/0.06485** in November. The highest XGBoost hourly MAE occurs at Sydney target hours **13:00 ($0.08249/kWh)**, **12:00 ($0.07682/kWh)** and **18:00 ($0.07083/kWh)**. Full monthly and 24-hour results for every benchmark are in `models/price_xgb_v1/results.json`.

To probe anticipation, we compared each prediction with the latest completed price. On 8,827 evaluation intervals whose realised one-hour change was at least $0.05/kWh in magnitude, XGBoost predicted the correct direction **78.6%** of the time, versus **79.5%** for the simple hour-of-week median. It did better than persistence on absolute error for such moves, but this does **not** establish advance knowledge of an event: a typical-price forecast can get the direction right through mean reversion. Among **2,068** newly negative prices and **14** new spikes that were not present in the latest completed reading, XGBoost predicted **zero** events at their thresholds. The October 10 spike plot shows some delayed or misplaced high forecasts. Of XGBoost's 51 false evaluation spike predictions, 18 followed a spike already visible in the completed-price history available at that decision; the other 33 were also false at the specified one-hour target. Thus V1 mostly captures ordinary price level and reversion, with no demonstrated reliable event anticipation.

Model fitting took **2.40 s** on this machine after data loading; batch inference of 44,040 evaluation rows took **0.019 s** (about 0.44 µs per row). These are local CPU timings, exclude CSV parsing and process startup, and are not live-service latency guarantees. The native JSON model, exact feature order, fixed configuration, package versions, split counts, data SHA-256 fingerprints, training-only medians, best iteration and runtimes are in `models/price_xgb_v1/`. Saved-model inference checks both feature order and model checksum. Prediction CSV.gz files and full metrics are saved beside them. The plot folder includes learning curves, predicted-versus-actual, residuals, gain importance, and two days each for normal, negative-price and spike conditions. Days were chosen from **realised evaluation prices only**, never model error; selection rules and dates are in `plots/forecasting/xgb_v1/selection.json`. Gain is computed from the 61 trees actually used for prediction and describes their average training loss reduction; correlated features can share or swap gain, so the plot is not causal evidence.

XGBoost 3.4.1 supplies a Python 3.14-compatible Windows x86-64 wheel. `requirements-ml.txt` pins XGBoost, SciPy, scikit-learn and their resolved transitives with the existing NumPy 2.5.5 and Matplotlib 3.11.2. The project `.venv` remained Python 3.14.7; no interpreter replacement was needed. The model remains offline and has **not** been connected to MPC, the simulator, or hardware. Its inputs still assume zero extra AEMO publication delay and can reflect revisions in monthly historical files.

## XGBoost V1 error audit and four controlled V2 variants

**Evaluation status:** July–November 2025 was already examined when V1 was reviewed. Every V2 comparison on those 44,040 rows is therefore **reused development evaluation**, not a fresh independent test. All four V2 specifications were fixed before this comparison and the variant selection rule was lowest MAE on the 50,100 eligible January–June tuning rows. The December 2025–February 2026 held-out summer data remains untouched. The 41 features, one-hour target, training rows, seed, tree parameters, development-week exclusions and split boundaries are unchanged. V1 model and results files remain untouched in `models/price_xgb_v1/`.

### Demonstrated V1 findings

`models/price_xgb_v2/v1_diagnostics.json` contains full quantiles, market conditions, price-range and monthly/hourly bias, feature gain, squared-error shares and learning-curve values for the frozen V1 model. The key evaluation distribution is:

| July–November AUD/kWh | Minimum | 1st percentile | Median | 95th percentile | 99th percentile | Maximum | Mean | Population SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actual | −0.99999 | −0.03055 | 0.07879 | 0.21962 | 0.29999 | 20.30 | 0.08480 | 0.27743 |
| V1 predicted | **+0.00509** | +0.01963 | 0.08748 | 0.18751 | 0.32271 | 7.92209 | 0.09790 | 0.10973 |

V1's evaluation predictions are all positive, even though **8,261 of 44,040 actual targets (18.76%)** are negative. Of the 8,261 decisions whose latest completed price is negative, V1 predicts a positive **$0.0393/kWh on average**. Its prediction correlates with the latest completed price (Pearson **0.632**), but the linear slope is only **0.250**: the model responds to the latest reading and pulls it strongly toward typical prices. It is not simply copying persistence. V1 made 14 negative predictions on January–June tuning, none correct, and none on training or the reused evaluation.

V1's overall evaluation bias is **+$0.01310/kWh**. It is positive in every evaluation month (about **+$0.010 to +$0.015/kWh**) and is largest at Sydney target hours **08:00 (+$0.0591)** and **07:00 (+$0.0424)**. Conditional on actual price, bias is **+$0.0634** for negative intervals and **+$0.0223** for prices from $0 to $0.10; it turns **−$0.0125** for $0.10–$0.30, **−$0.1369** for $0.30–$1, and **−$12.51** for the 14 spikes. This is shrinkage toward ordinary positive prices, with missed high extremes as well as missed negative values.

| Market slice | Negative targets | ≥$1/kWh targets | Mean latest regional demand | Actual price SD | Mean absolute consecutive 5-minute price change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023–2024 training | 7.07% (14,762) | 0.259% (541) | 7,488 MW | $0.529/kWh | $0.0292/kWh |
| Jan–Jun tuning | 8.94% (4,481) | 0.445% (223) | 7,598 MW | $0.614/kWh | $0.0329/kWh |
| Jul–Nov reused evaluation | **18.76% (8,261)** | **0.0318% (14)** | 7,489 MW | $0.277/kWh | $0.0179/kWh |

Thus evaluation has many more negative and far fewer spike intervals than training/tuning. Regional demand mean is similar to training but lower than tuning; price volatility is lower. These are observed distribution differences, not a causal explanation on their own. The 541 training spikes are only **0.259%** of rows, yet contribute **94.9%** of the selected V1 model's training squared error and **97.8%** of a constant-training-median reference's squared error. This establishes strong squared-error concentration in rare spikes; it does not prove that it alone caused V1's positive predictions.

The latest completed price has the largest **total** V1 split gain at **8.5%**; the top five features together account for **34.7%**. `local_fold` has zero gain. Several lags, rolling statistics and calendar features also contribute. Gain is not a causal or independent feature-necessity test, so there is no evidence that one feature alone dominates the model. The V1 training MAE was **0.05162** at its selected round 60 versus tuning **0.06817**. After 50 more rounds training MAE fell to **0.04902** while tuning rose to **0.06996**. That is evidence of overfitting **after** the selected round. The remaining event failures at the selected round do not by themselves diagnose underfitting.

**Working hypothesis tested next:** squared-error training and a positive typical-price anchor may make rare high spikes more influential and negative-price forecasts less likely. A different loss may improve ordinary and negative prices. The loss comparison below supports an effect of objective choice, while the changing market mix and correlated features prevent assigning a single exclusive cause.

### V2 design and mathematics

| Variant | Regression target | Built-in objective | Difference from V1 |
| --- | --- | --- | --- |
| V2-A | Future absolute price | `reg:absoluteerror` | L1 instead of squared loss |
| V2-B | Future absolute price | `reg:pseudohubererror` | Smooth robust loss; explicit default `huber_slope=1.0` AUD/kWh |
| V2-C | Future price minus latest completed price | `reg:squarederror` | Residual target |
| V2-D | Future price minus latest completed price | `reg:absoluteerror` | Residual target and L1 loss |

For a decision at `t`, let `p` be the price of `[t-5,t)` and `y` the actual future price of `[t+60,t+65)`. Residual variants fit `r=y-p` and reconstruct `y_hat=p+r_hat`. The **same completed-price feature value** is used in both operations. Hence `|y-y_hat| = |(y-p)-r_hat|`: XGBoost's `eval_metric=mae` on the residual label is the same absolute-price MAE used for early stopping and comparison, apart from floating-point rounding. Tests verify the identity and the installed library's reported early-stopping metric. V2-B uses XGBoost's documented default pseudo-Huber slope of 1.0 without tuning it. All variants retain the other V1 hyperparameters, including the 1,000-estimator cap, 50-round early stopping, and seed 42. Genuine negative and extreme targets remain unweighted and uncapped. The installed XGBoost 3.4.1 build successfully fitted all three objectives on a small signed-price example before the full runs.

### Matched results

The best V2 by **January–June tuning MAE** is **V2-A (0.05490/kWh)**. Its best iteration is 536 (zero-based, 537 trees used). The other best iterations are B 145, C 8, D 610. The reused July–November comparison is:

| Model | MAE | RMSE | Median absolute error | Mean bias | Negative precision / recall / F1 | Spike true / false predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 0.04324 | 0.28968 | 0.02780 | +0.01310 | undefined / 0 / 0 | 0 / 51 |
| **V2-A** | **0.02947** | **0.27039** | 0.01571 | −0.00511 | 90.3% / 46.0% / 61.0% | 0 / 0 |
| V2-B | 0.03254 | 0.27159 | 0.01832 | +0.00111 | 94.9% / 34.5% / 50.6% | 0 / 4 |
| V2-C | 0.03951 | 0.36147 | 0.01600 | +0.00195 | 77.1% / 72.7% / 74.8% | 0 / 14 |
| V2-D | 0.03278 | 0.35986 | **0.01513** | −0.00301 | 90.7% / 52.3% / 66.3% | 0 / 10 |
| Persistence | 0.04005 | 0.38550 | 0.01579 | +0.00003 | 75.0% / 75.0% / 75.0% | 0 / 14 |
| Previous-day seasonal | 0.05289 | 0.38633 | 0.02920 | +0.00125 | 65.3% / 64.3% / 64.8% | 0 / 14 |
| Training constant median | 0.06057 | 0.27744 | 0.04441 | −0.00127 | undefined / 0 / 0 | 0 / 0 |
| Training hour-of-week median | 0.04874 | 0.27408 | 0.03370 | −0.00147 | undefined / 0 / 0 | 0 / 0 |

All error/bias columns are AUD/kWh. The 14 actual July–November spikes were missed by **every** model. Spike precision and recall are zero for models making high predictions and precision is undefined for models making none; all spike F1 values are zero. `v2_results.json` records true/false counts and precision/recall/F1 for both events across **both** splits, with monthly, hourly, conditional-range and onset details. On tuning, V2-A caught zero of 223 spikes and made no high predictions, versus V1's 56 true and 176 false high predictions. This is a material tradeoff, not an across-the-board improvement.

V2-A predicted **3,804** of the 8,261 negative targets with **407** false negative-price signals. It correctly flagged **327** of the 2,068 newly negative intervals whose latest price was nonnegative; persistence flags none at their decision times. Conditional MAE for actual negative prices fell from **0.06338 (V1)** to **0.02540 (V2-A)**, versus **0.03463** for persistence. V2-A also improves MAE in the $0–$0.10 range (**0.01674** versus V1 **0.02598**) and in every evaluation month. It still misses spikes badly (MAE about **$12.53/kWh** on the 14 actual spikes). Its worst hourly MAE is at 18:00 (**0.05668/kWh**), with all 24 hourly figures in the results file.

The tuning improvement persists in January–March (**V2-A 0.03258** versus V1 **0.04289**) and April–June (**0.07523** versus **0.09119**). On the reused evaluation, V2-A scores **0.02433** versus V1 **0.03590** in July–September, and **0.03722** versus **0.05430** in October–November. These subperiods show that the MAE difference is not confined to one month; they are still exploratory development comparisons, not independent confirmation.

| Variant | Training fit time | Evaluation batch inference | Best iteration (zero-based) |
| --- | ---: | ---: | ---: |
| V2-A | 13.84 s | 0.099 s / 44,040 rows | 536 |
| V2-B | 4.48 s | 0.037 s / 44,040 rows | 145 |
| V2-C | 1.49 s | 0.008 s / 44,040 rows | 8 |
| V2-D | 15.30 s | 0.148 s / 44,040 rows | 610 |

These local batch timings exclude feature CSV loading, Python process startup and any live service overhead. V2-A's gain in MAE comes with more trees and slower inference than V1. `models/price_xgb_v2/` holds per-variant native JSON models, ordered-feature schemas, exact objectives/parameters, data fingerprints, runtimes and learning curves, plus matched tuning/evaluation prediction files and metrics. A fresh V2-A fit reproduced its model checksum and best iteration; reloaded inference from all four artifacts matches saved predictions. `../plots/forecasting/xgb_v2/` contains learning curves, prediction/residual distributions and two ordinary, two negative and two spike days. Those days reuse V1's previously chosen outcome-only dates. Six new tests cover residual algebra, the early-stopping metric, future-data leakage, saved inference and artifact isolation; **all 70 tests pass**.

The **demonstrated** result is better one-hour MAE and nonzero negative-price forecasts from direct L1 regression on these development periods. The proposed mechanism is reduced sensitivity of L1 updates to extreme targets, supported by V1's squared-error concentration and the controlled objective change. This experiment does **not** establish a deployable model, reliable spike forecasting, or financial benefit: AEMO publication-delay/revision assumptions remain, no MPC backtest was run, and the held-out summer test was not used.

## Alternative tabular models: Ridge, Lasso, Random Forest, LightGBM, CatBoost

This next offline comparison keeps the same 41 ordered decision-time features and the price of the five-minute interval starting exactly 60 minutes after decision time. It reads the frozen feature CSVs and V2 matched prediction files, not the held-out summer files. There are **208,800 training** (2023–2024), **50,100 tuning** (January–June 2025) and **44,040 reused development evaluation** (July–November 2025) origins. The development week and 12 June/July boundary origins remain excluded. Prices are AUD/kWh; genuine negatives and extreme spikes are untouched. Linear scaling is fitted only on training rows. Only January–June MAE chooses linear regularisation and stops boosting trees. July–November has been inspected in earlier experiments and is **not an independent test**.

### What the five architectures learn

| Model | Mathematics and learning | Difference from XGBoost; useful settings and limits |
| --- | --- | --- |
| Ridge | Linear forecast `b + Σ w_j z_j`; minimises `Σ(y−ŷ)² + αΣw_j²` on standardised `z`. A closed-form or numerical linear solver shrinks correlated coefficients. | No trees or nonlinear feature interactions unless already encoded. α controls shrinkage. Fast and compact, but squared loss gives large spikes disproportionate influence. |
| Lasso | Same linear forecast; minimises `Σ(y−ŷ)² + αΣ|w_j|`. Coordinate descent can drive coefficients exactly to zero. | Sparse, interpretable linear reference. α and convergence tolerance matter. Correlated lags can exchange which coefficient survives; squared residual loss is still spike-sensitive. |
| Random Forest | Fifty bootstrap decision trees choose squared-error-reducing splits; their numeric outputs are averaged. | Trees train independently rather than sequentially correcting errors. Depth 12, leaf minimum 20, feature fraction 0.7, seed 42, one worker. Squared-error splits chase extreme targets; averaging reduces variance but can miss rare events. This one-worker Windows run is substantially slower. |
| LightGBM | Sequential histogram-binned, leaf-wise gradient-boosted trees minimise direct L1 regression loss. | Unlike XGBoost's V2-A tree growth, LightGBM expands the leaf with largest gain; 31 leaves, learning rate 0.05, minimum leaf 30, up to 800 trees and 50-round tuning-MAE patience. Efficient, nonlinear, but leaf-wise growth can overfit and rare spikes remain difficult under MAE. |
| CatBoost | Sequential symmetric (oblivious) trees minimise direct MAE; each tree uses the same split at each level. | Symmetric trees differ structurally from XGBoost's trees; depth 6, learning rate 0.05, L2 leaf regularisation 3, up to 800 iterations, 50-round tuning-MAE patience. Compact and fast for this numeric dataset, but its categorical-feature advantages are not exercised here. |

Ridge tried α = 0.1, 1, 10 and selected **0.1** on tuning MAE. Lasso tried α = 0.0001, 0.001, 0.01 and selected **0.01**. Random Forest uses a single fixed configuration. LightGBM stopped at **267 trees**; CatBoost retained **288 trees** (zero-based best iteration 287). The installed LightGBM 4.7.0 accepted `regression_l1` with `l1` validation metric; CatBoost 1.2.10 accepted `MAE` loss and metric. Both have Python 3.14 Windows wheels. Versions and resolved transitive dependencies are pinned in `../requirements-ml.txt`; the Python environment was not replaced. Random Forest ran with one worker after Windows sandbox permission denied joblib's worker creation. That affects the reported runtime, not the number or settings of trees.

### Matched results

The table compares all 14 models on the identical 44,040 reused development rows. Error and bias units are AUD/kWh. Negative-event columns are interval precision/recall/F1. Spike TP/FP uses the frozen **≥$1/kWh** definition; there are 14 true spikes. Precision is undefined when a model makes no spike predictions (shown as `—` in the machine-readable file), and every model's spike recall and F1 are zero.

| Model | Tune MAE | Eval MAE | RMSE | Median AE | Bias | Negative P/R/F1 | Spike TP/FP |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| V1 | 0.06817 | 0.04324 | 0.28968 | 0.02780 | +0.01310 | 0/0/0 | 0/51 |
| V2-A | **0.05490** | 0.02947 | 0.27039 | 0.01571 | −0.00511 | .903/.460/.610 | 0/0 |
| V2-B | 0.05795 | 0.03254 | 0.27159 | 0.01832 | +0.00111 | .949/.345/.506 | 0/4 |
| V2-C | 0.07086 | 0.03951 | 0.36147 | 0.01600 | +0.00195 | .771/.727/.748 | 0/14 |
| V2-D | 0.06233 | 0.03278 | 0.35986 | 0.01513 | −0.00301 | .907/.523/.663 | 0/10 |
| Persistence | 0.07226 | 0.04005 | 0.38550 | 0.01579 | +0.00003 | .750/.750/.750 | 0/14 |
| Daily seasonal | 0.09762 | 0.05289 | 0.38633 | 0.02920 | +0.00125 | .653/.643/.648 | 0/14 |
| Training median | 0.08304 | 0.06057 | 0.27744 | 0.04441 | −0.00127 | 0/0/0 | 0/0 |
| Hour-of-week median | 0.07337 | 0.04874 | 0.27408 | 0.03370 | −0.00147 | 0/0/0 | 0/0 |
| Ridge | 0.07896 | 0.05004 | 0.28432 | 0.03327 | +0.01041 | .763/.527/.623 | 0/20 |
| Lasso | 0.07281 | 0.04411 | 0.28202 | 0.02757 | +0.01353 | .916/.343/.499 | 0/17 |
| Random Forest | 0.06890 | 0.03930 | 0.28469 | 0.01937 | +0.01082 | .934/.336/.494 | 0/65 |
| LightGBM | 0.05497 | 0.02954 | **0.27036** | 0.01571 | −0.00533 | .915/.474/.625 | 0/0 |
| CatBoost | 0.05524 | **0.02945** | 0.27037 | **0.01543** | −0.00572 | .918/.482/.632 | 0/0 |

**V2-A remains the provisional selection by tuning MAE.** LightGBM is only $0.00007/kWh worse on tuning, CatBoost $0.00034/kWh worse. CatBoost's $0.00002/kWh reused-evaluation MAE edge over V2-A is too small and too reused to establish superiority. LightGBM and CatBoost use direct absolute loss like V2-A, so their comparison is more informative about architecture than the Ridge/Lasso/Forest comparisons; differences in other models' loss and regularisation confound any architecture-only conclusion. No model anticipates one of the 14 true evaluation spikes. CatBoost flags 3,984 of 8,261 negative targets with 356 false positives and 243 of 2,068 negative onsets; LightGBM flags 3,917 with 362 false positives and 291 onsets. Persistence has stronger negative F1 but no newly negative onset detection at decision time. Conditional spike MAE remains about **$12.5/kWh** for all leading direct-L1 models. The full [results JSON](models/price_tabular_comparison/results.json) includes conditional price ranges, every month and hour, event onset and false-positive counts, and chronological subperiods. CatBoost's MAE is not lowest in every evaluation month; November contributes notably to its small overall edge. Model-specific coefficients/importance are descriptive and use different definitions, so their bar lengths are not directly comparable or causal.

The local CPU training times were Ridge **0.68 s** and Lasso **2.52 s** across three α fits each, Random Forest **191.64 s**, LightGBM **2.52 s**, CatBoost **8.63 s**. On 44,040 rows, model-only batch inference took **0.0057, 0.0044, 0.0711, 0.0956, 0.0086 s**, respectively; median warmed one-row inference was **119, 117, 1,707, 259, 616 µs**. Saved model sizes were approximately **1.8 KB, 1.8 KB, 1.05 MB, 0.83 MB, 0.34 MB**. These timings exclude feature production, CSV loading, startup and service overhead; the Forest's one-worker setting makes direct speed comparisons machine-specific. Saved-model reload reproduced predictions for each model, with checksums and exact feature order in per-model metadata. Linear toy refits reproduce predictions with fixed inputs and solver settings. There is no separate full-data repeat fit of every nonlinear model; their fixed seeds and artifacts support rerunning the comparison, but bitwise cross-machine reproducibility is not claimed.

The machine-readable [14-model leaderboard](models/forecasting_leaderboard.json) stores configuration, loss, absolute/residual target type, feature fingerprints, split counts, full tuning/evaluation scores, runtimes and artifact locations. Model-only batch and warmed one-row inference timings were also measured for the preserved XGBoost artifacts. The four deterministic rules have no fitting time or saved model; their reported batch time is the combined generation of all four baseline arrays from prepared metadata, so it is not directly comparable to a single model's batch timing. Its nullable multi-horizon and downstream MPC fields allow later experiments without implying those have been performed. Matched prediction files are in `models/price_tabular_comparison/`; [12 plots](../plots/forecasting/tabular_comparison/manifest.json) show rankings, events, distributions, monthly/hourly errors, six outcome-selected days, computational cost and feature importance. Scatter and histogram axes explicitly clip extremes for readability; the metrics use every unmodified value. Seven new automated tests bring the suite to **77 passing tests**. No ML controller, financial backtest, neural network or held-out summer analysis was performed.

## First sequence comparison: LSTM, GRU and causal TCN

The [prespecified experiment specification](NEURAL_EXPERIMENT_SPEC.md) records the architecture equations, dimensions, parameter counts, optimiser, early-stopping rule, resource check and split rules **before training**. Every input is a lazy `(72, 41)` view of chronological five-minute feature rows, ending at decision `t`. The target is still the price for the five-minute interval starting at `t+60 minutes`. Thus the input spans 72 historical five-minute rows (six hours of feature rows), the **forecast horizon** is 60 minutes, and the **forecast resolution** is one five-minute price. Individual rows already contain daily and weekly lags, so the maximum underlying raw-data lookback exceeds six hours. The model does not produce 12 intermediate forecasts. Input features remain decision-time-safe under the existing optimistic AEMO publication assumption.

The 12 feature rows absent from the train CSV because their targets cross into January 2025 were reconstructed from the checksum-verified December 2024 processed observations **only as context**, using the existing feature formula. They have no training labels. The first **71 of 208,800** train origins lack 72 prior feature rows and are excluded, leaving **208,729** neural training samples. No tuning or evaluation origin is lost: **50,100 January–June tuning** and **44,040 July–November reused development evaluation** origins exactly match the prior 14-model comparison, including timestamps and actual target prices. Validation context can include preceding train, June/July boundary and development-week rows when available; development-week origins and targets remain excluded from scored rows. One shared `(304,992, 41)` array holds feature rows, and each window is sliced on demand. The scaler mean and standard deviation are fitted on the original **208,800 training feature rows only**. Negative and extreme prices remain unchanged.

PyTorch **2.15.0+cpu** installed into the existing Windows Python **3.14.7** environment; CUDA is unavailable in that wheel. At inspection, this machine had **16 logical CPU cores**, **13.81 GiB RAM** with about **2.95 GiB free**, and an RTX 3050 laptop GPU with **6 GiB VRAM**. The fixed CPU plan used four PyTorch threads, no DataLoader workers, 512-row batches, MAE loss, AdamW at learning rate 0.001 and weight decay 0.0001, gradient-norm clip 1, seed 42, at most eight epochs and two-epoch patience. A tuning improvement must be at least **0.0001 AUD/kWh** to replace the saved checkpoint or reset patience. The initial four-batch benchmark estimated approximately 25, 49 and 33 seconds per epoch for LSTM, GRU and TCN; no dataset, batch size or architecture was changed after that check. Model sizes were deliberately similar: **9,633 LSTM**, **10,001 GRU** and **10,081 TCN** trainable parameters. The TCN's causal, left-padded residual blocks have a **128-row receptive field**, covering all 72 input rows; tests verify future-row changes cannot alter earlier convolution outputs. All heads are unrestricted linear layers, so negative forecasts are possible.

### Matched forecast results

The table uses the identical 44,040 **reused development evaluation** rows; errors and bias are AUD/kWh. Full 17-model metrics, conditional price-range errors, 24 hourly scores, five monthly scores, material price changes, event onset and false-positive diagnostics are in [the results JSON](models/price_sequence_comparison/results.json). The [persistent leaderboard](models/forecasting_leaderboard.json) now has 17 entries and retains a byte-for-byte snapshot of the prior 14-entry version beside the neural results.

| Model | Tune MAE | Evaluation MAE | RMSE | Median AE | Bias | Negative P/R/F1 | Negative TP/FP | ≥$1 spike TP/FP |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| V2-A | **0.05490** | **0.02947** | **0.27039** | 0.01571 | −0.00511 | .903/.460/.610 | 3,804/407 | 0/0 |
| LSTM | 0.05708 | 0.03131 | 0.27095 | 0.01701 | −0.00508 | .884/.552/.680 | 4,558/596 | 0/0 |
| GRU | **0.05598** among neural | **0.03019** among neural | 0.27056 | **0.01627** among neural | −0.00616 | .874/.595/.708 | 4,918/706 | 0/0 |
| TCN | 0.05677 | 0.03232 | 0.27612 | 0.01746 | −0.00730 | .861/.588/.699 | 4,859/784 | 0/5 |

There are **8,261 actual negative-price targets**, **2,068 new negative onsets** relative to the latest completed price and **14 actual ≥$1/kWh spikes**. LSTM, GRU and TCN detect **341, 414 and 489** negative onsets, versus **327** for V2-A. All three have negative F1 above V2-A's 0.610 but below persistence's 0.750. **All three miss all 14 true spikes**; TCN adds five false spike alarms. Conditional MAE on actual negative targets is **0.02520 LSTM, 0.02439 GRU, 0.02819 TCN**, versus **0.02540 V2-A**; conditional MAE on actual spikes stays about **$12.5/kWh** for all four. GRU is worse than V2-A in **every evaluation month**, though it has lower conditional error on actual $0.30–$1.00 prices (0.16894 versus 0.20098). All four have their worst hourly MAE at **18:00 Sydney time**. These are demonstrated observations. A likely explanation for missed rare spikes is the combination of rare extreme events and MAE training; this experiment does not establish that cause.

**V2-A remains the provisional overall leader by January–June tuning MAE.** GRU is the best neural candidate under the same rule, but its tuning MAE is $0.00108/kWh higher than V2-A. The sequence networks see 72 historical feature vectors while V2-A sees one; any gain or loss cannot be attributed solely to LSTM/GRU/TCN architecture. The LSTM train MAE continued falling after epoch 1 while its tuning MAE stayed near 0.057, consistent with limited generalisation from later epochs; the curves alone do not identify a unique cause. GRU's best checkpoint is epoch **6 of 8**, LSTM **1 of 3**, TCN **4 of 6**. The TCN's epoch-6 raw tuning MAE was slightly lower than epoch 4 by under $0.0001/kWh, so the prespecified minimum-improvement rule retained epoch 4.

| Model | Fit wall time | Fit CPU time | Effective CPU cores | Evaluation batch / 44,040 | Warmed one-row median | Checkpoint size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | 54.61 s | 216.70 s | 3.97 | 0.567 s | 182 µs | 41,213 bytes |
| GRU | 314.64 s | 1,251.64 s | 3.98 | 1.672 s | 1,024 µs | 42,813 bytes |
| TCN | 110.21 s | 438.05 s | 3.97 | 1.276 s | 497 µs | 46,333 bytes |

A Windows process sample during training showed **574.8 MiB peak working set up to that sample**; it is a run-level observation, not a separately measured peak for each architecture. PyTorch allocated **0 MiB GPU memory** because the installed wheel and training device were CPU-only. Fit times exclude initial feature CSV loading. Batch inference includes lazy 72-row window assembly and DataLoader collation from **already computed** 41-feature rows; warmed one-row inference times only the forward pass on a preassembled window. Neither includes producing the 41 features from live observations. They are therefore not directly comparable to the prepared-matrix batch timings for the older tabular models. A checkpoint reload reproduces saved predictions to floating-point tolerance; feature order, model SHA-256 and training-only scaler parameters are checked. Seed 42, CPU deterministic algorithms and zero loader workers improve repeatability, but a second full training run and cross-machine bitwise reproducibility have **not** been demonstrated.

[Nine Matplotlib plots](../plots/forecasting/sequence_comparison/manifest.json) show loss curves, clipped prediction/residual distributions, the 17-model MAE ranking and two outcome-selected days each for ordinary, negative and extreme-spike conditions. Display clipping affects plots only, never metrics. **All 85 tests pass**, including the previous 77 and eight sequence tests. The held-out December 2025–February 2026 summer period was not accessed. No new features, multi-horizon target, Transformer, forecast-to-MPC link, RL or hardware integration was introduced.
