# AEMO forecast-horizon diagnostics — January 2025

This is **exploratory development analysis**, not independent validation. It uses all 8,928 five-minute realised NSW1 prices in Sydney-local January 2025, including the previously identified 13–19 January development week. It neither changes the 17-model leaderboard nor selects a new controller. The [machine-readable result](derived/aemo_horizon_2025_01.json) contains every matched score, coverage denominator, event count and forecast-age distribution. No December 2025–February 2026 held-out data was accessed.

## What is being compared

At decision time `t`, a P5MIN horizon `h` targets the **five-minute interval starting at `t+h`**. Its realised target is the existing `realised_spot_aud_per_kwh` for that interval. Persistence repeats the latest completed five-minute realised price, from the interval starting at `t−5`. Only rows with that observation, a realised target and an eligible AEMO forecast enter both scores. The published-time assumption is **`RUN_DATETIME + delay ≤ t`**, with delays of 0, 1, 5, 10 and 15 minutes. `RUN_DATETIME` is the nominal end of the first period, not a measured release time; these remain sensitivity calculations.

A 30-minute predispatch horizon of 1, 3, 6 or 12 hours targets the **half-hour starting at `t+h`**. Decisions are restricted to the half-hour grid. [AEMO describes `PREDISPATCHPRICE.RRP` as a 30-minute forecast spot price and spot prices as time-weighted](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/data-dashboard-nem). The corresponding realised target here is the arithmetic average of the six equal-length five-minute NSW1 prices in that half-hour. This is an explicit half-hour evaluation target, **not** an exact five-minute price forecast. Its persistence baseline repeats the latest completed five-minute price as a flat half-hour prediction. Predispatch availability is modeled as **`LASTCHANGED + delay ≤ t`**; the primary conservative case also requires nominal run time `≤ t`. AEMO documents `LASTCHANGED` as the time a **record was updated**, not a certified public publication or receipt timestamp, so none of these results proves operational availability.

All prices below are AUD/kWh. Positive bias means forecast minus actual is positive. A spike means **at least $1/kWh**, as in the existing project. Coverage and accuracy are shown separately because selection changes across horizons and delays. Even near-equal counts do not imply identical target rows.

Both forecast archives quote regional `RRP` in AUD/MWh; the parser divides by 1,000 before scoring. AEMO interval timestamps are fixed AEST even during Sydney daylight saving. UTC interval starts are the join keys throughout.

## P5MIN: exact five-minute targets

The primary table uses a **5-minute assumed delay**. The first 13 otherwise eligible origins lack a January archive vintage because the Sydney-local month starts before AEMO's fixed-AEST January archive; this is a source-boundary gap, not an imputed forecast. Target horizons at the end of the realised month are excluded from each denominator.

| Horizon (min) | Matched / eligible | AEMO MAE | RMSE | Median AE | Bias | Matched persistence MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8,914 / 8,927 | 0.0496 | 0.6475 | 0.0115 | +0.0052 | 0.0269 |
| 5 | 8,913 / 8,926 | 0.0576 | 0.7283 | 0.0117 | +0.0118 | 0.0358 |
| 10 | 8,912 / 8,925 | 0.0650 | 0.7802 | 0.0125 | +0.0186 | 0.0424 |
| 15 | 8,911 / 8,924 | 0.0784 | 0.9200 | 0.0128 | +0.0322 | 0.0442 |
| 20 | 8,910 / 8,923 | 0.0828 | 0.9346 | 0.0133 | +0.0362 | 0.0457 |
| 30 | 8,908 / 8,921 | 0.1033 | 1.1111 | 0.0141 | +0.0583 | 0.0484 |
| 40 | 8,906 / 8,919 | 0.1244 | 1.2355 | 0.0161 | +0.0721 | 0.0510 |
| 45 | 8,905 / 8,918 | 0.1280 | 1.2480 | 0.0160 | +0.0790 | 0.0523 |
| 50 | 0 / 8,917 | — | — | — | — | — |
| 55 | 0 / 8,916 | — | — | — | — | — |
| 60 | 0 / 8,915 | — | — | — | — | — |

Coverage at +0 through +45 minutes is 99.85% or better in this primary case. At +50, +55 and **+60**, no run is assumed available in time. The exact +60-minute finding from the previous milestone is preserved. The row count varies by horizon because each later target loses a few month-end origins; this is **not** a same-row comparison across horizons.

| Horizon (min) | Actual negative | Negative precision / recall / F1 | Actual spikes | Spike precision / recall / F1 | Spike TP / FP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,355 | .733 / .788 / .759 | 10 | .071 / .100 / .083 | 1 / 13 |
| 5 | 1,355 | .753 / .740 / .747 | 10 | .050 / .100 / .067 | 1 / 19 |
| 10 | 1,355 | .754 / .696 / .724 | 10 | .000 / .000 / .000 | 0 / 25 |
| 15 | 1,355 | .751 / .673 / .710 | 10 | .000 / .000 / .000 | 0 / 29 |
| 20 | 1,355 | .756 / .650 / .699 | 10 | .029 / .100 / .044 | 1 / 34 |
| 30 | 1,355 | .731 / .623 / .673 | 10 | .023 / .100 / .038 | 1 / 42 |
| 40 | 1,355 | .695 / .615 / .652 | 10 | .018 / .100 / .030 | 1 / 56 |
| 45 | 1,355 | .699 / .599 / .645 | 10 | .034 / .200 / .059 | 2 / 56 |
| 50–60 | — | undefined: no matched rows | — | undefined | — |

At a five-minute assumed delay, every selected run is nominally **5 minutes old** at the decision and has **0 minutes of age since assumed availability**. Target lead measured from that run is exactly `h+5` minutes; each min, median, p95 and max is that same value. The [JSON](derived/aemo_horizon_2025_01.json) records these distributions separately for every horizon and delay. This degeneracy follows the regular five-minute run and decision clocks; it does not verify release timing.

### Publication-delay sensitivity

The table gives matched counts at each exact target horizon. Every denominator is unchanged by the delay choice and is listed above for the 5-minute case.

| Assumed delay (min) | +0 | +5 | +10 | +15 | +20 | +30 | +40 | +45 | +50 | +55 | +60 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8,915 | 8,914 | 8,913 | 8,912 | 8,911 | 8,909 | 8,907 | 8,906 | 8,905 | 0 | 0 |
| 1 | 8,914 | 8,913 | 8,912 | 8,911 | 8,910 | 8,908 | 8,906 | 8,905 | 0 | 0 | 0 |
| 5 | 8,914 | 8,913 | 8,912 | 8,911 | 8,910 | 8,908 | 8,906 | 8,905 | 0 | 0 | 0 |
| 10 | 8,913 | 8,912 | 8,911 | 8,910 | 8,909 | 8,907 | 8,905 | 0 | 0 | 0 | 0 |
| 15 | 8,912 | 8,911 | 8,910 | 8,909 | 8,908 | 8,906 | 0 | 0 | 0 | 0 | 0 |

For example, at +30 minutes, AEMO MAE is **0.0964 / 0.1033 / 0.1138 / 0.1244** with 0 / 5 / 10 / 15-minute delays, respectively. These are partly different source vintages and slightly different boundary rows; the exact matched counts and paired persistence scores are in the JSON. No tested delay gives P5MIN exact +60-minute coverage.

## Complete 30-minute predispatch: half-hour targets

The [separate `PREDISP_ALL_DATA` directory](https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/2025/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/) does preserve complete forecast vintages. January 2025 `PREDISPATCHPRICE` has **82,584 NSW1 rows, 1,488 nominal runs and 1,536 distinct target half-hours**; `PREDISPATCHREGIONSUM` has the same NSW1 counts and matching `(sequence, target, intervention, runno)` keys. All sampled NSW1 rows have `INTERVENTION=0` and `RUNNO=1`; other dates could differ. The price and region ZIPs are **5,789,370 / 49,075,675 bytes compressed** and **69,067,456 / 178,335,068 bytes expanded**, respectively. Their immutable bytes, URLs, SHA-256 and retrieval headers are stored in `raw/aemo_predisp_all_2025_01/manifest.json`. The [index-only audit](derived/aemo_predisp_all_archive_index.json) finds both complete-table file names in all 35 monthly directories January 2023–November 2025. Only January 2025 contents were downloaded or parsed.

The primary diagnostic below uses **`LASTCHANGED + 10 minutes ≤ t` and nominal run time `≤ t`**. It has a ten-origin boundary gap at the start of Sydney-local January because December 2024 complete runs were not downloaded. The 30-minute realised target requires all six five-minute observations, and the last 1/3/6/12 hours of origin candidates are excluded when that target would cross the January boundary.

| Horizon (h) | Matched / eligible | AEMO MAE | RMSE | Median AE | Bias | Matched persistence MAE | Matched regional rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,475 / 1,485 | 0.1050 | 1.0710 | 0.0118 | +0.0578 | 0.0536 | 1,475 |
| 3 | 1,471 / 1,481 | 0.0990 | 1.0224 | 0.0131 | +0.0510 | 0.0698 | 1,471 |
| 6 | 1,465 / 1,475 | 0.1291 | 1.2446 | 0.0136 | +0.0811 | 0.0874 | 1,465 |
| 12 | 1,453 / 1,463 | 0.1730 | 1.4680 | 0.0144 | +0.1255 | 0.1003 | 1,453 |

| Horizon (h) | Actual negative | Negative precision / recall / F1 | Actual spikes | Spike precision / recall / F1 | Spike TP / FP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 207 | .745 / .676 / .709 | 6 | .100 / .167 / .125 | 1 / 9 |
| 3 | 205 | .725 / .644 / .682 | 6 | .143 / .167 / .154 | 1 / 6 |
| 6 | 199 | .717 / .648 / .681 | 6 | .077 / .167 / .105 | 1 / 12 |
| 12 | 195 | .678 / .605 / .640 | 6 | .133 / .333 / .190 | 2 / 13 |

For the primary case, the selected nominal run equals the decision half-hour: its nominal age is **0 minutes**, and target lead from that nominal run is exactly **60, 180, 360 or 720 minutes**. `LASTCHANGED` precedes the nominal stamp; after adding the ten-minute assumed margin, forecast age at decision has **minimum 15.98, median 17.77, p95 18.38 and maximum 18.73 minutes** across the one-hour matched rows. These are row-update-based proxies, not observed publication-to-decision ages. The JSON has the corresponding distribution for every horizon and assumption.

Delays of 0, 5 and 10 minutes, with or without the nominal-elapsed guard at ten minutes, select the same monthly-grid vintages and counts. With a 30-minute margin, the one-hour result changes to **1,474 matched rows and 0.1173 MAE**; the 3/6/12-hour MAEs become **0.1126 / 0.1299 / 0.1899**. Matched persistence scores and all counts for these sensitivity cases are in the JSON.

## Interpretation and limits

P5MIN has smaller **median** errors than persistence at several horizons, yet its **MAE and RMSE are worse** on the matched January rows, with positive bias that grows by horizon. Its spike precision is poor and false alarms increase with lead. Predispatch has a similar pattern: the one-hour half-hour median absolute error is low, but MAE is about twice matched persistence. Across all four tested predispatch horizons, its MAE exceeds matched persistence on this sample. Rare extreme prices strongly affect the means, but this diagnostic alone does not establish a causal explanation for every error.

The source files are hindsight monthly extracts. [AEMO's `PREDISPATCHPRICE` data model](https://visualisations.aemo.com.au/aemo/nemweb/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_276.htm) defines `PREDISPATCHSEQNO` as a `YYYYMMDDPP` nominal sequence, with `01` at **04:30 fixed AEST**; the parser computes nominal time as `04:30 + (PP−1) × 30 minutes` on the encoded market date. `PERIODID` is a count, and `DATETIME` is the target **interval end**. The parser subtracts 30 minutes for target start and converts AEST to UTC before matching. `RUNNO` identifies reruns; intervention 0 pricing rows are selected, and equal-timestamp duplicate vintages fail closed. The AEMO data model describes `LASTCHANGED` as the record's update time. [NEMSEER parses the sequence into a nominal run time](https://github.com/UNSW-CEEM/NEMSEER/blob/master/src/nemseer/data_handlers.py) and calls `LASTCHANGED` an often-reported actual run/published time, but neither establishes the exact time a January 2025 file became publicly reachable. These conditional metrics must not be treated as a verified live backtest.

The complete archive supports future research on 30-minute forecast trajectories and regional demand/supply fields. A 30-minute `RRP` cannot be inserted as the exact `t+60` five-minute label. Assigning one half-hour forecast to six five-minute intervals would be a **separate coarse-resolution approximation** requiring its own evaluation. The existing V2-A, LightGBM, CatBoost and GRU one-hour models have not been compared against these different targets.
