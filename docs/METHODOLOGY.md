# Forecast protocol

## Forecast contract
At the start of a day, use completed previous days to predict gross positive merchandise units for the next 28 days in each of five countries. Every prediction in that batch uses the same frozen origin. This is a sales forecast; stockout-censored latent demand is not observable.

## Source and target construction
Both sheets of UCI Online Retail II are concatenated in source order. Exact duplicate rows are dropped, keeping the first occurrence. This is an assumption: without line IDs, legitimate identical lines cannot always be distinguished. Cancelled invoices, nonpositive quantity or price, nonfinite required values, and stock codes outside `\d{5}[A-Za-z]*` are excluded. Administrative codes such as POST are not merchandise. Missing customer IDs are retained. Large orders are retained. The final partial day is excluded. Missing market/day combinations are filled with zero observed sales; missing reporting cannot be distinguished from genuine inactivity.

The five largest markets are chosen by unit sales strictly before the first validation origin. This prevents future-informed series selection. The exported panel contains no customer identifiers. Source audit, workbook SHA-256 and citation appear in `data/processed/provenance.json`.

## Direct global supervised model
For every seventh historical origin, create one row per market and horizon h=1..28. The label is y[origin+h-1] / scale(origin). The scale is max(mean of the last 56 observed days, 1). All features use data strictly before the origin or a calendar known at the origin. Train rows whose **label dates** fall at or after a fold cutoff are purged; filtering only origin dates would leak multi-step labels.

Features: market one-hot, origin lags, rolling mean/std/zero share, recent trend, weekday profile, target calendar and a 364-day aligned anchor when available. The 364-day anchor uses only observed history for all 28 horizons. Early training examples carry an indicator when the anchor does not exist. No fitted target encoder or full-data scaler is used. Ridge's feature scaler is fitted separately on each training fold. Market normalization at an origin uses only that market's trailing history.

Market-level normalization gives small markets influence during global fitting. It does not imply that the global model optimizes the final original-unit WAPE. Daily origin sampling at stride 7 reduces runtime and correlated duplicate labels; overlapping historical labels are allowed inside training, never across its cutoff.

## Candidates declared before the experiment
| Model | Configuration | Role |
|---|---|---|
| seasonal_naive | Repeat the last 7 completed days | Minimum credible baseline |
| weekday_mean_8w | Mean for each weekday over the last 56 days | Smoothed seasonal baseline |
| ets_weekly | Additive season, additive damped trend, period 7 | Classical statistical model |
| ridge_direct | StandardScaler + Ridge alpha=100 | Regularized linear multi-horizon model |
| gbm_shallow | Poisson HGB, 7 leaves, 180 iterations, lr=.05 | Nonlinear global model |
| gbm_deep | Poisson HGB, 15 leaves, 180 iterations, lr=.05 | Capacity comparison |

Both GBMs use min_samples_leaf=40, L2=10, seed=42; random early stopping is disabled. Poisson loss models a positive conditional mean; using it on scaled nonnegative quantities is a loss choice, not a claim that observed counts follow an exact Poisson distribution. Any negative Ridge/ETS output is clipped to zero consistently in validation and deployment.

The comparison of two tree capacities is the entire predefined tuning budget. There is no hidden exhaustive search or selection after seeing holdout errors. Seasonal baselines remain eligible to win. ETS optimizer failure raises an error; it is never silently replaced.

## Temporal validation and locking
The last 196 complete days are partitioned into seven adjacent non-overlapping 28-day blocks: four validation, two calibration and one holdout. All earlier data is training. Training expands at every origin, including previously observed evaluation blocks once those blocks are in the past.

For each market select the lowest average validation MASE among the six candidates. Four 28-day windows have equal weight. In exact ties, deterministic model-name ordering breaks the tie; no test values are consulted. The locked algorithm mapping is saved before computing calibration and holdout forecasts. The holdout is a single final period, not a claim of out-of-season generalization. Market-wise selection can still overfit four folds; publish all candidate scores and avoid claiming the selected validation score is unbiased.

## Metrics
- MAE and RMSE are in units/day; RMSE highlights large misses.
- WAPE = sum(abs(error)) / sum(actual); undefined when actual total is zero. Combined WAPE favours large markets.
- Seasonal MASE uses the mean absolute 7-day difference in that origin's training history. Equal observation counts give each market equal weight in overall MASE. MASE<1 compares against the in-sample scaling denominator; compare test baseline predictions explicitly to claim test improvement.
- Bias = sum(prediction - actual) / sum(actual); positive means overforecasting.
- Breakdown by market, weekday and week ahead; retain the 20 largest absolute misses.

## Uncertainty
After model selection, forecast the next two 28-day blocks with expanding refits of the locked algorithms. For each market, pool 56 absolute residuals and use the finite-sample order statistic ceil((n+1) * coverage), producing 80% and 95% symmetric error radii. Clamp the lower bound to zero. The same radius is used across horizons. No holdout errors are used in calibration.

This is split-conformal-style empirical calibration. Exchangeability is implausible with time dependence, nonstationarity and overlapping training; therefore no distribution-free time-series coverage guarantee is asserted. Two windows offer limited calibration diversity. Coverage and interval scores are reported on the holdout. Bounds across days are not a joint predictive region, and summing bounds is not a calibrated interval for the 28-day total. Final refitting on all observations changes the model fit, while the old calibration radii remain fixed and may become stale.

## Historical continuation
After holdout evaluation, refit the previously selected algorithms on the full panel and predict the next 28 historical dates. The dataset is old and the future period is unobserved, including the Christmas closure period. Do not describe it as a current business forecast or claim a measured continuation accuracy.

## Deployment scope and monitoring proposal
The included CLI, processed panel, run manifest and read-only dashboards make this a reproducible portfolio experiment. It is not a production service. A real service needs ingestion timestamps, data completeness checks, scheduler, persistence/versioning, drift alerts, retrain policy and a new external evaluation period. Proposed monitoring: daily completeness by market, weekly bias/WAPE once labels mature, rolling interval coverage, new/retired markets and alert thresholds set from an agreed business SLA. No invented inventory savings or causal business lift is reported.

## References
- Chen, D. (2012). [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), DOI: 10.24432/C5CG6D, CC BY 4.0.
- scikit-learn. [Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
- scikit-learn. [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
- statsmodels. [ExponentialSmoothing](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html).
