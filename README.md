# RetailCast — 28-day retail sales forecasting

[![Temporal integrity](https://github.com/ZackingZ2006/retailcast/actions/workflows/ci.yml/badge.svg)](https://github.com/ZackingZ2006/retailcast/actions/workflows/ci.yml)

Forecast daily units sold for the next 28 days in five countries, using real transactions from [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). The focus is on honest evaluation: time-based splits, strong baselines and tests that guard against data leakage.

🇻🇳 Báo cáo kết quả tiếng Việt: [reports/BAO_CAO_KET_QUA.md](reports/BAO_CAO_KET_QUA.md)

![Validation comparison](reports/figures/validation_leaderboard.png)

## Result

On the final holdout window **2011-11-11 → 2011-12-08** (28 days, never used for model choice):

| Strategy | WAPE | Mean seasonal MASE |
|---|---:|---:|
| Per-market champion (chosen on validation) | **26.6%** | **0.645** |
| 8-week weekday mean | 28.5% | 0.746 |
| Repeat last week (seasonal naive) | 35.4% | 0.867 |

What this does **not** show:
- WAPE is dominated by the UK. Denmark and Netherlands still have WAPE above 100%.
- The champion loses to a baseline in France (weekday mean) and Denmark (seasonal naive).
- One 28-day window just before Christmas is not evidence that the model works across seasons.

Details per market, bias and interval coverage: [reports/BAO_CAO_KET_QUA.md](reports/BAO_CAO_KET_QUA.md).

## Approach

| Step | What happens |
|---|---|
| Data | 1,067,371 transaction rows → daily positive merchandise units per market (2009-12-01 → 2011-12-08). Cancellations and non-merchandise codes removed; large orders kept. Top 5 markets chosen from the initial training period only. |
| Splits | The last 196 days form 4 validation windows, 2 calibration windows and 1 holdout, each 28 days. |
| Models | Seasonal naive, 8-week weekday mean, damped ETS, global direct Ridge, two global direct gradient boosting models (Poisson loss). |
| Selection | Lowest mean validation MASE, separately per market. Baselines are allowed to win (the UK picks the weekday mean). |
| Intervals | Empirical 80% / 95% intervals from 56 calibration errors per market. |

### Avoiding leakage in multi-step forecasting
- Features (lags, rolling means, weekday profile) use only data before the forecast origin and stay frozen for all 28 horizons.
- Training rows are filtered by **target date**, not origin date. Filtering only by origin would let labels from inside the validation window leak into training.
- `tests/test_temporal_integrity.py` checks this: changing future values must not change the features or the training set.

## Project structure

```text
src/retailcast/   data.py · features.py · models.py · evaluation.py · pipeline.py · reporting.py · cli.py
tests/            data contract, forecasting contract, temporal integrity
notebooks/        01_research.ipynb — exploration, diagnostics, one fold refit and checked against saved results
reports/          metrics, predictions, split dates, figures
data/processed/   daily panel, cleaning audit, data provenance
scripts/          audit_artifacts.py — recompute reported numbers from the processed data
app.py            Streamlit app to browse forecasts
DEMO.html         offline dashboard (download and open in a browser)
```

## Run it

Python 3.11–3.13.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
pip install -e . --no-deps

pytest -q                         # tests
python scripts/audit_artifacts.py # recheck reported numbers
retailcast run                    # rerun the full experiment from data/processed
streamlit run app.py              # interactive app
```

`retailcast run --refresh` downloads the raw workbook (~43.5 MiB) and rebuilds everything from scratch. See [data/raw/README.md](data/raw/README.md) for the file hash and source.

## Limitations and next steps

- Units sold are not demand: there is no stock, promotion, price plan or lead-time data.
- About two years of history, so yearly seasonality is weakly identified.
- Intervals use only 56 past errors per market and can under-cover when the level shifts. Daily bounds cannot be added up into a 28-day interval.
- Next: add decision-time features (promotions, holidays, stock), evaluate on more seasons, then move to SKU-level forecasts.

## Data and license

Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. DOI [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D), CC BY 4.0. Customer IDs are not exported.

Code: [MIT](LICENSE).
