import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_panel, prepare, sha256
from .evaluation import (add_intervals, attach_actuals, calibrate, grouped_metrics,
                         interval_metrics, keep_champions, make_splits, metrics, select_models)
from .features import make_design
from .models import CANDIDATES, forecast_models


def run(root, cfg=None, refresh=False):
    root = Path(root)
    cfg = cfg or Config()
    start_time = time.time()
    if refresh or not (root / "data/processed/daily_sales.csv").exists():
        panel = prepare(root, cfg)
    else:
        panel = load_panel(root)
    print(f"Panel: {panel.shape}, {panel.index.min().date()} to {panel.index.max().date()}", flush=True)
    splits = make_splits(panel, cfg)
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(splits).to_csv(report / "temporal_splits.csv", index=False)
    x, meta = make_design(panel, cfg)
    print(f"Direct-horizon design: {len(x):,} rows, {x.shape[1]} features", flush=True)
    cv_rows = []
    for split in splits[:cfg.cv_folds]:
        print(f"Validation {split['fold']}: {split['test_start']} to {split['test_end']}", flush=True)
        pred, _ = forecast_models(panel, split["cutoff_index"], x, meta, cfg)
        evaluated = attach_actuals(pred, panel, split["cutoff_index"])
        evaluated["fold"] = split["fold"]
        cv_rows.append(evaluated)
    cv = pd.concat(cv_rows, ignore_index=True)
    selection = select_models(cv)
    # Save selection before even computing calibration or holdout forecasts.
    selection_artifact = {"criterion": "mean validation MASE per market", "selection": selection,
                          "validation_end": splits[cfg.cv_folds - 1]["test_end"],
                          "candidates": CANDIDATES, "config": cfg.as_dict()}
    (report / "locked_selection.json").write_text(json.dumps(selection_artifact, indent=2), encoding="utf-8")
    print(f"Locked market champions: {selection}", flush=True)
    cv.to_csv(report / "validation_predictions.csv", index=False)
    grouped_metrics(cv, ["model"]).sort_values("MASE").to_csv(report / "validation_leaderboard.csv", index=False)
    grouped_metrics(cv, ["market", "model"]).to_csv(report / "validation_by_market.csv", index=False)
    grouped_metrics(cv, ["fold", "model"]).to_csv(report / "validation_by_fold.csv", index=False)
    used = list(dict.fromkeys(selection.values()))
    calibration_rows = []
    for split in splits[cfg.cv_folds:-1]:
        print(f"Calibration {split['fold']}: {split['test_start']} to {split['test_end']}", flush=True)
        pred, _ = forecast_models(panel, split["cutoff_index"], x, meta, cfg, used)
        evaluated = keep_champions(attach_actuals(pred, panel, split["cutoff_index"]), selection)
        evaluated["fold"] = split["fold"]
        calibration_rows.append(evaluated)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    radii = calibrate(calibration)
    calibration.to_csv(report / "calibration_predictions.csv", index=False)
    (report / "interval_calibration.json").write_text(json.dumps(radii, indent=2), encoding="utf-8")
    final_split = splits[-1]
    print(f"Final holdout (selection locked): {final_split['test_start']} to {final_split['test_end']}", flush=True)
    pred, _ = forecast_models(panel, final_split["cutoff_index"], x, meta, cfg,
                              list(dict.fromkeys(used + ["seasonal_naive", "weekday_mean_8w"])))
    actual = attach_actuals(pred, panel, final_split["cutoff_index"])
    champions = add_intervals(keep_champions(actual, selection), radii)
    comparisons = pd.concat([champions, actual.loc[actual["model"].isin(["seasonal_naive", "weekday_mean_8w"])]], ignore_index=True)
    comparisons.to_csv(report / "holdout_predictions.csv", index=False)
    holdout_summary = grouped_metrics(comparisons, ["model"])
    holdout_summary.to_csv(report / "holdout_metrics.csv", index=False)
    market_scores = grouped_metrics(comparisons, ["market", "model"])
    market_scores.to_csv(report / "holdout_by_market.csv", index=False)
    interval_scores = interval_metrics(champions)
    interval_scores.to_csv(report / "interval_metrics.csv", index=False)
    champions["week_ahead"] = ((champions["horizon"] - 1) // 7 + 1).astype(int)
    champions["weekday"] = champions["date"].dt.day_name()
    grouped_metrics(champions, ["week_ahead"]).to_csv(report / "errors_by_horizon.csv", index=False)
    grouped_metrics(champions, ["market", "weekday"]).to_csv(report / "errors_by_weekday.csv", index=False)
    champions.nlargest(20, "abs_error").to_csv(report / "largest_errors.csv", index=False)
    # After scoring, refit the preselected algorithms on all available history.
    # Calibration radii remain pre-holdout and are not refreshed using test errors.
    print("Refit locked algorithms on full history for illustrative future continuation", flush=True)
    pred, _ = forecast_models(panel, len(panel), x, meta, cfg, used)
    future = add_intervals(keep_champions(pred, selection), radii)
    future.to_csv(report / "forecast_28d.csv", index=False)
    summary = {
        "config": cfg.as_dict(), "markets": list(panel.columns), "selection": selection,
        "history_start": str(panel.index[0].date()), "history_end": str(panel.index[-1].date()),
        "holdout_start": final_split["test_start"], "holdout_end": final_split["test_end"],
        "holdout": holdout_summary.to_dict(orient="records"), "intervals": interval_scores.to_dict(orient="records"),
        "forecast_start": str(future["date"].min().date()), "forecast_end": str(future["date"].max().date()),
        "runtime_seconds": round(time.time() - start_time, 2),
        "processed_sha256": sha256(root / "data/processed/daily_sales.csv"),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "statsmodels", "matplotlib"]},
        "feature_names": x.columns.tolist(), "training_rows_full": len(x),
        "warning": "Historical continuation only. Intervals use 56 earlier errors/market; temporal dependence and seasonality shifts invalidate exchangeability guarantees."
    }
    (report / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    from .reporting import build_reports
    build_reports(root, panel, summary, cv, comparisons, future)
    print(json.dumps(holdout_summary.to_dict(orient="records"), indent=2), flush=True)
    return summary
