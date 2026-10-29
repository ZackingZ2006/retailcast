"""Reconcile the distributed experiment independently of report generation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"


def main():
    summary = json.loads((REPORT / "run_summary.json").read_text())
    provenance = json.loads((ROOT / "data/processed/provenance.json").read_text())
    locked = json.loads((REPORT / "locked_selection.json").read_text())
    panel = pd.read_csv(ROOT / "data/processed/daily_sales.csv", index_col="date", parse_dates=True)
    digest = hashlib.sha256((ROOT / "data/processed/daily_sales.csv").read_bytes()).hexdigest()
    assert digest == summary["processed_sha256"], "Processed panel hash mismatch"
    assert list(panel.columns) == summary["markets"]
    assert len(panel.columns) == summary["config"]["markets"]
    assert panel.index.is_unique and not panel.isna().any().any()
    assert panel.index.equals(pd.date_range(panel.index[0], panel.index[-1], freq="D"))
    audit = pd.read_csv(ROOT / "data/processed/cleaning_audit.csv")
    assert audit.removed.sum() + audit.iloc[-1].remaining == provenance["raw_rows"]
    assert provenance["raw_rows"] == 1067371
    splits = pd.read_csv(REPORT / "temporal_splits.csv", parse_dates=["train_end", "test_start", "test_end"])
    assert (splits.train_end < splits.test_start).all()
    assert (splits.test_end-splits.test_start).dt.days.eq(27).all()
    assert pd.Timestamp(provenance["market_selection_end"]) < splits.iloc[0].test_start
    assert pd.Timestamp(locked["validation_end"]) < splits.loc[splits.stage == "calibration", "test_start"].min()
    assert locked["selection"] == summary["selection"]
    cv = pd.read_csv(REPORT / "validation_predictions.csv", parse_dates=["date"])
    assert len(cv) == 4*5*6*28
    assert not cv.duplicated(["market", "model", "fold", "date"]).any()
    choices = cv.groupby(["market", "model"]).scaled_abs_error.mean().reset_index().sort_values(["market", "scaled_abs_error", "model"]).groupby("market").first()["model"].to_dict()
    assert choices == locked["selection"]
    cal = pd.read_csv(REPORT / "calibration_predictions.csv", parse_dates=["date"])
    holdout = pd.read_csv(REPORT / "holdout_predictions.csv", parse_dates=["date"])
    future = pd.read_csv(REPORT / "forecast_28d.csv", parse_dates=["date"])
    assert cv.date.max() < cal.date.min() <= cal.date.max() < holdout.date.min()
    assert holdout.date.max() == panel.index.max() < future.date.min()
    assert cal.groupby("market").size().eq(56).all()
    assert future.groupby("market").size().eq(28).all()
    champion = holdout.loc[holdout.model == "selected_champion"]
    assert champion.groupby("market").size().eq(28).all()
    for frame in [cv, cal, holdout]:
        expected = np.array([panel.loc[d, m] for d, m in zip(frame.date, frame.market)])
        np.testing.assert_allclose(frame.actual, expected)
        np.testing.assert_allclose(frame.error, frame.prediction-frame.actual)
        np.testing.assert_allclose(frame.abs_error, frame.error.abs())
    radii = json.loads((REPORT / "interval_calibration.json").read_text())
    for market, g in cal.groupby("market"):
        sorted_errors = np.sort(g.error.abs())
        for level in [.8, .95]:
            radius = sorted_errors[int(np.ceil((len(g)+1)*level))-1]
            np.testing.assert_allclose(radii[market][str(level)], radius)
    for frame in [champion, future]:
        assert np.isfinite(frame.prediction).all() and (frame.prediction >= 0).all()
        assert (frame.lower_95 <= frame.lower_80).all()
        assert (frame.lower_80 <= frame.prediction).all()
        assert (frame.prediction <= frame.upper_80).all()
        assert (frame.upper_80 <= frame.upper_95).all()
        assert (frame.lower_95 >= 0).all()
        assert frame.selected_model.eq(frame.market.map(locked["selection"])).all()
        for suffix, level in [("80", "0.8"), ("95", "0.95")]:
            expected = frame.market.map(lambda m: radii[m][level])
            np.testing.assert_allclose(frame[f"upper_{suffix}"]-frame.prediction, expected)
    saved = pd.read_csv(REPORT / "holdout_metrics.csv").set_index("model")
    for name, g in holdout.groupby("model"):
        errors = g.prediction-g.actual
        checks = {"WAPE": errors.abs().sum()/g.actual.sum(), "MAE": errors.abs().mean(),
                  "RMSE": np.sqrt((errors**2).mean()), "bias_pct": errors.sum()/g.actual.sum(),
                  "MASE": (errors.abs()/g.mase_scale).mean()}
        for metric, value in checks.items():
            np.testing.assert_allclose(value, saved.loc[name, metric], rtol=1e-9)
    print("PASS: panel hash, source row count, cleaning audit, splits, validation selection,")
    print("      actuals, metrics, 28-day forecasts and independent interval calibration.")


if __name__ == "__main__":
    main()
