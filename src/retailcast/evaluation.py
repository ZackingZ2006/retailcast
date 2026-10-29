import numpy as np
import pandas as pd


def make_splits(panel, cfg):
    n = len(panel)
    first = n - cfg.horizon * (cfg.cv_folds + cfg.calibration_folds + 1)
    splits = []
    for i in range(cfg.cv_folds + cfg.calibration_folds + 1):
        cutoff = first + i * cfg.horizon
        stage = "validation" if i < cfg.cv_folds else "calibration" if i < cfg.cv_folds + cfg.calibration_folds else "holdout"
        splits.append({"fold": i + 1, "stage": stage, "cutoff_index": cutoff,
                       "train_end": str(panel.index[cutoff - 1].date()),
                       "test_start": str(panel.index[cutoff].date()),
                       "test_end": str(panel.index[cutoff + cfg.horizon - 1].date())})
    return splits


def attach_actuals(pred, panel, cutoff):
    out = pred.copy()
    out["actual"] = [float(panel.loc[d, m]) for d, m in zip(out["date"], out["market"])]
    history = panel.iloc[:cutoff]
    # MASE uses in-sample seasonal (7-day) absolute differences.
    scales = {m: max(float(np.abs(history[m].to_numpy()[7:] - history[m].to_numpy()[:-7]).mean()), 1e-9) for m in panel.columns}
    out["mase_scale"] = out["market"].map(scales)
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["scaled_abs_error"] = out["abs_error"] / out["mase_scale"]
    return out


def metrics(frame):
    error = frame["prediction"].to_numpy() - frame["actual"].to_numpy()
    denominator = float(frame["actual"].abs().sum())
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt((error ** 2).mean())),
            "WAPE": float(np.abs(error).sum() / denominator) if denominator else None,
            "MASE": float(frame["scaled_abs_error"].mean()),
            "bias_pct": float(error.sum() / denominator) if denominator else None}


def grouped_metrics(frame, groups):
    rows = []
    for key, group in frame.groupby(groups, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(groups, key)) | metrics(group))
    return pd.DataFrame(rows)


def select_models(cv):
    scores = cv.groupby(["market", "model"])["scaled_abs_error"].mean().reset_index()
    # Deterministic model-name ordering breaks exact ties.
    return scores.sort_values(["market", "scaled_abs_error", "model"]).groupby("market", sort=False).first()["model"].to_dict()


def keep_champions(pred, selection):
    out = pred.loc[pred["model"] == pred["market"].map(selection)].copy()
    out["selected_model"] = out["model"]
    out["model"] = "selected_champion"
    return out


def conformal_radius(errors, coverage):
    values = np.sort(np.abs(np.asarray(errors, dtype=float)))
    if not 0 < coverage < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite calibration errors and coverage between 0 and 1")
    rank = int(np.ceil((len(values) + 1) * coverage))
    if rank > len(values):
        raise ValueError("Not enough calibration observations for requested coverage")
    return float(values[rank - 1])


def calibrate(calibration):
    radii = {}
    for market, g in calibration.groupby("market"):
        radii[market] = {str(level): conformal_radius(g["error"], level) for level in [0.8, 0.95]}
    return radii


def add_intervals(pred, radii):
    out = pred.copy()
    for level, suffix in [(0.8, "80"), (0.95, "95")]:
        q = out["market"].map(lambda m: radii[m][str(level)])
        out[f"lower_{suffix}"] = (out["prediction"] - q).clip(lower=0)
        out[f"upper_{suffix}"] = out["prediction"] + q
    return out


def interval_metrics(frame):
    rows = []
    for market, g in frame.groupby("market"):
        row = {"market": market, "n": len(g)}
        for level, suffix in [(0.8, "80"), (0.95, "95")]:
            lo, hi, y = g[f"lower_{suffix}"], g[f"upper_{suffix}"], g["actual"]
            alpha = 1 - level
            row[f"coverage_{suffix}"] = float(((y >= lo) & (y <= hi)).mean())
            row[f"width_{suffix}"] = float((hi - lo).mean())
            row[f"interval_score_{suffix}"] = float(((hi-lo) + 2/alpha*(lo-y).clip(lower=0) + 2/alpha*(y-hi).clip(lower=0)).mean())
        rows.append(row)
    return pd.DataFrame(rows)
