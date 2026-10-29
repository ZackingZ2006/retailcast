"""One row describes information at an origin and a known future calendar date.

history contains only observations strictly before origin. Every lag/rolling
statistic is frozen for the entire 28-day forecast. Annual anchors are also past.
"""
import numpy as np
import pandas as pd


def origin_features(history, origin_date, horizons, series_id, n_series):
    y = np.asarray(history, dtype=float)
    if len(y) < 112 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Need >=112 finite nonnegative historical observations")
    scale = max(float(y[-56:].mean()), 1.0)
    base = {f"market_{i}": float(i == series_id) for i in range(n_series)}
    base["log_level"] = np.log1p(scale)
    for lag in [1, 2, 3, 7, 14, 21, 28, 56, 84]:
        base[f"origin_lag_{lag}"] = y[-lag] / scale
    for window in [7, 14, 28, 56, 112]:
        a = y[-window:]
        base[f"mean_{window}"] = a.mean() / scale
        base[f"std_{window}"] = a.std() / scale
        base[f"zero_share_{window}"] = float((a == 0).mean())
    base["recent_trend"] = y[-28:].mean() / max(y[-56:-28].mean(), 1.0)
    weekday_profile = np.array([y[-56:][np.arange(len(y) - 56, len(y)) % 7 == k].mean() for k in range(7)])
    rows = []
    for h in horizons:
        if not 1 <= h <= 28:
            raise ValueError("Supported horizons are integers 1..28")
        target_date = pd.Timestamp(origin_date) + pd.Timedelta(days=int(h) - 1)
        row = dict(base)
        row.update({"horizon": h, "target_dow_sin": np.sin(2 * np.pi * target_date.dayofweek / 7),
                    "target_dow_cos": np.cos(2 * np.pi * target_date.dayofweek / 7),
                    "target_doy_sin": np.sin(2 * np.pi * target_date.dayofyear / 365.25),
                    "target_doy_cos": np.cos(2 * np.pi * target_date.dayofyear / 365.25),
                    "target_month": target_date.month,
                    "christmas": float(target_date.month == 12 and target_date.day in [25, 26]),
                    "new_year": float(target_date.month == 1 and target_date.day == 1),
                    "weekday_profile": weekday_profile[(len(y) + h - 1) % 7] / scale})
        anchor = len(y) + h - 1 - 364
        row["has_year_anchor"] = float(anchor >= 0)
        row["year_anchor"] = y[anchor] / scale if anchor >= 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows), scale


def make_design(panel, cfg):
    chunks, meta = [], []
    for s, market in enumerate(panel.columns):
        values = panel[market].to_numpy()
        for origin in range(cfg.min_history, len(panel) - 1, cfg.origin_stride):
            horizons = np.arange(1, min(cfg.horizon, len(panel) - origin) + 1)
            x, scale = origin_features(values[:origin], panel.index[origin], horizons, s, len(panel.columns))
            chunks.append(x)
            meta.append(pd.DataFrame({"market": market, "origin_index": origin,
                                      "target_index": origin + horizons - 1, "horizon": horizons,
                                      "scale": scale, "target": values[origin + horizons - 1] / scale}))
    return pd.concat(chunks, ignore_index=True), pd.concat(meta, ignore_index=True)


def training_slice(x, meta, cutoff):
    # Purging uses target dates, not just origin dates. This matters for h>1.
    mask = meta["target_index"] < cutoff
    if not mask.any():
        raise ValueError("No training examples before cutoff")
    return x.loc[mask], meta.loc[mask, "target"].to_numpy()


def future_design(history, start_date, cfg):
    chunks, info = [], []
    for i, market in enumerate(history.columns):
        x, scale = origin_features(history[market].to_numpy(), start_date, np.arange(1, cfg.horizon + 1), i, len(history.columns))
        chunks.append(x)
        info.append(pd.DataFrame({"market": market, "date": pd.date_range(start_date, periods=cfg.horizon),
                                  "horizon": np.arange(1, cfg.horizon + 1), "scale": scale}))
    return pd.concat(chunks, ignore_index=True), pd.concat(info, ignore_index=True)
