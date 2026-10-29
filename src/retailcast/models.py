import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits

from .features import future_design, training_slice

CANDIDATES = ["seasonal_naive", "weekday_mean_8w", "ets_weekly", "ridge_direct", "gbm_shallow", "gbm_deep"]


def forecast_models(panel, cutoff, x, meta, cfg, candidates=None):
    candidates = candidates or CANDIDATES
    history = panel.iloc[:cutoff]
    start = panel.index[0] + pd.Timedelta(days=cutoff)
    future_x, info = future_design(history, start, cfg)
    train_x, train_y = training_slice(x, meta, cutoff)
    forecasts = []
    fitted = {}
    for name in candidates:
        if name.startswith("gbm") or name == "ridge_direct":
            if name == "ridge_direct":
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=cfg.alpha_ridge))
            else:
                estimator = HistGradientBoostingRegressor(
                    loss="poisson", learning_rate=0.05, max_iter=cfg.gb_iterations,
                    max_leaf_nodes=7 if name == "gbm_shallow" else 15,
                    min_samples_leaf=40, l2_regularization=10.0,
                    early_stopping=False, random_state=cfg.seed)
            # Disable random internal validation; tuning happens on outer time folds.
            with threadpool_limits(limits=2):
                estimator.fit(train_x, train_y)
                pred = estimator.predict(future_x) * info["scale"].to_numpy()
            fitted[name] = estimator
        else:
            pieces = []
            for market in panel.columns:
                y = history[market].to_numpy()
                if name == "seasonal_naive":
                    p = np.resize(y[-7:], cfg.horizon)
                elif name == "weekday_mean_8w":
                    p = np.array([y[-56:][np.arange(len(y) - 56, len(y)) % 7 == (len(y) + h) % 7].mean() for h in range(cfg.horizon)])
                elif name == "ets_weekly":
                    scale = max(y.mean(), 1)
                    with warnings.catch_warnings(record=True) as captured:
                        fit = ExponentialSmoothing(y / scale, trend="add", damped_trend=True,
                                                   seasonal="add", seasonal_periods=7,
                                                   initialization_method="estimated").fit(optimized=True)
                    # Fail visibly instead of silently substituting a baseline.
                    if not fit.mle_retvals.get("success", True):
                        raise RuntimeError(f"ETS optimizer did not converge: {market}, cutoff={cutoff}")
                    p = np.asarray(fit.forecast(cfg.horizon)) * scale
                else:
                    raise ValueError(name)
                pieces.append(p)
            pred = np.concatenate(pieces)
        result = info[["market", "date", "horizon"]].copy()
        result["prediction"] = np.maximum(pred, 0)
        if not np.isfinite(result["prediction"]).all():
            raise ValueError(f"Nonfinite forecast: {name}")
        result["model"] = name
        result["cutoff"] = start
        forecasts.append(result)
    return pd.concat(forecasts, ignore_index=True), fitted
