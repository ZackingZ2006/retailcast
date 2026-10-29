import numpy as np
import pandas as pd
import pytest

from retailcast.config import Config
from retailcast.evaluation import (add_intervals, conformal_radius, make_splits,
                                   metrics, select_models)
from retailcast.features import make_design, origin_features, training_slice


def example_panel(n=740):
    dates = pd.date_range("2009-12-01", periods=n)
    a = np.arange(n)
    return pd.DataFrame({"A": 100 + a % 7 * 10 + a * 0.1, "B": 20 + a % 7}, index=dates)


def test_purge_labels_not_only_origins():
    panel = example_panel()
    cfg = Config()
    x, meta = make_design(panel, cfg)
    cutoff = 520
    tx, ty = training_slice(x, meta, cutoff)
    selected = meta.loc[tx.index]
    assert (selected.target_index < cutoff).all()
    assert (selected.origin_index < cutoff).all()
    assert len(ty) == len(tx)
    # Rows with past origins and future labels exist, and must be excluded.
    risky = meta[(meta.origin_index < cutoff) & (meta.target_index >= cutoff)]
    assert len(risky) > 0
    assert not tx.index.isin(risky.index).any()


def test_future_perturbation_cannot_change_training_or_features():
    p = example_panel()
    q = p.copy()
    cutoff = 520
    q.iloc[cutoff:] = 1e9
    x1, m1 = make_design(p, Config())
    x2, m2 = make_design(q, Config())
    a, ay = training_slice(x1, m1, cutoff)
    b, by = training_slice(x2, m2, cutoff)
    pd.testing.assert_frame_equal(a, b)
    np.testing.assert_array_equal(ay, by)
    f1, s1 = origin_features(p.A.iloc[:cutoff], p.index[cutoff], range(1, 29), 0, 2)
    f2, s2 = origin_features(q.A.iloc[:cutoff], q.index[cutoff], range(1, 29), 0, 2)
    pd.testing.assert_frame_equal(f1, f2)
    assert s1 == s2


def test_all_28_horizons_use_frozen_origin_lags():
    p = example_panel()
    f, s = origin_features(p.A.iloc[:520], p.index[520], range(1, 29), 0, 2)
    assert f.origin_lag_1.nunique() == 1
    assert f.mean_28.nunique() == 1
    assert list(f.horizon) == list(range(1, 29))
    assert f.year_anchor.iloc[-1] == pytest.approx(p.A.iloc[520 + 27 - 364] / s)


def test_nonoverlapping_evaluation_and_untouched_holdout():
    p, cfg = example_panel(), Config()
    s = make_splits(p, cfg)
    assert [x["stage"] for x in s] == ["validation"]*4 + ["calibration"]*2 + ["holdout"]
    for prev, nxt in zip(s[:-1], s[1:]):
        assert pd.Timestamp(prev["test_end"]) < pd.Timestamp(nxt["test_start"])
    assert s[-1]["test_end"] == str(p.index[-1].date())


def test_selection_uses_validation_error_only():
    cv = pd.DataFrame({"market": ["A"]*4, "model": ["simple", "complex"]*2,
                       "scaled_abs_error": [0.5, 0.8, 0.6, 0.7]})
    assert select_models(cv) == {"A": "simple"}


def test_finite_sample_conformal_order_statistic():
    assert conformal_radius(np.arange(1, 57), .8) == 46
    assert conformal_radius(np.arange(1, 57), .95) == 55
    with pytest.raises(ValueError):
        conformal_radius([1, 2, 3], .95)


def test_intervals_nonnegative_and_nested():
    p = pd.DataFrame({"market": ["A", "A"], "prediction": [0., 10.]})
    p = add_intervals(p, {"A": {"0.8": 5, "0.95": 9}})
    assert (p.lower_95 >= 0).all()
    assert (p.lower_95 <= p.lower_80).all()
    assert (p.upper_95 >= p.upper_80).all()
    assert (p.lower_80 <= p.prediction).all()


def test_zero_actuals_do_not_invent_percentage_metrics():
    f = pd.DataFrame({"prediction": [1, 2], "actual": [0, 0], "scaled_abs_error": [1, 2]})
    result = metrics(f)
    assert result["WAPE"] is None
    assert result["bias_pct"] is None
