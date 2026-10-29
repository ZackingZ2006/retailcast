import numpy as np
import pandas as pd

from retailcast.config import Config
from retailcast.evaluation import attach_actuals
from retailcast.features import make_design
from retailcast.models import forecast_models


def panel():
    t = np.arange(430)
    return pd.DataFrame({"A": 50 + 4*(t % 7) + .03*t, "B": 10 + 2*(t % 7)},
                        index=pd.date_range("2010-01-01", periods=len(t)))


def test_full_forecast_is_invariant_to_unseen_actuals():
    p = panel()
    cfg = Config(gb_iterations=10)
    cutoff = 390
    q = p.copy()
    q.iloc[cutoff:] = 999999
    x, meta = make_design(p, cfg)
    z, mz = make_design(q, cfg)
    models = ["seasonal_naive", "weekday_mean_8w", "ridge_direct", "gbm_shallow"]
    a, _ = forecast_models(p, cutoff, x, meta, cfg, models)
    b, _ = forecast_models(q, cutoff, z, mz, cfg, models)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(models)*2*28
    assert (a.prediction >= 0).all()
    assert np.isfinite(a.prediction).all()
    assert a.groupby(["market", "model"]).horizon.nunique().eq(28).all()


def test_seasonal_naive_repeats_only_completed_week():
    p = panel()
    cfg = Config()
    x, meta = make_design(p, cfg)
    pred, _ = forecast_models(p, 390, x, meta, cfg, ["seasonal_naive"])
    a = pred.loc[pred.market == "A", "prediction"].to_numpy()
    np.testing.assert_array_equal(a, np.tile(p.A.iloc[383:390], 4))


def test_mase_scaling_uses_training_seasonal_difference():
    p = panel()
    cfg = Config()
    x, meta = make_design(p, cfg)
    pred, _ = forecast_models(p, 390, x, meta, cfg, ["seasonal_naive"])
    out = attach_actuals(pred, p, 390)
    expected = np.abs(p.A.iloc[7:390].to_numpy() - p.A.iloc[:383].to_numpy()).mean()
    np.testing.assert_allclose(out.loc[out.market == "A", "mase_scale"], expected)
