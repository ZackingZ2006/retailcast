from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    horizon: int = 28
    cv_folds: int = 4
    calibration_folds: int = 2
    markets: int = 5
    min_history: int = 112
    origin_stride: int = 7
    seed: int = 42
    alpha_ridge: float = 100.0
    gb_iterations: int = 180

    def as_dict(self):
        return asdict(self)


ROOT = Path(__file__).resolve().parents[2]
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
DATA_SOURCE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
