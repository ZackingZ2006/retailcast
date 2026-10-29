import numpy as np
import pandas as pd
import pytest

from retailcast.data import clean_transactions, load_panel


def test_cleaning_rules_and_partial_day_are_explicit():
    df = pd.DataFrame({"Invoice": ["1", "1", "C2", "3", "4", "5", "6", "7"],
                       "StockCode": ["12345A", "12345A", "12345", "POST", "12345", "12345", "12345", "12345"],
                       "Quantity": [5, 5, 5, 5, -1, 5, 9, 5], "Price": [2, 2, 2, 2, 2, 0, 2, 2],
                       "InvoiceDate": ["2011-01-01"]*7 + ["2011-01-02"], "Country": ["UK"]*8})
    clean, audit, last = clean_transactions(df)
    assert clean.units.sum() == 14
    assert len(clean) == 2
    assert sum(a["removed"] for a in audit) + len(clean) == len(df)
    assert last == pd.Timestamp("2011-01-02")


def test_missing_calendar_is_rejected(tmp_path):
    dest = tmp_path / "data/processed"
    dest.mkdir(parents=True)
    pd.DataFrame({"date": ["2011-01-01", "2011-01-03"], "A": [1, 2]}).to_csv(dest / "daily_sales.csv", index=False)
    with pytest.raises(ValueError, match="calendar"):
        load_panel(tmp_path)


def test_negative_sales_are_rejected(tmp_path):
    dest = tmp_path / "data/processed"
    dest.mkdir(parents=True)
    pd.DataFrame({"date": ["2011-01-01", "2011-01-02"], "A": [1, -2]}).to_csv(dest / "daily_sales.csv", index=False)
    with pytest.raises(ValueError, match="nonnegative"):
        load_panel(tmp_path)
