"""Explicit, sequential cleaning rules; raw customer identifiers never exported."""
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_SOURCE, DATA_URL


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(root):
    dest = Path(root) / "data/raw/online_retail_II.xlsx"
    if dest.exists() and zipfile.is_zipfile(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "online_retail_ii.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        temp = archive.with_suffix(".part")
        with urllib.request.urlopen(DATA_URL, timeout=180) as r, temp.open("wb") as w:
            while block := r.read(1024 * 1024):
                w.write(block)
        temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        if len(names) != 1:
            raise ValueError("Expected exactly one Excel workbook in UCI archive")
        # Write the known filename directly, rather than extracting arbitrary paths.
        with z.open(names[0]) as r, dest.open("wb") as w:
            while block := r.read(1024 * 1024):
                w.write(block)
    return dest


def clean_transactions(raw):
    rename = {"Invoice": "invoice", "InvoiceNo": "invoice", "StockCode": "stock",
              "Quantity": "units", "InvoiceDate": "timestamp", "Price": "price",
              "UnitPrice": "price", "Country": "market"}
    raw = raw.rename(columns=rename).copy()
    cols = ["invoice", "stock", "units", "timestamp", "price", "market"]
    missing = set(cols) - set(raw)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    audit = [{"rule": "raw_rows", "removed": 0, "remaining": len(raw)}]
    # Duplicate detection uses all original fields, including customer/description.
    # Remove in source order: repeated rows from overlapping sheet boundaries
    # are attributed to the earlier copy, never removed retroactively.
    dup = raw.duplicated(keep="first")
    audit.append({"rule": "exact_duplicate_rows", "removed": int(dup.sum()), "remaining": int((~dup).sum())})
    df = raw.loc[~dup, cols].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["units", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    conditions = [
        ("missing_or_nonfinite_required", df[cols].notna().all(axis=1) & np.isfinite(df["units"]) & np.isfinite(df["price"])),
        ("cancellation_invoices", ~df["invoice"].astype(str).str.upper().str.startswith("C")),
        ("nonpositive_units_or_price", (df["units"] > 0) & (df["price"] > 0)),
        # UCI merchandise codes are five digits, optionally followed by letters.
        ("nonmerchandise_codes", df["stock"].astype(str).str.fullmatch(r"\d{5}[A-Za-z]*")),
    ]
    for name, mask in conditions:
        before = len(df)
        df = df.loc[mask.reindex(df.index).fillna(False)].copy()
        audit.append({"rule": name, "removed": before - len(df), "remaining": len(df)})
    df["date"] = df["timestamp"].dt.normalize()
    # Last date has only a partial trading day. No ad hoc clipping of large orders.
    last = raw["timestamp"].pipe(pd.to_datetime).max().normalize()
    before = len(df)
    df = df.loc[df["date"] < last].copy()
    audit.append({"rule": "incomplete_final_day", "removed": before - len(df), "remaining": len(df)})
    return df, audit, last


def prepare(root, cfg):
    root = Path(root)
    path = download(root)
    print("Reading both UCI workbook sheets ...", flush=True)
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    raw = pd.concat(sheets.values(), ignore_index=True)
    # Workbook sheets have identical schema; assert rather than silently misalign.
    if len({tuple(s.columns) for s in sheets.values()}) != 1:
        raise ValueError("Workbook sheets have different columns")
    clean, audit, last = clean_transactions(raw)
    dates = pd.date_range(clean["date"].min(), last - pd.Timedelta(days=1), freq="D")
    first_origin = len(dates) - cfg.horizon * (cfg.cv_folds + cfg.calibration_folds + 1)
    if first_origin < cfg.min_history + cfg.horizon:
        raise ValueError("Insufficient data for specified temporal design")
    # Market selection is based exclusively on initial training period.
    ranking = clean.loc[clean["date"] < dates[first_origin]].groupby("market")["units"].sum().sort_values(ascending=False)
    markets = ranking.head(cfg.markets).index.tolist()
    panel = clean.loc[clean["market"].isin(markets)].groupby(["date", "market"])["units"].sum().unstack("market")
    panel = panel.reindex(index=dates, columns=markets).fillna(0).astype(float)
    panel.index.name = "date"
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "daily_sales.csv")
    provenance = {
        "source": DATA_SOURCE, "download_url": DATA_URL, "doi": "10.24432/C5CG6D",
        "citation": "Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.",
        "license": "CC BY 4.0", "raw_sha256": sha256(path),
        "sheets": {k: len(v) for k, v in sheets.items()}, "raw_rows": len(raw),
        "audit": audit, "target": "daily gross positive merchandise units sold, excluding cancellations",
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "excluded_partial_date": str(last.date()), "markets": markets,
        "market_selection_end": str(dates[first_origin - 1].date()),
        "zero_fill_assumption": "No recorded eligible transactions means zero observed sales; outages are not identifiable.",
        "duplicate_assumption": "Exact repeated source rows are treated as accidental duplicates; no line-level transaction key exists.",
        "known_limitations": ["Sales are not unconstrained demand", "No stock, price-plan or promotion covariates", "Historical gift retailer with wholesale orders", "Final forecast is a historical continuation, not a forecast of 2026"],
        "panel_shape": list(panel.shape),
    }
    retrieval = path.parent / "retrieval.json"
    if retrieval.exists():
        provenance["retrieval"] = json.loads(retrieval.read_text(encoding="utf-8"))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    pd.DataFrame(audit).to_csv(out / "cleaning_audit.csv", index=False)
    return panel


def load_panel(root):
    panel = pd.read_csv(Path(root) / "data/processed/daily_sales.csv", index_col="date", parse_dates=True)
    expected = pd.date_range(panel.index[0], panel.index[-1], freq="D")
    if not panel.index.equals(expected) or panel.index.has_duplicates:
        raise ValueError("Panel must have a unique continuous daily calendar")
    if not np.isfinite(panel.to_numpy()).all() or (panel < 0).any().any():
        raise ValueError("Panel must contain finite nonnegative units")
    return panel
