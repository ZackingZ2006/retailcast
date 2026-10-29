import argparse
from pathlib import Path

from .config import ROOT, Config


def main():
    p = argparse.ArgumentParser(description="RetailCast forecasting pipeline")
    p.add_argument("command", choices=["run", "prepare"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--refresh", action="store_true", help="Rebuild processed data from raw workbook")
    args = p.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args.root, Config())
    else:
        from .pipeline import run
        run(args.root, Config(), refresh=args.refresh)


if __name__ == "__main__":
    main()
