"""Shared Stage 4 paths and options; reuses the Stage 2 match selection and raw StatsBomb data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from experiments.stage1.common import PALETTE, INK, INK_2, GRID, mpl  # noqa: E402
from experiments.stage2.common import selected_matches  # noqa: E402
from football import statsbomb  # noqa: E402
from football.dynamics import FIXED_WINDOWS, dynamic_ppr, substitution_windows  # noqa: E402

ALPHAS = (0.85, 0.99)
MODES = ("fixed", "substitutions")


def parse_args(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--selection", type=Path, default=ROOT / "experiments" / "stage2" / "selected_matches.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage4")
    parser.add_argument("--quick", action="store_true", help="five matches spanning all four selected competitions")
    parser.add_argument("--tol", type=float, default=1e-10, help="power-iteration step tolerance")
    args = parser.parse_args()
    args.data = statsbomb.data_root(args.data)
    args.output = args.output.resolve()
    (args.output / "figures").mkdir(parents=True, exist_ok=True)
    return args


def save_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    mpl().close(fig)
