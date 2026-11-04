"""Dedicated SDK entry for the pinned complete SuperMimic/mjlab recipe."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodiedforge.workflows.supermimic_reference_worker import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.config))
