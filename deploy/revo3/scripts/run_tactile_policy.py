#!/usr/bin/env python3
"""Run the installed Revo3 physical-taxel tactile policy CLI.

Overview:
This wrapper forwards all options to ``revo3_deploy.cli.run_tactile_policy``.

Quick Start:
    python scripts/run_tactile_policy.py --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from revo3_deploy.cli.run_tactile_policy import main


if __name__ == "__main__":
    raise SystemExit(main())
