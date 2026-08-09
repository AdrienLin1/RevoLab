#!/usr/bin/env python3
"""Print real Revo3 joint angles and numbered fingertip sensor values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from revo3_deploy.cli.inspect_hand import main


if __name__ == "__main__":
    raise SystemExit(main())
