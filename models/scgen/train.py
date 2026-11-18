"""Binary scGen preparation and training commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models._native_execution import train

if __name__ == "__main__":
    raise SystemExit(train("scgen"))
