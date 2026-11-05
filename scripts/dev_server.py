"""Run the coordinator locally on loopback with the fake runtime.

The database defaults to ~/.local/share/difa/dev.sqlite3 so it stays out of synced folders.
Usage: python scripts/dev_server.py [--port 8765] [--fresh]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "coordinator"))

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--fresh", action="store_true", help="start from an empty development database")
args = parser.parse_args()

data = Path(os.environ.get("DIFA_DATA_DIR", Path.home() / ".local" / "share" / "difa"))
data.mkdir(parents=True, exist_ok=True)
db = data / "dev.sqlite3"
if args.fresh:
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
os.environ.setdefault("DIFA_DB", str(db))

import uvicorn  # noqa: E402

uvicorn.run("difa.api.app:app_from_env", factory=True, host="127.0.0.1", port=args.port, log_level="info")
