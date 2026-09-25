"""Load settings from a local .env file (KEY=value lines) into the environment.

Variables already set in the shell win, and empty values are skipped, so the
blank entries copied from .env.example never mask real settings.

With no path, it reads ./.env and then the .env at the repository root (next to
the lha package), so commands started from another folder still find it.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: str | Path | None = None) -> None:
    if path is None:
        for p in dict.fromkeys([Path(".env").resolve(), REPO_ROOT / ".env"]):
            load_dotenv(p)
        return
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if value and key not in os.environ:
            os.environ[key] = value
