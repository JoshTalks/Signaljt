"""Paths, environment variables, and endpoint configuration.

Everything the rest of the package needs to know about *where* things live is
resolved here, so HPC users can redirect the cache with a single env var.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "signaljt"

ENV_API_KEY = "SIGNALJT_API_KEY"          # non-interactive credential (CI / Slurm batch)
ENV_CACHE_DIR = "SIGNALJT_CACHE_DIR"      # override the cache root (e.g. $SLURM_TMPDIR)
ENV_API_BASE = "SIGNALJT_API_BASE"        # override the API base URL (staging/testing)

DEFAULT_API_BASE = "https://data-collection-app.joshtalks.org/api/eval"


def cache_dir() -> Path:
    """Root cache directory.

    Precedence: SIGNALJT_CACHE_DIR > XDG_CACHE_HOME/signaljt > ~/.cache/signaljt.

    On HPC this lives on shared home by default, so a login-node ``signaljt login``
    and the first download are visible to every compute node. Point
    SIGNALJT_CACHE_DIR at node-local scratch ($SLURM_TMPDIR) for per-job speed.
    """
    override = os.environ.get(ENV_CACHE_DIR)
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / APP_NAME


def token_path() -> Path:
    return cache_dir() / "token"


def datasets_dir() -> Path:
    return cache_dir() / "datasets"


def api_base() -> str:
    return os.environ.get(ENV_API_BASE, DEFAULT_API_BASE).rstrip("/")
