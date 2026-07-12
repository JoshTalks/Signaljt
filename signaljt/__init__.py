"""signaljt — client for Signal (by Josh Talks), the OI-WER evaluation API.

Datasets: authenticated, cache-safe, parallel download.
Scoring:  submit predictions, get overall + per-language + per-chunk OI-WER.

    import signaljt
    signaljt.login()                       # or set SIGNALJT_API_KEY

    ds = signaljt.fetch_dataset("bn")      # download ground-truth audio
    for item in ds:
        item.index, item.audio_path        # absolute local path, ready to use

    result = signaljt.evaluate("predictions.json")   # score your model
    result.oiwer                           # weighted overall OI-WER
    wandb.log(result.metrics())            # overall + per-language, for logging
"""
from __future__ import annotations

__version__ = "0.1.1"

from .auth import get_api_key, login, logout, whoami
from .dataset import Dataset, DatasetItem, fetch_dataset, list_languages
from .scoring import EvalJob, EvalResult, evaluate, get_result, is_main_process, submit
from .errors import (
    ApiError,
    AuthenticationError,
    DownloadError,
    ManifestError,
    NotLoggedInError,
    SignaljtError,
)

__all__ = [
    "__version__",
    "login",
    "logout",
    "whoami",
    "get_api_key",
    "notebook_login",
    "fetch_dataset",
    "list_languages",
    "submit",
    "evaluate",
    "get_result",
    "is_main_process",
    "EvalJob",
    "EvalResult",
    "Dataset",
    "DatasetItem",
    "SignaljtError",
    "NotLoggedInError",
    "AuthenticationError",
    "ApiError",
    "DownloadError",
    "ManifestError",
]


def notebook_login(validate: bool = True) -> None:
    """Lazy wrapper so importing signaljt doesn't require ipywidgets."""
    from .notebook import notebook_login as _nl
    return _nl(validate=validate)