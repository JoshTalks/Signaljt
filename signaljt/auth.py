"""Session-style authentication, modelled on huggingface_hub.

There is no server-side session: the API authenticates every request with a
long-lived ``X-API-Key``. "Logging in" captures that key once, stores it with
0600 permissions, and every later call resolves it transparently.

Resolution precedence (highest first):
    1. explicit ``api_key=...`` argument
    2. ``SIGNALJT_API_KEY`` environment variable
    3. saved token file (``~/.cache/signaljt/token``)
"""
from __future__ import annotations

import getpass
import os
from typing import Optional

from . import config
from .errors import AuthenticationError, NotLoggedInError


def _write_token(token: str) -> str:
    """Atomically write the token with owner-only permissions."""
    path = config.token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    # O_CREAT with mode 0o600 so the secret is never briefly world-readable.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    os.replace(tmp, str(path))  # atomic; a killed login never leaves a half file
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass
    return str(path)


def _read_token_file() -> Optional[str]:
    try:
        with open(config.token_path(), "r", encoding="utf-8") as f:
            tok = f.read().strip()
        return tok or None
    except FileNotFoundError:
        return None


def get_api_key(api_key: Optional[str] = None, *, required: bool = True) -> Optional[str]:
    """Resolve the API key using the documented precedence."""
    if api_key:
        return api_key.strip()
    env = os.environ.get(config.ENV_API_KEY)
    if env and env.strip():
        return env.strip()
    tok = _read_token_file()
    if tok:
        return tok
    if required:
        raise NotLoggedInError(
            "No API key found. Run `signaljt login`, set the SIGNALJT_API_KEY "
            "environment variable, or pass api_key=... explicitly."
        )
    return None


def login(api_key: Optional[str] = None, *, validate: bool = True) -> None:
    """Store an API key for future calls.

    Works inline in scripts (``signaljt.login("...")``) or interactively
    (``signaljt.login()`` prompts without echoing).
    """
    if not api_key:
        api_key = getpass.getpass("Enter your Signal API key: ")
    api_key = (api_key or "").strip()
    if not api_key:
        raise AuthenticationError("Empty API key.")
    if validate:
        from ._http import validate_key
        validate_key(api_key)
    path = _write_token(api_key)
    print("Login successful. Token saved to %s" % path)


def logout() -> None:
    """Delete the saved token, if any."""
    path = config.token_path()
    try:
        os.remove(str(path))
        print("Logged out. Removed %s" % path)
    except FileNotFoundError:
        print("Not logged in; nothing to remove.")


def whoami(api_key: Optional[str] = None) -> dict:
    """Validate the resolved key against the API."""
    from ._http import validate_key
    key = get_api_key(api_key)
    info = validate_key(key)
    return info
