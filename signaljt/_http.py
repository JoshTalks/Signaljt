"""Thin HTTP layer for talking to the eval API.

Only two things happen here: validating a key and turning a language code into
a signed zip URL. The signed-URL field name is not yet confirmed against a live
key, so :func:`_find_url` is deliberately tolerant.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from . import config
from .errors import ApiError, AuthenticationError

# Generous read timeout: the fetch-audios call may mint a signed URL server-side.
_TIMEOUT = httpx.Timeout(30.0, read=120.0)

# Candidate keys for the signed URL, in priority order.
_URL_KEYS = (
    "url", "signed_url", "signedUrl", "download_url", "downloadUrl",
    "zip_url", "zipUrl", "presigned_url", "presignedUrl", "href", "download",
)


def _client() -> httpx.Client:
    # follow_redirects: some signed URLs 302 to the storage backend.
    # trust_env: honour HTTPS_PROXY/NO_PROXY, which matters on locked-down HPC nets.
    return httpx.Client(timeout=_TIMEOUT, follow_redirects=True, trust_env=True)


def _headers(api_key: str) -> dict:
    return {"X-API-Key": api_key}


def validate_key(api_key: str) -> dict:
    """Best-effort key validation.

    Sends a probe language code. An explicit 401/403 means the key is bad;
    anything else (including a 400/404 for the bogus language) means auth
    passed, so we accept the key rather than block a valid login.
    """
    url = config.api_base() + "/fetch-audios/"
    with _client() as c:
        r = c.get(url, headers=_headers(api_key), params={"language_code": "__signaljt_validate__"})
    if r.status_code in (401, 403):
        raise AuthenticationError("Invalid API key.")
    return {"authenticated": True, "status_code": r.status_code}


def get_signed_url(language_code: str, api_key: str) -> dict:
    """Call fetch-audios and return ``{"url": <signed zip url>, "raw": <payload>}``."""
    url = config.api_base() + "/fetch-audios/"
    with _client() as c:
        r = c.get(url, headers=_headers(api_key), params={"language_code": language_code})
    if r.status_code in (401, 403):
        raise AuthenticationError("Invalid API key. Run `signaljt login`.")
    if r.status_code >= 400:
        raise ApiError("fetch-audios failed (%s): %s" % (r.status_code, r.text[:500]))
    try:
        data = r.json()
    except ValueError:
        # Some deployments return the bare URL as text/plain.
        text = r.text.strip()
        if text.startswith("http"):
            return {"url": text, "raw": text}
        raise ApiError("fetch-audios returned non-JSON body: %s" % text[:200])
    return {"url": _find_url(data), "raw": data}


def list_languages(api_key: str) -> list:
    """Return the supported language codes.

    The API doesn't have a dedicated listing endpoint, but calling fetch-audios
    without a language_code returns ``supported_language_codes`` in the body.
    """
    url = config.api_base() + "/fetch-audios/"
    with _client() as c:
        r = c.get(url, headers=_headers(api_key))  # omit language_code on purpose
    if r.status_code in (401, 403):
        raise AuthenticationError("Invalid API key. Run `signaljt login`.")
    try:
        data = r.json()
    except ValueError:
        raise ApiError("Unexpected non-JSON response listing languages: %s" % r.text[:200])
    codes = data.get("supported_language_codes") if isinstance(data, dict) else None
    if not codes:
        raise ApiError("Could not find supported_language_codes in response: %r" % (data,))
    return list(codes)


def submit_predictions(api_key, *, file_path=None, file_bytes=None,
                       file_name="predictions.json", predictions_url=None,
                       eval_name=None, penalize_missing=True, tag_rows=True,
                       strip_bracketed=False) -> dict:
    """POST /submit/ — returns the 202 body ({eval_id, status, ...})."""
    url = config.api_base() + "/submit/"
    timeout = httpx.Timeout(60.0, read=300.0, write=300.0, pool=60.0)
    fh = None
    with _client() as c:
        if predictions_url:
            body = {
                "predictions_url": predictions_url,
                "penalize_missing": penalize_missing,
                "tag_rows": tag_rows,
                "strip_bracketed": strip_bracketed,
            }
            if eval_name:
                body["eval_name"] = eval_name
            r = c.post(url, headers=_headers(api_key), json=body, timeout=timeout)
        else:
            data = {  # multipart form fields are strings
                "penalize_missing": str(penalize_missing).lower(),
                "tag_rows": str(tag_rows).lower(),
                "strip_bracketed": str(strip_bracketed).lower(),
            }
            if eval_name:
                data["eval_name"] = eval_name
            try:
                if file_path is not None:
                    fh = open(file_path, "rb")  # streamed, not slurped into memory
                    files = {"file": (os.path.basename(file_path), fh, "application/json")}
                elif file_bytes is not None:
                    files = {"file": (file_name, file_bytes, "application/json")}
                else:
                    raise ValueError("Provide a predictions file/bytes or predictions_url.")
                r = c.post(url, headers=_headers(api_key), files=files, data=data, timeout=timeout)
            finally:
                if fh is not None:
                    fh.close()
    if r.status_code in (401, 403):
        raise AuthenticationError("Invalid API key. Run `signaljt login`.")
    if r.status_code >= 400:
        raise ApiError("submit failed (%s): %s" % (r.status_code, r.text[:500]))
    return r.json()


def get_eval_result(api_key, *, eval_id=None, eval_name=None) -> dict:
    """GET /result/ — returns the status/result metadata."""
    if not eval_id and not eval_name:
        raise ValueError("Provide eval_id or eval_name.")
    url = config.api_base() + "/result/"
    params = {}
    if eval_id:
        params["eval_id"] = eval_id
    if eval_name:
        params["eval_name"] = eval_name
    with _client() as c:
        r = c.get(url, headers=_headers(api_key), params=params)
    if r.status_code in (401, 403):
        raise AuthenticationError("Invalid API key. Run `signaljt login`.")
    if r.status_code >= 400:
        raise ApiError("result fetch failed (%s): %s" % (r.status_code, r.text[:500]))
    return r.json()


def download_json(url) -> dict:
    """Download and parse a (signed) JSON URL — the eval result file."""
    with _client() as c:
        r = c.get(url, timeout=httpx.Timeout(30.0, read=300.0))
    if r.status_code >= 400:
        raise ApiError("result download failed (HTTP %s)" % r.status_code)
    return r.json()


def _find_url(data: Any) -> str:
    if isinstance(data, str) and data.startswith("http"):
        return data
    if isinstance(data, dict):
        for k in _URL_KEYS:
            v = data.get(k)
            if isinstance(v, str) and v.startswith("http"):
                return v
        # Look one level down (e.g. {"data": {"url": ...}}).
        for v in data.values():
            if isinstance(v, dict):
                try:
                    return _find_url(v)
                except ApiError:
                    continue
    raise ApiError("Could not locate a signed URL in the response: %r" % (data,))
