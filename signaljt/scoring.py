"""OI-WER scoring: submit predictions, poll, and parse the result.

Flow: submit -> queued -> (poll) -> completed -> download result JSON.

The library route returns a dot-accessible :class:`EvalResult` (everything is
reachable by ``.``); the CLI route prints the signed result URL. A small
:meth:`EvalResult.metrics` helper produces a flat ``{name: scalar}`` dict ready
for ``wandb.log`` or TensorBoard ``add_scalar`` — overall and per-language.
"""
from __future__ import annotations

import json
import sys
import time
from typing import List, Optional

from . import _http
from .auth import get_api_key
from .errors import SignaljtError


def _is_url(s) -> bool:
    return isinstance(s, str) and s.startswith(("http://", "https://"))


class Box(dict):
    """A dict whose keys are also attributes.

    Nested dicts (and lists of dicts) are wrapped lazily on first access and
    memoized, so building a Box over a large payload is cheap — you only pay to
    wrap the parts you actually touch.
    """

    def __getitem__(self, key):
        val = super().__getitem__(key)
        wrapped = _boxify(val)
        if wrapped is not val and isinstance(key, str):
            super().__setitem__(key, wrapped)  # memoize
        return wrapped

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())


def _boxify(val):
    if isinstance(val, Box):
        return val
    if isinstance(val, dict):
        return Box(val)
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return [_boxify(v) for v in val]
    return val


class EvalResult:
    """Parsed OI-WER result — everything is dot-accessible.

        r.oiwer                       # weighted overall OI-WER
        r.overall.macro_oiwer
        r.overall.word.errors
        r.per_language["hi"].oiwer    # by ISO code
        r.per_language.hi.coverage.missing
        r.per_chunk[0].alignment.ops
        r.meta.penalize_missing
    """

    def __init__(self, data, *, eval_id=None, eval_name=None, result_url=None):
        self._data = data
        self.eval_id = eval_id
        self.eval_name = eval_name
        self.result_url = result_url
        self.overall = Box(data.get("overall", {}))
        self.meta = Box(data.get("meta", {}))
        # Re-key per_language by ISO code (matches fetch_dataset language codes),
        # keeping the full name available as .name.
        pl = {}
        for name, v in data.get("per_language", {}).items():
            box = Box(dict(v))
            box["name"] = name
            pl[v.get("iso", name)] = box
        self.per_language = Box(pl)
        self._per_chunk = None  # 14k+ rows: wrap lazily

    @property
    def per_chunk(self) -> List[Box]:
        if self._per_chunk is None:
            self._per_chunk = [Box(c) for c in self._data.get("per_chunk", [])]
        return self._per_chunk

    @property
    def oiwer(self):
        """Headline number: weighted overall OI-WER."""
        return self.overall.get("weighted_oiwer")

    def metrics(self, *, per_language=True, words=True, prefix="oiwer") -> dict:
        """Flat ``{name: scalar}`` dict for experiment loggers.

            wandb.log(result.metrics())
            for k, v in result.metrics().items():
                writer.add_scalar(k, v, step)

        Slash-namespaced so both wandb and TensorBoard group them (overall under
        ``oiwer/*`` and ``words/*``, per language under ``oiwer_by_language/*``).
        """
        m = {}
        o = self.overall
        if o.get("weighted_oiwer") is not None:
            m["%s/weighted" % prefix] = o["weighted_oiwer"]
        if o.get("macro_oiwer") is not None:
            m["%s/macro" % prefix] = o["macro_oiwer"]
        if words:
            w = o.get("word") or {}
            for k in ("errors", "sub", "ins", "del", "count"):
                if k in w:
                    m["words/%s" % k] = w[k]
        if per_language:
            for iso, lb in self.per_language.items():
                if lb.get("oiwer") is not None:
                    m["%s_by_language/%s" % (prefix, iso)] = lb["oiwer"]
        return m

    def to_dict(self) -> dict:
        return self._data

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    def __repr__(self):
        return "EvalResult(eval_id=%r, oiwer=%s, languages=%d, utterances=%s)" % (
            self.eval_id, self.oiwer, len(self.per_language),
            self.overall.get("total_utterances"),
        )


class EvalJob:
    """A submitted eval you can poll. Returned by :func:`submit`."""

    def __init__(self, eval_id=None, eval_name=None, status=None, *, api_key=None):
        self.eval_id = eval_id
        self.eval_name = eval_name
        self.status = status
        self._key = api_key or get_api_key()

    def refresh(self) -> dict:
        """One status poll; updates ``self.status`` and returns the raw metadata."""
        meta = _http.get_eval_result(self._key, eval_id=self.eval_id, eval_name=self.eval_name)
        self.status = meta.get("status", self.status)
        if not self.eval_id:
            self.eval_id = meta.get("eval_id")
        return meta

    def wait(self, *, poll_interval=3.0, timeout=1800.0, progress=False) -> EvalResult:
        """Poll until completed, then download + parse the result.

        Synchronous (blocks). ``progress`` is off by default for library use;
        when on, progress lines go to stderr so stdout stays clean.
        """
        waited = 0.0
        while True:
            meta = self.refresh()
            st = meta.get("status")
            if st == "completed":
                if progress:
                    print("eval %s completed" % self.eval_id, file=sys.stderr)
                return _parse_result(meta, self.eval_id, self.eval_name)
            if st == "failed":
                raise SignaljtError("Eval %s failed: %s" % (self.eval_id, meta.get("error")))
            if waited >= timeout:
                raise SignaljtError(
                    "Eval %s not finished after %.0fs (status=%s)"
                    % (self.eval_id, timeout, st))
            if progress:
                print("eval %s: %s ..." % (self.eval_id, st), file=sys.stderr)
            time.sleep(poll_interval)
            waited += poll_interval

    def result(self) -> EvalResult:
        """Fetch the result now; errors if the eval isn't completed yet."""
        meta = self.refresh()
        if meta.get("status") != "completed":
            raise SignaljtError(
                "Eval %s not completed (status=%s); use wait()." % (self.eval_id, self.status))
        return _parse_result(meta, self.eval_id, self.eval_name)


def _parse_result(meta, eval_id, eval_name) -> EvalResult:
    url = meta["result_file_url"]
    data = _http.download_json(url)
    return EvalResult(data, eval_id=eval_id or meta.get("eval_id"),
                      eval_name=eval_name or meta.get("eval_name"), result_url=url)


def submit(predictions=None, *, predictions_url=None, eval_name=None,
           penalize_missing=True, tag_rows=True, strip_bracketed=False,
           api_key=None) -> EvalJob:
    """Submit predictions for scoring; returns an :class:`EvalJob` (does not wait).

    ``predictions`` may be:
      * a file path (``"preds.json"`` / ``.jsonl``) — uploaded (streamed)
      * an http(s) URL — passed as ``predictions_url`` (no upload)
      * an in-memory ``dict`` (``{index: hypothesis}``) or ``list`` — serialized
        and uploaded
    """
    key = get_api_key(api_key)
    kw = dict(eval_name=eval_name, penalize_missing=penalize_missing,
              tag_rows=tag_rows, strip_bracketed=strip_bracketed)
    if predictions_url or _is_url(predictions):
        resp = _http.submit_predictions(key, predictions_url=predictions_url or predictions, **kw)
    elif isinstance(predictions, (dict, list)):
        blob = json.dumps(predictions, ensure_ascii=False).encode("utf-8")
        resp = _http.submit_predictions(key, file_bytes=blob, **kw)
    elif isinstance(predictions, str):
        resp = _http.submit_predictions(key, file_path=predictions, **kw)
    else:
        raise ValueError("predictions must be a file path, URL, dict, or list.")
    return EvalJob(resp.get("eval_id"), resp.get("eval_name"), resp.get("status"), api_key=key)


def evaluate(predictions=None, *, predictions_url=None, eval_name=None,
             penalize_missing=True, tag_rows=True, strip_bracketed=False,
             api_key=None, poll_interval=3.0, timeout=1800.0, progress=False) -> EvalResult:
    """Submit, wait for completion, and return the parsed :class:`EvalResult`.

    Synchronous — blocks until scoring finishes and returns the result. Silent
    by default; pass ``progress=True`` to print poll status to stderr.
    """
    job = submit(predictions, predictions_url=predictions_url, eval_name=eval_name,
                 penalize_missing=penalize_missing, tag_rows=tag_rows,
                 strip_bracketed=strip_bracketed, api_key=api_key)
    return job.wait(poll_interval=poll_interval, timeout=timeout, progress=progress)


def get_result(eval_id=None, *, eval_name=None, api_key=None,
               wait=False, poll_interval=3.0, timeout=1800.0) -> EvalResult:
    """Fetch a completed eval's result by ``eval_id`` (or ``eval_name``)."""
    job = EvalJob(eval_id, eval_name, api_key=api_key)
    return job.wait(poll_interval=poll_interval, timeout=timeout) if wait else job.result()