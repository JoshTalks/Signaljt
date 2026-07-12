"""Optimized, resumable, integrity-checked file download.

The transfer is I/O-bound, so the win is overlapping it: split the object into
byte ranges and pull them concurrently, each writing to its own offset in a
pre-allocated file. Falls back to a single stream when the server doesn't
advertise range support or the file is too small to bother.

Three durability features layer on top:
  * integrity  - verify the assembled file against the server's MD5 (GCS
                 x-goog-hash); fall back to a size check when absent.
  * versioning - expose a stable object "identity" (generation/md5/etag/size)
                 so callers can detect a re-published object.
  * resume     - a ".signaljtpart" sidecar records completed segments, so a killed
                 download continues instead of restarting.

Note: we never send HTTP HEAD. S3/GCS presigned URLs are signed for a specific
method, so a GET-signed URL 403s on HEAD; we probe with a 1-byte ranged GET.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import os
import threading
import time
from collections import namedtuple
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import httpx

from .errors import DownloadError

# Overridable defaults (every one is a parameter on download_file / fetch_dataset).
DEFAULT_CONNECTIONS = 1           # single stream by default; raise it for high-latency links
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB streaming chunk
DEFAULT_MIN_SEGMENT = 8 * 1024 * 1024    # don't split below this per segment
DEFAULT_MAX_SEGMENT = 64 * 1024 * 1024   # cap segment size so resume loses <= this
DEFAULT_RETRIES = 4
_PART_SUFFIX = ".signaljtpart"

ProgressCb = Optional[Callable[[int, Optional[int]], None]]

# What a probe tells us about the remote object.
RemoteInfo = namedtuple("RemoteInfo", "size supports_range identity md5 etag generation")
# What a completed download reports back.
DownloadResult = namedtuple("DownloadResult", "path size identity md5")


class _Progress:
    """Thread-safe progress aggregator; throttles printing to ~5% steps."""

    def __init__(self, total: Optional[int], cb):
        self.total = total
        self.done = 0
        self.cb = cb
        self._lock = threading.Lock()
        self._last_pct = -1

    def add(self, n: int) -> None:
        if not self.cb or n == 0:
            return
        with self._lock:
            self.done += n
            if callable(self.cb):
                self.cb(self.done, self.total)
            elif self.total:
                pct = int(self.done * 100 / self.total)
                if pct >= self._last_pct + 5 or pct >= 100:
                    self._last_pct = pct
                    mb, tot = self.done / 1048576, self.total / 1048576
                    print("  downloading: %3d%%  (%.1f / %.1f MiB)" % (pct, mb, tot))


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(30.0, read=300.0),
        follow_redirects=True,
        trust_env=True,
    )


def _parse_goog_hash(header: str) -> dict:
    """Parse ``x-goog-hash: crc32c=..,md5=..`` into a dict."""
    out = {}
    for part in header.split(","):
        part = part.strip()
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def probe(client: httpx.Client, url: str) -> RemoteInfo:
    """Learn size, range support, and a stable identity without downloading."""
    with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as r:
        if r.status_code == 206:
            cr = r.headers.get("Content-Range", "")
            size = int(cr.split("/")[-1]) if "/" in cr else None
            supports = True
        elif r.status_code == 200:
            cl = r.headers.get("Content-Length")
            size = int(cl) if cl else None
            supports = False
        else:
            raise DownloadError("Probe failed (HTTP %s) for the signed URL." % r.status_code)

        md5 = _parse_goog_hash(r.headers.get("x-goog-hash", "")).get("md5")
        etag = r.headers.get("ETag")
        generation = r.headers.get("x-goog-generation")
        # Prefer the most content-specific stable identifier available.
        identity = generation or md5 or etag or (("size:%d" % size) if size else None)
        return RemoteInfo(size, supports, identity, md5, etag, generation)


def _split(total: int, n: int) -> List[Tuple[int, int]]:
    seg = total // n
    bounds, start = [], 0
    for i in range(n):
        end = total - 1 if i == n - 1 else start + seg - 1
        bounds.append((start, end))
        start = end + 1
    return bounds


def _plan_segments(total: int, connections: int, min_segment: int, max_segment: int) -> List[Tuple[int, int]]:
    """Segment count decoupled from parallelism: pick a per-segment size in
    [min_segment, max_segment], so huge files get many small segments (fine
    resume granularity) while the pool still runs only ``connections`` at a time."""
    seg = max(min_segment, min(total // max(connections, 1) or total, max_segment))
    n = max(1, (total + seg - 1) // seg)
    return _split(total, n)


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def _download_segment(client, url, dest, start, end, prog: _Progress, chunk_size, retries) -> None:
    expected = end - start + 1
    attempt = 0
    while True:
        got = 0
        try:
            with client.stream("GET", url, headers={"Range": "bytes=%d-%d" % (start, end)}) as r:
                if r.status_code not in (206, 200):
                    raise DownloadError("segment HTTP %s" % r.status_code)
                with open(dest, "r+b") as f:
                    f.seek(start)
                    for chunk in r.iter_bytes(chunk_size):
                        f.write(chunk)
                        got += len(chunk)
                        prog.add(len(chunk))
            if got != expected:
                raise DownloadError("short read %d != %d" % (got, expected))
            return
        except (httpx.HTTPError, DownloadError) as e:
            prog.add(-got)  # don't double-count retried bytes
            attempt += 1
            if attempt > retries:
                raise DownloadError(
                    "segment %d-%d failed after %d attempts: %s" % (start, end, retries, e)
                )
            time.sleep(min(2 ** attempt * 0.25, 5.0))  # capped exponential backoff


def _download_single(client, url, dest, prog: _Progress, chunk_size) -> None:
    with client.stream("GET", url) as r:
        if r.status_code >= 400:
            raise DownloadError("download HTTP %s" % r.status_code)
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size):
                f.write(chunk)
                prog.add(len(chunk))


def _sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + _PART_SUFFIX)


def _read_sidecar(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def _write_sidecar(path: Path, meta: dict) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, str(path))


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _download_single_fast(client, url, dest, progress, chunk_size, verify) -> DownloadResult:
    """Single-stream path for ``connections <= 1``.

    Skips the probe entirely (size/identity/md5 come from the download response's
    own headers) and, when verifying, hashes inline as bytes are written — so
    there's no second read pass. One round-trip instead of two.
    """
    h = hashlib.md5() if verify else None
    prog = _Progress(None, progress)
    with client.stream("GET", url) as r:
        if r.status_code >= 400:
            raise DownloadError("download HTTP %s" % r.status_code)
        md5 = _parse_goog_hash(r.headers.get("x-goog-hash", "")).get("md5")
        etag = r.headers.get("ETag")
        generation = r.headers.get("x-goog-generation")
        cl = r.headers.get("Content-Length")
        size = int(cl) if cl else None
        identity = generation or md5 or etag or (("size:%d" % size) if size else None)
        prog.total = size
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size):
                f.write(chunk)
                if h is not None:
                    h.update(chunk)
                prog.add(len(chunk))
    actual = dest.stat().st_size
    if size and actual != size:
        raise DownloadError("size mismatch: got %d, expected %d" % (actual, size))
    if verify and md5:
        local = base64.b64encode(h.digest()).decode()
        if local != md5:
            raise DownloadError("checksum mismatch: local md5 %s != remote %s" % (local, md5))
    return DownloadResult(dest, actual, identity, md5)


def _finalize(dest: Path, info: RemoteInfo, verify: bool) -> DownloadResult:
    actual = dest.stat().st_size
    if info.size and actual != info.size:
        raise DownloadError("size mismatch: got %d, expected %d" % (actual, info.size))
    if verify and info.md5:
        local = _md5_b64(dest)
        if local != info.md5:
            raise DownloadError("checksum mismatch: local md5 %s != remote %s" % (local, info.md5))
    return DownloadResult(dest, actual, info.identity, info.md5)


def download_file(
    url: str,
    dest,
    *,
    connections: int = DEFAULT_CONNECTIONS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_segment: int = DEFAULT_MIN_SEGMENT,
    max_segment: int = DEFAULT_MAX_SEGMENT,
    retries: int = DEFAULT_RETRIES,
    progress: ProgressCb = None,
    resume: bool = True,
    verify: bool = True,
    client: Optional[httpx.Client] = None,
) -> DownloadResult:
    """Download ``url`` to ``dest``; return a :class:`DownloadResult`.

    ``connections`` parallel ranged streams (1 = single stream). ``chunk_size``
    is the streaming buffer; ``min_segment``/``max_segment`` bound per-segment
    size; ``retries`` is the per-segment attempt budget. ``resume`` continues a
    prior partial download; ``verify`` checks the assembled file against the
    server MD5; ``progress`` is ``None`` / ``True`` / a callable.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    side = _sidecar(dest)
    own = client is None
    client = client or _client()
    try:
        # Single stream: skip the probe (one round-trip) and hash inline.
        if connections <= 1:
            result = _download_single_fast(client, url, dest, progress, chunk_size, verify)
            _safe_unlink(side)
            return result

        info = probe(client, url)
        prog = _Progress(info.size, progress)

        # No-range or too-small to split -> one stream, no resume state.
        if not info.supports_range or not info.size or info.size < min_segment:
            _download_single(client, url, dest, prog, chunk_size)
            result = _finalize(dest, info, verify)
            _safe_unlink(side)
            return result

        # Decide fresh vs resume by matching the on-disk sidecar to the object.
        meta = _read_sidecar(side) if resume else None
        can_resume = bool(
            meta
            and dest.exists()
            and meta.get("size") == info.size
            and meta.get("identity") == info.identity
            and dest.stat().st_size == info.size
        )
        if can_resume:
            segments = [tuple(s) for s in meta["segments"]]
            done: Set[Tuple[int, int]] = {tuple(s) for s in meta.get("done", [])}
        else:
            segments = _plan_segments(info.size, connections, min_segment, max_segment)
            done = set()
            with open(dest, "wb") as f:  # pre-allocate for seek+write
                f.truncate(info.size)

        for (s, e) in done:  # account for already-downloaded bytes in progress
            prog.add(e - s + 1)

        meta = {
            "size": info.size,
            "identity": info.identity,
            "segments": [list(s) for s in segments],
            "done": [list(s) for s in done],
        }
        _write_sidecar(side, meta)
        remaining = [seg for seg in segments if seg not in done]
        sidecar_lock = threading.Lock()

        def run_segment(seg):
            _download_segment(client, url, dest, seg[0], seg[1], prog, chunk_size, retries)
            with sidecar_lock:  # persist progress so a kill here is resumable
                done.add(seg)
                meta["done"] = [list(s) for s in done]
                _write_sidecar(side, meta)

        errors = []
        workers = min(connections, len(remaining)) or 1
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_segment, seg): seg for seg in remaining}
            for fut in cf.as_completed(futs):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - surface first failure
                    errors.append(exc)
        if errors:
            raise errors[0]

        result = _finalize(dest, info, verify)
        _safe_unlink(side)  # only on success; a failure keeps it for resume
        return result
    finally:
        if own:
            client.close()