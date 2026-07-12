"""Download → cache → extract → manifest, exposed as a friendly Dataset object.

Concurrency-safe by design: a file lock plus atomic publish means a Slurm array
of N tasks all calling ``fetch_dataset("bn")`` results in exactly one download,
with the rest blocking on the lock and then reading from cache. A killed job
never leaves a half-extracted directory that looks complete.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from filelock import FileLock

from . import config
from ._http import get_signed_url, list_languages as _list_languages
from .auth import get_api_key
from .download import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CONNECTIONS, DEFAULT_MAX_SEGMENT,
    DEFAULT_MIN_SEGMENT, DEFAULT_RETRIES,
    _client, _safe_unlink, _sidecar, download_file, probe,
)
from .errors import ManifestError, SignaljtError

# The API ships one manifest per zip, named "<Language>_manifest.jsonl".
_MANIFEST_GLOB = "*_manifest.jsonl"
_COMPLETE_FLAG = ".signaljt_complete"
_META_NAME = ".signaljt_meta.json"    # remote identity of the cached zip

# Tolerant field detection for the manifest.
_INDEX_KEYS = ("Index", "index", "id", "ID", "utt_id", "utterance_id")
_PATH_KEYS = ("audio_filepath", "audio_path", "path", "relative_path", "rel_path",
              "audio", "file", "filepath", "wav")


class DatasetItem:
    """One manifest row: an index and a ready-to-use absolute audio path."""

    __slots__ = ("index", "audio_path", "extra")

    def __init__(self, index: str, audio_path: str, extra: Optional[Dict[str, Any]] = None):
        self.index = index
        self.audio_path = audio_path
        self.extra = extra or {}

    def __repr__(self) -> str:
        return "DatasetItem(index=%r, audio_path=%r)" % (self.index, self.audio_path)


class Dataset:
    """A downloaded dataset. Iterable, indexable by position or by Index."""

    def __init__(self, language_code: str, root: Path, items: List[DatasetItem]):
        self.language_code = language_code
        self.root = Path(root)
        self.items = items
        self._by_index = {it.index: it for it in items}

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[DatasetItem]:
        return iter(self.items)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.items[key]
        return self._by_index[key]

    def __contains__(self, index) -> bool:
        return index in self._by_index

    def get(self, index, default=None):
        return self._by_index.get(index, default)

    @property
    def indices(self) -> List[str]:
        return [it.index for it in self.items]

    def paths(self) -> Dict[str, str]:
        """Mapping of ``index -> absolute audio path``."""
        return {it.index: it.audio_path for it in self.items}

    def export(self, local_dir, *, mode: str = "auto") -> "Dataset":
        """Place this dataset's files into ``local_dir`` (see ``fetch_dataset``)."""
        if mode not in _PLACE_MODES:
            raise ValueError("mode must be one of %s" % (_PLACE_MODES,))
        dest_root = _place_into(self.root, local_dir, mode, progress=False)
        return Dataset(self.language_code, dest_root, _load_manifest(dest_root))

    def __repr__(self) -> str:
        return "Dataset(language_code=%r, items=%d, root=%r)" % (
            self.language_code, len(self.items), str(self.root),
        )


def _extract_parallel(zip_path: Path, dest: Path, workers: int = 1) -> None:
    """Extract a zip using a thread pool.

    Each worker opens its own ``ZipFile`` (the object isn't safe to share across
    threads). FLAC is already compressed, so this is I/O-bound file writing —
    exactly what threads overlap well.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]

    if not names:
        raise ManifestError("Downloaded zip is empty.")

    n = max(1, min(workers, len(names)))
    buckets: List[List[str]] = [[] for _ in range(n)]
    for i, name in enumerate(names):
        buckets[i % n].append(name)

    def worker(sublist: List[str]) -> None:
        with zipfile.ZipFile(zip_path) as zf:  # zf.extract sanitizes paths
            for name in sublist:
                zf.extract(name, dest)

    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(worker, b) for b in buckets if b]
        for fut in cf.as_completed(futs):
            fut.result()


def _pick(obj: Dict[str, Any], keys) -> Optional[str]:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _find_manifest(root: Path) -> Path:
    # Prefer the shallowest "*_manifest.jsonl"; fall back to any .jsonl.
    matches = list(root.rglob(_MANIFEST_GLOB)) or list(root.rglob("*.jsonl"))
    if not matches:
        raise ManifestError("No *_manifest.jsonl found under %s" % root)
    return min(matches, key=lambda p: len(p.parts))


def _load_manifest(root: Path) -> List[DatasetItem]:
    manifest = _find_manifest(root)
    base = manifest.parent  # relative paths are relative to the manifest
    items: List[DatasetItem] = []
    with open(manifest, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError("Bad JSON on line %d of manifest: %s" % (lineno, e))
            index = _pick(obj, _INDEX_KEYS)
            rel = _pick(obj, _PATH_KEYS)
            if index is None or rel is None:
                raise ManifestError(
                    "Manifest line %d missing index/path (keys: %s)" % (lineno, list(obj))
                )
            audio_abs = str((base / rel).resolve())
            extra = {k: v for k, v in obj.items()
                     if k not in _INDEX_KEYS and k not in _PATH_KEYS}
            items.append(DatasetItem(str(index), audio_abs, extra))
    if not items:
        raise ManifestError("Manifest %s has no rows." % manifest)
    return items


_PLACE_MODES = ("auto", "copy", "symlink", "hardlink")


def fetch_dataset(
    language,
    *,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    local_dir_mode: str = "auto",
    connections: int = DEFAULT_CONNECTIONS,
    extract_workers: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_segment: int = DEFAULT_MIN_SEGMENT,
    max_segment: int = DEFAULT_MAX_SEGMENT,
    retries: int = DEFAULT_RETRIES,
    force: bool = False,
    refresh: bool = False,
    max_parallel: int = 4,
    continue_on_error: bool = True,
    progress=True,
):
    """Download one language, several, or all.

    ``language`` may be:
      * a single code (``"bn"``)  -> returns a :class:`Dataset`
      * ``"all"``                 -> every supported language -> ``{code: Dataset}``
      * a list/tuple of codes     -> those languages          -> ``{code: Dataset}``

    Multiple languages download concurrently (``max_parallel``); each keeps its
    own cache/lock/resume, and with ``local_dir`` each lands under
    ``local_dir/<code>``.

    Performance knobs default to conservative single-threaded values; raise them
    for large files / high-latency links: ``connections`` parallel ranged
    streams, ``extract_workers`` unzip threads, ``chunk_size`` streaming buffer,
    ``min_segment``/``max_segment`` segment sizing, ``retries`` per segment.
    ``force`` always re-downloads; ``refresh`` re-downloads only if the remote
    object changed.
    """
    if local_dir_mode not in _PLACE_MODES:
        raise ValueError("local_dir_mode must be one of %s" % (_PLACE_MODES,))

    common = dict(
        cache_dir=cache_dir, local_dir_mode=local_dir_mode, connections=connections,
        extract_workers=extract_workers, chunk_size=chunk_size, min_segment=min_segment,
        max_segment=max_segment, retries=retries, force=force, refresh=refresh,
    )

    # Single language -> a Dataset.
    if isinstance(language, str) and language != "all":
        return _fetch_one(language, api_key=api_key, local_dir=local_dir,
                          progress=progress, **common)

    # "all" or an explicit list -> {code: Dataset}, downloaded concurrently.
    key = get_api_key(api_key)
    codes = _list_languages(key) if language == "all" else list(language)
    results: Dict[str, Dataset] = {}
    errors: Dict[str, Exception] = {}
    total, done = len(codes), 0

    def one(code):
        ld = os.path.join(local_dir, code) if local_dir else None
        return _fetch_one(code, api_key=key, local_dir=ld, progress=False, **common)

    with cf.ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        futs = {ex.submit(one, c): c for c in codes}
        for fut in cf.as_completed(futs):
            code = futs[fut]
            done += 1
            try:
                results[code] = fut.result()
                if progress:
                    print("[%d/%d] %s: %d items" % (done, total, code, len(results[code])))
            except Exception as e:  # noqa: BLE001 - collect per-language failures
                errors[code] = e
                if progress:
                    print("[%d/%d] %s: FAILED (%s)" % (done, total, code, e))
                if not continue_on_error:
                    raise
    if errors:
        print("fetch_dataset: %d/%d succeeded; failed: %s" % (len(results), total, list(errors)))
    return results


def _fetch_one(
    language_code: str,
    *,
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_dir: Optional[str] = None,
    local_dir_mode: str = "auto",
    connections: int = DEFAULT_CONNECTIONS,
    extract_workers: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_segment: int = DEFAULT_MIN_SEGMENT,
    max_segment: int = DEFAULT_MAX_SEGMENT,
    retries: int = DEFAULT_RETRIES,
    force: bool = False,
    refresh: bool = False,
    progress=True,
) -> Dataset:
    """Download (if needed), extract, and return a :class:`Dataset`.

    Cached by language code under the cache root; subsequent calls (including
    from other processes/nodes) return instantly. Point ``cache_dir`` or
    ``SIGNALJT_CACHE_DIR`` at node-local scratch for per-job speed.

    ``local_dir`` also places the files into a directory you choose (HF-style),
    without re-downloading. ``local_dir_mode`` controls how:
      * ``"auto"``    - hardlink if same filesystem (instant, no extra disk),
                        else copy. The safe default.
      * ``"copy"``    - always a real independent copy.
      * ``"symlink"`` - symlink back into the cache (breaks if cache is cleaned).
      * ``"hardlink"``- force hardlink (errors across filesystems).

    Performance knobs default to conservative single-threaded values; raise them
    for large files / high-latency links: ``connections`` parallel ranged
    streams, ``extract_workers`` unzip threads, ``chunk_size`` streaming buffer,
    ``min_segment``/``max_segment`` segment sizing, ``retries`` per segment.

    ``force`` always re-downloads. ``refresh`` does one cheap remote probe and
    re-downloads only if the object changed (else keeps the instant cache hit).
    """
    if local_dir_mode not in _PLACE_MODES:
        raise ValueError("local_dir_mode must be one of %s" % (_PLACE_MODES,))

    key = get_api_key(api_key)
    root = Path(cache_dir).expanduser() if cache_dir else config.datasets_dir()
    root.mkdir(parents=True, exist_ok=True)

    target = root / language_code
    lock_path = str(root / (language_code + ".lock"))

    with FileLock(lock_path):
        need = force or not (target / _COMPLETE_FLAG).exists()
        if not need and refresh:
            need = _remote_changed(language_code, key, target, progress)
        if need:
            _download_and_stage(
                language_code, key, root, target, progress, fresh=force,
                connections=connections, extract_workers=extract_workers,
                chunk_size=chunk_size, min_segment=min_segment,
                max_segment=max_segment, retries=retries,
            )

    if local_dir is not None:
        dest_root = _place_into(target, local_dir, local_dir_mode, progress)
        return Dataset(language_code, dest_root, _load_manifest(dest_root))

    return Dataset(language_code, target, _load_manifest(target))


def list_languages(api_key: Optional[str] = None) -> List[str]:
    """List all language codes the API supports."""
    return _list_languages(get_api_key(api_key))


def _read_meta(target: Path) -> Optional[dict]:
    try:
        return json.loads((target / _META_NAME).read_text())
    except (FileNotFoundError, ValueError):
        return None


def _place_file(src: Path, dest: Path, mode: str) -> None:
    if dest.exists() or dest.is_symlink():
        _safe_unlink(dest)
    if mode in ("auto", "hardlink"):
        try:
            os.link(str(src), str(dest))  # instant, shares inode with the cache
            return
        except OSError:
            if mode == "hardlink":
                raise SignaljtError(
                    "hardlink failed (cache and local_dir on different filesystems?); "
                    "use local_dir_mode='copy' or 'auto'."
                )
            # auto: fall through to copy across filesystems
    if mode == "symlink":
        os.symlink(os.path.abspath(str(src)), str(dest))
        return
    shutil.copy2(str(src), str(dest))


def _place_into(cache_target: Path, local_dir, mode: str, progress) -> Path:
    """Materialize the cached dataset into a user directory. Idempotent and
    lock-safe: concurrent jobs targeting the same dir place the files once."""
    dst = Path(local_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    meta = _read_meta(cache_target) or {}
    with FileLock(str(dst / ".signaljt.lock")):
        existing = _read_meta(dst)
        if existing and existing.get("identity") == meta.get("identity") \
                and (dst / _COMPLETE_FLAG).exists():
            return dst  # already materialized for this version

        for entry in dst.iterdir():  # clear a stale version (keep our lock file)
            if entry.name == ".signaljt.lock":
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                _safe_unlink(entry)

        if progress:
            print("Placing files into %s (mode=%s) ..." % (dst, mode))
        for src in sorted(cache_target.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(cache_target)
            if rel.parts[0].startswith(".signaljt"):
                continue  # skip internal flag/meta files; we rewrite our own
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            _place_file(src, out, mode)

        if meta:
            (dst / _META_NAME).write_text(json.dumps(meta))
        (dst / _COMPLETE_FLAG).write_text("ok")
    return dst


def _remote_changed(language_code, key, target, progress) -> bool:
    """Cheap identity check: has the published object changed since we cached?"""
    stored = _read_meta(target)
    if not stored:
        return True  # no recorded identity -> treat as stale
    try:
        info = get_signed_url(language_code, key)
        client = _client()
        try:
            identity = probe(client, info["url"]).identity
        finally:
            client.close()
    except SignaljtError:
        return False  # unreachable (e.g. offline node) -> keep the cache
    changed = identity != stored.get("identity")
    if changed and progress:
        print("Remote dataset changed (%s -> %s); refreshing."
              % (stored.get("identity"), identity))
    return changed


def _download_and_stage(language_code, key, root, target, progress, *, fresh=False,
                        connections=DEFAULT_CONNECTIONS, extract_workers=1,
                        chunk_size=DEFAULT_CHUNK_SIZE, min_segment=DEFAULT_MIN_SEGMENT,
                        max_segment=DEFAULT_MAX_SEGMENT, retries=DEFAULT_RETRIES) -> None:
    if target.exists():
        shutil.rmtree(target)

    if progress:
        print("Resolving signed URL for %r ..." % language_code)
    info = get_signed_url(language_code, key)

    tmp_zip = root / (language_code + ".part.zip")
    staging = root / (language_code + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    if fresh:  # force: discard any partial and its resume sidecar
        _safe_unlink(tmp_zip)
        _safe_unlink(_sidecar(tmp_zip))

    try:
        result = download_file(
            info["url"], tmp_zip, connections=connections, chunk_size=chunk_size,
            min_segment=min_segment, max_segment=max_segment, retries=retries,
            progress=progress, resume=not fresh,
        )
        if progress:
            print("Extracting ...")
        _extract_parallel(tmp_zip, staging, workers=extract_workers)
        _load_manifest(staging)  # validate before publishing
        (staging / _META_NAME).write_text(json.dumps({
            "language_code": language_code,
            "size": result.size,
            "identity": result.identity,
            "md5": result.md5,
        }))
        (staging / _COMPLETE_FLAG).write_text("ok")
        os.replace(str(staging), str(target))  # atomic publish
        _safe_unlink(tmp_zip)  # success: drop the zip (sidecar already removed)
    except BaseException:
        # Keep tmp_zip + its .signaljtpart sidecar so a rerun resumes; discard only
        # the half-extracted staging dir.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if progress:
        print("Ready: %s" % target)