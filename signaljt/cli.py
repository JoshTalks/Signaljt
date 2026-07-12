"""Command-line interface: ``signaljt login | logout | whoami | download``."""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .auth import login, logout, whoami
from .errors import SignaljtError


def _cmd_login(args) -> int:
    login(api_key=args.token, validate=not args.no_validate)
    return 0


def _cmd_logout(args) -> int:
    logout()
    return 0


def _cmd_whoami(args) -> int:
    whoami()
    print("Authenticated: your API key is valid.")
    return 0


def _parse_size(s):
    """Parse a size like '1M', '512K', '64M', or a plain byte count."""
    s = str(s).strip().upper()
    if s and s[-1] in "KMG":
        mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[s[-1]]
        return int(float(s[:-1]) * mult)
    return int(s)


def _cmd_download(args) -> int:
    from .dataset import fetch_dataset
    langs = args.languages
    target = "all" if langs == ["all"] else (langs[0] if len(langs) == 1 else langs)
    tuning = {}
    if args.chunk_size is not None:
        tuning["chunk_size"] = args.chunk_size
    if args.min_segment is not None:
        tuning["min_segment"] = args.min_segment
    if args.max_segment is not None:
        tuning["max_segment"] = args.max_segment
    if args.retries is not None:
        tuning["retries"] = args.retries
    res = fetch_dataset(
        target,
        cache_dir=args.dir,
        local_dir=args.local_dir,
        local_dir_mode=args.mode,
        connections=args.connections,
        extract_workers=args.extract_workers,
        max_parallel=args.parallel,
        force=args.force,
        refresh=args.refresh,
        **tuning,
    )
    if isinstance(res, dict):
        print("Done: %d languages ready." % len(res))
    else:
        print("Downloaded %d items to %s" % (len(res), res.root))
    return 0


def _cmd_languages(args) -> int:
    from .dataset import list_languages
    for code in list_languages():
        print(code)
    return 0


def _cmd_eval(args) -> int:
    from .scoring import submit
    job = submit(
        args.predictions,
        eval_name=args.name,
        penalize_missing=args.penalize_missing,
        tag_rows=args.tag_rows,
        strip_bracketed=args.strip_bracketed,
    )
    if args.no_wait:
        print(job.eval_id)          # nothing to resolve yet — emit the id
        return 0
    result = job.wait(poll_interval=args.poll, timeout=args.timeout, progress=not args.quiet)
    print(result.result_url)        # the URL itself
    return 0


def _cmd_result(args) -> int:
    from .auth import get_api_key
    from .scoring import EvalJob
    job = EvalJob(args.eval_id, args.eval_name, api_key=get_api_key())
    if args.wait:
        print(job.wait(poll_interval=args.poll, timeout=args.timeout,
                       progress=not args.quiet).result_url)
        return 0
    meta = job.refresh()
    if meta.get("status") == "completed":
        print(meta["result_file_url"])
        return 0
    print("status: %s" % meta.get("status"), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signaljt", description="OI-WER evaluation client.")
    p.add_argument("--version", action="version", version="signaljt %s" % __version__)
    sub = p.add_subparsers(dest="command", required=True)

    lp = sub.add_parser("login", help="Store your API key.")
    lp.add_argument("--token", help="API key (omit to be prompted without echo).")
    lp.add_argument("--no-validate", action="store_true", help="Skip server-side validation.")
    lp.set_defaults(func=_cmd_login)

    op = sub.add_parser("logout", help="Delete the saved API key.")
    op.set_defaults(func=_cmd_logout)

    wp = sub.add_parser("whoami", help="Check that the saved key is valid.")
    wp.set_defaults(func=_cmd_whoami)

    dp = sub.add_parser("download", help="Download one or more languages (or 'all').")
    dp.add_argument("languages", nargs="+", help="Language code(s), or 'all'.")
    dp.add_argument("--dir", help="Cache directory (default: ~/.cache/signaljt/datasets).")
    dp.add_argument("--local-dir", help="Place files here (per-language under <dir>/<code> for many).")
    dp.add_argument("--mode", default="auto", choices=["auto", "copy", "symlink", "hardlink"],
                    help="How to place files into --local-dir (default: auto).")
    dp.add_argument("--connections", type=int, default=1,
                    help="Parallel ranged connections per file (default 1).")
    dp.add_argument("--extract-workers", type=int, default=1,
                    help="Threads for zip extraction (default 1).")
    dp.add_argument("--parallel", type=int, default=4,
                    help="Languages downloaded concurrently (default 4).")
    dp.add_argument("--chunk-size", type=_parse_size, default=None, help="Streaming chunk, e.g. 1M.")
    dp.add_argument("--min-segment", type=_parse_size, default=None, help="Min split size, e.g. 8M.")
    dp.add_argument("--max-segment", type=_parse_size, default=None, help="Max segment size, e.g. 64M.")
    dp.add_argument("--retries", type=int, default=None, help="Per-segment retry attempts.")
    dp.add_argument("--force", action="store_true", help="Re-download even if cached.")
    dp.add_argument("--refresh", action="store_true", help="Re-download only if the remote changed.")
    dp.set_defaults(func=_cmd_download)

    gp = sub.add_parser("languages", help="List supported language codes.")
    gp.set_defaults(func=_cmd_languages)

    ep = sub.add_parser("eval", help="Score a predictions file/URL; prints the result URL.")
    ep.add_argument("predictions", help="Predictions file path or http(s) URL (JSON/JSONL).")
    ep.add_argument("--name", help="Human-readable eval name.")
    ep.add_argument("--no-penalize-missing", dest="penalize_missing", action="store_false",
                    help="Skip missing predictions instead of counting them as errors.")
    ep.add_argument("--no-tag-rows", dest="tag_rows", action="store_false",
                    help="Don't add per-chunk error-type tags.")
    ep.add_argument("--strip-bracketed", action="store_true",
                    help="Strip <noise>/[laughter] tokens before scoring.")
    ep.add_argument("--no-wait", action="store_true", help="Submit only; print the eval_id.")
    ep.add_argument("--poll", type=float, default=3.0, help="Poll interval seconds.")
    ep.add_argument("--timeout", type=float, default=1800.0, help="Max wait seconds.")
    ep.add_argument("--quiet", action="store_true", help="Suppress progress lines.")
    ep.set_defaults(func=_cmd_eval)

    rp = sub.add_parser("result", help="Fetch an eval result URL by id (or name).")
    rp.add_argument("eval_id", nargs="?", help="Eval id (or use --name).")
    rp.add_argument("--name", dest="eval_name", help="Eval name (fallback).")
    rp.add_argument("--wait", action="store_true", help="Wait until completed.")
    rp.add_argument("--poll", type=float, default=3.0)
    rp.add_argument("--timeout", type=float, default=1800.0)
    rp.add_argument("--quiet", action="store_true")
    rp.set_defaults(func=_cmd_result)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SignaljtError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())