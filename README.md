# signaljt

**Python client library for the Signal validation set** — Signal, by Josh Talks.

`signaljt` gives you two things through one small, dependency-light package:

1. **Datasets** — download the ground-truth evaluation audio for any of 15 Indic
   languages, with resumable parallel downloads, integrity checks, a shared
   cache, and ready-to-use absolute file paths.
2. **Scoring** — submit your ASR model's predictions and get **OI-WER** back:
   overall, per-language, and per-utterance, as a dot-accessible object that
   drops straight into wandb / TensorBoard.

Everything is available both as a **Python API** and a **`signaljt` CLI**.

> **Not using Python?** The three underlying REST endpoints are documented
> separately in **[API.md](API.md)**, with `curl` examples you can use from any
> language.

---

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Getting started (5 minutes)](#getting-started-5-minutes)
- [Authentication](#authentication)
- [Downloading datasets](#downloading-datasets)
  - [One language, many, or all](#one-language-many-or-all)
  - [The `Dataset` object](#the-dataset-object)
  - [Placing files where you want (`local_dir`)](#placing-files-where-you-want-local_dir)
  - [Caching, refresh, and force](#caching-refresh-and-force)
  - [Performance tuning](#performance-tuning)
  - [CLI: `download` / `languages`](#cli-download--languages)
- [Evaluating (OI-WER scoring)](#evaluating-oi-wer-scoring)
  - [Prediction input formats](#prediction-input-formats)
  - [The `EvalResult` object](#the-evalresult-object)
  - [Result schema](#result-schema)
  - [Scoring options](#scoring-options)
  - [Non-blocking submit / poll](#non-blocking-submit--poll)
  - [Logging to wandb / TensorBoard](#logging-to-wandb--tensorboard)
  - [CLI: `eval` / `result`](#cli-eval--result)
- [Using with ML frameworks](#using-with-ml-frameworks)
- [Configuration reference](#configuration-reference)
- [Error handling](#error-handling)
- [Full API reference](#full-api-reference)
- [End-to-end example](#end-to-end-example)

---

## Requirements

- **Python 3.8+**
- Runtime dependencies: `httpx`, `filelock` (installed automatically)
- Optional: `ipywidgets` (for the notebook login widget)

## Installation

```bash
pip install signaljt
```

Notebook login widget:

```bash
pip install "signaljt[notebook]"
```

From source (development):

```bash
git clone <repo> && cd meta_validation_package
pip install -e .            # or: pip install -e ".[notebook,dev]"
```

---

## Getting started (5 minutes)

```python
import signaljt

# 1. Authenticate once (or set the SIGNALJT_API_KEY env var)
signaljt.login("your-api-key")

# 2. Download a language's ground-truth audio
ds = signaljt.fetch_dataset("hi")           # Hindi
print(len(ds), "utterances")
print(ds[0].index, ds[0].audio_path)        # absolute path, ready to load

# 3. Run YOUR model over the audio -> {index: transcript}
predictions = {item.index: my_asr_model(item.audio_path) for item in ds}

# 4. Score it — returns overall + per-language + per-chunk OI-WER
result = signaljt.evaluate(predictions, eval_name="my-model-v1")
print("OI-WER:", result.oiwer)              # e.g. 0.1033
print("Hindi:", result.per_language["hi"].oiwer)

# 5. (optional) log to your experiment tracker
import wandb; wandb.log(result.metrics())
```

The same flow on the command line:

```bash
signaljt login
signaljt download hi
# ... produce predictions.json ...
signaljt eval predictions.json              # prints the signed result URL
```

---

## Authentication

Signal authenticates every request with a long-lived **API key** (`X-API-Key`).
`signaljt` stores it once and resolves it automatically for every call.

**Resolution order** (highest priority first):

1. An explicit `api_key=...` argument to any function
2. The `SIGNALJT_API_KEY` environment variable
3. The saved token file at `~/.cache/signaljt/token`

### Log in

Three interchangeable ways — use whichever fits your context:

```bash
# CLI (interactive, hidden prompt) — best on a terminal / HPC login node
signaljt login
```

```python
# In a script
import signaljt
signaljt.login("your-api-key")       # or signaljt.login() to be prompted
```

```python
# In a Jupyter / Colab notebook — masked widget, never echoed into the notebook
from signaljt import notebook_login
notebook_login()
```

```bash
# Non-interactive (CI, Slurm batch) — just set the env var, no login needed
export SIGNALJT_API_KEY="your-api-key"
```

The token is written to `~/.cache/signaljt/token` with `0600` permissions.

### Check and log out

```bash
signaljt whoami      # validates the saved key against the server
signaljt logout      # deletes the saved token
```

```python
signaljt.whoami()    # raises AuthenticationError if the key is invalid
signaljt.logout()
```

> **Note:** API keys have an expiry (1h / 1d / 2d / 7d / never, set on the
> dashboard). An expired key returns a clear `AuthenticationError` — just log in
> again with a fresh key.

---

## Downloading datasets

`fetch_dataset` fetches the signed URL, downloads the zip, verifies it, extracts
it, and returns a friendly object with **absolute** audio paths. It's cached, so
the second call (from any process or node) is instant.

```python
ds = signaljt.fetch_dataset("bn")     # Bengali
```

**Supported language codes** (via `signaljt.list_languages()`):

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `hi` | Hindi | `bn` | Bengali | `ta` | Tamil |
| `te` | Telugu | `mr` | Marathi | `gu` | Gujarati |
| `kn` | Kannada | `ml` | Malayalam | `pa` | Punjabi |
| `as` | Assamese | `or` | Odia | `mai` | Maithili |
| `bho`| Bhojpuri | `hne`| Chattisgarhi | `ur` | Urdu |

### One language, many, or all

The first argument accepts a single code, a list, or the string `"all"`:

```python
ds   = signaljt.fetch_dataset("hi")           # -> a single Dataset
data = signaljt.fetch_dataset(["hi", "bn"])   # -> {"hi": Dataset, "bn": Dataset}
data = signaljt.fetch_dataset("all")          # -> all 15 languages, as a dict
```

When you request multiple languages they download **concurrently** (bounded by
`max_parallel`, default 4); each language keeps its own cache/lock/resume, so a
partial run resumes cleanly.

### The `Dataset` object

```python
ds = signaljt.fetch_dataset("bn")

len(ds)                    # number of utterances
ds.language_code           # "bn"
ds.root                    # directory containing the extracted files

# Iterate
for item in ds:
    item.index             # utterance id (str), from the manifest
    item.audio_path        # ABSOLUTE path to the .flac, ready to load
    item.extra             # dict of any extra manifest columns

# Index access
ds[0]                      # by position -> DatasetItem
ds["100825"]               # by Index    -> DatasetItem
"100825" in ds             # membership test
ds.get("999", default=None)

# Bulk helpers
ds.indices                 # ["100825", "100826", ...]
ds.paths()                 # {"100825": "/abs/.../100825.flac", ...}
ds.items                   # list[DatasetItem]
```

The paths are computed **at load time** from wherever the data actually lives, so
they stay correct even if you move or copy the folder — always go through the
`Dataset` object rather than hand-parsing the manifest file.

### Placing files where you want (`local_dir`)

By default the data lives in the cache. To also materialize it into a directory
you choose (like Hugging Face's `local_dir`), without re-downloading:

```python
ds = signaljt.fetch_dataset("bn", local_dir="./bn_data")
ds.root            # ./bn_data
ds[0].audio_path   # absolute path inside ./bn_data

# Materialize an already-fetched dataset afterwards:
signaljt.fetch_dataset("bn").export("./bn_data", mode="copy")

# For multiple languages, each lands under local_dir/<code>/
signaljt.fetch_dataset("all", local_dir="./data")   # ./data/hi, ./data/bn, ...
```

**Placement modes** (`local_dir_mode=`):

| Mode | Behavior | Use when |
|------|----------|----------|
| `auto` *(default)* | **hardlink** if on the same filesystem (instant, no extra disk); **copy** across filesystems | almost always |
| `copy` | a real independent copy | you'll move / rsync / edit it |
| `symlink` | symlink back into the cache | smallest footprint; breaks if the cache is cleared |
| `hardlink` | force a hardlink (errors across filesystems) | you want to guarantee no copy |

### Caching, refresh, and force

The dataset is cached under `~/.cache/signaljt/datasets/<code>` (override with
`cache_dir=` or `SIGNALJT_CACHE_DIR`). Subsequent calls are instant and do no
network I/O.

```python
signaljt.fetch_dataset("bn")                 # instant if cached
signaljt.fetch_dataset("bn", refresh=True)   # one cheap probe; re-download only if the remote zip changed
signaljt.fetch_dataset("bn", force=True)     # always re-download from scratch
```

- **Integrity:** every download is verified against the server's MD5
  (`x-goog-hash`) when present, size otherwise.
- **Versioning:** the cached zip's identity is recorded; `refresh=True` detects a
  re-published dataset and refreshes automatically.
- **Resume:** a killed download continues where it stopped (segment-granular),
  which makes it safe under Slurm preemption / time limits.

### Performance tuning

Every knob is a per-call parameter. Defaults are conservative (single-threaded),
which is fastest on low-latency links; raise them for large files or
high-latency / on-prem → cloud transfers.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `connections` | `1` | parallel ranged download streams per file |
| `extract_workers` | `1` | threads used to unzip |
| `chunk_size` | `1 MiB` | streaming buffer size |
| `min_segment` | `8 MiB` | don't split a download below this |
| `max_segment` | `64 MiB` | cap per-segment size (bounds resume loss) |
| `retries` | `4` | per-segment retry attempts |
| `max_parallel` | `4` | languages downloaded concurrently (multi-language calls) |

```python
signaljt.fetch_dataset(
    "bn",
    connections=8,        # parallelize the download
    extract_workers=4,    # parallelize the unzip
    chunk_size=4 * 1024 * 1024,
    max_segment=32 * 1024 * 1024,
    retries=6,
)
```

> **Tip:** on a machine co-located with the storage (same-region cloud) a single
> connection already saturates the link, so the defaults are ideal. On a cluster
> pulling over the public internet, `connections=8` is usually a solid win —
> benchmark once on your actual nodes.

### CLI: `download` / `languages`

```bash
signaljt languages                      # list the 15 supported codes

signaljt download hi                    # one language
signaljt download hi bn te              # several
signaljt download all                   # everything

# Options
signaljt download bn \
  --dir ./cache \                       # cache directory
  --local-dir ./bn_data --mode auto \   # also place files here
  --connections 8 --extract-workers 4 \
  --parallel 4 \                        # languages at once (for multi/all)
  --chunk-size 4M --min-segment 8M --max-segment 32M --retries 6 \
  --force                               # or --refresh
```

Size flags accept `K`/`M`/`G` suffixes (e.g. `--chunk-size 4M`).

---

## Evaluating (OI-WER scoring)

Scoring is an asynchronous job on the server: you **submit** predictions, it
scores them against the ground truth, and you **fetch** the result. `evaluate()`
does the whole thing synchronously and hands you a parsed result object.

```python
result = signaljt.evaluate("predictions.json", eval_name="conformer-ckpt3800")
print(result.oiwer)                 # weighted overall OI-WER
```

`evaluate()` **blocks until scoring finishes** and returns silently (no console
output). Pass `progress=True` to print poll status to stderr.

### Prediction input formats

`predictions` (or `predictions_url=`) can be any of:

```python
signaljt.evaluate("preds.json")                       # JSON file path
signaljt.evaluate("preds.jsonl")                      # JSONL file path (streamed on upload)
signaljt.evaluate(predictions_url="https://.../p.json")  # a public / GCS URL (no upload)
signaljt.evaluate({"100000": "predicted text", ...})  # in-memory dict {index: hypothesis}
signaljt.evaluate([{"index": "100000", "hypothesis": "..."}, ...])  # in-memory list
```

The prediction set may span **any subset of languages** — the backend matches
each index to its language's ground truth and reports per-language scores.

> **Large prediction files:** the file is streamed on upload (not loaded into
> memory). For very large sets you can host the file on GCS / any public URL and
> pass `predictions_url=` to skip the upload entirely.

### The `EvalResult` object

Everything is reachable by attribute (`.`) **or** key (`[...]`) — nested dicts
are wrapped lazily, so it's cheap even with tens of thousands of chunks.

```python
r = signaljt.evaluate("predictions.json")

r.oiwer                          # shortcut for r.overall.weighted_oiwer
r.eval_id                        # server eval id
r.result_url                     # signed URL to the full result JSON (valid ~2 days)

# Overall
r.overall.weighted_oiwer
r.overall.macro_oiwer
r.overall.word.errors            # errors / count / sub / ins / del
r.overall.total_utterances
r.overall.total_duration_sec

# Per language (keyed by ISO code, same codes as fetch_dataset)
r.per_language["hi"].oiwer
r.per_language.hi.name           # "hindi"
r.per_language.hi.coverage.scored, r.per_language.hi.coverage.missing
for iso, lang in r.per_language.items():
    print(iso, lang.oiwer, dict(lang.word))

# Per utterance (lazy list — only built when you touch it)
c = r.per_chunk[0]
c.index, c.language, c.duration
c.word.errors
c.alignment.hyp                  # ["word1", "word2", ...]
c.alignment.ops                  # ["c", "s", "i", "d", ...]  (correct / sub / ins / del)
c.tags                           # ["correct"], ["wrong_script"], ...

# Run metadata
r.meta.penalize_missing, r.meta.tag_rows, r.meta.strip_bracketed

# Export
r.to_dict()                      # the raw result dict
r.save("eval_result.json")       # write the full result to disk
```

### Result schema

| Section | Fields |
|---------|--------|
| `overall` | `weighted_oiwer`, `macro_oiwer`, `weighted_by`, `word{errors,count,sub,ins,del}`, `total_utterances`, `total_duration_sec`, `languages_scored`, `extra_predictions` |
| `per_language[iso]` | `iso`, `name`, `oiwer`, `word{...}`, `utterances`, `coverage{scored,missing,extra}` |
| `per_chunk[i]` | `index`, `language`, `duration`, `cps`, `p808_mos`, `native_district`, `native_state`, `gender`, `speaker_id`, `predicted`, `status`, `word{...}`, `alignment{hyp[],ops[]}`, `tags[]` |
| `meta` | `scored_at`, `audio_version`, `lattice_version`, `penalize_missing`, `tag_rows`, `strip_bracketed` |

`alignment.ops` codes: `c` = correct, `s` = substitution, `i` = insertion,
`d` = deletion.

### Scoring options

| Option | Default | Effect |
|--------|---------|--------|
| `penalize_missing` | `True` | Count ground-truth utterances you **didn't** predict as full errors. Set `False` to score only what you predicted (missing ones are reported in `coverage` but not penalized). |
| `tag_rows` | `True` | Add error-type `tags` to each per-chunk entry (`correct`, `wrong_script`, `internal_deletion`, ...). |
| `strip_bracketed` | `False` | Strip bracketed tokens like `<noise>` / `[laughter]` from the hypothesis before scoring. |

```python
# "How good is my model only on what it transcribed?"
signaljt.evaluate(preds, penalize_missing=False)

# "How good over the whole benchmark, including gaps?" (default)
signaljt.evaluate(preds, penalize_missing=True)
```

### Non-blocking submit / poll

For fire-and-forget or batch pipelines, submit without waiting:

```python
job = signaljt.submit("predictions.json", eval_name="run-42")
job.eval_id                      # save this
job.status                       # "queued"

# ... later, same or different process ...
result = job.wait()                              # block until completed
result = signaljt.get_result(job.eval_id, wait=True)   # by id, from anywhere
```

`EvalJob` methods:

- `job.refresh()` — one status poll; updates `job.status`, returns raw metadata
- `job.wait(poll_interval=3.0, timeout=1800.0, progress=False)` — block, then return an `EvalResult`
- `job.result()` — fetch now; raises if not yet completed

### Logging to wandb / TensorBoard

`result.metrics()` returns a flat, slash-namespaced `{name: value}` dict — overall
**and** per-language — that both tools group nicely:

```python
metrics = result.metrics()
# {
#   "oiwer/weighted": 0.1033, "oiwer/macro": 0.103,
#   "words/errors": 18417, "words/sub": 9586, "words/ins": 3181,
#   "words/del": 5650, "words/count": 178265,
#   "oiwer_by_language/hi": 0.104, "oiwer_by_language/bn": 0.104, ...
# }

# Weights & Biases
import wandb
wandb.log(result.metrics())

# TensorBoard
for name, value in result.metrics().items():
    writer.add_scalar(name, value, step)
```

Control what's included:

```python
result.metrics(per_language=True, words=True, prefix="oiwer")
```

### CLI: `eval` / `result`

The CLI takes a predictions file or URL and prints the **result URL** on stdout
(progress goes to stderr, so `URL=$(...)` captures only the URL):

```bash
signaljt eval predictions.json                 # submit + wait, prints result URL
signaljt eval https://.../predictions.jsonl    # from a URL
signaljt eval predictions.json --name my-run --strip-bracketed
signaljt eval predictions.json --no-penalize-missing --no-tag-rows
signaljt eval predictions.json --no-wait       # submit only, prints eval_id
signaljt eval predictions.json --quiet         # no progress on stderr

# Fetch an existing eval later
signaljt result <eval_id>                       # prints result URL if completed, else status
signaljt result <eval_id> --wait                # block until completed
```

---

## Using with ML frameworks

`ds` is just an `Index → absolute audio path` mapping, so it drops into any stack.
The core eval loop needs no adapter:

```python
# transformers — pipelines accept a file path directly
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
preds = {it.index: asr(it.audio_path)["text"] for it in ds}
result = signaljt.evaluate(preds)
```

Thin adapters when you want a framework-native object (no extra deps in this package):

```python
# PyTorch DataLoader
import torch, torchaudio
class TorchDS(torch.utils.data.Dataset):
    def __init__(self, ds): self.items = ds.items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        wav, sr = torchaudio.load(self.items[i].audio_path)
        return self.items[i].index, wav, sr
loader = torch.utils.data.DataLoader(TorchDS(ds), batch_size=8, collate_fn=list)

# HuggingFace datasets (lazy audio decoding)
import datasets
hf = datasets.Dataset.from_dict(
    {"index": ds.indices, "audio": [it.audio_path for it in ds]}
).cast_column("audio", datasets.Audio())

# fairseq / wav2vec TSV manifest
import soundfile as sf, os
with open("eval.tsv", "w") as f:
    print(ds.root, file=f)
    for it in ds:
        print("%s\t%d" % (os.path.relpath(it.audio_path, ds.root),
                          sf.info(it.audio_path).frames), file=f)
```

---

## Configuration reference

**Environment variables** (all optional):

| Variable | Purpose |
|----------|---------|
| `SIGNALJT_API_KEY` | API key for non-interactive use (CI / Slurm batch) |
| `SIGNALJT_CACHE_DIR` | Override the cache root (default `~/.cache/signaljt`) |
| `SIGNALJT_API_BASE` | Override the API base URL (staging / testing) |
| `HTTPS_PROXY` / `NO_PROXY` | Standard proxy settings, honoured on all requests |

**Cache layout** (`~/.cache/signaljt/` by default):

```
~/.cache/signaljt/
├── token                        # your API key (chmod 600)
└── datasets/
    └── <code>/                  # e.g. bn/
        ├── <Language>_manifest.jsonl
        └── audio/*.flac
```

---

## Error handling

All exceptions derive from `signaljt.SignaljtError`:

| Exception | Raised when |
|-----------|-------------|
| `SignaljtError` | base class — catch this to handle any library error |
| `NotLoggedInError` | no API key found (arg / env / saved token) |
| `AuthenticationError` | the server rejected the key (invalid / expired) |
| `ApiError` | the API returned an unexpected response |
| `DownloadError` | a transfer failed after retries, or a checksum mismatch |
| `ManifestError` | the dataset manifest was missing or malformed |

```python
import signaljt
try:
    ds = signaljt.fetch_dataset("bn")
except signaljt.NotLoggedInError:
    signaljt.login()
except signaljt.SignaljtError as e:
    print("signaljt failed:", e)
```

---

## Full API reference

**Auth**

```python
signaljt.login(api_key=None, *, validate=True) -> None
signaljt.logout() -> None
signaljt.whoami(api_key=None) -> dict
signaljt.get_api_key(api_key=None, *, required=True) -> str | None
signaljt.notebook_login(validate=True) -> None
```

**Datasets**

```python
signaljt.list_languages(api_key=None) -> list[str]

signaljt.fetch_dataset(
    language,                      # str code | list[str] | "all"
    *, api_key=None, cache_dir=None,
    local_dir=None, local_dir_mode="auto",
    connections=1, extract_workers=1,
    chunk_size=1048576, min_segment=8388608, max_segment=67108864, retries=4,
    force=False, refresh=False,
    max_parallel=4, continue_on_error=True, progress=True,
) -> Dataset | dict[str, Dataset]

# Dataset
len(ds); iter(ds); ds[int]; ds[index]; index in ds
ds.get(index, default=None); ds.indices; ds.paths(); ds.items
ds.root; ds.language_code
ds.export(local_dir, *, mode="auto") -> Dataset

# DatasetItem
item.index; item.audio_path; item.extra
```

**Scoring**

```python
signaljt.evaluate(
    predictions=None, *, predictions_url=None, eval_name=None,
    penalize_missing=True, tag_rows=True, strip_bracketed=False,
    api_key=None, poll_interval=3.0, timeout=1800.0, progress=False,
) -> EvalResult

signaljt.submit(
    predictions=None, *, predictions_url=None, eval_name=None,
    penalize_missing=True, tag_rows=True, strip_bracketed=False, api_key=None,
) -> EvalJob

signaljt.get_result(
    eval_id=None, *, eval_name=None, api_key=None,
    wait=False, poll_interval=3.0, timeout=1800.0,
) -> EvalResult

# EvalJob
job.eval_id; job.eval_name; job.status
job.refresh() -> dict
job.wait(*, poll_interval=3.0, timeout=1800.0, progress=False) -> EvalResult
job.result() -> EvalResult

# EvalResult
r.oiwer; r.overall; r.per_language; r.per_chunk; r.meta
r.eval_id; r.eval_name; r.result_url
r.metrics(*, per_language=True, words=True, prefix="oiwer") -> dict
r.to_dict() -> dict
r.save(path) -> None
```

---

## End-to-end example

```python
import signaljt

signaljt.login("your-api-key")                # once (or SIGNALJT_API_KEY)

# Score a model across every language and log per-language OI-WER
scores = {}
for code in signaljt.list_languages():
    ds = signaljt.fetch_dataset(code)         # cached after first pull
    preds = {it.index: my_asr_model(it.audio_path) for it in ds}
    scores.update(preds)

result = signaljt.evaluate(scores, eval_name="whisper-large-v3")
print("Overall OI-WER:", result.oiwer)
for iso, lang in result.per_language.items():
    print(f"  {iso}: {lang.oiwer}")

result.save("eval_result.json")

import wandb
wandb.init(project="asr-eval")
wandb.log(result.metrics())
```

CLI equivalent for a single scored file:

```bash
export SIGNALJT_API_KEY="your-api-key"
signaljt download all
# ... run your model, write predictions.json ...
RESULT_URL=$(signaljt eval predictions.json --name whisper-large-v3)
echo "Full result: $RESULT_URL"
```