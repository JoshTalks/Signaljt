# Signal — REST API Reference

Raw HTTP reference for the Signal (by Josh Talks) OI-WER evaluation API. Use this
if you're not using the Python [`signaljt`](README.md) library — the endpoints
are plain HTTP and work from any language or from `curl`.

- **Base URL:** `https://data-collection-app.joshtalks.org`
- **Endpoints:**
  1. [`GET  /api/eval/fetch-audios/`](#1-fetch-ground-truth-audios) — get the ground-truth audio for a language
  2. [`POST /api/eval/submit/`](#2-submit-predictions) — submit predictions for scoring
  3. [`GET  /api/eval/result/`](#3-fetch-eval-result) — poll status and get the result

---

## Authentication

Every request requires your API key in a header:

```
X-API-Key: your-api-key-here
```

Keys are created on the dashboard and have an expiry (1h / 1d / 2d / 7d / never).
An invalid or expired key returns **401**:

```json
{ "error": "API key has expired. Please contact admin for a new key." }
```

## Supported language codes

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `hi` | Hindi | `bn` | Bengali | `ta` | Tamil |
| `te` | Telugu | `mr` | Marathi | `gu` | Gujarati |
| `kn` | Kannada | `ml` | Malayalam | `pa` | Punjabi |
| `as` | Assamese | `or` | Odia | `mai` | Maithili |
| `bho`| Bhojpuri | `hne`| Chattisgarhi | `ur` | Urdu |

You can also fetch this list live — see [listing languages](#listing-supported-languages).

---

## 1. Fetch ground-truth audios

Returns a **signed download URL** (valid ~1 day) for a language's ground-truth
package (a ZIP containing a manifest + audio files).

```
GET /api/eval/fetch-audios/?language_code=<code>
```

**Request**

```bash
curl -s \
  -H "X-API-Key: YOUR_KEY" \
  "https://data-collection-app.joshtalks.org/api/eval/fetch-audios/?language_code=hi"
```

**Response `200`**

```json
{
  "language_code": "hi",
  "url": "https://storage.googleapis.com/asr_eval/validation_language_zips/Hindi.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=..."
}
```

The `url` is a signed GCS link, valid ~24 hours, and supports HTTP **Range**
requests (so you can parallelize / resume the download).

**Download and unpack it**

```bash
URL=$(curl -s -H "X-API-Key: YOUR_KEY" \
  "https://data-collection-app.joshtalks.org/api/eval/fetch-audios/?language_code=hi" \
  | jq -r .url)

curl -o hindi.zip "$URL"
unzip hindi.zip -d hindi/
```

**ZIP structure**

```
Hindi_manifest.jsonl        # one JSON object per line
audio/
  100003.flac
  100004.flac
  ...
```

Each manifest line has the utterance index and the relative audio path:

```json
{"Index": "100003", "audio_filepath": "audio/100003.flac"}
```

You run your ASR model over these audio files to produce predictions for
[step 2](#2-submit-predictions).

**Errors**

| Status | Body | Cause |
|--------|------|-------|
| `400` | `{"error": "language_code query parameter is required.", "supported_language_codes": [...]}` | no `language_code` |
| `404` | `{"error": "Language code 'xyz' is not supported.", "supported_language_codes": [...]}` | unknown code |
| `401` | `{"error": "..."}` | bad / expired key |

### Listing supported languages

There is no dedicated endpoint — call `fetch-audios` **without** a
`language_code` and read `supported_language_codes` from the `400` body:

```bash
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://data-collection-app.joshtalks.org/api/eval/fetch-audios/" | jq .supported_language_codes
```

---

## 2. Submit predictions

Submits your model's predictions. The backend scores them against the ground
truth **asynchronously** and returns an `eval_id` to track the run.

```
POST /api/eval/submit/
```

Provide **one** of `file` (upload) or `predictions_url` (a public/GCS URL).

### Predictions file format

JSON or JSONL, mapping each utterance index to your model's transcript.

**JSON** (an object of `index → hypothesis`):

```json
{
  "100003": "आपका मॉडल आउटपुट यहाँ",
  "100004": "अगली प्रतिलिपि"
}
```

**JSONL** (one object per line):

```json
{"index": "100003", "hypothesis": "आपका मॉडल आउटपुट यहाँ"}
{"index": "100004", "hypothesis": "अगली प्रतिलिपि"}
```

Predictions may span **any subset of languages** — the backend matches each index
to its language's ground truth and reports per-language scores.

### Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file (multipart) | — | Predictions file (JSON or JSONL). *One of `file`/`predictions_url` required.* |
| `predictions_url` | string | — | Public/GCS URL of the predictions file (no upload). |
| `eval_name` | string | auto | Human-readable name for the run. |
| `penalize_missing` | bool | `true` | Count ground-truth utterances with **no** prediction as full errors. `false` skips them. |
| `tag_rows` | bool | `true` | Add error-type tags (`correct`, `wrong_script`, ...) to each per-chunk entry. |
| `strip_bracketed` | bool | `false` | Strip bracketed tokens like `<noise>` / `[laughter]` before scoring. |

### Request — file upload (`multipart/form-data`)

```bash
curl -s -X POST \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@predictions.json;type=application/json" \
  -F "eval_name=conformer-ctc-ckpt3800" \
  -F "penalize_missing=true" \
  -F "tag_rows=true" \
  -F "strip_bracketed=false" \
  "https://data-collection-app.joshtalks.org/api/eval/submit/"
```

### Request — by URL (`application/json`)

```bash
curl -s -X POST \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "predictions_url": "https://storage.googleapis.com/your-bucket/predictions.json",
        "eval_name": "conformer-ctc-ckpt3800",
        "penalize_missing": true
      }' \
  "https://data-collection-app.joshtalks.org/api/eval/submit/"
```

### Response `202` (queued)

```json
{
  "eval_id": "9c4b6282720",
  "eval_name": "conformer-ctc-ckpt3800",
  "status": "queued",
  "message": "Your eval is queued. You can see the result on the dashboard or call the get_result API with your eval ID: 9c4b6282720"
}
```

Save the `eval_id` and poll [step 3](#3-fetch-eval-result).

**Errors**

| Status | Body | Cause |
|--------|------|-------|
| `400` | `{"error": "Provide either a 'file' upload (JSON or JSONL) or a 'predictions_url'."}` | neither provided |
| `401` | `{"error": "..."}` | bad / expired key |

---

## 3. Fetch eval result

Poll the status of a submitted eval, and get the signed URL to the full result
once it's done.

```
GET /api/eval/result/?eval_id=<id>
GET /api/eval/result/?eval_name=<name>     # fallback: latest run with that name
```

`eval_id` takes priority; at least one of `eval_id` / `eval_name` is required.

**Request**

```bash
curl -s -H "X-API-Key: YOUR_KEY" \
  "https://data-collection-app.joshtalks.org/api/eval/result/?eval_id=9c4b6282720"
```

**Response `200` — still processing** (`status` is `queued` or `running`)

```json
{
  "eval_id": "9c4b6282720",
  "eval_name": "conformer-ctc-ckpt3800",
  "status": "running",
  "language_code": "",
  "submitted_at": "2026-07-11T10:00:00Z",
  "completed_at": null,
  "message": "Eval is still processing. Please check back shortly."
}
```

**Response `200` — completed**

```json
{
  "eval_id": "9c4b6282720",
  "eval_name": "conformer-ctc-ckpt3800",
  "status": "completed",
  "language_code": "",
  "submitted_at": "2026-07-11T10:00:00Z",
  "completed_at": "2026-07-11T10:05:32Z",
  "result_file_url": "https://storage.googleapis.com/asr_eval/eval_results/...?X-Goog-Signature=..."
}
```

`result_file_url` is a signed link (valid ~2 days) to the full
`eval_result.json` — download it for the complete results (schema below).

**Response `200` — failed**

```json
{
  "eval_id": "9c4b6282720",
  "eval_name": "conformer-ctc-ckpt3800",
  "status": "failed",
  "error": "Failed after 3 attempts. Last error: ..."
}
```

**Errors**

| Status | Body | Cause |
|--------|------|-------|
| `400` | `{"error": "Provide either eval_id or eval_name."}` | neither provided |
| `404` | `{"error": "No eval found with eval_id '...'."}` | unknown id |
| `401` | `{"error": "..."}` | bad / expired key |

### Poll until done and download (bash)

```bash
ID=9c4b6282720
while :; do
  STATUS=$(curl -s -H "X-API-Key: YOUR_KEY" \
    "https://data-collection-app.joshtalks.org/api/eval/result/?eval_id=$ID" | jq -r .status)
  echo "status: $STATUS"
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && { echo "eval failed"; exit 1; }
  sleep 3
done

URL=$(curl -s -H "X-API-Key: YOUR_KEY" \
  "https://data-collection-app.joshtalks.org/api/eval/result/?eval_id=$ID" | jq -r .result_file_url)
curl -o eval_result.json "$URL"
```

---

## Result file schema (`eval_result.json`)

Top-level keys: `overall`, `per_language`, `per_chunk`, `meta`.

```json
{
  "overall": {
    "weighted_oiwer": 0.1033,
    "macro_oiwer": 0.103,
    "weighted_by": "reference_words",
    "word": { "errors": 18417, "count": 178265, "sub": 9586, "ins": 3181, "del": 5650 },
    "total_utterances": 14714,
    "total_duration_sec": 76784.6,
    "languages_scored": 15,
    "extra_predictions": 0
  },
  "per_language": {
    "hindi": {
      "iso": "hi",
      "oiwer": 0.104,
      "word": { "errors": 1151, "count": 11129, "sub": 625, "ins": 206, "del": 320 },
      "utterances": 1034,
      "coverage": { "scored": 1034, "missing": 0, "extra": 0 }
    }
  },
  "per_chunk": [
    {
      "index": "100003",
      "language": "hindi",
      "duration": 4.68,
      "cps": 8.33,
      "p808_mos": 3.842,
      "native_district": "…",
      "native_state": "…",
      "gender": "M",
      "speaker_id": 106583,
      "predicted": true,
      "status": "scored",
      "word": { "errors": 0, "count": 9, "sub": 0, "ins": 0, "del": 0 },
      "alignment": { "hyp": ["…", "…"], "ops": ["c", "c"] },
      "tags": ["correct"]
    }
  ],
  "meta": {
    "scored_at": "2026-07-12T05:17:58Z",
    "audio_version": "0.1.0",
    "lattice_version": "0.1.0",
    "penalize_missing": true,
    "tag_rows": true,
    "strip_bracketed": false
  }
}
```

**Field reference**

| Section | Fields |
|---------|--------|
| `overall` | `weighted_oiwer`, `macro_oiwer`, `weighted_by`, `word{errors,count,sub,ins,del}`, `total_utterances`, `total_duration_sec`, `languages_scored`, `extra_predictions` |
| `per_language[<name>]` | `iso`, `oiwer`, `word{...}`, `utterances`, `coverage{scored,missing,extra}` |
| `per_chunk[i]` | `index`, `language`, `duration`, `cps`, `p808_mos`, `native_district`, `native_state`, `gender`, `speaker_id`, `predicted`, `status`, `word{...}`, `alignment{hyp[],ops[]}`, `tags[]` |
| `meta` | `scored_at`, `audio_version`, `lattice_version`, `penalize_missing`, `tag_rows`, `strip_bracketed` |

- **`weighted_oiwer`** — overall OI-WER weighted by reference words (the headline number).
- **`macro_oiwer`** — unweighted mean across languages.
- **`alignment.ops`** codes: `c` = correct, `s` = substitution, `i` = insertion, `d` = deletion.
- **`coverage.missing`** — ground-truth utterances you didn't predict (penalized only when `penalize_missing=true`).

---

## Typical flow (end-to-end, `curl`)

```bash
KEY="YOUR_KEY"
BASE="https://data-collection-app.joshtalks.org/api/eval"

# 1. Get + unpack the ground truth for a language
URL=$(curl -s -H "X-API-Key: $KEY" "$BASE/fetch-audios/?language_code=hi" | jq -r .url)
curl -o hi.zip "$URL" && unzip hi.zip -d hi/

# 2. Run your ASR model over hi/audio/*.flac -> predictions.json  (your code)

# 3. Submit
ID=$(curl -s -X POST -H "X-API-Key: $KEY" \
  -F "file=@predictions.json;type=application/json" \
  -F "eval_name=my-model-v1" \
  "$BASE/submit/" | jq -r .eval_id)

# 4. Poll until completed
while [ "$(curl -s -H "X-API-Key: $KEY" "$BASE/result/?eval_id=$ID" | jq -r .status)" != "completed" ]; do
  sleep 3
done

# 5. Download the full result
RESULT=$(curl -s -H "X-API-Key: $KEY" "$BASE/result/?eval_id=$ID" | jq -r .result_file_url)
curl -o eval_result.json "$RESULT"
jq '.overall.weighted_oiwer' eval_result.json
```