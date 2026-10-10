# TRIBE v2 on a Hugging Face Inference Endpoint

[TRIBE v2](https://huggingface.co/facebook/tribev2) (Meta) is a brain-encoding model. For a video it
predicts the **fMRI (BOLD) response of an "average subject"** on the fsaverage5 cortical mesh:
- one row per 1 s segment
- 20,484 columns, one per cortical vertex

It is not a retention, engagement, attention or virality model. Hookline stores its output as-is
and does not turn it into any score.

The model runs only on your own Hugging Face Inference Endpoint. Nothing about it is downloaded or
run on this Mac.

## Status

| Part | State |
|---|---|
| Hookline client (`src/hookline/tribe.py`), pipeline stage, CLI flags, artifacts | working; tested with a faked endpoint (27 tests, including a real localhost HTTP round trip) |
| Space job API (`endpoint/tribev2_space/`) and the `--tribe-api job` client | working code, tested with a fake model; the Docker image has not been built and nothing is deployed |
| Endpoint handler (`endpoint/tribev2/handler.py`, `requirements.txt`) | written against upstream `tribev2@af58661`; its request/response contract is tested with stand-in modules. **Not yet run on a real endpoint** |
| Endpoint deployment | **not done.** It needs your Hugging Face account, licence decisions and paid GPU hardware (steps below) |
| Real TRIBE output for any video | **none yet** |

## Licences: check before deploying

- **TRIBE v2 weights and code: CC BY-NC 4.0.** That means non-commercial use only. `neuralset`, which
  it depends on, is also CC BY-NC 4.0. Using it on a company's reels for a business purpose may not be
  allowed. Confirm your use is non-commercial (research or hackathon evaluation) with the organisers
  before deploying.
- **meta-llama/Llama-3.2-3B** (TRIBE's text features) is gated under the Llama 3.2 Community
  License. Request access on its model page with the Hugging Face account whose token the endpoint
  uses.
- facebook/vjepa2-vitg-fpc64-256 (video) and facebook/dinov2-large are Apache-2.0.
  facebook/w2v-bert-2.0 (audio) is MIT.
- Speech is transcribed by whisperX large-v3, which TRIBE runs at request time with `uvx`.

## Why a custom handler is required

`facebook/tribev2` has no `pipeline_tag` and no `library_name`. It loads through the `tribev2`
package (`TribeModel.from_pretrained` → `get_events_dataframe` → `predict`), not through
transformers. The default Inference Endpoints toolkit can't serve it, so the endpoint needs
`handler.py`.

The handler:
- loads `facebook/tribev2` at the pinned revision `f894e783…`
- writes the received video to a temporary file
- runs upstream's video path
- returns the prediction array (base64 float32) together with each row's segment start and duration

## What you need to set up on Hugging Face

1. **Account and billing.** Inference Endpoints bill per hour while running. Add a payment method,
   and check current prices on the Inference Endpoints pricing page.
2. **Llama 3.2 access.** Open https://huggingface.co/meta-llama/Llama-3.2-3B, accept the licence,
   and wait for approval.
3. **Two tokens** (Settings → Access Tokens):
   - for the endpoint: a *read* token that can access Llama 3.2.
   - for Hookline on your Mac: a *fine-grained* token with "Make calls to Inference Endpoints" for
     your endpoint.

   Never paste a token into a chat, a file in git, or a command that ends up in shell history.
4. **A model repo for the handler**, for example `<you>/hookline-tribev2-endpoint`, set to private.
   Upload `endpoint/tribev2/handler.py` and `endpoint/tribev2/requirements.txt` to its root:
   ```bash
   hf auth login                                   # paste the token at the prompt, not on the command line
   hf repo create <you>/hookline-tribev2-endpoint --private
   hf upload <you>/hookline-tribev2-endpoint endpoint/tribev2/handler.py handler.py
   hf upload <you>/hookline-tribev2-endpoint endpoint/tribev2/requirements.txt requirements.txt
   ```
5. **The endpoint** (https://endpoints.huggingface.co → New, choose that repo):
   - **Task:** Custom.
   - **Hardware:** one NVIDIA GPU with **24 GB or more** (for example L4 or A10G; L40S if 24 GB runs
     out). Upstream loads Llama 3.2-3B, V-JEPA2 ViT-g, Wav2Vec-BERT 2.0 and whisperX large-v3 on CUDA.
     Hookline has not measured how much memory this actually needs.
   - **Security:** Protected (only your token can call it).
   - **Environment secret:** `HF_TOKEN` = the read token from step 3. It downloads the gated Llama
     weights inside the endpoint.
   - **Autoscaling:** scale to zero after 15 minutes, so idle time doesn't bill. The first request
     after idling gets HTTP 503 while the endpoint starts. Hookline retries with backoff (up to about
     10 minutes by default).
   - **Request timeout:** raise it if your plan allows. One 60 s reel runs whisperX plus three
     feature extractors, and the first request also downloads ~20 GB of weights inside the endpoint.
6. **Hookline configuration**, in the untracked `.env` (or exported in your shell):
   ```bash
   TRIBE_ENDPOINT_URL=https://<id>.<region>.aws.endpoints.huggingface.cloud
   HF_TOKEN=hf_...            # the fine-grained token from step 3
   ENABLE_TRIBE=true          # optional; --tribe does the same
   ```

Not yet verified on a real endpoint (check these on the first deployment):
- **Python and torch:** `tribev2` needs Python 3.11 or newer and pins `torch<2.7`. If the endpoint
  image's Python or CUDA build conflicts, the dependency install fails at startup (see the endpoint
  logs). The fix is a custom container image that has those versions.
- **Request size:** a base64 body is about 1.33× the video. The company reels are up to 29 MB, and
  Hookline refuses files over 50 MB before sending (`TribeConfig.max_upload_mb`). The endpoint's own
  body limit hasn't been confirmed.
- **Internet access:** `uvx whisperx` needs the endpoint to have outbound internet the first time.

## Alternative: your own Space with a job API (`endpoint/tribev2_space/`)

A Docker Space on a GPU can serve TRIBE as a job API: `POST /analyze` returns a job id, and `GET
/report/{job_id}` returns `queued`, `running`, `done` (with the result) or `failed`. This suits a
slow model, because no single HTTP request has to stay open for minutes. The `result` is exactly what
`handler.py` returns, so Hookline validates it the same way.

- **Files:** `Dockerfile`, `app.py`, `requirements.txt` and `README.md` from `endpoint/tribev2_space/`, plus
  `endpoint/tribev2/handler.py`. All five go to the Space's root.
- **Space settings:** SDK Docker (blank template), **private**, a GPU hardware (for example A100 or L40S;
  check the price shown on the page), a **sleep time** so it doesn't bill while idle, and a secret
  `HF_TOKEN` (a read token whose account has Llama 3.2 access).
- **Hookline:** `TRIBE_ENDPOINT_URL=https://<owner>-<space-name>.hf.space`, `HF_TOKEN=<fine-grained token that can read the Space>`,
  and run with `--tribe-api job`.
- **Status:** the client, the job store and the FastAPI app are tested (including a real localhost HTTP
  round trip). The Dockerfile has not been built, and the model has not run on a Space. Check the
  Space's build and container logs on the first deploy.

## Running

TRIBE runs as an optional stage of `run_speech_ocr.py`. Task 2 features have to exist
first (`run_pipeline.py`). Upload rules:
- Nothing is uploaded without `--tribe-allow-remote`.
- Company reels under `01-Hookline/` also need `--tribe-approve <video_id>`. Only use it after the
  organisers allow uploading that reel.

```bash
# your own or licensed clip, TRIBE only (other model stages skipped)
.venv-models/bin/python scripts/run_speech_ocr.py --input my_clip.mp4 --skip-speech --skip-ocr --skip-librosa \
    --tribe --tribe-allow-remote            # add --tribe-api job when the URL is a Space (job API)
# a company reel, only after approval
.venv-models/bin/python scripts/run_speech_ocr.py --input 01-Hookline/face/videos/FC005.mp4 --artifacts artifacts/company \
    --skip-speech --skip-ocr --skip-librosa --tribe --tribe-allow-remote --tribe-approve FC005
```

`--tribe-language` sets whisperX's language: english (the default), french, spanish, dutch or
chinese. Those are the only languages upstream supports. Reels in other languages, such as Hindi,
get poor transcripts, and that weakens TRIBE's text features.

## Outputs

Per video, in `<artifacts>/features/<video_id>/`:

| File | Contents |
|---|---|
| `tribe.json` | Status (`available` or `failed`), whether anything was sent, the reason, the model repo and revision with its licence, the handler version and library versions, `timestamps_s` (segment starts), `segment_durations_s`, `tr_s`, `output_meaning`, request attempts, upload size, and the SHA-256 of the predictions file. It never contains the endpoint URL or the token. |
| `tribe_preds.npy` | float32 array, shape (n_segments, 20484), exactly as the endpoint returned it. |

How Hookline treats the output:
- The predictions stay on TRIBE's own 1 s segments. They are not merged into the 0.5 s
  `segments_merged.csv`, scores, recommendations or stage status.
- If TRIBE fails or is blocked, every other output is identical to a run without TRIBE, and an
  earlier successful `tribe.json` is kept.
- Cache key: source SHA-256, model revision, language and Hookline's TRIBE schema version. The
  endpoint URL isn't part of it.
- The response is checked before anything is stored:
  - model revision
  - the fsaverage5 vertex count
  - byte length
  - finite values
  - one segment per row, in time order
