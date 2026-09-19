---
title: Video Summarization
emoji: 🎬
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Hierarchical Temporal Transformer — upload a video, get an AI-generated keyshot summary
---

<!--
  The YAML block above is read by Hugging Face Spaces.
  Do NOT delete it if you deploy to HF Spaces.
  Everything below it is ordinary Markdown shown on the Space's About tab.
-->

# From Visual-Only to Vision-Language Multimodal Video Summarization

Reference implementation and live demo for two supervised video summarization
models. Upload a video, pick a model, and get a keyshot summary played back
next to the original in the browser.

**Saadman Sakib** · Chittagong University of Engineering & Technology

| | Part 1 — HiT-UNet | Part 2 — MM-HiT-UNet |
|---|---|---|
| Features | Visual-only (GoogLeNet + ViT) | Vision-language: BLIP-2 captions each frame, CLIP embeds image *and* caption, 1024-d concat |
| F1 (SumMe) | 65.8% | 59.27% |
| Params | — | 28.27M |
| Checkpoint | `best.pt` | `best_part2.pt` |
| Speed | ~5–10 s / video | Slower; BLIP-2 captioning dominates |

Both share the same backbone: a 3-level multi-scale Temporal U-Net with
hierarchical shot-aware transformers, KTS shot segmentation, and 0/1-knapsack
keyshot selection.

---

## What's in here

```
vsum-web/
├── backend/
│   ├── main.py            ← FastAPI server (upload, polling, range-streaming)
│   ├── inference.py       ← Models + full pipeline
│   ├── test_inference.py  ← CLI sanity check
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/            ← checkpoints go here (see below)
│   └── storage/           ← per-job files (auto-created, auto-expired)
├── frontend/
│   ├── index.html         ← Project page + demo UI
│   └── app.js             ← Upload, progress polling, video players
├── Dockerfile             ← Hugging Face Spaces build
├── setup.ps1 / run.ps1    ← Windows one-shot setup and launch
└── setup_cuda.ps1         ← Swap in a CUDA build of torch
```

No React, no build step, no bundler. Tailwind from CDN, vanilla JS.

---

## Quick start (Windows)

```powershell
.\setup.ps1     # installs Python 3.11 + ffmpeg if missing, builds venv, installs deps
.\run.ps1       # starts the server on http://localhost:8000
```

See [HOW_TO_RUN.md](HOW_TO_RUN.md) for details and troubleshooting.

## Quick start (macOS / Linux)

**Prerequisites:** Python 3.10 or 3.11, and `ffmpeg` on your PATH
(`brew install ffmpeg` / `sudo apt install ffmpeg`). ffmpeg is not optional —
without it, summary videos won't play in Chrome or Safari.

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the matching torch wheel *before* the other requirements:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Get the checkpoints

The weights are 432 MB each — too large for a normal git commit — so they're
attached to the **[latest release](../../releases/latest)**:

```bash
# GitHub CLI
gh release download --repo saadman110442/hierarchical-temporal-vsum \
    --pattern "*.pt" --dir backend/models

# or curl (Part 1 only; swap the filename for best_part2.pt)
curl -L -o backend/models/best.pt \
  https://github.com/saadman110442/hierarchical-temporal-vsum/releases/latest/download/best.pt
```

Part 1 needs `best.pt`; Part 2 needs `best_part2.pt`. The server starts with
whichever it finds and returns HTTP 503 for a variant whose checkpoint is missing.

### Sanity check, then run

```bash
cd backend
python test_inference.py path/to/any/video.mp4   # ends in SUCCESS if all is well
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000.

> **Heads-up on Part 2:** CLIP and BLIP-2 are lazy-loaded on the first Part 2
> request and download ~12 GB to the HuggingFace cache. Peak RAM is ~8 GB on
> CPU. If you only want Part 1, you can `pip uninstall transformers accelerate
> sentencepiece` after install. On Windows, `move_hf_cache.ps1` relocates that
> cache off your system drive.

---

## Configuration

All runtime options are environment variables (see `backend/.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `CHECKPOINT_PATH` | `./models/best.pt` | Part 1 checkpoint |
| `CHECKPOINT_PATH_PART2` | `./models/best_part2.pt` | Part 2 checkpoint |
| `SAMPLE_RATE` | `15` | Part 1 frame subsampling |
| `SAMPLE_RATE_PART2` | `30` | Part 2 frame subsampling |
| `SUMMARY_PROPORTION` | `0.20` | Knapsack budget as a fraction of video length |
| `MAX_UPLOAD_MB` | `200` | Larger uploads are rejected with 413 |
| `JOB_TTL_SECONDS` | `21600` | Auto-delete job data after 6h (`0` disables) |
| `VSUM_DEVICE` | unset | Force `cpu` or `cuda` |
| `CLIP_MODEL_ID` | `openai/clip-vit-base-patch32` | Part 2 vision encoder |
| `BLIP_MODEL_ID` | see `inference.py` | Part 2 captioner |
| `BLIP_MAX_TOKENS` | `24` | Caption length cap |

Model hyperparameters live in `backend/inference.py → BEST_CFG`. A checkpoint
trained with a different config fails to load with an explicit shape error.

---

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/summarize` | multipart: `file`, `variant` (`part1`\|`part2`), `device` (`auto`\|`cpu`\|`cuda`). Returns `{"job_id": …}` |
| `GET` | `/api/status/{job_id}` | Poll: `{status, stage, progress, overall, message, result?}` |
| `GET` | `/api/video/{job_id}/original` | Range-streamed original |
| `GET` | `/api/video/{job_id}/summary` | Range-streamed summary |
| `GET` | `/api/video/{job_id}/download` | Summary as an attachment |
| `GET` | `/api/health` | Per-variant load state, device, CUDA info, ffmpeg availability |
| `POST` | `/api/device` | Live-switch device: `{"device": "cpu"\|"cuda"\|"auto"}` |

Stages reported via `stage`:
`load_multimodal` (Part 2 only) `→ extract_features → kts → model → knapsack → encode → reencode → done`

Jobs run serially behind an `asyncio.Lock` — deliberate, so free-tier hardware
doesn't run out of memory. To allow concurrency, drop `async with _job_lock:`
in `main.py → _run_job_async`.

---

## Deploy to Hugging Face Spaces

The free CPU tier hosts this 24/7 without a credit card. Create a **Docker**
Space (CPU basic, Public), push this repo to it, and add the checkpoints
either with `git lfs track "*.pt"` or by pulling them from a private HF model
repo at build time using an `HF_TOKEN` secret. First build takes ~5–10 min.

Expect ~2–4 min end-to-end for a 2-minute video on free CPU. For live demos
that need to be fast, run the same server on a Colab T4 and expose it with a
`cloudflared` tunnel, then point `API_BASE` in `frontend/app.js` at the tunnel URL.

---

## Troubleshooting

**`RuntimeError: Error(s) in loading state_dict`** — the checkpoint's
hyperparameters don't match `BEST_CFG` in `inference.py`.

**HTTP 503 on summarize** — that variant's checkpoint wasn't found. Check
`/api/health`; it reports the exact paths it tried.

**Summary video won't play** — ffmpeg isn't on PATH. OpenCV's `mp4v` output
is rejected by most browsers; the server re-encodes to H.264 only if ffmpeg is
available. Confirm via `/api/health`.

**Part 2 is very slow or OOMs** — BLIP-2 captioning is the bottleneck and wants
~8 GB RAM on CPU. Use Part 1, or a GPU.

---

## Citation

```bibtex
@article{sakib2026hitunet,
  title   = {HiT-UNet: Multi-Scale Temporal U-Net with Hierarchical Shot-Aware
             Transformers for Video Summarization},
  author  = {Sakib, Saadman and Mahmud, Tanjim and Andersson, Karl and Deb, Kaushik},
  journal = {Under Review},
  year    = {2026},
}

@article{sakib2026multimodal,
  title   = {Multimodal Video Summarization Using Vision-Language Embeddings
             and Hierarchical Temporal Modeling},
  author  = {Sakib, Saadman and Deb, Kaushik},
  journal = {Applied AI Letters},
  year    = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
