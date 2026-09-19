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

# Video Summarization — Academic Project Page + Live Demo

This repo is a ready-to-deploy project page + working demo for your supervised
video summarization model (ViT encoder + Temporal U-Net + Hierarchical score head).

Upload a video → get a keyshot summary. Both videos play side-by-side in the browser.

```
vsum-web/
├── backend/
│   ├── main.py            ← FastAPI server (upload, polling, streaming)
│   ├── inference.py       ← Your notebook's model + pipeline, wrapped for serving
│   ├── requirements.txt
│   ├── .env.example       ← env vars you can tweak
│   ├── models/            ← put your .pt checkpoint here
│   └── storage/           ← per-job upload/summary files (auto-created)
├── frontend/
│   ├── index.html         ← Academic showcase page (hero, abstract, method, demo, BibTeX)
│   └── app.js             ← Upload, progress polling, video players
├── Dockerfile             ← For Hugging Face Spaces (free CPU)
├── .gitignore
└── README.md
```

No React, no build step, no bundler. Tailwind is loaded from CDN, JS is vanilla.

---

## ✏️ Before you do anything else — personalize the page

Open `frontend/index.html` and replace the placeholders:

| Search for | What to change |
|---|---|
| `Under Review · 2026` | Your venue (e.g. `CVPR 2026`, `IEEE TPAMI`, etc.) |
| `First Author`, `Second Author`, `Third Author` | Your author list + update `href="#"` to personal pages |
| `Your University`, `Your Lab / Affiliation` | Your affiliations |
| The four top buttons (Paper, arXiv, Code, Demo) | Real URLs |
| `<p>…Supervised video summarization aims to…</p>` | Your actual abstract |
| The `Results` table | Your F1 numbers + competitor baselines |
| The `BibTeX` block | Your real citation |

Everything else (method diagram, pipeline description) is already written to match
your actual architecture from the notebook, so it should need minimal edits.

---

## 🏃 Run locally

### 1. Prerequisites

- Python 3.10 or 3.11
- `ffmpeg` installed and on your PATH
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: `choco install ffmpeg` or download from ffmpeg.org
- Your trained `.pt` checkpoint (e.g. `tvsum_canonical_split4_best.pt` from your notebook)

### 2. Install

```bash
cd vsum-web/backend
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU note**: the default `requirements.txt` grabs the CPU wheel of torch on most
> platforms. If you have CUDA, install the matching wheel *first* before the other
> requirements, e.g.:
> ```bash
> pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

### 3. Download the trained checkpoint

The weights are too large for a normal git commit (432 MB each), so they are
attached to the **[latest GitHub Release](../../releases/latest)**. Download
`best.pt` into `backend/models/`:

```bash
# with the GitHub CLI
gh release download --repo saadman110442/hierarchical-temporal-vsum \
    --pattern "best.pt" --dir backend/models

# or with curl
curl -L -o backend/models/best.pt \
  https://github.com/saadman110442/hierarchical-temporal-vsum/releases/latest/download/best.pt
```

`best_part2.pt` is the second-stage checkpoint and is attached to the same
release; download it the same way if you need it.

To use a checkpoint stored somewhere else, set `CHECKPOINT_PATH` instead of
copying it in.

### 4. (Recommended) Sanity check

Before starting the web server, confirm the model loads and runs on a real
video. This catches config/checkpoint mismatches and missing ffmpeg in seconds:

```bash
cd backend
python test_inference.py path/to/any/video.mp4
```

Expected output ends with `SUCCESS` and writes two files to `./test_output/`.
If this works, the server will work. If it doesn't, the most common causes
are printed on screen (missing ffmpeg, checkpoint/config mismatch, etc.).

### 5. Run the server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 — upload a video — wait — watch the summary.

---

## ☁️ Deploy to Hugging Face Spaces (free, always-on)

This is the free-tier path: the free CPU tier hosts your site 24/7 without a credit
card. Inference will be slower than GPU (≈ 1.5–3× realtime for features on CPU)
but works fine for a ~1–2 min demo video.

### 1. Create the Space

1. Sign up at https://huggingface.co (free)
2. Click **New Space** → give it a name → choose **Docker** as the SDK → choose
   hardware **CPU basic (free)** → Public
3. Clone the empty Space repo HuggingFace creates for you

### 2. Push this project

```bash
cd <your-new-space-repo>
# copy everything from vsum-web/ into here
cp -r /path/to/vsum-web/* .
git add .
git commit -m "Initial demo"
git push
```

### 3. Upload the checkpoint — two options

**Option A (simple, public):** Your checkpoint is small enough to commit.
Track `.pt` via git-lfs:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
cp /path/to/tvsum_canonical_split4_best.pt backend/models/best.pt
git add backend/models/best.pt
git commit -m "Add trained weights"
git push
```

**Option B (private checkpoint):** Upload the `.pt` to a separate **private** HF
model repo, then download it at Space build time. Edit the end of `Dockerfile`:

```dockerfile
# Before CMD, add:
ARG HF_TOKEN
RUN pip install --user huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='your-username/your-private-repo', \
                    filename='best.pt', \
                    local_dir='./backend/models', \
                    token='$HF_TOKEN')"
```

Then in your Space **Settings → Secrets**, add `HF_TOKEN` = your HF access token.

### 4. Space settings

In **Settings → Variables**, add the optional env vars from `.env.example`
(CHECKPOINT_PATH, SAMPLE_RATE, SUMMARY_PROPORTION, MAX_UPLOAD_MB, JOB_TTL_SECONDS).
Defaults are fine if you just put `best.pt` in `backend/models/`.

First build takes ~5–10 min (downloads torch + timm pretrained weights).

---

## 🚀 Free GPU via Colab (for live-demo days)

If you need fast inference for a specific demo day — presentation, thesis defense,
job interview — you can keep the HF Spaces page as your permanent address but
temporarily route inference to a free Colab GPU.

The trick: run the FastAPI server inside Colab, expose it with a Cloudflare
tunnel, and point your HF Spaces frontend at the tunnel URL.

### 1. In Colab (free T4 GPU runtime)

Paste this into a cell:

```python
# Install deps
!pip install -q fastapi uvicorn python-multipart timm opencv-python-headless pillow
!pip install -q cloudflared || (wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared)
!apt-get install -y -qq ffmpeg

# Clone your space repo (or upload files manually)
!git clone https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME /content/vsum-web
%cd /content/vsum-web/backend

# Point to your Drive checkpoint
import os
os.environ['CHECKPOINT_PATH'] = '/content/drive/MyDrive/path/to/best.pt'

# Mount Drive (to read checkpoint)
from google.colab import drive
drive.mount('/content/drive')

# Start server in background
import subprocess, time
server = subprocess.Popen(['uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000'])
time.sleep(8)

# Open public tunnel
tunnel = subprocess.Popen(
    ['cloudflared', 'tunnel', '--url', 'http://localhost:8000'],
    stderr=subprocess.PIPE,
)
# Watch the stderr for the public URL — it prints like:
#   https://whatever-words.trycloudflare.com
```

The tunnel URL in the `cloudflared` output is now your GPU-backed demo.
Share that URL directly, or…

### 2. (Optional) Route your HF Space to the Colab backend

Temporarily make your HF Space frontend-only by pointing its API calls at
the Colab tunnel. Add this to the top of `frontend/app.js` and redeploy:

```js
const API_BASE = 'https://whatever-words.trycloudflare.com';
// then replace every `fetch('/api/...')` with `fetch(API_BASE + '/api/...')`
// and every video src with `${API_BASE}/api/video/...`
```

When you're done demoing, revert `API_BASE` back to `''`.

---

## ⚙️ Configuration

All runtime options are environment variables. See `backend/.env.example`:

| Variable | Default | Purpose |
|---|---|---|
| `CHECKPOINT_PATH` | `./models/best.pt` | Path to your trained `.pt` file |
| `SAMPLE_RATE` | `15` | Frame subsampling (same as training) |
| `SUMMARY_PROPORTION` | `0.20` | Knapsack budget as fraction of video length |
| `MAX_UPLOAD_MB` | `200` | Reject larger uploads with 413 |
| `JOB_TTL_SECONDS` | `21600` (6h) | Auto-delete old job data |

Model hyperparameters are hardcoded in `backend/inference.py → BEST_CFG`. This
matches the final TVSum config from your notebook exactly (ViT + UNet,
levels=3, base_channels=128, unet_use_attention=True, hier head, etc.).

If your best checkpoint was trained with *different* hyperparameters (e.g. a
SumMe split with a different head), edit `BEST_CFG`. The model will fail to load
with a clear shape error if the config doesn't match the `.pt` file.

---

## 🔧 API reference

| Method | Path | What it does |
|---|---|---|
| `POST` | `/api/summarize` | Upload a video (multipart/form-data, field `file`). Returns `{"job_id": "..."}` |
| `GET` | `/api/status/{job_id}` | Poll. Returns `{status, stage, progress, overall, message, result?}` |
| `GET` | `/api/video/{job_id}/original` | HTTP-Range-streamed original video |
| `GET` | `/api/video/{job_id}/summary` | HTTP-Range-streamed summary video |
| `GET` | `/api/video/{job_id}/download` | Summary with `Content-Disposition: attachment` |
| `GET` | `/api/health` | `{ok, model_loaded, device, ffmpeg_available}` |

Pipeline stages reported via `stage`:
`extract_features → kts → model → knapsack → encode → reencode → done`

---

## ❓ Troubleshooting

**"RuntimeError: Error(s) in loading state_dict"**
Your checkpoint was trained with different hyperparameters than `BEST_CFG`.
Print the config from your training notebook and update `BEST_CFG` accordingly.

**"Model is not loaded on the server" (HTTP 503)**
The server started without finding a checkpoint at `CHECKPOINT_PATH`. Check
`/api/health` — it'll tell you the path it tried.

**Summary video doesn't play in the browser**
Install `ffmpeg`. OpenCV's native `mp4v` encoding produces files that Chrome /
Safari often refuse to play; the server auto-reencodes to H.264 via ffmpeg,
but only if ffmpeg is available on PATH. Check with `GET /api/health`.

**Spaces build fails with "No space left on device"**
The torch+torchvision install is ~2 GB. The Dockerfile already uses CPU-only
wheels to keep the image small. If it still fails, try a smaller image base
(e.g. `python:3.11-slim` is what we use — don't switch to the full image).

**Inference is too slow on CPU**
A ~2 min video at 30 fps has ~3,600 frames → 240 features (at sample_rate=15).
GoogleNet on CPU runs those in ~60–120 s. Model forward + knapsack are <1 s.
The re-encode step depends on output length. On free CPU, ~2–4 min end-to-end
for a 2-min input is realistic. For faster demos, use the Colab path above.

**I want multiple users to upload simultaneously**
By default jobs run serially (one `asyncio.Lock`). This is intentional for
free-tier hardware. To lift it, remove `async with _job_lock:` in
`main.py → _run_job_async`. You may run out of VRAM or RAM on free tiers.
