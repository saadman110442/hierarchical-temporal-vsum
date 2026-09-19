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

Interactive demo for two published supervised video summarization models.
Upload a video, pick a model, and watch the keyshot summary play back next to
the original in the browser.

**Saadman Sakib** · Chittagong University of Engineering & Technology

> **Scope:** this repository is the **web demo** — a FastAPI backend that serves
> the trained models and a static project page that drives them. It is not the
> training code. For the methods, datasets, ablations and full results, see the
> papers below.

---

## Papers

**Part 1 — HybridHiT-UNet: Multi-Scale Temporal U-Net with Hierarchical Shot-Aware Transformers for Video Summarization**
Saadman Sakib, Tanjim Mahmud, Karl Andersson, Kaushik Deb
*Machine Learning and Knowledge Extraction* **8**(5), 135, 2026
[https://doi.org/10.3390/make8050135](https://doi.org/10.3390/make8050135) · [MDPI (open access)](https://www.mdpi.com/2504-4990/8/5/135)

**Part 2 — Multimodal Video Summarization Using Vision-Language Embeddings and Hierarchical Temporal Modeling**
Saadman Sakib, Kaushik Deb
*Applied AI Letters* **7**(3), e70039, 2026
[https://doi.org/10.1002/ail2.70039](https://doi.org/10.1002/ail2.70039) · [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/ail2.70039)

---

## The two models

| | Part 1 — HybridHiT-UNet | Part 2 — MM-HiT-UNet |
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
│   ├── inference.py       ← Model definitions + inference pipeline
│   ├── test_inference.py  ← CLI sanity check
│   └── models/            ← checkpoints go here
├── frontend/
│   ├── index.html         ← Project page + demo UI
│   └── app.js             ← Upload, progress polling, video players
├── Dockerfile             ← Hugging Face Spaces build
└── setup.ps1 / run.ps1    ← Windows setup and launch
```

No React, no build step, no bundler. Tailwind from CDN, vanilla JS.

## Running it

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)**. On Windows it's two commands:

```powershell
.\setup.ps1
.\run.ps1
```

Then open http://localhost:8000. `ffmpeg` is required — without it the
summaries won't play in Chrome or Safari.

## Checkpoints

The trained weights are 432 MB each, so they're attached to the
**[latest release](../../releases/latest)** rather than committed:

```bash
gh release download --repo saadman110442/hierarchical-temporal-vsum \
    --pattern "*.pt" --dir backend/models
```

Part 1 needs `best.pt`, Part 2 needs `best_part2.pt`. The server starts with
whichever it finds and returns HTTP 503 for a variant whose checkpoint is
missing. Part 2 additionally pulls CLIP and BLIP-2 (~12 GB) from the
HuggingFace hub on its first request.

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

Jobs run serially behind an `asyncio.Lock` so free-tier hardware doesn't run
out of memory. To allow concurrency, drop `async with _job_lock:` in
`main.py → _run_job_async`.

Runtime behaviour is configured with environment variables — checkpoint paths,
`SAMPLE_RATE` (15) and `SAMPLE_RATE_PART2` (30), `SUMMARY_PROPORTION` (0.20),
`MAX_UPLOAD_MB` (200), `JOB_TTL_SECONDS` (6h), `VSUM_DEVICE`, and the CLIP/BLIP
model ids. See [backend/.env.example](backend/.env.example).

---

## Citation

```bibtex
@article{sakib2026hybridhitunet,
  title   = {HybridHiT-UNet: Multi-Scale Temporal U-Net with Hierarchical
             Shot-Aware Transformers for Video Summarization},
  author  = {Sakib, Saadman and Mahmud, Tanjim and Andersson, Karl and Deb, Kaushik},
  journal = {Machine Learning and Knowledge Extraction},
  volume  = {8},
  number  = {5},
  pages   = {135},
  year    = {2026},
  doi     = {10.3390/make8050135},
}

@article{sakib2026multimodal,
  title   = {Multimodal Video Summarization Using Vision-Language Embeddings
             and Hierarchical Temporal Modeling},
  author  = {Sakib, Saadman and Deb, Kaushik},
  journal = {Applied AI Letters},
  volume  = {7},
  number  = {3},
  pages   = {e70039},
  year    = {2026},
  doi     = {10.1002/ail2.70039},
}
```

## License

MIT — see [LICENSE](LICENSE).
