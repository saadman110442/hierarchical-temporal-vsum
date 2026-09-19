"""
main.py
-------
FastAPI server for the video-summarization showcase.

Endpoints
---------
  GET  /                       -> serves frontend/index.html
  GET  /<static>               -> serves frontend/* (css, js, images)
  POST /api/summarize          -> upload a video, returns {"job_id": ...}
  GET  /api/status/{job_id}    -> poll: stage, progress, result (when done)
  GET  /api/video/{job_id}/original   -> HTTP-Range video stream
  GET  /api/video/{job_id}/summary    -> HTTP-Range video stream
  GET  /api/health             -> simple health check

Design notes
------------
* The model is loaded exactly once at startup.
* Jobs run on a SerialExecutor (one at a time) to keep VRAM usage predictable
  on free-tier hardware. Concurrent uploads queue.
* Videos are served with HTTP Range support so <video> can seek/stream without
  downloading the whole file.
* Storage layout:
      storage/<job_id>/input.<ext>      (user upload)
      storage/<job_id>/original.mp4     (H.264 re-encoded)
      storage/<job_id>/summary.mp4      (H.264 re-encoded)
      storage/<job_id>/meta.json        (result dict)
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import inference

# ------------------------------------------------------------
# Configuration (env-driven)
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = (ROOT.parent / "frontend").resolve()
STORAGE_DIR = (ROOT / "storage").resolve()
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    str(ROOT / "models" / "best.pt"),
)
CHECKPOINT_PATH_PART2 = os.environ.get(
    "CHECKPOINT_PATH_PART2",
    str(ROOT / "models" / "best_part2.pt"),
)
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "15"))
# Part 2 is expensive per-frame (BLIP-2 captioning), so we subsample more
# aggressively by default. Users can override by setting SAMPLE_RATE_PART2.
SAMPLE_RATE_PART2 = int(os.environ.get("SAMPLE_RATE_PART2", "30"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
SUMMARY_PROPORTION = float(os.environ.get("SUMMARY_PROPORTION", "0.20"))

# Retention: delete jobs older than this (seconds). 0 = never delete.
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", str(6 * 3600)))

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ------------------------------------------------------------
# In-memory job registry
# ------------------------------------------------------------

@dataclass
class JobState:
    job_id: str
    status: str = "queued"   # queued | running | done | error
    stage: str = "queued"    # extract_features | kts | model | knapsack | encode | reencode | done
    progress: float = 0.0    # 0..1 within current stage
    overall: float = 0.0     # 0..1 across pipeline
    message: str = ""
    variant: str = "part1"   # part1 | part2
    device: str = "auto"     # auto | cpu | cuda
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    original_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Don't ship huge arrays to the client on every poll
        if d.get("result"):
            r = dict(d["result"])
            r.pop("importance_scores", None)
            r.pop("picks", None)
            d["result"] = r
        return d


JOBS: Dict[str, JobState] = {}


# Stage-to-overall-progress mapping. These weights are approximate but give
# the user a believable progress bar. Feature extraction is by far the
# dominant cost on CPU; model forward is negligible.
STAGE_WEIGHTS = {
    "queued":           (0.00, 0.00),
    # Part 2 only: lazy-load CLIP + BLIP-2 on first Part 2 request
    "load_multimodal":  (0.00, 0.05),
    "extract_features": (0.05, 0.70),
    "kts":              (0.70, 0.78),
    "model":            (0.78, 0.82),
    "knapsack":         (0.82, 0.84),
    "encode":           (0.84, 0.94),
    "reencode":         (0.94, 1.00),
    "done":             (1.00, 1.00),
}


def _overall_progress(stage: str, stage_progress: float) -> float:
    lo, hi = STAGE_WEIGHTS.get(stage, (0.0, 1.0))
    p = max(0.0, min(1.0, stage_progress))
    return lo + (hi - lo) * p


def _stage_message(stage: str, meta: Dict[str, Any]) -> str:
    if stage == "load_multimodal":
        return meta.get("note") or "Loading CLIP + BLIP-2…"
    if stage == "extract_features":
        total = meta.get("total")
        frame = meta.get("frame")
        caption = meta.get("caption")
        base = (
            f"Extracting features… frame {frame:,}/{total:,}"
            if (total and frame)
            else "Extracting features…"
        )
        if caption:
            return f"{base}  ·  \u201C{caption[:60]}\u201D"
        return base
    if stage == "kts":
        n = meta.get("num_shots")
        return f"Detecting shots… {n} found" if n else "Detecting shot boundaries…"
    if stage == "model":
        return "Running transformer model…"
    if stage == "knapsack":
        sel = meta.get("selected_frames")
        return f"Selecting keyshots… {sel:,} frames" if sel else "Selecting keyshots…"
    if stage == "encode":
        return "Writing summary video…"
    if stage == "reencode":
        return "Finalizing (H.264)…"
    if stage == "done":
        return "Done"
    return stage


# ------------------------------------------------------------
# App
# ------------------------------------------------------------

app = FastAPI(title="Video Summarization — Live Demo")

# One job at a time — avoids GPU OOM on free tiers
_job_lock = asyncio.Lock()


@app.on_event("startup")
async def _startup() -> None:
    loaded_any = False

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[startup] Loading Part 1 model from {CHECKPOINT_PATH} …")
        try:
            inference.load_model_part1(CHECKPOINT_PATH)
            loaded_any = True
        except Exception as e:
            print(f"[startup] Part 1 load failed: {type(e).__name__}: {e}")
    else:
        print(f"[startup] Part 1 checkpoint not found at {CHECKPOINT_PATH}")

    if os.path.exists(CHECKPOINT_PATH_PART2):
        print(f"[startup] Loading Part 2 model from {CHECKPOINT_PATH_PART2} …")
        try:
            inference.load_model_part2(CHECKPOINT_PATH_PART2)
            loaded_any = True
        except Exception as e:
            print(f"[startup] Part 2 load failed: {type(e).__name__}: {e}")
    else:
        print(f"[startup] Part 2 checkpoint not found at {CHECKPOINT_PATH_PART2}")
        print("[startup] Part 2 will be hidden in the UI until best_part2.pt is added.")

    if not loaded_any:
        print("[startup] WARNING: no checkpoints loaded. /api/summarize will fail.")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        # Legacy field: still reports Part 1 status so existing clients keep working
        "model_loaded": inference._MODEL_PART1 is not None,
        "part1_loaded": inference._MODEL_PART1 is not None,
        "part2_loaded": inference._MODEL_PART2 is not None,
        "mm_extractor_warm": inference._MM_FEATS is not None,
        "device": str(inference._DEVICE) if inference._DEVICE else "unloaded",
        "cuda": inference.cuda_info(),
        "checkpoint": CHECKPOINT_PATH,
        "checkpoint_part2": CHECKPOINT_PATH_PART2,
        "ffmpeg_available": inference._have_ffmpeg(),
    }


@app.post("/api/device")
async def switch_device(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Live-switch the inference device. Body: {"device": "cpu" | "cuda" | "auto"}.

    Moves all loaded models to the new device. Returns the resolved device
    string. Useful when the user toggles the CPU/GPU switch in the UI without
    submitting a video.
    """
    name = str(payload.get("device") or "").strip()
    if not name:
        raise HTTPException(400, "Missing 'device' in body.")
    try:
        new_dev = inference.set_device(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"device": str(new_dev), "cuda": inference.cuda_info()}


# ------------------------------------------------------------
# Upload + background job
# ------------------------------------------------------------

def _run_job_sync(job_id: str, input_path: str, out_dir: str, variant: str, device: str) -> None:
    job = JOBS[job_id]
    job.status = "running"
    job.stage = "extract_features"
    job.started_at = time.time()

    def cb(stage: str, stage_progress: float, meta: Dict[str, Any]) -> None:
        job.stage = stage
        job.progress = stage_progress
        job.overall = _overall_progress(stage, stage_progress)
        job.message = _stage_message(stage, meta)

    sample_rate = SAMPLE_RATE_PART2 if variant == "part2" else SAMPLE_RATE
    try:
        result = inference.summarize_video(
            video_path=input_path,
            out_dir=out_dir,
            sample_rate=sample_rate,
            summary_proportion=SUMMARY_PROPORTION,
            progress_cb=cb,
            variant=variant,
            device=device,
        )
        job.result = result
        job.status = "done"
        job.stage = "done"
        job.progress = 1.0
        job.overall = 1.0
        job.message = "Done"
        job.finished_at = time.time()

        # Persist metadata for cleanup/debug
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "original_name": job.original_name,
                    **{k: v for k, v in result.items() if k not in ("importance_scores", "picks")},
                },
                f,
                indent=2,
            )
    except Exception as e:
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
        job.finished_at = time.time()
        print(f"[job {job_id}] ERROR: {job.error}")


async def _run_job_async(job_id: str, input_path: str, out_dir: str, variant: str, device: str) -> None:
    async with _job_lock:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_job_sync, job_id, input_path, out_dir, variant, device)


def _cleanup_old_jobs() -> None:
    if JOB_TTL_SECONDS <= 0:
        return
    now = time.time()
    for jid in list(JOBS.keys()):
        j = JOBS[jid]
        if now - j.created_at > JOB_TTL_SECONDS:
            d = STORAGE_DIR / jid
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            JOBS.pop(jid, None)


@app.post("/api/summarize")
async def summarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    variant: str = Form("part1"),
    device: str = Form("auto"),
) -> Dict[str, Any]:
    variant = (variant or "part1").lower()
    if variant not in ("part1", "part2"):
        raise HTTPException(400, f"Unknown variant '{variant}'. Use 'part1' or 'part2'.")

    # Validate the device string now so we fail fast (before saving the upload).
    # "auto"/"" is the soft default — pipeline resolves it against env + cuda.
    device = (device or "auto").strip().lower()
    if device not in ("auto", "cpu", "cuda", "gpu") and not device.startswith("cuda:"):
        raise HTTPException(400, f"Unknown device '{device}'. Use 'auto', 'cpu', or 'cuda'.")
    if device in ("cuda", "gpu") or device.startswith("cuda:"):
        import torch as _torch
        if not _torch.cuda.is_available():
            raise HTTPException(
                400,
                "GPU requested but PyTorch reports no CUDA device available on this machine. "
                "Either pick CPU, or install a CUDA build of torch (see setup_cuda.ps1).",
            )

    if variant == "part1" and inference._MODEL_PART1 is None:
        raise HTTPException(503, "Part 1 model is not loaded on the server.")
    if variant == "part2" and inference._MODEL_PART2 is None:
        raise HTTPException(
            503,
            "Part 2 model is not loaded on the server. Place best_part2.pt in models/ and restart.",
        )

    _cleanup_old_jobs()

    # Basic validation
    name = file.filename or "upload.mp4"
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}"
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{ext}"
    size_bytes = 0
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    try:
        with open(input_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    out.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        413, f"File exceeds {MAX_UPLOAD_MB} MB limit."
                    )
                out.write(chunk)
    finally:
        await file.close()

    job = JobState(job_id=job_id, original_name=name, variant=variant, device=device)
    JOBS[job_id] = job

    background_tasks.add_task(_run_job_async, job_id, str(input_path), str(job_dir), variant, device)
    return {"job_id": job_id, "variant": variant, "device": device}


@app.get("/api/status/{job_id}")
async def status(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return job.to_public_dict()


# ------------------------------------------------------------
# HTTP-Range video streaming
# ------------------------------------------------------------

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


def _stream_file(path: Path, request: Request) -> Response:
    if not path.exists():
        raise HTTPException(404, "File not found")

    file_size = path.stat().st_size
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "video/mp4"

    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header is None:
        # Whole-file response
        return FileResponse(path, media_type=mime)

    m = _RANGE_RE.search(range_header)
    if not m:
        raise HTTPException(416, "Invalid Range header")

    start = int(m.group(1))
    end_s = m.group(2)
    end = int(end_s) if end_s else file_size - 1
    end = min(end, file_size - 1)
    if start > end:
        raise HTTPException(416, "Range out of bounds")

    chunk_size = 1024 * 1024

    def iterfile():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }
    return StreamingResponse(iterfile(), status_code=206, media_type=mime, headers=headers)


@app.get("/api/video/{job_id}/original")
async def video_original(job_id: str, request: Request) -> Response:
    path = STORAGE_DIR / job_id / "original.mp4"
    return _stream_file(path, request)


@app.get("/api/video/{job_id}/summary")
async def video_summary(job_id: str, request: Request) -> Response:
    path = STORAGE_DIR / job_id / "summary.mp4"
    return _stream_file(path, request)


@app.get("/api/video/{job_id}/download")
async def video_download(job_id: str) -> Response:
    path = STORAGE_DIR / job_id / "summary.mp4"
    if not path.exists():
        raise HTTPException(404, "Summary not ready")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"summary_{job_id}.mp4",
    )


# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    idx = FRONTEND_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=500)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


# Serve everything else under /static/* from the frontend dir (js, css, images)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
