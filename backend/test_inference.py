"""
test_inference.py
-----------------
Standalone sanity check for inference.py. Run this BEFORE starting the web
server to verify:

  1. All Python imports resolve (torch, timm, cv2, etc.).
  2. ffmpeg / ffprobe are installed.
  3. Your checkpoint matches the BEST_CFG hyperparameters.
  4. Inference runs end-to-end on a real video and produces a playable mp4.

Usage
-----
    cd backend
    python test_inference.py path/to/any_video.mp4

    # Or use a custom checkpoint path / sample rate:
    CHECKPOINT_PATH=./models/best.pt python test_inference.py video.mp4

Exit codes
----------
    0  = everything worked, summary written to ./test_output/summary.mp4
    1  = anything failed (see printed traceback)

If this script succeeds, `uvicorn main:app` should also succeed.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

# Make sure we import the local inference module, not anything installed globally
sys.path.insert(0, str(Path(__file__).resolve().parent))

import inference  # noqa: E402


def main(video_path: str) -> int:
    print("=" * 70)
    print("  Video Summarizer — Sanity Check")
    print("=" * 70)

    # ----- 1. Environment ------------------------------------------------
    import torch
    print(f"\n[1/5] Environment")
    print(f"      Python          : {sys.version.split()[0]}")
    print(f"      PyTorch         : {torch.__version__}")
    print(f"      CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU             : {torch.cuda.get_device_name(0)}")

    # ----- 2. ffmpeg -----------------------------------------------------
    print(f"\n[2/5] ffmpeg")
    have = inference._have_ffmpeg()
    print(f"      ffmpeg+ffprobe on PATH : {have}")
    if not have:
        print("      ✗ ffmpeg is required. Install it:")
        print("        macOS:  brew install ffmpeg")
        print("        Ubuntu: sudo apt-get install ffmpeg")
        return 1

    # ----- 3. Checkpoint -------------------------------------------------
    ckpt = os.environ.get("CHECKPOINT_PATH", "./models/best.pt")
    print(f"\n[3/5] Checkpoint")
    print(f"      Looking for: {ckpt}")
    if not os.path.exists(ckpt):
        print(f"      ✗ not found.")
        print(f"        Copy your .pt file to backend/models/best.pt, or set")
        print(f"        CHECKPOINT_PATH=/abs/path/to/your.pt before running.")
        return 1
    size_mb = os.path.getsize(ckpt) / (1024 * 1024)
    print(f"      ✓ found ({size_mb:.1f} MB)")

    # ----- 4. Model loading ---------------------------------------------
    print(f"\n[4/5] Loading model (this downloads pretrained ViT + GoogleNet on first run)")
    t0 = time.time()
    try:
        inference.load_model(ckpt)
    except RuntimeError as e:
        msg = str(e)
        print(f"      ✗ load_state_dict failed: {msg[:300]}")
        if "size mismatch" in msg or "Missing key" in msg or "Unexpected key" in msg:
            print()
            print("      This almost always means your checkpoint was trained with")
            print("      different hyperparameters than inference.BEST_CFG.")
            print("      Open backend/inference.py and edit BEST_CFG to match the")
            print("      config your notebook used when saving this .pt file.")
        return 1
    print(f"      ✓ loaded in {time.time()-t0:.1f}s on {inference._DEVICE}")

    # ----- 5. End-to-end inference --------------------------------------
    print(f"\n[5/5] End-to-end inference on: {video_path}")
    if not os.path.exists(video_path):
        print(f"      ✗ video file not found")
        return 1

    out_dir = Path("./test_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def cb(stage, pct, meta):
        print(f"      [{stage:>18s}] {pct*100:5.1f}%  {meta or ''}")

    t0 = time.time()
    try:
        result = inference.summarize_video(
            video_path=video_path,
            out_dir=str(out_dir),
            sample_rate=int(os.environ.get("SAMPLE_RATE", "15")),
            summary_proportion=float(os.environ.get("SUMMARY_PROPORTION", "0.20")),
            progress_cb=cb,
        )
    except Exception as e:
        print(f"      ✗ inference failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print(f"      ✓ done in {time.time()-t0:.1f}s")

    # ----- Report -------------------------------------------------------
    print()
    print("=" * 70)
    print("  SUCCESS")
    print("=" * 70)
    print(f"  Duration input     : {result['duration_seconds']:.2f} s")
    print(f"  Duration summary   : {result['summary_duration_seconds']:.2f} s")
    print(f"  Compression        : {100*result['selected_ratio']:.1f}%")
    print(f"  Shots detected     : {result['num_shots']}")
    print(f"  Selected frames    : {result['selected_frames']} / {result['n_frames']}")
    print(f"  Output files:")
    print(f"    {out_dir / 'original.mp4'}")
    print(f"    {out_dir / 'summary.mp4'}")
    print()
    print("  If both mp4s play in VLC / your browser, the web server will work too.")
    print("  Next: `uvicorn main:app --host 0.0.0.0 --port 8000`")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
