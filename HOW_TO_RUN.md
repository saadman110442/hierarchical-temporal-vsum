# How to run the Video Summarizer

Two scripts in this folder handle everything:

- `setup.ps1` — one-shot environment setup (run once, or after changes)
- `run.ps1` — starts the web server

## First time (about 15-30 minutes, mostly downloads)

Open this folder in VS Code, open the integrated terminal (Ctrl+`), and run:

```powershell
.\setup.ps1
```

The script will:

1. Clean up any stray partial-download (`.crdownload`) files
2. Verify Python 3.11 is installed (installs it via winget if not)
3. Verify ffmpeg is installed (installs it via winget if not)
4. Remove the old Python 3.8 `venv` and build a fresh one with Python 3.11
5. Upgrade pip, install CPU-only PyTorch 2.4.1, then `requirements.txt`
6. Check that `backend\models\best.pt` is in place

If PowerShell blocks the script with an "execution policy" error, run this once
in an **Admin** PowerShell, then try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Starting the server

```powershell
.\run.ps1
```

Then open http://localhost:8000 in your browser. Drag a short `.mp4` onto the
upload zone and watch it summarize.

Options:

```powershell
.\run.ps1 -Port 8001                              # use a different port
.\run.ps1 -Checkpoint ".\models\my_model.pt"      # use a different checkpoint
```

Press **Ctrl+C** in the terminal to stop the server.

## Troubleshooting

- **"python is not recognized"** — Python 3.11 is not on PATH. Re-run the Python
  installer and tick "Add python.exe to PATH" on the first screen, then open a
  fresh terminal and re-run `.\setup.ps1`.
- **"ffmpeg is not recognized"** — close all terminals, open a fresh one
  (PATH changes only apply to new terminals), and re-run `.\setup.ps1`.
- **"size mismatch" on model load** — your checkpoint uses different
  hyperparameters than `BEST_CFG` in `backend\inference.py`. Edit that dict to
  match the config used when the checkpoint was saved.
- **"Address already in use"** — another process is on port 8000. Either close
  it, or run `.\run.ps1 -Port 8001`.

## Optional: sanity-check before starting the server

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python test_inference.py ..\my_test_video.mp4
```

Replace `my_test_video.mp4` with any short `.mp4` you have. On success it
writes `test_output\original.mp4` and `test_output\summary.mp4`.
