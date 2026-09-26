# Running Biomechanics Analyzer

# Dev/Creator-tubakhxn
A single-file, zero-configuration computer-vision pipeline that turns a plain video of a runner into a professional, sports-science style analysis video.

The output is a side-by-side render:

- **Left panel** — the original footage with a glowing HUD-style cyan/orange pose skeleton overlaid on the runner (comet trails on the legs, foot-strike shockwave pulses, HUD tracking brackets).
- **Right panel** — a dark analytics dashboard showing:
  - **Cadence** (steps per minute) with a live ring gauge
  - **Cadence-over-time** area graph
  - **Average knee shape at foot strike**, with an angle arc and mean ± std knee angle
  - **Ankle path visualization** (looping stride trace for each foot)
  - **Symmetry**, **Stability**, and **Speed** mini ring-gauges (speed via optional treadmill-display OCR)

No manual joint clicking, no config file, no browser UI — everything is detected and computed automatically from the input video using [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

---

## Features

- Automatic pose detection and primary-runner selection (works even with multiple people in frame)
- One Euro Filter–based landmark smoothing to kill jitter without adding lag
- Foot-strike detection from ankle/hip oscillation, with physiologically-plausible cadence bounds
- Exponentially-averaged knee shape accumulation at each strike
- Savitzky-Golay smoothed ankle path traces
- Optional OCR-based treadmill speed reading (`easyocr`)
- Automatic audio muxing back onto the output video (via `ffmpeg`, if available)
- Auto-downloads the required MediaPipe pose model on first run

---

## Requirements

- Python 3.9+
- `ffmpeg` / `ffprobe` on your `PATH` (optional, for keeping the original audio track)

Python dependencies are listed in `requirements.txt`:

```
opencv-python
mediapipe
numpy
scipy
Pillow
easyocr
```

`easyocr` is optional — it's only used for reading a treadmill's on-screen speed display. If it isn't installed, the script still runs fine and the SPEED gauge just shows `UNAVAILABLE`.

---

## Installation

```bash
git clone https://github.com/tubakhxn/running-biomechanics-analyzer.git
cd running-biomechanics-analyzer
pip install -r requirements.txt
```

---

## Usage

```bash
python running_biomechanics_analyzer.py input.mp4
```

This writes `input_analyzed.mp4` next to the input file.

### Options

| Flag | Description | Default |
|---|---|---|
| `-o`, `--output` | Custom output path | `<input>_analyzed.mp4` |
| `--model {lite,full,heavy}` | Pose model accuracy/speed tradeoff | `full` |
| `--num-poses N` | Max people detected per frame before picking the primary runner | `3` |
| `--no-ocr` | Disable treadmill speed-display OCR | OCR enabled |

Example:

```bash
python running_biomechanics_analyzer.py my_run.mp4 -o analyzed_run.mp4 --model heavy
```

---

## How it works (short version)

1. Each frame is run through MediaPipe's Pose Landmarker (video mode) to get 33 body landmarks.
2. If multiple people are detected, the one with the largest, most confident, most track-continuous bounding box is selected as the runner.
3. Landmarks are smoothed with a One Euro Filter, with a confidence-aware hold for brief occlusions.
4. Ankle-vs-hip vertical oscillation is analyzed to detect foot-strike events, which drive the cadence estimate.
5. Knee, ankle, and hip joint angles are computed each frame; the knee angle at each foot strike feeds a running average "knee shape."
6. All of this is rendered live into a dashboard panel next to a stylized HUD overlay on the original footage.

---

## Fork it / contribute

1. **Fork** this repository on GitHub (top-right "Fork" button).
2. **Clone your fork:**
   ```bash
   git clone https://github.com/<your-username>/running-biomechanics-analyzer.git
   cd running-biomechanics-analyzer
   ```
3. **Create a branch** for your change:
   ```bash
   git checkout -b feature/my-improvement
   ```
4. **Set up your environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. Make your changes, test on a sample running video, then commit:
   ```bash
   git add .
   git commit -m "Describe your change"
   git push origin feature/my-improvement
   ```
6. Open a **Pull Request** back to the original repository describing what you changed and why.

Ideas for contributions: additional metrics (stride length, ground contact time), a config file for the color palette, batch/folder processing, a lightweight web UI, or unit tests around the cadence/foot-strike detector.

---

## Relevant resources

- [MediaPipe Pose Landmarker documentation](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — the pose detection model this project is built on
- [OpenCV documentation](https://docs.opencv.org/) — used for all video I/O and drawing
- [One Euro Filter (original paper)](https://cristal.univ-lille.fr/~casiez/1euro/) — the smoothing algorithm used for landmarks and metrics
- [Savitzky–Golay filter (SciPy docs)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) — used to smooth the cadence graph and ankle path traces
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — optional OCR library used for treadmill speed reading
- [FFmpeg](https://ffmpeg.org/) — used to remux the original audio track onto the rendered output

---

## dev/creator = tubakhxn

Built by **Tuba Khan** ([@tubakhxn](https://github.com/tubakhxn)).

- GitHub: [github.com/tubakhxn](https://github.com/tubakhxn)
- LinkedIn: [linkedin.com/in/tubakhxn](https://www.linkedin.com/in/tubakhxn)
- Email: tubakhan.work@gmail.com

If you use or build on this project, a star on the repo and a credit back to `tubakhxn` is appreciated 🙌
