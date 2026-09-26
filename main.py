#!/usr/bin/env python3
# dev/creator-tubakhxn

import argparse
import math
import os
import sys
import time
import shutil
import subprocess
import urllib.request
from collections import deque

import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("ERROR: mediapipe is required. Install with:\n"
          "    pip install -r requirements.txt")
    sys.exit(1)

from scipy.signal import savgol_filter

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import easyocr
    _OCR_LIB_OK = True
except ImportError:
    _OCR_LIB_OK = False


NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32
NUM_LANDMARKS = 33


BONES = [
    (L_SHOULDER, R_SHOULDER, False),
    (L_SHOULDER, L_ELBOW, False), (L_ELBOW, L_WRIST, False),
    (R_SHOULDER, R_ELBOW, False), (R_ELBOW, R_WRIST, False),
    (L_SHOULDER, L_HIP, False), (R_SHOULDER, R_HIP, False),
    (L_HIP, R_HIP, True),
    (L_HIP, L_KNEE, True), (L_KNEE, L_ANKLE, True),
    (R_HIP, R_KNEE, True), (R_KNEE, R_ANKLE, True),
    (L_ANKLE, L_HEEL, True), (L_HEEL, L_FOOT, True), (L_ANKLE, L_FOOT, True),
    (R_ANKLE, R_HEEL, True), (R_HEEL, R_FOOT, True), (R_ANKLE, R_FOOT, True),
]

LOWER_BODY_JOINTS = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT, L_HEEL, R_HEEL]
LEG_TRAIL_JOINTS = [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT, L_HEEL, R_HEEL]


COL_BG = (18, 16, 14)
COL_PANEL_BG = (14, 12, 10)
COL_CARD_BG = (24, 22, 19)
COL_CYAN = (255, 219, 40)
COL_CYAN_DIM = (150, 120, 30)
COL_ORANGE = (10, 140, 255)
COL_ORANGE_DIM = (10, 80, 150)
COL_WHITE = (240, 240, 240)
COL_GRAY = (140, 140, 140)
COL_GREEN = (110, 220, 90)
COL_RED = (70, 70, 235)
COL_SEP = (55, 50, 45)
COL_GRID = (36, 33, 29)

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def scale_color(c, f):
    f = max(0.0, min(1.6, f))
    return tuple(int(min(255, ch * f)) for ch in c)


class OneEuroFilter:

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class LandmarkSmoother:

    def __init__(self, num_landmarks=NUM_LANDMARKS, min_cutoff=0.5, beta=0.18,
                 vis_threshold=0.35, max_hold_frames=12):
        self.fx = [OneEuroFilter(min_cutoff, beta) for _ in range(num_landmarks)]
        self.fy = [OneEuroFilter(min_cutoff, beta) for _ in range(num_landmarks)]
        self.last_valid = [None] * num_landmarks
        self.hold_count = [0] * num_landmarks
        self.vis_threshold = vis_threshold
        self.max_hold_frames = max_hold_frames

    def reset(self):
        for f in self.fx + self.fy:
            f.reset()
        self.last_valid = [None] * len(self.last_valid)
        self.hold_count = [0] * len(self.hold_count)

    def update(self, raw_xy_vis, t):
        n = len(raw_xy_vis)
        out = np.zeros((n, 3), dtype=np.float64)
        for i in range(n):
            x, y, vis = raw_xy_vis[i]
            if vis >= self.vis_threshold:
                sx = self.fx[i](x, t)
                sy = self.fy[i](y, t)
                self.last_valid[i] = (sx, sy)
                self.hold_count[i] = 0
                out[i] = (sx, sy, vis)
            else:
                if self.last_valid[i] is not None and self.hold_count[i] < self.max_hold_frames:
                    self.hold_count[i] += 1
                    decay = 1.0 - (self.hold_count[i] / self.max_hold_frames)
                    sx, sy = self.last_valid[i]


                    self.fx[i].x_prev, self.fy[i].x_prev = sx, sy
                    self.fx[i].t_prev = self.fy[i].t_prev = t
                    out[i] = (sx, sy, max(vis, 0.0) * decay + 1e-3)
                else:
                    out[i] = (x, y, 0.0)
        return out


def joint_angle(a, b, c):
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


class ScalarSmoother:

    def __init__(self, min_cutoff=0.8, beta=0.0):
        self.f = OneEuroFilter(min_cutoff, beta)
        self.value = None

    def update(self, x, t):
        if x is None:
            return self.value
        self.value = self.f(x, t)
        return self.value


class FootStrikeCadence:

    PLAUSIBLE_MIN_SPM = 100.0
    PLAUSIBLE_MAX_SPM = 220.0

    def __init__(self, fps, window_seconds=6.0, min_strike_interval=0.42,
                 cross_foot_refractory=0.18):
        self.fps = fps
        self.window_seconds = window_seconds
        self.min_strike_interval = min_strike_interval
        self.cross_foot_refractory = cross_foot_refractory

        buf_len = max(6, int(fps * 2.5))


        self.sig_raw = {"L": deque(maxlen=buf_len), "R": deque(maxlen=buf_len)}
        self.sig = {"L": deque(maxlen=buf_len), "R": deque(maxlen=buf_len)}
        self.t_buf = {"L": deque(maxlen=buf_len), "R": deque(maxlen=buf_len)}
        self.sig_filter = {"L": OneEuroFilter(min_cutoff=1.4, beta=0.0),
                            "R": OneEuroFilter(min_cutoff=1.4, beta=0.0)}
        self.last_strike_t = {"L": -999.0, "R": -999.0}
        self.last_any_strike_t = -999.0

        self.strike_times = deque(maxlen=400)
        self.strike_events = []

        self.cadence_filter = ScalarSmoother(min_cutoff=0.2, beta=0.0)
        self.total_steps = 0

        self.graph_history = deque(maxlen=600)
        self._last_graph_t = -999.0

    def update(self, t, left_sig, right_sig):
        self.strike_events = []
        for side, val in (("L", left_sig), ("R", right_sig)):
            if val is None:
                continue
            self.sig_raw[side].append(val)
            smoothed_val = self.sig_filter[side](val, t)
            self.sig[side].append(smoothed_val)
            self.t_buf[side].append(t)
            self._check_peak(side, t)


        window = min(self.window_seconds, max(t, 1e-3))
        cutoff = t - window
        while self.strike_times and self.strike_times[0] < cutoff:
            self.strike_times.popleft()
        n_strikes = len(self.strike_times)
        if t < 2.0 or n_strikes < 3:
            spm_instant = None
        else:
            span = max(self.strike_times[-1] - self.strike_times[0], window * 0.4)
            spm_instant = (n_strikes - 1) / span * 60.0 if span > 0 else None

            spm_window = n_strikes / window * 60.0
            if spm_instant is not None:
                spm_instant = 0.5 * spm_instant + 0.5 * spm_window
            if spm_instant is not None and not (self.PLAUSIBLE_MIN_SPM <= spm_instant <= self.PLAUSIBLE_MAX_SPM):


                spm_instant = None

        smoothed = self.cadence_filter.update(spm_instant, t) if spm_instant is not None else self.cadence_filter.value

        if smoothed is not None and t - self._last_graph_t >= 0.5:
            self.graph_history.append((t, smoothed))
            self._last_graph_t = t

        return smoothed

    def _check_peak(self, side, t_now):
        buf = self.sig[side]
        tb = self.t_buf[side]
        if len(buf) < 5:
            return


        window = list(buf)[-5:]
        a, b, c, d, e = window
        t_c = tb[-3]
        if not (c >= a and c >= b and c >= d and c >= e and c > min(a, e)):
            return
        recent = np.array(buf)
        rng = recent.max() - recent.min()
        if rng < 0.04:


            return
        prominence = c - recent.min()
        if prominence < 0.4 * rng:
            return
        if t_c - self.last_strike_t[side] < self.min_strike_interval:
            return
        if t_c - self.last_any_strike_t < self.cross_foot_refractory:
            return
        self.last_strike_t[side] = t_c
        self.last_any_strike_t = t_c
        self.strike_times.append(t_c)
        self.total_steps += 1
        self.strike_events.append(side)


class KneeShapeAccumulator:

    def __init__(self, alpha=0.12):
        self.alpha = alpha
        self.hip_dir = None
        self.ankle_dir = None
        self.angles = deque(maxlen=80)

    def update(self, hip, knee, ankle, angle_deg):


        if angle_deg is not None and not (80.0 <= angle_deg <= 195.0):
            return
        hip, knee, ankle = np.asarray(hip, float), np.asarray(knee, float), np.asarray(ankle, float)
        scale = np.linalg.norm(hip - knee) + np.linalg.norm(ankle - knee)
        if scale < 1e-6:
            return
        h = (hip - knee) / scale
        a = (ankle - knee) / scale
        if self.hip_dir is None:
            self.hip_dir, self.ankle_dir = h, a
        else:
            self.hip_dir = self.hip_dir * (1 - self.alpha) + h * self.alpha
            self.ankle_dir = self.ankle_dir * (1 - self.alpha) + a * self.alpha
        if angle_deg is not None:
            self.angles.append(angle_deg)

    @property
    def n_samples(self):
        return len(self.angles)

    def mean_std(self):
        if not self.angles:
            return None, None
        arr = np.array(self.angles)
        return float(arr.mean()), float(arr.std())


class FadingTrail:
    def __init__(self, maxlen):
        self.pts = deque(maxlen=maxlen)
        self.xmin = self.xmax = self.ymin = self.ymax = None

    def add(self, pt):
        x, y = pt
        self.pts.append((x, y))
        if self.xmin is None:
            self.xmin, self.xmax, self.ymin, self.ymax = x, x, y, y
        else:
            self.xmin, self.xmax = min(self.xmin, x), max(self.xmax, x)
            self.ymin, self.ymax = min(self.ymin, y), max(self.ymax, y)


class Fonts:
    def __init__(self):
        self.ok = _PIL_OK
        self.cache = {}
        self.path_bold = None
        self.path_reg = None
        if self.ok:
            candidates_bold = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
            candidates_reg = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
            self.path_bold = next((p for p in candidates_bold if os.path.exists(p)), None)
            self.path_reg = next((p for p in candidates_reg if os.path.exists(p)), None)
            self.ok = bool(self.path_bold and self.path_reg)

    def get(self, size, bold=False):
        if not self.ok:
            return None
        key = (size, bold)
        if key not in self.cache:
            path = self.path_bold if bold else self.path_reg
            self.cache[key] = ImageFont.truetype(path, size)
        return self.cache[key]


_DUMMY_IMG = Image.new("RGB", (1, 1)) if _PIL_OK else None
_DUMMY_DRAW = ImageDraw.Draw(_DUMMY_IMG) if _PIL_OK else None


def measure_text(s, font):
    if font is None or not _PIL_OK:
        return (len(s) * 8, 14)
    bbox = _DUMMY_DRAW.textbbox((0, 0), s, font=font)
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])


class TextLayer:

    def __init__(self, frame_bgr):
        self.frame = frame_bgr
        self.queue = []

    def text(self, xy, s, font, fill_bgr):
        self.queue.append((xy, s, font, fill_bgr))

    def text_size(self, s, font):
        return measure_text(s, font)

    def finish(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        for xy, s, font, fill_bgr in self.queue:
            if font is None:
                continue
            fill = (fill_bgr[2], fill_bgr[1], fill_bgr[0])
            draw.text(xy, s, font=font, fill=fill)
        rgb_out = np.array(img)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


def ensure_model(model_name):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "running_biomechanics_analyzer")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"pose_landmarker_{model_name}.task")
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    url = MODEL_URLS[model_name]
    print(f"Downloading pose model ({model_name}) — one-time setup...")
    try:
        def _hook(count, block_size, total_size):
            if total_size <= 0:
                return
            pct = min(100, int(count * block_size * 100 / total_size))
            bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
            sys.stdout.write(f"\r  [{bar}] {pct}%")
            sys.stdout.flush()
        urllib.request.urlretrieve(url, dest, _hook)
        print()
    except Exception as e:
        print(f"\nERROR: could not download the pose model automatically ({e}).")
        print(f"Please download it manually from:\n  {url}")
        print(f"and place it at:\n  {dest}")
        sys.exit(1)
    return dest


def select_primary(pose_list, prev_centroid, frame_w, frame_h):
    if not pose_list:
        return None, None
    best_idx, best_score = 0, -1e18
    for idx, lm in enumerate(pose_list):
        vis = lm[:, 2]
        valid = vis > 0.3
        if valid.sum() < 6:
            score = -1e6
        else:
            xs, ys = lm[valid, 0], lm[valid, 1]
            bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
            score = bbox_area * (vis[valid].mean())
            cx, cy = xs.mean(), ys.mean()
            if prev_centroid is not None:
                dist = math.hypot(cx - prev_centroid[0], cy - prev_centroid[1])
                continuity_bonus = max(0.0, 1.0 - dist / (0.35 * frame_w)) * bbox_area * 0.6
                score += continuity_bonus
        if score > best_score:
            best_score, best_idx = score, idx
    chosen = pose_list[best_idx]
    valid = chosen[:, 2] > 0.3
    if valid.sum() >= 6:
        centroid = (chosen[valid, 0].mean(), chosen[valid, 1].mean())
    else:
        centroid = prev_centroid
    return chosen, centroid


class RunningBiomechanicsAnalyzer:
    def __init__(self, args):
        self.args = args
        self.fonts = Fonts()
        self._vig_cache = None
        self._vig_cache_size = None


    def run(self):
        args = self.args
        if not os.path.isfile(args.input):
            print(f"ERROR: input file not found: {args.input}")
            sys.exit(1)

        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"ERROR: could not open video: {args.input}")
            sys.exit(1)

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 240:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0

        if src_w <= 0 or src_h <= 0:
            print("ERROR: invalid video (zero width/height).")
            sys.exit(1)

        out_path = args.output or self._default_output_path(args.input)

        self._print_banner(args.input, src_w, src_h, fps, duration)

        model_path = ensure_model(args.model)
        landmarker = self._build_landmarker(model_path, args.num_poses)


        video_h = max(480, min(src_h, 1080))
        video_w = int(round(video_h * (src_w / src_h)))
        dash_w = int(round(video_h * 0.58))
        total_w, total_h = video_w + dash_w, video_h
        if total_w > 1920:
            scale = 1920 / total_w
            total_w, total_h = int(total_w * scale), int(total_h * scale)
            video_w, video_h, dash_w = int(video_w * scale), int(total_h), int(dash_w * scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        tmp_video_path = out_path + ".tmp.mp4"
        writer = cv2.VideoWriter(tmp_video_path, fourcc, fps, (total_w, total_h))

        smoother = LandmarkSmoother()
        angle_smoothers = {k: ScalarSmoother(min_cutoff=1.2, beta=0.15)
                            for k in ("L_knee", "R_knee", "L_ankle", "R_ankle", "hip")}
        cadence_est = FootStrikeCadence(fps)
        knee_shapes = {"L": KneeShapeAccumulator(), "R": KneeShapeAccumulator()}
        strike_angle_hist = {"L": deque(maxlen=40), "R": deque(maxlen=40)}
        strike_time_hist = {"L": deque(maxlen=40), "R": deque(maxlen=40)}
        ankle_trail = {"L": FadingTrail(int(fps * 2.2)), "R": FadingTrail(int(fps * 2.2))}
        symmetry_filter = ScalarSmoother(min_cutoff=0.15, beta=0.0)
        stability_filter = ScalarSmoother(min_cutoff=0.15, beta=0.0)


        leg_trail_len = max(6, int(fps * 0.4))
        leg_trails = {j: deque(maxlen=leg_trail_len) for j in LEG_TRAIL_JOINTS}

        ocr_reader = None
        speed_value = None
        speed_last_check_frame = -10_000

        prev_centroid = None
        consecutive_no_pose = 0
        RUNNER_LOST_THRESHOLD = int(fps * 1.0)

        frame_idx = 0
        t0 = time.time()
        last_print = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            ts_ms = int(t * 1000)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = None
            try:
                result = landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                result = None

            pose_list = []
            if result is not None and result.pose_landmarks:
                for one in result.pose_landmarks:
                    arr = np.array([[lm.x * src_w, lm.y * src_h, lm.visibility] for lm in one])
                    pose_list.append(arr)

            chosen, prev_centroid = select_primary(pose_list, prev_centroid, src_w, src_h)

            runner_detected = chosen is not None
            if runner_detected:
                consecutive_no_pose = 0
                smoothed = smoother.update(chosen, t)
            else:
                consecutive_no_pose += 1
                smoothed = None
                if consecutive_no_pose > RUNNER_LOST_THRESHOLD:
                    smoother.reset()
                    prev_centroid = None

            metrics = self._compute_frame(
                smoothed, t, angle_smoothers, cadence_est, knee_shapes,
                strike_angle_hist, strike_time_hist, ankle_trail,
                symmetry_filter, stability_filter,
            )


            if args.ocr and _OCR_LIB_OK:
                if frame_idx - speed_last_check_frame >= int(fps * 2):
                    speed_last_check_frame = frame_idx
                    if ocr_reader is None:
                        try:
                            ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                        except Exception:
                            ocr_reader = False
                    if ocr_reader:
                        detected = self._try_ocr_speed(ocr_reader, frame)
                        if detected is not None:
                            speed_value = detected

            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            video_panel = self._render_video_panel(
                frame, smoothed, video_w, video_h, src_w, src_h,
                runner_detected or consecutive_no_pose <= RUNNER_LOST_THRESHOLD,
                leg_trails, t,
            )
            canvas[:, :video_w] = video_panel

            dash_panel = self._render_dashboard(
                dash_w, video_h, metrics, cadence_est, knee_shapes, ankle_trail,
                speed_value, t,
            )
            canvas[:, video_w:video_w + dash_w] = dash_panel
            self._draw_panel_divider(canvas, video_w)

            writer.write(canvas)
            frame_idx += 1

            now = time.time()
            if now - last_print > 0.2 or frame_idx == frame_count:
                self._print_progress(frame_idx, frame_count, t0)
                last_print = now

        cap.release()
        writer.release()
        landmarker.close()
        print()

        self._mux_audio(args.input, tmp_video_path, out_path)

        self._print_summary(out_path, metrics if 'metrics' in dir() else {}, cadence_est)
        return out_path


    @staticmethod
    def _default_output_path(input_path):
        base, _ext = os.path.splitext(input_path)
        return base + "_analyzed.mp4"

    @staticmethod
    def _print_banner(input_path, w, h, fps, duration):
        mins, secs = divmod(int(round(duration)), 60)
        print("=" * 47)
        print("RUNNING BIOMECHANICS ANALYZER")
        print("=" * 47)
        print()
        print(f"Input: {input_path}")
        print(f"Resolution: {w}x{h}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {mins:02d}:{secs:02d}")
        print()
        print("Initializing pose estimation...")
        print("Initializing biomechanics tracker...")
        print()
        print("Runner detection: READY")
        print("Pose estimation: READY")
        print("Biomechanics analysis: READY")
        print()
        print("Processing:")

    @staticmethod
    def _print_progress(frame_idx, frame_count, t0):
        elapsed = time.time() - t0
        if frame_count > 0:
            pct = min(100, int(frame_idx * 100 / frame_count))
            bar_len = 26
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "-" * (bar_len - filled)
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (frame_count - frame_idx) / rate if rate > 0 else 0
            eta_m, eta_s = divmod(int(remaining), 60)
            sys.stdout.write(f"\r[{bar}] {pct}%  ETA: {eta_m:02d}:{eta_s:02d}   ")
        else:
            sys.stdout.write(f"\rProcessed {frame_idx} frames ({elapsed:0.1f}s)   ")
        sys.stdout.flush()

    def _print_summary(self, out_path, metrics, cadence_est):
        print("=" * 47)
        print("ANALYSIS COMPLETE")
        print("=" * 47)
        cadence = cadence_est.cadence_filter.value
        print(f"Cadence: {cadence:.0f} SPM" if cadence else "Cadence: UNAVAILABLE")
        knee = metrics.get("display_knee_angle") if metrics else None
        print(f"Knee angle: {knee:.1f}°" if knee else "Knee angle: UNAVAILABLE")
        sym = metrics.get("symmetry") if metrics else None
        print(f"Symmetry: {sym:.0f}%" if sym else "Symmetry: UNAVAILABLE")
        print(f"Output: {out_path}")
        print("=" * 47)

    def _build_landmarker(self, model_path, num_poses):
        BaseOpt = BaseOptions
        VisionRunningMode = mp_vision.RunningMode
        try:
            base_options = BaseOpt(model_asset_path=model_path,
                                    delegate=BaseOpt.Delegate.GPU)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.VIDEO,
                num_poses=num_poses,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return mp_vision.PoseLandmarker.create_from_options(options)
        except Exception:
            base_options = BaseOpt(model_asset_path=model_path,
                                    delegate=BaseOpt.Delegate.CPU)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionRunningMode.VIDEO,
                num_poses=num_poses,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return mp_vision.PoseLandmarker.create_from_options(options)


    def _compute_frame(self, smoothed, t, angle_smoothers, cadence_est, knee_shapes,
                        strike_angle_hist, strike_time_hist, ankle_trail,
                        symmetry_filter, stability_filter):
        metrics = {"cadence": None, "display_knee_angle": None, "symmetry": None, "stability": None}
        if smoothed is None:
            metrics["cadence"] = cadence_est.cadence_filter.value
            return metrics

        def pt(i):
            return smoothed[i, :2]

        def vis(i):
            return smoothed[i, 2]


        l_knee = joint_angle(pt(L_HIP), pt(L_KNEE), pt(L_ANKLE)) if min(vis(L_HIP), vis(L_KNEE), vis(L_ANKLE)) > 0.15 else None
        r_knee = joint_angle(pt(R_HIP), pt(R_KNEE), pt(R_ANKLE)) if min(vis(R_HIP), vis(R_KNEE), vis(R_ANKLE)) > 0.15 else None
        l_ankle = joint_angle(pt(L_KNEE), pt(L_ANKLE), pt(L_FOOT)) if min(vis(L_KNEE), vis(L_ANKLE), vis(L_FOOT)) > 0.15 else None
        r_ankle = joint_angle(pt(R_KNEE), pt(R_ANKLE), pt(R_FOOT)) if min(vis(R_KNEE), vis(R_ANKLE), vis(R_FOOT)) > 0.15 else None
        hip_ang_src = None
        if min(vis(L_SHOULDER), vis(L_HIP), vis(L_KNEE)) > 0.15:
            hip_ang_src = joint_angle(pt(L_SHOULDER), pt(L_HIP), pt(L_KNEE))
        elif min(vis(R_SHOULDER), vis(R_HIP), vis(R_KNEE)) > 0.15:
            hip_ang_src = joint_angle(pt(R_SHOULDER), pt(R_HIP), pt(R_KNEE))

        l_knee_s = angle_smoothers["L_knee"].update(l_knee, t)
        r_knee_s = angle_smoothers["R_knee"].update(r_knee, t)
        angle_smoothers["L_ankle"].update(l_ankle, t)
        angle_smoothers["R_ankle"].update(r_ankle, t)
        angle_smoothers["hip"].update(hip_ang_src, t)

        candidates = [v for v in (l_knee_s, r_knee_s) if v is not None]
        metrics["display_knee_angle"] = float(np.mean(candidates)) if candidates else None


        if vis(L_ANKLE) > 0.2:
            ankle_trail["L"].add(tuple(pt(L_ANKLE)))
        if vis(R_ANKLE) > 0.2:
            ankle_trail["R"].add(tuple(pt(R_ANKLE)))


        body_h = None
        if vis(NOSE) > 0.2 and (vis(L_ANKLE) > 0.2 or vis(R_ANKLE) > 0.2):
            ay = pt(L_ANKLE)[1] if vis(L_ANKLE) > vis(R_ANKLE) else pt(R_ANKLE)[1]
            body_h = max(ay - pt(NOSE)[1], 1.0)
        left_sig = (pt(L_ANKLE)[1] - pt(L_HIP)[1]) / body_h if body_h and vis(L_ANKLE) > 0.2 and vis(L_HIP) > 0.2 else None
        right_sig = (pt(R_ANKLE)[1] - pt(R_HIP)[1]) / body_h if body_h and vis(R_ANKLE) > 0.2 and vis(R_HIP) > 0.2 else None

        cadence = cadence_est.update(t, left_sig, right_sig)
        metrics["cadence"] = cadence

        for side in cadence_est.strike_events:
            hip_i, knee_i, ankle_i = (L_HIP, L_KNEE, L_ANKLE) if side == "L" else (R_HIP, R_KNEE, R_ANKLE)
            ang = l_knee_s if side == "L" else r_knee_s
            if min(vis(hip_i), vis(knee_i), vis(ankle_i)) > 0.15:
                knee_shapes[side].update(pt(hip_i), pt(knee_i), pt(ankle_i), ang)
                if ang is not None:
                    strike_angle_hist[side].append(ang)
                strike_time_hist[side].append(t)


        if len(strike_angle_hist["L"]) >= 4 and len(strike_angle_hist["R"]) >= 4:
            mL = float(np.mean(list(strike_angle_hist["L"])[-8:]))
            mR = float(np.mean(list(strike_angle_hist["R"])[-8:]))
            denom = max(mL, mR, 1e-6)
            sym_raw = 100.0 * (1.0 - abs(mL - mR) / denom)
            sym_raw = float(np.clip(sym_raw, 0, 100))
            metrics["symmetry"] = symmetry_filter.update(sym_raw, t)
        else:
            metrics["symmetry"] = symmetry_filter.value


        if len(cadence_est.graph_history) >= 4:
            recent = [v for _, v in list(cadence_est.graph_history)[-10:]]
            cv = float(np.std(recent) / max(np.mean(recent), 1e-6))
            stab_raw = float(np.clip(100.0 * (1.0 - cv * 3.0), 0, 100))
            metrics["stability"] = stability_filter.update(stab_raw, t)
        else:
            metrics["stability"] = stability_filter.value

        return metrics


    @staticmethod
    def _try_ocr_speed(reader, frame):
        h, w = frame.shape[:2]
        roi = frame[0:int(h * 0.4), 0:int(w * 0.6)]
        try:
            results = reader.readtext(roi)
        except Exception:
            return None
        best = None
        for _, text, conf in results:
            clean = text.lower().replace(",", ".")
            has_unit = ("km" in clean) or ("mph" in clean) or ("kph" in clean) or ("mi" in clean)
            digits = "".join(ch for ch in clean if (ch.isdigit() or ch == "."))
            if digits and conf > 0.35:
                try:
                    val = float(digits)
                except ValueError:
                    continue
                if 1.0 <= val <= 30.0:
                    unit = "km/h" if "km" in clean or "kph" in clean or not has_unit else "mph"
                    if has_unit or best is None:
                        best = f"{val:.1f} {unit}"
        return best


    def _render_video_panel(self, frame, smoothed, video_w, video_h, src_w, src_h,
                             show_status_ok, leg_trails, t):
        panel = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_AREA)


        panel = cv2.convertScaleAbs(panel, alpha=0.78, beta=-6)
        sx, sy = video_w / src_w, video_h / src_h
        glow = np.zeros_like(panel)
        bbox = None

        if smoothed is not None:
            def p(i):
                x, y, v = smoothed[i]
                return (int(x * sx), int(y * sy)), v

            valid_pts = [p(i)[0] for i in range(NUM_LANDMARKS) if p(i)[1] > 0.15]
            if valid_pts:
                xs = [pt[0] for pt in valid_pts]
                ys = [pt[1] for pt in valid_pts]
                bbox = (min(xs) - 22, min(ys) - 22, max(xs) + 22, max(ys) + 22)


            for j in LEG_TRAIL_JOINTS:
                (px, py), v = p(j)
                if v > 0.2:
                    leg_trails[j].append((px, py))
                elif leg_trails[j]:
                    leg_trails[j].append(leg_trails[j][-1])

            for j in LEG_TRAIL_JOINTS:
                pts = list(leg_trails[j])
                n = len(pts)
                if n < 2:
                    continue
                for i in range(1, n):
                    alpha = (i / n) ** 1.5
                    thick = max(1, int(round(2 + 6 * alpha)))
                    col = scale_color(COL_ORANGE, 0.3 + 0.9 * alpha)
                    cv2.line(glow, pts[i - 1], pts[i], col, thick, cv2.LINE_AA)


            for a, b, lower in BONES:
                (pa, va), (pb, vb) = p(a), p(b)
                vmin_ab = min(va, vb)
                if vmin_ab <= 0.08:
                    continue
                fade = float(np.clip((vmin_ab - 0.08) / 0.35, 0.0, 1.0))
                base_color = COL_ORANGE if lower else COL_CYAN
                color = scale_color(base_color, 0.45 + 0.65 * fade)
                thick = 4 if lower else 2
                cv2.line(panel, pa, pb, color, thick, cv2.LINE_AA)
                cv2.line(glow, pa, pb, color, thick + 4, cv2.LINE_AA)

            for i in range(NUM_LANDMARKS):
                (px, py), v = p(i)
                if v <= 0.08:
                    continue
                fade = float(np.clip((v - 0.08) / 0.35, 0.0, 1.0))
                lower = i in LOWER_BODY_JOINTS
                r = 6 if lower else 4
                jcol = scale_color(COL_ORANGE if lower else COL_CYAN, 0.45 + 0.65 * fade)
                cv2.circle(panel, (px, py), r, COL_WHITE, -1, cv2.LINE_AA)
                cv2.circle(panel, (px, py), r, jcol, 2, cv2.LINE_AA)
                cv2.circle(glow, (px, py), r + 5, jcol, -1, cv2.LINE_AA)


            for side in ("L", "R"):
                ai = L_ANKLE if side == "L" else R_ANKLE
                (px, py), v = p(ai)
                if v > 0.15:
                    pulse = (math.sin(t * 10.0 + (0 if side == "L" else math.pi)) + 1) * 0.5
                    ring_r = int(10 + 14 * pulse)
                    ring_col = scale_color(COL_WHITE, 0.5 * (1.0 - pulse))
                    cv2.circle(glow, (px, py), ring_r, ring_col, 2, cv2.LINE_AA)
        elif not show_status_ok:
            self._draw_status_banner(panel, "RUNNER NOT DETECTED")

        if glow.any():
            blur_tight = cv2.GaussianBlur(glow, (0, 0), sigmaX=4, sigmaY=4)
            blur_wide = cv2.GaussianBlur(glow, (0, 0), sigmaX=15, sigmaY=15)
            bloom = cv2.addWeighted(blur_tight, 0.85, blur_wide, 0.55, 0)
            panel = cv2.add(panel, cv2.convertScaleAbs(bloom, alpha=0.75))

        if bbox is not None:
            self._draw_hud_brackets(panel, bbox)

        self._apply_vignette(panel)
        return panel

    @staticmethod
    def _draw_status_banner(panel, text):
        h, w = panel.shape[:2]
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
        cv2.putText(panel, text, (max(10, w // 2 - 170), h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_RED, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_hud_brackets(panel, bbox, color=COL_CYAN_DIM, size=18, thick=2):
        h, w = panel.shape[:2]
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        corners = [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]
        for cx, cy, dx, dy in corners:
            cv2.line(panel, (cx, cy), (cx + dx * size, cy), color, thick, cv2.LINE_AA)
            cv2.line(panel, (cx, cy), (cx, cy + dy * size), color, thick, cv2.LINE_AA)

    def _apply_vignette(self, panel):
        h, w = panel.shape[:2]
        if self._vig_cache is None or self._vig_cache_size != (w, h):
            yy, xx = np.mgrid[0:h, 0:w]
            cx, cy = w / 2.0, h / 2.0
            d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
            mask = np.clip(1.0 - 0.35 * np.clip(d - 0.55, 0, 1), 0.55, 1.0).astype(np.float32)
            self._vig_cache = mask
            self._vig_cache_size = (w, h)
        mask = self._vig_cache
        panel[:] = (panel.astype(np.float32) * mask[..., None]).astype(np.uint8)

    @staticmethod
    def _draw_panel_divider(canvas, video_w):
        h = canvas.shape[0]
        seam_w = 4
        x0 = max(0, video_w - seam_w // 2)
        seam_layer = np.zeros_like(canvas)
        for i in range(seam_w):
            x = x0 + i
            if x < 0 or x >= canvas.shape[1]:
                continue
            frac = i / max(seam_w - 1, 1)
            col = lerp_color(COL_CYAN, COL_ORANGE, frac)
            seam_layer[:, x] = col
        blurred = cv2.GaussianBlur(seam_layer, (0, 0), sigmaX=3, sigmaY=0)
        canvas[:] = cv2.add(canvas, cv2.convertScaleAbs(blurred, alpha=0.5))
        for i in range(seam_w):
            x = x0 + i
            if x < 0 or x >= canvas.shape[1]:
                continue
            frac = i / max(seam_w - 1, 1)
            col = lerp_color(COL_CYAN, COL_ORANGE, frac)
            canvas[:, x] = [int(v * 0.6) for v in col]

    @staticmethod
    def _rounded_rect(img, pt1, pt2, color, radius, thickness=-1, border_color=None):
        x0, y0 = pt1
        x1, y1 = pt2
        radius = max(0, min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
        if thickness < 0:
            cv2.rectangle(img, (x0 + radius, y0), (x1 - radius, y1), color, -1)
            cv2.rectangle(img, (x0, y0 + radius), (x1, y1 - radius), color, -1)
            for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                           (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
                cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x0 + radius, y0), (x1 - radius, y0), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x0 + radius, y1), (x1 - radius, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x0, y0 + radius), (x0, y1 - radius), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y0 + radius), (x1, y1 - radius), color, thickness, cv2.LINE_AA)
            for cx, cy, a1, a2 in ((x0 + radius, y0 + radius, 180, 270), (x1 - radius, y0 + radius, 270, 360),
                                   (x1 - radius, y1 - radius, 0, 90), (x0 + radius, y1 - radius, 90, 180)):
                cv2.ellipse(img, (cx, cy), (radius, radius), 0, a1, a2, color, thickness, cv2.LINE_AA)
        if border_color is not None:
            RunningBiomechanicsAnalyzer._rounded_rect(img, pt1, pt2, border_color, radius, thickness=1)

    @staticmethod
    def _draw_ring_gauge(panel, center, radius, frac, color, thickness=6, bg=(46, 42, 38)):
        start_angle, end_angle = 125, 415
        cv2.ellipse(panel, center, (radius, radius), 0, start_angle, end_angle, bg, thickness, cv2.LINE_AA)
        if frac is None:
            return
        frac = float(np.clip(frac, 0.0, 1.0))
        sweep_end = start_angle + frac * (end_angle - start_angle)
        cv2.ellipse(panel, center, (radius, radius), 0, start_angle, sweep_end, color, thickness, cv2.LINE_AA)

        tip_rad = math.radians(sweep_end)
        tip = (int(center[0] + radius * math.cos(tip_rad)), int(center[1] + radius * math.sin(tip_rad)))
        cv2.circle(panel, tip, max(2, thickness // 2 + 1), color, -1, cv2.LINE_AA)

    @staticmethod
    def _gradient_background(w, h):
        top = np.array((24, 21, 18), dtype=np.float32)
        bottom = np.array((9, 8, 7), dtype=np.float32)
        ramp = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
        grad = top.reshape(1, 1, 3) * (1 - ramp) + bottom.reshape(1, 1, 3) * ramp
        panel = np.repeat(grad, w, axis=1).astype(np.uint8)
        return panel

    def _render_dashboard(self, w, h, metrics, cadence_est, knee_shapes, ankle_trail,
                           speed_value, t):
        panel = self._gradient_background(w, h)
        pad = int(w * 0.06)
        cursor_y = int(h * 0.032)

        tl = TextLayer(panel) if self.fonts.ok else None
        f_label = self.fonts.get(max(11, int(h * 0.015)), bold=False)
        f_small = self.fonts.get(max(10, int(h * 0.012)), bold=False)
        f_big = self.fonts.get(int(h * 0.08), bold=True)
        f_unit = self.fonts.get(int(h * 0.02), bold=True)
        f_section = self.fonts.get(int(h * 0.0155), bold=True)
        f_title = self.fonts.get(int(h * 0.021), bold=True)
        f_tiny = self.fonts.get(max(9, int(h * 0.0105)), bold=False)

        def label(txt, y, color=COL_GRAY, font=f_label, accent=None):
            x = pad
            if accent is not None:
                tick_h = max(9, int(h * 0.013))
                cv2.rectangle(panel, (pad, y + 2), (pad + 3, y + 2 + tick_h), accent, -1, cv2.LINE_AA)
                x = pad + 11
            if tl:
                tl.text((x, y), txt, font, color)
            else:
                cv2.putText(panel, txt, (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        def sep(y):
            cv2.line(panel, (pad, y), (w - pad, y), COL_SEP, 1, cv2.LINE_AA)


        if tl:
            tl.text((pad, cursor_y), "RUNFORM", f_title, COL_WHITE)
            tw, _ = tl.text_size("RUNFORM", f_title)
            tl.text((pad + tw + 8, cursor_y + 2), "ANALYTICS", f_title, COL_GRAY)
        pulse = 0.5 + 0.5 * math.sin(t * 4.0)
        dot_col = scale_color(COL_GREEN, 0.7 + 0.6 * pulse)
        dot_center = (w - pad - 8, cursor_y + int(h * 0.012))
        cv2.circle(panel, dot_center, int(3 + 2 * pulse), dot_col, -1, cv2.LINE_AA)
        if tl:
            tl.text((dot_center[0] - 46, cursor_y), "LIVE", f_small, COL_GRAY)
        cursor_y += int(h * 0.048)
        sep(cursor_y)
        cursor_y += int(h * 0.03)


        cadence = metrics.get("cadence") if metrics else None
        label("CADENCE", cursor_y, COL_GRAY, f_label, accent=COL_CYAN)
        cursor_y += int(h * 0.022)
        big_y = cursor_y
        cad_text = f"{cadence:.0f}" if cadence else "--"
        if tl:
            tl.text((pad, big_y), cad_text, f_big, COL_WHITE)
            tw, th = tl.text_size(cad_text, f_big)
            tl.text((pad + tw + 8, big_y + th - int(h * 0.026)), "SPM", f_unit, COL_CYAN)
        else:
            cv2.putText(panel, cad_text, (pad, big_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COL_WHITE, 3, cv2.LINE_AA)
        gauge_r = max(18, int(h * 0.05))
        gauge_center = (w - pad - gauge_r - 4, big_y + int(h * 0.045))
        gauge_frac = None if cadence is None else (cadence - 80.0) / (210.0 - 80.0)
        self._draw_ring_gauge(panel, gauge_center, gauge_r, gauge_frac, COL_CYAN,
                               thickness=max(4, int(h * 0.009)))
        cursor_y += int(h * 0.095)
        steps_txt = f"{cadence_est.total_steps} steps"
        label(steps_txt, cursor_y, COL_GRAY, f_small)
        cursor_y += int(h * 0.035)
        sep(cursor_y)
        cursor_y += int(h * 0.03)


        label("AVG CADENCE OVER TIME", cursor_y, COL_GRAY, f_section, accent=COL_CYAN)
        cursor_y += int(h * 0.018)
        graph_h = int(h * 0.115)
        graph_rect = (pad, cursor_y, w - pad, cursor_y + graph_h)
        self._draw_line_graph(panel, graph_rect, cadence_est.graph_history, COL_CYAN)
        cursor_y += graph_h + int(h * 0.035)
        sep(cursor_y)
        cursor_y += int(h * 0.03)


        label("AVG KNEE SHAPE AT FOOT STRIKE", cursor_y, COL_GRAY, f_section, accent=COL_ORANGE)
        cursor_y += int(h * 0.018)
        knee_h = int(h * 0.165)
        knee_rect = (pad, cursor_y, w - pad, cursor_y + knee_h)
        side, mean_ang, std_ang = self._dominant_knee_side(knee_shapes)
        self._draw_knee_shape(panel, knee_rect, knee_shapes.get(side) if side else None)
        cursor_y += knee_h + int(h * 0.012)
        if side and mean_ang is not None:
            txt_side = "LEFT" if side == "L" else "RIGHT"
            color = COL_CYAN if side == "L" else COL_ORANGE
            full_txt = f"{txt_side}  {mean_ang:.1f}\u00b0 \u00b1 {std_ang:.1f}\u00b0"
            if tl:
                tl.text((pad, cursor_y), txt_side + "  ", f_small, color)
                tw, _ = tl.text_size(txt_side + "  ", f_small)
                tl.text((pad + tw, cursor_y), f"{mean_ang:.1f}\u00b0 \u00b1 {std_ang:.1f}\u00b0", f_small, COL_WHITE)
            else:
                cv2.putText(panel, full_txt, (pad, cursor_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_WHITE, 1, cv2.LINE_AA)
        else:
            label("collecting strikes...", cursor_y, COL_GRAY, f_small)
        cursor_y += int(h * 0.035)
        sep(cursor_y)
        cursor_y += int(h * 0.03)


        label("ANKLE PATH VISUALIZATION", cursor_y, COL_GRAY, f_section, accent=COL_CYAN)
        cursor_y += int(h * 0.018)
        path_h = int(h * 0.16)
        path_rect = (pad, cursor_y, w - pad, cursor_y + path_h)
        self._draw_ankle_paths(panel, path_rect, ankle_trail)
        cursor_y += path_h + int(h * 0.02)
        if tl:
            tl.text((pad, cursor_y), "\u25cf", f_small, COL_CYAN)
            tl.text((pad + 16, cursor_y), "LEFT", f_small, COL_GRAY)
            tl.text((pad + 80, cursor_y), "\u25cf", f_small, COL_ORANGE)
            tl.text((pad + 96, cursor_y), "RIGHT", f_small, COL_GRAY)
        cursor_y += int(h * 0.045)
        sep(cursor_y)
        cursor_y += int(h * 0.03)


        col_w = (w - 2 * pad) // 3
        metric_defs = [
            ("SYMMETRY", metrics.get("symmetry") if metrics else None, "%", COL_CYAN),
            ("STABILITY", metrics.get("stability") if metrics else None, "%", COL_ORANGE),
            ("SPEED", speed_value, "", COL_GRAY),
        ]
        mini_r = max(14, int(h * 0.026))
        for i, (name, val, unit, accent) in enumerate(metric_defs):
            x = pad + i * col_w
            if tl:
                tl.text((x, cursor_y), name, f_small, COL_GRAY)
            center = (x + mini_r + 2, cursor_y + int(h * 0.05))
            if val is None:
                self._draw_ring_gauge(panel, center, mini_r, None, accent, thickness=max(3, int(h * 0.007)))
                txt, col = "--", COL_GRAY
            elif isinstance(val, str):
                self._draw_ring_gauge(panel, center, mini_r, 1.0, accent, thickness=max(3, int(h * 0.007)))
                txt, col = val, COL_WHITE
            else:
                self._draw_ring_gauge(panel, center, mini_r, val / 100.0, accent, thickness=max(3, int(h * 0.007)))
                txt = f"{val:.0f}{unit}"
                col = COL_GREEN if val >= 80 else COL_WHITE
            if tl:
                tl.text((center[0] + mini_r + 8, cursor_y + int(h * 0.033)), txt, f_label, col)
            if name == "STABILITY" and not isinstance(val, str):
                if tl:
                    tl.text((center[0] + mini_r + 8, cursor_y + int(h * 0.058)), "EXPERIMENTAL", f_tiny, COL_GRAY)

        if tl:
            panel = tl.finish()
        return panel

    @staticmethod
    def _dominant_knee_side(knee_shapes):
        l_n = knee_shapes["L"].n_samples
        r_n = knee_shapes["R"].n_samples
        if l_n == 0 and r_n == 0:
            return None, None, None
        side = "R" if r_n >= l_n else "L"
        mean_ang, std_ang = knee_shapes[side].mean_std()
        return side, mean_ang, std_ang

    @staticmethod
    def _draw_line_graph(panel, rect, history, accent=COL_CYAN):
        x0, y0, x1, y1 = rect
        RunningBiomechanicsAnalyzer._rounded_rect(panel, (x0, y0), (x1, y1), COL_CARD_BG, 10)
        for gy in np.linspace(y0 + 6, y1 - 6, 4):
            cv2.line(panel, (x0 + 6, int(gy)), (x1 - 6, int(gy)), COL_GRID, 1, cv2.LINE_AA)
        if len(history) < 2:
            return
        times = np.array([p[0] for p in history])
        vals = np.array([p[1] for p in history])


        if len(vals) >= 5:
            win = min(9, len(vals) if len(vals) % 2 == 1 else len(vals) - 1)
            win = max(win, 5)
            try:
                vals = savgol_filter(vals, win, 2)
            except Exception:
                pass
        tmin, tmax = times.min(), times.max()
        vmin, vmax = vals.min(), vals.max()
        if tmax - tmin < 1e-3:
            return
        vpad = max((vmax - vmin) * 0.3, 3.0)
        vmin, vmax = vmin - vpad, vmax + vpad
        pts = []
        for tt, vv in zip(times, vals):
            px = x0 + 6 + int((tt - tmin) / (tmax - tmin) * (x1 - x0 - 12))
            py = y1 - 6 - int((vv - vmin) / max(vmax - vmin, 1e-6) * (y1 - y0 - 12))
            pts.append((px, py))

        fill_pts = pts + [(pts[-1][0], y1 - 4), (pts[0][0], y1 - 4)]
        overlay = panel.copy()
        cv2.fillPoly(overlay, [np.array(fill_pts, dtype=np.int32)], scale_color(accent, 0.35))
        cv2.addWeighted(overlay, 0.35, panel, 0.65, 0, panel)
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], accent, 2, cv2.LINE_AA)
        cv2.circle(panel, pts[-1], 5, COL_WHITE, -1, cv2.LINE_AA)
        cv2.circle(panel, pts[-1], 5, accent, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_knee_shape(panel, rect, acc):
        x0, y0, x1, y1 = rect
        RunningBiomechanicsAnalyzer._rounded_rect(panel, (x0, y0), (x1, y1), COL_CARD_BG, 10)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        if acc is None or acc.hip_dir is None:
            cv2.circle(panel, (cx, cy), 2, COL_SEP, -1, cv2.LINE_AA)
            return
        scale = min(x1 - x0, y1 - y0) * 0.38
        knee = np.array([cx, cy], dtype=float)
        hip_pt = knee + acc.hip_dir * scale
        ankle_pt = knee + acc.ankle_dir * scale
        p_hip = (int(hip_pt[0]), int(hip_pt[1]))
        p_knee = (int(knee[0]), int(knee[1]))
        p_ankle = (int(ankle_pt[0]), int(ankle_pt[1]))


        ang1 = math.degrees(math.atan2(acc.hip_dir[1], acc.hip_dir[0]))
        ang2 = math.degrees(math.atan2(acc.ankle_dir[1], acc.ankle_dir[0]))
        arc_r = max(10, int(scale * 0.32))
        cv2.ellipse(panel, p_knee, (arc_r, arc_r), 0, ang1, ang2, COL_WHITE, 1, cv2.LINE_AA)


        glow = np.zeros_like(panel)
        cv2.line(glow, p_hip, p_knee, COL_ORANGE, 8, cv2.LINE_AA)
        cv2.line(glow, p_knee, p_ankle, COL_ORANGE, 8, cv2.LINE_AA)
        blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=6, sigmaY=6)
        panel[:] = cv2.add(panel, cv2.convertScaleAbs(blurred, alpha=0.5))

        cv2.line(panel, p_hip, p_knee, COL_ORANGE, 4, cv2.LINE_AA)
        cv2.line(panel, p_knee, p_ankle, COL_ORANGE, 4, cv2.LINE_AA)
        for pnt in (p_hip, p_knee, p_ankle):
            cv2.circle(panel, pnt, 6, COL_WHITE, -1, cv2.LINE_AA)
            cv2.circle(panel, pnt, 6, COL_ORANGE, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_ankle_paths(panel, rect, ankle_trail):
        x0, y0, x1, y1 = rect
        RunningBiomechanicsAnalyzer._rounded_rect(panel, (x0, y0), (x1, y1), COL_CARD_BG, 10)
        for gx in np.linspace(x0 + 6, x1 - 6, 4):
            cv2.line(panel, (int(gx), y0 + 6), (int(gx), y1 - 6), COL_GRID, 1, cv2.LINE_AA)

        glow = np.zeros_like(panel)
        drew_any = False
        cached_lines = []

        for side, color in (("L", COL_CYAN), ("R", COL_ORANGE)):
            trail = ankle_trail[side]
            n = len(trail.pts)
            if n < 2:
                continue
            raw_x = np.array([p[0] for p in trail.pts], dtype=float)
            raw_y = np.array([p[1] for p in trail.pts], dtype=float)


            if n >= 7:
                win = min(21, n if n % 2 == 1 else n - 1)
                win = max(win, 5)
                poly = 3 if win > 3 else 2
                sm_x = savgol_filter(raw_x, win, poly)
                sm_y = savgol_filter(raw_y, win, poly)
            else:
                sm_x, sm_y = raw_x, raw_y


            xmin, xmax = np.percentile(sm_x, [2, 98])
            ymin, ymax = np.percentile(sm_y, [2, 98])
            rx = max(xmax - xmin, 1.0)
            ry = max(ymax - ymin, 1.0)
            margin = 10
            avail_w, avail_h = (x1 - x0 - 2 * margin), (y1 - y0 - 2 * margin)
            scale = min(avail_w / rx, avail_h / ry)
            ox = x0 + margin + (avail_w - rx * scale) / 2
            oy = y0 + margin + (avail_h - ry * scale) / 2

            def proj(px, py):
                return (int(ox + (px - xmin) * scale), int(oy + (py - ymin) * scale))

            pts = [proj(px, py) for px, py in zip(sm_x, sm_y)]
            cached_lines.append((color, pts))
            drew_any = True
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = scale_color(color, 0.3 + 0.9 * alpha)
                cv2.line(glow, pts[i - 1], pts[i], col, 4, cv2.LINE_AA)

        if drew_any:
            blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=4, sigmaY=4)
            panel[:] = cv2.add(panel, cv2.convertScaleAbs(blurred, alpha=0.55))

        for color, pts in cached_lines:
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = tuple(int(c * (0.25 + 0.75 * alpha)) for c in color)
                cv2.line(panel, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)
            cv2.circle(panel, pts[-1], 5, color, -1, cv2.LINE_AA)
            cv2.circle(panel, pts[-1], 5, COL_WHITE, 1, cv2.LINE_AA)


    @staticmethod
    def _mux_audio(input_path, tmp_video_path, out_path):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        has_audio = False
        if ffprobe:
            try:
                probe = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "csv=p=0", input_path],
                    capture_output=True, text=True, timeout=30,
                )
                has_audio = bool(probe.stdout.strip())
            except Exception:
                has_audio = False

        if ffmpeg and has_audio:
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-i", tmp_video_path, "-i", input_path,
                     "-map", "0:v:0", "-map", "1:a:0?",
                     "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
                    check=True, capture_output=True, timeout=600,
                )
                os.remove(tmp_video_path)
                return
            except Exception:
                pass

        if os.path.exists(out_path):
            os.remove(out_path)
        shutil.move(tmp_video_path, out_path)
        if not has_audio:
            print("(source video has no audio track — output is video-only)")
        elif not ffmpeg:
            print("(ffmpeg not found — output is video-only; install ffmpeg to keep audio)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Turn a runner video into a professional biomechanics analysis video.")
    p.add_argument("input", help="Path to the input video (e.g. input.mp4)")
    p.add_argument("-o", "--output", default=None,
                   help="Output path (default: <input>_analyzed.mp4)")
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="Pose model accuracy/speed tradeoff (default: full)")
    p.add_argument("--num-poses", type=int, default=3,
                   help="Max people to detect per frame before picking the primary runner (default: 3)")
    p.add_argument("--no-ocr", dest="ocr", action="store_false",
                   help="Disable treadmill speed-display OCR")
    p.set_defaults(ocr=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.ocr and not _OCR_LIB_OK:
        print("(easyocr not installed — treadmill SPEED will show UNAVAILABLE. "
              "Install with 'pip install easyocr' to enable it.)")
    analyzer = RunningBiomechanicsAnalyzer(args)
    analyzer.run()


if __name__ == "__main__":
    main()