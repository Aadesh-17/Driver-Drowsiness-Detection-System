import streamlit as st
import sys
import os
import traceback

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detector",
    page_icon="🚗",
    layout="wide",
)

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode,
    )
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.mediapipe_c_bindings import load_raw_library
    import numpy as np
    import av

    # Test loading the MediaPipe C library immediately to catch any missing system .so
    try:
        load_raw_library()
    except OSError as dl_err:
        import subprocess, glob
        so_files = glob.glob(os.path.join(os.path.dirname(mp.__file__), "**/*.so*"), recursive=True)
        ldd_info = ""
        for so in so_files:
            res = subprocess.run(["ldd", so], capture_output=True, text=True)
            if "not found" in res.stdout:
                ldd_info += f"\n--- ldd {os.path.basename(so)} ---\n{res.stdout}"
        st.error(f"⚠️ MediaPipe dlopen error: {dl_err}")
        if ldd_info:
            st.markdown("### Missing Shared Libraries (from `ldd`):")
            st.code(ldd_info)
        else:
            st.code(traceback.format_exc())
        st.stop()

except Exception as e:
    st.error(f"⚠️ Import failed: `{e}`")
    st.code(traceback.format_exc())
    st.markdown(f"**Python:** `{sys.executable}` — `{sys.version}`")
    st.stop()

import threading
import os
import urllib.request

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .status-box {
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .alert-green  { background: #1a3a1a; color: #4caf50; border: 2px solid #4caf50; }
    .alert-red    { background: #3a1a1a; color: #f44336; border: 2px solid #f44336; }
    .alert-yellow { background: #3a3a1a; color: #ffeb3b; border: 2px solid #ffeb3b; }
    .alert-gray   { background: #2a2a2a; color: #888;    border: 2px solid #444;    }
</style>
""", unsafe_allow_html=True)

# ── Download face landmarker model ────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

@st.cache_resource(show_spinner=False)
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading face landmarker model (~5 MB) — first run only…"):
            req = urllib.request.Request(
                MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp, open(MODEL_PATH, "wb") as f:
                f.write(resp.read())
    return MODEL_PATH

model_path = download_model()

# ── MediaPipe Face Mesh landmark indices ──────────────────────────────────────
# 6-point eye landmarks: [lateral, upper1, upper2, medial, lower2, lower1]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR
UPPER_LIP_IDX    = [82, 13, 312]
LOWER_LIP_IDX    = [87, 14, 317]
MOUTH_CORNER_IDX = [78, 308]

# Contour indices for drawing
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_CONTOUR = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185, 61,
]


# ── Ratio helpers ─────────────────────────────────────────────────────────────
def eye_aspect_ratio(landmarks, indices, w, h):
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, upper_idx, lower_idx, corner_idx, w, h):
    upper   = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in upper_idx])
    lower   = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in lower_idx])
    corners = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in corner_idx])
    A = np.linalg.norm(upper[0] - lower[0])
    B = np.linalg.norm(upper[1] - lower[1])
    C = np.linalg.norm(upper[2] - lower[2])
    D = np.linalg.norm(corners[0] - corners[1])
    if D == 0:
        return 0.0
    return (A + B + C) / (3.0 * D)


def get_contour_pts(landmarks, indices, w, h):
    return np.array(
        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices],
        dtype=np.int32,
    )


# ── Video Processor ───────────────────────────────────────────────────────────
class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.eye_counter   = 0
        self.mouth_counter = 0
        self._lock = threading.Lock()

        self.result = {
            "status": "WAITING",
            "ear": 0.0,
            "mar": 0.0,
            "eye_counter": 0,
            "mouth_counter": 0,
        }

        # Thresholds — updated from sidebar sliders
        self.ear_thresh  = 0.25
        self.ear_frames  = 30
        self.mar_thresh  = 0.6
        self.yawn_frames = 25

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face landmarks using the Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self.detector.detect(mp_image)

        status = "ALERT"
        ear    = 0.0
        mar    = 0.0

        if not detection.face_landmarks:
            cv2.putText(img, "No face detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            self.eye_counter   = 0
            self.mouth_counter = 0
            status = "WAITING"
        else:
            lm = detection.face_landmarks[0]  # First face

            # ── Compute ratios ────────────────────────────────────────────
            left_ear  = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(lm, UPPER_LIP_IDX, LOWER_LIP_IDX,
                                     MOUTH_CORNER_IDX, w, h)

            # ── Draw contours ─────────────────────────────────────────────
            le_pts = get_contour_pts(lm, LEFT_EYE_CONTOUR, w, h)
            re_pts = get_contour_pts(lm, RIGHT_EYE_CONTOUR, w, h)
            mp_pts = get_contour_pts(lm, LIPS_CONTOUR, w, h)

            cv2.polylines(img, [le_pts], True, (0, 255, 0), 1)
            cv2.polylines(img, [re_pts], True, (0, 255, 0), 1)
            cv2.polylines(img, [mp_pts], True, (0, 255, 255), 1)

            # ── Drowsiness detection ──────────────────────────────────────
            if ear < self.ear_thresh:
                self.eye_counter += 1
                if self.eye_counter >= self.ear_frames:
                    status = "DROWSY"
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(img, "WAKE UP!", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.eye_counter = 0

            # ── Yawn detection ────────────────────────────────────────────
            if mar > self.mar_thresh:
                self.mouth_counter += 1
                if self.mouth_counter >= self.yawn_frames:
                    if status == "ALERT":
                        status = "YAWNING"
                    cv2.putText(img, "YAWNING DETECTED!", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.mouth_counter = 0

            # ── Status dot (top-right) ────────────────────────────────────
            dot_color = {
                "ALERT":   (0, 220, 0),
                "DROWSY":  (0, 0, 255),
                "YAWNING": (0, 220, 220),
            }.get(status, (150, 150, 150))

            cx = w - 28
            cv2.circle(img, (cx, 28), 16, dot_color, -1)
            cv2.circle(img, (cx, 28), 16, (255, 255, 255), 2)

            # ── Metrics bar (bottom-left) ─────────────────────────────────
            cv2.rectangle(img, (0, h - 55), (230, h), (0, 0, 0), -1)
            cv2.putText(img, f"EAR: {ear:.3f}  (thresh {self.ear_thresh:.2f})",
                        (8, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(img, f"MAR: {mar:.3f}  (thresh {self.mar_thresh:.2f})",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Share result with UI
        with self._lock:
            self.result = {
                "status":        status,
                "ear":           ear,
                "mar":           mar,
                "eye_counter":   self.eye_counter,
                "mouth_counter": self.mouth_counter,
            }

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── RTC config (public STUN servers) ─────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚗 Driver Drowsiness Detection System")
st.markdown(
    "Real-time fatigue monitoring using **Eye Aspect Ratio (EAR)** and "
    "**Mouth Aspect Ratio (MAR)** from MediaPipe's 468-point face mesh. "
    "Allow camera access and click **START** to begin."
)
st.divider()

# ── Sidebar — Settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection Settings")
    st.subheader("👁️ Eye (Drowsiness)")
    ear_thresh = st.slider("EAR Threshold",   0.10, 0.50, 0.25, 0.01,
                           help="EAR drops below this → eye is considered closed")
    ear_frames = st.slider("Consec. Frames",  10, 60, 30,
                           help="How many frames eyes must stay closed (~30 = 1 sec at 30 fps)")

    st.subheader("😮 Mouth (Yawning)")
    mar_thresh  = st.slider("MAR Threshold",   0.30, 1.00, 0.60, 0.01,
                            help="MAR rises above this → mouth is considered open")
    yawn_frames = st.slider("Consec. Frames ",  10, 50, 25,
                            help="How many frames mouth must stay open")

    st.divider()
    st.caption("Adjust sliders to suit your lighting and face.")

# ── Layout ────────────────────────────────────────────────────────────────────
vid_col, stat_col = st.columns([3, 1])

with vid_col:
    ctx = webrtc_streamer(
        key="drowsiness-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=DrowsinessDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.ear_thresh  = ear_thresh
        ctx.video_processor.ear_frames  = ear_frames
        ctx.video_processor.mar_thresh  = mar_thresh
        ctx.video_processor.yawn_frames = yawn_frames

with stat_col:
    st.markdown("### 📊 Live Status")

    if ctx.state.playing and ctx.video_processor:
        with ctx.video_processor._lock:
            res = ctx.video_processor.result.copy()

        status = res["status"]
        css_map = {
            "ALERT":   ("alert-green",  "🟢 ALERT"),
            "DROWSY":  ("alert-red",    "🔴 DROWSY"),
            "YAWNING": ("alert-yellow", "🟡 YAWNING"),
            "WAITING": ("alert-gray",   "⚪ WAITING"),
        }
        css_cls, label = css_map.get(status, ("alert-gray", "⚪ WAITING"))
        st.markdown(
            f'<div class="status-box {css_cls}">{label}</div>',
            unsafe_allow_html=True,
        )

        st.metric("EAR",
                  f"{res['ear']:.3f}",
                  delta="⚠ closed" if 0 < res["ear"] < ear_thresh else "open",
                  delta_color="inverse")
        st.metric("MAR",
                  f"{res['mar']:.3f}",
                  delta="⚠ yawning" if res["mar"] > mar_thresh and res["mar"] > 0 else "normal",
                  delta_color="inverse")
        st.metric("Eye counter",   f"{res['eye_counter']} / {ear_frames}")
        st.metric("Mouth counter", f"{res['mouth_counter']} / {yawn_frames}")

        st.caption("Refresh the page or restart stream to reset counters.")
    else:
        st.markdown(
            '<div class="status-box alert-gray">⚪ WAITING</div>',
            unsafe_allow_html=True,
        )
        for label in ("EAR", "MAR", "Eye counter", "Mouth counter"):
            st.metric(label, "—")
        st.info("Start the camera to see live metrics.")

# ── How it works ──────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ How does it work?"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Eye Aspect Ratio (EAR)**

Computed from 6 eye landmark points:

> EAR = (‖p₂−p₆‖ + ‖p₃−p₅‖) / (2 · ‖p₁−p₄‖)

- Open eye → EAR ≈ 0.20–0.35
- Closed eye → EAR ≈ 0.0–0.15
- Alert fires when EAR stays below threshold for N consecutive frames
        """)
    with c2:
        st.markdown("""
**Mouth Aspect Ratio (MAR)**

Computed from 8 mouth landmark points:

> MAR = (‖u₁−l₁‖ + ‖u₂−l₂‖ + ‖u₃−l₃‖) / (3 · ‖c₁−c₂‖)

- Closed mouth → MAR ≈ 0.2–0.4
- Yawning mouth → MAR ≈ 0.7–1.0
- Alert fires when MAR stays above threshold for N consecutive frames
        """)
    st.markdown("""
**Stack:** `MediaPipe Tasks` · `OpenCV` · `streamlit-webrtc`
**Model:** MediaPipe FaceLandmarker — 468-point face mesh (float16)
    """)
