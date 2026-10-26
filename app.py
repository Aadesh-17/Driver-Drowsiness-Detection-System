import streamlit as st
import sys
import traceback

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detector",
    page_icon="🚗",
    layout="wide",
)

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
    import cv2
    import dlib
    import numpy as np
    from scipy.spatial import distance as dist
    from imutils import face_utils
    import av
except Exception as e:
    st.error(f"⚠️ Dependency Import Error: {e}")
    st.markdown("### Detailed Traceback:")
    st.code(traceback.format_exc())
    st.markdown(f"**Python executable:** `{sys.executable}`")
    st.markdown(f"**Python version:** `{sys.version}`")
    st.stop()

import threading
import os
import urllib.request
import bz2

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

# ── Model path & download ─────────────────────────────────────────────────────
MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"
MODEL_BZ2  = MODEL_PATH + ".bz2"
MODEL_URL  = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

@st.cache_resource(show_spinner=False)
def load_models():
    """Load dlib detector and predictor, downloading the landmark model if absent."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        with st.spinner("⬇️ Downloading facial landmark model (~95 MB) — first run only…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_BZ2)
            with bz2.open(MODEL_BZ2, "rb") as f_in, open(MODEL_PATH, "wb") as f_out:
                f_out.write(f_in.read())
            os.remove(MODEL_BZ2)

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    return detector, predictor

# ── Ratio helpers ─────────────────────────────────────────────────────────────
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# ── Video Processor ───────────────────────────────────────────────────────────
class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.detector, self.predictor = load_models()
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.eye_counter   = 0
        self.mouth_counter = 0
        self._lock = threading.Lock()

        # Live result shared with the UI thread
        self.result = {
            "status": "WAITING",
            "ear": 0.0,
            "mar": 0.0,
            "eye_counter": 0,
            "mouth_counter": 0,
        }

        # Thresholds — updated from UI
        self.ear_thresh  = 0.3
        self.ear_frames  = 30
        self.mar_thresh  = 0.5
        self.yawn_frames = 25

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img  = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        status = "ALERT"
        ear    = 0.0
        mar    = 0.0

        if len(faces) == 0:
            cv2.putText(img, "No face detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            self.eye_counter   = 0
            self.mouth_counter = 0
            status = "WAITING"

        for face in faces:
            shape = face_utils.shape_to_np(self.predictor(gray, face))

            leftEye  = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth    = shape[self.mStart:self.mEnd]

            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Draw facial contours
            cv2.drawContours(img, [cv2.convexHull(leftEye)],  -1, (0, 255, 0),   1)
            cv2.drawContours(img, [cv2.convexHull(rightEye)], -1, (0, 255, 0),   1)
            cv2.drawContours(img, [cv2.convexHull(mouth)],    -1, (0, 255, 255), 1)

            # Face bounding box
            cv2.rectangle(img,
                          (face.left(), face.top()),
                          (face.right(), face.bottom()),
                          (200, 200, 200), 1)

            # ── Drowsiness detection ──────────────────────────────────────────
            if ear < self.ear_thresh:
                self.eye_counter += 1
                if self.eye_counter >= self.ear_frames:
                    status = "DROWSY"
                    # Red tint overlay
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]),
                                  (0, 0, 200), -1)
                    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(img, "WAKE UP!", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.eye_counter = 0

            # ── Yawn detection ────────────────────────────────────────────────
            if mar > self.mar_thresh:
                self.mouth_counter += 1
                if self.mouth_counter >= self.yawn_frames:
                    if status == "ALERT":
                        status = "YAWNING"
                    cv2.putText(img, "YAWNING DETECTED!", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.mouth_counter = 0

            # ── Status dot (top-right) ────────────────────────────────────────
            dot_color = {
                "ALERT":   (0, 220, 0),
                "DROWSY":  (0, 0, 255),
                "YAWNING": (0, 220, 220),
                "WAITING": (150, 150, 150),
            }.get(status, (150, 150, 150))

            cx = img.shape[1] - 28
            cv2.circle(img, (cx, 28), 16, dot_color, -1)
            cv2.circle(img, (cx, 28), 16, (255, 255, 255), 2)

            # ── Metrics bar (bottom-left) ─────────────────────────────────────
            h = img.shape[0]
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
    "**Mouth Aspect Ratio (MAR)** from dlib's 68-point facial landmark detector. "
    "Allow camera access and click **START** to begin."
)
st.divider()

# ── Sidebar — Settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection Settings")
    st.subheader("👁️ Eye (Drowsiness)")
    ear_thresh = st.slider("EAR Threshold",   0.10, 0.50, 0.30, 0.01,
                           help="EAR drops below this → eye is considered closed")
    ear_frames = st.slider("Consec. Frames",  10, 60, 30,
                           help="How many frames eyes must stay closed (~30 = 1 sec at 30 fps)")

    st.subheader("😮 Mouth (Yawning)")
    mar_thresh  = st.slider("MAR Threshold",   0.30, 0.80, 0.50, 0.01,
                            help="MAR rises above this → mouth is considered open")
    yawn_frames = st.slider("Consec. Frames",  10, 50, 25,
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

    # Push updated thresholds into the processor
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
            unsafe_allow_html=True
        )

        st.metric("EAR",
                  f"{res['ear']:.3f}",
                  delta="⚠ closed" if res["ear"] < ear_thresh and res["ear"] > 0 else "open",
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
            unsafe_allow_html=True
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

- Open eye → EAR ≈ 0.25–0.35
- Closed eye → EAR ≈ 0.0–0.15
- Alert fires when EAR stays below threshold for N consecutive frames
        """)
    with c2:
        st.markdown("""
**Mouth Aspect Ratio (MAR)**

Computed from 8 mouth landmark points:

> MAR = (‖p₃−p₁₁‖ + ‖p₅−p₉‖) / (2 · ‖p₁−p₇‖)

- Closed mouth → MAR ≈ 0.2–0.4
- Yawning mouth → MAR ≈ 0.6–0.9
- Alert fires when MAR stays above threshold for N consecutive frames
        """)
    st.markdown("""
**Stack:** `dlib` · `OpenCV` · `imutils` · `scipy` · `streamlit-webrtc`  
**Model:** dlib's 68-point shape predictor trained on iBUG 300-W dataset
    """)
