# 🚗 Driver Drowsiness Detection System

> 🌐 **[Live Demo →](https://your-app-name.streamlit.app)** *(update this link after deploying)*

A real-time driver drowsiness detection system using computer vision and facial landmark analysis. It monitors eye closure and yawning to detect signs of fatigue and trigger audio alerts — helping prevent accidents caused by drowsy driving.

---

## 📸 Features

- 👁️ **Eye Drowsiness Detection** — Uses Eye Aspect Ratio (EAR) to detect prolonged eye closure
- 😮 **Yawn Detection** — Uses Mouth Aspect Ratio (MAR) to detect yawning
- 🔊 **Multi-level Audio Alerts** — Plays alarm sound (`pygame`), with a sine-wave beep fallback
- 📊 **Real-time Metrics Overlay** — Live EAR, MAR, and counter values on the video feed
- 🎨 **Visual Indicators** — Color-coded status circles (🟢 alert / 🔴 drowsy / 🟡 yawning)
- ⚡ **Alarm Throttling** — Prevents alert spam with configurable cooldown intervals

---

## 🧠 How It Works

The system uses **dlib's 68-point facial landmark detector** to identify key facial points. From these landmarks:

- **Eye Aspect Ratio (EAR)** — Computed from the vertical and horizontal eye landmark distances. A low EAR indicates closed eyes.

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

- **Mouth Aspect Ratio (MAR)** — Computed similarly for the mouth region. A high MAR indicates an open mouth (yawn).

| Condition | Threshold | Consecutive Frames | Action |
|---|---|---|---|
| Eyes closed (drowsy) | EAR < `0.3` | ≥ 15 frames | 🔊 Alarm + on-screen alert |
| Mouth open (yawning) | MAR > `0.5` | ≥ 15 frames | 🟡 On-screen warning + beep |

---

## 📁 Project Structure

```
Driver Drowsiness Detection System/
├── main.py                                   # Main application script
├── requirements.txt                          # Python dependencies
├── models/
│   └── shape_predictor_68_face_landmarks.dat # dlib landmark model
├── haarcascade/
│   ├── haarcascade_frontalface_default.xml   # Haar cascade (reference)
│   └── haarcascade_eye.xml
└── audio/
    └── alarm.wav                             # Alert sound file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- Webcam
- CMake (required to build `dlib`)

### 1. Clone the repository

```bash
git clone https://github.com/Aadesh-17/Driver-Drowsiness-Detection-System.git
cd Driver-Drowsiness-Detection-System
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dlib landmark model

Download `shape_predictor_68_face_landmarks.dat` from the [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it, and place it in the `models/` directory.

### 5. Add an alarm sound

Place an `alarm.wav` file inside the `audio/` directory. If absent, the system falls back to a built-in sine-wave beep.

---

## ▶️ Usage

### Desktop (original app)
```bash
python main.py
```

### Web demo (Streamlit)
```bash
streamlit run app.py
```

### Controls

| Key | Action |
|---|---|
| `ESC` | Quit the application |
| `T` | Test the alarm sound |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video capture & frame processing |
| `dlib` | Face detection & 68-point landmark prediction |
| `imutils` | Facial landmark index utilities |
| `numpy` | Numerical computations |
| `scipy` | Euclidean distance calculations |
| `pygame` | Audio playback for alarms |

---

## 🔧 Configuration

You can tune detection sensitivity by modifying these constants in `main.py`:

```python
EYE_AR_THRESH = 0.3        # EAR below this triggers drowsiness
EYE_AR_CONSEC_FRAMES = 15  # Consecutive frames before alert fires
MOUTH_AR_THRESH = 0.5      # MAR above this triggers yawn detection
YAWN_CONSEC_FRAMES = 15    # Consecutive frames before yawn warning
```

---

## 📄 License

This project is open-source. Feel free to use and modify it.

---

## 🙌 Acknowledgements

- [dlib](http://dlib.net/) — Face detection and landmark prediction
- [OpenCV](https://opencv.org/) — Computer vision framework
- [imutils](https://github.com/jrosebr1/imutils) — Facial landmark utilities
