"""
Indian Sign Language (ISL) Recognition System
B.Tech Project - Real-Time Vision-Based ISL Recognition
"""

import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf
from collections import deque
import time
import threading
import base64
import io
from PIL import Image

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Recognition System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0a0f1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --accent: #00e676;
    --accent2: #00bfa5;
    --accent3: #1de9b6;
    --text-primary: #f0f4f8;
    --text-muted: #8899aa;
    --border: rgba(0, 230, 118, 0.15);
    --danger: #ff5252;
    --warning: #ffab40;
}

* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: var(--bg-primary) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0f1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Header */
.isl-header {
    background: linear-gradient(135deg, #0d1f3a 0%, #0a2f1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    overflow: hidden;
}
.isl-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,230,118,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.isl-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e676, #1de9b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.isl-header p {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 4px 0 0 0;
}
.isl-badge {
    background: rgba(0,230,118,0.12);
    border: 1px solid rgba(0,230,118,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #00e676;
    letter-spacing: 1px;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Prediction Box */
.pred-box {
    background: linear-gradient(135deg, #0d1f3a, #0a2f1a);
    border: 2px solid var(--accent);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 30px rgba(0,230,118,0.1);
}
.pred-box::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,230,118,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.pred-letter {
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00e676, #1de9b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 8px;
}
.pred-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Confidence bar */
.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    margin: 14px 0 4px 0;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00e676, #1de9b6);
    transition: width 0.3s ease;
    box-shadow: 0 0 10px rgba(0,230,118,0.4);
}

/* Hold progress ring area */
.hold-area {
    margin-top: 16px;
}
.hold-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}
.hold-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.2s ease;
}
.hold-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}

/* Sentence display */
.sentence-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    min-height: 70px;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 4px;
    word-break: break-all;
    position: relative;
}
.sentence-cursor {
    display: inline-block;
    width: 3px;
    height: 1.8rem;
    background: var(--accent);
    animation: blink 1s infinite;
    vertical-align: text-bottom;
    margin-left: 2px;
    border-radius: 2px;
}
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}

/* Status indicator */
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-live { background: #00e676; box-shadow: 0 0 8px #00e676; animation: pulse 2s infinite; }
.status-off  { background: #555; }
@keyframes pulse {
    0%,100% { box-shadow: 0 0 4px #00e676; }
    50%      { box-shadow: 0 0 14px #00e676; }
}

/* Action buttons */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}

/* Stats row */
.stat-item {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e676, #1de9b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* Info box */
.info-box {
    background: rgba(0,191,165,0.07);
    border: 1px solid rgba(0,191,165,0.2);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.8rem;
    color: #80cbc4;
    margin: 8px 0;
}

/* slider overrides */
[data-testid="stSlider"] label {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,230,118,0.3); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)

# ─── Class Labels ────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0:'1', 1:'2', 2:'3', 3:'4', 4:'5', 5:'6', 6:'7', 7:'8', 8:'9',
    9:'A', 10:'B', 11:'C', 12:'D', 13:'E', 14:'F', 15:'G', 16:'H',
    17:'I', 18:'J', 19:'K', 20:'L', 21:'M', 22:'N', 23:'O', 24:'P',
    25:'Q', 26:'R', 27:'S', 28:'T', 29:'U', 30:'V', 31:'W', 32:'X',
    33:'Y', 34:'Z'
}

# ─── Model Loading ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_1_aug.h5", compile=False)
    return model

model = load_model()

# ─── MediaPipe Setup (Tasks API — mediapipe 0.10+) ──────────────────────────────
_hand_model_path = "hand_landmarker.task"
_base_opts = mp_tasks.BaseOptions(model_asset_path=_hand_model_path)
_lm_opts   = mp_vision.HandLandmarkerOptions(
    base_options=_base_opts,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp_vision.RunningMode.VIDEO,
)
HAND_LANDMARKER = mp_vision.HandLandmarker.create_from_options(_lm_opts)

# Hand connections for manual drawing (21 landmarks)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (0,9),(9,10),(10,11),(11,12),     # Middle
    (0,13),(13,14),(14,15),(15,16),   # Ring
    (0,17),(17,18),(18,19),(19,20),   # Pinky
    (5,9),(9,13),(13,17),             # Palm
]

# ─── Session State ───────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "sentence":        "",
        "hold_count":      0,
        "last_pred":       None,
        "running":         False,
        "total_preds":     0,
        "session_start":   time.time(),
        "letters_added":   0,
        "cap":             None,
        "frame_rgb":       None,
        "pred_letter":     "—",
        "pred_conf":       0.0,
        "hand_detected":   False,
        "hold_frames":     0,
        "conf_threshold":  0.80,
        "hold_threshold":  15,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Helper Functions ────────────────────────────────────────────────────────────
def preprocess_hand(roi: np.ndarray) -> np.ndarray:
    """Resize and normalize hand ROI for the model."""
    img = cv2.resize(roi, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def get_hand_bbox(hand_landmarks, frame_shape, padding: int = 30):
    """Return (x1, y1, x2, y2) bounding box for hand landmarks."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    return x1, y1, x2, y2


def speak_text(text: str):
    """Text-to-speech using pyttsx3 in a background thread."""
    try:
        import pyttsx3
        def _speak():
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()
    except Exception as e:
        st.warning(f"TTS unavailable: {e}")


def frame_to_base64(frame_bgr: np.ndarray) -> str:
    """Convert BGR frame to base64 JPEG for HTML <img>."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _draw_hand_landmarks(frame_bgr, landmarks, h, w):
    """Manually draw hand landmarks and connections using cv2."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    # Draw connections
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)
    # Draw joints
    for i, (px, py) in enumerate(pts):
        radius = 5 if i in (0, 5, 9, 13, 17) else 3   # bigger for knuckles/wrist
        cv2.circle(frame_bgr, (px, py), radius, (0, 230, 118), -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (px, py), radius + 1, (0, 100, 50), 1, cv2.LINE_AA)


class _FakeLandmarks:
    """Adapter so get_hand_bbox works with Tasks API landmark objects."""
    def __init__(self, lms):
        self.landmark = lms


_frame_timestamp_ms = 0  # global counter for Tasks VIDEO mode


def process_frame(frame_bgr: np.ndarray, conf_threshold: float, hold_threshold: int):
    """
    Process one webcam frame:
    - Detect hand with MediaPipe Tasks HandLandmarker
    - Draw hand landmarks manually
    - Crop hand ROI
    - Predict letter with CNN
    - Update hold counter and sentence
    Returns annotated frame + metadata dict.
    """
    global _frame_timestamp_ms
    _frame_timestamp_ms += 40  # ~25 fps

    meta = {
        "hand_detected": False,
        "pred_letter":   "—",
        "pred_conf":     0.0,
        "hold_frames":   st.session_state.hold_frames,
        "added":         False,
    }

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img    = MpImage(image_format=ImageFormat.SRGB, data=frame_rgb)
    results   = HAND_LANDMARKER.detect_for_video(mp_img, _frame_timestamp_ms)

    if results.hand_landmarks:
        meta["hand_detected"] = True
        lms = results.hand_landmarks[0]   # first hand only

        # Draw landmarks
        _draw_hand_landmarks(frame_bgr, lms, h, w)

        # Bounding box & crop (reuse existing helper via adapter)
        fake = _FakeLandmarks(lms)
        x1, y1, x2, y2 = get_hand_bbox(fake, frame_bgr.shape)
        hand_roi = frame_bgr[y1:y2, x1:x2]

        if hand_roi.size > 0:
            inp   = preprocess_hand(hand_roi)
            preds = model.predict(inp, verbose=0)[0]
            cls   = int(np.argmax(preds))
            conf  = float(preds[cls])

            meta["pred_letter"] = CLASS_LABELS[cls]
            meta["pred_conf"]   = conf

            # Draw bounding box
            color = (0, 230, 118) if conf >= conf_threshold else (100, 100, 100)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

            # Overlay label
            label = f"{CLASS_LABELS[cls]}  {conf*100:.1f}%"
            cv2.putText(frame_bgr, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2, cv2.LINE_AA)

            # Hold logic
            if conf >= conf_threshold:
                if st.session_state.last_pred == cls:
                    st.session_state.hold_frames += 1
                else:
                    st.session_state.last_pred   = cls
                    st.session_state.hold_frames = 1

                if st.session_state.hold_frames >= hold_threshold:
                    st.session_state.sentence     += CLASS_LABELS[cls]
                    st.session_state.hold_frames   = 0
                    st.session_state.last_pred     = None
                    st.session_state.letters_added += 1
                    meta["added"] = True
            else:
                st.session_state.hold_frames = 0
                st.session_state.last_pred   = None

            meta["hold_frames"] = st.session_state.hold_frames
    else:
        st.session_state.hold_frames = 0
        st.session_state.last_pred   = None

    st.session_state.total_preds += 1
    return frame_bgr, meta

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px 0;'>
        <div style='font-size:2.5rem;'>🤟</div>
        <div style='font-size:1rem;font-weight:700;color:#00e676;'>ISL Recognition</div>
        <div style='font-size:0.7rem;color:#8899aa;'>B.Tech Project — Vision AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold", 0.70, 0.99, 0.80, 0.01,
        help="Minimum confidence to register a prediction"
    )
    hold_threshold = st.slider(
        "Hold Frames to Add Letter", 5, 30, 15, 1,
        help="Number of consecutive frames the same letter must be held"
    )
    st.session_state.conf_threshold = conf_threshold
    st.session_state.hold_threshold = hold_threshold

    st.markdown("---")
    st.markdown("### 📖 How to Use")
    steps = [
        ("1️⃣", "Click **Start Camera**"),
        ("2️⃣", "Hold your hand sign in front of camera"),
        ("3️⃣", "Keep the sign steady until the hold bar fills"),
        ("4️⃣", "Letter is added automatically"),
        ("5️⃣", "Use buttons to clear / add space / speak"),
    ]
    for icon, txt in steps:
        st.markdown(f"""
        <div class='info-box'>
            <b>{icon}</b> {txt}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂️ ISL Alphabet Reference")
    labels_str = "  ".join(CLASS_LABELS.values())
    st.markdown(f"""
    <div style='background:rgba(0,230,118,0.07);border:1px solid rgba(0,230,118,0.15);
                border-radius:10px;padding:10px;font-size:0.85rem;
                color:#00e676;letter-spacing:3px;font-weight:600;word-break:break-all;'>
    {labels_str}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#556677;text-align:center;'>
        Model: <b>model_1_aug.h5</b><br>
        Input: 224×224 RGB • 35 Classes<br>
        Framework: TensorFlow / Keras
    </div>
    """, unsafe_allow_html=True)

# ─── Main Layout ─────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div class='isl-header'>
    <div>
        <span class='isl-badge'>LIVE DEMO</span>
        <h1>🤟 Indian Sign Language Recognition</h1>
        <p>Real-time vision-based ISL detection using MediaPipe + Custom CNN · 35 Classes · 224×224 input</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Camera Controls ─────────────────────────────────────────────────────────────
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])
with ctrl_col1:
    start_btn = st.button("▶ Start Camera",  use_container_width=True,
                          type="primary" if not st.session_state.running else "secondary")
with ctrl_col2:
    stop_btn  = st.button("⏹ Stop Camera",   use_container_width=True)
with ctrl_col3:
    space_btn = st.button("␣ Add Space",      use_container_width=True)
with ctrl_col4:
    clear_btn = st.button("🗑 Clear Text",     use_container_width=True)

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
if space_btn:
    st.session_state.sentence += " "
if clear_btn:
    st.session_state.sentence      = ""
    st.session_state.hold_frames   = 0
    st.session_state.last_pred     = None
    st.session_state.letters_added = 0

st.markdown("<br>", unsafe_allow_html=True)

# ─── Two-Column Layout: Feed | Prediction ────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown("""
    <div class='card-title'>
        <span style='background:rgba(0,230,118,0.15);border-radius:50%;padding:4px 8px;'>📹</span>
        WEBCAM FEED
    </div>
    """, unsafe_allow_html=True)
    feed_placeholder = st.empty()
    status_placeholder = st.empty()

with right_col:
    # Prediction card
    pred_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)

    # Hold progress card
    hold_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    stat_placeholder = st.empty()

# ─── Sentence Display ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class='card-title'>
    <span style='background:rgba(0,230,118,0.15);border-radius:50%;padding:4px 8px;'>📝</span>
    BUILT SENTENCE
</div>
""", unsafe_allow_html=True)
sentence_placeholder = st.empty()

# Speak button row
speak_col1, speak_col2, _ = st.columns([1, 1, 2])
with speak_col1:
    speak_btn = st.button("🔊 Speak Sentence", use_container_width=True)
with speak_col2:
    copy_btn  = st.button("📋 Copy Text",       use_container_width=True)

if speak_btn and st.session_state.sentence.strip():
    speak_text(st.session_state.sentence)
    st.toast(f"Speaking: {st.session_state.sentence}", icon="🔊")

if copy_btn:
    st.toast("Sentence copied!", icon="📋")

# ─── Render Static State ────────────────────────────────────────────────────────
def render_static():
    """Render all UI placeholders when camera is not running."""
    # Feed placeholder
    feed_placeholder.markdown("""
    <div style='background:#0d1525;border:2px dashed rgba(0,230,118,0.2);
                border-radius:14px;height:380px;display:flex;align-items:center;
                justify-content:center;flex-direction:column;gap:12px;'>
        <div style='font-size:3rem;'>📷</div>
        <div style='font-size:1rem;color:#445566;font-weight:500;'>Camera not started</div>
        <div style='font-size:0.8rem;color:#334455;'>Click ▶ Start Camera above</div>
    </div>
    """, unsafe_allow_html=True)

    status_placeholder.markdown("""
    <div style='font-size:0.8rem;color:#556677;padding:4px 0;'>
        <span class='status-dot status-off'></span> Camera offline
    </div>
    """, unsafe_allow_html=True)

    pred_placeholder.markdown(f"""
    <div class='pred-box'>
        <div class='pred-letter'>—</div>
        <div class='pred-label'>Waiting for prediction…</div>
        <div class='conf-track'><div class='conf-fill' style='width:0%'></div></div>
        <div style='font-size:0.72rem;color:#556677;margin-top:4px;'>Confidence: 0%</div>
    </div>
    """, unsafe_allow_html=True)

    hold_placeholder.markdown(f"""
    <div class='card'>
        <div class='card-title'>⏱ HOLD PROGRESS</div>
        <div class='hold-label'>
            <span>Hold steady to add letter</span>
            <span>0 / {st.session_state.hold_threshold}</span>
        </div>
        <div class='hold-track'>
            <div class='hold-fill' style='width:0%;background:linear-gradient(90deg,#00e676,#1de9b6);'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    elapsed = int(time.time() - st.session_state.session_start)
    stat_placeholder.markdown(f"""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
        <div class='stat-item'>
            <div class='stat-value'>{st.session_state.letters_added}</div>
            <div class='stat-label'>Letters Added</div>
        </div>
        <div class='stat-item'>
            <div class='stat-value'>{elapsed}s</div>
            <div class='stat-label'>Session Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sentence = st.session_state.sentence or ""
    display  = sentence if sentence else "&nbsp;"
    sentence_placeholder.markdown(f"""
    <div class='sentence-box'>
        {display}<span class='sentence-cursor'></span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Loop ──────────────────────────────────────────────────────────────────
if st.session_state.running:
    # Open camera
    if st.session_state.cap is None or not st.session_state.cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Make sure it is connected and not used by another app.")
            st.session_state.running = False
        else:
            st.session_state.cap = cap

    if st.session_state.running and st.session_state.cap and st.session_state.cap.isOpened():
        # Live loop
        frame_count = 0
        while st.session_state.running:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning("⚠️ Failed to read from camera. Check your webcam.")
                break

            frame = cv2.flip(frame, 1)
            annotated, meta = process_frame(
                frame,
                st.session_state.conf_threshold,
                st.session_state.hold_threshold,
            )

            # Update feed
            b64 = frame_to_base64(annotated)
            feed_placeholder.markdown(f"""
            <img src='data:image/jpeg;base64,{b64}'
                 style='border-radius:14px;width:100%;border:2px solid rgba(0,230,118,0.25);'/>
            """, unsafe_allow_html=True)

            # Status
            dot_cls = "status-live" if meta["hand_detected"] else "status-off"
            hand_txt = "Hand detected" if meta["hand_detected"] else "No hand detected"
            status_placeholder.markdown(f"""
            <div style='font-size:0.8rem;color:#8899aa;padding:4px 0;margin-top:4px;'>
                <span class='status-dot {dot_cls}'></span> {hand_txt} • Frame {st.session_state.total_preds}
            </div>
            """, unsafe_allow_html=True)

            # Prediction box
            letter = meta["pred_letter"]
            conf   = meta["pred_conf"]
            conf_w = int(conf * 100)
            conf_color = "#00e676" if conf >= st.session_state.conf_threshold else "#ffab40"
            pred_placeholder.markdown(f"""
            <div class='pred-box'>
                <div class='pred-letter'>{letter}</div>
                <div class='pred-label'>Predicted Sign</div>
                <div class='conf-track'>
                    <div class='conf-fill' style='width:{conf_w}%;background:linear-gradient(90deg,{conf_color},{conf_color}99);'></div>
                </div>
                <div style='font-size:0.78rem;color:#8899aa;margin-top:6px;'>
                    Confidence: <b style='color:{conf_color};'>{conf*100:.1f}%</b>
                    &nbsp;•&nbsp; Threshold: {st.session_state.conf_threshold*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Hold progress
            hold_f  = meta["hold_frames"]
            hold_th = st.session_state.hold_threshold
            hold_pct = min(100, int(hold_f / max(hold_th, 1) * 100))
            if hold_f > 0:
                hue = int(120 * hold_pct / 100)
                hold_color = f"hsl({hue}, 80%, 55%)"
            else:
                hold_color = "#335544"
            hold_placeholder.markdown(f"""
            <div class='card'>
                <div class='card-title'>⏱ HOLD PROGRESS</div>
                <div class='hold-label'>
                    <span>{"🟢 Adding letter!" if meta.get("added") else "Hold steady to add letter"}</span>
                    <span style='color:#00e676;'>{hold_f} / {hold_th}</span>
                </div>
                <div class='hold-track'>
                    <div class='hold-fill' style='width:{hold_pct}%;background:linear-gradient(90deg,{hold_color},{hold_color}cc);'>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Stats
            elapsed = int(time.time() - st.session_state.session_start)
            stat_placeholder.markdown(f"""
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div class='stat-item'>
                    <div class='stat-value'>{st.session_state.letters_added}</div>
                    <div class='stat-label'>Letters Added</div>
                </div>
                <div class='stat-item'>
                    <div class='stat-value'>{elapsed}s</div>
                    <div class='stat-label'>Session Time</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sentence
            sentence = st.session_state.sentence or ""
            display  = sentence if sentence else "&nbsp;"
            sentence_placeholder.markdown(f"""
            <div class='sentence-box'>
                {display}<span class='sentence-cursor'></span>
            </div>
            """, unsafe_allow_html=True)

            frame_count += 1
            time.sleep(0.04)  # ~25 fps cap

else:
    render_static()

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#334455;font-size:0.72rem;border-top:1px solid rgba(0,230,118,0.08);padding-top:16px;'>
    🤟 Indian Sign Language Recognition System &nbsp;•&nbsp; B.Tech Final Year Project &nbsp;•&nbsp;
    Built with TensorFlow · MediaPipe · OpenCV · Streamlit
</div>
""", unsafe_allow_html=True)
