# 🤟 Indian Sign Language (ISL) Recognition System
**Real-Time Vision-Based ISL Recognition — B.Tech Final Year Project**

---

## 📌 Overview

A Streamlit web application that recognises **35 Indian Sign Language gestures** (digits 1–9 + letters A–Z) in real time using a webcam.

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow / Keras — Custom CNN (`model_1_aug.h5`) |
| Hand Detection | MediaPipe Hands |
| Video Capture | OpenCV |
| UI Framework | Streamlit |
| Text-to-Speech | pyttsx3 |

---

## 🗂️ Repository Structure

```
ProjectOpenCv/
├── app.py              ← Main Streamlit application
├── model_1_aug.h5      ← Trained Keras model (35-class ISL CNN)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🚀 Quick Start (Local)

### 1. Clone / navigate to the project
```bash
cd ProjectOpenCv
```

### 2. Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate.bat    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note for Linux:** If `pyttsx3` fails, install the system dependency:
> ```bash
> sudo apt-get install espeak
> ```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎮 How to Use

1. **Sidebar** — Adjust *Confidence Threshold* and *Hold Frames* to match your environment.
2. Click **▶ Start Camera** — webcam feed appears with hand skeleton overlay.
3. **Show a sign** and hold it **steady** — watch the hold-progress bar fill up.
4. Once the bar is full (default: 15 frames), the **letter is added** to the sentence.
5. Use the action buttons:
   | Button | Action |
   |--------|--------|
   | ␣ Add Space | Insert a space into the sentence |
   | 🗑 Clear Text | Wipe the sentence |
   | 🔊 Speak Sentence | Read sentence aloud via TTS |
6. Click **⏹ Stop Camera** when done.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | Custom CNN (Sequential) |
| Input Size | 224 × 224 × 3 (RGB) |
| Output Classes | 35 (1–9, A–Z) |
| Validation Accuracy | ~99.7% |
| Parameters | ~58 K |

### Class Labels
```
0→1  1→2  2→3  3→4  4→5  5→6  6→7  7→8  8→9
9→A 10→B 11→C 12→D 13→E 14→F 15→G 16→H 17→I
18→J 19→K 20→L 21→M 22→N 23→O 24→P 25→Q 26→R
27→S 28→T 29→U 30→V 31→W 32→X 33→Y 34→Z
```

---

## ☁️ Streamlit Cloud Deployment

1. Push the repo to GitHub (include `model_1_aug.h5`).
2. On [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo.
3. Replace `pyttsx3` with `gTTS` in `app.py` (cloud has no audio output device):

```python
# Cloud TTS replacement — add to app.py
from gtts import gTTS
import tempfile, os

def speak_text(text: str):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        st.audio(f.name, format="audio/mp3")
```

4. Update `requirements.txt` — swap `pyttsx3` for `gTTS`.

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Webcam not found | Ensure no other app is using the camera; try a different USB port |
| `pyttsx3` error on Linux | `sudo apt-get install espeak libespeak-dev` |
| Low accuracy | Increase background contrast; ensure hand is fully visible |
| Slow prediction | Reduce *Hold Frames* threshold; ensure GPU drivers are installed |
| MediaPipe import error | `pip install mediapipe --upgrade` |

---

## 📄 License
MIT © 2024 — B.Tech Final Year Project
