# LipSync

**Real-Time Lip Reading and Hand Gesture Translation System for Industrial Communication**

Developed by **Karthik**.

A Streamlit-based dashboard that uses your webcam to detect a person, track
face/lip/hand landmarks, recognise industrial hand gestures, heuristically
read lip movement against a small vocabulary, fuse both into a sentence,
speak it aloud, and log everything to a searchable/exportable history.

---

## ⚠️ Please read before your viva / report

**Lip reading here is a heuristic mouth-shape matcher, not a trained deep
learning model.** True visual-speech-recognition (e.g. LipNet-style models)
requires training on thousands of labelled lip-video clips (e.g. the GRID
corpus) with a GPU. That's out of scope for a local mini project with no
dataset.

Instead, `detectors/lip_detector.py`:
1. Tracks how open your mouth is, frame by frame (normalised by face width).
2. Computes simple statistics over a rolling window (mean, std-dev, number
   of open/close cycles).
3. Matches those statistics against hand-authored templates for 7 words
   (`config.py → LIP_TEMPLATES`).

This is a legitimate, explainable engineering approach for a demo and is
genuinely real-time and functional — just be upfront in your report that it
is rule-based pattern matching, not a trained neural network. Section
"Extending to a Real ML Model" below explains how you could upgrade it.

**Hand gesture recognition is fully rule-based on real geometry** (finger-up/
down state, thumb orientation, fingertip distances) computed from MediaPipe's
21 hand landmarks — this part is accurate and works well live.

---

## Features

- Live webcam feed in the browser (no external camera app needed)
- Person detection gate: "No Person Detected" / "Multiple Persons Detected" /
  processes only when exactly one person is present
- Face detection with bounding box + confidence
- Lip landmark tracking + heuristic word matching, animated subtitle box
- Hand gesture recognition mapped to industrial meanings (STOP, HELP,
  EMERGENCY, OK, MOVE, WAIT, COME, GO)
- Combined sentence fusion (gesture + lip word → sentence)
- Subtitle history panel: scrollable, copy, clear, download TXT/CSV/PDF
- Offline text-to-speech (mute/volume/voice selection)
- SQLite-backed prediction history with CSV export
- Dashboard-style dark UI: glassmorphism cards, gradient accents, sidebar
  navigation, live FPS/CPU/time in the top bar
- Settings page (confidence thresholds, camera index, subtitle size,
  language, speech settings)
- About page (project/developer/version/license/tech stack)
- Unit tests for the gesture classifier, lip-template matcher, sentence
  combiner, and database layer (no camera required to run them)

---

## Tech Stack

| Purpose              | Library              |
|-----------------------|-----------------------|
| UI / dashboard         | Streamlit             |
| Live webcam in browser | streamlit-webrtc, av  |
| Computer vision        | OpenCV                |
| Face/hand landmarks    | MediaPipe             |
| Data                   | NumPy, Pandas         |
| Database               | SQLite (built-in)     |
| Speech                 | pyttsx3 (offline TTS) |
| PDF export              | fpdf2                 |
| System stats            | psutil                |

> We deliberately did **not** use YOLO/PyTorch/TensorFlow — MediaPipe already
> gives fast, accurate face/hand landmarks with no GPU and no extra model
> downloads, which keeps this runnable on a normal laptop. You can swap in a
> YOLO hand-detector or a trained lip-reading model later (see below) without
> restructuring the project, since detectors are isolated modules.

---

## Project Structure

```
lipsync/
├── lipsync.py                 # Streamlit entrypoint (UI, pages, camera pipeline)
├── config.py                  # All thresholds, vocab, and constants
├── requirements.txt
├── README.md
├── database/
│   └── db_manager.py          # SQLite history persistence
├── detectors/
│   ├── person_detector.py     # "is exactly one person visible?"
│   ├── face_detector.py       # face bbox + 468-pt mesh landmarks
│   ├── lip_detector.py        # heuristic lip-word matcher
│   └── hand_detector.py       # rule-based gesture classifier
├── translators/
│   └── combiner.py            # fuses gesture + lip word into a sentence
├── services/
│   ├── speech_service.py      # offline TTS (pyttsx3)
│   └── export_service.py      # TXT / CSV / PDF export
├── ui/
│   ├── styles.py               # professional dark UI theme (CSS)
│   └── components.py           # reusable Streamlit render helpers
├── utils/
│   └── logger.py               # rotating file + console logger
├── tests/
│   ├── test_hand_detector.py
│   ├── test_lip_detector.py
│   ├── test_combiner.py
│   └── test_db.py
├── logs/                       # lipsync.log written here at runtime
└── history_exports/            # generated exports land here
```

---

## Installation (VS Code / local machine)

### 1. Requirements
- Python 3.10 or 3.11 (MediaPipe does not yet support every 3.12/3.13 build —
  if `pip install mediapipe` fails, install Python 3.11 and retry)
- A working webcam
- Windows/macOS/Linux — pyttsx3 uses your OS's built-in TTS voices

### 2. Unzip and open in VS Code
Unzip `lipsync.zip`, then in VS Code: **File → Open Folder** →
select the `lipsync` folder.

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run lipsync.py
```
This opens `http://localhost:8501` in your browser. Click **START** on the
video widget to grant camera access and begin live detection.

### 6. Run the unit tests
```bash
pip install pytest
pytest tests/ -v
```
(These test the classifier logic and database layer directly — no camera
needed.)

---

## Using the App

1. **Dashboard / Live Detection** — start your camera, see live subtitles,
   the combined sentence, prediction cards, and (on Dashboard) a right-hand
   statistics panel with recent words/gestures and a confidence trend chart.
2. **Hand Signs** — reference table of every supported gesture and its
   industrial meaning, plus the currently detected gesture.
3. **Lip Reading** — current heuristic word, live mouth-aperture value, and
   the supported vocabulary list.
4. **Subtitle History** — full prediction log with copy/clear/download
   (TXT/CSV/PDF).
5. **Settings** — tune detection confidence thresholds, camera index,
   subtitle size, language, and speech (volume/mute/voice).
6. **About** — project, developer, version, license, and tech stack.

---

## Extending to a Real ML Model (future work)

If you want to go beyond the heuristic and build a genuinely trained
lip-reading model for your final-year project or beyond:

1. Collect/download a labelled lip-video dataset (e.g. GRID corpus, or
   record your own fixed-vocabulary dataset with a phone camera).
2. Extract per-frame lip-region crops using `detectors/lip_detector.py`'s
   landmark extraction as a starting point.
3. Train a small 3D-CNN + GRU/LSTM sequence model (similar to LipNet) on
   Google Colab (free GPU) using PyTorch or TensorFlow.
4. Export the trained model (ONNX or TorchScript) and swap it into
   `LipDetector.process()` in place of the template-matching logic — the
   rest of the app (UI, combiner, DB, TTS) doesn't need to change.

You could do the same for gestures — replace the rule-based classifier with
a small trained MLP on landmark vectors, or a YOLO hand-sign detector, if you
want higher accuracy on more complex signs.

---

## Troubleshooting

- **Camera doesn't start**: check your browser has granted camera
  permission to `localhost:8501`; try a different browser (Chrome/Edge
  recommended for WebRTC).
- **`mediapipe` fails to install**: use Python 3.10 or 3.11 in your venv.
- **No sound from Speak button**: pyttsx3 depends on OS TTS drivers
  (SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on Linux — install
  `espeak` via your package manager on Linux if voices list is empty).
- **Low FPS**: lower `frame_width`/`frame_height` in Settings, or close
  other apps using the camera/CPU.

---

## License

MIT — free to use, modify, and submit as your academic mini project. Please
retain the honesty note about the heuristic lip-reading approach in any
report or viva you give based on this project.
