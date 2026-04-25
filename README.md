# Real-Time Face Recognition System using OpenCV (LBPH)

A production-quality, beginner-friendly face recognition system built with Python and OpenCV. Uses the **Local Binary Patterns Histograms (LBPH)** algorithm for face recognition — no deep learning required.

---

## Project Structure

```
face_recognition_system/
├── dataset/               # Collected face images (auto-created)
├── models/
│   ├── model.yml          # Trained LBPH model
│   └── labels.json        # Label ↔ name mapping
├── src/
│   ├── __init__.py
│   ├── data_collection.py # Webcam face capture
│   ├── train_model.py     # LBPH model training
│   └── recognize.py       # Real-time recognition
├── main.py                # CLI menu entry point
├── requirements.txt
└── README.md
```

---

##  Installation

### Prerequisites
- Python 3.8+
- Webcam

### Steps

```bash
# Navigate to the project directory
cd face_recognition_system

# Install dependencies
pip install -r requirements.txt
```

> **Note:** `opencv-contrib-python` is required for `cv2.face.LBPHFaceRecognizer_create()`.

---

##  How to Run

### Option 1: CLI Menu (Recommended)

```bash
python main.py
```

This opens an interactive menu:
```
  1.  Collect Face Data
  2.  Train Recognition Model
  3.  Run Real-Time Recognition
  4.  Exit
```

### Option 2: Run Modules Individually

```bash
# Step 1 — Collect face images for a person
python src/data_collection.py

# Step 2 — Train the LBPH model
python src/train_model.py

# Step 3 — Run real-time recognition
python src/recognize.py
```

---

##  Workflow

1. **Collect Data** — Point the webcam at the person. The system captures 50 cropped grayscale face images automatically. Press `q` to stop early.

2. **Train Model** — Loads all images from `dataset/`, assigns labels, and trains the LBPH recognizer. Saves `model.yml` and `labels.json`.

3. **Recognize** — Opens the webcam and identifies faces in real-time. Known faces show a green box with name and confidence. Unknown faces show a red box.

---

##  Controls

| Module             | Key   | Action                  |
|--------------------|-------|-------------------------|
| Data Collection    | `q`   | Stop capturing          |
| Recognition        | `ESC` | Quit recognition        |
| Recognition        | `s`   | Save unknown face image |

---

##  Example Output

**Data Collection:**
```
[INFO] Collecting face data for 'Alice'.
[INFO] Saving images to: dataset/Alice
[DONE] Saved 50 face images for 'Alice'
```

**Training:**
```
  Loading  50 images for 'Alice' (label=0)
  Loading  50 images for 'Bob'   (label=1)

[INFO] Total images loaded: 100
[INFO] Training LBPH recognizer...
[SAVED] Model → models/model.yml
[SAVED] Labels → models/labels.json
```

**Recognition:**
```
Live webcam feed with:
  - Green bounding box: "Alice (87%)"
  - Red bounding box: "Unknown (120)"
  - FPS counter in top-left corner
```

---

##  How LBPH Works

LBPH (Local Binary Patterns Histograms) works by:

1. Dividing the face image into a grid of cells
2. Computing a Local Binary Pattern for each pixel by comparing it with neighbors
3. Building a histogram of patterns for each cell
4. Concatenating all histograms into a single feature vector
5. Comparing feature vectors using Chi-Square distance

**Confidence Score:** Lower distance = better match. A threshold (default: 80) separates known from unknown faces.

---

##  Limitations

- **Lighting Sensitivity** — Performance degrades in poor or uneven lighting
- **Pose Variation** — LBPH works best with frontal faces; side profiles may fail
- **Scale Sensitivity** — Very far or very close faces may not be detected
- **Single Cascade** — Only frontal face Haar Cascade is used
- **No Liveness Detection** — Cannot distinguish photos from real faces
- **Limited Accuracy** — LBPH is less accurate than deep learning methods for large datasets

---

##  Future Improvements

- Replace LBPH with deep learning models (FaceNet, ArcFace, dlib)
- Add face alignment preprocessing (eye detection + affine transform)
- Implement anti-spoofing / liveness detection
- Add a web-based interface (Flask / Streamlit)
- Support video file input (not just live webcam)
- Add database storage for face embeddings
- Implement face tracking to avoid re-detection every frame

---

##  License

This project is for educational purposes. Feel free to use and modify.
