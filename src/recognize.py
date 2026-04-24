"""
recognize.py — Real-Time Face Recognition Module
==================================================
Loads a pre-trained LBPH model and performs live face recognition
through the webcam feed.

Usage (standalone):
    python src/recognize.py

Controls:
    ESC   — Quit the application
    's'   — Save current frame of an unknown face to dataset/unknown/
"""

import os
import json
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
# Fixed face size (must match training dimensions)
FACE_SIZE = (200, 200)

# Confidence threshold — lower LBPH distance = better match
# Faces with confidence BELOW this value are considered "known"
CONFIDENCE_THRESHOLD = 80

# Minimum face size to avoid false positives from noise
MIN_FACE_SIZE = (100, 100)

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.yml")
LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "labels.json")
UNKNOWN_DIR = os.path.join(PROJECT_ROOT, "dataset", "unknown")

# Haar Cascade
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# Color palette for bounding boxes (BGR)
COLOR_KNOWN = (0, 255, 0)       # Green
COLOR_UNKNOWN = (0, 0, 255)     # Red
COLOR_TEXT = (255, 255, 255)     # White
COLOR_FPS = (255, 255, 0)       # Cyan


def load_model_and_labels() -> tuple:
    """
    Load the trained LBPH model and the label-name mapping from disk.

    Returns:
        recognizer  — Trained LBPHFaceRecognizer instance.
        label_map   — Dict mapping str(label_id) → person_name.

    Raises:
        FileNotFoundError: If model or labels file is missing.
    """
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}\n"
            "Run train_model.py first."
        )
    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(
            f"Label mapping not found at: {LABELS_PATH}\n"
            "Run train_model.py first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    print(f"[INFO] Model loaded from: {MODEL_PATH}")

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    print(f"[INFO] Labels loaded: {label_map}")

    return recognizer, label_map


def load_face_detector() -> cv2.CascadeClassifier:
    """
    Load Haar Cascade for frontal face detection.

    Returns:
        CascadeClassifier ready for use.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise IOError(f"Failed to load Haar Cascade from: {CASCADE_PATH}")
    return detector


def save_unknown_face(frame: np.ndarray, face_roi: np.ndarray) -> str:
    """
    Save a snapshot of an unknown face to dataset/unknown/ for later labeling.

    Args:
        frame:    The full BGR frame (saved for context).
        face_roi: The cropped grayscale face region.

    Returns:
        Path to the saved image.
    """
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    timestamp = int(time.time() * 1000)
    path = os.path.join(UNKNOWN_DIR, f"unknown_{timestamp}.jpg")
    cv2.imwrite(path, face_roi)
    return path


def run_recognition() -> None:
    """
    Main recognition loop — captures webcam frames, detects and identifies faces.
    """
    recognizer, label_map = load_model_and_labels()
    detector = load_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam.")
        return

    print("\n[INFO] Starting real-time face recognition...")
    print("[INFO] Press ESC to quit | 's' to save unknown face\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame grab failed — retrying...")
            continue

        # Mirror the frame for a natural view
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for detection + recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
        )

        for (x, y, w, h) in faces:
            # Extract and resize face ROI
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, FACE_SIZE)

            # Predict identity
            label_id, confidence = recognizer.predict(face_resized)

            # LBPH confidence = distance (lower is better)
            if confidence < CONFIDENCE_THRESHOLD:
                name = label_map.get(str(label_id), "Unknown")
                box_color = COLOR_KNOWN
                # Convert distance to a "match %" for display
                match_pct = max(0, 100 - confidence)
                display_text = f"{name} ({match_pct:.0f}%)"
            else:
                name = "Unknown"
                box_color = COLOR_UNKNOWN
                display_text = f"Unknown ({confidence:.0f})"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw label background for readability
            text_size = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )[0]
            cv2.rectangle(
                frame,
                (x, y - text_size[1] - 14),
                (x + text_size[0] + 6, y),
                box_color,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                display_text,
                (x + 3, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2,
            )

        # ---- HUD: FPS counter ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_FPS,
            2,
        )

        # ---- HUD: Instructions ----
        cv2.putText(
            frame,
            "ESC=Quit | 's'=Save Unknown",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Face Recognition — LBPH", frame)

        # ---- Key handling ----
        key = cv2.waitKey(1) & 0xFF

        # ESC to quit
        if key == 27:
            print("[INFO] ESC pressed — exiting.")
            break

        # 's' to save unknown face snapshot
        if key == ord("s") and len(faces) > 0:
            # Save the last detected face
            lx, ly, lw, lh = faces[-1]
            roi = gray[ly : ly + lh, lx : lx + lw]
            path = save_unknown_face(frame, cv2.resize(roi, FACE_SIZE))
            print(f"[SAVED] Unknown face → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Recognition session ended.")


# ---------------------------------------------------------------------------
#  Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_recognition()
