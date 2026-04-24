"""
train_model.py — LBPH Model Training Module
=============================================
Traverses the dataset/ directory, loads face images, assigns numeric
labels, and trains an LBPH face recognizer. The trained model and the
label mapping are saved to the models/ directory.

Usage (standalone):
    python src/train_model.py

Outputs:
    models/model.yml    — Trained LBPH recognizer
    models/labels.json  — Mapping of numeric label → person name
"""

import os
import json
import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
# Fixed face dimensions expected by the recognizer
FACE_SIZE = (200, 200)

# Project root (one level above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")

# Haar Cascade for optional face re-detection during training
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


def load_dataset() -> tuple:
    """
    Walk through dataset/ and load all face images with their labels.

    Directory layout expected:
        dataset/
        ├── Alice/
        │   ├── face_0000.jpg
        │   └── ...
        └── Bob/
            ├── face_0000.jpg
            └── ...

    Returns:
        faces       — List of numpy arrays (grayscale face images).
        labels      — List of integer labels corresponding to each face.
        label_map   — Dict mapping label_id (int) → person_name (str).
    """
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}\n"
            "Run data_collection.py first to create training data."
        )

    person_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    if not person_dirs:
        raise ValueError(
            "No person directories found inside dataset/.\n"
            "Run data_collection.py to collect face samples first."
        )

    print(f"[INFO] Found {len(person_dirs)} person(s) in dataset.\n")

    for person_name in person_dirs:
        person_path = os.path.join(DATASET_DIR, person_name)
        label_map[current_label] = person_name

        image_files = sorted([
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not image_files:
            print(f"  [WARNING] No images found for '{person_name}' — skipping.")
            continue

        print(f"  Loading {len(image_files):>3} images for '{person_name}' (label={current_label})")

        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"    [WARNING] Could not read: {img_path}")
                continue

            # Resize to fixed dimensions for consistency
            img_resized = cv2.resize(img, FACE_SIZE)
            faces.append(img_resized)
            labels.append(current_label)

        current_label += 1

    return faces, labels, label_map


def train_model(faces: list, labels: list) -> cv2.face.LBPHFaceRecognizer:
    """
    Create and train an LBPH face recognizer.

    Args:
        faces:  List of grayscale face images (numpy arrays).
        labels: List of integer labels.

    Returns:
        Trained LBPHFaceRecognizer instance.
    """
    print("\n[INFO] Training LBPH recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )
    recognizer.train(faces, np.array(labels))
    print("[INFO] Training complete.")
    return recognizer


def save_model(recognizer: cv2.face.LBPHFaceRecognizer, label_map: dict) -> None:
    """
    Persist the trained model and label mapping to disk.

    Args:
        recognizer: Trained LBPHFaceRecognizer.
        label_map:  Dict mapping label_id → person_name.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model weights
    recognizer.save(MODEL_PATH)
    print(f"[SAVED] Model → {MODEL_PATH}")

    # Save label mapping as JSON
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)
    print(f"[SAVED] Labels → {LABELS_PATH}")


def run_training() -> None:
    """
    Full training pipeline: load data → train → save.
    """
    print("=" * 55)
    print("  LBPH Face Recognizer — Training Pipeline")
    print("=" * 55)

    faces, labels, label_map = load_dataset()

    if len(faces) == 0:
        print("\n[ERROR] No face images were loaded. Cannot train.")
        return

    print(f"\n[INFO] Total images loaded: {len(faces)}")
    print(f"[INFO] Number of people:    {len(label_map)}")

    recognizer = train_model(faces, labels)
    save_model(recognizer, label_map)

    print("\n" + "=" * 55)
    print("  Training finished successfully!")
    print("=" * 55)
    print("\nLabel mapping:")
    for lid, name in label_map.items():
        count = labels.count(lid)
        print(f"  {lid} → {name}  ({count} samples)")


# ---------------------------------------------------------------------------
#  Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_training()
