"""
data_collection.py — Face Data Collection Module
==================================================
Captures face images from the webcam for a given user and stores them
as cropped, grayscale images inside  dataset/<person_name>/.

Usage (standalone):
    python src/data_collection.py

Workflow:
    1. Prompt user for their name.
    2. Open webcam feed.
    3. Detect faces using Haar Cascade.
    4. Crop, convert to grayscale, and save each detected face.
    5. Stop after MAX_SAMPLES images or when 'q' is pressed.
"""

import os
import cv2
import time

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
# Number of face samples to collect per person
MAX_SAMPLES = 50

# Minimum face size in pixels (filters out tiny false positives)
MIN_FACE_SIZE = (100, 100)

# Output size for each saved face image
FACE_SIZE = (200, 200)

# Base directory for all collected face datasets
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# Path to Haar Cascade — use OpenCV's bundled copy
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")


def create_dataset_directory(person_name: str) -> str:
    """
    Create (or reuse) a directory for the given person under dataset/.

    Args:
        person_name: Name of the person whose face data is being collected.

    Returns:
        Absolute path to the person's dataset directory.
    """
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def load_face_detector() -> cv2.CascadeClassifier:
    """
    Load the Haar Cascade classifier for frontal face detection.

    Returns:
        A ready-to-use CascadeClassifier object.

    Raises:
        IOError: If the cascade XML file cannot be loaded.
    """
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise IOError(
            f"Failed to load Haar Cascade from: {CASCADE_PATH}\n"
            "Make sure opencv-python is installed correctly."
        )
    return detector


def collect_faces(person_name: str, max_samples: int = MAX_SAMPLES) -> int:
    """
    Open the webcam, detect faces, and save cropped grayscale images.

    Args:
        person_name:  Name label for the person.
        max_samples:  Maximum number of face images to capture.

    Returns:
        The total number of face images saved.
    """
    save_dir = create_dataset_directory(person_name)
    detector = load_face_detector()

    # Determine the starting index so we don't overwrite existing images
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    start_index = len(existing)

    # Open webcam (index 0 = default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam. Is it connected?")
        return 0

    print(f"\n[INFO] Collecting face data for '{person_name}'.")
    print(f"[INFO] Saving images to: {save_dir}")
    print(f"[INFO] Will capture up to {max_samples} samples.")
    print("[INFO] Press 'q' to quit early.\n")

    count = 0
    prev_time = time.time()

    while count < max_samples:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame — retrying...")
            continue

        # Flip for mirror effect (more natural for the user)
        frame = cv2.flip(frame, 1)

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=MIN_FACE_SIZE,
        )

        for (x, y, w, h) in faces:
            # Crop and resize the face region
            face_roi = gray[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi, FACE_SIZE)

            # Save the face image
            img_name = f"face_{start_index + count:04d}.jpg"
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, face_resized)
            count += 1

            # Draw bounding box + counter on live feed
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Sample {count}/{max_samples}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if count >= max_samples:
                break

        # ---- HUD: FPS display ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Show instruction
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Face Data Collection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] 'q' pressed — stopping collection.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Saved {count} face images for '{person_name}' → {save_dir}")
    return count


# ---------------------------------------------------------------------------
#  Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    name = input("Enter the person's name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
    else:
        collect_faces(name)
