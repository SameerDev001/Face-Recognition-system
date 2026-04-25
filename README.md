# Simple Face Recognition System

This is a Python project that can recognize faces in real-time using a webcam. It uses the LBPH (Local Binary Patterns Histograms) algorithm from OpenCV. It's a classic way to do face recognition without needing heavy deep learning models.

## How to use it
The easiest way to run the project is using the main menu:

1. **Install requirements**:
   Run `pip install -r requirements.txt` or just install `opencv-contrib-python` and `numpy`.
   
2. **Run the program**:
   ```bash
   python main.py
   ```

3. **Follow the steps in the menu**:
   * **Step 1: Collect Data** - Type in your name and look at the camera. It will take about 50 photos of your face.
   * **Step 2: Train Model** - This button takes the photos you just took and trains the recognition model. It saves everything to the `models/` folder.
   * **Step 3: Run Recognition** - This opens the camera and tries to identify you based on the trained model.

## Controls
* While capturing or recognizing:
    * Press **q** to quit data collection.
    * Press **ESC** to stop the recognition window.
    * Press **s** (during recognition) to save a picture of an "unknown" person if the system doesn't recognize them.
