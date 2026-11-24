# Thai Banknote Detector

Real-time Thai banknote detection and recognition using **Python + OpenCV** with classical image processing (no deep learning / no model training).

## How it works

1. **Preprocessing** – resize and blur the camera frame
2. **HSV segmentation** – separate the banknote from the background (tunable with trackbars)
3. **Contour detection** – find the banknote outline
4. **Perspective transform** – warp the banknote to a flat, top-down view
5. **Classification** – take the median HSV color of the center area and compare it with saved samples (weighted HSV distance, hue handled as circular)
6. **Display** – show the detected value and total on screen
7. **Voice feedback** – speaks the result with `pyttsx3`

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Main loop: camera → segmentation → contours → warp → classify |
| `preprocessing.py` | Frame preprocessing and status text |
| `segmentation.py` | HSV mask + trackbars |
| `contours.py` | Banknote contour detection |
| `perspective.py` | Perspective warp |
| `classifier.py` | HSV sampling, learning, and classification |
| `voice_feedback.py` | Text-to-speech |
| `phone_camera.py` | Read video stream from a phone camera |
| `tune_hsv.py`, `camera_test.py`, `voice_test.py`, `test_image.py` | Testing / tuning tools |

## Installation

```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

**Camera:** `main.py` reads from a phone camera stream (`CAMERA_URL`). Change it to your own phone's IP, or use a webcam by setting `CAMERA_INDEX = 0`.

**Keys**

| Key | Action |
|---|---|
| `1` | Save the current banknote as a sample (100 Baht) |
| `q` | Quit |

Samples are saved to `hsv_samples.json`, which is created automatically the first time you save a sample.

## Tech stack

Python · OpenCV · NumPy · pyttsx3 · requests
