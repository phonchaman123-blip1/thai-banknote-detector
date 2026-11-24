import cv2


WIDTH = 960
HEIGHT = 540


def preprocess_frame(frame):
    # Resize
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Histogram Equalization
    equalized = cv2.equalizeHist(blur)

    # BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    return frame, gray, blur, equalized, hsv


def draw_status(frame, fps):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )