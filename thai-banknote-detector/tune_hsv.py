import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(
    0,
    cv2.CAP_AVFOUNDATION
)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


cv2.namedWindow("HSV Tune")

cv2.createTrackbar(
    "S Min",
    "HSV Tune",
    30,
    255,
    nothing
)

cv2.createTrackbar(
    "S Max",
    "HSV Tune",
    255,
    255,
    nothing
)

cv2.createTrackbar(
    "V Min",
    "HSV Tune",
    40,
    255,
    nothing
)

cv2.createTrackbar(
    "V Max",
    "HSV Tune",
    255,
    255,
    nothing
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (960, 540)
    )

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    s_min = cv2.getTrackbarPos(
        "S Min",
        "HSV Tune"
    )

    s_max = cv2.getTrackbarPos(
        "S Max",
        "HSV Tune"
    )

    v_min = cv2.getTrackbarPos(
        "V Min",
        "HSV Tune"
    )

    v_max = cv2.getTrackbarPos(
        "V Max",
        "HSV Tune"
    )

    lower = np.array(
        [0, s_min, v_min]
    )

    upper = np.array(
        [179, s_max, v_max]
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    cv2.imshow(
        "Original",
        frame
    )

    cv2.imshow(
        "HSV Mask",
        mask
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()