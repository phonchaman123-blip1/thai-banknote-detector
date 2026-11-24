import cv2
import numpy as np


DEFAULT_SETTINGS = {
    "s_min": 20,
    "s_max": 255,
    "v_min": 30,
    "v_max": 255,
}


def create_trackbars(window="HSV Calibration"):

    cv2.namedWindow(window)

    cv2.createTrackbar(
        "S Min",
        window,
        DEFAULT_SETTINGS["s_min"],
        255,
        lambda x: None
    )

    cv2.createTrackbar(
        "S Max",
        window,
        DEFAULT_SETTINGS["s_max"],
        255,
        lambda x: None
    )

    cv2.createTrackbar(
        "V Min",
        window,
        DEFAULT_SETTINGS["v_min"],
        255,
        lambda x: None
    )

    cv2.createTrackbar(
        "V Max",
        window,
        DEFAULT_SETTINGS["v_max"],
        255,
        lambda x: None
    )


def get_settings(window="HSV Calibration"):

    return {
        "s_min": cv2.getTrackbarPos(
            "S Min",
            window
        ),

        "s_max": cv2.getTrackbarPos(
            "S Max",
            window
        ),

        "v_min": cv2.getTrackbarPos(
            "V Min",
            window
        ),

        "v_max": cv2.getTrackbarPos(
            "V Max",
            window
        ),
    }


def segment_hsv(hsv, settings):

    lower = np.array([
        0,
        settings["s_min"],
        settings["v_min"]
    ])

    upper = np.array([
        179,
        settings["s_max"],
        settings["v_max"]
    ])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # ลบ noise ขนาดเล็ก
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # เติมรูเล็ก ๆ ภายในวัตถุ
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask