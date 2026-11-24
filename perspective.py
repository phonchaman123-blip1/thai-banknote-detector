import cv2
import numpy as np


MIN_RATIO = 1.6
MAX_RATIO = 3.4


def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(s)]       # top-left
    rect[2] = points[np.argmax(s)]       # bottom-right
    rect[1] = points[np.argmin(diff)]    # top-right
    rect[3] = points[np.argmax(diff)]    # bottom-left

    return rect


def crop_banknote(contour):
    rect = cv2.minAreaRect(contour)

    (center, (width, height), angle) = rect

    if width <= 0 or height <= 0:
        return None, None

    ratio = max(width, height) / min(width, height)

    # Reject shapes that don't look like a banknote
    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return None, None

    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    box = order_points(box)

    # Force landscape
    output_width = 600
    output_height = 300

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        box,
        destination
    )

    warped = cv2.warpPerspective(
        cv2.imread("__dummy__.jpg") if False else None,
        matrix,
        (output_width, output_height)
    )

    return matrix, box


def warp_banknote(frame, contour):
    rect = cv2.minAreaRect(contour)

    (center, (width, height), angle) = rect

    if width <= 0 or height <= 0:
        return None

    ratio = max(width, height) / min(width, height)

    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return None

    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    box = order_points(box)

    output_width = 600
    output_height = 300

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        box,
        destination
    )

    warped = cv2.warpPerspective(
        frame,
        matrix,
        (output_width, output_height)
    )

    return warped


def draw_perspective_rectangle(frame, contour):
    rect = cv2.minAreaRect(contour)

    width, height = rect[1]

    if min(width, height) <= 0:
        return

    ratio = max(width, height) / min(width, height)

    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        return

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.polylines(
        frame,
        [box],
        True,
        (0, 255, 0),
        3
    )