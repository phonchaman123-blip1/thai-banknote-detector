import cv2
import numpy as np
import os
import json


SAMPLE_FILE = "hsv_samples.json"

LEARN_KEYS = {
    "100": 100,
}


def sample_hsv(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h, w = hsv.shape[:2]

    # Center region
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    center = hsv[y1:y2, x1:x2]

    median = np.median(
        center.reshape(-1, 3),
        axis=0
    )

    return median.astype(float)


def load_samples():
    if not os.path.exists(SAMPLE_FILE):
        return {}

    with open(SAMPLE_FILE, "r") as f:
        return json.load(f)


def save_samples(samples):
    with open(SAMPLE_FILE, "w") as f:
        json.dump(samples, f, indent=2)


def add_sample(label, crop):
    hsv_value = sample_hsv(crop)

    samples = load_samples()

    if label not in samples:
        samples[label] = []

    samples[label].append(
        hsv_value.tolist()
    )

    save_samples(samples)


def hsv_distance(a, b):
    # HSV weights
    hue_weight = 1.0
    sat_weight = 0.7
    value_weight = 0.5

    # Hue is circular (0-179)
    hue_diff = abs(a[0] - b[0])

    if hue_diff > 90:
        hue_diff = 180 - hue_diff

    sat_diff = abs(a[1] - b[1])
    value_diff = abs(a[2] - b[2])

    distance = (
        hue_weight * hue_diff +
        sat_weight * sat_diff +
        value_weight * value_diff
    )

    return distance


def classify_hsv(hsv_value, samples):
    if not samples:
        return "UNKNOWN", None

    best_label = "UNKNOWN"
    best_distance = float("inf")

    for label, values in samples.items():

        for sample in values:
            sample = np.array(sample, dtype=float)

            distance = hsv_distance(
                hsv_value,
                sample
            )

            if distance < best_distance:
                best_distance = distance
                best_label = label

    return best_label, best_distance


def value_for_label(label):
    try:
        return int(label)
    except ValueError:
        return 0


def draw_classification(frame, label, total):
    cv2.putText(
        frame,
        f"Detected: {label} Baht",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Total: {total} Baht",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )