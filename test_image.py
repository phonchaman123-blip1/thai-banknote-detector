import cv2
import os

from preprocessing import preprocess_frame
from segmentation import segment_hsv, DEFAULT_SETTINGS
from contours import find_banknote_contours
from perspective import warp_banknote


IMAGE_PATH = "test.jpg"
OUTPUT_DIR = "debug_output"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


image = cv2.imread(IMAGE_PATH)

if image is None:
    print(
        f"Cannot open image: {IMAGE_PATH}"
    )
    exit()


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

frame, gray, blur, equalized, hsv = preprocess_frame(
    image
)

cv2.imwrite(
    f"{OUTPUT_DIR}/01_original.jpg",
    frame
)

cv2.imwrite(
    f"{OUTPUT_DIR}/02_gray.jpg",
    gray
)

cv2.imwrite(
    f"{OUTPUT_DIR}/03_blur.jpg",
    blur
)

cv2.imwrite(
    f"{OUTPUT_DIR}/04_equalized.jpg",
    equalized
)

cv2.imwrite(
    f"{OUTPUT_DIR}/05_hsv.jpg",
    hsv
)


# --------------------------------------------------
# Segmentation
# --------------------------------------------------

mask = segment_hsv(
    hsv,
    DEFAULT_SETTINGS
)

cv2.imwrite(
    f"{OUTPUT_DIR}/06_mask.jpg",
    mask
)


# --------------------------------------------------
# Contours
# --------------------------------------------------

contours = find_banknote_contours(
    mask
)

contour_image = frame.copy()

for contour in contours:

    x, y, w, h = cv2.boundingRect(
        contour
    )

    cv2.rectangle(
        contour_image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )


cv2.imwrite(
    f"{OUTPUT_DIR}/07_contours.jpg",
    contour_image
)


# --------------------------------------------------
# Perspective
# --------------------------------------------------

if contours:

    warped = warp_banknote(
        frame,
        contours[0]
    )

    if warped is not None:

        cv2.imwrite(
            f"{OUTPUT_DIR}/08_warped.jpg",
            warped
        )

        print(
            "Perspective crop saved."
        )

    else:

        print(
            "Contour rejected by aspect ratio."
        )

else:

    print(
        "No valid contours found."
    )


print(
    f"Debug images saved to: {OUTPUT_DIR}"
)