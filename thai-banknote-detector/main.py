import cv2
import time

from preprocessing import (
    preprocess_frame,
    draw_status
)

from segmentation import (
    segment_hsv,
    create_trackbars,
    get_settings
)

from contours import (
    find_banknote_contours
)

from perspective import (
    warp_banknote,
    draw_perspective_rectangle
)

from classifier import (
    sample_hsv,
    classify_hsv,
    load_samples,
    value_for_label,
    draw_classification
)


#CAMERA_INDEX = 0
CAMERA_URL = "http://172.20.10.3:8081/video"

def main():

    # ==========================================
    # 1. OPEN CAMERA
    # ==========================================

    cap = cv2.VideoCapture(
        #CAMERA_INDEX,
        #cv2.CAP_DSHOW
        CAMERA_URL

    )

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        960
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        540
    )

    print("Camera opened successfully!")
    print("Press Q to quit.")

    # ==========================================
    # 2. HSV TRACKBARS
    # ==========================================

    create_trackbars()

    # Load trained samples
    samples = load_samples()

    # FPS
    previous_time = time.time()

    # ==========================================
    # 3. MAIN CAMERA LOOP
    # ==========================================

    while True:

        # Read frame from camera
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Cannot read frame")
            break

        # ======================================
        # PREPROCESSING
        # ======================================

        frame, gray, blur, equalized, hsv = preprocess_frame(
            frame
        )

        # ======================================
        # SEGMENTATION
        # ======================================

        settings = get_settings()

        mask = segment_hsv(
            hsv,
            settings
        )

        # ======================================
        # CONTOUR DETECTION
        # ======================================

        contours = find_banknote_contours(
            mask
        )

        detected_label = "NONE"
        total_baht = 0

        # ======================================
        # BANKNOTE DETECTION
        # ======================================

        for contour in contours:

            # Check perspective / aspect ratio
            warped = warp_banknote(
                frame,
                contour
            )

            # Not a banknote shape
            if warped is None:
                continue

            # Draw rotated rectangle
            draw_perspective_rectangle(
                frame,
                contour
            )

            # ==================================
            # CLASSIFICATION
            # ==================================

            hsv_value = sample_hsv(
                warped
            )

            label, distance = classify_hsv(
                hsv_value,
                samples
            )

            detected_label = label

            if label != "UNKNOWN":

                total_baht = value_for_label(
                    label
                )

            # Use only the largest valid note
            break

        # ======================================
        # FPS
        # ======================================

        current_time = time.time()

        fps = 1.0 / max(
            current_time - previous_time,
            0.0001
        )

        previous_time = current_time

        # ======================================
        # DISPLAY
        # ======================================

        draw_status(
            frame,
            fps
        )

        draw_classification(
            frame,
            detected_label,
            total_baht
        )

        # Main camera window
        cv2.imshow(
            "Banknote Detection",
            frame
        )

        # Segmentation window
        cv2.imshow(
            "HSV Mask",
            mask
        )

        # ======================================
        # KEYBOARD
        # ======================================

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # ======================================
        # SAVE 100 BAHT SAMPLE
        # Press 1
        # ======================================

        if key == ord("1") and contours:

            warped = warp_banknote(
                frame,
                contours[0]
            )

            if warped is not None:

                from classifier import add_sample

                add_sample(
                    "100",
                    warped
                )

                samples = load_samples()

                print(
                    "100 Baht HSV sample saved!"
                )

    # ==========================================
    # 4. CLOSE CAMERA
    # ==========================================

    cap.release()

    cv2.destroyAllWindows()

    print("Camera closed.")


# ==============================================
# PROGRAM START
# ==============================================

if __name__ == "__main__":
    main()