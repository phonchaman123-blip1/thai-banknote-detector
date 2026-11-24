import cv2


def scan_cameras(max_cameras=10):

    print("Scanning cameras...")

    for index in range(max_cameras):

        print(f"Testing camera index {index}...")

        # Windows: DirectShow
        cap = cv2.VideoCapture(
            index,
            cv2.CAP_DSHOW
        )

        if not cap.isOpened():
            print(f"[--] Camera {index} unavailable")
            cap.release()
            continue

        ret, frame = cap.read()

        if ret and frame is not None:

            print(f"[FOUND] Camera index: {index}")

            cv2.imshow(
                f"Camera {index}",
                frame
            )

            print("Press any key to continue...")
            cv2.waitKey(1000)

        else:
            print(f"[--] Camera {index} cannot read frame")

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    scan_cameras()