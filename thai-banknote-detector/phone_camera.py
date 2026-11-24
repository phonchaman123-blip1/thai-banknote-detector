import cv2
import numpy as np
import requests


URL = "http://172.20.10.3:8081/video"


def phone_camera():

    print("Connecting to iPhone...")
    print(URL)

    response = requests.get(
        URL,
        stream=True,
        timeout=10
    )

    print("HTTP status:", response.status_code)

    if response.status_code != 200:
        print("ERROR: Cannot connect to iPhone stream")
        return

    print("Connected to iPhone!")
    print("Press Q to quit.")

    buffer = b""

    for chunk in response.iter_content(
        chunk_size=4096
    ):

        buffer += chunk

        # Find JPEG start
        start = buffer.find(b"\xff\xd8")

        # Find JPEG end
        end = buffer.find(
            b"\xff\xd9",
            start + 2
        )

        if start != -1 and end != -1:

            jpg = buffer[
                start:end + 2
            ]

            buffer = buffer[
                end + 2:
            ]

            # Convert JPEG → OpenCV image
            frame = cv2.imdecode(
                np.frombuffer(
                    jpg,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            frame = cv2.resize(
                frame,
                (960, 540)
            )

            cv2.imshow(
                "iPhone Camera",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    response.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    phone_camera()