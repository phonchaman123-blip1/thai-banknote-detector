import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Camera opened successfully!")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Cannot read frame")
        break

    frame = cv2.resize(
        frame,
        (960, 540)
    )

    cv2.imshow(
        "Camera Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()