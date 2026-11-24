import cv2


MIN_AREA = 2000


def find_banknote_contours(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area >= MIN_AREA:
            candidates.append(contour)

    # เรียงจากพื้นที่ใหญ่ไปเล็ก
    candidates.sort(
        key=cv2.contourArea,
        reverse=True
    )

    return candidates


def draw_contour_candidates(
    frame,
    contours
):

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Area: {int(area)}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )