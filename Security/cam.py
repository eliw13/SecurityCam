"""Utility to record webcam footage when a face or body is detected."""

from __future__ import annotations

import datetime
import time
from pathlib import Path

import cv2


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
BODY_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

SECONDS_TO_RECORD_AFTER_DETECTION = 5
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def _new_writer(frame_size: tuple[int, int]) -> cv2.VideoWriter:
    """Return a ``cv2.VideoWriter`` with a timestamped filename."""

    timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    filename = Path(f"{timestamp}.mp4")
    return cv2.VideoWriter(str(filename), FOURCC, 20, frame_size)


def main() -> None:
    cap = cv2.VideoCapture(0)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    recording = False
    last_detection = None
    out: cv2.VideoWriter | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        bodies = BODY_CASCADE.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0 or len(bodies) > 0:
            if not recording:
                out = _new_writer(frame_size)
                recording = True
                print("Started recording!")
            last_detection = time.time()

        if recording:
            assert out is not None  # for type checkers
            out.write(frame)
            if last_detection and time.time() - last_detection > SECONDS_TO_RECORD_AFTER_DETECTION:
                recording = False
                out.release()
                out = None
                print("Stopped recording!")

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
