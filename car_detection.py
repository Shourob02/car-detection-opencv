# ============================================================
#  Car Detection using Haar Cascade Classifier
#  Project: Car Detection and Counting using OpenCV
# ============================================================

import cv2
import sys

# ---------------------
# Configuration
# ---------------------
VIDEO_PATH      = 'cars.mp4'
CASCADE_PATH    = 'haarcascade_cars.xml'
SCALE_FACTOR    = 1.1
MIN_NEIGHBORS   = 3
RECT_COLOR      = (0, 255, 0)   # Green bounding boxes
RECT_THICKNESS  = 2
TEXT_COLOR      = (0, 200, 255) # Cyan overlay text

# ---------------------
# Load resources
# ---------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

car_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if car_cascade.empty():
    print(f"[ERROR] Cannot load cascade classifier: {CASCADE_PATH}")
    cap.release()
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25  # fallback to 25 if unreadable

print("[INFO] Starting car detection. Press 'Q' to quit.")

# ---------------------
# Main processing loop
# ---------------------
while True:
    ret, frame = cap.read()

    # End of video or read error
    if not ret or frame is None:
        print("[INFO] End of video or failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars
    cars = car_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS
    )

    # Draw bounding boxes around each detected car
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), RECT_COLOR, RECT_THICKNESS)

    # Overlay: car count and FPS
    car_count = len(cars)
    cv2.putText(
        frame,
        f"Cars Detected: {car_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2
    )

    cv2.imshow('Car Detection', frame)

    # Press 'Q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print("[INFO] User requested exit.")
        break

# ---------------------
# Cleanup
# ---------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection finished.")
