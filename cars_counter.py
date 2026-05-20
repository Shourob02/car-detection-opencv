# ============================================================
#  Car Counter using Frame Differencing
#  Project: Car Detection and Counting using OpenCV
# ============================================================

import cv2
import numpy as np
import sys

# ---------------------
# Configuration
# ---------------------
VIDEO_PATH          = 'cars.mp4'
MIN_CONTOUR_WIDTH   = 40
MIN_CONTOUR_HEIGHT  = 40
OFFSET              = 10        # Pixel tolerance around the counting line
LINE_HEIGHT         = 550       # Y-position of the virtual counting line
DISPLAY_WIDTH       = 1200      # Width used when drawing the counting line
BOX_COLOR           = (255, 0, 0)     # Blue bounding boxes
LINE_COLOR          = (0, 255, 0)     # Green counting line
CENTROID_COLOR      = (0, 255, 0)     # Green centroid dots
TEXT_COLOR          = (0, 170, 0)     # Green counter text

# ---------------------
# Helper: compute centroid
# ---------------------
def get_centroid(x, y, w, h):
    """Return the centre point of a bounding rectangle."""
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

# ---------------------
# Open video
# ---------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

# Optional: set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# ---------------------
# Initialise state
# ---------------------
matches     = []   # Centroids of active contours
cars        = 0    # Running vehicle count

ret, frame1 = cap.read()
ret, frame2 = cap.read()

if frame1 is None or frame2 is None:
    print("[ERROR] Could not read initial frames from video.")
    cap.release()
    sys.exit(1)

print("[INFO] Starting car counter. Press 'ESC' to quit.")

# ---------------------
# Main processing loop
# ---------------------
while ret:
    # --- Frame differencing pipeline ---
    diff    = cv2.absdiff(frame1, frame2)
    grey    = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur    = cv2.GaussianBlur(grey, (5, 5), 0)
    _, th   = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # --- Find contours ---
    contours, _ = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the virtual counting line
    cv2.line(frame1, (0, LINE_HEIGHT), (DISPLAY_WIDTH, LINE_HEIGHT), LINE_COLOR, 2)

    for contour in contours:
        (bx, by, bw, bh) = cv2.boundingRect(contour)

        # Filter out noise based on minimum size
        if bw < MIN_CONTOUR_WIDTH or bh < MIN_CONTOUR_HEIGHT:
            continue

        # Draw bounding box
        cv2.rectangle(
            frame1,
            (bx - 10, by - 10),
            (bx + bw + 10, by + bh + 10),
            BOX_COLOR, 2
        )

        # Compute and draw centroid
        centroid = get_centroid(bx, by, bw, bh)
        matches.append(centroid)
        cv2.circle(frame1, centroid, 5, CENTROID_COLOR, -1)

    # --- Check which centroids cross the counting line ---
    # Iterate over a copy so we can safely remove while iterating
    for (mx, my) in matches[:]:
        if (LINE_HEIGHT - OFFSET) < my < (LINE_HEIGHT + OFFSET):
            cars += 1
            matches.remove((mx, my))
            print(f"[COUNT] Vehicle #{cars} detected.")

    # --- Overlay counter text ---
    cv2.putText(
        frame1,
        f"Total Vehicles Detected: {cars}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        TEXT_COLOR,
        2
    )

    cv2.imshow("Car Counter", frame1)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        print("[INFO] User requested exit.")
        break

    # Advance frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Guard against end-of-video None frame
    if frame2 is None:
        break

# ---------------------
# Final report & cleanup
# ---------------------
print(f"[RESULT] Total vehicles counted: {cars}")
cv2.destroyAllWindows()
cap.release()
