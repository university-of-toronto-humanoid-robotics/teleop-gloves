import cv2
import time
import math
import urllib.request
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------
# 1. Download Holistic model
# ------------------------------------------------------------------
MODEL_PATH = "holistic_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "holistic_landmarker/holistic_landmarker/float16/1/"
             "holistic_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("Downloading holistic_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ------------------------------------------------------------------
# 2. Landmarker options
# ------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5
)

# ------------------------------------------------------------------
# 3. Angle helpers (15 joint angles)
# ------------------------------------------------------------------
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),        # thumb CMC, MCP, IP
    (0, 5, 6), (5, 6, 7), (6, 7, 8),        # index MCP, PIP, DIP
    (0, 9, 10), (9, 10, 11), (10, 11, 12),  # middle MCP, PIP, DIP
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# ring MCP, PIP, DIP
    (0, 17, 18), (17, 18, 19), (18, 19, 20),# pinky MCP, PIP, DIP
]

def angle_at(a, b, c):
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag == 0:
        return 0.0
    cos = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos))

def hand_angles(landmarks, w, h):
    pts = [(lm.x * w, lm.y * h) for lm in landmarks]
    return np.array([angle_at(pts[a], pts[b], pts[c])
                     for (a, b, c) in ANGLE_TRIPLETS], dtype=np.float32)

def flat_landmarks(maybe_nested):
    if not maybe_nested:
        return None
    first = maybe_nested[0]
    if isinstance(first, (list, tuple)):
        return first
    return maybe_nested

# ------------------------------------------------------------------
# 4. 2D simulated robotic hand (15 DOF)
# ------------------------------------------------------------------
def simulate_hand(angles, size=(450, 450), bend_sign=1):
    """
    Draw a 2D robotic hand from the 15 joint angles.
    bend = 180 - angle  →  straight = no bend, curled = large bend.
    """
    W, H = size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cx = W // 2
    palm_w, palm_h = 150, 120
    px1, px2 = cx - palm_w // 2, cx + palm_w // 2
    py1, py2 = H - 70 - palm_h, H - 70

    # Palm
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (70, 70, 90), -1)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (210, 210, 210), 2)
    cv2.putText(canvas, "SIMULATED HAND", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # name, base_x, base_y, initial_direction_deg, seg_lengths, angle_index
    fingers = [
        ('THU', px1 + 15, py2 - 45, -145, [40, 32, 26], 0),
        ('IND', cx - 47,   py1 + 5,  -95, [48, 32, 24], 3),
        ('MID', cx - 16,   py1,      -90, [52, 34, 26], 6),
        ('RIN', cx + 16,   py1 + 5,  -85, [48, 32, 24], 9),
        ('PIN', cx + 47,   py1 + 15, -78, [38, 26, 20], 12),
    ]

    if len(angles) < 15:
        return canvas

    for name, bx, by, base_angle_deg, seg_lengths, idx in fingers:
        # Clamp angles into sane range
        a = [max(0.0, min(180.0, float(angles[idx + k]))) for k in range(3)]
        bends = [(180 - a[k]) * bend_sign for k in range(3)]

        current_angle_deg = base_angle_deg
        pts = [(int(bx), int(by))]
        cur_x, cur_y = float(bx), float(by)

        for seg_len, bend in zip(seg_lengths, bends):
            current_angle_deg += bend
            rad = math.radians(current_angle_deg)
            cur_x += seg_len * math.cos(rad)
            cur_y += seg_len * math.sin(rad)
            pts.append((int(cur_x), int(cur_y)))

        # Segments
        for k in range(len(pts) - 1):
            cv2.line(canvas, pts[k], pts[k + 1], (0, 255, 0), 4)
        # Joints
        for p in pts:
            cv2.circle(canvas, p, 5, (0, 200, 255), -1)
        # Label
        cv2.putText(canvas, name, (pts[0][0] - 18, pts[0][1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Per-finger angle values under the fingertip
        for k in range(3):
            cv2.putText(canvas, f"{int(a[k])}",
                        (pts[k][0] + 6, pts[k][1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 220, 150), 1)

    return canvas

# ------------------------------------------------------------------
# 5. Main loop
# ------------------------------------------------------------------
# Flip to -1 if the simulated fingers curl the wrong way
BEND_SIGN = 1

with HolisticLandmarker.create_from_options(options) as landmarker:
    HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    previousTime = 0

    # Placeholder sim window so it exists before a hand shows up
    cv2.imshow("Simulated Hand (15 DOF)", simulate_hand([180] * 15))
    cv2.moveWindow("Simulated Hand (15 DOF)", 820, 50)

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        H, W = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        right_lms = flat_landmarks(results.right_hand_landmarks)
        left_lms = flat_landmarks(results.left_hand_landmarks)

        # Draw detected hands on webcam view
        if right_lms:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, right_lms, HAND_CONNECTIONS)
        if left_lms:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, left_lms, HAND_CONNECTIONS)

        # Pick ONE hand for the simulation (prefer right)
        tracked_angles = None
        tracked_label = ""
        if right_lms:
            tracked_angles = hand_angles(right_lms, W, H)
            tracked_label = "RIGHT"
        elif left_lms:
            tracked_angles = hand_angles(left_lms, W, H)
            tracked_label = "LEFT"

        # FPS
        currentTime = time.time()
        if previousTime != 0:
            fps = 1 / (currentTime - previousTime)
            cv2.putText(frame, f"{int(fps)} FPS", (10, H - 15),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        previousTime = currentTime

        # Info overlay + sim render
        if tracked_angles is not None:
            cv2.putText(frame, f"Tracking: {tracked_label}",
                        (W - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Print the array (rounded)
            print(f"{tracked_label}:", np.round(tracked_angles, 1).tolist())
            sim_img = simulate_hand(tracked_angles, bend_sign=BEND_SIGN)
            cv2.imshow("Simulated Hand (15 DOF)", sim_img)
        else:
            cv2.putText(frame, "No hand detected", (W - 220, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Hand Landmarks (Tasks API)", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()