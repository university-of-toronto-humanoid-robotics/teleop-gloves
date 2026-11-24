import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import time
import urllib.request
from collections.abc import Sequence
from typing import Protocol, TextIO, cast

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray
from .retarget import (DEFAULT_CAL, DEFAULT_PROFILE, FEATURE_POSES, FINGERS as FINGER_NAMES,
                       Features, Mapper, POSES, Profile, Samples, calibrate, emit, load)

Points = NDArray[np.float64]


class Landmark(Protocol):
    x: float
    y: float
    z: float

# ------------------------------------------------------------------
# 1. Download Holistic model
# ------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "holistic_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "holistic_landmarker/holistic_landmarker/float16/1/"
             "holistic_landmarker.task")
def download_model() -> None:
    if not MODEL_PATH.is_file():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading holistic_landmarker.task ...", file=sys.stderr)
        partial_path = MODEL_PATH.with_suffix(".task.part")
        try:
            urllib.request.urlretrieve(MODEL_URL, partial_path)
            partial_path.replace(MODEL_PATH)
        finally:
            partial_path.unlink(missing_ok=True)
        print("Download complete.", file=sys.stderr)

# ------------------------------------------------------------------
# 2. Landmarker options
# ------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
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

def point_at(points: Points, index: int) -> Points:
    """Indexing one row of an N×3 float array returns a vector, not a scalar."""
    return cast(Points, points[index])


def row_lengths(vectors: Points) -> Points:
    return cast(Points, np.linalg.norm(vectors, axis=1))


def hand_angles(points: Points) -> Points:
    a, b, c = (points.take(indices, axis=0) for indices in zip(*ANGLE_TRIPLETS))
    first, second = a - b, c - b
    cross = row_lengths(np.cross(first, second))
    dot = cast(Points, np.einsum("ij,ij->i", first, second))
    return np.degrees(np.arctan2(cross, dot))

# ------------------------------------------------------------------
# 4. Palm-relative preview of the 3D landmarks
# ------------------------------------------------------------------
SIM_WINDOW = "Hand Motion Preview"
PALM = (0, 1, 5, 9, 13, 17, 0)
FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
           (13, 14, 15, 16), (17, 18, 19, 20))
SMOOTHING_TIME = 0.045  # seconds


def palm_coordinates(landmarks: Sequence[Landmark] | None, *, is_left: bool) -> Points | None:
    """Remove wrist position, rotation, and overall hand scale."""
    if landmarks is None or len(landmarks) != 21:
        return None
    points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=float)
    if not np.isfinite(points).all():
        return None
    points -= point_at(points, 0).copy()

    along = point_at(points, 9).copy()  # wrist to middle MCP
    scale = np.linalg.norm(along)
    if scale < 1e-6:
        return None
    along /= scale
    across = point_at(points, 17) - point_at(points, 5)  # index to pinky
    across -= float(cast(np.float64, np.dot(across, along))) * along
    width = np.linalg.norm(across)
    if width < 1e-6:
        return None
    across /= width
    normal = np.cross(across, along)
    if is_left:
        normal = -normal  # use the same canonical (thumb-left) view for both hands
    return points @ np.column_stack((across, along, normal)) / scale


def project_hand(points: Points) -> Points:
    across, along, depth = (points.take(i, axis=1) for i in range(3))
    # Oblique view: depth remains visible when the thumb crosses the palm
    # or the fingers curl out of its plane.
    return np.column_stack((across + 0.40 * depth, -along + 0.55 * depth))


def motion_features(points: Points | None) -> Features | None:
    if points is None:
        return None
    for a, b, c in ANGLE_TRIPLETS:
        center = point_at(points, b)
        if min(np.linalg.norm(point_at(points, a) - center),
               np.linalg.norm(point_at(points, c) - center)) < 1e-5:
            return None
    flex = 180 - hand_angles(points)
    values: Features = {"thumb_mcp_deg": flex.item(1), "thumb_ip_deg": flex.item(2),
              # Proximal thumb direction projected onto index-to-pinky axis.
              # Unlike thumb tip position, this is independent of distal IP bend.
              "thumb_across": (points.item(2, 0) - points.item(1, 0)) /
                              float(np.linalg.norm(point_at(points, 2) - point_at(points, 1)))}
    for i, finger in enumerate(FINGER_NAMES, start=1):
        for j, joint in enumerate(("mcp", "pip", "dip")):
            values[f"{finger}_{joint}_deg"] = flex.item(3*i+j)
    return values


def raw_tip_gaps(world: Sequence[Landmark]) -> Points:
    xyz = np.array([(p.x, p.y, p.z) for p in world])
    return 1000 * row_lengths(xyz.take([8, 12, 16, 20], axis=0) - point_at(xyz, 4))


class PoseCapture:
    """Space starts a burst; only valid samples of the configured hand count."""
    prompts = {"open": "Open hand; thumb out", "curled": "Curl four fingers",
               "thumb_bend": "Bend thumb knuckles", "thumb_opposed": "Sweep thumb across palm"}

    def __init__(self, profile: Profile, operator: str, path: Path) -> None:
        self.profile, self.operator, self.path = profile, operator, path
        self.poses = [p for p in POSES if any(p in FEATURE_POSES[k][:2] for k in profile.features)]
        self.samples: Samples = {p: [] for p in self.poses}
        self.index = 0
        self.active = False

    @property
    def prompt(self) -> str:
        pose = self.poses[self.index]
        count = len(self.samples[pose])
        return f"{self.prompts[pose]}: " + (f"hold {count}/20" if self.active else "press SPACE when ready")

    def update(self, features: Features | None) -> bool:
        if not self.active or features is None:
            return False
        batch = self.samples[self.poses[self.index]]
        batch.append(features)
        if len(batch) < 20:
            return False
        self.active = False
        if self.index + 1 < len(self.poses):
            self.index += 1
            return False
        try:
            data = calibrate(self.profile, self.operator, self.samples)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            self.index = 0
            for values in self.samples.values():
                values.clear()
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x") as out:
            json.dump(data, out, indent=2)
            out.write("\n")
        print(f"Saved calibration to {self.path}", file=sys.stderr)
        return True


def simulate_hand(points: Points | None = None, size: tuple[int, int] = (450, 450)) -> NDArray[np.uint8]:
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(canvas, "PALM-RELATIVE HAND", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if points is None:
        cv2.putText(canvas, "Waiting for 3D hand landmarks", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return canvas

    origin = np.array([0.5 * width, 0.74 * height])
    pixels = np.rint(origin + min(width, height) * 0.25 * project_hand(points))
    pixels = np.clip(pixels, -32768, 32767).astype(np.int32)
    cv2.polylines(canvas, [pixels[list(PALM)]], False, (150, 150, 150),
                  2, cv2.LINE_AA)
    for finger in FINGERS:
        joints = pixels[list(finger)]
        color = (0, 190, 255) if finger[0] == 1 else (80, 220, 110)
        cv2.polylines(canvas, [joints], False, color, 4, cv2.LINE_AA)
        for i in range(len(joints)):
            cv2.circle(canvas, (joints.item(i, 0), joints.item(i, 1)), 4, color, -1, cv2.LINE_AA)
    return canvas

# ------------------------------------------------------------------
# 5. Main loop
# ------------------------------------------------------------------
def run_simulation(capture: cv2.VideoCapture, landmarker: HolisticLandmarker, profile: Profile,
                   mapper: Mapper | None, wizard: PoseCapture | None,
                   recordings: Sequence[TextIO] | None, diagnostics: bool) -> int:
    hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    previous_time = None
    timestamp_ms = -1
    smoothed_points = None

    cv2.imshow(SIM_WINDOW, simulate_hand())

    while True:
        ret, frame = capture.read()
        capture_ns = time.monotonic_ns()
        if not ret:
            print("Could not read a frame from the camera.", file=sys.stderr)
            return 1

        H, W = 600, 800
        frame = np.asarray(cv2.resize(frame, (W, H)), dtype=np.uint8)

        rgb_frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(timestamp_ms + 1, capture_ns // 1_000_000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        right_lms = results.right_hand_landmarks
        left_lms = results.left_hand_landmarks

        # Draw detected hands on webcam view
        if right_lms:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, right_lms, hand_connections)
        if left_lms:
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, left_lms, hand_connections)

        world = results.left_hand_world_landmarks if profile.side == "left" else results.right_hand_world_landmarks
        points = palm_coordinates(world, is_left=profile.side == "left")
        features = motion_features(points)

        # FPS
        current_time = time.perf_counter()
        elapsed = 0.0
        if previous_time is not None and current_time > previous_time:
            elapsed = current_time - previous_time
            fps = 1 / elapsed
            cv2.putText(frame, f"{int(fps)} FPS", (10, H - 15),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        previous_time = current_time

        # Info overlay + sim render
        if features is not None and points is not None:
            if smoothed_points is None:
                smoothed_points = points
            else:
                alpha = -math.expm1(-elapsed / SMOOTHING_TIME)
                smoothed_points += alpha * (points - smoothed_points)
            status, color = f"Tracking: {profile.side}", (0, 255, 255)
        else:
            smoothed_points = None
            status, color = f"No {profile.side} hand", (0, 0, 255)

        if wizard:
            if wizard.update(features):
                return 0
            label = wizard.prompt
        else:
            if mapper is None:
                raise ValueError("Streaming requires a mapper")
            msg = mapper.message(capture_ns, features)
            emit(sys.stdout, msg)
            if recordings:
                raw, targets = recordings
                feature_row = {k: v for k, v in msg.items() if k not in ("q_target", "tracking_valid")}
                emit(raw, {**feature_row, "features": features})
                emit(targets, msg)
            label = "Targets: " + (", ".join(f"{q:.2f}" for q in msg["q_target"])
                                    if msg["q_target"] is not None else "--")
        cv2.putText(frame, label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if diagnostics and features is not None:
            distances = raw_tip_gaps(world)
            gaps = " ".join(f"{name}:{distances.item(i):.0f}" for i, name in enumerate(FINGER_NAMES))
            cv2.putText(frame, f"Raw thumb gap (mm) {gaps}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, status, (W - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(SIM_WINDOW, simulate_hand(smoothed_points))
        cv2.imshow("Hand Landmarks (Tasks API)", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            return 0
        if key == ord('d'):
            diagnostics = not diagnostics
        if key == ord(' ') and wizard:
            wizard.active = True


class TrackerArgs(argparse.Namespace):
    camera: str
    mjpeg: bool
    profile: Path
    calibration: Path | None
    calibrate: str | None
    record: Path | None
    diagnostics: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the webcam hand simulator.")
    parser.add_argument("--camera", default="0",
                        help="camera index or Linux device path (default: 0)")
    parser.add_argument("--mjpeg", action="store_true",
                        help="request 640x480 MJPEG capture")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--calibration", type=Path, help="input calibration, or new output path with --calibrate")
    parser.add_argument("--calibrate", metavar="OPERATOR", help="collect calibration instead of streaming")
    parser.add_argument("--record", type=Path, help="new directory for features and commands")
    parser.add_argument("--diagnostics", action="store_true", help="show raw fingertip gaps (also toggle with d)")
    args = parser.parse_args(namespace=TrackerArgs())
    calibration: dict[str, object] | None = None
    try:
        profile_data = load(args.profile)
        profile = Profile(profile_data)
        wizard = mapper = None
        if args.calibrate is not None:
            if (not args.calibrate.strip() or Path(args.calibrate).name != args.calibrate
                    or args.calibrate in (".", "..") or args.record):
                parser.error("--calibrate requires an operator name and cannot be used with --record")
            path = args.calibration or DEFAULT_CAL.parent / f"{args.calibrate}-webcam-{profile.side}.json"
            if path.exists():
                parser.error(f"Calibration already exists: {path}; choose a new --calibration path")
            wizard = PoseCapture(profile, args.calibrate, path)
        else:
            calibration = load(args.calibration or DEFAULT_CAL)
            mapper = Mapper(profile, calibration)
            if calibration["source"] not in ("webcam", "synthetic"):
                parser.error("Webcam requires webcam or synthetic calibration")
            if calibration["source"] == "synthetic":
                print("Using synthetic demo calibration; use --calibrate for measured endpoints.", file=sys.stderr)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    source = int(args.camera) if args.camera.isdecimal() else args.camera

    is_linux = sys.platform.startswith("linux")
    backend = cv2.CAP_V4L2 if is_linux else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    try:
        if not capture.isOpened():
            hint = (" Check /dev/video* and your camera permissions."
                    if is_linux else "")
            print(f"Could not open camera {args.camera!r}.{hint}", file=sys.stderr)
            return 1

        if args.mjpeg:
            mjpeg_fourcc = cv2.VideoWriter.fourcc("M", "J", "P", "G")
            capture.set(cv2.CAP_PROP_FOURCC, mjpeg_fourcc)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if (capture.get(cv2.CAP_PROP_FOURCC) != mjpeg_fourcc
                    or capture.get(cv2.CAP_PROP_FRAME_WIDTH) != 640
                    or capture.get(cv2.CAP_PROP_FRAME_HEIGHT) != 480):
                print("Camera did not negotiate 640x480 MJPEG; trying the available mode.",
                      file=sys.stderr)

        download_model()
        with HolisticLandmarker.create_from_options(options) as landmarker, ExitStack() as stack:
            recordings = None
            if args.record and calibration is not None:
                args.record.mkdir(parents=True, exist_ok=False)
                for name, data in (("profile", profile_data), ("calibration", calibration)):
                    (args.record / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
                recordings = [stack.enter_context((args.record / f"{name}.jsonl").open("x"))
                              for name in ("features", "commands")]
            try:
                return run_simulation(capture, landmarker, profile, mapper, wizard, recordings, args.diagnostics)
            finally:
                cv2.destroyAllWindows()
    finally:
        capture.release()


if __name__ == "__main__":
    sys.exit(main())
