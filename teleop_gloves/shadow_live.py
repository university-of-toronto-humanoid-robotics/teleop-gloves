"""Webcam hand tracking -> Shadow Hand in MuJoCo (live)."""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np

from .hand_tracker import HolisticLandmarker, MODEL_PATH, download_model, motion_features, options, palm_coordinates
from .mujoco_input import HandInput
from .retarget import DEFAULT_CAL, ROOT, Mapper, Profile, load

SCENE = ROOT / "sim/mujoco_menagerie/shadow_hand/scene_right.xml"
PROFILE = ROOT / "config/hands/shadow_right.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Drive the MuJoCo Shadow Hand from your webcam.")
    p.add_argument("--scene", type=Path, default=SCENE)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    profile = Profile(load(PROFILE))
    mapper = Mapper(profile, load(args.calibration))
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    hand = HandInput(model, profile)

    cap = cv2.VideoCapture(args.camera)
    download_model()
    with HolisticLandmarker.create_from_options(options) as landmarker, \
            mujoco.viewer.launch_passive(model, data) as viewer:
        last = time.monotonic()
        while viewer.is_running():
            ok, frame = cap.read()
            if not ok:
                break
            t_ns = time.monotonic_ns()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(rgb, dtype=np.uint8)),
                t_ns // 1_000_000)

            # CV code: landmarks -> palm-relative points -> named joint-angle features
            features = motion_features(palm_coordinates(result.right_hand_world_landmarks, is_left=False))
            # Mapper: features -> joint targets (profile + calibration); None => tracking lost
            hand.apply(data, mapper.message(t_ns, features))

            # Advance physics by the real time that passed, then redraw
            now = time.monotonic()
            for _ in range(max(1, min(int((now - last) / model.opt.timestep), 200))):
                mujoco.mj_step(model, data)
            last = now
            viewer.sync()

            if result.right_hand_landmarks:
                mp.tasks.vision.drawing_utils.draw_landmarks(
                frame, result.right_hand_landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)
    
            cv2.imshow("camera (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
