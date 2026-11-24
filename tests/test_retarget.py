import copy
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import mujoco

from teleop_gloves import hand_tracker as tracker
from teleop_gloves.mujoco_input import HandInput
from teleop_gloves.retarget import (
    DEFAULT_CAL, DEFAULT_PROFILE, FEATURE_POSES, ROOT, Features, Mapper, Profile, Samples,
    calibrate, is_array, load, numeric, parse_record, record, replay,
)


class MappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = Profile(load(DEFAULT_PROFILE))
        self.cal = load(DEFAULT_CAL)
        self.mapper = Mapper(self.profile, self.cal)
        self.fixture = ROOT / "examples/demo/features.jsonl"
        self.rows = [parse_record(line) for line in self.fixture.read_text().splitlines()]

    def test_reference_recording(self) -> None:
        out = io.StringIO()
        with self.fixture.open() as lines:
            replay(self.mapper, lines, out)
        self.assertEqual(out.getvalue(), (self.fixture.parent / "commands.jsonl").read_text())
        vectors = [parse_record(line)["q_target"] for line in out.getvalue().splitlines()]
        for got, expected in zip(vectors, [0, 0.5, 1]):
            assert is_array(got)
            np.testing.assert_allclose([numeric(v) for v in got], [expected]*6)
        self.assertIsNone(vectors[-1])

    def test_dex3_order_and_clamping(self) -> None:
        dex = Profile(load(DEFAULT_PROFILE.with_name("uthr_competition_dex3_right.json")))
        mapper = Mapper(dex, self.cal)
        values = dict.fromkeys(dex.features, 500.)
        self.assertEqual(mapper.message(4, values)["q_target"],
                         [1.0472, -1.0472, -1.74533, 1.5708, 1.74533, 1.5708, 1.74533])
        values = dict.fromkeys(dex.features, -500.)
        self.assertEqual(mapper.message(5, values)["q_target"], [0]*7)

    def test_coupling_and_reversed_human_range(self) -> None:
        record(self.cal["endpoints"])["index_pip_deg"] = [90, 0]
        mapper = Mapper(self.profile, self.cal)
        values = record(self.rows[0]["features"])
        targets = mapper.message(1, values)["q_target"]
        assert targets is not None
        self.assertAlmostEqual(targets[0], 1/3)
        values["index_pip_deg"] = 45
        targets = mapper.message(2, values)["q_target"]
        assert targets is not None
        self.assertAlmostEqual(targets[0], 1/6)

    def test_missing_features_and_wrong_configuration(self) -> None:
        for features in [None, {}, {"index_pip_deg": float("nan")}]:
            msg = self.mapper.message(1, features)
            self.assertFalse(msg["tracking_valid"])
            self.assertIsNone(msg["q_target"])
        with self.assertRaises(ValueError):
            Profile(load(DEFAULT_PROFILE.with_name("sickkids_g1.template.json")))
        changed = copy.deepcopy(self.cal)
        a, b = self.mapper.endpoints["thumb_ip_deg"]
        record(changed["endpoints"])["thumb_ip_deg"] = [a, b + 1]
        with self.assertRaisesRegex(ValueError, "mismatch"):
            with self.fixture.open() as lines:
                replay(Mapper(self.profile, changed), lines, io.StringIO())

    def test_calibration_medians_and_weak_pose(self) -> None:
        samples: Samples = {}
        for k, (pose_a, pose_b, _) in FEATURE_POSES.items():
            for i, pose in enumerate((pose_a, pose_b)):
                frames = samples.setdefault(pose, [{} for _ in range(20)])
                for n, row in enumerate(frames):
                    row[k] = self.mapper.endpoints[k][i] if n != 0 else 10000
        data = calibrate(self.profile, "test", samples)
        self.assertEqual(data["endpoints"], self.cal["endpoints"])
        for row in samples["thumb_opposed"]:
            row["thumb_across"] = -0.79
        with self.assertRaisesRegex(ValueError, "insufficient span"):
            calibrate(self.profile, "test", samples)

    def test_named_simulator_indices_and_hold(self) -> None:
        profile = Profile(load(DEFAULT_PROFILE.with_name("uthr_competition_dex3_right.json")))
        mapper = Mapper(profile, self.cal)
        names = ["other", *(c["name"] for c in reversed(profile.controls))]
        bodies = "".join(f'<body><joint name="{name}"/><geom size=".01"/></body>' for name in names)
        servos = "".join(f'<position name="{name}" joint="{name}" ctrlrange="-2 2"/>' for name in names)
        model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody>"
                                                f"<actuator>{servos}</actuator></mujoco>")
        data = mujoco.MjData(model)
        hand = HandInput(model, profile)
        data.ctrl[:] = 0.75
        msg = mapper.message(1, record(self.rows[1]["features"]))
        hand.apply(data, msg)
        hand.apply(data, mapper.message(2, None))
        assert msg["q_target"] is not None
        np.testing.assert_allclose(data.ctrl.take([names.index(c["name"]) for c in profile.controls]), msg["q_target"])
        self.assertEqual(data.ctrl.item(0), 0.75)

    def test_guided_pose_capture_writes_usable_operator_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration/test-webcam-right.json"
            wizard = tracker.PoseCapture(self.profile, "test", path)
            values: Features = {}
            complete = False
            for pose in wizard.poses:
                values = {k: pair[int(FEATURE_POSES[k][1] == pose)]
                          for k, pair in self.mapper.endpoints.items()}
                self.assertFalse(wizard.update(values))  # Not armed.
                wizard.active = True
                self.assertFalse(wizard.update(None))  # Lost frame doesn't count.
                for _ in range(20):
                    complete = wizard.update(values)
                self.assertEqual(complete, pose == wizard.poses[-1])
            measured = load(path)
            self.assertEqual(measured["source"], "webcam")
            self.assertEqual(measured["operator"], "test")
            self.assertEqual(measured["endpoints"], self.cal["endpoints"])
            self.assertTrue(Mapper(self.profile, measured).message(1, values)["tracking_valid"])

    def test_opposite_hand_does_not_drive_live_output(self) -> None:
        cam = MagicMock(**{"read.return_value": (True, np.zeros((8, 8, 3), dtype=np.uint8))})
        result = MagicMock(left_hand_world_landmarks=[MagicMock()] * 21, right_hand_world_landmarks=[],
                           right_hand_landmarks=[], left_hand_landmarks=[])
        detect = MagicMock(return_value=result)
        model = MagicMock(detect_for_video=detect)
        raw, commands, stdout = io.StringIO(), io.StringIO(), io.StringIO()
        with patch.object(tracker.cv2, "imshow"), patch.object(tracker.cv2, "waitKey", return_value=ord("q")), \
                patch("sys.stdout", stdout), patch.object(tracker.time, "monotonic_ns", return_value=1250000001):
            tracker.run_simulation(cam, model, self.profile, self.mapper, None, [raw, commands], False)
        msg = parse_record(stdout.getvalue())
        self.assertEqual(msg["timestamp_ns"], 1250000001)
        from unittest.mock import ANY
        detect.assert_called_once_with(ANY, 1250)
        self.assertIsNone(msg["q_target"])
        self.assertIsNone(parse_record(raw.getvalue())["features"])
        self.assertEqual(commands.getvalue(), stdout.getvalue())
