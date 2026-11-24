from dataclasses import dataclass
import unittest

import numpy as np

from teleop_gloves.hand_tracker import Points, hand_angles, motion_features, palm_coordinates, project_hand, simulate_hand
from teleop_gloves.retarget import Features


@dataclass
class Point:
    x: float
    y: float
    z: float


def landmarks(points: Points) -> list[Point]:
    return [Point(points.item(i, 0), points.item(i, 1), points.item(i, 2)) for i in range(len(points))]


class HandMotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points: Points = np.array([
            (0, 0, 0),
            (-0.3, 0.2, 0), (-0.5, 0.4, 0), (-0.7, 0.6, 0), (-0.9, 0.8, 0),
            (-0.4, 0.9, 0), (-0.4, 1.2, 0), (-0.4, 1.4, 0), (-0.4, 1.6, 0),
            (0, 1, 0), (0, 1.4, 0), (0, 1.7, 0), (0, 2, 0),
            (0.2, 0.9, 0), (0.2, 1.2, 0), (0.2, 1.5, 0), (0.2, 1.7, 0),
            (0.4, 0.8, 0), (0.4, 1.1, 0), (0.4, 1.3, 0), (0.4, 1.5, 0),
        ])

    def test_palm_frame_ignores_wrist_pose_and_scale(self) -> None:
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        moved = 2 * self.points @ rotation.T + (3, 4, 5)
        original = palm_coordinates(landmarks(self.points), is_left=False)
        actual = palm_coordinates(landmarks(moved), is_left=False)
        assert actual is not None and original is not None
        np.testing.assert_allclose(actual, original, atol=1e-12)

    def test_thumb_depth_changes_preview_and_angles(self) -> None:
        bent = self.points.copy()
        bent[4, 2] = 0.4
        original = palm_coordinates(landmarks(self.points), is_left=False)
        actual = palm_coordinates(landmarks(bent), is_left=False)
        assert actual is not None and original is not None
        np.testing.assert_allclose(project_hand(actual)[5:], project_hand(original)[5:])
        self.assertFalse(np.allclose(project_hand(actual).take(4, axis=0), project_hand(original).take(4, axis=0)))
        angles = hand_angles(actual)
        self.assertEqual(len(angles), 15)
        self.assertAlmostEqual(hand_angles(original).item(2), 180)
        self.assertLess(angles.item(2), 180)
        self.assertFalse(np.array_equal(simulate_hand(actual), simulate_hand(original)))

    def test_mirrored_hands_share_preview(self) -> None:
        self.points[4, 2] = 0.4
        mirrored = self.points * (-1, 1, 1)
        right = palm_coordinates(landmarks(self.points), is_left=False)
        left = palm_coordinates(landmarks(mirrored), is_left=True)
        assert left is not None and right is not None
        np.testing.assert_allclose(left, right)

    def test_invalid_palm_is_not_rendered(self) -> None:
        self.assertIsNone(palm_coordinates(None, is_left=False))
        self.assertIsNone(palm_coordinates(landmarks(np.zeros((21, 3))), is_left=False))

    def test_proximal_sweep_and_thumb_tip_bend_are_distinct(self) -> None:
        def features(points: Points) -> Features:
            result = motion_features(palm_coordinates(landmarks(points), is_left=False))
            assert result is not None
            return result
        opened = features(self.points)
        bent = self.points.copy()
        bent[4, 2] = 0.4
        self.assertAlmostEqual(opened["index_pip_deg"], 0)
        self.assertGreater(features(bent)["thumb_ip_deg"], opened["thumb_ip_deg"])
        self.assertEqual(features(bent)["thumb_across"], opened["thumb_across"])
        swept = self.points.copy()
        swept[1:5, 0] = 2*self.points[1, 0] - swept[1:5, 0]
        self.assertAlmostEqual(features(swept)["thumb_ip_deg"], opened["thumb_ip_deg"])
        self.assertGreater(features(swept)["thumb_across"], opened["thumb_across"])
