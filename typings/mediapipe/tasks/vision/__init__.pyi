from contextlib import AbstractContextManager
from enum import Enum
from numpy import uint8
from numpy.typing import NDArray
from mediapipe import Image
from mediapipe.tasks import BaseOptions
from typing import Protocol

class _Point(Protocol):
    x: float
    y: float
    z: float

class _Connection(Protocol):
    start: int
    end: int

class HandLandmarksConnections:
    HAND_CONNECTIONS: list[_Connection]

class drawing_utils:
    @staticmethod
    def draw_landmarks(image: NDArray[uint8], landmark_list: list[_Point], connections: list[_Connection]) -> None: ...

class RunningMode(Enum):
    VIDEO = "VIDEO"

class HolisticLandmarkerOptions:
    def __init__(self, *, base_options: BaseOptions, running_mode: RunningMode,
                 min_pose_detection_confidence: float, min_hand_landmarks_confidence: float) -> None: ...

class HolisticLandmarkerResult:
    right_hand_landmarks: list[_Point]
    left_hand_landmarks: list[_Point]
    right_hand_world_landmarks: list[_Point]
    left_hand_world_landmarks: list[_Point]

class HolisticLandmarker(AbstractContextManager["HolisticLandmarker"]):
    @classmethod
    def create_from_options(cls, options: HolisticLandmarkerOptions) -> HolisticLandmarker: ...
    def detect_for_video(self, image: Image, timestamp_ms: int) -> HolisticLandmarkerResult: ...
