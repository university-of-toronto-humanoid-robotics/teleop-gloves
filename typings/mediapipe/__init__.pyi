from enum import Enum
from numpy import uint8
from numpy.typing import NDArray
from . import tasks as tasks

class ImageFormat(Enum):
    SRGB = 1

class Image:
    def __init__(self, image_format: ImageFormat, data: NDArray[uint8]) -> None: ...
