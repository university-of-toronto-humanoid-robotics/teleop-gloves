from numpy import float64, uint8
from numpy.typing import NDArray

class _Actuator:
    id: int
    ctrlrange: NDArray[float64]
    ctrllimited: NDArray[uint8]

class MjModel:
    @classmethod
    def from_xml_string(cls, xml: str) -> MjModel: ...
    @classmethod
    def from_xml_path(cls, filename: str) -> MjModel: ...
    def actuator(self, name: str) -> _Actuator: ...

class MjData:
    ctrl: NDArray[float64]
    def __init__(self, model: MjModel) -> None: ...

def mj_step(model: MjModel, data: MjData) -> None: ...
