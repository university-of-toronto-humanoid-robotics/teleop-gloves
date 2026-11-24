"""Apply a hand command to a MuJoCo simulation by actuator name."""
import argparse
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import mujoco

from .retarget import Profile, is_array, load, numeric, parse_record, require


class HandInput:
    def __init__(self, model: "mujoco.MjModel", profile: Profile) -> None:
        self.profile = profile
        self.indices: list[int] = []
        for c in profile.controls:
            require(c["unit"] == "rad", "Expected a radian joint-position profile")
            actuator = model.actuator(c["name"])
            lo, hi = actuator.ctrlrange.item(0), actuator.ctrlrange.item(1)
            require(bool(actuator.ctrllimited.item(0)) and lo <= c["min"] and c["max"] <= hi,
                    f"Model control limits do not match {c['name']}")
            self.indices.append(actuator.id)
        self.last_timestamp = -1

    def apply(self, data: "mujoco.MjData", row: Mapping[str, object]) -> bool:
        require(row.get("schema_version") == 1 and row.get("profile_id") == self.profile.id,
                "Command schema/profile mismatch")
        ts = row.get("timestamp_ns")
        if type(ts) is not int or ts <= self.last_timestamp:
            raise ValueError("Out-of-order command")
        valid, q = row.get("tracking_valid"), row["q_target"]
        if valid is False and q is None:
            self.last_timestamp = ts
            return False  # Hold existing controls, including other body actuators.
        if valid is not True or not is_array(q) or len(q) != len(self.indices):
            raise ValueError("Invalid target vector")
        values = [numeric(v) for v in q]
        require(all(c["min"] <= v <= c["max"] for v, c in zip(values, self.profile.controls)),
                "Invalid target vector")
        data.ctrl[self.indices] = values
        self.last_timestamp = ts
        return True


class SimulatorArgs(argparse.Namespace):
    model: str
    profile: Path


def main() -> None:
    import mujoco  # Optional simulator dependency, not needed for tracking/replay.
    parser = argparse.ArgumentParser(description="Feed simulator JSONL on stdin to a MuJoCo model.")
    parser.add_argument("model")
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args(namespace=SimulatorArgs())
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    hand = HandInput(model, Profile(load(args.profile)))
    frames = updated = 0
    for line in sys.stdin:
        updated += hand.apply(data, parse_record(line))
        mujoco.mj_step(model, data)
        frames += 1
    print(f"Frames: {frames}; commands applied: {updated}; final hand ctrl: "
          f"{data.ctrl[hand.indices].tolist()}", file=sys.stderr)


if __name__ == "__main__":
    main()
