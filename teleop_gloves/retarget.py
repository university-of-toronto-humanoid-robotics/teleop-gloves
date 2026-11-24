"""Simulator contract, sensor-independent mapping, and offline JSONL replay."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from collections.abc import Iterable, Mapping
from typing import TextIO, TypeGuard, TypedDict, cast

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "config/hands/generic_five_finger_demo.json"
DEFAULT_CAL = ROOT / "calibration/demo-synthetic-right.json"
POSES = ("open", "curled", "thumb_bend", "thumb_opposed")
FINGERS = ("index", "middle", "ring", "pinky")
FEATURE_POSES: dict[str, tuple[str, str, float]] = {
    **{f"{finger}_{joint}_deg": ("open", "curled", 5)
       for finger in FINGERS for joint in ("mcp", "pip", "dip")},
    **{f"thumb_{joint}_deg": ("open", "thumb_bend", 5) for joint in ("mcp", "ip")},
    "thumb_across": ("open", "thumb_opposed", 0.08),
}


Features = dict[str, float]
Samples = dict[str, list[Features]]


class Control(TypedDict):
    name: str
    unit: str
    min: float
    max: float
    a: float
    b: float
    weights: dict[str, float]


class Calibration(TypedDict):
    operator: str
    source: str
    side: str
    endpoints: dict[str, list[float]]
    pose_pairs: dict[str, list[str]]


class Command(TypedDict):
    schema_version: int
    timestamp_ns: int
    profile_id: str
    calibration_id: str
    tracking_valid: bool
    q_target: list[float] | None


def number(x: object) -> TypeGuard[float]:
    return (type(x) is float or type(x) is int) and math.isfinite(x)


def is_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def is_record(value: object) -> TypeGuard[dict[str, object]]:
    # JSON object keys are always strings.
    return isinstance(value, dict) and all(isinstance(key, str) for key in cast(Iterable[object], value))


def record(value: object) -> dict[str, object]:
    if not is_record(value):
        raise ValueError("Expected a JSON object")
    return value


def text(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {key}")
    return value


def numeric(value: object) -> float:
    if not number(value):
        raise ValueError("Expected a finite number")
    return value


def load(path: Path) -> dict[str, object]:
    with open(path) as stream:
        return parse_record(stream.read())


def parse_record(line: str) -> dict[str, object]:
    # json.loads has a dynamic return type; validate the outer object before use.
    return record(cast(object, json.loads(line)))


def revision(data: object) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()[:16]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Profile:
    def __init__(self, data: Mapping[str, object]) -> None:
        require(data.get("enabled") is True,
                "Incomplete or disabled hand profile")
        for key in ("model", "revision", "source"):
            text(data.get(key), key)
        self.side = text(data.get("side"), "side")
        require(self.side in ("left", "right"), "Expected left or right hand")
        self.id = f"{data['model']}:{self.side}:{data['revision']}:{revision(data)}"
        controls = data.get("controls")
        if not is_array(controls) or not controls:
            raise ValueError("Missing ordered controls")
        self.controls: list[Control] = []
        names: set[str] = set()
        self.features: set[str] = set()
        for item in controls:
            c = record(item)
            name = text(c.get("name"), "name")
            require(name not in names, "Duplicate control name")
            names.add(name)
            unit = text(c.get("unit"), "unit")
            lo, hi, a, b = (numeric(c.get(k)) for k in ("min", "max", "a", "b"))
            require(lo <= min(a, b) < max(a, b) <= hi,
                    f"{name}: invalid range or endpoints")
            weights = {k: numeric(w) for k, w in record(c.get("weights")).items()}
            require(bool(weights) and all(k in FEATURE_POSES and w > 0 for k, w in weights.items()),
                    f"{name}: invalid feature weights")
            self.features.update(weights)
            self.controls.append(Control(name=name, unit=unit, min=lo, max=hi, a=a, b=b, weights=weights))


def calibrate(profile: Profile, operator: str, samples: Samples) -> Calibration:
    endpoints: dict[str, list[float]] = {}
    for name in sorted(profile.features):
        a, b, min_span = FEATURE_POSES[name]
        values: list[float] = []
        for pose in (a, b):
            frames = samples.get(pose, [])
            readings = [frame[name] for frame in frames if number(frame.get(name))]
            require(len(readings) >= 10, f"{name}: need 10 valid frames for {pose}")
            values.append(median(readings))
        require(abs(values[1] - values[0]) >= min_span,
                f"{name}: insufficient span between {a} and {b}; repeat calibration")
        endpoints[name] = values
    return {"operator": operator, "source": "webcam", "side": profile.side,
            "pose_pairs": {k: list(FEATURE_POSES[k][:2]) for k in endpoints},
            "endpoints": endpoints}


class Mapper:
    def __init__(self, profile: Profile, calibration: Mapping[str, object]) -> None:
        self.profile = profile
        require(calibration.get("side") == profile.side,
                "Calibration hand does not match profile")
        for key in ("source", "operator"):
            text(calibration.get(key), key)
        self.cal_id = revision(calibration)
        endpoints = record(calibration.get("endpoints"))
        poses = record(calibration.get("pose_pairs"))
        self.endpoints: dict[str, tuple[float, float]] = {}
        for k in profile.features:
            pair = endpoints.get(k)
            if not is_array(pair) or len(pair) != 2:
                raise ValueError(f"{k}: missing calibration pair")
            a, b = numeric(pair[0]), numeric(pair[1])
            require(abs(b - a) >= FEATURE_POSES[k][2],
                    f"{k}: missing or insufficient calibration")
            require(poses.get(k) == list(FEATURE_POSES[k][:2]),
                    f"{k}: wrong calibration poses")
            self.endpoints[k] = a, b

    def message(self, timestamp_ns: int, features: Mapping[str, object] | None) -> Command:
        require(type(timestamp_ns) is int and timestamp_ns >= 0, "Invalid capture timestamp")
        targets: list[float] | None = None
        if features is not None and all(number(features.get(k)) for k in self.profile.features):
            u: dict[str, float] = {}
            for k in self.profile.features:
                a, b = self.endpoints[k]
                u[k] = min(1., max(0., (numeric(features[k]) - a) / (b - a)))
            targets = []
            for c in self.profile.controls:
                weights = c["weights"]
                f = sum(w * u[k] for k, w in weights.items()) / sum(weights.values())
                targets.append(min(c["max"], max(c["min"], c["a"] + f * (c["b"] - c["a"]))))
        return {"schema_version": 1, "timestamp_ns": timestamp_ns, "profile_id": self.profile.id,
                "calibration_id": self.cal_id, "tracking_valid": targets is not None, "q_target": targets}


def emit(stream: TextIO, row: object) -> None:
    print(json.dumps(row, allow_nan=False, separators=(",", ":")), file=stream, flush=True)


def replay(mapper: Mapper, lines: Iterable[str], stream: TextIO) -> None:
    last = -1
    for line in lines:
        row = parse_record(line)
        require(row.get("schema_version") == 1 and "features" in row
                and row.get("calibration_id") == mapper.cal_id and row.get("profile_id") == mapper.profile.id,
                "Feature recording schema/profile/calibration mismatch")
        ts = row.get("timestamp_ns")
        if type(ts) is not int:
            raise ValueError("Invalid capture timestamp")
        features = row["features"]
        msg = mapper.message(ts, features if is_record(features) else None)
        require(msg["timestamp_ns"] > last, "Expected increasing capture timestamps")
        last = msg["timestamp_ns"]
        emit(stream, msg)


class ReplayArgs(argparse.Namespace):
    features: Path
    profile: Path
    calibration: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a feature recording to simulator JSONL on stdout.")
    parser.add_argument("features", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    args = parser.parse_args(namespace=ReplayArgs())
    try:
        mapper = Mapper(Profile(load(args.profile)), load(args.calibration))
        with args.features.open() as source:
            replay(mapper, source, sys.stdout)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
