# teleop-gloves
Teleoperation gloves for manipulation data collection

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from the repo root:

```bash
uv sync --locked
# Try the six-control demo without a camera (prints JSON commands).
uv run --locked -m teleop_gloves.retarget examples/demo/features.jsonl
```

**Webcam → calibration → recording → replay** (Linux, right hand):

```bash
operator=Hasnain                    # Use your calibration name (case-sensitive).
trial=recordings/trial-02           # Choose a new directory for each recording.

# Calibrate once; skip if calibration/$operator-webcam-right.json already exists.
uv run --locked -m teleop_gloves.hand_tracker --mjpeg --calibrate "$operator"

# Track and record. Omit --record "$trial" for a live preview only.
uv run --locked -m teleop_gloves.hand_tracker --mjpeg \
  --calibration "calibration/$operator-webcam-right.json" --record "$trial"

# Replay the saved features using the same profile and calibration.
uv run --locked -m teleop_gloves.retarget "$trial/features.jsonl" \
  --profile "$trial/profile.json" --calibration "$trial/calibration.json"
```

Press **Space** for each calibration pose, **d** for fingertip distances, and **q** to quit.
For camera selection or WSL setup, see the [webcam guide](docs/hand-tracking.md#quick-start-on-linux).

**MuJoCo (optional):** Run the bundled Dex3 commands against the [UTHR simulator model](docs/retargeting.md#uthr-simulator). Replace the XML path with your local copy:

```bash
uv run --locked -m teleop_gloves.mujoco_input /path/to/g1_with_hands.xml \
  --profile config/hands/uthr_competition_dex3_right.json < examples/dex3/commands.jsonl
```

## Project layout

| Directory | Contents |
|---|---|
| `teleop_gloves/` | Tracking, calibration, mapping, replay, and simulator input |
| `config/hands/` | Ordered robot control profiles |
| `calibration/` | Operator calibration files and synthetic baseline |
| `examples/` | Synthetic JSONL recordings |
| `docs/` | Setup and tracking guide |
| `tests/` | Hand motion tests |
| `models/` | Downloaded MediaPipe model (created on first use) |

## Development checks

From the repo root:

```bash
uv run --locked pyright
uv run --locked basedpyright -p pyproject.toml
uv run --locked -m unittest discover -s tests
```

Pyright and VS Code/Pylance use `pyrightconfig.json` (strict, Python 3.11).
Select `.venv/bin/python` as the editor interpreter. The second checker also
reports inferred and explicit `Any`. The small stubs in `typings/` describe
the MediaPipe and MuJoCo APIs used here; check them when upgrading either library.
