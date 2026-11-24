# Hand mapping and simulator input

## Calibrate, stream, record

Run from the repository root. Defaults: right hand, six-control synthetic demo profile.

```bash
uv run --locked -m teleop_gloves.hand_tracker --mjpeg --calibrate alex
uv run --locked -m teleop_gloves.hand_tracker --mjpeg \
  --calibration calibration/alex-webcam-right.json --record recordings/trial-01
```

Hold each displayed pose and press **Space**: open with thumb out, curl the four
fingers, bend thumb knuckles, then sweep the thumb across the palm. Hold
the thumb relatively straight during the sweep. Each burst needs 20 valid
frames. Insufficient spans trigger another calibration pass. `q` cancels.
Use a new output path with `--calibration` to keep multiple calibrations.
The default synthetic calibration is only for trying the pipeline.

Use `--profile config/hands/uthr_competition_dex3_right.json` for the
three-finger simulator. Only the configured side drives commands, even
if both hands are visible. `d` toggles raw thumb-to-tip distances. They
are model estimates in mm, not contact measurements.

## JSON Lines contract (v1)

Stdout is one command per processed frame; diagnostics go to stderr.
Each command has `schema_version: 1`, integer `timestamp_ns` (host monotonic
time immediately after capture), `profile_id`, `calibration_id`,
`tracking_valid`, and `q_target`. Timestamps are comparable within a session,
not across clocks. `q_target[i]` uses the name, unit and limits from
`profile.controls[i]`. Only independently controlled hand actuators are included.
For lost tracking the vector is `null`, not a zero pose. The consumer must
handle that case explicitly; the example adapter holds its current controls.

IDs include a content hash: profile ID is `model:side:revision:hash`,
calibration ID is its hash. The hash is the first 16 hex digits of SHA-256
of the parsed JSON serialized with sorted keys and compact separators.
Save the supplied JSON profiles rather than reconstructing IDs from filenames.

`--record` creates a **new** directory containing:

- `profile.json`, `calibration.json`: exact inputs used.
- `features.jsonl`: timestamp, both IDs, schema version, and named human features (or `null`).
- `commands.jsonl`: simulator messages. These are requested targets, **not measured robot state**.

Replay without a camera or simulator:

```bash
uv run --locked -m teleop_gloves.retarget recordings/trial-01/features.jsonl \
  --profile recordings/trial-01/profile.json --calibration recordings/trial-01/calibration.json \
  > replayed.jsonl
diff recordings/trial-01/commands.jsonl replayed.jsonl
```

For an immediate example use `examples/demo/features.jsonl` with the defaults.
Its four frames represent open, midpoint, closed, and lost tracking.

## Mapping and profiles

For each non-thumb finger, `*_mcp_deg`, `*_pip_deg`, `*_dip_deg` approximate
flexion (straight ≈ 0°). MCP uses the wrist as the proximal reference.
`thumb_mcp_deg` and `thumb_ip_deg` measure bend. `thumb_across` is the
projection of the unit thumb CMC→MCP direction onto the palm's index→pinky
axis. This measures a component of opposition, not an anatomical rotation.

Each feature is normalized by its operator-specific median pose endpoints,
then clipped to `[0,1]`. A control uses a positive weighted average of its
normalized features; `a` and `b` map that fraction to robot units. The
result is clamped to `min`/`max`. Endpoints can decrease. Features and
calibration poses are shared between profiles; a glove can supply these
named measurements with its own source-specific calibration.

To configure the actual SickKids hand, copy `sickkids_g1.template.json`,
fill every field from its simulator/actuator specification, add one control
per independent input in simulator order, and enable the copy. Keep the wrist
separate. Distinguish simulator units from hardware protocol units.

## UTHR simulator

The Dex3 profile cites UTHR's pinned joint order and Menagerie MJCF. Each
listed hinge has its own radian position servo inheriting the joint range;
there are no tendon or equality couplings in that model. The human feature
weights and target endpoints within those limits are **prototype mappings**,
not manufacturer-calibrated grasp poses.

Use the existing UTHR simulator's `g1_with_hands.xml` (Menagerie revision
`cda5236f98383f6fbaa4820c6b18dba892cd087f`, with its `assets/` alongside):

```bash
uv run --locked --with 'mujoco>=3.4,<4' -m teleop_gloves.mujoco_input \
  /path/to/unitree_g1/g1_with_hands.xml \
  --profile config/hands/uthr_competition_dex3_right.json < examples/dex3/commands.jsonl
```

This headless smoke consumer advances one simulation step per line. In the
full simulator, instantiate `HandInput(model, profile)` once and call
`hand.apply(data, message)` inside its existing control loop. It writes only
named hand actuator slots; physics timing and other body controls stay with
the host simulator.
