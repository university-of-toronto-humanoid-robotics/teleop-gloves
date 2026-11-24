# Human-to-robot hand mapping

## Goal
Convert one tracked human hand into an ordered vector of position targets for a robotic hand simulator. Start with webcam tracking; a glove should later plug into the same mapping. The physical hand installed on the SickKids Unitree G1 is unconfirmed.

## Simulator contract
Emit one JSON Lines message per processed frame. Use the same format for live input, recording, and replay.

| Field | Meaning |
|---|---|
| `schema_version` | Message format version. |
| `timestamp_ns` | Monotonic time immediately after frame capture, before inference. An estimate, not a hardware timestamp. |
| `profile_id` | Identifies the robot hand, side, and mapping revision. |
| `tracking_valid` | Whether the selected hand produced usable features. |
| `q_target` | One position target per independent hand control, in profile order and units. `null` when tracking is lost. |

Never substitute the opposite hand or send zeros for missing data. Wrist and arm commands are outside this vector.

## Mapping
1. Extract named features from MediaPipe’s 21 world landmarks: individual finger flexion (straight ≈ 0°), thumb bend, and thumb movement across the palm. Keep the existing visual preview.
2. Guide the operator through open, curled, thumb-bend, and thumb-opposition poses. Use median readings across valid frames; reject missing or insufficient calibration spans.
3. Normalize each feature: `u = clamp((value - human_a) / (human_b - human_a), 0, 1)`.
4. For each robot control, use one normalized feature or a weighted average for coupled joints. Compute `target = robot_a + u_control × (robot_b - robot_a)` and clamp to the control limits. Reversed endpoints are valid.

## Configuration
Keep robot specifications separate from operator calibration. Do not hardcode dimensions, joint order, or units into the tracker.

| File | Purpose |
|---|---|
| `config/hands/sickkids_g1.template.json` | Fill in model, side, source revision, ordered control names, units, limits, pose endpoints, and feature weights. Reject incomplete profiles. |
| `config/hands/generic_five_finger_demo.json` | Runnable synthetic reference: four finger curls, thumb bend, and thumb opposition. Outputs six normalized targets. |
| `config/hands/uthr_competition_dex3_right.json` | Reference for [UTHR’s three-finger simulator](https://github.com/university-of-toronto-humanoid-robotics/Humanoid-Competition/blob/main/manipulation/sim/g1_manip/robot.py). Verify the seven controls, units, limits, and coupling against a pinned model revision before enabling it. |
| `calibration/<operator>-<source>-<side>.json` | Human feature endpoints and pose labels for one operator, sensing method, and hand side. |

## Diagnostics and acceptance
- Open, midpoint, and closed inputs produce expected targets, including reversed controls. All targets stay within limits.
- Changing profiles changes vector size, order, and scaling without tracker edits.
- Show tracking status and target values. Display raw thumb-to-fingertip distances on demand so tracking errors can be distinguished from mapping errors.
- Record human features and commanded targets separately with timestamps and profile/calibration revisions. Replaying the same features with the same configuration reproduces the target vector without a webcam.
- The simulator team can consume a sample recording using only the profile and message contract.

**Build order:** contract and profiles → mapper and replay tests → webcam features and calibration → live output and diagnostics → simulator integration.

**Out of scope:** physical robot control, arm mapping, model training, and guaranteeing fingertip contact across different hand geometries.
