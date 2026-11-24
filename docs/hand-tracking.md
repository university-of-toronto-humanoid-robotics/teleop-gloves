# Hand Landmark → 3D Motion Preview and Joint Angles

> ## ⚠️ STATUS: VIBE-CODED PROTOTYPE — READ THIS FIRST
>
> This code was **AI-generated (vibe-coded)** as a quick **feasibility spike**. It was
> written to answer one question: *"Can we get usable joint angles out of MediaPipe's
> hand landmarks and drive a simple simulated hand with them?"*
>
> **Treat this as a disposable experiment, not a codebase.**
>
> Its only job was to prove the concept was viable and give us something to play with.

---

## What it does

The OpenCV + MediaPipe tracker:

1. Grabs your webcam feed.
2. Runs Google's **MediaPipe Holistic Landmarker** (Tasks API) on each frame.
3. Detects hands and extracts the **21 hand landmarks** per hand.
4. Extracts named flexion and thumb-opposition features for the configured hand,
   maps them to robot controls, and emits one JSON command per frame.
5. Shows a **palm-relative motion preview** in a separate window. Its angled view
   uses the tracked joint positions directly, so thumb opposition and
   out-of-plane finger curl can be visible without forcing every joint to bend
   in the same direction.

Try touching your thumb to each fingertip, then rotating your wrist while
holding a pose. The thumb is highlighted, and the preview keeps the palm
in a consistent orientation. The preview validates the landmark motion,
not a robot's ability to reproduce the pose.


## Quick start on Linux

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
uv sync --locked
uv run --locked -m teleop_gloves.hand_tracker
```

`uv` uses Python 3.11 and creates a `.venv`; no manual activation is needed.
The demo requires a graphical desktop session and access to a V4L2 webcam.
If the default camera doesn't work, run `ls /dev/video*` and try another
device, for example:

```bash
uv run --locked -m teleop_gloves.hand_tracker --camera /dev/video2
```

If the camera opens but frame reads time out (for example, with a USB webcam
passed through to WSL), try requesting 640×480 MJPEG on a device that supports it:

```bash
uv run --locked -m teleop_gloves.hand_tracker --camera /dev/video0 --mjpeg
```

On the first run with a working camera, the script downloads
`holistic_landmarker.task` from Google and saves it in `models/` at the repository root.
The demo runs on the CPU; it does not require CUDA or ROCm and does not use
the BodyPose3D ONNX model.

**Controls:**

| Key | Action |
|-----|--------|
| `q` (in either window) | Quit |
| `d` | Toggle unsmoothed thumb-to-tip gaps in estimated mm |
| Space | Start sampling the displayed calibration pose |

---

## How the angles are computed

For each finger, three angles are measured at three joints. MediaPipe hand landmark
indices used:

| Finger | Triplets (a, b, c) → angle at `b` |
|--------|-----------------------------------|
| Thumb  | (0,1,2), (1,2,3), (2,3,4) |
| Index  | (0,5,6), (5,6,7), (6,7,8) |
| Middle | (0,9,10), (9,10,11), (10,11,12) |
| Ring   | (0,13,14), (13,14,15), (14,15,16) |
| Pinky  | (0,17,18), (17,18,19), (18,19,20) |

The included angle between `(a - b)` and `(c - b)` is near 180° for a
straight joint. For mapping we subtract it from 180° to get approximate
flexion (straight ≈ 0°). The thumb uses MCP/IP bend and a separate
across-palm feature rather than a CMC bend angle. Only the visual skeleton
is smoothed; calibration and mapping use unsmoothed features. See the
[mapping, recording and replay guide](retargeting.md).

---
