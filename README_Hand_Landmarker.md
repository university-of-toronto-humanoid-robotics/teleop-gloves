# Hand Landmark → 15-DOF Joint Angle Simulator

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

A single-file OpenCV + MediaPipe script that:

1. Grabs your webcam feed.
2. Runs Google's **MediaPipe Holistic Landmarker** (Tasks API) on each frame.
3. Detects hands and extracts the **21 hand landmarks** per hand.
4. Computes **15 joint angles** (3 per finger) from those landmarks.
5. Feeds those angles into a crude **2D simulated robotic hand** (15 DOF) drawn on a
   separate window, so you can visually check whether the angles track your real hand.

The point is **visual validation**: if the simulated fingers curl when your real fingers
curl, the angle extraction approach is sound enough to be worth pursuing.


## Quick start

```bash
# 1. Dependencies (Python 3.9–3.12 recommended for MediaPipe)
pip install opencv-python mediapipe numpy

# 2. Rename the file (it's currently a .txt)
mv "import cv2.txt" hand_sim.py

# 3. Run it — webcam required
python hand_sim.py
```

On first run it **downloads** `holistic_landmarker.task` (~10 MB) from Google's
storage bucket into the working directory. Needs internet for that first run only.

**Controls:**

| Key | Action |
|-----|--------|
| `q` (in either window) | Quit |

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

Each angle is the standard `acos` (arccos) of the dot product between the two vectors
`(a - b)` and `(c - b)`, computed **in image pixel space** (`x * width`, `y * height`).

The simulator then converts each angle to a "bend" (`bend = 180 - angle`) and walks
outward from each finger's base, rotating by the accumulated bend at each joint.

---