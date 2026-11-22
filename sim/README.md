# G1 Sim (MuJoCo)
how to get unitree g1 in mujoco in windows!

## Step 1: Clone repo and switch to the sim branch

```
git clone https://github.com/university-of-toronto-humanoid-robotics/teleop-gloves.git
cd teleop-gloves
git checkout g1-sim
cd sim
```

## Step 2: venv 

```
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` in your prompt.

Mac/Linux: use `source .venv/bin/activate` instead.

## Step 3: Install MuJoCo

```
pip install -r requirements.txt
```

## Step 4: Download the G1 model

```
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

## Step 5: Run the sim

```
python -m mujoco.viewer --mjcf=mujoco_menagerie/unitree_g1/scene.xml
```

A window with the G1 should open

## Note

The stock G1 hand is rigid (no finger joints) !
