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
might take a while. downloads collection of models from google deepmind 

## Step 5: Run the sim

```
python -m mujoco.viewer --mjcf=mujoco_menagerie/unitree_g1/scene.xml
```

A window with the G1 should open BUT the stock G1 hand is rigid (no finger joints) !

Alternatively you can run: 
```
python -m mujoco.viewer --mjcf=mujoco_menagerie/unitree_g1/scene_with_hands.xml
```
but these hands have 3 fingers. 

For a standalone hand you can try shadow hand model: 
```
python -m mujoco.viewer --mjcf=mujoco_menagerie/shadow_hand/right_hand.xml   
```

