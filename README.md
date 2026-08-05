# Overview
This is a repo for reinforcement learning sim2real rotation demo on SharpaWave, provides a step-by-step guide for training, visualizing and deploying.

<p align="center">
  <img src="resources/sim.gif" width="45%" />
  <img src="resources/real.gif" width="45%" />
</p>

# 1. Environment Setup
## 1.1. Follow the official  Isaaclab installation guide:
Install [Isaac Lab v2.3.2](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html).

Ubuntu 20.04/22.04, conda environment, Isaac Sim 4.5.0 and Isaac Lab v2.3.2 are supported. Isaac Lab 2.2 is no longer supported because it does not provide tactile contact positions.

CAUTION⚠️: A minimum of 32GB RAM is required. For specific requirements, please refer to [requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
## 1.2. Install this repo:  
```bash
conda activate env_isaaclab 
cd sharpa_tac_rl 
pip install -e .
```

# 2. Training
## 2.0. Configure experiment tracking (wandb)
Training metrics go to [Weights & Biases](https://wandb.ai). Edit `wandb_config.json` in the repo root:
```json
{
  "api_key": "your-wandb-api-key",
  "project": "sharpa",
  "entity": null,
  "mode": "online",
  "enabled": true
}
```
```bash
# INFOℹ️: Get your key from https://wandb.ai/authorize (40 hex characters, no prefix).
# INFOℹ️: "entity": null uses the account the key belongs to.
# INFOℹ️: "mode" accepts "online", "offline" or "disabled".
#         Use "offline" on a machine without internet, then upload later:
#           wandb sync logs/<experiment>/<timestamp>/stage1_wandb/wandb/offline-run-*
# INFOℹ️: Set "enabled": false to turn logging off entirely.
# INFOℹ️: Standard env vars override this file, which is handy for a one-off run:
#           WANDB_API_KEY=... WANDB_PROJECT=... WANDB_MODE=offline python .../train.py ...
# INFOℹ️: Point SHARPA_WANDB_CONFIG at another file to keep your key outside the repo.
```
Stage 1 and stage 2 of one experiment share a wandb group (`<experiment_name>/<timestamp>`),
so the PPO run and its distillation run appear together. If wandb is unavailable, training
prints a warning and continues without logging.

Offline check of the logging path (no simulator or GPU needed):
```bash
python tests/test_wandb_logger.py
```

## 2.1. Generate grasp cache
```bash
# CAUTION⚠️: Same object scale config will overwrite the older one.
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
```
## 2.2. Train the policy
```bash
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless
```
## 2.3. Distillation
```bash
# last.pth is recommended if curriculum is enabled
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --algorithm ProprioAdapt --load_path ${pth}
```

## 2.4. Bulb ring task
The ring task keeps the bulb fully free under gravity and uses a physical guide
collider instead of a revolute joint.

```bash
# Validate the ring asset, reward terms, gravity, collision, and scale matching.
python rl_isaaclab/scripts/smoke_test_bulb_ring.py --headless --device cuda:0

# Train with the five matched bulb/ring scales.
python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-Ring-v0 \
  --headless --num_envs 8192
```

# 3. Visualization
## 3.1. Visualize trained policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path ${pth}
```
## 3.2. Visualize distillated policy
```bash
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --algorithm ProprioAdapt --load_path ${pth}
```
## 3.3. Visualize bulb grasp cache
```bash
python rl_isaaclab/scripts/visualize_bulb_cache.py --num_envs 16
```

# 4. Deploy
## 4.1. Prepare SharpaWave and object
1. Calibrate SharpaWave through SharpaPilot. 
2. A cylinder with radius of 24mm and height of 60mm via 3D priting is recommended under default configuration.
## 4.2. Deploy on SharpaWave (HostComputer Tactile, Recommended)
### 4.2.1. Configure docker
```bash
# INFOℹ️: Install docker and nvidia-ctk following steps 1-4 in <Steps to Acquire 180 Hz High-Frame-Rate High-Performance Tactile Information>.
# Choose the docker config according to your CUDA version:
#   CUDA 12.4 -> rl_isaaclab/utils/docker/cu124
#   CUDA 12.8 -> rl_isaaclab/utils/docker/cu128
cd rl_isaaclab/utils/docker/cu124
# cd rl_isaaclab/utils/docker/cu128
export SHARPAWAVE_RL_LAB=$(git rev-parse --show-toplevel)
xhost +local:root
USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d
docker exec -it sharpawave_rl_dev bash
rm -r ~/sharpawave-rl-lab/rl_isaaclab/utils/python/
cp -r ~/sharpa-wave-sdk/python/sharpa/ ~/sharpawave-rl-lab/rl_isaaclab/utils/python/
cd ~/sharpawave-rl-lab/
python3 -m pip install -e .
```
### 4.2.2. Deploy
```bash
# INFOℹ️: Keyboard control is enabled by default. Press 'e' to start, press 'w' to freeze, press 'q' to go home.
python3 rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --hand_side ${0/1} --load_path ${pth}
```
## 4.3. Deploy on SharpaWave (OnBoard Tactile)
### 4.3.1. Configure SharpaWaveSDK (For Deploy)
```bash
# INFOℹ️: Install SharpaWaveSDK following the official user manual. ${SharpaWaveSDK} is the root path of the SDK.
rm -r rl_isaaclab/utils/python
cp -r ${SharpaWaveSDK}/python rl_isaaclab/utils/python
```
### 4.3.2. Deploy
```bash
# INFOℹ️: Keyboard control is enabled by default. Press 'e' to start, press 'w' to freeze, press 'q' to go home.
python rl_isaaclab/scripts/deploy.py --task Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0 --enable_on_board --hand_side ${0/1} --load_path ${pth}
```

# 5. Configure your own task via modifying the config file
Please refer to rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py and rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py for details.
