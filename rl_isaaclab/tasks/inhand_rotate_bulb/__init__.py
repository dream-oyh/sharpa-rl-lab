import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-v0",
    entry_point="rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_bulb_env_cfg:SharpaWaveBulbEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-Bulb-v0",
    entry_point="rl_isaaclab.tasks.inhand_rotate.sharpa_wave_grasp_env:SharpaWaveInhandRotateGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_bulb_grasp_env_cfg:SharpaWaveBulbGraspEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-Socket-v0",
    entry_point=(
        "rl_isaaclab.tasks.inhand_rotate_bulb.sharpa_wave_bulb_socket_env:"
        "SharpaWaveInhandRotateBulbSocketEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.sharpa_wave_bulb_socket_env_cfg:SharpaWaveBulbSocketEnvCfg"
        ),
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_socket_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-Bulb-Ring-v0",
    entry_point=(
        "rl_isaaclab.tasks.inhand_rotate_bulb.sharpa_wave_bulb_ring_env:"
        "SharpaWaveInhandRotateBulbRingEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.sharpa_wave_bulb_ring_env_cfg:SharpaWaveBulbRingEnvCfg"
        ),
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_ring_cfg.yaml",
    },
)
