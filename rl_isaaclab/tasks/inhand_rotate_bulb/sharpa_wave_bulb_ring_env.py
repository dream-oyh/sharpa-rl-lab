# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from .sharpa_wave_bulb_ring_env_cfg import SharpaWaveBulbRingEnvCfg
from .sharpa_wave_bulb_socket_env import SharpaWaveInhandRotateBulbSocketEnv


class SharpaWaveInhandRotateBulbRingEnv(SharpaWaveInhandRotateBulbSocketEnv):
    """Free bulb guided by a low-friction physical ring instead of a joint."""

    cfg: SharpaWaveBulbRingEnvCfg

    def _create_socket_constraint(self):
        """The ring asset itself supplies the only bulb--socket constraint."""
        return
