# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Traj-Stabilize-Cube-HX5-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_d20_env_cfg:HX5CubeTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HX5TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Traj-Stabilize-Cube-HX5-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_d20_env_cfg:HX5CubeTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HX5TeacherPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Traj-Stabilize-Cube-HX5-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_d20_env_cfg:HX5CubeDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:HX5DistillationRunnerRecurrentCfg"
        ),
    },
)

gym.register(
    id="Isaac-Traj-Stabilize-Cube-HX5-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hx5_d20_env_cfg:HX5CubeStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HX5StudentPPORunnerCfg",
    },
)
