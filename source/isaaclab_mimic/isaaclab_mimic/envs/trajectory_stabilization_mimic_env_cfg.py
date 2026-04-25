# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.trajectory_stabilization.config.hx5_d20.hx5_d20_env_cfg import (
    HX5CubeTeacherEnvCfg,
)


@configclass
class HX5TrajectoryStabilizationMimicEnvCfg(HX5CubeTeacherEnvCfg, MimicEnvCfg):
    """Mimic config scaffold for the HX5 trajectory stabilization task."""

    left_hand_action_dim = 20
    right_hand_action_dim = 20

    def __post_init__(self):
        super().__post_init__()

        self.subtask_configs = {
            "left_wrist": [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="stabilize",
                    description="Follow the left-hand reference while maintaining object stability.",
                )
            ],
            "right_wrist": [
                SubTaskConfig(
                    object_ref="object",
                    subtask_term_signal="stabilize",
                    description="Follow the right-hand reference while maintaining object stability.",
                )
            ],
        }
