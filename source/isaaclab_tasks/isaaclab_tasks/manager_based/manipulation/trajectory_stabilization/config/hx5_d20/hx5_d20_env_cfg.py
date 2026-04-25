# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_assets.robots import HX5_D20_LEFT_CFG, HX5_D20_RIGHT_CFG

from ...trajectory_stabilization_env_cfg import ObservationsCfg, TrajectoryStabilizationEnvCfg


@configclass
class HX5CubeTeacherEnvCfg(TrajectoryStabilizationEnvCfg):
    """Teacher environment with privileged critic observations."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.left_hand = HX5_D20_LEFT_CFG.replace(prim_path="{ENV_REGEX_NS}/LeftHand")
        self.scene.right_hand = HX5_D20_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/RightHand")
        self.scene.num_envs = 512
        self.scene.env_spacing = 1.25
        self.observations.student_policy = None
        self.observations.teacher_policy = None


class HX5CubeTeacherEnvCfg_PLAY(HX5CubeTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
        self.events.push_object = None


@configclass
class HX5CubeDistillEnvCfg(HX5CubeTeacherEnvCfg):
    """Distillation environment exposing both student and teacher observation groups."""

    def __post_init__(self):
        super().__post_init__()

        self.observations = ObservationsCfg()
        self.observations.policy = None
        self.observations.critic = None


@configclass
class HX5CubeStudentEnvCfg(HX5CubeTeacherEnvCfg):
    """Student fine-tuning environment with deployable observation groups only."""

    def __post_init__(self):
        super().__post_init__()

        self.observations = ObservationsCfg()
        self.observations.policy = self.observations.student_policy
        self.observations.critic = self.observations.student_policy
        self.observations.student_policy = None
        self.observations.teacher_policy = None
