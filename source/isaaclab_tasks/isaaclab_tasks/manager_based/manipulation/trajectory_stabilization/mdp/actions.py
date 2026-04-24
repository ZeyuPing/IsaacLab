# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class TrajectoryResidualActionCfg(ActionTermCfg):
    """Configuration for trajectory-conditioned residual actions."""

    class_type: type[ActionTerm] = MISSING
    left_hand_asset_name: str = MISSING
    right_hand_asset_name: str = MISSING
    left_joint_names: list[str] = MISSING
    right_joint_names: list[str] = MISSING
    wrist_pos_scale: float = 0.02
    wrist_rot_scale: float = 0.10
    joint_scale: float = 0.10


class TrajectoryResidualAction(ActionTerm):
    """Split policy outputs into left/right wrist residuals and joint residuals.

    Runtime note
    ------------

    This scaffold establishes the manager/action boundary expected by the task architecture.
    The actual wrist target tracking controller still needs to be finalized once the Isaac
    runtime is available for end-to-end validation.
    """

    cfg: TrajectoryResidualActionCfg
    _asset: Articulation

    def __init__(self, cfg: TrajectoryResidualActionCfg, env: ManagerBasedEnv):
        # Delay class_type self-reference until runtime import completes.
        if cfg.class_type is MISSING:
            cfg.class_type = TrajectoryResidualAction
        super().__init__(cfg, env)

        self._left_hand: Articulation = env.scene[cfg.left_hand_asset_name]
        self._right_hand: Articulation = env.scene[cfg.right_hand_asset_name]
        self._left_joint_ids, _ = self._left_hand.find_joints(cfg.left_joint_names)
        self._right_joint_ids, _ = self._right_hand.find_joints(cfg.right_joint_names)

        self._left_joint_dim = len(self._left_joint_ids)
        self._right_joint_dim = len(self._right_joint_ids)
        self._action_dim = 12 + self._left_joint_dim + self._right_joint_dim

        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._left_wrist_delta = torch.zeros((self.num_envs, 6), device=self.device)
        self._right_wrist_delta = torch.zeros((self.num_envs, 6), device=self.device)
        self._left_joint_residual = torch.zeros((self.num_envs, self._left_joint_dim), device=self.device)
        self._right_joint_residual = torch.zeros((self.num_envs, self._right_joint_dim), device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        cursor = 0
        self._left_wrist_delta[:] = actions[:, cursor : cursor + 6]
        cursor += 6
        self._right_wrist_delta[:] = actions[:, cursor : cursor + 6]
        cursor += 6
        self._left_joint_residual[:] = actions[:, cursor : cursor + self._left_joint_dim] * self.cfg.joint_scale
        cursor += self._left_joint_dim
        self._right_joint_residual[:] = actions[:, cursor : cursor + self._right_joint_dim] * self.cfg.joint_scale

        # Keep the controller boundary explicit for later runtime work.
        self._env.trajectory_residual_action_cache = {
            "left_wrist_delta": self._left_wrist_delta.clone(),
            "right_wrist_delta": self._right_wrist_delta.clone(),
            "left_joint_residual": self._left_joint_residual.clone(),
            "right_joint_residual": self._right_joint_residual.clone(),
        }

    def apply_actions(self):
        nominal = getattr(self._env, "nominal_replay_cache", {})
        left_nominal = nominal.get("left_joint_targets", self._left_hand.data.default_joint_pos[:, self._left_joint_ids])
        right_nominal = nominal.get(
            "right_joint_targets", self._right_hand.data.default_joint_pos[:, self._right_joint_ids]
        )

        self._left_hand.set_joint_position_target(left_nominal + self._left_joint_residual, joint_ids=self._left_joint_ids)
        self._right_hand.set_joint_position_target(
            right_nominal + self._right_joint_residual, joint_ids=self._right_joint_ids
        )

        # A future controller hook can consume these desired wrist deltas.
        self._env.desired_wrist_residual_targets = {
            "left": self._left_wrist_delta.clone(),
            "right": self._right_wrist_delta.clone(),
        }


TrajectoryResidualActionCfg.class_type = TrajectoryResidualAction
