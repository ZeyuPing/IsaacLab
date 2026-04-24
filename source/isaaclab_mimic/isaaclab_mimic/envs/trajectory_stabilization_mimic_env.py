# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class TrajectoryStabilizationMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for dual-wrist trajectory stabilization tasks."""

    _EEF_TO_ASSET = {
        "left_wrist": "left_hand",
        "right_wrist": "right_hand",
    }

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        asset_name = self._EEF_TO_ASSET[eef_name]
        state = self.scene[asset_name].data.root_state_w[env_ids, :7]
        return PoseUtils.make_pose(state[:, :3], PoseUtils.matrix_from_quat(state[:, 3:7]))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        action_parts = []
        for eef_name in ("left_wrist", "right_wrist"):
            target_pose = target_eef_pose_dict[eef_name]
            target_pos, target_rot = PoseUtils.unmake_pose(target_pose)

            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            delta_position = target_pos - curr_pos
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

            pose_action = torch.cat([delta_position, delta_rotation], dim=0)
            if action_noise_dict is not None and eef_name in action_noise_dict:
                pose_action = pose_action + action_noise_dict[eef_name] * torch.randn_like(pose_action)
            action_parts.append(pose_action)

        left_hand_action = gripper_action_dict.get("left_hand", torch.zeros(20, device=self.device))
        right_hand_action = gripper_action_dict.get("right_hand", torch.zeros(20, device=self.device))
        action_parts.extend([left_hand_action, right_hand_action])
        return torch.cat(action_parts, dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        left_delta = action[:, :6]
        right_delta = action[:, 6:12]
        result = {}
        for eef_name, delta in (("left_wrist", left_delta), ("right_wrist", right_delta)):
            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            target_pos = curr_pos + delta[:, :3]
            delta_rotation = delta[:, 3:6]
            delta_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
            delta_axis = torch.where(
                delta_angle > 1.0e-8,
                delta_rotation / delta_angle,
                torch.zeros_like(delta_rotation),
            )
            delta_quat = PoseUtils.quat_from_angle_axis(delta_angle.squeeze(-1), delta_axis).squeeze(0)
            delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
            target_rot = torch.matmul(delta_rot_mat, curr_rot)
            result[eef_name] = PoseUtils.make_pose(target_pos, target_rot)
        return result

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "left_hand": actions[:, :, 12:32],
            "right_hand": actions[:, :, 32:52],
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        if hasattr(self, "subtask_term_signals_cache"):
            return {key: value[env_ids] for key, value in self.subtask_term_signals_cache.items()}
        return {"stabilize": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)[env_ids]}
