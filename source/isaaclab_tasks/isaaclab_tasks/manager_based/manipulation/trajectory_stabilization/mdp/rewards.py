# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def _reference_object_state(env):
    ref = getattr(env, "reference_object_state_cache", None)
    if ref is None:
        ref = env.scene["object"].data.root_state_w[:, :13]
    return ref


def track_reference_object_pose_exp(env, std: float = 0.05) -> torch.Tensor:
    current = env.scene["object"].data.root_state_w[:, :7]
    target = _reference_object_state(env)[:, :7]
    error = torch.linalg.norm(current - target, dim=-1)
    return torch.exp(-(error**2) / (2 * std**2))


def track_reference_object_velocity_exp(env, std: float = 0.10) -> torch.Tensor:
    current = env.scene["object"].data.root_state_w[:, 7:13]
    target = _reference_object_state(env)[:, 7:13]
    error = torch.linalg.norm(current - target, dim=-1)
    return torch.exp(-(error**2) / (2 * std**2))


def stable_contact_reward(env) -> torch.Tensor:
    left = getattr(env.scene["left_hand_contacts"].data, "net_forces_w", None)
    right = getattr(env.scene["right_hand_contacts"].data, "net_forces_w", None)
    if left is None or right is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    left_mag = torch.linalg.norm(left.reshape(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
    right_mag = torch.linalg.norm(right.reshape(env.num_envs, -1, 3), dim=-1).sum(dim=-1)
    return torch.tanh(left_mag + right_mag)


def residual_action_l2(env) -> torch.Tensor:
    cache = getattr(env, "trajectory_residual_action_cache", None)
    if cache is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    components = torch.cat(
        [
            cache["left_wrist_delta"],
            cache["right_wrist_delta"],
            cache["left_joint_residual"],
            cache["right_joint_residual"],
        ],
        dim=-1,
    )
    return torch.sum(components**2, dim=-1)


def drop_penalty(env, minimum_height: float = 0.04) -> torch.Tensor:
    height = env.scene["object"].data.root_state_w[:, 2]
    return (height < minimum_height).float()
