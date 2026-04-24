# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def _zeros(env, dim: int) -> torch.Tensor:
    return torch.zeros((env.num_envs, dim), dtype=torch.float, device=env.device)


def bimanual_joint_pos(env) -> torch.Tensor:
    left = env.scene["left_hand"].data.joint_pos
    right = env.scene["right_hand"].data.joint_pos
    return torch.cat([left, right], dim=-1)


def bimanual_joint_vel(env) -> torch.Tensor:
    left = env.scene["left_hand"].data.joint_vel
    right = env.scene["right_hand"].data.joint_vel
    return torch.cat([left, right], dim=-1)


def tactile_observation(env, dim: int = 90) -> torch.Tensor:
    tactile = getattr(env, "tactile_observation_cache", None)
    if tactile is None:
        return _zeros(env, dim)
    return tactile


def reference_observation(env, dim: int = 64) -> torch.Tensor:
    reference = getattr(env, "reference_observation_cache", None)
    if reference is None:
        return _zeros(env, dim)
    return reference


def object_state_observation(env) -> torch.Tensor:
    state = env.scene["object"].data.root_state_w
    return state


def object_tracking_error_observation(env) -> torch.Tensor:
    current = env.scene["object"].data.root_state_w[:, :13]
    reference = getattr(env, "reference_object_state_cache", None)
    if reference is None:
        return torch.zeros((env.num_envs, 13), device=env.device)
    return current - reference
