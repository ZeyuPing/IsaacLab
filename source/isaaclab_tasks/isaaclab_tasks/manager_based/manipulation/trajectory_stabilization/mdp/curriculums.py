# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def _linear_progress(step_count: int, num_steps: int, start_scale: float, end_scale: float) -> float:
    if num_steps <= 0:
        return end_scale
    alpha = min(max(step_count / float(num_steps), 0.0), 1.0)
    return start_scale + alpha * (end_scale - start_scale)


def ramp_disturbance_scale(env, env_ids, start_scale: float = 0.0, end_scale: float = 1.0, num_steps: int = 50_000):
    """Record a disturbance curriculum scale and return it for logging."""

    scale = _linear_progress(env.common_step_counter, num_steps, start_scale, end_scale)
    env.disturbance_curriculum_scale = scale
    return {"disturbance_curriculum_scale": scale}


def ramp_gravity_scale(env, env_ids, start_scale: float = 0.0, end_scale: float = 1.0, num_steps: int = 100_000):
    """Record a gravity curriculum scale and expose the desired gravity vector.

    Runtime note
    ------------

    The actual physics-scene update must be wired once the task is executed inside an
    Isaac runtime. This helper already computes and stores the intended scale so the
    controller logic and logs can rely on a stable field name.
    """

    scale = _linear_progress(env.common_step_counter, num_steps, start_scale, end_scale)
    base_gravity = torch.tensor([0.0, 0.0, -9.81], device=env.device)
    env.gravity_curriculum_scale = scale
    env.desired_gravity_vector = base_gravity * scale
    return {"gravity_curriculum_scale": scale}


def push_object_by_setting_velocity(env, env_ids=None, velocity_range: dict[str, tuple[float, float]] | None = None):
    """Simple placeholder disturbance hook for the manipulated object."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if velocity_range is None:
        velocity_range = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)}

    env.pending_object_push = {
        "env_ids": env_ids,
        "velocity_range": velocity_range,
    }
