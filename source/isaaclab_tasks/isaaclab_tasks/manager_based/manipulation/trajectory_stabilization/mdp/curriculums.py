# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _linear_progress(step_count: int, num_steps: int, start_scale: float, end_scale: float) -> float:
    if num_steps <= 0:
        return end_scale
    alpha = min(max(step_count / float(num_steps), 0.0), 1.0)
    return start_scale + alpha * (end_scale - start_scale)


def linear_interpolate_value(env, env_ids, data, initial_value, final_value, num_steps: int):
    """Interpolate nested curriculum values using the official ``modify_term_cfg`` pattern."""

    alpha = _linear_progress(env.common_step_counter, num_steps, 0.0, 1.0)
    return _interpolate_nested(initial_value, final_value, data, alpha)


def _interpolate_nested(initial_value: Any, final_value: Any, data: Any, alpha: float):
    if isinstance(data, Mapping):
        return {
            key: _interpolate_nested(initial_value[key], final_value[key], value, alpha) for key, value in data.items()
        }
    if isinstance(data, list):
        return [
            _interpolate_nested(initial, final, item, alpha)
            for initial, final, item in zip(initial_value, final_value, data)
        ]
    if isinstance(data, tuple):
        return tuple(
            _interpolate_nested(initial, final, item, alpha)
            for initial, final, item in zip(initial_value, final_value, data)
        )

    value = float(initial_value) + alpha * (float(final_value) - float(initial_value))
    return int(value) if isinstance(data, int) else value
