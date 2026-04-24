# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lightweight abstractions for trajectory-stabilization task scalability."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class HandPlatformSpec:
    """Metadata for a dexterous hand platform used by this task family."""

    name: str
    num_joints_per_hand: int
    left_asset_name: str
    right_asset_name: str
    wrist_frame_names: tuple[str, str] = ("left_wrist", "right_wrist")


@dataclass(slots=True)
class TactileBackendSpec:
    """Metadata for a tactile observation backend."""

    name: str
    observation_dim: int
    modality: str = "pressure"
    notes: str = ""


@dataclass(slots=True)
class TrajectoryDatasetSpec:
    """Metadata for the trajectory reference source."""

    name: str
    format_name: str = "hdf5"
    supports_per_env_sampling: bool = True
    pairs_object_and_trajectory: bool = True
    required_fields: list[str] = field(
        default_factory=lambda: [
            "left_wrist_pose",
            "right_wrist_pose",
            "left_hand_joint_targets",
            "right_hand_joint_targets",
            "object_pose",
        ]
    )
