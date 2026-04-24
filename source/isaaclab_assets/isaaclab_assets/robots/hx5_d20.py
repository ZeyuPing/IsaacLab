# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration scaffolds for the HX5-D20 dexterous hands.

The following configurations are available:

* :obj:`HX5_D20_LEFT_CFG`: Left hand articulation scaffold
* :obj:`HX5_D20_RIGHT_CFG`: Right hand articulation scaffold

Notes
-----

These configurations intentionally point to local USD paths reserved for future
asset conversion. The USD files are not generated in this commit because Isaac Lab
is not installed on the current machine and Isaac conversion scripts must not be run.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


_HX5_D20_DATA_DIR = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "HX5_D20")


def _make_hx5_cfg(usd_filename: str) -> ArticulationCfg:
    usd_path = os.path.join(_HX5_D20_DATA_DIR, usd_filename)

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=1.0,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


HX5_D20_LEFT_CFG = _make_hx5_cfg("hx5_d20_left.usd")
"""Configuration scaffold for the left HX5-D20 hand."""


HX5_D20_RIGHT_CFG = _make_hx5_cfg("hx5_d20_right.usd")
"""Configuration scaffold for the right HX5-D20 hand."""
