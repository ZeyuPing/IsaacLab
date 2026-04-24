# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from . import mdp


@configclass
class TrajectoryStabilizationSceneCfg(InteractiveSceneCfg):
    """Generic scene for trajectory-conditioned bimanual stabilization."""

    left_hand: ArticulationCfg = MISSING
    right_hand: ArticulationCfg = MISSING

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=CuboidCfg(
            size=(0.6, 0.6, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    left_hand_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeftHand/.*",
        history_length=4,
        update_period=0.0,
    )
    right_hand_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RightHand/.*",
        history_length=4,
        update_period=0.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications."""

    residual = mdp.TrajectoryResidualActionCfg(
        asset_name="left_hand",
        left_hand_asset_name="left_hand",
        right_hand_asset_name="right_hand",
        left_joint_names=[".*"],
        right_joint_names=[".*"],
        wrist_pos_scale=0.02,
        wrist_rot_scale=0.10,
        joint_scale=0.10,
    )


@configclass
class ObservationsCfg:
    """Observation specifications."""

    @configclass
    class PolicyCfg(ObsGroup):
        reference = ObsTerm(func=mdp.reference_observation, params={"dim": 64}, noise=Gnoise(std=0.0))
        tactile = ObsTerm(func=mdp.tactile_observation, params={"dim": 90}, noise=Gnoise(std=0.0))
        joint_pos = ObsTerm(func=mdp.bimanual_joint_pos, noise=Gnoise(std=0.001))
        joint_vel = ObsTerm(func=mdp.bimanual_joint_vel, noise=Gnoise(std=0.01), scale=0.2)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        reference = ObsTerm(func=mdp.reference_observation, params={"dim": 64})
        tactile = ObsTerm(func=mdp.tactile_observation, params={"dim": 90})
        joint_pos = ObsTerm(func=mdp.bimanual_joint_pos)
        joint_vel = ObsTerm(func=mdp.bimanual_joint_vel, scale=0.2)
        last_action = ObsTerm(func=base_mdp.last_action)
        object_state = ObsTerm(func=mdp.object_state_observation)
        tracking_error = ObsTerm(func=mdp.object_tracking_error_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentPolicyCfg(ObsGroup):
        reference = ObsTerm(func=mdp.reference_observation, params={"dim": 64})
        tactile = ObsTerm(func=mdp.tactile_observation, params={"dim": 90})
        joint_pos = ObsTerm(func=mdp.bimanual_joint_pos)
        joint_vel = ObsTerm(func=mdp.bimanual_joint_vel, scale=0.2)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherPolicyCfg(ObsGroup):
        reference = ObsTerm(func=mdp.reference_observation, params={"dim": 64})
        tactile = ObsTerm(func=mdp.tactile_observation, params={"dim": 90})
        joint_pos = ObsTerm(func=mdp.bimanual_joint_pos)
        joint_vel = ObsTerm(func=mdp.bimanual_joint_vel, scale=0.2)
        last_action = ObsTerm(func=base_mdp.last_action)
        object_state = ObsTerm(func=mdp.object_state_observation)
        tracking_error = ObsTerm(func=mdp.object_tracking_error_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    student_policy: StudentPolicyCfg = StudentPolicyCfg()
    teacher_policy: TeacherPolicyCfg = TeacherPolicyCfg()


@configclass
class EventCfg:
    """Randomization and disturbance events."""

    left_hand_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("left_hand", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    right_hand_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    object_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.6, 1.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    object_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    object_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )
    push_object = EventTerm(
        func=mdp.push_object_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for trajectory stabilization."""

    track_object_pose = RewTerm(func=mdp.track_reference_object_pose_exp, weight=2.0, params={"std": 0.05})
    track_object_velocity = RewTerm(func=mdp.track_reference_object_velocity_exp, weight=1.0, params={"std": 0.10})
    stable_contact = RewTerm(func=mdp.stable_contact_reward, weight=0.25)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1.0e-2)
    residual_magnitude = RewTerm(func=mdp.residual_action_l2, weight=-1.0e-3)
    drop_penalty = RewTerm(func=mdp.drop_penalty, weight=-5.0, params={"minimum_height": 0.04})


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.04, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms.

    The gravity ramp is intentionally represented by a dedicated helper instead of an inlined
    lambda so future runtime work has one obvious place to connect the actual physics-scene update.
    """

    disturbance_scale = CurrTerm(
        func=mdp.ramp_disturbance_scale,
        params={"num_steps": 50_000, "start_scale": 0.0, "end_scale": 1.0},
    )
    gravity_scale = CurrTerm(
        func=mdp.ramp_gravity_scale,
        params={"num_steps": 100_000, "start_scale": 0.0, "end_scale": 1.0},
    )


@configclass
class TrajectoryStabilizationEnvCfg(ManagerBasedRLEnvCfg):
    """Generic manager-based environment scaffold for trajectory-conditioned stabilization."""

    scene: TrajectoryStabilizationSceneCfg = TrajectoryStabilizationSceneCfg(
        num_envs=256,
        env_spacing=1.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.8, 1.8, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.2)
