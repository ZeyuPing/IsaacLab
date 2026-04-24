# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class _BaseHX5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 100
    clip_actions = 1.0
    experiment_name = "hx5_traj_stabilization"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HX5TeacherPPORunnerCfg(_BaseHX5PPORunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class HX5StudentPPORunnerCfg(_BaseHX5PPORunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
