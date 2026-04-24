Code Map
========

Documentation
-------------

* ``docs/source/overview/developer-guide/hx5_traj_stabilization/index.rst``
* ``docs/source/overview/developer-guide/hx5_traj_stabilization/design_goals.rst``
* ``docs/source/overview/developer-guide/hx5_traj_stabilization/architecture.rst``
* ``docs/source/overview/developer-guide/hx5_traj_stabilization/implementation_status.rst``
* ``docs/source/overview/developer-guide/hx5_traj_stabilization/code_map.rst``


Assets
------

* ``source/isaaclab_assets/isaaclab_assets/robots/hx5_d20.py``
  HX5-D20 left/right articulation configuration scaffolds.
* ``source/isaaclab_assets/data/Robots/HX5_D20/README.md``
  Records the expected local asset conversion outputs and provenance.


Task Package
------------

* ``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/interfaces.py``
  Lightweight abstractions for hand platforms, tactile backends, and trajectory datasets.
* ``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/trajectory_stabilization_env_cfg.py``
  Generic manager-based environment scaffold.
* ``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/mdp/``
  Action, observation, reward, termination, and curriculum helper modules.


HX5-Specific Task Config
------------------------

* ``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/config/hx5_d20/hx5_d20_env_cfg.py``
  HX5-specific task configuration built on the generic task base.
* ``.../config/hx5_d20/__init__.py``
  Gym registration for teacher, distillation, and student-finetune task IDs.
* ``.../config/hx5_d20/agents/rsl_rl_ppo_cfg.py``
  Recurrent PPO configs.
* ``.../config/hx5_d20/agents/rsl_rl_distillation_cfg.py``
  Recurrent teacher-student distillation configs.


Mimic
-----

* ``source/isaaclab_mimic/isaaclab_mimic/envs/trajectory_stabilization_mimic_env.py``
  Mimic wrapper for dual-wrist target pose conversion.
* ``source/isaaclab_mimic/isaaclab_mimic/envs/trajectory_stabilization_mimic_env_cfg.py``
  Mimic config using the HX5 teacher environment as the base task.


Build Helper
------------

* ``Makefile``
  Root-level convenience wrapper for Sphinx builds.
