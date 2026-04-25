Architecture
============

System Overview
---------------

The V1 implementation is split into five layers:

1. **Asset layer**
   Robot and object asset configuration, beginning with HX5-D20 left and right hand scaffolds.
2. **Task layer**
   A manager-based manipulation environment that defines scene setup, observations, rewards,
   curriculum, and action layout.
3. **Reference layer**
   A trajectory-reference interface that provides the nominal wrist and hand targets for replay.
4. **Learning layer**
   Recurrent PPO and recurrent distillation configurations following official Isaac Lab RSL-RL patterns.
5. **Mimic layer**
   A Mimic-compatible environment wrapper for future demo recording and replay workflows.


Task Structure
--------------

The TacAda task package introduced in this implementation is:

``source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/trajectory_stabilization/``

It contains:

* generic task configuration and interfaces
* MDP helper modules for actions, observations, rewards, terminations, and curriculum
* an HX5-specific config package under ``config/hx5_d20/``

This follows the same general structure used by official manager-based tasks such as
the in-hand, lift, reach, and Dexsuite environments.


Observation Design
------------------

V1 observation design follows the "stay close to official patterns" directive.

The current observation split is:

* ``policy``:
  current reference, tactile vector, joint position, joint velocity, and last action
* ``critic``:
  policy observations plus privileged object state and tracking error
* ``student_policy``:
  deployable observation subset for later sim-to-real oriented distillation
* ``teacher_policy``:
  privileged teacher view used during distillation

In V1, tactile is represented as a simple concatenated vector.
No explicit temporal tactile preprocessor is added yet; recurrence is expected to carry the history.


Action Design
-------------

The action layout is designed for trajectory-conditioned residual control:

* left wrist residual pose delta: 6
* right wrist residual pose delta: 6
* left hand joint residuals: ``N_left``
* right hand joint residuals: ``N_right``

The nominal replay controller provides the reference.
The residual policy provides small corrections on top.

In the current scaffold, the action term is implemented as a manager action
term so the package already has a clear place to evolve into a fully functional controller.


Curriculum Design
-----------------

The implemented V1 curriculum is intentionally narrow and uses the same
``EventTerm`` plus ``modify_term_cfg`` pattern used by official Isaac Lab examples:

* object disturbance velocity range ramps from zero to the configured maximum
* gravity distribution ramps from zero gravity to nominal Earth gravity

The important design choice is that the sampling unit is a **trajectory-object pair**.
One trajectory is not assumed to be reusable with arbitrary objects.

The next curriculum items are planned but not implemented in this scaffold:

* trajectory bank difficulty
* tactile corruption strength


Scalability Boundaries
----------------------

The task should remain generic across:

* hand model
* tactile backend
* reference dataset source

To support that, the implementation introduces interface dataclasses in the task package for:

* hand platform metadata
* tactile observation metadata
* trajectory dataset metadata

These interfaces are intentionally lightweight in V1.
