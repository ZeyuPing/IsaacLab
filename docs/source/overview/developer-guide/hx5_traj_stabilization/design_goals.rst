Design Goals
============

Problem Framing
---------------

The first version of the task is **not** framed as generic grasp learning.
Instead, the environment should replay a demonstrated bimanual hand-object
interaction trajectory and learn a residual policy that keeps the object stable
while the replay is disturbed.

The task objective is therefore:

1. Replay nominal wrist poses and nominal hand joint targets from a reference trajectory.
2. Apply disturbances and model mismatch.
3. Learn residual corrections that preserve stable object behavior relative to the reference.


Confirmed V1 Decisions
----------------------

The following items were explicitly confirmed before implementation:

* Initial task: stable object behavior under disturbance while replaying a demonstrated trajectory.
* Nominal reference: replayed demonstration trajectory in wrist/object/hand space.
* Per-environment sampling: each environment samples one trajectory-object pair independently at reset.
* Control residuals: apply small residuals to both wrist pose and hand joints.
* Wrist control: dynamically tracked with a controller target, not kinematic teleportation.
* Tactile observations in V1: keep a simple concatenated tactile vector to stay close to official Isaac Lab patterns.
* Reference observations in V1: include current reference information only.
  Do not include next-step reference state or explicit phase signal.
* Curriculum direction: gravity should ramp from near-zero toward normal magnitude because gravity increases difficulty.
* Scalability constraint: do not hardcode the task architecture to HX5 only.
  Future hand platforms and future tactile backends should fit the same structure.
* Mimic compatibility: add the environment hooks from day one, but live teleoperation collection can remain a later step.
* Initial object for V1: cube.


Non-Goals For V1
----------------

The first implementation pass intentionally avoids the following:

* Full hardware-calibrated tactile modeling
* Visuo-tactile sensing
* Multi-object trajectory swapping within a single reference
* Full sim-to-real ROS integration
* Live Isaac teleoperation workflow validation on this machine

These are left for later phases once the basic architecture and code layout are in place.


Why The Architecture Stays Generic
----------------------------------

Even though HX5-D20 is the first concrete platform, this task is organized
around abstractions rather than around a single robot:

* **Hand platform abstraction**: left/right dexterous hands can be replaced later.
* **Tactile backend abstraction**: synthetic pressure tactile today, visuo-tactile later.
* **Trajectory dataset abstraction**: different demo collection or preprocessing pipelines can be plugged in later.

This keeps V1 aligned with the request for future extensibility while still
following the existing Isaac Lab package and task patterns.
