# SCA-AIFNav Baseline Freeze

## Frozen baseline

- Tag: `baseline-aimapp-aligned-v1`
- Commit: `900284bc486cead64920e3aa8a9952f24ecda2b0`
- Commit message: `feat: complete AIMAPP-aligned navigation baseline`
- Reference implementation: AIMAPP
- Purpose: freeze the AIMAPP-aligned baseline before SCA-specific innovations.

This tag is the rollback and comparison point for all subsequent SCA-AIFNav
research changes.

## Validation

### Core

- 534 passed
- 1 skipped
- 2 warnings

### ROS

- 321 passed
- 1 skipped
- 3 warnings

### Build

The following packages build successfully together:

- `sca_aifnav_core`
- `sca_aifnav_ros`
- `sca_aifnav_sim`

`sca_aifnav_sim` currently has no dedicated test directory.

## Gazebo validation

The complete runtime chain was verified:

Gazebo
-> odometry / LiDAR / three cameras
-> panorama acquisition
-> visual observation
-> cognitive-place observation
-> Active Inference state inference
-> MCTS planning
-> cognitive target prediction
-> Potential Field control
-> physical motion
-> next observation
-> A/B learning
-> posterior update
-> replanning

### Panorama

A complete panorama contains:

- 1 initial three-camera batch
- 3 rotated three-camera batches
- 4 batches total
- 12 images total

Visual stitching retry behavior was exercised successfully in Gazebo.

### Real physical navigation

The baseline demonstrated:

- initial heading alignment;
- forward motion;
- intermediate heading correction;
- approach-speed reduction;
- successful target completion.

A complete real closed loop was verified:

Observe -> Plan -> Act -> Observe -> Learn -> Replan

### Consecutive Gazebo actions

Three consecutive physical actions completed successfully.

Target-position errors:

- 0.049639 m
- 0.048863 m
- 0.048587 m

All were below the baseline 0.05 m completion tolerance.

Each subsequent decision correctly consumed the preceding executed action, and
the hidden-state belief remained normalized.

## Baseline conventions

### Action space

- Actions `0..11`: directional actions
- Action `12`: STAY
- 30 degree directional sectors
- Sector 0 center: 15 degrees
- Reverse directional action: offset by 6
- `reverse(12) = 12`

### Cognitive map

- Place-memory influence radius: 0.5 m
- Cognitive lookahead: 8
- Robot dimension: 0.3 m

The baseline influence radius is fixed.

Adaptive influence radius is reserved for the SCA innovation stage.

### Observation model

Two observation modalities are used:

- panoramic visual observation;
- cognitive-place observation.

The positional/place observation model `Ap` is included.

No `Bp` modality is introduced.

### Learning

Transition representation:

`B[next_state, previous_state, action]`

Real transition learning:

- direct transition: +10
- reverse transition: +7

Observation learning:

- pA evidence learning rate: +5

STAY is restored as an identity transition after updates.

### State inference

The baseline preserves the three-stage real-observation belief lifecycle:

preliminary belief
-> transition learning
-> learning belief
-> observation learning
-> final posterior belief

### Planning

Runtime MCTS parameters:

- simulations: 30
- rollout depth: 10
- exploration constant: 5

### Physical navigation

Potential Field baseline parameters include:

- maximum linear speed: 0.3 m/s
- maximum angular speed: 0.2 rad/s
- target tolerance: 0.05 m

The cognitive target is predicted before physical execution.

Physical-action failure triggers the baseline failure, rollback, return and
replanning path.

## Known runtime warnings

The frozen baseline may produce the following non-fatal warnings:

1. StitchingWarning indicating that not every supplied image is necessarily
   included in the final stitched panorama.

2. Initial visual stitching attempts may fail because of insufficient overlap
   or feature matching. The baseline reacquires the panorama with a reduced
   stitching confidence threshold.

3. Numba may disable its TBB threading layer because of the locally installed
   TBB interface version.

4. scikit-image may report the Python distutils deprecation warning.

5. Style tests may report the SelectableGroups dictionary-interface
   deprecation warning.

These warnings did not cause baseline regression failures.

## Freeze policy

SCA-specific innovations begin only after this baseline.

The frozen baseline must not be silently modified to introduce:

- spatial-complexity estimation;
- adaptive cognitive-node influence radius;
- adaptive cognitive-map density;
- SCA-specific planning heuristics.

The tag `baseline-aimapp-aligned-v1` must remain the reproducible comparison
point for subsequent SCA-AIFNav development.
