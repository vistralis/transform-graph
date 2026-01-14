# Pending Issues & Architectural Inconsistencies

This document tracks identified flaws, silent error sources, and mathematical inconsistencies in the `tgraph` library that need to be addressed to ensure production-grade rigor.

## 1. `transform_points` vs. `apply()` Inconsistency
**Issue:** The global `tf.transform_points(transform, points)` function simply performs a 4x4 matrix multiplication. It does **not** handle perspective division for `Projection` objects.
*   `Projection.apply(points)` -> Returns Nx2 $(u, v)$ (Correct).
*   `tf.transform_points(projection, points)` -> Returns Nx3 $(x, y, w)$ (Incorrect/Unexpected).

**Proposed Fix:** Move the `apply()` method into the `BaseTransform` abstract interface. Deprecate the global `transform_points` in favor of `transform.apply(points)`.

## 2. Invalid Dimensional Flow in Composition (`*`) [RESOLVED]
**Resolution:** Strict type checking implemented in `Transform.__mul__` to ban `Transform * CameraProjection`. `Projection.__mul__` correctly handles compositions with `MatrixTransform`.

## 3. Pose Frame-ID Mismatch
**Issue:** Composing two `Pose` objects (`pose_a * pose_b`) currently propagates frame IDs blindly without verifying if they are semantically linkable.
*   If `pose_a` is $T_{world \to base}$ and `pose_b` is $T_{camera \to image}$, `pose_a * pose_b` produces a result with `frame_id='world'` and `child_frame_id='image'`, which is semantically non-sensical.

**Proposed Fix:** Add a validation check in `Pose.__mul__`. Raise a `ValueError` if `self.child_frame_id != other.frame_id` (when both are defined).

## 4. `InverseProjection` Pseudo-inverse Ambiguity [RESOLVED]
**Resolution:** `InverseProjection` class implemented with explicit `unproject(pixels, depths)` method. `as_matrix()` returns pseudo-inverse for composition but unprojection requires depth.

## 5. Strict Type Validation in Constructors [RESOLVED]
**Resolution:** `ensure_translation` and `ensure_rotation` helpers are standardized and used in `Transform` and `Pose`. `CameraProjection` validates intrinsics shape.
