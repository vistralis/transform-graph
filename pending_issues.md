# Pending Issues & Architectural Inconsistencies

This document tracks identified flaws, silent error sources, and mathematical inconsistencies in the `tgraph` library that need to be addressed to ensure production-grade rigor.

## 1. `transform_points` vs. `apply()` Inconsistency
**Issue:** The global `tf.transform_points(transform, points)` function simply performs a 4x4 matrix multiplication. It does **not** handle perspective division for `Projection` objects.
*   `Projection.apply(points)` -> Returns Nx2 $(u, v)$ (Correct).
*   `tf.transform_points(projection, points)` -> Returns Nx3 $(x, y, w)$ (Incorrect/Unexpected).

**Proposed Fix:** Move the `apply()` method into the `BaseTransform` abstract interface. Deprecate the global `transform_points` in favor of `transform.apply(points)`.

## 2. Invalid Dimensional Flow in Composition (`*`)
**Issue:** The `*` operator allows any two transforms to be multiplied if they are 4x4, regardless of their geometric input/output dimensions.
*   **Invalid:** `Transform * Projection`. This implies $T(P(x))$, where $P(x)$ is a 2D pixel but $T$ expects a 3D point.
*   **Valid:** `Projection * Transform`. This is $P(T(x))$, transform then project.

**Proposed Fix:** Add `input_dim` and `output_dim` properties to `BaseTransform`. Validate that `left.input_dim == right.output_dim` during composition.

## 3. Pose Frame-ID Mismatch
**Issue:** Composing two `Pose` objects (`pose_a * pose_b`) currently propagates frame IDs blindly without verifying if they are semantically linkable.
*   If `pose_a` is $T_{world \to base}$ and `pose_b` is $T_{camera \to image}$, `pose_a * pose_b` produces a result with `frame_id='world'` and `child_frame_id='image'`, which is semantically non-sensical.

**Proposed Fix:** Add a validation check in `Pose.__mul__`. Raise a `ValueError` if `self.child_frame_id != other.frame_id` (when both are defined).

## 4. `InverseProjection` Pseudo-inverse Ambiguity
**Issue:** `InverseProjection.as_matrix()` returns a Moore-Penrose pseudo-inverse. While useful for internal matrix algebra, users might mistakenly use it to project pixels back to 3D without providing depth, leading to geometrically incorrect results.

**Proposed Fix:** Ensure `as_matrix()` or a new `apply()` on `InverseProjection` raises an explicit error or warning stating that depth is required, directing users to `unproject(pixels, depths)`.

## 5. Strict Type Validation in Constructors
**Issue:** While `**kwargs` were removed from `Rotation` and `Translation`, other constructors (like `Transform` or `Projection`) might still allow loose inputs that don't match the expected SE(3) or Projective constraints.

**Proposed Fix:** Standardize validation across all constructors using the `ensure_translation` and `ensure_rotation` pattern established in `transform.py`.
