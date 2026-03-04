# Deep Dive: MatrixTransform Proliferation in Transform Graphs

> [!NOTE]
> **This issue has been resolved.** The `CompositeProjection` and
> `InverseCompositeProjection` types now preserve intrinsic/extrinsic
> separation during graph traversal. This document is retained as
> architectural reference.

## Issue Description

Users may observe `MatrixTransform` objects appearing in the `TransformGraph`
when querying transforms between certain frames, even if the user only
explicitly added rigid `Transform` objects.

Specifically, querying the transform from a **2D Image Frame** (e.g.,
`cam_front`) to a **3D Frame** (e.g., `global`, `ego`, or `lidar`) returns
a `MatrixTransform` instead of a `Transform`.

## Architectural Cause

This behavior is mathematically correct and expected due to the inclusion of
**Projections** in the graph.

1. **Sensors as Projections**: Camera sensors are modeled using
   `CameraProjection` (3D → 2D).
2. **Graph Traversal**: To go from a 2D image pixel to a 3D world point,
   the graph must traverse the projection edge in reverse.
3. **Inverse Projection**: Determining a 3D ray from a 2D pixel requires an
   `InverseProjection`. This operation is **not a rigid body transformation**
   (it is a 3×4 or 4×4 homogeneous matrix operation).
4. **Composition**: The chain of transforms looks like:

   $$T_{\text{total}} = T_{\text{ego→global}} \cdot T_{\text{cam→ego}} \cdot P^{-1}_{\text{img→cam}}$$

   Since $P^{-1}$ is not a member of the Special Euclidean group SE(3), the
   result cannot be represented by the rigid `Transform` class.

## Resolution

The `CompositeProjection` system (Option E from the
[composition analysis](architecture/projection_composition_analysis.md))
was implemented:

| Composition | Result Type | Preserved Info |
|---|---|---|
| `Transform × Projection` | `CompositeProjection` | Extrinsics + Intrinsics |
| `Transform × InverseProjection` | `InverseCompositeProjection` | Extrinsics + InverseIntrinsics |

This means `get_transform("cam_front", "global")` now returns a structured
`InverseCompositeProjection` that visualization tools can decompose into
pose and intrinsics, rather than an opaque matrix.
