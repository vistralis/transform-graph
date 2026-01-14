# Agents & Framework Philosophy

This document outlines the core principles of the `transform-graph` library. All AI agents working on this codebase must adhere to these guidelines to ensure consistency, rigor, and maintainability.

## Core Philosophy

1.  **Strict Correctness**: Geometric operations must be mathematically rigorous.
    *   No "close enough" approximations for core transforms.
    *   Strict type composition (e.g., `Transform * Projection -> Projection`).
2.  **Explicit Semantics**:
    *   Use `source_frame` and `target_frame` (never `parent`/`child`).
    *   `CameraProjection` is strictly intrinsic. Extrinsics are separate `Transform` nodes.
3.  **Graph-First Design**:
    *   The graph is the source of truth.
    *   Connectivity determines validity.
    *   Edges are directed fundamental transforms; paths are derived.
4.  **No Legacy Bloat**:
    *   We do not support legacy patterns (e.g., 4x4 matrix init for Projections).
    *   Clean, modern Python 3.12+ code only.

## Architectural Guidelines

For detailed mathematical and implementation guidelines, refer to:
*   [Transforms Guidelines](docs/transforms_guidelines.md)

## Usage Guidelines for Agents

When using `transform-graph` as a dependency in other projects:

1.  **Strict Intrinsic Separation**: Never attempt to pass extrinsics (`R`, `t`) to `CameraProjection`. You MUST use a separate `Transform` node in the graph.
2.  **Composition Rules**: 
    *   `Transform * Transform -> Transform`
    *   `Projection * Transform -> Projection`
    *   `Transform * Projection -> ERROR` (Left-multiply by pose only).
3.  **Graph traversal**: Always prefer `graph.get_transform(source, target)` over manual chain multiplication.
4.  **Epipolar Geometry**: Do not implement manual fundamental matrix calculations. Use `graph.get_fundamental_matrix()`.

## Developer Workflow

*   **Testing**: All new features must be covered by `pytest`. Use `test_projections.py` for intrinsic logic and `test_transform_graph.py` for topological logic.
*   **Visualization**: Use the `visualization` module debug tools (frustums, connectivity) to verify complex chains.