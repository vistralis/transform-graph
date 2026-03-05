# Agents — Engineering Guide for transform-graph v0.1.1

This document is the authoritative reference for AI agents and developers working
on or consuming the `transform-graph` library. All code changes must adhere to
these guidelines.

## Core Philosophy

1. **Strict Correctness**: Geometric operations must be mathematically rigorous.
   No "close enough" approximations for core transforms.
2. **Explicit Semantics**: Use `source_frame` and `target_frame` (never
   `parent`/`child`). `CameraProjection` is strictly intrinsic — extrinsics
   are separate `Transform` edges.
3. **Graph-First Design**: The graph is the source of truth. Connectivity
   determines validity. Edges are directed fundamental transforms; paths are derived.
4. **No Legacy Bloat**: Clean, modern Python 3.12+ code only. No 4×4 matrix init
   for Projections, no legacy `.apply()` methods.

---

## Architecture

### Package: `tgraph`

Single-module library at `src/tgraph/transform.py` with visualization in
`src/tgraph/visualization.py`.

### Public API (`__all__`)

#### Transforms (3D → 3D)

| Class | Description |
|-------|-------------|
| `Transform` | Full SE(3) rigid body transform (translation + quaternion rotation) |
| `Translation` | Pure translation (identity rotation) |
| `Rotation` | Pure rotation (zero translation) |
| `Identity` | Neutral element — composes with anything, returns the other operand |
| `MatrixTransform` | Generic 4×4 homogeneous matrix (fallback for mixed compositions) |

#### Projections (3D → 2D)

| Class | Description |
|-------|-------------|
| `CameraProjection` | Pinhole camera model with intrinsics K and optional distortion D |
| `OrthographicProjection` | Orthographic (BEV / front / side) projection at fixed resolution |
| `CompositeProjection` | Result of `Projection × Transform` — projects from any 3D frame |

#### Inverse Projections (2D → 3D)

| Class | Description |
|-------|-------------|
| `InverseCameraProjection` | Unprojects pixels to 3D rays (use `.unproject(pixels, depths)`) |
| `InverseOrthographicProjection` | Lifts pixels back to 3D on the projection plane |
| `InverseCompositeProjection` | Result of `Transform × InverseProjection` |

#### Graph & Pose

| Class | Description |
|-------|-------------|
| `TransformGraph` | Frame graph with BFS pathfinding and automatic composition |
| `Pose` | User-friendly wrapper for position + orientation in a named frame |

#### Free Functions

| Function | Description |
|----------|-------------|
| `transform_points(points, source, ...)` | Transform Nx3 points through graph or transform |
| `project_points(points, source, ...)` | Project Nx3 points to Nx2 pixels through projection |
| `from_roll_pitch_yaw(roll, pitch, yaw)` | Create a rotation from Euler angles (ZYX convention) |
| `as_roll_pitch_yaw(rotation)` | Extract (roll, pitch, yaw) tuple from rotation |
| `serialize_transform(transform)` | Serialize any transform to a dict |
| `deserialize_transform(data)` | Deserialize a transform from a dict |
| `register_transform(cls)` | Decorator to register a custom transform class |

---

## Composition Algebra

The `*` operator composes transforms. Dimensional flow determines valid operations:

| Composition | Flow | Result | Use Case |
|-------------|------|--------|----------|
| `Transform × Transform` | 3D→3D→3D | `Transform` | Chain rigid body transforms |
| `Projection × Transform` | 3D→3D→2D | `CompositeProjection` | Project from any frame |
| `Transform × InverseProjection` | 2D→3D→3D | `InverseCompositeProjection` | Unproject + reposition |
| `Projection × InverseProjection` | 2D→3D→2D | `MatrixTransform` | Inter-image mapping |

**Invalid compositions** (raise `TypeError`):
- `Transform × Projection` — dimensional mismatch
- `Projection × Projection` — cannot compose two projections

---

## Usage Guidelines for Consumers

When using `transform-graph` as a dependency:

1. **Strict Intrinsic Separation**: Never pass extrinsics (R, t) to
   `CameraProjection`. Use a separate `Transform` edge in the graph.
2. **Graph Traversal**: Always prefer `graph.get_transform(source, target)`
   over manual chain multiplication.
3. **Point Transformation**: Use `tf.transform_points()` and `tf.project_points()`
   free functions — never call internal `_apply()` methods.
4. **Euler Angles**: Use `from_roll_pitch_yaw()` / `as_roll_pitch_yaw()`.
   The convention is intrinsic ZYX order (yaw → pitch → roll).
5. **Epipolar Geometry**: Use `graph.get_fundamental_matrix()` and
   `graph.get_essential_matrix()` — do not implement manual calculations.
6. **Serialization**: Use `graph.to_dict()` / `TransformGraph.from_dict()` for
   JSON-compatible persistence.

### Temporal Graph Patterns

When building temporal graphs (e.g., from autonomous driving datasets):

1. Create one `ego_{timestamp}` frame per sample.
2. Chain consecutive ego frames via relative transforms:
   `T_rel = solve(M_{i+1}, M_i)` where M is the ego→global pose.
3. Anchor exactly one ego frame to the global/city frame.
4. Connect sensor frames to the ego frame using calibration transforms.
5. Use `graph.get_transform(ego_k, global)` to recover absolute poses.

---

## Developer Workflow

### Project Structure

```
src/tgraph/
├── __init__.py           # Re-exports from transform.py
├── transform.py          # All transform classes, graph, and algebra
└── visualization.py      # Plotly-based 3D and topology visualization

tests/
├── conftest.py           # Shared fixtures
├── test_composite.py     # CompositeProjection / InverseCompositeProjection
├── test_coverage_gaps.py # Edge cases and coverage improvement
├── test_euler.py         # Euler angle conversions
├── test_frame_ids.py     # Frame ID validation
├── test_multi_camera.py  # Multi-camera graph scenarios
├── test_orthographic_projection.py  # OrthographicProjection
├── test_pose.py          # Pose class
├── test_projection.py    # CameraProjection / InverseCameraProjection
├── test_serialization.py # Roundtrip serialization
├── test_transform.py     # Core Transform / Translation / Rotation
├── test_transform_graph.py  # TransformGraph operations
└── test_visualization.py # Visualization module

docs/
├── Tutorial.md           # User-facing tutorial
├── transforms_guidelines.md  # Mathematical conventions and architecture
├── known_issues.md       # Known limitations (MatrixTransform proliferation)
├── matrix_transform_explanation.md  # Deep dive on type degradation
└── architecture/
    └── projection_composition_analysis.md  # Design decision record
```

### Quality Gates

```bash
# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Test with coverage
pytest --cov=tgraph --cov-report=term-missing

# API documentation
pdoc --math -t docs/templates -o docs/build tgraph
```

**Current Metrics (v0.1.1):**
- Test files: 14, Tests: 265
- Coverage: 94% overall, 96% on `transform.py`
- Docstring coverage: 100% public API
- TODOs / FIXMEs: 0

### Coding Standards

Follow [Vistralis Coding Standards](https://github.com/vistralis):
- `snake_case` functions/variables, `PascalCase` classes
- Google-style docstrings with type hints on all function signatures
- Line length: 100 characters
- Format with `ruff` (line-length=100)
- No abbreviations: `measurement` not `meas`, `translation` not `trans`

### Commit Messages

Imperative mood, natural language. Start with a verb:
```
Add OrthographicProjection with BEV/front/side presets
Fix relative transform direction in temporal graph builder
Remove deprecated apply() method from public API
```

---

## Additional Documentation

- **[Tutorial](docs/Tutorial.md)** — Getting started guide with code examples
- **[Transforms Guidelines](docs/transforms_guidelines.md)** — Mathematical
  conventions and architectural constraints
- **[Known Issues](docs/known_issues.md)** — MatrixTransform proliferation
  and workarounds
- **[Projection Analysis](docs/architecture/projection_composition_analysis.md)** —
  Design decision record for CompositeProjection

## License

Apache 2.0 — Vistralis Labs