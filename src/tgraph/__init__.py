# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
# transform-graph — Spatial Transformations for Robotics and Computer Vision

`transform-graph` is the foundational mathematical layer for Spatial AI and Robotics in Python.
It provides strict-typed handling of SE(3) rigid body transformations, camera and
orthographic projections, and a frame graph for automatic path composition.

**Target Environment:** Python 3.10+, NumPy 2.0+.

---

## Key Classes

### Transforms (3D → 3D)

| Class | Description |
|-------|-------------|
| `Transform` | Full SE(3) rigid body transform (translation + quaternion rotation) |
| `Translation` | Pure translation (identity rotation) |
| `Rotation` | Pure rotation (zero translation) |
| `Identity` | Neutral element — composes with anything and returns the other operand |
| `MatrixTransform` | Generic 4×4 homogeneous matrix (fallback for mixed compositions) |

**Convenience constructors** on `Transform` and `Rotation`::

    Transform.from_rotation_matrix(R, t=None, validate=True)
    Transform.from_quaternion(q, t=None, convention="wxyz")
    Transform.from_axis_angle(axis, angle, t=None)

### Quaternion Interop (`tgraph.quaternion`)

Conversion between numpy-quaternion (wxyz), scipy Rotation, and raw arrays::

    from tgraph.quaternion import to_xyzw, from_xyzw, to_scipy, from_scipy, normalize

### Projections (3D → 2D)

| Class | Description |
|-------|-------------|
| `CameraProjection` | Pinhole camera model with intrinsics K and optional distortion D |
| `OrthographicProjection` | Orthographic (BEV / front / side) projection at fixed resolution |
| `CompositeProjection` | Result of `Projection × Transform` — projects from any 3D frame |

### Inverse Projections (2D → 3D)

| Class | Description |
|-------|-------------|
| `InverseCameraProjection` | Unprojects pixels to 3D rays (use `.unproject(pixels, depths)`) |
| `InverseOrthographicProjection` | Lifts pixels back to 3D on the projection plane |
| `InverseCompositeProjection` | Result of `Transform × InverseProjection` |

### Graph & Pose

| Class | Description |
|-------|-------------|
| `TransformGraph` | Frame graph with BFS pathfinding and automatic composition |
| `Pose` | User-friendly wrapper for position + orientation in a named frame |

---

## Composition Algebra

The `*` operator composes transforms. The dimensional flow determines valid operations:

| Composition | Flow | Result | Use Case |
|-------------|------|--------|----------|
| `Transform × Transform` | 3D→3D→3D | `Transform` | Chain rigid body transforms |
| `Projection × Transform` | 3D→3D→2D | `CompositeProjection` | Project from any frame |
| `Transform × InvProjection` | 2D→3D→3D | `InverseCompositeProjection` | Unproject + reposition |
| `Projection × InverseProjection` | 2D→3D→2D | `MatrixTransform` | Inter-image mapping |

**Invalid compositions** (raise `TypeError`):
- `InverseProjection × Transform` — dimensional mismatch (2D→3D then 3D→3D)
- `Projection × Projection` — cannot compose two projections

---

## Quick Start

```python
import tgraph.transform as tf
import numpy as np

# Build a frame graph
graph = tf.TransformGraph()
graph.add_transform('world', 'robot', tf.Translation(x=1.0, y=2.0))
graph.add_transform('robot', 'camera', tf.Transform(
    translation=[0.1, 0, 0.5],
    rotation=tf.Rotation.from_roll_pitch_yaw(pitch=-0.1).rotation,
))

# Add a camera projection edge
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
graph.add_transform('camera', 'image', tf.CameraProjection(K=K))

# Query any transform — automatic path composition
world_to_image = graph.get_transform('world', 'image')

# Project 3D world points to pixels
points_world = np.array([[2.0, 3.0, 1.0]])
pixels = tf.transform_points(points_world, graph, 'world', 'image')
```

---

Apache 2.0 — Vistralis Labs
"""

from .transform import (
    BaseTransform,
    CameraProjection,
    CompositeProjection,
    Identity,
    InverseCameraProjection,
    InverseCompositeProjection,
    InverseOrthographicProjection,
    InverseProjection,
    MatrixTransform,
    OrthographicProjection,
    Pose,
    Projection,
    ProjectionModel,
    Rotation,
    Transform,
    TransformGraph,
    Translation,
    as_roll_pitch_yaw,
    deserialize_transform,
    from_roll_pitch_yaw,
    project_points,
    register_transform,
    serialize_transform,
    transform_points,
)

__version__ = "0.1.1"

__all__ = [
    "BaseTransform",
    "Transform",
    "Translation",
    "Rotation",
    "Identity",
    "MatrixTransform",
    "Projection",
    "InverseProjection",
    "CameraProjection",
    "InverseCameraProjection",
    "OrthographicProjection",
    "InverseOrthographicProjection",
    "CompositeProjection",
    "InverseCompositeProjection",
    "ProjectionModel",
    "TransformGraph",
    "register_transform",
    "serialize_transform",
    "deserialize_transform",
    "from_roll_pitch_yaw",
    "as_roll_pitch_yaw",
    "Pose",
    "transform_points",
    "project_points",
]
