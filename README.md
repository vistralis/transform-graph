# transform-graph

![CI](https://github.com/vistralis/transform-graph/actions/workflows/ci.yml/badge.svg)

**High-performance spatial transformations and frame graph management for robotics and computer vision.**

`transform-graph` (namespace `tgraph`) is the foundational mathematical layer for Spatial AI and Robotics in Python. It provides strict-typed handling of SE(3) rigid body transformations and projections.

**Target Environment:** Python 3.12+, NumPy 2.0+.

## Installation

```bash
pip install transform-graph
```

For visualization support (Plotly):
```bash
pip install "transform-graph[viz]"
```

## Usage

### Basic Transforms

```python
import tgraph.transform as tf
import numpy as np

# 1. Create simple transforms
# Translation: Move 1m in X
translation_x = tf.Translation(x=1.0)

# Rotation: 45 degrees yaw (heading) using aerospace convention
rotation_z = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi/4)

# 2. Compose transforms (Order matters!)
# Move then Rotate
combined_transform = translation_x * rotation_z

# 3. Transform points
points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
transformed_points = tf.transform_points(combined_transform, points)

print(f"Transformed Points:\n{transformed_points}")
```

### Transform Graph

```python
import tgraph.transform as tf

# Create a transform graph for a robot with a camera
graph = tf.TransformGraph()

# Define frame relationships (Source -> Target)
# Robot Base is 1m in X, 2m in Y relative to World
graph.add_transform('world', 'robot_base', tf.Translation(x=1.0, y=2.0))

# Camera is offset from Robot Base
graph.add_transform('robot_base', 'camera', tf.Transform(
    translation=[0.1, 0, 0.5],
    rotation=tf.Rotation.from_roll_pitch_yaw(pitch=-0.1).rotation
))

# Query transforms between any frames (auto-composes path)
world_to_camera = graph.get_transform('world', 'camera')

# Inverse traversal works automatically
camera_to_world = graph.get_transform('camera', 'world')
```

### Camera Projections

```python
import tgraph.transform as tf
import numpy as np

# Create a camera with intrinsic parameters
# Create a camera with intrinsic parameters (Strictly Intrinsic)
# Extrinsics (Position/Orientation) must be handled by a separate Transform.
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
camera = tf.CameraProjection(intrinsic_matrix=K, image_size=(640, 480))

# Project 3D points to 2D pixels (points must be in Camera Frame)
points_camera_frame = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]])
pixels = tf.project_points(camera, points_camera_frame)
print(f"Projected pixels:\n{pixels}")

# Unproject with known depth (returns points in Camera Frame)
inv_camera = camera.inverse()
depths = np.array([5.0, 5.0, 5.0])
points_recovered = inv_camera.unproject(pixels, depths)

# To handle Extrinsics, compose with a Transform
# Or use TransformGraph for automatic composition.
```

### Orthographic Projections

```python
import tgraph.transform as tf
import numpy as np

# Create a top-down (BEV) orthographic projection
# Maps 3D → 2D pixel coordinates without perspective division
ortho = tf.OrthographicProjection(
    axis="top",                    # "top" | "front" | "side"
    u_range=(-50, 50),             # column-axis extent (m)
    v_range=(-50, 50),             # row-axis extent (m)
    resolution=0.1,                # metres per pixel
)

# Register as a graph edge for unified transform_points API
graph = tf.TransformGraph()
graph.add_transform('ego', 'lidar', tf.Translation(x=2.0, z=1.0))
graph.add_transform('ego', 'bev', ortho)

# Project LiDAR points to BEV pixels — same API as camera projections
points_lidar = np.array([[5.0, 3.0, 0.5], [-2.0, -1.0, 0.0]])
pixels = tf.transform_points(points_lidar, graph, 'lidar', 'bev')[:, :2]

# Direct projection (without graph)
px = tf.transform_points(ortho, points_lidar)

# Inverse: lift pixel coordinates back to 3D (collapsed axis = 0)
pts_3d = tf.transform_points(ortho.inverse(), px)

# Grid metadata
print(f"Grid: {ortho.grid_shape}")       # (H, W) in pixels
print(f"Origin pixel: {ortho.origin_pixel}")  # (col, row) of world (0,0,0)
```

| Axis Preset | Drops | Col (u) | Row (v) | Use Case |
|-------------|-------|---------|---------|----------|
| `"top"` | Z | Y (left→right) | X (forward→back) | Bird's-eye view |
| `"front"` | X | Y (left→right) | Z (up→down) | Front elevation |
| `"side"` | Y | X (forward→back) | Z (up→down) | Side elevation |

## Transform Composition Rules

The library supports composing transforms with the `*` operator. The dimensional flow determines what operations are valid:

| Composition | Flow | Result | Use Case |
|-------------|------|--------|----------|
| `Transform * Transform` | 3D→3D→3D | `Transform` | Chain rigid body transforms |
| `Projection * Transform` | 3D→3D→2D | `Projection` | Project from any frame to image |
| `Transform * InverseProjection` | 2D→3D→3D | `MatrixTransform` | Unproject and transform rays |
| `Projection * InverseProjection` | 2D→3D→2D | `MatrixTransform` | Inter-image mapping |

**Key Principles:**

1. **Projections are one-way:** 3D→2D loses depth. Use `InverseProjection.unproject(pixels, depths)` when depth is known.

2. **Type degradation:** Composing SE(3) transforms with projections produces `MatrixTransform` or `Projection`, not `Transform`.

3. **No Homography type needed:** The Fundamental Matrix (`P₂ * T * P₁⁻¹`) maps points to epipolar *lines*, not points. Our `MatrixTransform` fallback correctly handles these cases.

### Epipolar Geometry
The library natively refines epipolar geometry from the graph structure:
```python
# Essential Matrix (E)
E = graph.get_essential_matrix("image1", "image2")

# Fundamental Matrix (F)
F = graph.get_fundamental_matrix("image1", "image2")

# Plane-Induced Homography (H)
H = graph.get_homography("image1", "image2", plane_normal=[0,0,1], plane_distance=1.0)
```

## Documentation & Tutorial

For a comprehensive guide on how to use `tgraph`, check out the [Tutorial](docs/Tutorial.md). It covers:
*   Creating and composing transforms
*   Managing complex frame graphs
*   3D spatial and 2D topology visualization
*   Camera models and projections
*   Serialization

### API Documentation

API documentation is auto-generated with [pdoc](https://pdoc.dev/) and published to
[vistralis.org/transform-graph/api](https://vistralis.org/transform-graph/api/tgraph.html).

To build locally:
```bash
pdoc --math -t docs/templates -o docs/build tgraph
```

## Development & Quality Control

We use `ruff` for linting/formatting and `pytest` for testing.

### 1. Installation

Install the project in editable mode with all development dependencies:
```bash
pip install -e ".[dev,viz]"
```

### 2. Linting & Formatting

We adhere to strict Python standards using **Ruff**.

**Check for issues:**
```bash
ruff check .
```

**Auto-fix linting issues and reformat code:**
```bash
ruff check . --fix
ruff format .
```

### 3. Testing & Coverage

We aim for high code coverage to ensure mathematical rigor.

**Run all tests:**
```bash
pytest
```

**Run tests with coverage report:**
```bash
pytest --cov=tgraph --cov-report=term-missing
```

## CI/CD Workflow

This project uses **GitHub Actions** for Continuous Integration.

*   **Workflow File:** `.github/workflows/ci.yml`
*   **Triggers:**
    *   Push to `main` branch.
    *   Pull Request to `main` branch.
*   **Jobs:**
    *   **Build & Test:**
        *   Sets up Python 3.12.
        *   Installs dependencies (including dev and viz extras).
        *   Runs the full test suite with coverage reporting.

## License

Apache 2.0 - Vistralis Labs