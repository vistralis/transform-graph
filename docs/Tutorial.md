# tgraph Tutorial

This tutorial covers the core features of `tgraph` (package name `transform-graph`), a high-performance library for spatial transformations and frame graph management.

## Setup

First, import the necessary modules. If you are working in a Jupyter notebook, configure the Plotly renderer.

```python
import numpy as np
import tgraph.transform as tf
import tgraph.visualization as vis
import plotly.io as pio

# Configure plotly for notebook rendering (optional)
pio.renderers.default = 'notebook'

# Configure numpy printing
np.set_printoptions(precision=3, suppress=True)
```

## 1. Basic Transforms

The core of `tgraph` is the `Transform` class, which represents SE(3) rigid body transformations (rotation + translation).

### Euler Angle Convention (Roll-Pitch-Yaw)

tgraph uses the **aerospace/robotics convention** for Euler angles:

| Angle | Axis | Description |
|-------|------|-------------|
| **Roll (φ)** | X-axis | Banking left/right |
| **Pitch (θ)** | Y-axis | Nose up/down |
| **Yaw (ψ)** | Z-axis | Heading direction |

Rotations are applied in **intrinsic ZYX order** (yaw → pitch → roll).

```python
# Create a simple translation
translation = tf.Translation(x=1.0, y=2.0, z=3.0)

# Create a rotation (90 degrees yaw)
rotation = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi/2)

# Create a full SE(3) transform
transform = tf.Transform(
    translation=[1.0, 0.0, 0.5],
    rotation=tf.Rotation.from_roll_pitch_yaw(pitch=0.1).rotation
)
```

### Transforming Points

```python
points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
transformed = tf.transform_points(transform, points)
```

## 2. Poses with Frame Semantics

A `Pose` represents the position and orientation of a child frame relative to a parent frame.

```python
# Create a pose: base_link relative to map
pose = tf.Pose(
    position=[1.0, 2.0, 0.0],
    orientation=tf.Rotation.from_roll_pitch_yaw(yaw=0.5).rotation,
    frame_id='map',
    child_frame_id='base_link'
)

# Inverting a pose automatically swaps frame IDs
# Result: map relative to base_link
inverse_pose = pose.inverse()
print(inverse_pose.frame_id)       # 'base_link'
print(inverse_pose.child_frame_id) # 'map'
```

## 3. Transform Graphs

The `TransformGraph` manages complex relationships between multiple coordinate frames.

```python
graph = tf.TransformGraph()

# Build a simple kinematic chain
graph.add_transform('world', 'base', tf.Translation(z=0.5))
graph.add_transform('base', 'arm', tf.Transform(
    translation=[0.2, 0, 0.3],
    rotation=tf.Rotation.from_roll_pitch_yaw(pitch=0.5).rotation
))
graph.add_transform('arm', 'gripper', tf.Translation(x=0.4))

# Query any transform automatically
world_to_gripper = graph.get_transform('world', 'gripper')

# Graph automatically handles inversion and path composition
gripper_to_world = graph.get_transform('gripper', 'world')
```

## 4. Visualization

`tgraph` provides two types of visualization using Plotly.

### Spatial Visualization (3D)

Visualizes the actual coordinate frames in 3D space. Each frame shows its RGB axes (X=Red, Y=Green, Z=Blue).

```python
fig_3d = vis.visualize_transform_graph(graph, root_frame='world', axis_scale=0.2)
fig_3d.show()
```

### Topology Visualization (2D)

Visualizes the logical structure of the graph, showing how frames are connected, the types of transforms used, and highlighting cached shortcuts.

```python
fig_topo = vis.visualize_graph(graph)
fig_topo.show()
```

## 5. Camera Projections

Project 3D points to 2D image coordinates.

```python
# Create camera with intrinsics
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
camera = tf.CameraProjection(
    intrinsic_matrix=K,
    image_size=(640, 480),
)

# Project points
points_3d = np.array([[0, 0, 5], [1, 0, 5]])
pixels = tf.project_points(camera, points_3d)
```

## 6. Serialization

Save and load graphs to/from JSON-compatible dictionaries.

```python
# Save
data = graph.to_dict()

# Load
new_graph = tf.TransformGraph.from_dict(data)
```
