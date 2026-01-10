#!/usr/bin/env python3
"""
Demonstration of what TransformGraph should do.
This shows the intended API and use cases.
"""

import numpy as np
import tgraph.transform as tf

print("=" * 70)
print("Transform Graph Demo - What We Want to Build")
print("=" * 70)

# Example robotics scenario:
# We have a robot with:
# - world: The global reference frame
# - base: Robot base
# - camera: Camera mounted on robot
# - object: An object detected in the camera view

print("\n" + "=" * 70)
print("SCENARIO: Robot with Camera Tracking an Object")
print("=" * 70)

# Define the transforms manually for now
print("\n1. Define transforms between frames:")
print("-" * 70)

# Transform from world to robot base (robot is at x=2, y=1 in world)
world_to_base = tf.Translation(x=2.0, y=1.0, z=0.0)
print(f"world → base: {world_to_base}")

# Transform from base to camera (camera mounted 0.5m above, facing forward)
base_to_camera = tf.Transform(
    translation=[0.5, 0.0, 0.5],  # 0.5m forward, 0.5m up
    rotation=[1.0, 0.0, 0.0, 0.0]  # No rotation for simplicity
)
print(f"base → camera: {base_to_camera}")

# Object detected at 1m in front of camera
camera_to_object = tf.Translation(x=1.0, y=0.0, z=0.0)
print(f"camera → object: {camera_to_object}")

# Manual composition to find object in world frame
print("\n2. Manual composition (what we have now):")
print("-" * 70)

world_to_camera = world_to_base * base_to_camera
print(f"world → camera (composed): {world_to_camera}")

world_to_object = world_to_camera * camera_to_object
print(f"world → object (composed): {world_to_object}")

# Apply to a point
object_point_in_object_frame = np.array([[0.0, 0.0, 0.0]])  # Origin of object frame
object_point_in_world = tf.transform_points(world_to_object, object_point_in_object_frame)
print(f"\nObject position in world frame: {object_point_in_world[0]}")

print("\n" + "=" * 70)
print("WHAT TransformGraph SHOULD PROVIDE")
print("=" * 70)

print("""
# Create graph
transform_graph = tf.TransformGraph()

# Add edges (transforms between frames)
transform_graph.add_transform("world", "base", world_to_base)
transform_graph.add_transform("base", "camera", base_to_camera)
transform_graph.add_transform("camera", "object", camera_to_object)

# Query any transform automatically!
world_to_object = transform_graph.get_transform("world", "object")
# ^ Automatically finds path and composes transforms

# Also works backwards (using inverse)
object_to_world = transform_graph.get_transform("object", "world")
# ^ Automatically inverts as needed

# Update transforms (e.g., robot moved)
transform_graph.update_transform("world", "base", new_transform)
# ^ Invalidates cached shortcuts that depend on this edge

# Clear all cached shortcuts (useful when doing bulk updates)
transform_graph.clear_cache()
# ^ Removes all shortcut edges, forcing recomputation on next query

# Serialize/deserialize
data = transform_graph.to_dict()
graph2 = tf.TransformGraph.from_dict(data)
""")

print("\n" + "=" * 70)
print("KEY FEATURES TO IMPLEMENT")
print("=" * 70)

features = [
    "1. Undirected graph with directional metadata",
    "2. Lazy inversion (compute inverse on-the-fly)",
    "3. Shortest path finding using networkx",
    "4. Shortcut caching for O(1) repeated queries",
    "5. Dependency-aware cache invalidation",
    "6. clear_cache() for bulk cache clearing",
    "7. Serialization to/from JSON-compatible dict",
    "8. Support for Projection edges (one-way only)",
]

for feature in features:
    print(f"  ✓ {feature}")

print("\n" + "=" * 70)
print("Ready to implement TransformGraph!")
print("=" * 70)
