#!/usr/bin/env python3
"""
Test the Projection classes and serialization system.
"""

import numpy as np
import tgraph.transform as tf
import pytest

def test_serialization_registry():
    """Test that all transform types are registered."""
    
    # Check registered types
    from tgraph.transform import _TRANSFORM_REGISTRY
    
    expected_types = [
        "Transform", "MatrixTransform", 
        "Projection", "InverseProjection",
        "CameraProjection",
    ]
    
    for t in expected_types:
        assert t in _TRANSFORM_REGISTRY, f"Missing: {t}"


def test_serialize_deserialize_all_types():
    """Test serialize_transform/deserialize_transform for all types."""
    
    # Create one of each type
    transforms = {
        "Transform": tf.Transform(
            translation=[1.0, 2.0, 3.0],
            rotation=[0.707, 0.0, 0.0, 0.707]
        ),
        "MatrixTransform": tf.MatrixTransform(np.eye(4)),
        "Projection": tf.Projection(
            np.array([
                [500, 0, 320, 0],
                [0, 500, 240, 0],
                [0, 0, 1, 0]
            ])
        ),
        "CameraProjection": tf.CameraProjection(
            intrinsic_matrix=np.array([
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1]
            ]),
            rotation_matrix=np.eye(3),
            translation=np.array([0, 0, 0])
        ),
    }
    
    for name, original in transforms.items():
        # Serialize
        data = tf.serialize_transform(original)
        
        # Deserialize
        restored = tf.deserialize_transform(data)
        
        # Verify matrices match
        assert np.allclose(original.as_matrix(), restored.as_matrix()), f"{name} matrix mismatch!"
    
    # Test inverse types
    projection = transforms["Projection"]
    inv_projection = projection.inverse()
    data = tf.serialize_transform(inv_projection)
    restored_inv = tf.deserialize_transform(data)
    assert isinstance(restored_inv, tf.InverseProjection)
    
    # CameraProjection.inverse() returns InverseProjection (parameters are lost)
    camera = transforms["CameraProjection"]
    inv_camera = camera.inverse()
    assert isinstance(inv_camera, tf.InverseProjection)
    
    # To recover CameraProjection: inv.inverse() -> Projection, then CameraProjection(matrix=...)
    recovered_proj = inv_camera.inverse()
    recovered_camera = tf.CameraProjection(matrix=recovered_proj.as_matrix())
    assert np.allclose(camera.focal_length, recovered_camera.focal_length, rtol=0.01)


def test_projection_basic():
    """Test basic Projection functionality."""
    
    # Simple projection matrix (identity intrinsics, no rotation/translation)
    projection_3x4 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    projection = tf.Projection(projection_3x4)
    
    # Project a point
    point = np.array([[1.0, 2.0, 5.0]])  # At z=5
    pixels = projection.apply(point)
    
    # With this simple projection, x/z=0.2, y/z=0.4
    expected = np.array([1.0/5.0, 2.0/5.0])
    assert np.allclose(pixels[0], expected), f"Expected {expected}, got {pixels[0]}"


def test_camera_projection():
    """Test CameraProjection with intrinsics."""
    
    # Typical camera intrinsics
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    rotation_matrix = np.eye(3)  # No rotation
    translation = np.array([0, 0, 0])  # At origin
    
    camera = tf.CameraProjection(
        intrinsic_matrix=intrinsic_matrix,
        rotation_matrix=rotation_matrix,
        translation=translation
    )
    
    # Project a point at (0, 0, 10) - should map to principal point
    point_center = np.array([[0.0, 0.0, 10.0]])
    pixels_center = camera.apply(point_center)
    assert np.allclose(pixels_center[0], [cx, cy]), "Center should project to principal point"
    
    # Project a point at (1, 0, 10)
    point_right = np.array([[1.0, 0.0, 10.0]])
    pixels_right = camera.apply(point_right)
    expected_x = cx + fx * (1.0 / 10.0)
    assert np.allclose(pixels_right[0, 0], expected_x), "X projection mismatch"


def test_camera_projection_from_matrix():
    """Test creating CameraProjection from a matrix."""
    
    # Create a projection matrix directly
    fx, fy = 600.0, 600.0
    cx, cy = 300.0, 200.0
    
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    rotation_matrix = np.eye(3)
    translation = np.array([[1.0], [2.0], [3.0]])
    
    extrinsic = np.hstack([rotation_matrix, translation])
    projection_matrix = intrinsic_matrix @ extrinsic
    
    # Create CameraProjection from the matrix
    camera = tf.CameraProjection(matrix=projection_matrix)
    
    # Check that K is approximately correct
    assert np.allclose(camera.focal_length, (fx, fy), rtol=0.01), "Focal length mismatch"
    assert np.allclose(camera.principal_point, (cx, cy), rtol=0.01), "Principal point mismatch"


def test_unprojection():
    """Test unprojecting 2D points back to 3D."""
    
    # Create camera
    intrinsic_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    camera = tf.CameraProjection(
        intrinsic_matrix=intrinsic_matrix,
        rotation_matrix=np.eye(3),
        translation=np.array([0, 0, 0])
    )
    
    # Original 3D points
    original_points = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 2.0, 5.0],
        [-2.0, 1.0, 8.0],
    ])
    
    # Project to 2D
    pixels = camera.apply(original_points)
    depths = original_points[:, 2]
    
    # Unproject back to 3D
    inv_camera = camera.inverse()
    recovered_points = inv_camera.unproject(pixels, depths)
    
    assert np.allclose(original_points, recovered_points, atol=1e-6), "Unprojection failed!"


def test_projection_in_graph():
    """Test using projections in TransformGraph."""
    
    graph = tf.TransformGraph()
    
    # World -> Camera (rigid transform)
    world_to_camera = tf.Transform(
        translation=[0, 0, 5],  # Camera 5m away
        rotation=[1, 0, 0, 0]
    )
    
    # Camera -> Image (projection)
    camera_projection = tf.CameraProjection(
        intrinsic_matrix=np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ]),
        rotation_matrix=np.eye(3),
        translation=np.array([0, 0, 0])
    )
    
    graph.add_transform("world", "camera", world_to_camera)
    graph.add_transform("camera", "image", camera_projection)
    
    # Get composed transform
    world_to_image = graph.get_transform("world", "image")
    
    # Serialize and restore the graph
    data = graph.to_dict()
    
    graph2 = tf.TransformGraph.from_dict(data)
    
    # Verify the camera projection was preserved
    restored_camera = graph2.get_transform("camera", "image")
    assert isinstance(restored_camera, tf.CameraProjection), "CameraProjection type not preserved!"


def test_error_handling():
    """Test error handling for unknown types."""
    
    # Try to deserialize unknown type
    with pytest.raises(ValueError, match="Unknown transform type: .*"):
        tf.deserialize_transform({"type": "UnknownType"})
    
    # Try to deserialize without type
    with pytest.raises(ValueError, match="Missing 'type' field .*"):
        tf.deserialize_transform({"matrix": [[1, 0], [0, 1]]})

