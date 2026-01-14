#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test the Projection classes and serialization system.
"""
import numpy as np
import pytest
import tgraph.transform as tf
def test_serialization_registry():
    """Test that all transform types are registered."""
    # Check registered types
    from tgraph.transform import _TRANSFORM_REGISTRY
    expected_types = [
        "Transform",
        "MatrixTransform",
        "Projection",
        "InverseProjection",
        "CameraProjection",
    ]

    for t in expected_types:
        assert t in _TRANSFORM_REGISTRY, f"Missing: {t}"


def test_serialize_deserialize_all_types():
    """Test serialize_transform/deserialize_transform for all types."""

    # Create one of each type
    transforms = {
        "Transform": tf.Transform(translation=[1.0, 2.0, 3.0], rotation=[0.707, 0.0, 0.0, 0.707]),
        "MatrixTransform": tf.MatrixTransform(np.eye(4)),
        "Projection": tf.Projection(np.array([[500, 0, 320, 0], [0, 500, 240, 0], [0, 0, 1, 0]])),
        "CameraProjection": tf.CameraProjection(
            intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),
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
    # DEPRECATED: CameraProjection(matrix=...) with full 3x4/4x4 is not supported directly.
    # But inv.inverse() returns a pure Projection.
    # If we want a CameraProjection, we must assert it's intrinsic-only or decompose.
    # In this test, we know inv_camera stores the intrinsics.
    recovered_proj = inv_camera.inverse()
    # recovered_proj is a Projection (wraps 3x3 or 3x4).
    # Since inv_camera was intrinsic-only (InverseCameraProjection), its inverse is K (3x3).
    
    # Verify it is effectively 3x3 (or 4x4 with identity extrinsic structure)
    P = recovered_proj.as_matrix()
    assert P.shape == (4, 4) or P.shape == (3, 3)
    
    # Extract K
    K_recovered = P[:3, :3]
    
    # Create object from explicit K
    recovered_camera = tf.CameraProjection(intrinsic_matrix=K_recovered)
    assert np.allclose(camera.focal_length, recovered_camera.focal_length, rtol=0.01)


def test_projection_basic():
    """Test basic Projection functionality."""

    # Simple projection matrix (identity intrinsics, no rotation/translation)
    projection_3x4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

    projection = tf.Projection(projection_3x4)

    # Project a point
    point = np.array([[1.0, 2.0, 5.0]])  # At z=5
    pixels = projection.apply(point)

    # With this simple projection, x/z=0.2, y/z=0.4
    expected = np.array([1.0 / 5.0, 2.0 / 5.0])
    assert np.allclose(pixels[0], expected), f"Expected {expected}, got {pixels[0]}"


def test_camera_projection():
    """Test CameraProjection with intrinsics."""

    # Typical camera intrinsics
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Intrinsic only
    camera = tf.CameraProjection(
        intrinsic_matrix=intrinsic_matrix
    )

    # Project a point at (0, 0, 10) in Camera Frame
    point_center = np.array([[0.0, 0.0, 10.0]])
    pixels_center = camera.apply(point_center)
    assert np.allclose(pixels_center[0], [cx, cy]), "Center should project to principal point"

    # Project a point at (1, 0, 10) in Camera Frame
    point_right = np.array([[1.0, 0.0, 10.0]])
    pixels_right = camera.apply(point_right)
    expected_x = cx + fx * (1.0 / 10.0)
    assert np.allclose(pixels_right[0, 0], expected_x), "X projection mismatch"


def test_decompose_projection():
    """Test decomposing a full projection matrix into K and Pose."""

    # Create a projection matrix directly
    fx, fy = 600.0, 600.0
    cx, cy = 300.0, 200.0

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    rotation_matrix = np.eye(3)
    translation = np.array([[1.0], [2.0], [3.0]])

    extrinsic = np.hstack([rotation_matrix, translation])
    projection_matrix = intrinsic_matrix @ extrinsic

    # Decompose
    cam, pose_tf = tf.decompose_projection_to_objects(projection_matrix)

    # Check that K is approximately correct
    assert np.allclose(cam.focal_length, (fx, fy), rtol=0.01), "Focal length mismatch"
    assert np.allclose(cam.principal_point, (cx, cy), rtol=0.01), "Principal point mismatch"
    
    # Check translation
    # decompose returns T_world_to_cam (which is extrinsic matrix logic R, t)
    # The extrinsic matrix constructed above is [R|t]. 
    # decompose logic: P = K [R|t]. 
    # So pose_tf.translation should be t if constructed this way?
    # verify implementation of decompose?
    # It solves t_vec = inv(K) @ P[:3,3].
    # P[:3,3] = K @ t_original. So t_vec should be t_original.
    assert np.allclose(pose_tf.translation.flatten(), translation.flatten()), "Translation mismatch"


def test_unprojection():
    """Test unprojecting 2D points back to 3D."""

    # Create camera
    intrinsic_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    camera = tf.CameraProjection(
        intrinsic_matrix=intrinsic_matrix
    )

    # Original 3D points
    original_points = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 2.0, 5.0],
            [-2.0, 1.0, 8.0],
        ]
    )

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
        rotation=[1, 0, 0, 0],
    )

    # Camera -> Image (projection)
    camera_projection = tf.CameraProjection(
        intrinsic_matrix=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    )

    graph.add_transform("world", "camera", world_to_camera)
    graph.add_transform("camera", "image", camera_projection)

    # Get composed transform
    graph.get_transform("world", "image")

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

def test_inverse_camera_shortcuts():
    """Test shortcut properties on CameraProjection/InverseCameraProjection."""
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
    cam = tf.CameraProjection(K=K, image_size=(1280, 720))

    assert cam.fx == 1000
    assert cam.fy == 1000
    assert cam.cx == 640
    assert cam.cy == 360
    
    inv_cam = cam.inverse()
    assert inv_cam.fx == 1000
    assert inv_cam.fy == 1000
    assert inv_cam.cx == 640
    assert inv_cam.cy == 360


def test_inverse_camera_type():
    """Test type relationships for InverseCameraProjection."""
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
    cam = tf.CameraProjection(K=K)
    
    inv_cam = cam.inverse()
    assert isinstance(inv_cam, tf.InverseCameraProjection)
    assert inv_cam.camera_projection is cam
    
    orig_cam = inv_cam.inverse()
    assert isinstance(orig_cam, tf.CameraProjection)
    assert orig_cam is cam


def test_inverse_camera_composition():
    """Test composition with InverseCameraProjection."""
    K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
    cam = tf.CameraProjection(K=K)
    inv_cam = cam.inverse()
    
    transform = tf.Transform(translation=[1, 0, 0])
    result = inv_cam * transform
    
    # Should fallback to MatrixTransform or similar logic for now
    assert hasattr(result, 'as_matrix')


def test_compound_graph_projection():
    """Test a complex chain involving Projection in the graph."""
    graph = tf.TransformGraph()
    
    # Chain: World -> Robot -> CameraMount -> Camera -> Image
    
    # 1. World -> Robot (Translation x=10)
    # T_robot_world implies Robot in World is at x=10? 
    # add_transform(source, target, T) -> T_source_target.
    # T_robot_world = Trans(10,0,0). Robot origin in World is 10,0,0.
    graph.add_transform("robot", "world", tf.Translation(x=10.0))
    
    # 2. Robot -> CameraMount (Rotation 90 deg yaw)
    graph.add_transform("mount", "robot", tf.Rotation.from_euler_angles(yaw=np.pi/2))
    
    # 3. Mount -> Camera (Translation z=2)
    graph.add_transform("camera", "mount", tf.Translation(z=2.0))
    
    # 4. Camera -> Image (Projection)
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float64)
    proj = tf.CameraProjection(K=K, image_size=(1000, 1000))
    
    # Edge: Image -> Camera (InverseProjection)
    # add_transform(source=image, target=camera, transform=proj.inverse())
    graph.add_transform("image", "camera", proj.inverse())
    
    # Test: Get projection from World to Image
    # T_world_to_image = get_transform("world", "image")
    # This transforms World Points -> Pixels.
    full_transform = graph.get_transform("world", "image")
    
    # Composition results in a full projection matrix (MatrixTransform or Projection)
    assert isinstance(full_transform, (tf.MatrixTransform, tf.Projection))
        
    # Verify correctness with a point
    # Camera structure (derived manually):
    # Robot at (10,0,0). Mount rot 90 (X->Y). Cam at Z=2 relative to Mount.
    # Cam origin in World: (10, 0, 2).
    # Cam Z axis = World Z axis (Up). Cam looks Up.
    
    # Test point: Directly above camera (Z=10 => dist=8)
    test_point = np.array([10, 0, 10]) 
    
    # Proj: (u, v) = (fx * X/Z + cx, fy * Y/Z + cy)
    # Relative to cam: X=0, Y=0, Z=8.
    # u = 500, v = 500.
    
    pixels = full_transform.apply(test_point)
    
    np.testing.assert_almost_equal(pixels[0,0], 500.0)
    np.testing.assert_almost_equal(pixels[0,1], 500.0)
