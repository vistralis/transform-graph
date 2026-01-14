#!/usr/bin/env python3
"""
Test CameraProjection with distortion and all initialization combinations.
"""

import numpy as np
import tgraph.transform as tf
from tgraph.transform import ProjectionModel
import pytest

def test_camera_projection_with_distortion():
    """Test initializing CameraProjection with distortion coefficients."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    R = np.eye(3)
    t = np.zeros(3)
    D = [0.1, -0.2, 0.001, 0.002, 0.05] # k1, k2, p1, p2, k3
    
    cam = tf.CameraProjection(
        intrinsic_matrix=K,
        rotation_matrix=R,
        translation=t,
        dist_coeffs=D,
        projection_model=ProjectionModel.PINHOLE_POLYNOMIAL
    )
    
    assert cam.projection_model == ProjectionModel.PINHOLE_POLYNOMIAL
    assert np.allclose(cam.dist_coeffs, D)
    assert np.allclose(cam.distortion_coefficients, D)
    
    # Test serialization
    data = cam.to_dict()
    assert data["dist_coeffs"] == D
    assert data["projection_model"] == "Pinhole+Polynomial"
    
    cam2 = tf.CameraProjection.from_dict(data)
    assert cam2.projection_model == ProjectionModel.PINHOLE_POLYNOMIAL
    assert np.allclose(cam2.dist_coeffs, D)

def test_camera_projection_default_model():
    """Test default values for model and distortion."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    R = np.eye(3)
    t = np.zeros(3)
    
    cam = tf.CameraProjection(K, R, t)
    assert cam.projection_model == ProjectionModel.PINHOLE
    assert len(cam.dist_coeffs) == 0
    
    data = cam.to_dict()
    assert data["dist_coeffs"] == []
    assert data["projection_model"] == "Pinhole"

def test_camera_projection_models():
    """Test different projection models."""
    K = np.eye(3)
    R = np.eye(3)
    t = np.zeros(3)
    
    for model in ProjectionModel:
        cam = tf.CameraProjection(K, R, t, projection_model=model)
        assert cam.projection_model == model
        
    # Test string init
    cam = tf.CameraProjection(K, R, t, projection_model="Fisheye")
    assert cam.projection_model == ProjectionModel.FISHEYE

def test_camera_projection_init_combinations():
    """Test all possible initialization combinations for CameraProjection."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    R = tf.Rotation.from_euler_angles(yaw=0.1).as_matrix()[:3, :3]
    t = np.array([1.0, 2.0, 3.0])
    
    # 1. Alias K alone
    cam1 = tf.CameraProjection(K=K)
    assert np.allclose(cam1.intrinsic_matrix, K)
    assert np.allclose(cam1.rotation_matrix, np.eye(3))
    
    # 2. Full name intrinsic_matrix alone
    cam2 = tf.CameraProjection(intrinsic_matrix=K)
    assert np.allclose(cam2.intrinsic_matrix, K)
    
    # 3. K + R
    cam3 = tf.CameraProjection(K=K, R=R)
    assert np.allclose(cam3.rotation_matrix, R)
    assert np.allclose(cam3.t, np.zeros(3))  # CV extrinsic t
    
    # 4. K + t
    cam4 = tf.CameraProjection(K=K, t=t)
    assert np.allclose(cam4.t, t)  # CV extrinsic t
    assert np.allclose(cam4.rotation_matrix, np.eye(3))
    
    # 5. K + R + t
    cam5 = tf.CameraProjection(K=K, R=R, t=t)
    assert np.allclose(cam5.intrinsic_matrix, K)
    assert np.allclose(cam5.rotation_matrix, R)
    assert np.allclose(cam5.t, t)  # CV extrinsic t
    
    # 6. Full names: intrinsic_matrix + rotation_matrix + translation
    cam6 = tf.CameraProjection(intrinsic_matrix=K, rotation_matrix=R, translation=t)
    assert np.allclose(cam6.intrinsic_matrix, K)
    assert np.allclose(cam6.rotation_matrix, R)
    assert np.allclose(cam6.t, t)  # CV extrinsic t
    
    # 7. Positional arguments (ordered: intrinsic, rotation, translation)
    cam7 = tf.CameraProjection(K, R, t)
    assert np.allclose(cam7.intrinsic_matrix, K)
    assert np.allclose(cam7.rotation_matrix, R)
    assert np.allclose(cam7.t, t)  # CV extrinsic t
    
    # 8. Matrix init (3x4)
    # Use a simpler R to avoid RQ decomposition sign ambiguities for this test
    R_simple = np.eye(3)
    P_3x4 = K @ np.hstack([R_simple, t.reshape(3, 1)])
    cam8 = tf.CameraProjection(matrix=P_3x4)
    assert np.allclose(cam8.as_matrix_3x4(), P_3x4)
    # Check decomposition consistency (note: R might have sign flips in RQ, 
    # but the full P should match)
    
    # 9. Matrix init (4x4)
    P_4x4 = np.eye(4)
    P_4x4[:3, :] = P_3x4
    cam9 = tf.CameraProjection(matrix=P_4x4)
    assert np.allclose(cam9.as_matrix(), P_4x4)

def test_camera_projection_init_errors():
    """Test error cases for CameraProjection initialization."""
    K = np.eye(3)
    
    # Missing both matrix and K
    with pytest.raises(ValueError, match="Must provide either 'matrix' or 'intrinsic_matrix'"):
        tf.CameraProjection()
        
    # Invalid K shape
    with pytest.raises(ValueError, match="Intrinsic matrix must be 3x3"):
        tf.CameraProjection(K=np.eye(4))
        
    # Invalid R shape
    with pytest.raises(ValueError, match="Rotation matrix must be 3x3"):
        tf.CameraProjection(K=K, R=np.eye(4))
        
    # Invalid matrix shape
    with pytest.raises(ValueError, match="Matrix must be 3x4 or 4x4"):
        tf.CameraProjection(matrix=np.eye(2))