#! /usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    D = [0.1, -0.2, 0.001, 0.002, 0.05] # k1, k2, p1, p2, k3
    cam = tf.CameraProjection(
        intrinsic_matrix=K,
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
    
    cam = tf.CameraProjection(K)
    assert cam.projection_model == ProjectionModel.PINHOLE
    assert len(cam.dist_coeffs) == 0
    
    data = cam.to_dict()
    assert data["dist_coeffs"] == []
    assert data["projection_model"] == "Pinhole"

def test_camera_projection_models():
    """Test different projection models."""
    K = np.eye(3)
    
    for model in ProjectionModel:
        cam = tf.CameraProjection(K, projection_model=model)
        assert cam.projection_model == model
        
    # Test string init
    cam = tf.CameraProjection(K, projection_model="Fisheye")
    assert cam.projection_model == ProjectionModel.FISHEYE

def test_camera_projection_init_combinations():
    """Test initialization combinations for CameraProjection (Intrinsics only)."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    
    # 1. Alias K alone
    cam1 = tf.CameraProjection(K=K)
    assert np.allclose(cam1.intrinsic_matrix, K)
    
    # 2. Full name intrinsic_matrix alone
    cam2 = tf.CameraProjection(intrinsic_matrix=K)
    assert np.allclose(cam2.intrinsic_matrix, K)
    
    # 3. Positional argument
    cam3 = tf.CameraProjection(K)
    assert np.allclose(cam3.intrinsic_matrix, K)
    
    # 4. Matrix init (3x4) - Deprecated/Removed checks if not supported.

def test_camera_projection_init_errors():
    """Test error cases for CameraProjection initialization."""
    K = np.eye(3)
    
    # Missing both matrix and K
    with pytest.raises(ValueError, match="Must provide .*intrinsic_matrix.*"):
        tf.CameraProjection()
        
    # Invalid K shape
    with pytest.raises(ValueError, match="Intrinsic matrix must be 3x3"):
        tf.CameraProjection(K=np.eye(4))