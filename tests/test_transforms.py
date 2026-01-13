#!/usr/bin/env python3
"""
Test script for tgraph.transform module
"""

import numpy as np
import pytest

import tgraph.transform as tf


def test_basic_transforms():
    """Test creating basic transforms."""

    # Translation only
    trans = tf.Translation(x=1.0, y=2.0, z=3.0)
    expected_matrix = np.eye(4)
    expected_matrix[0:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(trans.as_matrix(), expected_matrix)

    # Rotation only (90 degrees around Z-axis)
    # quaternion for 90° rotation around Z: w=cos(45°), z=sin(45°)
    rot_z_90 = tf.Rotation(w=np.cos(np.pi / 4), x=0, y=0, z=np.sin(np.pi / 4))

    # Check rotation matrix part
    matrix = rot_z_90.as_matrix()
    assert np.isclose(matrix[0, 0], 0.0, atol=1e-6)
    assert np.isclose(matrix[0, 1], -1.0, atol=1e-6)
    assert np.isclose(matrix[1, 0], 1.0, atol=1e-6)
    assert np.isclose(matrix[1, 1], 0.0, atol=1e-6)

    # Identity
    identity = tf.Identity()
    assert np.allclose(identity.as_matrix(), np.eye(4))


def test_transform_composition():
    """Test transform composition."""

    # Create two transforms
    transform_a = tf.Translation(x=1.0, y=0.0, z=0.0)
    transform_b = tf.Translation(x=0.0, y=1.0, z=0.0)

    # Compose them
    composed = transform_a * transform_b

    expected_translation = np.array([1.0, 1.0, 0.0])
    assert np.allclose(composed.translation.flatten(), expected_translation)


def test_transform_inversion():
    """Test transform inversion."""

    transform = tf.Transform(
        translation=[1.0, 2.0, 3.0],
        rotation=[1.0, 0.0, 0.0, 0.0],  # Identity rotation
    )
    inverse = transform.inverse()

    expected_inverse_translation = np.array([-1.0, -2.0, -3.0])
    assert np.allclose(inverse.translation.flatten(), expected_inverse_translation)

    # Verify: T * T^-1 = Identity
    should_be_identity = transform * inverse
    assert np.allclose(should_be_identity.as_matrix(), np.eye(4))


def test_transform_points():
    """Test transforming points."""

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    transform = tf.Translation(x=5.0, y=5.0, z=5.0)
    transformed_points = tf.transform_points(transform, points)

    expected_points = points + 5.0
    assert np.allclose(transformed_points, expected_points)


def test_complex_transformation():
    """Test complex transformation (translation + rotation)."""

    # Create a transform that translates and rotates
    # 90° around Z
    complex_transform = tf.Transform(
        translation=[1.0, 2.0, 3.0], rotation=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    )

    point = np.array([[1.0, 0.0, 0.0]])
    transformed = tf.transform_points(complex_transform, point)

    # 1. Rotate (1,0,0) 90 deg around Z -> (0,1,0)
    # 2. Translate by (1,2,3) -> (1,3,3)
    expected = np.array([1.0, 3.0, 3.0])

    assert np.allclose(transformed[0], expected, atol=1e-6)


def test_invalid_init_kwargs():
    """Verify that Rotation and Translation raise TypeError for unexpected kwargs."""

    with pytest.raises(TypeError, match="unexpected keyword argument 'yaw'"):
        tf.Rotation(yaw=0.5)

    with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
        tf.Translation(foo="bar")
