#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Transform/Rotation convenience constructors:
from_rotation_matrix, from_quaternion, from_axis_angle.

TDD RED phase — tests written before implementation.
"""

import numpy as np
import pytest
import quaternion as npq

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# from_rotation_matrix
# ---------------------------------------------------------------------------


class TestFromRotationMatrix:
    """Test Transform.from_rotation_matrix classmethods."""

    def test_identity_matrix(self):
        """Identity 3×3 produces identity transform."""
        t = tf.Transform.from_rotation_matrix(np.eye(3))
        np.testing.assert_allclose(t.as_matrix(), np.eye(4), atol=1e-10)

    def test_known_90z(self):
        """90° rotation about Z from matrix matches from_roll_pitch_yaw."""
        angle = np.pi / 2
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t = tf.Transform.from_rotation_matrix(R)
        expected = tf.Rotation.from_roll_pitch_yaw(yaw=angle)
        np.testing.assert_allclose(t.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_with_translation(self):
        """from_rotation_matrix with translation sets both R and t."""
        R = np.eye(3)
        t = tf.Transform.from_rotation_matrix(R, t=[1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.translation.flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.as_matrix()[:3, :3], np.eye(3), atol=1e-10)

    def test_non_so3_raises(self):
        """Non-orthogonal matrix raises ValueError with validate=True."""
        bad_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="[Rr]otation|SO\\(3\\)|orthogonal"):
            tf.Transform.from_rotation_matrix(bad_matrix)

    def test_negative_det_raises(self):
        """Matrix with det=-1 (reflection) raises ValueError."""
        reflection = np.diag([1.0, 1.0, -1.0])
        with pytest.raises(ValueError, match="[Rr]otation|SO\\(3\\)|determinant"):
            tf.Transform.from_rotation_matrix(reflection)

    def test_validate_false_skips_check(self):
        """validate=False skips SO(3) check on non-orthogonal input."""
        bad_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        # Should not raise
        t = tf.Transform.from_rotation_matrix(bad_matrix, validate=False)
        assert isinstance(t, tf.Transform)

    def test_wrong_shape_raises(self):
        """Non-3×3 matrix raises ValueError."""
        with pytest.raises(ValueError, match="3.3"):
            tf.Transform.from_rotation_matrix(np.eye(4))

    def test_roundtrip_arbitrary_rotation(self):
        """Arbitrary rotation matrix round-trips correctly."""
        original = tf.Rotation.from_roll_pitch_yaw(roll=0.3, pitch=-0.5, yaw=1.2)
        R = original.as_matrix()[:3, :3]
        recovered = tf.Transform.from_rotation_matrix(R)
        np.testing.assert_allclose(recovered.as_matrix()[:3, :3], R, atol=1e-10)


# ---------------------------------------------------------------------------
# from_quaternion
# ---------------------------------------------------------------------------


class TestFromQuaternion:
    """Test Transform.from_quaternion classmethod."""

    def test_wxyz_identity(self):
        """wxyz [1,0,0,0] produces identity rotation."""
        t = tf.Transform.from_quaternion([1, 0, 0, 0])
        np.testing.assert_allclose(t.as_matrix(), np.eye(4), atol=1e-10)

    def test_wxyz_known_90z(self):
        """wxyz quaternion for 90° Z matches expected matrix."""
        w = np.cos(np.pi / 4)
        z = np.sin(np.pi / 4)
        t = tf.Transform.from_quaternion([w, 0, 0, z])
        expected = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi / 2)
        np.testing.assert_allclose(t.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_xyzw_convention(self):
        """xyzw convention reorders correctly."""
        w = np.cos(np.pi / 4)
        z = np.sin(np.pi / 4)
        # xyzw: [x, y, z, w] = [0, 0, sin, cos]
        t = tf.Transform.from_quaternion([0, 0, z, w], convention="xyzw")
        expected = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi / 2)
        np.testing.assert_allclose(t.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_wxyz_and_xyzw_same_result(self):
        """Same rotation expressed in wxyz and xyzw produces identical transforms."""
        w, x, y, z = 0.5, 0.5, 0.5, 0.5
        t_wxyz = tf.Transform.from_quaternion([w, x, y, z], convention="wxyz")
        t_xyzw = tf.Transform.from_quaternion([x, y, z, w], convention="xyzw")
        np.testing.assert_allclose(t_wxyz.as_matrix(), t_xyzw.as_matrix(), atol=1e-10)

    def test_auto_normalizes(self):
        """Non-unit quaternion is auto-normalized."""
        t = tf.Transform.from_quaternion([2, 0, 0, 0])
        np.testing.assert_allclose(t.as_matrix(), np.eye(4), atol=1e-10)

    def test_zero_quaternion_raises(self):
        """Zero quaternion raises ValueError."""
        with pytest.raises(ValueError, match="[Zz]ero"):
            tf.Transform.from_quaternion([0, 0, 0, 0])

    def test_with_translation(self):
        """from_quaternion with translation sets both."""
        t = tf.Transform.from_quaternion([1, 0, 0, 0], t=[5.0, 6.0, 7.0])
        np.testing.assert_allclose(t.translation.flatten(), [5.0, 6.0, 7.0])

    def test_invalid_convention_raises(self):
        """Unknown convention string raises ValueError."""
        with pytest.raises(ValueError, match="convention"):
            tf.Transform.from_quaternion([1, 0, 0, 0], convention="zxyw")

    def test_numpy_quaternion_object_accepted(self):
        """Accepts a numpy-quaternion object directly."""
        q = npq.quaternion(1, 0, 0, 0)
        t = tf.Transform.from_quaternion(q)
        np.testing.assert_allclose(t.as_matrix(), np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# from_axis_angle
# ---------------------------------------------------------------------------


class TestFromAxisAngle:
    """Test Transform.from_axis_angle classmethod."""

    def test_z_axis_90(self):
        """90° about Z-axis matches from_roll_pitch_yaw(yaw=π/2)."""
        t = tf.Transform.from_axis_angle([0, 0, 1], np.pi / 2)
        expected = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi / 2)
        np.testing.assert_allclose(t.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_x_axis_180(self):
        """180° about X-axis."""
        t = tf.Transform.from_axis_angle([1, 0, 0], np.pi)
        expected = tf.Rotation.from_roll_pitch_yaw(roll=np.pi)
        np.testing.assert_allclose(t.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_non_unit_axis_normalized(self):
        """Non-unit axis vector is auto-normalized."""
        t = tf.Transform.from_axis_angle([0, 0, 5], np.pi / 2)
        expected = tf.Transform.from_axis_angle([0, 0, 1], np.pi / 2)
        np.testing.assert_allclose(t.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_zero_axis_raises(self):
        """Zero-length axis raises ValueError."""
        with pytest.raises(ValueError, match="[Zz]ero|axis"):
            tf.Transform.from_axis_angle([0, 0, 0], np.pi / 2)

    def test_zero_angle_identity(self):
        """Zero angle produces identity regardless of axis."""
        t = tf.Transform.from_axis_angle([1, 0, 0], 0.0)
        np.testing.assert_allclose(t.as_matrix(), np.eye(4), atol=1e-10)

    def test_with_translation(self):
        """from_axis_angle with translation sets both."""
        t = tf.Transform.from_axis_angle([0, 0, 1], np.pi / 4, t=[1, 2, 3])
        np.testing.assert_allclose(t.translation.flatten(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Rotation companions
# ---------------------------------------------------------------------------


class TestRotationCompanions:
    """Test Rotation.from_* classmethods return Rotation type with zero translation."""

    def test_rotation_from_rotation_matrix_type(self):
        """from_rotation_matrix returns Rotation, not Transform."""
        R = np.eye(3)
        r = tf.Rotation.from_rotation_matrix(R)
        assert type(r) is tf.Rotation
        np.testing.assert_allclose(r.translation.flatten(), [0, 0, 0])

    def test_rotation_from_quaternion_type(self):
        """from_quaternion returns Rotation with zero translation."""
        r = tf.Rotation.from_quaternion([1, 0, 0, 0])
        assert type(r) is tf.Rotation
        np.testing.assert_allclose(r.translation.flatten(), [0, 0, 0])

    def test_rotation_from_axis_angle_type(self):
        """from_axis_angle returns Rotation with zero translation."""
        r = tf.Rotation.from_axis_angle([0, 0, 1], np.pi / 4)
        assert type(r) is tf.Rotation
        np.testing.assert_allclose(r.translation.flatten(), [0, 0, 0])

    def test_rotation_from_rotation_matrix_correct(self):
        """Rotation.from_rotation_matrix produces correct rotation."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        r = tf.Rotation.from_rotation_matrix(R)
        np.testing.assert_allclose(r.as_matrix()[:3, :3], R, atol=1e-10)
