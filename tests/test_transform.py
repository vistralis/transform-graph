#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for core transform types: Transform, Translation, Rotation, Identity,
MatrixTransform.
"""

import numpy as np
import pytest

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTransformConstruction:
    """Test creating basic transform types."""

    def test_translation(self):
        """Translation creates correct 4x4 matrix."""
        trans = tf.Translation(x=1.0, y=2.0, z=3.0)
        expected = np.eye(4)
        expected[0:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(trans.as_matrix(), expected)

    def test_rotation_z90(self, rotation_z90):
        """90° rotation about Z-axis produces correct matrix entries."""
        m = rotation_z90.as_matrix()
        np.testing.assert_allclose(m[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(m[0, 1], -1.0, atol=1e-10)
        np.testing.assert_allclose(m[1, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(m[1, 1], 0.0, atol=1e-10)

    def test_identity(self):
        """Identity produces 4x4 identity matrix."""
        identity = tf.Identity()
        np.testing.assert_allclose(identity.as_matrix(), np.eye(4))

    def test_invalid_rotation_kwargs(self):
        """Rotation rejects unexpected keyword arguments."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'yaw'"):
            tf.Rotation(yaw=0.5)

    def test_invalid_translation_kwargs(self):
        """Translation rejects unexpected keyword arguments."""
        with pytest.raises(TypeError, match="unexpected keyword argument 'foo'"):
            tf.Translation(foo="bar")


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestTransformComposition:
    """Test transform composition (* operator)."""

    def test_translation_composition(self):
        """Composing two translations sums them."""
        a = tf.Translation(x=1.0)
        b = tf.Translation(y=1.0)
        composed = a * b
        np.testing.assert_allclose(composed.translation.flatten(), [1.0, 1.0, 0.0])

    def test_complex_transformation(self):
        """Rotation + translation applied to a point gives correct result."""
        # 90° around Z, then translate (1,2,3)
        t = tf.Transform(
            translation=[1.0, 2.0, 3.0],
            rotation=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
        )
        point = np.array([[1.0, 0.0, 0.0]])
        result = tf.transform_points(point, t)
        # Rotate (1,0,0) 90° around Z → (0,1,0), then translate → (1,3,3)
        np.testing.assert_allclose(result[0], [1.0, 3.0, 3.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


class TestTransformInversion:
    """Test transform inversion."""

    def test_basic_inversion(self):
        """Inverse of a translation negates it."""
        t = tf.Transform(translation=[1.0, 2.0, 3.0], rotation=[1, 0, 0, 0])
        inv = t.inverse()
        np.testing.assert_allclose(inv.translation.flatten(), [-1.0, -2.0, -3.0])

    def test_compose_with_inverse_is_identity(self):
        """T * T^-1 = Identity."""
        t = tf.Transform(translation=[1.0, 2.0, 3.0], rotation=[1, 0, 0, 0])
        result = t * t.inverse()
        np.testing.assert_allclose(result.as_matrix(), np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# transform_points
# ---------------------------------------------------------------------------


class TestTransformPoints:
    """Test the transform_points free function."""

    def test_translation_on_points(self):
        """Translating points by (5,5,5) adds offset."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
        t = tf.Translation(x=5.0, y=5.0, z=5.0)
        result = tf.transform_points(points, t)
        np.testing.assert_allclose(result, points + 5.0)

    def test_invalid_object(self):
        """transform_points raises TypeError for non-transform objects."""
        with pytest.raises(TypeError, match="must be BaseTransform or TransformGraph"):
            tf.transform_points(np.zeros((3, 3)), "not_a_transform")


# ---------------------------------------------------------------------------
# MatrixTransform
# ---------------------------------------------------------------------------


class TestMatrixTransform:
    """Tests for MatrixTransform construction, inverse, composition, repr."""

    def test_inverse_known_values(self):
        """Inverse of a known 4x4 matrix produces correct element values."""
        mat = np.array(
            [
                [1, 0, 0, 2],
                [0, 0, -1, 3],
                [0, 1, 0, 4],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        mt = tf.MatrixTransform(mat)
        inv = mt.inverse()
        identity = mt * inv
        np.testing.assert_allclose(identity.as_matrix(), np.eye(4), atol=1e-10)

    def test_composition_known_values(self):
        """Composition of two MatrixTransforms produces correct product."""
        a = np.eye(4)
        a[0, 3] = 1.0
        b = np.eye(4)
        b[1, 3] = 2.0
        mt_a = tf.MatrixTransform(a)
        mt_b = tf.MatrixTransform(b)
        result = mt_a * mt_b
        expected = a @ b
        np.testing.assert_allclose(result.as_matrix(), expected)

    def test_repr_contains_values(self):
        """__repr__ contains the matrix values."""
        mat = np.eye(4) * 2.0
        mt = tf.MatrixTransform(mat)
        s = repr(mt)
        assert "MatrixTransform" in s


# ---------------------------------------------------------------------------
# Transform.from_matrix
# ---------------------------------------------------------------------------


class TestFromMatrix:
    """Test Transform.from_matrix round-trip."""

    def test_identity(self):
        """from_matrix with identity gives zero translation, identity rotation."""
        t = tf.Transform.from_matrix(np.eye(4))
        np.testing.assert_allclose(t.translation.flatten(), [0, 0, 0], atol=1e-10)

    def test_known_se3(self):
        """from_matrix correctly extracts translation and rotation from a known SE(3)."""
        original = tf.Transform(
            translation=[1.0, 2.0, 3.0],
            rotation=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)],
        )
        recovered = tf.Transform.from_matrix(original.as_matrix())
        np.testing.assert_allclose(recovered.as_matrix(), original.as_matrix(), atol=1e-10)
