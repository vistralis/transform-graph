#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CompositeProjection, InverseCompositeProjection,
and the exhaustive NxN composition type matrix.
"""

import numpy as np
import pytest

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def T1():
    return tf.Transform(translation=[1, 2, 3], rotation=[1, 0, 0, 0])


@pytest.fixture
def T2():
    return tf.Transform(translation=[4, 5, 6], rotation=[0.707, 0.707, 0, 0])


@pytest.fixture
def K():
    return tf.CameraProjection(intrinsic_matrix=np.diag([1000, 1000, 1]))


@pytest.fixture
def K_inv(K):
    return K.inverse()


@pytest.fixture
def CP(K, T1):
    return K * T1


@pytest.fixture
def ICP(T1, K_inv):
    return T1 * K_inv


@pytest.fixture
def M():
    return tf.MatrixTransform(np.eye(4) * 1.5)


# ---------------------------------------------------------------------------
# CompositeProjection
# ---------------------------------------------------------------------------


class TestCompositeProjection:
    """Test CompositeProjection creation and composition."""

    def test_creation(self, K, T1):
        """CompositeProjection(K, T) stores components and computes matrix."""
        comp = tf.CompositeProjection(K, T1)
        assert isinstance(comp, tf.CompositeProjection)
        assert comp.projection is K
        assert comp.transform is T1
        expected = K.as_matrix() @ T1.as_matrix()
        np.testing.assert_allclose(comp.as_matrix(), expected)

    def test_right_mul(self, K, T1):
        """CompositeProjection * Transform absorbs the transform."""
        comp = tf.CompositeProjection(K, T1)
        T2 = tf.Transform(translation=[5, 0, 0])
        result = comp * T2
        assert isinstance(result, tf.CompositeProjection)
        np.testing.assert_allclose(
            result.transform.translation.flatten(),
            [6, 2, 3],
            atol=1e-6,
        )
        assert result.projection is K

    def test_inversion_cycle(self, K, T1):
        """CompositeProjection → inverse → inverse returns to original."""
        comp = tf.CompositeProjection(K, T1)
        inv = comp.inverse()
        assert isinstance(inv, tf.InverseCompositeProjection)
        recovered = inv.inverse()
        assert isinstance(recovered, tf.CompositeProjection)
        np.testing.assert_allclose(recovered.as_matrix(), comp.as_matrix(), atol=1e-10)


# ---------------------------------------------------------------------------
# InverseCompositeProjection
# ---------------------------------------------------------------------------


class TestInverseCompositeProjection:
    """Test InverseCompositeProjection."""

    def test_creation(self, T1, K_inv):
        """T * K_inv creates InverseCompositeProjection."""
        res = T1 * K_inv
        assert isinstance(res, tf.InverseCompositeProjection)
        assert res.transform is T1
        assert res.projection is K_inv

    def test_left_mul(self, T1, K_inv):
        """Transform * InverseCompositeProjection absorbs transform."""
        inv_comp = T1 * K_inv
        T2 = tf.Transform(translation=[0, 10, 0])
        res = T2 * inv_comp
        assert isinstance(res, tf.InverseCompositeProjection)
        expected_pos = [1, 12, 3]
        np.testing.assert_allclose(
            res.transform.translation.flatten(),
            expected_pos,
            atol=1e-6,
        )

    def test_visualization_attributes(self, K, T1):
        """CompositeProjection has attributes expected by visualization."""
        comp = tf.CompositeProjection(K, T1)
        assert hasattr(comp, "projection")
        assert isinstance(comp.projection, tf.CameraProjection)
        inv = comp.inverse()
        assert hasattr(inv, "projection")
        assert isinstance(inv.projection.inverse(), tf.CameraProjection)


# ---------------------------------------------------------------------------
# Exhaustive composition matrix
# ---------------------------------------------------------------------------


def _check_composition(left, right, expected_type, raises=None):
    """Helper to verify composition result type and matrix equivalence."""
    if raises:
        with pytest.raises(raises):
            _ = left * right
    else:
        res = left * right
        assert isinstance(res, expected_type), (
            f"Expected {expected_type.__name__}, got {type(res).__name__}"
            f" for {type(left).__name__} * {type(right).__name__}"
        )
        expected_matrix = left.as_matrix() @ right.as_matrix()
        np.testing.assert_allclose(
            res.as_matrix(),
            expected_matrix,
            err_msg=f"Matrix mismatch: {type(left).__name__} * {type(right).__name__}",
        )


class TestCompositionMatrix:
    """Test the 6×6 composition type matrix for all transform types."""

    def test_exhaustive_matrix(self, T1, T2, K, K_inv, CP, ICP, M):
        """Every valid combination returns the correct type; invalid ones raise."""
        types = [T1, K, K_inv, CP, ICP, M]
        expected = [
            # Right: T, K, K_inv, CP, ICP, M
            [
                tf.Transform,
                TypeError,
                tf.InverseCompositeProjection,
                TypeError,
                tf.InverseCompositeProjection,
                tf.MatrixTransform,
            ],
            [
                tf.CompositeProjection,
                TypeError,
                tf.MatrixTransform,
                TypeError,
                tf.MatrixTransform,
                tf.MatrixTransform,
            ],
            [
                TypeError,
                tf.MatrixTransform,
                TypeError,
                tf.MatrixTransform,
                TypeError,
                tf.MatrixTransform,
            ],
            [
                tf.CompositeProjection,
                TypeError,
                tf.MatrixTransform,
                TypeError,
                tf.MatrixTransform,
                tf.MatrixTransform,
            ],
            [
                TypeError,
                tf.MatrixTransform,
                TypeError,
                tf.MatrixTransform,
                TypeError,
                tf.MatrixTransform,
            ],
            [
                tf.MatrixTransform,
                tf.MatrixTransform,
                tf.MatrixTransform,
                tf.MatrixTransform,
                tf.MatrixTransform,
                tf.MatrixTransform,
            ],
        ]
        for i, left in enumerate(types):
            for j, right in enumerate(types):
                exp = expected[i][j]
                if exp is TypeError:
                    _check_composition(left, right, None, raises=TypeError)
                else:
                    _check_composition(left, right, exp)

    def test_associativity(self, T1, K_inv):
        """Composition chains are associative: (T1 * T2) * IP == T1 * (T2 * IP)."""
        res = T1 * K_inv
        assert isinstance(res, tf.InverseCompositeProjection)

        T2 = tf.Transform(translation=[10, 0, 0])
        res2 = T2 * res
        assert isinstance(res2, tf.InverseCompositeProjection)
        np.testing.assert_allclose(
            res2.transform.as_matrix(),
            (T2 * T1).as_matrix(),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestCompositeSerialization:
    """Test to_dict / from_dict for composite projection types."""

    def test_composite_roundtrip(self, K, T1):
        """CompositeProjection to_dict → from_dict roundtrip."""
        comp = tf.CompositeProjection(K, T1)
        data = comp.to_dict()
        recovered = tf.deserialize_transform(data)
        assert isinstance(recovered, tf.CompositeProjection)
        np.testing.assert_allclose(recovered.as_matrix(), comp.as_matrix(), atol=1e-10)

    def test_inverse_composite_roundtrip(self, T1, K_inv):
        """InverseCompositeProjection to_dict → from_dict roundtrip."""
        inv = T1 * K_inv
        assert isinstance(inv, tf.InverseCompositeProjection)
        data = inv.to_dict()
        recovered = tf.deserialize_transform(data)
        assert isinstance(recovered, tf.InverseCompositeProjection)
        np.testing.assert_allclose(recovered.as_matrix(), inv.as_matrix(), atol=1e-10)

    def test_composite_repr(self, K, T1):
        """CompositeProjection __repr__ is informative."""
        comp = tf.CompositeProjection(K, T1)
        assert "CompositeProjection" in repr(comp)

    def test_inverse_composite_repr(self, T1, K_inv):
        """InverseCompositeProjection __repr__ is informative."""
        inv = T1 * K_inv
        assert "InverseCompositeProjection" in repr(inv)
