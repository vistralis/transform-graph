#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tgraph.quaternion interop module.

TDD RED phase — tests written before implementation.
"""

import numpy as np
import pytest
import quaternion as npq
from scipy.spatial.transform import Rotation as ScipyRotation

from tgraph.quaternion import (
    from_scipy,
    from_wxyz,
    from_xyzw,
    normalize,
    to_scipy,
    to_wxyz,
    to_xyzw,
)

# ---------------------------------------------------------------------------
# to_xyzw / from_xyzw
# ---------------------------------------------------------------------------


class TestXyzwConversion:
    """Test wxyz (numpy-quaternion) ↔ xyzw (scipy/ROS) conversion."""

    def test_to_xyzw_known_values(self):
        """Identity quaternion w=1 maps to [0, 0, 0, 1] in xyzw."""
        q = npq.one
        result = to_xyzw(q)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 1.0])

    def test_to_xyzw_non_trivial(self):
        """Non-trivial quaternion preserves component mapping."""
        q = npq.quaternion(0.5, 0.1, 0.2, 0.3)
        result = to_xyzw(q)
        np.testing.assert_array_equal(result, [0.1, 0.2, 0.3, 0.5])

    def test_from_xyzw_known_values(self):
        """[0, 0, 0, 1] in xyzw maps to identity quaternion."""
        result = from_xyzw([0.0, 0.0, 0.0, 1.0])
        assert result == npq.one

    def test_roundtrip_xyzw(self):
        """to_xyzw → from_xyzw preserves the quaternion."""
        q = npq.quaternion(0.7071, 0.0, 0.7071, 0.0)
        recovered = from_xyzw(to_xyzw(q))
        assert recovered.w == pytest.approx(q.w)
        assert recovered.x == pytest.approx(q.x)
        assert recovered.y == pytest.approx(q.y)
        assert recovered.z == pytest.approx(q.z)

    def test_from_xyzw_wrong_size_raises(self):
        """Wrong-length input raises ValueError."""
        with pytest.raises(ValueError, match="4 elements"):
            from_xyzw([1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# to_wxyz / from_wxyz
# ---------------------------------------------------------------------------


class TestWxyzConversion:
    """Test quaternion ↔ [w,x,y,z] array conversion."""

    def test_to_wxyz_known_values(self):
        """Identity quaternion maps to [1, 0, 0, 0]."""
        q = npq.one
        result = to_wxyz(q)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0, 0.0])

    def test_from_wxyz_known_values(self):
        """[1, 0, 0, 0] maps to identity quaternion."""
        result = from_wxyz([1.0, 0.0, 0.0, 0.0])
        assert result == npq.one

    def test_roundtrip_wxyz(self):
        """to_wxyz → from_wxyz preserves the quaternion."""
        q = npq.quaternion(0.5, 0.5, 0.5, 0.5)
        recovered = from_wxyz(to_wxyz(q))
        assert recovered == q

    def test_from_wxyz_wrong_size_raises(self):
        """Wrong-length input raises ValueError."""
        with pytest.raises(ValueError, match="4 elements"):
            from_wxyz([1.0, 0.0])


# ---------------------------------------------------------------------------
# to_scipy / from_scipy
# ---------------------------------------------------------------------------


class TestScipyConversion:
    """Test numpy-quaternion ↔ scipy.spatial.transform.Rotation."""

    def test_to_scipy_identity(self):
        """Identity quaternion converts to identity scipy Rotation."""
        r = to_scipy(npq.one)
        np.testing.assert_allclose(r.as_matrix(), np.eye(3), atol=1e-10)

    def test_from_scipy_identity(self):
        """Identity scipy Rotation converts to identity quaternion."""
        r = ScipyRotation.identity()
        q = from_scipy(r)
        np.testing.assert_allclose([q.w, q.x, q.y, q.z], [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_roundtrip_scipy(self):
        """quaternion → scipy → quaternion preserves rotation."""
        q = npq.quaternion(np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # 90° about Z
        recovered = from_scipy(to_scipy(q))
        # Quaternions may differ by sign — compare rotation matrices
        original_matrix = npq.as_rotation_matrix(q)
        recovered_matrix = npq.as_rotation_matrix(recovered)
        np.testing.assert_allclose(recovered_matrix, original_matrix, atol=1e-10)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    """Test quaternion normalization."""

    def test_normalize_unit_unchanged(self):
        """Unit quaternion is returned unchanged."""
        q = npq.one
        result = normalize(q)
        assert result.w == pytest.approx(1.0)

    def test_normalize_non_unit(self):
        """Non-unit quaternion is normalized to unit length."""
        q = npq.quaternion(2.0, 0.0, 0.0, 0.0)
        result = normalize(q)
        norm = np.sqrt(result.w**2 + result.x**2 + result.y**2 + result.z**2)
        assert norm == pytest.approx(1.0)
        assert result.w == pytest.approx(1.0)

    def test_normalize_arbitrary(self):
        """Arbitrary non-unit quaternion normalizes correctly."""
        q = npq.quaternion(1.0, 1.0, 1.0, 1.0)
        result = normalize(q)
        norm = np.sqrt(result.w**2 + result.x**2 + result.y**2 + result.z**2)
        assert norm == pytest.approx(1.0)
        expected = 0.5  # 1/sqrt(4)
        assert result.w == pytest.approx(expected)

    def test_normalize_zero_raises(self):
        """Zero quaternion raises ValueError."""
        q = npq.quaternion(0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="[Zz]ero"):
            normalize(q)
