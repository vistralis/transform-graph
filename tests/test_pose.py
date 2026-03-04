#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Pose class: init, inversion, composition, conversions,
and frame-ID validation.
"""

import numpy as np
import pytest
import quaternion

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestPoseInit:
    """Test Pose initialization and properties."""

    def test_defaults(self):
        """Default Pose is identity with no frame IDs."""
        p = tf.Pose()
        np.testing.assert_allclose(p.position, [0, 0, 0])
        assert p.orientation == quaternion.one
        assert p.frame_id is None
        assert p.child_frame_id is None

    def test_with_values(self):
        """Pose stores position, orientation, and frame IDs."""
        pos = np.array([1.0, 2.0, 3.0])
        quat = tf.from_roll_pitch_yaw(yaw=np.pi / 2)
        p = tf.Pose(position=pos, orientation=quat, frame_id="map", child_frame_id="base_link")

        np.testing.assert_allclose(p.position, pos)
        assert np.isclose(p.orientation, quat)
        assert p.frame_id == "map"
        assert p.child_frame_id == "base_link"

    def test_position_setter(self):
        """position property setter works."""
        p = tf.Pose()
        p.position = [4, 5, 6]
        np.testing.assert_allclose(p.position, [4, 5, 6])

    def test_rvec_init(self):
        """3-element orientation is treated as rotation vector."""
        rvec = np.array([0.1, 0.2, 0.3])
        p = tf.Pose(orientation=rvec)
        expected_quat = quaternion.from_rotation_vector(rvec)
        assert np.isclose(p.orientation, expected_quat)

    def test_invalid_orientation(self):
        """Invalid orientation shape raises ValueError."""
        with pytest.raises(ValueError):
            tf.Pose(orientation=[1, 2])


# ---------------------------------------------------------------------------
# Inversion
# ---------------------------------------------------------------------------


class TestPoseInversion:
    """Test Pose inverse with frame swapping."""

    def test_inverse_swaps_frames(self):
        """Inverse swaps frame_id and child_frame_id."""
        p = tf.Pose(
            position=[1, 0, 0],
            orientation=quaternion.one,
            frame_id="map",
            child_frame_id="base_link",
        )
        inv = p.inverse()
        assert inv.frame_id == "base_link"
        assert inv.child_frame_id == "map"
        np.testing.assert_allclose(inv.position, [-1, 0, 0])

    def test_inverse_override_frames(self):
        """Inverse with explicit new frame IDs."""
        p = tf.Pose(position=[1, 0, 0], frame_id="map", child_frame_id="base_link")
        inv = p.inverse(new_frame_id="new_root", new_child_frame_id="old_root")
        assert inv.frame_id == "new_root"
        assert inv.child_frame_id == "old_root"

    def test_inverse_partial_override(self):
        """Partial override uses fallback for missing frame."""
        p = tf.Pose(position=[1, 0, 0], frame_id="map", child_frame_id="base_link")
        inv = p.inverse(new_frame_id="partial")
        assert inv.frame_id == "partial"
        assert inv.child_frame_id == "map"


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestPoseComposition:
    """Test Pose composition and frame-ID validation."""

    def test_pose_compose(self):
        """Pose * Pose composes positions and preserves outer frame IDs."""
        p1 = tf.Pose(position=[1, 0, 0], frame_id="world", child_frame_id="robot")
        p2 = tf.Pose(position=[0, 0, 1], frame_id="robot", child_frame_id="camera")
        p3 = p1 * p2
        np.testing.assert_allclose(p3.position, [1, 0, 1])
        assert p3.frame_id == "world"
        assert p3.child_frame_id == "camera"

    def test_pose_compose_transform(self):
        """Pose * Transform produces a Pose."""
        p = tf.Pose(position=[1, 0, 0], frame_id="world", child_frame_id="robot")
        t = tf.Translation(x=1)
        result = p * t
        assert isinstance(result, tf.Pose)
        np.testing.assert_allclose(result.position, [2, 0, 0])
        assert result.frame_id == "world"
        assert result.child_frame_id is None

    def test_frame_mismatch_error(self):
        """Composition raises ValueError when child != parent frame."""
        p1 = tf.Pose(frame_id="A", child_frame_id="B")
        p2 = tf.Pose(frame_id="D", child_frame_id="C")
        with pytest.raises(ValueError, match="Frame mismatch"):
            _ = p1 * p2

    def test_none_frames_permissive(self):
        """None frame IDs are permissive (no validation error)."""
        p1 = tf.Pose(frame_id="A", child_frame_id=None)
        p2 = tf.Pose(frame_id="B", child_frame_id="C")
        p3 = p1 * p2
        assert p3.frame_id == "A"
        assert p3.child_frame_id == "C"


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


class TestPoseConversions:
    """Test Pose conversion methods."""

    def test_as_transform(self):
        """as_transform returns a Transform with same data."""
        p = tf.Pose(position=[1, 2, 3], orientation=quaternion.x)
        t = p.as_transform()
        assert isinstance(t, tf.Transform)
        np.testing.assert_allclose(t.translation.flatten(), p.position)

    def test_from_transform(self):
        """from_transform creates Pose with frame IDs."""
        t = tf.Transform(translation=[1, 2, 3])
        p = tf.Pose.from_transform(t, frame_id="A", child_frame_id="B")
        assert p.frame_id == "A"
        assert p.child_frame_id == "B"

    def test_to_list(self):
        """to_list returns [px, py, pz, qw, qx, qy, qz]."""
        p = tf.Pose(position=[1, 2, 3], orientation=quaternion.x)
        lst = p.to_list()
        assert len(lst) == 7
        assert lst[0] == 1.0

    def test_to_matrix(self):
        """to_matrix returns 4x4 SE(3)."""
        p = tf.Pose(position=[1, 2, 3])
        mat = p.to_matrix()
        assert mat.shape == (4, 4)

    def test_repr(self):
        """__repr__ contains key fields."""
        p = tf.Pose(frame_id="A", child_frame_id="B")
        s = repr(p)
        assert "Pose(" in s
        assert "frame_id='A'" in s
        assert "child_frame_id='B'" in s
