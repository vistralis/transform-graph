#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for roll-pitch-yaw Euler angle functions.
"""

import numpy as np

import tgraph.transform as tf


class TestRollPitchYaw:
    """Test from_roll_pitch_yaw / as_roll_pitch_yaw with known angle values."""

    def test_yaw_90_matrix_values(self):
        """90° yaw produces correct rotation matrix entries."""
        rot = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi / 2)
        m = rot.as_matrix()
        np.testing.assert_allclose(m[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(m[0, 1], -1.0, atol=1e-10)
        np.testing.assert_allclose(m[1, 0], 1.0, atol=1e-10)
        np.testing.assert_allclose(m[1, 1], 0.0, atol=1e-10)

    def test_roundtrip(self):
        """from_roll_pitch_yaw → as_roll_pitch_yaw recovers original angles."""
        roll, pitch, yaw = 0.3, -0.2, 0.5
        rot = tf.Rotation.from_roll_pitch_yaw(roll=roll, pitch=pitch, yaw=yaw)
        r, p, y = rot.as_roll_pitch_yaw()
        np.testing.assert_allclose(r, roll, atol=1e-10)
        np.testing.assert_allclose(p, pitch, atol=1e-10)
        np.testing.assert_allclose(y, yaw, atol=1e-10)

    def test_free_function_roundtrip(self):
        """Free functions from_roll_pitch_yaw / as_roll_pitch_yaw roundtrip correctly."""
        quat = tf.from_roll_pitch_yaw(roll=0.1, pitch=-0.2, yaw=np.pi / 3)
        roll, pitch, yaw = tf.as_roll_pitch_yaw(quat)
        np.testing.assert_allclose(roll, 0.1, atol=1e-10)
        np.testing.assert_allclose(pitch, -0.2, atol=1e-10)
        np.testing.assert_allclose(yaw, np.pi / 3, atol=1e-10)

    def test_roll_only_produces_rx(self):
        """Roll-only produces R_x, not R_z (regression for old ZYZ bug)."""
        rot = tf.Rotation.from_roll_pitch_yaw(roll=np.pi / 2)
        m = rot.as_matrix()
        # R_x(90°): Y→Z, Z→-Y
        np.testing.assert_allclose(m[1, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(m[1, 2], -1.0, atol=1e-10)
        np.testing.assert_allclose(m[2, 1], 1.0, atol=1e-10)
        np.testing.assert_allclose(m[2, 2], 0.0, atol=1e-10)
        # X column should be unchanged
        np.testing.assert_allclose(m[0, 0], 1.0, atol=1e-10)

    def test_pitch_only_produces_ry(self):
        """Pitch-only produces R_y."""
        rot = tf.Rotation.from_roll_pitch_yaw(pitch=np.pi / 2)
        m = rot.as_matrix()
        # R_y(90°): X→Z, Z→-X
        np.testing.assert_allclose(m[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(m[0, 2], 1.0, atol=1e-10)
        np.testing.assert_allclose(m[2, 0], -1.0, atol=1e-10)
        np.testing.assert_allclose(m[2, 2], 0.0, atol=1e-10)
        # Y column should be unchanged
        np.testing.assert_allclose(m[1, 1], 1.0, atol=1e-10)

    def test_identity_angles(self):
        """Zero angles produce identity rotation."""
        rot = tf.Rotation.from_roll_pitch_yaw(roll=0, pitch=0, yaw=0)
        np.testing.assert_allclose(rot.as_matrix(), np.eye(4), atol=1e-10)

    def test_combined_angles(self):
        """Combined roll+pitch+yaw produces correct composition order."""
        # ZYX intrinsic: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        roll, pitch, yaw = 0.1, 0.2, 0.3
        rot = tf.Rotation.from_roll_pitch_yaw(roll=roll, pitch=pitch, yaw=yaw)

        # Build expected from individual rotations
        r_x = tf.Rotation.from_roll_pitch_yaw(roll=roll)
        r_y = tf.Rotation.from_roll_pitch_yaw(pitch=pitch)
        r_z = tf.Rotation.from_roll_pitch_yaw(yaw=yaw)
        expected = r_z * r_y * r_x  # ZYX intrinsic = extrinsic XYZ
        np.testing.assert_allclose(rot.as_matrix(), expected.as_matrix(), atol=1e-10)
