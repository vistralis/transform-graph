#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for multi-camera setups: epipolar geometry, Fundamental/Essential/Homography
matrices, projection frame detection, and strict composition rules.

Migrated from unittest to pytest.
"""

import numpy as np
import pytest

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_camera_graph():
    """Build a two-camera graph with a shared link frame."""
    K = np.array(
        [
            [1757.6433677063817, 0.0, 983.2146800742082],
            [0.0, 1761.2943542186904, 550.4079103066119],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    D = np.array(
        [0.027552, 0.032744, -0.002784, 0.000491, -0.016114],
        dtype=np.float64,
    )

    graph = tf.TransformGraph()
    graph.add_transform("robot", "world", tf.Translation(x=1.0))
    graph.add_transform("camera_link", "robot", tf.Translation(z=0.1))
    graph.add_transform("camera_left", "camera_link", tf.Translation(y=0.2))
    graph.add_transform("camera_right", "camera_link", tf.Translation(y=-0.2))

    optical_rot = tf.Rotation(x=-0.5, y=0.5, z=-0.5, w=0.5)
    graph.add_transform("camera_left_optical_frame", "camera_left", optical_rot)
    graph.add_transform(
        "camera_left_optical_frame",
        "IMAGE_camera_left",
        tf.CameraProjection(K=K, D=D),
    )
    graph.add_transform("camera_right_optical_frame", "camera_right", optical_rot)
    graph.add_transform(
        "camera_right_optical_frame",
        "IMAGE_camera_right",
        tf.CameraProjection(K=K, D=D),
    )

    return graph, K


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiCamera:
    """Test multi-camera geometry and composition rules."""

    def test_strict_composition(self, multi_camera_graph):
        """Transform * CameraProjection is forbidden; Projection * Transform is allowed."""
        _, K = multi_camera_graph
        T = tf.Translation(x=1.0)
        P = tf.CameraProjection(K)

        with pytest.raises(TypeError):
            _ = T * P

        result = P * T
        assert isinstance(result, tf.Projection)

    def test_inter_image_transform(self, multi_camera_graph):
        """get_transform between two image frames returns MatrixTransform."""
        graph, _ = multi_camera_graph
        T = graph.get_transform("IMAGE_camera_right", "IMAGE_camera_left")
        assert isinstance(T, tf.MatrixTransform)
        assert T.as_matrix().shape == (4, 4)

    def test_fundamental_matrix(self, multi_camera_graph):
        """Fundamental matrix has shape (3, 3)."""
        graph, _ = multi_camera_graph
        F = graph.get_fundamental_matrix("IMAGE_camera_left", "IMAGE_camera_right")
        assert F.shape == (3, 3)

    def test_essential_matrix(self, multi_camera_graph):
        """Essential matrix has shape (3, 3)."""
        graph, _ = multi_camera_graph
        E = graph.get_essential_matrix("IMAGE_camera_left", "IMAGE_camera_right")
        assert E.shape == (3, 3)

    def test_homography(self, multi_camera_graph):
        """Homography for a planar scene has shape (3, 3)."""
        graph, _ = multi_camera_graph
        n = np.array([0, 0, 1])
        d = 10.0
        H = graph.get_homography("IMAGE_camera_left", "IMAGE_camera_right", n, d)
        assert H.shape == (3, 3)

    def test_is_projection_frame(self, multi_camera_graph):
        """is_projection_frame correctly identifies IMAGE frames."""
        graph, _ = multi_camera_graph
        assert graph.is_projection_frame("IMAGE_camera_left")
        assert graph.is_projection_frame("IMAGE_camera_right")
        assert not graph.is_projection_frame("camera_left")
        assert not graph.is_projection_frame("robot")
