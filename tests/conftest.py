#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared test fixtures for the transform-graph test suite.
"""

import numpy as np
import pytest

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------


@pytest.fixture
def K_simple():
    """Simple 3x3 intrinsic matrix (500px focal length, 640x480)."""
    return np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)


@pytest.fixture
def K_identity():
    """Identity intrinsic matrix (unit focal length)."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def K_realistic():
    """Realistic camera intrinsics from a calibrated sensor (1920x1080)."""
    return np.array(
        [
            [1757.6433677063817, 0.0, 983.2146800742082],
            [0.0, 1761.2943542186904, 550.4079103066119],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def D_realistic():
    """Distortion coefficients matching K_realistic."""
    return np.array(
        [
            0.027552039300292,
            0.032744776829062,
            -0.002784825282852963,
            0.0004911277422916802,
            -0.016114769915598803,
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Common transforms
# ---------------------------------------------------------------------------


@pytest.fixture
def translation_x1():
    """Translation of 1.0 along X."""
    return tf.Translation(x=1.0)


@pytest.fixture
def rotation_z90():
    """90° rotation about Z-axis."""
    return tf.Rotation(w=np.cos(np.pi / 4), x=0, y=0, z=np.sin(np.pi / 4))


# ---------------------------------------------------------------------------
# Common graphs
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_graph():
    """A simple linear graph: world → robot → camera."""
    graph = tf.TransformGraph()
    graph.add_transform("world", "robot", tf.Translation(x=2.0, y=1.0))
    graph.add_transform("robot", "camera", tf.Translation(x=0.5, z=0.5))
    return graph


@pytest.fixture
def chain_graph():
    """A 4-node chain: A → B → C → D."""
    graph = tf.TransformGraph()
    graph.add_transform("A", "B", tf.Translation(x=1.0))
    graph.add_transform("B", "C", tf.Translation(x=1.0))
    graph.add_transform("C", "D", tf.Translation(x=1.0))
    return graph
