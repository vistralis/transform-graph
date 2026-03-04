#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the visualization module: 3D/2D graph rendering, frustum drawing,
heuristic root, and MatrixTransform axis traces.
"""

import numpy as np
import pytest

import tgraph.transform as tf

try:
    import plotly.graph_objects as go

    import tgraph.visualization as vis

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


pytestmark = pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not installed")


# ---------------------------------------------------------------------------
# 3D visualization
# ---------------------------------------------------------------------------


class TestVisualize3D:
    """Test 3D transform graph visualization."""

    def test_basic(self):
        """Simple graph produces valid figure with expected traces."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "robot", tf.Translation(x=1))
        fig = vis.visualize_transform_graph(graph, target_frame="world")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Transform Graph Visualization (3D)"
        assert len(fig.data) == 9  # 2 frames × 4 traces + 1 connection

    def test_empty_graph(self):
        """Empty graph produces 'Empty Graph' title."""
        fig = vis.visualize_transform_graph(tf.TransformGraph())
        assert fig.layout.title.text == "Empty Graph"

    def test_disconnected_components(self):
        """Disconnected nodes are not plotted when a target_frame is given."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "robot", tf.Translation(x=1))
        graph.add_transform("A", "B", tf.Translation(x=1))  # disconnected
        fig = vis.visualize_transform_graph(graph, target_frame="world")
        assert len(fig.data) == 9  # only world + robot

    def test_heuristic_root(self):
        """visualize_transform_graph without target_frame picks a root."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "robot", tf.Translation(x=1))
        fig = vis.visualize_transform_graph(graph)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# 2D topology visualization
# ---------------------------------------------------------------------------


class TestVisualize2D:
    """Test 2D topology visualization."""

    def test_basic(self):
        """2D topology creates a figure with edge and node traces."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        graph.add_transform("B", "C", tf.Translation(x=1))
        graph.get_transform("A", "C")  # creates cache edge
        fig = vis.visualize_graph(graph)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 7

    def test_empty(self):
        """Empty graph produces 'Empty Graph' title."""
        fig = vis.visualize_graph(tf.TransformGraph())
        assert fig.layout.title.text == "Empty Graph"


# ---------------------------------------------------------------------------
# Heuristic root
# ---------------------------------------------------------------------------


class TestHeuristicRoot:
    """Test _get_heuristic_root logic."""

    def test_picks_common_name(self):
        """Picks 'world' over arbitrary names."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        graph.add_transform("world", "A", tf.Translation(x=1))
        assert vis._get_heuristic_root(graph) == "world"

    def test_world_preferred_over_map(self):
        """'world' is preferred over 'map' when both present."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        graph.add_transform("world", "A", tf.Translation(x=1))
        graph.add_transform("map", "world", tf.Translation(x=1))
        assert vis._get_heuristic_root(graph) == "world"


# ---------------------------------------------------------------------------
# Axis traces
# ---------------------------------------------------------------------------


class TestAxisTraces:
    """Test axis trace creation for various transform types."""

    def test_matrix_transform(self):
        """MatrixTransform axis traces have correct origin."""
        mat = np.eye(4)
        mat[0, 3] = 10.0
        mt = tf.MatrixTransform(mat)
        traces = vis.create_axis_traces(mt, name="test")
        assert len(traces) == 4
        assert traces[0].x[0] == 10.0


# ---------------------------------------------------------------------------
# Frustum visualization
# ---------------------------------------------------------------------------


class TestFrustum:
    """Test frustum rendering with camera graphs."""

    def test_realistic_camera(self, K_realistic, D_realistic):
        """Frustum with realistic camera stays within reasonable bounds."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "robot", tf.Transform(translation=[0, 0, 0]))
        graph.add_transform("robot", "camera_link", tf.Transform(translation=[0.5, 0, 1.5]))
        graph.add_transform(
            "camera_link",
            "camera_optical",
            tf.Translation(z=0.1) * tf.Rotation(x=-0.5, y=0.5, z=-0.5, w=0.5),
        )
        graph.add_transform(
            "camera_optical",
            "IMAGE_front",
            tf.CameraProjection(
                intrinsic_matrix=K_realistic,
                dist_coeffs=D_realistic,
                image_size=(1920, 1080),
            ),
        )
        fig = vis.visualize_transform_graph(
            graph,
            target_frame="world",
            show_frustums=True,
            frustum_scale=0.5,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        all_x, all_y, all_z = [], [], []
        for trace in fig.data:
            if hasattr(trace, "x") and trace.x is not None:
                all_x.extend(x for x in trace.x if x is not None)
            if hasattr(trace, "y") and trace.y is not None:
                all_y.extend(y for y in trace.y if y is not None)
            if hasattr(trace, "z") and trace.z is not None:
                all_z.extend(z for z in trace.z if z is not None)

        max_extent = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z),
        )
        assert max_extent < 10, f"Frustum too large: {max_extent:.2f}m"

    def test_different_focal_lengths(self):
        """Wide-angle and telephoto frustums produce enough traces."""
        graph = tf.TransformGraph()

        K_wide = np.array([[300, 0, 640], [0, 300, 360], [0, 0, 1]])
        graph.add_transform("world", "wide_mount", tf.Translation(y=-2))
        graph.add_transform(
            "wide_mount", "wide_cam", tf.CameraProjection(K=K_wide, image_size=(1280, 720))
        )

        K_tele = np.array([[2000, 0, 640], [0, 2000, 360], [0, 0, 1]])
        graph.add_transform("world", "tele_mount", tf.Translation(y=2))
        graph.add_transform(
            "tele_mount", "tele_cam", tf.CameraProjection(K=K_tele, image_size=(1280, 720))
        )

        fig = vis.visualize_transform_graph(
            graph,
            target_frame="world",
            show_frustums=True,
            frustum_scale=1.0,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 20

    def test_frustum_scale(self, K_simple):
        """Different frustum_scale values produce valid figures."""
        cam = tf.CameraProjection(K=K_simple, image_size=(640, 480))
        graph = tf.TransformGraph()
        graph.add_transform("world", "camera", cam)

        for scale in [0.3, 0.5, 1.0, 2.0]:
            fig = vis.visualize_transform_graph(
                graph,
                target_frame="world",
                show_frustums=True,
                frustum_scale=scale,
            )
            assert isinstance(fig, go.Figure)
