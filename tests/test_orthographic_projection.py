# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OrthographicProjection and InverseOrthographicProjection."""

import numpy as np
import pytest

from tgraph import (
    InverseOrthographicProjection,
    OrthographicProjection,
    TransformGraph,
    Translation,
    deserialize_transform,
    project_points,
    transform_points,
)

# ------------------------------------------------------------------ #
# Construction & properties                                           #
# ------------------------------------------------------------------ #


class TestOrthographicProjectionConstruction:
    """Test construction, properties, and repr."""

    def test_default_top(self):
        p = OrthographicProjection()
        assert p.axis == "top"
        assert p.u_range == (-50.0, 50.0)
        assert p.v_range == (-50.0, 50.0)
        assert p.resolution == 0.1

    def test_grid_shape(self):
        p = OrthographicProjection("top", (-10, 10), (-20, 20), 0.5)
        H, W = p.grid_shape
        assert W == 40  # u_range = 20m / 0.5 = 40
        assert H == 80  # v_range = 40m / 0.5 = 80

    def test_origin_pixel(self):
        p = OrthographicProjection("top", (-10, 10), (-10, 10), 1.0)
        col, row = p.origin_pixel
        # For top-down: col = (y_max - 0) / res = 10, row = (x_max - 0) / res = 10
        assert col == 10
        assert row == 10

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="Unknown axis"):
            OrthographicProjection("diagonal")

    def test_repr(self):
        p = OrthographicProjection("top", (-5, 5), (-5, 5), 0.5)
        r = repr(p)
        assert "OrthographicProjection" in r
        assert "top" in r

    def test_matrix_shape(self):
        p = OrthographicProjection()
        assert p.as_matrix().shape == (4, 4)


# ------------------------------------------------------------------ #
# project_points — top-down BEV                                      #
# ------------------------------------------------------------------ #


class TestOrthographicProjectionTopDown:
    """Test top-down (BEV) projection with known points."""

    @pytest.fixture
    def bev(self):
        return OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)

    def test_origin_projects_to_center(self, bev):
        pts = np.array([[0.0, 0.0, 0.0]])
        px = project_points(pts, bev)
        # col = (50 - 0) / 0.1 = 500, row = (50 - 0) / 0.1 = 500
        np.testing.assert_allclose(px[0], [500.0, 500.0])

    def test_forward_point_projects_up(self, bev):
        """A point at x=10 (forward) should have a smaller row (toward top)."""
        pts = np.array([[10.0, 0.0, 0.0]])
        px = project_points(pts, bev)
        # row = (50 - 10) / 0.1 = 400
        assert px[0, 1] == pytest.approx(400.0)
        # col stays at center
        assert px[0, 0] == pytest.approx(500.0)

    def test_left_point_projects_left(self, bev):
        """A point at y=5 (left) should have a smaller col (toward left)."""
        pts = np.array([[0.0, 5.0, 0.0]])
        px = project_points(pts, bev)
        # col = (50 - 5) / 0.1 = 450
        assert px[0, 0] == pytest.approx(450.0)

    def test_z_is_ignored(self, bev):
        """Z should not affect pixel coordinates (orthographic)."""
        pts_low = np.array([[5.0, 5.0, -3.0]])
        pts_high = np.array([[5.0, 5.0, 10.0]])
        np.testing.assert_allclose(project_points(pts_low, bev), project_points(pts_high, bev))

    def test_batch_projection(self, bev):
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ]
        )
        px = project_points(pts, bev)
        assert px.shape == (3, 2)

    def test_homogeneous_input(self, bev):
        pts_3d = np.array([[10.0, 5.0, 0.0]])
        pts_4d = np.array([[10.0, 5.0, 0.0, 1.0]])
        np.testing.assert_allclose(project_points(pts_3d, bev), project_points(pts_4d, bev))


# ------------------------------------------------------------------ #
# project_points — front and side views                               #
# ------------------------------------------------------------------ #


class TestOrthographicProjectionFrontSide:
    """Test front and side axis presets."""

    def test_front_view(self):
        p = OrthographicProjection("front", (-10, 10), (-5, 5), 1.0)
        # Front: u=Y (col), v=Z (row)
        # Point at (x=99, y=3, z=2): col = (10-3)/1=7, row = (5-2)/1=3
        pts = np.array([[99.0, 3.0, 2.0]])
        px = project_points(pts, p)
        assert px[0, 0] == pytest.approx(7.0)
        assert px[0, 1] == pytest.approx(3.0)

    def test_side_view(self):
        p = OrthographicProjection("side", (-10, 10), (-5, 5), 1.0)
        # Side: u=X (col, no flip), v=Z (row, flipped)
        # Point at (x=3, y=99, z=2): col = (3-(-10))/1=13, row = (5-2)/1=3
        pts = np.array([[3.0, 99.0, 2.0]])
        px = project_points(pts, p)
        assert px[0, 0] == pytest.approx(13.0)
        assert px[0, 1] == pytest.approx(3.0)


# ------------------------------------------------------------------ #
# Inverse (round-trip)                                               #
# ------------------------------------------------------------------ #


class TestInverseOrthographicProjection:
    """Test inverse projection and round-trip accuracy."""

    def test_round_trip(self):
        p = OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)
        pts = np.array([[10.0, -5.0, 7.0]])  # z will be lost
        px = project_points(pts, p)
        pts_back = transform_points(px, p.inverse())

        # XY should round-trip, Z is zero (collapsed axis)
        np.testing.assert_allclose(pts_back[0, 0], pts[0, 0], atol=1e-10)  # x
        np.testing.assert_allclose(pts_back[0, 1], pts[0, 1], atol=1e-10)  # y
        assert pts_back[0, 2] == 0.0  # z collapsed

    def test_inverse_inverse(self):
        p = OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)
        p2 = p.inverse().inverse()
        assert isinstance(p2, OrthographicProjection)
        assert p2.axis == p.axis

    def test_inverse_repr(self):
        p = OrthographicProjection("front", (-10, 10), (-5, 5), 0.5)
        inv = p.inverse()
        assert isinstance(inv, InverseOrthographicProjection)
        assert "InverseOrthographicProjection" in repr(inv)

    def test_front_round_trip(self):
        p = OrthographicProjection("front", (-10, 10), (-5, 5), 0.5)
        pts = np.array([[99.0, 3.0, -2.0]])  # x is collapsed for front
        px = project_points(pts, p)
        pts_back = transform_points(px, p.inverse())
        # Y and Z should round-trip, X is zero
        np.testing.assert_allclose(pts_back[0, 1], pts[0, 1], atol=1e-10)
        np.testing.assert_allclose(pts_back[0, 2], pts[0, 2], atol=1e-10)
        assert pts_back[0, 0] == 0.0


# ------------------------------------------------------------------ #
# Serialization                                                      #
# ------------------------------------------------------------------ #


class TestOrthographicProjectionSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip(self):
        p = OrthographicProjection("top", (-30, 30), (-20, 20), 0.2)
        d = p.to_dict()
        assert d["type"] == "OrthographicProjection"
        p2 = OrthographicProjection.from_dict(d)
        assert p2.axis == p.axis
        assert p2.u_range == p.u_range
        assert p2.v_range == p.v_range
        assert p2.resolution == p.resolution

        # Same projection results
        pts = np.array([[5.0, -3.0, 1.0]])
        np.testing.assert_allclose(project_points(pts, p), project_points(pts, p2))

    def test_deserialize_via_registry(self):
        p = OrthographicProjection("side", (-10, 10), (-5, 5), 0.5)
        d = p.to_dict()
        p2 = deserialize_transform(d)
        assert isinstance(p2, OrthographicProjection)
        assert p2.axis == "side"

    def test_inverse_serialization(self):
        p = OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)
        inv = p.inverse()
        d = inv.to_dict()
        assert d["type"] == "InverseOrthographicProjection"
        inv2 = deserialize_transform(d)
        assert isinstance(inv2, InverseOrthographicProjection)


# ------------------------------------------------------------------ #
# TransformGraph integration                                         #
# ------------------------------------------------------------------ #


class TestOrthographicProjectionGraphIntegration:
    """Test using OrthographicProjection as a transform graph edge.

    Note: ``transform_points`` returns ``(N, 3)`` for projection edges
    by design (unnormalized ``[u*z, v*z, z]``).  For orthographic
    projections ``z == 1`` always, so ``result[:, :2]`` gives pixel coords.
    """

    def test_transform_points_through_graph(self):
        """lidar → ego → bev  should produce correct BEV pixel coordinates."""
        graph = TransformGraph()

        # ego → lidar is Translation(2, 0, 1)
        # So lidar → ego is Translation(-2, 0, -1)
        # A point at lidar origin (0,0,0) is at (-2, 0, -1) in ego frame
        graph.add_transform("ego", "lidar", Translation(2.0, 0.0, 1.0))

        # BEV projection from ego frame
        ortho = OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)
        graph.add_transform("ego", "bev", ortho)

        pts_lidar = np.array([[0.0, 0.0, 0.0]])
        result = transform_points(pts_lidar, graph, "lidar", "bev")

        # transform_points returns Nx3 for projections: [col, row, 1]
        assert result.shape == (1, 3)
        pixels = result[:, :2]

        # In ego frame: (-2, 0, -1)
        # BEV top: col = (50 - 0) / 0.1 = 500, row = (50 - (-2)) / 0.1 = 520
        np.testing.assert_allclose(pixels[0, 0], 500.0, atol=0.1)
        np.testing.assert_allclose(pixels[0, 1], 520.0, atol=0.1)
        # z-component should be 1.0 (orthographic)
        np.testing.assert_allclose(result[0, 2], 1.0, atol=1e-10)

    def test_multiple_sources_to_bev(self):
        """Multiple sensors should project correctly through the graph."""
        graph = TransformGraph()

        # ego → lidar: +2 forward, ego → radar: +3.5 forward
        graph.add_transform("ego", "lidar", Translation(2.0, 0.0, 1.0))
        graph.add_transform("ego", "radar", Translation(3.5, 0.0, 0.5))

        ortho = OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)
        graph.add_transform("ego", "bev", ortho)

        # Lidar origin in ego: (-2, 0, -1) → BEV: col=500, row=520
        px_lidar = transform_points(np.array([[0.0, 0.0, 0.0]]), graph, "lidar", "bev")[:, :2]
        # Radar origin in ego: (-3.5, 0, -0.5) → BEV: col=500, row=535
        px_radar = transform_points(np.array([[0.0, 0.0, 0.0]]), graph, "radar", "bev")[:, :2]

        # Both at col=500 (y=0)
        np.testing.assert_allclose(px_lidar[0, 0], 500.0, atol=0.1)
        np.testing.assert_allclose(px_radar[0, 0], 500.0, atol=0.1)
        # lidar: row = (50-(-2))/0.1 = 520, radar: row = (50-(-3.5))/0.1 = 535
        np.testing.assert_allclose(px_lidar[0, 1], 520.0, atol=0.1)
        np.testing.assert_allclose(px_radar[0, 1], 535.0, atol=0.1)


# ------------------------------------------------------------------ #
# Consistency with BEVProjection mapping                             #
# ------------------------------------------------------------------ #


class TestBEVConsistency:
    """Verify OrthographicProjection matches the existing BEVProjection mapping."""

    def test_matches_bev_manual_formula(self):
        """Compare against the hand-coded BEV formula."""
        x_range = (-50.0, 50.0)
        y_range = (-50.0, 50.0)
        res = 0.1

        ortho = OrthographicProjection("top", y_range, x_range, res)

        # The BEV formula from bev.py:
        #   col = (y_max - y) / res
        #   row = (x_max - x) / res
        test_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, -5.0, 3.0],
                [-20.0, 30.0, -1.0],
                [49.9, -49.9, 0.0],
            ]
        )

        for pt in test_points:
            x, y, z = pt
            expected_col = (y_range[1] - y) / res
            expected_row = (x_range[1] - x) / res

            px = project_points(pt.reshape(1, 3), ortho)
            np.testing.assert_allclose(
                px[0], [expected_col, expected_row], atol=1e-10, err_msg=f"Mismatch for point {pt}"
            )
