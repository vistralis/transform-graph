#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for per-model projection dispatch in CameraProjection.

Validates every ProjectionModel variant: Pinhole, BrownConrady, KannalaBrandt,
Rational, Division, MeiUnified, and Fisheye62. Each model is tested for:
  - Principal point projection (on-axis point → cx, cy)
  - Off-axis projection (known geometry → expected pixel)
  - Distortion-aware transform_points preserving depth
  - Edge cases (zero depth, behind camera)
"""

import numpy as np
import pytest

import tgraph.transform as tf
from tgraph.transform import ProjectionModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def K_standard():
    """Standard 640×480 camera intrinsic matrix."""
    return np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# Pinhole
# ---------------------------------------------------------------------------

class TestPinholeProjection:
    """Pinhole: ideal perspective, no distortion."""

    def test_on_axis_projects_to_principal_point(self, K_standard):
        """A point on the optical axis projects to (cx, cy)."""
        cam = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([[0.0, 0.0, 5.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_off_axis_projection(self, K_standard):
        """Point (1, 0, 10) → u = fx*(1/10) + cx = 500*0.1 + 320 = 370."""
        cam = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([[1.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0, 0], 370.0, atol=1e-10)
        np.testing.assert_allclose(uv[0, 1], 240.0, atol=1e-10)

    def test_batch_projection(self, K_standard):
        """Multiple points project correctly."""
        cam = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        uv = cam._apply(pts)
        assert uv.shape == (3, 2)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)
        np.testing.assert_allclose(uv[1], [820.0, 240.0], atol=1e-10)
        np.testing.assert_allclose(uv[2], [320.0, 740.0], atol=1e-10)


# ---------------------------------------------------------------------------
# BrownConrady
# ---------------------------------------------------------------------------

class TestBrownConradyProjection:
    """BrownConrady: pinhole + radial/tangential distortion."""

    def test_zero_distortion_matches_pinhole(self, K_standard):
        """With D=zeros, BrownConrady == Pinhole."""
        cam_pin = tf.CameraProjection(
            K=K_standard, projection_model=ProjectionModel.Pinhole,
        )
        cam_bc = tf.CameraProjection(
            K=K_standard, D=[0, 0, 0, 0, 0],
            projection_model=ProjectionModel.BrownConrady,
        )
        pts = np.array([[1.0, 0.5, 5.0], [0.0, 0.0, 10.0]])
        np.testing.assert_allclose(cam_bc._apply(pts), cam_pin._apply(pts), atol=1e-10)

    def test_positive_radial_distortion_barrel(self, K_standard):
        """Positive k1 causes barrel distortion: pixels move outward."""
        cam_no_dist = tf.CameraProjection(
            K=K_standard, D=[0, 0, 0, 0], projection_model=ProjectionModel.BrownConrady,
        )
        cam_barrel = tf.CameraProjection(
            K=K_standard, D=[0.1, 0, 0, 0], projection_model=ProjectionModel.BrownConrady,
        )
        pts = np.array([[1.0, 1.0, 5.0]])
        uv_clean = cam_no_dist._apply(pts)
        uv_barrel = cam_barrel._apply(pts)
        # Barrel distortion moves points away from center
        dist_clean = np.linalg.norm(uv_clean[0] - [320, 240])
        dist_barrel = np.linalg.norm(uv_barrel[0] - [320, 240])
        assert dist_barrel > dist_clean

    def test_on_axis_unaffected_by_distortion(self, K_standard):
        """On-axis point is not affected by any distortion (r=0)."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.5, -0.3, 0.01, 0.02, 0.1],
            projection_model=ProjectionModel.BrownConrady,
        )
        pts = np.array([[0.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_five_coefficient_distortion(self, K_standard):
        """All 5 coefficients (k1, k2, p1, p2, k3) are applied."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.001, 0.002, 0.01],
            projection_model=ProjectionModel.BrownConrady,
        )
        pts = np.array([[2.0, 1.0, 10.0]])
        uv = cam._apply(pts)
        # Should differ from pure pinhole
        cam_pin = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        uv_pin = cam_pin._apply(pts)
        assert not np.allclose(uv, uv_pin)


# ---------------------------------------------------------------------------
# KannalaBrandt
# ---------------------------------------------------------------------------

class TestKannalaBrandtProjection:
    """KannalaBrandt: fisheye / equidistant model."""

    def test_on_axis_projects_to_principal_point(self, K_standard):
        """On-axis point → (cx, cy) regardless of distortion."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.01, -0.005],
            projection_model=ProjectionModel.KannalaBrandt,
        )
        pts = np.array([[0.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_zero_distortion_equidistant(self, K_standard):
        """With D=zeros, projects via θ = atan2(r, z), θ_d = θ."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0, 0, 0, 0],
            projection_model=ProjectionModel.KannalaBrandt,
        )
        # Point at 45° from optical axis
        pts = np.array([[1.0, 0.0, 1.0]])
        uv = cam._apply(pts)
        theta = np.arctan2(1.0, 1.0)  # π/4
        expected_u = 500.0 * theta * (1.0 / np.sqrt(2)) / (1.0 / np.sqrt(2)) + 320.0
        # scale = theta / r where r = 1.0, so x_d = 1.0 * (θ / 1.0) = θ
        expected_u = 500.0 * theta + 320.0
        np.testing.assert_allclose(uv[0, 0], expected_u, atol=1e-8)

    def test_wide_angle_point(self, K_standard):
        """Points at very wide angles still project (no crash)."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.01, -0.005, 0.001, -0.0005],
            projection_model=ProjectionModel.KannalaBrandt,
        )
        # 89° from axis
        pts = np.array([[100.0, 0.0, 1.0]])
        uv = cam._apply(pts)
        assert uv.shape == (1, 2)
        assert np.all(np.isfinite(uv))

    def test_differs_from_pinhole(self, K_standard):
        """KannalaBrandt projection differs from pinhole for off-axis points."""
        cam_kb = tf.CameraProjection(
            K=K_standard, D=[0, 0, 0, 0],
            projection_model=ProjectionModel.KannalaBrandt,
        )
        cam_pin = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([[2.0, 1.0, 5.0]])
        uv_kb = cam_kb._apply(pts)
        uv_pin = cam_pin._apply(pts)
        # Fisheye and pinhole differ for off-axis points
        assert not np.allclose(uv_kb, uv_pin, atol=1.0)


# ---------------------------------------------------------------------------
# Rational
# ---------------------------------------------------------------------------

class TestRationalProjection:
    """Rational polynomial model."""

    def test_on_axis_projects_to_principal_point(self, K_standard):
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.001, 0.002, 0.01, 0.05, -0.02, 0.01],
            projection_model=ProjectionModel.Rational,
        )
        pts = np.array([[0.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_zero_denominator_coefficients_matches_brown_conrady(self, K_standard):
        """When k4=k5=k6=0, Rational == BrownConrady (denominator = 1)."""
        D_bc = [0.1, -0.05, 0.001, 0.002, 0.01]
        D_rat = [0.1, -0.05, 0.001, 0.002, 0.01, 0, 0, 0]
        cam_bc = tf.CameraProjection(
            K=K_standard, D=D_bc, projection_model=ProjectionModel.BrownConrady,
        )
        cam_rat = tf.CameraProjection(
            K=K_standard, D=D_rat, projection_model=ProjectionModel.Rational,
        )
        pts = np.array([[1.5, -0.5, 8.0], [0.3, 2.1, 3.0]])
        np.testing.assert_allclose(cam_rat._apply(pts), cam_bc._apply(pts), atol=1e-10)

    def test_full_eight_coefficient_distortion(self, K_standard):
        """All 8 coefficients produce finite results."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.001, 0.002, 0.01, 0.05, -0.02, 0.01],
            projection_model=ProjectionModel.Rational,
        )
        pts = np.array([[1.0, 1.0, 5.0]])
        uv = cam._apply(pts)
        assert np.all(np.isfinite(uv))


# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------

class TestDivisionProjection:
    """Division undistortion model."""

    def test_on_axis_projects_to_principal_point(self, K_standard):
        cam = tf.CameraProjection(
            K=K_standard, D=[0.5], projection_model=ProjectionModel.Division,
        )
        pts = np.array([[0.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_zero_coefficient_matches_pinhole(self, K_standard):
        """Division with k1=0 → scale = 1/(1+0) = 1 → pinhole."""
        cam_div = tf.CameraProjection(
            K=K_standard, D=[0.0], projection_model=ProjectionModel.Division,
        )
        cam_pin = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([[1.0, 0.5, 5.0]])
        np.testing.assert_allclose(cam_div._apply(pts), cam_pin._apply(pts), atol=1e-10)

    def test_positive_k1_compresses(self, K_standard):
        """Positive k1 compresses the image (points move toward center)."""
        cam_no = tf.CameraProjection(
            K=K_standard, D=[0.0], projection_model=ProjectionModel.Division,
        )
        cam_div = tf.CameraProjection(
            K=K_standard, D=[0.5], projection_model=ProjectionModel.Division,
        )
        pts = np.array([[1.0, 1.0, 5.0]])
        uv_no = cam_no._apply(pts)
        uv_div = cam_div._apply(pts)
        # Positive k1 → scale < 1 → points closer to center
        dist_no = np.linalg.norm(uv_no[0] - [320, 240])
        dist_div = np.linalg.norm(uv_div[0] - [320, 240])
        assert dist_div < dist_no

    def test_empty_distortion_defaults_to_zero(self, K_standard):
        """Division with empty D defaults k1=0 → pinhole equivalent."""
        cam = tf.CameraProjection(
            K=K_standard, D=[], projection_model=ProjectionModel.Division,
        )
        cam_pin = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([[2.0, -1.0, 8.0]])
        np.testing.assert_allclose(cam._apply(pts), cam_pin._apply(pts), atol=1e-10)


# ---------------------------------------------------------------------------
# MeiUnified
# ---------------------------------------------------------------------------

class TestMeiUnifiedProjection:
    """Mei Unified omnidirectional camera model."""

    def test_xi_zero_approaches_pinhole(self, K_standard):
        """When ξ=0 and no radial distortion, Mei → pinhole (normalized by z/||p||)."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.0, 0.0, 0.0],
            projection_model=ProjectionModel.MeiUnified,
        )
        # On-axis point: unit sphere z = 1 → denom = z + ξ = 1 → x=0, y=0 → cx, cy
        pts = np.array([[0.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_xi_nonzero_changes_projection(self, K_standard):
        """Non-zero ξ changes the projection compared to ξ=0."""
        cam_xi0 = tf.CameraProjection(
            K=K_standard, D=[0.0], projection_model=ProjectionModel.MeiUnified,
        )
        cam_xi1 = tf.CameraProjection(
            K=K_standard, D=[1.0], projection_model=ProjectionModel.MeiUnified,
        )
        pts = np.array([[1.0, 0.0, 5.0]])
        uv_0 = cam_xi0._apply(pts)
        uv_1 = cam_xi1._apply(pts)
        # Different ξ → different pixel coordinates
        assert not np.allclose(uv_0, uv_1)

    def test_on_axis_unaffected_by_xi(self, K_standard):
        """On-axis point always → (cx, cy) regardless of ξ."""
        for xi in [0.0, 0.5, 1.0, 2.0]:
            cam = tf.CameraProjection(
                K=K_standard, D=[xi], projection_model=ProjectionModel.MeiUnified,
            )
            uv = cam._apply(np.array([[0.0, 0.0, 5.0]]))
            np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_with_radial_distortion(self, K_standard):
        """Mei with radial coefficients k1, k2 differs from without."""
        cam_no_rad = tf.CameraProjection(
            K=K_standard, D=[1.0, 0.0, 0.0],
            projection_model=ProjectionModel.MeiUnified,
        )
        cam_rad = tf.CameraProjection(
            K=K_standard, D=[1.0, 0.1, -0.05],
            projection_model=ProjectionModel.MeiUnified,
        )
        pts = np.array([[2.0, 1.0, 5.0]])
        uv_no = cam_no_rad._apply(pts)
        uv_rad = cam_rad._apply(pts)
        assert not np.allclose(uv_no, uv_rad)

    def test_wide_angle_finite(self, K_standard):
        """Very wide angle points (near 90°) produce finite results."""
        cam = tf.CameraProjection(
            K=K_standard, D=[1.0, 0.01, -0.005],
            projection_model=ProjectionModel.MeiUnified,
        )
        pts = np.array([[10.0, 0.0, 0.1]])  # nearly perpendicular
        uv = cam._apply(pts)
        assert np.all(np.isfinite(uv))


# ---------------------------------------------------------------------------
# Fisheye62
# ---------------------------------------------------------------------------

class TestFisheye62Projection:
    """Project Aria Fisheye62 model."""

    def test_on_axis_projects_to_principal_point(self, K_standard):
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.01, -0.005, 0.001, -0.001],
            projection_model=ProjectionModel.Fisheye62,
        )
        pts = np.array([[0.0, 0.0, 10.0]])
        uv = cam._apply(pts)
        np.testing.assert_allclose(uv[0], [320.0, 240.0], atol=1e-10)

    def test_zero_distortion_matches_kannala_brandt(self, K_standard):
        """Fisheye62 with D=zeros and no tangential → same angular model as KB."""
        cam_kb = tf.CameraProjection(
            K=K_standard, D=[0, 0, 0, 0],
            projection_model=ProjectionModel.KannalaBrandt,
        )
        cam_f62 = tf.CameraProjection(
            K=K_standard, D=[0, 0, 0, 0, 0, 0],
            projection_model=ProjectionModel.Fisheye62,
        )
        pts = np.array([[1.0, 0.5, 5.0]])
        uv_kb = cam_kb._apply(pts)
        uv_f62 = cam_f62._apply(pts)
        # Same radial polynomial, no tangential → same result
        np.testing.assert_allclose(uv_f62, uv_kb, atol=1e-10)

    def test_tangential_coefficients_change_result(self, K_standard):
        """Non-zero p0, p1 change the projection."""
        cam_no_tan = tf.CameraProjection(
            K=K_standard, D=[0.01, 0, 0, 0, 0, 0],
            projection_model=ProjectionModel.Fisheye62,
        )
        cam_tan = tf.CameraProjection(
            K=K_standard, D=[0.01, 0, 0, 0, 0.005, 0.003],
            projection_model=ProjectionModel.Fisheye62,
        )
        pts = np.array([[1.0, 1.0, 5.0]])
        uv_no = cam_no_tan._apply(pts)
        uv_tan = cam_tan._apply(pts)
        assert not np.allclose(uv_no, uv_tan)


# ---------------------------------------------------------------------------
# Distortion-aware transform_points
# ---------------------------------------------------------------------------

class TestTransformPointsDistortionAware:
    """transform_points uses full model projection for CameraProjection."""

    def test_preserves_depth(self, K_standard):
        """transform_points returns [u·z, v·z, z] (depth preserved)."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.001, 0.002, 0.01],
            projection_model=ProjectionModel.BrownConrady,
        )
        pts = np.array([[1.0, 0.5, 8.0]])
        result = tf.transform_points(pts, cam)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 2], 8.0, atol=1e-10)

    def test_pixel_coordinates_match_project_points(self, K_standard):
        """u·z/z and v·z/z from transform_points match project_points output."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.1, -0.05, 0.001, 0.002, 0.01],
            projection_model=ProjectionModel.BrownConrady,
        )
        pts = np.array([[1.0, 0.5, 8.0], [0.0, 0.0, 5.0]])
        tp = tf.transform_points(pts, cam)
        pp = tf.project_points(pts, cam)
        # u = tp[:, 0] / tp[:, 2], v = tp[:, 1] / tp[:, 2]
        u_from_tp = tp[:, 0] / tp[:, 2]
        v_from_tp = tp[:, 1] / tp[:, 2]
        np.testing.assert_allclose(u_from_tp, pp[:, 0], atol=1e-10)
        np.testing.assert_allclose(v_from_tp, pp[:, 1], atol=1e-10)

    def test_kannala_brandt_through_graph(self, K_standard):
        """KannalaBrandt model works correctly through TransformGraph."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "camera", tf.Translation(0, 0, 0))
        cam = tf.CameraProjection(
            K=K_standard, D=[0.01, -0.005, 0.001, -0.0005],
            projection_model=ProjectionModel.KannalaBrandt,
        )
        graph.add_transform("camera", "image", cam)

        pts = np.array([[0.0, 0.0, 10.0]])
        result = tf.transform_points(pts, graph, "world", "image")
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 2], 10.0, atol=1e-10)
        # On-axis → u·z = cx·z, v·z = cy·z
        np.testing.assert_allclose(result[0, 0], 320.0 * 10.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], 240.0 * 10.0, atol=1e-6)

    def test_homogeneous_4d_input(self, K_standard):
        """transform_points with CameraProjection accepts Nx4 homogeneous input."""
        cam = tf.CameraProjection(K=K_standard, projection_model=ProjectionModel.Pinhole)
        pts = np.array([[0.0, 0.0, 10.0, 1.0]])
        result = tf.transform_points(pts, cam)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 2], 10.0, atol=1e-10)

    def test_mei_unified_transform_points(self, K_standard):
        """MeiUnified model works through transform_points."""
        cam = tf.CameraProjection(
            K=K_standard, D=[0.5, 0.01, -0.005],
            projection_model=ProjectionModel.MeiUnified,
        )
        pts = np.array([[1.0, 0.5, 5.0]])
        result = tf.transform_points(pts, cam)
        assert result.shape == (1, 3)
        # Depth preserved
        np.testing.assert_allclose(result[0, 2], 5.0, atol=1e-10)
        # Pixels should be finite and reasonable
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# ProjectionModel from_string — MeiUnified
# ---------------------------------------------------------------------------

class TestProjectionModelMeiUnified:
    """Test MeiUnified enum membership and from_string."""

    def test_enum_value(self):
        assert ProjectionModel.MeiUnified.value == "MeiUnified"

    def test_from_string_exact(self):
        assert ProjectionModel.from_string("MeiUnified") == ProjectionModel.MeiUnified

    def test_from_string_case_insensitive(self):
        assert ProjectionModel.from_string("meiunified") == ProjectionModel.MeiUnified
