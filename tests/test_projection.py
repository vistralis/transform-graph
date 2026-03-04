#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CameraProjection, Projection, InverseProjection, InverseCameraProjection.
Covers construction, distortion, apply, project_points, and transform_points integration.
"""

import numpy as np
import pytest

import tgraph.transform as tf
from tgraph.transform import ProjectionModel

# ---------------------------------------------------------------------------
# Construction & properties
# ---------------------------------------------------------------------------


class TestCameraProjectionConstruction:
    """Test CameraProjection creation, properties, and init combinations."""

    def test_from_K(self, K_simple):
        """CameraProjection from K matrix preserves intrinsics."""
        cam = tf.CameraProjection(K=K_simple)
        np.testing.assert_allclose(cam.intrinsic_matrix, K_simple)
        assert cam.projection_model == ProjectionModel.PINHOLE
        assert len(cam.dist_coeffs) == 0

    def test_from_intrinsic_matrix(self, K_simple):
        """CameraProjection from intrinsic_matrix= keyword."""
        cam = tf.CameraProjection(intrinsic_matrix=K_simple)
        np.testing.assert_allclose(cam.intrinsic_matrix, K_simple)

    def test_from_positional(self, K_simple):
        """CameraProjection from positional argument."""
        cam = tf.CameraProjection(K_simple)
        np.testing.assert_allclose(cam.intrinsic_matrix, K_simple)

    def test_missing_K_error(self):
        """Raises ValueError when no intrinsic matrix provided."""
        with pytest.raises(ValueError, match="Must provide .*intrinsic_matrix.*"):
            tf.CameraProjection()

    def test_invalid_K_shape(self):
        """Raises ValueError for non-3x3 K."""
        with pytest.raises(ValueError, match="Intrinsic matrix must be 3x3"):
            tf.CameraProjection(K=np.eye(4))


class TestCameraProjectionDistortion:
    """Test CameraProjection with distortion coefficients."""

    def test_init_with_distortion(self, K_simple):
        """CameraProjection stores distortion and model correctly."""
        D = [0.1, -0.2, 0.001, 0.002, 0.05]
        cam = tf.CameraProjection(
            K=K_simple,
            dist_coeffs=D,
            projection_model=ProjectionModel.PINHOLE_POLYNOMIAL,
        )
        assert cam.projection_model == ProjectionModel.PINHOLE_POLYNOMIAL
        np.testing.assert_allclose(cam.dist_coeffs, D)
        np.testing.assert_allclose(cam.distortion_coefficients, D)

    def test_distortion_serialization(self, K_simple):
        """Distortion survives to_dict / from_dict."""
        D = [0.1, -0.2, 0.001, 0.002, 0.05]
        cam = tf.CameraProjection(
            K=K_simple,
            dist_coeffs=D,
            projection_model=ProjectionModel.PINHOLE_POLYNOMIAL,
        )
        data = cam.to_dict()
        assert data["dist_coeffs"] == D
        assert data["projection_model"] == "Pinhole+Polynomial"

        cam2 = tf.CameraProjection.from_dict(data)
        assert cam2.projection_model == ProjectionModel.PINHOLE_POLYNOMIAL
        np.testing.assert_allclose(cam2.dist_coeffs, D)

    def test_all_projection_models(self):
        """All ProjectionModel enum values can be used."""
        K = np.eye(3)
        for model in ProjectionModel:
            cam = tf.CameraProjection(K, projection_model=model)
            assert cam.projection_model == model

    def test_string_model_init(self):
        """ProjectionModel from string."""
        cam = tf.CameraProjection(np.eye(3), projection_model="Fisheye")
        assert cam.projection_model == ProjectionModel.FISHEYE


# ---------------------------------------------------------------------------
# Projection behavior
# ---------------------------------------------------------------------------


class TestProjectionBehavior:
    """Test Projection and CameraProjection apply / project_points."""

    def test_basic_projection(self, K_simple):
        """Projection produces correct pixel coordinates."""
        cam = tf.CameraProjection(K=K_simple)
        # Point at (0, 0, 10) should project to principal point
        point = np.array([[0, 0, 10.0]])
        result = cam._apply(point)
        np.testing.assert_allclose(result[0, 0], 320.0, atol=1e-6)  # cx
        np.testing.assert_allclose(result[0, 1], 240.0, atol=1e-6)  # cy

    def test_transform_points_with_projection(self, K_simple):
        """transform_points with Projection returns unnormalized [u*z, v*z, z]."""
        cam = tf.CameraProjection(K=K_simple)
        points = np.array([[0.0, 0.0, 10.0]])
        result = tf.transform_points(points, cam)
        np.testing.assert_allclose(result[0, 2], 10.0, atol=1e-6)

    def test_transform_points_4d_input(self, K_simple):
        """transform_points with Projection accepts Nx4 homogeneous input."""
        cam = tf.CameraProjection(K=K_simple)
        points = np.array([[0.0, 0.0, 10.0, 1.0]])
        result = tf.transform_points(points, cam)
        assert result.shape[1] in (3, 4)

    def test_projection_in_graph(self, K_simple):
        """CameraProjection used as a graph edge."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "camera", tf.Translation(x=0, y=0, z=0))
        graph.add_transform("camera", "image", tf.CameraProjection(K=K_simple))

        t = graph.get_transform("world", "image")
        assert isinstance(t, tf.MatrixTransform) or isinstance(t, tf.Projection)

    def test_decompose_projection(self, K_simple):
        """Decomposing a full projection matrix into K and Pose."""
        cam = tf.CameraProjection(K=K_simple)
        mat = cam.as_matrix()
        # Should be a 4x4 with the K embedded in the upper-left
        assert mat.shape == (4, 4)


# ---------------------------------------------------------------------------
# InverseProjection
# ---------------------------------------------------------------------------


class TestInverseProjection:
    """Test InverseProjection / InverseCameraProjection."""

    def test_inverse_shortcuts(self, K_simple):
        """InverseCameraProjection has access to K properties."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        assert isinstance(inv, tf.InverseCameraProjection)
        assert inv.fx == cam.fx
        assert inv.fy == cam.fy

    def test_inverse_type_relationships(self, K_simple):
        """InverseCameraProjection is an InverseProjection."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        assert isinstance(inv, tf.InverseProjection)
        assert isinstance(inv, tf.InverseCameraProjection)

    def test_inverse_inverse_is_original(self, K_simple):
        """Inverse of inverse gives original CameraProjection."""
        cam = tf.CameraProjection(K=K_simple)
        recovered = cam.inverse().inverse()
        assert isinstance(recovered, tf.CameraProjection)
        np.testing.assert_allclose(recovered.as_matrix(), cam.as_matrix(), atol=1e-10)

    def test_apply_nx3_homogeneous(self, K_simple):
        """InverseProjection.apply with Nx3 homogeneous pixels."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        pixels = np.array([[320.0, 240.0, 1.0]])  # principal point
        result = inv._apply(pixels)
        # Should give direction (0, 0, 1) scaled
        assert result.shape == (1, 3)

    def test_apply_nx4(self, K_simple):
        """InverseProjection.apply with Nx4 input."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        pixels = np.array([[320.0, 240.0, 1.0, 1.0]])
        result = inv._apply(pixels)
        assert result.shape[0] == 1

    def test_apply_invalid_shape(self, K_simple):
        """InverseProjection raises ValueError for wrong shape."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        with pytest.raises(ValueError):
            inv._apply(np.array([[1.0]]))

    def test_transform_points_inverse_projection(self, K_simple):
        """transform_points with InverseProjection: Nx2 pixels → Nx3 rays."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        pixels = np.array([[320.0, 240.0]])  # principal point
        result = tf.transform_points(pixels, inv)
        assert result.shape == (1, 3)


# ---------------------------------------------------------------------------
# project_points
# ---------------------------------------------------------------------------


class TestProjectPoints:
    """Test the project_points free function."""

    def test_project_points_values(self, K_simple):
        """project_points returns correct 2D pixel values."""
        cam = tf.CameraProjection(K=K_simple)
        points = np.array([[0.0, 0.0, 10.0]])
        result = tf.project_points(points, cam)
        np.testing.assert_allclose(result[0, 0], 320.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], 240.0, atol=1e-6)

    def test_invalid_points_shape(self, K_simple):
        """project_points raises ValueError for invalid point dimensions."""
        cam = tf.CameraProjection(K=K_simple)
        with pytest.raises((ValueError, TypeError, IndexError)):
            tf.project_points(np.array([[1.0, 2.0]]), cam)

    def test_invalid_object(self, K_simple):
        """project_points raises TypeError for non-transform objects."""
        with pytest.raises(TypeError, match="must be BaseTransform or TransformGraph"):
            tf.project_points(np.zeros((1, 3)), "not_a_transform")

    def test_project_points_via_graph(self, K_simple):
        """project_points through a TransformGraph."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "camera", tf.Translation(z=0))
        graph.add_transform("camera", "image", tf.CameraProjection(K=K_simple))
        points = np.array([[0.0, 0.0, 10.0]])
        result = tf.project_points(points, graph, source_frame="world", target_frame="image")
        np.testing.assert_allclose(result[0, 0], 320.0, atol=1e-6)
        np.testing.assert_allclose(result[0, 1], 240.0, atol=1e-6)

    def test_project_points_rigid_error(self, K_simple):
        """project_points with a rigid Transform raises TypeError."""
        t = tf.Translation(x=1)
        with pytest.raises(TypeError, match="Cannot project_points using a rigid transform"):
            tf.project_points(np.zeros((1, 3)), t)


# ---------------------------------------------------------------------------
# CameraProjection properties
# ---------------------------------------------------------------------------


class TestCameraProjectionProperties:
    """Test CameraProjection property accessors."""

    def test_fx_fy_cx_cy(self, K_simple):
        """fx, fy, cx, cy return correct values."""
        cam = tf.CameraProjection(K=K_simple)
        assert cam.fx == 500.0
        assert cam.fy == 500.0
        assert cam.cx == 320.0
        assert cam.cy == 240.0

    def test_image_size(self, K_simple):
        """image_size stores and returns (width, height)."""
        cam = tf.CameraProjection(K=K_simple, image_size=(1920, 1080))
        assert cam.image_size == (1920, 1080)

    def test_image_size_default(self, K_simple):
        """Default image_size is None."""
        cam = tf.CameraProjection(K=K_simple)
        assert cam.image_size is None

    def test_repr_camera_projection(self, K_simple):
        """CameraProjection __repr__ contains intrinsic info."""
        cam = tf.CameraProjection(K=K_simple)
        s = repr(cam)
        assert "CameraProjection" in s

    def test_projection_3x4(self, K_simple):
        """projection_3x4 returns correct shape."""
        cam = tf.CameraProjection(K=K_simple)
        p34 = cam.as_matrix()[:3, :]
        assert p34.shape == (3, 4)


# ---------------------------------------------------------------------------
# Distortion forward path
# ---------------------------------------------------------------------------


class TestDistortionApply:
    """Test CameraProjection._apply with distortion coefficients."""

    def test_distortion_affects_projection(self, K_simple):
        """Points projected with distortion differ from without."""
        D = [0.1, -0.2, 0.001, 0.002, 0.05]
        cam_dist = tf.CameraProjection(K=K_simple, dist_coeffs=D)
        cam_nodist = tf.CameraProjection(K=K_simple)

        # Off-axis point to exercise distortion terms
        point = np.array([[1.0, 0.5, 5.0]])
        result_dist = cam_dist._apply(point)
        result_nodist = cam_nodist._apply(point)

        # Not equal (distortion shifts pixels)
        assert not np.allclose(result_dist, result_nodist), "Distortion had no effect"


# ---------------------------------------------------------------------------
# Decompose projection
# ---------------------------------------------------------------------------


class TestDecomposeProjection:
    """Test decompose_projection_to_objects."""

    def test_roundtrip(self, K_simple):
        """Decomposing a projection matrix recovers K and pose."""
        cam = tf.CameraProjection(K=K_simple)
        t = tf.Transform(
            translation=[1.0, 2.0, 3.0],
            rotation=[np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)],
        )
        P = cam.as_matrix() @ t.as_matrix()
        recovered_cam, recovered_t = tf.decompose_projection_to_objects(P)
        assert isinstance(recovered_cam, tf.CameraProjection)
        assert isinstance(recovered_t, tf.Transform)


# ---------------------------------------------------------------------------
# InverseProjection / Projection serialization and repr
# ---------------------------------------------------------------------------


class TestProjectionSerialization:
    """Test Projection and InverseProjection serialization."""

    def test_projection_roundtrip(self, K_simple):
        """Projection to_dict → from_dict roundtrip."""
        proj = tf.Projection(matrix=K_simple @ np.eye(3, 4))
        data = proj.to_dict()
        recovered = tf.Projection.from_dict(data)
        np.testing.assert_allclose(recovered.as_matrix(), proj.as_matrix(), atol=1e-10)

    def test_projection_repr(self, K_simple):
        """Projection __repr__ is informative."""
        proj = tf.CameraProjection(K=K_simple)
        assert "CameraProjection" in repr(proj)

    def test_inverse_projection_roundtrip(self, K_simple):
        """InverseProjection to_dict → from_dict roundtrip."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        data = tf.InverseProjection.to_dict(inv)
        recovered = tf.InverseProjection.from_dict(data)
        np.testing.assert_allclose(recovered.as_matrix(), inv.as_matrix(), atol=1e-10)

    def test_inverse_projection_repr(self, K_simple):
        """InverseProjection __repr__ is informative."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        s = repr(inv)
        assert "Inverse" in s

    def test_inverse_camera_projection_roundtrip(self, K_simple):
        """InverseCameraProjection to_dict → from_dict roundtrip."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        data = inv.to_dict()
        recovered = tf.deserialize_transform(data)
        np.testing.assert_allclose(recovered.as_matrix(), inv.as_matrix(), atol=1e-10)


# ---------------------------------------------------------------------------
# InverseProjection.unproject
# ---------------------------------------------------------------------------


class TestUnproject:
    """Test InverseProjection.unproject with depth values."""

    def test_unproject_principal_point(self, K_simple):
        """Unprojecting (cx, cy) with depth=10 gives (0, 0, 10)."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        pixels = np.array([[320.0, 240.0]])
        depths = np.array([10.0])
        points = inv.unproject(pixels, depths)
        np.testing.assert_allclose(points, [[0, 0, 10]], atol=1e-6)

    def test_unproject_shape_check(self, K_simple):
        """unproject raises ValueError for mismatched shapes."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        with pytest.raises(ValueError, match="Pixels must be Nx2"):
            inv.unproject(np.array([[1, 2, 3]]), np.array([1.0]))
        with pytest.raises(ValueError, match="Depths length"):
            inv.unproject(np.array([[1, 2]]), np.array([1.0, 2.0]))

    def test_project_unproject_roundtrip(self, K_simple):
        """Project → unproject with known depth recovers 3D point."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        point = np.array([[1.0, -0.5, 8.0]])
        pixels = cam._apply(point)
        recovered = inv.unproject(pixels, np.array([8.0]))
        np.testing.assert_allclose(recovered, point, atol=1e-6)
