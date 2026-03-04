#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Targeted tests to push coverage toward 95%.

Covers: ProjectionModel.from_string edge cases, deserialize_transform errors,
BaseTransform._apply Nx4, Projection._apply Nx4, InverseProjection properties,
CameraProjection aliases, InverseCameraProjection properties, OrthographicProjection
edge paths, CompositeProjection/InverseCompositeProjection from_dict validation,
transform_points/project_points edge cases, get_basis_vectors, is_projection_frame.
"""

import numpy as np
import pytest

import tgraph.transform as tf
from tgraph.transform import ProjectionModel

# ---------------------------------------------------------------------------
# ProjectionModel.from_string
# ---------------------------------------------------------------------------


class TestProjectionModelFromString:
    """Test ProjectionModel.from_string edge cases."""

    def test_exact_value_match(self):
        """Exact enum value string matches."""
        assert ProjectionModel.from_string("Pinhole") == ProjectionModel.PINHOLE
        assert ProjectionModel.from_string("Fisheye") == ProjectionModel.FISHEYE

    def test_case_insensitive_name(self):
        """Case-insensitive name match (uppercase fallback path)."""
        assert ProjectionModel.from_string("pinhole") == ProjectionModel.PINHOLE
        assert ProjectionModel.from_string("PINHOLE") == ProjectionModel.PINHOLE

    def test_unknown_raises(self):
        """Unknown projection model string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown projection model"):
            ProjectionModel.from_string("UnknownModel")


# ---------------------------------------------------------------------------
# deserialize_transform error paths
# ---------------------------------------------------------------------------


class TestDeserializeErrors:
    """Test deserialize_transform error branches."""

    def test_missing_type_field(self):
        """Missing 'type' field raises ValueError."""
        with pytest.raises(ValueError, match="Missing 'type' field"):
            tf.deserialize_transform({"data": "something"})

    def test_empty_type_field(self):
        """Empty 'type' field raises ValueError."""
        with pytest.raises(ValueError, match="Missing 'type' field"):
            tf.deserialize_transform({"type": ""})


# ---------------------------------------------------------------------------
# BaseTransform._apply Nx4
# ---------------------------------------------------------------------------


class TestBaseTransformApply:
    """Test BaseTransform._apply with Nx4 homogeneous input."""

    def test_nx4_passthrough(self):
        """Nx4 input is multiplied without dehomogenization."""
        t = tf.Translation(x=5)
        points = np.array([[1.0, 0.0, 0.0, 1.0]])
        result = t._apply(points)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0, 0], 6.0)  # 1 + 5

    def test_invalid_shape(self):
        """Input with wrong column count raises ValueError."""
        t = tf.Translation(x=1)
        with pytest.raises(ValueError, match="Nx3 or Nx4"):
            t._apply(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# Projection._apply Nx4
# ---------------------------------------------------------------------------


class TestProjectionApplyNx4:
    """Test Projection._apply with Nx4 homogeneous input."""

    def test_nx4_input(self, K_simple):
        """Projection._apply with Nx4 homogeneous input."""
        proj = tf.Projection(matrix=K_simple @ np.eye(3, 4))
        points = np.array([[0.0, 0.0, 10.0, 1.0]])
        result = proj._apply(points)
        assert result.shape == (1, 2)

    def test_invalid_shape(self, K_simple):
        """Projection._apply with invalid shape raises ValueError."""
        proj = tf.Projection(matrix=K_simple @ np.eye(3, 4))
        with pytest.raises(ValueError, match="Nx3 or Nx4"):
            proj._apply(np.array([[1.0, 2.0]]))

    def test_projection_project_points(self, K_simple):
        """Projection.project_points alias calls _apply."""
        proj = tf.CameraProjection(K=K_simple)
        points = np.array([[0.0, 0.0, 10.0]])
        result = proj.project_points(points)
        assert result.shape == (1, 2)

    def test_projection_repr(self, K_simple):
        """Projection __repr__ contains class name."""
        proj = tf.Projection(matrix=K_simple @ np.eye(3, 4))
        assert "Projection" in repr(proj)


# ---------------------------------------------------------------------------
# InverseProjection properties and composition error paths
# ---------------------------------------------------------------------------


class TestInverseProjectionEdgeCases:
    """Test InverseProjection.original_matrix, inverse(), __mul__ errors."""

    def test_original_matrix_property(self, K_simple):
        """InverseProjection.original_matrix returns the stored matrix."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        np.testing.assert_allclose(inv.original_matrix, cam.as_matrix())

    def test_inverse_returns_projection(self, K_simple):
        """InverseProjection.inverse() returns a plain Projection."""
        inv = tf.InverseProjection(original_matrix=K_simple @ np.eye(3, 4))
        # Note: This returns a Projection, not CameraProjection
        recovered = inv.inverse()
        assert isinstance(recovered, tf.Projection)

    def test_mul_transform_error(self, K_simple):
        """InverseProjection * Transform raises TypeError."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        t = tf.Translation(x=1)
        with pytest.raises(TypeError, match="dimensional mismatch"):
            _ = inv * t

    def test_repr(self, K_simple):
        """InverseProjection __repr__ is informative."""
        inv = tf.InverseProjection(original_matrix=K_simple @ np.eye(3, 4))
        assert "InverseProjection" in repr(inv)


# ---------------------------------------------------------------------------
# CameraProjection aliases
# ---------------------------------------------------------------------------


class TestCameraProjectionAliases:
    """Test CameraProjection K, D, from_intrinsics_and_transform."""

    def test_K_alias(self, K_simple):
        """CameraProjection.K returns intrinsic matrix."""
        cam = tf.CameraProjection(K=K_simple)
        np.testing.assert_allclose(cam.K, K_simple)

    def test_D_alias(self, K_simple):
        """CameraProjection.D returns distortion coefficients."""
        D = [0.1, -0.2, 0.001, 0.002]
        cam = tf.CameraProjection(K=K_simple, dist_coeffs=D)
        np.testing.assert_allclose(cam.D, D)

    def test_from_intrinsics_and_transform_disabled(self):
        """from_intrinsics_and_transform raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Intrinsic-only"):
            tf.CameraProjection.from_intrinsics_and_transform()


# ---------------------------------------------------------------------------
# InverseCameraProjection properties and from_dict branch
# ---------------------------------------------------------------------------


class TestInverseCameraProjectionProperties:
    """Test InverseCameraProjection shortcuts and from_dict edge case."""

    def test_camera_projection_property(self, K_simple):
        """camera_projection returns the original CameraProjection."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        assert inv.camera_projection is cam

    def test_cx_cy_shortcuts(self, K_simple):
        """cx, cy shortcuts delegate to original CameraProjection."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        assert inv.cx == 320.0
        assert inv.cy == 240.0

    def test_intrinsic_matrix(self, K_simple):
        """intrinsic_matrix delegates to original CameraProjection."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        np.testing.assert_allclose(inv.intrinsic_matrix, K_simple)

    def test_from_dict_adds_type(self, K_simple):
        """from_dict fixes missing type in camera_projection sub-dict."""
        cam = tf.CameraProjection(K=K_simple)
        inv = cam.inverse()
        data = inv.to_dict()
        # Remove type from nested dict to test fallback
        del data["camera_projection"]["type"]
        recovered = tf.InverseCameraProjection.from_dict(data)
        np.testing.assert_allclose(recovered.as_matrix(), inv.as_matrix(), atol=1e-10)

    def test_repr(self, K_simple):
        """InverseCameraProjection __repr__ contains class name."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        assert "InverseCameraProjection" in repr(inv)


# ---------------------------------------------------------------------------
# CameraProjection image_size serialization
# ---------------------------------------------------------------------------


class TestCameraProjectionImageSizeSerialization:
    """Test CameraProjection to_dict/from_dict with image_size."""

    def test_with_image_size(self, K_simple):
        """image_size survives serialization roundtrip."""
        cam = tf.CameraProjection(K=K_simple, image_size=(1920, 1080))
        data = cam.to_dict()
        assert data["image_size"] == [1920, 1080]
        recovered = tf.CameraProjection.from_dict(data)
        assert recovered.image_size == (1920, 1080)

    def test_without_image_size(self, K_simple):
        """No image_size in serialization when None."""
        cam = tf.CameraProjection(K=K_simple)
        data = cam.to_dict()
        assert "image_size" not in data


# ---------------------------------------------------------------------------
# CameraProjection _apply Nx4 path
# ---------------------------------------------------------------------------


class TestCameraProjectionApplyNx4:
    """Test CameraProjection._apply with Nx4 input."""

    def test_nx4_homogeneous(self, K_simple):
        """CameraProjection._apply handles Nx4 by dehomogenizing."""
        cam = tf.CameraProjection(K=K_simple)
        points = np.array([[0.0, 0.0, 10.0, 1.0]])
        result = cam._apply(points)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0, 0], 320.0, atol=1e-6)


# ---------------------------------------------------------------------------
# transform_points edge cases
# ---------------------------------------------------------------------------


class TestTransformPointsEdgeCases:
    """Test transform_points for graph-path and Nx4 cases."""

    def test_graph_missing_frames_error(self, K_simple):
        """transform_points with graph requires both frames."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        with pytest.raises(ValueError, match="source_frame.*target_frame"):
            tf.transform_points(np.zeros((1, 3)), graph)

    def test_nx4_rigid_transform(self):
        """transform_points with Nx4 input and rigid Transform."""
        t = tf.Translation(x=5)
        points = np.array([[1.0, 0.0, 0.0, 1.0]])
        result = tf.transform_points(points, t)
        assert result.shape == (1, 4)
        np.testing.assert_allclose(result[0, 0], 6.0)

    def test_unsupported_type_error(self):
        """transform_points with CompositeProjection works (inherits Projection)."""
        K = tf.CameraProjection(K=np.diag([100, 100, 1]).astype(float))
        T = tf.Transform(translation=[1, 0, 0])
        cp = K * T
        assert isinstance(cp, tf.CompositeProjection)
        # CompositeProjection inherits from Projection, so it's valid
        result = tf.transform_points(np.array([[0.0, 0.0, 10.0]]), cp)
        assert result.shape == (1, 3)

    def test_invalid_nx5_points(self):
        """transform_points with Nx5 raises ValueError."""
        t = tf.Translation(x=1)
        with pytest.raises(ValueError, match="Nx2.*Nx3.*Nx4"):
            tf.transform_points(np.zeros((1, 5)), t)

    def test_projection_nx4(self, K_simple):
        """transform_points with Projection and Nx4 input."""
        cam = tf.CameraProjection(K=K_simple)
        points = np.array([[0.0, 0.0, 10.0, 1.0]])
        result = tf.transform_points(points, cam)
        assert result.shape == (1, 3)

    def test_projection_invalid_shape(self, K_simple):
        """transform_points with Projection and invalid points raises ValueError."""
        cam = tf.CameraProjection(K=K_simple)
        with pytest.raises(ValueError, match="Nx3 or Nx4"):
            tf.transform_points(np.zeros((1, 2)), cam)


# ---------------------------------------------------------------------------
# project_points fallback path
# ---------------------------------------------------------------------------


class TestProjectPointsFallback:
    """Test project_points fallback _apply path (L2098-2103)."""

    def test_graph_missing_frames_error(self, K_simple):
        """project_points with graph requires both frames."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.CameraProjection(K=K_simple))
        with pytest.raises(ValueError, match="source_frame.*target_frame"):
            tf.project_points(np.zeros((1, 3)), graph)


# ---------------------------------------------------------------------------
# get_basis_vectors
# ---------------------------------------------------------------------------


class TestGetBasisVectors:
    """Test the get_basis_vectors free function."""

    def test_identity(self):
        """Identity transform gives unit basis at origin."""
        origin, x, y, z = tf.get_basis_vectors(tf.Identity())
        np.testing.assert_allclose(origin, [0, 0, 0])
        np.testing.assert_allclose(x, [1, 0, 0])
        np.testing.assert_allclose(y, [0, 1, 0])
        np.testing.assert_allclose(z, [0, 0, 1])

    def test_translated(self):
        """Translated transform shifts origin but not direction."""
        origin, x, y, z = tf.get_basis_vectors(tf.Translation(x=10))
        np.testing.assert_allclose(origin, [10, 0, 0])
        np.testing.assert_allclose(x, [11, 0, 0])

    def test_custom_length(self):
        """Custom length scales basis vectors."""
        origin, x, y, z = tf.get_basis_vectors(tf.Identity(), length=2.0)
        np.testing.assert_allclose(x, [2, 0, 0])


# ---------------------------------------------------------------------------
# is_projection_frame
# ---------------------------------------------------------------------------


class TestIsProjectionFrame:
    """Test TransformGraph.is_projection_frame."""

    def test_image_frame_is_projection(self, K_simple):
        """Frame connected via CameraProjection is a projection frame."""
        graph = tf.TransformGraph()
        graph.add_transform("camera", "image", tf.CameraProjection(K=K_simple))
        assert graph.is_projection_frame("image") is True
        assert graph.is_projection_frame("camera") is False

    def test_rigid_frame_is_not_projection(self):
        """Frame connected via Translation is not a projection frame."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        assert graph.is_projection_frame("B") is False

    def test_unknown_frame(self, K_simple):
        """Non-existent frame returns False."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.CameraProjection(K=K_simple))
        assert graph.is_projection_frame("Z") is False

    def test_isolated_frame_returns_false(self, K_simple):
        """Frame with no neighbors returns False."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.CameraProjection(K=K_simple))
        graph.remove_transform("A", "B")
        # After removal, A should have no neighbors
        # (B gets removed too since it's isolated)
        assert graph.is_projection_frame("A") is False


# ---------------------------------------------------------------------------
# Identity neutrality
# ---------------------------------------------------------------------------


class TestIdentityNeutrality:
    """Test Identity as the algebraic neutral element."""

    def test_identity_left_mul(self):
        """Identity * T = T."""
        t = tf.Translation(x=5)
        result = tf.Identity() * t
        np.testing.assert_allclose(result.as_matrix(), t.as_matrix())

    def test_identity_right_mul_inverse_projection(self, K_simple):
        """InverseProjection * Identity = InverseProjection (Identity path)."""
        inv = tf.CameraProjection(K=K_simple).inverse()
        # Currently __mul__ checks isinstance(other, Identity) → return self
        result = inv * tf.Identity()
        assert result is inv


# ---------------------------------------------------------------------------
# OrthographicProjection extra coverage
# ---------------------------------------------------------------------------


class TestOrthographicEdgePaths:
    """Cover ortho _apply Nx4 error, project_points alias, unproject side axis."""

    def test_apply_invalid_shape(self):
        """OrthographicProjection._apply with Nx5 raises ValueError."""
        ortho = tf.OrthographicProjection("top", (-10, 10), (-10, 10), 1.0)
        with pytest.raises(ValueError, match="Nx3 or Nx4"):
            ortho._apply(np.zeros((1, 5)))

    def test_ortho_project_points_alias(self):
        """OrthographicProjection.project_points is an alias for _apply."""
        ortho = tf.OrthographicProjection("top", (-10, 10), (-10, 10), 1.0)
        pts = np.array([[5.0, 3.0, 0.0]])
        expected = ortho._apply(pts)
        result = ortho.project_points(pts)
        np.testing.assert_allclose(result, expected)

    def test_side_unproject_round_trip(self):
        """Side-view unproject round-trip covers non-flip_u path in unproject."""
        # side: flip_u=False, flip_v=True
        ortho = tf.OrthographicProjection("side", (-10, 10), (-5, 5), 0.5)
        pts = np.array([[3.0, 99.0, -2.0]])  # Y is collapsed for side
        px = tf.project_points(pts, ortho)
        pts_back = tf.transform_points(px, ortho.inverse())
        # X and Z should round-trip, Y is zero
        np.testing.assert_allclose(pts_back[0, 0], pts[0, 0], atol=1e-10)  # x
        np.testing.assert_allclose(pts_back[0, 2], pts[0, 2], atol=1e-10)  # z
        assert pts_back[0, 1] == 0.0  # y collapsed

    def test_inverse_ortho_property(self):
        """InverseOrthographicProjection.orthographic_projection returns original."""
        ortho = tf.OrthographicProjection("front", (-10, 10), (-5, 5), 0.5)
        inv = ortho.inverse()
        assert inv.orthographic_projection is ortho

    def test_inverse_ortho_invalid_input_shape(self):
        """InverseOrthographicProjection._apply with wrong shape raises ValueError."""
        ortho = tf.OrthographicProjection("top", (-10, 10), (-10, 10), 1.0)
        inv = ortho.inverse()
        with pytest.raises(ValueError, match="Nx2"):
            inv._apply(np.zeros((1, 3)))

    def test_ortho_nx4_input(self):
        """OrthographicProjection._apply handles Nx4 homogeneous input."""
        ortho = tf.OrthographicProjection("top", (-10, 10), (-10, 10), 1.0)
        pts_3d = np.array([[5.0, 3.0, 0.0]])
        pts_4d = np.array([[5.0, 3.0, 0.0, 1.0]])
        np.testing.assert_allclose(ortho._apply(pts_3d), ortho._apply(pts_4d), atol=1e-10)


# ---------------------------------------------------------------------------
# CompositeProjection / InverseCompositeProjection from_dict validation
# ---------------------------------------------------------------------------


class TestCompositeFromDictValidation:
    """Cover from_dict validation error branches (L1691-1794)."""

    def test_composite_invalid_projection_type(self):
        """CompositeProjection.from_dict rejects non-Projection projection."""
        bad_data = {
            "type": "CompositeProjection",
            "projection": tf.Transform(translation=[1, 0, 0]).to_dict(),
            "transform": tf.Transform(translation=[0, 1, 0]).to_dict(),
        }
        with pytest.raises(ValueError, match="must be a Projection"):
            tf.CompositeProjection.from_dict(bad_data)

    def test_composite_invalid_transform_type(self, K_simple):
        """CompositeProjection.from_dict rejects non-Transform transform."""
        proj = tf.Projection(matrix=K_simple @ np.eye(3, 4))
        bad_data = {
            "type": "CompositeProjection",
            "projection": proj.to_dict(),
            "transform": proj.to_dict(),  # Projection, not Transform
        }
        with pytest.raises(ValueError, match="must be a Transform"):
            tf.CompositeProjection.from_dict(bad_data)

    def test_inverse_composite_invalid_transform(self, K_simple):
        """InverseCompositeProjection.from_dict rejects non-Transform transform."""
        inv = tf.InverseProjection(original_matrix=K_simple @ np.eye(3, 4))
        bad_data = {
            "type": "InverseCompositeProjection",
            "transform": inv.to_dict(),  # InverseProjection, not Transform
            "projection": inv.to_dict(),
        }
        with pytest.raises(ValueError, match="must be a Transform"):
            tf.InverseCompositeProjection.from_dict(bad_data)

    def test_inverse_composite_invalid_projection(self, K_simple):
        """InverseCompositeProjection.from_dict rejects non-InverseProjection projection."""
        t = tf.Transform(translation=[1, 0, 0])
        bad_data = {
            "type": "InverseCompositeProjection",
            "transform": t.to_dict(),
            "projection": t.to_dict(),  # Transform, not InverseProjection
        }
        with pytest.raises(ValueError, match="must be an InverseProjection"):
            tf.InverseCompositeProjection.from_dict(bad_data)


# ---------------------------------------------------------------------------
# project_points with rigid transform error
# ---------------------------------------------------------------------------


class TestProjectPointsRigidError:
    """project_points with rigid Transform raises TypeError."""

    def test_rigid_transform_rejected(self):
        """project_points rejects rigid transforms (needs projection)."""
        t = tf.Translation(x=1)
        with pytest.raises(TypeError, match="Cannot project_points"):
            tf.project_points(np.zeros((1, 3)), t)


# ---------------------------------------------------------------------------
# _get_camera_intrinsics_and_pose
# ---------------------------------------------------------------------------


class TestGetCameraIntrinsicsAndPose:
    """Test TransformGraph._get_camera_intrinsics_and_pose."""

    def test_basic_lookup(self, K_simple):
        """Finds K and camera_frame for a valid image frame."""
        graph = tf.TransformGraph()
        graph.add_transform("camera", "image", tf.CameraProjection(K=K_simple))
        K, cam_frame = graph._get_camera_intrinsics_and_pose("image")
        np.testing.assert_allclose(K, K_simple)
        assert cam_frame == "camera"

    def test_not_found_error(self):
        """Raises ValueError for unknown frame."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        with pytest.raises(ValueError, match="not found"):
            graph._get_camera_intrinsics_and_pose("Z")

    def test_not_projection_error(self):
        """Raises ValueError for frame not connected to camera."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        with pytest.raises(ValueError, match="not a valid projection frame"):
            graph._get_camera_intrinsics_and_pose("B")


# ---------------------------------------------------------------------------
# BaseTransform.__repr__ (L405)
# ---------------------------------------------------------------------------


class TestBaseTransformRepr:
    """Test the BaseTransform default __repr__ for subclasses that don't override."""

    def test_matrix_transform_repr(self):
        """MatrixTransform repr contains class name."""
        mt = tf.MatrixTransform(np.eye(4))
        assert "MatrixTransform" in repr(mt)
