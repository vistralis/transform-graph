# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from uuid import UUID

import networkx as nx
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as ScipyRotation

# -----------------------------------------------------------------------------
# Transform Registry for Serialization
# -----------------------------------------------------------------------------
# Global registry mapping type names to classes
_TRANSFORM_REGISTRY: dict[str, type[BaseTransform]] = {}


def register_transform(cls: type[BaseTransform]) -> type[BaseTransform]:
    """
    Decorator to register a transform class for serialization.

    Usage:
        @register_transform
        class MyTransform(BaseTransform):
            ...
    """
    _TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


def serialize_transform(transform: BaseTransform) -> dict[str, Any]:
    """
    Serialize any transform to a JSON-compatible dictionary.

    Args:
        transform: Any BaseTransform subclass instance.

    Returns:
        Dict containing the serialized transform with a "type" key.
    """
    return transform.to_dict()


def from_roll_pitch_yaw(
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> quaternion.quaternion:
    """
    Create a quaternion from roll-pitch-yaw angles.

    Uses the aerospace/robotics intrinsic **ZYX** (Tait-Bryan) convention:
    yaw (Z) → pitch (Y) → roll (X).

    For other conventions (ZYZ, XYZ, etc.), use scipy directly::

        from scipy.spatial.transform import Rotation as R
        q_scipy = R.from_euler('ZYZ', [alpha, beta, gamma])

    Args:
        roll: Rotation about X-axis in radians.
        pitch: Rotation about Y-axis in radians.
        yaw: Rotation about Z-axis in radians.

    Returns:
        quaternion.quaternion: The resulting quaternion.

    Warning:
        This function uses ``scipy.spatial.transform.Rotation`` with true
        ZYX intrinsic ordering.  It is **not** compatible with
        ``quaternion.from_euler_angles(alpha, beta, gamma)``, which uses
        ZYZ convention.
    """
    scipy_rot = ScipyRotation.from_euler("ZYX", [yaw, pitch, roll])
    # scipy uses [x, y, z, w], numpy-quaternion uses [w, x, y, z]
    x, y, z, w = scipy_rot.as_quat()
    return quaternion.quaternion(w, x, y, z)


def as_roll_pitch_yaw(
    q: quaternion.quaternion,
) -> tuple[float, float, float]:
    """
    Extract roll, pitch, yaw from a quaternion.

    Uses the aerospace/robotics intrinsic **ZYX** (Tait-Bryan) convention.

    For other conventions (ZYZ, XYZ, etc.), use scipy directly::

        from scipy.spatial.transform import Rotation as R
        angles = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('ZYZ')

    Args:
        q: The input quaternion.

    Returns:
        Tuple[float, float, float]: ``(roll, pitch, yaw)`` in radians.
    """
    scipy_rot = ScipyRotation.from_quat([q.x, q.y, q.z, q.w])
    yaw, pitch, roll = scipy_rot.as_euler("ZYX")
    return (roll, pitch, yaw)


# ---------------------------------------------------------------------------
# Numerical constants — projection
# ---------------------------------------------------------------------------
_DEPTH_EPS = 1e-10
"""Guard for z-division in perspective projection (prevents divide-by-zero)."""

_RADIAL_EPS = 1e-10
"""Guard for r-division in equidistant models (on-axis singularity)."""

_NORM_EPS = 1e-10
"""Guard for point norm in spherical models (zero-length point)."""

_DENOM_EPS = 1e-10
"""Guard for rational model denominators (prevents division instability)."""


class ProjectionModel(Enum):
    """
    Supported camera projection models.

    Each member represents a complete 3D → 2D projection function, covering
    both the ideal projection geometry and its associated distortion model.

    Models:

    - Pinhole: Ideal perspective projection, no distortion.
      Parameters: fx, fy, cx, cy.
    - BrownConrady: Pinhole + radial/tangential distortion.
      D = (k1, k2, p1, p2, k3). OpenCV default, ROS ``plumb_bob``.
    - KannalaBrandt: Fisheye / equidistant.
      D = (k1, k2, k3, k4). ``cv2.fisheye``, ROS ``kannala_brandt``.
    - Division: Simple wide-angle with single division coefficient.
      D = (k1,).
    - Rational: Full rational polynomial.
      D = (k1, k2, p1, p2, k3, k4, k5, k6). ROS ``rational_polynomial``.
    - Fisheye62: Project Aria fisheye model.
      D = (k0, k1, k2, k3, p0, p1).
    - MeiUnified: Unified omnidirectional camera model (Mei 2007).
      D = (xi, k1, k2). Used by KITTI-360 fisheye cameras.
    """

    Pinhole = "Pinhole"
    BrownConrady = "BrownConrady"
    KannalaBrandt = "KannalaBrandt"
    Division = "Division"
    Rational = "Rational"
    Fisheye62 = "Fisheye62"
    MeiUnified = "MeiUnified"

    @classmethod
    def from_string(cls, model_str: str) -> ProjectionModel:
        """Convert a string to a ProjectionModel enum value.

        Accepts ROS ``distortion_model`` names and legacy tgraph names.
        """
        _aliases = {
            # ROS camera_info distortion_model names
            "plumb_bob": cls.BrownConrady,
            "rational_polynomial": cls.Rational,
            "kannala_brandt": cls.KannalaBrandt,
            "fisheye62": cls.Fisheye62,
            # Legacy tgraph names
            "Fisheye": cls.KannalaBrandt,
            "Omnidirectional": cls.Division,
            "Pinhole+Polynomial": cls.BrownConrady,
        }
        if model_str in _aliases:
            return _aliases[model_str]
        # Exact value match
        for model in cls:
            if model.value == model_str:
                return model
        # Case-insensitive name match
        lower = model_str.lower().replace("_", "").replace("-", "").replace("+", "")
        for model in cls:
            if model.name.lower() == lower:
                return model
        raise ValueError(
            f"Unknown projection model: {model_str}. "
            f"Valid: {[m.value for m in cls]}"
        )


def deserialize_transform(data: dict[str, Any]) -> BaseTransform:
    """
    Deserialize a transform from a dictionary.

    Automatically determines the correct class from the "type" field
    and calls its from_dict() method.

    Args:
        data: Dictionary previously created by serialize_transform() or to_dict().

    Returns:
        BaseTransform: The deserialized transform instance.

    Raises:
        ValueError: If the transform type is not registered.
    """
    transform_type = data.get("type")
    if not transform_type:
        raise ValueError("Missing 'type' field in transform data")

    if transform_type not in _TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform type: '{transform_type}'. "
            f"Registered types: {list(_TRANSFORM_REGISTRY.keys())}"
        )

    cls = _TRANSFORM_REGISTRY[transform_type]
    return cls.from_dict(data)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def decompose_projection_to_objects(
    projection_matrix: np.ndarray, dtype: np.dtype = np.float64
) -> tuple[CameraProjection, Transform]:
    """
    Decompose a 3x4 projection matrix P into Intrinsic (CameraProjection)
    and Extrinsic (Transform) objects.

    P = K @ [R | t]

    Returns:
        (CameraProjection, Transform)
        - CameraProjection: Holds K (Intrinsic only)
        - Transform: Extrinsic Transform T_world_to_cam (OpenCV convention: R, t).
          Note: This Transform represents the conversion from World to Camera frame.
          If you want Camera Pose (Camera to World), take the inverse of this transform.
    """
    from scipy.linalg import rq

    P = np.asarray(projection_matrix, dtype=dtype)
    if P.shape == (4, 4):
        P = P[:3, :]

    if P.shape != (3, 4):
        raise ValueError(f"Projection matrix must be 3x4 or 4x4, got {P.shape}")

    # Extract M = P[:3,:3] = K @ R
    M = P[:3, :3]

    # RQ decomposition: M = R @ Q (scipy uses K, R notation where K is upper tri)
    intrinsic_matrix, rotation_matrix = rq(M)

    # Ensure K has positive diagonal
    for i in range(3):
        if intrinsic_matrix[i, i] < 0:
            intrinsic_matrix[i, :] *= -1
            rotation_matrix[:, i] *= -1

    # Normalize K so that K[2,2] = 1
    scale = intrinsic_matrix[2, 2]
    if abs(scale) > 1e-10:
        intrinsic_matrix = intrinsic_matrix / scale

    # Ensure R is proper rotation
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix = -rotation_matrix
        intrinsic_matrix = -intrinsic_matrix

    # Compute translation: P[:3,3] = K @ t -> t = inv(K) @ P[:3,3]
    t_vec = np.linalg.solve(intrinsic_matrix, P[:3, 3])
    t_vec = t_vec.reshape(3, 1)

    # Convert Rotation Matrix to Quaternion
    rot_quat = quaternion.from_rotation_matrix(rotation_matrix)

    # Create Objects
    cam_proj = CameraProjection(intrinsic_matrix=intrinsic_matrix, dtype=dtype)
    extrinsic_tf = Transform(translation=t_vec, rotation=rot_quat, dtype=dtype)

    return cam_proj, extrinsic_tf


def ensure_translation(
    translation: np.ndarray | list | tuple | None, dtype: np.dtype
) -> np.ndarray:
    """
    Ensures translation is a 3x1 numpy array of the specified dtype.
    Optimized to avoid copies if input already matches requirements.
    """
    if translation is None:
        return np.zeros((3, 1), dtype=dtype)

    if (
        isinstance(translation, np.ndarray)
        and translation.shape == (3, 1)
        and translation.dtype == dtype
    ):
        return translation

    array = np.array(translation, dtype=dtype)
    if array.size != 3:
        raise ValueError(f"Translation must have 3 elements, got {array.size}")
    return array.reshape(3, 1)


def ensure_rotation(
    rotation: np.ndarray | list | tuple | None, dtype: np.dtype
) -> quaternion.quaternion:
    """
    Ensures rotation is a quaternion of the specified dtype.
    Optimized to avoid copies if input already matches requirements.
    """
    if rotation is None:
        return quaternion.one

    if isinstance(rotation, quaternion.quaternion):
        return rotation

    array = np.array(rotation, dtype=dtype).flatten()
    if array.size == 4:
        return quaternion.quaternion(*array)
    else:
        raise ValueError(f"Rotation must be a quaternion or 4 elements [w,x,y,z], got {array.size}")


def skew(vector: np.ndarray | list | tuple) -> np.ndarray:
    """
    Returns the 3x3 skew-symmetric matrix of a 3-element vector.

    [v]x = [[  0, -v3,  v2],
            [ v3,   0, -v1],
            [-v2,  v1,   0]]

    Args:
        vector: A 3-element sequence (shape (3,) or (3,1)).

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix.
    """
    v = np.asarray(vector, dtype=np.float64).flatten()
    if v.size != 3:
        raise ValueError(f"Vector must have 3 elements, got {v.size}")

    vx, vy, vz = v
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)


class BaseTransform(ABC):
    """
    Abstract interface for all spatial transformations.
    """

    def __init__(self, dtype: np.dtype = np.float64):
        self.dtype = dtype

    @abstractmethod
    def as_matrix() -> np.ndarray:
        """
        Returns the 4x4 homogeneous representation of the transform.

        Returns:
            np.ndarray: 4x4 matrix of the transform's dtype.
        """
        pass

    @abstractmethod
    def inverse() -> BaseTransform:
        """
        Returns the mathematical inverse of the transformation.

        Returns:
            BaseTransform: The inverse transformation.
        """
        pass

    def _apply(self, vector: np.ndarray | list | tuple) -> np.ndarray:
        """
        Apply the transform to 3D vectors (Nx3 or Nx4).

        Args:
            vector: Nx3 or Nx4 array of vectors.
                    - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform,
                      checking subclass logic).
                      Standard BaseTransform behavior:
                      Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
                    - If Nx4: Treated as homogeneous vectors.
                      Multiply -> Return Nx4.

        Returns:
            np.ndarray: Transformed vectors (Nx3 or Nx4).
        """
        vector = np.atleast_2d(vector)

        if vector.shape[1] == 3:
            # Homogenize (w=1)
            hom_vector = np.hstack([vector, np.ones((vector.shape[0], 1), dtype=self.dtype)])
            # Apply
            transformed_hom = (self.as_matrix() @ hom_vector.T).T
            # Dehomogenize (return 3D)
            return transformed_hom[:, :3]

        elif vector.shape[1] == 4:
            # Generic 4x4
            return (self.as_matrix() @ vector.T).T
        else:
            raise ValueError(f"Input vector must be Nx3 or Nx4, got {vector.shape}")

    @abstractmethod
    def __mul__(self, other: BaseTransform) -> BaseTransform:
        """
        Composes this transform with another.
        Composition follows standard matrix multiplication order: (T1 * T2) * p = T1 * (T2 * p).

        Args:
            other: The transform to apply second.

        Returns:
            BaseTransform: The composed transformation.
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the transform to a JSON-compatible dictionary.

        The dictionary MUST include a "type" key with the class name
        to enable proper deserialization.

        Returns:
            Dict[str, Any]: JSON-compatible dictionary representation.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseTransform:
        """
        Deserialize a transform from a dictionary.

        Args:
            data: Dictionary previously created by to_dict().

        Returns:
            BaseTransform: The deserialized transform instance.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


@register_transform
class Transform(BaseTransform):
    """
    Standard SE(3) rigid body transformation.
    Consists of a translation (3x1) and a rotation (quaternion).
    """

    def __init__(
        self,
        translation: np.ndarray | list | tuple | None = None,
        rotation: quaternion.quaternion | np.ndarray | list | tuple | Transform | None = None,
        dtype: np.dtype = np.float64,
    ):
        super().__init__(dtype=dtype)

        self.translation = ensure_translation(translation, self.dtype)

        # Handle rotation: can be quaternion, array, or Transform/Rotation object
        if isinstance(rotation, BaseTransform):
            # It's a Transform or Rotation object - extract the quaternion
            rotation = rotation.rotation

        self.rotation = ensure_rotation(rotation, self.dtype)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, dtype: np.dtype | None = None) -> Transform:
        """
        Creates a Transform from a 4x4 matrix.
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got {matrix.shape}")

        target_dtype = dtype if dtype is not None else matrix.dtype
        translation = matrix[:3, 3]
        rot_mat = matrix[:3, :3]
        rot_quat = quaternion.from_rotation_matrix(rot_mat)
        return cls(translation=translation, rotation=rot_quat, dtype=target_dtype)

    def as_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transformation matrix [R|t; 0 1]."""
        matrix = np.eye(4, dtype=self.dtype)
        matrix[:3, :3] = quaternion.as_rotation_matrix(self.rotation).astype(self.dtype)
        matrix[:3, 3] = self.translation.ravel()
        return matrix

    def inverse(self) -> Transform:
        """Return the inverse SE(3) transform: T^-1 = [-R^T t; R^T]."""
        inv_rotation = self.rotation.conjugate()
        # Rotate expects (..., 3) vectors. flatten() ensures we get (3,) if translation is (3, 1)
        inv_translation = -quaternion.rotate_vectors(inv_rotation, self.translation.flatten())
        return Transform(translation=inv_translation, rotation=inv_rotation, dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        if isinstance(other, Transform):
            # T1 * T2 = [R1*R2, R1*t2 + t1]
            new_rotation = self.rotation * other.rotation
            new_translation = (
                quaternion.rotate_vectors(self.rotation, other.translation.flatten())
                + self.translation.flatten()
            )
            return Transform(translation=new_translation, rotation=new_rotation, dtype=self.dtype)

        if isinstance(other, (Projection, CameraProjection)):
            # Transform * Projection -> Invalid
            # 3D->3D * 3D->2D = dimensional mismatch if interpreted strictly as flow?
            # Guidelines say: Transform * CameraProjection = Forbidden.
            raise TypeError(
                "Composition 'Transform * CameraProjection' is invalid. "
                "Transforms (Spatial) cannot pre-multiply Projections (Sensor). "
                "Did you mean 'CameraProjection * Transform'?"
            )

        if isinstance(other, InverseProjection):
            # Transform * InverseProjection -> InverseCompositeProjection
            # T * P_inv
            # Logic: P_inv unprojects to frame A. T transforms A to B. Result unprojects to B.

            # If other is already InverseCompositeProjection: T * (T_old * K_inv)
            # The __rmul__ of InverseCompositeProjection should handle
            # this if we return NotImplemented?
            # But BaseTransform doesn't implement __rmul__ universally.
            # Let's handle it explicitly or rely on commutation if possible.
            # Transform doesn't know about InverseCompositeProjection class structure necessarily?
            # But we just added it to this module.

            if isinstance(other, InverseCompositeProjection):
                # T * ICA(T_old, K_inv) = ICA(T * T_old, K_inv)
                new_transform = self * other.transform
                if isinstance(new_transform, Transform):
                    return InverseCompositeProjection(
                        new_transform, other.projection, dtype=self.dtype
                    )

            # Generic InverseProjection (e.g. InverseCameraProjection or raw InverseProjection)
            # Treat 'self' as the extrinsics T.
            # Result = T * P_inv.
            # We create InverseCompositeProjection(T, P_inv).
            return InverseCompositeProjection(self, other, dtype=self.dtype)

        # Fallback to matrix multiplication
        return MatrixTransform(self.as_matrix() @ other.as_matrix(), dtype=self.dtype)

    def __repr__(self) -> str:
        return f"Transform(translation={self.translation.flatten()!r}, rotation={self.rotation!r})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize transform to a JSON-compatible dictionary."""
        t = self.translation.flatten()
        q = self.rotation
        return {
            "type": "Transform",
            "translation": [float(t[0]), float(t[1]), float(t[2])],
            "rotation": [float(q.w), float(q.x), float(q.y), float(q.z)],
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transform:
        """Deserialize transform from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(
            translation=data["translation"],
            rotation=data["rotation"],
            dtype=dtype,
        )


class Translation(Transform):
    """A Transform with only translation (Identity rotation)."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        translation: np.ndarray | list | tuple | None = None,
    ):
        if translation is not None:
            super().__init__(translation=translation)
        else:
            super().__init__(translation=[x, y, z])


class Rotation(Transform):
    """
    A Transform with only rotation (Zero translation).

    Supports multiple construction patterns:
    - Quaternion components: Rotation(w=1, x=0, y=0, z=0)
    - Quaternion object: Rotation(rotation=q)
    - Euler angles: Rotation.from_roll_pitch_yaw(roll=0, pitch=0, yaw=0)
    ...
    """

    def __init__(
        self,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        rotation: quaternion.quaternion | np.ndarray | list | tuple | None = None,
    ):
        if rotation is not None:
            super().__init__(rotation=rotation)
        else:
            super().__init__(rotation=[w, x, y, z])

    @classmethod
    def from_roll_pitch_yaw(
        cls,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> Rotation:
        """
        Create a Rotation from roll-pitch-yaw angles.

        Uses the aerospace/robotics intrinsic **ZYX** (Tait-Bryan) convention:
        yaw (Z) → pitch (Y) → roll (X).

        Args:
            roll: Rotation about X-axis in radians.
            pitch: Rotation about Y-axis in radians.
            yaw: Rotation about Z-axis in radians.

        Returns:
            Rotation: A rotation-only transform.

        Example:
            >>> attitude = tf.Rotation.from_roll_pitch_yaw(pitch=np.radians(10))
            >>> heading = tf.Rotation.from_roll_pitch_yaw(yaw=np.pi/4)
        """
        q = from_roll_pitch_yaw(roll=roll, pitch=pitch, yaw=yaw)
        return cls(rotation=q)

    def as_roll_pitch_yaw(self) -> tuple[float, float, float]:
        """
        Extract roll, pitch, yaw from the rotation.

        Uses the aerospace/robotics intrinsic **ZYX** (Tait-Bryan) convention.

        Returns:
            Tuple[float, float, float]: ``(roll, pitch, yaw)`` in radians.

        Warning:
            Euler angles have a singularity (gimbal lock) when pitch = ±90°.

        Example:
            >>> rotation = tf.Rotation.from_roll_pitch_yaw(roll=0.1, pitch=0.2, yaw=0.3)
            >>> roll, pitch, yaw = rotation.as_roll_pitch_yaw()
        """
        return as_roll_pitch_yaw(self.rotation)


class Identity(Transform):
    """The identity transform (0 translation, identity rotation)."""

    def __init__(self):
        super().__init__()

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        """Identity is the neutral element: I * X = X."""
        return other


@register_transform
class MatrixTransform(BaseTransform):
    """
    A generic transform held as a raw 4x4 matrix.
    Used when SE(3) structure is lost or not applicable.
    """

    def __init__(self, matrix: np.ndarray, dtype: np.dtype = np.float64):
        super().__init__(dtype=dtype)
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got {matrix.shape}")
        self.matrix = matrix.astype(self.dtype)

    def as_matrix(self) -> np.ndarray:
        """Return the stored 4x4 matrix."""
        return self.matrix

    def inverse(self) -> MatrixTransform:
        """Return the inverse via np.linalg.inv.

        .. warning::
            Uses raw ``np.linalg.inv`` for convenience. This method is intended
            for quick inspection and non-critical paths. For near-singular or
            ill-conditioned matrices, prefer decomposing back into structured
            types (``Transform``, ``Projection``) that have numerically stable
            inverses.
        """
        return MatrixTransform(np.linalg.inv(self.matrix), dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> MatrixTransform:
        return MatrixTransform(self.matrix @ other.as_matrix(), dtype=self.dtype)

    def to_dict(self) -> dict[str, Any]:
        """Serialize transform to a JSON-compatible dictionary."""
        return {
            "type": "MatrixTransform",
            "matrix": self.matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatrixTransform:
        """Deserialize transform from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(matrix=np.array(data["matrix"]), dtype=dtype)

    def __repr__(self) -> str:
        # Format matrix with numpy's array_repr for better readability
        matrix_str = np.array_repr(self.matrix, precision=4, suppress_small=True)
        return f"MatrixTransform(matrix={matrix_str})"


# -----------------------------------------------------------------------------
# Projection Classes (3D -> 2D)
# -----------------------------------------------------------------------------


def _ensure_4x4_projection(matrix: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Convert a 3x4 or 4x4 projection matrix to 4x4 format.

    For 3x4 matrices, adds bottom row [0, 0, 0, 1].
    """
    matrix = np.asarray(matrix, dtype=dtype)

    if matrix.shape == (4, 4):
        return matrix
    elif matrix.shape == (3, 4):
        result = np.zeros((4, 4), dtype=dtype)
        result[:3, :] = matrix
        result[3, 3] = 1.0
        return result
    else:
        raise ValueError(f"Projection matrix must be 3x4 or 4x4, got {matrix.shape}")


@register_transform
class Projection(BaseTransform):
    """
    A 3D to 2D projection transformation.

    Stores a projection matrix P that maps 3D homogeneous points to 2D.
    Internally stored as 4x4 matrix with bottom row [0, 0, 0, 1] for compatibility.

    The project_points() method projects 3D points to 2D pixel coordinates.

    Note: Projections are generally non-invertible. The inverse() method returns
    an InverseProjection which represents the conceptual inverse but requires
    additional depth information to actually unproject points.
    """

    def __init__(
        self,
        matrix: np.ndarray | list,
        dtype: np.dtype = np.float64,
    ):
        """
        Create a Projection from a 3x4 or 4x4 matrix.

        Args:
            matrix: 3x4 or 4x4 projection matrix.
            dtype: Data type for the matrix.
        """
        super().__init__(dtype=dtype)
        self.matrix = _ensure_4x4_projection(np.asarray(matrix), self.dtype)

    def as_matrix(self) -> np.ndarray:
        """Returns the 4x4 projection matrix."""
        return self.matrix

    def as_matrix_3x4(self) -> np.ndarray:
        """Returns the 3x4 projection matrix (top 3 rows)."""
        return self.matrix[:3, :]

    def inverse(self) -> InverseProjection:
        """
        Returns an InverseProjection representing P^-1.

        Note: The inverse projection requires depth information to actually
        unproject 2D points to 3D.
        """
        return InverseProjection(self.matrix, dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        """Compose projection with another transform."""
        result_matrix = self.matrix @ other.as_matrix()

        # Compose with Rigid Transform -> CompositeProjection
        if isinstance(other, Transform):
            return CompositeProjection(self, other, dtype=self.dtype)

        if isinstance(other, (Projection, CompositeProjection)):
            raise TypeError(
                f"Composition '{type(self).__name__} * "
                f"{type(other).__name__}' is invalid "
                "(dimensional mismatch). "
                "Both transformations map to 2D; you cannot compose them in this order."
            )

        # Fallback for all other types (MatrixTransform, InverseProjection)
        # These compositions (e.g. P * P_inv) stay within 2D->2D or
        # 3D->3D bounds but are not specialized.
        return MatrixTransform(result_matrix, dtype=self.dtype)

    def _apply(self, vector: np.ndarray | list | tuple) -> np.ndarray:
        """
        Project 3D vectors to 2D pixel coordinates.

        Args:
            vector: Nx3 (points) or Nx4 (homogeneous) array.

        Returns:
            np.ndarray: Nx2 pixel coordinates.
        """
        vector = np.atleast_2d(vector)

        if vector.shape[1] == 3:
            # Homogenize (w=1 implicit for points)
            hom_vec = np.hstack([vector, np.ones((vector.shape[0], 1), dtype=self.dtype)])
        elif vector.shape[1] == 4:
            hom_vec = vector
        else:
            raise ValueError(f"Input must be Nx3 or Nx4, got {vector.shape}")

        # Project: (3x4 or 4x4) @ 4x1 -> 4x1 (if 4x4) or 3x1 (if 3x4?)
        # Base class stores 4x4 with bottom [0,0,0,1].
        # Result will be [u*w, v*w, w, 1].
        projected = (self.matrix @ hom_vec.T).T

        # We need [x, y, w] part.
        # projected is Nx4.

        # Perspective division
        w = projected[:, 2:3]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)
        pixels = projected[:, :2] / w

        return pixels

    def project_points(self, points: np.ndarray | list | tuple) -> np.ndarray:
        """
        Project 3D points (Nx3 or Nx4) to 2D pixel coordinates.
        alias for _apply(points).

        Args:
             points: Nx3 or Nx4 array of points.

        Returns:
             np.ndarray: Nx2 pixel coordinates.
        """
        return self._apply(points)

    def to_dict(self) -> dict[str, Any]:
        """Serialize projection to a JSON-compatible dictionary."""
        return {
            "type": "Projection",
            "matrix": self.matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Projection:
        """Deserialize projection from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(matrix=np.array(data["matrix"]), dtype=dtype)

    def __repr__(self) -> str:
        return f"Projection(matrix_shape={self.matrix.shape})"


@register_transform
class InverseProjection(BaseTransform):
    """
    Represents the conceptual inverse of a Projection (P^-1).

    This class tracks that an inverse operation was requested, but actual
    unprojection requires depth information. Use unproject() with depth values
    to convert 2D pixels back to 3D points.

    Useful for:
    - Tracking transform logic in a graph
    - Composing with other transforms
    - Unprojecting when depth is available
    """

    def __init__(
        self,
        original_matrix: np.ndarray | list,
        dtype: np.dtype = np.float64,
    ):
        """
        Create an InverseProjection from the original projection matrix.

        Args:
            original_matrix: The original 3x4 or 4x4 projection matrix.
            dtype: Data type for the matrix.
        """
        super().__init__(dtype=dtype)
        self._original_matrix = _ensure_4x4_projection(np.asarray(original_matrix), self.dtype)

    @property
    def original_matrix(self) -> np.ndarray:
        """The original projection matrix that was inverted."""
        return self._original_matrix

    def as_matrix(self) -> np.ndarray:
        """
        Returns a pseudo-inverse matrix for composition purposes.

        Warning: This is the Moore-Penrose pseudo-inverse and may not
        produce geometrically meaningful results for all operations.
        """
        return np.linalg.pinv(self._original_matrix)

    def inverse(self) -> Projection:
        """Returns the original Projection."""
        return Projection(self._original_matrix, dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        """Compose with another transform using pseudo-inverse."""
        if isinstance(other, Identity):
            return self
        if isinstance(other, Transform):
            raise TypeError(
                f"Composition '{type(self).__name__} * Transform' "
                "is invalid (dimensional mismatch). "
                "InverseProjections (2D->3D) cannot post-multiply Transforms (3D->3D). "
                "Did you mean 'Transform * InverseProjection'?"
            )
        if isinstance(other, (InverseProjection, InverseCompositeProjection)):
            raise TypeError(
                f"Composition '{type(self).__name__} * "
                f"{type(other).__name__}' is invalid "
                "(dimensional mismatch)."
            )

        return MatrixTransform(self.as_matrix() @ other.as_matrix(), dtype=self.dtype)

    def _apply(self, vector: np.ndarray | list | tuple) -> np.ndarray:
        """
        Unproject 2D/3D vectors using pseudo-inverse.

        Args:
            vector: Nx2 (pixels), Nx3 (homogenous pixels), or Nx4.

        Returns:
            np.ndarray: Transformed vectors (Nx3 or Nx4).
        """
        vector = np.atleast_2d(vector)
        cols = vector.shape[1]

        input_vec = None
        if cols == 2:
            # Nx2 pixels -> Nx3 homogeneous [u, v, 1]
            input_vec = np.hstack([vector, np.ones((vector.shape[0], 1), dtype=self.dtype)])
        elif cols == 3:
            input_vec = vector
        elif cols == 4:
            input_vec = vector
        else:
            raise ValueError(f"Input must be Nx2, Nx3 or Nx4, got {vector.shape}")

        # If input is 3D, and P_inv is 4x4, we need 4D input?
        # P_inv is 4x4 (from BaseTransform.as_matrix which pinv's 4x4 P).
        # We need to pad to 4D if it's 3D.
        if input_vec.shape[1] == 3:
            # Pad with 0? or 1?
            # Let's assume w=1 for "point-like" unprojection (ray).
            input_vec = np.hstack([input_vec, np.ones((input_vec.shape[0], 1), dtype=self.dtype)])

        result = (self.as_matrix() @ input_vec.T).T

        # User wants "homogenize when needed and dehomogenize".
        if cols < 4:
            return result[:, :3]
        return result

    def unproject(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Unproject 2D pixels to 3D points using depth values.

        Args:
            pixels: Nx2 array of 2D pixel coordinates.
            depths: N array of depth values (Z coordinate in camera frame).

        Returns:
            np.ndarray: Nx3 array of 3D points.

        Note: This assumes a standard pinhole camera model where the
        projection matrix can be decomposed into K[R|t] form.
        """
        pixels = np.atleast_2d(pixels)
        depths = np.atleast_1d(depths).flatten()

        if pixels.shape[1] != 2:
            raise ValueError(f"Pixels must be Nx2, got {pixels.shape}")
        if len(depths) != len(pixels):
            raise ValueError(f"Depths length {len(depths)} must match pixels length {len(pixels)}")

        # Extract K matrix (intrinsics) from projection matrix P = K[R|t]
        # For simple unprojection, we assume P[:3,:3] contains K*R
        # and use the pseudo-inverse approach
        projection_3x3 = self._original_matrix[:3, :3]
        projection_t = self._original_matrix[:3, 3]

        # Homogeneous pixel coordinates scaled by depth
        hom_pixels = np.column_stack([pixels[:, 0] * depths, pixels[:, 1] * depths, depths])

        # Solve for 3D points: P[:3,:3] * X = hom_pixels - P[:3,3]
        # Using solve instead of inv for numerical stability and performance.
        points_3d = np.linalg.solve(projection_3x3, (hom_pixels - projection_t).T).T

        return points_3d

    def to_dict(self) -> dict[str, Any]:
        """Serialize inverse projection to a JSON-compatible dictionary."""
        return {
            "type": "InverseProjection",
            "original_matrix": self._original_matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InverseProjection:
        """Deserialize inverse projection from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(original_matrix=np.array(data["original_matrix"]), dtype=dtype)

    def __repr__(self) -> str:
        return f"InverseProjection(original_matrix_shape={self._original_matrix.shape})"


@register_transform
class CameraProjection(Projection):
    """
    A camera projection with strict Intrinsic-only parameters.

    Adheres to the architectural guideline that CameraProjection represents
    the internal geometry of the optical sensor (K, D) ONLY.

    Spatial pose (Extrinsics) must be managed via separate Transform objects.

    Structure:
        - K: 3x3 intrinsic matrix (focal length, principal point)
        - D: Distortion coefficients (OpenCV convention)

    Can be constructed from:
        - Explicit K parameters (with optional distortion)
        - Flexible aliases: K/intrinsic_matrix, D/dist_coeffs
    """

    def __init__(
        self,
        intrinsic_matrix: np.ndarray | list | None = None,
        dist_coeffs: list | np.ndarray | None = None,
        projection_model: ProjectionModel | str | None = None,
        image_size: tuple[int, int] | None = None,
        dtype: np.dtype = np.float64,
        # Flexible aliases (OpenCV-style)
        K: np.ndarray | list | None = None,
        D: list | np.ndarray | None = None,
    ):
        """
        Create a CameraProjection (Intrinsic-only).

        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            dist_coeffs: Distortion coefficients (OpenCV ordering: k1,k2,p1,p2,k3,...).
            projection_model: Camera model (PINHOLE, BROWN_CONRADY, KANNALA_BRANDT, etc.).
            image_size: Image dimensions (width, height) in pixels.
            dtype: Data type for matrices.
            K: Alias for intrinsic_matrix.
            D: Alias for dist_coeffs.
        """
        self._dtype = dtype

        # Handle aliases
        if K is not None:
            intrinsic_matrix = K
        if D is not None:
            dist_coeffs = D

        # Store image size
        self._image_size = image_size

        # Handle distortion coefficients
        if dist_coeffs is None:
            self._dist_coeffs = np.array([], dtype=dtype)
        else:
            self._dist_coeffs = np.asarray(dist_coeffs, dtype=dtype).flatten()

        # Handle projection model
        if projection_model is None:
            if len(self._dist_coeffs) > 0:
                self._projection_model = ProjectionModel.BrownConrady
            else:
                self._projection_model = ProjectionModel.Pinhole
        elif isinstance(projection_model, str):
            self._projection_model = ProjectionModel.from_string(projection_model)
        else:
            self._projection_model = projection_model

        if intrinsic_matrix is None:
            raise ValueError("Must provide 'intrinsic_matrix' or alias 'K'")

        self._intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=dtype)
        if self._intrinsic_matrix.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {self._intrinsic_matrix.shape}")

        # The internal matrix is just K with a [0,0,0,1] row to make it 4x4
        # (Scale/Intrinsic transform)
        # We assume points are in Camera Frame already.
        matrix_4x4 = np.eye(4, dtype=dtype)
        matrix_4x4[:3, :3] = self._intrinsic_matrix

        super().__init__(matrix=matrix_4x4, dtype=dtype)

    @classmethod
    def from_intrinsics_and_transform(cls, *args, **kwargs):
        """Disabled — CameraProjection is intrinsic-only. Use separate Transform objects."""
        raise NotImplementedError(
            "CameraProjection is now Intrinsic-only. Use separate Transform objects for Extrinsics."
        )

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """The 3x3 camera intrinsic matrix K."""
        return self._intrinsic_matrix

    @property
    def fx(self) -> float:
        """Focal length x."""
        return float(self._intrinsic_matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length y."""
        return float(self._intrinsic_matrix[1, 1])

    @property
    def cx(self) -> float:
        """Principal point x."""
        return float(self._intrinsic_matrix[0, 2])

    @property
    def cy(self) -> float:
        """Principal point y."""
        return float(self._intrinsic_matrix[1, 2])

    @property
    def focal_length(self) -> tuple[float, float]:
        """Focal lengths (fx, fy) from the intrinsic matrix."""
        return (self.fx, self.fy)

    @property
    def principal_point(self) -> tuple[float, float]:
        """Principal point (cx, cy) from the intrinsic matrix."""
        return (self.cx, self.cy)

    @property
    def dist_coeffs(self) -> np.ndarray:
        """Distortion coefficients (OpenCV ordering: k1, k2, p1, p2, k3, ...)."""
        return self._dist_coeffs

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Alias for dist_coeffs."""
        return self._dist_coeffs

    @property
    def projection_model(self) -> ProjectionModel:
        """The camera projection model (PINHOLE, BROWN_CONRADY, KANNALA_BRANDT, etc.)."""
        return self._projection_model

    @property
    def image_size(self) -> tuple[int, int] | None:
        """Image dimensions (width, height) in pixels, if specified."""
        return self._image_size

    # OpenCV-style shortcuts
    @property
    def K(self) -> np.ndarray:
        """Alias for intrinsic_matrix (OpenCV-style)."""
        return self._intrinsic_matrix

    @property
    def D(self) -> np.ndarray:
        """Alias for dist_coeffs (OpenCV-style)."""
        return self._dist_coeffs

    def _apply(self, vector: np.ndarray | list | tuple) -> np.ndarray:
        """
        Project 3D points to 2D pixel coordinates using the full projection model.

        Dispatches to the correct projection function based on ``projection_model``.
        Each model implements the complete 3D → 2D pipeline including distortion.

        Args:
            vector: Nx3 (or Nx4 homogeneous) points in camera frame.

        Returns:
            np.ndarray: Nx2 pixel coordinates.
        """
        pts = np.atleast_2d(vector)
        if pts.shape[1] == 4:
            pts = pts[:, :3] / pts[:, 3:4]

        model = self._projection_model
        if model == ProjectionModel.Pinhole:
            return self._project_pinhole(pts)
        elif model == ProjectionModel.BrownConrady:
            return self._project_brown_conrady(pts)
        elif model == ProjectionModel.KannalaBrandt:
            return self._project_kannala_brandt(pts)
        elif model == ProjectionModel.Rational:
            return self._project_rational(pts)
        elif model == ProjectionModel.Division:
            return self._project_division(pts)
        elif model == ProjectionModel.MeiUnified:
            return self._project_mei_unified(pts)
        elif model == ProjectionModel.Fisheye62:
            return self._project_fisheye62(pts)
        else:
            raise NotImplementedError(f"Projection not implemented for {model}")

    # ------------------------------------------------------------------
    # Per-model projection implementations
    #
    # All methods use named numerical constants instead of magic numbers.
    # Polynomial evaluation uses Horner form: p(θ²) = ((k4·θ² + k3)·θ² + k2)·θ² + k1
    # ------------------------------------------------------------------

    def _project_pinhole(self, pts: np.ndarray) -> np.ndarray:
        """Ideal pinhole: normalize by z, apply K. No distortion."""
        z = np.where(np.abs(pts[:, 2]) < _DEPTH_EPS, _DEPTH_EPS, pts[:, 2])
        x = pts[:, 0] / z
        y = pts[:, 1] / z
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def _project_brown_conrady(self, pts: np.ndarray) -> np.ndarray:
        """Pinhole + Brown-Conrady radial/tangential distortion (OpenCV default).

        D = (k1, k2, p1, p2 [, k3])
        """
        z = np.where(np.abs(pts[:, 2]) < _DEPTH_EPS, _DEPTH_EPS, pts[:, 2])
        x = pts[:, 0] / z
        y = pts[:, 1] / z

        d = self._dist_coeffs
        if len(d) >= 4:
            r2 = x * x + y * y
            k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
            k3 = d[4] if len(d) > 4 else 0.0

            # Horner form: 1 + r²·(k1 + r²·(k2 + r²·k3))
            radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            y_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = x * radial + x_tan
            y = y * radial + y_tan

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def _project_kannala_brandt(self, pts: np.ndarray) -> np.ndarray:
        """Kannala-Brandt equidistant fisheye model (OpenCV cv2.fisheye).

        D = (k1, k2, k3, k4)

        Projects via θ_d = θ·(1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
        where θ = atan2(r, z), and the distorted point is scaled by θ_d/r.
        """
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(r, z)

        d = self._dist_coeffs
        k1 = d[0] if len(d) > 0 else 0.0
        k2 = d[1] if len(d) > 1 else 0.0
        k3 = d[2] if len(d) > 2 else 0.0
        k4 = d[3] if len(d) > 3 else 0.0

        theta2 = theta * theta
        # Horner form: θ·(1 + θ²·(k1 + θ²·(k2 + θ²·(k3 + θ²·k4))))
        theta_d = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))

        # Scale factor: θ_d / r (safe division for on-axis points where r → 0)
        safe_r = np.where(r < _RADIAL_EPS, 1.0, r)
        scale = np.where(r < _RADIAL_EPS, 1.0, theta_d / safe_r)
        x_d = x * scale
        y_d = y * scale

        u = self.fx * x_d + self.cx
        v = self.fy * y_d + self.cy
        return np.column_stack([u, v])

    def _project_rational(self, pts: np.ndarray) -> np.ndarray:
        """Rational polynomial model (OpenCV CALIB_RATIONAL_MODEL).

        D = (k1, k2, p1, p2, k3, k4, k5, k6)

        Radial: (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶)
        Plus tangential distortion (p1, p2).
        """
        z = np.where(np.abs(pts[:, 2]) < _DEPTH_EPS, _DEPTH_EPS, pts[:, 2])
        x = pts[:, 0] / z
        y = pts[:, 1] / z

        d = self._dist_coeffs
        k1 = d[0] if len(d) > 0 else 0.0
        k2 = d[1] if len(d) > 1 else 0.0
        p1 = d[2] if len(d) > 2 else 0.0
        p2 = d[3] if len(d) > 3 else 0.0
        k3 = d[4] if len(d) > 4 else 0.0
        k4 = d[5] if len(d) > 5 else 0.0
        k5 = d[6] if len(d) > 6 else 0.0
        k6 = d[7] if len(d) > 7 else 0.0

        r2 = x * x + y * y

        # Horner form for numerator and denominator
        numerator = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        denominator = 1.0 + r2 * (k4 + r2 * (k5 + r2 * k6))
        safe_denom = np.where(np.abs(denominator) < _DENOM_EPS, _DENOM_EPS, denominator)
        radial = numerator / safe_denom

        x_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = x * radial + x_tan
        y = y * radial + y_tan

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def _project_division(self, pts: np.ndarray) -> np.ndarray:
        """Division undistortion model.

        D = (k1,)

        Distorted: x_d = x / (1 + k1·r²). Simple single-parameter wide-angle model.
        """
        z = np.where(np.abs(pts[:, 2]) < _DEPTH_EPS, _DEPTH_EPS, pts[:, 2])
        x = pts[:, 0] / z
        y = pts[:, 1] / z

        d = self._dist_coeffs
        k1 = d[0] if len(d) > 0 else 0.0

        r2 = x * x + y * y
        denom = 1.0 + k1 * r2
        safe_denom = np.where(np.abs(denom) < _DENOM_EPS, _DENOM_EPS, denom)
        x = x / safe_denom
        y = y / safe_denom

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def _project_mei_unified(self, pts: np.ndarray) -> np.ndarray:
        """Mei Unified omnidirectional camera model.

        D = (xi [, k1, k2])

        Projects onto unit sphere, then divides by (z + xi) to model the
        mirror/sphere, applies radial distortion, then scales by focal
        length and principal point.
        """
        norm = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
        safe_norm = np.where(norm < _NORM_EPS, _NORM_EPS, norm)
        x = pts[:, 0] / safe_norm
        y = pts[:, 1] / safe_norm
        z = pts[:, 2] / safe_norm

        d = self._dist_coeffs
        xi = d[0] if len(d) > 0 else 0.0
        k1 = d[1] if len(d) > 1 else 0.0
        k2 = d[2] if len(d) > 2 else 0.0

        denom = z + xi
        safe_denom = np.where(np.abs(denom) < _DENOM_EPS, _DENOM_EPS, denom)
        x = x / safe_denom
        y = y / safe_denom

        # Radial distortion (unconditional — branch-free)
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * k2)  # Horner form
        x = x * radial
        y = y * radial

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.column_stack([u, v])

    def _project_fisheye62(self, pts: np.ndarray) -> np.ndarray:
        """Project Aria Fisheye62 model.

        D = (k0, k1, k2, k3, p0, p1)

        Equidistant-style with 4 radial + 2 tangential coefficients.
        """
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(r, z)

        d = self._dist_coeffs
        k0 = d[0] if len(d) > 0 else 0.0
        k1 = d[1] if len(d) > 1 else 0.0
        k2 = d[2] if len(d) > 2 else 0.0
        k3 = d[3] if len(d) > 3 else 0.0
        p0 = d[4] if len(d) > 4 else 0.0
        p1 = d[5] if len(d) > 5 else 0.0

        theta2 = theta * theta
        # Horner form: θ·(1 + θ²·(k0 + θ²·(k1 + θ²·(k2 + θ²·k3))))
        theta_d = theta * (1 + theta2 * (k0 + theta2 * (k1 + theta2 * (k2 + theta2 * k3))))

        safe_r = np.where(r < _RADIAL_EPS, 1.0, r)
        scale = np.where(r < _RADIAL_EPS, 1.0, theta_d / safe_r)
        x_d = x * scale
        y_d = y * scale

        # Tangential
        r2_d = x_d * x_d + y_d * y_d
        x_d = x_d + 2 * p0 * x_d * y_d + p1 * (r2_d + 2 * x_d * x_d)
        y_d = y_d + p0 * (r2_d + 2 * y_d * y_d) + 2 * p1 * x_d * y_d

        u = self.fx * x_d + self.cx
        v = self.fy * y_d + self.cy
        return np.column_stack([u, v])

    def to_dict(self) -> dict[str, Any]:
        """Serialize camera projection to a JSON-compatible dictionary."""
        result = {
            "type": "CameraProjection",
            "intrinsic_matrix": self._intrinsic_matrix.tolist(),
            "dist_coeffs": self._dist_coeffs.tolist(),
            "projection_model": self._projection_model.value,
            "dtype": np.dtype(self.dtype).name,
        }
        if self._image_size is not None:
            result["image_size"] = list(self._image_size)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraProjection:
        """Deserialize camera projection from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        dist_coeffs = data.get("dist_coeffs", [])
        projection_model_str = data.get("projection_model", "Pinhole")
        image_size = data.get("image_size", None)
        if image_size is not None:
            image_size = tuple(image_size)

        return cls(
            intrinsic_matrix=np.array(data["intrinsic_matrix"]),
            dist_coeffs=dist_coeffs,
            projection_model=projection_model_str,
            image_size=image_size,
            dtype=dtype,
        )

    def __repr__(self) -> str:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        return f"CameraProjection(fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f})"

    def inverse(self) -> InverseCameraProjection:
        """
        Returns an InverseCameraProjection for unprojection.

        Preserves the camera parameters (intrinsics) in the inverse object.
        """
        return InverseCameraProjection(self)


@register_transform
class InverseCameraProjection(InverseProjection):
    """
    The inverse of a CameraProjection.

    Preserves the internal CameraProjection instance to maintain access to
    intrinsics (K) and distortion coefficients.
    """

    def __init__(self, camera_projection: CameraProjection):
        """
        Create an InverseCameraProjection.

        Args:
            camera_projection: The original CameraProjection instance.
        """
        self._camera_projection = camera_projection
        # Initialize parent with the matrix (will be pseudo-inverted by parent's as_matrix)
        super().__init__(original_matrix=camera_projection.matrix, dtype=camera_projection.dtype)

    @property
    def camera_projection(self) -> CameraProjection:
        """The original CameraProjection instance."""
        return self._camera_projection

    # Shortcuts exposing the camera parameters
    @property
    def fx(self) -> float:
        """Focal length in x from the original CameraProjection."""
        return self._camera_projection.fx

    @property
    def fy(self) -> float:
        """Focal length in y from the original CameraProjection."""
        return self._camera_projection.fy

    @property
    def cx(self) -> float:
        """Principal point x from the original CameraProjection."""
        return self._camera_projection.cx

    @property
    def cy(self) -> float:
        """Principal point y from the original CameraProjection."""
        return self._camera_projection.cy

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """The 3x3 intrinsic matrix K from the original CameraProjection."""
        return self._camera_projection.intrinsic_matrix

    def inverse(self) -> CameraProjection:
        """Returns the original CameraProjection."""
        return self._camera_projection

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary using the contained CameraProjection."""
        return {
            "type": "InverseCameraProjection",
            "camera_projection": self._camera_projection.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InverseCameraProjection:
        """Deserialize from dictionary."""
        cam_data = data["camera_projection"]
        if cam_data.get("type") != "CameraProjection":
            cam_data["type"] = "CameraProjection"

        camera_proj = CameraProjection.from_dict(cam_data)
        return cls(camera_proj)

    def __repr__(self) -> str:
        return f"InverseCameraProjection({self._camera_projection})"


# Axis presets: maps axis name → (u_world_axis, v_world_axis, flip_u, flip_v)
# u is the column axis, v is the row axis in the output image.
_ORTHO_AXIS_PRESETS: dict[str, tuple[int, int, bool, bool]] = {
    # top-down: x=forward→top (row, flipped), y=left→left (col, flipped)
    "top": (1, 0, True, True),
    # front view: y=left→left (col, flipped), z=up→top (row, flipped)
    "front": (1, 2, True, True),
    # side view: x=forward→right (col, not flipped), z=up→top (row, flipped)
    "side": (0, 2, False, True),
}


@register_transform
class OrthographicProjection(Projection):
    """
    Orthographic (parallel) projection — maps 3D to 2D without perspective.

    Unlike perspective ``CameraProjection``, this applies a pure affine
    mapping: the output pixel coordinates are a linear function of the input
    3D coordinates, with no division by depth.

    Axis conventions (default ``"top"`` / BEV):
        * **+x** (forward) → image **top** (smaller row)
        * **+y** (left)    → image **left** (smaller col)

    Supported axis presets:
        * ``"top"``   — Bird's-eye view (drops Z)
        * ``"front"`` — Front view (drops X)
        * ``"side"``  — Side view (drops Y)

    Usage::

        ortho = OrthographicProjection("top", (-50, 50), (-50, 50), 0.1)
        graph.add_transform("ego", "bev", ortho)
        pixels = transform_points(pts, graph, "lidar", "bev")

    Args:
        axis: Projection axis preset (``"top"``, ``"front"``, ``"side"``).
        u_range: World-coordinate extent along the column axis (metres).
        v_range: World-coordinate extent along the row axis (metres).
        resolution: Metres per pixel.
        dtype: Numeric data type.
    """

    def __init__(
        self,
        axis: str = "top",
        u_range: tuple[float, float] = (-50.0, 50.0),
        v_range: tuple[float, float] = (-50.0, 50.0),
        resolution: float = 0.1,
        dtype: np.dtype = np.float64,
    ):
        if axis not in _ORTHO_AXIS_PRESETS:
            raise ValueError(f"Unknown axis '{axis}', must be one of {list(_ORTHO_AXIS_PRESETS)}")

        self._axis = axis
        self._u_range = tuple(u_range)
        self._v_range = tuple(v_range)
        self._resolution = float(resolution)

        u_idx, v_idx, flip_u, flip_v = _ORTHO_AXIS_PRESETS[axis]
        self._u_idx = u_idx
        self._v_idx = v_idx

        # Build the 3x4 affine projection matrix.
        # col = (u_max - world[u_idx]) / res   if flip_u  else  (world[u_idx] - u_min) / res
        # row = (v_max - world[v_idx]) / res   if flip_v  else  (world[v_idx] - v_min) / res
        inv_res = 1.0 / resolution
        mat = np.zeros((3, 4), dtype=dtype)

        if flip_u:
            mat[0, u_idx] = -inv_res
            mat[0, 3] = u_range[1] * inv_res
        else:
            mat[0, u_idx] = inv_res
            mat[0, 3] = -u_range[0] * inv_res

        if flip_v:
            mat[1, v_idx] = -inv_res
            mat[1, 3] = v_range[1] * inv_res
        else:
            mat[1, v_idx] = inv_res
            mat[1, 3] = -v_range[0] * inv_res

        # Third row: constant 1 (homogeneous w)
        mat[2, 3] = 1.0

        super().__init__(matrix=mat, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def axis(self) -> str:
        """Projection axis preset."""
        return self._axis

    @property
    def u_range(self) -> tuple[float, float]:
        """World-coordinate extent along the column axis (metres)."""
        return self._u_range

    @property
    def v_range(self) -> tuple[float, float]:
        """World-coordinate extent along the row axis (metres)."""
        return self._v_range

    @property
    def resolution(self) -> float:
        """Metres per pixel."""
        return self._resolution

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Output image dimensions ``(H, W)`` in pixels."""
        W = int((self._u_range[1] - self._u_range[0]) / self._resolution)
        H = int((self._v_range[1] - self._v_range[0]) / self._resolution)
        return H, W

    @property
    def origin_pixel(self) -> tuple[int, int]:
        """Pixel coordinates ``(col, row)`` of the world origin ``(0, 0, 0)``."""
        px = self._apply(np.array([[0.0, 0.0, 0.0]]))[0]
        return int(px[0]), int(px[1])

    # ------------------------------------------------------------------ #
    # Core projection (override — NO perspective division)                #
    # ------------------------------------------------------------------ #

    def _apply(self, vector: np.ndarray | list | tuple) -> np.ndarray:
        """
        Project 3D points to 2D pixel coordinates (affine, no perspective).

        Args:
            vector: ``(N, 3)`` or ``(N, 4)`` array of 3D points.

        Returns:
            ``(N, 2)`` array of ``[col, row]`` pixel coordinates.
        """
        vector = np.atleast_2d(np.asarray(vector, dtype=self.dtype))

        if vector.shape[1] == 3:
            hom = np.hstack([vector, np.ones((vector.shape[0], 1), dtype=self.dtype)])
        elif vector.shape[1] == 4:
            hom = vector
        else:
            raise ValueError(f"Input must be Nx3 or Nx4, got {vector.shape}")

        # Affine projection: result[:, :2] are pixel coords, result[:, 2] == 1
        projected = (self.matrix @ hom.T).T
        return projected[:, :2]

    def project_points(self, points: np.ndarray | list | tuple) -> np.ndarray:
        """Alias for :meth:`_apply`."""
        return self._apply(points)

    # ------------------------------------------------------------------ #
    # Inverse                                                             #
    # ------------------------------------------------------------------ #

    def inverse(self) -> InverseOrthographicProjection:
        """
        Return the inverse projection.

        The inverse lifts 2D pixel coordinates back to 3D, placing them on the
        projection plane (the collapsed axis coordinate is set to 0).
        """
        return InverseOrthographicProjection(self)

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "OrthographicProjection",
            "axis": self._axis,
            "u_range": list(self._u_range),
            "v_range": list(self._v_range),
            "resolution": self._resolution,
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrthographicProjection:
        """Deserialize from a dictionary."""
        return cls(
            axis=data["axis"],
            u_range=tuple(data["u_range"]),
            v_range=tuple(data["v_range"]),
            resolution=data["resolution"],
            dtype=np.dtype(data.get("dtype", "float64")),
        )

    def __repr__(self) -> str:
        H, W = self.grid_shape
        return (
            f"OrthographicProjection(axis={self._axis!r}, "
            f"u_range={self._u_range}, v_range={self._v_range}, "
            f"res={self._resolution}, grid={W}x{H})"
        )


@register_transform
class InverseOrthographicProjection(InverseProjection):
    """
    Inverse of an :class:`OrthographicProjection`.

    Lifts 2D pixel coordinates back to 3D by inverting the affine mapping.
    The collapsed axis coordinate is set to 0 (i.e. the point is placed on
    the projection plane).
    """

    def __init__(self, ortho: OrthographicProjection):
        self._ortho = ortho
        super().__init__(original_matrix=ortho.matrix, dtype=ortho.dtype)

    @property
    def orthographic_projection(self) -> OrthographicProjection:
        """The original OrthographicProjection."""
        return self._ortho

    def _apply(self, vector: np.ndarray | list | tuple) -> np.ndarray:
        """
        Unproject 2D pixel coordinates to 3D (on the projection plane).

        Args:
            vector: ``(N, 2)`` pixel coords ``[col, row]``.

        Returns:
            ``(N, 3)`` array of 3D points (collapsed axis = 0).
        """
        vector = np.atleast_2d(np.asarray(vector, dtype=self.dtype))
        if vector.shape[1] != 2:
            raise ValueError(f"Expected Nx2 pixel coordinates, got {vector.shape}")

        cols = vector[:, 0]
        rows = vector[:, 1]

        u_idx, v_idx, flip_u, flip_v = _ORTHO_AXIS_PRESETS[self._ortho.axis]
        res = self._ortho.resolution
        u_range = self._ortho.u_range
        v_range = self._ortho.v_range

        # Invert the affine mapping
        if flip_u:
            world_u = u_range[1] - cols * res
        else:
            world_u = u_range[0] + cols * res

        if flip_v:
            world_v = v_range[1] - rows * res
        else:
            world_v = v_range[0] + rows * res

        # Build 3D points (collapsed axis = 0)
        N = len(cols)
        points = np.zeros((N, 3), dtype=self.dtype)
        points[:, u_idx] = world_u
        points[:, v_idx] = world_v
        return points

    def inverse(self) -> OrthographicProjection:
        """Return the original OrthographicProjection."""
        return self._ortho

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "type": "InverseOrthographicProjection",
            "orthographic_projection": self._ortho.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InverseOrthographicProjection:
        """Deserialize from a dictionary produced by to_dict."""
        ortho = OrthographicProjection.from_dict(data["orthographic_projection"])
        return cls(ortho)

    def __repr__(self) -> str:
        return f"InverseOrthographicProjection({self._ortho})"


@register_transform
class CompositeProjection(Projection):
    """
    Represents a composition of a Projection (Intrinsics) and a Transform (Extrinsics).

    Equivalent to: Projection * Transform
    P_composite = K * T

    Projects from the source frame of T directly to 2D.

    Structure:
    - projection: The Projection component (applied last/leftmost)
    - transform: The Transform component (applied first/rightmost)
    """

    def __init__(self, projection: Projection, transform: Transform, dtype: np.dtype = np.float64):
        self._projection = projection
        self._transform = transform

        # Calculate matrix for BaseTransform compatibility
        # Matrix = K * T
        matrix = projection.as_matrix() @ transform.as_matrix()
        super().__init__(matrix=matrix, dtype=dtype)

    @property
    def projection(self) -> Projection:
        """The intrinsic Projection component (applied last in the chain)."""
        return self._projection

    @property
    def transform(self) -> Transform:
        """The extrinsic Transform component (applied first in the chain)."""
        return self._transform

    def inverse(self) -> InverseCompositeProjection:
        """Return the inverse as an InverseCompositeProjection (T_inv * K_inv)."""
        return InverseCompositeProjection(self._transform.inverse(), self._projection.inverse())

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        if isinstance(other, Transform):
            # Composite * Transform = (K * T_old) * T_new = K * (T_old * T_new)
            # Update transform component
            new_transform = self._transform * other
            # Result must be a Transform (T_old * T_new is Transform * Transform -> Transform)
            if not isinstance(new_transform, Transform):
                # Fallback if transform degrades (unlikely)
                return super().__mul__(other)

            return CompositeProjection(self._projection, new_transform, dtype=self.dtype)

        # Inherit strict checks from Projection
        return super().__mul__(other)

    def to_dict(self) -> dict[str, Any]:
        """Serialize CompositeProjection (projection + transform) to dictionary."""
        return {
            "type": "CompositeProjection",
            "projection": self._projection.to_dict(),
            "transform": self._transform.to_dict(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositeProjection:
        """Deserialize CompositeProjection from dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        projection = deserialize_transform(data["projection"])
        transform = deserialize_transform(data["transform"])
        if not isinstance(projection, Projection):
            raise ValueError("CompositeProjection projection must be a Projection")
        if not isinstance(transform, Transform):
            raise ValueError("CompositeProjection transform must be a Transform")
        return cls(projection, transform, dtype=dtype)

    def __repr__(self) -> str:
        return f"CompositeProjection(projection={self._projection}, transform={self._transform})"


@register_transform
class InverseCompositeProjection(InverseProjection):
    """
    Represents the inverse of a CompositeProjection.

    Equivalent to: Transform * InverseProjection
    P_inv_composite = T * K_inv

    Unprojects from 2D to the source frame of T.

    Structure:
    - transform: The Transform component (applied last/leftmost)
    - projection: The InverseProjection component (applied first/rightmost)
    """

    def __init__(
        self, transform: Transform, projection: InverseProjection, dtype: np.dtype = np.float64
    ):
        self._transform = transform
        self._projection = projection

        # Calculate matrix for BaseTransform compatibility
        # Matrix = T * K_inv
        transform.as_matrix() @ projection.as_matrix()
        # Pass a dummy matrix to super, we override everything anyway.
        # But for correctness, passing what we think is the "original" is hard.
        # Let's assume standard behavior.
        # super needs "original_matrix" that is the PROJECTION matrix.
        # P_comp = K * T_inv.
        # So original = (K * T_inv).matrix

        # NOTE: We can't easily construct the unified projection
        # matrix just from pieces without logic,
        # but let's try.
        try:
            # K * T_inv
            orig_proj_mat = projection.inverse().as_matrix() @ transform.inverse().as_matrix()
            super().__init__(original_matrix=orig_proj_mat, dtype=dtype)
        except Exception:
            # Fallback
            super().__init__(original_matrix=np.eye(4), dtype=dtype)

    @property
    def transform(self) -> Transform:
        """The extrinsic Transform component (applied first in the chain)."""
        return self._transform

    @property
    def projection(self) -> InverseProjection:
        """The inverse Projection component (applied last in the chain)."""
        return self._projection

    def as_matrix(self) -> np.ndarray:
        """Return the combined T * K_inv matrix."""
        return self._transform.as_matrix() @ self._projection.as_matrix()

    def inverse(self) -> CompositeProjection:
        """Return the inverse as a CompositeProjection (K * T_inv)."""
        return CompositeProjection(
            self._projection.inverse(), self._transform.inverse(), dtype=self.dtype
        )

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        # Inherit strict checks from InverseProjection
        return super().__mul__(other)

    def __rmul__(self, other: BaseTransform) -> BaseTransform:
        # Handle Transform * InverseCompositeProjection
        # T_new * (T_old * K_inv) = (T_new * T_old) * K_inv
        if isinstance(other, Transform):
            new_transform = other * self._transform
            if isinstance(new_transform, Transform):
                return InverseCompositeProjection(new_transform, self._projection, dtype=self.dtype)

        return NotImplemented

    def to_dict(self) -> dict[str, Any]:
        """Serialize InverseCompositeProjection to dictionary."""
        return {
            "type": "InverseCompositeProjection",
            "transform": self._transform.to_dict(),
            "projection": self._projection.to_dict(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InverseCompositeProjection:
        """Deserialize InverseCompositeProjection from dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        transform = deserialize_transform(data["transform"])
        projection = deserialize_transform(data["projection"])
        if not isinstance(transform, Transform):
            raise ValueError("InverseCompositeProjection transform must be a Transform")
        if not isinstance(projection, InverseProjection):
            raise ValueError("InverseCompositeProjection projection must be an InverseProjection")
        return cls(transform, projection, dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"InverseCompositeProjection("
            f"transform={self._transform}, "
            f"projection={self._projection})"
        )


class Pose:
    """
    A user-friendly wrapper around Transform for pose representation.

    Represents the pose of 'child_frame_id' relative to 'frame_id'.
    """

    def __init__(
        self,
        position: np.ndarray | list | tuple | None = None,
        orientation: quaternion.quaternion | np.ndarray | list | tuple | None = None,
        frame_id: str | int | UUID | None = None,
        child_frame_id: str | int | UUID | None = None,
    ):
        # Use ensure_rotation logic but for rvec too
        quat = quaternion.one
        if orientation is not None:
            if isinstance(orientation, quaternion.quaternion):
                quat = orientation.normalized()
            elif isinstance(orientation, (list, tuple, np.ndarray)):
                arr = np.array(orientation, dtype=np.float64).flatten()
                if arr.size == 4:
                    quat = quaternion.quaternion(*arr).normalized()
                elif arr.size == 3:
                    # Rotation vector
                    theta = np.linalg.norm(arr)
                    if theta > 1e-8:
                        quat = quaternion.from_rotation_vector(arr)
                else:
                    raise ValueError("Orientation must be quaternion (4) or rvec (3)")

        self._transform = Transform(translation=position, rotation=quat)
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id

    @property
    def position(self) -> np.ndarray:
        """The 3D position [x, y, z] in the parent frame."""
        return self._transform.translation.flatten()

    @position.setter
    def position(self, value: np.ndarray | list | tuple):
        """Set the 3D position [x, y, z]."""
        self._transform.translation = ensure_translation(value, self._transform.dtype)

    @property
    def orientation(self) -> quaternion.quaternion:
        """The orientation as a unit quaternion."""
        return self._transform.rotation

    @orientation.setter
    def orientation(self, value: quaternion.quaternion | np.ndarray | list | tuple):
        """Set the orientation from a quaternion or [w, x, y, z] array."""
        self._transform.rotation = ensure_rotation(value, self._transform.dtype)

    def as_transform(self) -> Transform:
        """Returns the underlying Transform object."""
        return self._transform

    @classmethod
    def from_transform(
        cls,
        tf: Transform,
        frame_id: str | int | UUID | None = None,
        child_frame_id: str | int | UUID | None = None,
    ) -> Pose:
        """Creates a Pose from a Transform."""
        return cls(
            position=tf.translation.flatten(),
            orientation=tf.rotation,
            frame_id=frame_id,
            child_frame_id=child_frame_id,
        )

    def inverse(
        self,
        new_frame_id: str | int | UUID | None = None,
        new_child_frame_id: str | int | UUID | None = None,
    ) -> Pose:
        """
        Returns the inverse pose.

        By default, swaps frame_id and child_frame_id:
        Inverse(T_A->B) = T_B->A
        """
        # Default behavior: swap frames
        target_frame_id = new_frame_id if new_frame_id is not None else self.child_frame_id
        target_child_frame_id = (
            new_child_frame_id if new_child_frame_id is not None else self.frame_id
        )

        return Pose.from_transform(
            self._transform.inverse(),
            frame_id=target_frame_id,
            child_frame_id=target_child_frame_id,
        )

    def compose(self, other):
        """Returns self * other"""
        # Logic: T_A_C = T_A_B * T_B_C
        new_frame_id = self.frame_id
        new_child_frame_id = None

        if isinstance(other, Pose):
            # Strict Frame Check
            # Only check if both are explicitly defined (not None)
            if (
                self.child_frame_id is not None
                and other.frame_id is not None
                and self.child_frame_id != other.frame_id
            ):
                raise ValueError(
                    f"Frame mismatch in composition: "
                    f"Pose 1 ends in '{self.child_frame_id}' but "
                    f"Pose 2 starts in '{other.frame_id}'."
                )

            new_child_frame_id = other.child_frame_id
            return Pose.from_transform(
                self._transform * other.as_transform(),
                frame_id=new_frame_id,
                child_frame_id=new_child_frame_id,
            )

        return Pose.from_transform(
            self._transform * other, frame_id=new_frame_id, child_frame_id=new_child_frame_id
        )

    def __mul__(self, other: Pose | Transform) -> Pose:
        return self.compose(other)

    def to_list(self) -> list[float]:
        """Returns [px, py, pz, qw, qx, qy, qz]"""
        q = self.orientation
        p = self.position
        return [
            float(p[0]),
            float(p[1]),
            float(p[2]),
            float(q.w),
            float(q.x),
            float(q.y),
            float(q.z),
        ]

    def to_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transformation matrix."""
        return self._transform.as_matrix()

    def __repr__(self) -> str:
        elements = [f"position={self.position!r}", f"orientation={self.orientation!r}"]
        if self.frame_id:
            elements.append(f"frame_id={self.frame_id!r}")
        if self.child_frame_id:
            elements.append(f"child_frame_id={self.child_frame_id!r}")
        return f"Pose({', '.join(elements)})"


def transform_points(
    points: np.ndarray,
    transform_object: BaseTransform | TransformGraph,
    source_frame: str | None = None,
    target_frame: str | None = None,
) -> np.ndarray:
    """
    Applies a transformation to a set of 3D points.

    Supports polymorphic second argument:
    1. transform_points(points, transform)
       Directly applies a transform object.
    2. transform_points(points, graph, source_frame, target_frame)
       Uses the graph to find the transform from source to target.
       If target is a projection frame (e.g. image), returns unnormalized 3D
       coordinates [u*z, v*z, z].

    Args:
        points: Nx3 points array.
        transform_object: BaseTransform object OR TransformGraph.
        source_frame: Source frame ID (required if using graph).
        target_frame: Target frame ID (required if using graph).

    Returns:
        np.ndarray: Nx3 array of transformed points.
    """
    points = np.atleast_2d(points)

    # CASE 1: TransformGraph
    if hasattr(transform_object, "get_transform"):
        graph: TransformGraph = transform_object  # type: ignore
        if source_frame is None or target_frame is None:
            raise ValueError(
                "When using TransformGraph, both 'source_frame' and "
                "'target_frame' must be provided. "
                "Usage: transform_points(points, graph, "
                "source_frame='A', target_frame='B')"
            )
        transform = graph.get_transform(source_frame, target_frame)
        # Recurse with resolved transform
        return transform_points(points, transform)

    # CASE 2: BaseTransform
    elif isinstance(transform_object, BaseTransform):
        transform = transform_object

        # Check for projection (special handling for "transform_points" vs "project_points")
        if isinstance(transform, Projection):
            # CameraProjection needs full model-dispatched projection (distortion-aware).
            # Other Projection subclasses (e.g., OrthographicProjection) are linear.
            if isinstance(transform, CameraProjection):
                pts = points
                if pts.shape[1] == 4:
                    pts = pts[:, :3] / pts[:, 3:4]
                elif pts.shape[1] != 3:
                    raise ValueError("Points must be Nx3 or Nx4")

                z = pts[:, 2]
                uv = transform._apply(pts)  # full model projection → [u, v]
                return np.column_stack([uv[:, 0] * z, uv[:, 1] * z, z])
            else:
                # Linear projection (OrthographicProjection etc.) — use matrix path
                N = points.shape[0]
                if points.shape[1] == 3:
                    hom_points = np.hstack(
                        [points, np.ones((N, 1), dtype=transform.dtype)]
                    )
                elif points.shape[1] == 4:
                    hom_points = points
                else:
                    raise ValueError("Points must be Nx3 or Nx4")
                res_hom = (transform.as_matrix() @ hom_points.T).T
                return res_hom[:, :3]

        if not isinstance(
            transform,
            (Transform, Rotation, Translation, Identity, MatrixTransform, InverseProjection),
        ):
            raise TypeError(
                f"Unsupported transform type: {type(transform).__name__}. "
                "Supported: Rigid transforms, InverseProjection, or Projection "
                "(for unnormalized 3D output)."
            )
        # InverseProjection accepts Nx2 (pixel coords) → Nx3
        if isinstance(transform, InverseProjection) and points.shape[1] == 2:
            return transform._apply(points)

        if points.shape[1] == 3:
            hom_points = np.hstack([points, np.ones((points.shape[0], 1), dtype=transform.dtype)])
            transformed_hom = (transform.as_matrix() @ hom_points.T).T
            return transformed_hom[:, :3]
        elif points.shape[1] == 4:
            transformed = (transform.as_matrix() @ points.T).T
            return transformed
        else:
            raise ValueError("Points must be Nx2 (for InverseProjection), Nx3, or Nx4")

    else:
        obj_type = type(transform_object).__name__
        raise TypeError(f"transform_object must be BaseTransform or TransformGraph, got {obj_type}")


def project_points(
    points: np.ndarray,
    transform_object: BaseTransform | TransformGraph,
    source_frame: str | None = None,
    target_frame: str | None = None,
) -> np.ndarray:
    """
    Projects 3D points to 2D coordinates (homogenized).

    Signatures:
    1. project_points(points, projection_transform)
    2. project_points(points, graph, source_frame, target_frame)

    Returns:
        np.ndarray: Nx2 array of pixel coordinates [u, v].
    """
    points = np.atleast_2d(points)

    # CASE 1: TransformGraph
    if hasattr(transform_object, "get_transform"):
        graph: TransformGraph = transform_object  # type: ignore
        if source_frame is None or target_frame is None:
            raise ValueError(
                "When using TransformGraph, both 'source_frame' and "
                "'target_frame' must be provided."
            )
        transform = graph.get_transform(source_frame, target_frame)
        return project_points(points, transform)

    # CASE 2: BaseTransform
    elif isinstance(transform_object, BaseTransform):
        transform = transform_object

        # If transform is Projection, _apply() does homogenization (div by z) -> 2D
        if isinstance(transform, Projection):
            return transform._apply(points)
        elif isinstance(transform, (Transform, Rotation, Translation, Identity, MatrixTransform)):
            raise TypeError(
                "Cannot project_points using a rigid transform. "
                "Target frame must be a projection frame."
            )
        else:
            # Try _apply anyway if it supports it?
            res = transform._apply(points)
            if res.shape[1] != 2:
                raise ValueError(
                    f"Transform returned {res.shape[1]}D points, expected 2D for project_points."
                )
            return res

    else:
        obj_type = type(transform_object).__name__
        raise TypeError(f"transform_object must be BaseTransform or TransformGraph, got {obj_type}")


def get_basis_vectors(
    transform: BaseTransform, length: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the origin and basis vectors (x, y, z) of the transform's coordinate frame.

    Args:
        transform: The transform defining the coordinate system.
        length: Length of the basis vectors.

    Returns:
        Tuple: (origin, x_vec, y_vec, z_vec) as 3-element numpy arrays in global frame.
    """
    matrix = transform.as_matrix()
    origin = matrix[:3, 3]
    rotation = matrix[:3, :3]

    x_vec = origin + rotation @ np.array([length, 0, 0])
    y_vec = origin + rotation @ np.array([0, length, 0])
    z_vec = origin + rotation @ np.array([0, 0, length])

    return origin, x_vec, y_vec, z_vec


# -----------------------------------------------------------------------------
# Transform Graph
# -----------------------------------------------------------------------------


class TransformGraph:
    """
    Manages a graph of coordinate frames connected by spatial transformations.

    Uses an undirected NetworkX graph with directional metadata on edges.
    Each edge stores:
        - transform: The BaseTransform object
        - parent: The source frame (defines the "natural" direction)
        - is_cache: Whether this is a cached shortcut (True) or added edge (False)
        - weight: 1.0 for added edges, 0.1 for cached shortcuts

    Features:
        - Automatic path finding and transform composition
        - Lazy inversion when traversing against natural direction
        - Shortcut caching for O(1) repeated queries
        - Dependency-aware cache invalidation
        - JSON-compatible serialization
    """

    # Edge weights for path finding
    ADDED_EDGE_WEIGHT = 1.0
    CACHED_EDGE_WEIGHT = 1.0

    def __init__(self):
        self._graph = nx.Graph()
        # Maps (source, target) added edges to list of cache edges that depend on them
        self._dependency_map: dict[tuple[str, str], list[tuple[str, str]]] = {}

    @property
    def graph(self) -> nx.Graph:
        """Returns the internal NetworkX graph (read-only)."""
        return self._graph

    @property
    def frames(self) -> list[str]:
        """Returns list of all frame IDs in the graph."""
        return list(self._graph.nodes())

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Returns list of all added edges as (reference_frame, target_frame) tuples."""
        result = []
        for u, v, data in self._graph.edges(data=True):
            if not data.get("is_cache", False):
                reference_frame = data["reference_frame"]
                target_frame = v if reference_frame == u else u
                result.append((reference_frame, target_frame))
        return result

    def has_frame(self, frame_id: str) -> bool:
        """Check if a frame exists in the graph."""
        return frame_id in self._graph

    def has_transform(self, source_frame: str, target_frame: str) -> bool:
        """Check if a direct transform (edge) exists between two frames."""
        return self._graph.has_edge(source_frame, target_frame)

    def add_transform(
        self,
        source_frame: str,
        target_frame: str,
        transform: BaseTransform,
    ) -> None:
        """
        Add a transform between two frames.

        API: add_transform(source, target, transform)
        - SOURCE: The domain frame (where vectors start).
        - TARGET: The codomain/reference frame (where vectors end).
        - TRANSFORM: Source→Target operator.

        The transform maps Source coordinates to Target coordinates:
        P_target = transform * P_source

        Args:
            source_frame: The source/domain frame ID.
            target_frame: The target/reference frame ID.
            transform: The transform from source to target.

        Raises:
            ValueError: If an edge already exists between these frames.
        """
        if self._graph.has_edge(source_frame, target_frame):
            raise ValueError(
                f"Transform between '{source_frame}' and '{target_frame}' already exists. "
                "Use update_transform() to modify it."
            )

        # Store edge between nodes.
        # We store "reference_frame" to indicate which node is the Target (Codomain)
        # of the transform.
        # If reference_frame = target_frame, then the transform is source -> target.
        self._graph.add_edge(
            target_frame,
            source_frame,
            transform=transform,
            reference_frame=target_frame,
            is_cache=False,
            weight=self.ADDED_EDGE_WEIGHT,
        )

    def update_transform(
        self,
        source_frame: str,
        target_frame: str,
        transform: BaseTransform,
    ) -> None:
        """
        Update an existing transform between two frames.

        Automatically invalidates any cached shortcuts that depend on this edge.

        Args:
            source_frame: The source frame ID.
            target_frame: The target frame ID.
            transform: The new transform from source to target.

        Raises:
            ValueError: If no edge exists between these frames.
        """
        if not self._graph.has_edge(target_frame, source_frame):
            raise ValueError(
                f"No transform between '{source_frame}' and '{target_frame}'. "
                "Use add_transform() to create it."
            )

        # Invalidate dependent caches
        self._invalidate_caches_for_edge(target_frame, source_frame)

        # Update the transform
        self._graph[target_frame][source_frame]["transform"] = transform
        self._graph[target_frame][source_frame]["reference_frame"] = target_frame

    def remove_transform(self, frame_a: str, frame_b: str) -> None:
        """
        Remove a transform (edge) between two frames.

        Args:
            frame_a: First frame ID.
            frame_b: Second frame ID.

        Raises:
            ValueError: If no edge exists between these frames.
        """
        if not self._graph.has_edge(frame_a, frame_b):
            raise ValueError(f"No transform between '{frame_a}' and '{frame_b}'.")

        # Invalidate dependent caches
        self._invalidate_caches_for_edge(frame_a, frame_b)

        # Remove the edge
        self._graph.remove_edge(frame_a, frame_b)

        # Clean up isolated nodes
        for frame in [frame_a, frame_b]:
            if self._graph.degree(frame) == 0:
                self._graph.remove_node(frame)

    def get_transform(self, source_frame: str, target_frame: str) -> BaseTransform:
        """
        Get the transform from source_frame to target_frame.

        Automatically finds the shortest path and composes transforms.
        Results are cached as shortcut edges for O(1) subsequent lookups.

        Args:
            source_frame: The source frame ID.
            target_frame: The target frame ID.

        Returns:
            BaseTransform: The composed transform T_source_to_target.

        Raises:
            ValueError: If either frame doesn't exist or no path exists.
        """
        if source_frame == target_frame:
            return Identity()

        if source_frame not in self._graph:
            raise ValueError(f"Frame '{source_frame}' not found in graph.")
        if target_frame not in self._graph:
            raise ValueError(f"Frame '{target_frame}' not found in graph.")

        # Check for direct edge (including cached shortcuts)
        if self._graph.has_edge(source_frame, target_frame):
            edge_data = self._graph[source_frame][target_frame]
            transform = edge_data["transform"]
            reference_frame = edge_data["reference_frame"]

            if reference_frame == source_frame:
                # source is reference_frame (Target). Going to target_frame (Source).
                # Direction: Target -> Source.
                # Use inverse.
                return transform.inverse()
            return transform  # source is Source. Going to reference_frame (Target). Use direct.

        # Find shortest path
        try:
            path = nx.shortest_path(self._graph, source_frame, target_frame, weight="weight")
        except nx.NetworkXNoPath:
            raise ValueError(f"No path from '{source_frame}' to '{target_frame}'.")

        # Compose transforms along path
        composed_transform = Identity()
        added_edges = []

        for i in range(len(path) - 1):
            current_frame = path[i]
            next_frame = path[i + 1]

            edge_data = self._graph[current_frame][next_frame]
            transform = edge_data["transform"]
            reference_frame = edge_data["reference_frame"]

            if not edge_data.get("is_cache", False):
                # Normalize edge key (always smaller, larger)
                # Use str(frame) for consistent sorting across mixed types (int vs str vs UUID)
                sorted_frames = sorted([current_frame, next_frame], key=str)
                edge_key = tuple(sorted_frames)
                added_edges.append(edge_key)

            # Traversal Logic:
            # If current is Ref (Target): Going to Source. Use Inverse.
            # If current is Source: Going to Ref (Target). Use Direct.
            if reference_frame == current_frame:
                step_transform = transform.inverse()
            else:
                step_transform = transform

            # Compose in reverse order: new_step * accumulated
            # This ensures: (T3 * T2 * T1) transforms correctly
            composed_transform = step_transform * composed_transform

        # Cache the result as a shortcut edge
        self._add_cache_edge(source_frame, target_frame, composed_transform, added_edges)

        return composed_transform

    def _add_cache_edge(
        self,
        source_frame: str,
        target_frame: str,
        transform: BaseTransform,
        added_edges: list[tuple[str, str]],
    ) -> None:
        """
        Add a cached shortcut edge and register dependencies.

        The transform is source→target, edge goes from target→source.
        """
        self._graph.add_edge(
            target_frame,  # Edge from ref (target)
            source_frame,  # to source
            transform=transform,  # source→target transform
            reference_frame=target_frame,  # target is the reference_frame
            is_cache=True,
            weight=self.CACHED_EDGE_WEIGHT,
        )

        # Register cache dependency for all constituent edges
        # Use str() key for consistent sorting
        sorted_frames = sorted([source_frame, target_frame], key=str)
        cache_edge = tuple(sorted_frames)
        for added_edge in added_edges:
            if added_edge not in self._dependency_map:
                self._dependency_map[added_edge] = []
            self._dependency_map[added_edge].append(cache_edge)

    def _invalidate_caches_for_edge(self, frame_a: str, frame_b: str) -> None:
        """
        Remove all cached edges that depend on the edge (frame_a, frame_b).
        """
        sorted_frames = sorted([frame_a, frame_b], key=str)
        edge_key = tuple(sorted_frames)
        if edge_key in self._dependency_map:
            for cache_u, cache_v in self._dependency_map[edge_key]:
                if self._graph.has_edge(cache_u, cache_v) and self._graph[cache_u][cache_v].get(
                    "is_cache", False
                ):
                    self._graph.remove_edge(cache_u, cache_v)
            # Clear dependencies for this edge
            del self._dependency_map[edge_key]

    def is_projection_frame(self, frame_id: str) -> bool:
        """
        Check if a frame is a 2D projection frame (e.g., an Image frame).

        Rule: A frame is a projection frame if ALL edges connected to it treat it
        as a projection space.
        - If transform maps INTO frame (frame is Target), transform must be a Projection.
        - If transform maps OUT OF frame (frame is Source), transform must be an InverseProjection.
        """
        if frame_id not in self._graph:
            return False

        neighbors = list(self._graph.neighbors(frame_id))
        if not neighbors:
            return False

        for neighbor in neighbors:
            edge_data = self._graph[frame_id][neighbor]
            transform = edge_data["transform"]
            reference_frame = edge_data["reference_frame"]

            # Use 'reference_frame' to determine direction.
            # reference_frame is the TARGET of the transform.

            if reference_frame == frame_id:
                # Transform is Neighbor -> Frame (Source -> Target)
                # For Frame to be 2D, this must be a Projection (3D -> 2D)
                if not isinstance(transform, Projection):
                    return False
            else:
                # Transform is Frame -> Neighbor (Source -> Target)
                # For Frame to be 2D, this must be an InverseProjection (2D -> 3D)
                if not isinstance(transform, InverseProjection):
                    return False

        return True

    def _get_camera_intrinsics_and_pose(self, image_frame: str) -> tuple[np.ndarray, str]:
        """
        Helper: Find the connected 3D camera frame and K matrix for an image frame.

        Returns:
            (K, camera_frame_id)
        """
        if image_frame not in self._graph:
            raise ValueError(f"Frame '{image_frame}' not found.")

        for neighbor in self._graph.neighbors(image_frame):
            edge_data = self._graph[image_frame][neighbor]
            transform = edge_data["transform"]
            reference_frame = edge_data["reference_frame"]

            K = None
            cam_frame = None

            # Logic: If frame is projection frame, the neighbor must be the camera
            if isinstance(transform, CameraProjection):
                if reference_frame == image_frame:  # Proj maps Neighbor->Image
                    K = transform.intrinsic_matrix
                    cam_frame = neighbor

            elif isinstance(transform, InverseCameraProjection):
                if reference_frame != image_frame:  # InvProj maps Image->Neighbor
                    K = transform.intrinsic_matrix
                    cam_frame = neighbor

            if K is not None:
                return K, cam_frame

        raise ValueError(
            f"Frame '{image_frame}' is not a valid projection frame connected to a camera."
        )

    def get_essential_matrix(self, image_frame_1: str, image_frame_2: str) -> np.ndarray:
        """
        Compute the Essential Matrix E between two image frames.

        E = [t]_x R
        where R, t describe method to transform points from Camera 1 to Camera 2.
        X2 = R X1 + t.
        """
        _, c1 = self._get_camera_intrinsics_and_pose(image_frame_1)
        _, c2 = self._get_camera_intrinsics_and_pose(image_frame_2)

        # Get transform from C1 to C2
        # T_c1_to_c2: converts X_c1 to X_c2.
        T_12 = self.get_transform(c1, c2)
        if not isinstance(T_12, Transform):
            raise ValueError(f"Transform between '{c1}' and '{c2}' is not a spatial Transform.")

        R = quaternion.as_rotation_matrix(T_12.rotation)
        t = T_12.translation.flatten()

        # Skew symmetric matrix of t
        t_x = skew(t)

        return t_x @ R

    def get_fundamental_matrix(self, image_frame_1: str, image_frame_2: str) -> np.ndarray:
        """
        Compute the Fundamental Matrix F between two image frames.

        F = K2^-T E K1^-1
        x2^T F x1 = 0
        """
        K1, _ = self._get_camera_intrinsics_and_pose(image_frame_1)
        K2, _ = self._get_camera_intrinsics_and_pose(image_frame_2)

        E = self.get_essential_matrix(image_frame_1, image_frame_2)

        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)

        return K2_inv.T @ E @ K1_inv

    def get_homography(
        self,
        image_frame_1: str,
        image_frame_2: str,
        plane_normal: np.ndarray,
        plane_distance: float,
    ) -> np.ndarray:
        """
        Compute Homography H mapping pixels from image 1 to image 2 induced by a plane.

        x2 ~ H x1

        Plane equation in Camera 1 frame: n^T X = d
        H = K2 (R + t n^T / d) K1^-1

        Args:
            image_frame_1: Source image frame.
            image_frame_2: Target image frame.
            plane_normal: Normal vector of the plane in Camera 1 frame (3,).
            plane_distance: Distance to the plane in Camera 1 frame (scalar).
        """
        K1, c1 = self._get_camera_intrinsics_and_pose(image_frame_1)
        K2, c2 = self._get_camera_intrinsics_and_pose(image_frame_2)

        # T_12: C1 -> C2
        T_12 = self.get_transform(c1, c2)
        R = quaternion.as_rotation_matrix(T_12.rotation)
        t = T_12.translation.flatten().reshape(3, 1)

        n = np.asarray(plane_normal).reshape(3, 1)
        d = float(plane_distance)

        H_euclidean = R + (t @ n.T) / d

        return K2 @ H_euclidean @ np.linalg.inv(K1)

    @staticmethod
    def estimate_skew(intrinsic_matrix: np.ndarray) -> float:
        """
        Estimate the skew parameter from an intrinsic matrix K.

        K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
        Returns s.
        """
        return float(intrinsic_matrix[0, 1])

    def clear_cache(self) -> None:
        """
        Clear all cached shortcut transforms.

        Removes edges marked with is_cache=True.
        """
        edges_to_remove = [
            (u, v) for u, v, data in self._graph.edges(data=True) if data.get("is_cache", False)
        ]
        self._graph.remove_edges_from(edges_to_remove)
        self._dependency_map.clear()

    def get_connected_components(self) -> list[set[str]]:
        """
        Get all connected components in the graph.

        Returns:
            List of sets, where each set contains frame IDs of a connected component.
        """
        return list(nx.connected_components(self._graph))

    def get_connected_nodes(self, frame_id: str) -> set[str]:
        """
        Get the set of all nodes connected to the given frame (its connected component).

        Args:
            frame_id: The frame to start searching from.

        Returns:
            Set of connected frame IDs.

        Raises:
            ValueError: If frame_id is not in the graph.
        """
        if frame_id not in self._graph:
            raise ValueError(f"Frame '{frame_id}' is not in the graph.")
        return nx.node_connected_component(self._graph, frame_id)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the entire graph to a dictionary.

        Returns:
            Dict containing 'frames' and 'edges' (only explicit, non-cached edges).

        Frame IDs that are not JSON-native (tuples, datetime, UUID) are encoded
        as tagged dicts with ``__type__`` and ``value`` keys so they survive
        ``json.dumps``/``json.loads`` roundtrip without information loss.
        """
        frames = [self._encode_frame_id(f) for f in self.frames]
        edges = []
        for u, v, data in self._graph.edges(data=True):
            if not data.get("is_cache", False):
                transform = data["transform"]
                reference = data.get("reference_frame")
                source = v if reference == u else u
                edges.append(
                    {
                        "source": self._encode_frame_id(source),
                        "target": self._encode_frame_id(reference),
                        "transform": transform.to_dict(),
                    }
                )

        return {"frames": frames, "edges": edges}

    # -----------------------------------------------------------------------
    # Frame ID Serialization Helpers
    #
    # Non-JSON-native Python types used as frame IDs are encoded as tagged
    # dicts so the graph can be losslessly transmitted as JSON (e.g., via
    # HTTP requests).
    #
    # Supported types:
    #   tuple      → {"type": "tuple",      "value": [encoded elements...]}
    #   datetime   → {"type": "datetime",   "value": "2026-01-01T12:00:00+00:00"}
    #   datetime64 → {"type": "datetime64", "value": "2026-01-01T12:00:00.000000000", "unit": "ns"}
    #   UUID       → {"type": "uuid",       "value": "550e8400-..."}
    #
    # Primitives (str, int, float, bool, None) pass through unchanged.
    # -----------------------------------------------------------------------

    @staticmethod
    def _encode_frame_id(frame_id: Any) -> Any:
        """Encode a frame ID into a JSON-safe representation.

        Non-JSON-native types are wrapped in ``{"type": ..., "value": ...}``
        tagged dicts.  The encoding is recursive for compound types (tuples).
        """
        import datetime as dt

        if isinstance(frame_id, tuple):
            return {
                "type": "tuple",
                "value": [TransformGraph._encode_frame_id(item) for item in frame_id],
            }
        if isinstance(frame_id, np.datetime64):
            unit = np.datetime_data(frame_id)[0]
            return {"type": "datetime64", "value": str(frame_id), "unit": unit}
        if isinstance(frame_id, dt.datetime):
            return {"type": "datetime", "value": frame_id.isoformat()}
        if isinstance(frame_id, UUID):
            return {"type": "uuid", "value": str(frame_id)}
        # str, int, float, bool, None — JSON-native
        return frame_id

    @staticmethod
    def _decode_frame_id(frame_id: Any) -> Any:
        """Decode a JSON-deserialized frame ID back to its original Python type.

        Recognizes tagged dicts produced by ``_encode_frame_id`` and converts
        them back.  Plain lists (from untagged JSON arrays) are converted to
        tuples for backward compatibility.
        """
        import datetime as dt

        if isinstance(frame_id, dict) and "type" in frame_id:
            type_tag = frame_id["type"]
            value = frame_id["value"]
            if type_tag == "tuple":
                return tuple(TransformGraph._decode_frame_id(item) for item in value)
            if type_tag == "datetime":
                return dt.datetime.fromisoformat(value)
            if type_tag == "datetime64":
                unit = frame_id.get("unit", "ns")
                return np.datetime64(value, unit)
            if type_tag == "uuid":
                return UUID(value)
            raise ValueError(f"Unknown frame ID type tag: {type_tag!r}")
        # Backward compatibility: plain JSON arrays → tuples
        if isinstance(frame_id, list):
            return tuple(TransformGraph._decode_frame_id(item) for item in frame_id)
        return frame_id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformGraph:
        """
        Deserialize a graph from a dictionary.

        Args:
            data: Dictionary produced by to_dict().

        Returns:
            New TransformGraph instance.
        """
        graph = cls()
        for edge_data in data.get("edges", []):
            src = cls._decode_frame_id(edge_data["source"])
            tgt = cls._decode_frame_id(edge_data["target"])
            transform = deserialize_transform(edge_data["transform"])
            graph.add_transform(src, tgt, transform)

        return graph

    def __repr__(self) -> str:
        """
        String representation of the graph.
        Shows basic stats and the largest connected component.
        """
        nodes = self.frames
        if not nodes:
            return "TransformGraph(Empty)"

        edges = self.edges
        components = self.get_connected_components()

        # Sort by size, descending
        components.sort(key=len, reverse=True)
        largest = components[0]
        largest_nodes = sorted(list(largest))

        return (
            f"TransformGraph(\n"
            f"  nodes={len(nodes)}, edges={len(edges)}, components={len(components)},\n"
            f"  largest_component={largest_nodes}\n"
            f")"
        )
