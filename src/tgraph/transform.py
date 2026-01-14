from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union

import networkx as nx
import numpy as np
import quaternion

# -----------------------------------------------------------------------------
# Transform Registry for Serialization
# -----------------------------------------------------------------------------

# Global registry mapping type names to classes
_TRANSFORM_REGISTRY: dict[str, type["BaseTransform"]] = {}


def register_transform(cls: type["BaseTransform"]) -> type["BaseTransform"]:
    """
    Decorator to register a transform class for serialization.

    Usage:
        @register_transform
        class MyTransform(BaseTransform):
            ...
    """
    _TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


def serialize_transform(transform: "BaseTransform") -> dict[str, Any]:
    """
    Serialize any transform to a JSON-compatible dictionary.

    Args:
        transform: Any BaseTransform subclass instance.

    Returns:
        Dict containing the serialized transform with a "type" key.
    """
    return transform.to_dict()


def from_euler_angles(roll: float, pitch: float, yaw: float) -> quaternion.quaternion:
    """
    Create a quaternion from roll-pitch-yaw Euler angles.

    Uses the aerospace/robotics convention with intrinsic ZYX rotation order
    (yaw → pitch → roll). See Rotation class docstring for full convention details.

    The numpy-quaternion library's from_euler_angles(alpha, beta, gamma) uses
    intrinsic ZYZ convention by default, but we use it with ZYX ordering:
    - alpha (first rotation): yaw about Z
    - beta (second rotation): pitch about Y
    - gamma (third rotation): roll about X

    Args:
        roll: Rotation about X-axis in radians (banking/roll)
        pitch: Rotation about Y-axis in radians (nose up/down)
        yaw: Rotation about Z-axis in radians (heading/yaw)

    Returns:
        quaternion.quaternion: The resulting quaternion.
    """
    return quaternion.from_euler_angles(yaw, pitch, roll)


def as_euler_angles(q: quaternion.quaternion) -> tuple[float, float, float]:
    """
    Extract roll-pitch-yaw Euler angles from a quaternion.

    Uses the aerospace/robotics convention with intrinsic ZYX rotation order.
    See Rotation class docstring for full convention details.

    Args:
        q: The input quaternion.

    Returns:
        Tuple[float, float, float]: (roll, pitch, yaw) in radians.
    """
    # quaternion.as_euler_angles returns (alpha, beta, gamma) for ZYZ convention
    # which corresponds to (yaw, pitch, roll) in our ZYX convention
    yaw, pitch, roll = quaternion.as_euler_angles(q)
    return (roll, pitch, yaw)


class ProjectionModel(Enum):
    """
    Enum for different camera projection models.
    
    Supported models:
    - PINHOLE: Standard pinhole camera model (no distortion)
    - PINHOLE_POLYNOMIAL: Pinhole with polynomial radial/tangential distortion (OpenCV-style)
    - FISHEYE: Fisheye camera model
    - OMNIDIRECTIONAL: Omnidirectional camera model
    """
    PINHOLE = "Pinhole"
    PINHOLE_POLYNOMIAL = "Pinhole+Polynomial"
    FISHEYE = "Fisheye"
    OMNIDIRECTIONAL = "Omnidirectional"
    
    @classmethod
    def from_string(cls, model_str: str) -> "ProjectionModel":
        """Convert a string to a ProjectionModel enum value."""
        # Try exact match first
        for model in cls:
            if model.value == model_str:
                return model
        # Try case-insensitive name match
        model_str_upper = model_str.upper().replace("+", "_").replace("-", "_")
        for model in cls:
            if model.name == model_str_upper:
                return model
        raise ValueError(
            f"Unknown projection model: {model_str}. "
            f"Valid options: {[m.value for m in cls]}"
        )


def deserialize_transform(data: dict[str, Any]) -> "BaseTransform":
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
) -> tuple["CameraProjection", "Transform"]:
    """
    Decompose a 3x4 projection matrix P into Intrinsic (CameraProjection) and Extrinsic (Transform) objects.
    
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
    return np.array([
        [0.0, -vz, vy],
        [vz, 0.0, -vx],
        [-vy, vx, 0.0]
    ], dtype=np.float64)


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
    def inverse() -> "BaseTransform":
        """
        Returns the mathematical inverse of the transformation.

        Returns:
            BaseTransform: The inverse transformation.
        """
        pass

    @abstractmethod
    def __mul__(self, other: "BaseTransform") -> "BaseTransform":
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
    def from_dict(cls, data: dict[str, Any]) -> "BaseTransform":
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
        rotation: "quaternion.quaternion | np.ndarray | list | tuple | Transform | None" = None,
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
    def from_matrix(cls, matrix: np.ndarray, dtype: np.dtype | None = None) -> "Transform":
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
        matrix = np.eye(4, dtype=self.dtype)
        matrix[:3, :3] = quaternion.as_rotation_matrix(self.rotation).astype(self.dtype)
        matrix[:3, 3] = self.translation.ravel()
        return matrix

    def inverse(self) -> "Transform":
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

        if isinstance(other, InverseCameraProjection):
             # Transform * InverseCameraProjection -> Matrix (4x3) unprojection
             # Handled effectively by MatrixTransform fallback below but let's be explicit if needed.
             # T * K_inv -> resulting math P_inv @ T_mat (wait, dimensions).
             # T (4x4) * K_inv (3x3)? K_inv usually treated as 4x4 or 3x4 padded?
             # InverseCameraProjection.as_matrix returns 3x3 or 4x3?
             pass

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
    def from_dict(cls, data: dict[str, Any]) -> "Transform":
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
    - Euler angles: Rotation.from_euler_angles(roll=0, pitch=0, yaw=0)
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
    def from_euler_angles(
        cls,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> "Rotation":
        """
        Create a Rotation from roll-pitch-yaw Euler angles.

        Uses the aerospace/robotics convention with intrinsic ZYX rotation order
        (yaw → pitch → roll). See class docstring for full convention details.

        Args:
            roll: Rotation about X-axis in radians (banking/roll)
            pitch: Rotation about Y-axis in radians (nose up/down)
            yaw: Rotation about Z-axis in radians (heading/yaw)

        Returns:
            Rotation: A rotation-only transform.

        Example:
            >>> # Aircraft heading north, 10° pitch up, wings level
            >>> attitude = tf.Rotation.from_euler_angles(roll=0, pitch=np.radians(10), yaw=0)
            >>>
            >>> # Robot turning left 45°
            >>> heading = tf.Rotation.from_euler_angles(yaw=np.pi/4)
        """
        # Use quaternion.from_euler_angles with ZYX ordering
        # (yaw about Z, pitch about Y, roll about X)
        return cls(rotation=quaternion.from_euler_angles(yaw, pitch, roll))

    def as_euler_angles(self) -> tuple[float, float, float]:
        """
        Extract roll-pitch-yaw Euler angles from the rotation.

        Uses the aerospace/robotics convention with intrinsic ZYX rotation order.
        See class docstring for full convention details.

        Returns:
            Tuple[float, float, float]: (roll, pitch, yaw) in radians.

        Warning:
            Euler angles have a singularity (gimbal lock) when pitch = ±90°.
            At this singularity, roll and yaw become coupled and the
            decomposition is not unique.

        Example:
            >>> rotation = tf.Rotation.from_euler_angles(roll=0.1, pitch=0.2, yaw=0.3)
            >>> roll, pitch, yaw = rotation.as_euler_angles()
            >>> print(
            ...     f"Roll: {np.degrees(roll):.1f}°, "
            ...     f"Pitch: {np.degrees(pitch):.1f}°, "
            ...     f"Yaw: {np.degrees(yaw):.1f}°"
            ... )
        """
        # quaternion.as_euler_angles returns (alpha, beta, gamma)
        # which corresponds to (yaw, pitch, roll) in our ZYX convention
        yaw, pitch, roll = quaternion.as_euler_angles(self.rotation)
        return (roll, pitch, yaw)


class Identity(Transform):
    """The identity transform (0 translation, identity rotation)."""

    def __init__(self):
        super().__init__()


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
        return self.matrix

    def inverse(self) -> "MatrixTransform":
        return MatrixTransform(np.linalg.inv(self.matrix), dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> "MatrixTransform":
        return MatrixTransform(self.matrix @ other.as_matrix(), dtype=self.dtype)

    def to_dict(self) -> dict[str, Any]:
        """Serialize transform to a JSON-compatible dictionary."""
        return {
            "type": "MatrixTransform",
            "matrix": self.matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatrixTransform":
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

    The apply() method projects 3D points to 2D pixel coordinates.

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

    def inverse(self) -> "InverseProjection":
        """
        Returns an InverseProjection representing P^-1.

        Note: The inverse projection requires depth information to actually
        unproject 2D points to 3D.
        """
        return InverseProjection(self.matrix, dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> BaseTransform:
        """Compose projection with another transform."""
        result_matrix = self.matrix @ other.as_matrix()
        
        # If we are composing with a generic MatrixTransform, we lose the strict
        # Projection semantics (e.g. if it becomes an Image->Image homography).
        # To remain safe and flexible, return MatrixTransform in this case.
        if isinstance(other, MatrixTransform):
            return MatrixTransform(result_matrix, dtype=self.dtype)
            
        # Otherwise (e.g. composed with Rigid Transform), we consider it a new Projection
        return Projection(result_matrix, dtype=self.dtype)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D pixel coordinates.

        Args:
            points: Nx3 array of 3D points.

        Returns:
            np.ndarray: Nx2 array of 2D pixel coordinates.
        """
        points = np.atleast_2d(points)
        if points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3, got {points.shape}")

        # Homogeneous coordinates
        hom_points = np.hstack([points, np.ones((points.shape[0], 1))])
        projected = (self.matrix[:3, :] @ hom_points.T).T

        # Normalize by w (perspective division)
        w = projected[:, 2:3]
        w = np.where(np.abs(w) < 1e-10, 1e-10, w)  # Avoid division by zero
        pixels = projected[:, :2] / w

        return pixels

    def to_dict(self) -> dict[str, Any]:
        """Serialize projection to a JSON-compatible dictionary."""
        return {
            "type": "Projection",
            "matrix": self.matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Projection":
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

    def inverse(self) -> "Projection":
        """Returns the original Projection."""
        return Projection(self._original_matrix, dtype=self.dtype)

    def __mul__(self, other: BaseTransform) -> "MatrixTransform":
        """Compose with another transform using pseudo-inverse."""
        return MatrixTransform(self.as_matrix() @ other.as_matrix(), dtype=self.dtype)

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

        # Solve for 3D points: P[:3,:3] * X + P[:3,3] = hom_pixels
        # X = inv(P[:3,:3]) * (hom_pixels - P[:3,3])
        inv_projection = np.linalg.inv(projection_3x3)
        points_3d = (inv_projection @ (hom_pixels - projection_t).T).T

        return points_3d

    def to_dict(self) -> dict[str, Any]:
        """Serialize inverse projection to a JSON-compatible dictionary."""
        return {
            "type": "InverseProjection",
            "original_matrix": self._original_matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InverseProjection":
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
            projection_model: Camera model (PINHOLE, PINHOLE_POLYNOMIAL, FISHEYE, etc.).
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
                self._projection_model = ProjectionModel.PINHOLE_POLYNOMIAL
            else:
                self._projection_model = ProjectionModel.PINHOLE
        elif isinstance(projection_model, str):
            self._projection_model = ProjectionModel.from_string(projection_model)
        else:
            self._projection_model = projection_model

        if intrinsic_matrix is None:
             raise ValueError("Must provide 'intrinsic_matrix' or alias 'K'")

        self._intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=dtype)
        if self._intrinsic_matrix.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {self._intrinsic_matrix.shape}")

        # The internal matrix is just K with a [0,0,0,1] row to make it 4x4 (Scale/Intrinsic transform)
        # We assume points are in Camera Frame already.
        matrix_4x4 = np.eye(4, dtype=dtype)
        matrix_4x4[:3, :3] = self._intrinsic_matrix
        
        super().__init__(matrix=matrix_4x4, dtype=dtype)

    @classmethod
    def from_intrinsics_and_transform(cls, *args, **kwargs):
        raise NotImplementedError(
            "CameraProjection is now Intrinsic-only. "
            "Use separate Transform objects for Extrinsics."
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
        """The camera projection model (PINHOLE, PINHOLE_POLYNOMIAL, FISHEYE, etc.)."""
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
    def from_dict(cls, data: dict[str, Any]) -> "CameraProjection":
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
        return (
            f"CameraProjection(fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f})"
        )

    def inverse(self) -> "InverseCameraProjection":
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
        return self._camera_projection.fx

    @property
    def fy(self) -> float:
        return self._camera_projection.fy

    @property
    def cx(self) -> float:
        return self._camera_projection.cx

    @property
    def cy(self) -> float:
        return self._camera_projection.cy

    @property
    def intrinsic_matrix(self) -> np.ndarray:
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
    def from_dict(cls, data: dict[str, Any]) -> "InverseCameraProjection":
        """Deserialize from dictionary."""
        cam_data = data["camera_projection"]
        if cam_data.get("type") != "CameraProjection":
            cam_data["type"] = "CameraProjection"
            
        camera_proj = CameraProjection.from_dict(cam_data)
        return cls(camera_proj)

    def __repr__(self) -> str:
        return f"InverseCameraProjection({self._camera_projection})"


class Pose:
    """
    A user-friendly wrapper around Transform for pose representation.

    Represents the pose of 'child_frame_id' relative to 'frame_id'.
    """

    def __init__(
        self,
        position: np.ndarray | list | tuple | None = None,
        orientation: quaternion.quaternion | np.ndarray | list | tuple | None = None,
        frame_id: str | None = None,
        child_frame_id: str | None = None,
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
        return self._transform.translation.flatten()

    @position.setter
    def position(self, value: np.ndarray | list | tuple):
        self._transform.translation = ensure_translation(value, self._transform.dtype)

    @property
    def orientation(self) -> quaternion.quaternion:
        return self._transform.rotation

    @orientation.setter
    def orientation(self, value: quaternion.quaternion | np.ndarray | list | tuple):
        self._transform.rotation = ensure_rotation(value, self._transform.dtype)

    def as_transform(self) -> Transform:
        """Returns the underlying Transform object."""
        return self._transform

    @classmethod
    def from_transform(
        cls, tf: Transform, frame_id: str | None = None, child_frame_id: str | None = None
    ) -> "Pose":
        """Creates a Pose from a Transform."""
        return cls(
            position=tf.translation.flatten(),
            orientation=tf.rotation,
            frame_id=frame_id,
            child_frame_id=child_frame_id,
        )

    def inverse(
        self, new_frame_id: str | None = None, new_child_frame_id: str | None = None
    ) -> "Pose":
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
            # Check frame consistency (optional, but good practice)
            # if self.child_frame_id and other.frame_id and self.child_frame_id != other.frame_id:
            #     warn("Frame mismatch in composition")
            new_child_frame_id = other.child_frame_id
            return Pose.from_transform(
                self._transform * other.as_transform(),
                frame_id=new_frame_id,
                child_frame_id=new_child_frame_id,
            )

        return Pose.from_transform(
            self._transform * other, frame_id=new_frame_id, child_frame_id=new_child_frame_id
        )

    def __mul__(self, other: Union["Pose", Transform]) -> "Pose":
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
        return self._transform.as_matrix()

    def __repr__(self) -> str:
        elements = [f"position={self.position!r}", f"orientation={self.orientation!r}"]
        if self.frame_id:
            elements.append(f"frame_id={self.frame_id!r}")
        if self.child_frame_id:
            elements.append(f"child_frame_id={self.child_frame_id!r}")
        return f"Pose({', '.join(elements)})"


def transform_points(transform: BaseTransform, points: np.ndarray) -> np.ndarray:
    """
    Applies a transformation to a set of 3D points.

    Args:
        transform: The transformation to apply.
        points: Nx3 array of points.

    Returns:
        np.ndarray: Nx3 array of transformed points.
    """
    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got {points.shape}")

    # Homogeneous coordinates
    hom_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_hom = (transform.as_matrix() @ hom_points.T).T

    return transformed_hom[:, :3]


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
        # We store "reference_frame" to indicate which node is the Target (Codomain) of the transform.
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

            # Store Direction: reference_frame->neighbor (opposite of reference_frame) is direction of transform?
            # No, add_transform(Source, Target, T). Edge Target->Source. reference_frame=Target.
            # T maps Source->Target.
            
            # If traversing reference_frame -> other (Target -> Source):
            # We want T_target_to_source.
            # Stored is T_source_to_target.
            # So traverse down (Ref->Source) = Inverse.
            
            # If traversing other -> reference_frame (Source -> Target):
            # We want T_source_to_target.
            # Stored is T_source_to_target.
            # So traverse up (Source->Ref) = Direct.
            
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

            # Track added edges for dependency mapping
            if not edge_data.get("is_cache", False):
                # Normalize edge key (always smaller, larger)
                edge_key = tuple(sorted([current_frame, next_frame]))
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
        cache_edge = tuple(sorted([source_frame, target_frame]))
        for added_edge in added_edges:
            if added_edge not in self._dependency_map:
                self._dependency_map[added_edge] = []
            self._dependency_map[added_edge].append(cache_edge)

    def _invalidate_caches_for_edge(self, frame_a: str, frame_b: str) -> None:
        """
        Remove all cached edges that depend on the edge (frame_a, frame_b).
        """
        edge_key = tuple(sorted([frame_a, frame_b]))
        if edge_key in self._dependency_map:
            for cache_u, cache_v in self._dependency_map[edge_key]:
                if self._graph.has_edge(cache_u, cache_v) and self._graph[cache_u][cache_v].get("is_cache", False):
                    self._graph.remove_edge(cache_u, cache_v)
            # Clear dependencies for this edge
            del self._dependency_map[edge_key]

    def is_projection_frame(self, frame_id: str) -> bool:
        """
        Check if a frame is a 2D projection frame (e.g., an Image frame).
        
        Rule: A frame is a projection frame if ALL edges connected to it treat it as a projection space.
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
                if reference_frame == image_frame: # Proj maps Neighbor->Image
                    K = transform.intrinsic_matrix
                    cam_frame = neighbor
            
            elif isinstance(transform, InverseCameraProjection):
                if reference_frame != image_frame: # InvProj maps Image->Neighbor
                    K = transform.intrinsic_matrix
                    cam_frame = neighbor
            
            if K is not None:
                return K, cam_frame

        raise ValueError(f"Frame '{image_frame}' is not a valid projection frame connected to a camera.")



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
        plane_distance: float
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
