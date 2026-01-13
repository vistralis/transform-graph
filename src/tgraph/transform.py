import numpy as np
import quaternion
import networkx as nx
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List, Tuple, Type


# -----------------------------------------------------------------------------
# Transform Registry for Serialization
# -----------------------------------------------------------------------------

# Global registry mapping type names to classes
_TRANSFORM_REGISTRY: Dict[str, Type["BaseTransform"]] = {}


def register_transform(cls: Type["BaseTransform"]) -> Type["BaseTransform"]:
    """
    Decorator to register a transform class for serialization.
    
    Usage:
        @register_transform
        class MyTransform(BaseTransform):
            ...
    """
    _TRANSFORM_REGISTRY[cls.__name__] = cls
    return cls


def serialize_transform(transform: "BaseTransform") -> Dict[str, Any]:
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

def as_euler_angles(q: quaternion.quaternion) -> Tuple[float, float, float]:
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


def deserialize_transform(data: Dict[str, Any]) -> "BaseTransform":
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

def ensure_translation(translation: Optional[Union[np.ndarray, list, tuple]], dtype: np.dtype) -> np.ndarray:
    """
    Ensures translation is a 3x1 numpy array of the specified dtype.
    Optimized to avoid copies if input already matches requirements.
    """
    if translation is None:
        return np.zeros((3, 1), dtype=dtype)
        
    if isinstance(translation, np.ndarray) and translation.shape == (3, 1) and translation.dtype == dtype:
        return translation

    array = np.array(translation, dtype=dtype)
    if array.size != 3:
        raise ValueError(f"Translation must have 3 elements, got {array.size}")
    return array.reshape(3, 1)

def ensure_rotation(rotation: Optional[Union[np.ndarray, list, tuple]], dtype: np.dtype) -> quaternion.quaternion:
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
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "BaseTransform":
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
        translation: Optional[Union[np.ndarray, list, tuple]] = None,
        rotation: Optional[Union[quaternion.quaternion, np.ndarray, list, tuple]] = None,
        dtype: np.dtype = np.float64,
    ):
        super().__init__(dtype=dtype)
        
        self.translation = ensure_translation(translation, self.dtype)
        self.rotation = ensure_rotation(rotation, self.dtype)


    @classmethod
    def from_matrix(cls, matrix: np.ndarray, dtype: Optional[np.dtype] = None) -> "Transform":
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
            new_translation = quaternion.rotate_vectors(self.rotation, other.translation.flatten()) + self.translation.flatten()
            return Transform(translation=new_translation, rotation=new_rotation, dtype=self.dtype)
        
        # Fallback to matrix multiplication
        return MatrixTransform(self.as_matrix() @ other.as_matrix(), dtype=self.dtype)

    def __repr__(self) -> str:
        return f"Transform(translation={self.translation.flatten()!r}, rotation={self.rotation!r})"

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "Transform":
        """Deserialize transform from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(
            translation=data["translation"],
            rotation=data["rotation"],
            dtype=dtype,
        )


class Translation(Transform):
    """A Transform with only translation (Identity rotation)."""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, translation: Optional[Union[np.ndarray, list, tuple]] = None,
                 **kwargs):
        if "translation" in kwargs:
            super().__init__(translation=kwargs["translation"])
        elif translation is not None:
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
    
    Euler Angle Convention (Roll-Pitch-Yaw / RPY):
    ------------------------------------------------
    This library uses the aerospace/robotics convention for Euler angles:
    
    - **Roll (φ)**: Rotation about the X-axis (forward axis)
    - **Pitch (θ)**: Rotation about the Y-axis (right/lateral axis)  
    - **Yaw (ψ)**: Rotation about the Z-axis (up/vertical axis)
    
    The rotations are applied in **intrinsic ZYX order** (yaw → pitch → roll),
    which is equivalent to **extrinsic XYZ order**. This means:
    1. First rotate about Z (yaw) - heading direction
    2. Then rotate about the new Y' (pitch) - nose up/down
    3. Finally rotate about the new X'' (roll) - banking
    
    This convention is standard in:
    - Aerospace (aircraft attitude: heading, pitch, bank)
    - Robotics (ROS REP-103, mobile robot navigation)
    - Marine vehicles (ship orientation)
    
    All angles are in **radians**.
    
    Example:
        >>> # Aircraft heading east (yaw=90°), level flight
        >>> attitude = tf.Rotation.from_euler_angles(roll=0, pitch=0, yaw=np.pi/2)
        >>> 
        >>> # Get angles back
        >>> roll, pitch, yaw = attitude.as_euler_angles()
    """
    def __init__(
        self,
        w: float = 1.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        q: Optional[quaternion.quaternion] = None,
        **kwargs
    ):
        if "rotation" in kwargs:
            super().__init__(rotation=kwargs["rotation"])
        elif q is not None:
            super().__init__(rotation=q)
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

    def as_euler_angles(self) -> Tuple[float, float, float]:
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
            >>> print(f"Roll: {np.degrees(roll):.1f}°, Pitch: {np.degrees(pitch):.1f}°, Yaw: {np.degrees(yaw):.1f}°")
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transform to a JSON-compatible dictionary."""
        return {
            "type": "MatrixTransform",
            "matrix": self.matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatrixTransform":
        """Deserialize transform from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(matrix=np.array(data["matrix"]), dtype=dtype)


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
        matrix: Union[np.ndarray, list],
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
    
    def __mul__(self, other: BaseTransform) -> "Projection":
        """Compose projection with another transform."""
        return Projection(self.matrix @ other.as_matrix(), dtype=self.dtype)
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize projection to a JSON-compatible dictionary."""
        return {
            "type": "Projection",
            "matrix": self.matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Projection":
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
        original_matrix: Union[np.ndarray, list],
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
        hom_pixels = np.column_stack([
            pixels[:, 0] * depths,
            pixels[:, 1] * depths,
            depths
        ])
        
        # Solve for 3D points: P[:3,:3] * X + P[:3,3] = hom_pixels
        # X = inv(P[:3,:3]) * (hom_pixels - P[:3,3])
        inv_projection = np.linalg.inv(projection_3x3)
        points_3d = (inv_projection @ (hom_pixels - projection_t).T).T
        
        return points_3d
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize inverse projection to a JSON-compatible dictionary."""
        return {
            "type": "InverseProjection",
            "original_matrix": self._original_matrix.tolist(),
            "dtype": np.dtype(self.dtype).name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InverseProjection":
        """Deserialize inverse projection from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(original_matrix=np.array(data["original_matrix"]), dtype=dtype)
    
    def __repr__(self) -> str:
        return f"InverseProjection(original_matrix_shape={self._original_matrix.shape})"


@register_transform
class CameraProjection(Projection):
    """
    A camera projection with explicit intrinsic and extrinsic parameters.
    
    The full projection is P = K @ [R | t] where:
        - K: 3x3 intrinsic matrix (focal length, principal point)
        - R: 3x3 rotation matrix (camera orientation)
        - t: 3x1 translation vector (camera position)
    
    Can be constructed from:
        - Explicit K, R, t parameters
        - A 3x4 or 4x4 projection matrix (decomposed via RQ decomposition)
    """
    
    def __init__(
        self,
        intrinsic_matrix: Optional[Union[np.ndarray, list]] = None,
        rotation_matrix: Optional[Union[np.ndarray, list]] = None,
        translation: Optional[Union[np.ndarray, list]] = None,
        matrix: Optional[Union[np.ndarray, list]] = None,
        dtype: np.dtype = np.float64,
    ):
        """
        Create a CameraProjection.
        
        Either provide (intrinsic_matrix, rotation_matrix, translation) OR matrix.
        
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            rotation_matrix: 3x3 rotation matrix R.
            translation: 3x1 translation vector t.
            matrix: 3x4 or 4x4 projection matrix (will be decomposed).
            dtype: Data type for matrices.
        """
        self._dtype = dtype
        
        if matrix is not None:
            # Decompose the provided matrix
            matrix = np.asarray(matrix, dtype=dtype)
            if matrix.shape == (4, 4):
                matrix = matrix[:3, :]
            elif matrix.shape != (3, 4):
                raise ValueError(f"Matrix must be 3x4 or 4x4, got {matrix.shape}")
            
            self._intrinsic_matrix, self._rotation_matrix, self._translation = \
                self._decompose_projection_matrix(matrix)
        else:
            # Use explicit parameters
            if intrinsic_matrix is None or rotation_matrix is None or translation is None:
                raise ValueError(
                    "Must provide either 'matrix' or all of "
                    "(intrinsic_matrix, rotation_matrix, translation)"
                )
            
            self._intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=dtype)
            self._rotation_matrix = np.asarray(rotation_matrix, dtype=dtype)
            self._translation = np.asarray(translation, dtype=dtype).reshape(3, 1)
            
            # Validate shapes
            if self._intrinsic_matrix.shape != (3, 3):
                raise ValueError(f"Intrinsic matrix must be 3x3, got {self._intrinsic_matrix.shape}")
            if self._rotation_matrix.shape != (3, 3):
                raise ValueError(f"Rotation matrix must be 3x3, got {self._rotation_matrix.shape}")
        
        # Compute the full projection matrix P = K @ [R | t]
        extrinsic = np.hstack([self._rotation_matrix, self._translation])
        projection_3x4 = self._intrinsic_matrix @ extrinsic
        
        # Initialize parent with 4x4 matrix
        super().__init__(matrix=projection_3x4, dtype=dtype)
    
    @classmethod
    def from_intrinsics_and_transform(
        cls,
        intrinsic_matrix: Union[np.ndarray, list],
        extrinsic_transform: "Transform",
        dtype: np.dtype = np.float64,
    ) -> "CameraProjection":
        """
        Create a CameraProjection from intrinsic matrix and a world→camera Transform.
        
        This is the recommended way to construct a CameraProjection when you have
        the camera's pose as a Transform object.
        
        Args:
            intrinsic_matrix: 3x3 camera intrinsic matrix K.
            extrinsic_transform: Transform from world to camera frame.
            dtype: Data type for matrices.
            
        Returns:
            CameraProjection: The complete projection.
            
        Example:
            >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
            >>> world_to_camera = tf.Transform(translation=[0, 0, 5])
            >>> camera = tf.CameraProjection.from_intrinsics_and_transform(K, world_to_camera)
        """
        # Extract R and t from the transform
        matrix_4x4 = extrinsic_transform.as_matrix()
        rotation_matrix = matrix_4x4[:3, :3]
        translation = matrix_4x4[:3, 3:4]
        
        return cls(
            intrinsic_matrix=intrinsic_matrix,
            rotation_matrix=rotation_matrix,
            translation=translation,
            dtype=dtype,
        )
    
    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """The 3x3 camera intrinsic matrix K."""
        return self._intrinsic_matrix
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """The 3x3 rotation matrix R (world → camera)."""
        return self._rotation_matrix
    
    @property
    def translation(self) -> np.ndarray:
        """The 3x1 translation vector t (world → camera)."""
        return self._translation
    
    @property
    def extrinsic_transform(self) -> "Transform":
        """
        The world → camera Transform (extrinsic parameters as a Transform).
        
        This Transform converts points from world coordinates to camera coordinates:
            X_camera = extrinsic_transform.apply(X_world)
        
        Returns:
            Transform: The SE(3) transform from world to camera frame.
        """
        rot_quat = quaternion.from_rotation_matrix(self._rotation_matrix)
        return Transform(
            translation=self._translation.flatten(),
            rotation=rot_quat,
            dtype=self.dtype,
        )
    
    @property
    def focal_length(self) -> Tuple[float, float]:
        """Focal lengths (fx, fy) from the intrinsic matrix."""
        return (float(self._intrinsic_matrix[0, 0]), float(self._intrinsic_matrix[1, 1]))
    
    @property
    def principal_point(self) -> Tuple[float, float]:
        """Principal point (cx, cy) from the intrinsic matrix."""
        return (float(self._intrinsic_matrix[0, 2]), float(self._intrinsic_matrix[1, 2]))
    
    def _decompose_projection_matrix(
        self, projection_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose a 3x4 projection matrix into K, R, t using RQ decomposition.
        
        P = K @ [R | t]
        P[:3,:3] = K @ R
        
        Returns:
            Tuple of (K, R, t)
        """
        from scipy.linalg import rq
        
        # Extract M = P[:3,:3] = K @ R
        M = projection_matrix[:3, :3]
        
        # RQ decomposition: M = R @ Q where R is upper triangular, Q is orthogonal
        # In our case: M = K @ R_rot, so we want K (upper tri) and R_rot (rotation)
        intrinsic_matrix, rotation_matrix = rq(M)
        
        # Ensure K has positive diagonal by flipping signs
        # For each negative diagonal element, flip the sign of that row of K
        # and the corresponding column of R
        for i in range(3):
            if intrinsic_matrix[i, i] < 0:
                intrinsic_matrix[i, :] *= -1
                rotation_matrix[:, i] *= -1
        
        # Normalize K so that K[2,2] = 1
        scale = intrinsic_matrix[2, 2]
        if abs(scale) > 1e-10:
            intrinsic_matrix = intrinsic_matrix / scale
        
        # Ensure R is a proper rotation (det = +1)
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix = -rotation_matrix
            intrinsic_matrix = -intrinsic_matrix
        
        # Compute translation: P[:3,3] = K @ t, so t = K^-1 @ P[:3,3]
        translation = np.linalg.solve(intrinsic_matrix, projection_matrix[:3, 3])
        translation = translation.reshape(3, 1)
        
        return intrinsic_matrix.astype(self._dtype), rotation_matrix.astype(self._dtype), translation.astype(self._dtype)
    
    def inverse(self) -> "InverseProjection":
        """
        Returns an InverseProjection for unprojection.
        
        Note: The structured K, R, t parameters are not preserved in the inverse.
        If you need to recover a CameraProjection from an InverseProjection,
        call inv_proj.inverse() to get a Projection, then create a new
        CameraProjection with CameraProjection(matrix=proj.as_matrix()).
        """
        return InverseProjection(self.matrix, dtype=self.dtype)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize camera projection to a JSON-compatible dictionary."""
        return {
            "type": "CameraProjection",
            "intrinsic_matrix": self._intrinsic_matrix.tolist(),
            "rotation_matrix": self._rotation_matrix.tolist(),
            "translation": self._translation.flatten().tolist(),
            "dtype": np.dtype(self.dtype).name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraProjection":
        """Deserialize camera projection from a dictionary."""
        dtype = np.dtype(data.get("dtype", "float64"))
        return cls(
            intrinsic_matrix=np.array(data["intrinsic_matrix"]),
            rotation_matrix=np.array(data["rotation_matrix"]),
            translation=np.array(data["translation"]),
            dtype=dtype,
        )
    
    def __repr__(self) -> str:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        return f"CameraProjection(fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f})"


class Pose:
    """
    A user-friendly wrapper around Transform for pose representation.
    
    Represents the pose of 'child_frame_id' relative to 'frame_id'.
    """
    def __init__(
        self, 
        position: Optional[Union[np.ndarray, list, tuple]] = None, 
        orientation: Optional[Union[quaternion.quaternion, np.ndarray, list, tuple]] = None,
        frame_id: Optional[str] = None,
        child_frame_id: Optional[str] = None,
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
    def position(self, value: Union[np.ndarray, list, tuple]):
        self._transform.translation = ensure_translation(value, self._transform.dtype)

    @property
    def orientation(self) -> quaternion.quaternion:
        return self._transform.rotation
    
    @orientation.setter
    def orientation(self, value: Union[quaternion.quaternion, np.ndarray, list, tuple]):
        self._transform.rotation = ensure_rotation(value, self._transform.dtype)

    def as_transform(self) -> Transform:
        """Returns the underlying Transform object."""
        return self._transform

    @classmethod
    def from_transform(
        cls, 
        tf: Transform, 
        frame_id: Optional[str] = None, 
        child_frame_id: Optional[str] = None
    ) -> "Pose":
        """Creates a Pose from a Transform."""
        return cls(
            position=tf.translation.flatten(), 
            orientation=tf.rotation,
            frame_id=frame_id,
            child_frame_id=child_frame_id
        )

    def inverse(
        self, 
        new_frame_id: Optional[str] = None, 
        new_child_frame_id: Optional[str] = None
    ) -> "Pose":
        """
        Returns the inverse pose.
        
        By default, swaps frame_id and child_frame_id:
        Inverse(T_A->B) = T_B->A
        """
        # Default behavior: swap frames
        target_frame_id = new_frame_id if new_frame_id is not None else self.child_frame_id
        target_child_frame_id = new_child_frame_id if new_child_frame_id is not None else self.frame_id
        
        return Pose.from_transform(
            self._transform.inverse(), 
            frame_id=target_frame_id,
            child_frame_id=target_child_frame_id
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
                child_frame_id=new_child_frame_id
            )
            
        return Pose.from_transform(
            self._transform * other, 
            frame_id=new_frame_id,
            child_frame_id=new_child_frame_id
        )

    def __mul__(self, other: Union["Pose", Transform]) -> "Pose":
        return self.compose(other)

    def to_list(self) -> List[float]:
        """Returns [px, py, pz, qw, qx, qy, qz]"""
        q = self.orientation
        p = self.position
        return [float(p[0]), float(p[1]), float(p[2]), float(q.w), float(q.x), float(q.y), float(q.z)]
    
    def to_matrix(self) -> np.ndarray:
        return self._transform.as_matrix()

    def __repr__(self) -> str:
        elements = [
            f"position={self.position!r}",
            f"orientation={self.orientation!r}"
        ]
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


def get_basis_vectors(transform: BaseTransform, length: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        self._dependency_map: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}

    @property
    def graph(self) -> nx.Graph:
        """Returns the internal NetworkX graph (read-only)."""
        return self._graph

    @property
    def frames(self) -> List[str]:
        """Returns list of all frame IDs in the graph."""
        return list(self._graph.nodes())

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Returns list of all added edges as (parent, child) tuples."""
        result = []
        for u, v, data in self._graph.edges(data=True):
            if not data.get("is_cache", False):
                parent = data["parent"]
                child = v if parent == u else u
                result.append((parent, child))
        return result

    def has_frame(self, frame_id: str) -> bool:
        """Check if a frame exists in the graph."""
        return frame_id in self._graph

    def has_transform(self, source_frame: str, target_frame: str) -> bool:
        """Check if a direct transform (edge) exists between two frames."""
        return self._graph.has_edge(source_frame, target_frame)

    def add_transform(
        self,
        parent_frame: str,
        child_frame: str,
        transform: BaseTransform,
    ) -> None:
        """
        Add a transform between two frames.
        
        Args:
            parent_frame: The source/parent frame ID.
            child_frame: The target/child frame ID.
            transform: The transform from parent to child (T_parent_to_child).
        
        Raises:
            ValueError: If an edge already exists between these frames.
        """
        if self._graph.has_edge(parent_frame, child_frame):
            raise ValueError(
                f"Transform between '{parent_frame}' and '{child_frame}' already exists. "
                "Use update_transform() to modify it."
            )
        
        self._graph.add_edge(
            parent_frame,
            child_frame,
            transform=transform,
            parent=parent_frame,
            is_cache=False,
            weight=self.ADDED_EDGE_WEIGHT,
        )

    def update_transform(
        self,
        parent_frame: str,
        child_frame: str,
        transform: BaseTransform,
    ) -> None:
        """
        Update an existing transform between two frames.
        
        Automatically invalidates any cached shortcuts that depend on this edge.
        
        Args:
            parent_frame: The source/parent frame ID.
            child_frame: The target/child frame ID.
            transform: The new transform from parent to child.
        
        Raises:
            ValueError: If no edge exists between these frames.
        """
        if not self._graph.has_edge(parent_frame, child_frame):
            raise ValueError(
                f"No transform between '{parent_frame}' and '{child_frame}'. "
                "Use add_transform() to create it."
            )
        
        # Invalidate dependent caches
        self._invalidate_caches_for_edge(parent_frame, child_frame)
        
        # Update the transform
        self._graph[parent_frame][child_frame]["transform"] = transform
        self._graph[parent_frame][child_frame]["parent"] = parent_frame

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
            parent = edge_data["parent"]
            
            # If we're going against the natural direction, invert
            if parent == target_frame:
                return transform.inverse()
            return transform
        
        # Find shortest path
        try:
            path = nx.shortest_path(
                self._graph, source_frame, target_frame, weight="weight"
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"No path from '{source_frame}' to '{target_frame}'."
            )
        
        # Compose transforms along path
        composed_transform = Identity()
        added_edges = []
        
        for i in range(len(path) - 1):
            current_frame = path[i]
            next_frame = path[i + 1]
            
            edge_data = self._graph[current_frame][next_frame]
            transform = edge_data["transform"]
            parent = edge_data["parent"]
            
            # Track added edges for dependency mapping
            if not edge_data.get("is_cache", False):
                # Normalize edge key (always smaller, larger)
                edge_key = tuple(sorted([current_frame, next_frame]))
                added_edges.append(edge_key)
            
            # If we're going against the natural direction, invert
            if parent == next_frame:
                step_transform = transform.inverse()
            else:
                step_transform = transform
            
            composed_transform = composed_transform * step_transform
        
        # Cache the result as a shortcut edge
        self._add_cache_edge(source_frame, target_frame, composed_transform, added_edges)
        
        return composed_transform

    def _add_cache_edge(
        self,
        source_frame: str,
        target_frame: str,
        transform: BaseTransform,
        added_edges: List[Tuple[str, str]],
    ) -> None:
        """Add a cached shortcut edge and register dependencies."""
        self._graph.add_edge(
            source_frame,
            target_frame,
            transform=transform,
            parent=source_frame,
            is_cache=True,
            weight=self.CACHED_EDGE_WEIGHT,
        )
        
        # Register this cache edge as dependent on all added edges in the path
        cache_edge = tuple(sorted([source_frame, target_frame]))
        for added_edge in added_edges:
            if added_edge not in self._dependency_map:
                self._dependency_map[added_edge] = []
            if cache_edge not in self._dependency_map[added_edge]:
                self._dependency_map[added_edge].append(cache_edge)

    def _invalidate_caches_for_edge(self, frame_a: str, frame_b: str) -> None:
        """Invalidate all cache edges that depend on an added edge."""
        edge_key = tuple(sorted([frame_a, frame_b]))
        
        if edge_key not in self._dependency_map:
            return
        
        # Remove all dependent cache edges
        cache_edges_to_remove = self._dependency_map[edge_key].copy()
        for cache_edge in cache_edges_to_remove:
            cache_u, cache_v = cache_edge
            if self._graph.has_edge(cache_u, cache_v):
                edge_data = self._graph[cache_u][cache_v]
                if edge_data.get("is_cache", False):
                    self._graph.remove_edge(cache_u, cache_v)
        
        # Clear the dependency list
        del self._dependency_map[edge_key]
        
        # Clean up other dependency entries that referenced removed cache edges
        for added_edge, cache_list in list(self._dependency_map.items()):
            self._dependency_map[added_edge] = [
                ce for ce in cache_list if ce not in cache_edges_to_remove
            ]

    def clear_cache(self) -> None:
        """Remove all cached shortcut edges, forcing recomputation on next query."""
        cache_edges = [
            (u, v) for u, v, data in self._graph.edges(data=True)
            if data.get("is_cache", False)
        ]
        
        for u, v in cache_edges:
            self._graph.remove_edge(u, v)
        
        self._dependency_map.clear()

    def get_connected_nodes(self, node: str) -> List[str]:
        """
        Get all nodes connected to the given node (connected component).
        
        Args:
            node: The node to start from.
            
        Returns:
            List[str]: List of connected node IDs (including the start node).
            
        Raises:
            ValueError: If the node is not in the graph.
        """
        if node not in self._graph:
            raise ValueError(f"Frame '{node}' not found in graph.")
        return list(nx.node_connected_component(self._graph, node))

    def get_connected_components(self) -> List[List[str]]:
        """
        Get all connected components in the graph.
        
        Returns:
            List[List[str]]: List of lists, where each inner list contains
                             frame IDs belonging to a connected component.
        """
        return [list(c) for c in nx.connected_components(self._graph)]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the graph to a JSON-compatible dictionary.
        
        Only ground-truth edges are serialized; cache edges are transient.
        """
        edges = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("is_cache", False):
                continue  # Skip cache edges
            
            edges.append({
                "parent": data["parent"],
                "child": v if data["parent"] == u else u,
                "transform": data["transform"].to_dict(),
            })
        
        return {
            "version": "1.0",
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformGraph":
        """
        Deserialize a graph from a dictionary.
        
        Args:
            data: Dictionary previously created by to_dict().
            
        Returns:
            TransformGraph: Reconstructed graph.
        """
        graph = cls()
        
        for edge_data in data.get("edges", []):
            parent = edge_data["parent"]
            child = edge_data["child"]
            transform = deserialize_transform(edge_data["transform"])
            graph.add_transform(parent, child, transform)
        
        return graph

    def __repr__(self) -> str:
        num_frames = self._graph.number_of_nodes()
        num_edges = sum(
            1 for _, _, d in self._graph.edges(data=True)
            if not d.get("is_cache", False)
        )
        num_cached = sum(
            1 for _, _, d in self._graph.edges(data=True)
            if d.get("is_cache", False)
        )
        return f"TransformGraph(frames={num_frames}, edges={num_edges}, cached={num_cached})"
