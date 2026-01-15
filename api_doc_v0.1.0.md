# tgraph API Reference (v0.1.0)

This document was automatically generated from the source code docstrings.

## Module: `tgraph.transform`

### Classes

#### `BaseTransform`

```python
class BaseTransform(dtype: numpy.dtype = <class 'numpy.float64'>)
```

Abstract interface for all spatial transformations.

**Methods:**

*   `__init__(self, dtype: numpy.dtype = <class 'numpy.float64'>)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Apply the transform to 3D vectors (Nx3 or Nx4).
    > 
    > Args:
    >     vector: Nx3 or Nx4 array of vectors.
    >             - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform, but checking subclass logic).
    >               Standard BaseTransform behavior:
    >               Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
    >             - If Nx4: Treated as homogeneous vectors.
    >               Multiply -> Return Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix() -> numpy.ndarray`
    > Returns the 4x4 homogeneous representation of the transform.
    > 
    > Returns:
    >     np.ndarray: 4x4 matrix of the transform's dtype.

*   `from_dict(data: dict[str, typing.Any]) -> 'BaseTransform'`
    > Deserialize a transform from a dictionary.
    > 
    > Args:
    >     data: Dictionary previously created by to_dict().
    > 
    > Returns:
    >     BaseTransform: The deserialized transform instance.

*   `inverse() -> 'BaseTransform'`
    > Returns the mathematical inverse of the transformation.
    > 
    > Returns:
    >     BaseTransform: The inverse transformation.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize the transform to a JSON-compatible dictionary.
    > 
    > The dictionary MUST include a "type" key with the class name
    > to enable proper deserialization.
    > 
    > Returns:
    >     Dict[str, Any]: JSON-compatible dictionary representation.

#### `CameraProjection`

```python
class CameraProjection(intrinsic_matrix: numpy.ndarray | list | None = None, dist_coeffs: list | numpy.ndarray | None = None, projection_model: tgraph.transform.ProjectionModel | str | None = None, image_size: tuple[int, int] | None = None, dtype: numpy.dtype = <class 'numpy.float64'>, K: numpy.ndarray | list | None = None, D: list | numpy.ndarray | None = None)
```

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

**Methods:**

*   `__init__(self, intrinsic_matrix: numpy.ndarray | list | None = None, dist_coeffs: list | numpy.ndarray | None = None, projection_model: tgraph.transform.ProjectionModel | str | None = None, image_size: tuple[int, int] | None = None, dtype: numpy.dtype = <class 'numpy.float64'>, K: numpy.ndarray | list | None = None, D: list | numpy.ndarray | None = None)`
    > Create a CameraProjection (Intrinsic-only).
    > 
    > Args:
    >     intrinsic_matrix: 3x3 camera intrinsic matrix K.
    >     dist_coeffs: Distortion coefficients (OpenCV ordering: k1,k2,p1,p2,k3,...).
    >     projection_model: Camera model (PINHOLE, PINHOLE_POLYNOMIAL, FISHEYE, etc.).
    >     image_size: Image dimensions (width, height) in pixels.
    >     dtype: Data type for matrices.
    >     K: Alias for intrinsic_matrix.
    >     D: Alias for dist_coeffs.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Project 3D vectors to 2D pixel coordinates.
    > 
    > Args:
    >     vector: Nx3 (points) or Nx4 (homogeneous) array.
    > 
    > Returns:
    >     np.ndarray: Nx2 pixel coordinates.

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 projection matrix.

*   `as_matrix_3x4(self) -> numpy.ndarray`
    > Returns the 3x4 projection matrix (top 3 rows).

*   `from_dict(data: dict[str, typing.Any]) -> 'CameraProjection'`
    > Deserialize camera projection from a dictionary.

*   `from_intrinsics_and_transform(*args, **kwargs)`

*   `inverse(self) -> 'InverseCameraProjection'`
    > Returns an InverseCameraProjection for unprojection.
    > 
    > Preserves the camera parameters (intrinsics) in the inverse object.

*   `project_points(self, points: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Project 3D points (Nx3 or Nx4) to 2D pixel coordinates.
    > alias for apply(points).
    > 
    > Args:
    >      points: Nx3 or Nx4 array of points.
    > 
    > Returns:
    >      np.ndarray: Nx2 pixel coordinates.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize camera projection to a JSON-compatible dictionary.

#### `Identity`

```python
class Identity()
```

The identity transform (0 translation, identity rotation).

**Methods:**

*   `__init__(self)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Apply the transform to 3D vectors (Nx3 or Nx4).
    > 
    > Args:
    >     vector: Nx3 or Nx4 array of vectors.
    >             - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform, but checking subclass logic).
    >               Standard BaseTransform behavior:
    >               Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
    >             - If Nx4: Treated as homogeneous vectors.
    >               Multiply -> Return Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 homogeneous representation of the transform.
    > 
    > Returns:
    >     np.ndarray: 4x4 matrix of the transform's dtype.

*   `from_dict(data: dict[str, typing.Any]) -> 'Transform'`
    > Deserialize transform from a dictionary.

*   `from_matrix(matrix: numpy.ndarray, dtype: numpy.dtype | None = None) -> 'Transform'`
    > Creates a Transform from a 4x4 matrix.

*   `inverse(self) -> 'Transform'`
    > Returns the mathematical inverse of the transformation.
    > 
    > Returns:
    >     BaseTransform: The inverse transformation.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize transform to a JSON-compatible dictionary.

#### `InverseCameraProjection`

```python
class InverseCameraProjection(camera_projection: tgraph.transform.CameraProjection)
```

The inverse of a CameraProjection.

Preserves the internal CameraProjection instance to maintain access to
intrinsics (K) and distortion coefficients.

**Methods:**

*   `__init__(self, camera_projection: tgraph.transform.CameraProjection)`
    > Create an InverseCameraProjection.
    > 
    > Args:
    >     camera_projection: The original CameraProjection instance.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Unproject 2D/3D vectors using pseudo-inverse.
    > 
    > Args:
    >     vector: Nx2 (pixels), Nx3 (homogenous pixels), or Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix(self) -> numpy.ndarray`
    > Returns a pseudo-inverse matrix for composition purposes.
    > 
    > Warning: This is the Moore-Penrose pseudo-inverse and may not
    > produce geometrically meaningful results for all operations.

*   `from_dict(data: dict[str, typing.Any]) -> 'InverseCameraProjection'`
    > Deserialize from dictionary.

*   `inverse(self) -> tgraph.transform.CameraProjection`
    > Returns the original CameraProjection.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize to dictionary using the contained CameraProjection.

*   `unproject(self, pixels: numpy.ndarray, depths: numpy.ndarray) -> numpy.ndarray`
    > Unproject 2D pixels to 3D points using depth values.
    > 
    > Args:
    >     pixels: Nx2 array of 2D pixel coordinates.
    >     depths: N array of depth values (Z coordinate in camera frame).
    > 
    > Returns:
    >     np.ndarray: Nx3 array of 3D points.
    > 
    > Note: This assumes a standard pinhole camera model where the
    > projection matrix can be decomposed into K[R|t] form.

#### `InverseProjection`

```python
class InverseProjection(original_matrix: numpy.ndarray | list, dtype: numpy.dtype = <class 'numpy.float64'>)
```

Represents the conceptual inverse of a Projection (P^-1).

This class tracks that an inverse operation was requested, but actual
unprojection requires depth information. Use unproject() with depth values
to convert 2D pixels back to 3D points.

Useful for:
- Tracking transform logic in a graph
- Composing with other transforms
- Unprojecting when depth is available

**Methods:**

*   `__init__(self, original_matrix: numpy.ndarray | list, dtype: numpy.dtype = <class 'numpy.float64'>)`
    > Create an InverseProjection from the original projection matrix.
    > 
    > Args:
    >     original_matrix: The original 3x4 or 4x4 projection matrix.
    >     dtype: Data type for the matrix.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Unproject 2D/3D vectors using pseudo-inverse.
    > 
    > Args:
    >     vector: Nx2 (pixels), Nx3 (homogenous pixels), or Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix(self) -> numpy.ndarray`
    > Returns a pseudo-inverse matrix for composition purposes.
    > 
    > Warning: This is the Moore-Penrose pseudo-inverse and may not
    > produce geometrically meaningful results for all operations.

*   `from_dict(data: dict[str, typing.Any]) -> 'InverseProjection'`
    > Deserialize inverse projection from a dictionary.

*   `inverse(self) -> 'Projection'`
    > Returns the original Projection.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize inverse projection to a JSON-compatible dictionary.

*   `unproject(self, pixels: numpy.ndarray, depths: numpy.ndarray) -> numpy.ndarray`
    > Unproject 2D pixels to 3D points using depth values.
    > 
    > Args:
    >     pixels: Nx2 array of 2D pixel coordinates.
    >     depths: N array of depth values (Z coordinate in camera frame).
    > 
    > Returns:
    >     np.ndarray: Nx3 array of 3D points.
    > 
    > Note: This assumes a standard pinhole camera model where the
    > projection matrix can be decomposed into K[R|t] form.

#### `MatrixTransform`

```python
class MatrixTransform(matrix: numpy.ndarray, dtype: numpy.dtype = <class 'numpy.float64'>)
```

A generic transform held as a raw 4x4 matrix.
Used when SE(3) structure is lost or not applicable.

**Methods:**

*   `__init__(self, matrix: numpy.ndarray, dtype: numpy.dtype = <class 'numpy.float64'>)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Apply the transform to 3D vectors (Nx3 or Nx4).
    > 
    > Args:
    >     vector: Nx3 or Nx4 array of vectors.
    >             - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform, but checking subclass logic).
    >               Standard BaseTransform behavior:
    >               Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
    >             - If Nx4: Treated as homogeneous vectors.
    >               Multiply -> Return Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 homogeneous representation of the transform.
    > 
    > Returns:
    >     np.ndarray: 4x4 matrix of the transform's dtype.

*   `from_dict(data: dict[str, typing.Any]) -> 'MatrixTransform'`
    > Deserialize transform from a dictionary.

*   `inverse(self) -> 'MatrixTransform'`
    > Returns the mathematical inverse of the transformation.
    > 
    > Returns:
    >     BaseTransform: The inverse transformation.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize transform to a JSON-compatible dictionary.

#### `Pose`

```python
class Pose(position: numpy.ndarray | list | tuple | None = None, orientation: quaternion.quaternion | numpy.ndarray | list | tuple | None = None, frame_id: str | None = None, child_frame_id: str | None = None)
```

A user-friendly wrapper around Transform for pose representation.

Represents the pose of 'child_frame_id' relative to 'frame_id'.

**Methods:**

*   `__init__(self, position: numpy.ndarray | list | tuple | None = None, orientation: quaternion.quaternion | numpy.ndarray | list | tuple | None = None, frame_id: str | None = None, child_frame_id: str | None = None)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `as_transform(self) -> tgraph.transform.Transform`
    > Returns the underlying Transform object.

*   `compose(self, other)`
    > Returns self * other

*   `from_transform(tf: tgraph.transform.Transform, frame_id: str | None = None, child_frame_id: str | None = None) -> 'Pose'`
    > Creates a Pose from a Transform.

*   `inverse(self, new_frame_id: str | None = None, new_child_frame_id: str | None = None) -> 'Pose'`
    > Returns the inverse pose.
    > 
    > By default, swaps frame_id and child_frame_id:
    > Inverse(T_A->B) = T_B->A

*   `to_list(self) -> list[float]`
    > Returns [px, py, pz, qw, qx, qy, qz]

*   `to_matrix(self) -> numpy.ndarray`

#### `Projection`

```python
class Projection(matrix: numpy.ndarray | list, dtype: numpy.dtype = <class 'numpy.float64'>)
```

A 3D to 2D projection transformation.

Stores a projection matrix P that maps 3D homogeneous points to 2D.
Internally stored as 4x4 matrix with bottom row [0, 0, 0, 1] for compatibility.

The apply() method projects 3D points to 2D pixel coordinates.

Note: Projections are generally non-invertible. The inverse() method returns
an InverseProjection which represents the conceptual inverse but requires
additional depth information to actually unproject points.

**Methods:**

*   `__init__(self, matrix: numpy.ndarray | list, dtype: numpy.dtype = <class 'numpy.float64'>)`
    > Create a Projection from a 3x4 or 4x4 matrix.
    > 
    > Args:
    >     matrix: 3x4 or 4x4 projection matrix.
    >     dtype: Data type for the matrix.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Project 3D vectors to 2D pixel coordinates.
    > 
    > Args:
    >     vector: Nx3 (points) or Nx4 (homogeneous) array.
    > 
    > Returns:
    >     np.ndarray: Nx2 pixel coordinates.

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 projection matrix.

*   `as_matrix_3x4(self) -> numpy.ndarray`
    > Returns the 3x4 projection matrix (top 3 rows).

*   `from_dict(data: dict[str, typing.Any]) -> 'Projection'`
    > Deserialize projection from a dictionary.

*   `inverse(self) -> 'InverseProjection'`
    > Returns an InverseProjection representing P^-1.
    > 
    > Note: The inverse projection requires depth information to actually
    > unproject 2D points to 3D.

*   `project_points(self, points: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Project 3D points (Nx3 or Nx4) to 2D pixel coordinates.
    > alias for apply(points).
    > 
    > Args:
    >      points: Nx3 or Nx4 array of points.
    > 
    > Returns:
    >      np.ndarray: Nx2 pixel coordinates.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize projection to a JSON-compatible dictionary.

#### `ProjectionModel`

```python
class ProjectionModel(*values)
```

Enum for different camera projection models.

Supported models:
- PINHOLE: Standard pinhole camera model (no distortion)
- PINHOLE_POLYNOMIAL: Pinhole with polynomial radial/tangential distortion (OpenCV-style)
- FISHEYE: Fisheye camera model
- OMNIDIRECTIONAL: Omnidirectional camera model

#### `Rotation`

```python
class Rotation(w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, rotation: quaternion.quaternion | numpy.ndarray | list | tuple | None = None)
```

A Transform with only rotation (Zero translation).

Supports multiple construction patterns:
- Quaternion components: Rotation(w=1, x=0, y=0, z=0)
- Quaternion object: Rotation(rotation=q)
- Euler angles: Rotation.from_euler_angles(roll=0, pitch=0, yaw=0)
...

**Methods:**

*   `__init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0, rotation: quaternion.quaternion | numpy.ndarray | list | tuple | None = None)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Apply the transform to 3D vectors (Nx3 or Nx4).
    > 
    > Args:
    >     vector: Nx3 or Nx4 array of vectors.
    >             - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform, but checking subclass logic).
    >               Standard BaseTransform behavior:
    >               Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
    >             - If Nx4: Treated as homogeneous vectors.
    >               Multiply -> Return Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_euler_angles(self) -> tuple[float, float, float]`
    > Extract roll-pitch-yaw Euler angles from the rotation.
    > 
    > Uses the aerospace/robotics convention with intrinsic ZYX rotation order.
    > See class docstring for full convention details.
    > 
    > Returns:
    >     Tuple[float, float, float]: (roll, pitch, yaw) in radians.
    > 
    > Warning:
    >     Euler angles have a singularity (gimbal lock) when pitch = ±90°.
    >     At this singularity, roll and yaw become coupled and the
    >     decomposition is not unique.
    > 
    > Example:
    >     >>> rotation = tf.Rotation.from_euler_angles(roll=0.1, pitch=0.2, yaw=0.3)
    >     >>> roll, pitch, yaw = rotation.as_euler_angles()
    >     >>> print(
    >     ...     f"Roll: {np.degrees(roll):.1f}°, "
    >     ...     f"Pitch: {np.degrees(pitch):.1f}°, "
    >     ...     f"Yaw: {np.degrees(yaw):.1f}°"
    >     ... )

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 homogeneous representation of the transform.
    > 
    > Returns:
    >     np.ndarray: 4x4 matrix of the transform's dtype.

*   `from_dict(data: dict[str, typing.Any]) -> 'Transform'`
    > Deserialize transform from a dictionary.

*   `from_euler_angles(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> 'Rotation'`
    > Create a Rotation from roll-pitch-yaw Euler angles.
    > 
    > Uses the aerospace/robotics convention with intrinsic ZYX rotation order
    > (yaw → pitch → roll). See class docstring for full convention details.
    > 
    > Args:
    >     roll: Rotation about X-axis in radians (banking/roll)
    >     pitch: Rotation about Y-axis in radians (nose up/down)
    >     yaw: Rotation about Z-axis in radians (heading/yaw)
    > 
    > Returns:
    >     Rotation: A rotation-only transform.
    > 
    > Example:
    >     >>> # Aircraft heading north, 10° pitch up, wings level
    >     >>> attitude = tf.Rotation.from_euler_angles(roll=0, pitch=np.radians(10), yaw=0)
    >     >>>
    >     >>> # Robot turning left 45°
    >     >>> heading = tf.Rotation.from_euler_angles(yaw=np.pi/4)

*   `from_matrix(matrix: numpy.ndarray, dtype: numpy.dtype | None = None) -> 'Transform'`
    > Creates a Transform from a 4x4 matrix.

*   `inverse(self) -> 'Transform'`
    > Returns the mathematical inverse of the transformation.
    > 
    > Returns:
    >     BaseTransform: The inverse transformation.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize transform to a JSON-compatible dictionary.

#### `Transform`

```python
class Transform(translation: numpy.ndarray | list | tuple | None = None, rotation: 'quaternion.quaternion | np.ndarray | list | tuple | Transform | None' = None, dtype: numpy.dtype = <class 'numpy.float64'>)
```

Standard SE(3) rigid body transformation.
Consists of a translation (3x1) and a rotation (quaternion).

**Methods:**

*   `__init__(self, translation: numpy.ndarray | list | tuple | None = None, rotation: 'quaternion.quaternion | np.ndarray | list | tuple | Transform | None' = None, dtype: numpy.dtype = <class 'numpy.float64'>)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Apply the transform to 3D vectors (Nx3 or Nx4).
    > 
    > Args:
    >     vector: Nx3 or Nx4 array of vectors.
    >             - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform, but checking subclass logic).
    >               Standard BaseTransform behavior:
    >               Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
    >             - If Nx4: Treated as homogeneous vectors.
    >               Multiply -> Return Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 homogeneous representation of the transform.
    > 
    > Returns:
    >     np.ndarray: 4x4 matrix of the transform's dtype.

*   `from_dict(data: dict[str, typing.Any]) -> 'Transform'`
    > Deserialize transform from a dictionary.

*   `from_matrix(matrix: numpy.ndarray, dtype: numpy.dtype | None = None) -> 'Transform'`
    > Creates a Transform from a 4x4 matrix.

*   `inverse(self) -> 'Transform'`
    > Returns the mathematical inverse of the transformation.
    > 
    > Returns:
    >     BaseTransform: The inverse transformation.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize transform to a JSON-compatible dictionary.

#### `TransformGraph`

```python
class TransformGraph()
```

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

**Methods:**

*   `__init__(self)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `add_transform(self, source_frame: str, target_frame: str, transform: tgraph.transform.BaseTransform) -> None`
    > Add a transform between two frames.
    > 
    > API: add_transform(source, target, transform)
    > - SOURCE: The domain frame (where vectors start).
    > - TARGET: The codomain/reference frame (where vectors end).
    > - TRANSFORM: Source→Target operator.
    > 
    > The transform maps Source coordinates to Target coordinates:
    > P_target = transform * P_source
    > 
    > Args:
    >     source_frame: The source/domain frame ID.
    >     target_frame: The target/reference frame ID.
    >     transform: The transform from source to target.
    > 
    > Raises:
    >     ValueError: If an edge already exists between these frames.

*   `clear_cache(self) -> None`
    > Clear all cached shortcut transforms.
    > 
    > Removes edges marked with is_cache=True.

*   `estimate_skew(intrinsic_matrix: numpy.ndarray) -> float`
    > Estimate the skew parameter from an intrinsic matrix K.
    > 
    > K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]
    > Returns s.

*   `from_dict(data: dict[str, typing.Any]) -> 'TransformGraph'`
    > Deserialize a graph from a dictionary.
    > 
    > Args:
    >     data: Dictionary produced by to_dict().
    >     
    > Returns:
    >     New TransformGraph instance.

*   `get_connected_components(self) -> list[set[str]]`
    > Get all connected components in the graph.
    > 
    > Returns:
    >     List of sets, where each set contains frame IDs of a connected component.

*   `get_connected_nodes(self, frame_id: str) -> set[str]`
    > Get the set of all nodes connected to the given frame (its connected component).
    > 
    > Args:
    >     frame_id: The frame to start searching from.
    >     
    > Returns:
    >     Set of connected frame IDs.
    >     
    > Raises:
    >     ValueError: If frame_id is not in the graph.

*   `get_essential_matrix(self, image_frame_1: str, image_frame_2: str) -> numpy.ndarray`
    > Compute the Essential Matrix E between two image frames.
    > 
    > E = [t]_x R
    > where R, t describe method to transform points from Camera 1 to Camera 2.
    > X2 = R X1 + t.

*   `get_fundamental_matrix(self, image_frame_1: str, image_frame_2: str) -> numpy.ndarray`
    > Compute the Fundamental Matrix F between two image frames.
    > 
    > F = K2^-T E K1^-1
    > x2^T F x1 = 0

*   `get_homography(self, image_frame_1: str, image_frame_2: str, plane_normal: numpy.ndarray, plane_distance: float) -> numpy.ndarray`
    > Compute Homography H mapping pixels from image 1 to image 2 induced by a plane.
    > 
    > x2 ~ H x1
    > 
    > Plane equation in Camera 1 frame: n^T X = d
    > H = K2 (R + t n^T / d) K1^-1
    > 
    > Args:
    >     image_frame_1: Source image frame.
    >     image_frame_2: Target image frame.
    >     plane_normal: Normal vector of the plane in Camera 1 frame (3,).
    >     plane_distance: Distance to the plane in Camera 1 frame (scalar).

*   `get_transform(self, source_frame: str, target_frame: str) -> tgraph.transform.BaseTransform`
    > Get the transform from source_frame to target_frame.
    > 
    > Automatically finds the shortest path and composes transforms.
    > Results are cached as shortcut edges for O(1) subsequent lookups.
    > 
    > Args:
    >     source_frame: The source frame ID.
    >     target_frame: The target frame ID.
    > 
    > Returns:
    >     BaseTransform: The composed transform T_source_to_target.
    > 
    > Raises:
    >     ValueError: If either frame doesn't exist or no path exists.

*   `has_frame(self, frame_id: str) -> bool`
    > Check if a frame exists in the graph.

*   `has_transform(self, source_frame: str, target_frame: str) -> bool`
    > Check if a direct transform (edge) exists between two frames.

*   `is_projection_frame(self, frame_id: str) -> bool`
    > Check if a frame is a 2D projection frame (e.g., an Image frame).
    > 
    > Rule: A frame is a projection frame if ALL edges connected to it treat it as a projection space.
    > - If transform maps INTO frame (frame is Target), transform must be a Projection.
    > - If transform maps OUT OF frame (frame is Source), transform must be an InverseProjection.

*   `remove_transform(self, frame_a: str, frame_b: str) -> None`
    > Remove a transform (edge) between two frames.
    > 
    > Args:
    >     frame_a: First frame ID.
    >     frame_b: Second frame ID.
    > 
    > Raises:
    >     ValueError: If no edge exists between these frames.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize the entire graph to a dictionary.
    > 
    > Returns:
    >     Dict containing 'frames' and 'edges' (only explicit, non-cached edges).

*   `update_transform(self, source_frame: str, target_frame: str, transform: tgraph.transform.BaseTransform) -> None`
    > Update an existing transform between two frames.
    > 
    > Automatically invalidates any cached shortcuts that depend on this edge.
    > 
    > Args:
    >     source_frame: The source frame ID.
    >     target_frame: The target frame ID.
    >     transform: The new transform from source to target.
    > 
    > Raises:
    >     ValueError: If no edge exists between these frames.

#### `Translation`

```python
class Translation(x: float = 0.0, y: float = 0.0, z: float = 0.0, translation: numpy.ndarray | list | tuple | None = None)
```

A Transform with only translation (Identity rotation).

**Methods:**

*   `__init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, translation: numpy.ndarray | list | tuple | None = None)`
    > Initialize self.  See help(type(self)) for accurate signature.

*   `apply(self, vector: numpy.ndarray | list | tuple) -> numpy.ndarray`
    > Apply the transform to 3D vectors (Nx3 or Nx4).
    > 
    > Args:
    >     vector: Nx3 or Nx4 array of vectors.
    >             - If Nx3: Treated as 3D points/vectors (w=1 implied if Transform, but checking subclass logic).
    >               Standard BaseTransform behavior:
    >               Homogenize (w=1) -> Multiply -> Dehomogenize (w division).
    >             - If Nx4: Treated as homogeneous vectors.
    >               Multiply -> Return Nx4.
    > 
    > Returns:
    >     np.ndarray: Transformed vectors (Nx3 or Nx4).

*   `as_matrix(self) -> numpy.ndarray`
    > Returns the 4x4 homogeneous representation of the transform.
    > 
    > Returns:
    >     np.ndarray: 4x4 matrix of the transform's dtype.

*   `from_dict(data: dict[str, typing.Any]) -> 'Transform'`
    > Deserialize transform from a dictionary.

*   `from_matrix(matrix: numpy.ndarray, dtype: numpy.dtype | None = None) -> 'Transform'`
    > Creates a Transform from a 4x4 matrix.

*   `inverse(self) -> 'Transform'`
    > Returns the mathematical inverse of the transformation.
    > 
    > Returns:
    >     BaseTransform: The inverse transformation.

*   `to_dict(self) -> dict[str, typing.Any]`
    > Serialize transform to a JSON-compatible dictionary.

### Functions

#### `as_euler_angles`

```python
def as_euler_angles(q: quaternion.quaternion) -> tuple[float, float, float]
```

Extract roll-pitch-yaw Euler angles from a quaternion.

Uses the aerospace/robotics convention with intrinsic ZYX rotation order.
See Rotation class docstring for full convention details.

Args:
    q: The input quaternion.

Returns:
    Tuple[float, float, float]: (roll, pitch, yaw) in radians.

#### `decompose_projection_to_objects`

```python
def decompose_projection_to_objects(projection_matrix: numpy.ndarray, dtype: numpy.dtype = <class 'numpy.float64'>) -> tuple['CameraProjection', 'Transform']
```

Decompose a 3x4 projection matrix P into Intrinsic (CameraProjection) and Extrinsic (Transform) objects.

P = K @ [R | t]

Returns:
    (CameraProjection, Transform)
    - CameraProjection: Holds K (Intrinsic only)
    - Transform: Extrinsic Transform T_world_to_cam (OpenCV convention: R, t).
      Note: This Transform represents the conversion from World to Camera frame.
      If you want Camera Pose (Camera to World), take the inverse of this transform.

#### `deserialize_transform`

```python
def deserialize_transform(data: dict[str, typing.Any]) -> 'BaseTransform'
```

Deserialize a transform from a dictionary.

Automatically determines the correct class from the "type" field
and calls its from_dict() method.

Args:
    data: Dictionary previously created by serialize_transform() or to_dict().

Returns:
    BaseTransform: The deserialized transform instance.

Raises:
    ValueError: If the transform type is not registered.

#### `ensure_rotation`

```python
def ensure_rotation(rotation: numpy.ndarray | list | tuple | None, dtype: numpy.dtype) -> quaternion.quaternion
```

Ensures rotation is a quaternion of the specified dtype.
Optimized to avoid copies if input already matches requirements.

#### `ensure_translation`

```python
def ensure_translation(translation: numpy.ndarray | list | tuple | None, dtype: numpy.dtype) -> numpy.ndarray
```

Ensures translation is a 3x1 numpy array of the specified dtype.
Optimized to avoid copies if input already matches requirements.

#### `from_euler_angles`

```python
def from_euler_angles(roll: float, pitch: float, yaw: float) -> quaternion.quaternion
```

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

#### `get_basis_vectors`

```python
def get_basis_vectors(transform: tgraph.transform.BaseTransform, length: float = 1.0) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
```

Get the origin and basis vectors (x, y, z) of the transform's coordinate frame.

Args:
    transform: The transform defining the coordinate system.
    length: Length of the basis vectors.

Returns:
    Tuple: (origin, x_vec, y_vec, z_vec) as 3-element numpy arrays in global frame.

#### `register_transform`

```python
def register_transform(cls: type['BaseTransform']) -> type['BaseTransform']
```

Decorator to register a transform class for serialization.

Usage:
    @register_transform
    class MyTransform(BaseTransform):
        ...

#### `serialize_transform`

```python
def serialize_transform(transform: 'BaseTransform') -> dict[str, typing.Any]
```

Serialize any transform to a JSON-compatible dictionary.

Args:
    transform: Any BaseTransform subclass instance.

Returns:
    Dict containing the serialized transform with a "type" key.

#### `skew`

```python
def skew(vector: numpy.ndarray | list | tuple) -> numpy.ndarray
```

Returns the 3x3 skew-symmetric matrix of a 3-element vector.

[v]x = [[  0, -v3,  v2],
        [ v3,   0, -v1],
        [-v2,  v1,   0]]
        
Args:
    vector: A 3-element sequence (shape (3,) or (3,1)).
    
Returns:
    np.ndarray: 3x3 skew-symmetric matrix.

#### `transform_points`

```python
def transform_points(transform: tgraph.transform.BaseTransform, points: numpy.ndarray) -> numpy.ndarray
```

Applies a transformation to a set of 3D points.

Strictly supports Transform, Rotation, Translation, MatrixTransform.
For Projections, use projection.apply(vector).

Args:
    transform: The transformation to apply.
    points: Nx3 array of points.

Returns:
    np.ndarray: Nx3 array of transformed points.
    
Raises:
    TypeError: If transform is not a rigid/matrix transform.


---

## Module: `tgraph.visualization`

Visualization module for TransformGraph using Plotly.

### Functions

#### `visualize_graph`

```python
def visualize_graph(transform_graph: tgraph.transform.TransformGraph, title: str = 'Graph Topology (2D)') -> 'go.Figure'
```

Visualize the graph topology in 2D using NetworkX layout.

Shows nodes, edges, transform types, and caching status.

Args:
    transform_graph: The TransformGraph instance.
    title: Plot title.

Returns:
    plotly.graph_objects.Figure

#### `visualize_transform_graph`

```python
def visualize_transform_graph(transform_graph: tgraph.transform.TransformGraph, root_frame: str | None = None, axis_scale: float = 1.0, show_connections: bool = True, show_frustums: bool = True, frustum_scale: float = 1.0, title: str = 'Transform Graph Visualization (3D)') -> 'go.Figure'
```

Visualize the transform graph in 3D.

