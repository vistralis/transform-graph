# Transform Framework — Architectural Guidelines

## 1. Mathematical Standard: Active Transform Convention

Our system operates on the **Active Transform** convention, where a transformation
matrix moves a coordinate from a local source frame into a global or parent target frame.

$$X_{target} = T_{source \to target} \cdot X_{source}$$

**Domain and Codomain:**
- **Domain**: The Source Coordinate Frame (e.g., $X_{robot}$) — the space where a vector
  is originally defined relative to a local origin.
- **Codomain**: The Target Coordinate Frame (e.g., $X_{world}$) — the space into which
  the vector is mapped, usually a global or parent coordinate system.

**The Concept of Pose:**
The Pose of an entity is the transformation required to map its local origin and axes
into the World Frame:
- Robot Pose = $T_{robot \to world}$
- Camera Pose = $T_{camera \to world}$

The translation component of the pose represents the position of the source origin
in the target frame, while the rotation component represents its orientation.

## 2. Intrinsic/Extrinsic Separation

`CameraProjection` is **strictly intrinsic-only**. It represents the internal geometry
of the optical sensor. All spatial information (position and orientation) must be
managed as standard Transform objects.

**Constraints:**
1. **Storage**: `CameraProjection` holds the $3 \times 3$ intrinsic matrix $K$
   and OpenCV distortion coefficients $D$. No rotation or translation data.
2. **State Management**: Extrinsics ($R|t$) must be held separately as a `Transform`
   object, ensuring camera motion is treated identically to any other graph edge.
3. **Inversion**: Because $P = K [R|t]$ uses a World-to-Camera mapping, conversion
   utilities invert this relationship to return a Camera-to-World `Transform` (Pose).

**Object Properties:**
- `CameraProjection`: Holds $K$, $D$, and provides quick access to $f_x, f_y, c_x, c_y$.
- `InverseCameraProjection`: Preserves the same parameters but returns $K^{-1}$
  via `.as_matrix()`.

## 3. Algebraic Composition Rules

| Operation | Result | Semantic Meaning | Status |
|-----------|--------|------------------|--------|
| `Projection * Transform` | `CompositeProjection` | Full projection from any frame | Valid |
| `Transform * Projection` | N/A | Invalid dimensionality/logic | Forbidden |
| `Transform * InverseProjection` | `InverseCompositeProjection` | Unproject to any frame | Valid |

## 4. Decompose Projection to Objects

The `decompose_projection_to_objects` utility extracts intrinsics and extrinsics
from a raw $3 \times 4$ projection matrix:

1. RQ Decomposition on the $3 \times 3$ sub-matrix to separate $K$ from $R$.
2. Standardize $K$: Ensure diagonal elements are positive.
3. Normalize $K$: Ensure scale factor $K_{2,2} = 1.0$.
4. Extract translation by solving $P_{col4} = K \cdot t$.
5. Invert for Pose: Construct extrinsic matrix from $R$ and $t$,
   then compute its inverse to obtain Camera-to-World Pose.

**Output**: A tuple of `(CameraProjection, Transform)`.

## 5. Orthographic Projection Standard

`OrthographicProjection` extends the `Projection` class with a pure affine mapping
(no perspective division). It overrides `_apply()` to skip normalization.

**Key Properties:**
- **No Perspective Division**: Output coordinates are a linear function of input:
  `pixel = A * world + b`.
- **Axis Presets**: `"top"` (drops Z), `"front"` (drops X), `"side"` (drops Y).
- **Invertible**: The inverse places points on the projection plane (collapsed axis = 0).

| Operation | Result | Semantic Meaning | Status |
|-----------|--------|------------------|--------|
| `OrthographicProjection * Transform` | `CompositeProjection` | Affine map from any frame | Valid |
| `Transform * InverseOrthographicProjection` | `InverseCompositeProjection` | Lift pixels to any frame | Valid |

**Graph Integration:**
- Registered as a standard graph edge between a 3D frame and a 2D frame.
- `transform_points` returns `[col, row, 1]` (constant z=1, unlike perspective).
- `project_points` returns `(N, 2)` pixel coordinates directly.