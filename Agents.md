# Agents.md - The Foundation of `tgraph`

## Vision
To build a high-performance spatial transformation and frame graph library: **`transform-graph`** (Package Name) / **`tgraph`** (Namespace).
This library is a foundational tool for Robotics and Spatial AI, providing the fundamental mathematical layer for Space (SE3 Transforms, Projections, Poses) with production-grade rigor.

**Target Environment:** Python 3.12+, NumPy 2.0+ (Leveraging modern performance, typing, and the latest numerical standards).

## Core Identity
*   **Package Name:** `transform-graph` (on PyPI).
*   **Import Namespace:** `tgraph` (Short, standard, avoids collisions).
    *   `import tgraph.transform as tf`
    *   `import tgraph.visualization as vis`
*   **Philosophy:**
    *   **Monolith Modules:** We prioritize readability and minimalism. We use single, self-contained files for major modules (e.g., `transform.py`).
    *   **Strict Types:** Transforms are mathematical operators. Poses are states with frames.
    *   **High Performance:** Powered by `numpy` and `numpy-quaternion`.
    *   **Unified Graph:** Supports Rigid Transforms (invertible) and Projections (one-way).
    *   **Usability:** First-class Visualization (Plotly).

## Technical Architecture

### 1. Mathematical Core (`tgraph.transform`)
The foundation uses `numpy` and `numpy-quaternion`.

#### A. Smart Constructors
*   **Simplicity of Use:** Constructors support multiple input modes (kwargs, lists, numpy) without verbosity.
*   **Validation:** 
    *   `Translation` inputs must resolve to exactly 3 elements.
    *   `Rotation` inputs must resolve to exactly 4 elements (w, x, y, z) or be a `quaternion` object.
    *   `Rotation.from_euler_angles(roll, pitch, yaw)`: Preferred factory method for rotations.
*   **No Matrix Constructor:** Direct initialization from matrices in the main `Transform` classes is forbidden to preserve type clarity. Use `from_matrix()` factory methods instead.

#### B. Euler Angle Convention (Roll-Pitch-Yaw)

This library uses the **aerospace/robotics convention** for Euler angles:

| Angle | Axis | Description | Domain |
|-------|------|-------------|--------|
| **Roll (φ)** | X-axis | Banking left/right | Forward axis |
| **Pitch (θ)** | Y-axis | Nose up/down | Lateral axis |
| **Yaw (ψ)** | Z-axis | Heading direction | Vertical axis |

**Rotation Order:** Intrinsic ZYX (yaw → pitch → roll), equivalent to extrinsic XYZ.

This means rotations are applied:
1. First rotate about Z (yaw) - heading direction
2. Then rotate about the new Y' (pitch) - nose up/down
3. Finally rotate about the new X'' (roll) - banking

**Why This Convention?**
*   **Aerospace Standard:** Used universally for aircraft attitude (heading, pitch, bank).
*   **Robotics Standard:** Aligns with ROS REP-103 for mobile robot navigation.
*   **Intuitive:** Yaw = "where am I pointing?", Pitch = "tilted up/down?", Roll = "tilted sideways?"

**API:**
```python
# Create rotation from Euler angles (radians)
rotation = tf.Rotation.from_euler_angles(roll=0.1, pitch=0.2, yaw=0.3)

# Extract Euler angles from rotation
roll, pitch, yaw = rotation.as_euler_angles()
```

**Warning:** Gimbal lock occurs when pitch = ±90°. At this singularity, roll and yaw become coupled.

#### C. The Hierarchy
*   **`BaseTransform`**: Abstract Interface.
    *   **Data Types:** Supports configurable `dtype` (default is `np.float64`).
*   **`Transform`**: The workhorse (SE3).
    *   **Structure:** `translation` (3x1 `np.array`) + `rotation` (`quaternion`).
    *   **Operations:** Composition (`*`), Inversion, Application to points.
*   **Specializations:**
    *   `Translation(x=1)`: Rotation is Identity.
    *   `Rotation(w=1)`: Translation is Zero.
    *   `Identity()`: The Unity transform. Signals "intentionally neutral/empty".
*   **`MatrixTransform`**: 
    *   The most general fallback. Holds a raw `4x4` matrix.
    *   Used when structure (t/r) is lost or irrelevant.
*   **`Projection`**: 
    *   Represents 3D -> 2D (Camera Intrinsics).
    *   Stored as 4x4 matrix (bottom row `[0,0,0,1]`) for compatibility.
    *   Accepts 3x4 or 4x4 matrices; internally converts to 4x4.
    *   `apply(points)` returns 2D pixels with perspective division.
*   **`CameraProjection`**:
    *   A `Projection` with explicit intrinsic (K) and extrinsic (R, t) parameters.
    *   Can be constructed from K, R, t or decomposed from a 3x4/4x4 matrix via RQ decomposition.
    *   Provides `focal_length` and `principal_point` properties.
*   **`InverseProjection`**: 
    *   Result of `Projection.inverse()` or `CameraProjection.inverse()`.
    *   **Behavior:** Represents the conceptual operation $P^{-1}$. Useful for tracking logic in the graph.
    *   **Limitation:** Projections are not truly invertible (3D→2D loses depth). The `unproject(pixels, depths)` method requires depth values to recover 3D points.
    *   **Recovery:** To recover a `CameraProjection` from an `InverseProjection`, use: `CameraProjection(matrix=inv_proj.inverse().as_matrix())`.

#### C. The State (`Pose`)
*   **Concept:** A "Noun" (Where am I?).
*   **Structure:** `position`, `orientation`, **`frame_id`** (Parent).
*   **API:**
    *   `as_transform()`: Views the Pose as the Transform $T_{parent 	o body}$.
    *   `as_matrix()`: Helper.

### 2. Data Types: Points and Vectors
The distinction between points and vectors is fundamental.

*   **Points:** Represent a position in space.
    *   **Representation:** `Nx3` numpy arrays.
*   **Vectors:** Represent directions or displacements.
    *   **Representation:** Column vectors, `Nx1`.
*   **Utilities:**
    *   `tf.transform_points(transform, points)`: Primary function for applying a `Transform` to a point set (`Nx3`).

### 3. The Graph (`tgraph.transform.TransformGraph`)
*   **Role:** Manages the tree/graph of frames.
*   **Features:**
    *   Stores `Transform` objects on edges.
    *   Resolves paths (shortest path).
    *   Handles Inversion ($A 	o B$ becomes $B 	o A$).
    *   Can traverse `Projection` edges (but not invert them).

---

#### Architecture & Core Principles

**1. Undirected Topology with Directional Metadata**
* To optimize memory, the graph uses an undirected `nx.Graph`.
* Each edge stores a `BaseTransform` object and a `parent` attribute identifying the original source frame.
* This prevents redundant storage of inverse transforms and ensures a single "Source of Truth" for every spatial relationship.

**2. Lazy Inversion & Composition**
* **Traversal:** When querying a transform between two frames, the graph finds the shortest path.
* **Inversion:** For each step $(u, v)$ in the path, if $u \neq parent$, the `.inverse()` method of the `Transform` object is called on-the-fly.
* **Composition:** Transforms are chained using the `__mul__` operator, following the convention: $(T_1 * T_2) * p = T_1 * (T_2 * p)$.

**3. Dependency-Aware Shortcut Caching**
* **Performance:** Computed paths are injected back into the graph as "shortcut" edges with a lower weight (`0.1` vs `1.0`). Subsequent queries result in $O(1)$ lookups.
* **Invalidation:** The class maintains a dependency map. Updating or modifying a "Ground Truth" edge triggers a surgical invalidation of all shortcut edges that relied on that specific transform.

#### Class Interface Integration

The graph expects objects implementing the `BaseTransform` abstract base class:
* `as_matrix()`: For numerical validation or external library compatibility.
* `inverse()`: For traversing edges against their natural direction.
* `__mul__`: For recursive composition of frame chains.
* `to_dict()`: For serialization via `nx.node_link_data`.

#### Serialization Workflow

The graph is designed to be serialized into a JSON-compatible dictionary. 
* **Export:** Transient cache edges are filtered out. Only "Ground Truth" edges are exported. `Transform` objects are converted to dictionaries using their internal `to_dict()` methods.
* **Import:** The graph is reconstructed, and the dependency map is initialized fresh.

### 4. Visualization (`tgraph.visualization`)
*   **Engine:** Plotly (Interactive, Web-ready).
*   **Optional Dependency:** `plotly` is only required for this module.

### 5. Dependency Strategy
We prioritize a lightweight core.

*   **Core (`tgraph.transform`):**
    *   `numpy >= 2.0`
    *   `numpy-quaternion`
    *   `scipy`
    *   `networkx`
*   **Visualization (`tgraph.visualization`):**
    *   **Optional:** `plotly`
    *   **Mechanism:** `import tgraph.visualization` raises a helpful `ImportError` if `plotly` is not installed. Users install via `pip install transform-graph[viz]`.

### 6. Serialization System

#### A. Design Philosophy
The serialization system uses a **registry pattern** to support extensibility while maintaining type safety. Every `BaseTransform` subclass can be serialized and deserialized without modification to core code.

#### B. The Registry
```python
# Global registry maps type names to classes
_TRANSFORM_REGISTRY: Dict[str, Type[BaseTransform]] = {}

@register_transform  # Decorator auto-registers the class
class MyTransform(BaseTransform):
    ...
```

#### C. Public API
| Function | Description |
|----------|-------------|
| `serialize_transform(transform)` | Serialize any `BaseTransform` to a JSON-compatible dict |
| `deserialize_transform(data)` | Deserialize any transform from dict (auto-detects type) |

#### D. Contract for `to_dict()` / `from_dict()`
Every `BaseTransform` subclass **MUST** implement:
* `to_dict() -> Dict[str, Any]`: Returns a dictionary with a `"type"` key matching the class name.
* `from_dict(cls, data) -> BaseTransform`: Class method to reconstruct from dictionary.

**Example:**
```python
def to_dict(self) -> Dict[str, Any]:
    return {
        "type": "Transform",  # MUST match class name
        "translation": [...],
        "rotation": [...],
        "dtype": "float64",
    }
```

#### E. Registered Types
| Type | Description |
|------|-------------|
| `Transform` | SE(3) rigid body transform (translation + quaternion) |
| `MatrixTransform` | Raw 4x4 matrix |
| `Projection` | 3D→2D projection (3x4/4x4 matrix) |
| `InverseProjection` | Conceptual inverse of a projection |
| `CameraProjection` | Projection with explicit K, R, t parameters |

### 7. Transform Operations & Composition Rules

#### A. The `*` Operator (Composition)
Composition follows standard matrix multiplication semantics: $(T_1 * T_2) * p = T_1(T_2(p))$.

#### B. Dimensional Analysis

Each transform type operates between spaces of specific dimensions:

| Type | Input Dimension | Output Dimension | Invertible |
|------|-----------------|------------------|------------|
| `Transform` | 3D | 3D | ✅ Yes |
| `MatrixTransform` | 3D | 3D | ✅ Yes (if non-singular) |
| `Projection` | 3D | 2D | ❌ No (loses depth) |
| `InverseProjection` | 2D | 3D | ⚠️ Requires depth |

#### C. Permissible Operations Matrix

| Left Operand | Right Operand | Result Type | Dimension Flow | Notes |
|--------------|---------------|-------------|----------------|-------|
| `Transform` | `Transform` | `Transform` | 3D→3D→3D | Optimized quaternion math |
| `Transform` | `MatrixTransform` | `MatrixTransform` | 3D→3D→3D | Falls back to matrix mult |
| `Projection` | `Transform` | `Projection` | 3D→3D→2D | Projects from different frame |
| `Projection` | `Projection` | `Projection` | 3D→2D→2D | Cascaded (unusual) |
| `MatrixTransform` | `*` (any) | `MatrixTransform` | varies | Always matrix multiplication |
| `InverseProjection` | `Projection` | `MatrixTransform` | 3D→2D→3D | Projective 3D transform |
| `Projection` | `InverseProjection` | `MatrixTransform` | 2D→3D→2D | 2D homography-like |

#### D. Inversion Rules

| Type | `inverse()` Returns | Invertible? |
|------|---------------------|-------------|
| `Transform` | `Transform` | ✅ Yes (exact) |
| `MatrixTransform` | `MatrixTransform` | ✅ Yes (if non-singular) |
| `Projection` | `InverseProjection` | ⚠️ Conceptual only |
| `CameraProjection` | `InverseProjection` | ⚠️ Parameters lost |
| `InverseProjection` | `Projection` | ✅ Returns original |

#### E. Key Insights

**1. Projections Are One-Way**
* A `Projection` maps 3D → 2D, losing the depth dimension.
* `InverseProjection` tracks the *intent* to unproject but cannot actually recover 3D points without depth information.
* Use `inv_projection.unproject(pixels, depths)` when depth is available.

**2. CameraProjection Parameter Loss**
* `CameraProjection` stores structured parameters (K, R, t).
* When `.inverse()` is called, only the raw matrix is preserved as `InverseProjection`.
* To recover a `CameraProjection` from an inverse:
  ```python
  inv = camera.inverse()  # InverseProjection
  proj = inv.inverse()    # Projection (original matrix)
  camera_recovered = CameraProjection(matrix=proj.as_matrix())  # Decompose
  ```

**3. Type Degradation in Composition**
* Composing a `Transform` with a `Projection` produces a `MatrixTransform` (or `Projection`), not a `Transform`.
* This is intentional: the result is no longer a pure SE(3) rigid body transform.

**4. Graph Traversal with Projections**
* The `TransformGraph` can store `Projection` edges.
* Forward traversal works: `world → camera → image`.
* Reverse traversal returns `InverseProjection`: useful for tracking but requires depth to unproject.

#### F. Why No Homography2D Type?

A **homography** is a projective transformation between spaces of the same dimension:
* 2D Homography: 3×3 matrix (2D projective → 2D projective)
* 3D Homography: 4×4 matrix (3D projective → 3D projective)

**We do not implement a separate Homography2D type because:**

1. **3D Homography Already Exists:** `MatrixTransform` (4×4) is a general 3D projective transform. `Transform` (SE3) is a constrained 3D homography (rigid body, 6 DOF).

2. **2D Homographies Don't Naturally Arise:** Our composition operations produce `MatrixTransform` as the fallback, which correctly represents the algebra.

3. **Fundamental Matrix ≠ Homography:** The composition `P₂ * T * P₁⁻¹` produces a **Fundamental Matrix**, not a homography. The Fundamental Matrix maps points to *epipolar lines* (`x'ᵀFx = 0`), not points to points.

4. **Specialized Use Cases:** True 2D homographies arise in specific scenarios (image stitching, planar AR). Users can work with 3×3 numpy arrays directly for these cases.

**Verdict:** Compositions fall back to `MatrixTransform` when structure is lost, which is mathematically correct. No special Homography types are needed.

## Coding Conventions
*   **Typing:** Use explicit types (`np.float64`).
*   **Conversions:**
    *   `as_X()`: Representation/View (e.g., `pose.as_transform()`).
    *   `to_X()`: Conversion/Export (e.g., `pose.to_dict()`).
*   **Constructors:** Simplified args (`Translation(1, 0, 0)`), matching `__repr__`.
*   **Variable Naming:** Use descriptive, semantic names. Avoid single-letter or cryptic abbreviations.
    *   **Unacceptable:** `T_trans`, `T_rot`, `T_combined`, `p_new`.
    *   **Required:** `translation_x`, `rotation_z`, `camera_transform`, `transformed_point`.

### 8. Development & Testing Standards

*   **Framework:** `pytest` is the authoritative testing framework.
*   **Coverage:** Code changes must be accompanied by tests. We aim for high coverage, using `pytest-cov`.
*   **CI/CD:** 
    *   Automated workflows run on GitHub Actions for every push and pull request to `main`.
    *   Tests run against the supported Python versions (currently 3.12).
    *   Linting and type checking should be performed locally before pushing.
*   **Test Structure:**
    *   Tests reside in the `tests/` directory.
    *   Test files match the pattern `test_*.py`.
    *   Use `pytest` fixtures for setup/teardown where appropriate.
    *   Prefer assertions (`assert x == y`) over unittest-style checks.