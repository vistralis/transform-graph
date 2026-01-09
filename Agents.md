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
*   **No Matrix Constructor:** Direct initialization from matrices in the main `Transform` classes is forbidden to preserve type clarity. Use `from_matrix()` factory methods instead.

#### B. The Hierarchy
*   **`BaseTransform`**: Abstract Interface.
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
    *   `apply()` returns 2D pixels.
*   **`InverseProjection`**: 
    *   Result of `Projection.inverse`.
    *   **Behavior:** Represents the operation $P^{-1}$. Useful for tracking logic, but applying it typically requires extra data (depth).

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

## Coding Conventions
*   **Typing:** Use explicit types (`np.float64`).
*   **Conversions:**
    *   `as_X()`: Representation/View (e.g., `pose.as_transform()`).
    *   `to_X()`: Conversion/Export (e.g., `pose.to_dict()`).
*   **Constructors:** Simplified args (`Translation(1, 0, 0)`), matching `__repr__`.
*   **Variable Naming:** Use descriptive, semantic names. Avoid single-letter or cryptic abbreviations.
    *   **Unacceptable:** `T_trans`, `T_rot`, `T_combined`, `p_new`.
    *   **Required:** `translation_x`, `rotation_z`, `camera_transform`, `transformed_point`.