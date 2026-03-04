# Analysis: The Transform-Projection Composition Dilemma

> **Status: Resolved** — Option E (`CompositeProjection`) was adopted and
> implemented. This document is retained as the architectural decision record.

## 1. The Problem
In a unified graph where nodes can be 2D (Images) or 3D (Spatial), traversing an edge often involves composing a **Rigid Transform** ($T \in SE(3)$) with a **Projective Transform** ($P \in \mathbb{RP}^2$ or $K$).

Mathematically:
- **Projection**: $x_{2D} = K \cdot T^{-1} \cdot X_{3D}$ (World $\to$ Cam $\to$ Image)
- **Unprojection**: $Direction_{3D} = T \cdot K^{-1} \cdot x_{2D}$ (Image $\to$ Cam $\to$ World)

### Current Behavior
In `transform-graph`:
- `Transform * Projection` $\to$ `Projection` (wraps the resulting $3 \times 4$ matrix).
- `Transform * InverseProjection` $\to$ `MatrixTransform` (wraps the resulting $4 \times 4$ matrix).

This creates an asymmetry. The semantic richness of "Intrinsics + Extrinsics" is lost in the second case, degrading into a raw matrix. This breaks tools (like visualization) that expect separated Pose and K information.

## 2. Multi-Perspective Analysis

### A. Linear Algebra / Projective Geometry
From a strict math perspective, `MatrixTransform` ($4 \times 4$ Homography) is the correct closure for these operations. The set of projective transformations forms a group. Distinguishing between "Intrinsic" and "Extrinsic" components is a **decomposition** problem, not a representation problem.
* **Verdict**: The current behavior is mathematically "correct" but semantically "lossy".

### B. Robotics & SLAM (e.g., GTSAM, ORB-SLAM)
These fields almost **never** multiply $K$ and $T$ into a single static matrix for storage.
- A **Camera** is an object composed of `Pose` (R, t) and `Calibration` (K).
- Optimization (Bundle Adjustment) adjusts `Pose` while keeping `Calibration` constrained (or optimizing strictly its parameters).
- **Verdict**: We should maintain separation. A generic `MatrixTransform` prevents us from using the object for optimization or geometry-aware tasks (e.g., limiting unprojection to valid depth ranges).

### C. Computer Graphics (OpenGL/Three.js)
CG uses the **ModelViewProjection (MVP)** matrix.
- `MVP = Projection * View * Model`.
- The shader receives a single $4 \times 4$ matrix.
- **Verdict**: For *rendering*, a combined matrix is fine. For *reasoning*, it is not. Since `transform-graph` is a reasoning library, we shouldn't follow the CG pattern blindly.

### D. Deep Learning (PyTorch3D, Kornia)
These libraries often use Tensor representations but provide specific `PinholeCamera` classes that hold batched $R, t, K$ tensors separately.
- **Verdict**: Separation is key for differentiability and parameter updates.

## 3. Evaluated Alternatives

### Option A: `CompositeTransform` (Lazy Evaluation)
Create a generic container that stores the chain of operations without multiplying them.
`CompositeTransform([InverseCameraProjection(K), Transform(T)])`
- **Pros**: Zero information loss. Perfect reconstruction of components.
- **Cons**: High complexity. Every method (inverse, apply, multiply) needs to handle the chain logic or collapse it on demand. `apply()` would be slower (multiple multiplies per point).

### Option B: `CameraTransform` / `PinholeCameraModel` (Specialized Class)
Create a specific class to represent the result of `Transform * CameraProjection` (or its inverse).
- **Structure**:
    ```python
    class PosedCameraProjection(Projection):
        pose: Transform  # T_world_to_cam (Extrinsics)
        intrinsic: CameraProjection # K (Intrinsics)
        
        def as_matrix(self):
            return self.intrinsic.as_matrix() @ self.pose.inverse().as_matrix()
    ```
- **Pros**: Explicit semantics. Visualization works trivially (`viz(cam)` uses `cam.pose` and `cam.intrinsic`). Lightweight.
- **Cons**: Adds a new type. Must define composition rules carefully.

### Option C: RQ Decomposition (The "Decompose" Approach)
Keep `MatrixTransform`, but provide a robust `decompose()` method that uses RQ decomposition to recover $K$ and $T$ only when needed (e.g., for visualization).
- **Pros**: No new classes.
- **Cons**: Numerical instability. Sign ambiguity (is $f_x$ negative or is camera looking backwards?). Loss of metadata (distortion coefficients are lost if we only store the matrix).

### Option E: `CompositeProjection` (User Proposal)
A refined version of Option B, focusing on generic composition rules.

1.  **`CompositeProjection`**:
    -   Represents `Projection * Transform`.
    -   Stores `(intrinsics: Projection, extrinsics: Transform)`.
    -   **Multiplication Rule**: Multiplying on the right by `Transform` updates the internal `extrinsics` object ($T_{new} = T_{old} \cdot T_{update}$).
    -   `as_matrix()`: Returns `intrinsics.matrix @ extrinsics.inverse().matrix` (or appropriate composition).

2.  **`InverseCompositeProjection`**:
    -   Represents `Transform * InverseProjection`.
    -   Stores `(extrinsics: Transform, inverse_intrinsics: InverseProjection)`.
    -   **Multiplication Rule**: Multiplying on the left by `Transform` updates the internal `extrinsics` object ($T_{new} = T_{update} \cdot T_{old}$).

**Benefits**:
-   **Generic**: Works for any `Projection` subclass (Pinhole, Fisheye, Orthographic).
-   **Visualization Ready**: `viz(tf)` simply checks for `CompositeProjection`, extracts `extrinsics` for the cone pose, and `intrinsics` for the shape.
-   **Algebraic Closure**: Defines a clear algebra for mixing SE(3) and Projective elements without degrading to raw matrices immediately.

## 4. Recommendation

**Adopt Option E (`CompositeProjection`).**

It provides the necessary semantic container to separate "Where the sensor is" (Transform) from "How the sensor projects" (Intrinsics), while robustly handling the group algebra.

### Proposed Hierarchy Update:
1.  **`CompositeProjection`**: Container for `(Projection, Transform)`.
2.  **`InverseCompositeProjection`**: Container for `(Transform, InverseProjection)`.
3.  Update `BaseTransform.__mul__` to detect these cases and return the Composite objects instead of fallback `MatrixTransform`.
