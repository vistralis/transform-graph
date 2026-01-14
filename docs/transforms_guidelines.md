Transform Framework and Architectural Instructions
1. Mathematical Summary: The Transform Standard
Our system operates on the Active Transform convention, where a transformation matrix moves a coordinate from a local source frame into a global or parent target frame.
The Equation
$X_{target} = T_{source \to target} \cdot X_{source}$
Domain and Codomain
* Domain: The Source Coordinate Frame (e.g., $X_{robot}$). This is the space where a vector is originally defined relative to a local origin.
* Codomain: The Target Coordinate Frame (e.g., $X_{world}$). This is the space into which the vector is mapped, usually representing a global or parent coordinate system.
The Concept of Pose
The Pose of an entity is defined as the transformation required to map its local origin and axes into the World Frame.
* Robot Pose = $T_{robot \to world}$
* Camera Pose = $T_{camera \to world}$
The translation component of the pose represents the position of the source origin in the target frame, while the rotation component represents its orientation.
2. Redacted Instruction to Agent: Antigravity
Subject: Architectural Decoupling of Vision and Spatial Transforms
Protocol Update:
The entity CameraProjection is hereby redefined to be strictly Intrinsic-only. It represents the internal geometry of the optical sensor. All spatial information (position and orientation) must be extracted and managed as standard $4 \times 4$ Transform objects.
Implementation Constraints:
1. Storage: CameraProjection must not store rotation or translation data. It holds the $3 \times 3$ intrinsic matrix $K$ and OpenCV distortion coefficients $D$.
2. State Management: Extrinsics ($R|t$) must be held separately as a Transform object to ensure camera motion is treated identically to any other robot joint.
3. Inversion Logic: Because $P = K [R|t]$ uses a World-to-Camera mapping, conversion utilities must invert this relationship to return a Camera-to-World Transform (Pose).
Object Properties:
* CameraProjection: Holds $K$, $D$, and provides quick access to parameters: $f_x, f_y, c_x, c_y$.
* InverseCameraProjection: Represents the inverse mapping. It preserves the same parameters but returns $K^{-1}$ when evaluated via .as_matrix().
3. Algebraic Composition Rules
To maintain coordinate integrity, the following rules for multiplication are enforced:
Operation
	Result
	Semantic Meaning
	Status
	CameraProjection * Transform
	Matrix ($3 \times 4$)
	Full Projection ($K[R \vert t]$)
	Valid
	Transform * CameraProjection
	N/A
	Invalid dimensionality/logic
	Forbidden
	Transform * InverseCameraProjection
	Matrix ($4 \times 3$)
	Unprojection to World ($(K[R \vert t])^{-1}$)
	Valid
	InverseCameraProjection * Transform
	Matrix ($3 \times 4$)
	Transform in Normalized Image Space
	Valid
	4. Functional Specification: decompose_projection_to_objects
The legacy decomposition logic is refactored into a standalone utility within the transform package.
Function Requirements:
* Input: A $3 \times 4$ projection matrix ($P$).
* Process:
   1. Perform RQ Decomposition on the $3 \times 3$ sub-matrix to separate $K$ from $R$.
   2. Standardize $K$: Ensure diagonal elements are positive; adjust $R$ to maintain the product.
   3. Normalize $K$: Ensure scale factor $K_{2,2} = 1.0$.
   4. Extract Translation: Solve $P_{col4} = K \cdot t$ to find translation vector $t$.
   5. Invert for Pose: Construct the $4 \times 4$ extrinsic matrix from $R$ and $t$, then calculate its inverse to obtain the Camera-to-World Pose.
* Output: A tuple of (CameraProjection, Transform).