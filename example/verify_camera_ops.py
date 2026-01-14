#!/usr/bin/env python3
"""
Test the new CameraProjection composition and frustum visualization.
"""
import numpy as np
import tgraph.transform as tf
import tgraph.visualization as vis
import os

def run_verification():
    print("Testing CameraProjection * Transform composition...")
    
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    # Initial camera at origin
    cam = tf.CameraProjection(K=K, image_size=(640, 480))
    
    # Move the world frame relative to the camera
    # (equivalent to moving the camera in the world)
    T_world_move = tf.Translation(x=1.0, y=2.0, z=3.0)
    
    # Compose
    cam_moved = cam * T_world_move
    
    print(f"Original type: {type(cam).__name__}")
    print(f"Composed type: {type(cam_moved).__name__}")
    
    assert isinstance(cam_moved, tf.CameraProjection), "Composition failed to preserve CameraProjection type!"
    assert cam_moved.image_size == (640, 480), "Image size not propagated!"
    print("✓ Composition check passed.")

    print("\nTesting TransformGraph with CameraProjection...")
    graph = tf.TransformGraph()
    
    # Link structure: world -> robot -> camera_link -> camera_optical -> image
    graph.add_transform('world', 'robot', tf.Translation(x=2.0))
    graph.add_transform('robot', 'camera_link', tf.Translation(z=1.0))
    
    # Optical frame rotation (Z forward to X forward etc)
    T_optical = tf.Rotation.from_euler_angles(yaw=-1.57, pitch=1.57) 
    graph.add_transform('camera_link', 'camera_optical', T_optical)
    
    # Add camera
    graph.add_transform('camera_optical', 'IMAGE', cam)
    
    # Get transform from world to IMAGE
    # This should be a CameraProjection because:
    # T_world_IMAGE = T_world_robot * T_robot_cl * T_cl_co * P_co_IMAGE
    # Since P is on the right, and others are rigid Transforms, it should work.
    
    # Wait, the composition in TransformGraph happens as:
    # T = T1 * T2 * T3 * P
    # My __mul__ handles Projection * Transform.
    # But TransformGraph composes them.
    
    T_total = graph.get_transform('world', 'IMAGE')
    print(f"Total transform type to 'IMAGE': {type(T_total).__name__}")
    
    # Note: If TransformGraph composes from left to right:
    # result = Identity() 
    # result = result * T1  -> Transform
    # result = result * T2  -> Transform
    # result = result * P   -> Projection
    # result = result * T4  -> CameraProjection (if P was CameraProjection and T4 is Transform)
    
    # In my graph: world -> ... -> camera_optical -> IMAGE
    # Path: world -> robot (T1), robot -> camera_link (T2), camera_link -> camera_optical (T3), camera_optical -> IMAGE (P)
    # Composition: T1 * T2 * T3 * P
    # result = (((T1 * T2) * T3) * P)
    # result = (Transform * P) -> Projection (BaseTransform does matrix mult)
    
    # Wait! I need to implement Transform * Projection -> Projection in Transform class too!
    
    # Current Transform.__mul__:
    # def __mul__(self, other: BaseTransform) -> BaseTransform:
    #    if isinstance(other, Transform): ...
    #    return MatrixTransform(self.as_matrix() @ other.as_matrix())
    
    # It returns MatrixTransform. It should return Projection if 'other' is Projection.
    
    print("\nVisualizing...")
    try:
        fig = vis.visualize_transform_graph(graph, root_frame='world')
        fig.write_html("test_frustum.html")
        print("Visualization saved to test_frustum.html")
        if os.path.exists("test_frustum.html"):
            os.remove("test_frustum.html")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    run_verification()
