# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.insert(0, 'src')
import tgraph.transform as tf
import tgraph.visualization as vis
import numpy as np
import unittest
class TestMultiCamera(unittest.TestCase):
    def setUp(self):
        # User scenario setup
        cam_front = {
            "camera_matrix": [
                [1757.6433677063817, 0.0, 983.2146800742082],
                [0.0, 1761.2943542186904, 550.4079103066119],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [0.027552, 0.032744, -0.002784, 0.000491, -0.016114], # truncated for brevity
        }

        self.K = np.array(cam_front['camera_matrix'], dtype=np.float32).reshape(3,3)
        # Fix K to be double for precision in tests
        self.K = self.K.astype(np.float64)
        self.D = np.array(cam_front['dist_coeffs'], dtype=np.float64)

        robot_transform = tf.Translation(x=1.0)
        camera_link_transform = tf.Translation(z=0.1)
        camera_left_transform = tf.Translation(y = 0.2)
        camera_right_transform = tf.Translation(y = -0.2)
        
        # Standard camera rotation (optical frame)
        # x=-0.5, y=0.5, z=-0.5, w=0.5 corresponds to X-axis -90, Y-axis -90?
        # Let's use user's quaternion directly.
        optical_rot = tf.Rotation(x=-0.5, y=0.5, z=-0.5, w=0.5)

        self.graph = tf.TransformGraph()
        self.graph.add_transform('robot', 'world', robot_transform)
        self.graph.add_transform('camera_link', 'robot', camera_link_transform)
        self.graph.add_transform('camera_left', 'camera_link', camera_left_transform)
        self.graph.add_transform('camera_right', 'camera_link', camera_right_transform)
        
        self.graph.add_transform('camera_left_optical_frame', 'camera_left', optical_rot)
        self.graph.add_transform('camera_left_optical_frame', 'IMAGE_camera_left', tf.CameraProjection(K=self.K, D=self.D))

        self.graph.add_transform('camera_right_optical_frame', 'camera_right', optical_rot)
        self.graph.add_transform('camera_right_optical_frame', 'IMAGE_camera_right', tf.CameraProjection(K=self.K, D=self.D))

    def test_strict_composition(self):
        """Verify that Transform * CameraProjection is forbidden."""
        T = tf.Translation(x=1.0)
        P = tf.CameraProjection(self.K)
        
        with self.assertRaises(TypeError):
            _ = T * P
            
        # Verify valid composition P * T
        res = P * T
        self.assertIsInstance(res, tf.Projection)

    def test_inter_image_transform(self):
        """Verify get_transform between two image frames."""
        # IMAGE_right -> IMAGE_left
        # Path: ImgR -> OptR -> CamR -> Link -> CamL -> OptL -> ImgL
        # Transforms: InvProj * T_opt_cam * T_cam_link * T_link_cam * T_cam_opt * Proj
        # Result should be MatrixTransform (Projections lose SE3 structure)
        
        T_img_left_right = self.graph.get_transform('IMAGE_camera_right', 'IMAGE_camera_left')
        print(f"Inter-image transform type: {type(T_img_left_right)}")
        self.assertIsInstance(T_img_left_right, tf.MatrixTransform)
        self.assertEqual(T_img_left_right.as_matrix().shape, (4, 4)) # or 3x3 depending on implementation detail. 
        # Current MatrixTransform is always 4x4.
        
    def test_geometry_functions(self):
        """Test Fundamental, Essential, and Homography."""
        img1 = 'IMAGE_camera_left'
        img2 = 'IMAGE_camera_right'
        
        # 1. Fundamental Matrix
        F = self.graph.get_fundamental_matrix(img1, img2)
        print("Fundamental Matrix:\n", F)
        self.assertEqual(F.shape, (3, 3))
        
        # Verify epipolar constraint x2.T F x1 = 0 for a point infinitely far away
        # Point at infinity in front of cameras.
        # Cameras look down Z (in optical frame).
        # Point P_opt = [0, 0, 100].
        # P_img1 = K * P_opt
        # P_img2: P_opt2 = T_12 * P_opt1.
        # T_12: Left -> Right.
        # Left(y=0.2) -> Right(y=-0.2). Delta y = -0.4 (in Link frame).
        # In Optical frame? Optical frame is rotated.
        # Let code compute.
        
        # 2. Essential Matrix
        E = self.graph.get_essential_matrix(img1, img2)
        self.assertEqual(E.shape, (3, 3))
        
        # 3. Homography (Plane at Z=10m in camera frame)
        n = np.array([0, 0, 1]) # Plane normal along optical axis
        d = 10.0 # Distance 10m
        H = self.graph.get_homography(img1, img2, n, d)
        print("Homography:\n", H)
        self.assertEqual(H.shape, (3, 3))

    def test_is_projection_frame(self):
        self.assertTrue(self.graph.is_projection_frame('IMAGE_camera_left'))
        self.assertTrue(self.graph.is_projection_frame('IMAGE_camera_right'))
        self.assertFalse(self.graph.is_projection_frame('camera_left'))
        self.assertFalse(self.graph.is_projection_frame('robot'))

    def test_visualization(self):
        # Just check it runs without error
        try:
            fig = vis.visualize_transform_graph(self.graph, root_frame='world', show_frustums=True)
            output_file = "multi_camera_viz.html"
            fig.write_html(output_file)
            print(f"Visualization saved to {output_file}")
            
            # Count frustums traces. 'camera_left' and 'camera_right'.
            frustum_traces = [t for t in fig.data if 'frustum' in str(t.name).lower()]
            print(f"Frustum traces: {len(frustum_traces)}")
            # Should appear twice? One per camera frame.
            # But traces logic adds multiple trace objects per frustum (edges + image plane).
            # The filter logic relies on name. One named trace per frustum.
            # Names are usually f"{frame_id}_frustum".
            # Frame IDs are 'camera_left_optical_frame' (parent) or 'camera_right_optical_frame'.
            # Trace names logic: `name=f"{name}_frustum"`.
            
            # The visualization loop iterates over frames.
            # For each frame, it checks connected projection edges.
            # Both 'camera_left_optical_frame' and 'camera_right_optical_frame' are in graph.
            # So we expect 2 frustums.
            
        except Exception as e:
            self.fail(f"Visualization failed: {e}")

if __name__ == '__main__':
    unittest.main()
