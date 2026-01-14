import sys
sys.path.insert(0, 'src')
import numpy as np
import tgraph.transform as tf
import unittest

class TestInverseCameraProjection(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.array([0, 0, 0], dtype=np.float64)
        self.cam = tf.CameraProjection(K=self.K, R=self.R, t=self.t, image_size=(1280, 720))

    def test_shortcuts(self):
        print("\nTesting shortcuts...")
        self.assertEqual(self.cam.fx, 1000)
        self.assertEqual(self.cam.fy, 1000)
        self.assertEqual(self.cam.cx, 640)
        self.assertEqual(self.cam.cy, 360)
        inv_cam = self.cam.inverse()
        self.assertEqual(inv_cam.fx, 1000)
        self.assertEqual(inv_cam.fy, 1000)
        self.assertEqual(inv_cam.cx, 640)
        self.assertEqual(inv_cam.cy, 360)
        print("Shortcuts OK")

    def test_inverse_type(self):
        print("\nTesting inverse type...")
        inv_cam = self.cam.inverse()
        self.assertIsInstance(inv_cam, tf.InverseCameraProjection)
        self.assertIs(inv_cam.camera_projection, self.cam)
        
        orig_cam = inv_cam.inverse()
        self.assertIsInstance(orig_cam, tf.CameraProjection)
        self.assertIs(orig_cam, self.cam)
        print("Inverse type OK")

    def test_composition_structure(self):
        print("\nTesting composition structure (TODO logic)...")
        # Currently we haven't implemented special __mul__ for InverseCamera yet, 
        # so it should fallback to MatrixTransform (or behave as InverseProjection)
        # But let's verify it doesn't crash
        inv_cam = self.cam.inverse()
        transform = tf.Transform(translation=[1, 0, 0])
        result = inv_cam * transform
        print(f"Result type: {type(result)}")
        # Ideally this becomes something meaningful later, but for now ensure it runs
        self.assertTrue(hasattr(result, 'as_matrix'))

if __name__ == '__main__':
    unittest.main()
