import sys
sys.path.insert(0, 'src')
import numpy as np
import tgraph.transform as tf
import unittest

class TestCompoundProjection(unittest.TestCase):
    def test_compound_graph_projection(self):
        graph = tf.TransformGraph()
        
        # Chain: World -> Robot -> CameraMount -> Camera -> Image
        
        # 1. World -> Robot (Translation x=10)
        graph.add_transform("robot", "world", tf.Translation(x=10.0))
        
        # 2. Robot -> CameraMount (Rotation 90 deg yaw)
        graph.add_transform("mount", "robot", tf.Rotation.from_euler_angles(yaw=np.pi/2))
        
        # 3. Mount -> Camera (Translation z=2)
        graph.add_transform("camera", "mount", tf.Translation(z=2.0))
        
        # 4. Camera -> Image (Projection)
        K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float64)
        # Standard camera looking down +Z (identity orientation in camera frame)
        proj = tf.CameraProjection(K=K, image_size=(1000, 1000))
        # Note: add_transform takes T_source_to_target. 
        # Source=Image, Target=Camera. So we need T_image_to_camera = proj.inverse()
        graph.add_transform("image", "camera", proj.inverse())
        
        # Test: Get projection from World to Image
        # Logic: P_world = get_transform(world, image) * P_in_world
        # Transform should be: Proj_cam * T_cam_mount * T_mount_robot * T_robot_world
        # This is Proj * T_world_to_cam (inverse of path transforms?)
        
        # get_transform(source, target) returns T that converts Source coords -> Target coords.
        # So get_transform("world", "image") should return P_world_to_image.
        
        full_transform = graph.get_transform("world", "image")
        
        print(f"\nFull transform type: {type(full_transform)}")
        # Under new architecture, CameraProjection * Transform -> MatrixTransform
        # because CameraProjection is strict intrinsic-only.
        # Composition results in a full projection matrix.
        self.assertTrue(isinstance(full_transform, (tf.MatrixTransform, tf.Projection)), 
            f"Should return MatrixTransform/Projection, got {type(full_transform)}")
            
        # Verify correctness with a point
        # Point in world: at robot (x=10), then locally rotated/translated...
        # Let's trace a point.
        # Robot is at x=10 in World.
        # Mount is rotated 90 deg wrt Robot.
        # Camera is z=2 above Mount. 
        # So Camera in World is at (10, 0, 2)? No, wait.
        
        # T_world_robot: x=10. (Robot frame origin in World is (10,0,0)?) NO.
        # add_transform("robot", "world", T) usually means T is position of Robot in World? 
        # OR T converts Robot points to World points?
        # definition: add_transform(source, target, transform): transform represents Source pose relative to Target.
        # T_robot_to_world.
        # So Robot origin is at T * [0,0,0] = [10, 0, 0] in World.
        
        # T_mount_to_robot: Rot 90 deg yaw.
        # Mount origin in Robot: [0,0,0].
        
        # T_cam_to_mount: z=2.
        # Cam origin in Mount: [0,0,2].
        
        # So Cam pos in World: 
        # P_world = T_r_w * T_m_r * T_c_m * P_cam_local
        # Cam origin (0,0,0 local) -> World (10, 0, 2).
        
        # Now consider a point using get_transform("world", "image").
        # This transform converts World Points -> Pixels.
        # So if I take the camera center in world (10, 0, 2), it should project to... something singular? 
        # Or let's take a point 5m in front of the camera.
        
        # Camera orientation:
        # T_r_w: Identity rot.
        # T_m_r: 90 deg yaw (Rot Z). X -> Y, Y -> -X.
        # T_c_m: Identity rot.
        # So Camera X axis points in World Y. Camera Z axis points in World Z.
        # This is weird for a camera (usually look down Z), but let's stick to it.
        # If I put a point at World (10, 5, 2).
        # Relative to Camera (10, 0, 2): Delta is (0, 5, 0).
        # In Camera frame (X=Y_world): This point is X=5?
        # Let's verify with manual calc.
        
        # P_curr = [10, 5, 2]
        # To Robot: T_w_r = inv(T_r_w). T_r_w is Trans(10,0,0). Inv is Trans(-10,0,0).
        # P_rob = [0, 5, 2]
        
        # To Mount: T_r_m = inv(T_m_r) = Rot(-90).
        # Rot(90): x->y, y->-x. Rot(-90): x->-y, y->x.
        # P_rob = (0, 5, 2). x=0, y=5.
        # P_mnt: x = 5, y = 0, z = 2.
        
        # To Camera: T_m_c = inv(T_c_m) = Trans(0,0,-2).
        # P_cam = P_mnt - (0,0,2) = (5, 0, 0).
        
        # So point is at (5, 0, 0) in Camera Frame.
        # Proj = K [I|0].
        # P_img = K * (5, 0, 0)^T = (5000, 0, 0). Normalize by Z=0? 
        # Wait, Z=0 leads to singular projection.
        # Ah, Camera convention: Look down Z.
        # Point (5, 0, 0) has Z=0.
        
        # Let's move point to have Z > 0 in Camera frame.
        # We need point with P_cam_z > 0.
        # P_cam = (x, y, z).
        # P_mnt = (x, y, z+2).
        # P_rob = (y, -x, z+2). (Rot 90 applied to P_mnt gives P_rob for vectors? No transforms are tricky manually).
        
        # Let's rely on the code to be consistent and just check types and basic values.
        # Assume valid point in front of camera.
        # Camera is at (10, 0, 2), facing +Y (World) because X_cam points +Y_world?
        # Wait:
        # T_m_r (Rot 90 Z): Basis X_m tied to Y_r. Basis Y_m tied to -X_r.
        # T_c_m (Ident): X_c = X_m = Y_r = Y_w.
        # Z_c = Z_m = Z_r = Z_w.
        # So Camera Z axis is World Z axis (Up). Camera looks Up.
        # So we need a point with World Z > 2.
        
        test_point = np.array([10, 0, 10]) # Directly above camera (Z=10, Delta Z=8).
        # P_cam should be roughly (0, 0, 8).
        # Proj: (u, v) = (fx * X/Z + cx, fy * Y/Z + cy)
        # x=0, y=0. u = cx = 500, v = cy = 500.
        
        pixels = full_transform.apply(test_point)
        print(f"Projected pixels: {pixels}")
        
        self.assertAlmostEqual(pixels[0,0], 500.0)
        self.assertAlmostEqual(pixels[0,1], 500.0)

if __name__ == '__main__':
    unittest.main()
