import numpy as np
import sys
sys.path.insert(0, 'src')
import tgraph.transform as tf
import tgraph.visualization as vis

# User provided parameters
cam_front = {
    "camera_matrix": [
        [1757.6433677063817, 0.0, 983.2146800742082],
        [0.0, 1761.2943542186904, 550.4079103066119],
        [0.0, 0.0, 1.0],
    ],
    "dist_coeffs": [
        0.027552039300292,
        0.032744776829062,
        -0.002784825282852963,
        0.0004911277422916802,
        -0.016114769915598803,
    ],
    "image_height": 1080,
    "image_width": 1920,
    "rms_error": 0.3943461153503168,
    "sensor_id": "CAM_FRONT",
}

K = np.array(cam_front['camera_matrix'], dtype=np.float32).reshape(3,3)
D = np.array(cam_front['dist_coeffs'], dtype=np.float32)

robot_transform = tf.Translation(x=1.0)
camera_transform = tf.Translation(z=0.1)

transform_graph = tf.TransformGraph()
transform_graph.add_transform('robot', 'world', robot_transform)
transform_graph.add_transform('camera_front_link', 'robot', camera_transform)
transform_graph.add_transform('camera_optical_frame', 'camera_front_link',  tf.Rotation(x = -0.5,y =  0.5,z = -0.5,w =  0.5))

# User's way of adding projection: Source=Cam, Target=Image. 
# This means: Image = Proj * Cam. 
# In Graph: Parent=Image, Child=Camera.
transform_graph.add_transform('camera_optical_frame', 'IMAGE_camera_front', tf.CameraProjection(K=K, D=D))

print("Graph Edges:")
for u, v, data in transform_graph.graph.edges(data=True):
    print(f"{u} -> {v} (Parent: {data.get('parent')}, Type: {type(data.get('transform')).__name__})")

try:
    fig = vis.visualize_transform_graph(
        transform_graph, 
        root_frame='world', 
        show_frustums=True, 
        frustum_scale=0.5
    )
    
    # Check if we have frustum traces
    frustum_traces = [t for t in fig.data if 'frustum' in str(t.name).lower()]
    print(f"Number of frustum traces: {len(frustum_traces)}")
    
    output_file = "camera_front_viz.html"
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    
except Exception as e:
    print(f"Visualization failed: {e}")
    import traceback
    traceback.print_exc()
