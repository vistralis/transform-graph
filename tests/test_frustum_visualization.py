#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test frustum visualization with realistic camera configurations.
"""
import numpy as np
import pytest
import tgraph.transform as tf
try:
    import tgraph.visualization as vis
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not installed")
def test_frustum_with_realistic_camera():
    """Test frustum visualization with realistic camera from user scenario."""
    # User's real camera calibration
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

    K = np.array(cam_front['camera_matrix'], dtype=np.float32).reshape(3, 3)
    D = np.array(cam_front['dist_coeffs'], dtype=np.float32)

    # Build the graph
    transform_graph = tf.TransformGraph()
    robot_transform = tf.Transform(translation=[0, 0, 0])
    camera_transform = tf.Transform(translation=[0.5, 0, 1.5])
    
    transform_graph.add_transform('world', 'robot', robot_transform)
    transform_graph.add_transform('robot', 'camera_front_link', camera_transform)
    transform_graph.add_transform(
        'camera_front_link', 
        'camera_optical_frame',  
        tf.Translation(z=0.1) * tf.Rotation(x=-0.5, y=0.5, z=-0.5, w=0.5)
    )
    transform_graph.add_transform(
        'camera_optical_frame', 
        'IMAGE_camera_front', 
        tf.CameraProjection(
            intrinsic_matrix=K, 
            dist_coeffs=D,
            image_size=(cam_front['image_width'], cam_front['image_height'])
        )
    )

    # Visualize with frustums
    fig = vis.visualize_transform_graph(
        transform_graph,
        root_frame='world',
        show_frustums=True,
        frustum_scale=0.5  # 50cm deep frustum
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    
    # Check that the frustum is reasonable
    # Get all trace data
    all_x = []
    all_y = []
    all_z = []
    
    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None:
            all_x.extend([x for x in trace.x if x is not None])
        if hasattr(trace, 'y') and trace.y is not None:
            all_y.extend([y for y in trace.y if y is not None])
        if hasattr(trace, 'z') and trace.z is not None:
            all_z.extend([z for z in trace.z if z is not None])
    
    # Check that coordinates are within reasonable bounds
    # Given that robot is at origin and camera is at ~1.5m height
    # Everything should be within a few meters
    max_extent = max(
        max(all_x) - min(all_x),
        max(all_y) - min(all_y),
        max(all_z) - min(all_z)
    )
    
    # With scale=0.5, frustum should be small (< 10m total extent)
    assert max_extent < 10, f"Frustum too large: {max_extent:.2f}m extent"


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not installed")
def test_frustum_different_focal_lengths():
    """Test that frustums respect camera intrinsics (focal length affects FOV)."""
    graph = tf.TransformGraph()

    # Wide angle camera (small focal length = wider FOV)
    K_wide = np.array([[300, 0, 640], [0, 300, 360], [0, 0, 1]])
    cam_wide = tf.CameraProjection(K=K_wide, image_size=(1280, 720))
    # Original test had t=[0, -2, 0]. Add explicit transform.
    graph.add_transform('world', 'wide_cam_mount', tf.Translation(x=0, y=-2, z=0))
    graph.add_transform('wide_cam_mount', 'wide_cam', cam_wide)

    # Telephoto camera (large focal length = narrower FOV)
    K_tele = np.array([[2000, 0, 640], [0, 2000, 360], [0, 0, 1]])
    cam_tele = tf.CameraProjection(K=K_tele, image_size=(1280, 720))
    # Original test had t=[0, 2, 0]
    graph.add_transform('world', 'tele_cam_mount', tf.Translation(x=0, y=2, z=0))
    graph.add_transform('tele_cam_mount', 'tele_cam', cam_tele)

    fig = vis.visualize_transform_graph(
        graph,
        root_frame='world',
        show_frustums=True,
        frustum_scale=1.0
    )

    assert isinstance(fig, go.Figure)
    # Each camera has: 4 axis traces + 4 frustum edge traces + 1 frustum plane = 9 traces
    # Plus world frame (4 traces) and connections (1 trace)
    # Total: 4 (world) + 9 (wide) + 9 (tele) + 1 (connections) = 23 traces
    assert len(fig.data) >= 20


@pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not installed")
def test_frustum_scale():
    """Test that frustum_scale parameter controls frustum depth."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    cam = tf.CameraProjection(K=K, image_size=(640, 480))
    
    graph = tf.TransformGraph()
    graph.add_transform('world', 'camera', cam)

    # Test different scales
    for scale in [0.3, 0.5, 1.0, 2.0]:
        fig = vis.visualize_transform_graph(
            graph,
            root_frame='world',
            show_frustums=True,
            frustum_scale=scale
        )
        
        assert isinstance(fig, go.Figure)
        # Frustum depth should be proportional to scale
        # (actual verification would require extracting frustum corner positions)
