#!/usr/bin/env python3
"""
Test the Pose class and its frame_id semantics.
"""

import numpy as np
import quaternion
import tgraph.transform as tf
import pytest

def test_pose_init_and_properties():
    """Test Pose initialization and properties."""
    # Default init (Identity)
    p1 = tf.Pose()
    assert np.allclose(p1.position, [0, 0, 0])
    assert p1.orientation == quaternion.one
    assert p1.frame_id is None
    assert p1.child_frame_id is None
    
    # Init with values
    pos = np.array([1.0, 2.0, 3.0])
    quat = quaternion.from_euler_angles(0, 0, np.pi/2)
    p2 = tf.Pose(position=pos, orientation=quat, frame_id="map", child_frame_id="base_link")
    
    assert np.allclose(p2.position, pos)
    assert np.isclose(p2.orientation, quat) # Quaternion close check
    assert p2.frame_id == "map"
    assert p2.child_frame_id == "base_link"
    
    # Test property setters
    p2.position = [4, 5, 6]
    assert np.allclose(p2.position, [4, 5, 6])
    
    # Test rvec init (3 elements for orientation)
    rvec = np.array([0.1, 0.2, 0.3])
    p3 = tf.Pose(orientation=rvec)
    expected_quat = quaternion.from_rotation_vector(rvec)
    assert np.isclose(p3.orientation, expected_quat)

    # Test invalid init
    with pytest.raises(ValueError):
        tf.Pose(orientation=[1, 2]) # Invalid shape

def test_pose_inverse():
    """Test Pose inversion logic with frame swapping."""
    # T_map_base
    p = tf.Pose(
        position=[1, 0, 0], 
        orientation=quaternion.one, 
        frame_id="map", 
        child_frame_id="base_link"
    )
    
    # Default inverse: T_base_map
    inv = p.inverse()
    assert inv.frame_id == "base_link"
    assert inv.child_frame_id == "map"
    assert np.allclose(inv.position, [-1, 0, 0])
    
    # Override inverse frames
    inv2 = p.inverse(new_frame_id="new_root", new_child_frame_id="old_root")
    assert inv2.frame_id == "new_root"
    assert inv2.child_frame_id == "old_root"
    
    # Partial override
    inv3 = p.inverse(new_frame_id="partial")
    assert inv3.frame_id == "partial"
    assert inv3.child_frame_id == "map" # Default fallback to self.frame_id

def test_pose_composition():
    """Test Pose composition (* operator)."""
    # T_world_robot
    p1 = tf.Pose(position=[1, 0, 0], frame_id="world", child_frame_id="robot")
    # T_robot_cam
    p2 = tf.Pose(position=[0, 0, 1], frame_id="robot", child_frame_id="camera")
    
    # Compose: T_world_cam
    p3 = p1 * p2
    
    assert np.allclose(p3.position, [1, 0, 1])
    assert p3.frame_id == "world"
    assert p3.child_frame_id == "camera"
    
    # Compose with Transform
    t = tf.Translation(x=1)
    p4 = p1 * t
    assert isinstance(p4, tf.Pose)
    assert np.allclose(p4.position, [2, 0, 0])
    assert p4.frame_id == "world"
    assert p4.child_frame_id is None # Transform has no frame info, so child becomes None

def test_pose_conversions():
    """Test conversions to list/matrix/transform."""
    p = tf.Pose(position=[1, 2, 3], orientation=quaternion.x)
    
    # as_transform
    t = p.as_transform()
    assert isinstance(t, tf.Transform)
    assert np.allclose(t.translation.flatten(), p.position)
    
    # from_transform
    p2 = tf.Pose.from_transform(t, frame_id="A", child_frame_id="B")
    assert p2.frame_id == "A"
    
    # to_list
    lst = p.to_list()
    assert len(lst) == 7 # px, py, pz, qw, qx, qy, qz
    assert lst[0] == 1.0
    
    # to_matrix
    mat = p.to_matrix()
    assert mat.shape == (4, 4)
    
    # __repr__
    s = repr(p)
    assert "Pose(" in s
    assert "position=" in s
    
    p_frames = tf.Pose(frame_id="A", child_frame_id="B")
    s_frames = repr(p_frames)
    assert "frame_id='A'" in s_frames
    assert "child_frame_id='B'" in s_frames
