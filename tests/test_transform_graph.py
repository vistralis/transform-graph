#!/usr/bin/env python3
"""
Test the TransformGraph implementation.
"""

import numpy as np
import tgraph.transform as tf
import pytest

def test_basic_graph_operations():
    """Test basic add/get operations."""
    
    graph = tf.TransformGraph()
    
    # Add transforms
    world_to_base = tf.Translation(x=2.0, y=1.0, z=0.0)
    base_to_camera = tf.Transform(
        translation=[0.5, 0.0, 0.5],
        rotation=[1.0, 0.0, 0.0, 0.0]
    )
    camera_to_object = tf.Translation(x=1.0, y=0.0, z=0.0)
    
    graph.add_transform("world", "base", world_to_base)
    graph.add_transform("base", "camera", base_to_camera)
    graph.add_transform("camera", "object", camera_to_object)
    
    # Get direct transforms
    retrieved = graph.get_transform("world", "base")
    
    # Verify matrix equivalence
    assert np.allclose(retrieved.as_matrix(), world_to_base.as_matrix()), "Transform mismatch!"


def test_path_finding_and_composition():
    """Test automatic path finding and composition."""
    
    graph = tf.TransformGraph()
    
    # Build a chain: world -> base -> camera -> object
    graph.add_transform("world", "base", tf.Translation(x=2.0, y=1.0, z=0.0))
    graph.add_transform("base", "camera", tf.Translation(x=0.5, y=0.0, z=0.5))
    graph.add_transform("camera", "object", tf.Translation(x=1.0, y=0.0, z=0.0))
    
    # Query a composed transform
    world_to_object = graph.get_transform("world", "object")
    
    # Manual verification
    expected_translation = np.array([2.0 + 0.5 + 1.0, 1.0 + 0.0 + 0.0, 0.0 + 0.5 + 0.0])
    actual_translation = world_to_object.translation.flatten()
    
    assert np.allclose(actual_translation, expected_translation), "Composition failed!"


def test_inverse_traversal():
    """Test traversing edges in reverse direction."""
    
    graph = tf.TransformGraph()
    
    world_to_base = tf.Translation(x=5.0, y=0.0, z=0.0)
    graph.add_transform("world", "base", world_to_base)
    
    # Get forward transform
    forward = graph.get_transform("world", "base")
    
    # Get inverse transform
    backward = graph.get_transform("base", "world")
    
    # Verify inverse
    expected_inverse_translation = np.array([-5.0, 0.0, 0.0])
    actual_inverse_translation = backward.translation.flatten()
    
    assert np.allclose(actual_inverse_translation, expected_inverse_translation), "Inverse failed!"
    
    # Verify composition is identity
    identity_test = graph.get_transform("world", "base") * graph.get_transform("base", "world")
    assert np.allclose(identity_test.as_matrix(), np.eye(4)), "Composition not identity!"


def test_caching():
    """Test that caching works correctly."""
    
    graph = tf.TransformGraph()
    
    # Build a long chain
    graph.add_transform("A", "B", tf.Translation(x=1.0))
    graph.add_transform("B", "C", tf.Translation(x=1.0))
    graph.add_transform("C", "D", tf.Translation(x=1.0))
    graph.add_transform("D", "E", tf.Translation(x=1.0))
    
    # First query - should create cache
    result1 = graph.get_transform("A", "E")
    
    # Second query - should use cache
    result2 = graph.get_transform("A", "E")
    
    # Verify results are the same
    assert np.allclose(result1.as_matrix(), result2.as_matrix()), "Cache returned different result!"


def test_cache_invalidation():
    """Test that cache invalidation works."""
    
    graph = tf.TransformGraph()
    
    graph.add_transform("A", "B", tf.Translation(x=1.0))
    graph.add_transform("B", "C", tf.Translation(x=1.0))
    
    # Query to create cache
    result1 = graph.get_transform("A", "C")
    
    # Update an edge
    graph.update_transform("A", "B", tf.Translation(x=5.0))
    
    # Query again - should recompute
    result2 = graph.get_transform("A", "C")
    
    expected_new = np.array([6.0, 0.0, 0.0])  # 5 + 1
    actual_new = result2.translation.flatten()
    
    assert np.allclose(actual_new, expected_new), "Cache invalidation failed!"


def test_clear_cache():
    """Test clear_cache() method."""
    
    graph = tf.TransformGraph()
    
    graph.add_transform("A", "B", tf.Translation(x=1.0))
    graph.add_transform("B", "C", tf.Translation(x=1.0))
    graph.add_transform("C", "D", tf.Translation(x=1.0))
    
    # Create some caches
    graph.get_transform("A", "C")
    graph.get_transform("A", "D")
    graph.get_transform("B", "D")
    
    # Clear cache
    graph.clear_cache()
    
    # Verify no cache edges remain
    for u, v, data in graph._graph.edges(data=True):
        assert not data.get("is_cache", False), "Cache edge still exists!"


def test_serialization():
    """Test to_dict() and from_dict()."""
    
    graph = tf.TransformGraph()
    
    graph.add_transform("world", "robot", tf.Translation(x=1.0, y=2.0, z=3.0))
    graph.add_transform("robot", "camera", tf.Rotation(w=0.707, z=0.707))
    
    # Create a cache (should NOT be serialized)
    graph.get_transform("world", "camera")
    
    # Serialize
    data = graph.to_dict()
    
    # Deserialize
    graph2 = tf.TransformGraph.from_dict(data)
    
    # Verify same transforms
    orig_transform = graph.get_transform("world", "robot")
    new_transform = graph2.get_transform("world", "robot")
    
    assert np.allclose(orig_transform.as_matrix(), new_transform.as_matrix()), "Serialization mismatch!"


def test_identity_transform():
    """Test that same source and target returns identity."""
    
    graph = tf.TransformGraph()
    graph.add_transform("A", "B", tf.Translation(x=1.0))
    
    identity = graph.get_transform("A", "A")
    
    assert np.allclose(identity.as_matrix(), np.eye(4)), "Not identity!"


def test_error_handling():
    """Test error conditions."""
    
    graph = tf.TransformGraph()
    graph.add_transform("A", "B", tf.Translation(x=1.0))
    
    # Try to add duplicate edge
    with pytest.raises(ValueError, match="Transform between .* already exists"):
        graph.add_transform("A", "B", tf.Translation(x=2.0))
    
    # Try to update non-existent edge
    with pytest.raises(ValueError, match="No transform between .*"):
        graph.update_transform("X", "Y", tf.Translation(x=1.0))
    
    # Try to get transform for non-existent frame
    with pytest.raises(ValueError, match="Frame .* not found in graph"):
        graph.get_transform("Z", "A")
        
    with pytest.raises(ValueError, match="Frame .* not found in graph"):
        graph.get_transform("A", "Z")
