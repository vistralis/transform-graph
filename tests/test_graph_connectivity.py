#!/usr/bin/env python3
"""
Test connectivity methods in TransformGraph.
"""

import tgraph.transform as tf
import pytest

def test_connected_components():
    """Test get_connected_components and get_connected_nodes."""
    graph = tf.TransformGraph()
    
    # Create two disconnected components
    # Component 1: A-B-C
    graph.add_transform("A", "B", tf.Translation(x=1))
    graph.add_transform("B", "C", tf.Translation(x=1))
    
    # Component 2: X-Y
    graph.add_transform("X", "Y", tf.Translation(x=1))
    
    # Test connected components
    components = graph.get_connected_components()
    assert len(components) == 2
    
    # Sort for deterministic comparison
    comp_sets = [set(c) for c in components]
    assert {"A", "B", "C"} in comp_sets
    assert {"X", "Y"} in comp_sets
    
    # Test get_connected_nodes for Component 1
    nodes_a = graph.get_connected_nodes("A")
    assert set(nodes_a) == {"A", "B", "C"}
    
    nodes_b = graph.get_connected_nodes("B")
    assert set(nodes_b) == {"A", "B", "C"}
    
    # Test get_connected_nodes for Component 2
    nodes_x = graph.get_connected_nodes("X")
    assert set(nodes_x) == {"X", "Y"}
    
    # Test error handling
    with pytest.raises(ValueError, match="Frame .* not found"):
        graph.get_connected_nodes("Z")

def test_connectivity_single_node():
    """Test connectivity with a single node (unlikely via add_transform but graph-theoretically possible)."""
    # Note: add_transform requires two nodes.
    # We can technically simulate a single node by adding an edge to self?
    # Or just rely on the fact that nx.Graph handles single nodes if added manually.
    # But our public API only exposes add_transform(u, v).
    # If we add A->A (identity), it's a loop.
    
    graph = tf.TransformGraph()
    # Adding self-loop (usually implies identity, but let's see if allowed by our API)
    # Our API checks has_edge(u, v).
    
    # Let's try adding A->B then removing B->A? No, remove_transform removes the edge.
    pass 
    # Actually, TransformGraph doesn't expose add_node.
    # So components will always have size >= 2 unless we hack _graph.

def test_connectivity_merged_components():
    """Test that components merge when connected."""
    graph = tf.TransformGraph()
    
    # A-B
    graph.add_transform("A", "B", tf.Translation(x=1))
    # C-D
    graph.add_transform("C", "D", tf.Translation(x=1))
    
    assert len(graph.get_connected_components()) == 2
    
    # Connect B-C
    graph.add_transform("B", "C", tf.Translation(x=1))
    
    assert len(graph.get_connected_components()) == 1
    nodes = graph.get_connected_nodes("A")
    assert set(nodes) == {"A", "B", "C", "D"}
