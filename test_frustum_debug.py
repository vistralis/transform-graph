import numpy as np
import sys
sys.path.insert(0, 'src')
import tgraph.transform as tf
import tgraph.visualization as vis

# Patch visualize_transform_graph to add debug output
orig_viz = vis.visualize_transform_graph

def debug_viz(*args, **kwargs):
    print("=== visualize_transform_graph called ===")
    print(f"show_frustums: {kwargs.get('show_frustums', True)}")
    
    # Call original but with custom trace creation
    transform_graph = args[0]
    root_frame = kwargs.get('root_frame')
    show_frustums = kwargs.get('show_frustums', True)
    
    print(f"\nshow_frustums check: {show_frustums}")
    print(f"Edges in graph:")
    for u, v, edge_data in transform_graph.graph.edges(data=True):
        edge_transform = edge_data["transform"]
        print(f"  {u}→{v}: {type(edge_transform).__name__}")
    
    return orig_viz(*args, **kwargs)

vis.visualize_transform_graph = debug_viz

graph = tf.TransformGraph()
graph.add_transform(source_frame="robot", target_frame="world", transform=tf.Translation(x=1.0))
K = np.array([[1757.6, 0.0, 983.2], [0.0, 1761.3, 550.4], [0.0, 0.0, 1.0]])
graph.add_transform(
    source_frame="IMAGE",
    target_frame="robot",
    transform=tf.CameraProjection(intrinsic_matrix=K, image_size=(1920, 1080)).inverse()
)

fig = vis.visualize_transform_graph(graph, root_frame='world', show_frustums=True, frustum_scale=0.5)
print(f"\nTotal traces: {len(fig.data)}")
frustum_traces = [t for t in fig.data if 'frustum' in str(t.name).lower()]
print(f"Frustum traces: {len(frustum_traces)}")
