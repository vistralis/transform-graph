# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization module for TransformGraph using Plotly.
"""
import networkx as nx
import numpy as np
import quaternion
try:
    import plotly.graph_objects as go
except ImportError:
    go = None
from tgraph.transform import CameraProjection, Identity, InverseCameraProjection, InverseProjection, Projection, Transform, TransformGraph
def _check_plotly():
    if go is None:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install it with: pip install 'transform-graph[viz]'"
        )


def _get_heuristic_root(transform_graph: TransformGraph) -> str:
    """
    Determine a suitable root frame using heuristics.

    Priority:
    1. Common names: 'world', 'map', 'odom', 'base_link', 'base', 'root'
    2. Node with highest degree (most connections)
    3. First alphabetical node
    """
    frames = transform_graph.frames
    if not frames:
        raise ValueError("Graph is empty")

    # 1. Check common names
    candidates = ["world", "map", "odom", "base_link", "base", "root", "origin"]
    for candidate in candidates:
        if transform_graph.has_frame(candidate):
            return candidate

    # 2. Max degree
    degrees = dict(transform_graph.graph.degree())
    if degrees:
        return max(degrees, key=degrees.get)

    return frames[0]


def _create_axis_traces(
    transform: Transform, scale: float = 1.0, name: str = "", visible: bool = True
) -> list[go.Scatter3d]:
    """
    Create RGB axes for a specific transform.
    """
    # Get origin
    if hasattr(transform, 'translation'):
        origin = transform.translation.flatten()
    else:
        transform_matrix = transform.as_matrix()
        origin = transform_matrix[:3, 3]
    
    # Get rotation matrix
    transform_matrix = transform.as_matrix()
    rotation_matrix = transform_matrix[:3, :3]

    # Axis end points in local frame
    x_end_local = np.array([scale, 0, 0])
    y_end_local = np.array([0, scale, 0])
    z_end_local = np.array([0, 0, scale])

    # Transform to global frame
    x_end = origin + rotation_matrix @ x_end_local
    y_end = origin + rotation_matrix @ y_end_local
    z_end = origin + rotation_matrix @ z_end_local

    traces = []
    line_width = 5

    # X Axis (Red)
    traces.append(go.Scatter3d(
        x=[origin[0], x_end[0]], y=[origin[1], x_end[1]], z=[origin[2], x_end[2]],
        mode="lines", line=dict(color="red", width=line_width),
        name=f"{name}_X", hoverinfo="text", text=f"{name} X", showlegend=False, visible=visible
    ))

    # Y Axis (Green)
    traces.append(go.Scatter3d(
        x=[origin[0], y_end[0]], y=[origin[1], y_end[1]], z=[origin[2], y_end[2]],
        mode="lines", line=dict(color="green", width=line_width),
        name=f"{name}_Y", hoverinfo="text", text=f"{name} Y", showlegend=False, visible=visible
    ))

    # Z Axis (Blue)
    traces.append(go.Scatter3d(
        x=[origin[0], z_end[0]], y=[origin[1], z_end[1]], z=[origin[2], z_end[2]],
        mode="lines", line=dict(color="blue", width=line_width),
        name=f"{name}_Z", hoverinfo="text", text=f"{name} Z", showlegend=False, visible=visible
    ))

    # Origin marker
    traces.append(go.Scatter3d(
        x=[origin[0]], y=[origin[1]], z=[origin[2]],
        mode="markers", marker=dict(size=3, color="black"),
        name=name, hoverinfo="text", text=name, showlegend=True, visible=visible
    ))

    return traces


def _create_frustum_traces(
    intrinsic: CameraProjection, 
    pose: Transform, 
    scale: float = 1.0, 
    name: str = "", 
    visible: bool = True
) -> list[go.Scatter3d]:
    """
    Create a frustum pyramid for a CameraProjection.

    Args:
        intrinsic: CameraProjection (intrinsic parameters).
        pose: Transform (Camera to World pose).
        scale: Depth of the frustum (distance from camera center to image plane).
        name: Label for the frustum.
        visible: Whether the frustum is initially visible.
    """
    # 1. Get camera position and orientation from Pose (Cam -> World)
    camera_center = pose.translation.flatten()
    R_cam_to_world = quaternion.as_rotation_matrix(pose.rotation)

    # 2. Define image corners in pixel coordinates
    if intrinsic.image_size is not None:
        w, h = intrinsic.image_size
    else:
        # Infer from principal point (assume it's roughly at image center)
        cx, cy = intrinsic.principal_point
        w = int(2 * cx) if cx > 0 else 640
        h = int(2 * cy) if cy > 0 else 480

    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    # 3. Unproject corners to normalized camera frame directions
    K = intrinsic.intrinsic_matrix
    K_inv = np.linalg.inv(K)

    corners_cam_frame = []
    for u, v in corners_2d:
        # Get ray direction in camera frame
        ray_dir = K_inv @ np.array([u, v, 1])
        # Normalize and scale to desired depth
        ray_dir = ray_dir / np.linalg.norm(ray_dir) * scale
        corners_cam_frame.append(ray_dir)

    # 4. Transform corners from camera frame to world frame
    # X_world = C + R_cam_to_world @ X_cam
    corners_world = [camera_center + R_cam_to_world @ p for p in corners_cam_frame]

    traces = []

    # Draw pyramid edges
    for p in corners_world:
        traces.append(go.Scatter3d(
            x=[camera_center[0], p[0]], y=[camera_center[1], p[1]], z=[camera_center[2], p[2]],
            mode="lines", line=dict(color="orange", width=2),
            showlegend=False, visible=visible, hoverinfo="none"
        ))

    # Draw the image plane rectangle
    cx = [p[0] for p in corners_world] + [corners_world[0][0]]
    cy = [p[1] for p in corners_world] + [corners_world[0][1]]
    cz = [p[2] for p in corners_world] + [corners_world[0][2]]

    traces.append(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="lines", line=dict(color="orange", width=3),
        name=f"{name}_frustum", showlegend=False, visible=visible, hoverinfo="text",
        text=f"{name} Frustum"
    ))

    return traces


def visualize_transform_graph(
    transform_graph: TransformGraph,
    root_frame: str | None = None,
    axis_scale: float = 1.0,
    show_connections: bool = True,
    show_frustums: bool = True,
    frustum_scale: float = 1.0,
    title: str = "Transform Graph Visualization (3D)",
) -> "go.Figure":
    """
    Visualize the transform graph in 3D.
    """
    _check_plotly()

    if not transform_graph.frames:
        return go.Figure(layout=dict(title="Empty Graph"))

    if root_frame is None:
        root_frame = _get_heuristic_root(transform_graph)

    if not transform_graph.has_frame(root_frame):
        raise ValueError(f"Root frame '{root_frame}' not found in graph.")

    traces = []
    frame_transforms = {}
    projection_map_2d_to_3d = {}

    # Pre-pass: Identify projection edges which are directional (3D -> 2D)
    # We want 2D frames (Image) to use the position of 3D frames (Camera) for visualization
    for u, v, edge_data in transform_graph.graph.edges(data=True):
        tr = edge_data["transform"]
        if isinstance(tr, (Projection, InverseProjection)):
            reference_frame = edge_data.get("reference_frame")
            other_frame = v if reference_frame == u else u
            
            target_2d = None
            source_3d = None
            
            # Logic: Identify which frame is 2D and which is 3D
            if isinstance(tr, InverseProjection) and not isinstance(tr, CameraProjection):
                # InverseProjection (Source=2D, Target=3D)
                # Edge: reference_frame=Target(3D), other=Source(2D)
                target_2d = other_frame
                source_3d = reference_frame
            else:
                # Projection/CameraProjection (Source=3D, Target=2D)
                # Edge: reference_frame=Target(2D), other=Source(3D)
                target_2d = reference_frame
                source_3d = other_frame
            
            if target_2d and source_3d:
                projection_map_2d_to_3d[target_2d] = source_3d

    # Compute global transforms relative to root
    for frame_id in transform_graph.frames:
        try:
            if frame_id == root_frame:
                transform = Identity()
            else:
                target_frame_for_transform = frame_id
                if frame_id in projection_map_2d_to_3d:
                    target_frame_for_transform = projection_map_2d_to_3d[frame_id]
                
                transform = transform_graph.get_transform(target_frame_for_transform, root_frame)
            
            frame_transforms[frame_id] = transform
        except ValueError:
            pass

    # Create traces
    for frame_id, transform in frame_transforms.items():
        traces.extend(_create_axis_traces(transform, scale=axis_scale, name=frame_id))

        if show_frustums:
            # Check for camera edges connected to this frame (this frame as 3D Camera)
            for u, v, edge_data in transform_graph.graph.edges(data=True):
                # We need to determine if frame_id is the 3D Camera side of this edge.
                
                if u != frame_id and v != frame_id:
                    continue

                if u != frame_id and v != frame_id:
                    continue

                reference_frame = edge_data.get("reference_frame")
                other_frame = v if reference_frame == u else u
                
                edge_transform = edge_data["transform"]
                intrinsic_to_draw = None # Only set if frame_id is the CAMERA (3D) side

                # Case 1: Edge holds InverseCameraProjection (Image -> Camera)
                # Transform Source=Image, Target=Camera.
                # add_transform(Image, Camera, InvProj) -> reference_frame=Camera, other=Image.
                # So Camera is reference_frame.
                if isinstance(edge_transform, InverseCameraProjection):
                     if reference_frame == frame_id:
                         intrinsic_to_draw = edge_transform.camera_projection

                # Case 2: Edge holds CameraProjection (Camera -> Image)
                # Transform Source=Camera, Target=Image.
                # add_transform(Camera, Image, Proj) -> reference_frame=Image, other=Camera.
                # So Camera is other_frame.
                elif isinstance(edge_transform, CameraProjection):
                     if other_frame == frame_id:
                         intrinsic_to_draw = edge_transform
                
                if intrinsic_to_draw is not None:
                    # Draw frustum using the Camera Frame's pose
                    pose = frame_transforms[frame_id] # This is T_cam_to_root (Pose)
                    
                    # Ensure it is a Transform (has rotation/translation props)
                    if hasattr(pose, 'rotation') and hasattr(pose, 'translation'):
                        frustum_traces = _create_frustum_traces(
                            intrinsic=intrinsic_to_draw,
                            pose=pose,
                            scale=frustum_scale,
                            name=frame_id
                        )
                        traces.extend(frustum_traces)
                        # Don't break, allowing multiple frustums if multiple cameras attached (though unlikely for one frame)

    # Create traces for connections
    if show_connections:
        edge_x = []
        edge_y = []
        edge_z = []

        for u, v in transform_graph.edges:
            if u in frame_transforms and v in frame_transforms:
                # Get translation, handling objects without .translation property
                u_transform = frame_transforms[u]
                v_transform = frame_transforms[v]
                
                if hasattr(u_transform, 'translation'):
                    p1 = u_transform.translation.flatten()
                else:
                    p1 = u_transform.as_matrix()[:3, 3]
                
                if hasattr(v_transform, 'translation'):
                    p2 = v_transform.translation.flatten()
                else:
                    p2 = v_transform.as_matrix()[:3, 3]

                edge_x.extend([p1[0], p2[0], None])
                edge_y.extend([p1[1], p2[1], None])
                edge_z.extend([p1[2], p2[2], None])

        if edge_x:
            traces.append(
                go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode="lines",
                    line=dict(color="grey", width=2, dash="dash"),
                    name="Connections",
                    hoverinfo="none",
                )
            )

    # Layout
    layout = go.Layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0, y=1),
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


def visualize_graph(
    transform_graph: TransformGraph, title: str = "Graph Topology (2D)"
) -> "go.Figure":
    """
    Visualize the graph topology in 2D using NetworkX layout.

    Shows nodes, edges, transform types, and caching status.

    Args:
        transform_graph: The TransformGraph instance.
        title: Plot title.

    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    # Access internal networkx graph to get all edges (including cached)
    G = transform_graph.graph

    if G.number_of_nodes() == 0:
        return go.Figure(layout=dict(title="Empty Graph"))

    # Compute 2D layout
    pos = nx.spring_layout(G, seed=42)

    # Create traces
    edge_traces = []
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode="markers+text",
        text=[],
        textposition="top center",
        hoverinfo="text",
        marker=dict(showscale=False, color="skyblue", size=20, line_width=2),
        name="Frames",
    )

    # Edges
    # We iterate edges to create lines and annotations
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        is_cache = data.get("is_cache", False)
        transform = data.get("transform")
        transform_type = type(transform).__name__ if transform else "Unknown"

        # Style
        color = "lightgrey" if is_cache else "#888"
        width = 1 if is_cache else 2
        dash = "dot" if is_cache else "solid"

        # Edge line
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color, dash=dash),
                hoverinfo="none",
                showlegend=False,
            )
        )

        # Annotation (middle of edge)
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2

        label = f"{transform_type}"
        if is_cache:
            label += " *cached"

        edge_traces.append(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=[label],
                textposition="middle center",
                textfont=dict(size=8, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Nodes
    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)
        node_trace["text"] += (node,)

    # Combine traces (edges first so nodes are on top)
    data = edge_traces + [node_trace]

    layout = go.Layout(
        title=title,
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )

    return go.Figure(data=data, layout=layout)
