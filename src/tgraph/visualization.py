"""
Visualization module for TransformGraph using Plotly.
"""

import networkx as nx
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from tgraph.transform import Transform, TransformGraph


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
    # Use matrix form to be robust against different Transform types (e.g. MatrixTransform)
    transform_matrix = transform.as_matrix()
    origin = transform_matrix[:3, 3]
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

    # Common attributes
    line_width = 5

    # X Axis (Red)
    traces.append(
        go.Scatter3d(
            x=[origin[0], x_end[0]],
            y=[origin[1], x_end[1]],
            z=[origin[2], x_end[2]],
            mode="lines",
            line=dict(color="red", width=line_width),
            name=f"{name}_X",
            hoverinfo="text",
            text=f"{name} X",
            showlegend=False,
            visible=visible,
        )
    )

    # Y Axis (Green)
    traces.append(
        go.Scatter3d(
            x=[origin[0], y_end[0]],
            y=[origin[1], y_end[1]],
            z=[origin[2], y_end[2]],
            mode="lines",
            line=dict(color="green", width=line_width),
            name=f"{name}_Y",
            hoverinfo="text",
            text=f"{name} Y",
            showlegend=False,
            visible=visible,
        )
    )

    # Z Axis (Blue)
    traces.append(
        go.Scatter3d(
            x=[origin[0], z_end[0]],
            y=[origin[1], z_end[1]],
            z=[origin[2], z_end[2]],
            mode="lines",
            line=dict(color="blue", width=line_width),
            name=f"{name}_Z",
            hoverinfo="text",
            text=f"{name} Z",
            showlegend=False,
            visible=visible,
        )
    )

    # Origin marker (for hover info)
    traces.append(
        go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode="markers",
            marker=dict(size=3, color="black"),
            name=name,
            hoverinfo="text",
            text=name,
            showlegend=True,
            visible=visible,
        )
    )

    return traces


def visualize_transform_graph(
    transform_graph: TransformGraph,
    root_frame: str | None = None,
    axis_scale: float = 1.0,
    show_connections: bool = True,
    title: str = "Transform Graph Visualization (3D)",
) -> "go.Figure":
    """
    Visualize the transform graph in 3D.

    Args:
        transform_graph: The TransformGraph instance.
        root_frame: The frame to use as the origin (0,0,0).
                    If None, determined heuristically.
        axis_scale: Length of the RGB axes for each frame.
        show_connections: Whether to draw lines between connected frames.
        title: Plot title.

    Returns:
        plotly.graph_objects.Figure
    """
    _check_plotly()

    if not transform_graph.frames:
        return go.Figure(layout=dict(title="Empty Graph"))

    if root_frame is None:
        root_frame = _get_heuristic_root(transform_graph)

    if not transform_graph.has_frame(root_frame):
        raise ValueError(f"Root frame '{root_frame}' not found in graph.")

    traces = []

    # Map frame_id -> Transform (relative to root)
    frame_transforms = {}

    # Calculate global transforms for all frames reachable from root
    # We rely on the graph's internal pathfinding and caching
    for frame_id in transform_graph.frames:
        try:
            # get_transform handles finding the path and composing transforms
            transform = transform_graph.get_transform(root_frame, frame_id)
            frame_transforms[frame_id] = transform
        except ValueError:
            # Frame is part of a disconnected component not reachable from root
            pass

    # Create traces for frames
    for frame_id, transform in frame_transforms.items():
        traces.extend(_create_axis_traces(transform, scale=axis_scale, name=frame_id))

    # Create traces for connections
    if show_connections:
        edge_x = []
        edge_y = []
        edge_z = []

        for u, v in transform_graph.edges:
            if u in frame_transforms and v in frame_transforms:
                p1 = frame_transforms[u].translation.flatten()
                p2 = frame_transforms[v].translation.flatten()

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
