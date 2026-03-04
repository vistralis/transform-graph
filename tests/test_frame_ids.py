import json
import uuid
from datetime import UTC, datetime

import numpy as np

from tgraph.transform import Transform, TransformGraph


def test_integer_frame_ids():
    """Verify that integer frame IDs work correctly."""
    graph = TransformGraph()

    # Frames as integers
    frame_a = 1
    frame_b = 2
    frame_c = 3

    # Transform A -> B: Translation(1, 0, 0)
    t_ab = Transform(translation=[1, 0, 0])
    graph.add_transform(frame_a, frame_b, t_ab)

    # Transform B -> C: Translation(0, 1, 0)
    t_bc = Transform(translation=[0, 1, 0])
    graph.add_transform(frame_b, frame_c, t_bc)

    # Check A -> C: Should be Translation(1, 1, 0)
    t_ac = graph.get_transform(frame_a, frame_c)

    expected_pos = np.array([1.0, 1.0, 0.0])
    assert np.allclose(t_ac.translation.flatten(), expected_pos)

    # Check frames listing
    frames = graph.frames
    assert frame_a in frames
    assert frame_b in frames
    assert frame_c in frames


def test_uuid_frame_ids():
    """Verify that UUID frame IDs work correctly."""
    graph = TransformGraph()

    # Frames as UUIDs
    frame_a = uuid.uuid4()
    frame_b = uuid.uuid4()

    # Transform A -> B
    t_ab = Transform(translation=[2.0, 0.0, 0.0])
    graph.add_transform(frame_a, frame_b, t_ab)

    # Retrieve
    t_retrieved = graph.get_transform(frame_a, frame_b)
    assert np.allclose(t_retrieved.translation.flatten(), t_ab.translation.flatten())

    # Inverse retrieval
    t_inv = graph.get_transform(frame_b, frame_a)
    assert np.allclose(t_inv.translation.flatten(), [-2.0, 0.0, 0.0])


def test_mixed_frame_ids():
    """Verify mixing string, int, and UUID frame IDs."""
    graph = TransformGraph()

    frame_str = "root"
    frame_int = 42
    frame_uuid = uuid.uuid4()

    # Str -> Int
    graph.add_transform(frame_str, frame_int, Transform(translation=[1, 0, 0]))

    # Int -> UUID
    graph.add_transform(frame_int, frame_uuid, Transform(translation=[0, 1, 0]))

    # Str -> UUID
    t_res = graph.get_transform(frame_str, frame_uuid)

    assert np.allclose(t_res.translation.flatten(), [1.0, 1.0, 0.0])


def test_timestamp_frame_ids():
    """Verify that datetime timestamp frame IDs work for temporal graphs."""
    graph = TransformGraph()

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, 2, tzinfo=UTC)

    graph.add_transform(t0, t1, Transform(translation=[1.0, 0.0, 0.0]))
    graph.add_transform(t1, t2, Transform(translation=[0.0, 2.0, 0.0]))

    # Chain: t0 -> t2
    t_02 = graph.get_transform(t0, t2)
    assert np.allclose(t_02.translation.flatten(), [1.0, 2.0, 0.0])

    # Inverse: t2 -> t0
    t_20 = graph.get_transform(t2, t0)
    assert np.allclose(t_20.translation.flatten(), [-1.0, -2.0, 0.0])

    # Frames listing
    assert t0 in graph.frames
    assert t1 in graph.frames
    assert t2 in graph.frames


def test_float_frame_ids():
    """Verify that float frame IDs work (e.g., Unix timestamps)."""
    graph = TransformGraph()

    t0 = 1735689600.0  # Unix timestamp
    t1 = 1735689601.0
    t2 = 1735689602.0

    graph.add_transform(t0, t1, Transform(translation=[1.0, 0.0, 0.0]))
    graph.add_transform(t1, t2, Transform(translation=[0.0, 1.0, 0.0]))

    t_02 = graph.get_transform(t0, t2)
    assert np.allclose(t_02.translation.flatten(), [1.0, 1.0, 0.0])

    assert t0 in graph.frames
    assert t2 in graph.frames


def test_tuple_frame_ids():
    """Verify that tuple frame IDs work (e.g., (sensor, timestamp) pairs)."""
    graph = TransformGraph()

    frame_a = ("lidar", 0)
    frame_b = ("ego", 0)
    frame_c = ("ego", 1)

    graph.add_transform(frame_a, frame_b, Transform(translation=[1.0, 0.0, 2.0]))
    graph.add_transform(frame_b, frame_c, Transform(translation=[3.0, 0.0, 0.0]))

    # Chain: ("lidar", 0) -> ("ego", 1)
    t_ac = graph.get_transform(frame_a, frame_c)
    assert np.allclose(t_ac.translation.flatten(), [4.0, 0.0, 2.0])

    # Frames listing
    assert frame_a in graph.frames
    assert frame_b in graph.frames
    assert frame_c in graph.frames


# ---------------------------------------------------------------------------
# JSON Serialization Roundtrip Tests
#
# Verifies that a graph survives: to_dict → json.dumps → json.loads → from_dict
# This is the critical path for sending graphs via HTTP requests.
# ---------------------------------------------------------------------------


def _json_roundtrip(graph: TransformGraph) -> TransformGraph:
    """Serialize a graph to JSON string and back."""
    data = graph.to_dict()
    json_string = json.dumps(data)
    restored_data = json.loads(json_string)
    return TransformGraph.from_dict(restored_data)


def test_json_roundtrip_string_frames():
    """JSON roundtrip preserves string frame IDs and transform values."""
    graph = TransformGraph()
    graph.add_transform("world", "robot", Transform(translation=[1.0, 2.0, 3.0]))
    graph.add_transform("robot", "sensor", Transform(translation=[0.5, 0.0, 1.0]))

    restored = _json_roundtrip(graph)

    # Same frame set
    assert set(restored.frames) == {"world", "robot", "sensor"}

    # Transform values preserved
    t_orig = graph.get_transform("world", "sensor")
    t_rest = restored.get_transform("world", "sensor")
    assert np.allclose(t_orig.as_matrix(), t_rest.as_matrix())


def test_json_roundtrip_integer_frames():
    """JSON roundtrip preserves integer frame IDs."""
    graph = TransformGraph()
    graph.add_transform(1, 2, Transform(translation=[1.0, 0.0, 0.0]))
    graph.add_transform(2, 3, Transform(translation=[0.0, 1.0, 0.0]))

    restored = _json_roundtrip(graph)

    # Integers survive JSON roundtrip natively
    assert 1 in restored.frames
    assert 2 in restored.frames
    assert 3 in restored.frames

    t_orig = graph.get_transform(1, 3)
    t_rest = restored.get_transform(1, 3)
    assert np.allclose(t_orig.as_matrix(), t_rest.as_matrix())


def test_json_roundtrip_float_frames():
    """JSON roundtrip preserves float frame IDs (Unix timestamps)."""
    graph = TransformGraph()
    graph.add_transform(1735689600.0, 1735689601.0, Transform(translation=[5.0, 0.0, 0.0]))

    restored = _json_roundtrip(graph)

    assert 1735689600.0 in restored.frames
    assert 1735689601.0 in restored.frames

    t_rest = restored.get_transform(1735689600.0, 1735689601.0)
    assert np.allclose(t_rest.translation.flatten(), [5.0, 0.0, 0.0])


def test_json_roundtrip_tuple_frames():
    """JSON roundtrip preserves tuple frame IDs.

    JSON has no tuple type — json.dumps converts tuples to JSON arrays,
    which json.loads deserializes as Python lists. TransformGraph.from_dict
    coerces lists back to tuples so compound frame IDs like ("ego", 0)
    survive the full roundtrip.
    """
    graph = TransformGraph()
    graph.add_transform(("lidar", 0), ("ego", 0), Transform(translation=[1.0, 0.0, 2.0]))
    graph.add_transform(("ego", 0), ("ego", 1), Transform(translation=[3.0, 0.0, 0.0]))

    restored = _json_roundtrip(graph)

    # Tuples survive roundtrip via list→tuple coercion
    assert ("lidar", 0) in restored.frames
    assert ("ego", 0) in restored.frames
    assert ("ego", 1) in restored.frames

    # Chain composition preserved
    t_orig = graph.get_transform(("lidar", 0), ("ego", 1))
    t_rest = restored.get_transform(("lidar", 0), ("ego", 1))
    assert np.allclose(t_orig.as_matrix(), t_rest.as_matrix())


def test_json_roundtrip_mixed_frames():
    """JSON roundtrip with mixed string + integer frame IDs."""
    graph = TransformGraph()
    graph.add_transform("global", 0, Transform(translation=[10.0, 0.0, 0.0]))
    graph.add_transform(0, 1, Transform(translation=[3.0, 0.0, 0.0]))
    graph.add_transform(1, "sensor", Transform(translation=[0.5, 0.5, 0.0]))

    restored = _json_roundtrip(graph)

    assert set(restored.frames) == {"global", 0, 1, "sensor"}

    t_orig = graph.get_transform("global", "sensor")
    t_rest = restored.get_transform("global", "sensor")
    assert np.allclose(t_orig.as_matrix(), t_rest.as_matrix())


def test_json_roundtrip_temporal_ego_chain():
    """JSON roundtrip of a temporal ego chain using integer timestamps.

    Simulates the pattern used by real dataset builders: ego frames named by
    integer timestamp, chained with relative transforms, one anchored to global.
    """
    graph = TransformGraph()

    timestamps = [1000000, 1000100, 1000200, 1000300, 1000400]

    # Chain ego frames
    for i in range(len(timestamps) - 1):
        graph.add_transform(
            timestamps[i],
            timestamps[i + 1],
            Transform(translation=[2.0, 0.1 * i, 0.0]),
        )

    # Anchor first to global
    graph.add_transform(timestamps[0], "global", Transform(translation=[0.0, 0.0, 0.0]))

    # Roundtrip
    restored = _json_roundtrip(graph)

    # Verify chain composition survives
    t_orig = graph.get_transform(timestamps[0], timestamps[-1])
    t_rest = restored.get_transform(timestamps[0], timestamps[-1])
    assert np.allclose(t_orig.as_matrix(), t_rest.as_matrix(), atol=1e-12)

    # Verify global anchor
    t_global_orig = graph.get_transform(timestamps[-1], "global")
    t_global_rest = restored.get_transform(timestamps[-1], "global")
    assert np.allclose(t_global_orig.as_matrix(), t_global_rest.as_matrix(), atol=1e-12)


def test_json_roundtrip_datetime_frames():
    """JSON roundtrip preserves datetime frame IDs via ISO 8601 encoding."""
    graph = TransformGraph()

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 12, 0, 1, tzinfo=UTC)

    graph.add_transform(t0, t1, Transform(translation=[3.0, 0.0, 0.0]))

    restored = _json_roundtrip(graph)

    assert t0 in restored.frames
    assert t1 in restored.frames

    t_rest = restored.get_transform(t0, t1)
    assert np.allclose(t_rest.translation.flatten(), [3.0, 0.0, 0.0])


def test_json_roundtrip_uuid_frames():
    """JSON roundtrip preserves UUID frame IDs."""
    graph = TransformGraph()

    frame_a = uuid.uuid4()
    frame_b = uuid.uuid4()

    graph.add_transform(frame_a, frame_b, Transform(translation=[1.0, 2.0, 3.0]))

    restored = _json_roundtrip(graph)

    assert frame_a in restored.frames
    assert frame_b in restored.frames

    t_rest = restored.get_transform(frame_a, frame_b)
    assert np.allclose(t_rest.translation.flatten(), [1.0, 2.0, 3.0])


def test_json_roundtrip_compound_ego_datetime_tuple():
    """JSON roundtrip of ("ego", datetime) compound frame IDs.

    This is the canonical pattern for temporal vehicle graphs: each ego pose
    is identified by a (sensor_name, timestamp) tuple. Tests that both the
    tuple structure and the datetime values survive the full JSON pipeline.
    """
    graph = TransformGraph()

    t0 = datetime(2026, 3, 5, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 3, 5, 12, 0, 1, tzinfo=UTC)
    t2 = datetime(2026, 3, 5, 12, 0, 2, tzinfo=UTC)

    # Ego chain with compound (name, datetime) keys
    graph.add_transform(("ego", t0), ("ego", t1), Transform(translation=[2.0, 0.0, 0.0]))
    graph.add_transform(("ego", t1), ("ego", t2), Transform(translation=[2.0, 0.1, 0.0]))

    # Sensor attached to first ego
    graph.add_transform(("cam_front", t0), ("ego", t0), Transform(translation=[1.5, 0.0, 1.2]))

    # Global anchor
    graph.add_transform(("ego", t0), "global", Transform(translation=[100.0, 200.0, 0.0]))

    restored = _json_roundtrip(graph)

    # All compound frames survived
    assert ("ego", t0) in restored.frames
    assert ("ego", t1) in restored.frames
    assert ("ego", t2) in restored.frames
    assert ("cam_front", t0) in restored.frames
    assert "global" in restored.frames

    # Chain composition preserved
    t_chain_orig = graph.get_transform(("ego", t0), ("ego", t2))
    t_chain_rest = restored.get_transform(("ego", t0), ("ego", t2))
    assert np.allclose(t_chain_orig.as_matrix(), t_chain_rest.as_matrix(), atol=1e-12)

    # Cross-type path: cam -> global
    t_cam_orig = graph.get_transform(("cam_front", t0), "global")
    t_cam_rest = restored.get_transform(("cam_front", t0), "global")
    assert np.allclose(t_cam_orig.as_matrix(), t_cam_rest.as_matrix(), atol=1e-12)


def test_datetime64_frame_ids():
    """Verify that np.datetime64 frame IDs work with nanosecond precision."""
    graph = TransformGraph()

    t0 = np.datetime64("2026-03-05T13:20:51.123456789", "ns")
    t1 = np.datetime64("2026-03-05T13:20:52.123456789", "ns")

    graph.add_transform(t0, t1, Transform(translation=[0.1, 0.0, 0.0]))

    t_res = graph.get_transform(t0, t1)
    assert np.allclose(t_res.translation.flatten(), [0.1, 0.0, 0.0])
    assert t0 in graph.frames


def test_json_roundtrip_datetime64_frames():
    """JSON roundtrip preserves np.datetime64 frame IDs with unit metadata."""
    graph = TransformGraph()

    t0 = np.datetime64("2026-03-05T13:20:51.123456789", "ns")
    t1 = np.datetime64("2026-03-05T13:20:52.123456789", "ns")

    graph.add_transform(t0, t1, Transform(translation=[0.1, 0.0, 0.0]))

    restored = _json_roundtrip(graph)

    assert t0 in restored.frames
    assert t1 in restored.frames

    t_rest = restored.get_transform(t0, t1)
    assert np.allclose(t_rest.translation.flatten(), [0.1, 0.0, 0.0])


def test_json_roundtrip_compound_datetime64_tuple():
    """JSON roundtrip of ("base_link", np.datetime64) compound frame IDs.

    Tests the nanosecond-precision temporal pattern used in ROS2-style graphs.
    """
    graph = TransformGraph()

    t0 = np.datetime64("2026-03-05T13:20:51.000000000", "ns")
    t1 = np.datetime64("2026-03-05T13:20:52.000000000", "ns")

    graph.add_transform(
        ("base_link", t0), ("base_link", t1), Transform(translation=[0.1, 0.0, 0.0])
    )
    graph.add_transform(("base_link", t0), "world", Transform(translation=[1.0, 2.0, 0.0]))

    restored = _json_roundtrip(graph)

    assert ("base_link", t0) in restored.frames
    assert ("base_link", t1) in restored.frames
    assert "world" in restored.frames

    t_orig = graph.get_transform(("base_link", t1), "world")
    t_rest = restored.get_transform(("base_link", t1), "world")
    assert np.allclose(t_orig.as_matrix(), t_rest.as_matrix(), atol=1e-12)
