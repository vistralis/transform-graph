#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for serialization: registry, to_dict/from_dict round-trips, from_matrix.
"""

import numpy as np
import pytest

import tgraph.transform as tf

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestSerializationRegistry:
    """Test that all transform types are registered and can roundtrip."""

    def test_all_types_registered(self):
        """All concrete types exist in the global registry."""
        types_to_check = [
            "Transform",
            "Projection",
            "CameraProjection",
            "InverseProjection",
            "InverseCameraProjection",
            "MatrixTransform",
            "OrthographicProjection",
            "InverseOrthographicProjection",
            "CompositeProjection",
            "InverseCompositeProjection",
        ]
        for tname in types_to_check:
            assert tname in tf._TRANSFORM_REGISTRY, f"{tname} not registered"

    def test_serialize_deserialize_all_types(self):
        """serialize_transform/deserialize_transform roundtrips for all types."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)

        instances = [
            tf.Transform(translation=[1, 2, 3], rotation=[1, 0, 0, 0]),
            tf.Rotation(w=1, x=0, y=0, z=0),
            tf.Translation(x=1.0, y=2.0, z=3.0),
            tf.Identity(),
            tf.CameraProjection(K=K),
            tf.MatrixTransform(np.eye(4)),
        ]

        for inst in instances:
            data = tf.serialize_transform(inst)
            recovered = tf.deserialize_transform(data)
            np.testing.assert_allclose(
                recovered.as_matrix(),
                inst.as_matrix(),
                atol=1e-10,
                err_msg=f"Roundtrip failed for {type(inst).__name__}",
            )

    def test_unknown_type_error(self):
        """deserialize_transform raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown transform type"):
            tf.deserialize_transform({"type": "NonExistentType"})


# ---------------------------------------------------------------------------
# TransformGraph serialization
# ---------------------------------------------------------------------------


class TestTransformGraphSerialization:
    """Test TransformGraph to_dict/from_dict."""

    def test_basic_roundtrip(self):
        """to_dict → from_dict preserves transform values."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "robot", tf.Translation(x=1.0, y=2.0, z=3.0))
        graph.add_transform("robot", "camera", tf.Rotation(w=0.707, z=0.707))

        data = graph.to_dict()
        graph2 = tf.TransformGraph.from_dict(data)

        orig = graph.get_transform("world", "robot")
        recovered = graph2.get_transform("world", "robot")
        np.testing.assert_allclose(recovered.as_matrix(), orig.as_matrix())

    def test_mixed_types_roundtrip(self, K_simple):
        """to_dict/from_dict preserves Translation + Rotation + CameraProjection."""
        graph = tf.TransformGraph()
        graph.add_transform("world", "robot", tf.Translation(x=3.0, y=4.0, z=5.0))
        graph.add_transform("robot", "camera", tf.Rotation.from_roll_pitch_yaw(yaw=np.pi / 2))
        graph.add_transform("camera", "image", tf.CameraProjection(K=K_simple))

        data = graph.to_dict()
        graph2 = tf.TransformGraph.from_dict(data)

        for source, target in [("world", "robot"), ("robot", "camera")]:
            orig = graph.get_transform(source, target)
            recovered = graph2.get_transform(source, target)
            np.testing.assert_allclose(
                recovered.as_matrix(),
                orig.as_matrix(),
                atol=1e-10,
            )

    def test_repr_counts(self):
        """__repr__ reports correct node and edge counts."""
        graph = tf.TransformGraph()
        graph.add_transform("A", "B", tf.Translation(x=1))
        graph.add_transform("B", "C", tf.Translation(x=1))
        s = repr(graph)
        assert "nodes=3" in s or "3 frames" in s
        assert "edges=2" in s or "2 edges" in s

    def test_repr_empty(self):
        """__repr__ for an empty graph."""
        s = repr(tf.TransformGraph())
        assert "Empty" in s or "0" in s
