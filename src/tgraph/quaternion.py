# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quaternion convention conversion utilities.

Provides lossless conversion between numpy-quaternion (wxyz),
scipy.spatial.transform.Rotation, and raw arrays in either
wxyz or xyzw layout.

All functions operate on single quaternions — no batch support.

Convention reference:

- **wxyz** (Hamilton): ``[w, x, y, z]`` — used by numpy-quaternion
- **xyzw** (JPL / ROS / scipy): ``[x, y, z, w]`` — used by scipy, ROS tf2

Example::

    import quaternion as npq
    from tgraph.quaternion import to_xyzw, from_xyzw, normalize

    q = npq.quaternion(0.7071, 0.0, 0.7071, 0.0)
    arr = to_xyzw(q)       # [0.0, 0.7071, 0.0, 0.7071]
    q2 = from_xyzw(arr)    # round-trip
    q3 = normalize(q2)     # ensure unit length
"""

from __future__ import annotations

import numpy as np
import quaternion as npq
from scipy.spatial.transform import Rotation as ScipyRotation


def to_xyzw(q: npq.quaternion) -> np.ndarray:
    """Convert a numpy-quaternion to a ``[x, y, z, w]`` array.

    Args:
        q: A numpy-quaternion instance.

    Returns:
        np.ndarray: Shape ``(4,)`` array in ``[x, y, z, w]`` order.
    """
    return np.array([q.x, q.y, q.z, q.w])


def from_xyzw(array: np.ndarray | list | tuple) -> npq.quaternion:
    """Create a numpy-quaternion from a ``[x, y, z, w]`` array.

    Args:
        array: 4-element sequence in ``[x, y, z, w]`` order.

    Returns:
        quaternion.quaternion: The corresponding quaternion.
    """
    a = np.asarray(array, dtype=np.float64).ravel()
    if a.size != 4:
        raise ValueError(f"Expected 4 elements, got {a.size}")
    x, y, z, w = a
    return npq.quaternion(w, x, y, z)


def to_wxyz(q: npq.quaternion) -> np.ndarray:
    """Convert a numpy-quaternion to a ``[w, x, y, z]`` array.

    Args:
        q: A numpy-quaternion instance.

    Returns:
        np.ndarray: Shape ``(4,)`` array in ``[w, x, y, z]`` order.
    """
    return np.array([q.w, q.x, q.y, q.z])


def from_wxyz(array: np.ndarray | list | tuple) -> npq.quaternion:
    """Create a numpy-quaternion from a ``[w, x, y, z]`` array.

    Args:
        array: 4-element sequence in ``[w, x, y, z]`` order.

    Returns:
        quaternion.quaternion: The corresponding quaternion.
    """
    a = np.asarray(array, dtype=np.float64).ravel()
    if a.size != 4:
        raise ValueError(f"Expected 4 elements, got {a.size}")
    return npq.quaternion(*a)


def to_scipy(q: npq.quaternion) -> ScipyRotation:
    """Convert a numpy-quaternion to a scipy Rotation.

    Args:
        q: A numpy-quaternion instance (unit quaternion expected).

    Returns:
        scipy.spatial.transform.Rotation: The equivalent rotation.
    """
    return ScipyRotation.from_quat([q.x, q.y, q.z, q.w])


def from_scipy(rotation: ScipyRotation) -> npq.quaternion:
    """Convert a scipy Rotation to a numpy-quaternion.

    Args:
        rotation: A scipy Rotation instance.

    Returns:
        quaternion.quaternion: The equivalent quaternion.
    """
    x, y, z, w = rotation.as_quat()
    return npq.quaternion(w, x, y, z)


def normalize(q: npq.quaternion) -> npq.quaternion:
    """Normalize a quaternion to unit length.

    Args:
        q: A numpy-quaternion instance.

    Returns:
        quaternion.quaternion: The unit quaternion.

    Raises:
        ValueError: If the quaternion has zero norm (cannot represent a rotation).
    """
    norm = np.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
    if norm < 1e-15:
        raise ValueError(
            "Zero-norm quaternion cannot be normalized — it does not represent a valid rotation."
        )
    return q / norm
