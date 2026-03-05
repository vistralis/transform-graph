# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-03-05

### Added
- 7 camera projection models: Pinhole, BrownConrady, KannalaBrandt, Rational, Division, MeiUnified, Fisheye62.
- Per-model projection dispatch in `CameraProjection._apply()`.
- `ProjectionModel.from_string()` with aliases for ROS `distortion_model` names and legacy names.
- Distortion-aware `transform_points` for `CameraProjection` — returns `[u·z, v·z, z]`.
- `_get_padded_dist_coeffs()` helper for safe coefficient extraction.
- 34 new tests in `test_projection_models.py`.

### Changed
- `ProjectionModel` enum rewritten to PascalCase (was snake_case).
- Numerical hardening: named constants (`_DEPTH_EPS`, `_RADIAL_EPS`, `_NORM_EPS`, `_DENOM_EPS`), Horner form polynomial evaluation, safe divisions.
- No exact float comparisons — all use `np.abs(value) < epsilon` guards.

### Removed
- Copilot review workflow (requires Enterprise, not available on Pro).
- Autogeneration notice from README.

## [0.1.0] — 2026-03-04

Initial release — spatial transformation library for robotics and computer vision.

### Added
- `Transform`, `Rotation`, `Translation`, `MatrixTransform`, `Identity` types.
- `CameraProjection`, `OrthographicProjection` with inverse support.
- `CompositeProjection` preserving intrinsic/extrinsic separation.
- `TransformGraph` with BFS pathfinding and automatic SE(3) composition.
- Serialization (to_dict / from_dict) for all transform types and graphs.
- Visualization via Plotly.
- 231 tests, 94% coverage.
