# Known Issues & Architectural Notes

This document tracks known behaviors, limitations, and architectural decisions
that may affect usage or development.

## Resolved Issues

### MatrixTransform Proliferation (Resolved)

**Status**: ✅ Resolved in v0.1.0 via `CompositeProjection` / `InverseCompositeProjection`.

**Original problem**: Querying transforms across projection edges (e.g.,
`cam_front` → `global`) returned opaque `MatrixTransform` objects, losing
the intrinsic/extrinsic decomposition.

**Solution**: Composition now returns structured types that preserve separation:

| Composition | Result Type |
|---|---|
| `Transform × Projection` | `CompositeProjection` |
| `Transform × InverseProjection` | `InverseCompositeProjection` |
| `Projection × Transform` | `CompositeProjection` (extrinsics updated) |

See [projection_composition_analysis.md](architecture/projection_composition_analysis.md)
for the design evaluation.

## Current Limitations

### Tuple Frame IDs and JSON Serialization

Tuple frame IDs like `("ego", datetime(...))` are fully supported in-memory
and survive JSON roundtrip via the typed encoding in `to_dict`/`from_dict`.
However, the encoding adds structural overhead compared to simple string keys.

**Recommendation**: For maximum interoperability, prefer string or integer
frame IDs. Use compound tuples when the semantic richness justifies it
(e.g., temporal ego graphs).

### GeodeticPoint and ENUConverter

`GeodeticPoint` and `ENUConverter` are defined as stubs. Geo-referencing
functionality is planned but not yet implemented.
