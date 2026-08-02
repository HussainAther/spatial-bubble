# ADR 0001: Scientific state is authoritative

- Status: accepted
- Date: 2026-07-21

## Context

The platform must support solvers, ParaView, spectral renderers, Blender, XR, and
AI explanations without making any one presentation tool the source of truth.

## Decision

Versioned mesh/field state with units and provenance is authoritative. Rendered
assets are derived products. User interaction changes simulation input through
explicit parameters, boundary conditions, or forcing operations.

## Consequences

Adapters must preserve identifiers and provenance. Blender files and glTF files
cannot serve as scientific checkpoints by themselves. Re-rendering or changing
a colormap never changes the underlying physical result.
