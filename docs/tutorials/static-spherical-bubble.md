# Tutorial: canonical static spherical bubble

## Purpose

This reference study exercises the scientific state model, plugin discovery,
discrete surface geometry, Young–Laplace pressure, coherent thin-film optics,
validation evidence, scientific serialization, ParaView export, and derived
Blender/glTF views. It does not simulate time evolution.

## Reproduce from a clean checkout

```bash
./scripts/reproduce_static_sphere.sh
```

The script creates `.venv`, installs the package, discovers the bundled plugin
through package metadata, and writes `outputs/static-spherical-bubble`.

For an already configured environment:

```bash
.venv/bin/python -m openphenomena.reference \
  --output outputs/static-spherical-bubble
```

The command exits with status 0 only when every required evidence record passes.

## Canonical configuration

- radius: `0.01 m`
- surface tension on each interface: `0.03 N m^-1`
- external pressure: `101325 Pa`
- internal pressure: `101337 Pa`
- total film thickness: `450 nm`
- material stack: air (`1.0`) – film (`1.333`) – air (`1.0`)
- wavelength samples: 81 uniformly spaced samples from 380–780 nm
- meshes: projected icospheres at refinement levels 1–4
- randomness: none; the recorded mesh seed is 0 for interface consistency

Illumination is two opposed incoherent collimated equal-energy unpolarized
beams. Each vertex uses the incident beam in its outward hemisphere. Optics is
evaluated in a local tangent plane without transport between surface points.

## Outputs

| Path | Meaning |
|---|---|
| `scientific/manifest.json` + `arrays.npz` | authoritative immutable run bundle |
| `convergence/convergence.json` | reusable machine-readable convergence data |
| `convergence/convergence.csv` | tabular convergence data |
| `reports/validation.md` | generated evidence report |
| `reports/convergence.md` | generated measured convergence report |
| `exports/bubble_finest.vtp` | scientific ParaView interchange |
| `exports/bubble_blender_vo.ply` | Blender-compatible **VO** mesh |
| `exports/bubble_vo.gltf` | glTF 2.0 **VO** artifact |
| `exports/visualization_manifest.json` | field and classification inventory |

## Inspect in ParaView

Open `bubble_finest.vtp`, choose a point or cell field from the coloring menu,
and use the Information panel to inspect array dimensions. Spectral arrays have
one component per wavelength. The `.vtp` file is an interchange copy; the
authoritative descriptors and provenance remain in the scientific bundle.
