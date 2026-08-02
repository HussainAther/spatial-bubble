# Current limitations

## Engineering approximations

- polygonal mid-surface and discrete geometry;
- one uniform total thickness with no distinct interface geometry;
- real, constant, wavelength-independent refractive indices;
- local tangent-plane optics with no global light transport;
- artificial two-sided collimated illumination;
- approximate color matching for display;
- `tracemalloc` peak Python memory rather than total process RSS.
- dependency lower bounds permit newer compatible environments; every run
  records the exact resolved versions, but a cross-platform lock file is not
  yet maintained;
- VTP round-trip verification uses the project's independent reader; external
  ParaView/VTK parser validation remains to be added;
- Python 3.14 is excluded because the current build backend's editable-install
  path file is hidden and therefore skipped by Python 3.14 site initialization.

## Visualization-only

XYZ-to-sRGB conversion, gamut clipping, tone mapping, PLY/glTF colors, and all
ParaView/Blender shading or smoothing are **VO**.

## Absent/speculative

No drainage, surfactant transport, Marangoni flow, evaporation, airflow,
vibration, rupture, disjoining pressure, absorption, polarization transport,
roughness, XR, Vulkan, or AI functionality is included.
