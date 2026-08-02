"""ParaView-readable VTK XML PolyData scientific export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import numpy.typing as npt

from openphenomena.data import Domain, Field, FieldAssociation


def write_vtp(domain: Domain, path: Path) -> Path:
    """Write all compatible authoritative fields to ASCII VTK PolyData."""

    vtk = ET.Element(
        "VTKFile", type="PolyData", version="1.0", byte_order="LittleEndian"
    )
    polydata = ET.SubElement(vtk, "PolyData")
    piece = ET.SubElement(
        polydata,
        "Piece",
        NumberOfPoints=str(len(domain.positions_m)),
        NumberOfVerts="0",
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys=str(len(domain.faces)),
    )
    points = ET.SubElement(piece, "Points")
    _data_array(points, "geometry.position", domain.positions_m, components=3)
    polys = ET.SubElement(piece, "Polys")
    _data_array(polys, "connectivity", domain.faces, components=1)
    offsets = 3 * np.arange(1, len(domain.faces) + 1, dtype=np.int64)
    _data_array(polys, "offsets", offsets, components=1)

    point_data = ET.SubElement(piece, "PointData")
    cell_data = ET.SubElement(piece, "CellData")
    field_data = ET.SubElement(piece, "FieldData")
    for semantic_id in sorted(domain.fields):
        sampled_field = domain.fields[semantic_id]
        target = {
            FieldAssociation.VERTEX: point_data,
            FieldAssociation.FACE: cell_data,
            FieldAssociation.GLOBAL: field_data,
        }[sampled_field.descriptor.association]
        _field_data_array(target, sampled_field)

    ET.indent(vtk, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)
    return path


def read_vtp(
    path: Path,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    dict[str, npt.NDArray[np.generic]],
]:
    """Read the subset of VTP emitted by :func:`write_vtp` for round-trip tests."""

    root = ET.parse(path).getroot()
    piece = root.find("./PolyData/Piece")
    if piece is None:
        raise ValueError("VTP file has no PolyData Piece")
    positions_node = piece.find("./Points/DataArray")
    connectivity_node = piece.find("./Polys/DataArray[@Name='connectivity']")
    offsets_node = piece.find("./Polys/DataArray[@Name='offsets']")
    if positions_node is None or connectivity_node is None or offsets_node is None:
        raise ValueError("VTP file is missing positions, connectivity, or offsets")
    positions = _parse_data_array(positions_node).astype(np.float64).reshape(-1, 3)
    connectivity = _parse_data_array(connectivity_node).astype(np.int64)
    offsets = _parse_data_array(offsets_node).astype(np.int64)
    expected_offsets = 3 * np.arange(1, len(offsets) + 1, dtype=np.int64)
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("VTP topology is not an all-triangle connectivity stream")
    if len(connectivity) != 3 * len(offsets):
        raise ValueError("VTP connectivity and offsets have inconsistent lengths")
    faces = connectivity.reshape(-1, 3)
    if len(positions) != int(piece.attrib["NumberOfPoints"]):
        raise ValueError("VTP point count does not match Piece metadata")
    if len(faces) != int(piece.attrib["NumberOfPolys"]):
        raise ValueError("VTP polygon count does not match Piece metadata")
    fields: dict[str, npt.NDArray[np.generic]] = {}
    for group_name in ("PointData", "CellData", "FieldData"):
        for node in piece.findall(f"./{group_name}/DataArray"):
            name = node.attrib["Name"]
            values = _parse_data_array(node)
            components = int(node.attrib.get("NumberOfComponents", "1"))
            if components > 1:
                values = values.reshape(-1, components)
            fields[name] = values
    return positions, faces, fields


def _field_data_array(parent: ET.Element, sampled_field: Field) -> None:
    values = sampled_field.values
    components = 1 if values.ndim <= 1 else int(np.prod(values.shape[1:]))
    node = _data_array(
        parent,
        sampled_field.descriptor.semantic_id,
        values,
        components=components,
    )
    node.set("Unit", sampled_field.descriptor.unit)
    node.set(
        "UnitDimension",
        " ".join(str(value) for value in sampled_field.descriptor.unit_dimension),
    )
    node.set("Association", sampled_field.descriptor.association.value)
    node.set("CoordinateFrame", sampled_field.descriptor.coordinate_frame)
    node.set("GeneratingModel", sampled_field.descriptor.generating_model)
    node.set(
        "GeneratingImplementation",
        sampled_field.descriptor.generating_implementation,
    )
    node.set("Fidelity", sampled_field.descriptor.fidelity.value)
    node.set("ValidationStatus", sampled_field.descriptor.validation_status.value)
    node.set("ComponentNames", "|".join(sampled_field.descriptor.component_names))
    node.set("CoordinateAxes", "|".join(sampled_field.descriptor.coordinate_axes))


def _data_array(
    parent: ET.Element,
    name: str,
    values: npt.NDArray[np.generic],
    *,
    components: int,
) -> ET.Element:
    array = np.asarray(values)
    vtk_type = {
        "f": "Float64" if array.dtype.itemsize == 8 else "Float32",
        "i": "Int64" if array.dtype.itemsize == 8 else "Int32",
        "u": "UInt64" if array.dtype.itemsize == 8 else "UInt32",
        "b": "UInt8",
    }.get(array.dtype.kind)
    if vtk_type is None:
        raise TypeError(f"unsupported VTK dtype: {array.dtype}")
    node = ET.SubElement(
        parent,
        "DataArray",
        type=vtk_type,
        Name=name,
        NumberOfComponents=str(components),
        format="ascii",
    )
    if array.dtype.kind == "f":
        node.text = " ".join(f"{float(item):.17g}" for item in array.ravel())
    else:
        node.text = " ".join(str(int(item)) for item in array.ravel())
    return node


def _parse_data_array(node: ET.Element) -> npt.NDArray[np.generic]:
    dtype = {
        "Float64": np.float64,
        "Float32": np.float32,
        "Int64": np.int64,
        "Int32": np.int32,
        "UInt64": np.uint64,
        "UInt32": np.uint32,
        "UInt8": np.uint8,
    }[node.attrib["type"]]
    return np.fromstring(node.text or "", sep=" ", dtype=dtype)
