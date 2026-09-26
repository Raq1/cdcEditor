from __future__ import annotations

import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from mathutils import Matrix, Vector

from ...core.log import logger
from ...core.material_ui import get_material_draw_group
from ..common.section import SectionContextCache
from .tr8_model import (
    TR8_VERTEX_POSITION,
    TR8_VERTEX_NORMAL,
    TR8_VERTEX_TANGENT,
    TR8_VERTEX_BINORMAL,
    TR8_VERTEX_SKIN_WEIGHTS,
    TR8_VERTEX_SKIN_INDICES,
    TR8_VERTEX_TEXCOORD1,
    TR8_VERTEX_TEXCOORD2,
    TR8_VERTEX_COLOR1,
)

_SECTION_NAME_RE = re.compile(r'^(?P<index>\d+)_(?P<id>[0-9a-fA-F]+)\.(?P<ext>[^.]+)$')
_TR8_MAX_WEIGHTED_BONES_PER_BATCH = 42
_TR8_MAX_WEIGHTS_PER_VERTEX = 4
_TR8_MAX_BATCH_VERTICES = 0xFFFF
_TR8_VERTEX_STRIDE = 44
_RELOCATION_POINTER = 0


@dataclass(slots=True)
class UnderworldExportResult:
    path: Path
    render_id: int
    batch_count: int
    vertex_count: int
    triangle_count: int
    material_count: int


@dataclass(slots=True)
class _Relocation:
    target_section_index: int
    offset: int
    type_specific: int = 0
    relocation_type: int = _RELOCATION_POINTER


@dataclass(slots=True)
class _SectionTemplate:
    section_type: int = 12
    skip_flags: int = 0
    version_id: int = 0
    packed_low: int = 0
    section_id: int = 0
    spec_mask: int = 0xFFFFFFFF
    section_index: int = 0


@dataclass
class _TR8VertexPayload:
    source_vertex: int
    position: Vector
    normal: Vector
    uv: tuple[float, float]
    color: tuple[float, float, float, float]
    weights: list[tuple[int, float]]


@dataclass
class _TR8Group:
    material_index: int
    draw_group: int
    indices: list[int] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return len(self.indices) // 3


@dataclass
class _TR8Batch:
    index: int
    skin_map: list[int] = field(default_factory=list)
    vertices: list[_TR8VertexPayload] = field(default_factory=list)
    key_map: dict[tuple, int] = field(default_factory=dict)
    groups: list[_TR8Group] = field(default_factory=list)

    def triangle_count(self) -> int:
        return sum(group.triangle_count for group in self.groups)


def _warn(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        try:
            callback(message)
            return
        except Exception:
            pass
    logger.warning(message)


def _section_index_from_filename(path: str | Path) -> int:
    stem = Path(path).stem
    head = stem.split('_', 1)[0]
    return int(head)


def _find_tr8mesh_template(directory: Path, render_id: int) -> Path | None:
    directory = Path(directory)
    rid = int(render_id) & 0xFFFFFFFF
    candidates: list[tuple[int, Path]] = []
    for path in sorted(directory.iterdir() if directory.exists() else []):
        if not path.is_file() or path.suffix.lower() not in {'.tr8mesh', '.gnc'}:
            continue
        try:
            info = _read_section_template(path)
        except Exception:
            continue
        if int(info.section_type) != 12:
            continue
        if int(info.section_id) & 0xFFFFFFFF != rid:
            continue
        candidates.append((-path.stat().st_size, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[0][1]


def _read_section_template(path: Path) -> _SectionTemplate:
    path = Path(path)
    blob = path.read_bytes()[:24]
    if len(blob) < 24 or blob[:4] not in {b'SECT', b'TCES'}:
        raise ValueError(f'{path.name} is not a standalone section')
    _stored_size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from('<iBBHIII', blob, 4)
    return _SectionTemplate(
        section_type=int(section_type),
        skip_flags=int(skip_flags),
        version_id=int(version_id),
        packed_low=int(packed_data) & 0xFF,
        section_id=int(section_id) & 0xFFFFFFFF,
        spec_mask=int(spec_mask) & 0xFFFFFFFF,
        section_index=_section_index_from_filename(path),
    )




def _read_standalone_section_payload(path: Path) -> tuple[_SectionTemplate, bytes, list[_Relocation]]:
    path = Path(path)
    blob = path.read_bytes()
    if len(blob) < 24 or blob[:4] not in {b'SECT', b'TCES'}:
        raise ValueError(f'{path.name} is not a standalone section')
    stored_size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from('<iBBHIII', blob, 4)
    reloc_count = (int(packed_data) >> 8) & 0x00FFFFFF
    table_end = 24 + reloc_count * 8
    if table_end > len(blob):
        reloc_count = 0
        table_end = 24
    relocations: list[_Relocation] = []
    for index in range(reloc_count):
        type_and_section, type_specific, offset = struct.unpack_from('<HhI', blob, 24 + index * 8)
        relocations.append(_Relocation(
            target_section_index=(int(type_and_section) >> 3) & 0x1FFF,
            offset=int(offset),
            type_specific=int(type_specific),
            relocation_type=int(type_and_section) & 0x7,
        ))
    declared_size = max(0, int(stored_size))
    payload_end = min(len(blob), table_end + declared_size) if declared_size else len(blob)
    template = _SectionTemplate(
        section_type=int(section_type),
        skip_flags=int(skip_flags),
        version_id=int(version_id),
        packed_low=int(packed_data) & 0xFF,
        section_id=int(section_id) & 0xFFFFFFFF,
        spec_mask=int(spec_mask) & 0xFFFFFFFF,
        section_index=_section_index_from_filename(path),
    )
    return template, bytes(blob[table_end:payload_end]), relocations


def _u32_at(data: bytes, offset: int, default: int = 0) -> int:
    if int(offset) < 0 or int(offset) + 4 > len(data):
        return int(default)
    return int(struct.unpack_from('<I', data, int(offset))[0])


def _valid_range(data: bytes, offset: int, size: int) -> bool:
    return int(offset) >= 0 and int(size) >= 0 and int(offset) + int(size) <= len(data)


def _extract_skeleton_records_from_template(path: Path, warning_cb=None) -> tuple[bytes, int] | None:
    """Copy the source TR8 cdcModelData skeleton records into the rebuilt mesh section.

    The records contain hierarchy/pivot data and two signed 16-bit fields whose exact
    runtime purpose is still not fully proven.  Preserving the template records is safer
    than generating global vertex ranges that can overflow the signed fields on larger
    rebuilt meshes.
    """
    try:
        template, data, relocs = _read_standalone_section_payload(Path(path))
    except Exception as exc:
        _warn(warning_cb, f'Could not read source Underworld skeleton template from {Path(path).name}: {exc}')
        return None
    if int(template.section_type) != 12 or len(data) < 0x84:
        return None
    header_local = data.find(b'Mesh')
    if header_local < 0 or header_local + 0x84 > len(data):
        return None
    expected_bones = int(struct.unpack_from('<H', data, header_local + 0x80)[0])
    if expected_bones <= 0 or expected_bones > 4096:
        return None
    self_targets = {0, int(template.section_index), int(template.section_id) & 0x1FFF, int(template.section_id) & 0xFFFFFFFF}
    candidates: list[int] = []
    for reloc in reversed(relocs):
        if int(reloc.relocation_type) != _RELOCATION_POINTER:
            continue
        target = int(reloc.target_section_index)
        if target not in self_targets:
            continue
        field_local = int(reloc.offset)
        if not _valid_range(data, field_local, 4):
            continue
        value = _u32_at(data, field_local)
        for candidate in (int(value) - 16, int(value) - 8, field_local - 16, field_local - 4):
            if candidate not in candidates and _valid_range(data, candidate, 8):
                candidates.append(candidate)
    for header in candidates:
        count = _u32_at(data, header)
        bones_offset = _u32_at(data, header + 4)
        if int(count) != int(expected_bones) or count <= 0 or count > 4096:
            continue
        records_local = int(header) + 0x10
        if not _valid_range(data, records_local, int(count) * 64):
            records_local = int(bones_offset)
        if not _valid_range(data, records_local, int(count) * 64):
            continue
        out = bytearray(0x10 + int(count) * 64)
        struct.pack_into('<IIII', out, 0, int(count), 0x10, 0, 0)
        out[0x10:0x10 + int(count) * 64] = data[records_local:records_local + int(count) * 64]
        return bytes(out), int(count)
    return None


def _fit_i16_or_sentinel(value: int, *, sentinel: int = -1) -> int:
    value = int(value)
    if -32768 <= value <= 32767:
        return value
    return int(sentinel)

def _write_standalone_section(path: Path, template: _SectionTemplate, data: bytes, relocations: Sequence[_Relocation]) -> None:
    path = Path(path)
    relocs_by_offset: dict[int, _Relocation] = {}
    for reloc in relocations:
        relocs_by_offset[int(reloc.offset)] = reloc
    ordered_relocs = [relocs_by_offset[offset] for offset in sorted(relocs_by_offset)]
    packed_data = (int(template.packed_low) & 0xFF) | (len(ordered_relocs) << 8)
    payload = bytearray()
    payload.extend(b'SECT')
    payload.extend(struct.pack('<iBBHIII', len(data), int(template.section_type) & 0xFF, int(template.skip_flags) & 0xFF, int(template.version_id) & 0xFFFF, int(packed_data) & 0xFFFFFFFF, int(template.section_id) & 0xFFFFFFFF, int(template.spec_mask) & 0xFFFFFFFF))
    for reloc in ordered_relocs:
        type_and_section = ((int(reloc.target_section_index) & 0x1FFF) << 3) | (int(reloc.relocation_type) & 0x7)
        payload.extend(struct.pack('<HhI', type_and_section & 0xFFFF, int(reloc.type_specific), int(reloc.offset) & 0xFFFFFFFF))
    payload.extend(data)
    path.write_bytes(bytes(payload))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith('0x'):
                return int(text, 16)
        return int(value)
    except Exception:
        return int(default)


def _align(data: bytearray, alignment: int = 16) -> int:
    alignment = max(1, int(alignment))
    padding = (-len(data)) % alignment
    if padding:
        data.extend(b'\x00' * padding)
    return len(data)


def _normal_to_tuple(normal) -> tuple[float, float, float]:
    try:
        vec = Vector((float(normal.x), float(normal.y), float(normal.z)))
    except Exception:
        vec = Vector((0.0, 0.0, 1.0))
    if vec.length <= 1e-8:
        vec = Vector((0.0, 0.0, 1.0))
    else:
        vec.normalize()
    return (float(vec.x), float(vec.y), float(vec.z))


def _normal_to_byte(value: float) -> int:
    value = max(-1.0, min(1.0, float(value)))
    return int(round((value * 0.5 + 0.5) * 255.0)) & 0xFF


def _uv_to_tr8_raw(value: float) -> int:
    value = max(-32768.0, min(32767.0, float(value) * 2048.0))
    return int(round(value)) & 0xFFFF


def _rgba_float_to_bytes(color: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    result = []
    for index in range(4):
        try:
            component = float(color[index])
        except Exception:
            component = 1.0
        result.append(max(0, min(255, int(round(component * 255.0)))))
    return tuple(result)


def _material_prop(material, name: str, default=None):
    if material is None:
        return default
    try:
        if name in material:
            return material.get(name, default)
    except Exception:
        pass
    try:
        value = getattr(material, name, None)
        if value is not None:
            return value
    except Exception:
        pass
    return default


def _underworld_material_id(material, material_index: int) -> int:
    value = _material_prop(material, 'trlau_tr8_material_id', None)
    if value is not None:
        return _safe_int(value, material_index) & 0xFFFFFFFF
    name = str(getattr(material, 'name', '') or '') if material is not None else ''
    match = re.match(r'^Material_(\d+)_', name)
    if match is not None:
        return _safe_int(match.group(1), material_index) & 0xFFFFFFFF
    return int(material_index) & 0xFFFFFFFF


def _underworld_material_double_sided(material) -> bool:
    if material is None:
        return False
    for name in ('trlau_ui_tr8_double_sided', 'trlau_tr8_double_sided'):
        try:
            return bool(getattr(material, name)) if hasattr(material, name) else bool(material.get(name, False))
        except Exception:
            continue
    return False


def _underworld_material_draw_group(material) -> int:
    try:
        return int(get_material_draw_group(material))
    except Exception:
        try:
            return int(_material_prop(material, 'trlau_draw_group', 0) or 0)
        except Exception:
            return 0


def _active_color_attribute(mesh):
    try:
        color_attrs = getattr(mesh, 'color_attributes', None)
        if color_attrs is not None:
            active = getattr(color_attrs, 'active_color', None) or getattr(color_attrs, 'active', None)
            if active is not None:
                return active
            for attr in color_attrs:
                return attr
    except Exception:
        pass
    try:
        vertex_colors = getattr(mesh, 'vertex_colors', None)
        if vertex_colors:
            active = vertex_colors.active
            if active is not None:
                return active
            for attr in vertex_colors:
                return attr
    except Exception:
        pass
    return None


def _sample_color(color_attr, loop_index: int, vertex_index: int) -> tuple[float, float, float, float]:
    if color_attr is None:
        return (1.0, 1.0, 1.0, 1.0)
    try:
        domain = str(getattr(color_attr, 'domain', '') or '').upper()
        data_index = int(vertex_index) if domain == 'POINT' else int(loop_index)
        color = color_attr.data[data_index].color
        return (
            _safe_float(color[0], 1.0),
            _safe_float(color[1], 1.0),
            _safe_float(color[2], 1.0),
            _safe_float(color[3] if len(color) > 3 else 1.0, 1.0),
        )
    except Exception:
        return (1.0, 1.0, 1.0, 1.0)


def _bone_index_from_group_name(name: str) -> int | None:
    text = str(name or '')
    if text.startswith('bone_'):
        try:
            return int(text.split('_', 1)[1])
        except Exception:
            return None
    try:
        return int(text)
    except Exception:
        return None


def _bone_weights_for_vertex(mesh_obj, vertex_index: int, *, warning_counter: dict[str, int] | None = None) -> list[tuple[int, float]]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None or vertex_index < 0 or vertex_index >= len(mesh.vertices):
        return []
    groups = getattr(mesh_obj, 'vertex_groups', None)
    if groups is None:
        return []
    result: list[tuple[int, float]] = []
    for group_ref in getattr(mesh.vertices[int(vertex_index)], 'groups', []) or []:
        try:
            group = groups[int(group_ref.group)]
            bone_index = _bone_index_from_group_name(getattr(group, 'name', ''))
            if bone_index is None:
                continue
            weight = float(getattr(group_ref, 'weight', 0.0) or 0.0)
            if weight > 0.0001:
                result.append((int(bone_index), weight))
        except Exception:
            continue
    result.sort(key=lambda item: (-item[1], item[0]))
    if warning_counter is not None and len(result) > _TR8_MAX_WEIGHTS_PER_VERTEX:
        warning_counter['over_four'] = int(warning_counter.get('over_four', 0)) + 1
    result = result[:_TR8_MAX_WEIGHTS_PER_VERTEX]
    total = sum(weight for _bone, weight in result)
    if total > 1e-9:
        result = [(bone, weight / total) for bone, weight in result]
    return result


def _weights_or_default(weights: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    clean = [(int(bone), float(weight)) for bone, weight in list(weights or []) if float(weight) > 0.0001]
    if not clean:
        return [(0, 1.0)]
    total = sum(weight for _bone, weight in clean)
    if total > 1e-9:
        clean = [(bone, weight / total) for bone, weight in clean]
    return clean[:_TR8_MAX_WEIGHTS_PER_VERTEX]


def _triangle_skin_bones(payloads: Sequence[_TR8VertexPayload]) -> list[int]:
    result: list[int] = []
    for payload in payloads:
        for bone_index, _weight in _weights_or_default(payload.weights):
            if int(bone_index) not in result:
                result.append(int(bone_index))
    return result


def _vertex_key(payload: _TR8VertexPayload) -> tuple:
    weights = tuple((int(b), round(float(w), 6)) for b, w in _weights_or_default(payload.weights))
    normal = _normal_to_tuple(payload.normal)
    return (
        int(payload.source_vertex),
        round(float(payload.position.x), 6), round(float(payload.position.y), 6), round(float(payload.position.z), 6),
        round(normal[0], 6), round(normal[1], 6), round(normal[2], 6),
        round(float(payload.uv[0]), 6), round(float(payload.uv[1]), 6),
        tuple(round(float(v), 6) for v in payload.color),
        weights,
    )


def _can_add_triangle_to_batch(batch: _TR8Batch, payloads: Sequence[_TR8VertexPayload]) -> bool:
    new_vertices = 0
    for payload in payloads:
        if _vertex_key(payload) not in batch.key_map:
            new_vertices += 1
    if len(batch.vertices) + int(new_vertices) > _TR8_MAX_BATCH_VERTICES:
        return False
    skin_map = list(batch.skin_map)
    for bone_index in _triangle_skin_bones(payloads):
        if int(bone_index) not in skin_map:
            skin_map.append(int(bone_index))
    return len(skin_map) <= _TR8_MAX_WEIGHTED_BONES_PER_BATCH


def _append_triangle(batch: _TR8Batch, material_index: int, draw_group: int, payloads: Sequence[_TR8VertexPayload], *, double_sided: bool) -> None:
    local_indices: list[int] = []
    for payload in payloads:
        key = _vertex_key(payload)
        existing = batch.key_map.get(key)
        if existing is None:
            existing = len(batch.vertices)
            batch.vertices.append(payload)
            batch.key_map[key] = existing
            for bone_index in _triangle_skin_bones([payload]):
                if int(bone_index) not in batch.skin_map:
                    batch.skin_map.append(int(bone_index))
        local_indices.append(int(existing))

    if not batch.groups or int(batch.groups[-1].material_index) != int(material_index) or int(batch.groups[-1].draw_group) != int(draw_group):
        batch.groups.append(_TR8Group(material_index=int(material_index), draw_group=int(draw_group)))
    group = batch.groups[-1]
    tri = tuple(local_indices[:3])
    group.indices.extend(tri)
    if bool(double_sided):
        group.indices.extend((tri[2], tri[1], tri[0]))


def _material_slots(mesh_obj) -> list:
    try:
        return list(getattr(getattr(mesh_obj, 'data', None), 'materials', []) or [])
    except Exception:
        return []


def _collect_triangles_by_material(mesh_obj, arm_obj, model_root, warning_cb=None) -> tuple[dict[int, list[tuple[Sequence[_TR8VertexPayload], bool, int]]], list, list[Vector]]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        raise ValueError(f'Underworld mesh object {getattr(mesh_obj, "name", "<mesh>")} has no mesh data')
    try:
        mesh.calc_loop_triangles()
        mesh.calc_normals_split()
    except Exception:
        pass

    uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
    color_attr = _active_color_attribute(mesh)
    arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else model_root.matrix_world.inverted_safe()
    warning_counter: dict[str, int] = {}

    local_position: dict[int, Vector] = {}
    local_weights: dict[int, list[tuple[int, float]]] = {}
    all_positions: list[Vector] = []
    for vertex in mesh.vertices:
        index = int(vertex.index)
        world = mesh_obj.matrix_world @ vertex.co
        position = arm_inv @ world
        local_position[index] = Vector((float(position.x), float(position.y), float(position.z)))
        all_positions.append(local_position[index])
        local_weights[index] = _weights_or_default(_bone_weights_for_vertex(mesh_obj, index, warning_counter=warning_counter))

    if int(warning_counter.get('over_four', 0)) > 0:
        _warn(warning_cb, f'Underworld mesh "{getattr(mesh_obj, "name", "<mesh>")}" has {warning_counter["over_four"]} vertex/vertices with more than four weights; export kept the strongest four and renormalized them.')

    materials = _material_slots(mesh_obj)
    triangles_by_material: dict[int, list[tuple[Sequence[_TR8VertexPayload], bool, int]]] = defaultdict(list)
    for tri in getattr(mesh, 'loop_triangles', []) or []:
        material_index = int(getattr(tri, 'material_index', 0) or 0)
        material = materials[material_index] if 0 <= material_index < len(materials) else None
        double_sided = _underworld_material_double_sided(material)
        draw_group = _underworld_material_draw_group(material)
        payloads: list[_TR8VertexPayload] = []
        for loop_index in tri.loops:
            try:
                loop = mesh.loops[int(loop_index)]
                vertex_index = int(loop.vertex_index)
            except Exception:
                continue
            uv = (0.0, 0.0)
            if uv_layer is not None:
                try:
                    luv = uv_layer[int(loop_index)].uv
                    uv = (float(luv.x), float(luv.y))
                except Exception:
                    uv = (0.0, 0.0)
            try:
                normal = Vector(mesh.loops[int(loop_index)].normal)
            except Exception:
                try:
                    normal = Vector(mesh.vertices[vertex_index].normal)
                except Exception:
                    normal = Vector((0.0, 0.0, 1.0))
            payloads.append(_TR8VertexPayload(
                source_vertex=int(vertex_index),
                position=local_position.get(vertex_index, Vector((0.0, 0.0, 0.0))),
                normal=normal,
                uv=uv,
                color=_sample_color(color_attr, int(loop_index), int(vertex_index)),
                weights=local_weights.get(vertex_index, [(0, 1.0)]),
            ))
        if len(payloads) != 3:
            continue
        if len({payload.source_vertex for payload in payloads}) < 3:
            continue
        triangles_by_material[int(material_index)].append((payloads, bool(double_sided), int(draw_group)))
    return triangles_by_material, materials, all_positions


def _build_batches(mesh_obj, arm_obj, model_root, warning_cb=None) -> tuple[list[_TR8Batch], list, list[Vector]]:
    triangles_by_material, materials, positions = _collect_triangles_by_material(mesh_obj, arm_obj, model_root, warning_cb=warning_cb)
    batches: list[_TR8Batch] = []
    current: _TR8Batch | None = None

    def new_batch() -> _TR8Batch:
        batch = _TR8Batch(index=len(batches))
        batches.append(batch)
        return batch

    for material_index in sorted(triangles_by_material.keys()):
        for payloads, double_sided, draw_group in triangles_by_material[material_index]:
            triangle_bones = _triangle_skin_bones(payloads)
            if len(triangle_bones) > _TR8_MAX_WEIGHTED_BONES_PER_BATCH:
                raise ValueError(
                    f'Underworld triangle on material slot {material_index} needs {len(triangle_bones)} unique weighted bones, exceeding the engine batch limit of {_TR8_MAX_WEIGHTED_BONES_PER_BATCH}.'
                )
            if current is None or not _can_add_triangle_to_batch(current, payloads):
                current = new_batch()
            if not _can_add_triangle_to_batch(current, payloads):
                raise ValueError(
                    f'Underworld triangle on material slot {material_index} cannot fit into an empty batch with the {_TR8_MAX_WEIGHTED_BONES_PER_BATCH}-bone palette limit.'
                )
            _append_triangle(current, material_index, draw_group, payloads, double_sided=double_sided)

    batches = [batch for batch in batches if batch.triangle_count() > 0 and batch.vertices]
    for index, batch in enumerate(batches):
        batch.index = int(index)
    return batches, materials, positions


def _compute_bounds(points: Sequence[Vector]) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], float]:
    if not points:
        zero = (0.0, 0.0, 0.0, 0.0)
        return zero, zero, zero, 0.0
    xs = [float(p.x) for p in points]
    ys = [float(p.y) for p in points]
    zs = [float(p.z) for p in points]
    mn = Vector((min(xs), min(ys), min(zs)))
    mx = Vector((max(xs), max(ys), max(zs)))
    center = (mn + mx) * 0.5
    radius = max((p - center).length for p in points) if points else 0.0
    return (float(center.x), float(center.y), float(center.z), 0.0), (float(mn.x), float(mn.y), float(mn.z), 0.0), (float(mx.x), float(mx.y), float(mx.z), 0.0), float(radius)


def _pack_vertex_format() -> bytes:
    components = [
        (TR8_VERTEX_POSITION, 0, 0x02, 0),
        (TR8_VERTEX_NORMAL, 12, 0x05, 0),
        (TR8_VERTEX_SKIN_WEIGHTS, 16, 0x06, 0),
        (TR8_VERTEX_SKIN_INDICES, 20, 0x07, 0),
        (TR8_VERTEX_TEXCOORD1, 24, 0x13, 0),
        (TR8_VERTEX_TEXCOORD2, 28, 0x13, 0),
        (TR8_VERTEX_COLOR1, 32, 0x04, 0),
        (TR8_VERTEX_BINORMAL, 36, 0x05, 0),
        (TR8_VERTEX_TANGENT, 40, 0x05, 0),
    ]
    stride = 44
    data = bytearray(0x20)
    struct.pack_into('<II', data, 0x10, 0xBC7532C8, 0x2FB19541)
    struct.pack_into('<HBB', data, 0x18, len(components), stride, 0)
    for semantic, offset, entry_type, entry_null in components:
        data.extend(struct.pack('<IHBB', int(semantic) & 0xFFFFFFFF, int(offset) & 0xFFFF, int(entry_type) & 0xFF, int(entry_null) & 0xFF))
    return bytes(data)


def _scratch_vertex_format_template() -> '_TR8VertexFormatTemplate':
    blob = _pack_vertex_format()
    stride, components = _parse_source_vertex_format(blob, 0)
    if stride <= 0 or not components:
        raise ValueError('Internal Underworld scratch vertex declaration is invalid')
    return _TR8VertexFormatTemplate(blob=blob, stride=int(stride), components=dict(components))


def _source_mesh_header_opaque_block(source_template: Path | None) -> bytes:
    if source_template is None:
        return bytes(0x24)
    try:
        _template, payload, _relocs = _read_standalone_section_payload(Path(source_template))
    except Exception:
        return bytes(0x24)
    header = payload.find(b'Mesh')
    if header < 0 or header + 0x6C > len(payload):
        return bytes(0x24)
    return bytes(payload[header + 0x48:header + 0x6C])


def _source_mesh_bone_count(source_template: Path | None) -> int:
    if source_template is None:
        return 0
    try:
        _template, payload, _relocs = _read_standalone_section_payload(Path(source_template))
    except Exception:
        return 0
    header = payload.find(b'Mesh')
    if header < 0 or header + 0x82 > len(payload):
        return 0
    try:
        return int(struct.unpack_from('<H', payload, header + 0x80)[0])
    except Exception:
        return 0


@dataclass(slots=True)
class _TR8SourceVertexComponent:
    semantic: int
    offset: int
    entry_type: int
    entry_null: int


@dataclass(slots=True)
class _TR8SourceBatch:
    index: int
    vertex_components_offset: int
    vertex_stream_offset: int
    vertex_stride: int
    num_vertices: int
    num_faces: int
    skin_map_offset: int
    skin_map_size: int
    skin_map: list[int]
    components: dict[int, _TR8SourceVertexComponent]


@dataclass(slots=True)
class _TR8SourceGroupRecord:
    base_index: int
    num_faces: int
    num_vertices: int
    flags: int
    draw_group: int
    order: int
    material_index: int


@dataclass(slots=True)
class _TR8VertexFormatTemplate:
    blob: bytes
    stride: int
    components: dict[int, _TR8SourceVertexComponent]


def _parse_source_vertex_format(data: bytes | bytearray, vertex_components_offset: int) -> tuple[int, dict[int, _TR8SourceVertexComponent]]:
    vertex_components_offset = int(vertex_components_offset)
    if not _valid_range(data, vertex_components_offset, 0x20):
        return 0, {}
    component_count = int(struct.unpack_from('<H', data, vertex_components_offset + 0x18)[0])
    stride = int(data[vertex_components_offset + 0x1A])
    if component_count <= 0 or component_count > 64 or stride <= 0 or stride > 256:
        return 0, {}
    entries_offset = vertex_components_offset + 0x20
    if not _valid_range(data, entries_offset, component_count * 8):
        return 0, {}
    components: dict[int, _TR8SourceVertexComponent] = {}
    for entry_index in range(component_count):
        semantic, offset, entry_type, entry_null = struct.unpack_from('<IHBB', data, entries_offset + (entry_index * 8))
        components[int(semantic)] = _TR8SourceVertexComponent(
            semantic=int(semantic),
            offset=int(offset),
            entry_type=int(entry_type),
            entry_null=int(entry_null),
        )
    if TR8_VERTEX_POSITION not in components:
        return 0, {}
    return int(stride), components



def _source_vertex_format_template_at(data: bytes | bytearray, vertex_components_offset: int) -> _TR8VertexFormatTemplate | None:
    stride, components = _parse_source_vertex_format(data, int(vertex_components_offset))
    if stride <= 0 or not components:
        return None
    count = len(components)
    block_size = 0x20 + (int(count) * 8)
    if not _valid_range(data, int(vertex_components_offset), block_size):
        return None
    return _TR8VertexFormatTemplate(
        blob=bytes(data[int(vertex_components_offset):int(vertex_components_offset) + block_size]),
        stride=int(stride),
        components=dict(components),
    )


def _choose_source_vertex_format_template(data: bytes | bytearray, header_offset: int, warning_cb=None) -> _TR8VertexFormatTemplate:
    candidates: list[_TR8VertexFormatTemplate] = []
    mesh_info_offset = _u32_at(data, int(header_offset) + 0x70)
    mesh_record_count = 0
    if _valid_range(data, int(header_offset) + 0x7E, 2):
        mesh_record_count = int(struct.unpack_from('<H', data, int(header_offset) + 0x7E)[0])
    if mesh_record_count > 0 and mesh_record_count < 4096 and _valid_range(data, mesh_info_offset, mesh_record_count * 52):
        for record_index in range(mesh_record_count):
            fmt = _u32_at(data, mesh_info_offset + (record_index * 52))
            tpl = _source_vertex_format_template_at(data, fmt)
            if tpl is not None:
                candidates.append(tpl)

    if not candidates:
        search_limit = min(len(data), 0x200000)
        offset = 0
        while offset + 0x20 <= search_limit:
            tpl = _source_vertex_format_template_at(data, offset)
            if tpl is not None:
                candidates.append(tpl)
                offset += max(0x20, len(tpl.blob))
                continue
            offset += 0x10

    if candidates:
        candidates.sort(key=lambda tpl: (
            1 if TR8_VERTEX_TEXCOORD2 in tpl.components else 0,
            1 if TR8_VERTEX_COLOR1 in tpl.components else 0,
            len(tpl.components),
            int(tpl.stride),
        ), reverse=True)
        chosen = candidates[0]
        logger.debug(
            'Using source TR8 vertex declaration for scratch export: stride=%d components=%d hasUV2=%d hasColor=%d',
            int(chosen.stride),
            len(chosen.components),
            1 if TR8_VERTEX_TEXCOORD2 in chosen.components else 0,
            1 if TR8_VERTEX_COLOR1 in chosen.components else 0,
        )
        return chosen

    _warn(warning_cb, 'Could not find a source TR8 vertex declaration; using generic scratch declaration.')
    stride = _TR8_VERTEX_STRIDE
    components = {
        TR8_VERTEX_POSITION: _TR8SourceVertexComponent(TR8_VERTEX_POSITION, 0, 0x02, 0),
        TR8_VERTEX_NORMAL: _TR8SourceVertexComponent(TR8_VERTEX_NORMAL, 12, 0x05, 0),
        TR8_VERTEX_SKIN_WEIGHTS: _TR8SourceVertexComponent(TR8_VERTEX_SKIN_WEIGHTS, 16, 0x06, 0),
        TR8_VERTEX_SKIN_INDICES: _TR8SourceVertexComponent(TR8_VERTEX_SKIN_INDICES, 20, 0x07, 0),
        TR8_VERTEX_TEXCOORD1: _TR8SourceVertexComponent(TR8_VERTEX_TEXCOORD1, 24, 0x13, 0),
        TR8_VERTEX_TEXCOORD2: _TR8SourceVertexComponent(TR8_VERTEX_TEXCOORD2, 28, 0x13, 0),
        TR8_VERTEX_COLOR1: _TR8SourceVertexComponent(TR8_VERTEX_COLOR1, 32, 0x04, 0),
        TR8_VERTEX_BINORMAL: _TR8SourceVertexComponent(TR8_VERTEX_BINORMAL, 36, 0x05, 0),
        TR8_VERTEX_TANGENT: _TR8SourceVertexComponent(TR8_VERTEX_TANGENT, 40, 0x05, 0),
    }
    return _TR8VertexFormatTemplate(blob=_pack_vertex_format(), stride=stride, components=components)

def _read_source_group_records(data: bytes | bytearray, header_offset: int) -> list[_TR8SourceGroupRecord]:
    if not _valid_range(data, int(header_offset) + 0x80, 2):
        return []
    group_offset = _u32_at(data, int(header_offset) + 0x6C)
    group_count = int(struct.unpack_from('<H', data, int(header_offset) + 0x7C)[0])
    groups: list[_TR8SourceGroupRecord] = []
    if group_count <= 0 or not _valid_range(data, int(group_offset), int(group_count) * 64):
        return groups
    for index in range(int(group_count)):
        off = int(group_offset) + (index * 64) + 0x20
        base_index, num_faces, num_vertices, flags, draw_group, order, material_index = struct.unpack_from('<IIIIiiI', data, off)
        groups.append(_TR8SourceGroupRecord(
            base_index=int(base_index),
            num_faces=int(num_faces),
            num_vertices=int(num_vertices),
            flags=int(flags),
            draw_group=int(draw_group),
            order=int(order),
            material_index=int(material_index),
        ))
    return groups


def _read_source_mesh_record_metadata(data: bytes | bytearray, header_offset: int) -> list[dict[str, int]]:
    mesh_info_offset = _u32_at(data, int(header_offset) + 0x70)
    record_count = int(struct.unpack_from('<H', data, int(header_offset) + 0x7E)[0]) if _valid_range(data, int(header_offset) + 0x7E, 2) else 0
    records: list[dict[str, int]] = []
    if record_count <= 0 or not _valid_range(data, int(mesh_info_offset), int(record_count) * 52):
        return records
    for index in range(int(record_count)):
        off = int(mesh_info_offset) + (index * 52)
        vertex_components_offset, num_vertices, base_index, num_faces, _unused0, sort_value, submesh_count, skin_map_size, skin_map_offset, vertex_offset = struct.unpack_from('<IIIIIIIIII', data, off)
        records.append({
            'index': int(index),
            'vertex_components_offset': int(vertex_components_offset),
            'num_vertices': int(num_vertices),
            'base_index': int(base_index),
            'num_faces': int(num_faces),
            'sort_value': int(sort_value),
            'submesh_count': int(submesh_count),
            'skin_map_size': int(skin_map_size),
            'skin_map_offset': int(skin_map_offset),
            'vertex_offset': int(vertex_offset),
        })
    return records


def _read_source_skin_palette(data: bytes | bytearray, skin_map_offset: int, skin_map_size: int) -> list[int]:
    skin_map_size = int(skin_map_size)
    if skin_map_size <= 0:
        return []
    for start in (int(skin_map_offset) + 0x10, int(skin_map_offset)):
        if _valid_range(data, start, skin_map_size * 4):
            return [int(struct.unpack_from('<I', data, start + (i * 4))[0]) for i in range(skin_map_size)]
    return []


def _source_batch_vertex_format_templates(data: bytes | bytearray, header_offset: int, warning_cb=None) -> list[_TR8VertexFormatTemplate]:
    records = _read_source_mesh_record_metadata(data, int(header_offset))
    fallback = _choose_source_vertex_format_template(data, int(header_offset), warning_cb=warning_cb)
    result: list[_TR8VertexFormatTemplate] = []
    for index, record in enumerate(records):
        fmt_offset = 0
        if index + 1 < len(records):
            fmt_offset = int(records[index + 1].get('vertex_components_offset', 0) or 0)
        else:
            vertex_offset = int(record.get('vertex_offset', 0) or 0)
            fmt_offset = _find_source_vertex_format_before_stream(data, vertex_offset)
            if fmt_offset < 0:
                fmt_offset = int(record.get('vertex_components_offset', 0) or 0)
        tpl = _source_vertex_format_template_at(data, fmt_offset) if fmt_offset > 0 else None
        result.append(tpl if tpl is not None else fallback)
    return result or [fallback]


def _would_start_new_tr8_group(batch: _TR8Batch, material_index: int, draw_group: int) -> bool:
    if not batch.groups:
        return True
    return int(batch.groups[-1].material_index) != int(material_index) or int(batch.groups[-1].draw_group) != int(draw_group)


def _can_add_triangle_to_template_batch(batch: _TR8Batch, payloads: Sequence[_TR8VertexPayload], material_index: int, draw_group: int, group_capacity: int) -> bool:
    if int(group_capacity) <= 0:
        return False
    if _would_start_new_tr8_group(batch, int(material_index), int(draw_group)) and len(batch.groups) >= int(group_capacity):
        return False
    return _can_add_triangle_to_batch(batch, payloads)


def _append_triangle_to_existing_group(batch: _TR8Batch, group_index: int, payloads: Sequence[_TR8VertexPayload], *, double_sided: bool) -> None:
    local_indices: list[int] = []
    for payload in payloads:
        key = _vertex_key(payload)
        existing = batch.key_map.get(key)
        if existing is None:
            existing = len(batch.vertices)
            batch.vertices.append(payload)
            batch.key_map[key] = existing
            for bone_index in _triangle_skin_bones([payload]):
                if int(bone_index) not in batch.skin_map:
                    batch.skin_map.append(int(bone_index))
        local_indices.append(int(existing))
    while len(batch.groups) <= int(group_index):
        batch.groups.append(_TR8Group(material_index=0, draw_group=0))
    tri = tuple(local_indices[:3])
    batch.groups[int(group_index)].indices.extend(tri)
    if bool(double_sided):
        batch.groups[int(group_index)].indices.extend((tri[2], tri[1], tri[0]))


def _build_batches_for_source_layout(mesh_obj, arm_obj, model_root, source_records: Sequence[dict[str, int]], source_groups: Sequence[_TR8SourceGroupRecord], warning_cb=None) -> tuple[list[_TR8Batch], list, list[Vector]]:
    triangles_by_material, materials, positions = _collect_triangles_by_material(mesh_obj, arm_obj, model_root, warning_cb=warning_cb)
    capacities = [max(0, int(record.get('submesh_count', 0) or 0)) for record in source_records]
    if not capacities or not source_groups:
        return _build_batches(mesh_obj, arm_obj, model_root, warning_cb=warning_cb)

    batches = [_TR8Batch(index=index) for index in range(len(capacities))]
    source_slots_by_material: dict[int, list[tuple[int, int, _TR8SourceGroupRecord]]] = defaultdict(list)
    source_group_cursor = 0
    for batch_index, capacity in enumerate(capacities):
        for local_group_index in range(int(capacity)):
            if source_group_cursor < len(source_groups):
                src = source_groups[source_group_cursor]
            else:
                src = _TR8SourceGroupRecord(0, 0, 0, 0, 0, int(source_group_cursor), 0)
            batches[batch_index].groups.append(_TR8Group(material_index=int(src.material_index), draw_group=int(src.draw_group)))
            source_slots_by_material[int(src.material_index)].append((batch_index, local_group_index, src))
            source_group_cursor += 1

    def emitted_triangle_count(double_sided: bool) -> int:
        return 2 if bool(double_sided) else 1

    def group_emitted_triangles(batch: _TR8Batch, local_group_index: int) -> int:
        if local_group_index < 0 or local_group_index >= len(batch.groups):
            return 0
        return len(batch.groups[local_group_index].indices) // 3

    def can_add_to_slot(batch_index: int, local_group_index: int, src: _TR8SourceGroupRecord, payloads: Sequence[_TR8VertexPayload], double_sided: bool, *, respect_source_face_quota: bool) -> bool:
        batch = batches[int(batch_index)]
        if respect_source_face_quota:
            quota = max(0, int(src.num_faces))
            if quota > 0 and group_emitted_triangles(batch, int(local_group_index)) + emitted_triangle_count(bool(double_sided)) > quota:
                return False
        return _can_add_triangle_to_batch(batch, payloads)

    unassigned: list[tuple[int, Sequence[_TR8VertexPayload], bool, int]] = []
    for material_index in sorted(triangles_by_material.keys()):
        slots = list(source_slots_by_material.get(int(material_index), []))
        if not slots:
            unassigned.extend((int(material_index), payloads, bool(double_sided), int(draw_group)) for payloads, double_sided, draw_group in triangles_by_material[material_index])
            continue
        slot_cursor = 0
        for payloads, double_sided, draw_group in triangles_by_material[material_index]:
            triangle_bones = _triangle_skin_bones(payloads)
            if len(triangle_bones) > _TR8_MAX_WEIGHTED_BONES_PER_BATCH:
                raise ValueError(
                    f'Underworld triangle on material slot {material_index} needs {len(triangle_bones)} unique weighted bones, exceeding the engine batch limit of {_TR8_MAX_WEIGHTED_BONES_PER_BATCH}.'
                )
            placed = False
            for pass_respect_quota in (True, False):
                for probe in range(len(slots)):
                    slot_index = (slot_cursor + probe) % len(slots)
                    batch_index, local_group_index, src = slots[slot_index]
                    if not can_add_to_slot(batch_index, local_group_index, src, payloads, bool(double_sided), respect_source_face_quota=pass_respect_quota):
                        continue
                    _append_triangle_to_existing_group(batches[batch_index], local_group_index, payloads, double_sided=bool(double_sided))
                    slot_cursor = slot_index
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                unassigned.append((int(material_index), payloads, bool(double_sided), int(draw_group)))

    if unassigned:
        all_slots: list[tuple[int, int, _TR8SourceGroupRecord]] = []
        for entries in source_slots_by_material.values():
            all_slots.extend(entries)
        for material_index, payloads, double_sided, draw_group in unassigned:
            placed = False
            for batch_index, local_group_index, src in all_slots:
                if not _can_add_triangle_to_batch(batches[batch_index], payloads):
                    continue
                group = batches[batch_index].groups[local_group_index]
                if group.indices and int(group.material_index) != int(material_index):
                    continue
                group.material_index = int(material_index)
                group.draw_group = int(draw_group)
                _append_triangle_to_existing_group(batches[batch_index], local_group_index, payloads, double_sided=bool(double_sided))
                placed = True
                break
            if not placed:
                raise ValueError(
                    'This is ugly '
                )

    for index, batch in enumerate(batches):
        batch.index = int(index)
        if not batch.skin_map:
            batch.skin_map = [0]
        if len(batch.skin_map) > _TR8_MAX_WEIGHTED_BONES_PER_BATCH:
            raise ValueError(f'Underworld batch {index} uses {len(batch.skin_map)} weighted bones, exceeding the engine limit of {_TR8_MAX_WEIGHTED_BONES_PER_BATCH}.')
    return batches, materials, positions

def _read_source_skin_map_for_record(data: bytes | bytearray, record: dict[str, int]) -> list[int]:
    return _read_source_skin_palette(data, int(record.get('skin_map_offset', 0) or 0), int(record.get('skin_map_size', 0) or 0))


def _find_source_vertex_format_before_stream(data: bytes | bytearray, vertex_stream_offset: int, *, max_components: int = 64) -> int:
    vertex_stream_offset = int(vertex_stream_offset)
    if vertex_stream_offset <= 0:
        return -1
    for component_count in range(1, int(max_components) + 1):
        candidate = vertex_stream_offset - (0x20 + (component_count * 8))
        if candidate < 0:
            continue
        stride, components = _parse_source_vertex_format(data, candidate)
        if stride <= 0 or not components:
            continue
        format_end = (candidate + 0x20 + (len(components) * 8) + 15) & ~15
        if format_end == vertex_stream_offset:
            return int(candidate)
    return -1


def _source_vertex_stream_start(vertex_components_offset: int, vertex_offset: int, component_count: int) -> int:
    format_end = int(vertex_components_offset) + 0x20 + (int(component_count) * 8)
    aligned_format_end = (format_end + 15) & ~15
    return int(aligned_format_end)


def _read_source_mesh_groups(data: bytes | bytearray, mesh_group_offset: int, group_count: int) -> list[tuple[int, int, int, int, int, int, int]]:
    groups: list[tuple[int, int, int, int, int, int, int]] = []
    if int(group_count) <= 0 or not _valid_range(data, int(mesh_group_offset), int(group_count) * 64):
        return groups
    for index in range(int(group_count)):
        off = int(mesh_group_offset) + (index * 64) + 0x20
        base_index, num_faces, num_vertices, flags, draw_group, order, material_index = struct.unpack_from('<IIIIiiI', data, off)
        groups.append((int(base_index), int(num_faces), int(num_vertices), int(flags), int(draw_group), int(order), int(material_index)))
    return groups


def _read_source_skin_map(data: bytes | bytearray, skin_map_offset: int, skin_map_size: int) -> list[int]:
    if int(skin_map_size) <= 0:
        return []
    palette_offset = int(skin_map_offset) + 0x10
    if not _valid_range(data, palette_offset, int(skin_map_size) * 4):
        palette_offset = int(skin_map_offset)
        if not _valid_range(data, palette_offset, int(skin_map_size) * 4):
            return []
    return [int(struct.unpack_from('<I', data, palette_offset + (index * 4))[0]) for index in range(int(skin_map_size))]


def _parse_source_batches_from_template(data: bytes | bytearray, header_offset: int, warning_cb=None) -> tuple[list[_TR8SourceBatch], int, int, int]:
    if not _valid_range(data, int(header_offset), 0x84):
        return [], 0, 0, 0
    num_indices = _u32_at(data, int(header_offset) + 0x0C)
    mesh_group_offset, mesh_info_offset, _bone_map_offset, face_data_offset = struct.unpack_from('<IIII', data, int(header_offset) + 0x6C)
    mesh_group_count, mesh_record_count, _bone_count = struct.unpack_from('<HHH', data, int(header_offset) + 0x7C)
    if int(mesh_record_count) <= 0 or not _valid_range(data, int(mesh_info_offset), int(mesh_record_count) * 52):
        return [], int(num_indices), int(mesh_group_count), 0
    groups = _read_source_mesh_groups(data, int(mesh_group_offset), int(mesh_group_count))

    records: list[dict[str, int]] = []
    for index in range(int(mesh_record_count)):
        off = int(mesh_info_offset) + (index * 52)
        vertex_components_offset, num_vertices, base_index, num_faces, _u0, _sort, submesh_count, skin_map_size, skin_map_offset, vertex_offset = struct.unpack_from('<IIIIIIIII I'.replace(' ', ''), data, off)[:10]
        records.append({
            'index': index,
            'vertex_components_offset': int(vertex_components_offset),
            'num_vertices': int(num_vertices),
            'base_index': int(base_index),
            'num_faces': int(num_faces),
            'submesh_count': int(submesh_count),
            'skin_map_size': int(skin_map_size),
            'skin_map_offset': int(skin_map_offset),
            'vertex_offset': int(vertex_offset),
        })

    batches: list[_TR8SourceBatch] = []
    group_cursor = 0
    for record_index, meta in enumerate(records):
        group_count = max(0, int(meta.get('submesh_count', 0)))
        if group_count <= 0:
            continue
        batch_groups = groups[group_cursor:group_cursor + group_count]
        group_face_sum = sum(group[1] for group in batch_groups)
        geometry = records[record_index + 1] if record_index + 1 < len(records) else None

        vertex_components_offset = int(geometry['vertex_components_offset']) if geometry is not None else -1
        num_vertices = int(geometry['num_vertices']) if geometry is not None else 0
        num_faces = int(group_face_sum or (geometry['num_faces'] if geometry is not None else 0))
        if geometry is None or vertex_components_offset <= 0 or num_vertices <= 0:
            vertex_components_offset = _find_source_vertex_format_before_stream(data, int(meta['vertex_offset']))
            num_faces = int(group_face_sum)
            if vertex_components_offset >= 0:
                stride_probe, comps_probe = _parse_source_vertex_format(data, vertex_components_offset)
                stream_probe = _source_vertex_stream_start(vertex_components_offset, int(meta['vertex_offset']), len(comps_probe)) if comps_probe else int(meta['vertex_offset'])
                if stride_probe > 0:
                    available = max(0, (int(face_data_offset) + 0x10) - int(stream_probe))
                    num_vertices = max(0, available // int(stride_probe))
                    group_vertex_sum = sum(group[2] for group in batch_groups)
                    num_vertices = max(int(num_vertices), int(group_vertex_sum))

        stride, components = _parse_source_vertex_format(data, vertex_components_offset)
        vertex_stream_offset = _source_vertex_stream_start(vertex_components_offset, int(meta['vertex_offset']), len(components)) if components else int(meta['vertex_offset'])
        if stride <= 0 or not components or num_vertices <= 0:
            _warn(warning_cb, f'Underworld source batch {record_index} has no usable vertex stream; leaving it unchanged.')
            group_cursor += group_count
            continue
        if not _valid_range(data, vertex_stream_offset, int(num_vertices) * int(stride)):
            available = max(0, (int(face_data_offset) + 0x10) - int(vertex_stream_offset))
            clamped = available // int(stride)
            if clamped > 0:
                num_vertices = int(clamped)
        skin_map = _read_source_skin_map(data, int(meta['skin_map_offset']), int(meta['skin_map_size']))
        batches.append(_TR8SourceBatch(
            index=int(record_index),
            vertex_components_offset=int(vertex_components_offset),
            vertex_stream_offset=int(vertex_stream_offset),
            vertex_stride=int(stride),
            num_vertices=int(num_vertices),
            num_faces=int(num_faces),
            skin_map_offset=int(meta['skin_map_offset']),
            skin_map_size=int(meta['skin_map_size']),
            skin_map=skin_map,
            components=components,
        ))
        group_cursor += group_count
    return batches, int(num_indices), int(mesh_group_count), int(mesh_record_count)


def _collect_vertex_payloads_for_inplace(mesh_obj, arm_obj, model_root, warning_cb=None) -> tuple[list[_TR8VertexPayload], list[Vector]]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        raise ValueError(f'Underworld mesh object {getattr(mesh_obj, "name", "<mesh>")} has no mesh data')
    try:
        mesh.calc_loop_triangles()
        mesh.calc_normals_split()
        mesh.update(calc_edges=True)
    except Exception:
        pass

    uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
    color_attr = _active_color_attribute(mesh)
    arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else model_root.matrix_world.inverted_safe()
    warning_counter: dict[str, int] = {}
    first_loop_for_vertex: dict[int, int] = {}
    try:
        for loop in mesh.loops:
            first_loop_for_vertex.setdefault(int(loop.vertex_index), int(loop.index))
    except Exception:
        pass

    payloads: list[_TR8VertexPayload] = []
    positions: list[Vector] = []
    for vertex in mesh.vertices:
        index = int(vertex.index)
        world = mesh_obj.matrix_world @ vertex.co
        position = arm_inv @ world
        pos = Vector((float(position.x), float(position.y), float(position.z)))
        positions.append(pos)
        loop_index = int(first_loop_for_vertex.get(index, 0))
        uv = (0.0, 0.0)
        if uv_layer is not None:
            try:
                luv = uv_layer[loop_index].uv
                uv = (float(luv.x), float(luv.y))
            except Exception:
                uv = (0.0, 0.0)
        try:
            normal = Vector(mesh.loops[loop_index].normal) if mesh.loops and loop_index < len(mesh.loops) else Vector(vertex.normal)
        except Exception:
            try:
                normal = Vector(vertex.normal)
            except Exception:
                normal = Vector((0.0, 0.0, 1.0))
        payloads.append(_TR8VertexPayload(
            source_vertex=int(index),
            position=pos,
            normal=normal,
            uv=uv,
            color=_sample_color(color_attr, loop_index, index),
            weights=_weights_or_default(_bone_weights_for_vertex(mesh_obj, index, warning_counter=warning_counter)),
        ))

    if int(warning_counter.get('over_four', 0)) > 0:
        _warn(warning_cb, f'Underworld mesh "{getattr(mesh_obj, "name", "<mesh>")}" has {warning_counter["over_four"]} vertex/vertices with more than four weights; in-place export kept the strongest four and renormalized them.')
    return payloads, positions


def _write_payload_to_existing_vertex(data: bytearray, vertex_abs: int, stride: int, components: dict[int, _TR8SourceVertexComponent], payload: _TR8VertexPayload, skin_map: Sequence[int], warning_state: dict[str, int]) -> None:
    def component(semantic: int, size: int) -> int | None:
        comp = components.get(int(semantic))
        if comp is None:
            return None
        off = int(comp.offset)
        if off < 0 or off + int(size) > int(stride):
            return None
        return int(vertex_abs) + off

    off = component(TR8_VERTEX_POSITION, 12)
    if off is not None and _valid_range(data, off, 12):
        struct.pack_into('<fff', data, off, float(payload.position.x), float(payload.position.y), float(payload.position.z))

    off = component(TR8_VERTEX_NORMAL, 4)
    if off is not None and _valid_range(data, off, 4):
        nx, ny, nz = _normal_to_tuple(payload.normal)
        data[off:off + 4] = bytes((_normal_to_byte(nz), _normal_to_byte(ny), _normal_to_byte(nx), data[off + 3]))

    weights = _weights_or_default(payload.weights)
    if skin_map:
        remapped: list[tuple[int, float]] = []
        failed = False
        for bone, weight in weights:
            try:
                local_index = list(skin_map).index(int(bone))
            except ValueError:
                failed = True
                break
            remapped.append((int(local_index), float(weight)))
        if not failed and remapped:
            byte_weights = [max(0, min(255, int(round(float(weight) * 255.0)))) for _idx, weight in remapped[:4]]
            local_indices = [max(0, min(255, int(idx))) for idx, _weight in remapped[:4]]
            while len(byte_weights) < 4:
                byte_weights.append(0)
            while len(local_indices) < 4:
                local_indices.append(local_indices[0] if local_indices else 0)
            if sum(byte_weights) > 0:
                delta = 255 - sum(byte_weights)
                best = max(range(4), key=lambda idx: byte_weights[idx])
                byte_weights[best] = max(0, min(255, byte_weights[best] + delta))
            off = component(TR8_VERTEX_SKIN_WEIGHTS, 4)
            if off is not None and _valid_range(data, off, 4):
                data[off:off + 4] = bytes(byte_weights[:4])
            off = component(TR8_VERTEX_SKIN_INDICES, 4)
            if off is not None and _valid_range(data, off, 4):
                data[off:off + 4] = bytes(local_indices[:4])
        elif failed:
            warning_state['palette_miss'] = int(warning_state.get('palette_miss', 0)) + 1

    off = component(TR8_VERTEX_TEXCOORD1, 4)
    if off is not None and _valid_range(data, off, 4):
        u, v = payload.uv
        struct.pack_into('<HH', data, off, _uv_to_tr8_raw(float(u)), _uv_to_tr8_raw(1.0 - float(v)))

    off = component(TR8_VERTEX_COLOR1, 4)
    if off is not None and _valid_range(data, off, 4):
        r, g, b, a = _rgba_float_to_bytes(payload.color)
        data[off:off + 4] = bytes((b, g, r, a))


def _pack_tr8_mesh_data_inplace(mesh_obj, arm_obj, model_root, warning_cb=None, source_template: Path | None = None) -> tuple[bytes, list[_Relocation], int, int, int, int] | None:
    if source_template is None:
        return None
    try:
        _source_header, source_data, source_relocs = _read_standalone_section_payload(Path(source_template))
    except Exception as exc:
        _warn(warning_cb, f'Could not read Underworld source mesh template for safe in-place export: {exc}')
        return None
    header_offset = source_data.find(b'Mesh')
    if header_offset < 0 or not _valid_range(source_data, header_offset, 0x84):
        return None
    source_batches, num_indices, mesh_group_count, mesh_record_count = _parse_source_batches_from_template(source_data, header_offset, warning_cb=warning_cb)
    if not source_batches:
        return None
    source_vertex_count = sum(max(0, int(batch.num_vertices)) for batch in source_batches)
    payloads, positions = _collect_vertex_payloads_for_inplace(mesh_obj, arm_obj, model_root, warning_cb=warning_cb)
    if len(payloads) < int(source_vertex_count):
        raise ValueError(
            f'Underworld safe mesh export requires at least the original {source_vertex_count} imported vertices; current mesh has {len(payloads)}. '
            'Topology-changing Underworld export is not stable yet, so re-import the original model or avoid deleting vertices.'
        )

    out = bytearray(source_data)
    warning_state: dict[str, int] = {}
    global_index = 0
    patched_vertices = 0
    for batch in source_batches:
        for local_index in range(int(batch.num_vertices)):
            if global_index >= len(payloads):
                break
            vertex_abs = int(batch.vertex_stream_offset) + (local_index * int(batch.vertex_stride))
            if not _valid_range(out, vertex_abs, int(batch.vertex_stride)):
                global_index += 1
                continue
            _write_payload_to_existing_vertex(out, vertex_abs, int(batch.vertex_stride), batch.components, payloads[global_index], batch.skin_map, warning_state)
            patched_vertices += 1
            global_index += 1

    center, box_min, box_max, radius = _compute_bounds(positions[:max(1, min(len(positions), source_vertex_count))])
    struct.pack_into('<4f', out, header_offset + 0x10, *center)
    struct.pack_into('<4f', out, header_offset + 0x20, *box_min)
    struct.pack_into('<4f', out, header_offset + 0x30, *box_max)
    struct.pack_into('<f', out, header_offset + 0x40, float(radius))

    material_offset = _u32_at(out, header_offset + 0x08)
    material_count = _u32_at(out, material_offset + 0x14, 0) if _valid_range(out, material_offset, 0x18) else 0
    if int(warning_state.get('palette_miss', 0)) > 0:
        _warn(
            warning_cb,
            f'Underworld in-place export kept original skin weights for {warning_state["palette_miss"]} vertex/vertices whose edited bone weights do not fit the original 42-bone batch palette.'
        )
    logger.info(
        'Patched Underworld cdcModelData in-place from template %s: batches=%d meshRecords=%d meshGroups=%d vertices=%d indices=%d materials=%d',
        Path(source_template).name,
        len(source_batches),
        int(mesh_record_count),
        int(mesh_group_count),
        int(patched_vertices),
        int(num_indices),
        int(material_count),
    )
    return bytes(out), list(source_relocs), len(source_batches), int(patched_vertices), int(num_indices) // 3, int(material_count)
def _pack_vertex(payload: _TR8VertexPayload, skin_map: Sequence[int], fmt: _TR8VertexFormatTemplate | None = None) -> bytes:
    if fmt is None:
        components = {
            TR8_VERTEX_POSITION: _TR8SourceVertexComponent(TR8_VERTEX_POSITION, 0, 0x02, 0),
            TR8_VERTEX_NORMAL: _TR8SourceVertexComponent(TR8_VERTEX_NORMAL, 12, 0x05, 0),
            TR8_VERTEX_SKIN_WEIGHTS: _TR8SourceVertexComponent(TR8_VERTEX_SKIN_WEIGHTS, 16, 0x06, 0),
            TR8_VERTEX_SKIN_INDICES: _TR8SourceVertexComponent(TR8_VERTEX_SKIN_INDICES, 20, 0x07, 0),
            TR8_VERTEX_TEXCOORD1: _TR8SourceVertexComponent(TR8_VERTEX_TEXCOORD1, 24, 0x13, 0),
            TR8_VERTEX_TEXCOORD2: _TR8SourceVertexComponent(TR8_VERTEX_TEXCOORD2, 28, 0x13, 0),
            TR8_VERTEX_COLOR1: _TR8SourceVertexComponent(TR8_VERTEX_COLOR1, 32, 0x04, 0),
            TR8_VERTEX_BINORMAL: _TR8SourceVertexComponent(TR8_VERTEX_BINORMAL, 36, 0x05, 0),
            TR8_VERTEX_TANGENT: _TR8SourceVertexComponent(TR8_VERTEX_TANGENT, 40, 0x05, 0),
        }
        stride = _TR8_VERTEX_STRIDE
    else:
        components = fmt.components
        stride = int(fmt.stride)
    data = bytearray(max(0, int(stride)))

    def component(semantic: int, size: int) -> int | None:
        comp = components.get(int(semantic))
        if comp is None:
            return None
        off = int(comp.offset)
        if off < 0 or off + int(size) > len(data):
            return None
        return off

    off = component(TR8_VERTEX_POSITION, 12)
    if off is not None:
        struct.pack_into('<fff', data, off, float(payload.position.x), float(payload.position.y), float(payload.position.z))

    nx, ny, nz = _normal_to_tuple(payload.normal)
    encoded_normal = bytes((_normal_to_byte(nz), _normal_to_byte(ny), _normal_to_byte(nx), 0))
    off = component(TR8_VERTEX_NORMAL, 4)
    if off is not None:
        data[off:off + 4] = encoded_normal

    weights = _weights_or_default(payload.weights)
    byte_weights: list[int] = []
    local_indices: list[int] = []
    palette = list(skin_map)
    for bone, weight in weights[:_TR8_MAX_WEIGHTS_PER_VERTEX]:
        byte_weights.append(max(0, min(255, int(round(float(weight) * 255.0)))))
        try:
            local_indices.append(palette.index(int(bone)))
        except ValueError:
            local_indices.append(0)
    while len(byte_weights) < _TR8_MAX_WEIGHTS_PER_VERTEX:
        byte_weights.append(0)
    while len(local_indices) < _TR8_MAX_WEIGHTS_PER_VERTEX:
        local_indices.append(local_indices[0] if local_indices else 0)
    if sum(byte_weights) > 0:
        delta = 255 - sum(byte_weights)
        best = max(range(_TR8_MAX_WEIGHTS_PER_VERTEX), key=lambda idx: byte_weights[idx])
        byte_weights[best] = max(0, min(255, byte_weights[best] + delta))
    off = component(TR8_VERTEX_SKIN_WEIGHTS, 4)
    if off is not None:
        data[off:off + 4] = bytes(byte_weights[:4])
    off = component(TR8_VERTEX_SKIN_INDICES, 4)
    if off is not None:
        data[off:off + 4] = bytes(max(0, min(255, int(value))) for value in local_indices[:4])

    u, v = payload.uv
    uv_bytes = struct.pack('<HH', _uv_to_tr8_raw(float(u)), _uv_to_tr8_raw(1.0 - float(v)))
    off = component(TR8_VERTEX_TEXCOORD1, 4)
    if off is not None:
        data[off:off + 4] = uv_bytes
    off = component(TR8_VERTEX_TEXCOORD2, 4)
    if off is not None:
        data[off:off + 4] = uv_bytes

    off = component(TR8_VERTEX_COLOR1, 4)
    if off is not None:
        r, g, b, a = _rgba_float_to_bytes(payload.color)
        data[off:off + 4] = bytes((b, g, r, a))

    binormal = bytes((_normal_to_byte(0.0), _normal_to_byte(1.0), _normal_to_byte(0.0), 0))
    tangent = bytes((_normal_to_byte(0.0), _normal_to_byte(0.0), _normal_to_byte(1.0), 0))
    off = component(TR8_VERTEX_BINORMAL, 4)
    if off is not None:
        data[off:off + 4] = binormal
    off = component(TR8_VERTEX_TANGENT, 4)
    if off is not None:
        data[off:off + 4] = tangent
    return bytes(data)


def _pack_material_table(materials: Sequence) -> tuple[bytes, int]:
    material_ids = [_underworld_material_id(material, index) for index, material in enumerate(materials)]
    count = len(material_ids)
    data = bytearray(0x18 + (count * 4))
    struct.pack_into('<III', data, 0x00, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into('<I', data, 0x14, count)
    for index, material_id in enumerate(material_ids):
        struct.pack_into('<I', data, 0x18 + index * 4, int(material_id) & 0xFFFFFFFF)
    return bytes(data), count


def _pack_skeleton_records(arm_obj, batches: Sequence[_TR8Batch], warning_cb=None, source_template: Path | None = None, minimum_count: int = 0) -> tuple[bytes, int]:
    if source_template is not None:
        copied = _extract_skeleton_records_from_template(Path(source_template), warning_cb=warning_cb)
        if copied is not None:
            return copied

    if arm_obj is None or getattr(arm_obj, 'data', None) is None:
        count = max(1, int(minimum_count or 0))
        data = bytearray(0x10 + (count * 64))
        struct.pack_into('<IIII', data, 0, int(count), 0x10, 0, 0)
        for index in range(int(count)):
            record = 0x10 + (index * 64)
            struct.pack_into('<fff', data, record + 0x20, 0.0, 0.0, 0.0)
            struct.pack_into('<hhii', data, record + 0x34, 0, -1, -1, 0)
        return bytes(data), int(count)

    bones_by_index: dict[int, object] = {}
    for bone in getattr(arm_obj.data, 'bones', []) or []:
        bone_index = _bone_index_from_group_name(getattr(bone, 'name', ''))
        if bone_index is not None:
            bones_by_index[int(bone_index)] = bone
    if not bones_by_index:
        return _pack_skeleton_records(None, batches, minimum_count=minimum_count)
    count = max(max(bones_by_index) + 1, int(minimum_count or 0))
    data = bytearray(0x10 + (count * 64))
    struct.pack_into('<II', data, 0, int(count), 0x10)
    for index in range(count):
        bone = bones_by_index.get(index)
        parent_index = -1
        pivot = Vector((0.0, 0.0, 0.0))
        if bone is not None:
            try:
                head = Vector(getattr(bone, 'head_local', (0.0, 0.0, 0.0)))
                parent = getattr(bone, 'parent', None)
                if parent is not None:
                    pidx = _bone_index_from_group_name(getattr(parent, 'name', ''))
                    if pidx is not None:
                        parent_index = int(pidx)
                        parent_head = Vector(getattr(parent, 'head_local', (0.0, 0.0, 0.0)))
                        pivot = head - parent_head
                    else:
                        pivot = head
                else:
                    pivot = head
            except Exception:
                pivot = Vector((0.0, 0.0, 0.0))
        record = 0x10 + (index * 64)
        struct.pack_into('<fff', data, record + 0x20, float(pivot.x), float(pivot.y), float(pivot.z))
        struct.pack_into('<hhii', data, record + 0x34, 0, -1, int(parent_index), 0)
    return bytes(data), int(count)


def _pack_tr8_mesh_data(mesh_obj, arm_obj, model_root, render_id: int, warning_cb=None, source_template: Path | None = None) -> tuple[bytes, list[_Relocation], int, int, int, int, int]:
    del render_id
    source_template_path = Path(source_template) if source_template is not None else None
    header_opaque_block = _source_mesh_header_opaque_block(source_template_path)
    source_bone_count = _source_mesh_bone_count(source_template_path)

    batches, materials, positions = _build_batches(mesh_obj, arm_obj, model_root, warning_cb=warning_cb)
    if not batches or not any(batch.triangle_count() > 0 for batch in batches):
        raise ValueError(f'Underworld mesh "{getattr(mesh_obj, "name", "<mesh>")}" contains no exportable triangles')

    material_table, material_count = _pack_material_table(materials)
    skeleton_blob, bone_count = _pack_skeleton_records(arm_obj, batches, warning_cb=warning_cb, source_template=None, minimum_count=source_bone_count)
    bone_count = max(1, int(bone_count))
    vertex_format_template = _scratch_vertex_format_template()
    center, box_min, box_max, radius = _compute_bounds(positions)

    data = bytearray()
    relocations: list[_Relocation] = []

    def add_self_reloc(offset: int) -> None:
        relocations.append(_Relocation(target_section_index=0, offset=int(offset), relocation_type=_RELOCATION_POINTER))

    runtime_root_offset = _align(data, 16)
    data.extend(b'\x00' * 0x10)

    header_offset = _align(data, 16)
    data.extend(b'\x00' * 0x84)

    bone_map_offset = _align(data, 16)
    for bone_index in range(bone_count):
        data.extend(struct.pack('<I', int(bone_index) & 0xFFFFFFFF))

    mesh_info_offset = _align(data, 32)
    mesh_record_count = len(batches)
    data.extend(b'\x00' * (mesh_record_count * 52))

    skin_offsets: list[int] = []
    vertex_format_offsets: list[int] = []
    vertex_offsets: list[int] = []
    for batch in batches:
        skin_offset = _align(data, 16)
        skin_offsets.append(skin_offset)
        data.extend(b'\x00' * 0x10)
        for bone in batch.skin_map:
            data.extend(struct.pack('<I', int(bone) & 0xFFFFFFFF))

        fmt_offset = _align(data, 16)
        vertex_format_offsets.append(fmt_offset)
        data.extend(vertex_format_template.blob)
        vertex_offset = _align(data, 16)
        vertex_offsets.append(vertex_offset)
        for payload in batch.vertices:
            data.extend(_pack_vertex(payload, batch.skin_map, vertex_format_template))

    face_data_offset = _align(data, 16)
    data.extend(struct.pack('<IIII', 0x01F70672, 0xFFFFFFFF, 0x7FA08D05, 0x7F61FB85))

    num_indices = 0
    batch_index_starts: list[int] = []
    batch_face_counts: list[int] = []
    group_records: list[tuple[_TR8Group, int, int, int]] = []
    for batch_index, batch in enumerate(batches):
        batch_index_starts.append(num_indices)
        batch_index_count = 0
        for group in batch.groups:
            group_base = num_indices
            for index in group.indices:
                data.extend(struct.pack('<H', int(index) & 0xFFFF))
                num_indices += 1
                batch_index_count += 1
            group_records.append((group, int(group_base), int(batch_index), int(len(group_records))))
        batch_face_counts.append(batch_index_count // 3)

    mesh_group_offset = _align(data, 16)
    total_groups = len(group_records)
    data.extend(b'\x00' * (total_groups * 64))

    for group_cursor, (group, group_base, batch_index, _global_order) in enumerate(group_records):
        group_offset = mesh_group_offset + (group_cursor * 64)
        unique_indices = sorted({int(i) for i in group.indices})
        group_vertex_count = len(unique_indices)
        material_index = int(group.material_index)
        if material_index < 0 or material_index >= int(material_count):
            _warn(warning_cb, f'Underworld scratch export material slot {material_index} is outside the generated material table; remapped to slot 0.')
            material_index = 0
        center_x = center_y = center_z = 0.0
        if 0 <= int(batch_index) < len(batches) and unique_indices:
            verts = batches[int(batch_index)].vertices
            samples = [verts[i].position for i in unique_indices if 0 <= int(i) < len(verts)]
            if samples:
                inv = 1.0 / float(len(samples))
                center_x = sum(float(v.x) for v in samples) * inv
                center_y = sum(float(v.y) for v in samples) * inv
                center_z = sum(float(v.z) for v in samples) * inv
        struct.pack_into('<4f', data, group_offset + 0x00, 0.0, 0.0, 0.0, 0.0)
        struct.pack_into('<4f', data, group_offset + 0x10, float(center_x), float(center_y), float(center_z), 1.0)
        struct.pack_into('<IIIIiiII', data, group_offset + 0x20,
            int(group_base),
            int(group.triangle_count),
            int(group_vertex_count),
            0,
            int(group.draw_group),
            0,
            int(material_index),
            0xFFFFFFFF,
        )

    for batch_index, batch in enumerate(batches):
        rec = mesh_info_offset + (batch_index * 52)
        if batch_index > 0:
            geom_batch = batches[batch_index - 1]
            geom_fmt = int(vertex_format_offsets[batch_index - 1])
            geom_vertices = len(geom_batch.vertices)
            geom_base_index = int(batch_index_starts[batch_index - 1])
            geom_faces = int(batch_face_counts[batch_index - 1])
        else:
            geom_fmt = 0
            geom_vertices = 0
            geom_base_index = 0
            geom_faces = 0
        struct.pack_into('<IIIIIIIII', data, rec + 0x00,
            int(geom_fmt),
            int(geom_vertices),
            int(geom_base_index),
            int(geom_faces),
            0,
            0x7F7FFFFF,
            len(batch.groups),
            len(batch.skin_map),
            int(skin_offsets[batch_index]),
        )
        struct.pack_into('<I', data, rec + 0x24, int(vertex_offsets[batch_index]))

    material_offset = _align(data, 16)
    data.extend(material_table)

    runtime_aux_offset = _align(data, 16)
    top_array_offset = runtime_aux_offset + 0x08
    top_record_offset = top_array_offset
    node_count = max(1, total_groups)
    node_array_offset = _align(bytearray(b'\x00' * (top_record_offset + 0x14)), 16)
    node_array_offset = (top_record_offset + 0x14 + 0x0F) & ~0x0F
    edge_table_offset = node_array_offset + (node_count * 0x30)
    edge_table_size = max(0x10, int(node_count) * 8)
    shape_table_offset = (edge_table_offset + edge_table_size + 0x0F) & ~0x0F
    runtime_end_min = shape_table_offset + (node_count * 0x50)
    if len(data) < runtime_end_min:
        data.extend(b'\x00' * (runtime_end_min - len(data)))

    struct.pack_into('<II', data, runtime_aux_offset + 0x00, int(top_array_offset), 1)
    add_self_reloc(runtime_aux_offset + 0x00)
    packed_runtime_flags = 0x00060000 | (int(node_count) & 0xFFFF)
    struct.pack_into('<IIIII', data, top_record_offset,
        int(node_array_offset),
        int(edge_table_offset),
        0x71,
        int(packed_runtime_flags) & 0xFFFFFFFF,
        int(node_count) & 0xFFFF,
    )
    add_self_reloc(top_record_offset + 0x00)
    add_self_reloc(top_record_offset + 0x04)
    for edge_index in range(int(node_count)):
        struct.pack_into('<fHH', data, edge_table_offset + (edge_index * 8), 0.0, int(edge_index) & 0xFFFF, int(edge_index) & 0xFFFF)

    cx, cy, cz, _cw = center
    for node_index in range(node_count):
        node_offset = node_array_offset + (node_index * 0x30)
        shape_offset = shape_table_offset + (node_index * 0x50)
        struct.pack_into('<4f', data, node_offset + 0x00, float(cx), float(cy), float(cz), 0.0)
        struct.pack_into('<II', data, node_offset + 0x10, 0, 0)
        struct.pack_into('<I', data, node_offset + 0x18, int(shape_offset))
        struct.pack_into('<I', data, node_offset + 0x1C, ((0x74 + node_index) & 0xFFFF) << 16)
        struct.pack_into('<I', data, node_offset + 0x20, int(node_index))
        add_self_reloc(node_offset + 0x18)
        struct.pack_into('<IIII', data, shape_offset, 4, 0, 0, 0)
        vectors = [
            (float(box_min[0]), float(box_min[1]), float(box_min[2]), 0.0),
            (float(box_max[0]), float(box_min[1]), float(box_min[2]), 1.0),
            (float(box_min[0]), float(box_max[1]), float(box_max[2]), 2.0),
            (float(box_max[0]), float(box_max[1]), float(box_max[2]), 3.0),
        ]
        for vec_index, vec in enumerate(vectors):
            struct.pack_into('<4f', data, shape_offset + 0x10 + (vec_index * 0x10), *vec)

    skeleton_offset = _align(data, 16)
    data.extend(skeleton_blob)
    struct.pack_into('<I', data, skeleton_offset + 0x04, int(skeleton_offset + 0x10))
    add_self_reloc(skeleton_offset + 0x04)

    struct.pack_into('<IIII', data, runtime_root_offset,
        int(header_offset),
        int(material_offset) + 0x10,
        int(material_offset) + 0x14,
        int(runtime_aux_offset),
    )
    add_self_reloc(runtime_root_offset + 0x00)
    add_self_reloc(runtime_root_offset + 0x04)
    add_self_reloc(runtime_root_offset + 0x08)
    add_self_reloc(runtime_root_offset + 0x0C)

    struct.pack_into('<4sIII', data, header_offset + 0x00, b'Mesh', 1, int(material_offset), int(num_indices))
    struct.pack_into('<4f', data, header_offset + 0x10, *center)
    struct.pack_into('<4f', data, header_offset + 0x20, *box_min)
    struct.pack_into('<4f', data, header_offset + 0x30, *box_max)
    struct.pack_into('<f', data, header_offset + 0x40, float(radius))
    struct.pack_into('<I', data, header_offset + 0x44, 1)
    if len(header_opaque_block) == 0x24:
        data[header_offset + 0x48:header_offset + 0x6C] = header_opaque_block
    struct.pack_into('<IIII', data, header_offset + 0x6C, int(mesh_group_offset), int(mesh_info_offset), int(bone_map_offset), int(face_data_offset))
    struct.pack_into('<HHH', data, header_offset + 0x7C, int(total_groups), int(mesh_record_count), int(bone_count))

    vertex_count = sum(len(batch.vertices) for batch in batches)
    triangle_count = sum(batch.triangle_count() for batch in batches)
    logger.info(
        'Built Underworld cdcModelData fully from scratch for %s: batches=%d groups=%d vertices=%d triangles=%d materials=%d bones=%d relocs=%d',
        getattr(mesh_obj, 'name', '<mesh>'),
        len(batches),
        int(total_groups),
        int(vertex_count),
        int(triangle_count),
        int(material_count),
        int(bone_count),
        len(relocations),
    )
    return bytes(data), relocations, len(batches), vertex_count, triangle_count, int(material_count), int(skeleton_offset)


def _patch_tr8_oldmodel_skeleton_references(directory: Path, target_section_index: int, skeleton_offset: int, *, warning_cb=None) -> int:
    directory = Path(directory)
    target_section_index = int(target_section_index)
    skeleton_offset = int(skeleton_offset) & 0xFFFFFFFF
    patched = 0
    for path in sorted(directory.iterdir() if directory.exists() else []):
        if not path.is_file() or path.suffix.lower() not in {'.gnc', '.obj'}:
            continue
        try:
            template, payload, relocs = _read_standalone_section_payload(path)
        except Exception:
            continue
        if int(template.section_type) != 7:
            continue
        changed = False
        data = bytearray(payload)
        for reloc in relocs:
            if int(reloc.relocation_type) != _RELOCATION_POINTER:
                continue
            if int(reloc.target_section_index) != target_section_index:
                continue
            field_local = int(reloc.offset)
            if field_local != 0x1A:
                continue
            if field_local < 0 or field_local + 4 > len(data):
                continue
            struct.pack_into('<I', data, field_local, skeleton_offset)
            changed = True
        if changed:
            _write_standalone_section(path, template, bytes(data), relocs)
            patched += 1
            logger.info('Patched Underworld oldModel skeleton relocation %s -> cdcModelData section %d local 0x%X', path.name, target_section_index, skeleton_offset)
    if patched == 0:
        _warn(warning_cb, f'No Underworld oldModel skeleton external relocation was found for cdcModelData section {target_section_index}; export will rely only on cdcModelData self-relocations.')
    return patched


def export_pc_underworld_model_section(
    context,
    directory: Path,
    model_root,
    arm_obj,
    mesh_obj,
    render_id: int,
    *,
    section_list=None,
    template_path: Path | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> UnderworldExportResult | None:
    del context
    directory = Path(directory)
    render_id = int(render_id) & 0xFFFFFFFF
    if render_id == 0 or mesh_obj is None:
        return None

    source_template = Path(template_path) if template_path is not None else _find_tr8mesh_template(directory, render_id)
    if source_template is None:
        raise FileNotFoundError(f'Could not find an Underworld cdcModelData/tr8mesh section with resource ID 0x{render_id:X}')
    template = _read_section_template(source_template)
    section_index = _section_index_from_filename(source_template)
    template.section_index = int(section_index)
    if int(template.section_id) == 0:
        template.section_id = int(render_id) & 0xFFFFFFFF
    data, relocations, batch_count, vertex_count, triangle_count, material_count, skeleton_offset = _pack_tr8_mesh_data(mesh_obj, arm_obj, model_root, render_id, warning_cb=warning_callback, source_template=source_template)
    for reloc in relocations:
        reloc.target_section_index = int(section_index)
    _write_standalone_section(source_template, template, data, relocations)
    _patch_tr8_oldmodel_skeleton_references(directory, int(section_index), int(skeleton_offset), warning_cb=warning_callback)
    if section_list is not None:
        try:
            section_list.ensure_index(section_index, source_template.name)
        except Exception:
            pass
    logger.info(
        'Wrote Underworld cdcModelData section %s RenderID=0x%X batches=%d vertices=%d triangles=%d materials=%d',
        source_template.name,
        int(render_id),
        int(batch_count),
        int(vertex_count),
        int(triangle_count),
        int(material_count),
    )
    return UnderworldExportResult(source_template, render_id, batch_count, vertex_count, triangle_count, material_count)
