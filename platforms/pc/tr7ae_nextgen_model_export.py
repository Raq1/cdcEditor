from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import bpy
from mathutils import Matrix, Vector

from ...core.log import logger
from ...core.material_ui import decode_tpage_flags, get_material_draw_group, get_material_platform_name, get_material_tpageid, sync_pc_nextgen_material_from_panel, _pc_nextgen_panel_storage_value, set_material_panel_value
from ..common.section import SectionContextCache
from .tr7ae_nextgen_model import (
    D3DDECLTYPE_D3DCOLOR,
    D3DDECLTYPE_FLOAT1,
    D3DDECLTYPE_FLOAT2,
    D3DDECLTYPE_FLOAT3,
    D3DDECLTYPE_SHORT2,
    D3DDECLTYPE_UNUSED,
    D3DDECLUSAGE_BLENDINDICES,
    D3DDECLUSAGE_BLENDWEIGHT,
    D3DDECLUSAGE_COLOR,
    D3DDECLUSAGE_NORMAL,
    D3DDECLUSAGE_POSITION,
    D3DDECLUSAGE_TEXCOORD,
    TRNextGenModelParser,
)


_SECTION_NAME_RE = None
_PC_NEXTGEN_MATERIAL_STRIDE = 456
_PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS = 0x00221A00
_PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS_SPECULAR = 0x00225A00
_PC_NEXTGEN_DEFAULT_SHADER_INDICES_LAYERED = (0, 1, 2, 3, 4, 6, 0, 0)
_PC_NEXTGEN_DEFAULT_SHADER_INDICES_DIFFUSE_ONLY = (7, 8, 9, 10, 11, 12, 0, 0)
_PC_NEXTGEN_NO_FX_MATERIAL_OFFSET = 0xFFFFFFFF
_PC_NEXTGEN_DEFAULT_SHADER_SECTION_IDS = [int(value) for value in range(660, 683)]



@dataclass(slots=True)
class NextGenExportResult:
    path: Path
    render_id: int
    vertex_count: int
    face_count: int
    material_count: int


def _warn(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        try:
            callback(message)
            return
        except Exception:
            pass
    logger.warning(message)


def is_pc_nextgen_mesh_object(obj) -> bool:
    if obj is None or getattr(obj, 'type', None) != 'MESH':
        return False
    # Classify the mesh object itself, not the materials assigned to it.
    #
    # Older builds treated any mesh using a PC Next-Gen material as a next-gen
    # mesh.  That is too broad now that materials can be converted/authored
    # from the main Material panel: an ordinary model mesh may temporarily hold
    # PC Next-Gen materials, and the old-gen exporter must still see that mesh.
    #
    # Next-gen render meshes are identified by explicit object/data metadata or
    # the import/export naming convention.
    try:
        if bool(getattr(obj, 'trlau_pc_nextgen_mesh', False)) or bool(obj.get('trlau_pc_nextgen_mesh', False)):
            return True
    except Exception:
        pass
    try:
        data = getattr(obj, 'data', None)
        if data is not None and (bool(getattr(data, 'trlau_pc_nextgen_mesh', False)) or bool(data.get('trlau_pc_nextgen_mesh', False))):
            return True
    except Exception:
        pass
    name = str(getattr(obj, 'name', '') or '')
    if '_Mesh_NextGen_' in name or name.endswith('_Mesh_NextGen') or name.endswith('_NextGen'):
        return True
    return False


def _iter_descendants(root) -> Iterable[object]:
    for child in list(getattr(root, 'children_recursive', []) or []):
        yield child


def find_pc_nextgen_meshes(model_root, render_id: int | None = None) -> List[object]:
    result: List[object] = []
    for obj in _iter_descendants(model_root):
        if not is_pc_nextgen_mesh_object(obj):
            continue
        if render_id is not None and int(render_id or 0) > 0:
            try:
                obj_render_id = int(getattr(obj, 'trlau_ui_cdc_render_data_id', obj.get('trlau_cdc_render_data_id', 0)) or 0) & 0xFFFFFFFF
            except Exception:
                obj_render_id = 0
            if obj_render_id not in {0, int(render_id) & 0xFFFFFFFF}:
                continue
        result.append(obj)
    return result


def _section_index_from_filename(path: Path) -> int:
    stem = Path(path).stem
    head = stem.split('_', 1)[0]
    return int(head)


def _standalone_section_info(path: Path):
    cache = SectionContextCache(str(path), endian='<')
    try:
        ctx = cache.get_root_context()
        return ctx.section_info
    finally:
        cache.close()


def find_nextgen_section_template(directory: Path, render_id: int) -> Path | None:
    directory = Path(directory)
    rid = int(render_id) & 0xFFFFFFFF
    candidates: list[tuple[int, int, Path]] = []
    for path in sorted(directory.iterdir() if directory.exists() else []):
        if not path.is_file() or path.suffix.lower() not in {'.gnc', '.ngm', '.ngmesh', '.tr7ngmesh'}:
            continue
        try:
            info = _standalone_section_info(path)
        except Exception:
            continue
        try:
            if int(getattr(info, 'section_type', -1)) != 7:
                continue
            if (int(getattr(info, 'section_id', 0)) & 0xFFFFFFFF) != rid:
                continue
            spec_rank = 0 if (int(getattr(info, 'spec_mask', 0)) & 0x80000000) else 1
            size_rank = -int(getattr(info, 'size', 0))
            candidates.append((spec_rank, size_rank, path))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[0][2]


def _read_template_parser(path: Path, render_id: int):
    parser = TRNextGenModelParser(str(path), cdc_render_data_id=int(render_id))
    # parse() sets the parser's resolved standalone-data offsets and validates
    # that this is really a PCD9 next-gen section.  The returned ModelData is not
    # used by the exporter; the parser state is used as a template map below.
    parser.parse()
    header_offsets = parser._read_nextgen_header_offsets()
    pc_base = parser._find_pcmodeldata_base(header_offsets[0])
    header = parser._parse_header(pc_base)
    pixmaps = parser._parse_pixmaps(header_offsets[1])
    special_flags = parser._parse_int_table(header_offsets[3])
    materials = parser._parse_materials(header, pixmaps, special_flags)
    prim_groups = parser._parse_prim_groups(header)
    batches = parser._parse_batches(header)
    index_data_offset, index_data_extra, _score = parser._infer_index_data_offset(header, batches, prim_groups)
    raw_indices = parser._read_raw_indices_at(index_data_offset, int(header.num_indices))
    return parser, header_offsets, header, pixmaps, materials, prim_groups, batches, index_data_offset, index_data_extra, raw_indices


def _safe_int(value, default: int = 0) -> int:
    try:
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith('0x'):
                return int(text, 16)
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _csv_ints(value, count: int, default: int = 0) -> list[int]:
    result: list[int] = []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',')]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = []
    for part in parts:
        if part == '':
            continue
        result.append(_safe_int(part, default))
    while len(result) < int(count):
        result.append(int(default))
    return result[:int(count)]


def _json_colors(value, count: int, default=(0.0, 0.0, 0.0, 0.0)) -> list[tuple[float, float, float, float]]:
    parsed = []
    try:
        parsed = json.loads(value) if isinstance(value, str) else list(value or [])
    except Exception:
        parsed = []
    result = []
    for item in parsed:
        try:
            values = list(item)
            rgba = tuple(_safe_float(values[i], default[i] if i < len(default) else 0.0) for i in range(4))
        except Exception:
            rgba = tuple(float(v) for v in default)
        result.append(rgba)
    while len(result) < int(count):
        result.append(tuple(float(v) for v in default))
    return result[:int(count)]


def _material_prop(material, name: str, default=None):
    if material is None:
        return default
    try:
        if name in material:
            return material.get(name, default)
    except Exception:
        pass
    if str(name).startswith('trlau_pc_nextgen_'):
        try:
            value = _pc_nextgen_panel_storage_value(material, name, None)
            if value is not None:
                return value
        except Exception:
            pass
    try:
        value = getattr(material, name, None)
        if value is not None:
            return value
    except Exception:
        pass
    # Compatibility fallback: a next-gen mesh may intentionally use a PC
    # Old-Gen/PS3/Xbox material.  Treat the material Type as an authoring UI
    # layout, not a blocker.  Missing PC Next-Gen fields are synthesized here
    # so export can continue without forcing the user to convert materials.
    try:
        fallback = _pc_nextgen_fallback_material_prop(material, name, default)
        return fallback
    except Exception:
        return default


def _has_pc_nextgen_layout(material) -> bool:
    if material is None:
        return False
    try:
        if str(get_material_platform_name(material) or '').lower() == 'pc_nextgen':
            return True
    except Exception:
        pass
    try:
        return any(key in material for key in (
            'trlau_pc_nextgen_material_id',
            'trlau_pc_nextgen_layer_texture_ids',
            'trlau_pc_nextgen_diffuse_texture_id',
            'trlau_pc_nextgen_shader_applied',
        ))
    except Exception:
        return False


def _pc_nextgen_maybe_sync_material(material) -> None:
    # sync_pc_nextgen_material_from_panel() intentionally switches the material
    # platform to PC Next-Gen.  Only call it when the material already has that
    # layout.  Old-gen/console materials assigned to a next-gen mesh are adapted
    # non-destructively through fallback properties instead.
    if not _has_pc_nextgen_layout(material):
        return
    try:
        sync_pc_nextgen_material_from_panel(material)
    except Exception:
        pass


def _generic_tpage_flags(material) -> dict:
    try:
        return dict(decode_tpage_flags(get_material_tpageid(material)))
    except Exception:
        return {'texture_id': 0, 'blend_value': 0, 'single_sided': 0, 'alpha_ref': 0}


def _generic_material_texture_ids(material) -> list[int]:
    if material is None:
        return [-1] * 8
    if _has_pc_nextgen_layout(material):
        try:
            raw_ids = _material_prop(material, 'trlau_pc_nextgen_layer_texture_ids', '')
        except Exception:
            raw_ids = ''
        if raw_ids:
            ids = _csv_ints(raw_ids, 8, -1)
            if any(int(value) >= 0 for value in ids):
                return [int(value) for value in ids[:8]]
    platform = str(get_material_platform_name(material) or '').lower()
    diffuse_id = -1
    normal_id = -1
    specular_id = -1
    if platform in {'ps3', 'xbox360'}:
        diffuse_id = _safe_int(_material_prop(material, 'trlau_ui_ps3_diffuse_texture_id', -1), -1)
        normal_id = _safe_int(_material_prop(material, 'trlau_ui_ps3_normal_texture_id', -1), -1)
        specular_id = _safe_int(_material_prop(material, 'trlau_ui_ps3_specular_texture_id', -1), -1)
    if diffuse_id < 0:
        diffuse_id = int(_generic_tpage_flags(material).get('texture_id', 0) or 0)
    ids = [diffuse_id, normal_id, specular_id] + [-1] * 5
    return [int(value) for value in ids[:8]]


def _generic_pc_nextgen_layer_enabled(material) -> list[int]:
    if _has_pc_nextgen_layout(material):
        values = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_enabled', ''), 8, 0)
        if any(int(value) != 0 for value in values):
            return [int(value) for value in values[:8]]
    ids = _generic_material_texture_ids(material)
    return [1 if int(value) >= 0 else 0 for value in ids[:8]]


def _pc_nextgen_fallback_material_prop(material, name: str, default=None):
    ids = None
    if name in {'trlau_pc_nextgen_layer_texture_ids', 'trlau_pc_nextgen_layer_enabled', 'trlau_pc_nextgen_layer_num_textures'}:
        ids = _generic_material_texture_ids(material)
    if name == 'trlau_pc_nextgen_layer_texture_ids':
        return _csv_join_ints(ids, 8)
    if name == 'trlau_pc_nextgen_layer_enabled':
        return _csv_join_ints([1 if int(value) >= 0 else 0 for value in ids[:8]], 8)
    if name == 'trlau_pc_nextgen_layer_num_textures':
        return _csv_join_ints([1 if int(value) >= 0 else 0 for value in ids[:8]], 8)
    if name == 'trlau_pc_nextgen_diffuse_texture_id':
        return _generic_material_texture_ids(material)[0]
    if name == 'trlau_pc_nextgen_normal_texture_id':
        return _generic_material_texture_ids(material)[1]
    if name == 'trlau_pc_nextgen_specular_texture_id':
        return _generic_material_texture_ids(material)[2]
    if name == 'trlau_pc_nextgen_material_id':
        return int(get_material_draw_group(material)) + 1
    if name == 'trlau_pc_nextgen_blend_mode':
        flags = _generic_tpage_flags(material)
        if int(flags.get('alpha_ref', 0) or 0):
            return 1
        if int(flags.get('blend_value', 0) or 0):
            return 2
        return 0
    if name == 'trlau_pc_nextgen_double_sided':
        flags = _generic_tpage_flags(material)
        return not bool(int(flags.get('single_sided', 0) or 0))
    if name == 'trlau_pc_nextgen_material_flags':
        layer_count = sum(1 for value in _generic_material_texture_ids(material)[:3] if int(value) >= 0)
        return _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS_SPECULAR if layer_count >= 3 else _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS
    if name == 'trlau_pc_nextgen_shader_indices':
        layer_count = sum(1 for value in _generic_material_texture_ids(material)[:3] if int(value) >= 0)
        values = _PC_NEXTGEN_DEFAULT_SHADER_INDICES_LAYERED if layer_count >= 3 else _PC_NEXTGEN_DEFAULT_SHADER_INDICES_DIFFUSE_ONLY
        return _csv_join_ints(values, 8)
    if name == 'trlau_pc_nextgen_local_num_pixmaps':
        return sum(1 for value in _generic_material_texture_ids(material) if int(value) >= 0)
    return default


def _material_table_index_for_slot(material, fallback_index: int, header) -> int:
    record_offset = _safe_int(_material_prop(material, 'trlau_pc_nextgen_material_record_offset', -1), -1)
    if record_offset >= int(header.material_offset):
        delta = int(record_offset) - int(header.material_offset)
        if delta % _PC_NEXTGEN_MATERIAL_STRIDE == 0:
            idx = delta // _PC_NEXTGEN_MATERIAL_STRIDE
            if idx >= 0:
                return int(idx)
    return int(fallback_index)


def _pc_nextgen_material_double_sided(material) -> bool:
    _pc_nextgen_maybe_sync_material(material)
    return bool(_material_prop(material, 'trlau_pc_nextgen_double_sided', False))


def _triangle_key(triangle: Sequence[int]) -> tuple[int, int, int]:
    return tuple(sorted(int(value) for value in triangle[:3]))


def _triangle_winding_parity(triangle: Sequence[int]) -> int:
    tri = [int(value) for value in triangle[:3]]
    if len(set(tri)) != 3:
        return 0
    order = sorted(tri)
    ranks = [order.index(value) for value in tri]
    inversions = 0
    for a in range(3):
        for b in range(a + 1, 3):
            if ranks[a] > ranks[b]:
                inversions += 1
    return inversions & 1


def _count_double_wound_pairs(triangles: Sequence[Sequence[int]]) -> int:
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for tri in triangles:
        if len(tri) < 3 or len(set(int(v) for v in tri[:3])) != 3:
            continue
        key = _triangle_key(tri)
        parity = _triangle_winding_parity(tri)
        buckets.setdefault(key, [0, 0])[parity] += 1
    return sum(min(counts[0], counts[1]) for counts in buckets.values())


def _select_double_sided_triangles(triangles: list[tuple[int, int, int]], needed: int) -> list[tuple[int, int, int]]:
    needed = max(0, int(needed))
    if needed <= 0:
        return []
    unique: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for tri in triangles:
        tri3 = tuple(int(v) for v in tri[:3])
        if len(set(tri3)) != 3:
            continue
        key = _triangle_key(tri3)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tri3)

    if len(unique) * 2 > needed:
        return list(triangles[:needed])

    expanded: list[tuple[int, int, int]] = []
    for a, b, c in unique:
        expanded.append((a, b, c))
        if len(expanded) >= needed:
            return expanded[:needed]
        expanded.append((a, c, b))
        if len(expanded) >= needed:
            return expanded[:needed]

    # If the selected template range is not perfectly pair-sized, preserve any
    # remaining source triangles after writing as many explicit pairs as possible.
    for tri in triangles:
        if len(expanded) >= needed:
            break
        tri3 = tuple(int(v) for v in tri[:3])
        expanded.append(tri3)
    return expanded[:needed]

def _pack_i32(blob: bytearray, offset: int, value: int) -> None:
    struct.pack_into('<i', blob, int(offset), int(value))


def _pack_u32(blob: bytearray, offset: int, value: int) -> None:
    struct.pack_into('<I', blob, int(offset), int(value) & 0xFFFFFFFF)


def _pack_u16(blob: bytearray, offset: int, value: int) -> None:
    struct.pack_into('<H', blob, int(offset), int(value) & 0xFFFF)


def _pack_u8(blob: bytearray, offset: int, value: int) -> None:
    struct.pack_into('<B', blob, int(offset), int(value) & 0xFF)


def _pack_s8(blob: bytearray, offset: int, value: int) -> None:
    value = max(-128, min(127, int(value)))
    struct.pack_into('<b', blob, int(offset), value)


def _pack_f32(blob: bytearray, offset: int, value: float) -> None:
    struct.pack_into('<f', blob, int(offset), float(value))


def _pc_nextgen_material_id_for_export(material, fallback_index: int = 0) -> int:
    try:
        drawgroup = int(get_material_draw_group(material))
    except Exception:
        drawgroup = int(fallback_index)
    return int(drawgroup) + 1


def _pc_nextgen_layer_num_textures_for_export(layer_is_enabled: bool, texture_id: int, texture_index: int) -> int:
    if not bool(layer_is_enabled):
        return 0
    if int(texture_id) < 0:
        return 0
    if not (0 <= int(texture_index) < 0xFFFF):
        return 0
    return 1


def _rgba_float_to_bgra_bytes(color: Sequence[float]) -> bytes:
    values = [0.0, 0.0, 0.0, 1.0]
    for index in range(min(4, len(color or []))):
        values[index] = max(0.0, min(1.0, _safe_float(color[index], values[index])))
    r, g, b, a = [int(round(component * 255.0)) & 0xFF for component in values]
    return bytes((b, g, r, a))


def _patch_materials(blob: bytearray, header, header_offsets, data_start: int, pixmaps: list[int], materials, mesh_obj, warning_cb=None) -> int:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        return 0
    try:
        material_slots = list(getattr(mesh, 'materials', []) or [])
    except Exception:
        material_slots = []

    # Keep the PCD9 pixmap table in sync when a layer keeps a valid textureIndex
    # but the user edits the texture ID in the PC Next Gen material panel.
    pixmap_table_base = None
    if header_offsets and len(header_offsets) > 1:
        for candidate in (int(data_start) + int(header_offsets[1]), int(header_offsets[1])):
            try:
                if 0 <= candidate <= len(blob) - 4:
                    count = struct.unpack_from('<i', blob, candidate)[0]
                    if count == len(pixmaps) and candidate + 4 + count * 4 <= len(blob):
                        pixmap_table_base = candidate
                        break
            except Exception:
                pass

    patched = 0
    for slot_index, material in enumerate(material_slots):
        if material is None:
            continue
        try:
            sync_pc_nextgen_material_from_panel(material)
        except Exception:
            pass
        material_index = _material_table_index_for_slot(material, slot_index, header)
        if material_index < 0 or material_index >= int(header.num_materials):
            continue
        offset = int(header.base) + int(header.material_offset) + material_index * _PC_NEXTGEN_MATERIAL_STRIDE
        if offset < 0 or offset + _PC_NEXTGEN_MATERIAL_STRIDE > len(blob):
            continue

        _pack_i32(blob, offset + 0x00, _pc_nextgen_material_id_for_export(material, material_index))
        struct.pack_into('<Q', blob, offset + 0x08, 0)
        _pack_u16(blob, offset + 0x10, 0)
        _pack_i32(blob, offset + 0x18, _safe_int(_material_prop(material, 'trlau_pc_nextgen_blend_mode', 0), 0))
        _pack_i32(blob, offset + 0x1C, _safe_int(_material_prop(material, 'trlau_pc_nextgen_combiner_type', 0), 0))
        _pack_u32(blob, offset + 0x20, _safe_int(_material_prop(material, 'trlau_pc_nextgen_material_flags', 0), 0))
        _pack_f32(blob, offset + 0x24, _safe_float(_material_prop(material, 'trlau_pc_nextgen_opacity', 1.0), 1.0))
        _pack_u32(blob, offset + 0x28, 0)
        _pack_u16(blob, offset + 0x2C, _safe_int(_material_prop(material, 'trlau_pc_nextgen_uv_auto_scroll_speed', 0), 0))
        _pack_f32(blob, offset + 0x30, _safe_float(_material_prop(material, 'trlau_pc_nextgen_sort_bias', 0.0), 0.0))
        _pack_f32(blob, offset + 0x34, _safe_float(_material_prop(material, 'trlau_pc_nextgen_detail_range_mul', 0.0), 0.0))
        _pack_f32(blob, offset + 0x38, _safe_float(_material_prop(material, 'trlau_pc_nextgen_detail_scale', 0.0), 0.0))
        _pack_f32(blob, offset + 0x3C, _safe_float(_material_prop(material, 'trlau_pc_nextgen_parallax_scale', 0.0), 0.0))
        _pack_f32(blob, offset + 0x40, _safe_float(_material_prop(material, 'trlau_pc_nextgen_parallax_offset', 0.0), 0.0))
        _pack_f32(blob, offset + 0x44, _safe_float(_material_prop(material, 'trlau_pc_nextgen_specular_power', 0.0), 0.0))
        _pack_f32(blob, offset + 0x48, _safe_float(_material_prop(material, 'trlau_pc_nextgen_specular_shift0', 0.0), 0.0))
        _pack_f32(blob, offset + 0x4C, _safe_float(_material_prop(material, 'trlau_pc_nextgen_specular_shift1', 0.0), 0.0))
        rim = _json_colors(_material_prop(material, 'trlau_pc_nextgen_rim_light_color', '[]'), 1, (0.0, 0.0, 0.0, 0.0))[0]
        struct.pack_into('<ffff', blob, offset + 0x50, *rim)
        _pack_f32(blob, offset + 0x60, _safe_float(_material_prop(material, 'trlau_pc_nextgen_rim_light_intensity', 0.0), 0.0))
        _pack_f32(blob, offset + 0x64, _safe_float(_material_prop(material, 'trlau_pc_nextgen_water_blend_bias', 0.0), 0.0))
        _pack_f32(blob, offset + 0x68, _safe_float(_material_prop(material, 'trlau_pc_nextgen_water_blend_exponent', 0.0), 0.0))
        water = _json_colors(_material_prop(material, 'trlau_pc_nextgen_water_deep_color', '[]'), 1, (0.0, 0.0, 0.0, 0.0))[0]
        struct.pack_into('<ffff', blob, offset + 0x6C, *water)
        local_pixmap_count = 0

        layer_texture_ids = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_texture_ids', ''), 8, -1)
        layer_texture_indices = _pc_nextgen_original_layer_texture_indices(material)
        layer_enabled = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_enabled', ''), 8, 0)
        layer_colors = _json_colors(_material_prop(material, 'trlau_pc_nextgen_layer_colors', '[]'), 8, (1.0, 1.0, 1.0, 1.0))
        layer_texcoord_sources = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_texcoord_sources', ''), 8, 0)
        layer_modifiers = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_modifiers', ''), 8, 0)
        layer_param_ids = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_param_ids', ''), 8, 0)
        layer_constants = _json_colors(_material_prop(material, 'trlau_pc_nextgen_layer_constants', '[]'), 8, (0.0, 0.0, 0.0, 0.0))

        for layer_index in range(8):
            layer_offset = offset + 0x80 + layer_index * 36
            blob[layer_offset:layer_offset + 4] = _rgba_float_to_bgra_bytes(layer_colors[layer_index])
            _pack_i32(blob, layer_offset + 0x04, layer_texcoord_sources[layer_index])
            _pack_i32(blob, layer_offset + 0x08, layer_modifiers[layer_index])
            _pack_i32(blob, layer_offset + 0x0C, layer_param_ids[layer_index])
            struct.pack_into('<ffff', blob, layer_offset + 0x10, *layer_constants[layer_index])
            texture_id = int(layer_texture_ids[layer_index])
            layer_is_enabled = int(layer_enabled[layer_index]) != 0
            if not layer_is_enabled:
                texture_index = 0xFFFF
            else:
                texture_index = int(layer_texture_indices[layer_index]) & 0xFFFF
            num_textures = _pc_nextgen_layer_num_textures_for_export(layer_is_enabled, texture_id, texture_index)
            if num_textures <= 0:
                texture_index = 0xFFFF
                layer_enabled[layer_index] = 0
            else:
                local_pixmap_count += 1
            _pack_u16(blob, layer_offset + 0x20, texture_index)
            _pack_u8(blob, layer_offset + 0x22, num_textures)
            _pack_s8(blob, layer_offset + 0x23, layer_enabled[layer_index])
            if num_textures > 0 and pixmap_table_base is not None and 0 <= texture_index < len(pixmaps):
                struct.pack_into('<i', blob, pixmap_table_base + 4 + texture_index * 4, texture_id)

        _pack_u8(blob, offset + 0x7C, local_pixmap_count)

        shader_indices = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_shader_indices', ''), 8, 0)
        for shader_index, value in enumerate(shader_indices[:8]):
            _pack_u32(blob, offset + 0x1A0 + shader_index * 4, value)
        patched += 1
    return patched


def _element_by_usage(elements: list[dict], usage: int, usage_index: int = 0) -> dict | None:
    for element in elements:
        if int(element.get('usage', -1)) == int(usage) and int(element.get('usage_index', 0)) == int(usage_index):
            return element
    for element in elements:
        if int(element.get('usage', -1)) == int(usage):
            return element
    return None


def _normal_to_tuple(value) -> tuple[float, float, float]:
    try:
        vec = Vector(value)
        if vec.length > 1e-9:
            vec.normalize()
        return (float(vec.x), float(vec.y), float(vec.z))
    except Exception:
        return (0.0, 0.0, 1.0)


def _uv_by_vertex(mesh) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    try:
        uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
    except Exception:
        uv_layer = None
    if uv_layer is None:
        return result
    for poly in getattr(mesh, 'polygons', []) or []:
        for loop_index in poly.loop_indices:
            try:
                loop = mesh.loops[loop_index]
                vi = int(loop.vertex_index)
                uv = uv_layer[loop_index].uv
                result.setdefault(vi, (float(uv.x), float(uv.y)))
            except Exception:
                continue
    return result


def _point_color_by_vertex(mesh) -> dict[int, tuple[float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float]] = {}
    attr = None
    try:
        attrs = getattr(mesh, 'color_attributes', None)
        attr = attrs.get('Color') if attrs is not None else None
        if attr is None:
            attr = getattr(attrs, 'active_color', None)
    except Exception:
        attr = None
    if attr is None:
        return result
    try:
        domain = str(getattr(attr, 'domain', '') or '').upper()
        data = getattr(attr, 'data', None)
        if data is None:
            return result
        if domain == 'POINT':
            for index, item in enumerate(data):
                color = getattr(item, 'color', (1.0, 1.0, 1.0, 1.0))
                result[int(index)] = tuple(float(color[i]) for i in range(4))
        else:
            for poly in getattr(mesh, 'polygons', []) or []:
                for loop_index in poly.loop_indices:
                    loop = mesh.loops[loop_index]
                    color = getattr(data[loop_index], 'color', (1.0, 1.0, 1.0, 1.0))
                    result.setdefault(int(loop.vertex_index), tuple(float(color[i]) for i in range(4)))
    except Exception:
        return {}
    return result


def _bone_weights_for_vertex(mesh_obj, vertex_index: int) -> list[tuple[int, float]]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None or vertex_index < 0 or vertex_index >= len(mesh.vertices):
        return []
    groups = getattr(mesh_obj, 'vertex_groups', None)
    if groups is None:
        return []
    result = []
    for group_ref in getattr(mesh.vertices[vertex_index], 'groups', []) or []:
        try:
            group = groups[int(group_ref.group)]
            name = str(getattr(group, 'name', '') or '')
            if not name.startswith('bone_'):
                continue
            bone_index = int(name.split('_', 1)[1])
            weight = float(getattr(group_ref, 'weight', 0.0) or 0.0)
            if weight > 0.0001:
                result.append((bone_index, weight))
        except Exception:
            continue
    result.sort(key=lambda item: item[1], reverse=True)
    total = sum(weight for _bone, weight in result[:2])
    if total > 1e-9:
        result = [(bone, weight / total) for bone, weight in result[:2]]
    return result[:2]


def _skin_local_index(batch, bone_index: int) -> int:
    skin_map = list(getattr(batch, 'skin_map', []) or [])
    if skin_map:
        try:
            return skin_map.index(int(bone_index))
        except ValueError:
            return max(0, min(len(skin_map) - 1, 0))
    return int(bone_index)


def _patch_vertices(blob: bytearray, header, batches, mesh_obj, arm_obj, warning_cb=None) -> int:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        return 0
    total_vertices = sum(max(0, int(batch.num_vertices)) for batch in batches)
    if len(mesh.vertices) != total_vertices:
        raise ValueError(
            f'PC Next Gen mesh "{getattr(mesh_obj, "name", "<mesh>")}" has {len(mesh.vertices)} vertices, but the template expects {total_vertices}. '
            'falling back to generic section rebuild.'
        )

    try:
        mesh.calc_normals()
    except Exception:
        pass
    uv_lookup = _uv_by_vertex(mesh)
    color_lookup = _point_color_by_vertex(mesh)
    arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else Matrix.Identity(4)
    normal_world_to_arm = arm_inv.to_3x3()

    patched = 0
    for batch in batches:
        vertex_base = int(header.base) + int(batch.vertex_data_offset)
        stride = int(batch.vertex_stride)
        if stride <= 0:
            continue
        position_elem = _element_by_usage(batch.vertex_elements, D3DDECLUSAGE_POSITION)
        normal_elem = _element_by_usage(batch.vertex_elements, D3DDECLUSAGE_NORMAL)
        uv_elem = _element_by_usage(batch.vertex_elements, D3DDECLUSAGE_TEXCOORD)
        color_elem = _element_by_usage(batch.vertex_elements, D3DDECLUSAGE_COLOR)
        weight_elem = _element_by_usage(batch.vertex_elements, D3DDECLUSAGE_BLENDWEIGHT)
        indices_elem = _element_by_usage(batch.vertex_elements, D3DDECLUSAGE_BLENDINDICES)
        for local_index in range(int(batch.num_vertices)):
            global_index = int(batch.vertex_start) + int(local_index)
            vertex = mesh.vertices[global_index]
            record = vertex_base + local_index * stride
            world = mesh_obj.matrix_world @ vertex.co
            local = arm_inv @ world
            if position_elem is not None and int(position_elem.get('type', -1)) == D3DDECLTYPE_FLOAT3:
                struct.pack_into('<fff', blob, record + int(position_elem.get('offset', 0)), float(local.x), float(local.y), float(local.z))
            if normal_elem is not None and int(normal_elem.get('type', -1)) == D3DDECLTYPE_FLOAT3:
                normal = normal_world_to_arm @ (mesh_obj.matrix_world.to_3x3() @ vertex.normal)
                nx, ny, nz = _normal_to_tuple(normal)
                struct.pack_into('<fff', blob, record + int(normal_elem.get('offset', 0)), nx, ny, nz)
            if uv_elem is not None and int(uv_elem.get('type', -1)) == D3DDECLTYPE_FLOAT2:
                uv = uv_lookup.get(global_index, (0.0, 0.0))
                struct.pack_into('<ff', blob, record + int(uv_elem.get('offset', 0)), float(uv[0]), 1.0 - float(uv[1]))
            if color_elem is not None and int(color_elem.get('type', -1)) == D3DDECLTYPE_D3DCOLOR:
                color = color_lookup.get(global_index)
                if color is not None:
                    color_offset = record + int(color_elem.get('offset', 0))
                    blob[color_offset:color_offset + 4] = _rgba_float_to_bgra_bytes(color)
            weights = _bone_weights_for_vertex(mesh_obj, global_index)
            if indices_elem is not None and int(indices_elem.get('type', -1)) == D3DDECLTYPE_SHORT2 and weights:
                bone0 = _skin_local_index(batch, weights[0][0])
                bone1 = _skin_local_index(batch, weights[1][0] if len(weights) > 1 else weights[0][0])
                struct.pack_into('<hh', blob, record + int(indices_elem.get('offset', 0)), int(bone0), int(bone1))
            if weight_elem is not None and int(weight_elem.get('type', -1)) == D3DDECLTYPE_FLOAT1 and weights:
                second_weight = float(weights[1][1]) if len(weights) > 1 else 0.0
                struct.pack_into('<f', blob, record + int(weight_elem.get('offset', 0)), max(0.0, min(1.0, second_weight)))
            patched += 1
    return patched


def _material_index_by_slot(mesh, header) -> dict[int, int]:
    result = {}
    for slot_index, material in enumerate(list(getattr(mesh, 'materials', []) or [])):
        result[int(slot_index)] = _material_table_index_for_slot(material, slot_index, header)
    return result


def _material_by_table_index(mesh, header) -> dict[int, object]:
    result = {}
    for slot_index, material in enumerate(list(getattr(mesh, 'materials', []) or [])):
        if material is None:
            continue
        result[_material_table_index_for_slot(material, slot_index, header)] = material
    return result


def _batch_for_triangle(indices: Sequence[int], batches) -> object | None:
    for batch in batches:
        start = int(batch.vertex_start)
        end = start + int(batch.num_vertices)
        if all(start <= int(index) < end for index in indices):
            return batch
    return None


def _encode_reversed_index(index: int) -> int:
    # Signed-negative spelling that the importer decodes as -index-1.
    value = -int(index) - 1
    if value < -32768:
        return int(index) | 0x8000
    return value


def _patch_indices(blob: bytearray, parser: TRNextGenModelParser, header, prim_groups, batches, index_data_offset: int, raw_indices: list[int], mesh_obj, warning_cb=None) -> int:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        return 0
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass
    material_by_slot = _material_index_by_slot(mesh, header)
    material_by_index = _material_by_table_index(mesh, header)
    triangles_by_key: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for tri in list(getattr(mesh, 'loop_triangles', []) or []):
        try:
            vertices = tuple(int(value) for value in tri.vertices)
            polygon = mesh.polygons[int(tri.polygon_index)]
            material_slot = int(getattr(polygon, 'material_index', 0))
            material_index = int(material_by_slot.get(material_slot, material_slot))
            batch = _batch_for_triangle(vertices, batches)
            if batch is None:
                continue
            local = tuple(int(value) - int(batch.vertex_start) for value in vertices)
            triangles_by_key[(int(batch.index), material_index)].append(local)
        except Exception:
            continue

    cursors: dict[tuple[int, int], int] = defaultdict(int)
    patched_faces = 0
    group_cursor = 0
    for batch in batches:
        for _local_group_index in range(int(batch.num_prim_groups)):
            if group_cursor >= len(prim_groups):
                break
            group = prim_groups[group_cursor]
            group_cursor += 1
            key = (int(batch.index), int(group.material_index))
            needed = int(group.num_primitives)
            start = cursors[key]
            available = triangles_by_key.get(key, [])
            source_window = available[start:start + needed]
            selected = list(source_window)
            material_obj = material_by_index.get(int(group.material_index))
            double_sided = _pc_nextgen_material_double_sided(material_obj)
            if double_sided:
                # Blender import may have normal-oriented the reverse half of a
                # double-wound pair, leaving two polygons with the same winding.
                # Rebuild the face-data pair explicitly instead of trusting the
                # mesh polygon winding verbatim.
                selected = _select_double_sided_triangles(available[start:], needed)
                if needed >= 2 and _count_double_wound_pairs(selected) <= 0:
                    _warn(
                        warning_cb,
                        f'PC Next Gen material {group.material_index} is marked Double Sided, but template group {batch.index}:{group.material_index} '
                        'does not have spare paired face slots; preserving original primitive count without growing the index buffer.'
                    )
            if len(selected) != needed:
                raise ValueError(
                    f'PC Next Gen mesh "{getattr(mesh_obj, "name", "<mesh>")}" has {len(selected)} triangle(s) for batch {batch.index}, material {group.material_index}; '
                    f'the template expects {needed}. falling back to generic section rebuild.'
                )
            cursors[key] += min(len(source_window), needed)
            raw_start = int(group.base_index)
            if raw_start < 0 or raw_start + needed * 3 > len(raw_indices):
                raise ValueError('PC Next Gen template index range is outside the original index buffer')
            for face_index, tri_local in enumerate(selected):
                original_raw = raw_indices[raw_start + face_index * 3:raw_start + face_index * 3 + 3]
                _decoded, was_reversed = parser._decode_nextgen_triangle(original_raw, batch)
                a, b, c = (int(tri_local[0]), int(tri_local[1]), int(tri_local[2]))
                if max(a, b, c) >= int(batch.num_vertices) or min(a, b, c) < 0:
                    raise ValueError('PC Next Gen triangle references a vertex outside its original batch')
                if was_reversed:
                    to_write = (_encode_reversed_index(a), c, b)
                else:
                    to_write = (a, b, c)
                out = int(index_data_offset) + (raw_start + face_index * 3) * 2
                for component_index, value in enumerate(to_write):
                    if int(value) < 0:
                        struct.pack_into('<h', blob, out + component_index * 2, int(value))
                    else:
                        struct.pack_into('<H', blob, out + component_index * 2, int(value) & 0xFFFF)
                patched_faces += 1
    return patched_faces


def _copy_template_to_directory(template_path: Path, directory: Path, render_id: int, section_list=None) -> Path:
    directory = Path(directory)
    render_id = int(render_id) & 0xFFFFFFFF
    try:
        existing_indices = []
        for candidate in directory.iterdir():
            if not candidate.is_file():
                continue
            try:
                existing_indices.append(_section_index_from_filename(candidate))
            except Exception:
                pass
        index = (max(existing_indices) + 1) if existing_indices else 0
    except Exception:
        index = 0
    filename = f'{index}_{render_id:x}.gnc'
    if section_list is not None:
        try:
            index = int(section_list.allocate(filename))
            filename = f'{index}_{render_id:x}.gnc'
        except Exception:
            pass
    out_path = directory / filename
    blob = bytearray(Path(template_path).read_bytes())
    if len(blob) >= 20:
        struct.pack_into('<I', blob, 16, render_id)
    out_path.write_bytes(bytes(blob))
    return out_path


def _source_template_from_mesh(mesh_obj, render_id: int) -> Path | None:
    for target in (mesh_obj, getattr(mesh_obj, 'data', None)):
        if target is None:
            continue
        try:
            value = str(getattr(target, 'trlau_pc_nextgen_source_section_file', target.get('trlau_pc_nextgen_source_section_file', '')) or '').strip()
        except Exception:
            value = ''
        if not value:
            continue
        path = Path(value)
        if path.exists():
            try:
                info = _standalone_section_info(path)
                if int(getattr(info, 'section_type', -1)) == 7:
                    return path
            except Exception:
                return path
    return None




def _align(value: int, alignment: int = 16) -> int:
    alignment = max(1, int(alignment))
    return (int(value) + alignment - 1) & ~(alignment - 1)


def _section_index_for_output(path: Path, fallback: int = 0) -> int:
    try:
        return _section_index_from_filename(Path(path))
    except Exception:
        return int(fallback)



def _read_counted_i32_table_from_payload(data: bytes | bytearray, offset: int, *, max_count: int = 4096) -> list[int]:
    try:
        offset = int(offset)
        if offset < 0 or offset + 4 > len(data):
            return []
        count = struct.unpack_from('<i', data, offset)[0]
        if count < 0 or count > int(max_count):
            return []
        end = offset + 4 + int(count) * 4
        if end > len(data):
            return []
        return [int(struct.unpack_from('<i', data, offset + 4 + index * 4)[0]) for index in range(int(count))]
    except Exception:
        return []


def _pc_nextgen_shader_table_from_template(template_path: Path | None) -> list[int]:
    if template_path is None:
        return []
    path = Path(template_path)
    if not path.exists():
        return []
    cache = SectionContextCache(str(path), endian='<')
    try:
        context = cache.get_root_context()
        blob = Path(context.filepath).read_bytes()
        data_start = int(context.data_start)
        if data_start < 0 or data_start + 12 > len(blob):
            return []
        offset_shaders = struct.unpack_from('<I', blob, data_start + 0x08)[0]
        table = _read_counted_i32_table_from_payload(blob[data_start:int(context.data_end)], int(offset_shaders))
        if table:
            return table
        return _read_counted_i32_table_from_payload(blob, int(offset_shaders))
    except Exception:
        return []
    finally:
        cache.close()


def _pc_nextgen_shader_table_from_materials(materials: Sequence[object | None]) -> list[int]:
    for material in materials:
        value = _material_prop(material, 'trlau_pc_nextgen_shader_table_ids', '')
        table = _csv_ints(value, 0, 0) if value else []
        if isinstance(value, str) and value.strip():
            parsed = []
            for part in value.split(','):
                part = part.strip()
                if not part:
                    continue
                parsed.append(_safe_int(part, 0))
            if parsed:
                return parsed
        elif isinstance(value, (list, tuple)) and value:
            try:
                return [int(item) for item in value]
            except Exception:
                pass
    return []


def _max_shader_index_used_by_materials(materials: Sequence[object | None]) -> int:
    max_index = -1
    for material in materials:
        values = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_shader_indices', ''), 8, 0)
        for value in values:
            value = int(value)
            if 0 <= value < 4096:
                max_index = max(max_index, value)
    return max_index


def _pc_nextgen_shader_table_for_export(materials: Sequence[object | None], reference_template_path: Path | None = None) -> list[int]:
    table = _pc_nextgen_shader_table_from_template(reference_template_path)
    if not table:
        table = _pc_nextgen_shader_table_from_materials(materials)
    if not table:
        table = list(_PC_NEXTGEN_DEFAULT_SHADER_SECTION_IDS)

    required_max = _max_shader_index_used_by_materials(materials)
    if required_max >= len(table):
        for shader_id in _PC_NEXTGEN_DEFAULT_SHADER_SECTION_IDS:
            if len(table) > required_max:
                break
            if int(shader_id) not in table:
                table.append(int(shader_id))
        while len(table) <= required_max:
            table.append(int(table[-1] if table else _PC_NEXTGEN_DEFAULT_SHADER_SECTION_IDS[0]))
    return [int(value) for value in table]


def _pc_nextgen_relocations_for_payload(data: bytes | bytearray, self_index: int) -> dict[int, tuple[int, int, int]]:
    relocs_by_offset: dict[int, tuple[int, int, int]] = {
        0: (int(self_index), 0, 0),
        4: (int(self_index), 0, 0),
        8: (int(self_index), 0, 0),
    }
    try:
        offset_pixmaps = struct.unpack_from('<I', data, 0x04)[0]
        pixmaps = _read_counted_i32_table_from_payload(data, int(offset_pixmaps))
        for index, _texture_id in enumerate(pixmaps):
            relocs_by_offset[int(offset_pixmaps) + 4 + index * 4] = (5, 0, 2)
    except Exception:
        pass
    try:
        offset_shaders = struct.unpack_from('<I', data, 0x08)[0]
        shaders = _read_counted_i32_table_from_payload(data, int(offset_shaders))
        for index, _shader_id in enumerate(shaders):
            relocs_by_offset[int(offset_shaders) + 4 + index * 4] = (9, 0, 1)
    except Exception:
        pass
    return relocs_by_offset


def _write_pc_nextgen_standalone_section(path: Path, data: bytes, render_id: int, *, section_index: int | None = None, template_path: Path | None = None) -> None:
    section_type = 7
    skip_flags = 0
    version_id = 0
    has_debug_info = 0
    resource_type = 0
    spec_mask = 0x7FFFFFFF
    if template_path is not None and Path(template_path).exists():
        try:
            info = _standalone_section_info(Path(template_path))
            section_type = int(getattr(info, 'section_type', section_type))
            skip_flags = int(getattr(info, 'skip_flags', skip_flags))
            version_id = int(getattr(info, 'version_id', version_id))
            has_debug_info = int(getattr(info, 'has_debug_info', has_debug_info))
            resource_type = int(getattr(info, 'resource_type', resource_type))
            spec_mask = int(getattr(info, 'spec_mask', spec_mask))
        except Exception:
            pass

    self_index = int(section_index if section_index is not None else _section_index_for_output(path, int(render_id) & 0x1FFF))
    relocs_by_offset = _pc_nextgen_relocations_for_payload(data, self_index)
    packed_data = ((int(has_debug_info) & 0x1) | ((int(resource_type) & 0x7F) << 1) | ((len(relocs_by_offset) & 0x00FFFFFF) << 8))
    payload = bytearray()
    payload.extend(b'SECT')
    payload.extend(struct.pack('<iBBHIII', int(len(data)), int(section_type) & 0xFF, int(skip_flags) & 0xFF, int(version_id) & 0xFFFF, int(packed_data) & 0xFFFFFFFF, int(render_id) & 0xFFFFFFFF, int(spec_mask) & 0xFFFFFFFF))
    for offset in sorted(relocs_by_offset):
        target, type_specific, reloc_type = relocs_by_offset[offset]
        type_and_section = ((int(target) & 0x1FFF) << 3) | (int(reloc_type) & 0x7)
        payload.extend(struct.pack('<HhI', type_and_section & 0xFFFF, int(type_specific), int(offset) & 0xFFFFFFFF))
    payload.extend(data)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(bytes(payload))


def _allocate_nextgen_output_path(directory: Path, render_id: int, section_list=None) -> Path:
    directory = Path(directory)
    render_id = int(render_id) & 0xFFFFFFFF
    existing_indices: set[int] = set()
    try:
        for candidate in directory.iterdir():
            if not candidate.is_file():
                continue
            try:
                existing_indices.add(_section_index_from_filename(candidate))
            except Exception:
                continue
    except Exception:
        pass
    if section_list is not None:
        try:
            existing_indices.update(int(value) for value in section_list.used_indices())
        except Exception:
            pass
    index = (max(existing_indices) + 1) if existing_indices else 0
    filename = f'{index}_{render_id:x}.gnc'
    if section_list is not None:
        try:
            section_list.ensure_index(index, filename)
        except Exception:
            pass
    return directory / filename


def _material_slot_records(mesh) -> list[object | None]:
    try:
        materials = list(getattr(mesh, 'materials', []) or [])
    except Exception:
        materials = []
    return materials or [None]


def _material_table_index_for_polygon(mesh, polygon) -> int:
    try:
        slot = int(getattr(polygon, 'material_index', 0) or 0)
    except Exception:
        slot = 0
    material_count = max(1, len(_material_slot_records(mesh)))
    return max(0, min(material_count - 1, slot))


def _pc_nextgen_material_drawgroup(material, fallback: int) -> int:
    try:
        return int(get_material_draw_group(material))
    except Exception:
        pass
    material_id = _safe_int(_material_prop(material, 'trlau_pc_nextgen_material_id', -1), -1)
    if material_id > 0:
        return int(material_id) - 1
    return int(fallback)



def _csv_join_ints(values: Sequence[int], count: int = 8) -> str:
    result = []
    for index in range(int(count)):
        try:
            result.append(str(int(values[index])))
        except Exception:
            result.append('-1')
    return ','.join(result)


def _parse_texture_id_from_image_text(text: str) -> int | None:
    text = str(text or '').strip()
    if not text:
        return None
    try:
        name = Path(text).name
    except Exception:
        name = text
    candidates = [name, text]
    for candidate in candidates:
        candidate = str(candidate or '')
        # ?
        match = re.search(r'(?i)(?:^|[^0-9a-f])0x([0-9a-f]+)(?:[^0-9a-f]|$)', candidate)
        if match:
            return int(match.group(1), 16)
        # ?
        match = re.match(r'(?i)^\d+_([0-9a-f]+)(?:\.[^.]+)?$', Path(candidate).name)
        if match:
            return int(match.group(1), 16)
        stem = Path(candidate).stem
        match = re.search(r'(?i)(?:^|[^a-z0-9])(?:tex|texture|pcd|diffuse|albedo|base|basecolor|normal|spec|specular|gloss|layer)[_\s-]*(?:id[_\s-]*)?([0-9a-f]+)$', stem)
        if match:
            return int(match.group(1), 16)
    return None

def _texture_id_from_image(image) -> int | None:
    if image is None:
        return None
    for key in ('trlau_texture_id', 'texture_id', 'trlau_section_id'):
        try:
            if key in image:
                value = _safe_int(image.get(key), -1)
                if value >= 0:
                    return int(value) & 0xFFFFFFFF
        except Exception:
            pass
    # Name/path parsing is deliberately strict. If no explicit texture-section
    # ID is present, fall back to the PCMaterialData layer ID instead of guessing
    # from a material/slot number in the image name.
    for attr in ('filepath', 'filepath_raw', 'name'):
        try:
            value = getattr(image, attr, '')
        except Exception:
            value = ''
        parsed = _parse_texture_id_from_image_text(value)
        if parsed is not None and parsed >= 0:
            return int(parsed) & 0xFFFFFFFF
    return None

def _pc_nextgen_role_to_layer(value: str) -> int | None:
    text = str(value or '').strip().lower()
    if text in {'diffuse', 'base', 'basecolor', 'base_color', 'albedo', 'color'}:
        return 0
    if text in {'normal', 'normalmap', 'normal_map', 'bump'}:
        return 1
    if text in {'specular', 'spec', 'gloss', 'glossmap', 'specular_gloss'}:
        return 2
    match = re.match(r'^layer[_\s-]*(\d+)$', text)
    if match:
        index = int(match.group(1))
        if 0 <= index < 8:
            return index
    return None


def _pc_nextgen_layer_images_from_nodes(material) -> list[object | None]:
    images: list[object | None] = [None] * 8
    if material is None:
        return images
    try:
        node_tree = getattr(material, 'node_tree', None)
        nodes = list(getattr(node_tree, 'nodes', []) or []) if node_tree is not None else []
    except Exception:
        nodes = []
    for node in nodes:
        try:
            if getattr(node, 'bl_idname', '') != 'ShaderNodeTexImage':
                continue
            image = getattr(node, 'image', None)
            if image is None:
                continue
        except Exception:
            continue
        layer_index = None
        for source in (node, image):
            try:
                if 'trlau_pc_nextgen_layer_index' in source:
                    candidate = _safe_int(source.get('trlau_pc_nextgen_layer_index'), -1)
                    if 0 <= candidate < 8:
                        layer_index = int(candidate)
                        break
            except Exception:
                pass
        if layer_index is None:
            for key in ('trlau_pc_nextgen_role', 'trlau_pc_nextgen_texture_role'):
                for source in (node, image):
                    try:
                        if key in source:
                            layer_index = _pc_nextgen_role_to_layer(str(source.get(key)))
                            if layer_index is not None:
                                break
                    except Exception:
                        pass
                if layer_index is not None:
                    break
        if layer_index is None:
            # Last-resort role inference from image or node text.  This keeps
            # manually assigned Base/Normal/Specular image nodes exportable even
            # if the custom metadata was lost.
            texts = []
            for source in (node, image):
                for attr in ('name', 'label'):
                    try:
                        texts.append(str(getattr(source, attr, '') or ''))
                    except Exception:
                        pass
            joined = ' '.join(texts).lower()
            if 'normal' in joined:
                layer_index = 1
            elif 'spec' in joined or 'gloss' in joined:
                layer_index = 2
            elif 'diffuse' in joined or 'base' in joined or 'albedo' in joined:
                layer_index = 0
        if layer_index is None or not (0 <= int(layer_index) < 8):
            continue
        current = images[int(layer_index)]
        # Prefer the first image with an explicit/parseable texture ID; otherwise
        # keep the first node for that layer.
        if current is None or (_texture_id_from_image(current) is None and _texture_id_from_image(image) is not None):
            images[int(layer_index)] = image
    return images


def _resolved_pc_nextgen_layer_texture_ids(material) -> list[int]:
    """Resolve exported texture IDs from bound image nodes plus panel metadata.

    The PC Next Gen panel remains the fallback/source for non-node workflows, but
    an actually bound Image Texture node with a valid `trlau_texture_id` or a
    parseable texture section ID now wins for its layer.  This means Base Color,
    Normal Map, and Specular node swaps export without requiring manual texture
    ID edits in the panel.
    """
    _pc_nextgen_maybe_sync_material(material)
    ids = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_texture_ids', ''), 8, -1)
    images = _pc_nextgen_layer_images_from_nodes(material)
    changed = False
    for layer_index, image in enumerate(images):
        resolved = _texture_id_from_image(image)
        if resolved is None:
            continue
        resolved = int(resolved) & 0xFFFFFFFF
        if int(ids[layer_index]) != resolved:
            ids[layer_index] = resolved
            changed = True
    if changed and material is not None:
        try:
            for layer_index, value in enumerate(ids[:8]):
                set_material_panel_value(material, f'trlau_ui_pcng_layer{layer_index}_texture_id', int(value))
            diffuse_id = ids[0] if len(ids) > 0 and int(ids[0]) >= 0 else -1
            normal_id = ids[1] if len(ids) > 1 and int(ids[1]) >= 0 else -1
            specular_id = ids[2] if len(ids) > 2 and int(ids[2]) >= 0 else -1
            set_material_panel_value(material, 'trlau_ui_pcng_diffuse_texture_id', int(diffuse_id))
            set_material_panel_value(material, 'trlau_ui_pcng_normal_texture_id', int(normal_id))
            set_material_panel_value(material, 'trlau_ui_pcng_specular_texture_id', int(specular_id))
        except Exception:
            pass
    return [int(value) for value in ids[:8]]

def _pc_nextgen_template_pixmaps(template_path: Path | None) -> list[int]:
    if template_path is None:
        return []
    path = Path(template_path)
    if not path.exists():
        return []
    try:
        parser = TRNextGenModelParser(str(path), cdc_render_data_id=0)
        parser.parse()
        header_offsets = parser._read_nextgen_header_offsets()
        return [int(value) for value in parser._parse_pixmaps(header_offsets[1])]
    except Exception:
        logger.debug('Could not read PC Next Gen template pixmap table from %s', path, exc_info=True)
        return []


def _pc_nextgen_original_layer_texture_indices(material) -> list[int]:
    """Return the imported per-layer textureIndex values when available."""
    template = _pc_nextgen_material_template_record_bytes(material)
    result = [0xFFFF] * 8
    if template is not None and len(template) >= _PC_NEXTGEN_MATERIAL_STRIDE:
        for layer_index in range(8):
            layer_offset = 0x80 + layer_index * 36
            try:
                result[layer_index] = int(struct.unpack_from('<H', template, layer_offset + 0x20)[0])
            except Exception:
                result[layer_index] = 0xFFFF
        return result
    try:
        return _csv_ints(material.get('trlau_pc_nextgen_layer_texture_indices', ''), 8, 0xFFFF)
    except Exception:
        return [0xFFFF] * 8


def _pc_nextgen_build_pixmap_layout(materials: Sequence[object | None], template_path: Path | None = None) -> tuple[list[int], list[list[int]]]:
    template_pixmaps = _pc_nextgen_template_pixmaps(template_path)
    pixmaps = list(template_pixmaps)
    layer_indices_by_material: list[list[int]] = []
    assigned_indices: dict[int, int] = {}
    used_indices: set[int] = set()

    for material in materials:
        ids = _resolved_pc_nextgen_layer_texture_ids(material)
        enabled = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_enabled', ''), 8, 0)
        original_indices = _pc_nextgen_original_layer_texture_indices(material)
        material_indices = [0xFFFF] * 8
        for layer_index in range(8):
            texture_id = int(ids[layer_index]) if layer_index < len(ids) else -1
            if int(enabled[layer_index]) == 0 or texture_id < 0:
                continue

            chosen_index = None
            original_index = int(original_indices[layer_index]) if layer_index < len(original_indices) else 0xFFFF
            if 0 <= original_index < len(pixmaps):
                already_assigned = assigned_indices.get(original_index)
                if already_assigned is None or int(already_assigned) == texture_id:
                    chosen_index = original_index
                    pixmaps[chosen_index] = texture_id
                    assigned_indices[chosen_index] = texture_id

            if chosen_index is None:
                chosen_index = len(pixmaps)
                pixmaps.append(texture_id)
                assigned_indices[chosen_index] = texture_id

            chosen_index = int(chosen_index)
            material_indices[layer_index] = chosen_index
            used_indices.add(chosen_index)
        layer_indices_by_material.append(material_indices)

    if pixmaps:
        keep_count = len(template_pixmaps)
        if used_indices:
            keep_count = max(keep_count, max(used_indices) + 1)
        if keep_count < len(pixmaps):
            pixmaps = pixmaps[:keep_count]

    return pixmaps, layer_indices_by_material


def _mesh_uv_lookup_by_loop(mesh) -> dict[int, tuple[float, float]]:
    result: dict[int, tuple[float, float]] = {}
    try:
        layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
    except Exception:
        layer = None
    if layer is None:
        return result
    for index, item in enumerate(layer):
        try:
            uv = item.uv
            result[int(index)] = (float(uv.x), float(uv.y))
        except Exception:
            continue
    return result


def _mesh_color_lookup_by_loop(mesh) -> dict[int, tuple[float, float, float, float]]:
    result: dict[int, tuple[float, float, float, float]] = {}
    attr = None
    try:
        attrs = getattr(mesh, 'color_attributes', None)
        attr = attrs.get('Color') if attrs is not None else None
        if attr is None:
            attr = getattr(attrs, 'active_color', None)
    except Exception:
        attr = None
    if attr is None:
        return result
    try:
        data = getattr(attr, 'data', None)
        if data is None:
            return result
        domain = str(getattr(attr, 'domain', '') or '').upper()
        if domain == 'POINT':
            for poly in getattr(mesh, 'polygons', []) or []:
                for loop_index in poly.loop_indices:
                    vi = int(mesh.loops[loop_index].vertex_index)
                    color = getattr(data[vi], 'color', (1.0, 1.0, 1.0, 1.0))
                    result[int(loop_index)] = tuple(float(color[i]) for i in range(4))
        else:
            for loop_index, item in enumerate(data):
                color = getattr(item, 'color', (1.0, 1.0, 1.0, 1.0))
                result[int(loop_index)] = tuple(float(color[i]) for i in range(4))
    except Exception:
        return {}
    return result


def _tri_loop_indices(mesh, loop_triangle) -> tuple[int, int, int]:
    try:
        return tuple(int(v) for v in loop_triangle.loops[:3])
    except Exception:
        try:
            polygon = mesh.polygons[int(loop_triangle.polygon_index)]
            return tuple(int(v) for v in list(polygon.loop_indices)[:3])
        except Exception:
            return (0, 0, 0)


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


def _bone_weights_for_vertex_any(mesh_obj, vertex_index: int) -> list[tuple[int, float]]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None or vertex_index < 0 or vertex_index >= len(mesh.vertices):
        return []
    groups = getattr(mesh_obj, 'vertex_groups', None)
    if groups is None:
        return []
    result: list[tuple[int, float]] = []
    for group_ref in getattr(mesh.vertices[vertex_index], 'groups', []) or []:
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
    result.sort(key=lambda item: item[1], reverse=True)
    result = result[:2]
    total = sum(weight for _bone, weight in result)
    if total > 1e-9:
        result = [(bone, weight / total) for bone, weight in result]
    return result


def _normal_for_vertex(mesh, vertex_index: int):
    try:
        return mesh.vertices[int(vertex_index)].normal
    except Exception:
        return Vector((0.0, 0.0, 1.0))


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


def _pack_pc_nextgen_vertex_record(position: Vector, normal, uv: tuple[float, float], color: tuple[float, float, float, float], weights: list[tuple[int, float]], skin_map: list[int], *, include_tangent_basis: bool = True) -> bytes:
    stride = 68 if include_tangent_basis else 44
    data = bytearray(stride)
    struct.pack_into('<fff', data, 0, float(position.x), float(position.y), float(position.z))
    data[12:16] = _rgba_float_to_bgra_bytes(color)
    struct.pack_into('<ff', data, 16, float(uv[0]), 1.0 - float(uv[1]))
    nx, ny, nz = _normal_to_tuple(normal)
    struct.pack_into('<fff', data, 24, nx, ny, nz)
    if weights:
        bone0 = int(weights[0][0])
        bone1 = int(weights[1][0]) if len(weights) > 1 else bone0
        try:
            local0 = skin_map.index(bone0)
        except ValueError:
            local0 = 0
        try:
            local1 = skin_map.index(bone1)
        except ValueError:
            local1 = local0
        second_weight = float(weights[1][1]) if len(weights) > 1 else 0.0
    else:
        local0 = 0
        local1 = 0
        second_weight = 0.0
    struct.pack_into('<f', data, 36, max(0.0, min(1.0, second_weight)))
    struct.pack_into('<hh', data, 40, int(local0), int(local1))
    if include_tangent_basis:
        # Conservative tangent/binormal basis.  Tangents are currently not
        # exposed in the Blender import path, so write a stable orthogonal
        # fallback rather than carrying arbitrary stale template bytes.
        struct.pack_into('<fff', data, 44, 1.0, 0.0, 0.0)
        struct.pack_into('<fff', data, 56, 0.0, 1.0, 0.0)
    return bytes(data)


def _pc_nextgen_vertex_elements_blob(*, include_tangent_basis: bool = True) -> bytes:
    entries = [
        (0, 0, D3DDECLTYPE_FLOAT3, 0, D3DDECLUSAGE_POSITION, 0),
        (0, 12, D3DDECLTYPE_D3DCOLOR, 0, D3DDECLUSAGE_COLOR, 0),
        (0, 16, D3DDECLTYPE_FLOAT2, 0, D3DDECLUSAGE_TEXCOORD, 0),
        (0, 24, D3DDECLTYPE_FLOAT3, 0, D3DDECLUSAGE_NORMAL, 0),
        (0, 36, D3DDECLTYPE_FLOAT1, 0, D3DDECLUSAGE_BLENDWEIGHT, 0),
        (0, 40, D3DDECLTYPE_SHORT2, 0, D3DDECLUSAGE_BLENDINDICES, 0),
    ]
    if include_tangent_basis:
        entries.extend([
            (0, 44, D3DDECLTYPE_FLOAT3, 0, 6, 0),
            (0, 56, D3DDECLTYPE_FLOAT3, 0, 7, 0),
        ])
    entries.append((0xFF, 0, D3DDECLTYPE_UNUSED, 0, 0, 0))
    blob = bytearray()
    for stream, offset, elem_type, method, usage, usage_index in entries:
        blob.extend(struct.pack('<HHBBBB', int(stream), int(offset), int(elem_type), int(method), int(usage), int(usage_index)))
    if len(blob) < 16 * 8:
        blob.extend(b'\x00' * ((16 * 8) - len(blob)))
    return bytes(blob[:16 * 8])


def _pc_nextgen_material_uses_compact_vertex_layout(material) -> bool:
    """Return true for materials that use the 44-byte alpha vertex stream.

    Vanilla TR7 PC next-gen alpha/diffuse-only batches omit tangent and binormal
    elements and use vertexFormat 0x003F, stride 44.  The supplied sample's
    alpha material is one of those.  Rebuilding it with the normal-mapped
    0xC03F/68-byte declaration makes the batch incompatible with the preserved
    alpha shader mapping.
    """
    try:
        return int(_material_prop(material, 'trlau_pc_nextgen_blend_mode', 0)) != 0 and _pc_nextgen_export_layer_count(material) <= 1
    except Exception:
        return False



def _pc_nextgen_enabled_layer_count(material) -> int:
    _pc_nextgen_maybe_sync_material(material)
    ids = _resolved_pc_nextgen_layer_texture_ids(material)
    enabled = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_enabled', ''), 8, 0)
    count = 0
    for index in range(min(8, len(ids), len(enabled))):
        if int(enabled[index]) != 0 and int(ids[index]) >= 0:
            count += 1
    return int(count)


def _pc_nextgen_export_layer_count(material) -> int:
    return _pc_nextgen_enabled_layer_count(material)


def _pc_nextgen_safe_material_flags(material) -> int:
    raw = _safe_int(_material_prop(material, 'trlau_pc_nextgen_material_flags', 0), 0) & 0xFFFFFFFF
    layer_count = _pc_nextgen_export_layer_count(material)
    # Vanilla TR7 PC PCD9 diffuse-only alpha materials use the plain diffuse
    # flag set (0x00221A00).  If a material was converted or duplicated from a
    # layered/specular material and later reduced to one active layer, keeping
    # flags such as 0x00235A00 makes the game bind a shader state that ignores
    # the diffuse texture alpha; transparent white RGB then renders visibly
    # in-game.  Normalize one-layer exports to the observed diffuse-only state.
    if layer_count <= 1:
        return _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS
    if raw != 0:
        return int(raw)
    # Zero appears in newly converted placeholder materials, not in the vanilla
    # TR7 PC Next-Gen material set we have.  Use a conservative lit material
    # configuration instead of exporting a zeroed shader-control word.
    return _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS_SPECULAR if layer_count >= 3 else _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS


def _pc_nextgen_safe_shader_indices(material, shader_table_count: int | None = None) -> list[int]:
    values = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_shader_indices', ''), 8, 0)
    layer_count = _pc_nextgen_export_layer_count(material)

    # Imported PCD9 materials already contain the shader-program mapping used by
    # the game.  Preserve that mapping on a rebuilt export instead of replacing
    # every one-layer material with the generic diffuse-only fallback.  The
    # alpha/eyelash material in the supplied sample uses shader indices
    # 10..15; changing those to 7..12 pairs the material with the wrong vertex
    # declaration and can crash during render-data setup.
    has_imported_record = _pc_nextgen_material_template_record_bytes(material) is not None
    has_explicit_mapping = any(int(value) != 0 for value in values[:6])
    if not has_imported_record and not has_explicit_mapping:
        # Converted/new materials often have all zeros, which is not a useful
        # shader mapping. Pick the observed vanilla fallback based on enabled
        # layer count.
        values = list(_PC_NEXTGEN_DEFAULT_SHADER_INDICES_LAYERED if layer_count >= 3 else _PC_NEXTGEN_DEFAULT_SHADER_INDICES_DIFFUSE_ONLY)

    table_count = int(shader_table_count or 0)
    if table_count > 0:
        values = [int(value) if 0 <= int(value) < table_count else 0 for value in values[:8]]
    while len(values) < 8:
        values.append(0)
    return [int(value) for value in values[:8]]



def _pc_nextgen_material_template_record_bytes(material) -> bytes | None:
    """Return the imported 456-byte PCMaterialData record when available.

    Starting from the original material record preserves currently unknown
    per-material bytes.  This is especially important for alpha-blended
    vanilla materials: the documented fields can look sufficient, but the game
    appears to depend on bytes we had been zeroing during generic rebuilds.
    """
    value = _material_prop(material, 'trlau_pc_nextgen_material_record_hex', '')
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= _PC_NEXTGEN_MATERIAL_STRIDE * 2:
            try:
                data = bytes.fromhex(text[:_PC_NEXTGEN_MATERIAL_STRIDE * 2])
                if len(data) == _PC_NEXTGEN_MATERIAL_STRIDE:
                    return data
            except Exception:
                pass
    return None


def _pc_nextgen_asset_padding_bytes(material) -> bytes:
    value = _material_prop(material, 'trlau_pc_nextgen_asset_id_padding_hex', '')
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 12:
            try:
                data = bytes.fromhex(text[:12])
                if len(data) == 6:
                    return data
            except Exception:
                pass
    template = _pc_nextgen_material_template_record_bytes(material)
    if template is not None and len(template) >= 0x18:
        return bytes(template[0x12:0x18])
    return bytes.fromhex('000073bc0510')

def _pack_pc_nextgen_material_record(material, material_index: int, layer_pixmap_indices: Sequence[int] | None = None, shader_table_count: int | None = None) -> bytes:
    _pc_nextgen_maybe_sync_material(material)
    template_record = _pc_nextgen_material_template_record_bytes(material)
    blob = bytearray(template_record if template_record is not None else (b'\x00' * _PC_NEXTGEN_MATERIAL_STRIDE))
    export_layer_count = _pc_nextgen_export_layer_count(material)
    drawgroup = _pc_nextgen_material_drawgroup(material, material_index)
    material_id = _pc_nextgen_material_id_for_export(material, material_index)
    _pack_i32(blob, 0x00, material_id)
    struct.pack_into('<Q', blob, 0x08, 0)
    _pack_u16(blob, 0x10, 0)
    blob[0x12:0x18] = _pc_nextgen_asset_padding_bytes(material)
    _pack_i32(blob, 0x18, _safe_int(_material_prop(material, 'trlau_pc_nextgen_blend_mode', 0), 0))
    _pack_i32(blob, 0x1C, _safe_int(_material_prop(material, 'trlau_pc_nextgen_combiner_type', 0), 0))
    _pack_u32(blob, 0x20, _pc_nextgen_safe_material_flags(material))
    _pack_f32(blob, 0x24, _safe_float(_material_prop(material, 'trlau_pc_nextgen_opacity', 1.0), 1.0))
    _pack_u32(blob, 0x28, 0)
    _pack_u16(blob, 0x2C, _safe_int(_material_prop(material, 'trlau_pc_nextgen_uv_auto_scroll_speed', 0), 0))
    _pack_f32(blob, 0x30, _safe_float(_material_prop(material, 'trlau_pc_nextgen_sort_bias', 0.0), 0.0))
    _pack_f32(blob, 0x34, _safe_float(_material_prop(material, 'trlau_pc_nextgen_detail_range_mul', 0.0), 0.0))
    _pack_f32(blob, 0x38, _safe_float(_material_prop(material, 'trlau_pc_nextgen_detail_scale', 0.0), 0.0))
    _pack_f32(blob, 0x3C, _safe_float(_material_prop(material, 'trlau_pc_nextgen_parallax_scale', 0.0), 0.0))
    _pack_f32(blob, 0x40, _safe_float(_material_prop(material, 'trlau_pc_nextgen_parallax_offset', 0.0), 0.0))
    specular_power = 0.0 if export_layer_count <= 1 else _safe_float(_material_prop(material, 'trlau_pc_nextgen_specular_power', 0.0), 0.0)
    _pack_f32(blob, 0x44, specular_power)
    _pack_f32(blob, 0x48, _safe_float(_material_prop(material, 'trlau_pc_nextgen_specular_shift0', 0.0), 0.0))
    _pack_f32(blob, 0x4C, _safe_float(_material_prop(material, 'trlau_pc_nextgen_specular_shift1', 0.0), 0.0))
    rim = _json_colors(_material_prop(material, 'trlau_pc_nextgen_rim_light_color', '[]'), 1, (0.0, 0.0, 0.0, 0.0))[0]
    struct.pack_into('<ffff', blob, 0x50, *rim)
    _pack_f32(blob, 0x60, _safe_float(_material_prop(material, 'trlau_pc_nextgen_rim_light_intensity', 0.0), 0.0))
    _pack_f32(blob, 0x64, _safe_float(_material_prop(material, 'trlau_pc_nextgen_water_blend_bias', 0.0), 0.0))
    _pack_f32(blob, 0x68, _safe_float(_material_prop(material, 'trlau_pc_nextgen_water_blend_exponent', 0.0), 0.0))
    water = _json_colors(_material_prop(material, 'trlau_pc_nextgen_water_deep_color', '[]'), 1, (0.0, 0.0, 0.0, 0.0))[0]
    struct.pack_into('<ffff', blob, 0x6C, *water)

    layer_texture_ids = _resolved_pc_nextgen_layer_texture_ids(material)
    layer_enabled = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_enabled', ''), 8, 0)
    layer_colors = _json_colors(_material_prop(material, 'trlau_pc_nextgen_layer_colors', '[]'), 8, (1.0, 1.0, 1.0, 1.0))
    layer_texcoord_sources = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_texcoord_sources', ''), 8, 0)
    layer_modifiers = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_modifiers', ''), 8, 0)
    layer_param_ids = _csv_ints(_material_prop(material, 'trlau_pc_nextgen_layer_param_ids', ''), 8, 0)
    layer_constants = _json_colors(_material_prop(material, 'trlau_pc_nextgen_layer_constants', '[]'), 8, (0.0, 0.0, 0.0, 0.0))

    local_pixmap_count = 0
    for layer_index in range(8):
        layer_offset = 0x80 + layer_index * 36
        blob[layer_offset:layer_offset + 4] = _rgba_float_to_bgra_bytes(layer_colors[layer_index])
        _pack_i32(blob, layer_offset + 0x04, layer_texcoord_sources[layer_index])
        _pack_i32(blob, layer_offset + 0x08, layer_modifiers[layer_index])
        _pack_i32(blob, layer_offset + 0x0C, layer_param_ids[layer_index])
        struct.pack_into('<ffff', blob, layer_offset + 0x10, *layer_constants[layer_index])
        texture_id = int(layer_texture_ids[layer_index])
        layer_is_enabled = int(layer_enabled[layer_index]) != 0
        mapped_texture_index = 0xFFFF
        if layer_pixmap_indices is not None and layer_index < len(layer_pixmap_indices):
            try:
                mapped_texture_index = int(layer_pixmap_indices[layer_index])
            except Exception:
                mapped_texture_index = 0xFFFF
        if layer_is_enabled and texture_id >= 0 and 0 <= mapped_texture_index < 0xFFFF:
            texture_index = int(mapped_texture_index)
        else:
            # Disabled/unused PCD9 layers use textureIndex 0xFFFF.  Do not
            # auto-enable a layer just because a stale texture ID remained in
            # Blender metadata.
            texture_index = 0xFFFF
            layer_enabled[layer_index] = 0
        num_textures = _pc_nextgen_layer_num_textures_for_export(layer_is_enabled, texture_id, texture_index)
        if num_textures > 0:
            local_pixmap_count += 1
        else:
            layer_enabled[layer_index] = 0
        _pack_u16(blob, layer_offset + 0x20, texture_index)
        _pack_u8(blob, layer_offset + 0x22, num_textures)
        _pack_s8(blob, layer_offset + 0x23, layer_enabled[layer_index])
    _pack_u8(blob, 0x7C, local_pixmap_count)
    shader_indices = _pc_nextgen_safe_shader_indices(material, shader_table_count)
    for shader_index, value in enumerate(shader_indices[:8]):
        _pack_u32(blob, 0x1A0 + shader_index * 4, int(value))
    _pack_u32(blob, 0x1C0, _PC_NEXTGEN_NO_FX_MATERIAL_OFFSET)
    return bytes(blob)


_PC_NEXTGEN_MAX_BATCH_VERTICES = 32767
_PC_NEXTGEN_MAX_SKIN_MAP_SIZE = 16


class _GeneratedPrimGroup:
    __slots__ = ('material_index', 'triangles', 'index_base', 'double_sided_keys')
    def __init__(self, material_index: int):
        self.material_index = int(material_index)
        self.triangles: list[tuple[int, int, int]] = []
        self.index_base = 0
        # For PC Next-Gen Double Sided materials, the mesh may already contain
        # both windings from the imported face data.  Track each logical
        # triangle so the generic writer emits one front/back pair, not a new
        # pair for every already-duplicated Blender polygon.
        self.double_sided_keys: set[tuple[int, int, int]] = set()


class _GeneratedBatch:
    __slots__ = ('vertices', 'triangles', 'groups', 'skin_map', 'skin_offset', 'vertex_data_offset', 'index_base', 'key_map', 'include_tangent_basis')
    def __init__(self, *, include_tangent_basis: bool = True):
        self.vertices: list[dict] = []
        # Compatibility aggregate. The writer uses groups, but keeping this
        # populated makes debug/introspection code simpler and avoids older call
        # paths assuming a flat triangle list.
        self.triangles: list[tuple[int, int, int]] = []
        self.groups: list[_GeneratedPrimGroup] = []
        self.skin_map: list[int] = []
        self.skin_offset = 0
        self.vertex_data_offset = 0
        self.index_base = 0
        self.key_map: dict[tuple[int, tuple[float, float]], int] = {}
        self.include_tangent_basis = bool(include_tangent_basis)

    def triangle_count(self) -> int:
        return sum(len(group.triangles) for group in self.groups)


def _weights_or_default(weights: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if weights:
        return [(int(bone), float(weight)) for bone, weight in weights[:2]]
    return [(0, 1.0)]


def _triangle_skin_bones(vertex_payloads: Sequence[dict]) -> list[int]:
    result: list[int] = []
    for payload in vertex_payloads:
        for bone_index, _weight in _weights_or_default(payload.get('weights') or []):
            bone_index = int(bone_index)
            if bone_index not in result:
                result.append(bone_index)
    return result


def _can_add_triangle_to_batch(batch: _GeneratedBatch, needed_keys: Sequence[tuple[int, tuple[float, float], int]], vertex_payloads: Sequence[dict]) -> bool:
    new_vertex_count = sum(1 for source_vertex, uv_key, _loop_index in needed_keys if (int(source_vertex), uv_key) not in batch.key_map)
    if len(batch.vertices) + int(new_vertex_count) > _PC_NEXTGEN_MAX_BATCH_VERTICES:
        return False
    skin_map = list(batch.skin_map or [])
    for bone_index in _triangle_skin_bones(vertex_payloads):
        if int(bone_index) not in skin_map:
            skin_map.append(int(bone_index))
            if len(skin_map) > _PC_NEXTGEN_MAX_SKIN_MAP_SIZE:
                return False
    return True


def _append_group_triangle(batch: _GeneratedBatch, material_index: int, tri: tuple[int, int, int], *, double_sided: bool) -> None:
    if not batch.groups or int(batch.groups[-1].material_index) != int(material_index):
        batch.groups.append(_GeneratedPrimGroup(material_index))
    group = batch.groups[-1]
    tri3 = tuple(int(v) for v in tri)
    if len(set(tri3)) != 3:
        return
    if double_sided:
        # Imported PC Next-Gen files can already contain explicit double-wound
        # faces.  Earlier generic-writer builds appended a reverse face for
        # every Blender polygon, so an imported double-wound pair became four
        # triangles on export.  Alpha-blended materials are especially sensitive
        # to that overdraw and render white/opaque in game.  Emit exactly one
        # front/back pair per unique logical triangle instead.
        key = tuple(sorted(tri3))
        if key in group.double_sided_keys:
            return
        group.double_sided_keys.add(key)
        group.triangles.append(tri3)
        batch.triangles.append(tri3)
        reversed_tri = (int(tri3[0]), int(tri3[2]), int(tri3[1]))
        group.triangles.append(reversed_tri)
        batch.triangles.append(reversed_tri)
        return
    group.triangles.append(tri3)
    batch.triangles.append(tri3)


def _build_generated_nextgen_batches(mesh_obj, arm_obj) -> list[_GeneratedBatch]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        return []
    try:
        mesh.calc_normals()
    except Exception:
        pass
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass
    uv_by_loop = _mesh_uv_lookup_by_loop(mesh)
    color_by_loop = _mesh_color_lookup_by_loop(mesh)
    arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else Matrix.Identity(4)
    normal_world_to_arm = arm_inv.to_3x3()
    material_count = max(1, len(_material_slot_records(mesh)))

    generated: list[_GeneratedBatch] = []
    current_batch: _GeneratedBatch | None = None

    def new_batch(*, include_tangent_basis: bool = True) -> _GeneratedBatch:
        batch = _GeneratedBatch(include_tangent_basis=include_tangent_basis)
        generated.append(batch)
        return batch

    for tri in list(getattr(mesh, 'loop_triangles', []) or []):
        try:
            polygon = mesh.polygons[int(tri.polygon_index)]
            material_index = max(0, min(material_count - 1, _material_table_index_for_polygon(mesh, polygon)))
            material = _material_slot_records(mesh)[material_index]
            double_sided = _pc_nextgen_material_double_sided(material)
            include_tangent_basis = not _pc_nextgen_material_uses_compact_vertex_layout(material)
            loop_indices = _tri_loop_indices(mesh, tri)
            source_vertices = tuple(int(v) for v in tri.vertices[:3])

            needed_keys: list[tuple[int, tuple[float, float], int]] = []
            vertex_payloads: list[dict] = []
            for corner, source_vertex in enumerate(source_vertices):
                loop_index = int(loop_indices[corner]) if corner < len(loop_indices) else int(loop_indices[0])
                uv = uv_by_loop.get(loop_index, (0.0, 0.0))
                uv_key = (round(float(uv[0]), 8), round(float(uv[1]), 8))
                key = (int(source_vertex), uv_key)
                vertex = mesh.vertices[int(source_vertex)]
                world = mesh_obj.matrix_world @ vertex.co
                local_pos = arm_inv @ world
                normal = normal_world_to_arm @ (mesh_obj.matrix_world.to_3x3() @ _normal_for_vertex(mesh, source_vertex))
                color = color_by_loop.get(loop_index, (1.0, 1.0, 1.0, 1.0))
                weights = _bone_weights_for_vertex_any(mesh_obj, int(source_vertex))
                needed_keys.append((int(source_vertex), uv_key, int(loop_index)))
                vertex_payloads.append({'key': key, 'position': local_pos, 'normal': normal, 'uv': uv_key, 'color': color, 'weights': weights})

            if len(set(source_vertices)) != 3:
                continue
            triangle_bones = _triangle_skin_bones(vertex_payloads)
            if len(triangle_bones) > _PC_NEXTGEN_MAX_SKIN_MAP_SIZE:
                # This should be practically unreachable because the exporter only
                # writes the two strongest influences per vertex, but keep a clear
                # error instead of silently truncating the PCModelBatch skin map.
                raise ValueError(
                    f'PC Next Gen triangle needs {len(triangle_bones)} unique bones, exceeding the PCModelBatch skin map limit of {_PC_NEXTGEN_MAX_SKIN_MAP_SIZE}.'
                )

            if (
                current_batch is None
                or bool(getattr(current_batch, 'include_tangent_basis', True)) != bool(include_tangent_basis)
                or (current_batch.triangles and not _can_add_triangle_to_batch(current_batch, needed_keys, vertex_payloads))
            ):
                current_batch = new_batch(include_tangent_basis=include_tangent_basis)
            elif current_batch is None:
                current_batch = new_batch(include_tangent_basis=include_tangent_basis)

            # If the current empty batch still cannot accept the triangle, fail
            # explicitly.  That means a single triangle exceeded a hard format
            # constraint and no amount of batch splitting can repair it.
            if not _can_add_triangle_to_batch(current_batch, needed_keys, vertex_payloads):
                current_batch = new_batch(include_tangent_basis=include_tangent_basis)
                if not _can_add_triangle_to_batch(current_batch, needed_keys, vertex_payloads):
                    raise ValueError('PC Next Gen triangle cannot fit in a generated PCModelBatch')

            for bone_index in triangle_bones:
                if int(bone_index) not in current_batch.skin_map:
                    current_batch.skin_map.append(int(bone_index))
            if not current_batch.skin_map:
                current_batch.skin_map.append(0)

            local_tri: list[int] = []
            for payload in vertex_payloads:
                key = payload['key']
                local_index = current_batch.key_map.get(key)
                if local_index is None:
                    current_batch.vertices.append({
                        'position': payload['position'],
                        'normal': payload['normal'],
                        'uv': payload['uv'],
                        'color': payload['color'],
                        'weights': payload['weights'],
                    })
                    local_index = len(current_batch.vertices) - 1
                    current_batch.key_map[key] = local_index
                local_tri.append(int(local_index))
            if len(set(local_tri)) != 3:
                continue
            _append_group_triangle(current_batch, material_index, (local_tri[0], local_tri[1], local_tri[2]), double_sided=double_sided)
        except Exception:
            logger.debug('Skipped PC Next Gen export triangle', exc_info=True)
            continue

    result: list[_GeneratedBatch] = []
    for batch in generated:
        if not batch.triangles or not batch.groups:
            continue
        if len(batch.vertices) > _PC_NEXTGEN_MAX_BATCH_VERTICES:
            raise ValueError(f'Generated PC Next Gen batch uses {len(batch.vertices)} vertices; maximum is {_PC_NEXTGEN_MAX_BATCH_VERTICES}.')
        if len(batch.skin_map) > _PC_NEXTGEN_MAX_SKIN_MAP_SIZE:
            raise ValueError(f'Generated PC Next Gen batch has skin map size {len(batch.skin_map)}; maximum is {_PC_NEXTGEN_MAX_SKIN_MAP_SIZE}.')
        if not batch.skin_map:
            batch.skin_map = [0]
        result.append(batch)
    return result



def _pc_nextgen_bone_table_from_template(template_path: Path | None) -> list[int]:
    """Read the global PCModelData bone table from a source next-gen section.

    The PC batch skin maps used by TR7 PC are stored as the values consumed by
    the runtime vertex palette.  In vanilla files the global bone table for Lara
    is identity (0..108).  Rebuilding it as a compact "used bone" list makes
    otherwise-valid skin-map entries point at the wrong global bones in game,
    producing long spiked/skinned triangles.  Preserve the template table when
    available; otherwise generate an identity table large enough for the skin
    map entries we write.
    """
    if template_path is None:
        return []
    path = Path(template_path)
    if not path.exists():
        return []
    try:
        parser = TRNextGenModelParser(str(path), cdc_render_data_id=0)
        parser.parse()
        header_offsets = parser._read_nextgen_header_offsets()
        pc_base = parser._find_pcmodeldata_base(header_offsets[0])
        header = parser._parse_header(pc_base)
        base = int(header.base) + int(header.bone_offset)
        count = max(0, int(header.num_bones))
        if count <= 0 or not parser._valid_abs(base, count * 4):
            return []
        return [int(parser._u32(base + index * 4)) for index in range(count)]
    except Exception:
        logger.debug('Could not read PC Next Gen template bone table from %s', path, exc_info=True)
        return []


def _pc_nextgen_bone_table_for_export(batches: Sequence[_GeneratedBatch], template_path: Path | None = None) -> list[int]:
    used_bones: list[int] = []
    max_bone = 0
    for batch in batches:
        for bone in list(getattr(batch, 'skin_map', []) or []):
            bone = int(bone)
            if bone < 0:
                continue
            max_bone = max(max_bone, bone)
            if bone not in used_bones:
                used_bones.append(bone)
    template_table = _pc_nextgen_bone_table_from_template(template_path)
    if template_table and len(template_table) > max_bone:
        return [int(value) for value in template_table]
    if used_bones:
        return [int(index) for index in range(max_bone + 1)]
    return [0]


def _pack_pc_nextgen_generic_payload(mesh_obj, arm_obj, render_id: int, *, template_path: Path | None = None) -> tuple[bytes, int, int, int]:
    mesh = getattr(mesh_obj, 'data', None)
    if mesh is None:
        raise ValueError('PC Next Gen export requires a mesh object')
    materials = _material_slot_records(mesh)
    pixmaps, layer_pixmap_indices_by_material = _pc_nextgen_build_pixmap_layout(materials, template_path)
    shader_table_ids = _pc_nextgen_shader_table_for_export(materials, template_path)
    batches = _build_generated_nextgen_batches(mesh_obj, arm_obj)
    if not batches:
        raise ValueError('PC Next Gen mesh has no exportable triangles')

    pc_base = 0x10
    header_size = 0x80
    material_offset = header_size
    material_table_size = len(materials) * _PC_NEXTGEN_MATERIAL_STRIDE
    bone_set = _pc_nextgen_bone_table_for_export(batches, template_path)
    bone_offset = _align(material_offset + material_table_size, 16)
    bone_table_size = len(bone_set) * 4
    batch_offset = _align(bone_offset + bone_table_size, 16)
    batch_table_size = len(batches) * 0xAC
    skin_cursor = _align(batch_offset + batch_table_size, 16)
    for batch in batches:
        setattr(batch, 'skin_offset', skin_cursor)
        skin_cursor = _align(skin_cursor + len(batch.skin_map) * 4, 4)
    vertex_cursor = _align(skin_cursor, 16)
    for batch in batches:
        batch.vertex_data_offset = vertex_cursor
        stride = 68 if bool(getattr(batch, 'include_tangent_basis', True)) else 44
        vertex_cursor = _align(vertex_cursor + len(batch.vertices) * stride, 16)
    index_offset = _align(vertex_cursor, 16)
    index_cursor = index_offset
    num_prim_groups = sum(len(batch.groups) for batch in batches)
    for batch in batches:
        batch.index_base = (index_cursor - index_offset) // 2
        for group in batch.groups:
            group.index_base = (index_cursor - index_offset) // 2
            index_cursor += len(group.triangles) * 3 * 2
    num_indices = (index_cursor - index_offset) // 2
    prim_group_offset = _align(index_cursor, 16)
    prim_group_size = int(num_prim_groups) * 20
    # Vanilla TR7 PC next-gen sections keep the auxiliary counted tables after
    # the PCModelData block in this order:
    #   specialMaterialFlags -> pixMaps -> shaders
    # and PCModelData.totalDataSize points to the start of the special-material
    # table, not to the physical end of the section.  Earlier generic-writer
    # builds placed pixMaps first and wrote totalDataSize as the full section
    # size.  The rebuilt payload could parse in our tools, but the game can use
    # totalDataSize/table ordering when setting up render data, so write the
    # vanilla layout here.
    pc_model_data_end = _align(prim_group_offset + prim_group_size, 16)
    special_flags_offset = pc_model_data_end
    special_flags: list[int] = []
    for index, material in enumerate(materials):
        if bool(_material_prop(material, 'trlau_pc_nextgen_special_material_flag', False)):
            special_flags.append(int(index))
    special_flags_size = 4 + len(special_flags) * 4
    pixmap_table_offset = _align(special_flags_offset + special_flags_size, 16)
    pixmap_table_size = 4 + len(pixmaps) * 4
    shaders_offset = _align(pixmap_table_offset + pixmap_table_size, 16)
    shaders_size = 4 + len(shader_table_ids) * 4
    total_pc_size = _align(shaders_offset + shaders_size, 16)
    pc_model_total_data_size = pc_base + special_flags_offset
    data = bytearray(pc_base + total_pc_size)

    # nextGenSpecificData offsets are relative to the section data start.  The
    # PCModelData field keeps the observed 0x0C raw value with a self relocation,
    # while the parser/game reaches the PCD9 block at data+0x10.
    struct.pack_into('<IIII', data, 0, 0x0C, pc_base + pixmap_table_offset, pc_base + shaders_offset, pc_base + special_flags_offset)

    # Header.
    positions: list[Vector] = []
    for batch in batches:
        for item in batch.vertices:
            positions.append(item['position'])
    center, box_min, box_max, radius = _compute_bounds(positions)
    struct.pack_into('<4sIII', data, pc_base + 0x00, b'PCD9', 0x18, int(pc_model_total_data_size), int(num_indices))
    struct.pack_into('<4f', data, pc_base + 0x10, *center)
    struct.pack_into('<4f', data, pc_base + 0x20, *box_min)
    struct.pack_into('<4f', data, pc_base + 0x30, *box_max)
    struct.pack_into('<f', data, pc_base + 0x40, float(radius))
    struct.pack_into('<I', data, pc_base + 0x44, 1)  # kTypeSkinned-compatible vertex declaration
    struct.pack_into('<f', data, pc_base + 0x48, 0.0)
    struct.pack_into('<IIIIII', data, pc_base + 0x4C, int(prim_group_offset), int(batch_offset), int(bone_offset), int(material_offset), int(index_offset), 0xFFFFFFFF)
    struct.pack_into('<HHHHHH', data, pc_base + 0x64, int(num_prim_groups), len(batches), len(bone_set), len(materials), len(pixmaps), 0)

    # Materials.
    for material_index, material in enumerate(materials):
        layer_pixmap_indices = layer_pixmap_indices_by_material[material_index] if material_index < len(layer_pixmap_indices_by_material) else None
        record = _pack_pc_nextgen_material_record(material, material_index, layer_pixmap_indices, len(shader_table_ids))
        start = pc_base + material_offset + material_index * _PC_NEXTGEN_MATERIAL_STRIDE
        data[start:start + _PC_NEXTGEN_MATERIAL_STRIDE] = record

    # Global bone table.
    for index, bone in enumerate(bone_set):
        struct.pack_into('<I', data, pc_base + bone_offset + index * 4, int(bone) & 0xFFFFFFFF)

    # Batches, skin maps, vertices, prim groups, indices.
    group_cursor = 0
    for batch_index, batch in enumerate(batches):
        boff = pc_base + batch_offset + batch_index * 0xAC
        if len(batch.skin_map) > _PC_NEXTGEN_MAX_SKIN_MAP_SIZE:
            raise ValueError(f'Generated PC Next Gen batch {batch_index} has skin map size {len(batch.skin_map)}; maximum is {_PC_NEXTGEN_MAX_SKIN_MAP_SIZE}.')
        # PCModelBatch.flags is not a skinning flag.  Vanilla TR7 PC Next-Gen
        # Lara uses flags=1 for fully opaque batches, but flags=0 for batches
        # that contain alpha/blended primitive groups.  Earlier generic writer
        # builds derived this from whether vertices had bone weights, so alpha
        # materials were exported in flags=1 batches.  The game then rendered
        # diffuse alpha incorrectly, exposing the white RGB in transparent pixels
        # on eyelashes/hair.  Match the observed vanilla rule instead.
        batch_has_alpha_material = False
        for group in batch.groups:
            try:
                material = materials[int(group.material_index)]
                if int(_material_prop(material, 'trlau_pc_nextgen_blend_mode', 0)) != 0:
                    batch_has_alpha_material = True
                    break
            except Exception:
                continue
        flags = 0 if batch_has_alpha_material else 1
        include_tangent_basis = bool(getattr(batch, 'include_tangent_basis', True))
        vertex_elements = _pc_nextgen_vertex_elements_blob(include_tangent_basis=include_tangent_basis)
        vertex_format = 0xC03F if include_tangent_basis else 0x003F
        vertex_stride = 68 if include_tangent_basis else 44
        total_batch_triangles = batch.triangle_count()
        struct.pack_into('<IIHxxII', data, boff + 0x00, int(flags), len(batch.groups), len(batch.skin_map), int(getattr(batch, 'skin_offset')), int(batch.vertex_data_offset))
        data[boff + 0x18:boff + 0x18 + len(vertex_elements)] = vertex_elements
        struct.pack_into('<IIIII', data, boff + 0x98, int(vertex_format), int(vertex_stride), len(batch.vertices), int(batch.index_base), int(total_batch_triangles))
        for skin_index, bone in enumerate(batch.skin_map):
            struct.pack_into('<I', data, pc_base + int(getattr(batch, 'skin_offset')) + skin_index * 4, int(bone) & 0xFFFFFFFF)
        for vertex_index, item in enumerate(batch.vertices):
            record = _pack_pc_nextgen_vertex_record(item['position'], item['normal'], item['uv'], item['color'], item['weights'], batch.skin_map, include_tangent_basis=include_tangent_basis)
            start = pc_base + batch.vertex_data_offset + vertex_index * vertex_stride
            data[start:start + vertex_stride] = record
        for group in batch.groups:
            goff = pc_base + prim_group_offset + group_cursor * 20
            unique_group_vertices = len({int(index) for tri in group.triangles for index in tri})
            struct.pack_into('<IIIHxxI', data, goff, int(group.index_base), len(group.triangles), int(unique_group_vertices), 1, int(group.material_index))
            for tri_index, (a, b, c) in enumerate(group.triangles):
                ioff = pc_base + index_offset + (int(group.index_base) + tri_index * 3) * 2
                struct.pack_into('<hhh', data, ioff, int(a), int(b), int(c))
            group_cursor += 1

    # Pixmaps, special flags, placeholder shader table.
    struct.pack_into('<i', data, pc_base + pixmap_table_offset, len(pixmaps))
    for index, texture_id in enumerate(pixmaps):
        struct.pack_into('<i', data, pc_base + pixmap_table_offset + 4 + index * 4, int(texture_id))
    struct.pack_into('<i', data, pc_base + special_flags_offset, len(special_flags))
    for index, material_index in enumerate(special_flags):
        struct.pack_into('<i', data, pc_base + special_flags_offset + 4 + index * 4, int(material_index))
    struct.pack_into('<i', data, pc_base + shaders_offset, len(shader_table_ids))
    for index, shader_id in enumerate(shader_table_ids):
        struct.pack_into('<i', data, pc_base + shaders_offset + 4 + index * 4, int(shader_id))
    return bytes(data), sum(len(batch.vertices) for batch in batches), sum(batch.triangle_count() for batch in batches), len(materials)


def _write_generic_pc_nextgen_model_section(
    directory: Path,
    mesh_obj,
    arm_obj,
    render_id: int,
    *,
    section_list=None,
    template_path: Path | None = None,
    reference_template_path: Path | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> NextGenExportResult:
    directory = Path(directory)
    render_id = int(render_id) & 0xFFFFFFFF
    out_path = Path(template_path) if template_path is not None else _allocate_nextgen_output_path(directory, render_id, section_list=section_list)
    reference_path = Path(reference_template_path) if reference_template_path is not None else (Path(template_path) if template_path is not None else None)
    if template_path is None:
        _warn(warning_callback, f'No existing next-gen section matched RenderID 0x{render_id:X}; creating a new section {out_path.name}.')
    data, vertex_count, face_count, material_count = _pack_pc_nextgen_generic_payload(mesh_obj, arm_obj, render_id, template_path=reference_path)
    section_index = _section_index_for_output(out_path, int(render_id) & 0x1FFF)
    _write_pc_nextgen_standalone_section(out_path, data, render_id, section_index=section_index, template_path=reference_path)
    if section_list is not None:
        try:
            section_list.ensure_index(section_index, out_path.name)
        except Exception:
            pass
    logger.info('Wrote generic PC next-gen model section %s RenderID=0x%X vertices=%d faces=%d materials=%d', out_path.name, render_id, vertex_count, face_count, material_count)
    return NextGenExportResult(out_path, render_id, vertex_count, face_count, material_count)
def export_pc_nextgen_model_section(
    context,
    directory: Path,
    model_root,
    arm_obj,
    mesh_obj,
    render_id: int,
    *,
    section_list=None,
    warning_callback: Callable[[str], None] | None = None,
) -> NextGenExportResult | None:
    directory = Path(directory)
    render_id = int(render_id) & 0xFFFFFFFF
    if render_id == 0 or mesh_obj is None:
        return None

    template_path = find_nextgen_section_template(directory, render_id)
    if template_path is None:
        source_template = _source_template_from_mesh(mesh_obj, render_id)
        if source_template is not None:
            output_path = _allocate_nextgen_output_path(directory, render_id, section_list=section_list)
            _warn(warning_callback, f'No matching next-gen section existed in the target DRM; creating {output_path.name} for RenderID 0x{render_id:X}.')
            return _write_generic_pc_nextgen_model_section(
                directory,
                mesh_obj,
                arm_obj,
                render_id,
                section_list=section_list,
                template_path=output_path,
                reference_template_path=source_template,
                warning_callback=warning_callback,
            )
        return _write_generic_pc_nextgen_model_section(
            directory,
            mesh_obj,
            arm_obj,
            render_id,
            section_list=section_list,
            template_path=None,
            warning_callback=warning_callback,
        )

    return _write_generic_pc_nextgen_model_section(
        directory,
        mesh_obj,
        arm_obj,
        render_id,
        section_list=section_list,
        template_path=template_path,
        reference_template_path=template_path,
        warning_callback=warning_callback,
    )
