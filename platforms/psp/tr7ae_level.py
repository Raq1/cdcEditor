from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import bpy

from ...core.log import logger
from ..common.drm_container import DRMContainerParser
from ..pc.tr7ae_level import (
    BGInstance,
    BGObject,
    LevelData,
    LevelStrip,
    TerrainGroup,
    TRLevelParser,
    _LEVEL_VERSION,
    _RawBinaryReader,
    _RawSection,
    _RawSectionList,
    _apply_bgobject_reflection_flags,
    _bgobject_has_render_data,
    _convert_bgobject_translation,
    _convert_level_position,
    _pick_bgobject_position,
)
from .texture import decode_supported_psp_texture

_PSP_TERRAIN_GROUP_SIZE = 0xA0
_PSP_TERRAIN_MATERIAL_SIZE = 0x10
_PSP_STRIP_MAX_VERTEX_COUNT = 0x4000
_PSP_MAX_STRIP_CHAIN = 100000
_PSP_MAX_OCTREE_NODES = 200000
_PSP_BGOBJECT_SIZE = 0x60
_PSP_BGINSTANCE_SIZE = 0x100
_PSP_BGOBJECT_FIRST_PRIMITIVE_SIZE = 0x30
_PSP_BGOBJECT_EXTRA_PRIMITIVE_SIZE = 0x30
_PSP_BGOBJECT_MAX_VERTEX_COUNT = 2_000_000
_PSP_ANNIVERSARY_LEVEL_VERSION = _LEVEL_VERSION
_PSP_LEGEND_LEVEL_VERSION = 0x04C204BF
_PSP_SUPPORTED_LEVEL_VERSIONS = {_PSP_ANNIVERSARY_LEVEL_VERSION, _PSP_LEGEND_LEVEL_VERSION}


@dataclass(slots=True)
class _PSPBGPrimitive:
    index: int
    offset: int
    vertex_count: int
    mode_word: int
    texture_id: int
    blend: int
    vertex_format_flags: int
    data_offset: int
    compact: bool = False

    @property
    def mode(self) -> int:
        return int(self.mode_word) & 0xFF

    @property
    def draw_group(self) -> int:
        return (int(self.mode_word) >> 8) & 0xFF



def _safe_u16(data: bytearray | bytes, offset: int) -> int:
    if offset < 0 or offset + 2 > len(data):
        return 0
    return struct.unpack_from('<H', data, offset)[0]


def _safe_u32(data: bytearray | bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return struct.unpack_from('<I', data, offset)[0]


def _safe_i32(data: bytearray | bytes, offset: int) -> int:
    if offset < 0 or offset + 4 > len(data):
        return 0
    return struct.unpack_from('<i', data, offset)[0]


def _safe_f32(data: bytearray | bytes, offset: int) -> float:
    if offset < 0 or offset + 4 > len(data):
        return 0.0
    try:
        value = struct.unpack_from('<f', data, offset)[0]
        return float(value) if math.isfinite(float(value)) else 0.0
    except Exception:
        return 0.0


def _in_range(data: bytearray | bytes, offset: int, size: int = 1) -> bool:
    offset = int(offset or 0)
    size = max(0, int(size or 0))
    return 0 <= offset and (offset + size) <= len(data)


def _s8(value: int) -> int:
    value = int(value) & 0xFF
    return value - 0x100 if value >= 0x80 else value


_PSP_LEVEL_UV_SCALE = 1.0 / 2048.0


def _psp_level_uv_tile_base(raw_values: List[int]) -> int:
    if not raw_values:
        return 0
    # PSP terrain strips store Q11 UVs and commonly put texture-edge values one
    # LSB below the next integer tile, e.g. 0x3FFF and 0x47FF for a full 0..1
    # span.  Add that LSB when selecting the integer tile origin so the decoded
    # UVs land on the local texture square instead of importing as a half-size
    # 12-bit fragment.
    return int(math.floor((float(min(int(v) for v in raw_values)) + 1.0) * _PSP_LEVEL_UV_SCALE))


def _decode_psp_level_uv16(raw_u: int, raw_v: int, u_tile_base: int, v_tile_base: int) -> Tuple[float, float]:
    return (
        ((float(int(raw_u)) + 1.0) * _PSP_LEVEL_UV_SCALE) - float(u_tile_base),
        ((float(int(raw_v)) + 1.0) * _PSP_LEVEL_UV_SCALE) - float(v_tile_base),
    )


def _psp_mode_from_blob_size(blob_size: int) -> int:
    size = int(blob_size)
    if size >= 22:
        return 8
    if size >= 20:
        return 6
    if size >= 18:
        return 4
    if size >= 16:
        return 2
    return 1


def _psp_weight_prefix_size(mode: int, blob_size: int) -> int:
    mode = int(mode) & 0xFF
    size = int(blob_size)
    if mode <= 0:
        return 0
    if mode == 1:
        return 1 if size >= 16 else 0
    if mode > 8:
        mode = 8
    min_tail = 14 + (1 if (mode & 1) else 0)
    return min(mode, max(0, size - min_tail))


def _psp_uv_offset(mode: int, blob_size: int) -> int:
    return _psp_weight_prefix_size(mode, blob_size)


def _psp_normal_offset(blob_size: int) -> int:
    return max(0, int(blob_size) - 10)


def _psp_color_offset(blob_size: int) -> int:
    return int(_psp_normal_offset(blob_size)) - 2


def _decode_psp_level_rgb565_vertex_color(value: int) -> Tuple[int, int, int, int]:
    value = int(value) & 0xFFFF

    # PSP level terrain stores this word as RGB565 vertex lighting.  The upper
    # five bits are the blue channel, not alpha/mask data.  Decoding them as
    # alpha makes level vertex colors partly transparent and collapses the
    # actual lighting down to a grayscale value.  This matches the PSP PCD
    # RGB565 texture channel order used elsewhere in the importer.
    r5 = value & 0x001F
    g6 = (value >> 5) & 0x003F
    b5 = (value >> 11) & 0x001F

    r = int((r5 * 255 + 15) // 31)
    g = int((g6 * 255 + 31) // 63)
    b = int((b5 * 255 + 15) // 31)
    return (r, g, b, 255)


def _read_psp_vertex_color(blob: bytes) -> Optional[Tuple[int, int, int, int]]:
    color_offset = _psp_color_offset(len(blob))
    if color_offset < 0 or color_offset + 2 > len(blob):
        return None
    try:
        return _decode_psp_level_rgb565_vertex_color(struct.unpack_from('<H', blob, color_offset)[0])
    except Exception:
        return None


def _psp_strip_vertex_key(level: LevelData, index: int) -> tuple:
    if 0 <= int(index) < len(level.vertices):
        pos = level.vertices[int(index)]
        # Topology restarts/degenerates are position-based.  Some PSP terrain
        # seam vertices duplicate xyz while changing UVs, so including UVs in
        # this key lets artificial bridge triangles survive.
        return (
            round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2]), 6),
        )
    return ('idx', int(index))


def _vec_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _vec_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        (float(a[1]) * float(b[2])) - (float(a[2]) * float(b[1])),
        (float(a[2]) * float(b[0])) - (float(a[0]) * float(b[2])),
        (float(a[0]) * float(b[1])) - (float(a[1]) * float(b[0])),
    )


def _vec_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return (float(a[0]) * float(b[0])) + (float(a[1]) * float(b[1])) + (float(a[2]) * float(b[2]))


def _convert_psp_level_normal(nx: int, ny: int, nz: int) -> Tuple[float, float, float]:
    # Terrain vertices use the same axis convention as their signed xyz shorts.
    # _convert_level_position(x, y, z) is equivalent to (-x, -y, z), so use
    # the same linear transform for byte-packed normals.
    return (-float(nx), -float(ny), float(nz))


def _orient_psp_triangle_to_normals(
    tri: Tuple[int, int, int],
    level: LevelData,
    normals: Optional[dict[int, Tuple[float, float, float]]] = None,
) -> Tuple[int, int, int]:
    if not normals:
        return tri
    try:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        if a not in normals or b not in normals or c not in normals:
            return tri
        pa = level.vertices[a]; pb = level.vertices[b]; pc = level.vertices[c]
        face_normal = _vec_cross(_vec_sub(pb, pa), _vec_sub(pc, pa))
        normal_sum = (
            float(normals[a][0]) + float(normals[b][0]) + float(normals[c][0]),
            float(normals[a][1]) + float(normals[b][1]) + float(normals[c][1]),
            float(normals[a][2]) + float(normals[b][2]) + float(normals[c][2]),
        )
        if abs(normal_sum[0]) + abs(normal_sum[1]) + abs(normal_sum[2]) <= 0.000001:
            return tri
        if _vec_dot(face_normal, normal_sum) < 0.0:
            return (a, c, b)
    except Exception:
        return tri
    return tri


def _looks_like_psp_strip_header(data: bytearray | bytes, offset: int, material_count: int = 1) -> bool:
    if not _in_range(data, offset, 0x38):
        return False
    strip_vertex_count = _safe_u16(data, offset + 0x00)
    list_vertex_count = _safe_u16(data, offset + 0x02)
    total_vertex_count = int(strip_vertex_count) + int(list_vertex_count)
    if total_vertex_count <= 0 or total_vertex_count > (_PSP_STRIP_MAX_VERTEX_COUNT * 2):
        return False
    # Valid PSP terrain draw-strip records observed in TRA/TRL retail data use
    # 0xFFFFFFFF as the third header word.  Pointer tables and material arrays
    # can otherwise look like huge strips when only the first half-word is used.
    if _safe_u32(data, offset + 0x08) != 0xFFFFFFFF:
        return False
    mat = _safe_i32(data, offset + 0x18)
    if mat < 0 or mat >= max(1, int(material_count)):
        return False
    # First-pass sanity: at least one 16-byte vertex should fit after the fixed
    # header.  Full count validation happens in _emit_psp_strip.
    return _in_range(data, offset + 0x38, 16)


def _strip_to_triangles(
    sequence: List[int],
    level: LevelData,
    normals: Optional[dict[int, Tuple[float, float, float]]] = None,
) -> List[int]:
    out: List[int] = []
    window: List[int] = []
    previous_key = None

    for raw_index in sequence:
        index = int(raw_index)
        key = _psp_strip_vertex_key(level, index)
        if previous_key is not None and key == previous_key:
            # PSP terrain chains use duplicated vertex records as strip-island
            # separators.  Keep the duplicate as the first vertex of the new
            # window so the next non-degenerate triangle starts locally.
            window = [index]
            previous_key = key
            continue

        window.append(index)
        previous_key = key
        if len(window) < 3:
            continue

        a, b, c = window[-3], window[-2], window[-1]
        key_a = _psp_strip_vertex_key(level, a)
        key_b = _psp_strip_vertex_key(level, b)
        key_c = _psp_strip_vertex_key(level, c)
        if key_a == key_b or key_b == key_c or key_a == key_c:
            continue

        local_i = len(window) - 1
        tri = (b, a, c) if (local_i & 1) else (a, b, c)
        tri = _orient_psp_triangle_to_normals(tri, level, normals)
        out.extend([int(tri[0]), int(tri[1]), int(tri[2])])
    return out


def _triangle_list_to_triangles(
    sequence: List[int],
    level: LevelData,
    normals: Optional[dict[int, Tuple[float, float, float]]] = None,
) -> List[int]:
    out: List[int] = []
    usable = (len(sequence) // 3) * 3
    for tri_start in range(0, usable, 3):
        tri = (int(sequence[tri_start]), int(sequence[tri_start + 1]), int(sequence[tri_start + 2]))
        key_a = _psp_strip_vertex_key(level, tri[0])
        key_b = _psp_strip_vertex_key(level, tri[1])
        key_c = _psp_strip_vertex_key(level, tri[2])
        if key_a == key_b or key_b == key_c or key_a == key_c:
            continue
        tri = _orient_psp_triangle_to_normals(tri, level, normals)
        out.extend([int(tri[0]), int(tri[1]), int(tri[2])])
    return out


def _is_finite_triplet(values: Tuple[float, float, float]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except Exception:
        return False


def _canonicalize_psp_bg_tpageid(texture_id: int, blend: int = 0) -> int:
    return (int(texture_id) & 0x1FFF) | ((int(blend) & 0xF) << 13)


def _looks_like_psp_bgobject_primitive_entry(data: bytearray | bytes, offset: int) -> bool:
    if not _in_range(data, offset, _PSP_BGOBJECT_EXTRA_PRIMITIVE_SIZE):
        return False
    vertex_count = _safe_u16(data, offset + 0x00)
    mode = _safe_u16(data, offset + 0x02) & 0xFF
    texture_id = _safe_u16(data, offset + 0x04)
    data_offset = _safe_u32(data, offset + 0x0C)
    if vertex_count <= 0 or vertex_count > _PSP_BGOBJECT_MAX_VERTEX_COUNT:
        return False
    if mode <= 0 or mode > 8:
        return False
    if texture_id >= 8192:
        return False
    if data_offset <= 0 or not _in_range(data, data_offset, 8):
        return False
    return True


def _looks_like_psp_bgobject_first_primitive(data: bytearray | bytes, offset: int) -> bool:
    return _looks_like_psp_bgobject_primitive_entry(data, offset)


def _looks_like_psp_bgobject_extra_primitive(data: bytearray | bytes, offset: int) -> bool:
    return _looks_like_psp_bgobject_primitive_entry(data, offset)


def _read_psp_bgobject_primitive_entry(data: bytearray | bytes, offset: int, index: int = 0) -> Optional[_PSPBGPrimitive]:
    if not _looks_like_psp_bgobject_primitive_entry(data, offset):
        return None
    return _PSPBGPrimitive(
        index=int(index),
        offset=int(offset),
        vertex_count=int(_safe_u16(data, offset + 0x00)),
        mode_word=int(_safe_u16(data, offset + 0x02)),
        texture_id=int(_safe_u16(data, offset + 0x04)),
        blend=int(_safe_u16(data, offset + 0x06)),
        vertex_format_flags=int(_safe_u32(data, offset + 0x08)),
        data_offset=int(_safe_u32(data, offset + 0x0C)),
        compact=False,
    )


def _read_psp_bgobject_first_primitive(data: bytearray | bytes, offset: int, index: int = 0) -> Optional[_PSPBGPrimitive]:
    return _read_psp_bgobject_primitive_entry(data, offset, index)


def _read_psp_bgobject_extra_primitive(data: bytearray | bytes, offset: int, index: int) -> Optional[_PSPBGPrimitive]:
    return _read_psp_bgobject_primitive_entry(data, offset, index)


def _looks_like_psp_direct_stream_header(data: bytearray | bytes, offset: int) -> bool:
    if not _in_range(data, offset, 8):
        return False
    if _safe_u32(data, offset + 4) != 0:
        return False
    value = _safe_f32(data, offset)
    return math.isfinite(float(value)) and 1.0e-6 < abs(float(value)) < 1.0e6


def _psp_bgobject_default_vertex_stride(mode: int) -> int:
    mode = int(mode) & 0xFF
    # Observed PSP BGObject draw streams use compact direct vertices:
    #   uv8x2, aux/color16, xyz16x3
    # i.e. 10 bytes after an optional 8-byte stream header.
    if mode == 1:
        return 10
    if mode == 2:
        return 12
    if mode in {3, 4}:
        return 16
    if mode in {5, 6}:
        return 18
    return 20


def _psp_bgobject_stride_candidates(mode: int) -> List[int]:
    mode = int(mode) & 0xFF
    if mode == 1:
        return [10, 12, 14, 16]
    if mode == 2:
        return [12, 10, 14, 16]
    if mode in {3, 4}:
        return [16, 14, 18, 20]
    if mode in {5, 6}:
        return [18, 16, 20, 22]
    if mode in {7, 8}:
        return [20, 18, 22, 24]
    return [10, 12, 14, 16, 18, 20, 22]


def _psp_bgobject_vertex_key(vertices: List[Tuple[float, float, float]], index: int) -> tuple:
    if 0 <= int(index) < len(vertices):
        pos = vertices[int(index)]
        return (round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2]), 6))
    return ('idx', int(index))


def _psp_bgobject_strip_to_triangles(sequence: List[int], vertices: List[Tuple[float, float, float]]) -> List[int]:
    out: List[int] = []
    window: List[int] = []
    previous_key = None
    for raw_index in sequence:
        index = int(raw_index)
        key = _psp_bgobject_vertex_key(vertices, index)
        if previous_key is not None and key == previous_key:
            window = [index]
            previous_key = key
            continue
        window.append(index)
        previous_key = key
        if len(window) < 3:
            continue
        a, b, c = window[-3], window[-2], window[-1]
        key_a = _psp_bgobject_vertex_key(vertices, a)
        key_b = _psp_bgobject_vertex_key(vertices, b)
        key_c = _psp_bgobject_vertex_key(vertices, c)
        if key_a == key_b or key_b == key_c or key_a == key_c:
            continue
        local_i = len(window) - 1
        if local_i & 1:
            out.extend([b, a, c])
        else:
            out.extend([a, b, c])
    return out



class TRPSPLevelParser(TRLevelParser):

    @staticmethod
    def _decode_container_bytes(data: bytes) -> bytes:
        decoded = bytes(data)
        for _ in range(4):
            if decoded.startswith(DRMContainerParser.DERICKW_MAGIC):
                decoded = DRMContainerParser.decompress_derickw_bytes(decoded)
                continue
            if DRMContainerParser._looks_like_cdrm_bytes(decoded[:8]):
                decoded = DRMContainerParser.decompress_cdrm_bytes(decoded)
                continue
            break
        return decoded

    @classmethod
    def _read_decoded_file(cls, filepath: str) -> bytes:
        with open(filepath, 'rb') as fh:
            return cls._decode_container_bytes(fh.read())

    @staticmethod
    def is_level_file(filepath: str) -> bool:
        suffix = Path(filepath).suffix.lower()
        if suffix == '.drm':
            try:
                data = TRPSPLevelParser._read_decoded_file(filepath)
                reader = _RawBinaryReader(data)
                sections = _RawSectionList(reader, relocate=False)
                first_section = sections.get_section(0)
                return _safe_u32(reader.data, int(first_section.offset) + 168) in _PSP_SUPPORTED_LEVEL_VERSIONS
            except Exception:
                return False
        return TRLevelParser.is_level_file(filepath)

    def parse(self) -> LevelData:
        if Path(self.filepath).suffix.lower() == '.drm':
            return self._parse_psp_drm()
        return super().parse()

    def _parse_psp_drm(self) -> LevelData:
        data = self._read_decoded_file(self.filepath)

        type_reader = _RawBinaryReader(data)
        type_sections = _RawSectionList(type_reader, relocate=False)
        root_type_section = type_sections.get_section(0)
        level_magic = _safe_u32(type_reader.data, int(root_type_section.offset) + 168)
        if level_magic not in _PSP_SUPPORTED_LEVEL_VERSIONS:
            raise ValueError(f'File does not appear to contain PSP TRLAU level data (marker=0x{level_magic:08X})')

        reader = _RawBinaryReader(data)
        sections = _RawSectionList(reader, relocate=True)
        root_section = sections.get_section(0)
        root_offset = int(root_section.offset)
        terrain_offset = _safe_u32(reader.data, root_offset)
        if terrain_offset <= 0 or not _in_range(reader.data, terrain_offset, 0x50):
            raise ValueError(f'PSP level terrain pointer is invalid: 0x{terrain_offset:08X}')

        level = LevelData(filepath=self.filepath)
        # TRA PSP level roots use marker 0x04C204BB; TRL PSP level roots use
        # 0x04C204BF.  The importer UI has historically passed an Anniversary
        # hint for all PSP level DRMs, so prefer the root marker when selecting
        # Legend-vs-Anniversary metadata layouts such as MarkUp entries.
        if int(level_magic) == _PSP_LEGEND_LEVEL_VERSION:
            level.source_game = 'legend'
        elif int(level_magic) == _PSP_ANNIVERSARY_LEVEL_VERSION:
            level.source_game = 'anniversary'
        else:
            level.source_game = self._normalize_level_game(self.game_hint)
            if level.source_game == 'unknown':
                level.source_game = 'anniversary'
        self.level_cdc_render_data_id = 1 if level.source_game == 'legend' else 0
        level.cdc_render_data_id = int(self.level_cdc_render_data_id)
        level.level_metadata = {
            'psp_level_import': True,
            'psp_level_version': int(level_magic),
            'psp_detected_game': str(level.source_game),
            'psp_root_section_offset': int(root_offset),
            'psp_terrain_offset': int(terrain_offset),
        }

        self._read_psp_terrain(reader, sections, level, terrain_offset)

        # The PSP DRM level root keeps the PC/LAU Level fields for MarkUp,
        # camera metadata, terrain lights and IntroData.  The terrain payload is
        # platform-specific, so it is parsed above, but the shared root metadata
        # path can be reused after relocation has converted DRM pointers to
        # absolute offsets.
        psp_metadata = dict(getattr(level, 'level_metadata', {}) or {})
        try:
            self._populate_drm_level_metadata(reader, sections, root_section, level)
            self._populate_raw_intro_data_blocks(reader, level)
            self._populate_psp_unit_data_fallback(reader, sections, level)
        except Exception as exc:
            logger.warning('Failed to import PSP level root metadata from %s: %s', Path(self.filepath).name, exc)
            try:
                self._populate_psp_unit_data_fallback(reader, sections, level)
            except Exception:
                pass
        finally:
            merged_metadata = dict(getattr(level, 'level_metadata', {}) or {})
            merged_metadata.update(psp_metadata)
            merged_metadata.update({
                'psp_markups_imported': int(len(getattr(level, 'markups', []) or [])),
                'psp_intro_data_imported': int(len(getattr(level, 'intro_data', []) or [])),
                'psp_camera_keys_imported': int(len(getattr(level, 'camera_data', []) or [])),
                'psp_terrain_lights_imported': int(len(getattr(level, 'terrain_lights', []) or [])),
            })
            level.level_metadata = merged_metadata

        if self.import_textures:
            level.texture_images = self._load_drm_texture_images(reader, sections)
        return level

    def _populate_psp_unit_data_fallback(self, reader: _RawBinaryReader, sections: _RawSectionList, level: LevelData) -> None:
        existing = dict(getattr(level, 'unit_data', {}) or {})
        if existing and (int(existing.get('num_cines', 0) or 0) > 0 or existing.get('cine_entries')):
            return
        data = reader.data
        for _section_index, _section in enumerate(getattr(sections, 'sections', []) or []):
            try:
                setattr(_section, 'index', int(_section_index))
            except Exception:
                pass
        best = None
        best_score = -1
        for section in getattr(sections, 'sections', []) or []:
            try:
                section_type = int(getattr(section, 'type', 0) or 0)
                size = int(getattr(section, 'size', 0) or 0)
                offset = int(getattr(section, 'offset', 0) or 0)
            except Exception:
                continue
            if section_type != 7 or size < 0x1C0 or not _in_range(data, offset, min(size, 16)):
                continue
            num_fsfx = _safe_u32(data, offset + 0)
            p_fsfx = _safe_u32(data, offset + 4)
            num_cines = _safe_u32(data, offset + 8)
            p_cines = _safe_u32(data, offset + 12)
            if num_fsfx > 8192 or num_cines > 8192:
                continue
            score = 0
            reloc_offsets = {int(getattr(reloc, 'offset', -1)) for reloc in getattr(section, 'relocations', []) or []}
            if 4 in reloc_offsets:
                score += 1
            if 12 in reloc_offsets:
                score += 4
            if 0 < num_cines <= 128 and _in_range(data, p_cines, 4):
                score += 8
                first_cine = _safe_u32(data, p_cines)
                if self._find_raw_section_for_absolute_offset(sections, first_cine) is not None:
                    score += 8
            if 0 < num_fsfx <= 128 and _in_range(data, p_fsfx, 12):
                score += 2
            if score > best_score:
                best_score = score
                best = section
        if best is None or best_score <= 0:
            return
        try:
            parsed = self._parse_unit_data_blob(
                data,
                int(getattr(best, 'offset', 0) or 0),
                getattr(level, 'source_game', self.game_hint),
                section_id=int(getattr(best, 'id', 0) or 0),
                parse_pointer_arrays=True,
                raw_sections=sections,
            )
        except Exception as exc:
            logger.warning('Failed to parse PSP UnitData fallback section in %s: %s', Path(self.filepath).name, exc)
            return
        if not parsed:
            return
        if existing:
            merged = dict(parsed)
            merged.update(existing)
            if not existing.get('cine_entries') and parsed.get('cine_entries'):
                merged['cine_entries'] = parsed.get('cine_entries')
            if int(existing.get('num_cines', 0) or 0) <= 0 and int(parsed.get('num_cines', 0) or 0) > 0:
                merged['num_cines'] = parsed.get('num_cines')
            level.unit_data = merged
        else:
            level.unit_data = parsed

    def _read_psp_terrain(self, reader: _RawBinaryReader, sections: _RawSectionList, level: LevelData, terrain_offset: int) -> None:
        data = reader.data
        num_groups = _safe_i32(data, terrain_offset + 0x14)
        group_list = _safe_u32(data, terrain_offset + 0x18)
        if num_groups < 0 or num_groups > 8192 or not _in_range(data, group_list, max(0, num_groups) * _PSP_TERRAIN_GROUP_SIZE):
            raise ValueError(f'Invalid PSP TerrainGroup table at 0x{group_list:08X} count={num_groups}')

        signal_group = _safe_u32(data, terrain_offset + 0x1C)
        level.terrain_signal_list_abs = int(_safe_u32(data, terrain_offset + 0x20))
        level.level_metadata.update({
            'psp_num_terrain_groups': int(num_groups),
            'psp_terrain_groups_offset': int(group_list),
            'psp_signal_terrain_group_offset': int(signal_group),
            'psp_bg_instance_count': int(_safe_i32(data, terrain_offset + 0x28)),
            'psp_bg_instance_list_offset': int(_safe_u32(data, terrain_offset + 0x2C)),
            'psp_bg_object_count': int(_safe_i32(data, terrain_offset + 0x30)),
            'psp_bg_object_list_offset': int(_safe_u32(data, terrain_offset + 0x34)),
            'psp_vertex_stream_offset': int(_safe_u32(data, terrain_offset + 0x48)),
        })

        for index in range(int(num_groups)):
            group_offset = int(group_list) + (index * _PSP_TERRAIN_GROUP_SIZE)
            group = self._read_psp_terrain_group(reader, sections, level, index, group_offset)
            if group is not None:
                level.terrain_groups.append(group)

        num_bg_objects = int(_safe_i32(data, terrain_offset + 0x30))
        bg_object_list = int(_safe_u32(data, terrain_offset + 0x34))
        num_bg_instances = int(_safe_i32(data, terrain_offset + 0x28))
        bg_instance_list = int(_safe_u32(data, terrain_offset + 0x2C))
        level.bg_objects = self._read_psp_bg_objects(reader, bg_object_list, num_bg_objects)
        level.bg_instances = self._read_psp_bg_instances(reader, bg_instance_list, num_bg_instances, bg_object_list, num_bg_objects)
        level.level_metadata.update({
            'psp_bg_objects_imported': int(len(level.bg_objects)),
            'psp_bg_instances_imported': int(len(level.bg_instances)),
        })

    def _read_psp_terrain_group(self, reader: _RawBinaryReader, sections: _RawSectionList, level: LevelData, index: int, group_offset: int) -> Optional[TerrainGroup]:
        data = reader.data
        if not _in_range(data, group_offset, _PSP_TERRAIN_GROUP_SIZE):
            logger.warning('Skipping truncated PSP TerrainGroup %d at 0x%X', index, group_offset)
            return None

        global_raw = tuple(_safe_f32(data, group_offset + i * 4) for i in range(3))
        local_raw = tuple(_safe_f32(data, group_offset + 0x10 + i * 4) for i in range(3))
        global_offset = _convert_level_position(float(global_raw[0]), float(global_raw[1]), float(global_raw[2]))
        local_offset = _convert_level_position(float(local_raw[0]), float(local_raw[1]), float(local_raw[2]))
        material_list = _safe_u32(data, group_offset + 0x90)
        sorted_material_list = _safe_u32(data, group_offset + 0x94)
        octree = _safe_u32(data, group_offset + 0x44)
        mesh_or_collision = _safe_u32(data, group_offset + 0x38)

        group = TerrainGroup(
            index=int(index),
            position=tuple(float(v) for v in global_offset),
            global_offset=tuple(float(v) for v in global_offset),
            local_offset=tuple(float(v) for v in local_offset),
            flags=int(_safe_i32(data, group_offset + 0x20)),
            terrain_id=int(_safe_i32(data, group_offset + 0x24)),
            unique_id=int(_safe_i32(data, group_offset + 0x28)),
            spline_id=int(_safe_i32(data, group_offset + 0x2C)),
            group_origin=tuple(float(v) for v in global_offset),
        )
        group.material_entry_offsets = self._read_psp_materials(reader, group, material_list, sorted_material_list)
        self._read_psp_octree_strips(reader, level, group, octree, mesh_or_collision)
        return group

    def _read_psp_bgobject_primitives(self, reader: _RawBinaryReader, primitive_table: int, geometry_start: int) -> List[_PSPBGPrimitive]:
        data = reader.data
        primitives: List[_PSPBGPrimitive] = []
        if primitive_table <= 0 or not _in_range(data, primitive_table, _PSP_BGOBJECT_EXTRA_PRIMITIVE_SIZE):
            return primitives

        # PSP BGObject primitive tables are a sequence of uniform 0x30-byte
        # draw records followed by a small footer/terminator block.  The first
        # v7 pass treated the first record as a 0x6C PC-like header and then
        # read later records starting mid-entry; that mixed vertex counts and
        # data pointers from different records, producing exploded BGObject
        # meshes.
        table_end = int(geometry_start) if geometry_start > primitive_table else int(primitive_table) + _PSP_BGOBJECT_EXTRA_PRIMITIVE_SIZE
        table_end = min(max(0, table_end), len(data))
        current = int(primitive_table)
        primitive_index = 0
        while current + _PSP_BGOBJECT_EXTRA_PRIMITIVE_SIZE <= table_end:
            primitive = _read_psp_bgobject_primitive_entry(data, current, primitive_index)
            if primitive is None:
                break
            primitives.append(primitive)
            primitive_index += 1
            current += _PSP_BGOBJECT_EXTRA_PRIMITIVE_SIZE
        return primitives

    def _estimate_psp_bgobject_vertex_stride(self, data: bytearray | bytes, primitives: List[_PSPBGPrimitive], primitive_index: int, header_size: int) -> int:
        primitive = primitives[primitive_index]
        primitive_offset = int(primitive.data_offset)
        default_stride = _psp_bgobject_default_vertex_stride(primitive.mode)
        candidates = _psp_bgobject_stride_candidates(primitive.mode)
        preferred_stride = int(default_stride)

        first_vertex_offset = primitive_offset + max(0, int(header_size))
        first_vertex_prefix = bytes(data[first_vertex_offset:first_vertex_offset + 4]) if _in_range(data, first_vertex_offset, 1) else b''
        mode = int(primitive.mode) & 0xFF
        if mode == 1:
            preferred_stride = 10
            candidates = [10, 12, 14, 16]
        elif mode == 2:
            preferred_stride = 12
            candidates = [12, 10, 14, 16]

        candidate_limits: List[int] = []
        for other in primitives[primitive_index + 1:]:
            other_offset = int(other.data_offset)
            if other_offset > primitive_offset:
                candidate_limits.append(other_offset)
        stream_limit = min(candidate_limits) if candidate_limits else len(data)
        stream_span = max(0, int(stream_limit) - int(primitive_offset))
        if stream_span <= 0 or primitive.vertex_count <= 0:
            return int(preferred_stride)

        header_size = max(0, int(header_size))
        preferred_required = header_size + (int(primitive.vertex_count) * int(preferred_stride))
        if preferred_required <= stream_span:
            return int(preferred_stride)

        best_stride = int(preferred_stride)
        best_score = None
        for stride in candidates:
            required = header_size + (int(primitive.vertex_count) * int(stride))
            if required > stream_span:
                continue
            aligned_required = (required + 15) & ~15
            if aligned_required <= stream_span:
                padding_after_aligned_stream = stream_span - aligned_required
                misalignment = 0
            else:
                padding_after_aligned_stream = stream_span - required
                misalignment = abs((stream_span - required) % 16)
            score = (
                0 if int(stride) == int(preferred_stride) else 1,
                padding_after_aligned_stream,
                misalignment,
                abs(int(stride) - int(preferred_stride)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_stride = int(stride)
        return int(best_stride)

    def _emit_psp_bgobject_primitive(self, reader: _RawBinaryReader, bg_object: BGObject, primitives: List[_PSPBGPrimitive], primitive_index: int) -> None:
        data = reader.data
        primitive = primitives[primitive_index]
        data_offset = int(primitive.data_offset)
        if data_offset <= 0 or not _in_range(data, data_offset, 8):
            return

        header_size = 8 if _looks_like_psp_direct_stream_header(data, data_offset) else 0
        stride = self._estimate_psp_bgobject_vertex_stride(data, primitives, primitive_index, header_size)
        if stride <= 0:
            return
        vertex_start = data_offset + header_size
        safe_count = min(int(primitive.vertex_count), max(0, (len(data) - vertex_start) // int(stride)))
        if safe_count < 3:
            return

        sequence: List[int] = []
        for _vertex_index in range(int(safe_count)):
            blob_offset = vertex_start + (_vertex_index * int(stride))
            blob = bytes(data[blob_offset:blob_offset + int(stride)])
            if len(blob) < 6:
                continue

            # PSP BGObject streams follow the model-style GE coordinate
            # order, not the PSP terrain layout.  The compact 10-byte BGObject
            # record is uv8x2, aux/color16, xyz16x3.  v8 decoded the first four
            # bytes as signed uv16 fixed-point values, which produced wildly
            # oversized UV islands.
            raw_x, raw_y, raw_z = struct.unpack_from('<3h', blob, len(blob) - 6)
            uv_offset = _psp_uv_offset(int(primitive.mode), len(blob))
            if uv_offset + 2 <= len(blob):
                u = float(int(blob[uv_offset]) & 0xFF) / 254.0
                v = float(int(blob[uv_offset + 1]) & 0xFF) / 254.0
            else:
                u = v = 0.0

            source_index = len(bg_object.vertices)
            bg_object.vertices.append((float(raw_x), float(raw_y), float(raw_z)))
            bg_object.uvs.append((float(u), float(v)))

            color = None
            if len(blob) == 10:
                try:
                    color = _decode_psp_level_rgb565_vertex_color(struct.unpack_from('<H', blob, 2)[0])
                except Exception:
                    color = None
            if color is None:
                color = _read_psp_vertex_color(blob) or (255, 255, 255, 255)
            bg_object.vertex_colors.append(tuple(max(0, min(255, int(value))) for value in color))
            sequence.append(int(source_index))

        triangles = _psp_bgobject_strip_to_triangles(sequence, bg_object.vertices)
        if not triangles:
            return

        strip = LevelStrip(
            material_index=int(primitive.index),
            tpageid=_canonicalize_psp_bg_tpageid(primitive.texture_id, primitive.blend),
            flags=int(primitive.vertex_format_flags) & 0xFFFFFFFF,
            vertex_base_offset=0,
            indices=triangles,
            raw_count=int(primitive.vertex_count),
            sort_push=0,
            scroll_offset=0.0,
        )
        # Preserve PSP material state for round-tripping/debugging without making
        # PC terrain flags drive the topology path.
        try:
            strip.psp_mode_word = int(primitive.mode_word)  # type: ignore[attr-defined]
            strip.psp_vertex_mode = int(primitive.mode)  # type: ignore[attr-defined]
            strip.psp_texture_id = int(primitive.texture_id)  # type: ignore[attr-defined]
            strip.psp_blend = int(primitive.blend)  # type: ignore[attr-defined]
            strip.psp_vertex_format_flags = int(primitive.vertex_format_flags) & 0xFFFFFFFF  # type: ignore[attr-defined]
        except Exception:
            pass
        bg_object.strips.append(strip)

    def _read_psp_bg_objects(self, reader: _RawBinaryReader, list_offset: int, count: int) -> List[BGObject]:
        data = reader.data
        bg_objects: List[BGObject] = []
        if list_offset <= 0 or count <= 0:
            return bg_objects
        if count < 0 or count > 100000:
            logger.warning('Skipping PSP BGObjects due to suspicious count %d', count)
            return bg_objects
        if not _in_range(data, list_offset, int(count) * _PSP_BGOBJECT_SIZE):
            logger.warning('Skipping PSP BGObjects at invalid table 0x%X count=%d', list_offset, count)
            return bg_objects

        for bg_index in range(int(count)):
            entry_offset = int(list_offset) + (bg_index * _PSP_BGOBJECT_SIZE)
            if not _in_range(data, entry_offset, _PSP_BGOBJECT_SIZE):
                continue
            try:
                scale = tuple(float(v) for v in struct.unpack_from('<3f', data, entry_offset + 0x00))
            except Exception:
                scale = (1.0, 1.0, 1.0)
            if not _is_finite_triplet(scale):
                scale = (1.0, 1.0, 1.0)

            position_candidates: List[Tuple[int, Tuple[float, float, float]]] = []
            try:
                candidate_raw = tuple(float(v) for v in struct.unpack_from('<3f', data, entry_offset + 0x10))
                if _is_finite_triplet(candidate_raw):
                    position_candidates.append((0x10, _convert_bgobject_translation(candidate_raw[0], candidate_raw[1], candidate_raw[2])))
            except Exception:
                pass

            flags = int(_safe_i32(data, entry_offset + 0x20))
            primitive_table = int(_safe_u32(data, entry_offset + 0x30))
            geometry_start = int(_safe_u32(data, entry_offset + 0x44))
            cdc_render_data_id = int(_safe_u32(data, entry_offset + 0x58))

            bg_object = BGObject(
                index=int(bg_index),
                scale=tuple(float(v) for v in scale),
                position=_pick_bgobject_position(tuple(float(v) for v in scale), position_candidates),
                stride=_PSP_BGOBJECT_SIZE,
                flags=int(flags),
                cdc_render_data_id=int(cdc_render_data_id),
                position_candidates=position_candidates,
            )

            primitives = self._read_psp_bgobject_primitives(reader, primitive_table, geometry_start)
            for primitive_index in range(len(primitives)):
                self._emit_psp_bgobject_primitive(reader, bg_object, primitives, primitive_index)
            if not _bgobject_has_render_data(bg_object):
                continue
            bg_object.env_mapped_vertices = []
            bg_object.eye_ref_env_mapped_vertices = []
            _apply_bgobject_reflection_flags(bg_object)
            bg_objects.append(bg_object)
        return bg_objects

    def _read_psp_bg_instances(self, reader: _RawBinaryReader, list_offset: int, count: int, bg_object_list: int, bg_object_count: int) -> List[BGInstance]:
        data = reader.data
        instances: List[BGInstance] = []
        if list_offset <= 0 or count <= 0:
            return instances
        if count < 0 or count > 100000:
            logger.warning('Skipping PSP BGInstances due to suspicious count %d', count)
            return instances
        if not _in_range(data, list_offset, int(count) * _PSP_BGINSTANCE_SIZE):
            logger.warning('Skipping PSP BGInstances at invalid table 0x%X count=%d', list_offset, count)
            return instances

        for instance_index in range(int(count)):
            entry_offset = int(list_offset) + (instance_index * _PSP_BGINSTANCE_SIZE)
            if not _in_range(data, entry_offset, _PSP_BGINSTANCE_SIZE):
                continue
            try:
                matrix_values = struct.unpack_from('<16f', data, entry_offset)
                matrix_rows = (
                    tuple(float(v) for v in matrix_values[0:4]),
                    tuple(float(v) for v in matrix_values[4:8]),
                    tuple(float(v) for v in matrix_values[8:12]),
                    tuple(float(v) for v in matrix_values[12:16]),
                )
            except Exception:
                matrix_rows = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))

            bg_object_offset = int(_safe_u32(data, entry_offset + 0xC0))
            if bg_object_offset >= int(bg_object_list) and _PSP_BGOBJECT_SIZE > 0:
                bg_object_index = (bg_object_offset - int(bg_object_list)) // _PSP_BGOBJECT_SIZE
            else:
                bg_object_index = -1
            if bg_object_index < 0 or bg_object_index >= max(0, int(bg_object_count)):
                continue

            flags = int(_safe_u32(data, entry_offset + 0xC4))
            multi_spline_offset = int(_safe_u32(data, entry_offset + 0xC8))
            original_radius = float(_safe_f32(data, entry_offset + 0xCC))
            radius = float(_safe_f32(data, entry_offset + 0xD0))
            instance_id = int(_safe_u16(data, entry_offset + 0xD4))
            bg_flags = int(_safe_u16(data, entry_offset + 0xD6))
            target_frame = int(struct.unpack_from('<h', data, entry_offset + 0xD8)[0]) if _in_range(data, entry_offset + 0xD8, 2) else 0
            clip_beg = int(struct.unpack_from('<h', data, entry_offset + 0xDA)[0]) if _in_range(data, entry_offset + 0xDA, 2) else 0
            clip_end = int(struct.unpack_from('<h', data, entry_offset + 0xDC)[0]) if _in_range(data, entry_offset + 0xDC, 2) else 0
            link_seg = int(struct.unpack_from('<h', data, entry_offset + 0xDE)[0]) if _in_range(data, entry_offset + 0xDE, 2) else 0
            link_instance_offset = int(_safe_u32(data, entry_offset + 0xE0))
            color_data_index = int(struct.unpack_from('<h', data, entry_offset + 0xE4)[0]) if _in_range(data, entry_offset + 0xE4, 2) else 0
            lod = int(struct.unpack_from('<h', data, entry_offset + 0xE6)[0]) if _in_range(data, entry_offset + 0xE6, 2) else 0
            active_light_bitfield = int(_safe_u32(data, entry_offset + 0xEC))

            instances.append(BGInstance(
                index=int(instance_index),
                bg_object_index=int(bg_object_index),
                bg_object_offset=int(bg_object_offset),
                instance_id=int(instance_id),
                flags=int(flags),
                multi_spline_offset=int(multi_spline_offset),
                original_radius=float(original_radius),
                radius=float(radius),
                bg_flags=int(bg_flags),
                target_frame=int(target_frame),
                clip_beg=int(clip_beg),
                clip_end=int(clip_end),
                link_seg=int(link_seg),
                link_instance_offset=int(link_instance_offset),
                color_data_index=int(color_data_index),
                lod=int(lod),
                active_light_bitfield=int(active_light_bitfield),
                matrix_rows=matrix_rows,
            ))
        return instances

    def _read_psp_materials(self, reader: _RawBinaryReader, group: TerrainGroup, material_list: int, sorted_material_list: int) -> List[int]:
        data = reader.data
        if material_list <= 0 or not _in_range(data, material_list, 4):
            group.strips.append(LevelStrip(material_index=0, tpageid=0, flags=0, vertex_base_offset=0))
            return [0]

        count = _safe_i32(data, material_list)
        if count <= 0:
            # Some PSP terrain groups intentionally point at an empty shared list.
            # Keep a fallback material slot so geometry from the octree chain can still import.
            group.strips.append(LevelStrip(material_index=0, tpageid=0, flags=0, vertex_base_offset=0))
            return [0]
        if count > 4096 or not _in_range(data, material_list + 4, int(count) * _PSP_TERRAIN_MATERIAL_SIZE):
            logger.warning('Invalid PSP material list for group %d at 0x%X count=%d', group.index, material_list, count)
            group.strips.append(LevelStrip(material_index=0, tpageid=0, flags=0, vertex_base_offset=0))
            return [0]

        entry_offsets: List[int] = []
        for material_index in range(int(count)):
            entry_offset = int(material_list) + 4 + (material_index * _PSP_TERRAIN_MATERIAL_SIZE)
            tpageid, flags, vertex_base_offset, _strip_data = struct.unpack_from('<IIII', data, entry_offset)
            entry_offsets.append(int(entry_offset))
            group.strips.append(LevelStrip(
                material_index=int(material_index),
                tpageid=int(tpageid) & 0xFFFFFFFF,
                flags=int(flags) & 0xFFFFFFFF,
                vertex_base_offset=int(vertex_base_offset) & 0xFFFFFFFF,
            ))

        sorted_count = _safe_i32(data, sorted_material_list) if _in_range(data, sorted_material_list, 4) else 0
        if 0 < sorted_count <= count and _in_range(data, sorted_material_list + 4, sorted_count * _PSP_TERRAIN_MATERIAL_SIZE):
            sorted_indices: List[int] = []
            for sorted_i in range(int(sorted_count)):
                sorted_entry = int(sorted_material_list) + 4 + sorted_i * _PSP_TERRAIN_MATERIAL_SIZE
                raw_tpage = _safe_u32(data, sorted_entry)
                for candidate, strip in enumerate(group.strips):
                    if int(strip.tpageid) == int(raw_tpage) and candidate not in sorted_indices:
                        sorted_indices.append(candidate)
                        break
            group.sorted_material_indices = sorted_indices
        return entry_offsets

    def _read_psp_octree_strips(self, reader: _RawBinaryReader, level: LevelData, group: TerrainGroup, octree_offset: int, fallback_strip_offset: int = 0) -> None:
        data = reader.data
        candidates: List[int] = []
        if _in_range(data, octree_offset, 0x18):
            seen_nodes: set[int] = set()
            stack = [int(octree_offset)]
            while stack and len(seen_nodes) < _PSP_MAX_OCTREE_NODES:
                node = stack.pop()
                if node <= 0 or node in seen_nodes or not _in_range(data, node, 0x18):
                    continue
                seen_nodes.add(node)
                first_strip = _safe_u32(data, node + 0x10)
                if first_strip > 0 and _looks_like_psp_strip_header(data, first_strip, len(group.strips)):
                    candidates.append(int(first_strip))
                child_count = _safe_i32(data, node + 0x14)
                if 0 <= child_count <= 8 and _in_range(data, node + 0x18, child_count * 4):
                    for child_i in range(int(child_count)):
                        child = _safe_u32(data, node + 0x18 + child_i * 4)
                        if child > 0 and child not in seen_nodes:
                            stack.append(int(child))
        elif _in_range(data, fallback_strip_offset, 0x34):
            candidates.append(int(fallback_strip_offset))

        seen_strips: set[int] = set()
        for strip_offset in candidates:
            current = int(strip_offset)
            chain_guard = 0
            while current > 0 and _in_range(data, current, 0x34) and current not in seen_strips:
                if not _looks_like_psp_strip_header(data, current, len(group.strips)):
                    break
                seen_strips.add(current)
                chain_guard += 1
                if chain_guard > _PSP_MAX_STRIP_CHAIN:
                    logger.warning('Aborting PSP strip chain for group %d at 0x%X due to excessive length', group.index, strip_offset)
                    break
                next_offset = self._emit_psp_strip(reader, level, group, current)
                if next_offset <= 0 or next_offset == current:
                    break
                current = int(next_offset)

    @staticmethod
    def _psp_stride_candidates(_mode: int = 0) -> List[int]:
        # Retail PSP level terrain samples use a 0x38-byte strip header followed
        # by 16-byte vertices.  Some strip headers have non-zero upper bits in
        # the first word and leave auxiliary/padding bytes before the next strip;
        # choosing the stride from the next-strip span turns those records into
        # 18/20/22-byte vertices and corrupts positions/faces.  Keep the wider
        # candidates for recovery, but strongly prefer the observed 16-byte path.
        return [16, 18, 20, 22, 14, 24, 12]

    def _choose_psp_strip_layout(self, data: bytearray, strip_offset: int, vertex_count: int, next_offset: int) -> Tuple[int, int]:
        data_len = len(data)
        if vertex_count > 0 and strip_offset + 0x38 + (int(vertex_count) * 16) <= data_len:
            return 0x38, 16

        candidates: List[Tuple[tuple, int, int]] = []
        span = int(next_offset) - int(strip_offset) if next_offset > strip_offset and next_offset <= data_len else 0
        for header_size in (0x38, 0x48, 0x30, 0x40, 0x34):
            for stride in self._psp_stride_candidates():
                required = int(header_size) + int(vertex_count) * int(stride)
                if required <= 0 or strip_offset + required > data_len:
                    continue
                if span > 0:
                    if required > span:
                        continue
                    padding = span - required
                    score = (0 if stride == 16 else 1, 0 if header_size == 0x38 else 1, padding, abs(stride - 16), header_size)
                else:
                    score = (0 if stride == 16 else 1, 0 if header_size == 0x38 else 1, abs(stride - 16), header_size, required)
                candidates.append((score, int(header_size), int(stride)))
        if not candidates:
            return 0x38, 16
        candidates.sort(key=lambda item: item[0])
        return int(candidates[0][1]), int(candidates[0][2])

    def _emit_psp_strip(self, reader: _RawBinaryReader, level: LevelData, group: TerrainGroup, strip_offset: int) -> int:
        data = reader.data
        if not _looks_like_psp_strip_header(data, strip_offset, len(group.strips)):
            return 0

        strip_vertex_count = int(_safe_u16(data, strip_offset + 0x00))
        list_vertex_count = int(_safe_u16(data, strip_offset + 0x02))
        material_count = max(1, len(group.strips))
        raw_material_index = _safe_i32(data, strip_offset + 0x18)
        material_index = int(raw_material_index) if 0 <= int(raw_material_index) < material_count else 0
        vertex_count = int(strip_vertex_count) + int(list_vertex_count)
        if vertex_count <= 0 or vertex_count > (_PSP_STRIP_MAX_VERTEX_COUNT * 2):
            return _safe_u32(data, strip_offset + 0x30)

        next_offset = _safe_u32(data, strip_offset + 0x30)
        if next_offset <= 0 or not _in_range(data, next_offset, 4):
            next_offset = 0

        header_size, stride = self._choose_psp_strip_layout(data, strip_offset, vertex_count, next_offset)
        data_offset = int(strip_offset) + int(header_size)
        safe_count = min(vertex_count, max(0, (len(data) - data_offset) // max(1, stride)))
        if next_offset > data_offset:
            safe_count = min(safe_count, max(0, (next_offset - data_offset) // max(1, stride)))
        if safe_count < 3:
            return next_offset

        sequence: List[int] = []
        strip_normals: dict[int, Tuple[float, float, float]] = {}
        vertex_records: List[Tuple[Tuple[float, float, float], Optional[Tuple[int, int]], Tuple[float, float], Tuple[float, float, float], Tuple[int, int, int, int]]] = []
        raw_u_values: List[int] = []
        raw_v_values: List[int] = []
        for vertex_index in range(int(safe_count)):
            blob_offset = data_offset + vertex_index * stride
            blob = bytes(data[blob_offset:blob_offset + stride])
            if len(blob) < 6:
                continue
            x, y, z = struct.unpack_from('<3h', blob, len(blob) - 6)
            position = _convert_level_position(float(x), float(y), float(z))

            raw_uv: Optional[Tuple[int, int]] = None
            fallback_uv = (0.0, 0.0)
            if len(blob) >= 16:
                # PSP level terrain uses a distinct compact 16-byte format:
                #   uv16x2, color16, normal8x3, pad, xyz16x3
                # The UV shorts are Q11 fixed-point texture coordinates.  Their
                # integer tile part is local to this draw-strip and should not
                # be kept as a PC-style 1/4096 coordinate or masked down to 12
                # bits; both make full-tile PSP faces import too small.
                raw_u = int(struct.unpack_from('<H', blob, 0)[0])
                raw_v = int(struct.unpack_from('<H', blob, 2)[0])
                raw_uv = (raw_u, raw_v)
                raw_u_values.append(raw_u)
                raw_v_values.append(raw_v)
                normal = _convert_psp_level_normal(_s8(blob[6]), _s8(blob[7]), _s8(blob[8]))
            else:
                mode = _psp_mode_from_blob_size(len(blob))
                uv_offset = _psp_uv_offset(mode, len(blob))
                if uv_offset + 2 <= len(blob):
                    fallback_uv = (float(blob[uv_offset]) / 255.0, float(blob[uv_offset + 1]) / 255.0)
                normal = (0.0, 0.0, 0.0)

            color = _read_psp_vertex_color(blob) or (255, 255, 255, 255)
            vertex_records.append((
                (float(position[0]), float(position[1]), float(position[2])),
                raw_uv,
                fallback_uv,
                normal,
                tuple(int(v) for v in color),
            ))

        u_tile_base = _psp_level_uv_tile_base(raw_u_values)
        v_tile_base = _psp_level_uv_tile_base(raw_v_values)
        for position, raw_uv, fallback_uv, normal, color in vertex_records:
            uv = _decode_psp_level_uv16(raw_uv[0], raw_uv[1], u_tile_base, v_tile_base) if raw_uv is not None else fallback_uv
            source_index = len(level.vertices)
            level.vertices.append(position)
            level.uvs.append((float(uv[0]), float(uv[1])))
            level.vertex_colors.append(color)
            strip_normals[int(source_index)] = normal
            sequence.append(source_index)

        # The two 16-bit count fields are different primitive batches packed
        # back-to-back: h0 is a triangle-strip vertex stream and h1 is an
        # independent triangle-list vertex stream.  h1 is almost always a
        # multiple of three and the record span is 0x38 + (h0 + h1) * 16.
        # Treating h1 as strip continuation creates long diagonal artifacts;
        # ignoring it leaves random holes.
        triangles: List[int] = []
        if strip_vertex_count > 0:
            triangles.extend(_strip_to_triangles(sequence[:strip_vertex_count], level, strip_normals))
        if list_vertex_count > 0:
            triangles.extend(_triangle_list_to_triangles(sequence[strip_vertex_count:strip_vertex_count + list_vertex_count], level, strip_normals))

        if triangles:
            try:
                group.strips[int(material_index)].indices.extend(triangles)
            except Exception:
                group.strips[0].indices.extend(triangles)
        return next_offset

    def _create_image_from_texture_section(self, reader: _RawBinaryReader, section_index: int, section: _RawSection) -> Optional[bpy.types.Image]:
        base = int(section.offset)
        size = int(section.size)
        if base <= 0 or size <= 0 or not _in_range(reader.data, base, size):
            return None
        payload = bytes(reader.data[base:base + size])
        decoded = decode_supported_psp_texture(payload)
        if not decoded or not decoded.get('supported'):
            return None

        image_name = f'{int(section_index)}_{int(section.id):x}'
        existing = bpy.data.images.get(image_name)
        if existing is not None:
            return existing

        image = bpy.data.images.new(
            image_name,
            width=int(decoded['width']),
            height=int(decoded['height']),
            alpha=bool(decoded.get('has_alpha', True)),
        )
        image.pixels[:] = decoded['pixels']
        try:
            image.pack()
        except Exception:
            pass
        return image
