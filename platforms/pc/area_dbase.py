from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


def _sign_extend(value: int, bits: int) -> int:
    value = int(value) & ((1 << int(bits)) - 1)
    sign_bit = 1 << (int(bits) - 1)
    return value - (1 << int(bits)) if value & sign_bit else value


def _read_c_string(data: bytes, offset: int, size: int) -> str:
    offset = max(0, int(offset))
    size = max(0, int(size))
    if offset >= len(data) or size <= 0:
        return ''
    chunk = bytes(data[offset:offset + size])
    nul = chunk.find(b'\x00')
    if nul >= 0:
        chunk = chunk[:nul]
    return chunk.decode('utf-8', errors='replace')


def _convert_level_position(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (-float(x), -float(y), float(z))


def convert_area_dbase_center_position(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (-float(x), -float(y), float(z))


def convert_area_dbase_center_from_blender_position(x: float, y: float, z: float) -> Tuple[int, int, int]:
    return (int(round(-float(x))), int(round(-float(y))), int(round(float(z))))




def _pack_signed(value: int, bits: int) -> int:
    return int(value) & ((1 << int(bits)) - 1)


def _clamp_i16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def _align4(value: int) -> int:
    return (int(value) + 3) & ~3


def convert_area_dbase_child_position_from_blender(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (-float(x), -float(y), float(z))


def _build_generated_kd_records(area_infos: list[dict]) -> bytes:
    if not area_infos:
        return b''

    def centroid(info: dict, axis: int) -> float:
        values = [float(v[axis]) for v in info.get('vertices', [])]
        return sum(values) / float(len(values) or 1)

    def bounds(infos: list[dict], axis: int) -> tuple[float, float]:
        vals = [float(v[axis]) for info in infos for v in info.get('vertices', [])]
        if not vals:
            return 0.0, 0.0
        return min(vals), max(vals)

    def build(infos: list[dict]) -> list[tuple[int, int, int, int, bool]]:
        if len(infos) <= 1:
            ref = int(infos[0].get('area_ref', 0)) if infos else 0
            return [(0, ref & 0xFFFF, 0, 0, True)]
        ranges = []
        for axis in range(3):
            mn, mx = bounds(infos, axis)
            ranges.append(mx - mn)
        axis = max(range(3), key=lambda item: ranges[item])
        ordered = sorted(infos, key=lambda info: centroid(info, axis))
        mid = max(1, min(len(ordered) - 1, len(ordered) // 2))
        neg_infos = ordered[:mid]
        pos_infos = ordered[mid:]
        neg_records = build(neg_infos)
        pos_records = build(pos_infos)
        pos_child_offset = 1 + len(neg_records)
        neg_min, neg_max = bounds(neg_infos, axis)
        pos_min, pos_max = bounds(pos_infos, axis)
        split = (neg_max + pos_min) * 0.5
        # Keep these conservative.  They are signed 16-bit offsets; exact values
        # are not needed for serialization consistency, only a valid tree shape.
        neg_offset = _clamp_i16(round((neg_max - split) / 8.0))
        pos_offset = _clamp_i16(round((pos_min - split) / 8.0))
        return [(pos_child_offset, int(axis), neg_offset, pos_offset, False)] + neg_records + pos_records

    out = bytearray()
    for a, b, c, d, is_leaf in build(list(area_infos)):
        if is_leaf:
            out.extend(struct.pack('<HHHH', int(a) & 0xFFFF, int(b) & 0xFFFF, int(c) & 0xFFFF, int(d) & 0xFFFF))
        else:
            out.extend(struct.pack('<HHhh', int(a) & 0xFFFF, int(b) & 0xFFFF, _clamp_i16(c), _clamp_i16(d)))
    return bytes(out)


def build_area_dbase_section_content_from_polygons(
    name: str,
    center: Tuple[int, int, int],
    polygons: list[list[Tuple[float, float, float]]],
    *,
    prefix_size: int = 0x10,
    version: int = 5,
    endian: str = '<',
) -> tuple[bytes, int]:
    clean_polygons: list[list[tuple[int, int, int]]] = []
    for polygon in polygons or []:
        verts: list[tuple[int, int, int]] = []
        for vertex in polygon or []:
            gx, gy, gz = convert_area_dbase_child_position_from_blender(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            verts.append((int(round(gx)), int(round(gy)), int(round(gz))))
        # Drop duplicate closing vertex if the mesh provides one.
        if len(verts) >= 2 and verts[0] == verts[-1]:
            verts = verts[:-1]
        # Remove adjacent duplicate vertices.
        deduped: list[tuple[int, int, int]] = []
        for vertex in verts:
            if not deduped or deduped[-1] != vertex:
                deduped.append(vertex)
        if len(deduped) >= 3:
            clean_polygons.append(deduped)

    if not clean_polygons:
        raise ValueError('AreaDBase export requires at least one polygon with three vertices')

    area_offsets: list[int] = []
    cursor = 0
    for poly in clean_polygons:
        if len(poly) > 32:
            raise ValueError('AreaDBase areas support at most 32 edges')
        area_offsets.append(cursor)
        cursor += AreaDBaseParser.AREA_SIZE + (len(poly) * AreaDBaseParser.EDGE_SIZE)
    area_block_bytes = cursor
    area_refs = [offset // 4 for offset in area_offsets]

    # Match edges by XY only.  Imported display surfaces may have stitched or
    # averaged Z, while the AreaDBase topology connects areas by horizontal edge.
    edge_owner_by_xy: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]] = {}
    for area_index, poly in enumerate(clean_polygons):
        for edge_index, p0 in enumerate(poly):
            p1 = poly[(edge_index + 1) % len(poly)]
            key = tuple(sorted(((int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])))))
            edge_owner_by_xy.setdefault(key, []).append((area_index, edge_index))

    connected_to: dict[tuple[int, int], tuple[int, int]] = {}
    for owners in edge_owner_by_xy.values():
        if len(owners) == 2:
            a, b = owners
            connected_to[a] = b
            connected_to[b] = a

    area_block = bytearray()
    area_infos: list[dict] = []
    for area_index, poly in enumerate(clean_polygons):
        xs = [v[0] for v in poly]
        ys = [v[1] for v in poly]
        zs = [v[2] for v in poly]
        pos_x = int(round((sum(xs) / float(len(xs))) / AreaDBaseParser.AREA_XY_WORLD_SCALE))
        pos_y = int(round((sum(ys) / float(len(ys))) / AreaDBaseParser.AREA_XY_WORLD_SCALE))
        pos_z = int(round((sum(zs) / float(len(zs))) / AreaDBaseParser.AREA_Z_WORLD_SCALE))
        packed_pos = (
            _pack_signed(pos_x, 10)
            | (_pack_signed(pos_y, 10) << 10)
            | (_pack_signed(pos_z, 12) << 20)
        )
        flags = int(len(poly)) & 0x3F
        search = 0xFFFF
        area_block.extend(struct.pack(f'{endian}IIHH', int(packed_pos) & 0xFFFFFFFF, 0, flags, search))
        area_infos.append({'area_ref': area_refs[area_index], 'vertices': poly})
        for edge_index, vertex in enumerate(poly):
            sx = int(round((int(vertex[0]) - (pos_x * AreaDBaseParser.AREA_XY_WORLD_SCALE)) / AreaDBaseParser.EDGE_START_WORLD_SCALE))
            sy = int(round((int(vertex[1]) - (pos_y * AreaDBaseParser.AREA_XY_WORLD_SCALE)) / AreaDBaseParser.EDGE_START_WORLD_SCALE))
            if not (-4096 <= sx <= 4095 and -4096 <= sy <= 4095):
                raise ValueError('AreaDBase vertex offset exceeds signed 13-bit AEdge range')
            other = connected_to.get((area_index, edge_index))
            if other is None:
                edge_type = 1  # BLOCKED_EDGE
                adj_area_ref = -1
            else:
                edge_type = 0  # CONNECTED_EDGE
                adj_area_ref = int(area_refs[int(other[0])])
            raw_x = _pack_signed(sx, 13)
            raw_y = _pack_signed(sy, 13) | ((int(edge_type) & 0x7) << 13)
            area_block.extend(struct.pack(f'{endian}HHhH', raw_x, raw_y, int(adj_area_ref), 0))

    tree_bytes_blob = _build_generated_kd_records(area_infos)
    area_block_offset = AreaDBaseParser.HEADER_SIZE
    tree_offset = _align4(area_block_offset + len(area_block))
    dyn_ob_heap_offset = _align4(tree_offset + len(tree_bytes_blob))
    dyn_ob_heap_bytes = 0
    gateway_offset = dyn_ob_heap_offset + dyn_ob_heap_bytes
    num_gateways = 0
    num_bytes = gateway_offset

    all_vertices = [v for poly in clean_polygons for v in poly]
    min_x = min(v[0] for v in all_vertices)
    min_y = min(v[1] for v in all_vertices)
    min_z = min(v[2] for v in all_vertices)
    max_x = max(v[0] for v in all_vertices)
    max_y = max(v[1] for v in all_vertices)
    max_z = max(v[2] for v in all_vertices)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1
    if min_z == max_z:
        max_z += 1

    header = bytearray(AreaDBaseParser.HEADER_SIZE)
    struct.pack_into(f'{endian}ii', header, 0x00, int(version), int(num_bytes))
    encoded_name = str(name or 'AreaDBase').encode('ascii', errors='ignore')[:23]
    header[0x08:0x08 + len(encoded_name)] = encoded_name
    cx, cy, cz = (int(center[0]), int(center[1]), int(center[2]))
    struct.pack_into(f'{endian}iii', header, 0x20, cx, cy, cz)
    struct.pack_into(f'{endian}iii', header, 0x2C, int(min_x), int(min_y), int(min_z))
    struct.pack_into(f'{endian}iii', header, 0x38, int(max_x), int(max_y), int(max_z))
    struct.pack_into(
        f'{endian}IIIIIIII',
        header,
        0x44,
        int(area_block_offset),
        int(len(area_block)),
        int(tree_offset),
        int(len(tree_bytes_blob)),
        int(dyn_ob_heap_offset),
        int(dyn_ob_heap_bytes),
        int(gateway_offset),
        int(num_gateways),
    )

    body = bytearray(header)
    body.extend(area_block)
    while len(body) < tree_offset:
        body.append(0)
    body.extend(tree_bytes_blob)
    while len(body) < num_bytes:
        body.append(0)

    return (bytes(bytearray(int(prefix_size)) + body), int(prefix_size))

def patch_area_dbase_section_blob_center(blob: bytes | bytearray, center: Tuple[int, int, int], *, endian: str = '<') -> bytes:
    data = bytearray(blob or b'')
    info_size, _section_type, _version_id, _section_id, _size, _spec_mask, _num_relocations = AreaDBaseParser._parse_section_header(data, endian)
    header = AreaDBaseParser.find_header_offset(data, payload_base=info_size, endian=endian)
    if header is None:
        raise ValueError('section blob does not contain an AreaDBase header')
    cx, cy, cz = (int(center[0]), int(center[1]), int(center[2]))
    struct.pack_into(f'{endian}iii', data, int(header) + 0x20, cx, cy, cz)
    return bytes(data)


def area_dbase_content_header_offset(blob: bytes | bytearray, *, endian: str = '<') -> Optional[int]:
    data = bytes(blob or b'')
    info_size, _section_type, _version_id, _section_id, _size, _spec_mask, _num_relocations = AreaDBaseParser._parse_section_header(data, endian)
    header = AreaDBaseParser.find_header_offset(data, payload_base=info_size, endian=endian)
    if header is None:
        return None
    return int(header) - int(info_size)


@dataclass(slots=True)
class AreaDBasePortal:
    area_index: int
    portal_index: int
    src_x: int
    src_y: int
    src_z: int
    last_portal: int
    usable: int
    adj_area_ref: int
    user_data: int
    portal_id: int
    sibling_portal_index: int
    raw_src_x: int
    raw_src_y: int
    raw_src_z: int
    raw_portal_id: int
    game_position: Tuple[float, float, float]
    blender_position: Tuple[float, float, float]


@dataclass(slots=True)
class AreaDBaseEdge:
    area_index: int
    edge_index: int
    start_x: int
    start_y: int
    edge_type: int
    adj_area_ref: int
    pad: int
    raw_start_x: int
    raw_start_y: int
    game_position: Tuple[float, float, float]
    blender_position: Tuple[float, float, float]


@dataclass(slots=True)
class AreaDBaseArea:
    index: int
    block_offset: int
    pos_x: int
    pos_y: int
    pos_z: int
    dyn_areas_raw: int
    num_edges: int
    has_portals: int
    island: int
    flood_island: int
    search_cost: int
    in_open_list: int
    edges: List[AreaDBaseEdge] = field(default_factory=list)
    portals: List[AreaDBasePortal] = field(default_factory=list)

    @property
    def area_ref(self) -> int:
        # cdc::AreaRef stores an area-block byte offset divided by four.
        return int(self.block_offset) // 4


@dataclass(slots=True)
class AreaDBaseKDRecord:
    index: int
    pos_child_offset_or_zero: int
    axis_or_prim: int
    neg_offset_or_pad: int
    pos_offset_or_pad2: int


@dataclass(slots=True)
class AreaDBaseGateway:
    index: int
    name: str
    dest_unit_name: str
    sibling_gateway_name: str
    pos_x: int
    pos_y: int
    pos_z: int
    num_links: int
    in_open_list: int
    static_area_ref: int
    pad: int
    search_cost: float
    game_position: Tuple[float, float, float]
    blender_position: Tuple[float, float, float]


@dataclass(slots=True)
class AreaDBaseData:
    filepath: str
    section_id: int = 0
    section_type: int = 0
    section_version_id: int = 0
    section_spec_mask: int = 0xFFFFFFFF
    section_info_size: int = 0
    header_offset: int = 0
    version: int = 0
    num_bytes: int = 0
    name: str = ''
    center: Tuple[int, int, int] = (0, 0, 0)
    box_min: Tuple[int, int, int] = (0, 0, 0)
    box_max: Tuple[int, int, int] = (0, 0, 0)
    area_block_offset: int = 0
    area_block_bytes: int = 0
    tree_offset: int = 0
    tree_bytes: int = 0
    dyn_ob_heap_offset: int = 0
    dyn_ob_heap_bytes: int = 0
    gateway_offset: int = 0
    num_gateways: int = 0
    areas: List[AreaDBaseArea] = field(default_factory=list)
    kd_records: List[AreaDBaseKDRecord] = field(default_factory=list)
    gateways: List[AreaDBaseGateway] = field(default_factory=list)
    raw_data: bytes = b''

    @property
    def edge_count(self) -> int:
        return sum(len(area.edges) for area in self.areas)

    @property
    def portal_count(self) -> int:
        return sum(len(area.portals) for area in self.areas)

    @property
    def game_bbox(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return self.box_min, self.box_max

    @property
    def blender_bbox(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        a = _convert_level_position(*self.box_min)
        b = _convert_level_position(*self.box_max)
        return (
            tuple(min(a[i], b[i]) for i in range(3)),
            tuple(max(a[i], b[i]) for i in range(3)),
        )


class AreaDBaseParser:

    HEADER_SIZE = 0x64
    AREA_SIZE = 12
    EDGE_SIZE = 8
    PORTAL_SIZE = 12
    AREA_XY_WORLD_SCALE = 128
    AREA_Z_WORLD_SCALE = 32
    EDGE_START_WORLD_SCALE = 8
    GATEWAY_SIZE = 0x88
    SECTION_HEADER_SIZE = 24

    def __init__(self, filepath: str | Path, *, endian: str = '<'):
        self.filepath = Path(filepath)
        self.endian = endian

    @staticmethod
    def _parse_section_header(data: bytes, endian: str = '<') -> tuple[int, int, int, int, int, int, int]:
        if len(data) < AreaDBaseParser.SECTION_HEADER_SIZE or bytes(data[:4]) not in {b'SECT', b'TCES'}:
            return 0, 0, 0, 0, 0, 0xFFFFFFFF, 0
        size, section_type, _skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from(f'{endian}iBBHIII', data, 4)
        num_relocations = max(0, int((packed_data >> 8) & 0x00FFFFFF))
        info_size = AreaDBaseParser.SECTION_HEADER_SIZE + (num_relocations * 8)
        return int(info_size), int(section_type), int(version_id), int(section_id), int(size), int(spec_mask), int(num_relocations)

    @classmethod
    def _header_candidate_valid(cls, data: bytes, base: int, endian: str = '<') -> bool:
        if base < 0 or base + cls.HEADER_SIZE > len(data):
            return False
        try:
            version, num_bytes = struct.unpack_from(f'{endian}ii', data, base)
            area_off, area_size, tree_off, tree_size, dyn_off, dyn_size, gateway_off, num_gateways = struct.unpack_from(f'{endian}IIIIIIII', data, base + 0x44)
        except struct.error:
            return False
        if not (1 <= int(version) <= 32):
            return False
        if int(num_bytes) < cls.HEADER_SIZE or base + int(num_bytes) > len(data):
            return False
        if int(area_off) < cls.HEADER_SIZE or int(area_size) <= 0:
            return False
        if base + int(area_off) + int(area_size) > len(data):
            return False
        if int(area_size) % 4 != 0:
            return False
        for off, size in ((tree_off, tree_size), (dyn_off, dyn_size)):
            if int(size) and (int(off) <= 0 or base + int(off) + int(size) > len(data)):
                return False
        if int(num_gateways) > 0 and (int(gateway_off) <= 0 or base + int(gateway_off) > len(data)):
            return False
        name = _read_c_string(data, base + 8, 0x18)
        if not name:
            return False
        return True

    @classmethod
    def find_header_offset(cls, data: bytes, *, payload_base: int = 0, endian: str = '<') -> Optional[int]:
        # Extracted TRA/TRL AreaDBase sections have a 0x20-byte section-local
        # prefix before cdc::AreaDBaseHeader.
        candidates = [int(payload_base) + 0x20, int(payload_base) + 0x10, int(payload_base), 0x38, 0x28, 0x20, 0x10, 0]
        seen: set[int] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if cls._header_candidate_valid(data, candidate, endian=endian):
                return candidate
        return None

    @classmethod
    def looks_like_area_dbase(cls, filepath: str | Path, *, endian: str = '<') -> bool:
        try:
            data = Path(filepath).read_bytes()
        except Exception:
            return False
        info_size, _section_type, _version_id, _section_id, _size, _spec_mask, _num_relocations = cls._parse_section_header(data, endian)
        return cls.find_header_offset(data, payload_base=info_size, endian=endian) is not None

    @classmethod
    def parse_blob(cls, data: bytes | bytearray, *, filepath: str | Path = '<memory>', endian: str = '<') -> AreaDBaseData:
        parser = cls(filepath, endian=endian)
        return parser._parse_data(bytes(data or b''))

    def parse(self) -> AreaDBaseData:
        return self._parse_data(self.filepath.read_bytes())

    def _parse_data(self, data: bytes) -> AreaDBaseData:
        info_size, section_type, version_id, section_id, _section_size, spec_mask, _num_relocations = self._parse_section_header(data, self.endian)
        header = self.find_header_offset(data, payload_base=info_size, endian=self.endian)
        if header is None:
            raise ValueError(f'{self.filepath.name} does not look like a cdc::AreaDBase section')

        version, num_bytes = struct.unpack_from(f'{self.endian}ii', data, header)
        name = _read_c_string(data, header + 8, 0x18)
        center = tuple(int(v) for v in struct.unpack_from(f'{self.endian}iii', data, header + 0x20))
        box_min = tuple(int(v) for v in struct.unpack_from(f'{self.endian}iii', data, header + 0x2C))
        box_max = tuple(int(v) for v in struct.unpack_from(f'{self.endian}iii', data, header + 0x38))
        area_off, area_size, tree_off, tree_size, dyn_off, dyn_size, gateway_off, num_gateways = struct.unpack_from(f'{self.endian}IIIIIIII', data, header + 0x44)

        parsed = AreaDBaseData(
            filepath=str(self.filepath),
            section_id=int(section_id),
            section_type=int(section_type),
            section_version_id=int(version_id),
            section_spec_mask=int(spec_mask),
            section_info_size=int(info_size),
            header_offset=int(header),
            version=int(version),
            num_bytes=int(num_bytes),
            name=name,
            center=center,
            box_min=box_min,
            box_max=box_max,
            area_block_offset=int(area_off),
            area_block_bytes=int(area_size),
            tree_offset=int(tree_off),
            tree_bytes=int(tree_size),
            dyn_ob_heap_offset=int(dyn_off),
            dyn_ob_heap_bytes=int(dyn_size),
            gateway_offset=int(gateway_off),
            num_gateways=int(num_gateways),
            raw_data=data,
        )
        self._parse_areas(data, header, parsed)
        self._parse_kd_records(data, header, parsed)
        self._parse_gateways(data, header, parsed)
        return parsed

    def _parse_areas(self, data: bytes, header: int, parsed: AreaDBaseData) -> None:
        area_start = int(header) + int(parsed.area_block_offset)
        area_end = area_start + int(parsed.area_block_bytes)
        cursor = area_start
        area_index = 0
        while cursor < area_end:
            if cursor + self.AREA_SIZE > area_end:
                raise ValueError(f'Truncated cdc::Area at 0x{cursor:X}')
            block_offset = cursor - area_start
            packed_pos, dyn_areas_raw, flags, search = struct.unpack_from(f'{self.endian}IIHH', data, cursor)
            num_edges = int(flags) & 0x3F
            has_portals = (int(flags) >> 6) & 0x1
            island = (int(flags) >> 7) & 0xFF
            flood_island = (int(flags) >> 15) & 0x1
            search_cost = int(search) & 0x7FFF
            in_open_list = (int(search) >> 15) & 0x1
            if num_edges <= 0 or num_edges > 32:
                raise ValueError(f'Invalid cdc::Area edge count {num_edges} at 0x{cursor:X}')
            cursor += self.AREA_SIZE
            if cursor + (num_edges * self.EDGE_SIZE) > area_end:
                raise ValueError(f'Truncated cdc::AEdge array for area {area_index}')

            pos_x = _sign_extend(int(packed_pos) & 0x3FF, 10)
            pos_y = _sign_extend((int(packed_pos) >> 10) & 0x3FF, 10)
            pos_z = _sign_extend((int(packed_pos) >> 20) & 0xFFF, 12)
            area = AreaDBaseArea(
                index=int(area_index),
                block_offset=int(block_offset),
                pos_x=int(pos_x),
                pos_y=int(pos_y),
                pos_z=int(pos_z),
                dyn_areas_raw=int(dyn_areas_raw),
                num_edges=int(num_edges),
                has_portals=int(has_portals),
                island=int(island),
                flood_island=int(flood_island),
                search_cost=int(search_cost),
                in_open_list=int(in_open_list),
            )
            for edge_index in range(num_edges):
                raw_x, raw_y, adj_area_ref, pad = struct.unpack_from(f'{self.endian}HHhH', data, cursor)
                start_x = _sign_extend(int(raw_x) & 0x1FFF, 13)
                start_y = _sign_extend(int(raw_y) & 0x1FFF, 13)
                edge_type = (int(raw_y) >> 13) & 0x7
                game_pos = (
                    float((int(pos_x) * self.AREA_XY_WORLD_SCALE) + (int(start_x) * self.EDGE_START_WORLD_SCALE)),
                    float((int(pos_y) * self.AREA_XY_WORLD_SCALE) + (int(start_y) * self.EDGE_START_WORLD_SCALE)),
                    float(int(pos_z) * self.AREA_Z_WORLD_SCALE),
                )
                area.edges.append(AreaDBaseEdge(
                    area_index=int(area_index),
                    edge_index=int(edge_index),
                    start_x=int(start_x),
                    start_y=int(start_y),
                    edge_type=int(edge_type),
                    adj_area_ref=int(adj_area_ref),
                    pad=int(pad),
                    raw_start_x=int(raw_x),
                    raw_start_y=int(raw_y),
                    game_position=game_pos,
                    blender_position=_convert_level_position(*game_pos),
                ))
                cursor += self.EDGE_SIZE

            if has_portals:
                cursor = self._parse_portals(data, cursor, area, area_end, pos_x, pos_y, pos_z)

            parsed.areas.append(area)
            area_index += 1

        if cursor != area_end:
            raise ValueError(f'Area block parse ended at 0x{cursor:X}, expected 0x{area_end:X}')

    def _parse_portals(self, data: bytes, cursor: int, area: AreaDBaseArea, area_end: int, pos_x: int, pos_y: int, pos_z: int) -> int:
        portal_index = 0
        while True:
            if cursor + self.PORTAL_SIZE > area_end:
                raise ValueError(f'Truncated cdc::APortal array for area {area.index}')
            raw_src_x, raw_src_y, raw_src_z, adj_area_ref, user_data, raw_portal_id = struct.unpack_from(f'{self.endian}HHHhHH', data, cursor)
            src_x = _sign_extend(int(raw_src_x) & 0x7FFF, 15)
            src_y = _sign_extend(int(raw_src_y) & 0x7FFF, 15)
            src_z = _sign_extend(int(raw_src_z) & 0x7FFF, 15)
            last_portal = (int(raw_src_x) >> 15) & 0x1
            usable = (int(raw_src_z) >> 15) & 0x1
            portal_id = int(raw_portal_id) & 0x01FF
            sibling_portal_index = (int(raw_portal_id) >> 9) & 0x7F
            game_pos = (
                float((int(pos_x) * 128) + int(src_x)),
                float((int(pos_y) * 128) + int(src_y)),
                float((int(pos_z) * 32) + int(src_z)),
            )
            area.portals.append(AreaDBasePortal(
                area_index=int(area.index),
                portal_index=int(portal_index),
                src_x=int(src_x),
                src_y=int(src_y),
                src_z=int(src_z),
                last_portal=int(last_portal),
                usable=int(usable),
                adj_area_ref=int(adj_area_ref),
                user_data=int(user_data),
                portal_id=int(portal_id),
                sibling_portal_index=int(sibling_portal_index),
                raw_src_x=int(raw_src_x),
                raw_src_y=int(raw_src_y),
                raw_src_z=int(raw_src_z),
                raw_portal_id=int(raw_portal_id),
                game_position=game_pos,
                blender_position=_convert_level_position(*game_pos),
            ))
            cursor += self.PORTAL_SIZE
            portal_index += 1
            if last_portal:
                break
            if portal_index > 256:
                raise ValueError(f'Unterminated cdc::APortal array for area {area.index}')
        return cursor

    def _parse_kd_records(self, data: bytes, header: int, parsed: AreaDBaseData) -> None:
        tree_size = int(parsed.tree_bytes or 0)
        if tree_size <= 0:
            return
        tree_start = int(header) + int(parsed.tree_offset)
        tree_end = tree_start + tree_size
        if tree_start < 0 or tree_end > len(data):
            return
        count = tree_size // 8
        for index in range(count):
            offset = tree_start + (index * 8)
            a, b, c, d = struct.unpack_from(f'{self.endian}HHhh', data, offset)
            parsed.kd_records.append(AreaDBaseKDRecord(
                index=int(index),
                pos_child_offset_or_zero=int(a),
                axis_or_prim=int(b),
                neg_offset_or_pad=int(c),
                pos_offset_or_pad2=int(d),
            ))


    def _parse_gateways(self, data: bytes, header: int, parsed: AreaDBaseData) -> None:
        count = int(parsed.num_gateways or 0)
        if count <= 0:
            return
        gateway_start = int(header) + int(parsed.gateway_offset)
        gateway_end = gateway_start + (count * self.GATEWAY_SIZE)
        if gateway_start < 0 or gateway_end > len(data):
            return
        for index in range(count):
            offset = gateway_start + (index * self.GATEWAY_SIZE)
            name = _read_c_string(data, offset, 24)
            dest_unit_name = _read_c_string(data, offset + 24, 24)
            sibling_gateway_name = _read_c_string(data, offset + 48, 24)
            pos_x, pos_y, pos_z = struct.unpack_from(f'{self.endian}hhh', data, offset + 0x48)
            num_links = int(data[offset + 0x4E])
            in_open_list = int(data[offset + 0x4F])
            static_area_ref, pad = struct.unpack_from(f'{self.endian}hH', data, offset + 0x7C)
            search_cost = struct.unpack_from(f'{self.endian}f', data, offset + 0x80)[0]
            game_pos = (float(pos_x), float(pos_y), float(pos_z))
            parsed.gateways.append(AreaDBaseGateway(
                index=int(index),
                name=name,
                dest_unit_name=dest_unit_name,
                sibling_gateway_name=sibling_gateway_name,
                pos_x=int(pos_x),
                pos_y=int(pos_y),
                pos_z=int(pos_z),
                num_links=int(num_links),
                in_open_list=int(in_open_list),
                static_area_ref=int(static_area_ref),
                pad=int(pad),
                search_cost=float(search_cost),
                game_position=game_pos,
                blender_position=_convert_level_position(*game_pos),
            ))
