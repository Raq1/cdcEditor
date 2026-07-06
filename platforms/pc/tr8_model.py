from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

from ...core.log import logger
from ...core.model_types import BoneMirrorEntry, HBox, HCapsule, HMarker, HSphere, MVertex, ModelData, Segment, TextureStrip
from ..common.section import SectionContext, SectionContextCache
try:
    from ..ps2.texture import parse_tr8_ps2_pcd_bytes, decode_supported_ps2_texture
except Exception:
    parse_tr8_ps2_pcd_bytes = None
    decode_supported_ps2_texture = None


TR8_VERTEX_POSITION = 0xD2F7D823
TR8_VERTEX_NORMAL = 0x36F5E414
TR8_VERTEX_TANGENT = 0xF1ED11C3
TR8_VERTEX_BINORMAL = 0x64A86F01
TR8_VERTEX_SKIN_WEIGHTS = 0x48E691C0
TR8_VERTEX_SKIN_INDICES = 0x5156D8D3
TR8_VERTEX_TEXCOORD1 = 0x8317902A
TR8_VERTEX_TEXCOORD2 = 0x8E54B6F3
TR8_VERTEX_TEXCOORD3 = 0x8A95AB44
TR8_VERTEX_COLOR1 = 0x7E7DD623
TR8_VERTEX_COLOR2 = 0x733EF0FA


@dataclass(slots=True)
class TR8VertexComponent:
    semantic: int
    offset: int
    entry_type: int
    entry_null: int


@dataclass(slots=True)
class TR8MeshInfo:
    index: int
    vertex_components_offset: int
    num_vertices: int
    base_index: int
    num_faces: int
    submesh_count: int
    skin_map_size: int
    skin_map_offset: int
    vertex_offset: int
    global_vertex_start: int = 0
    group_start: int = 0
    group_count: int = 0
    palette_source_index: int = -1
    geometry_source_index: int = -1


@dataclass(slots=True)
class TR8MeshGroupInfo:
    index: int
    base_index: int
    num_faces: int
    num_vertices: int
    flags: int
    draw_group_id: int
    order: int
    material_index: int


@dataclass(slots=True)
class TR8ShaderTextureInfo:
    texture_id: int
    texture_type: int
    slot: int
    filter: int


@dataclass(slots=True)
class TR8MaterialInfo:
    resource_id: int
    filepath: str
    shader_ids: List[int]
    shader_files: List[str]
    shader_strings: List[str]
    textures: List[TR8ShaderTextureInfo]


class TR8ModelParser:

    def __init__(self, filepath: str, import_hinfo: bool = True, import_markups: bool = True, endian: str = "<", parse_segments_only: bool = False):
        self.filepath = filepath
        self.import_hinfo = import_hinfo
        self.import_markups = import_markups
        self.endian = endian
        self.parse_segments_only = parse_segments_only

    @staticmethod
    def looks_like_tr8mesh(filepath: str | Path) -> bool:
        path = Path(filepath)
        if path.suffix.lower() == '.tr8mesh':
            return True
        try:
            with open(path, 'rb') as fh:
                header = fh.read(0x18)
            return len(header) >= 0x18 and header[:4] == b'SECT' and header[8] == 0x0C
        except Exception:
            return False

    def parse(self) -> ModelData:
        cache = SectionContextCache(self.filepath, endian=self.endian)
        try:
            context = cache.get_root_context()
            return self._parse_context(context)
        finally:
            cache.close()

    @staticmethod
    def _valid_local(context: SectionContext, local_offset: int, size: int = 1) -> bool:
        return 0 <= int(local_offset) and context.data_start + int(local_offset) + max(0, int(size)) <= context.file_size

    @staticmethod
    def _read_u32_at(context: SectionContext, local_offset: int, default: int = 0) -> int:
        if not TR8ModelParser._valid_local(context, local_offset, 4):
            return int(default)
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            return int(context.reader.u32())
        finally:
            context.reader.seek(previous)


    @staticmethod
    def _read_u8_at(context: SectionContext, local_offset: int, default: int = 0) -> int:
        if not TR8ModelParser._valid_local(context, local_offset, 1):
            return int(default)
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            return int(context.reader.u8())
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _read_i16_at(context: SectionContext, local_offset: int, default: int = 0) -> int:
        if not TR8ModelParser._valid_local(context, local_offset, 2):
            return int(default)
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            return int(context.reader.i16())
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _read_u16_at(context: SectionContext, local_offset: int, default: int = 0) -> int:
        if not TR8ModelParser._valid_local(context, local_offset, 2):
            return int(default)
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            return int(context.reader.u16())
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _read_i32_at(context: SectionContext, local_offset: int, default: int = 0) -> int:
        if not TR8ModelParser._valid_local(context, local_offset, 4):
            return int(default)
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            return int(context.reader.i32())
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _read_f32_at(context: SectionContext, local_offset: int, default: float = 0.0) -> float:
        if not TR8ModelParser._valid_local(context, local_offset, 4):
            return float(default)
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            value = float(context.reader.f32())
            return value if math.isfinite(value) else float(default)
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _read_vec4_at(context: SectionContext, local_offset: int) -> tuple[float, float, float, float]:
        return (
            TR8ModelParser._read_f32_at(context, int(local_offset) + 0),
            TR8ModelParser._read_f32_at(context, int(local_offset) + 4),
            TR8ModelParser._read_f32_at(context, int(local_offset) + 8),
            TR8ModelParser._read_f32_at(context, int(local_offset) + 12),
        )

    @staticmethod
    def _rotate_vec3_by_quaternion(quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
        qx, qy, qz, qw = quaternion
        vx, vy, vz = vector
        tx = 2.0 * ((qy * vz) - (qz * vy))
        ty = 2.0 * ((qz * vx) - (qx * vz))
        tz = 2.0 * ((qx * vy) - (qy * vx))
        rx = vx + (qw * tx) + ((qy * tz) - (qz * ty))
        ry = vy + (qw * ty) + ((qz * tx) - (qx * tz))
        rz = vz + (qw * tz) + ((qx * ty) - (qy * tx))
        return (float(rx), float(ry), float(rz))

    def _parse_tr8_hspheres(self, context: SectionContext, list_local: int, count: int, owner_segment: int, start_index: int) -> List[HSphere]:
        spheres: List[HSphere] = []
        record_size = 36
        if count <= 0 or not self._valid_local(context, list_local, int(count) * record_size):
            return spheres
        for local_index in range(int(count)):
            base = int(list_local) + (local_index * record_size)
            radius = self._read_u16_at(context, base + 0x06)
            spheres.append(
                HSphere(
                    global_index=start_index + local_index,
                    owner_segment=int(owner_segment),
                    flags=self._read_u16_at(context, base + 0x00),
                    id=self._read_u8_at(context, base + 0x02),
                    rank=self._read_u8_at(context, base + 0x03),
                    radius=radius,
                    x=self._read_i16_at(context, base + 0x08),
                    y=self._read_i16_at(context, base + 0x0A),
                    z=self._read_i16_at(context, base + 0x0C),
                    radius_sq=self._read_u32_at(context, base + 0x10, int(radius) * int(radius)),
                    mass=self._read_u16_at(context, base + 0x14),
                    buoyancy_factor=self._read_u8_at(context, base + 0x16),
                    explosion_factor=self._read_u8_at(context, base + 0x18),
                    material_type=self._read_i32_at(context, base + 0x1A),
                    pad=self._read_u8_at(context, base + 0x04),
                    damage=self._read_i16_at(context, base + 0x1E),
                )
            )
        return spheres

    def _parse_tr8_hmarkers(self, context: SectionContext, list_local: int, count: int, owner_segment: int, start_index: int) -> List[HMarker]:
        markers: List[HMarker] = []
        record_size = 32
        if count <= 0 or not self._valid_local(context, list_local, int(count) * record_size):
            return markers
        for local_index in range(int(count)):
            base = int(list_local) + (local_index * record_size)
            marker_segment = self._read_i32_at(context, base + 0x00, int(owner_segment))
            markers.append(
                HMarker(
                    global_index=start_index + local_index,
                    owner_segment=int(owner_segment),
                    bone=marker_segment,
                    index=self._read_u16_at(context, base + 0x04),
                    position=(
                        self._read_f32_at(context, base + 0x08),
                        self._read_f32_at(context, base + 0x0C),
                        self._read_f32_at(context, base + 0x10),
                    ),
                    rotation=(
                        self._read_f32_at(context, base + 0x14),
                        self._read_f32_at(context, base + 0x18),
                        self._read_f32_at(context, base + 0x1C),
                    ),
                )
            )
        return markers

    def _parse_tr8_hboxes(self, context: SectionContext, list_local: int, count: int, owner_segment: int, start_index: int) -> List[HBox]:
        boxes: List[HBox] = []
        record_size = 72
        if count <= 0 or not self._valid_local(context, list_local, int(count) * record_size):
            return boxes
        for local_index in range(int(count)):
            base = int(list_local) + (local_index * record_size)
            boxes.append(
                HBox(
                    global_index=start_index + local_index,
                    owner_segment=int(owner_segment),
                    flags=self._read_u16_at(context, base + 0x30),
                    id=self._read_u8_at(context, base + 0x32),
                    rank=self._read_u8_at(context, base + 0x33),
                    mass=self._read_u16_at(context, base + 0x36),
                    buoyancy_factor=self._read_u8_at(context, base + 0x38),
                    explosion_factor=self._read_u8_at(context, base + 0x3A),
                    material_type=self._read_i32_at(context, base + 0x3C),
                    pad=self._read_u8_at(context, base + 0x34),
                    damage=self._read_i16_at(context, base + 0x40),
                    dimensions=self._read_vec4_at(context, base + 0x00),
                    position=self._read_vec4_at(context, base + 0x10),
                    quaternion=self._read_vec4_at(context, base + 0x20),
                )
            )
        return boxes

    def _parse_tr8_hcapsules(self, context: SectionContext, list_local: int, count: int, owner_segment: int, start_index: int) -> List[HCapsule]:
        capsules: List[HCapsule] = []
        record_size = 60
        if count <= 0 or not self._valid_local(context, list_local, int(count) * record_size):
            return capsules
        for local_index in range(int(count)):
            base = int(list_local) + (local_index * record_size)
            position = self._read_vec4_at(context, base + 0x00)
            quaternion = self._read_vec4_at(context, base + 0x10)
            radius = self._read_u16_at(context, base + 0x26)
            length = self._read_u16_at(context, base + 0x28)
            position3 = (float(position[0]), float(position[1]), float(position[2]))
            direction = self._rotate_vec3_by_quaternion(
                (float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3])),
                (0.0, 0.0, 1.0),
            )
            half_length = float(length) * 0.5
            start = (
                position3[0] - (direction[0] * half_length),
                position3[1] - (direction[1] * half_length),
                position3[2] - (direction[2] * half_length),
            )
            end = (
                position3[0] + (direction[0] * half_length),
                position3[1] + (direction[1] * half_length),
                position3[2] + (direction[2] * half_length),
            )
            capsules.append(
                HCapsule(
                    global_index=start_index + local_index,
                    owner_segment=int(owner_segment),
                    flags=self._read_u16_at(context, base + 0x20),
                    id=self._read_u8_at(context, base + 0x22),
                    rank=self._read_u8_at(context, base + 0x23),
                    radius=radius,
                    length=length,
                    mass=self._read_u16_at(context, base + 0x2A),
                    buoyancy_factor=self._read_u8_at(context, base + 0x2C),
                    explosion_factor=self._read_u8_at(context, base + 0x2E),
                    material_type=self._read_i32_at(context, base + 0x30),
                    pad=self._read_u8_at(context, base + 0x24),
                    damage=self._read_i16_at(context, base + 0x34),
                    position=position,
                    quaternion=quaternion,
                    start=start,
                    end=end,
                )
            )
        return capsules

    def _read_tr8_local_pointer_at(self, context: SectionContext, field_local_offset: int) -> int:
        return int(self._read_u32_at(context, int(field_local_offset), 0) or 0)

    def _parse_tr8_bone_mirror_entries(self, context: SectionContext, table_local: int) -> List[BoneMirrorEntry]:
        entries: List[BoneMirrorEntry] = []
        table_local = int(table_local)
        if table_local <= 0 or not self._valid_local(context, table_local, 2):
            return entries

        br = context.reader
        previous = br.tell()
        try:
            br.seek(context.data_start + table_local)
            data_limit = min(context.data_end, context.file_size)
            seen = 0
            while br.tell() + 3 <= data_limit and seen < 1024:
                if br.tell() + 2 <= data_limit and br.peek_u16() == 0:
                    break
                bone1 = int(br.u8())
                bone2 = int(br.u8())
                count = int(br.u8())
                if count <= 0:
                    break
                entries.append(BoneMirrorEntry(bone1=bone1, bone2=bone2, count=count))
                seen += 1
                if br.tell() + 2 <= data_limit and br.peek_u16() == 0:
                    break
        finally:
            br.seek(previous)
        return entries

    def attach_underworld_model_hinfo(self, model: ModelData, model_filepath: str | Path) -> ModelData:
        cache = SectionContextCache(str(model_filepath), endian=self.endian)
        try:
            context = cache.get_root_context()
            magic = self._read_u32_at(context, 0x00)
            if magic != 0x04C20453:
                logger.debug('Skipping TR8 HInfo read from %s: unexpected model magic 0x%X', context.file_name, int(magic))
                return model

            bone_mirror_data_local = self._read_tr8_local_pointer_at(context, 0x4C)
            bone_mirror_entries = self._parse_tr8_bone_mirror_entries(context, bone_mirror_data_local)
            if bone_mirror_entries:
                model.bone_mirror_entries = list(bone_mirror_entries)
                logger.info(
                    'Imported TR8 bone mirror table from %s: entries=%d local=0x%X',
                    context.file_name,
                    len(bone_mirror_entries),
                    int(bone_mirror_data_local),
                )
            else:
                logger.debug('No importable TR8 bone mirror table found in %s local=0x%X', context.file_name, int(bone_mirror_data_local))

            if not self.import_hinfo:
                return model

            num_segments = self._read_u32_at(context, 0x04)
            segment_list_local = self._read_tr8_local_pointer_at(context, 0x0C)
            segment_record_size = 64
            if num_segments <= 0 or num_segments > 4096 or not self._valid_local(context, segment_list_local, int(num_segments) * segment_record_size):
                logger.debug(
                    'Skipping TR8 HInfo read from %s: invalid segment list count=%d local=0x%X',
                    context.file_name,
                    int(num_segments),
                    int(segment_list_local),
                )
                return model

            hmarkers: List[HMarker] = []
            hspheres: List[HSphere] = []
            hboxes: List[HBox] = []
            hcapsules: List[HCapsule] = []
            hinfo_segments = 0
            for segment_index in range(int(num_segments)):
                segment_local = int(segment_list_local) + (segment_index * segment_record_size)
                hinfo_local = self._read_tr8_local_pointer_at(context, segment_local + 0x3C)
                if hinfo_local <= 0 or not self._valid_local(context, hinfo_local, 40):
                    continue

                num_hspheres = self._read_i32_at(context, hinfo_local + 0x00)
                hspheres_local = self._read_tr8_local_pointer_at(context, hinfo_local + 0x04)
                num_hboxes = self._read_i32_at(context, hinfo_local + 0x08)
                hboxes_local = self._read_tr8_local_pointer_at(context, hinfo_local + 0x0C)
                num_hmarkers = self._read_i32_at(context, hinfo_local + 0x10)
                hmarkers_local = self._read_tr8_local_pointer_at(context, hinfo_local + 0x14)
                num_hcapsules = self._read_i32_at(context, hinfo_local + 0x18)
                hcapsules_local = self._read_tr8_local_pointer_at(context, hinfo_local + 0x1C)
                num_hgeoms = self._read_i32_at(context, hinfo_local + 0x20)

                if any(value < 0 or value > 65535 for value in (num_hspheres, num_hboxes, num_hmarkers, num_hcapsules, num_hgeoms)):
                    logger.debug(
                        'Skipping invalid TR8 HInfo header in %s segment %d local=0x%X counts=(%d,%d,%d,%d,%d)',
                        context.file_name,
                        int(segment_index),
                        int(hinfo_local),
                        int(num_hspheres),
                        int(num_hboxes),
                        int(num_hmarkers),
                        int(num_hcapsules),
                        int(num_hgeoms),
                    )
                    continue

                start_counts = (len(hspheres), len(hboxes), len(hmarkers), len(hcapsules))
                hspheres.extend(self._parse_tr8_hspheres(context, hspheres_local, num_hspheres, segment_index, len(hspheres)))
                hboxes.extend(self._parse_tr8_hboxes(context, hboxes_local, num_hboxes, segment_index, len(hboxes)))
                hmarkers.extend(self._parse_tr8_hmarkers(context, hmarkers_local, num_hmarkers, segment_index, len(hmarkers)))
                hcapsules.extend(self._parse_tr8_hcapsules(context, hcapsules_local, num_hcapsules, segment_index, len(hcapsules)))
                end_counts = (len(hspheres), len(hboxes), len(hmarkers), len(hcapsules))
                if end_counts != start_counts:
                    hinfo_segments += 1

            if hmarkers or hspheres or hboxes or hcapsules:
                model.hmarkers.extend(hmarkers)
                model.hspheres.extend(hspheres)
                model.hboxes.extend(hboxes)
                model.hcapsules.extend(hcapsules)
                logger.info(
                    'Imported TR8 model HInfo from %s: hinfoSegments=%d hmarkers=%d hspheres=%d hboxes=%d hcapsules=%d',
                    context.file_name,
                    int(hinfo_segments),
                    len(hmarkers),
                    len(hspheres),
                    len(hboxes),
                    len(hcapsules),
                )
            else:
                logger.debug('No importable TR8 model HInfo found in %s', context.file_name)
            return model
        finally:
            cache.close()

    def _find_mesh_header_local(self, context: SectionContext) -> int:
        previous = context.reader.tell()
        try:
            start = context.data_start
            end = min(context.data_end, context.file_size)
            if end - start < 0x90:
                return -1
            context.reader.seek(start)
            data = context.reader.read(end - start)
            magic = b'Mesh'
            offset = data.find(magic)
            while offset >= 0:
                if offset + 0x84 <= len(data):
                    num_indices = int.from_bytes(data[offset + 0x0C:offset + 0x10], 'little', signed=False)
                    num_mesh_groups = int.from_bytes(data[offset + 0x7C:offset + 0x7E], 'little', signed=False)
                    num_meshes = int.from_bytes(data[offset + 0x7E:offset + 0x80], 'little', signed=False)
                    num_bones = int.from_bytes(data[offset + 0x80:offset + 0x82], 'little', signed=False)
                    if 0 <= num_indices <= 50_000_000 and 0 <= num_mesh_groups <= 100_000 and 0 <= num_meshes <= 4096 and 0 <= num_bones <= 4096:
                        return int(offset)
                offset = data.find(magic, offset + 1)
            return -1
        finally:
            context.reader.seek(previous)

    def _read_header(self, context: SectionContext, header_local: int) -> dict[str, int | float | tuple[float, float, float, float]]:
        br = context.reader
        br.seek(context.data_start + header_local)
        magic = br.u32()
        flags = br.u32()
        offset_mat_info = br.u32()
        num_indices = br.u32()
        bounding_sphere_center = br.vec4()
        box_min = br.vec4()
        box_max = br.vec4()
        bounding_sphere_radius = br.f32()
        model_type = br.u32()
        br.skip(36)
        offset_mesh_group_info = br.u32()
        offset_mesh_info = br.u32()
        offset_bone_map = br.u32()
        offset_face_data = br.u32()
        num_mesh_groups = br.u16()
        num_meshes = br.u16()
        num_bones = br.u16()
        return {
            'magic': int(magic),
            'flags': int(flags),
            'offset_mat_info': int(offset_mat_info),
            'num_indices': int(num_indices),
            'bounding_sphere_center': bounding_sphere_center,
            'box_min': box_min,
            'box_max': box_max,
            'bounding_sphere_radius': float(bounding_sphere_radius),
            'model_type': int(model_type),
            'offset_mesh_group_info': int(offset_mesh_group_info),
            'offset_mesh_info': int(offset_mesh_info),
            'offset_bone_map': int(offset_bone_map),
            'offset_face_data': int(offset_face_data),
            'num_mesh_groups': int(num_mesh_groups),
            'num_meshes': int(num_meshes),
            'num_bones': int(num_bones),
        }

    @staticmethod
    def _find_section_file_by_resource_id(directory: Path, resource_id: int) -> Path | None:
        try:
            target = f'{int(resource_id) & 0xFFFFFFFF:x}'
        except Exception:
            return None
        try:
            entries = sorted(Path(directory).iterdir(), key=lambda path: path.name)
        except Exception:
            return None
        for candidate in entries:
            if not candidate.is_file():
                continue
            parts = candidate.stem.split('_', 1)
            if len(parts) < 2:
                continue
            if parts[1].lower() == target:
                return candidate
        return None

    @staticmethod
    def _printable_strings(data: bytes, *, min_length: int = 4, limit: int = 32) -> List[str]:
        strings: List[str] = []
        current = bytearray()
        for value in data:
            if 32 <= int(value) < 127:
                current.append(int(value))
                continue
            if len(current) >= int(min_length):
                text = current.decode('ascii', errors='ignore')
                if text and text not in strings:
                    strings.append(text)
                    if len(strings) >= int(limit):
                        break
            current.clear()
        if len(current) >= int(min_length) and len(strings) < int(limit):
            text = current.decode('ascii', errors='ignore')
            if text and text not in strings:
                strings.append(text)
        return strings

    def _parse_shader_strings(self, shader_path: Path | None) -> List[str]:
        if shader_path is None:
            return []
        try:
            data = Path(shader_path).read_bytes()
        except Exception:
            return []
        # The TR8 shader sections contain compiled D3D9 bytecode with CTAB
        # comment chunks.  Full bytecode decompilation is unnecessary for material
        # binding: extracting the CTAB/ASCII names gives shader and sampler context
        # for diagnostics while the .matd texture table supplies the actual refs.
        strings = self._printable_strings(data, min_length=4, limit=64)
        return [text for text in strings if text not in {'SECT', 'CTAB'}]

    def _parse_tr8_material_ids(self, context: SectionContext, offset_mat_info: int) -> List[int]:
        material_ids: List[int] = []
        if not self._valid_local(context, int(offset_mat_info), 0x18):
            return material_ids

        count = self._read_u32_at(context, int(offset_mat_info) + 0x14)
        if count <= 0 or count > 4096:
            return material_ids
        list_local = int(offset_mat_info) + 0x18
        if not self._valid_local(context, list_local, int(count) * 4):
            return []
        for index in range(int(count)):
            value = self._read_u32_at(context, list_local + (index * 4))
            material_ids.append(int(value) & 0xFFFFFFFF)
        return material_ids

    def _parse_tr8_material_file(self, material_path: Path, resource_id: int) -> TR8MaterialInfo | None:
        cache = SectionContextCache(str(material_path), endian=self.endian)
        try:
            context = cache.get_root_context()
            if context.section_info.section_type != 10 or context.data_size < 0x60:
                return None

            version = self._read_i16_at(context, 0x00)
            if version <= 0 or version > 256:
                return None

            shader_ids: List[int] = []
            shader_files: List[str] = []
            shader_strings: List[str] = []
            for shader_index in range(4):
                shader_id = self._read_u32_at(context, 0x28 + (shader_index * 4), 0)
                if shader_id in {0, 0xFFFFFFFF}:
                    continue
                shader_ids.append(int(shader_id))
                shader_path = self._find_section_file_by_resource_id(context.filepath.parent, int(shader_id))
                if shader_path is not None:
                    shader_files.append(shader_path.name)
                    for text in self._parse_shader_strings(shader_path):
                        if text not in shader_strings:
                            shader_strings.append(text)
                else:
                    shader_files.append('')

            textures: List[TR8ShaderTextureInfo] = []
            for pass_index in range(2):
                pass_local = 0x50 + (pass_index * 0x10)
                if not self._valid_local(context, pass_local, 0x10):
                    continue
                num_textures = self._read_u16_at(context, pass_local + 0x02)
                texture_list_local = self._read_u32_at(context, pass_local + 0x04)
                if num_textures <= 0 or num_textures > 128:
                    continue
                if not self._valid_local(context, texture_list_local, int(num_textures) * 16):
                    logger.debug(
                        'Skipping invalid TR8 material texture list in %s pass=%d count=%d local=0x%X',
                        context.file_name,
                        pass_index,
                        int(num_textures),
                        int(texture_list_local),
                    )
                    continue
                for texture_index in range(int(num_textures)):
                    entry_local = int(texture_list_local) + (texture_index * 16)
                    texture_id = self._read_u32_at(context, entry_local + 0x00, 0)
                    texture_type = self._read_u8_at(context, entry_local + 0x08, 0)
                    slot = self._read_u8_at(context, entry_local + 0x0D, texture_index)
                    texture_filter = self._read_u16_at(context, entry_local + 0x0E, 0)
                    if texture_id in {0, 0xFFFFFFFF}:
                        continue
                    textures.append(
                        TR8ShaderTextureInfo(
                            texture_id=int(texture_id) & 0xFFFFFFFF,
                            texture_type=int(texture_type) & 0xFF,
                            slot=int(slot) & 0xFF,
                            filter=int(texture_filter) & 0xFFFF,
                        )
                    )

            return TR8MaterialInfo(
                resource_id=int(resource_id) & 0xFFFFFFFF,
                filepath=str(material_path),
                shader_ids=shader_ids,
                shader_files=shader_files,
                shader_strings=shader_strings,
                textures=textures,
            )
        except Exception as exc:
            logger.debug('Could not parse TR8 material %s: %s', Path(material_path).name, exc)
            return None
        finally:
            cache.close()

    @staticmethod
    def _first_texture_id_by_type(material: TR8MaterialInfo, texture_type: int, *, exclude: set[int] | None = None) -> int:
        exclude = exclude or set()
        for texture in material.textures:
            texture_id = int(texture.texture_id)
            if int(texture.texture_type) == int(texture_type) and texture_id not in exclude:
                return texture_id
        return -1

    def _attach_underworld_materials(self, context: SectionContext, strips: List[TextureStrip], material_ids: List[int]) -> None:
        if not strips or not material_ids:
            return

        material_cache: Dict[int, TR8MaterialInfo | None] = {}
        resolved_materials = 0
        diffuse_count = 0
        normal_count = 0
        for strip in strips:
            material_index = int(getattr(strip, 'material_group', -1))
            if material_index < 0 or material_index >= len(material_ids):
                continue
            material_id = int(material_ids[material_index]) & 0xFFFFFFFF
            strip.tr8_material_resource_id = int(material_id)
            if material_id not in material_cache:
                material_path = self._find_section_file_by_resource_id(context.filepath.parent, material_id)
                material_cache[material_id] = self._parse_tr8_material_file(material_path, material_id) if material_path is not None else None
            material = material_cache.get(material_id)
            if material is None:
                continue

            resolved_materials += 1
            strip.tr8_material_file = Path(material.filepath).name
            strip.tr8_shader_ids = list(material.shader_ids)
            strip.tr8_shader_files = list(material.shader_files)
            strip.tr8_shader_strings = list(material.shader_strings[:16])
            strip.tr8_texture_stage_ids = [int(texture.texture_id) for texture in material.textures]
            strip.tr8_texture_stage_types = [int(texture.texture_type) for texture in material.textures]
            strip.tr8_texture_stage_slots = [int(texture.slot) for texture in material.textures]

            diffuse_id = self._first_texture_id_by_type(material, 1)
            # Some simple TR8 materials in the current samples have no explicit
            # type-1 Diffuse entry; their color map is the first type-0 entry.
            # Preserve type-0 as Mask metadata too, but use it as a diffuse
            # fallback so the material is not left untextured.
            mask_id = self._first_texture_id_by_type(material, 0)
            if diffuse_id < 0:
                diffuse_id = mask_id
            normal_id = self._first_texture_id_by_type(material, 3, exclude={diffuse_id} if diffuse_id >= 0 else set())

            strip.tr8_diffuse_texture_id = int(diffuse_id)
            strip.tr8_normal_texture_id = int(normal_id)
            strip.tr8_ao_texture_id = self._first_texture_id_by_type(material, 7)
            strip.tr8_detail_texture_id = self._first_texture_id_by_type(material, 5)
            strip.tr8_detail_ao_texture_id = self._first_texture_id_by_type(material, 2)
            strip.tr8_mask_texture_id = int(mask_id)
            strip.tr8_reflection_texture_id = self._first_texture_id_by_type(material, 4)
            if strip.tr8_diffuse_texture_id >= 0:
                diffuse_count += 1
            if strip.tr8_normal_texture_id >= 0:
                normal_count += 1

        unique_materials = {int(value) for value in material_ids if int(value) not in {0, 0xFFFFFFFF}}
        logger.info(
            'Parsed TR8 material bindings for %s: materialIds=%d unique=%d resolvedUses=%d strips=%d diffuseUses=%d normalUses=%d',
            context.file_name,
            len(material_ids),
            len(unique_materials),
            int(resolved_materials),
            len(strips),
            int(diffuse_count),
            int(normal_count),
        )

    def _parse_mesh_infos(self, context: SectionContext, offset_mesh_info: int, num_meshes: int) -> List[TR8MeshInfo]:
        infos: List[TR8MeshInfo] = []
        if num_meshes <= 0 or not self._valid_local(context, offset_mesh_info, int(num_meshes) * 52):
            return infos
        br = context.reader
        for index in range(int(num_meshes)):
            br.seek(context.data_start + int(offset_mesh_info) + (index * 52))
            vertex_components_offset = int(br.u32())
            num_vertices = int(br.u32())
            base_index = int(br.u32())
            num_faces = int(br.u32())
            br.skip(8)
            submesh_count = int(br.u32())
            skin_map_size = int(br.u32())
            skin_map_offset = int(br.u32())
            vertex_offset = int(br.u32())
            br.skip(12)
            infos.append(
                TR8MeshInfo(
                    index=index,
                    vertex_components_offset=vertex_components_offset,
                    num_vertices=num_vertices,
                    base_index=base_index,
                    num_faces=num_faces,
                    submesh_count=submesh_count,
                    skin_map_size=skin_map_size,
                    skin_map_offset=skin_map_offset,
                    vertex_offset=vertex_offset,
                )
            )
        return infos

    def _parse_mesh_groups(self, context: SectionContext, offset_mesh_group_info: int, num_mesh_groups: int) -> List[TR8MeshGroupInfo]:
        groups: List[TR8MeshGroupInfo] = []
        if num_mesh_groups <= 0 or not self._valid_local(context, offset_mesh_group_info, int(num_mesh_groups) * 64):
            return groups
        br = context.reader
        for index in range(int(num_mesh_groups)):
            group_abs = context.data_start + int(offset_mesh_group_info) + (index * 64)
            br.seek(group_abs + 0x20)
            base_index = int(br.u32())
            num_faces = int(br.u32())
            num_vertices = int(br.u32())
            flags = int(br.u32())
            draw_group_id = int(br.i32())
            order = int(br.i32())
            material_index = int(br.u32())
            if num_faces <= 0:
                continue
            groups.append(
                TR8MeshGroupInfo(
                    index=index,
                    base_index=base_index,
                    num_faces=num_faces,
                    num_vertices=num_vertices,
                    flags=flags,
                    draw_group_id=draw_group_id,
                    order=order,
                    material_index=material_index,
                )
            )
        return groups

    def _parse_bone_map(self, context: SectionContext, offset_bone_map: int, num_bones: int) -> List[int]:
        if num_bones <= 0 or not self._valid_local(context, offset_bone_map, int(num_bones) * 4):
            return []
        br = context.reader
        br.seek(context.data_start + int(offset_bone_map))
        return [int(br.u32()) for _ in range(int(num_bones))]

    def _parse_skin_map(self, context: SectionContext, mesh: TR8MeshInfo, global_bone_map: List[int]) -> List[int]:
        del global_bone_map
        if mesh.skin_map_size <= 0:
            return []

        palette_local = int(mesh.skin_map_offset) + 0x10
        if not self._valid_local(context, palette_local, int(mesh.skin_map_size) * 4):
            palette_local = int(mesh.skin_map_offset)
            if not self._valid_local(context, palette_local, int(mesh.skin_map_size) * 4):
                return []

        br = context.reader
        br.seek(context.data_start + palette_local)
        return [int(br.u32()) for _ in range(int(mesh.skin_map_size))]

    def _parse_vertex_format(self, context: SectionContext, mesh: TR8MeshInfo) -> tuple[int, Dict[int, TR8VertexComponent]]:
        if not self._valid_local(context, mesh.vertex_components_offset, 0x20):
            return 0, {}
        br = context.reader
        base_abs = context.data_start + int(mesh.vertex_components_offset)
        br.seek(base_abs + 0x18)
        num_components = int(br.u16())
        stride = int(br.u8())
        if num_components <= 0 or num_components > 64 or stride <= 0 or stride > 256:
            logger.debug(
                'TR8 mesh[%d] has invalid vertex format header at local 0x%X: components=%d stride=%d',
                mesh.index,
                mesh.vertex_components_offset,
                num_components,
                stride,
            )
            return 0, {}

        entries: Dict[int, TR8VertexComponent] = {}
        br.seek(base_abs + 0x20)
        for _ in range(num_components):
            semantic = int(br.u32())
            offset = int(br.u16())
            entry_type = int(br.u8())
            entry_null = int(br.u8())
            entries[semantic] = TR8VertexComponent(semantic=semantic, offset=offset, entry_type=entry_type, entry_null=entry_null)
        return stride, entries


    def _infer_vertex_stride(
        self,
        context: SectionContext,
        mesh: TR8MeshInfo,
        format_stride: int,
        components: Dict[int, TR8VertexComponent],
        vertex_data_local: int,
    ) -> int:
        pos_component = components.get(TR8_VERTEX_POSITION)
        if pos_component is None or mesh.num_vertices <= 1:
            return int(format_stride)
        min_stride = 0
        for semantic in (TR8_VERTEX_POSITION, TR8_VERTEX_NORMAL, TR8_VERTEX_SKIN_WEIGHTS, TR8_VERTEX_SKIN_INDICES, TR8_VERTEX_TEXCOORD1, TR8_VERTEX_COLOR1):
            component = components.get(semantic)
            if component is None:
                continue
            size = 12 if semantic == TR8_VERTEX_POSITION else 4
            min_stride = max(min_stride, int(component.offset) + size)
        candidate_values = {int(format_stride), int(format_stride) + 4, int(format_stride) - 4, min_stride, min_stride + 4, min_stride + 8, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64}
        candidates = sorted(value for value in candidate_values if value >= max(12, min_stride) and value <= 128)
        sample_count = min(int(mesh.num_vertices), 96)
        br = context.reader
        previous = br.tell()
        best_stride = int(format_stride)
        best_score: tuple[int, int, int, int] = (-1, -1, -1, -999)
        try:
            for stride in candidates:
                if not self._valid_local(context, vertex_data_local, ((sample_count - 1) * stride) + int(pos_component.offset) + 12):
                    continue
                sane = 0
                finite = 0
                nonzero = 0
                for index in range(sample_count):
                    br.seek(context.data_start + vertex_data_local + (index * stride) + int(pos_component.offset))
                    x = float(br.f32())
                    y = float(br.f32())
                    z = float(br.f32())
                    if x != x or y != y or z != z:
                        continue
                    if not (-100000.0 < x < 100000.0 and -100000.0 < y < 100000.0 and -100000.0 < z < 100000.0):
                        continue
                    finite += 1
                    magnitude = abs(x) + abs(y) + abs(z)
                    if magnitude > 1.0:
                        nonzero += 1
                    if -5000.0 < x < 5000.0 and -5000.0 < y < 5000.0 and -5000.0 < z < 5000.0 and magnitude > 1.0:
                        sane += 1
                format_bonus = 1 if stride == int(format_stride) else 0
                score = (sane, nonzero, finite, format_bonus)
                if score > best_score:
                    best_score = score
                    best_stride = int(stride)
        finally:
            br.seek(previous)
        if best_stride != int(format_stride):
            logger.debug('Adjusted TR8 mesh[%d] vertex stride from %d to %d', mesh.index, int(format_stride), best_stride)
        return best_stride

    @staticmethod
    def _decode_normal_byte(value: int) -> int:
        normal = (float(int(value) & 0xFF) / 255.0) * 2.0 - 1.0
        return int(round(max(-1.0, min(1.0, normal)) * 127.0))

    @staticmethod
    def _decode_uv_component(value: int) -> float:
        raw = int(value) & 0xFFFF
        if raw >= 0x8000:
            raw -= 0x10000
        return float(raw) / 2048.0


    def _score_vertex_stream_start(
        self,
        context: SectionContext,
        mesh: TR8MeshInfo,
        stride: int,
        components: Dict[int, TR8VertexComponent],
        vertex_data_local: int,
    ) -> tuple[int, int, int, int]:
        pos_component = components.get(TR8_VERTEX_POSITION)
        if pos_component is None or stride <= 0 or mesh.num_vertices <= 0:
            return (-1, -1, -1, -1)
        sample_count = min(int(mesh.num_vertices), 96)
        if not self._valid_local(context, vertex_data_local, ((sample_count - 1) * int(stride)) + int(pos_component.offset) + 12):
            return (-1, -1, -1, -1)
        br = context.reader
        previous = br.tell()
        sane = 0
        finite = 0
        nonzero = 0
        try:
            for index in range(sample_count):
                br.seek(context.data_start + int(vertex_data_local) + (index * int(stride)) + int(pos_component.offset))
                x = float(br.f32())
                y = float(br.f32())
                z = float(br.f32())
                if x != x or y != y or z != z:
                    continue
                if not (-100000.0 < x < 100000.0 and -100000.0 < y < 100000.0 and -100000.0 < z < 100000.0):
                    continue
                finite += 1
                magnitude = abs(x) + abs(y) + abs(z)
                if magnitude > 1.0:
                    nonzero += 1
                if -5000.0 < x < 5000.0 and -5000.0 < y < 5000.0 and -5000.0 < z < 5000.0 and magnitude > 1.0:
                    sane += 1
        finally:
            br.seek(previous)
        return (sane, nonzero, finite, sample_count)

    def _infer_vertex_stream_start(
        self,
        context: SectionContext,
        mesh: TR8MeshInfo,
        stride: int,
        components: Dict[int, TR8VertexComponent],
    ) -> int:
        if stride <= 0 or not components:
            return -1
        component_count = len(components)
        format_end = int(mesh.vertex_components_offset) + 0x20 + (component_count * 8)
        aligned_format_end = (format_end + 15) & ~15
        preferred_candidates = [aligned_format_end, aligned_format_end + 0x10]
        fallback_candidates = [int(mesh.vertex_offset), int(mesh.vertex_offset) + 0x10]

        candidates: list[int] = []
        for candidate in preferred_candidates + fallback_candidates:
            candidate = int(candidate)
            if candidate < 0 or candidate in candidates:
                continue
            candidates.append(candidate)

        best_candidate = -1
        best_score: tuple[int, int, int, int, int] = (-1, -1, -1, -1, -999999999)
        for candidate in candidates:
            sane, nonzero, finite, sample_count = self._score_vertex_stream_start(context, mesh, stride, components, candidate)
            if sane < 0:
                continue
            locality_bonus = 1000000 if candidate in preferred_candidates else -abs(candidate - aligned_format_end)
            score = (sane, nonzero, finite, sample_count, locality_bonus)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate >= 0:
            logger.debug(
                'TR8 mesh[%d] vertex stream start local=0x%X score=%s format_end=0x%X stored_vertexOffset=0x%X',
                mesh.index,
                best_candidate,
                best_score[:4],
                aligned_format_end,
                int(mesh.vertex_offset),
            )
        return best_candidate

    def _parse_vertices_for_mesh(
        self,
        context: SectionContext,
        mesh: TR8MeshInfo,
        skin_map: List[int],
    ) -> tuple[List[MVertex], List[Tuple[int, int, int, int]]]:
        stride, components = self._parse_vertex_format(context, mesh)
        if stride <= 0 or not components or mesh.num_vertices <= 0:
            return [], []
        if TR8_VERTEX_POSITION not in components:
            return [], []
        vertex_data_local = self._infer_vertex_stream_start(context, mesh, stride, components)
        if vertex_data_local < 0:
            return [], []
        stride = self._infer_vertex_stride(context, mesh, stride, components, vertex_data_local)
        vertex_data_local = self._infer_vertex_stream_start(context, mesh, stride, components)
        if vertex_data_local < 0 or not self._valid_local(context, vertex_data_local, int(mesh.num_vertices) * int(stride)):
            return [], []

        br = context.reader
        vertices: List[MVertex] = []
        colors: List[Tuple[int, int, int, int]] = []
        pos_component = components.get(TR8_VERTEX_POSITION)
        normal_component = components.get(TR8_VERTEX_NORMAL)
        weight_component = components.get(TR8_VERTEX_SKIN_WEIGHTS)
        index_component = components.get(TR8_VERTEX_SKIN_INDICES)
        uv_component = components.get(TR8_VERTEX_TEXCOORD1)
        color_component = components.get(TR8_VERTEX_COLOR1)

        for local_index in range(int(mesh.num_vertices)):
            vertex_abs = context.data_start + vertex_data_local + (local_index * stride)
            assert pos_component is not None
            br.seek(vertex_abs + int(pos_component.offset))
            x = float(br.f32())
            y = float(br.f32())
            z = float(br.f32())

            normal_raw = (0, 0, 127)
            if normal_component is not None and int(normal_component.offset) + 4 <= stride:
                br.seek(vertex_abs + int(normal_component.offset))
                nz = self._decode_normal_byte(br.u8())
                ny = self._decode_normal_byte(br.u8())
                nx = self._decode_normal_byte(br.u8())
                br.u8()
                normal_raw = (nx, ny, nz)

            raw_weights = [0, 0, 0, 0]
            if weight_component is not None and int(weight_component.offset) + 4 <= stride:
                br.seek(vertex_abs + int(weight_component.offset))
                raw_weights = [int(br.u8()) for _ in range(4)]

            raw_indices = [0, 0, 0, 0]
            if index_component is not None and int(index_component.offset) + 4 <= stride:
                br.seek(vertex_abs + int(index_component.offset))
                raw_indices = [int(br.u8()) for _ in range(4)]

            uv_raw = (0, 0)
            uv_decoded = (0.0, 0.0)
            if uv_component is not None and int(uv_component.offset) + 4 <= stride:
                br.seek(vertex_abs + int(uv_component.offset))
                raw_u = int(br.u16())
                raw_v = int(br.u16())
                uv_raw = (raw_u, raw_v)
                uv_decoded = (self._decode_uv_component(raw_u), 1.0 - self._decode_uv_component(raw_v))

            color_rgba = (255, 255, 255, 255)
            if color_component is not None and int(color_component.offset) + 4 <= stride:
                br.seek(vertex_abs + int(color_component.offset))
                b = int(br.u8())
                g = int(br.u8())
                r = int(br.u8())
                a = int(br.u8())
                color_rgba = (r, g, b, a)
            colors.append(color_rgba)

            weighted: List[Tuple[int, float]] = []
            for raw_index, raw_weight in zip(raw_indices, raw_weights):
                if raw_weight <= 0:
                    continue
                bone_index = skin_map[int(raw_index)] if 0 <= int(raw_index) < len(skin_map) else int(raw_index)
                if bone_index < 0:
                    continue
                weighted.append((int(bone_index), float(raw_weight) / 255.0))
            if not weighted and skin_map:
                weighted.append((int(skin_map[0]), 1.0))

            total = sum(weight for _bone, weight in weighted)
            if total > 0.0:
                weighted = [(bone, weight / total) for bone, weight in weighted]

            vertex = MVertex(
                index=int(mesh.global_vertex_start) + local_index,
                position_raw=(x, y, z),
                normal_raw=normal_raw,
                segment=weighted[0][0] if weighted else 0,
                uv_raw=uv_raw,
                uv_decoded=uv_decoded,
                skin_weights=weighted or None,
            )
            vertices.append(vertex)

        return vertices, colors

    def _mesh_for_group(self, meshes: List[TR8MeshInfo], group: TR8MeshGroupInfo) -> TR8MeshInfo | None:
        group_index = int(group.index)
        candidates = [
            mesh for mesh in meshes
            if mesh.num_vertices > 0
            and mesh.num_faces > 0
            and int(mesh.group_count) > 0
            and int(mesh.group_start) <= group_index < int(mesh.group_start) + int(mesh.group_count)
        ]
        if candidates:
            return candidates[0]

        # Fallback for layout variants that do not expose batch/group counts.
        group_base = int(group.base_index)
        candidates = [
            mesh for mesh in meshes
            if mesh.num_vertices > 0 and mesh.num_faces > 0 and int(mesh.base_index) <= group_base < int(mesh.base_index) + (int(mesh.num_faces) * 3)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda mesh: int(mesh.base_index))

    def _score_face_index_data_start(
        self,
        context: SectionContext,
        meshes: List[TR8MeshInfo],
        groups: List[TR8MeshGroupInfo],
        index_data_local: int,
        num_indices: int,
    ) -> tuple[int, int, int, int, int]:
        if num_indices <= 0 or not self._valid_local(context, int(index_data_local), int(num_indices) * 2):
            return (-1, -1, -1, -1, -1)

        br = context.reader
        previous = br.tell()
        valid = 0
        degenerate = 0
        out_of_bounds = 0
        unreadable = 0
        sampled = 0
        try:
            for group in groups:
                mesh = self._mesh_for_group(meshes, group)
                if mesh is None or int(mesh.num_vertices) <= 0:
                    continue
                index_count = int(group.num_faces) * 3
                if index_count <= 0:
                    continue
                start = int(group.base_index) * 2
                if start < 0 or start + (index_count * 2) > int(num_indices) * 2:
                    unreadable += int(group.num_faces)
                    continue

                # Sampling every face is still cheap for the current samples, but
                # cap nothing here: this scoring is run only a few times and it
                # prevents selecting a shifted stream that merely looks plausible
                # at the beginning of a large model.
                br.seek(context.data_start + int(index_data_local) + start)
                for _ in range(int(group.num_faces)):
                    i0 = int(br.u16())
                    i1 = int(br.u16())
                    i2 = int(br.u16())
                    sampled += 1
                    if i0 >= int(mesh.num_vertices) or i1 >= int(mesh.num_vertices) or i2 >= int(mesh.num_vertices):
                        out_of_bounds += 1
                    elif i0 == i1 or i1 == i2 or i0 == i2:
                        degenerate += 1
                    else:
                        valid += 1
        finally:
            br.seek(previous)

        return (valid, -degenerate, -out_of_bounds, -unreadable, sampled)

    def _infer_face_index_data_local(
        self,
        context: SectionContext,
        meshes: List[TR8MeshInfo],
        groups: List[TR8MeshGroupInfo],
        offset_face_data: int,
        num_indices: int,
    ) -> int:
        if num_indices <= 0:
            return -1

        # TR8 PC face data can start directly at offsetFaceData, or it can be
        # preceded by a small normal/winding-order blob.  Most currently tested
        # character meshes have a 16-byte prefix; some resources do not.  The
        # mesh groups still index the triangle index buffer from zero, so choose
        # the stream start that produces valid local indices for every group.
        candidate_offsets = (0x10, 0x00, 0x08, 0x04, 0x0C, 0x14, 0x20)
        best_local = -1
        best_score: tuple[int, int, int, int, int, int] = (-1, -999999, -999999, -999999, -1, -999999)
        for extra in candidate_offsets:
            candidate_local = int(offset_face_data) + int(extra)
            score = self._score_face_index_data_start(context, meshes, groups, candidate_local, int(num_indices))
            if score[0] < 0:
                continue
            # Prefer streams with no out-of-bounds/degenerate triangles, then
            # use +0x10 only as a weak tie-breaker because that is the common
            # normal/winding-prefix layout in the verified TRU samples.
            tie_break = 1 if int(extra) == 0x10 else 0
            ranked_score = (score[0], score[1], score[2], score[3], score[4], tie_break)
            if ranked_score > best_score:
                best_score = ranked_score
                best_local = candidate_local

        if best_local >= 0 and best_local != int(offset_face_data) + 0x10:
            logger.info(
                'TR8 mesh %s uses face index data at offsetFaceData+0x%X instead of +0x10; selected by validation score=%s',
                context.file_name,
                best_local - int(offset_face_data),
                best_score[:5],
            )
        elif best_local >= 0:
            logger.debug(
                'TR8 mesh %s uses face index data at offsetFaceData+0x10 score=%s',
                context.file_name,
                best_score[:5],
            )
        return best_local


    @staticmethod
    def _triangle_winding_parity(triangle: List[int] | Tuple[int, int, int]) -> int:
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

    @classmethod
    def _count_double_wound_triangle_pairs(cls, triangles: List[List[int] | Tuple[int, int, int]]) -> int:
        buckets: Dict[Tuple[int, int, int], List[int]] = {}
        for tri in triangles:
            tri3 = [int(value) for value in tri[:3]]
            if len(tri3) != 3 or len(set(tri3)) != 3:
                continue
            key = tuple(sorted(tri3))
            parity = cls._triangle_winding_parity(tri3)
            if key not in buckets:
                buckets[key] = [0, 0]
            buckets[key][parity] += 1
        return sum(min(counts[0], counts[1]) for counts in buckets.values())

    @classmethod
    def _collapse_double_wound_triangle_pairs(cls, triangles: List[List[int] | Tuple[int, int, int]]) -> List[List[int]]:
        if not triangles:
            return []

        counts: Dict[Tuple[int, int, int], List[int]] = {}
        preferred_parity: Dict[Tuple[int, int, int], int] = {}
        for tri in triangles:
            tri3 = [int(value) for value in tri[:3]]
            if len(tri3) != 3 or len(set(tri3)) != 3:
                continue
            key = tuple(sorted(tri3))
            parity = cls._triangle_winding_parity(tri3)
            counts.setdefault(key, [0, 0])[parity] += 1
            preferred_parity.setdefault(key, parity)

        pair_quota: Dict[Tuple[int, int, int], int] = {
            key: min(value[0], value[1])
            for key, value in counts.items()
            if min(value[0], value[1]) > 0
        }
        if not pair_quota:
            return [[int(value) for value in tri[:3]] for tri in triangles]

        consumed: Dict[Tuple[int, int, int], List[int]] = {key: [0, 0] for key in pair_quota}
        emitted_preferred: Dict[Tuple[int, int, int], int] = {key: 0 for key in pair_quota}
        collapsed: List[List[int]] = []

        for tri in triangles:
            tri3 = [int(value) for value in tri[:3]]
            if len(tri3) != 3 or len(set(tri3)) != 3:
                collapsed.append(tri3)
                continue
            key = tuple(sorted(tri3))
            quota = int(pair_quota.get(key, 0))
            if quota <= 0:
                collapsed.append(tri3)
                continue

            parity = cls._triangle_winding_parity(tri3)
            if consumed[key][parity] < quota:
                consumed[key][parity] += 1
                if parity == preferred_parity.get(key, parity) and emitted_preferred[key] < quota:
                    emitted_preferred[key] += 1
                    collapsed.append(tri3)
                continue

            collapsed.append(tri3)

        return collapsed

    @staticmethod
    def _propagate_tr8_double_wound_material_flags(strips: List[TextureStrip]) -> None:
        pair_counts_by_material: Dict[int, int] = {}
        for strip_index, strip in enumerate(strips):
            try:
                material_group = int(getattr(strip, 'material_group', -1))
            except Exception:
                material_group = -1
            if material_group < 0:
                material_group = int(strip_index)
            pair_count = int(getattr(strip, 'tr8_double_wound_pair_count', 0) or 0)
            if pair_count > 0:
                pair_counts_by_material[material_group] = pair_counts_by_material.get(material_group, 0) + pair_count
        if not pair_counts_by_material:
            return
        for strip_index, strip in enumerate(strips):
            try:
                material_group = int(getattr(strip, 'material_group', -1))
            except Exception:
                material_group = -1
            if material_group < 0:
                material_group = int(strip_index)
            pair_count = int(pair_counts_by_material.get(material_group, 0))
            if pair_count > 0:
                strip.tr8_double_sided = True
                strip.tr8_double_wound_pair_count = pair_count

    def _parse_triangle_strips(
        self,
        context: SectionContext,
        meshes: List[TR8MeshInfo],
        groups: List[TR8MeshGroupInfo],
        offset_face_data: int,
        num_indices: int,
    ) -> List[TextureStrip]:
        strips: List[TextureStrip] = []
        index_data_local = self._infer_face_index_data_local(context, meshes, groups, int(offset_face_data), int(num_indices))
        if index_data_local < 0 or not self._valid_local(context, index_data_local, int(num_indices) * 2):
            return strips

        br = context.reader
        index_data_abs = context.data_start + int(index_data_local)
        for group in groups:
            mesh = self._mesh_for_group(meshes, group)
            if mesh is None:
                continue
            index_count = int(group.num_faces) * 3
            if index_count <= 0:
                continue
            start = int(group.base_index) * 2
            if start < 0 or start + (index_count * 2) > int(num_indices) * 2:
                continue
            br.seek(index_data_abs + start)
            local_triangles: List[List[int]] = []
            for _ in range(int(group.num_faces)):
                tri = [int(br.u16()), int(br.u16()), int(br.u16())]
                local_triangles.append(tri)

            double_wound_pair_count = self._count_double_wound_triangle_pairs(local_triangles)
            if double_wound_pair_count > 0:
                local_triangles = self._collapse_double_wound_triangle_pairs(local_triangles)

            indices: List[int] = []
            for tri in local_triangles:
                indices.extend(int(mesh.global_vertex_start) + int(local_index) for local_index in tri[:3])

            strips.append(
                TextureStrip(
                    offset=int(group.base_index),
                    vertex_count=len(indices),
                    draw_group=int(group.draw_group_id),
                    tpageid=int(group.material_index),
                    sort_push=0.0,
                    scroll_offset=0.0,
                    indices=indices,
                    material_group=int(group.material_index),
                    source_file=context.file_name,
                    tr8_batch_index=int(mesh.index),
                    tr8_vertex_format_offset=int(mesh.vertex_components_offset),
                    tr8_palette_source_index=int(getattr(mesh, 'palette_source_index', -1)),
                    tr8_geometry_source_index=int(getattr(mesh, 'geometry_source_index', -1)),
                    tr8_double_sided=bool(double_wound_pair_count > 0),
                    tr8_double_wound_pair_count=int(double_wound_pair_count),
                )
            )
        self._propagate_tr8_double_wound_material_flags(strips)
        return strips

    def _parse_skeleton(self, context: SectionContext, expected_bones: int) -> List[Segment]:
        if expected_bones <= 0 or not context.section_info.relocations:
            return []
        br = context.reader
        previous = br.tell()
        try:
            # The last self-relocation in the observed TR8 mesh files points to
            # a skeleton-array ref.  The array header sits 16 bytes before that
            # ref target: uint count; uint bonesOffset; uint[2] reserved.
            candidates: List[int] = []
            for relocation in reversed(context.section_info.relocations):
                if int(relocation.section_index_or_type) not in {0, int(context.section_info.section_id), int(context.filepath.stem.split('_', 1)[0]) if '_' in context.filepath.stem else -1}:
                    continue
                field_local = int(relocation.offset)
                if not self._valid_local(context, field_local, 4):
                    continue
                value = self._read_u32_at(context, field_local)
                # Observed TR8 PC meshes use two closely related skeleton
                # reference layouts.  Most samples have a 16-byte array header
                # immediately before the relocated pointer target:
                #   uint count; uint bonesOffset; uint[2] reserved
                # The zipped-Lara style DRM instead stores only:
                #   uint count; uint bonesOffset
                # immediately before the bone records.  The relocation field
                # itself points at bonesOffset, so value - 8 / field - 4 is the
                # actual header.  Try both layouts; validation below rejects
                # non-skeleton candidates.
                for header_local in (int(value) - 16, int(value) - 8):
                    if self._valid_local(context, header_local, 8):
                        candidates.append(header_local)
                if self._valid_local(context, field_local - 16, 16):
                    candidates.append(field_local - 16)
                if self._valid_local(context, field_local - 4, 8):
                    candidates.append(field_local - 4)

            for header_local in candidates:
                br.seek(context.data_start + int(header_local))
                count = int(br.u32())
                bones_offset = int(br.u32())
                if count != int(expected_bones) or count <= 0 or count > 4096:
                    continue
                records_local = int(header_local) + 0x10
                if not self._valid_local(context, records_local, count * 64):
                    # Some variants store the bone record array at the pointer.
                    records_local = int(bones_offset)
                if not self._valid_local(context, records_local, count * 64):
                    continue

                segments: List[Segment] = []
                for index in range(count):
                    rec_abs = context.data_start + records_local + (index * 64)
                    br.seek(rec_abs + 0x20)
                    x = float(br.f32())
                    y = float(br.f32())
                    z = float(br.f32())
                    br.skip(8)
                    first_vertex = int(br.i16())
                    last_vertex = int(br.i16())
                    parent = int(br.i32())
                    br.i32()
                    segments.append(
                        Segment(
                            index=index,
                            min_v=(0.0, 0.0, 0.0, 0.0),
                            max_v=(0.0, 0.0, 0.0, 0.0),
                            pivot=(x, y, z, 1.0),
                            flags=0,
                            first_vertex=first_vertex,
                            last_vertex=last_vertex,
                            parent=parent,
                            hinfo=0,
                        )
                    )
                valid_parent_count = sum(1 for segment in segments if -1 <= int(segment.parent) < len(segments) and int(segment.parent) != int(segment.index))
                nonzero_pivot_count = sum(1 for segment in segments if abs(float(segment.pivot[0])) + abs(float(segment.pivot[1])) + abs(float(segment.pivot[2])) > 0.001)
                root_parent = int(segments[0].parent) if segments else 0
                if root_parent != -1 or valid_parent_count < max(1, len(segments) // 2) or nonzero_pivot_count < max(1, len(segments) // 8):
                    logger.debug(
                        'Rejected TR8 skeleton candidate in %s at header_local=0x%X records_local=0x%X: root_parent=%d validParents=%d nonzeroPivots=%d',
                        context.file_name,
                        header_local,
                        records_local,
                        root_parent,
                        valid_parent_count,
                        nonzero_pivot_count,
                    )
                    continue
                logger.debug('Parsed TR8 skeleton from %s: bones=%d header_local=0x%X records_local=0x%X', context.file_name, len(segments), header_local, records_local)
                return segments
            return []
        finally:
            br.seek(previous)

    def _find_vertex_format_before_stream(
        self,
        context: SectionContext,
        vertex_data_local: int,
        *,
        max_components: int = 32,
    ) -> int:
        vertex_data_local = int(vertex_data_local)
        if vertex_data_local <= 0:
            return -1
        for component_count in range(1, int(max_components) + 1):
            candidate = vertex_data_local - (0x20 + (component_count * 8))
            if candidate < 0:
                continue
            stride, components = self._parse_vertex_format(
                context,
                TR8MeshInfo(
                    index=-1,
                    vertex_components_offset=candidate,
                    num_vertices=0,
                    base_index=0,
                    num_faces=0,
                    submesh_count=0,
                    skin_map_size=0,
                    skin_map_offset=0,
                    vertex_offset=0,
                ),
            )
            if stride <= 0 or TR8_VERTEX_POSITION not in components:
                continue
            format_end = (candidate + 0x20 + (len(components) * 8) + 15) & ~15
            if format_end == vertex_data_local:
                return int(candidate)
        return -1

    def _infer_final_batch_geometry(
        self,
        context: SectionContext,
        meta: TR8MeshInfo,
        batch_groups: List[TR8MeshGroupInfo],
        offset_face_data: int,
    ) -> TR8MeshInfo | None:
        if not batch_groups:
            return None

        vertex_data_local = int(meta.vertex_offset)
        vertex_components_offset = self._find_vertex_format_before_stream(context, vertex_data_local)
        if vertex_components_offset < 0:
            logger.debug(
                'Skipping final TR8 batch metadata record %d: could not find vertex format before vertexOffset=0x%X',
                int(meta.index),
                vertex_data_local,
            )
            return None

        probe = TR8MeshInfo(
            index=int(meta.index),
            vertex_components_offset=int(vertex_components_offset),
            num_vertices=1,
            base_index=int(batch_groups[0].base_index),
            num_faces=sum(int(group.num_faces) for group in batch_groups),
            submesh_count=int(meta.submesh_count),
            skin_map_size=int(meta.skin_map_size),
            skin_map_offset=int(meta.skin_map_offset),
            vertex_offset=int(meta.vertex_offset),
        )
        stride, components = self._parse_vertex_format(context, probe)
        if stride <= 0 or not components:
            return None
        stream_start = self._infer_vertex_stream_start(context, probe, stride, components)
        if stream_start < 0:
            stream_start = vertex_data_local
        stride = self._infer_vertex_stride(context, probe, stride, components, stream_start)
        if stride <= 0:
            return None

        # The final vertex stream can run right up to the actual index buffer.
        # offsetFaceData may point at a 16-byte winding/normal prefix before the
        # indices, so allow that prefix-sized slack when estimating the count.
        available_bytes = max(0, (int(offset_face_data) + 0x10) - int(stream_start))
        inferred_vertices = available_bytes // int(stride)
        group_vertex_sum = sum(int(group.num_vertices) for group in batch_groups)
        max_group_vertices = max((int(group.num_vertices) for group in batch_groups), default=0)
        if inferred_vertices <= 0:
            inferred_vertices = group_vertex_sum
        if group_vertex_sum > inferred_vertices:
            inferred_vertices = group_vertex_sum
        if max_group_vertices > inferred_vertices:
            inferred_vertices = max_group_vertices

        if inferred_vertices <= 0:
            return None

        logger.debug(
            'Inferred final TR8 render batch from metadata record %d: vertexFormat=0x%X vertexStream=0x%X stride=%d vertices=%d groups=%d faces=%d',
            int(meta.index),
            int(vertex_components_offset),
            int(stream_start),
            int(stride),
            int(inferred_vertices),
            len(batch_groups),
            sum(int(group.num_faces) for group in batch_groups),
        )
        return TR8MeshInfo(
            index=int(meta.index),
            vertex_components_offset=int(vertex_components_offset),
            num_vertices=int(inferred_vertices),
            base_index=int(batch_groups[0].base_index),
            num_faces=sum(int(group.num_faces) for group in batch_groups),
            submesh_count=int(meta.submesh_count),
            skin_map_size=int(meta.skin_map_size),
            skin_map_offset=int(meta.skin_map_offset),
            vertex_offset=int(meta.vertex_offset),
            palette_source_index=int(meta.index),
            geometry_source_index=-1,
        )

    def _build_render_batches(
        self,
        context: SectionContext,
        mesh_records: List[TR8MeshInfo],
        groups: List[TR8MeshGroupInfo],
        offset_face_data: int,
    ) -> List[TR8MeshInfo]:
        batches: List[TR8MeshInfo] = []
        group_cursor = 0
        group_total = len(groups)
        for record_index, meta in enumerate(mesh_records):
            group_count = max(0, int(meta.submesh_count))
            if group_count <= 0:
                continue
            if group_cursor >= group_total:
                break

            actual_group_count = min(group_count, group_total - group_cursor)
            batch_groups = groups[group_cursor:group_cursor + actual_group_count]
            group_face_sum = sum(int(group.num_faces) for group in batch_groups)

            geometry: TR8MeshInfo | None = None
            if record_index + 1 < len(mesh_records):
                candidate = mesh_records[record_index + 1]
                if int(candidate.num_vertices) > 0 and int(candidate.num_faces) > 0 and int(candidate.vertex_components_offset) > 0:
                    geometry = candidate
            else:
                geometry = self._infer_final_batch_geometry(context, meta, batch_groups, int(offset_face_data))

            if geometry is None:
                logger.debug(
                    'Skipping TR8 batch metadata record %d: groups=%d has no usable geometry record',
                    int(meta.index),
                    group_count,
                )
                group_cursor += group_count
                continue

            if group_face_sum != int(geometry.num_faces):
                logger.debug(
                    'TR8 batch %d face-count mismatch: groups %d..%d sum=%d geometry record %d faces=%d',
                    int(meta.index),
                    group_cursor,
                    group_cursor + actual_group_count - 1,
                    group_face_sum,
                    int(geometry.index),
                    int(geometry.num_faces),
                )

            batches.append(TR8MeshInfo(
                index=int(meta.index),
                vertex_components_offset=int(geometry.vertex_components_offset),
                num_vertices=int(geometry.num_vertices),
                base_index=int(batch_groups[0].base_index if batch_groups else geometry.base_index),
                num_faces=int(group_face_sum or geometry.num_faces),
                submesh_count=int(meta.submesh_count),
                skin_map_size=int(meta.skin_map_size),
                skin_map_offset=int(meta.skin_map_offset),
                vertex_offset=int(meta.vertex_offset),
                group_start=group_cursor,
                group_count=actual_group_count,
                palette_source_index=int(meta.index),
                geometry_source_index=int(getattr(geometry, 'geometry_source_index', geometry.index) if int(getattr(geometry, 'geometry_source_index', geometry.index)) >= 0 else -1),
            ))
            group_cursor += group_count

        if batches:
            covered_groups = sum(max(0, int(batch.group_count)) for batch in batches)
            logger.debug(
                'Built %d TR8 render batches from %d mesh records and %d mesh groups; coveredGroups=%d/%d',
                len(batches),
                len(mesh_records),
                len(groups),
                covered_groups,
                len(groups),
            )
        return batches


    @staticmethod
    def _find_ps2_mesh_header_local(context: SectionContext) -> int:
        # Underworld PS2 cdcModelData sections use a compact binary header with
        # the marker 0xFFEEDDCC at local offset 0x10 in all samples observed so
        # far.  Keep a small scan fallback so extracted/repacked sections with
        # a different pointer prefix still parse.
        if context.data_size >= 0x70 and TR8ModelParser._read_u32_at(context, 0x10) == 0xFFEEDDCC:
            return 0x10
        limit = min(int(context.data_size) - 0x10, 0x200)
        for local in range(0, max(0, limit), 4):
            if TR8ModelParser._read_u32_at(context, local) != 0xFFEEDDCC:
                continue
            if TR8ModelParser._read_u32_at(context, local + 0x0C, 0) == 0x60:
                return int(local)
        return -1

    @staticmethod
    def _tr8ps2_vif_unpack_element_size(cmd: int) -> tuple[int, int] | None:
        if not (0x60 <= int(cmd) <= 0x6F):
            return None
        vn = (int(cmd) >> 2) & 0x3
        vl = int(cmd) & 0x3
        components = (1, 2, 3, 4)[vn]
        if vl == 0:
            scalar_size = 4
        elif vl == 1:
            scalar_size = 2
        elif vl == 2:
            scalar_size = 1
        else:
            return None
        return components, scalar_size

    @staticmethod
    def _tr8ps2_normal_s12_to_byte(normal: tuple[int, int, int]) -> tuple[int, int, int]:
        nx, ny, nz = (float(normal[0]), float(normal[1]), float(normal[2]))
        length = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
        if length <= 1e-8:
            return (0, 0, 127)
        return (
            max(-127, min(127, int(round((nx / length) * 127.0)))),
            max(-127, min(127, int(round((ny / length) * 127.0)))),
            max(-127, min(127, int(round((nz / length) * 127.0)))),
        )

    @staticmethod
    def _tr8ps2_matrix_slot_from_flag(flag: int) -> int:
        matrix_word = int(flag) & 0x3FFF
        if matrix_word % 12 == 0:
            return matrix_word // 12
        return matrix_word

    @staticmethod
    def _tr8ps2_strip_flag_adc(flag: int) -> bool:
        return (int(flag) & 0x8000) != 0

    @staticmethod
    def _tr8ps2_strip_flag_flip(flag: int) -> bool:
        return (int(flag) & 0x4000) != 0

    def _parse_tr8ps2_vif_chunk(self, context: SectionContext, start_local: int, max_bytes: int = 0x4000) -> dict | None:
        if not self._valid_local(context, start_local, 16):
            return None
        start_abs = context.data_start + int(start_local)
        end_abs = min(context.data_end, start_abs + max(0x40, int(max_bytes)))
        br = context.reader
        previous = br.tell()
        streams = {
            'sort_key': 0.0,
            'positions': [],
            'colors': [],
            'normals': [],
            'uvs': [],
            'start_local': int(start_local),
            'end_local': int(start_local),
        }
        try:
            pos = start_abs
            safety = 0
            saw_positions = False
            saw_uvs = False
            while pos + 4 <= end_abs and safety < 32:
                safety += 1
                br.seek(pos)
                word = int(br.u32())
                if word == 0:
                    break
                cmd = (word >> 24) & 0xFF
                num = (word >> 16) & 0xFF
                unpack = self._tr8ps2_vif_unpack_element_size(cmd)
                if unpack is None or num <= 0:
                    break
                components, scalar_size = unpack
                data_len = int(num) * int(components) * int(scalar_size)
                data_len_padded = (data_len + 3) & ~3
                payload_abs = pos + 4
                if payload_abs + data_len > end_abs or payload_abs + data_len > context.file_size:
                    break
                payload = br.peek(data_len, offset=payload_abs)

                if cmd == 0x64 and num == 1 and data_len >= 8:
                    try:
                        streams['sort_key'] = float(struct.unpack_from('<f', payload, 0)[0])
                    except Exception:
                        streams['sort_key'] = 0.0
                elif cmd == 0x6D:
                    streams['positions'] = [struct.unpack_from('<4h', payload, i * 8) for i in range(num)]
                    saw_positions = True
                elif cmd == 0x6E:
                    streams['colors'] = [tuple(int(v) for v in payload[i * 4:i * 4 + 4]) for i in range(num)]
                elif cmd == 0x69:
                    streams['normals'] = [struct.unpack_from('<3h', payload, i * 6) for i in range(num)]
                elif cmd == 0x65:
                    streams['uvs'] = [struct.unpack_from('<2h', payload, i * 4) for i in range(num)]
                    saw_uvs = True

                pos = payload_abs + data_len_padded
                streams['end_local'] = int(pos - context.data_start)
                if saw_positions and saw_uvs:
                    # Observed PS2 Underworld packets use exactly these five
                    # unpack blocks.  Stopping at UV prevents unrelated VIF-like
                    # words from being folded into the chunk.
                    break
            if not streams.get('positions'):
                return None
            return streams
        finally:
            br.seek(previous)

    def _tr8ps2_parse_segments_from_records(self, context: SectionContext, start_local: int, count: int) -> list[Segment]:
        if count <= 0 or count > 4096 or not self._valid_local(context, start_local, int(count) * 64):
            return []
        segments: list[Segment] = []
        for index in range(int(count)):
            base = int(start_local) + (index * 64)
            min_v = self._read_vec4_at(context, base + 0x00)
            max_v = self._read_vec4_at(context, base + 0x10)
            pivot = self._read_vec4_at(context, base + 0x20)
            if not all(math.isfinite(float(v)) and abs(float(v)) < 1.0e7 for v in (*min_v, *max_v, *pivot)):
                return []
            first = self._read_i16_at(context, base + 0x38, -1)
            last = self._read_i16_at(context, base + 0x3A, -1)
            raw_parent = self._read_i32_at(context, base + 0x3C, -1)
            if index == 0 and first < 0:
                parent = -1
            elif -1 <= int(first) < index:
                parent = int(first)
            elif -1 <= int(raw_parent) < index:
                parent = int(raw_parent)
            else:
                parent = -1 if index == 0 else 0
            segments.append(Segment(
                index=index,
                min_v=min_v,
                max_v=max_v,
                pivot=pivot,
                flags=0,
                first_vertex=-1,
                last_vertex=-1,
                parent=parent,
                hinfo=-1,
            ))
        return segments

    def _tr8ps2_score_segment_table(self, segments: list[Segment]) -> float:
        if not segments:
            return -1.0
        nonzero_pivots = sum(1 for segment in segments if any(abs(float(v)) > 1.0e-6 for v in segment.pivot[:3]))
        valid_parent_count = 0
        root_count = 0
        for segment in segments:
            parent = int(segment.parent)
            if parent < 0:
                root_count += 1
                valid_parent_count += 1
            elif 0 <= parent < int(segment.index):
                valid_parent_count += 1
        # Prefer real record tables over pointer/metadata blocks that happen to
        # satisfy the broad parent checks.  A useful skeleton generally has a
        # single root and non-zero pivots for most child bones.
        score = float(valid_parent_count)
        score += float(nonzero_pivots) * 4.0
        if root_count == 1:
            score += 16.0
        elif root_count == 0:
            score -= 16.0
        else:
            score -= float(root_count)
        return score

    def _parse_tr8ps2_skeleton(self, context: SectionContext, header_local: int) -> list[Segment]:
        packed_count = self._read_u32_at(context, int(header_local) + 0x48, 0)
        header_bone_count = (int(packed_count) >> 16) & 0xFFFF
        group_count = int(packed_count) & 0xFFFF
        header_count_2 = self._read_u32_at(context, int(header_local) + 0x4C, 0)
        root_pointers = [self._read_u32_at(context, local, 0) for local in (0x00, 0x04, 0x08, 0x0C)]
        candidates: list[tuple[int, int, int]] = []

        def add_candidate(start: int, count: int, priority: int) -> None:
            start = int(start)
            count = int(count)
            if start <= 0 or count <= 0:
                return
            if not self._valid_local(context, start, count * 64):
                return
            item = (priority, start, count)
            if item not in candidates:
                candidates.append(item)

        # Root pointer 3 points at a compact metadata block in the smaller PS2
        # Underworld samples.  Its fourth dword is the 64-byte segment table.
        root3 = root_pointers[3] if len(root_pointers) > 3 else 0
        if self._valid_local(context, root3, 0x10):
            meta_count = self._read_u32_at(context, int(root3) + 0x08, 0)
            meta_records = self._read_u32_at(context, int(root3) + 0x0C, 0)
            for count in (header_bone_count, meta_count, header_count_2):
                add_candidate(meta_records, count, 0)

        # In PS2 Underworld samples seen so far, the canonical skeleton table
        # is often referenced by a compact {count, records_ptr} pair elsewhere
        # in the mesh section.  The previous heuristic missed this in the Yeti
        # Thrall sample and instead accepted the per-part helper table at root3,
        # producing a scrambled armature.  Scan for an exact bone_count/pointer
        # pair before falling back to broader pointer-region guesses.
        try:
            previous_pos = context.reader.tell()
            context.reader.seek(context.data_start)
            data = context.reader.read(context.data_size)
        finally:
            try:
                context.reader.seek(previous_pos)
            except Exception:
                pass
        for count in (header_count_2, header_bone_count):
            count = int(count)
            if count <= 0 or count > 4096:
                continue
            for local in range(0, max(0, len(data) - 8), 4):
                candidate_count = int.from_bytes(data[local:local + 4], 'little', signed=False)
                if candidate_count != count:
                    continue
                records = int.from_bytes(data[local + 4:local + 8], 'little', signed=False)
                add_candidate(records, count, -4)

        # Some larger Lara sections use a deeper metadata block; the records are
        # commonly adjacent to dword pointers in the final root-pointer region.
        for pointer in root_pointers:
            if self._valid_local(context, pointer, 0x20):
                for rel in range(0, 0x80, 4):
                    current_local = int(pointer) + rel
                    nested = self._read_u32_at(context, current_local, 0)
                    next_value = self._read_u32_at(context, current_local + 4, 0)
                    # Metadata often stores {count, records_ptr}; this catches
                    # the larger Lara sample where count=0x79 and records_ptr
                    # follows immediately in a nested block.
                    if 0 < int(nested) <= 4096:
                        add_candidate(next_value, int(nested), 0)
                    if self._valid_local(context, nested, 0x20):
                        for nested_rel in range(0, 0x80, 4):
                            nested_count = self._read_u32_at(context, int(nested) + nested_rel, 0)
                            nested_records = self._read_u32_at(context, int(nested) + nested_rel + 4, 0)
                            if 0 < int(nested_count) <= 4096:
                                add_candidate(nested_records, int(nested_count), -1)
                    for count in (header_bone_count, header_count_2, group_count):
                        add_candidate(nested, count, 1)
                for count in (header_bone_count, header_count_2, group_count):
                    add_candidate(pointer, count, 2)

        best_segments: list[Segment] = []
        best_score = -1.0
        for priority, start, count in candidates:
            segments = self._tr8ps2_parse_segments_from_records(context, start, count)
            score = self._tr8ps2_score_segment_table(segments)
            if score < 0:
                continue
            # Priority keeps direct metadata matches ahead of broad pointer scans
            # when scores tie.
            score -= float(priority) * 0.25
            if score > best_score:
                best_score = score
                best_segments = segments
        if best_segments:
            logger.debug('Parsed TR8 PS2 skeleton from %s: bones=%d score=%.2f', context.file_name, len(best_segments), best_score)
        return best_segments

    def _parse_tr8ps2_blend_records(self, context: SectionContext, header_local: int) -> list[list[tuple[int, float]]]:
        # The first table after the compact PS2 mesh header is a skin-blend
        # table.  Each 8-byte record is four u8 bone ids followed by four u8
        # weights.  The active VIF packets reference these records through their
        # chunk-local matrix/blend palette.
        packed_count = self._read_u32_at(context, int(header_local) + 0x48, 0)
        blend_count = (int(packed_count) >> 16) & 0xFFFF
        bone_count = self._read_u32_at(context, int(header_local) + 0x4C, 0)
        table_local = int(header_local) + 0x60
        if blend_count <= 0 or blend_count > 4096 or bone_count <= 0:
            return []
        if not self._valid_local(context, table_local, int(blend_count) * 8):
            return []

        records: list[list[tuple[int, float]]] = []
        for record_index in range(int(blend_count)):
            base = table_local + (record_index * 8)
            bones = [self._read_u8_at(context, base + i, 0) for i in range(4)]
            weights = [self._read_u8_at(context, base + 4 + i, 0) for i in range(4)]
            pairs: list[tuple[int, int]] = []
            for bone, weight in zip(bones, weights):
                bone = int(bone)
                weight = int(weight)
                if weight <= 0:
                    continue
                if 0 <= bone < int(bone_count):
                    pairs.append((bone, weight))
            total = sum(weight for _, weight in pairs)
            if total <= 0:
                records.append([])
                continue
            records.append([(bone, float(weight) / float(total)) for bone, weight in pairs])
        logger.debug(
            'Parsed TR8 PS2 blend table from %s: records=%d bones=%d',
            context.file_name,
            len(records),
            int(bone_count),
        )
        return records

    @staticmethod
    def _tr8ps2_find_vif_preamble(data: bytes, start_local: int, vertex_count: int) -> tuple[list[int], list[int]]:
        # A render chunk has a small CPU/VU preamble immediately before the VIF
        # attribute unpacks.  The preamble repeats the first sort-key float, then at
        # +0x14 stores: u8 vertexCount, u8 unknown/destination, u8 paletteCount,
        # followed by paletteCount u8 blend-record references.  At +0x20 it also
        # stores the VU destination/source ids for the packet vertices.  These ids
        # are useful for research but the skinning comes from the blend table.
        start_local = int(start_local)
        vertex_count = int(vertex_count)
        if start_local < 12 or start_local + 12 > len(data):
            return [], []
        sort_key = data[start_local + 4:start_local + 8]
        best = -1
        for local in range(max(0, start_local - 0x300), start_local, 4):
            if data[local:local + 4] != sort_key:
                continue
            if local + 0x20 > start_local:
                continue
            pre_vertex_count = data[local + 0x14]
            palette_count = data[local + 0x16]
            if pre_vertex_count != vertex_count:
                continue
            if not (0 < palette_count <= 32):
                continue
            if local + 0x17 + palette_count > start_local:
                continue
            best = int(local)
        if best < 0:
            return [], []

        palette_count = int(data[best + 0x16])
        palette = [int(v) for v in data[best + 0x17:best + 0x17 + palette_count]]
        source_ids: list[int] = []
        ids_start = best + 0x20
        ids_end = ids_start + (vertex_count * 2)
        if ids_end <= start_local:
            source_ids = [int(struct.unpack_from('<H', data, ids_start + (i * 2))[0]) for i in range(vertex_count)]
        return palette, source_ids


    @staticmethod
    def _tr8ps2_segment_global_pivots(segments: list[Segment]) -> list[tuple[float, float, float]]:
        globals_: list[tuple[float, float, float]] = []
        for segment in segments or []:
            px, py, pz = float(segment.pivot[0]), float(segment.pivot[1]), float(segment.pivot[2])
            parent = int(segment.parent)
            if 0 <= parent < len(globals_):
                gx, gy, gz = globals_[parent]
                globals_.append((gx + px, gy + py, gz + pz))
            else:
                globals_.append((px, py, pz))
        return globals_

    @staticmethod
    def _tr8ps2_make_spatial_bone_map(samples: dict[int, list[float]], segments: list[Segment], label: str) -> dict[int, int]:
        globals_ = TR8ModelParser._tr8ps2_segment_global_pivots(segments)
        if not globals_:
            return {}
        result: dict[int, int] = {}
        for node, accum in (samples or {}).items():
            if len(accum) < 4 or float(accum[3]) <= 0.0:
                continue
            cx = float(accum[0]) / float(accum[3])
            cy = float(accum[1]) / float(accum[3])
            cz = float(accum[2]) / float(accum[3])
            if len(accum) >= 7:
                min_x = float(accum[5])
                max_x = float(accum[6])
                # Some PS2 blend nodes are shared across both sides of the body.
                # Their weighted centroid can drift left or right depending on
                # how much surface area was sampled.  If the node clearly spans
                # both sides, score it as a centre-line node instead of letting it
                # collapse onto a thigh/arm solely because the mesh is asymmetric.
                if min_x < -25.0 and max_x > 25.0 and abs(cx) < 55.0:
                    cx = 0.0

            best_index = -1
            best_score = float('inf')
            for index, (gx, gy, gz) in enumerate(globals_):
                segment = segments[index] if 0 <= index < len(segments) else None
                if segment is not None:
                    # The recovered PS2 segment table includes root / helper
                    # placeholders that are valid for the hierarchy but should
                    # not receive skinned vertices.
                    if int(segment.parent) < 0:
                        continue
                    if index > 0 and abs(gx) < 1e-5 and abs(gy) < 1e-5 and abs(gz) < 1e-5:
                        continue
                dx = cx - gx
                dy = cy - gy
                dz = cz - gz
                score = (dx * dx) + (dy * dy) + (dz * dz)
                # Left/right mismatches are almost always wrong for arms, hands,
                # and legs.  Keep centre-line bones eligible by only applying the
                # penalty when both positions are clearly off centre.
                if abs(cx) > 18.0 and abs(gx) > 18.0 and ((cx < 0.0) != (gx < 0.0)):
                    score += 250000.0
                if score < best_score:
                    best_score = score
                    best_index = index
            if best_index >= 0:
                result[int(node)] = int(best_index)

        if result:
            logger.info('Derived TR8 PS2 %s skin-node remap entries=%d', str(label), len(result))
        return result

    @staticmethod
    def _tr8ps2_normalize_weights(weights: list[tuple[int, float]]) -> list[tuple[int, float]]:
        totals: dict[int, float] = {}
        for bone, weight in weights or []:
            weight = float(weight)
            if weight <= 0.0:
                continue
            totals[int(bone)] = totals.get(int(bone), 0.0) + weight
        total = sum(totals.values())
        if total <= 0.0:
            return []
        return [(int(bone), float(weight) / float(total)) for bone, weight in sorted(totals.items())]

    def _parse_tr8ps2_texture_ids(self, context: SectionContext) -> list[int]:
        texture_list_local = int(self._read_u32_at(context, 0x04, 0))
        if not self._valid_local(context, texture_list_local, 4):
            return []
        count = int(self._read_u32_at(context, texture_list_local, 0))
        if count <= 0 or count > 256:
            return []
        if not self._valid_local(context, texture_list_local + 4, count * 4):
            return []
        texture_ids: list[int] = []
        for index in range(count):
            texture_id = int(self._read_u32_at(context, texture_list_local + 4 + (index * 4), -1))
            if texture_id < 0:
                continue
            if texture_id not in texture_ids:
                texture_ids.append(texture_id)
        return texture_ids

    def _tr8ps2_texture_metadata_by_id(self, context: SectionContext, texture_ids: list[int]) -> dict[int, dict]:
        if parse_tr8_ps2_pcd_bytes is None or not texture_ids:
            return {}
        wanted = {int(texture_id) for texture_id in texture_ids}
        result: dict[int, dict] = {}
        try:
            directory = Path(context.filepath).parent
            for path in directory.glob('*.pcd'):
                try:
                    stem = path.stem
                    texture_id = int(stem.split('_', 1)[1], 16) if '_' in stem else int(stem, 16)
                except Exception:
                    continue
                if texture_id not in wanted or texture_id in result:
                    continue
                try:
                    pcd_bytes = path.read_bytes()
                    parsed = parse_tr8_ps2_pcd_bytes(pcd_bytes)
                except Exception:
                    parsed = None
                    pcd_bytes = b''
                if parsed is None:
                    continue

                has_alpha = False
                alpha_coverage = 0.0
                try:
                    if decode_supported_ps2_texture is not None:
                        decoded = decode_supported_ps2_texture(pcd_bytes)
                        if decoded is not None and decoded.get('supported', False):
                            has_alpha = bool(decoded.get('has_alpha', False))
                            pixels = decoded.get('pixels') or []
                            if pixels:
                                alpha_values = [float(pixels[index]) for index in range(3, len(pixels), 4)]
                                if alpha_values:
                                    alpha_coverage = sum(1 for alpha in alpha_values if alpha > 0.01) / float(len(alpha_values))
                except Exception:
                    has_alpha = False
                    alpha_coverage = 0.0

                result[int(texture_id)] = {
                    'width': int(parsed.width),
                    'height': int(parsed.height),
                    'area': int(parsed.width) * int(parsed.height),
                    'format_id': int(parsed.format_id),
                    'bits_per_pixel': int(parsed.bits_per_pixel),
                    'has_alpha': bool(has_alpha),
                    'alpha_coverage': float(alpha_coverage),
                }
        except Exception:
            logger.debug('Unable to read TR8 PS2 texture metadata beside %s', context.file_name, exc_info=True)
        return result

    def _tr8ps2_diffuse_texture_slots(self, context: SectionContext, texture_ids: list[int]) -> list[int]:
        slots = list(range(len(texture_ids)))
        if len(texture_ids) < 3:
            return slots

        metadata = self._tr8ps2_texture_metadata_by_id(context, texture_ids)
        if not metadata:
            return slots

        skip_slots: set[int] = set()
        first_tail_slot = max(1, len(texture_ids) - 3)
        for slot in range(first_tail_slot, len(texture_ids) - 1):
            current = metadata.get(int(texture_ids[slot]))
            nxt = metadata.get(int(texture_ids[slot + 1]))
            if not current or not nxt:
                continue
            area = int(current.get('area', 0) or 0)
            next_area = int(nxt.get('area', 0) or 0)
            if area <= 0 or next_area <= 0:
                continue
            # Observed auxiliary slots in the supplied PS2 TR8 samples are
            # tail entries: 64x64 or smaller lookup/reflection maps immediately
            # before substantially larger 128/256px diffuse maps.  Restricting
            # this to the tail avoids removing ordinary early small diffuse
            # textures such as backpack, eye, hair, and accessory pieces.
            if area <= 0x1000 and next_area >= 0x4000 and next_area >= area * 8:
                skip_slots.add(int(slot))

        if not skip_slots:
            return slots

        filtered = [slot for slot in slots if slot not in skip_slots]
        logger.info(
            'TR8 PS2 %s filtered auxiliary texture slots from diffuse material list: %s',
            context.file_name,
            ', '.join(f'{slot}:0x{int(texture_ids[slot]):X}' for slot in sorted(skip_slots)),
        )
        return filtered or slots

    def _parse_tr8ps2_material_runs(self, context: SectionContext, header_local: int, texture_ids: list[int]) -> list[dict]:
        packed_count = int(self._read_u32_at(context, int(header_local) + 0x48, 0))
        run_count = packed_count & 0xFFFF
        table_base = int(self._read_u32_at(context, int(header_local) + 0x08, 0))
        table_local = int(header_local) + table_base + 0x20

        if run_count <= 0 or run_count > 4096 or table_base <= 0:
            return []
        if not self._valid_local(context, table_local, 0x10):
            return []

        diffuse_slots = self._tr8ps2_diffuse_texture_slots(context, texture_ids)
        texture_metadata = self._tr8ps2_texture_metadata_by_id(context, texture_ids)

        chunk_starts: list[int] = []
        try:
            previous_pos = context.reader.tell()
            try:
                data = context.reader.peek(context.data_size, offset=context.data_start)
            finally:
                context.reader.seek(previous_pos)
            for local in range(0, max(0, len(data) - 16), 4):
                if struct.unpack_from('<I', data, local)[0] != 0x64018001:
                    continue
                if ((struct.unpack_from('<I', data, local + 12)[0] >> 24) & 0xFF) != 0x6D:
                    continue
                chunk_starts.append(int(local))
        except Exception:
            chunk_starts = []
        # Some PS2 TR8 material runs use a paired texture-page setup: the run
        # carries a primary material value plus a secondary material value and a
        # small stage/control word (observed 0x842).  The next material value in
        # the run table can then alias back to that secondary texture instead of
        # advancing to the next raw texture-list slot.  Lara eye materials are
        # the clearest supplied case: the eyelash/eye-page run is value 5+6,
        # while the following eye draw uses value 7 but still resolves to the
        # texture referenced by secondary value 6.  Keep this as a value->slot
        # remap while walking the already sorted run table.
        # First collect enough context to choose the PS2 material-indexing mode.
        # The early Lara PS2 tables use material value 0 as the first/root
        # texture and then use several low positive values as absolute texture
        # slots.  Tables that never reference 0 behave as the older one-based
        # material list.  A paired 0x842 stage can also alias only the next draw
        # run; it is not a global value remap.
        has_primary_zero = False
        for probe_run_index in range(int(run_count)):
            probe_base = table_local + (probe_run_index * 0x28)
            if not self._valid_local(context, probe_base, 0x10):
                break
            probe_start = int(self._read_u32_at(context, probe_base + 0x04, 0))
            probe_primary = int(self._read_u32_at(context, probe_base + 0x08, 0xFFFFFFFF))
            probe_secondary = int(self._read_u32_at(context, probe_base + 0x0C, 0xFFFFFFFF))
            probe_flags = int(self._read_u32_at(context, probe_base + 0x14, 0))
            if probe_start <= 0 or not self._valid_local(context, probe_start, 1):
                continue
            if probe_primary == 0 and not (probe_secondary == 0 and probe_flags == 0):
                has_primary_zero = True
                break

        pending_stage_alias_value: int | None = None
        pending_stage_alias_slot: int | None = None

        def _material_value_valid(value: int) -> bool:
            value = int(value)
            if value == 0xFFFFFFFF or value < 0:
                return True
            if value == 0:
                return bool(texture_ids)
            # Positive PS2 material values are interpreted against the compact
            # diffuse-material list.  Some meshes store one trailing sentinel
            # record after the real run list; its material fields are floats,
            # pointers, or unrelated dwords.  Reject those here instead of
            # allowing the fallback mapping to turn them into real materials.
            if 0 <= (value - 1) < len(diffuse_slots):
                return True
            if 0 <= (value - 1) < len(texture_ids):
                return True
            return False

        runs: list[dict] = []
        for run_index in range(int(run_count)):
            base = table_local + (run_index * 0x28)
            if not self._valid_local(context, base, 0x10):
                break
            start_local = int(self._read_u32_at(context, base + 0x04, 0))
            primary_index = int(self._read_u32_at(context, base + 0x08, 0xFFFFFFFF))
            secondary_index = int(self._read_u32_at(context, base + 0x0C, 0xFFFFFFFF))
            run_flags = int(self._read_u32_at(context, base + 0x14, 0))
            stage_control = int(self._read_u32_at(context, base + 0x1C, 0))
            stage_variant = int(self._read_u32_at(context, base + 0x20, 0))
            if start_local <= 0 or not self._valid_local(context, start_local, 1):
                continue
            if not _material_value_valid(primary_index) or not _material_value_valid(secondary_index):
                logger.debug(
                    'TR8 PS2 %s skipped invalid material-run sentinel #%d: start=0x%X primary=0x%X secondary=0x%X flags=0x%X',
                    context.file_name,
                    int(run_index),
                    int(start_local),
                    int(primary_index),
                    int(secondary_index),
                    int(run_flags),
                )
                continue

            # The PS2 draw-run material fields are not zero-based texture-list
            # indices.  In the observed Underworld PS2 meshes, material value 0
            # is the default first texture, while positive values are one-based
            # texture slots.  Treating value 1 as texture_ids[1] made the main
            # tiger body use its tiny auxiliary map and made Lara's lower-body
            # run use the metal/detail map.
            stage_indices: list[int] = []

            def _slot_looks_like_auxiliary_after_diffuse(raw_slot: int) -> bool:
                raw_slot = int(raw_slot)
                if raw_slot <= 0 or raw_slot >= len(texture_ids):
                    return False
                current = texture_metadata.get(int(texture_ids[raw_slot]))
                previous = texture_metadata.get(int(texture_ids[raw_slot - 1]))
                if not current or not previous:
                    return False
                area = int(current.get('area', 0) or 0)
                prev_area = int(previous.get('area', 0) or 0)
                if area <= 0 or prev_area <= 0:
                    return False
                return area <= 0x1000 and prev_area >= 0x4000 and prev_area >= area * 8

            def _one_based_texture_slot_from_material_value(value: int) -> int | None:
                value = int(value)
                if value == 0:
                    return 0 if texture_ids else None
                if value == 0xFFFFFFFF or value < 0:
                    return None
                diffuse_index = value - 1
                if 0 <= diffuse_index < len(diffuse_slots):
                    return int(diffuse_slots[diffuse_index])
                raw_slot = value - 1
                if 0 <= raw_slot < len(texture_ids):
                    return int(raw_slot)
                return None

            def _texture_slot_from_material_value(value: int, *, paired_stage: bool = False, allow_pending_alias: bool = True) -> int | None:
                nonlocal pending_stage_alias_value, pending_stage_alias_slot
                value = int(value)
                if value == 0xFFFFFFFF or value < 0:
                    return None
                if value == 0:
                    return 0 if texture_ids else None

                if (
                    allow_pending_alias
                    and pending_stage_alias_value is not None
                    and int(value) == int(pending_stage_alias_value)
                    and pending_stage_alias_slot is not None
                    and 0 <= int(pending_stage_alias_slot) < len(texture_ids)
                    # Runs with the high 0x00200000 state bit return to the
                    # normal texture-slot namespace.  This is what separates
                    # Lara's iris draw from the following hair/eyelid draw, even
                    # though both use material value 7.
                    and not (int(run_flags) & 0x00200000)
                ):
                    alias_slot = int(pending_stage_alias_slot)
                    pending_stage_alias_value = None
                    pending_stage_alias_slot = None
                    return alias_slot

                if paired_stage:
                    return _one_based_texture_slot_from_material_value(value)

                if has_primary_zero:
                    # Zero-inclusive TR8 PS2 material tables use ordinary
                    # primary material values as zero-based entries in the
                    # mesh-local texture list.  The earlier builds only applied
                    # this to a few low values, which fixed some Lara head runs
                    # but shifted later body/outfit materials back to the wrong
                    # one-based slots.  Keep the special paired-stage path above
                    # for 0x842 eye/lash records; ordinary records stay
                    # zero-based here.
                    raw_slot = value
                    if 0 <= raw_slot < len(texture_ids):
                        # Some two-texture meshes include a tiny lookup/detail
                        # texture directly after a large diffuse map.  Those
                        # are not intended as the visible base texture for a
                        # primary material value, so retain the conservative
                        # fallback used for tiger/merc-style cases.
                        if _slot_looks_like_auxiliary_after_diffuse(raw_slot):
                            return raw_slot - 1
                        return int(raw_slot)

                return _one_based_texture_slot_from_material_value(value)

            paired_stage_record = int(stage_control) == 0x842
            for candidate in (primary_index, secondary_index):
                slot = _texture_slot_from_material_value(candidate, paired_stage=paired_stage_record)
                if slot is not None and slot not in stage_indices:
                    stage_indices.append(int(slot))
            if not stage_indices and texture_ids:
                stage_indices.append(0)

            diffuse_slot = int(stage_indices[0]) if stage_indices else -1
            stage_blend_mode = ''

            # Paired material-stage runs should keep the primary texture as the
            # visible diffuse stage.  The previous preview build promoted the
            # secondary stage for Lara's eye pair (for example B1+B2 -> B2),
            # which put the iris texture on the eyelash geometry.  The secondary
            # is real data, but it is not the base diffuse texture for this run.
            # Preserve it as metadata and use it to resolve the following alias
            # material value where observed.
            if len(stage_indices) >= 2 and int(stage_control) == 0x842:
                try:
                    secondary_slot_for_alias = int(stage_indices[1])
                    alias_value = int(secondary_index) + 1
                    if 0 <= secondary_slot_for_alias < len(texture_ids) and alias_value > 0:
                        # This alias is consumed by the next matching run only.
                        # Keeping it global made later material-7 runs inherit
                        # the iris texture, which put eye texture data on Lara's
                        # eyelid/hair geometry.
                        pending_stage_alias_value = int(alias_value)
                        pending_stage_alias_slot = int(secondary_slot_for_alias)
                        logger.info(
                            'TR8 PS2 %s material run #%d paired stage primary=0x%X secondary=0x%X; next material value %d aliases to secondary',
                            context.file_name,
                            int(run_index),
                            int(texture_ids[int(stage_indices[0])]),
                            int(texture_ids[secondary_slot_for_alias]),
                            int(alias_value),
                        )
                except Exception:
                    pass

            if diffuse_slot >= 0 and diffuse_slot in stage_indices:
                stage_indices = [diffuse_slot] + [slot for slot in stage_indices if int(slot) != int(diffuse_slot)]

            diffuse_texture_id = int(texture_ids[diffuse_slot]) if 0 <= diffuse_slot < len(texture_ids) else -1
            stage_ids = [int(texture_ids[index]) for index in stage_indices]
            # Do not treat the low-byte 0x71 state as Blender alpha
            # blending.  That bit pattern appears on ordinary PS2 character
            # hair/skin/detail runs as well as candidate translucent draws, and
            # forcing BLEND in Blender made opaque body materials render
            # incorrectly.  Keep the raw run state as metadata; let texture
            # alpha use the existing CLIP path until the real GS alpha state
            # table is mapped.
            alpha_blend = False
            # Build a material-binding key from the PS2 material state, not just
            # from the resolved diffuse texture.  Multiple draw runs can share a
            # true material and should merge, but runs with the same diffuse
            # texture and different secondary stage / render-state words must
            # stay separate.  Grouping only by texture made selected Blender
            # materials contain unrelated UV islands from different PS2 states.
            material_binding_key = (
                int(primary_index),
                int(secondary_index),
                int(run_flags),
                int(stage_control),
                int(stage_variant),
                tuple(int(value) for value in stage_ids),
                int(diffuse_texture_id),
            )
            material_index = -1
            material_group = -1

            runs.append({
                'material_binding_key': material_binding_key,
                'run_index': int(run_index),
                'start_local': int(start_local),
                'primary_index': int(primary_index),
                'secondary_index': int(secondary_index),
                'material_index': int(material_index),
                'material_group': int(material_group),
                'run_flags': int(run_flags),
                'stage_control': int(stage_control),
                'stage_variant': int(stage_variant),
                'ps2_alpha_blend': bool(alpha_blend),
                'ps2_stage_blend_mode': str(stage_blend_mode),
                'diffuse_texture_id': int(diffuse_texture_id),
                'stage_ids': stage_ids,
            })

        runs.sort(key=lambda item: int(item.get('start_local', 0)))

        if chunk_starts:
            active_runs: list[dict] = []
            for index, run in enumerate(runs):
                start_local = int(run.get('start_local', 0) or 0)
                next_start = int(runs[index + 1].get('start_local', 0) or 0) if index + 1 < len(runs) else 0x7FFFFFFF
                owns_packet = any(start_local <= int(chunk_start) < next_start for chunk_start in chunk_starts)
                if owns_packet:
                    active_runs.append(run)
                else:
                    logger.debug(
                        'TR8 PS2 %s skipped material-run #%d because it owns no VIF packets: start=0x%X next=0x%X',
                        context.file_name,
                        int(run.get('run_index', -1)),
                        int(start_local),
                        int(next_start),
                    )
            runs = active_runs

        group_by_key: dict[tuple, int] = {}
        for run in runs:
            key = run.get('material_binding_key')
            if not isinstance(key, tuple):
                key = (
                    int(run.get('primary_index', -1)),
                    int(run.get('secondary_index', -1)),
                    int(run.get('run_flags', 0)),
                    int(run.get('stage_control', 0)),
                    int(run.get('stage_variant', 0)),
                    tuple(int(value) for value in (run.get('stage_ids') or [])),
                    int(run.get('diffuse_texture_id', -1)),
                )
            if key not in group_by_key:
                group_by_key[key] = len(group_by_key)
            group_index = int(group_by_key[key])
            run['material_group'] = group_index
            run['material_index'] = group_index

        return runs

    @staticmethod
    def _tr8ps2_material_run_for_strip(runs: list[dict], strip_start_local: int) -> dict | None:
        selected: dict | None = None
        strip_start_local = int(strip_start_local)
        for run in runs or []:
            if int(run.get('start_local', 0)) <= strip_start_local:
                selected = run
            else:
                break
        return selected

    def _attach_tr8ps2_texture_stages(self, context: SectionContext, header_local: int, strips: list[TextureStrip]) -> None:
        texture_ids = self._parse_tr8ps2_texture_ids(context)
        if not texture_ids:
            return

        material_runs = self._parse_tr8ps2_material_runs(context, header_local, texture_ids)
        has_material_runs = bool(material_runs)
        fallback_texture_id = int(texture_ids[0])

        for strip_index, strip in enumerate(strips):
            run = self._tr8ps2_material_run_for_strip(material_runs, int(strip.offset))
            if run is not None:
                diffuse_texture_id = int(run.get('diffuse_texture_id', fallback_texture_id))
                stage_ids = [int(texture_id) for texture_id in (run.get('stage_ids') or [])]
                # Group strips by the resolved material key, not by render-run
                # index.  Multiple PS2 render runs can share one material/texture.
                material_group = int(run.get('material_group', diffuse_texture_id if diffuse_texture_id >= 0 else strip_index))
                strip.tr8_ps2_material_index = int(run.get('material_index', material_group))
                strip.tr8_ps2_run_flags = int(run.get('run_flags', 0) or 0)
                strip.tr8_ps2_alpha_blend = bool(run.get('ps2_alpha_blend', False))
                strip.tr8_ps2_stage_blend_mode = str(run.get('ps2_stage_blend_mode', '') or '')
            else:
                diffuse_texture_id = fallback_texture_id
                stage_ids = [fallback_texture_id]
                material_group = int(fallback_texture_id) if not has_material_runs else int(strip_index)
                strip.tr8_ps2_material_index = int(material_group)
                strip.tr8_ps2_stage_blend_mode = ''

            if not stage_ids:
                stage_ids = [diffuse_texture_id]

            strip.material_group = int(material_group)
            strip.tpageid = max(0, diffuse_texture_id) & 0x1FFF
            strip.tr8_material_resource_id = -1
            strip.tr8_material_file = ''
            strip.tr8_texture_stage_ids = list(stage_ids)
            strip.tr8_texture_stage_slots = list(range(len(stage_ids)))
            strip.tr8_texture_stage_types = [1] + ([0] * max(0, len(stage_ids) - 1))
            strip.tr8_diffuse_texture_id = int(diffuse_texture_id)
            strip.tr8_normal_texture_id = -1

        if material_runs:
            logger.info(
                'TR8 PS2 %s material runs: %s',
                context.file_name,
                ', '.join(
                    f"#{int(run.get('run_index', -1))}@0x{int(run.get('start_local', 0)):X}->0x{int(run.get('diffuse_texture_id', -1)):X}"
                    f"{'/ABE' if bool(run.get('ps2_alpha_blend', False)) else ''}"
                    for run in material_runs
                ),
            )
        else:
            logger.info(
                'TR8 PS2 %s texture stages: %s',
                context.file_name,
                ', '.join(f'0x{texture_id:X}' for texture_id in texture_ids),
            )

    def _parse_tr8ps2_geometry(self, context: SectionContext, header_local: int, segments: list[Segment] | None = None) -> tuple[list[MVertex], list[TextureStrip], list[tuple[int, int, int, int]], int]:
        br = context.reader
        previous = br.tell()
        starts: list[int] = []
        try:
            data = br.peek(context.data_size, offset=context.data_start)
        finally:
            br.seek(previous)

        # Find PS2 VIF chunks.  The observed Underworld packets start with a
        # V2_32 sort-key unpack (0x64) and the next unpack is V4_16 positions.
        for local in range(0, max(0, len(data) - 16), 4):
            word = struct.unpack_from('<I', data, local)[0]
            if word != 0x64018001:
                continue
            next_word = struct.unpack_from('<I', data, local + 12)[0]
            if ((next_word >> 24) & 0xFF) != 0x6D:
                continue
            starts.append(local)

        chunk_entries: list[tuple[int, dict, list[int], list[int]]] = []
        blend_records = self._parse_tr8ps2_blend_records(context, header_local)
        tr8ps2_blend_count = len(blend_records)
        tr8ps2_bone_count = int(self._read_u32_at(context, int(header_local) + 0x4C, 0))

        # First pass: decode packet streams and attach the chunk-local palette.
        # The palette uses a compact split namespace:
        #   0 .. bone_count - 1                 => direct rigid skeleton bone id
        #   bone_count .. bone_count+blend_count => index into the 8-byte blend table
        # Previous preview builds treated low values as blend-record ids and high
        # values as bone_count/blend_count aliases; that produced the random
        # weighted islands seen in Blender.
        for start_local in starts:
            streams = self._parse_tr8ps2_vif_chunk(context, start_local)
            if streams is None:
                continue
            positions = streams.get('positions') or []
            if not positions:
                continue
            blend_palette, source_ids = self._tr8ps2_find_vif_preamble(data, start_local, len(positions))
            streams['blend_palette'] = blend_palette
            streams['source_vertex_ids'] = source_ids
            chunk_entries.append((int(start_local), streams, blend_palette, source_ids))

        vertices: list[MVertex] = []
        vertex_colors: list[tuple[int, int, int, int]] = []
        strips: list[TextureStrip] = []
        vertex_map: dict[tuple, int] = {}
        max_segment = -1

        def resolve_skin_weights(streams: dict, matrix_slot: int) -> list[tuple[int, float]]:
            palette = streams.get('blend_palette') or []
            if 0 <= int(matrix_slot) < len(palette):
                palette_ref = int(palette[int(matrix_slot)])

                # Direct rigid bone reference.  These low palette values are
                # already in the same order as the recovered PS2 segment table.
                if 0 <= palette_ref < tr8ps2_bone_count:
                    return [(int(palette_ref), 1.0)]

                # Blended reference.  The blend-table namespace starts exactly
                # after the direct bone namespace, not at zero and not after the
                # blend table itself.  Example from Lara main: bone_count=121,
                # palette value 121 => blend record 0, 122 => blend record 1,
                # etc.
                blend_index = int(palette_ref) - int(tr8ps2_bone_count)
                if 0 <= blend_index < tr8ps2_blend_count:
                    normalized = self._tr8ps2_normalize_weights(blend_records[blend_index])
                    if normalized:
                        return normalized

            # Last-resort fallback for malformed chunks: this keeps geometry
            # visible without manufacturing unrelated bone assignments.
            if 0 <= int(matrix_slot) < tr8ps2_bone_count:
                return [(int(matrix_slot), 1.0)]
            return []

        def primary_segment_from_weights(weights: list[tuple[int, float]], fallback: int) -> int:
            if weights:
                return int(max(weights, key=lambda item: float(item[1]))[0])
            return int(fallback)

        def materialize_vertex(streams: dict, stream_index: int) -> int | None:
            nonlocal max_segment
            positions = streams.get('positions') or []
            if not (0 <= stream_index < len(positions)):
                return None
            x, y, z, flag_signed = positions[stream_index]
            flag = int(flag_signed) & 0xFFFF
            matrix_slot = self._tr8ps2_matrix_slot_from_flag(flag)
            skin_weights = resolve_skin_weights(streams, matrix_slot)
            segment = primary_segment_from_weights(skin_weights, matrix_slot)
            for bone_index, _weight in skin_weights:
                max_segment = max(max_segment, int(bone_index))
            max_segment = max(max_segment, int(segment))
            normals = streams.get('normals') or []
            normal_raw = self._tr8ps2_normal_s12_to_byte(normals[stream_index]) if stream_index < len(normals) else (0, 0, 127)
            uvs = streams.get('uvs') or []
            uv_raw = tuple(int(v) for v in uvs[stream_index]) if stream_index < len(uvs) else (0, 0)
            colors = streams.get('colors') or []
            color = tuple(int(v) for v in colors[stream_index]) if stream_index < len(colors) else (255, 255, 255, 255)
            weights_key = tuple((int(bone), round(float(weight), 6)) for bone, weight in skin_weights)
            # V4_8 colors have been observed as ABGR/RGBA depending on asset.
            # Keep source order stable for now; it is primarily useful as a
            # color attribute and should not block geometry import.
            key = (
                int(x), int(y), int(z),
                int(normal_raw[0]), int(normal_raw[1]), int(normal_raw[2]),
                int(uv_raw[0]), int(uv_raw[1]),
                weights_key,
                int(color[0]), int(color[1]), int(color[2]), int(color[3]),
            )
            existing = vertex_map.get(key)
            if existing is not None:
                return existing
            vertex_index = len(vertices)
            u = float(uv_raw[0]) / 4096.0
            v = 1.0 - (float(uv_raw[1]) / 4096.0)
            vertices.append(MVertex(
                index=vertex_index,
                position_raw=(int(x), int(y), int(z)),
                normal_raw=normal_raw,
                segment=int(segment),
                uv_raw=(int(uv_raw[0]), int(uv_raw[1])),
                uv_decoded=(u, v),
                gc_transform_id=int(matrix_slot),
                gc_bind_segment=int(segment),
                gc_primary_segment=int(segment),
                gc_secondary_segment=-1,
                gc_secondary_weight=0.0,
                skin_weights=skin_weights if skin_weights else None,
            ))
            vertex_colors.append((int(color[0]), int(color[1]), int(color[2]), int(color[3])))
            vertex_map[key] = vertex_index
            return vertex_index

        for chunk_index, (start_local, streams, _blend_palette, _source_ids) in enumerate(chunk_entries):
            positions = streams.get('positions') or []
            if not positions:
                continue
            strip_indices: list[int] = []
            window: list[int] = []
            for stream_index, position in enumerate(positions):
                flag = int(position[3]) & 0xFFFF
                window.append(stream_index)
                if len(window) > 3:
                    window = window[-3:]
                if len(window) < 3 or self._tr8ps2_strip_flag_adc(flag):
                    continue
                a_i, b_i, c_i = window
                a = materialize_vertex(streams, a_i)
                b = materialize_vertex(streams, b_i)
                c = materialize_vertex(streams, c_i)
                if a is None or b is None or c is None:
                    continue
                if len({a, b, c}) != 3:
                    continue
                if self._tr8ps2_strip_flag_flip(flag):
                    tri = [b, a, c]
                else:
                    tri = [a, b, c]
                strip_indices.extend(tri)
            if not strip_indices:
                continue
            strips.append(TextureStrip(
                offset=int(start_local),
                vertex_count=len(strip_indices),
                draw_group=0,
                tpageid=0,
                sort_push=float(streams.get('sort_key', 0.0) or 0.0),
                scroll_offset=0.0,
                env_mapping=0,
                next_texture=-1,
                indices=strip_indices,
                bone_ids=[],
                material_group=0,
                source_file=context.file_name,
                tr8_batch_index=0,
                tr8_vertex_format_offset=0,
            ))

        return vertices, strips, vertex_colors, max_segment

    def _tr8ps2_derive_position_scale(self, context: SectionContext, header_local: int, vertices: list[MVertex]) -> float:
        if not vertices:
            return 1.0 / 60.0

        try:
            raw_min = [min(int(vertex.position_raw[axis]) for vertex in vertices) for axis in range(3)]
            raw_max = [max(int(vertex.position_raw[axis]) for vertex in vertices) for axis in range(3)]
        except Exception:
            return 1.0 / 60.0

        header_min = [self._read_f32_at(context, int(header_local) + 0x20 + (axis * 4), 0.0) for axis in range(3)]
        header_max = [self._read_f32_at(context, int(header_local) + 0x30 + (axis * 4), 0.0) for axis in range(3)]

        ratios: list[float] = []
        for axis in range(3):
            raw_extent = float(raw_max[axis] - raw_min[axis])
            header_extent = float(header_max[axis] - header_min[axis])
            if raw_extent <= 1.0 or header_extent <= 1.0:
                continue
            ratio = header_extent / raw_extent
            if math.isfinite(ratio) and 0.0001 <= ratio <= 10.0:
                ratios.append(float(ratio))

        if not ratios:
            return 1.0 / 60.0

        ratios.sort()
        median = ratios[len(ratios) // 2]
        if len(ratios) % 2 == 0:
            median = (ratios[(len(ratios) // 2) - 1] + ratios[len(ratios) // 2]) * 0.5

        # Reject clearly non-uniform header/raw comparisons.  This keeps the
        # importer from inventing an arbitrary scale if a future packet parser
        # only recovers a partial mesh.
        max_deviation = max(abs((ratio / median) - 1.0) for ratio in ratios if median > 0.0)
        if max_deviation > 0.08:
            logger.warning(
                'TR8 PS2 %s header/raw bounds produced non-uniform position scales %s; using median %.8f',
                context.file_name,
                ', '.join(f'{ratio:.8f}' for ratio in ratios),
                median,
            )

        return float(median) if math.isfinite(float(median)) and median > 0.0 else (1.0 / 60.0)


    @staticmethod
    def _make_tr8ps2_placeholder_segments(count: int) -> list[Segment]:
        segments: list[Segment] = []
        for index in range(max(0, int(count))):
            segments.append(Segment(
                index=index,
                min_v=(0.0, 0.0, 0.0, 0.0),
                max_v=(0.0, 0.0, 0.0, 0.0),
                pivot=(0.0, 0.0, 0.0, 0.0),
                flags=0,
                first_vertex=-1,
                last_vertex=-1,
                parent=-1 if index == 0 else 0,
                hinfo=-1,
            ))
        return segments

    def _parse_tr8ps2_context(self, context: SectionContext, header_local: int) -> ModelData:
        segments = [] if self.parse_segments_only else self._parse_tr8ps2_skeleton(context, header_local)
        vertices, strips, vertex_colors, max_segment = self._parse_tr8ps2_geometry(context, header_local, segments)
        self._attach_tr8ps2_texture_stages(context, header_local, strips)
        required_segments = max(max_segment + 1, 1 if vertices else 0)
        if len(segments) < required_segments:
            if segments:
                # Keep recovered bones and append simple placeholders for any
                # transform slots used by geometry but absent from the table.
                for index in range(len(segments), required_segments):
                    segments.append(Segment(
                        index=index,
                        min_v=(0.0, 0.0, 0.0, 0.0),
                        max_v=(0.0, 0.0, 0.0, 0.0),
                        pivot=(0.0, 0.0, 0.0, 0.0),
                        flags=0,
                        first_vertex=-1,
                        last_vertex=-1,
                        parent=0,
                        hinfo=-1,
                    ))
            else:
                segments = self._make_tr8ps2_placeholder_segments(required_segments)

        for segment in segments:
            assigned = [vertex.index for vertex in vertices if int(vertex.segment) == int(segment.index)]
            segment.first_vertex = min(assigned) if assigned else -1
            segment.last_vertex = max(assigned) if assigned else -1

        radius = self._read_f32_at(context, int(header_local) + 0x40, 0.0)
        logger.info(
            'Parsed TR8 PS2 mesh %s: vifChunks=%d vertices=%d strips=%d triangles=%d bones=%d',
            context.file_name,
            len(strips),
            len(vertices),
            len(strips),
            sum(len(strip.indices) // 3 for strip in strips),
            len(segments),
        )

        ps2_position_scale = self._tr8ps2_derive_position_scale(context, header_local, vertices)
        logger.debug('TR8 PS2 %s position scale %.8f', context.file_name, ps2_position_scale)

        return ModelData(
            version=19,
            model_scale=(ps2_position_scale, ps2_position_scale, ps2_position_scale, 1.0),
            segments=segments,
            virt_segments=[],
            vertices=vertices,
            faces=[],
            strips=strips,
            vertex_colors=vertex_colors if len(vertex_colors) == len(vertices) and vertices else None,
            hmarkers=[],
            hspheres=[],
            hboxes=[],
            hcapsules=[],
            targets=[],
            markups=[],
            max_rad=float(radius or 0.0),
            max_rad_sq=float(radius or 0.0) ** 2,
            cdc_render_data_id=int(context.section_info.section_id),
            uv_format='underworld',
        )

    def _parse_context(self, context: SectionContext) -> ModelData:
        ps2_header_local = self._find_ps2_mesh_header_local(context)
        if ps2_header_local >= 0:
            return self._parse_tr8ps2_context(context, ps2_header_local)

        header_local = self._find_mesh_header_local(context)
        if header_local < 0:
            raise ValueError(f'{context.file_name} does not contain a TR8 Mesh header')
        header = self._read_header(context, header_local)
        mesh_records = self._parse_mesh_infos(context, int(header['offset_mesh_info']), int(header['num_meshes']))
        groups = self._parse_mesh_groups(context, int(header['offset_mesh_group_info']), int(header['num_mesh_groups']))
        global_bone_map = self._parse_bone_map(context, int(header['offset_bone_map']), int(header['num_bones']))
        meshes = self._build_render_batches(context, mesh_records, groups, int(header['offset_face_data'])) or mesh_records

        vertices: List[MVertex] = []
        vertex_colors: List[Tuple[int, int, int, int]] = []
        global_start = 0
        for mesh in meshes:
            mesh.global_vertex_start = global_start
            if mesh.num_vertices <= 0:
                continue
            skin_map = self._parse_skin_map(context, mesh, global_bone_map)
            mesh_vertices, mesh_colors = self._parse_vertices_for_mesh(context, mesh, skin_map)
            if not mesh_vertices:
                continue
            vertices.extend(mesh_vertices)
            vertex_colors.extend(mesh_colors)
            global_start += len(mesh_vertices)

        strips = self._parse_triangle_strips(context, meshes, groups, int(header['offset_face_data']), int(header['num_indices']))
        material_ids = self._parse_tr8_material_ids(context, int(header.get('offset_mat_info', 0) or 0))
        self._attach_underworld_materials(context, strips, material_ids)
        segments = self._parse_skeleton(context, int(header['num_bones']))

        logger.info(
            'Parsed TR8 mesh %s: batches=%d meshGroups=%d vertices=%d strips=%d triangles=%d bones=%d',
            context.file_name,
            len([mesh for mesh in meshes if mesh.num_vertices > 0]),
            len(groups),
            len(vertices),
            len(strips),
            sum(len(strip.indices) // 3 for strip in strips),
            len(segments),
        )

        return ModelData(
            version=19,
            model_scale=(1.0, 1.0, 1.0, 1.0),
            segments=segments,
            virt_segments=[],
            vertices=vertices,
            faces=[],
            strips=strips,
            vertex_colors=vertex_colors if len(vertex_colors) == len(vertices) and vertices else None,
            hmarkers=[],
            hspheres=[],
            hboxes=[],
            hcapsules=[],
            targets=[],
            markups=[],
            max_rad=float(header.get('bounding_sphere_radius', 0.0) or 0.0),
            max_rad_sq=float(header.get('bounding_sphere_radius', 0.0) or 0.0) ** 2,
            cdc_render_data_id=int(context.section_info.section_id),
            uv_format='underworld',
        )
