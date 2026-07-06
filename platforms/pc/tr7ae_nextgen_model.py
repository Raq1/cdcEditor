from __future__ import annotations

import math
import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ...core.log import logger
from ...core.model_types import MVertex, ModelData, Segment, TextureStrip
from ..common.section import SectionContextCache


D3DDECLUSAGE_POSITION = 0
D3DDECLUSAGE_BLENDWEIGHT = 1
D3DDECLUSAGE_BLENDINDICES = 2
D3DDECLUSAGE_NORMAL = 3
D3DDECLUSAGE_TEXCOORD = 5
D3DDECLUSAGE_COLOR = 10

D3DDECLTYPE_FLOAT1 = 0
D3DDECLTYPE_FLOAT2 = 1
D3DDECLTYPE_FLOAT3 = 2
D3DDECLTYPE_FLOAT4 = 3
D3DDECLTYPE_D3DCOLOR = 4
D3DDECLTYPE_UBYTE4 = 5
D3DDECLTYPE_SHORT2 = 6
D3DDECLTYPE_SHORT4 = 7
D3DDECLTYPE_UNUSED = 17


@dataclass(slots=True)
class _Header:
    base: int
    flags: int
    total_data_size: int
    num_indices: int
    box_min: Tuple[float, float, float, float]
    box_max: Tuple[float, float, float, float]
    prim_group_offset: int
    model_batch_offset: int
    bone_offset: int
    material_offset: int
    index_offset: int
    num_prim_groups: int
    num_batches: int
    num_bones: int
    num_materials: int
    num_pixmaps: int


@dataclass(slots=True)
class _PrimGroup:
    base_index: int
    num_primitives: int
    num_vertices: int
    vertex_shader_flags: int
    material_index: int


@dataclass(slots=True)
class _Batch:
    index: int
    flags: int
    num_prim_groups: int
    skin_map_size: int
    skin_map_offset: int
    vertex_data_offset: int
    vertex_elements: List[dict]
    vertex_format: int
    vertex_stride: int
    num_vertices: int
    base_index: int
    num_primitives: int
    vertex_start: int = 0
    skin_map: List[int] | None = None


@dataclass(slots=True)
class _MaterialLayer:
    index: int
    color: tuple[float, float, float, float]
    texcoord_source: int
    modifier: int
    param_id: int
    constant: tuple[float, float, float, float]
    texture_index: int
    texture_id: int
    num_textures: int
    enabled: int


@dataclass(slots=True)
class _MaterialData:
    index: int
    record_offset: int = 0
    material_id: int = -1
    asset_id_hi: int = 0
    asset_id_lo: int = 0
    asset_id_padding_hex: str = ''
    material_record_hex: str = ''
    blend_mode: int = 0
    combiner_type: int = 0
    flags: int = 0
    opacity: float = 1.0
    poly_flags: int = 0
    uv_auto_scroll_speed: int = 0
    sort_bias: float = 0.0
    detail_range_mul: float = 0.0
    detail_scale: float = 0.0
    parallax_scale: float = 0.0
    parallax_offset: float = 0.0
    specular_power: float = 0.0
    specular_shift0: float = 0.0
    specular_shift1: float = 0.0
    rim_light_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    rim_light_intensity: float = 0.0
    water_blend_bias: float = 0.0
    water_blend_exponent: float = 0.0
    water_deep_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    local_num_pixmaps: int = 0
    layers: list[_MaterialLayer] = None
    shader_indices: list[int] = None
    fx_material_data_offset: int = 0
    special_material_flag: bool = False
    diffuse_id: int = -1
    normal_id: int = -1
    specular_id: int = -1

    def __post_init__(self):
        if self.layers is None:
            self.layers = []
        if self.shader_indices is None:
            self.shader_indices = []


class TRNextGenModelParser:
    """Parser for TR7 PC next-generation render data sections.

    These sections are referenced from the ordinary PC model by cdcRenderDataID
    and have section type 7 in a DRM.  The layout is the PCD9 render-data block
    described by the 010 template shipped with this add-on patch.
    """

    def __init__(self, filepath: str, *, source_model: ModelData | None = None, cdc_render_data_id: int = 0):
        self.filepath = str(filepath)
        self.source_model = source_model
        self.cdc_render_data_id = int(cdc_render_data_id or 0)
        self._blob = b''
        self._data_start = 0
        self._data_end = 0
        self._file_name = Path(filepath).name

    def parse(self) -> ModelData:
        cache = SectionContextCache(self.filepath, endian='<')
        try:
            context = cache.get_root_context()
            self._blob = Path(context.filepath).read_bytes()
            self._data_start = int(context.data_start)
            self._data_end = int(context.data_end)
            self._file_name = context.file_name
            header_offsets = self._read_nextgen_header_offsets()
            pc_base = self._find_pcmodeldata_base(header_offsets[0])
            header = self._parse_header(pc_base)
            pixmaps = self._parse_pixmaps(header_offsets[1])
            shader_table_ids = self._parse_int_table(header_offsets[2])
            special_material_flags = self._parse_int_table(header_offsets[3])
            materials = self._parse_materials(header, pixmaps, special_material_flags)
            prim_groups = self._parse_prim_groups(header)
            batches = self._parse_batches(header)
            vertices, vertex_colors = self._parse_vertices(header, batches)
            strips = self._parse_strips(header, batches, prim_groups, materials, shader_table_ids)
            segments = self._make_segments(header, len(vertices))

            logger.info(
                'Parsed next-gen render model %s cdcRenderDataID=0x%X vertices=%d strips=%d primGroups=%d batches=%d materials=%d',
                self._file_name,
                self.cdc_render_data_id,
                len(vertices),
                len(strips),
                len(prim_groups),
                len(batches),
                len(materials),
            )
            return ModelData(
                version=0x39444350,
                model_scale=(1.0, 1.0, 1.0, 1.0),
                segments=segments,
                virt_segments=[],
                vertices=vertices,
                faces=[],
                strips=strips,
                vertex_colors=vertex_colors or None,
                cdc_render_data_id=self.cdc_render_data_id,
                uv_format='pc_nextgen',
            )
        finally:
            cache.close()

    def _valid_abs(self, offset: int, size: int = 1) -> bool:
        return 0 <= int(offset) <= len(self._blob) and int(offset) + int(size) <= len(self._blob)

    def _u8(self, offset: int) -> int:
        return self._blob[int(offset)]

    def _u16(self, offset: int) -> int:
        return struct.unpack_from('<H', self._blob, int(offset))[0]

    def _i16(self, offset: int) -> int:
        return struct.unpack_from('<h', self._blob, int(offset))[0]

    def _u32(self, offset: int) -> int:
        return struct.unpack_from('<I', self._blob, int(offset))[0]

    def _i32(self, offset: int) -> int:
        return struct.unpack_from('<i', self._blob, int(offset))[0]

    def _f32(self, offset: int) -> float:
        return struct.unpack_from('<f', self._blob, int(offset))[0]

    def _f32_tuple(self, offset: int, count: int) -> tuple[float, ...]:
        return struct.unpack_from('<' + ('f' * int(count)), self._blob, int(offset))

    def _read_nextgen_header_offsets(self) -> tuple[int, int, int, int]:
        if not self._valid_abs(self._data_start, 16):
            raise ValueError(f'Next-gen section {self._file_name} is too small for its header')
        return struct.unpack_from('<IIII', self._blob, self._data_start)

    def _find_pcmodeldata_base(self, offset_pc_model_data: int) -> int:
        candidates = [self._data_start + 0x10]
        if int(offset_pc_model_data) >= 0:
            candidates.extend([
                self._data_start + int(offset_pc_model_data),
                self._data_start + int(offset_pc_model_data) + 4,
            ])
        for candidate in candidates:
            if self._valid_abs(candidate, 4) and self._blob[candidate:candidate + 4] == b'PCD9':
                return int(candidate)
        search_end = min(self._data_start + 0x80, self._data_end)
        found = self._blob.find(b'PCD9', self._data_start, search_end)
        if found >= 0:
            return int(found)
        raise ValueError(f'Could not locate PCD9 PCModelData block in {self._file_name}')

    def _parse_header(self, base: int) -> _Header:
        if not self._valid_abs(base, 0x70):
            raise ValueError(f'PCModelData header in {self._file_name} is outside the section bounds')
        magic = self._blob[base:base + 4]
        if magic != b'PCD9':
            raise ValueError(f'Unsupported next-gen model magic {magic!r} in {self._file_name}')
        flags = self._u32(base + 0x04)
        total_data_size = self._u32(base + 0x08)
        num_indices = self._u32(base + 0x0C)
        box_min = self._f32_tuple(base + 0x20, 4)
        box_max = self._f32_tuple(base + 0x30, 4)
        prim_group_offset = self._u32(base + 0x4C)
        model_batch_offset = self._u32(base + 0x50)
        bone_offset = self._u32(base + 0x54)
        material_offset = self._u32(base + 0x58)
        index_offset = self._u32(base + 0x5C)
        num_prim_groups = self._u16(base + 0x64)
        num_batches = self._u16(base + 0x66)
        num_bones = self._u16(base + 0x68)
        num_materials = self._u16(base + 0x6A)
        num_pixmaps = self._u16(base + 0x6C)
        return _Header(
            base=int(base),
            flags=int(flags),
            total_data_size=int(total_data_size),
            num_indices=int(num_indices),
            box_min=tuple(float(v) for v in box_min),
            box_max=tuple(float(v) for v in box_max),
            prim_group_offset=int(prim_group_offset),
            model_batch_offset=int(model_batch_offset),
            bone_offset=int(bone_offset),
            material_offset=int(material_offset),
            index_offset=int(index_offset),
            num_prim_groups=int(num_prim_groups),
            num_batches=int(num_batches),
            num_bones=int(num_bones),
            num_materials=int(num_materials),
            num_pixmaps=int(num_pixmaps),
        )

    def _parse_pixmaps(self, offset_pixmaps: int) -> list[int]:
        candidates = [self._data_start + int(offset_pixmaps), int(offset_pixmaps)]
        for base in candidates:
            if not self._valid_abs(base, 4):
                continue
            count = self._i32(base)
            if count < 0 or count > 4096:
                continue
            end = base + 4 + count * 4
            if not self._valid_abs(base, 4 + count * 4):
                continue
            try:
                return [self._i32(base + 4 + index * 4) for index in range(count)]
            except Exception:
                continue
        return []

    def _parse_int_table(self, relative_offset: int) -> list[int]:
        candidates = [self._data_start + int(relative_offset), int(relative_offset)]
        for base in candidates:
            if not self._valid_abs(base, 4):
                continue
            try:
                count = self._i32(base)
            except Exception:
                continue
            if count < 0 or count > 4096:
                continue
            if not self._valid_abs(base, 4 + count * 4):
                continue
            try:
                return [self._i32(base + 4 + index * 4) for index in range(count)]
            except Exception:
                continue
        return []

    @staticmethod
    def _color_byte_rgba(raw_bgra: bytes) -> tuple[float, float, float, float]:
        if len(raw_bgra) < 4:
            return (1.0, 1.0, 1.0, 1.0)
        b, g, r, a = raw_bgra[:4]
        return (int(r) / 255.0, int(g) / 255.0, int(b) / 255.0, int(a) / 255.0)

    def _parse_material_layer(self, offset: int, layer_index: int, pixmaps: list[int]) -> _MaterialLayer:
        texture_index = 0xFFFF
        texture_id = -1
        enabled = 0
        num_textures = 0
        color = (1.0, 1.0, 1.0, 1.0)
        texcoord_source = 0
        modifier = 0
        param_id = 0
        constant = (0.0, 0.0, 0.0, 0.0)
        if self._valid_abs(offset, 36):
            color = self._color_byte_rgba(self._blob[offset:offset + 4])
            texcoord_source = self._i32(offset + 0x04)
            modifier = self._i32(offset + 0x08)
            param_id = self._i32(offset + 0x0C)
            constant = tuple(float(value) for value in self._f32_tuple(offset + 0x10, 4))
            texture_index = self._u16(offset + 0x20)
            num_textures = self._u8(offset + 0x22)
            enabled = struct.unpack_from('<b', self._blob, offset + 0x23)[0]
            if 0 <= int(texture_index) < len(pixmaps):
                resolved = int(pixmaps[int(texture_index)])
                if resolved >= 0:
                    texture_id = resolved
        return _MaterialLayer(
            index=int(layer_index),
            color=color,
            texcoord_source=int(texcoord_source),
            modifier=int(modifier),
            param_id=int(param_id),
            constant=constant,
            texture_index=int(texture_index),
            texture_id=int(texture_id),
            num_textures=int(num_textures),
            enabled=int(enabled),
        )

    def _parse_materials(self, header: _Header, pixmaps: list[int], special_material_flags: list[int]) -> list[_MaterialData]:
        result: list[_MaterialData] = []
        material_base = header.base + header.material_offset
        if header.num_materials <= 0 or not self._valid_abs(material_base, 1):
            return result

        # The documented PCMaterialData record is fixed-size: 0x1C8 / 456 bytes.
        # Some files have padding between the material table and the next block, so
        # deriving stride from the table span can misalign later materials.
        material_stride = 456

        special_set = {int(value) for value in special_material_flags}
        for material_index in range(header.num_materials):
            offset = material_base + material_index * material_stride
            material = _MaterialData(index=int(material_index), record_offset=int(offset - header.base), special_material_flag=material_index in special_set)
            if not self._valid_abs(offset, min(material_stride, 456)):
                result.append(material)
                continue

            material.material_id = self._i32(offset + 0x00)
            material.material_record_hex = self._blob[offset:offset + material_stride].hex()
            material.asset_id_hi = struct.unpack_from('<Q', self._blob, offset + 0x08)[0]
            material.asset_id_lo = self._u16(offset + 0x10)
            material.asset_id_padding_hex = self._blob[offset + 0x12:offset + 0x18].hex()
            material.blend_mode = self._i32(offset + 0x18)
            material.combiner_type = self._i32(offset + 0x1C)
            material.flags = self._u32(offset + 0x20)
            material.opacity = float(self._f32(offset + 0x24))
            material.poly_flags = self._u32(offset + 0x28)
            material.uv_auto_scroll_speed = self._u16(offset + 0x2C)
            material.sort_bias = float(self._f32(offset + 0x30))
            material.detail_range_mul = float(self._f32(offset + 0x34))
            material.detail_scale = float(self._f32(offset + 0x38))
            material.parallax_scale = float(self._f32(offset + 0x3C))
            material.parallax_offset = float(self._f32(offset + 0x40))
            material.specular_power = float(self._f32(offset + 0x44))
            material.specular_shift0 = float(self._f32(offset + 0x48))
            material.specular_shift1 = float(self._f32(offset + 0x4C))
            material.rim_light_color = tuple(float(value) for value in self._f32_tuple(offset + 0x50, 4))
            material.rim_light_intensity = float(self._f32(offset + 0x60))
            material.water_blend_bias = float(self._f32(offset + 0x64))
            material.water_blend_exponent = float(self._f32(offset + 0x68))
            material.water_deep_color = tuple(float(value) for value in self._f32_tuple(offset + 0x6C, 4))
            material.local_num_pixmaps = int(self._u8(offset + 0x7C))
            material.layers = [self._parse_material_layer(offset + 0x80 + layer_index * 36, layer_index, pixmaps) for layer_index in range(8)]
            material.shader_indices = [self._u32(offset + 0x1A0 + shader_index * 4) for shader_index in range(8)]
            material.fx_material_data_offset = self._u32(offset + 0x1C0)

            enabled_texture_layers = [layer for layer in material.layers if int(layer.enabled) != 0 and int(layer.texture_id) >= 0]
            if enabled_texture_layers:
                material.diffuse_id = int(enabled_texture_layers[0].texture_id)
                if len(enabled_texture_layers) > 1:
                    material.normal_id = int(enabled_texture_layers[1].texture_id)
                if len(enabled_texture_layers) > 2:
                    material.specular_id = int(enabled_texture_layers[2].texture_id)
            elif 0 <= material_index < len(pixmaps):
                material.diffuse_id = int(pixmaps[material_index])
            result.append(material)
        return result

    def _parse_prim_groups(self, header: _Header) -> list[_PrimGroup]:
        groups: list[_PrimGroup] = []
        base = header.base + header.prim_group_offset
        for index in range(header.num_prim_groups):
            offset = base + index * 20
            if not self._valid_abs(offset, 20):
                break
            groups.append(_PrimGroup(
                base_index=self._u32(offset + 0x00),
                num_primitives=self._u32(offset + 0x04),
                num_vertices=self._u32(offset + 0x08),
                vertex_shader_flags=self._u16(offset + 0x0C),
                material_index=self._u32(offset + 0x10),
            ))
        return groups

    def _parse_vertex_elements(self, offset: int) -> list[dict]:
        elements: list[dict] = []
        for element_index in range(16):
            element_offset = offset + element_index * 8
            if not self._valid_abs(element_offset, 8):
                break
            stream, elem_offset = struct.unpack_from('<HH', self._blob, element_offset)
            elem_type = self._u8(element_offset + 4)
            method = self._u8(element_offset + 5)
            usage = self._u8(element_offset + 6)
            usage_index = self._u8(element_offset + 7)
            if stream == 0xFF or elem_type == D3DDECLTYPE_UNUSED:
                break
            elements.append({
                'stream': int(stream),
                'offset': int(elem_offset),
                'type': int(elem_type),
                'method': int(method),
                'usage': int(usage),
                'usage_index': int(usage_index),
            })
        return elements

    def _parse_batches(self, header: _Header) -> list[_Batch]:
        batches: list[_Batch] = []
        base = header.base + header.model_batch_offset
        vertex_start = 0
        for index in range(header.num_batches):
            offset = base + index * 0xAC
            if not self._valid_abs(offset, 0xAC):
                break
            batch = _Batch(
                index=index,
                flags=self._u32(offset + 0x00),
                num_prim_groups=self._u32(offset + 0x04),
                skin_map_size=self._u16(offset + 0x08),
                skin_map_offset=self._u32(offset + 0x0C),
                vertex_data_offset=self._u32(offset + 0x10),
                vertex_elements=self._parse_vertex_elements(offset + 0x18),
                vertex_format=self._u32(offset + 0x98),
                vertex_stride=self._u32(offset + 0x9C),
                num_vertices=self._u32(offset + 0xA0),
                base_index=self._u32(offset + 0xA4),
                num_primitives=self._u32(offset + 0xA8),
                vertex_start=vertex_start,
            )
            vertex_start += max(0, int(batch.num_vertices))
            batch.skin_map = self._parse_skin_map(header, batch)
            batches.append(batch)
        return batches

    def _parse_skin_map(self, header: _Header, batch: _Batch) -> list[int]:
        size = max(0, int(batch.skin_map_size))
        if size <= 0:
            return []
        offset = header.base + int(batch.skin_map_offset)
        if not self._valid_abs(offset, size * 4):
            return []
        return [self._u32(offset + index * 4) for index in range(size)]

    @staticmethod
    def _element_by_usage(elements: list[dict], usage: int, usage_index: int = 0) -> dict | None:
        for element in elements:
            if int(element.get('usage', -1)) == int(usage) and int(element.get('usage_index', 0)) == int(usage_index):
                return element
        for element in elements:
            if int(element.get('usage', -1)) == int(usage):
                return element
        return None

    @staticmethod
    def _normal_to_raw(normal: tuple[float, float, float]) -> tuple[int, int, int]:
        values = []
        for value in normal:
            if not math.isfinite(float(value)):
                value = 0.0
            value = max(-1.0, min(1.0, float(value)))
            values.append(int(round(value * 127.0)))
        return (values[0], values[1], values[2])

    @staticmethod
    def _bone_from_skin_map(batch: _Batch, bone_index: int) -> int:
        skin_map = batch.skin_map or []
        raw = int(bone_index)
        if 0 <= raw < len(skin_map):
            return int(skin_map[raw])
        return raw

    @staticmethod
    def _valid_weight(value: float) -> float:
        if not math.isfinite(float(value)):
            return 0.0
        return max(0.0, min(1.0, float(value)))

    def _parse_vertices(self, header: _Header, batches: list[_Batch]) -> tuple[list[MVertex], list[tuple[int, int, int, int]]]:
        vertices: list[MVertex] = []
        vertex_colors: list[tuple[int, int, int, int]] = []

        for batch in batches:
            position_elem = self._element_by_usage(batch.vertex_elements, D3DDECLUSAGE_POSITION)
            uv_elem = self._element_by_usage(batch.vertex_elements, D3DDECLUSAGE_TEXCOORD)
            normal_elem = self._element_by_usage(batch.vertex_elements, D3DDECLUSAGE_NORMAL)
            color_elem = self._element_by_usage(batch.vertex_elements, D3DDECLUSAGE_COLOR)
            weight_elem = self._element_by_usage(batch.vertex_elements, D3DDECLUSAGE_BLENDWEIGHT)
            indices_elem = self._element_by_usage(batch.vertex_elements, D3DDECLUSAGE_BLENDINDICES)

            if position_elem is None or batch.vertex_stride <= 0 or batch.num_vertices <= 0:
                logger.warning('Skipping next-gen batch %d in %s: missing position element or invalid stride', batch.index, self._file_name)
                continue
            vertex_base = header.base + int(batch.vertex_data_offset)
            if not self._valid_abs(vertex_base, batch.vertex_stride * max(0, int(batch.num_vertices))):
                logger.warning('Skipping next-gen batch %d in %s: vertex stream is outside file bounds', batch.index, self._file_name)
                continue

            for local_index in range(batch.num_vertices):
                offset = vertex_base + local_index * batch.vertex_stride
                vertex_index = len(vertices)
                pos_offset = offset + int(position_elem.get('offset', 0))
                x, y, z = (0.0, 0.0, 0.0)
                if self._valid_abs(pos_offset, 12):
                    x, y, z = self._f32_tuple(pos_offset, 3)

                normal_raw = (0, 0, 127)
                if normal_elem is not None:
                    normal_offset = offset + int(normal_elem.get('offset', 0))
                    if self._valid_abs(normal_offset, 12):
                        normal_raw = self._normal_to_raw(self._f32_tuple(normal_offset, 3))

                color = (255, 255, 255, 255)
                if color_elem is not None:
                    color_offset = offset + int(color_elem.get('offset', 0))
                    if self._valid_abs(color_offset, 4):
                        # D3DCOLOR is stored little-endian as BGRA.
                        b, g, r, a = self._blob[color_offset:color_offset + 4]
                        color = (int(r), int(g), int(b), int(a))

                uv_decoded = (0.0, 0.0)
                if uv_elem is not None:
                    uv_offset = offset + int(uv_elem.get('offset', 0))
                    if self._valid_abs(uv_offset, 8):
                        u, v = self._f32_tuple(uv_offset, 2)
                        uv_decoded = (float(u), 1.0 - float(v))

                skin_weights: list[tuple[int, float]] = []
                if indices_elem is not None:
                    indices_offset = offset + int(indices_elem.get('offset', 0))
                    if self._valid_abs(indices_offset, 4):
                        bone0 = self._bone_from_skin_map(batch, self._i16(indices_offset + 0))
                        bone1 = self._bone_from_skin_map(batch, self._i16(indices_offset + 2))
                        weight = 0.0
                        if weight_elem is not None:
                            weight_offset = offset + int(weight_elem.get('offset', 0))
                            if self._valid_abs(weight_offset, 4):
                                weight = self._valid_weight(self._f32(weight_offset))
                        if int(bone1) == int(bone0):
                            skin_weights = [(int(bone0), 1.0)]
                        else:
                            skin_weights = [(int(bone0), 1.0 - weight)]
                            if weight > 0.0:
                                skin_weights.append((int(bone1), weight))
                        skin_weights = [(bone, value) for bone, value in skin_weights if bone >= 0 and value > 0.0001]

                segment = skin_weights[0][0] if skin_weights else 0
                vertices.append(MVertex(
                    index=vertex_index,
                    position_raw=(float(x), float(y), float(z)),
                    normal_raw=normal_raw,
                    segment=int(segment),
                    uv_raw=(0, 0),
                    uv_decoded=uv_decoded,
                    ps3_color_rgba=color,
                    skin_weights=skin_weights or None,
                ))
                vertex_colors.append(color)
        return vertices, vertex_colors

    def _read_raw_indices_at(self, offset: int, count: int) -> list[int]:
        count = max(0, int(count))
        offset = int(offset)
        if count <= 0 or not self._valid_abs(offset, count * 2):
            return []
        # The 010 template declares PCModelData.Indices as signed short.  Keep
        # the signed spelling here so negative/reversed-winding markers are not
        # lost before triangle decode.
        return [self._i16(offset + index * 2) for index in range(count)]

    @staticmethod
    def _decode_nextgen_index(raw_value: int, batch_vertex_count: int) -> tuple[int, bool]:
        """Decode a PC next-gen face index.

        The documented index stream is signed 16-bit.  Ordinary indices are
        non-negative.  Reversed-winding faces may be encoded either in the
        signed-negative form used by several Crystal Dynamics face streams
        (-index-1), or by setting bit 15 on an otherwise unsigned local index.
        The order below matters: 0xFFFF must decode to index 0 for the signed
        form, not to 1.
        """
        vertex_count = max(0, int(batch_vertex_count))
        signed_value = int(raw_value)
        if signed_value > 0x7FFF:
            signed_value -= 0x10000
        raw = signed_value & 0xFFFF
        if 0 <= signed_value < vertex_count:
            return int(signed_value), False
        if signed_value < 0 or (raw & 0x8000):
            candidates = []
            if signed_value < 0:
                candidates.append((-signed_value) - 1)
            candidates.append(raw & 0x7FFF)
            if signed_value < 0:
                candidates.append(-signed_value)
            for candidate in candidates:
                if 0 <= int(candidate) < vertex_count:
                    return int(candidate), True
        return int(raw), False

    @classmethod
    def _decode_nextgen_triangle(cls, raw_values: list[int] | tuple[int, int, int], batch: _Batch) -> tuple[list[int], bool]:
        decoded: list[int] = []
        reverse = False
        for raw_value in raw_values[:3]:
            index, flagged = cls._decode_nextgen_index(int(raw_value), int(batch.num_vertices))
            decoded.append(index)
            reverse = reverse or bool(flagged)
        if reverse and len(decoded) == 3:
            decoded = [decoded[0], decoded[2], decoded[1]]
        return decoded, reverse

    @staticmethod
    def _triangle_winding_parity(triangle: list[int] | tuple[int, int, int]) -> int:
        """Return a stable parity bit for a triangle's vertex order.

        For a fixed set of three distinct vertex indices, opposite winding has
        the opposite permutation parity. This lets the importer detect PC
        next-gen double-wound face pairs without relying on geometry normals or
        adjacency.
        """
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
    def _count_double_wound_triangle_pairs(cls, triangles: list[list[int] | tuple[int, int, int]]) -> int:
        """Count paired faces with identical vertices and opposite winding.

        A material marked this way should be exported with both windings for
        every affected face. Import collapses those explicit face-data pairs to
        one editable Blender polygon and stores the material-level flag so the
        exporter can regenerate the second winding when Double Sided is enabled.
        """
        buckets: dict[tuple[int, int, int], list[int]] = {}
        for tri in triangles:
            if len(tri) != 3 or len(set(int(v) for v in tri)) != 3:
                continue
            key = tuple(sorted(int(v) for v in tri))
            parity = cls._triangle_winding_parity(tri)
            if key not in buckets:
                buckets[key] = [0, 0]
            buckets[key][parity] += 1
        return sum(min(counts[0], counts[1]) for counts in buckets.values())

    @classmethod
    def _collapse_double_wound_triangle_pairs(cls, triangles: list[list[int] | tuple[int, int, int]]) -> list[list[int]]:
        """Collapse explicit PC next-gen front/back face pairs for Blender import.

        TR7 PC next-gen stores double-sided geometry as two triangle records
        with the same three vertices and opposite winding.  Blender only needs
        one editable polygon; the imported material's Double Sided flag records
        the intent, and export expands it back to the paired index data.

        When a key has extra unpaired duplicates, only the matched opposite-
        winding pairs are collapsed.  The kept representative uses the first
        observed winding for that vertex set, preserving the source orientation
        as much as possible.
        """
        if not triangles:
            return []

        counts: dict[tuple[int, int, int], list[int]] = {}
        preferred_parity: dict[tuple[int, int, int], int] = {}
        for tri in triangles:
            if len(tri) != 3 or len(set(int(v) for v in tri)) != 3:
                continue
            key = tuple(sorted(int(v) for v in tri))
            parity = cls._triangle_winding_parity(tri)
            counts.setdefault(key, [0, 0])[parity] += 1
            preferred_parity.setdefault(key, parity)

        pair_quota: dict[tuple[int, int, int], int] = {
            key: min(value[0], value[1])
            for key, value in counts.items()
            if min(value[0], value[1]) > 0
        }
        if not pair_quota:
            return [[int(value) for value in tri[:3]] for tri in triangles]

        consumed: dict[tuple[int, int, int], list[int]] = {key: [0, 0] for key in pair_quota}
        emitted_preferred: dict[tuple[int, int, int], int] = {key: 0 for key in pair_quota}
        collapsed: list[list[int]] = []

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
                # Opposite winding half of a matched pair: skip it.
                continue

            # Additional duplicates beyond the matched pair quota are not part
            # of a front/back pair, so preserve them as authored.
            collapsed.append(tri3)

        return collapsed

    def _score_index_data_start(self, header: _Header, batches: list[_Batch], prim_groups: list[_PrimGroup], absolute_offset: int) -> tuple[int, int, int, int, int]:
        if not self._valid_abs(absolute_offset, max(0, int(header.num_indices)) * 2):
            # The selected groups may not consume every declared index in some
            # malformed files.  Still reject candidates that cannot cover the
            # highest referenced group range below.
            pass
        valid = 0
        degenerate = 0
        out_of_bounds = 0
        unreadable = 0
        reversed_faces = 0
        group_cursor = 0
        for batch in batches:
            for _local_group_index in range(int(batch.num_prim_groups)):
                if group_cursor >= len(prim_groups):
                    break
                group = prim_groups[group_cursor]
                group_cursor += 1
                index_start = int(group.base_index)
                face_count = int(group.num_primitives)
                if index_start < 0 or face_count <= 0:
                    continue
                group_abs = int(absolute_offset) + index_start * 2
                if not self._valid_abs(group_abs, face_count * 3 * 2):
                    unreadable += max(0, face_count)
                    continue
                for face_index in range(face_count):
                    tri_abs = group_abs + face_index * 6
                    raw = (self._u16(tri_abs), self._u16(tri_abs + 2), self._u16(tri_abs + 4))
                    tri, reversed_winding = self._decode_nextgen_triangle(raw, batch)
                    if reversed_winding:
                        reversed_faces += 1
                    if min(tri) < 0 or max(tri) >= int(batch.num_vertices):
                        out_of_bounds += 1
                    elif tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                        degenerate += 1
                    else:
                        valid += 1
        return (valid, -degenerate, -out_of_bounds, -unreadable, reversed_faces)

    def _infer_index_data_offset(self, header: _Header, batches: list[_Batch], prim_groups: list[_PrimGroup]) -> tuple[int, int, tuple[int, int, int, int, int]]:
        base_offset = header.base + int(header.index_offset)
        candidate_extras = (0x00, 0x10, 0x08, 0x04, 0x0C, 0x14, 0x20)
        best_offset = base_offset
        best_extra = 0
        best_score: tuple[int, int, int, int, int, int] = (-1, -999999, -999999, -999999, -1, -999999)
        for extra in candidate_extras:
            candidate = base_offset + int(extra)
            score = self._score_index_data_start(header, batches, prim_groups, candidate)
            if score[0] < 0:
                continue
            # Prefer streams with all triangles valid; use the documented direct
            # offset as the tie-breaker unless +0x10 gives a strictly better
            # score.  This supports Underworld-style prefixed face data without
            # shifting clean PCD9 index buffers.
            tie_break = 1 if int(extra) == 0 else 0
            ranked = (score[0], score[1], score[2], score[3], score[4], tie_break)
            if ranked > best_score:
                best_score = ranked
                best_offset = candidate
                best_extra = int(extra)
        if best_offset != base_offset:
            logger.info(
                'Next-gen model %s uses index data at indexOffset+0x%X; selected by validation score=%s',
                self._file_name,
                best_offset - base_offset,
                best_score[:5],
            )
        return int(best_offset), int(best_extra), best_score[:5]

    def _read_indices(self, header: _Header, batches: list[_Batch], prim_groups: list[_PrimGroup]) -> tuple[list[int], int, tuple[int, int, int, int, int]]:
        offset, extra, score = self._infer_index_data_offset(header, batches, prim_groups)
        count = max(0, int(header.num_indices))
        return self._read_raw_indices_at(offset, count), int(extra), score

    @staticmethod
    def _copy_material_to_strip(strip: TextureStrip, material: _MaterialData) -> None:
        strip.pc_nextgen_material_id = int(material.material_id)
        strip.pc_nextgen_material_record_offset = int(material.record_offset)
        strip.pc_nextgen_asset_id_hi = int(material.asset_id_hi)
        strip.pc_nextgen_asset_id_lo = int(material.asset_id_lo)
        strip.pc_nextgen_asset_id_padding_hex = str(getattr(material, 'asset_id_padding_hex', '') or '')
        strip.pc_nextgen_material_record_hex = str(getattr(material, 'material_record_hex', '') or '')
        strip.pc_nextgen_blend_mode = int(material.blend_mode)
        strip.pc_nextgen_combiner_type = int(material.combiner_type)
        strip.pc_nextgen_material_flags = int(material.flags)
        strip.pc_nextgen_opacity = float(material.opacity)
        strip.pc_nextgen_poly_flags = int(material.poly_flags)
        strip.pc_nextgen_uv_auto_scroll_speed = int(material.uv_auto_scroll_speed)
        strip.pc_nextgen_sort_bias = float(material.sort_bias)
        strip.pc_nextgen_detail_range_mul = float(material.detail_range_mul)
        strip.pc_nextgen_detail_scale = float(material.detail_scale)
        strip.pc_nextgen_parallax_scale = float(material.parallax_scale)
        strip.pc_nextgen_parallax_offset = float(material.parallax_offset)
        strip.pc_nextgen_specular_power = float(material.specular_power)
        strip.pc_nextgen_specular_shift0 = float(material.specular_shift0)
        strip.pc_nextgen_specular_shift1 = float(material.specular_shift1)
        strip.pc_nextgen_rim_light_color = tuple(float(value) for value in material.rim_light_color)
        strip.pc_nextgen_rim_light_intensity = float(material.rim_light_intensity)
        strip.pc_nextgen_water_blend_bias = float(material.water_blend_bias)
        strip.pc_nextgen_water_blend_exponent = float(material.water_blend_exponent)
        strip.pc_nextgen_water_deep_color = tuple(float(value) for value in material.water_deep_color)
        strip.pc_nextgen_local_num_pixmaps = int(material.local_num_pixmaps)
        strip.pc_nextgen_layer_texture_ids = [int(layer.texture_id) for layer in material.layers]
        strip.pc_nextgen_layer_texture_indices = [int(layer.texture_index) for layer in material.layers]
        strip.pc_nextgen_layer_enabled = [int(layer.enabled) for layer in material.layers]
        strip.pc_nextgen_layer_colors = [tuple(float(value) for value in layer.color) for layer in material.layers]
        strip.pc_nextgen_layer_texcoord_sources = [int(layer.texcoord_source) for layer in material.layers]
        strip.pc_nextgen_layer_modifiers = [int(layer.modifier) for layer in material.layers]
        strip.pc_nextgen_layer_param_ids = [int(layer.param_id) for layer in material.layers]
        strip.pc_nextgen_layer_constants = [tuple(float(value) for value in layer.constant) for layer in material.layers]
        strip.pc_nextgen_layer_num_textures = [int(layer.num_textures) for layer in material.layers]
        strip.pc_nextgen_shader_indices = [int(value) for value in material.shader_indices]
        strip.pc_nextgen_special_material_flag = bool(material.special_material_flag)
        strip.pc_nextgen_diffuse_texture_id = int(material.diffuse_id)
        strip.pc_nextgen_normal_texture_id = int(material.normal_id)
        strip.pc_nextgen_specular_texture_id = int(material.specular_id)

    def _parse_strips(self, header: _Header, batches: list[_Batch], prim_groups: list[_PrimGroup], materials: list[_MaterialData], shader_table_ids: list[int] | None = None) -> list[TextureStrip]:
        indices, index_data_extra, index_score = self._read_indices(header, batches, prim_groups)
        strips: list[TextureStrip] = []
        group_cursor = 0
        for batch in batches:
            for _local_group_index in range(int(batch.num_prim_groups)):
                if group_cursor >= len(prim_groups):
                    break
                group = prim_groups[group_cursor]
                group_cursor += 1
                index_start = int(group.base_index)
                index_count = int(group.num_primitives) * 3
                if index_start < 0 or index_count <= 0 or index_start + index_count > len(indices):
                    continue
                tri_indices: list[int] = []
                local_triangles: list[list[int]] = []
                reversed_winding_count = 0
                raw_values = indices[index_start:index_start + index_count]
                for face_index in range(0, len(raw_values) - (len(raw_values) % 3), 3):
                    local_triangle, reversed_winding = self._decode_nextgen_triangle(raw_values[face_index:face_index + 3], batch)
                    if reversed_winding:
                        reversed_winding_count += 1
                    local_triangles.append([int(value) for value in local_triangle])
                    tri_indices.extend(int(batch.vertex_start) + int(value) for value in local_triangle)
                double_wound_pair_count = self._count_double_wound_triangle_pairs(local_triangles)
                if double_wound_pair_count > 0:
                    collapsed_triangles = self._collapse_double_wound_triangle_pairs(local_triangles)
                    tri_indices = []
                    for local_triangle in collapsed_triangles:
                        tri_indices.extend(int(batch.vertex_start) + int(value) for value in local_triangle[:3])
                material_index = int(group.material_index)
                material = materials[material_index] if 0 <= material_index < len(materials) else _MaterialData(index=material_index)
                diffuse_id = int(material.diffuse_id)
                normal_id = int(material.normal_id)
                specular_id = int(material.specular_id)
                material_id = int(material.material_id)
                # PC next-gen does not have the old texture-strip draw_group
                # field.  In observed TR7 next-gen render data, PCMaterialData.id
                # stores drawgroup + 1 (id 1 = drawgroup 0, id 6 = drawgroup 5).
                # Falling back to the source model only when the id is absent keeps
                # drawgroup separation and visibility consistent with old-gen.
                draw_group = material_id - 1 if material_id > 0 else 0
                if material_id <= 0 and self.source_model is not None:
                    try:
                        source_strips = list(getattr(self.source_model, 'strips', []) or [])
                        if 0 <= (group_cursor - 1) < len(source_strips):
                            draw_group = int(getattr(source_strips[group_cursor - 1], 'draw_group', 0) or 0)
                    except Exception:
                        draw_group = 0
                strip = TextureStrip(
                    offset=group_cursor - 1,
                    vertex_count=len(tri_indices),
                    draw_group=int(draw_group),
                    tpageid=diffuse_id if diffuse_id >= 0 else 0,
                    sort_push=float(material.sort_bias),
                    scroll_offset=0.0,
                    indices=tri_indices,
                    material_group=material_index,
                    source_file=self._file_name,
                    tr8_batch_index=int(batch.index),
                    tr8_vertex_format_offset=int(batch.vertex_format),
                    tr8_geometry_source_index=int(batch.index),
                    pc_nextgen_reversed_winding_count=int(reversed_winding_count),
                    pc_nextgen_double_sided=bool(double_wound_pair_count > 0),
                    pc_nextgen_double_wound_pair_count=int(double_wound_pair_count),
                    pc_nextgen_index_data_offset_extra=int(index_data_extra),
                    pc_nextgen_index_data_validation_score=','.join(str(int(value)) for value in index_score),
                )
                self._copy_material_to_strip(strip, material)
                strip.pc_nextgen_shader_table_ids = [int(value) for value in (shader_table_ids or [])]
                # Keep these legacy fields populated as a broad compatibility fallback
                # for tools that inspect TextureStrip directly, but Blender material
                # creation now uses the pc_nextgen_* material state above.
                strip.tr8_diffuse_texture_id = diffuse_id
                strip.tr8_normal_texture_id = normal_id
                strip.tr8_mask_texture_id = specular_id
                strips.append(strip)
        return strips

    def _make_segments(self, header: _Header, vertex_count: int) -> list[Segment]:
        source_model = self.source_model
        if source_model is not None and getattr(source_model, 'segments', None):
            segments = deepcopy(list(source_model.segments))
            for segment in segments:
                if int(getattr(segment, 'first_vertex', -1)) < 0:
                    segment.first_vertex = 0
                if int(getattr(segment, 'last_vertex', -1)) < int(getattr(segment, 'first_vertex', 0)):
                    segment.last_vertex = max(0, int(vertex_count) - 1)
            return segments
        return [Segment(
            index=0,
            min_v=header.box_min,
            max_v=header.box_max,
            pivot=(0.0, 0.0, 0.0, 0.0),
            flags=0,
            first_vertex=0,
            last_vertex=max(0, int(vertex_count) - 1),
            parent=-1,
            hinfo=0,
        )]
