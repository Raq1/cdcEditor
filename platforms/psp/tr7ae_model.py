from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Tuple

from ...core.log import logger
from ...core.model_types import BoneMirrorEntry, HBox, HCapsule, HMarker, HSphere, MFace, MVertex, ModelData, Segment, Target, TextureStrip, VirtSegment
from ..pc.tr7ae_model import TRModelParser
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..common.model import make_segments_only_model_data, parse_compact_segment, parse_compact_virt_segment


@dataclass(slots=True)
class PSPPrimitiveInfo:
    index: int
    offset: int
    vertex_count: int
    mode_word: int
    texture_id: int
    blend: int
    vertex_format_flags: int
    data_offset: int
    first_vertex_hint: int
    bone_palette: List[int]
    uv_rect: Tuple[float, float, float, float]
    raw: bytes
    data_context: SectionContext | None = None
    data_local_offset: int = 0

    @property
    def mode(self) -> int:
        return int(self.mode_word) & 0xFF

    @property
    def render_flags(self) -> int:
        return (int(self.mode_word) >> 8) & 0xFF

    @property
    def draw_group(self) -> int:
        # PSP stores the drawgroup in the high byte of the mode word.  The Lara
        # sample uses this to separate drawgroups 0 and 27.
        return self.render_flags

    @property
    def blend_value(self) -> int:
        return int(self.blend)

    @property
    def blend_shim(self) -> int:
        # Compatibility shim for the existing PC-style 4-bit tpage blend field.
        return int(self.blend) & 0xF


class TRPSPModelParser(TRModelParser):

    PSP_MODEL_VERSION = 0x04C20453
    PRIMITIVE_STRIDES = (0xA0, 0x9C)
    PRIMITIVE_STRIDE = 0xA0  # Fallback/default only; real stride is detected per table.
    PSP_ENV_MAPPING_FLAG = 0x00000400
    PSP_ENV_MAPPING_VALUE = 64

    def _segment_record_size(self) -> int:
        return 32

    def _virt_segment_record_size(self) -> int:
        return 32

    def _parse_segment(self, context: SectionContext, index: int) -> Segment:
        return parse_compact_segment(context, index)

    def _parse_virt_segment(self, context: SectionContext, virt_index: int) -> VirtSegment:
        return parse_compact_virt_segment(context, virt_index)


    @staticmethod
    def _canonicalize_psp_tpageid(texture_id: int, blend: int = 0) -> int:
        return (int(texture_id) & 0x1FFF) | ((int(blend) & 0xF) << 13)

    @classmethod
    def _psp_primitive_env_mapping_value(cls, primitive: PSPPrimitiveInfo) -> int:
        return cls.PSP_ENV_MAPPING_VALUE if (int(primitive.vertex_format_flags) & cls.PSP_ENV_MAPPING_FLAG) else 0

    def _parse_psp_primitive_info_table_with_stride(self, context: SectionContext, absolute_offset: int, stride: int) -> List[PSPPrimitiveInfo]:
        if absolute_offset <= 0 or absolute_offset >= context.file_size:
            return []
        stride = int(stride)
        br = context.reader
        previous = br.tell()
        entries: List[PSPPrimitiveInfo] = []
        try:
            current = int(absolute_offset)
            index = 0
            while current + 4 <= context.data_end:
                first_word = int.from_bytes(br.peek(4, offset=current), 'little', signed=False)
                if first_word == 0:
                    break
                if current + stride > context.data_end:
                    break
                raw = br.peek(stride, offset=current)
                vertex_count = struct.unpack_from('<H', raw, 0x00)[0]
                mode_word = struct.unpack_from('<H', raw, 0x02)[0]
                texture_id = struct.unpack_from('<H', raw, 0x04)[0]
                blend = struct.unpack_from('<H', raw, 0x06)[0]
                vertex_format_flags = struct.unpack_from('<I', raw, 0x08)[0]
                data_offset = struct.unpack_from('<I', raw, 0x0C)[0]
                first_vertex_hint = struct.unpack_from('<H', raw, 0x10)[0]
                bone_palette = [int(struct.unpack_from('<h', raw, 0x10 + (slot * 2))[0]) for slot in range(8)]
                uv_rect_offset = 0x24 if int(stride) >= 0xA0 else 0x20
                uv_rect = tuple(float(v) for v in struct.unpack_from('<4f', raw, uv_rect_offset))

                mode = int(mode_word) & 0xFF
                if vertex_count <= 0:
                    break
                if mode <= 0 or mode > 8:
                    break

                entries.append(PSPPrimitiveInfo(
                    index=index,
                    offset=current,
                    vertex_count=int(vertex_count),
                    mode_word=int(mode_word),
                    texture_id=int(texture_id),
                    blend=int(blend),
                    vertex_format_flags=int(vertex_format_flags),
                    data_offset=int(data_offset),
                    first_vertex_hint=int(first_vertex_hint),
                    bone_palette=bone_palette,
                    uv_rect=uv_rect,
                    raw=raw,
                ))
                current += stride
                index += 1
        finally:
            br.seek(previous)
        return entries

    @staticmethod
    def _psp_uv_rect_score(rect: Tuple[float, float, float, float]) -> int:
        try:
            u0, u1, v0, v1 = [float(value) for value in rect]
        except Exception:
            return 0
        if not all(math.isfinite(value) for value in (u0, u1, v0, v1)):
            return 0
        du = u1 - u0
        dv = v1 - v0
        if du <= 0.0 or dv <= 0.0:
            return 1
        if abs(du) > 16.0 or abs(dv) > 16.0:
            return 1
        return 3

    @staticmethod
    def _score_psp_primitive_table(context: SectionContext, primitives: List[PSPPrimitiveInfo]) -> tuple[int, int, int, int, int]:
        valid_modes = 0
        relocations = 0
        uv_rect_score = 0
        plausible_textures = 0
        for primitive in primitives:
            mode = int(primitive.mode)
            if 1 <= mode <= 8:
                valid_modes += 1
            if int(primitive.texture_id) < 8192:
                plausible_textures += 1
            field_local_offset = (int(primitive.offset) - int(context.data_start)) + 0x0C
            if field_local_offset in context.section_info.relocations_by_offset:
                relocations += 1
            uv_rect_score += TRPSPModelParser._psp_uv_rect_score(primitive.uv_rect)
        return (
            len(primitives),
            relocations,
            valid_modes,
            uv_rect_score,
            plausible_textures,
        )

    def _parse_psp_primitive_info_table(self, context: SectionContext, absolute_offset: int) -> List[PSPPrimitiveInfo]:
        best_stride = self.PRIMITIVE_STRIDE
        best_entries: List[PSPPrimitiveInfo] = []
        best_score: tuple[int, int, int, int, int] | None = None
        for stride in self.PRIMITIVE_STRIDES:
            entries = self._parse_psp_primitive_info_table_with_stride(context, absolute_offset, int(stride))
            score = self._score_psp_primitive_table(context, entries)
            if best_score is None or score > best_score:
                best_score = score
                best_entries = entries
                best_stride = int(stride)
        logger.debug(
            'Read %d PSP PrimitiveInfo records from %s:0x%X using stride=0x%X score=%s',
            len(best_entries),
            context.file_name,
            absolute_offset,
            int(best_stride),
            best_score,
        )
        return best_entries

    @staticmethod
    def _s8(value: int) -> int:
        value = int(value) & 0xFF
        return value - 0x100 if value >= 0x80 else value

    @staticmethod
    def _psp_default_vertex_stride(mode: int) -> int:
        mode = int(mode) & 0xFF
        if mode in {1, 2}:
            return 16
        if mode in {3, 4}:
            return 18
        if mode in {5, 6}:
            return 20
        return 22

    @staticmethod
    def _psp_stride_candidates(mode: int) -> List[int]:
        mode = int(mode) & 0xFF
        if mode in {1, 2}:
            return [16, 14, 18]
        if mode in {3, 4}:
            return [18, 20, 16]
        if mode in {5, 6}:
            return [20, 18, 22]
        if mode in {7, 8}:
            return [22, 20, 18]
        return [16, 18, 20, 22, 14]

    @staticmethod
    def _psp_primitive_data_context(primitive: PSPPrimitiveInfo, fallback: SectionContext) -> SectionContext:
        return getattr(primitive, 'data_context', None) or fallback

    @staticmethod
    def _psp_primitive_data_offset(primitive: PSPPrimitiveInfo) -> int:
        if getattr(primitive, 'data_context', None) is not None:
            return int(getattr(primitive, 'data_local_offset', 0) or 0)
        return int(primitive.data_offset)

    def _resolve_psp_primitive_data_pointers(
        self,
        cache: SectionContextCache,
        primitive_context: SectionContext,
        primitives: List[PSPPrimitiveInfo],
    ) -> None:
        for primitive in primitives:
            field_local_offset = (int(primitive.offset) - int(primitive_context.data_start)) + 0x0C
            relocation = primitive_context.section_info.relocations_by_offset.get(field_local_offset)
            if relocation is None:
                primitive.data_context = None
                primitive.data_local_offset = int(primitive.data_offset)
                continue
            try:
                target_context, absolute_offset = resolve_pointer(
                    cache,
                    primitive_context,
                    field_local_offset,
                    int(primitive.data_offset),
                )
                primitive.data_context = target_context
                primitive.data_local_offset = max(0, int(absolute_offset) - int(target_context.data_start)) if absolute_offset else 0
            except Exception:
                logger.debug(
                    'Failed to resolve PSP primitive data pointer in %s primitive=%d field=0x%X',
                    primitive_context.file_name,
                    int(primitive.index),
                    int(field_local_offset),
                    exc_info=True,
                )
                primitive.data_context = None
                primitive.data_local_offset = int(primitive.data_offset)

    def _estimate_direct_vertex_stride(
        self,
        primitives: List[PSPPrimitiveInfo],
        primitive_index: int,
        geometry_context: SectionContext,
        header_size: int = 8,
        stream_end_offset: int = 0,
    ) -> int:
        primitive = primitives[primitive_index]
        primitive_offset = self._psp_primitive_data_offset(primitive)
        default_stride = self._psp_default_vertex_stride(primitive.mode)
        candidates = self._psp_stride_candidates(primitive.mode)

        first_vertex_prefix = b''
        first_vertex_offset = int(primitive_offset) + max(0, int(header_size))
        if 0 <= first_vertex_offset < int(geometry_context.data_size):
            try:
                first_vertex_prefix = geometry_context.reader.peek(
                    min(4, int(geometry_context.data_size) - first_vertex_offset),
                    offset=geometry_context.data_start + first_vertex_offset,
                )
            except Exception:
                first_vertex_prefix = b''

        preferred_stride = default_stride
        mode = int(primitive.mode) & 0xFF
        if mode == 1 and (not first_vertex_prefix or first_vertex_prefix[0] != 0x80):
            preferred_stride = 14
            candidates = [14, 16, 18]
        elif mode == 2:
            preferred_stride = 16
            candidates = [16, 18, 14]

        stream_end_offset = int(stream_end_offset or 0)
        candidate_limits = []
        for other in primitives[primitive_index + 1:]:
            other_context = self._psp_primitive_data_context(other, geometry_context)
            if other_context is not geometry_context:
                continue
            other_offset = self._psp_primitive_data_offset(other)
            if int(other_offset) > int(primitive_offset):
                candidate_limits.append(int(other_offset))
        if int(primitive_offset) < stream_end_offset <= int(geometry_context.data_size):
            candidate_limits.append(stream_end_offset)
        stream_limit = min(candidate_limits) if candidate_limits else int(geometry_context.data_size)
        stream_span = max(0, int(stream_limit) - int(primitive_offset))
        if stream_span <= 0 or primitive.vertex_count <= 0:
            return default_stride

        header_size = max(0, int(header_size))
        preferred_required = header_size + (int(primitive.vertex_count) * int(preferred_stride))
        if preferred_required <= stream_span:
            return int(preferred_stride)

        best_stride = preferred_stride
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
                # Fallback for any raw/non-aligned stream variant.
                padding_after_aligned_stream = stream_span - required
                misalignment = abs((stream_span - required) % 16)
            score = (
                0 if stride == preferred_stride else 1,
                padding_after_aligned_stream,
                misalignment,
                abs(int(stride) - int(preferred_stride)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_stride = int(stride)
        return int(best_stride)


    @staticmethod
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

    @staticmethod
    def _psp_normal_offset(blob_size: int) -> int:
        # PSP direct records keep xyz in the final 6 bytes, with packed signed
        # byte normals immediately before that final position group.  The byte
        # following nz is padding/aux.
        return max(0, int(blob_size) - 10)

    @classmethod
    def _psp_color_offset(cls, blob_size: int) -> int:
        return int(cls._psp_normal_offset(blob_size)) - 2

    @staticmethod
    def _decode_psp_r5g5a5_lighting(value: int) -> Tuple[int, int, int, int]:
        value = int(value) & 0xFFFF

        r5 = (value >> 0) & 0x1F
        g5 = (value >> 5) & 0x1F
        a5 = (value >> 11) & 0x1F

        r = int((r5 * 255 + 15) // 31)
        g = int((g5 * 255 + 15) // 31)
        a = int((a5 * 255 + 15) // 31)

        # PSP model vertex lighting is effectively neutral intensity data here.
        # Write the same half-intensity value to RGB so mask/packing noise does
        # not tint the imported mesh green.
        shade = int((r + g + 1) // 2) // 2
        return (shade, shade, shade, a)

    @classmethod
    def _read_psp_vertex_color(cls, blob: bytes) -> Tuple[int, int, int, int] | None:
        color_offset = cls._psp_color_offset(len(blob))
        if color_offset < 0 or color_offset + 2 > len(blob):
            return None
        try:
            return cls._decode_psp_r5g5a5_lighting(struct.unpack_from('<H', blob, color_offset)[0])
        except Exception:
            return None

    @staticmethod
    def _psp_weight_prefix_size(mode: int, blob_size: int) -> int:
        mode = int(mode) & 0xFF
        size = int(blob_size)
        if mode <= 0:
            return 0
        if mode == 1:
            # Rigid/static PSP records are 14 bytes with no weight byte.
            # Skinned one-weight records are 16 bytes and start with 0x80.
            return 1 if size >= 16 else 0
        if mode > 8:
            mode = 8
        # Leave room for UV(2), optional pad(1), aux16(2), normal/pad(4), xyz(6).
        min_tail = 14 + (1 if (mode & 1) else 0)
        return min(mode, max(0, size - min_tail))

    @classmethod
    def _psp_uv_offset(cls, mode: int, blob_size: int) -> int:
        # UV is two 8-bit GE texture-coordinate bytes immediately after the
        # actual weight prefix.  Odd weight counts have their alignment pad
        # after UV, not before it.
        return cls._psp_weight_prefix_size(mode, blob_size)

    @classmethod
    def _psp_weight_slot_count(cls, mode: int, blob_size: int) -> int:
        mode = int(mode) & 0xFF
        uv_offset = cls._psp_uv_offset(mode, blob_size)
        if 2 <= mode <= 8 and uv_offset > 0:
            return min(mode, uv_offset)
        return 0

    @staticmethod
    def _psp_palette_segment(value: int) -> int:
        # The palette stores sign-extended bone ids.  The sign is not part of
        # the segment index; it appears to encode handedness/matrix-side state.
        return abs(int(value))

    def _resolve_psp_vertex_skin(self, primitive: PSPPrimitiveInfo | None, blob: bytes) -> tuple[int, int, int, float]:
        fallback_segment = int(getattr(primitive, 'first_vertex_hint', 0) or 0) if primitive is not None else 0
        fallback_segment = self._psp_palette_segment(fallback_segment)
        if primitive is None or not blob:
            return fallback_segment, fallback_segment, -1, 0.0

        palette = list(getattr(primitive, 'bone_palette', []) or [])
        slot_count = min(len(palette), self._psp_weight_slot_count(primitive.mode, len(blob)))
        influences: list[tuple[int, int, int]] = []
        for slot in range(slot_count):
            weight = int(blob[slot])
            if weight <= 0:
                continue
            segment = self._psp_palette_segment(palette[slot] if slot < len(palette) else fallback_segment)
            influences.append((slot, segment, weight))

        if not influences:
            return fallback_segment, fallback_segment, -1, 0.0

        if len(influences) > 2:
            influences = influences[:2]

        primary_segment = int(influences[0][1])
        if len(influences) == 1:
            return primary_segment, primary_segment, -1, 0.0

        secondary = influences[1]
        total = max(1, int(influences[0][2]) + int(secondary[2]))
        secondary_weight = float(secondary[2]) / float(total)
        return primary_segment, primary_segment, int(secondary[1]), secondary_weight

    @staticmethod
    def _psp_uv_rect_looks_normalized(rect: Tuple[float, float, float, float]) -> bool:
        try:
            values = [float(value) for value in rect]
        except Exception:
            return False
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            return False
        # Older 0x9C tables use these floats as a normalized UV rectangle.
        # Newer 0xA0 tables commonly contain values like 43, 117, or 132 in
        # the first slot; applying those as UV bounds explodes the unwrap.
        if min(values) < -0.25 or max(values) > 1.25:
            return False
        u0, u1, v0, v1 = values
        if abs(u1 - u0) > 1.5 or abs(v1 - v0) > 1.5:
            return False
        return True

    def _read_psp_vertex_blob(self, blob: bytes, index: int, segment: int = 0, primitive: PSPPrimitiveInfo | None = None) -> MVertex:
        if len(blob) < 6:
            x = y = z = 0
        else:
            x, y, z = struct.unpack_from('<3h', blob, len(blob) - 6)

        bind_segment, primary_segment, secondary_segment, secondary_weight = self._resolve_psp_vertex_skin(primitive, blob)
        if primary_segment < 0:
            primary_segment = int(segment)
        if bind_segment < 0:
            bind_segment = int(primary_segment)

        effective_mode = primitive.mode if primitive is not None else self._psp_mode_from_blob_size(len(blob))
        uv_offset = self._psp_uv_offset(effective_mode, len(blob))
        normal_offset = self._psp_normal_offset(len(blob))
        nx = self._s8(blob[normal_offset]) if normal_offset < len(blob) else 0
        ny = self._s8(blob[normal_offset + 1]) if normal_offset + 1 < len(blob) else 0
        nz = self._s8(blob[normal_offset + 2]) if normal_offset + 2 < len(blob) else 0

        if uv_offset + 2 <= len(blob):
            uvx = int(blob[uv_offset])
            uvy = int(blob[uv_offset + 1])
        else:
            uvx = uvy = 0

        uv_decoded = None
        if primitive is not None:
            try:
                u0, u1, v0, v1 = primitive.uv_rect
                if all(math.isfinite(float(value)) for value in (u0, u1, v0, v1)):
                    uf = float(int(uvx) & 0xFF) / 254.0
                    vf = float(int(uvy) & 0xFF) / 254.0
                    uv_decoded = (
                        float(u0) + (float(u1) - float(u0)) * uf,
                        float(v0) + (float(v1) - float(v0)) * vf,
                    )
            except Exception:
                uv_decoded = None

        vertex_color = self._read_psp_vertex_color(blob)

        vertex = MVertex(
            index=index,
            position_raw=(int(x), int(y), int(z)),
            normal_raw=(int(nx), int(ny), int(nz)),
            segment=int(primary_segment),
            uv_raw=(int(uvx), int(uvy)),
            uv_decoded=uv_decoded,
            gc_bind_segment=int(bind_segment),
            gc_primary_segment=int(primary_segment),
            gc_secondary_segment=int(secondary_segment),
            gc_secondary_weight=float(secondary_weight),
            psp_color_rgba=vertex_color,
        )
        return vertex

    def _read_psp_vertex_16(self, blob: bytes, index: int, segment: int = 0, primitive: PSPPrimitiveInfo | None = None) -> MVertex:
        return self._read_psp_vertex_blob(blob, index, segment=segment, primitive=primitive)

    def _read_psp_vertex_20(self, blob: bytes, index: int, segment: int = 0, primitive: PSPPrimitiveInfo | None = None) -> MVertex:
        return self._read_psp_vertex_blob(blob, index, segment=segment, primitive=primitive)

    def _read_psp_vertex_22(self, blob: bytes, index: int, segment: int = 0, primitive: PSPPrimitiveInfo | None = None) -> MVertex:
        return self._read_psp_vertex_blob(blob, index, segment=segment, primitive=primitive)

    def _read_segment_ids(self, context: SectionContext, absolute_offset: int, count: int, num_segments: int) -> List[int]:
        if absolute_offset <= 0 or count <= 0 or absolute_offset >= context.file_size:
            return [0] * max(0, count)
        size = min(max(0, count), max(0, context.file_size - absolute_offset))
        data = context.reader.peek(size, offset=absolute_offset)
        ids: List[int] = []
        for value in data[:count]:
            ivalue = int(value)
            ids.append(ivalue if 0 <= ivalue < max(1, int(num_segments)) else 0)
        if len(ids) < count:
            ids.extend([0] * (count - len(ids)))
        return ids

    def _read_vertices(
        self,
        context: SectionContext,
        absolute_offset: int,
        count: int,
        stride: int,
        segment_ids: List[int],
    ) -> List[MVertex]:
        vertices: List[MVertex] = []
        if count <= 0 or absolute_offset <= 0:
            return vertices
        max_count = max(0, (context.file_size - absolute_offset) // max(1, stride))
        safe_count = min(int(count), max_count)
        if safe_count < count:
            logger.warning('PSP vertex read truncated in %s: requested=%d safe=%d stride=%d', context.file_name, count, safe_count, stride)
        for index in range(safe_count):
            blob = context.reader.peek(stride, offset=absolute_offset + (index * stride))
            segment = segment_ids[index] if index < len(segment_ids) else 0
            if stride == 16:
                vertices.append(self._read_psp_vertex_16(blob, index, segment=segment))
            elif stride == 20:
                vertices.append(self._read_psp_vertex_20(blob, index, segment=segment))
            else:
                vertices.append(self._read_psp_vertex_22(blob, index, segment=segment))
        return vertices

    @staticmethod
    def _strip_vertex_key(vertices: List[MVertex], index: int) -> tuple:
        if 0 <= int(index) < len(vertices):
            vertex = vertices[int(index)]
            return (
                tuple(int(v) for v in vertex.position_raw),
                int(vertex.segment),
                int(getattr(vertex, 'gc_primary_segment', -1)),
                int(getattr(vertex, 'gc_secondary_segment', -1)),
            )
        return ('idx', int(index))

    @classmethod
    def _strip_to_triangles(cls, sequence: List[int], vertices: List[MVertex] | None = None) -> List[int]:
        out: List[int] = []
        window: List[int] = []
        previous_key = None

        for raw_index in sequence:
            index = int(raw_index)
            key = cls._strip_vertex_key(vertices, index) if vertices is not None else ('idx', index)

            if previous_key is not None and key == previous_key:
                window = [index]
                previous_key = key
                continue

            window.append(index)
            previous_key = key
            if len(window) < 3:
                continue

            a, b, c = window[-3], window[-2], window[-1]
            key_a = cls._strip_vertex_key(vertices, a) if vertices is not None else ('idx', a)
            key_b = cls._strip_vertex_key(vertices, b) if vertices is not None else ('idx', b)
            key_c = cls._strip_vertex_key(vertices, c) if vertices is not None else ('idx', c)
            if key_a == key_b or key_b == key_c or key_a == key_c:
                continue

            local_i = len(window) - 1
            if local_i & 1:
                out.extend([b, a, c])
            else:
                out.extend([a, b, c])
        return out

    def _read_index_sequence(self, context: SectionContext, offset: int, count: int, num_vertices: int) -> tuple[List[int], int]:
        values, source_offset, _ratio = self._read_index_sequence_scored(context, offset, count, num_vertices)
        return values, source_offset

    def _read_index_sequence_scored(self, context: SectionContext, offset: int, count: int, num_vertices: int) -> tuple[List[int], int, float]:
        candidates = []
        for candidate in (offset, offset + 8, offset + 4, offset + 16):
            if candidate < 0 or candidate + (count * 2) > context.data_size:
                continue
            abs_off = context.data_start + candidate
            raw = context.reader.peek(count * 2, offset=abs_off)
            values = [struct.unpack_from('<H', raw, i * 2)[0] for i in range(count)]
            valid = sum(1 for value in values if 0 <= int(value) < max(1, int(num_vertices)))
            ratio = valid / max(1, count)
            # Prefer fully valid candidates; if tied, prefer smaller offsets to
            # preserve the old behaviour for streams that intentionally include
            # leading strip restart zeros.
            score = (ratio, -candidate)
            candidates.append((score, values, candidate, ratio))
        if not candidates:
            return [], -1, 0.0
        candidates.sort(key=lambda item: item[0], reverse=True)
        values = [int(value) for value in candidates[0][1] if 0 <= int(value) < max(1, int(num_vertices))]
        return values, int(candidates[0][2]), float(candidates[0][3])

    @staticmethod
    def _looks_like_stream_header(blob: bytes) -> bool:
        if len(blob) < 8:
            return False
        if struct.unpack_from('<I', blob, 4)[0] != 0:
            return False
        try:
            value = struct.unpack_from('<f', blob, 0)[0]
        except Exception:
            return False
        return math.isfinite(value) and 1.0e-6 < abs(float(value)) < 1.0e6

    def _append_direct_vertices(
        self,
        vertices: List[MVertex],
        context: SectionContext,
        stream_offset: int,
        count: int,
        stride: int,
        header_size: int,
        segment: int = 0,
        primitive: PSPPrimitiveInfo | None = None,
    ) -> List[int]:
        sequence: List[int] = []
        if count <= 0 or stream_offset < 0:
            return sequence
        start = int(stream_offset) + int(header_size)
        for _ in range(int(count)):
            if start + int(stride) > context.data_size:
                break
            blob = context.reader.peek(int(stride), offset=context.data_start + start)
            index = len(vertices)
            if int(stride) == 16:
                vertex = self._read_psp_vertex_16(blob, index, segment=segment, primitive=primitive)
            elif int(stride) == 20:
                vertex = self._read_psp_vertex_20(blob, index, segment=segment, primitive=primitive)
            else:
                vertex = self._read_psp_vertex_22(blob, index, segment=segment, primitive=primitive)
            vertices.append(vertex)
            sequence.append(index)
            start += int(stride)
        return sequence

    def _psp_indexed_geometry_score(self, primitives: List[PSPPrimitiveInfo], geometry_context: SectionContext, num_vertices: int) -> float:
        ratios: List[float] = []
        for primitive in primitives:
            if primitive.vertex_count < 3:
                continue
            primitive_context = self._psp_primitive_data_context(primitive, geometry_context)
            local_offset = self._psp_primitive_data_offset(primitive)
            _sequence, _source_offset, ratio = self._read_index_sequence_scored(
                primitive_context,
                local_offset,
                primitive.vertex_count,
                num_vertices,
            )
            ratios.append(float(ratio))
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    def _build_psp_strips(
        self,
        primitives: List[PSPPrimitiveInfo],
        geometry_context: SectionContext,
        vertices: List[MVertex],
        vertex_stride: int,
        indexed_geometry: bool,
        compact_vertex_base: int = 0,
    ) -> tuple[List[MFace], List[TextureStrip]]:
        faces: List[MFace] = []
        strips: List[TextureStrip] = []
        material_group_by_key: dict[tuple[int, int, int, int, int, int], int] = {}
        direct_stream_end = int(compact_vertex_base or 0)

        def material_group_for(primitive: PSPPrimitiveInfo, canonical_tpageid: int, env_mapping: int) -> int:
            key = (
                int(primitive.draw_group),
                int(primitive.blend_value),
                int(canonical_tpageid) & 0xFFFFFFFF,
                int(env_mapping),
                int(primitive.render_flags),
                int(primitive.vertex_format_flags) & 0xFFFFFFFF,
            )
            group = material_group_by_key.get(key)
            if group is None:
                group = len(material_group_by_key)
                material_group_by_key[key] = group
            return group

        def emit_strip(primitive: PSPPrimitiveInfo, source_offset: int, sequence: List[int], canonical_tpageid: int) -> None:
            strip_indices = self._strip_to_triangles(sequence, vertices)
            if not strip_indices:
                return
            for i in range(0, len(strip_indices), 3):
                tri = strip_indices[i:i + 3]
                if len(tri) != 3:
                    continue
                faces.append(MFace(index=len(faces), v0=tri[0], v1=tri[1], v2=tri[2], same_vert_bits=0))
            env_mapping = self._psp_primitive_env_mapping_value(primitive)
            strips.append(TextureStrip(
                offset=int(source_offset),
                vertex_count=len(strip_indices),
                draw_group=int(primitive.draw_group),
                tpageid=int(canonical_tpageid),
                sort_push=0.0,
                scroll_offset=0.0,
                env_mapping=int(env_mapping),
                next_texture=-1,
                indices=strip_indices,
                material_group=material_group_for(primitive, canonical_tpageid, env_mapping),
                ps2_tpageid_raw=int((primitive.texture_id & 0xFFFF) | ((primitive.mode_word & 0xFFFF) << 16)),
                psp_mode_word=int(primitive.mode_word),
                psp_vertex_mode=int(primitive.mode),
                psp_render_flags=int(primitive.render_flags),
                psp_blend=int(primitive.blend_value),
                psp_vertex_format_flags=int(primitive.vertex_format_flags),
                psp_texture_id=int(primitive.texture_id),
            ))

        for primitive in primitives:
            if primitive.vertex_count < 3:
                continue
            canonical_tpageid = self._canonicalize_psp_tpageid(primitive.texture_id, primitive.blend_shim)
            primitive_context = self._psp_primitive_data_context(primitive, geometry_context)
            local_offset = self._psp_primitive_data_offset(primitive)

            if indexed_geometry:
                sequence, source_offset = self._read_index_sequence(
                    primitive_context,
                    local_offset,
                    primitive.vertex_count,
                    len(vertices),
                )
                emit_strip(primitive, source_offset, sequence, canonical_tpageid)
                continue

            header_size = 8
            if 0 <= local_offset < primitive_context.data_size:
                header_blob = primitive_context.reader.peek(
                    min(8, max(0, primitive_context.data_size - local_offset)),
                    offset=primitive_context.data_start + local_offset,
                )
                if not self._looks_like_stream_header(header_blob):
                    header_size = 0

            direct_stride = self._estimate_direct_vertex_stride(
                primitives,
                primitive.index,
                primitive_context,
                header_size=header_size,
                stream_end_offset=direct_stream_end if primitive_context is geometry_context else 0,
            )

            sequence = self._append_direct_vertices(
                vertices,
                primitive_context,
                local_offset,
                primitive.vertex_count,
                direct_stride,
                header_size,
                segment=primitive.first_vertex_hint if primitive.first_vertex_hint >= 0 else 0,
                primitive=primitive,
            )

            emit_strip(primitive, local_offset + header_size, sequence, canonical_tpageid)
        return faces, strips

    @staticmethod
    def _psp_vertex_dedupe_key(vertex: MVertex) -> tuple:
        uv_decoded = getattr(vertex, 'uv_decoded', None)
        if uv_decoded is not None:
            uv_key = tuple(round(float(value), 7) for value in uv_decoded)
        else:
            uv_key = None
        return (
            tuple(int(value) for value in vertex.position_raw),
            tuple(int(value) for value in vertex.normal_raw),
            int(vertex.segment),
            tuple(int(value) for value in vertex.uv_raw),
            uv_key,
            int(getattr(vertex, 'gc_bind_segment', -1)),
            int(getattr(vertex, 'gc_primary_segment', -1)),
            int(getattr(vertex, 'gc_secondary_segment', -1)),
            round(float(getattr(vertex, 'gc_secondary_weight', 0.0)), 7),
            int(getattr(vertex, 'gc_transform_id', -1)),
            tuple(int(value) for value in (getattr(vertex, 'psp_color_rgba', None) or (255, 255, 255, 255))),
        )

    def _dedupe_psp_vertices(
        self,
        vertices: List[MVertex],
        faces: List[MFace],
        strips: List[TextureStrip],
    ) -> List[MVertex]:
        if not vertices:
            return vertices

        key_to_new_index: dict[tuple, int] = {}
        remap: List[int] = [0] * len(vertices)
        deduped: List[MVertex] = []

        for old_index, vertex in enumerate(vertices):
            key = self._psp_vertex_dedupe_key(vertex)
            new_index = key_to_new_index.get(key)
            if new_index is None:
                new_index = len(deduped)
                key_to_new_index[key] = new_index
                vertex.index = new_index
                deduped.append(vertex)
            remap[old_index] = new_index

        if len(deduped) == len(vertices):
            return vertices

        for face in faces:
            if 0 <= face.v0 < len(remap):
                face.v0 = remap[face.v0]
            if 0 <= face.v1 < len(remap):
                face.v1 = remap[face.v1]
            if 0 <= face.v2 < len(remap):
                face.v2 = remap[face.v2]

        for strip in strips:
            remapped_indices = [remap[index] if 0 <= int(index) < len(remap) else int(index) for index in strip.indices]
            filtered_indices: List[int] = []
            usable_count = len(remapped_indices) - (len(remapped_indices) % 3)
            for i in range(0, usable_count, 3):
                tri = remapped_indices[i:i + 3]
                if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                    continue
                filtered_indices.extend(tri)
            strip.indices = filtered_indices
            strip.vertex_count = len(filtered_indices)

        filtered_faces: List[MFace] = []
        for face in faces:
            if face.v0 == face.v1 or face.v1 == face.v2 or face.v0 == face.v2:
                continue
            face.index = len(filtered_faces)
            filtered_faces.append(face)
        if len(filtered_faces) != len(faces):
            faces[:] = filtered_faces

        logger.debug('Deduped PSP vertices: %d -> %d', len(vertices), len(deduped))
        return deduped

    @staticmethod
    def _psp_model_vertex_colors(vertices: List[MVertex]) -> List[Tuple[int, int, int, int]] | None:
        if not vertices:
            return None
        colors: List[Tuple[int, int, int, int]] = []
        has_non_white = False
        for vertex in vertices:
            color = getattr(vertex, 'psp_color_rgba', None) or (255, 255, 255, 255)
            rgba = tuple(max(0, min(255, int(value))) for value in color)
            colors.append(rgba)
            if rgba != (255, 255, 255, 255):
                has_non_white = True
        return colors if has_non_white else None

    def _read_u32_pointer_at(self, cache: SectionContextCache, context: SectionContext, local_offset: int) -> Tuple[int, SectionContext, int]:
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            raw = context.reader.u32()
            target_ctx, abs_off = resolve_pointer(cache, context, int(local_offset), raw)
            return raw, target_ctx, abs_off
        finally:
            context.reader.seek(previous)

    def _parse_model(self, cache: SectionContextCache, context: SectionContext) -> ModelData:
        br = context.reader
        br.seek(context.data_start)

        version = br.i32()
        num_segments = br.i32()
        num_virt_segments = br.i32()
        segment_list_raw, segment_list_ctx, segment_list_abs = self._read_u32_pointer(cache, context)

        segments: List[Segment] = []
        virt_segments: List[VirtSegment] = []
        hmarkers: List[HMarker] = []
        hspheres: List[HSphere] = []
        hboxes: List[HBox] = []
        hcapsules: List[HCapsule] = []
        targets: List[Target] = []

        if num_segments > 0 and segment_list_abs:
            segment_list_ctx.reader.seek(segment_list_abs)
            for index in range(num_segments):
                segment_record_start = segment_list_ctx.reader.tell()
                segment = self._parse_segment(segment_list_ctx, index)
                segments.append(segment)
                if self.import_hinfo:
                    hinfo_field_local_offset = (segment_record_start - segment_list_ctx.data_start) + (self._segment_record_size() - 4)
                    if segment.hinfo or hinfo_field_local_offset in segment_list_ctx.section_info.relocations_by_offset:
                        hinfo_ctx, hinfo_abs = resolve_pointer(cache, segment_list_ctx, hinfo_field_local_offset, segment.hinfo)
                        if hinfo_abs:
                            hspheres.extend(self._parse_hinfo_hspheres(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_sphere_index_start=len(hspheres)))
                            hboxes.extend(self._parse_hinfo_hboxes(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_hbox_index_start=len(hboxes)))
                            hmarkers.extend(self._parse_hinfo_hmarkers(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_marker_index_start=len(hmarkers)))
                            hcapsules.extend(self._parse_hinfo_hcapsules(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_capsule_index_start=len(hcapsules)))
            for virt_index in range(max(0, int(num_virt_segments))):
                virt_segments.append(self._parse_virt_segment(segment_list_ctx, virt_index))

        if self.parse_segments_only:
            return make_segments_only_model_data(
                version=version,
                segments=segments,
                virt_segments=virt_segments,
                hmarkers=hmarkers,
                hspheres=hspheres,
                hboxes=hboxes,
                hcapsules=hcapsules,
                uv_format='psp',
            )

        br.seek(context.data_start + 0x10)
        model_scale = br.vec4()
        num_vertices = br.i32()
        segment_ids_raw, segment_ids_ctx, segment_ids_abs = self._read_u32_pointer(cache, context)
        _unknown_count_or_flags = br.i32()
        _aux_raw, _aux_ctx, _aux_abs = self._read_u32_pointer(cache, context)
        vertex_data_raw, vertex_data_ctx, vertex_data_abs = self._read_u32_pointer(cache, context)
        _vertex_end_raw, _vertex_end_ctx, _vertex_end_abs = self._read_u32_pointer(cache, context)

        max_rad = 0.0
        max_rad_sq = 0.0
        bone_mirror_entries: List[BoneMirrorEntry] = []
        cdc_render_data_id = 0
        target_list_raw = 0
        target_list_ctx = context
        target_list_abs = 0
        primitive_info_raw = 0
        primitive_info_ctx = context
        primitive_info_abs = 0

        try:
            br.seek(context.data_start + 0x60)
            max_rad = br.f32()
            max_rad_sq = br.f32()
        except Exception:
            pass

        try:
            bone_mirror_raw, bone_mirror_ctx, bone_mirror_abs = self._read_u32_pointer_at(cache, context, 0x98)
            if bone_mirror_abs:
                bone_mirror_entries = self._parse_bone_mirror_entries(bone_mirror_ctx, bone_mirror_abs)
        except Exception:
            logger.debug('PSP bone mirror read failed in %s', context.file_name, exc_info=True)

        try:
            primitive_info_raw, primitive_info_ctx, primitive_info_abs = self._read_u32_pointer_at(cache, context, 0x7C)
        except Exception:
            primitive_info_raw, primitive_info_ctx, primitive_info_abs = 0, context, 0

        try:
            target_list_raw, target_list_ctx, target_list_abs = self._read_u32_pointer_at(cache, context, 0x88)
            br.seek(context.data_start + 0x8C)
            cdc_render_data_id = br.u32()
        except Exception:
            target_list_raw, target_list_ctx, target_list_abs = 0, context, 0
            cdc_render_data_id = 0

        primitives = self._parse_psp_primitive_info_table(primitive_info_ctx, primitive_info_abs)
        self._resolve_psp_primitive_data_pointers(cache, primitive_info_ctx, primitives)
        all_mode8 = bool(primitives) and all(primitive.mode == 8 for primitive in primitives)
        indexed_geometry = False
        indexed_geometry_score = 0.0
        compact_vertex_base = 0

        if all_mode8:
            vertex_stride = 22
            has_global_vertex_buffer = bool(vertex_data_abs and int(num_vertices) > 0)
            if has_global_vertex_buffer:
                indexed_geometry_score = self._psp_indexed_geometry_score(primitives, vertex_data_ctx, int(num_vertices))
                indexed_geometry = indexed_geometry_score >= 0.85
                segment_ids = self._read_segment_ids(segment_ids_ctx, segment_ids_abs, int(num_vertices), int(num_segments))
                vertices = self._read_vertices(vertex_data_ctx, vertex_data_abs, int(num_vertices), vertex_stride, segment_ids)
                if not indexed_geometry:
                    vertices = []
            else:
                segment_ids = []
                vertices = []
        else:
            vertex_stride = 16
            if _aux_ctx is not context:
                vertex_data_ctx = _aux_ctx
            elif segment_ids_ctx is not context:
                vertex_data_ctx = segment_ids_ctx
            elif vertex_data_ctx is context and primitives:
                try:
                    vertex_data_ctx, _unused_abs = resolve_pointer(cache, primitive_info_ctx, 0x0C, int(primitives[0].data_offset))
                except Exception:
                    vertex_data_ctx = context
            segment_ids = self._read_segment_ids(segment_ids_ctx, segment_ids_abs, int(num_vertices), int(num_segments))
            vertices: List[MVertex] = []

        direct_stream_end = 0
        if not indexed_geometry and vertex_data_ctx is _aux_ctx and 0 < int(_aux_raw) <= int(vertex_data_ctx.data_size):
            direct_stream_end = int(_aux_raw)

        faces, strips = self._build_psp_strips(
            primitives,
            vertex_data_ctx,
            vertices,
            vertex_stride,
            indexed_geometry,
            compact_vertex_base=direct_stream_end,
        )

        vertices = self._dedupe_psp_vertices(vertices, faces, strips)

        if not indexed_geometry and segment_ids_abs > 0:
            for vertex in vertices:
                if int(getattr(vertex, 'gc_primary_segment', -1)) >= 0:
                    continue
                if 0 <= vertex.index < len(segment_ids):
                    vertex.segment = segment_ids[vertex.index]

        # Keep segment ranges useful even when PSP segment id arrays are partial.
        for segment in segments:
            assigned = [vertex.index for vertex in vertices if int(vertex.segment) == int(segment.index)]
            segment.first_vertex = min(assigned) if assigned else -1
            segment.last_vertex = max(assigned) if assigned else -1

        if target_list_abs:
            # PSP sample does not expose a reliable target count in the guessed
            # header yet; keep this disabled until another sample confirms it.
            targets = []

        vertex_colors = self._psp_model_vertex_colors(vertices)

        logger.debug(
            'PSP model %s version=0x%X segments=%d vertices=%d primitiveInfos=%d strips=%d faces=%d stride=%d indexed=%s indexScore=%.3f vertexData=0x%X',
            context.file_name,
            int(version) & 0xFFFFFFFF,
            len(segments),
            len(vertices),
            len(primitives),
            len(strips),
            len(faces),
            vertex_stride,
            bool(indexed_geometry),
            float(indexed_geometry_score),
            int(vertex_data_raw),
        )

        return ModelData(
            version=version,
            model_scale=model_scale,
            segments=segments,
            virt_segments=virt_segments,
            vertices=vertices,
            faces=faces,
            strips=strips,
            vertex_colors=vertex_colors,
            env_mapped_face_indices=[],
            eye_ref_env_mapped_face_indices=[],
            hmarkers=hmarkers,
            hspheres=hspheres,
            hboxes=hboxes,
            hcapsules=hcapsules,
            targets=targets,
            markups=[],
            max_rad=max_rad,
            max_rad_sq=max_rad_sq,
            bone_mirror_entries=bone_mirror_entries,
            cdc_render_data_id=cdc_render_data_id,
            uv_format='psp',
        )
