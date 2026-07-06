from __future__ import annotations

from typing import List, Tuple
from math import sqrt
from collections import Counter

from ...core.log import logger
from ...core.model_types import HBox, HCapsule, HMarker, HSphere, MVertex, ModelData, Segment, Target, TextureStrip, VirtSegment
from ..pc.tr7ae_model import TRModelParser
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..common.model import make_segments_only_model_data, make_weighted_virt_segment, resolve_weighted_transform


def _sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    mask = (1 << bits) - 1
    value &= mask
    return value - (1 << bits) if (value & sign_bit) else value


class TRXboxModelParser(TRModelParser):

    XBOX_VERTEX_SIZE = 20
    XBOX_FACE_DATA_RECORD_MAGIC = b"\xFC\x17\x04\x00"
    XBOX_UV_SCALE = 4096.0

    def __init__(self, filepath: str, import_hinfo: bool = True, import_markups: bool = True, parse_segments_only: bool = False):
        super().__init__(filepath, import_hinfo=import_hinfo, import_markups=import_markups, endian="<", parse_segments_only=parse_segments_only)

    def _is_gamecube_layout(self) -> bool:
        return True


    @staticmethod
    def _decode_xbox_packed_normal(raw: int) -> tuple[int, int, int]:
        # Not too sure about this one
        nx = _sign_extend(raw & 0x7FF, 11) / 1023.0
        ny = _sign_extend((raw >> 11) & 0x7FF, 11) / 1023.0
        nz = _sign_extend((raw >> 22) & 0x3FF, 10) / 511.0

        length = sqrt((nx * nx) + (ny * ny) + (nz * nz))
        if length <= 1e-8:
            return (0, 0, 127)

        nx /= length
        ny /= length
        nz /= length
        dx = max(-127, min(127, int(round(nx * 127.0))))
        dy = max(-127, min(127, int(round(ny * 127.0))))
        dz = max(-127, min(127, int(round(nz * 127.0))))
        return (dx, dy, dz)

    def _resolve_xbox_transform(self, raw_group: int, palette: List[int], segments: List[Segment], virt_segments: List[VirtSegment]) -> tuple[int, int, int, int, float]:
        total_transforms = len(segments) + len(virt_segments)
        transform_id = -1

        if len(palette) == 1:
            transform_id = int(palette[0])
        else:
            slot_candidates = [int(raw_group) // 3, int(raw_group)]
            for slot in slot_candidates:
                if 0 <= slot < len(palette):
                    transform_id = int(palette[slot])
                    break

        if transform_id < 0 and 0 <= int(raw_group) < total_transforms:
            transform_id = int(raw_group)

        return resolve_weighted_transform(transform_id, segments, virt_segments)

    def _read_vertex_data(self, target_context: SectionContext, vertex_data_abs: int) -> Tuple[List[MVertex], list[tuple[int, int, int, int]]]:
        if vertex_data_abs <= 0 or vertex_data_abs + 4 > target_context.file_size:
            return [], []

        br = target_context.reader
        previous = br.tell()
        try:
            br.seek(vertex_data_abs)
            vertex_count = br.u32()
            data_end = vertex_data_abs + 4 + (vertex_count * self.XBOX_VERTEX_SIZE)
            if data_end > target_context.file_size:
                logger.warning(
                    'Skipping Xbox vertex stream read in %s: abs=0x%X count=%d exceeds file size 0x%X',
                    target_context.file_name,
                    vertex_data_abs,
                    vertex_count,
                    target_context.file_size,
                )
                return [], []

            vertices: List[MVertex] = []
            colors: list[tuple[int, int, int, int]] = []
            for index in range(vertex_count):
                x = br.i16()
                y = br.i16()
                z = br.i16()
                color = (br.u8(), br.u8(), br.u8(), br.u8())
                u = br.i16()
                v = br.i16()
                raw_group = br.u16()
                packed_normal = br.u32()

                vertices.append(
                    MVertex(
                        index=index,
                        position_raw=(x, y, z),
                        normal_raw=self._decode_xbox_packed_normal(packed_normal),
                        segment=-1,
                        uv_raw=(u, v),
                        gc_transform_id=int(raw_group),
                    )
                )
                colors.append(color)

            logger.debug(
                'Read %d Xbox vertices from %s:0x%X',
                vertex_count,
                target_context.file_name,
                vertex_data_abs,
            )
            return vertices, colors
        finally:
            br.seek(previous)

    def _parse_xbox_primdata_entry(
        self,
        cache: SectionContextCache,
        source_context: SectionContext,
        prim_abs: int,
    ) -> dict | None:
        if prim_abs <= 0 or prim_abs + 0x34 > source_context.file_size:
            return None

        br = source_context.reader
        previous = br.tell()
        try:
            br.seek(prim_abs)
            valid_prim = br.u32()
            if valid_prim == 0:
                return None

            _unk0 = br.u32()
            _unk1 = br.u32()
            draw_group = br.i32()
            tpageid = br.i32()
            blend_mode_something = br.i32()
            _unk2 = br.u32()
            _unk3 = br.u32()
            offset_next_primdata = br.u32()
            _unk4 = br.u32()
            _unk5 = br.u32()
            face_data_field_local_offset = br.tell() - source_context.data_start
            face_data_offset = br.u32()
            face_ctx, face_abs = resolve_pointer(cache, source_context, face_data_field_local_offset, face_data_offset)
            bone_id_count = br.u16()
            bone_ids = [br.u16() for _ in range(bone_id_count)]

            nested_palettes: List[dict] = []
            while br.tell() + 2 <= source_context.file_size:
                entry_pos = br.tell()
                marker = br.peek_u16(offset=entry_pos)
                if marker == 0xFFFF:
                    br.u16()
                    break

                if (entry_pos & 0x3) != 0:
                    if br.tell() + 2 > source_context.file_size:
                        break
                    br.u16()

                if br.tell() + 14 > source_context.file_size:
                    break

                nested_unk0 = br.u32()
                nested_unk1 = br.u32()
                nested_face_field_local_offset = br.tell() - source_context.data_start
                nested_face_data_offset = br.u32()
                nested_face_ctx, nested_face_abs = resolve_pointer(cache, source_context, nested_face_field_local_offset, nested_face_data_offset)
                nested_bone_count = br.u16()
                if br.tell() + (nested_bone_count * 2) > source_context.file_size:
                    break
                nested_bone_ids = [br.u16() for _ in range(nested_bone_count)]

                nested_palettes.append({
                    'absolute_offset': entry_pos,
                    'unk0': nested_unk0,
                    'unk1': nested_unk1,
                    'face_data_offset': nested_face_data_offset,
                    'face_context': nested_face_ctx,
                    'face_absolute_offset': nested_face_abs,
                    'bone_ids': nested_bone_ids,
                })

            return {
                'record_absolute_offset': prim_abs,
                'draw_group': draw_group,
                'tpageid': tpageid,
                'env_mapping': blend_mode_something,
                'offset_next_primdata': offset_next_primdata,
                'face_data_offset': face_data_offset,
                'face_context': face_ctx,
                'face_absolute_offset': face_abs,
                'bone_ids': bone_ids,
                'nested_palettes': nested_palettes,
            }
        finally:
            br.seek(previous)

    @classmethod
    def _triangle_strip_to_triangles(cls, indices: list[int]) -> list[int]:
        triangles: list[int] = []
        for i in range(len(indices) - 2):
            a = int(indices[i])
            b = int(indices[i + 1])
            c = int(indices[i + 2])
            if a == b or b == c or a == c:
                continue
            if (i % 2) == 0:
                triangles.extend((a, b, c))
            else:
                triangles.extend((b, a, c))
        return triangles

    @staticmethod
    def _triangle_list_to_triangles(indices: list[int]) -> list[int]:
        triangles: list[int] = []
        usable_count = len(indices) - (len(indices) % 3)
        for i in range(0, usable_count, 3):
            a = int(indices[i])
            b = int(indices[i + 1])
            c = int(indices[i + 2])
            if len({a, b, c}) == 3:
                triangles.extend((a, b, c))
        return triangles

    @staticmethod
    def _quads_to_triangles(indices: list[int]) -> list[int]:
        triangles: list[int] = []
        usable_count = len(indices) - (len(indices) % 4)
        for i in range(0, usable_count, 4):
            a = int(indices[i])
            b = int(indices[i + 1])
            c = int(indices[i + 2])
            d = int(indices[i + 3])
            if len({a, b, c}) == 3:
                triangles.extend((a, b, c))
            if len({a, c, d}) == 3:
                triangles.extend((a, c, d))
        return triangles

    def _decode_xbox_face_record(self, format_code: int, raw_indices: list[int]) -> list[int]:
        if not raw_indices:
            return []
        if format_code == 0x05:
            return self._triangle_list_to_triangles(raw_indices)
        if format_code == 0x06:
            return self._triangle_strip_to_triangles(raw_indices)
        if format_code == 0x08:
            return self._quads_to_triangles(raw_indices)
        logger.debug('Skipping unsupported Xbox face format 0x%X', format_code)
        return []

    @staticmethod
    def _is_xbox_index_inline_command(command: int) -> bool:
        command = int(command) & 0xFFFFFFFF
        return (command & 0xFFFF) == 0x1800 and (command >> 16) >= 0x4000

    @staticmethod
    def _xbox_index_inline_byte_count(command: int) -> int:
        return max(0, ((int(command) >> 16) & 0xFFFF) - 0x4000)

    def _find_xbox_face_record_start(self, face_context: SectionContext, face_abs: int) -> int:
        if face_abs <= 0 or face_abs >= face_context.file_size:
            return 0

        br = face_context.reader
        for candidate in (face_abs, face_abs + 32):
            if candidate + 4 <= face_context.file_size:
                br.seek(candidate)
                if br.u32() == 0x000417FC:
                    return candidate

        search_end = min(face_context.file_size, face_abs + 96)
        if search_end <= face_abs:
            return 0
        br.seek(face_abs)
        window = br.read(search_end - face_abs)
        relative = window.find(self.XBOX_FACE_DATA_RECORD_MAGIC)
        return 0 if relative < 0 else face_abs + relative

    def _read_xbox_face_data_records(self, face_context: SectionContext, face_abs: int) -> list[list[int]]:
        if face_context is None or face_abs <= 0 or face_abs >= face_context.file_size:
            return []

        br = face_context.reader
        previous = br.tell()
        records: list[list[int]] = []
        try:
            current_abs = self._find_xbox_face_record_start(face_context, face_abs)
            if current_abs <= 0:
                logger.debug('No Xbox face record marker found in %s near 0x%X', face_context.file_name, face_abs)
                return []

            safety = 0
            while current_abs + 8 <= face_context.file_size and safety < 1024:
                safety += 1
                br.seek(current_abs)
                identifier = br.u32()
                if identifier != 0x000417FC:
                    logger.debug(
                        'Stopping Xbox face parse in %s at 0x%X: expected identifier 0x000417FC, got 0x%08X',
                        face_context.file_name,
                        current_abs,
                        identifier,
                    )
                    break

                format_code = br.u32()
                if format_code == 0:
                    next_abs = br.tell()
                    if next_abs + 8 <= face_context.file_size:
                        br.seek(next_abs)
                        if br.u32() == 0x000417FC:
                            current_abs = next_abs
                            continue
                    break

                raw_indices: list[int] = []
                command_count = 0
                next_record_abs = 0

                while br.tell() + 4 <= face_context.file_size:
                    command_abs = br.tell()
                    command = br.u32()
                    if command == 0x000417FC:
                        next_record_abs = command_abs
                        break

                    if not self._is_xbox_index_inline_command(command):
                        logger.debug(
                            'Stopping Xbox face index payload in %s at 0x%X: unexpected command 0x%08X',
                            face_context.file_name,
                            command_abs,
                            command,
                        )
                        break

                    command_count += 1
                    byte_count = self._xbox_index_inline_byte_count(command)
                    if byte_count <= 0:
                        continue
                    if br.tell() + byte_count > face_context.file_size:
                        logger.warning(
                            'Truncating Xbox face index payload in %s at 0x%X: byte_count=0x%X exceeds file size',
                            face_context.file_name,
                            command_abs,
                            byte_count,
                        )
                        byte_count = max(0, face_context.file_size - br.tell())
                    if byte_count & 1:
                        byte_count -= 1
                    raw_indices.extend(br.u16() for _ in range(byte_count // 2))

                triangles = self._decode_xbox_face_record(format_code, raw_indices)
                if triangles:
                    records.append(triangles)

                logger.debug(
                    'Read Xbox face record from %s:0x%X format=0x%X commands=%d (%d raw indices -> %d triangle indices)',
                    face_context.file_name,
                    current_abs,
                    format_code,
                    command_count,
                    len(raw_indices),
                    len(triangles),
                )

                if next_record_abs <= 0 or next_record_abs <= current_abs:
                    break
                current_abs = next_record_abs

            return records
        finally:
            br.seek(previous)

    def _parse_xbox_primdata_chain(
        self,
        cache: SectionContextCache,
        source_context: SectionContext,
        first_ptr_value: int,
        first_target_context: SectionContext,
        first_absolute_offset: int,
    ) -> List[TextureStrip]:
        if first_ptr_value == 0 or first_absolute_offset == 0:
            return []

        strips: List[TextureStrip] = []
        visited_offsets: set[tuple[str, int]] = set()
        current_context = first_target_context
        current_absolute_offset = first_absolute_offset
        primitive_entry_index = 0

        while current_absolute_offset:
            visit_key = (current_context.file_name, current_absolute_offset)
            if visit_key in visited_offsets:
                logger.debug('Stopping Xbox PrimData parse due to loop at %s:0x%X', current_context.file_name, current_absolute_offset)
                break
            visited_offsets.add(visit_key)

            prim = self._parse_xbox_primdata_entry(cache, current_context, current_absolute_offset)
            if prim is None:
                break

            primitive_material_index = int(primitive_entry_index)
            palette_sources = [{
                'offset': prim['record_absolute_offset'],
                'face_context': prim.get('face_context'),
                'face_absolute_offset': prim.get('face_absolute_offset', 0),
                'bone_ids': list(prim.get('bone_ids') or []),
            }]
            palette_sources.extend(prim.get('nested_palettes') or [])

            emitted_any = False
            for palette_index, palette_source in enumerate(palette_sources):
                face_records = self._read_xbox_face_data_records(
                    palette_source.get('face_context'),
                    int(palette_source.get('face_absolute_offset', 0) or 0),
                )
                if not face_records:
                    continue

                emitted_any = True
                palette_bone_ids = list(palette_source.get('bone_ids') or [])
                base_offset = int(palette_source.get('absolute_offset', prim['record_absolute_offset']) or prim['record_absolute_offset'])
                for record_index, indices in enumerate(face_records):
                    strips.append(
                        TextureStrip(
                            offset=base_offset + (palette_index * 0x100000) + (record_index * 0x1000),
                            vertex_count=len(indices),
                            draw_group=prim['draw_group'],
                            tpageid=prim['tpageid'],
                            sort_push=0.0,
                            scroll_offset=0.0,
                            env_mapping=prim['env_mapping'],
                            next_texture=prim['offset_next_primdata'],
                            indices=indices,
                            bone_ids=palette_bone_ids,
                            material_group=primitive_material_index,
                        )
                    )

            if not emitted_any:
                strips.append(
                    TextureStrip(
                        offset=prim['record_absolute_offset'],
                        vertex_count=0,
                        draw_group=prim['draw_group'],
                        tpageid=prim['tpageid'],
                        sort_push=0.0,
                        scroll_offset=0.0,
                        env_mapping=prim['env_mapping'],
                        next_texture=prim['offset_next_primdata'],
                        indices=[],
                        bone_ids=list(prim.get('bone_ids') or []),
                        material_group=primitive_material_index,
                    )
                )

            primitive_entry_index += 1

            next_prim_raw = prim['offset_next_primdata']
            if next_prim_raw == 0:
                break

            next_field_local_offset = (prim['record_absolute_offset'] - current_context.data_start) + 32
            current_context, current_absolute_offset = resolve_pointer(
                cache,
                current_context,
                next_field_local_offset,
                next_prim_raw,
            )

        return strips



    def _materialize_xbox_vertices_by_strip_usage(
        self,
        vertices: List[MVertex],
        vertex_colors: list[tuple[int, int, int, int]],
        strips: List[TextureStrip],
        segments: List[Segment],
        virt_segments: List[VirtSegment],
    ) -> tuple[List[MVertex], list[tuple[int, int, int, int]], List[TextureStrip], List[Segment], List[VirtSegment]]:
        if not vertices or not strips:
            return vertices, vertex_colors, strips, segments, virt_segments

        source_virt_segments = virt_segments
        output_vertices: List[MVertex] = []
        output_colors: list[tuple[int, int, int, int]] = []
        output_strips: List[TextureStrip] = []
        output_virt_segments: List[VirtSegment] = []
        vertex_map: dict[tuple[int, int, int, int, int, int], int] = {}

        def materialize_vertex(source_index: int, palette: List[int]) -> int | None:
            if not (0 <= int(source_index) < len(vertices)):
                return None

            source_vertex = vertices[int(source_index)]
            transform_id, bind_segment, primary_segment, secondary_segment, secondary_weight = self._resolve_xbox_transform(
                int(source_vertex.gc_transform_id),
                palette,
                segments,
                source_virt_segments,
            )

            weight_key = int(round(max(0.0, min(1.0, float(secondary_weight))) * 1000000.0))
            key = (
                int(source_index),
                int(transform_id),
                int(bind_segment),
                int(primary_segment),
                int(secondary_segment),
                weight_key,
            )

            existing = vertex_map.get(key)
            if existing is not None:
                return existing

            initial_segment = -1
            if 0 <= int(primary_segment) < len(segments):
                initial_segment = int(primary_segment)
            elif 0 <= int(bind_segment) < len(segments):
                initial_segment = int(bind_segment)

            vertex_index = len(output_vertices)
            output_vertices.append(MVertex(
                index=vertex_index,
                position_raw=source_vertex.position_raw,
                normal_raw=source_vertex.normal_raw,
                segment=initial_segment,
                uv_raw=source_vertex.uv_raw,
                gc_transform_id=int(transform_id),
                gc_bind_segment=int(bind_segment),
                gc_primary_segment=int(primary_segment),
                gc_secondary_segment=int(secondary_segment),
                gc_secondary_weight=float(secondary_weight),
            ))

            if 0 <= int(source_index) < len(vertex_colors):
                output_colors.append(vertex_colors[int(source_index)])
            else:
                output_colors.append((255, 255, 255, 255))

            if (
                0 <= int(primary_segment) < len(segments)
                and 0 <= int(secondary_segment) < len(segments)
                and 0.0 < float(secondary_weight) < 1.0
            ):
                output_virt_segments.append(make_weighted_virt_segment(
                    vertex_index,
                    primary_segment,
                    secondary_segment,
                    secondary_weight,
                    len(output_virt_segments),
                ))

            vertex_map[key] = vertex_index
            return vertex_index

        for strip in strips:
            palette = list(getattr(strip, 'bone_ids', []) or [])
            source_indices = [int(index) for index in getattr(strip, 'indices', [])]
            rebuilt_indices: list[int] = []
            skipped_triangles = 0

            usable_count = len(source_indices) - (len(source_indices) % 3)
            for triangle_start in range(0, usable_count, 3):
                triangle = source_indices[triangle_start:triangle_start + 3]
                materialized_triangle: list[int] = []
                for source_index in triangle:
                    materialized = materialize_vertex(int(source_index), palette)
                    if materialized is None:
                        materialized_triangle = []
                        break
                    materialized_triangle.append(int(materialized))

                if len(materialized_triangle) != 3:
                    skipped_triangles += 1
                    continue
                if len(set(materialized_triangle)) != 3:
                    skipped_triangles += 1
                    continue
                rebuilt_indices.extend(materialized_triangle)

            if skipped_triangles:
                logger.debug(
                    'Skipped %d Xbox triangles with invalid or degenerate materialized indices for strip at 0x%X',
                    skipped_triangles,
                    int(getattr(strip, 'offset', 0)),
                )

            output_strips.append(TextureStrip(
                offset=int(getattr(strip, 'offset', 0)),
                vertex_count=len(rebuilt_indices),
                draw_group=int(getattr(strip, 'draw_group', 0)),
                tpageid=int(getattr(strip, 'tpageid', 0)),
                sort_push=float(getattr(strip, 'sort_push', 0.0)),
                scroll_offset=float(getattr(strip, 'scroll_offset', 0.0)),
                env_mapping=int(getattr(strip, 'env_mapping', 0)),
                next_texture=int(getattr(strip, 'next_texture', -1)),
                indices=rebuilt_indices,
                bone_ids=list(getattr(strip, 'bone_ids', []) or []),
                material_group=int(getattr(strip, 'material_group', -1)),
            ))

        for segment in segments:
            assigned = [vertex.index for vertex in output_vertices if int(vertex.segment) == int(segment.index)]
            segment.first_vertex = min(assigned) if assigned else -1
            segment.last_vertex = max(assigned) if assigned else -1

        logger.debug(
            'Materialized Xbox vertices by palette usage: %d source vertices -> %d output vertices across %d strips',
            len(vertices),
            len(output_vertices),
            len(output_strips),
        )
        return output_vertices, output_colors, output_strips, segments, output_virt_segments

    def _assign_xbox_vertex_segments(
        self,
        vertices: List[MVertex],
        strips: List[TextureStrip],
        segments: List[Segment],
        virt_segments: List[VirtSegment],
    ) -> None:
        if not vertices or not strips or not segments:
            return

        segment_ref_counts: List[Counter[int]] = [Counter() for _ in range(len(vertices))]

        for strip in strips:
            palette = list(getattr(strip, 'bone_ids', []) or [])
            if not palette:
                continue
            for vertex_index in getattr(strip, 'indices', []):
                if not (0 <= int(vertex_index) < len(vertices)):
                    continue
                vertex = vertices[int(vertex_index)]
                transform_id, bind_segment, primary_segment, secondary_segment, secondary_weight = self._resolve_xbox_transform(
                    int(vertex.gc_transform_id),
                    palette,
                    segments,
                    virt_segments,
                )
                vertex.gc_bind_segment = int(bind_segment)
                vertex.gc_primary_segment = int(primary_segment)
                vertex.gc_secondary_segment = int(secondary_segment)
                vertex.gc_secondary_weight = float(secondary_weight)
                if vertex.gc_transform_id < 0 and transform_id >= 0:
                    vertex.gc_transform_id = int(transform_id)
                if 0 <= int(bind_segment) < len(segments):
                    segment_ref_counts[int(vertex_index)][int(bind_segment)] += 1

        for vertex_index, vertex in enumerate(vertices):
            chosen_segment = int(vertex.gc_bind_segment)
            counter = segment_ref_counts[vertex_index]
            if counter:
                chosen_segment = max(counter.items(), key=lambda item: (int(item[1]), -int(item[0])))[0]

            if 0 <= int(chosen_segment) < len(segments):
                vertex.segment = int(chosen_segment)
            elif 0 <= int(vertex.gc_primary_segment) < len(segments):
                vertex.segment = int(vertex.gc_primary_segment)
            else:
                vertex.segment = -1


    def _compact_vertices_to_strip_usage(
        self,
        vertices: List[MVertex],
        vertex_colors: list[tuple[int, int, int, int]],
        strips: List[TextureStrip],
        segments: List[Segment],
        virt_segments: List[VirtSegment],
    ) -> tuple[List[MVertex], list[tuple[int, int, int, int]], List[TextureStrip], List[Segment], List[VirtSegment]]:
        valid_vertex_count = len(vertices)
        used_vertex_indices = sorted({
            int(index)
            for strip in strips
            for index in getattr(strip, 'indices', [])
            if 0 <= int(index) < valid_vertex_count
        })
        if not vertices or not used_vertex_indices:
            return vertices, vertex_colors, strips, segments, virt_segments
        if len(used_vertex_indices) == len(vertices) and used_vertex_indices[0] == 0 and used_vertex_indices[-1] == len(vertices) - 1:
            return vertices, vertex_colors, strips, segments, virt_segments

        for strip in strips:
            strip.indices = [int(index) for index in getattr(strip, 'indices', []) if 0 <= int(index) < valid_vertex_count]

        vertex_remap = {source_index: new_index for new_index, source_index in enumerate(used_vertex_indices)}

        compact_vertices: List[MVertex] = []
        for source_index in used_vertex_indices:
            vertex = vertices[source_index]
            compact_vertices.append(
                MVertex(
                    index=vertex_remap[source_index],
                    position_raw=vertex.position_raw,
                    normal_raw=vertex.normal_raw,
                    segment=vertex.segment,
                    uv_raw=vertex.uv_raw,
                    gc_transform_id=vertex.gc_transform_id,
                    gc_bind_segment=vertex.gc_bind_segment,
                    gc_primary_segment=vertex.gc_primary_segment,
                    gc_secondary_segment=vertex.gc_secondary_segment,
                    gc_secondary_weight=vertex.gc_secondary_weight,
                )
            )

        compact_colors = [vertex_colors[source_index] for source_index in used_vertex_indices if source_index < len(vertex_colors)] if vertex_colors else vertex_colors

        compact_strips: List[TextureStrip] = []
        for strip in strips:
            strip.indices = [vertex_remap[int(index)] for index in strip.indices if int(index) in vertex_remap]
            compact_strips.append(strip)

        for segment in segments:
            remapped = [vertex_remap[source_index] for source_index in used_vertex_indices if segment.first_vertex <= source_index <= segment.last_vertex]
            segment.first_vertex = min(remapped) if remapped else -1
            segment.last_vertex = max(remapped) if remapped else -1

        compact_virt_segments: List[VirtSegment] = []
        for virt_segment in virt_segments:
            remapped = [vertex_remap[source_index] for source_index in used_vertex_indices if virt_segment.first_vertex <= source_index <= virt_segment.last_vertex]
            if not remapped:
                continue
            virt_segment.first_vertex = min(remapped)
            virt_segment.last_vertex = max(remapped)
            compact_virt_segments.append(virt_segment)

        logger.debug(
            'Compacted Xbox vertex set from %d to %d referenced vertices',
            len(vertices),
            len(compact_vertices),
        )
        return compact_vertices, compact_colors, compact_strips, segments, compact_virt_segments

    def _parse_model(self, cache: SectionContextCache, context: SectionContext) -> ModelData:
        br = context.reader
        br.seek(context.data_start)

        version = br.i32()
        num_segments = br.i32()
        num_virt_segments = br.i32()
        segment_list_raw, segment_list_ctx, segment_list_abs = self._read_u32_pointer(cache, context)

        logger.debug(
            'Starting Xbox model parse from %s: version=%d num_segments=%d num_virt_segments=%d segment_list=0x%X',
            context.file_name,
            version,
            num_segments,
            num_virt_segments,
            segment_list_raw,
        )

        header_resume_offset = br.tell()
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
                    hinfo_ctx, hinfo_abs = resolve_pointer(cache, segment_list_ctx, hinfo_field_local_offset, segment.hinfo)
                    if hinfo_abs:
                        hspheres.extend(self._parse_hinfo_hspheres(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_sphere_index_start=len(hspheres)))
                        hboxes.extend(self._parse_hinfo_hboxes(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_hbox_index_start=len(hboxes)))
                        hmarkers.extend(self._parse_hinfo_hmarkers(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_marker_index_start=len(hmarkers)))
                        hcapsules.extend(self._parse_hinfo_hcapsules(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_capsule_index_start=len(hcapsules)))

            for virt_index in range(num_virt_segments):
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
                uv_format='xbox',
            )

        br.seek(header_resume_offset)
        model_scale = br.vec4()

        _num_vertices_hint = br.i32()
        _vertex_list_raw, _vertex_list_ctx, _vertex_list_abs = self._read_u32_pointer(cache, context)
        _num_normals = br.i32()
        _normal_list_raw, _normal_list_ctx, _normal_list_abs = self._read_u32_pointer(cache, context)
        _num_faces_hint = br.i32()
        _face_list_raw, _face_list_ctx, _face_list_abs = self._read_u32_pointer(cache, context)

        _obsolete_ani_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        max_rad = br.f32()
        max_rad_sq = br.f32()
        _obsolete_start_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _obsolete_end_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _animated_list_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _animated_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _scroll_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        primitive_data_raw, primitive_ctx, primitive_abs = self._read_u32_pointer(cache, context)

        _a_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _b_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        vertex_data_raw, vertex_data_ctx, vertex_data_abs = self._read_u32_pointer(cache, context)
        _unknown_after_vertex_data = br.u32() if br.tell() + 4 <= context.data_end else 0

        bone_mirror_data_raw = 0
        bone_mirror_data_ctx = context
        bone_mirror_data_abs = 0
        if br.tell() + 4 <= context.data_end:
            bone_mirror_field_local_offset = self._field_local_offset(context)
            bone_mirror_data_raw = br.u32()
            if bone_mirror_data_raw:
                bone_mirror_data_ctx, bone_mirror_data_abs = resolve_pointer(cache, context, bone_mirror_field_local_offset, bone_mirror_data_raw)

        _normal_data_offset = br.i32() if br.tell() + 4 <= context.data_end else 0

        env_mapped_vertices_raw, env_mapped_vertices_ctx, env_mapped_vertices_abs = 0, context, 0
        eye_ref_env_mapped_vertices_raw, eye_ref_env_mapped_vertices_ctx, eye_ref_env_mapped_vertices_abs = 0, context, 0
        _drawgroup_center_list_raw, _ctx, _abs = 0, context, 0

        num_markups = br.u32() if br.tell() + 4 <= context.data_end else 0
        mark_up_list_raw, mark_up_list_ctx, mark_up_list_abs = self._read_u32_pointer(cache, context) if br.tell() + 4 <= context.data_end else (0, context, 0)

        if not bone_mirror_data_abs and br.tell() + 4 <= context.data_end:
            fallback_offset = br.tell()
            bone_mirror_field_local_offset = self._field_local_offset(context)
            bone_mirror_signed = br.i32()
            if bone_mirror_signed > 0:
                bone_mirror_data_raw = int(bone_mirror_signed) & 0xFFFFFFFF
                bone_mirror_data_ctx, bone_mirror_data_abs = resolve_pointer(cache, context, bone_mirror_field_local_offset, bone_mirror_data_raw)
            if not bone_mirror_data_abs:
                br.seek(fallback_offset)

        num_targets = br.i32() if br.tell() + 4 <= context.data_end else 0
        target_list_raw, target_list_ctx, target_list_abs = self._read_u32_pointer(cache, context) if br.tell() + 4 <= context.data_end else (0, context, 0)
        cdc_render_data_id = 0
        _xbox_unknown_after_target_list = br.u32() if br.tell() + 4 <= context.data_end else 0
        _cdc_render_model_data_raw = br.u32() if br.tell() + 4 <= context.data_end else 0
        if br.tell() + 12 <= context.data_end:
            br.skip(12)

        logger.debug(
            'Xbox model geometry pointers in %s: primitive_data=0x%X vertex_data=0x%X bone_mirror=0x%X num_markups=%d num_targets=%d target_list=0x%X cdcRenderDataID=0x%X',
            context.file_name,
            primitive_data_raw,
            vertex_data_raw,
            bone_mirror_data_raw,
            num_markups,
            num_targets,
            target_list_raw,
            cdc_render_data_id,
        )

        vertices, vertex_colors = self._read_vertex_data(vertex_data_ctx, vertex_data_abs)
        strips = self._parse_xbox_primdata_chain(cache, context, primitive_data_raw, primitive_ctx, primitive_abs)
        vertices, vertex_colors, strips, segments, virt_segments = self._materialize_xbox_vertices_by_strip_usage(
            vertices,
            vertex_colors,
            strips,
            segments,
            virt_segments,
        )
        vertices, vertex_colors, strips, segments, virt_segments = self._compact_vertices_to_strip_usage(
            vertices,
            vertex_colors,
            strips,
            segments,
            virt_segments,
        )

        env_mapped_face_indices: list[int] = []
        if env_mapped_vertices_abs:
            env_mapped_face_indices.extend(self._parse_index_list(env_mapped_vertices_ctx, env_mapped_vertices_abs, env_mapped_vertices_raw, 'envMappedVertices'))

        eye_ref_env_mapped_face_indices: list[int] = []
        if eye_ref_env_mapped_vertices_abs:
            eye_ref_env_mapped_face_indices.extend(self._parse_index_list(eye_ref_env_mapped_vertices_ctx, eye_ref_env_mapped_vertices_abs, eye_ref_env_mapped_vertices_raw, 'eyeRefEnvMappedVertices'))

        markups = []
        if num_markups > 0 and mark_up_list_abs:
            markups = self._parse_section_markups(cache, mark_up_list_ctx, mark_up_list_abs, num_markups, cdc_render_data_id)

        if num_targets > 0 and target_list_abs:
            targets.extend(self._parse_targets(target_list_ctx, target_list_abs, num_targets))

        bone_mirror_entries = []
        if bone_mirror_data_abs:
            bone_mirror_entries = self._parse_bone_mirror_entries(bone_mirror_data_ctx, bone_mirror_data_abs)

        return ModelData(
            version=version,
            model_scale=model_scale,
            segments=segments,
            virt_segments=virt_segments,
            vertices=vertices,
            faces=[],
            strips=strips,
            vertex_colors=vertex_colors,
            env_mapped_face_indices=sorted({int(index) for index in env_mapped_face_indices if int(index) >= 0}),
            eye_ref_env_mapped_face_indices=sorted({int(index) for index in eye_ref_env_mapped_face_indices if int(index) >= 0}),
            hmarkers=hmarkers,
            hspheres=hspheres,
            hboxes=hboxes,
            hcapsules=hcapsules,
            targets=targets,
            markups=markups,
            max_rad=max_rad,
            max_rad_sq=max_rad_sq,
            bone_mirror_entries=bone_mirror_entries,
            cdc_render_data_id=cdc_render_data_id,
            uv_format='xbox',
        )
