from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Tuple

from ...core.log import logger
from ...core.model_types import BoneMirrorEntry, HBox, HCapsule, HMarker, HSphere, MFace, MVertex, ModelData, Segment, Target, TextureStrip, VirtSegment
from ..pc.tr7ae_model import TRModelParser
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..common.model import make_segments_only_model_data, make_weighted_virt_segment, resolve_weighted_transform


@dataclass(slots=True)
class PS2PrimitiveInfo:
    index: int
    record_absolute_offset: int
    vertex_count: int
    qword_count: int
    palette_count: int
    flags: int
    tpageid: int
    primitive_data_offset: int
    primitive_context: SectionContext
    primitive_absolute_offset: int
    sort_key: float
    palette: List[int]


class TRPS2ModelParser(TRModelParser):
    """PS2 little-endian model parser.

    Observed PS2 geometry is stored in VIF UNPACK packets referenced from a
    32-byte PrimitiveInfo table.  This is intentionally import-only and keeps
    the assumptions narrow: one material/primitive entry becomes one triangle
    list strip, and vertices are materialized from the per-primitive stream.
    """

    PS2_UV_SCALE = 4096.0
    PS2_MATRIX_SLOT_STRIDE = 12
    PS2_BONE_MIRROR_FIELD_OFFSET = 0x6C

    def __init__(self, filepath: str, import_hinfo: bool = True, import_markups: bool = True, parse_segments_only: bool = False):
        super().__init__(filepath, import_hinfo=import_hinfo, import_markups=import_markups, endian="<", parse_segments_only=parse_segments_only)

    def _is_gamecube_layout(self) -> bool:
        return True


    @staticmethod
    def _canonicalize_ps2_tpageid(raw_tpageid: int) -> int:
        raw = int(raw_tpageid) & 0xFFFFFFFF

        texture_id = raw & 0xFFFF
        blend_value = (raw >> 16) & 0xF

        canonical = texture_id & 0x1FFF
        canonical |= (blend_value & 0xF) << 13

        render_mode = (raw >> 20) & 0x7
        double_sided = (raw >> 31) & 0x1
        canonical |= (render_mode & 0x7) << 17   # PS2 render mode
        if not double_sided:
            canonical |= 1 << 21                 # shared UI/backface-cull hint

        canonical |= ((raw >> 23) & 0x1) << 20   # PS2 state bit 23
        canonical |= ((raw >> 24) & 0x1) << 24   # PS2 state bit 24
        canonical |= ((raw >> 25) & 0x3) << 22   # texture address mode
        canonical |= ((raw >> 27) & 0x1) << 25   # mip/LOD-ish flag
        canonical |= ((raw >> 28) & 0x1) << 30   # alpha/texture-state flag
        canonical |= ((raw >> 29) & 0x1) << 26   # flat_shading-ish
        canonical |= ((raw >> 30) & 0x1) << 27   # sort_z/render-pass-ish
        canonical |= double_sided << 28          # PS2 double-sided flag, low bit only

        return canonical & 0xFFFFFFFF

    @staticmethod
    def _normal_s12_to_byte(normal: Tuple[int, int, int]) -> tuple[int, int, int]:
        nx, ny, nz = (float(normal[0]), float(normal[1]), float(normal[2]))
        length = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
        if length <= 1e-8:
            return (0, 0, 127)
        return (
            max(-127, min(127, int(round((nx / length) * 127.0)))),
            max(-127, min(127, int(round((ny / length) * 127.0)))),
            max(-127, min(127, int(round((nz / length) * 127.0)))),
        )

    @classmethod
    def _matrix_slot_from_flag(cls, flag: int) -> int:
        matrix_word = int(flag) & 0x3FFF
        if matrix_word % cls.PS2_MATRIX_SLOT_STRIDE == 0:
            return matrix_word // cls.PS2_MATRIX_SLOT_STRIDE
        return matrix_word

    @classmethod
    def _strip_flag_adc(cls, flag: int) -> bool:
        # The top bit behaves like the PS2 ADC/no-draw marker used by strip starts.
        return (int(flag) & 0x8000) != 0

    @classmethod
    def _strip_flag_flip(cls, flag: int) -> bool:
        # The 0x4000 bit alternates on drawn vertices and is a better winding
        # indicator than global vertex parity once the packet restarts strips.
        return (int(flag) & 0x4000) != 0

    def _resolve_ps2_transform(self, flag: int, palette: List[int], segments: List[Segment], virt_segments: List[VirtSegment]) -> tuple[int, int, int, int, float]:
        slot = self._matrix_slot_from_flag(flag)
        total_transforms = len(segments) + len(virt_segments)
        transform_id = -1

        if 0 <= slot < len(palette):
            transform_id = int(palette[slot])
        elif 0 <= slot < total_transforms:
            transform_id = slot

        return resolve_weighted_transform(transform_id, segments, virt_segments)

    def _parse_ps2_primitive_info_table(
        self,
        cache: SectionContextCache,
        table_context: SectionContext,
        table_absolute_offset: int,
    ) -> List[PS2PrimitiveInfo]:
        if table_absolute_offset <= 0 or table_absolute_offset >= table_context.file_size:
            return []

        br = table_context.reader
        previous = br.tell()
        entries: List[PS2PrimitiveInfo] = []
        try:
            current = table_absolute_offset
            index = 0
            while current + 4 <= table_context.file_size:
                br.seek(current)
                first_word = br.u32()
                if first_word == 0:
                    break
                if current + 32 > table_context.file_size:
                    break

                record = br.peek(32, offset=current)
                vertex_count = int(record[0])
                qword_count = int(record[1])
                palette_count = int(record[2])
                flags = int(record[3])
                tpageid = struct.unpack_from('<I', record, 4)[0]
                primitive_data_offset = struct.unpack_from('<I', record, 8)[0]
                sort_key = struct.unpack_from('<f', record, 12)[0]
                palette = [int(value) for value in record[0x12:0x12 + min(palette_count, 14)]]

                primitive_field_local_offset = (current - table_context.data_start) + 8
                primitive_context, primitive_absolute_offset = resolve_pointer(
                    cache,
                    table_context,
                    primitive_field_local_offset,
                    primitive_data_offset,
                )

                entries.append(PS2PrimitiveInfo(
                    index=index,
                    record_absolute_offset=current,
                    vertex_count=vertex_count,
                    qword_count=qword_count,
                    palette_count=palette_count,
                    flags=flags,
                    tpageid=tpageid,
                    primitive_data_offset=primitive_data_offset,
                    primitive_context=primitive_context,
                    primitive_absolute_offset=primitive_absolute_offset,
                    sort_key=sort_key,
                    palette=palette,
                ))
                index += 1
                current += 32
        finally:
            br.seek(previous)

        logger.debug('Read %d PS2 PrimitiveInfo records from %s:0x%X', len(entries), table_context.file_name, table_absolute_offset)
        return entries

    @staticmethod
    def _vif_unpack_element_size(cmd: int) -> tuple[int, int] | None:
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

    def _parse_ps2_vif_chunk(self, context: SectionContext, primitive: PS2PrimitiveInfo) -> dict | None:
        start_abs = int(primitive.primitive_absolute_offset)
        byte_count = int(primitive.qword_count) * 16
        if start_abs <= 0 or byte_count <= 0 or start_abs + byte_count > context.file_size:
            logger.debug('Skipping PS2 primitive %d: invalid chunk bounds %s:0x%X qwc=%d', primitive.index, context.file_name, start_abs, primitive.qword_count)
            return None

        br = context.reader
        previous = br.tell()
        streams = {
            'sort_key': float(primitive.sort_key),
            'positions': [],
            'colors': [],
            'normals': [],
            'uvs': [],
        }
        try:
            pos = start_abs
            end = start_abs + byte_count
            safety = 0
            while pos + 4 <= end and safety < 16:
                safety += 1
                br.seek(pos)
                word = br.u32()
                if word == 0:
                    break
                cmd = (word >> 24) & 0xFF
                num = (word >> 16) & 0xFF
                imm = word & 0xFFFF
                unpack = self._vif_unpack_element_size(cmd)
                if unpack is None:
                    logger.debug('Stopping PS2 VIF parse in %s at 0x%X: unsupported VIF cmd=0x%X word=0x%08X', context.file_name, pos, cmd, word)
                    break
                components, scalar_size = unpack
                data_len = int(num) * int(components) * int(scalar_size)
                data_len_padded = (data_len + 3) & ~3
                payload_abs = pos + 4
                if payload_abs + data_len > end or payload_abs + data_len > context.file_size:
                    logger.debug('Stopping PS2 VIF parse in %s at 0x%X: payload exceeds primitive bounds', context.file_name, pos)
                    break
                payload = br.peek(data_len, offset=payload_abs)

                if cmd == 0x64 and num == 1 and data_len >= 8:
                    streams['sort_key'] = float(struct.unpack_from('<f', payload, 0)[0])
                elif cmd == 0x6D:
                    # V4_16: x, y, z, packed matrix/strip flag.
                    streams['positions'] = [struct.unpack_from('<4h', payload, i * 8) for i in range(num)]
                elif cmd == 0x6E:
                    # V4_8: vertex color, observed as RGBA-like bytes.
                    streams['colors'] = [tuple(payload[i * 4:i * 4 + 4]) for i in range(num)]
                elif cmd == 0x69:
                    # V3_16: signed normals, approximately S12 normalized.
                    streams['normals'] = [struct.unpack_from('<3h', payload, i * 6) for i in range(num)]
                elif cmd == 0x65:
                    # V2_16: texture coordinates, scale 4096 like GC/Xbox.
                    streams['uvs'] = [struct.unpack_from('<2h', payload, i * 4) for i in range(num)]

                pos = payload_abs + data_len_padded
            return streams
        finally:
            br.seek(previous)

    def _parse_ps2_geometry(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        primitive_info_raw: int,
        primitive_info_ctx: SectionContext,
        primitive_info_abs: int,
        segments: List[Segment],
        virt_segments: List[VirtSegment],
    ) -> tuple[List[MVertex], List[MFace], List[TextureStrip], list[tuple[int, int, int, int]], List[VirtSegment]]:
        primitive_infos = self._parse_ps2_primitive_info_table(cache, primitive_info_ctx, primitive_info_abs)
        if not primitive_infos:
            return [], [], [], [], []

        source_virt_segments = virt_segments
        output_virt_segments: List[VirtSegment] = []
        vertices: List[MVertex] = []
        vertex_colors: list[tuple[int, int, int, int]] = []
        faces: List[MFace] = []
        strips: List[TextureStrip] = []
        vertex_map: dict[tuple, int] = {}
        material_group_by_key: dict[tuple[int, int], int] = {}

        def material_group_for(draw_group: int, canonical_tpageid: int) -> int:
            key = (int(draw_group), int(canonical_tpageid) & 0xFFFFFFFF)
            material_group = material_group_by_key.get(key)
            if material_group is None:
                material_group = len(material_group_by_key)
                material_group_by_key[key] = material_group
            return material_group

        def materialize_vertex(primitive: PS2PrimitiveInfo, streams: dict, stream_index: int) -> int | None:
            positions = streams.get('positions') or []
            if not (0 <= stream_index < len(positions)):
                return None

            x, y, z, flag_signed = positions[stream_index]
            flag = int(flag_signed) & 0xFFFF
            palette = list(primitive.palette or [])
            transform_id, bind_segment, primary_segment, secondary_segment, secondary_weight = self._resolve_ps2_transform(
                flag,
                palette,
                segments,
                source_virt_segments,
            )

            normals = streams.get('normals') or []
            normal_raw = self._normal_s12_to_byte(normals[stream_index]) if stream_index < len(normals) else (0, 0, 127)
            uvs = streams.get('uvs') or []
            uv_raw = tuple(int(v) for v in uvs[stream_index]) if stream_index < len(uvs) else (0, 0)
            colors = streams.get('colors') or []
            color = tuple(int(v) for v in colors[stream_index]) if stream_index < len(colors) else (255, 255, 255, 255)
            weight_key = int(round(max(0.0, min(1.0, float(secondary_weight))) * 1000000.0))
            key = (
                int(x), int(y), int(z),
                int(normal_raw[0]), int(normal_raw[1]), int(normal_raw[2]),
                int(uv_raw[0]), int(uv_raw[1]),
                int(transform_id), int(bind_segment), int(primary_segment), int(secondary_segment), weight_key,
                int(color[0]), int(color[1]), int(color[2]), int(color[3]),
            )
            existing = vertex_map.get(key)
            if existing is not None:
                return existing

            initial_segment = -1
            if 0 <= int(primary_segment) < len(segments):
                initial_segment = int(primary_segment)
            elif 0 <= int(bind_segment) < len(segments):
                initial_segment = int(bind_segment)

            vertex_index = len(vertices)
            vertices.append(MVertex(
                index=vertex_index,
                position_raw=(int(x), int(y), int(z)),
                normal_raw=normal_raw,
                segment=initial_segment,
                uv_raw=(int(uv_raw[0]), int(uv_raw[1])),
                gc_transform_id=int(transform_id),
                gc_bind_segment=int(bind_segment),
                gc_primary_segment=int(primary_segment),
                gc_secondary_segment=int(secondary_segment),
                gc_secondary_weight=float(secondary_weight),
            ))
            vertex_colors.append(color)
            vertex_map[key] = vertex_index

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
            return vertex_index

        for primitive in primitive_infos:
            streams = self._parse_ps2_vif_chunk(primitive.primitive_context, primitive)
            if streams is None:
                continue
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
                if len(window) < 3 or self._strip_flag_adc(flag):
                    continue

                a_i, b_i, c_i = window
                a = materialize_vertex(primitive, streams, a_i)
                b = materialize_vertex(primitive, streams, b_i)
                c = materialize_vertex(primitive, streams, c_i)
                if a is None or b is None or c is None:
                    continue
                if len({a, b, c}) != 3:
                    continue

                if self._strip_flag_flip(flag):
                    tri = [b, a, c]
                else:
                    tri = [a, b, c]
                faces.append(MFace(index=len(faces), v0=tri[0], v1=tri[1], v2=tri[2], same_vert_bits=0))
                strip_indices.extend(tri)

            draw_group = int(primitive.flags)
            canonical_tpageid = self._canonicalize_ps2_tpageid(primitive.tpageid)
            strips.append(TextureStrip(
                offset=int(primitive.primitive_data_offset),
                vertex_count=len(strip_indices),
                draw_group=draw_group,
                tpageid=canonical_tpageid,
                sort_push=float(streams.get('sort_key', primitive.sort_key)),
                scroll_offset=0.0,
                env_mapping=0,
                next_texture=-1,
                indices=strip_indices,
                bone_ids=list(primitive.palette or []),
                material_group=material_group_for(draw_group, canonical_tpageid),
                ps2_tpageid_raw=int(primitive.tpageid),
            ))

        # Keep segment ranges useful for armature binding.
        for segment in segments:
            assigned = [vertex.index for vertex in vertices if int(vertex.segment) == int(segment.index)]
            segment.first_vertex = min(assigned) if assigned else -1
            segment.last_vertex = max(assigned) if assigned else -1

        logger.debug(
            'PS2 geometry from %s primitiveInfo=0x%X: primitiveInfos=%d vertices=%d faces=%d strips=%d',
            context.file_name,
            primitive_info_raw,
            len(primitive_infos),
            len(vertices),
            len(faces),
            len(strips),
        )
        return vertices, faces, strips, vertex_colors, output_virt_segments

    def _parse_ps2_bone_mirrors(
        self,
        cache: SectionContextCache,
        context: SectionContext,
    ) -> List[BoneMirrorEntry]:
        source_previous = context.reader.tell()
        mirror_previous = None
        mirror_ctx = context
        try:
            try:
                raw, mirror_ctx, mirror_abs = self._read_u32_pointer_at(cache, context, self.PS2_BONE_MIRROR_FIELD_OFFSET)
            except Exception as exc:
                logger.debug('Could not read PS2 bone mirror pointer from %s: %s', context.file_name, exc)
                return []
            if not raw or not mirror_abs:
                return []

            # The pointer must resolve inside a known data section.  Invalid
            # values can exist in non-standard/template headers, so fail closed.
            if mirror_abs < mirror_ctx.data_start or mirror_abs >= min(mirror_ctx.data_end, mirror_ctx.file_size):
                logger.debug(
                    'Ignoring PS2 bone mirror pointer outside section data: %s raw=0x%X abs=0x%X',
                    mirror_ctx.file_name, int(raw), int(mirror_abs)
                )
                return []

            mirror_previous = mirror_ctx.reader.tell()
            try:
                entries = self._parse_bone_mirror_entries(mirror_ctx, mirror_abs)
            except Exception as exc:
                logger.debug('Could not parse PS2 bone mirror table from %s:0x%X: %s', mirror_ctx.file_name, int(mirror_abs), exc)
                return []

            # Avoid treating arbitrary pointed data as a huge mirror table.
            # Known PS2 mirror tables are small; 64 entries is intentionally
            # generous and keeps malformed pointers from poisoning import.
            if len(entries) > 64:
                logger.debug(
                    'Ignoring suspicious PS2 bone mirror table from %s:0x%X with %d entries',
                    mirror_ctx.file_name, int(mirror_abs), len(entries)
                )
                return []

            logger.debug('Read %d PS2 bone mirror entries from %s:0x%X', len(entries), mirror_ctx.file_name, int(mirror_abs))
            return entries
        finally:
            if mirror_previous is not None:
                try:
                    mirror_ctx.reader.seek(mirror_previous)
                except Exception:
                    pass
            context.reader.seek(source_previous)

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
        bone_mirror_entries: List[BoneMirrorEntry] = self._parse_ps2_bone_mirrors(cache, context)

        header_resume_offset = br.tell()
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
                uv_format='ps2',
            )

        br.seek(header_resume_offset)
        model_scale = br.vec4()

        _num_vertices_hint = br.i32()
        _vertex_list_raw, _vertex_list_ctx, _vertex_list_abs = self._read_u32_pointer(cache, context)
        _num_normals = br.i32()
        _normal_list_raw, _normal_ctx, _normal_abs = self._read_u32_pointer(cache, context)
        _num_faces_hint = br.i32()
        _face_list_raw, _face_ctx, _face_abs = self._read_u32_pointer(cache, context)

        _obsolete_ani_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        max_rad = br.f32()
        max_rad_sq = br.f32()
        _obsolete_start_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _obsolete_end_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _animated_list_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _animated_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        _scroll_info_raw, _scroll_info_ctx, _scroll_info_abs = self._read_u32_pointer(cache, context)
        primitive_info_raw, primitive_info_ctx, primitive_info_abs = self._read_u32_pointer(cache, context)

        vertices, faces, strips, vertex_colors, output_virt_segments = self._parse_ps2_geometry(
            cache,
            context,
            primitive_info_raw,
            primitive_info_ctx,
            primitive_info_abs,
            segments,
            virt_segments,
        )
        if output_virt_segments and not virt_segments:
            virt_segments = output_virt_segments

        return ModelData(
            version=version,
            model_scale=model_scale,
            segments=segments,
            virt_segments=virt_segments,
            vertices=vertices,
            faces=faces,
            strips=strips,
            vertex_colors=vertex_colors,
            hmarkers=hmarkers,
            hspheres=hspheres,
            hboxes=hboxes,
            hcapsules=hcapsules,
            targets=targets,
            markups=[],
            max_rad=max_rad,
            max_rad_sq=max_rad_sq,
            bone_mirror_entries=bone_mirror_entries,
            cdc_render_data_id=0,
            uv_format='ps2',
        )
