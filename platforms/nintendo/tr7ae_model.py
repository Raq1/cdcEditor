from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from ...core.log import logger
from ...core.model_types import MFace, MVertex, ModelData, Segment, TextureStrip, VirtSegment
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..common.model import make_weighted_virt_segment, resolve_weighted_transform


class GameCubeModelParserMixin:

    def _read_gamecube_positions(self, ctx: SectionContext, start_abs: int, end_abs: int) -> List[Tuple[int, int, int]]:
        positions: List[Tuple[int, int, int]] = []
        if start_abs <= 0 or end_abs <= start_abs:
            return positions
        previous = ctx.reader.tell()
        try:
            ctx.reader.seek(start_abs)
            while ctx.reader.tell() + 6 <= end_abs:
                positions.append((ctx.reader.i16(), ctx.reader.i16(), ctx.reader.i16()))
        finally:
            ctx.reader.seek(previous)
        return positions

    def _read_gamecube_uvs(self, ctx: SectionContext, start_abs: int, end_abs: int) -> List[Tuple[int, int]]:
        uvs: List[Tuple[int, int]] = []
        if start_abs <= 0 or end_abs <= start_abs:
            return uvs
        previous = ctx.reader.tell()
        try:
            ctx.reader.seek(start_abs)
            while ctx.reader.tell() + 4 <= end_abs:
                uvs.append((ctx.reader.i16(), ctx.reader.i16()))
        finally:
            ctx.reader.seek(previous)
        return uvs

    def _read_gamecube_colors(self, ctx: SectionContext, start_abs: int, end_abs: int) -> List[Tuple[int, int, int, int]]:
        colors: List[Tuple[int, int, int, int]] = []
        if start_abs <= 0 or end_abs <= start_abs:
            return colors
        previous = ctx.reader.tell()
        try:
            ctx.reader.seek(start_abs)
            while ctx.reader.tell() + 4 <= end_abs:
                r = ctx.reader.u8()
                g = ctx.reader.u8()
                b = ctx.reader.u8()
                a = ctx.reader.u8()
                colors.append((r, g, b, a))
        finally:
            ctx.reader.seek(previous)
        return colors

    def _read_gamecube_normals(self, ctx: SectionContext, start_abs: int, end_abs: int) -> List[Tuple[int, int, int]]:
        normals: List[Tuple[int, int, int]] = []
        if start_abs <= 0 or end_abs <= start_abs:
            return normals
        previous = ctx.reader.tell()
        try:
            ctx.reader.seek(start_abs)
            while ctx.reader.tell() + 3 <= end_abs:
                normals.append((ctx.reader.i8(), ctx.reader.i8(), ctx.reader.i8()))
        finally:
            ctx.reader.seek(previous)
        return normals

    def _gamecube_is_valid_primitive_command(self, ctx: SectionContext, absolute_offset: int, region_end: int) -> bool:
        if absolute_offset + 3 > region_end or absolute_offset + 3 > ctx.file_size:
            return False

        opcode = ctx.reader.peek_u8(offset=absolute_offset)
        if opcode not in (0x81, 0x91, 0x99):
            return False
        count = ctx.reader.peek_u16(offset=absolute_offset + 1)

        if opcode == 0x81 and (count <= 0 or (count % 4) != 0):
            return False
        if opcode == 0x91 and (count <= 0 or (count % 3) != 0):
            return False
        if opcode == 0x99 and count < 3:
            return False

        command_size = 3 + (count * 9)
        return absolute_offset + command_size <= region_end

    def _parse_gamecube_primitive_commands(self, ctx: SectionContext, start_abs: int, end_abs: int) -> List[dict]:
        commands: List[dict] = []
        if start_abs <= 0 or end_abs <= start_abs:
            return commands

        previous = ctx.reader.tell()
        try:
            pos = start_abs
            while pos < end_abs:
                if self._gamecube_is_valid_primitive_command(ctx, pos, end_abs):
                    ctx.reader.seek(pos)
                    opcode = ctx.reader.u8()
                    count = ctx.reader.u16()
                    vertices = []
                    for _ in range(count):
                        matrix_index = ctx.reader.u8()
                        position_index = ctx.reader.u16()
                        normal_index = ctx.reader.u16()
                        color_index = ctx.reader.u16()
                        texcoord_index = ctx.reader.u16()
                        vertices.append((matrix_index, position_index, normal_index, color_index, texcoord_index))
                    commands.append({
                        'offset': pos,
                        'opcode': opcode,
                        'count': count,
                        'vertices': vertices,
                    })
                    pos = ctx.reader.tell()
                    continue
                pos += 1
        finally:
            ctx.reader.seek(previous)
        return commands

    def _iter_gamecube_triangles(self, command: dict):
        vertices = command['vertices']
        opcode = command['opcode']
        if opcode == 0x81:
            for i in range(0, len(vertices), 4):
                quad = vertices[i:i + 4]
                if len(quad) == 4:
                    yield (quad[0], quad[1], quad[2])
                    yield (quad[0], quad[2], quad[3])
        elif opcode == 0x91:
            for i in range(0, len(vertices), 3):
                tri = vertices[i:i + 3]
                if len(tri) == 3:
                    yield (tri[0], tri[1], tri[2])
        elif opcode == 0x99:
            for i in range(max(0, len(vertices) - 2)):
                a = vertices[i]
                b = vertices[i + 1]
                c = vertices[i + 2]
                if (i % 2) == 0:
                    yield (a, b, c)
                else:
                    yield (b, a, c)


    def _parse_gamecube_primdata_entry(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int) -> dict | None:
        br = context.reader
        previous = br.tell()
        try:
            if absolute_offset < context.data_start or absolute_offset + 0x2E > context.data_end:
                return None
            br.seek(absolute_offset)
            valid_prim = br.u32()
            if valid_prim != 1:
                return None

            br.u32()
            br.u32()
            draw_group = br.i32()
            tpageid = br.i32()
            env_mapping = br.i32()
            br.u32()
            br.u32()
            offset_next_primdata = br.u32()
            primitive_data_size = br.u32()
            primitive_data_offset_field_offset = br.tell() - context.data_start
            primitive_data_offset = br.u32()
            primitive_target_context, primitive_absolute_offset = resolve_pointer(cache, context, primitive_data_offset_field_offset, primitive_data_offset)
            bone_id_count = br.u16()
            bone_ids = [br.u16() for _ in range(bone_id_count)]

            nested_palettes: List[dict] = []
            while br.tell() + 2 <= context.data_end:
                entry_pos = br.tell()
                marker = br.peek_u16(offset=entry_pos)
                if marker == 0xFFFF:
                    break

                # wtf?
                has_alignment_padding = (entry_pos % 4) != 0
                if has_alignment_padding:
                    if entry_pos + 12 > context.data_end:
                        break
                    br.u16()

                if br.tell() + 10 > context.data_end:
                    break

                primitive_size = br.u32()
                nested_primitive_offset_field_offset = br.tell() - context.data_start
                primitive_offset = br.u32()
                nested_target_context, nested_primitive_absolute_offset = resolve_pointer(cache, context, nested_primitive_offset_field_offset, primitive_offset)
                nested_bone_count = br.u16()
                if br.tell() + (nested_bone_count * 2) > context.data_end:
                    break

                nested_bone_ids = [br.u16() for _ in range(nested_bone_count)]
                nested_palettes.append({
                    'absolute_offset': entry_pos,
                    'primitive_data_size': primitive_size,
                    'primitive_data_offset': primitive_offset,
                    'primitive_context': nested_target_context,
                    'primitive_absolute_offset': nested_primitive_absolute_offset,
                    'bone_ids': nested_bone_ids,
                })

            return {
                'record_absolute_offset': absolute_offset,
                'draw_group': draw_group,
                'tpageid': tpageid,
                'env_mapping': env_mapping,
                'offset_next_primdata': offset_next_primdata,
                'primitive_data_size': primitive_data_size,
                'primitive_data_offset': primitive_data_offset,
                'primitive_context': primitive_target_context,
                'primitive_absolute_offset': primitive_absolute_offset,
                'bone_ids': bone_ids,
                'nested_palettes': nested_palettes,
            }
        finally:
            br.seek(previous)

    def _assign_gamecube_palettes(self, record: dict, commands: List[dict]) -> None:
        base_palette = list(record.get('bone_ids') or [])
        for command in commands:
            command['palette'] = list(base_palette)

        if not commands:
            return

        sorted_offsets = [int(command.get('offset', -1)) for command in commands]
        record_primitive_context = record.get('primitive_context')
        for nested in record.get('nested_palettes', []):
            bone_ids = list(nested.get('bone_ids') or [])
            if not bone_ids:
                continue

            nested_context = nested.get('primitive_context')
            nested_abs = int(nested.get('primitive_absolute_offset', 0) or 0)
            if nested_abs <= 0 or nested_context is None:
                continue
            if record_primitive_context is not None and nested_context.file_name != record_primitive_context.file_name:
                continue

            start_index = None
            for idx, cmd_offset in enumerate(sorted_offsets):
                if cmd_offset >= nested_abs:
                    start_index = idx
                    break
            if start_index is None:
                continue

            primitive_count = max(0, int(nested.get('primitive_data_size', 0) or 0))
            if primitive_count <= 0:
                continue

            end_index = min(len(commands), start_index + primitive_count)
            for idx in range(start_index, end_index):
                commands[idx]['palette'] = list(bone_ids)

    def _resolve_gamecube_transform(self, matrix_index: int, palette: List[int], segments: List[Segment], virt_segments: List[VirtSegment]) -> tuple[int, int, int, int, float]:
        total_transforms = len(segments) + len(virt_segments)
        transform_id = -1

        slot_candidates = [int(matrix_index) // 3, int(matrix_index)]
        for slot in slot_candidates:
            if 0 <= slot < len(palette):
                transform_id = int(palette[slot])
                break

        if transform_id < 0 and 0 <= int(matrix_index) < total_transforms:
            transform_id = int(matrix_index)

        return resolve_weighted_transform(transform_id, segments, virt_segments)

    def _find_gamecube_primitive_records(self, cache: SectionContextCache, context: SectionContext) -> List[dict]:
        records: List[dict] = []
        seen_targets: set[tuple[str, int]] = set()
        previous = context.reader.tell()
        try:
            scan_start = context.data_start
            scan_end = max(scan_start, context.data_end - 52)
            for absolute_offset in range(scan_start, scan_end + 1, 4):
                context.reader.seek(absolute_offset)
                if context.reader.u32() != 1:
                    continue
                field_local_offset = (absolute_offset - context.data_start) + 0x28
                primitive_raw, primitive_ctx, primitive_abs = self._read_u32_pointer_at(cache, context, field_local_offset)
                if primitive_abs <= 0 or primitive_abs >= primitive_ctx.file_size:
                    continue
                primitive_reader_prev = primitive_ctx.reader.tell()
                try:
                    primitive_ctx.reader.seek(primitive_abs)
                    first_opcode = primitive_ctx.reader.u8()
                finally:
                    primitive_ctx.reader.seek(primitive_reader_prev)
                if first_opcode not in (0x81, 0x91, 0x99):
                    continue
                key = (primitive_ctx.file_name, primitive_abs)
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                record = {
                    'record_absolute_offset': absolute_offset,
                    'record_local_offset': absolute_offset - context.data_start,
                    'primitive_raw': primitive_raw,
                    'primitive_context': primitive_ctx,
                    'primitive_absolute_offset': primitive_abs,
                }
                primdata = self._parse_gamecube_primdata_entry(cache, context, absolute_offset)
                if primdata is not None:
                    record.update(primdata)
                records.append(record)
        finally:
            context.reader.seek(previous)

        records.sort(key=lambda entry: (entry['primitive_context'].file_name, entry['primitive_absolute_offset']))
        return records

    def _parse_gamecube_geometry(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        *,
        version: int,
        segments: List[Segment],
        virt_segments: List[VirtSegment],
        hmarkers: List[HMarker],
        hspheres: List[HSphere],
        hboxes: List[HBox],
        hcapsules: List[HCapsule],
        targets: List[Target],
        max_rad: float = 0.0,
        max_rad_sq: float = 0.0,
        bone_mirror_entries: List[BoneMirrorEntry] | None = None,
    ) -> ModelData:
        br = context.reader
        previous = br.tell()
        try:
            br.seek(context.data_start + 0x10)
            model_scale = br.vec4()
        finally:
            br.seek(previous)

        primitive_records = self._find_gamecube_primitive_records(cache, context)
        if not primitive_records:
            logger.warning('No valid Gamecube primitive records found in %s', context.file_name)

        positions_raw, positions_ctx, positions_abs = self._read_u32_pointer_at(cache, context, 0x64)
        uv_raw, uv_ctx, uv_abs = self._read_u32_pointer_at(cache, context, 0x68)
        vertex_colors_raw, vertex_colors_ctx, vertex_colors_abs = self._read_u32_pointer_at(cache, context, 0x6C)
        normal_raw, normal_ctx, normal_abs = self._read_u32_pointer_at(cache, context, 0x70)

        component_entries = [
            ('positions', positions_ctx, positions_abs, 6),
            ('vertex_colors', vertex_colors_ctx, vertex_colors_abs, 4),
            ('uvs', uv_ctx, uv_abs, 4),
            ('normals', normal_ctx, normal_abs, 3),
        ]
        bounds_by_context: dict[str, List[int]] = {}
        for _name, target_ctx, absolute_offset, _stride in component_entries:
            if absolute_offset > 0:
                bounds_by_context.setdefault(target_ctx.file_name, []).append(absolute_offset)
        for entry in primitive_records:
            prim_ctx = entry['primitive_context']
            prim_abs = entry['primitive_absolute_offset']
            bounds_by_context.setdefault(prim_ctx.file_name, []).append(prim_abs)

        def next_boundary(target_ctx: SectionContext, start_abs: int) -> int:
            candidates = [target_ctx.data_end]
            for bound in bounds_by_context.get(target_ctx.file_name, []):
                if bound > start_abs:
                    candidates.append(bound)
            return min(candidates) if candidates else target_ctx.data_end

        positions_end = next_boundary(positions_ctx, positions_abs) if positions_abs > 0 else 0
        colors_end = next_boundary(vertex_colors_ctx, vertex_colors_abs) if vertex_colors_abs > 0 else 0
        uvs_end = next_boundary(uv_ctx, uv_abs) if uv_abs > 0 else 0
        normals_end = next_boundary(normal_ctx, normal_abs) if normal_abs > 0 else 0

        positions = self._read_gamecube_positions(positions_ctx, positions_abs, positions_end)
        vertex_color_table = self._read_gamecube_colors(vertex_colors_ctx, vertex_colors_abs, colors_end)
        uv_table = self._read_gamecube_uvs(uv_ctx, uv_abs, uvs_end)
        normals = self._read_gamecube_normals(normal_ctx, normal_abs, normals_end)

        logger.debug(
            'Gamecube geometry in %s: primitive_records=%d positions=%d uvs=%d vertex_colors=%d normals=%d (raw pointers: prim=%s pos=0x%X uv=0x%X col=0x%X nrm=0x%X)',
            context.file_name,
            len(primitive_records),
            len(positions),
            len(uv_table),
            len(vertex_color_table),
            len(normals),
            ', '.join(f"0x{entry['primitive_raw']:X}" for entry in primitive_records) or 'none',
            positions_raw,
            uv_raw,
            vertex_colors_raw,
            normal_raw,
        )

        commands: List[dict] = []
        for index, entry in enumerate(primitive_records):
            prim_ctx = entry['primitive_context']
            prim_abs = entry['primitive_absolute_offset']
            next_primitive_offsets = [
                other['primitive_absolute_offset']
                for other in primitive_records
                if other['primitive_context'].file_name == prim_ctx.file_name and other['primitive_absolute_offset'] > prim_abs
            ]
            candidates = [prim_ctx.data_end]
            if next_primitive_offsets:
                candidates.append(min(next_primitive_offsets))
            for _name, comp_ctx, comp_abs, _stride in component_entries:
                if comp_ctx.file_name == prim_ctx.file_name and comp_abs > prim_abs:
                    candidates.append(comp_abs)
            region_end = min(candidates)
            region_commands = self._parse_gamecube_primitive_commands(prim_ctx, prim_abs, region_end)
            for command in region_commands:
                command['record_index'] = index
                command['draw_group'] = int(entry.get('draw_group', index))
                command['tpageid'] = int(entry.get('tpageid', 0))
            self._assign_gamecube_palettes(entry, region_commands)
            commands.extend(region_commands)

        source_virt_segments = virt_segments
        output_virt_segments: List[VirtSegment] = []

        vertex_map: dict[tuple[int, int, int, int, int, int], int] = {}
        vertices: List[MVertex] = []
        vertex_colors: List[Tuple[int, int, int, int]] = []
        faces: List[MFace] = []
        strips: List[TextureStrip] = []
        segment_ref_counts: List[Counter[int]] = []

        def materialize_vertex(vertex_ref: tuple[int, int, int, int, int], palette: List[int]) -> int | None:
            matrix_index, position_index, normal_index, color_index, texcoord_index = vertex_ref
            transform_id, bind_segment, primary_segment, secondary_segment, secondary_weight = self._resolve_gamecube_transform(
                int(matrix_index),
                palette,
                segments,
                source_virt_segments,
            )
            key = (transform_id, position_index, normal_index, color_index, texcoord_index, bind_segment)
            existing = vertex_map.get(key)
            if existing is not None:
                if 0 <= int(bind_segment) < len(segments):
                    segment_ref_counts[existing][int(bind_segment)] += 1
                return existing
            if position_index < 0 or position_index >= len(positions):
                return None
            if normal_index < 0 or normal_index >= len(normals):
                return None
            if color_index < 0 or color_index >= len(vertex_color_table):
                return None
            if texcoord_index < 0 or texcoord_index >= len(uv_table):
                return None
            vertex_index = len(vertices)
            initial_segment = int(primary_segment) if 0 <= int(primary_segment) < len(segments) else -1
            if initial_segment < 0 and 0 <= int(bind_segment) < len(segments):
                initial_segment = int(bind_segment)
            vertices.append(MVertex(
                index=vertex_index,
                position_raw=positions[position_index],
                normal_raw=normals[normal_index],
                segment=initial_segment,
                uv_raw=uv_table[texcoord_index],
            ))
            if 0 <= int(primary_segment) < len(segments) and 0 <= int(secondary_segment) < len(segments) and 0.0 < float(secondary_weight) < 1.0:
                output_virt_segments.append(make_weighted_virt_segment(
                    vertex_index,
                    primary_segment,
                    secondary_segment,
                    secondary_weight,
                    len(output_virt_segments),
                ))
            vertex_colors.append(vertex_color_table[color_index])
            segment_counter: Counter[int] = Counter()
            if initial_segment >= 0:
                segment_counter[initial_segment] += 1
            segment_ref_counts.append(segment_counter)
            vertex_map[key] = vertex_index
            return vertex_index

        strip_triangle_indices: dict[int, List[int]] = {index: [] for index in range(len(primitive_records))}

        for command in commands:
            record_index = int(command.get('record_index', 0))
            palette = list(command.get('palette') or [])
            for triangle in self._iter_gamecube_triangles(command):
                tri_indices = []
                reject_triangle = False
                position_keys = []
                for vertex_ref in triangle:
                    position_keys.append((vertex_ref[1], vertex_ref[2], vertex_ref[3], vertex_ref[4], vertex_ref[0]))
                    vertex_index = materialize_vertex(vertex_ref, palette)
                    if vertex_index is None:
                        reject_triangle = True
                        break
                    tri_indices.append(vertex_index)
                if reject_triangle or len(set(position_keys)) < 3 or len(set(tri_indices)) < 3:
                    continue
                faces.append(MFace(index=len(faces), v0=tri_indices[0], v1=tri_indices[1], v2=tri_indices[2], same_vert_bits=0))
                strip_triangle_indices.setdefault(record_index, []).extend(tri_indices)

        self._assign_gamecube_vertex_segments(vertices, segment_ref_counts, len(segments))
        vertices, vertex_colors, faces, strips, segments, output_virt_segments = self._reindex_gamecube_vertices_by_segment(
            vertices,
            vertex_colors,
            faces,
            [
                TextureStrip(
                    offset=index,
                    vertex_count=len(indices),
                    draw_group=int(primitive_records[index].get('draw_group', index)),
                    tpageid=int(primitive_records[index].get('tpageid', 0)),
                    sort_push=0.0,
                    scroll_offset=0.0,
                    env_mapping=int(primitive_records[index].get('env_mapping', 0)),
                    next_texture=-1,
                    indices=list(indices),
                )
                for index, indices in sorted(strip_triangle_indices.items()) if indices
            ],
            segments,
            output_virt_segments,
        )

        return ModelData(
            version=version,
            model_scale=model_scale,
            segments=segments,
            virt_segments=output_virt_segments,
            vertices=vertices,
            faces=faces,
            strips=strips,
            vertex_colors=vertex_colors,
            hmarkers=hmarkers,
            hspheres=hspheres,
            hboxes=hboxes,
            hcapsules=hcapsules,
            targets=targets,
            max_rad=float(max_rad),
            max_rad_sq=float(max_rad_sq),
            bone_mirror_entries=list(bone_mirror_entries or []),
            cdc_render_data_id=0,
            uv_format='gamecube',
        )


    def _assign_gamecube_vertex_segments(
        self,
        vertices: List[MVertex],
        segment_ref_counts: List[Counter[int]],
        num_segments: int,
    ) -> None:
        if not vertices:
            return

        for vertex_index, vertex in enumerate(vertices):
            chosen_segment = int(vertex.segment)
            counter = segment_ref_counts[vertex_index] if vertex_index < len(segment_ref_counts) else None
            if counter:
                best_segment = -1
                best_count = -1
                for segment_index, count in counter.items():
                    if 0 <= int(segment_index) < num_segments and int(count) > best_count:
                        best_segment = int(segment_index)
                        best_count = int(count)
                if best_segment >= 0:
                    chosen_segment = best_segment

            if not (0 <= chosen_segment < num_segments):
                chosen_segment = -1
            vertex.segment = chosen_segment

    def _reindex_gamecube_vertices_by_segment(
        self,
        vertices: List[MVertex],
        vertex_colors: List[Tuple[int, int, int, int]],
        faces: List[MFace],
        strips: List[TextureStrip],
        segments: List[Segment],
        virt_segments: List[VirtSegment],
    ) -> tuple[List[MVertex], List[Tuple[int, int, int, int]], List[MFace], List[TextureStrip], List[Segment], List[VirtSegment]]:
        if not vertices:
            return vertices, vertex_colors, faces, strips, segments, virt_segments

        old_vertex_count = len(vertices)
        valid_segment_indices = set(range(len(segments)))
        grouped_old_indices: List[int] = []
        segment_to_old_indices: dict[int, List[int]] = {segment.index: [] for segment in segments}
        unassigned_old_indices: List[int] = []

        for old_index, vertex in enumerate(vertices):
            segment_index = int(vertex.segment)
            if segment_index in valid_segment_indices:
                segment_to_old_indices.setdefault(segment_index, []).append(old_index)
            else:
                unassigned_old_indices.append(old_index)

        for segment in segments:
            grouped_old_indices.extend(segment_to_old_indices.get(segment.index, []))
        grouped_old_indices.extend(unassigned_old_indices)

        if len(grouped_old_indices) != old_vertex_count:
            seen = set(grouped_old_indices)
            grouped_old_indices.extend(old_index for old_index in range(old_vertex_count) if old_index not in seen)

        old_to_new = {old_index: new_index for new_index, old_index in enumerate(grouped_old_indices)}

        reordered_vertices: List[MVertex] = []
        reordered_colors: List[Tuple[int, int, int, int]] = []
        for new_index, old_index in enumerate(grouped_old_indices):
            vertex = vertices[old_index]
            vertex.index = new_index
            reordered_vertices.append(vertex)
            if old_index < len(vertex_colors):
                reordered_colors.append(vertex_colors[old_index])

        remapped_faces: List[MFace] = []
        for face_index, face in enumerate(faces):
            remapped_faces.append(
                MFace(
                    index=face_index,
                    v0=old_to_new[int(face.v0)],
                    v1=old_to_new[int(face.v1)],
                    v2=old_to_new[int(face.v2)],
                    same_vert_bits=int(face.same_vert_bits),
                )
            )

        remapped_strips: List[TextureStrip] = []
        for strip in strips:
            remapped_strip = TextureStrip(
                offset=int(strip.offset),
                vertex_count=int(strip.vertex_count),
                draw_group=int(strip.draw_group),
                tpageid=int(strip.tpageid),
                sort_push=float(strip.sort_push),
                scroll_offset=float(strip.scroll_offset),
                env_mapping=int(getattr(strip, 'env_mapping', 0)),
                next_texture=int(strip.next_texture),
                indices=[old_to_new[int(index)] for index in strip.indices if int(index) in old_to_new],
            )
            remapped_strip.vertex_count = len(remapped_strip.indices)
            remapped_strips.append(remapped_strip)

        for segment in segments:
            segment_indices = segment_to_old_indices.get(segment.index, [])
            if segment_indices:
                remapped = [old_to_new[old_index] for old_index in segment_indices]
                segment.first_vertex = min(remapped)
                segment.last_vertex = max(remapped)
            else:
                segment.first_vertex = 0
                segment.last_vertex = -1

        for virt_segment in virt_segments:
            start = max(0, int(virt_segment.first_vertex))
            end = min(old_vertex_count - 1, int(virt_segment.last_vertex))
            if start <= end:
                remapped = [old_to_new[old_index] for old_index in range(start, end + 1)]
                virt_segment.first_vertex = min(remapped)
                virt_segment.last_vertex = max(remapped)
            else:
                virt_segment.first_vertex = 0
                virt_segment.last_vertex = -1

        return reordered_vertices, reordered_colors, remapped_faces, remapped_strips, segments, virt_segments

