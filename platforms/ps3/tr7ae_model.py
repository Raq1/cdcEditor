from __future__ import annotations

from typing import Dict, List, Tuple

from ...core.log import logger
from ...core.model_types import (
    BoneMirrorEntry,
    HBox,
    HCapsule,
    HMarker,
    MFace,
    MVertex,
    ModelData,
    Segment,
    Target,
    TextureStrip,
    VirtSegment,
)
from ..common.model import make_segments_only_model_data
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..pc.tr7ae_model import TRModelParser


class TRPS3ModelParser(TRModelParser):
    """Importer for PS3 TRLAU model sections.

    PS3 DRM containers and section headers are big-endian. The model skeleton
    records use the normal 64-byte TRLAU segment layout, but most rendered
    geometry is stored in a PS3 render stream referenced from the model header:

    - header field 0x58 -> big-endian u16 triangle-strip index stream
    - header field 0x5C -> stream vertex-buffer header
    - vertex-buffer header +0x00: vertex count
    - vertex-buffer header +0x04: pointer to 16-byte vertex records

    The first six bytes of each PS3 stream vertex are signed big-endian XYZ
    coordinates.  Bytes 8 and 9 hold the two compact skin slots observed in
    character streams, while bytes 12 and 13 hold the corresponding 8-bit
    blend weights.  The stream appears to be limited to two weights per
    vertex.  The remaining payload bytes are still treated conservatively
    until normals/UVs are fully mapped.
    """

    PS3_STREAM_VERTEX_SIZE = 16
    PS3_STREAM_EXTRA_VERTEX_SIZE = 20
    PS3_STREAM_ALIGNMENT = 0x20
    PS3_RESTART_INDICES = {0xFFFF, 0xFFFE}

    def __init__(self, filepath: str, import_hinfo: bool = True, import_markups: bool = True, parse_segments_only: bool = False):
        super().__init__(filepath, import_hinfo=import_hinfo, import_markups=import_markups, endian='>', parse_segments_only=parse_segments_only)

    def _is_gamecube_layout(self) -> bool:
        # PS3 is big-endian, but it does not use Gamecube/Wii compact segment
        # records or the Nintendo-specific geometry block.
        return False

    def _read_u32_pointer_at(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        field_local_offset: int,
    ) -> Tuple[int, SectionContext, int]:
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(field_local_offset))
            raw_value = context.reader.u32()
            target_context, absolute_offset = resolve_pointer(cache, context, int(field_local_offset), raw_value)
            return raw_value, target_context, absolute_offset
        finally:
            context.reader.seek(previous)

    def _parse_segments_and_hinfo(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        segment_list_ctx: SectionContext,
        segment_list_abs: int,
        num_segments: int,
        num_virt_segments: int,
    ) -> Tuple[List[Segment], List[VirtSegment], List[HMarker], List[HSphere], List[HBox], List[HCapsule], List[Target]]:
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
                if not self.import_hinfo:
                    continue
                hinfo_field_local_offset = (segment_record_start - segment_list_ctx.data_start) + (self._segment_record_size() - 4)
                if segment.hinfo or hinfo_field_local_offset in segment_list_ctx.section_info.relocations_by_offset:
                    hinfo_ctx, hinfo_abs = resolve_pointer(cache, segment_list_ctx, hinfo_field_local_offset, segment.hinfo)
                    if hinfo_abs:
                        hspheres.extend(self._parse_hinfo_hspheres(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_sphere_index_start=len(hspheres)))
                        hboxes.extend(self._parse_hinfo_hboxes(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_hbox_index_start=len(hboxes)))
                        hmarkers.extend(self._parse_hinfo_hmarkers(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_marker_index_start=len(hmarkers)))
                        hcapsules.extend(self._parse_hinfo_hcapsules(cache, hinfo_ctx, hinfo_abs, owner_segment=index, global_capsule_index_start=len(hcapsules)))

            for virt_index in range(num_virt_segments):
                virt_segments.append(self._parse_virt_segment(segment_list_ctx, virt_index))

        return segments, virt_segments, hmarkers, hspheres, hboxes, hcapsules, targets

    @staticmethod
    def _segment_lookup_from_ranges(segments: List[Segment]) -> Dict[int, int]:
        lookup: Dict[int, int] = {}
        for segment in segments:
            try:
                start = int(segment.first_vertex)
                end = int(segment.last_vertex)
            except Exception:
                continue
            if start < 0 or end < start:
                continue
            # Avoid one bad segment record claiming an implausibly huge range.
            if end - start > 1000000:
                continue
            for vertex_index in range(start, end + 1):
                lookup.setdefault(vertex_index, int(segment.index))
        return lookup

    @staticmethod
    def _ps3_read_u16_from_context(context: SectionContext, absolute_offset: int) -> int:
        previous = context.reader.tell()
        try:
            context.reader.seek(int(absolute_offset))
            return int(context.reader.u16())
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _ps3_read_u32_from_context(context: SectionContext, absolute_offset: int) -> int:
        previous = context.reader.tell()
        try:
            context.reader.seek(int(absolute_offset))
            return int(context.reader.u32())
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _decode_ps3_texture_id(tpage_value: int) -> int:
        """Return the texture-resource id packed into an observed PS3 tpage word."""
        texture_id = int(tpage_value) & 0x1FFF
        if texture_id <= 0 or texture_id == 0x1FFF:
            return -1
        return int(texture_id)

    @classmethod
    def _read_ps3_material_texture_bindings(cls, material_context: SectionContext, record_abs: int) -> Dict[str, object]:
        """Extract observed PS3 material texture stages.

        Character PS3 material records use stable texture slots rather than the
        generic PC tpage-only material.  The supplied samples show that the
        slot previously guessed as the normal/bump map is actually the
        specular/gloss stage:

        - +0x10: primary diffuse/base-color texture
        - +0x18: specular/gloss texture stage

        Other stage words are still preserved as raw metadata.  We do not bind
        any of them as a normal map automatically because the exact PS3 normal
        map slot/encoding is not confirmed yet, and wiring the specular stage
        into Blender's Normal input produces visibly incorrect shading.  The
        PS3 Material panel exposes the editable Normal Texture ID field for
        manual assignment once a specific model's normal stage is identified.
        """
        stage_offsets = (0x14, 0x18, 0x20, 0x24, 0x28)
        diffuse_raw = cls._ps3_read_u32_from_context(material_context, record_abs + 0x10)
        diffuse_texture_id = cls._decode_ps3_texture_id(diffuse_raw)

        stage_tpageids: List[int] = []
        stage_texture_ids: List[int] = []
        stage_by_offset: Dict[int, tuple[int, int]] = {}
        for offset in stage_offsets:
            raw = cls._ps3_read_u32_from_context(material_context, record_abs + offset)
            texture_id = cls._decode_ps3_texture_id(raw)
            stage_tpageids.append(int(raw))
            stage_texture_ids.append(int(texture_id))
            stage_by_offset[int(offset)] = (int(raw), int(texture_id))

        def _valid_stage(offset: int, used: set[int]) -> tuple[int, int]:
            raw, texture_id = stage_by_offset.get(int(offset), (-1, -1))
            if texture_id < 0 or texture_id in used:
                return -1, -1
            return int(raw), int(texture_id)

        used: set[int] = {diffuse_texture_id} if diffuse_texture_id >= 0 else set()

        # The observed +0x18 stage is spec/gloss, not a tangent normal map.
        # Newer PS3 material records also have a valid +0x14 auxiliary stage;
        # keep it available as a candidate, but do not treat it as specular
        # before +0x18 unless +0x18 is absent.
        specular_raw, specular_texture_id = _valid_stage(0x18, used)
        if specular_texture_id >= 0:
            used.add(int(specular_texture_id))
        else:
            for offset in (0x14, 0x24, 0x28, 0x20):
                specular_raw, specular_texture_id = _valid_stage(offset, used)
                if specular_texture_id >= 0:
                    used.add(int(specular_texture_id))
                    break

        # Treat the first remaining auxiliary stage as the normal-map
        # candidate and bind it as the imported PS3 normal texture.  The raw
        # candidate fields are preserved so the mapping can still be audited.
        normal_candidate_raw, normal_candidate_texture_id = -1, -1
        for offset in (0x14, 0x20, 0x28, 0x24):
            candidate_raw, candidate_texture_id = _valid_stage(offset, used)
            if candidate_texture_id >= 0:
                normal_candidate_raw = int(candidate_raw)
                normal_candidate_texture_id = int(candidate_texture_id)
                used.add(int(candidate_texture_id))
                break

        normal_raw, normal_texture_id = int(normal_candidate_raw), int(normal_candidate_texture_id)

        return {
            'diffuse_tpageid_raw': int(diffuse_raw),
            'diffuse_texture_id': int(diffuse_texture_id),
            'normal_tpageid_raw': int(normal_raw),
            'normal_texture_id': int(normal_texture_id),
            'normal_candidate_tpageid_raw': int(normal_candidate_raw),
            'normal_candidate_texture_id': int(normal_candidate_texture_id),
            'specular_tpageid_raw': int(specular_raw),
            'specular_texture_id': int(specular_texture_id),
            'stage_tpageids': stage_tpageids,
            'stage_texture_ids': stage_texture_ids,
        }

    def _read_ps3_material_draw_packets(
        self,
        material_context: SectionContext,
        material_abs: int,
        num_segments: int,
        total_index_values: int,
    ) -> List[Dict[str, object]]:
        """Parse PS3 material draw packets and their compact-bone palettes.

        The PS3 render path does not store global segment ids in the 16-byte
        vertex records.  It stores compact matrix slots.  The material/draw
        records referenced by model header field 0x4C carry the actual palette
        for each index-stream range:

            u32 index_start
            u32 index_count
            u16 palette_count
            u16 palette[palette_count]

        Multiple packets can be embedded between one material record and the
        next material-record pointer.  Parsing these packets is more reliable
        than assigning one guessed palette to an entire mesh or strip.

        Material record +0x0C is the observed PS3 drawgroup value.  It is not
        generated from packet/material order; game scripting uses this value to
        hide/show material groups.
        """
        if material_abs <= 0 or material_abs + 0x94 > material_context.file_size:
            return []

        packets: List[Dict[str, object]] = []
        data_start = int(material_context.data_start)
        data_end = min(int(material_context.data_end), int(material_context.file_size))
        data_size = max(0, data_end - data_start)
        if not (data_start <= int(material_abs) < data_end):
            return []

        visited: set[int] = set()
        record_abs = int(material_abs)
        for _record_index in range(1024):
            if record_abs in visited:
                break
            if not (data_start <= record_abs + 0x94 <= data_end):
                break
            visited.add(record_abs)

            first_word = self._ps3_read_u32_from_context(material_context, record_abs)
            if first_word != 1:
                break

            bindings = self._read_ps3_material_texture_bindings(material_context, record_abs)
            texture_word = int(bindings.get('diffuse_tpageid_raw', -1))
            tpageid = texture_word if int(bindings.get('diffuse_texture_id', -1)) >= 0 else -1
            draw_group = int(self._ps3_read_u32_from_context(material_context, record_abs + 0x0C))

            next_raw = self._ps3_read_u32_from_context(material_context, record_abs + 0x88)
            next_abs = data_start + int(next_raw) if 0 < int(next_raw) < data_size else 0
            record_end = next_abs if next_abs > record_abs else data_end
            record_end = min(record_end, data_end)

            cursor = record_abs + 0x8C
            packet_index = 0
            while cursor + 10 <= record_end:
                index_start = self._ps3_read_u32_from_context(material_context, cursor)
                index_count = self._ps3_read_u32_from_context(material_context, cursor + 4)
                palette_count = self._ps3_read_u16_from_context(material_context, cursor + 8)

                if int(index_count) <= 0:
                    break
                if int(index_start) < 0 or int(index_start) >= max(1, int(total_index_values)):
                    break
                if int(index_start) + int(index_count) > int(total_index_values) + 8:
                    break
                if int(palette_count) < 0 or int(palette_count) > 32:
                    break

                palette: List[int] = []
                palette_abs = cursor + 10
                if int(palette_count) > 0:
                    if palette_abs + (int(palette_count) * 2) > record_end:
                        break
                    valid_palette = True
                    for i in range(int(palette_count)):
                        value = self._ps3_read_u16_from_context(material_context, palette_abs + (i * 2))
                        if not (0 <= int(value) < int(num_segments)):
                            valid_palette = False
                            break
                        palette.append(int(value))
                    if not valid_palette:
                        break

                packets.append(
                    {
                        'record_index': len(visited) - 1,
                        'record_offset': record_abs - data_start,
                        'packet_index': packet_index,
                        'draw_group': int(draw_group),
                        'index_start': int(index_start),
                        'index_count': int(index_count),
                        'palette': palette,
                        'tpageid': int(tpageid),
                        'ps3_diffuse_tpageid_raw': int(bindings.get('diffuse_tpageid_raw', -1)),
                        'ps3_normal_tpageid_raw': int(bindings.get('normal_tpageid_raw', -1)),
                        'ps3_specular_tpageid_raw': int(bindings.get('specular_tpageid_raw', -1)),
                        'ps3_diffuse_texture_id': int(bindings.get('diffuse_texture_id', -1)),
                        'ps3_normal_texture_id': int(bindings.get('normal_texture_id', -1)),
                        'ps3_normal_candidate_tpageid_raw': int(bindings.get('normal_candidate_tpageid_raw', -1)),
                        'ps3_normal_candidate_texture_id': int(bindings.get('normal_candidate_texture_id', -1)),
                        'ps3_specular_texture_id': int(bindings.get('specular_texture_id', -1)),
                        'ps3_texture_stage_ids': [int(value) for value in bindings.get('stage_texture_ids', [])],
                        'ps3_texture_stage_tpageids': [int(value) for value in bindings.get('stage_tpageids', [])],
                    }
                )

                cursor = palette_abs + (int(palette_count) * 2)
                cursor = (cursor + 3) & ~3
                packet_index += 1

            if not next_abs or next_abs <= record_abs or next_abs >= data_end:
                break
            record_abs = next_abs

        if packets:
            logger.debug(
                'Parsed %d PS3 material draw packet(s) from %s',
                len(packets),
                material_context.file_name,
            )
        return packets

    @staticmethod
    def _triangle_strip_to_triangles(indices: List[int]) -> List[int]:
        triangles: List[int] = []
        for i in range(len(indices) - 2):
            a, b, c = int(indices[i]), int(indices[i + 1]), int(indices[i + 2])
            if a == b or b == c or a == c:
                continue
            if i % 2:
                tri = (b, a, c)
            else:
                tri = (a, b, c)
            triangles.extend(tri)
        return triangles

    def _read_ps3_index_values(
        self,
        stream_context: SectionContext,
        index_stream_abs: int,
        vertex_header_abs: int,
    ) -> Tuple[List[int], int]:
        if index_stream_abs <= 0 or vertex_header_abs <= index_stream_abs:
            return [], 0
        data_end = min(int(vertex_header_abs), stream_context.file_size)
        if data_end <= index_stream_abs:
            return [], 0
        br = stream_context.reader
        previous = br.tell()
        values: List[int] = []
        restart_count = 0
        try:
            br.seek(index_stream_abs)
            while br.tell() + 2 <= data_end:
                value = int(br.u16())
                values.append(value)
                if value in self.PS3_RESTART_INDICES:
                    restart_count += 1
        finally:
            br.seek(previous)
        return values, restart_count

    def _read_ps3_index_triangles(
        self,
        stream_context: SectionContext,
        index_stream_abs: int,
        vertex_header_abs: int,
    ) -> Tuple[List[int], int]:
        values, restart_count = self._read_ps3_index_values(stream_context, index_stream_abs, vertex_header_abs)
        triangles: List[int] = []
        strip: List[int] = []
        for value in values:
            if int(value) in self.PS3_RESTART_INDICES:
                if len(strip) >= 3:
                    triangles.extend(self._triangle_strip_to_triangles(strip))
                strip = []
                continue
            strip.append(int(value))
        if len(strip) >= 3:
            triangles.extend(self._triangle_strip_to_triangles(strip))
        return triangles, restart_count


    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        alignment = max(1, int(alignment))
        return (int(value) + alignment - 1) & ~(alignment - 1)

    @staticmethod
    def _i8_from_byte(value: int) -> int:
        value = int(value) & 0xFF
        return value - 0x100 if value & 0x80 else value

    @classmethod
    def _decode_ps3_extra_normal(cls, record: bytes) -> Tuple[int, int, int]:
        """Decode the best observed PS3 vector candidate from the secondary stream.

        The PS3 render payload for the supplied TRLAU character samples stores
        the primary 16-byte stream first, followed by a 20-byte-per-vertex
        attribute stream aligned to 0x20.  In that attribute stream bytes 4..7
        are UVs and bytes 8..11 are the first signed 8-bit vector candidate.

        Earlier builds used X/Y/Z = byte10/byte9/byte8.  Cross-checking the
        same stream against generated geometric normals across the Lara, Natla,
        Demon Natla, Kid, and Kold samples shows that the least-bad ordering is
        X/Y/Z = -byte11/-byte10/+byte8.  This is still treated as provisional
        by the mesh builder; if the vector set fails a face-normal sanity check,
        Blender-generated smooth normals are used instead.
        """
        if len(record) < 12:
            return (0, 0, 127)
        nx = -cls._i8_from_byte(record[11])
        ny = -cls._i8_from_byte(record[10])
        nz = cls._i8_from_byte(record[8])
        if nx == 0 and ny == 0 and nz == 0:
            return (0, 0, 127)
        return (nx, ny, nz)

    @staticmethod
    def _decode_ps3_extra_color(record: bytes) -> Tuple[int, int, int, int] | None:
        """Decode the observed PS3 secondary-stream vertex color.

        The first four bytes of the 20-byte PS3 secondary vertex record behave
        like ARGB color/modulation data in the supplied character and prop
        samples.  Import RGB as Blender color data and keep the first byte as
        alpha metadata, but the PS3 shader does not use this alpha for
        transparency automatically.
        """
        if len(record) < 4:
            return None
        a = int(record[0]) & 0xFF
        r = int(record[1]) & 0xFF
        g = int(record[2]) & 0xFF
        b = int(record[3]) & 0xFF
        return (r, g, b, a)

    def _read_ps3_extra_vertex_records(
        self,
        vertex_data_ctx: SectionContext,
        primary_vertex_data_abs: int,
        vertex_count: int,
    ) -> Tuple[List[bytes], int]:
        """Read the PS3 secondary attribute stream when present.

        The model header only points at the first 16-byte stream.  The observed
        PS3 character files place an aligned 20-byte stream immediately after it:

            bytes 0..3   vertex color / render attribute
            bytes 4..7   UV, big-endian u16 S/T
            bytes 8..11  normal vector, signed byte Z/Y/X/W
            bytes 12..15 tangent-like vector
            bytes 16..19 bitangent-like vector
        """
        count = int(vertex_count)
        if count <= 0:
            return [], 0
        primary_local = int(primary_vertex_data_abs) - int(vertex_data_ctx.data_start)
        if primary_local < 0:
            return [], 0
        extra_local = self._align_up(primary_local + (count * self.PS3_STREAM_VERTEX_SIZE), self.PS3_STREAM_ALIGNMENT)
        extra_abs = int(vertex_data_ctx.data_start) + int(extra_local)
        required_end = extra_abs + (count * self.PS3_STREAM_EXTRA_VERTEX_SIZE)
        data_limit = min(int(vertex_data_ctx.data_end), int(vertex_data_ctx.file_size))
        if extra_abs <= 0 or required_end > data_limit:
            return [], int(extra_abs)

        br = vertex_data_ctx.reader
        previous = br.tell()
        records: List[bytes] = []
        try:
            br.seek(extra_abs)
            for _index in range(count):
                records.append(br.read(self.PS3_STREAM_EXTRA_VERTEX_SIZE))
        finally:
            br.seek(previous)
        return records, int(extra_abs)

    def _read_ps3_external_stream_vertices(
        self,
        cache: SectionContextCache,
        vertex_stream_ctx: SectionContext,
        vertex_header_abs: int,
        vertex_count: int,
        segments: List[Segment],
    ) -> Tuple[List[MVertex], int, int]:
        """Read the PS3 external render-stream variant.

        Some PS3 models store indices and vertices in a separate type-3 render
        section.  Model header field +0x64 points at the u16 index stream and
        +0x68 points at a small vertex-stream header inside that same section.
        The first word of that header is a relocated/local pointer to the
        16-byte vertex records; the vertex count still comes from the model
        header at +0x20.
        """
        count = int(vertex_count)
        if count <= 0 or count > 10000000:
            return [], int(count), 0
        if vertex_header_abs <= 0 or vertex_header_abs + 4 > vertex_stream_ctx.file_size:
            return [], int(count), 0

        br = vertex_stream_ctx.reader
        previous = br.tell()
        try:
            br.seek(vertex_header_abs)
            vertex_data_raw = int(br.u32())
            field_local_offset = int(vertex_header_abs) - int(vertex_stream_ctx.data_start)
            vertex_data_ctx, vertex_data_abs = resolve_pointer(cache, vertex_stream_ctx, field_local_offset, vertex_data_raw)
        except Exception:
            try:
                br.seek(previous)
            except Exception:
                pass
            return [], int(count), 0
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

        required_end = int(vertex_data_abs) + (count * self.PS3_STREAM_VERTEX_SIZE)
        if vertex_data_abs <= 0 or required_end > min(int(vertex_data_ctx.data_end), int(vertex_data_ctx.file_size)):
            logger.debug(
                'Skipping PS3 external vertex stream; data outside bounds in %s abs=0x%X count=%d end=0x%X data_end=0x%X',
                vertex_data_ctx.file_name,
                int(vertex_data_abs),
                count,
                int(required_end),
                int(vertex_data_ctx.data_end),
            )
            return [], int(count), int(vertex_data_raw)

        extra_vertex_records, extra_vertex_abs = self._read_ps3_extra_vertex_records(
            vertex_data_ctx,
            int(vertex_data_abs),
            count,
        )
        if extra_vertex_records:
            logger.debug(
                'Read PS3 external secondary vertex attributes from %s:0x%X (%d records)',
                vertex_data_ctx.file_name,
                int(extra_vertex_abs),
                len(extra_vertex_records),
            )

        segment_by_vertex = self._segment_lookup_from_ranges(segments)
        vertices: List[MVertex] = []
        vbr = vertex_data_ctx.reader
        vprev = vbr.tell()
        try:
            vbr.seek(int(vertex_data_abs))
            for index in range(count):
                record = vbr.read(self.PS3_STREAM_VERTEX_SIZE)
                x = int.from_bytes(record[0:2], 'big', signed=True)
                y = int.from_bytes(record[2:4], 'big', signed=True)
                z = int.from_bytes(record[4:6], 'big', signed=True)
                fallback_segment = int(segment_by_vertex.get(index, 0))
                extra_record = extra_vertex_records[index] if index < len(extra_vertex_records) else b''
                normal_raw = self._decode_ps3_extra_normal(extra_record) if extra_record else (0, 0, 127)
                ps3_color_rgba = self._decode_ps3_extra_color(extra_record) if extra_record else None
                uv_raw = (0, 0)
                uv_decoded = (0.0, 0.0)
                if len(extra_record) >= 8:
                    uvx = int.from_bytes(extra_record[4:6], 'big', signed=True)
                    uvy = int.from_bytes(extra_record[6:8], 'big', signed=True)
                    uv_raw = (uvx, uvy)
                    uv_decoded = (float(uvx) / 4096.0, 1.0 - (float(uvy) / 4096.0))

                primary_slot = int(record[8])
                secondary_slot = int(record[9])
                primary_weight_u8 = int(record[12])
                secondary_weight_u8 = int(record[13])
                total_weight = primary_weight_u8 + secondary_weight_u8
                secondary_weight = 0.0
                if total_weight > 0 and secondary_slot != primary_slot:
                    secondary_weight = float(secondary_weight_u8) / float(total_weight)

                vertices.append(
                    MVertex(
                        index=index,
                        position_raw=(x, y, z),
                        normal_raw=normal_raw,
                        segment=fallback_segment if 0 <= fallback_segment < len(segments) else 0,
                        uv_raw=uv_raw,
                        uv_decoded=uv_decoded,
                        ps3_color_rgba=ps3_color_rgba,
                        gc_transform_id=primary_slot,
                        gc_bind_segment=fallback_segment if 0 <= fallback_segment < len(segments) else 0,
                        gc_primary_segment=primary_slot,
                        gc_secondary_segment=secondary_slot,
                        gc_secondary_weight=secondary_weight,
                    )
                )
        finally:
            vbr.seek(vprev)
        return vertices, count, int(vertex_data_raw)

    def _read_ps3_stream_vertices(
        self,
        cache: SectionContextCache,
        vertex_header_ctx: SectionContext,
        vertex_header_abs: int,
        segments: List[Segment],
        model_scale: Tuple[float, float, float, float],
    ) -> Tuple[List[MVertex], int, int]:
        if vertex_header_abs <= 0 or vertex_header_abs + 8 > vertex_header_ctx.file_size:
            return [], 0, 0
        br = vertex_header_ctx.reader
        previous = br.tell()
        vertices: List[MVertex] = []
        try:
            br.seek(vertex_header_abs)
            vertex_count = br.u32()
            vertex_data_field_local_offset = self._field_local_offset(vertex_header_ctx)
            vertex_data_raw = br.u32()
            vertex_data_ctx, vertex_data_abs = resolve_pointer(cache, vertex_header_ctx, vertex_data_field_local_offset, vertex_data_raw)
            if vertex_count <= 0 or vertex_count > 10000000:
                logger.warning('Skipping PS3 vertex stream with suspicious vertex count %d at %s:0x%X', vertex_count, vertex_header_ctx.file_name, vertex_header_abs)
                return [], int(vertex_count), int(vertex_data_raw)
            required_end = vertex_data_abs + (int(vertex_count) * self.PS3_STREAM_VERTEX_SIZE)
            if vertex_data_abs <= 0 or required_end > vertex_data_ctx.file_size:
                logger.warning(
                    'Skipping PS3 vertex stream; vertex data outside bounds in %s abs=0x%X count=%d end=0x%X file_size=0x%X',
                    vertex_data_ctx.file_name,
                    vertex_data_abs,
                    vertex_count,
                    required_end,
                    vertex_data_ctx.file_size,
                )
                return [], int(vertex_count), int(vertex_data_raw)

            extra_vertex_records, extra_vertex_abs = self._read_ps3_extra_vertex_records(
                vertex_data_ctx,
                vertex_data_abs,
                int(vertex_count),
            )
            if extra_vertex_records:
                logger.debug(
                    'Read PS3 secondary vertex attributes from %s:0x%X (%d records)',
                    vertex_data_ctx.file_name,
                    extra_vertex_abs,
                    len(extra_vertex_records),
                )

            segment_by_vertex = self._segment_lookup_from_ranges(segments)
            vbr = vertex_data_ctx.reader
            vprev = vbr.tell()
            try:
                vbr.seek(vertex_data_abs)
                for index in range(int(vertex_count)):
                    record = vbr.read(self.PS3_STREAM_VERTEX_SIZE)
                    x = int.from_bytes(record[0:2], 'big', signed=True)
                    y = int.from_bytes(record[2:4], 'big', signed=True)
                    z = int.from_bytes(record[4:6], 'big', signed=True)
                    fallback_segment = int(segment_by_vertex.get(index, 0))
                    extra_record = extra_vertex_records[index] if index < len(extra_vertex_records) else b''
                    normal_raw = self._decode_ps3_extra_normal(extra_record) if extra_record else (0, 0, 127)
                    ps3_color_rgba = self._decode_ps3_extra_color(extra_record) if extra_record else None
                    uv_raw = (0, 0)
                    uv_decoded = (0.0, 0.0)
                    if len(extra_record) >= 8:
                        uvx = int.from_bytes(extra_record[4:6], 'big', signed=True)
                        uvy = int.from_bytes(extra_record[6:8], 'big', signed=True)
                        uv_raw = (uvx, uvy)
                        uv_decoded = (float(uvx) / 4096.0, 1.0 - (float(uvy) / 4096.0))

                    # PS3 stream vertices store compact skin slots.  The real
                    # segment ids are supplied by the draw packet's bone
                    # palette, so keep the compact slots on the intermediate
                    # vertex and resolve them during packet materialization.
                    primary_slot = int(record[8])
                    secondary_slot = int(record[9])
                    primary_weight_u8 = int(record[12])
                    secondary_weight_u8 = int(record[13])
                    total_weight = primary_weight_u8 + secondary_weight_u8
                    secondary_weight = 0.0
                    if total_weight > 0 and secondary_slot != primary_slot:
                        secondary_weight = float(secondary_weight_u8) / float(total_weight)

                    vertices.append(
                        MVertex(
                            index=index,
                            position_raw=(x, y, z),
                            normal_raw=normal_raw,
                            segment=fallback_segment if 0 <= fallback_segment < len(segments) else 0,
                            uv_raw=uv_raw,
                            uv_decoded=uv_decoded,
                            ps3_color_rgba=ps3_color_rgba,
                            gc_transform_id=primary_slot,
                            gc_bind_segment=fallback_segment if 0 <= fallback_segment < len(segments) else 0,
                            gc_primary_segment=primary_slot,
                            gc_secondary_segment=secondary_slot,
                            gc_secondary_weight=secondary_weight,
                        )
                    )
            finally:
                vbr.seek(vprev)
        finally:
            br.seek(previous)
        return vertices, int(vertex_count), int(vertex_data_raw)

    def _materialize_ps3_vertex(
        self,
        source: MVertex,
        output_index: int,
        palette: List[int],
        segments: List[Segment],
    ) -> MVertex:
        primary_slot = int(source.gc_primary_segment if source.gc_primary_segment >= 0 else source.gc_transform_id)
        secondary_slot = int(source.gc_secondary_segment)
        fallback_segment = int(source.segment if 0 <= source.segment < len(segments) else 0)

        primary_segment = fallback_segment
        if 0 <= primary_slot < len(palette):
            primary_segment = int(palette[primary_slot])
        elif 0 <= primary_slot < len(segments):
            primary_segment = int(primary_slot)

        secondary_segment = -1
        secondary_weight = float(source.gc_secondary_weight or 0.0)
        if secondary_weight > 0.0:
            if 0 <= secondary_slot < len(palette):
                secondary_segment = int(palette[secondary_slot])
            elif 0 <= secondary_slot < len(segments):
                secondary_segment = int(secondary_slot)

        if not (0 <= primary_segment < len(segments)):
            primary_segment = fallback_segment
        if not (0 <= secondary_segment < len(segments)) or secondary_segment == primary_segment:
            secondary_segment = -1
            secondary_weight = 0.0

        return MVertex(
            index=output_index,
            position_raw=source.position_raw,
            normal_raw=source.normal_raw,
            segment=primary_segment,
            uv_raw=source.uv_raw,
            uv_decoded=source.uv_decoded,
            psp_color_rgba=source.psp_color_rgba,
            ps3_color_rgba=source.ps3_color_rgba,
            gc_transform_id=primary_slot,
            gc_bind_segment=primary_segment,
            gc_primary_segment=primary_segment,
            gc_secondary_segment=secondary_segment,
            gc_secondary_weight=secondary_weight,
        )

    def _materialize_ps3_draw_geometry(
        self,
        source_vertices: List[MVertex],
        index_values: List[int],
        packets: List[Dict[str, object]],
        segments: List[Segment],
        index_stream_abs: int,
        source_file: str,
    ) -> Tuple[List[MVertex], List[TextureStrip]]:
        if not source_vertices or not index_values or not packets:
            return [], []

        output_vertices: List[MVertex] = []
        strips: List[TextureStrip] = []

        for packet_number, packet in enumerate(packets):
            try:
                start = int(packet.get('index_start', 0))
                count = int(packet.get('index_count', 0))
                palette = [int(value) for value in packet.get('palette', [])]  # type: ignore[arg-type]
                tpageid = int(packet.get('tpageid', -1))
                material_group = int(packet.get('record_index', packet_number))
                ps3_diffuse_tpageid_raw = int(packet.get('ps3_diffuse_tpageid_raw', -1))
                ps3_normal_tpageid_raw = int(packet.get('ps3_normal_tpageid_raw', -1))
                ps3_specular_tpageid_raw = int(packet.get('ps3_specular_tpageid_raw', -1))
                ps3_diffuse_texture_id = int(packet.get('ps3_diffuse_texture_id', -1))
                ps3_normal_texture_id = int(packet.get('ps3_normal_texture_id', -1))
                ps3_normal_candidate_tpageid_raw = int(packet.get('ps3_normal_candidate_tpageid_raw', -1))
                ps3_normal_candidate_texture_id = int(packet.get('ps3_normal_candidate_texture_id', -1))
                ps3_specular_texture_id = int(packet.get('ps3_specular_texture_id', -1))
                ps3_texture_stage_ids = [int(value) for value in packet.get('ps3_texture_stage_ids', [])]
                ps3_texture_stage_tpageids = [int(value) for value in packet.get('ps3_texture_stage_tpageids', [])]
                draw_group = int(packet.get('draw_group', 0))
            except Exception:
                continue
            if count <= 0 or start < 0 or start >= len(index_values):
                continue

            raw_indices = index_values[start:min(len(index_values), start + count)]
            if not raw_indices:
                continue

            vertex_map: Dict[int, int] = {}
            mapped_strip: List[int] = []
            mapped_triangles: List[int] = []

            def materialized_index(source_index: int) -> int:
                existing = vertex_map.get(source_index)
                if existing is not None:
                    return existing
                source = source_vertices[source_index]
                new_index = len(output_vertices)
                output_vertices.append(self._materialize_ps3_vertex(source, new_index, palette, segments))
                vertex_map[source_index] = new_index
                return new_index

            for value in raw_indices:
                if int(value) in self.PS3_RESTART_INDICES:
                    if len(mapped_strip) >= 3:
                        mapped_triangles.extend(self._triangle_strip_to_triangles(mapped_strip))
                    mapped_strip = []
                    continue
                source_index = int(value)
                if not (0 <= source_index < len(source_vertices)):
                    continue
                mapped_strip.append(materialized_index(source_index))

            if len(mapped_strip) >= 3:
                mapped_triangles.extend(self._triangle_strip_to_triangles(mapped_strip))

            if not mapped_triangles:
                continue

            strips.append(
                TextureStrip(
                    offset=int(index_stream_abs) + (start * 2),
                    vertex_count=len(mapped_triangles),
                    draw_group=draw_group,
                    tpageid=tpageid,
                    sort_push=0.0,
                    scroll_offset=0.0,
                    next_texture=0,
                    indices=mapped_triangles,
                    bone_ids=palette,
                    material_group=material_group,
                    source_file=source_file,
                    ps3_diffuse_tpageid_raw=ps3_diffuse_tpageid_raw,
                    ps3_normal_tpageid_raw=ps3_normal_tpageid_raw,
                    ps3_specular_tpageid_raw=ps3_specular_tpageid_raw,
                    ps3_diffuse_texture_id=ps3_diffuse_texture_id,
                    ps3_normal_texture_id=ps3_normal_texture_id,
                    ps3_normal_candidate_tpageid_raw=ps3_normal_candidate_tpageid_raw,
                    ps3_normal_candidate_texture_id=ps3_normal_candidate_texture_id,
                    ps3_specular_texture_id=ps3_specular_texture_id,
                    ps3_texture_stage_ids=ps3_texture_stage_ids,
                    ps3_texture_stage_tpageids=ps3_texture_stage_tpageids,
                )
            )

        return output_vertices, strips


    def _read_ps3_primary_draw_group(self, material_context: SectionContext, material_abs: int) -> int:
        if material_abs <= 0 or material_abs + 0x90 > material_context.file_size:
            return 0
        try:
            first_word = self._ps3_read_u32_from_context(material_context, material_abs)
            if int(first_word) != 1:
                return 0
            # Some header pointers lead to compact dependency/id lists that also
            # start with 1.  Only treat +0x0C as drawgroup when the surrounding
            # words look like a real PS3 material record.
            sampler_word = int(self._ps3_read_u32_from_context(material_context, material_abs + 0x14))
            if (sampler_word & 0xFFFF0000) not in {0x00050000, 0x00058000}:
                return 0
            draw_group = int(self._ps3_read_u32_from_context(material_context, material_abs + 0x0C))
            return draw_group if draw_group >= 0 else 0
        except Exception:
            return 0

    def _read_ps3_primary_texture_id(self, material_context: SectionContext, material_abs: int) -> int:
        if material_abs <= 0 or material_abs + 0x18 > material_context.file_size:
            return -1
        br = material_context.reader
        previous = br.tell()
        try:
            br.seek(material_abs)
            material_count = br.u32()
            if material_count <= 0 or material_count > 4096:
                return -1
            br.seek(material_abs + 0x10)
            raw_texture_field = br.u32()
            return self._decode_ps3_texture_id(raw_texture_field)
        except Exception:
            return -1
        finally:
            br.seek(previous)

    def _parse_pc_fallback_geometry(
        self,
        vertex_list_ctx: SectionContext,
        vertex_list_abs: int,
        num_vertices: int,
        face_list_ctx: SectionContext,
        face_list_abs: int,
        num_faces: int,
    ) -> Tuple[List[MVertex], List[MFace]]:
        vertices: List[MVertex] = []
        faces: List[MFace] = []
        if num_vertices > 0 and vertex_list_abs:
            vertex_list_ctx.reader.seek(vertex_list_abs)
            for index in range(int(num_vertices)):
                if vertex_list_ctx.reader.tell() + 16 > vertex_list_ctx.file_size:
                    break
                vertices.append(self._parse_vertex(vertex_list_ctx, index))
        if num_faces > 0 and face_list_abs:
            face_list_ctx.reader.seek(face_list_abs)
            for index in range(int(num_faces)):
                if face_list_ctx.reader.tell() + 8 > face_list_ctx.file_size:
                    break
                faces.append(self._parse_face(face_list_ctx, index))
        return vertices, faces

    def _parse_model(self, cache: SectionContextCache, context: SectionContext) -> ModelData:
        br = context.reader
        br.seek(context.data_start)

        version = br.i32()
        num_segments = br.i32()
        num_virt_segments = br.i32()
        segment_list_raw, segment_list_ctx, segment_list_abs = self._read_u32_pointer(cache, context)

        logger.debug(
            'PS3 model header in %s: version=0x%X num_segments=%d num_virt_segments=%d segment_list=0x%X',
            context.file_name,
            version & 0xFFFFFFFF,
            num_segments,
            num_virt_segments,
            segment_list_raw,
        )

        segments, virt_segments, hmarkers, hspheres, hboxes, hcapsules, targets = self._parse_segments_and_hinfo(
            cache,
            context,
            segment_list_ctx,
            segment_list_abs,
            num_segments,
            num_virt_segments,
        )

        if self.parse_segments_only:
            return make_segments_only_model_data(
                version=version,
                segments=segments,
                virt_segments=virt_segments,
                strips=[],
                hmarkers=hmarkers,
                hspheres=hspheres,
                hboxes=hboxes,
                hcapsules=hcapsules,
                uv_format='ps3',
            )

        # PS3 model header is close to PC, but has no obsoleteAniTextures field
        # before maxRad. Fields below are fixed local offsets relative to the
        # model section payload.
        br.seek(context.data_start + 0x10)
        model_scale = br.vec4()
        num_vertices = br.i32()
        vertex_list_raw, vertex_list_ctx, vertex_list_abs = self._read_u32_pointer(cache, context)
        num_normals = br.i32()
        normal_list_raw, normal_list_ctx, normal_list_abs = self._read_u32_pointer(cache, context)
        num_faces = br.i32()
        face_list_raw, face_list_ctx, face_list_abs = self._read_u32_pointer(cache, context)
        max_rad = br.f32() if br.tell() + 4 <= context.file_size else 0.0
        max_rad_sq = br.f32() if br.tell() + 4 <= context.file_size else 0.0

        material_info_raw, material_info_ctx, material_info_abs = self._read_u32_pointer_at(cache, context, 0x4C)
        primary_texture_id = self._read_ps3_primary_texture_id(material_info_ctx, material_info_abs)
        primary_draw_group = self._read_ps3_primary_draw_group(material_info_ctx, material_info_abs)
        index_stream_raw, index_stream_ctx, index_stream_abs = self._read_u32_pointer_at(cache, context, 0x58)
        vertex_header_raw, vertex_header_ctx, vertex_header_abs = self._read_u32_pointer_at(cache, context, 0x5C)
        external_index_raw, external_index_ctx, external_index_abs = self._read_u32_pointer_at(cache, context, 0x64)
        external_vertex_header_raw, external_vertex_header_ctx, external_vertex_header_abs = self._read_u32_pointer_at(cache, context, 0x68)
        bone_mirror_raw, bone_mirror_ctx, bone_mirror_abs = self._read_u32_pointer_at(cache, context, 0x84)

        logger.debug(
            'PS3 model geometry fields: numVertices=%d vertexList=0x%X numFaces=%d faceList=0x%X normalList=0x%X materialInfo=0x%X indexStream=0x%X vertexHeader=0x%X externalIndex=0x%X externalVertexHeader=0x%X',
            num_vertices,
            vertex_list_raw,
            num_faces,
            face_list_raw,
            normal_list_raw,
            material_info_raw,
            index_stream_raw,
            vertex_header_raw,
            external_index_raw,
            external_vertex_header_raw,
        )

        vertices: List[MVertex] = []
        faces: List[MFace] = []
        strips: List[TextureStrip] = []

        stream_vertices, stream_vertex_count, vertex_data_raw = self._read_ps3_stream_vertices(
            cache,
            vertex_header_ctx,
            vertex_header_abs,
            segments,
            model_scale,
        )
        active_index_ctx = index_stream_ctx
        active_index_abs = index_stream_abs
        active_index_end_abs = vertex_header_abs if vertex_header_ctx.filepath == index_stream_ctx.filepath else index_stream_ctx.data_end

        preloaded_index_values: List[int] | None = None
        preloaded_restart_count = 0
        used_external_render_stream = False

        if not stream_vertices and external_index_abs and external_vertex_header_abs:
            external_index_end_abs = external_vertex_header_abs if external_vertex_header_ctx.filepath == external_index_ctx.filepath else external_index_ctx.data_end
            external_index_values, external_restart_count = self._read_ps3_index_values(
                external_index_ctx,
                external_index_abs,
                external_index_end_abs,
            )
            external_max_index = max(
                (int(value) for value in external_index_values if int(value) not in self.PS3_RESTART_INDICES),
                default=-1,
            )
            # In the external PS3 render-stream variant, the model-header
            # vertex count is sometimes the logical/CPU count, not the GPU
            # stream count.  The u16 index stream and the packed vertex/extra
            # streams line up exactly when the GPU count is max(index)+1.
            # Using the smaller header count drops valid indexed vertices and
            # makes these models import with zero geometry.
            external_stream_vertex_count = max(int(num_vertices), int(external_max_index) + 1)
            external_vertices, external_vertex_count, external_vertex_data_raw = self._read_ps3_external_stream_vertices(
                cache,
                external_vertex_header_ctx,
                external_vertex_header_abs,
                external_stream_vertex_count,
                segments,
            )
            if external_vertices:
                stream_vertices = external_vertices
                stream_vertex_count = external_vertex_count
                vertex_data_raw = external_vertex_data_raw
                active_index_ctx = external_index_ctx
                active_index_abs = external_index_abs
                active_index_end_abs = external_index_end_abs
                preloaded_index_values = external_index_values
                preloaded_restart_count = external_restart_count
                used_external_render_stream = True
                logger.debug(
                    'Using PS3 external render stream for %s: vertices=%d headerVertices=%d maxIndex=%d index_source=%s:0x%X vertex_header=%s:0x%X',
                    context.file_name,
                    len(stream_vertices),
                    int(num_vertices),
                    int(external_max_index),
                    active_index_ctx.file_name,
                    int(active_index_abs),
                    external_vertex_header_ctx.file_name,
                    int(external_vertex_header_abs),
                )

        if preloaded_index_values is not None:
            index_values, restart_count = preloaded_index_values, preloaded_restart_count
        else:
            index_values, restart_count = self._read_ps3_index_values(
                active_index_ctx,
                active_index_abs,
                active_index_end_abs,
            )
        draw_packets = self._read_ps3_material_draw_packets(
            material_info_ctx,
            material_info_abs,
            len(segments),
            len(index_values),
        )
        if not draw_packets and index_stream_abs:
            # Some PS3 DRMs point materialInfo at a tiny render-data stub while
            # the actual material records live at the model header's indexStream
            # pointer.  The index buffer itself can still be external; the
            # packet offsets/counts are relative to the active index stream.
            draw_packets = self._read_ps3_material_draw_packets(
                index_stream_ctx,
                index_stream_abs,
                len(segments),
                len(index_values),
            )
            if draw_packets:
                logger.debug(
                    'Parsed PS3 material/draw packets for %s from indexStream records instead of materialInfo',
                    context.file_name,
                )

        if stream_vertices and index_values and draw_packets:
            vertices, strips = self._materialize_ps3_draw_geometry(
                stream_vertices,
                index_values,
                draw_packets,
                segments,
                active_index_abs,
                active_index_ctx.file_name,
            )
            if vertices and strips:
                logger.debug(
                    'Parsed PS3 packet geometry for %s: source_vertices=%d materialized_vertices=%d triangles=%d packets=%d restarts=%d vertexDataRaw=0x%X',
                    context.file_name,
                    len(stream_vertices),
                    len(vertices),
                    sum(len(strip.indices) for strip in strips) // 3,
                    len(draw_packets),
                    restart_count,
                    vertex_data_raw,
                )

        if not vertices:
            stream_triangles, restart_count = self._read_ps3_index_triangles(
                active_index_ctx,
                active_index_abs,
                active_index_end_abs,
            )
            if stream_vertices and stream_triangles:
                max_index = max(stream_triangles, default=-1)
                if max_index < len(stream_vertices):
                    vertices = stream_vertices
                    strips = [
                        TextureStrip(
                            offset=int(active_index_abs),
                            vertex_count=len(stream_triangles),
                            draw_group=primary_draw_group,
                            tpageid=primary_texture_id if primary_texture_id >= 0 else -1,
                            sort_push=0.0,
                            scroll_offset=0.0,
                            next_texture=0,
                            indices=stream_triangles,
                            material_group=0,
                            source_file=active_index_ctx.file_name,
                            ps3_diffuse_texture_id=primary_texture_id if primary_texture_id >= 0 else -1,
                            ps3_diffuse_tpageid_raw=primary_texture_id if primary_texture_id >= 0 else -1,
                        )
                    ]
                    logger.debug(
                        'Parsed PS3 stream geometry for %s without material packets: vertices=%d triangles=%d restarts=%d vertexDataRaw=0x%X',
                        context.file_name,
                        len(vertices),
                        len(stream_triangles) // 3,
                        restart_count,
                        vertex_data_raw,
                    )
                else:
                    logger.warning(
                        'PS3 stream index max %d exceeds vertex count %d in %s; falling back to standard geometry lists',
                        max_index,
                        len(stream_vertices),
                        context.file_name,
                    )

        if not vertices:
            vertices, faces = self._parse_pc_fallback_geometry(
                vertex_list_ctx,
                vertex_list_abs,
                num_vertices,
                face_list_ctx,
                face_list_abs,
                num_faces,
            )
            logger.debug('Parsed PS3 fallback geometry for %s: vertices=%d faces=%d', context.file_name, len(vertices), len(faces))

        bone_mirror_entries: List[BoneMirrorEntry] = []
        if bone_mirror_abs and 0 < bone_mirror_abs < bone_mirror_ctx.file_size:
            try:
                bone_mirror_entries = self._parse_bone_mirror_entries(bone_mirror_ctx, bone_mirror_abs)
            except Exception as exc:
                logger.debug('Failed to parse PS3 bone mirror data from %s: %s', bone_mirror_ctx.file_name, exc)

        ps3_vertex_colors = None
        if vertices:
            decoded_colors = [getattr(vertex, 'ps3_color_rgba', None) for vertex in vertices]
            if any(color is not None for color in decoded_colors):
                ps3_vertex_colors = [
                    tuple(int(c) for c in color) if color is not None else (255, 255, 255, 255)
                    for color in decoded_colors
                ]

        # The usual target/markup block offsets are not confirmed for PS3; skip
        # them instead of reading unrelated render data as objects.
        return ModelData(
            version=version,
            model_scale=model_scale,
            segments=segments,
            virt_segments=virt_segments,
            vertices=vertices,
            faces=faces,
            strips=strips,
            vertex_colors=ps3_vertex_colors,
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
            cdc_render_data_id=0,
            uv_format='ps3',
            ps3_external_render_stream=bool(used_external_render_stream),
        )
