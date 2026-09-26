from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ...core.log import logger
from ...core.model_types import BoneMirrorEntry, HBox, HCapsule, HMarker, HSphere, MFace, MVertex, ModelData, Segment, Target, TextureStrip, VirtSegment
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..common.model import (
    make_segments_only_model_data,
    parse_compact_segment,
    parse_compact_virt_segment,
    parse_standard_segment,
    parse_standard_virt_segment,
)
from ..nintendo.tr7ae_model import GameCubeModelParserMixin


@dataclass(slots=True)
class ModelCameraAnticData:
    use_antic_camera: int = 0
    new_dtp_camera_antic_data_id: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)


@dataclass(slots=True)
class ModelMarkupData:
    index: int
    game: str = 'unknown'
    override_movement_camera: int = 0
    dtp_camera_data_id: int = 0
    dtp_markup_data_id: int = 0
    animated_segment: int = 0
    camera_antic: Optional[ModelCameraAnticData] = None
    flags: int = 0
    intro_id: int = -1
    markup_id: int = -1
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    polyline_offset: int = 0
    polyline: List[Tuple[float, float, float, float]] = field(default_factory=list)


class TRModelParser(GameCubeModelParserMixin):
    def __init__(self, filepath: str, import_hinfo: bool = True, import_markups: bool = True, endian: str = "<", parse_segments_only: bool = False):
        self.filepath = filepath
        self.import_hinfo = import_hinfo
        self.import_markups = import_markups
        self.endian = endian
        self.parse_segments_only = parse_segments_only

    def parse(self) -> ModelData:
        cache = SectionContextCache(self.filepath, endian=self.endian)
        try:
            context = cache.get_root_context()
            logger.debug(
                'Starting model parse from %s (type=%d, data_start=0x%X)',
                context.file_name,
                context.section_info.section_type,
                context.data_start,
            )
            return self._parse_model(cache, context)
        finally:
            cache.close()

    def _field_local_offset(self, context: SectionContext, field_size: int = 0) -> int:
        return context.reader.tell() - context.data_start - field_size

    def _read_u32_pointer(self, cache: SectionContextCache, context: SectionContext) -> Tuple[int, SectionContext, int]:
        field_local_offset = self._field_local_offset(context)
        raw_value = context.reader.u32()
        target_context, absolute_offset = resolve_pointer(cache, context, field_local_offset, raw_value)
        return raw_value, target_context, absolute_offset

    def _is_gamecube_layout(self) -> bool:
        return self.endian == ">"

    def _segment_record_size(self) -> int:
        return 32 if self._is_gamecube_layout() else 64

    def _virt_segment_record_size(self) -> int:
        return 32 if self._is_gamecube_layout() else 64

    def _parse_segment(self, context: SectionContext, index: int) -> Segment:
        if self._is_gamecube_layout():
            return parse_compact_segment(context, index)
        return parse_standard_segment(context, index)

    def _parse_virt_segment(self, context: SectionContext, virt_index: int) -> VirtSegment:
        if self._is_gamecube_layout():
            return parse_compact_virt_segment(context, virt_index)
        return parse_standard_virt_segment(context, virt_index)

    def _parse_vertex(self, context: SectionContext, index: int) -> MVertex:
        br = context.reader
        x = br.i16()
        y = br.i16()
        z = br.i16()
        nx = br.i8()
        ny = br.i8()
        nz = br.i8()
        br.u8()
        segment = br.i16()
        uvx = br.u16()
        uvy = br.u16()
        return MVertex(
            index=index,
            position_raw=(x, y, z),
            normal_raw=(nx, ny, nz),
            segment=segment,
            uv_raw=(uvx, uvy),
        )

    def _parse_face(self, context: SectionContext, index: int) -> MFace:
        br = context.reader
        return MFace(
            index=index,
            v0=br.u16(),
            v1=br.u16(),
            v2=br.u16(),
            same_vert_bits=br.u16(),
        )

    def _parse_texture_strip_chain(
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

        while current_absolute_offset:
            visit_key = (current_context.file_name, current_absolute_offset)
            if visit_key in visited_offsets:
                logger.debug('Stopping texture strip parse due to loop at %s:0x%X', current_context.file_name, current_absolute_offset)
                break
            visited_offsets.add(visit_key)

            if current_absolute_offset >= current_context.data_end:
                logger.warning(
                    'Texture strip pointer 0x%X is outside data bounds for %s (data_end=0x%X)',
                    current_absolute_offset,
                    current_context.file_name,
                    current_context.data_end,
                )
                break

            br = current_context.reader
            br.seek(current_absolute_offset)
            if br.tell() + 2 > current_context.file_size:
                logger.warning('Cannot read texture strip header at 0x%X in %s', current_absolute_offset, current_context.file_name)
                break

            if br.peek_i16() == 0:
                logger.debug('Texture strip chain terminated by zero entry at %s:0x%X', current_context.file_name, current_absolute_offset)
                break

            entry_offset = br.tell()
            vertex_count = br.i16()
            draw_group = br.i16()
            tpageid = br.i32()
            sort_push = br.f32()
            scroll_offset = br.f32()
            next_field_local_offset = br.tell() - current_context.data_start
            next_texture = br.u32()

            logger.debug(
                'Texture strip entry at %s:0x%X vertex_count=%d draw_group=%d tpageid=%d nextTexture=0x%X',
                current_context.file_name,
                entry_offset,
                vertex_count,
                draw_group,
                tpageid,
                next_texture,
            )

            if vertex_count < 0:
                logger.warning('Negative texture strip vertex count %d at %s:0x%X', vertex_count, current_context.file_name, entry_offset)
                break

            index_bytes_end = br.tell() + (vertex_count * 2)
            if index_bytes_end > current_context.file_size:
                logger.warning(
                    'Texture strip indices overflow file bounds at %s:0x%X (vertex_count=%d)',
                    current_context.file_name,
                    entry_offset,
                    vertex_count,
                )
                break
            indices = [br.u16() for _ in range(vertex_count)]

            if br.tell() + 2 <= current_context.file_size and br.peek_i16() == -1:
                br.skip(2)

            strips.append(
                TextureStrip(
                    offset=entry_offset,
                    vertex_count=vertex_count,
                    draw_group=draw_group,
                    tpageid=tpageid,
                    sort_push=sort_push,
                    scroll_offset=scroll_offset,
                    next_texture=next_texture,
                    indices=indices,
                    source_file=current_context.file_name,
                )
            )

            if next_texture == 0:
                break

            current_context, current_absolute_offset = resolve_pointer(
                cache,
                current_context,
                next_field_local_offset,
                next_texture,
            )

        return strips

    def _parse_model_scroll_info(
        self,
        cache: SectionContextCache,
        scroll_context: SectionContext,
        absolute_offset: int,
        strips: List[TextureStrip],
    ) -> None:
        if absolute_offset <= 0 or not self._is_valid_abs(scroll_context, absolute_offset, 4):
            return

        br = scroll_context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            try:
                count = br.i32()
            except EOFError:
                return

            if count <= 0:
                return
            if count > 100000:
                logger.debug(
                    'Model scrollInfo has suspicious count %d at %s:0x%X',
                    count,
                    scroll_context.file_name,
                    absolute_offset,
                )
                return

            unresolved = 0
            for entry_index in range(count):
                entry_abs = absolute_offset + 4 + (entry_index * 20)
                if not self._is_valid_abs(scroll_context, entry_abs, 20):
                    unresolved += 1
                    break

                br.seek(entry_abs)
                br.u32()
                scroll_speed = br.f32()
                scroll_num_tiles = br.i32()
                scroll_tile = br.i32()

                fixup_field_local_offset = self._field_local_offset(scroll_context)
                fixup_raw = br.u32()
                fixup_context, fixup_abs = resolve_pointer(
                    cache,
                    scroll_context,
                    fixup_field_local_offset,
                    fixup_raw,
                )

                strip_index = self._resolve_model_scroll_strip_index(
                    fixup_context,
                    fixup_abs,
                    fixup_raw,
                    strips,
                )
                if strip_index is None:
                    unresolved += 1
                    continue

                strip = strips[int(strip_index)]
                strip.scroll_entry_count += 1
                if not strip.scroll_speeds or all(abs(float(existing) - float(scroll_speed)) > 1e-6 for existing in strip.scroll_speeds):
                    strip.scroll_speeds.append(float(scroll_speed))
                if strip.scroll_num_tiles == 0:
                    strip.scroll_num_tiles = int(scroll_num_tiles)
                if strip.scroll_tile == 0:
                    strip.scroll_tile = int(scroll_tile)

            if unresolved:
                logger.debug(
                    'Model scrollInfo unresolved entries: %d/%d at %s:0x%X',
                    unresolved,
                    count,
                    scroll_context.file_name,
                    absolute_offset,
                )
        finally:
            br.seek(previous)

    def _resolve_model_scroll_strip_index(
        self,
        fixup_context: SectionContext,
        fixup_abs: int,
        fixup_raw: int,
        strips: List[TextureStrip],
    ) -> Optional[int]:
        if not strips:
            return None

        candidates: List[Tuple[int, int]] = []

        preferred_deltas = (12, 8, 0, 16, 4, 20, -4, -8)
        for strip_index, strip in enumerate(strips):
            source_file = str(getattr(strip, 'source_file', '') or '')
            if source_file and source_file != fixup_context.file_name:
                continue
            try:
                strip_abs = int(strip.offset)
            except Exception:
                continue
            for rank, delta in enumerate(preferred_deltas):
                if int(fixup_abs) == strip_abs + int(delta):
                    score = 100 - (rank * 10)
                    if delta == 12:
                        score += 25
                    candidates.append((score, int(strip_index)))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]

        for strip_index, strip in enumerate(strips):
            try:
                strip_local = int(strip.offset) - int(fixup_context.data_start)
            except Exception:
                continue
            for rank, delta in enumerate(preferred_deltas):
                if int(fixup_raw) == strip_local + int(delta):
                    score = 50 - rank
                    if delta == 12:
                        score += 10
                    candidates.append((score, int(strip_index)))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return None


    def _parse_bone_mirror_entries(
        self,
        context: SectionContext,
        bone_mirror_data_abs: int,
    ) -> List[BoneMirrorEntry]:
        entries: List[BoneMirrorEntry] = []
        if bone_mirror_data_abs <= 0 or bone_mirror_data_abs >= context.file_size:
            return entries

        br = context.reader
        data_limit = min(context.file_size, context.data_end)
        br.seek(bone_mirror_data_abs)
        while br.tell() + 5 <= data_limit:
            if br.peek_u16() == 0:
                break

            bone1 = br.u8()
            bone2 = br.u8()
            count = br.u8()
            entries.append(BoneMirrorEntry(bone1=bone1, bone2=bone2, count=count))

            if br.peek_u16() == 0:
                break

        return entries

    def _parse_vertex_colors(
        self,
        context: SectionContext,
        num_vertices: int,
    ) -> List[Tuple[int, int, int, int]]:
        br = context.reader
        colors: List[Tuple[int, int, int, int]] = []
        for index in range(num_vertices):
            b = br.u8()
            g = br.u8()
            r = br.u8()
            a = br.u8()
            colors.append((r, g, b, a))
        logger.debug(
            'Read %d BGRA vertex colors from %s at absolute 0x%X',
            num_vertices,
            context.file_name,
            br.tell() - (num_vertices * 4),
        )
        return colors


    def _parse_index_list(
        self,
        context: SectionContext,
        absolute_offset: int,
        raw_value: int,
        label: str,
    ) -> List[int]:
        if absolute_offset <= 0 or absolute_offset + 4 > context.file_size:
            return []

        br = context.reader
        br.seek(absolute_offset)
        try:
            index_count = br.i32()
        except EOFError:
            logger.warning('%s count read failed at %s:0x%X', label, context.file_name, absolute_offset)
            return []

        if index_count <= 0:
            logger.debug('%s list is empty at %s:0x%X (raw=0x%X)', label, context.file_name, absolute_offset, raw_value)
            return []

        data_end = br.tell() + (index_count * 2)
        if data_end > context.file_size:
            logger.warning(
                'Skipping %s read: ptr=0x%X abs=0x%X count=%d would end at 0x%X beyond %s file_size=0x%X',
                label,
                raw_value,
                absolute_offset,
                index_count,
                data_end,
                context.file_name,
                context.file_size,
            )
            return []

        indices = [br.u16() for _ in range(index_count)]
        logger.debug(
            'Read %d %s indices from %s at absolute 0x%X',
            len(indices),
            label,
            context.file_name,
            absolute_offset,
        )
        return indices


    def _parse_hinfo_hspheres(
        self,
        cache: SectionContextCache,
        source_context: SectionContext,
        hinfo_absolute_offset: int,
        owner_segment: int,
        global_sphere_index_start: int,
    ) -> List[HSphere]:
        if hinfo_absolute_offset <= 0 or hinfo_absolute_offset + 32 > source_context.file_size:
            return []

        br = source_context.reader
        previous_offset = br.tell()
        try:
            br.seek(hinfo_absolute_offset)

            num_hspheres = br.i32()
            hspheres_field_local_offset = self._field_local_offset(source_context)
            hspheres_raw, hspheres_ctx, hspheres_abs = self._read_u32_pointer(cache, source_context)
            num_hboxes = br.i32()
            _hboxes_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hmarkers = br.i32()
            _hmarkers_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hcapsules = br.i32()
            _hcapsules_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)

            logger.debug(
                'HInfo HSpheres for segment %d at %s:0x%X hspheres=%d hboxes=%d hmarkers=%d hcapsules=%d',
                owner_segment,
                source_context.file_name,
                hinfo_absolute_offset,
                num_hspheres,
                num_hboxes,
                num_hmarkers,
                num_hcapsules,
            )

            if num_hspheres <= 0 or hspheres_abs <= 0:
                return []

            record_size = 28
            if hspheres_abs + (num_hspheres * record_size) > hspheres_ctx.file_size:
                logger.warning(
                    'Skipping HSphere read for segment %d: ptr=0x%X abs=0x%X count=%d exceeds %s file_size=0x%X',
                    owner_segment,
                    hspheres_raw,
                    hspheres_abs,
                    num_hspheres,
                    hspheres_ctx.file_name,
                    hspheres_ctx.file_size,
                )
                return []

            logger.debug(
                'Reading %d HSpheres for segment %d from %s field local 0x%X -> %s absolute 0x%X',
                num_hspheres,
                owner_segment,
                source_context.file_name,
                hspheres_field_local_offset,
                hspheres_ctx.file_name,
                hspheres_abs,
            )

            spheres: List[HSphere] = []
            hspheres_ctx.reader.seek(hspheres_abs)
            for local_index in range(num_hspheres):
                flags = hspheres_ctx.reader.u16()
                sphere_id = hspheres_ctx.reader.u8()
                rank = hspheres_ctx.reader.u8()
                radius = hspheres_ctx.reader.u16()
                x = hspheres_ctx.reader.i16()
                y = hspheres_ctx.reader.i16()
                z = hspheres_ctx.reader.i16()
                radius_sq = hspheres_ctx.reader.u32()
                mass = hspheres_ctx.reader.u16()
                buoyancy_factor = hspheres_ctx.reader.u8()
                explosion_factor = hspheres_ctx.reader.u8()
                material_type = hspheres_ctx.reader.u8()
                pad = hspheres_ctx.reader.u8()
                damage = hspheres_ctx.reader.i16()
                spheres.append(
                    HSphere(
                        global_index=global_sphere_index_start + local_index,
                        owner_segment=owner_segment,
                        flags=flags,
                        id=sphere_id,
                        rank=rank,
                        radius=radius,
                        x=x,
                        y=y,
                        z=z,
                        radius_sq=radius_sq,
                        mass=mass,
                        buoyancy_factor=buoyancy_factor,
                        explosion_factor=explosion_factor,
                        material_type=material_type,
                        pad=pad,
                        damage=damage,
                    )
                )
            return spheres
        finally:
            br.seek(previous_offset)

    def _parse_hinfo_hboxes(
        self,
        cache: SectionContextCache,
        source_context: SectionContext,
        hinfo_absolute_offset: int,
        owner_segment: int,
        global_hbox_index_start: int,
    ) -> List[HBox]:
        if hinfo_absolute_offset <= 0 or hinfo_absolute_offset + 32 > source_context.file_size:
            return []

        br = source_context.reader
        previous_offset = br.tell()
        try:
            br.seek(hinfo_absolute_offset)

            num_hspheres = br.i32()
            _hspheres_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hboxes = br.i32()
            hboxes_field_local_offset = self._field_local_offset(source_context)
            hboxes_raw, hboxes_ctx, hboxes_abs = self._read_u32_pointer(cache, source_context)
            num_hmarkers = br.i32()
            _hmarkers_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hcapsules = br.i32()
            _hcapsules_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)

            logger.debug(
                'HInfo HBoxes for segment %d at %s:0x%X hspheres=%d hboxes=%d hmarkers=%d hcapsules=%d',
                owner_segment,
                source_context.file_name,
                hinfo_absolute_offset,
                num_hspheres,
                num_hboxes,
                num_hmarkers,
                num_hcapsules,
            )

            if num_hboxes <= 0 or hboxes_abs <= 0:
                return []

            record_size = 64
            if hboxes_abs + (num_hboxes * record_size) > hboxes_ctx.file_size:
                logger.warning(
                    'Skipping HBox read for segment %d: ptr=0x%X abs=0x%X count=%d exceeds %s file_size=0x%X',
                    owner_segment,
                    hboxes_raw,
                    hboxes_abs,
                    num_hboxes,
                    hboxes_ctx.file_name,
                    hboxes_ctx.file_size,
                )
                return []

            logger.debug(
                'Reading %d HBoxes for segment %d from %s field local 0x%X -> %s absolute 0x%X',
                num_hboxes,
                owner_segment,
                source_context.file_name,
                hboxes_field_local_offset,
                hboxes_ctx.file_name,
                hboxes_abs,
            )

            hboxes: List[HBox] = []
            hboxes_ctx.reader.seek(hboxes_abs)
            for local_index in range(num_hboxes):
                dimensions = hboxes_ctx.reader.vec4()
                position = hboxes_ctx.reader.vec4()
                quaternion = hboxes_ctx.reader.vec4()
                flags = hboxes_ctx.reader.u16()
                hbox_id = hboxes_ctx.reader.u8()
                rank = hboxes_ctx.reader.u8()
                mass = hboxes_ctx.reader.u16()
                buoyancy_factor = hboxes_ctx.reader.u8()
                explosion_factor = hboxes_ctx.reader.u8()
                material_type = hboxes_ctx.reader.u8()
                pad = hboxes_ctx.reader.u8()
                damage = hboxes_ctx.reader.i16()
                _pad2 = hboxes_ctx.reader.i32()

                hboxes.append(
                    HBox(
                        global_index=global_hbox_index_start + local_index,
                        owner_segment=owner_segment,
                        flags=flags,
                        id=hbox_id,
                        rank=rank,
                        mass=mass,
                        buoyancy_factor=buoyancy_factor,
                        explosion_factor=explosion_factor,
                        material_type=material_type,
                        pad=pad,
                        damage=damage,
                        dimensions=dimensions,
                        position=position,
                        quaternion=quaternion,
                    )
                )
            return hboxes
        finally:
            br.seek(previous_offset)


    @staticmethod
    def _normalize_vec3(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, z = vector
        length = math.sqrt((x * x) + (y * y) + (z * z))
        if length <= 1e-8:
            return (0.0, 0.0, 1.0)
        return (x / length, y / length, z / length)

    @classmethod
    def _rotate_vec3_by_quaternion(
        cls,
        quaternion: Tuple[float, float, float, float],
        vector: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        qx, qy, qz, qw = quaternion
        vx, vy, vz = vector

        tx = 2.0 * ((qy * vz) - (qz * vy))
        ty = 2.0 * ((qz * vx) - (qx * vz))
        tz = 2.0 * ((qx * vy) - (qy * vx))

        rx = vx + (qw * tx) + ((qy * tz) - (qz * ty))
        ry = vy + (qw * ty) + ((qz * tx) - (qx * tz))
        rz = vz + (qw * tz) + ((qx * ty) - (qy * tx))
        return cls._normalize_vec3((rx, ry, rz))

    def _parse_hinfo_hcapsules(
        self,
        cache: SectionContextCache,
        source_context: SectionContext,
        hinfo_absolute_offset: int,
        owner_segment: int,
        global_capsule_index_start: int,
    ) -> List[HCapsule]:
        if hinfo_absolute_offset <= 0 or hinfo_absolute_offset + 32 > source_context.file_size:
            return []

        br = source_context.reader
        previous_offset = br.tell()
        try:
            br.seek(hinfo_absolute_offset)

            num_hspheres = br.i32()
            _hspheres_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hboxes = br.i32()
            _hboxes_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hmarkers = br.i32()
            _hmarkers_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hcapsules = br.i32()
            hcapsules_field_local_offset = self._field_local_offset(source_context)
            hcapsules_raw, hcapsules_ctx, hcapsules_abs = self._read_u32_pointer(cache, source_context)

            logger.debug(
                'HInfo HCapsules for segment %d at %s:0x%X hspheres=%d hboxes=%d hmarkers=%d hcapsules=%d',
                owner_segment,
                source_context.file_name,
                hinfo_absolute_offset,
                num_hspheres,
                num_hboxes,
                num_hmarkers,
                num_hcapsules,
            )

            if num_hcapsules <= 0 or hcapsules_abs <= 0:
                return []

            record_size = 40
            if hcapsules_abs + (num_hcapsules * record_size) > hcapsules_ctx.file_size:
                logger.warning(
                    'Skipping HCapsule read for segment %d: ptr=0x%X abs=0x%X count=%d exceeds %s file_size=0x%X',
                    owner_segment,
                    hcapsules_raw,
                    hcapsules_abs,
                    num_hcapsules,
                    hcapsules_ctx.file_name,
                    hcapsules_ctx.file_size,
                )
                return []

            logger.debug(
                'Reading %d HCapsules for segment %d from %s field local 0x%X -> %s absolute 0x%X',
                num_hcapsules,
                owner_segment,
                source_context.file_name,
                hcapsules_field_local_offset,
                hcapsules_ctx.file_name,
                hcapsules_abs,
            )

            capsules: List[HCapsule] = []
            hcapsules_ctx.reader.seek(hcapsules_abs)
            for local_index in range(num_hcapsules):
                position = hcapsules_ctx.reader.vec4()
                quaternion = hcapsules_ctx.reader.vec4()
                flags = hcapsules_ctx.reader.u16()
                capsule_id = hcapsules_ctx.reader.u8()
                rank = hcapsules_ctx.reader.u8()
                radius = hcapsules_ctx.reader.u16()
                length = hcapsules_ctx.reader.u16()
                mass = hcapsules_ctx.reader.u16()
                buoyancy_factor = hcapsules_ctx.reader.u8()
                explosion_factor = hcapsules_ctx.reader.u8()
                material_type = hcapsules_ctx.reader.u8()
                pad = hcapsules_ctx.reader.u8()
                damage = hcapsules_ctx.reader.i16()

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
                        global_index=global_capsule_index_start + local_index,
                        owner_segment=owner_segment,
                        flags=flags,
                        id=capsule_id,
                        rank=rank,
                        radius=radius,
                        length=length,
                        mass=mass,
                        buoyancy_factor=buoyancy_factor,
                        explosion_factor=explosion_factor,
                        material_type=material_type,
                        pad=pad,
                        damage=damage,
                        position=position,
                        quaternion=quaternion,
                        start=start,
                        end=end,
                    )
                )
            return capsules
        finally:
            br.seek(previous_offset)


    def _parse_hinfo_hmarkers(
        self,
        cache: SectionContextCache,
        source_context: SectionContext,
        hinfo_absolute_offset: int,
        owner_segment: int,
        global_marker_index_start: int,
    ) -> List[HMarker]:
        if hinfo_absolute_offset <= 0 or hinfo_absolute_offset + 32 > source_context.file_size:
            return []

        br = source_context.reader
        previous_offset = br.tell()
        try:
            br.seek(hinfo_absolute_offset)

            num_hspheres = br.i32()
            hspheres_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hboxes = br.i32()
            hboxes_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)
            num_hmarkers = br.i32()
            hmarkers_field_local_offset = self._field_local_offset(source_context)
            hmarkers_raw, hmarkers_ctx, hmarkers_abs = self._read_u32_pointer(cache, source_context)
            num_hcapsules = br.i32()
            hcapsules_raw, _ctx, _abs = self._read_u32_pointer(cache, source_context)

            logger.debug(
                'HInfo for segment %d at %s:0x%X hspheres=%d hboxes=%d hmarkers=%d hcapsules=%d',
                owner_segment,
                source_context.file_name,
                hinfo_absolute_offset,
                num_hspheres,
                num_hboxes,
                num_hmarkers,
                num_hcapsules,
            )

            if num_hmarkers <= 0 or hmarkers_abs <= 0:
                return []

            if hmarkers_abs + (num_hmarkers * 32) > hmarkers_ctx.file_size:
                logger.warning(
                    'Skipping HMarker read for segment %d: ptr=0x%X abs=0x%X count=%d exceeds %s file_size=0x%X',
                    owner_segment,
                    hmarkers_raw,
                    hmarkers_abs,
                    num_hmarkers,
                    hmarkers_ctx.file_name,
                    hmarkers_ctx.file_size,
                )
                return []

            logger.debug(
                'Reading %d HMarkers for segment %d from %s field local 0x%X -> %s absolute 0x%X',
                num_hmarkers,
                owner_segment,
                source_context.file_name,
                hmarkers_field_local_offset,
                hmarkers_ctx.file_name,
                hmarkers_abs,
            )

            markers: List[HMarker] = []
            hmarkers_ctx.reader.seek(hmarkers_abs)
            for local_index in range(num_hmarkers):
                bone = hmarkers_ctx.reader.i32()
                index = hmarkers_ctx.reader.i32()
                px = hmarkers_ctx.reader.f32()
                py = hmarkers_ctx.reader.f32()
                pz = hmarkers_ctx.reader.f32()
                rx = hmarkers_ctx.reader.f32()
                ry = hmarkers_ctx.reader.f32()
                rz = hmarkers_ctx.reader.f32()
                markers.append(
                    HMarker(
                        global_index=global_marker_index_start + local_index,
                        owner_segment=owner_segment,
                        bone=bone,
                        index=index,
                        position=(px, py, pz),
                        rotation=(rx, ry, rz),
                    )
                )
            return markers
        finally:
            br.seek(previous_offset)



    def _read_u32_pointer_at(self, cache: SectionContextCache, context: SectionContext, field_local_offset: int) -> Tuple[int, SectionContext, int]:
        previous = context.reader.tell()
        try:
            absolute = context.data_start + field_local_offset
            if absolute + 4 > context.file_size:
                return 0, context, 0
            context.reader.seek(absolute)
            raw_value = context.reader.u32()
            target_context, absolute_offset = resolve_pointer(cache, context, field_local_offset, raw_value)
            return raw_value, target_context, absolute_offset
        finally:
            context.reader.seek(previous)

    @staticmethod
    def _is_valid_abs(context: SectionContext, absolute_offset: int, size: int = 1) -> bool:
        return absolute_offset >= context.data_start and (absolute_offset + max(size, 0)) <= min(context.data_end, context.file_size)

    @staticmethod
    def _normalize_markup_game(cdc_render_data_id: int) -> str:
        return 'legend' if int(cdc_render_data_id) != 0 else 'anniversary'

    def _resolve_pointer_at_absolute(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int):
        if not self._is_valid_abs(context, absolute_offset, 4):
            return 0, context, 0
        previous = context.reader.tell()
        try:
            context.reader.seek(absolute_offset)
            raw_value = context.reader.u32()
        finally:
            context.reader.seek(previous)
        field_local_offset = absolute_offset - context.data_start
        if field_local_offset < 0:
            return raw_value, context, raw_value
        target_context, target_abs = resolve_pointer(cache, context, field_local_offset, raw_value)
        return raw_value, target_context, target_abs

    def _read_markup_polyline_section(self, context: SectionContext, absolute_offset: int) -> List[Tuple[float, float, float, float]]:
        polyline: List[Tuple[float, float, float, float]] = []
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 16):
            return polyline
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            num_points = br.i32()
            if num_points <= 0 or num_points > 8192:
                return polyline
            data_start = absolute_offset + 16
            if not self._is_valid_abs(context, data_start, num_points * 16):
                return polyline
            br.seek(data_start)
            for _ in range(num_points):
                polyline.append((float(br.f32()), float(br.f32()), float(br.f32()), float(br.f32())))
            return polyline
        finally:
            br.seek(previous)

    def _parse_section_markups(
        self,
        cache: SectionContextCache,
        list_ctx: SectionContext,
        list_abs: int,
        count: int,
        cdc_render_data_id: int,
    ) -> List[ModelMarkupData]:
        markups: List[ModelMarkupData] = []
        if not self.import_markups or count <= 0 or list_abs <= 0:
            return markups
        game = self._normalize_markup_game(cdc_render_data_id)
        entry_size = 76 if game == 'anniversary' else 48
        requested_count = max(0, min(int(count), 65535))
        parse_count = 0
        for index in range(requested_count):
            if not self._is_valid_abs(list_ctx, list_abs + (index * entry_size), entry_size):
                break
            parse_count += 1
        if parse_count < requested_count:
            logger.warning(
                'Model MarkUp list at 0x%X in %s has count=%d but only %d complete entries fit; importing the complete entries',
                list_abs,
                list_ctx.file_name,
                count,
                parse_count,
            )
        if parse_count <= 0:
            return markups

        br = list_ctx.reader
        previous = br.tell()
        try:
            for index in range(parse_count):
                entry_abs = list_abs + (index * entry_size)
                br.seek(entry_abs)
                override_movement_camera = br.i32()
                dtp_camera_data_id = br.i32()
                dtp_markup_data_id = br.i32()
                animated_segment = 0
                camera_antic = None
                if game == 'anniversary':
                    animated_segment = br.i32()
                    use_antic_camera = br.i32()
                    antic_ids = tuple(int(br.i32()) for _ in range(5))
                    camera_antic = ModelCameraAnticData(
                        use_antic_camera=int(use_antic_camera),
                        new_dtp_camera_antic_data_id=antic_ids,
                    )
                flags = br.u32()
                intro_id = br.i16()
                markup_id = br.i16()
                px = br.f32(); py = br.f32(); pz = br.f32()
                bbox = tuple(int(br.i16()) for _ in range(6))
                polyline_field_abs = br.tell()
                polyline_raw, polyline_ctx, polyline_abs = self._resolve_pointer_at_absolute(cache, list_ctx, polyline_field_abs)
                polyline_target = polyline_abs if polyline_abs > 0 else polyline_raw
                polyline = self._read_markup_polyline_section(polyline_ctx, polyline_target)
                markups.append(ModelMarkupData(
                    index=index,
                    game=game,
                    override_movement_camera=int(override_movement_camera),
                    dtp_camera_data_id=int(dtp_camera_data_id),
                    dtp_markup_data_id=int(dtp_markup_data_id),
                    animated_segment=int(animated_segment),
                    camera_antic=camera_antic,
                    flags=int(flags),
                    intro_id=int(intro_id),
                    markup_id=int(markup_id),
                    position=(float(px), float(py), float(pz)),
                    bbox=bbox,
                    polyline_offset=int(polyline_target),
                    polyline=polyline,
                ))
        finally:
            br.seek(previous)
        return markups

    def _parse_targets(
        self,
        target_ctx: SectionContext,
        target_abs: int,
        num_targets: int,
    ) -> List[Target]:
        if num_targets <= 0 or target_abs <= 0:
            return []

        entry_size = 28
        data_end = target_abs + (num_targets * entry_size)
        if data_end > target_ctx.file_size:
            logger.warning(
                'Skipping targetList read: abs=0x%X count=%d would end at 0x%X beyond %s file_size=0x%X',
                target_abs,
                num_targets,
                data_end,
                target_ctx.file_name,
                target_ctx.file_size,
            )
            return []

        previous_offset = target_ctx.reader.tell()
        try:
            target_ctx.reader.seek(target_abs)
            targets: List[Target] = []
            for index in range(num_targets):
                segment = target_ctx.reader.u16()
                flags = target_ctx.reader.u16()
                px = target_ctx.reader.f32()
                py = target_ctx.reader.f32()
                pz = target_ctx.reader.f32()
                rx = target_ctx.reader.f32()
                ry = target_ctx.reader.f32()
                rz = target_ctx.reader.f32()
                unique_id = target_ctx.reader.u32()
                targets.append(
                    Target(
                        global_index=index,
                        segment=segment,
                        flags=flags,
                        position=(px, py, pz),
                        rotation=(rx, ry, rz),
                        unique_id=unique_id,
                    )
                )
            logger.debug('Read %d targets from %s absolute 0x%X', len(targets), target_ctx.file_name, target_abs)
            return targets
        finally:
            target_ctx.reader.seek(previous_offset)

    def _parse_model(self, cache: SectionContextCache, context: SectionContext) -> ModelData:
        br = context.reader
        br.seek(context.data_start)

        version = br.i32()
        num_segments = br.i32()
        num_virt_segments = br.i32()
        segment_list_raw, segment_list_ctx, segment_list_abs = self._read_u32_pointer(cache, context)

        logger.debug(
            'Header values in %s: version=%d num_segments=%d num_virt_segments=%d segment_list=0x%X',
            context.file_name,
            version,
            num_segments,
            num_virt_segments,
            segment_list_raw,
        )

        return_offset = br.tell()
        segments: List[Segment] = []
        virt_segments: List[VirtSegment] = []
        hmarkers: List[HMarker] = []
        hspheres: List[HSphere] = []
        hboxes: List[HBox] = []
        hcapsules: List[HCapsule] = []
        targets: List[Target] = []
        if num_segments > 0 and segment_list_abs:
            logger.debug(
                'Reading %d Segment records and %d VirtSegment records from %s local 0x%X',
                num_segments,
                num_virt_segments,
                segment_list_ctx.file_name,
                segment_list_raw,
            )
            segment_list_ctx.reader.seek(segment_list_abs)
            for index in range(num_segments):
                segment_record_start = segment_list_ctx.reader.tell()
                segment = self._parse_segment(segment_list_ctx, index)
                segments.append(segment)
                if self.import_hinfo:
                    if segment.hinfo:
                        hinfo_field_local_offset = (segment_record_start - segment_list_ctx.data_start) + (self._segment_record_size() - 4)
                        hinfo_ctx, hinfo_abs = resolve_pointer(cache, segment_list_ctx, hinfo_field_local_offset, segment.hinfo)
                        if hinfo_abs:
                            hspheres.extend(
                                self._parse_hinfo_hspheres(
                                    cache,
                                    hinfo_ctx,
                                    hinfo_abs,
                                    owner_segment=index,
                                    global_sphere_index_start=len(hspheres),
                                )
                            )
                            hboxes.extend(
                                self._parse_hinfo_hboxes(
                                    cache,
                                    hinfo_ctx,
                                    hinfo_abs,
                                    owner_segment=index,
                                    global_hbox_index_start=len(hboxes),
                                )
                            )
                            hmarkers.extend(
                                self._parse_hinfo_hmarkers(
                                    cache,
                                    hinfo_ctx,
                                    hinfo_abs,
                                    owner_segment=index,
                                    global_marker_index_start=len(hmarkers),
                                )
                            )
                            hcapsules.extend(
                                self._parse_hinfo_hcapsules(
                                    cache,
                                    hinfo_ctx,
                                    hinfo_abs,
                                    owner_segment=index,
                                    global_capsule_index_start=len(hcapsules),
                                )
                            )
                    elif segment.hinfo == 0:
                        hinfo_field_local_offset = (segment_record_start - segment_list_ctx.data_start) + (self._segment_record_size() - 4)
                        relocation = segment_list_ctx.section_info.relocations_by_offset.get(hinfo_field_local_offset)
                        if relocation is not None:
                            hinfo_ctx, hinfo_abs = resolve_pointer(cache, segment_list_ctx, hinfo_field_local_offset, segment.hinfo)
                            if hinfo_abs:
                                hspheres.extend(
                                    self._parse_hinfo_hspheres(
                                        cache,
                                        hinfo_ctx,
                                        hinfo_abs,
                                        owner_segment=index,
                                        global_sphere_index_start=len(hspheres),
                                    )
                                )
                                hboxes.extend(
                                    self._parse_hinfo_hboxes(
                                        cache,
                                        hinfo_ctx,
                                        hinfo_abs,
                                        owner_segment=index,
                                        global_hbox_index_start=len(hboxes),
                                    )
                                )
                                hmarkers.extend(
                                    self._parse_hinfo_hmarkers(
                                        cache,
                                        hinfo_ctx,
                                        hinfo_abs,
                                        owner_segment=index,
                                        global_marker_index_start=len(hmarkers),
                                    )
                                )
                                hcapsules.extend(
                                    self._parse_hinfo_hcapsules(
                                        cache,
                                        hinfo_ctx,
                                        hinfo_abs,
                                        owner_segment=index,
                                        global_capsule_index_start=len(hcapsules),
                                    )
                                )
                
            if self.parse_segments_only and self._is_gamecube_layout():
                logger.debug('Skipping Gamecube virt segment parsing in segments-only mode')
            else:
                for virt_index in range(num_virt_segments):
                    virt_segments.append(self._parse_virt_segment(segment_list_ctx, virt_index))

        if self.parse_segments_only:
            return make_segments_only_model_data(
                version=version,
                segments=segments,
                virt_segments=virt_segments,
                strips=strips,
                hmarkers=hmarkers,
                hspheres=hspheres,
                hboxes=hboxes,
                hcapsules=hcapsules,
                uv_format='gamecube' if self._is_gamecube_layout() else 'pc',
            )

        if self._is_gamecube_layout():
            max_rad = 0.0
            max_rad_sq = 0.0
            bone_mirror_entries: List[BoneMirrorEntry] = []

            previous = br.tell()
            try:
                if context.data_start + 0x40 <= context.data_end:
                    br.seek(context.data_start + 0x3C)
                    max_rad = br.f32()
                    max_rad_sq = br.f32()

                bone_mirror_raw, bone_mirror_ctx, bone_mirror_abs = self._read_u32_pointer_at(cache, context, 0x7C)
                if bone_mirror_abs:
                    bone_mirror_entries = self._parse_bone_mirror_entries(bone_mirror_ctx, bone_mirror_abs)

                if context.data_start + 0x90 <= context.data_end:
                    br.seek(context.data_start + 0x8C)
                    num_targets = br.i32()
                    target_list_raw, target_list_ctx, target_list_abs = self._read_u32_pointer(cache, context)
                    if num_targets > 0 and target_list_abs:
                        targets.extend(self._parse_targets(target_list_ctx, target_list_abs, num_targets))

            finally:
                br.seek(previous)

            return self._parse_gamecube_geometry(
                cache,
                context,
                version=version,
                segments=segments,
                virt_segments=virt_segments,
                hmarkers=hmarkers,
                hspheres=hspheres,
                hboxes=hboxes,
                hcapsules=hcapsules,
                targets=targets,
                max_rad=max_rad,
                max_rad_sq=max_rad_sq,
                bone_mirror_entries=bone_mirror_entries,
            )

        br.seek(return_offset)
        model_scale = br.vec4()

        num_vertices = br.i32()
        vertex_list_raw, vertex_list_ctx, vertex_list_abs = self._read_u32_pointer(cache, context)
        return_offset = br.tell()
        vertices: List[MVertex] = []
        if num_vertices > 0 and vertex_list_abs:
            logger.debug(
                'Reading %d vertices from %s local 0x%X',
                num_vertices,
                vertex_list_ctx.file_name,
                vertex_list_raw,
            )
            vertex_list_ctx.reader.seek(vertex_list_abs)
            for index in range(num_vertices):
                vertices.append(self._parse_vertex(vertex_list_ctx, index))

        br.seek(return_offset)
        num_normals = br.i32()
        normal_list_raw, _normal_ctx, _normal_abs = self._read_u32_pointer(cache, context)
        num_faces = br.i32()
        face_list_raw, face_list_ctx, face_list_abs = self._read_u32_pointer(cache, context)
        return_offset = br.tell()
        logger.debug(
            'num_normals=%d normal_list=0x%X num_faces=%d face_list=0x%X',
            num_normals,
            normal_list_raw,
            num_faces,
            face_list_raw,
        )

        faces = []
        if num_faces > 0 and face_list_abs:
            logger.debug(
                'Reading %d faces from %s local 0x%X',
                num_faces,
                face_list_ctx.file_name,
                face_list_raw,
            )
            face_list_ctx.reader.seek(face_list_abs)
            for index in range(num_faces):
                faces.append(self._parse_face(face_list_ctx, index))

        br.seek(return_offset)
        obsolete_ani_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        max_rad = br.f32()
        max_rad_sq = br.f32()
        obsolete_start_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        obsolete_end_textures_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        animated_list_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        animated_info_raw, _ctx, _abs = self._read_u32_pointer(cache, context)
        scroll_info_raw, scroll_info_ctx, scroll_info_abs = self._read_u32_pointer(cache, context)
        texture_strip_info_raw, texture_strip_ctx, texture_strip_abs = self._read_u32_pointer(cache, context)

        env_mapped_vertices_field_local_offset = None
        env_mapped_vertices_raw = 0
        env_mapped_vertices_ctx = context
        env_mapped_vertices_abs = 0
        eye_ref_env_mapped_vertices_field_local_offset = None
        eye_ref_env_mapped_vertices_raw = 0
        eye_ref_env_mapped_vertices_ctx = context
        eye_ref_env_mapped_vertices_abs = 0
        material_vertex_colors_field_local_offset = None
        material_vertex_colors_raw = 0
        material_vertex_colors_ctx = context
        material_vertex_colors_abs = 0

        spectral_vertex_colors_raw = 0
        spectral_vertex_colors_ctx = context
        spectral_vertex_colors_abs = 0
        pn_shadow_faces = 0
        pn_shadow_edges = 0
        bone_mirror_data_raw = 0
        bone_mirror_data_ctx = context
        bone_mirror_data_abs = 0
        drawgroup_center_list_raw = 0
        drawgroup_center_list_ctx = context
        drawgroup_center_list_abs = 0
        gamecube_positions_raw = 0
        gamecube_uv_raw = 0
        gamecube_normal_data_raw = 0
        markuplist_raw = 0
        markuplist_ctx = context
        markuplist_abs = 0
        gamecube_cdc_render_model_data_raw = 0

        if self._is_gamecube_layout():
            if br.tell() + 4 <= context.data_end:
                gamecube_positions_raw, _gamecube_positions_ctx, _gamecube_positions_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                gamecube_uv_raw, _gamecube_uv_ctx, _gamecube_uv_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                material_vertex_colors_field_local_offset = self._field_local_offset(context)
                material_vertex_colors_raw, material_vertex_colors_ctx, material_vertex_colors_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                gamecube_normal_data_raw = br.i32()
            if br.tell() + 4 <= context.data_end:
                env_mapped_vertices_field_local_offset = self._field_local_offset(context)
                env_mapped_vertices_raw, env_mapped_vertices_ctx, env_mapped_vertices_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                eye_ref_env_mapped_vertices_field_local_offset = self._field_local_offset(context)
                eye_ref_env_mapped_vertices_raw, eye_ref_env_mapped_vertices_ctx, eye_ref_env_mapped_vertices_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                bone_mirror_data_raw, bone_mirror_data_ctx, bone_mirror_data_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                drawgroup_center_list_raw, drawgroup_center_list_ctx, drawgroup_center_list_abs = self._read_u32_pointer(cache, context)
        else:
            if br.tell() + 4 <= context.data_end:
                env_mapped_vertices_field_local_offset = self._field_local_offset(context)
                env_mapped_vertices_raw, env_mapped_vertices_ctx, env_mapped_vertices_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                eye_ref_env_mapped_vertices_field_local_offset = self._field_local_offset(context)
                eye_ref_env_mapped_vertices_raw, eye_ref_env_mapped_vertices_ctx, eye_ref_env_mapped_vertices_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                material_vertex_colors_field_local_offset = self._field_local_offset(context)
                material_vertex_colors_raw, material_vertex_colors_ctx, material_vertex_colors_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                spectral_vertex_colors_raw, spectral_vertex_colors_ctx, spectral_vertex_colors_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 2 <= context.data_end:
                pn_shadow_faces = br.i16()
            if br.tell() + 2 <= context.data_end:
                br.u16()
            if br.tell() + 2 <= context.data_end:
                pn_shadow_edges = br.i16()
            if br.tell() + 2 <= context.data_end:
                br.u16()
            if br.tell() + 4 <= context.data_end:
                bone_mirror_data_raw, bone_mirror_data_ctx, bone_mirror_data_abs = self._read_u32_pointer(cache, context)
            if br.tell() + 4 <= context.data_end:
                drawgroup_center_list_raw, drawgroup_center_list_ctx, drawgroup_center_list_abs = self._read_u32_pointer(cache, context)

        logger.debug(
            'texture_strip_info=0x%X env_mapped_vertices=0x%X eye_ref_env_mapped_vertices=0x%X material_vertex_colors=0x%X max_rad=%s max_rad_sq=%s obsoleteAni=0x%X startTex=0x%X endTex=0x%X animatedList=0x%X animatedInfo=0x%X scrollInfo=0x%X',
            texture_strip_info_raw,
            env_mapped_vertices_raw,
            eye_ref_env_mapped_vertices_raw,
            material_vertex_colors_raw,
            max_rad,
            max_rad_sq,
            obsolete_ani_textures_raw,
            obsolete_start_textures_raw,
            obsolete_end_textures_raw,
            animated_list_info_raw,
            animated_info_raw,
            scroll_info_raw,
        )
        if self._is_gamecube_layout():
            logger.debug(
                'positionsOffset=0x%X uvOffset=0x%X vertexColorsOffset=0x%X normalDataOffset=0x%X boneMirrorData=0x%X drawgroupCenterList=0x%X',
                gamecube_positions_raw,
                gamecube_uv_raw,
                material_vertex_colors_raw,
                gamecube_normal_data_raw,
                bone_mirror_data_raw,
                drawgroup_center_list_raw,
            )
        else:
            logger.debug(
                'spectralVertexColors=0x%X pnShadowFaces=%d pnShadowEdges=%d boneMirrorData=0x%X drawgroupCenterList=0x%X',
                spectral_vertex_colors_raw,
                pn_shadow_faces,
                pn_shadow_edges,
                bone_mirror_data_raw,
                drawgroup_center_list_raw,
            )

        if env_mapped_vertices_field_local_offset is not None:
            if env_mapped_vertices_abs:
                logger.debug(
                    'Found envMappedVertices pointer field at %s local 0x%X absolute 0x%X -> %s absolute 0x%X (raw=0x%X)',
                    context.file_name,
                    env_mapped_vertices_field_local_offset,
                    context.data_start + env_mapped_vertices_field_local_offset,
                    env_mapped_vertices_ctx.file_name,
                    env_mapped_vertices_abs,
                    env_mapped_vertices_raw,
                )
            else:
                logger.debug(
                    'envMappedVertices pointer field at %s local 0x%X absolute 0x%X is null',
                    context.file_name,
                    env_mapped_vertices_field_local_offset,
                    context.data_start + env_mapped_vertices_field_local_offset,
                )
        if eye_ref_env_mapped_vertices_field_local_offset is not None:
            if eye_ref_env_mapped_vertices_abs:
                logger.debug(
                    'Found eyeRefEnvMappedVertices pointer field at %s local 0x%X absolute 0x%X -> %s absolute 0x%X (raw=0x%X)',
                    context.file_name,
                    eye_ref_env_mapped_vertices_field_local_offset,
                    context.data_start + eye_ref_env_mapped_vertices_field_local_offset,
                    eye_ref_env_mapped_vertices_ctx.file_name,
                    eye_ref_env_mapped_vertices_abs,
                    eye_ref_env_mapped_vertices_raw,
                )
            else:
                logger.debug(
                    'eyeRefEnvMappedVertices pointer field at %s local 0x%X absolute 0x%X is null',
                    context.file_name,
                    eye_ref_env_mapped_vertices_field_local_offset,
                    context.data_start + eye_ref_env_mapped_vertices_field_local_offset,
                )
        if material_vertex_colors_field_local_offset is not None and material_vertex_colors_abs:
            logger.debug(
                'materialVertexColors resolved from field local 0x%X absolute 0x%X to %s absolute 0x%X (raw=0x%X)',
                material_vertex_colors_field_local_offset,
                context.data_start + material_vertex_colors_field_local_offset,
                material_vertex_colors_ctx.file_name,
                material_vertex_colors_abs,
                material_vertex_colors_raw,
            )

        stream_resume_offset = br.tell()
        strips = self._parse_texture_strip_chain(
            cache,
            context,
            texture_strip_info_raw,
            texture_strip_ctx,
            texture_strip_abs,
        )
        if scroll_info_abs:
            self._parse_model_scroll_info(cache, scroll_info_ctx, scroll_info_abs, strips)
        br.seek(stream_resume_offset)

        env_mapped_face_indices = []
        if env_mapped_vertices_abs:
            env_mapped_face_indices.extend(
                self._parse_index_list(
                    env_mapped_vertices_ctx,
                    env_mapped_vertices_abs,
                    env_mapped_vertices_raw,
                    'envMappedVertices',
                )
            )
        br.seek(stream_resume_offset)

        eye_ref_env_mapped_face_indices = []
        if eye_ref_env_mapped_vertices_abs:
            eye_ref_env_mapped_face_indices.extend(
                self._parse_index_list(
                    eye_ref_env_mapped_vertices_ctx,
                    eye_ref_env_mapped_vertices_abs,
                    eye_ref_env_mapped_vertices_raw,
                    'eyeRefEnvMappedVertices',
                )
            )
        br.seek(stream_resume_offset)

        if br.tell() + 4 <= context.data_end:
            num_markups = br.i32()
        else:
            num_markups = 0
        if br.tell() + 4 <= context.data_end:
            if self._is_gamecube_layout():
                markuplist_raw, markuplist_ctx, markuplist_abs = self._read_u32_pointer(cache, context)
            else:
                markuplist_raw, markuplist_ctx, markuplist_abs = self._read_u32_pointer(cache, context)
        if br.tell() + 4 <= context.data_end:
            num_targets = br.i32()
        else:
            num_targets = 0
        if br.tell() + 4 <= context.data_end:
            target_list_raw, target_list_ctx, target_list_abs = self._read_u32_pointer(cache, context)
        else:
            target_list_raw, target_list_ctx, target_list_abs = 0, context, 0
        if br.tell() + 4 <= context.data_end:
            cdc_render_data_id = br.u32()
        else:
            cdc_render_data_id = 0
        if self._is_gamecube_layout() and br.tell() + 4 <= context.data_end:
            gamecube_cdc_render_model_data_raw = br.u32()

        logger.debug(
            'numMarkUps=%d numTargets=%d targetList=0x%X cdcRenderDataID=0x%X',
            num_markups,
            num_targets,
            target_list_raw,
            cdc_render_data_id,
        )

        markups = []
        if num_markups > 0 and markuplist_abs:
            markups = self._parse_section_markups(cache, markuplist_ctx, markuplist_abs, num_markups, cdc_render_data_id)

        if num_targets > 0 and target_list_abs:
            targets.extend(self._parse_targets(target_list_ctx, target_list_abs, num_targets))

        vertex_colors = None
        if num_vertices > 0 and material_vertex_colors_abs:
            color_data_end = material_vertex_colors_abs + (num_vertices * 4)
            if color_data_end <= material_vertex_colors_ctx.file_size:
                if color_data_end > material_vertex_colors_ctx.data_end:
                    logger.warning(
                        'materialVertexColors read extends beyond declared data range but remains within file bounds: ptr=0x%X abs=0x%X count=%d end=0x%X data_end=0x%X file_size=0x%X',
                        material_vertex_colors_raw,
                        material_vertex_colors_abs,
                        num_vertices,
                        color_data_end,
                        material_vertex_colors_ctx.data_end,
                        material_vertex_colors_ctx.file_size,
                    )
                logger.debug(
                    'Reading %d vertex colors from %s local 0x%X',
                    num_vertices,
                    material_vertex_colors_ctx.file_name,
                    material_vertex_colors_raw,
                )
                material_vertex_colors_ctx.reader.seek(material_vertex_colors_abs)
                vertex_colors = self._parse_vertex_colors(material_vertex_colors_ctx, num_vertices)
            else:
                logger.warning(
                    'Skipping materialVertexColors read: ptr=0x%X abs=0x%X count=%d would end at 0x%X beyond %s file_size=0x%X (data_end=0x%X)',
                    material_vertex_colors_raw,
                    material_vertex_colors_abs,
                    num_vertices,
                    color_data_end,
                    material_vertex_colors_ctx.file_name,
                    material_vertex_colors_ctx.file_size,
                    material_vertex_colors_ctx.data_end,
                )

        bone_mirror_entries = []
        if bone_mirror_data_abs:
            bone_mirror_entries = self._parse_bone_mirror_entries(bone_mirror_data_ctx, bone_mirror_data_abs)

        return ModelData(
            version=version,
            model_scale=model_scale,
            segments=segments,
            virt_segments=virt_segments,
            vertices=vertices,
            faces=faces,
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
            uv_format='pc',
        )
