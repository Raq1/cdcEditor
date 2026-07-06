from __future__ import annotations

import math
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.log import logger
from ...core.material_ui import decode_tpage_flags, get_material_tpageid
from ...core.model_types import MVertex, ModelData, TextureStrip
from ..pc.tr7ae_model_export import (
    TRLAUModelExporter,
    _MODEL_FLAT_SHADED_WEIGHT_STEPS,
    _decode_pc_uv_component,
    _OriginalModelLayout,
    _Relocation,
    _SectionList,
    _StandaloneSectionBuffer,
    _section_index_from_filename,
    _should_ignore_model_export_object,
    _write_standalone_section_file,
    _iter_collection_objects,
)
from .texture import (
    build_ps2_indexed8_pcd_body,
    build_ps2_rgba32_pcd_body,
)
from ..common.section import SectionContextCache, resolve_pointer


_PS2_MODEL_VERSION_MAGIC = 79823955
_PS2_MODEL_HEADER_SIZE = 0x90
_PS2_BONE_MIRROR_FIELD_OFFSET = 0x6C
_PS2_VIF_UNPACK_COUNT_LIMIT = 255
_PS2_SAFE_PRIMITIVE_VERTEX_LIMIT = 54
_PS2_MAX_PALETTE_ENTRIES = 7
_PS2_MAX_TEXTURE_DIMENSION = 512
_PS2_TPAGE_ADDRESS_SHIFT = 25
_PS2_TPAGE_ADDRESS_MASK = 0x3 << _PS2_TPAGE_ADDRESS_SHIFT
_PS2_TPAGE_ADDRESS_DEFAULT = 0
_PS2_TPAGE_ADDRESS_REPEAT = 1
_PS2_COMPACT_SEGMENT_SIZE = 32
_PS2_COMPACT_VIRT_SEGMENT_SIZE = 32


@dataclass(slots=True)
class _PS2PrimitiveRecord:
    vertex_count: int
    qword_count: int
    palette: List[int]
    flags: int
    tpageid: int
    sort_key: float
    payload: bytes
    unknown10: bytes = b'\x00\x00'


@dataclass(slots=True)
class _PS2TemplatePrimitiveRecord:
    index: int
    vertex_count: int
    qword_count: int
    palette_count: int
    flags: int
    tpageid: int
    sort_key: float
    palette: List[int]
    unknown10: bytes


def _clamp_i16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def _clamp_i8(value: int) -> int:
    return max(-128, min(127, int(value)))


def _encode_ps2_uv_component(value: float) -> int:
    try:
        scaled = int(round(float(value) * 4096.0))
    except Exception:
        scaled = 0
    return _clamp_i16(scaled)


def _align_bytes(data: bytearray, alignment: int) -> None:
    pad = (-len(data)) % max(1, int(alignment))
    if pad:
        data.extend(b'\x00' * pad)


class TRLAUPS2ModelExporter(TRLAUModelExporter):

    platform_name = 'PS2'

    @staticmethod
    def _normalise_model_platform(value: object) -> str:
        platform = str(value or '').strip().lower()
        aliases = {
            'xbox 360': 'xbox360',
            'xenon': 'xbox360',
            'game cube': 'gamecube',
            'gc': 'gamecube',
            'nintendo': 'gamecube',
            'tr8': 'underworld',
        }
        return aliases.get(platform, platform)

    @classmethod
    def collection_is_ps2_model(cls, collection) -> bool:
        if collection is None:
            return False

        explicit_platforms: List[str] = []
        objects = list(_iter_collection_objects(collection))
        for obj in objects:
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not bool(getattr(obj, 'trlau_is_model_empty', False)):
                continue
            try:
                platform_value = getattr(obj, 'trlau_model_platform', None)
                if platform_value not in (None, ''):
                    explicit_platforms.append(cls._normalise_model_platform(platform_value))
                elif 'trlau_model_platform_id' in obj:
                    explicit_platforms.append(cls._normalise_model_platform(obj.get('trlau_model_platform_id')))
            except Exception:
                pass

        if explicit_platforms:
            return any(platform == 'ps2' for platform in explicit_platforms)

        for obj in objects:
            if _should_ignore_model_export_object(obj):
                continue
            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            for material in getattr(mesh, 'materials', []) or []:
                if material is None:
                    continue
                try:
                    if cls._normalise_model_platform(getattr(material, 'trlau_ui_material_platform', '')) == 'ps2':
                        return True
                except Exception:
                    pass
        return False

    def _build_model_data_from_scene(self, context, model_root) -> ModelData:
        model = super()._build_model_data_from_scene(context, model_root)
        try:
            model.uv_format = 'ps2'
        except Exception:
            pass
        return model

    @staticmethod
    def _ps2_max_virt_segments_for_segments(segments: Sequence[object]) -> int:
        return max(0, 256 - len(segments or []))

    def _collect_mesh_geometry(self, context, model_root, arm_obj, mesh_objects, segments, pivot_by_segment, model_scale):
        vertices, faces, strips, vertex_colors, virt_segments, env_indices, eye_ref_indices, vertex_weights = super()._collect_mesh_geometry(
            context,
            model_root,
            arm_obj,
            mesh_objects,
            segments,
            pivot_by_segment,
            model_scale,
        )

        max_virt_segments = self._ps2_max_virt_segments_for_segments(segments)
        if len(virt_segments or []) <= max_virt_segments:
            return vertices, faces, strips, vertex_colors, virt_segments, env_indices, eye_ref_indices, vertex_weights

        previous_weights = list(vertex_weights or [])
        optimized_weights, message = self._optimize_ps2_virt_segment_weights(
            model_root,
            previous_weights,
            segments,
            max_virt_segments,
        )
        self._rebase_vertex_positions_for_primary_changes(
            vertices,
            previous_weights,
            optimized_weights,
            pivot_by_segment,
            model_scale,
        )
        virt_segments = self._build_virt_segments(vertices, optimized_weights, segments, pivot_by_segment, model_scale)
        vertex_weights = optimized_weights

        if message:
            logger.warning(message)
            self.warnings.append(message)

        return vertices, faces, strips, vertex_colors, virt_segments, env_indices, eye_ref_indices, vertex_weights

    def _optimize_ps2_virt_segment_weights(
        self,
        model_root,
        vertex_weights: Sequence[tuple[int, int, float]],
        segments: Sequence[object],
        max_virt_segments: int,
    ) -> tuple[List[tuple[int, int, float]], Optional[str]]:
        max_virt_segments = max(0, int(max_virt_segments))
        segment_count = max(1, len(segments or []))
        normalized = [self._normalize_export_weight(weight_tuple, segment_count) for weight_tuple in (vertex_weights or [])]
        original_count = self._estimate_virt_segment_count_from_weights(normalized, segments)

        if original_count <= max_virt_segments:
            return list(normalized), None

        optimized = list(normalized)
        final_count = original_count
        notes: List[str] = []

        for step in _MODEL_FLAT_SHADED_WEIGHT_STEPS:
            candidate = self._quantize_flat_shaded_weights(normalized, segment_count, float(step))
            candidate_count = self._estimate_virt_segment_count_from_weights(candidate, segments)
            optimized = candidate
            final_count = candidate_count
            notes.append(f'weights snapped to {float(step):.2f} increments')
            if candidate_count <= max_virt_segments:
                break

        if final_count > max_virt_segments:
            before_reduce = final_count
            optimized = self._reduce_weight_keys_to_limit(optimized, max_virt_segments, segment_count, float(_MODEL_FLAT_SHADED_WEIGHT_STEPS[-1]))
            final_count = self._estimate_virt_segment_count_from_weights(optimized, segments)
            if final_count < before_reduce:
                notes.append('low-impact bone-pair buckets were merged or baked')

        if final_count > max_virt_segments:
            before_collapse = final_count
            optimized = self._collapse_lowest_impact_weight_keys(optimized, segments, max_virt_segments)
            final_count = self._estimate_virt_segment_count_from_weights(optimized, segments)
            if final_count < before_collapse:
                notes.append('remaining low-impact weights were collapsed to the nearest single bone')

        if not self._weights_changed(normalized, optimized, segment_count):
            return list(normalized), None

        model_name = getattr(model_root, 'name', '<unnamed model>')
        detail = '; '.join(dict.fromkeys(notes)) or 'vertex weights were optimized'
        return list(optimized), (
            f'Model "{model_name}" exceeded the PS2 transform palette limit. '
            f'PS2 PrimitiveInfo can address only 256 Segment/VirtSegment transforms, so export optimized weighted vertices '
            f'({detail}). VirtSegments: {original_count} -> {final_count} (limit {max_virt_segments}).'
        )

    def _ps2_primitive_layout_targets(self, directory: Path, layout: _OriginalModelLayout) -> tuple[int, int, Optional[int]]:
        table_target = int(layout.target_for(88))
        payload_target = table_target
        material_target: Optional[int] = None

        source_path = Path(directory) / str(layout.model_filename)
        if not source_path.exists():
            return table_target, payload_target, material_target

        cache = None
        try:
            cache = SectionContextCache(str(source_path))
            root_ctx = cache.get_root_context()
            br = root_ctx.reader
            br.seek(root_ctx.data_start + 88)
            primitive_table_raw = int(br.u32())
            table_ctx, table_abs = resolve_pointer(cache, root_ctx, 88, primitive_table_raw)
            try:
                table_target = int(_section_index_from_filename(table_ctx.file_name))
            except Exception:
                pass
            table_local = int(table_abs) - int(table_ctx.data_start)

            for record_index in range(0, 16):
                record_local = table_local + (record_index * 32)
                if record_local < 0 or record_local + 12 > int(table_ctx.data_size):
                    break
                try:
                    table_ctx.reader.seek(table_ctx.data_start + record_local)
                    first_word = int(table_ctx.reader.u32())
                except Exception:
                    first_word = 0
                if first_word == 0:
                    break

                material_reloc = table_ctx.section_info.relocations_by_offset.get(record_local + 4)
                if material_reloc is not None and int(getattr(material_reloc, 'type', 0)) == 2:
                    material_target = int(getattr(material_reloc, 'section_index_or_type', table_target))

                payload_reloc = table_ctx.section_info.relocations_by_offset.get(record_local + 8)
                if payload_reloc is not None:
                    payload_target = int(getattr(payload_reloc, 'section_index_or_type', payload_target))

                if material_target is not None and payload_target != table_target:
                    break
        except Exception as exc:
            logger.debug('Could not infer PS2 primitive relocation targets from %s: %s', source_path.name, exc)
        finally:
            try:
                if cache is not None:
                    cache.close()
            except Exception:
                pass

        return int(table_target), int(payload_target), material_target

    def _append_ps2_zero_block(self, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, target_section_index: int, preferred_offset: int = 0) -> int:
        buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, int(target_section_index))
        preferred_offset = int(preferred_offset or 0)
        if preferred_offset > 0:
            if preferred_offset >= len(buffer.data):
                buffer.data.extend(b'\x00' * (preferred_offset - len(buffer.data)))
                offset = preferred_offset
                buffer.data.extend(b'\x00\x00\x00\x00')
                return int(offset)
            if preferred_offset + 4 <= len(buffer.data) and bytes(buffer.data[preferred_offset:preferred_offset + 4]) == b'\x00\x00\x00\x00':
                return int(preferred_offset)
        return int(buffer.append(b'\x00\x00\x00\x00', alignment=4))

    def _write_ps2_optional_zero_pointer(self, root: _StandaloneSectionBuffer, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, field_offset: int) -> None:
        target = int(layout.target_for(int(field_offset)))
        value = int(getattr(layout, 'pointer_values', {}).get(int(field_offset), 0) or 0)
        if target == int(root.section_index) and value == 0:
            if int(field_offset) + 4 <= len(root.data):
                struct.pack_into('<I', root.data, int(field_offset), 0)
            return

        zero_offset = self._append_ps2_zero_block(buffers, directory, layout, section_list, target, preferred_offset=value)
        self._write_pointer(root, int(field_offset), target, zero_offset)

    def _write_model_sections(self, directory: Path, layout: _OriginalModelLayout, model: ModelData, section_list: _SectionList) -> List[Path]:
        buffers: Dict[int, _StandaloneSectionBuffer] = {}
        root = self._make_buffer(directory, layout, section_list, layout.model_section_index, root=True, default_ext='tr7aemesh')
        root.reserve(_PS2_MODEL_HEADER_SIZE, alignment=16)
        buffers[root.section_index] = root

        struct.pack_into('<iii', root.data, 0, _PS2_MODEL_VERSION_MAGIC, len(model.segments), len(model.virt_segments))
        self._write_ps2_bone_mirrors(root, buffers, directory, layout, section_list, model)
        self._write_segment_data(root, buffers, directory, layout, section_list, model)
        primitive_offsets = self._write_ps2_primitive_data(root, buffers, directory, layout, section_list, model)

        max_rad = self._generate_max_radius(model)
        struct.pack_into('<4f', root.data, 16, *[float(v) for v in model.model_scale])
        struct.pack_into('<i', root.data, 32, len(model.vertices))
        struct.pack_into('<i', root.data, 40, len(model.vertices))
        struct.pack_into('<i', root.data, 48, len(model.faces))
        struct.pack_into('<ff', root.data, 60, float(max_rad), float(max_rad) * float(max_rad))

        for field_offset in (36, 44, 52, 56, 68, 72, 76, 80, 84, 92, 96, 100, 104, 116, 120, 128, 136):
            try:
                self._write_ps2_optional_zero_pointer(root, buffers, directory, layout, section_list, field_offset)
            except Exception:
                if field_offset + 4 <= len(root.data):
                    struct.pack_into('<I', root.data, field_offset, 0)

        if not primitive_offsets:
            self._add_warning('PS2 model export wrote no primitive records; check material assignments and mesh triangles.')

        written: List[Path] = []
        for section_index, buffer in sorted(buffers.items(), key=lambda item: item[0]):
            out_path = directory / buffer.filename
            _write_standalone_section_file(out_path, buffer)
            written.append(out_path)
            logger.info('Wrote PS2 TRLAU model export section %s', out_path.name)
        return written

    def _write_ps2_bone_mirrors(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        entries = list(getattr(model, 'bone_mirror_entries', []) or [])
        if not entries:
            return

        payload = bytearray()
        for entry in entries:
            payload.extend(bytes((
                int(getattr(entry, 'bone1', 0)) & 0xFF,
                int(getattr(entry, 'bone2', 0)) & 0xFF,
                int(getattr(entry, 'count', 0)) & 0xFF,
            )))
        payload.extend(b'\x00\x00')

        target = int(layout.target_for(_PS2_BONE_MIRROR_FIELD_OFFSET))
        preferred = self._preferred_pointer_offset(layout, _PS2_BONE_MIRROR_FIELD_OFFSET) if target == int(layout.model_section_index) else None
        if target == int(layout.model_section_index) and preferred is None:
            preferred = 0xBC

        offset, _buffer = self._write_payload_to_target(
            buffers,
            directory,
            layout,
            section_list,
            target,
            bytes(payload),
            alignment=4,
            preferred_offset=preferred,
        )
        self._write_pointer(root, _PS2_BONE_MIRROR_FIELD_OFFSET, target, offset)

    def _write_segment_data(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        target = layout.target_for(12)
        seg_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, target)

        if seg_buffer is root and int(target) == int(root.section_index):
            preferred_offset = 0
            try:
                preferred_offset = int(getattr(layout, 'pointer_values', {}).get(12, 0) or 0)
            except Exception:
                preferred_offset = 0
            if preferred_offset < _PS2_MODEL_HEADER_SIZE:
                preferred_offset = _PS2_MODEL_HEADER_SIZE
            desired_offset = max(_PS2_MODEL_HEADER_SIZE, preferred_offset)
            if len(seg_buffer.data) < desired_offset:
                seg_buffer.data.extend(b'\x00' * (desired_offset - len(seg_buffer.data)))
        segment_list_offset = seg_buffer.align(16)

        segments = sorted(list(model.segments or []), key=lambda item: int(item.index))
        for ordinal, segment in enumerate(segments):
            expected_index = int(getattr(segment, 'index', ordinal))
            if expected_index != ordinal:
                expected_index = ordinal
            offset = segment_list_offset + (expected_index * _PS2_COMPACT_SEGMENT_SIZE)
            if len(seg_buffer.data) < offset + _PS2_COMPACT_SEGMENT_SIZE:
                seg_buffer.data.extend(b'\x00' * ((offset + _PS2_COMPACT_SEGMENT_SIZE) - len(seg_buffer.data)))
            seg_buffer.pack_at(offset, '<4fihhiI',
                *[float(v) for v in getattr(segment, 'pivot', (0.0, 0.0, 0.0, 1.0))],
                int(getattr(segment, 'flags', 0) or 0), # probably wrong
                int(getattr(segment, 'first_vertex', -1) or -1),
                int(getattr(segment, 'last_vertex', -1) or -1),
                int(getattr(segment, 'parent', -1) or -1),
                0)

        virt_base = segment_list_offset + (len(segments) * _PS2_COMPACT_SEGMENT_SIZE)
        for virt_index, virt in enumerate(list(model.virt_segments or [])):
            offset = virt_base + (virt_index * _PS2_COMPACT_VIRT_SEGMENT_SIZE)
            if len(seg_buffer.data) < offset + _PS2_COMPACT_VIRT_SEGMENT_SIZE:
                seg_buffer.data.extend(b'\x00' * ((offset + _PS2_COMPACT_VIRT_SEGMENT_SIZE) - len(seg_buffer.data)))
            seg_buffer.pack_at(offset, '<4fihhhhf',
                *[float(v) for v in getattr(virt, 'pivot', (0.0, 0.0, 0.0, 1.0))],
                int(getattr(virt, 'flags', 8) or 0),
                int(getattr(virt, 'first_vertex', -1) or -1),
                int(getattr(virt, 'last_vertex', -1) or -1),
                int(getattr(virt, 'index', -1) or -1),
                int(getattr(virt, 'weight_index', -1) or -1),
                float(getattr(virt, 'weight', 0.0) or 0.0))

        hinfo_payloads = self._build_hinfo_payloads(model)
        for segment in segments:
            hinfo_payload = hinfo_payloads.get(int(segment.index))
            if not hinfo_payload:
                continue
            hinfo_target = layout.hinfo_target_for(int(segment.index))
            hinfo_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, hinfo_target)
            hinfo_offset = hinfo_buffer.align(4)
            hinfo_data = bytearray(hinfo_payload)
            hinfo_pointer_fields: List[int] = []
            for pointer_field in (4, 12, 20, 28):
                value = struct.unpack_from('<I', hinfo_data, pointer_field)[0]
                if value:
                    struct.pack_into('<I', hinfo_data, pointer_field, int(hinfo_offset + value) & 0xFFFFFFFF)
                    hinfo_pointer_fields.append(pointer_field)
            hinfo_buffer.data.extend(hinfo_data)
            for pointer_field in hinfo_pointer_fields:
                hinfo_buffer.relocations.append(_Relocation(hinfo_target, int(hinfo_offset + pointer_field)))
            segment_record_offset = segment_list_offset + (int(segment.index) * _PS2_COMPACT_SEGMENT_SIZE)
            field_offset = segment_record_offset + (_PS2_COMPACT_SEGMENT_SIZE - 4)
            seg_buffer.pack_at(field_offset, '<I', int(hinfo_offset) & 0xFFFFFFFF)
            seg_buffer.relocations.append(_Relocation(hinfo_target, int(field_offset)))

        self._write_pointer(root, 12, target, segment_list_offset)

    def _read_ps2_template_primitive_records(self, directory: Path, layout: _OriginalModelLayout) -> List[_PS2TemplatePrimitiveRecord]:
        source_path = Path(directory) / str(layout.model_filename)
        if not source_path.exists():
            return []
        cache = None
        records: List[_PS2TemplatePrimitiveRecord] = []
        try:
            cache = SectionContextCache(str(source_path))
            root_ctx = cache.get_root_context()
            br = root_ctx.reader
            br.seek(root_ctx.data_start + 88)
            primitive_table_raw = int(br.u32())
            table_ctx, table_abs = resolve_pointer(cache, root_ctx, 88, primitive_table_raw)
            if table_abs <= 0:
                return []
            table_local = int(table_abs) - int(table_ctx.data_start)
            br = table_ctx.reader
            for index in range(0, 4096):
                local = table_local + (index * 32)
                absolute = table_ctx.data_start + local
                if local < 0 or absolute + 32 > table_ctx.file_size:
                    break
                br.seek(absolute)
                blob = br.read(32)
                if struct.unpack_from('<I', blob, 0)[0] == 0:
                    break
                vertex_count, qword_count, palette_count, flags = struct.unpack_from('<BBBB', blob, 0)
                tpageid = struct.unpack_from('<I', blob, 4)[0]
                sort_key = struct.unpack_from('<f', blob, 12)[0]
                palette_count = max(0, min(int(palette_count), 14))
                palette = [int(value) for value in blob[0x12:0x12 + palette_count]]
                records.append(_PS2TemplatePrimitiveRecord(
                    index=int(index),
                    vertex_count=int(vertex_count),
                    qword_count=int(qword_count),
                    palette_count=int(palette_count),
                    flags=int(flags),
                    tpageid=int(tpageid) & 0xFFFFFFFF,
                    sort_key=float(sort_key),
                    palette=palette,
                    unknown10=bytes(blob[0x10:0x12]),
                ))
        except Exception as exc:
            logger.debug('Could not read PS2 template primitive records from %s: %s', source_path.name, exc)
        finally:
            try:
                if cache is not None:
                    cache.close()
            except Exception:
                pass
        return records

    @staticmethod
    def _patch_ps2_payload_sort_key(payload: bytes, sort_key: float) -> bytes:
        if len(payload) < 12:
            return payload
        try:
            word = struct.unpack_from('<I', payload, 0)[0]
            cmd = (word >> 24) & 0xFF
            num = (word >> 16) & 0xFF
            imm = word & 0xFFFF
            if cmd == 0x64 and num == 1 and imm == 0x8001:
                data = bytearray(payload)
                struct.pack_into('<fI', data, 4, float(sort_key), 0)
                return bytes(data)
        except Exception:
            pass
        return payload

    def _apply_ps2_template_primitive_metadata(self, records: List[_PS2PrimitiveRecord], directory: Path, layout: _OriginalModelLayout) -> None:
        template_records = self._read_ps2_template_primitive_records(directory, layout)
        if not template_records or not records:
            return

        by_tpage: Dict[int, List[_PS2TemplatePrimitiveRecord]] = {}
        for template in template_records:
            by_tpage.setdefault(int(template.tpageid) & 0xFFFFFFFF, []).append(template)
        used_by_tpage: Dict[int, int] = {}
        used_global = 0

        for record_index, record in enumerate(records):
            tpageid = int(record.tpageid) & 0xFFFFFFFF
            candidates = by_tpage.get(tpageid) or []
            template: Optional[_PS2TemplatePrimitiveRecord] = None
            if candidates:
                cursor = int(used_by_tpage.get(tpageid, 0))
                if cursor < len(candidates):
                    template = candidates[cursor]
                    used_by_tpage[tpageid] = cursor + 1
                else:
                    template = candidates[-1]
            elif record_index < len(template_records):
                template = template_records[record_index]
            elif template_records:
                template = template_records[-1]

            if template is None:
                continue
            record.sort_key = float(template.sort_key)
            record.payload = self._patch_ps2_payload_sort_key(record.payload, record.sort_key)
            if template.unknown10 and template.unknown10 != b'\x00\x00':
                record.unknown10 = bytes(template.unknown10[:2]).ljust(2, b'\x00')

    def _write_ps2_primitive_data(self, root: _StandaloneSectionBuffer, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, model: ModelData) -> List[Tuple[int, int]]:
        records = self._build_ps2_primitive_records(model)
        if not records:
            return []
        self._apply_ps2_template_primitive_metadata(records, directory, layout)

        table_target, payload_target, material_target = self._ps2_primitive_layout_targets(directory, layout)
        primitive_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, table_target)
        payload_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, payload_target)

        table_offset = primitive_buffer.align(16)
        table_size = (len(records) + 1) * 32
        primitive_buffer.data.extend(b'\x00' * table_size)

        written_offsets: List[Tuple[int, int]] = []
        for index, record in enumerate(records):
            payload_offset = payload_buffer.append(record.payload, alignment=16)
            written_offsets.append((int(payload_target), int(payload_offset)))
            entry_offset = table_offset + (index * 32)
            primitive_buffer.pack_at(entry_offset + 0, '<BBBB', int(record.vertex_count) & 0xFF, int(record.qword_count) & 0xFF, len(record.palette) & 0xFF, int(record.flags) & 0xFF)
            primitive_buffer.pack_at(entry_offset + 4, '<I', int(record.tpageid) & 0xFFFFFFFF)
            if material_target is not None:
                primitive_buffer.relocations.append(_Relocation(int(material_target), entry_offset + 4, relocation_type=2))
            primitive_buffer.pack_at(entry_offset + 8, '<I', int(payload_offset) & 0xFFFFFFFF)
            primitive_buffer.relocations.append(_Relocation(int(payload_target), entry_offset + 8))
            primitive_buffer.pack_at(entry_offset + 12, '<f', float(record.sort_key))
            unknown10 = bytes(getattr(record, 'unknown10', b'\x00\x00') or b'\x00\x00')[:2].ljust(2, b'\x00')
            primitive_buffer.data[entry_offset + 0x10:entry_offset + 0x12] = unknown10
            for palette_index, bone_id in enumerate(record.palette[:_PS2_MAX_PALETTE_ENTRIES]):
                primitive_buffer.data[entry_offset + 0x12 + palette_index] = int(bone_id) & 0xFF

        self._write_pointer(root, 88, table_target, table_offset)
        return written_offsets

    def _build_ps2_primitive_records(self, model: ModelData) -> List[_PS2PrimitiveRecord]:
        records: List[_PS2PrimitiveRecord] = []
        vertices = list(model.vertices or [])
        if not vertices:
            return records

        split_for_size = 0
        split_for_palette = 0
        skipped_for_size = 0

        for strip_index, strip in enumerate(model.strips or []):
            indices = [int(index) for index in list(getattr(strip, 'indices', []) or []) if 0 <= int(index) < len(vertices)]
            if len(indices) < 3:
                continue
            triangle_indices = indices[:len(indices) - (len(indices) % 3)]
            if not triangle_indices:
                continue

            chunks = self._split_ps2_triangle_chunks(vertices, triangle_indices, strip)
            if len(chunks) > 1:
                split_for_size += max(0, len(triangle_indices) // _PS2_SAFE_PRIMITIVE_VERTEX_LIMIT)
            for chunk in chunks:
                if len(chunk) < 3:
                    continue
                stream = self._stripify_ps2_triangle_indices(chunk)
                if len(stream) < 3:
                    continue
                palette = self._palette_for_indices(vertices, [entry[0] for entry in stream], strip)
                payload = self._build_ps2_vif_payload(vertices, stream, palette, vertex_colors=getattr(model, 'vertex_colors', None), sort_key=float(getattr(strip, 'sort_push', 0.0) or 0.0))
                qword_count = (len(payload) + 15) // 16
                if qword_count > _PS2_VIF_UNPACK_COUNT_LIMIT:
                    sub_records = self._records_for_oversized_ps2_chunk(model, vertices, chunk, strip)
                    if sub_records:
                        records.extend(sub_records)
                    else:
                        skipped_for_size += 1
                    continue
                if len(palette) >= _PS2_MAX_PALETTE_ENTRIES and self._chunk_needs_more_palette_entries(vertices, [entry[0] for entry in stream], palette):
                    split_for_palette += 1
                records.append(_PS2PrimitiveRecord(
                    vertex_count=len(stream),
                    qword_count=qword_count,
                    palette=palette,
                    flags=int(getattr(strip, 'draw_group', 0) or 0) & 0xFF,
                    tpageid=self._ps2_tpageid_for_stream(strip, vertices, [entry[0] for entry in stream]),
                    sort_key=float(getattr(strip, 'sort_push', 0.0) or 0.0),
                    payload=payload,
                ))

        if split_for_size:
            self._add_warning(f'PS2 exporter split large primitive streams into smaller VIF packets to stay under the 255-qword limit.')
        if split_for_palette:
            self._add_warning('Some PS2 primitive chunks still need more than 7 bone/matrix palette entries; affected vertices use palette slot 0.')
        if skipped_for_size:
            self._add_warning(f'PS2 exporter skipped {skipped_for_size} primitive chunk(s) that still exceeded the 255-qword limit after extra splitting.')
        return records

    def _split_ps2_triangle_chunks(self, vertices: Sequence[MVertex], triangle_indices: Sequence[int], strip: TextureStrip) -> List[List[int]]:
        chunks: List[List[int]] = []
        current: List[int] = []
        current_palette: List[int] = []

        for start in range(0, len(triangle_indices), 3):
            tri = [int(value) for value in triangle_indices[start:start + 3]]
            if len(tri) < 3:
                continue
            tri_palette = self._palette_candidates_for_indices(vertices, tri, strip)
            merged_palette = list(current_palette)
            palette_overflow = False
            for bone_id in tri_palette:
                if bone_id in merged_palette:
                    continue
                if len(merged_palette) >= _PS2_MAX_PALETTE_ENTRIES:
                    palette_overflow = True
                    break
                merged_palette.append(bone_id)

            size_overflow = current and (len(current) + len(tri) > _PS2_SAFE_PRIMITIVE_VERTEX_LIMIT)
            if current and (size_overflow or palette_overflow):
                chunks.append(current)
                current = []
                current_palette = []
                merged_palette = []
                palette_overflow = False
                for bone_id in tri_palette:
                    if bone_id in merged_palette:
                        continue
                    if len(merged_palette) >= _PS2_MAX_PALETTE_ENTRIES:
                        palette_overflow = True
                        break
                    merged_palette.append(bone_id)

            current.extend(tri)
            current_palette = merged_palette

        if current:
            chunks.append(current)
        return chunks

    def _records_for_oversized_ps2_chunk(self, model: ModelData, vertices: Sequence[MVertex], chunk: Sequence[int], strip: TextureStrip) -> List[_PS2PrimitiveRecord]:
        records: List[_PS2PrimitiveRecord] = []
        if len(chunk) <= 3:
            return records
        midpoint = (len(chunk) // 6) * 3
        if midpoint <= 0 or midpoint >= len(chunk):
            midpoint = (len(chunk) // 2) - ((len(chunk) // 2) % 3)
        if midpoint <= 0 or midpoint >= len(chunk):
            return records
        for sub_chunk in (list(chunk[:midpoint]), list(chunk[midpoint:])):
            if len(sub_chunk) < 3:
                continue
            stream = self._stripify_ps2_triangle_indices(sub_chunk)
            if len(stream) < 3:
                continue
            palette = self._palette_for_indices(vertices, [entry[0] for entry in stream], strip)
            payload = self._build_ps2_vif_payload(vertices, stream, palette, vertex_colors=getattr(model, 'vertex_colors', None), sort_key=float(getattr(strip, 'sort_push', 0.0) or 0.0))
            qword_count = (len(payload) + 15) // 16
            if qword_count > _PS2_VIF_UNPACK_COUNT_LIMIT:
                records.extend(self._records_for_oversized_ps2_chunk(model, vertices, sub_chunk, strip))
                continue
            records.append(_PS2PrimitiveRecord(
                vertex_count=len(stream),
                qword_count=qword_count,
                palette=palette,
                flags=int(getattr(strip, 'draw_group', 0) or 0) & 0xFF,
                tpageid=self._ps2_tpageid_for_stream(strip, vertices, [entry[0] for entry in stream]),
                sort_key=float(getattr(strip, 'sort_push', 0.0) or 0.0),
                payload=payload,
            ))
        return records

    def _palette_for_indices(self, vertices: Sequence[MVertex], indices: Sequence[int], strip: TextureStrip) -> List[int]:
        palette: List[int] = []
        overflow = False
        for value in self._palette_candidates_for_indices(vertices, indices, strip):
            if value in palette:
                continue
            if len(palette) >= _PS2_MAX_PALETTE_ENTRIES:
                overflow = True
                continue
            palette.append(value)
        if overflow:
            pass
        if not palette:
            palette.append(0)
        return palette[:_PS2_MAX_PALETTE_ENTRIES]

    def _palette_candidates_for_indices(self, vertices: Sequence[MVertex], indices: Sequence[int], strip: TextureStrip) -> List[int]:
        candidates: List[int] = []
        for vertex_index in indices:
            vertex = vertices[int(vertex_index)]
            for value in self._vertex_bone_candidates(vertex):
                if value >= 0 and value not in candidates:
                    candidates.append(value)
        if candidates:
            return candidates

        for value in [int(v) for v in list(getattr(strip, 'bone_ids', []) or []) if int(v) >= 0]:
            if value not in candidates:
                candidates.append(value)
        return candidates

    @staticmethod
    def _vertex_bone_candidates(vertex: MVertex) -> List[int]:
        values: List[int] = []
        for value in (
            int(getattr(vertex, 'gc_transform_id', -1)),
            int(getattr(vertex, 'segment', -1)),
            int(getattr(vertex, 'gc_primary_segment', -1)),
            int(getattr(vertex, 'gc_bind_segment', -1)),
        ):
            if value >= 0 and value not in values:
                values.append(value)
        return values

    def _chunk_needs_more_palette_entries(self, vertices: Sequence[MVertex], indices: Sequence[int], palette: Sequence[int]) -> bool:
        palette_set = {int(value) for value in palette}
        for vertex_index in indices:
            for value in self._vertex_bone_candidates(vertices[int(vertex_index)]):
                if value >= 0 and value not in palette_set:
                    return True
        return False

    @staticmethod
    def _vertex_palette_slot(vertex: MVertex, palette: Sequence[int]) -> int:
        candidates = [
            int(getattr(vertex, 'gc_transform_id', -1)),
            int(getattr(vertex, 'segment', -1)),
            int(getattr(vertex, 'gc_primary_segment', -1)),
            int(getattr(vertex, 'gc_bind_segment', -1)),
        ]
        for candidate in candidates:
            if candidate < 0:
                continue
            try:
                return list(palette).index(candidate)
            except ValueError:
                continue
        return 0

    @staticmethod
    def _same_oriented_triangle(a: Sequence[int], b: Sequence[int]) -> bool:
        if len(a) != 3 or len(b) != 3:
            return False
        aa = (int(a[0]), int(a[1]), int(a[2]))
        bb = (int(b[0]), int(b[1]), int(b[2]))
        return aa == bb or aa == (bb[1], bb[2], bb[0]) or aa == (bb[2], bb[0], bb[1])

    @staticmethod
    def _third_vertex_for_edge(tri: Sequence[int], edge_a: int, edge_b: int) -> Optional[int]:
        values = [int(v) for v in tri]
        if int(edge_a) not in values or int(edge_b) not in values:
            return None
        for value in values:
            if value != int(edge_a) and value != int(edge_b):
                return int(value)
        return None

    def _stripify_ps2_triangle_indices(self, triangle_indices: Sequence[int]) -> List[Tuple[int, int]]:
        triangles: List[Tuple[int, int, int]] = []
        usable = len(triangle_indices) - (len(triangle_indices) % 3)
        for start in range(0, usable, 3):
            tri = (int(triangle_indices[start]), int(triangle_indices[start + 1]), int(triangle_indices[start + 2]))
            if len({tri[0], tri[1], tri[2]}) == 3:
                triangles.append(tri)
        if not triangles:
            return []

        unused = set(range(len(triangles)))
        stream: List[Tuple[int, int]] = []

        while unused:
            tri_index = min(unused)
            unused.remove(tri_index)
            tri = triangles[tri_index]
            current_vertices: List[int] = [tri[0], tri[1], tri[2]]
            current_flags: List[int] = [0x8000, 0x8000, 0x0000]

            while unused:
                edge_a = current_vertices[-2]
                edge_b = current_vertices[-1]
                chosen_index: Optional[int] = None
                chosen_vertex: Optional[int] = None
                chosen_flip = 0

                for candidate_index in sorted(unused):
                    candidate = triangles[candidate_index]
                    third = self._third_vertex_for_edge(candidate, edge_a, edge_b)
                    if third is None:
                        continue
                    no_flip = (edge_a, edge_b, third)
                    flip = (edge_b, edge_a, third)
                    if self._same_oriented_triangle(candidate, no_flip):
                        chosen_index = candidate_index
                        chosen_vertex = third
                        chosen_flip = 0x0000
                        break
                    if self._same_oriented_triangle(candidate, flip):
                        chosen_index = candidate_index
                        chosen_vertex = third
                        chosen_flip = 0x4000
                        break

                if chosen_index is None or chosen_vertex is None:
                    break
                unused.remove(chosen_index)
                current_vertices.append(int(chosen_vertex))
                current_flags.append(int(chosen_flip))

            for vertex_index, flag in zip(current_vertices, current_flags):
                stream.append((int(vertex_index), int(flag) & 0xC000))

        return stream

    def _build_ps2_vif_payload(self, vertices: Sequence[MVertex], stream: Sequence[Tuple[int, int]], palette: Sequence[int], *, vertex_colors: Sequence[Tuple[int, int, int, int]] | None = None, sort_key: float = 0.0) -> bytes:
        count = len(stream)
        positions = bytearray()
        colors = bytearray()
        normals = bytearray()
        uvs = bytearray()

        for local_index, stream_entry in enumerate(stream):
            vertex_index, extra_flag = int(stream_entry[0]), int(stream_entry[1])
            vertex = vertices[int(vertex_index)]
            x, y, z = (int(v) for v in getattr(vertex, 'position_raw', (0, 0, 0)))
            flag = self._ps2_position_flag(vertex, palette) | (int(extra_flag) & 0xC000)
            positions.extend(struct.pack('<hhhH', _clamp_i16(x), _clamp_i16(y), _clamp_i16(z), int(flag) & 0xFFFF))

            color = self._vertex_color_for_export(vertex, int(vertex_index), vertex_colors)
            colors.extend(bytes((color[0] & 0xFF, color[1] & 0xFF, color[2] & 0xFF, color[3] & 0xFF)))

            nx, ny, nz = (int(v) for v in getattr(vertex, 'normal_raw', (0, 0, 127)))
            normals.extend(struct.pack('<hhh', _clamp_i16(_clamp_i8(nx) * 32), _clamp_i16(_clamp_i8(ny) * 32), _clamp_i16(_clamp_i8(nz) * 32)))

            u, v = (int(value) for value in getattr(vertex, 'uv_raw', (0, 0)))
            uvs.extend(struct.pack('<hh', _clamp_i16(u), _clamp_i16(v)))

        payload = bytearray()
        self._append_vif_unpack(payload, 0x64, 1, struct.pack('<fI', float(sort_key), 0), imm=0x8001)
        self._append_vif_unpack(payload, 0x6D, count, bytes(positions), imm=0x8066)
        self._append_vif_unpack(payload, 0x6E, count, bytes(colors), imm=0xC065)
        self._append_vif_unpack(payload, 0x69, count, bytes(normals), imm=0x8064)
        if any(value != 0 for value in uvs):
            self._append_vif_unpack(payload, 0x65, count, bytes(uvs), imm=0x8063)
        _align_bytes(payload, 16)
        return bytes(payload)

    @staticmethod
    def _append_vif_unpack(payload: bytearray, cmd: int, num: int, data: bytes, *, imm: int = 0) -> None:
        if not (0 <= int(num) <= 255):
            raise ValueError(f'PS2 VIF UNPACK count outside 0..255: {num}')
        word = ((int(cmd) & 0xFF) << 24) | ((int(num) & 0xFF) << 16) | (int(imm) & 0xFFFF)
        payload.extend(struct.pack('<I', word))
        payload.extend(data)
        _align_bytes(payload, 4)

    def _ps2_position_flag(self, vertex: MVertex, palette: Sequence[int]) -> int:
        slot = self._vertex_palette_slot(vertex, palette)
        return (int(slot) * 12) & 0x3FFF

    def _vertex_color_for_export(self, vertex: MVertex, vertex_index: int, vertex_colors: Sequence[Tuple[int, int, int, int]] | None = None) -> Tuple[int, int, int, int]:
        if vertex_colors is not None and 0 <= int(vertex_index) < len(vertex_colors):
            try:
                source = vertex_colors[int(vertex_index)]
                if source is not None and len(source) >= 4:
                    return tuple(max(0, min(255, int(value))) for value in source[:4])  # type: ignore[return-value]
            except Exception:
                pass
        color = None
        try:
            color = getattr(vertex, 'psp_color_rgba', None) or getattr(vertex, 'ps3_color_rgba', None)
        except Exception:
            color = None
        if color is not None and len(color) >= 4:
            return tuple(max(0, min(255, int(value))) for value in color[:4])  # type: ignore[return-value]
        return (255, 255, 255, 128)

    def _ps2_tpageid_for_strip(self, strip: TextureStrip) -> int:
        raw = int(getattr(strip, 'ps2_tpageid_raw', -1) or -1)
        if raw >= 0:
            return raw & 0xFFFFFFFF
        return self._canonical_tpageid_to_ps2_raw(int(getattr(strip, 'tpageid', 0) or 0))

    def _ps2_tpageid_for_stream(self, strip: TextureStrip, vertices: Sequence[MVertex], stream_indices: Sequence[int]) -> int:
        return self._ps2_tpageid_for_strip(strip) & 0xFFFFFFFF

    @staticmethod
    def _ps2_stream_uses_out_of_tile_uv(vertices: Sequence[MVertex], stream_indices: Sequence[int]) -> bool:
        for vertex_index in stream_indices:
            if int(vertex_index) < 0 or int(vertex_index) >= len(vertices):
                continue
            try:
                u, v = vertices[int(vertex_index)].uv_raw
                u = int(u)
                v = int(v)
            except Exception:
                continue
            if u < 0 or u > 4096 or v < 0 or v > 4096:
                return True
        return False

    @staticmethod
    def _material_panel_bool(material, name: str, default: bool = False) -> bool:
        try:
            return bool(getattr(material, name))
        except Exception:
            return bool(default)

    @staticmethod
    def _material_panel_int(material, name: str, default: int = 0) -> int:
        try:
            return int(getattr(material, name))
        except Exception:
            return int(default)

    def _material_ps2_raw_tpageid(self, material) -> int:
        if material is None:
            return -1

        has_panel = False
        try:
            has_panel = hasattr(material, 'trlau_ui_texture_id')
        except Exception:
            has_panel = False
        if has_panel:
            flags = {
                'texture_id': self._material_panel_int(material, 'trlau_ui_texture_id', 0) & 0xFFFF,
                'blend_value': self._material_panel_int(material, 'trlau_ui_blend_value', 0) & 0xF,
                'cull_mode': self._material_panel_int(material, 'trlau_ui_cull_mode', 0) & 0x7,
                'unknown_1': 1 if self._material_panel_bool(material, 'trlau_ui_unknown_1', False) else 0,
                'single_sided': 1 if self._material_panel_bool(material, 'trlau_ui_single_sided', False) else 0,
                'double_sided': 1 if self._material_panel_bool(material, 'trlau_ui_ps2_double_sided', False) else 0,
                'texture_wrap': self._material_panel_int(material, 'trlau_ui_texture_wrap', 0) & 0x3,
                'unknown_2': 1 if self._material_panel_bool(material, 'trlau_ui_unknown_2', False) else 0,
                'unknown_3': 1 if self._material_panel_bool(material, 'trlau_ui_unknown_3', False) else 0,
                'flat_shading': 1 if self._material_panel_bool(material, 'trlau_ui_flat_shading', False) else 0,
                'sort_z': 1 if self._material_panel_bool(material, 'trlau_ui_sort_z', False) else 0,
                'stencil_pass': self._material_panel_int(material, 'trlau_ui_stencil_pass', 0) & 0x3,
                'stencil_func': 1 if self._material_panel_bool(material, 'trlau_ui_stencil_func', False) else 0,
                'alpha_ref': 1 if self._material_panel_bool(material, 'trlau_ui_alpha_ref', False) else 0,
            }
            return self._ps2_flags_to_raw_tpageid(flags)

        return -1

    @staticmethod
    def _ps2_flags_to_raw_tpageid(flags: dict) -> int:
        raw = int(flags.get('texture_id', 0)) & 0xFFFF
        raw |= (int(flags.get('blend_value', 0)) & 0xF) << 16

        render_mode = int(flags.get('cull_mode', 0)) & 0x7
        raw |= (render_mode & 0x7) << 20

        raw |= (int(flags.get('unknown_1', 0)) & 0x1) << 23
        raw |= (int(flags.get('unknown_2', 0)) & 0x1) << 24
        raw |= (int(flags.get('texture_wrap', 0)) & 0x3) << 25
        raw |= (int(flags.get('unknown_3', 0)) & 0x1) << 27
        raw |= (int(flags.get('stencil_func', 0)) & 0x1) << 28
        raw |= (int(flags.get('flat_shading', 0)) & 0x1) << 29
        raw |= (int(flags.get('sort_z', 0)) & 0x1) << 30
        double_sided = int(flags.get('double_sided', int(flags.get('stencil_pass', 0)) & 0x1)) & 0x1
        raw |= double_sided << 31
        return raw & 0xFFFFFFFF

    @staticmethod
    def _canonical_tpageid_to_ps2_raw(canonical_tpageid: int) -> int:
        flags = decode_tpage_flags(int(canonical_tpageid) & 0xFFFFFFFF)
        return TRLAUPS2ModelExporter._ps2_flags_to_raw_tpageid(flags)

    def _material_tpageid(self, material) -> int:
        if material is None:
            return 0
        raw = self._material_ps2_raw_tpageid(material)
        if raw >= 0:
            return self._ps2_raw_tpageid_to_canonical(raw)
        try:
            return int(get_material_tpageid(material)) & 0xFFFFFFFF
        except Exception:
            return 0

    @staticmethod
    def _ps2_raw_tpageid_to_canonical(raw_tpageid: int) -> int:
        raw = int(raw_tpageid) & 0xFFFFFFFF
        texture_id = raw & 0xFFFF
        blend_value = (raw >> 16) & 0xF
        canonical = texture_id & 0x1FFF
        canonical |= (blend_value & 0xF) << 13
        render_mode = (raw >> 20) & 0x7
        double_sided = (raw >> 31) & 0x1
        canonical |= (render_mode & 0x7) << 17
        if not double_sided:
            canonical |= 1 << 21
        canonical |= ((raw >> 23) & 0x1) << 20
        canonical |= ((raw >> 24) & 0x1) << 24
        canonical |= ((raw >> 25) & 0x3) << 22
        canonical |= ((raw >> 27) & 0x1) << 25
        canonical |= ((raw >> 28) & 0x1) << 30
        canonical |= ((raw >> 29) & 0x1) << 26
        canonical |= ((raw >> 30) & 0x1) << 27
        canonical |= double_sided << 28
        return canonical & 0xFFFFFFFF

    def _export_model_textures(self, collection, directory: Path, section_list: _SectionList) -> List[Path]:
        written: List[Path] = []
        seen: set[int] = set()
        for obj in _iter_collection_objects(collection):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'MESH':
                continue
            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            for material in getattr(mesh, 'materials', []) or []:
                if material is None:
                    continue
                raw_tpageid = self._material_ps2_raw_tpageid(material)
                texture_id = int(raw_tpageid) & 0xFFFF if int(raw_tpageid) >= 0 else 0
                if texture_id <= 0 or texture_id in seen:
                    continue
                image = self._find_material_image_texture(material)
                if image is None:
                    continue
                section_index, filename = self._texture_section_for_id(directory, section_list, texture_id)
                template_body = self._read_existing_ps2_pcd_body(directory / filename)
                pcd_body = self._image_to_ps2_pcd_body(image, template_body=template_body)
                buffer = _StandaloneSectionBuffer(section_index=section_index, filename=filename, section_type=5, section_id=texture_id, spec_mask=0xFFFFFFFF)
                buffer.data.extend(pcd_body)
                out_path = directory / filename
                _write_standalone_section_file(out_path, buffer)
                section_list.ensure_index(section_index, filename)
                written.append(out_path)
                seen.add(texture_id)
        return written

    @staticmethod
    def _read_existing_ps2_pcd_body(path: Path) -> Optional[bytes]:
        try:
            blob = Path(path).read_bytes()
        except Exception:
            return None
        if len(blob) < 24 or blob[:4] not in {b'SECT', b'TCES'}:
            return bytes(blob) if blob else None
        try:
            packed_data = struct.unpack_from('<I', blob, 12)[0]
            relocation_count = (int(packed_data) >> 8) & 0x00FFFFFF
            data_start = 24 + (int(relocation_count) * 8)
            if data_start <= len(blob):
                return bytes(blob[data_start:])
        except Exception:
            return None
        return None

    def _image_to_ps2_pcd_body(self, image, *, template_body: Optional[bytes] = None) -> bytes:
        rgba, width, height = self._read_image_rgba8(image)
        if width <= 0 or height <= 0 or (width & (width - 1)) != 0 or (height & (height - 1)) != 0:
            raise ValueError(f'PS2 texture export requires power-of-two dimensions; got {width}x{height}')

        rgba, width, height = self._clamp_ps2_texture_size(rgba, width, height, image_name=getattr(image, 'name', 'Image'))

        indexed = self._try_make_indexed8(rgba, width, height)
        if indexed is not None:
            indices, palette = indexed
            return build_ps2_indexed8_pcd_body(indices, palette, width, height, template_body=template_body)

        self._add_warning(f'PS2 texture "{getattr(image, "name", "Image")}" has more than 256 exact RGBA colors; quantizing to indexed PS2 PSMT8/CLUT')
        indices, palette = self._quantize_indexed8(rgba, width, height)
        return build_ps2_indexed8_pcd_body(indices, palette, width, height, template_body=template_body)

    def _clamp_ps2_texture_size(self, rgba: bytes, width: int, height: int, *, image_name: str = 'Image') -> Tuple[bytes, int, int]:
        width = int(width)
        height = int(height)
        if width <= _PS2_MAX_TEXTURE_DIMENSION and height <= _PS2_MAX_TEXTURE_DIMENSION:
            return rgba, width, height

        factor = 1
        while (width // factor) > _PS2_MAX_TEXTURE_DIMENSION or (height // factor) > _PS2_MAX_TEXTURE_DIMENSION:
            factor *= 2

        new_width = max(1, width // factor)
        new_height = max(1, height // factor)
        if new_width < 4 or new_height < 4:
            raise ValueError(
                f'PS2 texture "{image_name}" is too large to clamp safely: {width}x{height}. '
                f'Maximum supported PS2 export size is {_PS2_MAX_TEXTURE_DIMENSION}x{_PS2_MAX_TEXTURE_DIMENSION}.'
            )

        self._add_warning(
            f'PS2 texture "{image_name}" is {width}x{height}; downscaling to {new_width}x{new_height} for PS2 export '
            f'because textures above {_PS2_MAX_TEXTURE_DIMENSION}x{_PS2_MAX_TEXTURE_DIMENSION} cause runtime rendering issues.'
        )
        return self._downsample_rgba8_power_of_two(rgba, width, height, factor), new_width, new_height

    @staticmethod
    def _downsample_rgba8_power_of_two(rgba: bytes, width: int, height: int, factor: int) -> bytes:
        width = int(width)
        height = int(height)
        factor = max(1, int(factor))
        new_width = max(1, width // factor)
        new_height = max(1, height // factor)
        required = width * height * 4
        if len(rgba) < required:
            raise ValueError(f'Truncated RGBA source: expected {required} bytes, got {len(rgba)}')
        if factor <= 1:
            return bytes(rgba[:required])

        out = bytearray(new_width * new_height * 4)
        for y in range(new_height):
            src_y0 = y * factor
            for x in range(new_width):
                src_x0 = x * factor
                r = g = b = a = count = 0
                for yy in range(factor):
                    sy = src_y0 + yy
                    if sy >= height:
                        continue
                    row = sy * width
                    for xx in range(factor):
                        sx = src_x0 + xx
                        if sx >= width:
                            continue
                        off = ((row + sx) * 4)
                        r += int(rgba[off + 0])
                        g += int(rgba[off + 1])
                        b += int(rgba[off + 2])
                        a += int(rgba[off + 3])
                        count += 1
                count = max(1, count)
                dst = ((y * new_width) + x) * 4
                out[dst + 0] = int(round(r / count))
                out[dst + 1] = int(round(g / count))
                out[dst + 2] = int(round(b / count))
                out[dst + 3] = int(round(a / count))
        return bytes(out)

    @staticmethod
    def _read_image_rgba8(image) -> Tuple[bytes, int, int]:
        try:
            size = getattr(image, 'size', None)
            width = max(1, int(size[0] or 1))
            height = max(1, int(size[1] or 1))
        except Exception:
            width = height = 1
        try:
            channels = max(1, min(4, int(getattr(image, 'channels', 4) or 4)))
        except Exception:
            channels = 4
        expected = width * height * channels
        pixels_obj = getattr(image, 'pixels', None)
        if pixels_obj is None:
            raise ValueError('Image has no readable pixel buffer')
        pixels = array('f', [0.0]) * expected
        if hasattr(pixels_obj, 'foreach_get'):
            pixels_obj.foreach_get(pixels)
        else:
            for index in range(expected):
                pixels[index] = float(pixels_obj[index])

        rgba = bytearray(width * height * 4)
        for pixel_index in range(width * height):
            src = pixel_index * channels
            dst = pixel_index * 4
            if channels == 1:
                value = TRLAUPS2ModelExporter._float_to_u8(pixels[src])
                rgba[dst:dst + 4] = bytes((value, value, value, 255))
            elif channels == 2:
                value = TRLAUPS2ModelExporter._float_to_u8(pixels[src])
                alpha = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 1])
                rgba[dst:dst + 4] = bytes((value, value, value, alpha))
            elif channels == 3:
                rgba[dst + 0] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 0])
                rgba[dst + 1] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 1])
                rgba[dst + 2] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 2])
                rgba[dst + 3] = 255
            else:
                rgba[dst + 0] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 0])
                rgba[dst + 1] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 1])
                rgba[dst + 2] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 2])
                rgba[dst + 3] = TRLAUPS2ModelExporter._float_to_u8(pixels[src + 3])
        return bytes(rgba), width, height

    @staticmethod
    def _float_to_u8(value: float) -> int:
        try:
            value = float(value)
        except Exception:
            value = 0.0
        if value <= 0.0:
            return 0
        if value >= 1.0:
            return 255
        return int(value * 255.0 + 0.5)

    @staticmethod
    def _quantize_indexed8(rgba: bytes, width: int, height: int) -> Tuple[bytes, bytes]:
        pixel_count = int(width) * int(height)
        samples = [bytes(rgba[i * 4:i * 4 + 4]) for i in range(pixel_count)]
        if not samples:
            return b'', b'\x00' * 1024

        from collections import Counter, defaultdict
        bucket_counts: Counter[tuple[int, int, int, int]] = Counter()
        bucket_sums: dict[tuple[int, int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
        for color in samples:
            r, g, b, a = color
            if a < 8:
                key = (0, 0, 0, 0)
            else:
                key = (int(r) >> 3, int(g) >> 3, int(b) >> 3, 1 if int(a) >= 128 else 0)
            bucket_counts[key] += 1
            acc = bucket_sums[key]
            acc[0] += int(r)
            acc[1] += int(g)
            acc[2] += int(b)
            acc[3] += int(a)
            acc[4] += 1

        ranked = [key for key, _count in bucket_counts.most_common(256)]
        palette_colors: list[tuple[int, int, int, int]] = []
        for key in ranked:
            acc = bucket_sums[key]
            count = max(1, int(acc[4]))
            palette_colors.append((
                max(0, min(255, int(round(acc[0] / count)))),
                max(0, min(255, int(round(acc[1] / count)))),
                max(0, min(255, int(round(acc[2] / count)))),
                max(0, min(255, int(round(acc[3] / count)))),
            ))
        while len(palette_colors) < 256:
            palette_colors.append((0, 0, 0, 0))

        bucket_to_index = {key: index for index, key in enumerate(ranked[:256])}

        def nearest_index(color: bytes) -> int:
            r, g, b, a = (int(color[0]), int(color[1]), int(color[2]), int(color[3]))
            best_index = 0
            best_score = None
            for index, pal in enumerate(palette_colors[:256]):
                pr, pg, pb, pa = pal
                score = ((r - pr) * (r - pr)) + ((g - pg) * (g - pg)) + ((b - pb) * (b - pb)) + (((a - pa) * (a - pa)) * 2)
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = index
                    if score == 0:
                        break
            return int(best_index)

        indices = bytearray(pixel_count)
        for pixel_index, color in enumerate(samples):
            r, g, b, a = color
            if a < 8:
                key = (0, 0, 0, 0)
            else:
                key = (int(r) >> 3, int(g) >> 3, int(b) >> 3, 1 if int(a) >= 128 else 0)
            palette_index = bucket_to_index.get(key)
            if palette_index is None:
                palette_index = nearest_index(color)
            indices[pixel_index] = int(palette_index) & 0xFF

        palette = bytearray(1024)
        for palette_index, color in enumerate(palette_colors[:256]):
            palette[palette_index * 4:palette_index * 4 + 4] = bytes(color)
        return bytes(indices), bytes(palette)


    @staticmethod
    def _try_make_indexed8(rgba: bytes, width: int, height: int) -> Optional[Tuple[bytes, bytes]]:
        palette_map: Dict[bytes, int] = {}
        indices = bytearray(width * height)
        for pixel in range(width * height):
            color = rgba[pixel * 4:pixel * 4 + 4]
            palette_index = palette_map.get(color)
            if palette_index is None:
                if len(palette_map) >= 256:
                    return None
                palette_index = len(palette_map)
                palette_map[bytes(color)] = palette_index
            indices[pixel] = int(palette_index)
        palette = bytearray(1024)
        for color, palette_index in palette_map.items():
            palette[palette_index * 4:palette_index * 4 + 4] = color
        return bytes(indices), bytes(palette)
