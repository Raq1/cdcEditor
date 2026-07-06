from __future__ import annotations

from typing import List, Tuple

from ...core.log import logger
from ...core.model_types import MVertex, ModelData, Segment
from ..common.section import SectionContext, SectionContextCache, resolve_pointer
from ..ps3.tr7ae_model import TRPS3ModelParser


class TRXbox360ModelParser(TRPS3ModelParser):

    X360_EXTERNAL_VERTEX_SIZE = 36

    @staticmethod
    def _decode_x360_external_color(record: bytes) -> Tuple[int, int, int, int] | None:
        if len(record) < 12:
            return None
        a = int(record[8]) & 0xFF
        r = int(record[9]) & 0xFF
        g = int(record[10]) & 0xFF
        b = int(record[11]) & 0xFF
        return (r, g, b, a)

    def _read_ps3_external_stream_vertices(
        self,
        cache: SectionContextCache,
        vertex_stream_ctx: SectionContext,
        vertex_header_abs: int,
        vertex_count: int,
        segments: List[Segment],
    ) -> Tuple[List[MVertex], int, int]:
        """Read the Xbox 360 external render-stream variant.

        The PS3 external reader treats the first word at +0x68 as a vertex-data
        pointer.  Xbox 360 files store a small header instead:

            +0x00 u32 vertex_count
            +0x04 u32 vertex_record_pointer

        The vertex records are interleaved 36-byte entries.  Reading this stream
        as PS3-style 16-byte primary records plus a detached 20-byte secondary
        stream makes the model explode into a cube of random triangles.
        """
        if vertex_header_abs <= 0 or vertex_header_abs + 8 > vertex_stream_ctx.file_size:
            return [], int(vertex_count), 0

        br = vertex_stream_ctx.reader
        previous = br.tell()
        try:
            br.seek(vertex_header_abs)
            header_count = int(br.u32())
            vertex_data_raw = int(br.u32())
            field_local_offset = int(vertex_header_abs) - int(vertex_stream_ctx.data_start) + 4
            vertex_data_ctx, vertex_data_abs = resolve_pointer(cache, vertex_stream_ctx, field_local_offset, vertex_data_raw)
        except Exception:
            try:
                br.seek(previous)
            except Exception:
                pass
            return [], int(vertex_count), 0
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

        count = int(header_count) if 0 < int(header_count) <= 10000000 else int(vertex_count)
        if count <= 0 or count > 10000000:
            return [], int(count), int(vertex_data_raw)

        required_end = int(vertex_data_abs) + (count * self.X360_EXTERNAL_VERTEX_SIZE)
        data_limit = min(int(vertex_data_ctx.data_end), int(vertex_data_ctx.file_size))
        if vertex_data_abs <= 0 or required_end > data_limit:
            direct_vertex_abs = int(vertex_header_abs)
            direct_limit = min(int(vertex_stream_ctx.data_end), int(vertex_stream_ctx.file_size))
            direct_available = max(0, int(direct_limit) - int(direct_vertex_abs))
            direct_max_count = int(direct_available) // self.X360_EXTERNAL_VERTEX_SIZE
            direct_count = min(int(vertex_count), int(direct_max_count)) if int(vertex_count) > 0 else int(direct_max_count)
            if direct_vertex_abs > 0 and direct_count > 0:
                logger.debug(
                    'Using direct Xbox 360 external vertex records in %s:0x%X because header pointer was invalid '
                    '(header_count=%d raw=0x%X resolved=0x%X count=%d data_end=0x%X); direct_count=%d',
                    vertex_stream_ctx.file_name,
                    int(direct_vertex_abs),
                    int(header_count),
                    int(vertex_data_raw),
                    int(vertex_data_abs),
                    int(count),
                    int(data_limit),
                    int(direct_count),
                )
                vertex_data_ctx = vertex_stream_ctx
                vertex_data_abs = direct_vertex_abs
                vertex_data_raw = int(direct_vertex_abs) - int(vertex_stream_ctx.data_start)
                count = int(direct_count)
            else:
                logger.debug(
                    'Skipping Xbox 360 external vertex stream; data outside bounds in %s abs=0x%X count=%d stride=%d end=0x%X data_end=0x%X',
                    vertex_data_ctx.file_name,
                    int(vertex_data_abs),
                    count,
                    self.X360_EXTERNAL_VERTEX_SIZE,
                    int(required_end),
                    int(data_limit),
                )
                return [], int(count), int(vertex_data_raw)

        segment_by_vertex = self._segment_lookup_from_ranges(segments)
        vertices: List[MVertex] = []
        vbr = vertex_data_ctx.reader
        vprev = vbr.tell()
        try:
            vbr.seek(int(vertex_data_abs))
            for index in range(count):
                record = vbr.read(self.X360_EXTERNAL_VERTEX_SIZE)
                x = int.from_bytes(record[0:2], 'big', signed=True)
                y = int.from_bytes(record[2:4], 'big', signed=True)
                z = int.from_bytes(record[4:6], 'big', signed=True)

                fallback_segment = int(segment_by_vertex.get(index, 0))
                ps3_color_rgba = self._decode_x360_external_color(record)

                uv_raw = (0, 0)
                uv_decoded = (0.0, 0.0)
                if len(record) >= 16:
                    uvx = int.from_bytes(record[12:14], 'big', signed=True)
                    uvy = int.from_bytes(record[14:16], 'big', signed=True)
                    uv_raw = (uvx, uvy)
                    uv_decoded = (float(uvx) / 4096.0, 1.0 - (float(uvy) / 4096.0))

                primary_slot = int(record[28]) if len(record) > 28 else fallback_segment
                secondary_slot = int(record[29]) if len(record) > 29 else primary_slot
                primary_weight_u8 = int(record[32]) if len(record) > 32 else 255
                secondary_weight_u8 = int(record[33]) if len(record) > 33 else 0
                total_weight = primary_weight_u8 + secondary_weight_u8
                secondary_weight = 0.0
                if total_weight > 0 and secondary_slot != primary_slot:
                    secondary_weight = float(secondary_weight_u8) / float(total_weight)

                vertices.append(
                    MVertex(
                        index=index,
                        position_raw=(x, y, z),
                        normal_raw=(0, 0, 127),
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

        logger.debug(
            'Read Xbox 360 external interleaved vertex stream from %s:0x%X (%d records, stride=%d)',
            vertex_data_ctx.file_name,
            int(vertex_data_abs),
            len(vertices),
            self.X360_EXTERNAL_VERTEX_SIZE,
        )
        return vertices, count, int(vertex_data_raw)

    def _parse_model(self, cache: SectionContextCache, context: SectionContext) -> ModelData:
        model = super()._parse_model(cache, context)
        model.uv_format = 'xbox360'
        return model
