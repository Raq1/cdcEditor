from __future__ import annotations

import shutil
import struct
import tempfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from ...core.binary import BinaryReader
from ...core.log import logger


@dataclass(slots=True)
class DRMRelocationEntry:
    type_and_section_info: int
    type_specific: int
    offset: int
    referenced_offset: int | None = None


@dataclass(slots=True)
class DRMSectionEntry:
    index: int
    size: int
    section_type: int
    pad: int
    version_id: int
    packed_data: int
    has_debug_info: int
    resource_type: int
    num_relocations: int
    section_id: int
    spec_mask: int
    relocations: List[DRMRelocationEntry] = field(default_factory=list)
    content_offset: int = 0
    drm_version: int = 0
    tr8_relocation_table_size: int = 0
    tr8_relocation_table: bytes = b''


class DRMContainerParser:

    STANDALONE_MAGIC = b'SECT'
    DERICKW_MAGIC = b'derickw\0'
    DERICKW_CHUNK_SIZE = 0x6000
    DERICKW_HEADER_SIZE = 20
    DERICKW_OUT_SIZE = 0x30000
    CDRM_MAGIC = b'CDRM'
    CDRM_MAGIC_BE = b'MRDC'
    CDRM_ALIGNMENT = 0x10
    CDRM_MAX_COMPRESSED_BLOCKS = 0x00FFFFFF
    CDRM_FLAG_UNCOMPRESSED = 1
    CDRM_FLAG_COMPRESSED = 2

    @staticmethod
    def _relocation_count_candidates(packed_data: int) -> list[int]:
        packed = int(packed_data) & 0xFFFFFFFF
        candidates = [
            (packed >> 8) & 0x00FFFFFF,
            (packed >> 8) & 0x0000FFFF,
            (packed >> 16) & 0x0000FFFF,
            packed & 0x0000FFFF,
            packed & 0x000000FF,
            0,
        ]
        result: list[int] = []
        for value in candidates:
            value = int(value)
            if value < 0 or value in result:
                continue
            result.append(value)
        return result

    @staticmethod
    def _choose_relocation_count_for_container(packed_data: int, section_start: int, section_size: int, file_size: int) -> int:
        default_count = (int(packed_data) >> 8) & 0x00FFFFFF
        for count in DRMContainerParser._relocation_count_candidates(packed_data):
            content_offset = int(section_start) + (int(count) * 8)
            end_offset = content_offset + max(0, int(section_size))
            if content_offset <= int(file_size) and end_offset <= int(file_size):
                if int(count) != int(default_count):
                    logger.warning(
                        'Adjusted DRM relocation count decode at payload offset 0x%X from %d to %d because the default table does not fit file_size=0x%X',
                        int(section_start),
                        int(default_count),
                        int(count),
                        int(file_size),
                    )
                return int(count)
        return int(default_count)

    @staticmethod
    def _normalise_packed_data_relocation_count(packed_data: int, relocation_count: int) -> int:
        return (int(packed_data) & 0xFF) | ((int(relocation_count) & 0x00FFFFFF) << 8)

    def __init__(self, filepath: str, endian: str = "<", decompress_derickw: bool = False):
        if endian not in ("<", ">"):
            raise ValueError(f"Unsupported endianness: {endian}")
        self.filepath = Path(filepath)
        self.endian = endian
        self.decompress_derickw = bool(decompress_derickw)


    @staticmethod
    def looks_like_derickw_file(filepath: str | Path) -> bool:
        try:
            with open(filepath, 'rb') as fh:
                return fh.read(8) == DRMContainerParser.DERICKW_MAGIC
        except Exception:
            return False

    @staticmethod
    def looks_like_cdrm_file(filepath: str | Path) -> bool:
        try:
            with open(filepath, 'rb') as fh:
                return fh.read(4) in {DRMContainerParser.CDRM_MAGIC, DRMContainerParser.CDRM_MAGIC_BE}
        except Exception:
            return False

    @staticmethod
    def _looks_like_cdrm_bytes(data: bytes) -> bool:
        return bytes(data[:4]) in {DRMContainerParser.CDRM_MAGIC, DRMContainerParser.CDRM_MAGIC_BE}

    @staticmethod
    def _align(value: int, alignment: int = 0x10) -> int:
        alignment = max(1, int(alignment))
        return (int(value) + alignment - 1) & ~(alignment - 1)

    @classmethod
    def decompress_cdrm_bytes(cls, data: bytes) -> bytes:
        if not cls._looks_like_cdrm_bytes(data):
            return data
        if len(data) < 0x10:
            raise ValueError('CDRM file is too small for a block header')

        magic, version, num_blocks, _padding = struct.unpack_from('<4sIII', data, 0)
        if magic not in {cls.CDRM_MAGIC, cls.CDRM_MAGIC_BE}:
            raise ValueError(f'Unsupported CDRM magic: {magic!r}')
        if num_blocks <= 0 or num_blocks > cls.CDRM_MAX_COMPRESSED_BLOCKS:
            raise ValueError(f'Unreasonable CDRM block count: {num_blocks}')

        table_end = 0x10 + (int(num_blocks) * 8)
        if table_end > len(data):
            raise ValueError('CDRM block table exceeds file size')

        entries: list[tuple[int, int, int]] = []
        pos = 0x10
        for block_index in range(int(num_blocks)):
            packed_size, compressed_size = struct.unpack_from('<II', data, pos)
            pos += 8
            block_type = int(packed_size) & 0xFF
            uncompressed_size = int(packed_size) >> 8
            if uncompressed_size < 0 or compressed_size < 0:
                raise ValueError(f'Invalid CDRM block sizes at block {block_index}')
            entries.append((block_type, uncompressed_size, int(compressed_size)))

        payload_pos = cls._align(table_end, cls.CDRM_ALIGNMENT)
        output = bytearray()
        for block_index, (block_type, uncompressed_size, compressed_size) in enumerate(entries):
            if payload_pos + compressed_size > len(data):
                raise ValueError(f'CDRM block {block_index} exceeds file size')
            payload = data[payload_pos:payload_pos + compressed_size]
            payload_pos = cls._align(payload_pos + compressed_size, cls.CDRM_ALIGNMENT)

            if block_type == cls.CDRM_FLAG_COMPRESSED:
                decoded = zlib.decompress(payload)
            elif block_type == cls.CDRM_FLAG_UNCOMPRESSED:
                decoded = payload
            elif block_type == 0 and uncompressed_size == 0 and compressed_size == 0:
                decoded = b''
            else:
                raise ValueError(f'Unsupported CDRM block compression flag {block_type} at block {block_index}')

            if len(decoded) != uncompressed_size:
                logger.warning(
                    'CDRM block %d decoded size mismatch: got 0x%X, expected 0x%X; normalising length',
                    block_index,
                    len(decoded),
                    uncompressed_size,
                )
                if len(decoded) > uncompressed_size:
                    decoded = decoded[:uncompressed_size]
                else:
                    decoded = decoded + (b'\0' * (uncompressed_size - len(decoded)))
            output.extend(decoded)

        logger.debug('Decompressed CDRM version=%d blocks=%d output_size=0x%X', int(version), int(num_blocks), len(output))
        return bytes(output)

    @staticmethod
    def _lzf_decompress_chunk(payload: bytes, out_len: int) -> bytes:
        ip = 0
        out = bytearray()
        in_len = len(payload)
        while ip < in_len:
            ctrl = payload[ip]
            ip += 1
            if ctrl < (1 << 5):
                literal_len = ctrl + 1
                if ip + literal_len > in_len:
                    raise ValueError('Truncated derickw/LZF literal run')
                if len(out) + literal_len > out_len:
                    raise ValueError('derickw/LZF output buffer exceeded')
                out.extend(payload[ip:ip + literal_len])
                ip += literal_len
                continue

            match_len = ctrl >> 5
            if match_len == 7:
                if ip >= in_len:
                    raise ValueError('Truncated derickw/LZF extended length')
                match_len += payload[ip]
                ip += 1
            if ip >= in_len:
                raise ValueError('Truncated derickw/LZF back-reference offset')
            ref = len(out) - ((ctrl & 0x1F) << 8) - 1
            ref -= payload[ip]
            ip += 1
            if ref < 0:
                raise ValueError('Invalid derickw/LZF back-reference')

            match_len += 2
            if len(out) + match_len > out_len:
                raise ValueError('derickw/LZF output buffer exceeded')
            for _ in range(match_len):
                out.append(out[ref])
                ref += 1
        return bytes(out)

    @classmethod
    def decompress_derickw_bytes(cls, data: bytes) -> bytes:
        if not data.startswith(cls.DERICKW_MAGIC):
            return data
        output = bytearray()
        for chunk_offset in range(0, len(data), cls.DERICKW_CHUNK_SIZE):
            chunk = data[chunk_offset:chunk_offset + cls.DERICKW_CHUNK_SIZE]
            if not chunk:
                break
            if len(chunk) < cls.DERICKW_HEADER_SIZE or chunk[:8] != cls.DERICKW_MAGIC:
                raise ValueError(f'Invalid derickw chunk at 0x{chunk_offset:X}')
            output.extend(cls._lzf_decompress_chunk(chunk[cls.DERICKW_HEADER_SIZE:], cls.DERICKW_OUT_SIZE))
        return bytes(output)

    @contextmanager
    def _temporary_decoded_source(self):
        needs_decode = False
        try:
            with open(self.filepath, 'rb') as fh:
                prefix = fh.read(8)
            needs_decode = self._looks_like_cdrm_bytes(prefix) or (self.decompress_derickw and prefix == self.DERICKW_MAGIC)
        except Exception:
            needs_decode = False

        if not needs_decode:
            yield self.filepath
            return

        temp_dir = Path(tempfile.mkdtemp(prefix='trlau_editor_drm_decode_'))
        try:
            with open(self.filepath, 'rb') as fh:
                decoded = fh.read()
            stages: list[str] = []
            guard = 0
            while guard < 4:
                guard += 1
                if self.decompress_derickw and decoded.startswith(self.DERICKW_MAGIC):
                    decoded = self.decompress_derickw_bytes(decoded)
                    stages.append('derickw')
                    continue
                if self._looks_like_cdrm_bytes(decoded):
                    magic_name = 'CDRM' if decoded[:4] == self.CDRM_MAGIC else 'MRDC'
                    decoded = self.decompress_cdrm_bytes(decoded)
                    stages.append(magic_name)
                    continue
                break

            out_path = temp_dir / self.filepath.name
            with open(out_path, 'wb') as fh:
                fh.write(decoded)
            logger.debug(
                'Decoded DRM source %s via %s -> %s (%d bytes)',
                self.filepath.name,
                '+'.join(stages) or 'none',
                out_path,
                len(decoded),
            )
            yield out_path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @contextmanager
    def _temporary_derickw_source(self):
        with self._temporary_decoded_source() as source_path:
            yield source_path

    @staticmethod
    def _decode_tr8_extern_relocation(raw: int) -> tuple[int, int, int]:
        section_index = int(raw) & 0x3FFF
        offset = (int(raw) >> 14) & 0x00FFFFFF
        referenced_offset = (int(raw) >> 38) & 0x03FFFFFF
        return section_index, offset, referenced_offset

    def parse(self) -> List[DRMSectionEntry]:
        with self._temporary_decoded_source() as source_path:
            if source_path != self.filepath:
                return DRMContainerParser(str(source_path), endian=self.endian, decompress_derickw=False).parse()

        sections: List[DRMSectionEntry] = []
        with open(self.filepath, 'rb') as fh:
            file_prefix = fh.read(4)
            if self._looks_like_cdrm_bytes(file_prefix):
                raise ValueError(f'Raw CDRM wrapper was not decompressed before DRM parsing: magic={file_prefix!r}')
            fh.seek(0)
            br = BinaryReader(fh, endian=self.endian)

            version_number = br.i32()
            is_tr8 = self.endian == '<' and int(version_number) == 19
            tr8_name_size = 0
            tr8_padding_size = 0
            tr8_unknown0 = 0
            tr8_unknown1 = 0
            if is_tr8:
                tr8_name_size = br.u32()
                tr8_padding_size = br.u32()
                tr8_unknown0 = br.u32()
                tr8_unknown1 = br.u32()
                num_sections = br.u32()
            else:
                num_sections = br.i32()
            logger.debug(
                'Parsing DRM %s version=%d num_sections=%d tr8=%s name_size=0x%X padding_size=0x%X unk0=0x%X unk1=0x%X',
                self.filepath.name,
                version_number,
                num_sections,
                is_tr8,
                tr8_name_size,
                tr8_padding_size,
                tr8_unknown0,
                tr8_unknown1,
            )

            for index in range(max(0, num_sections)):
                size = br.i32()
                section_type = br.u8()
                pad = br.u8()
                version_id = br.u16()
                packed_data = br.u32()
                has_debug_info = packed_data & 0x1
                resource_type = (packed_data >> 1) & 0x7F
                encoded_count_or_size = (packed_data >> 8) & 0x00FFFFFF
                section_id = br.u32()
                spec_mask = br.u32()
                sections.append(
                    DRMSectionEntry(
                        index=index,
                        size=size,
                        section_type=section_type,
                        pad=pad,
                        version_id=version_id,
                        packed_data=packed_data,
                        has_debug_info=has_debug_info,
                        resource_type=resource_type,
                        num_relocations=0 if is_tr8 else encoded_count_or_size,
                        section_id=section_id,
                        spec_mask=spec_mask,
                        drm_version=int(version_number),
                        tr8_relocation_table_size=encoded_count_or_size if is_tr8 else 0,
                    )
                )

            file_size = self.filepath.stat().st_size
            section_header_size = 24 if is_tr8 else 8
            section_payload_start = section_header_size + (len(sections) * 20)
            if is_tr8:
                section_payload_start += max(0, int(tr8_name_size)) + max(0, int(tr8_padding_size))
                if section_payload_start > file_size:
                    raise EOFError(
                        f'TR8 DRM header/name/padding area exceeds file size: '
                        f'0x{section_payload_start:X} > 0x{file_size:X}'
                    )

            if is_tr8:
                current_offset = section_payload_start
                for section in sections:
                    table_size = max(0, int(section.tr8_relocation_table_size))
                    relocations: List[DRMRelocationEntry] = []
                    table_blob = b''
                    if table_size:
                        if current_offset + table_size > file_size:
                            raise EOFError(
                                f'TR8 relocation table for section {section.index} exceeds file size: '
                                f'0x{current_offset + table_size:X} > 0x{file_size:X}'
                            )
                        fh.seek(current_offset)
                        table_blob = fh.read(table_size)
                        if len(table_blob) != table_size:
                            raise EOFError(f'TR8 relocation table for section {section.index} is truncated')
                        br.seek(current_offset)
                        counts = [br.u32() for _ in range(5)]
                        expected_table_size = 20 + counts[0] * 8 + counts[1] * 8 + counts[2] * 4 + counts[3] * 4 + counts[4] * 4
                        if expected_table_size > table_size:
                            raise ValueError(
                                f'TR8 relocation counts for section {section.index} require 0x{expected_table_size:X} bytes, '
                                f'but header stores 0x{table_size:X}'
                            )

                        for _ in range(counts[0]):
                            field_offset = br.u32()
                            _referenced_offset = br.u32()
                            relocations.append(
                                DRMRelocationEntry(
                                    type_and_section_info=(int(section.index) << 3),
                                    type_specific=0,
                                    offset=field_offset,
                                    referenced_offset=_referenced_offset,
                                )
                            )
                        for _ in range(counts[1]):
                            raw = int.from_bytes(br.read(8), 'little', signed=False)
                            target_section_index, field_offset, _referenced_offset = self._decode_tr8_extern_relocation(raw)
                            relocations.append(
                                DRMRelocationEntry(
                                    type_and_section_info=(int(target_section_index) << 3),
                                    type_specific=0,
                                    offset=field_offset,
                                    referenced_offset=_referenced_offset,
                                )
                            )

                        for count in counts[2:]:
                            br.skip(int(count) * 4)

                    section.relocations = relocations
                    section.num_relocations = len(relocations)
                    section.tr8_relocation_table = bytes(table_blob)
                    section.content_offset = current_offset + table_size
                    current_offset = section.content_offset + max(0, int(section.size))
                logger.debug(
                    'Parsed TR8 DRM sections: sections=%d total_relocation_table_size=0x%X converted_relocations=%d content_end=0x%X',
                    len(sections),
                    sum(max(0, int(section.tr8_relocation_table_size)) for section in sections),
                    sum(max(0, int(section.num_relocations)) for section in sections),
                    current_offset,
                )
                return sections

            ps3_count_total_end = section_payload_start + sum(
                max(0, int(section.size)) + (((int(section.packed_data) & 0xFFFF) * 8) if self.endian == '>' else 0)
                for section in sections
            )
            use_ps3_relocation_count = self.endian == '>' and bool(sections) and ps3_count_total_end == file_size
            if use_ps3_relocation_count:
                logger.debug(
                    'Using PS3 16-bit relocation-count decode for %s because section sizes plus relocation tables match file size 0x%X',
                    self.filepath.name,
                    file_size,
                )

            current_offset = section_payload_start
            for section in sections:
                if use_ps3_relocation_count:
                    adjusted_relocations = int(section.packed_data) & 0xFFFF
                else:
                    adjusted_relocations = self._choose_relocation_count_for_container(
                        section.packed_data,
                        current_offset,
                        section.size,
                        file_size,
                    )
                    if adjusted_relocations != section.num_relocations:
                        section.packed_data = self._normalise_packed_data_relocation_count(section.packed_data, adjusted_relocations)

                if adjusted_relocations != section.num_relocations:
                    section.num_relocations = int(adjusted_relocations)
                br.seek(current_offset)
                relocations: List[DRMRelocationEntry] = []
                for _ in range(section.num_relocations):
                    relocations.append(
                        DRMRelocationEntry(
                            type_and_section_info=br.u16(),
                            type_specific=br.i16(),
                            offset=br.u32(),
                        )
                    )
                section.relocations = relocations
                section.content_offset = current_offset + (section.num_relocations * 8)
                current_offset = section.content_offset + section.size
            logger.debug(
                'Parsed DRM sections: sections=%d relocations=%d content_end=0x%X',
                len(sections),
                sum(max(0, int(section.num_relocations)) for section in sections),
                current_offset,
            )

        return sections


    def _section_relocation_at(self, section: DRMSectionEntry, field_local_offset: int) -> DRMRelocationEntry | None:
        for relocation in section.relocations:
            if int(relocation.offset) == int(field_local_offset):
                return relocation
        return None

    @staticmethod
    def _relocation_target_section_index(relocation: DRMRelocationEntry) -> int:
        return (int(relocation.type_and_section_info) >> 3) & 0x1FFF

    def _read_section_u32(self, fh, section: DRMSectionEntry, field_local_offset: int) -> int:
        if field_local_offset < 0 or field_local_offset + 4 > int(section.size):
            return 0
        byteorder = 'big' if self.endian == '>' else 'little'
        fh.seek(int(section.content_offset) + int(field_local_offset))
        data = fh.read(4)
        if len(data) != 4:
            return 0
        return int.from_bytes(data, byteorder, signed=False)

    def _read_section_i32(self, fh, section: DRMSectionEntry, field_local_offset: int) -> int:
        if field_local_offset < 0 or field_local_offset + 4 > int(section.size):
            return 0
        byteorder = 'big' if self.endian == '>' else 'little'
        fh.seek(int(section.content_offset) + int(field_local_offset))
        data = fh.read(4)
        if len(data) != 4:
            return 0
        return int.from_bytes(data, byteorder, signed=True)

    def _drm_pointer_field_plausible(
        self,
        sections_by_index: dict[int, DRMSectionEntry],
        section: DRMSectionEntry,
        field_local_offset: int,
        raw_value: int,
    ) -> bool:
        relocation = self._section_relocation_at(section, field_local_offset)
        if relocation is not None:
            target = sections_by_index.get(self._relocation_target_section_index(relocation))
            return target is not None and 0 <= int(raw_value) <= max(0, int(target.size))
        return int(raw_value) != 0 and 0 <= int(raw_value) < max(0, int(section.size))

    def _looks_like_model_section(self, fh, sections_by_index: dict[int, DRMSectionEntry], section: DRMSectionEntry) -> bool:
        if int(section.size) < 0x6C:
            return False

        version = self._read_section_i32(fh, section, 0x00)
        num_segments = self._read_section_i32(fh, section, 0x04)
        num_virt_segments = self._read_section_i32(fh, section, 0x08)
        segment_list_raw = self._read_section_u32(fh, section, 0x0C)

        if not (-1 <= int(version) <= 0x10000):
            return False
        if not (0 < int(num_segments) <= 4096):
            return False
        if not (0 <= int(num_virt_segments) <= 16384):
            return False
        return self._drm_pointer_field_plausible(sections_by_index, section, 0x0C, segment_list_raw)

    def _detect_big_endian_game_from_external_streams(self, fh, sections: List[DRMSectionEntry]) -> str | None:
        if self.endian != '>':
            return None

        sections_by_index = {int(section.index): section for section in sections}
        for section in sections:
            if not self._looks_like_model_section(fh, sections_by_index, section):
                continue

            external_index_raw = self._read_section_u32(fh, section, 0x64)
            external_vertex_header_raw = self._read_section_u32(fh, section, 0x68)
            has_external_index = self._drm_pointer_field_plausible(sections_by_index, section, 0x64, external_index_raw)
            has_external_vertex_header = self._drm_pointer_field_plausible(sections_by_index, section, 0x68, external_vertex_header_raw)
            if has_external_index and has_external_vertex_header:
                logger.debug(
                    'Detected Legend DRM from alternate external render stream in section[%d] externalIndex=0x%X externalVertexHeader=0x%X',
                    int(section.index),
                    int(external_index_raw),
                    int(external_vertex_header_raw),
                )
                return 'legend'

        return 'anniversary'

    def detect_game(self, sections: List[DRMSectionEntry] | None = None) -> str:
        sections = sections or self.parse()
        if not sections:
            return 'anniversary'
        if int(getattr(sections[0], 'drm_version', 0)) == 19:
            return 'underworld'

        try:
            with self._temporary_derickw_source() as source_path:
                with open(source_path, 'rb') as fh:
                    external_stream_game = self._detect_big_endian_game_from_external_streams(fh, sections)
                    if external_stream_game is not None:
                        return external_stream_game

                    root_section = sections[0]
                    fh.seek(int(root_section.content_offset) + 80)
                    cdc_render_data_id = int.from_bytes(fh.read(4), 'little', signed=False)
        except Exception:
            cdc_render_data_id = 0

        return 'legend' if cdc_render_data_id != 0 else 'anniversary'

    def extract_sections(self) -> tuple[Path, List[Path], List[DRMSectionEntry]]:
        return self.extract_sections_to(self._make_output_dir(), clear=True)

    @contextmanager
    def temporary_extract_sections(self):
        output_dir = Path(tempfile.mkdtemp(prefix='trlau_editor_drm_'))
        try:
            with self._temporary_decoded_source() as source_path:
                if source_path != self.filepath:
                    yield DRMContainerParser(str(source_path), endian=self.endian, decompress_derickw=False).extract_sections_to(output_dir, clear=False)
                else:
                    yield self.extract_sections_to(output_dir, clear=False)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def extract_sections_to(self, output_dir: str | Path, *, clear: bool = True) -> tuple[Path, List[Path], List[DRMSectionEntry]]:
        with self._temporary_decoded_source() as source_path:
            if source_path != self.filepath:
                return DRMContainerParser(str(source_path), endian=self.endian, decompress_derickw=False).extract_sections_to(output_dir, clear=clear)

        sections = self.parse()
        output_dir = Path(output_dir)
        if clear and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted_paths: List[Path] = []
        level_section_index = self.find_level_section_index(sections)
        with open(self.filepath, 'rb') as src:
            for section in sections:
                out_path = output_dir / self._make_section_filename(section, level_section_index=level_section_index)
                src.seek(section.content_offset)
                content = src.read(section.size)
                if len(content) != section.size:
                    raise EOFError(
                        f'Truncated DRM section {section.index}: expected {section.size} bytes, got {len(content)}'
                    )
                content = bytearray(content)
                if int(getattr(section, 'drm_version', 0)) == 19:
                    for reloc in section.relocations:
                        referenced_offset = getattr(reloc, 'referenced_offset', None)
                        if referenced_offset is None:
                            continue
                        field_offset = int(reloc.offset)
                        if 0 <= field_offset <= len(content) - 4:
                            content[field_offset:field_offset + 4] = int(referenced_offset).to_bytes(4, 'little', signed=False)

                standalone_size = 24 + (section.num_relocations * 8) + section.size
                with open(out_path, 'wb') as dst:
                    byteorder = 'little' if self.endian == '<' else 'big'
                    dst.write(self.STANDALONE_MAGIC)
                    dst.write(int(standalone_size).to_bytes(4, byteorder, signed=True))
                    dst.write(int(section.section_type).to_bytes(1, 'little', signed=False))
                    dst.write(int(section.pad).to_bytes(1, 'little', signed=False))
                    dst.write(int(section.version_id).to_bytes(2, byteorder, signed=False))
                    packed_data_out = int(section.packed_data)
                    if int(getattr(section, 'drm_version', 0)) == 19:
                        packed_data_out = self._normalise_packed_data_relocation_count(section.packed_data, section.num_relocations)
                    dst.write(int(packed_data_out).to_bytes(4, byteorder, signed=False))
                    dst.write(int(section.section_id).to_bytes(4, byteorder, signed=False))
                    dst.write(int(section.spec_mask).to_bytes(4, byteorder, signed=False))
                    for reloc in section.relocations:
                        dst.write(int(reloc.type_and_section_info).to_bytes(2, byteorder, signed=False))
                        dst.write(int(reloc.type_specific).to_bytes(2, byteorder, signed=True))
                        dst.write(int(reloc.offset).to_bytes(4, byteorder, signed=False))
                    dst.write(content)
                extracted_paths.append(out_path)

        logger.debug('Extracted %d DRM sections into %s', len(extracted_paths), output_dir)
        return output_dir, extracted_paths, sections

    def _make_output_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix=f'trlau_editor_drm_{self.filepath.stem}_'))

    def find_level_section_index(self, sections: List[DRMSectionEntry] | None = None) -> int | None:
        sections = sections or self.parse()
        if not sections:
            return None
        try:
            with open(self.filepath, 'rb') as fh:
                br = BinaryReader(fh, endian=self.endian)
                file_size = self.filepath.stat().st_size
                for section in sections:
                    sig_offset = section.content_offset + 168
                    if sig_offset + 4 > file_size:
                        continue
                    br.seek(sig_offset)
                    if br.u32() != 0x04C204BB:
                        continue

                    return section.index
        except Exception:
            logger.exception('Failed while probing DRM %s for level sections', self.filepath)
        return None

    def _make_section_filename(self, section: DRMSectionEntry, level_section_index: int | None = None) -> str:
        suffix = '.gnc'
        if section.section_type == 2:
            suffix = '.ani'
        elif section.section_type == 5:
            suffix = '.pcd'
        elif section.section_type == 12:
            suffix = '.tr8mesh'
        elif int(getattr(section, 'drm_version', 0)) == 19 and section.section_type == 10:
            suffix = '.matd'
        elif int(getattr(section, 'drm_version', 0)) == 19 and section.section_type == 9:
            suffix = '.shad'
        elif level_section_index is not None and section.index == level_section_index:
            suffix = '.level'
        elif section.index == 0:
            suffix = '.obj'

        if int(getattr(section, 'drm_version', 0)) == 19:
            return f'{section.index}_{section.section_id:x}{suffix}'
        if suffix == '.pcd':
            return f'{section.index}_{section.section_id:x}{suffix}'
        return f'{section.index}_0{suffix}'



def _section_index_from_standalone_name(path: Path) -> int:
    stem = path.stem
    head = stem.split('_', 1)[0]
    try:
        return int(head)
    except Exception as exc:
        raise ValueError(f'Cannot infer DRM section index from {path.name}') from exc



def _read_standalone_section_parts(path: Path, *, endian: str = '<'):
    blob = Path(path).read_bytes()
    if len(blob) < 24:
        raise ValueError(f'{Path(path).name} is too small to be a standalone section')
    if blob[:4] not in {b'SECT', b'TCES'}:
        raise ValueError(f'{Path(path).name} does not start with a SECT/TCES header')
    try:
        _stored_size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from(f'{endian}iBBHIII', blob, 4)
    except struct.error as exc:
        raise ValueError(f'{Path(path).name} has a truncated section header') from exc
    num_relocations = (int(packed_data) >> 8) & 0x00FFFFFF
    relocation_start = 24
    relocation_end = relocation_start + (num_relocations * 8)
    if relocation_end > len(blob):
        raise ValueError(f'{Path(path).name} relocation table exceeds file size')
    relocation_blob = bytes(blob[relocation_start:relocation_end])
    data = bytes(blob[relocation_end:])
    return int(section_type), int(skip_flags), int(version_id), int(packed_data), int(section_id), int(spec_mask), relocation_blob, data


def _u32_from(data: bytes, offset: int, *, endian: str = '<') -> int:
    if int(offset) < 0 or int(offset) + 4 > len(data):
        return 0
    return int(struct.unpack_from(f'{endian}I', data, int(offset))[0])


def _tr8_relocation_tail_from_source(source_section) -> tuple[tuple[int, int, int], bytes]:
    table = bytes(getattr(source_section, 'tr8_relocation_table', b'') or b'') if source_section is not None else b''
    if len(table) < 20:
        return (0, 0, 0), b''
    try:
        c0, c1, c2, c3, c4 = struct.unpack_from('<IIIII', table, 0)
    except struct.error:
        return (0, 0, 0), b''
    tail_start = 20 + int(c0) * 8 + int(c1) * 8
    if tail_start > len(table):
        return (0, 0, 0), b''
    return (int(c2), int(c3), int(c4)), bytes(table[tail_start:])


def _build_tr8_relocation_table_from_standalone(section_index: int, relocation_blob: bytes, data: bytes, source_section=None) -> bytes:
    internal: list[tuple[int, int]] = []
    external: list[tuple[int, int, int]] = []
    for offset in range(0, len(relocation_blob), 8):
        if offset + 8 > len(relocation_blob):
            break
        type_and_section, _type_specific, field_offset = struct.unpack_from('<HhI', relocation_blob, offset)
        relocation_type = int(type_and_section) & 0x7
        if relocation_type != 0:
            continue
        target_section_index = (int(type_and_section) >> 3) & 0x1FFF
        field_offset = int(field_offset)
        referenced_offset = _u32_from(data, field_offset, endian='<') & 0x03FFFFFF
        if int(target_section_index) == int(section_index):
            internal.append((field_offset & 0xFFFFFFFF, referenced_offset & 0xFFFFFFFF))
        else:
            external.append((target_section_index & 0x3FFF, field_offset & 0x00FFFFFF, referenced_offset & 0x03FFFFFF))

    (c2, c3, c4), tail = _tr8_relocation_tail_from_source(source_section)
    expected_tail_size = int(c2) * 4 + int(c3) * 4 + int(c4) * 4
    if expected_tail_size and len(tail) < expected_tail_size:
        c2 = c3 = c4 = 0
        tail = b''
    elif expected_tail_size:
        tail = tail[:expected_tail_size]
    else:
        tail = b''

    out = bytearray()
    out.extend(struct.pack('<IIIII', len(internal), len(external), int(c2), int(c3), int(c4)))
    for field_offset, referenced_offset in internal:
        out.extend(struct.pack('<II', int(field_offset) & 0xFFFFFFFF, int(referenced_offset) & 0xFFFFFFFF))
    for target_section_index, field_offset, referenced_offset in external:
        raw = (int(target_section_index) & 0x3FFF) | ((int(field_offset) & 0x00FFFFFF) << 14) | ((int(referenced_offset) & 0x03FFFFFF) << 38)
        out.extend(int(raw).to_bytes(8, 'little', signed=False))
    out.extend(tail)
    return bytes(out)


def build_tr8_drm_from_standalone_sections(section_paths: Sequence[str | Path], *, source_sections: Sequence[DRMSectionEntry] | None = None, endian: str = '<') -> bytes:
    if endian != '<':
        raise ValueError('TR8 DRM writing is currently supported only for little-endian PC files')

    ordered_paths = sorted((Path(path) for path in section_paths), key=_section_index_from_standalone_name)
    if not ordered_paths:
        raise ValueError('No standalone DRM sections were found to pack')

    indexed = [(_section_index_from_standalone_name(path), path) for path in ordered_paths]
    actual = [index for index, _path in indexed]
    duplicate_indices = sorted({index for index in actual if actual.count(index) > 1})
    if duplicate_indices:
        duplicate_names = {
            index: [path.name for item_index, path in indexed if item_index == index]
            for index in duplicate_indices[:8]
        }
        raise ValueError(f'DRM section file indices must be unique; duplicates: {duplicate_names}')
    expected = list(range(len(indexed)))
    if actual != expected:
        actual_set = set(actual)
        max_index = max(actual) if actual else -1
        missing = [index for index in range(max_index + 1) if index not in actual_set]
        raise ValueError(f'DRM section files must be contiguous from 0; found {len(actual)} section file(s), highest index {max_index}, missing indices {missing[:32]}')

    records: list[tuple[int, int, int, int, int, int, int, bytes, bytes]] = []
    source_by_index = {int(getattr(section, 'index', -1)): section for section in (source_sections or [])}
    for section_index, path in indexed:
        section_type, skip_flags, version_id, packed_data, section_id, spec_mask, relocation_blob, data = _read_standalone_section_parts(path, endian=endian)
        source_section = source_by_index.get(int(section_index))
        tr8_table = _build_tr8_relocation_table_from_standalone(int(section_index), relocation_blob, data, source_section=source_section)
        fixed_packed_data = (int(packed_data) & 0xFF) | ((len(tr8_table) & 0x00FFFFFF) << 8)
        records.append((len(data), section_type, skip_flags, version_id, fixed_packed_data, section_id, spec_mask, tr8_table, data))

    payload = bytearray()
    payload.extend(struct.pack('<IIIIII', 19, 0, 0, 0, 0, len(records)))
    for size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask, _relocations, _data in records:
        payload.extend(struct.pack(
            '<iBBHIII',
            int(size),
            int(section_type) & 0xFF,
            int(skip_flags) & 0xFF,
            int(version_id) & 0xFFFF,
            int(packed_data) & 0xFFFFFFFF,
            int(section_id) & 0xFFFFFFFF,
            int(spec_mask) & 0xFFFFFFFF,
        ))
    for _size, _section_type, _skip_flags, _version_id, _packed_data, _section_id, _spec_mask, relocations, data in records:
        payload.extend(relocations)
        payload.extend(data)
    return bytes(payload)

def build_drm_from_standalone_sections(section_paths: Sequence[str | Path], *, endian: str = '<') -> bytes:
    if endian not in ('<', '>'):
        raise ValueError(f'Unsupported endianness: {endian}')

    ordered_paths = sorted((Path(path) for path in section_paths), key=_section_index_from_standalone_name)
    if not ordered_paths:
        raise ValueError('No standalone DRM sections were found to pack')

    indexed = [(_section_index_from_standalone_name(path), path) for path in ordered_paths]
    actual = [index for index, _path in indexed]
    duplicate_indices = sorted({index for index in actual if actual.count(index) > 1})
    if duplicate_indices:
        duplicate_names = {
            index: [path.name for item_index, path in indexed if item_index == index]
            for index in duplicate_indices[:8]
        }
        raise ValueError(f'DRM section file indices must be unique; duplicates: {duplicate_names}')
    expected = list(range(len(indexed)))
    if actual != expected:
        actual_set = set(actual)
        max_index = max(actual) if actual else -1
        missing = [index for index in range(max_index + 1) if index not in actual_set]
        tail = actual[-8:] if len(actual) > 32 else []
        details = f'found {len(actual)} section file(s), highest index {max_index}, missing indices {missing[:32]}'
        if len(missing) > 32:
            details += f' (+{len(missing) - 32} more)'
        if tail:
            details += f', first indices {actual[:32]}, last indices {tail}'
        else:
            details += f', indices {actual[:32]}'
        raise ValueError(f'DRM section files must be contiguous from 0; {details}')

    section_records: list[tuple[int, int, int, int, int, int, int, bytes, bytes]] = []
    for _index, path in indexed:
        blob = path.read_bytes()
        if len(blob) < 24:
            raise ValueError(f'{path.name} is too small to be a standalone section')
        if blob[:4] not in {b'SECT', b'TCES'}:
            raise ValueError(f'{path.name} does not start with a SECT/TCES header')
        try:
            _stored_size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from(f'{endian}iBBHIII', blob, 4)
        except struct.error as exc:
            raise ValueError(f'{path.name} has a truncated section header') from exc
        num_relocations = (int(packed_data) >> 8) & 0x00FFFFFF
        relocation_start = 24
        relocation_end = relocation_start + (num_relocations * 8)
        if relocation_end > len(blob):
            raise ValueError(f'{path.name} relocation table exceeds file size')
        relocation_blob = blob[relocation_start:relocation_end]
        data = blob[relocation_end:]
        fixed_packed_data = (int(packed_data) & 0xFF) | ((len(relocation_blob) // 8) << 8)
        section_records.append((len(data), int(section_type), int(skip_flags), int(version_id), int(fixed_packed_data), int(section_id), int(spec_mask), relocation_blob, data))

    payload = bytearray()
    payload.extend(struct.pack(f'{endian}iI', 14, len(section_records)))
    for size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask, _relocations, _data in section_records:
        payload.extend(struct.pack(
            f'{endian}IBBHIII',
            int(size),
            int(section_type) & 0xFF,
            int(skip_flags) & 0xFF,
            int(version_id) & 0xFFFF,
            int(packed_data) & 0xFFFFFFFF,
            int(section_id) & 0xFFFFFFFF,
            int(spec_mask) & 0xFFFFFFFF,
        ))
    for _size, _section_type, _skip_flags, _version_id, _packed_data, _section_id, _spec_mask, relocations, data in section_records:
        payload.extend(relocations)
        payload.extend(data)
    return bytes(payload)


def write_drm_from_standalone_directory(directory: str | Path, filepath: str | Path, *, endian: str = '<', source_sections: Sequence[DRMSectionEntry] | None = None) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(str(directory))
    section_paths: list[Path] = []
    for candidate in directory.iterdir():
        if not candidate.is_file():
            continue
        try:
            _section_index_from_standalone_name(candidate)
        except ValueError:
            continue
        if candidate.suffix.lower() not in {'.obj', '.gnc', '.ani', '.tr7aemesh', '.tr8mesh', '.pcd', '.level', '.matd', '.shad'}:
            continue
        section_paths.append(candidate)
    source_is_tr8 = bool(source_sections) and any(int(getattr(section, 'drm_version', 0)) == 19 for section in (source_sections or []))
    if source_is_tr8:
        payload = build_tr8_drm_from_standalone_sections(section_paths, source_sections=source_sections, endian=endian)
    else:
        payload = build_drm_from_standalone_sections(section_paths, endian=endian)
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return sorted(section_paths, key=_section_index_from_standalone_name)
