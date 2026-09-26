from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ...core.binary import BinaryReader
from ...core.log import logger
from ...core.model_types import RelocationEntry, SectionInfo


def _relocation_count_candidates(packed_data: int) -> list[int]:
    """Return plausible relocation-count decodes for known TR section headers.

    The usual LAU layout stores debug/resource flags in the low byte and a
    24-bit relocation count in bits 8..31.  Some platform/repacked sections seen
    during research can present the count in narrower subfields.  The default
    decode remains first; fallbacks are only used when the default relocation
    table cannot fit in the standalone section file.
    """
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


def _reader_file_size(br: BinaryReader) -> int | None:
    fh = getattr(br, '_fh', None)
    if fh is None:
        return None
    try:
        current = fh.tell()
        fh.seek(0, 2)
        size = int(fh.tell())
        fh.seek(current, 0)
        return size
    except Exception:
        return None


def _standalone_relocation_table_plausible(br: BinaryReader, header_start: int, file_size: int, count: int) -> bool:
    if count < 0:
        return False
    table_start = int(header_start) + 0x18
    table_end = table_start + (int(count) * 8)
    if table_end > int(file_size):
        return False
    if count == 0:
        return True

    data_size = max(0, int(file_size) - table_end)
    if data_size <= 0:
        return False
    previous = br.tell()
    try:
        br.seek(table_start)
        valid_offsets = 0
        for _index in range(int(count)):
            type_and_section_info = br.u16()
            _type_specific = br.u16()
            offset = br.u32()
            relocation_type = int(type_and_section_info) & 0x7
            if relocation_type > 7:
                return False
            if int(offset) < int(data_size):
                valid_offsets += 1
        return valid_offsets == int(count)
    except Exception:
        return False
    finally:
        try:
            br.seek(previous)
        except Exception:
            pass


def _choose_relocation_count_for_standalone(br: BinaryReader, packed_data: int, header_start: int, file_size: int | None) -> int:
    default_count = (int(packed_data) >> 8) & 0x00FFFFFF
    if file_size is None:
        return int(default_count)

    if getattr(br, 'endian', '<') == '>':
        ps3_count = int(packed_data) & 0xFFFF
        if ps3_count and _standalone_relocation_table_plausible(br, header_start, int(file_size), ps3_count):
            if ps3_count != default_count:
                logger.debug(
                    'Using PS3 16-bit relocation count at section offset 0x%X: %d instead of default %d',
                    int(header_start),
                    int(ps3_count),
                    int(default_count),
                )
            return int(ps3_count)

    for count in _relocation_count_candidates(packed_data):
        if int(header_start) + 0x18 + (int(count) * 8) <= int(file_size):
            if int(count) != int(default_count):
                logger.warning(
                    'Adjusted relocation count decode at section offset 0x%X from %d to %d because the default table does not fit file_size=0x%X',
                    int(header_start),
                    int(default_count),
                    int(count),
                    int(file_size),
                )
            return int(count)
    return int(default_count)


class TRSectionParser:
    @staticmethod
    def parse(br: BinaryReader) -> SectionInfo:
        magic = br.read(4)
        size = br.i32()
        section_type = br.u8()
        skip_flags = br.u8()
        version_id = br.u16()
        packed_data = br.u32()
        has_debug_info = packed_data & 0x1
        resource_type = (packed_data >> 1) & 0x7F
        num_relocations = (packed_data >> 8) & 0x00FFFFFF
        section_id = br.i32()
        spec_mask = br.u32()

        header_start = br.tell() - 0x18
        file_size = _reader_file_size(br)
        num_relocations = _choose_relocation_count_for_standalone(br, packed_data, header_start, file_size)

        relocations = []
        relocations_by_offset = {}
        for index in range(num_relocations):
            type_and_section_info = br.u16()
            relocation_type = type_and_section_info & 0x7
            section_index_or_type = (type_and_section_info >> 3) & 0x1FFF
            type_specific = br.u16()
            offset = br.u32()
            entry = RelocationEntry(
                type=relocation_type,
                section_index_or_type=section_index_or_type,
                type_specific=type_specific,
                offset=offset,
            )
            relocations.append(entry)
            relocations_by_offset[offset] = entry

        info_size = 0x18 + (num_relocations * 8)
        logger.debug(
            'Parsed section header: magic=%r size=%d type=%d skip_flags=0x%X version=%d has_debug_info=%d resource_type=%d relocations=%d info_size=0x%X id=%d spec_mask=0x%X',
            magic,
            size,
            section_type,
            skip_flags,
            version_id,
            has_debug_info,
            resource_type,
            num_relocations,
            info_size,
            section_id,
            spec_mask,
        )

        return SectionInfo(
            magic=magic,
            size=size,
            section_type=section_type,
            skip_flags=skip_flags,
            version_id=version_id,
            packed_data=packed_data,
            has_debug_info=has_debug_info,
            resource_type=resource_type,
            num_relocations=num_relocations,
            section_id=section_id,
            spec_mask=spec_mask,
            info_size=info_size,
            relocations=relocations,
            relocations_by_offset=relocations_by_offset,
        )


@dataclass(slots=True)
class SectionContext:
    filepath: Path
    reader: BinaryReader
    section_info: SectionInfo
    section_offset: int
    data_start: int
    data_end: int
    file_size: int

    @property
    def file_name(self) -> str:
        return self.filepath.name

    @property
    def data_size(self) -> int:
        return max(0, self.data_end - self.data_start)


class SectionContextCache:
    def __init__(self, root_filepath: str, endian: str = "<"):
        self.root_path = Path(root_filepath)
        self.base_dir = self.root_path.parent
        self.endian = endian
        self._contexts: Dict[Path, Tuple[object, SectionContext]] = {}

    def close(self) -> None:
        for fh, _ctx in self._contexts.values():
            try:
                fh.close()
            except Exception:
                pass
        self._contexts.clear()

    def get_root_context(self) -> SectionContext:
        return self.get_context(self.root_path)

    def get_context(self, path: Path) -> SectionContext:
        path = path.resolve()
        cached = self._contexts.get(path)
        if cached is not None:
            return cached[1]

        fh = open(path, 'rb')
        br = BinaryReader(fh, endian=self.endian)
        section_offset = 0
        section_info = TRSectionParser.parse(br)
        file_size = path.stat().st_size
        data_start = section_offset + section_info.info_size
        data_end = data_start + section_info.size
        if data_end > file_size and section_offset + section_info.size <= file_size:
            data_end = section_offset + section_info.size
        data_end = min(file_size, data_end)
        if data_end < data_start:
            logger.warning(
                'Section size is smaller than section info in %s: size=0x%X info_size=0x%X',
                path.name,
                section_info.size,
                section_info.info_size,
            )
            data_end = data_start

        ctx = SectionContext(
            filepath=path,
            reader=br,
            section_info=section_info,
            section_offset=section_offset,
            data_start=data_start,
            data_end=data_end,
            file_size=file_size,
        )
        self._contexts[path] = (fh, ctx)
        logger.debug(
            'Opened section context %s (type=%d, data_start=0x%X, data_end=0x%X, file_size=0x%X)',
            path.name,
            section_info.section_type,
            data_start,
            data_end,
            file_size,
        )
        return ctx

    def resolve_target_context(self, source_context: SectionContext, section_index_or_type: int) -> SectionContext:
        current_stem = source_context.filepath.stem
        source_suffix = source_context.filepath.suffix
        preferred_name = f'{section_index_or_type}_0{source_suffix}'
        preferred_path = (self.base_dir / preferred_name).resolve()

        if preferred_path.exists():
            logger.debug(
                'Resolved relocation target for sectionIndexOrType=%d using matching suffix: %s',
                section_index_or_type,
                preferred_path.name,
            )
            return self.get_context(preferred_path)

        target_stem = f'{section_index_or_type}_0'
        target_prefix = f'{section_index_or_type}_'
        for candidate in sorted(self.base_dir.iterdir()):
            if not candidate.is_file():
                continue
            if candidate.stem == target_stem or candidate.stem.startswith(target_prefix):
                logger.debug(
                    'Resolved relocation target for sectionIndexOrType=%d using first matching file: %s',
                    section_index_or_type,
                    candidate.name,
                )
                return self.get_context(candidate)

        fallback_path = (self.base_dir / f'{section_index_or_type}_0.gnc').resolve()
        logger.debug(
            'Falling back to .gnc relocation target for sectionIndexOrType=%d: %s',
            section_index_or_type,
            fallback_path.name,
        )
        return self.get_context(fallback_path)


def resolve_pointer(
    cache: SectionContextCache,
    source_context: SectionContext,
    field_local_offset: int,
    pointer_value: int,
) -> Tuple[SectionContext, int]:
    relocation = source_context.section_info.relocations_by_offset.get(field_local_offset)
    target_context = source_context
    if relocation is not None:
        target_context = cache.resolve_target_context(source_context, relocation.section_index_or_type)
        logger.debug(
            'Relocation hit in %s at field local offset 0x%X -> %s (raw=0x%X)',
            source_context.file_name,
            field_local_offset,
            target_context.file_name,
            pointer_value,
        )
    elif pointer_value == 0:
        logger.debug(
            'Pointer at field local offset 0x%X in %s is null (no relocation)',
            field_local_offset,
            source_context.file_name,
        )
        return source_context, 0

    absolute_offset = target_context.data_start + pointer_value
    logger.debug(
        'Resolved pointer from %s field local offset 0x%X with value 0x%X -> %s absolute 0x%X',
        source_context.file_name,
        field_local_offset,
        pointer_value,
        target_context.file_name,
        absolute_offset,
    )

    if pointer_value >= target_context.data_size:
        logger.warning(
            'Pointer value 0x%X exceeds target data size 0x%X in %s',
            pointer_value,
            target_context.data_size,
            target_context.file_name,
        )
    if absolute_offset >= target_context.file_size:
        logger.warning(
            'Absolute pointer 0x%X exceeds file size 0x%X in %s',
            absolute_offset,
            target_context.file_size,
            target_context.file_name,
        )

    return target_context, absolute_offset
