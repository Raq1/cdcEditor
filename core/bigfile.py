from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import shutil
import struct
import zlib
import uuid
from functools import lru_cache
from typing import BinaryIO, Iterable

SECTOR_SIZE = 0x800
LEGEND_ALIGNMENT = 0x9600000
UNDERWORLD_ALIGNMENT = 0x7FF00000
BIGFILE_ALIGNMENT_CANDIDATES = (LEGEND_ALIGNMENT, UNDERWORLD_ALIGNMENT)
MANIFEST_FORMAT = 'TRLAU_BIGFILE_LEGEND'
MANIFEST_VERSION = 1
BIGFILE_PRIMARY_SUFFIXES = {'.000', '.dat'}
MAX_BIGFILE_RECORDS = 1_000_000


BIGFILE_SPECIALIZATION_FLAGS: tuple[tuple[int, str, str], ...] = (
    (0x00000001, 'English', 'en'),
    (0x00000002, 'French', 'fr'),
    (0x00000004, 'German', 'de'),
    (0x00000008, 'Italian', 'it'),
    (0x00000010, 'Spanish', 'es'),
    (0x00000020, 'Dutch', 'nl'),
    (0x80000000, 'NextGen', 'nextgen'),
)
_BIGFILE_KNOWN_SPECIALIZATION_MASK = 0
for _trlau_spec_bit, _trlau_spec_name, _trlau_spec_suffix in BIGFILE_SPECIALIZATION_FLAGS:
    _BIGFILE_KNOWN_SPECIALIZATION_MASK |= int(_trlau_spec_bit)

_BIGFILE_LANGUAGE_SPECIALIZATION_FLAGS = tuple(
    (bit, name, suffix) for bit, name, suffix in BIGFILE_SPECIALIZATION_FLAGS if int(bit) != 0x80000000
)
_PSP_SPECIALIZATION_BASE_MASK = 0xFFFFFE00
_PSP_SPECIALIZATION_LOW_MASK = 0x000001FF


def _normalize_bigfile_specialization_mask(spec_mask: int) -> int:
    mask = int(spec_mask) & 0xFFFFFFFF
    if mask in {0, 0xFFFFFFFF}:
        return 0
    if (mask & _PSP_SPECIALIZATION_BASE_MASK) == _PSP_SPECIALIZATION_BASE_MASK:
        low = mask & _PSP_SPECIALIZATION_LOW_MASK
        if low == _PSP_SPECIALIZATION_LOW_MASK:
            return 0
        for bit, _name, _suffix in _BIGFILE_LANGUAGE_SPECIALIZATION_FLAGS:
            if low & int(bit):
                return int(bit)
    return mask


def _is_plausible_bigfile_specialization_mask(spec_mask: int) -> bool:
    mask = int(spec_mask) & 0xFFFFFFFF
    if mask in {0, 0xFFFFFFFF}:
        return True
    normalized = _normalize_bigfile_specialization_mask(mask)
    return bool(normalized) and (int(normalized) & ~_BIGFILE_KNOWN_SPECIALIZATION_MASK) == 0


def bigfile_specialization_names(spec_mask: int) -> list[str]:
    mask = _normalize_bigfile_specialization_mask(spec_mask)
    return [name for bit, name, _suffix in BIGFILE_SPECIALIZATION_FLAGS if mask & int(bit)]


def format_bigfile_specialization(spec_mask: int) -> str:
    mask = _normalize_bigfile_specialization_mask(spec_mask)
    if mask == 0:
        return 'None'
    parts = bigfile_specialization_names(mask)
    return ', '.join(parts) if parts else 'unknown'


def bigfile_first_specialization_suffix(spec_mask: int) -> str:
    mask = _normalize_bigfile_specialization_mask(spec_mask)
    if mask == 0:
        return ''
    for bit, _name, suffix in BIGFILE_SPECIALIZATION_FLAGS:
        if mask & int(bit):
            return suffix
    return ''


def specialize_bigfile_record_path(path_text: str, spec_mask: int) -> str:
    """Add a language/NextGen suffix using the first known specialization flag.

    Bigfiles can contain multiple records with the same filename hash and different
    specialization masks. Using the first known flag keeps multi-specialization
    assets distinct in the UI and extraction output without making long filenames.
    """
    path_text = str(path_text or '')
    suffix = bigfile_first_specialization_suffix(spec_mask)
    if not suffix:
        return path_text
    normalized = path_text.replace('\\', '/')
    folder, sep, filename = normalized.rpartition('/')
    if not filename:
        filename = folder
        folder = ''
        sep = ''
    stem, dot, ext = filename.rpartition('.')
    if not dot:
        stem = filename
        ext = ''
    lower_stem = stem.lower()
    known_suffixes = {f'_{item_suffix.lower()}' for _bit, _name, item_suffix in BIGFILE_SPECIALIZATION_FLAGS}
    if not any(lower_stem.endswith(item_suffix) for item_suffix in known_suffixes):
        stem = f'{stem}_{suffix}'
    renamed = f'{stem}.{ext}' if dot else stem
    return f'{folder}{sep}{renamed}' if folder or sep else renamed


FILELISTS_DIRNAME = 'filelists'


def bundled_filelists_dir() -> Path:
    return Path(__file__).resolve().parents[1] / FILELISTS_DIRNAME


def available_bigfile_filelists() -> list[str]:
    directory = bundled_filelists_dir()
    if not directory.exists():
        return []
    return sorted(child.name for child in directory.iterdir() if child.is_file() and child.suffix.lower() == '.txt')


def bundled_filelist_path(name: str) -> Path | None:
    name = str(name or '').strip()
    if not name:
        return None
    directory = bundled_filelists_dir()
    path = (directory / name).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError:
        return None
    if not path.is_file() or path.suffix.lower() != '.txt':
        return None
    return path


def _crc32_standard(data: bytes, init: int = 0) -> int:
    return zlib.crc32(data, init) & 0xFFFFFFFF


@lru_cache(maxsize=1)
def _crc32_direct_table() -> tuple[int, ...]:
    table = []
    for index in range(256):
        crc = int(index) << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return tuple(table)


def _crc32_direct(data: bytes, init: int = 0xFFFFFFFF, xor_out: int = 0xFFFFFFFF) -> int:
    crc = int(init) & 0xFFFFFFFF
    table = _crc32_direct_table()
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ (int(byte) & 0xFF)) & 0xFF]
    return (crc ^ int(xor_out)) & 0xFFFFFFFF


def _byteswap32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    return int.from_bytes(value.to_bytes(4, 'little'), 'big')


def _crc32_name_values(text: str) -> set[int]:
    values: set[int] = set()
    encodings = []
    for encoding in ('utf-8', 'cp1252'):
        try:
            data = text.encode(encoding)
        except UnicodeEncodeError:
            continue
        if data not in encodings:
            encodings.append(data)

    for data in encodings:
        variants = {
            _crc32_standard(data),
            _crc32_standard(data, 0xFFFFFFFF) ^ 0xFFFFFFFF,
            _crc32_standard(data, 0xFFFFFFFF),
            _crc32_standard(data) ^ 0xFFFFFFFF,
            _crc32_direct(data, 0xFFFFFFFF, 0xFFFFFFFF),
            _crc32_direct(data, 0xFFFFFFFF, 0),
            _crc32_direct(data, 0, 0xFFFFFFFF),
            _crc32_direct(data, 0, 0),
        }
        for value in list(variants):
            variants.add(_byteswap32(value))
            variants.add((~value) & 0xFFFFFFFF)
        values.update(value & 0xFFFFFFFF for value in variants)
    return values


def _filelist_hash_candidates(path_text: str) -> set[str]:
    text = str(path_text or '').strip().lstrip('\ufeff')
    if not text:
        return set()
    slash = text.replace('\\', '/')
    backslash = text.replace('/', '\\')
    bases = {text, slash, backslash}
    stripped = {item.strip('\\/') for item in bases if item}
    bases |= stripped
    bases |= {f'\\{item}' for item in stripped if item}
    bases |= {f'/{item}' for item in stripped if item}
    expanded: set[str] = set()
    for item in bases:
        if not item:
            continue
        expanded.add(item)
        expanded.add(item.lower())
        expanded.add(item.upper())
        expanded.add(item.replace('\\', '/'))
        expanded.add(item.replace('/', '\\'))
        expanded.add(item.replace('\\', '/').lower())
        expanded.add(item.replace('/', '\\').lower())
        expanded.add(item.replace('\\', '/').upper())
        expanded.add(item.replace('/', '\\').upper())
    return {item for item in expanded if item}


def _display_filelist_path(path_text: str) -> str:
    text = str(path_text or '').strip().lstrip('\ufeff').replace('\\', '/')
    text = text.strip('/')
    return text or path_text


@lru_cache(maxsize=32)
def _load_bigfile_filename_map_cached(filelist_path: str) -> dict[int, str]:
    path = Path(filelist_path)
    if not path.is_file():
        raise BigfileError(f'File list not found: {path}')
    mapping: dict[int, str] = {}
    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        text = path.read_text(encoding='cp1252')
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        display = _display_filelist_path(line)
        for candidate in _filelist_hash_candidates(line):
            for hash_value in _crc32_name_values(candidate):
                mapping.setdefault(hash_value, display)
    return mapping


def load_bigfile_filename_map(filelist_path: str | Path | None) -> dict[int, str]:
    if filelist_path is None or not str(filelist_path):
        return {}
    return dict(_load_bigfile_filename_map_cached(str(Path(filelist_path).resolve())))


def load_bigfile_filename_map_for_hashes(filelist_path: str | Path | None, hash_values: Iterable[int]) -> dict[int, str]:
    """Resolve only the hashes present in an opened Bigfile.

    Building the full filename map is intentionally broad: it indexes many path
    spellings, CRC variants, byte-swapped variants, and inverted variants for
    every line in the file list.  That is useful for tools, but the browser only
    needs names for hashes that are actually in the archive header.  This targeted
    resolver avoids allocating a huge all-variants dictionary during Bigfile open
    while keeping the same matching behavior.
    """
    if filelist_path is None or not str(filelist_path):
        return {}
    wanted = {int(value) & 0xFFFFFFFF for value in hash_values}
    if not wanted:
        return {}
    path = Path(filelist_path)
    if not path.is_file():
        raise BigfileError(f'File list not found: {path}')

    mapping: dict[int, str] = {}
    try:
        text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        text = path.read_text(encoding='cp1252')

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        display = _display_filelist_path(line)
        for candidate in _filelist_hash_candidates(line):
            for hash_value in _crc32_name_values(candidate):
                if hash_value in wanted and hash_value not in mapping:
                    mapping[hash_value] = display
        if len(mapping) >= len(wanted):
            break
    return mapping


def _safe_relative_record_path(name: str) -> Path:
    text = str(name or '').replace('\\', '/').strip('/')
    parts = []
    for part in text.split('/'):
        part = part.strip()
        if not part or part in {'.', '..'}:
            continue
        part = part.replace(':', '_')
        parts.append(part)
    if not parts:
        return Path('unknown.bin')
    return Path(*parts)


def is_legend_primary_part(path: str | Path) -> bool:
    source_path = Path(path)
    if source_path.suffix.lower() == '.dat':
        return True
    name = source_path.name
    match = re.match(r'^(.*)\.(\d{3})$', name)
    return bool(match and int(match.group(2)) == 0)


def _require_legend_primary_part(path: str | Path) -> Path:
    source_path = Path(path)
    if not is_legend_primary_part(source_path):
        raise BigfileError('Select the first Bigfile segment (.000) or a .dat Bigfile archive')
    return source_path


class BigfileError(Exception):
    pass


@dataclass
class BigfileRecord:
    index: int
    hash_value: int
    size: int
    offset: int
    spec_mask: int
    compressed_length: int
    endian: str = '<'
    alignment: int = LEGEND_ALIGNMENT

    @property
    def is_compressed(self) -> bool:
        return int(self.compressed_length) > 0

    @property
    def stored_size(self) -> int:
        return int(self.compressed_length) if self.is_compressed else int(self.size)


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return (int(value) + alignment - 1) // alignment * alignment


def _is_single_file_bigfile(base_path: Path) -> bool:
    # BIGFILE.DAT
    return Path(base_path).suffix.lower() == '.dat'


def _part_path(base_path: Path, part_index: int) -> Path:
    base_path = Path(base_path)
    if _is_single_file_bigfile(base_path):
        return base_path
    match = re.match(r'^(.*)\.(\d{3})$', base_path.name)
    if match:
        return base_path.with_name(f'{match.group(1)}.{part_index:03d}')
    if part_index == 0:
        return base_path
    return base_path.with_name(f'{base_path.name}.{part_index:03d}')


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise BigfileError(f'Unexpected end of file while reading {size} bytes')
    return data


def _normalize_bigfile_endian(value: str | None, default: str = '<') -> str:
    text = str(value or '').strip().lower()
    if text in {'<', 'le', 'little', 'little-endian', 'little_endian'}:
        return '<'
    if text in {'>', 'be', 'big', 'big-endian', 'big_endian'}:
        return '>'
    return default if default in {'<', '>'} else '<'


def _bigfile_endian_label(endian: str) -> str:
    return 'big' if endian == '>' else 'little'


def _archive_virtual_size(base_path: Path, alignment: int = LEGEND_ALIGNMENT) -> int:
    base_path = Path(base_path)
    if _is_single_file_bigfile(base_path):
        return base_path.stat().st_size if base_path.exists() else 0
    parts = _matching_part_paths(base_path)
    if not parts:
        return base_path.stat().st_size if base_path.exists() else 0
    virtual_size = 0
    for part in parts:
        match = re.match(r'^(.*)\.(\d{3})$', part.name)
        part_index = int(match.group(2)) if match else 0
        virtual_size = max(virtual_size, part_index * int(alignment) + part.stat().st_size)
    return int(virtual_size)


def _split_alignment_errors(base_path: Path, alignment: int) -> list[str]:
    if _is_single_file_bigfile(base_path):
        return []
    errors: list[str] = []
    for part in _matching_part_paths(base_path):
        size = part.stat().st_size
        if int(alignment) > 0 and size > int(alignment):
            errors.append(f'{part.name} is larger than split alignment 0x{int(alignment):X}')
    return errors


def _score_legend_header_records(
    records: list[BigfileRecord],
    header_size: int,
    archive_size: int,
    *,
    allow_incomplete: bool = False,
) -> tuple[int, list[str]]:
    data_floor = _align(header_size, SECTOR_SIZE)
    score = 0
    invalid: list[str] = []
    previous_absolute = -1
    complete_records = 0
    incomplete_records = 0
    for record in records:
        size = int(record.size)
        compressed_length = int(record.compressed_length)
        stored_size = int(record.stored_size)
        absolute_offset = int(record.offset) << 11
        if size < 0 or stored_size < 0:
            invalid.append(f'record {record.index} has negative size')
            continue
        if stored_size and absolute_offset < data_floor:
            invalid.append(f'record {record.index} points into header')
            continue
        if archive_size and stored_size and absolute_offset + stored_size > archive_size:
            if not allow_incomplete:
                invalid.append(f'record {record.index} exceeds archive size')
                continue
            incomplete_records += 1
        else:
            complete_records += 1
        score += 10
        if absolute_offset >= previous_absolute:
            score += 1
        previous_absolute = absolute_offset
        if int(record.hash_value) not in {0, 0xFFFFFFFF}:
            score += 1
        if _is_plausible_bigfile_specialization_mask(int(record.spec_mask)):
            score += 1
    if allow_incomplete and incomplete_records and not complete_records and records:
        invalid.append('archive fragment does not contain any complete records')
    if allow_incomplete and incomplete_records:
        score -= 1
    if not records:
        score += 1
    return score, invalid


def _read_legend_header_candidate(path: Path, endian: str, *, allow_incomplete: bool = False) -> tuple[list[BigfileRecord], int]:
    endian = _normalize_bigfile_endian(endian)
    first_part_size = path.stat().st_size if path.exists() else 0
    with path.open('rb') as handle:
        header = _read_exact(handle, 4)
        (num_records,) = struct.unpack(f'{endian}I', header)
        if num_records > MAX_BIGFILE_RECORDS:
            raise BigfileError(f'Unreasonable record count for {_bigfile_endian_label(endian)} endian: {num_records}')
        header_size = 4 + int(num_records) * 4 + int(num_records) * 16
        if header_size > first_part_size:
            raise BigfileError(
                f'Header for {_bigfile_endian_label(endian)} endian would exceed first archive part: '
                f'{header_size:,} > {first_part_size:,}'
            )
        hashes = list(struct.unpack(f'{endian}{num_records}I', _read_exact(handle, num_records * 4))) if num_records else []
        records: list[BigfileRecord] = []
        for index in range(num_records):
            size, offset, spec_mask, compressed_length = struct.unpack(f'{endian}IIIi', _read_exact(handle, 16))
            records.append(BigfileRecord(index, hashes[index], size, offset, spec_mask, compressed_length, endian=endian))

    alignments = (LEGEND_ALIGNMENT,) if _is_single_file_bigfile(path) else BIGFILE_ALIGNMENT_CANDIDATES
    candidates: list[tuple[int, int]] = []
    errors: list[str] = []
    for alignment in alignments:
        invalid = _split_alignment_errors(path, alignment)
        archive_size = _archive_virtual_size(path, alignment)
        score, record_errors = _score_legend_header_records(records, header_size, archive_size, allow_incomplete=allow_incomplete)
        invalid.extend(record_errors)
        if invalid:
            preview = '; '.join(invalid[:3])
            if len(invalid) > 3:
                preview += f'; {len(invalid) - 3} more'
            errors.append(f'0x{alignment:X}: {preview}')
            continue
        parts = _matching_part_paths(path) if not _is_single_file_bigfile(path) else []
        if parts and any(part.stat().st_size > LEGEND_ALIGNMENT for part in parts):
            if alignment == UNDERWORLD_ALIGNMENT:
                score += 1000
        candidates.append((score, alignment))

    if not candidates:
        detail = ' | '.join(errors) if errors else 'no valid split alignment'
        raise BigfileError(f'Invalid {_bigfile_endian_label(endian)} endian Bigfile header: {detail}')

    candidates.sort(key=lambda item: (item[0], 1 if item[1] == LEGEND_ALIGNMENT else 0), reverse=True)
    score, alignment = candidates[0]
    for record in records:
        record.alignment = int(alignment)
    return records, score


def read_legend_header(path: str | Path, endian: str | None = None, *, allow_incomplete: bool = False) -> list[BigfileRecord]:
    path = _require_legend_primary_part(path)
    requested = _normalize_bigfile_endian(endian, default='') if endian else None
    if requested in {'<', '>'}:
        records, _score = _read_legend_header_candidate(path, requested, allow_incomplete=allow_incomplete)
        return records

    candidates: list[tuple[int, str, list[BigfileRecord]]] = []
    errors: list[str] = []
    for candidate_endian in ('<', '>'):
        try:
            records, score = _read_legend_header_candidate(path, candidate_endian, allow_incomplete=allow_incomplete)
        except BigfileError as exc:
            errors.append(str(exc))
            continue
        candidates.append((int(score), candidate_endian, records))

    if not candidates:
        detail = ' | '.join(errors) if errors else 'no valid little-endian or big-endian header candidate'
        raise BigfileError(f'Unable to read Bigfile header: {detail}')

    candidates.sort(key=lambda item: (item[0], 1 if item[1] == '<' else 0), reverse=True)
    return candidates[0][2]


def _records_endian(records: list[BigfileRecord], default: str = '<') -> str:
    if records:
        return _normalize_bigfile_endian(getattr(records[0], 'endian', default), default=default)
    return _normalize_bigfile_endian(default)


def _records_alignment(records: list[BigfileRecord], default: int = LEGEND_ALIGNMENT) -> int:
    if records:
        try:
            alignment = int(getattr(records[0], 'alignment', default))
        except Exception:
            alignment = int(default)
        if alignment > 0:
            return alignment
    return int(default)


def _read_from_parts(base_path: Path, absolute_offset: int, size: int, alignment: int = LEGEND_ALIGNMENT) -> bytes:
    base_path = Path(base_path)
    if _is_single_file_bigfile(base_path):
        if not base_path.exists():
            raise BigfileError(f'Missing archive file: {base_path}')
        with base_path.open('rb') as handle:
            handle.seek(int(absolute_offset))
            return _read_exact(handle, int(size))

    remaining = int(size)
    cursor = int(absolute_offset)
    chunks: list[bytes] = []
    while remaining > 0:
        part_index = cursor // alignment
        part_offset = cursor % alignment
        part_path = _part_path(base_path, part_index)
        if not part_path.exists():
            raise BigfileError(f'Missing archive part: {part_path}')
        amount = min(remaining, alignment - part_offset)
        with part_path.open('rb') as handle:
            handle.seek(part_offset)
            chunks.append(_read_exact(handle, amount))
        cursor += amount
        remaining -= amount
    return b''.join(chunks)




def _record_data_available(base_path: Path, absolute_offset: int, size: int, alignment: int = LEGEND_ALIGNMENT) -> bool:
    """Return True when the record's stored bytes can be read from available archive parts."""
    base_path = Path(base_path)
    size = int(size)
    absolute_offset = int(absolute_offset)
    if size <= 0:
        return True
    if absolute_offset < 0:
        return False
    if _is_single_file_bigfile(base_path):
        return base_path.exists() and absolute_offset + size <= base_path.stat().st_size

    remaining = size
    cursor = absolute_offset
    while remaining > 0:
        part_index = cursor // alignment
        part_offset = cursor % alignment
        part_path = _part_path(base_path, part_index)
        if not part_path.exists():
            return False
        amount = min(remaining, alignment - part_offset)
        if part_offset + amount > part_path.stat().st_size:
            return False
        cursor += amount
        remaining -= amount
    return True

def _open_part_for_write(base_path: Path, part_index: int, handles: dict[int, BinaryIO]) -> BinaryIO:
    base_path = Path(base_path)
    handle_key = 0 if _is_single_file_bigfile(base_path) else int(part_index)
    handle = handles.get(handle_key)
    if handle is None:
        part_path = _part_path(base_path, handle_key)
        part_path.parent.mkdir(parents=True, exist_ok=True)
        handle = part_path.open('w+b')
        handles[handle_key] = handle
    return handle


def _write_to_parts(base_path: Path, absolute_offset: int, data: bytes, handles: dict[int, BinaryIO], alignment: int = LEGEND_ALIGNMENT) -> None:
    base_path = Path(base_path)
    view = memoryview(data)
    if _is_single_file_bigfile(base_path):
        handle = _open_part_for_write(base_path, 0, handles)
        handle.seek(int(absolute_offset))
        handle.write(view)
        return

    cursor = int(absolute_offset)
    written = 0
    remaining = len(data)
    while remaining > 0:
        part_index = cursor // alignment
        part_offset = cursor % alignment
        amount = min(remaining, alignment - part_offset)
        handle = _open_part_for_write(base_path, part_index, handles)
        handle.seek(part_offset)
        handle.write(view[written:written + amount])
        cursor += amount
        written += amount
        remaining -= amount


def _clear_existing_parts(base_path: Path) -> None:
    base_path = Path(base_path)
    if _is_single_file_bigfile(base_path):
        if base_path.exists():
            base_path.unlink()
        return

    match = re.match(r'^(.*)\.(\d{3})$', base_path.name)
    if match:
        prefix = match.group(1)
        pattern = re.compile(rf'^{re.escape(prefix)}\.\d{{3}}$')
        for child in base_path.parent.iterdir():
            if child.is_file() and pattern.match(child.name):
                child.unlink()
        return

    if base_path.exists():
        base_path.unlink()


def _matching_part_paths(base_path: Path) -> list[Path]:
    base_path = Path(base_path)
    if _is_single_file_bigfile(base_path):
        return [base_path] if base_path.exists() else []

    match = re.match(r'^(.*)\.(\d{3})$', base_path.name)
    if match:
        prefix = match.group(1)
        pattern = re.compile(rf'^{re.escape(prefix)}\.\d{{3}}$')
        return sorted(child for child in base_path.parent.iterdir() if child.is_file() and pattern.match(child.name))

    return [base_path] if base_path.exists() else []


def legend_bigfile_backup_dir(path: str | Path) -> Path:
    source_path = _require_legend_primary_part(path)
    return source_path.parent / 'bigfile_backup'


def legend_bigfile_backup_exists(path: str | Path) -> bool:
    source_path = _require_legend_primary_part(path)
    backup_dir = legend_bigfile_backup_dir(source_path)
    if not backup_dir.is_dir():
        return False
    source_parts = _matching_part_paths(source_path)
    if not source_parts:
        return False
    for part in source_parts:
        backup_part = backup_dir / part.name
        if not backup_part.is_file():
            return False
    return True


def create_legend_bigfile_backup(path: str | Path, *, overwrite: bool = False) -> Path:
    source_path = _require_legend_primary_part(path)
    source_parts = _matching_part_paths(source_path)
    if not source_parts:
        raise BigfileError(f'No Bigfile parts found for: {source_path}')
    backup_dir = legend_bigfile_backup_dir(source_path)
    existing = [backup_dir / part.name for part in source_parts if (backup_dir / part.name).exists()]
    if existing and not overwrite:
        raise BigfileError(f'Backup already exists: {backup_dir}')
    backup_dir.mkdir(parents=True, exist_ok=True)
    for part in source_parts:
        shutil.copy2(part, backup_dir / part.name)
    return backup_dir


def _replace_parts(temp_base: Path, output_base: Path) -> None:
    temp_parts = _matching_part_paths(temp_base)
    if not temp_parts:
        raise BigfileError(f'Temporary Bigfile was not written: {temp_base}')
    _clear_existing_parts(output_base)
    if _is_single_file_bigfile(output_base):
        temp_part = temp_parts[0]
        output_base.parent.mkdir(parents=True, exist_ok=True)
        temp_part.replace(output_base)
        for extra_part in temp_parts[1:]:
            try:
                extra_part.unlink()
            except FileNotFoundError:
                pass
        return

    for temp_part in temp_parts:
        match = re.match(r'^(.*)\.(\d{3})$', temp_part.name)
        part_index = int(match.group(2)) if match else 0
        dest_part = _part_path(output_base, part_index)
        dest_part.parent.mkdir(parents=True, exist_ok=True)
        temp_part.replace(dest_part)


def _temporary_primary_part_near(path: Path) -> Path:
    path = Path(path)
    stem = re.sub(r'\.000$', '', path.name, flags=re.IGNORECASE)
    if _is_single_file_bigfile(path):
        dat_stem = path.name[:-len(path.suffix)] if path.suffix else path.name
        return path.with_name(f'{dat_stem}.trlau_tmp_{uuid.uuid4().hex[:8]}{path.suffix}')
    return path.with_name(f'{stem}.trlau_tmp_{uuid.uuid4().hex[:8]}.000')


def _record_filename(index: int, hash_value: int) -> str:
    return f'{index:05d}_{hash_value:08X}.bin'


def _unpack_legend_records(
    path: str | Path,
    output_dir: str | Path | None = None,
    *,
    record_indices: set[int] | None = None,
    filename_map: dict[int, str] | None = None,
    allow_incomplete: bool = False,
) -> Path:
    source_path = _require_legend_primary_part(path)
    if output_dir is None or not str(output_dir):
        output_dir = source_path.with_name(f'{source_path.stem}_unpacked')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    files_dir = output_path / 'files'
    files_dir.mkdir(parents=True, exist_ok=True)

    records = read_legend_header(source_path, allow_incomplete=allow_incomplete)
    archive_alignment = _records_alignment(records)
    selected = set(record_indices) if record_indices is not None else None
    filename_map = filename_map or {}
    manifest_records = []
    skipped_unavailable: list[int] = []
    for record in records:
        if selected is not None and int(record.index) not in selected:
            continue
        absolute_offset = int(record.offset) << 11
        if not _record_data_available(source_path, absolute_offset, record.stored_size, archive_alignment):
            if allow_incomplete:
                skipped_unavailable.append(int(record.index))
                continue
            raise BigfileError(
                f'Record {record.index} is outside the available Bigfile data. '
                'Open the complete archive or use the Bigfile browser to extract records that are present.'
            )
        stored = _read_from_parts(source_path, absolute_offset, record.stored_size, archive_alignment)
        raw = stored
        if record.is_compressed:
            try:
                raw = zlib.decompress(stored)
            except Exception as exc:
                raise BigfileError(f'Failed to decompress record {record.index} hash {record.hash_value:08X}: {exc}') from exc
            if len(raw) != record.size:
                raise BigfileError(f'Decompressed record {record.index} size mismatch: got {len(raw)}, expected {record.size}')

        matched_name = filename_map.get(int(record.hash_value))
        filename = matched_name or _record_filename(record.index, record.hash_value)
        filename = specialize_bigfile_record_path(filename, int(record.spec_mask))
        relative_path = _safe_relative_record_path(filename)
        output_file = files_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(raw)
        manifest_records.append({
            'index': int(record.index),
            'hash': f'{record.hash_value:08X}',
            'size': int(record.size),
            'offset': int(record.offset),
            'specMask': f'{record.spec_mask:08X}',
            'compressedLength': int(record.compressed_length),
            'compressed': bool(record.is_compressed),
            'storedSize': int(record.stored_size),
            'matchedName': matched_name or '',
            'file': 'files/' + str(relative_path).replace('\\', '/'),
        })

    manifest = {
        'format': MANIFEST_FORMAT,
        'version': MANIFEST_VERSION,
        'game': 'Tomb Raider Legend',
        'sectorSize': SECTOR_SIZE,
        'alignment': archive_alignment,
        'endianness': _bigfile_endian_label(_records_endian(records)),
        'source': str(source_path),
        'recordCount': len(manifest_records),
        'archiveRecordCount': len(records),
        'partial': selected is not None or bool(skipped_unavailable),
        'skippedUnavailableCount': len(skipped_unavailable),
        'skippedUnavailableRecords': skipped_unavailable[:1000],
        'records': manifest_records,
    }
    if skipped_unavailable and len(skipped_unavailable) > 1000:
        manifest['skippedUnavailableRecordsTruncated'] = True
    manifest_path = output_path / (
        'manifest_selected.json' if selected is not None else ('manifest_available.json' if skipped_unavailable else 'manifest.json')
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest_path


def unpack_legend_bigfile(
    path: str | Path,
    output_dir: str | Path | None = None,
    *,
    filename_map: dict[int, str] | None = None,
    allow_incomplete: bool = False,
) -> Path:
    return _unpack_legend_records(
        path,
        output_dir,
        record_indices=None,
        filename_map=filename_map,
        allow_incomplete=allow_incomplete,
    )


def unpack_legend_bigfile_selection(
    path: str | Path,
    output_dir: str | Path | None,
    record_indices: Iterable[int],
    *,
    filename_map: dict[int, str] | None = None,
    allow_incomplete: bool = False,
) -> Path:
    indices = {int(index) for index in record_indices}
    if not indices:
        raise BigfileError('No Bigfile records selected')
    return _unpack_legend_records(path, output_dir, record_indices=indices, filename_map=filename_map, allow_incomplete=allow_incomplete)

def _parse_int(value, field_name: str) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith('0x'):
            return int(text, 16)
        if field_name in {'hash', 'specMask'}:
            return int(text, 16)
        if re.fullmatch(r'[0-9a-fA-F]+', text) and any(c.isalpha() for c in text):
            return int(text, 16)
        return int(text, 10)
    raise BigfileError(f'Invalid integer value for {field_name}: {value!r}')


def repack_legend_bigfile(manifest_path: str | Path, output_path: str | Path | None = None, *, compress_marked_records: bool = True) -> Path:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != MANIFEST_FORMAT:
        raise BigfileError('Manifest is not a TRLAU Legend Bigfile manifest')
    if int(manifest.get('version', 0)) != MANIFEST_VERSION:
        raise BigfileError(f'Unsupported manifest version: {manifest.get("version")}')

    records_data = list(manifest.get('records') or [])
    records_data.sort(key=lambda item: int(item.get('index', 0)))
    num_records = len(records_data)
    alignment = int(manifest.get('alignment') or LEGEND_ALIGNMENT)
    sector_size = int(manifest.get('sectorSize') or SECTOR_SIZE)
    if sector_size != SECTOR_SIZE:
        raise BigfileError(f'Unsupported sector size: {sector_size}')
    endian = _normalize_bigfile_endian(manifest.get('endianness') or manifest.get('endian'), default='<')

    if output_path is None or not str(output_path):
        output_path = manifest_path.parent / 'bigfile_repacked.000'
    output_base = _require_legend_primary_part(output_path)
    _clear_existing_parts(output_base)

    prepared = []
    data_cursor = _align(4 + num_records * 4 + num_records * 16, SECTOR_SIZE)
    for item in records_data:
        rel_file = item.get('file')
        if not rel_file:
            raise BigfileError(f'Manifest record {item.get("index")} has no file path')
        file_path = (manifest_path.parent / rel_file).resolve()
        if not file_path.exists():
            raise BigfileError(f'Missing unpacked file for record {item.get("index")}: {file_path}')
        raw = file_path.read_bytes()
        hash_value = _parse_int(item.get('hash'), 'hash')
        spec_mask = _parse_int(item.get('specMask', 0), 'specMask')
        should_compress = bool(item.get('compressed', False)) and compress_marked_records
        if should_compress:
            stored = zlib.compress(raw, level=9)
            compressed_length = len(stored)
        else:
            stored = raw
            compressed_length = 0
        data_cursor = _align(data_cursor, SECTOR_SIZE)
        offset_sector = data_cursor >> 11
        prepared.append({
            'hash': hash_value & 0xFFFFFFFF,
            'size': len(raw),
            'offset': offset_sector,
            'specMask': spec_mask & 0xFFFFFFFF,
            'compressedLength': int(compressed_length),
            'stored': stored,
            'absoluteOffset': data_cursor,
        })
        data_cursor += _align(len(stored), SECTOR_SIZE)

    header = bytearray()
    header += struct.pack(f'{endian}I', num_records)
    for item in prepared:
        header += struct.pack(f'{endian}I', item['hash'])
    for item in prepared:
        header += struct.pack(f'{endian}IIIi', item['size'], item['offset'], item['specMask'], item['compressedLength'])

    handles: dict[int, BinaryIO] = {}
    try:
        _write_to_parts(output_base, 0, bytes(header), handles, alignment)
        for item in prepared:
            _write_to_parts(output_base, item['absoluteOffset'], item['stored'], handles, alignment)
    finally:
        for handle in handles.values():
            handle.close()

    return _part_path(output_base, 0)


def repack_legend_bigfile_with_replacements(
    source_path: str | Path,
    replacements: dict[int, bytes | bytearray | memoryview],
    output_path: str | Path | None = None,
    *,
    compress_replaced_records: bool = True,
) -> Path:
    source_path = _require_legend_primary_part(source_path)
    if not replacements:
        raise BigfileError('No replacement records were provided')
    replacement_bytes = {int(index): bytes(data) for index, data in replacements.items()}
    records = read_legend_header(source_path)
    archive_alignment = _records_alignment(records)
    valid_indices = {int(record.index) for record in records}
    invalid = sorted(index for index in replacement_bytes if index not in valid_indices)
    if invalid:
        raise BigfileError(f'Replacement record index not found in Bigfile: {invalid[0]}')

    if output_path is None or not str(output_path):
        output_path = source_path
    output_base = _require_legend_primary_part(output_path)
    inplace = output_base.resolve() == source_path.resolve()
    write_base = _temporary_primary_part_near(output_base) if inplace else output_base
    _clear_existing_parts(write_base)

    num_records = len(records)
    endian = _records_endian(records, default='<')
    data_cursor = _align(4 + num_records * 4 + num_records * 16, SECTOR_SIZE)
    prepared = []
    for record in records:
        index = int(record.index)
        if index in replacement_bytes:
            raw = replacement_bytes[index]
            if record.is_compressed and compress_replaced_records:
                stored = zlib.compress(raw, level=9)
                compressed_length = len(stored)
            else:
                stored = raw
                compressed_length = 0
            size = len(raw)
            stored_size = len(stored)
            replacement_stored = stored
        else:
            size = int(record.size)
            compressed_length = int(record.compressed_length)
            stored_size = int(record.stored_size)
            replacement_stored = None

        data_cursor = _align(data_cursor, SECTOR_SIZE)
        offset_sector = data_cursor >> 11
        prepared.append({
            'record': record,
            'hash': int(record.hash_value) & 0xFFFFFFFF,
            'size': int(size),
            'offset': int(offset_sector),
            'specMask': int(record.spec_mask) & 0xFFFFFFFF,
            'compressedLength': int(compressed_length),
            'storedSize': int(stored_size),
            'stored': replacement_stored,
            'absoluteOffset': int(data_cursor),
        })
        data_cursor += _align(stored_size, SECTOR_SIZE)

    header = bytearray()
    header += struct.pack(f'{endian}I', num_records)
    for item in prepared:
        header += struct.pack(f'{endian}I', item['hash'])
    for item in prepared:
        header += struct.pack(f'{endian}IIIi', item['size'], item['offset'], item['specMask'], item['compressedLength'])

    handles: dict[int, BinaryIO] = {}
    try:
        _write_to_parts(write_base, 0, bytes(header), handles, archive_alignment)
        for item in prepared:
            stored = item['stored']
            if stored is None:
                record = item['record']
                stored = _read_from_parts(source_path, int(record.offset) << 11, int(record.stored_size), archive_alignment)
            _write_to_parts(write_base, item['absoluteOffset'], stored, handles, archive_alignment)
    finally:
        for handle in handles.values():
            handle.close()

    if inplace:
        _replace_parts(write_base, output_base)

    return _part_path(output_base, 0)


def _records_expected_virtual_size(records: list[BigfileRecord]) -> int:
    expected_size = 0
    for record in records:
        if int(record.stored_size) <= 0:
            continue
        expected_size = max(expected_size, (int(record.offset) << 11) + int(record.stored_size))
    return int(expected_size)


def inspect_legend_bigfile(path: str | Path) -> dict:
    source_path = _require_legend_primary_part(path)
    records = read_legend_header(source_path, allow_incomplete=True)
    archive_alignment = _records_alignment(records)
    compressed = sum(1 for record in records if record.is_compressed)
    total_size = sum(record.size for record in records)
    total_stored = sum(record.stored_size for record in records)
    available_size = _archive_virtual_size(source_path, archive_alignment)
    expected_size = _records_expected_virtual_size(records)
    return {
        'path': str(source_path),
        'recordCount': len(records),
        'endianness': _bigfile_endian_label(_records_endian(records)),
        'alignment': f'0x{archive_alignment:X}',
        'compressedCount': compressed,
        'totalSize': total_size,
        'totalStoredSize': total_stored,
        'availableVirtualSize': available_size,
        'expectedVirtualSize': expected_size,
        'archiveComplete': available_size >= expected_size,
        'firstHash': f'{records[0].hash_value:08X}' if records else None,
        'lastHash': f'{records[-1].hash_value:08X}' if records else None,
    }
