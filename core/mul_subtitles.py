from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Mapping


MUL_SUBTITLE_TEXT_PROP = 'trlau_mul_subtitle_text'
MUL_SUBTITLE_TEXT_MAGIC = 'cdcEditor MUL subtitles v2'
MUL_SUBTITLE_TEXT_MAGIC_V1 = '# cdcEditor MUL subtitles v1'


MUL_SUBTITLE_LANGUAGE_ID_TO_NAME = {
    '0': 'English',
    '1': 'French',
    '2': 'German',
    '3': 'Italian',
    '4': 'Spanish',
    '7': 'Polish',
    '9': 'Russian',
    '10': 'Czech',
}
MUL_SUBTITLE_LANGUAGE_NAME_TO_ID = {
    name.casefold(): language_id
    for language_id, name in MUL_SUBTITLE_LANGUAGE_ID_TO_NAME.items()
}


def subtitle_language_to_id(value: object) -> str:
    token = str(value if value is not None else '').strip()
    if not token:
        return ''
    mapped = MUL_SUBTITLE_LANGUAGE_NAME_TO_ID.get(token.casefold())
    return mapped if mapped is not None else token


def subtitle_language_display_name(value: object) -> str:
    token = str(value if value is not None else '').strip()
    return MUL_SUBTITLE_LANGUAGE_ID_TO_NAME.get(token, token)


# Backward-compatible parser for Text datablocks made by the first subtitle
# implementation. New imports use the JSON dialogue representation below.
_FRAME_RE = re.compile(r'^frame\s+(-?\d+)\s*=\s*(.+?)\s*$', re.IGNORECASE)
_FRAMES_RE = re.compile(r'^frames\s+(-?\d+)\s*-\s*(-?\d+)\s*=\s*(.+?)\s*$', re.IGNORECASE)
_COUNT_RE = re.compile(r'^num_subtitles\s*=\s*(-?\d+)\s*$', re.IGNORECASE)


def _coerce_blob(value) -> bytes:
    if value is None:
        return b''
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        text = ''.join(value.strip().split())
        if text.lower().startswith('hex:'):
            text = text[4:]
        if text.lower().startswith('0x'):
            number = int(text, 16)
            return int(number & 0xFFFFFFFF).to_bytes(4, 'little', signed=False)
        return bytes.fromhex(text) if text else b''
    return bytes(value)


def normalize_subtitle_frames(frames) -> dict[int, bytes]:
    result: dict[int, bytes] = {}
    if isinstance(frames, Mapping):
        iterable: Iterable = frames.items()
    else:
        iterable = frames or []

    for item in iterable:
        frame = None
        blob = None
        if isinstance(frames, Mapping):
            try:
                frame, blob = item
            except Exception:
                continue
        elif isinstance(item, Mapping):
            frame = item.get('frame')
            blob = item.get('data', item.get('blob', item.get('data_hex')))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            frame, blob = item[0], item[1]
        else:
            continue
        try:
            frame_index = int(frame)
            result[frame_index] = _coerce_blob(blob)
        except Exception:
            continue
    return result


def subtitle_blob_has_data(blob: bytes | bytearray | memoryview | None) -> bool:
    raw = _coerce_blob(blob)
    return any(byte != 0 for byte in raw)


def _subtitle_endian(endian: str) -> str:
    return '>' if str(endian or '<').startswith('>') else '<'


def decode_subtitle_set(blob: bytes | bytearray | memoryview, endian: str = '<') -> tuple[list[dict[str, str]], bytes] | None:
    raw = _coerce_blob(blob)
    if len(raw) < 4:
        return None

    endian = _subtitle_endian(endian)
    try:
        size = int(struct.unpack_from(endian + 'i', raw, 0)[0])
    except Exception:
        return None
    if size < 0 or 4 + size > len(raw):
        return None

    content = raw[4:4 + size]
    suffix = raw[4 + size:]
    entries: list[dict[str, str]] = []
    offset = 0
    while offset < len(content):
        language_end = content.find(b'\r', offset)
        if language_end < 0:
            return None
        text_start = language_end + 1
        text_end = content.find(b'\r', text_start)
        if text_end < 0:
            return None
        try:
            language = content[offset:language_end].decode('utf-8', errors='strict')
            text = content[text_start:text_end].decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            return None
        entries.append({'language': language, 'text': text})
        offset = text_end + 1

    return entries, suffix


def encode_subtitle_set(entries, endian: str = '<', suffix: bytes | bytearray | memoryview | None = None) -> bytes:
    endian = _subtitle_endian(endian)
    content = bytearray()
    for entry_index, entry in enumerate(entries or []):
        if isinstance(entry, Mapping):
            language = subtitle_language_to_id(entry.get('language', ''))
            dialogue = str(entry.get('text', entry.get('dialogue', '')))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            language = subtitle_language_to_id(entry[0])
            dialogue = str(entry[1])
        else:
            raise ValueError(f'subtitle entry {entry_index} must contain language and text')

        # 0x0D is the actual on-disk string terminator, so embedding it would
        # create a second field and corrupt the language/text pairing.
        if '\r' in language:
            raise ValueError(f'subtitle entry {entry_index} language contains a carriage return')
        if '\r' in dialogue:
            raise ValueError(f'subtitle entry {entry_index} text contains a carriage return')

        content.extend(language.encode('utf-8'))
        content.append(0x0D)
        content.extend(dialogue.encode('utf-8'))
        content.append(0x0D)

    result = bytearray(struct.pack(endian + 'i', len(content)))
    result.extend(content)
    if suffix:
        result.extend(bytes(suffix))
    return bytes(result)


def format_subtitle_text(num_subtitles: int, frames, endian: str = '<') -> str:
    endian = _subtitle_endian(endian)
    frame_map = normalize_subtitle_frames(frames)
    document: dict = {
        'frames': [],
    }

    for frame, blob in sorted(frame_map.items()):
        if int(frame) < 0 or not subtitle_blob_has_data(blob):
            continue
        decoded = decode_subtitle_set(blob, endian=endian)
        if decoded is None:
            # Do not expose opaque binary data in the Scripting/Text Editor.
            continue
        entries, _suffix = decoded
        if not entries:
            continue
        display_entries = [
            {
                'language': subtitle_language_display_name(entry.get('language', '')),
                'text': str(entry.get('text', '') or ''),
            }
            for entry in entries
        ]
        document['frames'].append({
            'frame': int(frame),
            'subtitles': display_entries,
        })

    return json.dumps(document, ensure_ascii=False, indent=2) + '\n'

def _parse_hex_blob(token: str, label: str) -> bytes:
    value = ''.join(str(token or '').split()).replace('_', '')
    if value.lower().startswith('hex:'):
        value = value[4:]
    if len(value) % 2:
        raise ValueError(f'{label} must contain an even number of hex digits')
    try:
        return bytes.fromhex(value)
    except Exception as exc:
        raise ValueError(f'invalid {label}: {exc}') from exc


def _parse_subtitle_json(text: str, endian: str = '<') -> tuple[int, dict[int, bytes]]:
    try:
        document = json.loads(text)
    except Exception as exc:
        raise ValueError(f'invalid JSON: {exc}') from exc
    if not isinstance(document, Mapping):
        raise ValueError('subtitle document must be a JSON object')

    if 'num_subtitles' in document:
        try:
            num_subtitles = int(document.get('num_subtitles', 0))
        except Exception as exc:
            raise ValueError('num_subtitles must be an integer') from exc
        if num_subtitles < 0:
            raise ValueError('num_subtitles cannot be negative')
    else:
        # Clean v2 subtitle documents omit this binary/header field.
        num_subtitles = -1

    frame_items = document.get('frames', [])
    if frame_items is None:
        frame_items = []
    if not isinstance(frame_items, list):
        raise ValueError('frames must be a JSON array')

    frames: dict[int, bytes] = {}
    for item_index, item in enumerate(frame_items):
        if not isinstance(item, Mapping):
            raise ValueError(f'frames[{item_index}] must be an object')
        if 'frame' not in item:
            raise ValueError(f'frames[{item_index}] is missing frame')
        try:
            frame = int(item.get('frame'))
        except Exception as exc:
            raise ValueError(f'frames[{item_index}].frame must be an integer') from exc
        if frame < 0:
            raise ValueError(f'frames[{item_index}].frame cannot be negative')

        if 'raw_hex' in item:
            frames[frame] = _parse_hex_blob(str(item.get('raw_hex', '')), f'frames[{item_index}].raw_hex')
            continue

        entries = item.get('subtitles', item.get('entries', []))
        if entries is None:
            entries = []
        if not isinstance(entries, list):
            raise ValueError(f'frames[{item_index}].subtitles must be an array')
        suffix = b''
        if item.get('suffix_hex'):
            suffix = _parse_hex_blob(str(item.get('suffix_hex', '')), f'frames[{item_index}].suffix_hex')
        try:
            frames[frame] = encode_subtitle_set(entries, endian=endian, suffix=suffix)
        except Exception as exc:
            raise ValueError(f'frames[{item_index}]: {exc}') from exc

    return int(num_subtitles), frames


def _parse_v1_blob(value: str, line_number: int) -> bytes:
    token = str(value or '').strip()
    if not token:
        return b'\x00\x00\x00\x00'
    try:
        if token.lower().startswith('0x'):
            number = int(token, 16)
            if number < 0 or number > 0xFFFFFFFF:
                raise ValueError('32-bit value is outside 0x00000000..0xFFFFFFFF')
            return int(number).to_bytes(4, 'little', signed=False)
        if token.lower().startswith('hex:'):
            token = token[4:]
        token = ''.join(token.split()).replace('_', '')
        if len(token) % 2:
            raise ValueError('hex data must contain an even number of digits')
        return bytes.fromhex(token)
    except Exception as exc:
        raise ValueError(f'line {line_number}: invalid SubtitleSet value: {exc}') from exc


def _parse_subtitle_text_v1(text: str) -> tuple[int, dict[int, bytes]]:
    num_subtitles = 0
    frames: dict[int, bytes] = {}
    saw_count = False

    for line_number, raw_line in enumerate(str(text or '').splitlines(), 1):
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue

        match = _COUNT_RE.match(line)
        if match:
            num_subtitles = int(match.group(1))
            if num_subtitles < 0:
                raise ValueError(f'line {line_number}: num_subtitles cannot be negative')
            saw_count = True
            continue

        match = _FRAME_RE.match(line)
        if match:
            frame = int(match.group(1))
            if frame < 0:
                raise ValueError(f'line {line_number}: subtitle frame cannot be negative')
            frames[frame] = _parse_v1_blob(match.group(2), line_number)
            continue

        match = _FRAMES_RE.match(line)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            if start_frame < 0:
                raise ValueError(f'line {line_number}: subtitle frame cannot be negative')
            if end_frame < start_frame:
                raise ValueError(f'line {line_number}: frame range end is before start')
            if end_frame - start_frame > 1000000:
                raise ValueError(f'line {line_number}: frame range is unreasonably large')
            blob = _parse_v1_blob(match.group(3), line_number)
            for frame in range(start_frame, end_frame + 1):
                frames[frame] = blob
            continue

        raise ValueError(f'line {line_number}: expected num_subtitles, frame, or frames range')

    if not saw_count:
        raise ValueError('num_subtitles = ... is missing')
    return int(num_subtitles), frames


def parse_subtitle_text(text: str, endian: str = '<') -> tuple[int, dict[int, bytes]]:
    source = str(text or '').lstrip('\ufeff \t\r\n')
    if source.startswith('{'):
        return _parse_subtitle_json(source, endian=_subtitle_endian(endian))
    return _parse_subtitle_text_v1(str(text or ''))
