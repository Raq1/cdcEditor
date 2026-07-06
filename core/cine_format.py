from __future__ import annotations

import struct
from typing import Any, Dict, List, Tuple

CINE_SPLINE_DIRECTOR = 1
CINE_SPLINE_FOV = 2
CINE_SPLINE_MASTER_COMMAND = 3
CINE_SPLINE_3D = 4
CINE_MARKER = 0x1234

_SPLINE_TYPE_NAMES = {
    CINE_SPLINE_DIRECTOR: 'DirectorSpline',
    CINE_SPLINE_FOV: 'FOVSpline',
    CINE_SPLINE_MASTER_COMMAND: 'MasterCommandSpline',
    CINE_SPLINE_3D: 'CineThreeDSpline',
}


def _read_c_string(data: bytes, offset: int, *, limit: int | None = None) -> Tuple[str, int]:
    if offset < 0 or offset >= len(data):
        raise ValueError(f'Cine string offset 0x{offset:X} is outside payload size 0x{len(data):X}')
    max_end = len(data) if limit is None else min(len(data), offset + max(1, int(limit)))
    end = data.find(b'\0', offset, max_end)
    if end < 0:
        raise ValueError(f'Cine string at 0x{offset:X} is not null-terminated')
    raw = data[offset:end]
    try:
        text = raw.decode('utf-8', errors='strict')
    except UnicodeDecodeError as exc:
        raise ValueError(f'Cine string at 0x{offset:X} is not valid UTF-8/ASCII') from exc
    if any(ord(ch) < 32 or ord(ch) >= 127 for ch in text):
        raise ValueError(f'Cine string at 0x{offset:X} contains non-printable characters')
    return text, end + 1


# A lot of hopes and prayers here
def _align(value: int, alignment: int = 4) -> int:
    alignment = max(1, int(alignment))
    return (int(value) + alignment - 1) & ~(alignment - 1)


def _skip_zero_padding(data: bytes, offset: int, alignment: int = 4) -> int:
    aligned = _align(offset, alignment)
    if aligned <= len(data) and all(byte == 0 for byte in data[offset:aligned]):
        return aligned
    return offset


def _require(data: bytes, offset: int, size: int, what: str) -> None:
    if offset < 0 or offset + size > len(data):
        raise ValueError(f'Cine {what} at 0x{offset:X} exceeds payload size 0x{len(data):X}')


def _u16(data: bytes, offset: int) -> int:
    _require(data, offset, 2, 'u16')
    return int(struct.unpack_from('<H', data, offset)[0])


def _i16(data: bytes, offset: int) -> int:
    _require(data, offset, 2, 'i16')
    return int(struct.unpack_from('<h', data, offset)[0])


def _u32(data: bytes, offset: int) -> int:
    _require(data, offset, 4, 'u32')
    return int(struct.unpack_from('<I', data, offset)[0])


def _i32(data: bytes, offset: int) -> int:
    _require(data, offset, 4, 'i32')
    return int(struct.unpack_from('<i', data, offset)[0])


def _f32(data: bytes, offset: int) -> float:
    _require(data, offset, 4, 'f32')
    return float(struct.unpack_from('<f', data, offset)[0])


def _read_marker(data: bytes, offset: int, what: str) -> int:
    marker = _u16(data, offset)
    if marker != CINE_MARKER:
        raise ValueError(f'Cine {what} marker mismatch at 0x{offset:X}: got 0x{marker:04X}, expected 0x{CINE_MARKER:04X}')
    return offset + 2


def strip_standalone_section_header(raw_data: bytes) -> Tuple[bytes, int, int]:
    raw_data = bytes(raw_data or b'')
    if len(raw_data) >= 24 and raw_data[:4] in {b'SECT', b'TCES'}:
        try:
            size, _section_type, _skip_flags, _version_id, packed_data, _section_id, _spec_mask = struct.unpack_from('<iBBHIII', raw_data, 4)
            relocation_count = max(0, int((packed_data >> 8) & 0x00FFFFFF))
            payload_offset = 24 + (relocation_count * 8)
            if 0 <= payload_offset <= len(raw_data):
                declared = max(0, int(size))
                end = min(len(raw_data), payload_offset + declared) if declared else len(raw_data)
                return raw_data[payload_offset:end], payload_offset, declared
        except Exception:
            pass
        return raw_data[24:], 24, max(0, len(raw_data) - 24)
    return raw_data, 0, len(raw_data)


def _read_command_name_and_align(data: bytes, offset: int) -> Tuple[str, int, int]:
    name_offset = int(offset)
    name, offset = _read_c_string(data, offset)
    offset = _skip_zero_padding(data, offset, 4)
    return name, offset, name_offset


def parse_cinematic_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    unit_id = _i32(data, offset)
    load = _i16(data, offset + 4)
    camera_control = _i16(data, offset + 6)
    channels = _i16(data, offset + 8)
    positions_after_playback = _i16(data, offset + 10)
    end_trigger_id = _i32(data, offset + 12)
    data_pointer = _u32(data, offset + 16)
    data_size = _i32(data, offset + 20)
    cinematic_name, offset = _read_c_string(data, offset + 24)
    offset = _skip_zero_padding(data, offset, 4)
    command = {
        'type': 'Cinematic',
        'offset': start,
        'unit_id': int(unit_id),
        'load': int(load),
        'camera_control': int(camera_control),
        'channels': int(channels),
        'positions_after_playback': int(positions_after_playback),
        'end_trigger_id': int(end_trigger_id),
        'data_pointer': int(data_pointer),
        'data_size': int(data_size),
        'cinematic_name': str(cinematic_name),
    }
    return command, offset


def parse_script_message_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({
        'type': 'ScriptMessage',
        'offset': start,
        'unit_id': int(_i32(data, offset)),
        'trigger_id': int(_i32(data, offset + 4)),
    }, offset + 8)


def parse_effect_once_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    count = max(0, min(int(_i32(data, offset)), 8192))
    offset += 4
    entries: List[int] = []
    for _ in range(count):
        entries.append(int(_u32(data, offset)))
        offset += 4
    return ({
        'type': 'EffectOnce',
        'offset': start,
        'effect_count': int(count),
        'effect_data': entries,
    }, offset)


def parse_sound2d_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({
        'type': 'Sound2D',
        'offset': start,
        'mode': int(_i16(data, offset)),
        'id': int(_i32(data, offset + 4)),
    }, offset + 8)


def parse_sound_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({
        'type': 'Sound',
        'offset': start,
        'mode': int(_i16(data, offset)),
        'id': int(_i16(data, offset + 2)),
    }, offset + 4)


def parse_screen_wipe_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({
        'type': 'ScreenWipe',
        'offset': start,
        'mode': int(_i16(data, offset)),
        'action': int(_i16(data, offset + 2)),
        'duration': int(_i16(data, offset + 4)),
    }, offset + 6)


def parse_hide_cine_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({'type': 'HideCine', 'offset': start, 'hidden': int(data[offset])}, offset + 1)


def parse_floor_command_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({'type': 'Floor', 'offset': start, 'active': int(data[offset])}, offset + 1)


def parse_camera_select_payload(data: bytes, offset: int) -> Tuple[Dict[str, Any], int]:
    start = int(offset)
    return ({'type': 'CameraSelect', 'offset': start, 'resource': int(_i32(data, offset))}, offset + 4)


_COMMAND_PARSERS = {
    'Cinematic': parse_cinematic_command_payload,
    'ScriptMessage': parse_script_message_command_payload,
    'EffectOnce': parse_effect_once_command_payload,
    'Sound2D': parse_sound2d_command_payload,
    'Sound': parse_sound_command_payload,
    'ScreenWipe': parse_screen_wipe_command_payload,
    'HideCine': parse_hide_cine_command_payload,
    'Floor': parse_floor_command_payload,
    'CameraSelect': parse_camera_select_payload,
}


def parse_cine_commands(data: bytes, offset: int) -> Tuple[List[Dict[str, Any]], int]:
    command_count = _u16(data, offset)
    offset += 2
    commands: List[Dict[str, Any]] = []
    for command_index in range(command_count):
        command_name, body_offset, command_name_offset = _read_command_name_and_align(data, offset)
        parser = _COMMAND_PARSERS.get(command_name)
        if parser is None:
            raise ValueError(f'Unsupported Cine command {command_name!r} at 0x{command_name_offset:X}')
        command, offset = parser(data, body_offset)
        command['type'] = str(command_name)
        command['command_index'] = int(command_index)
        command['command_name_offset'] = int(command_name_offset)
        command['body_offset'] = int(body_offset)
        command['end_offset'] = int(offset)
        commands.append(command)
    return commands, offset


def parse_command_nodes(data: bytes, offset: int, node_count: int) -> Tuple[List[Dict[str, Any]], int]:
    nodes: List[Dict[str, Any]] = []
    if int(node_count) > 0:
        offset = _align(offset, 4)
    for node_index in range(int(node_count)):
        offset = _align(offset, 4)
        node_offset = offset
        time_value = _f32(data, offset)
        offset += 4
        commands, offset = parse_cine_commands(data, offset)
        nodes.append({
            'index': int(node_index),
            'offset': int(node_offset),
            'time': float(time_value),
            'commands': commands,
        })
    return nodes, offset


def parse_one_axis_nodes(data: bytes, offset: int, node_count: int) -> Tuple[List[Dict[str, Any]], int]:
    nodes: List[Dict[str, Any]] = []
    if int(node_count) > 0:
        offset = _align(offset, 4)
    for node_index in range(int(node_count)):
        node_offset = offset
        nodes.append({
            'index': int(node_index),
            'offset': int(node_offset),
            'time': float(_f32(data, offset)),
            'x': float(_f32(data, offset + 4)),
            'v': float(_f32(data, offset + 8)),
        })
        offset += 12
    return nodes, offset


def _parse_cine_3d_command_only_layout(data: bytes, offset: int, spline_offset: int) -> Tuple[Dict[str, Any], int]:
    header_offset = int(offset)
    _require(data, header_offset, 0x24, 'CineThreeDSpline header')
    header_blob = bytes(data[header_offset:header_offset + 0x24])
    focus_spline = int(_i32(data, header_offset + 0x00))
    clone_source = int(_i32(data, header_offset + 0x04))
    actor_id = int(_i32(data, header_offset + 0x10))
    command_node_count = int(_u16(data, header_offset + 0x22))
    nodes, offset = parse_command_nodes(data, header_offset + 0x24, command_node_count)
    spline = {
        'offset': int(spline_offset),
        'type': CINE_SPLINE_3D,
        'type_name': _SPLINE_TYPE_NAMES[CINE_SPLINE_3D],
        'node_count': int(command_node_count),
        'focus_spline': int(focus_spline),
        'clone_source': int(clone_source),
        'actor_id': int(actor_id),
        'raw_header': header_blob.hex(),
        'nodes': nodes,
    }
    return spline, offset


def _collect_commands(splines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    commands: List[Dict[str, Any]] = []
    for spline in splines:
        for node in spline.get('nodes') or []:
            for command in node.get('commands') or []:
                command_copy = dict(command)
                command_copy.setdefault('spline_index', spline.get('index'))
                command_copy.setdefault('spline_type', spline.get('type_name'))
                command_copy.setdefault('node_index', node.get('index'))
                command_copy.setdefault('node_time', node.get('time'))
                commands.append(command_copy)
    return commands


def parse_cine_payload(payload: bytes | bytearray) -> Dict[str, Any]:
    data = bytes(payload or b'')
    if len(data) < 0x20:
        raise ValueError(f'Cine payload too small: 0x{len(data):X}')

    offset = 0
    version_minor = _u16(data, offset)
    version_major = _u16(data, offset + 2)
    offset += 4
    if version_minor <= 0 or version_minor > 0x100 or version_major <= 0 or version_major > 0x100:
        raise ValueError(f'Unsupported Cine payload version pair {version_major}.{version_minor}')

    name, offset = _read_c_string(data, offset)
    offset = _skip_zero_padding(data, offset, 4)

    stream_unit_id = _i32(data, offset)
    cine_id = _i32(data, offset + 4)
    end_time = _f32(data, offset + 8)
    offset += 12

    spline_count = _u16(data, offset)
    offset += 2
    offset = _read_marker(data, offset, 'spline-list')

    splines: List[Dict[str, Any]] = []
    for spline_index in range(int(spline_count)):
        spline_offset = offset
        spline_type = _u16(data, offset)
        offset += 2
        if spline_type in {CINE_SPLINE_DIRECTOR, CINE_SPLINE_MASTER_COMMAND}:
            node_count = _u16(data, offset)
            offset += 2
            nodes, offset = parse_command_nodes(data, offset, node_count)
            offset = _skip_zero_padding(data, offset, 2)
            offset = _read_marker(data, offset, _SPLINE_TYPE_NAMES.get(int(spline_type), 'CommandSpline'))
            spline = {
                'index': int(spline_index),
                'offset': int(spline_offset),
                'type': int(spline_type),
                'type_name': _SPLINE_TYPE_NAMES.get(int(spline_type), f'UnknownSpline{int(spline_type)}'),
                'node_count': int(node_count),
                'nodes': nodes,
            }
        elif spline_type == CINE_SPLINE_FOV:
            node_count = _u16(data, offset)
            offset += 2
            nodes, offset = parse_one_axis_nodes(data, offset, node_count)
            offset = _skip_zero_padding(data, offset, 2)
            offset = _read_marker(data, offset, 'FOVSpline')
            spline = {
                'index': int(spline_index),
                'offset': int(spline_offset),
                'type': int(spline_type),
                'type_name': _SPLINE_TYPE_NAMES[CINE_SPLINE_FOV],
                'node_count': int(node_count),
                'nodes': nodes,
            }
        elif spline_type == CINE_SPLINE_3D:
            spline, offset = _parse_cine_3d_command_only_layout(data, offset, spline_offset)
            spline['index'] = int(spline_index)
            offset = _skip_zero_padding(data, offset, 2)
            offset = _read_marker(data, offset, 'CineThreeDSpline')
        else:
            raise ValueError(f'Unsupported Cine spline type={spline_type} at 0x{spline_offset:X}')
        splines.append(spline)

    trailing = data[offset:]
    if trailing and any(byte != 0 for byte in trailing):
        raise ValueError(f'Cine payload has non-zero trailing bytes at 0x{offset:X}: {trailing[:16].hex()}')

    commands = _collect_commands(splines)
    command_types = sorted({str(command.get('type', '')) for command in commands if command.get('type')})
    rebuildable = (
        len(splines) == 3
        and [int(s.get('type', 0)) for s in splines] == [CINE_SPLINE_DIRECTOR, CINE_SPLINE_FOV, CINE_SPLINE_MASTER_COMMAND]
        and len(commands) == 1
        and str(commands[0].get('type', '')) == 'Cinematic'
        and int(commands[0].get('data_size', 0) or 0) == 0
        and int(splines[0].get('node_count', 0) or 0) == 0
        and int(splines[1].get('node_count', 0) or 0) == 0
    )

    return {
        'format': 'cdc::CineDataResource',
        'version_major': int(version_major),
        'version_minor': int(version_minor),
        'name': str(name),
        'stream_unit_id': int(stream_unit_id),
        'cine_id': int(cine_id),
        'end_time': float(end_time),
        'spline_count': int(spline_count),
        'splines': splines,
        'commands': commands,
        'command_count': int(len(commands)),
        'command_types': ', '.join(command_types),
        'rebuildable': bool(rebuildable),
        'parsed_size': int(offset),
        'payload_size': int(len(data)),
    }


def parse_cine_section(raw_data: bytes | bytearray) -> Dict[str, Any]:
    payload, payload_offset, declared_size = strip_standalone_section_header(bytes(raw_data or b''))
    parsed = parse_cine_payload(payload)
    parsed['payload_offset'] = int(payload_offset)
    parsed['payload_size'] = int(len(payload))
    parsed['standalone_declared_size'] = int(declared_size)
    return parsed


def first_cinematic_command(parsed: Dict[str, Any]) -> Dict[str, Any] | None:
    for command in parsed.get('commands') or []:
        if str(command.get('type', '')) == 'Cinematic':
            return command
    for spline in parsed.get('splines') or []:
        for node in spline.get('nodes') or []:
            for command in node.get('commands') or []:
                if str(command.get('type', '')) == 'Cinematic':
                    return command
    return None


def cine_name_from_section(raw_data: bytes | bytearray) -> str:
    try:
        parsed = parse_cine_section(raw_data)
        return str(parsed.get('name', '') or '')
    except Exception:
        return ''


def _pack_cstring(text: object) -> bytes:
    value = str(text or '')
    encoded = value.encode('utf-8', errors='strict')
    if b'\0' in encoded:
        encoded = encoded.split(b'\0', 1)[0]
    return encoded + b'\0'


def _pad_to_alignment(out: bytearray, alignment: int = 4) -> None:
    while len(out) % max(1, int(alignment)):
        out.append(0)


def build_cine_payload_from_metadata(meta: Dict[str, Any]) -> bytes:
    name = str(meta.get('name', '') or meta.get('cinematic_name', '') or 'Cinematic')
    version_minor = int(meta.get('version_minor', 2) or 2)
    version_major = int(meta.get('version_major', 1) or 1)
    stream_unit_id = int(meta.get('stream_unit_id', 0) or 0)
    cine_id = int(meta.get('cine_id', 0) or 0)
    end_time = float(meta.get('end_time', 0.0) or 0.0)

    command = dict(meta.get('cinematic_command') or {})
    cinematic_name = str(command.get('cinematic_name', '') or meta.get('cinematic_name', '') or name)
    unit_id = int(command.get('unit_id', meta.get('command_unit_id', 0)) or 0)
    load = int(command.get('load', meta.get('command_load', 0)) or 0)
    camera_control = int(command.get('camera_control', meta.get('command_camera_control', 0)) or 0)
    channels = int(command.get('channels', meta.get('command_channels', 0)) or 0)
    positions_after_playback = int(command.get('positions_after_playback', meta.get('command_positions_after_playback', 0)) or 0)
    end_trigger_id = int(command.get('end_trigger_id', meta.get('command_end_trigger_id', 0)) or 0)
    data_pointer = int(command.get('data_pointer', meta.get('command_data_pointer', 0)) or 0)
    data_size = int(command.get('data_size', meta.get('command_data_size', 0)) or 0)
    command_time = float(meta.get('master_command_time', command.get('time', 0.0)) or 0.0)

    out = bytearray()
    out += struct.pack('<HH', version_minor & 0xFFFF, version_major & 0xFFFF)
    out += _pack_cstring(name)
    _pad_to_alignment(out, 4)
    out += struct.pack('<ii f', stream_unit_id, cine_id, end_time)

    # not too sure about everything that's going on here, but it somewhat works
    out += struct.pack('<HH', 3, CINE_MARKER)
    out += struct.pack('<HHH', CINE_SPLINE_DIRECTOR, 0, CINE_MARKER)
    out += struct.pack('<HHH', CINE_SPLINE_FOV, 0, CINE_MARKER)
    out += struct.pack('<HH', CINE_SPLINE_MASTER_COMMAND, 1)
    _pad_to_alignment(out, 4)
    out += struct.pack('<fH', command_time, 1)
    out += _pack_cstring('Cinematic')
    _pad_to_alignment(out, 4)
    out += struct.pack(
        '<ihhhhIIi',
        unit_id,
        max(-0x8000, min(0x7FFF, load)),
        max(-0x8000, min(0x7FFF, camera_control)),
        max(-0x8000, min(0x7FFF, channels)),
        max(-0x8000, min(0x7FFF, positions_after_playback)),
        end_trigger_id & 0xFFFFFFFF,
        data_pointer & 0xFFFFFFFF,
        max(-0x80000000, min(0x7FFFFFFF, data_size)),
    )
    out += _pack_cstring(cinematic_name)
    _pad_to_alignment(out, 2)
    out += struct.pack('<H', CINE_MARKER)
    return bytes(out)
