from __future__ import annotations

import base64
import copy
import json
import struct
import zlib
from typing import Any, Dict, Iterable, Optional, Tuple

ROOT_COMBAT_FIELDS = (
    # Level::splineCameraData and the marker lists are runtime dependencies of
    # combat waves. They are embedded and relinked alongside the wave lists.
    ('splineCameraData', 0x98),
    ('attackWaveList', 0xC8),
    ('attackWaveGroupList', 0xCC),
    ('combatDoorsList', 0xD0),
    ('vmarkerList', 0xE0),
    ('pmarkerList', 0xE4),
)

COMBAT_BLOB_MAGIC = b'TRCB2\x00'
COMBAT_BLOB_VERSION = 3

_ATTACK_WAVE_DEFINITION_FIELDS = (
    'flags',
    'numAttackerSpawns',
    'numAttackersPerWave',
    'ignoreTimeForFirst',
    'probability',
    'triggerRemaining',
    'numLinkedAttackWaves',
    'numAttackers',
    'numAttackWaveVars',
    'numBlockingAttackWaves',
    'numFinishedAttackWaves',
    'musicRank',
)

_ATTACK_WAVE_RUNTIME_BYTE_FIELDS = (
    'flags',
    'rtFlags',
    'probability',
    'numAttackerSpawns',
    'numSpawnsThisLoad',
    'currentAttacker',
    'triggerRemaining',
    'numAttackersPerWave',
    'numLinkedAttackWaves',
    'numBlockedAttackWaves',
    'numFinishedAttackWaves',
    'numAttackers',
    'musicRank',
)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(default) if value is None else int(value)
    except Exception:
        return int(default)


def _json_default(value: Any):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'__bytes__': base64.b64encode(bytes(value)).decode('ascii')}
    raise TypeError(f'Unsupported combat metadata value: {type(value)!r}')


def _json_object_hook(value: Dict[str, Any]):
    encoded = value.get('__bytes__')
    if isinstance(encoded, str) and len(value) == 1:
        try:
            return base64.b64decode(encoded.encode('ascii'))
        except Exception:
            return b''
    return value


def encode_combat_graph(graph: Dict[str, Any]) -> bytes:
    payload = json.dumps(graph or {}, separators=(',', ':'), ensure_ascii=False, default=_json_default).encode('utf-8')
    return COMBAT_BLOB_MAGIC + zlib.compress(payload, 9)


def decode_combat_graph(blob: bytes | bytearray | memoryview) -> Dict[str, Any]:
    raw = bytes(blob or b'')
    if not raw:
        return {}
    try:
        if raw.startswith(COMBAT_BLOB_MAGIC):
            raw = zlib.decompress(raw[len(COMBAT_BLOB_MAGIC):])
        result = json.loads(raw.decode('utf-8'), object_hook=_json_object_hook)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def clone_combat_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(graph or {})


def _section_record(graph: Dict[str, Any], section_index: int) -> Optional[Dict[str, Any]]:
    section_index = int(section_index)
    if section_index == 0:
        tail_start = _as_int(graph.get('root_tail_start'), -1)
        tail_data = graph.get('root_tail_data', b'')
        if tail_start < 0 or not isinstance(tail_data, (bytes, bytearray, memoryview)):
            return None
        if not isinstance(tail_data, bytearray):
            tail_data = bytearray(bytes(tail_data))
            graph['root_tail_data'] = tail_data
        relocations = graph.get('root_tail_relocations')
        if not isinstance(relocations, list):
            relocations = []
            graph['root_tail_relocations'] = relocations
        return {
            'source_index': 0,
            'base_offset': tail_start,
            'data': tail_data,
            'relocations': relocations,
        }
    for section in list(graph.get('sections', []) or []):
        if _as_int(section.get('source_index'), -1) == section_index:
            return section
    return None


def _section_base(section: Dict[str, Any]) -> int:
    return _as_int(section.get('base_offset'), 0)


def _section_data(section: Dict[str, Any]) -> bytes:
    value = section.get('data', b'')
    return bytes(value) if isinstance(value, (bytes, bytearray, memoryview)) else b''


def _section_mutable_data(section: Dict[str, Any]) -> bytearray:
    value = section.get('data', b'')
    if isinstance(value, bytearray):
        return value
    mutable = bytearray(bytes(value) if isinstance(value, (bytes, memoryview)) else b'')
    section['data'] = mutable
    return mutable


def _local_index(section: Dict[str, Any], target_offset: int) -> int:
    return int(target_offset) - _section_base(section)


def _read_at(graph: Dict[str, Any], ref: Dict[str, Any], size: int) -> bytes:
    section = _section_record(graph, _as_int(ref.get('section'), -1))
    if section is None:
        return b''
    index = _local_index(section, _as_int(ref.get('offset'), 0))
    data = _section_data(section)
    if index < 0 or index + int(size) > len(data):
        return b''
    return data[index:index + int(size)]


def _write_at(graph: Dict[str, Any], ref: Dict[str, Any], payload: bytes) -> bool:
    section = _section_record(graph, _as_int(ref.get('section'), -1))
    if section is None:
        return False
    index = _local_index(section, _as_int(ref.get('offset'), 0))
    data = _section_mutable_data(section)
    if index < 0 or index + len(payload) > len(data):
        return False
    data[index:index + len(payload)] = payload
    return True


def _relocations_for_section(graph: Dict[str, Any], section_index: int) -> Iterable[Dict[str, Any]]:
    section = _section_record(graph, section_index)
    if section is None:
        return ()
    return list(section.get('relocations', []) or [])


def resolve_pointer_ref(graph: Dict[str, Any], owner_section_index: int, pointer_field_offset: int) -> Optional[Dict[str, int]]:
    owner_section_index = int(owner_section_index)
    pointer_field_offset = int(pointer_field_offset)
    for relocation in _relocations_for_section(graph, owner_section_index):
        if _as_int(relocation.get('source_offset'), -1) != pointer_field_offset:
            continue
        if _as_int(relocation.get('relocation_type'), 0) != 0:
            continue
        return {
            'section': _as_int(relocation.get('target_section'), -1),
            'offset': _as_int(relocation.get('target_offset'), 0),
        }
    owner = _section_record(graph, owner_section_index)
    if owner is None:
        return None
    local = pointer_field_offset - _section_base(owner)
    data = _section_data(owner)
    if local < 0 or local + 4 > len(data):
        return None
    raw_target = int(struct.unpack_from('<I', data, local)[0])
    section_base = _section_base(owner)
    if raw_target == 0 or raw_target == 0xFFFFFFFF:
        return None
    if raw_target < section_base or raw_target >= section_base + len(data):
        return None
    return {'section': owner_section_index, 'offset': raw_target}


def _pointer_values(graph: Dict[str, Any], owner_section: int, field_offset: int, prefix: str) -> Dict[str, int]:
    ref = resolve_pointer_ref(graph, owner_section, field_offset)
    return {
        f'{prefix}TargetSection': _as_int(ref.get('section'), -1) if ref else -1,
        f'{prefix}TargetOffset': _as_int(ref.get('offset'), 0) if ref else 0,
    }


def _patch_pointer_ref(
    graph: Dict[str, Any],
    owner_section_index: int,
    pointer_field_offset: int,
    target_section_index: int,
    target_offset: int,
) -> None:
    owner = _section_record(graph, owner_section_index)
    if owner is None:
        return
    relocations = owner.get('relocations')
    if not isinstance(relocations, list):
        relocations = []
        owner['relocations'] = relocations
        if int(owner_section_index) == 0:
            graph['root_tail_relocations'] = relocations
    matching = [
        relocation for relocation in relocations
        if _as_int(relocation.get('source_offset'), -1) == int(pointer_field_offset)
        and _as_int(relocation.get('relocation_type'), 0) == 0
    ]
    target_section_index = int(target_section_index)
    target_offset = int(target_offset)
    if target_section_index < 0:
        relocations[:] = [relocation for relocation in relocations if relocation not in matching]
        _write_at(graph, {'section': owner_section_index, 'offset': pointer_field_offset}, struct.pack('<I', 0))
        return
    if matching:
        relocation = matching[0]
        relocation['target_section'] = target_section_index
        relocation['target_offset'] = target_offset
        for duplicate in matching[1:]:
            try:
                relocations.remove(duplicate)
            except ValueError:
                pass
    else:
        relocations.append({
            'source_offset': int(pointer_field_offset),
            'target_section': target_section_index,
            'target_offset': target_offset,
            'relocation_type': 0,
            'type_specific': 0,
        })
    _write_at(
        graph,
        {'section': owner_section_index, 'offset': pointer_field_offset},
        struct.pack('<I', target_offset & 0xFFFFFFFF),
    )


def _patch_c_string_pointer(
    graph: Dict[str, Any],
    owner_section: int,
    pointer_field_offset: int,
    value: Any,
    *,
    limit: int = 256,
) -> None:
    current_ref = resolve_pointer_ref(graph, owner_section, pointer_field_offset)
    current_name = _read_c_string(graph, current_ref, limit) if current_ref else ''
    encoded = str(value or '').encode('utf-8', errors='ignore')[:max(0, int(limit) - 1)]
    current_capacity = max(1, len(current_name.encode('utf-8', errors='ignore')) + 1)
    if current_ref and len(encoded) + 1 <= current_capacity:
        _write_at(graph, current_ref, encoded + (b'\x00' * (current_capacity - len(encoded))))
        return
    if not encoded and current_ref is None:
        return
    target_section_index = _as_int(current_ref.get('section'), owner_section) if current_ref else int(owner_section)
    target_section = _section_record(graph, target_section_index)
    if target_section is None:
        target_section_index = int(owner_section)
        target_section = _section_record(graph, target_section_index)
    if target_section is None:
        return
    target_data = _section_mutable_data(target_section)
    target_offset = _section_base(target_section) + len(target_data)
    target_data.extend(encoded + b'\x00')
    _patch_pointer_ref(graph, owner_section, pointer_field_offset, target_section_index, target_offset)


def _patch_pointer_fields(
    graph: Dict[str, Any],
    owner_section: int,
    base_offset: int,
    values: Dict[str, Any],
    fields: Iterable[Tuple[str, int]],
) -> None:
    for prefix, relative_offset in fields:
        section_key = f'{prefix}TargetSection'
        offset_key = f'{prefix}TargetOffset'
        if section_key not in values and offset_key not in values:
            continue
        current_ref = resolve_pointer_ref(graph, owner_section, base_offset + int(relative_offset))
        current_section = _as_int(current_ref.get('section'), -1) if current_ref else -1
        current_offset = _as_int(current_ref.get('offset'), 0) if current_ref else 0
        new_section = _as_int(values.get(section_key), current_section)
        new_offset = _as_int(values.get(offset_key), current_offset)
        if new_section == current_section and new_offset == current_offset:
            continue
        _patch_pointer_ref(
            graph,
            owner_section,
            base_offset + int(relative_offset),
            new_section,
            new_offset,
        )


def _read_c_string(graph: Dict[str, Any], ref: Optional[Dict[str, Any]], limit: int = 512) -> str:
    if not ref:
        return ''
    section = _section_record(graph, _as_int(ref.get('section'), -1))
    if section is None:
        return ''
    index = _local_index(section, _as_int(ref.get('offset'), 0))
    data = _section_data(section)
    if index < 0 or index >= len(data):
        return ''
    end = index
    max_end = min(len(data), index + max(1, int(limit)))
    while end < max_end and data[end] != 0:
        end += 1
    try:
        return data[index:end].decode('utf-8', errors='ignore')
    except Exception:
        return ''


def _parse_attack_wave_definition(graph: Dict[str, Any], ref: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ref:
        return {}
    raw = _read_at(graph, ref, 36)
    if len(raw) < 36:
        return {}
    values = struct.unpack_from('<12BHhhhIIII', raw, 0)
    result = {name: int(values[index]) for index, name in enumerate(_ATTACK_WAVE_DEFINITION_FIELDS)}
    result.update({
        'radiusCheckMarker': int(values[12]),
        'radiusCheckTime': int(values[13]),
        'radiusCheckRadius': int(values[14]),
        'pad': int(values[15]),
    })
    owner_section = _as_int(ref.get('section'), -1)
    base_offset = _as_int(ref.get('offset'), 0)
    for prefix, relative_offset in (
        ('enableSignal', 20),
        ('disableSignal', 24),
        ('killSignal', 28),
        ('data', 32),
    ):
        result.update(_pointer_values(graph, owner_section, base_offset + relative_offset, prefix))
    return result


def _parse_runtime_marker_name(graph: Dict[str, Any], owner_section: int, field_offset: int) -> str:
    return _read_c_string(graph, resolve_pointer_ref(graph, owner_section, field_offset), 256)


def _parse_attack_wave_runtime(graph: Dict[str, Any], ref: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ref:
        return {}
    raw = _read_at(graph, ref, 804)
    if len(raw) < 804:
        return {}
    result: Dict[str, Any] = {'radiusCheckTimer': float(struct.unpack_from('<f', raw, 0)[0])}
    for index, field_name in enumerate(_ATTACK_WAVE_RUNTIME_BYTE_FIELDS):
        result[field_name] = int(raw[4 + index])
    result['pad'] = int(struct.unpack_from('<b', raw, 17)[0])
    result['radiusCheckMarker'] = int(struct.unpack_from('<H', raw, 18)[0])
    result['radiusCheckRadius'] = int(struct.unpack_from('<h', raw, 20)[0])
    owner_section = _as_int(ref.get('section'), -1)
    base_offset = _as_int(ref.get('offset'), 0)
    for prefix, relative_offset in (
        ('nextEnableAttackWave', 24),
        ('nextDisableAttackWave', 28),
        ('nextKillAttackWave', 32),
        ('data', 800),
    ):
        result.update(_pointer_values(graph, owner_section, base_offset + relative_offset, prefix))

    cursor = 36
    num_limits = int(struct.unpack_from('<i', raw, cursor)[0]); cursor += 4
    limits = []
    for index in range(6):
        values = struct.unpack_from('<5f', raw, cursor); cursor += 20
        if index < max(0, min(6, num_limits)):
            limits.append(dict(zip(('x', 'y', 'z', 'rad', 'pursueRad'), (float(v) for v in values))))
    result['numLimits'] = num_limits
    result['limits'] = limits

    num_box = int(struct.unpack_from('<i', raw, cursor)[0]); cursor += 4
    box_limits = []
    for index in range(6):
        values = struct.unpack_from('<4f4h', raw, cursor); cursor += 24
        if index < max(0, min(6, num_box)):
            box_limits.append({
                'x': float(values[0]), 'y': float(values[1]), 'z': float(values[2]), 'zrot': float(values[3]),
                'width': int(values[4]), 'length': int(values[5]), 'pursueWidth': int(values[6]), 'pursueLength': int(values[7]),
            })
    result['numBoxLimits'] = num_box
    result['boxLimits'] = box_limits

    num_plane = int(struct.unpack_from('<i', raw, cursor)[0]); cursor += 4
    plane_limits = []
    for index in range(1):
        values = struct.unpack_from('<6f', raw, cursor); cursor += 24
        if index < max(0, min(1, num_plane)):
            plane_limits.append(dict(zip(('px', 'py', 'pz', 'nx', 'ny', 'nz'), (float(v) for v in values))))
    result['numPlaneLimits'] = num_plane
    result['planeLimits'] = plane_limits

    num_run = int(struct.unpack_from('<i', raw, cursor)[0]); cursor += 4
    run_positions = []
    for index in range(6):
        values = struct.unpack_from('<3f3hH', raw, cursor)
        marker_field_offset = base_offset + cursor + 20
        marker_name = _parse_runtime_marker_name(graph, owner_section, marker_field_offset)
        cursor += 24
        if index < max(0, min(6, num_run)):
            run_positions.append({
                'x': float(values[0]), 'y': float(values[1]), 'z': float(values[2]),
                'priority': int(values[3]), 'waittime': int(values[4]), 'failAction': int(values[5]),
                'markerName': marker_name,
            })
    result['numRunAndGunPos'] = num_run
    result['runAndGunPositions'] = run_positions

    num_patrol, patrol_type = struct.unpack_from('<hh', raw, cursor); cursor += 4
    patrol_positions = []
    for index in range(8):
        values = struct.unpack_from('<6f4h', raw, cursor); cursor += 32
        if index < max(0, min(8, int(num_patrol))):
            patrol_positions.append({
                'x': float(values[0]), 'y': float(values[1]), 'z': float(values[2]),
                'lookx': float(values[3]), 'looky': float(values[4]), 'lookz': float(values[5]),
                'waittime': int(values[6]), 'anim': int(values[7]), 'mode': int(values[8]), 'look': int(values[9]),
            })
    result['numPatrolPos'] = int(num_patrol)
    result['patrolType'] = int(patrol_type)
    result['patrolPositions'] = patrol_positions

    num_messages = int(struct.unpack_from('<i', raw, 744)[0])
    messages = []
    cursor = 748
    for index in range(6):
        message, data_value = struct.unpack_from('<ii', raw, cursor); cursor += 8
        if index < max(0, min(6, num_messages)):
            messages.append({'message': int(message), 'data': int(data_value)})
    result['numMessages'] = num_messages
    result['messages'] = messages
    result['endMarker'] = int(struct.unpack_from('<i', raw, 796)[0])
    return result


def _parse_packed_attack_wave_list(graph: Dict[str, Any], list_name: str, list_ref: Dict[str, Any]) -> Dict[str, Any]:
    header = _read_at(graph, list_ref, 4)
    if len(header) < 4:
        return {'name': list_name, 'count': 0, 'entries': []}
    count = int(struct.unpack_from('<i', header, 0)[0])
    if count < 0 or count > 4096:
        return {'name': list_name, 'count': count, 'entries': [], 'layout': 'unknown'}
    owner_section = _as_int(list_ref.get('section'), -1)
    base = _as_int(list_ref.get('offset'), 0)
    ids_start = base + 4
    cursor = (ids_start + (count * 2) + 3) & ~3
    runtime_ptr_start = cursor
    definition_ptr_start = runtime_ptr_start + (count * 4)
    name_ptr_start = definition_ptr_start + (count * 4)
    required_end = name_ptr_start + (count * 4)
    owner = _section_record(graph, owner_section)
    if owner is None or _local_index(owner, required_end) > len(_section_data(owner)):
        return {'name': list_name, 'count': count, 'entries': [], 'layout': 'unknown'}

    entries = []
    for index in range(count):
        id_ref = {'section': owner_section, 'offset': ids_start + (index * 2)}
        raw_id = _read_at(graph, id_ref, 2)
        entry_id = int(struct.unpack_from('<H', raw_id, 0)[0]) if len(raw_id) == 2 else 0
        runtime_field = runtime_ptr_start + (index * 4)
        definition_field = definition_ptr_start + (index * 4)
        name_field = name_ptr_start + (index * 4)
        runtime_ref = resolve_pointer_ref(graph, owner_section, runtime_field)
        definition_ref = resolve_pointer_ref(graph, owner_section, definition_field)
        name_ref = resolve_pointer_ref(graph, owner_section, name_field)
        name_value = _read_c_string(graph, name_ref, 256)
        entries.append({
            'index': index,
            'id': entry_id,
            'name': name_value,
            'id_ref': id_ref,
            'runtime_pointer_field': {'section': owner_section, 'offset': runtime_field},
            'definition_pointer_field': {'section': owner_section, 'offset': definition_field},
            'name_pointer_field': {'section': owner_section, 'offset': name_field},
            'runtime_ref': runtime_ref,
            'definition_ref': definition_ref,
            'name_ref': name_ref,
            'name_capacity': max(1, len(name_value.encode('utf-8', errors='ignore')) + 1),
            'runtime': _parse_attack_wave_runtime(graph, runtime_ref),
            'definition': _parse_attack_wave_definition(graph, definition_ref),
        })
    return {
        'name': list_name,
        'count': count,
        'layout': 'packed_attack_wave_v1',
        'list_ref': dict(list_ref),
        'entries': entries,
    }



_PMARKER_STRUCT = struct.Struct('<6f6hH4hBb')
_PMARKER_FIELDS = (
    'lx', 'ly', 'lz', 'lrotx', 'lroty', 'lrotz',
    'mx', 'my', 'mz', 'px', 'py', 'pz',
    'flags', 'minDist', 'maxDist', 'maxCombatDist', 'pauseTime',
    'spawnType', 'padding',
)


def _parse_pmarker_list(graph: Dict[str, Any], list_ref: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not list_ref:
        return {'name': 'pmarkerList', 'count': 0, 'entries': []}
    header = _read_at(graph, list_ref, 4)
    if len(header) < 4:
        return {'name': 'pmarkerList', 'count': 0, 'entries': []}
    count = int(struct.unpack_from('<i', header, 0)[0])
    if count < 0 or count > 4096:
        return {'name': 'pmarkerList', 'count': count, 'entries': [], 'layout': 'unknown'}
    owner_section = _as_int(list_ref.get('section'), -1)
    base = _as_int(list_ref.get('offset'), 0)
    ids_start = base + 4
    records_start = (ids_start + (count * 2) + 3) & ~3
    names_start = records_start + (count * _PMARKER_STRUCT.size)
    required_end = names_start + (count * 4)
    owner = _section_record(graph, owner_section)
    if owner is None or _local_index(owner, required_end) > len(_section_data(owner)):
        return {'name': 'pmarkerList', 'count': count, 'entries': [], 'layout': 'unknown'}
    entries = []
    for index in range(count):
        id_ref = {'section': owner_section, 'offset': ids_start + (index * 2)}
        record_ref = {'section': owner_section, 'offset': records_start + (index * _PMARKER_STRUCT.size)}
        name_field = names_start + (index * 4)
        raw_id = _read_at(graph, id_ref, 2)
        raw_record = _read_at(graph, record_ref, _PMARKER_STRUCT.size)
        if len(raw_record) != _PMARKER_STRUCT.size:
            continue
        values = _PMARKER_STRUCT.unpack(raw_record)
        item = {field_name: values[field_index] for field_index, field_name in enumerate(_PMARKER_FIELDS)}
        for field_name in ('mx', 'my', 'mz', 'px', 'py', 'pz', 'flags', 'minDist', 'maxDist', 'maxCombatDist', 'pauseTime', 'spawnType', 'padding'):
            item[field_name] = int(item[field_name])
        name_ref = resolve_pointer_ref(graph, owner_section, name_field)
        item.update({
            'index': index,
            'id': int(struct.unpack_from('<H', raw_id, 0)[0]) if len(raw_id) == 2 else 0,
            'name': _read_c_string(graph, name_ref, 256),
            'id_ref': id_ref,
            'record_ref': record_ref,
            'name_pointer_field': {'section': owner_section, 'offset': name_field},
            'name_ref': name_ref,
        })
        entries.append(item)
    return {
        'name': 'pmarkerList',
        'count': count,
        'layout': 'packed_pmarker_v1',
        'list_ref': dict(list_ref),
        'entries': entries,
    }


def _patch_pmarker_entry(graph: Dict[str, Any], source_entry: Dict[str, Any], edited: Dict[str, Any]) -> None:
    id_ref = source_entry.get('id_ref')
    if isinstance(id_ref, dict):
        _write_at(graph, id_ref, struct.pack('<H', _clamp_int(edited.get('id', source_entry.get('id', 0)), 0, 65535)))
    record_ref = source_entry.get('record_ref')
    if isinstance(record_ref, dict):
        current = bytearray(_read_at(graph, record_ref, _PMARKER_STRUCT.size))
        if len(current) == _PMARKER_STRUCT.size:
            old = list(_PMARKER_STRUCT.unpack(current))
            floats = [float(edited.get(name, source_entry.get(name, old[i])) or 0.0) for i, name in enumerate(_PMARKER_FIELDS[:6])]
            signed16_names = ('mx', 'my', 'mz', 'px', 'py', 'pz')
            signed16 = [_clamp_int(edited.get(name, source_entry.get(name, old[6 + i])), -32768, 32767) for i, name in enumerate(signed16_names)]
            flags = _clamp_int(edited.get('flags', source_entry.get('flags', old[12])), 0, 65535)
            trailing16_names = ('minDist', 'maxDist', 'maxCombatDist', 'pauseTime')
            trailing16 = [_clamp_int(edited.get(name, source_entry.get(name, old[13 + i])), -32768, 32767) for i, name in enumerate(trailing16_names)]
            spawn_type = _clamp_int(edited.get('spawnType', source_entry.get('spawnType', old[17])), 0, 255)
            padding = _clamp_int(edited.get('padding', source_entry.get('padding', old[18])), -128, 127)
            _write_at(graph, record_ref, _PMARKER_STRUCT.pack(*(floats + signed16 + [flags] + trailing16 + [spawn_type, padding])))
    name_pointer_field = source_entry.get('name_pointer_field')
    if isinstance(name_pointer_field, dict):
        _patch_c_string_pointer(
            graph,
            _as_int(name_pointer_field.get('section'), -1),
            _as_int(name_pointer_field.get('offset'), 0),
            edited.get('name', source_entry.get('name', '')),
            limit=256,
        )

def rebuild_combat_summary(graph: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    field_map = {str(field.get('name', '')): field for field in list(graph.get('root_fields', []) or [])}
    for list_name in ('attackWaveList', 'attackWaveGroupList'):
        field = field_map.get(list_name)
        if not field:
            summary[list_name] = {'name': list_name, 'count': 0, 'entries': []}
            continue
        list_ref = {'section': _as_int(field.get('target_section'), -1), 'offset': _as_int(field.get('target_offset'), 0)}
        summary[list_name] = _parse_packed_attack_wave_list(graph, list_name, list_ref)

    doors_field = field_map.get('combatDoorsList')
    if doors_field:
        doors_ref = {'section': _as_int(doors_field.get('target_section'), -1), 'offset': _as_int(doors_field.get('target_offset'), 0)}
        raw = _read_at(graph, doors_ref, 4)
        count = int(struct.unpack_from('<i', raw, 0)[0]) if len(raw) == 4 else 0
        summary['combatDoorsList'] = {'name': 'combatDoorsList', 'count': count, 'list_ref': doors_ref, 'layout': 'raw'}
    else:
        summary['combatDoorsList'] = {'name': 'combatDoorsList', 'count': 0, 'layout': 'raw'}

    pmarker_field = field_map.get('pmarkerList')
    if pmarker_field:
        pmarker_ref = {'section': _as_int(pmarker_field.get('target_section'), -1), 'offset': _as_int(pmarker_field.get('target_offset'), 0)}
        summary['pmarkerList'] = _parse_pmarker_list(graph, pmarker_ref)
    else:
        summary['pmarkerList'] = {'name': 'pmarkerList', 'count': 0, 'entries': []}

    summary['splineCameraDataPresent'] = bool(field_map.get('splineCameraData'))

    vmarker_field = field_map.get('vmarkerList')
    if vmarker_field:
        vmarker_ref = {'section': _as_int(vmarker_field.get('target_section'), -1), 'offset': _as_int(vmarker_field.get('target_offset'), 0)}
        raw = _read_at(graph, vmarker_ref, 4)
        count = int(struct.unpack_from('<i', raw, 0)[0]) if len(raw) == 4 else 0
        summary['vmarkerList'] = {'name': 'vmarkerList', 'count': count, 'list_ref': vmarker_ref, 'layout': 'raw'}
    else:
        summary['vmarkerList'] = {'name': 'vmarkerList', 'count': 0, 'layout': 'raw'}
    graph['summary'] = summary
    return summary


def _clamp_int(value: Any, minimum: int, maximum: int, default: int = 0) -> int:
    try:
        result = int(value)
    except Exception:
        result = int(default)
    return max(int(minimum), min(int(maximum), result))


def _patch_definition(graph: Dict[str, Any], ref: Optional[Dict[str, Any]], values: Dict[str, Any]) -> None:
    if not ref or not isinstance(values, dict):
        return
    current = bytearray(_read_at(graph, ref, 36))
    if len(current) < 36:
        return
    for index, field_name in enumerate(_ATTACK_WAVE_DEFINITION_FIELDS):
        current[index] = _clamp_int(values.get(field_name, current[index]), 0, 255)
    struct.pack_into('<Hhhh', current, 12,
        _clamp_int(values.get('radiusCheckMarker', struct.unpack_from('<H', current, 12)[0]), 0, 65535),
        _clamp_int(values.get('radiusCheckTime', struct.unpack_from('<h', current, 14)[0]), -32768, 32767),
        _clamp_int(values.get('radiusCheckRadius', struct.unpack_from('<h', current, 16)[0]), -32768, 32767),
        _clamp_int(values.get('pad', struct.unpack_from('<h', current, 18)[0]), -32768, 32767),
    )
    _write_at(graph, ref, current)
    owner_section = _as_int(ref.get('section'), -1)
    base_offset = _as_int(ref.get('offset'), 0)
    _patch_pointer_fields(graph, owner_section, base_offset, values, (
        ('enableSignal', 20),
        ('disableSignal', 24),
        ('killSignal', 28),
        ('data', 32),
    ))


def _patch_runtime(graph: Dict[str, Any], ref: Optional[Dict[str, Any]], values: Dict[str, Any]) -> None:
    if not ref or not isinstance(values, dict):
        return
    current = bytearray(_read_at(graph, ref, 804))
    if len(current) < 804:
        return
    owner_section = _as_int(ref.get('section'), -1)
    base_offset = _as_int(ref.get('offset'), 0)
    try:
        struct.pack_into('<f', current, 0, float(values.get('radiusCheckTimer', struct.unpack_from('<f', current, 0)[0]) or 0.0))
    except Exception:
        pass
    for index, field_name in enumerate(_ATTACK_WAVE_RUNTIME_BYTE_FIELDS):
        current[4 + index] = _clamp_int(values.get(field_name, current[4 + index]), 0, 255)
    struct.pack_into('<b', current, 17, _clamp_int(values.get('pad', struct.unpack_from('<b', current, 17)[0]), -128, 127))
    struct.pack_into('<Hh', current, 18,
        _clamp_int(values.get('radiusCheckMarker', struct.unpack_from('<H', current, 18)[0]), 0, 65535),
        _clamp_int(values.get('radiusCheckRadius', struct.unpack_from('<h', current, 20)[0]), -32768, 32767),
    )

    cursor = 36
    limits = list(values.get('limits', []) or [])
    num_limits = _clamp_int(values.get('numLimits', len(limits)), 0, 6)
    struct.pack_into('<i', current, cursor, num_limits); cursor += 4
    for index in range(6):
        item = limits[index] if index < len(limits) and isinstance(limits[index], dict) else {}
        existing = struct.unpack_from('<5f', current, cursor)
        struct.pack_into('<5f', current, cursor, *(float(item.get(name, existing[i]) or 0.0) for i, name in enumerate(('x', 'y', 'z', 'rad', 'pursueRad')))); cursor += 20

    box_limits = list(values.get('boxLimits', []) or [])
    num_box = _clamp_int(values.get('numBoxLimits', len(box_limits)), 0, 6)
    struct.pack_into('<i', current, cursor, num_box); cursor += 4
    for index in range(6):
        item = box_limits[index] if index < len(box_limits) and isinstance(box_limits[index], dict) else {}
        existing = struct.unpack_from('<4f4h', current, cursor)
        floats = [float(item.get(name, existing[i]) or 0.0) for i, name in enumerate(('x', 'y', 'z', 'zrot'))]
        ints = [_clamp_int(item.get(name, existing[4 + i]), -32768, 32767) for i, name in enumerate(('width', 'length', 'pursueWidth', 'pursueLength'))]
        struct.pack_into('<4f4h', current, cursor, *(floats + ints)); cursor += 24

    plane_limits = list(values.get('planeLimits', []) or [])
    num_plane = _clamp_int(values.get('numPlaneLimits', len(plane_limits)), 0, 1)
    struct.pack_into('<i', current, cursor, num_plane); cursor += 4
    for index in range(1):
        item = plane_limits[index] if index < len(plane_limits) and isinstance(plane_limits[index], dict) else {}
        existing = struct.unpack_from('<6f', current, cursor)
        struct.pack_into('<6f', current, cursor, *(float(item.get(name, existing[i]) or 0.0) for i, name in enumerate(('px', 'py', 'pz', 'nx', 'ny', 'nz')))); cursor += 24

    run_positions = list(values.get('runAndGunPositions', []) or [])
    num_run = _clamp_int(values.get('numRunAndGunPos', len(run_positions)), 0, 6)
    struct.pack_into('<i', current, cursor, num_run); cursor += 4
    for index in range(6):
        item = run_positions[index] if index < len(run_positions) and isinstance(run_positions[index], dict) else {}
        existing = struct.unpack_from('<3f3hH', current, cursor)
        floats = [float(item.get(name, existing[i]) or 0.0) for i, name in enumerate(('x', 'y', 'z'))]
        ints = [_clamp_int(item.get(name, existing[3 + i]), -32768, 32767) for i, name in enumerate(('priority', 'waittime', 'failAction'))]
        struct.pack_into('<3f3hH', current, cursor, *(floats + ints + [existing[6]]))
        if 'markerName' in item:
            _patch_c_string_pointer(
                graph,
                owner_section,
                base_offset + cursor + 20,
                item.get('markerName', ''),
                limit=256,
            )
        cursor += 24

    patrol_positions = list(values.get('patrolPositions', []) or [])
    num_patrol = _clamp_int(values.get('numPatrolPos', len(patrol_positions)), 0, 8)
    patrol_type = _clamp_int(values.get('patrolType', struct.unpack_from('<h', current, cursor + 2)[0]), -32768, 32767)
    struct.pack_into('<hh', current, cursor, num_patrol, patrol_type); cursor += 4
    for index in range(8):
        item = patrol_positions[index] if index < len(patrol_positions) and isinstance(patrol_positions[index], dict) else {}
        existing = struct.unpack_from('<6f4h', current, cursor)
        floats = [float(item.get(name, existing[i]) or 0.0) for i, name in enumerate(('x', 'y', 'z', 'lookx', 'looky', 'lookz'))]
        ints = [_clamp_int(item.get(name, existing[6 + i]), -32768, 32767) for i, name in enumerate(('waittime', 'anim', 'mode', 'look'))]
        struct.pack_into('<6f4h', current, cursor, *(floats + ints)); cursor += 32

    messages = list(values.get('messages', []) or [])
    num_messages = _clamp_int(values.get('numMessages', len(messages)), 0, 6)
    struct.pack_into('<i', current, 744, num_messages)
    cursor = 748
    for index in range(6):
        item = messages[index] if index < len(messages) and isinstance(messages[index], dict) else {}
        old_message, old_data = struct.unpack_from('<ii', current, cursor)
        struct.pack_into('<ii', current, cursor,
            _clamp_int(item.get('message', old_message), -2147483648, 2147483647),
            _clamp_int(item.get('data', old_data), -2147483648, 2147483647),
        )
        cursor += 8
    struct.pack_into('<i', current, 796, _clamp_int(values.get('endMarker', struct.unpack_from('<i', current, 796)[0]), -2147483648, 2147483647))
    _write_at(graph, ref, current)
    _patch_pointer_fields(graph, owner_section, base_offset, values, (
        ('nextEnableAttackWave', 24),
        ('nextDisableAttackWave', 28),
        ('nextKillAttackWave', 32),
        ('data', 800),
    ))


def patch_combat_graph(graph: Dict[str, Any], editor_data: Dict[str, Any]) -> Dict[str, Any]:
    if not graph or not isinstance(editor_data, dict):
        return graph
    summary = graph.get('summary') if isinstance(graph.get('summary'), dict) else rebuild_combat_summary(graph)
    for list_name in ('attackWaveList', 'attackWaveGroupList'):
        source_list = summary.get(list_name, {}) if isinstance(summary, dict) else {}
        edited_list = editor_data.get(list_name, {})
        if not isinstance(source_list, dict) or not isinstance(edited_list, dict):
            continue
        source_entries = list(source_list.get('entries', []) or [])
        edited_entries = list(edited_list.get('entries', []) or [])
        for index, source_entry in enumerate(source_entries):
            if index >= len(edited_entries) or not isinstance(edited_entries[index], dict):
                continue
            edited = edited_entries[index]
            id_ref = source_entry.get('id_ref')
            if isinstance(id_ref, dict):
                _write_at(graph, id_ref, struct.pack('<H', _clamp_int(edited.get('id', source_entry.get('id', 0)), 0, 65535)))
            name_pointer_field = source_entry.get('name_pointer_field')
            if isinstance(name_pointer_field, dict):
                _patch_c_string_pointer(
                    graph,
                    _as_int(name_pointer_field.get('section'), -1),
                    _as_int(name_pointer_field.get('offset'), 0),
                    edited.get('name', source_entry.get('name', '')),
                    limit=256,
                )
            _patch_definition(graph, source_entry.get('definition_ref'), dict(edited.get('definition', {}) or {}))
            _patch_runtime(graph, source_entry.get('runtime_ref'), dict(edited.get('runtime', {}) or {}))

    source_pmarkers = dict(summary.get('pmarkerList', {}) or {}) if isinstance(summary, dict) else {}
    edited_pmarkers = dict(editor_data.get('pmarkerList', {}) or {})
    source_entries = list(source_pmarkers.get('entries', []) or [])
    edited_entries = list(edited_pmarkers.get('entries', []) or [])
    for index, source_entry in enumerate(source_entries):
        if index >= len(edited_entries) or not isinstance(edited_entries[index], dict):
            continue
        _patch_pmarker_entry(graph, source_entry, edited_entries[index])
    rebuild_combat_summary(graph)
    return graph
