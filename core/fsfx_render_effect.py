from __future__ import annotations

import base64
import struct
from collections import OrderedDict
from typing import Any, Iterable

FSFX_PROP_PREFIX = 'trlau_fsfx_'
FSFX_FORMAT_NAME = 'cdc::RenderFSEffect'
RENDER_FS_EFFECT_SIZE = 0x1CC8
COLOUR_DOF_SIZE = 0x1320
FINISH_SIZE = 0x910

COLOUR_DOF_TABLES: tuple[tuple[str, int], ...] = (
    ('m_anBBRed', 3),
    ('m_anBBGreen', 3),
    ('m_anBBBlue', 3),
    ('m_anSBRed', 3),
    ('m_anSBGreen', 3),
    ('m_anSBBlue', 3),
    ('m_anDOFCurve', 1),
)
FINISH_TABLES: tuple[tuple[str, int], ...] = (
    ('m_anSBRed', 3),
    ('m_anSBGreen', 3),
    ('m_anSBBlue', 3),
)

# Curve/Palette table UI is absolutely horrible and needs to be improved
# or maybe not because I don't even know how else Blender can handle it
SCALAR_FIELDS: tuple[tuple[str, str, str, int, str], ...] = (
    ('sColourDOF', 'm_vBlur', 'colour_dof_blur', 0x1300, 'f32'),
    ('sColourDOF', 'm_vAlpha', 'colour_dof_alpha', 0x1304, 'f32'),
    ('sColourDOF', 'm_nColour', 'colour_dof_colour', 0x1308, 'u32'),
    ('sColourDOF', 'm_isUseZBuffer', 'colour_dof_use_zbuffer', 0x130C, 'bool'),
    ('sColourDOF', 'm_isColourConvertBackBuffer', 'colour_dof_colour_convert_backbuffer', 0x130D, 'bool'),
    ('sColourDOF', 'm_isEnabled', 'colour_dof_enabled', 0x130E, 'bool'),
    ('sColourDOF', 'm_isUseObjectPointer', 'colour_dof_use_object_pointer', 0x130F, 'bool'),
    ('sColourDOF', 'm_objectPoint', 'colour_dof_object_point', 0x1310, 'vec3'),
    ('sColourDOF', 'm_objectRadius', 'colour_dof_object_radius', 0x131C, 'f32'),

    ('sFinish', 'm_vBlurStrength', 'finish_blur_strength', 0x1320 + 0x900, 'f32'),
    ('sFinish', 'm_vBright', 'finish_bright', 0x1320 + 0x904, 'f32'),
    ('sFinish', 'm_isColourConvert', 'finish_colour_convert', 0x1320 + 0x908, 'bool'),
    ('sFinish', 'm_isEnabled', 'finish_enabled', 0x1320 + 0x909, 'bool'),
    ('sFinish', 'm_pad1', 'finish_pad1', 0x1320 + 0x90A, 'u16'),
    ('sFinish', 'm_pad2', 'finish_pad2', 0x1320 + 0x90C, 'u32'),

    ('sNoise', 'm_nColour', 'noise_colour', 0x1C30, 'u32'),
    ('sNoise', 'm_nMinAlpha', 'noise_min_alpha', 0x1C34, 'u32'),
    ('sNoise', 'm_nMaxAlpha', 'noise_max_alpha', 0x1C38, 'u32'),
    ('sNoise', 'm_nTexStep', 'noise_tex_step', 0x1C3C, 'u32'),
    ('sNoise', 'm_isEnabled', 'noise_enabled', 0x1C40, 'bool'),

    ('sBloom', 'm_vBlurStrength', 'bloom_blur_strength', 0x1C44, 'f32'),
    ('sBloom', 'm_vBright', 'bloom_bright', 0x1C48, 'f32'),
    ('sBloom', 'm_isEnabled', 'bloom_enabled', 0x1C4C, 'bool'),

    ('sFeedback', 'm_vFrontAlpha', 'feedback_front_alpha', 0x1C50, 'f32'),
    ('sFeedback', 'm_vBackAlpha', 'feedback_back_alpha', 0x1C54, 'f32'),
    ('sFeedback', 'm_vBlurStrength', 'feedback_blur_strength', 0x1C58, 'f32'),
    ('sFeedback', 'm_vScaleX', 'feedback_scale_x', 0x1C5C, 'f32'),
    ('sFeedback', 'm_vScaleY', 'feedback_scale_y', 0x1C60, 'f32'),
    ('sFeedback', 'm_vShearX', 'feedback_shear_x', 0x1C64, 'f32'),
    ('sFeedback', 'm_vShearY', 'feedback_shear_y', 0x1C68, 'f32'),
    ('sFeedback', 'm_nColour', 'feedback_colour', 0x1C6C, 'u32'),
    ('sFeedback', 'm_isEnabled', 'feedback_enabled', 0x1C70, 'bool'),

    ('sWater', 'm_vBlur', 'water_blur', 0x1C74, 'f32'),
    ('sWater', 'm_vAmplitude', 'water_amplitude', 0x1C78, 'f32'),
    ('sWater', 'm_vSpeed', 'water_speed', 0x1C7C, 'f32'),
    ('sWater', 'm_nColour', 'water_colour', 0x1C80, 'u32'),
    ('sWater', 'm_isEnabled', 'water_enabled', 0x1C84, 'bool'),

    ('sHDR', 'm_vTargetBrightness', 'hdr_target_brightness', 0x1C88, 'f32'),
    ('sHDR', 'm_vNormalBrightness', 'hdr_normal_brightness', 0x1C8C, 'f32'),
    ('sHDR', 'm_vBrightnessAdjustRateUp', 'hdr_brightness_adjust_rate_up', 0x1C90, 'f32'),
    ('sHDR', 'm_vBrightnessAdjustRateDown', 'hdr_brightness_adjust_rate_down', 0x1C94, 'f32'),
    ('sHDR', 'm_vTargetBrightnessWait', 'hdr_target_brightness_wait', 0x1C98, 'f32'),
    ('sHDR', 'm_vMaxBlur', 'hdr_max_blur', 0x1C9C, 'f32'),
    ('sHDR', 'm_bReduceToNormalBrightness', 'hdr_reduce_to_normal_brightness', 0x1CA0, 'bool'),
    ('sHDR', 'm_isEnabled', 'hdr_enabled', 0x1CA1, 'bool'),

    ('sHeatHaze', 'm_vHorizontalCompression', 'heat_haze_horizontal_compression', 0x1CA4, 'f32'),
    ('sHeatHaze', 'm_vVerticalCompression', 'heat_haze_vertical_compression', 0x1CA8, 'f32'),
    ('sHeatHaze', 'm_vTurbulence', 'heat_haze_turbulence', 0x1CAC, 'f32'),
    ('sHeatHaze', 'm_vSpeed', 'heat_haze_speed', 0x1CB0, 'f32'),
    ('sHeatHaze', 'm_vOpacity', 'heat_haze_opacity', 0x1CB4, 'f32'),
    ('sHeatHaze', 'm_nColour', 'heat_haze_colour', 0x1CB8, 'u32'),
    ('sHeatHaze', 'm_isFullscreen', 'heat_haze_fullscreen', 0x1CBC, 'bool'),
    ('sHeatHaze', 'm_isEnabled', 'heat_haze_enabled', 0x1CBD, 'bool'),

    ('sMultiFocusBlur', 'm_vBlurStrength', 'multi_focus_blur_strength', 0x1CC0, 'f32'),
    ('sMultiFocusBlur', 'm_isEnabled', 'multi_focus_blur_enabled', 0x1CC4, 'bool'),
)

_SCALAR_BY_SUFFIX = {suffix: (struct_name, field_label, offset, kind) for struct_name, field_label, suffix, offset, kind in SCALAR_FIELDS}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith('0x'):
                return int(text, 16)
        return int(value)
    except Exception:
        return int(default or 0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default or 0.0)


def _clamp_u8(value: Any) -> int:
    return max(0, min(255, _safe_int(value, 0)))


def _clamp_u16(value: Any) -> int:
    return max(0, min(0xFFFF, _safe_int(value, 0)))


def _clamp_u32(value: Any) -> int:
    return max(0, min(0xFFFFFFFF, _safe_int(value, 0)))


def _read_scalar(content: bytes, offset: int, kind: str) -> Any:
    if kind == 'f32':
        return struct.unpack_from('<f', content, offset)[0]
    if kind == 'u16':
        return struct.unpack_from('<H', content, offset)[0]
    if kind == 'u32':
        return struct.unpack_from('<I', content, offset)[0]
    if kind == 'bool':
        return bool(content[offset])
    if kind == 'vec3':
        return tuple(float(v) for v in struct.unpack_from('<fff', content, offset))
    raise ValueError(f'Unsupported FSFX scalar kind: {kind}')


def _write_scalar(content: bytearray, offset: int, kind: str, value: Any) -> None:
    if kind == 'f32':
        struct.pack_into('<f', content, offset, _safe_float(value, 0.0))
    elif kind == 'u16':
        struct.pack_into('<H', content, offset, _clamp_u16(value))
    elif kind == 'u32':
        struct.pack_into('<I', content, offset, _clamp_u32(value))
    elif kind == 'bool':
        content[offset] = 1 if bool(value) else 0
    elif kind == 'vec3':
        values = list(value) if isinstance(value, (list, tuple)) else []
        while len(values) < 3:
            values.append(0.0)
        struct.pack_into('<fff', content, offset, _safe_float(values[0]), _safe_float(values[1]), _safe_float(values[2]))
    else:
        raise ValueError(f'Unsupported FSFX scalar kind: {kind}')


def _section_content_from_blob(blob: bytes) -> bytes:
    raw = bytes(blob or b'')
    if len(raw) >= 24 and raw[:4] in {b'SECT', b'DRM\x00'}:
        try:
            section_size, _section_type, _skip_flags, _version_id, packed_data, _section_id, _spec_mask = struct.unpack_from('<iBBHIII', raw, 4)
        except struct.error:
            return b''
        reloc_count = max(0, int((int(packed_data) >> 8) & 0x00FFFFFF))
        info_size = 24 + reloc_count * 8
        if len(raw) < info_size:
            return b''
        payload_end = info_size + max(0, int(section_size))
        if payload_end <= len(raw):
            end = payload_end
        elif 0 < int(section_size) <= len(raw):
            end = int(section_size)
        else:
            end = len(raw)
        return bytes(raw[info_size:end])
    return raw


def _table_prop_key(struct_name: str, table_name: str, channel_index: int | None = None) -> str:
    stem = f'{FSFX_PROP_PREFIX}table_{struct_name}_{table_name}'
    return stem if channel_index is None else f'{stem}_{int(channel_index)}'


def _encode_table(values: Iterable[int]) -> str:
    return base64.b64encode(bytes(_clamp_u8(v) for v in list(values)[:256]).ljust(256, b'\x00')).decode('ascii')


def _decode_table(obj: Any, key: str, default: bytes | None = None) -> bytes:
    raw_default = bytes(default or b'\x00' * 256)[:256].ljust(256, b'\x00')
    try:
        encoded = obj.get(key, '')
    except Exception:
        encoded = ''
    if not encoded:
        return raw_default
    try:
        data = base64.b64decode(str(encoded).encode('ascii'), validate=False)
    except Exception:
        return raw_default
    return bytes(data[:256]).ljust(256, b'\x00')


def _table_summary(values: bytes) -> str:
    values = bytes(values[:256]).ljust(256, b'\x00')
    if all(v == i for i, v in enumerate(values)):
        kind = 'identity'
    elif all(v == 0 for v in values):
        kind = 'zero'
    else:
        kind = 'custom'
    return f'{kind}; min={min(values)} max={max(values)} unique={len(set(values))}'


def _store_table(obj: Any, struct_name: str, table_name: str, channel_index: int | None, values: bytes) -> None:
    obj[_table_prop_key(struct_name, table_name, channel_index)] = base64.b64encode(bytes(values[:256]).ljust(256, b'\x00')).decode('ascii')


def _iter_table_specs() -> Iterable[tuple[str, str, int | None, int]]:
    offset = 0
    for table_name, count in COLOUR_DOF_TABLES:
        if count == 1:
            yield ('sColourDOF', table_name, None, offset)
            offset += 256
        else:
            for channel_index in range(count):
                yield ('sColourDOF', table_name, channel_index, offset)
                offset += 256
    offset = 0x1320
    for table_name, count in FINISH_TABLES:
        for channel_index in range(count):
            yield ('sFinish', table_name, channel_index, offset)
            offset += 256


def _table_identifier(struct_name: str, table_name: str, channel_index: int | None) -> str:
    suffix = 'main' if channel_index is None else str(int(channel_index))
    return f'{struct_name}|{table_name}|{suffix}'


def _table_label(struct_name: str, table_name: str, channel_index: int | None) -> str:
    if channel_index is None:
        return f'{struct_name}.{table_name}'
    return f'{struct_name}.{table_name}[{int(channel_index)}]'


def _parse_table_identifier(identifier: str) -> tuple[str, str, int | None] | None:
    parts = str(identifier or '').split('|')
    if len(parts) != 3:
        return None
    struct_name, table_name, channel_text = parts
    channel_index = None
    if channel_text != 'main':
        try:
            channel_index = int(channel_text)
        except Exception:
            return None
    known = {
        _table_identifier(spec_struct, spec_table, spec_channel)
        for spec_struct, spec_table, spec_channel, _offset in _iter_table_specs()
    }
    parsed_id = _table_identifier(struct_name, table_name, channel_index)
    if parsed_id not in known:
        return None
    return struct_name, table_name, channel_index


def fsfx_table_ui_items() -> list[tuple[str, str, str]]:
    return [
        (_table_identifier(struct_name, table_name, channel_index), _table_label(struct_name, table_name, channel_index), '')
        for struct_name, table_name, channel_index, _offset in _iter_table_specs()
    ]


def fsfx_first_table_identifier(obj: Any | None = None) -> str:
    for identifier, _label, _description in fsfx_table_ui_items():
        if obj is None:
            return identifier
        parsed = _parse_table_identifier(identifier)
        if parsed is None:
            continue
        struct_name, table_name, channel_index = parsed
        if _table_prop_key(struct_name, table_name, channel_index) in obj:
            return identifier
    items = fsfx_table_ui_items()
    return items[0][0] if items else ''


def fsfx_normalize_table_identifier(obj: Any, identifier: str) -> str:
    if _parse_table_identifier(identifier) is not None:
        return str(identifier)
    return fsfx_first_table_identifier(obj)


def fsfx_table_label(identifier: str) -> str:
    parsed = _parse_table_identifier(identifier)
    if parsed is None:
        return ''
    return _table_label(*parsed)


def fsfx_get_table_bytes(obj: Any, identifier: str) -> bytes:
    parsed = _parse_table_identifier(fsfx_normalize_table_identifier(obj, identifier))
    if parsed is None:
        return bytes(b'\x00' * 256)
    key = _table_prop_key(*parsed)
    return _decode_table(obj, key)


def fsfx_set_table_bytes(obj: Any, identifier: str, values: Iterable[int]) -> None:
    parsed = _parse_table_identifier(fsfx_normalize_table_identifier(obj, identifier))
    if parsed is None:
        return
    key = _table_prop_key(*parsed)
    encoded = _encode_table(values)
    obj[key] = encoded
    _refresh_table_summary(obj)


def fsfx_get_table_value(obj: Any, identifier: str, index: int) -> int:
    values = fsfx_get_table_bytes(obj, identifier)
    try:
        idx = max(0, min(255, int(index)))
    except Exception:
        idx = 0
    return int(values[idx])


def fsfx_set_table_value(obj: Any, identifier: str, index: int, value: Any) -> None:
    table_id = fsfx_normalize_table_identifier(obj, identifier)
    values = bytearray(fsfx_get_table_bytes(obj, table_id))
    try:
        idx = max(0, min(255, int(index)))
    except Exception:
        idx = 0
    values[idx] = _clamp_u8(value)
    fsfx_set_table_bytes(obj, table_id, values)


def fsfx_table_summary_for_identifier(obj: Any, identifier: str) -> str:
    table_id = fsfx_normalize_table_identifier(obj, identifier)
    return _table_summary(fsfx_get_table_bytes(obj, table_id))


def _refresh_table_summary(obj: Any) -> None:
    summaries: list[str] = []
    for struct_name, table_name, channel_index, _offset in _iter_table_specs():
        key = _table_prop_key(struct_name, table_name, channel_index)
        if key not in obj:
            continue
        label = _table_label(struct_name, table_name, channel_index)
        summaries.append(f'{label}: {_table_summary(_decode_table(obj, key))}')
    obj[f'{FSFX_PROP_PREFIX}table_summary'] = '\n'.join(summaries)


def fsfx_u32_colour_to_rgb(value: Any) -> tuple[float, float, float]:
    colour = _clamp_u32(value)
    return (
        float((colour >> 16) & 0xFF) / 255.0,
        float((colour >> 8) & 0xFF) / 255.0,
        float(colour & 0xFF) / 255.0,
    )


def fsfx_rgb_to_u32(rgb: Iterable[float], preserve_value: Any = 0) -> int:
    values = list(rgb) if isinstance(rgb, (list, tuple)) else []
    while len(values) < 3:
        values.append(0.0)
    r = max(0, min(255, int(round(float(values[0]) * 255.0))))
    g = max(0, min(255, int(round(float(values[1]) * 255.0))))
    b = max(0, min(255, int(round(float(values[2]) * 255.0))))
    high = _clamp_u32(preserve_value) & 0xFF000000
    return int(high | (r << 16) | (g << 8) | b)


def has_fsfx_render_effect_properties(obj: Any) -> bool:
    try:
        return str(obj.get(f'{FSFX_PROP_PREFIX}format', '') or '') == FSFX_FORMAT_NAME
    except Exception:
        return False


def parse_fsfx_render_effect_content(content: bytes) -> dict[str, Any]:
    data = bytes(content or b'')
    if len(data) < RENDER_FS_EFFECT_SIZE:
        raise ValueError(f'FSFX content is too small for {FSFX_FORMAT_NAME}: 0x{len(data):X} < 0x{RENDER_FS_EFFECT_SIZE:X}')
    parsed: dict[str, Any] = OrderedDict()
    for struct_name, field_label, suffix, offset, kind in SCALAR_FIELDS:
        parsed.setdefault(struct_name, OrderedDict())
        parsed[struct_name][field_label] = _read_scalar(data, offset, kind)
    tables: dict[str, list[int]] = OrderedDict()
    for struct_name, table_name, channel_index, offset in _iter_table_specs():
        label = f'{struct_name}.{table_name}' if channel_index is None else f'{struct_name}.{table_name}[{channel_index}]'
        tables[label] = list(data[offset:offset + 256])
    return {
        'format': FSFX_FORMAT_NAME,
        'render_fs_effect_size': RENDER_FS_EFFECT_SIZE,
        'content_size': len(data),
        'trailing_padding_size': max(0, len(data) - RENDER_FS_EFFECT_SIZE),
        'scalars': parsed,
        'tables': tables,
    }


def store_fsfx_render_effect_properties(obj: Any, section_blob_or_content: bytes) -> bool:
    content = _section_content_from_blob(section_blob_or_content)
    if len(content) < RENDER_FS_EFFECT_SIZE:
        return False
    obj[f'{FSFX_PROP_PREFIX}format'] = FSFX_FORMAT_NAME

    for _struct_name, _field_label, suffix, offset, kind in SCALAR_FIELDS:
        obj[f'{FSFX_PROP_PREFIX}{suffix}'] = _read_scalar(content, offset, kind)

    for struct_name, table_name, channel_index, offset in _iter_table_specs():
        values = content[offset:offset + 256]
        _store_table(obj, struct_name, table_name, channel_index, values)
    _refresh_table_summary(obj)
    return True


def build_fsfx_render_effect_content_from_object(obj: Any, base_content: bytes) -> bytes:
    if not has_fsfx_render_effect_properties(obj):
        return bytes(base_content or b'')
    content = bytearray(bytes(base_content or b''))
    if len(content) < RENDER_FS_EFFECT_SIZE:
        content.extend(b'\x00' * (RENDER_FS_EFFECT_SIZE - len(content)))
    for _struct_name, _field_label, suffix, offset, kind in SCALAR_FIELDS:
        key = f'{FSFX_PROP_PREFIX}{suffix}'
        if key not in obj:
            continue
        _write_scalar(content, offset, kind, obj.get(key))

    for struct_name, table_name, channel_index, offset in _iter_table_specs():
        key = _table_prop_key(struct_name, table_name, channel_index)
        if key not in obj:
            continue
        current = bytes(content[offset:offset + 256])
        content[offset:offset + 256] = _decode_table(obj, key, current)
    return bytes(content)


def fsfx_table_summary(obj: Any) -> str:
    try:
        return str(obj.get(f'{FSFX_PROP_PREFIX}table_summary', '') or '')
    except Exception:
        return ''



def decode_stored_section_blob_from_object(obj: Any, *, prefix: str = 'trlau_section_blob_') -> bytes:
    try:
        chunk_count = int(obj.get('trlau_section_blob_chunk_count', 0) or 0)
    except Exception:
        chunk_count = 0
    if chunk_count <= 0:
        return b''
    encoded_parts: list[str] = []
    for index in range(chunk_count):
        try:
            encoded_parts.append(str(obj.get(f'{prefix}{index:04d}', '') or ''))
        except Exception:
            encoded_parts.append('')
    encoded = ''.join(encoded_parts)
    if not encoded:
        return b''
    try:
        data = base64.b64decode(encoded.encode('ascii'), validate=False)
    except Exception:
        return b''
    try:
        expected_size = int(obj.get('trlau_section_blob_size', len(data)) or len(data))
    except Exception:
        expected_size = len(data)
    if expected_size >= 0:
        data = data[:expected_size]
    return bytes(data)
