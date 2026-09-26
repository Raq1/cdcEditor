from __future__ import annotations

from struct import unpack_from

from ..pc.texture import _create_dds_header

DDS_FOURCC_DXT1 = 0x31545844
DDS_FOURCC_DXT5 = 0x35545844
PS3T_MAGIC = b'PS3T'


def _calc_mipmap_count(width: int, height: int, block_bytes: int, payload_size: int) -> int:
    if width <= 0 or height <= 0 or block_bytes <= 0 or payload_size <= 0:
        return 1
    mip_count = 0
    total = 0
    w = int(width)
    h = int(height)
    while True:
        total += max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * int(block_bytes)
        mip_count += 1
        if total >= payload_size:
            return max(1, mip_count)
        if w == 1 and h == 1:
            return max(1, mip_count)
        w = max(1, w // 2)
        h = max(1, h // 2)


def _standalone_section_payload_offset(pcd_bytes: bytes) -> int | None:
    if len(pcd_bytes) < 24 or pcd_bytes[:4] not in {b'SECT', b'TCES'}:
        return None

    try:
        packed_data = int(unpack_from('>I', pcd_bytes, 12)[0])
    except Exception:
        return 24

    candidates = []
    for count in (
        int(packed_data) & 0xFFFF,
        (int(packed_data) >> 8) & 0x00FFFFFF,
        (int(packed_data) >> 8) & 0xFFFF,
        0,
    ):
        offset = 24 + (int(count) * 8)
        if offset not in candidates and 24 <= offset <= len(pcd_bytes):
            candidates.append(offset)

    return candidates[0] if candidates else 24


def _iter_ps3_texture_header_candidates(pcd_bytes: bytes):
    if not pcd_bytes:
        return

    magic_offset = pcd_bytes.find(PS3T_MAGIC, 0, min(len(pcd_bytes), 0x200))
    if magic_offset >= 0:
        yield {
            'header_offset': int(magic_offset),
            'payload_size_offset': int(magic_offset) + 0x04,
            'format_offset': int(magic_offset) + 0x0C,
            'width_offset': int(magic_offset) + 0x14,
            'height_offset': int(magic_offset) + 0x16,
            'data_offset': int(magic_offset) + 0x24,
            'magic': 'PS3T',
        }

    section_payload_offset = _standalone_section_payload_offset(pcd_bytes)
    for base in (section_payload_offset, 0):
        if base is None:
            continue
        base = int(base)
        if base < 0 or base + 0x20 > len(pcd_bytes):
            continue
        # Markerless form: same header as PS3T but without the 4-byte magic.
        # PS3T +0x04 payloadSize becomes +0x00, +0x0C format becomes +0x08,
        # and texture data begins at +0x20.
        yield {
            'header_offset': base,
            'payload_size_offset': base + 0x00,
            'format_offset': base + 0x08,
            'width_offset': base + 0x10,
            'height_offset': base + 0x12,
            'data_offset': base + 0x20,
            'magic': 'markerless',
        }


def _decode_ps3_header_candidate(pcd_bytes: bytes, candidate: dict) -> dict | None:
    try:
        payload_size = int(unpack_from('>I', pcd_bytes, int(candidate['payload_size_offset']))[0])
        format_word = int(unpack_from('>I', pcd_bytes, int(candidate['format_offset']))[0])
        format_id = (format_word >> 24) & 0xFF
        width = int(unpack_from('>H', pcd_bytes, int(candidate['width_offset']))[0])
        height = int(unpack_from('>H', pcd_bytes, int(candidate['height_offset']))[0])
        data_offset = int(candidate['data_offset'])
    except Exception:
        return None

    if format_id not in {0x86, 0x88}:
        return None
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        return None
    if data_offset < 0 or data_offset > len(pcd_bytes):
        return None
    if payload_size <= 0 or data_offset + payload_size > len(pcd_bytes):
        payload_size = max(0, len(pcd_bytes) - data_offset)
    if payload_size <= 0:
        return None

    if format_id == 0x86:
        dds_format = DDS_FOURCC_DXT1
        block_bytes = 8
        format_name = 'BC1/DXT1'
        has_alpha = False
    elif format_id == 0x88:
        dds_format = DDS_FOURCC_DXT5
        block_bytes = 16
        format_name = 'BC3/DXT5'
        has_alpha = True
    else:
        return None

    payload = bytes(pcd_bytes[data_offset:data_offset + payload_size])
    mipmaps = _calc_mipmap_count(width, height, block_bytes, len(payload))
    dds_bytes = _create_dds_header(width, height, mipmaps, dds_format, len(payload)) + payload
    return {
        'supported': True,
        'format_name': format_name,
        'format_id': format_id,
        'width': width,
        'height': height,
        'mipmaps': mipmaps,
        'has_alpha': has_alpha,
        'dds_bytes': dds_bytes,
        'bitmap_data': payload,
        'header_offset': int(candidate.get('header_offset', 0)),
        'header_kind': str(candidate.get('magic', 'unknown')),
    }


def decode_supported_ps3_texture(pcd_bytes: bytes) -> dict | None:
    """Parse observed PS3 PCD texture containers.

    The common form is an extracted SECT wrapper containing a PS3T payload, but
    some files/tools omit the literal PS3T marker while preserving the same
    big-endian texture header at the section payload.  The decoder validates
    candidate headers instead of requiring the marker unconditionally.
    """
    if not pcd_bytes:
        return None

    first_unsupported: dict | None = None
    for candidate in _iter_ps3_texture_header_candidates(pcd_bytes):
        decoded = _decode_ps3_header_candidate(pcd_bytes, candidate)
        if decoded is not None:
            return decoded
        if first_unsupported is None:
            try:
                fmt = (int(unpack_from('>I', pcd_bytes, int(candidate['format_offset']))[0]) >> 24) & 0xFF
                width = int(unpack_from('>H', pcd_bytes, int(candidate['width_offset']))[0])
                height = int(unpack_from('>H', pcd_bytes, int(candidate['height_offset']))[0])
            except Exception:
                fmt, width, height = 0, 0, 0
            first_unsupported = {
                'supported': False,
                'format_name': f'PS3T-0x{fmt:02X}',
                'format_id': fmt,
                'width': width,
                'height': height,
            }

    return first_unsupported
