from __future__ import annotations

from math import isqrt, sqrt
from struct import unpack_from

from ..pc.texture import D3DFMT_A8B8G8R8, _create_dds_header
from ..ps3.texture import decode_supported_ps3_texture

DDS_FOURCC_DXT1 = 0x31545844
DDS_FOURCC_DXT5 = 0x35545844
X360_FMT_DXT1 = 0x05
X360_FMT_DXT5 = 0x09
X360_FMT_ARGB8_LINEAR = 0x0D
X360_FMT_BC5_NORMAL = 0x13


def _section_payload_offset(data: bytes) -> int:
    if len(data) >= 0x18 and data[:4] in {b'SECT', b'TCES'}:
        return 0x18
    return 0


def _swap_16bit_pairs(payload: bytes) -> bytes:
    out = bytearray(payload)
    end = len(out) - (len(out) & 1)
    for i in range(0, end, 2):
        out[i], out[i + 1] = out[i + 1], out[i]
    return bytes(out)


def _expected_dxt_size(width: int, height: int, block_bytes: int) -> int:
    return ((max(1, width) + 3) // 4) * ((max(1, height) + 3) // 4) * int(block_bytes)


def _infer_square_dimension_from_dxt_payload(data_size: int, block_bytes: int) -> int | None:
    if data_size <= 0 or block_bytes <= 0:
        return None
    block_count = int(data_size) // int(block_bytes)
    if block_count * int(block_bytes) != int(data_size):
        return None
    blocks_side = isqrt(block_count)
    if blocks_side * blocks_side != block_count:
        return None
    dimension = blocks_side * 4
    if dimension <= 0 or dimension > 8192:
        return None
    if dimension & (dimension - 1):
        return None
    return int(dimension)


def _decode_bc4_block(block: bytes) -> list[int]:
    if len(block) < 8:
        return [0] * 16
    a0 = int(block[0]) & 0xFF
    a1 = int(block[1]) & 0xFF
    values = [a0, a1]
    if a0 > a1:
        values.extend([
            (6 * a0 + 1 * a1) // 7,
            (5 * a0 + 2 * a1) // 7,
            (4 * a0 + 3 * a1) // 7,
            (3 * a0 + 4 * a1) // 7,
            (2 * a0 + 5 * a1) // 7,
            (1 * a0 + 6 * a1) // 7,
        ])
    else:
        values.extend([
            (4 * a0 + 1 * a1) // 5,
            (3 * a0 + 2 * a1) // 5,
            (2 * a0 + 3 * a1) // 5,
            (1 * a0 + 4 * a1) // 5,
            0,
            255,
        ])

    bits = int.from_bytes(block[2:8], 'little', signed=False)
    decoded: list[int] = []
    for index in range(16):
        decoded.append(int(values[(bits >> (3 * index)) & 0x7]))
    return decoded


def _decode_xbox360_bc5_normal_to_rgba(payload: bytes, width: int, height: int) -> bytes:
    # X360 BC5 decoding is reeeeaaally slow, gotta figure it out at some point.
    width = max(1, int(width))
    height = max(1, int(height))
    width_in_blocks = (width + 3) // 4
    height_in_blocks = (height + 3) // 4
    out = bytearray(width * height * 4)
    for block_y in range(height_in_blocks):
        for block_x in range(width_in_blocks):
            offset = ((block_y * width_in_blocks) + block_x) * 16
            block = payload[offset:offset + 16]
            if len(block) < 16:
                block = block + bytes(16 - len(block))
            red_values = _decode_bc4_block(block[:8])
            green_values = _decode_bc4_block(block[8:16])
            for py in range(4):
                for px in range(4):
                    x = block_x * 4 + px
                    y = block_y * 4 + py
                    if x >= width or y >= height:
                        continue
                    i = py * 4 + px
                    r = int(red_values[i]) & 0xFF
                    g = int(green_values[i]) & 0xFF
                    nx = (float(r) / 127.5) - 1.0
                    ny = (float(g) / 127.5) - 1.0
                    nz = sqrt(max(0.0, 1.0 - (nx * nx) - (ny * ny)))
                    b = max(0, min(255, int(round((nz * 0.5 + 0.5) * 255.0))))
                    out_offset = ((y * width) + x) * 4
                    out[out_offset:out_offset + 4] = bytes((r, g, b, 255))
    return bytes(out)


def _decode_xbox360_argb8_linear_to_rgba(payload: bytes, width: int, height: int) -> bytes:
    width = max(1, int(width))
    height = max(1, int(height))
    pixel_count = width * height
    source = bytes(payload[:pixel_count * 4])
    if len(source) < pixel_count * 4:
        source = source + bytes((pixel_count * 4) - len(source))
    out = bytearray(pixel_count * 4)
    for index in range(pixel_count):
        a = int(source[index * 4 + 0]) & 0xFF
        r = int(source[index * 4 + 1]) & 0xFF
        g = int(source[index * 4 + 2]) & 0xFF
        b = int(source[index * 4 + 3]) & 0xFF
        out[index * 4:index * 4 + 4] = bytes((r, g, b, a))
    return bytes(out)


def _xg_address_2d_tiled_x(block_offset: int, width_in_blocks: int, texel_byte_pitch: int) -> int:
    aligned_width = (int(width_in_blocks) + 31) & ~31
    if aligned_width <= 0:
        return 0
    log_bpp = (int(texel_byte_pitch) >> 2) + ((int(texel_byte_pitch) >> 1) >> (int(texel_byte_pitch) >> 2))
    offset_byte = int(block_offset) << log_bpp
    offset_tile = ((offset_byte & ~0xFFF) >> 3) + ((offset_byte & 0x700) >> 2) + (offset_byte & 0x3F)
    offset_macro = offset_tile >> (7 + log_bpp)
    macro_x = (offset_macro % max(1, aligned_width >> 5)) << 2
    tile = ((((offset_tile >> (5 + log_bpp)) & 2) + (offset_byte >> 6)) & 3)
    macro = (macro_x + tile) << 3
    micro = (((((offset_tile >> 1) & ~0xF) + (offset_tile & 0xF)) & ((int(texel_byte_pitch) << 3) - 1))) >> log_bpp
    return int(macro + micro)


def _xg_address_2d_tiled_y(block_offset: int, width_in_blocks: int, texel_byte_pitch: int) -> int:
    aligned_width = (int(width_in_blocks) + 31) & ~31
    if aligned_width <= 0:
        return 0
    log_bpp = (int(texel_byte_pitch) >> 2) + ((int(texel_byte_pitch) >> 1) >> (int(texel_byte_pitch) >> 2))
    offset_byte = int(block_offset) << log_bpp
    offset_tile = ((offset_byte & ~0xFFF) >> 3) + ((offset_byte & 0x700) >> 2) + (offset_byte & 0x3F)
    offset_macro = offset_tile >> (7 + log_bpp)
    macro_y = (offset_macro // max(1, aligned_width >> 5)) << 2
    tile = ((offset_tile >> (6 + log_bpp)) & 1) + ((offset_byte & 0x800) >> 10)
    macro = (macro_y + tile) << 3
    micro = ((((offset_tile & (((int(texel_byte_pitch) << 6) - 1) & ~0x1F)) + ((offset_tile & 0xF) << 1)) >> (3 + log_bpp)) & ~1)
    return int(macro + micro + ((offset_tile & 0x10) >> 4))


def _untile_xbox360_dxt_base_level(payload: bytes, width: int, height: int, block_bytes: int) -> bytes:
    width_in_blocks = max(1, (int(width) + 3) // 4)
    height_in_blocks = max(1, (int(height) + 3) // 4)
    block_bytes = int(block_bytes)
    top_size = width_in_blocks * height_in_blocks * block_bytes
    source = bytes(payload[:top_size])
    if len(source) < top_size:
        source = source + bytes(top_size - len(source))

    dest = bytearray(top_size)
    for y in range(height_in_blocks):
        for x in range(width_in_blocks):
            block_offset = y * width_in_blocks + x
            linear_offset = block_offset * block_bytes
            tiled_x = _xg_address_2d_tiled_x(block_offset, width_in_blocks, block_bytes)
            tiled_y = _xg_address_2d_tiled_y(block_offset, width_in_blocks, block_bytes)
            if tiled_x < 0 or tiled_y < 0 or tiled_x >= width_in_blocks or tiled_y >= height_in_blocks:
                continue
            tiled_offset = (tiled_y * width_in_blocks + tiled_x) * block_bytes
            if linear_offset + block_bytes <= len(source) and tiled_offset + block_bytes <= len(dest):
                dest[tiled_offset:tiled_offset + block_bytes] = source[linear_offset:linear_offset + block_bytes]
    return bytes(dest)


def _dds_format_for_x360_format(format_id: int) -> tuple[int | None, int | None, str, bool]:
    if int(format_id) == X360_FMT_DXT1:
        return DDS_FOURCC_DXT1, 8, 'BC1/DXT1', False
    if int(format_id) == X360_FMT_DXT5:
        return DDS_FOURCC_DXT5, 16, 'BC3/DXT5', True
    return None, None, f'X360-0x{int(format_id):X}', True


def _decode_markerless_x360_texture(data: bytes) -> dict | None:
    base = _section_payload_offset(data)
    if base + 0x10 > len(data):
        return None

    try:
        format_id = int(unpack_from('>I', data, base + 0x00)[0])
        width = int(unpack_from('>H', data, base + 0x04)[0])
        height = int(unpack_from('>H', data, base + 0x06)[0])
        data_size = int(unpack_from('>I', data, base + 0x08)[0])
        extra_word = int(unpack_from('>I', data, base + 0x0C)[0])
    except Exception:
        return None

    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        return None
    if data_size <= 0 or base + 0x10 + data_size > len(data):
        return None

    payload = bytes(data[base + 0x10:base + 0x10 + data_size])

    if int(format_id) == X360_FMT_ARGB8_LINEAR:
        top_size = int(width) * int(height) * 4
        if top_size <= 0 or len(payload) < top_size:
            return {
                'supported': False,
                'format_id': int(format_id),
                'format_name': 'X360-ARGB8-truncated',
                'width': int(width),
                'height': int(height),
            }
        rgba_payload = _decode_xbox360_argb8_linear_to_rgba(payload, int(width), int(height))
        dds_bytes = _create_dds_header(int(width), int(height), 1, D3DFMT_A8B8G8R8, len(rgba_payload)) + rgba_payload
        return {
            'supported': True,
            'format_id': int(format_id),
            'format_name': 'ARGB8/RGBA32',
            'width': int(width),
            'height': int(height),
            'mipmaps': 1,
            'has_alpha': True,
            'dds_bytes': dds_bytes,
            'bitmap_data': rgba_payload,
            'header_offset': int(base),
            'header_kind': 'markerless-x360',
            'x360_extra_word': int(extra_word),
            'x360_linearized_top_mip_only': True,
            'x360_source_format': 'argb8-linear',
        }

    if int(format_id) == X360_FMT_BC5_NORMAL:
        linear_bc5 = _untile_xbox360_dxt_base_level(_swap_16bit_pairs(payload), int(width), int(height), 16)
        rgba_payload = _decode_xbox360_bc5_normal_to_rgba(linear_bc5, int(width), int(height))
        dds_bytes = _create_dds_header(int(width), int(height), 1, D3DFMT_A8B8G8R8, len(rgba_payload)) + rgba_payload
        return {
            'supported': True,
            'format_id': int(format_id),
            'format_name': 'BC5/ATI2 normal decoded to RGB',
            'width': int(width),
            'height': int(height),
            'mipmaps': 1,
            'has_alpha': False,
            'dds_bytes': dds_bytes,
            'bitmap_data': rgba_payload,
            'header_offset': int(base),
            'header_kind': 'markerless-x360',
            'x360_extra_word': int(extra_word),
            'x360_endian_swap_16bit': True,
            'x360_tiled': True,
            'x360_linearized_top_mip_only': True,
            'x360_source_format': 'bc5-normal',
        }

    dds_format, block_bytes, format_name, has_alpha = _dds_format_for_x360_format(format_id)
    if dds_format is None or block_bytes is None:
        return {
            'supported': False,
            'format_id': int(format_id),
            'format_name': format_name,
            'width': int(width),
            'height': int(height),
        }

    payload = _swap_16bit_pairs(payload)
    payload = _untile_xbox360_dxt_base_level(payload, int(width), int(height), int(block_bytes))
    dds_bytes = _create_dds_header(int(width), int(height), 1, int(dds_format), len(payload)) + payload
    return {
        'supported': True,
        'format_id': int(format_id),
        'format_name': format_name,
        'width': int(width),
        'height': int(height),
        'mipmaps': 1,
        'has_alpha': bool(has_alpha),
        'dds_bytes': dds_bytes,
        'bitmap_data': payload,
        'header_offset': int(base),
        'header_kind': 'markerless-x360',
        'x360_extra_word': int(extra_word),
        'x360_endian_swap_16bit': True,
        'x360_tiled': True,
        'x360_linearized_top_mip_only': True,
    }


def decode_supported_xbox360_texture(pcd_bytes: bytes) -> dict | None:
    if not pcd_bytes:
        return None

    markerless = _decode_markerless_x360_texture(pcd_bytes)
    if markerless is not None:
        return markerless

    ps3_like = decode_supported_ps3_texture(pcd_bytes)
    if ps3_like is not None:
        if ps3_like.get('supported', False):
            ps3_like = dict(ps3_like)
            ps3_like['format_name'] = f"X360/{ps3_like.get('format_name', 'PS3T-like')}"
            ps3_like['header_kind'] = f"x360-{ps3_like.get('header_kind', 'ps3t-like')}"
        return ps3_like

    return None
