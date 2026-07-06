from __future__ import annotations

from struct import unpack
from typing import Optional, Tuple

PSP_PCD_HEADER_SIZE = 0x20
PSP_PCD_PALETTE_RGBA8888_SIZE = 0x400
PSP_PCD_KIND_T8_RGBA8888_CLUT = 0x02600000
PSP_PCD_KIND_T8_RGBA8888_CLUT_ALT = 0x02000000
PSP_PCD_KIND_T4_RGBA8888_CLUT = 0x00680000
PSP_PCD_KIND_T4_RGBA8888_CLUT_ALT = 0x00080000
PSP_PCD_KIND_RGB565 = 0x10000000
PSP_PCD_KIND_RGB565_BLOCK = 0x10600000
PSP_PCD_KIND_RGBA4444 = 0x20000000
PSP_PCD_KIND_RGBA4444_BLOCK = 0x20600000
PSP_PCD_KIND_RGBA8888 = 0x00700000
PSP_PCD_KIND_RGBA8888_ALT = 0x00100000
PSP_PCD_T8_KINDS = {PSP_PCD_KIND_T8_RGBA8888_CLUT, PSP_PCD_KIND_T8_RGBA8888_CLUT_ALT}
PSP_PCD_RGB565_KINDS = {PSP_PCD_KIND_RGB565, PSP_PCD_KIND_RGB565_BLOCK}
PSP_PCD_RGBA4444_KINDS = {PSP_PCD_KIND_RGBA4444, PSP_PCD_KIND_RGBA4444_BLOCK}



def _pcd_section_payload_offset(pcd_bytes: bytes) -> int:
    if len(pcd_bytes) >= 24 and pcd_bytes[:4] in {b'SECT', b'TCES'}:
        return 24
    return 0


def _read_psp_pcd_header(pcd_bytes: bytes) -> Optional[dict]:
    payload_offset = _pcd_section_payload_offset(pcd_bytes)
    if len(pcd_bytes) < payload_offset + PSP_PCD_HEADER_SIZE:
        return None

    header = pcd_bytes[payload_offset:payload_offset + PSP_PCD_HEADER_SIZE]
    kind = unpack('<I', header[0:4])[0]
    if kind not in (PSP_PCD_T8_KINDS | PSP_PCD_RGB565_KINDS | PSP_PCD_RGBA4444_KINDS | {PSP_PCD_KIND_T4_RGBA8888_CLUT, PSP_PCD_KIND_T4_RGBA8888_CLUT_ALT, PSP_PCD_KIND_RGBA8888, PSP_PCD_KIND_RGBA8888_ALT}):
        return None

    width, height = unpack('<HH', header[4:8])
    log_width, log_height = unpack('<HH', header[8:12])
    bitmap_size = unpack('<I', header[0x10:0x14])[0]

    if width <= 0 or height <= 0 or width > 2048 or height > 2048:
        return None

    if bitmap_size <= 0:
        if kind in PSP_PCD_T8_KINDS:
            bitmap_size = _psp_top_mip_index_size(width, height, 8)
        elif kind in {PSP_PCD_KIND_T4_RGBA8888_CLUT, PSP_PCD_KIND_T4_RGBA8888_CLUT_ALT}:
            bitmap_size = _psp_top_mip_index_size(width, height, 4)
        elif kind in PSP_PCD_RGB565_KINDS:
            bitmap_size = int(width) * int(height) * 2
        elif kind in PSP_PCD_RGBA4444_KINDS:
            bitmap_size = int(width) * int(height) * 2
        elif kind in {PSP_PCD_KIND_RGBA8888, PSP_PCD_KIND_RGBA8888_ALT}:
            bitmap_size = int(width) * int(height) * 4
        else:
            return None

    return {
        'payload_offset': payload_offset,
        'kind': int(kind),
        'width': int(width),
        'height': int(height),
        'log_width': int(log_width),
        'log_height': int(log_height),
        'bitmap_size': int(bitmap_size),
    }


def _psp_pcd_dimensions(pcd_bytes: bytes) -> Tuple[int, int] | None:
    header = _read_psp_pcd_header(pcd_bytes)
    if header is None:
        return None
    return int(header['width']), int(header['height'])


def _unswizzle_psp_texture(data: bytes, width: int, height: int, bytes_per_pixel: int) -> bytes:
    if width <= 0 or height <= 0 or bytes_per_pixel <= 0:
        return bytes(data)

    block_width_pixels = max(1, 16 // int(bytes_per_pixel))
    block_height_pixels = 8
    required = int(width) * int(height) * int(bytes_per_pixel)
    source = bytes(data[:required])
    if len(source) < required:
        raise ValueError(f'Truncated PSP texture data: expected {required} bytes, got {len(source)}')

    output = bytearray(required)
    source_offset = 0
    for block_y in range(0, int(height), block_height_pixels):
        for block_x in range(0, int(width), block_width_pixels):
            for y in range(block_height_pixels):
                for x in range(block_width_pixels):
                    pixel_x = block_x + x
                    pixel_y = block_y + y
                    if pixel_x < width and pixel_y < height:
                        dest_offset = ((pixel_y * width) + pixel_x) * bytes_per_pixel
                        output[dest_offset:dest_offset + bytes_per_pixel] = source[source_offset:source_offset + bytes_per_pixel]
                    source_offset += bytes_per_pixel
    return bytes(output)


def _psp_top_mip_index_size(width: int, height: int, bits_per_pixel: int) -> int:
    return max(1, (int(width) * int(height) * int(bits_per_pixel) + 7) // 8)


def _flip_rgba_float_y(pixels: list[float], width: int, height: int) -> list[float]:
    row_size = max(0, int(width)) * 4
    height = max(0, int(height))
    if row_size <= 0 or height <= 1:
        return list(pixels)

    flipped = [0.0] * len(pixels)
    for y in range(height):
        src = y * row_size
        dst = (height - 1 - y) * row_size
        flipped[dst:dst + row_size] = pixels[src:src + row_size]
    return flipped


def _psp_pcd_image_header_size(header: dict, pcd_bytes: bytes, palette_size: int, top_mip_size: int) -> int:
    payload_offset = int(header['payload_offset'])
    width = int(header['width'])
    height = int(header['height'])
    file_len = len(pcd_bytes)

    extended_offset = payload_offset + 0x30 + int(palette_size)
    if file_len >= extended_offset + int(top_mip_size):
        extension = pcd_bytes[payload_offset + 0x20:payload_offset + 0x30]
        if len(extension) == 0x10 and (extension[:4] == b'\x40\x00\x00\x00' or extension[4:8] == b'\x00\x00\x00\x08'):
            return 0x30

    compact_offset = payload_offset + 0x20 + int(palette_size)
    if file_len >= compact_offset + int(top_mip_size):
        return 0x20

    # ?
    return 0x20


def _psp_palette_alpha_denominator(palette: bytes) -> float | None:
    if not palette:
        return 255.0
    max_alpha = max(int(palette[i + 3]) for i in range(0, len(palette) - 3, 4))
    if max_alpha == 0:
        return None
    return 128.0 if max_alpha <= 128 else 255.0


def _palette_rgba_to_float(palette: bytes, palette_offset: int, alpha_denominator: float | None) -> tuple[float, float, float, float]:
    r = palette[palette_offset]
    g = palette[palette_offset + 1]
    b = palette[palette_offset + 2]
    a = palette[palette_offset + 3]
    alpha = 1.0 if alpha_denominator is None else max(0.0, min(1.0, float(a) / float(alpha_denominator)))
    return (
        r / 255.0,
        g / 255.0,
        b / 255.0,
        alpha,
    )


def _decode_psp_pcd_rgba8888_clut(header: dict, pcd_bytes: bytes) -> list[float]:
    payload_offset = int(header['payload_offset'])
    width = int(header['width'])
    height = int(header['height'])
    required_bitmap_size = _psp_top_mip_index_size(width, height, 8)
    header_size = _psp_pcd_image_header_size(header, pcd_bytes, PSP_PCD_PALETTE_RGBA8888_SIZE, required_bitmap_size)
    palette_offset = payload_offset + header_size
    index_offset = palette_offset + PSP_PCD_PALETTE_RGBA8888_SIZE

    if len(pcd_bytes) < palette_offset + PSP_PCD_PALETTE_RGBA8888_SIZE:
        raise ValueError('Truncated PSP PCD palette data')
    if len(pcd_bytes) < index_offset + required_bitmap_size:
        raise ValueError('Truncated PSP paletted PCD texture data')

    palette = pcd_bytes[palette_offset:palette_offset + PSP_PCD_PALETTE_RGBA8888_SIZE]
    indices = _unswizzle_psp_texture(
        pcd_bytes[index_offset:index_offset + required_bitmap_size],
        width,
        height,
        1,
    )
    alpha_denominator = _psp_palette_alpha_denominator(palette)

    rgba_pixels: list[float] = [0.0] * (width * height * 4)
    out_index = 0
    for palette_index in indices[:width * height]:
        entry_offset = int(palette_index) * 4
        r, g, b, a = _palette_rgba_to_float(palette, entry_offset, alpha_denominator)
        rgba_pixels[out_index] = r
        rgba_pixels[out_index + 1] = g
        rgba_pixels[out_index + 2] = b
        rgba_pixels[out_index + 3] = a
        out_index += 4
    return rgba_pixels


def _decode_psp_pcd_rgba8888_clut4(header: dict, pcd_bytes: bytes) -> list[float]:
    payload_offset = int(header['payload_offset'])
    width = int(header['width'])
    height = int(header['height'])
    palette_size = 0x40
    row_bytes = (width + 1) // 2
    required_bitmap_size = row_bytes * height
    header_size = _psp_pcd_image_header_size(header, pcd_bytes, palette_size, required_bitmap_size)
    palette_offset = payload_offset + header_size
    index_offset = palette_offset + palette_size

    if len(pcd_bytes) < palette_offset + palette_size:
        raise ValueError('Truncated PSP 4-bit PCD palette data')
    if len(pcd_bytes) < index_offset + required_bitmap_size:
        raise ValueError('Truncated PSP 4-bit paletted PCD texture data')

    palette = pcd_bytes[palette_offset:palette_offset + palette_size]
    index_bytes = _unswizzle_psp_texture(
        pcd_bytes[index_offset:index_offset + required_bitmap_size],
        row_bytes,
        height,
        1,
    )
    alpha_denominator = _psp_palette_alpha_denominator(palette)

    rgba_pixels: list[float] = [0.0] * (width * height * 4)
    out_index = 0
    for y in range(height):
        row_offset = y * row_bytes
        for x in range(width):
            packed = int(index_bytes[row_offset + (x // 2)])
            palette_index = (packed & 0x0F) if (x & 1) == 0 else ((packed >> 4) & 0x0F)
            entry_offset = palette_index * 4
            r, g, b, a = _palette_rgba_to_float(palette, entry_offset, alpha_denominator)
            rgba_pixels[out_index] = r
            rgba_pixels[out_index + 1] = g
            rgba_pixels[out_index + 2] = b
            rgba_pixels[out_index + 3] = a
            out_index += 4
    return rgba_pixels


def _decode_psp_pcd_rgba8888(header: dict, pcd_bytes: bytes) -> list[float]:
    payload_offset = int(header['payload_offset'])
    width = int(header['width'])
    height = int(header['height'])
    required_bitmap_size = width * height * 4
    data_offset = payload_offset + PSP_PCD_HEADER_SIZE

    if len(pcd_bytes) < data_offset + required_bitmap_size:
        raise ValueError('Truncated PSP RGBA8888 PCD texture data')

    pixels = _unswizzle_psp_texture(
        pcd_bytes[data_offset:data_offset + required_bitmap_size],
        width,
        height,
        4,
    )

    rgba_pixels: list[float] = [0.0] * (width * height * 4)
    out_index = 0
    for offset in range(0, required_bitmap_size, 4):
        r = pixels[offset]
        g = pixels[offset + 1]
        b = pixels[offset + 2]
        a = pixels[offset + 3]
        rgba_pixels[out_index] = r / 255.0
        rgba_pixels[out_index + 1] = g / 255.0
        rgba_pixels[out_index + 2] = b / 255.0
        rgba_pixels[out_index + 3] = a / 255.0
        out_index += 4
    return rgba_pixels


def _decode_psp_pcd_rgb565(header: dict, pcd_bytes: bytes) -> list[float]:
    payload_offset = int(header['payload_offset'])
    width = int(header['width'])
    height = int(header['height'])
    data_offset = payload_offset + PSP_PCD_HEADER_SIZE
    required_bitmap_size = width * height * 2

    if len(pcd_bytes) < data_offset + required_bitmap_size:
        raise ValueError('Truncated PSP RGB565 PCD texture data')

    pixels = _unswizzle_psp_texture(
        pcd_bytes[data_offset:data_offset + required_bitmap_size],
        width,
        height,
        2,
    )

    rgba_pixels: list[float] = [0.0] * (width * height * 4)
    out_index = 0
    for offset in range(0, required_bitmap_size, 2):
        value = pixels[offset] | (pixels[offset + 1] << 8)
        # A mess, not sure if this is the right way
        r = (value & 0x001F) * 255 // 31
        g = ((value >> 5) & 0x003F) * 255 // 63
        b = ((value >> 11) & 0x001F) * 255 // 31
        rgba_pixels[out_index] = r / 255.0
        rgba_pixels[out_index + 1] = g / 255.0
        rgba_pixels[out_index + 2] = b / 255.0
        rgba_pixels[out_index + 3] = 1.0
        out_index += 4
    return rgba_pixels


def _decode_psp_pcd_rgba4444(header: dict, pcd_bytes: bytes) -> list[float]:
    payload_offset = int(header['payload_offset'])
    width = int(header['width'])
    height = int(header['height'])
    data_offset = payload_offset + PSP_PCD_HEADER_SIZE
    required_bitmap_size = width * height * 2

    if len(pcd_bytes) < data_offset + required_bitmap_size:
        raise ValueError('Truncated PSP RGBA4444 PCD texture data')

    pixels = _unswizzle_psp_texture(
        pcd_bytes[data_offset:data_offset + required_bitmap_size],
        width,
        height,
        2,
    )

    rgba_pixels: list[float] = [0.0] * (width * height * 4)
    out_index = 0
    for offset in range(0, required_bitmap_size, 2):
        value = pixels[offset] | (pixels[offset + 1] << 8)
        r = (value & 0x000F) * 17
        g = ((value >> 4) & 0x000F) * 17
        b = ((value >> 8) & 0x000F) * 17
        a = ((value >> 12) & 0x000F) * 17
        rgba_pixels[out_index] = r / 255.0
        rgba_pixels[out_index + 1] = g / 255.0
        rgba_pixels[out_index + 2] = b / 255.0
        rgba_pixels[out_index + 3] = a / 255.0
        out_index += 4
    return rgba_pixels


def decode_supported_psp_texture(pcd_bytes: bytes) -> Optional[dict]:
    header = _read_psp_pcd_header(pcd_bytes)
    if header is None:
        return None

    kind = int(header['kind'])
    if kind in PSP_PCD_T8_KINDS:
        pixels = _decode_psp_pcd_rgba8888_clut(header, pcd_bytes)
        return {
            'supported': True,
            'format_name': 'PSP_T8_RGBA8888_CLUT',
            'width': int(header['width']),
            'height': int(header['height']),
            'pixels': _flip_rgba_float_y(pixels, int(header['width']), int(header['height'])),
            'has_alpha': True,
        }

    if kind in {PSP_PCD_KIND_T4_RGBA8888_CLUT, PSP_PCD_KIND_T4_RGBA8888_CLUT_ALT}:
        pixels = _decode_psp_pcd_rgba8888_clut4(header, pcd_bytes)
        return {
            'supported': True,
            'format_name': 'PSP_T4_RGBA8888_CLUT',
            'width': int(header['width']),
            'height': int(header['height']),
            'pixels': _flip_rgba_float_y(pixels, int(header['width']), int(header['height'])),
            'has_alpha': True,
        }

    if kind in PSP_PCD_RGB565_KINDS:
        pixels = _decode_psp_pcd_rgb565(header, pcd_bytes)
        return {
            'supported': True,
            'format_name': 'PSP_RGB565',
            'width': int(header['width']),
            'height': int(header['height']),
            'pixels': _flip_rgba_float_y(pixels, int(header['width']), int(header['height'])),
            'has_alpha': True,
        }

    if kind in PSP_PCD_RGBA4444_KINDS:
        pixels = _decode_psp_pcd_rgba4444(header, pcd_bytes)
        return {
            'supported': True,
            'format_name': 'PSP_RGBA4444',
            'width': int(header['width']),
            'height': int(header['height']),
            'pixels': _flip_rgba_float_y(pixels, int(header['width']), int(header['height'])),
            'has_alpha': True,
        }

    if kind in {PSP_PCD_KIND_RGBA8888, PSP_PCD_KIND_RGBA8888_ALT}:
        pixels = _decode_psp_pcd_rgba8888(header, pcd_bytes)
        return {
            'supported': True,
            'format_name': 'PSP_RGBA8888',
            'width': int(header['width']),
            'height': int(header['height']),
            'pixels': _flip_rgba_float_y(pixels, int(header['width']), int(header['height'])),
            'has_alpha': True,
        }

    return {
        'supported': False,
        'format_name': f'PSP_UNKNOWN_0x{kind:X}',
        'width': int(header['width']),
        'height': int(header['height']),
    }
