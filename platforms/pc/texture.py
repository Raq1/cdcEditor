from __future__ import annotations

from struct import pack, unpack, pack_into


NORMAL_SCALE = 127.0
PCD_MAGIC = 0x39444350
DDS_FOURCC_FLAG = 0x00000004
DDS_CAPS_TEXTURE = 0x00001000
DDS_RGB_FLAG = 0x00000040
DDS_RGBA_FLAG = 0x00000041

D3DFMT_A8R8G8B8 = 0x15
D3DFMT_X8R8G8B8 = 0x16
D3DFMT_A8B8G8R8 = 0x20
D3DFMT_X8B8G8R8 = 0x21


def _create_dds_header(width: int, height: int, mipmaps: int, dds_format: int, data_size: int) -> bytes:
    header = bytearray(128)
    pack_into('<4sI', header, 0, b'DDS ', 124)
    pack_into('<I', header, 8, 0x0002100F)
    pack_into('<I', header, 12, height)
    pack_into('<I', header, 16, width)
    pack_into('<I', header, 20, max(1, data_size))
    pack_into('<I', header, 24, 0)
    pack_into('<I', header, 28, mipmaps if mipmaps > 0 else 1)
    pack_into('<I', header, 76, 32)

    if dds_format in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DFMT_A8B8G8R8, D3DFMT_X8B8G8R8):
        has_alpha = dds_format in (D3DFMT_A8R8G8B8, D3DFMT_A8B8G8R8)
        pack_into('<I', header, 80, DDS_RGBA_FLAG if has_alpha else DDS_RGB_FLAG)
        pack_into('<I', header, 88, 32)
        if dds_format in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8):
            pack_into('<I', header, 92, 0x00FF0000)
            pack_into('<I', header, 96, 0x0000FF00)
            pack_into('<I', header, 100, 0x000000FF)
        else:
            pack_into('<I', header, 92, 0x000000FF)
            pack_into('<I', header, 96, 0x0000FF00)
            pack_into('<I', header, 100, 0x00FF0000)
        pack_into('<I', header, 104, 0xFF000000 if has_alpha else 0x00000000)
    else:
        pack_into('<I', header, 80, DDS_FOURCC_FLAG)
        pack_into('<4s', header, 84, pack('<I', dds_format))

    pack_into('<I', header, 108, DDS_CAPS_TEXTURE)
    return bytes(header)


def _convert_pcd_argb32_to_rgba(pixels: bytes, width: int, height: int, alpha_mode: str) -> list[float]:
    pixel_count = width * height
    required_size = pixel_count * 4
    if len(pixels) < required_size:
        raise ValueError(f'Truncated 32-bit PCD texture data: expected {required_size} bytes, got {len(pixels)}')

    rgba_pixels: list[float] = [0.0] * (pixel_count * 4)
    out_index = 0
    for offset in range(0, required_size, 4):
        b = pixels[offset]
        g = pixels[offset + 1]
        r = pixels[offset + 2]
        a = pixels[offset + 3]
        rgba_pixels[out_index] = r / 255.0
        rgba_pixels[out_index + 1] = g / 255.0
        rgba_pixels[out_index + 2] = b / 255.0
        rgba_pixels[out_index + 3] = (a / 255.0) if alpha_mode == 'alpha' else 1.0
        out_index += 4
    return rgba_pixels


def _decode_pcd_bytes(pcd_bytes: bytes) -> dict:
    if len(pcd_bytes) < 48:
        raise ValueError('PCD file is too small to contain texture data')

    magic_number, dds_format, bitmap_size, _palette_size, width, height, _depth, mipmaps, _flags = unpack(
        '<I i I I H H B B H',
        pcd_bytes[24:48],
    )
    if magic_number != PCD_MAGIC:
        raise ValueError(f'Unexpected PCD magic: {hex(magic_number)}')

    bitmap_data = pcd_bytes[48:48 + bitmap_size]
    if len(bitmap_data) != bitmap_size:
        raise ValueError(f'Truncated PCD texture data: expected {bitmap_size} bytes, got {len(bitmap_data)}')

    return {
        'dds_format': dds_format,
        'bitmap_data': bitmap_data,
        'width': width,
        'height': height,
        'mipmaps': mipmaps,
    }


def _extract_dds_from_pcd_bytes(pcd_bytes: bytes) -> bytes:
    decoded = _decode_pcd_bytes(pcd_bytes)
    dds_mipmaps = max(1, int(decoded['mipmaps']) + 1)
    return _create_dds_header(
        decoded['width'],
        decoded['height'],
        dds_mipmaps,
        decoded['dds_format'],
        len(decoded['bitmap_data']),
    ) + decoded['bitmap_data']
