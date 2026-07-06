from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SECTION_HEADER_SIZE = 0x18
TEXTURE_HEADER_SIZE = 0x10

XBOX_FMT_DXT1 = 0x05
XBOX_FMT_DXT5 = 0x09

D3DFMT_DXT1 = 0x31545844
D3DFMT_DXT5 = 0x35545844

_HAS_ALPHA = {
    D3DFMT_DXT1: False,
    D3DFMT_DXT5: True,
}


@dataclass
class XboxTextureData:
    format_id: int
    width: int
    height: int
    data_size: int
    unk_0c: int
    unk_0e: int
    bitmap_data: bytes


def _read_le_u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 2], 'little')


def _read_le_u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], 'little')


def parse_xbox_pcd_bytes(pcd_bytes: bytes) -> Optional[XboxTextureData]:
    minimum_size = SECTION_HEADER_SIZE + TEXTURE_HEADER_SIZE
    if len(pcd_bytes) < minimum_size:
        return None

    header_offset = SECTION_HEADER_SIZE
    format_id = _read_le_u16(pcd_bytes, header_offset + 2)
    width = _read_le_u16(pcd_bytes, header_offset + 4)
    height = _read_le_u16(pcd_bytes, header_offset + 6)
    data_size = _read_le_u32(pcd_bytes, header_offset + 8)
    unk_0c = _read_le_u16(pcd_bytes, header_offset + 12)
    unk_0e = _read_le_u16(pcd_bytes, header_offset + 14)

    if width <= 0 or height <= 0 or width > 4096 or height > 4096 or data_size <= 0:
        return None

    data_offset = SECTION_HEADER_SIZE + TEXTURE_HEADER_SIZE
    data_end = data_offset + data_size
    if data_end > len(pcd_bytes):
        return None

    return XboxTextureData(
        format_id=format_id,
        width=width,
        height=height,
        data_size=data_size,
        unk_0c=unk_0c,
        unk_0e=unk_0e,
        bitmap_data=pcd_bytes[data_offset:data_end],
    )


def _expected_dxt_level_size(width: int, height: int, block_bytes: int) -> int:
    return ((max(1, width) + 3) // 4) * ((max(1, height) + 3) // 4) * block_bytes


def _expected_full_mip_chain_size(width: int, height: int, block_bytes: int) -> tuple[int, int]:
    total_size = 0
    mipmaps = 0
    level_width = max(1, int(width))
    level_height = max(1, int(height))

    while True:
        total_size += _expected_dxt_level_size(level_width, level_height, block_bytes)
        mipmaps += 1
        if level_width == 1 and level_height == 1:
            break
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)

    return total_size, mipmaps


def _infer_mipmap_count(width: int, height: int, data_size: int, block_bytes: int) -> Optional[int]:
    top_level_size = _expected_dxt_level_size(width, height, block_bytes)
    if data_size == top_level_size:
        return 1

    full_chain_size, full_chain_mipmaps = _expected_full_mip_chain_size(width, height, block_bytes)
    if data_size == full_chain_size:
        return full_chain_mipmaps

    remaining_size = int(data_size)
    level_width = max(1, int(width))
    level_height = max(1, int(height))
    mipmaps = 0
    while remaining_size > 0:
        level_size = _expected_dxt_level_size(level_width, level_height, block_bytes)
        if remaining_size < level_size:
            return None
        remaining_size -= level_size
        mipmaps += 1
        if remaining_size == 0:
            return mipmaps
        if level_width == 1 and level_height == 1:
            return None
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)

    return None


def _map_format(format_id: int) -> tuple[Optional[int], Optional[int]]:
    if format_id == XBOX_FMT_DXT1:
        return D3DFMT_DXT1, 8
    if format_id == XBOX_FMT_DXT5:
        return D3DFMT_DXT5, 16
    return None, None


def decode_supported_xbox_texture(pcd_bytes: bytes) -> Optional[dict]:
    texture = parse_xbox_pcd_bytes(pcd_bytes)
    if texture is None:
        return None

    dds_format, block_bytes = _map_format(int(texture.format_id))
    if dds_format is None or block_bytes is None:
        return {
            'supported': False,
            'format_id': texture.format_id,
            'width': texture.width,
            'height': texture.height,
        }

    mipmaps = _infer_mipmap_count(texture.width, texture.height, texture.data_size, block_bytes)
    if mipmaps is None:
        return {
            'supported': False,
            'format_id': texture.format_id,
            'width': texture.width,
            'height': texture.height,
        }

    return {
        'supported': True,
        'format_id': texture.format_id,
        'dds_format': dds_format,
        'width': texture.width,
        'height': texture.height,
        'has_alpha': _HAS_ALPHA.get(dds_format, True),
        'bitmap_data': texture.bitmap_data,
        'mipmaps': mipmaps,
    }
