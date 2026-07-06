from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SECTION_BODY_OFFSET = 0x18
GC_CMPR_FORMAT = 0x05
GC_CMPR_ALPHA_FORMAT = 0x09


@dataclass
class GameCubeTextureData:
    format_id: int
    width: int
    height: int
    data_size: int
    unk_0c: int
    unk_0e: int
    bitmap_data: bytes


def _read_be_u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 2], 'big')


def _read_be_u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset:offset + 4], 'big')


def parse_gamecube_pcd_bytes(pcd_bytes: bytes) -> Optional[GameCubeTextureData]:
    if len(pcd_bytes) < SECTION_BODY_OFFSET + 16:
        return None

    body_offset = SECTION_BODY_OFFSET
    format_id = _read_be_u32(pcd_bytes, body_offset + 0)
    width = _read_be_u16(pcd_bytes, body_offset + 4)
    height = _read_be_u16(pcd_bytes, body_offset + 6)
    data_size = _read_be_u32(pcd_bytes, body_offset + 8)
    unk_0c = _read_be_u16(pcd_bytes, body_offset + 12)
    unk_0e = _read_be_u16(pcd_bytes, body_offset + 14)

    if format_id < 0 or format_id > 0x20:
        return None
    if width <= 0 or height <= 0 or width > 4096 or height > 4096 or data_size <= 0:
        return None

    data_offset = body_offset + 16
    data_end = data_offset + data_size
    if data_end > len(pcd_bytes):
        return None

    return GameCubeTextureData(
        format_id=format_id,
        width=width,
        height=height,
        data_size=data_size,
        unk_0c=unk_0c,
        unk_0e=unk_0e,
        bitmap_data=pcd_bytes[data_offset:data_end],
    )


def _rgb565_to_rgba8888(color: int) -> tuple[int, int, int, int]:
    r = ((color >> 11) & 0x1F) * 255 // 31
    g = ((color >> 5) & 0x3F) * 255 // 63
    b = (color & 0x1F) * 255 // 31
    return (r, g, b, 255)


def _expand_cmpr_palette(color0: int, color1: int) -> list[tuple[int, int, int, int]]:
    palette0 = _rgb565_to_rgba8888(color0)
    palette1 = _rgb565_to_rgba8888(color1)
    if color0 > color1:
        palette2 = tuple((2 * palette0[i] + palette1[i]) // 3 for i in range(4))
        palette3 = tuple((palette0[i] + 2 * palette1[i]) // 3 for i in range(4))
    else:
        palette2 = tuple((palette0[i] + palette1[i]) // 2 for i in range(4))
        palette3 = (0, 0, 0, 0)
    return [palette0, palette1, palette2, palette3]


def _decode_cmpr_subblock(block: bytes) -> list[tuple[int, int, int, int]]:
    color0 = int.from_bytes(block[0:2], 'big')
    color1 = int.from_bytes(block[2:4], 'big')
    palette = _expand_cmpr_palette(color0, color1)
    pixels: list[tuple[int, int, int, int]] = []
    for row in range(4):
        selectors = block[4 + row]
        for col in range(4):
            palette_index = (selectors >> (6 - (col * 2))) & 0x3
            pixels.append(palette[palette_index])
    return pixels


def _shift_coord(index: int, count: int, shift: int, wrap: bool) -> int:
    if wrap:
        return ((index + shift) % count + count) % count if count > 0 else 0
    return index + shift


def _decode_i4_tiled(
    texture_bytes: bytes,
    width: int,
    height: int,
    tile_width: int = 8,
    tile_height: int = 8,
    tile_shift_x: int = 0,
    tile_shift_y: int = 0,
    pixel_shift_x: int = 0,
    pixel_shift_y: int = 0,
    wrap_shift: bool = True,
) -> bytearray:
    required_bytes = (width * height) // 2
    if len(texture_bytes) < required_bytes:
        raise ValueError(f'Truncated GameCube I4 plane: expected {required_bytes} bytes, got {len(texture_bytes)}')

    alpha8 = bytearray(width * height)
    tiles_w = (width + tile_width - 1) // tile_width
    tiles_h = (height + tile_height - 1) // tile_height
    offset = 0
    for src_tile_y in range(tiles_h):
        for src_tile_x in range(tiles_w):
            dst_tile_x = _shift_coord(src_tile_x, tiles_w, tile_shift_x, wrap_shift)
            dst_tile_y = _shift_coord(src_tile_y, tiles_h, tile_shift_y, wrap_shift)
            tile_x = (dst_tile_x * tile_width) + pixel_shift_x
            tile_y = (dst_tile_y * tile_height) + pixel_shift_y
            for row in range(tile_height):
                for col in range(0, tile_width, 2):
                    packed = texture_bytes[offset]
                    offset += 1
                    hi = (packed >> 4) & 0xF
                    lo = packed & 0xF

                    dst_y = tile_y + row
                    dst_x0 = tile_x + col
                    dst_x1 = dst_x0 + 1

                    if 0 <= dst_y < height and 0 <= dst_x0 < width:
                        alpha8[dst_y * width + dst_x0] = hi * 17
                    if 0 <= dst_y < height and 0 <= dst_x1 < width:
                        alpha8[dst_y * width + dst_x1] = lo * 17
    return alpha8


def _gamecube_cmpr_top_mip_size(width: int, height: int) -> int:
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    tiles_x = (blocks_x + 1) // 2
    tiles_y = (blocks_y + 1) // 2
    return tiles_x * tiles_y * 32

def decode_gamecube_cmpr(texture: GameCubeTextureData, payload: bytes | None = None) -> bytearray:
    width = int(texture.width)
    height = int(texture.height)
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4
    tiles_x = (blocks_x + 1) // 2
    tiles_y = (blocks_y + 1) // 2
    top_mip_size = _gamecube_cmpr_top_mip_size(width, height)
    source = texture.bitmap_data if payload is None else payload
    if len(source) < top_mip_size:
        raise ValueError(f'Truncated GameCube CMPR top mip: expected {top_mip_size} bytes, got {len(source)}')

    rgba8 = bytearray(width * height * 4)
    payload = source[:top_mip_size]
    offset = 0
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile_origin_x = tile_x * 8
            tile_origin_y = tile_y * 8
            for subblock_index in range(4):
                block = payload[offset:offset + 8]
                offset += 8
                decoded = _decode_cmpr_subblock(block)
                sub_origin_x = tile_origin_x + (4 if (subblock_index & 1) else 0)
                sub_origin_y = tile_origin_y + (4 if (subblock_index & 2) else 0)
                for py in range(4):
                    for px in range(4):
                        dst_x = sub_origin_x + px
                        dst_y = sub_origin_y + py
                        if dst_x >= width or dst_y >= height:
                            continue
                        r, g, b, a = decoded[(py * 4) + px]
                        dst = ((dst_y * width) + dst_x) * 4
                        rgba8[dst + 0] = r
                        rgba8[dst + 1] = g
                        rgba8[dst + 2] = b
                        rgba8[dst + 3] = a

    return rgba8


def decode_gamecube_format_09(texture: GameCubeTextureData) -> bytearray:
    width = int(texture.width)
    height = int(texture.height)
    top_mip_size = _gamecube_cmpr_top_mip_size(width, height)
    source = texture.bitmap_data

    if len(source) < top_mip_size * 2:
        raise ValueError(
            f'Truncated GameCube format-09 texture: expected at least {top_mip_size * 2} bytes, got {len(source)}'
        )
    color_payload = source[:top_mip_size]
    alpha_payload_offset = len(source) // 2 if (len(source) % 2) == 0 else top_mip_size

    # Very random stuff, I have no idea about this

    if texture.unk_0c == 0:
        alpha_tile_shift_x = 0
        alpha_tile_shift_y = 0
        alpha_pixel_shift_x = 0
        alpha_pixel_shift_y = 0
        alpha_wrap_shift = True
    else:
        alpha_payload_offset = max(0, alpha_payload_offset - 32)
        alpha_tile_shift_x = 0
        alpha_tile_shift_y = 1
        alpha_pixel_shift_x = 0
        alpha_pixel_shift_y = -8
        alpha_wrap_shift = False

    if alpha_payload_offset + top_mip_size > len(source):
        alpha_payload_offset = max(0, len(source) - top_mip_size)
    alpha_payload = source[alpha_payload_offset:alpha_payload_offset + top_mip_size]

    color_rgba = decode_gamecube_cmpr(texture, color_payload)
    alpha8 = _decode_i4_tiled(
        alpha_payload,
        width,
        height,
        8,
        8,
        tile_shift_x=alpha_tile_shift_x,
        tile_shift_y=alpha_tile_shift_y,
        pixel_shift_x=alpha_pixel_shift_x,
        pixel_shift_y=alpha_pixel_shift_y,
        wrap_shift=alpha_wrap_shift,
    )

    for pixel_index, alpha_value in enumerate(alpha8):
        color_rgba[pixel_index * 4 + 3] = alpha_value
    return color_rgba




def _flip_rgba_y(rgba: bytearray, width: int, height: int) -> bytearray:
    out = bytearray(len(rgba))
    row = width * 4
    for y in range(height):
        src_y = height - 1 - y
        out[y * row:(y + 1) * row] = rgba[src_y * row:(src_y + 1) * row]
    return out

def decode_supported_gamecube_texture(pcd_bytes: bytes) -> Optional[dict]:
    texture = parse_gamecube_pcd_bytes(pcd_bytes)
    if texture is None:
        return None

    if texture.format_id == GC_CMPR_FORMAT:
        rgba8 = decode_gamecube_cmpr(texture)
    elif texture.format_id == GC_CMPR_ALPHA_FORMAT:
        rgba8 = decode_gamecube_format_09(texture)
    else:
        return {
            'platform': 'gamecube',
            'format_id': texture.format_id,
            'width': texture.width,
            'height': texture.height,
            'supported': False,
        }

    rgba8 = _flip_rgba_y(rgba8, texture.width, texture.height)

    return {
        'platform': 'gamecube',
        'format_id': texture.format_id,
        'width': texture.width,
        'height': texture.height,
        'supported': True,
        'pixels': [channel / 255.0 for channel in rgba8],
        'has_alpha': True,
    }
