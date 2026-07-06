from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

SECTION_HEADER_SIZE = 0x18
PS2_PCD_MAGIC_INDEXED8 = 0x02600000
PS2_PCD_MAGIC_INDEXED8_ALT = 0x02400000
PS2_PCD_MAGIC_RGBA32 = 0x00700000
PS2_PCD_INDEXED8_MAGICS = (PS2_PCD_MAGIC_INDEXED8, PS2_PCD_MAGIC_INDEXED8_ALT)
PS2_PCD_MAGICS = (*PS2_PCD_INDEXED8_MAGICS, PS2_PCD_MAGIC_RGBA32)
PS2_PCD_HEADER_SIZE = 0x48
PS2_PCD_PALETTE_OFFSET = 0x48
PS2_PCD_PALETTE_SIZE = 0x400
PS2_PCD_IMAGE_PACKET_OFFSET = PS2_PCD_PALETTE_OFFSET + PS2_PCD_PALETTE_SIZE
PS2_PCD_IMAGE_PACKET_SIZE = 0x10
PS2_PCD_IMAGE_DATA_OFFSET = PS2_PCD_IMAGE_PACKET_OFFSET + PS2_PCD_IMAGE_PACKET_SIZE

TR8_PS2_PCD_MAGIC = 0x4053474C
TR8_PS2_PCD_HEADER_OFFSET = 0x18
TR8_PS2_PCD_WIDTH_OFFSET = 0x24
TR8_PS2_PCD_HEIGHT_OFFSET = 0x26
TR8_PS2_PCD_LOG_WIDTH_OFFSET = 0x28
TR8_PS2_PCD_LOG_HEIGHT_OFFSET = 0x2A
TR8_PS2_PCD_FORMAT_OFFSET = 0x2C
TR8_PS2_PCD_SETUP_OFFSET = 0x2E
TR8_PS2_PCD_FLAG_OFFSET = 0x20
TR8_PS2_PCD_PALETTE_OFFSET = 0x50
TR8_PS2_PCD_8BIT_IMAGE_PACKET_OFFSET = TR8_PS2_PCD_PALETTE_OFFSET + 0x400
TR8_PS2_PCD_8BIT_IMAGE_DATA_OFFSET = TR8_PS2_PCD_8BIT_IMAGE_PACKET_OFFSET + 0x10
TR8_PS2_PCD_4BIT_PALETTE_SIZE = 0x40
TR8_PS2_PCD_4BIT_IMAGE_PACKET_OFFSET = TR8_PS2_PCD_PALETTE_OFFSET + TR8_PS2_PCD_4BIT_PALETTE_SIZE
TR8_PS2_PCD_4BIT_IMAGE_DATA_OFFSET = TR8_PS2_PCD_4BIT_IMAGE_PACKET_OFFSET + 0x10


@dataclass
class Ps2TextureData:
    width: int
    height: int
    log_width: int
    log_height: int
    format_id: int
    palette_rgba: bytes
    index_data: bytes


@dataclass
class Ps2TextureRgba32Data:
    width: int
    height: int
    log_width: int
    log_height: int
    format_id: int
    rgba_data: bytes


@dataclass
class Ps2UnderworldTextureData:
    width: int
    height: int
    log_width: int
    log_height: int
    format_id: int
    setup_id: int
    palette_rgba: bytes
    index_data: bytes
    bits_per_pixel: int


def _read_le_u16(data: bytes, offset: int) -> int:
    if offset + 2 > len(data):
        return 0
    return int.from_bytes(data[offset:offset + 2], 'little')


def _read_le_u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return 0
    return int.from_bytes(data[offset:offset + 4], 'little')


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def looks_like_ps2_pcd_bytes(pcd_bytes: bytes) -> bool:
    """Cheap guard used to keep PS2 PCDs out of the PC/DDS decoder.

    Known PS2 samples use a SECT wrapper, width/height/log2 fields at +0x1C,
    and one of the observed GS setup values at +0x18.  The secondary shape
    check catches future PS2 texture setup values so they log as unsupported
    PS2 textures instead of falling into the PC PCD decoder.
    """
    if len(pcd_bytes) < PS2_PCD_HEADER_SIZE:
        return False
    if _read_le_u32(pcd_bytes, 0x18) in PS2_PCD_MAGICS:
        return True
    if _read_le_u32(pcd_bytes, TR8_PS2_PCD_HEADER_OFFSET) == TR8_PS2_PCD_MAGIC:
        width = _read_le_u16(pcd_bytes, TR8_PS2_PCD_WIDTH_OFFSET)
        height = _read_le_u16(pcd_bytes, TR8_PS2_PCD_HEIGHT_OFFSET)
        log_width = _read_le_u16(pcd_bytes, TR8_PS2_PCD_LOG_WIDTH_OFFSET)
        log_height = _read_le_u16(pcd_bytes, TR8_PS2_PCD_LOG_HEIGHT_OFFSET)
        if width <= 0 or height <= 0 or width > 4096 or height > 4096:
            return False
        if not _is_power_of_two(width) or not _is_power_of_two(height):
            return False
        return log_width == (width.bit_length() - 1) and log_height == (height.bit_length() - 1)
    if pcd_bytes[:4] != b'SECT':
        return False

    width = _read_le_u16(pcd_bytes, 0x1C)
    height = _read_le_u16(pcd_bytes, 0x1E)
    log_width = _read_le_u16(pcd_bytes, 0x20)
    log_height = _read_le_u16(pcd_bytes, 0x22)
    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return False
    if not _is_power_of_two(width) or not _is_power_of_two(height):
        return False
    return log_width == (width.bit_length() - 1) and log_height == (height.bit_length() - 1)


def parse_ps2_pcd_bytes(pcd_bytes: bytes) -> Optional[Ps2TextureData]:
    """Parse the observed PS2 8-bit indexed PCD variant.

    Observed file body:
      0x00 SECT section header
      0x18 0x02600000 or 0x02400000 PS2 texture marker / GS texture setup value
      0x1C u16 width
      0x1E u16 height
      0x20 u16 log2(width)
      0x22 u16 log2(height)
      0x24 u16 format-ish value
      0x48 256-entry RGBA CLUT, PS2 alpha range 0..128
      0x448 16-byte image upload packet header
      0x458 PSMT8-swizzled top mip indices

    The validation here is intentionally permissive.  Earlier builds required
    the image upload qword count to match width*height/16 exactly; that rejects
    otherwise PS2-shaped PCDs and lets them crash in the PC decoder.
    """
    if len(pcd_bytes) < PS2_PCD_IMAGE_DATA_OFFSET:
        return None
    if _read_le_u32(pcd_bytes, 0x18) not in PS2_PCD_INDEXED8_MAGICS:
        return None

    width = _read_le_u16(pcd_bytes, 0x1C)
    height = _read_le_u16(pcd_bytes, 0x1E)
    log_width = _read_le_u16(pcd_bytes, 0x20)
    log_height = _read_le_u16(pcd_bytes, 0x22)
    format_id = _read_le_u16(pcd_bytes, 0x24)

    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return None
    if not _is_power_of_two(width) or not _is_power_of_two(height):
        return None

    pixel_count = width * height
    palette_end = PS2_PCD_PALETTE_OFFSET + PS2_PCD_PALETTE_SIZE
    image_end = PS2_PCD_IMAGE_DATA_OFFSET + pixel_count
    if palette_end > len(pcd_bytes) or image_end > len(pcd_bytes):
        return None

    image_qwords = _read_le_u32(pcd_bytes, PS2_PCD_IMAGE_PACKET_OFFSET)
    upload_marker = _read_le_u32(pcd_bytes, PS2_PCD_IMAGE_PACKET_OFFSET + 4)
    if upload_marker not in (0, 0x08000000):
        return None
    if image_qwords not in (0, pixel_count // 16) and image_qwords * 16 > len(pcd_bytes):
        return None

    return Ps2TextureData(
        width=width,
        height=height,
        log_width=log_width,
        log_height=log_height,
        format_id=format_id,
        palette_rgba=pcd_bytes[PS2_PCD_PALETTE_OFFSET:palette_end],
        index_data=pcd_bytes[PS2_PCD_IMAGE_DATA_OFFSET:image_end],
    )


def parse_ps2_rgba32_pcd_bytes(pcd_bytes: bytes) -> Optional[Ps2TextureRgba32Data]:
    """Parse the observed PS2 32-bit RGBA PCD variant.

    Observed file body:
      0x00 SECT section header
      0x18 0x00700000 PS2 texture marker / GS texture setup value
      0x1C u16 width
      0x1E u16 height
      0x20 u16 log2(width)
      0x22 u16 log2(height)
      0x24 u16 format-ish value, 0 in the current sample
      0x48 linear top mip RGBA pixels

    This variant has no 256-entry CLUT.  For the supplied 128x128 sample, the
    bytes after +0x48 are already row-major RGBA pixels.  Earlier builds tried
    to GS-unswizzle this payload; that produced block/diagonal corruption.
    """
    if len(pcd_bytes) < PS2_PCD_HEADER_SIZE:
        return None
    if _read_le_u32(pcd_bytes, 0x18) != PS2_PCD_MAGIC_RGBA32:
        return None

    width = _read_le_u16(pcd_bytes, 0x1C)
    height = _read_le_u16(pcd_bytes, 0x1E)
    log_width = _read_le_u16(pcd_bytes, 0x20)
    log_height = _read_le_u16(pcd_bytes, 0x22)
    format_id = _read_le_u16(pcd_bytes, 0x24)

    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return None
    if not _is_power_of_two(width) or not _is_power_of_two(height):
        return None

    pixel_bytes = width * height * 4
    image_end = PS2_PCD_HEADER_SIZE + pixel_bytes
    if image_end > len(pcd_bytes):
        return None

    return Ps2TextureRgba32Data(
        width=width,
        height=height,
        log_width=log_width,
        log_height=log_height,
        format_id=format_id,
        rgba_data=pcd_bytes[PS2_PCD_HEADER_SIZE:image_end],
    )



def parse_tr8_ps2_pcd_bytes(pcd_bytes: bytes) -> Optional[Ps2UnderworldTextureData]:
    """Parse the Underworld PS2 LGS@ PCD top mip.

    TR8 PS2 PCDs are still wrapped in SECT, but their texture body starts at
    +0x18 with the LGS@ marker instead of the older TRL/TRA GS setup word.  The
    supplied Underworld samples contain indexed 8-bit and indexed 4-bit top
    mips.  This parser exposes the largest/top mip only; smaller mips and GS
    packet metadata are intentionally ignored by the Blender importer.
    """
    if len(pcd_bytes) < TR8_PS2_PCD_PALETTE_OFFSET:
        return None
    if _read_le_u32(pcd_bytes, TR8_PS2_PCD_HEADER_OFFSET) != TR8_PS2_PCD_MAGIC:
        return None

    width = _read_le_u16(pcd_bytes, TR8_PS2_PCD_WIDTH_OFFSET)
    height = _read_le_u16(pcd_bytes, TR8_PS2_PCD_HEIGHT_OFFSET)
    log_width = _read_le_u16(pcd_bytes, TR8_PS2_PCD_LOG_WIDTH_OFFSET)
    log_height = _read_le_u16(pcd_bytes, TR8_PS2_PCD_LOG_HEIGHT_OFFSET)
    format_id = _read_le_u16(pcd_bytes, TR8_PS2_PCD_FORMAT_OFFSET)
    setup_id = _read_le_u16(pcd_bytes, TR8_PS2_PCD_SETUP_OFFSET)
    mode_flag = _read_le_u16(pcd_bytes, TR8_PS2_PCD_FLAG_OFFSET)

    if width <= 0 or height <= 0 or width > 4096 or height > 4096:
        return None
    if not _is_power_of_two(width) or not _is_power_of_two(height):
        return None
    if log_width != (width.bit_length() - 1) or log_height != (height.bit_length() - 1):
        return None

    pixel_count = int(width) * int(height)
    image8_end = TR8_PS2_PCD_8BIT_IMAGE_DATA_OFFSET + pixel_count
    image4_end = TR8_PS2_PCD_4BIT_IMAGE_DATA_OFFSET + ((pixel_count + 1) // 2)
    can_8bit = image8_end <= len(pcd_bytes)
    can_4bit = image4_end <= len(pcd_bytes)
    use_4bit = False
    if int(mode_flag) == 0x14:
        use_4bit = can_4bit
    elif not can_8bit and can_4bit:
        use_4bit = True

    if use_4bit:
        palette_end = TR8_PS2_PCD_PALETTE_OFFSET + TR8_PS2_PCD_4BIT_PALETTE_SIZE
        if palette_end > len(pcd_bytes) or image4_end > len(pcd_bytes):
            return None
        return Ps2UnderworldTextureData(
            width=width,
            height=height,
            log_width=log_width,
            log_height=log_height,
            format_id=format_id,
            setup_id=setup_id,
            palette_rgba=pcd_bytes[TR8_PS2_PCD_PALETTE_OFFSET:palette_end],
            index_data=pcd_bytes[TR8_PS2_PCD_4BIT_IMAGE_DATA_OFFSET:image4_end],
            bits_per_pixel=4,
        )

    if can_8bit:
        palette_end = TR8_PS2_PCD_PALETTE_OFFSET + PS2_PCD_PALETTE_SIZE
        if palette_end > len(pcd_bytes):
            return None
        return Ps2UnderworldTextureData(
            width=width,
            height=height,
            log_width=log_width,
            log_height=log_height,
            format_id=format_id,
            setup_id=setup_id,
            palette_rgba=pcd_bytes[TR8_PS2_PCD_PALETTE_OFFSET:palette_end],
            index_data=pcd_bytes[TR8_PS2_PCD_8BIT_IMAGE_DATA_OFFSET:image8_end],
            bits_per_pixel=8,
        )

    return None


def _expand_indexed4_nibbles(raw: bytes, width: int, height: int) -> bytearray:
    pixel_count = int(width) * int(height)
    indices = bytearray(pixel_count)
    out_index = 0
    for value in raw:
        if out_index < pixel_count:
            indices[out_index] = int(value) & 0x0F
            out_index += 1
        if out_index < pixel_count:
            indices[out_index] = (int(value) >> 4) & 0x0F
            out_index += 1
        if out_index >= pixel_count:
            break
    return indices


def _unswizzle_ps2_palette_rgba(raw_palette: bytes) -> bytearray:
    """Undo the standard PS2 8-bit CLUT block swap.

    PS2 CSM1 8-bit palettes store colors in 32-entry groups where entries
    8..15 and 16..23 are exchanged.  The texture indices are already in the
    logical palette domain, so the CLUT must be rearranged before expansion.
    """
    if len(raw_palette) < PS2_PCD_PALETTE_SIZE:
        raise ValueError(f'Truncated PS2 PCD palette: expected 1024 bytes, got {len(raw_palette)}')

    palette = bytearray(PS2_PCD_PALETTE_SIZE)
    for index in range(256):
        stored_index = index
        if (index & 0x18) == 0x08:
            stored_index = index + 8
        elif (index & 0x18) == 0x10:
            stored_index = index - 8
        palette[index * 4:index * 4 + 4] = raw_palette[stored_index * 4:stored_index * 4 + 4]
    return palette


def _unswizzle_psmt8(swizzled: bytes, width: int, height: int) -> bytearray:
    """Convert PS2 PSMT8 texture memory order to linear row-major indices."""
    required_size = width * height
    if len(swizzled) < required_size:
        raise ValueError(f'Truncated PS2 PSMT8 texture data: expected {required_size} bytes, got {len(swizzled)}')

    if width < 16 or height < 4:
        return bytearray(swizzled[:required_size])

    linear = bytearray(required_size)
    for y in range(height):
        for x in range(width):
            block_location = (y & ~0x0F) * width + (x & ~0x0F) * 2
            swap_selector = (((y + 2) >> 2) & 0x01) * 4
            pos_y = (((y & ~0x03) >> 1) + (y & 0x01)) & 0x07
            column_location = pos_y * width * 2 + ((x + swap_selector) & 0x07) * 4
            byte_number = ((y >> 1) & 0x01) + ((x >> 2) & 0x02)
            src_offset = block_location + column_location + byte_number
            if src_offset < required_size:
                linear[(y * width) + x] = swizzled[src_offset]
    return linear


def _normalize_ps2_rgba32(rgba8: bytearray, width: int, height: int) -> tuple[bytearray, bool]:
    has_alpha = False
    pixel_count = width * height
    for pixel_index in range(pixel_count):
        alpha_offset = pixel_index * 4 + 3
        alpha = _scale_ps2_alpha(rgba8[alpha_offset])
        if alpha < 255:
            has_alpha = True
        rgba8[alpha_offset] = alpha
    return rgba8, has_alpha


def _flip_rgba_y(rgba8: bytearray, width: int, height: int) -> bytearray:
    row_size = width * 4
    flipped = bytearray(len(rgba8))
    for y in range(height):
        src = y * row_size
        dst = (height - 1 - y) * row_size
        flipped[dst:dst + row_size] = rgba8[src:src + row_size]
    return flipped


def _scale_ps2_alpha(alpha: int) -> int:
    # PS2 GS texture alpha is nominally 0..0x80, but some direct-color assets
    # top out at 0x7F.  Treat 0x7F/0x80 as fully opaque to avoid nearly-opaque
    # materials being flagged/blended unnecessarily.
    if alpha >= 0x7F:
        return 255
    return min(255, alpha * 2)


def _expand_indexed_rgba(indices: bytes, palette: bytes, width: int, height: int) -> tuple[bytearray, bool]:
    pixel_count = width * height
    rgba8 = bytearray(pixel_count * 4)
    has_alpha = False
    for pixel_index, palette_index in enumerate(indices[:pixel_count]):
        src = palette_index * 4
        dst = pixel_index * 4
        r = palette[src + 0]
        g = palette[src + 1]
        b = palette[src + 2]
        # GS alpha is 0..0x80, with 0x7F also used as max in some assets.
        a = _scale_ps2_alpha(palette[src + 3])
        if a < 255:
            has_alpha = True
        rgba8[dst + 0] = r
        rgba8[dst + 1] = g
        rgba8[dst + 2] = b
        rgba8[dst + 3] = a
    return rgba8, has_alpha


def decode_supported_ps2_texture(pcd_bytes: bytes) -> Optional[dict]:
    if not looks_like_ps2_pcd_bytes(pcd_bytes):
        return None

    tr8_texture = parse_tr8_ps2_pcd_bytes(pcd_bytes)
    if tr8_texture is not None:
        if int(tr8_texture.bits_per_pixel) == 8:
            palette = _unswizzle_ps2_palette_rgba(tr8_texture.palette_rgba)
            indices = _unswizzle_psmt8(tr8_texture.index_data, tr8_texture.width, tr8_texture.height)
            variant = 'tr8_lgs_psmt8_clut'
        else:
            # First-pass PSMT4 support: the supplied assets that use this path are
            # auxiliary/small maps.  Expand nibbles linearly; if future models expose
            # obvious PSMT4 swizzle corruption, this should be replaced with a full GS
            # PSMT4 unswizzler.
            palette = bytearray(tr8_texture.palette_rgba)
            indices = _expand_indexed4_nibbles(tr8_texture.index_data, tr8_texture.width, tr8_texture.height)
            variant = 'tr8_lgs_psmt4_clut_linear'
        rgba8, has_alpha = _expand_indexed_rgba(indices, palette, tr8_texture.width, tr8_texture.height)
        rgba8 = _flip_rgba_y(rgba8, tr8_texture.width, tr8_texture.height)

        return {
            'platform': 'ps2',
            'variant': variant,
            'magic': _read_le_u32(pcd_bytes, TR8_PS2_PCD_HEADER_OFFSET),
            'format_id': tr8_texture.format_id,
            'setup_id': tr8_texture.setup_id,
            'width': tr8_texture.width,
            'height': tr8_texture.height,
            'supported': True,
            'pixels': [channel / 255.0 for channel in rgba8],
            'has_alpha': has_alpha,
        }

    texture = parse_ps2_pcd_bytes(pcd_bytes)
    if texture is not None:
        palette = _unswizzle_ps2_palette_rgba(texture.palette_rgba)
        indices = _unswizzle_psmt8(texture.index_data, texture.width, texture.height)
        rgba8, has_alpha = _expand_indexed_rgba(indices, palette, texture.width, texture.height)
        rgba8 = _flip_rgba_y(rgba8, texture.width, texture.height)

        return {
            'platform': 'ps2',
            'variant': 'psmt8_clut',
            'magic': _read_le_u32(pcd_bytes, 0x18),
            'format_id': texture.format_id,
            'width': texture.width,
            'height': texture.height,
            'supported': True,
            'pixels': [channel / 255.0 for channel in rgba8],
            'has_alpha': has_alpha,
        }

    texture_rgba32 = parse_ps2_rgba32_pcd_bytes(pcd_bytes)
    if texture_rgba32 is not None:
        rgba8 = bytearray(texture_rgba32.rgba_data)
        rgba8, has_alpha = _normalize_ps2_rgba32(rgba8, texture_rgba32.width, texture_rgba32.height)
        rgba8 = _flip_rgba_y(rgba8, texture_rgba32.width, texture_rgba32.height)

        return {
            'platform': 'ps2',
            'variant': 'rgba32_linear',
            'format_id': texture_rgba32.format_id,
            'width': texture_rgba32.width,
            'height': texture_rgba32.height,
            'supported': True,
            'pixels': [channel / 255.0 for channel in rgba8],
            'has_alpha': has_alpha,
        }

    if _read_le_u32(pcd_bytes, TR8_PS2_PCD_HEADER_OFFSET) == TR8_PS2_PCD_MAGIC:
        return {
            'platform': 'ps2',
            'variant': 'tr8_lgs_unsupported',
            'format_id': _read_le_u16(pcd_bytes, TR8_PS2_PCD_FORMAT_OFFSET),
            'setup_id': _read_le_u16(pcd_bytes, TR8_PS2_PCD_SETUP_OFFSET),
            'width': _read_le_u16(pcd_bytes, TR8_PS2_PCD_WIDTH_OFFSET),
            'height': _read_le_u16(pcd_bytes, TR8_PS2_PCD_HEIGHT_OFFSET),
            'supported': False,
            'magic': _read_le_u32(pcd_bytes, TR8_PS2_PCD_HEADER_OFFSET),
        }

    return {
        'platform': 'ps2',
        'format_id': _read_le_u16(pcd_bytes, 0x24),
        'width': _read_le_u16(pcd_bytes, 0x1C),
        'height': _read_le_u16(pcd_bytes, 0x1E),
        'supported': False,
        'magic': _read_le_u32(pcd_bytes, 0x18),
    }



def _log2_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f'PS2 textures must use power-of-two dimensions; got {value}')
    return value.bit_length() - 1


def _scale_to_ps2_alpha(alpha: int) -> int:
    alpha = max(0, min(255, int(alpha)))
    if alpha >= 254:
        return 0x80
    return max(0, min(0x80, int(round(alpha / 2.0))))


def _swizzle_ps2_palette_rgba(logical_palette: bytes) -> bytearray:
    """Apply the standard PS2 8-bit CLUT block swap used by the decoder above."""
    if len(logical_palette) < PS2_PCD_PALETTE_SIZE:
        raise ValueError(f'Truncated PS2 palette: expected 1024 bytes, got {len(logical_palette)}')
    raw = bytearray(PS2_PCD_PALETTE_SIZE)
    for index in range(256):
        stored_index = index
        if (index & 0x18) == 0x08:
            stored_index = index + 8
        elif (index & 0x18) == 0x10:
            stored_index = index - 8
        raw[stored_index * 4:stored_index * 4 + 4] = logical_palette[index * 4:index * 4 + 4]
    return raw


def _swizzle_psmt8(linear: bytes, width: int, height: int) -> bytearray:
    """Convert row-major 8-bit indices to the PS2 PSMT8 memory order."""
    required_size = int(width) * int(height)
    if len(linear) < required_size:
        raise ValueError(f'Truncated linear PS2 PSMT8 source data: expected {required_size} bytes, got {len(linear)}')
    if width < 16 or height < 4:
        return bytearray(linear[:required_size])

    swizzled = bytearray(required_size)
    for y in range(int(height)):
        for x in range(int(width)):
            block_location = (y & ~0x0F) * width + (x & ~0x0F) * 2
            swap_selector = (((y + 2) >> 2) & 0x01) * 4
            pos_y = (((y & ~0x03) >> 1) + (y & 0x01)) & 0x07
            column_location = pos_y * width * 2 + ((x + swap_selector) & 0x07) * 4
            byte_number = ((y >> 1) & 0x01) + ((x >> 2) & 0x02)
            dst_offset = block_location + column_location + byte_number
            if dst_offset < required_size:
                swizzled[dst_offset] = linear[(y * width) + x]
    return swizzled


def _flip_linear_indices_y(indices: bytes, width: int, height: int) -> bytearray:
    row_size = int(width)
    flipped = bytearray(len(indices))
    for y in range(int(height)):
        src = y * row_size
        dst = (int(height) - 1 - y) * row_size
        flipped[dst:dst + row_size] = indices[src:src + row_size]
    return flipped


def build_ps2_rgba32_pcd_body(rgba_top_left: bytes, width: int, height: int, *, format_id: int = 0) -> bytes:
    """Build a PS2 RGBA32 PCD section payload body.

    The returned bytes are the standalone-section payload, not the SECT wrapper.
    Existing readers see the magic at full-file offset +0x18 after the section
    wrapper is added by the exporter.
    """
    width = int(width)
    height = int(height)
    log_width = _log2_power_of_two(width)
    log_height = _log2_power_of_two(height)
    required = width * height * 4
    if len(rgba_top_left) < required:
        raise ValueError(f'Truncated RGBA32 source: expected {required} bytes, got {len(rgba_top_left)}')

    body = bytearray(PS2_PCD_HEADER_SIZE - SECTION_HEADER_SIZE)
    body[0:4] = int(PS2_PCD_MAGIC_RGBA32).to_bytes(4, 'little')
    body[4:6] = width.to_bytes(2, 'little')
    body[6:8] = height.to_bytes(2, 'little')
    body[8:10] = log_width.to_bytes(2, 'little')
    body[10:12] = log_height.to_bytes(2, 'little')
    body[12:14] = (int(format_id) & 0xFFFF).to_bytes(2, 'little')

    # Import flips decoded PS2 textures to Blender's top-left view. Store the
    # inverse orientation and scale Blender alpha 0..255 back to GS alpha 0..128.
    src = bytearray(rgba_top_left[:required])
    for pixel in range(width * height):
        src[(pixel * 4) + 3] = _scale_to_ps2_alpha(src[(pixel * 4) + 3])
    body.extend(_flip_rgba_y(src, width, height))
    return bytes(body)


def _ps2_indexed8_mip_dimensions(width: int, height: int) -> list[tuple[int, int]]:
    """Return the PS2 indexed top-mip chain used by observed TRA/TRL PCDs.

    The game assets store the top level plus smaller levels down to 4x4.
    Earlier exporter builds wrote only the top mip.  The add-on could still
    re-import that, but the PS2 loader/runtime expects the full packet chain
    described by the texture header and can hang when those packets are absent.
    """
    dims: list[tuple[int, int]] = []
    w = int(width)
    h = int(height)
    while w >= 4 and h >= 4:
        dims.append((w, h))
        if w == 4 and h == 4:
            break
        w = max(1, w // 2)
        h = max(1, h // 2)
    return dims


def _generate_ps2_indexed8_header(width: int, height: int, *, magic: int = PS2_PCD_MAGIC_INDEXED8) -> bytearray:
    width = int(width)
    height = int(height)
    log_width = _log2_power_of_two(width)
    log_height = _log2_power_of_two(height)
    mip_count = max(0, min(log_width, log_height) - 2)

    body = bytearray(PS2_PCD_HEADER_SIZE - SECTION_HEADER_SIZE)
    body[0:4] = (int(magic) & 0xFFFFFFFF).to_bytes(4, 'little')
    body[4:6] = width.to_bytes(2, 'little')
    body[6:8] = height.to_bytes(2, 'little')
    body[8:10] = log_width.to_bytes(2, 'little')
    body[10:12] = log_height.to_bytes(2, 'little')
    body[12:14] = int(mip_count & 0xFFFF).to_bytes(2, 'little')
    # Observed indexed PS2 textures use 0xFF34, 0xFF44, 0xFF54, ... for
    # 16, 32, 64, ... top mips.  This field appears to be part of the GS/mip
    # setup, not a Blender-visible image value.
    setup_word = 0xFF00 | ((((max(log_width, log_height) - 1) & 0xF) << 4) | 0x4)
    body[14:16] = int(setup_word & 0xFFFF).to_bytes(2, 'little')

    dims = _ps2_indexed8_mip_dimensions(width, height)
    for index in range(min(8, len(dims))):
        w, h = dims[index]
        pages = max(1, (int(w) * int(h)) // 256)
        off = 0x10 + (index * 2)
        if off + 2 <= len(body):
            body[off:off + 2] = int(pages & 0xFFFF).to_bytes(2, 'little')
    # Constant values observed on indexed texture sections.  Preserve exact
    # template headers when possible, but provide sane generated values for new
    # texture sections.
    body[0x20:0x22] = (0x40).to_bytes(2, 'little')
    body[0x26:0x28] = (0x0800).to_bytes(2, 'little')
    return body


def _ps2_header_compatible(template_body: bytes | None, width: int, height: int, magic: int) -> bool:
    if not template_body or len(template_body) < (PS2_PCD_HEADER_SIZE - SECTION_HEADER_SIZE):
        return False
    if _read_le_u32(template_body, 0) not in PS2_PCD_INDEXED8_MAGICS:
        return False
    return (
        _read_le_u16(template_body, 4) == int(width)
        and _read_le_u16(template_body, 6) == int(height)
        and _read_le_u32(template_body, 0) == (int(magic) & 0xFFFFFFFF)
    )


def _palette_color(logical_palette_rgba: bytes, index: int) -> tuple[int, int, int, int]:
    offset = (int(index) & 0xFF) * 4
    return (
        int(logical_palette_rgba[offset + 0]),
        int(logical_palette_rgba[offset + 1]),
        int(logical_palette_rgba[offset + 2]),
        int(logical_palette_rgba[offset + 3]),
    )


def _nearest_palette_index(color: tuple[int, int, int, int], logical_palette_rgba: bytes) -> int:
    r, g, b, a = (int(color[0]), int(color[1]), int(color[2]), int(color[3]))
    best_index = 0
    best_score = None
    for index in range(256):
        pr, pg, pb, pa = _palette_color(logical_palette_rgba, index)
        score = ((r - pr) * (r - pr)) + ((g - pg) * (g - pg)) + ((b - pb) * (b - pb)) + (((a - pa) * (a - pa)) * 2)
        if best_score is None or score < best_score:
            best_score = score
            best_index = index
            if score == 0:
                break
    return int(best_index)


def _downsample_indexed8_level(prev_indices: bytes, prev_width: int, prev_height: int, logical_palette_rgba: bytes) -> tuple[bytes, int, int]:
    new_width = max(1, int(prev_width) // 2)
    new_height = max(1, int(prev_height) // 2)
    out = bytearray(new_width * new_height)
    for y in range(new_height):
        for x in range(new_width):
            acc = [0, 0, 0, 0, 0]
            for yy in range(2):
                sy = min(int(prev_height) - 1, (y * 2) + yy)
                for xx in range(2):
                    sx = min(int(prev_width) - 1, (x * 2) + xx)
                    idx = int(prev_indices[(sy * int(prev_width)) + sx]) & 0xFF
                    color = _palette_color(logical_palette_rgba, idx)
                    acc[0] += color[0]
                    acc[1] += color[1]
                    acc[2] += color[2]
                    acc[3] += color[3]
                    acc[4] += 1
            count = max(1, acc[4])
            avg = (
                int(round(acc[0] / count)),
                int(round(acc[1] / count)),
                int(round(acc[2] / count)),
                int(round(acc[3] / count)),
            )
            out[(y * new_width) + x] = _nearest_palette_index(avg, logical_palette_rgba)
    return bytes(out), new_width, new_height


def build_ps2_indexed8_pcd_body(index_top_left: bytes, logical_palette_rgba: bytes, width: int, height: int, *, format_id: int = 0, magic: int = PS2_PCD_MAGIC_INDEXED8, template_body: bytes | None = None) -> bytes:
    """Build an observed PS2 PSMT8+RGBA8888-CLUT PCD section payload body.

    The returned body includes the PS2 texture setup header, CLUT, and every
    8-bit mip packet down to 4x4.  The first-pass exporter originally wrote
    only the top mip and zeroed the runtime header fields at 0x0C..0x2F.  That
    was enough for the add-on decoder, but not for the game's texture loader.
    """
    width = int(width)
    height = int(height)
    log_width = _log2_power_of_two(width)
    log_height = _log2_power_of_two(height)
    pixel_count = width * height
    if len(index_top_left) < pixel_count:
        raise ValueError(f'Truncated PS2 indexed source: expected {pixel_count} bytes, got {len(index_top_left)}')
    if len(logical_palette_rgba) < PS2_PCD_PALETTE_SIZE:
        raise ValueError(f'Truncated PS2 palette source: expected 1024 bytes, got {len(logical_palette_rgba)}')

    if template_body and _read_le_u32(template_body, 0) in PS2_PCD_INDEXED8_MAGICS:
        magic = _read_le_u32(template_body, 0)

    palette = bytearray(logical_palette_rgba[:PS2_PCD_PALETTE_SIZE])
    for entry in range(256):
        palette[(entry * 4) + 3] = _scale_to_ps2_alpha(palette[(entry * 4) + 3])

    if _ps2_header_compatible(template_body, width, height, magic):
        body = bytearray(template_body[:PS2_PCD_HEADER_SIZE - SECTION_HEADER_SIZE])
    else:
        body = _generate_ps2_indexed8_header(width, height, magic=magic)
    # Always refresh dimensions/logs in case a compatible template had stale or
    # partially edited values.  For same-size replacements this leaves the
    # unknown runtime setup bytes intact.
    body[0:4] = (int(magic) & 0xFFFFFFFF).to_bytes(4, 'little')
    body[4:6] = width.to_bytes(2, 'little')
    body[6:8] = height.to_bytes(2, 'little')
    body[8:10] = log_width.to_bytes(2, 'little')
    body[10:12] = log_height.to_bytes(2, 'little')

    logical_palette_for_mips = bytes(palette)
    body.extend(_swizzle_ps2_palette_rgba(bytes(palette)))

    level_indices = bytes(index_top_left[:pixel_count])
    level_width = width
    level_height = height
    for _level, (expected_width, expected_height) in enumerate(_ps2_indexed8_mip_dimensions(width, height)):
        if int(level_width) != int(expected_width) or int(level_height) != int(expected_height):
            # Defensive only; the level chain generation should already match.
            break
        level_pixel_count = int(level_width) * int(level_height)
        body.extend(int(level_pixel_count // 16).to_bytes(4, 'little'))
        body.extend(int(0x08000000).to_bytes(4, 'little'))
        body.extend(b'\x00' * 8)
        body.extend(_swizzle_psmt8(_flip_linear_indices_y(level_indices[:level_pixel_count], level_width, level_height), level_width, level_height))
        if level_width == 4 and level_height == 4:
            break
        level_indices, level_width, level_height = _downsample_indexed8_level(level_indices, level_width, level_height, logical_palette_for_mips)
    return bytes(body)
