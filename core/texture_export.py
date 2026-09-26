from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Optional

import bpy

from ..platforms.pc.texture import (
    D3DFMT_A8B8G8R8,
    D3DFMT_A8R8G8B8,
    D3DFMT_X8B8G8R8,
    D3DFMT_X8R8G8B8,
)
from .log import logger

PCD_MAGIC = 0x39444350
D3DFMT_DXT1 = 0x31545844
D3DFMT_DXT3 = 0x33545844
D3DFMT_DXT5 = 0x35545844
DDS_FOURCC_FLAG = 0x00000004
DDS_RGB_FLAG = 0x00000040
DDS_ALPHA_PIXELS_FLAG = 0x00000001
DDS_DX10_FOURCC = 0x30315844

_SUPPORTED_DDS_FORMATS = {
    D3DFMT_DXT1,
    D3DFMT_DXT3,
    D3DFMT_DXT5,
    D3DFMT_A8R8G8B8,
    D3DFMT_X8R8G8B8,
    D3DFMT_A8B8G8R8,
    D3DFMT_X8B8G8R8,
}

_SRGB_SOURCE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.tga'}
_DATA_COLORSPACE_NAMES = {
    'non-color',
    'non color',
    'raw',
    'linear',
}

_ALPHA_OPAQUE_THRESHOLD = 0.999
_ALPHA_FOREACH_GET_MAX_FLOATS = 16_777_216
_MIN_SAFE_DXT_DIMENSION = 4
_TEXTURE_EXPORT_CACHE: dict[tuple, bytes] = {}


def clear_texture_export_cache() -> None:
    _TEXTURE_EXPORT_CACHE.clear()


def _is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _safe_texture_dimensions(width: int, height: int) -> tuple[int, int, bool]:
    src_width = max(1, int(width or 1))
    src_height = max(1, int(height or 1))
    safe_width = max(_MIN_SAFE_DXT_DIMENSION, _next_power_of_two(src_width))
    safe_height = max(_MIN_SAFE_DXT_DIMENSION, _next_power_of_two(src_height))
    changed = safe_width != src_width or safe_height != src_height
    return safe_width, safe_height, changed




def _expected_full_mip_count(width: int, height: int) -> int:
    mip_width = max(1, int(width or 1))
    mip_height = max(1, int(height or 1))
    levels = 1
    while mip_width > 1 or mip_height > 1:
        mip_width = max(1, mip_width // 2)
        mip_height = max(1, mip_height // 2)
        levels += 1
    return levels


def _expected_trlau_mip_count(width: int, height: int) -> int:
    mip_width = max(1, int(width or 1))
    mip_height = max(1, int(height or 1))
    levels = 1
    while mip_width > 1 and mip_height > 1:
        mip_width = max(1, mip_width // 2)
        mip_height = max(1, mip_height // 2)
        levels += 1
    return levels


def _image_dimensions(image) -> tuple[int, int]:
    try:
        size = getattr(image, 'size', None)
        if size is not None and len(size) >= 2:
            return max(1, int(size[0] or 1)), max(1, int(size[1] or 1))
    except Exception:
        pass
    return 1, 1


def _safe_stem(value: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() or ch in {'_', '-', '.'} else '_' for ch in str(value or '').strip())
    cleaned = cleaned.strip('._')
    return cleaned or 'texture'


def _image_cache_identity(image) -> int:
    try:
        as_pointer = getattr(image, 'as_pointer', None)
        if callable(as_pointer):
            return int(as_pointer())
    except Exception:
        pass
    return id(image)


def _encoded_texture_cache_key(
    image,
    *,
    dds_format: str,
    resize_to: Optional[tuple[int, int]],
    mip_count: int,
    separate_alpha: bool,
) -> tuple:
    width, height = _image_dimensions(image)
    return (
        _image_cache_identity(image),
        int(width),
        int(height),
        str(dds_format),
        tuple(resize_to) if resize_to is not None else None,
        int(mip_count),
        bool(separate_alpha),
        _normalize_colorspace_name(_image_colorspace_name(image)),
    )


def _resolve_existing_image_path(image) -> Optional[Path]:
    candidates: list[str] = []
    filepath_from_user = getattr(image, 'filepath_from_user', None)
    if callable(filepath_from_user):
        try:
            resolved_user_path = filepath_from_user()
        except TypeError:
            resolved_user_path = ''
        if resolved_user_path:
            candidates.append(str(resolved_user_path))
    for raw in (
        getattr(image, 'filepath', ''),
        getattr(image, 'filepath_raw', ''),
    ):
        if raw:
            candidates.append(str(raw))
    for raw in candidates:
        try:
            resolved = Path(bpy.path.abspath(raw))
        except Exception:
            resolved = Path(raw)
        if resolved.is_file():
            return resolved
    return None


def _get_packed_image_bytes(image) -> Optional[bytes]:
    packed = getattr(image, 'packed_file', None)
    if packed is None:
        return None
    data = getattr(packed, 'data', None)
    if data is None:
        return None
    try:
        return bytes(data)
    except Exception:
        return None


def _get_existing_dds_bytes(image) -> Optional[bytes]:
    packed_bytes = _get_packed_image_bytes(image)
    if packed_bytes and packed_bytes[:4] == b'DDS ':
        return packed_bytes

    source_path = _resolve_existing_image_path(image)
    if source_path is not None and source_path.suffix.lower() == '.dds':
        try:
            return source_path.read_bytes()
        except Exception as exc:
            logger.warning('Failed to read DDS source %s: %s', source_path, exc)
    return None


def _image_has_alpha(image) -> bool:
    """Return True only when the image contains meaningful alpha data.

    Blender often exposes PNG/JPEG data through an RGBA pixel buffer even when
    the source texture is effectively opaque. Choosing DXT5 from the channel
    count alone therefore bloats those textures and makes re-imported PCDs look
    like they have alpha. DXT5 is only required when at least one pixel is not
    fully opaque.
    """
    try:
        channels = int(getattr(image, 'channels', 4) or 4)
    except Exception:
        channels = 4
    if channels < 4:
        return False

    try:
        alpha_mode = str(getattr(image, 'alpha_mode', '') or '').upper()
    except Exception:
        alpha_mode = ''
    if alpha_mode == 'NONE':
        return False

    try:
        depth = int(getattr(image, 'depth', 0) or 0)
    except Exception:
        depth = 0
    if depth in {8, 16, 24, 48}:
        return False

    pixels = getattr(image, 'pixels', None)
    if pixels is None:
        return True

    try:
        pixel_count = len(pixels)
    except Exception:
        return True
    if pixel_count <= 0:
        return False

    try:
        if pixel_count <= _ALPHA_FOREACH_GET_MAX_FLOATS and hasattr(pixels, 'foreach_get'):
            buffer = array('f', [0.0]) * pixel_count
            pixels.foreach_get(buffer)
            return any(
                float(buffer[index]) < _ALPHA_OPAQUE_THRESHOLD
                for index in range(channels - 1, pixel_count, channels)
            )
    except Exception as exc:
        logger.debug(
            'Fast alpha scan failed for image %s: %s',
            getattr(image, 'name', '<unnamed>'),
            exc,
        )

    try:
        for index in range(channels - 1, pixel_count, channels):
            if float(pixels[index]) < _ALPHA_OPAQUE_THRESHOLD:
                return True
        return False
    except Exception as exc:
        logger.warning(
            'Could not inspect alpha for image %s; using DXT5 to avoid losing transparency: %s',
            getattr(image, 'name', '<unnamed>'),
            exc,
        )
        return True


def _image_colorspace_name(image) -> str:
    settings = getattr(image, 'colorspace_settings', None)
    if settings is None:
        return ''
    try:
        return str(getattr(settings, 'name', '') or '').strip()
    except Exception:
        return ''


def _normalize_colorspace_name(value: str) -> str:
    return str(value or '').strip().replace('_', '-').lower()


def _image_is_non_color_or_linear(image) -> bool:
    colorspace = _normalize_colorspace_name(_image_colorspace_name(image))
    return (
        colorspace in _DATA_COLORSPACE_NAMES
        or colorspace.startswith('linear')
        or colorspace.startswith('raw')
        or 'non-color' in colorspace
    )


def _should_preserve_srgb_for_texconv(image, input_path: Path) -> bool:
    """Return True when texconv should keep externally authored sRGB data.

    TRLAU PCD/DDS payloads are legacy D3D9 texture bytes. When exporting from
    Blender's pixel buffer we must keep those values as raw game-space UNORM
    data; applying an sRGB encode makes the DXT color endpoints too bright
    compared with the original files.
    """
    try:
        suffix = Path(input_path).suffix.lower()
    except Exception:
        suffix = ''
    if suffix not in _SRGB_SOURCE_EXTENSIONS:
        return False

    if suffix == '.tga':
        return False

    if _image_is_non_color_or_linear(image):
        return False

    colorspace = _normalize_colorspace_name(_image_colorspace_name(image))
    if not colorspace:
        return suffix in {'.png'}
    return 'srgb' in colorspace


def _image_file_format_for_path(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == '.tga':
        return 'TARGA_RAW'
    if suffix in {'.tif', '.tiff'}:
        return 'TIFF'
    return 'PNG'


def _clamp_float01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _linear_float_to_srgb_float(value: float) -> float:
    value = _clamp_float01(value)
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055


def _float_to_unorm8(value: float, *, srgb_encode: bool = False) -> int:
    if srgb_encode:
        value = _linear_float_to_srgb_float(value)
    else:
        value = _clamp_float01(value)
    return max(0, min(255, int(round(value * 255.0))))


def _read_image_pixels(image) -> tuple[array, int, int, int]:
    width, height = _image_dimensions(image)
    try:
        channels = int(getattr(image, 'channels', 4) or 4)
    except Exception:
        channels = 4
    channels = max(1, min(4, channels))
    expected = max(1, width * height * channels)

    pixels = getattr(image, 'pixels', None)
    if pixels is None:
        raise ValueError('Image has no readable pixel buffer')

    buffer = array('f', [0.0]) * expected
    try:
        if hasattr(pixels, 'foreach_get'):
            pixels.foreach_get(buffer)
        else:
            for index in range(expected):
                buffer[index] = float(pixels[index])
    except Exception as exc:
        raise ValueError(f'Could not read image pixels: {exc}') from exc

    return buffer, width, height, channels


def _float_to_unorm8_fast(value: float) -> int:
    if value <= 0.0:
        return 0
    if value >= 1.0:
        return 255
    return int(value * 255.0 + 0.5)


def _write_tga_from_image_pixels(image, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pixels, width, height, channels = _read_image_pixels(image)

    header = bytearray(18)
    header[2] = 2
    header[12:14] = int(width).to_bytes(2, 'little')
    header[14:16] = int(height).to_bytes(2, 'little')
    header[16] = 32
    header[17] = 0x28

    q = _float_to_unorm8_fast
    row_bytes = int(width) * 4

    with output_path.open('wb') as fh:
        fh.write(header)
        for y in range(height - 1, -1, -1):
            row = bytearray(row_bytes)
            row_base = y * width * channels
            out_offset = 0
            if channels == 1:
                for x in range(width):
                    v = pixels[row_base + x]
                    b = q(v)
                    row[out_offset] = b
                    row[out_offset + 1] = b
                    row[out_offset + 2] = b
                    row[out_offset + 3] = 255
                    out_offset += 4
            elif channels == 2:
                for x in range(width):
                    offset = row_base + x * 2
                    v = pixels[offset]
                    b = q(v)
                    row[out_offset] = b
                    row[out_offset + 1] = b
                    row[out_offset + 2] = b
                    row[out_offset + 3] = q(pixels[offset + 1])
                    out_offset += 4
            elif channels == 3:
                for x in range(width):
                    offset = row_base + x * 3
                    row[out_offset] = q(pixels[offset + 2])
                    row[out_offset + 1] = q(pixels[offset + 1])
                    row[out_offset + 2] = q(pixels[offset])
                    row[out_offset + 3] = 255
                    out_offset += 4
            else:
                for x in range(width):
                    offset = row_base + x * 4
                    row[out_offset] = q(pixels[offset + 2])
                    row[out_offset + 1] = q(pixels[offset + 1])
                    row[out_offset + 2] = q(pixels[offset])
                    row[out_offset + 3] = q(pixels[offset + 3])
                    out_offset += 4
            fh.write(row)

    return output_path

def _export_image_copy(image, output_path: Path, *, force_copy: bool = False, preserve_zero_alpha_rgb: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_path = _resolve_existing_image_path(image)
    if not force_copy and source_path is not None and source_path.suffix.lower() != '.dds':
        return source_path

    if output_path.suffix.lower() == '.tga':
        return _write_tga_from_image_pixels(image, output_path)

    previous_filepath_raw = getattr(image, 'filepath_raw', '')
    previous_file_format = getattr(image, 'file_format', 'PNG')
    previous_filepath = getattr(image, 'filepath', '')
    previous_alpha_mode = getattr(image, 'alpha_mode', None)
    try:
        image.filepath_raw = str(output_path)
        image.file_format = _image_file_format_for_path(output_path)
        if preserve_zero_alpha_rgb:
            try:
                image.alpha_mode = 'CHANNEL_PACKED'
            except Exception:
                pass
        try:
            image.save()
        except Exception:
            image.save_render(str(output_path), scene=bpy.context.scene)
    finally:
        try:
            image.filepath_raw = previous_filepath_raw
        except Exception:
            pass
        try:
            image.filepath = previous_filepath
        except Exception:
            pass
        try:
            image.file_format = previous_file_format
        except Exception:
            pass
        if previous_alpha_mode is not None:
            try:
                image.alpha_mode = previous_alpha_mode
            except Exception:
                pass
    return output_path


def find_texconv_executable() -> Optional[Path]:
    env_value = os.environ.get('TRLAU_TEXCONV')
    if env_value:
        env_path = Path(env_value)
        if env_path.is_file():
            return env_path

    addon_root = Path(__file__).resolve().parents[1]
    search_paths = [
        addon_root / 'bin' / 'Texconv.exe',
        addon_root / 'bin' / 'texconv.exe',
        addon_root / 'Texconv.exe',
        addon_root / 'texconv.exe',
    ]
    for candidate in search_paths:
        if candidate.is_file():
            return candidate

    which_candidate = shutil.which('texconv.exe') or shutil.which('texconv')
    if which_candidate:
        return Path(which_candidate)
    return None


def _dds_mip_count_to_pcd_num_mipmaps(dds_mipmaps: int) -> int:
    return max(0, int(dds_mipmaps) - 1)


def _dds_level_size(width: int, height: int, dds_format: int) -> int:
    level_width = max(1, int(width or 1))
    level_height = max(1, int(height or 1))
    dds_format = int(dds_format) & 0xFFFFFFFF
    if dds_format == D3DFMT_DXT1:
        block_bytes = 8
    elif dds_format in {D3DFMT_DXT3, D3DFMT_DXT5}:
        block_bytes = 16
    elif dds_format in {D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8, D3DFMT_A8B8G8R8, D3DFMT_X8B8G8R8}:
        return level_width * level_height * 4
    else:
        raise ValueError(f'Unsupported DDS format 0x{dds_format:08X}')
    return ((level_width + 3) // 4) * ((level_height + 3) // 4) * block_bytes


def _dds_payload_size_for_mip_count(width: int, height: int, dds_format: int, mipmaps: int) -> int:
    total_size = 0
    level_width = max(1, int(width or 1))
    level_height = max(1, int(height or 1))
    for _level_index in range(max(1, int(mipmaps or 1))):
        total_size += _dds_level_size(level_width, level_height, dds_format)
        level_width = max(1, level_width // 2)
        level_height = max(1, level_height // 2)
    return total_size


def _trim_parsed_dds_mipmaps(parsed: dict, max_mipmaps: Optional[int]) -> dict:
    if max_mipmaps is None:
        return parsed

    source_mipmaps = max(1, int(parsed.get('mipmaps', 1) or 1))
    target_mipmaps = max(1, min(source_mipmaps, int(max_mipmaps or 1)))
    if target_mipmaps >= source_mipmaps:
        return parsed

    target_size = _dds_payload_size_for_mip_count(
        int(parsed.get('width', 1) or 1),
        int(parsed.get('height', 1) or 1),
        int(parsed.get('format', 0) or 0),
        target_mipmaps,
    )
    bitmap_data = bytes(parsed.get('bitmap_data', b''))
    if len(bitmap_data) < target_size:
        raise ValueError(
            f'Truncated DDS payload: expected at least {target_size} bytes for {target_mipmaps} mip level(s), got {len(bitmap_data)}'
        )

    trimmed = dict(parsed)
    trimmed['mipmaps'] = target_mipmaps
    trimmed['bitmap_data'] = bitmap_data[:target_size]
    return trimmed


def _dxt5_alpha_palette(alpha0: int, alpha1: int) -> tuple[int, ...]:
    alpha0 = int(alpha0) & 0xFF
    alpha1 = int(alpha1) & 0xFF
    if alpha0 > alpha1:
        return (
            alpha0,
            alpha1,
            (6 * alpha0 + 1 * alpha1) // 7,
            (5 * alpha0 + 2 * alpha1) // 7,
            (4 * alpha0 + 3 * alpha1) // 7,
            (3 * alpha0 + 4 * alpha1) // 7,
            (2 * alpha0 + 5 * alpha1) // 7,
            (1 * alpha0 + 6 * alpha1) // 7,
        )
    return (
        alpha0,
        alpha1,
        (4 * alpha0 + 1 * alpha1) // 5,
        (3 * alpha0 + 2 * alpha1) // 5,
        (2 * alpha0 + 3 * alpha1) // 5,
        (1 * alpha0 + 4 * alpha1) // 5,
        0,
        255,
    )


def _dxt5_block_has_meaningful_alpha(block: bytes) -> bool:
    if len(block) < 8:
        return True
    palette = _dxt5_alpha_palette(block[0], block[1])
    alpha_indices = int.from_bytes(block[2:8], 'little')
    for pixel_index in range(16):
        palette_index = (alpha_indices >> (pixel_index * 3)) & 0x7
        if palette[palette_index] < 255:
            return True
    return False


def _dxt3_block_has_meaningful_alpha(block: bytes) -> bool:
    if len(block) < 8:
        return True
    return any(byte != 0xFF for byte in block[:8])


def _compressed_dds_has_meaningful_alpha(parsed: dict) -> Optional[bool]:
    """Inspect legacy compressed DDS alpha blocks.

    Returns True when alpha is used, False when all encoded alpha is fully
    opaque, and None for formats where this helper does not apply.
    """
    dds_format = int(parsed.get('format', 0) or 0)
    if dds_format not in {D3DFMT_DXT3, D3DFMT_DXT5}:
        return None

    width = max(1, int(parsed.get('width', 1) or 1))
    height = max(1, int(parsed.get('height', 1) or 1))
    mipmaps = max(1, int(parsed.get('mipmaps', 1) or 1))
    bitmap_data = bytes(parsed.get('bitmap_data', b''))
    offset = 0
    block_size = 16

    for _mip_index in range(mipmaps):
        block_width = max(1, (width + 3) // 4)
        block_height = max(1, (height + 3) // 4)
        mip_size = block_width * block_height * block_size
        mip_data = bitmap_data[offset:offset + mip_size]
        if len(mip_data) < mip_size:
            return True

        for block_offset in range(0, mip_size, block_size):
            block = mip_data[block_offset:block_offset + block_size]
            if dds_format == D3DFMT_DXT5:
                if _dxt5_block_has_meaningful_alpha(block):
                    return True
            elif _dxt3_block_has_meaningful_alpha(block):
                return True

        offset += mip_size
        width = max(1, width // 2)
        height = max(1, height // 2)

    return False


def _run_texconv(
    texconv_path: Path,
    input_path: Path,
    output_dir: Path,
    dds_format: str,
    *,
    preserve_srgb: bool = False,
    resize_to: Optional[tuple[int, int]] = None,
    mip_count: Optional[int] = None,
    separate_alpha: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(texconv_path),
        '-nologo',
        '-y',
        '-ft', 'dds',
        '-f', str(dds_format),
        '-dx9',
    ]
    if mip_count is not None:
        command.extend(['-m', str(max(1, int(mip_count)))])
    if separate_alpha:
        command.append('-sepalpha')
    if preserve_srgb:
        command.append('-srgb')
    if resize_to is not None:
        target_width, target_height = resize_to
        command.extend(['-w', str(int(target_width)), '-h', str(int(target_height))])
    command.extend([
        '-o', str(output_dir),
        str(input_path),
    ])

    if sys.platform != 'win32' and texconv_path.suffix.lower() == '.exe':
        wine = shutil.which('wine')
        if wine:
            command.insert(0, wine)

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f'Failed to launch texconv: {exc}') from exc
    if completed.returncode != 0:
        stdout = (completed.stdout or '').strip()
        stderr = (completed.stderr or '').strip()
        details = stderr or stdout or f'exit code {completed.returncode}'
        raise RuntimeError(f'texconv failed for {input_path.name}: {details}')

    output_path = output_dir / f'{input_path.stem}.dds'
    if not output_path.is_file():
        produced = sorted(output_dir.glob('*.dds'))
        if produced:
            output_path = produced[0]
    if not output_path.is_file():
        raise RuntimeError(f'texconv did not produce a DDS file for {input_path.name}')
    return output_path


def _parse_dds_bytes(dds_bytes: bytes) -> dict:
    if len(dds_bytes) < 128 or dds_bytes[:4] != b'DDS ':
        raise ValueError('DDS header is missing or truncated')
    header_size = struct.unpack_from('<I', dds_bytes, 4)[0]
    if int(header_size) != 124:
        raise ValueError(f'Unsupported DDS header size: {header_size}')

    height = int(struct.unpack_from('<I', dds_bytes, 12)[0])
    width = int(struct.unpack_from('<I', dds_bytes, 16)[0])
    mipmaps = int(struct.unpack_from('<I', dds_bytes, 28)[0]) or 1
    pixel_flags = int(struct.unpack_from('<I', dds_bytes, 80)[0])
    fourcc = int(struct.unpack_from('<I', dds_bytes, 84)[0])
    rgb_bit_count = int(struct.unpack_from('<I', dds_bytes, 88)[0])
    r_mask = int(struct.unpack_from('<I', dds_bytes, 92)[0])
    g_mask = int(struct.unpack_from('<I', dds_bytes, 96)[0])
    b_mask = int(struct.unpack_from('<I', dds_bytes, 100)[0])
    a_mask = int(struct.unpack_from('<I', dds_bytes, 104)[0])

    if pixel_flags & DDS_FOURCC_FLAG:
        if fourcc == DDS_DX10_FOURCC:
            raise ValueError('DX10 DDS headers need conversion through texconv before export')
        dds_format = fourcc
    elif (pixel_flags & DDS_RGB_FLAG) and rgb_bit_count == 32:
        if (r_mask, g_mask, b_mask, a_mask) == (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000):
            dds_format = D3DFMT_A8R8G8B8
        elif (r_mask, g_mask, b_mask, a_mask) == (0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000):
            dds_format = D3DFMT_X8R8G8B8
        elif (r_mask, g_mask, b_mask, a_mask) == (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000):
            dds_format = D3DFMT_A8B8G8R8
        elif (r_mask, g_mask, b_mask, a_mask) == (0x000000FF, 0x0000FF00, 0x00FF0000, 0x00000000):
            dds_format = D3DFMT_X8B8G8R8
        else:
            raise ValueError('Unsupported 32-bit DDS channel masks')
    else:
        raise ValueError('Unsupported DDS pixel format')

    if dds_format not in _SUPPORTED_DDS_FORMATS:
        raise ValueError(f'Unsupported DDS format 0x{dds_format:08X}')

    bitmap_data = dds_bytes[128:]
    if not bitmap_data:
        raise ValueError('DDS payload is empty')

    return {
        'format': dds_format,
        'width': max(1, width),
        'height': max(1, height),
        'mipmaps': max(1, mipmaps),
        'bitmap_data': bitmap_data,
    }


def dds_to_pcd_bytes(dds_bytes: bytes, *, max_mipmaps: Optional[int] = None) -> bytes:
    parsed = _trim_parsed_dds_mipmaps(_parse_dds_bytes(dds_bytes), max_mipmaps)
    header = struct.pack(
        '<I I I I H H B B H',
        int(PCD_MAGIC),
        int(parsed['format']) & 0xFFFFFFFF,
        len(parsed['bitmap_data']),
        0,
        int(parsed['width']) & 0xFFFF,
        int(parsed['height']) & 0xFFFF,
        0,
        _dds_mip_count_to_pcd_num_mipmaps(int(parsed['mipmaps'])) & 0xFF,
        3,
    )
    return header + bytes(parsed['bitmap_data'])


def image_to_pcd_bytes(image, *, texture_id: int, material_name: str = '') -> bytes:
    if image is None:
        raise ValueError('Material has no image texture assigned')

    force_dxt1 = False
    known_has_alpha: Optional[bool] = None
    needs_mip_generation = False
    target_mipmaps: Optional[int] = None
    resize_to: Optional[tuple[int, int]] = None
    existing_dds = _get_existing_dds_bytes(image)
    if existing_dds is not None:
        try:
            parsed_existing_dds = _parse_dds_bytes(existing_dds)
            dds_width = int(parsed_existing_dds.get('width', 1) or 1)
            dds_height = int(parsed_existing_dds.get('height', 1) or 1)
            safe_width, safe_height, needs_resize = _safe_texture_dimensions(dds_width, dds_height)
            if needs_resize:
                resize_to = (safe_width, safe_height)
                logger.warning(
                    'Resizing texture %s from %dx%d to %dx%d for TRLAU-safe DXT export.',
                    getattr(image, 'name', '<unnamed>'),
                    dds_width,
                    dds_height,
                    safe_width,
                    safe_height,
                )
            target_mipmaps = _expected_trlau_mip_count(safe_width, safe_height)
            existing_mipmaps = max(1, int(parsed_existing_dds.get('mipmaps', 1) or 1))
            existing_format = int(parsed_existing_dds.get('format', 0) or 0)
            if existing_format == D3DFMT_DXT1:
                known_has_alpha = False
            needs_mip_generation = existing_mipmaps < target_mipmaps
            if needs_mip_generation:
                logger.info(
                    'Regenerating mipmaps for texture %s: DDS has %d level(s), expected %d TRLAU level(s).',
                    getattr(image, 'name', '<unnamed>'),
                    existing_mipmaps,
                    target_mipmaps,
                )
            elif existing_mipmaps > target_mipmaps:
                logger.info(
                    'Dropping non-TRLAU mip level(s) for texture %s: DDS has %d level(s), exporting %d TRLAU level(s).',
                    getattr(image, 'name', '<unnamed>'),
                    existing_mipmaps,
                    target_mipmaps,
                )
            compressed_alpha = _compressed_dds_has_meaningful_alpha(parsed_existing_dds)
            if compressed_alpha is not None:
                known_has_alpha = bool(compressed_alpha)
            if compressed_alpha is False:
                force_dxt1 = True
                logger.info(
                    'Recompressing opaque %s texture %s as DXT1 instead of copying the existing alpha-capable DDS payload.',
                    'DXT5' if int(parsed_existing_dds.get('format', 0)) == D3DFMT_DXT5 else 'DXT3',
                    getattr(image, 'name', '<unnamed>'),
                )
        except Exception as exc:
            logger.warning(
                'Direct DDS->PCD conversion failed for image %s (texture_id=0x%X): %s. Falling back to texconv.',
                getattr(image, 'name', '<unnamed>'),
                int(texture_id),
                exc,
            )

    if target_mipmaps is None:
        if resize_to is None:
            image_width, image_height = _image_dimensions(image)
            safe_width, safe_height, needs_resize = _safe_texture_dimensions(image_width, image_height)
            if needs_resize:
                resize_to = (safe_width, safe_height)
                logger.warning(
                    'Resizing texture %s from %dx%d to %dx%d for TRLAU-safe DXT export.',
                    getattr(image, 'name', '<unnamed>'),
                    image_width,
                    image_height,
                    safe_width,
                    safe_height,
                )
        else:
            safe_width, safe_height = resize_to

        target_mipmaps = _expected_trlau_mip_count(safe_width, safe_height)

    texconv_path = find_texconv_executable()
    if texconv_path is None:
        raise RuntimeError(
            'Texture export requires texconv. Place Texconv.exe in trlau_editor/bin or set the TRLAU_TEXCONV environment variable.'
        )

    if force_dxt1:
        has_alpha = False
    elif known_has_alpha is not None:
        has_alpha = bool(known_has_alpha)
    else:
        has_alpha = _image_has_alpha(image)
    alpha_format = 'DXT5' if has_alpha else 'DXT1'
    cache_key = _encoded_texture_cache_key(
        image,
        dds_format=alpha_format,
        resize_to=resize_to,
        mip_count=target_mipmaps,
        separate_alpha=has_alpha,
    )
    cached_pcd = _TEXTURE_EXPORT_CACHE.get(cache_key)
    if cached_pcd is not None:
        return cached_pcd

    temp_root = Path(tempfile.mkdtemp(prefix='trlau_texture_export_'))
    input_name = f'{int(texture_id):04x}_{_safe_stem(material_name or getattr(image, "name", "texture"))}'
    input_ext = '.tga'
    try:
        input_path = _export_image_copy(
            image,
            temp_root / f'{input_name}{input_ext}',
            force_copy=has_alpha,
            preserve_zero_alpha_rgb=has_alpha,
        )
        preserve_srgb = _should_preserve_srgb_for_texconv(image, input_path)
        dds_output = _run_texconv(
            texconv_path,
            input_path,
            temp_root / 'dds',
            alpha_format,
            preserve_srgb=preserve_srgb,
            resize_to=resize_to,
            mip_count=target_mipmaps,
            separate_alpha=has_alpha,
        )
        pcd_data = dds_to_pcd_bytes(dds_output.read_bytes(), max_mipmaps=target_mipmaps)
        _TEXTURE_EXPORT_CACHE[cache_key] = pcd_data
        return pcd_data
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
