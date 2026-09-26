from __future__ import annotations

import tempfile
from pathlib import Path
from struct import unpack_from
from typing import Optional, Tuple

import bpy

from ...core.log import logger
from ..nintendo.texture import decode_supported_gamecube_texture
from ..pc.texture import (
    D3DFMT_A8R8G8B8,
    D3DFMT_X8R8G8B8,
    _convert_pcd_argb32_to_rgba,
    _create_dds_header,
    _decode_pcd_bytes,
    _extract_dds_from_pcd_bytes,
)
from ..ps2.texture import decode_supported_ps2_texture, looks_like_ps2_pcd_bytes
from ..psp.texture import _psp_pcd_dimensions, decode_supported_psp_texture
from ..ps3.texture import decode_supported_ps3_texture, _iter_ps3_texture_header_candidates
from ..xbox.texture import decode_supported_xbox_texture
from ..xbox360.texture import decode_supported_xbox360_texture, _section_payload_offset as _x360_section_payload_offset


def _valid_texture_dimensions(width: int, height: int) -> Tuple[int, int] | None:
    try:
        width = int(width)
        height = int(height)
    except Exception:
        return None
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        return None
    return (width, height)


def _ps3_texture_dimensions_from_bytes(pcd_bytes: bytes) -> Tuple[int, int] | None:
    try:
        for candidate in _iter_ps3_texture_header_candidates(pcd_bytes):
            width = int(unpack_from('>H', pcd_bytes, int(candidate['width_offset']))[0])
            height = int(unpack_from('>H', pcd_bytes, int(candidate['height_offset']))[0])
            dimensions = _valid_texture_dimensions(width, height)
            if dimensions is not None:
                return dimensions
    except Exception:
        return None
    return None


def _xbox360_texture_dimensions_from_bytes(pcd_bytes: bytes) -> Tuple[int, int] | None:
    try:
        base = _x360_section_payload_offset(pcd_bytes)
        if base + 0x10 <= len(pcd_bytes):
            width = int(unpack_from('>H', pcd_bytes, base + 0x04)[0])
            height = int(unpack_from('>H', pcd_bytes, base + 0x06)[0])
            data_size = int(unpack_from('>I', pcd_bytes, base + 0x08)[0])
            dimensions = _valid_texture_dimensions(width, height)
            if dimensions is not None and data_size > 0 and base + 0x10 + data_size <= len(pcd_bytes):
                return dimensions
    except Exception:
        pass
    return _ps3_texture_dimensions_from_bytes(pcd_bytes)

class PlatformTextureLoaderMixin:
    def _texture_import_enabled(self) -> bool:
        settings = getattr(self, 'settings', None)
        if settings is None:
            return True
        return bool(getattr(settings, 'import_textures', True))

    def _read_texture_dimensions_from_pcd(self, texture_id: int) -> Tuple[int, int] | None:
        if not self._texture_import_enabled():
            return None

        texture_path = self._find_texture_path(texture_id)
        if texture_path is None:
            return None

        cache_key = str(texture_path)
        if cache_key in self._texture_dimension_cache:
            return self._texture_dimension_cache[cache_key]

        result: Tuple[int, int] | None = None
        try:
            with open(texture_path, 'rb') as fh:
                pcd_bytes = fh.read()

            platform_hint = str(getattr(self, '_active_uv_format', '') or '').lower()
            if platform_hint == 'xbox360':
                result = _xbox360_texture_dimensions_from_bytes(pcd_bytes)
                self._texture_dimension_cache[cache_key] = result
                return result
            if platform_hint == 'ps3':
                result = _ps3_texture_dimensions_from_bytes(pcd_bytes)
                self._texture_dimension_cache[cache_key] = result
                return result
            if platform_hint == 'underworld' and looks_like_ps2_pcd_bytes(pcd_bytes):
                try:
                    ps2_decoded = decode_supported_ps2_texture(pcd_bytes)
                except Exception:
                    ps2_decoded = None
                if ps2_decoded is not None and ps2_decoded.get('supported', False):
                    result = (int(ps2_decoded['width']), int(ps2_decoded['height']))
                    self._texture_dimension_cache[cache_key] = result
                    return result

            psp_dimensions = _psp_pcd_dimensions(pcd_bytes)
            if psp_dimensions is not None:
                result = psp_dimensions
            else:
                try:
                    ps2_decoded = decode_supported_ps2_texture(pcd_bytes)
                except Exception:
                    ps2_decoded = None
                if ps2_decoded is not None and ps2_decoded.get('supported', False):
                    result = (int(ps2_decoded['width']), int(ps2_decoded['height']))
                else:
                    ps3_decoded = decode_supported_ps3_texture(pcd_bytes)
                    if ps3_decoded is not None and ps3_decoded.get('supported', False):
                        result = (int(ps3_decoded['width']), int(ps3_decoded['height']))
                    else:
                        gamecube_decoded = decode_supported_gamecube_texture(pcd_bytes)
                        if gamecube_decoded is not None and gamecube_decoded.get('supported', False):
                            result = (int(gamecube_decoded['width']), int(gamecube_decoded['height']))
                        else:
                            xbox_decoded = decode_supported_xbox_texture(pcd_bytes)
                            if xbox_decoded is not None and xbox_decoded.get('supported', False):
                                result = (int(xbox_decoded['width']), int(xbox_decoded['height']))
                            else:
                                try:
                                    decoded = _decode_pcd_bytes(pcd_bytes)
                                    result = (int(decoded['width']), int(decoded['height']))
                                except Exception:
                                    result = None
        except Exception as exc:
            logger.debug('Failed reading texture dimensions for texture id %d: %s', int(texture_id), exc)
            result = None

        self._texture_dimension_cache[cache_key] = result
        return result

    def _load_packed_image_from_pcd(self, texture_id: int, platform_hint: str | None = None) -> Optional[bpy.types.Image]:
        if not self._texture_import_enabled():
            return None

        texture_path = self._find_texture_path(texture_id)
        if texture_path is None:
            return None

        cache_key = str(texture_path)
        cached = self._loaded_texture_cache.get(cache_key)
        if cached is not None and cached.name in bpy.data.images:
            return cached

        with open(texture_path, 'rb') as fh:
            pcd_bytes = fh.read()

        platform_hint = str(platform_hint or self._active_uv_format or '').lower()

        def _image_from_decoded_texture(decoded: dict) -> bpy.types.Image:
            image_name = texture_path.stem
            decoded_platform = str(decoded.get('platform') or '').lower()

            existing = bpy.data.images.get(image_name)
            if existing is not None:
                try:
                    existing_source = str(existing.get('trlau_source_pcd_path', '') or '')
                except Exception:
                    existing_source = ''
                try:
                    existing_platform = str(existing.get('trlau_texture_platform', '') or '').lower()
                except Exception:
                    existing_platform = ''

                if (
                    existing_source == str(texture_path)
                    and (not decoded_platform or not existing_platform or existing_platform == decoded_platform)
                ):
                    self._loaded_texture_cache[cache_key] = existing
                    return existing

                if (
                    decoded_platform == 'ps2'
                    and existing_source == str(texture_path)
                    and existing_platform and existing_platform != 'ps2'
                    and int(getattr(existing, 'users', 0) or 0) == 0
                ):
                    try:
                        bpy.data.images.remove(existing)
                        existing = None
                    except Exception:
                        pass

            image = bpy.data.images.new(
                image_name,
                width=int(decoded['width']),
                height=int(decoded['height']),
                alpha=bool(decoded.get('has_alpha', True)),
            )
            image.pixels[:] = decoded['pixels']
            try:
                image['trlau_source_pcd_path'] = str(texture_path)
                try:
                    image['trlau_texture_id'] = int(texture_path.stem.rsplit('_', 1)[-1], 16)
                except Exception:
                    pass
                if decoded.get('platform'):
                    image['trlau_texture_platform'] = str(decoded.get('platform'))
                if decoded.get('variant'):
                    image['trlau_texture_variant'] = str(decoded.get('variant'))
            except Exception:
                pass
            image.pack()
            self._loaded_texture_cache[cache_key] = image
            return image

        def _image_from_dds_bytes(dds_bytes: bytes, *, pack_image: bool = True, metadata: dict | None = None) -> bpy.types.Image:
            temp_dir = Path(tempfile.gettempdir()) / "trlau_editor_pcd_import"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dds_path = temp_dir / f"{texture_path.stem}.dds"
            with open(temp_dds_path, 'wb') as fh:
                fh.write(dds_bytes)

            image = bpy.data.images.load(str(temp_dds_path), check_existing=True)
            image.name = texture_path.stem
            try:
                image['trlau_source_pcd_path'] = str(texture_path)
                image['trlau_cached_dds_path'] = str(temp_dds_path)
                if metadata:
                    for key, value in metadata.items():
                        image[str(key)] = value
            except Exception:
                pass

            if pack_image:
                image.pack()
                try:
                    temp_dds_path.unlink(missing_ok=True)
                except Exception:
                    pass

            self._loaded_texture_cache[cache_key] = image
            return image

        def _try_load_psp_texture() -> Optional[bpy.types.Image]:
            try:
                psp_decoded = decode_supported_psp_texture(pcd_bytes)
            except Exception as exc:
                logger.warning('Skipping PSP texture %s: %s', texture_path.name, exc)
                return None

            if psp_decoded is None:
                return None

            if not psp_decoded.get('supported', False):
                logger.warning(
                    'Skipping unsupported PSP texture format %s (%sx%s) for %s',
                    psp_decoded.get('format_name', 'unknown'),
                    int(psp_decoded.get('width', 0)),
                    int(psp_decoded.get('height', 0)),
                    texture_path.name,
                )
                return None

            return _image_from_decoded_texture(psp_decoded)

        def _try_load_ps2_texture() -> Optional[bpy.types.Image]:
            try:
                ps2_decoded = decode_supported_ps2_texture(pcd_bytes)
            except Exception as exc:
                if looks_like_ps2_pcd_bytes(pcd_bytes):
                    logger.warning('Skipping PS2 texture %s: %s', texture_path.name, exc)
                    return None
                return None

            if ps2_decoded is None:
                return None

            if not ps2_decoded.get('supported', False):
                logger.warning(
                    'Skipping unsupported PS2 texture format 0x%X (%sx%s) for %s',
                    int(ps2_decoded.get('format_id', 0)),
                    int(ps2_decoded.get('width', 0)),
                    int(ps2_decoded.get('height', 0)),
                    texture_path.name,
                )
                return None

            return _image_from_decoded_texture(ps2_decoded)

        def _try_load_ps3_texture() -> Optional[bpy.types.Image]:
            try:
                ps3_decoded = decode_supported_ps3_texture(pcd_bytes)
            except Exception as exc:
                logger.warning('Skipping PS3 texture %s: %s', texture_path.name, exc)
                return None

            if ps3_decoded is None:
                return None

            if not ps3_decoded.get('supported', False):
                logger.warning(
                    'Skipping unsupported PS3 texture format %s (%sx%s) for %s',
                    ps3_decoded.get('format_name', 'unknown'),
                    int(ps3_decoded.get('width', 0)),
                    int(ps3_decoded.get('height', 0)),
                    texture_path.name,
                )
                return None

            return _image_from_dds_bytes(
                ps3_decoded['dds_bytes'],
                pack_image=False,
                metadata={
                    'trlau_texture_platform': 'ps3',
                    'trlau_ps3_texture_format_id': int(ps3_decoded.get('format_id', 0)),
                    'trlau_ps3_texture_format_name': str(ps3_decoded.get('format_name', '')),
                    'trlau_ps3_texture_has_alpha': bool(ps3_decoded.get('has_alpha', False)),
                    'trlau_ps3_texture_mipmaps': int(ps3_decoded.get('mipmaps', 1)),
                },
            )

        if platform_hint == 'ps3':
            return _try_load_ps3_texture()
        if platform_hint == 'xbox360':
            try:
                x360_decoded = decode_supported_xbox360_texture(pcd_bytes)
            except Exception as exc:
                logger.warning('Skipping Xbox 360 texture %s: %s', texture_path.name, exc)
                return None
            if x360_decoded is None:
                return None
            if not x360_decoded.get('supported', False):
                logger.warning(
                    'Skipping unsupported Xbox 360 texture format %s (%sx%s) for %s',
                    x360_decoded.get('format_name', 'unknown'),
                    int(x360_decoded.get('width', 0)),
                    int(x360_decoded.get('height', 0)),
                    texture_path.name,
                )
                return None
            return _image_from_dds_bytes(
                x360_decoded['dds_bytes'],
                pack_image=False,
                metadata={
                    'trlau_texture_platform': 'xbox360',
                    'trlau_xbox360_texture_format_id': int(x360_decoded.get('format_id', 0)),
                    'trlau_xbox360_texture_format_name': str(x360_decoded.get('format_name', '')),
                    'trlau_xbox360_texture_has_alpha': bool(x360_decoded.get('has_alpha', False)),
                    'trlau_xbox360_texture_mipmaps': int(x360_decoded.get('mipmaps', 1)),
                    'trlau_xbox360_texture_header_kind': str(x360_decoded.get('header_kind', '')),
                },
            )
        if platform_hint == 'ps2':
            image = _try_load_ps2_texture()
            if image is not None:
                return image
            if looks_like_ps2_pcd_bytes(pcd_bytes):
                return None
        elif platform_hint == 'underworld' and looks_like_ps2_pcd_bytes(pcd_bytes):
            image = _try_load_ps2_texture()
            if image is not None:
                return image
            return None
        elif platform_hint == 'psp':
            image = _try_load_psp_texture()
            if image is not None:
                return image
            return None
        else:
            image = _try_load_psp_texture()
            if image is not None:
                return image

            image = _try_load_ps2_texture()
            if image is not None:
                return image

            if looks_like_ps2_pcd_bytes(pcd_bytes):
                logger.warning('Skipping unsupported PS2 texture %s', texture_path.name)
                return None

        image = _try_load_ps3_texture()
        if image is not None:
            return image

        gamecube_decoded = decode_supported_gamecube_texture(pcd_bytes)
        if gamecube_decoded is not None:
            if not gamecube_decoded.get('supported', False):
                logger.warning(
                    'Skipping unsupported GameCube texture format 0x%X for %s',
                    int(gamecube_decoded.get('format_id', 0)),
                    texture_path.name,
                )
                return None

            image = bpy.data.images.new(
                texture_path.stem,
                width=int(gamecube_decoded['width']),
                height=int(gamecube_decoded['height']),
                alpha=bool(gamecube_decoded.get('has_alpha', True)),
            )
            image.pixels[:] = gamecube_decoded['pixels']
            image.pack()
            self._loaded_texture_cache[cache_key] = image
            return image

        xbox_decoded = decode_supported_xbox_texture(pcd_bytes)
        if xbox_decoded is not None:
            if not xbox_decoded.get('supported', False):
                logger.warning(
                    'Skipping unsupported Xbox texture format 0x%X for %s',
                    int(xbox_decoded.get('format_id', 0)),
                    texture_path.name,
                )
                return None

            if 'pixels' in xbox_decoded:
                image = bpy.data.images.new(
                    texture_path.stem,
                    width=int(xbox_decoded['width']),
                    height=int(xbox_decoded['height']),
                    alpha=bool(xbox_decoded.get('has_alpha', True)),
                )
                image.pixels[:] = xbox_decoded['pixels']
                image.pack()
                self._loaded_texture_cache[cache_key] = image
                return image

            dds_bytes = _create_dds_header(
                int(xbox_decoded['width']),
                int(xbox_decoded['height']),
                int(xbox_decoded.get('mipmaps', 1)),
                int(xbox_decoded['dds_format']),
                len(xbox_decoded['bitmap_data']),
            ) + bytes(xbox_decoded['bitmap_data'])
            return _image_from_dds_bytes(dds_bytes)

        try:
            decoded = _decode_pcd_bytes(pcd_bytes)
        except Exception as exc:
            logger.warning('Skipping unsupported PCD texture %s: %s', texture_path.name, exc)
            return None
        dds_format = int(decoded['dds_format'])

        if dds_format in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8):
            image = bpy.data.images.new(
                texture_path.stem,
                width=decoded['width'],
                height=decoded['height'],
                alpha=(dds_format == D3DFMT_A8R8G8B8),
            )
            image.pixels[:] = _convert_pcd_argb32_to_rgba(
                decoded['bitmap_data'],
                decoded['width'],
                decoded['height'],
                'alpha' if dds_format == D3DFMT_A8R8G8B8 else 'opaque',
            )
            image.pack()
            self._loaded_texture_cache[cache_key] = image
            return image

        return _image_from_dds_bytes(_extract_dds_from_pcd_bytes(pcd_bytes))
