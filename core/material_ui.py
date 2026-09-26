from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import bpy

from ..platforms.nintendo.texture import decode_supported_gamecube_texture
from ..platforms.xbox.texture import decode_supported_xbox_texture
from ..platforms.psp.texture import decode_supported_psp_texture
from ..platforms.ps2.texture import decode_supported_ps2_texture, looks_like_ps2_pcd_bytes
from ..platforms.ps3.texture import decode_supported_ps3_texture
from ..platforms.xbox360.texture import decode_supported_xbox360_texture
from ..builders.mesh_builder import MeshBuilder
from ..platforms.pc.texture import (
    D3DFMT_A8R8G8B8,
    D3DFMT_X8R8G8B8,
    _convert_pcd_argb32_to_rgba,
    _create_dds_header,
    _decode_pcd_bytes,
    _extract_dds_from_pcd_bytes,
)

_MATERIAL_SYNC_GUARD = False


_FLAG_NAMES = (
    'texture_id',
    'blend_value',
    'cull_mode',
    'unknown_1',
    'single_sided',
    'texture_wrap',
    'unknown_2',
    'unknown_3',
    'flat_shading',
    'sort_z',
    'stencil_pass',
    'stencil_func',
    'alpha_ref',
)

_GAME_SCROLL_TO_BLENDER_SCALE = 60.0 / 2048.0
_BLENDER_SCROLL_TO_GAME_SCALE = 2048.0 / 60.0


_PC_NEXTGEN_PANEL_VERSION = 13
_PC_NEXTGEN_LAYER_COUNT = 8
_PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS = 0x00221A00
_PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS_SPECULAR = 0x00225A00
_PC_NEXTGEN_DEFAULT_SHADER_INDICES_LAYERED = (0, 1, 2, 3, 4, 6, 0, 0)
_PC_NEXTGEN_DEFAULT_SHADER_INDICES_DIFFUSE_ONLY = (7, 8, 9, 10, 11, 12, 0, 0)
_PC_NEXTGEN_SHADER_INDEX_LABELS = (
    'MultiPass Light VS 2.0',
    'MultiPass Light VS 3.0',
    'SinglePass Light VS 2.0',
    'SinglePass Light VS 3.0',
    'SinglePass Light PS 2.0',
    'SinglePass Light PS 3.0',
    'SinglePass Light FX PS 2.0',
    'SinglePass Light FX PS 3.0',
)
_PC_NEXTGEN_BLEND_MODE_ITEMS = (
    ('0', 'Opaque', 'kPCBlendModeOpaque'),
    ('1', 'Alpha Test', 'kPCBlendModeAlphaTest'),
    ('2', 'Alpha Blend', 'kPCBlendModeAlphaBlend'),
    ('3', 'Additive', 'kPCBlendModeAdditive'),
    ('4', 'Subtract', 'kPCBlendModeSubtract'),
    ('5', 'Dest Alpha', 'kPCBlendModeDestAlpha'),
    ('6', 'Dest Add', 'kPCBlendModeDestAdd'),
    ('7', 'Modulate', 'kPCBlendModeModulate'),
    ('8', 'Blend 50/50', 'kPCBlendModeBlend5050'),
    ('9', 'Dest Alpha Src Only', 'kPCBlendModeDestAlphaSrcOnly'),
    ('10', 'Color Modulate', 'kPCBlendModeColorModulate'),
    ('13', 'Multipass Alpha', 'kPCBlendModeMultipassAlpha'),
    ('20', 'Light Pass Additive', 'kPCBlendModeLightPassAdditive'),
)
_PC_NEXTGEN_COMBINER_TYPE_ITEMS = (
    ('0', 'Default', 'PC_CT_DEFAULT'),
    ('1', 'Lightmap', 'PC_CT_LIGHTMAP'),
    ('2', 'Reflection', 'PC_CT_REFLECTION'),
    ('3', 'Masked Reflection', 'PC_CT_MASKEDREFLECTION'),
    ('4', 'Stencil Reflection', 'PC_CT_STENCILREFLECTION'),
    ('5', 'Diffuse', 'PC_CT_DIFFUSE'),
    ('6', 'Masked Diffuse', 'PC_CT_MASKEDDIFFUSE'),
    ('7', 'Immediate Draw', 'PC_CT_IMMEDIATEDRAW'),
    ('8', 'Immediate Draw Predator', 'PC_CT_IMMEDIATEDRAW_PREDATOR'),
    ('9', 'Depth of Field', 'PC_CT_DEPTHOFFIELD'),
    ('10', 'Count', 'PC_CT_COUNT'),
)
_PC_NEXTGEN_TEXCOORD_SOURCE_ITEMS = (
    ('0', 'TexCoord 0', 'kTCSTexCoord0'),
    ('1', 'TexCoord 1', 'kTCSTexCoord1'),
    ('2', 'TexCoord 2', 'kTCSTexCoord2'),
    ('3', 'TexCoord 3', 'kTCSTexCoord3'),
    ('4', 'Camera Position', 'kTCSCameraSpacePosition'),
    ('5', 'Camera Normal', 'kTCSCameraSpaceNormal'),
    ('6', 'Camera Reflection', 'kTCSCameraSpaceReflectionVector'),
    ('7', 'World Position', 'kTCSWorldSpacePosition'),
    ('8', 'World Normal', 'kTCSWorldSpaceNormal'),
    ('9', 'World Reflection', 'kTCSWorldSpaceReflectionVector'),
)
_PC_NEXTGEN_TEXCOORD_MODIFIER_ITEMS = (
    ('0', 'None', 'kTCMNone'),
    ('1', 'Scroll', 'kTCMScroll'),
    ('2', 'Auto Scroll', 'kTCMAutoScroll'),
)
_PC_NEXTGEN_PARAM_ID_ITEMS = (
    ('0', 'Constant', 'kMPIConstant'),
    ('1', 'Param 0', 'kMPIParam0'),
    ('2', 'Param 1', 'kMPIParam1'),
    ('3', 'Param 2', 'kMPIParam2'),
    ('4', 'Param 3', 'kMPIParam3'),
    ('5', 'Param 4', 'kMPIParam4'),
    ('6', 'Param 5', 'kMPIParam5'),
    ('7', 'Param 6', 'kMPIParam6'),
    ('8', 'Param 7', 'kMPIParam7'),
)



_FLAG_UI_PROPS = {
    'texture_id': 'trlau_ui_texture_id',
    'blend_value': 'trlau_ui_blend_value',
    'cull_mode': 'trlau_ui_cull_mode',
    'unknown_1': 'trlau_ui_unknown_1',
    'single_sided': 'trlau_ui_single_sided',
    'texture_wrap': 'trlau_ui_texture_wrap',
    'unknown_2': 'trlau_ui_unknown_2',
    'unknown_3': 'trlau_ui_unknown_3',
    'flat_shading': 'trlau_ui_flat_shading',
    'sort_z': 'trlau_ui_sort_z',
    'stencil_pass': 'trlau_ui_stencil_pass',
    'stencil_func': 'trlau_ui_stencil_func',
    'alpha_ref': 'trlau_ui_alpha_ref',
}


def _has_registered_material_panel_props(material) -> bool:
    return material is not None and hasattr(material, 'trlau_ui_texture_id')


def _get_panel_prop(material, name: str, default=0):
    if material is None:
        return default
    try:
        return getattr(material, name)
    except Exception:
        return default


def _set_panel_prop(material, name: str, value) -> None:
    if material is None:
        return
    global _MATERIAL_SYNC_GUARD
    previous_guard = _MATERIAL_SYNC_GUARD
    _MATERIAL_SYNC_GUARD = True
    try:
        setattr(material, name, value)
    except Exception:
        pass
    finally:
        _MATERIAL_SYNC_GUARD = previous_guard


def _pc_nextgen_apply_backface_culling_from_double_sided(material) -> None:
    if material is None:
        return
    try:
        double_sided = bool(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_double_sided', False))
        material.use_backface_culling = not double_sided
    except Exception:
        pass

def set_material_panel_value(material, name: str, value) -> None:
    _set_panel_prop(material, name, value)


_MATERIAL_PLATFORM_VALUES = {'', 'pc', 'pc_nextgen', 'ps2', 'psp', 'ps3', 'xbox360'}
_MATERIAL_TYPE_TO_PLATFORM = {
    'pc_oldgen': 'pc',
    'pc_nextgen': 'pc_nextgen',
    'ps2': 'ps2',
    'psp': 'psp',
    'ps3': 'ps3',
    'xbox360': 'xbox360',
}
_MATERIAL_PLATFORM_TO_TYPE = {
    'pc': 'pc_oldgen',
    'pc_nextgen': 'pc_nextgen',
    'ps2': 'ps2',
    'psp': 'psp',
    'ps3': 'ps3',
    'xbox360': 'xbox360',
}
_MATERIAL_TYPE_VALUES = set(_MATERIAL_TYPE_TO_PLATFORM)


def _material_type_from_platform(platform: str | None) -> str:
    return _MATERIAL_PLATFORM_TO_TYPE.get(str(platform or 'pc').lower(), 'pc_oldgen')


def get_material_type_name(material) -> str:
    if material is None:
        return 'pc_oldgen'
    try:
        value = str(getattr(material, 'trlau_ui_material_type', '') or '').lower()
    except Exception:
        value = ''
    if value in _MATERIAL_TYPE_VALUES:
        return value
    return _material_type_from_platform(get_material_platform_name(material))


def ensure_material_type_panel_prop(material) -> None:
    if material is None:
        return
    target = _material_type_from_platform(get_material_platform_name(material))
    try:
        value = str(getattr(material, 'trlau_ui_material_type', '') or '').lower()
    except Exception:
        value = ''
    if value == target and value in _MATERIAL_TYPE_VALUES:
        return
    _set_panel_prop(material, 'trlau_ui_material_type', target)


def get_material_platform_name(material) -> str:
    if material is None:
        return ''
    try:
        panel_platform = str(getattr(material, 'trlau_ui_material_platform', '') or '').lower()
    except Exception:
        panel_platform = ''
    if panel_platform in {'', 'pc'}:
        try:
            if any(key in material for key in (
                'trlau_pc_nextgen_material_id',
                'trlau_pc_nextgen_layer_texture_ids',
                'trlau_pc_nextgen_diffuse_texture_id',
                'trlau_pc_nextgen_shader_applied',
            )):
                return 'pc_nextgen'
        except Exception:
            pass
    return panel_platform if panel_platform in _MATERIAL_PLATFORM_VALUES else ''

def set_material_platform_name(material, platform: str | None) -> None:
    if material is None:
        return
    platform_name = str(platform or 'pc').lower()
    if platform_name not in {'pc', 'pc_nextgen', 'ps2', 'psp', 'ps3', 'xbox360'}:
        platform_name = 'pc'
    _set_panel_prop(material, 'trlau_ui_material_platform', platform_name)
    try:
        _set_panel_prop(material, 'trlau_ui_material_type', _material_type_from_platform(platform_name))
    except Exception:
        pass

def _get_panel_int(material, name: str, default: int = 0) -> int:
    try:
        return int(_get_panel_prop(material, name, default))
    except Exception:
        return int(default)


def _get_panel_bool(material, name: str, default: bool = False) -> bool:
    try:
        return bool(_get_panel_prop(material, name, default))
    except Exception:
        return bool(default)


def _flags_from_panel_props(material) -> Dict[str, int]:
    ps2_double_sided = 1 if bool(_get_panel_prop(material, 'trlau_ui_ps2_double_sided', False)) else 0
    single_sided = 1 if bool(_get_panel_prop(material, 'trlau_ui_single_sided', False)) else 0
    if _normalise_platform_name(get_material_platform_name(material)) == 'ps2':
        single_sided = 0 if ps2_double_sided else 1
    return {
        'texture_id': int(_get_panel_prop(material, 'trlau_ui_texture_id', 0)) & _texture_id_mask_for_material(material),
        'blend_value': int(_get_panel_prop(material, 'trlau_ui_blend_value', 0)) & 0xF,
        'cull_mode': int(_get_panel_prop(material, 'trlau_ui_cull_mode', 0)) & 0x7,
        'unknown_1': 1 if bool(_get_panel_prop(material, 'trlau_ui_unknown_1', False)) else 0,
        'single_sided': single_sided,
        'texture_wrap': int(_get_panel_prop(material, 'trlau_ui_texture_wrap', 0)) & 0x3,
        'unknown_2': 1 if bool(_get_panel_prop(material, 'trlau_ui_unknown_2', False)) else 0,
        'unknown_3': 1 if bool(_get_panel_prop(material, 'trlau_ui_unknown_3', False)) else 0,
        'flat_shading': 1 if bool(_get_panel_prop(material, 'trlau_ui_flat_shading', False)) else 0,
        'sort_z': 1 if bool(_get_panel_prop(material, 'trlau_ui_sort_z', False)) else 0,
        'stencil_pass': ps2_double_sided if _normalise_platform_name(get_material_platform_name(material)) == 'ps2' else (int(_get_panel_prop(material, 'trlau_ui_stencil_pass', 0)) & 0x3),
        'stencil_func': 1 if bool(_get_panel_prop(material, 'trlau_ui_stencil_func', False)) else 0,
        'alpha_ref': 1 if bool(_get_panel_prop(material, 'trlau_ui_alpha_ref', False)) else 0,
    }


def _write_panel_flags(material, flags: Dict[str, int]) -> None:
    _set_panel_prop(material, 'trlau_ui_texture_id', int(flags.get('texture_id', 0)) & _texture_id_mask_for_material(material))
    _set_panel_prop(material, 'trlau_ui_blend_value', int(flags.get('blend_value', 0)) & 0xF)
    _set_panel_prop(material, 'trlau_ui_cull_mode', int(flags.get('cull_mode', 0)) & 0x7)
    _set_panel_prop(material, 'trlau_ui_unknown_1', bool(int(flags.get('unknown_1', 0))))
    _set_panel_prop(material, 'trlau_ui_single_sided', bool(int(flags.get('single_sided', 0))))
    _set_panel_prop(material, 'trlau_ui_texture_wrap', int(flags.get('texture_wrap', 0)) & 0x3)
    _set_panel_prop(material, 'trlau_ui_unknown_2', bool(int(flags.get('unknown_2', 0))))
    _set_panel_prop(material, 'trlau_ui_unknown_3', bool(int(flags.get('unknown_3', 0))))
    _set_panel_prop(material, 'trlau_ui_flat_shading', bool(int(flags.get('flat_shading', 0))))
    _set_panel_prop(material, 'trlau_ui_sort_z', bool(int(flags.get('sort_z', 0))))
    _set_panel_prop(material, 'trlau_ui_stencil_pass', int(flags.get('stencil_pass', 0)) & 0x3)
    if _normalise_platform_name(get_material_platform_name(material)) == 'ps2':
        _set_panel_prop(material, 'trlau_ui_ps2_double_sided', bool(int(flags.get('stencil_pass', 0)) & 0x1))
        _set_panel_prop(material, 'trlau_ui_single_sided', not bool(int(flags.get('stencil_pass', 0)) & 0x1))
    _set_panel_prop(material, 'trlau_ui_stencil_func', bool(int(flags.get('stencil_func', 0))))
    _set_panel_prop(material, 'trlau_ui_alpha_ref', bool(int(flags.get('alpha_ref', 0))))


def _as_u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _texture_id_mask_for_platform(material_platform: str | None) -> int:
    return 0xFFFF if _normalise_platform_name(material_platform) == 'ps2' else 0x1FFF


def _texture_id_mask_for_material(material) -> int:
    return _texture_id_mask_for_platform(get_material_platform_name(material))


def get_material_tpageid(material) -> int:
    if _has_registered_material_panel_props(material):
        flags = _flags_from_panel_props(material)
        try:
            if _is_pc_nextgen_material(material):
                flags = _oldgen_tpage_flags_from_pc_nextgen_material(material, flags)
            elif get_material_platform_name(material) in {'ps3', 'xbox360'}:
                flags = dict(flags)
                flags['texture_id'] = _material_primary_texture_id(material) & 0x1FFF
        except Exception:
            pass
        return encode_tpage_flags_for_platform(flags, _material_platform_name(material))
    return 0



def _oldgen_tpage_flags_from_pc_nextgen_material(material, base_flags: Dict[str, int]) -> Dict[str, int]:
    flags = dict(base_flags or {})
    try:
        flags['texture_id'] = int(_material_primary_texture_id(material)) & 0x1FFF
    except Exception:
        pass
    try:
        blend_mode = int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_blend_mode', 0) or 0)
    except Exception:
        blend_mode = 0
    try:
        pc_flags = int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_material_flags', 0) or 0)
    except Exception:
        pc_flags = 0
    render_flags = (int(pc_flags) >> 8) & 0x7FF
    if blend_mode == 1 or bool(render_flags & 0x200):
        flags['alpha_ref'] = 1
    if blend_mode != 0 or bool(render_flags & 0x200):
        flags['blend_value'] = int(flags.get('blend_value', 0) or 1)
    try:
        if bool(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_double_sided', False)):
            flags['single_sided'] = 0
    except Exception:
        pass
    return flags

def set_material_tpageid(material, value: int) -> None:
    _write_panel_flags(material, decode_tpage_flags_for_platform(value, _material_platform_name(material)))


def make_material_name_from_tpageid(tpageid: int, strip_index: int | None = None, name_suffix: str | None = None) -> str:
    tpageid_u32 = _as_u32(tpageid)
    base_name = f'Material_{tpageid_u32}'
    if strip_index is not None and int(strip_index) >= 0:
        base_name = f'Material_{int(strip_index)}_{tpageid_u32}'
    if name_suffix:
        return f'{base_name}_{name_suffix}'
    return base_name


def find_material_image(material):
    if not material.use_nodes or material.node_tree is None:
        return None
    for node in material.node_tree.nodes:
        if node.bl_idname == 'ShaderNodeTexImage' and getattr(node, 'image', None) is not None:
            return node.image
    return None


def rename_material_from_tpageid(material) -> None:
    strip_index = None

    material.name = make_material_name_from_tpageid(get_material_tpageid(material), strip_index)


def decode_tpage_flags(tpageid: int) -> Dict[str, int]:
    tpageid = _as_u32(tpageid)
    return {
        'texture_id': tpageid & 0x1FFF,
        'blend_value': (tpageid >> 13) & 0xF,
        'cull_mode': (tpageid >> 17) & 0x7,
        'unknown_1': (tpageid >> 20) & 0x1,
        'single_sided': (tpageid >> 21) & 0x1,
        'texture_wrap': (tpageid >> 22) & 0x3,
        'unknown_2': (tpageid >> 24) & 0x1,
        'unknown_3': (tpageid >> 25) & 0x1,
        'flat_shading': (tpageid >> 26) & 0x1,
        'sort_z': (tpageid >> 27) & 0x1,
        'stencil_pass': (tpageid >> 28) & 0x3,
        'stencil_func': (tpageid >> 30) & 0x1,
        'alpha_ref': (tpageid >> 31) & 0x1,
    }


_CONSOLE_RENDER_PLATFORMS = {'ps3', 'xbox360'}


def _normalise_platform_name(material_platform: str | None) -> str:
    try:
        return str(material_platform or '').lower()
    except Exception:
        return ''


def _uses_console_render_flags(material_platform: str | None) -> bool:
    return _normalise_platform_name(material_platform) in _CONSOLE_RENDER_PLATFORMS


def _apply_console_cull_single_sided_rule(flags: Dict[str, int], material_platform: str = 'ps3') -> Dict[str, int]:
    adjusted = dict(flags)
    if _uses_console_render_flags(material_platform):
        console_single_sided_value = int(adjusted.get('cull_mode', 0)) & 0x7
        adjusted['single_sided'] = 1 if console_single_sided_value == 2 else 0
        adjusted['cull_mode'] = 0
    return adjusted


def decode_tpage_flags_for_platform(tpageid: int, material_platform: str | None = None) -> Dict[str, int]:
    return _apply_console_cull_single_sided_rule(decode_tpage_flags(tpageid), material_platform)


def encode_tpage_flags_for_platform(flags: Dict[str, int], material_platform: str | None = None) -> int:
    encoded_flags = dict(flags)
    if _uses_console_render_flags(material_platform):
        encoded_flags['cull_mode'] = 2 if int(encoded_flags.get('single_sided', 0)) else 0
        encoded_flags['single_sided'] = 0
    return encode_tpage_flags(encoded_flags)


def _material_platform_name(material) -> str:
    return get_material_platform_name(material)


def encode_tpage_flags(flags: Dict[str, int]) -> int:
    value = 0
    value |= int(flags.get('texture_id', 0)) & 0x1FFF
    value |= (int(flags.get('blend_value', 0)) & 0xF) << 13
    value |= (int(flags.get('cull_mode', 0)) & 0x7) << 17
    value |= (int(flags.get('unknown_1', 0)) & 0x1) << 20
    value |= (int(flags.get('single_sided', 0)) & 0x1) << 21
    value |= (int(flags.get('texture_wrap', 0)) & 0x3) << 22
    value |= (int(flags.get('unknown_2', 0)) & 0x1) << 24
    value |= (int(flags.get('unknown_3', 0)) & 0x1) << 25
    value |= (int(flags.get('flat_shading', 0)) & 0x1) << 26
    value |= (int(flags.get('sort_z', 0)) & 0x1) << 27
    value |= (int(flags.get('stencil_pass', 0)) & 0x3) << 28
    value |= (int(flags.get('stencil_func', 0)) & 0x1) << 30
    value |= (int(flags.get('alpha_ref', 0)) & 0x1) << 31
    return _as_u32(value)


def initialize_material_flag_properties(material, tpageid: int) -> None:
    global _MATERIAL_SYNC_GUARD
    previous_guard = _MATERIAL_SYNC_GUARD
    _MATERIAL_SYNC_GUARD = True
    try:
        set_material_tpageid(material, tpageid)
    finally:
        _MATERIAL_SYNC_GUARD = previous_guard
    rename_material_from_tpageid(material)


def _is_helper_material(material) -> bool:
    if material is None:
        return False
    try:
        if bool(material.get('trlau_signal_material', False)):
            return True
    except Exception:
        pass
    try:
        if 'trlau_collision_client_flag' in material:
            return True
    except Exception:
        pass
    return False


def _is_trlau_material(material) -> bool:
    return material is not None and not _is_helper_material(material)


_STALE_MATERIAL_METADATA_KEYS = (
    'trlau_texture_width',
    'trlau_texture_height',
    'trlau_texture_path',
    'trlau_texture_image_name',
    'trlau_texture_strip_index',
    'trlau_material_group',
    'trlau_source_dir',
    'trlau_assigned_texture_image_name',
    'trlau_assigned_texture_id',
    'trlau_material_name_index',
    'trlau_material_name_texture_id',
    'trlau_material_name_suffix',

    'trlau_ui_pcng_material_id',
    'trlau_ui_pcng_asset_id_hi',
    'trlau_ui_pcng_asset_id_lo',
    'trlau_ui_pcng_local_num_pixmaps',

    'trlau_reflective',
    'trlau_env_mapping_value',
    'trlau_sort_push',
    'trlau_scroll_offset',
    'trlau_scroll_entry_count',
    'trlau_scroll_num_tiles',
    'trlau_scroll_tile',
    'trlau_anim_u_speed',
    'trlau_anim_v_speed',
    'trlau_anim_loop_seconds',
    'trlau_anim_loop_tiles_u',
    'trlau_anim_loop_tiles_v',
    'trlau_game_scroll_speed',
    'trlau_level_strip_flags',
    'trlau_texture_id',
    'trlau_blend_value',
    'trlau_cull_mode',
    'trlau_unknown_1',
    'trlau_single_sided',
    'trlau_texture_wrap',
    'trlau_unknown_2',
    'trlau_unknown_3',
    'trlau_flat_shading',
    'trlau_sort_z',
    'trlau_stencil_pass',
    'trlau_stencil_func',
    'trlau_alpha_ref',
    'trlau_draw_group',
    'trlau_animated_material',
    'trlau_use_vertex_colors',
    'trlau_env_mapping',
    'trlau_eye_ref_env_mapping',
    'trlau_scroll_speed',
    'trlau_vertex_color_attribute_name',

    'trlau_ps3_normal_candidate_texture_id',
    'trlau_ps3_external_render_stream',
    'trlau_ps3_texture_stage_ids',
    'trlau_ps3_texture_stage_tpageids',
    'trlau_ps3_has_normal_map',
    'trlau_ps3_has_specular_map',
    'trlau_ps3_role_mapping_version',
    'trlau_ps3_normal_shader_mode',
    'trlau_ps3_normal_green_channel_inverted',
    'trlau_ps2_tpageid_raw',
)


def cleanup_material_metadata(material) -> None:
    if material is None:
        return
    for key in _STALE_MATERIAL_METADATA_KEYS:
        try:
            if key in material:
                del material[key]
        except Exception:
            pass
    try:
        keys = [str(key) for key in material.keys()]
    except Exception:
        keys = []
    for key in keys:
        try:
            if key.startswith('trlau_ui_pcng_layer') and key.endswith('_num_textures'):
                del material[key]
        except Exception:
            pass

def cleanup_all_material_metadata() -> None:
    try:
        materials = list(bpy.data.materials)
    except Exception:
        materials = []
    for material in materials:
        cleanup_material_metadata(material)



def get_material_draw_group(material) -> int:
    try:
        return max(-32768, min(int(_get_panel_prop(material, 'trlau_ui_draw_group', 0)), 32767))
    except Exception:
        try:
            return max(-32768, min(int(material.get('trlau_draw_group', 0)), 32767))
        except Exception:
            return 0


def get_material_scroll_enabled(material) -> bool:
    try:
        return bool(_get_panel_prop(material, 'trlau_ui_scroll_enabled', False))
    except Exception:
        return False


def get_material_scroll_speed(material) -> float:
    try:
        return float(_get_panel_prop(material, 'trlau_ui_scroll_speed', 0.0))
    except Exception:
        return 0.0


def get_material_env_mapping(material) -> bool:
    return bool(_get_panel_prop(material, 'trlau_ui_env_mapping', False))


def get_material_eye_ref_env_mapping(material) -> bool:
    return bool(_get_panel_prop(material, 'trlau_ui_eye_ref_env_mapping', False))


def material_uses_vertex_colors(material) -> bool:
    return _get_material_vertex_color_assignment_name(material) is not None


def _is_psp_material(material) -> bool:
    if material is None:
        return False
    if get_material_platform_name(material) == 'psp':
        return True
    return any(key in material for key in (
        'trlau_psp_texture_id',
        'trlau_psp_blend',
        'trlau_psp_mode_word',
        'trlau_psp_vertex_format_flags',
    ))




def _is_ps3_material(material) -> bool:
    return material is not None and get_material_platform_name(material) in {'ps3', 'xbox360'}

def _get_material_flags(material) -> Dict[str, int]:
    return decode_tpage_flags_for_platform(get_material_tpageid(material), _material_platform_name(material))


def _store_material_flags(material, flags: Dict[str, int]) -> None:
    _write_panel_flags(material, flags)
    if _is_psp_material(material):
        texture_id = int(flags.get('texture_id', 0)) & 0x1FFF
        blend_value = int(flags.get('blend_value', 0)) & 0xF
        material['trlau_psp_texture_id'] = texture_id
        material['trlau_psp_blend'] = blend_value
        material['trlau_psp_blend_low_nibble'] = blend_value
    rename_material_from_tpageid(material)


def _set_psp_alpha_blend_hint(material) -> None:
    try:
        psp_blend = int(material.get('trlau_psp_blend', 0)) & 0xFF
    except Exception:
        psp_blend = 0
    try:
        psp_render_flags = int(material.get('trlau_psp_render_flags', 0)) & 0xFF
    except Exception:
        psp_render_flags = 0
    try:
        psp_vertex_format_flags = int(material.get('trlau_psp_vertex_format_flags', 0)) & 0xFFFFFFFF
    except Exception:
        psp_vertex_format_flags = 0
    material['trlau_psp_alpha_blend_hint'] = bool(psp_blend != 0 or psp_render_flags != 0 or (psp_vertex_format_flags & 0x40))


def _sync_psp_mode_word(material) -> None:
    try:
        vertex_mode = int(material.get('trlau_psp_vertex_mode', int(material.get('trlau_psp_mode_word', 0)) & 0xFF)) & 0xFF
    except Exception:
        vertex_mode = 0
    try:
        render_flags = int(material.get('trlau_psp_render_flags', (int(material.get('trlau_psp_mode_word', 0)) >> 8) & 0xFF)) & 0xFF
    except Exception:
        render_flags = 0
    material['trlau_psp_vertex_mode'] = vertex_mode
    material['trlau_psp_render_flags'] = render_flags
    material['trlau_psp_mode_word'] = vertex_mode | (render_flags << 8)


def _sync_psp_vertex_format_flags(material) -> None:
    try:
        flags = int(material.get('trlau_psp_vertex_format_flags', 0)) & 0xFFFFFFFF
    except Exception:
        flags = 0
    material['trlau_psp_vertex_format_flags'] = flags
    material['trlau_psp_envmap_flag'] = bool(flags & 0x400)
    material['trlau_psp_unknown_800_flag'] = bool(flags & 0x800)
    try:
        material.trlau_ui_env_mapping = bool(flags & 0x400)
    except Exception:
        pass
    _set_psp_alpha_blend_hint(material)


def _find_existing_image(material) -> Optional[bpy.types.Image]:
    if not material.use_nodes or material.node_tree is None:
        return None
    try:
        for node in material.node_tree.nodes:
            if node.bl_idname == 'ShaderNodeTexImage' and getattr(node, 'image', None) is not None:
                return node.image
    except Exception:
        return None
    return None



def _is_texture_id_assignment_enabled() -> bool:
    context = getattr(bpy, 'context', None)
    scene = getattr(context, 'scene', None) if context is not None else None
    if scene is None:
        return False
    try:
        return bool(getattr(scene, 'trlau_texture_id_assignment', False))
    except Exception:
        return False


def _normalize_vertex_color_attribute_name(name: object) -> Optional[str]:
    value = str(name or '').strip()
    return value or None



def _get_mesh_vertex_color_name(mesh) -> Optional[str]:
    if mesh is None:
        return None

    color_attributes = getattr(mesh, 'color_attributes', None)
    if color_attributes is not None:
        try:
            active_color = getattr(color_attributes, 'active_color', None)
            active_name = _normalize_vertex_color_attribute_name(getattr(active_color, 'name', None))
            if active_name is not None:
                return active_name
        except Exception:
            pass

        for index_attr in ('render_color_index', 'active_color_index'):
            try:
                color_index = int(getattr(color_attributes, index_attr, -1))
                if color_index >= 0 and color_index < len(color_attributes):
                    attr_name = _normalize_vertex_color_attribute_name(getattr(color_attributes[color_index], 'name', None))
                    if attr_name is not None:
                        return attr_name
            except Exception:
                pass

        try:
            for color_attr in color_attributes:
                attr_name = _normalize_vertex_color_attribute_name(getattr(color_attr, 'name', None))
                if attr_name is not None:
                    return attr_name
        except Exception:
            pass

    vertex_color_layers = getattr(mesh, 'vertex_colors', None)
    if vertex_color_layers is not None:
        for attr_name in ('active', 'active_index'):
            try:
                active_value = getattr(vertex_color_layers, attr_name, None)
                if attr_name == 'active_index':
                    color_index = int(active_value)
                    if color_index >= 0 and color_index < len(vertex_color_layers):
                        layer_name = _normalize_vertex_color_attribute_name(getattr(vertex_color_layers[color_index], 'name', None))
                        if layer_name is not None:
                            return layer_name
                else:
                    layer_name = _normalize_vertex_color_attribute_name(getattr(active_value, 'name', None))
                    if layer_name is not None:
                        return layer_name
            except Exception:
                pass

        try:
            for color_layer in vertex_color_layers:
                layer_name = _normalize_vertex_color_attribute_name(getattr(color_layer, 'name', None))
                if layer_name is not None:
                    return layer_name
        except Exception:
            pass

    return None



def _get_material_vertex_color_assignment_name(material) -> Optional[str]:
    node_tree = getattr(material, 'node_tree', None)
    if getattr(material, 'use_nodes', False) and node_tree is not None:
        try:
            unnamed_assignment_found = False
            for node in node_tree.nodes:
                bl_idname = getattr(node, 'bl_idname', '')
                if bl_idname == 'ShaderNodeVertexColor':
                    layer_name = _normalize_vertex_color_attribute_name(getattr(node, 'layer_name', None))
                    if layer_name is not None:
                        return layer_name
                    unnamed_assignment_found = True
                elif bl_idname == 'ShaderNodeAttribute':
                    attribute_name = _normalize_vertex_color_attribute_name(getattr(node, 'attribute_name', None))
                    if attribute_name is not None:
                        return attribute_name
                    unnamed_assignment_found = True
            if unnamed_assignment_found:
                return 'Color'
        except Exception:
            pass



    return None



def _material_has_vertex_color_assignment(material) -> bool:
    return _get_material_vertex_color_assignment_name(material) is not None


def _assign_current_material_nodes(material, texture_id: int | None = None) -> None:
    if texture_id is not None:
        _assign_current_image_to_texture_id(material, texture_id)

    vertex_color_name = _get_material_vertex_color_assignment_name(material)
    # ?


def _get_material_assigned_texture_id(material) -> Optional[int]:
    try:
        if 'trlau_assigned_texture_id' not in material:
            return None
        return int(material.get('trlau_assigned_texture_id', 0)) & 0x1FFF
    except Exception:
        return None


def _get_material_assigned_image(material, texture_id: int) -> Optional[bpy.types.Image]:
    target_id = int(texture_id) & 0x1FFF
    assigned_id = _get_material_assigned_texture_id(material)
    if assigned_id != target_id:
        return None

    assigned_name = str(material.get('trlau_assigned_texture_image_name', '') or '').strip()
    if not assigned_name:
        return None

    try:
        return bpy.data.images.get(assigned_name)
    except Exception:
        return None


def _assign_current_image_to_texture_id(material, texture_id: int) -> None:
    # Do not store material-side texture assignment metadata.  Texture binding
    # is represented by the material's image node and the Texture ID field.
    cleanup_material_metadata(material)


def _strip_trailing_blender_copy_suffix(text: str) -> str:
    value = str(text or '').strip()
    while True:
        root, ext = os.path.splitext(value)
        if ext.startswith('.') and ext[1:].isdigit():
            value = root
            continue
        return value


def _iter_texture_id_name_candidates(name: str):
    value = _strip_trailing_blender_copy_suffix(str(name or '').strip())
    if not value:
        return

    candidates = [value]
    basename = Path(value).name
    if basename and basename not in candidates:
        candidates.append(basename)

    stem = Path(basename).stem
    if stem and stem not in candidates:
        candidates.append(stem)

    for candidate in tuple(candidates):
        normalized = candidate.lower()
        yield normalized
        if '_' in normalized:
            yield normalized.rsplit('_', 1)[-1]
        if '-' in normalized:
            yield normalized.rsplit('-', 1)[-1]


def _find_loaded_image_by_texture_id(texture_id: int) -> Optional[bpy.types.Image]:
    target_id = int(texture_id) & 0x1FFF
    try:
        exact_matches = []
        suffix_matches = []
        for image in bpy.data.images:
            image_candidates = [str(getattr(image, 'name', '') or '')]
            raw_path = str(getattr(image, 'filepath_raw', '') or getattr(image, 'filepath', '') or '')
            if raw_path:
                try:
                    image_candidates.append(Path(bpy.path.abspath(raw_path)).name)
                except Exception:
                    image_candidates.append(Path(raw_path).name)

            matched = False
            explicit_id = None
            try:
                if 'trlau_texture_id' in image:
                    explicit_id = int(image.get('trlau_texture_id', 0)) & 0x1FFF
            except Exception:
                explicit_id = None
            if explicit_id == target_id:
                exact_matches.append(image)
                continue

            for candidate in image_candidates:
                parsed_id = _parse_texture_id_from_name(candidate)
                if parsed_id != target_id:
                    continue
                normalized = _strip_trailing_blender_copy_suffix(str(candidate or '').strip()).lower()
                stem = Path(normalized).stem
                if normalized in {f'{target_id:x}', f'0x{target_id:x}'} or stem in {f'{target_id:x}', f'0x{target_id:x}'}:
                    exact_matches.append(image)
                else:
                    suffix_matches.append(image)
                matched = True
                break
            if matched:
                continue
        if exact_matches:
            return sorted(exact_matches, key=lambda item: item.name)[0]
        if suffix_matches:
            return sorted(suffix_matches, key=lambda item: item.name)[0]
    except Exception:
        return None
    return None


def _parse_texture_id_from_name(name: str) -> Optional[int]:
    for candidate in _iter_texture_id_name_candidates(name):
        if re.fullmatch(r'0x[0-9a-f]+', candidate):
            try:
                return int(candidate, 16) & 0x1FFF
            except Exception:
                continue
        if re.fullmatch(r'[0-9a-f]+', candidate):
            try:
                return int(candidate, 16) & 0x1FFF
            except Exception:
                continue
    return None


def _iter_candidate_texture_directories(material=None, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None):
    seen: set[str] = set()

    def _add(path_value):
        if not path_value:
            return
        try:
            path = Path(path_value)
        except Exception:
            return
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved)
        if key in seen or not resolved.exists() or not resolved.is_dir():
            return
        seen.add(key)
        yield resolved

    for value in extra_dirs or ():
        yield from _add(value)

    if material is not None:
        image = _find_existing_image(material)
        if image is not None:
            for attr_name in ('trlau_source_pcd_path', 'filepath_raw', 'filepath'):
                try:
                    raw_path = str(image.get(attr_name, '') if attr_name == 'trlau_source_pcd_path' else getattr(image, attr_name, '') or '')
                except Exception:
                    raw_path = ''
                if raw_path:
                    yield from _add(Path(raw_path).parent)

    if owner is not None:
        raw_source_dir = str(owner.get('trlau_source_dir', '') or '')
        if raw_source_dir:
            yield from _add(raw_source_dir)


_SUPPORTED_TEXTURE_EXTENSIONS = ('.pcd', '.dds', '.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff')


def _find_texture_path_from_directories(texture_id: int, directories: Iterable[Path]) -> Optional[Path]:
    if int(texture_id) == 0:
        return None

    target_id = int(texture_id) & 0x1FFF
    for directory in directories:
        try:
            for entry in sorted(directory.rglob('*')):
                if not entry.is_file() or entry.suffix.lower() not in _SUPPORTED_TEXTURE_EXTENSIONS:
                    continue
                if _parse_texture_id_from_name(entry.name) == target_id:
                    return entry
        except Exception:
            continue
    return None


def find_texture_path_for_material(material, texture_id: int, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> Optional[Path]:
    path_from_metadata = _find_texture_path_from_metadata(material, texture_id)
    if path_from_metadata is not None:
        return path_from_metadata
    return _find_texture_path_from_directories(texture_id, _iter_candidate_texture_directories(material, owner, extra_dirs))


def _find_texture_path_from_metadata(material, texture_id: int) -> Optional[Path]:
    image = _find_existing_image(material)
    if image is None:
        return None
    target_id = int(texture_id) & 0x1FFF
    for attr_name in ('trlau_source_pcd_path', 'filepath_raw', 'filepath'):
        try:
            raw_path = str(image.get(attr_name, '') if attr_name == 'trlau_source_pcd_path' else getattr(image, attr_name, '') or '')
        except Exception:
            raw_path = ''
        if not raw_path:
            continue
        current_path = Path(raw_path)
        model_dir = current_path.parent
        try:
            for entry in sorted(model_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in _SUPPORTED_TEXTURE_EXTENSIONS and _parse_texture_id_from_name(entry.name) == target_id:
                    return entry
        except Exception:
            pass
    return None


def _load_image_for_texture_id(material, texture_id: int, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> Optional[bpy.types.Image]:
    if int(texture_id) == 0:
        return None

    assigned_image = _get_material_assigned_image(material, texture_id)
    if assigned_image is not None:
        return assigned_image

    existing_image = _find_existing_image(material)
    if existing_image is not None:
        try:
            if 'trlau_texture_id' in existing_image and (int(existing_image.get('trlau_texture_id', 0)) & 0x1FFF) == (int(texture_id) & 0x1FFF):
                return existing_image
        except Exception:
            pass
        if _parse_texture_id_from_name(str(getattr(existing_image, 'name', '') or '')) == (int(texture_id) & 0x1FFF):
            return existing_image
        raw_path = str(getattr(existing_image, 'filepath_raw', '') or getattr(existing_image, 'filepath', '') or '')
        if raw_path and _parse_texture_id_from_name(Path(raw_path).name) == (int(texture_id) & 0x1FFF):
            return existing_image

    loaded_image = _find_loaded_image_by_texture_id(texture_id)
    if loaded_image is not None:
        return loaded_image

    texture_path = find_texture_path_for_material(material, texture_id, owner=owner, extra_dirs=extra_dirs)
    if texture_path is None:
        return None

    if texture_path.suffix.lower() != '.pcd':
        try:
            image = bpy.data.images.load(str(texture_path), check_existing=True)
            return image
        except Exception:
            return None

    try:
        with open(texture_path, 'rb') as fh:
            pcd_bytes = fh.read()
    except Exception:
        return None

    platform_name = get_material_platform_name(material)

    if platform_name == 'ps2':
        try:
            ps2_decoded = decode_supported_ps2_texture(pcd_bytes)
        except Exception:
            ps2_decoded = None
        if ps2_decoded is not None:
            if not ps2_decoded.get('supported', False):
                return None
            try:
                image_name = texture_path.stem
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
                    if existing_source == str(texture_path) and existing_platform in {'', 'ps2'}:
                        return existing
                    if existing_source == str(texture_path) and existing_platform and existing_platform != 'ps2' and int(getattr(existing, 'users', 0) or 0) == 0:
                        try:
                            bpy.data.images.remove(existing)
                        except Exception:
                            pass

                image = bpy.data.images.new(
                    image_name,
                    width=int(ps2_decoded['width']),
                    height=int(ps2_decoded['height']),
                    alpha=bool(ps2_decoded.get('has_alpha', True)),
                )
                image.pixels[:] = ps2_decoded['pixels']
                image['trlau_source_pcd_path'] = str(texture_path)
                image['trlau_texture_platform'] = 'ps2'
                try:
                    image['trlau_texture_id'] = int(texture_path.stem.rsplit('_', 1)[-1], 16)
                except Exception:
                    pass
                if ps2_decoded.get('variant'):
                    image['trlau_texture_variant'] = str(ps2_decoded.get('variant'))
                image.pack()
                return image
            except Exception:
                return None
        if looks_like_ps2_pcd_bytes(pcd_bytes):
            return None

    ps3_decoded = decode_supported_xbox360_texture(pcd_bytes) if platform_name == 'xbox360' else decode_supported_ps3_texture(pcd_bytes)
    if ps3_decoded is not None:
        if not ps3_decoded.get('supported', False):
            return None
        temp_dds_path = None
        try:
            temp_dir = Path(tempfile.gettempdir()) / 'trlau_editor_pcd_import'
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_dds_path = temp_dir / f'{texture_path.stem}.dds'
            with open(temp_dds_path, 'wb') as fh:
                fh.write(ps3_decoded['dds_bytes'])
            image = bpy.data.images.load(str(temp_dds_path), check_existing=True)
            image.name = texture_path.stem
            image['trlau_source_pcd_path'] = str(texture_path)
            image['trlau_cached_dds_path'] = str(temp_dds_path)
            image['trlau_texture_platform'] = 'xbox360' if platform_name == 'xbox360' else 'ps3'
            image['trlau_ps3_texture_format_id'] = int(ps3_decoded.get('format_id', 0))
            image['trlau_ps3_texture_format_name'] = str(ps3_decoded.get('format_name', ''))
            image['trlau_ps3_texture_has_alpha'] = bool(ps3_decoded.get('has_alpha', False))
            image['trlau_ps3_texture_mipmaps'] = int(ps3_decoded.get('mipmaps', 1))
            return image
        except Exception:
            return None

    psp_decoded = decode_supported_psp_texture(pcd_bytes)
    if psp_decoded is not None:
        if not psp_decoded.get('supported', False):
            return None
        try:
            image = bpy.data.images.new(
                texture_path.stem,
                width=int(psp_decoded['width']),
                height=int(psp_decoded['height']),
                alpha=bool(psp_decoded.get('has_alpha', True)),
            )
            image.pixels[:] = psp_decoded['pixels']
            image.pack()
            return image
        except Exception:
            return None

    gamecube_decoded = decode_supported_gamecube_texture(pcd_bytes)
    if gamecube_decoded is not None:
        if not gamecube_decoded.get('supported', False):
            return None
        try:
            image = bpy.data.images.new(
                texture_path.stem,
                width=int(gamecube_decoded['width']),
                height=int(gamecube_decoded['height']),
                alpha=bool(gamecube_decoded.get('has_alpha', True)),
            )
            image.pixels[:] = gamecube_decoded['pixels']
            image.pack()
            return image
        except Exception:
            return None

    xbox_decoded = decode_supported_xbox_texture(pcd_bytes)
    if xbox_decoded is not None:
        if not xbox_decoded.get('supported', False):
            return None
        try:
            if 'pixels' in xbox_decoded:
                image = bpy.data.images.new(
                    texture_path.stem,
                    width=int(xbox_decoded['width']),
                    height=int(xbox_decoded['height']),
                    alpha=bool(xbox_decoded.get('has_alpha', True)),
                )
                image.pixels[:] = xbox_decoded['pixels']
                image.pack()
                return image

            dds_bytes = _create_dds_header(
                int(xbox_decoded['width']),
                int(xbox_decoded['height']),
                int(xbox_decoded.get('mipmaps', 1)),
                int(xbox_decoded['dds_format']),
                len(xbox_decoded['bitmap_data']),
            ) + bytes(xbox_decoded['bitmap_data'])
        except Exception:
            return None

        temp_dir = Path(tempfile.gettempdir()) / 'trlau_editor_pcd_import'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dds_path = temp_dir / f'{texture_path.stem}.dds'
        try:
            with open(temp_dds_path, 'wb') as fh:
                fh.write(dds_bytes)
            image = bpy.data.images.load(str(temp_dds_path), check_existing=True)
            image.name = texture_path.stem
            image.pack()
            return image
        except Exception:
            return None
        finally:
            try:
                temp_dds_path.unlink(missing_ok=True)
            except Exception:
                pass

    try:
        decoded = _decode_pcd_bytes(pcd_bytes)
    except Exception:
        return None

    dds_format = int(decoded['dds_format'])
    if dds_format in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8):
        try:
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
            return image
        except Exception:
            return None

    try:
        dds_bytes = _extract_dds_from_pcd_bytes(pcd_bytes)
    except Exception:
        return None

    temp_dir = Path(tempfile.gettempdir()) / 'trlau_editor_pcd_import'
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dds_path = temp_dir / f'{texture_path.stem}.dds'
    try:
        with open(temp_dds_path, 'wb') as fh:
            fh.write(dds_bytes)
        image = bpy.data.images.load(str(temp_dds_path), check_existing=True)
        image.name = texture_path.stem
        image.pack()
        return image
    except Exception:
        return None
    finally:
        try:
            temp_dds_path.unlink(missing_ok=True)
        except Exception:
            pass



def _get_reflection_mode(material) -> str:
    env = bool(get_material_env_mapping(material))
    eye_ref = bool(get_material_eye_ref_env_mapping(material))
    if env and eye_ref:
        return 'both'
    if eye_ref:
        return 'eye_ref'
    if env:
        return 'env'
    return 'none'


def get_material_reflection_mode(material) -> str:
    return _get_reflection_mode(material)



def _safe_get_int(material, key: str, default: int = -1) -> int:
    try:
        return int(material.get(key, default))
    except Exception:
        return int(default)


def _safe_get_float(material, key: str, default: float) -> float:
    try:
        return float(material.get(key, default))
    except Exception:
        return float(default)


def _clamp_texture_id(value: int, *, allow_disabled: bool = True) -> int:
    try:
        value = int(value)
    except Exception:
        value = -1 if allow_disabled else 0
    lower = -1 if allow_disabled else 0
    return max(lower, min(value, 0x1FFF))


def _set_ps3_texture_role_metadata(image, texture_id: int, role: str) -> None:
    if image is None:
        return
    try:
        image['trlau_texture_id'] = int(texture_id) & 0x1FFF
        image['trlau_ps3_texture_role'] = str(role)
        if role == 'normal' and 'trlau_ps3_treat_as_height_map' not in image:
            image['trlau_ps3_treat_as_height_map'] = False
    except Exception:
        pass


def _is_pc_nextgen_material(material) -> bool:
    if material is None:
        return False
    if get_material_platform_name(material) == 'pc_nextgen':
        return True
    return any(key in material for key in (
        'trlau_pc_nextgen_material_id',
        'trlau_pc_nextgen_layer_texture_ids',
        'trlau_pc_nextgen_diffuse_texture_id',
        'trlau_pc_nextgen_shader_applied',
    ))


def _csv_ints(value) -> list[int]:
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            try:
                result.append(int(item))
            except Exception:
                result.append(-1)
        return result
    text = str(value or '').strip()
    if not text:
        return []
    result = []
    for part in text.split(','):
        try:
            result.append(int(part.strip()))
        except Exception:
            result.append(-1)
    return result


def _set_pc_nextgen_texture_role_metadata(image, texture_id: int, layer_index: int, role: str) -> None:
    if image is None:
        return
    try:
        image['trlau_texture_id'] = int(texture_id) & 0x1FFF
    except Exception:
        pass


def _pc_nextgen_role_for_layer(layer_index: int) -> str:
    if int(layer_index) == 0:
        return 'diffuse'
    if int(layer_index) == 1:
        return 'normal'
    if int(layer_index) == 2:
        return 'specular'
    return f'layer_{int(layer_index)}'


def _pc_nextgen_get_layer_image_names(material) -> list[str]:
    if material is None:
        return []
    raw = _pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_image_names', '')
    if isinstance(raw, str):
        try:
            value = json.loads(raw) if raw.strip() else []
        except Exception:
            value = []
    elif isinstance(raw, (list, tuple)):
        value = raw
    else:
        value = []
    result: list[str] = []
    for item in value:
        result.append(str(item or ''))
    return result


def _pc_nextgen_store_layer_image_names(material, layer_images) -> None:
    if material is None:
        return
    names: list[str] = []
    for image in layer_images or []:
        try:
            names.append(str(getattr(image, 'name', '') or '') if image is not None else '')
        except Exception:
            names.append('')
    while len(names) < _PC_NEXTGEN_LAYER_COUNT:
        names.append('')
    try:
        _set_panel_prop(material, 'trlau_ui_pcng_layer_image_names', json.dumps(names[:_PC_NEXTGEN_LAYER_COUNT]))
    except Exception:
        pass


def _pc_nextgen_capture_current_layer_images(material) -> list[Optional[bpy.types.Image]]:
    images: list[Optional[bpy.types.Image]] = [None] * _PC_NEXTGEN_LAYER_COUNT
    if material is None:
        return images
    try:
        if not material.use_nodes or material.node_tree is None:
            return images
        for node in material.node_tree.nodes:
            if getattr(node, 'bl_idname', '') != 'ShaderNodeTexImage':
                continue
            image = getattr(node, 'image', None)
            if image is None:
                continue
            layer_index = None
            try:
                if 'trlau_pc_nextgen_layer_index' in node:
                    layer_index = int(node.get('trlau_pc_nextgen_layer_index'))
            except Exception:
                layer_index = None
            if layer_index is None:
                try:
                    if 'trlau_pc_nextgen_layer_index' in image:
                        layer_index = int(image.get('trlau_pc_nextgen_layer_index'))
                except Exception:
                    layer_index = None
            if layer_index is None:
                label = str(getattr(node, 'label', '') or getattr(node, 'name', '') or '')
                match = re.search(r'(?:Layer|L)\s*([0-7])\b', label, re.IGNORECASE)
                if match:
                    try:
                        layer_index = int(match.group(1))
                    except Exception:
                        layer_index = None
            if layer_index is None or not (0 <= int(layer_index) < _PC_NEXTGEN_LAYER_COUNT):
                continue
            images[int(layer_index)] = image
    except Exception:
        pass
    _pc_nextgen_store_layer_image_names(material, images)
    return images


def _pc_nextgen_image_matches_texture_id(image, texture_id: int) -> bool:
    if image is None or int(texture_id) < 0:
        return False
    target_id = int(texture_id) & 0x1FFF
    try:
        if 'trlau_texture_id' in image and (int(image.get('trlau_texture_id', 0)) & 0x1FFF) == target_id:
            return True
    except Exception:
        pass
    try:
        if _parse_texture_id_from_name(str(getattr(image, 'name', '') or '')) == target_id:
            return True
    except Exception:
        pass
    try:
        raw_path = str(getattr(image, 'filepath_raw', '') or getattr(image, 'filepath', '') or '')
        if raw_path and _parse_texture_id_from_name(Path(raw_path).name) == target_id:
            return True
    except Exception:
        pass
    return False


def _load_pc_nextgen_layer_images(material, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> list[Optional[bpy.types.Image]]:
    layer_texture_ids = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_texture_ids', ''))
    layer_enabled = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_enabled', ''))

    if not layer_texture_ids:
        layer_texture_ids = [
            int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_diffuse_texture_id', -1)),
            int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_normal_texture_id', -1)),
            int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_specular_texture_id', -1)),
        ]
        layer_enabled = [1 if int(value) >= 0 else 0 for value in layer_texture_ids]

    if len(layer_texture_ids) < _PC_NEXTGEN_LAYER_COUNT:
        layer_texture_ids.extend([-1] * (_PC_NEXTGEN_LAYER_COUNT - len(layer_texture_ids)))
    if len(layer_enabled) < len(layer_texture_ids):
        layer_enabled.extend([0] * (len(layer_texture_ids) - len(layer_enabled)))

    captured_images = _pc_nextgen_capture_current_layer_images(material)
    saved_names = _pc_nextgen_get_layer_image_names(material)
    if len(saved_names) < _PC_NEXTGEN_LAYER_COUNT:
        saved_names.extend([''] * (_PC_NEXTGEN_LAYER_COUNT - len(saved_names)))

    images: list[Optional[bpy.types.Image]] = []
    for layer_index, texture_id in enumerate(layer_texture_ids[:_PC_NEXTGEN_LAYER_COUNT]):
        try:
            enabled = int(layer_enabled[layer_index]) != 0
        except Exception:
            enabled = False
        texture_id = _clamp_texture_id(texture_id, allow_disabled=True)
        if not enabled or texture_id < 0:
            images.append(None)
            continue

        image = _load_image_for_texture_id(material, texture_id, owner=owner, extra_dirs=extra_dirs)

        if image is None and layer_index < len(captured_images):
            candidate = captured_images[layer_index]
            if candidate is not None:
                image = candidate

        if image is None and layer_index < len(saved_names) and saved_names[layer_index]:
            try:
                candidate = bpy.data.images.get(saved_names[layer_index])
            except Exception:
                candidate = None
            if candidate is not None:
                image = candidate

        if image is None:
            image = _find_loaded_image_by_texture_id(texture_id)

        role = _pc_nextgen_role_for_layer(layer_index)
        _set_pc_nextgen_texture_role_metadata(image, texture_id, layer_index, role)
        images.append(image)

    _pc_nextgen_store_layer_image_names(material, images)
    return images



def is_pc_nextgen_material(material) -> bool:
    return _is_pc_nextgen_material(material)


def _pc_nextgen_enum_value(value, fallback: int = 0) -> int:
    try:
        return int(str(value), 0)
    except Exception:
        return int(fallback)


def _pc_nextgen_parse_int_text(value, fallback: int = 0) -> int:
    try:
        if isinstance(value, str):
            text = value.strip()
            return int(text, 0) if text else int(fallback)
        return int(value)
    except Exception:
        return int(fallback)


def _pc_nextgen_to_int_text(value, *, hex_format: bool = False) -> str:
    try:
        integer = int(value)
    except Exception:
        integer = 0
    if hex_format:
        return f'0x{integer & 0xFFFFFFFF:08X}'
    return str(integer)


def _pc_nextgen_json_float_tuples(value, fallback=None) -> list[tuple[float, float, float, float]]:
    if fallback is None:
        fallback = []
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except Exception:
            return list(fallback)
    if not isinstance(value, (list, tuple)):
        return list(fallback)
    result: list[tuple[float, float, float, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)):
            continue
        parts = list(item)[:4]
        while len(parts) < 4:
            parts.append(1.0 if len(parts) == 3 else 0.0)
        try:
            result.append(tuple(float(component) for component in parts[:4]))
        except Exception:
            pass
    return result


def _pc_nextgen_json_from_float_tuples(values) -> str:
    try:
        return json.dumps([[float(component) for component in value[:4]] for value in values])
    except Exception:
        return '[]'



def _pc_nextgen_panel_storage_value(material, name: str, default=None):
    if material is None:
        return default
    try:
        if name == 'trlau_pc_nextgen_material_id':
            return int(_get_panel_int(material, 'trlau_ui_draw_group', 0)) + 1
        if name == 'trlau_pc_nextgen_material_record_offset':
            return _get_panel_int(material, 'trlau_ui_pcng_material_record_offset', default if default is not None else 0)
        if name == 'trlau_pc_nextgen_asset_id_hi':
            return 0
        if name == 'trlau_pc_nextgen_asset_id_lo':
            return 0
        if name == 'trlau_pc_nextgen_asset_id_padding_hex':
            return str(_get_panel_prop(material, 'trlau_ui_pcng_asset_id_padding_hex', default or '') or '')
        if name == 'trlau_pc_nextgen_material_record_hex':
            return str(_get_panel_prop(material, 'trlau_ui_pcng_material_record_hex', default or '') or '')
        if name == 'trlau_pc_nextgen_blend_mode':
            return _pc_nextgen_enum_value(_get_panel_prop(material, 'trlau_ui_pcng_blend_mode', '0'), default if default is not None else 0)
        if name == 'trlau_pc_nextgen_combiner_type':
            return _pc_nextgen_enum_value(_get_panel_prop(material, 'trlau_ui_pcng_combiner_type', '0'), default if default is not None else 0)
        if name == 'trlau_pc_nextgen_material_flags':
            return _pc_nextgen_parse_int_text(_get_panel_prop(material, 'trlau_ui_pcng_material_flags', '0'), default if default is not None else 0)
        if name == 'trlau_pc_nextgen_opacity':
            return max(0.0, min(1.0, float(_get_panel_prop(material, 'trlau_ui_pcng_opacity', default if default is not None else 1.0))))
        if name == 'trlau_pc_nextgen_poly_flags':
            return _pc_nextgen_parse_int_text(_get_panel_prop(material, 'trlau_ui_pcng_poly_flags', '0'), default if default is not None else 0)
        if name == 'trlau_pc_nextgen_uv_auto_scroll_speed':
            return _get_panel_int(material, 'trlau_ui_pcng_uv_auto_scroll_speed', default if default is not None else 0) & 0xFFFF
        if name in {
            'trlau_pc_nextgen_sort_bias',
            'trlau_pc_nextgen_detail_range_mul',
            'trlau_pc_nextgen_detail_scale',
            'trlau_pc_nextgen_parallax_scale',
            'trlau_pc_nextgen_parallax_offset',
            'trlau_pc_nextgen_specular_power',
            'trlau_pc_nextgen_specular_shift0',
            'trlau_pc_nextgen_specular_shift1',
            'trlau_pc_nextgen_rim_light_intensity',
            'trlau_pc_nextgen_water_blend_bias',
            'trlau_pc_nextgen_water_blend_exponent',
        }:
            suffix = name[len('trlau_pc_nextgen_'):]
            return float(_get_panel_prop(material, f'trlau_ui_pcng_{suffix}', default if default is not None else 0.0))
        if name == 'trlau_pc_nextgen_rim_light_color':
            return _pc_nextgen_json_from_float_tuples([_pc_nextgen_color_value(material, 'trlau_ui_pcng_rim_light_color', (0.0, 0.0, 0.0, 0.0))])
        if name == 'trlau_pc_nextgen_water_deep_color':
            return _pc_nextgen_json_from_float_tuples([_pc_nextgen_color_value(material, 'trlau_ui_pcng_water_deep_color', (0.0, 0.0, 0.0, 0.0))])
        if name == 'trlau_pc_nextgen_local_num_pixmaps':
            return _pc_nextgen_calculated_local_num_pixmaps(material)
        if name == 'trlau_pc_nextgen_layer_texture_ids':
            return ','.join(str(_get_panel_int(material, f'trlau_ui_pcng_layer{index}_texture_id', -1)) for index in range(_PC_NEXTGEN_LAYER_COUNT))
        if name == 'trlau_pc_nextgen_layer_texture_indices':
            return ','.join(str(value) for value in _pc_nextgen_calculated_layer_texture_indices(material))
        if name == 'trlau_pc_nextgen_layer_enabled':
            return ','.join('1' if bool(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_enabled', False)) else '0' for index in range(_PC_NEXTGEN_LAYER_COUNT))
        if name == 'trlau_pc_nextgen_layer_colors':
            return _pc_nextgen_json_from_float_tuples([_pc_nextgen_color_value(material, f'trlau_ui_pcng_layer{index}_color', (1.0, 1.0, 1.0, 1.0)) for index in range(_PC_NEXTGEN_LAYER_COUNT)])
        if name == 'trlau_pc_nextgen_layer_texcoord_sources':
            return ','.join(str(_pc_nextgen_enum_value(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_texcoord_source', '0'), 0)) for index in range(_PC_NEXTGEN_LAYER_COUNT))
        if name == 'trlau_pc_nextgen_layer_modifiers':
            return ','.join(str(_pc_nextgen_enum_value(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_modifier', '0'), 0)) for index in range(_PC_NEXTGEN_LAYER_COUNT))
        if name == 'trlau_pc_nextgen_layer_param_ids':
            return ','.join(str(_pc_nextgen_enum_value(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_param_id', '0'), 0)) for index in range(_PC_NEXTGEN_LAYER_COUNT))
        if name == 'trlau_pc_nextgen_layer_constants':
            return _pc_nextgen_json_from_float_tuples([_pc_nextgen_color_value(material, f'trlau_ui_pcng_layer{index}_constant', (0.0, 0.0, 0.0, 0.0)) for index in range(_PC_NEXTGEN_LAYER_COUNT)])
        if name == 'trlau_pc_nextgen_layer_num_textures':
            return ','.join(str(value) for value in _pc_nextgen_calculated_layer_num_textures(material))
        if name == 'trlau_pc_nextgen_shader_indices':
            return ','.join(str(_get_panel_int(material, f'trlau_ui_pcng_shader_index{index}', 0)) for index in range(len(_PC_NEXTGEN_SHADER_INDEX_LABELS)))
        if name == 'trlau_pc_nextgen_shader_table_ids':
            return str(_get_panel_prop(material, 'trlau_ui_pcng_shader_table_ids', default or '') or '')
        if name == 'trlau_pc_nextgen_special_material_flag':
            return bool(_get_panel_prop(material, 'trlau_ui_pcng_special_material_flag', default if default is not None else False))
        if name == 'trlau_pc_nextgen_double_sided':
            return bool(_get_panel_prop(material, 'trlau_ui_pcng_double_sided', default if default is not None else False))
        if name == 'trlau_pc_nextgen_double_wound_pair_count':
            return _get_panel_int(material, 'trlau_ui_pcng_double_wound_pair_count', default if default is not None else 0)
        if name == 'trlau_pc_nextgen_layer_image_names':
            return str(_get_panel_prop(material, 'trlau_ui_pcng_layer_image_names', default or '') or '')
        if name == 'trlau_pc_nextgen_last_update_error':
            return str(_get_panel_prop(material, 'trlau_ui_pcng_last_update_error', default or '') or '')
        if name == 'trlau_pc_nextgen_diffuse_texture_id':
            return _get_panel_int(material, 'trlau_ui_pcng_diffuse_texture_id', default if default is not None else -1)
        if name == 'trlau_pc_nextgen_normal_texture_id':
            return _get_panel_int(material, 'trlau_ui_pcng_normal_texture_id', default if default is not None else -1)
        if name == 'trlau_pc_nextgen_specular_texture_id':
            return _get_panel_int(material, 'trlau_ui_pcng_specular_texture_id', default if default is not None else -1)
        if name == 'trlau_pc_nextgen_panel_sync_version':
            return _get_panel_int(material, 'trlau_ui_pcng_panel_sync_version', default if default is not None else 0)
    except Exception:
        return default
    return default

def _pc_nextgen_color_value(material, name: str, fallback=(1.0, 1.0, 1.0, 1.0)) -> tuple[float, float, float, float]:
    try:
        value = list(getattr(material, name))[:4]
    except Exception:
        value = list(fallback)
    while len(value) < 4:
        value.append(1.0 if len(value) == 3 else 0.0)
    try:
        return tuple(float(component) for component in value[:4])
    except Exception:
        return tuple(float(component) for component in fallback)


def _pc_nextgen_set_custom_prop(material, name: str, value) -> None:
    return


def _pc_nextgen_get_array_int(material, key: str, length: int, default: int = 0) -> list[int]:
    values = _csv_ints(material.get(key, ''))
    if len(values) < length:
        values.extend([int(default)] * (length - len(values)))
    return [int(value) for value in values[:length]]


def _pc_nextgen_get_array_color(material, key: str, length: int, default=(1.0, 1.0, 1.0, 1.0)) -> list[tuple[float, float, float, float]]:
    values = _pc_nextgen_json_float_tuples(material.get(key, '[]'))
    if len(values) < length:
        values.extend([tuple(default)] * (length - len(values)))
    return values[:length]


def _pc_nextgen_panel_mismatches_imported_data(material) -> bool:
    if material is None or not _is_pc_nextgen_material(material):
        return False
    checks: list[tuple[str, object, object]] = []
    if 'trlau_pc_nextgen_specular_power' in material:
        checks.append(('trlau_ui_pcng_specular_power', _safe_get_float(material, 'trlau_pc_nextgen_specular_power', 0.0), 0.0))
    if 'trlau_pc_nextgen_opacity' in material:
        checks.append(('trlau_ui_pcng_opacity', _safe_get_float(material, 'trlau_pc_nextgen_opacity', 1.0), 1.0))

    imported_layer_ids = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_layer_texture_ids', _PC_NEXTGEN_LAYER_COUNT, -1)
    for layer_index in range(min(3, len(imported_layer_ids))):
        imported_id = int(imported_layer_ids[layer_index])
        if imported_id >= 0:
            checks.append((f'trlau_ui_pcng_layer{layer_index}_texture_id', imported_id, -1))

    for prop_name, imported_value, _default_value in checks:
        try:
            panel_value = getattr(material, prop_name)
        except Exception:
            return True
        try:
            if isinstance(imported_value, float):
                if abs(float(panel_value) - float(imported_value)) > 1e-5 and abs(float(panel_value) - float(_default_value)) <= 1e-5:
                    return True
            elif int(panel_value) != int(imported_value) and int(panel_value) == int(_default_value):
                return True
        except Exception:
            if str(panel_value) != str(imported_value) and str(panel_value) == str(_default_value):
                return True
    return False


def ensure_pc_nextgen_material_panel_props(material, *, refresh_stale_defaults: bool = True) -> None:
    if material is None or not _is_pc_nextgen_material(material):
        return
    try:
        current_version = int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_panel_sync_version', material.get('trlau_pc_nextgen_panel_sync_version', 0)))
    except Exception:
        current_version = 0
    if current_version >= _PC_NEXTGEN_PANEL_VERSION:
        if not refresh_stale_defaults or not _pc_nextgen_panel_mismatches_imported_data(material):
            _pc_nextgen_apply_backface_culling_from_double_sided(material)
            return

    def enum_identifier(value: int, items, fallback='0') -> str:
        text = str(int(value))
        return text if any(item[0] == text for item in items) else fallback

    _set_panel_prop(material, 'trlau_ui_material_platform', 'pc_nextgen')
    _clear_removed_pc_nextgen_panel_fields(material)
    try:
        legacy_material_id = _safe_get_int(material, 'trlau_pc_nextgen_material_id', -1)
        if legacy_material_id > 0 and 'trlau_ui_draw_group' not in material and 'trlau_draw_group' not in material:
            _set_panel_prop(material, 'trlau_ui_draw_group', int(legacy_material_id) - 1)
    except Exception:
        pass
    _set_panel_prop(material, 'trlau_ui_pcng_material_record_offset', _safe_get_int(material, 'trlau_pc_nextgen_material_record_offset', 0))
    _set_panel_prop(material, 'trlau_ui_pcng_material_record_hex', str(material.get('trlau_pc_nextgen_material_record_hex', '') or ''))
    _set_panel_prop(material, 'trlau_ui_pcng_asset_id_padding_hex', str(material.get('trlau_pc_nextgen_asset_id_padding_hex', '') or ''))
    _set_panel_prop(material, 'trlau_ui_pcng_shader_table_ids', str(material.get('trlau_pc_nextgen_shader_table_ids', '') or ''))
    _set_panel_prop(material, 'trlau_ui_pcng_double_wound_pair_count', _safe_get_int(material, 'trlau_pc_nextgen_double_wound_pair_count', 0))
    _set_panel_prop(material, 'trlau_ui_pcng_blend_mode', enum_identifier(_safe_get_int(material, 'trlau_pc_nextgen_blend_mode', 0), _PC_NEXTGEN_BLEND_MODE_ITEMS))
    _set_panel_prop(material, 'trlau_ui_pcng_combiner_type', enum_identifier(_safe_get_int(material, 'trlau_pc_nextgen_combiner_type', 0), _PC_NEXTGEN_COMBINER_TYPE_ITEMS))
    _set_panel_prop(material, 'trlau_ui_pcng_material_flags', _pc_nextgen_to_int_text(material.get('trlau_pc_nextgen_material_flags', 0), hex_format=True))
    _set_panel_prop(material, 'trlau_ui_pcng_opacity', max(0.0, min(1.0, _safe_get_float(material, 'trlau_pc_nextgen_opacity', 1.0))))
    _set_panel_prop(material, 'trlau_ui_pcng_poly_flags', '0x00000000')
    _set_panel_prop(material, 'trlau_ui_pcng_uv_auto_scroll_speed', _safe_get_int(material, 'trlau_pc_nextgen_uv_auto_scroll_speed', 0) & 0xFFFF)
    _set_panel_prop(material, 'trlau_ui_pcng_sort_bias', _safe_get_float(material, 'trlau_pc_nextgen_sort_bias', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_detail_range_mul', _safe_get_float(material, 'trlau_pc_nextgen_detail_range_mul', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_detail_scale', _safe_get_float(material, 'trlau_pc_nextgen_detail_scale', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_parallax_scale', _safe_get_float(material, 'trlau_pc_nextgen_parallax_scale', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_parallax_offset', _safe_get_float(material, 'trlau_pc_nextgen_parallax_offset', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_specular_power', _safe_get_float(material, 'trlau_pc_nextgen_specular_power', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_specular_shift0', _safe_get_float(material, 'trlau_pc_nextgen_specular_shift0', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_specular_shift1', _safe_get_float(material, 'trlau_pc_nextgen_specular_shift1', 0.0))
    rim_colors = _pc_nextgen_json_float_tuples(material.get('trlau_pc_nextgen_rim_light_color', '[]'), [(0.0, 0.0, 0.0, 0.0)])
    _set_panel_prop(material, 'trlau_ui_pcng_rim_light_color', rim_colors[0] if rim_colors else (0.0, 0.0, 0.0, 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_rim_light_intensity', _safe_get_float(material, 'trlau_pc_nextgen_rim_light_intensity', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_water_blend_bias', _safe_get_float(material, 'trlau_pc_nextgen_water_blend_bias', 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_water_blend_exponent', _safe_get_float(material, 'trlau_pc_nextgen_water_blend_exponent', 0.0))
    water_colors = _pc_nextgen_json_float_tuples(material.get('trlau_pc_nextgen_water_deep_color', '[]'), [(0.0, 0.0, 0.0, 0.0)])
    _set_panel_prop(material, 'trlau_ui_pcng_water_deep_color', water_colors[0] if water_colors else (0.0, 0.0, 0.0, 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_special_material_flag', bool(material.get('trlau_pc_nextgen_special_material_flag', False)))
    _set_panel_prop(material, 'trlau_ui_pcng_double_sided', bool(material.get('trlau_pc_nextgen_double_sided', False)))
    _pc_nextgen_apply_backface_culling_from_double_sided(material)
    _set_panel_prop(material, 'trlau_ui_pcng_diffuse_texture_id', _safe_get_int(material, 'trlau_pc_nextgen_diffuse_texture_id', -1))
    _set_panel_prop(material, 'trlau_ui_pcng_normal_texture_id', _safe_get_int(material, 'trlau_pc_nextgen_normal_texture_id', -1))
    _set_panel_prop(material, 'trlau_ui_pcng_specular_texture_id', _safe_get_int(material, 'trlau_pc_nextgen_specular_texture_id', -1))

    shader_indices = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_shader_indices', len(_PC_NEXTGEN_SHADER_INDEX_LABELS), 0)
    for index, value in enumerate(shader_indices):
        _set_panel_prop(material, f'trlau_ui_pcng_shader_index{index}', int(value))

    layer_texture_ids = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_layer_texture_ids', _PC_NEXTGEN_LAYER_COUNT, -1)
    layer_enabled = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_layer_enabled', _PC_NEXTGEN_LAYER_COUNT, 0)
    layer_colors = _pc_nextgen_get_array_color(material, 'trlau_pc_nextgen_layer_colors', _PC_NEXTGEN_LAYER_COUNT, (1.0, 1.0, 1.0, 1.0))
    layer_texcoord_sources = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_layer_texcoord_sources', _PC_NEXTGEN_LAYER_COUNT, 0)
    layer_modifiers = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_layer_modifiers', _PC_NEXTGEN_LAYER_COUNT, 0)
    layer_param_ids = _pc_nextgen_get_array_int(material, 'trlau_pc_nextgen_layer_param_ids', _PC_NEXTGEN_LAYER_COUNT, 0)
    layer_constants = _pc_nextgen_get_array_color(material, 'trlau_pc_nextgen_layer_constants', _PC_NEXTGEN_LAYER_COUNT, (0.0, 0.0, 0.0, 0.0))
    texcoord_ids = {item[0] for item in _PC_NEXTGEN_TEXCOORD_SOURCE_ITEMS}
    modifier_ids = {item[0] for item in _PC_NEXTGEN_TEXCOORD_MODIFIER_ITEMS}
    param_ids = {item[0] for item in _PC_NEXTGEN_PARAM_ID_ITEMS}
    for index in range(_PC_NEXTGEN_LAYER_COUNT):
        enabled = bool(layer_enabled[index])
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_enabled', enabled)
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_texture_id', int(layer_texture_ids[index]))
        texcoord = str(int(layer_texcoord_sources[index]))
        modifier = str(int(layer_modifiers[index]))
        param = str(int(layer_param_ids[index]))
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_texcoord_source', texcoord if texcoord in texcoord_ids else '0')
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_modifier', modifier if modifier in modifier_ids else '0')
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_param_id', param if param in param_ids else '0')
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_color', layer_colors[index])
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_constant', layer_constants[index])

    _set_panel_prop(material, 'trlau_ui_pcng_panel_sync_version', _PC_NEXTGEN_PANEL_VERSION)
    _clear_pc_nextgen_metadata(material)


def sync_pc_nextgen_material_from_panel(material) -> None:
    if material is None:
        return
    set_material_platform_name(material, 'pc_nextgen')

    layer_texture_ids: list[int] = []
    layer_enabled: list[int] = []
    layer_colors: list[tuple[float, float, float, float]] = []
    layer_texcoord_sources: list[int] = []
    layer_modifiers: list[int] = []
    layer_param_ids: list[int] = []
    layer_constants: list[tuple[float, float, float, float]] = []
    for index in range(_PC_NEXTGEN_LAYER_COUNT):
        enabled = 1 if bool(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_enabled', False)) else 0
        layer_enabled.append(enabled)
        layer_texture_ids.append(_get_panel_int(material, f'trlau_ui_pcng_layer{index}_texture_id', -1))
        layer_texcoord_sources.append(_pc_nextgen_enum_value(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_texcoord_source', '0'), 0))
        layer_modifiers.append(_pc_nextgen_enum_value(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_modifier', '0'), 0))
        layer_param_ids.append(_pc_nextgen_enum_value(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_param_id', '0'), 0))
        layer_colors.append(_pc_nextgen_color_value(material, f'trlau_ui_pcng_layer{index}_color', (1.0, 1.0, 1.0, 1.0)))
        layer_constants.append(_pc_nextgen_color_value(material, f'trlau_ui_pcng_layer{index}_constant', (0.0, 0.0, 0.0, 0.0)))

    shader_indices = [_get_panel_int(material, f'trlau_ui_pcng_shader_index{index}', 0) for index in range(len(_PC_NEXTGEN_SHADER_INDEX_LABELS))]

    _clear_removed_pc_nextgen_panel_fields(material)
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_material_id', int(_get_panel_int(material, 'trlau_ui_draw_group', 0)) + 1)
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_asset_id_hi', 0)
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_asset_id_lo', 0)
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_blend_mode', _pc_nextgen_enum_value(_get_panel_prop(material, 'trlau_ui_pcng_blend_mode', '0'), 0))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_combiner_type', _pc_nextgen_enum_value(_get_panel_prop(material, 'trlau_ui_pcng_combiner_type', '0'), 0))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_material_flags', _pc_nextgen_parse_int_text(_get_panel_prop(material, 'trlau_ui_pcng_material_flags', '0'), 0))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_opacity', max(0.0, min(1.0, float(_get_panel_prop(material, 'trlau_ui_pcng_opacity', 1.0)))))
    _set_panel_prop(material, 'trlau_ui_pcng_poly_flags', '0x00000000')
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_poly_flags', 0)
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_uv_auto_scroll_speed', _get_panel_int(material, 'trlau_ui_pcng_uv_auto_scroll_speed', 0) & 0xFFFF)
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_sort_bias', float(_get_panel_prop(material, 'trlau_ui_pcng_sort_bias', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_detail_range_mul', float(_get_panel_prop(material, 'trlau_ui_pcng_detail_range_mul', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_detail_scale', float(_get_panel_prop(material, 'trlau_ui_pcng_detail_scale', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_parallax_scale', float(_get_panel_prop(material, 'trlau_ui_pcng_parallax_scale', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_parallax_offset', float(_get_panel_prop(material, 'trlau_ui_pcng_parallax_offset', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_specular_power', float(_get_panel_prop(material, 'trlau_ui_pcng_specular_power', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_specular_shift0', float(_get_panel_prop(material, 'trlau_ui_pcng_specular_shift0', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_specular_shift1', float(_get_panel_prop(material, 'trlau_ui_pcng_specular_shift1', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_rim_light_color', _pc_nextgen_json_from_float_tuples([_pc_nextgen_color_value(material, 'trlau_ui_pcng_rim_light_color', (0.0, 0.0, 0.0, 0.0))]))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_rim_light_intensity', float(_get_panel_prop(material, 'trlau_ui_pcng_rim_light_intensity', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_water_blend_bias', float(_get_panel_prop(material, 'trlau_ui_pcng_water_blend_bias', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_water_blend_exponent', float(_get_panel_prop(material, 'trlau_ui_pcng_water_blend_exponent', 0.0)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_water_deep_color', _pc_nextgen_json_from_float_tuples([_pc_nextgen_color_value(material, 'trlau_ui_pcng_water_deep_color', (0.0, 0.0, 0.0, 0.0))]))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_local_num_pixmaps', _pc_nextgen_calculated_local_num_pixmaps(material))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_texture_ids', ','.join(str(int(value)) for value in layer_texture_ids))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_enabled', ','.join(str(int(value)) for value in layer_enabled))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_colors', _pc_nextgen_json_from_float_tuples(layer_colors))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_texcoord_sources', ','.join(str(int(value)) for value in layer_texcoord_sources))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_modifiers', ','.join(str(int(value)) for value in layer_modifiers))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_param_ids', ','.join(str(int(value)) for value in layer_param_ids))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_constants', _pc_nextgen_json_from_float_tuples(layer_constants))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_layer_num_textures', ','.join(str(int(value)) for value in _pc_nextgen_calculated_layer_num_textures(material)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_shader_indices', ','.join(str(int(value)) for value in shader_indices))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_special_material_flag', bool(_get_panel_prop(material, 'trlau_ui_pcng_special_material_flag', False)))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_double_sided', bool(_get_panel_prop(material, 'trlau_ui_pcng_double_sided', False)))
    _pc_nextgen_apply_backface_culling_from_double_sided(material)

    diffuse_id = layer_texture_ids[0] if len(layer_texture_ids) > 0 and layer_enabled[0] else -1
    normal_id = layer_texture_ids[1] if len(layer_texture_ids) > 1 and layer_enabled[1] else -1
    specular_id = layer_texture_ids[2] if len(layer_texture_ids) > 2 and layer_enabled[2] else -1
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_diffuse_texture_id', int(diffuse_id))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_normal_texture_id', int(normal_id))
    _pc_nextgen_set_custom_prop(material, 'trlau_pc_nextgen_specular_texture_id', int(specular_id))
    _set_panel_prop(material, 'trlau_ui_pcng_diffuse_texture_id', int(diffuse_id))
    _set_panel_prop(material, 'trlau_ui_pcng_normal_texture_id', int(normal_id))
    _set_panel_prop(material, 'trlau_ui_pcng_specular_texture_id', int(specular_id))
    _set_panel_prop(material, 'trlau_ui_pcng_panel_sync_version', _PC_NEXTGEN_PANEL_VERSION)
    _clear_pc_nextgen_metadata(material)

def apply_pc_nextgen_material_settings(material, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> None:
    global _MATERIAL_SYNC_GUARD
    if _MATERIAL_SYNC_GUARD or material is None:
        return

    # PC next-gen materials are driven by PCMaterialData/PCMaterialDataLayer.
    # Do not pass through the legacy tpage material rebuild path.
    set_material_platform_name(material, 'pc_nextgen')
    _pc_nextgen_capture_current_layer_images(material)
    cleanup_material_metadata(material)

    flags = decode_tpage_flags_for_platform(0, 'pc_nextgen')
    diffuse_texture_id = int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_diffuse_texture_id', -1) or -1)
    if diffuse_texture_id < 0:
        layer_ids = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_texture_ids', ''))
        diffuse_texture_id = next((int(value) for value in layer_ids if int(value) >= 0), 0) if layer_ids else 0
    flags['texture_id'] = max(0, int(diffuse_texture_id)) & 0x1FFF

    _set_panel_prop(material, 'trlau_ui_texture_id', int(flags['texture_id']))
    _set_panel_prop(material, 'trlau_ui_material_platform', 'pc_nextgen')
    _pc_nextgen_apply_backface_culling_from_double_sided(material)
    _set_panel_prop(material, 'trlau_ui_blend_value', 0)
    _set_panel_prop(material, 'trlau_ui_cull_mode', 0)

    vertex_color_attribute_name = _get_material_vertex_color_assignment_name(material)
    use_vertex_colors = vertex_color_attribute_name is not None
    owner_mesh = getattr(owner, 'data', None) if owner is not None else None
    owner_vertex_color_name = _get_mesh_vertex_color_name(owner_mesh)
    if owner_vertex_color_name is not None:
        use_vertex_colors = True
        vertex_color_attribute_name = owner_vertex_color_name

    layer_images = _load_pc_nextgen_layer_images(material, owner=owner, extra_dirs=extra_dirs)

    builder = MeshBuilder.__new__(MeshBuilder)
    previous_guard = _MATERIAL_SYNC_GUARD
    _MATERIAL_SYNC_GUARD = True
    try:
        builder._setup_pc_nextgen_material_nodes(
            material,
            layer_images,
            flags,
            use_vertex_colors=use_vertex_colors,
            vertex_color_attribute_name=vertex_color_attribute_name,
        )
        _pc_nextgen_apply_backface_culling_from_double_sided(material)
        apply_texture_scroll_settings(material)
    finally:
        _MATERIAL_SYNC_GUARD = previous_guard


def _load_ps3_role_image(material, texture_id: int, role: str, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> Optional[bpy.types.Image]:
    texture_id = _clamp_texture_id(texture_id, allow_disabled=True)
    if texture_id < 0:
        return None

    image = _load_image_for_texture_id(material, texture_id, owner=owner, extra_dirs=extra_dirs)
    _set_ps3_texture_role_metadata(image, texture_id, role)
    return image


def apply_ps3_material_settings(material, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> None:
    global _MATERIAL_SYNC_GUARD
    if _MATERIAL_SYNC_GUARD or material is None:
        return

    material_platform = get_material_platform_name(material) or 'ps3'
    if material_platform not in {'ps3', 'xbox360'}:
        material_platform = 'ps3'
    flags = decode_tpage_flags_for_platform(get_material_tpageid(material), material_platform)
    diffuse_texture_id = _clamp_texture_id(
        _get_panel_int(material, 'trlau_ui_ps3_diffuse_texture_id', flags.get('texture_id', 0)),
        allow_disabled=False,
    )
    normal_texture_id = _clamp_texture_id(
        _get_panel_int(material, 'trlau_ui_ps3_normal_texture_id', -1),
        allow_disabled=True,
    )
    normal_candidate_texture_id = _clamp_texture_id(
        _get_panel_int(material, 'trlau_ui_ps3_normal_candidate_texture_id', -1),
        allow_disabled=True,
    )
    specular_texture_id = _clamp_texture_id(
        _get_panel_int(material, 'trlau_ui_ps3_specular_texture_id', -1),
        allow_disabled=True,
    )
    ps3_external_render_stream = _get_panel_bool(material, 'trlau_ui_ps3_external_render_stream', False)

    effective_normal_texture_id = int(normal_texture_id)
    if ps3_external_render_stream and effective_normal_texture_id < 0 and normal_candidate_texture_id >= 0 and normal_candidate_texture_id != specular_texture_id:
        effective_normal_texture_id = int(normal_candidate_texture_id)
    elif not ps3_external_render_stream:
        effective_normal_texture_id = -1

    flags['texture_id'] = diffuse_texture_id
    set_material_tpageid(material, encode_tpage_flags_for_platform(flags, material_platform))
    set_material_platform_name(material, material_platform)
    _set_panel_prop(material, 'trlau_ui_ps3_diffuse_texture_id', int(diffuse_texture_id))
    _set_panel_prop(material, 'trlau_ui_ps3_normal_texture_id', int(effective_normal_texture_id if ps3_external_render_stream else normal_texture_id))
    _set_panel_prop(material, 'trlau_ui_ps3_normal_candidate_texture_id', int(normal_candidate_texture_id))
    _set_panel_prop(material, 'trlau_ui_ps3_specular_texture_id', int(specular_texture_id))
    _set_panel_prop(material, 'trlau_ui_ps3_external_render_stream', bool(ps3_external_render_stream))
    _set_panel_prop(material, 'trlau_ui_ps3_role_mapping_version', 4)
    cleanup_material_metadata(material)
    rename_material_from_tpageid(material)

    # whatever I was doing here, delete this stuff, it's useless
    for key in (
        'trlau_ps3_normal_mode',
        'trlau_ps3_specular_source_mode',
        'trlau_ps3_normal_strength',
        'trlau_ps3_bump_strength',
        'trlau_ps3_bump_distance',
        'trlau_ps3_specular_strength',
        'trlau_ps3_roughness_dark',
        'trlau_ps3_roughness_bright',
    ):
        try:
            if key in material:
                del material[key]
        except Exception:
            pass

    material.use_backface_culling = bool(flags['single_sided'])
    diffuse_image = _load_ps3_role_image(material, diffuse_texture_id, 'diffuse', owner=owner, extra_dirs=extra_dirs)
    normal_image = _load_ps3_role_image(material, effective_normal_texture_id, 'normal', owner=owner, extra_dirs=extra_dirs)
    specular_image = _load_ps3_role_image(material, specular_texture_id, 'specular', owner=owner, extra_dirs=extra_dirs)

    vertex_color_attribute_name = _get_material_vertex_color_assignment_name(material)
    use_vertex_colors = vertex_color_attribute_name is not None
    owner_mesh = getattr(owner, 'data', None) if owner is not None else None
    owner_vertex_color_name = _get_mesh_vertex_color_name(owner_mesh)
    if owner_vertex_color_name is not None:
        use_vertex_colors = True
        vertex_color_attribute_name = owner_vertex_color_name

    reflection_mode = _get_reflection_mode(material)
    reflective = reflection_mode != 'none'

    builder = MeshBuilder.__new__(MeshBuilder)
    previous_guard = _MATERIAL_SYNC_GUARD
    _MATERIAL_SYNC_GUARD = True
    try:
        builder._setup_material_nodes(
            material,
            diffuse_image,
            flags,
            use_vertex_colors=use_vertex_colors,
            vertex_color_attribute_name=vertex_color_attribute_name,
            reflective=reflective,
            reflection_mode=reflection_mode,
            normal_image=normal_image,
            specular_image=specular_image,
        )
        apply_texture_scroll_settings(material)
    finally:
        _MATERIAL_SYNC_GUARD = previous_guard

def apply_trlau_material_settings(material, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> None:
    global _MATERIAL_SYNC_GUARD
    if _MATERIAL_SYNC_GUARD:
        return
    if _is_helper_material(material):
        return

    if _is_pc_nextgen_material(material):
        apply_pc_nextgen_material_settings(material, owner=owner, extra_dirs=extra_dirs)
        return

    if _is_ps3_material(material):
        apply_ps3_material_settings(material, owner=owner, extra_dirs=extra_dirs)
        return

    cleanup_material_metadata(material)
    flags = _get_material_flags(material)
    _store_material_flags(material, flags)
    for name in _FLAG_NAMES:
        flag_key = f'trlau_{name}'
        if flag_key in material and not (_is_psp_material(material) and flag_key == 'trlau_texture_id'):
            del material[flag_key]

    material.use_backface_culling = bool(flags['single_sided'])

    image = _load_image_for_texture_id(material, flags['texture_id'], owner=owner, extra_dirs=extra_dirs)
    vertex_color_attribute_name = _get_material_vertex_color_assignment_name(material)
    use_vertex_colors = vertex_color_attribute_name is not None
    owner_mesh = getattr(owner, 'data', None) if owner is not None else None
    owner_vertex_color_name = _get_mesh_vertex_color_name(owner_mesh)
    if owner_vertex_color_name is not None:
        use_vertex_colors = True
        vertex_color_attribute_name = owner_vertex_color_name
    reflection_mode = _get_reflection_mode(material)
    reflective = reflection_mode != 'none'

    builder = MeshBuilder.__new__(MeshBuilder)
    previous_guard = _MATERIAL_SYNC_GUARD
    _MATERIAL_SYNC_GUARD = True
    try:
        builder._setup_material_nodes(
            material,
            image,
            flags,
            use_vertex_colors=use_vertex_colors,
            vertex_color_attribute_name=vertex_color_attribute_name,
            reflective=reflective,
            reflection_mode=reflection_mode,
        )
        apply_texture_scroll_settings(material)
    finally:
        _MATERIAL_SYNC_GUARD = previous_guard


def _get_flag_value(self, key: str) -> int:
    flags = _get_material_flags(self)
    return int(flags[key])


def _set_flag_value(self, key: str, value: int) -> None:
    flags = _get_material_flags(self)
    flags[key] = int(value)
    if _is_psp_material(self):
        if key == 'texture_id':
            self['trlau_psp_texture_id'] = int(value) & 0x1FFF
        elif key == 'blend_value':
            self['trlau_psp_blend'] = int(value) & 0xF
            self['trlau_psp_blend_low_nibble'] = int(value) & 0xF
            _set_psp_alpha_blend_hint(self)
    _store_material_flags(self, flags)
    apply_trlau_material_settings(self)


def _get_texture_id(self):
    return _get_flag_value(self, 'texture_id')


def _get_texture_id_assignment(self):
    return _is_texture_id_assignment_enabled()


def _set_texture_id_assignment(self, value):
    context = getattr(bpy, 'context', None)
    scene = getattr(context, 'scene', None) if context is not None else None
    if scene is None:
        return
    scene.trlau_texture_id_assignment = bool(value)
    if bool(value):
        texture_id = _get_flag_value(self, 'texture_id')
        _assign_current_material_nodes(self, texture_id)


def _set_texture_id(self, value):
    texture_id = max(0, min(int(value), 0x1FFF))
    flags = _get_material_flags(self)
    flags['texture_id'] = texture_id
    if _is_psp_material(self):
        self['trlau_psp_texture_id'] = texture_id
        self['trlau_texture_id'] = texture_id
    _store_material_flags(self, flags)

    owner = None
    context = getattr(bpy, 'context', None)
    active_object = getattr(context, 'object', None) if context is not None else None
    if active_object is not None and getattr(active_object, 'type', None) == 'MESH':
        materials = getattr(getattr(active_object, 'data', None), 'materials', None)
        if materials is not None:
            try:
                if any(material is self for material in materials):
                    owner = active_object
            except Exception:
                owner = None

    if owner is not None:
        owner_vertex_color_name = _get_mesh_vertex_color_name(getattr(owner, 'data', None))
        if owner_vertex_color_name is not None:
            pass

    if _is_texture_id_assignment_enabled():
        _assign_current_material_nodes(self, texture_id)
        image = _find_existing_image(self)
        if image is not None:
            try:
                image['trlau_texture_id'] = texture_id
            except Exception:
                pass

    apply_trlau_material_settings(self, owner=owner)


def _get_blend_value(self):
    return _get_flag_value(self, 'blend_value')


def _set_blend_value(self, value):
    _set_flag_value(self, 'blend_value', max(0, min(int(value), 0xF)))


def _get_psp_blend(self):
    return int(self.get('trlau_psp_blend', _get_flag_value(self, 'blend_value'))) & 0xFF


def _set_psp_blend(self, value):
    psp_blend = max(0, min(int(value), 0xFF))
    self['trlau_psp_blend'] = psp_blend
    self['trlau_psp_blend_low_nibble'] = psp_blend & 0xF
    _set_psp_alpha_blend_hint(self)
    flags = _get_material_flags(self)
    flags['blend_value'] = psp_blend & 0xF
    _store_material_flags(self, flags)
    apply_trlau_material_settings(self)


def _get_psp_render_flags(self):
    return int(self.get('trlau_psp_render_flags', (int(self.get('trlau_psp_mode_word', 0)) >> 8) & 0xFF)) & 0xFF


def _set_psp_render_flags(self, value):
    self['trlau_psp_render_flags'] = max(0, min(int(value), 0xFF))
    _sync_psp_mode_word(self)
    _set_psp_alpha_blend_hint(self)
    apply_trlau_material_settings(self)


def _get_psp_vertex_mode(self):
    return int(self.get('trlau_psp_vertex_mode', int(self.get('trlau_psp_mode_word', 0)) & 0xFF)) & 0xFF


def _set_psp_vertex_mode(self, value):
    self['trlau_psp_vertex_mode'] = max(0, min(int(value), 0xFF))
    _sync_psp_mode_word(self)
    apply_trlau_material_settings(self)


def _get_psp_vertex_format_flags(self):
    return int(self.get('trlau_psp_vertex_format_flags', 0)) & 0xFFFFFFFF


def _set_psp_vertex_format_flags(self, value):
    self['trlau_psp_vertex_format_flags'] = max(0, min(int(value), 0xFFFFFFFF))
    _sync_psp_vertex_format_flags(self)
    apply_trlau_material_settings(self)


def _get_psp_env_mapping(self):
    return bool((_get_psp_vertex_format_flags(self) & 0x400) or get_material_env_mapping(self))


def _set_psp_env_mapping(self, value):
    flags = _get_psp_vertex_format_flags(self)
    if value:
        flags |= 0x400
    else:
        flags &= ~0x400
    self['trlau_psp_vertex_format_flags'] = flags
    _sync_psp_vertex_format_flags(self)
    apply_trlau_material_settings(self)


def _get_psp_unknown_800(self):
    return bool(_get_psp_vertex_format_flags(self) & 0x800)


def _set_psp_unknown_800(self, value):
    flags = _get_psp_vertex_format_flags(self)
    if value:
        flags |= 0x800
    else:
        flags &= ~0x800
    self['trlau_psp_vertex_format_flags'] = flags
    _sync_psp_vertex_format_flags(self)
    apply_trlau_material_settings(self)


def _get_cull_mode(self):
    return _get_flag_value(self, 'cull_mode')


def _set_cull_mode(self, value):
    _set_flag_value(self, 'cull_mode', max(0, min(int(value), 0x7)))


def _get_unknown_1(self):
    return bool(_get_flag_value(self, 'unknown_1'))


def _set_unknown_1(self, value):
    _set_flag_value(self, 'unknown_1', 1 if value else 0)


def _get_single_sided(self):
    return bool(_get_flag_value(self, 'single_sided'))


def _set_single_sided(self, value):
    _set_flag_value(self, 'single_sided', 1 if value else 0)


def _get_texture_wrap(self):
    return _get_flag_value(self, 'texture_wrap')


def _set_texture_wrap(self, value):
    _set_flag_value(self, 'texture_wrap', max(0, min(int(value), 0x3)))


def _get_unknown_2(self):
    return bool(_get_flag_value(self, 'unknown_2'))


def _set_unknown_2(self, value):
    _set_flag_value(self, 'unknown_2', 1 if value else 0)


def _get_unknown_3(self):
    return bool(_get_flag_value(self, 'unknown_3'))


def _set_unknown_3(self, value):
    _set_flag_value(self, 'unknown_3', 1 if value else 0)


def _get_flat_shading(self):
    return bool(_get_flag_value(self, 'flat_shading'))


def _set_flat_shading(self, value):
    _set_flag_value(self, 'flat_shading', 1 if value else 0)


def _get_sort_z(self):
    return bool(_get_flag_value(self, 'sort_z'))


def _set_sort_z(self, value):
    _set_flag_value(self, 'sort_z', 1 if value else 0)


def _get_stencil_pass(self):
    return _get_flag_value(self, 'stencil_pass')


def _set_stencil_pass(self, value):
    _set_flag_value(self, 'stencil_pass', max(0, min(int(value), 0x3)))


def _get_stencil_func(self):
    return bool(_get_flag_value(self, 'stencil_func'))


def _set_stencil_func(self, value):
    _set_flag_value(self, 'stencil_func', 1 if value else 0)


def _get_alpha_ref(self):
    return bool(_get_flag_value(self, 'alpha_ref'))


def _set_alpha_ref(self, value):
    _set_flag_value(self, 'alpha_ref', 1 if value else 0)


def _get_env_mapping(self):
    return bool(_get_panel_prop(self, 'trlau_ui_env_mapping', False))


def _set_env_mapping(self, value):
    _set_panel_prop(self, 'trlau_ui_env_mapping', bool(value))
    apply_trlau_material_settings(self)


def _get_eye_ref_env_mapping(self):
    return bool(_get_panel_prop(self, 'trlau_ui_eye_ref_env_mapping', False))


def _set_eye_ref_env_mapping(self, value):
    _set_panel_prop(self, 'trlau_ui_eye_ref_env_mapping', bool(value))
    apply_trlau_material_settings(self)


def _get_draw_group(self):
    try:
        return int(_get_panel_prop(self, 'trlau_ui_draw_group', 0))
    except Exception:
        return 0


def _set_draw_group(self, value):
    _set_panel_prop(self, 'trlau_ui_draw_group', max(-32768, min(int(value), 32767)))


def _ensure_texture_scroll_defaults(material) -> None:
    for key in (
        'trlau_anim_u_speed',
        'trlau_anim_v_speed',
        'trlau_anim_loop_seconds',
        'trlau_anim_loop_tiles_u',
        'trlau_anim_loop_tiles_v',
        'trlau_scroll_entry_count',
        'trlau_scroll_num_tiles',
        'trlau_scroll_tile',
        'trlau_scroll_offset',
        'trlau_game_scroll_speed',
    ):
        try:
            if key in material:
                del material[key]
        except Exception:
            pass


def apply_texture_scroll_settings(material) -> None:
    _ensure_texture_scroll_defaults(material)

    builder = MeshBuilder.__new__(MeshBuilder)
    enabled = bool(_get_panel_prop(material, 'trlau_ui_scroll_enabled', False))
    if enabled:
        try:
            builder._apply_uv_scroll_animation_to_material(material)
        except Exception:
            pass


def _get_scroll_enabled(self):
    return bool(_get_panel_prop(self, 'trlau_ui_scroll_enabled', False))


def _set_scroll_enabled(self, value):
    _set_panel_prop(self, 'trlau_ui_scroll_enabled', bool(value))
    apply_texture_scroll_settings(self)


def _get_scroll_speed(self):
    try:
        return float(_get_panel_prop(self, 'trlau_ui_scroll_speed', 0.0))
    except Exception:
        return 0.0


def _set_scroll_speed(self, value):
    _set_panel_prop(self, 'trlau_ui_scroll_speed', float(value))
    apply_texture_scroll_settings(self)


def reload_textures_for_material(material, owner=None, extra_dirs: Optional[Iterable[Path | str]] = None) -> bool:
    if not _is_trlau_material(material):
        return False

    flags = _get_material_flags(material)
    texture_id = int(flags.get('texture_id', 0))
    if texture_id <= 0:
        apply_trlau_material_settings(material, owner=owner, extra_dirs=extra_dirs)
        return False

    before_image = _find_existing_image(material)
    before_name = str(getattr(before_image, 'name', '') or '') if before_image is not None else ''

    image = _load_image_for_texture_id(material, texture_id, owner=owner, extra_dirs=extra_dirs)

    apply_trlau_material_settings(material, owner=owner, extra_dirs=extra_dirs)

    after_image = _find_existing_image(material)
    after_name = str(getattr(after_image, 'name', '') or '') if after_image is not None else ''
    return bool(after_name and after_name != before_name)


def _mesh_has_color_attribute(mesh) -> bool:
    return _get_mesh_vertex_color_name(mesh) is not None



def reload_textures_for_objects(objects, extra_dirs: Optional[Iterable[Path | str]] = None) -> dict[str, int]:
    processed_materials: set[int] = set()
    processed_images = 0
    processed_material_count = 0

    for obj in objects or ():
        data = getattr(obj, 'data', None)
        materials = getattr(data, 'materials', None)
        if materials is None:
            continue

        vertex_color_name = _get_mesh_vertex_color_name(data)
        use_vertex_colors = vertex_color_name is not None

        for material in materials:
            if material is None:
                continue
            material_key = id(material)
            if material_key in processed_materials or not _is_trlau_material(material):
                continue
            processed_materials.add(material_key)
            processed_material_count += 1

            if reload_textures_for_material(material, owner=obj, extra_dirs=extra_dirs):
                processed_images += 1

    return {
        'materials': processed_material_count,
        'updated': processed_images,
    }


def reload_textures_for_scene(scene=None, selected_only: bool = False) -> dict[str, int]:
    context = bpy.context
    if selected_only and context is not None:
        objects = list(getattr(context, 'selected_objects', []) or [])
    else:
        scene = scene or getattr(context, 'scene', None)
        objects = list(getattr(scene, 'objects', []) or []) if scene is not None else []
    return reload_textures_for_objects(objects)



def _get_ps3_diffuse_texture_id(self):
    return _clamp_texture_id(
        _get_panel_int(self, 'trlau_ui_ps3_diffuse_texture_id', _get_flag_value(self, 'texture_id')),
        allow_disabled=False,
    )


def _set_ps3_diffuse_texture_id(self, value):
    value = _clamp_texture_id(value, allow_disabled=False)
    _set_panel_prop(self, 'trlau_ui_ps3_diffuse_texture_id', int(value))
    flags = _get_material_flags(self)
    flags['texture_id'] = int(value)
    set_material_tpageid(self, encode_tpage_flags_for_platform(flags, get_material_platform_name(self)))
    apply_ps3_material_settings(self)


def _get_ps3_normal_texture_id(self):
    return _clamp_texture_id(
        _get_panel_int(self, 'trlau_ui_ps3_normal_texture_id', -1),
        allow_disabled=True,
    )


def _set_ps3_normal_texture_id(self, value):
    value = _clamp_texture_id(value, allow_disabled=True)
    _set_panel_prop(self, 'trlau_ui_ps3_normal_texture_id', int(value))
    apply_ps3_material_settings(self)


def _get_ps3_specular_texture_id(self):
    return _clamp_texture_id(
        _get_panel_int(self, 'trlau_ui_ps3_specular_texture_id', -1),
        allow_disabled=True,
    )


def _set_ps3_specular_texture_id(self, value):
    value = _clamp_texture_id(value, allow_disabled=True)
    _set_panel_prop(self, 'trlau_ui_ps3_specular_texture_id', int(value))
    apply_ps3_material_settings(self)


_PS3_NORMAL_MODE_TO_INDEX = {'auto': 0, 'normal_map': 1, 'bump_height': 2, 'none': 3}
_PS3_NORMAL_INDEX_TO_MODE = {value: key for key, value in _PS3_NORMAL_MODE_TO_INDEX.items()}
_PS3_SPECULAR_SOURCE_TO_INDEX = {'auto': 0, 'luminance': 1, 'alpha': 2, 'none': 3}
_PS3_SPECULAR_INDEX_TO_SOURCE = {value: key for key, value in _PS3_SPECULAR_SOURCE_TO_INDEX.items()}


def _get_ps3_normal_mode(self):
    return _PS3_NORMAL_MODE_TO_INDEX.get(str(self.get('trlau_ps3_normal_mode', 'auto') or 'auto').lower(), 0)


def _set_ps3_normal_mode(self, value):
    self['trlau_ps3_normal_mode'] = _PS3_NORMAL_INDEX_TO_MODE.get(int(value), 'auto')
    apply_ps3_material_settings(self)


def _get_ps3_specular_source_mode(self):
    return _PS3_SPECULAR_SOURCE_TO_INDEX.get(str(self.get('trlau_ps3_specular_source_mode', 'auto') or 'auto').lower(), 0)


def _set_ps3_specular_source_mode(self, value):
    self['trlau_ps3_specular_source_mode'] = _PS3_SPECULAR_INDEX_TO_SOURCE.get(int(value), 'auto')
    apply_ps3_material_settings(self)


def _get_ps3_normal_strength(self):
    return _safe_get_float(self, 'trlau_ps3_normal_strength', 1.0)


def _set_ps3_normal_strength(self, value):
    self['trlau_ps3_normal_strength'] = max(0.0, min(float(value), 10.0))
    apply_ps3_material_settings(self)


def _get_ps3_bump_strength(self):
    return _safe_get_float(self, 'trlau_ps3_bump_strength', 0.075)


def _set_ps3_bump_strength(self, value):
    self['trlau_ps3_bump_strength'] = max(0.0, min(float(value), 10.0))
    apply_ps3_material_settings(self)


def _get_ps3_bump_distance(self):
    return _safe_get_float(self, 'trlau_ps3_bump_distance', 0.08)


def _set_ps3_bump_distance(self, value):
    self['trlau_ps3_bump_distance'] = max(0.0, min(float(value), 10.0))
    apply_ps3_material_settings(self)


def _get_ps3_specular_strength(self):
    return _safe_get_float(self, 'trlau_ps3_specular_strength', 1.0)


def _set_ps3_specular_strength(self, value):
    self['trlau_ps3_specular_strength'] = max(0.0, min(float(value), 10.0))
    apply_ps3_material_settings(self)


def _get_ps3_roughness_dark(self):
    return _safe_get_float(self, 'trlau_ps3_roughness_dark', 0.82)


def _set_ps3_roughness_dark(self, value):
    self['trlau_ps3_roughness_dark'] = max(0.0, min(float(value), 1.0))
    apply_ps3_material_settings(self)


def _get_ps3_roughness_bright(self):
    return _safe_get_float(self, 'trlau_ps3_roughness_bright', 0.18)


def _set_ps3_roughness_bright(self, value):
    self['trlau_ps3_roughness_bright'] = max(0.0, min(float(value), 1.0))
    apply_ps3_material_settings(self)




def _clear_pc_nextgen_metadata(material) -> None:
    if material is None:
        return
    try:
        keys = [str(key) for key in material.keys()]
    except Exception:
        keys = []
    for key in keys:
        if key.startswith('trlau_pc_nextgen_'):
            try:
                del material[key]
            except Exception:
                pass


def _clear_removed_pc_nextgen_panel_fields(material) -> None:
    if material is None:
        return
    names = {
        'trlau_ui_pcng_material_id',
        'trlau_ui_pcng_asset_id_hi',
        'trlau_ui_pcng_asset_id_lo',
        'trlau_ui_pcng_local_num_pixmaps',
    }
    names.update(f'trlau_ui_pcng_layer{index}_num_textures' for index in range(_PC_NEXTGEN_LAYER_COUNT))
    names.update(f'trlau_ui_pcng_layer{index}_texture_index' for index in range(_PC_NEXTGEN_LAYER_COUNT))
    for name in names:
        try:
            if name in material:
                del material[name]
                continue
        except Exception:
            pass
        try:
            material.property_unset(name)
        except Exception:
            pass


def _pc_nextgen_layer_has_export_texture(material, index: int) -> bool:
    try:
        if not bool(_get_panel_prop(material, f'trlau_ui_pcng_layer{index}_enabled', False)):
            return False
        if _get_panel_int(material, f'trlau_ui_pcng_layer{index}_texture_id', -1) < 0:
            return False
        return True
    except Exception:
        return False


def _pc_nextgen_calculated_layer_num_textures(material) -> list[int]:
    return [1 if _pc_nextgen_layer_has_export_texture(material, index) else 0 for index in range(_PC_NEXTGEN_LAYER_COUNT)]


def _pc_nextgen_calculated_layer_texture_indices(material) -> list[int]:
    indices: list[int] = []
    next_index = 0
    for index in range(_PC_NEXTGEN_LAYER_COUNT):
        if _pc_nextgen_layer_has_export_texture(material, index):
            indices.append(next_index)
            next_index += 1
        else:
            indices.append(0xFFFF)
    return indices


def _pc_nextgen_calculated_local_num_pixmaps(material) -> int:
    return sum(1 for value in _pc_nextgen_calculated_layer_num_textures(material) if int(value) > 0)


def _material_primary_texture_id(material) -> int:
    if material is None:
        return 0
    if _is_pc_nextgen_material(material):
        try:
            layer_ids = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_texture_ids', ''))
            layer_enabled = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_enabled', ''))
            for index, value in enumerate(layer_ids):
                enabled = True
                if index < len(layer_enabled):
                    enabled = int(layer_enabled[index]) != 0
                if enabled and int(value) >= 0:
                    return int(value) & 0x1FFF
        except Exception:
            pass
        try:
            value = int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_diffuse_texture_id', -1))
            if value >= 0:
                return value & 0x1FFF
        except Exception:
            pass
    if get_material_platform_name(material) in {'ps3', 'xbox360'}:
        try:
            return int(_get_panel_prop(material, 'trlau_ui_ps3_diffuse_texture_id', 0)) & 0x1FFF
        except Exception:
            pass
    try:
        return int(_get_panel_prop(material, 'trlau_ui_texture_id', 0)) & _texture_id_mask_for_material(material)
    except Exception:
        pass
    try:
        return int(decode_tpage_flags_for_platform(get_material_tpageid(material), get_material_platform_name(material)).get('texture_id', 0)) & 0x1FFF
    except Exception:
        return 0


def _material_normal_texture_id(material) -> int:
    if material is None:
        return -1
    if _is_pc_nextgen_material(material):
        try:
            ids = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_texture_ids', ''))
            enabled = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_enabled', ''))
            if len(ids) > 1 and ids[1] >= 0 and (len(enabled) <= 1 or enabled[1]):
                return int(ids[1]) & 0x1FFF
        except Exception:
            pass
        return int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_normal_texture_id', -1) or -1)
    if get_material_platform_name(material) in {'ps3', 'xbox360'}:
        try:
            return int(_get_panel_prop(material, 'trlau_ui_ps3_normal_texture_id', -1))
        except Exception:
            return -1
    return -1


def _material_specular_texture_id(material) -> int:
    if material is None:
        return -1
    if _is_pc_nextgen_material(material):
        try:
            ids = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_texture_ids', ''))
            enabled = _csv_ints(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_layer_enabled', ''))
            if len(ids) > 2 and ids[2] >= 0 and (len(enabled) <= 2 or enabled[2]):
                return int(ids[2]) & 0x1FFF
        except Exception:
            pass
        return int(_pc_nextgen_panel_storage_value(material, 'trlau_pc_nextgen_specular_texture_id', -1) or -1)
    if get_material_platform_name(material) in {'ps3', 'xbox360'}:
        try:
            return int(_get_panel_prop(material, 'trlau_ui_ps3_specular_texture_id', -1))
        except Exception:
            return -1
    return -1


def initialize_pc_nextgen_material_from_current(material) -> None:
    if material is None:
        return
    diffuse_id = _clamp_texture_id(_material_primary_texture_id(material), allow_disabled=False)
    normal_id = _clamp_texture_id(_material_normal_texture_id(material), allow_disabled=True)
    specular_id = _clamp_texture_id(_material_specular_texture_id(material), allow_disabled=True)
    try:
        flags = decode_tpage_flags_for_platform(get_material_tpageid(material), get_material_platform_name(material))
    except Exception:
        flags = _flags_from_panel_props(material)
    blend_mode = 0
    try:
        if int(flags.get('alpha_ref', 0)):
            blend_mode = 1
        elif int(flags.get('blend_value', 0)):
            blend_mode = 2
    except Exception:
        blend_mode = 0
    # Observed PC Next-Gen/TR7 PCD9 materials keep polyFlags at 0.
    poly_flags = 0

    layer_ids = [diffuse_id, normal_id, specular_id] + [-1] * (_PC_NEXTGEN_LAYER_COUNT - 3)
    layer_enabled = [1, 1 if normal_id >= 0 else 0, 1 if specular_id >= 0 else 0] + [0] * (_PC_NEXTGEN_LAYER_COUNT - 3)
    layer_colors = [(1.0, 1.0, 1.0, 1.0)] * _PC_NEXTGEN_LAYER_COUNT
    layer_constants = [(0.0, 0.0, 0.0, 0.0)] * _PC_NEXTGEN_LAYER_COUNT
    zeros = [0] * _PC_NEXTGEN_LAYER_COUNT

    _set_panel_prop(material, 'trlau_ui_material_platform', 'pc_nextgen')
    _set_panel_prop(material, 'trlau_ui_material_type', 'pc_nextgen')
    _set_panel_prop(material, 'trlau_ui_pcng_blend_mode', str(blend_mode))
    _set_panel_prop(material, 'trlau_ui_pcng_combiner_type', '0')
    default_pcng_flags = _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS_SPECULAR if int(normal_id) >= 0 and int(specular_id) >= 0 else _PC_NEXTGEN_DEFAULT_MATERIAL_FLAGS
    _set_panel_prop(material, 'trlau_ui_pcng_material_flags', f'0x{default_pcng_flags:08X}')
    _set_panel_prop(material, 'trlau_ui_pcng_opacity', 1.0)
    _set_panel_prop(material, 'trlau_ui_pcng_poly_flags', f'0x{poly_flags & 0xFFFFFFFF:08X}')
    _set_panel_prop(material, 'trlau_ui_pcng_uv_auto_scroll_speed', 0)
    _set_panel_prop(material, 'trlau_ui_pcng_sort_bias', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_detail_range_mul', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_detail_scale', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_parallax_scale', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_parallax_offset', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_specular_power', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_specular_shift0', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_specular_shift1', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_rim_light_color', (0.0, 0.0, 0.0, 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_rim_light_intensity', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_water_blend_bias', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_water_blend_exponent', 0.0)
    _set_panel_prop(material, 'trlau_ui_pcng_water_deep_color', (0.0, 0.0, 0.0, 0.0))
    _set_panel_prop(material, 'trlau_ui_pcng_special_material_flag', False)
    _set_panel_prop(material, 'trlau_ui_pcng_double_sided', bool(not int(flags.get('single_sided', 0))))
    default_shader_indices = _PC_NEXTGEN_DEFAULT_SHADER_INDICES_LAYERED if int(normal_id) >= 0 and int(specular_id) >= 0 else _PC_NEXTGEN_DEFAULT_SHADER_INDICES_DIFFUSE_ONLY
    for index in range(len(_PC_NEXTGEN_SHADER_INDEX_LABELS)):
        _set_panel_prop(material, f'trlau_ui_pcng_shader_index{index}', int(default_shader_indices[index]) if index < len(default_shader_indices) else 0)
    for index in range(_PC_NEXTGEN_LAYER_COUNT):
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_enabled', bool(layer_enabled[index]))
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_texture_id', int(layer_ids[index]))
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_texcoord_source', str(zeros[index]))
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_modifier', str(zeros[index]))
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_param_id', str(zeros[index]))
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_color', layer_colors[index])
        _set_panel_prop(material, f'trlau_ui_pcng_layer{index}_constant', layer_constants[index])
    sync_pc_nextgen_material_from_panel(material)


def convert_material_to_type(material, material_type: str, owner=None) -> None:
    global _MATERIAL_SYNC_GUARD
    if material is None:
        return
    material_type = str(material_type or 'pc_oldgen').lower()
    if material_type not in _MATERIAL_TYPE_VALUES:
        material_type = 'pc_oldgen'
    texture_id = _material_primary_texture_id(material)
    normal_id = _material_normal_texture_id(material)
    specular_id = _material_specular_texture_id(material)
    draw_group = get_material_draw_group(material)

    previous_guard = _MATERIAL_SYNC_GUARD
    _MATERIAL_SYNC_GUARD = True
    try:
        if material_type == 'pc_nextgen':
            initialize_pc_nextgen_material_from_current(material)
        else:
            _clear_pc_nextgen_metadata(material)
            platform = _MATERIAL_TYPE_TO_PLATFORM.get(material_type, 'pc')
            set_material_platform_name(material, platform)
            _set_panel_prop(material, 'trlau_ui_material_type', material_type)
            _set_panel_prop(material, 'trlau_ui_texture_id', int(texture_id) & _texture_id_mask_for_platform(platform))
            _set_panel_prop(material, 'trlau_ui_draw_group', int(draw_group))
            if material_type in {'ps3', 'xbox360'}:
                _set_panel_prop(material, 'trlau_ui_ps3_diffuse_texture_id', int(texture_id) & 0x1FFF)
                _set_panel_prop(material, 'trlau_ui_ps3_normal_texture_id', int(normal_id) if int(normal_id) >= 0 else -1)
                _set_panel_prop(material, 'trlau_ui_ps3_specular_texture_id', int(specular_id) if int(specular_id) >= 0 else -1)
                _set_panel_prop(material, 'trlau_ui_ps3_external_render_stream', bool(int(normal_id) >= 0))
            flags = _flags_from_panel_props(material)
            flags['texture_id'] = int(texture_id) & _texture_id_mask_for_platform(platform)
            _store_material_flags(material, flags)
    finally:
        _MATERIAL_SYNC_GUARD = previous_guard

    if material_type == 'pc_nextgen':
        apply_pc_nextgen_material_settings(material, owner=owner)
    elif material_type in {'ps3', 'xbox360'}:
        apply_ps3_material_settings(material, owner=owner)
    else:
        apply_trlau_material_settings(material, owner=owner)


def _update_material_type_setting(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    convert_material_to_type(self, getattr(self, 'trlau_ui_material_type', 'pc_oldgen'), owner=_material_owner_from_context(self, context))

def _material_owner_from_context(material, context=None):
    context = context or getattr(bpy, 'context', None)
    active_object = getattr(context, 'object', None) if context is not None else None
    if active_object is not None and getattr(active_object, 'type', None) == 'MESH':
        materials = getattr(getattr(active_object, 'data', None), 'materials', None)
        if materials is not None:
            try:
                if any(existing is material for existing in materials):
                    return active_object
            except Exception:
                pass
    return None


def _update_material_panel_settings(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    apply_trlau_material_settings(self, owner=_material_owner_from_context(self, context))


def _update_texture_id_panel_setting(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    texture_id = int(_get_panel_prop(self, 'trlau_ui_texture_id', 0)) & _texture_id_mask_for_material(self)
    if _is_texture_id_assignment_enabled():
        _assign_current_material_nodes(self, texture_id)
        image = _find_existing_image(self)
        if image is not None:
            try:
                image['trlau_texture_id'] = texture_id
            except Exception:
                pass
    apply_trlau_material_settings(self, owner=_material_owner_from_context(self, context))


def _update_scroll_panel_settings(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    apply_texture_scroll_settings(self)




def _update_tr8_material_panel_settings(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    try:
        double_sided = bool(getattr(self, 'trlau_ui_tr8_double_sided', False))
    except Exception:
        double_sided = False
    try:
        self['trlau_tr8_double_sided'] = bool(double_sided)
    except Exception:
        pass
    try:
        self.use_backface_culling = not bool(double_sided)
    except Exception:
        pass

def _update_ps3_material_panel_settings(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    apply_ps3_material_settings(self, owner=_material_owner_from_context(self, context))


def _update_pc_nextgen_material_panel_settings(self, context):
    if _MATERIAL_SYNC_GUARD:
        return
    ensure_pc_nextgen_material_panel_props(self, refresh_stale_defaults=False)
    _pc_nextgen_capture_current_layer_images(self)
    try:
        sync_pc_nextgen_material_from_panel(self)
        apply_pc_nextgen_material_settings(self, owner=_material_owner_from_context(self, context))
        _set_panel_prop(self, 'trlau_ui_pcng_last_update_error', '')
    except Exception as exc:
        try:
            _set_panel_prop(self, 'trlau_ui_pcng_last_update_error', str(exc))
        except Exception:
            pass


def register_material_properties():
    bpy.types.Material.trlau_ui_material_type = bpy.props.EnumProperty(
        name='Type',
        items=(
            ('pc_oldgen', 'PC Old-Gen', 'Legacy PC tpage material'),
            ('pc_nextgen', 'PC Next-Gen', 'PC next-generation PCMaterialData material'),
            ('ps2', 'PS2', 'PS2 / GS-state material layout'),
            ('psp', 'PSP', 'PSP material layout'),
            ('ps3', 'PS3', 'PS3 next-generation material layout'),
            ('xbox360', 'Xbox 360', 'Xbox 360 / Xenon material layout'),
        ),
        default='pc_oldgen',
        update=_update_material_type_setting,
    )
    bpy.types.Material.trlau_ui_material_platform = bpy.props.EnumProperty(
        name='TRLAU Platform',
        items=(
            ('pc', 'PC', ''),
            ('pc_nextgen', 'PC Next Gen', ''),
            ('ps2', 'PS2', ''),
            ('psp', 'PSP', ''),
            ('ps3', 'PS3', ''),
            ('xbox360', 'Xbox 360', ''),
        ),
        default='pc',
        options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_texture_id = bpy.props.IntProperty(
        name='Texture ID', min=0, max=0xFFFF, default=0, update=_update_texture_id_panel_setting,
    )
    bpy.types.Material.trlau_ui_blend_value = bpy.props.IntProperty(
        name='Blend', min=0, max=0xF, default=0, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_cull_mode = bpy.props.IntProperty(
        name='Cull Mode', min=0, max=0x7, default=0, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_unknown_1 = bpy.props.BoolProperty(
        name='Unknown 1', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_single_sided = bpy.props.BoolProperty(
        name='Single Sided', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_texture_wrap = bpy.props.IntProperty(
        name='Texture Wrap', min=0, max=0x3, default=0, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_ps2_double_sided = bpy.props.BoolProperty(
        name='Double Sided', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_unknown_2 = bpy.props.BoolProperty(
        name='Unknown 2', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_unknown_3 = bpy.props.BoolProperty(
        name='Unknown 3', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_flat_shading = bpy.props.BoolProperty(
        name='Flat Shading', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_sort_z = bpy.props.BoolProperty(
        name='Sort Z', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_stencil_pass = bpy.props.IntProperty(
        name='Stencil Pass', min=0, max=0x3, default=0, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_stencil_func = bpy.props.BoolProperty(
        name='Stencil Func', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_alpha_ref = bpy.props.BoolProperty(
        name='Alpha Ref', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_env_mapping = bpy.props.BoolProperty(
        name='Environment Mapping', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_eye_ref_env_mapping = bpy.props.BoolProperty(
        name='Eye Reflection Environment Mapping', default=False, update=_update_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_draw_group = bpy.props.IntProperty(
        name='Drawgroup', min=-32768, max=32767, default=0,
    )
    bpy.types.Material.trlau_ui_scroll_enabled = bpy.props.BoolProperty(
        name='Enabled', default=False, update=_update_scroll_panel_settings,
    )
    bpy.types.Material.trlau_ui_scroll_speed = bpy.props.FloatProperty(
        name='Speed', default=0.0, precision=4, update=_update_scroll_panel_settings,
    )
    bpy.types.Material.trlau_ui_tr8_double_sided = bpy.props.BoolProperty(
        name='Double Sided', default=False, update=_update_tr8_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_ps3_diffuse_texture_id = bpy.props.IntProperty(
        name='Diffuse Texture ID', min=0, max=0x1FFF, default=0, update=_update_ps3_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_ps3_normal_texture_id = bpy.props.IntProperty(
        name='Normal Texture ID', min=-1, max=0x1FFF, default=-1, update=_update_ps3_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_ps3_normal_candidate_texture_id = bpy.props.IntProperty(
        name='TRLAU Normal Candidate Texture ID', min=-1, max=0x1FFF, default=-1, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_ps3_specular_texture_id = bpy.props.IntProperty(
        name='Specular Texture ID', min=-1, max=0x1FFF, default=-1, update=_update_ps3_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_ps3_external_render_stream = bpy.props.BoolProperty(
        name='TRLAU External Render Stream', default=False, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_ps3_role_mapping_version = bpy.props.IntProperty(
        name='TRLAU Role Mapping Version', default=4, options={'HIDDEN'},
    )

    bpy.types.Material.trlau_ui_pcng_material_record_offset = bpy.props.IntProperty(
        name='Material Record Offset', default=0, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_material_record_hex = bpy.props.StringProperty(
        name='Material Record Hex', default='', options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_asset_id_padding_hex = bpy.props.StringProperty(
        name='Asset ID Padding Hex', default='', options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_shader_table_ids = bpy.props.StringProperty(
        name='Shader Table IDs', default='', options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_double_wound_pair_count = bpy.props.IntProperty(
        name='Double Wound Pair Count', default=0, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_panel_sync_version = bpy.props.IntProperty(
        name='PCNG Panel Sync Version', default=0, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_layer_image_names = bpy.props.StringProperty(
        name='PCNG Layer Image Names', default='', options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_last_update_error = bpy.props.StringProperty(
        name='PCNG Last Update Error', default='', options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_blend_mode = bpy.props.EnumProperty(
        name='Blend Mode', items=_PC_NEXTGEN_BLEND_MODE_ITEMS, default='0', update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_combiner_type = bpy.props.EnumProperty(
        name='Combiner Type', items=_PC_NEXTGEN_COMBINER_TYPE_ITEMS, default='0', update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_material_flags = bpy.props.StringProperty(
        name='Material Flags', default='0x00000000', update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_opacity = bpy.props.FloatProperty(
        name='Opacity', min=0.0, max=1.0, default=1.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_poly_flags = bpy.props.StringProperty(
        name='Poly Flags', default='0x00000000', update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_uv_auto_scroll_speed = bpy.props.IntProperty(
        name='UV Auto Scroll Speed', min=0, max=0xFFFF, default=0, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_sort_bias = bpy.props.FloatProperty(
        name='Sort Bias', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_detail_range_mul = bpy.props.FloatProperty(
        name='Detail Range Mul', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_detail_scale = bpy.props.FloatProperty(
        name='Detail Scale', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_parallax_scale = bpy.props.FloatProperty(
        name='Parallax Scale', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_parallax_offset = bpy.props.FloatProperty(
        name='Parallax Offset', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_specular_power = bpy.props.FloatProperty(
        name='Specular Power', min=0.0, default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_specular_shift0 = bpy.props.FloatProperty(
        name='Specular Shift 0', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_specular_shift1 = bpy.props.FloatProperty(
        name='Specular Shift 1', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_rim_light_color = bpy.props.FloatVectorProperty(
        name='Rim Light Color', size=4, subtype='COLOR', min=0.0, max=16.0, default=(0.0, 0.0, 0.0, 0.0), update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_rim_light_intensity = bpy.props.FloatProperty(
        name='Rim Light Intensity', min=0.0, default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_water_blend_bias = bpy.props.FloatProperty(
        name='Water Blend Bias', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_water_blend_exponent = bpy.props.FloatProperty(
        name='Water Blend Exponent', default=0.0, precision=4, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_water_deep_color = bpy.props.FloatVectorProperty(
        name='Water Deep Color', size=4, subtype='COLOR', min=0.0, max=16.0, default=(0.0, 0.0, 0.0, 0.0), update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_special_material_flag = bpy.props.BoolProperty(
        name='Special Material Flag', default=False, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_double_sided = bpy.props.BoolProperty(
        name='Double Sided', default=False, update=_update_pc_nextgen_material_panel_settings,
    )
    bpy.types.Material.trlau_ui_pcng_diffuse_texture_id = bpy.props.IntProperty(
        name='Derived Diffuse Texture ID', min=-1, max=0x1FFF, default=-1, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_normal_texture_id = bpy.props.IntProperty(
        name='Derived Normal Texture ID', min=-1, max=0x1FFF, default=-1, options={'HIDDEN'},
    )
    bpy.types.Material.trlau_ui_pcng_specular_texture_id = bpy.props.IntProperty(
        name='Derived Specular Texture ID', min=-1, max=0x1FFF, default=-1, options={'HIDDEN'},
    )
    for index, label in enumerate(_PC_NEXTGEN_SHADER_INDEX_LABELS):
        setattr(bpy.types.Material, f'trlau_ui_pcng_shader_index{index}', bpy.props.IntProperty(
            name=label, min=0, default=0, update=_update_pc_nextgen_material_panel_settings,
        ))
    for index in range(_PC_NEXTGEN_LAYER_COUNT):
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_enabled', bpy.props.BoolProperty(
            name='Enabled', default=False, update=_update_pc_nextgen_material_panel_settings,
        ))
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_texture_id', bpy.props.IntProperty(
            name='Texture ID', min=-1, max=0x1FFF, default=-1, update=_update_pc_nextgen_material_panel_settings,
        ))
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_texcoord_source', bpy.props.EnumProperty(
            name='TexCoord Source', items=_PC_NEXTGEN_TEXCOORD_SOURCE_ITEMS, default='0', update=_update_pc_nextgen_material_panel_settings,
        ))
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_modifier', bpy.props.EnumProperty(
            name='Modifier', items=_PC_NEXTGEN_TEXCOORD_MODIFIER_ITEMS, default='0', update=_update_pc_nextgen_material_panel_settings,
        ))
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_param_id', bpy.props.EnumProperty(
            name='Param ID', items=_PC_NEXTGEN_PARAM_ID_ITEMS, default='0', update=_update_pc_nextgen_material_panel_settings,
        ))
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_color', bpy.props.FloatVectorProperty(
            name='Color', size=4, subtype='COLOR', min=0.0, max=16.0, default=(1.0, 1.0, 1.0, 1.0), update=_update_pc_nextgen_material_panel_settings,
        ))
        setattr(bpy.types.Material, f'trlau_ui_pcng_layer{index}_constant', bpy.props.FloatVectorProperty(
            name='Constant', size=4, min=-1024.0, max=1024.0, default=(0.0, 0.0, 0.0, 0.0), update=_update_pc_nextgen_material_panel_settings,
        ))

    cleanup_all_material_metadata()

def unregister_material_properties():
    property_names = (
        'trlau_ui_material_type',
        'trlau_ui_material_platform',
        'trlau_ui_texture_id',
        'trlau_ui_blend_value',
        'trlau_ui_cull_mode',
        'trlau_ui_unknown_1',
        'trlau_ui_single_sided',
        'trlau_ui_texture_wrap',
        'trlau_ui_ps2_double_sided',
        'trlau_ui_unknown_2',
        'trlau_ui_unknown_3',
        'trlau_ui_flat_shading',
        'trlau_ui_sort_z',
        'trlau_ui_stencil_pass',
        'trlau_ui_stencil_func',
        'trlau_ui_alpha_ref',
        'trlau_ui_env_mapping',
        'trlau_ui_eye_ref_env_mapping',
        'trlau_ui_draw_group',
        'trlau_ui_scroll_enabled',
        'trlau_ui_scroll_speed',
        'trlau_ui_tr8_double_sided',
        'trlau_ui_ps3_diffuse_texture_id',
        'trlau_ui_ps3_normal_texture_id',
        'trlau_ui_ps3_specular_texture_id',
        'trlau_ui_ps3_normal_candidate_texture_id',
        'trlau_ui_ps3_external_render_stream',
        'trlau_ui_ps3_role_mapping_version',
        'trlau_ui_ps3_normal_mode',
        'trlau_ui_ps3_specular_source_mode',
        'trlau_ui_ps3_normal_strength',
        'trlau_ui_ps3_bump_strength',
        'trlau_ui_ps3_bump_distance',
        'trlau_ui_ps3_specular_strength',
        'trlau_ui_ps3_roughness_dark',
        'trlau_ui_ps3_roughness_bright',
        'trlau_ui_pcng_material_record_offset',
        'trlau_ui_pcng_material_record_hex',
        'trlau_ui_pcng_asset_id_padding_hex',
        'trlau_ui_pcng_shader_table_ids',
        'trlau_ui_pcng_double_wound_pair_count',
        'trlau_ui_pcng_panel_sync_version',
        'trlau_ui_pcng_layer_image_names',
        'trlau_ui_pcng_last_update_error',
        'trlau_ui_pcng_blend_mode',
        'trlau_ui_pcng_combiner_type',
        'trlau_ui_pcng_material_flags',
        'trlau_ui_pcng_opacity',
        'trlau_ui_pcng_poly_flags',
        'trlau_ui_pcng_uv_auto_scroll_speed',
        'trlau_ui_pcng_sort_bias',
        'trlau_ui_pcng_detail_range_mul',
        'trlau_ui_pcng_detail_scale',
        'trlau_ui_pcng_parallax_scale',
        'trlau_ui_pcng_parallax_offset',
        'trlau_ui_pcng_specular_power',
        'trlau_ui_pcng_specular_shift0',
        'trlau_ui_pcng_specular_shift1',
        'trlau_ui_pcng_rim_light_color',
        'trlau_ui_pcng_rim_light_intensity',
        'trlau_ui_pcng_water_blend_bias',
        'trlau_ui_pcng_water_blend_exponent',
        'trlau_ui_pcng_water_deep_color',
        'trlau_ui_pcng_special_material_flag',
        'trlau_ui_pcng_double_sided',
        'trlau_ui_pcng_diffuse_texture_id',
        'trlau_ui_pcng_normal_texture_id',
        'trlau_ui_pcng_specular_texture_id',
    )
    property_names = tuple(property_names) + tuple(
        f'trlau_ui_pcng_shader_index{index}' for index in range(len(_PC_NEXTGEN_SHADER_INDEX_LABELS))
    ) + tuple(
        f'trlau_ui_pcng_layer{layer}_{suffix}'
        for layer in range(_PC_NEXTGEN_LAYER_COUNT)
        for suffix in (
            'enabled',
            'texture_id',
            'texcoord_source',
            'modifier',
            'param_id',
            'color',
            'constant',
        )
    )
    for name in property_names:
        if hasattr(bpy.types.Material, name):
            delattr(bpy.types.Material, name)
