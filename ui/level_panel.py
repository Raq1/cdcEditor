from __future__ import annotations

from pathlib import Path
import re

import bmesh
import bpy
import mathutils
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty

from ..core.hinfo_ui import PANEL_TYPES
from ..core.object_utils import trlau_object_type
from ..core.fsfx_render_effect import (
    FSFX_PROP_PREFIX,
    decode_stored_section_blob_from_object,
    fsfx_get_table_value,
    fsfx_set_table_value,
    fsfx_first_table_identifier,
    fsfx_normalize_table_identifier,
    fsfx_table_label,
    fsfx_table_summary,
    fsfx_table_summary_for_identifier,
    fsfx_table_ui_items,
    fsfx_u32_colour_to_rgb,
    fsfx_rgb_to_u32,
    has_fsfx_render_effect_properties,
    store_fsfx_render_effect_properties,
)
from . import model_panel
from .hinfo_panel import _draw_hinfo_creation_content, _draw_hinfo_details_content, _has_hinfo_creation_context
from .level_properties import (
    ADMD_DATA_PROP_PREFIX,
    INTRO_GENERIC_FIELDS,
    INTRO_SOUND_SFX_FIELDS,
    INTRO_SPECIFIC_FIELDS,
    SFX_MARKER_FIELDS,
    SFX_REFERENCE_FIELDS,
    SFX_SOUND_BASE_FIELDS,
    SFX_SOUND_EVENT_FIELDS,
    SFX_SOUND_PERIODIC_FIELDS,
    SFX_SOUND_STREAM_FIELDS,
    SFX_SPEAKER_FIELDS,
    WAVE_METADATA_FIELDS,
    LEVEL_GAME_ITEMS,
    LEVEL_METADATA_GROUPS,
    LEVEL_METADATA_PROP_PREFIX,
    MARKUP_BBOX_FLAGS,
    MARKUP_FLAG_LEDGE,
    MARKUP_TYPE_DEFS,
    MARKUP_TYPE_ITEMS,
    UNIT_DATA_PROP_PREFIX,
    _find_related_level_root,
    _level_game_for_object,
    _normalize_level_game_value,
    _next_index_for_prop,
    _next_unique_int_for_prop,
    _unitdata_field_visible_for_game,
    _write_int_prop,
)


FSFX_TABLE_PAGE_SIZE = 16
FSFX_COLOUR_UI_FIELDS = (
    ('colour_dof_colour', 'm_nColour'),
    ('noise_colour', 'm_nColour'),
    ('feedback_colour', 'm_nColour'),
    ('water_colour', 'm_nColour'),
    ('heat_haze_colour', 'm_nColour'),
)

FSFX_SECTION_ROLES = {'FSFXLink', 'WaterFSFX'}



def _fsfx_colour_prop_name(suffix: str) -> str:
    return f'trlau_fsfx_{suffix}_ui'


def _get_fsfx_colour_value(obj, suffix: str):
    return fsfx_u32_colour_to_rgb(obj.get(f'{FSFX_PROP_PREFIX}{suffix}', 0))


def _set_fsfx_colour_value(obj, suffix: str, value) -> None:
    key = f'{FSFX_PROP_PREFIX}{suffix}'
    obj[key] = fsfx_rgb_to_u32(value, obj.get(key, 0))


def _make_fsfx_colour_getter(suffix: str):
    def getter(self):
        return _get_fsfx_colour_value(self, suffix)
    return getter


def _make_fsfx_colour_setter(suffix: str):
    def setter(self, value):
        _set_fsfx_colour_value(self, suffix, value)
    return setter


def _fsfx_table_enum_items(self, context):
    return fsfx_table_ui_items()


def _fsfx_active_table_identifier(obj) -> str:
    try:
        identifier = str(getattr(obj, 'trlau_fsfx_table_selector', '') or '')
    except Exception:
        identifier = ''
    return fsfx_normalize_table_identifier(obj, identifier)


def _get_fsfx_table_page(obj) -> int:
    try:
        page = int(getattr(obj, 'trlau_fsfx_table_page', 0) or 0)
    except Exception:
        page = 0
    return max(0, min(15, page))


def _make_fsfx_table_entry_getter(slot_index: int):
    def getter(self):
        return fsfx_get_table_value(self, _fsfx_active_table_identifier(self), _get_fsfx_table_page(self) * FSFX_TABLE_PAGE_SIZE + int(slot_index))
    return getter


def _make_fsfx_table_entry_setter(slot_index: int):
    def setter(self, value):
        fsfx_set_table_value(self, _fsfx_active_table_identifier(self), _get_fsfx_table_page(self) * FSFX_TABLE_PAGE_SIZE + int(slot_index), value)
    return setter


def register_fsfx_panel_properties():
    bpy.types.Object.trlau_fsfx_table_selector = EnumProperty(
        name='Table',
        description='RenderFSEffect 256-byte curve/palette table to edit',
        items=_fsfx_table_enum_items,
        options={'SKIP_SAVE'},
    )
    bpy.types.Object.trlau_fsfx_table_page = IntProperty(
        name='Page',
        description='Table page. Each page exposes 16 of the 256 byte entries.',
        min=0,
        max=15,
        default=0,
        options={'SKIP_SAVE'},
    )
    for slot_index in range(FSFX_TABLE_PAGE_SIZE):
        setattr(
            bpy.types.Object,
            f'trlau_fsfx_table_value_{slot_index:02d}',
            IntProperty(
                name=f'{slot_index:02d}',
                description='Selected RenderFSEffect table byte value',
                min=0,
                max=255,
                get=_make_fsfx_table_entry_getter(slot_index),
                set=_make_fsfx_table_entry_setter(slot_index),
            ),
        )
    for suffix, _label in FSFX_COLOUR_UI_FIELDS:
        setattr(
            bpy.types.Object,
            _fsfx_colour_prop_name(suffix),
            FloatVectorProperty(
                name='Colour',
                description='RGB colour packed back into the original 32-bit FSFX colour value; the high byte is preserved',
                subtype='COLOR',
                size=3,
                min=0.0,
                max=1.0,
                get=_make_fsfx_colour_getter(suffix),
                set=_make_fsfx_colour_setter(suffix),
            ),
        )


def unregister_fsfx_panel_properties():
    for suffix, _label in FSFX_COLOUR_UI_FIELDS:
        prop_name = _fsfx_colour_prop_name(suffix)
        if hasattr(bpy.types.Object, prop_name):
            delattr(bpy.types.Object, prop_name)
    for slot_index in range(FSFX_TABLE_PAGE_SIZE):
        prop_name = f'trlau_fsfx_table_value_{slot_index:02d}'
        if hasattr(bpy.types.Object, prop_name):
            delattr(bpy.types.Object, prop_name)
    if hasattr(bpy.types.Object, 'trlau_fsfx_table_page'):
        del bpy.types.Object.trlau_fsfx_table_page
    if hasattr(bpy.types.Object, 'trlau_fsfx_table_selector'):
        del bpy.types.Object.trlau_fsfx_table_selector


SFX_PANEL_STREAM_FILENAME_SLOTS = 256
SFX_PANEL_STRING_NAMES = {
    'trlau_sfx_role',
    'trlau_sfx_kind',
    'trlau_sfx_ids',
    'trlau_sfx_resolved_json',
    'trlau_sfx_name',
    'trlau_sfx_music_vars',
    'trlau_sfx_ref_section_file',
    'trlau_sfx_ref_section_type',
    'trlau_sfx_ref_filename',
    'trlau_wave_wave_section_file',
    'trlau_wave_decoded_wav_path',
    'trlau_sfx_filename',
}
SFX_PANEL_VECTOR_NAMES = {
    'trlau_sfx_marker_raw_position',
}
SFX_PANEL_FLOAT_NAMES = {
    'trlau_sfx_note',
    'trlau_sfx_pitch',
    'trlau_sfx_pitch_variation',
    'trlau_sfx_initial_delay',
    'trlau_sfx_initial_delay_variation',
    'trlau_sfx_on_time',
    'trlau_sfx_on_time_variation',
    'trlau_sfx_off_time',
    'trlau_sfx_off_time_variation',
    'trlau_sfx_delay_variation',
    'trlau_sfx_choose_chance',
    'trlau_sfx_play_chance',
    'trlau_sfx_max_volume',
    'trlau_sfx_max_vol_variation',
    'trlau_sfx_ref_note',
    'trlau_sfx_ref_pitch_variation',
    'trlau_sfx_ref_initial_delay',
    'trlau_sfx_ref_initial_delay_variation',
    'trlau_sfx_ref_group_weight',
}


def _sfx_panel_property_specs():
    names: dict[str, str] = {
        'trlau_sfx_role': 'SFX Role',
        'trlau_sfx_marker_count': 'markerCount',
        'trlau_sfx_marker_sound_instance_offset': 'soundInstanceOffset',
        'trlau_sfx_resolved_json': 'resolvedJSON',
        'trlau_sfx_filename': 'filename',
        'trlau_sfx_name_offset': 'nameOffset',
        'trlau_sfx_name_has_relocation': 'nameHasRelocation',
    }
    for attr, label in SFX_MARKER_FIELDS:
        names[f'trlau_sfx_{attr}'] = label
    for attr, label in SFX_SOUND_BASE_FIELDS:
        names[f'trlau_sfx_{attr}'] = label
    for fields in (SFX_SOUND_PERIODIC_FIELDS, SFX_SOUND_EVENT_FIELDS, SFX_SOUND_STREAM_FIELDS, SFX_SPEAKER_FIELDS):
        for attr, label in fields:
            names[f'trlau_sfx_{attr}'] = label
    for attr, label in SFX_REFERENCE_FIELDS:
        names[f'trlau_sfx_ref_{attr}'] = label
    for attr, label in WAVE_METADATA_FIELDS:
        names[f'trlau_wave_{attr}'] = label
    for attr, label in INTRO_SOUND_SFX_FIELDS:
        names[f'trlau_sfx_{attr}'] = label
    for index in range(SFX_PANEL_STREAM_FILENAME_SLOTS):
        names[f'trlau_sfx_stream_filename_{index:02d}'] = f'streamFilename{index:02d}'
    return names


SFX_PANEL_PROPERTY_NAMES = tuple(_sfx_panel_property_specs().keys())


def register_sfx_panel_properties():
    for prop_name, label in _sfx_panel_property_specs().items():
        if hasattr(bpy.types.Object, prop_name):
            continue
        if prop_name in SFX_PANEL_VECTOR_NAMES:
            setattr(
                bpy.types.Object,
                prop_name,
                FloatVectorProperty(name=label, size=3, default=(0.0, 0.0, 0.0), options={'HIDDEN'}),
            )
        elif prop_name in SFX_PANEL_STRING_NAMES or prop_name.startswith('trlau_sfx_stream_filename_'):
            setattr(
                bpy.types.Object,
                prop_name,
                StringProperty(name=label, default='', options={'HIDDEN'}),
            )
        elif prop_name in SFX_PANEL_FLOAT_NAMES:
            setattr(
                bpy.types.Object,
                prop_name,
                FloatProperty(name=label, default=0.0, options={'HIDDEN'}),
            )
        else:
            setattr(
                bpy.types.Object,
                prop_name,
                IntProperty(name=label, default=0, options={'HIDDEN'}),
            )


def unregister_sfx_panel_properties():
    for prop_name in reversed(SFX_PANEL_PROPERTY_NAMES):
        if hasattr(bpy.types.Object, prop_name):
            delattr(bpy.types.Object, prop_name)




MULTISPLINE_TARGET_ITEMS = (
    ('positional', 'Positional', 'Spline positional keys'),
    ('rotational', 'Rotational', 'RSpline rotational keys'),
    ('scaling', 'Scaling', 'Spline scaling keys'),
)


def _identity_matrix16() -> tuple[float, ...]:
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def _float_seq(values, size: int, default) -> tuple[float, ...]:
    try:
        seq = list(values or [])
    except Exception:
        seq = []
    out: list[float] = []
    for index in range(int(size)):
        try:
            value = seq[index]
        except Exception:
            value = default[index] if index < len(default) else 0.0
        try:
            value = float(value)
        except Exception:
            value = float(default[index] if index < len(default) else 0.0)
        out.append(value)
    return tuple(out)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class TRLAU_PG_multispline_key(bpy.types.PropertyGroup):
    point: FloatVectorProperty(name='point', size=4, default=(0.0, 0.0, 0.0, 0.0))
    d1: FloatVectorProperty(name='d1', size=4, default=(0.0, 0.0, 0.0, 0.0))
    d2: FloatVectorProperty(name='d2', size=4, default=(0.0, 0.0, 0.0, 0.0))
    q: FloatVectorProperty(name='q', size=4, default=(0.0, 0.0, 0.0, 1.0))
    t0: FloatProperty(name='t0', default=0.0)
    tf: FloatProperty(name='tf', default=0.0)
    count: FloatProperty(name='count', default=0.0)
    invCount: FloatProperty(name='invCount', default=0.0)


class TRLAU_PG_multispline_spline(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Enabled', default=False)
    type: IntProperty(name='type', min=0, max=255, default=0)
    flags: IntProperty(name='flags', min=0, max=255, default=0)
    count: FloatProperty(name='count', default=0.0)
    source_offset: IntProperty(name='sourceOffset', default=0)
    active_key_index: IntProperty(name='Key', min=0, default=0, options={'SKIP_SAVE'})
    keys: CollectionProperty(type=TRLAU_PG_multispline_key)


class TRLAU_PG_multispline_state(bpy.types.PropertyGroup):
    t: FloatProperty(name='t', default=0.0)
    curr_key: IntProperty(name='currKey', min=0, max=65535, default=0)
    pad: IntProperty(name='pad', min=0, max=65535, default=0)


class TRLAU_PG_multispline(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Has MultiSpline', default=False)
    source_offset: IntProperty(name='sourceOffset', default=0)
    source_raw_value: IntProperty(name='sourceRawValue', default=0)
    cur_rot_matrix: FloatVectorProperty(name='curRotMatrix', size=16, default=_identity_matrix16())
    positional: PointerProperty(type=TRLAU_PG_multispline_spline)
    rotational: PointerProperty(type=TRLAU_PG_multispline_spline)
    scaling: PointerProperty(type=TRLAU_PG_multispline_spline)
    cur_positional: PointerProperty(type=TRLAU_PG_multispline_state)
    cur_rotational: PointerProperty(type=TRLAU_PG_multispline_state)
    cur_scaling: PointerProperty(type=TRLAU_PG_multispline_state)


class TRLAU_PG_markup_data(bpy.types.PropertyGroup):
    intro_id: IntProperty(
        name='Intro ID',
        description='MarkUp introID / linked water-volume IntroData index',
        default=0,
        min=-32768,
        max=32767,
    )


class TRLAU_PG_intro_data(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Enabled', default=False)
    index: IntProperty(name='index', default=-1)
    object_id: IntProperty(name='objectID', default=-1)
    intro_num: IntProperty(name='introNum', default=0)
    unique_id: IntProperty(name='uniqueID', default=0)
    max_radius: FloatProperty(name='maxRad', default=0.0, min=0.0)
    intro_flags: StringProperty(name='introFlags', default='0')
    start_frame: IntProperty(name='startFrame', default=0)
    end_frame: IntProperty(name='endFrame', default=0)
    attached_vmo: IntProperty(name='attachedVMO', default=0)
    multi_spline: IntProperty(name='multiSpline', default=0)
    position: FloatVectorProperty(name='position', size=3, default=(0.0, 0.0, 0.0))
    rotation: FloatVectorProperty(name='rotation', size=4, default=(0.0, 0.0, 0.0, 0.0))
    rotation_mode: StringProperty(name='rotationMode', default='XYZ')
    rotation_basis: StringProperty(name='rotationBasis', default='RAW_XYZ')
    scale: FloatVectorProperty(name='scale', size=4, default=(1.0, 1.0, 1.0, 0.0))
    uses_player_object_id: BoolProperty(name='usesPlayerObjectID', default=False)
    player_object_id: IntProperty(name='playerObjectID', default=-1)


class TRLAU_PG_intro_generic_data(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Enabled', default=False)
    parent_index: IntProperty(name='parentIndex', default=-1)
    in_view_remove_dist: FloatProperty(name='InViewRemoveDist', default=0.0)
    out_of_view_remove_dist: FloatProperty(name='OutOfViewRemoveDist', default=0.0)
    use_model: IntProperty(name='UseModel', default=0)
    pad0: IntProperty(name='pad0', default=0)
    pad1: IntProperty(name='pad1', default=0)
    flags: IntProperty(name='Flags', default=0)
    attached_instance: IntProperty(name='AttachedInstance', default=0)
    swing_length: FloatProperty(name='SwingLength', default=0.0)
    dtp_camera_id: IntProperty(name='DTPCameraID', default=0)


class TRLAU_PG_intro_specific_data(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Enabled', default=False)
    parent_index: IntProperty(name='parentIndex', default=-1)
    data_type: IntProperty(name='type', default=0)
    struct_name: StringProperty(name='struct', default='')

    reward_type: IntProperty(name='rewardType', default=0, min=0, max=3)
    unique_id: IntProperty(name='uniqueID', default=0)
    sound_id: IntProperty(name='soundID', default=0)

    top_connect_type: IntProperty(name='topConnectType', default=0)
    bottom_connect_type: IntProperty(name='bottomConnectType', default=0)
    top_connect_instance: IntProperty(name='topConnectInstance', default=0)
    top_connect_model_index: IntProperty(name='topConnectModelIndex', default=0)
    top_connect_model_marker_index: IntProperty(name='topConnectModelMarkerIndex', default=0)
    bottom_connect_instance: IntProperty(name='bottomConnectInstance', default=0)
    bottom_connect_model_index: IntProperty(name='bottomConnectModelIndex', default=0)
    bottom_connect_model_marker_index: IntProperty(name='bottomConnectModelMarkerIndex', default=0)
    collision_plane_instance0: IntProperty(name='collisionPlaneInstance0', default=0)
    collision_plane_instance1: IntProperty(name='collisionPlaneInstance1', default=0)
    rope_camera_dtpid: IntProperty(name='ropeCameraDTPID', default=0)
    rope_camera_overrides_movement: IntProperty(name='ropeCameraOverridesMovement', default=0)
    sound_input_min: FloatProperty(name='soundInputMin', default=0.0)
    sound_input_max: FloatProperty(name='soundInputMax', default=0.0)
    render_width: FloatProperty(name='renderWidth', default=0.0)
    render_length_per_v: FloatProperty(name='renderLengthPerV', default=0.0)
    render_u_width: FloatProperty(name='renderUWidth', default=0.0)
    render_color: StringProperty(name='renderColor', default='0')
    render_model: IntProperty(name='renderModel', default=0)
    render_texture_main: IntProperty(name='renderTextureMain', default=0)
    render_texture_top: IntProperty(name='renderTextureTop', default=0)
    render_texture_bottom: IntProperty(name='renderTextureBottom', default=0)

    water_depth: FloatProperty(name='waterDepth', default=0.0)
    water_inflow: IntProperty(name='waterInflow', default=0)
    water_outflow: IntProperty(name='waterOutflow', default=0)
    water_speed: FloatProperty(name='waterSpeed', default=0.0)
    flow_radius: FloatProperty(name='flowRadius', default=0.0)
    bob_height: IntProperty(name='bobHeight', default=0)
    bob_frequency: IntProperty(name='bobFrequency', default=0)
    bob_grid_size: FloatProperty(name='bobGridSize', default=0.0)
    priority: IntProperty(name='priority', default=0)
    water_flags: IntProperty(name='waterFlags', default=0, min=0, max=65535)


class TRLAU_PG_intro_sound_data(bpy.types.PropertyGroup):
    enabled: BoolProperty(name='Enabled', default=False)
    parent_index: IntProperty(name='parentIndex', default=-1)
    sound_id: IntProperty(name='soundID / sectionID', default=0)
    format: StringProperty(name='format', default='')
    file_name: StringProperty(name='fileName', default='')
    section_type: IntProperty(name='sectionType', default=0)
    skip_flags: IntProperty(name='skipFlags', default=0)
    version_id: IntProperty(name='versionID', default=0)
    has_debug_info: IntProperty(name='hasDebugInfo', default=0)
    resource_type: IntProperty(name='resourceType', default=0)
    spec_mask: StringProperty(name='specMask', default='4294967295')
    num_relocations: IntProperty(name='numRelocations', default=0)
    standalone_size: IntProperty(name='standaloneSize', default=0)


_INTRO_SPECIFIC_TYPE_NAME_BY_ID = {
    12: 'RopeObjIntroData',
    13: 'WaterVolumeIntroData',
    17: 'RewardIntroData',
}
_INTRO_SPECIFIC_TYPE_ID_BY_NAME = {value: key for key, value in _INTRO_SPECIFIC_TYPE_NAME_BY_ID.items()}
def _assign_pg_value(group, attr: str, value) -> None:
    if group is None or not hasattr(group, attr):
        return
    try:
        prop = group.bl_rna.properties.get(attr)
        if prop is not None and prop.type == 'STRING':
            setattr(group, attr, str(value if value is not None else ''))
            return
    except Exception:
        pass
    try:
        setattr(group, attr, value)
    except Exception:
        try:
            setattr(group, attr, _safe_int(value, 0))
        except Exception:
            try:
                setattr(group, attr, _safe_float(value, 0.0))
            except Exception:
                pass


def _assign_pg_vector(group, attr: str, value, size: int, default) -> None:
    try:
        setattr(group, attr, _float_seq(value, size, default))
    except Exception:
        pass


def _intro_data_group(obj):
    return getattr(obj, 'trlau_intro_data', None) if obj is not None else None


def _intro_generic_group(obj):
    return getattr(obj, 'trlau_intro_generic_data', None) if obj is not None else None


def _intro_specific_group(obj):
    return getattr(obj, 'trlau_intro_specific_data', None) if obj is not None else None


def _intro_sound_group(obj):
    return getattr(obj, 'trlau_intro_sound_data', None) if obj is not None else None


def trlau_set_intro_data_property(obj, **values) -> None:
    data = _intro_data_group(obj)
    if data is None:
        return
    data.enabled = True
    for attr, value in values.items():
        if attr == 'position':
            _assign_pg_vector(data, attr, value, 3, (0.0, 0.0, 0.0))
        elif attr in {'rotation', 'scale'}:
            default = (0.0, 0.0, 0.0, 0.0) if attr == 'rotation' else (1.0, 1.0, 1.0, 0.0)
            _assign_pg_vector(data, attr, value, 4, default)
        else:
            _assign_pg_value(data, attr, value)


def trlau_set_intro_generic_property(obj, *, parent_index: int = -1, values: dict | None = None) -> None:
    data = _intro_generic_group(obj)
    if data is None:
        return
    data.enabled = True
    data.parent_index = int(parent_index)
    for attr, _label in INTRO_GENERIC_FIELDS:
        if values is not None and attr in values:
            _assign_pg_value(data, attr, values.get(attr))


def trlau_set_intro_specific_property(obj, *, parent_index: int = -1, data_type: int = 0, struct_name: str = '', values: dict | None = None) -> None:
    data = _intro_specific_group(obj)
    if data is None:
        return
    data.enabled = True
    data.parent_index = int(parent_index)
    data.data_type = int(data_type or 0)
    data.struct_name = str(struct_name or _INTRO_SPECIFIC_TYPE_NAME_BY_ID.get(int(data_type or 0), f'IntroType{int(data_type or 0)}Data'))
    for attr, _label in INTRO_SPECIFIC_FIELDS.get(data.struct_name, ()):  # known structured types
        if values is not None and attr in values:
            _assign_pg_value(data, attr, values.get(attr))


def trlau_set_intro_sound_property(obj, **values) -> None:
    data = _intro_sound_group(obj)
    if data is None:
        return
    data.enabled = True
    for attr, value in values.items():
        _assign_pg_value(data, attr, value)


def _intro_data_for_ui(obj):
    return _intro_data_group(obj)


def _intro_generic_data_for_ui(obj):
    return _intro_generic_group(obj)


def _intro_specific_data_for_ui(obj):
    return _intro_specific_group(obj)


def _intro_sound_data_for_ui(obj):
    return _intro_sound_group(obj)


def trlau_intro_data_to_dict(obj) -> dict[str, object]:
    data = _intro_data_group(obj)
    if data is None or not bool(getattr(data, 'enabled', False)):
        return {}
    return {
        'index': int(getattr(data, 'index', -1) or -1),
        'object_id': int(getattr(data, 'object_id', -1) or -1),
        'intro_num': int(getattr(data, 'intro_num', 0) or 0),
        'unique_id': int(getattr(data, 'unique_id', 0) or 0),
        'start_frame': int(getattr(data, 'start_frame', 0) or 0),
        'end_frame': int(getattr(data, 'end_frame', 0) or 0),
        'intro_flags': _safe_int(getattr(data, 'intro_flags', 0), 0),
        'attached_vmo': int(getattr(data, 'attached_vmo', 0) or 0),
        'multi_spline': int(getattr(data, 'multi_spline', 0) or 0),
        'max_radius': float(getattr(data, 'max_radius', 0.0) or 0.0),
        'position': tuple(float(v) for v in getattr(data, 'position', (0.0, 0.0, 0.0))),
        'rotation': tuple(float(v) for v in getattr(data, 'rotation', (0.0, 0.0, 0.0, 0.0))),
        'scale': tuple(float(v) for v in getattr(data, 'scale', (1.0, 1.0, 1.0, 0.0))),
        'rotation_mode': str(getattr(data, 'rotation_mode', '') or ''),
        'rotation_basis': str(getattr(data, 'rotation_basis', '') or ''),
        'uses_player_object_id': bool(getattr(data, 'uses_player_object_id', False)),
        'player_object_id': int(getattr(data, 'player_object_id', -1) or -1),
    }


def trlau_intro_generic_data_to_dict(obj) -> dict[str, object]:
    data = _intro_generic_group(obj)
    if data is None or not bool(getattr(data, 'enabled', False)):
        return {}
    result: dict[str, object] = {}
    for attr, _label in INTRO_GENERIC_FIELDS:
        value = getattr(data, attr, 0)
        result[attr] = float(value) if isinstance(value, float) else int(value or 0)
    return result


def trlau_intro_specific_data_type(obj) -> int:
    data = _intro_specific_group(obj)
    if data is not None and bool(getattr(data, 'enabled', False)):
        data_type = int(getattr(data, 'data_type', 0) or 0)
        if data_type > 0:
            return data_type
        struct_name = str(getattr(data, 'struct_name', '') or '')
        return int(_INTRO_SPECIFIC_TYPE_ID_BY_NAME.get(struct_name, 0))
    return 0


def trlau_intro_specific_struct_name(obj) -> str:
    data = _intro_specific_group(obj)
    if data is not None and bool(getattr(data, 'enabled', False)):
        struct_name = str(getattr(data, 'struct_name', '') or '')
        if struct_name:
            return struct_name
        data_type = int(getattr(data, 'data_type', 0) or 0)
        return _INTRO_SPECIFIC_TYPE_NAME_BY_ID.get(data_type, f'IntroType{data_type}Data' if data_type else '')
    return ''


def trlau_intro_specific_data_to_dict(obj, data_type: int | None = None) -> dict[str, object]:
    data = _intro_specific_group(obj)
    if data is None or not bool(getattr(data, 'enabled', False)):
        return {}
    own_type = trlau_intro_specific_data_type(obj)
    if data_type is not None and int(data_type) > 0 and int(own_type) != int(data_type):
        return {}
    struct_name = trlau_intro_specific_struct_name(obj)
    result: dict[str, object] = {}
    for attr, _label in INTRO_SPECIFIC_FIELDS.get(struct_name, ()):  # known structured types
        value = getattr(data, attr, 0)
        if isinstance(value, str):
            result[attr] = _safe_int(value, 0)
        elif isinstance(value, float):
            result[attr] = float(value)
        else:
            result[attr] = int(value or 0)
    return result


def trlau_intro_sound_data_to_dict(obj) -> dict[str, object]:
    data = _intro_sound_group(obj)
    if data is None or not bool(getattr(data, 'enabled', False)):
        return {}
    return {
        'parent_index': int(getattr(data, 'parent_index', -1) or -1),
        'sound_id': int(getattr(data, 'sound_id', 0) or 0),
        'format': str(getattr(data, 'format', '') or ''),
        'file_name': str(getattr(data, 'file_name', '') or ''),
        'section_type': int(getattr(data, 'section_type', 0) or 0),
        'skip_flags': int(getattr(data, 'skip_flags', 0) or 0),
        'version_id': int(getattr(data, 'version_id', 0) or 0),
        'has_debug_info': int(getattr(data, 'has_debug_info', 0) or 0),
        'resource_type': int(getattr(data, 'resource_type', 0) or 0),
        'spec_mask': _safe_int(getattr(data, 'spec_mask', 0), 0xFFFFFFFF),
        'num_relocations': int(getattr(data, 'num_relocations', 0) or 0),
        'standalone_size': int(getattr(data, 'standalone_size', 0) or 0),
    }


def trlau_intro_is_intro_data(obj) -> bool:
    data = _intro_data_group(obj)
    return bool(data is not None and getattr(data, 'enabled', False))


def trlau_intro_is_generic_data(obj) -> bool:
    data = _intro_generic_group(obj)
    return bool(data is not None and getattr(data, 'enabled', False))


def trlau_intro_is_specific_data(obj) -> bool:
    data = _intro_specific_group(obj)
    return bool(data is not None and getattr(data, 'enabled', False))


def trlau_intro_is_sound_data(obj) -> bool:
    data = _intro_sound_group(obj)
    return bool(data is not None and getattr(data, 'enabled', False))


def _next_intro_data_index(collection) -> int:
    max_index = -1
    for obj in list(getattr(collection, 'all_objects', None) or collection.objects):
        data = _intro_data_group(obj)
        if data is not None and bool(getattr(data, 'enabled', False)):
            max_index = max(max_index, int(getattr(data, 'index', -1) or -1))
    return max_index + 1


def _next_intro_unique_id(collection) -> int:
    max_id = -1
    for obj in list(getattr(collection, 'all_objects', None) or collection.objects):
        data = _intro_data_group(obj)
        if data is not None and bool(getattr(data, 'enabled', False)):
            max_id = max(max_id, int(getattr(data, 'unique_id', -1) or -1))
    return max_id + 1


def _markup_data_for_ui(obj):
    return getattr(obj, 'trlau_markup_data', None)


def _multispline_clear_spline(spline) -> None:
    spline.enabled = False
    spline.type = 0
    spline.flags = 0
    spline.count = 0.0
    spline.source_offset = 0
    spline.active_key_index = 0
    spline.keys.clear()


def _multispline_load_spline(spline, data, *, rotational: bool = False) -> None:
    _multispline_clear_spline(spline)
    if not isinstance(data, dict) or not data:
        return
    spline.enabled = True
    spline.type = _safe_int(data.get('type', 0)) & 0xFF
    spline.flags = _safe_int(data.get('flags', 0)) & 0xFF
    spline.count = _safe_float(data.get('count', data.get('numKeys', 0.0)))
    spline.source_offset = _safe_int(data.get('sourceOffset', 0))
    raw_keys = data.get('keys', [])
    keys = list(raw_keys or []) if isinstance(raw_keys, list) else []
    for key_data in keys:
        if not isinstance(key_data, dict):
            continue
        key = spline.keys.add()
        if rotational:
            key.q = _float_seq(key_data.get('q'), 4, (0.0, 0.0, 0.0, 1.0))
        else:
            key.point = _float_seq(key_data.get('point'), 4, (0.0, 0.0, 0.0, 0.0))
            key.d1 = _float_seq(key_data.get('d1'), 4, (0.0, 0.0, 0.0, 0.0))
            key.d2 = _float_seq(key_data.get('d2'), 4, (0.0, 0.0, 0.0, 0.0))
        key.t0 = _safe_float(key_data.get('t0', 0.0))
        key.tf = _safe_float(key_data.get('tf', 0.0))
        key.count = _safe_float(key_data.get('count', 0.0))
        key.invCount = _safe_float(key_data.get('invCount', key_data.get('inv_count', 0.0)))


def _multispline_state_from_dict(state, data) -> None:
    if isinstance(data, dict):
        state.t = _safe_float(data.get('t', 0.0))
        state.curr_key = _safe_int(data.get('currKey', data.get('curr_key', 0))) & 0xFFFF
        state.pad = _safe_int(data.get('pad', 0)) & 0xFFFF
    else:
        state.t = 0.0
        state.curr_key = 0
        state.pad = 0


def trlau_load_multispline_property(obj, data) -> None:
    if obj is None or not hasattr(obj, 'trlau_multi_spline'):
        return
    ms = obj.trlau_multi_spline
    if not isinstance(data, dict) or not data:
        ms.enabled = False
        _multispline_clear_spline(ms.positional)
        _multispline_clear_spline(ms.rotational)
        _multispline_clear_spline(ms.scaling)
        return
    ms.enabled = True
    ms.source_offset = _safe_int(data.get('sourceOffset', 0))
    ms.source_raw_value = _safe_int(data.get('sourceRawValue', 0))
    ms.cur_rot_matrix = _float_seq(data.get('curRotMatrix'), 16, _identity_matrix16())
    _multispline_load_spline(ms.positional, data.get('positional'), rotational=False)
    _multispline_load_spline(ms.rotational, data.get('rotational'), rotational=True)
    _multispline_load_spline(ms.scaling, data.get('scaling'), rotational=False)
    _multispline_state_from_dict(ms.cur_positional, data.get('curPositional'))
    _multispline_state_from_dict(ms.cur_rotational, data.get('curRotational'))
    _multispline_state_from_dict(ms.cur_scaling, data.get('curScaling'))


def _multispline_spline_to_dict(spline, *, rotational: bool = False):
    if spline is None or not bool(getattr(spline, 'enabled', False)):
        return None
    keys = []
    for key in list(getattr(spline, 'keys', []) or []):
        if rotational:
            item = {
                'q': [float(v) for v in key.q],
                't0': float(key.t0),
                'tf': float(key.tf),
                'count': float(key.count),
                'invCount': float(key.invCount),
            }
        else:
            item = {
                'point': [float(v) for v in key.point],
                'd1': [float(v) for v in key.d1],
                'd2': [float(v) for v in key.d2],
                't0': float(key.t0),
                'tf': float(key.tf),
                'count': float(key.count),
                'invCount': float(key.invCount),
            }
        keys.append(item)
    return {
        'numKeys': len(keys),
        'type': int(getattr(spline, 'type', 0)) & 0xFF,
        'flags': int(getattr(spline, 'flags', 0)) & 0xFF,
        'count': float(getattr(spline, 'count', float(len(keys))) or 0.0),
        'keys': keys,
        'sourceOffset': int(getattr(spline, 'source_offset', 0) or 0),
    }


def _multispline_state_to_dict(state) -> dict[str, object]:
    return {
        't': float(getattr(state, 't', 0.0) or 0.0),
        'currKey': int(getattr(state, 'curr_key', 0) or 0) & 0xFFFF,
        'pad': int(getattr(state, 'pad', 0) or 0) & 0xFFFF,
    }


def trlau_multispline_property_to_dict(obj):
    if obj is None or not hasattr(obj, 'trlau_multi_spline'):
        return None
    ms = obj.trlau_multi_spline
    if not bool(getattr(ms, 'enabled', False)):
        return None
    return {
        'format': 'TRLAU.MultiSpline.v1',
        'sourceOffset': int(getattr(ms, 'source_offset', 0) or 0),
        'sourceRawValue': int(getattr(ms, 'source_raw_value', 0) or 0),
        'curRotMatrix': [float(v) for v in ms.cur_rot_matrix],
        'positional': _multispline_spline_to_dict(ms.positional, rotational=False),
        'rotational': _multispline_spline_to_dict(ms.rotational, rotational=True),
        'scaling': _multispline_spline_to_dict(ms.scaling, rotational=False),
        'curPositional': _multispline_state_to_dict(ms.cur_positional),
        'curRotational': _multispline_state_to_dict(ms.cur_rotational),
        'curScaling': _multispline_state_to_dict(ms.cur_scaling),
    }


def _multispline_get_target(obj, target: str):
    if obj is None or not hasattr(obj, 'trlau_multi_spline'):
        return None
    ms = obj.trlau_multi_spline
    if target == 'rotational':
        return ms.rotational
    if target == 'scaling':
        return ms.scaling
    return ms.positional


class TRLAU_OT_multispline_add_key(bpy.types.Operator):
    bl_idname = 'trlau.multispline_add_key'
    bl_label = 'Add MultiSpline Key'
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(name='Spline', items=MULTISPLINE_TARGET_ITEMS, default='positional')

    def execute(self, context):
        obj = context.object
        spline = _multispline_get_target(obj, self.target)
        if spline is None:
            return {'CANCELLED'}
        obj.trlau_multi_spline.enabled = True
        spline.enabled = True
        key = spline.keys.add()
        key.q = (0.0, 0.0, 0.0, 1.0)
        spline.active_key_index = max(0, len(spline.keys) - 1)
        if not spline.count:
            spline.count = float(len(spline.keys))
        return {'FINISHED'}


class TRLAU_OT_multispline_remove_key(bpy.types.Operator):
    bl_idname = 'trlau.multispline_remove_key'
    bl_label = 'Remove MultiSpline Key'
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(name='Spline', items=MULTISPLINE_TARGET_ITEMS, default='positional')

    def execute(self, context):
        spline = _multispline_get_target(context.object, self.target)
        if spline is None or len(spline.keys) == 0:
            return {'CANCELLED'}
        index = max(0, min(int(spline.active_key_index), len(spline.keys) - 1))
        spline.keys.remove(index)
        spline.active_key_index = max(0, min(index, len(spline.keys) - 1))
        return {'FINISHED'}


def _draw_multispline_key(layout, key, index: int, *, rotational: bool = False) -> None:
    key_box = layout.box()
    key_box.label(text=f'Key {index}')
    if rotational:
        key_box.prop(key, 'q')
    else:
        key_box.prop(key, 'point')
        key_box.prop(key, 'd1')
        key_box.prop(key, 'd2')
    row = key_box.row(align=True)
    row.prop(key, 't0')
    row.prop(key, 'tf')
    row = key_box.row(align=True)
    row.prop(key, 'count')
    row.prop(key, 'invCount')


def _draw_multispline_spline(layout, obj, label: str, target: str, spline, *, rotational: bool = False) -> None:
    box = layout.box()
    header = box.row(align=True)
    header.prop(spline, 'enabled', text=label)
    add_op = header.operator('trlau.multispline_add_key', text='', icon='ADD')
    add_op.target = target
    remove_op = header.operator('trlau.multispline_remove_key', text='', icon='REMOVE')
    remove_op.target = target
    if not bool(spline.enabled):
        return
    row = box.row(align=True)
    row.prop(spline, 'type')
    row.prop(spline, 'flags')
    box.prop(spline, 'count')
    box.label(text=f'keys: {len(spline.keys)}')
    if len(spline.keys):
        active_index = max(0, min(int(spline.active_key_index), len(spline.keys) - 1))
        box.prop(spline, 'active_key_index', text='Key Index')
        _draw_multispline_key(box, spline.keys[active_index], active_index, rotational=rotational)


def draw_multispline_panel(layout, obj) -> None:
    if obj is None or not hasattr(obj, 'trlau_multi_spline'):
        return
    ms = obj.trlau_multi_spline
    box = layout.box()
    box.label(text='MultiSpline')
    box.prop(ms, 'enabled', text='Has MultiSpline')
    if not bool(ms.enabled):
        return
    row = box.row(align=True)
    row.prop(ms, 'source_offset', text='sourceOffset')
    row.prop(ms, 'source_raw_value', text='sourceRawValue')
    matrix_box = box.box()
    matrix_box.label(text='curRotMatrix')
    for row_index in range(4):
        row = matrix_box.row(align=True)
        for col_index in range(4):
            row.prop(ms, 'cur_rot_matrix', index=(row_index * 4) + col_index, text='')
    state_box = box.box()
    state_box.label(text='SplineState')
    for state_attr, label in (('cur_positional', 'curPositional'), ('cur_rotational', 'curRotational'), ('cur_scaling', 'curScaling')):
        state = getattr(ms, state_attr)
        row = state_box.row(align=True)
        row.label(text=label)
        row.prop(state, 't', text='t')
        row.prop(state, 'curr_key', text='currKey')
        row.prop(state, 'pad', text='pad')
    _draw_multispline_spline(box, obj, 'positional', 'positional', ms.positional, rotational=False)
    _draw_multispline_spline(box, obj, 'rotational', 'rotational', ms.rotational, rotational=True)
    _draw_multispline_spline(box, obj, 'scaling', 'scaling', ms.scaling, rotational=False)


class TRLAU_PG_signal_spline_camera_level_data(bpy.types.PropertyGroup):
    CameraFollowSmooth: IntProperty(name='CameraFollowSmooth')
    CameraFollowTilt: IntProperty(name='CameraFollowTilt')
    CameraFollowDistance: IntProperty(name='CameraFollowDistance')
    CameraFollowRotation: IntProperty(name='CameraFollowRotation')
    CameraZOffset: IntProperty(name='CameraZOffset')
    CameraCutAngle: IntProperty(name='CameraCutAngle')
    CameraCombatMinDistance: IntProperty(name='CameraCombatMinDistance')
    CameraCombatDistance: IntProperty(name='CameraCombatDistance')
    CameraCombatZOffset: IntProperty(name='CameraCombatZOffset')
    CameraCombatLockedOut: IntProperty(name='CameraCombatLockedOut')
    CameraDisableLookaroundFlag: IntProperty(name='CameraDisableLookaroundFlag')
    CameraSpline0: IntProperty(name='CameraSpline0')
    CameraSpline1: IntProperty(name='CameraSpline1')
    CameraSpline2: IntProperty(name='CameraSpline2')
    CameraAlternateSpline0: IntProperty(name='CameraAlternateSpline0')
    CameraAlternateSpline1: IntProperty(name='CameraAlternateSpline1')
    CameraAlternateSpline2: IntProperty(name='CameraAlternateSpline2')
    CameraMode: IntProperty(name='CameraMode')
    CameraLeadAmount: IntProperty(name='CameraLeadAmount')
    CameraSpline0Width: IntProperty(name='CameraSpline0Width')
    CameraSpline1Width: IntProperty(name='CameraSpline1Width')
    CameraSpline0WidthZ: IntProperty(name='CameraSpline0WidthZ')
    CameraSpline1WidthZ: IntProperty(name='CameraSpline1WidthZ')
    CameraZoomDist: IntProperty(name='CameraZoomDist')
    CameraVelocity: IntProperty(name='CameraVelocity')
    CameraDampening: IntProperty(name='CameraDampening')
    CameraTargetVelocity: IntProperty(name='CameraTargetVelocity')
    CameraCombatFraming: IntProperty(name='CameraCombatFraming')
    CameraTargetDampening: IntProperty(name='CameraTargetDampening')
    CameraCutFlag: IntProperty(name='CameraCutFlag')
    CameraFollowCamOverrideEnabled: IntProperty(name='CameraFollowCamOverrideEnabled')
    CameraCrossfade: IntProperty(name='CameraCrossfade')
    CameraInterestInstId: IntProperty(name='CameraInterestInstId')
    CameraInterestTune: IntProperty(name='CameraInterestTune')
    CameraSwitchToFollowDist: IntProperty(name='CameraSwitchToFollowDist')
    CameraFollowVerticalBias: IntProperty(name='CameraFollowVerticalBias')
    CameraFollowHighTiltDistance: IntProperty(name='CameraFollowHighTiltDistance')
    CameraFollowHighTiltAngle: IntProperty(name='CameraFollowHighTiltAngle')
    CameraFollowMedTiltDistance: IntProperty(name='CameraFollowMedTiltDistance')
    CameraFollowMedTiltAngle: IntProperty(name='CameraFollowMedTiltAngle')
    CameraFollowZeroTiltDistance: IntProperty(name='CameraFollowZeroTiltDistance')
    CameraFollowZeroTiltAngle: IntProperty(name='CameraFollowZeroTiltAngle')
    CameraFollowLowTiltDistance: IntProperty(name='CameraFollowLowTiltDistance')
    CameraFollowLowTiltAngle: IntProperty(name='CameraFollowLowTiltAngle')


class TRLAU_PG_signal_spline_camera_pointer(bpy.types.PropertyGroup):
    slot: IntProperty(name='slot', default=-1)
    present: BoolProperty(name='present', default=False)
    data: PointerProperty(type=TRLAU_PG_signal_spline_camera_level_data)


class TRLAU_PG_signal_attack_wave_limit(bpy.types.PropertyGroup):
    x: FloatProperty(name='x')
    y: FloatProperty(name='y')
    z: FloatProperty(name='z')
    rad: FloatProperty(name='rad')
    pursueRad: FloatProperty(name='pursueRad')


class TRLAU_PG_signal_attack_wave_box_limit(bpy.types.PropertyGroup):
    x: FloatProperty(name='x')
    y: FloatProperty(name='y')
    z: FloatProperty(name='z')
    zrot: FloatProperty(name='zrot')
    width: IntProperty(name='width')
    length: IntProperty(name='length')
    pursueWidth: IntProperty(name='pursueWidth')
    pursueLength: IntProperty(name='pursueLength')


class TRLAU_PG_signal_attack_wave_plane_limit(bpy.types.PropertyGroup):
    px: FloatProperty(name='px')
    py: FloatProperty(name='py')
    pz: FloatProperty(name='pz')
    nx: FloatProperty(name='nx')
    ny: FloatProperty(name='ny')
    nz: FloatProperty(name='nz')


class TRLAU_PG_signal_attack_wave_run_and_gun_pos(bpy.types.PropertyGroup):
    x: FloatProperty(name='x')
    y: FloatProperty(name='y')
    z: FloatProperty(name='z')
    priority: IntProperty(name='priority')
    waittime: IntProperty(name='waittime')
    failAction: IntProperty(name='failAction')
    markerName: StringProperty(name='markerName')


class TRLAU_PG_signal_attack_wave_patrol_pos(bpy.types.PropertyGroup):
    x: FloatProperty(name='x')
    y: FloatProperty(name='y')
    z: FloatProperty(name='z')
    lookx: FloatProperty(name='lookx')
    looky: FloatProperty(name='looky')
    lookz: FloatProperty(name='lookz')
    waittime: IntProperty(name='waittime')
    anim: IntProperty(name='anim')
    mode: IntProperty(name='mode')
    look: IntProperty(name='look')


class TRLAU_PG_signal_attack_wave_message(bpy.types.PropertyGroup):
    message: IntProperty(name='message')
    data: IntProperty(name='data')


class TRLAU_PG_signal_attack_wave_runtime(bpy.types.PropertyGroup):
    chain_index: IntProperty(name='chain index', default=-1)
    radiusCheckTimer: FloatProperty(name='radiusCheckTimer')
    flags: IntProperty(name='flags', min=0, max=255)
    rtFlags: IntProperty(name='rtFlags', min=0, max=255)
    probability: IntProperty(name='probability', min=0, max=255)
    numAttackerSpawns: IntProperty(name='numAttackerSpawns', min=0, max=255)
    numSpawnsThisLoad: IntProperty(name='numSpawnsThisLoad', min=0, max=255)
    currentAttacker: IntProperty(name='currentAttacker', min=0, max=255)
    triggerRemaining: IntProperty(name='triggerRemaining', min=0, max=255)
    numAttackersPerWave: IntProperty(name='numAttackersPerWave', min=0, max=255)
    numLinkedAttackWaves: IntProperty(name='numLinkedAttackWaves', min=0, max=255)
    numBlockedAttackWaves: IntProperty(name='numBlockedAttackWaves', min=0, max=255)
    numFinishedAttackWaves: IntProperty(name='numFinishedAttackWaves', min=0, max=255)
    numAttackers: IntProperty(name='numAttackers', min=0, max=255)
    musicRank: IntProperty(name='musicRank', min=0, max=255)
    pad: IntProperty(name='pad', min=-128, max=127)
    radiusCheckMarker: IntProperty(name='radiusCheckMarker', min=0, max=65535)
    radiusCheckRadius: IntProperty(name='radiusCheckRadius')
    nextEnableAttackWaveTargetSection: IntProperty(name='nextEnableAttackWave section', default=-1)
    nextEnableAttackWaveTargetOffset: IntProperty(name='nextEnableAttackWave offset', min=0)
    nextDisableAttackWaveTargetSection: IntProperty(name='nextDisableAttackWave section', default=-1)
    nextDisableAttackWaveTargetOffset: IntProperty(name='nextDisableAttackWave offset', min=0)
    nextKillAttackWaveTargetSection: IntProperty(name='nextKillAttackWave section', default=-1)
    nextKillAttackWaveTargetOffset: IntProperty(name='nextKillAttackWave offset', min=0)
    dataTargetSection: IntProperty(name='data[0] section', default=-1)
    dataTargetOffset: IntProperty(name='data[0] offset', min=0)
    endMarker: IntProperty(name='endMarker')
    numLimits: IntProperty(name='numLimits')
    limits: CollectionProperty(type=TRLAU_PG_signal_attack_wave_limit)
    numBoxLimits: IntProperty(name='numBoxLimits')
    boxLimits: CollectionProperty(type=TRLAU_PG_signal_attack_wave_box_limit)
    numPlaneLimits: IntProperty(name='numPlaneLimits')
    planeLimits: CollectionProperty(type=TRLAU_PG_signal_attack_wave_plane_limit)
    numRunAndGunPos: IntProperty(name='numRunAndGunPos')
    runAndGunPositions: CollectionProperty(type=TRLAU_PG_signal_attack_wave_run_and_gun_pos)
    numPatrolPos: IntProperty(name='numPatrolPos')
    patrolType: IntProperty(name='patrolType')
    patrolPositions: CollectionProperty(type=TRLAU_PG_signal_attack_wave_patrol_pos)
    numMessages: IntProperty(name='numMessages')
    messages: CollectionProperty(type=TRLAU_PG_signal_attack_wave_message)




class TRLAU_PG_level_attack_wave_definition(bpy.types.PropertyGroup):
    flags: IntProperty(name='flags', min=0, max=255)
    numAttackerSpawns: IntProperty(name='numAttackerSpawns', min=0, max=255)
    numAttackersPerWave: IntProperty(name='numAttackersPerWave', min=0, max=255)
    ignoreTimeForFirst: IntProperty(name='ignoreTimeForFirst', min=0, max=255)
    probability: IntProperty(name='probability', min=0, max=255)
    triggerRemaining: IntProperty(name='triggerRemaining', min=0, max=255)
    numLinkedAttackWaves: IntProperty(name='numLinkedAttackWaves', min=0, max=255)
    numAttackers: IntProperty(name='numAttackers', min=0, max=255)
    numAttackWaveVars: IntProperty(name='numAttackWaveVars', min=0, max=255)
    numBlockingAttackWaves: IntProperty(name='numBlockingAttackWaves', min=0, max=255)
    numFinishedAttackWaves: IntProperty(name='numFinishedAttackWaves', min=0, max=255)
    musicRank: IntProperty(name='musicRank', min=0, max=255)
    radiusCheckMarker: IntProperty(name='radiusCheckMarker', min=0, max=65535)
    radiusCheckTime: IntProperty(name='radiusCheckTime', min=-32768, max=32767)
    radiusCheckRadius: IntProperty(name='radiusCheckRadius', min=-32768, max=32767)
    pad: IntProperty(name='pad', min=-32768, max=32767)
    enableSignalTargetSection: IntProperty(name='enableSignal section', default=-1)
    enableSignalTargetOffset: IntProperty(name='enableSignal offset', min=0)
    disableSignalTargetSection: IntProperty(name='disableSignal section', default=-1)
    disableSignalTargetOffset: IntProperty(name='disableSignal offset', min=0)
    killSignalTargetSection: IntProperty(name='killSignal section', default=-1)
    killSignalTargetOffset: IntProperty(name='killSignal offset', min=0)
    dataTargetSection: IntProperty(name='data[0] section', default=-1)
    dataTargetOffset: IntProperty(name='data[0] offset', min=0)


class TRLAU_PG_level_attack_wave_entry(bpy.types.PropertyGroup):
    index: IntProperty(name='index', default=-1)
    wave_id: IntProperty(name='ID', min=0, max=65535)
    name: StringProperty(name='name', maxlen=256)
    definition: PointerProperty(type=TRLAU_PG_level_attack_wave_definition)
    runtime: PointerProperty(type=TRLAU_PG_signal_attack_wave_runtime)




class TRLAU_PG_level_pmarker_entry(bpy.types.PropertyGroup):
    index: IntProperty(name='index', default=-1)
    marker_id: IntProperty(name='ID', min=0, max=65535)
    name: StringProperty(name='name', maxlen=256)
    lx: FloatProperty(name='lx')
    ly: FloatProperty(name='ly')
    lz: FloatProperty(name='lz')
    lrotx: FloatProperty(name='lrotx')
    lroty: FloatProperty(name='lroty')
    lrotz: FloatProperty(name='lrotz')
    mx: IntProperty(name='mx', min=-32768, max=32767)
    my: IntProperty(name='my', min=-32768, max=32767)
    mz: IntProperty(name='mz', min=-32768, max=32767)
    px: IntProperty(name='px', min=-32768, max=32767)
    py: IntProperty(name='py', min=-32768, max=32767)
    pz: IntProperty(name='pz', min=-32768, max=32767)
    flags: IntProperty(name='flags', min=0, max=65535)
    minDist: IntProperty(name='minDist', min=-32768, max=32767)
    maxDist: IntProperty(name='maxDist', min=-32768, max=32767)
    maxCombatDist: IntProperty(name='maxCombatDist', min=-32768, max=32767)
    pauseTime: IntProperty(name='pauseTime', min=-32768, max=32767)
    spawnType: IntProperty(name='spawnType', min=0, max=255)
    padding: IntProperty(name='padding', min=-128, max=127)

class TRLAU_PG_level_combat_data(bpy.types.PropertyGroup):
    embedded: BoolProperty(name='Embedded combat data', default=False)
    attack_wave_count: IntProperty(name='Attack waves', min=0)
    attack_waves: CollectionProperty(type=TRLAU_PG_level_attack_wave_entry)
    attack_wave_group_count: IntProperty(name='Attack-wave groups', min=0)
    attack_wave_groups: CollectionProperty(type=TRLAU_PG_level_attack_wave_entry)
    combat_door_count: IntProperty(name='Combat doors', min=0)
    pmarker_count: IntProperty(name='PMarkers', min=0)
    pmarkers: CollectionProperty(type=TRLAU_PG_level_pmarker_entry)
    vmarker_count: IntProperty(name='VMarkers', min=0)
    spline_camera_data_present: BoolProperty(name='Global spline-camera data present', default=False)


_LEVEL_ATTACK_WAVE_DEFINITION_FIELDS = (
    'flags', 'numAttackerSpawns', 'numAttackersPerWave', 'ignoreTimeForFirst',
    'probability', 'triggerRemaining', 'numLinkedAttackWaves', 'numAttackers',
    'numAttackWaveVars', 'numBlockingAttackWaves', 'numFinishedAttackWaves',
    'musicRank', 'radiusCheckMarker', 'radiusCheckTime', 'radiusCheckRadius', 'pad',
    'enableSignalTargetSection', 'enableSignalTargetOffset',
    'disableSignalTargetSection', 'disableSignalTargetOffset',
    'killSignalTargetSection', 'killSignalTargetOffset',
    'dataTargetSection', 'dataTargetOffset',
)

_LEVEL_ATTACK_WAVE_RUNTIME_FIELDS = (
    'chain_index', 'radiusCheckTimer', 'flags', 'rtFlags', 'probability',
    'numAttackerSpawns', 'numSpawnsThisLoad', 'currentAttacker',
    'triggerRemaining', 'numAttackersPerWave', 'numLinkedAttackWaves',
    'numBlockedAttackWaves', 'numFinishedAttackWaves', 'numAttackers',
    'musicRank', 'pad', 'radiusCheckMarker', 'radiusCheckRadius',
    'nextEnableAttackWaveTargetSection', 'nextEnableAttackWaveTargetOffset',
    'nextDisableAttackWaveTargetSection', 'nextDisableAttackWaveTargetOffset',
    'nextKillAttackWaveTargetSection', 'nextKillAttackWaveTargetOffset',
    'dataTargetSection', 'dataTargetOffset', 'endMarker',
    'numLimits', 'numBoxLimits', 'numPlaneLimits', 'numRunAndGunPos',
    'numPatrolPos', 'patrolType', 'numMessages',
)

_LEVEL_ATTACK_WAVE_RUNTIME_COLLECTIONS = (
    ('limits', ('x', 'y', 'z', 'rad', 'pursueRad')),
    ('boxLimits', ('x', 'y', 'z', 'zrot', 'width', 'length', 'pursueWidth', 'pursueLength')),
    ('planeLimits', ('px', 'py', 'pz', 'nx', 'ny', 'nz')),
    ('runAndGunPositions', ('x', 'y', 'z', 'priority', 'waittime', 'failAction', 'markerName')),
    ('patrolPositions', ('x', 'y', 'z', 'lookx', 'looky', 'lookz', 'waittime', 'anim', 'mode', 'look')),
    ('messages', ('message', 'data')),
)


def _load_attack_wave_runtime_property(target, values: dict[str, object]) -> None:
    if target is None:
        return
    values = dict(values or {})
    for field_name in _LEVEL_ATTACK_WAVE_RUNTIME_FIELDS:
        if field_name not in values:
            continue
        try:
            setattr(target, field_name, values[field_name])
        except Exception:
            pass
    for collection_name, field_names in _LEVEL_ATTACK_WAVE_RUNTIME_COLLECTIONS:
        collection = getattr(target, collection_name, None)
        if collection is None:
            continue
        try:
            collection.clear()
        except Exception:
            while len(collection):
                collection.remove(len(collection) - 1)
        for item_values in list(values.get(collection_name, []) or []):
            if not isinstance(item_values, dict):
                continue
            item = collection.add()
            for field_name in field_names:
                if field_name not in item_values:
                    continue
                try:
                    setattr(item, field_name, item_values[field_name])
                except Exception:
                    pass


def _attack_wave_runtime_property_to_dict(source) -> dict[str, object]:
    result: dict[str, object] = {}
    if source is None:
        return result
    for field_name in _LEVEL_ATTACK_WAVE_RUNTIME_FIELDS:
        try:
            result[field_name] = getattr(source, field_name)
        except Exception:
            pass
    for collection_name, field_names in _LEVEL_ATTACK_WAVE_RUNTIME_COLLECTIONS:
        values = []
        for item in list(getattr(source, collection_name, []) or []):
            entry = {}
            for field_name in field_names:
                try:
                    entry[field_name] = getattr(item, field_name)
                except Exception:
                    pass
            values.append(entry)
        result[collection_name] = values
    return result


def trlau_load_level_combat_property(obj, summary: dict[str, object], *, embedded: bool = True) -> None:
    data = getattr(obj, 'trlau_combat_data', None)
    if data is None:
        return
    data.embedded = bool(embedded)
    summary = dict(summary or {})
    for summary_name, count_name, collection_name in (
        ('attackWaveList', 'attack_wave_count', 'attack_waves'),
        ('attackWaveGroupList', 'attack_wave_group_count', 'attack_wave_groups'),
    ):
        list_data = dict(summary.get(summary_name, {}) or {})
        entries = list(list_data.get('entries', []) or [])
        try:
            setattr(data, count_name, max(0, int(list_data.get('count', len(entries)) or 0)))
        except Exception:
            pass
        collection = getattr(data, collection_name, None)
        if collection is None:
            continue
        try:
            collection.clear()
        except Exception:
            while len(collection):
                collection.remove(len(collection) - 1)
        for index, entry_values in enumerate(entries):
            if not isinstance(entry_values, dict):
                continue
            entry = collection.add()
            entry.index = int(entry_values.get('index', index) or index)
            entry.wave_id = int(entry_values.get('id', 0) or 0)
            entry.name = str(entry_values.get('name', '') or '')
            definition_values = dict(entry_values.get('definition', {}) or {})
            for field_name in _LEVEL_ATTACK_WAVE_DEFINITION_FIELDS:
                if field_name not in definition_values:
                    continue
                try:
                    setattr(entry.definition, field_name, definition_values[field_name])
                except Exception:
                    pass
            runtime_values = dict(entry_values.get('runtime', {}) or {})
            runtime_values.setdefault('chain_index', int(entry.index))
            _load_attack_wave_runtime_property(entry.runtime, runtime_values)
    doors = dict(summary.get('combatDoorsList', {}) or {})
    try:
        data.combat_door_count = max(0, int(doors.get('count', 0) or 0))
    except Exception:
        pass

    pmarker_data = dict(summary.get('pmarkerList', {}) or {})
    pmarker_entries = list(pmarker_data.get('entries', []) or [])
    try:
        data.pmarker_count = max(0, int(pmarker_data.get('count', len(pmarker_entries)) or 0))
    except Exception:
        pass
    try:
        data.pmarkers.clear()
    except Exception:
        while len(data.pmarkers):
            data.pmarkers.remove(len(data.pmarkers) - 1)
    for index, values in enumerate(pmarker_entries):
        if not isinstance(values, dict):
            continue
        item = data.pmarkers.add()
        item.index = int(values.get('index', index) or index)
        item.marker_id = int(values.get('id', 0) or 0)
        item.name = str(values.get('name', '') or '')
        for field_name in (
            'lx', 'ly', 'lz', 'lrotx', 'lroty', 'lrotz',
            'mx', 'my', 'mz', 'px', 'py', 'pz', 'flags',
            'minDist', 'maxDist', 'maxCombatDist', 'pauseTime', 'spawnType', 'padding',
        ):
            if field_name not in values:
                continue
            try:
                setattr(item, field_name, values[field_name])
            except Exception:
                pass
    vmarkers = dict(summary.get('vmarkerList', {}) or {})
    try:
        data.vmarker_count = max(0, int(vmarkers.get('count', 0) or 0))
    except Exception:
        pass
    data.spline_camera_data_present = bool(summary.get('splineCameraDataPresent', False))


def trlau_level_combat_property_to_dict(obj) -> dict[str, object]:
    data = getattr(obj, 'trlau_combat_data', None)
    if data is None:
        return {}
    result: dict[str, object] = {}
    for summary_name, count_name, collection_name in (
        ('attackWaveList', 'attack_wave_count', 'attack_waves'),
        ('attackWaveGroupList', 'attack_wave_group_count', 'attack_wave_groups'),
    ):
        entries = []
        for item in list(getattr(data, collection_name, []) or []):
            definition = {}
            for field_name in _LEVEL_ATTACK_WAVE_DEFINITION_FIELDS:
                try:
                    definition[field_name] = getattr(item.definition, field_name)
                except Exception:
                    pass
            entries.append({
                'index': int(getattr(item, 'index', len(entries))),
                'id': int(getattr(item, 'wave_id', 0)),
                'name': str(getattr(item, 'name', '') or ''),
                'definition': definition,
                'runtime': _attack_wave_runtime_property_to_dict(getattr(item, 'runtime', None)),
            })
        result[summary_name] = {
            'count': int(getattr(data, count_name, len(entries))),
            'entries': entries,
        }
    result['combatDoorsList'] = {'count': int(getattr(data, 'combat_door_count', 0))}
    pmarkers = []
    for item in list(getattr(data, 'pmarkers', []) or []):
        entry = {
            'index': int(getattr(item, 'index', len(pmarkers))),
            'id': int(getattr(item, 'marker_id', 0)),
            'name': str(getattr(item, 'name', '') or ''),
        }
        for field_name in (
            'lx', 'ly', 'lz', 'lrotx', 'lroty', 'lrotz',
            'mx', 'my', 'mz', 'px', 'py', 'pz', 'flags',
            'minDist', 'maxDist', 'maxCombatDist', 'pauseTime', 'spawnType', 'padding',
        ):
            try:
                entry[field_name] = getattr(item, field_name)
            except Exception:
                pass
        pmarkers.append(entry)
    result['pmarkerList'] = {
        'count': int(getattr(data, 'pmarker_count', len(pmarkers))),
        'entries': pmarkers,
    }
    return result

class TRLAU_PG_signal_data(bpy.types.PropertyGroup):
    signal_id: IntProperty(name='Signal ID', default=-1)
    list_index: IntProperty(name='Signal list index', default=-1)
    face_id: IntProperty(name='SignalFace.id', default=-1)
    raw_flags: IntProperty(name='flags', min=0, max=65535)
    exitPortal: BoolProperty(name='exitPortal')
    entryPortal: BoolProperty(name='entryPortal')
    autoStream: BoolProperty(name='autoStream')
    ResetFSFXToDefaultOut: BoolProperty(name='ResetFSFXToDefaultOut')
    ResetFSFXToDefaultIn: BoolProperty(name='ResetFSFXToDefaultIn')
    MaterialOnly: BoolProperty(name='MaterialOnly')
    SpectralOnly: BoolProperty(name='SpectralOnly')
    ObjectHitSignal: BoolProperty(name='ObjectHitSignal')
    ResetSlideAngle: BoolProperty(name='ResetSlideAngle')
    triggered: BoolProperty(name='triggered')
    startGoingIntoWaterSignal: BoolProperty(name='startGoingIntoWaterSignal')
    startGoingOutOfWaterSignal: BoolProperty(name='startGoingOutOfWaterSignal')
    CameraLock: IntProperty(name='CameraLock')
    CameraUnlock: IntProperty(name='CameraUnlock')
    CameraSave: IntProperty(name='CameraSave')
    CameraRestore: IntProperty(name='CameraRestore')
    CameraShakeScale: IntProperty(name='CameraShakeScale', min=0, max=65535)
    CameraShakeTime: IntProperty(name='CameraShakeTime')
    splineCamParams: PointerProperty(type=TRLAU_PG_signal_spline_camera_level_data)
    streamName: StringProperty(name='streamName', maxlen=20)
    streamPortalIndex: IntProperty(name='streamPortalIndex')
    SetSlideAngle: IntProperty(name='SetSlideAngle')
    ReverbType: IntProperty(name='ReverbType')
    ReverbVolume: IntProperty(name='ReverbVolume')
    setMusicVar: IntProperty(name='setMusicVar')
    setMusicVarValue: IntProperty(name='setMusicVarValue')
    fatalType: IntProperty(name='fatalType')
    CameraStackType: IntProperty(name='CameraStackType')
    CameraStackData: IntProperty(name='CameraStackData', min=0, max=15)
    CameraStackCamInUse: IntProperty(name='CameraStackCamInUse', min=0, max=15)
    CameraStackSignalFlags0: IntProperty(name='CameraStackSignalFlags[0]', min=0, max=255)
    CameraStackSignalFlags1: IntProperty(name='CameraStackSignalFlags[1]', min=0, max=255)
    DTPCameraID0: IntProperty(name='DTPCameraID[0]')
    DTPCameraID1: IntProperty(name='DTPCameraID[1]')
    FSFXInDTPID: IntProperty(name='FSFXInDTPID')
    FSFXExitDTPID: IntProperty(name='FSFXExitDTPID')
    FSFXInActiveDist: IntProperty(name='FSFXInActiveDist', min=0, max=65535)
    FSFXExitActiveDist: IntProperty(name='FSFXExitActiveDist', min=0, max=65535)
    splineCamDataPtr0: PointerProperty(type=TRLAU_PG_signal_spline_camera_pointer)
    splineCamDataPtr1: PointerProperty(type=TRLAU_PG_signal_spline_camera_pointer)
    splineCamDataPtr2: PointerProperty(type=TRLAU_PG_signal_spline_camera_pointer)
    enableAttackWaves: CollectionProperty(type=TRLAU_PG_signal_attack_wave_runtime)
    disableAttackWaves: CollectionProperty(type=TRLAU_PG_signal_attack_wave_runtime)
    killAttackWaves: CollectionProperty(type=TRLAU_PG_signal_attack_wave_runtime)
    cameraLinkSignalIndex: IntProperty(name='CameraLinkSignal index', default=-1)


SIGNAL_FLAG_FIELD_NAMES = (
    'exitPortal',
    'entryPortal',
    'autoStream',
    'ResetFSFXToDefaultOut',
    'ResetFSFXToDefaultIn',
    'MaterialOnly',
    'SpectralOnly',
    'ObjectHitSignal',
    'ResetSlideAngle',
    'triggered',
)

SIGNAL_VALUE_FIELD_NAMES = (
    'CameraLock',
    'CameraUnlock',
    'CameraSave',
    'CameraRestore',
    'CameraShakeScale',
    'CameraShakeTime',
    'streamName',
    'streamPortalIndex',
    'SetSlideAngle',
    'ReverbType',
    'ReverbVolume',
    'setMusicVar',
    'setMusicVarValue',
    'fatalType',
    'CameraStackType',
    'CameraStackData',
    'CameraStackCamInUse',
    'CameraStackSignalFlags0',
    'CameraStackSignalFlags1',
    'DTPCameraID0',
    'DTPCameraID1',
    'FSFXInDTPID',
    'FSFXExitDTPID',
    'FSFXInActiveDist',
    'FSFXExitActiveDist',
)

SIGNAL_POINTER_FIELD_GROUPS = (
    ('splineCamDataPtr0', 'splineCamDataPtr[0]'),
    ('splineCamDataPtr1', 'splineCamDataPtr[1]'),
    ('splineCamDataPtr2', 'splineCamDataPtr[2]'),
)

SIGNAL_ATTACK_WAVE_COLLECTIONS = (
    ('enableAttackWaves', 'enableAttackWaves'),
    ('disableAttackWaves', 'disableAttackWaves'),
    ('killAttackWaves', 'killAttackWaves'),
)

SIGNAL_SPLINE_CAMERA_FIELD_NAMES = (
    'CameraFollowSmooth',
    'CameraFollowTilt',
    'CameraFollowDistance',
    'CameraFollowRotation',
    'CameraZOffset',
    'CameraCutAngle',
    'CameraCombatMinDistance',
    'CameraCombatDistance',
    'CameraCombatZOffset',
    'CameraCombatLockedOut',
    'CameraDisableLookaroundFlag',
    'CameraSpline0',
    'CameraSpline1',
    'CameraSpline2',
    'CameraAlternateSpline0',
    'CameraAlternateSpline1',
    'CameraAlternateSpline2',
    'CameraMode',
    'CameraLeadAmount',
    'CameraSpline0Width',
    'CameraSpline1Width',
    'CameraSpline0WidthZ',
    'CameraSpline1WidthZ',
    'CameraZoomDist',
    'CameraVelocity',
    'CameraDampening',
    'CameraTargetVelocity',
    'CameraCombatFraming',
    'CameraTargetDampening',
    'CameraCutFlag',
    'CameraFollowCamOverrideEnabled',
    'CameraCrossfade',
    'CameraInterestInstId',
    'CameraInterestTune',
    'CameraSwitchToFollowDist',
    'CameraFollowVerticalBias',
    'CameraFollowHighTiltDistance',
    'CameraFollowHighTiltAngle',
    'CameraFollowMedTiltDistance',
    'CameraFollowMedTiltAngle',
    'CameraFollowZeroTiltDistance',
    'CameraFollowZeroTiltAngle',
    'CameraFollowLowTiltDistance',
    'CameraFollowLowTiltAngle',
)



def _draw_signal_spline_fields(layout, spline_data) -> None:
    for field_name in SIGNAL_SPLINE_CAMERA_FIELD_NAMES:
        layout.prop(spline_data, field_name)


def _draw_signal_spline_pointer(layout, signal_data, prop_name: str, label: str) -> None:
    ptr = getattr(signal_data, prop_name, None)
    box = layout.box()
    if ptr is None or not bool(getattr(ptr, 'present', False)):
        box.label(text=f'{label}: null')
        return
    box.label(text=label)
    _draw_signal_spline_fields(box, ptr.data)


def _draw_simple_collection(layout, title: str, collection, field_groups: tuple[tuple[str, ...], ...]) -> None:
    if collection is None or len(collection) == 0:
        return
    box = layout.box()
    box.label(text=f'{title}: {len(collection)}')
    for index, item in enumerate(collection):
        item_box = box.box()
        item_box.label(text=f'{title}[{index}]')
        for fields in field_groups:
            row = item_box.row(align=True)
            for field_name in fields:
                row.prop(item, field_name)


def _draw_signal_attack_wave_collection(layout, signal_data, prop_name: str, label: str) -> None:
    waves = getattr(signal_data, prop_name, None)
    box = layout.box()
    count = len(waves) if waves is not None else 0
    box.label(text=f'{label}: {count}')
    if waves is None or len(waves) == 0:
        return
    for wave in waves:
        wave_box = box.box()
        wave_box.label(text=f'{label}[{wave.chain_index}]')
        for field_name in (
            'radiusCheckTimer', 'flags', 'rtFlags', 'probability',
            'numAttackerSpawns', 'numSpawnsThisLoad', 'currentAttacker',
            'triggerRemaining', 'numAttackersPerWave', 'numLinkedAttackWaves',
            'numBlockedAttackWaves', 'numFinishedAttackWaves', 'numAttackers',
            'musicRank', 'radiusCheckMarker', 'radiusCheckRadius', 'endMarker',
        ):
            wave_box.prop(wave, field_name)
        _draw_simple_collection(wave_box, 'limit', wave.limits, (('x', 'y', 'z'), ('rad', 'pursueRad')))
        _draw_simple_collection(wave_box, 'boxLimit', wave.boxLimits, (('x', 'y', 'z'), ('zrot',), ('width', 'length'), ('pursueWidth', 'pursueLength')))
        _draw_simple_collection(wave_box, 'planeLimit', wave.planeLimits, (('px', 'py', 'pz'), ('nx', 'ny', 'nz')))
        _draw_simple_collection(wave_box, 'runAndGunPos', wave.runAndGunPositions, (('x', 'y', 'z'), ('priority', 'waittime', 'failAction'), ('markerName',)))
        _draw_simple_collection(wave_box, 'patrolPos', wave.patrolPositions, (('x', 'y', 'z'), ('lookx', 'looky', 'lookz'), ('waittime', 'anim', 'mode', 'look')))
        _draw_simple_collection(wave_box, 'message', wave.messages, (('message', 'data'),))


def _draw_level_attack_wave_entry(layout, entry, label: str) -> None:
    box = layout.box()
    box.label(text=label)
    row = box.row(align=True)
    row.prop(entry, 'wave_id')
    row.prop(entry, 'name')
    definition_box = box.box()
    definition_box.label(text='AttackWave')
    for field_name in _LEVEL_ATTACK_WAVE_DEFINITION_FIELDS:
        definition_box.prop(entry.definition, field_name)
    runtime_box = box.box()
    runtime_box.label(text='AttackWaveRuntime')
    for field_name in (
        'radiusCheckTimer', 'flags', 'rtFlags', 'probability', 'numAttackerSpawns',
        'numSpawnsThisLoad', 'currentAttacker', 'triggerRemaining', 'numAttackersPerWave',
        'numLinkedAttackWaves', 'numBlockedAttackWaves', 'numFinishedAttackWaves',
        'numAttackers', 'musicRank', 'pad', 'radiusCheckMarker', 'radiusCheckRadius', 'endMarker',
    ):
        runtime_box.prop(entry.runtime, field_name)
    pointer_box = runtime_box.box()
    pointer_box.label(text='Runtime pointers (section / offset)')
    for section_field, offset_field in (
        ('nextEnableAttackWaveTargetSection', 'nextEnableAttackWaveTargetOffset'),
        ('nextDisableAttackWaveTargetSection', 'nextDisableAttackWaveTargetOffset'),
        ('nextKillAttackWaveTargetSection', 'nextKillAttackWaveTargetOffset'),
        ('dataTargetSection', 'dataTargetOffset'),
    ):
        row = pointer_box.row(align=True)
        row.prop(entry.runtime, section_field)
        row.prop(entry.runtime, offset_field)
    _draw_simple_collection(runtime_box, 'limit', entry.runtime.limits, (('x', 'y', 'z'), ('rad', 'pursueRad')))
    _draw_simple_collection(runtime_box, 'boxLimit', entry.runtime.boxLimits, (('x', 'y', 'z'), ('zrot',), ('width', 'length'), ('pursueWidth', 'pursueLength')))
    _draw_simple_collection(runtime_box, 'planeLimit', entry.runtime.planeLimits, (('px', 'py', 'pz'), ('nx', 'ny', 'nz')))
    _draw_simple_collection(runtime_box, 'runAndGunPos', entry.runtime.runAndGunPositions, (('x', 'y', 'z'), ('priority', 'waittime', 'failAction'), ('markerName',)))
    _draw_simple_collection(runtime_box, 'patrolPos', entry.runtime.patrolPositions, (('x', 'y', 'z'), ('lookx', 'looky', 'lookz'), ('waittime', 'anim', 'mode', 'look')))
    _draw_simple_collection(runtime_box, 'message', entry.runtime.messages, (('message', 'data'),))



def _draw_level_pmarker_entry(layout, entry, label: str) -> None:
    box = layout.box()
    box.label(text=label)
    row = box.row(align=True)
    row.prop(entry, 'marker_id')
    row.prop(entry, 'name')
    row = box.row(align=True)
    row.prop(entry, 'lx'); row.prop(entry, 'ly'); row.prop(entry, 'lz')
    row = box.row(align=True)
    row.prop(entry, 'lrotx'); row.prop(entry, 'lroty'); row.prop(entry, 'lrotz')
    row = box.row(align=True)
    row.prop(entry, 'mx'); row.prop(entry, 'my'); row.prop(entry, 'mz')
    row = box.row(align=True)
    row.prop(entry, 'px'); row.prop(entry, 'py'); row.prop(entry, 'pz')
    row = box.row(align=True)
    row.prop(entry, 'minDist'); row.prop(entry, 'maxDist'); row.prop(entry, 'maxCombatDist')
    row = box.row(align=True)
    row.prop(entry, 'flags'); row.prop(entry, 'pauseTime'); row.prop(entry, 'spawnType'); row.prop(entry, 'padding')

def _get_terrain_light_radius_ui(self):
    light_data = getattr(self, 'data', None)
    if light_data is not None and hasattr(light_data, 'shadow_soft_size'):
        try:
            return max(0, int(round(float(light_data.shadow_soft_size) * 10.0)))
        except Exception:
            pass
    return max(0, int(round(float(self.get('trlau_terrain_light_radius', 0) or 0))))


def _set_terrain_light_radius_ui(self, value):
    radius = max(0, int(round(float(value))))
    self['trlau_terrain_light_radius'] = radius
    light_data = getattr(self, 'data', None)
    if light_data is not None and hasattr(light_data, 'shadow_soft_size'):
        try:
            light_data.shadow_soft_size = max(0.0, float(radius) / 10.0)
        except Exception:
            pass


def _get_terrain_light_multiplier_ui(self):
    light_data = getattr(self, 'data', None)
    if light_data is not None and hasattr(light_data, 'energy'):
        try:
            return int(round(float(light_data.energy) / 100.0))
        except Exception:
            pass
    return int(round(float(self.get('trlau_terrain_light_multiplier', 0) or 0)))


def _set_terrain_light_multiplier_ui(self, value):
    multiplier = int(round(float(value)))
    self['trlau_terrain_light_multiplier'] = multiplier
    light_data = getattr(self, 'data', None)
    if light_data is not None and hasattr(light_data, 'energy'):
        try:
            light_data.energy = float(multiplier) * 100.0
        except Exception:
            pass




LEVEL_METADATA_TAB_ITEMS = tuple((identifier, label, label) for identifier, label, _fields in LEVEL_METADATA_GROUPS)

UNIT_DATA_GROUPS = (
    ('General', (
        ('id', 'ID'),
        ('num_cines', 'm_numCines'),
        ('script', 'm_script'),
    )),
    ('DepthData', (
        ('depth_backcolor', 'backcolor'),
        ('depth_watercolor', 'watercolor'),
        ('depth_ambientcolor', 'ambientcolor'),
        ('depth_objectambientcolor', 'objectambientcolor'),
        ('depth_waterblend', 'waterblend'),
        ('depth_fogfar', 'fogfar'),
        ('depth_fogmax', 'fogmax'),
        ('depth_fognear', 'fognear'),
        ('depth_waterfogfar', 'waterfogfar'),
        ('depth_waterfogmax', 'waterfogmax'),
        ('depth_waterfognear', 'waterfognear'),
        ('depth_underwaterfxalpha', 'underwaterfxalpha'),
        ('depth_underwaterfxmovement', 'underwaterfxmovement'),
        ('depth_underwaterfxspeed', 'underwaterfxspeed'),
        ('depth_underwaterfxrandscale', 'underwaterfxrandscale'),
        ('depth_underwaterfxscale', 'underwaterfxscale'),
        ('depth_underwaterfxcolor', 'underwaterfxcolor'),
        ('depth_underwaterfxadditive', 'underwaterfxadditive'),
        ('depth_underwaterfxbeforezsort', 'underwaterfxbeforezsort'),
        ('depth_underwaterfxvertalpha', 'underwaterfxvertalpha'),
        ('depth_underwaterplayercolor', 'underwaterplayercolor'),
        ('depth_waterfxalpha', 'waterfxalpha'),
        ('depth_waterfxmovement', 'waterfxmovement'),
        ('depth_waterfxspeed', 'waterfxspeed'),
        ('depth_waterfxrandscale', 'waterfxrandscale'),
        ('depth_waterfxscale', 'waterfxscale'),
        ('depth_waterfxcolor', 'waterfxcolor'),
        ('depth_waterfxadditive', 'waterfxadditive'),
        ('depth_waterfxbeforezsort', 'waterfxbeforezsort'),
        ('depth_waterfxvertalpha', 'waterfxvertalpha'),
    )),
    ('BendData', (
        ('bend_black_offset', 'Black offset'),
        ('bend_black_time_offset', 'Black timeOffset'),
        ('bend_black_time_scale', 'Black timeScale'),
        ('bend_blue_offset', 'Blue offset'),
        ('bend_blue_time_offset', 'Blue timeOffset'),
        ('bend_blue_time_scale', 'Blue timeScale'),
        ('bend_green_offset', 'Green offset'),
        ('bend_green_time_offset', 'Green timeOffset'),
        ('bend_green_time_scale', 'Green timeScale'),
        ('bend_bluegreen_offset', 'BlueGreen offset'),
        ('bend_bluegreen_time_offset', 'BlueGreen timeOffset'),
        ('bend_bluegreen_time_scale', 'BlueGreen timeScale'),
        ('bend_red_offset', 'Red offset'),
        ('bend_red_time_offset', 'Red timeOffset'),
        ('bend_red_time_scale', 'Red timeScale'),
        ('bend_redblue_offset', 'RedBlue offset'),
        ('bend_redblue_time_offset', 'RedBlue timeOffset'),
        ('bend_redblue_time_scale', 'RedBlue timeScale'),
        ('bend_redgreen_offset', 'RedGreen offset'),
        ('bend_redgreen_time_offset', 'RedGreen timeOffset'),
        ('bend_redgreen_time_scale', 'RedGreen timeScale'),
        ('bend_redgreenblue_offset', 'RedGreenBlue offset'),
        ('bend_redgreenblue_time_offset', 'RedGreenBlue timeOffset'),
        ('bend_redgreenblue_time_scale', 'RedGreenBlue timeScale'),
    )),
    ('BaseCamera', (
        ('base_camera_use_camera_stack_system', 'UseCameraStackSystem'),
        ('base_camera_basecam', 'basecam'),
        ('base_camera_ledgecam', 'ledgecam'),
        ('base_camera_crawlcam', 'crawlcam'),
        ('base_camera_vehiclecam', 'vehiclecam'),
        ('base_camera_deathcam', 'deathcam'),
    )),
    ('NextGenData', (
        ('nextgen_vertex_color_percent', 'm_vertexColorPercent'),
        ('nextgen_global_ambient_color', 'm_globalAmbientColor'),
        ('nextgen_global_ambient_intensity', 'm_globalAmbientIntensity'),
        ('nextgen_override_fog_color', 'm_bOverrideFogColor'),
        ('nextgen_fog_color', 'm_fogColor'),
        ('nextgen_fog_start', 'm_fogStart'),
        ('nextgen_fog_end', 'm_fogEnd'),
        ('nextgen_enable_pls_spot', 'm_bEnablePlsSpot'),
        ('nextgen_enable_pls_spot_shadows', 'm_bEnablePlsSpotShadows'),
        ('nextgen_light_fade_time', 'm_lightFadeTime'),
    )),
    ('PSPData', (
        ('psp_override_fog_color', 'm_bOverrideFogColor'),
        ('psp_fog_color', 'm_fogColor'),
        ('psp_fog_start', 'm_fogStart'),
        ('psp_fog_end', 'm_fogEnd'),
    )),
    ('OverlayData', (
        ('overlay_snow_level', 'm_bSnowLevel'),
        ('overlay_num_effects', 'm_numOverlayEffects'),
        ('overlay_p_effects_storage', 'm_pOverlayEffectsStorage'),
        ('overlay_effect_identification', 'Identification'),
        ('overlay_effect_type', 'Type'),
        ('overlay_effect_feet', 'FeetEffect'),
        ('overlay_effect_legs', 'LegsEffect'),
        ('overlay_effect_torso', 'TorsoEffect'),
        ('overlay_effect_face', 'FaceEffect'),
        ('overlay_effect_hair', 'HairEffect'),
        ('overlay_effect_feet_cap', 'FeetEffectCap'),
        ('overlay_effect_legs_cap', 'LegsEffectCap'),
        ('overlay_effect_torso_cap', 'TorsoEffectCap'),
        ('overlay_effect_face_cap', 'FaceEffectCap'),
        ('overlay_effect_hair_cap', 'HairEffectCap'),
    )),
)




def _level_collection_for_root(root, context=None):
    collections = list(getattr(root, 'users_collection', []) or []) if root is not None else []
    if collections:
        return collections[0]
    if context is not None and getattr(context, 'collection', None) is not None:
        return context.collection
    return bpy.context.collection


def _prefixed_level_name(root, component_name: str) -> str:
    drm_name = str(root.get('trlau_drm_name', getattr(root, 'name', '') or 'Level') or 'Level').strip()
    suffix = '_Level'
    if drm_name.endswith(suffix):
        drm_name = drm_name[:-len(suffix)]
    return f'{drm_name}_{component_name}' if drm_name else component_name


def _get_or_create_level_component_empty(root, collection, component_name: str, display_size: float = 56.0):
    expected_name = _prefixed_level_name(root, component_name)
    for child in getattr(root, 'children', []) or []:
        if bool(child.get('trlau_component_empty')) and str(child.get('trlau_component_name', '') or '') == component_name:
            return child
        if getattr(child, 'name', '') == expected_name:
            return child
    empty = bpy.data.objects.get(expected_name)
    if empty is None:
        empty = bpy.data.objects.new(expected_name, None)
    if all(coll != collection for coll in getattr(empty, 'users_collection', []) or []):
        collection.objects.link(empty)
    empty.empty_display_type = 'PLAIN_AXES'
    empty.empty_display_size = float(display_size)
    empty.parent = root
    empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    empty.location = (0.0, 0.0, 0.0)
    empty.rotation_mode = 'XYZ'
    empty.rotation_euler = (0.0, 0.0, 0.0)
    empty.scale = (1.0, 1.0, 1.0)
    empty['trlau_level'] = True
    empty['trlau_component_empty'] = True
    empty['trlau_component_name'] = component_name
    return empty


def _next_index_for_prop(collection, prop_name: str) -> int:
    max_index = -1
    for obj in getattr(collection, 'all_objects', []) or getattr(collection, 'objects', []):
        if prop_name not in obj:
            continue
        try:
            max_index = max(max_index, int(obj.get(prop_name)))
        except Exception:
            continue
    return max_index + 1


def _next_unique_int_for_prop(collection, prop_name: str, minimum: int = 0) -> int:
    used: set[int] = set()
    for obj in getattr(collection, 'all_objects', []) or getattr(collection, 'objects', []):
        if prop_name not in obj:
            continue
        try:
            used.add(int(obj.get(prop_name)))
        except Exception:
            continue
    if not used:
        return int(minimum)
    return max(max(used) + 1, int(minimum))


def _next_terrain_group_index(collection) -> int:
    max_index = -1
    regex = re.compile(r'(?:^|_)TerrainGroup_(\d+)(?:$|_|\.)')
    objects = getattr(collection, 'all_objects', []) or getattr(collection, 'objects', [])
    for obj in objects:
        if getattr(obj, 'type', None) != 'EMPTY':
            continue
        try:
            if 'trlau_terrain_group_index' in obj:
                max_index = max(max_index, int(obj.get('trlau_terrain_group_index')))
                continue
        except Exception:
            pass
        match = regex.search(str(getattr(obj, 'name', '') or ''))
        if match is not None:
            try:
                max_index = max(max_index, int(match.group(1)))
            except Exception:
                pass
    return max_index + 1


def _cursor_position_in_level_space(context, root):
    cursor_location = getattr(getattr(context, 'scene', None), 'cursor', None)
    if cursor_location is not None:
        world = getattr(cursor_location, 'location', mathutils.Vector((0.0, 0.0, 0.0)))
    else:
        world = mathutils.Vector((0.0, 0.0, 0.0))
    try:
        return root.matrix_world.inverted_safe() @ mathutils.Vector(world)
    except Exception:
        return mathutils.Vector((0.0, 0.0, 0.0))


def _selected_mesh_points_in_level_space(context, root):
    obj = getattr(context, 'edit_object', None) or getattr(context, 'object', None)
    if obj is None or getattr(obj, 'type', None) != 'MESH':
        return []
    root_inv = root.matrix_world.inverted_safe()
    points = []
    if getattr(context, 'mode', '') == 'EDIT_MESH':
        bm = bmesh.from_edit_mesh(obj.data)
        selected = [vert for vert in bm.verts if vert.select]
        selected_set = set(selected)
        selected_edges = [edge for edge in bm.edges if edge.select and edge.verts[0] in selected_set and edge.verts[1] in selected_set]
        ordered = []
        if selected_edges:
            adjacency = {vert: [] for vert in selected}
            for edge in selected_edges:
                a, b = edge.verts[0], edge.verts[1]
                adjacency.setdefault(a, []).append(b)
                adjacency.setdefault(b, []).append(a)
            endpoints = [vert for vert, neighbors in adjacency.items() if len(neighbors) == 1]
            current = endpoints[0] if endpoints else selected_edges[0].verts[0]
            previous = None
            visited = set()
            while current is not None and current not in visited:
                ordered.append(current)
                visited.add(current)
                next_vert = None
                for candidate in adjacency.get(current, []):
                    if candidate is previous:
                        continue
                    if candidate not in visited:
                        next_vert = candidate
                        break
                previous, current = current, next_vert
            for vert in selected:
                if vert not in visited:
                    ordered.append(vert)
        else:
            ordered = selected
        for vert in ordered:
            try:
                points.append(root_inv @ (obj.matrix_world @ vert.co))
            except Exception:
                pass
    else:
        for vert in getattr(obj.data, 'vertices', []) or []:
            if getattr(vert, 'select', False):
                try:
                    points.append(root_inv @ (obj.matrix_world @ vert.co))
                except Exception:
                    pass
    return points


def _sanitize_name_fragment(value: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() or ch in {'_', '-'} else '_' for ch in str(value or '').strip())
    return cleaned or 'Object'


def _objectlist_path_for_game(game: str) -> Path:
    filename = 'tra_objectlist_pc.txt' if _normalize_level_game_value(game) == 'anniversary' else 'tr7_objectlist_pc.txt'
    return Path(__file__).resolve().parent.parent / 'objectlists' / filename


def _load_packaged_objectlist(game: str) -> list[tuple[int, str]]:
    path = _objectlist_path_for_game(game)
    entries: list[tuple[int, str]] = []
    try:
        for raw_line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            line = raw_line.strip()
            if not line or ',' not in line:
                continue
            object_id_text, object_name = line.split(',', 1)
            try:
                entries.append((int(object_id_text.strip()), object_name.strip()))
            except Exception:
                continue
    except Exception:
        return []
    return entries


def _intro_object_items(self, context):
    root = _find_related_level_root(getattr(context, 'object', None), context) if context is not None else None
    game = _level_game_for_object(root, context) if root is not None else 'legend'
    entries = _load_packaged_objectlist(game)
    query = str(getattr(self, 'object_search', '') or '').strip().lower()
    if query:
        entries = [
            (object_id, object_name)
            for object_id, object_name in entries
            if query in str(object_id).lower() or query in str(object_name).lower()
        ]
    if not entries:
        return (('__NO_MATCH__', 'No matches', 'Clear or change the search text'),)
    return tuple((f'{object_id}|{object_name}', f'{object_id}: {object_name}', object_name) for object_id, object_name in entries)


def _markup_flag_from_identifier(identifier: str) -> int:
    identifier = str(identifier or '').strip()
    for item_identifier, _label, _description, value in MARKUP_TYPE_DEFS:
        if item_identifier == identifier:
            return int(value)
    return MARKUP_FLAG_LEDGE


def _add_polyline_spline(curve_data, points) -> None:
    points = [mathutils.Vector(point) for point in (points or [])]
    if not points:
        return
    spline = curve_data.splines.new('POLY')
    if len(points) > 1:
        spline.points.add(len(points) - 1)
    for point_index, point in enumerate(points):
        spline.points[point_index].co = (float(point.x), float(point.y), float(point.z), 1.0)



def _reverse_spline_points(spline) -> bool:
    if getattr(spline, 'type', '') == 'BEZIER':
        points = list(getattr(spline, 'bezier_points', []) or [])
        if len(points) < 2:
            return False
        snapshot = []
        for point in points:
            snapshot.append({
                'co': point.co.copy(),
                'handle_left': point.handle_left.copy(),
                'handle_right': point.handle_right.copy(),
                'handle_left_type': point.handle_left_type,
                'handle_right_type': point.handle_right_type,
                'radius': point.radius,
                'tilt': point.tilt,
                'weight_softbody': point.weight_softbody,
            })
        for point, source in zip(points, reversed(snapshot)):
            point.co = source['co']
            point.handle_left = source['handle_right']
            point.handle_right = source['handle_left']
            point.handle_left_type = source['handle_right_type']
            point.handle_right_type = source['handle_left_type']
            point.radius = source['radius']
            point.tilt = source['tilt']
            point.weight_softbody = source['weight_softbody']
        return True

    points = list(getattr(spline, 'points', []) or [])
    if len(points) < 2:
        return False
    snapshot = []
    for point in points:
        snapshot.append({
            'co': point.co.copy(),
            'radius': point.radius,
            'tilt': point.tilt,
            'weight_softbody': point.weight_softbody,
        })
    for point, source in zip(points, reversed(snapshot)):
        point.co = source['co']
        point.radius = source['radius']
        point.tilt = source['tilt']
        point.weight_softbody = source['weight_softbody']
    return True


def _reverse_curve_direction(obj) -> bool:
    data = getattr(obj, 'data', None)
    if getattr(obj, 'type', None) != 'CURVE' or data is None:
        return False
    changed = False
    for spline in getattr(data, 'splines', []) or []:
        changed = _reverse_spline_points(spline) or changed
    if changed:
        data.update_tag()
    return changed


def _points_bounds(points):
    points = [mathutils.Vector(point) for point in (points or [])]
    if not points:
        return None
    min_x = min(float(point.x) for point in points)
    min_y = min(float(point.y) for point in points)
    min_z = min(float(point.z) for point in points)
    max_x = max(float(point.x) for point in points)
    max_y = max(float(point.y) for point in points)
    max_z = max(float(point.z) for point in points)
    return (min_x, min_y, min_z, max_x, max_y, max_z)


def _expanded_bounds_for_template(points, center, *, default_half_x=128.0, default_half_y=128.0, default_half_z=128.0):
    bounds = _points_bounds(points)
    if bounds is None:
        cx, cy, cz = float(center.x), float(center.y), float(center.z)
        return (cx - default_half_x, cy - default_half_y, cz - default_half_z, cx + default_half_x, cy + default_half_y, cz + default_half_z)
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    if abs(max_x - min_x) < 1e-4:
        min_x -= default_half_x
        max_x += default_half_x
    if abs(max_y - min_y) < 1e-4:
        min_y -= default_half_y
        max_y += default_half_y
    if abs(max_z - min_z) < 1e-4:
        min_z -= default_half_z
        max_z += default_half_z
    return (min_x, min_y, min_z, max_x, max_y, max_z)


def _add_box_curve_splines(curve_data, center, points=None) -> None:
    min_x, min_y, min_z, max_x, max_y, max_z = _expanded_bounds_for_template(points, center)
    corners = [
        mathutils.Vector((min_x, min_y, min_z)),
        mathutils.Vector((max_x, min_y, min_z)),
        mathutils.Vector((max_x, max_y, min_z)),
        mathutils.Vector((min_x, max_y, min_z)),
        mathutils.Vector((min_x, min_y, max_z)),
        mathutils.Vector((max_x, min_y, max_z)),
        mathutils.Vector((max_x, max_y, max_z)),
        mathutils.Vector((min_x, max_y, max_z)),
    ]
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    center = mathutils.Vector(center)
    for first, second in edge_indices:
        _add_polyline_spline(curve_data, (corners[second] - center, corners[first] - center))


def _add_markup_template_splines(curve_data, markup_identifier: str, center, selected_points=None) -> int:
    markup_identifier = str(markup_identifier or 'LEDGE').strip() or 'LEDGE'
    center = mathutils.Vector(center)
    selected_points = [mathutils.Vector(point) for point in (selected_points or [])]

    if markup_identifier in {'PERCH', 'WATER'}:
        _add_box_curve_splines(curve_data, center, selected_points)
        return 0

    if markup_identifier in {'LADDER', 'VERTICAL_POLE', 'WALL_VERTICAL_POLE'}:
        if selected_points:
            min_x, min_y, min_z, max_x, max_y, max_z = _expanded_bounds_for_template(
                selected_points,
                center,
                default_half_x=32.0,
                default_half_y=0.0,
                default_half_z=128.0,
            )
            bottom_z = min_z
            top_z = max_z
            mid_x = (min_x + max_x) * 0.5
            mid_y = (min_y + max_y) * 0.5
            half_width = max(abs(max_x - min_x) * 0.5, 32.0)
        else:
            bottom_z = center.z - 128.0
            top_z = center.z + 128.0
            mid_x = center.x
            mid_y = center.y
            half_width = 32.0
        world_points = (
            mathutils.Vector((mid_x + half_width, mid_y, bottom_z)),
            mathutils.Vector((mid_x, mid_y, bottom_z)),
            mathutils.Vector((mid_x, mid_y, top_z)),
        )
        _add_polyline_spline(curve_data, [point - center for point in world_points])
        return len(world_points)

    if selected_points:
        _add_polyline_spline(curve_data, [point - center for point in reversed(selected_points)])
        return len(selected_points)

    _add_polyline_spline(curve_data, (mathutils.Vector((128.0, 0.0, 0.0)), mathutils.Vector((-128.0, 0.0, 0.0))))
    return 2



class TRLAU_OT_reverse_markup_direction(bpy.types.Operator):
    bl_idname = 'trlau.reverse_markup_direction'
    bl_label = 'Reverse Direction'
    bl_description = 'Reverse the point order of the selected Markup curve direction'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and getattr(obj, 'type', None) == 'CURVE' and str(obj.get('trlau_type', '') or '') == 'Markup'

    def execute(self, context):
        selected = [obj for obj in getattr(context, 'selected_objects', []) or [] if getattr(obj, 'type', None) == 'CURVE' and str(obj.get('trlau_type', '') or '') == 'Markup']
        if not selected:
            selected = [context.object] if context.object is not None else []
        changed_count = 0
        previous_mode = getattr(context, 'mode', 'OBJECT')
        try:
            if previous_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        for obj in selected:
            if _reverse_curve_direction(obj):
                changed_count += 1
        try:
            context.view_layer.update()
        except Exception:
            pass
        if changed_count <= 0:
            self.report({'WARNING'}, 'No curve direction was changed')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Reversed {changed_count} Markup curve direction(s)')
        return {'FINISHED'}


class TRLAU_OT_add_level_terrain_group(bpy.types.Operator):
    bl_idname = 'trlau.add_level_terrain_group'
    bl_label = 'Add TerrainGroup'
    bl_description = 'Create an empty TerrainGroup under the selected level Terrain component'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and _find_related_level_root(obj, context) is not None

    def execute(self, context):
        root = _find_related_level_root(getattr(context, 'object', None), context)
        if root is None:
            self.report({'WARNING'}, 'Select a level object')
            return {'CANCELLED'}

        collection = _level_collection_for_root(root, context)
        terrain_empty = _get_or_create_level_component_empty(root, collection, 'Terrain', display_size=72.0)
        terrain_empty['trlau_type'] = 'Terrain'

        index = _next_terrain_group_index(collection)
        group_name = _prefixed_level_name(root, f'TerrainGroup_{index:03d}')
        group_empty = bpy.data.objects.new(group_name, None)
        collection.objects.link(group_empty)
        group_empty.empty_display_type = 'PLAIN_AXES'
        group_empty.empty_display_size = 64.0
        group_empty.parent = terrain_empty
        group_empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        location = _cursor_position_in_level_space(context, root)
        group_empty.location = (float(location.x), float(location.y), float(location.z))
        group_empty.rotation_mode = 'XYZ'
        group_empty.rotation_euler = (0.0, 0.0, 0.0)
        group_empty.scale = (1.0, 1.0, 1.0)

        group_empty['trlau_terrain_group_flags'] = 0
        group_empty['trlau_terrain_group_id'] = 0
        group_empty['trlau_terrain_group_unique_id'] = int(index)
        group_empty['trlau_terrain_group_spline_id'] = 0

        bpy.ops.object.select_all(action='DESELECT')
        group_empty.select_set(True)
        context.view_layer.objects.active = group_empty
        self.report({'INFO'}, f'Added TerrainGroup {index:03d}')
        return {'FINISHED'}


class TRLAU_OT_add_level_markup(bpy.types.Operator):
    bl_idname = 'trlau.add_level_markup'
    bl_label = 'Add Markup'
    bl_description = 'Create a Markup curve under the selected level'
    bl_options = {'REGISTER', 'UNDO'}

    markup_type: EnumProperty(name='Type', items=MARKUP_TYPE_ITEMS, default='LEDGE')

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and _find_related_level_root(obj, context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        self.layout.prop(self, 'markup_type', text='Type')

    def execute(self, context):
        source_obj = getattr(context, 'object', None)
        root = _find_related_level_root(source_obj, context)
        if root is None:
            self.report({'WARNING'}, 'Select a level object')
            return {'CANCELLED'}
        collection = _level_collection_for_root(root, context)
        parent = _get_or_create_level_component_empty(root, collection, 'Markup')
        index = _next_index_for_prop(collection, 'trlau_markup_index')
        points = _selected_mesh_points_in_level_space(context, root)
        if points:
            center = sum(points, mathutils.Vector((0.0, 0.0, 0.0))) / float(len(points))
        else:
            center = _cursor_position_in_level_space(context, root)
        curve_name = _prefixed_level_name(root, f'Markup_{index:03d}')
        curve_data = bpy.data.curves.new(curve_name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 1
        point_count = _add_markup_template_splines(curve_data, self.markup_type, center, points)
        curve_obj = bpy.data.objects.new(curve_name, curve_data)
        collection.objects.link(curve_obj)
        curve_obj.parent = parent
        curve_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        curve_obj.location = (float(center.x), float(center.y), float(center.z))
        curve_obj.rotation_euler = (0.0, 0.0, 0.0)
        curve_obj.scale = (1.0, 1.0, 1.0)
        curve_obj['trlau_level'] = True
        curve_obj['trlau_type'] = 'Markup'
        curve_obj['trlau_markup_index'] = int(index)
        curve_obj['trlau_markup_game'] = _level_game_for_object(root, context)
        flags = _markup_flag_from_identifier(self.markup_type)
        _write_int_prop(curve_obj, 'trlau_markup_flags', flags)
        curve_obj['trlau_markup_point_count'] = int(point_count)
        if flags & MARKUP_BBOX_FLAGS:
            curve_obj['trlau_markup_bbox_shape'] = True
        curve_obj['trlau_markup_animated_segment'] = -1
        try:
            curve_obj.trlau_markup_data.intro_id = 0
        except Exception:
            pass
        try:
            if getattr(context, 'mode', '') != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
        bpy.ops.object.select_all(action='DESELECT')
        curve_obj.select_set(True)
        context.view_layer.objects.active = curve_obj
        self.report({'INFO'}, f'Added Markup {index}')
        return {'FINISHED'}


class TRLAU_OT_add_level_light(bpy.types.Operator):
    bl_idname = 'trlau.add_level_light'
    bl_label = 'Add Light'
    bl_description = 'Create a TerrainLight under the selected level'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and _find_related_level_root(obj, context) is not None

    def execute(self, context):
        root = _find_related_level_root(getattr(context, 'object', None), context)
        if root is None:
            self.report({'WARNING'}, 'Select a level object')
            return {'CANCELLED'}
        collection = _level_collection_for_root(root, context)
        parent = _get_or_create_level_component_empty(root, collection, 'TerrainLight')
        index = _next_index_for_prop(collection, 'trlau_terrain_light_index')
        light_name = _prefixed_level_name(root, f'TerrainLight_{index:03d}')
        light_data = bpy.data.lights.new(light_name, type='POINT')
        light_data.energy = 2048.0 * 100.0
        if hasattr(light_data, 'exposure'):
            light_data.exposure = 10.0
        light_data.shadow_soft_size = 200.0
        light_obj = bpy.data.objects.new(light_name, light_data)
        collection.objects.link(light_obj)
        light_obj.parent = parent
        light_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        location = _cursor_position_in_level_space(context, root)
        light_obj.location = (float(location.x), float(location.y), float(location.z))
        light_obj['trlau_level'] = True
        light_obj['trlau_type'] = 'TerrainLight'
        light_obj['trlau_terrain_light_index'] = int(index)
        light_obj['trlau_terrain_light_type'] = 24
        light_obj['trlau_terrain_light_id'] = 0
        light_obj['trlau_terrain_light_radius'] = 2000
        light_obj['trlau_terrain_light_multiplier'] = 2048
        light_obj['trlau_terrain_light_hotspot_angle'] = 0
        light_obj['trlau_terrain_light_falloff_angle'] = 0
        light_obj['trlau_terrain_light_direction'] = (0.0, 0.0, -4096.0)
        bpy.ops.object.select_all(action='DESELECT')
        light_obj.select_set(True)
        context.view_layer.objects.active = light_obj
        self.report({'INFO'}, f'Added TerrainLight {index}')
        return {'FINISHED'}




class TRLAU_OT_fsfx_decode_stored_blob(bpy.types.Operator):
    bl_idname = 'trlau.fsfx_decode_stored_blob'
    bl_label = 'Decode Stored FSFX Blob'
    bl_description = 'Parse the raw stored FSFX section blob into editable RenderFSEffect properties'

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        if obj is None:
            return False
        try:
            return bool(obj.get('trlau_section_metadata_empty')) and str(obj.get('trlau_section_role', '') or '') in FSFX_SECTION_ROLES
        except Exception:
            return False

    def execute(self, context):
        obj = context.object
        blob = decode_stored_section_blob_from_object(obj)
        if not blob:
            self.report({'ERROR'}, 'Selected section has no stored raw blob')
            return {'CANCELLED'}
        try:
            ok = store_fsfx_render_effect_properties(obj, blob)
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to decode FSFX blob: {exc}')
            return {'CANCELLED'}
        if not ok:
            self.report({'ERROR'}, 'Stored section is not a supported RenderFSEffect blob')
            return {'CANCELLED'}
        self.report({'INFO'}, 'Decoded FSFX RenderFSEffect data')
        return {'FINISHED'}


class TRLAU_OT_add_level_intro_data(bpy.types.Operator):
    bl_idname = 'trlau.add_level_intro_data'
    bl_label = 'Add IntroData'
    bl_description = 'Create an IntroData empty and optionally import the selected object DRM when it can be found next to the level source file'
    bl_options = {'REGISTER', 'UNDO'}

    object_search: StringProperty(name='Search', default='', options={'SKIP_SAVE'})
    object_choice: EnumProperty(name='Object', items=_intro_object_items)

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and _find_related_level_root(obj, context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'object_search', text='Search')
        layout.prop(self, 'object_choice', text='Object')

    def _source_dir_for_root(self, root) -> Path | None:
        for key in ('trlau_level_source_dir', 'trlau_level_source_path'):
            value = str(root.get(key, '') or '').strip()
            if not value:
                continue
            path = Path(value)
            if key.endswith('_path'):
                path = path.parent
            if path.exists() and path.is_dir():
                return path
        return None

    def _find_named_drm(self, directory: Path | None, object_name: str) -> Path | None:
        if directory is None:
            return None
        candidates = [directory / f'{object_name}.drm', directory / f'{object_name.lower()}.drm', directory / f'{object_name.upper()}.drm']
        for candidate in candidates:
            if candidate.exists():
                return candidate
        lowered = object_name.strip().lower()
        try:
            for candidate in directory.glob('*.drm'):
                if candidate.stem.lower() == lowered:
                    return candidate
        except Exception:
            return None
        return None

    def _try_import_model(self, context, drm_path: Path, parent_empty) -> None:
        from ..core.log import logger
        from ..operators.animation_import import AnimationImporterMixin
        from ..operators.import_utils import ImportUtilityMixin
        from ..operators.model_importer import ModelImporterMixin
        from ..platforms.common.drm_container import DRMContainerParser
        from ..platforms.pc.tr7ae_object import TRObjectParser

        class _IntroDataModelImporter(ImportUtilityMixin, ModelImporterMixin, AnimationImporterMixin):
            @staticmethod
            def _link_object_to_collection(obj, collection) -> None:
                if obj is None or collection is None:
                    return
                if all(coll != collection for coll in getattr(obj, 'users_collection', ())):
                    try:
                        collection.objects.link(obj)
                    except Exception:
                        pass

            @staticmethod
            def _section_id_matches_anim_id(section_id, anim_id):
                if section_id is None:
                    return False
                return int(section_id) == int(anim_id) or (int(section_id) & 0xFFFF) == (int(anim_id) & 0xFFFF)

            @staticmethod
            def _standalone_section_id(path):
                try:
                    from ..core.binary import BinaryReader
                    from ..platforms.common.section import TRSectionParser
                    with open(path, 'rb') as fh:
                        info = TRSectionParser.parse(BinaryReader(fh, endian='<'))
                    return int(getattr(info, 'section_id', 0))
                except Exception:
                    return None

            def _find_animation_path_for_id(self, extracted_paths, anim_id):
                direct_section_match = None
                for animation_path in sorted((Path(path) for path in extracted_paths), key=lambda item: item.name):
                    if animation_path.suffix.lower() in {'.obj', '.level', '.pcd'}:
                        continue
                    section_id = self._standalone_section_id(animation_path)
                    if self._section_id_matches_anim_id(section_id, anim_id) and direct_section_match is None:
                        direct_section_match = animation_path
                    parsed = self._probe_animation_file(
                        str(animation_path),
                        default_endianness='<',
                        platform=getattr(self, 'platform', 'PC'),
                    )
                    if parsed is not None and int(parsed.get('anim_id', -1)) == int(anim_id):
                        return animation_path
                return direct_section_match

            def _import_first_intro_animation(self, context, extracted_paths, imported_results, anim_id, action_prefix, parent_empty) -> bool:
                for stale_key in ('trlau_intro_animation_id', 'trlau_intro_animation_action', 'trlau_intro_animation_nla_strip'):
                    try:
                        if parent_empty is not None and stale_key in parent_empty:
                            del parent_empty[stale_key]
                    except Exception:
                        pass

                armatures = [result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None]
                armature = armatures[0] if armatures else None
                if armature is None:
                    logger.debug('Skipping IntroData animation import for %s: imported model has no armature', getattr(parent_empty, 'name', '<unknown>'))
                    return False

                animation_path = self._find_animation_path_for_id(extracted_paths, anim_id)
                if animation_path is None:
                    logger.debug('Skipping IntroData animation import for %s: animation ID %d was not found in %s', getattr(parent_empty, 'name', '<unknown>'), int(anim_id), Path(getattr(parent_empty, 'name', 'IntroData')).name)
                    return False

                animation_result = self._import_animation_file(
                    context,
                    str(animation_path),
                    armature=armature,
                    default_endianness='<',
                    action_name=f'{action_prefix}_Anim_{int(anim_id)}',
                    platform=getattr(self, 'platform', 'PC'),
                )
                action = animation_result.get('animation_action')
                final_frame = int(animation_result.get('animation_final_frame', 0) or 0)
                strip = self._create_looping_nla_strip(
                    context,
                    armature,
                    action,
                    final_frame,
                    track_name='IntroData First Animation',
                    strip_name=f'{action.name}_Loop' if action is not None else None,
                )
                return strip is not None

        collection = (list(getattr(parent_empty, 'users_collection', []) or []) or [getattr(context, 'collection', None)])[0]
        if collection is None:
            collection = context.scene.collection

        importer = _IntroDataModelImporter()
        importer.platform = 'PC'
        importer.import_textures = True
        importer.import_hinfo = False
        importer.import_cloth = False
        importer.import_markups = False
        importer.import_main_model_only = True
        importer.import_armature_only = False
        importer.import_bounding_boxes = False
        importer.import_all_textures = False
        importer.import_all_animations = False
        importer.separate_by_drawgroup = False
        importer.separate_by_material = False
        importer._trlau_skip_targets = True

        before = set(bpy.data.objects)
        unique_prefix = f'{_sanitize_name_fragment(getattr(parent_empty, "name", "IntroData"))}_{_sanitize_name_fragment(drm_path.stem)}'

        drm_parser = DRMContainerParser(str(drm_path))
        with drm_parser.temporary_extract_sections() as (_extract_dir, extracted_paths, _sections):
            if not extracted_paths:
                return
            first_anim_id = None
            try:
                first_anim_id = TRObjectParser(str(extracted_paths[0])).parse_first_animation_id()
            except Exception as exc:
                logger.debug('Unable to read first IntroData animation from %s: %s', drm_path.name, exc)

            imported_results = importer._import_from_object_refs(
                context,
                drm_path,
                str(extracted_paths[0]),
                collection_name=unique_prefix,
                collection=collection,
            )
            if first_anim_id is not None:
                try:
                    importer._import_first_intro_animation(
                        context,
                        extracted_paths,
                        imported_results,
                        first_anim_id,
                        unique_prefix,
                        parent_empty,
                    )
                except Exception as exc:
                    logger.warning('IntroData animation import failed for %s animation ID %s: %s', drm_path.name, first_anim_id, exc)

        if importer.import_textures:
            source_dir = drm_path.resolve().parent
            for imported_result in imported_results:
                try:
                    importer._reload_imported_result_textures(imported_result, source_dir)
                except Exception:
                    pass

        importer._disable_relationship_lines(context)

        after = set(bpy.data.objects)
        created = [obj for obj in after - before]
        created_set = set(created)
        root_objects = [obj for obj in created if getattr(obj, 'parent', None) not in created_set]
        for obj in root_objects:
            try:
                obj.parent = parent_empty
                obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                obj.location = (0.0, 0.0, 0.0)
                obj.rotation_euler = (0.0, 0.0, 0.0)
                obj.scale = (1.0, 1.0, 1.0)
            except Exception:
                pass

    def execute(self, context):
        root = _find_related_level_root(getattr(context, 'object', None), context)
        if root is None:
            self.report({'WARNING'}, 'Select a level object')
            return {'CANCELLED'}
        choice = str(getattr(self, 'object_choice', '') or '')
        if choice == '__NO_MATCH__':
            self.report({'WARNING'}, 'No IntroData object matches the current search')
            return {'CANCELLED'}
        try:
            object_id_text, object_name = choice.split('|', 1)
            object_id = int(object_id_text)
        except Exception:
            object_id, object_name = 0, 'Object_000'
        collection = _level_collection_for_root(root, context)
        parent = _get_or_create_level_component_empty(root, collection, 'IntroData')
        index = _next_intro_data_index(collection)
        safe_name = _sanitize_name_fragment(object_name)
        empty_name = _prefixed_level_name(root, f'Intro_{index:03d}_{safe_name}')
        intro_empty = bpy.data.objects.new(empty_name, None)
        collection.objects.link(intro_empty)
        intro_empty.empty_display_type = 'PLAIN_AXES'
        intro_empty.empty_display_size = 48.0
        intro_empty.parent = parent
        intro_empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        location = _cursor_position_in_level_space(context, root)
        intro_empty.location = (float(location.x), float(location.y), float(location.z))
        intro_empty.rotation_mode = 'XYZ'
        intro_empty.rotation_euler = (0.0, 0.0, 0.0)
        intro_empty.scale = (1.0, 1.0, 1.0)
        intro_empty['trlau_level'] = True
        intro_empty['trlau_type'] = 'IntroData'
        trlau_set_intro_data_property(
            intro_empty,
            index=int(index),
            object_id=int(object_id),
            intro_num=0,
            unique_id=_next_intro_unique_id(collection),
            start_frame=0,
            end_frame=0,
            intro_flags=0,
            attached_vmo=0,
            multi_spline=0,
            rotation=(0.0, 0.0, 0.0, 0.0),
            rotation_mode='XYZ',
            rotation_basis='RAW_XYZ',
            scale=(1.0, 1.0, 1.0, 0.0),
        )
        drm_path = self._find_named_drm(self._source_dir_for_root(root), object_name)
        if drm_path is not None:
            try:
                self._try_import_model(context, drm_path, intro_empty)
            except Exception as exc:
                self.report({'WARNING'}, f'IntroData created, but model import failed: {exc}')
        bpy.ops.object.select_all(action='DESELECT')
        intro_empty.select_set(True)
        context.view_layer.objects.active = intro_empty
        self.report({'INFO'}, f'Added IntroData {index}')
        return {'FINISHED'}


class VIEW3D_PT_trlau_level_editing(bpy.types.Panel):
    bl_label = 'Level Editing'
    bl_idname = 'VIEW3D_PT_trlau_level_editing'
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and _find_related_level_root(obj, context) is not None

    def draw(self, context):
        layout = self.layout
        layout.operator('trlau.add_level_terrain_group', text='Add TerrainGroup')
        layout.operator('trlau.add_level_markup', text='Add Markup')
        layout.operator('trlau.add_level_light', text='Add Light')
        layout.operator('trlau.add_level_intro_data', text='Add IntroData')



class VIEW3D_PT_trlau_level_editor(bpy.types.Panel):
    bl_label = 'cdcEditor'
    bl_idname = 'VIEW3D_PT_trlau_level_editor'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None:
            return False
        # The Markup metadata panel is a child of this root editor panel.
        # Model Markups do not have a level root, so keep the parent visible
        # when a model MarkUp is selected.
        if str(obj.get('trlau_type', '') or '') == 'Markup':
            return True
        if any(
            str(selected.get('trlau_type', '') or '') == 'Multiplex'
            or bool(selected.get('trlau_multiplex_empty'))
            or int(selected.get('trlau_mul_data_chunk_count', 0) or 0) > 0
            for selected in getattr(context, 'selected_objects', []) or []
        ):
            return True
        helper_type = trlau_object_type(obj)
        helper_panel_types = set(PANEL_TYPES) | {
            'Cloth', 'ClothCollision', 'ClothPoint', 'ClothCollisionRule',
            'ClothCollisionRules', 'ClothCapsuleEndpoint', 'ClothPlaneRule',
            'ClothPlaneRulePlane', 'ClothPlaneRuleSelector', 'ClothPinCollision',
        }
        return (
            _find_related_level_root(obj, context) is not None
            or model_panel._is_model_empty(obj)
            or getattr(obj, 'active_material', None) is not None
            or helper_type in helper_panel_types
            or str(obj.get('trlau_type', '') or '') == 'Multiplex'
            or int(obj.get('trlau_mul_data_chunk_count', 0) or 0) > 0
            or _has_hinfo_creation_context(context)
        )

    def draw(self, context):
        # This panel is only a parent/header for the contextual TRLAU child panels.
        # Model controls live in the Model child panel; level tools live in Level Editing.
        pass



_COMPONENT_NAME_TYPES = {
    'TerrainGroup': 'TerrainGroup',
    'TerrainLight': 'TerrainLight',
    'CameraData': 'CameraData',
    'BGInstance': 'BGInstance',
    'IntroData': 'IntroData',
    'GenericIntroData': 'GenericIntroData',
    'IntroSound': 'IntroSound',
    'UnitData': 'UnitData',
    'ADMDData': 'ADMDData',
    'Markup': 'Markup',
}
_INTRO_SPECIFIC_TYPE_NAMES = {'RewardIntroData', 'RopeObjIntroData', 'WaterVolumeIntroData'}
_BGINSTANCE_NAME_RE = re.compile(r'(?:^|_)BGObject_\d+_Instance_\d+(?:$|_|\.)')
_BGOBJECT_NAME_RE = re.compile(r'(?:^|_)BGObject_\d+(?:$|_|\.)')
_MARKUP_NAME_RE = re.compile(r'(?:^|_)Markup_\d+(?:$|_|\.)')
_INTRO_NAME_RE = re.compile(r'(?:^|_)IntroData_\d+(?:_|$)')


def _obj_name(obj) -> str:
    return str(getattr(obj, 'name', '') or '')


def _parent_name(obj) -> str:
    return str(getattr(getattr(obj, 'parent', None), 'name', '') or '')


def _name_is_component(obj, component_name: str) -> bool:
    name = _obj_name(obj)
    return name == component_name or name.endswith(f'_{component_name}')


def _has_prop_prefix(obj, prefix: str) -> bool:
    try:
        return any(str(key).startswith(prefix) for key in obj.keys())
    except Exception:
        return False


def _infer_intro_specific_type_from_name(obj) -> str:
    struct_name = trlau_intro_specific_struct_name(obj)
    if struct_name in _INTRO_SPECIFIC_TYPE_NAMES:
        return struct_name
    name = _obj_name(obj)
    for type_name in _INTRO_SPECIFIC_TYPE_NAMES:
        if name == type_name or name.endswith(f'_{type_name}'):
            return type_name
    return ''


def _infer_level_metadata_object_type(obj) -> str:
    if obj is None:
        return ''
    try:
        sfx_role = str(getattr(obj, 'trlau_sfx_role', '') or '')
    except Exception:
        sfx_role = ''
    if sfx_role in {'SFXMarker', 'SFXSound', 'SFXSpeaker'}:
        return sfx_role
    try:
        obj_type = str(obj.get('trlau_type', '') or '')
    except Exception:
        obj_type = ''
    if obj_type:
        return obj_type

    name = _obj_name(obj)
    parent_name = _parent_name(obj)

    if bool(obj.get('trlau_level_root')) or _name_is_component(obj, 'Level') or _has_prop_prefix(obj, LEVEL_METADATA_PROP_PREFIX):
        return 'Level'
    if bool(obj.get('trlau_terrain_group')) or _name_is_component(obj, 'TerrainGroup') or _has_prop_prefix(obj, 'trlau_terrain_group_'):
        return 'TerrainGroup'
    if _name_is_component(obj, 'TerrainLight') or _has_prop_prefix(obj, 'trlau_terrain_light_'):
        return 'TerrainLight'
    if _name_is_component(obj, 'CameraData') or _has_prop_prefix(obj, 'trlau_camera_'):
        return 'CameraData'
    if bool(obj.get('trlau_bginstance')) or bool(obj.get('trlau_bginstance_empty')) or _has_prop_prefix(obj, 'trlau_bginstance_') or _BGINSTANCE_NAME_RE.search(name):
        return 'BGInstance'
    if bool(obj.get('trlau_section_metadata_empty')):
        return 'SectionMetadata'
    if bool(getattr(obj, 'trlau_sfx_marker_count', 0)) or bool(getattr(obj, 'trlau_sfx_marker_unique_id', 0)) or _has_prop_prefix(obj, 'trlau_sfx_marker_') or 'trlau_sfx_marker_count' in obj:
        return 'SFXMarker'
    if bool(getattr(obj, 'trlau_sfx_wave_id', 0)) or bool(getattr(obj, 'trlau_wave_decoded_wav_path', '')) or 'trlau_sfx_wave_id' in obj or _has_prop_prefix(obj, 'trlau_wave_') or _has_prop_prefix(obj, 'trlau_sfx_ref_'):
        return 'SFXSpeaker'
    if bool(getattr(obj, 'trlau_sfx_kind', '')) or bool(getattr(obj, 'trlau_sfx_ids', '')) or bool(getattr(obj, 'trlau_sfx_speaker_count', 0)) or 'trlau_sfx_kind' in obj or 'trlau_sfx_ids' in obj or 'trlau_sfx_speaker_count' in obj:
        return 'SFXSound'
    if trlau_intro_is_sound_data(obj) or (name.endswith('_Sound') and ('IntroData_' in parent_name or parent_name.endswith('_IntroData') or _parent_name(getattr(obj, 'parent', None)).endswith('_IntroData'))):
        return 'IntroSound'
    if trlau_intro_is_generic_data(obj) or _name_is_component(obj, 'GenericIntroData'):
        return 'GenericIntroData'
    intro_specific_type = _infer_intro_specific_type_from_name(obj)
    if trlau_intro_is_specific_data(obj) or intro_specific_type:
        return intro_specific_type or 'IntroSpecificData'
    if trlau_intro_is_intro_data(obj) or _INTRO_NAME_RE.search(name):
        return 'IntroData'
    if bool(obj.get('trlau_signal')):
        return 'Signal'
    if bool(obj.get('trlau_combat_data_empty')) or _name_is_component(obj, 'CombatData'):
        return 'CombatData'
    if _MARKUP_NAME_RE.search(name) or _has_prop_prefix(obj, 'trlau_markup_'):
        return 'Markup'
    if bool(obj.get('trlau_unitdata_cine_empty')) or _has_prop_prefix(obj, 'trlau_unitdata_cine_') or '_UnitData_Cine_' in name:
        return 'UnitDataCine'
    if bool(obj.get('trlau_unitdata_fsfx_link_empty')) or _has_prop_prefix(obj, 'trlau_unitdata_fsfx_') or '_UnitData_FSFXLink_' in name:
        return 'UnitDataFSFXLink'
    if bool(obj.get('trlau_unitdata_event_variable_empty')) or _has_prop_prefix(obj, 'trlau_unitdata_event_variable_') or '_UnitData_EventVariableStorage_' in name:
        return 'UnitDataEventVariableStorage'
    if bool(obj.get('trlau_unitdata_empty')) or _name_is_component(obj, 'UnitData') or _has_prop_prefix(obj, UNIT_DATA_PROP_PREFIX):
        return 'UnitData'
    if bool(obj.get('trlau_admd_empty')) or _name_is_component(obj, 'ADMDData') or _has_prop_prefix(obj, ADMD_DATA_PROP_PREFIX):
        return 'ADMDData'
    return ''


def _is_bgobject_geometry_object(obj) -> bool:
    if obj is None or getattr(obj, 'type', '') != 'MESH':
        return False
    try:
        if obj.get('trlau_type') == 'BGObject' or bool(obj.get('trlau_bgobject')):
            return True
    except Exception:
        pass
    name = _obj_name(obj)
    parent_type = _infer_level_metadata_object_type(getattr(obj, 'parent', None))
    return (_BGOBJECT_NAME_RE.search(name) is not None) and parent_type in {'BGInstance', ''}


def _level_data_panel_label(obj) -> str:
    if obj is None:
        return 'Level Data'
    obj_type = _infer_level_metadata_object_type(obj)
    if obj_type in {'Level', 'LevelRoot'}:
        return 'Level'
    if obj_type == 'TerrainGroup' or bool(obj.get('trlau_terrain_group')):
        return 'TerrainGroup'
    if obj_type == 'TerrainLight':
        return 'TerrainLight'
    if obj_type == 'CameraData':
        return 'CameraData'
    if obj_type == 'BGInstance' or bool(obj.get('trlau_bginstance')):
        return 'BGInstance'
    if obj_type == 'IntroData' or trlau_intro_is_intro_data(obj):
        return 'IntroData'
    if obj_type == 'IntroSound' or trlau_intro_is_sound_data(obj):
        return 'IntroSound'
    if obj_type == 'GenericIntroData' or trlau_intro_is_generic_data(obj):
        return 'GenericIntroData'
    if trlau_intro_is_specific_data(obj) or obj_type in {'RewardIntroData', 'RopeObjIntroData', 'WaterVolumeIntroData'}:
        return trlau_intro_specific_struct_name(obj) or str(obj_type or 'IntroData')
    if obj_type == 'Signal' or bool(obj.get('trlau_signal')):
        return 'Signal'
    if obj_type == 'CombatData' or bool(obj.get('trlau_combat_data_empty')):
        return 'Combat Data'
    if obj_type == 'SFXMarker':
        return 'SFX Marker'
    if obj_type == 'SFXSound':
        return 'SFX Sound'
    if obj_type == 'SFXSpeaker':
        return 'SFX Speaker'
    if obj_type == 'Markup':
        return 'Markup'
    if obj_type == 'UnitData' or bool(obj.get('trlau_unitdata_empty')):
        return 'UnitData'
    if obj_type == 'UnitDataCine' or bool(obj.get('trlau_unitdata_cine_empty')):
        return 'Cine'
    if obj_type == 'UnitDataFSFXLink' or bool(obj.get('trlau_unitdata_fsfx_link_empty')):
        return 'FSFXLink'
    if obj_type == 'UnitDataEventVariableStorage' or bool(obj.get('trlau_unitdata_event_variable_empty')):
        return 'EventVariableStorage'
    if obj_type == 'ADMDData' or bool(obj.get('trlau_admd_empty')):
        return 'ADMDData'
    if bool(obj.get('trlau_section_metadata_empty')):
        role = str(obj.get('trlau_section_role', '') or '')
        if role in FSFX_SECTION_ROLES:
            return 'WaterFSFX Render Effect' if role == 'WaterFSFX' else 'FSFX Render Effect'
        return f'{role} Metadata' if role else 'Section Metadata'
    return obj_type or 'Level Data'


class VIEW3D_PT_trlau_level_metadata(bpy.types.Panel):
    bl_label = ''
    bl_idname = 'VIEW3D_PT_trlau_level_metadata'
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None:
            return False
        # Hide this panel for BGObject geometry meshes. BGInstance metadata is
        # edited on the BGInstance empty, not on the child strip meshes.
        if _is_bgobject_geometry_object(obj):
            return False
        return _infer_level_metadata_object_type(obj) in {'TerrainLight', 'CameraData', 'Level', 'LevelRoot', 'TerrainGroup', 'BGInstance', 'IntroData', 'GenericIntroData', 'RewardIntroData', 'RopeObjIntroData', 'WaterVolumeIntroData', 'IntroSpecificData', 'IntroSound', 'SFXMarker', 'SFXSound', 'SFXSpeaker', 'UnitData', 'UnitDataCine', 'UnitDataFSFXLink', 'UnitDataEventVariableStorage', 'ADMDData', 'Markup', 'Signal', 'CombatData', 'SectionMetadata'}

    def draw_header(self, context):
        obj = getattr(context, 'object', None)
        self.layout.label(text=_level_data_panel_label(obj))

    def _draw_idprop(self, layout, obj, key: str, text: str):
        if str(key).startswith(('trlau_sfx_', 'trlau_wave_')) and hasattr(obj, key):
            layout.prop(obj, key, text=text)
        elif key in obj:
            layout.prop(obj, f'["{key}"]', text=text)

    def _find_cine_section_metadata_child(self, obj):
        try:
            children = list(getattr(obj, 'children', []) or [])
        except Exception:
            children = []
        for child in children:
            try:
                if str(child.get('trlau_section_role', '') or '') == 'Cine':
                    return child
                if bool(child.get('trlau_section_metadata_empty')) and str(child.get('trlau_type', '') or '') == 'CineSection':
                    return child
            except Exception:
                continue
        return None

    def _draw_cine_data_resource(self, layout, obj, *, prefer_section_child: bool = False):
        data_obj = obj
        prefix = 'trlau_unitdata_cine_'
        name_key = 'trlau_unitdata_cine_name'
        structured_key = 'trlau_unitdata_cine_structured'

        if prefer_section_child:
            section_obj = self._find_cine_section_metadata_child(obj)
            if section_obj is not None and ('trlau_cine_structured' in section_obj or 'trlau_cine_name' in section_obj):
                data_obj = section_obj
                prefix = 'trlau_cine_'
                name_key = 'trlau_cine_name'
                structured_key = 'trlau_cine_structured'
            elif 'trlau_cine_structured' in obj or 'trlau_cine_name' in obj:
                prefix = 'trlau_cine_'
                name_key = 'trlau_cine_name'
                structured_key = 'trlau_cine_structured'
        elif 'trlau_cine_structured' in obj or 'trlau_cine_name' in obj:
            prefix = 'trlau_cine_'
            name_key = 'trlau_cine_name'
            structured_key = 'trlau_cine_structured'

        structured = False
        try:
            structured = bool(int(data_obj.get(structured_key, 0) or 0))
        except Exception:
            structured = bool(data_obj.get(structured_key, False))

        cine_box = layout.box()
        cine_box.label(text='CineData Resource')
        if not structured:
            cine_box.label(text='No parsed CineData resource found for this Cine entry.', icon='INFO')
            if data_obj is not obj and 'trlau_section_file_name' in data_obj:
                self._draw_idprop(cine_box, data_obj, 'trlau_section_file_name', 'Section File')
            return

        self._draw_idprop(cine_box, data_obj, name_key, 'Name')
        for suffix, label in (
            ('version_major', 'Version Major'),
            ('version_minor', 'Version Minor'),
            ('stream_unit_id', 'Stream Unit ID'),
            ('id', 'Cine ID'),
            ('end_time', 'End Time'),
            ('spline_count', 'Spline Count'),
            ('command_count', 'Command Count'),
            ('command_types', 'Command Types'),
            ('rebuildable', 'Editable/Rebuildable'),
        ):
            self._draw_idprop(cine_box, data_obj, f'{prefix}{suffix}', label)

        command_keys = [
            ('command_type', 'Command Type'),
            ('command_unit_id', 'Unit ID'),
            ('command_load', 'Load'),
            ('command_camera_control', 'Camera Control'),
            ('command_channels', 'Channels'),
            ('command_positions_after_playback', 'Positions After Playback'),
            ('command_end_trigger_id', 'End Trigger ID'),
            ('command_data_pointer', 'Data Pointer'),
            ('command_data_size', 'Data Size'),
            ('command_cinematic_name', 'Cinematic Name'),
        ]
        if any(f'{prefix}{suffix}' in data_obj for suffix, _label in command_keys):
            command_box = cine_box.box()
            command_box.label(text='CinematicCineCommand')
            for suffix, label in command_keys:
                self._draw_idprop(command_box, data_obj, f'{prefix}{suffix}', label)

    def _draw_fsfx_scalar(self, layout, obj, suffix: str, text: str):
        key = f'{FSFX_PROP_PREFIX}{suffix}'
        if key not in obj:
            return
        prop_name = _fsfx_colour_prop_name(suffix)
        if suffix in {item[0] for item in FSFX_COLOUR_UI_FIELDS} and hasattr(obj, prop_name):
            layout.prop(obj, prop_name, text=text)
        else:
            layout.prop(obj, f'["{key}"]', text=text)

    def _draw_fsfx_render_effect(self, layout, obj):
        if not has_fsfx_render_effect_properties(obj):
            try:
                is_fsfx_section = bool(obj.get('trlau_section_metadata_empty')) and str(obj.get('trlau_section_role', '') or '') in FSFX_SECTION_ROLES
            except Exception:
                is_fsfx_section = False
            if is_fsfx_section:
                fsfx_box = layout.box()
                fsfx_box.label(text='RenderFSEffect')
                fsfx_box.label(text='No typed FSFX properties found on this section.', icon='INFO')
                fsfx_box.operator('trlau.fsfx_decode_stored_blob', text='Decode Stored FSFX Blob', icon='IMPORT')
            return
        fsfx_box = layout.box()
        fsfx_box.label(text='RenderFSEffect')

        colour_box = fsfx_box.box()
        colour_box.label(text='sColourDOF')
        for key, label in (
            ('colour_dof_enabled', 'm_isEnabled'),
            ('colour_dof_use_zbuffer', 'm_isUseZBuffer'),
            ('colour_dof_colour_convert_backbuffer', 'm_isColourConvertBackBuffer'),
            ('colour_dof_use_object_pointer', 'm_isUseObjectPointer'),
            ('colour_dof_blur', 'm_vBlur'),
            ('colour_dof_alpha', 'm_vAlpha'),
            ('colour_dof_colour', 'm_nColour'),
            ('colour_dof_object_point', 'm_objectPoint'),
            ('colour_dof_object_radius', 'm_objectRadius'),
        ):
            self._draw_fsfx_scalar(colour_box, obj, key, label)

        finish_box = fsfx_box.box()
        finish_box.label(text='sFinish')
        for key, label in (
            ('finish_enabled', 'm_isEnabled'),
            ('finish_colour_convert', 'm_isColourConvert'),
            ('finish_blur_strength', 'm_vBlurStrength'),
            ('finish_bright', 'm_vBright'),
            ('finish_pad1', 'm_pad1'),
            ('finish_pad2', 'm_pad2'),
        ):
            self._draw_fsfx_scalar(finish_box, obj, key, label)

        extra_box = fsfx_box.box()
        extra_box.label(text='Other FS Effects')
        for group_label, fields in (
            ('Noise', (
                ('noise_enabled', 'enabled'), ('noise_colour', 'colour'), ('noise_min_alpha', 'minAlpha'), ('noise_max_alpha', 'maxAlpha'), ('noise_tex_step', 'texStep'),
            )),
            ('Bloom', (
                ('bloom_enabled', 'enabled'), ('bloom_blur_strength', 'blurStrength'), ('bloom_bright', 'bright'),
            )),
            ('Feedback', (
                ('feedback_enabled', 'enabled'), ('feedback_front_alpha', 'frontAlpha'), ('feedback_back_alpha', 'backAlpha'), ('feedback_blur_strength', 'blurStrength'),
                ('feedback_scale_x', 'scaleX'), ('feedback_scale_y', 'scaleY'), ('feedback_shear_x', 'shearX'), ('feedback_shear_y', 'shearY'), ('feedback_colour', 'colour'),
            )),
            ('Water', (
                ('water_enabled', 'enabled'), ('water_blur', 'blur'), ('water_amplitude', 'amplitude'), ('water_speed', 'speed'), ('water_colour', 'colour'),
            )),
            ('HDR', (
                ('hdr_enabled', 'enabled'), ('hdr_reduce_to_normal_brightness', 'reduceToNormalBrightness'), ('hdr_target_brightness', 'targetBrightness'),
                ('hdr_normal_brightness', 'normalBrightness'), ('hdr_brightness_adjust_rate_up', 'adjustRateUp'), ('hdr_brightness_adjust_rate_down', 'adjustRateDown'),
                ('hdr_target_brightness_wait', 'targetBrightnessWait'), ('hdr_max_blur', 'maxBlur'),
            )),
            ('HeatHaze', (
                ('heat_haze_enabled', 'enabled'), ('heat_haze_fullscreen', 'fullscreen'), ('heat_haze_horizontal_compression', 'horizontalCompression'),
                ('heat_haze_vertical_compression', 'verticalCompression'), ('heat_haze_turbulence', 'turbulence'), ('heat_haze_speed', 'speed'),
                ('heat_haze_opacity', 'opacity'), ('heat_haze_colour', 'colour'),
            )),
            ('MultiFocusBlur', (
                ('multi_focus_blur_enabled', 'enabled'), ('multi_focus_blur_strength', 'blurStrength'),
            )),
        ):
            sub = extra_box.box()
            sub.label(text=group_label)
            for key, label in fields:
                self._draw_fsfx_scalar(sub, obj, key, label)

        table_box = fsfx_box.box()
        table_box.label(text='Curve/Palette Tables')
        table_box.prop(obj, 'trlau_fsfx_table_selector', text='Table')
        table_box.prop(obj, 'trlau_fsfx_table_page', text='Page')
        active_table = _fsfx_active_table_identifier(obj)
        page = _get_fsfx_table_page(obj)
        start_index = page * FSFX_TABLE_PAGE_SIZE
        table_box.label(text=f'{fsfx_table_label(active_table)} indices {start_index}-{start_index + FSFX_TABLE_PAGE_SIZE - 1}: {fsfx_table_summary_for_identifier(obj, active_table)}')
        for row_index in range(4):
            row = table_box.row(align=True)
            for col_index in range(4):
                slot_index = row_index * 4 + col_index
                absolute_index = start_index + slot_index
                row.prop(obj, f'trlau_fsfx_table_value_{slot_index:02d}', text=str(absolute_index))
    def _draw_level_metadata_prop(self, layout, obj, field_name: str):
        self._draw_idprop(layout, obj, f'{LEVEL_METADATA_PROP_PREFIX}{field_name}', field_name)

    def _draw_unitdata_prop(self, layout, obj, field_name: str, label: str):
        self._draw_idprop(layout, obj, f'{UNIT_DATA_PROP_PREFIX}{field_name}', label)

    def _draw_admd_instance(self, layout, obj, index: int):
        base = f'{ADMD_DATA_PROP_PREFIX}light_{index:03d}_'
        box = layout.box()
        box.label(text=f'Light {index}')
        for field_name, label in (
            ('transform_row0', 'transform row 0'),
            ('transform_row1', 'transform row 1'),
            ('transform_row2', 'transform row 2'),
            ('transform_row3', 'transform row 3'),
            ('property_type', 'propertyType'),
            ('light_resource', 'LightResource'),
            ('intensity', 'Intensity'),
            ('rim_intensity', 'rimIntensity'),
            ('specular_intensity', 'specularIntensity'),
            ('enable_per_instance_intensity', 'bEnablePerInstanceIntensity'),
            ('light_color', 'lightColor'),
            ('enable_per_instance_color', 'bEnablePerInstanceColor'),
            ('range', 'range'),
            ('enable_per_instance_range', 'bEnablePerInstanceRange'),
            ('ambient_color', 'ambientColor'),
            ('ambient_percentage', 'ambientPercentage'),
            ('enable_per_instance_ambient', 'bEnablePerInstanceAmbient'),
            ('enable_per_instance_shadow_bias', 'bEnablePerInstanceShadowBias'),
            ('shadow_map_bias', 'shadowMapBias'),
            ('shadow_map_slope_bias', 'shadowMapSlopeBias'),
            ('cull_light_distance', 'cullLightDistance'),
            ('enable_light_culling', 'bEnableLightCulling'),
            ('cull_light_fade_distance', 'cullLightFadeDistance'),
            ('shadow_cull_distance', 'shadowCullDistance'),
            ('enable_shadow_lod', 'bEnableShadowLOD'),
            ('active_in_gameplay', 'bActiveInGameplay'),
            ('active_in_cinematics', 'bActiveInCinematics'),
            ('affects_player', 'bAffectsPlayer'),
            ('affects_intros', 'bAffectsIntros'),
            ('affects_bgobjects', 'bAffectsBGObjects'),
            ('affects_terrain', 'bAffectsTerrain'),
            ('affects_neighboring_units', 'bAffectsNeighboringUnits'),
            ('affects_water', 'bAffectsWater'),
            ('cinematic_name_ptr', 'cinematicName ptr'),
            ('cinematic_name', 'cinematicName'),
            ('scene_light', 'pSceneLight'),
        ):
            key = base + field_name
            if key in obj:
                self._draw_idprop(box, obj, key, label)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        obj_type = _infer_level_metadata_object_type(obj)

        if obj_type == 'TerrainGroup' or bool(obj.get('trlau_terrain_group')):
            data_box = layout.box()
            data_box.label(text='TerrainGroup Properties')
            data_box.prop(obj, 'trlau_terrain_group_flags_ui', text='Flags')
            self._draw_idprop(data_box, obj, 'trlau_terrain_group_unique_id', 'uniqueID')
            self._draw_idprop(data_box, obj, 'trlau_terrain_group_spline_id', 'splineID')
        elif obj_type == 'TerrainLight':
            data_box = layout.box()
            data_box.label(text='Light Properties')
            self._draw_idprop(data_box, obj, 'trlau_terrain_light_type', 'Type')
            self._draw_idprop(data_box, obj, 'trlau_terrain_light_id', 'ID')
            data_box.prop(obj, 'trlau_terrain_light_radius_ui', text='Radius')
            data_box.prop(obj, 'trlau_terrain_light_multiplier_ui', text='Multiplier')
            self._draw_idprop(data_box, obj, 'trlau_terrain_light_hotspot_angle', 'Hotspot Angle')
            self._draw_idprop(data_box, obj, 'trlau_terrain_light_falloff_angle', 'Falloff Angle')
        elif obj_type == 'CameraData':
            data_box = layout.box()
            data_box.label(text='Camera Properties')
            self._draw_idprop(data_box, obj, 'trlau_camera_id', 'ID')
            self._draw_idprop(data_box, obj, 'trlau_camera_flags', 'Flags')
            self._draw_idprop(data_box, obj, 'trlau_camera_rx', 'Rot X')
            self._draw_idprop(data_box, obj, 'trlau_camera_ry', 'Rot Y')
            self._draw_idprop(data_box, obj, 'trlau_camera_rz', 'Rot Z')
            self._draw_idprop(data_box, obj, 'trlau_camera_tx', 'Target X')
            self._draw_idprop(data_box, obj, 'trlau_camera_ty', 'Target Y')
            self._draw_idprop(data_box, obj, 'trlau_camera_tz', 'Target Z')
        elif obj_type in {'Level', 'LevelRoot'}:
            data_box = layout.box()
            data_box.label(text='Level Properties')
            data_box.prop(obj, 'trlau_level_game', text='Game')
            scene = context.scene
            row = data_box.row(align=True)
            row.scale_y = 0.9
            for tab_id, tab_label, _tab_fields in LEVEL_METADATA_GROUPS:
                row.prop_enum(scene, 'trlau_level_metadata_tab', tab_id, text=tab_label)
            active_tab = getattr(scene, 'trlau_level_metadata_tab', LEVEL_METADATA_GROUPS[0][0])
            active_group = next((group for group in LEVEL_METADATA_GROUPS if group[0] == active_tab), LEVEL_METADATA_GROUPS[0])
            group_box = data_box.box()
            group_box.label(text=active_group[1])
            for field_name in active_group[2]:
                self._draw_level_metadata_prop(group_box, obj, field_name)
        elif obj_type == 'BGInstance' or obj.get('trlau_bginstance_empty') or obj.get('trlau_bginstance'):
            data_box = layout.box()
            data_box.label(text='BGInstance Properties')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_flags', 'flags')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_id', 'id')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_bgflags', 'BGflags')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_target_frame', 'targetFrame')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_clip_beg', 'clipBeg')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_clip_end', 'clipEnd')
            self._draw_idprop(data_box, obj, 'trlau_bginstance_link_seg', 'linkSeg')
            draw_multispline_panel(layout, obj)
        elif obj_type == 'SectionMetadata' or obj.get('trlau_section_metadata_empty'):
            role = str(obj.get('trlau_section_role', obj_type) or obj_type)
            if role in FSFX_SECTION_ROLES:
                self._draw_fsfx_render_effect(layout, obj)
            elif role == 'Cine':
                self._draw_cine_data_resource(layout, obj)
            else:
                data_box = layout.box()
                data_box.label(text=f'{role} Metadata')
                self._draw_idprop(data_box, obj, 'trlau_section_role', 'role')
                self._draw_idprop(data_box, obj, 'trlau_section_original_id', 'sectionID')
                self._draw_idprop(data_box, obj, 'trlau_section_type', 'sectionType')
                self._draw_idprop(data_box, obj, 'trlau_section_file_name', 'sourceFile')
                self._draw_idprop(data_box, obj, 'trlau_section_version_id', 'versionID')
                self._draw_idprop(data_box, obj, 'trlau_section_resource_type', 'resourceType')
                self._draw_idprop(data_box, obj, 'trlau_section_spec_mask', 'specMask')
        elif obj_type == 'IntroData' or trlau_intro_is_intro_data(obj):
            intro_data = _intro_data_for_ui(obj)
            data_box = layout.box()
            data_box.label(text='IntroData Properties')
            if intro_data is None:
                data_box.label(text='No registered IntroData fields are available.', icon='INFO')
            else:
                data_box.prop(intro_data, 'object_id', text='objectID')
                data_box.prop(intro_data, 'intro_num', text='introNum')
                data_box.prop(intro_data, 'unique_id', text='uniqueID')
                data_box.prop(intro_data, 'max_radius', text='maxRad')
                data_box.prop(intro_data, 'intro_flags', text='introFlags')
            draw_multispline_panel(layout, obj)
        elif obj_type == 'SFXMarker':
            data_box = layout.box()
            data_box.label(text='SFX Marker')
            for attr, label in SFX_MARKER_FIELDS:
                self._draw_idprop(data_box, obj, f'trlau_sfx_{attr}', label)
        elif obj_type == 'SFXSound':
            data_box = layout.box()
            data_box.label(text='SFX Sound Entry')
            row = data_box.row(align=True)
            row.operator('trlau.play_sfx_sound', text='Play First Decoded Wave', icon='PLAY')
            row.operator('trlau.stop_sfx_sound', text='Stop', icon='PAUSE')
            for attr, label in SFX_SOUND_BASE_FIELDS:
                self._draw_idprop(data_box, obj, f'trlau_sfx_{attr}', label)
            kind = str(getattr(obj, 'trlau_sfx_kind', obj.get('trlau_sfx_kind', '')) or '')
            if kind == 'periodic':
                fields = SFX_SOUND_PERIODIC_FIELDS
            elif kind in {'event', 'one_shot'}:
                fields = SFX_SOUND_EVENT_FIELDS
            elif kind == 'stream':
                fields = SFX_SOUND_STREAM_FIELDS
            else:
                fields = SFX_SOUND_PERIODIC_FIELDS + SFX_SOUND_EVENT_FIELDS + SFX_SOUND_STREAM_FIELDS
            for attr, label in fields:
                self._draw_idprop(data_box, obj, f'trlau_sfx_{attr}', label)
        elif obj_type == 'SFXSpeaker':
            data_box = layout.box()
            data_box.label(text='SFX Speaker')
            row = data_box.row(align=True)
            row.operator('trlau.play_sfx_sound', text='Play', icon='PLAY')
            row.operator('trlau.stop_sfx_sound', text='Stop', icon='PAUSE')
            for attr, label in SFX_SPEAKER_FIELDS:
                self._draw_idprop(data_box, obj, f'trlau_sfx_{attr}', label)
            ref_box = layout.box()
            ref_box.label(text='Resolved SFX Reference')
            for attr, label in SFX_REFERENCE_FIELDS:
                self._draw_idprop(ref_box, obj, f'trlau_sfx_ref_{attr}', label)
            wave_box = layout.box()
            wave_box.label(text='Decoded Wave')
            for attr, label in WAVE_METADATA_FIELDS:
                self._draw_idprop(wave_box, obj, f'trlau_wave_{attr}', label)
        elif obj_type == 'IntroSound' or trlau_intro_is_sound_data(obj):
            sound_data = _intro_sound_data_for_ui(obj)
            data_box = layout.box()
            data_box.label(text='Sfx Sound Properties')
            if sound_data is None:
                data_box.label(text='No registered IntroSound fields are available.', icon='INFO')
            else:
                data_box.prop(sound_data, 'sound_id', text='soundID / sectionID')
                data_box.prop(sound_data, 'format', text='format')
            for attr, label in INTRO_SOUND_SFX_FIELDS:
                self._draw_idprop(data_box, obj, f'trlau_sfx_{attr}', label)
        elif obj_type == 'GenericIntroData' or trlau_intro_is_generic_data(obj):
            generic_data = _intro_generic_data_for_ui(obj)
            data_box = layout.box()
            data_box.label(text='GenericIntroData Properties')
            if generic_data is None:
                data_box.label(text='No registered GenericIntroData fields are available.', icon='INFO')
            else:
                for attr, label in INTRO_GENERIC_FIELDS:
                    data_box.prop(generic_data, attr, text=label)
        elif trlau_intro_is_specific_data(obj) or obj_type in {'RewardIntroData', 'RopeObjIntroData', 'WaterVolumeIntroData', 'IntroSpecificData'}:
            specific_data = _intro_specific_data_for_ui(obj)
            data_box = layout.box()
            struct_name = trlau_intro_specific_struct_name(obj) or str(obj_type)
            data_box.label(text=f'{struct_name} Properties')
            if specific_data is None:
                data_box.label(text='No registered IntroSpecificData fields are available.', icon='INFO')
            else:
                data_box.prop(specific_data, 'data_type', text='type')
                for attr, label in INTRO_SPECIFIC_FIELDS.get(struct_name, ()):
                    if struct_name == 'RewardIntroData' and attr == 'reward_type':
                        data_box.prop(obj, 'trlau_intro_reward_type_ui', text=label)
                        continue
                    data_box.prop(specific_data, attr, text=label)
        elif obj_type == 'CombatData' or obj.get('trlau_combat_data_empty'):
            data = getattr(obj, 'trlau_combat_data', None)
            box = layout.box()
            box.label(text='Level Combat Data')
            if data is None:
                box.label(text='No typed combat data is registered for this object.', icon='INFO')
                return
            box.label(text='Embedded in this .blend' if data.embedded else 'No embedded backing data')
            counts = box.column(align=True)
            counts.label(text=f'Attack waves: {int(data.attack_wave_count)}')
            counts.label(text=f'Attack-wave groups: {int(data.attack_wave_group_count)}')
            counts.label(text=f'Combat doors: {int(data.combat_door_count)}')
            counts.label(text=f'PMarkers: {int(data.pmarker_count)}')
            counts.label(text=f'VMarkers: {int(data.vmarker_count)}')
            counts.label(text='Global spline-camera data: present' if data.spline_camera_data_present else 'Global spline-camera data: absent')
            box.label(text='Known fields are editable; variable payloads and relocations are preserved.', icon='INFO')
            for index, entry in enumerate(list(data.attack_waves or [])):
                _draw_level_attack_wave_entry(layout, entry, f'Attack wave {index}')
            for index, entry in enumerate(list(data.attack_wave_groups or [])):
                _draw_level_attack_wave_entry(layout, entry, f'Attack-wave group {index}')
            for index, entry in enumerate(list(data.pmarkers or [])):
                _draw_level_pmarker_entry(layout, entry, f'PMarker {index}')
        elif obj_type == 'Signal' or obj.get('trlau_signal'):
            signal_data = getattr(obj, 'trlau_signal_data', None)
            if signal_data is None:
                data_box = layout.box()
                data_box.label(text='Signal Properties')
                data_box.label(text='No typed signal data is registered for this object.', icon='INFO')
                return

            data_box = layout.box()
            data_box.label(text='Signal')
            data_box.prop(signal_data, 'signal_id')
            data_box.prop(signal_data, 'list_index')
            data_box.prop(signal_data, 'face_id')
            data_box.prop(signal_data, 'startGoingIntoWaterSignal')
            data_box.prop(signal_data, 'startGoingOutOfWaterSignal')

            flags_box = layout.box()
            flags_box.label(text='Signal Flags')
            flags_box.prop(signal_data, 'raw_flags')
            for field_name in SIGNAL_FLAG_FIELD_NAMES:
                flags_box.prop(signal_data, field_name)

            values_box = layout.box()
            values_box.label(text='Signal Data')
            for field_name in SIGNAL_VALUE_FIELD_NAMES:
                values_box.prop(signal_data, field_name)

            pointer_box = layout.box()
            pointer_box.label(text='Signal Pointers')
            pointer_box.prop(signal_data, 'cameraLinkSignalIndex')
            for prop_name, label in SIGNAL_POINTER_FIELD_GROUPS:
                _draw_signal_spline_pointer(pointer_box, signal_data, prop_name, label)
            for prop_name, label in SIGNAL_ATTACK_WAVE_COLLECTIONS:
                _draw_signal_attack_wave_collection(pointer_box, signal_data, prop_name, label)

            spline_box = layout.box()
            spline_box.label(text='splineCamParams')
            _draw_signal_spline_fields(spline_box, signal_data.splineCamParams)
        elif obj_type == 'Markup':
            data_box = layout.box()
            data_box.label(text='Markup Properties')
            data_box.prop(obj, 'trlau_markup_type_ui', text='Type')
            row = data_box.row(align=True)
            row.operator('trlau.reverse_markup_direction', text='Reverse Direction', icon='ARROW_LEFTRIGHT')
            markup_game = _level_game_for_object(obj, context)
            if markup_game == 'anniversary' and 'trlau_markup_animated_segment' in obj:
                self._draw_idprop(data_box, obj, 'trlau_markup_animated_segment', 'AnimatedSegment')
            if markup_game in {'legend', 'anniversary'}:
                markup_data = _markup_data_for_ui(obj)
                if markup_data is not None:
                    data_box.prop(markup_data, 'intro_id', text='Intro ID')
        elif obj_type == 'UnitDataCine' or obj.get('trlau_unitdata_cine_empty'):
            # The UnitData Cine entry is the CineData resource referenced by m_pCines.
            # Show the parsed resource fields, not pointer/index bookkeeping.
            self._draw_cine_data_resource(layout, obj, prefer_section_child=True)
        elif obj_type == 'UnitDataFSFXLink' or obj.get('trlau_unitdata_fsfx_link_empty'):
            data_box = layout.box()
            data_box.label(text='FSFXLink Properties')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_fsfx_index', 'Index')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_fsfx_id', 'ID')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_fsfx_alpha', 'alpha')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_fsfx_enabled', 'bEnabled')
        elif obj_type == 'UnitDataEventVariableStorage' or obj.get('trlau_unitdata_event_variable_empty'):
            data_box = layout.box()
            data_box.label(text='EventVariableStorage Properties')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_event_variable_index', 'Index')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_event_variable_variable', 'Variable')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_event_variable_medium_value', 'MediumValue')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_event_variable_easy_value', 'EasyValue')
            self._draw_idprop(data_box, obj, 'trlau_unitdata_event_variable_hard_value', 'HardValue')
        elif obj_type == 'UnitData' or obj.get('trlau_unitdata_empty'):
            data_box = layout.box()
            data_box.label(text='UnitData Properties')
            level_game = _level_game_for_object(obj, context)
            for group_label, fields in UNIT_DATA_GROUPS:
                available_fields = [
                    (field_name, label)
                    for field_name, label in fields
                    if f'{UNIT_DATA_PROP_PREFIX}{field_name}' in obj and _unitdata_field_visible_for_game(group_label, field_name, level_game)
                ]
                if not available_fields:
                    continue
                group_box = data_box.box()
                group_box.label(text=group_label)
                for field_name, label in available_fields:
                    self._draw_idprop(group_box, obj, f'{UNIT_DATA_PROP_PREFIX}{field_name}', label)
        elif obj_type == 'ADMDData' or obj.get('trlau_admd_empty'):
            data_box = layout.box()
            data_box.label(text='ADMDData Properties')
            for field_name, label in (
                ('id', 'ID'),
                ('light_instance_count', 'm_lightInstanceCount'),
            ):
                key = f'{ADMD_DATA_PROP_PREFIX}{field_name}'
                if key in obj:
                    self._draw_idprop(data_box, obj, key, label)

classes = (
    TRLAU_OT_fsfx_decode_stored_blob,
    TRLAU_OT_reverse_markup_direction,
    TRLAU_OT_add_level_terrain_group,
    TRLAU_OT_add_level_markup,
    TRLAU_OT_add_level_light,
    TRLAU_OT_add_level_intro_data,
    VIEW3D_PT_trlau_level_editor,
    VIEW3D_PT_trlau_level_editing,
    VIEW3D_PT_trlau_level_metadata,
)
