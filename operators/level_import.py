from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
import tempfile
import wave
from contextlib import ExitStack
from pathlib import Path

import bpy
import mathutils

try:
    import aud
except Exception:
    aud = None

from ..core.binary import BinaryReader
from ..core.log import logger
from ..core.markup_utils import add_markup_bbox_wire
from ..core.fsfx_render_effect import store_fsfx_render_effect_properties
from ..core.cine_format import parse_cine_section, first_cinematic_command
from ..platforms.common.drm_container import DRMContainerParser
from ..platforms.pc.area_dbase import AreaDBaseData, AreaDBaseParser, _convert_level_position, convert_area_dbase_center_position
from ..platforms.pc.tr7ae_level import TRLevelBuilder, TRLevelParser
from ..platforms.psp.tr7ae_level import TRPSPLevelParser
from ..platforms.pc.tr7ae_object import TRObjectParser
from ..platforms.common.section import TRSectionParser
from ..ui.level_panel import (
    trlau_load_multispline_property,
    trlau_set_intro_data_property,
    trlau_set_intro_generic_property,
    trlau_set_intro_specific_property,
    trlau_set_intro_sound_property,
    trlau_intro_data_to_dict,
    SFX_PANEL_FLOAT_NAMES,
    SFX_PANEL_STRING_NAMES,
    SFX_PANEL_VECTOR_NAMES,
)


LEVEL_METADATA_FIELDS = (
    'waterZLevel',
    'backColorR',
    'backColorG',
    'backColorB',
    'spectralColorR',
    'spectralColorG',
    'spectralColorB',
    'spectralFXAlways',
    'waterColorR',
    'waterColorG',
    'waterColorB',
    'waterBlend',
    'farPlane',
    'fogFar',
    'fogNear',
    'spectralFarPlane',
    'spectralFogFar',
    'spectralFogNear',
    'waterFarPlane',
    'waterFogFar',
    'waterFogNear',
    'UnderwaterFXAlpha',
    'UnderwaterFXMovement',
    'UnderwaterFXSpeed',
    'SpectralFXAlpha',
    'SpectralFXIncrease',
    'SpectralFXCenter',
    'WaterFXAlpha',
    'WaterFXMovement',
    'WaterFXSpeed',
    'WaterFSFX',
    'flags',
    'worldName',
    'unitFlags',
    'guiID',
    'streamUnitID',
    'playerName',
    'playerObjectID',
)

LEVEL_METADATA_PROP_PREFIX = 'trlau_level_'


SECTION_BLOB_CHUNK_SIZE = 30000
SECTION_BLOB_PROP_PREFIX = 'trlau_section_blob_'
TERRAIN_LIGHT_GRID_BLOB_PROP_PREFIX = 'trlau_terrain_light_grid_blob_'

_SFX_STREAM_STRUCT = '<IBBBBIBbBxfHHfffIb'
_SFX_STREAM_SIZE = struct.calcsize(_SFX_STREAM_STRUCT)
_SFX_TONE_STRUCT = '<IBBBBIBBBxfHHfffIBBBB'
_SFX_TONE_SIZE = struct.calcsize(_SFX_TONE_STRUCT)
_SFX_NAME_POINTER_OFFSET = 4
_SFX_DEFAULT_NAME_OFFSET = 4 + ((_SFX_STREAM_SIZE + 3) & ~3)

INTRO_GENERIC_FIELDS = (
    ('in_view_remove_dist', 'InViewRemoveDist', 0.0),
    ('out_of_view_remove_dist', 'OutOfViewRemoveDist', 0.0),
    ('use_model', 'UseModel', 0),
    ('pad0', 'pad0', 0),
    ('pad1', 'pad1', 0),
    ('flags', 'Flags', 0),
    ('attached_instance', 'AttachedInstance', 0),
    ('swing_length', 'SwingLength', 0.0),
    ('dtp_camera_id', 'DTPCameraID', 0),
)
INTRO_SPECIFIC_TYPE_NAMES = {12: 'RopeObjIntroData', 13: 'WaterVolumeIntroData', 17: 'RewardIntroData'}
INTRO_SPECIFIC_FIELDS = {
    17: (('reward_type', 'rewardType', 0), ('unique_id', 'uniqueID', 0), ('sound_id', 'soundID', 0)),
    12: (
        ('top_connect_type', 'topConnectType', 0), ('bottom_connect_type', 'bottomConnectType', 0),
        ('top_connect_instance', 'topConnectInstance', 0), ('top_connect_model_index', 'topConnectModelIndex', 0),
        ('top_connect_model_marker_index', 'topConnectModelMarkerIndex', 0), ('bottom_connect_instance', 'bottomConnectInstance', 0),
        ('bottom_connect_model_index', 'bottomConnectModelIndex', 0), ('bottom_connect_model_marker_index', 'bottomConnectModelMarkerIndex', 0),
        ('collision_plane_instance0', 'collisionPlaneInstance0', 0), ('collision_plane_instance1', 'collisionPlaneInstance1', 0),
        ('rope_camera_dtpid', 'ropeCameraDTPID', 0), ('rope_camera_overrides_movement', 'ropeCameraOverridesMovement', 0),
        ('sound_input_min', 'soundInputMin', 0.0), ('sound_input_max', 'soundInputMax', 0.0),
        ('render_width', 'renderWidth', 0.0), ('render_length_per_v', 'renderLengthPerV', 0.0),
        ('render_u_width', 'renderUWidth', 0.0), ('render_color', 'renderColor', 0),
        ('render_model', 'renderModel', 0), ('render_texture_main', 'renderTextureMain', 0),
        ('render_texture_top', 'renderTextureTop', 0), ('render_texture_bottom', 'renderTextureBottom', 0),
    ),
    13: (
        ('water_depth', 'waterDepth', 0.0), ('water_inflow', 'waterInflow', 0), ('water_outflow', 'waterOutflow', 0),
        ('water_speed', 'waterSpeed', 0.0), ('flow_radius', 'flowRadius', 0.0),
        ('bob_height', 'bobHeight', 0), ('bob_frequency', 'bobFrequency', 0), ('bob_grid_size', 'bobGridSize', 0.0),
        ('priority', 'priority', 0), ('water_flags', 'waterFlags', 0),
    ),
}


UNIT_DATA_PROP_PREFIX = 'trlau_unitdata_'
ADMD_DATA_PROP_PREFIX = 'trlau_admd_'
MARKUP_FLAG_PERCH = 262144
MARKUP_FLAG_WATER = 2147483648
MARKUP_BBOX_FLAGS = MARKUP_FLAG_PERCH | MARKUP_FLAG_WATER
_BGOBJECT_NAME_RE = re.compile(r'(?:^|_)BGObject_(\d+)(?:$|_|\.)')
_BGINSTANCE_NAME_RE = re.compile(r'(?:^|_)BGObject_(\d+)_Instance_(\d+)(?:$|_|\.)')
_MARKUP_NAME_RE = re.compile(r'(?:^|_)Markup_(\d+)(?:$|_|\.)')
_TERRAIN_GROUP_NAME_RE = re.compile(r'(?:^|_)TerrainGroup_(\d+)(?:$|_|\.)')
_TERRAIN_GROUP_MESH_NAME_RE = re.compile(r'(?:^|_)TerrainGroup_(\d+)_(?:Mesh|Strip_\d+)(?:$|_|\.)')
_TERRAIN_GROUP_COLLISION_NAME_RE = re.compile(r'(?:^|_)TerrainGroup(?:_Collision)?_(\d+)(?:_Collision)?(?:$|_|\.)')
_INTRO_NAME_RE = re.compile(r'(?:^|_)Intro_(\d+)(?:_|$)')


def _add_markup_bbox_wire(curve_data, bbox, markup_position, *, level_space: bool, bbox_is_local: bool = False) -> bool:
    return add_markup_bbox_wire(curve_data, bbox, markup_position, level_space=level_space, bbox_is_local=bbox_is_local)


class LevelImporterMixin:

    @staticmethod
    def _prefixed_name(prefix: str, name: str) -> str:
        prefix = str(prefix or '').strip()
        if not prefix:
            return name
        normalized = f'{prefix}_'
        return name if name.startswith(normalized) else f'{prefix}_{name}'

    def _get_or_create_intro_empty(self, collection, drm_name: str, intro, object_name: str):
        empty_name = self._prefixed_name(drm_name, f'Intro_{int(getattr(intro, "index", 0)):03d}_{object_name}')
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 48.0
        empty.parent = self._get_or_create_component_empty(collection, 'IntroData')
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        self._set_custom_property(empty, 'trlau_level', True)
        self._set_custom_property(empty, 'trlau_type', 'IntroData')
        trlau_set_intro_data_property(
            empty,
            index=int(getattr(intro, 'index', -1)),
            object_id=int(getattr(intro, 'object_id', -1)),
            intro_num=int(getattr(intro, 'intro_num', -1)),
            unique_id=int(getattr(intro, 'unique_id', -1)),
            start_frame=int(getattr(intro, 'start_frame', 0)),
            end_frame=int(getattr(intro, 'end_frame', 0)),
            intro_flags=int(getattr(intro, 'intro_flags', 0)),
            attached_vmo=int(getattr(intro, 'attached_vmo', 0)),
            multi_spline=int(getattr(intro, 'multi_spline', 0)),
        )
        self._create_intro_data_metadata_empties(collection, empty, intro)
        self._store_multi_spline_json_property(empty, 'trlau_intro_multi_spline_json', getattr(intro, 'multi_spline_data', None))
        return empty

    @staticmethod
    def _set_custom_property(obj, key: str, value) -> None:
        try:
            if obj is not None and str(key).startswith(('trlau_sfx_', 'trlau_wave_')) and hasattr(bpy.types.Object, key):
                try:
                    if key in obj:
                        del obj[key]
                except Exception:
                    pass
                if key in SFX_PANEL_VECTOR_NAMES:
                    try:
                        seq = list(value or [])[:3]
                    except Exception:
                        seq = []
                    while len(seq) < 3:
                        seq.append(0.0)
                    setattr(obj, key, tuple(float(item) for item in seq[:3]))
                elif key in SFX_PANEL_STRING_NAMES or str(key).startswith('trlau_sfx_stream_filename_'):
                    if isinstance(value, (tuple, list)):
                        try:
                            text = json.dumps(list(value), separators=(',', ':'))
                        except Exception:
                            text = ','.join(str(item) for item in value)
                    else:
                        text = str(value if value is not None else '')
                    setattr(obj, key, text)
                elif key in SFX_PANEL_FLOAT_NAMES:
                    try:
                        setattr(obj, key, float(value if value is not None else 0.0))
                    except Exception:
                        setattr(obj, key, 0.0)
                else:
                    try:
                        setattr(obj, key, int(value if value is not None else 0))
                    except Exception:
                        setattr(obj, key, 0)
                return
            if obj is not None and str(key).startswith(('trlau_sfx_', 'trlau_wave_')):
                # Do not create visible ID custom properties for SFX/audio data.
                return
            if isinstance(value, bool):
                obj[key] = value
                return
            if isinstance(value, int):
                if -(2**31) <= value <= (2**31 - 1):
                    obj[key] = value
                else:
                    obj[key] = str(value)
                return
            if isinstance(value, float):
                obj[key] = float(value) if math.isfinite(float(value)) else 0.0
                return
            if isinstance(value, (tuple, list)):
                converted = []
                for item in value:
                    if isinstance(item, bool):
                        converted.append(item)
                    elif isinstance(item, int):
                        if -(2**31) <= item <= (2**31 - 1):
                            converted.append(item)
                        else:
                            converted.append(str(item))
                    elif isinstance(item, float):
                        converted.append(float(item) if math.isfinite(float(item)) else 0.0)
                    else:
                        converted.append(str(item))
                if all(isinstance(item, (int, float, bool, str)) for item in converted):
                    obj[key] = tuple(converted)
                else:
                    obj[key] = str(value)
                return
            obj[key] = value
        except Exception:
            try:
                obj[key] = str(value)
            except Exception:
                pass

    @staticmethod
    def _delete_custom_property(obj, key: str) -> None:
        try:
            if obj is not None and key in obj:
                del obj[key]
        except Exception:
            pass

    def _store_multi_spline_json_property(self, obj, key: str, multi_spline) -> None:
        # Legacy name retained for call-site compatibility.  MultiSpline is now
        # stored in bpy PropertyGroups so it can be edited from the level panel.
        try:
            trlau_load_multispline_property(obj, multi_spline)
        except Exception:
            pass
        self._delete_custom_property(obj, key)

    def _get_or_create_intro_metadata_child(self, collection, parent_empty, name: str, type_name: str):
        child_name = f'{getattr(parent_empty, "name", "IntroData")}_{name}'
        empty = bpy.data.objects.get(child_name)
        if empty is None:
            empty = bpy.data.objects.new(child_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 24.0
        empty.parent = parent_empty
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        self._set_custom_property(empty, 'trlau_level', True)
        self._set_custom_property(empty, 'trlau_type', type_name)
        return empty

    def _create_intro_data_metadata_empties(self, collection, intro_empty, intro) -> None:
        block = getattr(intro, 'intro_data_block', None)
        if block is None:
            return
        data_type = int(getattr(block, 'data_type', 0) or 0)

        generic = getattr(block, 'generic', None)
        if generic is not None:
            generic_empty = self._get_or_create_intro_metadata_child(collection, intro_empty, 'GenericIntroData', 'GenericIntroData')
            parent_data = trlau_intro_data_to_dict(intro_empty)
            trlau_set_intro_generic_property(
                generic_empty,
                parent_index=int(parent_data.get('index', -1)),
                values={attr: getattr(generic, attr, default) for attr, _label, default in INTRO_GENERIC_FIELDS},
            )

        specific_name = INTRO_SPECIFIC_TYPE_NAMES.get(data_type, f'IntroType{data_type}Data')
        specific_empty = self._get_or_create_intro_metadata_child(collection, intro_empty, specific_name, specific_name)
        specific = dict(getattr(block, 'specific', {}) or {})
        parent_data = trlau_intro_data_to_dict(intro_empty)
        trlau_set_intro_specific_property(
            specific_empty,
            parent_index=int(parent_data.get('index', -1)),
            data_type=data_type,
            struct_name=specific_name,
            values={attr: specific.get(attr, default) for attr, _label, default in INTRO_SPECIFIC_FIELDS.get(data_type, ())},
        )

    @staticmethod
    def _read_sfx_c_string(data: bytes, offset: int) -> str:
        try:
            offset = int(offset)
        except Exception:
            return ''
        if offset < 0 or offset >= len(data):
            return ''
        end = data.find(b'\x00', offset)
        if end < 0:
            end = len(data)
        return data[offset:end].decode('utf-8', errors='replace')

    @staticmethod
    def _safe_sfx_float(value, default: float = 0.0) -> float:
        try:
            value = float(value)
        except Exception:
            return float(default)
        return value if math.isfinite(value) else float(default)

    @staticmethod
    def _decode_adsr_fields(adsr0: int, adsr1: int) -> dict[str, int]:
        return {
            'armode': int(adsr0) & 0x1,
            'ar': (int(adsr0) >> 1) & 0x7F,
            'dr': (int(adsr0) >> 8) & 0x0F,
            'sl': (int(adsr0) >> 12) & 0x0F,
            'srmode': int(adsr1) & 0x1,
            'srsign': (int(adsr1) >> 1) & 0x1,
            'srpad': (int(adsr1) >> 2) & 0x1,
            'rr': (int(adsr1) >> 3) & 0x1F,
            'sr': (int(adsr1) >> 8) & 0x7F,
            'rrmode': (int(adsr1) >> 15) & 0x1,
        }

    @staticmethod
    def _parse_sfx_sound_section(raw_data: bytes, info) -> dict[str, object]:
        if info is None or not raw_data:
            return {}
        try:
            info_size = int(getattr(info, 'info_size', 24) or 24)
            total_size = min(len(raw_data), max(info_size, int(getattr(info, 'size', len(raw_data)) or len(raw_data))))
        except Exception:
            return {}
        payload = bytes(raw_data[info_size:total_size])
        if len(payload) < 4:
            return {}
        try:
            calltype = int(struct.unpack_from('<i', payload, 0)[0])
        except struct.error:
            return {}

        metadata: dict[str, object] = {'calltype': int(calltype)}
        if calltype == 0:
            if len(payload) < 4 + _SFX_TONE_SIZE:
                return metadata
            try:
                (
                    wave_id,
                    priority,
                    volume,
                    volume_variation,
                    mode,
                    sound_group,
                    delay,
                    max_voices,
                    cur_voices,
                    note,
                    adsr0,
                    adsr1,
                    pitch_variation,
                    initial_delay,
                    initial_delay_variation,
                    max_distance,
                    pan_position,
                    pad1,
                    pad2,
                    pad3,
                ) = struct.unpack_from(_SFX_TONE_STRUCT, payload, 4)
            except struct.error:
                return metadata
            metadata.update({
                'wave_id': int(wave_id),
                'priority': int(priority),
                'volume': int(volume),
                'volume_variation': int(volume_variation),
                'mode': int(mode),
                'sound_group': int(sound_group),
                'delay': int(delay),
                'max_voices': int(max_voices),
                'cur_voices': int(cur_voices),
                'note': LevelImporterMixin._safe_sfx_float(note),
                'pitch_variation': LevelImporterMixin._safe_sfx_float(pitch_variation),
                'initial_delay': LevelImporterMixin._safe_sfx_float(initial_delay),
                'initial_delay_variation': LevelImporterMixin._safe_sfx_float(initial_delay_variation),
                'max_distance': int(max_distance),
                'pan_position': int(pan_position),
                'pad1': int(pad1),
                'pad2': int(pad2),
                'pad3': int(pad3),
            })
            metadata.update(LevelImporterMixin._decode_adsr_fields(int(adsr0), int(adsr1)))
            return metadata

        if calltype == 1:
            if len(payload) < 4 + _SFX_STREAM_SIZE:
                return metadata
            try:
                (
                    name_offset,
                    priority,
                    volume,
                    volume_variation,
                    mode,
                    sound_group,
                    delay,
                    loops,
                    cur_voices,
                    note,
                    adsr0,
                    adsr1,
                    pitch_variation,
                    initial_delay,
                    initial_delay_variation,
                    max_distance,
                    subtitlemode,
                ) = struct.unpack_from(_SFX_STREAM_STRUCT, payload, 4)
            except struct.error:
                return metadata

            filename = LevelImporterMixin._read_sfx_c_string(payload, int(name_offset))
            metadata.update({
                'filename': filename,
                'name_offset': int(name_offset),
                'name_has_relocation': int(_SFX_NAME_POINTER_OFFSET in getattr(info, 'relocations_by_offset', {}) or any(int(getattr(r, 'offset', -1)) == _SFX_NAME_POINTER_OFFSET for r in getattr(info, 'relocations', []) or [])),
                'priority': int(priority),
                'volume': int(volume),
                'volume_variation': int(volume_variation),
                'mode': int(mode),
                'sound_group': int(sound_group),
                'delay': int(delay),
                'loops': int(loops),
                'cur_voices': int(cur_voices),
                'note': LevelImporterMixin._safe_sfx_float(note),
                'pitch_variation': LevelImporterMixin._safe_sfx_float(pitch_variation),
                'initial_delay': LevelImporterMixin._safe_sfx_float(initial_delay),
                'initial_delay_variation': LevelImporterMixin._safe_sfx_float(initial_delay_variation),
                'max_distance': int(max_distance),
                'subtitlemode': int(subtitlemode),
            })
            metadata.update(LevelImporterMixin._decode_adsr_fields(int(adsr0), int(adsr1)))
            return metadata

        if calltype == 2:
            if len(payload) >= 8:
                count = max(0, min(int(struct.unpack_from('<I', payload, 4)[0]), 1024))
                choices: list[dict[str, int]] = []
                for entry_index in range(count):
                    entry_offset = 8 + (entry_index * 8)
                    if entry_offset + 8 > len(payload):
                        break
                    choices.append({
                        'tone_id': int(struct.unpack_from('<I', payload, entry_offset)[0]),
                        'chance': int(struct.unpack_from('<I', payload, entry_offset + 4)[0]),
                    })
                metadata['choice_count'] = int(count)
                metadata['choices'] = choices
            return metadata

        if calltype == 3:
            if len(payload) >= 8:
                count = max(0, min(int(struct.unpack_from('<I', payload, 4)[0]), 1024))
                maps: list[dict[str, int]] = []
                for entry_index in range(count):
                    entry_offset = 8 + (entry_index * 8)
                    if entry_offset + 6 > len(payload):
                        break
                    maps.append({
                        'tone_id': int(struct.unpack_from('<I', payload, entry_offset)[0]),
                        'material': int(struct.unpack_from('<H', payload, entry_offset + 4)[0]),
                    })
                metadata['material_count'] = int(count)
                metadata['materials'] = maps
            return metadata

        if calltype == 4:
            if len(payload) >= 8:
                count = max(0, min(int(struct.unpack_from('<I', payload, 4)[0]), 1024))
                maps: list[dict[str, int]] = []
                for entry_index in range(count):
                    entry_offset = 8 + (entry_index * 8)
                    if entry_offset + 8 > len(payload):
                        break
                    maps.append({
                        'tone_id': int(struct.unpack_from('<I', payload, entry_offset)[0]),
                        'costume': int(struct.unpack_from('<i', payload, entry_offset + 4)[0]),
                    })
                metadata['costume_count'] = int(count)
                metadata['costumes'] = maps
            return metadata

        return metadata

    @staticmethod
    def _section_payload(raw_data: bytes, info) -> bytes:
        if info is None or not raw_data:
            return b''
        try:
            info_size = int(getattr(info, 'info_size', 24) or 24)
            total_size = min(len(raw_data), max(info_size, int(getattr(info, 'size', len(raw_data)) or len(raw_data))))
            return bytes(raw_data[info_size:total_size])
        except Exception:
            return b''

    @staticmethod
    def _adpcm_decode_sample(code: int, predicted: int, step: int) -> int:
        delta = int(step) >> 3
        if int(code) & 1:
            delta += int(step) >> 2
        if int(code) & 2:
            delta += int(step) >> 1
        if int(code) & 4:
            delta += int(step)
        if int(code) & 8:
            delta = -delta
        delta += int(predicted)
        return max(-32768, min(32767, int(delta)))

    @staticmethod
    def _adpcm_next_step_index(step_index: int, code: int) -> int:
        index_table = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)
        return max(0, min(88, int(step_index) + int(index_table[int(code) & 0x0F])))

    @staticmethod
    def _decode_cd_adpcm_block(block: bytes) -> bytes:
        if len(block) < 36:
            return b''
        step_table = (
            7, 8, 9, 10, 11, 12, 13, 14,
            16, 17, 19, 21, 23, 25, 28, 31,
            34, 37, 41, 45, 50, 55, 60, 66,
            73, 80, 88, 97, 107, 118, 130, 143,
            157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658,
            724, 796, 876, 963, 1060, 1166, 1282, 1411,
            1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
            3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484,
            7132, 7845, 8191, 8191, 8191, 8191, 8191, 8191,
            8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191,
            8191,
        )
        predicted = struct.unpack_from('<h', block, 0)[0]
        step_index = int(block[2])
        samples = [int(predicted)]
        code = int(block[4]) >> 4
        step = step_table[max(0, min(88, step_index))]
        predicted = LevelImporterMixin._adpcm_decode_sample(code, predicted, step)
        step_index = LevelImporterMixin._adpcm_next_step_index(step_index, code)
        samples.append(int(predicted))
        for byte_index in range(5, 36):
            packed = int(block[byte_index])
            for code in (packed & 0x0F, packed >> 4):
                step = step_table[max(0, min(88, step_index))]
                predicted = LevelImporterMixin._adpcm_decode_sample(code, predicted, step)
                step_index = LevelImporterMixin._adpcm_next_step_index(step_index, code)
                samples.append(int(predicted))
        return struct.pack('<' + ('h' * len(samples)), *samples[:64])

    @staticmethod
    def _decode_wave_section_to_pcm_wav(raw_data: bytes, info) -> tuple[bytes, dict[str, int]]:
        payload = LevelImporterMixin._section_payload(raw_data, info)
        if len(payload) < 12:
            return b'', {}
        sample_rate, loop_start, loop_end = struct.unpack_from('<III', payload, 0)
        adpcm_payload = payload[12:]
        pcm = bytearray()
        block_count = 0
        for offset in range(0, len(adpcm_payload) - 35, 36):
            pcm.extend(LevelImporterMixin._decode_cd_adpcm_block(adpcm_payload[offset:offset + 36]))
            block_count += 1
        if not pcm:
            return b'', {}
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(bytes(pcm))
        return wav_io.getvalue(), {
            'sample_rate': int(sample_rate),
            'loop_start': int(loop_start),
            'loop_end': int(loop_end),
            'adpcm_block_count': int(block_count),
            'sample_count': int(len(pcm) // 2),
        }

    def _audio_temp_dir(self) -> Path:
        temp_dir = getattr(self, '_trlau_audio_temp_dir', None)
        if temp_dir:
            try:
                path = Path(temp_dir)
                path.mkdir(parents=True, exist_ok=True)
                return path
            except Exception:
                pass
        path = Path(tempfile.mkdtemp(prefix='trlau_editor_audio_'))
        self._trlau_audio_temp_dir = str(path)
        return path

    def _load_wave_sound_for_section_id(self, source_dir: Path, wave_id: int):
        try:
            wave_id = int(wave_id)
        except Exception:
            return None, {}
        cache = getattr(self, '_trlau_wave_sound_cache', None)
        if cache is None:
            cache = {}
            self._trlau_wave_sound_cache = cache
        cache_key = (str(Path(source_dir)), int(wave_id))
        if cache_key in cache:
            return cache[cache_key]
        section_path = self._find_section_file_by_id(Path(source_dir), wave_id)
        if section_path is None:
            logger.warning('Could not find Wave section file with id %s in %s', wave_id, source_dir)
            cache[cache_key] = (None, {})
            return cache[cache_key]
        try:
            raw_data = section_path.read_bytes()
            info = self._read_section_info(section_path)
            if info is None or int(getattr(info, 'section_type', -1)) != 6:
                cache[cache_key] = (None, {})
                return cache[cache_key]
            wav_bytes, metadata = self._decode_wave_section_to_pcm_wav(raw_data, info)
            if not wav_bytes:
                cache[cache_key] = (None, {})
                return cache[cache_key]
            out_path = self._audio_temp_dir() / f'trlau_wave_{wave_id}.wav'
            out_path.write_bytes(wav_bytes)
            sound = bpy.data.sounds.load(str(out_path), check_existing=True)
            try:
                sound.name = f'TRLAU_Wave_{wave_id}'
            except Exception:
                pass
            try:
                sound.pack()
            except Exception:
                pass
            metadata = dict(metadata)
            metadata['wave_section_id'] = int(wave_id)
            metadata['wave_section_file'] = str(section_path.name)
            metadata['decoded_wav_path'] = str(out_path)
            cache[cache_key] = (sound, metadata)
        except Exception as exc:
            logger.warning('Failed to decode Wave section id %s from %s: %s', wave_id, section_path, exc)
            cache[cache_key] = (None, {})
        return cache[cache_key]

    def _parse_sfx_section_reference(self, source_dir: Path, sfx_id: int, *, _seen: set[int] | None = None) -> list[dict[str, object]]:
        try:
            sfx_id = int(sfx_id)
        except Exception:
            return []
        if sfx_id <= 0:
            return []
        seen = set(_seen or set())
        if sfx_id in seen:
            return []
        seen.add(sfx_id)
        section_path = self._find_section_file_by_id(Path(source_dir), sfx_id)
        if section_path is None:
            return [{'sfx_id': int(sfx_id), 'missing': True}]
        try:
            raw_data = section_path.read_bytes()
            info = self._read_section_info(section_path)
            payload = self._section_payload(raw_data, info)
            if len(payload) < 4:
                return [{'sfx_id': int(sfx_id), 'section_file': str(section_path.name), 'invalid': True}]
            calltype = int(struct.unpack_from('<i', payload, 0)[0])
            base = {
                'sfx_id': int(sfx_id),
                'section_file': str(section_path.name),
                'section_type': int(getattr(info, 'section_type', 0) if info is not None else 0),
                'calltype': int(calltype),
            }
            if calltype in {0, 1}:
                result = dict(base)
                result.update(self._parse_sfx_sound_section(raw_data, info))
                if calltype == 0 and 'wave_id' not in result and len(payload) >= 8:
                    result['wave_id'] = int(struct.unpack_from('<I', payload, 4)[0])
                return [result]
            if calltype == 2 and len(payload) >= 8:
                count = max(0, min(int(struct.unpack_from('<I', payload, 4)[0]), 1024))
                results: list[dict[str, object]] = []
                for entry_index in range(count):
                    entry_offset = 8 + (entry_index * 8)
                    if entry_offset + 8 > len(payload):
                        break
                    child_id = int(struct.unpack_from('<I', payload, entry_offset)[0])
                    weight = int(struct.unpack_from('<I', payload, entry_offset + 4)[0])
                    for child in self._parse_sfx_section_reference(source_dir, child_id, _seen=set(seen)):
                        child = dict(child)
                        child['group_sfx_id'] = int(sfx_id)
                        child['group_entry_index'] = int(entry_index)
                        child['group_weight'] = int(weight)
                        results.append(child)
                if results:
                    return results
                result = dict(base)
                result['group_count'] = int(count)
                return [result]
            return [base]
        except Exception as exc:
            logger.warning('Failed to parse SFX section id %s in %s: %s', sfx_id, source_dir, exc)
            return [{'sfx_id': int(sfx_id), 'error': str(exc)}]

    def _set_sfx_properties(self, obj, prefix: str, values: dict[str, object]) -> None:
        for key, value in dict(values or {}).items():
            prop_name = f'{prefix}{key}'
            if isinstance(value, (str, int, float, bool)):
                self._set_custom_property(obj, prop_name, value)
            elif isinstance(value, (tuple, list)) and all(isinstance(item, (str, int, float, bool)) for item in value):
                self._set_custom_property(obj, prop_name, tuple(value))
            elif value is not None:
                try:
                    self._set_custom_property(obj, prop_name, json.dumps(value, separators=(',', ':')))
                except Exception:
                    self._set_custom_property(obj, prop_name, str(value))

    def _configure_speaker_from_sfx(self, speaker_data, sound_entry, resolved: dict[str, object]) -> None:
        props = dict(getattr(sound_entry, 'properties', {}) or {})
        sfx_volume = resolved.get('volume', None)
        max_volume = props.get('max_volume', None)
        volume_value = sfx_volume if sfx_volume is not None else max_volume
        try:
            if volume_value is not None and hasattr(speaker_data, 'volume'):
                speaker_data.volume = max(0.0, min(2.0, float(volume_value) / 127.0))
        except Exception:
            pass
        try:
            pitch = self._safe_sfx_float(props.get('pitch', 0.0) or 0.0)
            note = self._safe_sfx_float(resolved.get('note', 0.0) or 0.0)
            if hasattr(speaker_data, 'pitch'):
                speaker_data.pitch = max(0.01, min(16.0, math.pow(2.0, (pitch + note) / 12.0)))
        except Exception:
            pass
        try:
            distance_max = int(resolved.get('max_distance', 0) or 0)
            min_dist = int(props.get('min_vol_distance', 0) or 0)
            if hasattr(speaker_data, 'distance_max') and max(distance_max, min_dist) > 0:
                speaker_data.distance_max = float(max(distance_max, min_dist))
            if hasattr(speaker_data, 'distance_reference') and min_dist > 0:
                speaker_data.distance_reference = float(min_dist)
        except Exception:
            pass

    def _import_level_sfx_markers(self, collection, source_dir: Path, level) -> list[object]:
        if not getattr(self, 'import_audio', True):
            return []
        if str(getattr(self, 'platform', 'PC')).upper() != 'PC':
            return []
        markers = list(getattr(level, 'sfx_markers', []) or [])
        if not markers:
            return []
        print('SFXMarkers found, importing audio markers...')
        root_obj = self._get_level_root_empty(collection)
        drm_name = str(root_obj.get('trlau_drm_name', getattr(collection, 'name', '') or 'Level') or 'Level')
        parent = self._get_or_create_component_empty(collection, 'SFXMarker', display_size=48.0)
        parent.trlau_sfx_role = 'SFXRoot'
        self._set_custom_property(parent, 'trlau_sfx_marker_count', len(markers))
        imported_audio_sections: set[tuple[str, int]] = set()
        created: list[object] = []
        for marker in markers:
            marker_name = self._prefixed_name(drm_name, f'SFXMarker_{int(getattr(marker, "index", 0)):03d}_{int(getattr(marker, "unique_id", 0))}')
            marker_obj = bpy.data.objects.get(marker_name)
            if marker_obj is None:
                marker_obj = bpy.data.objects.new(marker_name, None)
            self._link_object_to_collection(marker_obj, collection)
            marker_obj.empty_display_type = 'SPHERE'
            marker_obj.empty_display_size = 36.0
            marker_obj.location = tuple(float(v) for v in getattr(marker, 'position', (0.0, 0.0, 0.0)))
            marker_obj.rotation_mode = 'XYZ'
            marker_obj.rotation_euler = (0.0, 0.0, 0.0)
            marker_obj.scale = (1.0, 1.0, 1.0)
            marker_obj.parent = parent
            marker_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            marker_obj.trlau_sfx_role = 'SFXMarker'
            for _key in ('trlau_level', 'trlau_type'):
                try:
                    if _key in marker_obj:
                        del marker_obj[_key]
                except Exception:
                    pass
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_index', int(getattr(marker, 'index', -1)))
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_unique_id', int(getattr(marker, 'unique_id', 0)))
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_plane', int(getattr(marker, 'plane', 0)))
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_spline_id', int(getattr(marker, 'spline_id', 0)))
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_sound_instance_offset', int(getattr(marker, 'sound_instance_offset', 0)))
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_source_offset', int(getattr(marker, 'source_offset', 0)))
            self._set_custom_property(marker_obj, 'trlau_sfx_marker_raw_position', tuple(float(v) for v in getattr(marker, 'raw_position', (0.0, 0.0, 0.0))))
            created.append(marker_obj)

            sound_entries = []
            for kind_attr in ('periodic_sounds', 'event_sounds', 'one_shot_sounds', 'stream_sounds'):
                sound_entries.extend(list(getattr(marker, kind_attr, []) or []))
            for sound_entry in sound_entries:
                kind = str(getattr(sound_entry, 'kind', 'sound') or 'sound')
                entry_index = int(getattr(sound_entry, 'index', 0) or 0)
                entry_name = self._prefixed_name(drm_name, f'SFXMarker_{int(getattr(marker, "index", 0)):03d}_{kind}_{entry_index:02d}')
                entry_obj = bpy.data.objects.get(entry_name)
                if entry_obj is None:
                    entry_obj = bpy.data.objects.new(entry_name, None)
                self._link_object_to_collection(entry_obj, collection)
                entry_obj.empty_display_type = 'PLAIN_AXES'
                entry_obj.empty_display_size = 18.0
                entry_obj.parent = marker_obj
                entry_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                entry_obj.location = (0.0, 0.0, 0.0)
                entry_obj.rotation_mode = 'XYZ'
                entry_obj.rotation_euler = (0.0, 0.0, 0.0)
                entry_obj.scale = (1.0, 1.0, 1.0)
                entry_obj.trlau_sfx_role = 'SFXSound'
                for _key in ('trlau_level', 'trlau_type'):
                    try:
                        if _key in entry_obj:
                            del entry_obj[_key]
                    except Exception:
                        pass
                self._set_custom_property(entry_obj, 'trlau_sfx_kind', kind)
                self._set_custom_property(entry_obj, 'trlau_sfx_entry_index', entry_index)
                self._set_custom_property(entry_obj, 'trlau_sfx_source_offset', int(getattr(sound_entry, 'source_offset', 0) or 0))
                self._set_custom_property(entry_obj, 'trlau_sfx_ids', tuple(int(v) for v in list(getattr(sound_entry, 'sound_ids', []) or [])))
                self._set_sfx_properties(entry_obj, 'trlau_sfx_', dict(getattr(sound_entry, 'properties', {}) or {}))
                created.append(entry_obj)

                resolved_refs: list[dict[str, object]] = []
                for sfx_id in list(getattr(sound_entry, 'sound_ids', []) or []):
                    resolved_refs.extend(self._parse_sfx_section_reference(Path(source_dir), int(sfx_id)))
                if resolved_refs:
                    self._set_custom_property(entry_obj, 'trlau_sfx_resolved_json', json.dumps(resolved_refs, separators=(',', ':')))

                sfx_section_ids: set[int] = set()
                for value in list(getattr(sound_entry, 'sound_ids', []) or []):
                    try:
                        sfx_value = int(value)
                    except Exception:
                        continue
                    if sfx_value > 0:
                        sfx_section_ids.add(sfx_value)
                wave_section_ids: set[int] = set()
                for resolved in resolved_refs:
                    try:
                        sfx_id = int(resolved.get('sfx_id', 0) or 0)
                    except Exception:
                        sfx_id = 0
                    if sfx_id > 0:
                        sfx_section_ids.add(sfx_id)
                    try:
                        group_sfx_id = int(resolved.get('group_sfx_id', 0) or 0)
                    except Exception:
                        group_sfx_id = 0
                    if group_sfx_id > 0:
                        sfx_section_ids.add(group_sfx_id)
                    try:
                        wave_id = int(resolved.get('wave_id', 0) or 0)
                    except Exception:
                        wave_id = 0
                    if wave_id > 0:
                        wave_section_ids.add(wave_id)

                for sfx_section_id in sorted(sfx_section_ids):
                    key = ('SFX', int(sfx_section_id))
                    if key in imported_audio_sections:
                        continue
                    section_path = self._find_section_file_by_id(Path(source_dir), int(sfx_section_id))
                    if section_path is not None:
                        self._import_section_metadata_empty(
                            collection,
                            parent,
                            self._prefixed_name(drm_name, f'SFXSection_{int(sfx_section_id):04d}'),
                            'SFX',
                            int(sfx_section_id),
                            section_path,
                            type_name='SFXSection',
                        )
                        imported_audio_sections.add(key)

                for wave_section_id in sorted(wave_section_ids):
                    key = ('Wave', int(wave_section_id))
                    if key in imported_audio_sections:
                        continue
                    section_path = self._find_section_file_by_id(Path(source_dir), int(wave_section_id))
                    if section_path is not None:
                        self._import_section_metadata_empty(
                            collection,
                            parent,
                            self._prefixed_name(drm_name, f'WaveSection_{int(wave_section_id):04d}'),
                            'Wave',
                            int(wave_section_id),
                            section_path,
                            type_name='WaveSection',
                        )
                        imported_audio_sections.add(key)

                speaker_count = 0
                for resolved in resolved_refs:
                    wave_id = int(resolved.get('wave_id', 0) or 0)
                    if wave_id <= 0:
                        if resolved.get('filename'):
                            self._set_custom_property(entry_obj, f'trlau_sfx_stream_filename_{speaker_count:02d}', str(resolved.get('filename', '')))
                        continue
                    sound, wave_meta = self._load_wave_sound_for_section_id(Path(source_dir), wave_id)
                    if sound is None:
                        continue
                    speaker_name = self._prefixed_name(drm_name, f'SFXMarker_{int(getattr(marker, "index", 0)):03d}_{kind}_{entry_index:02d}_Wave_{wave_id}')
                    speaker_data = bpy.data.speakers.new(speaker_name)
                    speaker_obj = bpy.data.objects.new(speaker_name, speaker_data)
                    speaker_data.sound = sound
                    self._configure_speaker_from_sfx(speaker_data, sound_entry, resolved)
                    self._link_object_to_collection(speaker_obj, collection)
                    speaker_obj.parent = entry_obj
                    speaker_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                    speaker_obj.location = (0.0, 0.0, 0.0)
                    speaker_obj.rotation_mode = 'XYZ'
                    speaker_obj.rotation_euler = (0.0, 0.0, 0.0)
                    speaker_obj.scale = (1.0, 1.0, 1.0)
                    speaker_obj.trlau_sfx_role = 'SFXSpeaker'
                    for _key in ('trlau_level', 'trlau_type'):
                        try:
                            if _key in speaker_obj:
                                del speaker_obj[_key]
                        except Exception:
                            pass
                    self._set_custom_property(speaker_obj, 'trlau_sfx_wave_id', wave_id)
                    self._set_custom_property(speaker_obj, 'trlau_sfx_id', int(resolved.get('sfx_id', 0) or 0))
                    self._set_sfx_properties(speaker_obj, 'trlau_sfx_ref_', resolved)
                    self._set_sfx_properties(speaker_obj, 'trlau_wave_', wave_meta)
                    created.append(speaker_obj)
                    speaker_count += 1
                self._set_custom_property(entry_obj, 'trlau_sfx_speaker_count', speaker_count)
        return created


    def _import_intro_sound_metadata(self, collection, source_dir: Path, intro_empty, intro) -> None:
        block = getattr(intro, 'intro_data_block', None)
        if block is None or int(getattr(block, 'data_type', 0) or 0) != 17:
            return
        specific = dict(getattr(block, 'specific', {}) or {})
        try:
            sound_id = int(specific.get('sound_id', 0) or 0)
        except Exception:
            sound_id = 0
        if sound_id <= 0:
            return

        sound_path = self._find_section_file_by_id(Path(source_dir), sound_id)
        if sound_path is None:
            logger.warning('Could not find Sound section file with id %s for IntroData %s in %s', sound_id, getattr(intro_empty, 'name', '<unknown>'), source_dir)
            return

        empty_name = f'{getattr(intro_empty, "name", "IntroData")}_Sound'
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 32.0
        empty.parent = intro_empty
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)

        # Internal export payload holder only. Do not mark it as
        # trlau_section_metadata_empty, otherwise it appears in the editable
        # metadata panel like WaterFSFX/basecam.
        self._set_custom_property(empty, 'trlau_level', True)
        self._set_custom_property(empty, 'trlau_type', 'IntroSound')
        parent_data = trlau_intro_data_to_dict(intro_empty)
        trlau_set_intro_sound_property(
            empty,
            parent_index=int(parent_data.get('index', -1)),
            file_name=str(sound_path.name),
        )

        try:
            raw_data = sound_path.read_bytes()
        except Exception as exc:
            logger.warning('Failed to read Sound section file %s: %s', sound_path.name, exc)
            return

        info = self._read_section_info(sound_path)
        if info is not None:
            trlau_set_intro_sound_property(
                empty,
                section_type=int(getattr(info, 'section_type', 0)),
                skip_flags=int(getattr(info, 'skip_flags', 0)),
                version_id=int(getattr(info, 'version_id', 0)),
                has_debug_info=int(getattr(info, 'has_debug_info', 0)),
                resource_type=int(getattr(info, 'resource_type', 0)),
                spec_mask=int(getattr(info, 'spec_mask', 0)),
                num_relocations=int(getattr(info, 'num_relocations', 0)),
                standalone_size=int(getattr(info, 'size', len(raw_data))),
            )

        self._clear_section_blob_metadata(empty)
        self._store_section_blob_metadata(empty, raw_data)

        sfx_metadata = self._parse_sfx_sound_section(raw_data, info)
        if sfx_metadata:
            self._set_custom_property(empty, 'trlau_sfx_calltype', sfx_metadata.get('calltype', 0))
            self._set_custom_property(empty, 'trlau_sfx_filename', sfx_metadata.get('filename', ''))
            self._set_custom_property(empty, 'trlau_sfx_name_offset', max(4 + _SFX_STREAM_SIZE, int(sfx_metadata.get('name_offset', _SFX_DEFAULT_NAME_OFFSET) or _SFX_DEFAULT_NAME_OFFSET)))
            self._set_custom_property(empty, 'trlau_sfx_name_has_relocation', int(sfx_metadata.get('name_has_relocation', 0) or 0))
            for field_name in (
                'priority', 'volume', 'volume_variation', 'mode', 'sound_group', 'delay', 'loops', 'cur_voices',
                'note', 'armode', 'ar', 'dr', 'sl', 'srmode', 'srsign', 'srpad', 'rr', 'sr', 'rrmode',
                'pitch_variation', 'initial_delay', 'initial_delay_variation', 'max_distance', 'subtitlemode',
            ):
                self._set_custom_property(empty, f'trlau_sfx_{field_name}', sfx_metadata.get(field_name, 0))

    @staticmethod
    def _clear_section_blob_metadata(obj) -> None:
        for key in list(obj.keys()):
            if str(key).startswith(SECTION_BLOB_PROP_PREFIX):
                try:
                    del obj[key]
                except Exception:
                    pass

    @staticmethod
    def _store_section_blob_metadata(obj, data: bytes) -> None:
        encoded = base64.b64encode(bytes(data)).decode('ascii') if data else ''
        chunk_count = 0
        if encoded:
            for chunk_index, start in enumerate(range(0, len(encoded), SECTION_BLOB_CHUNK_SIZE)):
                obj[f'{SECTION_BLOB_PROP_PREFIX}{chunk_index:04d}'] = encoded[start:start + SECTION_BLOB_CHUNK_SIZE]
                chunk_count += 1
        obj['trlau_section_blob_encoding'] = 'base64'
        obj['trlau_section_blob_chunk_count'] = int(chunk_count)
        obj['trlau_section_blob_size'] = int(len(data))

    @staticmethod
    def _clear_terrain_light_grid_metadata(obj) -> None:
        for key in list(obj.keys()):
            if str(key).startswith(TERRAIN_LIGHT_GRID_BLOB_PROP_PREFIX):
                try:
                    del obj[key]
                except Exception:
                    pass
        for key in ('trlau_terrain_light_grid_blob_encoding', 'trlau_terrain_light_grid_blob_chunk_count', 'trlau_terrain_light_grid_blob_size'):
            try:
                if key in obj:
                    del obj[key]
            except Exception:
                pass

    @staticmethod
    def _encode_terrain_light_grid_cells(cells) -> bytes:
        normalized = []
        for cell in list(cells or []):
            try:
                cell_index = int(cell[0])
                values = tuple(int(v) & 0xFFFF for v in (cell[1] or ()))
            except Exception:
                continue
            if 0 <= cell_index < 1024 and values:
                normalized.append((cell_index, values[:256]))
        normalized.sort(key=lambda item: item[0])
        blob = bytearray(b'TLG1')
        blob.extend(struct.pack('<I', len(normalized)))
        for cell_index, values in normalized:
            blob.extend(struct.pack('<HH', int(cell_index), len(values)))
            for value in values:
                blob.extend(struct.pack('<H', int(value) & 0xFFFF))
        return bytes(blob)

    @staticmethod
    def _store_terrain_light_grid_metadata(obj, cells) -> None:
        LevelImporterMixin._clear_terrain_light_grid_metadata(obj)
        blob = LevelImporterMixin._encode_terrain_light_grid_cells(cells)
        encoded = base64.b64encode(blob).decode('ascii') if blob else ''
        chunk_count = 0
        if encoded:
            for chunk_index, start in enumerate(range(0, len(encoded), SECTION_BLOB_CHUNK_SIZE)):
                obj[f'{TERRAIN_LIGHT_GRID_BLOB_PROP_PREFIX}{chunk_index:04d}'] = encoded[start:start + SECTION_BLOB_CHUNK_SIZE]
                chunk_count += 1
        obj['trlau_terrain_light_grid_blob_encoding'] = 'base64'
        obj['trlau_terrain_light_grid_blob_chunk_count'] = int(chunk_count)
        obj['trlau_terrain_light_grid_blob_size'] = int(len(blob))

    @staticmethod
    def _read_section_info(filepath: Path):
        try:
            with open(filepath, 'rb') as fh:
                br = BinaryReader(fh, endian='<')
                return TRSectionParser.parse(br)
        except Exception:
            return None

    def _find_section_file_by_id(self, directory: Path, section_id: int) -> Path | None:
        target_id = int(section_id or 0)
        if target_id <= 0 or not directory.exists() or not directory.is_dir():
            return None
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file() or candidate.suffix.lower() in {'.drm', '.txt', '.json', '.py', '.md'}:
                continue
            info = self._read_section_info(candidate)
            if info is not None and int(getattr(info, 'section_id', 0)) == target_id:
                return candidate
        return None

    @staticmethod
    def _section_index_from_filename(path: Path) -> int:
        try:
            return int(str(path.stem).split('_', 1)[0])
        except Exception:
            return -1

    def _find_section_file_by_index(self, directory: Path, section_index: int) -> Path | None:
        try:
            target_index = int(section_index)
        except Exception:
            target_index = -1
        if target_index < 0 or not directory.exists() or not directory.is_dir():
            return None
        for candidate in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if not candidate.is_file() or candidate.suffix.lower() in {'.drm', '.txt', '.json', '.py', '.md'}:
                continue
            if self._section_index_from_filename(candidate) == target_index:
                return candidate
        return None

    @staticmethod
    def _parse_cine_section_metadata(raw_data: bytes, info=None) -> dict[str, object]:
        payload_offset = 0
        payload_size = len(raw_data)
        try:
            if info is not None:
                payload_offset = int(getattr(info, 'data_start', 0) or 0)
                if payload_offset <= 0 or payload_offset > len(raw_data):
                    payload_offset = 0x18 if raw_data[:4] == b'SECT' and len(raw_data) >= 0x18 else 0
                standalone_size = int(getattr(info, 'size', len(raw_data)) or len(raw_data))
                if standalone_size > 0:
                    payload_size = max(0, min(len(raw_data) - payload_offset, standalone_size))
        except Exception:
            payload_offset = 0x18 if raw_data[:4] == b'SECT' and len(raw_data) >= 0x18 else 0
            payload_size = max(0, len(raw_data) - payload_offset)
        if raw_data[:4] == b'SECT' and payload_offset == 0:
            payload_offset = 0x18
            payload_size = max(0, len(raw_data) - payload_offset)
        payload = raw_data[payload_offset:payload_offset + payload_size]
        version_word = 0
        if len(payload) >= 4:
            try:
                version_word = int(struct.unpack_from('<I', payload, 0)[0])
            except Exception:
                version_word = 0

        meta: dict[str, object] = {
            'name': '',
            'payload_offset': int(payload_offset),
            'payload_size': int(len(payload)),
            'version_word': int(version_word),
            'structured': False,
        }
        try:
            parsed = parse_cine_section(raw_data)
            command = first_cinematic_command(parsed)
            meta.update({
                'structured': True,
                'name': str(parsed.get('name', '') or ''),
                'version_major': int(parsed.get('version_major', 0) or 0),
                'version_minor': int(parsed.get('version_minor', 0) or 0),
                'stream_unit_id': int(parsed.get('stream_unit_id', 0) or 0),
                'cine_id': int(parsed.get('cine_id', 0) or 0),
                'end_time': float(parsed.get('end_time', 0.0) or 0.0),
                'spline_count': int(parsed.get('spline_count', 0) or 0),
                'command_count': int(parsed.get('command_count', 0) or 0),
                'command_types': str(parsed.get('command_types', '') or ''),
                'rebuildable': int(1 if parsed.get('rebuildable') else 0),
                'parsed_size': int(parsed.get('parsed_size', 0) or 0),
            })
            if command:
                meta.update({
                    'command_type': 'Cinematic',
                    'command_unit_id': int(command.get('unit_id', 0) or 0),
                    'command_load': int(command.get('load', 0) or 0),
                    'command_camera_control': int(command.get('camera_control', 0) or 0),
                    'command_channels': int(command.get('channels', 0) or 0),
                    'command_positions_after_playback': int(command.get('positions_after_playback', 0) or 0),
                    'command_end_trigger_id': int(command.get('end_trigger_id', 0) or 0),
                    'command_data_pointer': int(command.get('data_pointer', 0) or 0),
                    'command_data_size': int(command.get('data_size', 0) or 0),
                    'command_cinematic_name': str(command.get('cinematic_name', '') or ''),
                })
        except Exception:
            # Unknown Cine command/spline class.  Leave raw section passthrough
            # intact instead of guessing from arbitrary strings.
            try:
                meta['name'] = TRLevelParser._parse_cine_payload_name(payload, 0)
            except Exception:
                meta['name'] = ''
        return meta

    @staticmethod
    def _store_cine_metadata_properties(obj, cine_meta: dict[str, object]) -> None:
        if obj is None or not isinstance(cine_meta, dict):
            return
        setters = getattr(LevelImporterMixin, '_set_custom_property', None)
        def setp(key, value):
            try:
                LevelImporterMixin._set_custom_property(obj, key, value)
            except Exception:
                try:
                    obj[key] = value
                except Exception:
                    pass
        setp('trlau_cine_structured', int(1 if cine_meta.get('structured') else 0))
        LevelImporterMixin._delete_custom_property(obj, 'trlau_cine_format')
        LevelImporterMixin._delete_custom_property(obj, 'trlau_unitdata_cine_format')
        for key, prop, cast in (
            ('version_major', 'trlau_cine_version_major', int),
            ('version_minor', 'trlau_cine_version_minor', int),
            ('stream_unit_id', 'trlau_cine_stream_unit_id', int),
            ('cine_id', 'trlau_cine_id', int),
            ('end_time', 'trlau_cine_end_time', float),
            ('spline_count', 'trlau_cine_spline_count', int),
            ('parsed_size', 'trlau_cine_parsed_size', int),
            ('command_count', 'trlau_cine_command_count', int),
            ('command_types', 'trlau_cine_command_types', str),
            ('rebuildable', 'trlau_cine_rebuildable', int),
            ('command_type', 'trlau_cine_command_type', str),
            ('command_unit_id', 'trlau_cine_command_unit_id', int),
            ('command_load', 'trlau_cine_command_load', int),
            ('command_camera_control', 'trlau_cine_command_camera_control', int),
            ('command_channels', 'trlau_cine_command_channels', int),
            ('command_positions_after_playback', 'trlau_cine_command_positions_after_playback', int),
            ('command_end_trigger_id', 'trlau_cine_command_end_trigger_id', int),
            ('command_data_pointer', 'trlau_cine_command_data_pointer', int),
            ('command_data_size', 'trlau_cine_command_data_size', int),
            ('command_cinematic_name', 'trlau_cine_command_cinematic_name', str),
        ):
            if key in cine_meta:
                try:
                    setp(prop, cast(cine_meta.get(key)))
                except Exception:
                    pass

    @staticmethod
    def _store_cine_entry_metadata_properties(obj, entry: dict[str, object]) -> None:
        if obj is None or not isinstance(entry, dict):
            return
        cine_meta: dict[str, object] = {
            'structured': bool(entry.get('cine_structured')),
            'name': str(entry.get('name', '') or ''),
            'version_major': int(entry.get('cine_version_major', 0) or 0),
            'version_minor': int(entry.get('cine_version_minor', 0) or 0),
            'stream_unit_id': int(entry.get('cine_stream_unit_id', 0) or 0),
            'cine_id': int(entry.get('cine_id', 0) or 0),
            'end_time': float(entry.get('cine_end_time', 0.0) or 0.0),
            'spline_count': int(entry.get('cine_spline_count', 0) or 0),
            'parsed_size': int(entry.get('cine_parsed_size', 0) or 0),
            'command_count': int(entry.get('cine_command_count', 0) or 0),
            'command_types': str(entry.get('cine_command_types', '') or ''),
            'rebuildable': int(entry.get('cine_rebuildable', 0) or 0),
        }
        if entry.get('cine_command_type'):
            cine_meta.update({
                'command_type': str(entry.get('cine_command_type', '') or ''),
                'command_unit_id': int(entry.get('cine_command_unit_id', 0) or 0),
                'command_load': int(entry.get('cine_command_load', 0) or 0),
                'command_camera_control': int(entry.get('cine_command_camera_control', 0) or 0),
                'command_channels': int(entry.get('cine_command_channels', 0) or 0),
                'command_positions_after_playback': int(entry.get('cine_command_positions_after_playback', 0) or 0),
                'command_end_trigger_id': int(entry.get('cine_command_end_trigger_id', 0) or 0),
                'command_data_pointer': int(entry.get('cine_command_data_pointer', 0) or 0),
                'command_data_size': int(entry.get('cine_command_data_size', 0) or 0),
                'command_cinematic_name': str(entry.get('cine_command_cinematic_name', '') or ''),
            })
        LevelImporterMixin._set_custom_property(obj, 'trlau_cine_name', str(cine_meta.get('name', '') or ''))
        LevelImporterMixin._set_custom_property(obj, 'trlau_cine_payload_offset', int(entry.get('target_offset', 0) or 0))
        LevelImporterMixin._set_custom_property(obj, 'trlau_cine_payload_size', int(entry.get('cine_parsed_size', 0) or 0))
        version_word = (int(cine_meta.get('version_major', 0) or 0) << 16) | (int(cine_meta.get('version_minor', 0) or 0) & 0xFFFF)
        LevelImporterMixin._set_custom_property(obj, 'trlau_cine_version_word', int(version_word))
        LevelImporterMixin._store_cine_metadata_properties(obj, cine_meta)

    def _import_section_metadata_empty(self, collection, parent_empty, empty_name: str, role: str, section_id: int, section_path: Path, *, type_name: str = 'SectionMetadata'):
        if parent_empty is None or section_path is None or not section_path.exists():
            return None
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 40.0
        empty.parent = parent_empty
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        self._set_custom_property(empty, 'trlau_level', True)
        self._set_custom_property(empty, 'trlau_type', type_name)
        self._set_custom_property(empty, 'trlau_section_metadata_empty', True)
        self._set_custom_property(empty, 'trlau_section_role', str(role))
        self._set_custom_property(empty, 'trlau_section_file_name', str(section_path.name))
        self._set_custom_property(empty, 'trlau_section_source_path', str(section_path))
        self._set_custom_property(empty, 'trlau_section_original_id', int(section_id))

        try:
            raw_data = section_path.read_bytes()
        except Exception as exc:
            logger.warning('Failed to read section metadata file %s: %s', section_path.name, exc)
            return empty

        info = self._read_section_info(section_path)
        if info is not None:
            self._set_custom_property(empty, 'trlau_section_type', int(getattr(info, 'section_type', 0)))
            self._set_custom_property(empty, 'trlau_section_skip_flags', int(getattr(info, 'skip_flags', 0)))
            self._set_custom_property(empty, 'trlau_section_version_id', int(getattr(info, 'version_id', 0)))
            self._set_custom_property(empty, 'trlau_section_has_debug_info', int(getattr(info, 'has_debug_info', 0)))
            self._set_custom_property(empty, 'trlau_section_resource_type', int(getattr(info, 'resource_type', 0)))
            self._set_custom_property(empty, 'trlau_section_spec_mask', int(getattr(info, 'spec_mask', 0)))
            self._set_custom_property(empty, 'trlau_section_num_relocations', int(getattr(info, 'num_relocations', 0)))
            self._set_custom_property(empty, 'trlau_section_standalone_size', int(getattr(info, 'size', len(raw_data))))

        self._clear_section_blob_metadata(empty)
        self._store_section_blob_metadata(empty, raw_data)
        if str(role) in {'FSFXLink', 'WaterFSFX'}:
            try:
                store_fsfx_render_effect_properties(empty, raw_data)
            except Exception as exc:
                logger.warning('Failed to parse %s RenderFSEffect metadata from %s: %s', role, section_path.name, exc)
        if str(role) == 'Cine':
            try:
                cine_meta = self._parse_cine_section_metadata(raw_data, info)
                self._set_custom_property(empty, 'trlau_cine_name', str(cine_meta.get('name', '') or ''))
                self._set_custom_property(empty, 'trlau_cine_payload_offset', int(cine_meta.get('payload_offset', 0) or 0))
                self._set_custom_property(empty, 'trlau_cine_payload_size', int(cine_meta.get('payload_size', 0) or 0))
                self._set_custom_property(empty, 'trlau_cine_version_word', int(cine_meta.get('version_word', 0) or 0))
                self._set_custom_property(empty, 'trlau_cine_section_index', int(self._section_index_from_filename(section_path)))
                self._store_cine_metadata_properties(empty, cine_meta)
            except Exception as exc:
                logger.warning('Failed to parse Cine metadata from %s: %s', section_path.name, exc)
        return empty

    def _build_reloc_module_metadata(self, collection, root_obj, level) -> None:
        reloc_module = getattr(level, 'reloc_module', None)
        if reloc_module is None:
            return
        try:
            blob = bytes(getattr(reloc_module, 'section_blob', b'') or b'')
        except Exception:
            blob = b''
        if not blob:
            return
        drm_name = str(root_obj.get('trlau_drm_name', getattr(collection, 'name', '') or 'Level') or 'Level')
        empty_name = self._prefixed_name(drm_name, 'RelocModule')
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 48.0
        empty.parent = root_obj
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)

        self._set_custom_property(empty, 'trlau_level', True)
        self._set_custom_property(empty, 'trlau_type', 'RelocModule')
        self._set_custom_property(empty, 'trlau_reloc_module_empty', True)
        self._set_custom_property(empty, 'trlau_reloc_module_target_offset', int(getattr(reloc_module, 'target_offset', 0) or 0))
        self._set_custom_property(empty, 'trlau_reloc_module_section_index', int(getattr(reloc_module, 'section_index', -1) or -1))
        self._set_custom_property(empty, 'trlau_reloc_module_file_name', str(getattr(reloc_module, 'file_name', '') or ''))
        self._set_custom_property(empty, 'trlau_section_metadata_empty', True)
        self._set_custom_property(empty, 'trlau_section_role', 'RelocModule')
        self._set_custom_property(empty, 'trlau_section_file_name', str(getattr(reloc_module, 'file_name', '') or empty_name))
        self._set_custom_property(empty, 'trlau_section_source_path', str(getattr(reloc_module, 'file_name', '') or ''))
        self._set_custom_property(empty, 'trlau_section_original_id', int(getattr(reloc_module, 'section_id', 0) or 0))
        self._set_custom_property(empty, 'trlau_section_type', int(getattr(reloc_module, 'section_type', 0) or 0))
        self._set_custom_property(empty, 'trlau_section_skip_flags', int(getattr(reloc_module, 'skip_flags', 0) or 0))
        self._set_custom_property(empty, 'trlau_section_version_id', int(getattr(reloc_module, 'version_id', 0) or 0))
        self._set_custom_property(empty, 'trlau_section_has_debug_info', int(getattr(reloc_module, 'has_debug_info', 0) or 0))
        self._set_custom_property(empty, 'trlau_section_resource_type', int(getattr(reloc_module, 'resource_type', 0) or 0))
        self._set_custom_property(empty, 'trlau_section_spec_mask', int(getattr(reloc_module, 'spec_mask', 0xFFFFFFFF) or 0xFFFFFFFF))

        self._clear_section_blob_metadata(empty)
        self._store_section_blob_metadata(empty, blob)

    def _build_section_reference_metadata(self, collection, source_dir: Path, level) -> None:
        root_obj = self._get_level_root_empty(collection)
        drm_name = str(root_obj.get('trlau_drm_name', getattr(collection, 'name', '') or 'Level') or 'Level')

        level_meta = dict(getattr(level, 'level_metadata', {}) or {})
        water_fsfx_id = int(level_meta.get('WaterFSFX', 0) or 0)
        if water_fsfx_id > 0:
            water_path = self._find_section_file_by_id(source_dir, water_fsfx_id)
            if water_path is not None:
                self._import_section_metadata_empty(
                    collection,
                    root_obj,
                    self._prefixed_name(drm_name, 'WaterFSFX'),
                    'WaterFSFX',
                    water_fsfx_id,
                    water_path,
                    type_name='WaterFSFXSection',
                )
            else:
                logger.warning('Could not find WaterFSFX section file with id %s in %s', water_fsfx_id, source_dir)

        unit_data = dict(getattr(level, 'unit_data', {}) or {})
        if unit_data:
            unit_obj = self._get_or_create_component_empty(collection, 'UnitData')
            self._set_custom_property(unit_obj, 'trlau_type', 'UnitData')
            self._set_custom_property(unit_obj, 'trlau_unitdata_empty', True)

            basecam_id = int(unit_data.get('base_camera_basecam', 0) or 0)
            if basecam_id > 0:
                basecam_path = self._find_section_file_by_id(source_dir, basecam_id)
                if basecam_path is not None:
                    self._import_section_metadata_empty(
                        collection,
                        unit_obj,
                        self._prefixed_name(drm_name, 'basecam'),
                        'basecam',
                        basecam_id,
                        basecam_path,
                        type_name='BaseCameraSection',
                    )
                else:
                    logger.warning('Could not find BaseCamera/basecam section file with id %s in %s', basecam_id, source_dir)

            self._import_unit_data_fsfx_section_metadata(
                collection,
                source_dir,
                unit_obj,
                unit_data.get('fsfx_links') or [],
                drm_name,
            )
            self._import_unit_data_cine_section_metadata(
                collection,
                source_dir,
                unit_obj,
                unit_data.get('cine_entries') or [],
                drm_name,
            )

    @staticmethod
    def _root_objects_from_result(imported_result: dict):
        model_root_obj = imported_result.get('model_root_obj')
        if model_root_obj is not None:
            return [model_root_obj]

        arm_obj = imported_result.get('arm_obj')
        if arm_obj is not None:
            return [arm_obj]

        mesh_objects = list(imported_result.get('mesh_objects', []) or [])
        if not mesh_objects and imported_result.get('mesh_obj') is not None:
            mesh_objects = [imported_result['mesh_obj']]
        mesh_object_set = set(mesh_objects)
        root_objects = [obj for obj in mesh_objects if getattr(obj, 'parent', None) not in mesh_object_set]
        return root_objects or mesh_objects

    @staticmethod
    def _normalize_angle(value: float) -> float:
        return ((float(value) + math.pi) % (2.0 * math.pi)) - math.pi

    @staticmethod
    def _intro_rotation_raw_xyz(rotation_values) -> tuple[float, float, float]:
        rotation = tuple(float(v) for v in rotation_values)
        rx = rotation[0] if len(rotation) > 0 else 0.0
        ry = rotation[1] if len(rotation) > 1 else 0.0
        rz = rotation[2] if len(rotation) > 2 else 0.0
        return (float(rx), float(ry), float(rz))

    @staticmethod
    def _intro_rotation_to_blender(rotation_values, *, model_basis: bool = False) -> tuple[str, tuple[float, ...]]:
        rx, ry, rz = LevelImporterMixin._intro_rotation_raw_xyz(rotation_values)
        if model_basis:
            rz = LevelImporterMixin._normalize_angle(math.pi - rz)
        return ('XYZ', (float(rx), float(ry), float(rz)))

    @staticmethod
    def _intro_rotation_uses_model_basis(root_objects) -> bool:
        return any(getattr(obj, 'type', '') == 'ARMATURE' for obj in list(root_objects or []))

    def _apply_intro_transform(self, imported_result: dict, intro, parent_empty=None) -> None:
        position = tuple(float(v) for v in getattr(intro, 'position', (0.0, 0.0, 0.0)))
        raw_rotation = tuple(float(v) for v in getattr(intro, 'rotation', (0.0, 0.0, 0.0, 1.0)))
        scale_values = (1.0, 1.0, 1.0, 0.0)
        scale = (1.0, 1.0, 1.0)
        root_objects = self._root_objects_from_result(imported_result)
        model_basis = self._intro_rotation_uses_model_basis(root_objects)
        rotation_mode, rotation = self._intro_rotation_to_blender(raw_rotation, model_basis=model_basis)
        rotation_basis = 'MODEL_ARMATURE_Z180' if model_basis else 'RAW_XYZ'

        transformed_rotation = rotation

        if parent_empty is not None:
            try:
                parent_empty.location = position
                parent_empty.rotation_mode = rotation_mode
                if rotation_mode == 'QUATERNION':
                    parent_empty.rotation_quaternion = transformed_rotation
                else:
                    parent_empty.rotation_euler = tuple(float(v) for v in transformed_rotation[:3])
                parent_empty.scale = scale
                trlau_set_intro_data_property(
                    parent_empty,
                    position=position,
                    rotation=raw_rotation,
                    rotation_mode=rotation_mode,
                    rotation_basis=rotation_basis,
                    scale=scale_values,
                )
            except Exception:
                pass
        for obj in root_objects:
            try:
                if parent_empty is not None:
                    obj.parent = parent_empty
                    obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                    obj.location = (0.0, 0.0, 0.0)
                    obj.rotation_mode = 'XYZ'
                    obj.rotation_euler = (0.0, 0.0, 0.0)
                    obj.scale = (1.0, 1.0, 1.0)
                else:
                    obj.parent = None
                    obj.location = position
                    obj.rotation_mode = rotation_mode
                    if rotation_mode == 'QUATERNION':
                        obj.rotation_quaternion = transformed_rotation
                    else:
                        obj.rotation_euler = tuple(float(v) for v in transformed_rotation[:3])
                    obj.scale = scale
                self._set_custom_property(obj, 'trlau_type', 'IntroData')
                trlau_set_intro_data_property(
                    obj,
                    index=int(getattr(intro, 'index', -1)),
                    object_id=int(getattr(intro, 'object_id', -1)),
                    intro_num=int(getattr(intro, 'intro_num', -1)),
                    unique_id=int(getattr(intro, 'unique_id', -1)),
                    position=position,
                    rotation=raw_rotation,
                    rotation_mode=rotation_mode,
                    rotation_basis=rotation_basis,
                    scale=scale_values,
                    start_frame=int(getattr(intro, 'start_frame', 0)),
                    end_frame=int(getattr(intro, 'end_frame', 0)),
                    intro_flags=int(getattr(intro, 'intro_flags', 0)),
                    attached_vmo=int(getattr(intro, 'attached_vmo', 0)),
                    multi_spline=int(getattr(intro, 'multi_spline', 0)),
                )
                self._store_multi_spline_json_property(obj, 'trlau_intro_multi_spline_json', getattr(intro, 'multi_spline_data', None))
            except Exception:
                pass

    @staticmethod
    def _link_object_to_collection(obj, collection) -> None:
        if obj is None or collection is None:
            return
        if all(coll != collection for coll in getattr(obj, 'users_collection', ())):
            try:
                collection.objects.link(obj)
            except Exception:
                pass

    def _get_level_root_empty(self, collection):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        root_name = self._prefixed_name(drm_name, 'Level')
        root = bpy.data.objects.get(root_name)
        if root is None:
            root = bpy.data.objects.new(root_name, None)
        self._link_object_to_collection(root, collection)
        root.empty_display_type = 'PLAIN_AXES'
        root.empty_display_size = 96.0
        root.parent = None
        scene_center = getattr(self, '_current_level_scene_center_offset', None)
        if scene_center is None:
            scene_center = root.get('trlau_scene_center_offset', (0.0, 0.0, 0.0, 0.0))
        try:
            scene_center_tuple = tuple(float(v) for v in scene_center)
        except Exception:
            scene_center_tuple = (0.0, 0.0, 0.0, 0.0)
        if len(scene_center_tuple) < 4:
            scene_center_tuple = tuple(scene_center_tuple) + ((0.0,) * (4 - len(scene_center_tuple)))
        root.location = (scene_center_tuple[0], scene_center_tuple[1], scene_center_tuple[2])
        root.rotation_mode = 'XYZ'
        root.rotation_euler = (0.0, 0.0, 0.0)
        root.scale = (1.0, 1.0, 1.0)
        self._set_custom_property(root, 'trlau_level', True)
        self._set_custom_property(root, 'trlau_level_root', True)
        self._set_custom_property(root, 'trlau_drm_name', drm_name)
        for stale_key in ('trlau_scene_center_offset', 'trlau_scene_center_offset_w'):
            if stale_key in root:
                try:
                    del root[stale_key]
                except Exception:
                    pass
        return root

    def _get_or_create_component_empty(self, collection, component_name: str, *, display_size: float = 56.0):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        empty_name = self._prefixed_name(drm_name, component_name)
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = float(display_size)
        empty.parent = self._get_level_root_empty(collection)
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        self._set_custom_property(empty, 'trlau_level', True)
        self._set_custom_property(empty, 'trlau_component_empty', True)
        self._set_custom_property(empty, 'trlau_component_name', component_name)
        return empty

    def _parent_result_to_empty(self, imported_result: dict, empty_obj) -> None:
        if empty_obj is None:
            return
        for obj in self._root_objects_from_result(imported_result):
            try:
                obj.parent = empty_obj
                obj.matrix_parent_inverse.identity()
                obj.location = (0.0, 0.0, 0.0)
                obj.rotation_euler = (0.0, 0.0, 0.0)
                obj.scale = (1.0, 1.0, 1.0)
            except Exception:
                pass

    @staticmethod
    def _standalone_section_id(path: Path, endian: str = '<') -> int | None:
        try:
            with open(path, 'rb') as fh:
                info = TRSectionParser.parse(BinaryReader(fh, endian=endian))
            return int(getattr(info, 'section_id', 0))
        except Exception:
            return None

    @staticmethod
    def _section_id_matches_anim_id(section_id: int | None, anim_id: int) -> bool:
        if section_id is None:
            return False
        return int(section_id) == int(anim_id) or (int(section_id) & 0xFFFF) == (int(anim_id) & 0xFFFF)

    def _find_intro_animation_path_for_id(self, extracted_paths, anim_id: int) -> Path | None:
        probe_animation_file = getattr(self, '_probe_animation_file', None)
        direct_section_match: Path | None = None

        for animation_path in sorted((Path(path) for path in extracted_paths), key=lambda item: item.name):
            if animation_path.suffix.lower() in {'.obj', '.level', '.pcd'}:
                continue

            section_id = self._standalone_section_id(animation_path, '<')
            if self._section_id_matches_anim_id(section_id, int(anim_id)) and direct_section_match is None:
                direct_section_match = animation_path

            if probe_animation_file is None:
                continue
            parsed = probe_animation_file(
                str(animation_path),
                default_endianness='<',
                platform=getattr(self, 'platform', 'PC'),
            )
            if parsed is not None and int(parsed.get('anim_id', -1)) == int(anim_id):
                return animation_path

        return direct_section_match

    def _import_intro_first_animation(self, context, drm_path: Path, imported_results: list[dict], parent_empty, action_prefix: str | None = None) -> bool:
        for stale_key in ('trlau_intro_animation_id', 'trlau_intro_animation_action', 'trlau_intro_animation_nla_strip'):
            try:
                if parent_empty is not None and stale_key in parent_empty:
                    del parent_empty[stale_key]
            except Exception:
                pass

        if not imported_results:
            return False

        import_animation_file = getattr(self, '_import_animation_file', None)
        create_looping_nla_strip = getattr(self, '_create_looping_nla_strip', None)
        if import_animation_file is None or create_looping_nla_strip is None:
            return False

        armatures = [result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None]
        armature = armatures[0] if armatures else None
        if armature is None:
            logger.debug('Skipping IntroData animation import for %s: imported model has no armature', getattr(parent_empty, 'name', '<unknown>'))
            return False

        try:
            with DRMContainerParser(
                str(drm_path),
                decompress_derickw=(str(getattr(self, 'platform', '')).upper() == 'PSP'),
            ).temporary_extract_sections() as (_extract_dir, extracted_paths, _sections):
                if not extracted_paths:
                    return False
                object_path = Path(extracted_paths[0])
                finder = getattr(self, '_find_object_root_with_model_refs', None)
                if callable(finder):
                    object_path = Path(finder(extracted_paths))
                first_anim_id = TRObjectParser(str(object_path)).parse_first_animation_id()
                if first_anim_id is None:
                    logger.debug('Skipping IntroData animation import for %s: no first animation entry found', getattr(parent_empty, 'name', '<unknown>'))
                    return False

                animation_path = self._find_intro_animation_path_for_id(extracted_paths, int(first_anim_id))
                if animation_path is None:
                    logger.debug('Skipping IntroData animation import for %s: animation ID %d was not found in %s', getattr(parent_empty, 'name', '<unknown>'), int(first_anim_id), drm_path.name)
                    return False

                safe_prefix = str(action_prefix or getattr(parent_empty, 'name', drm_path.stem) or drm_path.stem)
                animation_result = import_animation_file(
                    context,
                    str(animation_path),
                    armature=armature,
                    default_endianness='<',
                    action_name=f'{safe_prefix}_Anim_{int(first_anim_id)}',
                    platform=getattr(self, 'platform', 'PC'),
                )
                action = animation_result.get('animation_action')
                final_frame = int(animation_result.get('animation_final_frame', 0) or 0)
                strip = create_looping_nla_strip(
                    context,
                    armature,
                    action,
                    final_frame,
                    track_name='IntroData First Animation',
                    strip_name=f'{action.name}_Loop' if action is not None else None,
                )
                return strip is not None
        except Exception as exc:
            logger.warning('IntroData animation import failed for %s: %s', drm_path.name, exc)
            return False

    def _create_transform_empty(self, collection, name: str, location=(0.0, 0.0, 0.0), rotation_quaternion=None, scale=(1.0, 1.0, 1.0), display_size: float = 32.0):
        empty = bpy.data.objects.get(name)
        if empty is None:
            empty = bpy.data.objects.new(name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = float(display_size)
        empty.parent = None
        empty.location = tuple(float(v) for v in location)
        if rotation_quaternion is not None:
            empty.rotation_mode = 'QUATERNION'
            empty.rotation_quaternion = rotation_quaternion
        else:
            empty.rotation_mode = 'XYZ'
            empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = tuple(float(v) for v in scale)
        return empty

    def _create_player_object_empty(self, collection, level):
        player_object_id = int(getattr(level, 'player_object_id', -1))
        empty = self._create_transform_empty(collection, f'PlayerObject_{player_object_id:03d}', display_size=48.0)
        empty['trlau_type'] = 'PlayerObject'
        empty['trlau_player_object_id'] = player_object_id
        return empty

    def _create_intro_empty(self, collection, intro, object_name: str):
        raw_rotation = tuple(float(v) for v in getattr(intro, 'rotation', (0.0, 0.0, 0.0, 1.0)))
        rotation_mode, rotation = self._intro_rotation_to_blender(raw_rotation, model_basis=False)
        rotation_basis = 'RAW_XYZ'
        scale = (1.0, 1.0, 1.0)
        empty = self._create_transform_empty(
            collection,
            f'IntroData_{int(getattr(intro, "index", 0)):03d}_{object_name}',
            location=tuple(float(v) for v in getattr(intro, 'position', (0.0, 0.0, 0.0))),
            rotation_quaternion=rotation if rotation_mode == 'QUATERNION' else None,
            scale=scale if len(scale) == 3 else (1.0, 1.0, 1.0),
            display_size=40.0,
        )
        if rotation_mode != 'QUATERNION':
            empty.rotation_mode = rotation_mode
            empty.rotation_euler = rotation[:3]
        self._set_custom_property(empty, 'trlau_type', 'IntroData')
        trlau_set_intro_data_property(
            empty,
            index=int(getattr(intro, 'index', -1)),
            object_id=int(getattr(intro, 'object_id', -1)),
            intro_num=int(getattr(intro, 'intro_num', -1)),
            unique_id=int(getattr(intro, 'unique_id', -1)),
            position=tuple(float(v) for v in getattr(intro, 'position', (0.0, 0.0, 0.0))),
            rotation=raw_rotation,
            rotation_mode=rotation_mode,
            rotation_basis=rotation_basis,
            scale=(1.0, 1.0, 1.0, 0.0),
            start_frame=int(getattr(intro, 'start_frame', 0)),
            end_frame=int(getattr(intro, 'end_frame', 0)),
            intro_flags=int(getattr(intro, 'intro_flags', 0)),
            attached_vmo=int(getattr(intro, 'attached_vmo', 0)),
            multi_spline=int(getattr(intro, 'multi_spline', 0)),
        )
        self._create_intro_data_metadata_empties(collection, empty, intro)
        self._store_multi_spline_json_property(empty, 'trlau_intro_multi_spline_json', getattr(intro, 'multi_spline_data', None))
        return empty

    @staticmethod
    def _orient_object_towards_target(obj, target_position) -> None:
        try:
            direction = mathutils.Vector(target_position) - obj.location
            if direction.length > 1e-6:
                obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        except Exception:
            pass

    def _level_objectlist_game_dir(self, level) -> str | None:
        source_game = str(getattr(level, 'source_game', '') or '').strip().lower()
        if source_game in {'anniversary', 'tra', 'trae'}:
            return 'trae'
        if source_game in {'legend', 'tr7'}:
            return 'tr7'
        return None

    def _player_intro_object_id(self, level) -> int | None:
        game_dir = self._level_objectlist_game_dir(level)
        if game_dir == 'trae':
            return 72
        if game_dir == 'tr7':
            return 171
        return None


    def _level_objectlist_candidate_dirs(self, filepath: str, level) -> list[Path]:
        source_path = Path(filepath)
        candidates: list[Path] = []
        game_dir = self._level_objectlist_game_dir(level)
        if game_dir:
            candidates.append(source_path.parent.parent / game_dir / 'pc-w')
            candidates.append(source_path.parent / game_dir / 'pc-w')
        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    def _load_level_objectlist_mapping(self, filepath: str, level) -> tuple[dict[int, str], Path | None]:
        for directory in self._level_objectlist_candidate_dirs(filepath, level):
            mapping = self._load_objectlist_mapping(directory)
            if mapping:
                logger.debug('Loaded objectlist.txt for level import from %s', directory / 'objectlist.txt')
                return mapping, directory
        return {}, None

    def _find_level_named_drm(self, filepath: str, objectlist_dir: Path | None, object_name: str) -> Path | None:
        source_dir = Path(filepath).parent
        directories = [objectlist_dir, source_dir]
        seen: set[str] = set()
        for directory in directories:
            if directory is None:
                continue
            try:
                key = str(directory.resolve())
            except Exception:
                key = str(directory)
            if key in seen:
                continue
            seen.add(key)
            drm_path = self._find_named_drm(directory, object_name)
            if drm_path is not None:
                return drm_path
        return None

    def _import_player_object(self, context, filepath: str, collection, level) -> list[dict]:
        player_object_id = int(getattr(level, 'player_object_id', -1))
        if player_object_id < 0:
            return []
        objectlist, objectlist_dir = self._load_level_objectlist_mapping(filepath, level)
        if not objectlist:
            logger.warning('playerObjectID found in %s but objectlist.txt was not found in the expected game object-list folder; skipping player object import', Path(filepath).name)
            return []
        object_name = objectlist.get(player_object_id)
        if not object_name:
            logger.warning('No objectlist.txt entry for playerObjectID %s', player_object_id)
            return []
        drm_path = self._find_level_named_drm(filepath, objectlist_dir, object_name)
        if drm_path is None:
            logger.warning('Could not find playerObjectID DRM for object %s (%s)', player_object_id, object_name)
            return []

        player_empty = self._get_or_create_component_empty(collection, 'PlayerObject')
        imported_results = self._import_drm_first_model(context, drm_path, collection, include_hinfo=False)
        for result in imported_results:
            for obj in self._root_objects_from_result(result):
                try:
                    obj.parent = player_empty
                    obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                    obj.location = (0.0, 0.0, 0.0)
                    obj.rotation_mode = 'XYZ'
                    obj.rotation_euler = (0.0, 0.0, 0.0)
                    obj.scale = (1.0, 1.0, 1.0)
                    self._set_custom_property(obj, 'trlau_type', 'PlayerObject')
                    self._set_custom_property(obj, 'trlau_player_object_id', player_object_id)
                except Exception:
                    pass
        return imported_results


    def _build_unit_data_metadata(self, collection, level):
        unit_data = dict(getattr(level, 'unit_data', {}) or {})
        if not unit_data:
            return None
        unit_obj = self._get_or_create_component_empty(collection, 'UnitData')
        self._set_custom_property(unit_obj, 'trlau_type', 'UnitData')
        self._set_custom_property(unit_obj, 'trlau_unitdata_empty', True)
        skipped_keys = {
            'num_fsfx',
            'num_event_variable_storage',
            'p_fsfx',
            'p_event_variable_storage',
            'num_cines',
            'p_cines',
            'cine_entries',
            'fsfx_links',
            'event_variable_storage_entries',
        }
        for key, value in unit_data.items():
            if key in skipped_keys:
                continue
            self._set_custom_property(unit_obj, f'{UNIT_DATA_PROP_PREFIX}{key}', value)
        self._build_unit_data_fsfx_link_empties(collection, unit_obj, unit_data.get('fsfx_links') or [])
        self._build_unit_data_cine_empties(collection, unit_obj, unit_data.get('cine_entries') or [])
        self._build_unit_data_event_variable_empties(collection, unit_obj, unit_data.get('event_variable_storage_entries') or [])
        return unit_obj

    @staticmethod
    def _clean_unitdata_cine_wrapper_props(obj) -> None:
        if obj is None:
            return
        keep = {'trlau_unitdata_cine_empty', 'trlau_unitdata_cine_index'}
        try:
            for key in list(obj.keys()):
                if key.startswith('trlau_unitdata_cine_') and key not in keep:
                    try:
                        del obj[key]
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def _remove_stale_cine_section_children(cine_obj) -> None:
        if cine_obj is None:
            return
        try:
            children = list(getattr(cine_obj, 'children', []) or [])
        except Exception:
            children = []
        for child in children:
            try:
                is_old_cine_section = (
                    getattr(child, 'type', None) == 'EMPTY'
                    and (
                        str(child.get('trlau_section_role', '') or '') == 'Cine'
                        or str(child.get('trlau_type', '') or '') == 'CineSection'
                        or str(getattr(child, 'name', '') or '').endswith('_Section')
                    )
                )
            except Exception:
                is_old_cine_section = False
            if not is_old_cine_section:
                continue
            try:
                bpy.data.objects.remove(child, do_unlink=True)
            except Exception:
                pass

    def _build_unit_data_cine_empties(self, collection, unit_obj, entries):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        for index, entry in enumerate(list(entries or [])):
            if not isinstance(entry, dict):
                continue
            try:
                cine_index = int(entry.get('index', index) or index)
            except Exception:
                cine_index = int(index)
            empty_name = self._prefixed_name(drm_name, f'UnitData_Cine_{cine_index:03d}')
            empty = bpy.data.objects.get(empty_name)
            if empty is None:
                empty = bpy.data.objects.new(empty_name, None)
            self._link_object_to_collection(empty, collection)
            empty.empty_display_type = 'PLAIN_AXES'
            empty.empty_display_size = 32.0
            empty.parent = unit_obj
            empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            empty.location = (0.0, 0.0, 0.0)
            empty.rotation_mode = 'XYZ'
            empty.rotation_euler = (0.0, 0.0, 0.0)
            empty.scale = (1.0, 1.0, 1.0)
            self._set_custom_property(empty, 'trlau_level', True)
            self._set_custom_property(empty, 'trlau_type', 'UnitDataCine')
            self._set_custom_property(empty, 'trlau_unitdata_cine_empty', True)
            self._set_custom_property(empty, 'trlau_unitdata_cine_index', int(cine_index))
            self._clean_unitdata_cine_wrapper_props(empty)
            self._remove_stale_cine_section_children(empty)

            try:
                section_index = int(entry.get('section_index', -1) or -1)
            except Exception:
                section_index = -1
            try:
                section_id = int(entry.get('section_id', 0) or 0)
            except Exception:
                section_id = 0

            # A UnitData Cine array entry is itself the CineData resource in the
            # Blender hierarchy.  Older builds created a wrapper plus a child
            # "_Section" empty with the real data, which duplicated the same
            # cinematic and made selection/export ambiguous.
            if section_index >= 0 or section_id > 0 or entry.get('cine_structured') or entry.get('name'):
                self._set_custom_property(empty, 'trlau_section_metadata_empty', True)
                self._set_custom_property(empty, 'trlau_section_role', 'Cine')
                self._set_custom_property(empty, 'trlau_section_original_id', int(section_id))
                self._set_custom_property(empty, 'trlau_cine_section_index', int(section_index))
            if entry.get('cine_structured') or entry.get('name'):
                self._store_cine_entry_metadata_properties(empty, entry)

    def _build_unit_data_fsfx_link_empties(self, collection, unit_obj, entries):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        for index, entry in enumerate(list(entries or [])):
            if not isinstance(entry, dict):
                continue
            empty_name = self._prefixed_name(drm_name, f'UnitData_FSFXLink_{index:03d}')
            empty = bpy.data.objects.get(empty_name)
            if empty is None:
                empty = bpy.data.objects.new(empty_name, None)
            self._link_object_to_collection(empty, collection)
            empty.empty_display_type = 'PLAIN_AXES'
            empty.empty_display_size = 24.0
            empty.parent = unit_obj
            empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            empty.location = (0.0, 0.0, 0.0)
            empty.rotation_mode = 'XYZ'
            empty.rotation_euler = (0.0, 0.0, 0.0)
            empty.scale = (1.0, 1.0, 1.0)
            self._set_custom_property(empty, 'trlau_level', True)
            self._set_custom_property(empty, 'trlau_type', 'UnitDataFSFXLink')
            self._set_custom_property(empty, 'trlau_unitdata_fsfx_link_empty', True)
            self._set_custom_property(empty, 'trlau_unitdata_fsfx_index', int(entry.get('index', index) or 0))
            self._set_custom_property(empty, 'trlau_unitdata_fsfx_id', int(entry.get('id', 0) or 0))
            self._set_custom_property(empty, 'trlau_unitdata_fsfx_alpha', float(entry.get('alpha', 0.0) or 0.0))
            self._set_custom_property(empty, 'trlau_unitdata_fsfx_enabled', int(1 if entry.get('enabled', 0) else 0))

    def _find_unit_data_fsfx_link_empty(self, collection, unit_obj, index: int, fsfx_id: int = 0):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if unit_obj is not None:
                parent = getattr(obj, 'parent', None)
                is_descendant = parent == unit_obj
                while parent is not None and not is_descendant:
                    parent = getattr(parent, 'parent', None)
                    is_descendant = parent == unit_obj
                if not is_descendant:
                    continue
            if not bool(obj.get('trlau_unitdata_fsfx_link_empty')):
                continue
            try:
                obj_index = int(obj.get('trlau_unitdata_fsfx_index', -1) or -1)
            except Exception:
                obj_index = -1
            if obj_index == int(index):
                return obj
        return None

    def _find_unit_data_cine_empty(self, collection, unit_obj, index: int, section_index: int = -1, section_id: int = 0):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if unit_obj is not None:
                parent = getattr(obj, 'parent', None)
                is_descendant = parent == unit_obj
                while parent is not None and not is_descendant:
                    parent = getattr(parent, 'parent', None)
                    is_descendant = parent == unit_obj
                if not is_descendant:
                    continue
            if not bool(obj.get('trlau_unitdata_cine_empty')):
                continue
            try:
                obj_index = int(obj.get('trlau_unitdata_cine_index', -1) or -1)
            except Exception:
                obj_index = -1
            if obj_index == int(index):
                return obj
        return None

    def _import_unit_data_cine_section_metadata(self, collection, source_dir: Path, unit_obj, entries, drm_name: str) -> None:
        if source_dir is None or not Path(source_dir).exists():
            return
        source_dir = Path(source_dir)
        for entry_index, entry in enumerate(list(entries or [])):
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get('index', entry_index) or entry_index)
            except Exception:
                index = int(entry_index)
            try:
                section_index = int(entry.get('section_index', -1) or -1)
            except Exception:
                section_index = -1
            try:
                section_id = int(entry.get('section_id', 0) or 0)
            except Exception:
                section_id = 0
            cine_empty = self._find_unit_data_cine_empty(collection, unit_obj, index, section_index, section_id)
            if cine_empty is None:
                continue
            section_path = None
            if section_index >= 0:
                section_path = self._find_section_file_by_index(source_dir, section_index)
            if section_path is None:
                section_file = str(entry.get('section_file', '') or '').strip()
                if section_file:
                    candidate = source_dir / section_file
                    if candidate.exists():
                        section_path = candidate
            if section_path is None and section_id > 0:
                section_path = self._find_section_file_by_id(source_dir, section_id)
            if section_path is None:
                if entry.get('cine_structured') or entry.get('name'):
                    self._set_custom_property(cine_empty, 'trlau_section_metadata_empty', True)
                    self._set_custom_property(cine_empty, 'trlau_section_role', 'Cine')
                    self._set_custom_property(cine_empty, 'trlau_section_original_id', int(section_id))
                    self._set_custom_property(cine_empty, 'trlau_cine_section_index', int(section_index))
                    self._store_cine_entry_metadata_properties(cine_empty, entry)
                    self._clean_unitdata_cine_wrapper_props(cine_empty)
                    self._remove_stale_cine_section_children(cine_empty)
                    continue
                logger.warning('Could not find Cine section file index %s id %s in %s', section_index, section_id, source_dir)
                continue

            cine_empty = self._import_section_metadata_empty(
                collection,
                unit_obj,
                self._prefixed_name(drm_name, f'UnitData_Cine_{index:03d}'),
                'Cine',
                section_id,
                section_path,
                type_name='UnitDataCine',
            ) or cine_empty
            self._set_custom_property(cine_empty, 'trlau_unitdata_cine_empty', True)
            self._set_custom_property(cine_empty, 'trlau_unitdata_cine_index', int(index))
            self._set_custom_property(cine_empty, 'trlau_cine_section_index', int(self._section_index_from_filename(section_path)))
            self._clean_unitdata_cine_wrapper_props(cine_empty)
            self._remove_stale_cine_section_children(cine_empty)

    def _import_unit_data_fsfx_section_metadata(self, collection, source_dir: Path, unit_obj, entries, drm_name: str) -> None:
        if source_dir is None or not Path(source_dir).exists():
            return
        for entry_index, entry in enumerate(list(entries or [])):
            if not isinstance(entry, dict):
                continue
            try:
                fsfx_id = int(entry.get('id', 0) or 0)
            except Exception:
                fsfx_id = 0
            if fsfx_id <= 0:
                continue
            try:
                index = int(entry.get('index', entry_index) or entry_index)
            except Exception:
                index = int(entry_index)
            link_empty = self._find_unit_data_fsfx_link_empty(collection, unit_obj, index, fsfx_id)
            if link_empty is None:
                continue
            section_path = self._find_section_file_by_id(Path(source_dir), fsfx_id)
            if section_path is None:
                logger.warning('Could not find FSFXLink section file with id %s in %s', fsfx_id, source_dir)
                continue
            self._import_section_metadata_empty(
                collection,
                link_empty,
                self._prefixed_name(drm_name, f'UnitData_FSFXLink_{index:03d}_Section'),
                'FSFXLink',
                fsfx_id,
                section_path,
                type_name='FSFXLinkSection',
            )

    def _build_unit_data_event_variable_empties(self, collection, unit_obj, entries):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        for index, entry in enumerate(list(entries or [])):
            if not isinstance(entry, dict):
                continue
            empty_name = self._prefixed_name(drm_name, f'UnitData_EventVariableStorage_{index:03d}')
            empty = bpy.data.objects.get(empty_name)
            if empty is None:
                empty = bpy.data.objects.new(empty_name, None)
            self._link_object_to_collection(empty, collection)
            empty.empty_display_type = 'PLAIN_AXES'
            empty.empty_display_size = 24.0
            empty.parent = unit_obj
            empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            empty.location = (0.0, 0.0, 0.0)
            empty.rotation_mode = 'XYZ'
            empty.rotation_euler = (0.0, 0.0, 0.0)
            empty.scale = (1.0, 1.0, 1.0)
            self._set_custom_property(empty, 'trlau_level', True)
            self._set_custom_property(empty, 'trlau_type', 'UnitDataEventVariableStorage')
            self._set_custom_property(empty, 'trlau_unitdata_event_variable_empty', True)
            self._set_custom_property(empty, 'trlau_unitdata_event_variable_index', int(entry.get('index', index) or 0))
            self._set_custom_property(empty, 'trlau_unitdata_event_variable_variable', int(entry.get('variable', 0) or 0))
            self._set_custom_property(empty, 'trlau_unitdata_event_variable_medium_value', int(entry.get('medium_value', 0) or 0))
            self._set_custom_property(empty, 'trlau_unitdata_event_variable_easy_value', int(entry.get('easy_value', 0) or 0))
            self._set_custom_property(empty, 'trlau_unitdata_event_variable_hard_value', int(entry.get('hard_value', 0) or 0))

    def _build_admd_metadata(self, collection, level):
        admd_data = dict(getattr(level, 'admd_data', {}) or {})
        if not admd_data:
            return None
        admd_obj = self._get_or_create_component_empty(collection, 'ADMDData')
        self._set_custom_property(admd_obj, 'trlau_type', 'ADMDData')
        self._set_custom_property(admd_obj, 'trlau_admd_empty', True)
        for key, value in admd_data.items():
            self._set_custom_property(admd_obj, f'{ADMD_DATA_PROP_PREFIX}{key}', value)
        return admd_obj

    @staticmethod
    def _area_dbase_section_index_from_name(path: Path) -> int:
        try:
            return int(str(path.stem).split('_', 1)[0])
        except Exception:
            return -1

    def _find_area_dbase_section_paths(self, source_dir: Path, expected_name: str | None = None) -> list[Path]:
        if source_dir is None or not Path(source_dir).is_dir():
            return []
        expected = str(expected_name or '').strip().lower()
        matches: list[tuple[int, Path]] = []
        for candidate in sorted(Path(source_dir).iterdir(), key=lambda item: item.name.lower()):
            if not candidate.is_file() or candidate.suffix.lower() != '.gnc':
                continue
            try:
                if not AreaDBaseParser.looks_like_area_dbase(candidate):
                    continue
                score = 10
                if expected:
                    try:
                        parsed = AreaDBaseParser(candidate).parse()
                        if str(parsed.name or '').strip().lower() == expected:
                            score = 0
                    except Exception:
                        pass
                matches.append((score, candidate))
            except Exception:
                continue
        matches.sort(key=lambda item: (item[0], self._area_dbase_section_index_from_name(item[1]), item[1].name.lower()))
        return [path for _score, path in matches]

    @staticmethod
    def _clear_area_dbase_object_metadata(obj, keep: set[str] | None = None) -> None:
        keep = set(keep or set())
        prefixes = (
            'trlau_area_dbase_',
            'trlau_section_blob_',
        )
        exact = {
            'trlau_section_blob_encoding',
            'trlau_section_blob_chunk_count',
            'trlau_section_blob_size',
            'trlau_section_metadata_empty',
            'trlau_section_role',
            'trlau_section_file_name',
            'trlau_section_source_path',
            'trlau_section_original_id',
            'trlau_section_type',
            'trlau_section_version_id',
            'trlau_section_spec_mask',
            'trlau_type',
            'trlau_component_empty',
            'trlau_component_name',
            'trlau_level',
        }
        try:
            keys = list(obj.keys())
        except Exception:
            return
        for key in keys:
            key_s = str(key)
            if key_s in keep:
                continue
            if key_s in exact or any(key_s.startswith(prefix) for prefix in prefixes):
                try:
                    del obj[key]
                except Exception:
                    pass

    def _get_or_create_area_dbase_empty(self, collection, area_dbase: AreaDBaseData, section_path: Path | None = None):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        empty_name = self._prefixed_name(drm_name, 'AreaDBase')
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
        self._link_object_to_collection(empty, collection)
        empty.empty_display_type = 'CUBE'
        empty.empty_display_size = 96.0
        empty.parent = self._get_level_root_empty(collection)
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = convert_area_dbase_center_position(*area_dbase.center)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        self._clear_area_dbase_object_metadata(empty)
        return empty

    @staticmethod
    def _get_or_create_area_dbase_material(name: str, color: tuple[float, float, float, float]):
        material = bpy.data.materials.get(name)
        if material is None:
            material = bpy.data.materials.new(name)
        try:
            material.diffuse_color = tuple(float(v) for v in color)
        except Exception:
            pass
        return material

    def _build_area_dbase_stitched_positions(self, area_dbase: AreaDBaseData) -> dict[tuple[int, int], tuple[float, float, float]]:
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        position: dict[tuple[int, int], tuple[float, float, float]] = {}

        def key(area_index: int, edge_index: int) -> tuple[int, int]:
            return (int(area_index), int(edge_index))

        def find(item: tuple[int, int]) -> tuple[int, int]:
            parent.setdefault(item, item)
            root = item
            while parent[root] != root:
                root = parent[root]
            while parent[item] != item:
                next_item = parent[item]
                parent[item] = root
                item = next_item
            return root

        def union(a: tuple[int, int], b: tuple[int, int]) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        def xy_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
            return abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))

        areas = list(getattr(area_dbase, 'areas', []) or [])
        by_ref = {int(area.area_ref): area for area in areas}
        for area in areas:
            for edge_index, edge in enumerate(area.edges):
                k = key(area.index, edge_index)
                parent[k] = k
                position[k] = tuple(float(v) for v in edge.blender_position)

        for area in areas:
            edges = list(area.edges)
            if len(edges) < 2:
                continue
            for edge_index, edge in enumerate(edges):
                if int(getattr(edge, 'edge_type', -1)) != 0:
                    continue
                target = by_ref.get(int(getattr(edge, 'adj_area_ref', -1)))
                if target is None or len(target.edges) < 2:
                    continue

                a0 = position[key(area.index, edge_index)]
                a1 = position[key(area.index, (edge_index + 1) % len(edges))]
                best_index = -1
                best_distance = 1.0e18
                target_edges = list(target.edges)
                for target_index, _target_edge in enumerate(target_edges):
                    b0 = position[key(target.index, target_index)]
                    b1 = position[key(target.index, (target_index + 1) % len(target_edges))]
                    distance = xy_distance(a0, b1) + xy_distance(a1, b0)
                    if distance < best_distance:
                        best_distance = distance
                        best_index = target_index

                if best_index >= 0 and best_distance <= 0.001:
                    union(key(area.index, edge_index), key(target.index, (best_index + 1) % len(target_edges)))
                    union(key(area.index, (edge_index + 1) % len(edges)), key(target.index, best_index))

        groups: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        for item, pos in position.items():
            groups.setdefault(find(item), []).append(pos)

        group_pos: dict[tuple[int, int], tuple[float, float, float]] = {}
        for root, values in groups.items():
            count = float(len(values)) or 1.0
            group_pos[root] = (
                sum(v[0] for v in values) / count,
                sum(v[1] for v in values) / count,
                sum(v[2] for v in values) / count,
            )

        return {item: group_pos[find(item)] for item in position}

    def _build_area_dbase_mesh(self, collection, area_dbase: AreaDBaseData, parent_empty):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        mesh_name = self._prefixed_name(drm_name, 'AreaDBase_Areas')
        display_positions = self._build_area_dbase_stitched_positions(area_dbase)
        vertices: list[tuple[float, float, float]] = []
        vertex_index_by_position: dict[tuple[float, float, float], int] = {}
        faces: list[list[int]] = []
        face_area_indices: list[int] = []

        def add_vertex(pos: tuple[float, float, float]) -> int:
            # Round only for stable deduplication after averaged display Z.
            rounded = (round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2]), 6))
            existing = vertex_index_by_position.get(rounded)
            if existing is not None:
                return int(existing)
            vertex_index = len(vertices)
            vertices.append(tuple(float(v) for v in pos))
            vertex_index_by_position[rounded] = vertex_index
            return vertex_index

        for area in area_dbase.areas:
            if len(area.edges) < 3:
                continue
            face: list[int] = []
            for edge_index, edge in enumerate(area.edges):
                pos = display_positions.get((int(area.index), int(edge_index)), tuple(float(v) for v in edge.blender_position))
                face.append(add_vertex(pos))
            if len(set(face)) >= 3:
                faces.append(face)
                face_area_indices.append(int(area.index))

        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        try:
            mesh.materials.append(self._get_or_create_area_dbase_material('TRLAU AreaDBase Area', (0.25, 0.65, 1.0, 0.22)))
        except Exception:
            pass

        obj = bpy.data.objects.new(mesh_name, mesh)
        self._link_object_to_collection(obj, collection)
        obj.parent = parent_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        try:
            obj.show_wire = True
            obj.show_in_front = True
        except Exception:
            pass
        # Keep the editable mesh clean as well.  Export finds it by hierarchy/name
        # and derives all AreaDBase records from its polygons.
        self._clear_area_dbase_object_metadata(obj)
        return obj

    def _build_area_dbase_edge_curve(self, collection, area_dbase: AreaDBaseData, parent_empty, edge_type: int, label: str, color: tuple[float, float, float, float]):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        curve_name = self._prefixed_name(drm_name, f'AreaDBase_{label}Edges')
        curve_data = bpy.data.curves.new(curve_name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 1
        try:
            curve_data.bevel_depth = 2.0
            curve_data.bevel_resolution = 0
        except Exception:
            pass
        display_positions = self._build_area_dbase_stitched_positions(area_dbase)
        segment_count = 0
        for area in area_dbase.areas:
            edges = list(area.edges)
            if len(edges) < 2:
                continue
            for index, edge in enumerate(edges):
                if int(edge.edge_type) != int(edge_type):
                    continue
                p0 = display_positions.get((int(area.index), int(index)), tuple(float(v) for v in edge.blender_position))
                p1 = display_positions.get((int(area.index), int((index + 1) % len(edges))), tuple(float(v) for v in edges[(index + 1) % len(edges)].blender_position))
                spline = curve_data.splines.new('POLY')
                spline.points.add(1)
                spline.points[0].co = (float(p0[0]), float(p0[1]), float(p0[2]), 1.0)
                spline.points[1].co = (float(p1[0]), float(p1[1]), float(p1[2]), 1.0)
                segment_count += 1
        if segment_count <= 0:
            try:
                bpy.data.curves.remove(curve_data)
            except Exception:
                pass
            return None
        try:
            curve_data.materials.append(self._get_or_create_area_dbase_material(f'TRLAU AreaDBase {label} Edge', color))
        except Exception:
            pass
        obj = bpy.data.objects.new(curve_name, curve_data)
        self._link_object_to_collection(obj, collection)
        obj.parent = parent_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        try:
            obj.show_in_front = True
        except Exception:
            pass
        self._set_custom_property(obj, 'trlau_level', True)
        self._set_custom_property(obj, 'trlau_type', f'AreaDBase{label}Edges')
        self._set_custom_property(obj, 'trlau_area_dbase_edge_curve', True)
        self._set_custom_property(obj, 'trlau_area_dbase_generated_helper', True)
        self._set_custom_property(obj, 'trlau_export_ignore', True)
        self._set_custom_property(obj, 'trlau_area_dbase_edge_type', int(edge_type))
        self._set_custom_property(obj, 'trlau_area_dbase_edge_segment_count', int(segment_count))
        return obj

    def _build_area_dbase_portal_curve(self, collection, area_dbase: AreaDBaseData, parent_empty):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        curve_name = self._prefixed_name(drm_name, 'AreaDBase_Portals')
        curve_data = bpy.data.curves.new(curve_name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 1
        try:
            curve_data.bevel_depth = 3.0
            curve_data.bevel_resolution = 0
        except Exception:
            pass
        segment_count = 0
        portal_count = 0
        for area in area_dbase.areas:
            portals = list(getattr(area, 'portals', []) or [])
            portal_count += len(portals)
            if len(portals) < 2:
                continue
            spline = curve_data.splines.new('POLY')
            spline.points.add(len(portals) - 1)
            for index, portal in enumerate(portals):
                pos = tuple(float(v) for v in portal.blender_position)
                spline.points[index].co = (pos[0], pos[1], pos[2], 1.0)
            segment_count += max(0, len(portals) - 1)
        if portal_count <= 0 or segment_count <= 0:
            try:
                bpy.data.curves.remove(curve_data)
            except Exception:
                pass
            return None
        try:
            curve_data.materials.append(self._get_or_create_area_dbase_material('TRLAU AreaDBase Portal', (1.0, 0.75, 0.15, 1.0)))
        except Exception:
            pass
        obj = bpy.data.objects.new(curve_name, curve_data)
        self._link_object_to_collection(obj, collection)
        obj.parent = parent_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        try:
            obj.show_in_front = True
        except Exception:
            pass
        self._set_custom_property(obj, 'trlau_level', True)
        self._set_custom_property(obj, 'trlau_type', 'AreaDBasePortals')
        self._set_custom_property(obj, 'trlau_area_dbase_portal_curve', True)
        self._set_custom_property(obj, 'trlau_area_dbase_generated_helper', True)
        self._set_custom_property(obj, 'trlau_export_ignore', True)
        self._set_custom_property(obj, 'trlau_area_dbase_portal_count', int(portal_count))
        self._set_custom_property(obj, 'trlau_area_dbase_portal_segment_count', int(segment_count))
        return obj

    def _build_area_dbase_gateway_empties(self, collection, area_dbase: AreaDBaseData, parent_empty):
        gateways = list(getattr(area_dbase, 'gateways', []) or [])
        if not gateways:
            return []
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        group_name = self._prefixed_name(drm_name, 'AreaDBase_Gateways')
        group = bpy.data.objects.get(group_name)
        if group is None:
            group = bpy.data.objects.new(group_name, None)
        self._link_object_to_collection(group, collection)
        group.empty_display_type = 'PLAIN_AXES'
        group.empty_display_size = 96.0
        group.parent = parent_empty
        group.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        self._set_custom_property(group, 'trlau_level', True)
        self._set_custom_property(group, 'trlau_type', 'AreaDBaseGateways')
        self._set_custom_property(group, 'trlau_area_dbase_gateway_group', True)
        self._set_custom_property(group, 'trlau_area_dbase_gateway_count', len(gateways))

        objects = [group]
        for gateway in gateways:
            name = str(getattr(gateway, 'name', '') or f'gateway_{gateway.index:02d}')
            obj_name = self._prefixed_name(drm_name, f'AreaDBase_Gateway_{gateway.index:02d}_{name}')
            empty = bpy.data.objects.get(obj_name)
            if empty is None:
                empty = bpy.data.objects.new(obj_name, None)
            self._link_object_to_collection(empty, collection)
            empty.empty_display_type = 'SPHERE'
            empty.empty_display_size = 64.0
            empty.parent = group
            empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            empty.location = tuple(float(v) for v in gateway.blender_position)
            self._set_custom_property(empty, 'trlau_level', True)
            self._set_custom_property(empty, 'trlau_type', 'AreaDBaseGateway')
            self._set_custom_property(empty, 'trlau_area_dbase_gateway', True)
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_index', int(gateway.index))
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_name', name)
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_dest_unit_name', str(getattr(gateway, 'dest_unit_name', '') or ''))
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_sibling_name', str(getattr(gateway, 'sibling_gateway_name', '') or ''))
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_num_links', int(getattr(gateway, 'num_links', 0)))
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_static_area_ref', int(getattr(gateway, 'static_area_ref', -1)))
            self._set_custom_property(empty, 'trlau_area_dbase_gateway_game_position', tuple(int(v) for v in gateway.game_position))
            objects.append(empty)
        return objects

    def _build_area_dbase_bbox_curve(self, collection, area_dbase: AreaDBaseData, parent_empty):
        drm_name = str(getattr(collection, 'name', '') or '').strip()
        curve_name = self._prefixed_name(drm_name, 'AreaDBase_Bounds')
        curve_data = bpy.data.curves.new(curve_name, type='CURVE')
        curve_data.dimensions = '3D'
        try:
            curve_data.bevel_depth = 1.0
            curve_data.bevel_resolution = 0
        except Exception:
            pass
        min_x, min_y, min_z = area_dbase.box_min
        max_x, max_y, max_z = area_dbase.box_max
        game_corners = [
            (min_x, min_y, min_z), (max_x, min_y, min_z), (max_x, max_y, min_z), (min_x, max_y, min_z),
            (min_x, min_y, max_z), (max_x, min_y, max_z), (max_x, max_y, max_z), (min_x, max_y, max_z),
        ]
        blender_corners = [_convert_level_position(float(x), float(y), float(z)) for x, y, z in game_corners]
        edge_indices = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
        for a, b in edge_indices:
            spline = curve_data.splines.new('POLY')
            spline.points.add(1)
            va = blender_corners[a]
            vb = blender_corners[b]
            spline.points[0].co = (va[0], va[1], va[2], 1.0)
            spline.points[1].co = (vb[0], vb[1], vb[2], 1.0)
        try:
            curve_data.materials.append(self._get_or_create_area_dbase_material('TRLAU AreaDBase Bounds', (1.0, 1.0, 1.0, 0.5)))
        except Exception:
            pass
        obj = bpy.data.objects.new(curve_name, curve_data)
        self._link_object_to_collection(obj, collection)
        obj.parent = parent_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        try:
            obj.show_in_front = True
        except Exception:
            pass
        self._set_custom_property(obj, 'trlau_level', True)
        self._set_custom_property(obj, 'trlau_type', 'AreaDBaseBounds')
        self._set_custom_property(obj, 'trlau_area_dbase_bounds_curve', True)
        self._set_custom_property(obj, 'trlau_area_dbase_generated_helper', True)
        self._set_custom_property(obj, 'trlau_export_ignore', True)
        return obj

    def _import_area_dbase_file(self, context, filepath: str | Path, collection=None, parent_empty=None):
        source_path = Path(filepath)
        area_dbase = AreaDBaseParser(source_path).parse()
        collection = collection or self._get_or_create_collection(context, str(source_path), collection_name=area_dbase.name or source_path.stem)
        parent_empty = parent_empty or self._get_or_create_area_dbase_empty(collection, area_dbase, source_path)
        mesh_obj = self._build_area_dbase_mesh(collection, area_dbase, parent_empty)

        portal_obj = self._build_area_dbase_portal_curve(collection, area_dbase, parent_empty)
        gateway_objects = self._build_area_dbase_gateway_empties(collection, area_dbase, parent_empty)
        generated_count = 0
        for area in area_dbase.areas:
            for edge in area.edges:
                if int(getattr(edge, 'edge_type', -1)) in {0, 1}:
                    generated_count += 1

        print(f'AreaDBase found, importing {len(area_dbase.areas)} areas, {area_dbase.edge_count} edges, {getattr(area_dbase, "portal_count", 0)} portals, and {len(getattr(area_dbase, "gateways", []))} gateways from {source_path.name}...')
        return {
            'area_dbase': area_dbase,
            'mesh_obj': mesh_obj,
            'mesh_objects': [obj for obj in [mesh_obj, portal_obj, *gateway_objects] if obj is not None],
            'arm_obj': None,
            'faces': len(area_dbase.areas),
            'source': str(source_path),
            'area_dbase_generated_edge_count': generated_count,
        }

    def _import_area_dbase_from_source_dir(self, context, filepath: str, collection, section_source_dir: Path | None = None) -> list[dict]:
        if not getattr(self, 'import_area_dbase', True):
            return []
        source_dir = Path(section_source_dir or Path(filepath).parent)
        expected_name = Path(filepath).stem
        matches = self._find_area_dbase_section_paths(source_dir, expected_name=expected_name)
        if not matches:
            return []
        results = []
        parent_empty = None
        for section_path in matches:
            try:
                area_dbase = AreaDBaseParser(section_path).parse()
                parent_empty = self._get_or_create_area_dbase_empty(collection, area_dbase, section_path)
                results.append(self._import_area_dbase_file(context, section_path, collection=collection, parent_empty=parent_empty))
            except Exception as exc:
                logger.warning('Failed to import AreaDBase section %s: %s', section_path.name, exc)
        return results

    def _build_level_metadata(self, collection, level):
        self._current_level_scene_center_offset = tuple(float(v) for v in getattr(level, 'scene_center_offset', (0.0, 0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0, 0.0))
        if len(self._current_level_scene_center_offset) < 4:
            self._current_level_scene_center_offset = tuple(self._current_level_scene_center_offset) + ((0.0,) * (4 - len(self._current_level_scene_center_offset)))
        root_obj = self._get_level_root_empty(collection)
        source_path = Path(str(getattr(level, 'filepath', '') or ''))
        if str(source_path):
            self._set_custom_property(root_obj, 'trlau_level_source_path', str(source_path))
            self._set_custom_property(root_obj, 'trlau_level_source_dir', str(source_path.parent))
        drm_name = str(root_obj.get('trlau_drm_name', getattr(collection, 'name', '') or '') or '').strip()
        self._build_unit_data_metadata(collection, level)
        self._build_admd_metadata(collection, level)
        self._build_reloc_module_metadata(collection, root_obj, level)
        source_game = str(getattr(level, 'source_game', '') or '').strip().lower()
        if source_game not in {'legend', 'anniversary'}:
            source_game = 'legend'
        self._set_custom_property(root_obj, 'trlau_level_game_id', source_game)
        try:
            root_obj.trlau_level_game = source_game
        except Exception:
            pass

        metadata = dict(getattr(level, 'level_metadata', {}) or {})
        for field_name in LEVEL_METADATA_FIELDS:
            if field_name in metadata:
                self._set_custom_property(root_obj, f'{LEVEL_METADATA_PROP_PREFIX}{field_name}', metadata[field_name])
        stream_unit_portals = list(getattr(level, 'stream_unit_portals', []) or [])
        if stream_unit_portals:
            try:
                self._set_custom_property(root_obj, 'trlau_stream_unit_portals_json', json.dumps(stream_unit_portals, separators=(',', ':')))
                self._set_custom_property(root_obj, 'trlau_stream_unit_portal_count', len(stream_unit_portals))
            except Exception:
                self._delete_custom_property(root_obj, 'trlau_stream_unit_portals_json')
                self._delete_custom_property(root_obj, 'trlau_stream_unit_portal_count')
        else:
            self._delete_custom_property(root_obj, 'trlau_stream_unit_portals_json')
            self._delete_custom_property(root_obj, 'trlau_stream_unit_portal_count')

        self._clear_terrain_light_grid_metadata(root_obj)

        terrain_lights = list(getattr(level, 'terrain_lights', [])) if self.import_terrain_lights else []
        if terrain_lights:
            print('TerrainLights found, importing...')
            light_parent = self._get_or_create_component_empty(collection, 'TerrainLight')
            for terrain_light in terrain_lights:
                light_type = int(getattr(terrain_light, 'type', 0))
                blender_light_type = 'SPOT' if light_type == 1 else 'SUN' if light_type == 2 else 'POINT'
                light_name = self._prefixed_name(drm_name, f'TerrainLight_{terrain_light.index:03d}')
                light_data = bpy.data.lights.new(light_name, type=blender_light_type)
                light_data.color = terrain_light.color
                light_data.energy = float(terrain_light.multiplier) * 100.0
                if hasattr(light_data, 'exposure'):
                    light_data.exposure = 10.0
                light_data.shadow_soft_size = max(0.0, float(max(terrain_light.radius, 0)) / 10.0)
                if blender_light_type == 'SPOT':
                    light_data.spot_size = max(0.0174533, math.radians(max(int(terrain_light.falloff_angle), 1)))
                    light_data.spot_blend = 0.15
                light_obj = bpy.data.objects.new(light_name, light_data)
                light_obj.location = tuple(float(v) for v in terrain_light.position)
                self._set_custom_property(light_obj, 'trlau_type', 'TerrainLight')
                self._set_custom_property(light_obj, 'trlau_terrain_light_index', int(terrain_light.index))
                self._set_custom_property(light_obj, 'trlau_terrain_light_id', int(terrain_light.light_id))
                self._set_custom_property(light_obj, 'trlau_terrain_light_type', int(terrain_light.type))
                self._set_custom_property(light_obj, 'trlau_terrain_light_radius', int(terrain_light.radius))
                self._set_custom_property(light_obj, 'trlau_terrain_light_multiplier', int(terrain_light.multiplier))
                self._set_custom_property(light_obj, 'trlau_terrain_light_hotspot_angle', int(terrain_light.hotspot_angle))
                self._set_custom_property(light_obj, 'trlau_terrain_light_falloff_angle', int(terrain_light.falloff_angle))
                self._set_custom_property(light_obj, 'trlau_terrain_light_direction', tuple(float(v) for v in getattr(terrain_light, 'direction', (0.0, 0.0, -1.0))))
                self._link_object_to_collection(light_obj, collection)
                light_obj.parent = light_parent
                light_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                direction = tuple(float(v) for v in getattr(terrain_light, 'direction', (0.0, 0.0, -1.0)))
                self._orient_object_towards_target(light_obj, (light_obj.location.x + direction[0], light_obj.location.y + direction[1], light_obj.location.z + direction[2]))

        if self.import_camera_data:
            camera_keys = list(getattr(level, 'camera_data', []))
            if camera_keys:
                print('CameraData found, importing...')
                camera_parent = self._get_or_create_component_empty(collection, 'CameraData')
                for camera_key in camera_keys:
                    camera_name = self._prefixed_name(drm_name, f'CameraKey_{camera_key.index:03d}_{camera_key.camera_id}')
                    camera_data = bpy.data.cameras.new(camera_name)
                    camera_obj = bpy.data.objects.new(camera_name, camera_data)
                    camera_obj.location = tuple(float(v) for v in camera_key.position)
                    camera_obj.scale = (100.0, 100.0, 100.0)
                    self._set_custom_property(camera_obj, 'trlau_type', 'CameraData')
                    self._set_custom_property(camera_obj, 'trlau_camera_id', int(camera_key.camera_id))
                    self._set_custom_property(camera_obj, 'trlau_camera_flags', int(camera_key.flags))
                    self._set_custom_property(camera_obj, 'trlau_camera_rx', int(camera_key.rotation[0]))
                    self._set_custom_property(camera_obj, 'trlau_camera_ry', int(camera_key.rotation[1]))
                    self._set_custom_property(camera_obj, 'trlau_camera_rz', int(camera_key.rotation[2]))
                    self._set_custom_property(camera_obj, 'trlau_camera_tx', int(camera_key.target[0]))
                    self._set_custom_property(camera_obj, 'trlau_camera_ty', int(camera_key.target[1]))
                    self._set_custom_property(camera_obj, 'trlau_camera_tz', int(camera_key.target[2]))
                    self._link_object_to_collection(camera_obj, collection)
                    camera_obj.parent = camera_parent
                    camera_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                    self._orient_object_towards_target(camera_obj, (
                        camera_obj.location.x + float(camera_key.target[0]),
                        camera_obj.location.y + float(camera_key.target[1]),
                        camera_obj.location.z + float(camera_key.target[2]),
                    ))

        if getattr(self, 'import_hinfo', True) and getattr(self, 'import_markups', True):
            markups = list(getattr(level, 'markups', []))
            if markups:
                print('Markups found, importing...')
                markup_parent = self._get_or_create_component_empty(collection, 'Markup')
                for markup in markups:
                    polyline = list(getattr(markup, 'polyline', []) or [])
                    curve_name = self._prefixed_name(drm_name, f'Markup_{int(getattr(markup, "index", 0)):03d}')
                    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
                    curve_data.dimensions = '3D'
                    curve_data.resolution_u = 1
                    markup_position = tuple(float(v) for v in getattr(markup, 'position', (0.0, 0.0, 0.0)))
                    markup_flags = int(getattr(markup, 'flags', 0) or 0)
                    bbox_used_for_shape = False
                    if markup_flags & MARKUP_BBOX_FLAGS:
                        bbox_used_for_shape = _add_markup_bbox_wire(
                            curve_data,
                            getattr(markup, 'bbox', ()),
                            markup_position,
                            level_space=True,
                            bbox_is_local=True,
                        )
                    if polyline and not bbox_used_for_shape:
                        spline = curve_data.splines.new('POLY')
                        if len(polyline) > 1:
                            spline.points.add(len(polyline) - 1)
                        base_x, base_y, base_z = markup_position
                        for point_index, point in enumerate(polyline):
                            px, py, pz = (float(point[0]), float(point[1]), float(point[2]))
                            pw = float(point[3]) if len(point) > 3 else 1.0
                            spline.points[point_index].co = (px - base_x, py - base_y, pz - base_z, pw)
                    curve_obj = bpy.data.objects.new(curve_name, curve_data)
                    curve_obj.location = markup_position
                    curve_obj.rotation_euler = (0.0, 0.0, 0.0)
                    curve_obj.scale = (1.0, 1.0, 1.0)
                    self._set_custom_property(curve_obj, 'trlau_type', 'Markup')
                    self._set_custom_property(curve_obj, 'trlau_markup_index', int(getattr(markup, 'index', -1)))
                    self._set_custom_property(curve_obj, 'trlau_markup_game', str(getattr(markup, 'game', getattr(level, 'source_game', 'unknown'))))
                    self._set_custom_property(curve_obj, 'trlau_markup_flags', markup_flags)
                    self._set_custom_property(curve_obj, 'trlau_markup_point_count', len(polyline))
                    if bbox_used_for_shape:
                        self._set_custom_property(curve_obj, 'trlau_markup_bbox_shape', True)
                    self._set_custom_property(curve_obj, 'trlau_markup_animated_segment', int(getattr(markup, 'animated_segment', 0)))
                    try:
                        curve_obj.trlau_markup_data.intro_id = int(getattr(markup, 'intro_id', 0) or 0)
                    except Exception:
                        pass
                    self._link_object_to_collection(curve_obj, collection)
                    curve_obj.parent = markup_parent
                    curve_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)

        return root_obj

    @staticmethod
    def _remove_custom_properties(obj, *, exact: set[str] | None = None, prefixes: tuple[str, ...] = ()) -> None:
        exact = set(exact or set())
        try:
            keys = list(obj.keys())
        except Exception:
            return
        for key in keys:
            key_s = str(key)
            if key_s in exact or any(key_s.startswith(prefix) for prefix in prefixes):
                try:
                    del obj[key]
                except Exception:
                    pass

    def _clean_level_component_custom_properties(self, collection, root_obj=None) -> None:
        """Remove import-only selector/cache properties from level scene objects.

        Export should derive component identity from the normal import hierarchy,
        object names, transforms, materials and geometry.  Keep custom properties
        only for values that are not represented in Blender scene data.
        """
        objects = list(getattr(collection, 'all_objects', None) or collection.objects)
        for obj in objects:
            try:
                name = getattr(obj, 'name', '') or ''
                parent_name = getattr(getattr(obj, 'parent', None), 'name', '') or ''
                is_component_empty = getattr(obj, 'type', None) == 'EMPTY' and any(name.endswith(f'_{suffix}') or name == suffix for suffix in (
                    'Terrain', 'BGInstance', 'Markup', 'IntroData', 'CameraData', 'TerrainLight', 'UnitData', 'ADMDData', 'AreaDBase'
                ))
                is_bg = _BGOBJECT_NAME_RE.search(name) is not None or _BGINSTANCE_NAME_RE.search(name) is not None
                is_markup = _MARKUP_NAME_RE.search(name) is not None or str(obj.get('trlau_type', '') or '') == 'Markup'
                try:
                    intro_data = trlau_intro_data_to_dict(obj)
                except Exception:
                    intro_data = {}
                is_intro = _INTRO_NAME_RE.search(name) is not None or bool(intro_data) or str(obj.get('trlau_type', '') or '') == 'IntroData'
                is_area = 'AreaDBase' in name or str(obj.get('trlau_type', '') or '').startswith('AreaDBase')
                is_terrain_group_empty = (
                    getattr(obj, 'type', None) == 'EMPTY'
                    and (_TERRAIN_GROUP_NAME_RE.search(name) is not None or bool(obj.get('trlau_terrain_group')) or str(obj.get('trlau_type', '') or '') == 'TerrainGroup')
                )
                is_terrain_group_mesh = (
                    getattr(obj, 'type', None) == 'MESH'
                    and (_TERRAIN_GROUP_MESH_NAME_RE.search(name) is not None or bool(obj.get('trlau_terrain_mesh')))
                )
                is_terrain_collision = (
                    _TERRAIN_GROUP_COLLISION_NAME_RE.search(name) is not None
                    or bool(obj.get('trlau_terrain_collision'))
                    or bool(obj.get('trlau_terrain_collision_empty'))
                )

                exact = {
                    'trlau_level', 'trlau_type', 'trlau_component_empty', 'trlau_component_name',
                    'trlau_export_ignore',
                }
                prefixes: list[str] = []
                if is_component_empty:
                    exact.update({'trlau_unitdata_empty', 'trlau_admd_empty'})
                if is_bg:
                    exact.update({
                        'trlau_bgobject', 'trlau_bgobject_empty', 'trlau_bgobject_mesh', 'trlau_bginstance', 'trlau_bginstance_empty',
                        'trlau_bgobject_index', 'trlau_bginstance_index', 'trlau_bginstance_bgobject_index', 'trlau_bginstance_bgobject_offset',
                        'trlau_bgobject_position', 'trlau_bgobject_scale',
                        'trlau_bgobject_position_x', 'trlau_bgobject_position_y', 'trlau_bgobject_position_z',
                        'trlau_bgobject_scale_x', 'trlau_bgobject_scale_y', 'trlau_bgobject_scale_z',
                        'trlau_bgobject_stride', 'trlau_bgobject_vertex_count', 'trlau_vertex_color_count',
                        'trlau_bgobject_color_data_index', 'trlau_bgobject_color_slot_count',
                        'trlau_bgobject_source_indices', 'trlau_bgobject_slot_count',
                        'trlau_level_vertex_buffer', 'trlau_vertex_base_offset', 'trlau_bgobject_strip_index',
                        'trlau_bgobject_strip_sort_vertex_x', 'trlau_bgobject_strip_sort_vertex_y', 'trlau_bgobject_strip_sort_vertex_z',
                        'trlau_bgobject_strip_sort_push', 'trlau_bgobject_strip_scroll_offset', 'trlau_bgobject_strip_raw_count',
                    })
                    prefixes.extend(('trlau_bgobject_blob_', 'trlau_bgobject_slot_', 'trlau_bgobject_position_candidate_', 'trlau_bginstance_m'))
                    exact.update({'trlau_bgobject_blob_encoding', 'trlau_bgobject_blob_chunk_count', 'trlau_bgobject_blob_size'})
                if is_terrain_group_empty:
                    # Keep only fields that are edited or not reliably derivable.
                    exact.update({
                        'trlau_terrain_group', 'trlau_drawgroup', 'trlau_terrain_group_index',
                        'trlau_terrain_group_global_offset', 'trlau_terrain_group_local_offset', 'trlau_terrain_group_origin',
                        'trlau_terrain_group_sorted_material_indices',
                    })
                if is_terrain_group_mesh:
                    exact.update({
                        'trlau_terrain_mesh', 'trlau_drawgroup', 'trlau_terrain_group_index',
                        'trlau_group_position_x', 'trlau_group_position_y', 'trlau_group_position_z',
                        'trlau_terrain_group_flags', 'trlau_terrain_group_id', 'trlau_terrain_group_unique_id',
                        'trlau_terrain_group_spline_id', 'trlau_terrain_group_global_offset',
                        'trlau_terrain_group_local_offset', 'trlau_terrain_group_origin',
                        'trlau_terrain_group_texture_morph_value', 'trlau_terrain_group_texture_morph_step',
                        'trlau_strip_slot_count', 'trlau_strip_flags',
                        'trlau_level_vertex_buffer', 'trlau_vertex_base_offset',
                    })
                    prefixes.extend(('trlau_strip_slot_',))
                if is_terrain_collision:
                    exact.update({
                        'trlau_terrain_collision', 'trlau_terrain_collision_empty', 'trlau_terrain_group_index',
                        'trlau_terrain_collision_position', 'trlau_terrain_collision_bbox_min', 'trlau_terrain_collision_bbox_max',
                    })
                if is_markup:
                    exact.update({'trlau_markup_index', 'trlau_markup_game', 'trlau_markup_point_count', 'trlau_markup_bbox_shape'})
                if is_area:
                    prefixes.extend(('trlau_area_dbase_',))
                if parent_name.endswith('_IntroData') or parent_name == 'IntroData' or '_Intro_' in parent_name or parent_name.startswith('Intro_'):
                    exact.update({'trlau_level', 'trlau_type'})
                self._remove_custom_properties(obj, exact=exact, prefixes=tuple(prefixes))
            except Exception:
                continue

    def _import_intro_data(self, context, filepath: str, collection, level, section_source_dir: Path | None = None) -> list[dict]:
        if not self.import_intro_data:
            return []
        intro_entries = list(getattr(level, 'intro_data', []))
        if not intro_entries:
            return []

        objectlist, objectlist_dir = self._load_level_objectlist_mapping(filepath, level)
        if not objectlist:
            logger.warning('IntroData found in %s but objectlist.txt was not found in the expected game object-list folder; importing placeholders only', Path(filepath).name)

        print('IntroData found, importing...')
        drm_name = Path(filepath).stem
        intro_parent = self._get_or_create_component_empty(collection, 'IntroData')
        player_intro_object_id = self._player_intro_object_id(level)
        player_object_id = int(getattr(level, 'player_object_id', -1))
        imported_results = []
        for intro in intro_entries:
            object_id = int(getattr(intro, 'object_id', -1))
            mapped_name = objectlist.get(object_id) if objectlist else None
            object_name = mapped_name or (f'Object_{object_id:03d}' if object_id >= 0 else f'Object_{abs(object_id):03d}')
            import_object_id = object_id
            import_mapped_name = mapped_name
            imports_level_player_model = False

            if object_id == player_intro_object_id:
                imports_level_player_model = True
                import_object_id = player_object_id
                import_mapped_name = objectlist.get(player_object_id) if objectlist and player_object_id >= 0 else None
                if import_mapped_name:
                    object_name = import_mapped_name

            intro_empty = self._get_or_create_intro_empty(collection, drm_name, intro, object_name)
            intro_empty.parent = intro_parent
            intro_empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            self._apply_intro_transform({}, intro, parent_empty=intro_empty)
            self._import_intro_sound_metadata(collection, section_source_dir or Path(filepath).parent, intro_empty, intro)
            if imports_level_player_model:
                trlau_set_intro_data_property(intro_empty, uses_player_object_id=True, player_object_id=player_object_id)
            else:
                trlau_set_intro_data_property(intro_empty, uses_player_object_id=False, player_object_id=-1)

            if not import_mapped_name:
                if imports_level_player_model:
                    logger.warning(
                        'Player IntroData object id %s uses playerObjectID %s, but no objectlist.txt entry was found for playerObjectID; importing placeholder empty only',
                        object_id,
                        player_object_id,
                    )
                else:
                    logger.warning('No objectlist.txt entry for IntroData object id %s; importing placeholder empty only', object_id)
                continue

            drm_path = self._find_level_named_drm(filepath, objectlist_dir, import_mapped_name)
            if drm_path is None:
                if imports_level_player_model:
                    logger.warning(
                        'Could not find player IntroData DRM for intro object %s using playerObjectID %s (%s); keeping placeholder empty',
                        object_id,
                        import_object_id,
                        import_mapped_name,
                    )
                else:
                    logger.warning('Could not find intro DRM for object %s (%s); keeping placeholder empty', object_id, import_mapped_name)
                continue

            try:
                imported = self._import_drm_first_model(context, drm_path, collection, include_hinfo=False, include_targets=False)
            except Exception as exc:
                logger.warning(
                    'Failed to import IntroData object %s using object %s (%s) from %s; keeping placeholder empty: %s',
                    object_id,
                    import_object_id,
                    import_mapped_name,
                    drm_path.name,
                    exc,
                )
                continue
            self._import_intro_first_animation(context, drm_path, imported, intro_empty, action_prefix=getattr(intro_empty, 'name', import_mapped_name))
            for result in imported:
                self._apply_intro_transform(result, intro, parent_empty=intro_empty)
                if imports_level_player_model:
                    for obj in self._root_objects_from_result(result):
                        try:
                            trlau_set_intro_data_property(obj, uses_player_object_id=True, player_object_id=player_object_id)
                        except Exception:
                            pass
            imported_results.extend(imported)
        return imported_results

    def _import_level_file(self, context, filepath: str, collection=None, drm_game: str | None = None):
        with ExitStack() as cleanup_stack:
            source_path = Path(filepath)
            section_source_dir = source_path.parent
            if source_path.suffix.lower() == '.drm':
                try:
                    section_parser = DRMContainerParser(
                        filepath,
                        decompress_derickw=(str(getattr(self, 'platform', '')).upper() == 'PSP'),
                    )
                    section_source_dir, _extracted_paths, _sections = cleanup_stack.enter_context(
                        section_parser.temporary_extract_sections()
                    )
                except Exception as exc:
                    logger.warning('Failed to extract DRM sections for section-id metadata import from %s: %s', source_path.name, exc)
            collection = collection or self._get_or_create_collection(context, filepath)
            parser_cls = TRPSPLevelParser if str(getattr(self, 'platform', '')).upper() == 'PSP' else TRLevelParser
            level = parser_cls(
                filepath,
                import_textures=self.import_textures,
                import_bgobjects=self.import_bgobjects,
                import_collisions=getattr(self, 'import_collisions', True),
                import_markups=getattr(self, 'import_markups', True),
                import_signals=getattr(self, 'import_signals', True),
                game_hint=drm_game,
            ).parse()
            builder = TRLevelBuilder(
                context,
                filepath,
                collection=collection,
                import_textures=self.import_textures,
                import_bgobjects=self.import_bgobjects,
                import_collisions=getattr(self, 'import_collisions', True),
                import_signals=getattr(self, 'import_signals', True),
                import_kdnodes=getattr(self, 'import_kdnodes', False),
                split_terrain_groups_by_strip=getattr(self, 'split_terrain_groups_by_strip', False),
            )
            builder.import_terrain_lights = self.import_terrain_lights
            mesh_objects = builder.build(level)
            root_obj = self._build_level_metadata(collection, level)
            self._build_section_reference_metadata(collection, section_source_dir, level)
            audio_objects = self._import_level_sfx_markers(collection, section_source_dir, level)
            area_dbase_results = self._import_area_dbase_from_source_dir(context, filepath, collection, section_source_dir)

            results = [{
                'model': level,
                'mesh_obj': mesh_objects[0] if mesh_objects else None,
                'mesh_objects': mesh_objects,
                'arm_obj': None,
                'audio_objects': audio_objects,
                'faces': (
                    sum(len(getattr(group_strip, 'indices', [])) // 3 for group in level.terrain_groups for group_strip in group.strips)
                    + sum(len(getattr(bg_strip, 'indices', [])) // 3 for bg in getattr(level, 'bg_objects', []) for bg_strip in bg.strips)
                ),
                'source': str(source_path),
            }]
            results.extend(area_dbase_results)
            results.extend(self._import_intro_data(context, filepath, collection, level, section_source_dir))
            self._clean_level_component_custom_properties(collection, root_obj)
            print('')
            return results

_TRLAU_AUDIO_DEVICE = None
_TRLAU_AUDIO_HANDLE = None
_TRLAU_AUDIO_TEMP_DIR: str | None = None


def _selected_sfx_speaker_object(context):
    obj = getattr(context, 'object', None)
    if obj is None:
        return None
    if getattr(obj, 'type', '') == 'SPEAKER':
        return obj
    stack = list(getattr(obj, 'children', []) or [])
    while stack:
        child = stack.pop(0)
        if getattr(child, 'type', '') == 'SPEAKER':
            return child
        stack.extend(list(getattr(child, 'children', []) or []))
    return None


def _sound_filepath_for_playback(sound) -> str:
    if sound is None:
        return ''
    for attr in ('filepath', 'filepath_raw'):
        try:
            path = str(getattr(sound, attr, '') or '')
        except Exception:
            path = ''
        if path:
            try:
                abs_path = bpy.path.abspath(path)
            except Exception:
                abs_path = path
            if abs_path and Path(abs_path).exists():
                return abs_path
    try:
        packed = getattr(sound, 'packed_file', None)
        data = getattr(packed, 'data', None) if packed is not None else None
        if data:
            global _TRLAU_AUDIO_TEMP_DIR
            if not _TRLAU_AUDIO_TEMP_DIR:
                _TRLAU_AUDIO_TEMP_DIR = tempfile.mkdtemp(prefix='trlau_editor_audio_play_')
            safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(getattr(sound, 'name', 'sound') or 'sound'))
            out_path = Path(_TRLAU_AUDIO_TEMP_DIR) / f'{safe_name}.wav'
            out_path.write_bytes(bytes(data))
            return str(out_path)
    except Exception:
        pass
    return ''


class TRLAU_OT_play_sfx_sound(bpy.types.Operator):
    bl_idname = 'trlau.play_sfx_sound'
    bl_label = 'Play SFX Sound'
    bl_description = 'Play the selected imported TRLAU SFX speaker, or the first speaker below the selected SFX entry'
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _selected_sfx_speaker_object(context) is not None

    def execute(self, context):
        global _TRLAU_AUDIO_DEVICE, _TRLAU_AUDIO_HANDLE
        speaker_obj = _selected_sfx_speaker_object(context)
        if speaker_obj is None or getattr(speaker_obj, 'type', '') != 'SPEAKER':
            self.report({'WARNING'}, 'Select an imported SFX speaker or SFX entry with a child speaker')
            return {'CANCELLED'}
        speaker_data = getattr(speaker_obj, 'data', None)
        sound = getattr(speaker_data, 'sound', None)
        path = str(getattr(speaker_obj, 'trlau_wave_decoded_wav_path', '') or speaker_obj.get('trlau_wave_decoded_wav_path', '') or '')
        if not path or not Path(path).exists():
            path = _sound_filepath_for_playback(sound)
        if aud is None:
            self.report({'ERROR'}, 'Blender aud module is unavailable; use timeline playback on the Speaker instead')
            return {'CANCELLED'}
        if not path or not Path(path).exists():
            self.report({'ERROR'}, 'No decoded WAV file is available for this speaker')
            return {'CANCELLED'}
        try:
            if _TRLAU_AUDIO_HANDLE is not None:
                try:
                    _TRLAU_AUDIO_HANDLE.stop()
                except Exception:
                    pass
            if _TRLAU_AUDIO_DEVICE is None:
                _TRLAU_AUDIO_DEVICE = aud.Device()
            handle = _TRLAU_AUDIO_DEVICE.play(aud.Sound(path))
            try:
                handle.volume = max(0.0, float(getattr(speaker_data, 'volume', 1.0) or 1.0))
            except Exception:
                pass
            try:
                handle.pitch = max(0.01, float(getattr(speaker_data, 'pitch', 1.0) or 1.0))
            except Exception:
                pass
            _TRLAU_AUDIO_HANDLE = handle
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to play SFX sound: {exc}')
            return {'CANCELLED'}


class TRLAU_OT_stop_sfx_sound(bpy.types.Operator):
    bl_idname = 'trlau.stop_sfx_sound'
    bl_label = 'Stop SFX Sound'
    bl_description = 'Stop the current TRLAU SFX preview sound'
    bl_options = {'REGISTER'}

    def execute(self, _context):
        global _TRLAU_AUDIO_HANDLE
        if _TRLAU_AUDIO_HANDLE is not None:
            try:
                _TRLAU_AUDIO_HANDLE.stop()
            except Exception:
                pass
            _TRLAU_AUDIO_HANDLE = None
        return {'FINISHED'}

