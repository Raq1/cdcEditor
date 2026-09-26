from __future__ import annotations

import base64
import json
import math
import re
import struct
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import List, Optional, Tuple

import bpy
import mathutils
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Collection, Operator
from bpy_extras.io_utils import ExportHelper

from ..core.game_utils import normalize_game_value
from ..core.log import configure_logging, logger
from ..core.object_utils import is_descendant_of
from ..core.material_ui import decode_tpage_flags, get_material_tpageid, get_material_scroll_enabled, get_material_scroll_speed, get_material_env_mapping, get_material_eye_ref_env_mapping
from ..core.texture_export import clear_texture_export_cache, image_to_pcd_bytes
from ..core.fsfx_render_effect import build_fsfx_render_effect_content_from_object, has_fsfx_render_effect_properties
from ..core.cine_format import build_cine_payload_from_metadata
from ..core.level_combat import decode_combat_graph, patch_combat_graph
from ..platforms.pc.tr7ae_model_export import TRLAUModelExporter
from ..platforms.ps2.tr7ae_model_export import TRLAUPS2ModelExporter
from ..platforms.pc.area_dbase import (
    build_area_dbase_section_content_from_polygons,
    convert_area_dbase_center_from_blender_position,
)
from .animation_export import TRLAUAnimationExporter
from ..ui.level_panel import (
    trlau_multispline_property_to_dict,
    trlau_intro_data_to_dict,
    trlau_intro_generic_data_to_dict,
    trlau_intro_specific_data_to_dict,
    trlau_intro_specific_data_type,
    trlau_intro_specific_struct_name,
    trlau_intro_sound_data_to_dict,
    trlau_intro_is_intro_data,
    trlau_intro_is_generic_data,
    trlau_intro_is_specific_data,
    trlau_intro_is_sound_data,
    SFX_SOUND_PERIODIC_FIELDS,
    SFX_SOUND_EVENT_FIELDS,
    SFX_SOUND_STREAM_FIELDS,
    trlau_level_combat_property_to_dict,
)
from ..platforms.pc.tr7ae_level_export import (
    ExportCollision,
    ExportCollisionFace,
    ExportCollisionKDNode,
    ExportSignal,
    ExportSignalFace,
    ExportSignalMesh,
    ExportBGInstance,
    ExportBGObject,
    ExportIntroData,
    ExportLevel,
    ExportSFXMarker,
    ExportSFXPerimeter,
    ExportSFXSound,
    ExportMarkupData,
    ExportPassthroughSection,
    ExportStrip,
    ExportTerrainGroup,
    ExportTerrainLightData,
    ExportVertex,
    TRLevelDRMWriter,
)

_LEVEL_SUFFIX_RE = re.compile(r'^(?P<prefix>.+?)_Level$')
_GROUP_NAME_RE = re.compile(r'TerrainGroup_(\d+)$')
_STRIP_NAME_RE = re.compile(r'_Strip_(\d+)$')
_BGOBJECT_NAME_RE = re.compile(r'(?:^|_)BGObject_(\d+)(?:$|_|\.)')
_BGINSTANCE_NAME_RE = re.compile(r'(?:^|_)BGObject_(\d+)_Instance_(\d+)(?:$|_|\.)')
_MARKUP_NAME_RE = re.compile(r'(?:^|_)Markup_(\d+)(?:$|_|\.)')
_INTRO_NAME_RE = re.compile(r'(?:^|_)Intro_(\d+)(?:_|$)')
_COLLISION_NAME_BLENDER_SUFFIX_RE = re.compile(r'\.\d{3}$')
_COLLISION_CLIENT_FLAG_RE = re.compile(r'(?:^|[_\s-])ClientFlag[_\s-]*(Wall|Ground|Slope|Water|Snow|\d+)(?:$|[._\s-])', re.IGNORECASE)
_COLLISION_CLIENT_FLAG_NAMES = {
    'wall': 0,
    'ground': 32,
    'slope': 4,
    'water': 1,
    'water1': 16,
    'water2': 17,
    'snow': 48,
}
_COLLECTION_ENUM_CACHE: list[tuple[str, str, str]] = []
_ACTION_ENUM_CACHE: list[tuple[str, str, str]] = []
_ARMATURE_ENUM_CACHE: list[tuple[str, str, str]] = []


SECTION_BLOB_PROP_PREFIX = 'trlau_section_blob_'
TERRAIN_LIGHT_GRID_BLOB_PROP_PREFIX = 'trlau_terrain_light_grid_blob_'
BGOBJECT_BLOB_PROP_PREFIX = 'trlau_bgobject_blob_'
COMBAT_BLOB_PROP_PREFIX = 'trlau_combat_blob_'
MARKUP_FLAG_PERCH = 262144
MARKUP_FLAG_WATER = 2147483648
MARKUP_BBOX_FLAGS = MARKUP_FLAG_PERCH | MARKUP_FLAG_WATER

_SFX_STREAM_STRUCT = '<IBBBBIBbBxfHHfffIb'
_SFX_STREAM_SIZE = struct.calcsize(_SFX_STREAM_STRUCT)
_SFX_NAME_POINTER_OFFSET = 4
_SFX_DEFAULT_NAME_OFFSET = 4 + ((_SFX_STREAM_SIZE + 3) & ~3)

INTRO_GENERIC_FIELDS = ('in_view_remove_dist', 'out_of_view_remove_dist', 'use_model', 'pad0', 'pad1', 'flags', 'attached_instance', 'swing_length', 'dtp_camera_id')
INTRO_SPECIFIC_FIELDS_BY_TYPE = {
    17: ('reward_type', 'unique_id', 'sound_id'),
    12: ('top_connect_type', 'bottom_connect_type', 'top_connect_instance', 'top_connect_model_index', 'top_connect_model_marker_index', 'bottom_connect_instance', 'bottom_connect_model_index', 'bottom_connect_model_marker_index', 'collision_plane_instance0', 'collision_plane_instance1', 'rope_camera_dtpid', 'rope_camera_overrides_movement', 'sound_input_min', 'sound_input_max', 'render_width', 'render_length_per_v', 'render_u_width', 'render_color', 'render_model', 'render_texture_main', 'render_texture_top', 'render_texture_bottom'),
    13: ('water_depth', 'water_inflow', 'water_outflow', 'water_speed', 'flow_radius', 'bob_height', 'bob_frequency', 'bob_grid_size', 'priority', 'water_flags'),
}

_UV_EXPORT_SCALE = 4096.0
_UV_RAW_MIN = -32768
_UV_RAW_MAX = 32767
_UV_TILE_UNITS = 4096
_UV_EXPORT_EPSILON = 1e-6

_SIGNAL_FLAG_FIELD_NAMES = (
    'exitPortal', 'entryPortal', 'autoStream', 'ResetFSFXToDefaultOut',
    'ResetFSFXToDefaultIn', 'MaterialOnly', 'SpectralOnly', 'ObjectHitSignal',
    'ResetSlideAngle', 'triggered',
)
_SIGNAL_VALUE_FIELD_NAMES = (
    'raw_flags', 'CameraLock', 'CameraUnlock', 'CameraSave', 'CameraRestore',
    'CameraShakeScale', 'CameraShakeTime', 'streamName', 'streamPortalIndex',
    'SetSlideAngle', 'ReverbType', 'ReverbVolume', 'setMusicVar',
    'setMusicVarValue', 'fatalType', 'CameraStackType', 'CameraStackData',
    'CameraStackCamInUse', 'CameraStackSignalFlags0', 'CameraStackSignalFlags1',
    'DTPCameraID0', 'DTPCameraID1', 'cameraLinkSignalIndex', 'FSFXInDTPID',
    'FSFXExitDTPID', 'FSFXInActiveDist', 'FSFXExitActiveDist',
    'startGoingIntoWaterSignal', 'startGoingOutOfWaterSignal',
)
_SIGNAL_SPLINE_FIELD_NAMES = (
    'CameraFollowSmooth', 'CameraFollowTilt', 'CameraFollowDistance',
    'CameraFollowRotation', 'CameraZOffset', 'CameraCutAngle',
    'CameraCombatMinDistance', 'CameraCombatDistance', 'CameraCombatZOffset',
    'CameraCombatLockedOut', 'CameraDisableLookaroundFlag', 'CameraSpline0',
    'CameraSpline1', 'CameraSpline2', 'CameraAlternateSpline0',
    'CameraAlternateSpline1', 'CameraAlternateSpline2', 'CameraMode',
    'CameraLeadAmount', 'CameraSpline0Width', 'CameraSpline1Width',
    'CameraSpline0WidthZ', 'CameraSpline1WidthZ', 'CameraZoomDist',
    'CameraVelocity', 'CameraDampening', 'CameraTargetVelocity',
    'CameraCombatFraming', 'CameraTargetDampening', 'CameraCutFlag',
    'CameraFollowCamOverrideEnabled', 'CameraCrossfade', 'CameraInterestInstId',
    'CameraInterestTune', 'CameraSwitchToFollowDist', 'CameraFollowVerticalBias',
    'CameraFollowHighTiltDistance', 'CameraFollowHighTiltAngle',
    'CameraFollowMedTiltDistance', 'CameraFollowMedTiltAngle',
    'CameraFollowZeroTiltDistance', 'CameraFollowZeroTiltAngle',
    'CameraFollowLowTiltDistance', 'CameraFollowLowTiltAngle',
)


def _find_level_root_in_collection(collection: Collection):
    objects = getattr(collection, 'all_objects', None) or collection.objects
    for obj in objects:
        if obj.type == 'EMPTY' and bool(obj.get('trlau_level_root')):
            return obj
    for obj in objects:
        if obj.type != 'EMPTY':
            continue
        if _LEVEL_SUFFIX_RE.match(obj.name):
            return obj
    return None


def _enum_collection_items(_self=None, context=None):
    global _COLLECTION_ENUM_CACHE
    items: list[tuple[str, str, str]] = []
    for collection in bpy.data.collections:
        if collection.name.startswith('Master Collection'):
            continue
        has_root = _find_level_root_in_collection(collection) is not None
        has_model = (not has_root) and TRLAUModelExporter.collection_has_models(collection)
        if not (has_root or has_model):
            continue
        suffix = ' [Level]' if has_root else ' [Model]'
        label = f'{collection.name}{suffix}'
        description = 'Export level DRM' if has_root else 'Export model/object section'
        items.append((collection.name, label, f'{description}: {collection.name}'))
    _COLLECTION_ENUM_CACHE = items or [('__TRLAU_NO_EXPORT_COLLECTION__', 'No Model/Level collections', 'No collection contains a TRLAU Model or Level empty')]
    return _COLLECTION_ENUM_CACHE


def _enum_animation_action_items(_self=None, context=None):
    global _ACTION_ENUM_CACHE
    items: list[tuple[str, str, str]] = []
    active_armature = _active_armature(context) if context is not None else None
    active_action = getattr(getattr(active_armature, 'animation_data', None), 'action', None) if active_armature is not None else None

    if active_action is not None:
        items.append((active_action.name, active_action.name, f'Export action: {active_action.name}'))

    for action in bpy.data.actions:
        if active_action is not None and action.name == active_action.name:
            continue
        items.append((action.name, action.name, f'Export action: {action.name}'))

    _ACTION_ENUM_CACHE = items or [('__TRLAU_NO_ACTION__', 'No actions available', 'No Blender actions are available to export')]
    return _ACTION_ENUM_CACHE



def _enum_animation_armature_items(_self=None, context=None):
    global _ARMATURE_ENUM_CACHE
    items: list[tuple[str, str, str]] = []
    active_armature = _active_armature(context) if context is not None else None

    if active_armature is not None:
        items.append((active_armature.name, active_armature.name, f'Export using armature: {active_armature.name}'))

    for obj in bpy.data.objects:
        if getattr(obj, 'type', None) != 'ARMATURE':
            continue
        if active_armature is not None and obj.name == active_armature.name:
            continue
        items.append((obj.name, obj.name, f'Export using armature: {obj.name}'))

    _ARMATURE_ENUM_CACHE = items or [('__TRLAU_NO_ARMATURE__', 'No armatures available', 'No armature is available to export animation')]
    return _ARMATURE_ENUM_CACHE


def _trlau_filepath_suffix(filepath: str) -> str:
    text = str(filepath or '').replace('\\', '/')
    name = text.rsplit('/', 1)[-1]
    dot = name.rfind('.')
    if dot <= 0:
        return ''
    return name[dot:].lower()


def _trlau_replace_filepath_suffix(filepath: str, desired_ext: str) -> str:
    text = str(filepath or '')
    desired_ext = str(desired_ext or '')
    if desired_ext and not desired_ext.startswith('.'):
        desired_ext = '.' + desired_ext

    slash = max(text.rfind('/'), text.rfind('\\'))
    dot = text.rfind('.')
    if dot > slash:
        text = text[:dot]
    return bpy.path.ensure_ext(text, desired_ext)


def _trlau_filepath_name(filepath: str) -> str:
    text = str(filepath or '').replace('\\', '/').rstrip('/')
    return text.rsplit('/', 1)[-1] if text else ''


def _is_animation_export_filepath(filepath: str) -> bool:
    return _trlau_filepath_suffix(filepath) == '.ani'


def _active_armature(context):
    obj = getattr(context, 'object', None) or getattr(context, 'active_object', None)
    if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
        return obj
    return None


class EXPORT_SCENE_OT_trlau_level(Operator, ExportHelper):

    bl_idname = 'export_scene.trlau_level'
    bl_label = 'Tomb Raider LAU Export'
    bl_description = '\u00A0'
    bl_options = {'UNDO'}

    filename_ext = '.drm'
    filter_glob: StringProperty(default='*.drm;*.obj;*.tr7aemesh;*.ani', options={'HIDDEN'})

    collection_name: EnumProperty(
        name='Mesh Collection',
        description='Collection to export. Level collections export .drm; regular model collections can export to .drm or back to an extracted .obj section.',
        items=_enum_collection_items,
    )
    export_textures: BoolProperty(
        name='Export Textures',
        default=True,
        description='Export Image Textures binded to the model materials',
    )
    export_cloth: BoolProperty(
        name='Export Cloth',
        default=True,
        description='Export ClothSetup binded to the model',
    )
    export_area_dbase: BoolProperty(
        name='Export AreaDBase',
        default=True,
        description='Generate AreaDBase planner data WHAT ARE WE DOING',
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    animation_armature_name: EnumProperty(
        name='Armature',
        description='Armature to use when exporting animation',
        items=_enum_animation_armature_items,
    )
    animation_action_name: EnumProperty(
        name='Action',
        description='Action to export when exporting animation',
        items=_enum_animation_action_items,
    )
    animation_big_endian: BoolProperty(
        name='Big Endian',
        default=False,
        description='Export animation data in big endian for platforms such as Nintendo, PS3, and Xbox',
    )
    animation_underworld_format: BoolProperty(
        name='Underworld Format',
        default=False,
        description='Export animation data to Underworld format',
    )
    debug: BoolProperty(
        name='Debug',
        default=False,
        description='Enable console debug log',
    )

    @classmethod
    def poll(cls, context):
        if bool(getattr(bpy.data, 'collections', [])):
            return True
        return any(getattr(obj, 'type', None) == 'ARMATURE' for obj in getattr(bpy.data, 'objects', []) or [])

    def invoke(self, context, event):
        default_name = self._default_collection_name(context)

        enum_ids = [item[0] for item in _enum_collection_items(self, context)]
        if default_name not in enum_ids:
            default_name = next((item_id for item_id in enum_ids if item_id != '__TRLAU_NO_EXPORT_COLLECTION__'), '')

        if default_name:
            self.collection_name = default_name

        active_armature = _active_armature(context)
        first_armature = next((obj for obj in bpy.data.objects if getattr(obj, 'type', None) == 'ARMATURE'), None)
        chosen_armature = active_armature or first_armature
        if chosen_armature is not None:
            try:
                self.animation_armature_name = chosen_armature.name
            except Exception:
                pass

        active_action = getattr(getattr(active_armature, 'animation_data', None), 'action', None) if active_armature is not None else None
        first_action = next(iter(bpy.data.actions), None)
        chosen_action = active_action or first_action
        if chosen_action is not None:
            try:
                self.animation_action_name = chosen_action.name
            except Exception:
                pass

        if not self.filepath:
            if active_armature is not None:
                action_name = chosen_action.name if chosen_action is not None else active_armature.name
                self.filepath = bpy.path.ensure_ext(str(Path('//') / f'{action_name}.ani'), '.ani')
            else:
                collection = bpy.data.collections.get(default_name) if default_name else None
                if self._is_model_export_collection(collection):
                    source_drm = TRLAUModelExporter.find_source_drm_for_collection(collection)
                    if source_drm is not None:
                        self.filepath = bpy.path.ensure_ext(str(Path('//') / source_drm.name), '.drm')
                    else:
                        object_name = self._default_object_section_name(collection)
                        self.filepath = bpy.path.ensure_ext(str(Path('//') / object_name), '.obj')
                else:
                    drm_name = self._default_drm_name(context, default_name)
                    self.filepath = bpy.path.ensure_ext(str(Path('//') / drm_name), '.drm')
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def check(self, context):
        if _is_animation_export_filepath(self.filepath):
            return False

        collection = bpy.data.collections.get(self.collection_name)
        if self._is_model_export_collection(collection):
            if self.filepath and _trlau_filepath_suffix(self.filepath) not in {'.obj', '.drm'}:
                source_drm = TRLAUModelExporter.find_source_drm_for_collection(collection)
                desired_ext = '.drm' if source_drm is not None else '.obj'
                self.filepath = _trlau_replace_filepath_suffix(self.filepath, desired_ext)
                return True
        elif self.filepath and _trlau_filepath_suffix(self.filepath) != '.drm':
            self.filepath = _trlau_replace_filepath_suffix(self.filepath, '.drm')
            return True
        return False

    def draw(self, context):
        layout = self.layout
        if _is_animation_export_filepath(self.filepath):
            layout.label(text='Animation Export')
            if any(getattr(obj, 'type', None) == 'ARMATURE' for obj in bpy.data.objects):
                layout.prop(self, 'animation_armature_name')
            else:
                layout.label(text='Armature: none available', icon='ERROR')
            layout.prop(self, 'animation_big_endian')
            layout.prop(self, 'animation_underworld_format')
            if len(bpy.data.actions) > 0:
                layout.prop(self, 'animation_action_name')
            else:
                layout.label(text='Action: none available', icon='ERROR')
            layout.prop(self, 'debug')
            return

        layout.label(text='Mesh Collection')
        layout.prop(self, 'collection_name', text='')
        collection = bpy.data.collections.get(self.collection_name)
        if collection is None:
            layout.label(text='No exportable Model/Level collection found', icon='ERROR')
        else:
            layout.prop(self, 'export_textures')
            if self._is_model_export_collection(collection):
                layout.prop(self, 'export_cloth')
        layout.prop(self, 'debug')

    def execute(self, context):
        configure_logging(bool(self.debug))

        if _is_animation_export_filepath(self.filepath):
            armature = None
            armature_name = str(getattr(self, 'animation_armature_name', '') or '')
            if armature_name and armature_name != '__TRLAU_NO_ARMATURE__':
                candidate = bpy.data.objects.get(armature_name)
                if candidate is not None and getattr(candidate, 'type', None) == 'ARMATURE':
                    armature = candidate
            if armature is None:
                armature = _active_armature(context)
            if armature is None:
                armature = next((obj for obj in bpy.data.objects if getattr(obj, 'type', None) == 'ARMATURE'), None)
            if armature is None:
                self.report({'ERROR'}, 'Choose an armature to export animation')
                return {'CANCELLED'}

            action = None
            action_name = str(getattr(self, 'animation_action_name', '') or '')
            if action_name and action_name != '__TRLAU_NO_ACTION__':
                action = bpy.data.actions.get(action_name)
            if action is None:
                action = getattr(getattr(armature, 'animation_data', None), 'action', None)
            if action is None:
                action = next(iter(bpy.data.actions), None)
            if action is None:
                self.report({'ERROR'}, 'Choose an action to export')
                return {'CANCELLED'}

            exporter = TRLAUAnimationExporter(debug=bool(self.debug))
            try:
                prepared = exporter.export_animation(
                    context,
                    armature,
                    action,
                    self.filepath,
                    big_endian=bool(self.animation_big_endian),
                    underworld=bool(self.animation_underworld_format),
                )
            except Exception as exc:
                logger.exception('TRLAU animation export failed')
                self.report({'ERROR'}, f'TRLAU animation export failed: {exc}')
                return {'CANCELLED'}

            for message in exporter.warnings:
                self.report({'WARNING'}, message)
            self.report({'INFO'}, f'Exported TRLAU animation {prepared.anim_id}: {prepared.frame_count} frame(s), TimePerKey {prepared.time_per_key}')
            return {'FINISHED'}

        collection = bpy.data.collections.get(self.collection_name)
        if collection is None:
            self.report({'ERROR'}, 'Choose a valid collection to export')
            return {'CANCELLED'}

        try:
            if self._is_model_export_collection(collection):
                exporter_cls = TRLAUPS2ModelExporter if TRLAUPS2ModelExporter.collection_is_ps2_model(collection) else TRLAUModelExporter
                exporter = exporter_cls(debug=bool(self.debug))
                suffix = _trlau_filepath_suffix(self.filepath)
                if suffix == '.drm':
                    written = exporter.export_collection_to_drm(context, collection, self.filepath, export_textures=bool(self.export_textures), export_cloth=bool(self.export_cloth))
                    for message in getattr(exporter, 'warnings', []) or []:
                        self.report({'WARNING'}, message)
                    self.report({'INFO'}, f'Exported TRLAU model DRM: {_trlau_filepath_name(self.filepath)}')
                    return {'FINISHED'}
                if suffix == '.obj':
                    written = exporter.export_collection(context, collection, self.filepath, export_textures=bool(self.export_textures), export_cloth=bool(self.export_cloth))
                    for message in getattr(exporter, 'warnings', []) or []:
                        self.report({'WARNING'}, message)
                    self.report({'INFO'}, f'Exported {len(written)} TRLAU model/texture section file(s)')
                    return {'FINISHED'}
                raise ValueError('Model export requires a .drm target or an extracted .obj object section file')

            level = self._build_export_level(context, collection, export_textures=bool(self.export_textures), export_area_dbase=True)
            validation_errors, validation_warnings = self._validate_export_level(level)
            if validation_warnings:
                for message in validation_warnings:
                    logger.warning(message)
                    self.report({'WARNING'}, message)
            if validation_errors:
                for message in validation_errors:
                    logger.warning(message)
                raise ValueError(validation_errors[0])
            writer = TRLevelDRMWriter()
            writer.write(level, self.filepath)
        except Exception as exc:
            logger.exception('TRLAU export failed')
            self.report({'ERROR'}, f'TRLAU export failed: {exc}')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Exported TRLAU level DRM: {_trlau_filepath_name(self.filepath)}')
        return {'FINISHED'}

    def _build_export_level(self, context, collection: Collection, *, export_textures: bool = True, export_area_dbase: bool = True) -> ExportLevel:
        root = self._find_level_root(collection)
        if root is None:
            raise ValueError(f'Collection "{collection.name}" does not contain a <DRMName>_Level empty')

        drm_name = self._determine_drm_name(root, collection)
        level_game = self._level_game_from_root(root)
        root_matrix_inv = root.matrix_world.inverted_safe()
        selected_names = None

        terrain_groups: List[ExportTerrainGroup] = []
        for group_empty in self._iter_group_objects(collection, root):
            if selected_names is not None and group_empty.name not in selected_names:
                child_selected = any(child.name in selected_names for child in group_empty.children_recursive)
                if not child_selected:
                    continue
            export_group = self._export_group(group_empty, root_matrix_inv, selected_names, drm_name)
            if export_group is not None:
                terrain_groups.append(export_group)

        if not terrain_groups:
            raise ValueError('No TerrainGroup data was found in the chosen collection')

        terrain_groups.sort(key=lambda group: int(group.index))
        for default_index, group in enumerate(terrain_groups):
            group.index = int(default_index if group.index < 0 else group.index)
            if not group.group_origin or all(abs(float(v)) < 1e-6 for v in group.group_origin):
                group.group_origin = tuple(float(v) for v in group.position)
            if not group.global_offset or all(abs(float(v)) < 1e-6 for v in group.global_offset):
                group.global_offset = tuple(float(v) for v in group.position)

        metadata = self._gather_root_metadata(root)
        unit_data = self._gather_unit_data_metadata(collection, root)
        combat_data = self._gather_combat_data_metadata(collection, root)
        admd_data = self._gather_component_metadata(collection, root, 'ADMDData', 'trlau_type', 'ADMDData', 'trlau_admd_empty', 'trlau_admd_')
        passthrough_sections = self._gather_passthrough_sections(collection, root, metadata, unit_data, export_area_dbase=export_area_dbase)
        if export_textures:
            passthrough_sections.extend(self._gather_texture_passthrough_sections(collection, root, selected_names))
        bg_objects, bg_instances = self._gather_bg_objects_and_instances(collection, root, root_matrix_inv)
        signals, signal_mesh = self._gather_signals(collection, root, root_matrix_inv)
        sfx_markers = self._gather_sfx_markers(collection, root, root_matrix_inv)

        return ExportLevel(
            drm_name=drm_name,
            terrain_groups=terrain_groups,
            signals=signals,
            signal_mesh=signal_mesh,
            bg_objects=bg_objects,
            bg_instances=bg_instances,
            intro_data=self._gather_intro_data(collection, root, root_matrix_inv),
            terrain_lights=self._gather_terrain_lights(collection, root, root_matrix_inv),
            markups=self._gather_markups(collection, root, root_matrix_inv, level_game),
            sfx_markers=sfx_markers,
            terrain_light_grid_cells=[],
            metadata=metadata,
            unit_data=unit_data,
            admd_data=admd_data,
            combat_data=combat_data,
            scene_center_offset=self._extract_scene_center_offset(root),
            passthrough_sections=passthrough_sections,
            game=level_game,
        )

    def _validate_export_level(self, level: ExportLevel) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        level_game = self._normalize_level_game(getattr(level, 'game', 'legend'))
        required_player_intro_object_id = 72 if level_game == 'anniversary' else 171 if level_game == 'legend' else None
        if required_player_intro_object_id is not None:
            intro_object_ids: set[int] = set()
            for intro in getattr(level, 'intro_data', []) or []:
                try:
                    intro_object_ids.add(int(getattr(intro, 'object_id', -1)))
                except Exception:
                    continue
            if required_player_intro_object_id not in intro_object_ids:
                warnings.append('No player introData found, game will crash!')

        for group in level.terrain_groups:
            if len(group.strips) > 0xFFFF:
                errors.append(f'TerrainGroup {group.index} has {len(group.strips)} strips/materials, which exceeds the format limit of 65535.')

            for strip in group.strips:
                if len(strip.vertices) > 0x7FFF:
                    errors.append(f'Strip "{strip.name}" in TerrainGroup {group.index} has {len(strip.vertices)} vertices, which exceeds the format limit of 32767.')
                if strip.indices:
                    max_index = max(int(index) for index in strip.indices)
                    if max_index >= len(strip.vertices):
                        errors.append(f'Strip "{strip.name}" in TerrainGroup {group.index} contains an index that does not reference an exported vertex.')
                    if max_index > 0x7FFF:
                        errors.append(f'Strip "{strip.name}" in TerrainGroup {group.index} references vertex index {max_index}, which exceeds the format limit of 32767.')

            collision = group.collision
            if collision is not None:
                if len(collision.vertices) > 0xFFFF:
                    errors.append(f'Collision in TerrainGroup {group.index} has {len(collision.vertices)} vertices, which exceeds the format limit of 65535.')
                if len(collision.faces) > 0xFFFF:
                    errors.append(f'Collision in TerrainGroup {group.index} has {len(collision.faces)} faces, which exceeds the format limit of 65535.')
                if len(collision.kd_nodes) > 0xFFFF:
                    errors.append(f'Collision in TerrainGroup {group.index} has {len(collision.kd_nodes)} KDNodes, which exceeds the format limit of 65535.')

        signal_mesh = getattr(level, 'signal_mesh', None)
        stream_portal_count = len(list((getattr(level, 'metadata', {}) or {}).get('streamUnitPortals', []) or []))
        if getattr(level, 'signals', None) and stream_portal_count <= 0:
            signal_needs_stream_portal = False
            for signal in list(getattr(level, 'signals', []) or []):
                data = dict(getattr(signal, 'data', {}) or {})
                if str(data.get('streamName', '') or '').strip() or bool(data.get('entryPortal', False)) or bool(data.get('exitPortal', False)) or bool(data.get('autoStream', False)):
                    signal_needs_stream_portal = True
                    break
            if signal_needs_stream_portal:
                warnings.append('Signal stream/portal data exists, but streamUnitPortals metadata is missing. Re-import the original DRM with this addon version before re-exporting stream-connected signals.')
        if signal_mesh is not None:
            if len(getattr(signal_mesh, 'vertices', []) or []) > 0xFFFF:
                errors.append(f'Signal mesh has {len(signal_mesh.vertices)} vertices, which exceeds the format limit of 65535.')
            if len(getattr(signal_mesh, 'faces', []) or []) > 0xFFFF:
                errors.append(f'Signal mesh has {len(signal_mesh.faces)} faces, which exceeds the format limit of 65535.')
        signal_indices = []
        for signal in getattr(level, 'signals', []) or []:
            try:
                signal_indices.append(int(getattr(signal, 'index', -1)))
            except Exception:
                continue
        if signal_indices:
            if min(signal_indices) < 0 or max(signal_indices) > 0x7FFF:
                errors.append('Signal list indices must be in the range 0..32767.')
            missing_indices = [index for index in range(max(signal_indices) + 1) if index not in set(signal_indices)]
            if missing_indices:
                warnings.append('Signal list has gaps; missing Signal structs will be exported as zeroed entries.')

        for bg_object in getattr(level, 'bg_objects', []) or []:
            if len(bg_object.vertices) > 0xFFFF:
                errors.append(f'BGObject {bg_object.index} has {len(bg_object.vertices)} vertices, which exceeds the format limit of 65535.')
            for strip in bg_object.strips:
                if strip.indices:
                    max_index = max(int(index) for index in strip.indices)
                    if max_index >= len(bg_object.vertices):
                        errors.append(f'BGObject {bg_object.index} strip "{strip.name}" contains an index that does not reference an exported vertex.')
                    if max_index > 0xFFFF:
                        errors.append(f'BGObject {bg_object.index} strip "{strip.name}" references vertex index {max_index}, which exceeds the format limit of 65535.')
                    if len(strip.indices) > 0x7FFFFFFF:
                        errors.append(f'BGObject {bg_object.index} strip "{strip.name}" has too many indices.')

        def _dedupe(messages: list[str]) -> list[str]:
            deduped: list[str] = []
            seen: set[str] = set()
            for message in messages:
                if message in seen:
                    continue
                seen.add(message)
                deduped.append(message)
            return deduped

        return _dedupe(errors), _dedupe(warnings)

    @staticmethod
    def _property_value(source, field_name: str, default=None):
        if source is None:
            return default
        try:
            if hasattr(source, field_name):
                return getattr(source, field_name)
        except Exception:
            pass
        try:
            return source.get(field_name, default)
        except Exception:
            return default

    @classmethod
    def _spline_property_group_to_dict(cls, spline_data) -> dict[str, object]:
        values: dict[str, object] = {}
        if spline_data is None:
            return values
        for field_name in _SIGNAL_SPLINE_FIELD_NAMES:
            values[field_name] = cls._property_value(spline_data, field_name, 0)
        return values

    @classmethod
    def _simple_property_collection_to_dicts(cls, collection, field_names: Sequence[str]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in list(collection or []):
            result.append({field_name: cls._property_value(item, field_name, 0) for field_name in field_names})
        return result

    @classmethod
    def _attack_wave_collection_to_dicts(cls, collection) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for wave in list(collection or []):
            value = {
                field_name: cls._property_value(wave, field_name, 0)
                for field_name in (
                    'chain_index', 'radiusCheckTimer', 'flags', 'rtFlags', 'probability',
                    'numAttackerSpawns', 'numSpawnsThisLoad', 'currentAttacker',
                    'triggerRemaining', 'numAttackersPerWave', 'numLinkedAttackWaves',
                    'numBlockedAttackWaves', 'numFinishedAttackWaves', 'numAttackers',
                    'musicRank', 'radiusCheckMarker', 'radiusCheckRadius', 'endMarker',
                    'numLimits', 'numBoxLimits', 'numPlaneLimits', 'numRunAndGunPos',
                    'numPatrolPos', 'patrolType', 'numMessages',
                )
            }
            value['limits'] = cls._simple_property_collection_to_dicts(getattr(wave, 'limits', []), ('x', 'y', 'z', 'rad', 'pursueRad'))
            value['boxLimits'] = cls._simple_property_collection_to_dicts(getattr(wave, 'boxLimits', []), ('x', 'y', 'z', 'zrot', 'width', 'length', 'pursueWidth', 'pursueLength'))
            value['planeLimits'] = cls._simple_property_collection_to_dicts(getattr(wave, 'planeLimits', []), ('px', 'py', 'pz', 'nx', 'ny', 'nz'))
            value['runAndGunPositions'] = cls._simple_property_collection_to_dicts(getattr(wave, 'runAndGunPositions', []), ('x', 'y', 'z', 'priority', 'waittime', 'failAction', 'markerName'))
            value['patrolPositions'] = cls._simple_property_collection_to_dicts(getattr(wave, 'patrolPositions', []), ('x', 'y', 'z', 'lookx', 'looky', 'lookz', 'waittime', 'anim', 'mode', 'look'))
            value['messages'] = cls._simple_property_collection_to_dicts(getattr(wave, 'messages', []), ('message', 'data'))
            result.append(value)
        result.sort(key=lambda item: int(item.get('chain_index', 0) or 0))
        return result

    @classmethod
    def _signal_data_to_export_dict(cls, signal_data) -> dict[str, object]:
        data: dict[str, object] = {}
        if signal_data is None:
            return data
        for field_name in _SIGNAL_VALUE_FIELD_NAMES:
            data[field_name] = cls._property_value(signal_data, field_name, False if field_name.startswith('startGoing') else 0)
        for field_name in _SIGNAL_FLAG_FIELD_NAMES:
            data[field_name] = bool(cls._property_value(signal_data, field_name, False))
        data['splineCamParams'] = cls._spline_property_group_to_dict(getattr(signal_data, 'splineCamParams', None))
        spline_ptrs: list[object] = []
        for slot in range(3):
            ptr = getattr(signal_data, f'splineCamDataPtr{slot}', None)
            if ptr is not None and bool(getattr(ptr, 'present', False)):
                spline_ptrs.append(cls._spline_property_group_to_dict(getattr(ptr, 'data', None)))
            else:
                spline_ptrs.append(None)
        data['splineCamDataPtr'] = spline_ptrs
        for collection_name in ('enableAttackWaves', 'disableAttackWaves', 'killAttackWaves'):
            data[collection_name] = cls._attack_wave_collection_to_dicts(getattr(signal_data, collection_name, []))
        return data

    @staticmethod
    def _face_int_attribute(mesh, attr_name: str, polygon_index: int, default: int) -> int:
        attributes = getattr(mesh, 'attributes', None)
        if attributes is None:
            return int(default)
        attr = attributes.get(attr_name)
        if attr is None:
            return int(default)
        try:
            if getattr(attr, 'domain', '') == 'FACE':
                return int(getattr(attr.data[int(polygon_index)], 'value', default))
        except Exception:
            return int(default)
        return int(default)

    def _iter_signal_objects(self, collection: Collection, root) -> list[object]:
        objects = list(getattr(collection, 'all_objects', []) or [])
        signal_objects = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'MESH':
                continue
            if not is_descendant_of(obj, root) and obj.parent is not root:
                continue
            obj_type = str(obj.get('trlau_type', '') or '') if hasattr(obj, 'get') else ''
            if obj_type == 'Signal' or bool(obj.get('trlau_signal', False)):
                signal_objects.append(obj)
        signal_objects.sort(key=lambda obj: (
            int(getattr(getattr(obj, 'trlau_signal_data', None), 'list_index', obj.get('trlau_signal_list_index', 0)) or 0),
            str(getattr(obj, 'name', '')),
        ))
        return signal_objects

    def _gather_signals(self, collection: Collection, root, root_matrix_inv) -> tuple[list[ExportSignal], Optional[ExportSignalMesh]]:
        signal_objects = self._iter_signal_objects(collection, root)
        if not signal_objects:
            return [], None

        vertices: list[tuple[float, float, float]] = []
        faces: list[ExportSignalFace] = []
        vertex_map: dict[tuple[float, float, float], int] = {}
        signals: list[ExportSignal] = []
        used_indices: set[int] = set()

        for signal_order, obj in enumerate(signal_objects):
            signal_data = getattr(obj, 'trlau_signal_data', None)
            if signal_data is None:
                continue
            signal_index = int(getattr(signal_data, 'list_index', -1))
            if signal_index < 0:
                signal_index = signal_order
            signal_id = int(getattr(signal_data, 'signal_id', signal_index))
            while signal_index in used_indices:
                signal_index += 1
            used_indices.add(signal_index)

            export_data = self._signal_data_to_export_dict(signal_data)
            signals.append(ExportSignal(index=int(signal_index), signal_id=int(signal_id), data=export_data))

            mesh, eval_obj = self._evaluated_mesh(obj)
            if mesh is None:
                continue
            try:
                mesh.calc_loop_triangles()
                for tri in mesh.loop_triangles:
                    face_indices: list[int] = []
                    for loop_index in tri.loops:
                        vertex_index = mesh.loops[loop_index].vertex_index
                        local_co = root_matrix_inv @ (obj.matrix_world @ mesh.vertices[vertex_index].co)
                        vertex_key = (round(float(local_co.x), 6), round(float(local_co.y), 6), round(float(local_co.z), 6))
                        out_index = vertex_map.get(vertex_key)
                        if out_index is None:
                            out_index = len(vertices)
                            vertex_map[vertex_key] = out_index
                            vertices.append((float(local_co.x), float(local_co.y), float(local_co.z)))
                        face_indices.append(out_index)
                    if len(face_indices) != 3 or len(set(face_indices)) != 3:
                        continue
                    polygon_index = int(getattr(tri, 'polygon_index', 0) or 0)
                    faces.append(ExportSignalFace(
                        i0=int(face_indices[0]),
                        i1=int(face_indices[1]),
                        i2=int(face_indices[2]),
                        signal_index=int(signal_index),
                        adjacency_flags=self._face_int_attribute(mesh, 'trlau_signal_adjacency_flags', polygon_index, 24),
                        collision_flags=self._face_int_attribute(mesh, 'trlau_signal_collision_flags', polygon_index, 255),
                    ))
            finally:
                self._free_evaluated_mesh(eval_obj)

        if not vertices or not faces:
            return signals, None
        return signals, ExportSignalMesh(vertices=vertices, faces=faces)


    def _export_group(self, group_empty, root_matrix_inv, selected_names: Optional[set[str]], drm_name: str = '') -> Optional[ExportTerrainGroup]:
        group_index = self._get_group_index(group_empty)
        group_position = self._relative_translation(group_empty, root_matrix_inv)

        export_group = ExportTerrainGroup(
            index=group_index,
            position=group_position,
            global_offset=self._vector_prop(group_empty, 'trlau_terrain_group_global_offset', default=group_position),
            local_offset=self._vector_prop(group_empty, 'trlau_terrain_group_local_offset', default=(0.0, 0.0, 0.0)),
            flags=int(group_empty.get('trlau_terrain_group_flags', 0)),
            terrain_id=int(group_empty.get('trlau_terrain_group_id', 0)),
            unique_id=int(group_empty.get('trlau_terrain_group_unique_id', group_index)),
            spline_id=int(group_empty.get('trlau_terrain_group_spline_id', 0)),
            texture_morph_value=float(group_empty.get('trlau_terrain_group_texture_morph_value', 0.0)),
            texture_morph_step=float(group_empty.get('trlau_terrain_group_texture_morph_step', 0.0)),
            group_origin=self._vector_prop(group_empty, 'trlau_terrain_group_origin', default=group_position),
        )

        group_collection = self._group_collection(group_empty)
        collision_empty = self._find_collision_empty(group_empty, group_collection, drm_name)
        if collision_empty is not None:
            collision = self._export_collision(collision_empty, selected_names, group_collection, drm_name=drm_name, group_index=group_index)
            if collision is not None:
                export_group.collision = collision

        strip_meshes = self._iter_group_strip_meshes(group_collection, group_empty, drm_name)

        strip_index = 0
        for strip_obj in strip_meshes:
            if selected_names is not None and strip_obj.name not in selected_names and group_empty.name not in selected_names:
                continue
            strips = self._export_strips_for_object(group_empty, strip_obj, strip_index)
            if not strips:
                continue
            export_group.strips.extend(strips)
            strip_index += len(strips)

        if not export_group.strips and export_group.collision is None:
            return None
        return export_group

    @staticmethod
    def _resolve_strip_material(obj, material_slot_index: Optional[int] = None):
        if material_slot_index is not None:
            slots = tuple(getattr(obj, 'material_slots', None) or ())
            if 0 <= int(material_slot_index) < len(slots):
                material = getattr(slots[int(material_slot_index)], 'material', None)
                if material is not None:
                    return material

        material = getattr(obj, 'active_material', None)
        if material is not None:
            return material

        for slot in getattr(obj, 'material_slots', None) or ():
            material = getattr(slot, 'material', None)
            if material is not None:
                return material
        return None

    @staticmethod
    def _parse_collision_client_flag_label(label: object) -> Optional[int]:
        text = str(label or '').strip()
        if not text:
            return None
        text = re.sub(r'\.\d{3}$', '', text).strip()
        normalized = re.sub(r'[_\s-]+', '', text.lower())
        if normalized in _COLLISION_CLIENT_FLAG_NAMES:
            return int(_COLLISION_CLIENT_FLAG_NAMES[normalized])
        try:
            return int(text, 10)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _resolve_collision_client_flag(cls, obj, material_slot_index: Optional[int] = None) -> int:
        material = cls._resolve_strip_material(obj, material_slot_index)
        if material is None:
            return 0

        if 'trlau_collision_client_flag' in material:
            try:
                return int(material['trlau_collision_client_flag'])
            except (TypeError, ValueError):
                pass

        material_name = re.sub(r'\.\d{3}$', '', str(getattr(material, 'name', '') or '').strip())
        direct_value = cls._parse_collision_client_flag_label(material_name)
        if direct_value is not None:
            return direct_value

        match = _COLLISION_CLIENT_FLAG_RE.search(material_name)
        if match:
            parsed = cls._parse_collision_client_flag_label(match.group(1))
            if parsed is not None:
                return parsed

        return 0

    def _iter_used_strip_materials(self, obj) -> list[tuple[int, object]]:
        mesh, eval_obj = self._evaluated_mesh(obj)
        if mesh is None:
            return []

        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                return []

            used_material_indices = sorted({int(getattr(tri, 'material_index', 0)) for tri in mesh.loop_triangles})
            materials: list[tuple[int, object]] = []
            for material_index in used_material_indices:
                material = self._resolve_strip_material(obj, material_index)
                if material is not None:
                    materials.append((int(material_index), material))
            return materials
        finally:
            self._free_evaluated_mesh(eval_obj)

    @staticmethod
    def _resolve_strip_tpageid(obj, material=None) -> int:
        material = material or EXPORT_SCENE_OT_trlau_level._resolve_strip_material(obj)
        if material is not None:
            return int(get_material_tpageid(material))

        return 0

    @staticmethod
    def _resolve_strip_scroll_settings(obj, material=None) -> tuple[bool, float, int, int]:
        material = material or EXPORT_SCENE_OT_trlau_level._resolve_strip_material(obj)
        if material is None:
            return False, 0.0, 0, 0

        enabled = bool(get_material_scroll_enabled(material))
        raw_speed = float(get_material_scroll_speed(material))
        return enabled, raw_speed, 0, 0

    @staticmethod
    def _resolve_strip_flags(obj, scroll_enabled: bool, material=None, material_slot_index: Optional[int] = None) -> int:
        material = material or EXPORT_SCENE_OT_trlau_level._resolve_strip_material(obj, material_slot_index)

        flags = None
        if obj is not None and material_slot_index is not None:
            slot_key = f'trlau_strip_slot_{int(material_slot_index):03d}_flags'
            if slot_key in obj:
                try:
                    flags = int(obj.get(slot_key, 0))
                except Exception:
                    flags = None
        if flags is None and obj is not None and 'trlau_strip_flags' in obj:
            flags = int(obj.get('trlau_strip_flags', 0))
        elif flags is None and material is not None and 'trlau_level_strip_flags' in material:
            flags = int(material.get('trlau_level_strip_flags', 0))
        elif flags is None:
            flags = 0

        if scroll_enabled:
            flags |= 0x1
        else:
            flags &= ~0x1

        return flags

    @staticmethod
    def _choose_uv_tile_shift(values: Sequence[float]) -> tuple[int, bool]:
        if not values:
            return 0, True

        raw_values = [int(round(float(value) * _UV_EXPORT_SCALE)) for value in values]
        min_raw = min(raw_values)
        max_raw = max(raw_values)

        lower = math.ceil((_UV_RAW_MIN - min_raw) / _UV_TILE_UNITS)
        upper = math.floor((_UV_RAW_MAX - max_raw) / _UV_TILE_UNITS)

        if lower <= upper:
            center_shift = int(round(-((min_raw + max_raw) * 0.5) / _UV_TILE_UNITS))
            return max(lower, min(upper, center_shift)), True

        unclamped_shift = int(round(-((min_raw + max_raw) * 0.5) / _UV_TILE_UNITS))
        return unclamped_shift, False

    @classmethod
    def _normalize_strip_uvs(
        cls,
        corners: Sequence[dict[str, object]],
        triangles: Sequence[tuple[int, int, int]],
        object_name: str,
    ) -> list[tuple[float, float]]:
        if not corners:
            return []

        node_indices: dict[tuple[int, int, int], int] = {}
        node_uvs: list[tuple[float, float]] = []
        corner_to_node: list[int] = [0] * len(corners)

        for corner_index, corner in enumerate(corners):
            vertex_index = int(corner['vertex_index'])
            uv = corner['uv']
            key = (
                vertex_index,
                round(float(uv[0]) * _UV_EXPORT_SCALE),
                round(float(uv[1]) * _UV_EXPORT_SCALE),
            )
            node_index = node_indices.get(key)
            if node_index is None:
                node_index = len(node_uvs)
                node_indices[key] = node_index
                node_uvs.append((float(uv[0]), float(uv[1])))
            corner_to_node[corner_index] = node_index

        adjacency: list[set[int]] = [set() for _ in range(len(node_uvs))]
        for corner_a, corner_b, corner_c in triangles:
            node_a = corner_to_node[corner_a]
            node_b = corner_to_node[corner_b]
            node_c = corner_to_node[corner_c]
            adjacency[node_a].add(node_b)
            adjacency[node_a].add(node_c)
            adjacency[node_b].add(node_a)
            adjacency[node_b].add(node_c)
            adjacency[node_c].add(node_a)
            adjacency[node_c].add(node_b)

        normalized_nodes = list(node_uvs)
        visited = [False] * len(node_uvs)
        warned = False
        for start_index in range(len(node_uvs)):
            if visited[start_index]:
                continue

            stack = [start_index]
            component: list[int] = []
            visited[start_index] = True
            while stack:
                node_index = stack.pop()
                component.append(node_index)
                for neighbor in adjacency[node_index]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

            component_us = [node_uvs[node_index][0] for node_index in component]
            component_vs = [node_uvs[node_index][1] for node_index in component]
            shift_u, fits_u = cls._choose_uv_tile_shift(component_us)
            shift_v, fits_v = cls._choose_uv_tile_shift(component_vs)

            for node_index in component:
                u, v = node_uvs[node_index]
                normalized_nodes[node_index] = (u + shift_u, v + shift_v)

            if not fits_u or not fits_v:
                warned = True

        normalized_corners = [normalized_nodes[corner_to_node[corner_index]] for corner_index in range(len(corners))]

        return normalized_corners


    def _collect_strip_geometry(self, group_empty, obj, mesh, triangle_source: Sequence[object]) -> tuple[list[dict[str, object]], list[tuple[int, int, int]]]:
        parent_matrix_inv = group_empty.matrix_world.inverted_safe()
        uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
        color_attr = self._active_color_attribute(mesh)

        corners: List[dict[str, object]] = []
        triangles: List[tuple[int, int, int]] = []
        for tri in triangle_source:
            face_corner_indices: List[int] = []
            for loop_index in tri.loops:
                vertex_index = mesh.loops[loop_index].vertex_index
                local_co = parent_matrix_inv @ (obj.matrix_world @ mesh.vertices[vertex_index].co)
                uv = (0.0, 0.0)
                if uv_layer is not None:
                    uv_value = uv_layer[loop_index].uv
                    uv = (float(uv_value.x), 1.0 - float(uv_value.y))
                color = self._sample_color(color_attr, loop_index, vertex_index)
                face_corner_indices.append(len(corners))
                corners.append({
                    'vertex_index': int(vertex_index),
                    'position': (float(local_co.x), float(local_co.y), float(local_co.z)),
                    'uv': uv,
                    'color': color,
                })
            if len(face_corner_indices) == 3:
                triangles.append((face_corner_indices[0], face_corner_indices[1], face_corner_indices[2]))

        return corners, triangles

    def _build_export_strip(self, obj, strip_name: str, corners: Sequence[dict[str, object]], triangles: Sequence[tuple[int, int, int]], material, material_slot_index: Optional[int] = None) -> Optional[ExportStrip]:
        if not triangles or not corners:
            return None

        normalized_uvs = self._normalize_strip_uvs(corners, triangles, strip_name)

        vertices: List[ExportVertex] = []
        indices: List[int] = []
        vertex_map: dict[tuple[float, float, float, float, float, int, int, int, int], int] = {}
        for triangle in triangles:
            face_indices: List[int] = []
            for corner_index in triangle:
                corner = corners[corner_index]
                position = corner['position']
                uv = normalized_uvs[corner_index]
                color = corner['color']
                vertex_key = (
                    round(float(position[0]), 6),
                    round(float(position[1]), 6),
                    round(float(position[2]), 6),
                    round(float(uv[0]), 6),
                    round(float(uv[1]), 6),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    int(color[3]),
                )
                deduped_index = vertex_map.get(vertex_key)
                if deduped_index is None:
                    deduped_index = len(vertices)
                    vertex_map[vertex_key] = deduped_index
                    vertices.append(
                        ExportVertex(
                            position=(float(position[0]), float(position[1]), float(position[2])),
                            uv=(float(uv[0]), float(uv[1])),
                            color=(int(color[0]), int(color[1]), int(color[2]), int(color[3])),
                        )
                    )
                face_indices.append(deduped_index)
            if len(face_indices) == 3 and len(set(face_indices)) == 3:
                indices.extend(face_indices)

        if not indices or not vertices:
            return None

        tpageid = self._resolve_strip_tpageid(obj, material=material)
        scroll_enabled, scroll_speed, scroll_num_tiles, scroll_tile = self._resolve_strip_scroll_settings(obj, material=material)
        strip_flags = self._resolve_strip_flags(obj, scroll_enabled, material=material, material_slot_index=material_slot_index)

        return ExportStrip(
            name=strip_name,
            tpageid=tpageid,
            flags=strip_flags,
            vertices=vertices,
            indices=indices,
            scroll_enabled=scroll_enabled,
            scroll_speed=scroll_speed,
            scroll_num_tiles=scroll_num_tiles,
            scroll_tile=scroll_tile,
        )

    def _export_strips_for_object(self, group_empty, obj, start_strip_index: int) -> List[ExportStrip]:
        mesh, eval_obj = self._evaluated_mesh(obj)
        if mesh is None:
            return []

        try:
            mesh.calc_loop_triangles()
            if not mesh.loop_triangles:
                return []

            triangles_by_material: dict[int, list[object]] = {}
            for tri in mesh.loop_triangles:
                material_index = int(getattr(tri, 'material_index', 0))
                triangles_by_material.setdefault(material_index, []).append(tri)

            if not triangles_by_material:
                return []

            used_material_indices = sorted(triangles_by_material.keys())
            if len(used_material_indices) > 1:
                logger.info(
                    'Splitting strip object %s into %d exported strips by material slot',
                    getattr(obj, 'name', '<unnamed>'),
                    len(used_material_indices),
                )

            strips: List[ExportStrip] = []
            multiple_materials = len(used_material_indices) > 1
            for subset_offset, material_index in enumerate(used_material_indices):
                corners, triangles = self._collect_strip_geometry(group_empty, obj, mesh, triangles_by_material[material_index])
                material = self._resolve_strip_material(obj, material_index)
                strip_name = obj.name if not multiple_materials else f'{obj.name}_Mat_{material_index:03d}'
                strip = self._build_export_strip(obj, strip_name, corners, triangles, material, material_slot_index=material_index)
                if strip is not None:
                    strips.append(strip)

            return strips
        finally:
            self._free_evaluated_mesh(eval_obj)

    def _export_collision(self, collision_empty, selected_names: Optional[set[str]], collection: Optional[Collection] = None, *, drm_name: str = '', group_index: Optional[int] = None) -> Optional[ExportCollision]:
        if group_index is None:
            group_index = self._get_group_index(collision_empty)
        if bool(collision_empty.get('trlau_terrain_group')) or str(collision_empty.get('trlau_type', '') or '') == 'TerrainGroup':
            collision_meshes = [
                child for child in collision_empty.children
                if child.type == 'MESH' and self._is_collision_mesh_for_group(child, int(group_index), drm_name)
            ]
        else:
            collision_meshes = [
                child for child in collision_empty.children
                if child.type == 'MESH' and (
                    bool(child.get('trlau_terrain_collision'))
                    or self._is_collision_mesh_for_group(child, int(group_index), drm_name)
                )
            ]
        if selected_names is not None:
            collision_meshes = [obj for obj in collision_meshes if obj.name in selected_names or collision_empty.name in selected_names]
        if not collision_meshes:
            return None

        parent_matrix_inv = collision_empty.matrix_world.inverted_safe()
        vertices: List[Tuple[float, float, float]] = []
        faces: List[ExportCollisionFace] = []
        vertex_map: dict[tuple[float, float, float], int] = {}

        for mesh_obj in collision_meshes:
            mesh, eval_obj = self._evaluated_mesh(mesh_obj)
            if mesh is None:
                continue
            try:
                mesh.calc_loop_triangles()
                if not mesh.loop_triangles:
                    continue
                for tri in mesh.loop_triangles:
                    face_indices: List[int] = []
                    client_flag = self._resolve_collision_client_flag(mesh_obj, int(getattr(tri, 'material_index', 0) or 0))
                    for loop_index in tri.loops:
                        vertex_index = mesh.loops[loop_index].vertex_index
                        local_co = parent_matrix_inv @ (mesh_obj.matrix_world @ mesh.vertices[vertex_index].co)
                        vertex_key = (
                            round(float(local_co.x), 6),
                            round(float(local_co.y), 6),
                            round(float(local_co.z), 6),
                        )
                        deduped_index = vertex_map.get(vertex_key)
                        if deduped_index is None:
                            deduped_index = len(vertices)
                            vertex_map[vertex_key] = deduped_index
                            vertices.append((float(local_co.x), float(local_co.y), float(local_co.z)))
                        face_indices.append(deduped_index)
                    if len(face_indices) == 3 and len(set(face_indices)) == 3:
                        faces.append(
                            ExportCollisionFace(
                                face_indices[0],
                                face_indices[1],
                                face_indices[2],
                                collision_flags=3,
                                client_flags=client_flag,
                            )
                        )
            finally:
                self._free_evaluated_mesh(eval_obj)

        if not vertices or not faces:
            return None

        kd_nodes, kd_max_depth = self._export_collision_kd_nodes(collision_empty, collection, selected_names)
        bbox_min, bbox_max = self._compute_bounds(vertices)
        return ExportCollision(
            position=self._relative_translation(collision_empty, collision_empty.parent.matrix_world.inverted_safe() if collision_empty.parent is not None else None),
            vertices=vertices,
            faces=faces,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            kd_nodes=kd_nodes,
            kd_max_depth=kd_max_depth,
        )

    @staticmethod
    def _blender_to_raw_float_triplet(value: Sequence[float]) -> Tuple[float, float, float]:
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        return (-x, -y, z)

    @classmethod
    def _level_bbox_to_raw_bounds(cls, bbox_min: Sequence[float], bbox_max: Sequence[float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        raw_a = cls._blender_to_raw_float_triplet(bbox_min)
        raw_b = cls._blender_to_raw_float_triplet(bbox_max)
        return (
            tuple(min(float(raw_a[index]), float(raw_b[index])) for index in range(3)),
            tuple(max(float(raw_a[index]), float(raw_b[index])) for index in range(3)),
        )

    def _export_collision_kd_nodes(self, collision_empty, collection: Optional[Collection], selected_names: Optional[set[str]]) -> Tuple[List[ExportCollisionKDNode], int]:
        kd_root = self._find_collision_kd_empty(collision_empty, collection)
        if kd_root is None:
            return [], 0

        node_objects = [child for child in kd_root.children if child.type == 'EMPTY' and bool(child.get('trlau_terrain_collision_kd_node'))]
        if selected_names is not None and kd_root.name not in selected_names and collision_empty.name not in selected_names:
            node_objects = [obj for obj in node_objects if obj.name in selected_names]
        if not node_objects:
            return [], int(kd_root.get('trlau_collision_kd_depth', 0) or 0)

        node_info_by_index: dict[int, dict[str, object]] = {}
        for obj in node_objects:
            node_index = int(obj.get('trlau_collision_kd_node_index', -1))
            if node_index < 0:
                continue
            bbox_min = self._vector_prop(obj, 'trlau_collision_kd_bbox_min', obj.location)
            bbox_max = self._vector_prop(obj, 'trlau_collision_kd_bbox_max', obj.location)
            raw_min, raw_max = self._level_bbox_to_raw_bounds(bbox_min, bbox_max)
            node_info_by_index[node_index] = {
                'axis': int(obj.get('trlau_collision_kd_axis', 0) or 0),
                'num_faces': int(obj.get('trlau_collision_kd_num_faces', 0) or 0),
                'ref_index': int(obj.get('trlau_collision_kd_ref_index', 0) or 0),
                'face_start': int(obj.get('trlau_collision_kd_face_start', obj.get('trlau_collision_kd_ref_index', 0)) or 0),
                'depth': int(obj.get('trlau_collision_kd_depth', 0) or 0),
                'raw_min': raw_min,
                'raw_max': raw_max,
                'neg_offset': float(obj.get('trlau_collision_kd_neg_offset', 0.0) or 0.0) if 'trlau_collision_kd_neg_offset' in obj else None,
                'pos_offset': float(obj.get('trlau_collision_kd_pos_offset', 0.0) or 0.0) if 'trlau_collision_kd_pos_offset' in obj else None,
            }

        if not node_info_by_index:
            return [], 0

        exported_nodes: List[ExportCollisionKDNode] = []
        max_depth = max((int(info.get('depth', 0) or 0) for info in node_info_by_index.values()), default=0)
        for node_index in sorted(node_info_by_index):
            info = node_info_by_index[node_index]
            num_faces = int(info.get('num_faces', 0) or 0)
            axis = int(info.get('axis', 0) or 0)
            ref_index = int(info.get('ref_index', 0) or 0)
            if num_faces > 0:
                exported_nodes.append(
                    ExportCollisionKDNode(
                        neg_offset=float(info.get('neg_offset', 0.0) or 0.0),
                        pos_offset=float(info.get('pos_offset', 0.0) or 0.0),
                        index=int(info.get('face_start', ref_index) or 0),
                        axis=axis,
                        num_faces=num_faces,
                    )
                )
                continue

            neg_offset = info.get('neg_offset')
            pos_offset = info.get('pos_offset')
            if neg_offset is None or pos_offset is None:
                split_value = None
                negative_child = node_info_by_index.get(node_index + 1)
                positive_child = node_info_by_index.get(node_index + ref_index) if ref_index > 0 else None
                if negative_child is not None:
                    split_value = float(negative_child['raw_max'][axis])
                elif positive_child is not None:
                    split_value = float(positive_child['raw_min'][axis])
                else:
                    split_value = (float(info['raw_min'][axis]) + float(info['raw_max'][axis])) * 0.5
                neg_offset = float(split_value + 1.0)
                pos_offset = float(split_value - 1.0)

            exported_nodes.append(
                ExportCollisionKDNode(
                    neg_offset=float(neg_offset),
                    pos_offset=float(pos_offset),
                    index=ref_index,
                    axis=axis,
                    num_faces=0,
                )
            )

        if max_depth <= 0:
            max_depth = int(kd_root.get('trlau_collision_kd_depth', 0) or 0)
            if max_depth <= 0 and exported_nodes:
                max_depth = 8
        return exported_nodes, max_depth

    @staticmethod
    def _default_collection_name(context) -> str:
        selected_collection = EXPORT_SCENE_OT_trlau_level._export_collection_for_selected_object(context)
        if selected_collection is not None:
            return selected_collection.name

        active = getattr(context, 'collection', None)
        if active is not None:
            active_name = str(getattr(active, 'name', '') or '')
            if active_name and bpy.data.collections.get(active_name) is not None:
                if EXPORT_SCENE_OT_trlau_level._is_exportable_collection(active):
                    return active_name
        for collection in bpy.data.collections:
            if _find_level_root_in_collection(collection) is not None:
                return collection.name
        for collection in bpy.data.collections:
            if TRLAUModelExporter.collection_has_models(collection):
                return collection.name
        return ''

    @staticmethod
    def _is_exportable_collection(collection: Collection | None) -> bool:
        return collection is not None and (_find_level_root_in_collection(collection) is not None or TRLAUModelExporter.collection_has_models(collection))

    @staticmethod
    def _collection_contains_object(collection: Collection, obj) -> bool:
        if collection is None or obj is None:
            return False
        try:
            if any(candidate == obj for candidate in collection.objects):
                return True
        except Exception:
            pass
        try:
            return any(candidate == obj for candidate in collection.all_objects)
        except Exception:
            return False

    @staticmethod
    def _export_collection_for_selected_object(context) -> Collection | None:
        active_obj = getattr(context, 'object', None) or getattr(context, 'active_object', None)
        selected_objects = list(getattr(context, 'selected_objects', []) or [])
        candidates_to_check = []
        for obj in [active_obj] + selected_objects:
            if obj is not None and obj not in candidates_to_check:
                candidates_to_check.append(obj)

        for obj in candidates_to_check:
            for collection in getattr(obj, 'users_collection', []) or []:
                if EXPORT_SCENE_OT_trlau_level._is_exportable_collection(collection):
                    return collection

            containing_collections = [
                collection for collection in bpy.data.collections
                if EXPORT_SCENE_OT_trlau_level._is_exportable_collection(collection)
                and EXPORT_SCENE_OT_trlau_level._collection_contains_object(collection, obj)
            ]
            if containing_collections:
                containing_collections.sort(key=lambda collection: len(list(getattr(collection, 'all_objects', []) or [])))
                return containing_collections[0]
        return None

    def _default_drm_name(self, context, collection_name: Optional[str] = None) -> str:
        collection = bpy.data.collections.get(collection_name or self._default_collection_name(context))
        if collection is None:
            return 'exported_level.drm'
        root = self._find_level_root(collection)
        base_name = self._determine_drm_name(root, collection) if root is not None else collection.name
        return f'{base_name}.drm'

    @staticmethod
    def _is_model_export_collection(collection: Collection | None) -> bool:
        return collection is not None and _find_level_root_in_collection(collection) is None and TRLAUModelExporter.collection_has_models(collection)

    @staticmethod
    def _default_object_section_name(collection: Collection | None) -> str:
        if collection is None:
            return '0_0.obj'
        return f'{collection.name}.obj'

    @staticmethod
    def _find_level_root(collection: Collection):
        return _find_level_root_in_collection(collection)

    @staticmethod
    def _determine_drm_name(root, collection: Collection) -> str:
        if root is not None:
            drm_name = str(root.get('trlau_drm_name', '') or '').strip()
            if drm_name:
                return drm_name
            match = _LEVEL_SUFFIX_RE.match(root.name)
            if match is not None:
                return match.group('prefix')
        return str(collection.name or 'Level').strip() or 'Level'

    @staticmethod
    def _iter_group_objects(collection: Collection, root) -> Iterable:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        result = []
        for obj in objects:
            if obj.type != 'EMPTY':
                continue
            if obj == root:
                continue
            if not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            if bool(obj.get('trlau_terrain_group')) or _GROUP_NAME_RE.search(obj.name):
                result.append(obj)
        result.sort(key=EXPORT_SCENE_OT_trlau_level._group_sort_key)
        return result

    @staticmethod
    def _is_descendant_of(obj, ancestor) -> bool:
        return is_descendant_of(obj, ancestor)

    @staticmethod
    def _object_index_from_name(obj, regex, group_index: int = 1, default: int = -1) -> int:
        try:
            match = regex.search(getattr(obj, 'name', '') or '')
            if match is not None:
                return int(match.group(group_index))
        except Exception:
            pass
        return int(default)

    @staticmethod
    def _bgobject_index_from_object(obj, default: int = -1) -> int:
        if obj is not None:
            try:
                if 'trlau_bgobject_index' in obj:
                    return int(obj.get('trlau_bgobject_index', default) or default)
            except Exception:
                pass
            value = EXPORT_SCENE_OT_trlau_level._object_index_from_name(obj, _BGOBJECT_NAME_RE, 1, default)
            if value >= 0:
                return value
        return int(default)

    @staticmethod
    def _bginstance_index_from_object(obj, default: int = -1) -> int:
        if obj is not None:
            try:
                if 'trlau_bginstance_index' in obj:
                    return int(obj.get('trlau_bginstance_index', default) or default)
            except Exception:
                pass
            value = EXPORT_SCENE_OT_trlau_level._object_index_from_name(obj, _BGINSTANCE_NAME_RE, 2, default)
            if value >= 0:
                return value
        return int(default)

    @staticmethod
    def _markup_index_from_object(obj, default: int = 0) -> int:
        if obj is not None:
            try:
                if 'trlau_markup_index' in obj:
                    return int(obj.get('trlau_markup_index', default) or default)
            except Exception:
                pass
            value = EXPORT_SCENE_OT_trlau_level._object_index_from_name(obj, _MARKUP_NAME_RE, 1, default)
            if value >= 0:
                return value
        return int(default)

    @staticmethod
    def _intro_index_from_object(obj, default: int = 0) -> int:
        if obj is not None:
            try:
                data = trlau_intro_data_to_dict(obj)
                if data:
                    return int(data.get('index', default) or default)
            except Exception:
                pass
            value = EXPORT_SCENE_OT_trlau_level._object_index_from_name(obj, _INTRO_NAME_RE, 1, default)
            if value >= 0:
                return value
        return int(default)

    @staticmethod
    def _find_component_empty(collection: Collection, root, component_name: str, type_key: str, type_value: str, empty_flag: str):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        suffix = f'_{component_name}'
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            if str(obj.get(type_key, '') or '') == type_value or bool(obj.get(empty_flag)):
                return obj
            if str(obj.get('trlau_component_name', '') or '') == component_name:
                return obj
            if obj.name.endswith(suffix):
                return obj
        return None

    @staticmethod
    def _gather_component_metadata(collection: Collection, root, component_name: str, type_key: str, type_value: str, empty_flag: str, prop_prefix: str) -> Optional[dict[str, object]]:
        obj = EXPORT_SCENE_OT_trlau_level._find_component_empty(collection, root, component_name, type_key, type_value, empty_flag)
        if obj is None:
            return None
        metadata: dict[str, object] = {}
        for key in obj.keys():
            if not key.startswith(prop_prefix):
                continue
            metadata[key[len(prop_prefix):]] = obj.get(key)
        return metadata or None

    @staticmethod
    def _gather_unitdata_child_entries(collection: Collection, unit_obj, type_value: str, empty_flag: str, prop_prefix: str, fields: Sequence[tuple[str, str, object]]) -> list[dict[str, object]]:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        entries: list[dict[str, object]] = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if unit_obj is not None and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, unit_obj):
                continue
            if str(obj.get('trlau_type', '') or '') != type_value and not bool(obj.get(empty_flag)):
                continue
            entry: dict[str, object] = {}
            for output_key, prop_key, default_value in fields:
                entry[output_key] = obj.get(f'{prop_prefix}{prop_key}', default_value)
            entries.append(entry)
        entries.sort(key=lambda entry: (int(entry.get('index', 0) or 0), str(entry)))
        return entries

    @staticmethod
    def _gather_unitdata_cine_entries(collection: Collection, unit_obj) -> list[dict[str, object]]:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        entries: list[dict[str, object]] = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if unit_obj is not None and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, unit_obj):
                continue
            if str(obj.get('trlau_type', '') or '') != 'UnitDataCine' and not bool(obj.get('trlau_unitdata_cine_empty')):
                continue
            try:
                index = int(obj.get('trlau_unitdata_cine_index', 0) or 0)
            except Exception:
                index = 0
            try:
                raw_section_index = obj.get('trlau_cine_section_index', obj.get('trlau_unitdata_cine_section_index', -1))
                section_index = -1 if raw_section_index is None else int(raw_section_index)
            except Exception:
                section_index = -1
            try:
                section_id = int(obj.get('trlau_section_original_id', obj.get('trlau_unitdata_cine_section_id', 0)) or 0)
            except Exception:
                section_id = 0
            try:
                target_offset = int(obj.get('trlau_cine_payload_offset', obj.get('trlau_unitdata_cine_target_offset', 0)) or 0)
            except Exception:
                target_offset = 0
            name = str(obj.get('trlau_cine_name', obj.get('trlau_unitdata_cine_name', '')) or '')
            entries.append({
                'index': index,
                'section_index': section_index,
                'section_id': section_id,
                'name': name,
                'target_offset': target_offset,
            })
        entries.sort(key=lambda entry: (int(entry.get('index', 0) or 0), str(entry)))
        return entries

    @staticmethod
    def _gather_combat_data_metadata(collection: Collection, root) -> Optional[dict[str, object]]:
        combat_obj = EXPORT_SCENE_OT_trlau_level._find_component_empty(
            collection,
            root,
            'CombatData',
            'trlau_type',
            'CombatData',
            'trlau_combat_data_empty',
        )
        if combat_obj is None:
            return None
        try:
            chunk_count = int(combat_obj.get('trlau_combat_blob_chunk_count', 0) or 0)
        except Exception:
            chunk_count = 0
        if chunk_count <= 0:
            return None
        encoded = ''.join(str(combat_obj.get(f'{COMBAT_BLOB_PROP_PREFIX}{index:04d}', '') or '') for index in range(chunk_count))
        if not encoded:
            return None
        try:
            blob = base64.b64decode(encoded.encode('ascii'))
        except Exception:
            logger.warning('Failed to decode embedded combat metadata for %s', getattr(combat_obj, 'name', '<unknown>'))
            return None
        graph = decode_combat_graph(blob)
        if not graph:
            logger.warning('Embedded combat metadata for %s is invalid', getattr(combat_obj, 'name', '<unknown>'))
            return None
        if int(graph.get('version', 0) or 0) < 3:
            raise ValueError(
                'This scene contains legacy embedded combat data that does not include PMarker and global spline-camera links. '
                'Re-import the original DRM with the current addon before exporting.'
            )
        editor_data = trlau_level_combat_property_to_dict(combat_obj)
        patch_combat_graph(graph, editor_data)
        return graph

    @staticmethod
    def _gather_unit_data_metadata(collection: Collection, root) -> Optional[dict[str, object]]:
        unit_obj = EXPORT_SCENE_OT_trlau_level._find_component_empty(collection, root, 'UnitData', 'trlau_type', 'UnitData', 'trlau_unitdata_empty')
        if unit_obj is None:
            return None
        metadata: dict[str, object] = {}
        ignored_scalar_keys = {
            'num_fsfx',
            'num_event_variable_storage',
            'p_fsfx',
            'p_event_variable_storage',
            'num_cines',
            'p_cines',
            'cine_entries',
        }
        for key in unit_obj.keys():
            if not key.startswith('trlau_unitdata_'):
                continue
            field_name = key[len('trlau_unitdata_'):]
            if field_name in ignored_scalar_keys:
                continue
            metadata[field_name] = unit_obj.get(key)
        metadata['fsfx_links'] = EXPORT_SCENE_OT_trlau_level._gather_unitdata_child_entries(
            collection,
            unit_obj,
            'UnitDataFSFXLink',
            'trlau_unitdata_fsfx_link_empty',
            'trlau_unitdata_fsfx_',
            (
                ('index', 'index', 0),
                ('id', 'id', 0),
                ('alpha', 'alpha', 0.0),
                ('enabled', 'enabled', 0),
            ),
        )
        metadata['cine_entries'] = EXPORT_SCENE_OT_trlau_level._gather_unitdata_cine_entries(
            collection,
            unit_obj,
        )
        metadata['event_variable_storage_entries'] = EXPORT_SCENE_OT_trlau_level._gather_unitdata_child_entries(
            collection,
            unit_obj,
            'UnitDataEventVariableStorage',
            'trlau_unitdata_event_variable_empty',
            'trlau_unitdata_event_variable_',
            (
                ('index', 'index', 0),
                ('variable', 'variable', 0),
                ('medium_value', 'medium_value', 0),
                ('easy_value', 'easy_value', 0),
                ('hard_value', 'hard_value', 0),
            ),
        )
        return metadata or None

    @staticmethod
    def _find_section_reference_empty(collection: Collection, root, role: str, parent=None):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not bool(obj.get('trlau_section_metadata_empty')):
                continue
            if str(obj.get('trlau_section_role', '') or '') != str(role):
                continue
            if parent is not None:
                if getattr(obj, 'parent', None) != parent:
                    continue
            elif not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            return obj
        return None

    @staticmethod
    def _find_intro_sound_empty(collection: Collection, parent):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if getattr(obj, 'parent', None) != parent:
                continue
            if trlau_intro_is_sound_data(obj) or str(obj.get('trlau_type', '') or '') == 'IntroSound' or str(getattr(obj, 'name', '') or '').endswith('_Sound'):
                return obj
        return None

    @staticmethod
    def _intro_sound_id(sound_obj, default: int = 0) -> int:
        if sound_obj is None:
            return int(default or 0)
        try:
            data = trlau_intro_sound_data_to_dict(sound_obj)
            value = int(data.get('sound_id', 0) or 0) if data else 0
        except Exception:
            value = 0
        return value if value > 0 else int(default or 0)

    @staticmethod
    def _decode_section_blob(obj) -> bytes:
        chunk_count = int(obj.get('trlau_section_blob_chunk_count', 0) or 0)
        if chunk_count <= 0:
            return b''
        encoded = ''.join(str(obj.get(f'{SECTION_BLOB_PROP_PREFIX}{index:04d}', '') or '') for index in range(chunk_count))
        if not encoded:
            return b''
        try:
            return base64.b64decode(encoded.encode('ascii'))
        except Exception:
            logger.warning('Failed to decode section blob metadata for %s', getattr(obj, 'name', '<unknown>'))
            return b''

    @staticmethod
    def _decode_terrain_light_grid_cells(obj) -> list[tuple[int, tuple[int, ...]]]:
        chunk_count = int(obj.get('trlau_terrain_light_grid_blob_chunk_count', 0) or 0)
        if chunk_count <= 0:
            return []
        encoded = ''.join(str(obj.get(f'{TERRAIN_LIGHT_GRID_BLOB_PROP_PREFIX}{index:04d}', '') or '') for index in range(chunk_count))
        if not encoded:
            return []
        try:
            blob = base64.b64decode(encoded.encode('ascii'))
        except Exception:
            logger.warning('Failed to decode TerrainLightGrid metadata for %s', getattr(obj, 'name', '<unknown>'))
            return []
        if len(blob) < 8 or blob[:4] != b'TLG1':
            return []
        cells: list[tuple[int, tuple[int, ...]]] = []
        try:
            count = struct.unpack_from('<I', blob, 4)[0]
            cursor = 8
            for _ in range(min(int(count), 1024)):
                if cursor + 4 > len(blob):
                    break
                cell_index, value_count = struct.unpack_from('<HH', blob, cursor)
                cursor += 4
                if cursor + (int(value_count) * 2) > len(blob):
                    break
                values = tuple(int(v) for v in struct.unpack_from('<' + ('H' * int(value_count)), blob, cursor)) if value_count else ()
                cursor += int(value_count) * 2
                if 0 <= int(cell_index) < 1024 and values:
                    cells.append((int(cell_index), values))
        except Exception:
            logger.warning('Failed to parse TerrainLightGrid metadata for %s', getattr(obj, 'name', '<unknown>'))
            return []
        return cells

    @staticmethod
    def _parse_passthrough_section_blob(
        blob: bytes,
        export_name: str,
        override_section_id: Optional[int] = None,
        original_section_index: int = -1,
    ) -> Optional[ExportPassthroughSection]:
        if len(blob) < 24 or blob[:4] not in {b'SECT', b'DRM\x00'}:
            return None
        try:
            total_size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from('<iBBHIII', blob, 4)
        except struct.error:
            return None
        section_size = max(0, int(total_size))
        num_relocations = max(0, int((packed_data >> 8) & 0x00FFFFFF))
        info_size = 24 + (num_relocations * 8)
        if len(blob) < info_size:
            return None
        payload_end = info_size + section_size
        if payload_end <= len(blob):
            data_end = payload_end
        elif 0 < section_size <= len(blob):
            data_end = section_size
        else:
            data_end = len(blob)
        relocations: list[tuple[int, int, int]] = []
        reloc_offset = 24
        for _ in range(num_relocations):
            try:
                type_and_section_info, type_specific, offset = struct.unpack_from('<HhI', blob, reloc_offset)
            except struct.error:
                return None
            relocations.append((int(type_and_section_info), int(type_specific), int(offset)))
            reloc_offset += 8
        content = bytes(blob[info_size:data_end])
        return ExportPassthroughSection(
            name=str(export_name),
            data=content,
            section_type=int(section_type),
            section_id=int(section_id if override_section_id is None else override_section_id),
            skip_flags=int(skip_flags),
            version_id=int(version_id),
            has_debug_info=int(packed_data & 0x1),
            resource_type=int((packed_data >> 1) & 0x7F),
            spec_mask=int(spec_mask),
            original_section_index=int(original_section_index),
            relocations=relocations,
        )

    @staticmethod
    def _object_prop(obj, key: str, default=None):
        if obj is None:
            return default
        value = default
        has_registered = False
        try:
            if hasattr(obj, key):
                value = getattr(obj, key)
                has_registered = True
        except Exception:
            value = default
        try:
            if key in obj:
                if (not has_registered) or value in (None, '', 0, 0.0, False):
                    return obj.get(key, default)
        except Exception:
            pass
        return value

    @staticmethod
    def _int_prop(obj, key: str, default: int = 0) -> int:
        try:
            return int(EXPORT_SCENE_OT_trlau_level._object_prop(obj, key, default) if obj is not None else default)
        except Exception:
            return int(default or 0)

    @staticmethod
    def _float_prop(obj, key: str, default: float = 0.0) -> float:
        try:
            return float(EXPORT_SCENE_OT_trlau_level._object_prop(obj, key, default) if obj is not None else default)
        except Exception:
            return float(default or 0.0)

    @staticmethod
    def _sfx_u8(value: object) -> int:
        try:
            return max(0, min(0xFF, int(value)))
        except Exception:
            return 0

    @staticmethod
    def _sfx_i8(value: object) -> int:
        try:
            value = int(value)
        except Exception:
            value = 0
        return max(-0x80, min(0x7F, value))

    @staticmethod
    def _sfx_u16(value: object) -> int:
        try:
            return max(0, min(0xFFFF, int(value)))
        except Exception:
            return 0

    @staticmethod
    def _sfx_i32(value: object) -> int:
        try:
            value = int(value)
        except Exception:
            value = 0
        return max(-0x80000000, min(0x7FFFFFFF, value))

    @staticmethod
    def _sfx_u32(value: object) -> int:
        try:
            return max(0, min(0xFFFFFFFF, int(value)))
        except Exception:
            return 0

    @staticmethod
    def _build_intro_sound_passthrough_section(obj, export_name: str, override_section_id: Optional[int] = None) -> Optional[ExportPassthroughSection]:
        has_sfx_metadata = bool(obj is not None and (EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_calltype', 0) or EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_filename', '') or 'trlau_sfx_calltype' in obj or 'trlau_sfx_filename' in obj))
        if not has_sfx_metadata:
            return EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(obj, export_name, override_section_id=override_section_id)

        raw_section = EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(obj, export_name, override_section_id=override_section_id)
        sound_data = trlau_intro_sound_data_to_dict(obj)
        section_type = int(sound_data.get('section_type', raw_section.section_type if raw_section is not None else 0) if sound_data else (raw_section.section_type if raw_section is not None else 0))
        section_id = int(override_section_id if override_section_id is not None else (sound_data.get('sound_id', raw_section.section_id if raw_section is not None else 0) if sound_data else (raw_section.section_id if raw_section is not None else 0)))
        skip_flags = int(sound_data.get('skip_flags', raw_section.skip_flags if raw_section is not None else 0) if sound_data else (raw_section.skip_flags if raw_section is not None else 0))
        version_id = int(sound_data.get('version_id', raw_section.version_id if raw_section is not None else 0) if sound_data else (raw_section.version_id if raw_section is not None else 0))
        has_debug_info = int(sound_data.get('has_debug_info', raw_section.has_debug_info if raw_section is not None else 0) if sound_data else (raw_section.has_debug_info if raw_section is not None else 0))
        resource_type = int(sound_data.get('resource_type', raw_section.resource_type if raw_section is not None else 0) if sound_data else (raw_section.resource_type if raw_section is not None else 0))
        spec_mask = int(sound_data.get('spec_mask', raw_section.spec_mask if raw_section is not None else 0xFFFFFFFF) if sound_data else (raw_section.spec_mask if raw_section is not None else 0xFFFFFFFF))

        armode = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_armode', 0)) & 0x1
        ar = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_ar', 0)) & 0x7F
        dr = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_dr', 0)) & 0x0F
        sl = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_sl', 0)) & 0x0F
        adsr0 = armode | (ar << 1) | (dr << 8) | (sl << 12)

        srmode = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_srmode', 0)) & 0x1
        srsign = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_srsign', 0)) & 0x1
        srpad = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_srpad', 0)) & 0x1
        rr = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_rr', 0)) & 0x1F
        sr = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_sr', 0)) & 0x7F
        rrmode = EXPORT_SCENE_OT_trlau_level._sfx_u16(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_rrmode', 0)) & 0x1
        adsr1 = srmode | (srsign << 1) | (srpad << 2) | (rr << 3) | (sr << 8) | (rrmode << 15)

        data = bytearray()
        data.extend(struct.pack('<i', EXPORT_SCENE_OT_trlau_level._sfx_i32(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_calltype', 0))))
        data.extend(struct.pack(
            _SFX_STREAM_STRUCT,
            0,
            EXPORT_SCENE_OT_trlau_level._sfx_u8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_priority', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_u8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_volume', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_u8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_volume_variation', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_u8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_mode', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_u32(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_sound_group', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_u8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_delay', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_i8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_loops', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_u8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_cur_voices', 0)),
            EXPORT_SCENE_OT_trlau_level._float_prop(obj, 'trlau_sfx_note', 0.0),
            adsr0,
            adsr1,
            EXPORT_SCENE_OT_trlau_level._float_prop(obj, 'trlau_sfx_pitch_variation', 0.0),
            EXPORT_SCENE_OT_trlau_level._float_prop(obj, 'trlau_sfx_initial_delay', 0.0),
            EXPORT_SCENE_OT_trlau_level._float_prop(obj, 'trlau_sfx_initial_delay_variation', 0.0),
            EXPORT_SCENE_OT_trlau_level._sfx_u32(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_max_distance', 0)),
            EXPORT_SCENE_OT_trlau_level._sfx_i8(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_subtitlemode', 0)),
        ))

        min_name_offset = len(data)
        name_offset = EXPORT_SCENE_OT_trlau_level._int_prop(obj, 'trlau_sfx_name_offset', _SFX_DEFAULT_NAME_OFFSET)
        if name_offset < min_name_offset:
            name_offset = (min_name_offset + 3) & ~3
        if len(data) < name_offset:
            data.extend(b'\x00' * (int(name_offset) - len(data)))
        filename = str(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_filename', '') or '')
        data.extend(filename.encode('utf-8', errors='ignore') + b'\x00')

        return ExportPassthroughSection(
            name=str(export_name),
            data=bytes(data),
            section_type=int(section_type),
            section_id=int(section_id),
            skip_flags=int(skip_flags),
            version_id=int(version_id),
            has_debug_info=int(has_debug_info),
            resource_type=int(resource_type),
            spec_mask=int(spec_mask),
            relocations=[],
            pointer_relocations=[(_SFX_NAME_POINTER_OFFSET, str(export_name), int(name_offset))],
        )

    @staticmethod
    def _parse_passthrough_section(obj, export_name: str, override_section_id: Optional[int] = None) -> Optional[ExportPassthroughSection]:
        blob = EXPORT_SCENE_OT_trlau_level._decode_section_blob(obj)
        original_section_index = EXPORT_SCENE_OT_trlau_level._int_prop(obj, 'trlau_section_original_index', -1)
        if original_section_index < 0:
            # Backward compatibility for scenes imported before the explicit
            # property existed. Extracted section files are named INDEX_ID.ext.
            file_name = str(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_section_file_name', '') or '')
            try:
                original_section_index = int(Path(file_name).stem.split('_', 1)[0])
            except Exception:
                original_section_index = -1
        section = EXPORT_SCENE_OT_trlau_level._parse_passthrough_section_blob(
            blob,
            export_name,
            override_section_id=override_section_id,
            original_section_index=original_section_index,
        )
        if section is not None and has_fsfx_render_effect_properties(obj):
            try:
                section.data = build_fsfx_render_effect_content_from_object(obj, section.data)
            except Exception as exc:
                logger.warning('Failed to build edited FSFX section for %s: %s', getattr(obj, 'name', '<unknown>'), exc)
        return section

    @staticmethod
    def _cine_value(cine_obj, section_obj, parent_key: str, section_key: str, default=None):
        for obj, key in ((cine_obj, parent_key), (section_obj, section_key)):
            try:
                if obj is not None and key in obj:
                    return obj.get(key)
            except Exception:
                pass
        return default

    @staticmethod
    def _build_cine_passthrough_section(cine_obj, section_obj, export_name: str) -> Optional[ExportPassthroughSection]:
        section = EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(section_obj, export_name, override_section_id=None)
        if section is None:
            return None
        structured = EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_structured', 'trlau_cine_structured', 0)
        try:
            structured = int(structured or 0) != 0
        except Exception:
            structured = False
        if not structured:
            return section
        rebuildable = EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_rebuildable', 'trlau_cine_rebuildable', 0)
        try:
            rebuildable = int(rebuildable or 0) != 0
        except Exception:
            rebuildable = False
        if not rebuildable:
            return section

        command_type = str(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_type', 'trlau_cine_command_type', 'Cinematic') or 'Cinematic')
        data_size = EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_data_size', 'trlau_cine_command_data_size', 0)
        try:
            data_size = int(data_size or 0)
        except Exception:
            data_size = 0
        if command_type != 'Cinematic' or data_size != 0:
            logger.warning('Cine section %s uses unsupported command type/data block (%s, data_size=%s); exporting original raw section.', getattr(section_obj, 'name', '<unknown>'), command_type, data_size)
            return section

        try:
            meta = {
                'name': str(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_name', 'trlau_cine_name', '') or ''),
                'version_major': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_version_major', 'trlau_cine_version_major', 1) or 1),
                'version_minor': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_version_minor', 'trlau_cine_version_minor', 2) or 2),
                'stream_unit_id': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_stream_unit_id', 'trlau_cine_stream_unit_id', 0) or 0),
                'cine_id': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_id', 'trlau_cine_id', 0) or 0),
                'end_time': float(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_end_time', 'trlau_cine_end_time', 0.0) or 0.0),
                'master_command_time': 0.0,
                'cinematic_command': {
                    'cinematic_name': str(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_cinematic_name', 'trlau_cine_command_cinematic_name', '') or ''),
                    'unit_id': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_unit_id', 'trlau_cine_command_unit_id', 0) or 0),
                    'load': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_load', 'trlau_cine_command_load', 0) or 0),
                    'camera_control': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_camera_control', 'trlau_cine_command_camera_control', 0) or 0),
                    'channels': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_channels', 'trlau_cine_command_channels', 0) or 0),
                    'positions_after_playback': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_positions_after_playback', 'trlau_cine_command_positions_after_playback', 0) or 0),
                    'end_trigger_id': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_end_trigger_id', 'trlau_cine_command_end_trigger_id', 0) or 0),
                    'data_pointer': int(EXPORT_SCENE_OT_trlau_level._cine_value(cine_obj, section_obj, 'trlau_unitdata_cine_command_data_pointer', 'trlau_cine_command_data_pointer', 0) or 0),
                    'data_size': data_size,
                },
            }
            if not meta['name']:
                meta['name'] = meta['cinematic_command']['cinematic_name'] or 'Cinematic'
            if not meta['cinematic_command']['cinematic_name']:
                meta['cinematic_command']['cinematic_name'] = meta['name']
            section.data = build_cine_payload_from_metadata(meta)
            section.section_type = 0
            section.section_id = 0
            section.relocations = []
        except Exception as exc:
            logger.warning('Failed to rebuild structured Cine section %s; exporting original raw section: %s', getattr(section_obj, 'name', '<unknown>'), exc)
        return section

    @staticmethod
    def _gather_texture_passthrough_sections(collection: Collection, root, selected_names: Optional[set[str]] = None) -> List[ExportPassthroughSection]:
        clear_texture_export_cache()
        sections: List[ExportPassthroughSection] = []
        seen_texture_ids: set[int] = set()
        exported_sources: dict[int, str] = {}

        def _append_textures_from_mesh(strip_obj, owner_name: str) -> None:
            if selected_names is not None and strip_obj.name not in selected_names and owner_name not in selected_names:
                return
            used_materials = EXPORT_SCENE_OT_trlau_level._iter_used_strip_materials(EXPORT_SCENE_OT_trlau_level, strip_obj)
            if not used_materials:
                material = EXPORT_SCENE_OT_trlau_level._resolve_strip_material(strip_obj)
                used_materials = [] if material is None else [(0, material)]

            for material_index, material in used_materials:
                flags = decode_tpage_flags(get_material_tpageid(material))
                texture_id = int(flags.get('texture_id', 0) or 0)
                if texture_id <= 0:
                    continue

                image = EXPORT_SCENE_OT_trlau_level._find_material_image_texture(material)
                if image is None:
                    logger.warning(
                        'Skipping texture export for material %s (slot=%s, texture_id=0x%X): no Image Texture node with an image was found',
                        getattr(material, 'name', '<unnamed>'),
                        material_index,
                        texture_id,
                    )
                    continue

                image_name = str(getattr(image, 'name', '<unnamed>'))
                if texture_id in seen_texture_ids:
                    previous = exported_sources.get(texture_id)
                    if previous and previous != image_name:
                        logger.warning(
                            'Texture ID 0x%X is used by multiple images (%s, %s). Keeping the first export.',
                            texture_id,
                            previous,
                            image_name,
                        )
                    continue

                try:
                    pcd_data = image_to_pcd_bytes(image, texture_id=texture_id, material_name=getattr(material, 'name', ''))
                except Exception as exc:
                    raise RuntimeError(
                        f'Failed to export texture 0x{texture_id:X} for material {getattr(material, "name", "<unnamed>")}: {exc}'
                    ) from exc

                seen_texture_ids.add(texture_id)
                exported_sources[texture_id] = image_name
                sections.append(ExportPassthroughSection(
                    name=f'texture_{texture_id:04x}',
                    data=pcd_data,
                    section_type=5,
                    section_id=texture_id,
                ))

        for group_empty in EXPORT_SCENE_OT_trlau_level._iter_group_objects(collection, root):
            if selected_names is not None and group_empty.name not in selected_names:
                child_selected = any(child.name in selected_names for child in group_empty.children_recursive)
                if not child_selected:
                    continue

            strip_meshes = EXPORT_SCENE_OT_trlau_level._iter_group_strip_meshes(
                EXPORT_SCENE_OT_trlau_level._group_collection(group_empty),
                group_empty,
            )
            for strip_obj in strip_meshes:
                _append_textures_from_mesh(strip_obj, group_empty.name)

        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if root is not None and obj != root and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            if not EXPORT_SCENE_OT_trlau_level._is_bgobject_mesh(obj):
                continue
            parent = getattr(obj, 'parent', None)
            owner_name = getattr(parent, 'name', '') if parent is not None else getattr(obj, 'name', '')
            _append_textures_from_mesh(obj, owner_name)

        return sections

    @staticmethod
    def _find_material_image_texture(material):
        node_tree = getattr(material, 'node_tree', None)
        if node_tree is None:
            return None

        nodes = list(getattr(node_tree, 'nodes', []) or [])
        active = getattr(node_tree.nodes, 'active', None) if getattr(node_tree, 'nodes', None) is not None else None
        if active is not None and getattr(active, 'type', '') == 'TEX_IMAGE' and getattr(active, 'image', None) is not None:
            return active.image

        linked_images = []
        unlinked_images = []
        for node in nodes:
            if getattr(node, 'type', '') != 'TEX_IMAGE':
                continue
            image = getattr(node, 'image', None)
            if image is None:
                continue
            outputs = tuple(getattr(node, 'outputs', ()) or ())
            if any(bool(getattr(output, 'is_linked', False)) for output in outputs):
                linked_images.append(image)
            else:
                unlinked_images.append(image)

        if linked_images:
            return linked_images[0]
        if unlinked_images:
            return unlinked_images[0]
        return None

    @staticmethod
    def _area_dbase_pointer_offsets_for_header(header_content_offset: Optional[int]) -> tuple[Optional[int], Optional[int], Optional[int]]:
        try:
            header_offset = int(header_content_offset) if header_content_offset is not None else 0x10
        except Exception:
            header_offset = 0x10
        if header_offset < 0:
            header_offset = 0x10
        return 0, header_offset, None

    @staticmethod
    def _find_area_dbase_empty(collection: Collection, root):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        candidates = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            name = str(getattr(obj, 'name', '') or '')
            if name == 'AreaDBase' or name.endswith('_AreaDBase'):
                candidates.append(obj)
        if not candidates:
            return None
        candidates.sort(key=lambda obj: (0 if str(getattr(obj, 'name', '') or '').endswith('_AreaDBase') else 1, str(getattr(obj, 'name', '') or '')))
        return candidates[0]

    @staticmethod
    def _find_area_dbase_mesh(area_obj):
        if area_obj is None:
            return None
        candidates = []
        try:
            candidates.extend(list(getattr(area_obj, 'children', []) or []))
        except Exception:
            pass
        try:
            for obj in bpy.data.objects:
                if obj in candidates:
                    continue
                if 'AreaDBase_Areas' in str(getattr(obj, 'name', '')):
                    candidates.append(obj)
        except Exception:
            pass
        for obj in candidates:
            if getattr(obj, 'type', '') == 'MESH' and 'AreaDBase_Areas' in str(getattr(obj, 'name', '')):
                return obj
        for obj in candidates:
            if getattr(obj, 'type', '') == 'MESH':
                return obj
        return None

    @staticmethod
    def _area_dbase_polygons_from_mesh(area_obj, mesh_obj) -> list[list[tuple[float, float, float]]]:
        if area_obj is None or mesh_obj is None or getattr(mesh_obj, 'type', '') != 'MESH':
            return []
        mesh = getattr(mesh_obj, 'data', None)
        if mesh is None:
            return []
        try:
            to_area_local = area_obj.matrix_world.inverted() @ mesh_obj.matrix_world
        except Exception:
            to_area_local = mathutils.Matrix.Identity(4)
        polygons: list[list[tuple[float, float, float]]] = []
        vertices = list(getattr(mesh, 'vertices', []) or [])
        for polygon in list(getattr(mesh, 'polygons', []) or []):
            face: list[tuple[float, float, float]] = []
            for vertex_index in list(getattr(polygon, 'vertices', []) or []):
                try:
                    co = to_area_local @ vertices[int(vertex_index)].co
                    face.append((float(co.x), float(co.y), float(co.z)))
                except Exception:
                    continue
            if len(face) >= 3:
                polygons.append(face)
        return polygons

    @staticmethod
    def _area_dbase_export_name(collection: Collection, root, area_obj) -> str:
        for value in (
            getattr(root, 'get', lambda _k, _d=None: None)('trlau_drm_name', None) if root is not None else None,
            getattr(collection, 'name', ''),
            getattr(area_obj, 'name', '') if area_obj is not None else '',
        ):
            name = str(value or '').strip()
            if not name:
                continue
            if name.endswith('_Level'):
                name = name[:-6]
            if name.endswith('_AreaDBase'):
                name = name[:-10]
            if name:
                return name[:23]
        return 'AreaDBase'

    @staticmethod
    def _build_area_dbase_passthrough_section(collection: Collection, root, metadata: dict[str, object]) -> Optional[ExportPassthroughSection]:
        area_obj = EXPORT_SCENE_OT_trlau_level._find_area_dbase_empty(collection, root)
        if area_obj is None:
            return None

        use_embedded_source = bool(area_obj.get('trlau_area_dbase_use_embedded_source', True))
        embedded_blob = EXPORT_SCENE_OT_trlau_level._decode_section_blob(area_obj) if use_embedded_source else b''
        if embedded_blob:
            embedded_section = EXPORT_SCENE_OT_trlau_level._parse_passthrough_section_blob(
                embedded_blob,
                'area_dbase',
            )
            if embedded_section is None:
                logger.warning('Embedded AreaDBase section on %s is invalid; rebuilding from the mesh instead', getattr(area_obj, 'name', '<unknown>'))
            elif embedded_section.relocations:
                logger.warning(
                    'Embedded AreaDBase section on %s has %d unresolved relocation(s); rebuilding from the mesh instead',
                    getattr(area_obj, 'name', '<unknown>'),
                    len(embedded_section.relocations),
                )
            else:
                try:
                    move_offset = int(metadata.get('areaDBaseMoveDataOffset', 0))
                except Exception:
                    move_offset = 0
                try:
                    planner_offset = int(metadata.get('areaDBasePlannerDataOffset', 0x10))
                except Exception:
                    planner_offset = 0x10
                try:
                    runtime_area_dbase_offset = int(metadata.get('areaDBaseRuntimeObjectOffset', -1))
                except Exception:
                    runtime_area_dbase_offset = -1
                try:
                    header_content_offset = int(metadata.get('areaDBaseHeaderContentOffset', planner_offset))
                except Exception:
                    header_content_offset = planner_offset
                metadata['areaDBaseSectionName'] = str(embedded_section.name)
                metadata['areaDBaseMoveDataOffset'] = int(move_offset)
                metadata['areaDBasePlannerDataOffset'] = int(planner_offset)
                metadata['areaDBaseRuntimeObjectOffset'] = int(runtime_area_dbase_offset)
                metadata['areaDBaseHeaderContentOffset'] = int(header_content_offset)
                logger.info(
                    'Exporting lossless embedded AreaDBase section from %s (%d bytes)',
                    getattr(area_obj, 'name', '<unknown>'),
                    len(embedded_section.data),
                )
                return embedded_section

        mesh_obj = EXPORT_SCENE_OT_trlau_level._find_area_dbase_mesh(area_obj)
        if mesh_obj is None:
            logger.warning('Skipping AreaDBase export: %s has no AreaDBase_Areas mesh', getattr(area_obj, 'name', '<unknown>'))
            return None

        polygons = EXPORT_SCENE_OT_trlau_level._area_dbase_polygons_from_mesh(area_obj, mesh_obj)
        if not polygons:
            logger.warning('Skipping AreaDBase export: %s contains no exportable area polygons', getattr(mesh_obj, 'name', '<unknown>'))
            return None

        center = convert_area_dbase_center_from_blender_position(
            float(getattr(area_obj, 'location', (0.0, 0.0, 0.0))[0]),
            float(getattr(area_obj, 'location', (0.0, 0.0, 0.0))[1]),
            float(getattr(area_obj, 'location', (0.0, 0.0, 0.0))[2]),
        )
        export_name = EXPORT_SCENE_OT_trlau_level._area_dbase_export_name(collection, root, area_obj)
        try:
            content, header_content_offset = build_area_dbase_section_content_from_polygons(
                export_name,
                center,
                polygons,
                prefix_size=0x10,
            )
        except Exception as exc:
            logger.warning('Skipping AreaDBase export: failed to generate section from %s: %s', getattr(mesh_obj, 'name', '<unknown>'), exc)
            return None

        section = ExportPassthroughSection(
            name='area_dbase',
            data=bytes(content),
            section_type=0,
            section_id=0,
            skip_flags=0,
            version_id=0,
            has_debug_info=0,
            resource_type=0,
            spec_mask=0xFFFFFFFF,
            relocations=[],
        )

        _move_offset, planner_offset, runtime_area_dbase_offset = EXPORT_SCENE_OT_trlau_level._area_dbase_pointer_offsets_for_header(header_content_offset)
        metadata['areaDBaseSectionName'] = str(section.name)
        metadata['areaDBaseMoveDataOffset'] = int(_move_offset) if _move_offset is not None else -1
        metadata['areaDBasePlannerDataOffset'] = int(planner_offset) if planner_offset is not None else int(header_content_offset)
        metadata['areaDBaseRuntimeObjectOffset'] = int(runtime_area_dbase_offset) if runtime_area_dbase_offset is not None else -1
        metadata['areaDBaseHeaderContentOffset'] = int(header_content_offset)
        return section

    @staticmethod
    def _find_unitdata_fsfx_link_empty(collection: Collection, unit_obj, index: int, fsfx_id: int = 0):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if unit_obj is not None and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, unit_obj):
                continue
            if not bool(obj.get('trlau_unitdata_fsfx_link_empty')):
                continue
            try:
                raw_index = obj.get('trlau_unitdata_fsfx_index', -1)
                obj_index = -1 if raw_index is None else int(raw_index)
            except Exception:
                obj_index = -1
            if obj_index == int(index):
                return obj
        return None

    @staticmethod
    def _find_unitdata_fsfx_section_empty(collection: Collection, fsfx_link_obj):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not bool(obj.get('trlau_section_metadata_empty')):
                continue
            if getattr(obj, 'parent', None) != fsfx_link_obj:
                continue
            if str(obj.get('trlau_section_role', '') or '') == 'FSFXLink':
                return obj
        return None

    @staticmethod
    def _find_unitdata_cine_empty(collection: Collection, unit_obj, index: int, section_index: int = -1, section_id: int = 0):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if unit_obj is not None and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, unit_obj):
                continue
            if not bool(obj.get('trlau_unitdata_cine_empty')):
                continue
            try:
                raw_index = obj.get('trlau_unitdata_cine_index', -1)
                obj_index = -1 if raw_index is None else int(raw_index)
            except Exception:
                obj_index = -1
            if obj_index == int(index):
                return obj
        return None

    @staticmethod
    def _find_unitdata_cine_section_empty(collection: Collection, cine_obj):
        if cine_obj is None:
            return None
        try:
            if (
                str(cine_obj.get('trlau_section_role', '') or '') == 'Cine'
                or 'trlau_cine_structured' in cine_obj
                or 'trlau_cine_name' in cine_obj
                or int(cine_obj.get('trlau_section_blob_chunk_count', 0) or 0) > 0
            ):
                return cine_obj
        except Exception:
            pass
        objects = getattr(collection, 'all_objects', None) or collection.objects
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not bool(obj.get('trlau_section_metadata_empty')):
                continue
            if getattr(obj, 'parent', None) != cine_obj:
                continue
            if str(obj.get('trlau_section_role', '') or '') == 'Cine':
                return obj
        return None

    @staticmethod
    def _gather_passthrough_sections(collection: Collection, root, metadata: dict[str, object], unit_data: Optional[dict[str, object]], *, export_area_dbase: bool = True) -> List[ExportPassthroughSection]:
        sections: List[ExportPassthroughSection] = []
        seen_keys: set[tuple[int, int, str]] = set()

        def _append_unique(section: Optional[ExportPassthroughSection]) -> None:
            if section is None:
                return
            section_id = int(section.section_id)
            name_key = str(section.name or '') if section_id <= 0 else ''
            key = (int(section.section_type), section_id, name_key)
            if key in seen_keys:
                return
            seen_keys.add(key)
            sections.append(section)

        reloc_module_obj = EXPORT_SCENE_OT_trlau_level._find_section_reference_empty(collection, root, 'RelocModule', parent=root)
        if reloc_module_obj is not None:
            reloc_module_id = EXPORT_SCENE_OT_trlau_level._int_prop(reloc_module_obj, 'trlau_section_original_id', 0)
            reloc_module_section = EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(
                reloc_module_obj,
                'reloc_module',
                override_section_id=reloc_module_id if reloc_module_id > 0 else None,
            )
            if reloc_module_section is not None:
                metadata['relocModuleSectionName'] = str(reloc_module_section.name)
                metadata['relocModuleTargetOffset'] = EXPORT_SCENE_OT_trlau_level._int_prop(reloc_module_obj, 'trlau_reloc_module_target_offset', 0)
                _append_unique(reloc_module_section)

        if export_area_dbase:
            _append_unique(EXPORT_SCENE_OT_trlau_level._build_area_dbase_passthrough_section(collection, root, metadata))

        water_section_id = int(metadata.get('WaterFSFX', 0) or 0)
        if water_section_id > 0:
            water_obj = EXPORT_SCENE_OT_trlau_level._find_section_reference_empty(collection, root, 'WaterFSFX', parent=root)
            if water_obj is not None:
                _append_unique(EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(water_obj, 'water_fsfx', override_section_id=water_section_id))

        if unit_data:
            unit_obj = EXPORT_SCENE_OT_trlau_level._find_component_empty(collection, root, 'UnitData', 'trlau_type', 'UnitData', 'trlau_unitdata_empty')
            basecam_id = int(unit_data.get('base_camera_basecam', 0) or 0)
            if unit_obj is not None and basecam_id > 0:
                basecam_obj = EXPORT_SCENE_OT_trlau_level._find_section_reference_empty(collection, root, 'basecam', parent=unit_obj)
                if basecam_obj is not None:
                    _append_unique(EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(basecam_obj, 'basecam', override_section_id=basecam_id))

            for entry_index, entry in enumerate(list(unit_data.get('fsfx_links') or [])):
                if not isinstance(entry, dict):
                    continue
                try:
                    fsfx_id = int(entry.get('id', 0) or 0)
                except Exception:
                    fsfx_id = 0
                if fsfx_id <= 0:
                    continue
                try:
                    fsfx_index = int(entry.get('index', entry_index) or entry_index)
                except Exception:
                    fsfx_index = int(entry_index)
                fsfx_link_obj = EXPORT_SCENE_OT_trlau_level._find_unitdata_fsfx_link_empty(collection, unit_obj, fsfx_index, fsfx_id)
                if fsfx_link_obj is None:
                    continue
                fsfx_section_obj = EXPORT_SCENE_OT_trlau_level._find_unitdata_fsfx_section_empty(collection, fsfx_link_obj)
                if fsfx_section_obj is None:
                    logger.warning('FSFXLink id %s has no imported section metadata child; it will stay referenced but the section will not be embedded.', fsfx_id)
                    continue
                _append_unique(EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(fsfx_section_obj, f'fsfxlink_{fsfx_index:03d}', override_section_id=fsfx_id))

            for entry_index, entry in enumerate(list(unit_data.get('cine_entries') or [])):
                if not isinstance(entry, dict):
                    continue
                try:
                    cine_index = int(entry.get('index', entry_index) or entry_index)
                except Exception:
                    cine_index = int(entry_index)
                try:
                    raw_section_index = entry.get('section_index', -1)
                    cine_section_index = -1 if raw_section_index is None else int(raw_section_index)
                except Exception:
                    cine_section_index = -1
                try:
                    cine_section_id = int(entry.get('section_id', 0) or 0)
                except Exception:
                    cine_section_id = 0
                cine_obj = EXPORT_SCENE_OT_trlau_level._find_unitdata_cine_empty(collection, unit_obj, cine_index, cine_section_index, cine_section_id)
                if cine_obj is None:
                    continue
                cine_section_obj = EXPORT_SCENE_OT_trlau_level._find_unitdata_cine_section_empty(collection, cine_obj)
                if cine_section_obj is None:
                    logger.warning('Cine index %s has no imported CineData section payload; it will stay referenced but the section will not be embedded.', cine_index)
                    continue
                section = EXPORT_SCENE_OT_trlau_level._build_cine_passthrough_section(cine_obj, cine_section_obj, f'cine_{cine_index:03d}')
                if section is None:
                    continue
                entry['export_section_name'] = str(section.name)
                if cine_section_index >= 0:
                    entry['section_index'] = int(cine_section_index)
                _append_unique(section)

        objects = getattr(collection, 'all_objects', None) or collection.objects
        intro_empties = [
            obj for obj in objects
            if getattr(obj, 'type', None) == 'EMPTY'
            and (trlau_intro_is_intro_data(obj) or str(obj.get('trlau_type', '')) == 'IntroData')
            and EXPORT_SCENE_OT_trlau_level._intro_data_type_from_metadata(obj) == 17
        ]
        intro_empties.sort(key=lambda obj: EXPORT_SCENE_OT_trlau_level._intro_index_from_object(obj, 0))
        for intro_obj in intro_empties:
            specific_empty = EXPORT_SCENE_OT_trlau_level._child_intro_metadata_empty(intro_obj, specific=True, data_type=17)
            try:
                sound_id = int((trlau_intro_specific_data_to_dict(specific_empty, 17).get('sound_id', 0) if specific_empty is not None else 0) or 0)
            except Exception:
                sound_id = 0
            sound_obj = EXPORT_SCENE_OT_trlau_level._find_intro_sound_empty(collection, intro_obj)
            if sound_obj is None:
                continue
            if sound_id <= 0:
                sound_id = EXPORT_SCENE_OT_trlau_level._intro_sound_id(sound_obj, 0)
            if sound_id <= 0:
                continue
            intro_index = EXPORT_SCENE_OT_trlau_level._intro_index_from_object(intro_obj, 0)
            _append_unique(EXPORT_SCENE_OT_trlau_level._build_intro_sound_passthrough_section(sound_obj, f'intro_{intro_index:03d}_sound', override_section_id=sound_id))

        for section_obj in EXPORT_SCENE_OT_trlau_level._iter_sfx_audio_section_metadata(collection, root):
            role = str(section_obj.get('trlau_section_role', '') or '')
            section_id = EXPORT_SCENE_OT_trlau_level._int_prop(section_obj, 'trlau_section_original_id', 0)
            export_name = f'{role.lower()}_{section_id:04d}' if section_id > 0 else f'{role.lower()}_{len(sections):04d}'
            _append_unique(EXPORT_SCENE_OT_trlau_level._parse_passthrough_section(section_obj, export_name, override_section_id=section_id if section_id > 0 else None))

        return sections

    @staticmethod
    def _group_collection(group_empty):
        users_collection = tuple(getattr(group_empty, 'users_collection', ()) or ())
        if users_collection:
            return users_collection[0]
        return None

    @staticmethod
    def _iter_group_strip_meshes(collection: Optional[Collection], group_empty, drm_name: str = ''):
        group_index = EXPORT_SCENE_OT_trlau_level._get_group_index(group_empty)
        seen_ids: set[int] = set()
        result = []

        for child in group_empty.children:
            if child.type != 'MESH' or EXPORT_SCENE_OT_trlau_level._is_collision_mesh_for_group(child, int(group_index), drm_name):
                continue
            marker = id(child)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            result.append(child)

        objects = getattr(collection, 'all_objects', None) if collection is not None else None
        if objects is None:
            objects = getattr(bpy.context, 'scene', None)
            objects = getattr(objects, 'objects', []) if objects is not None else []

        for obj in objects:
            if getattr(obj, 'type', None) != 'MESH':
                continue
            if EXPORT_SCENE_OT_trlau_level._is_collision_mesh_for_group(obj, int(group_index), drm_name):
                continue
            if int(obj.get('trlau_drawgroup', -999999)) != int(group_index):
                continue
            marker = id(obj)
            if marker in seen_ids:
                continue
            seen_ids.add(marker)
            result.append(obj)

        result.sort(key=EXPORT_SCENE_OT_trlau_level._strip_sort_key)
        return result

    @staticmethod
    def _normalized_object_name(name: object) -> str:
        return _COLLISION_NAME_BLENDER_SUFFIX_RE.sub('', str(name or '').strip())

    @staticmethod
    def _is_collision_mesh_for_group(obj, group_index: int, drm_name: str = '') -> bool:
        if getattr(obj, 'type', None) != 'MESH':
            return False
        if bool(obj.get('trlau_terrain_collision')):
            return True

        name = EXPORT_SCENE_OT_trlau_level._normalized_object_name(getattr(obj, 'name', ''))
        if not name:
            return False

        try:
            group_index_int = int(group_index)
        except Exception:
            group_index_int = -1

        normalized_drm_name = str(drm_name or '').strip()
        if normalized_drm_name:
            for index_text in (str(group_index_int), f'{group_index_int:03d}'):
                if name in {
                    f'{normalized_drm_name}_TerrainGroup_{index_text}_Collision',
                    f'{normalized_drm_name}_Level_Group_{index_text}_Collision',
                }:
                    return True

        return False

    @staticmethod
    def _find_collision_empty(group_empty, collection: Optional[Collection] = None, drm_name: str = ''):
        group_index = EXPORT_SCENE_OT_trlau_level._get_group_index(group_empty)
        if any(
            child.type == 'MESH' and EXPORT_SCENE_OT_trlau_level._is_collision_mesh_for_group(child, int(group_index), drm_name)
            for child in getattr(group_empty, 'children', []) or []
        ):
            return group_empty
        for child in group_empty.children:
            if child.type == 'EMPTY' and bool(child.get('trlau_terrain_collision_empty')):
                return child
        for child in group_empty.children:
            if child.type == 'EMPTY' and bool(child.get('trlau_terrain_collision')):
                return child
        for child in group_empty.children:
            if child.type == 'EMPTY' and 'Collision' in child.name:
                return child

        objects = getattr(collection, 'all_objects', None) if collection is not None else None
        if objects is None:
            objects = getattr(bpy.context, 'scene', None)
            objects = getattr(objects, 'objects', []) if objects is not None else []

        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if int(obj.get('trlau_terrain_group_index', -999999)) != int(group_index):
                continue
            if bool(obj.get('trlau_terrain_collision_empty')) or bool(obj.get('trlau_terrain_collision')):
                return obj

        return None

    @staticmethod
    def _find_collision_kd_empty(collision_empty, collection: Optional[Collection] = None):
        group_index = EXPORT_SCENE_OT_trlau_level._get_group_index(collision_empty)
        for child in collision_empty.children:
            if child.type == 'EMPTY' and bool(child.get('trlau_terrain_collision_kd')):
                return child

        objects = getattr(collection, 'all_objects', None) if collection is not None else None
        if objects is None:
            objects = getattr(bpy.context, 'scene', None)
            objects = getattr(objects, 'objects', []) if objects is not None else []

        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if int(obj.get('trlau_terrain_group_index', -999999)) != int(group_index):
                continue
            if bool(obj.get('trlau_terrain_collision_kd')):
                return obj

        return None

    @staticmethod
    def _relative_translation(obj, reference_matrix_inv=None) -> Tuple[float, float, float]:
        vector = obj.matrix_world.translation
        if reference_matrix_inv is not None:
            vector = reference_matrix_inv @ vector
        return (float(vector.x), float(vector.y), float(vector.z))

    @staticmethod
    def _group_sort_key(obj) -> tuple[int, str]:
        return EXPORT_SCENE_OT_trlau_level._get_group_index(obj), obj.name

    @staticmethod
    def _get_group_index(obj) -> int:
        if 'trlau_terrain_group_index' in obj:
            return int(obj.get('trlau_terrain_group_index', 0))
        match = _GROUP_NAME_RE.search(obj.name)
        if match is not None:
            return int(match.group(1))
        return -1

    @staticmethod
    def _strip_sort_key(obj) -> tuple[int, str]:
        match = _STRIP_NAME_RE.search(obj.name)
        if match is not None:
            return int(match.group(1)), obj.name
        return int(obj.get('trlau_material_index', 0)), obj.name

    @staticmethod
    def _vector_prop(obj, name: str, default=(0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
        value = obj.get(name, default)
        try:
            if isinstance(value, (str, bytes, dict)) or len(value) < 3:
                raise TypeError
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return (float(default[0]), float(default[1]), float(default[2]))

    @staticmethod
    def _int_list_prop(obj, name: str) -> list[int]:
        value = obj.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, dict)):
            return []
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except Exception:
                return []
        return result

    @staticmethod
    def _extract_scene_center_offset(root) -> Tuple[float, float, float, float]:
        try:
            loc = getattr(root, 'location', (0.0, 0.0, 0.0))
            x, y, z = float(loc[0]), float(loc[1]), float(loc[2])
        except Exception:
            x, y, z = 0.0, 0.0, 0.0
        return (x, y, z, 0.0)

    @staticmethod
    def _normalize_level_game(value: object) -> str:
        return normalize_game_value(value)

    @staticmethod
    def _level_game_from_root(root) -> str:
        for key in ('trlau_level_game_id',):
            try:
                if key in root:
                    return EXPORT_SCENE_OT_trlau_level._normalize_level_game(root.get(key))
            except Exception:
                continue
        return 'legend'

    @staticmethod
    def _gather_root_metadata(root) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for key in root.keys():
            if not key.startswith('trlau_level_'):
                continue
            field_name = key[len('trlau_level_'):]
            value = root.get(key)
            if field_name in {'objectNameList', 'game', 'game_id'}:
                continue
            metadata[field_name] = value
        if 'trlau_level_flags' in root:
            metadata['flags'] = int(root.get('trlau_level_flags', 0))
        if 'trlau_unit_flags' in root:
            metadata['unitFlags'] = int(root.get('trlau_unit_flags', 0))
        if 'trlau_stream_unit_id' in root:
            metadata['streamUnitID'] = int(root.get('trlau_stream_unit_id', 0))
        if 'trlau_player_object_id' in root:
            metadata['playerObjectID'] = int(root.get('trlau_player_object_id', -1))
        if 'trlau_drm_name' in root and 'worldName' not in metadata:
            metadata['worldName'] = str(root.get('trlau_drm_name', ''))
        portals_json = str(root.get('trlau_stream_unit_portals_json', '') or '')
        if portals_json:
            try:
                decoded_portals = json.loads(portals_json)
                if isinstance(decoded_portals, list):
                    metadata['streamUnitPortals'] = decoded_portals
            except Exception:
                logger.warning('Failed to decode trlau_stream_unit_portals_json; stream unit portals will not be exported')
        return metadata

    @staticmethod
    def _evaluated_mesh(obj):
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            return eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph), eval_obj
        except Exception:
            try:
                eval_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                return eval_obj.to_mesh(), eval_obj
            except Exception:
                logger.exception('Failed to evaluate mesh for %s', getattr(obj, 'name', '<unknown>'))
                return None, None

    @staticmethod
    def _free_evaluated_mesh(eval_obj) -> None:
        if eval_obj is not None:
            try:
                eval_obj.to_mesh_clear()
            except Exception:
                pass

    @staticmethod
    def _active_color_attribute(mesh):
        attributes = getattr(mesh, 'color_attributes', None)
        if not attributes:
            return None
        active = getattr(attributes, 'active_color', None) or getattr(attributes, 'active', None)
        if active is not None:
            return active
        return attributes[0] if len(attributes) else None

    @staticmethod
    def _u8(value: object) -> int:
        return max(0, min(0xFF, int(value)))

    @staticmethod
    def _sample_color(color_attr, loop_index: int, vertex_index: int, alpha_scale: float = 255.0) -> Tuple[int, int, int, int]:
        if color_attr is None:
            return (255, 255, 255, 255)
        try:
            domain = getattr(color_attr, 'domain', 'CORNER')
            source_index = loop_index if domain == 'CORNER' else vertex_index
            color_data = color_attr.data[source_index]
            raw = getattr(color_data, 'color', None)
            if raw is None:
                raw = getattr(color_data, 'color_srgb', None)
            rgba = []
            for channel_index, channel in enumerate(raw[:4]):
                scale = float(alpha_scale) if channel_index == 3 else 255.0
                rgba.append(max(0, min(255, int(round(float(channel) * scale)))))
            while len(rgba) < 4:
                rgba.append(255)
            return rgba[0], rgba[1], rgba[2], rgba[3]
        except Exception:
            return (255, 255, 255, 255)

    @staticmethod
    def _imported_level_color_attribute(mesh):
        attributes = getattr(mesh, 'color_attributes', None)
        if attributes:
            try:
                color_attr = attributes.get('Color')
                if color_attr is not None:
                    return color_attr
            except Exception:
                pass
        return EXPORT_SCENE_OT_trlau_level._active_color_attribute(mesh)

    @staticmethod
    def _compute_bounds(vertices: Sequence[Tuple[float, float, float]]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if not vertices:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        min_x = min(vertex[0] for vertex in vertices)
        min_y = min(vertex[1] for vertex in vertices)
        min_z = min(vertex[2] for vertex in vertices)
        max_x = max(vertex[0] for vertex in vertices)
        max_y = max(vertex[1] for vertex in vertices)
        max_z = max(vertex[2] for vertex in vertices)
        return (float(min_x), float(min_y), float(min_z)), (float(max_x), float(max_y), float(max_z))


    @staticmethod
    def _terrain_light_radius_from_object(obj) -> int:
        light_data = getattr(obj, 'data', None)
        if light_data is not None and hasattr(light_data, 'shadow_soft_size'):
            try:
                return max(0, int(round(float(light_data.shadow_soft_size) * 10.0)))
            except Exception:
                pass
        return max(0, int(round(float(obj.get('trlau_terrain_light_radius', 0) or 0))))

    @staticmethod
    def _terrain_light_multiplier_from_object(obj) -> int:
        light_data = getattr(obj, 'data', None)
        if light_data is not None and hasattr(light_data, 'energy'):
            try:
                return int(round(float(light_data.energy) / 100.0))
            except Exception:
                pass
        return int(round(float(obj.get('trlau_terrain_light_multiplier', 0) or 0)))

    @staticmethod
    def _terrain_light_type_from_object(obj) -> int:
        if 'trlau_terrain_light_type' in obj:
            return int(obj.get('trlau_terrain_light_type', 0))
        light_data = getattr(obj, 'data', None)
        blender_type = str(getattr(light_data, 'type', '') or '').upper()
        if blender_type == 'SPOT':
            return 1
        if blender_type == 'SUN':
            return 2
        return 0

    @staticmethod
    def _terrain_light_color_from_object(obj) -> Tuple[float, float, float]:
        light_data = getattr(obj, 'data', None)
        color = getattr(light_data, 'color', None) if light_data is not None else None
        if color is not None and len(color) >= 3:
            return (float(color[0]), float(color[1]), float(color[2]))
        return (1.0, 1.0, 1.0)

    @staticmethod
    def _terrain_light_falloff_angle_from_object(obj) -> int:
        if 'trlau_terrain_light_falloff_angle' in obj:
            return int(obj.get('trlau_terrain_light_falloff_angle', 0))
        light_data = getattr(obj, 'data', None)
        if light_data is not None and hasattr(light_data, 'spot_size'):
            try:
                return int(round(math.degrees(float(light_data.spot_size))))
            except Exception:
                pass
        return 0

    @staticmethod
    def _terrain_light_direction_from_object(obj, root_matrix_inv=None) -> Tuple[float, float, float]:
        if 'trlau_terrain_light_direction' in obj:
            return EXPORT_SCENE_OT_trlau_level._vector_prop(obj, 'trlau_terrain_light_direction', default=(0.0, 0.0, -4096.0))
        try:
            direction_world = obj.matrix_world.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))
            if root_matrix_inv is not None:
                direction = root_matrix_inv.to_3x3() @ direction_world
            else:
                direction = direction_world
            if direction.length > 1e-6:
                direction.normalize()
                direction *= 4096.0
                return (float(direction.x), float(direction.y), float(direction.z))
        except Exception:
            pass
        return (0.0, 0.0, -4096.0)

    @staticmethod
    def _markup_curve_points(obj, root_matrix_inv=None) -> list[tuple[float, float, float, float]]:
        points: list[tuple[float, float, float, float]] = []
        data = getattr(obj, 'data', None)
        if getattr(obj, 'type', None) != 'CURVE' or data is None:
            return points
        for spline in getattr(data, 'splines', []) or []:
            if getattr(spline, 'type', '') == 'BEZIER':
                source_points = getattr(spline, 'bezier_points', []) or []
                for point in source_points:
                    co = getattr(point, 'co', (0.0, 0.0, 0.0))
                    local = mathutils.Vector((float(co[0]), float(co[1]), float(co[2])))
                    world = obj.matrix_world @ local
                    if root_matrix_inv is not None:
                        world = root_matrix_inv @ world
                    points.append((float(world.x), float(world.y), float(world.z), 1.0))
            else:
                source_points = getattr(spline, 'points', []) or []
                for point in source_points:
                    co = getattr(point, 'co', (0.0, 0.0, 0.0, 1.0))
                    local = mathutils.Vector((float(co[0]), float(co[1]), float(co[2])))
                    world = obj.matrix_world @ local
                    if root_matrix_inv is not None:
                        world = root_matrix_inv @ world
                    pw = float(co[3]) if len(co) > 3 else 1.0
                    points.append((float(world.x), float(world.y), float(world.z), pw))
        return points

    @staticmethod
    def _markup_object_bbox_points(obj, root_matrix_inv=None) -> list[tuple[float, float, float, float]]:
        points: list[tuple[float, float, float, float]] = []
        bound_box = getattr(obj, 'bound_box', None)
        if not bound_box:
            return points
        try:
            raw_corners = [tuple(float(coord) for coord in corner[:3]) for corner in bound_box]
        except Exception:
            return points
        if not raw_corners:
            return points
        first = raw_corners[0]
        if all(all(abs(corner[index] - first[index]) <= 1e-6 for index in range(3)) for corner in raw_corners):
            return points
        for corner in raw_corners:
            local = mathutils.Vector(corner)
            world = obj.matrix_world @ local
            if root_matrix_inv is not None:
                world = root_matrix_inv @ world
            points.append((float(world.x), float(world.y), float(world.z), 1.0))
        return points

    @staticmethod
    def _i16_clamped(value: object) -> int:
        try:
            value = int(round(float(value)))
        except Exception:
            value = 0
        return max(-0x8000, min(0x7FFF, value))

    @classmethod
    def _markup_bbox_from_points(
        cls,
        points: Sequence[Sequence[float]],
        default_position: Sequence[float],
        *,
        relative_to_position: bool = False,
    ) -> tuple[int, int, int, int, int, int]:
        sample_points = list(points or [])
        if not sample_points:
            sample_points = [default_position]
        if relative_to_position:
            base = (
                float(default_position[0]) if len(default_position) > 0 else 0.0,
                float(default_position[1]) if len(default_position) > 1 else 0.0,
                float(default_position[2]) if len(default_position) > 2 else 0.0,
            )
            raw_points = [
                cls._blender_to_raw_float_triplet((
                    float(point[0]) - base[0],
                    float(point[1]) - base[1],
                    float(point[2]) - base[2],
                ))
                for point in sample_points
            ]
        else:
            raw_points = [cls._blender_to_raw_float_triplet(point[:3]) for point in sample_points]
        min_x = min(float(point[0]) for point in raw_points)
        min_y = min(float(point[1]) for point in raw_points)
        min_z = min(float(point[2]) for point in raw_points)
        max_x = max(float(point[0]) for point in raw_points)
        max_y = max(float(point[1]) for point in raw_points)
        max_z = max(float(point[2]) for point in raw_points)
        return (
            cls._i16_clamped(min_x),
            cls._i16_clamped(min_y),
            cls._i16_clamped(min_z),
            cls._i16_clamped(max_x),
            cls._i16_clamped(max_y),
            cls._i16_clamped(max_z),
        )

    @staticmethod
    def _gather_markups(collection: Collection, root, root_matrix_inv=None, level_game: str = 'legend') -> List[ExportMarkupData]:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        markup_root = EXPORT_SCENE_OT_trlau_level._find_component_empty(collection, root, 'Markup', 'trlau_type', 'MarkupRoot', 'trlau_markup_root')
        markup_objects = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'CURVE':
                continue
            is_markup = str(obj.get('trlau_type', '') or '') == 'Markup' or _MARKUP_NAME_RE.search(getattr(obj, 'name', '') or '') is not None
            if not is_markup:
                continue
            if markup_root is not None:
                if obj != markup_root and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, markup_root):
                    continue
            elif root is not None and obj != root and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            markup_objects.append(obj)

        markup_objects.sort(key=lambda obj: (EXPORT_SCENE_OT_trlau_level._markup_index_from_object(obj, 0), getattr(obj, 'name', '')))
        result: List[ExportMarkupData] = []
        game = str(level_game or 'legend').strip().lower()
        for default_index, obj in enumerate(markup_objects):
            position = EXPORT_SCENE_OT_trlau_level._relative_translation(obj, root_matrix_inv)
            flags = int(obj.get('trlau_markup_flags', 0) or 0)
            points = EXPORT_SCENE_OT_trlau_level._markup_curve_points(obj, root_matrix_inv)
            shape_points = points
            bbox_relative_to_position = False
            if flags & MARKUP_BBOX_FLAGS:
                shape_points = points or EXPORT_SCENE_OT_trlau_level._markup_object_bbox_points(obj, root_matrix_inv)
                bbox_relative_to_position = True
            bbox = EXPORT_SCENE_OT_trlau_level._markup_bbox_from_points(
                shape_points,
                position,
                relative_to_position=bbox_relative_to_position,
            )
            animated_segment = int(obj.get('trlau_markup_animated_segment', 0) or 0) if game == 'anniversary' else 0
            result.append(ExportMarkupData(
                index=EXPORT_SCENE_OT_trlau_level._markup_index_from_object(obj, default_index),
                flags=flags,
                animated_segment=animated_segment,
                intro_id=EXPORT_SCENE_OT_trlau_level._markup_intro_id_from_object(obj) if game in {'legend', 'anniversary'} else 0,
                position=position,
                bbox=bbox,
                polyline=[] if (flags & MARKUP_BBOX_FLAGS) else points,
            ))
        result.sort(key=lambda item: int(item.index))
        for normalized_index, item in enumerate(result):
            item.index = normalized_index
        return result


    @staticmethod
    def _markup_intro_id_from_object(obj) -> int:
        try:
            data = getattr(obj, 'trlau_markup_data', None)
            if data is not None:
                return int(getattr(data, 'intro_id', 0) or 0)
        except Exception:
            pass
        return 0

    @staticmethod
    def _parse_int_sequence_property(value: object) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                decoded = json.loads(text)
                if isinstance(decoded, (list, tuple)):
                    value = decoded
                else:
                    value = [decoded]
            except Exception:
                text = text.strip('()[]')
                value = [part.strip() for part in text.split(',') if part.strip()]
        if not isinstance(value, (list, tuple)):
            value = [value]
        result: list[int] = []
        for item in value:
            try:
                number = int(item)
            except Exception:
                continue
            if number > 0:
                result.append(number)
        return result

    @staticmethod
    def _sfx_sound_properties_from_object(obj) -> dict[str, object]:
        properties: dict[str, object] = {}
        kind = str(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_kind', '') or '').strip().lower()
        if kind == 'periodic':
            fields = SFX_SOUND_PERIODIC_FIELDS
        elif kind in {'event', 'one_shot', 'oneshot', 'one-shot'}:
            fields = SFX_SOUND_EVENT_FIELDS
        elif kind == 'stream':
            fields = SFX_SOUND_STREAM_FIELDS
        else:
            fields = SFX_SOUND_PERIODIC_FIELDS + SFX_SOUND_EVENT_FIELDS + SFX_SOUND_STREAM_FIELDS
        for field_name, _label in fields:
            prop_name = f'trlau_sfx_{field_name}'
            value = EXPORT_SCENE_OT_trlau_level._object_prop(obj, prop_name, None)
            if value is not None:
                properties[field_name] = value
        try:
            for key in list(obj.keys()):
                if not str(key).startswith('trlau_sfx_'):
                    continue
                field_name = str(key)[len('trlau_sfx_'):]
                if field_name in {'kind', 'entry_index', 'source_offset', 'ids', 'speaker_count', 'resolved_json'} or field_name.startswith('ref_') or field_name.startswith('stream_filename_'):
                    continue
                properties.setdefault(field_name, obj.get(key))
        except Exception:
            pass
        return properties

    @staticmethod
    def _gather_sfx_sound_children(marker_obj) -> tuple[list[ExportSFXSound], list[ExportSFXSound], list[ExportSFXSound], list[ExportSFXSound]]:
        event_sounds: list[ExportSFXSound] = []
        periodic_sounds: list[ExportSFXSound] = []
        one_shot_sounds: list[ExportSFXSound] = []
        stream_sounds: list[ExportSFXSound] = []
        for obj in list(getattr(marker_obj, 'children', []) or []):
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            role = str(getattr(obj, 'trlau_sfx_role', '') or obj.get('trlau_type', '') or '')
            if role != 'SFXSound':
                continue
            kind = str(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_kind', '') or '').strip().lower()
            if not kind:
                continue
            try:
                entry_index = int(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_entry_index', 0) or 0)
            except Exception:
                entry_index = 0
            sound = ExportSFXSound(
                index=entry_index,
                kind=kind,
                sound_ids=EXPORT_SCENE_OT_trlau_level._parse_int_sequence_property(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_ids', ())),
                properties=EXPORT_SCENE_OT_trlau_level._sfx_sound_properties_from_object(obj),
            )
            if kind == 'event':
                event_sounds.append(sound)
            elif kind == 'periodic':
                periodic_sounds.append(sound)
            elif kind in {'one_shot', 'oneshot', 'one-shot'}:
                sound.kind = 'one_shot'
                one_shot_sounds.append(sound)
            elif kind == 'stream':
                stream_sounds.append(sound)
        for sounds in (event_sounds, periodic_sounds, one_shot_sounds, stream_sounds):
            sounds.sort(key=lambda item: int(item.index))
            for default_index, sound in enumerate(sounds):
                if int(sound.index) < 0:
                    sound.index = default_index
        return event_sounds, periodic_sounds, one_shot_sounds, stream_sounds

    @staticmethod
    def _gather_sfx_markers(collection: Collection, root, root_matrix_inv=None) -> List[ExportSFXMarker]:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        marker_objects = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if root is not None and obj != root and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            role = str(getattr(obj, 'trlau_sfx_role', '') or obj.get('trlau_type', '') or '')
            if role != 'SFXMarker':
                continue
            marker_objects.append(obj)
        marker_objects.sort(key=lambda obj: (int(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_marker_index', 0) or 0), getattr(obj, 'name', '')))
        result: list[ExportSFXMarker] = []
        for default_index, obj in enumerate(marker_objects):
            event_sounds, periodic_sounds, one_shot_sounds, stream_sounds = EXPORT_SCENE_OT_trlau_level._gather_sfx_sound_children(obj)
            try:
                marker_index = int(EXPORT_SCENE_OT_trlau_level._object_prop(obj, 'trlau_sfx_marker_index', default_index) or default_index)
            except Exception:
                marker_index = int(default_index)
            result.append(ExportSFXMarker(
                index=marker_index,
                position=EXPORT_SCENE_OT_trlau_level._relative_translation(obj, root_matrix_inv),
                unique_id=EXPORT_SCENE_OT_trlau_level._int_prop(obj, 'trlau_sfx_marker_unique_id', 0),
                plane=EXPORT_SCENE_OT_trlau_level._int_prop(obj, 'trlau_sfx_marker_plane', 0),
                spline_id=EXPORT_SCENE_OT_trlau_level._int_prop(obj, 'trlau_sfx_marker_spline_id', 0),
                sound_instance_offset=EXPORT_SCENE_OT_trlau_level._int_prop(obj, 'trlau_sfx_marker_sound_instance_offset', 0),
                event_sounds=event_sounds,
                periodic_sounds=periodic_sounds,
                one_shot_sounds=one_shot_sounds,
                stream_sounds=stream_sounds,
            ))
        result.sort(key=lambda item: int(item.index))
        for normalized_index, item in enumerate(result):
            item.index = int(normalized_index)
        return result

    @staticmethod
    def _iter_sfx_audio_section_metadata(collection: Collection, root):
        objects = getattr(collection, 'all_objects', None) or collection.objects
        result = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if root is not None and obj != root and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            if not bool(obj.get('trlau_section_metadata_empty')):
                continue
            role = str(obj.get('trlau_section_role', '') or '')
            if role not in {'SFX', 'Wave'}:
                continue
            result.append(obj)
        result.sort(key=lambda obj: (str(obj.get('trlau_section_role', '') or ''), int(obj.get('trlau_section_original_id', 0) or 0), getattr(obj, 'name', '')))
        return result


    @staticmethod
    def _gather_terrain_lights(collection: Collection, root, root_matrix_inv=None) -> List[ExportTerrainLightData]:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        light_objects = []
        for obj in objects:
            if getattr(obj, 'type', None) != 'LIGHT':
                continue
            if root is not None and obj != root and not EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root):
                continue
            if str(obj.get('trlau_type', '') or '') != 'TerrainLight' and 'trlau_terrain_light_id' not in obj:
                continue
            light_objects.append(obj)

        light_objects.sort(key=lambda obj: (int(obj.get('trlau_terrain_light_index', 0) or 0), getattr(obj, 'name', '')))
        result: List[ExportTerrainLightData] = []
        for default_index, obj in enumerate(light_objects):
            result.append(ExportTerrainLightData(
                index=int(obj.get('trlau_terrain_light_index', default_index) or default_index),
                position=EXPORT_SCENE_OT_trlau_level._relative_translation(obj, root_matrix_inv),
                radius=EXPORT_SCENE_OT_trlau_level._terrain_light_radius_from_object(obj),
                color=EXPORT_SCENE_OT_trlau_level._terrain_light_color_from_object(obj),
                type=EXPORT_SCENE_OT_trlau_level._terrain_light_type_from_object(obj),
                multiplier=EXPORT_SCENE_OT_trlau_level._terrain_light_multiplier_from_object(obj),
                hotspot_angle=int(obj.get('trlau_terrain_light_hotspot_angle', 0) or 0),
                direction=EXPORT_SCENE_OT_trlau_level._terrain_light_direction_from_object(obj, root_matrix_inv),
                falloff_angle=EXPORT_SCENE_OT_trlau_level._terrain_light_falloff_angle_from_object(obj),
                light_id=int(obj.get('trlau_terrain_light_id', default_index) or 0),
            ))
        result.sort(key=lambda item: int(item.index))
        for normalized_index, item in enumerate(result):
            item.index = normalized_index
        return result

    @staticmethod
    def _is_bgobject_mesh(obj) -> bool:
        if getattr(obj, 'type', None) != 'MESH':
            return False
        if bool(obj.get('trlau_bgobject')) and (str(obj.get('trlau_type', '') or '') == 'BGObject' or 'trlau_bgobject_index' in obj):
            return True
        return _BGOBJECT_NAME_RE.search(getattr(obj, 'name', '') or '') is not None

    @staticmethod
    def _is_bginstance_empty(obj) -> bool:
        if getattr(obj, 'type', None) != 'EMPTY':
            return False
        if (bool(obj.get('trlau_bginstance_empty')) or str(obj.get('trlau_type', '') or '') == 'BGInstance') and ('trlau_bgobject_index' in obj or 'trlau_bginstance_bgobject_index' in obj):
            return True
        return _BGINSTANCE_NAME_RE.search(getattr(obj, 'name', '') or '') is not None

    @staticmethod
    def _is_bgobject_empty(obj) -> bool:
        if getattr(obj, 'type', None) != 'EMPTY':
            return False
        if (bool(obj.get('trlau_bgobject_empty')) or (bool(obj.get('trlau_bgobject')) and not bool(obj.get('trlau_bginstance')))) and 'trlau_bgobject_index' in obj:
            return True
        return _BGOBJECT_NAME_RE.search(getattr(obj, 'name', '') or '') is not None and _BGINSTANCE_NAME_RE.search(getattr(obj, 'name', '') or '') is None

    @staticmethod
    def _coerce_prop_triplet(value, default: tuple[float, float, float]) -> tuple[float, float, float]:
        if value is None or isinstance(value, (str, bytes, dict)):
            return default
        try:
            if len(value) >= 3:
                return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            pass
        return default

    @staticmethod
    def _bgobject_scale_from_source(source) -> tuple[float, float, float]:
        if source is None:
            return (1.0, 1.0, 1.0)
        keys = ('trlau_bgobject_scale_x', 'trlau_bgobject_scale_y', 'trlau_bgobject_scale_z')
        if all(key in source for key in keys):
            try:
                return (float(source.get(keys[0])), float(source.get(keys[1])), float(source.get(keys[2])))
            except Exception:
                pass
        value = source.get('trlau_bgobject_scale') if 'trlau_bgobject_scale' in source else None
        coerced = EXPORT_SCENE_OT_trlau_level._coerce_prop_triplet(value, (float('nan'), float('nan'), float('nan')))
        if all(math.isfinite(v) and abs(v) > 1.0e-9 for v in coerced):
            return coerced
        try:
            scale = source.matrix_local.to_scale()
            result = (abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)))
            if all(math.isfinite(v) and abs(v) > 1.0e-9 for v in result):
                return result
        except Exception:
            pass
        try:
            scale = getattr(source, 'scale', (1.0, 1.0, 1.0))
            return (abs(float(scale[0])), abs(float(scale[1])), abs(float(scale[2])))
        except Exception:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _bgobject_position_from_source(source) -> tuple[float, float, float]:
        if source is None:
            return (0.0, 0.0, 0.0)
        keys = ('trlau_bgobject_position_x', 'trlau_bgobject_position_y', 'trlau_bgobject_position_z')
        if all(key in source for key in keys):
            try:
                return (float(source.get(keys[0])), float(source.get(keys[1])), float(source.get(keys[2])))
            except Exception:
                pass
        value = source.get('trlau_bgobject_position') if 'trlau_bgobject_position' in source else None
        coerced = EXPORT_SCENE_OT_trlau_level._coerce_prop_triplet(value, (float('nan'), float('nan'), float('nan')))
        if all(math.isfinite(v) for v in coerced):
            return coerced
        try:
            loc = source.matrix_local.to_translation()
            return (-float(loc.x), float(loc.z), float(loc.y))
        except Exception:
            try:
                loc = getattr(source, 'location', (0.0, 0.0, 0.0))
                return (-float(loc[0]), float(loc[2]), float(loc[1]))
            except Exception:
                return (0.0, 0.0, 0.0)

    @staticmethod
    def _decode_bgobject_raw_blob(obj):
        if obj is None:
            return None
        try:
            chunk_count = int(obj.get('trlau_bgobject_blob_chunk_count', 0) or 0)
        except Exception:
            chunk_count = 0
        if chunk_count <= 0:
            return None
        encoded = ''.join(str(obj.get(f'{BGOBJECT_BLOB_PROP_PREFIX}{index:04d}', '') or '') for index in range(chunk_count))
        if not encoded:
            return None
        try:
            blob = base64.b64decode(encoded.encode('ascii'))
        except Exception:
            logger.warning('Failed to decode BGObject raw blob for %s', getattr(obj, 'name', '<unknown>'))
            return None

        magic = blob[:4]
        if magic not in {b'BGO2', b'BGO3'}:
            return None
        try:
            if magic == b'BGO3':
                if len(blob) < 32:
                    return None
                vertex_count, raw_vertex_len, raw_color_len, color_slot_count, strip_count, env_count, eye_count = struct.unpack_from('<IIIIIII', blob, 4)
                cursor = 32
            else:
                if len(blob) < 24:
                    return None
                vertex_count, raw_vertex_len, raw_color_len, color_slot_count, strip_count = struct.unpack_from('<IIIII', blob, 4)
                env_count = 0
                eye_count = 0
                cursor = 24

            raw_vertex_len = int(raw_vertex_len)
            raw_color_len = int(raw_color_len)
            vertex_count = int(vertex_count)
            env_count = int(env_count)
            eye_count = int(eye_count)
            if vertex_count <= 0 or raw_vertex_len < vertex_count * 12:
                return None
            if raw_vertex_len < 0 or raw_color_len < 0 or env_count < 0 or eye_count < 0:
                return None
            env_byte_len = env_count * 2
            eye_byte_len = eye_count * 2
            if cursor + raw_vertex_len + raw_color_len + env_byte_len + eye_byte_len > len(blob):
                return None

            raw_vertices = bytes(blob[cursor:cursor + raw_vertex_len])
            cursor += raw_vertex_len
            raw_colors = bytes(blob[cursor:cursor + raw_color_len])
            cursor += raw_color_len
            env_indices = list(struct.unpack_from('<' + ('H' * env_count), blob, cursor)) if env_count else []
            cursor += env_byte_len
            eye_indices = list(struct.unpack_from('<' + ('H' * eye_count), blob, cursor)) if eye_count else []
            cursor += eye_byte_len

            vertices: list[ExportVertex] = []
            for vertex_index in range(vertex_count):
                item_offset = vertex_index * 12
                x, y, z, _w, u, v = struct.unpack_from('<hhhhhh', raw_vertices, item_offset)
                color = (255, 255, 255, 255)
                if raw_colors and (vertex_index * 4 + 4) <= len(raw_colors):
                    color_bgra = struct.unpack_from('<I', raw_colors, vertex_index * 4)[0]
                    b = color_bgra & 0xFF
                    g = (color_bgra >> 8) & 0xFF
                    r = (color_bgra >> 16) & 0xFF
                    a = (color_bgra >> 24) & 0xFF
                    color = (r, g, b, a)
                vertices.append(ExportVertex(position=(float(x), float(y), float(z)), uv=(float(u) / _UV_EXPORT_SCALE, float(v) / _UV_EXPORT_SCALE), color=color))

            strips: list[ExportStrip] = []
            for strip_index in range(min(int(strip_count), 100000)):
                if cursor + 28 > len(blob):
                    break
                stored_count = struct.unpack_from('<i', blob, cursor)[0]
                cursor += 4
                sx, sy, sz, _pad = struct.unpack_from('<hhhH', blob, cursor)
                cursor += 8
                tpageid, sort_push, scroll_offset, index_count = struct.unpack_from('<IifI', blob, cursor)
                cursor += 16
                index_count = int(index_count)
                if index_count < 0 or cursor + (index_count * 2) > len(blob):
                    break
                indices = list(struct.unpack_from('<' + ('H' * index_count), blob, cursor)) if index_count else []
                cursor += index_count * 2
                is_terminator = int(stored_count) <= 0
                if is_terminator:
                    indices = []
                strip = ExportStrip(
                    name=f'{getattr(obj, "name", "BGObject")}_RawStrip_{strip_index:03d}',
                    tpageid=int(tpageid),
                    indices=[int(value) for value in indices],
                    sort_vertex=(int(sx), int(sy), int(sz)),
                    sort_push=int(sort_push),
                    scroll_offset=float(scroll_offset),
                    raw_count=int(stored_count),
                    is_terminator=bool(is_terminator),
                )
                strips.append(strip)
            if not vertices or not strips:
                return None
            env_indices = sorted({int(value) for value in env_indices if int(value) >= 0 and int(value) < len(vertices)})
            eye_indices = sorted({int(value) for value in eye_indices if int(value) >= 0 and int(value) < len(vertices)})
            return vertices, strips, raw_vertices, raw_colors, max(1, int(color_slot_count or 1)), env_indices, eye_indices
        except Exception:
            logger.warning('Failed to parse BGObject raw blob for %s', getattr(obj, 'name', '<unknown>'))
            return None

    @staticmethod
    def _material_reflection_flags(material) -> tuple[bool, bool]:
        if material is None:
            return (False, False)
        env = bool(get_material_env_mapping(material))
        eye = bool(get_material_eye_ref_env_mapping(material))
        reflective = bool(env or eye)
        if reflective and not env and not eye:
            env = True
        return (env, eye)

    def _bgobject_reflection_modes_from_meshes(self, mesh_objects: Sequence[object]) -> dict[int, tuple[bool, bool]]:
        modes: dict[int, tuple[bool, bool]] = {}
        for default_index, obj in enumerate(sorted(list(mesh_objects or []), key=self._strip_sort_key)):
            match = _STRIP_NAME_RE.search(getattr(obj, 'name', '') or '')
            strip_index = int(match.group(1)) if match is not None else default_index
            env = False
            eye = False
            for _material_index, material in self._iter_used_strip_materials(obj):
                mat_env, mat_eye = self._material_reflection_flags(material)
                env = env or mat_env
                eye = eye or mat_eye
            if not env and not eye:
                material = self._resolve_strip_material(obj)
                env, eye = self._material_reflection_flags(material)
            modes[int(strip_index)] = (bool(env), bool(eye))
        return modes

    @staticmethod
    def _bgobject_reflection_modes_from_indices(strips: Sequence[ExportStrip], env_indices: Sequence[int], eye_indices: Sequence[int]) -> dict[int, tuple[bool, bool]]:
        env_set = {int(value) for value in list(env_indices or []) if int(value) >= 0}
        eye_set = {int(value) for value in list(eye_indices or []) if int(value) >= 0}
        modes: dict[int, tuple[bool, bool]] = {}
        visible_index = 0
        for strip in list(strips or []):
            if bool(getattr(strip, 'is_terminator', False)):
                continue
            base = int(getattr(strip, 'vertex_base_offset', 0) or 0)
            indices = [base + int(value) for value in list(getattr(strip, 'indices', []) or [])]
            modes[visible_index] = (
                any(index in env_set for index in indices),
                any(index in eye_set for index in indices),
            )
            visible_index += 1
        return modes

    @staticmethod
    def _bgobject_indices_from_reflection_modes(strips: Sequence[ExportStrip], modes: dict[int, tuple[bool, bool]]) -> tuple[list[int], list[int]]:
        env_indices: list[int] = []
        eye_indices: list[int] = []
        visible_index = 0
        for strip in list(strips or []):
            if bool(getattr(strip, 'is_terminator', False)):
                continue
            env, eye = modes.get(visible_index, (False, False))
            if env or eye:
                base = int(getattr(strip, 'vertex_base_offset', 0) or 0)
                indices = [base + int(value) for value in list(getattr(strip, 'indices', []) or []) if int(value) >= 0]
                if env:
                    env_indices.extend(indices)
                if eye:
                    eye_indices.extend(indices)
                strip.env_mapping = bool(env)
                strip.eye_ref_env_mapping = bool(eye)
            else:
                strip.env_mapping = False
                strip.eye_ref_env_mapping = False
            visible_index += 1
        return sorted(set(env_indices)), sorted(set(eye_indices))

    @staticmethod
    def _bgobject_source_indices_for_mesh(obj, strip: ExportStrip, vertex_count: int) -> list[int]:
        try:
            raw_mapping = obj.get('trlau_bgobject_source_indices') if obj is not None and 'trlau_bgobject_source_indices' in obj else None
        except Exception:
            raw_mapping = None

        if raw_mapping is not None and not isinstance(raw_mapping, (str, bytes, dict)):
            try:
                mapping = [int(value) for value in list(raw_mapping)]
            except Exception:
                mapping = []
            if mapping and all(0 <= value < int(vertex_count) for value in mapping):
                return mapping
        return []

    @staticmethod
    def _bgobject_source_indices_prop(obj) -> list[int]:
        if obj is None:
            return []
        try:
            raw_mapping = obj.get('trlau_bgobject_source_indices') if 'trlau_bgobject_source_indices' in obj else None
        except Exception:
            raw_mapping = None
        if raw_mapping is not None and not isinstance(raw_mapping, (str, bytes, dict)):
            try:
                mapping = [int(value) for value in list(raw_mapping)]
            except Exception:
                mapping = []
            mapping = [value for value in mapping if value >= 0]
            if mapping:
                return mapping

        return []

    @staticmethod
    def _bgobject_color_slot_from_mesh(obj) -> int:
        for candidate in (obj, getattr(obj, 'parent', None) if obj is not None else None):
            if candidate is None:
                continue
            for key in ('trlau_bgobject_color_data_index', 'trlau_bginstance_color_data_index'):
                try:
                    if key in candidate:
                        return int(candidate.get(key, 0) or 0)
                except Exception:
                    pass
        return 0

    @staticmethod
    def _bgobject_imported_vertex_count(mesh_objects: Sequence[object]) -> int:
        vertex_count = 0
        for obj in list(mesh_objects or []):
            try:
                if 'trlau_bgobject_vertex_count' in obj:
                    vertex_count = max(vertex_count, int(obj.get('trlau_bgobject_vertex_count', 0) or 0))
            except Exception:
                pass
            try:
                if 'trlau_vertex_color_count' in obj:
                    vertex_count = max(vertex_count, int(obj.get('trlau_vertex_color_count', 0) or 0))
            except Exception:
                pass
            mapping = EXPORT_SCENE_OT_trlau_level._bgobject_source_indices_prop(obj)
            if mapping:
                vertex_count = max(vertex_count, max(mapping) + 1)
        return int(vertex_count)

    def _export_imported_bgobject_geometry_and_colors(
        self,
        geometry_meshes: Sequence[object],
        color_meshes: Sequence[object],
        parent,
    ) -> Optional[tuple[list[ExportVertex], list[ExportStrip], bytes, int]]:
        ordered_color_meshes: list[object] = []
        seen_ids: set[int] = set()
        for obj in list(color_meshes or []) + list(geometry_meshes or []):
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen_ids:
                continue
            seen_ids.add(ident)
            ordered_color_meshes.append(obj)

        vertex_count = self._bgobject_imported_vertex_count(ordered_color_meshes)
        if vertex_count <= 0:
            return None

        parent_inv = None
        if parent is not None:
            try:
                parent_inv = parent.matrix_world.inverted_safe()
            except Exception:
                parent_inv = None

        positions: list[Optional[tuple[float, float, float]]] = [None] * vertex_count
        uvs: list[Optional[tuple[float, float]]] = [None] * vertex_count

        def _uvs_by_vertex(mesh) -> dict[int, tuple[float, float]]:
            result: dict[int, tuple[float, float]] = {}
            uv_layer = None
            try:
                uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
            except Exception:
                uv_layer = None
            if uv_layer is None:
                return result
            try:
                loops = getattr(mesh, 'loops', None)
                if loops is None:
                    return result
                for poly in list(getattr(mesh, 'polygons', []) or []):
                    for loop_index in list(getattr(poly, 'loop_indices', []) or []):
                        vertex_index = int(loops[loop_index].vertex_index)
                        if vertex_index in result:
                            continue
                        uv_value = uv_layer[loop_index].uv
                        result[vertex_index] = (float(uv_value.x), 1.0 - float(uv_value.y))
            except Exception:
                return result
            return result

        for obj in sorted(list(geometry_meshes or []), key=self._strip_sort_key):
            mapping = self._bgobject_source_indices_prop(obj)
            if not mapping:
                continue
            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            local_uvs = _uvs_by_vertex(mesh)
            for local_index, src_index in enumerate(mapping):
                if src_index < 0 or src_index >= vertex_count:
                    continue
                try:
                    if local_index < len(mesh.vertices):
                        co = mesh.vertices[local_index].co.copy()
                        if parent_inv is not None:
                            co = parent_inv @ (obj.matrix_world @ co)
                        positions[src_index] = (float(co.x), float(co.y), float(co.z))
                except Exception:
                    pass
                if local_index in local_uvs:
                    uvs[src_index] = local_uvs[local_index]

        vertices: list[ExportVertex] = []
        for index in range(vertex_count):
            pos = positions[index] if positions[index] is not None else (0.0, 0.0, 0.0)
            uv = uvs[index] if uvs[index] is not None else (0.0, 0.0)
            vertices.append(ExportVertex(position=pos, uv=uv, color=(255, 255, 255, 128)))

        strips: list[ExportStrip] = []
        for obj in sorted(list(geometry_meshes or []), key=self._strip_sort_key):
            mapping = self._bgobject_source_indices_prop(obj)
            if not mapping:
                continue
            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            try:
                mesh.calc_loop_triangles()
                loops = getattr(mesh, 'loops', None)
                if loops is None:
                    continue
                triangles_by_material: dict[int, list[object]] = {}
                for tri in list(getattr(mesh, 'loop_triangles', []) or []):
                    material_index = int(getattr(tri, 'material_index', 0) or 0)
                    triangles_by_material.setdefault(material_index, []).append(tri)
            except Exception:
                logger.warning('Failed reading BGObject triangles from %s', getattr(obj, 'name', '<unknown>'))
                continue

            for material_index in sorted(triangles_by_material.keys()):
                slot_prefix = f'trlau_bgobject_slot_{int(material_index):03d}_'
                try:
                    base = int(obj.get(slot_prefix + 'vertex_base_offset', obj.get('trlau_vertex_base_offset', 0)) or 0)
                except Exception:
                    base = 0
                indices: list[int] = []
                for tri in triangles_by_material[material_index]:
                    tri_indices: list[int] = []
                    for loop_index in list(getattr(tri, 'loops', []) or []):
                        vertex_index = int(loops[loop_index].vertex_index)
                        if vertex_index < 0 or vertex_index >= len(mapping):
                            tri_indices = []
                            break
                        src_index = int(mapping[vertex_index])
                        raw_index = src_index - base
                        if raw_index < 0 or raw_index > 0xFFFF:
                            tri_indices = []
                            break
                        tri_indices.append(raw_index)
                    if len(tri_indices) == 3 and len(set(tri_indices)) == 3:
                        indices.extend(tri_indices)
                if not indices:
                    continue

                material = self._resolve_strip_material(obj, material_index)
                try:
                    sort_vertex = (
                        int(obj.get(slot_prefix + 'sort_vertex_x', obj.get('trlau_bgobject_strip_sort_vertex_x', 0)) or 0),
                        int(obj.get(slot_prefix + 'sort_vertex_y', obj.get('trlau_bgobject_strip_sort_vertex_y', 0)) or 0),
                        int(obj.get(slot_prefix + 'sort_vertex_z', obj.get('trlau_bgobject_strip_sort_vertex_z', 0)) or 0),
                    )
                except Exception:
                    sort_vertex = (0, 0, 0)
                try:
                    source_strip_index = int(obj.get(slot_prefix + 'strip_index', 0) or 0)
                except Exception:
                    source_strip_index = 0
                try:
                    sort_push = int(obj.get(slot_prefix + 'sort_push', obj.get('trlau_bgobject_strip_sort_push', 0)) or 0)
                except Exception:
                    sort_push = 0
                try:
                    scroll_offset = float(obj.get(slot_prefix + 'scroll_offset', obj.get('trlau_bgobject_strip_scroll_offset', 0.0)) or 0.0)
                except Exception:
                    scroll_offset = 0.0
                try:
                    raw_count = int(obj.get(slot_prefix + 'raw_count', obj.get('trlau_bgobject_strip_raw_count', len(indices))) or len(indices))
                except Exception:
                    raw_count = len(indices)
                flags = self._resolve_strip_flags(obj, False, material=material, material_slot_index=material_index)
                if slot_prefix + 'flags' in obj:
                    try:
                        flags = int(obj.get(slot_prefix + 'flags', flags) or flags)
                    except Exception:
                        pass
                strips.append(ExportStrip(
                    name=f'{getattr(obj, "name", "BGObject")}_BGStrip_{int(source_strip_index):03d}',
                    tpageid=self._resolve_strip_tpageid(obj, material=material),
                    flags=flags,
                    indices=indices,
                    vertex_base_offset=base,
                    sort_vertex=sort_vertex,
                    sort_push=sort_push,
                    scroll_offset=scroll_offset,
                    raw_count=raw_count,
                    env_mapping=False,
                    eye_ref_env_mapping=False,
                ))

        if not strips:
            return None

        slot_count = 1
        for obj in ordered_color_meshes:
            try:
                slot_count = max(slot_count, int(obj.get('trlau_bgobject_color_slot_count', 1) or 1))
            except Exception:
                pass
            slot_index = self._bgobject_color_slot_from_mesh(obj)
            if slot_index >= 0:
                slot_count = max(slot_count, slot_index + 1)

        per_slot_size = vertex_count * 4
        raw_color_data = bytearray((b'\xFF\xFF\xFF\x80' * vertex_count) * slot_count)
        totals: dict[tuple[int, int], list[int]] = {}
        counts: dict[tuple[int, int], int] = {}
        for obj in ordered_color_meshes:
            mapping = self._bgobject_source_indices_prop(obj)
            if not mapping:
                continue
            slot_index = self._bgobject_color_slot_from_mesh(obj)
            if slot_index < 0:
                continue
            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            color_attr = self._imported_level_color_attribute(mesh)
            if color_attr is None:
                continue
            try:
                domain = str(getattr(color_attr, 'domain', 'CORNER') or 'CORNER').upper()
                if domain == 'POINT':
                    for vertex_index in range(min(len(mapping), len(getattr(mesh, 'vertices', []) or []))):
                        src_index = int(mapping[vertex_index])
                        if src_index < 0 or src_index >= vertex_count:
                            continue
                        rgba = self._sample_color(color_attr, 0, int(vertex_index), alpha_scale=128.0)
                        key = (slot_index, src_index)
                        bucket = totals.setdefault(key, [0, 0, 0, 0])
                        bucket[0] += int(rgba[0])
                        bucket[1] += int(rgba[1])
                        bucket[2] += int(rgba[2])
                        bucket[3] += int(rgba[3])
                        counts[key] = counts.get(key, 0) + 1
                else:
                    loops = getattr(mesh, 'loops', None)
                    if loops is None:
                        continue
                    for poly in list(getattr(mesh, 'polygons', []) or []):
                        for loop_index in list(getattr(poly, 'loop_indices', []) or []):
                            vertex_index = int(loops[loop_index].vertex_index)
                            if vertex_index < 0 or vertex_index >= len(mapping):
                                continue
                            src_index = int(mapping[vertex_index])
                            if src_index < 0 or src_index >= vertex_count:
                                continue
                            rgba = self._sample_color(color_attr, int(loop_index), int(vertex_index), alpha_scale=128.0)
                            key = (slot_index, src_index)
                            bucket = totals.setdefault(key, [0, 0, 0, 0])
                            bucket[0] += int(rgba[0])
                            bucket[1] += int(rgba[1])
                            bucket[2] += int(rgba[2])
                            bucket[3] += int(rgba[3])
                            counts[key] = counts.get(key, 0) + 1
            except Exception:
                logger.warning('Failed reading BGObject vertex colors from %s', getattr(obj, 'name', '<unknown>'))
                continue

        for (slot_index, src_index), color_totals in totals.items():
            if slot_index < 0 or slot_index >= slot_count or src_index < 0 or src_index >= vertex_count:
                continue
            count = max(1, int(counts.get((slot_index, src_index), 1)))
            r = self._u8(round(color_totals[0] / count))
            g = self._u8(round(color_totals[1] / count))
            b = self._u8(round(color_totals[2] / count))
            a = self._u8(round(color_totals[3] / count))
            item_offset = (slot_index * per_slot_size) + (src_index * 4)
            struct.pack_into('<I', raw_color_data, item_offset, (a << 24) | (r << 16) | (g << 8) | b)
            if slot_index == 0:
                vertices[src_index].color = (r, g, b, a)

        return vertices, strips, bytes(raw_color_data), int(slot_count)

    def _update_bgobject_raw_color_data_from_meshes(
        self,
        vertices: Sequence[ExportVertex],
        strips: Sequence[ExportStrip],
        raw_color_data: bytes,
        raw_color_slot_count: int,
        mesh_objects: Sequence[object],
    ) -> bytes:
        vertex_count = len(list(vertices or []))
        per_slot_size = int(vertex_count) * 4
        if vertex_count <= 0 or per_slot_size <= 0 or not raw_color_data:
            return raw_color_data

        slot_count = max(1, int(raw_color_slot_count or 1), len(raw_color_data) // per_slot_size)
        data = bytearray(raw_color_data)
        required_len = slot_count * per_slot_size
        if len(data) < required_len:
            data.extend((b'\xFF\xFF\xFF\xFF' * ((required_len - len(data) + 3) // 4))[:required_len - len(data)])

        for obj in list(mesh_objects or []):
            try:
                slot_index = int(obj.get('trlau_bgobject_color_data_index', 0) or 0)
            except Exception:
                slot_index = 0
            if slot_index < 0:
                continue
            if slot_index >= slot_count:
                slot_count = slot_index + 1
                required_len = slot_count * per_slot_size
                if len(data) < required_len:
                    data.extend((b'\xFF\xFF\xFF\xFF' * ((required_len - len(data) + 3) // 4))[:required_len - len(data)])

            try:
                if 'trlau_bgobject_strip_index' in obj:
                    strip_index = int(obj.get('trlau_bgobject_strip_index', 0) or 0)
                else:
                    match = _STRIP_NAME_RE.search(getattr(obj, 'name', '') or '')
                    if match is None:
                        continue
                    strip_index = int(match.group(1))
            except Exception:
                continue
            if strip_index < 0 or strip_index >= len(strips):
                continue
            strip = strips[strip_index]
            if bool(getattr(strip, 'is_terminator', False)):
                continue

            source_indices = self._bgobject_source_indices_for_mesh(obj, strip, vertex_count)
            if not source_indices:
                continue

            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            color_attr = self._imported_level_color_attribute(mesh)
            if color_attr is None:
                continue

            accum: dict[int, list[int]] = {}
            sample_count: dict[int, int] = {}
            try:
                loops = getattr(mesh, 'loops', None)
                if loops is None:
                    continue
                for poly in list(getattr(mesh, 'polygons', []) or []):
                    for loop_index in list(getattr(poly, 'loop_indices', []) or []):
                        vertex_index = int(loops[loop_index].vertex_index)
                        if vertex_index < 0 or vertex_index >= len(source_indices):
                            continue
                        src_index = int(source_indices[vertex_index])
                        rgba = self._sample_color(color_attr, int(loop_index), int(vertex_index), alpha_scale=128.0)
                        bucket = accum.setdefault(src_index, [0, 0, 0, 0])
                        bucket[0] += int(rgba[0])
                        bucket[1] += int(rgba[1])
                        bucket[2] += int(rgba[2])
                        bucket[3] += int(rgba[3])
                        sample_count[src_index] = sample_count.get(src_index, 0) + 1
            except Exception:
                logger.warning('Failed reading BGObject vertex colors from %s for export', getattr(obj, 'name', '<unknown>'))
                continue

            slot_offset = slot_index * per_slot_size
            for src_index, totals in accum.items():
                count = max(1, int(sample_count.get(src_index, 1)))
                r = self._u8(round(totals[0] / count))
                g = self._u8(round(totals[1] / count))
                b = self._u8(round(totals[2] / count))
                a = self._u8(round(totals[3] / count))
                item_offset = slot_offset + (int(src_index) * 4)
                if item_offset + 4 <= len(data):
                    original_a = int(data[item_offset + 3])
                    if a >= 128 and original_a >= 128:
                        a = original_a
                    struct.pack_into('<I', data, item_offset, (a << 24) | (r << 16) | (g << 8) | b)
        return bytes(data)

    def _gather_bg_objects_and_instances(self, collection: Collection, root, root_matrix_inv=None) -> tuple[List[ExportBGObject], List[ExportBGInstance]]:
        objects = list(getattr(collection, 'all_objects', None) or collection.objects)
        scoped_objects = [obj for obj in objects if root is None or obj == root or self._is_descendant_of(obj, root)]
        mesh_objects = [obj for obj in scoped_objects if self._is_bgobject_mesh(obj)]
        bgobject_empties: dict[int, object] = {}
        for obj in scoped_objects:
            if not self._is_bgobject_empty(obj):
                continue
            try:
                bgobject_empties[int(EXPORT_SCENE_OT_trlau_level._bgobject_index_from_object(obj, -1))] = obj
            except Exception:
                continue

        meshes_by_parent: dict[int, dict[object, list[object]]] = {}
        for obj in mesh_objects:
            try:
                bg_index = int(EXPORT_SCENE_OT_trlau_level._bgobject_index_from_object(obj, -1))
            except Exception:
                continue
            if bg_index < 0:
                continue
            parent = getattr(obj, 'parent', None)
            meshes_by_parent.setdefault(bg_index, {}).setdefault(parent, []).append(obj)

        def _parent_score(parent) -> tuple[int, int, str]:
            if parent is None:
                return (3, 0, '')
            try:
                if bool(parent.get('trlau_bgobject_empty')) and not bool(parent.get('trlau_bginstance')):
                    return (0, EXPORT_SCENE_OT_trlau_level._bgobject_index_from_object(parent, 0), getattr(parent, 'name', ''))
                if bool(parent.get('trlau_bginstance_empty')) or str(parent.get('trlau_type', '') or '') == 'BGInstance':
                    return (1, EXPORT_SCENE_OT_trlau_level._bginstance_index_from_object(parent, 0), getattr(parent, 'name', ''))
            except Exception:
                pass
            return (2, 0, getattr(parent, 'name', ''))

        bg_objects: list[ExportBGObject] = []
        for bg_index in sorted(set(meshes_by_parent.keys()) | set(bgobject_empties.keys())):
            parent_groups = meshes_by_parent.get(int(bg_index), {})
            chosen_parent = None
            chosen_meshes: list[object] = []
            if parent_groups:
                chosen_parent = min(parent_groups.keys(), key=_parent_score)
                chosen_meshes = sorted(parent_groups[chosen_parent], key=self._strip_sort_key)
            object_empty = bgobject_empties.get(int(bg_index))
            if object_empty is not None:
                source = object_empty
            elif chosen_parent is not None and self._is_bgobject_empty(chosen_parent) and not self._is_bginstance_empty(chosen_parent):
                source = chosen_parent
            else:
                source = None
            scale = self._bgobject_scale_from_source(source)
            position = self._bgobject_position_from_source(source)
            flags = int(source.get('trlau_bgobject_flags', 0) or 0) if source is not None else 0
            stride = int(source.get('trlau_bgobject_stride', 96) or 96) if source is not None else 96
            cdc_render_data_id = int(source.get('trlau_bgobject_cdc_render_data_id', 0) or 0) if source is not None else 0

            all_bgobject_meshes: list[object] = []
            for _parent, grouped_meshes in parent_groups.items():
                all_bgobject_meshes.extend(list(grouped_meshes or []))

            vertices: list[ExportVertex] = []
            strips: list[ExportStrip] = []
            raw_vertex_data = b''
            raw_color_data = b''
            raw_color_slot_count = 1
            env_mapped_vertices: list[int] = []
            eye_ref_env_mapped_vertices: list[int] = []

            imported_bgobject = None
            if chosen_meshes:
                imported_bgobject = self._export_imported_bgobject_geometry_and_colors(
                    chosen_meshes,
                    all_bgobject_meshes,
                    chosen_parent,
                )
            if imported_bgobject is not None:
                vertices, strips, raw_color_data, raw_color_slot_count = imported_bgobject
            else:
                raw_bgobject = None
                raw_blob_sources: list[object] = []
                for candidate in (source, chosen_parent, bgobject_empties.get(int(bg_index))):
                    if candidate is not None and candidate not in raw_blob_sources:
                        raw_blob_sources.append(candidate)
                for parent in sorted(parent_groups.keys(), key=_parent_score):
                    if parent is not None and parent not in raw_blob_sources:
                        raw_blob_sources.append(parent)
                for candidate in raw_blob_sources:
                    raw_bgobject = self._decode_bgobject_raw_blob(candidate)
                    if raw_bgobject is not None:
                        break
                if raw_bgobject is not None:
                    vertices, strips, raw_vertex_data, raw_color_data, raw_color_slot_count, _raw_env_indices, _raw_eye_indices = raw_bgobject
                    raw_color_data = self._update_bgobject_raw_color_data_from_meshes(
                        vertices,
                        strips,
                        raw_color_data,
                        raw_color_slot_count,
                        all_bgobject_meshes,
                    )
                    if len(vertices) > 0:
                        raw_color_slot_count = max(1, len(raw_color_data) // (len(vertices) * 4)) if raw_color_data else raw_color_slot_count
                else:
                    vertices, strips = self._export_bgobject_geometry(chosen_meshes, chosen_parent)
            for strip in list(strips or []):
                strip.env_mapping = False
                strip.eye_ref_env_mapping = False
            if not vertices or not strips:
                continue
            bg_objects.append(ExportBGObject(
                index=int(bg_index),
                scale=scale,
                position=position,
                flags=flags,
                stride=stride,
                cdc_render_data_id=cdc_render_data_id,
                vertices=vertices,
                strips=strips,
                raw_vertex_data=raw_vertex_data,
                raw_color_data=raw_color_data,
                raw_color_slot_count=raw_color_slot_count,
                env_mapped_vertices=env_mapped_vertices,
                eye_ref_env_mapped_vertices=eye_ref_env_mapped_vertices,
            ))

        object_index_set = {int(bg.index) for bg in bg_objects}
        instance_empties = [obj for obj in scoped_objects if self._is_bginstance_empty(obj)]
        instance_empties.sort(key=lambda obj: (EXPORT_SCENE_OT_trlau_level._bginstance_index_from_object(obj, 0), getattr(obj, 'name', '')))
        bg_by_index = {int(bg.index): bg for bg in bg_objects}
        bg_instances: list[ExportBGInstance] = []
        for default_index, obj in enumerate(instance_empties):
            try:
                bg_index = int(obj.get('trlau_bginstance_bgobject_index', EXPORT_SCENE_OT_trlau_level._bgobject_index_from_object(obj, -1)))
            except Exception:
                continue
            if bg_index not in object_index_set:
                continue
            bg_object = bg_by_index[bg_index]
            matrix_rows = self._bginstance_matrix_rows_from_object(obj, bg_object.scale, root_matrix_inv)
            calculated_radius = self._calculate_bginstance_radius(bg_object, matrix_rows)
            bg_instances.append(ExportBGInstance(
                index=EXPORT_SCENE_OT_trlau_level._bginstance_index_from_object(obj, default_index),
                bg_object_index=int(bg_index),
                instance_id=int(obj.get('trlau_bginstance_id', default_index) or 0),
                flags=int(obj.get('trlau_bginstance_flags', 0) or 0),
                multi_spline_data=EXPORT_SCENE_OT_trlau_level._multi_spline_data_from_object(obj, 'trlau_bginstance_multi_spline_json'),
                original_radius=calculated_radius,
                radius=calculated_radius,
                bg_flags=int(obj.get('trlau_bginstance_bgflags', 0) or 0),
                target_frame=int(obj.get('trlau_bginstance_target_frame', 0) or 0),
                clip_beg=int(obj.get('trlau_bginstance_clip_beg', 0) or 0),
                clip_end=int(obj.get('trlau_bginstance_clip_end', 0) or 0),
                link_seg=int(obj.get('trlau_bginstance_link_seg', 0) or 0),
                color_data_index=int(obj.get('trlau_bginstance_color_data_index', 0) or 0),
                lod=int(obj.get('trlau_bginstance_lod', 0) or 0),
                active_light_bitfield=int(obj.get('trlau_bginstance_active_light_bitfield', 0) or 0),
                matrix_rows=matrix_rows,
            ))
        bg_instances.sort(key=lambda item: int(item.index))
        for normalized_index, item in enumerate(bg_instances):
            item.index = normalized_index
        return bg_objects, bg_instances

    def _export_bgobject_geometry(self, mesh_objects: Sequence[object], parent) -> tuple[list[ExportVertex], list[ExportStrip]]:
        vertices: list[ExportVertex] = []
        strips: list[ExportStrip] = []
        vertex_map: dict[tuple[float, float, float, float, float, int, int, int, int], int] = {}
        parent_inv = None
        if parent is not None:
            try:
                parent_inv = parent.matrix_world.inverted_safe()
            except Exception:
                parent_inv = None

        strip_index = 0
        for obj in mesh_objects:
            mesh, eval_obj = self._evaluated_mesh(obj)
            if mesh is None:
                continue
            try:
                mesh.calc_loop_triangles()
                if not mesh.loop_triangles:
                    continue
                triangles_by_material: dict[int, list[object]] = {}
                for tri in mesh.loop_triangles:
                    material_index = int(getattr(tri, 'material_index', 0))
                    triangles_by_material.setdefault(material_index, []).append(tri)
                for material_index in sorted(triangles_by_material.keys()):
                    strip = self._export_bgobject_strip_geometry(
                        obj,
                        mesh,
                        triangles_by_material[material_index],
                        material_index,
                        parent_inv,
                        vertices,
                        vertex_map,
                        strip_index,
                    )
                    if strip is not None:
                        strips.append(strip)
                        strip_index += 1
            finally:
                self._free_evaluated_mesh(eval_obj)
        return vertices, strips

    def _export_bgobject_strip_geometry(
        self,
        obj,
        mesh,
        triangle_source: Sequence[object],
        material_index: int,
        parent_inv,
        vertices: list[ExportVertex],
        vertex_map: dict[tuple[float, float, float, float, float, int, int, int, int], int],
        strip_index: int,
    ) -> Optional[ExportStrip]:
        uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
        color_attr = self._active_color_attribute(mesh)
        corners: list[dict[str, object]] = []
        triangles: list[tuple[int, int, int]] = []

        for tri in triangle_source:
            face_corner_indices: list[int] = []
            for loop_index in tri.loops:
                vertex_index = mesh.loops[loop_index].vertex_index
                co = obj.matrix_world @ mesh.vertices[vertex_index].co
                if parent_inv is not None:
                    co = parent_inv @ co
                uv = (0.0, 0.0)
                if uv_layer is not None:
                    uv_value = uv_layer[loop_index].uv
                    uv = (float(uv_value.x), 1.0 - float(uv_value.y))
                color = self._sample_color(color_attr, loop_index, vertex_index, alpha_scale=128.0)
                face_corner_indices.append(len(corners))
                corners.append({
                    'vertex_index': int(vertex_index),
                    'position': (float(co.x), float(co.y), float(co.z)),
                    'uv': uv,
                    'color': color,
                })
            if len(face_corner_indices) == 3:
                triangles.append((face_corner_indices[0], face_corner_indices[1], face_corner_indices[2]))

        if not corners or not triangles:
            return None
        normalized_uvs = self._normalize_strip_uvs(corners, triangles, getattr(obj, 'name', 'BGObject'))
        indices: list[int] = []
        for triangle in triangles:
            face_indices: list[int] = []
            for corner_index in triangle:
                corner = corners[corner_index]
                position = corner['position']
                uv = normalized_uvs[corner_index]
                color = corner['color']
                key = (
                    round(float(position[0]), 6),
                    round(float(position[1]), 6),
                    round(float(position[2]), 6),
                    round(float(uv[0]), 6),
                    round(float(uv[1]), 6),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                    int(color[3]),
                )
                vertex_index = vertex_map.get(key)
                if vertex_index is None:
                    vertex_index = len(vertices)
                    vertex_map[key] = vertex_index
                    vertices.append(ExportVertex(
                        position=(float(position[0]), float(position[1]), float(position[2])),
                        uv=(float(uv[0]), float(uv[1])),
                        color=(int(color[0]), int(color[1]), int(color[2]), int(color[3])),
                    ))
                face_indices.append(vertex_index)
            if len(face_indices) == 3 and len(set(face_indices)) == 3:
                indices.extend(face_indices)
        if not indices:
            return None

        material = self._resolve_strip_material(obj, material_index)
        return ExportStrip(
            name=f'{getattr(obj, "name", "BGObject")}_BGStrip_{strip_index:03d}',
            tpageid=self._resolve_strip_tpageid(obj, material=material),
            flags=self._resolve_strip_flags(obj, False, material=material, material_slot_index=material_index),
            indices=indices,
            env_mapping=False,
            eye_ref_env_mapping=False,
        )

    @staticmethod
    def _calculate_bginstance_radius(bg_object: ExportBGObject, matrix_rows: Sequence[Sequence[float]]) -> float:
        vertices = list(getattr(bg_object, 'vertices', []) or [])
        if not vertices:
            return 0.0
        try:
            rows = tuple(tuple(float(matrix_rows[row][col]) for col in range(4)) for row in range(4))
        except Exception:
            rows = (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        scale = tuple(float(value) for value in (getattr(bg_object, 'scale', (1.0, 1.0, 1.0)) or (1.0, 1.0, 1.0)))
        sx = scale[0] if len(scale) > 0 and math.isfinite(scale[0]) else 1.0
        sy = scale[1] if len(scale) > 1 and math.isfinite(scale[1]) else 1.0
        sz = scale[2] if len(scale) > 2 and math.isfinite(scale[2]) else 1.0

        max_sq = 0.0
        for vertex in vertices:
            pos = getattr(vertex, 'position', (0.0, 0.0, 0.0))
            try:
                x = float(pos[0]) * sx
                y = float(pos[1]) * sy
                z = float(pos[2]) * sz
            except Exception:
                continue
            tx = (x * rows[0][0]) + (y * rows[1][0]) + (z * rows[2][0])
            ty = (x * rows[0][1]) + (y * rows[1][1]) + (z * rows[2][1])
            tz = (x * rows[0][2]) + (y * rows[1][2]) + (z * rows[2][2])
            radius_sq = (tx * tx) + (ty * ty) + (tz * tz)
            if math.isfinite(radius_sq) and radius_sq > max_sq:
                max_sq = radius_sq
        if max_sq <= 0.0:
            return 0.0
        return float(math.sqrt(max_sq) * 1.003 + 0.001)


    @staticmethod
    def _bginstance_rows_from_props(obj) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]:
        rows = []
        try:
            for row_index in range(4):
                row = []
                for col_index in range(4):
                    key = f'trlau_bginstance_m{row_index}{col_index}'
                    if key not in obj:
                        raise KeyError(key)
                    row.append(float(obj.get(key)))
                rows.append(tuple(row))
            return tuple(rows)
        except Exception:
            return (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )

    @staticmethod
    def _bginstance_matrix_rows_from_object(obj, scale: Sequence[float], reference_matrix_inv=None) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float], tuple[float, float, float, float]]:
        try:
            sx = float(scale[0]) if len(scale) > 0 and abs(float(scale[0])) > 1e-9 else 1.0
            sy = float(scale[1]) if len(scale) > 1 and abs(float(scale[1])) > 1e-9 else 1.0
            sz = float(scale[2]) if len(scale) > 2 and abs(float(scale[2])) > 1e-9 else 1.0
            basis = mathutils.Matrix((
                (-1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ))
            level_rotation_inv = mathutils.Matrix.Rotation(math.radians(-90.0), 4, 'X')
            scale_inv = mathutils.Matrix.Diagonal((1.0 / sx, 1.0 / sy, 1.0 / sz, 1.0))
            object_matrix = obj.matrix_world.copy()
            if reference_matrix_inv is not None:
                object_matrix = reference_matrix_inv @ object_matrix
            source_matrix = basis @ level_rotation_inv @ object_matrix @ scale_inv
            row_matrix = source_matrix.transposed()
            return tuple(tuple(float(row_matrix[row][col]) for col in range(4)) for row in range(4))
        except Exception:
            return EXPORT_SCENE_OT_trlau_level._bginstance_rows_from_props(obj)


    @staticmethod
    def _intro_data_type_from_metadata(obj) -> int:
        specific_empty = EXPORT_SCENE_OT_trlau_level._child_intro_metadata_empty(obj, specific=True)
        if specific_empty is not None:
            value = trlau_intro_specific_data_type(specific_empty)
            if value > 0:
                return value
            for type_name in (
                str(specific_empty.get('trlau_type', '') or ''),
                getattr(specific_empty, 'name', '') or '',
            ):
                named_type = EXPORT_SCENE_OT_trlau_level._intro_specific_type_from_name(type_name)
                if named_type > 0:
                    return named_type

        return 0

    @staticmethod
    def _iter_descendants(obj):
        stack = list(getattr(obj, 'children', []) or [])
        while stack:
            child = stack.pop(0)
            yield child
            stack[0:0] = list(getattr(child, 'children', []) or [])

    @staticmethod
    def _calculate_intro_max_radius(obj) -> float:
        stored_radius = 0.0
        try:
            intro_data = trlau_intro_data_to_dict(obj)
            stored_radius = max(0.0, float(intro_data.get('max_radius', 0.0) or 0.0)) if intro_data else 0.0
        except Exception:
            stored_radius = 0.0
        try:
            origin_inv = obj.matrix_world.inverted_safe()
        except Exception:
            origin_inv = mathutils.Matrix.Identity(4)

        max_sq = 0.0
        candidates = [obj] + list(EXPORT_SCENE_OT_trlau_level._iter_descendants(obj))
        for child in candidates:
            child_type = getattr(child, 'type', None)
            if child_type == 'EMPTY':
                continue
            try:
                matrix = origin_inv @ child.matrix_world
            except Exception:
                matrix = mathutils.Matrix.Identity(4)

            vertices = []
            data = getattr(child, 'data', None)
            if child_type == 'MESH' and data is not None:
                mesh_vertices = getattr(data, 'vertices', None)
                try:
                    has_vertices = mesh_vertices is not None and len(mesh_vertices) > 0
                except Exception:
                    has_vertices = mesh_vertices is not None
                if has_vertices:
                    vertices = [vertex.co for vertex in mesh_vertices]
            if not vertices:
                bound_box = getattr(child, 'bound_box', None)
                if bound_box:
                    vertices = [mathutils.Vector(corner) for corner in bound_box]
            for vertex in vertices:
                try:
                    point = matrix @ vertex
                    radius_sq = (float(point.x) * float(point.x)) + (float(point.y) * float(point.y)) + (float(point.z) * float(point.z))
                except Exception:
                    continue
                if math.isfinite(radius_sq) and radius_sq > max_sq:
                    max_sq = radius_sq

        if max_sq <= 0.0:
            return stored_radius
        return float(math.sqrt(max_sq) * 1.003 + 0.001)

    @staticmethod
    def _intro_specific_type_from_name(name: str) -> int:
        name_s = str(name or '')
        type_map = {'RewardIntroData': 17, 'RopeObjIntroData': 12, 'WaterVolumeIntroData': 13}
        for type_name, type_id in type_map.items():
            if type_name in name_s:
                return int(type_id)
        match = re.search(r'IntroType(\d+)Data', name_s)
        if match is not None:
            try:
                return int(match.group(1))
            except Exception:
                pass
        return 0

    @staticmethod
    def _child_intro_metadata_empty(parent, *, generic: bool = False, specific: bool = False, data_type: int | None = None):
        for child in getattr(parent, 'children', []) or []:
            child_name = getattr(child, 'name', '') or ''
            if generic and (trlau_intro_is_generic_data(child) or 'GenericIntroData' in child_name):
                return child
            if specific:
                is_specific = trlau_intro_is_specific_data(child) or EXPORT_SCENE_OT_trlau_level._intro_specific_type_from_name(child_name) > 0
                if not is_specific:
                    continue
                child_type = trlau_intro_specific_data_type(child)
                if child_type <= 0:
                    child_type = EXPORT_SCENE_OT_trlau_level._intro_specific_type_from_name(child_name)
                if data_type is None or int(child_type) == int(data_type):
                    return child
        return None

    @staticmethod
    def _intro_metadata_dict(obj, prefix: str, fields: Sequence[str]) -> dict[str, object]:
        if obj is None:
            return {}
        if prefix == 'trlau_intro_generic_':
            typed = trlau_intro_generic_data_to_dict(obj)
            if typed:
                return {field_name: typed.get(field_name) for field_name in fields if field_name in typed}
        if prefix == 'trlau_intro_specific_':
            typed = trlau_intro_specific_data_to_dict(obj)
            if typed:
                return {field_name: typed.get(field_name) for field_name in fields if field_name in typed}
        return {}

    @staticmethod
    def _multi_spline_data_from_object(obj, key: str) -> Optional[dict[str, object]]:
        if obj is None:
            return None
        try:
            panel_data = trlau_multispline_property_to_dict(obj)
        except Exception:
            panel_data = None
        if isinstance(panel_data, dict) and panel_data:
            return panel_data
        value = obj.get(key, None)
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except Exception:
                return None
            return dict(decoded) if isinstance(decoded, dict) else None
        return None

    @staticmethod
    def _gather_intro_data(collection: Collection, root=None, root_matrix_inv=None) -> List[ExportIntroData]:
        objects = getattr(collection, 'all_objects', None) or collection.objects
        intro_empties = [
            obj for obj in objects
            if getattr(obj, 'type', None) == 'EMPTY'
            and ((trlau_intro_is_intro_data(obj) or str(obj.get('trlau_type', '')) == 'IntroData') or _INTRO_NAME_RE.search(getattr(obj, 'name', '') or '') is not None)
            and (trlau_intro_data_to_dict(obj).get('object_id', -1) != -1 or _INTRO_NAME_RE.search(getattr(obj, 'name', '') or '') is not None)
            and (root is None or obj == root or EXPORT_SCENE_OT_trlau_level._is_descendant_of(obj, root))
        ]
        intro_empties.sort(key=lambda obj: EXPORT_SCENE_OT_trlau_level._intro_index_from_object(obj, 0))
        result: List[ExportIntroData] = []
        for obj in intro_empties:
            intro_type = EXPORT_SCENE_OT_trlau_level._intro_data_type_from_metadata(obj)
            specific_intro_data = EXPORT_SCENE_OT_trlau_level._intro_metadata_dict(
                EXPORT_SCENE_OT_trlau_level._child_intro_metadata_empty(obj, specific=True, data_type=intro_type),
                'trlau_intro_specific_',
                INTRO_SPECIFIC_FIELDS_BY_TYPE.get(intro_type, ()),
            )
            if intro_type == 17:
                sound_obj = EXPORT_SCENE_OT_trlau_level._find_intro_sound_empty(collection, obj)
                if sound_obj is not None:
                    sound_id = EXPORT_SCENE_OT_trlau_level._intro_sound_id(sound_obj, int(specific_intro_data.get('sound_id', 0) or 0))
                    if sound_id > 0:
                        specific_intro_data['sound_id'] = sound_id

            intro_data = trlau_intro_data_to_dict(obj)
            result.append(ExportIntroData(
                index=EXPORT_SCENE_OT_trlau_level._intro_index_from_object(obj, len(result)),
                object_id=int(intro_data.get('object_id', -1)) if intro_data else -1,
                intro_num=int(intro_data.get('intro_num', 0)) if intro_data else 0,
                unique_id=int(intro_data.get('unique_id', 0)) if intro_data else 0,
                position=EXPORT_SCENE_OT_trlau_level._relative_translation(obj, root_matrix_inv),
                rotation=EXPORT_SCENE_OT_trlau_level._intro_raw_rotation_from_object(obj, root_matrix_inv),
                dummy1=(0.0, 0.0, 0.0, 0.0),
                dummy2=(0.0, 0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0, 0.0),
                start_frame=int(intro_data.get('start_frame', 0)) if intro_data else 0,
                end_frame=int(intro_data.get('end_frame', 0)) if intro_data else 0,
                intro_flags=int(intro_data.get('intro_flags', 0)) if intro_data else 0,
                attached_vmo=int(intro_data.get('attached_vmo', 0)) if intro_data else 0,
                data=0,
                intro_data_type=intro_type,
                generic_intro_data=EXPORT_SCENE_OT_trlau_level._intro_metadata_dict(EXPORT_SCENE_OT_trlau_level._child_intro_metadata_empty(obj, generic=True), 'trlau_intro_generic_', INTRO_GENERIC_FIELDS),
                specific_intro_data=specific_intro_data,
                multi_spline=int(intro_data.get('multi_spline', 0)) if intro_data else 0,
                multi_spline_data=EXPORT_SCENE_OT_trlau_level._multi_spline_data_from_object(obj, 'trlau_intro_multi_spline_json'),
                max_radius=EXPORT_SCENE_OT_trlau_level._calculate_intro_max_radius(obj),
            ))
        return result

    @staticmethod
    def _normalize_angle(value: float) -> float:
        return ((float(value) + math.pi) % (2.0 * math.pi)) - math.pi

    @staticmethod
    def _intro_rotation_uses_model_basis(obj) -> bool:
        try:
            intro_data = trlau_intro_data_to_dict(obj)
            basis = str(intro_data.get('rotation_basis', '') or '').upper() if intro_data else ''
        except Exception:
            basis = ''
        if basis == 'MODEL_ARMATURE_Z180':
            return True
        if basis == 'RAW_XYZ':
            return False
        try:
            descendants = list(EXPORT_SCENE_OT_trlau_level._iter_descendants(obj))
        except Exception:
            descendants = []
        return any(getattr(child, 'type', '') == 'ARMATURE' for child in descendants)

    @staticmethod
    def _intro_raw_rotation_from_object(obj, root_matrix_inv=None) -> Tuple[float, float, float, float]:
        try:
            intro_data = trlau_intro_data_to_dict(obj)
            stored_rotation = tuple(float(v) for v in intro_data.get('rotation', (0.0, 0.0, 0.0, 0.0))) if intro_data else (0.0, 0.0, 0.0, 0.0)
        except Exception:
            stored_rotation = (0.0, 0.0, 0.0, 0.0)
        raw_w = float(stored_rotation[3]) if len(stored_rotation) > 3 else 0.0
        try:
            matrix = obj.matrix_world.copy()
            if root_matrix_inv is not None:
                matrix = root_matrix_inv @ matrix
            _translation, rotation_quat, _scale = matrix.decompose()
            display_euler = rotation_quat.to_euler('XYZ')
            rx = float(display_euler.x)
            ry = float(display_euler.y)
            rz = float(display_euler.z)
            if EXPORT_SCENE_OT_trlau_level._intro_rotation_uses_model_basis(obj):
                rz = EXPORT_SCENE_OT_trlau_level._normalize_angle(math.pi - rz)
            return (rx, ry, rz, raw_w)
        except Exception:
            if len(stored_rotation) >= 3:
                rx = float(stored_rotation[0])
                ry = float(stored_rotation[1])
                rz = float(stored_rotation[2])
                return (rx, ry, rz, raw_w)
            try:
                e = obj.rotation_euler
                rz = float(e.z)
                if EXPORT_SCENE_OT_trlau_level._intro_rotation_uses_model_basis(obj):
                    rz = EXPORT_SCENE_OT_trlau_level._normalize_angle(math.pi - rz)
                return (float(e.x), float(e.y), rz, raw_w)
            except Exception:
                return (0.0, 0.0, 0.0, raw_w)

    @staticmethod
    def _float_tuple_prop(obj, name: str, default=(0.0, 0.0, 0.0, 0.0)) -> Tuple[float, ...]:
        value = obj.get(name, default)
        if not isinstance(value, Sequence):
            return tuple(float(v) for v in default)
        result = []
        for item in value:
            try:
                result.append(float(item))
            except Exception:
                result.append(0.0)
        if not result:
            return tuple(float(v) for v in default)
        return tuple(result)

classes = (EXPORT_SCENE_OT_trlau_level,)


def menu_func_export(self, _context):
    self.layout.operator(EXPORT_SCENE_OT_trlau_level.bl_idname, text='Tomb Raider LAU Export')


def register():
    configure_logging(False)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
