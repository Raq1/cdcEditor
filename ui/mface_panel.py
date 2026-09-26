from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import bpy

from ..core.game_utils import normalize_game_value
from .level_properties import LEVEL_METADATA_PROP_PREFIX

MFACE_ATTR = 'MFacePoint'
MFACE_SORT_TOP_TO_BOTTOM_KEY = 'trlau_mface_sort_top_to_bottom'
_MAX_MFACE_POINT_VERTICES = 31
_MFACE_CLEANUP_PREFIX = 'MFace'




def _is_model_empty(obj) -> bool:
    try:
        return obj is not None and bool(getattr(obj, 'trlau_is_model_empty', False))
    except Exception:
        return False


def _find_related_model_root(obj, context=None):
    current = obj
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_model_empty(current):
            return current
        current = getattr(current, 'parent', None)

    collections = []
    try:
        collections.extend(list(getattr(obj, 'users_collection', []) or []))
    except Exception:
        pass
    try:
        if context is not None and getattr(context, 'collection', None) is not None:
            collections.append(context.collection)
    except Exception:
        pass
    for collection in collections:
        try:
            candidates = getattr(collection, 'all_objects', None) or getattr(collection, 'objects', [])
        except Exception:
            candidates = []
        for candidate in candidates:
            if _is_model_empty(candidate):
                return candidate
    return None


def _model_game_for_object(obj, context=None) -> str:
    root = _find_related_model_root(obj, context)
    if root is not None:
        try:
            return normalize_game_value(root.get('trlau_model_game_id', 'legend'))
        except Exception:
            pass
    try:
        return normalize_game_value(obj.get('trlau_model_game_id', 'legend'))
    except Exception:
        return 'legend'


def _is_underworld_model_context(obj, context=None) -> bool:
    return _model_game_for_object(obj, context) == 'underworld'


def _has_id_property(obj, key: str) -> bool:
    try:
        return obj is not None and key in obj
    except Exception:
        return False


def _id_property_value(obj, key: str, default=None):
    try:
        return obj.get(key, default)
    except Exception:
        return default


def _has_id_property_prefix(obj, prefixes) -> bool:
    try:
        keys = list(obj.keys())
    except Exception:
        return False
    for key in keys:
        key_s = str(key)
        if any(key_s.startswith(prefix) for prefix in prefixes):
            return True
    return False


_LEVEL_CONTEXT_TYPES = {
    'Level', 'TerrainGroup', 'TerrainLight', 'CameraData', 'BGObject', 'BGInstance',
    'IntroData', 'GenericIntroData', 'IntroSound', 'UnitData', 'ADMDData',
    'Markup', 'SFXMarker', 'SFXSound', 'SFXSpeaker', 'SectionMetadata',
    'AreaDBase', 'AreaDBaseEdges', 'AreaDBasePortals',
}
_LEVEL_CONTEXT_PROP_PREFIXES = (
    LEVEL_METADATA_PROP_PREFIX,
    'trlau_terrain_',
    'trlau_camera_',
    'trlau_bgobject_',
    'trlau_bginstance_',
    'trlau_intro_',
    'trlau_sfx_',
    'trlau_wave_',
    'trlau_unitdata_',
    'trlau_admd_',
    'trlau_area_dbase_',
)
_LEVEL_CONTEXT_EXACT_PROPS = (
    'trlau_level',
    'trlau_level_root',
    'trlau_component_empty',
    'trlau_component_name',
    'trlau_terrain_mesh',
    'trlau_terrain_group',
    'trlau_bgobject',
    'trlau_bgobject_empty',
    'trlau_bgobject_mesh',
    'trlau_bginstance',
    'trlau_bginstance_empty',
    'trlau_section_metadata_empty',
    'trlau_sfx_marker_count',
    'trlau_sfx_wave_id',
)


def _is_level_related_context(obj, context=None) -> bool:
    current = obj
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for key in _LEVEL_CONTEXT_EXACT_PROPS:
            if _has_id_property(current, key):
                value = _id_property_value(current, key, None)
                if isinstance(value, str):
                    if value.strip():
                        return True
                elif value is not None:
                    return True
        obj_type = str(_id_property_value(current, 'trlau_type', '') or '')
        if obj_type in _LEVEL_CONTEXT_TYPES or any(obj_type.startswith(prefix) for prefix in ('AreaDBase',)):
            return True
        if _has_id_property_prefix(current, _LEVEL_CONTEXT_PROP_PREFIXES):
            return True
        current = getattr(current, 'parent', None)
    return False


def _is_render_mesh(obj) -> bool:
    return obj is not None and getattr(obj, 'type', None) == 'MESH'


def _attribute_is_point_int(attr, mesh) -> bool:
    if attr is None:
        return False
    if str(getattr(attr, 'domain', '') or '') != 'POINT':
        return False
    data_type = str(getattr(attr, 'data_type', getattr(attr, 'type', '')) or '')
    if data_type and data_type != 'INT':
        return False
    try:
        return len(attr.data) == len(mesh.vertices)
    except Exception:
        return False


def _get_mface_attr(mesh):
    try:
        attributes = getattr(mesh, 'attributes', None)
        attr = attributes.get(MFACE_ATTR) if attributes is not None else None
    except Exception:
        attr = None
    return attr if _attribute_is_point_int(attr, mesh) else None


def _ensure_mface_attr(mesh):
    attributes = getattr(mesh, 'attributes', None)
    if attributes is None:
        raise ValueError('This Blender version does not expose mesh attributes')
    attr = attributes.get(MFACE_ATTR)
    if attr is not None and not _attribute_is_point_int(attr, mesh):
        try:
            attributes.remove(attr)
        except Exception as exc:
            raise ValueError('MFacePoint exists but is not a POINT-domain integer attribute; delete it and create MFace again') from exc
        attr = None
    if attr is None:
        attr = attributes.new(name=MFACE_ATTR, type='INT', domain='POINT')
    if not _attribute_is_point_int(attr, mesh):
        raise ValueError('MFacePoint must be a POINT-domain integer attribute')
    return attr


def _switch_to_object_mode_for_mesh_writes(obj) -> Optional[str]:
    mode = str(getattr(obj, 'mode', '') or '')
    if mode == 'EDIT':
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
            return 'EDIT'
        except Exception:
            return None
    return None


def _restore_mode(_obj, mode: Optional[str]) -> None:
    if mode:
        try:
            bpy.ops.object.mode_set(mode=mode)
        except Exception:
            pass


def _fit_values_to_mesh(values: Iterable[int], vertex_count: int) -> List[int]:
    fitted = [int(value) for value in values]
    if len(fitted) > vertex_count:
        return fitted[:vertex_count]
    next_id = (max((value for value in fitted if value >= 0), default=-1) + 1)
    while len(fitted) < vertex_count:
        fitted.append(next_id)
        next_id += 1
    return fitted


def _read_mface_values(mesh) -> Optional[List[int]]:
    attr = _get_mface_attr(mesh)
    if attr is None:
        return None
    values: List[int] = []
    for item in attr.data:
        try:
            values.append(int(item.value))
        except Exception:
            values.append(-1)
    return values


def _write_mface_values(obj, values: Iterable[int], *, top_to_bottom: bool = False) -> None:
    restore_mode = _switch_to_object_mode_for_mesh_writes(obj)
    try:
        mesh = obj.data
        values = _fit_values_to_mesh(values, len(mesh.vertices))
        attr = _ensure_mface_attr(mesh)
        try:
            attr.data.foreach_set('value', values)
        except Exception:
            for item, value in zip(attr.data, values):
                item.value = int(value)
        try:
            mesh[MFACE_SORT_TOP_TO_BOTTOM_KEY] = bool(top_to_bottom)
        except Exception:
            pass
        # Remove obsolete source-snapshot properties left by older addon
        # versions.  Current exports always regenerate MFace declarations from
        # the editable mesh and MFacePoint layer.
        for key in (
            'trlau_mface_source_face_points_v2',
            'trlau_mface_source_vertex_count_v2',
            'trlau_mface_source_polygon_count_v2',
            'trlau_mface_source_point_count_v2',
        ):
            _delete_custom_prop(mesh, key)
        mesh.update()
    finally:
        _restore_mode(obj, restore_mode)


def _delete_custom_prop(id_data, key: str) -> None:
    try:
        if id_data is not None and key in id_data:
            del id_data[key]
    except Exception:
        pass


def _delete_mface_data(mesh) -> int:
    removed = 0

    attributes = getattr(mesh, 'attributes', None)
    if attributes is not None:
        try:
            names = [getattr(attr, 'name', '') for attr in attributes]
        except Exception:
            names = []
        for name in names:
            if name == MFACE_ATTR or str(name).startswith(_MFACE_CLEANUP_PREFIX):
                try:
                    attr = attributes.get(name)
                    if attr is not None:
                        attributes.remove(attr)
                        removed += 1
                except Exception:
                    pass

    color_attributes = getattr(mesh, 'color_attributes', None)
    if color_attributes is not None:
        try:
            names = [getattr(attr, 'name', '') for attr in color_attributes]
        except Exception:
            names = []
        for name in names:
            if str(name).startswith(_MFACE_CLEANUP_PREFIX):
                try:
                    attr = color_attributes.get(name)
                    if attr is not None:
                        color_attributes.remove(attr)
                        removed += 1
                except Exception:
                    pass

    legacy_vertex_colors = getattr(mesh, 'vertex_colors', None)
    if legacy_vertex_colors is not None:
        try:
            names = [getattr(attr, 'name', '') for attr in legacy_vertex_colors]
        except Exception:
            names = []
        for name in names:
            if str(name).startswith(_MFACE_CLEANUP_PREFIX):
                try:
                    attr = legacy_vertex_colors.get(name)
                    if attr is not None:
                        legacy_vertex_colors.remove(attr)
                        removed += 1
                except Exception:
                    pass

    for key in list(getattr(mesh, 'keys', lambda: [])()):
        try:
            if str(key).startswith('trlau_mface_'):
                del mesh[key]
        except Exception:
            pass

    try:
        mesh.update()
    except Exception:
        pass
    return removed


def _chunk_indices(indices: Iterable[int], max_size: int = _MAX_MFACE_POINT_VERTICES) -> List[List[int]]:
    ordered = sorted({int(index) for index in indices if int(index) >= 0})
    if not ordered:
        return []
    max_size = max(1, int(max_size))
    return [ordered[start:start + max_size] for start in range(0, len(ordered), max_size)]


def _values_from_groups(vertex_count: int, groups: Iterable[Iterable[int]]) -> Tuple[List[int], int]:
    values = [-1] * int(vertex_count)
    covered = set()
    next_point = 0
    for group in groups:
        for chunk in _chunk_indices(group):
            for index in chunk:
                if 0 <= index < vertex_count and index not in covered:
                    values[index] = int(next_point)
                    covered.add(index)
            next_point += 1
    for index in range(vertex_count):
        if values[index] < 0:
            values[index] = int(next_point)
            next_point += 1
    return values, next_point


def _mesh_position_groups(mesh, tolerance: float) -> List[List[int]]:
    tolerance = max(float(tolerance or 0.0), 1e-9)
    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for vertex in getattr(mesh, 'vertices', []) or []:
        co = vertex.co
        key = (
            int(round(float(co.x) / tolerance)),
            int(round(float(co.y) / tolerance)),
            int(round(float(co.z) / tolerance)),
        )
        grouped[key].append(int(vertex.index))
    return [indices for _key, indices in sorted(grouped.items(), key=lambda item: (min(item[1]) if item[1] else 0, item[0]))]


def _mface_point_average_heights(obj, values: List[int]) -> Dict[int, float]:
    mesh = getattr(obj, 'data', None)
    matrix_world = getattr(obj, 'matrix_world', None)
    if mesh is None:
        return {}
    sums: Dict[int, float] = defaultdict(float)
    counts: Dict[int, int] = defaultdict(int)
    for index, value in enumerate(values):
        point = int(value)
        if point < 0 or index >= len(mesh.vertices):
            continue
        co = mesh.vertices[index].co
        try:
            z = float((matrix_world @ co).z) if matrix_world is not None else float(co.z)
        except Exception:
            z = float(getattr(co, 'z', 0.0))
        sums[point] += z
        counts[point] += 1
    return {point: sums[point] / max(1, counts[point]) for point in sums.keys()}


def _renumber_mface_points_top_to_bottom(obj, values: List[int]) -> List[int]:
    heights = _mface_point_average_heights(obj, values)
    points = sorted({int(value) for value in values if int(value) >= 0}, key=lambda point: (-float(heights.get(point, 0.0)), int(point)))
    remap = {old: new for new, old in enumerate(points)}
    return [remap.get(int(value), int(value)) for value in values]


def _create_mface_from_position(obj, tolerance: float, top_to_bottom: bool) -> Tuple[int, int]:
    restore_mode = _switch_to_object_mode_for_mesh_writes(obj)
    try:
        mesh = obj.data
        groups = _mesh_position_groups(mesh, float(tolerance))
        values, point_count = _values_from_groups(len(mesh.vertices), groups)
        if bool(top_to_bottom):
            values = _renumber_mface_points_top_to_bottom(obj, values)
        _write_mface_values(obj, values, top_to_bottom=top_to_bottom)
        return int(len(mesh.vertices)), int(point_count)
    finally:
        _restore_mode(obj, restore_mode)


class TRLAU_OT_create_mface(bpy.types.Operator):
    bl_idname = 'trlau.create_mface'
    bl_label = 'Create MFace'
    bl_description = 'Create MFacePoint data by grouping vertices at matching local positions'
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: bpy.props.FloatProperty(
        name='Tolerance',
        description='Local-space position tolerance used to consider vertices the same MFace point',
        default=0.00001,
        min=0.000000001,
        soft_min=0.000001,
        soft_max=0.01,
        precision=6,
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return _is_render_mesh(obj) and not _is_level_related_context(obj, context) and not _is_underworld_model_context(obj, context)

    def execute(self, context):
        try:
            vertex_count, point_count = _create_mface_from_position(context.object, float(self.tolerance), False)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Created MFace for {vertex_count} vertex/vertices with {point_count} point(s)')
        return {'FINISHED'}


class TRLAU_OT_create_mface_top_to_bottom(bpy.types.Operator):
    bl_idname = 'trlau.create_mface_top_to_bottom'
    bl_label = 'Create MFace Top to Bottom'
    bl_description = 'Create MFacePoint data by position, then order generated MFace data from top to bottom'
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: bpy.props.FloatProperty(
        name='Tolerance',
        description='Local-space position tolerance used to consider vertices the same MFace point',
        default=0.00001,
        min=0.000000001,
        soft_min=0.000001,
        soft_max=0.01,
        precision=6,
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return _is_render_mesh(obj) and not _is_level_related_context(obj, context) and not _is_underworld_model_context(obj, context)

    def execute(self, context):
        try:
            vertex_count, point_count = _create_mface_from_position(context.object, float(self.tolerance), True)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f'Created top-to-bottom MFace for {vertex_count} vertex/vertices with {point_count} point(s)')
        return {'FINISHED'}


class TRLAU_OT_delete_mface(bpy.types.Operator):
    bl_idname = 'trlau.delete_mface'
    bl_label = 'Delete MFace'
    bl_description = 'Remove MFacePoint data and old MFace authoring leftovers from the selected mesh'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return _is_render_mesh(obj) and not _is_level_related_context(obj, context) and not _is_underworld_model_context(obj, context)

    def execute(self, context):
        obj = context.object
        restore_mode = _switch_to_object_mode_for_mesh_writes(obj)
        try:
            removed = _delete_mface_data(obj.data)
        finally:
            _restore_mode(obj, restore_mode)
        self.report({'INFO'}, 'Deleted MFace data' if removed else 'No MFace data found')
        return {'FINISHED'}


class VIEW3D_PT_trlau_mface(bpy.types.Panel):
    bl_label = 'MFace'
    bl_idname = 'VIEW3D_PT_trlau_mface'
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return _is_render_mesh(obj) and not _is_level_related_context(obj, context) and not _is_underworld_model_context(obj, context)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False
        obj = getattr(context, 'object', None)
        mesh = getattr(obj, 'data', None)
        values = _read_mface_values(mesh) if mesh is not None else None

        if values is None:
            layout.label(text='MFace: none')
        else:
            point_count = len({int(value) for value in values if int(value) >= 0})
            layout.label(text=f'MFace: {point_count} point(s)')

        col = layout.column(align=True)
        col.operator('trlau.create_mface', text='Create MFace')
        col.operator('trlau.create_mface_top_to_bottom', text='Create MFace Top to Bottom')
        col.operator('trlau.delete_mface', text='Delete MFace')

        box = layout.box()
        box.label(text='Create MFace groups matching vertex positions.')
        box.label(text='Use Top to Bottom for new meshes that should dry downward.')


classes = (
    TRLAU_OT_create_mface,
    TRLAU_OT_create_mface_top_to_bottom,
    TRLAU_OT_delete_mface,
    VIEW3D_PT_trlau_mface,
)
