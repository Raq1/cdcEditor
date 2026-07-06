from __future__ import annotations

from .object_utils import trlau_object_type, hmarker_index_from_name
from .hinfo_properties import (
    HINFO_DATA_KEYS,
    TRLAU_HInfoDataProperties,
    cleanup_hinfo_custom_props,
    get_hinfo_int_prop,
    set_hinfo_component_type,
    set_hinfo_int_prop,
)
from typing import Iterable, Optional

import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

HINFO_TYPES = {'HMarker', 'HSphere', 'HBox', 'HCapsule'}
PANEL_TYPES = HINFO_TYPES | {'Target'}
CHILD_OF_NAMES = {
    'HMarker': 'TRLau HMarker Bone Follow',
    'HSphere': 'TRLau HSphere Bone Follow',
    'HBox': 'TRLau HBox Bone Follow',
    'HCapsule': 'TRLau HCapsule Bone Follow',
    'Target': 'TRLau Target Bone Follow',
}

# The whole matrix syncing thing is pretty useless, I'm not sure why I was doing it in the first place
# I guess it's just pretty
_SYNC_GUARD = False
_LAST_SYNC_MATRIX_BY_OBJECT_POINTER = {}


def _object_cache_key(obj):
    try:
        return int(obj.as_pointer())
    except Exception:
        return id(obj)


def _bone_index_from_name(name: str, default: int = 0) -> int:
    text = str(name or '')
    if text.startswith('bone_'):
        suffix = text[5:]
        if suffix.isdigit():
            return int(suffix)
    digits = ''
    for char in reversed(text):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else int(default)


def _constraint_bone_index(obj, default: int = 0) -> int:
    try:
        for constraint in obj.constraints:
            if constraint.type == 'CHILD_OF' and constraint.target is not None and constraint.target.type == 'ARMATURE':
                subtarget = str(getattr(constraint, 'subtarget', '') or '')
                if subtarget:
                    return _bone_index_from_name(subtarget, default)
    except Exception:
        pass
    return int(default)


def _is_hinfo_object(obj) -> bool:
    return obj is not None and trlau_object_type(obj) in HINFO_TYPES


def _is_target_object(obj) -> bool:
    return obj is not None and trlau_object_type(obj) == 'Target'


def _is_panel_object(obj) -> bool:
    return obj is not None and trlau_object_type(obj) in PANEL_TYPES


def _safe_tuple(values: Iterable[float], size: int, default=0.0):
    seq = list(values) if values is not None else []
    if len(seq) < size:
        seq.extend([default] * (size - len(seq)))
    return tuple(float(seq[i]) for i in range(size))


def _bone_name_from_index(index: int) -> str:
    return f'bone_{int(index)}'


def _find_armature(obj):
    if obj is None:
        return None
    for constraint in obj.constraints:
        if constraint.type == 'CHILD_OF' and constraint.target and constraint.target.type == 'ARMATURE':
            return constraint.target
    current = obj.parent
    while current is not None:
        if current.type == 'ARMATURE':
            return current
        current = current.parent
    return None


def _get_child_of_constraint(obj, component_type: str):
    expected_name = CHILD_OF_NAMES.get(component_type)
    for constraint in obj.constraints:
        if constraint.type != 'CHILD_OF' or constraint.target is None or constraint.target.type != 'ARMATURE':
            continue
        if expected_name is None or constraint.name == expected_name:
            return constraint
    return None


def _get_component_bone_index(obj) -> int:
    component_type = trlau_object_type(obj)
    if component_type in {'HMarker', 'HSphere', 'HBox', 'HCapsule'}:
        return _constraint_bone_index(obj, 0)
    if component_type == 'Target':
        return int(obj.get('trlau_segment', _constraint_bone_index(obj, 0)))
    return _constraint_bone_index(obj, 0)




def _get_last_available_bone_index(obj) -> int:
    arm_obj = _find_armature(obj)
    if arm_obj is None or arm_obj.data is None:
        return max(_get_component_bone_index(obj), 0)
    max_index = -1
    for bone in arm_obj.data.bones:
        name = bone.name or ''
        if name.startswith('bone_'):
            suffix = name[5:]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    if max_index >= 0:
        return max_index
    bone_count = len(arm_obj.data.bones)
    return max(bone_count - 1, 0)

def _get_bone(obj, bone_index: Optional[int] = None):
    arm_obj = _find_armature(obj)
    if arm_obj is None or arm_obj.data is None:
        return None
    if bone_index is None:
        bone_index = _get_component_bone_index(obj)
    return arm_obj.data.bones.get(_bone_name_from_index(bone_index))


def _get_reference_world_matrix(obj, bone_index: Optional[int] = None) -> Matrix:
    arm_obj = _find_armature(obj)
    bone = _get_bone(obj, bone_index)
    if arm_obj is not None and bone is not None:
        return arm_obj.matrix_world @ bone.matrix_local
    if obj.parent is not None:
        return obj.parent.matrix_world.copy()
    return Matrix.Identity(4)


def _get_component_local_matrix(obj, bone_index: Optional[int] = None) -> Matrix:
    bone = _get_bone(obj, bone_index)
    if bone is not None:
        return bone.matrix_local.inverted() @ obj.matrix_local.copy()
    return obj.matrix_local.copy()


def _set_local_matrix(obj, local_matrix: Matrix, bone_index: Optional[int] = None):
    bone = _get_bone(obj, bone_index)
    if bone is not None:
        obj.matrix_local = bone.matrix_local @ local_matrix
    else:
        obj.matrix_local = local_matrix


def _update_constraint_target(obj, bone_index: int):
    arm_obj = _find_armature(obj)
    if arm_obj is None:
        return
    constraint = _get_child_of_constraint(obj, trlau_object_type(obj))
    if constraint is None:
        return
    bone_name = _bone_name_from_index(bone_index)
    constraint.target = arm_obj
    constraint.subtarget = bone_name
    bone = arm_obj.data.bones.get(bone_name) if arm_obj.data else None
    parent_world = obj.parent.matrix_world if obj.parent is not None else Matrix.Identity(4)
    if bone is not None:
        constraint.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ parent_world


def _set_component_world_matrix(obj, world_matrix: Matrix, bone_index: Optional[int] = None):
    arm_obj = _find_armature(obj)
    bone = _get_bone(obj, bone_index)
    parent_world = obj.parent.matrix_world.copy() if obj.parent is not None else Matrix.Identity(4)
    constraint = _get_child_of_constraint(obj, trlau_object_type(obj))
    if constraint is not None:
        was_enabled = getattr(constraint, 'enabled', True)
        try:
            constraint.enabled = False
        except Exception:
            was_enabled = None
        obj.matrix_world = world_matrix
        if bone is not None and arm_obj is not None:
            local_matrix = bone.matrix_local.inverted() @ arm_obj.matrix_world.inverted() @ world_matrix
            obj.matrix_local = parent_world.inverted() @ (arm_obj.matrix_world @ bone.matrix_local @ local_matrix)
        else:
            obj.matrix_local = parent_world.inverted() @ world_matrix
        if bone is not None and arm_obj is not None:
            constraint.target = arm_obj
            constraint.subtarget = _bone_name_from_index(bone_index if bone_index is not None else _get_component_bone_index(obj))
            try:
                constraint.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ parent_world
            except Exception:
                pass
        if was_enabled is not None:
            try:
                constraint.enabled = was_enabled
            except Exception:
                pass
        try:
            bpy.context.view_layer.update()
            obj.matrix_world = world_matrix
        except Exception:
            pass
        return
    obj.matrix_world = world_matrix
    if bone is not None and arm_obj is not None:
        local_matrix = bone.matrix_local.inverted() @ arm_obj.matrix_world.inverted() @ world_matrix
        obj.matrix_local = bone.matrix_local @ local_matrix
    else:
        obj.matrix_local = parent_world.inverted() @ world_matrix




def _scale_matrix(scale_values) -> Matrix:
    scale = _safe_tuple(scale_values, 3, 1.0)
    return Matrix.Diagonal((float(scale[0]), float(scale[1]), float(scale[2]), 1.0))


def _compose_local_matrix(location, rotation=None, scale_values=(1.0, 1.0, 1.0)) -> Matrix:
    matrix = Matrix.Translation(Vector(_safe_tuple(location, 3)))
    if rotation is not None:
        matrix = matrix @ rotation.to_matrix().to_4x4()
    return matrix @ _scale_matrix(scale_values)


def _tag_view3d_redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            area.tag_redraw()
            for region in area.regions:
                if region.type == 'UI':
                    region.tag_redraw()


def _get_eval_local_matrix(obj, depsgraph=None, bone_index: Optional[int] = None) -> Matrix:
    try:
        parent_space_matrix = obj.matrix_local.copy()
    except Exception:
        parent_space_matrix = Matrix.Identity(4)

    bone = _get_bone(obj, bone_index)
    if bone is not None:
        try:
            return bone.matrix_local.inverted_safe() @ parent_space_matrix
        except Exception:
            return bone.matrix_local.inverted() @ parent_space_matrix

    return parent_space_matrix


def _matrix_signature(matrix: Matrix):
    return tuple(round(float(v), 6) for row in matrix for v in row)


def _store_last_sync_matrix(obj, depsgraph=None):
    matrix = _get_eval_local_matrix(obj, depsgraph)
    _LAST_SYNC_MATRIX_BY_OBJECT_POINTER[_object_cache_key(obj)] = _matrix_signature(matrix)
    try:
        if 'trlau_last_sync_matrix' in obj:
            del obj['trlau_last_sync_matrix']
    except Exception:
        pass
    return matrix


def _object_transform_changed(obj, depsgraph=None) -> bool:
    current = _matrix_signature(_get_eval_local_matrix(obj, depsgraph))
    previous = _LAST_SYNC_MATRIX_BY_OBJECT_POINTER.get(_object_cache_key(obj))
    if previous is None:
        _LAST_SYNC_MATRIX_BY_OBJECT_POINTER[_object_cache_key(obj)] = current
        return False
    return tuple(current) != tuple(previous)


def _quat_from_xyzw(values) -> Quaternion:
    x, y, z, w = _safe_tuple(values, 4)
    quat = Quaternion((w, x, y, z))
    if quat.length <= 1e-8:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    quat.normalize()
    return quat


def _xyzw_from_quat(quat: Quaternion):
    quat = quat.normalized()
    return (float(quat.x), float(quat.y), float(quat.z), float(quat.w))


def _local_decompose(obj, depsgraph=None):
    try:
        return _get_eval_local_matrix(obj, depsgraph).decompose()
    except Exception:
        return Vector((0.0, 0.0, 0.0)), Quaternion((1.0, 0.0, 0.0, 0.0)), Vector((1.0, 1.0, 1.0))


def _get_marker_position(self):
    loc, _rot, _scale = _local_decompose(self)
    return (float(loc.x), float(loc.y), float(loc.z))


def _set_marker_position(self, value):
    obj = self
    pos = Vector(_safe_tuple(value, 3))
    _loc, rot, _scale = _local_decompose(obj)
    _set_local_matrix(obj, Matrix.Translation(pos) @ rot.to_matrix().to_4x4())
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_marker_rotation(self):
    _loc, rot, _scale = _local_decompose(self)
    euler = rot.to_euler('ZYX')
    return (float(euler.x), float(euler.y), float(euler.z))


def _set_marker_rotation(self, value):
    obj = self
    rot = Euler(_safe_tuple(value, 3), 'ZYX')
    loc, _old_rot, _scale = _local_decompose(obj)
    _set_local_matrix(obj, Matrix.Translation(loc) @ rot.to_matrix().to_4x4())
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_target_position(self):
    return _safe_tuple(self.get('trlau_target_position', (0.0, 0.0, 0.0)), 3)


def _set_target_position(self, value):
    obj = self
    pos = Vector(_safe_tuple(value, 3))
    obj['trlau_target_position'] = tuple(float(v) for v in pos)
    euler = Euler(_safe_tuple(obj.get('trlau_target_rotation', (0.0, 0.0, 0.0)), 3), 'ZYX')
    _set_local_matrix(obj, Matrix.Translation(pos) @ euler.to_matrix().to_4x4())
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_target_rotation(self):
    return _safe_tuple(self.get('trlau_target_rotation', (0.0, 0.0, 0.0)), 3)


def _set_target_rotation(self, value):
    obj = self
    rot = Euler(_safe_tuple(value, 3), 'ZYX')
    obj['trlau_target_rotation'] = tuple(float(v) for v in rot)
    pos = Vector(_safe_tuple(obj.get('trlau_target_position', (0.0, 0.0, 0.0)), 3))
    _set_local_matrix(obj, Matrix.Translation(pos) @ rot.to_matrix().to_4x4())
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_hsphere_position(self):
    loc, _rot, _scale = _local_decompose(self)
    return (int(round(float(loc.x))), int(round(float(loc.y))), int(round(float(loc.z))))


def _set_hsphere_position(self, value):
    obj = self
    pos = Vector(tuple(float(int(round(float(v)))) for v in _safe_tuple(value, 3, 0)))
    _loc, _rot, scale = _local_decompose(obj)
    radius = max(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z)), 1e-6)
    local_matrix = Matrix.Translation(pos) @ Matrix.Diagonal((radius, radius, radius, 1.0))
    _set_local_matrix(obj, local_matrix)
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_hsphere_radius(self):
    _loc, _rot, scale = _local_decompose(self)
    return max(int(round(max(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z))))), 0)


def _set_hsphere_radius(self, value):
    obj = self
    radius = max(int(round(float(value))), 0)
    loc, _rot, _scale = _local_decompose(obj)
    radius_f = max(float(radius), 1e-6)
    _set_local_matrix(obj, Matrix.Translation(loc) @ Matrix.Diagonal((radius_f, radius_f, radius_f, 1.0)))
    try:
        set_hinfo_int_prop(obj, 'trlau_hsphere_radius_sq', int(radius) * int(radius))
    except Exception:
        pass
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_hbox_position(self):
    loc, _rot, _scale = _local_decompose(self)
    return (float(loc.x), float(loc.y), float(loc.z))


def _apply_component_local_transform(obj, local_matrix: Matrix):
    bone = _get_bone(obj)
    parent_space_matrix = bone.matrix_local @ local_matrix if bone is not None else local_matrix.copy()
    location, rotation, scale = parent_space_matrix.decompose()
    obj.location = location
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = rotation
    obj.scale = scale
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass


def _set_hbox_position(self, value):
    obj = self
    pos = Vector(_safe_tuple(value, 3))
    _loc, rot, scale = _local_decompose(obj)
    local_matrix = Matrix.Translation(pos) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal((max(abs(float(scale.x)), 1e-6), max(abs(float(scale.y)), 1e-6), max(abs(float(scale.z)), 1e-6), 1.0))
    _set_local_matrix(obj, local_matrix)
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_hbox_quaternion(self):
    _loc, rot, _scale = _local_decompose(self)
    return _xyzw_from_quat(rot)


def _set_hbox_quaternion(self, value):
    obj = self
    quat = _quat_from_xyzw(_safe_tuple(value, 4))
    loc, _old_rot, scale = _local_decompose(obj)
    local_matrix = Matrix.Translation(loc) @ quat.to_matrix().to_4x4() @ Matrix.Diagonal((max(abs(float(scale.x)), 1e-6), max(abs(float(scale.y)), 1e-6), max(abs(float(scale.z)), 1e-6), 1.0))
    _set_local_matrix(obj, local_matrix)
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_hbox_dimensions(self):
    _loc, _rot, scale = _local_decompose(self)
    return (abs(float(scale.x)) * 2.0, abs(float(scale.y)) * 2.0, abs(float(scale.z)) * 2.0, 1.0)


def _set_hbox_dimensions(self, value):
    obj = self
    dims = list(_safe_tuple(value, 4))
    loc, rot, _scale = _local_decompose(obj)
    local_matrix = Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal((max(abs(float(dims[0])) * 0.5, 1e-6), max(abs(float(dims[1])) * 0.5, 1e-6), max(abs(float(dims[2])) * 0.5, 1e-6), 1.0))
    _set_local_matrix(obj, local_matrix)
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()



def _hcapsule_shape_from_transform(obj):
    loc, rot, scale = _local_decompose(obj)
    length = max(abs(float(scale.z)) * 2.0, 0.0)
    radius = max(abs(float(scale.x)), abs(float(scale.y)), 0.0)
    axis = rot @ Vector((0.0, 0.0, 1.0))
    if axis.length <= 1e-8:
        axis = Vector((0.0, 0.0, 1.0))
    else:
        axis.normalize()
    half = axis * (length * 0.5)
    return loc, rot, length, radius, loc - half, loc + half


def _set_hcapsule_shape(obj, midpoint: Vector, quat: Quaternion, length: float, radius: float):
    local_matrix = (
        Matrix.Translation(midpoint)
        @ quat.to_matrix().to_4x4()
        @ Matrix.Diagonal((max(float(radius), 1e-6), max(float(radius), 1e-6), max(float(length) * 0.5, 1e-6), 1.0))
    )
    _set_local_matrix(obj, local_matrix)
    _strip_hinfo_derived_props(obj)
    _store_last_sync_matrix(obj)
    _tag_view3d_redraw()


def _get_hcapsule_position(self):
    loc, _rot, _length, _radius, _start, _end = _hcapsule_shape_from_transform(self)
    return (float(loc.x), float(loc.y), float(loc.z))


def _get_hcapsule_quaternion(self):
    _loc, rot, _length, _radius, _start, _end = _hcapsule_shape_from_transform(self)
    return _xyzw_from_quat(rot)


def _set_hcapsule_position(self, value):
    obj = self
    _loc, rot, length, radius, _start, _end = _hcapsule_shape_from_transform(obj)
    _set_hcapsule_shape(obj, Vector(_safe_tuple(value, 3)), rot, length, radius)


def _set_hcapsule_quaternion(self, value):
    obj = self
    loc, _rot, length, radius, _start, _end = _hcapsule_shape_from_transform(obj)
    quat = _quat_from_xyzw(_safe_tuple(value, 4))
    _set_hcapsule_shape(obj, loc, quat, length, radius)


def _capsule_apply_endpoints(obj, start: Vector, end: Vector, update_position_quaternion: bool = True):
    direction = end - start
    length = direction.length
    axis = direction.normalized() if length > 1e-8 else Vector((0.0, 0.0, 1.0))
    rotation = axis.to_track_quat('Z', 'Y')
    midpoint = start.lerp(end, 0.5)
    _loc, _rot, _old_length, radius, _old_start, _old_end = _hcapsule_shape_from_transform(obj)
    _set_hcapsule_shape(obj, midpoint, rotation, length, radius)


def _get_hcapsule_start(self):
    _loc, _rot, _length, _radius, start, _end = _hcapsule_shape_from_transform(self)
    return (float(start.x), float(start.y), float(start.z))


def _set_hcapsule_start(self, value):
    obj = self
    start = Vector(_safe_tuple(value, 3))
    _loc, _rot, _length, _radius, _old_start, end = _hcapsule_shape_from_transform(obj)
    _capsule_apply_endpoints(obj, start, end)


def _get_hcapsule_end(self):
    _loc, _rot, _length, _radius, _start, end = _hcapsule_shape_from_transform(self)
    return (float(end.x), float(end.y), float(end.z))


def _set_hcapsule_end(self, value):
    obj = self
    _loc, _rot, _length, _radius, start, _old_end = _hcapsule_shape_from_transform(obj)
    end = Vector(_safe_tuple(value, 3))
    _capsule_apply_endpoints(obj, start, end)


def _get_hcapsule_radius(self):
    _loc, _rot, _length, radius, _start, _end = _hcapsule_shape_from_transform(self)
    return int(round(float(radius)))


def _set_hcapsule_radius(self, value):
    obj = self
    loc, rot, length, _old_radius, _start, _end = _hcapsule_shape_from_transform(obj)
    radius = max(int(round(float(value))), 0)
    _set_hcapsule_shape(obj, loc, rot, length, float(radius))


def _get_hcapsule_length_raw(self):
    _loc, _rot, length, _radius, _start, _end = _hcapsule_shape_from_transform(self)
    return int(round(float(length)))


def _set_hcapsule_length_raw(self, value):
    obj = self
    loc, rot, _length, radius, _start, _end = _hcapsule_shape_from_transform(obj)
    new_length = max(int(round(float(value))), 0)
    _set_hcapsule_shape(obj, loc, rot, float(new_length), radius)




def _get_hmarker_index(self):
    return get_hinfo_int_prop(self, 'trlau_marker_index', hmarker_index_from_name(self, 0), min_value=0)


def _set_hmarker_index(self, value):
    obj = self
    if trlau_object_type(obj) != 'HMarker':
        return
    try:
        index = max(int(value), 0)
    except Exception:
        index = 0
    name = str(getattr(obj, 'name', '') or '')
    stem = name.split('.', 1)[0]
    if '_HMarker_' in stem:
        prefix = stem.split('_HMarker_', 1)[0]
        obj.name = f'{prefix}_HMarker_{index}'
    else:
        obj.name = f'HMarker_{index}'
    set_hinfo_component_type(obj, 'HMarker')
    set_hinfo_int_prop(obj, 'trlau_marker_index', index)
    cleanup_hinfo_custom_props(obj)

def _get_component_bone(self):
    return min(max(_get_component_bone_index(self), 0), _get_last_available_bone_index(self))


def _set_component_bone(self, value):
    obj = self
    if not _is_panel_object(obj):
        return
    current_bone = _get_component_bone_index(obj)
    new_bone = min(max(int(value), 0), _get_last_available_bone_index(obj))
    keep_position = bool(getattr(obj, 'trlau_keep_position', False))
    world_matrix = obj.matrix_world.copy() if keep_position else None
    local_matrix = _get_eval_local_matrix(obj, None, current_bone)
    if trlau_object_type(obj) == 'Target':
        obj['trlau_segment'] = new_bone
    else:
        _strip_hinfo_derived_props(obj)
    _update_constraint_target(obj, new_bone)
    if keep_position and world_matrix is not None:
        _set_component_world_matrix(obj, world_matrix, new_bone)
    else:
        _set_local_matrix(obj, local_matrix, new_bone)
    sync_hinfo_object(obj)
    _tag_view3d_redraw()


def _strip_hinfo_derived_props(obj):
    for key in (
        'trlau_hsphere_x', 'trlau_hsphere_y', 'trlau_hsphere_z', 'trlau_hsphere_radius',
        'trlau_hbox_position', 'trlau_hbox_quaternion', 'trlau_hbox_dimensions',
        'trlau_hcapsule_radius_sq', 'trlau_hcapsule_length_raw', 'trlau_hcapsule_position',
        'trlau_hcapsule_quaternion', 'trlau_hcapsule_start', 'trlau_hcapsule_end', 'trlau_hcapsule_length',
        'trlau_marker_position', 'trlau_marker_rotation',
        'trlau_last_sync_matrix',
    ):
        if key in HINFO_DATA_KEYS:
            continue
        try:
            if key in obj:
                del obj[key]
        except Exception:
            pass
    cleanup_hinfo_custom_props(obj)


def _sync_marker_metadata_from_object(obj, depsgraph=None):
    _strip_hinfo_derived_props(obj)


def _sync_hsphere_metadata_from_object(obj, depsgraph=None):
    _strip_hinfo_derived_props(obj)


def _sync_hbox_metadata_from_object(obj, depsgraph=None):
    _strip_hinfo_derived_props(obj)


def _sync_hcapsule_metadata_from_object(obj, depsgraph=None):
    obj.lock_rotation = (False, False, False)
    obj.lock_rotation_w = False
    obj.lock_rotations_4d = False
    _strip_hinfo_derived_props(obj)


def _sync_target_metadata_from_object(obj, depsgraph=None):
    local_matrix = _get_eval_local_matrix(obj, depsgraph)
    loc, rot, _scale = local_matrix.decompose()
    obj['trlau_target_position'] = tuple(float(v) for v in loc)
    obj['trlau_target_rotation'] = tuple(float(v) for v in rot.to_euler('ZYX'))



def sync_hinfo_object(obj, depsgraph=None):
    if not (_is_hinfo_object(obj) or _is_target_object(obj)):
        return
    component_type = trlau_object_type(obj)
    if component_type == 'HMarker':
        _sync_marker_metadata_from_object(obj, depsgraph)
    elif component_type == 'HSphere':
        _sync_hsphere_metadata_from_object(obj, depsgraph)
    elif component_type == 'HBox':
        _sync_hbox_metadata_from_object(obj, depsgraph)
    elif component_type == 'HCapsule':
        _sync_hcapsule_metadata_from_object(obj, depsgraph)
    elif component_type == 'Target':
        _sync_target_metadata_from_object(obj, depsgraph)
    _store_last_sync_matrix(obj, depsgraph)


def sync_hinfo_scene(_scene=None, depsgraph=None):
    global _SYNC_GUARD
    if _SYNC_GUARD:
        return
    _SYNC_GUARD = True
    try:
        changed = False
        if depsgraph is not None:
            seen = set()
            for update in depsgraph.updates:
                obj = getattr(update, 'id', None)
                if not isinstance(obj, bpy.types.Object):
                    continue
                if obj.name in seen:
                    continue
                seen.add(obj.name)
                if not (_is_hinfo_object(obj) or _is_target_object(obj)):
                    continue
                if _object_cache_key(obj) not in _LAST_SYNC_MATRIX_BY_OBJECT_POINTER:
                    if _is_hinfo_object(obj):
                        _strip_hinfo_derived_props(obj)
                    _store_last_sync_matrix(obj, depsgraph)
                    continue
                if _object_transform_changed(obj, depsgraph):
                    sync_hinfo_object(obj, depsgraph)
                    changed = True
        else:
            for obj in bpy.data.objects:
                if not (_is_hinfo_object(obj) or _is_target_object(obj)):
                    continue
                if _object_cache_key(obj) not in _LAST_SYNC_MATRIX_BY_OBJECT_POINTER:
                    if _is_hinfo_object(obj):
                        _strip_hinfo_derived_props(obj)
                    _store_last_sync_matrix(obj)
                    continue
                if _object_transform_changed(obj):
                    sync_hinfo_object(obj)
                    changed = True
        if changed:
            _tag_view3d_redraw()
    finally:
        _SYNC_GUARD = False
_LAST_SYNC_MATRIX_BY_OBJECT_POINTER = {}


def _object_cache_key(obj):
    try:
        return int(obj.as_pointer())
    except Exception:
        return id(obj)


def _bone_index_from_name(name: str, default: int = 0) -> int:
    text = str(name or '')
    if text.startswith('bone_'):
        suffix = text[5:]
        if suffix.isdigit():
            return int(suffix)
    digits = ''
    for char in reversed(text):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else int(default)


def _constraint_bone_index(obj, default: int = 0) -> int:
    try:
        for constraint in obj.constraints:
            if constraint.type == 'CHILD_OF' and constraint.target is not None and constraint.target.type == 'ARMATURE':
                subtarget = str(getattr(constraint, 'subtarget', '') or '')
                if subtarget:
                    return _bone_index_from_name(subtarget, default)
    except Exception:
        pass
    return int(default)





def _timer_sync_hinfo_scene():
    try:
        sync_hinfo_scene()
    except Exception:
        pass
    return 0.1

def register_hinfo_properties():
    try:
        bpy.utils.register_class(TRLAU_HInfoDataProperties)
    except ValueError:
        pass
    if not hasattr(bpy.types.Object, 'trlau_hinfo_data'):
        bpy.types.Object.trlau_hinfo_data = bpy.props.PointerProperty(type=TRLAU_HInfoDataProperties)
    bpy.types.Object.trlau_keep_position = bpy.props.BoolProperty(name='Keep Position', description='Preserve current world position when changing assigned bone', default=False)
    bpy.types.Object.trlau_hinfo_bone = bpy.props.IntProperty(
        name='Bone', get=_get_component_bone, set=_set_component_bone,
    )
    bpy.types.Object.trlau_hmarker_index_ui = bpy.props.IntProperty(
        name='Index', min=0, get=_get_hmarker_index, set=_set_hmarker_index,
    )
    bpy.types.Object.trlau_hinfo_position = bpy.props.FloatVectorProperty(
        name='Position', size=3, get=_get_marker_position, set=_set_marker_position,
    )
    bpy.types.Object.trlau_hinfo_rotation = bpy.props.FloatVectorProperty(
        name='Rotation', size=3, get=_get_marker_rotation, set=_set_marker_rotation,
    )
    bpy.types.Object.trlau_hsphere_position_ui = bpy.props.IntVectorProperty(
        name='Position', size=3, get=_get_hsphere_position, set=_set_hsphere_position,
    )
    bpy.types.Object.trlau_hsphere_radius_ui = bpy.props.IntProperty(
        name='Radius', min=0, get=_get_hsphere_radius, set=_set_hsphere_radius,
    )
    bpy.types.Object.trlau_hbox_position_ui = bpy.props.FloatVectorProperty(
        name='Position', size=3, get=_get_hbox_position, set=_set_hbox_position,
    )
    bpy.types.Object.trlau_hbox_quaternion_ui = bpy.props.FloatVectorProperty(
        name='Quaternion', size=4, subtype='QUATERNION', get=_get_hbox_quaternion, set=_set_hbox_quaternion,
    )
    bpy.types.Object.trlau_hbox_dimensions_ui = bpy.props.FloatVectorProperty(
        name='Dimensions', size=4, get=_get_hbox_dimensions, set=_set_hbox_dimensions,
    )
    bpy.types.Object.trlau_hcapsule_position_ui = bpy.props.FloatVectorProperty(
        name='Position', size=3, get=_get_hcapsule_position, set=_set_hcapsule_position,
    )
    bpy.types.Object.trlau_hcapsule_quaternion_ui = bpy.props.FloatVectorProperty(
        name='Quaternion', size=4, subtype='QUATERNION', get=_get_hcapsule_quaternion, set=_set_hcapsule_quaternion,
    )
    bpy.types.Object.trlau_hcapsule_start_ui = bpy.props.FloatVectorProperty(
        name='Start', size=3, get=_get_hcapsule_start, set=_set_hcapsule_start,
    )
    bpy.types.Object.trlau_hcapsule_end_ui = bpy.props.FloatVectorProperty(
        name='End', size=3, get=_get_hcapsule_end, set=_set_hcapsule_end,
    )
    bpy.types.Object.trlau_hcapsule_radius_ui = bpy.props.IntProperty(
        name='Radius', min=0, get=_get_hcapsule_radius, set=_set_hcapsule_radius,
    )
    bpy.types.Object.trlau_hcapsule_length_raw_ui = bpy.props.IntProperty(
        name='Length', min=0, get=_get_hcapsule_length_raw, set=_set_hcapsule_length_raw,
    )
    bpy.types.Object.trlau_target_position_ui = bpy.props.FloatVectorProperty(
        name='Position', size=3, get=_get_target_position, set=_set_target_position,
    )
    bpy.types.Object.trlau_target_rotation_ui = bpy.props.FloatVectorProperty(
        name='Rotation', size=3, get=_get_target_rotation, set=_set_target_rotation,
    )


def unregister_hinfo_properties():
    property_names = (
        'trlau_hinfo_data',
        'trlau_keep_position',
        'trlau_hinfo_bone',
        'trlau_hmarker_index_ui',
        'trlau_hinfo_position',
        'trlau_hinfo_rotation',
        'trlau_hsphere_position_ui',
        'trlau_hsphere_radius_ui',
        'trlau_hbox_position_ui',
        'trlau_hbox_quaternion_ui',
        'trlau_hbox_dimensions_ui',
        'trlau_hcapsule_position_ui',
        'trlau_hcapsule_quaternion_ui',
        'trlau_hcapsule_start_ui',
        'trlau_hcapsule_end_ui',
        'trlau_hcapsule_radius_ui',
        'trlau_hcapsule_length_raw_ui',
        'trlau_target_position_ui',
        'trlau_target_rotation_ui',
    )
    for name in property_names:
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)
    try:
        bpy.utils.unregister_class(TRLAU_HInfoDataProperties)
    except Exception:
        pass
