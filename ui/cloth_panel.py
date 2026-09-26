from __future__ import annotations

from ..core.object_utils import trlau_object_type, hmarker_index_from_name
import bpy
import mathutils
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from ..core.blender_mesh_utils import lock_object_rotation
from .level_properties import _find_related_model_root

def _cloth_base_name_from_armature(arm_obj) -> str:
    name = str(getattr(arm_obj, 'name', '') or 'TRLAU')
    for suffix in ('_Armature', '_Skeleton'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name or 'TRLAU'


def _find_cloth_root_for_armature(arm_obj):
    if arm_obj is None:
        return None
    for child in list(getattr(arm_obj, 'children', []) or []):
        try:
            if trlau_object_type(child) == 'Cloth':
                return child
        except Exception:
            pass
    return None


def _preferred_cloth_collection_for_armature(context, arm_obj):
    model_root = _find_related_model_root(arm_obj, context) if arm_obj is not None else None
    model_collections = list(getattr(model_root, 'users_collection', []) or []) if model_root is not None else []
    arm_collections = list(getattr(arm_obj, 'users_collection', []) or []) if arm_obj is not None else []

    if model_collections:
        for collection in model_collections:
            if collection in arm_collections:
                return collection
        return model_collections[0]

    if arm_collections:
        return arm_collections[0]

    collection = getattr(context, 'collection', None) if context is not None else None
    if collection is not None:
        return collection

    scene = getattr(context, 'scene', None) if context is not None else None
    return getattr(scene, 'collection', None) or bpy.context.scene.collection


def _link_object_to_collection(obj, collection) -> None:
    if obj is None or collection is None:
        return
    try:
        if all(existing != collection for existing in getattr(obj, 'users_collection', []) or []):
            collection.objects.link(obj)
    except Exception:
        pass


_STALE_CLOTH_ROOT_PROPS = (
    'trlau_cloth_gravity',
    'trlau_cloth_drag',
    'trlau_cloth_preview_enabled',
)


def _cleanup_cloth_root_custom_props(root) -> None:
    for key in _STALE_CLOTH_ROOT_PROPS:
        try:
            if root is not None and key in root:
                del root[key]
        except Exception:
            pass


def _get_uniform_cloth_radius(obj) -> float:
    try:
        scale = getattr(obj, 'scale', (1.0, 1.0, 1.0))
        return max(abs(float(scale[0])), abs(float(scale[1])), abs(float(scale[2])))
    except Exception:
        return 1.0


def _set_uniform_cloth_radius(obj, value) -> None:
    try:
        radius = max(0.0, float(value))
    except Exception:
        radius = 0.0
    try:
        obj.scale = (radius, radius, radius)
    except Exception:
        pass


def register_cloth_panel_properties() -> None:
    if not hasattr(bpy.types.Object, 'trlau_cloth_radius_ui'):
        bpy.types.Object.trlau_cloth_radius_ui = FloatProperty(
            name='Radius',
            description='Uniform transform scale used as the cloth collision or collision-rule radius',
            min=0.0,
            soft_min=0.0,
            soft_max=128.0,
            options={'SKIP_SAVE'},
            get=_get_uniform_cloth_radius,
            set=_set_uniform_cloth_radius,
        )


def unregister_cloth_panel_properties() -> None:
    if hasattr(bpy.types.Object, 'trlau_cloth_radius_ui'):
        try:
            del bpy.types.Object.trlau_cloth_radius_ui
        except Exception:
            pass


def _ensure_cloth_root(context, arm_obj):
    collection = _preferred_cloth_collection_for_armature(context, arm_obj)
    root = _find_cloth_root_for_armature(arm_obj)
    if root is not None:
        _link_object_to_collection(root, collection)
        _cleanup_cloth_root_custom_props(root)
        return root
    base_name = _cloth_base_name_from_armature(arm_obj)
    root = bpy.data.objects.new(f'{base_name}_Cloth', None)
    _link_object_to_collection(root, collection)
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 32.0
    root.parent = arm_obj
    root.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    _cleanup_cloth_root_custom_props(root)
    return root

def _cloth_collision_name_base(root) -> str:
    name = str(getattr(root, 'name', '') or 'TRLAU_Cloth')
    if '.' in name and name.rsplit('.', 1)[1].isdigit():
        name = name.rsplit('.', 1)[0]
    if name.endswith('_Cloth'):
        name = name[:-len('_Cloth')]
    return name or 'TRLAU'


def _safe_cloth_name_fragment(value: str, default: str = 'Chain') -> str:
    text = str(value or '').strip()
    safe = ''.join(char if char.isalnum() or char in {'_', '-'} else '_' for char in text)
    safe = safe.strip('_')
    return safe or str(default)


def _cloth_collision_rules_group_name(root, chain_root_segment: int | None = None, chain_root_bone_name: str = '') -> str:
    base = f'{_cloth_collision_name_base(root)}_Cloth_CollisionRules'
    if chain_root_segment is None:
        return base
    bone_part = _safe_cloth_name_fragment(chain_root_bone_name, f'B{int(chain_root_segment):03d}')
    return f'{base}_B{int(chain_root_segment):03d}_{bone_part}'


def _is_object_cloth_collision_rules_group(obj) -> bool:
    try:
        if trlau_object_type(obj) == 'ClothCollisionRules':
            return True
    except Exception:
        pass
    try:
        return '_Cloth_CollisionRules' in str(getattr(obj, 'name', '') or '')
    except Exception:
        return False


def _pose_bone_for_cloth_segment(arm_obj, segment: int):
    pose_bones = getattr(getattr(arm_obj, 'pose', None), 'bones', {}) if arm_obj is not None else {}
    if pose_bones is None:
        return None
    for pose_bone in list(pose_bones or []):
        try:
            bone_segment = _cloth_bone_segment_from_name(getattr(pose_bone, 'name', ''), -1)
            if int(bone_segment) == int(segment):
                return pose_bone
        except Exception:
            pass
    return None


def _pose_bone_cloth_state(arm_obj, pose_bone) -> tuple[bool, bool]:
    bone_name = str(getattr(pose_bone, 'name', '') or '')
    data_bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name) if arm_obj is not None else None
    enabled = False
    pinned = False
    for candidate in (pose_bone, data_bone):
        if candidate is None:
            continue
        try:
            if bool(candidate.get('trlau_cloth_enabled', False)):
                enabled = True
            if 'trlau_cloth_pinned' in candidate and bool(candidate.get('trlau_cloth_pinned', False)):
                pinned = True
        except Exception:
            pass
    return bool(enabled), bool(pinned)


def _cloth_chain_root_pose_bone(arm_obj, pose_bone):
    if arm_obj is None or pose_bone is None:
        return pose_bone
    current = pose_bone
    selected_owner = pose_bone
    visited = set()
    while current is not None:
        name = str(getattr(current, 'name', '') or '')
        if name in visited:
            break
        visited.add(name)
        enabled, pinned = _pose_bone_cloth_state(arm_obj, current)
        if enabled:
            selected_owner = current
            if pinned:
                return current
        parent = getattr(current, 'parent', None)
        if parent is None:
            break
        current = parent
    return selected_owner


def _ensure_cloth_collision_rules_group(context, root, arm_obj=None, pose_bone=None, chain_root_segment: int | None = None, chain_root_bone_name: str = ''):
    if root is None:
        return None
    if chain_root_segment is None and pose_bone is not None:
        chain_root = _cloth_chain_root_pose_bone(arm_obj, pose_bone)
        chain_root_bone_name = str(getattr(chain_root, 'name', '') or '')
        chain_root_segment = _cloth_bone_segment_from_name(chain_root_bone_name, 0)
    elif chain_root_segment is not None and not chain_root_bone_name:
        chain_root_bone_name = f'bone_{int(chain_root_segment)}'

    group_name = _cloth_collision_rules_group_name(root, chain_root_segment, chain_root_bone_name)
    group = None
    for child in list(getattr(root, 'children', []) or []):
        if not _is_object_cloth_collision_rules_group(child):
            continue
        try:
            if chain_root_segment is not None and 'trlau_cloth_chain_root_segment' in child:
                if int(child.get('trlau_cloth_chain_root_segment')) == int(chain_root_segment):
                    group = child
                    break
            elif str(getattr(child, 'name', '') or '') == group_name:
                group = child
                break
        except Exception:
            pass
    if group is None:
        group = bpy.data.objects.get(group_name)
        if group is not None and getattr(group, 'parent', None) is not root:
            group = None
    collection = None
    try:
        collection = (list(getattr(root, 'users_collection', []) or []) or [getattr(context, 'collection', None)])[0]
    except Exception:
        collection = None
    if collection is None:
        try:
            collection = context.scene.collection
        except Exception:
            collection = bpy.context.scene.collection
    if group is None:
        group = bpy.data.objects.new(group_name, None)
        _link_object_to_collection(group, collection)
        group.parent = root
        group.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        group.empty_display_type = 'PLAIN_AXES'
        group.empty_display_size = 1.0
        group.show_name = False
        group.hide_select = True
    else:
        _link_object_to_collection(group, collection)
        try:
            group.parent = root
            group.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        except Exception:
            pass
    group.name = group_name
    if chain_root_segment is not None:
        group['trlau_cloth_chain_root_segment'] = int(chain_root_segment)
        group['trlau_cloth_chain_root_bone'] = str(chain_root_bone_name or f'bone_{int(chain_root_segment)}')
    return group



def _cloth_collision_name(root, index: int) -> str:
    return f'{_cloth_collision_name_base(root)}_Cloth_Collision_{int(index):03d}'


def _cloth_collision_index_from_name(root, name: str) -> int | None:
    prefix = f'{_cloth_collision_name_base(root)}_Cloth_Collision_'
    value = str(name or '')
    if '.' in value:
        value = value.split('.', 1)[0]
    if not value.startswith(prefix):
        return None
    suffix = value[len(prefix):]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    try:
        return int(''.join(digits))
    except Exception:
        return None


def _next_cloth_collision_index(root) -> int:
    max_index = -1
    for obj in list(bpy.data.objects):
        try:
            if getattr(obj, 'parent', None) is not root:
                continue
            if trlau_object_type(obj) not in {'ClothCollision', 'ClothPoint'}:
                continue
            index = _cloth_collision_index_from_name(root, getattr(obj, 'name', ''))
            if index is not None:
                max_index = max(max_index, index)
        except Exception:
            pass
    return max_index + 1


def _unique_cloth_collision_name(root) -> str:
    index = _next_cloth_collision_index(root)
    while bpy.data.objects.get(_cloth_collision_name(root, index)) is not None:
        index += 1
    return _cloth_collision_name(root, index)


def _cloth_bone_segment_from_name(name: str, default: int = 0) -> int:
    value = str(name or '')
    if value.startswith('bone_'):
        try:
            return int(value.split('_', 1)[1])
        except Exception:
            pass
    return int(default)


def _is_object_cloth_collision(obj) -> bool:
    try:
        return trlau_object_type(obj) in {'ClothCollision', 'ClothPoint'}
    except Exception:
        return False


def _is_object_cloth_collision_rule(obj) -> bool:
    try:
        if trlau_object_type(obj) == 'ClothCollisionRule':
            return True
    except Exception:
        pass
    try:
        return '_Cloth_CollisionRule_' in str(getattr(obj, 'name', '') or '')
    except Exception:
        return False


def _iter_object_descendants(obj):
    if obj is None:
        return
    for child in list(getattr(obj, 'children', []) or []):
        yield child
        yield from _iter_object_descendants(child)


def _iter_cloth_collision_rule_objects(root):
    if root is None:
        return
    for child in list(getattr(root, 'children', []) or []):
        if _is_object_cloth_collision_rule(child):
            yield child
        elif _is_object_cloth_collision_rules_group(child):
            for descendant in _iter_object_descendants(child):
                if _is_object_cloth_collision_rule(descendant):
                    yield descendant


def _move_existing_cloth_collision_rules_to_groups(context, root, arm_obj) -> None:
    if root is None:
        return
    for rule_obj in list(_iter_cloth_collision_rule_objects(root)):
        _order, _point_index, bone_segment, _collision_index = _cloth_collision_rule_parts_from_name(getattr(rule_obj, 'name', ''))
        if bone_segment is None:
            continue
        pose_bone = _pose_bone_for_cloth_segment(arm_obj, int(bone_segment))
        target_group = _ensure_cloth_collision_rules_group(
            context,
            root,
            arm_obj=arm_obj,
            pose_bone=pose_bone,
            chain_root_segment=int(bone_segment) if pose_bone is None else None,
        )
        if target_group is None or getattr(rule_obj, 'parent', None) is target_group:
            continue
        try:
            world_matrix = rule_obj.matrix_world.copy()
        except Exception:
            world_matrix = None
        try:
            rule_obj.parent = target_group
            rule_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            if world_matrix is not None:
                rule_obj.matrix_world = world_matrix
        except Exception:
            pass



def _cloth_collision_radius(obj, default: float = 1.0) -> float:
    try:
        values = tuple(float(v) for v in getattr(obj, 'scale', (default, default, default)))
        radius = max(abs(v) for v in values) if values else float(default)
        if radius > 0.0:
            return float(radius)
    except Exception:
        pass
    return float(default)


def _cloth_collision_rule_name(root, index: int, bone_segment: int, collision_index: int) -> str:
    return f'{_cloth_collision_name_base(root)}_Cloth_CollisionRule_{int(index):03d}_B{int(bone_segment):03d}_C{int(collision_index):03d}'


def _cloth_collision_rule_parts_from_name(name: str) -> tuple[int | None, int | None, int | None, int | None]:
    try:
        value = str(name or '').split('.', 1)[0]
        marker = '_Cloth_CollisionRule_'
        if marker not in value:
            return None, None, None, None
        suffix = value.split(marker, 1)[1]
        parts = suffix.split('_')
        order = int(parts[0]) if parts and parts[0].isdigit() else None
        point_index = None
        bone_segment = None
        collision_index = None
        for part in parts[1:]:
            if len(part) >= 2 and part[0].upper() == 'P' and part[1:].isdigit():
                point_index = int(part[1:])
            elif len(part) >= 2 and part[0].upper() == 'B' and part[1:].isdigit():
                bone_segment = int(part[1:])
            elif len(part) >= 2 and part[0].upper() == 'C' and part[1:].isdigit():
                collision_index = int(part[1:])
        return order, point_index, bone_segment, collision_index
    except Exception:
        return None, None, None, None


def _next_cloth_collision_rule_index(root) -> int:
    max_index = -1
    for child in _iter_cloth_collision_rule_objects(root):
        if not _is_object_cloth_collision_rule(child):
            continue
        order, _point_index, _bone_segment, _collision_index = _cloth_collision_rule_parts_from_name(getattr(child, 'name', ''))
        if order is not None:
            max_index = max(max_index, int(order))
    return max_index + 1


def _cloth_collision_rule_existing_keys(root) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for child in _iter_cloth_collision_rule_objects(root):
        if not _is_object_cloth_collision_rule(child):
            continue
        _order, _point_index, bone_segment, collision_index = _cloth_collision_rule_parts_from_name(getattr(child, 'name', ''))
        if bone_segment is not None and collision_index is not None:
            keys.add((int(bone_segment), int(collision_index)))
    return keys


def _pose_bone_is_cloth_unpinned(arm_obj, pose_bone) -> bool:
    bone_name = str(getattr(pose_bone, 'name', '') or '')
    if not bone_name:
        return False
    data_bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name)
    enabled = False
    pinned = False
    for candidate in (pose_bone, data_bone):
        if candidate is None:
            continue
        try:
            if bool(candidate.get('trlau_cloth_enabled', False)):
                enabled = True
            if 'trlau_cloth_pinned' in candidate and bool(candidate.get('trlau_cloth_pinned', False)):
                pinned = True
        except Exception:
            pass
    return bool(enabled) and not bool(pinned)


def _all_unpinned_cloth_pose_bones(arm_obj) -> list:
    pose_bones = getattr(getattr(arm_obj, 'pose', None), 'bones', {}) if arm_obj is not None else {}
    data_bones = getattr(getattr(arm_obj, 'data', None), 'bones', {}) if arm_obj is not None else {}
    result = []
    for default_index, data_bone in enumerate(data_bones or []):
        name = str(getattr(data_bone, 'name', '') or '')
        pose_bone = pose_bones.get(name) if pose_bones is not None else None
        if pose_bone is None:
            continue
        if _pose_bone_is_cloth_unpinned(arm_obj, pose_bone):
            result.append(pose_bone)
    result.sort(key=lambda pb: (_cloth_bone_segment_from_name(getattr(pb, 'name', ''), 0), str(getattr(pb, 'name', '') or '')))
    return result


def _collision_helpers_for_rule_generation(root, only_collision_obj=None) -> list:
    helpers = []
    candidate_children = [only_collision_obj] if only_collision_obj is not None else list(getattr(root, 'children', []) or [])
    for child in candidate_children:
        if child is None:
            continue
        if not _is_object_cloth_collision(child):
            continue
        index = _cloth_collision_index_from_name(root, getattr(child, 'name', ''))
        if index is None:
            index = len(helpers)
        helpers.append((int(index), child))
    helpers.sort(key=lambda item: (int(item[0]), str(getattr(item[1], 'name', '') or '')))
    return helpers


def _ensure_cloth_collision_rule_helpers(context, root, arm_obj, pose_bones=None, only_collision_obj=None) -> int:
    if root is None or arm_obj is None:
        return 0
    if pose_bones is None:
        pose_bones = _all_unpinned_cloth_pose_bones(arm_obj)
    else:
        pose_bones = [pose_bone for pose_bone in pose_bones if _pose_bone_is_cloth_unpinned(arm_obj, pose_bone)]
    if not pose_bones:
        return 0
    collision_helpers = _collision_helpers_for_rule_generation(root, only_collision_obj=only_collision_obj)
    if not collision_helpers:
        return 0

    collection = (getattr(root, 'users_collection', None) or [getattr(context, 'collection', None)])[0]
    if collection is None:
        collection = context.scene.collection
    _move_existing_cloth_collision_rules_to_groups(context, root, arm_obj)

    existing_keys = _cloth_collision_rule_existing_keys(root)
    created_count = 0
    next_index = _next_cloth_collision_rule_index(root)
    data_bones = getattr(getattr(arm_obj, 'data', None), 'bones', {})
    for pose_bone in pose_bones:
        bone_name = str(getattr(pose_bone, 'name', '') or '')
        if not bone_name:
            continue
        bone_segment = _cloth_bone_segment_from_name(bone_name, len(existing_keys))
        data_bone = data_bones.get(bone_name) if data_bones is not None else None
        for collision_index, collision_obj in collision_helpers:
            key = (int(bone_segment), int(collision_index))
            if key in existing_keys:
                continue
            rule_name = _cloth_collision_rule_name(root, next_index, bone_segment, collision_index)
            while bpy.data.objects.get(rule_name) is not None:
                next_index += 1
                rule_name = _cloth_collision_rule_name(root, next_index, bone_segment, collision_index)
            rule_parent = _ensure_cloth_collision_rules_group(context, root, arm_obj=arm_obj, pose_bone=pose_bone) or root
            rule_obj = bpy.data.objects.new(rule_name, None)
            collection.objects.link(rule_obj)
            rule_obj.parent = rule_parent
            rule_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            rule_obj.empty_display_type = 'SPHERE'
            rule_obj.empty_display_size = 0.1
            rule_obj.show_name = False
            rule_obj.show_in_front = True
            try:
                if data_bone is not None:
                    rule_obj.matrix_local = data_bone.matrix_local.copy()
                else:
                    rule_obj.matrix_world.translation = getattr(collision_obj, 'matrix_world', mathutils.Matrix.Identity(4)).translation.copy()
            except Exception:
                pass
            if data_bone is not None:
                try:
                    child_of = rule_obj.constraints.new(type='CHILD_OF')
                    child_of.name = 'TRLAU Cloth Collision Rule Target Follow'
                    child_of.target = arm_obj
                    child_of.subtarget = bone_name
                    child_of.inverse_matrix = (arm_obj.matrix_world @ data_bone.matrix_local).inverted() @ getattr(rule_parent, 'matrix_world', root.matrix_world)
                except Exception:
                    pass
            radius = _cloth_collision_radius(collision_obj, 1.0)
            rule_obj.scale = (radius, radius, radius)
            _lock_cloth_collision_rotation(rule_obj)
            existing_keys.add(key)
            created_count += 1
            next_index += 1
    return int(created_count)


def _cloth_capsule_name(root, index: int) -> str:
    return f'{_cloth_collision_name_base(root)}_Cloth_Capsule_{int(index):03d}'


def _cloth_capsule_endpoint_name(root, capsule_index: int, endpoint: str) -> str:
    endpoint = 'B' if str(endpoint).upper() == 'B' else 'A'
    return f'{_cloth_capsule_name(root, capsule_index)}_{endpoint}'


def _cloth_capsule_index_from_name(root, name: str) -> int | None:
    prefix = f'{_cloth_collision_name_base(root)}_Cloth_Capsule_'
    value = str(name or '')
    if '.' in value:
        value = value.split('.', 1)[0]
    if not value.startswith(prefix):
        return None
    suffix = value[len(prefix):]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    try:
        return int(''.join(digits))
    except Exception:
        return None


def _next_cloth_capsule_index(root) -> int:
    max_index = -1
    for obj in list(bpy.data.objects):
        try:
            if getattr(obj, 'parent', None) is not root:
                continue
            if trlau_object_type(obj) != 'ClothCapsuleEndpoint':
                continue
            index = _cloth_capsule_index_from_name(root, getattr(obj, 'name', ''))
            if index is not None:
                max_index = max(max_index, int(index))
        except Exception:
            pass
    return max_index + 1


def _unique_cloth_capsule_index(root) -> int:
    index = _next_cloth_capsule_index(root)
    while (
        bpy.data.objects.get(_cloth_capsule_endpoint_name(root, index, 'A')) is not None
        or bpy.data.objects.get(_cloth_capsule_endpoint_name(root, index, 'B')) is not None
    ):
        index += 1
    return index


def _hide_cloth_collision_viewport_names() -> None:
    for obj in list(getattr(bpy.data, 'objects', []) or []):
        try:
            if trlau_object_type(obj) in {'ClothCollision', 'ClothPoint', 'ClothCapsuleEndpoint'}:
                obj.show_name = False
        except Exception:
            pass


def _apply_cloth_bone_color(arm_obj, bone_name: str, pinned: bool | None) -> None:
    palette = 'DEFAULT' if pinned is None else ('THEME01' if pinned else 'THEME08')
    for candidate in (
        getattr(getattr(arm_obj, 'pose', None), 'bones', {}).get(bone_name) if arm_obj is not None else None,
        getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name) if arm_obj is not None else None,
    ):
        if candidate is None or not hasattr(candidate, 'color'):
            continue
        try:
            candidate.color.palette = palette
        except Exception:
            pass


def _lock_cloth_collision_rotation(obj) -> None:
    lock_object_rotation(obj)



def _iter_object_ancestors(obj):
    current = getattr(obj, 'parent', None)
    seen = set()
    while current is not None:
        try:
            key = int(current.as_pointer())
        except Exception:
            key = id(current)
        if key in seen:
            break
        seen.add(key)
        yield current
        current = getattr(current, 'parent', None)


def _find_ancestor_with_type(obj, trlau_type: str):
    current = obj
    while current is not None:
        try:
            if trlau_object_type(current) == trlau_type:
                return current
        except Exception:
            pass
        current = getattr(current, 'parent', None)
    return None


def _armature_from_related_object(obj):
    current = obj
    while current is not None:
        if getattr(current, 'type', None) == 'ARMATURE':
            return current
        current = getattr(current, 'parent', None)
    return None


def _is_hmarker_object(obj) -> bool:
    try:
        if trlau_object_type(obj) == 'HMarker':
            return True
    except Exception:
        pass
    try:
        return '_HMarker_' in str(getattr(obj, 'name', '') or '')
    except Exception:
        return False


def _hmarker_stored_index(obj, default: int = 0) -> int:
    return abs(int(hmarker_index_from_name(obj, default)))


def _selected_hmarkers(context) -> list:
    selected = []
    seen = set()
    active = getattr(context, 'object', None)
    for obj in [active] + list(getattr(context, 'selected_objects', []) or []):
        if obj is None or not _is_hmarker_object(obj):
            continue
        try:
            key = int(obj.as_pointer())
        except Exception:
            key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        selected.append(obj)
    return selected


def _collection_for_cloth_root(context, root):
    collections = list(getattr(root, 'users_collection', []) or []) if root is not None else []
    if collections:
        return collections[0]
    collection = getattr(context, 'collection', None) if context is not None else None
    if collection is not None:
        return collection
    scene = getattr(context, 'scene', None) if context is not None else None
    return getattr(scene, 'collection', None) or bpy.context.scene.collection


def _next_cloth_plane_rule_index(root) -> int:
    max_index = -1
    for child in list(getattr(root, 'children', []) or []):
        try:
            if trlau_object_type(child) != 'ClothPlaneRule':
                continue
            value = child.get('trlau_cloth_plane_rule_index', None)
            if value is None:
                name = str(getattr(child, 'name', '') or '')
                marker = '_PlaneRule_'
                if marker in name:
                    suffix = name.split(marker, 1)[1].split('_', 1)[0]
                    value = int(suffix) if suffix.isdigit() else None
            if value is not None:
                max_index = max(max_index, int(value))
        except Exception:
            pass
    return max_index + 1


def _find_cloth_plane_rule_parent(obj):
    current = obj
    while current is not None:
        try:
            if trlau_object_type(current) == 'ClothPlaneRule':
                return current
        except Exception:
            pass
        current = getattr(current, 'parent', None)
    return None


def _iter_descendants(obj):
    stack = list(getattr(obj, 'children', []) or [])
    while stack:
        child = stack.pop(0)
        yield child
        stack[0:0] = list(getattr(child, 'children', []) or [])


def _find_rule_plane_children(rule_obj) -> list:
    children = []
    for child in _iter_descendants(rule_obj):
        try:
            if trlau_object_type(child) == 'ClothPlaneRulePlane':
                children.append(child)
        except Exception:
            pass
    children.sort(key=lambda obj: str(getattr(obj, 'name', '') or ''))
    return children


def _find_rule_selector(rule_obj):
    for child in _iter_descendants(rule_obj):
        try:
            if trlau_object_type(child) == 'ClothPlaneRuleSelector':
                return child
        except Exception:
            pass
        try:
            if str(getattr(child, 'name', '') or '').endswith('_Selector'):
                return child
        except Exception:
            pass
    return None


def _marker_from_plane_child(plane_obj):
    for constraint in list(getattr(plane_obj, 'constraints', []) or []):
        try:
            target = getattr(constraint, 'target', None)
            if target is not None and _is_hmarker_object(target):
                return target
        except Exception:
            pass
    marker_index = None
    try:
        marker_index = int(plane_obj.get('trlau_cloth_plane_marker_index'))
    except Exception:
        marker_index = None
    if marker_index is None:
        return None
    for obj in list(getattr(bpy.data, 'objects', []) or []):
        if _is_hmarker_object(obj) and abs(_hmarker_stored_index(obj, 0)) == abs(int(marker_index)):
            return obj
    return None


def _markers_for_rule(rule_obj) -> list:
    markers = []
    seen = set()
    for plane_obj in _find_rule_plane_children(rule_obj):
        marker = _marker_from_plane_child(plane_obj)
        if marker is None:
            continue
        try:
            key = int(marker.as_pointer())
        except Exception:
            key = id(marker)
        if key in seen:
            continue
        seen.add(key)
        markers.append(marker)
    if len(markers) >= 2:
        return markers[:2]

    name = str(getattr(rule_obj, 'name', '') or '')
    wanted = []
    try:
        import re
        found = re.findall(r'_M(\d+)', name)
        wanted = [int(v) for v in found[:2]]
    except Exception:
        wanted = []
    for marker_index in wanted:
        for obj in list(getattr(bpy.data, 'objects', []) or []):
            if not _is_hmarker_object(obj):
                continue
            if abs(_hmarker_stored_index(obj, 0)) != abs(int(marker_index)):
                continue
            try:
                key = int(obj.as_pointer())
            except Exception:
                key = id(obj)
            if key in seen:
                continue
            seen.add(key)
            markers.append(obj)
            break
    return markers[:2]


def _make_or_get_cloth_plane_material():
    material = bpy.data.materials.get('TRLAU Cloth PlaneRule')
    if material is None:
        material = bpy.data.materials.new('TRLAU Cloth PlaneRule')
        try:
            material.diffuse_color = (0.15, 0.55, 1.0, 0.35)
            material.use_nodes = False
            material.blend_method = 'BLEND'
            material.show_transparent_back = True
        except Exception:
            pass
    return material


def _make_cloth_plane_face_mesh(mesh_name: str, width: float = 24.0, height: float = 24.0):
    width = max(0.1, float(width))
    height = max(0.1, float(height))
    hx = width * 0.5
    hy = height * 0.5
    vertices = [(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    faces = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(vertices, edges, faces)
    try:
        mesh.update()
    except Exception:
        pass
    return mesh


def _bounds_from_world_points(points, *, minimum_extent: float = 8.0, margin: float | None = None):
    positions = [mathutils.Vector(tuple(point)) for point in list(points or [])]
    if not positions:
        return None
    xs = [float(point.x) for point in positions]
    ys = [float(point.y) for point in positions]
    zs = [float(point.z) for point in positions]
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0e-5)
    pad = max(2.0, min(extent * 0.08, 10.0)) if margin is None else max(0.0, float(margin))
    min_v = mathutils.Vector((min(xs) - pad, min(ys) - pad, min(zs) - pad))
    max_v = mathutils.Vector((max(xs) + pad, max(ys) + pad, max(zs) + pad))
    for axis in range(3):
        if float(max_v[axis]) - float(min_v[axis]) < float(minimum_extent):
            center = (float(max_v[axis]) + float(min_v[axis])) * 0.5
            min_v[axis] = center - float(minimum_extent) * 0.5
            max_v[axis] = center + float(minimum_extent) * 0.5
    return min_v, max_v


def _make_selector_mesh_from_bounds(mesh_name: str, bounds, parent_world_inverse):
    if bounds is None:
        return None
    min_v, max_v = bounds
    world_points = [
        mathutils.Vector((min_v.x, min_v.y, min_v.z)),
        mathutils.Vector((max_v.x, min_v.y, min_v.z)),
        mathutils.Vector((max_v.x, max_v.y, min_v.z)),
        mathutils.Vector((min_v.x, max_v.y, min_v.z)),
        mathutils.Vector((min_v.x, min_v.y, max_v.z)),
        mathutils.Vector((max_v.x, min_v.y, max_v.z)),
        mathutils.Vector((max_v.x, max_v.y, max_v.z)),
        mathutils.Vector((min_v.x, max_v.y, max_v.z)),
    ]
    vertices = []
    for point in world_points:
        try:
            vertices.append(tuple(parent_world_inverse @ point))
        except Exception:
            vertices.append(tuple(point))
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    mesh = bpy.data.meshes.new(mesh_name)
    mesh.from_pydata(vertices, edges, [])
    try:
        mesh.update()
    except Exception:
        pass
    return mesh


def _set_selector_mesh_from_bounds(selector_obj, bounds, parent_obj):
    if selector_obj is None or bounds is None:
        return False
    try:
        parent_inv = parent_obj.matrix_world.inverted_safe()
    except Exception:
        parent_inv = mathutils.Matrix.Identity(4)
    mesh = _make_selector_mesh_from_bounds(f'{selector_obj.name}_Mesh', bounds, parent_inv)
    if mesh is None:
        return False
    old_mesh = getattr(selector_obj, 'data', None)
    selector_obj.data = mesh
    try:
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    except Exception:
        pass
    return True


def _pose_bone_world_position(arm_obj, pose_bone):
    try:
        return arm_obj.matrix_world @ pose_bone.matrix.translation
    except Exception:
        pass
    try:
        bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(pose_bone.name)
        if bone is not None:
            return arm_obj.matrix_world @ bone.matrix_local.translation
    except Exception:
        pass
    return None


def _data_bone_is_cloth_point(bone) -> bool:
    try:
        if bool(bone.get('trlau_cloth_enabled', False)):
            return True
    except Exception:
        pass
    return False


def _data_bone_is_movable_cloth_point(bone) -> bool:
    if not _data_bone_is_cloth_point(bone):
        return False
    try:
        if bool(bone.get('trlau_cloth_pinned', False)):
            return False
    except Exception:
        pass
    return True


def _selected_cloth_point_world_positions(context, arm_obj=None) -> list[mathutils.Vector]:
    positions = []
    seen = set()

    def add_position(point):
        if point is None:
            return
        key = tuple(round(float(v), 5) for v in point)
        if key in seen:
            return
        seen.add(key)
        positions.append(mathutils.Vector(point))

    if arm_obj is None:
        arm_obj = getattr(context, 'object', None) if getattr(getattr(context, 'object', None), 'type', None) == 'ARMATURE' else None

    for pose_bone in list(getattr(context, 'selected_pose_bones', []) or []):
        try:
            data_bone = getattr(getattr(pose_bone, 'id_data', None), 'data', None).bones.get(pose_bone.name)
        except Exception:
            data_bone = None
        if data_bone is not None and _data_bone_is_movable_cloth_point(data_bone):
            add_position(_pose_bone_world_position(pose_bone.id_data, pose_bone))

    for obj in list(getattr(context, 'selected_objects', []) or []):
        try:
            obj_type = trlau_object_type(obj)
        except Exception:
            obj_type = ''
        if obj_type in {'ClothCollision', 'ClothPoint', 'ClothCapsuleEndpoint'}:
            add_position(obj.matrix_world.translation)
        if getattr(obj, 'type', None) == 'ARMATURE':
            arm_obj = obj
            active = getattr(getattr(obj.data, 'bones', None), 'active', None)
            if active is not None and _data_bone_is_movable_cloth_point(active):
                pose = getattr(getattr(obj, 'pose', None), 'bones', {}).get(active.name)
                if pose is not None:
                    add_position(_pose_bone_world_position(obj, pose))

    return positions


def _all_movable_cloth_point_world_positions(root, arm_obj=None) -> list[mathutils.Vector]:
    positions = []
    if arm_obj is None:
        arm_obj = getattr(root, 'parent', None) if getattr(getattr(root, 'parent', None), 'type', None) == 'ARMATURE' else None
    if arm_obj is not None:
        pose_bones = getattr(getattr(arm_obj, 'pose', None), 'bones', {}) or {}
        for data_bone in list(getattr(getattr(arm_obj, 'data', None), 'bones', []) or []):
            if not _data_bone_is_movable_cloth_point(data_bone):
                continue
            pose_bone = pose_bones.get(data_bone.name)
            if pose_bone is not None:
                point = _pose_bone_world_position(arm_obj, pose_bone)
            else:
                try:
                    point = arm_obj.matrix_world @ data_bone.matrix_local.translation
                except Exception:
                    point = None
            if point is not None:
                positions.append(point)
    for child in _iter_descendants(root):
        try:
            obj_type = trlau_object_type(child)
        except Exception:
            obj_type = ''
        if obj_type not in {'ClothCollision', 'ClothPoint', 'ClothCapsuleEndpoint'}:
            continue
        try:
            positions.append(child.matrix_world.translation.copy())
        except Exception:
            pass
    return positions


def _default_selector_bounds_for_markers(marker_a, marker_b):
    try:
        point_a = marker_a.matrix_world.translation.copy()
    except Exception:
        point_a = mathutils.Vector((0.0, 0.0, 0.0))
    try:
        point_b = marker_b.matrix_world.translation.copy()
    except Exception:
        point_b = point_a.copy()
    center = (point_a + point_b) * 0.5
    distance = max(0.0, float((point_b - point_a).length))
    extent = max(16.0, min(96.0, distance * 0.75 if distance > 1.0e-5 else 32.0))
    half = extent * 0.5
    return mathutils.Vector((center.x - half, center.y - half, center.z - half)), mathutils.Vector((center.x + half, center.y + half, center.z + half))


def _create_cloth_plane_rule_objects(context, root, marker_a, marker_b, *, selector_positions=None, display_width: float = 24.0, display_height: float = 24.0):
    collection = _collection_for_cloth_root(context, root)
    material = _make_or_get_cloth_plane_material()
    marker_a_index = _hmarker_stored_index(marker_a, 0)
    marker_b_index = _hmarker_stored_index(marker_b, 0)
    rule_index = _next_cloth_plane_rule_index(root)
    base = _cloth_collision_name_base(root)
    rule_name = f'{base}_Cloth_PlaneRule_{int(rule_index):03d}_M{abs(int(marker_a_index)):03d}_M{abs(int(marker_b_index)):03d}'

    rule_obj = bpy.data.objects.new(rule_name, None)
    _link_object_to_collection(rule_obj, collection)
    rule_obj.parent = root
    rule_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    try:
        rule_obj.matrix_local = mathutils.Matrix.Identity(4)
    except Exception:
        pass
    rule_obj.empty_display_type = 'PLAIN_AXES'
    rule_obj.empty_display_size = 4.0
    rule_obj.show_name = False
    rule_obj.show_in_front = True
    rule_obj['trlau_cloth_enabled'] = True
    rule_obj['trlau_export_ignore'] = True
    rule_obj['trlau_cloth_plane_rule_index'] = int(rule_index)
    rule_obj['trlau_cloth_plane_display_width'] = float(display_width)
    rule_obj['trlau_cloth_plane_display_height'] = float(display_height)

    for label, marker, marker_index in (('A', marker_a, marker_a_index), ('B', marker_b, marker_b_index)):
        mesh = _make_cloth_plane_face_mesh(f'{rule_name}_M{abs(int(marker_index)):03d}_{label}_Mesh', display_width, display_height)
        plane_obj = bpy.data.objects.new(f'{rule_name}_M{abs(int(marker_index)):03d}_{label}', mesh)
        _link_object_to_collection(plane_obj, collection)
        plane_obj.parent = rule_obj
        try:
            plane_obj.matrix_world = marker.matrix_world.copy()
        except Exception:
            plane_obj.matrix_local = mathutils.Matrix.Identity(4)
        try:
            plane_obj.data.materials.append(material)
        except Exception:
            pass
        plane_obj.show_name = False
        plane_obj.show_in_front = True
        try:
            plane_obj.show_wire = True
        except Exception:
            pass
        plane_obj['trlau_cloth_enabled'] = True
        plane_obj['trlau_export_ignore'] = True
        plane_obj['trlau_cloth_plane_rule_index'] = int(rule_index)
        plane_obj['trlau_cloth_plane_marker_index'] = int(abs(int(marker_index)))
        try:
            follow = plane_obj.constraints.new(type='COPY_TRANSFORMS')
            follow.name = 'TRLAU Cloth PlaneRule HMarker Follow'
            follow.target = marker
            follow.target_space = 'WORLD'
            follow.owner_space = 'WORLD'
        except Exception:
            pass

    if selector_positions:
        bounds = _bounds_from_world_points(selector_positions, minimum_extent=8.0)
    else:
        bounds = _default_selector_bounds_for_markers(marker_a, marker_b)
    try:
        parent_inv = rule_obj.matrix_world.inverted_safe()
    except Exception:
        parent_inv = mathutils.Matrix.Identity(4)
    selector_mesh = _make_selector_mesh_from_bounds(f'{rule_name}_Selector_Mesh', bounds, parent_inv)
    if selector_mesh is not None:
        selector_obj = bpy.data.objects.new(f'{rule_name}_Selector', selector_mesh)
        _link_object_to_collection(selector_obj, collection)
        selector_obj.parent = rule_obj
        selector_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        try:
            selector_obj.matrix_local = mathutils.Matrix.Identity(4)
        except Exception:
            pass
        try:
            selector_obj.data.materials.append(material)
        except Exception:
            pass
        selector_obj.show_name = False
        selector_obj.show_in_front = True
        try:
            selector_obj.show_wire = True
            selector_obj.display_type = 'WIRE'
        except Exception:
            pass
        selector_obj['trlau_cloth_enabled'] = True
        selector_obj['trlau_export_ignore'] = True
        selector_obj['trlau_cloth_plane_rule_index'] = int(rule_index)

    return rule_obj


def _rule_objects_from_context(context) -> list:
    rules = []
    seen = set()
    for obj in [getattr(context, 'object', None)] + list(getattr(context, 'selected_objects', []) or []):
        rule = _find_cloth_plane_rule_parent(obj)
        if rule is None:
            continue
        try:
            key = int(rule.as_pointer())
        except Exception:
            key = id(rule)
        if key in seen:
            continue
        seen.add(key)
        rules.append(rule)
    if rules:
        return rules
    obj = getattr(context, 'object', None)
    root = _find_ancestor_with_type(obj, 'Cloth') if obj is not None else None
    if root is None and obj is not None:
        try:
            if trlau_object_type(obj) == 'Cloth':
                root = obj
        except Exception:
            pass
    if root is not None:
        for child in list(getattr(root, 'children', []) or []):
            try:
                if trlau_object_type(child) == 'ClothPlaneRule':
                    rules.append(child)
            except Exception:
                pass
    return rules


def _rebuild_rule_plane_meshes(rule_obj, *, display_width: float | None = None, display_height: float | None = None) -> int:
    markers = _markers_for_rule(rule_obj)
    if len(markers) < 2:
        return 0
    collection = None
    try:
        collection = list(rule_obj.users_collection or [])[0]
    except Exception:
        collection = None
    if collection is None:
        collection = bpy.context.collection or bpy.context.scene.collection
    material = _make_or_get_cloth_plane_material()
    try:
        width = float(display_width if display_width is not None else rule_obj.get('trlau_cloth_plane_display_width', 24.0))
        height = float(display_height if display_height is not None else rule_obj.get('trlau_cloth_plane_display_height', 24.0))
    except Exception:
        width, height = 24.0, 24.0
    rule_obj['trlau_cloth_plane_display_width'] = float(width)
    rule_obj['trlau_cloth_plane_display_height'] = float(height)

    existing = _find_rule_plane_children(rule_obj)
    rebuilt = 0
    for plane_obj in existing:
        old_mesh = getattr(plane_obj, 'data', None)
        marker_index = 0
        try:
            marker_index = int(plane_obj.get('trlau_cloth_plane_marker_index', 0) or 0)
        except Exception:
            marker_index = 0
        plane_obj.data = _make_cloth_plane_face_mesh(f'{plane_obj.name}_Mesh', width, height)
        try:
            if old_mesh is not None and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
        except Exception:
            pass
        try:
            if material is not None and len(plane_obj.data.materials) == 0:
                plane_obj.data.materials.append(material)
        except Exception:
            pass
        plane_obj['trlau_cloth_plane_marker_index'] = int(abs(marker_index))
        rebuilt += 1

    if len(existing) >= 2:
        return rebuilt

    rule_index = int(rule_obj.get('trlau_cloth_plane_rule_index', _next_cloth_plane_rule_index(getattr(rule_obj, 'parent', None))))
    for label, marker in (('A', markers[0]), ('B', markers[1])):
        marker_index = _hmarker_stored_index(marker, 0)
        name = f'{rule_obj.name}_M{abs(int(marker_index)):03d}_{label}'
        if bpy.data.objects.get(name) is not None:
            continue
        plane_obj = bpy.data.objects.new(name, _make_cloth_plane_face_mesh(f'{name}_Mesh', width, height))
        _link_object_to_collection(plane_obj, collection)
        plane_obj.parent = rule_obj
        try:
            plane_obj.matrix_world = marker.matrix_world.copy()
        except Exception:
            pass
        try:
            plane_obj.data.materials.append(material)
        except Exception:
            pass
        plane_obj.show_name = False
        plane_obj.show_in_front = True
        plane_obj['trlau_cloth_enabled'] = True
        plane_obj['trlau_export_ignore'] = True
        plane_obj['trlau_cloth_plane_rule_index'] = int(rule_index)
        plane_obj['trlau_cloth_plane_marker_index'] = int(abs(int(marker_index)))
        try:
            follow = plane_obj.constraints.new(type='COPY_TRANSFORMS')
            follow.name = 'TRLAU Cloth PlaneRule HMarker Follow'
            follow.target = marker
            follow.target_space = 'WORLD'
            follow.owner_space = 'WORLD'
        except Exception:
            pass
        rebuilt += 1
    return rebuilt


class TRLAU_OT_set_cloth_bone_state(bpy.types.Operator):
    bl_idname = 'trlau.set_cloth_bone_state'
    bl_label = 'Set Cloth Bone State'
    bl_description = 'Pin, unpin, or clear the selected bones for cloth export'
    bl_options = {'REGISTER', 'UNDO'}

    state: EnumProperty(
        name='State',
        items=(
            ('PIN', 'Pin', 'Export selected bones as pinned cloth bones'),
            ('UNPIN', 'Unpin', 'Export selected bones as movable cloth bones'),
            ('CLEAR', 'Clear', 'Remove selected bones from cloth export'),
        ),
        default='UNPIN',
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return True
        if obj is not None and trlau_object_type(obj) == 'Cloth':
            return getattr(getattr(obj, 'parent', None), 'type', None) == 'ARMATURE'
        return getattr(context, 'active_pose_bone', None) is not None

    @staticmethod
    def _armature_from_context(context):
        obj = getattr(context, 'object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return obj
        if obj is not None and trlau_object_type(obj) == 'Cloth':
            parent = getattr(obj, 'parent', None)
            if getattr(parent, 'type', None) == 'ARMATURE':
                return parent
        return None

    @staticmethod
    def _selected_pose_bones(context, arm_obj):
        bones = list(getattr(context, 'selected_pose_bones', []) or [])
        active = getattr(context, 'active_pose_bone', None)
        if active is not None and active not in bones:
            bones.append(active)
        if bones:
            return bones
        selected = []
        for bone in list(getattr(getattr(arm_obj, 'data', None), 'bones', []) or []):
            try:
                if bool(getattr(bone, 'select', False)):
                    pose_bone = getattr(getattr(arm_obj, 'pose', None), 'bones', {}).get(bone.name)
                    selected.append(pose_bone or bone)
            except Exception:
                pass
        return selected

    def execute(self, context):
        arm_obj = self._armature_from_context(context)
        if arm_obj is None:
            self.report({'WARNING'}, 'Select an armature or its Cloth empty')
            return {'CANCELLED'}
        bones = self._selected_pose_bones(context, arm_obj)
        if not bones:
            self.report({'WARNING'}, 'Select one or more bones')
            return {'CANCELLED'}

        if self.state != 'CLEAR':
            _ensure_cloth_root(context, arm_obj)

        pinned = self.state == 'PIN'
        for pose_bone in bones:
            bone_name = str(getattr(pose_bone, 'name', '') or '')
            data_bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name)
            for target in (data_bone, pose_bone):
                if target is None:
                    continue
                if self.state == 'CLEAR':
                    for key in (
                        'trlau_type', 'trlau_cloth_enabled', 'trlau_cloth_pinned', 'trlau_cloth_point_flags',
                        'trlau_cloth_point_key', 'trlau_cloth_point_index', 'trlau_cloth_point_segment',
                        'trlau_cloth_point_jointOrder', 'trlau_cloth_point_upTo', 'trlau_cloth_chain_id',
                        'trlau_cloth_chain_order', 'trlau_cloth_map_axis',
                    ):
                        try:
                            if key in target:
                                del target[key]
                        except Exception:
                            pass
                    continue
                target['trlau_cloth_enabled'] = True
                target['trlau_cloth_pinned'] = bool(pinned)
            _apply_cloth_bone_color(arm_obj, bone_name, None if self.state == 'CLEAR' else pinned)
        return {'FINISHED'}


class TRLAU_OT_add_cloth_point(bpy.types.Operator):
    bl_idname = 'trlau.add_cloth_point'
    bl_label = 'Add Cloth Collision Sphere'
    bl_description = 'Add a collision sphere to the selected bone'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if getattr(context, 'mode', '') != 'POSE':
            return False
        pose_bone = getattr(context, 'active_pose_bone', None)
        arm_obj = getattr(pose_bone, 'id_data', None) if pose_bone is not None else None
        return getattr(arm_obj, 'type', None) == 'ARMATURE'

    @staticmethod
    def _active_armature_and_root(context):
        obj = getattr(context, 'object', None)
        if obj is not None and trlau_object_type(obj) == 'Cloth':
            arm_obj = getattr(obj, 'parent', None) if getattr(getattr(obj, 'parent', None), 'type', None) == 'ARMATURE' else None
            return arm_obj, obj
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            root = _ensure_cloth_root(context, obj)
            return obj, root
        pose_bone = getattr(context, 'active_pose_bone', None)
        arm_obj = getattr(pose_bone, 'id_data', None) if pose_bone is not None else None
        if arm_obj is not None:
            root = _ensure_cloth_root(context, arm_obj)
            return arm_obj, root
        return None, None

    def execute(self, context):
        arm_obj, root = self._active_armature_and_root(context)
        if root is None:
            self.report({'WARNING'}, 'Select a Cloth empty or armature')
            return {'CANCELLED'}
        active_pose_bone = getattr(context, 'active_pose_bone', None)
        segment = 0
        bone_name = ''
        if active_pose_bone is not None:
            bone_name = str(active_pose_bone.name)
        elif arm_obj is not None:
            try:
                bone_name = str(getattr(getattr(arm_obj, 'data', None), 'bones', [])[0].name)
            except Exception:
                bone_name = ''
        if bone_name.startswith('bone_'):
            try:
                segment = int(bone_name.split('_', 1)[1])
            except Exception:
                segment = 0

        radius = 24.0
        location = getattr(root, 'matrix_world', mathutils.Matrix.Identity(4)).translation
        if arm_obj is not None and bone_name:
            bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name)
            if bone is not None:
                try:
                    location = arm_obj.matrix_world @ bone.head_local
                except Exception:
                    pass
        point_obj = bpy.data.objects.new(_unique_cloth_collision_name(root), None)
        target_collection = getattr(root, 'users_collection', None)
        collection = target_collection[0] if target_collection else getattr(context, 'collection', None)
        if collection is not None:
            collection.objects.link(point_obj)
        else:
            context.scene.collection.objects.link(point_obj)
        point_obj.location = location
        point_obj.parent = root
        point_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        point_obj.empty_display_type = 'SPHERE'
        point_obj.empty_display_size = 1.0
        point_obj.show_name = False
        point_obj.show_in_front = True
        point_obj.scale = (radius, radius, radius)
        _lock_cloth_collision_rotation(point_obj)
        if arm_obj is not None and bone_name:
            try:
                child_of = point_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLAU Cloth Collision Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
            except Exception:
                pass
        _ensure_cloth_collision_rule_helpers(context, root, arm_obj, only_collision_obj=point_obj)
        return {'FINISHED'}


class TRLAU_OT_add_cloth_capsule(bpy.types.Operator):
    bl_idname = 'trlau.add_cloth_capsule'
    bl_label = 'Add Cloth Collision Capsule'
    bl_description = 'Add a collision capsule to the selected bone'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return TRLAU_OT_add_cloth_point.poll(context)

    @staticmethod
    def _axis_vector(axis: str):
        return {
            'X': mathutils.Vector((1.0, 0.0, 0.0)),
            'NEG_X': mathutils.Vector((-1.0, 0.0, 0.0)),
            'Y': mathutils.Vector((0.0, 1.0, 0.0)),
            'NEG_Y': mathutils.Vector((0.0, -1.0, 0.0)),
            'Z': mathutils.Vector((0.0, 0.0, 1.0)),
            'NEG_Z': mathutils.Vector((0.0, 0.0, -1.0)),
        }.get(axis, mathutils.Vector((0.0, 1.0, 0.0)))

    @staticmethod
    def _add_bone_follow_constraint(endpoint_obj, arm_obj, bone_name: str) -> None:
        if arm_obj is None or not bone_name:
            return
        try:
            child_of = endpoint_obj.constraints.new(type='CHILD_OF')
            child_of.name = 'TRLAU Cloth Capsule Endpoint Bone Follow'
            child_of.target = arm_obj
            child_of.subtarget = bone_name
            try:
                bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name)
                parent_obj = getattr(endpoint_obj, 'parent', None)
                if bone is not None and parent_obj is not None:
                    child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ parent_obj.matrix_world
            except Exception:
                pass
        except Exception:
            pass

    def execute(self, context):
        arm_obj, root = TRLAU_OT_add_cloth_point._active_armature_and_root(context)
        if root is None:
            self.report({'WARNING'}, 'Select a Cloth empty or armature')
            return {'CANCELLED'}

        active_pose_bone = getattr(context, 'active_pose_bone', None)
        bone_name = str(getattr(active_pose_bone, 'name', '') or '') if active_pose_bone is not None else ''
        if not bone_name and arm_obj is not None:
            try:
                bone_name = str(getattr(getattr(arm_obj, 'data', None), 'bones', [])[0].name)
            except Exception:
                bone_name = ''

        radius = 24.0
        length = 32.0
        axis_vector = mathutils.Vector((0.0, 1.0, 0.0))
        if axis_vector.length <= 0.000001:
            axis_vector = mathutils.Vector((0.0, 1.0, 0.0))
        axis_vector.normalize()

        start_world = getattr(root, 'matrix_world', mathutils.Matrix.Identity(4)).translation.copy()
        end_world = start_world + axis_vector * length
        if arm_obj is not None and bone_name:
            bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name)
            if bone is not None:
                try:
                    start_world = arm_obj.matrix_world @ bone.head_local
                    bone_dir = (bone.tail_local - bone.head_local)
                    if bone_dir.length > 0.000001:
                        end_world = arm_obj.matrix_world @ (bone.head_local + bone_dir.normalized() * length)
                    else:
                        end_world = start_world + (arm_obj.matrix_world.to_3x3() @ axis_vector) * length
                except Exception:
                    end_world = start_world + axis_vector * length

        capsule_index = _unique_cloth_capsule_index(root)
        collection = (getattr(root, 'users_collection', None) or [getattr(context, 'collection', None)])[0]
        if collection is None:
            collection = context.scene.collection

        created = []
        for endpoint, world_location in (('A', start_world), ('B', end_world)):
            endpoint_obj = bpy.data.objects.new(_cloth_capsule_endpoint_name(root, capsule_index, endpoint), None)
            collection.objects.link(endpoint_obj)
            endpoint_obj.parent = root
            endpoint_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            try:
                endpoint_obj.matrix_world.translation = world_location
            except Exception:
                endpoint_obj.location = world_location
            endpoint_obj.empty_display_type = 'SPHERE'
            endpoint_obj.empty_display_size = 1.0
            endpoint_obj.show_name = False
            endpoint_obj.show_in_front = True
            endpoint_obj.scale = (radius, radius, radius)
            _lock_cloth_collision_rotation(endpoint_obj)
            self._add_bone_follow_constraint(endpoint_obj, arm_obj, bone_name)
            created.append(endpoint_obj)

        try:
            context.view_layer.objects.active = created[0]
            created[0].select_set(True)
            created[1].select_set(True)
        except Exception:
            pass
        return {'FINISHED'}


CLOTH_ALIGN_AXIS_ITEMS = (
    ('X', 'X', 'Align the selected bone chain to +X'),
    ('NEG_X', '-X', 'Align the selected bone chain to -X'),
    ('Y', 'Y', 'Align the selected bone chain to +Y'),
    ('NEG_Y', '-Y', 'Align the selected bone chain to -Y'),
    ('Z', 'Z', 'Align the selected bone chain to +Z'),
    ('NEG_Z', '-Z', 'Align the selected bone chain to -Z'),
)


class TRLAU_OT_align_cloth_chain_to_axis(bpy.types.Operator):
    bl_idname = 'trlau.align_cloth_chain_to_axis'
    bl_label = 'Align to Axis'
    bl_description = 'Rotate selected bone chain(s) so each parent-to-child segment points along the chosen axis\n\nLeaf point-bones are left to inherit the chain rotation so the final bone is not twisted independently'
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(
        name='Axis',
        items=CLOTH_ALIGN_AXIS_ITEMS,
        default='Y',
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        pose_bone = getattr(context, 'active_pose_bone', None)
        if pose_bone is not None and getattr(getattr(pose_bone, 'id_data', None), 'type', None) == 'ARMATURE':
            return getattr(context, 'mode', '') == 'POSE'
        return obj is not None and getattr(obj, 'type', None) == 'ARMATURE' and getattr(context, 'mode', '') == 'POSE'

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=220)

    def draw(self, _context):
        self.layout.prop(self, 'axis')

    @staticmethod
    def _active_armature(context):
        pose_bone = getattr(context, 'active_pose_bone', None)
        arm_obj = getattr(pose_bone, 'id_data', None) if pose_bone is not None else None
        if getattr(arm_obj, 'type', None) == 'ARMATURE':
            return arm_obj
        obj = getattr(context, 'object', None)
        return obj if getattr(obj, 'type', None) == 'ARMATURE' else None

    @staticmethod
    def _axis_vector(axis: str):
        return {
            'X': mathutils.Vector((1.0, 0.0, 0.0)),
            'NEG_X': mathutils.Vector((-1.0, 0.0, 0.0)),
            'Y': mathutils.Vector((0.0, 1.0, 0.0)),
            'NEG_Y': mathutils.Vector((0.0, -1.0, 0.0)),
            'Z': mathutils.Vector((0.0, 0.0, 1.0)),
            'NEG_Z': mathutils.Vector((0.0, 0.0, -1.0)),
        }.get(axis, mathutils.Vector((0.0, 1.0, 0.0)))

    @staticmethod
    def _collect_pose_chain(root_pose_bone):
        ordered_bones = []

        def collect(pose_bone):
            ordered_bones.append(pose_bone)
            children = list(getattr(pose_bone, 'children', []) or [])
            children.sort(key=lambda bone: getattr(bone, 'name', ''))
            for child in children:
                collect(child)

        collect(root_pose_bone)
        return ordered_bones

    @staticmethod
    def _selected_pose_bones(context, arm_obj):
        selected = None
        for attr_name in ('selected_pose_bones_from_active_object', 'selected_pose_bones'):
            try:
                selected = getattr(context, attr_name, None)
            except Exception:
                selected = None
            if selected:
                break

        result = []
        seen = set()
        for pose_bone in list(selected or []):
            if getattr(pose_bone, 'id_data', None) != arm_obj:
                continue
            name = getattr(pose_bone, 'name', None)
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(pose_bone)

        if not result:
            active_pose_bone = getattr(context, 'active_pose_bone', None)
            if active_pose_bone is not None and getattr(active_pose_bone, 'id_data', None) == arm_obj:
                result.append(active_pose_bone)

        result.sort(key=lambda bone: getattr(bone, 'name', ''))
        return result

    @staticmethod
    def _pose_bone_has_selected_ancestor(pose_bone, selected_names):
        parent = getattr(pose_bone, 'parent', None)
        while parent is not None:
            if getattr(parent, 'name', None) in selected_names:
                return True
            parent = getattr(parent, 'parent', None)
        return False

    @classmethod
    def _selected_pose_chain_roots(cls, context, arm_obj):
        selected_bones = cls._selected_pose_bones(context, arm_obj)
        selected_names = {getattr(bone, 'name', '') for bone in selected_bones}
        roots = [bone for bone in selected_bones if not cls._pose_bone_has_selected_ancestor(bone, selected_names)]
        roots.sort(key=lambda bone: getattr(bone, 'name', ''))
        return roots

    @staticmethod
    def _safe_normalized(vector, default):
        try:
            vector = mathutils.Vector(vector)
            if vector.length > 0.000001:
                return vector.normalized()
        except Exception:
            pass
        return mathutils.Vector(default).normalized()

    @staticmethod
    def _orthogonal_axis(axis_vector, preferred_vector=None):
        axis_vector = TRLAU_OT_align_cloth_chain_to_axis._safe_normalized(axis_vector, (0.0, 1.0, 0.0))
        if preferred_vector is not None:
            try:
                preferred = mathutils.Vector(preferred_vector)
                projected = preferred - axis_vector * preferred.dot(axis_vector)
                if projected.length > 0.000001:
                    return projected.normalized()
            except Exception:
                pass

        for candidate in (
            mathutils.Vector((1.0, 0.0, 0.0)),
            mathutils.Vector((0.0, 0.0, 1.0)),
            mathutils.Vector((0.0, 1.0, 0.0)),
        ):
            projected = candidate - axis_vector * candidate.dot(axis_vector)
            if projected.length > 0.000001:
                return projected.normalized()
        return mathutils.Vector((1.0, 0.0, 0.0))

    @staticmethod
    def _bone_length(pose_bone):
        try:
            length = (mathutils.Vector(pose_bone.tail) - mathutils.Vector(pose_bone.head)).length
            if length > 0.000001:
                return float(length)
        except Exception:
            pass
        try:
            length = float(getattr(getattr(pose_bone, 'bone', None), 'length', 0.0))
            if length > 0.000001:
                return length
        except Exception:
            pass
        return 0.1

    @staticmethod
    def _rotation_between_vectors(source_vector, target_vector):
        source = TRLAU_OT_align_cloth_chain_to_axis._safe_normalized(source_vector, (0.0, 1.0, 0.0))
        target = TRLAU_OT_align_cloth_chain_to_axis._safe_normalized(target_vector, (0.0, 1.0, 0.0))
        try:
            return source.rotation_difference(target).to_matrix().to_4x4()
        except Exception:
            pass

        dot = max(-1.0, min(1.0, float(source.dot(target))))
        if dot > 0.999999:
            return mathutils.Matrix.Identity(4)

        if dot < -0.999999:
            axis = TRLAU_OT_align_cloth_chain_to_axis._orthogonal_axis(source, None)
            return mathutils.Matrix.Rotation(3.141592653589793, 4, axis)

        axis = source.cross(target)
        if axis.length <= 0.000001:
            return mathutils.Matrix.Identity(4)
        axis.normalize()
        return mathutils.Matrix.Rotation(source.angle(target), 4, axis)

    @staticmethod
    def _pose_bone_head(pose_bone):
        try:
            return mathutils.Vector(pose_bone.head)
        except Exception:
            try:
                return pose_bone.matrix.translation.copy()
            except Exception:
                return mathutils.Vector((0.0, 0.0, 0.0))

    @staticmethod
    def _pose_bone_tail(pose_bone):
        try:
            return mathutils.Vector(pose_bone.tail)
        except Exception:
            try:
                head = TRLAU_OT_align_cloth_chain_to_axis._pose_bone_head(pose_bone)
                direction = pose_bone.matrix.to_3x3() @ mathutils.Vector((0.0, 1.0, 0.0))
                direction = TRLAU_OT_align_cloth_chain_to_axis._safe_normalized(direction, (0.0, 1.0, 0.0))
                return head + direction * TRLAU_OT_align_cloth_chain_to_axis._bone_length(pose_bone)
            except Exception:
                return TRLAU_OT_align_cloth_chain_to_axis._pose_bone_head(pose_bone)

    @staticmethod
    def _ordered_children(pose_bone, bone_set, order_index):
        children = [child for child in list(getattr(pose_bone, 'children', []) or []) if child in bone_set]
        children.sort(key=lambda bone: (order_index.get(bone, 10**9), getattr(bone, 'name', '')))
        return children

    @staticmethod
    def _update_view_layer(context):
        if context is None:
            return
        try:
            context.view_layer.update()
        except Exception:
            pass

    @staticmethod
    def _rotate_pose_bone_about_head(pose_bone, rotation_matrix):
        head = TRLAU_OT_align_cloth_chain_to_axis._pose_bone_head(pose_bone)
        pose_bone.matrix = (
            mathutils.Matrix.Translation(head)
            @ rotation_matrix
            @ mathutils.Matrix.Translation(-head)
            @ pose_bone.matrix
        )

    @staticmethod
    def _rotate_pose_bone_source_to_axis(pose_bone, source_vector, axis_vector, context=None):
        try:
            source = mathutils.Vector(source_vector)
            if source.length <= 0.000001:
                return False
            rotation_matrix = TRLAU_OT_align_cloth_chain_to_axis._rotation_between_vectors(source, axis_vector)
            TRLAU_OT_align_cloth_chain_to_axis._rotate_pose_bone_about_head(pose_bone, rotation_matrix)
            TRLAU_OT_align_cloth_chain_to_axis._update_view_layer(context)
            return True
        except Exception:
            return False

    @staticmethod
    def _align_pose_chain_to_axis(ordered_bones, axis_vector, context=None):
        if not ordered_bones:
            return 0

        axis_vector = TRLAU_OT_align_cloth_chain_to_axis._safe_normalized(axis_vector, (0.0, 1.0, 0.0))
        bone_set = set(ordered_bones)
        order_index = {bone: index for index, bone in enumerate(ordered_bones)}

        aligned_count = 0

        def align_branch(pose_bone):
            nonlocal aligned_count
            children = TRLAU_OT_align_cloth_chain_to_axis._ordered_children(pose_bone, bone_set, order_index)
            source_vector = None

            if children:
                child_head = TRLAU_OT_align_cloth_chain_to_axis._pose_bone_head(children[0])
                parent_head = TRLAU_OT_align_cloth_chain_to_axis._pose_bone_head(pose_bone)
                source_vector = child_head - parent_head
            else:
                source_vector = None

            if source_vector is not None and source_vector.length > 0.000001:
                if TRLAU_OT_align_cloth_chain_to_axis._rotate_pose_bone_source_to_axis(
                    pose_bone,
                    source_vector,
                    axis_vector,
                    context=context,
                ):
                    aligned_count += 1

            for child in children:
                align_branch(child)

        align_branch(ordered_bones[0])
        return aligned_count

    def execute(self, context):
        arm_obj = self._active_armature(context)
        if arm_obj is None:
            self.report({'WARNING'}, 'Select an armature in Pose Mode')
            return {'CANCELLED'}

        if getattr(context, 'mode', '') != 'POSE':
            self.report({'WARNING'}, 'Align to Axis must be run in Pose Mode')
            return {'CANCELLED'}

        root_pose_bones = self._selected_pose_chain_roots(context, arm_obj)
        if not root_pose_bones:
            self.report({'WARNING'}, 'Select one or more pose bones')
            return {'CANCELLED'}

        axis_vector = self._axis_vector(self.axis).normalized()

        chains = []
        processed_names = set()
        for root_pose_bone in root_pose_bones:
            ordered_bones = []
            for pose_bone in self._collect_pose_chain(root_pose_bone):
                name = getattr(pose_bone, 'name', '')
                if name in processed_names:
                    continue
                processed_names.add(name)
                ordered_bones.append(pose_bone)
            if ordered_bones:
                chains.append(ordered_bones)

        if not chains:
            self.report({'WARNING'}, 'No pose bones found')
            return {'CANCELLED'}

        previous_active = getattr(context.view_layer.objects, 'active', None)
        previous_mode = getattr(context, 'mode', 'POSE')

        try:
            context.view_layer.objects.active = arm_obj
            if previous_mode != 'POSE':
                bpy.ops.object.mode_set(mode='POSE')

            aligned_count = 0
            for ordered_bones in chains:
                aligned_count += self._align_pose_chain_to_axis(ordered_bones, axis_vector, context=context)

            try:
                context.view_layer.update()
            except Exception:
                pass

            axis_label = dict((item[0], item[1]) for item in CLOTH_ALIGN_AXIS_ITEMS).get(self.axis, self.axis)
            chain_word = 'chain' if len(chains) == 1 else 'chains'
            self.report({'INFO'}, f'Rotated {aligned_count} bone(s) across {len(chains)} {chain_word} toward {axis_label}')
            return {'FINISHED'}
        finally:
            try:
                if previous_active is not None:
                    context.view_layer.objects.active = previous_active
            except Exception:
                pass
            try:
                if previous_mode == 'POSE':
                    bpy.ops.object.mode_set(mode='POSE')
            except Exception:
                pass


class TRLAU_OT_apply_cloth_pose_as_rest(bpy.types.Operator):
    bl_idname = 'trlau.apply_cloth_pose_as_rest'
    bl_label = 'Apply Pose as Rest Pose'
    bl_description = 'Bake the current armature deformation as rest pose'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return TRLAU_OT_align_cloth_chain_to_axis.poll(context)

    @staticmethod
    def _active_armature(context):
        return TRLAU_OT_align_cloth_chain_to_axis._active_armature(context)

    @staticmethod
    def _armature_mesh_users(arm_obj):
        users = []
        if arm_obj is None:
            return users
        for obj in list(getattr(bpy.data, 'objects', []) or []):
            if getattr(obj, 'type', None) != 'MESH':
                continue
            uses_armature = False
            for modifier in list(getattr(obj, 'modifiers', []) or []):
                try:
                    if getattr(modifier, 'type', None) == 'ARMATURE' and getattr(modifier, 'object', None) == arm_obj:
                        uses_armature = True
                        break
                except Exception:
                    pass
            if not uses_armature:
                try:
                    uses_armature = getattr(obj, 'parent', None) == arm_obj and getattr(obj, 'parent_type', '') == 'ARMATURE'
                except Exception:
                    uses_armature = False
            if uses_armature:
                users.append(obj)
        return users

    @staticmethod
    def _armature_modifiers(mesh_obj, arm_obj):
        modifiers = []
        for modifier in list(getattr(mesh_obj, 'modifiers', []) or []):
            try:
                if getattr(modifier, 'type', None) == 'ARMATURE' and getattr(modifier, 'object', None) == arm_obj:
                    modifiers.append(modifier)
            except Exception:
                pass
        return modifiers

    @staticmethod
    def _set_selected_objects(objects, active=None):
        try:
            for obj in list(getattr(bpy.context, 'selected_objects', []) or []):
                obj.select_set(False)
        except Exception:
            pass
        for obj in objects:
            try:
                obj.select_set(True)
            except Exception:
                pass
        if active is not None:
            try:
                bpy.context.view_layer.objects.active = active
            except Exception:
                pass

    @staticmethod
    def _modifier_names(mesh_obj):
        return [str(getattr(modifier, 'name', '') or '') for modifier in list(getattr(mesh_obj, 'modifiers', []) or [])]

    def _copy_and_apply_armature_modifier(self, context, mesh_obj, modifier):
        modifier_name = str(getattr(modifier, 'name', '') or '')
        if not modifier_name:
            return False, 'modifier has no name'

        before_names = set(self._modifier_names(mesh_obj))
        copied_modifier_name = None

        self._set_selected_objects([mesh_obj], active=mesh_obj)

        try:
            with context.temp_override(object=mesh_obj, active_object=mesh_obj, selected_objects=[mesh_obj], selected_editable_objects=[mesh_obj]):
                copy_result = bpy.ops.object.modifier_copy(modifier=modifier_name)
        except Exception as exc:
            return False, f'could not copy Armature modifier {modifier_name}: {exc}'

        if isinstance(copy_result, set) and 'CANCELLED' in copy_result:
            return False, f'could not copy Armature modifier {modifier_name}'

        after_names = self._modifier_names(mesh_obj)
        for name in after_names:
            if name not in before_names:
                copied_modifier_name = name
                break
        if not copied_modifier_name:
            copied_candidates = [name for name in after_names if name != modifier_name and name.startswith(modifier_name)]
            copied_modifier_name = copied_candidates[-1] if copied_candidates else None

        try:
            with context.temp_override(object=mesh_obj, active_object=mesh_obj, selected_objects=[mesh_obj], selected_editable_objects=[mesh_obj]):
                apply_result = bpy.ops.object.modifier_apply(modifier=modifier_name)
        except Exception as exc:
            if copied_modifier_name:
                try:
                    copied_modifier = getattr(mesh_obj, 'modifiers', {}).get(copied_modifier_name)
                    if copied_modifier is not None:
                        mesh_obj.modifiers.remove(copied_modifier)
                except Exception:
                    pass
            return False, f'could not apply Armature modifier {modifier_name}: {exc}'

        if isinstance(apply_result, set) and 'CANCELLED' in apply_result:
            if copied_modifier_name:
                try:
                    copied_modifier = getattr(mesh_obj, 'modifiers', {}).get(copied_modifier_name)
                    if copied_modifier is not None:
                        mesh_obj.modifiers.remove(copied_modifier)
                except Exception:
                    pass
            return False, f'could not apply Armature modifier {modifier_name}'

        return True, ''

    def _bake_mesh_armature_deformation(self, context, arm_obj, mesh_obj):
        armature_modifiers = self._armature_modifiers(mesh_obj, arm_obj)
        if not armature_modifiers:
            return False, 'no Armature modifier targets this armature'

        baked_count = 0
        for modifier in list(armature_modifiers):
            modifier_name = str(getattr(modifier, 'name', '') or '')
            current_modifier = getattr(mesh_obj, 'modifiers', {}).get(modifier_name)
            if current_modifier is None:
                continue
            ok, message = self._copy_and_apply_armature_modifier(context, mesh_obj, current_modifier)
            if not ok:
                return False, message
            baked_count += 1

        return baked_count > 0, ''

    def execute(self, context):
        arm_obj = self._active_armature(context)
        if arm_obj is None:
            self.report({'WARNING'}, 'Select an armature in Pose Mode')
            return {'CANCELLED'}
        if getattr(context, 'mode', '') != 'POSE':
            self.report({'WARNING'}, 'Apply Pose as Rest Pose must be run in Pose Mode')
            return {'CANCELLED'}

        mesh_users = self._armature_mesh_users(arm_obj)
        previous_active = getattr(context.view_layer.objects, 'active', None)
        previous_mode = getattr(context, 'mode', 'POSE')
        selected_objects = list(getattr(context, 'selected_objects', []) or [])

        baked_meshes = []
        skipped_meshes = []

        try:
            try:
                context.view_layer.update()
            except Exception:
                pass

            bpy.ops.object.mode_set(mode='OBJECT')
            for mesh_obj in mesh_users:
                ok, message = self._bake_mesh_armature_deformation(context, arm_obj, mesh_obj)
                if ok:
                    baked_meshes.append(mesh_obj)
                else:
                    skipped_meshes.append((mesh_obj, message))

            self._set_selected_objects([arm_obj], active=arm_obj)
            bpy.ops.object.mode_set(mode='POSE')

            try:
                with context.temp_override(object=arm_obj, active_object=arm_obj, selected_objects=[arm_obj]):
                    bpy.ops.pose.armature_apply(selected=False)
            except Exception:
                bpy.ops.pose.armature_apply(selected=False)

            try:
                context.view_layer.update()
            except Exception:
                pass

            if skipped_meshes:
                names = ', '.join(str(getattr(obj, 'name', 'Mesh')) for obj, _message in skipped_meshes[:4])
                if len(skipped_meshes) > 4:
                    names += f', +{len(skipped_meshes) - 4} more'
                self.report({'WARNING'}, f'Applied pose as rest. Baked {len(baked_meshes)} mesh object(s); skipped {len(skipped_meshes)}: {names}')
            else:
                self.report({'INFO'}, f'Applied pose as rest and baked {len(baked_meshes)} skinned mesh object(s)')
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Could not apply pose as rest pose: {exc}')
            return {'CANCELLED'}
        finally:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
            try:
                for obj in list(getattr(context, 'selected_objects', []) or []):
                    obj.select_set(False)
                for obj in selected_objects:
                    obj.select_set(True)
            except Exception:
                pass
            try:
                if previous_active is not None:
                    context.view_layer.objects.active = previous_active
            except Exception:
                pass
            try:
                if previous_mode == 'POSE' and previous_active is not None and getattr(previous_active, 'type', None) == 'ARMATURE':
                    bpy.ops.object.mode_set(mode='POSE')
                elif previous_mode and previous_mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:
                pass


class TRLAU_OT_create_cloth_from_active_bone(bpy.types.Operator):
    bl_idname = 'trlau.create_cloth_from_active_bone'
    bl_label = 'Create Cloth'
    bl_description = 'Create cloth from selected bone(s).\n\nEvery selected bone must be the root (first bone in the chain)'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return TRLAU_OT_align_cloth_chain_to_axis.poll(context)

    @staticmethod
    def _set_bone_state(arm_obj, pose_bone, pinned: bool) -> None:
        bone_name = str(getattr(pose_bone, 'name', '') or '')
        if not bone_name:
            return
        data_bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name)
        for target in (data_bone, pose_bone):
            if target is None:
                continue
            try:
                target['trlau_cloth_enabled'] = True
                target['trlau_cloth_pinned'] = bool(pinned)
            except Exception:
                pass
        _apply_cloth_bone_color(arm_obj, bone_name, bool(pinned))

    def execute(self, context):
        arm_obj = TRLAU_OT_align_cloth_chain_to_axis._active_armature(context)
        if arm_obj is None:
            self.report({'WARNING'}, 'Select an armature in Pose Mode')
            return {'CANCELLED'}
        if getattr(context, 'mode', '') != 'POSE':
            self.report({'WARNING'}, 'Create Cloth must be run in Pose Mode')
            return {'CANCELLED'}

        root_pose_bones = TRLAU_OT_align_cloth_chain_to_axis._selected_pose_chain_roots(context, arm_obj)
        if not root_pose_bones:
            active_pose_bone = getattr(context, 'active_pose_bone', None)
            if active_pose_bone is not None and getattr(active_pose_bone, 'id_data', None) == arm_obj:
                root_pose_bones = [active_pose_bone]
        if not root_pose_bones:
            self.report({'WARNING'}, 'Select one or more pose bones')
            return {'CANCELLED'}

        root = _ensure_cloth_root(context, arm_obj)
        all_authored_bones = []
        processed_names = set()
        pinned_count = 0
        unpinned_count = 0

        for root_pose_bone in root_pose_bones:
            ordered_bones = TRLAU_OT_align_cloth_chain_to_axis._collect_pose_chain(root_pose_bone)
            if not ordered_bones:
                continue
            for index, pose_bone in enumerate(ordered_bones):
                bone_name = str(getattr(pose_bone, 'name', '') or '')
                if not bone_name or bone_name in processed_names:
                    continue
                processed_names.add(bone_name)
                pinned = index == 0
                self._set_bone_state(arm_obj, pose_bone, pinned=pinned)
                all_authored_bones.append(pose_bone)
                if pinned:
                    pinned_count += 1
                else:
                    unpinned_count += 1

        if not all_authored_bones:
            self.report({'WARNING'}, 'No pose bones found')
            return {'CANCELLED'}

        created_rules = _ensure_cloth_collision_rule_helpers(context, root, arm_obj, pose_bones=all_authored_bones)

        try:
            context.view_layer.update()
        except Exception:
            pass

        suffix = f'; created {created_rules} per-point collision rule helper(s)' if created_rules else ''
        chain_word = 'chain' if pinned_count == 1 else 'chains'
        self.report({'INFO'}, f'Created {pinned_count} cloth {chain_word}: {pinned_count} pinned, {unpinned_count} unpinned{suffix}')
        return {'FINISHED'}


class TRLAU_OT_toggle_cloth_preview(bpy.types.Operator):
    bl_idname = 'trlau.toggle_cloth_preview'
    bl_label = 'Preview Cloth'
    bl_description = 'Enable real-time cloth previewing in Blender.\n\nThe simulation may not be entirely accurate to ingame result'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return True
        if obj is not None and trlau_object_type(obj) == 'Cloth':
            return getattr(getattr(obj, 'parent', None), 'type', None) == 'ARMATURE'
        return getattr(context, 'active_pose_bone', None) is not None

    @staticmethod
    def _active_armature(context):
        obj = getattr(context, 'object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return obj
        if obj is not None and trlau_object_type(obj) == 'Cloth':
            parent = getattr(obj, 'parent', None)
            if getattr(parent, 'type', None) == 'ARMATURE':
                return parent
        pose_bone = getattr(context, 'active_pose_bone', None)
        return getattr(pose_bone, 'id_data', None) if pose_bone is not None else None

    @staticmethod
    def preview_enabled_for_armature(context, arm_obj) -> bool:
        root = _find_cloth_root_for_armature(arm_obj)
        if root is None:
            return False
        try:
            from ..core.cloth_simulation import is_preview_enabled
            return bool(is_preview_enabled(root))
        except Exception:
            return False

    @staticmethod
    def _reset_cloth_runtime_state() -> None:
        try:
            from ..core.cloth_simulation import reset_simulation_state
            reset_simulation_state()
        except Exception:
            pass

    def execute(self, context):
        arm_obj = self._active_armature(context)
        if arm_obj is None:
            self.report({'WARNING'}, 'Select an armature or its Cloth empty')
            return {'CANCELLED'}

        root = _ensure_cloth_root(context, arm_obj)
        try:
            from ..core.cloth_simulation import is_preview_enabled, set_preview_enabled
            enabled = not bool(is_preview_enabled(root))
            set_preview_enabled(root, enabled)
        except Exception:
            enabled = True

        self._reset_cloth_runtime_state()
        try:
            context.view_layer.update()
        except Exception:
            pass

        self.report({'INFO'}, 'Cloth preview enabled; press Play on the timeline to simulate' if enabled else 'Cloth preview disabled')
        return {'FINISHED'}


class TRLAU_OT_add_cloth_plane_rule_from_hmarkers(bpy.types.Operator):
    bl_idname = 'trlau.add_cloth_plane_rule_from_hmarkers'
    bl_label = 'Add Cloth Plane Rule from HMarkers'
    bl_description = 'Create an FDPlaneRule authoring object from exactly two selected HMarkers. The HMarkers control the plane faces; the selector box controls affected cloth points on export'
    bl_options = {'REGISTER', 'UNDO'}

    display_width: FloatProperty(
        name='Plane Width',
        description='Initial viewport width for each HMarker plane face. The FDPlaneRule format itself has no width value',
        default=24.0,
        min=0.1,
        soft_max=256.0,
    )
    display_height: FloatProperty(
        name='Plane Height',
        description='Initial viewport height for each HMarker plane face. The FDPlaneRule format itself has no height value',
        default=24.0,
        min=0.1,
        soft_max=256.0,
    )
    selector_source: EnumProperty(
        name='Selector Fit',
        description='Initial source used to size the selector box',
        items=(
            ('DEFAULT', 'Default', 'Create a default selector box around the HMarker pair'),
            ('SELECTED', 'Selected Cloth Points', 'Fit the selector box to selected cloth points or selected cloth collision helpers'),
            ('ALL', 'All Movable Cloth Points', 'Fit the selector box to every movable cloth point in the Cloth root'),
        ),
        default='DEFAULT',
    )

    @classmethod
    def poll(cls, context):
        return len(_selected_hmarkers(context)) >= 2

    def execute(self, context):
        markers = _selected_hmarkers(context)
        if len(markers) != 2:
            self.report({'WARNING'}, 'Select exactly two HMarkers')
            return {'CANCELLED'}

        arm_obj = _armature_from_related_object(markers[0]) or _armature_from_related_object(markers[1])
        if arm_obj is None:
            self.report({'WARNING'}, 'Selected HMarkers are not under a model armature')
            return {'CANCELLED'}
        root = _ensure_cloth_root(context, arm_obj)
        if root is None:
            self.report({'WARNING'}, 'Could not create or find Cloth root')
            return {'CANCELLED'}

        selector_positions = None
        if self.selector_source == 'SELECTED':
            selector_positions = _selected_cloth_point_world_positions(context, arm_obj)
            if not selector_positions:
                self.report({'WARNING'}, 'No selected movable cloth points found; using default selector size')
        elif self.selector_source == 'ALL':
            selector_positions = _all_movable_cloth_point_world_positions(root, arm_obj)
            if not selector_positions:
                self.report({'WARNING'}, 'No movable cloth points found; using default selector size')

        rule_obj = _create_cloth_plane_rule_objects(
            context,
            root,
            markers[0],
            markers[1],
            selector_positions=selector_positions,
            display_width=float(self.display_width),
            display_height=float(self.display_height),
        )
        try:
            bpy.ops.object.select_all(action='DESELECT')
            rule_obj.select_set(True)
            context.view_layer.objects.active = rule_obj
        except Exception:
            pass
        self.report({'INFO'}, f'Created ClothPlaneRule from HMarkers {_hmarker_stored_index(markers[0])} and {_hmarker_stored_index(markers[1])}')
        return {'FINISHED'}


class TRLAU_OT_fit_cloth_plane_rule_selector(bpy.types.Operator):
    bl_idname = 'trlau.fit_cloth_plane_rule_selector'
    bl_label = 'Fit Cloth Plane Rule Selector'
    bl_description = 'Resize selected ClothPlaneRule selector boxes. The selector box controls which movable cloth points receive FDPlaneRule records on export'
    bl_options = {'REGISTER', 'UNDO'}

    source: EnumProperty(
        name='Source',
        items=(
            ('SELECTED', 'Selected Cloth Points', 'Fit to selected movable cloth points or selected cloth collision helpers'),
            ('ALL', 'All Movable Cloth Points', 'Fit to every movable cloth point in the rule\'s Cloth root'),
            ('MARKERS', 'Marker Pair Default', 'Reset to a default box around the two referenced HMarkers'),
        ),
        default='SELECTED',
    )

    @classmethod
    def poll(cls, context):
        return bool(_rule_objects_from_context(context))

    def execute(self, context):
        rules = _rule_objects_from_context(context)
        if not rules:
            self.report({'WARNING'}, 'Select a ClothPlaneRule, one of its child objects, or a Cloth root')
            return {'CANCELLED'}

        changed = 0
        for rule_obj in rules:
            root = _find_ancestor_with_type(rule_obj, 'Cloth')
            arm_obj = getattr(root, 'parent', None) if getattr(getattr(root, 'parent', None), 'type', None) == 'ARMATURE' else _armature_from_related_object(rule_obj)
            markers = _markers_for_rule(rule_obj)
            positions = []
            if self.source == 'SELECTED':
                positions = _selected_cloth_point_world_positions(context, arm_obj)
                if not positions:
                    continue
                bounds = _bounds_from_world_points(positions, minimum_extent=8.0)
            elif self.source == 'ALL':
                positions = _all_movable_cloth_point_world_positions(root, arm_obj) if root is not None else []
                if not positions:
                    continue
                bounds = _bounds_from_world_points(positions, minimum_extent=8.0)
            else:
                if len(markers) < 2:
                    continue
                bounds = _default_selector_bounds_for_markers(markers[0], markers[1])

            selector = _find_rule_selector(rule_obj)
            if selector is None:
                collection = _collection_for_cloth_root(context, root or rule_obj)
                try:
                    parent_inv = rule_obj.matrix_world.inverted_safe()
                except Exception:
                    parent_inv = mathutils.Matrix.Identity(4)
                mesh = _make_selector_mesh_from_bounds(f'{rule_obj.name}_Selector_Mesh', bounds, parent_inv)
                if mesh is None:
                    continue
                selector = bpy.data.objects.new(f'{rule_obj.name}_Selector', mesh)
                _link_object_to_collection(selector, collection)
                selector.parent = rule_obj
                selector.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                selector.show_name = False
                selector.show_in_front = True
                try:
                    selector.show_wire = True
                    selector.display_type = 'WIRE'
                    selector.data.materials.append(_make_or_get_cloth_plane_material())
                except Exception:
                    pass
                selector['trlau_cloth_enabled'] = True
                selector['trlau_export_ignore'] = True
                try:
                    selector['trlau_cloth_plane_rule_index'] = int(rule_obj.get('trlau_cloth_plane_rule_index', 0) or 0)
                except Exception:
                    pass
                changed += 1
            else:
                if _set_selector_mesh_from_bounds(selector, bounds, rule_obj):
                    changed += 1

        if changed <= 0:
            self.report({'WARNING'}, 'No selector boxes were changed')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Updated {changed} selector box(es)')
        return {'FINISHED'}


class TRLAU_OT_rebuild_cloth_plane_rule_visuals(bpy.types.Operator):
    bl_idname = 'trlau.rebuild_cloth_plane_rule_visuals'
    bl_label = 'Rebuild Cloth Plane Rule Visuals'
    bl_description = 'Rebuild selected ClothPlaneRule plane-face meshes and repair missing HMarker follow constraints where possible'
    bl_options = {'REGISTER', 'UNDO'}

    display_width: FloatProperty(
        name='Plane Width',
        default=24.0,
        min=0.1,
        soft_max=256.0,
    )
    display_height: FloatProperty(
        name='Plane Height',
        default=24.0,
        min=0.1,
        soft_max=256.0,
    )
    use_existing_size: BoolProperty(
        name='Use Existing Size',
        description='Use each rule object\'s stored display width/height instead of the values above',
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return bool(_rule_objects_from_context(context))

    def execute(self, context):
        rules = _rule_objects_from_context(context)
        rebuilt = 0
        missing = 0
        for rule_obj in rules:
            if len(_markers_for_rule(rule_obj)) < 2:
                missing += 1
                continue
            width = None if self.use_existing_size else float(self.display_width)
            height = None if self.use_existing_size else float(self.display_height)
            rebuilt += _rebuild_rule_plane_meshes(rule_obj, display_width=width, display_height=height)
            for plane_obj in _find_rule_plane_children(rule_obj):
                marker = _marker_from_plane_child(plane_obj)
                if marker is None:
                    continue
                has_follow = False
                for constraint in list(getattr(plane_obj, 'constraints', []) or []):
                    try:
                        if constraint.type == 'COPY_TRANSFORMS' and getattr(constraint, 'target', None) is marker:
                            has_follow = True
                            break
                    except Exception:
                        pass
                if not has_follow:
                    try:
                        follow = plane_obj.constraints.new(type='COPY_TRANSFORMS')
                        follow.name = 'TRLAU Cloth PlaneRule HMarker Follow'
                        follow.target = marker
                        follow.target_space = 'WORLD'
                        follow.owner_space = 'WORLD'
                    except Exception:
                        pass
        if rebuilt <= 0 and missing:
            self.report({'WARNING'}, f'No visuals rebuilt; {missing} rule(s) could not resolve two HMarkers')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Rebuilt {rebuilt} plane visual object(s)')
        return {'FINISHED'}


class TRLAU_OT_validate_cloth_plane_rules(bpy.types.Operator):
    bl_idname = 'trlau.validate_cloth_plane_rules'
    bl_label = 'Validate Cloth Plane Rules'
    bl_description = 'Check ClothPlaneRule authoring objects for missing HMarkers, missing selector boxes, and empty affected-point volumes'
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None

    def execute(self, context):
        rules = _rule_objects_from_context(context)
        if not rules:
            roots = []
            obj = getattr(context, 'object', None)
            arm_obj = _armature_from_related_object(obj)
            if arm_obj is not None:
                root = _find_cloth_root_for_armature(arm_obj)
                if root is not None:
                    roots.append(root)
            if not roots:
                roots = [obj for obj in list(getattr(bpy.data, 'objects', []) or []) if trlau_object_type(obj) == 'Cloth']
            for root in roots:
                for child in list(getattr(root, 'children', []) or []):
                    try:
                        if trlau_object_type(child) == 'ClothPlaneRule':
                            rules.append(child)
                    except Exception:
                        pass
        if not rules:
            self.report({'WARNING'}, 'No ClothPlaneRule objects found')
            return {'CANCELLED'}

        issue_count = 0
        valid_count = 0
        for rule_obj in rules:
            issues = []
            markers = _markers_for_rule(rule_obj)
            if len(markers) < 2:
                issues.append('missing HMarker pair')
            if len(_find_rule_plane_children(rule_obj)) < 2:
                issues.append('missing plane face child')
            if _find_rule_selector(rule_obj) is None:
                issues.append('missing selector box')
            root = _find_ancestor_with_type(rule_obj, 'Cloth')
            arm_obj = getattr(root, 'parent', None) if getattr(getattr(root, 'parent', None), 'type', None) == 'ARMATURE' else _armature_from_related_object(rule_obj)
            selector = _find_rule_selector(rule_obj)
            if selector is not None and root is not None:
                try:
                    world = selector.matrix_world.copy()
                    mesh = selector.data
                    used_points = [world @ vertex.co for vertex in list(getattr(mesh, 'vertices', []) or [])]
                    if len(used_points) >= 2:
                        xs = [p.x for p in used_points]
                        ys = [p.y for p in used_points]
                        zs = [p.z for p in used_points]
                        min_v = mathutils.Vector((min(xs), min(ys), min(zs)))
                        max_v = mathutils.Vector((max(xs), max(ys), max(zs)))
                        affected = 0
                        for point in _all_movable_cloth_point_world_positions(root, arm_obj):
                            if min_v.x <= point.x <= max_v.x and min_v.y <= point.y <= max_v.y and min_v.z <= point.z <= max_v.z:
                                affected += 1
                        if affected <= 0:
                            issues.append('selector contains no movable cloth points')
                    else:
                        issues.append('selector has no bounds vertices')
                except Exception:
                    issues.append('selector could not be evaluated')
            if issues:
                issue_count += 1
                self.report({'WARNING'}, f'{rule_obj.name}: {", ".join(issues)}')
            else:
                valid_count += 1

        if issue_count:
            self.report({'WARNING'}, f'Validated {len(rules)} PlaneRule(s): {valid_count} valid, {issue_count} with issue(s)')
        else:
            self.report({'INFO'}, f'Validated {len(rules)} PlaneRule(s): no issues found')
        return {'FINISHED'}


class VIEW3D_PT_trlau_cloth(bpy.types.Panel):
    bl_label = 'Cloth'
    bl_idname = 'VIEW3D_PT_trlau_cloth'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_order = 20
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return getattr(context, 'mode', '') == 'POSE'
        if obj is not None and trlau_object_type(obj) in {'Cloth', 'ClothCollision', 'ClothPoint', 'ClothCollisionRule', 'ClothCapsuleEndpoint', 'ClothPlaneRule', 'ClothPlaneRulePlane', 'ClothPlaneRuleSelector', 'HMarker'}:
            return True
        return getattr(context, 'active_pose_bone', None) is not None and getattr(context, 'mode', '') == 'POSE'

    @staticmethod
    def _draw_idprop(layout, obj, key: str, text: str):
        if obj is not None and key in obj:
            layout.prop(obj, f'["{key}"]', text=text)

    @staticmethod
    def _active_armature(context):
        obj = getattr(context, 'object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return obj
        if obj is not None and trlau_object_type(obj) == 'Cloth':
            parent = getattr(obj, 'parent', None)
            if getattr(parent, 'type', None) == 'ARMATURE':
                return parent
        pose_bone = getattr(context, 'active_pose_bone', None)
        return getattr(pose_bone, 'id_data', None) if pose_bone is not None else None

    def _draw_bone_authoring(self, layout, context):
        box = layout.box()
        box.label(text='Cloth Assignment')
        box.operator('trlau.create_cloth_from_active_bone', text='Create Cloth')
        preview_active = TRLAU_OT_toggle_cloth_preview.preview_enabled_for_armature(context, self._active_armature(context))
        try:
            box.operator('trlau.toggle_cloth_preview', text='Preview Cloth', depress=preview_active)
        except TypeError:
            box.operator('trlau.toggle_cloth_preview', text='Preview Cloth')
        row = box.row(align=True)
        op = row.operator('trlau.set_cloth_bone_state', text='Pin')
        op.state = 'PIN'
        op = row.operator('trlau.set_cloth_bone_state', text='Unpin')
        op.state = 'UNPIN'
        op = row.operator('trlau.set_cloth_bone_state', text='Clear')
        op.state = 'CLEAR'
        if getattr(context, 'active_pose_bone', None) is not None:
            box.operator('trlau.add_cloth_point', text='Add Collision Sphere')
            box.operator('trlau.add_cloth_capsule', text='Add Collision Capsule')
        box.operator('trlau.align_cloth_chain_to_axis', text='Align to Axis')
        box.operator('trlau.apply_cloth_pose_as_rest', text='Apply Pose as Rest Pose')

    def _draw_plane_rule_authoring(self, layout, context):
        box = layout.box()
        box.label(text='Cloth Plane Rules')
        box.operator('trlau.add_cloth_plane_rule_from_hmarkers', text='Add From Selected HMarkers')
        row = box.row(align=True)
        op = row.operator('trlau.fit_cloth_plane_rule_selector', text='Fit Selected')
        if op is not None:
            op.source = 'SELECTED'
        op = row.operator('trlau.fit_cloth_plane_rule_selector', text='Fit All')
        if op is not None:
            op.source = 'ALL'
        row = box.row(align=True)
        op = row.operator('trlau.fit_cloth_plane_rule_selector', text='Reset Selector')
        if op is not None:
            op.source = 'MARKERS'
        row.operator('trlau.rebuild_cloth_plane_rule_visuals', text='Rebuild Visuals')
        box.operator('trlau.validate_cloth_plane_rules', text='Validate Plane Rules')

    def draw(self, context):
        layout = self.layout
        obj = context.object
        obj_type = trlau_object_type(obj) if obj is not None else ''

        if obj_type == 'Cloth':
            _cleanup_cloth_root_custom_props(obj)
            data_box = layout.box()
            data_box.label(text='Cloth Setup')
            arm_obj = self._active_armature(context)
            preview_active = TRLAU_OT_toggle_cloth_preview.preview_enabled_for_armature(context, arm_obj)
            try:
                data_box.operator('trlau.toggle_cloth_preview', text='Preview Cloth', depress=preview_active)
            except TypeError:
                data_box.operator('trlau.toggle_cloth_preview', text='Preview Cloth')
            scene = getattr(context, 'scene', None)
            if scene is not None and hasattr(scene, 'trlau_cloth_preview_gravity'):
                data_box.prop(scene, 'trlau_cloth_preview_gravity', text='Gravity')
            if scene is not None and hasattr(scene, 'trlau_cloth_preview_drag'):
                data_box.prop(scene, 'trlau_cloth_preview_drag', text='Drag')
            self._draw_plane_rule_authoring(layout, context)
            return

        if obj_type in {'ClothPlaneRule', 'ClothPlaneRulePlane', 'ClothPlaneRuleSelector'}:
            rule_obj = _find_cloth_plane_rule_parent(obj)
            data_box = layout.box()
            data_box.label(text='Cloth Plane Rule')
            if rule_obj is not None:
                data_box.prop(rule_obj, '["trlau_cloth_enabled"]', text='Enabled')
                if 'trlau_cloth_plane_display_width' in rule_obj:
                    data_box.prop(rule_obj, '["trlau_cloth_plane_display_width"]', text='Display Width')
                if 'trlau_cloth_plane_display_height' in rule_obj:
                    data_box.prop(rule_obj, '["trlau_cloth_plane_display_height"]', text='Display Height')
                markers = _markers_for_rule(rule_obj)
                if markers:
                    text_value = ', '.join(str(_hmarker_stored_index(marker)) for marker in markers)
                    data_box.label(text=f'HMarkers: {text_value}')
                selector = _find_rule_selector(rule_obj)
                data_box.label(text='Selector: present' if selector is not None else 'Selector: missing', icon='CHECKMARK' if selector is not None else 'ERROR')
            self._draw_plane_rule_authoring(layout, context)
            hint = layout.box()
            hint.label(text='HMarkers move/rotate the plane faces.', icon='INFO')
            hint.label(text='Selector box controls which movable cloth points get FDPlaneRules.', icon='INFO')
            return

        if obj_type == 'HMarker':
            self._draw_plane_rule_authoring(layout, context)
            hint = layout.box()
            hint.label(text='Select exactly two HMarkers, then add a Cloth Plane Rule.', icon='INFO')
            return

        if obj_type == 'ClothCapsuleEndpoint':
            try:
                obj.show_name = False
            except Exception:
                pass
            _lock_cloth_collision_rotation(obj)
            data_box = layout.box()
            data_box.label(text='Cloth Collision Capsule Endpoint')
            if getattr(obj, 'type', None) == 'EMPTY':
                data_box.prop(obj, 'trlau_cloth_radius_ui', text='Radius')
            return

        if obj_type in {'ClothCollision', 'ClothPoint'}:
            try:
                obj.show_name = False
            except Exception:
                pass
            _lock_cloth_collision_rotation(obj)
            data_box = layout.box()
            data_box.label(text='Cloth Collision Sphere')
            if getattr(obj, 'type', None) == 'EMPTY':
                data_box.prop(obj, 'trlau_cloth_radius_ui', text='Radius')
            return

        if obj_type == 'ClothCollisionRule':
            try:
                obj.show_name = False
            except Exception:
                pass
            _lock_cloth_collision_rotation(obj)
            data_box = layout.box()
            data_box.label(text='Per-Point Cloth Collision Rule')
            if getattr(obj, 'type', None) == 'EMPTY':
                data_box.prop(obj, 'trlau_cloth_radius_ui', text='Radius')
            return

        if getattr(context, 'mode', '') == 'POSE':
            self._draw_bone_authoring(layout, context)

classes = (
    TRLAU_OT_set_cloth_bone_state,
    TRLAU_OT_add_cloth_point,
    TRLAU_OT_add_cloth_capsule,
    TRLAU_OT_align_cloth_chain_to_axis,
    TRLAU_OT_apply_cloth_pose_as_rest,
    TRLAU_OT_create_cloth_from_active_bone,
    TRLAU_OT_toggle_cloth_preview,
    TRLAU_OT_add_cloth_plane_rule_from_hmarkers,
    TRLAU_OT_fit_cloth_plane_rule_selector,
    TRLAU_OT_rebuild_cloth_plane_rule_visuals,
    TRLAU_OT_validate_cloth_plane_rules,
    VIEW3D_PT_trlau_cloth,
)
