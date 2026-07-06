from __future__ import annotations

from .object_utils import trlau_object_type
from dataclasses import dataclass, field
import math
from typing import Optional

import bpy
import mathutils
from bpy.app.handlers import persistent

_EPSILON = 1.0e-6
_MAX_DISTANCE_RULE = 999999.0
_COLLISION_HELPER_POINT_PREFIX = '__trlau_collision_helper__:'
_GAME_TO_BLENDER_RUNTIME_GRAVITY_SCALE = 0.1
_DEFAULT_PREVIEW_GRAVITY = 10.0
_DEFAULT_PREVIEW_DRAG = 0.1


def _cloth_preview_settings_update(_self=None, _context=None) -> None:
    reset_simulation_state()


def _scene_float(scene, attr: str, default: float) -> float:
    try:
        return float(getattr(scene, attr))
    except Exception:
        return float(default)


def preview_gravity(scene=None) -> float:
    scene = scene or getattr(bpy.context, 'scene', None)
    return _scene_float(scene, 'trlau_cloth_preview_gravity', _DEFAULT_PREVIEW_GRAVITY)


def preview_drag(scene=None) -> float:
    scene = scene or getattr(bpy.context, 'scene', None)
    value = _scene_float(scene, 'trlau_cloth_preview_drag', _DEFAULT_PREVIEW_DRAG)
    return max(0.0, min(1.0, float(value)))


@dataclass
class _FDPoint:
    segment: int
    flags: int
    joint_order: int = 0
    up_to: int = 0
    offset: mathutils.Vector = field(default_factory=lambda: mathutils.Vector((0.0, 0.0, 0.0)))
    # Runtime support for authored Blender cloth bones whose names are not bone_N.
    bone_name: str = ''


@dataclass
class _FDJointMap:
    segment: int
    flags: int
    axis: int
    joint_order: int
    center: int
    points: tuple[int, int, int, int]
    # Runtime support for authored Blender cloth bones whose names are not bone_N.
    bone_name: str = ''


@dataclass
class _FDDistanceRule:
    p0: int
    p1: int
    f0: int
    f1: int
    min_dist_sq: float
    max_dist_sq: float

# I had, like, no hopes for capsules to work, but wow they do
@dataclass
class _FDCapsuleRule:
    p0: int
    p1: int
    f0: int
    f1: int
    radius: float


@dataclass
class _FDPlaneRule:
    point: int
    flag: int
    plane_segment: tuple[int, int]


@dataclass
class _FDSphereRule:
    point: int
    radius: float


@dataclass
class _FDPointMoveRule:
    point: int
    count: int


@dataclass
class _FDSetup:
    points: list[_FDPoint] = field(default_factory=list)
    maps: list[_FDJointMap] = field(default_factory=list)
    dist_rules: list[_FDDistanceRule] = field(default_factory=list)
    capsule_rules: list[_FDCapsuleRule] = field(default_factory=list)
    plane_rules: list[_FDPlaneRule] = field(default_factory=list)
    sphere_rules: list[_FDSphereRule] = field(default_factory=list)
    move_rules: list[_FDPointMoveRule] = field(default_factory=list)
    signature: str = ''
    live_collision_rules_signature: str = ''


@dataclass
class _RootSimulationState:
    last_frame: float | None = None
    setup_signature: str = ''
    setup_point_keys: list[tuple] = field(default_factory=list)
    positions: list[mathutils.Vector] = field(default_factory=list)
    previous_positions: list[mathutils.Vector] = field(default_factory=list)
    rest_positions: list[mathutils.Vector] = field(default_factory=list)
    # Smoothed output is kept separate from the physics state.  It reduces
    # viewport jitter while posing without feeding the filtered values back into
    # Verlet integration.
    output_positions: list[mathutils.Vector] = field(default_factory=list)
    last_base_positions: list[mathutils.Vector] = field(default_factory=list)
    last_live_collision_rules_signature: str = ''

    # Per-root runtime caches.  Cloth simulation is called from both the depsgraph
    # handler and the pose-mode timer, so rebuilding FD tables and taking sqrt()s
    # for every rule on every tick is a noticeable viewport cost.  These caches are
    # invalidated by compact signatures derived from bone/helper metadata.
    setup_cache_key: tuple | None = None
    setup_cache: object | None = None
    prepared_distance_signature: tuple | None = None
    prepared_distance_rules: list[tuple] = field(default_factory=list)
    motion_limits_signature: tuple | None = None
    motion_limits: list[float] = field(default_factory=list)


@dataclass
class _ClothBoneInfo:
    name: str
    bone: object
    pose_bone: object
    pinned: bool


_states_by_root_pointer: dict[int, _RootSimulationState] = {}
_preview_enabled_root_pointers: set[int] = set()
_is_in_cloth_simulation = False


def _idprop(candidate, key: str, default=None):
    if candidate is None:
        return default
    try:
        if key in candidate:
            return candidate.get(key, default)
    except Exception:
        pass
    return default



def _object_type(obj) -> str:
    try:
        return trlau_object_type(obj)
    except Exception:
        return ''


def _is_cloth_root(obj) -> bool:
    try:
        return _object_type(obj) == 'Cloth'
    except Exception:
        return False


def _is_collision_enabled(obj) -> bool:
    try:
        return bool(obj.get('trlau_cloth_enabled', True))
    except Exception:
        return True


def _root_pointer(root) -> int | None:
    try:
        return int(root.as_pointer())
    except Exception:
        return None


def is_preview_enabled(root) -> bool:
    pointer = _root_pointer(root)
    return pointer is not None and pointer in _preview_enabled_root_pointers


def set_preview_enabled(root, enabled: bool) -> None:
    pointer = _root_pointer(root)
    if pointer is None:
        return
    if enabled:
        _preview_enabled_root_pointers.add(pointer)
    else:
        _preview_enabled_root_pointers.discard(pointer)


def _cleanup_stale_cloth_root_props(root) -> None:
    for key in ('trlau_cloth_gravity', 'trlau_cloth_drag', 'trlau_cloth_preview_enabled'):
        try:
            if root is not None and key in root:
                del root[key]
        except Exception:
            pass


def _is_preview_enabled(root) -> bool:
    return is_preview_enabled(root)


def _is_animation_playing() -> bool:
    try:
        window_manager = bpy.context.window_manager
        for window in list(getattr(window_manager, 'windows', []) or []):
            screen = getattr(window, 'screen', None)
            if bool(getattr(screen, 'is_animation_playing', False)):
                return True
    except Exception:
        pass
    return False


def _scene_frame(scene) -> float:
    try:
        return float(scene.frame_current_final)
    except Exception:
        try:
            return float(scene.frame_current) + float(getattr(scene, 'frame_subframe', 0.0) or 0.0)
        except Exception:
            return 0.0


def _world_matrix(obj, depsgraph=None) -> mathutils.Matrix:
    try:
        if depsgraph is not None:
            evaluated = obj.evaluated_get(depsgraph)
            return evaluated.matrix_world.copy()
    except Exception:
        pass
    try:
        return obj.matrix_world.copy()
    except Exception:
        return mathutils.Matrix.Identity(4)



def _evaluated_object(obj, depsgraph=None):
    try:
        if obj is not None and depsgraph is not None:
            return obj.evaluated_get(depsgraph)
    except Exception:
        pass
    return obj


def _world_radius(obj, depsgraph=None) -> float:
    # Cloth collision radius is authored with Object Properties > Transform > Scale.
    # Empty Data > Size is intentionally ignored so it can remain a viewport-only
    # display setting.  Using matrix_world.to_scale() is also avoided here because
    # the imported collision helpers often use Child Of constraints to follow
    # bones; evaluated constraint matrices can hide or distort the local object
    # scale that the artist is editing.
    try:
        scale_values = tuple(float(value) for value in getattr(obj, 'scale', (1.0, 1.0, 1.0)))
        if scale_values:
            return max(0.0, max(abs(value) for value in scale_values))
    except Exception:
        pass
    return 0.0


def _cloth_roots_for_scene(scene) -> list[object]:
    try:
        iterable = list(getattr(scene, 'objects', []) or [])
    except Exception:
        iterable = list(getattr(bpy.data, 'objects', []) or [])

    roots = []
    for obj in iterable:
        if not _is_cloth_root(obj):
            continue
        armature = getattr(obj, 'parent', None)
        if getattr(armature, 'type', None) != 'ARMATURE':
            continue
        roots.append(obj)
    return roots


def _as_bool(value, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return default


def _is_cloth_bone(data_bone, pose_bone) -> bool:
    for candidate in (pose_bone, data_bone):
        if candidate is None:
            continue
        if _as_bool(_idprop(candidate, 'trlau_cloth_enabled', False), False):
            return True
        try:
            if 'trlau_cloth_pinned' in candidate:
                return True
        except Exception:
            pass
    return False


def _is_pinned_cloth_bone(data_bone, pose_bone) -> bool:
    for candidate in (pose_bone, data_bone):
        try:
            if candidate is not None and 'trlau_cloth_pinned' in candidate:
                return bool(candidate.get('trlau_cloth_pinned', False))
        except Exception:
            pass
    return False


def _bone_sort_key(bone_name: str) -> tuple[int, str]:
    try:
        if bone_name.startswith('bone_'):
            return (int(bone_name.split('_', 1)[1]), bone_name)
    except Exception:
        pass
    return (10**9, bone_name)


def _segment_from_bone_name(name: str, default: int = 0) -> int:
    try:
        if str(name).startswith('bone_'):
            return int(str(name).split('_', 1)[1])
    except Exception:
        pass
    return int(default)


def _collect_cloth_bones(armature) -> dict[str, _ClothBoneInfo]:
    data_bones = getattr(getattr(armature, 'data', None), 'bones', {})
    pose_bones = getattr(getattr(armature, 'pose', None), 'bones', {})
    cloth_bones: dict[str, _ClothBoneInfo] = {}
    try:
        iterable = list(data_bones)
    except Exception:
        iterable = []
    for data_bone in iterable:
        name = str(getattr(data_bone, 'name', '') or '')
        if not name:
            continue
        pose_bone = pose_bones.get(name) if pose_bones is not None else None
        if pose_bone is None or not _is_cloth_bone(data_bone, pose_bone):
            continue
        cloth_bones[name] = _ClothBoneInfo(
            name=name,
            bone=data_bone,
            pose_bone=pose_bone,
            pinned=_is_pinned_cloth_bone(data_bone, pose_bone),
        )
    return cloth_bones


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


def _parse_point(data: dict) -> _FDPoint:
    return _FDPoint(
        segment=_safe_int(data.get('segment', 0)),
        flags=_safe_int(data.get('flags', 0)),
        joint_order=_safe_int(data.get('jointOrder', 0)),
        up_to=_safe_int(data.get('upTo', 0)),
        offset=mathutils.Vector((
            _safe_float(data.get('x', 0.0)),
            _safe_float(data.get('y', 0.0)),
            _safe_float(data.get('z', 0.0)),
        )),
    )


def _generated_setup_from_scene(
    armature,
    cloth_bones: dict[str, _ClothBoneInfo],
    *,
    exclude_bone_names: set[str] | None = None,
) -> _FDSetup | None:
    """Generate a runtime FD setup for authored Blender cloth bones."""
    if not cloth_bones:
        return None
    exclude_bone_names = set(exclude_bone_names or set())
    active_names = [str(name) for name in cloth_bones if str(name) not in exclude_bone_names]
    if not active_names:
        return None

    data_bones = getattr(getattr(armature, 'data', None), 'bones', {})

    def data_bone(name: str):
        try:
            return data_bones.get(str(name)) if data_bones is not None else None
        except Exception:
            return None

    def sort_key(name: str) -> tuple[int, str]:
        return _bone_sort_key(str(name))

    def segment_from_name(name: str, default: int = 0) -> int:
        try:
            return int(_segment_from_bone_name(str(name), int(default)))
        except Exception:
            return int(default)

    def bone_head_position(name: str) -> mathutils.Vector:
        try:
            bone = data_bone(name)
            if bone is not None:
                return mathutils.Vector(tuple(getattr(bone, 'head_local')))
        except Exception:
            pass
        return mathutils.Vector((0.0, 0.0, 0.0))

    def dominant_axis_code(vector: mathutils.Vector) -> int:
        try:
            x, y, z = abs(float(vector.x)), abs(float(vector.y)), abs(float(vector.z))
        except Exception:
            return 5
        if x >= y and x >= z:
            return 3
        if y >= x and y >= z:
            return 4
        return 5

    active_set = set(active_names)
    cloth_by_name: dict[str, dict] = {}
    children_by_parent: dict[str, list[str]] = {name: [] for name in active_names}
    chain_roots: list[str] = []
    chain_paths: list[list[str]] = []

    for default_index, name in enumerate(sorted(active_names, key=sort_key)):
        info = cloth_bones.get(name)
        bone = data_bone(name)
        parent_name = ''
        try:
            parent = getattr(bone, 'parent', None)
            parent_name = str(getattr(parent, 'name', '') or '') if parent is not None else ''
        except Exception:
            parent_name = ''
        segment = segment_from_name(name, default_index)
        cloth_by_name[name] = {
            'key': f'bone:{int(segment)}',
            'bone_name': str(name),
            'armature_parent_name': parent_name,
            'segment': int(segment),
            'flags': 5 if bool(getattr(info, 'pinned', False)) else 4,
            'position': bone_head_position(name),
            'children': [],
        }

    for name, entry in list(cloth_by_name.items()):
        parent_name = ''
        try:
            bone = data_bone(name)
            parent = getattr(bone, 'parent', None)
            parent_name = str(getattr(parent, 'name', '') or '') if parent is not None else ''
        except Exception:
            parent_name = ''
        if parent_name in active_set and int(entry.get('flags', 4)) != 5:
            children_by_parent.setdefault(parent_name, []).append(name)
        else:
            chain_roots.append(name)

    for names in children_by_parent.values():
        names.sort(key=sort_key)
    chain_roots.sort(key=sort_key)

    def walk(name: str, prefix: list[str]) -> None:
        current = prefix + [name]
        children = children_by_parent.get(name, [])
        if not children:
            chain_paths.append(current)
            return
        for child_name in children:
            walk(child_name, current)

    for root_name in chain_roots:
        walk(root_name, [])

    def chain_has_movable(path_names: list[str]) -> bool:
        return any(int(cloth_by_name.get(name, {}).get('flags', 0)) == 4 for name in path_names)

    def chain_root_is_pinned(path_names: list[str]) -> bool:
        return bool(path_names) and int(cloth_by_name.get(path_names[0], {}).get('flags', 0)) == 5

    def chain_order_key(path_names: list[str]) -> tuple[int, int, int, float, int, str]:
        root_name = path_names[0] if path_names else ''
        root_entry = cloth_by_name.get(root_name, {})
        root_segment = int(root_entry.get('segment', 0))
        parent_name = str(root_entry.get('armature_parent_name', '') or '')
        parent_segment = segment_from_name(parent_name, 0) if parent_name else 0
        try:
            delta = bone_head_position(root_name) - (bone_head_position(parent_name) if parent_name else mathutils.Vector((0.0, 0.0, 0.0)))
            x = float(delta.x)
            y = float(delta.y)
        except Exception:
            x = 0.0
            y = 0.0
        return (
            int(parent_segment),
            0 if y < 0.0 else 1,
            0 if x >= 0.0 else 1,
            -abs(x),
            int(root_segment),
            str(root_name),
        )

    ordered_chain_paths = sorted([list(path) for path in chain_paths if chain_has_movable(path)], key=chain_order_key)
    primary_chain_root_name = str(ordered_chain_paths[0][0]) if ordered_chain_paths else ''

    point_entries: list[dict] = []
    key_to_index: dict[str, int] = {}
    seen_bones: set[str] = set()

    def append_bone_name(bone_name: str) -> None:
        if not bone_name or bone_name in seen_bones or bone_name not in cloth_by_name:
            return
        seen_bones.add(bone_name)
        entry = dict(cloth_by_name[bone_name])
        entry['old_index'] = len(point_entries)
        point_entries.append(entry)

    reversed_paths = list(reversed(ordered_chain_paths))
    pinned_chain_roots = {
        str(path_names[0])
        for path_names in ordered_chain_paths
        if chain_root_is_pinned(path_names)
    }

    # Export writes movable points from leaf to root, walking the sorted paths in
    # reverse.  Matching that order is important because CollisionRule helpers
    # and joint maps are point-index sensitive after re-import.
    for path_names in reversed_paths:
        movable_names = [name for name in path_names if int(cloth_by_name.get(name, {}).get('flags', 0)) == 4]
        for bone_name in reversed(movable_names):
            append_bone_name(bone_name)

    isolated_pins = [
        name for name, entry in cloth_by_name.items()
        if int(entry.get('flags', 0)) == 5 and name not in pinned_chain_roots
    ]
    isolated_pins.sort(key=lambda name: (-int(cloth_by_name.get(name, {}).get('segment', 0)), str(name)))
    for bone_name in isolated_pins:
        append_bone_name(bone_name)

    for path_names in reversed_paths:
        root_name = str(path_names[0]) if path_names else ''
        if root_name == primary_chain_root_name:
            continue
        if chain_root_is_pinned(path_names):
            append_bone_name(root_name)

    for bone_name in sorted(cloth_by_name, key=sort_key):
        if bone_name != primary_chain_root_name:
            append_bone_name(bone_name)

    if primary_chain_root_name and primary_chain_root_name in cloth_by_name:
        try:
            if int(cloth_by_name.get(primary_chain_root_name, {}).get('flags', 0)) == 5:
                append_bone_name(primary_chain_root_name)
        except Exception:
            pass

    if not point_entries:
        return None

    # Generated-only data uses append order as the final order.
    setup = _FDSetup()
    setup.signature = 'generated_export_like:'
    for new_index, entry in enumerate(point_entries):
        key_to_index[str(entry['key'])] = int(new_index)
        setup.points.append(_FDPoint(
            segment=int(entry.get('segment', new_index)),
            flags=int(entry.get('flags', 4)),
            joint_order=0,
            up_to=0,
            offset=mathutils.Vector((0.0, 0.0, 0.0)),
            bone_name=str(entry.get('bone_name', f'bone_{int(entry.get("segment", new_index))}')),
        ))

    def index_for_bone_name(name: str) -> int | None:
        entry = cloth_by_name.get(str(name))
        if not entry:
            return None
        value = key_to_index.get(str(entry.get('key')))
        return int(value) if value is not None else None

    def axis_for_path(path_names: list[str]) -> int:
        if len(path_names) >= 2:
            return dominant_axis_code(bone_head_position(path_names[1]) - bone_head_position(path_names[0]))
        if path_names:
            entry = cloth_by_name.get(path_names[0], {})
            parent_name = str(entry.get('armature_parent_name', '') or '')
            if parent_name:
                return dominant_axis_code(bone_head_position(path_names[0]) - bone_head_position(parent_name))
        return 5

    indexed_paths: list[list[int]] = []
    path_axis_by_root_index: dict[int, int] = {}
    path_names_by_root_index: dict[int, list[str]] = {}
    for path_names in ordered_chain_paths:
        indices = [index_for_bone_name(name) for name in path_names]
        indices = [int(index) for index in indices if index is not None]
        if not indices:
            continue
        indexed_paths.append(indices)
        root_index = int(indices[0])
        path_axis_by_root_index[root_index] = int(axis_for_path(path_names))
        path_names_by_root_index[root_index] = list(path_names)

    def point_flag_value(index: int) -> int:
        try:
            return _rule_flag_for_point(setup, int(index))
        except Exception:
            return 0

    def distance_sq(index_a: int, index_b: int) -> float:
        try:
            return max(0.0, float((point_entries[int(index_a)]['position'] - point_entries[int(index_b)]['position']).length_squared))
        except Exception:
            return 0.0

    added_distance_keys: set[tuple[int, int, int]] = set()

    def add_distance(p0: int, p1: int, min_dist_sq: float, max_dist_sq: float, *, mode: int = 0) -> None:
        if p0 < 0 or p1 < 0 or p0 >= len(setup.points) or p1 >= len(setup.points):
            return
        dedupe = (min(int(p0), int(p1)), max(int(p0), int(p1)), int(mode))
        if dedupe in added_distance_keys:
            return
        added_distance_keys.add(dedupe)
        setup.dist_rules.append(_FDDistanceRule(
            p0=int(p0),
            p1=int(p1),
            f0=point_flag_value(p0),
            f1=point_flag_value(p1),
            min_dist_sq=max(0.0, float(min_dist_sq)),
            max_dist_sq=max(0.0, float(max_dist_sq)),
        ))

    mapped_indices: set[int] = set()
    for indices in indexed_paths:
        has_movable = any(int(setup.points[int(index)].flags) == 4 for index in indices)
        if not has_movable:
            continue
        axis_code = int(path_axis_by_root_index.get(int(indices[0]), 5))
        for order, current_index in enumerate(indices):
            if current_index in mapped_indices:
                continue
            mapped_indices.add(int(current_index))
            if order + 1 < len(indices):
                next_index = int(indices[order + 1])
                map_points = (int(current_index), int(next_index), 0, 0) if axis_code == 4 else (int(next_index), int(current_index), 0, 0)
            elif order > 0:
                # Terminal weighted bones still need a non-degenerate map.
                # Original game data maps the last point back to its parent
                # neighbor, e.g. center=tip and points=(tip,parent,0,0).
                # Previewing (tip,tip,0,0) leaves the last weighted bone with
                # no usable FD direction.
                previous_index = int(indices[order - 1])
                map_points = (int(current_index), int(previous_index), 0, 0)
            else:
                map_points = (int(current_index), int(current_index), 0, 0)
            current_entry = point_entries[int(current_index)]
            setup.maps.append(_FDJointMap(
                segment=int(current_entry.get('segment', current_index)),
                flags=0,
                axis=int(axis_code),
                joint_order=0,
                center=int(current_index),
                points=map_points,
                bone_name=str(current_entry.get('bone_name', '')),
            ))

    deferred_anchor_rules: list[tuple[int, int, float, float, int]] = []

    def collect_anchor_rules(indices: list[int], *, exact_axis: bool) -> None:
        if not indices or int(setup.points[int(indices[0])].flags) != 5:
            return
        root_index = int(indices[0])
        start = 1 if exact_axis else 2
        for descendant_index in indices[start:]:
            deferred_anchor_rules.append((
                root_index,
                int(descendant_index),
                0.0,
                distance_sq(root_index, int(descendant_index)),
                1,
            ))

    primary_path_root_index: int | None = index_for_bone_name(primary_chain_root_name) if primary_chain_root_name else None

    for indices in indexed_paths:
        has_movable = any(int(setup.points[int(index)].flags) == 4 for index in indices)
        if not has_movable:
            continue
        axis_code = int(path_axis_by_root_index.get(int(indices[0]), 5))
        exact_axis = axis_code == 4
        for order in range(max(0, len(indices) - 1)):
            p0, p1 = int(indices[order]), int(indices[order + 1])
            dist2 = distance_sq(p0, p1)
            add_distance(p0, p1, dist2 if exact_axis else 0.0, dist2, mode=0)
        if exact_axis and indices:
            last = int(indices[-1])
            add_distance(last, last, 0.0, 0.0, mode=0)

        is_primary_path = primary_path_root_index is not None and int(indices[0]) == int(primary_path_root_index)
        if is_primary_path:
            collect_anchor_rules(indices, exact_axis=exact_axis)
            for p0, p1, min_dist, max_dist, mode in list(deferred_anchor_rules):
                add_distance(p0, p1, min_dist, max_dist, mode=mode)
            deferred_anchor_rules.clear()
        else:
            collect_anchor_rules(indices, exact_axis=exact_axis)

    paths_by_parent: dict[str, list[list[int]]] = {}
    for path in indexed_paths:
        if not path or not any(int(setup.points[int(index)].flags) == 4 for index in path):
            continue
        parent_name = str(point_entries[int(path[0])].get('armature_parent_name', '') or '')
        paths_by_parent.setdefault(parent_name, []).append(path)

    for parent_name, paths in paths_by_parent.items():
        if len(paths) < 2:
            continue
        parent_pos = bone_head_position(parent_name) if parent_name else mathutils.Vector((0.0, 0.0, 0.0))

        def lateral_group_key(path: list[int]) -> tuple[int, float]:
            try:
                delta = point_entries[int(path[0])]['position'] - parent_pos
                return (0 if float(delta.y) < 0.0 else 1, -float(delta.x))
            except Exception:
                return (1, 0.0)

        grouped_paths: dict[int, list[list[int]]] = {}
        for path in paths:
            grouped_paths.setdefault(lateral_group_key(path)[0], []).append(path)

        for group_index in sorted(grouped_paths):
            ordered_paths = sorted(grouped_paths[group_index], key=lambda item: lateral_group_key(item)[1])
            if len(ordered_paths) < 2:
                continue
            for left, right in zip(ordered_paths, ordered_paths[1:]):
                max_depth = min(len(left), len(right))
                for depth in range(max_depth - 1, -1, -1):
                    p0, p1 = int(left[depth]), int(right[depth])
                    add_distance(p0, p1, 0.0, distance_sq(p0, p1), mode=2)

    for p0, p1, min_dist, max_dist, mode in list(deferred_anchor_rules):
        add_distance(p0, p1, min_dist, max_dist, mode=mode)

    for index, point in enumerate(setup.points):
        if int(point.flags) == 4:
            setup.move_rules.append(_FDPointMoveRule(point=int(index), count=1))

    signature_parts = []
    for index, entry in enumerate(point_entries):
        signature_parts.append(f'{index}:{entry.get("segment", 0)}:{entry.get("flags", 0)}:{entry.get("bone_name", "")}')
    for path in indexed_paths:
        signature_parts.append('path:' + ','.join(str(index) for index in path) + f':axis={path_axis_by_root_index.get(int(path[0]), 5)}')
    setup.signature += '|'.join(signature_parts)
    return setup


def _visible_collision_helper_name_index(obj) -> int | None:
    try:
        name = str(getattr(obj, 'name', '') or '')
        marker = '_Cloth_Collision_'
        if marker in name:
            suffix = name.split(marker, 1)[1].split('.', 1)[0]
            digits = []
            for char in suffix:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return int(''.join(digits))
    except Exception:
        pass
    return None



def _visible_pin_collision_helper_name_index(obj) -> int | None:
    try:
        name = str(getattr(obj, 'name', '') or '')
        marker = '_Cloth_PinCollision_'
        if marker in name:
            suffix = name.split(marker, 1)[1].split('.', 1)[0]
            digits = []
            for char in suffix:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                return int(''.join(digits))
    except Exception:
        pass
    return None


def _is_pin_collision_helper(obj) -> bool:
    try:
        if _object_type(obj) == 'ClothPinCollision':
            return True
        return _visible_pin_collision_helper_name_index(obj) is not None
    except Exception:
        return False


def _pin_collision_owner_segment(obj) -> int | None:
    try:
        for constraint in list(getattr(obj, 'constraints', []) or []):
            if getattr(constraint, 'type', '') != 'CHILD_OF':
                continue
            subtarget = str(getattr(constraint, 'subtarget', '') or '')
            if not subtarget:
                continue
            return int(_segment_from_bone_name(subtarget, 0))
    except Exception:
        pass
    return None


def _find_point_by_segment_and_flags(setup: _FDSetup, segment: int, allowed_flags: set[int]) -> int | None:
    try:
        target_segment = int(segment)
    except Exception:
        return None
    for index, point in enumerate(list(getattr(setup, 'points', []) or [])):
        try:
            if int(point.segment) == target_segment and int(point.flags) in allowed_flags:
                return int(index)
        except Exception:
            pass
    return None

def _collision_helper_marker_name(point: _FDPoint) -> str:
    try:
        name = str(getattr(point, 'bone_name', '') or '')
        if name.startswith(_COLLISION_HELPER_POINT_PREFIX):
            return name[len(_COLLISION_HELPER_POINT_PREFIX):]
    except Exception:
        pass
    return ''


def _find_generated_collision_point_index(setup: _FDSetup, object_name: str) -> int | None:
    target_name = str(object_name or '')
    if not target_name:
        return None
    for index, point in enumerate(list(getattr(setup, 'points', []) or [])):
        try:
            if int(point.flags) == 1 and _collision_helper_marker_name(point) == target_name:
                return int(index)
        except Exception:
            pass
    return None


def _visible_collision_point_for_helper(child, setup: _FDSetup) -> int | None:
    point_index = _visible_collision_point_index(child)
    if point_index is not None and 0 <= int(point_index) < len(setup.points):
        try:
            if int(setup.points[int(point_index)].flags) == 1:
                return int(point_index)
        except Exception:
            pass
    return _find_generated_collision_point_index(setup, str(getattr(child, 'name', '') or ''))


def _cloth_bone_names_from_setup(setup: _FDSetup) -> set[str]:
    names: set[str] = set()
    for point in list(getattr(setup, 'points', []) or []):
        try:
            if int(getattr(point, 'flags', 0)) not in {4, 5}:
                continue
            explicit = str(getattr(point, 'bone_name', '') or '')
            if explicit and not explicit.startswith(_COLLISION_HELPER_POINT_PREFIX):
                names.add(explicit)
            else:
                names.add(f'bone_{int(getattr(point, "segment", 0))}')
        except Exception:
            pass
    return names


def _append_visible_collision_points(root, armature, setup: _FDSetup, depsgraph=None) -> _FDSetup:
    """Append FD collision points for hand-authored visible colliders.

    Imported ClothSetup data already has collision ClothPoint records.  Newly
    authored Blender cloths do not, but visible ClothCollisionRule helpers need a
    fixed FD point to reference.  This adds generated flags=1 points for visible
    collision helpers that do not already resolve to an imported point.
    """
    if setup is None:
        return setup
    try:
        armature_inv = _safe_inverted(_world_matrix(armature, depsgraph))
    except Exception:
        armature_inv = mathutils.Matrix.Identity(4)
    signature_parts: list[str] = []
    for child in list(getattr(root, 'children', []) or []):
        if _object_type(child) not in {'ClothCollision', 'ClothPoint'}:
            continue
        if _visible_collision_point_for_helper(child, setup) is not None:
            continue
        try:
            local_pos = armature_inv @ _world_matrix(child, depsgraph).translation.copy()
        except Exception:
            local_pos = mathutils.Vector((0.0, 0.0, 0.0))
        point_index = len(setup.points)
        setup.points.append(_FDPoint(
            segment=0,
            flags=1,
            joint_order=0,
            up_to=3,
            offset=local_pos.copy(),
            bone_name=f'{_COLLISION_HELPER_POINT_PREFIX}{str(getattr(child, "name", "") or "")}',
        ))
        name_index = _visible_collision_helper_name_index(child)
        # Keep the runtime setup identity structural only.  The helper position is
        # a live base position, not topology; including it in the signature makes
        # every modal transform invalidate the simulation state and snaps cloth
        # bones back to their initialized/rest pose until the transform ends.
        signature_parts.append(f'{point_index}:{name_index}:{getattr(child, "name", "")}')
    if signature_parts:
        setup.signature = f'{setup.signature}|visible_collision_points:' + '|'.join(signature_parts)
    return setup

def _load_setup(root, armature, cloth_bones: dict[str, _ClothBoneInfo]) -> _FDSetup | None:
    setup = _generated_setup_from_scene(armature, cloth_bones)
    if setup is None:
        return None
    setup = _append_visible_collision_points(root, armature, setup)
    return _setup_with_visible_collision_rules(root, setup, armature)


def _cache_float(value, digits: int = 5) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return round(value, int(digits))
    except Exception:
        pass
    return 0.0


def _vector_cache_key(value, digits: int = 5) -> tuple[float, float, float]:
    try:
        return tuple(_cache_float(component, digits) for component in value)[:3]
    except Exception:
        return (0.0, 0.0, 0.0)


def _object_translation_cache_key(obj) -> tuple[float, float, float]:
    try:
        return _vector_cache_key(getattr(obj, 'matrix_world', mathutils.Matrix.Identity(4)).translation, 4)
    except Exception:
        return (0.0, 0.0, 0.0)


def _object_scale_cache_key(obj) -> tuple[float, float, float]:
    try:
        return _vector_cache_key(getattr(obj, 'scale', (1.0, 1.0, 1.0)), 5)
    except Exception:
        return (1.0, 1.0, 1.0)


def _idprop_cache_value(obj, key: str, default=None):
    try:
        value = obj.get(key, default)
    except Exception:
        return default
    if isinstance(value, float):
        return _cache_float(value)
    if isinstance(value, (int, bool)):
        return int(value)
    if value is None:
        return default
    return str(value)


def _cloth_setup_cache_key(root, armature, cloth_bones: dict[str, _ClothBoneInfo]) -> tuple:
    bone_parts: list[tuple] = []
    data_bones = getattr(getattr(armature, 'data', None), 'bones', {})
    for name in sorted((str(key) for key in cloth_bones.keys()), key=_bone_sort_key):
        info = cloth_bones.get(name)
        try:
            bone = data_bones.get(name) if data_bones is not None else None
        except Exception:
            bone = None
        try:
            parent = getattr(bone, 'parent', None)
            parent_name = str(getattr(parent, 'name', '') or '') if parent is not None else ''
        except Exception:
            parent_name = ''
        try:
            head_key = _vector_cache_key(getattr(bone, 'head_local', (0.0, 0.0, 0.0)), 5)
        except Exception:
            head_key = (0.0, 0.0, 0.0)
        bone_parts.append((name, bool(getattr(info, 'pinned', False)), parent_name, head_key))

    direct_helper_parts: list[tuple] = []
    for child in list(getattr(root, 'children', []) or []):
        child_type = _object_type(child)
        if child_type in {'ClothCollision', 'ClothPoint'}:
            direct_helper_parts.append((
                child_type,
                str(getattr(child, 'name', '') or ''),
                _visible_collision_helper_name_index(child),
                int(_is_collision_enabled(child)),
            ))
        elif _is_pin_collision_helper(child):
            direct_helper_parts.append((
                'ClothPinCollision',
                str(getattr(child, 'name', '') or ''),
                _visible_pin_collision_helper_name_index(child),
                _pin_collision_owner_segment(child),
                int(_is_collision_enabled(child)),
            ))
        elif _is_collision_rules_group(child):
            direct_helper_parts.append(('ClothCollisionRules', str(getattr(child, 'name', '') or '')))

    rule_parts: list[tuple] = []
    for child in list(_iter_collision_rule_helpers(root)):
        try:
            name_info = _collision_rule_name_info(child)
            _name_order, name_target, name_bone_segment, name_collision = name_info
            has_stable_target = name_target is not None or name_bone_segment is not None
            has_stable_collision = name_collision is not None
            translation_key = (0.0, 0.0, 0.0) if has_stable_target and has_stable_collision else _object_translation_cache_key(child)
            rule_parts.append((
                str(getattr(child, 'name', '') or ''),
                int(_is_collision_enabled(child)),
                name_info,
                _object_scale_cache_key(child),
                translation_key,
            ))
        except Exception:
            continue

    return (
        tuple(bone_parts),
        tuple(sorted(direct_helper_parts, key=lambda item: str(item))),
        tuple(sorted(rule_parts, key=lambda item: str(item))),
    )


def _load_setup_cached(root, armature, cloth_bones: dict[str, _ClothBoneInfo], state: _RootSimulationState) -> _FDSetup | None:
    try:
        cache_key = _cloth_setup_cache_key(root, armature, cloth_bones)
    except Exception:
        cache_key = None
    if cache_key is not None and state.setup_cache_key == cache_key and isinstance(state.setup_cache, _FDSetup):
        return state.setup_cache

    setup = _load_setup(root, armature, cloth_bones)
    if setup is not None and cache_key is not None:
        state.setup_cache_key = cache_key
        state.setup_cache = setup
    else:
        state.setup_cache_key = None
        state.setup_cache = None
    state.prepared_distance_signature = None
    state.motion_limits_signature = None
    return setup


def _safe_inverted(matrix: mathutils.Matrix) -> mathutils.Matrix:
    try:
        return matrix.inverted_safe()
    except Exception:
        try:
            return matrix.inverted()
        except Exception:
            return mathutils.Matrix.Identity(4)


def _rule_flag_for_point(setup: _FDSetup, index: int) -> int:
    try:
        return 1 if int(setup.points[int(index)].flags) in {1, 5} else 0
    except Exception:
        return 0


def _is_collision_rule_helper(obj) -> bool:
    try:
        if _object_type(obj) == 'ClothCollisionRule':
            return True
        name = str(getattr(obj, 'name', '') or '')
        return '_Cloth_CollisionRule_' in name
    except Exception:
        return False


def _is_collision_rules_group(obj) -> bool:
    try:
        if _object_type(obj) == 'ClothCollisionRules':
            return True
    except Exception:
        pass
    try:
        return '_Cloth_CollisionRules' in str(getattr(obj, 'name', '') or '')
    except Exception:
        return False


def _iter_object_descendants(obj):
    if obj is None:
        return
    for child in list(getattr(obj, 'children', []) or []):
        yield child
        yield from _iter_object_descendants(child)


def _iter_collision_rule_helpers(root):
    if root is None:
        return
    for child in list(getattr(root, 'children', []) or []):
        if _is_collision_rule_helper(child):
            yield child
        elif _is_collision_rules_group(child):
            for descendant in _iter_object_descendants(child):
                if _is_collision_rule_helper(descendant):
                    yield descendant


def _collision_rule_root(obj):
    current = obj
    seen = set()
    while current is not None:
        try:
            key = int(current.as_pointer())
        except Exception:
            key = id(current)
        if key in seen:
            break
        seen.add(key)
        if current is not obj and _is_cloth_root(current):
            return current
        current = getattr(current, 'parent', None)
    return None


def _collision_helper_maps_for_rule_cleanup(root) -> tuple[dict[str, object], dict[int, object]]:
    by_name: dict[str, object] = {}
    by_name_index: dict[int, object] = {}
    if root is None:
        return by_name, by_name_index
    for candidate in list(getattr(root, 'children', []) or []):
        try:
            candidate_type = _object_type(candidate)
            is_collision_helper = candidate_type in {'ClothCollision', 'ClothPoint'} or _is_pin_collision_helper(candidate)
        except Exception:
            is_collision_helper = False
        if not is_collision_helper:
            continue
        candidate_name = str(getattr(candidate, 'name', '') or '')
        if candidate_name:
            by_name[candidate_name] = candidate
        try:
            if _is_pin_collision_helper(candidate):
                name_index = _visible_pin_collision_helper_name_index(candidate)
            else:
                name_index = _visible_collision_helper_name_index(candidate)
        except Exception:
            name_index = None
        if name_index is not None:
            by_name_index[int(name_index)] = candidate
    return by_name, by_name_index


def _remove_orphaned_collision_rule_helpers() -> int:
    """Delete per-point collision rules whose named source collision helper was deleted."""
    removed_count = 0
    root_maps: dict[int, tuple[dict[str, object], dict[int, object]]] = {}
    for rule_obj in list(getattr(bpy.data, 'objects', []) or []):
        if not _is_collision_rule_helper(rule_obj):
            continue
        root = _collision_rule_root(rule_obj)
        if root is None:
            continue
        try:
            root_key = int(root.as_pointer())
        except Exception:
            root_key = id(root)
        if root_key not in root_maps:
            root_maps[root_key] = _collision_helper_maps_for_rule_cleanup(root)
        _helpers_by_name, helpers_by_name_index = root_maps[root_key]
        _order, _target, _bone_segment, collision_index = _collision_rule_name_info(rule_obj)
        if collision_index is not None and int(collision_index) in helpers_by_name_index:
            continue

        try:
            bpy.data.objects.remove(rule_obj, do_unlink=True)
            removed_count += 1
        except Exception:
            pass
    return removed_count

def _trlau_cloth_collision_rule_cleanup_timer():
    if _is_in_cloth_simulation:
        return 0.5
    try:
        _remove_orphaned_collision_rule_helpers()
    except Exception:
        pass
    return 0.5



def _collision_rule_name_info(obj) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (rule_order, target_point_index, target_bone_segment, collision_index) parsed from helper name.

    Imported helpers use names such as:
        *_Cloth_CollisionRule_012_B036_P004_C010

    The B### segment and C### collision helper identity are current scene data
    encoded in the object name, not legacy custom-property fallback. P### is
    kept only as an additional stable point hint for imported files.
    """
    try:
        name = str(getattr(obj, 'name', '') or '')
        marker = '_Cloth_CollisionRule_'
        if marker not in name:
            return None, None, None, None
        suffix = name.split(marker, 1)[1].split('.', 1)[0]
        parts = suffix.split('_')
        order = int(parts[0]) if parts and parts[0].isdigit() else None
        target = None
        bone_segment = None
        collision = None
        for part in parts[1:]:
            if len(part) >= 2 and part[0].upper() == 'P' and part[1:].isdigit():
                target = int(part[1:])
            elif len(part) >= 2 and part[0].upper() == 'B' and part[1:].isdigit():
                bone_segment = int(part[1:])
            elif len(part) >= 2 and part[0].upper() == 'C' and part[1:].isdigit():
                collision = int(part[1:])
        return order, target, bone_segment, collision
    except Exception:
        return None, None, None, None


def _collision_rule_helper_index(obj, default: int = 0) -> int:
    order, _target, _bone_segment, _collision = _collision_rule_name_info(obj)
    if order is not None:
        return int(order)
    return int(default)


def _collision_rule_helper_radius(obj) -> float:
    radius = _world_radius(obj)
    if radius > 0.0:
        return float(radius)
    return 0.0


def _visible_collision_rule_data(root, setup: _FDSetup, armature=None) -> tuple[list[dict], set[tuple[int, int]], str]:
    items: list[dict] = []
    known_pairs: set[tuple[int, int]] = set()
    signature_parts: list[str] = []

    collision_index_by_name_index: dict[int, int] = {}
    collision_enabled_by_name_index: dict[int, bool] = {}
    for candidate in list(getattr(root, 'children', []) or []):
        helper_index = None
        name_index = None
        if _object_type(candidate) in {'ClothCollision', 'ClothPoint'}:
            helper_index = _visible_collision_point_for_helper(candidate, setup)
            name_index = _visible_collision_helper_name_index(candidate)
        elif _is_pin_collision_helper(candidate):
            name_index = _visible_pin_collision_helper_name_index(candidate)
            owner_segment = _pin_collision_owner_segment(candidate)
            helper_index = _find_point_by_segment_and_flags(setup, int(owner_segment), {5}) if owner_segment is not None else None
        else:
            continue
        if helper_index is None or name_index is None:
            continue
        collision_index_by_name_index[int(name_index)] = int(helper_index)
        collision_enabled_by_name_index[int(name_index)] = _is_collision_enabled(candidate)

    base_positions: list[mathutils.Vector] | None = None

    def nearest_point_index_to_helper(helper_obj, allowed_flags: set[int]) -> int | None:
        nonlocal base_positions
        if armature is None:
            return None
        if base_positions is None:
            try:
                base_positions = _point_base_positions(armature, setup, _cloth_bone_names_from_setup(setup))
                _apply_visible_collision_points_to_base_positions(root, armature, setup, base_positions)
            except Exception:
                base_positions = []
        if not base_positions:
            return None
        try:
            armature_inv = _safe_inverted(_world_matrix(armature))
            helper_pos = armature_inv @ _world_matrix(helper_obj).translation.copy()
        except Exception:
            return None
        best_index = None
        best_dist = None
        for point_index, point in enumerate(list(getattr(setup, 'points', []) or [])):
            try:
                if int(point.flags) not in allowed_flags:
                    continue
                dist = float((base_positions[int(point_index)] - helper_pos).length_squared)
            except Exception:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_index = int(point_index)
        return best_index

    def valid_point(index: int, allowed_flags: set[int] | None = None) -> bool:
        try:
            if index < 0 or index >= len(setup.points):
                return False
            if allowed_flags is not None and int(setup.points[int(index)].flags) not in allowed_flags:
                return False
            return True
        except Exception:
            return False

    for raw_index, child in enumerate(list(_iter_collision_rule_helpers(root))):
        if not _is_collision_rule_helper(child):
            continue
        _name_order, name_target, name_bone_segment, name_collision = _collision_rule_name_info(child)
        target_segment = int(name_bone_segment) if name_bone_segment is not None else -1
        target_index = int(name_target) if name_target is not None else -1
        collision_index = int(name_collision) if name_collision is not None else -1

        if target_segment >= 0:
            segment_target = _find_point_by_segment_and_flags(setup, int(target_segment), {4})
            if segment_target is not None:
                target_index = int(segment_target)
        if not valid_point(target_index, {4}):
            nearest_target = nearest_point_index_to_helper(child, {4})
            if nearest_target is not None:
                target_index = int(nearest_target)

        has_stable_collision_reference = name_collision is not None
        if name_collision is not None:
            if int(name_collision) not in collision_index_by_name_index:
                order = _collision_rule_helper_index(child, raw_index)
                signature_parts.append(f'{order}:0:missing_collision:C{int(name_collision):03d}')
                continue
            if not bool(collision_enabled_by_name_index.get(int(name_collision), True)):
                order = _collision_rule_helper_index(child, raw_index)
                signature_parts.append(f'{order}:0:disabled_collision:C{int(name_collision):03d}')
                continue
            collision_index = int(collision_index_by_name_index[int(name_collision)])
        if not valid_point(collision_index, {1, 5}):
            if has_stable_collision_reference:
                continue
            nearest_collision = nearest_point_index_to_helper(child, {1, 5})
            if nearest_collision is not None:
                collision_index = int(nearest_collision)
        if not valid_point(target_index, {4}) or not valid_point(collision_index, {1, 5}):
            continue

        p0, p1 = int(target_index), int(collision_index)
        pair_key = (min(int(p0), int(p1)), max(int(p0), int(p1)))
        known_pairs.add(pair_key)
        radius = _collision_rule_helper_radius(child)
        min_dist_sq = radius * radius if radius > 0.0 else 0.0
        max_dist_sq = 1000000.0
        enabled = _is_collision_enabled(child) and min_dist_sq > 0.0
        f0, f1 = _rule_flag_for_point(setup, p0), _rule_flag_for_point(setup, p1)
        order = _collision_rule_helper_index(child, raw_index)
        signature_parts.append(f'{order}:{int(enabled)}:{p0}:{p1}:{f0}:{f1}:{min_dist_sq:.6f}:{max_dist_sq:.6f}')
        if not enabled:
            continue
        items.append({
            'order': int(order),
            'raw_index': int(raw_index),
            'p0': int(p0),
            'p1': int(p1),
            'f0': int(f0),
            'f1': int(f1),
            'min_dist_sq': float(min_dist_sq),
            'max_dist_sq': float(max_dist_sq),
            'pair_key': pair_key,
        })
    items.sort(key=lambda item: (int(item['order']), int(item['raw_index'])))
    signature = '|'.join(signature_parts)
    return items, known_pairs, signature

def _setup_with_visible_collision_rules(root, setup: _FDSetup, armature=None) -> _FDSetup:
    items, known_pairs, signature = _visible_collision_rule_data(root, setup, armature)
    if not signature:
        return setup

    replacement_by_pair: dict[tuple[int, int], list[dict]] = {}
    for item in items:
        replacement_by_pair.setdefault(item['pair_key'], []).append(item)

    new_rules: list[_FDDistanceRule] = []
    emitted_pairs: set[tuple[int, int]] = set()
    for rule in setup.dist_rules:
        pair_key = (min(int(rule.p0), int(rule.p1)), max(int(rule.p0), int(rule.p1)))
        is_collision_pair = False
        try:
            is_collision_pair = int(setup.points[int(rule.p0)].flags) == 1 or int(setup.points[int(rule.p1)].flags) == 1
        except Exception:
            is_collision_pair = False
        if pair_key in known_pairs and is_collision_pair:
            if pair_key not in emitted_pairs:
                for item in replacement_by_pair.get(pair_key, []):
                    new_rules.append(_FDDistanceRule(
                        p0=int(item['p0']),
                        p1=int(item['p1']),
                        f0=int(item['f0']),
                        f1=int(item['f1']),
                        min_dist_sq=float(item['min_dist_sq']),
                        max_dist_sq=float(item['max_dist_sq']),
                    ))
                emitted_pairs.add(pair_key)
            continue
        new_rules.append(rule)

    for pair_key, pair_items in replacement_by_pair.items():
        if pair_key in emitted_pairs:
            continue
        for item in pair_items:
            new_rules.append(_FDDistanceRule(
                p0=int(item['p0']),
                p1=int(item['p1']),
                f0=int(item['f0']),
                f1=int(item['f1']),
                min_dist_sq=float(item['min_dist_sq']),
                max_dist_sq=float(item['max_dist_sq']),
            ))

    setup.dist_rules = new_rules
    # Do not append the editable rule values to setup.signature. That signature
    # controls whether Verlet state is still valid; changing a CollisionRule
    # helper scale should update constraints live, not reinitialize the cloth.
    setup.live_collision_rules_signature = signature
    return setup


def _base_matrix_for_bone_name(
    armature,
    bone_name: str,
    cloth_bone_names: set[str],
    cache: dict[str, mathutils.Matrix],
    pose_source_armature=None,
) -> mathutils.Matrix:
    if bone_name in cache:
        return cache[bone_name].copy()
    pose_owner = pose_source_armature if pose_source_armature is not None else armature
    pose_bones = getattr(getattr(pose_owner, 'pose', None), 'bones', {})
    data_bones = getattr(getattr(armature, 'data', None), 'bones', {})
    data_bone = data_bones.get(bone_name) if data_bones is not None else None
    pose_bone = pose_bones.get(bone_name) if pose_bones is not None else None

    if bone_name not in cloth_bone_names and pose_bone is not None:
        try:
            matrix = pose_bone.matrix.copy()
            cache[bone_name] = matrix.copy()
            return matrix
        except Exception:
            pass

    if data_bone is None:
        matrix = mathutils.Matrix.Identity(4)
        cache[bone_name] = matrix.copy()
        return matrix

    try:
        parent = getattr(data_bone, 'parent', None)
        parent_name = str(getattr(parent, 'name', '') or '') if parent is not None else ''
        if parent_name:
            parent_base = _base_matrix_for_bone_name(armature, parent_name, cloth_bone_names, cache, pose_source_armature)
            parent_rest = getattr(parent, 'matrix_local', None)
            if parent_rest is not None:
                local_rest = _safe_inverted(parent_rest) @ data_bone.matrix_local
                matrix = parent_base @ local_rest
            else:
                matrix = data_bone.matrix_local.copy()
        else:
            matrix = data_bone.matrix_local.copy()
    except Exception:
        try:
            matrix = data_bone.matrix_local.copy()
        except Exception:
            matrix = mathutils.Matrix.Identity(4)
    cache[bone_name] = matrix.copy()
    return matrix


def _point_base_position_armature(
    armature,
    point: _FDPoint,
    cloth_bone_names: set[str],
    base_matrix_cache: dict[str, mathutils.Matrix],
    pose_source_armature=None,
) -> mathutils.Vector:
    bone_name = str(getattr(point, 'bone_name', '') or f'bone_{int(point.segment)}')
    if bone_name.startswith(_COLLISION_HELPER_POINT_PREFIX):
        return mathutils.Vector(tuple(point.offset))
    try:
        matrix = _base_matrix_for_bone_name(armature, bone_name, cloth_bone_names, base_matrix_cache, pose_source_armature)
        return matrix @ point.offset
    except Exception:
        return mathutils.Vector(tuple(point.offset))


def _point_base_positions(armature, setup: _FDSetup, cloth_bone_names: set[str], depsgraph=None) -> list[mathutils.Vector]:
    cache: dict[str, mathutils.Matrix] = {}
    pose_source = _evaluated_object(armature, depsgraph)
    return [_point_base_position_armature(armature, point, cloth_bone_names, cache, pose_source) for point in setup.points]


def _visible_collision_point_index(obj) -> int | None:
    try:
        name = str(getattr(obj, 'name', '') or '')
        marker = '_Cloth_Collision_'
        if marker in name:
            suffix = name.split(marker, 1)[1].split('.', 1)[0]
            if suffix.isdigit():
                return int(suffix)
    except Exception:
        pass
    return None


def _apply_visible_collision_points_to_base_positions(root, armature, setup: _FDSetup, base_positions: list[mathutils.Vector], depsgraph=None) -> None:
    if not base_positions:
        return
    try:
        armature_inv = _safe_inverted(_world_matrix(armature, depsgraph))
    except Exception:
        armature_inv = mathutils.Matrix.Identity(4)
    for child in list(getattr(root, 'children', []) or []):
        if _object_type(child) not in {'ClothCollision', 'ClothPoint'}:
            continue
        point_index = _visible_collision_point_for_helper(child, setup)
        if point_index is None or point_index < 0 or point_index >= len(base_positions):
            continue
        try:
            if int(setup.points[int(point_index)].flags) != 1:
                continue
        except Exception:
            continue
        try:
            world_pos = _world_matrix(child, depsgraph).translation.copy()
            base_positions[int(point_index)] = armature_inv @ world_pos
        except Exception:
            pass


def _movable_point_indices(setup: _FDSetup) -> set[int]:
    result: set[int] = set()
    for rule in setup.move_rules:
        start = int(rule.point)
        count = max(1, int(rule.count))
        for index in range(start, start + count):
            if 0 <= index < len(setup.points):
                result.add(index)
    if not result:
        for index, point in enumerate(setup.points):
            if int(point.flags) == 4:
                result.add(index)
    return result


def _is_rule_fixed(setup: _FDSetup, index: int, rule_flag: int, movable: set[int]) -> bool:
    if index < 0 or index >= len(setup.points):
        return True
    if int(rule_flag) != 0:
        return True
    if index not in movable:
        return True
    return int(setup.points[index].flags) != 4


def _apply_distance_rule(positions: list[mathutils.Vector], setup: _FDSetup, rule: _FDDistanceRule, movable: set[int], stiffness: float = 1.0) -> None:
    p0 = int(rule.p0)
    p1 = int(rule.p1)
    if p0 < 0 or p1 < 0 or p0 >= len(positions) or p1 >= len(positions):
        return
    if p0 == p1:
        return

    delta = positions[p1] - positions[p0]
    dist = float(delta.length)
    if dist <= _EPSILON:
        return

    min_len = math.sqrt(max(0.0, float(rule.min_dist_sq)))
    max_len = math.sqrt(max(0.0, float(rule.max_dist_sq))) if float(rule.max_dist_sq) > 0.0 and float(rule.max_dist_sq) < _MAX_DISTANCE_RULE else None
    target: float | None = None
    if min_len > 0.0 and dist < min_len:
        target = min_len
    elif max_len is not None and dist > max_len:
        target = max_len
    if target is None:
        return

    fixed0 = _is_rule_fixed(setup, p0, rule.f0, movable)
    fixed1 = _is_rule_fixed(setup, p1, rule.f1, movable)
    w0 = 0.0 if fixed0 else 1.0
    w1 = 0.0 if fixed1 else 1.0
    weight_sum = w0 + w1
    if weight_sum <= _EPSILON:
        return

    direction = delta / dist
    correction = direction * ((dist - target) * max(0.0, min(1.0, stiffness)))
    if w0 > 0.0:
        positions[p0] += correction * (w0 / weight_sum)
    if w1 > 0.0:
        positions[p1] -= correction * (w1 / weight_sum)


def _distance_rule_cache_signature(setup: _FDSetup, movable: set[int]) -> tuple:
    return (
        len(setup.points),
        tuple(sorted(int(index) for index in movable)),
        tuple(
            (
                int(rule.p0),
                int(rule.p1),
                int(rule.f0),
                int(rule.f1),
                _cache_float(rule.min_dist_sq, 6),
                _cache_float(rule.max_dist_sq, 6),
            )
            for rule in setup.dist_rules
        ),
    )


def _prepared_distance_rules_for_state(state: _RootSimulationState, setup: _FDSetup, movable: set[int]) -> list[tuple]:
    signature = _distance_rule_cache_signature(setup, movable)
    if state.prepared_distance_signature == signature:
        return state.prepared_distance_rules

    prepared: list[tuple] = []
    point_count = len(setup.points)
    for rule in setup.dist_rules:
        p0 = int(rule.p0)
        p1 = int(rule.p1)
        if p0 < 0 or p1 < 0 or p0 >= point_count or p1 >= point_count or p0 == p1:
            continue
        max_dist_sq = float(rule.max_dist_sq)
        prepared.append((
            p0,
            p1,
            _is_rule_fixed(setup, p0, int(rule.f0), movable),
            _is_rule_fixed(setup, p1, int(rule.f1), movable),
            math.sqrt(max(0.0, float(rule.min_dist_sq))),
            math.sqrt(max(0.0, max_dist_sq)) if max_dist_sq > 0.0 and max_dist_sq < _MAX_DISTANCE_RULE else -1.0,
        ))

    state.prepared_distance_signature = signature
    state.prepared_distance_rules = prepared
    return prepared


def _apply_prepared_distance_rule(positions: list[mathutils.Vector], rule: tuple, stiffness: float = 1.0) -> None:
    p0, p1, fixed0, fixed1, min_len, max_len = rule
    if p0 < 0 or p1 < 0 or p0 >= len(positions) or p1 >= len(positions):
        return

    delta = positions[p1] - positions[p0]
    dist = float(delta.length)
    if dist <= _EPSILON:
        return

    target: float | None = None
    if min_len > 0.0 and dist < min_len:
        target = min_len
    elif max_len >= 0.0 and dist > max_len:
        target = max_len
    if target is None:
        return

    w0 = 0.0 if fixed0 else 1.0
    w1 = 0.0 if fixed1 else 1.0
    weight_sum = w0 + w1
    if weight_sum <= _EPSILON:
        return

    correction = (delta / dist) * ((dist - target) * max(0.0, min(1.0, stiffness)))
    if w0 > 0.0:
        positions[p0] += correction * (w0 / weight_sum)
    if w1 > 0.0:
        positions[p1] -= correction * (w1 / weight_sum)


def _motion_limits_for_state(state: _RootSimulationState, setup: _FDSetup) -> list[float]:
    signature = (
        len(setup.points),
        tuple(
            (int(rule.p0), int(rule.p1), _cache_float(rule.min_dist_sq, 6), _cache_float(rule.max_dist_sq, 6))
            for rule in setup.dist_rules
        ),
    )
    if state.motion_limits_signature == signature and len(state.motion_limits) == len(setup.points):
        return state.motion_limits
    limits = _motion_limits_for_setup(setup)
    state.motion_limits_signature = signature
    state.motion_limits = limits
    return limits


def _closest_point_on_segment(point: mathutils.Vector, start: mathutils.Vector, end: mathutils.Vector) -> mathutils.Vector:
    segment = end - start
    length_sq = float(segment.length_squared)
    if length_sq <= _EPSILON:
        return start.copy()
    t = max(0.0, min(1.0, float((point - start).dot(segment) / length_sq)))
    return start + segment * t




def _capsule_endpoint_info_from_name(root, obj) -> tuple[int, str] | None:
    try:
        name = str(getattr(obj, 'name', '') or '')
        if '.' in name:
            name = name.split('.', 1)[0]
        root_name = str(getattr(root, 'name', '') or '')
        bases = [root_name]
        if root_name.endswith('_Cloth'):
            bases.append(root_name[:-len('_Cloth')])
        for base in bases:
            if not base:
                continue
            prefix = f'{base}_Cloth_Capsule_'
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            parts = suffix.split('_')
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            endpoint = parts[1].upper()
            if endpoint in {'A', 'B'}:
                return int(parts[0]), endpoint
    except Exception:
        pass
    return None

def _push_point_from_sphere(position: mathutils.Vector, center: mathutils.Vector, radius: float) -> mathutils.Vector:
    radius = max(0.0, float(radius))
    if radius <= 0.0:
        return position
    delta = position - center
    dist = float(delta.length)
    if dist >= radius:
        return position
    if dist <= _EPSILON:
        return center + mathutils.Vector((0.0, 0.0, radius))
    return center + (delta / dist) * radius


def _collect_colliders_armature(root, armature, depsgraph=None) -> tuple[list[tuple[mathutils.Vector, float]], list[tuple[mathutils.Vector, mathutils.Vector, float]]]:
    armature_inv = _safe_inverted(_world_matrix(armature, depsgraph))
    spheres: list[tuple[mathutils.Vector, float]] = []
    capsule_endpoints: dict[int, dict[str, tuple[mathutils.Vector, float]]] = {}
    for child in list(getattr(root, 'children', []) or []):
        if not _is_collision_enabled(child):
            continue
        trlau_type = _object_type(child)
        if trlau_type in {'ClothCollision', 'ClothPoint'}:
            try:
                world_pos = _world_matrix(child, depsgraph).translation.copy()
                spheres.append((armature_inv @ world_pos, _world_radius(child, depsgraph)))
            except Exception:
                pass
        elif trlau_type == 'ClothCapsuleEndpoint':
            endpoint_info = _capsule_endpoint_info_from_name(root, child)
            if endpoint_info is None:
                continue
            index, endpoint = endpoint_info
            try:
                world_pos = _world_matrix(child, depsgraph).translation.copy()
                capsule_endpoints.setdefault(index, {})[endpoint] = (armature_inv @ world_pos, _world_radius(child, depsgraph))
            except Exception:
                pass
    capsules: list[tuple[mathutils.Vector, mathutils.Vector, float]] = []
    for endpoints in capsule_endpoints.values():
        if 'A' not in endpoints or 'B' not in endpoints:
            continue
        a, ra = endpoints['A']
        b, rb = endpoints['B']
        capsules.append((a, b, max(0.0, max(float(ra), float(rb)))))
    return spheres, capsules


def _apply_external_collisions(positions: list[mathutils.Vector], movable, spheres, capsules) -> None:
    if not movable or (not spheres and not capsules):
        return
    for index in movable:
        if index < 0 or index >= len(positions):
            continue
        position = positions[index]
        for center, radius in spheres:
            position = _push_point_from_sphere(position, center, radius)
        for start, end, radius in capsules:
            closest = _closest_point_on_segment(position, start, end)
            position = _push_point_from_sphere(position, closest, radius)
        positions[index] = position


def _orthogonal_axis_for_forward(forward: mathutils.Vector) -> mathutils.Vector:
    for candidate in (
        mathutils.Vector((1.0, 0.0, 0.0)),
        mathutils.Vector((0.0, 0.0, 1.0)),
        mathutils.Vector((0.0, 1.0, 0.0)),
    ):
        projected = candidate - forward * candidate.dot(forward)
        if projected.length > _EPSILON:
            return projected.normalized()
    return mathutils.Vector((1.0, 0.0, 0.0))


def _aligned_pose_matrix(
    head_armature: mathutils.Vector,
    tail_armature: mathutils.Vector,
    preferred_x_armature: mathutils.Vector | None = None,
) -> mathutils.Matrix | None:
    forward = tail_armature - head_armature
    if forward.length <= _EPSILON:
        return None
    forward.normalize()

    if preferred_x_armature is not None and preferred_x_armature.length > _EPSILON:
        x_axis = preferred_x_armature - forward * preferred_x_armature.dot(forward)
        if x_axis.length <= _EPSILON:
            x_axis = _orthogonal_axis_for_forward(forward)
        else:
            x_axis.normalize()
    else:
        x_axis = _orthogonal_axis_for_forward(forward)

    z_axis = x_axis.cross(forward)
    if z_axis.length <= _EPSILON:
        z_axis = _orthogonal_axis_for_forward(forward).cross(forward)
    if z_axis.length <= _EPSILON:
        return None
    z_axis.normalize()
    x_axis = forward.cross(z_axis)
    if x_axis.length <= _EPSILON:
        return None
    x_axis.normalize()

    matrix = mathutils.Matrix(((
        x_axis.x, forward.x, z_axis.x, head_armature.x,
    ), (
        x_axis.y, forward.y, z_axis.y, head_armature.y,
    ), (
        x_axis.z, forward.z, z_axis.z, head_armature.z,
    ), (
        0.0, 0.0, 0.0, 1.0,
    )))
    return matrix


def _pose_bone_head_tail(pose_bone) -> tuple[mathutils.Vector, mathutils.Vector]:
    try:
        matrix = pose_bone.matrix.copy()
        head = matrix.translation.copy()
        length = max(0.0001, float(getattr(getattr(pose_bone, 'bone', None), 'length', 0.1) or 0.1))
        tail = matrix @ mathutils.Vector((0.0, length, 0.0))
        return head, tail
    except Exception:
        return mathutils.Vector((0.0, 0.0, 0.0)), mathutils.Vector((0.0, 0.1, 0.0))


def _preferred_x_axis(pose_bone) -> mathutils.Vector:
    try:
        matrix = pose_bone.matrix.copy()
        return (matrix.to_3x3() @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()
    except Exception:
        return mathutils.Vector((1.0, 0.0, 0.0))


def _set_pose_bone_from_matrix(pose_bone, matrix_armature: mathutils.Matrix | None) -> None:
    if matrix_armature is None:
        return
    try:
        pose_bone.matrix = matrix_armature
    except Exception:
        pass


def _matrices_nearly_equal(a: mathutils.Matrix, b: mathutils.Matrix, threshold: float = 1.0e-7) -> bool:
    try:
        rows = min(len(a), len(b))
        for row in range(rows):
            cols = min(len(a[row]), len(b[row]))
            for col in range(cols):
                if abs(float(a[row][col]) - float(b[row][col])) > float(threshold):
                    return False
        return True
    except Exception:
        return False


def _assign_pose_bone_matrix_basis_if_changed(pose_bone, matrix: mathutils.Matrix, threshold: float = 1.0e-7) -> bool:
    try:
        current = pose_bone.matrix_basis.copy()
        if _matrices_nearly_equal(current, matrix, threshold):
            return False
        pose_bone.matrix_basis = matrix
        return True
    except Exception:
        try:
            pose_bone.matrix_basis = matrix
            return True
        except Exception:
            return False


def _axis_local_vector(axis_code: int) -> mathutils.Vector:
    """Return the controlled local axis used by ClothJointMap.axis.

    TRA cloth maps use 3/4/5 for the dominant segment axis (X/Y/Z).  Older
    generated data may contain 0/1/2, so accept those as aliases.
    """
    axis = int(axis_code)
    if axis in {0, 3}:
        return mathutils.Vector((1.0, 0.0, 0.0))
    if axis in {1, 4}:
        return mathutils.Vector((0.0, 1.0, 0.0))
    return mathutils.Vector((0.0, 0.0, 1.0))


def _matrix_with_axis_aligned(
    base_matrix: mathutils.Matrix,
    local_axis: mathutils.Vector,
    target_axis_armature: mathutils.Vector,
) -> mathutils.Matrix | None:
    """Rotate a pose matrix so one authored local axis points at the FD target.

    The previous implementation always treated the Blender bone's +Y shaft as
    the controlled axis.  Imported TRA joint maps often use axis 5 (local Z) or
    axis 3 (local X).  Driving the wrong axis makes skinned cloth vertices fly
    into a cage even when the FD points themselves are stable.
    """
    if target_axis_armature.length <= _EPSILON:
        return None
    target = target_axis_armature.normalized()

    try:
        base_rot = base_matrix.to_3x3()
        rest_axis = base_rot @ local_axis
    except Exception:
        return None
    if rest_axis.length <= _EPSILON:
        return None
    rest_axis.normalize()

    try:
        delta = rest_axis.rotation_difference(target).to_matrix()
    except Exception:
        return None

    matrix = (delta @ base_rot).to_4x4()
    try:
        matrix.translation = base_matrix.translation.copy()
    except Exception:
        pass
    return matrix



def _map_vector_from_positions(
    map_rule: _FDJointMap,
    positions: list[mathutils.Vector],
) -> mathutils.Vector | None:
    """Return the authored map vector in armature space.

    A ClothJointMap does not mean "make Blender +Y/+Z point at this vector".
    It maps an existing game segment direction to a solved FD point direction.
    Using the map's rest vector as the source direction prevents the import rest
    pose from rotating at frame 0 and avoids the large cage-like corruption seen
    when axis 5 was interpreted as Blender +Z.
    """
    center = int(map_rule.center)
    if center < 0 or center >= len(positions):
        return None
    head = positions[center].copy()
    p0, p1, _p2, _p3 = map_rule.points
    candidate: Optional[mathutils.Vector] = None
    if p0 == center and 0 <= p1 < len(positions) and p1 != center:
        candidate = positions[p1] - head
    elif p1 == center and 0 <= p0 < len(positions) and p0 != center:
        candidate = positions[p0] - head
    elif 0 <= p0 < len(positions) and 0 <= p1 < len(positions) and p0 != p1:
        candidate = positions[p1] - positions[p0]
    if candidate is None or candidate.length <= _EPSILON:
        return None
    return candidate


def _bone_depth(data_bones, bone_name: str, cache: dict[str, int]) -> int:
    if bone_name in cache:
        return cache[bone_name]
    try:
        bone = data_bones.get(bone_name) if data_bones is not None else None
    except Exception:
        bone = None
    if bone is None:
        cache[bone_name] = 0
        return 0
    parent = getattr(bone, 'parent', None)
    parent_name = str(getattr(parent, 'name', '') or '') if parent is not None else ''
    if not parent_name:
        cache[bone_name] = 0
        return 0
    depth = _bone_depth(data_bones, parent_name, cache) + 1
    cache[bone_name] = depth
    return depth


def _no_basis_matrix_for_pose_bone(
    armature,
    pose_bone,
    pose_source_armature=None,
    mapped_parent_matrices: dict[str, mathutils.Matrix] | None = None,
) -> mathutils.Matrix | None:
    """Current armature-space matrix for the bone with its own basis removed.

    Setting pose_bone.matrix directly solves for an absolute child transform and
    can cancel the parent transform, which causes skinned cloth chains to shear
    or explode.  FD needs local segment rotations: parent transforms should carry
    child pivots naturally through the Blender hierarchy.
    """
    try:
        bone = pose_bone.bone
        bone_rest = bone.matrix_local.copy()
        parent = getattr(bone, 'parent', None)
        if parent is None:
            return bone_rest
        parent_name = str(getattr(parent, 'name', '') or '')
        if mapped_parent_matrices is not None and parent_name in mapped_parent_matrices:
            parent_matrix = mapped_parent_matrices[parent_name].copy()
        else:
            pose_owner = pose_source_armature if pose_source_armature is not None else armature
            pose_bones = getattr(getattr(pose_owner, 'pose', None), 'bones', {})
            parent_pose = pose_bones.get(parent.name) if pose_bones is not None else None
            if parent_pose is None:
                pose_bones = getattr(getattr(armature, 'pose', None), 'bones', {})
                parent_pose = pose_bones.get(parent.name) if pose_bones is not None else None
            if parent_pose is None:
                return bone_rest
            parent_matrix = parent_pose.matrix.copy()
        return parent_matrix @ _safe_inverted(parent.matrix_local) @ bone_rest
    except Exception:
        return None


def _set_pose_bone_local_rotation(
    armature,
    pose_bone,
    source_direction_local: mathutils.Vector,
    target_direction_armature: mathutils.Vector,
    *,
    blend: float = 1.0,
    max_angle_radians: float = math.radians(82.0),
    pose_source_armature=None,
    mapped_parent_matrices: dict[str, mathutils.Matrix] | None = None,
) -> mathutils.Matrix | None:
    no_basis = _no_basis_matrix_for_pose_bone(armature, pose_bone, pose_source_armature, mapped_parent_matrices)
    if no_basis is None:
        return
    if source_direction_local.length <= _EPSILON or target_direction_armature.length <= _EPSILON:
        return

    source_local = source_direction_local.normalized()
    target = target_direction_armature.normalized()
    try:
        current_source = no_basis.to_3x3() @ source_local
    except Exception:
        return
    if current_source.length <= _EPSILON:
        return
    current_source.normalize()

    try:
        delta_quat = current_source.rotation_difference(target)
    except Exception:
        return

    try:
        angle = float(delta_quat.angle)
    except Exception:
        angle = 0.0
    if not math.isfinite(angle):
        return
    if angle > float(max_angle_radians):
        try:
            axis = delta_quat.axis
            if axis.length <= _EPSILON:
                return
            delta_quat = mathutils.Quaternion(axis, float(max_angle_radians))
        except Exception:
            return

    try:
        desired_abs = (delta_quat.to_matrix().to_4x4() @ no_basis)
        desired_basis = _safe_inverted(no_basis) @ desired_abs
        desired_quat = desired_basis.to_quaternion()
    except Exception:
        return

    try:
        current_quat = pose_bone.matrix_basis.to_quaternion()
        blend = max(0.0, min(1.0, float(blend)))
        if blend < 1.0:
            desired_quat = current_quat.slerp(desired_quat, blend)
        basis = desired_quat.to_matrix().to_4x4()
        basis.translation = mathutils.Vector((0.0, 0.0, 0.0))
        changed = _assign_pose_bone_matrix_basis_if_changed(pose_bone, basis)
        if changed:
            try:
                pose_bone.location = (0.0, 0.0, 0.0)
            except Exception:
                pass
        try:
            return no_basis @ basis
        except Exception:
            return None
    except Exception:
        return None


def _clear_pose_bone_local_transform(pose_bone) -> None:
    try:
        changed = _assign_pose_bone_matrix_basis_if_changed(pose_bone, mathutils.Matrix.Identity(4))
        if changed:
            try:
                pose_bone.location = (0.0, 0.0, 0.0)
            except Exception:
                pass
    except Exception:
        pass


def _apply_joint_maps(
    armature,
    setup: _FDSetup,
    positions: list[mathutils.Vector],
    rest_positions: list[mathutils.Vector],
    cloth_bone_names: set[str],
    *,
    reset: bool = False,
    depsgraph=None,
) -> None:
    pose_bones = getattr(getattr(armature, 'pose', None), 'bones', {})
    data_bones = getattr(getattr(armature, 'data', None), 'bones', {})
    base_cache: dict[str, mathutils.Matrix] = {}
    depth_cache: dict[str, int] = {}
    mapped_bones: set[str] = set()
    pose_source = _evaluated_object(armature, depsgraph)
    applied_matrices: dict[str, mathutils.Matrix] = {}

    def sort_key(item: _FDJointMap):
        bone_name = str(getattr(item, 'bone_name', '') or f'bone_{int(item.segment)}')
        return (_bone_depth(data_bones, bone_name, depth_cache), int(item.joint_order), _bone_sort_key(bone_name))

    # Work from parent to child and write matrix_basis only.  This keeps the
    # original hierarchy intact instead of pinning every cloth pivot in absolute
    # armature space.
    for map_rule in sorted(setup.maps, key=sort_key):
        bone_name = str(getattr(map_rule, 'bone_name', '') or f'bone_{int(map_rule.segment)}')
        pose_bone = pose_bones.get(bone_name) if pose_bones is not None else None
        if pose_bone is None:
            continue
        mapped_bones.add(bone_name)

        rest_vector = _map_vector_from_positions(map_rule, rest_positions)
        target_vector = _map_vector_from_positions(map_rule, positions)
        if rest_vector is None or target_vector is None:
            _clear_pose_bone_local_transform(pose_bone)
            continue
        if rest_vector.length <= _EPSILON or target_vector.length <= _EPSILON:
            _clear_pose_bone_local_transform(pose_bone)
            continue

        base_matrix = _base_matrix_for_bone_name(armature, bone_name, cloth_bone_names, base_cache, pose_source)
        rest_local = _safe_inverted(base_matrix.to_3x3().to_4x4()).to_3x3() @ rest_vector
        if rest_local.length <= _EPSILON:
            rest_local = _axis_local_vector(int(map_rule.axis))

        # Imported game maps keep a conservative output clamp. Generated maps are
        # authored directly from the Blender chain and should be able to swing
        # like exported/re-imported cloth.
        is_generated_preview_map = bool(str(getattr(map_rule, 'bone_name', '') or ''))
        max_angle = math.radians(176.0 if is_generated_preview_map else 82.0)

        applied_matrix = _set_pose_bone_local_rotation(
            armature,
            pose_bone,
            rest_local,
            target_vector,
            blend=1.0 if reset else (0.96 if is_generated_preview_map else 0.90),
            max_angle_radians=max_angle,
            pose_source_armature=pose_source,
            mapped_parent_matrices=applied_matrices,
        )
        if applied_matrix is not None:
            applied_matrices[str(bone_name)] = applied_matrix.copy()

    # Clear any cloth bone that is marked but not driven by a map.  Leaving old
    # basis values here is a common cause of persistent corruption after a bad
    # simulation run.
    for bone_name in cloth_bone_names:
        if bone_name in mapped_bones:
            continue
        pose_bone = pose_bones.get(bone_name) if pose_bones is not None else None
        if pose_bone is not None:
            _clear_pose_bone_local_transform(pose_bone)

def _state_point_key(point: _FDPoint) -> tuple:
    try:
        bone_name = str(getattr(point, 'bone_name', '') or '')
    except Exception:
        bone_name = ''
    try:
        if bone_name.startswith(_COLLISION_HELPER_POINT_PREFIX):
            return ('collision', bone_name[len(_COLLISION_HELPER_POINT_PREFIX):])
    except Exception:
        pass
    try:
        offset = tuple(round(float(value), 6) for value in getattr(point, 'offset', (0.0, 0.0, 0.0)))
    except Exception:
        offset = (0.0, 0.0, 0.0)
    try:
        flags = int(getattr(point, 'flags', 0))
    except Exception:
        flags = 0
    try:
        joint_order = int(getattr(point, 'joint_order', 0))
    except Exception:
        joint_order = 0
    try:
        up_to = int(getattr(point, 'up_to', 0))
    except Exception:
        up_to = 0
    if bone_name:
        return ('bone', bone_name, flags, joint_order, up_to, offset)
    try:
        segment = int(getattr(point, 'segment', 0))
    except Exception:
        segment = 0
    return ('segment', segment, flags, joint_order, up_to, offset)


def _setup_point_keys(setup: _FDSetup) -> list[tuple]:
    return [_state_point_key(point) for point in list(getattr(setup, 'points', []) or [])]


def _has_valid_state(state: _RootSimulationState, setup: _FDSetup) -> bool:
    count = len(setup.points)
    return (
        state.setup_signature == setup.signature
        and getattr(state, 'setup_point_keys', []) == _setup_point_keys(setup)
        and len(state.positions) == count
        and len(state.previous_positions) == count
        and len(state.rest_positions) == count
        and len(state.output_positions) == count
        and len(state.last_base_positions) == count
    )


def _copy_vector_or_base(values: list[mathutils.Vector], index: int | None, base: mathutils.Vector) -> mathutils.Vector:
    if index is not None and 0 <= int(index) < len(values):
        try:
            value = values[int(index)]
            if _is_finite_vector(value):
                return value.copy()
        except Exception:
            pass
    return base.copy()


def _migrate_state_to_setup(state: _RootSimulationState, setup: _FDSetup, base_positions: list[mathutils.Vector]) -> bool:
    old_keys = list(getattr(state, 'setup_point_keys', []) or [])
    new_keys = _setup_point_keys(setup)
    if not old_keys or not new_keys:
        return False
    if len(base_positions) != len(new_keys):
        return False

    key_to_old_indices: dict[tuple, list[int]] = {}
    for old_index, key in enumerate(old_keys):
        key_to_old_indices.setdefault(key, []).append(int(old_index))

    old_index_for_new: list[int | None] = []
    carried = 0
    for key in new_keys:
        indices = key_to_old_indices.get(key)
        if indices:
            old_index = int(indices.pop(0))
            old_index_for_new.append(old_index)
            carried += 1
        else:
            old_index_for_new.append(None)

    if carried <= 0:
        return False

    state.setup_signature = setup.signature
    state.setup_point_keys = list(new_keys)
    state.last_live_collision_rules_signature = getattr(setup, 'live_collision_rules_signature', '')
    state.positions = [_copy_vector_or_base(state.positions, old_index, base_positions[new_index]) for new_index, old_index in enumerate(old_index_for_new)]
    state.previous_positions = [_copy_vector_or_base(state.previous_positions, old_index, base_positions[new_index]) for new_index, old_index in enumerate(old_index_for_new)]
    state.rest_positions = [_copy_vector_or_base(state.rest_positions, old_index, base_positions[new_index]) for new_index, old_index in enumerate(old_index_for_new)]
    state.output_positions = [_copy_vector_or_base(state.output_positions, old_index, state.positions[new_index]) for new_index, old_index in enumerate(old_index_for_new)]
    state.last_base_positions = [_copy_vector_or_base(state.last_base_positions, old_index, base_positions[new_index]) for new_index, old_index in enumerate(old_index_for_new)]
    return True


def _initialize_state(state: _RootSimulationState, setup: _FDSetup, base_positions: list[mathutils.Vector]) -> None:
    state.setup_signature = setup.signature
    state.setup_point_keys = _setup_point_keys(setup)
    state.last_live_collision_rules_signature = getattr(setup, 'live_collision_rules_signature', '')
    state.positions = [p.copy() for p in base_positions]
    state.previous_positions = [p.copy() for p in base_positions]
    state.rest_positions = [p.copy() for p in base_positions]
    state.output_positions = [p.copy() for p in base_positions]
    state.last_base_positions = [p.copy() for p in base_positions]


def _sync_state_to_animated_pose(
    state: _RootSimulationState,
    setup: _FDSetup,
    base_positions: list[mathutils.Vector],
    movable: set[int],
    *,
    follow_strength: float = 0.92,
) -> None:
    follow_strength = max(0.0, min(1.0, float(follow_strength)))
    for index, base in enumerate(base_positions):
        if index >= len(state.positions):
            continue
        if index in movable and int(setup.points[index].flags) == 4:
            delta = base - state.rest_positions[index]
            # Do not move simulated points all the way to the new animated rest
            # pose.  Full shifting makes interactive pose edits look twitchy and
            # removes the small inertial lag expected from FD cloth.
            shifted = delta * follow_strength
            state.positions[index] += shifted
            state.previous_positions[index] += shifted
            state.rest_positions[index] = base.copy()
        else:
            state.positions[index] = base.copy()
            state.previous_positions[index] = base.copy()
            state.rest_positions[index] = base.copy()
            if index < len(state.output_positions):
                state.output_positions[index] = base.copy()
    state.last_base_positions = [p.copy() for p in base_positions]


def _is_finite_vector(value: mathutils.Vector) -> bool:
    try:
        return all(math.isfinite(float(component)) for component in value)
    except Exception:
        return False



def _base_positions_changed(state: _RootSimulationState, base_positions: list[mathutils.Vector], threshold: float = 1.0e-5) -> bool:
    if len(state.last_base_positions) != len(base_positions):
        return True
    threshold_sq = float(threshold) * float(threshold)
    for old, new in zip(state.last_base_positions, base_positions):
        try:
            if (new - old).length_squared > threshold_sq:
                return True
        except Exception:
            return True
    return False


def _update_smoothed_output_positions(
    state: _RootSimulationState,
    setup: _FDSetup,
    base_positions: list[mathutils.Vector],
    movable: set[int],
    *,
    reset: bool = False,
    blend: float = 0.62,
) -> list[mathutils.Vector]:
    count = len(state.positions)
    if len(state.output_positions) != count:
        state.output_positions = [p.copy() for p in state.positions]
    blend = max(0.0, min(1.0, float(blend)))
    for index in range(count):
        target = state.positions[index].copy()
        if index not in movable or int(setup.points[index].flags) != 4:
            try:
                target = base_positions[index].copy()
            except Exception:
                pass
        if reset or index >= len(state.output_positions):
            state.output_positions[index] = target.copy()
        elif index in movable and int(setup.points[index].flags) == 4:
            try:
                state.output_positions[index] = state.output_positions[index].lerp(target, blend)
            except Exception:
                state.output_positions[index] = target.copy()
        else:
            state.output_positions[index] = target.copy()
    return state.output_positions


def _motion_limits_for_setup(setup: _FDSetup) -> list[float]:
    limits = [64.0 for _ in setup.points]
    for rule in setup.dist_rules:
        values: list[float] = []
        for squared in (float(rule.min_dist_sq), float(rule.max_dist_sq)):
            if squared > 0.0 and squared < _MAX_DISTANCE_RULE:
                values.append(math.sqrt(max(0.0, squared)))
        if not values:
            continue
        link_len = max(values)
        for point_index in (int(rule.p0), int(rule.p1)):
            if 0 <= point_index < len(limits):
                limits[point_index] = max(limits[point_index], link_len * 3.0)
    return [max(32.0, min(512.0, float(value))) for value in limits]


def _stabilize_positions(
    positions: list[mathutils.Vector],
    previous_positions: list[mathutils.Vector],
    base_positions: list[mathutils.Vector],
    movable: set[int],
    motion_limits: list[float],
    movable_indices=None,
) -> None:
    indices = movable_indices if movable_indices is not None else range(len(positions))
    for index in indices:
        if index < 0 or index >= len(positions) or index >= len(base_positions):
            continue
        if not _is_finite_vector(positions[index]) or not _is_finite_vector(previous_positions[index]):
            positions[index] = base_positions[index].copy()
            previous_positions[index] = base_positions[index].copy()
            continue
        if movable_indices is None and index not in movable:
            continue
        limit = motion_limits[index] if index < len(motion_limits) else 128.0
        from_base = positions[index] - base_positions[index]
        if from_base.length > limit:
            positions[index] = base_positions[index] + from_base.normalized() * limit
        step = positions[index] - previous_positions[index]
        max_step = max(8.0, min(64.0, limit * 0.25))
        if step.length > max_step:
            positions[index] = previous_positions[index] + step.normalized() * max_step


def _solve_fd_step(
    setup: _FDSetup,
    positions: list[mathutils.Vector],
    previous_positions: list[mathutils.Vector],
    base_positions: list[mathutils.Vector],
    movable: set[int],
    prepared_distance_rules: list[tuple],
    gravity: mathutils.Vector,
    drag: float,
    frame_delta: float,
    spheres,
    capsules,
    motion_limits: list[float],
    *,
    fast_preview: bool = False,
) -> tuple[list[mathutils.Vector], list[mathutils.Vector]]:
    # Full-quality playback still uses the original 6x18 projection budget.
    # Interactive same-frame pose updates use a smaller budget; those updates arrive
    # at timer/depsgraph frequency and are immediately corrected on the next frame.
    if fast_preview:
        substeps = 3
        iterations = 10
    else:
        substeps = 6
        iterations = 18
    # TRA's values are authored in game-frame units, not Blender seconds. Keep
    # frame_delta in frames so gravity=10 has roughly the same order of magnitude
    # as the imported point distances.
    step_dt = max(0.0, min(1.0, float(frame_delta))) / max(1, substeps)
    drag_scale = max(0.0, min(1.0, 1.0 - float(drag))) ** (1.0 / max(1, substeps))
    gravity_step = gravity * (step_dt * step_dt)

    current = [p.copy() for p in positions]
    previous = [p.copy() for p in previous_positions]
    movable_indices = [
        int(index)
        for index in movable
        if 0 <= int(index) < len(current) and int(setup.points[int(index)].flags) == 4
    ]
    movable_index_set = set(movable_indices)
    fixed_indices = [index for index in range(len(current)) if index not in movable_index_set]
    has_external_colliders = bool(spheres or capsules)

    def pin_fixed_points(*arrays) -> None:
        for index in fixed_indices:
            if index >= len(base_positions):
                continue
            base = base_positions[index]
            for values in arrays:
                values[index] = base.copy()

    for _substep in range(substeps):
        for index in movable_indices:
            position = current[index].copy()
            velocity = (position - previous[index]) * drag_scale
            previous[index] = position.copy()
            current[index] = position + velocity + gravity_step
        pin_fixed_points(current, previous)
        _stabilize_positions(current, previous, base_positions, movable, motion_limits, movable_indices)

        # Keep non-movable points aligned to the current animated pose before and
        # during constraint projection. This is the game-style meaning of pinned:
        # it is an animated attachment point, while mapped bones can still rotate
        # because their other points are dynamic.
        pin_fixed_points(current)

        for _iteration in range(iterations):
            for rule in prepared_distance_rules:
                _apply_prepared_distance_rule(current, rule, stiffness=1.0)
            if has_external_colliders:
                _apply_external_collisions(current, movable_indices, spheres, capsules)
                # Keep the Verlet previous position out of colliders as well.  If
                # only the current point is projected out, the next frame sees the
                # projection as velocity and the cloth bounces/jitters vertically.
                _apply_external_collisions(previous, movable_indices, spheres, capsules)
            _stabilize_positions(current, previous, base_positions, movable, motion_limits, movable_indices)
            pin_fixed_points(current, previous)

    return current, previous


def _simulate_root(scene, root, depsgraph=None, *, interactive: bool = False) -> None:
    _cleanup_stale_cloth_root_props(root)
    armature = getattr(root, 'parent', None)
    if getattr(armature, 'type', None) != 'ARMATURE':
        return

    root_pointer = int(root.as_pointer())
    state = _states_by_root_pointer.setdefault(root_pointer, _RootSimulationState())
    cloth_bones = _collect_cloth_bones(armature)
    setup = _load_setup_cached(root, armature, cloth_bones, state)
    if setup is None or not setup.points:
        return

    current_frame = _scene_frame(scene)
    last_frame = state.last_frame
    reset = False
    same_frame = False
    if last_frame is None:
        reset = True
        frame_delta = 1.0
    else:
        frame_delta = current_frame - float(last_frame)
        if abs(frame_delta) <= 1.0e-5:
            # Blender can call frame_change_post more than once for the same
            # timeline frame while playback is active.  Treating that as a reset
            # makes cloth alternate between rest and simulated poses.
            same_frame = True
            frame_delta = 0.0
        elif frame_delta < 0.0 or frame_delta > 2.0:
            # Timeline loops, seeks, and dropped-frame jumps should restart the
            # simulation.  A duplicate evaluation of the current frame should not.
            reset = True
            frame_delta = 1.0
    state.last_frame = current_frame

    base_positions = _point_base_positions(armature, setup, set(cloth_bones.keys()), depsgraph)
    if len(base_positions) != len(setup.points):
        return
    _apply_visible_collision_points_to_base_positions(root, armature, setup, base_positions, depsgraph)
    movable = _movable_point_indices(setup)
    state_valid = _has_valid_state(state, setup)
    live_collision_rules_signature = getattr(setup, 'live_collision_rules_signature', '')
    live_collision_rules_changed = state_valid and state.last_live_collision_rules_signature != live_collision_rules_signature
    base_changed = state_valid and _base_positions_changed(state, base_positions)
    if reset:
        _initialize_state(state, setup, base_positions)
        base_changed = False
    elif not state_valid:
        if not _migrate_state_to_setup(state, setup, base_positions):
            _initialize_state(state, setup, base_positions)
        base_changed = False
    elif not same_frame:
        _sync_state_to_animated_pose(state, setup, base_positions, movable, follow_strength=0.92)
    elif base_changed:
        # Same frame, different pose: this happens while manipulating the model
        # in Pose Mode.  Step the solver gently instead of reusing the previous
        # pose or hard-resetting to rest.
        _sync_state_to_animated_pose(state, setup, base_positions, movable, follow_strength=0.68)
    elif interactive and not live_collision_rules_changed:
        # An unrelated pose-bone transform can still make Blender re-evaluate the
        # whole armature and momentarily display cloth bones from their keyed/rest
        # pose.  Re-apply the last solved cloth pose instead of returning without
        # touching the mapped bones.  Pose-bone writes below are idempotent, so
        # this does not create a self-triggered depsgraph loop when nothing really
        # changed.
        output_positions = state.output_positions if len(state.output_positions) == len(setup.points) else state.positions
        _apply_joint_maps(armature, setup, output_positions, state.rest_positions, set(cloth_bones.keys()), reset=True, depsgraph=depsgraph)
        return

    gravity_value = preview_gravity(scene)
    drag_value = preview_drag(scene)
    # The generated setup uses the game-authored/default gravity. Blender's live
    # preview needs the same value divided by 10 to match in-game behavior.
    runtime_gravity_value = gravity_value * _GAME_TO_BLENDER_RUNTIME_GRAVITY_SCALE
    gravity = mathutils.Vector((0.0, 0.0, -runtime_gravity_value))
    spheres, capsules = _collect_colliders_armature(root, armature, depsgraph)
    prepared_distance_rules = _prepared_distance_rules_for_state(state, setup, movable)
    motion_limits = _motion_limits_for_state(state, setup)

    if same_frame and _has_valid_state(state, setup) and not base_changed and not live_collision_rules_changed:
        # Do not integrate or reset on duplicate evaluations of the same frame.
        # Re-applying the last solved pose is enough and avoids visible
        # up/down flicker during playback.
        pass
    elif not reset:
        solve_delta = 0.35 if same_frame and (base_changed or live_collision_rules_changed) else frame_delta
        state.positions, state.previous_positions = _solve_fd_step(
            setup,
            state.positions,
            state.previous_positions,
            base_positions,
            movable,
            prepared_distance_rules,
            gravity,
            drag_value,
            solve_delta,
            spheres,
            capsules,
            motion_limits,
            fast_preview=bool(same_frame and (base_changed or live_collision_rules_changed)),
        )
    else:
        # Even on the first frame, project constraints once so imported FD rules
        # produce an authored/rest pose instead of a tail-only approximation.
        movable_indices = [
            int(index)
            for index in movable
            if 0 <= int(index) < len(state.positions) and int(setup.points[int(index)].flags) == 4
        ]
        has_external_colliders = bool(spheres or capsules)
        for _ in range(8):
            for rule in prepared_distance_rules:
                _apply_prepared_distance_rule(state.positions, rule, stiffness=1.0)
            if has_external_colliders:
                _apply_external_collisions(state.positions, movable_indices, spheres, capsules)
                _apply_external_collisions(state.previous_positions, movable_indices, spheres, capsules)
            _stabilize_positions(state.positions, state.previous_positions, base_positions, movable, motion_limits, movable_indices)

    output_blend = 1.0 if reset else (0.48 if same_frame and base_changed else 0.62)
    output_positions = _update_smoothed_output_positions(
        state,
        setup,
        base_positions,
        movable,
        reset=reset,
        blend=output_blend,
    )
    state.last_base_positions = [p.copy() for p in base_positions]
    state.last_live_collision_rules_signature = live_collision_rules_signature
    _apply_joint_maps(armature, setup, output_positions, state.rest_positions, set(cloth_bones.keys()), reset=reset, depsgraph=depsgraph)
    try:
        armature.update_tag(refresh={'OBJECT', 'DATA'})
    except Exception:
        try:
            armature.update_tag()
        except Exception:
            pass


def simulate_scene(scene, depsgraph=None, *, interactive: bool = False, require_preview_enabled: bool = False) -> None:
    roots = _cloth_roots_for_scene(scene)
    live_pointers = set()
    for root in roots:
        try:
            if require_preview_enabled and not _is_preview_enabled(root):
                continue
            live_pointers.add(int(root.as_pointer()))
            _simulate_root(scene, root, depsgraph, interactive=interactive)
        except Exception:
            # Timeline handlers should not interrupt playback. Broken or partial
            # cloth authoring data is skipped until the user fixes it.
            pass

    for stale_pointer in [pointer for pointer in _states_by_root_pointer.keys() if pointer not in live_pointers]:
        try:
            del _states_by_root_pointer[stale_pointer]
        except Exception:
            pass


def reset_simulation_state() -> None:
    _states_by_root_pointer.clear()



def _active_pose_mode_needs_cloth_refresh(scene) -> bool:
    try:
        if str(getattr(bpy.context, 'mode', '') or '') != 'POSE':
            return False
    except Exception:
        return False
    try:
        active = getattr(bpy.context, 'active_object', None)
        if getattr(active, 'type', None) != 'ARMATURE':
            return False
    except Exception:
        return False
    try:
        for child in list(getattr(active, 'children', []) or []):
            if _is_cloth_root(child):
                return True
    except Exception:
        pass
    return False


def _trlau_cloth_pose_mode_timer():
    # Cloth preview is playback-gated by the Preview Cloth button.  Keep this
    # function as a safe no-op for sessions that may still have the timer
    # registered from a previous addon version.
    return 0.25


@persistent
def _trlau_cloth_frame_change_post(scene, depsgraph=None) -> None:
    global _is_in_cloth_simulation
    if not _is_animation_playing() or _is_in_cloth_simulation:
        return
    _is_in_cloth_simulation = True
    try:
        simulate_scene(scene, depsgraph, interactive=False, require_preview_enabled=True)
    finally:
        _is_in_cloth_simulation = False


@persistent
def _trlau_cloth_depsgraph_update_post(scene, depsgraph=None) -> None:
    # Preview Cloth now runs only from frame_change_post while the timeline plays.
    # This avoids cloth bones fighting normal pose editing between playback passes.
    return


@persistent
def _trlau_cloth_animation_playback_pre(scene, depsgraph=None) -> None:
    reset_simulation_state()


@persistent
def _trlau_cloth_animation_playback_post(scene, depsgraph=None) -> None:
    reset_simulation_state()


def _append_handler(handler_list, handler) -> None:
    if handler not in handler_list:
        handler_list.append(handler)


def _remove_handler(handler_list, handler) -> None:
    if handler in handler_list:
        handler_list.remove(handler)


def register_cloth_simulation() -> None:
    if not hasattr(bpy.types.Scene, 'trlau_cloth_preview_gravity'):
        bpy.types.Scene.trlau_cloth_preview_gravity = bpy.props.FloatProperty(
            name='Gravity',
            description='Runtime cloth preview gravity. This is not stored on Cloth helper objects',
            default=_DEFAULT_PREVIEW_GRAVITY,
            soft_min=0.0,
            soft_max=100.0,
            options={'SKIP_SAVE'},
            update=_cloth_preview_settings_update,
        )
    if not hasattr(bpy.types.Scene, 'trlau_cloth_preview_drag'):
        bpy.types.Scene.trlau_cloth_preview_drag = bpy.props.FloatProperty(
            name='Drag',
            description='Runtime cloth preview drag. This is not stored on Cloth helper objects',
            default=_DEFAULT_PREVIEW_DRAG,
            min=0.0,
            max=1.0,
            soft_min=0.0,
            soft_max=1.0,
            options={'SKIP_SAVE'},
            update=_cloth_preview_settings_update,
        )
    _append_handler(bpy.app.handlers.frame_change_post, _trlau_cloth_frame_change_post)
    try:
        # Old addon builds used a pose-mode timer.  Remove it when present so
        # Preview Cloth remains a timeline-playback feature.
        if bpy.app.timers.is_registered(_trlau_cloth_pose_mode_timer):
            bpy.app.timers.unregister(_trlau_cloth_pose_mode_timer)
    except Exception:
        pass
    try:
        if not bpy.app.timers.is_registered(_trlau_cloth_collision_rule_cleanup_timer):
            bpy.app.timers.register(_trlau_cloth_collision_rule_cleanup_timer, first_interval=0.5, persistent=True)
    except Exception:
        pass
    if hasattr(bpy.app.handlers, 'depsgraph_update_post'):
        _remove_handler(bpy.app.handlers.depsgraph_update_post, _trlau_cloth_depsgraph_update_post)
    if hasattr(bpy.app.handlers, 'animation_playback_pre'):
        _append_handler(bpy.app.handlers.animation_playback_pre, _trlau_cloth_animation_playback_pre)
    if hasattr(bpy.app.handlers, 'animation_playback_post'):
        _append_handler(bpy.app.handlers.animation_playback_post, _trlau_cloth_animation_playback_post)


def unregister_cloth_simulation() -> None:
    for attr in ('trlau_cloth_preview_gravity', 'trlau_cloth_preview_drag'):
        try:
            if hasattr(bpy.types.Scene, attr):
                delattr(bpy.types.Scene, attr)
        except Exception:
            pass
    try:
        if bpy.app.timers.is_registered(_trlau_cloth_pose_mode_timer):
            bpy.app.timers.unregister(_trlau_cloth_pose_mode_timer)
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(_trlau_cloth_collision_rule_cleanup_timer):
            bpy.app.timers.unregister(_trlau_cloth_collision_rule_cleanup_timer)
    except Exception:
        pass
    _remove_handler(bpy.app.handlers.frame_change_post, _trlau_cloth_frame_change_post)
    if hasattr(bpy.app.handlers, 'depsgraph_update_post'):
        _remove_handler(bpy.app.handlers.depsgraph_update_post, _trlau_cloth_depsgraph_update_post)
    if hasattr(bpy.app.handlers, 'animation_playback_pre'):
        _remove_handler(bpy.app.handlers.animation_playback_pre, _trlau_cloth_animation_playback_pre)
    if hasattr(bpy.app.handlers, 'animation_playback_post'):
        _remove_handler(bpy.app.handlers.animation_playback_post, _trlau_cloth_animation_playback_post)
    reset_simulation_state()


__all__ = ['register_cloth_simulation', 'unregister_cloth_simulation', 'reset_simulation_state', 'simulate_scene']
