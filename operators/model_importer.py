from __future__ import annotations

from ..core.object_utils import trlau_object_type, hmarker_index_from_name
from pathlib import Path
import math

import bpy
import mathutils

from ..builders.armature_builder import ArmatureBuilder
from ..builders.mesh_builder import MeshBuildSettings, MeshBuilder
from ..core.blender_mesh_utils import lock_object_rotation
from ..core.log import logger
from ..core.markup_utils import add_markup_bbox_wire
from ..core.material_ui import decode_tpage_flags_for_platform, get_material_platform_name, get_material_tpageid
from ..platforms.common.drm_container import DRMContainerParser
from ..platforms.pc.tr7ae_model import TRModelParser
from ..platforms.pc.tr7ae_nextgen_model import TRNextGenModelParser
from ..platforms.pc.tr8_model import TR8ModelParser
from ..platforms.xbox.tr7ae_model import TRXboxModelParser
from ..platforms.ps2.tr7ae_model import TRPS2ModelParser
from ..platforms.psp.tr7ae_model import TRPSPModelParser
from ..platforms.ps3.tr7ae_model import TRPS3ModelParser
from ..platforms.xbox360.tr7ae_model import TRXbox360ModelParser
from ..platforms.pc.tr7ae_object import TRObjectParser


MARKUP_FLAG_PERCH = 262144
MARKUP_FLAG_WATER = 2147483648
MARKUP_BBOX_FLAGS = MARKUP_FLAG_PERCH | MARKUP_FLAG_WATER


def _add_markup_bbox_wire(curve_data, bbox, markup_position, *, bbox_is_local: bool = False) -> bool:
    return add_markup_bbox_wire(curve_data, bbox, markup_position, bbox_is_local=bbox_is_local)


class ModelImporterMixin:

    @staticmethod
    def _cloth_base_name(source_name: str | None, collection_name: str | None = None) -> str:
        base = str(source_name or collection_name or 'TRLAU').strip() or 'TRLAU'
        for suffix in ('_Armature', '_Model', '_Mesh'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        return base

    @staticmethod
    def _safe_cloth_name_fragment(value: str, default: str = 'Chain') -> str:
        text = str(value or '').strip()
        safe = ''.join(char if char.isalnum() or char in {'_', '-'} else '_' for char in text)
        safe = safe.strip('_')
        return safe or str(default)

    @staticmethod
    def _cloth_bone_segment_from_name(name: str, default: int = 0) -> int:
        value = str(name or '')
        if value.startswith('bone_'):
            try:
                return int(value.split('_', 1)[1])
            except Exception:
                pass
        return int(default)

    @staticmethod
    def _cloth_collision_rules_group_name(base_name: str, chain_root_segment: int | None = None, chain_root_bone_name: str = '') -> str:
        base = f"{str(base_name or 'TRLAU')}_Cloth_CollisionRules"
        if chain_root_segment is None:
            return base
        bone_part = ModelImporterMixin._safe_cloth_name_fragment(chain_root_bone_name, f'B{int(chain_root_segment):03d}')
        return f"{base}_B{int(chain_root_segment):03d}_{bone_part}"

    @staticmethod
    def _cloth_chain_root_for_segment(arm_obj, target_segment: int | None) -> tuple[int | None, str]:
        if arm_obj is None or target_segment is None:
            return target_segment, f'bone_{int(target_segment)}' if target_segment is not None else ''
        bones = getattr(getattr(arm_obj, 'data', None), 'bones', {})
        bone = None
        try:
            bone = bones.get(f'bone_{int(target_segment)}')
        except Exception:
            bone = None
        if bone is None:
            for candidate in list(bones or []):
                try:
                    if ModelImporterMixin._cloth_bone_segment_from_name(getattr(candidate, 'name', ''), -1) == int(target_segment):
                        bone = candidate
                        break
                except Exception:
                    pass
        current = bone
        selected = bone
        visited = set()
        while current is not None:
            name = str(getattr(current, 'name', '') or '')
            if name in visited:
                break
            visited.add(name)
            try:
                enabled = bool(current.get('trlau_cloth_enabled', False))
                pinned = bool(current.get('trlau_cloth_pinned', False))
            except Exception:
                enabled = False
                pinned = False
            if enabled:
                selected = current
                if pinned:
                    return ModelImporterMixin._cloth_bone_segment_from_name(name, int(target_segment)), name
            current = getattr(current, 'parent', None)
        if selected is not None:
            name = str(getattr(selected, 'name', '') or f'bone_{int(target_segment)}')
            return ModelImporterMixin._cloth_bone_segment_from_name(name, int(target_segment)), name
        return int(target_segment), f'bone_{int(target_segment)}'

    def _ensure_cloth_collision_rules_group(self, cloth_obj, collection, base_name: str, *, arm_obj=None, target_segment: int | None = None):
        if cloth_obj is None:
            return None
        chain_root_segment = None
        chain_root_bone_name = ''
        if target_segment is not None:
            chain_root_segment, chain_root_bone_name = self._cloth_chain_root_for_segment(arm_obj, int(target_segment))
        group_name = self._cloth_collision_rules_group_name(base_name, chain_root_segment, chain_root_bone_name)
        group = None
        for child in list(getattr(cloth_obj, 'children', []) or []):
            try:
                if trlau_object_type(child) != 'ClothCollisionRules':
                    continue
                if str(getattr(child, 'name', '') or '') == group_name:
                    group = child
                    break
            except Exception:
                pass
        if group is None:
            group = bpy.data.objects.new(group_name, None)
            self._link_object_to_collection(group, collection)
            group.parent = cloth_obj
            group.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            group.empty_display_type = 'PLAIN_AXES'
            group.empty_display_size = 1.0
            group.show_name = False
            group.hide_select = True
        else:
            self._link_object_to_collection(group, collection)
            try:
                group.parent = cloth_obj
                group.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            except Exception:
                pass
        group.name = group_name
        for key in ('trlau_cloth_chain_root_segment', 'trlau_cloth_chain_root_bone'):
            try:
                if key in group:
                    del group[key]
            except Exception:
                pass
        return group

    @staticmethod
    def _remove_object_tree(obj) -> None:
        if obj is None:
            return
        for child in list(getattr(obj, 'children', []) or []):
            ModelImporterMixin._remove_object_tree(child)
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass

    @staticmethod
    def _idprop_i32(value: int, default: int = 0) -> int:
        try:
            raw = int(value) & 0xFFFFFFFF
        except Exception:
            raw = int(default) & 0xFFFFFFFF
        return raw - 0x100000000 if raw & 0x80000000 else raw

    @staticmethod
    def _cloth_point_key(point_index: int, point) -> str:
        try:
            flags = int(getattr(point, 'flags', 0))
        except Exception:
            flags = 0
        try:
            segment = int(getattr(point, 'segment', 0))
        except Exception:
            segment = 0
        if flags == 1:
            return f'collision:{int(point_index)}'
        return f'bone:{segment}'

    @staticmethod
    def _cloth_rule_point_keys(points) -> list[str]:
        return [ModelImporterMixin._cloth_point_key(index, point) for index, point in enumerate(list(points or []))]

    @staticmethod
    def _make_cloth_material(name: str, color: tuple[float, float, float, float]):
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name)
        try:
            mat.diffuse_color = color
        except Exception:
            pass
        return mat

    @staticmethod
    def _apply_cloth_bone_color(arm_obj, bone_name: str, flags: int) -> None:
        palette = 'THEME01' if int(flags) == 5 else 'THEME08'
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

    @staticmethod
    def _lock_cloth_collision_rotation(obj) -> None:
        lock_object_rotation(obj)

    @staticmethod
    def _clear_imported_cloth_bone_marks(arm_obj) -> None:
        if arm_obj is None:
            return
        keys = (
            'trlau_type', 'trlau_cloth_point_key', 'trlau_cloth_point_index', 'trlau_cloth_point_segment',
            'trlau_cloth_point_flags', 'trlau_cloth_point_jointOrder', 'trlau_cloth_point_upTo',
            'trlau_cloth_chain_id', 'trlau_cloth_chain_order', 'trlau_cloth_map_axis', 'trlau_cloth_pinned',
        )
        for bone_container in (getattr(getattr(arm_obj, 'data', None), 'bones', None), getattr(getattr(arm_obj, 'pose', None), 'bones', None)):
            if bone_container is None:
                continue
            for bone in bone_container:
                if not bool(bone.get('trlau_cloth_enabled', False)):
                    continue
                for key in keys:
                    try:
                        if key in bone:
                            del bone[key]
                    except Exception:
                        pass
                try:
                    bone['trlau_cloth_enabled'] = False
                except Exception:
                    pass

    @staticmethod
    def _set_cloth_bone_props(arm_obj, bone_name: str, props: dict) -> None:
        for bone in (
            getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(bone_name) if arm_obj is not None else None,
            getattr(getattr(arm_obj, 'pose', None), 'bones', {}).get(bone_name) if arm_obj is not None else None,
        ):
            if bone is None:
                continue
            for key, value in props.items():
                try:
                    bone[key] = value
                except Exception:
                    pass

    @staticmethod
    def _create_sphere_mesh(name: str, radius: float = 1.0, segments: int = 16, rings: int = 8):
        vertices = []
        faces = []
        radius = float(radius) if float(radius) > 0 else 1.0
        for ring in range(rings + 1):
            theta = math.pi * ring / rings
            z = math.cos(theta) * radius
            r = math.sin(theta) * radius
            for segment in range(segments):
                phi = (2.0 * math.pi * segment) / segments
                vertices.append((math.cos(phi) * r, math.sin(phi) * r, z))
        for ring in range(rings):
            for segment in range(segments):
                a = ring * segments + segment
                b = ring * segments + ((segment + 1) % segments)
                c = (ring + 1) * segments + ((segment + 1) % segments)
                d = (ring + 1) * segments + segment
                if ring == 0:
                    faces.append((a, c, d))
                elif ring == rings - 1:
                    faces.append((a, b, d))
                else:
                    faces.append((a, b, c, d))
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        try:
            mesh.update()
        except Exception:
            pass
        return mesh

    def _import_cloth_setups(self, collection, arm_obj, cloth_setups, *, source_name: str | None = None):
        setups = list(cloth_setups or [])
        if not setups:
            return None
        setup = setups[0]
        base_name = self._cloth_base_name(source_name, getattr(collection, 'name', None))
        cloth_name = f'{base_name}_Cloth'
        force_unique_names = bool(getattr(self, 'force_unique_import_names', False))
        cloth_obj = None if force_unique_names else bpy.data.objects.get(cloth_name)
        if cloth_obj is not None and getattr(cloth_obj, 'type', None) != 'EMPTY':
            try:
                cloth_obj.name = f'{cloth_name}_Old'
            except Exception:
                pass
            cloth_obj = None
        if cloth_obj is None:
            cloth_obj = bpy.data.objects.new(cloth_name, None)
        self._link_object_to_collection(cloth_obj, collection)
        cloth_obj.empty_display_type = 'PLAIN_AXES'
        cloth_obj.empty_display_size = 32.0
        cloth_obj.location = (0.0, 0.0, 0.0)
        cloth_obj.rotation_mode = 'XYZ'
        cloth_obj.rotation_euler = (0.0, 0.0, 0.0)
        cloth_obj.scale = (1.0, 1.0, 1.0)
        for stale_key in ('trlau_cloth_gravity', 'trlau_cloth_drag', 'trlau_cloth_preview_enabled'):
            try:
                if stale_key in cloth_obj:
                    del cloth_obj[stale_key]
            except Exception:
                pass
        if arm_obj is not None:
            cloth_obj.parent = arm_obj
            cloth_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            try:
                arm_obj.show_in_front = True
            except Exception:
                pass
        else:
            cloth_obj.parent = None

        stale_cloth_helper_types = {'ClothPoint', 'ClothCollision', 'ClothPinCollision', 'ClothCollisionRule', 'ClothCollisionRules', 'ClothCapsuleEndpoint', 'ClothPlaneRule', 'ClothPlaneRulePlane', 'ClothPlaneRuleSelector', 'ClothPlaneRules'}
        for child in list(getattr(cloth_obj, 'children', []) or []):
            try:
                child_type = trlau_object_type(child)
            except Exception:
                child_type = ''
            if child_type in stale_cloth_helper_types:
                self._remove_object_tree(child)

        self._clear_imported_cloth_bone_marks(arm_obj)
        points = list(getattr(setup, 'points', []) or [])
        point_keys = self._cloth_rule_point_keys(points)

        referenced_point_indices: set[int] = set()

        def _mark_referenced_point(value) -> None:
            try:
                index = int(value)
            except Exception:
                return
            if 0 <= index < len(points):
                referenced_point_indices.add(int(index))

        for item in list(getattr(setup, 'maps', []) or []):
            _mark_referenced_point(getattr(item, 'center', -1))
            for value in list(getattr(item, 'points', []) or []):
                _mark_referenced_point(value)
        for attr in ('distRules', 'capsuleRules', 'planeRules', 'sideCollideRules'):
            for item in list(getattr(setup, attr, []) or []):
                for value in list(getattr(item, 'point', []) or []):
                    _mark_referenced_point(value)
        for item in list(getattr(setup, 'handRules', []) or []):
            _mark_referenced_point(getattr(item, 'movePt', -1))
            for value in list(getattr(item, 'point', []) or []):
                _mark_referenced_point(value)
        for attr in ('collideRules', 'sphereCollideRules', 'pointMoveRules'):
            for item in list(getattr(setup, attr, []) or []):
                _mark_referenced_point(getattr(item, 'point', -1))

        for key in (
            'trlau_cloth_dist_rules_json', 'trlau_cloth_capsule_rules_json', 'trlau_cloth_plane_rules_json',
            'trlau_cloth_hand_rules_json', 'trlau_cloth_collide_rules_json', 'trlau_cloth_sphere_collide_rules_json',
            'trlau_cloth_side_collide_rules_json',
        ):
            try:
                if key in cloth_obj:
                    del cloth_obj[key]
            except Exception:
                pass

        collision_material = self._make_cloth_material('TRLAU Cloth Collision', (1.0, 0.75, 0.05, 0.6))
        bone_lookup = getattr(getattr(arm_obj, 'data', None), 'bones', {}) if arm_obj is not None else {}

        collision_radius_candidates_by_key: dict[str, list[float]] = {}
        collision_distance_rules: list[dict] = []
        for source_rule_index, rule in enumerate(list(getattr(setup, 'distRules', []) or [])):
            try:
                p0, p1 = int(rule.point[0]), int(rule.point[1])
                min_dist = max(0.0, float(rule.minDist))
                if min_dist <= 0.0:
                    continue
                radius = math.sqrt(min_dist)
                if 0 <= p0 < len(points) and 0 <= p1 < len(points):
                    f0 = int(getattr(points[p0], 'flags', 0))
                    f1 = int(getattr(points[p1], 'flags', 0))
                    max_dist = max(0.0, float(getattr(rule, 'maxDist', 1000000.0)))
                    collision_like_0 = (f0 == 1) or (f0 == 5 and max_dist > min_dist + 1.0e-4)
                    collision_like_1 = (f1 == 1) or (f1 == 5 and max_dist > min_dist + 1.0e-4)
                    if (collision_like_0 and f1 == 4) or (collision_like_1 and f0 == 4):
                        collision_index = p0 if collision_like_0 else p1
                        target_index = p1 if collision_like_0 else p0
                        if int(getattr(points[collision_index], 'flags', 0)) == 1:
                            collision_radius_candidates_by_key.setdefault(point_keys[collision_index], []).append(radius)
                        collision_distance_rules.append({
                            'rule_index': int(source_rule_index),
                            'p0': int(p0),
                            'p1': int(p1),
                            'f0': int(getattr(rule, 'flags', (0, 0))[0]),
                            'f1': int(getattr(rule, 'flags', (0, 0))[1]),
                            'target_point_index': int(target_index),
                            'target_segment': int(getattr(points[target_index], 'segment', 0)),
                            'collision_point_index': int(collision_index),
                            'collision_point_flags': int(getattr(points[collision_index], 'flags', 0)),
                            'min_dist': float(min_dist),
                            'max_dist': float(max_dist),
                            'radius': float(radius),
                        })
                    else:
                        for point_index in (p0, p1):
                            if int(getattr(points[point_index], 'flags', 0)) == 1:
                                collision_radius_candidates_by_key.setdefault(point_keys[point_index], []).append(radius)
            except Exception:
                pass
        for sphere_rule in list(getattr(setup, 'sphereCollideRules', []) or []):
            try:
                point_index = int(getattr(sphere_rule, 'point', -1))
                radius = max(0.0, float(getattr(sphere_rule, 'radius', 0.0)))
                if radius > 0.0 and 0 <= point_index < len(points) and int(getattr(points[point_index], 'flags', 0)) == 1:
                    collision_radius_candidates_by_key[point_keys[point_index]] = [radius]
            except Exception:
                pass
        collision_radius_by_key: dict[str, float] = {
            key: min(values)
            for key, values in collision_radius_candidates_by_key.items()
            if values
        }
        collision_rule_count_by_point: dict[int, int] = {}
        for rule_info in collision_distance_rules:
            try:
                point_index = int(rule_info.get('collision_point_index', -1))
                collision_rule_count_by_point[point_index] = collision_rule_count_by_point.get(point_index, 0) + 1
            except Exception:
                pass

        capsule_point_indices: set[int] = set()
        capsule_rules = list(getattr(setup, 'capsuleRules', []) or [])
        for rule_index, capsule_rule in enumerate(capsule_rules):
            try:
                p0 = int(getattr(capsule_rule, 'point', (0, 0))[0])
                p1 = int(getattr(capsule_rule, 'point', (0, 0))[1])
                radius_value = max(0.0, float(getattr(capsule_rule, 'radius', 0.0) or 0.0)) or 1.0
            except Exception:
                continue
            if not (0 <= p0 < len(points) and 0 <= p1 < len(points)):
                continue
            capsule_point_indices.update({int(p0), int(p1)})
            for endpoint_label, point_index in (('A', p0), ('B', p1)):
                point = points[int(point_index)]
                segment = int(getattr(point, 'segment', 0))
                endpoint_obj = bpy.data.objects.new(f'{base_name}_Cloth_Capsule_{int(rule_index):03d}_{endpoint_label}', None)
                self._link_object_to_collection(endpoint_obj, collection)
                endpoint_obj.parent = cloth_obj
                endpoint_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                endpoint_obj.empty_display_type = 'SPHERE'
                endpoint_obj.empty_display_size = 1.0
                endpoint_obj.show_name = False
                endpoint_obj.show_in_front = True

                local_position = mathutils.Vector((float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), float(getattr(point, 'z', 0.0))))
                try:
                    bone = bone_lookup.get(f'bone_{segment}') if bone_lookup is not None else None
                    if bone is not None:
                        endpoint_obj.matrix_local = bone.matrix_local @ mathutils.Matrix.Translation(local_position)
                        child_of = endpoint_obj.constraints.new(type='CHILD_OF')
                        child_of.name = 'TRLAU Cloth Capsule Endpoint Bone Follow'
                        child_of.target = arm_obj
                        child_of.subtarget = f'bone_{segment}'
                        child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ cloth_obj.matrix_world
                    else:
                        endpoint_obj.location = local_position
                except Exception:
                    endpoint_obj.location = local_position

                endpoint_obj.scale = (radius_value, radius_value, radius_value)
                self._lock_cloth_collision_rotation(endpoint_obj)

        collision_spheres_by_point_index: dict[int, object] = {}
        for index, point in enumerate(points):
            segment = int(getattr(point, 'segment', 0))
            flags = int(getattr(point, 'flags', 0))
            if flags in {4, 5}:
                point_only_pin = bool(flags == 5 and int(index) not in referenced_point_indices)
                props = {
                    'trlau_cloth_pinned': bool(flags == 5),
                }
                if not point_only_pin:
                    props['trlau_cloth_enabled'] = True
                bone_name = f'bone_{segment}'
                self._set_cloth_bone_props(arm_obj, bone_name, props)
                self._apply_cloth_bone_color(arm_obj, bone_name, flags)
                continue

            props = {
                'trlau_cloth_enabled': True,
                'trlau_cloth_pinned': False,
            }

            if flags == 1:
                if int(index) in capsule_point_indices:
                    continue
                key = point_keys[index]
                display_radius = collision_radius_by_key.get(key, 8.0) or 8.0
                sphere_obj = bpy.data.objects.new(f'{base_name}_Cloth_Collision_{index:03d}', None)
                self._link_object_to_collection(sphere_obj, collection)
                sphere_obj.parent = cloth_obj
                sphere_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                sphere_obj.empty_display_type = 'SPHERE'
                sphere_obj.empty_display_size = 1.0
                sphere_obj.show_name = False
                sphere_obj.show_in_front = True
                local_position = mathutils.Vector((float(getattr(point, 'x', 0.0)), float(getattr(point, 'y', 0.0)), float(getattr(point, 'z', 0.0))))
                try:
                    bone = bone_lookup.get(f'bone_{segment}') if bone_lookup is not None else None
                    if bone is not None:
                        sphere_obj.matrix_local = bone.matrix_local @ mathutils.Matrix.Translation(local_position)
                        child_of = sphere_obj.constraints.new(type='CHILD_OF')
                        child_of.name = 'TRLAU Cloth Collision Bone Follow'
                        child_of.target = arm_obj
                        child_of.subtarget = f'bone_{segment}'
                        child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ cloth_obj.matrix_world
                    else:
                        sphere_obj.location = local_position
                except Exception:
                    sphere_obj.location = local_position

                radius_value = max(0.0, float(display_radius))
                sphere_obj.scale = (radius_value, radius_value, radius_value)
                self._lock_cloth_collision_rotation(sphere_obj)
                collision_spheres_by_point_index[int(index)] = sphere_obj


        collision_pin_anchors_by_point_index: dict[int, object] = {}
        pin_collision_point_indices = sorted({
            int(rule_info.get('collision_point_index', -1))
            for rule_info in collision_distance_rules
            if int(rule_info.get('collision_point_flags', 0)) == 5
        })
        for point_index in pin_collision_point_indices:
            if not (0 <= int(point_index) < len(points)):
                continue
            point = points[int(point_index)]
            segment = int(getattr(point, 'segment', 0))
            anchor_obj = bpy.data.objects.new(f'{base_name}_Cloth_PinCollision_{int(point_index):03d}', None)
            self._link_object_to_collection(anchor_obj, collection)
            anchor_obj.parent = cloth_obj
            anchor_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            anchor_obj.empty_display_type = 'SPHERE'
            anchor_obj.empty_display_size = 0.25
            anchor_obj.show_name = False
            anchor_obj.show_in_front = True

            local_position = mathutils.Vector((
                float(getattr(point, 'x', 0.0)),
                float(getattr(point, 'y', 0.0)),
                float(getattr(point, 'z', 0.0)),
            ))
            try:
                bone = bone_lookup.get(f'bone_{segment}') if bone_lookup is not None else None
                if bone is not None:
                    anchor_obj.matrix_local = bone.matrix_local @ mathutils.Matrix.Translation(local_position)
                    child_of = anchor_obj.constraints.new(type='CHILD_OF')
                    child_of.name = 'TRLAU Cloth Pin Collision Bone Follow'
                    child_of.target = arm_obj
                    child_of.subtarget = f'bone_{segment}'
                    child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ cloth_obj.matrix_world
                else:
                    anchor_obj.location = local_position
            except Exception:
                anchor_obj.location = local_position
            anchor_obj.scale = (1.0, 1.0, 1.0)
            self._lock_cloth_collision_rotation(anchor_obj)
            collision_pin_anchors_by_point_index[int(point_index)] = anchor_obj

        def _cloth_point_display_matrix(point_index: int) -> tuple[mathutils.Matrix | None, str | None]:
            try:
                point = points[int(point_index)]
                segment = int(getattr(point, 'segment', 0))
                local_position = mathutils.Vector((
                    float(getattr(point, 'x', 0.0)),
                    float(getattr(point, 'y', 0.0)),
                    float(getattr(point, 'z', 0.0)),
                ))
            except Exception:
                return None, None
            try:
                bone = bone_lookup.get(f'bone_{segment}') if bone_lookup is not None else None
                if bone is not None:
                    return bone.matrix_local @ mathutils.Matrix.Translation(local_position), f'bone_{segment}'
            except Exception:
                pass
            try:
                return mathutils.Matrix.Translation(local_position), None
            except Exception:
                return None, None

        for rule_order, rule_info in enumerate(collision_distance_rules):
            try:
                collision_point_index = int(rule_info.get('collision_point_index', -1))
                target_point_index = int(rule_info.get('target_point_index', -1))
                collision_obj = collision_spheres_by_point_index.get(collision_point_index) or collision_pin_anchors_by_point_index.get(collision_point_index)
                if collision_obj is None:
                    continue
                target_point = points[target_point_index]
                target_segment = int(getattr(target_point, 'segment', 0))
                radius_value = max(0.0, float(rule_info.get('radius', 0.0) or 0.0))
            except Exception:
                continue
            rules_group = self._ensure_cloth_collision_rules_group(
                cloth_obj,
                collection,
                base_name,
                arm_obj=arm_obj,
                target_segment=int(target_segment),
            )
            rule_obj = bpy.data.objects.new(f'{base_name}_Cloth_CollisionRule_{int(rule_order):03d}_B{int(target_segment):03d}_P{int(target_point_index):03d}_C{int(collision_point_index):03d}', None)
            self._link_object_to_collection(rule_obj, collection)
            rule_obj.parent = rules_group if rules_group is not None else cloth_obj
            rule_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            rule_obj.empty_display_type = 'SPHERE'
            rule_obj.empty_display_size = 1.0
            rule_obj.show_name = False
            rule_obj.show_in_front = True
            rule_obj.empty_display_size = 0.1

            display_matrix, target_bone_name = _cloth_point_display_matrix(target_point_index)
            if display_matrix is not None:
                try:
                    rule_obj.matrix_local = display_matrix
                except Exception:
                    try:
                        rule_obj.location = display_matrix.translation
                    except Exception:
                        pass
            else:
                try:
                    rule_obj.location = collision_obj.location.copy()
                except Exception:
                    pass
            if target_bone_name and arm_obj is not None:
                try:
                    bone = bone_lookup.get(target_bone_name) if bone_lookup is not None else None
                    child_of = rule_obj.constraints.new(type='CHILD_OF')
                    child_of.name = 'TRLAU Cloth Collision Rule Target Follow'
                    child_of.target = arm_obj
                    child_of.subtarget = target_bone_name
                    if bone is not None:
                        child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ getattr(rule_obj.parent, 'matrix_world', cloth_obj.matrix_world)
                except Exception:
                    pass
            rule_radius_value = float(radius_value)
            rule_obj.scale = (rule_radius_value, rule_radius_value, rule_radius_value)
            self._lock_cloth_collision_rotation(rule_obj)

        def _iter_hmarker_candidates():
            seen = set()

            def _yield_once(obj):
                if obj is None:
                    return
                try:
                    key = int(getattr(obj, 'as_pointer', lambda: id(obj))())
                except Exception:
                    key = id(obj)
                if key in seen:
                    return
                seen.add(key)
                yield obj

            if arm_obj is not None:
                for source in (getattr(arm_obj, 'children_recursive', None), getattr(arm_obj, 'children', None)):
                    try:
                        for obj in list(source or []):
                            yield from _yield_once(obj)
                    except Exception:
                        pass
            try:
                for obj in list(getattr(collection, 'all_objects', []) or []):
                    yield from _yield_once(obj)
            except Exception:
                pass
            try:
                for obj in list(getattr(bpy.data, 'objects', []) or []):
                    yield from _yield_once(obj)
            except Exception:
                pass

        def _safe_int(value, default=None):
            try:
                return int(value)
            except Exception:
                return default

        def _find_imported_hmarker_object(marker_value, preferred_segment=None):
            marker_index = _safe_int(marker_value, None)
            if marker_index is None:
                return None
            marker_index = abs(int(marker_index))
            if marker_index <= 0:
                return None
            exact_candidates = []
            for candidate in _iter_hmarker_candidates():
                try:
                    is_hmarker = trlau_object_type(candidate) == 'HMarker'
                except Exception:
                    is_hmarker = False
                name = str(getattr(candidate, 'name', '') or '')
                if not is_hmarker and '_HMarker_' not in name:
                    continue
                name_index = hmarker_index_from_name(candidate, 0)
                if name_index == marker_index:
                    exact_candidates.append(candidate)
                    continue
            candidates = exact_candidates
            if not candidates:
                return None
            return candidates[0]

        missing_hmarker_cache: dict[int, object] = {}

        def _get_or_create_hmarker_group():
            if arm_obj is not None:
                try:
                    _base, _collection, hinfo_obj = ArmatureBuilder._get_or_create_hinfo_objects(arm_obj)
                    return ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_HMarkers', hinfo_obj)
                except Exception:
                    pass
            hinfo_name = f'{base_name}_HInfo'
            hinfo_obj = bpy.data.objects.get(hinfo_name)
            if hinfo_obj is None:
                hinfo_obj = bpy.data.objects.new(hinfo_name, None)
                hinfo_obj.empty_display_type = 'PLAIN_AXES'
                hinfo_obj.empty_display_size = 10.0
            self._link_object_to_collection(hinfo_obj, collection)
            group_name = f'{base_name}_HMarkers'
            group_obj = bpy.data.objects.get(group_name)
            if group_obj is None:
                group_obj = bpy.data.objects.new(group_name, None)
                group_obj.empty_display_type = 'PLAIN_AXES'
                group_obj.empty_display_size = 10.0
            self._link_object_to_collection(group_obj, collection)
            group_obj.parent = hinfo_obj
            group_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            return group_obj

        def _create_placeholder_hmarker(marker_value, target_indices, preferred_segment=None):
            marker_index = _safe_int(marker_value, None)
            if marker_index is None:
                return None
            marker_index = abs(int(marker_index))
            if marker_index <= 0:
                return None
            existing = _find_imported_hmarker_object(marker_index, preferred_segment=preferred_segment)
            if existing is not None:
                return existing
            cached = missing_hmarker_cache.get(marker_index)
            if cached is not None:
                return cached

            target_center, _target_positions = _target_centroid(target_indices)
            if target_center is None:
                try:
                    target_center = cloth_obj.matrix_world.translation.copy()
                except Exception:
                    target_center = mathutils.Vector((0.0, 0.0, 0.0))

            bone_index = _safe_int(preferred_segment, 0) or 0
            marker_position = mathutils.Vector((0.0, 0.0, 0.0))
            marker_rotation = (0.0, 0.0, 0.0)
            marker_matrix_world = mathutils.Matrix.Translation(target_center)

            group_obj = _get_or_create_hmarker_group()
            marker_obj = bpy.data.objects.new(f'{base_name}_HMarker_Missing_{int(marker_index):03d}', None)
            marker_obj.empty_display_type = 'ARROWS'
            marker_obj.empty_display_size = 10.0
            marker_obj.show_in_front = True
            marker_obj.rotation_mode = 'ZYX'
            self._link_object_to_collection(marker_obj, collection)
            marker_obj.parent = group_obj
            marker_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)

            bone = None
            if arm_obj is not None and bone_index >= 0:
                try:
                    bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(f'bone_{int(bone_index)}')
                except Exception:
                    bone = None
            if arm_obj is not None and bone is not None:
                try:
                    bone_world = arm_obj.matrix_world @ bone.matrix_local
                    marker_position = bone_world.inverted_safe() @ target_center
                    local_transform = mathutils.Matrix.Translation(marker_position)
                    marker_obj.matrix_local = bone.matrix_local @ local_transform
                    child_of = marker_obj.constraints.new(type='CHILD_OF')
                    child_of.name = 'TRLau HMarker Bone Follow'
                    child_of.target = arm_obj
                    child_of.subtarget = f'bone_{int(bone_index)}'
                    child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted_safe() @ group_obj.matrix_world
                    marker_matrix_world = arm_obj.matrix_world @ bone.matrix_local @ local_transform
                except Exception:
                    try:
                        marker_obj.matrix_world = marker_matrix_world
                    except Exception:
                        pass
            else:
                try:
                    marker_obj.matrix_world = marker_matrix_world
                except Exception:
                    pass
                try:
                    marker_position = mathutils.Vector(tuple(marker_obj.location))
                except Exception:
                    pass

            marker_obj['trlau_hmarker_placeholder'] = True
            missing_hmarker_cache[marker_index] = marker_obj
            return marker_obj

        def _ensure_imported_hmarker_object(marker_value, target_indices, preferred_segment=None):
            existing = _find_imported_hmarker_object(marker_value, preferred_segment=preferred_segment)
            if existing is not None:
                return existing
            return _create_placeholder_hmarker(marker_value, target_indices, preferred_segment=preferred_segment)

        def _marker_matrix_world(marker_obj):
            if marker_obj is None:
                return None
            try:
                return marker_obj.matrix_world.copy()
            except Exception:
                try:
                    parent = getattr(marker_obj, 'parent', None)
                    if parent is not None:
                        return parent.matrix_world @ marker_obj.matrix_local
                except Exception:
                    pass
            return None

        def _safe_axis(vector, default=(1.0, 0.0, 0.0)):
            try:
                axis = mathutils.Vector(tuple(float(v) for v in vector))
                if axis.length > 1.0e-6:
                    axis.normalize()
                    return axis
            except Exception:
                pass
            axis = mathutils.Vector(tuple(float(v) for v in default))
            if axis.length <= 1.0e-6:
                axis = mathutils.Vector((1.0, 0.0, 0.0))
            axis.normalize()
            return axis

        def _cloth_point_world_position(point_index: int):
            try:
                display_matrix, _bone_name = _cloth_point_display_matrix(int(point_index))
                if display_matrix is not None:
                    return cloth_obj.matrix_world @ display_matrix.translation
            except Exception:
                pass
            try:
                return cloth_obj.matrix_world.translation.copy()
            except Exception:
                return mathutils.Vector((0.0, 0.0, 0.0))

        def _target_centroid(target_indices):
            positions = []
            for target_index in target_indices or []:
                try:
                    positions.append(_cloth_point_world_position(int(target_index)))
                except Exception:
                    pass
            if not positions:
                return None, []
            center = mathutils.Vector((0.0, 0.0, 0.0))
            for position in positions:
                center += position
            center /= max(1, len(positions))
            return center, positions

        def _marker_plane_basis(marker_obj, target_indices):
            matrix = _marker_matrix_world(marker_obj)
            target_center, target_positions = _target_centroid(target_indices)
            flipped_normal = False
            if matrix is not None:
                origin = matrix.translation.copy()
                basis = matrix.to_3x3()
                x_axis = _safe_axis(basis.col[0], (1.0, 0.0, 0.0))
                y_axis = _safe_axis(basis.col[1], (0.0, 1.0, 0.0))
                normal = _safe_axis(basis.col[2], (0.0, 0.0, 1.0))

                y_axis = normal.cross(x_axis)
                if y_axis.length <= 1.0e-6:
                    y_axis = _safe_axis(basis.col[1], (0.0, 1.0, 0.0))
                    x_axis = y_axis.cross(normal)
                if x_axis.length <= 1.0e-6:
                    x_axis = _safe_axis((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
                else:
                    x_axis.normalize()
                if y_axis.length <= 1.0e-6:
                    y_axis = normal.cross(x_axis)
                if y_axis.length <= 1.0e-6:
                    y_axis = _safe_axis((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
                else:
                    y_axis.normalize()
                normal = x_axis.cross(y_axis)
                if normal.length <= 1.0e-6:
                    normal = _safe_axis(basis.col[2], (0.0, 0.0, 1.0))
                else:
                    normal.normalize()
            else:
                origin = target_center.copy() if target_center is not None else getattr(cloth_obj, 'matrix_world', mathutils.Matrix.Identity(4)).translation.copy()
                x_axis = mathutils.Vector((1.0, 0.0, 0.0))
                y_axis = mathutils.Vector((0.0, 1.0, 0.0))
                normal = mathutils.Vector((0.0, 0.0, 1.0))

            if target_center is not None:
                try:
                    if float((target_center - origin).dot(normal)) < 0.0:
                        y_axis.negate()
                        normal.negate()
                        flipped_normal = True
                except Exception:
                    pass

            xs = []
            ys = []
            for position in target_positions:
                try:
                    offset = position - origin
                    xs.append(float(offset.dot(x_axis)))
                    ys.append(float(offset.dot(y_axis)))
                except Exception:
                    pass
            if not xs:
                xs = [-5.0, 5.0]
            if not ys:
                ys = [-5.0, 5.0]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)
            width = max(1.0e-5, float(x_max) - float(x_min))
            height = max(1.0e-5, float(y_max) - float(y_min))
            margin = max(3.0, min(max(width, height) * 0.12, 14.0))
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin
            if x_max - x_min < 6.0:
                center_x = (x_min + x_max) * 0.5
                x_min = center_x - 3.0
                x_max = center_x + 3.0
            if y_max - y_min < 6.0:
                center_y = (y_min + y_max) * 0.5
                y_min = center_y - 3.0
                y_max = center_y + 3.0

            return {
                'origin': origin,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'normal': normal,
                'x_min': float(x_min),
                'x_max': float(x_max),
                'y_min': float(y_min),
                'y_max': float(y_max),
                'marker_matrix': matrix,
                'flipped_normal': bool(flipped_normal),
            }

        def _make_marker_follow_plane_mesh(mesh_name: str, plane: dict):
            x_min = float(plane.get('x_min', -3.0))
            x_max = float(plane.get('x_max', 3.0))
            y_min = float(plane.get('y_min', -3.0))
            y_max = float(plane.get('y_max', 3.0))
            y_sign = -1.0 if bool(plane.get('flipped_normal', False)) else 1.0
            vertices = [
                (x_min, y_sign * y_min, 0.0),
                (x_max, y_sign * y_min, 0.0),
                (x_max, y_sign * y_max, 0.0),
                (x_min, y_sign * y_max, 0.0),
            ]
            face = (0, 3, 2, 1) if bool(plane.get('flipped_normal', False)) else (0, 1, 2, 3)
            edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
            mesh = bpy.data.meshes.new(mesh_name)
            mesh.from_pydata(vertices, edges, [face])
            try:
                mesh.update()
            except Exception:
                pass
            return mesh

        def _make_plane_rule_selector_mesh(mesh_name: str, target_positions, world_to_local):
            selector_positions = list(target_positions or [])
            if not selector_positions:
                return None

            def _to_local(point):
                try:
                    return tuple(world_to_local @ point)
                except Exception:
                    return tuple(point)

            try:
                xs = [float(p.x) for p in selector_positions]
                ys = [float(p.y) for p in selector_positions]
                zs = [float(p.z) for p in selector_positions]
                extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0e-5)
                margin = max(2.0, min(float(extent) * 0.08, 10.0))
                x0, x1 = min(xs) - margin, max(xs) + margin
                y0, y1 = min(ys) - margin, max(ys) + margin
                z0, z1 = min(zs) - margin, max(zs) + margin
                box_points = [
                    mathutils.Vector((x0, y0, z0)), mathutils.Vector((x1, y0, z0)),
                    mathutils.Vector((x1, y1, z0)), mathutils.Vector((x0, y1, z0)),
                    mathutils.Vector((x0, y0, z1)), mathutils.Vector((x1, y0, z1)),
                    mathutils.Vector((x1, y1, z1)), mathutils.Vector((x0, y1, z1)),
                ]
                vertices = [_to_local(point) for point in box_points]
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
            except Exception:
                return None

            mesh = bpy.data.meshes.new(mesh_name)
            mesh.from_pydata(vertices, edges, [])
            try:
                mesh.update()
            except Exception:
                pass
            return mesh

        plane_rules = list(getattr(setup, 'planeRules', []) or [])
        if plane_rules:
            plane_material = self._make_cloth_material('TRLAU Cloth PlaneRule', (0.15, 0.55, 1.0, 0.35))
            try:
                plane_material.blend_method = 'BLEND'
                plane_material.use_screen_refraction = False
                plane_material.show_transparent_back = True
            except Exception:
                pass
            grouped_plane_rules: dict[tuple[int, int], list[tuple[int, object]]] = {}
            group_order: list[tuple[int, int]] = []
            for rule_index, plane_rule in enumerate(plane_rules):
                try:
                    marker_a_raw = int(getattr(plane_rule, 'planeSegment', (0, 0))[0])
                    marker_b_raw = int(getattr(plane_rule, 'planeSegment', (0, 0))[1])
                    point_index = int(getattr(plane_rule, 'point', (0,))[0])
                except Exception:
                    continue
                if marker_a_raw == 0 and marker_b_raw == 0:
                    continue
                if not (0 <= point_index < len(points)):
                    continue
                key = (int(marker_a_raw), int(marker_b_raw))
                if key not in grouped_plane_rules:
                    grouped_plane_rules[key] = []
                    group_order.append(key)
                grouped_plane_rules[key].append((int(rule_index), plane_rule))

            unresolved_count = 0
            for plane_order, marker_pair in enumerate(group_order):
                records = grouped_plane_rules.get(marker_pair, [])
                if not records:
                    continue
                marker_a_raw, marker_b_raw = marker_pair
                target_point_indices = []
                target_point_segments = []
                target_flags = []
                source_rule_indices = []
                for source_rule_index, rule in records:
                    try:
                        target_index = int(getattr(rule, 'point', (0,))[0])
                    except Exception:
                        continue
                    if not (0 <= target_index < len(points)):
                        continue
                    target_point = points[int(target_index)]
                    target_point_indices.append(int(target_index))
                    target_point_segments.append(int(getattr(target_point, 'segment', -1)))
                    try:
                        target_flags.append(int(getattr(rule, 'flags', (0,))[0]))
                    except Exception:
                        target_flags.append(0)
                    source_rule_indices.append(int(source_rule_index))
                if not target_point_indices:
                    continue

                preferred_segment = int(target_point_segments[0]) if target_point_segments else None
                marker_a_obj = _ensure_imported_hmarker_object(marker_a_raw, target_point_indices, preferred_segment=preferred_segment)
                marker_b_obj = _ensure_imported_hmarker_object(marker_b_raw, target_point_indices, preferred_segment=preferred_segment)
                if marker_a_obj is None or marker_b_obj is None:
                    unresolved_count += 1

                _target_center_for_selector, target_positions = _target_centroid(target_point_indices)
                plane_a = _marker_plane_basis(marker_a_obj, target_point_indices)
                plane_b = _marker_plane_basis(marker_b_obj, target_point_indices)
                width = max(
                    1.0e-5,
                    float(plane_a['x_max']) - float(plane_a['x_min']),
                    float(plane_b['x_max']) - float(plane_b['x_min']),
                )
                height = max(
                    1.0e-5,
                    float(plane_a['y_max']) - float(plane_a['y_min']),
                    float(plane_b['y_max']) - float(plane_b['y_min']),
                )

                plane_name = f'{base_name}_Cloth_PlaneRule_{int(plane_order):03d}_M{abs(int(marker_a_raw)):03d}_M{abs(int(marker_b_raw)):03d}'
                try:
                    parent_inv = cloth_obj.matrix_world.inverted_safe()
                except Exception:
                    parent_inv = mathutils.Matrix.Identity(4)
                plane_obj = bpy.data.objects.new(plane_name, None)
                self._link_object_to_collection(plane_obj, collection)
                plane_obj.parent = cloth_obj
                plane_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                try:
                    plane_obj.matrix_local = mathutils.Matrix.Identity(4)
                except Exception:
                    pass
                plane_obj.empty_display_type = 'PLAIN_AXES'
                plane_obj.empty_display_size = 4.0
                plane_obj.show_name = False
                plane_obj.show_in_front = True
                plane_obj['trlau_cloth_enabled'] = True
                plane_obj['trlau_export_ignore'] = True
                plane_obj['trlau_cloth_plane_rule_index'] = int(plane_order)
                plane_obj['trlau_cloth_plane_display_width'] = float(width)
                plane_obj['trlau_cloth_plane_display_height'] = float(height)
                plane_obj['trlau_cloth_plane_resolved_hmarkers'] = bool(marker_a_obj is not None and marker_b_obj is not None)

                for face_label, marker_raw, marker_obj, plane_def in (
                    ('A', marker_a_raw, marker_a_obj, plane_a),
                    ('B', marker_b_raw, marker_b_obj, plane_b),
                ):
                    marker_abs = abs(int(marker_raw))
                    face_mesh = _make_marker_follow_plane_mesh(f'{plane_name}_M{marker_abs:03d}_{face_label}_Mesh', plane_def)
                    face_obj = bpy.data.objects.new(f'{plane_name}_M{marker_abs:03d}_{face_label}', face_mesh)
                    self._link_object_to_collection(face_obj, collection)
                    face_obj.parent = plane_obj
                    face_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                    try:
                        face_obj.matrix_local = mathutils.Matrix.Identity(4)
                    except Exception:
                        pass
                    try:
                        if plane_material is not None:
                            face_obj.data.materials.append(plane_material)
                    except Exception:
                        pass
                    try:
                        face_obj.show_wire = True
                        face_obj.show_in_front = True
                    except Exception:
                        pass
                    face_obj['trlau_cloth_enabled'] = True
                    face_obj['trlau_export_ignore'] = True
                    face_obj['trlau_cloth_plane_rule_index'] = int(plane_order)
                    face_obj['trlau_cloth_plane_marker_index'] = int(marker_abs)
                    if marker_obj is not None:
                        try:
                            marker_matrix = _marker_matrix_world(marker_obj)
                            if marker_matrix is not None:
                                face_obj.matrix_world = marker_matrix
                        except Exception:
                            pass
                        try:
                            follow = face_obj.constraints.new(type='COPY_TRANSFORMS')
                            follow.name = 'TRLAU Cloth PlaneRule HMarker Follow'
                            follow.target = marker_obj
                            follow.target_space = 'WORLD'
                            follow.owner_space = 'WORLD'
                        except Exception:
                            pass

                selector_mesh = _make_plane_rule_selector_mesh(f'{plane_name}_Selector_Mesh', target_positions, parent_inv)
                if selector_mesh is not None:
                    selector_obj = bpy.data.objects.new(f'{plane_name}_Selector', selector_mesh)
                    self._link_object_to_collection(selector_obj, collection)
                    selector_obj.parent = plane_obj
                    selector_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
                    try:
                        selector_obj.matrix_local = mathutils.Matrix.Identity(4)
                    except Exception:
                        pass
                    try:
                        selector_obj.data.materials.append(plane_material)
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
                    selector_obj['trlau_cloth_plane_rule_index'] = int(plane_order)

            if unresolved_count:
                logger.warning('Imported cloth PlaneRules: %d unique HMarker pair(s) could not fully resolve HMarkers', unresolved_count)

        logger.info('Imported cloth setup with %d point(s) into %s', len(points), cloth_name)
        return cloth_obj

    def _import_model_markups(self, collection, model, *, arm_obj=None, source_name: str | None = None):
        if not getattr(self, 'import_hinfo', True) or not getattr(self, 'import_markups', True):
            return []
        markups = list(getattr(model, 'markups', []) or [])
        if not markups:
            return []

        base_name = source_name or (arm_obj.name if arm_obj is not None else collection.name)
        if base_name.endswith('_Armature'):
            base_name = base_name[:-9]
        elif base_name.endswith('_Model'):
            base_name = base_name[:-6]
        parent_name = f'{base_name}_Markup'
        markup_parent = None if bool(getattr(self, 'force_unique_import_names', False)) else bpy.data.objects.get(parent_name)
        if markup_parent is None:
            markup_parent = bpy.data.objects.new(parent_name, None)
        self._link_object_to_collection(markup_parent, collection)
        markup_parent.empty_display_type = 'PLAIN_AXES'
        markup_parent.empty_display_size = 32.0
        markup_parent.rotation_mode = 'XYZ'
        markup_parent.location = (0.0, 0.0, 0.0)
        markup_parent.rotation_euler = (0.0, 0.0, 0.0)
        markup_parent.scale = (1.0, 1.0, 1.0)
        markup_parent['trlau_type'] = 'MarkupRoot'
        markup_parent['trlau_component_name'] = 'Markup'
        if arm_obj is not None:
            markup_parent.parent = arm_obj
            markup_parent.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        else:
            markup_parent.parent = None

        created = []
        for markup in markups:
            polyline = list(getattr(markup, 'polyline', []) or [])
            curve_name = f'{base_name}_Markup_{int(getattr(markup, "index", 0)):03d}'
            curve_data = bpy.data.curves.new(curve_name, type='CURVE')
            curve_data.dimensions = '3D'
            curve_data.resolution_u = 1
            markup_position = tuple(float(v) for v in getattr(markup, 'position', (0.0, 0.0, 0.0)))
            markup_flags = int(getattr(markup, 'flags', 0) or 0)
            bbox_used_for_shape = False
            if markup_flags & MARKUP_BBOX_FLAGS:
                bbox_used_for_shape = _add_markup_bbox_wire(curve_data, getattr(markup, 'bbox', ()), markup_position, bbox_is_local=True)
            if polyline and not bbox_used_for_shape:
                spline = curve_data.splines.new('POLY')
                if len(polyline) > 1:
                    spline.points.add(len(polyline) - 1)
                base_x, base_y, base_z = markup_position
                for point_index, point in enumerate(polyline):
                    px, py, pz = float(point[0]), float(point[1]), float(point[2])
                    pw = float(point[3]) if len(point) > 3 else 1.0
                    spline.points[point_index].co = (px - base_x, py - base_y, pz - base_z, pw)
            curve_obj = bpy.data.objects.new(curve_data.name, curve_data)
            curve_obj.location = markup_position
            curve_obj.rotation_euler = (0.0, 0.0, 0.0)
            curve_obj.scale = (1.0, 1.0, 1.0)
            curve_obj['trlau_type'] = 'Markup'
            curve_obj['trlau_markup_index'] = int(getattr(markup, 'index', -1))
            curve_obj['trlau_markup_game'] = str(getattr(markup, 'game', 'unknown'))
            curve_obj['trlau_markup_flags'] = markup_flags if -(2**31) <= int(markup_flags) <= (2**31 - 1) else str(int(markup_flags))
            curve_obj['trlau_markup_point_count'] = len(polyline)
            if bbox_used_for_shape:
                curve_obj['trlau_markup_bbox_shape'] = True
            curve_obj['trlau_markup_animated_segment'] = int(getattr(markup, 'animated_segment', 0))
            self._link_object_to_collection(curve_obj, collection)
            curve_obj.parent = markup_parent
            curve_obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            created.append(curve_obj)

        return created

    @staticmethod
    def _looks_like_pc_nextgen_mesh(filepath: str) -> bool:
        try:
            path = Path(filepath)
            if path.suffix.lower() not in {'.gnc', '.ngm', '.ngmesh', '.tr7ngmesh'}:
                return False
            data = path.read_bytes()[:0x100]
            return b'PCD9' in data
        except Exception:
            return False

    def _create_mesh_parser(self, filepath: str):
        platform = getattr(self, "platform", "PC")
        if platform == "PC" and self._looks_like_pc_nextgen_mesh(filepath):
            return TRNextGenModelParser(filepath, cdc_render_data_id=0)
        if Path(filepath).suffix.lower() == '.tr8mesh' or (platform == "PC" and TR8ModelParser.looks_like_tr8mesh(filepath)):
            return TR8ModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))
        if platform == "XBOX":
            return TRXboxModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))
        if platform == "PS2":
            return TRPS2ModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))
        if platform == "PSP":
            return TRPSPModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))
        if platform == "PS3":
            return TRPS3ModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))
        if platform == "XBOX360":
            return TRXbox360ModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))
        return TRModelParser(filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, "import_markups", True))

    @staticmethod
    def _clear_prefixed_custom_properties(obj, prefixes: tuple[str, ...]) -> None:
        for key in list(getattr(obj, 'keys', lambda: [])()):
            if any(str(key).startswith(prefix) for prefix in prefixes):
                try:
                    del obj[key]
                except Exception:
                    pass

    @staticmethod
    def _model_game_from_model(model) -> str:
        for markup in list(getattr(model, 'markups', []) or []):
            game = str(getattr(markup, 'game', '') or '').strip().lower()
            if game in {'legend', 'trl', 'tr7'}:
                return 'legend'
            if game in {'anniversary', 'tra', 'tr8'}:
                return 'anniversary'

        uv_format = str(getattr(model, 'uv_format', '') or '').strip().lower()
        if uv_format == 'underworld' or int(getattr(model, 'version', 0) or 0) == 19:
            return 'underworld'
        if uv_format == 'xbox':
            return 'legend'
        if uv_format in {'ps3', 'xbox360'}:
            return 'legend' if bool(getattr(model, 'ps3_external_render_stream', False)) else 'anniversary'

        try:
            return 'legend' if int(getattr(model, 'cdc_render_data_id', 0) or 0) != 0 else 'anniversary'
        except Exception:
            return 'legend'

    @staticmethod
    def _model_platform_from_model(model) -> str:
        uv_format = str(getattr(model, 'uv_format', '') or '').strip().lower()
        if uv_format in {'ps2', 'psp', 'ps3', 'xbox', 'xbox360', 'gamecube', 'pc_nextgen'}:
            return uv_format
        if uv_format == 'underworld' or int(getattr(model, 'version', 0) or 0) == 19:
            for strip in list(getattr(model, 'strips', []) or []):
                try:
                    if int(getattr(strip, 'tr8_ps2_material_index', -1)) >= 0:
                        return 'ps2'
                except Exception:
                    pass
                try:
                    if int(getattr(strip, 'ps2_tpageid_raw', -1)) >= 0:
                        return 'ps2'
                except Exception:
                    pass
            return 'pc'
        return 'pc'

    @staticmethod
    def _store_model_empty_metadata(obj, model) -> None:
        game_name = ModelImporterMixin._model_game_from_model(model)
        platform_name = ModelImporterMixin._model_platform_from_model(model)
        render_data_id = int(getattr(model, 'cdc_render_data_id', 0))

        try:
            obj.trlau_is_model_empty = True
        except Exception:
            pass
        try:
            obj.trlau_ui_cdc_render_data_id = render_data_id
        except Exception:
            pass
        try:
            obj.trlau_model_game = game_name
        except Exception:
            pass
        try:
            obj['trlau_model_game_id'] = game_name
        except Exception:
            pass
        try:
            obj.trlau_model_platform = platform_name
        except Exception:
            pass
        try:
            obj['trlau_model_platform_id'] = platform_name
        except Exception:
            pass


        bone_mirror_entries = list(getattr(model, 'bone_mirror_entries', []) or [])
        collection = getattr(obj, 'trlau_bone_mirror_entries', None)
        if collection is not None:
            collection.clear()

        for entry in bone_mirror_entries:
            if collection is not None:
                item = collection.add()
                item.bone1 = int(getattr(entry, 'bone1', 0))
                item.bone2 = int(getattr(entry, 'bone2', 0))
                item.count = int(getattr(entry, 'count', 0))

    def _get_or_create_model_empty(self, collection, name: str, model):
        force_unique_names = bool(getattr(self, 'force_unique_import_names', False))
        model_empty = None if force_unique_names else bpy.data.objects.get(name)
        if model_empty is not None and getattr(model_empty, 'type', None) != 'EMPTY':
            try:
                model_empty.name = f'{name}_Mesh'
            except Exception:
                pass
            model_empty = None
        if model_empty is None:
            model_empty = bpy.data.objects.new(name, None)
        self._link_object_to_collection(model_empty, collection)
        model_empty.empty_display_type = 'PLAIN_AXES'
        model_empty.empty_display_size = 32.0
        model_empty.parent = None
        model_empty.location = (0.0, 0.0, 0.0)
        model_empty.rotation_mode = 'XYZ'
        model_empty.rotation_euler = (0.0, 0.0, 0.0)
        model_empty.scale = (1.0, 1.0, 1.0)
        self._store_model_empty_metadata(model_empty, model)
        return model_empty

    @staticmethod
    def _parent_to_model_empty(obj, model_empty) -> None:
        if obj is None or model_empty is None:
            return
        obj.parent = model_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    def _import_mesh_file(self, context, filepath: str, collection=None, name_prefix: str | None = None, *, create_model_root: bool = True, underworld_model_filepath: str | None = None):
        parser = self._create_mesh_parser(filepath)
        model = parser.parse()
        if underworld_model_filepath and isinstance(parser, TR8ModelParser):
            model = parser.attach_underworld_model_hinfo(model, underworld_model_filepath)
        filepath_obj = Path(filepath)
        folder_name = name_prefix or filepath_obj.parent.name or filepath_obj.stem
        file_stem = filepath_obj.stem
        first_filename_part = file_stem.split('_')[0] if '_' in file_stem else file_stem
        collection = collection or self._get_or_create_collection(context, filepath)

        model_root_name = f'{folder_name}_{first_filename_part}_Model'
        mesh_name = f'{model_root_name}_Mesh' if create_model_root else model_root_name
        armature_name = f'{folder_name}_{first_filename_part}_Armature'
        model_root_obj = self._get_or_create_model_empty(collection, model_root_name, model) if create_model_root else None

        mesh_entries = []
        mesh_objects = []
        if not self.import_armature_only:
            mesh_entries = self._build_mesh_objects(context, filepath, model, collection, mesh_name)
            mesh_objects = [entry['object'] for entry in mesh_entries]

        arm_obj = None
        if model.segments:
            arm_obj = ArmatureBuilder(context, filepath, collection=collection, object_name=armature_name).build(model)
            self._parent_to_model_empty(arm_obj, model_root_obj)
            for mesh_entry in mesh_entries:
                ArmatureBuilder.bind_vertices(mesh_entry['object'], arm_obj, mesh_entry['bind_model'])
            if self.import_hinfo:
                print('Importing HInfo...')
                ArmatureBuilder.build_hmarkers(arm_obj, model)
                ArmatureBuilder.build_hspheres(arm_obj, model)
                ArmatureBuilder.build_hboxes(arm_obj, model)
                ArmatureBuilder.build_hcapsules(arm_obj, model)
            if self.import_bounding_boxes:
                ArmatureBuilder.build_bounds(arm_obj, model)
            if self.import_hinfo and not self.import_armature_only and not getattr(self, '_trlau_skip_targets', False):
                ArmatureBuilder.build_targets(arm_obj, model)
        elif model_root_obj is not None:
            for mesh_obj in mesh_objects:
                self._parent_to_model_empty(mesh_obj, model_root_obj)

        self._import_model_markups(collection, model, arm_obj=arm_obj, source_name=armature_name if arm_obj is not None else model_root_name)

        built_face_count = sum(len(entry.indices) // 3 for entry in model.strips) or len(model.faces)
        return {
            'model': model,
            'mesh_obj': mesh_objects[0] if mesh_objects else None,
            'mesh_objects': mesh_objects,
            'arm_obj': arm_obj,
            'model_root_obj': model_root_obj,
            'faces': built_face_count,
        }

    @staticmethod
    def _section_entry_matches_next_gen(section, cdc_render_data_id: int) -> bool:
        try:
            if int(getattr(section, 'section_type', -1)) != 7:
                return False
            return (int(getattr(section, 'section_id', 0)) & 0xFFFFFFFF) == (int(cdc_render_data_id) & 0xFFFFFFFF)
        except Exception:
            return False

    @classmethod
    def _find_next_gen_model_path(cls, cdc_render_data_id: int, extracted_paths=None, extracted_sections=None, directory: Path | None = None) -> Path | None:
        try:
            render_id = int(cdc_render_data_id) & 0xFFFFFFFF
        except Exception:
            render_id = 0
        if render_id == 0:
            return None

        candidates: list[tuple[int, int, Path]] = []
        if extracted_paths is not None and extracted_sections is not None:
            for path, section in zip(list(extracted_paths), list(extracted_sections)):
                if not cls._section_entry_matches_next_gen(section, render_id):
                    continue
                try:
                    spec_rank = 0 if (int(getattr(section, 'spec_mask', 0)) & 0x80000000) else 1
                except Exception:
                    spec_rank = 1
                try:
                    size_rank = -int(getattr(section, 'size', 0))
                except Exception:
                    size_rank = 0
                candidates.append((spec_rank, size_rank, Path(path)))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
                return candidates[0][2]

        if directory is not None:
            try:
                hex_id = f'{render_id:x}'
                for candidate in sorted(Path(directory).iterdir()):
                    if not candidate.is_file():
                        continue
                    stem = candidate.stem.lower()
                    suffix = candidate.suffix.lower()
                    if suffix not in {'.tr7ngmesh', '.ngmesh', '.ngm', '.gnc'}:
                        continue
                    if stem.endswith(f'_{hex_id}') or stem.endswith(f'_{render_id}'):
                        return candidate
            except Exception:
                pass
        return None

    @staticmethod
    def _tag_next_gen_mesh_object(mesh_obj, mesh_model, *, render_id: int, source_filepath: str, model_root_obj=None, source_arm_obj=None) -> None:
        if mesh_obj is None:
            return
        try:
            mesh_obj.trlau_pc_nextgen_mesh = True
            mesh_obj.trlau_ui_cdc_render_data_id = int(render_id) & 0xFFFFFFFF
            mesh_obj.trlau_pc_nextgen_source_section_file = str(source_filepath)
            mesh_obj.trlau_pc_nextgen_material_count = len(getattr(mesh_model, 'strips', []) or [])
        except Exception:
            pass
        try:
            mesh_data = getattr(mesh_obj, 'data', None)
            if mesh_data is not None:
                mesh_data.trlau_pc_nextgen_mesh = True
        except Exception:
            pass
        try:
            if model_root_obj is not None:
                mesh_obj.trlau_pc_nextgen_model_root = str(getattr(model_root_obj, 'name', '') or '')
        except Exception:
            pass
        try:
            if source_arm_obj is not None:
                mesh_obj.trlau_pc_nextgen_armature = str(getattr(source_arm_obj, 'name', '') or '')
        except Exception:
            pass

    @staticmethod
    def _next_gen_mesh_index_for_root(model_root_obj) -> int:
        if model_root_obj is None:
            return 0
        count = 0
        for child in list(getattr(model_root_obj, 'children_recursive', []) or []):
            try:
                if bool(getattr(child, 'trlau_pc_nextgen_mesh', False)) or bool(child.get('trlau_pc_nextgen_mesh', False)):
                    count += 1
                    continue
            except Exception:
                pass
            try:
                if getattr(child, 'type', None) == 'MESH' and '_Mesh_NextGen_' in str(getattr(child, 'name', '') or ''):
                    count += 1
            except Exception:
                pass
        return count

    def _import_next_gen_mesh_file(self, context, filepath: str, source_result, collection=None, name_prefix: str | None = None):
        source_model = source_result.get('model') if isinstance(source_result, dict) else source_result
        source_arm_obj = source_result.get('arm_obj') if isinstance(source_result, dict) else None
        source_model_root_obj = source_result.get('model_root_obj') if isinstance(source_result, dict) else None
        render_id = int(getattr(source_model, 'cdc_render_data_id', 0) or 0)
        parser = TRNextGenModelParser(filepath, source_model=source_model, cdc_render_data_id=render_id)
        model = parser.parse()

        filepath_obj = Path(filepath)
        collection = collection or self._get_or_create_collection(context, filepath)
        if source_model_root_obj is not None:
            model_root_obj = source_model_root_obj
            root_name = str(getattr(model_root_obj, 'name', '') or '')
        else:
            folder_name = name_prefix or filepath_obj.parent.name or filepath_obj.stem
            id_part = f'{render_id:x}' if render_id else filepath_obj.stem.split('_')[0]
            root_name = f'{folder_name}_{id_part}_Model'
            model_root_obj = self._get_or_create_model_empty(collection, root_name, model)

        mesh_name = f'{root_name}_Mesh_NextGen'

        mesh_entries = []
        mesh_objects = []
        if not self.import_armature_only:
            mesh_entries = self._build_mesh_objects(context, filepath, model, collection, mesh_name, force_single_mesh=True)
            mesh_objects = [entry['object'] for entry in mesh_entries]
            for mesh_obj in mesh_objects:
                self._tag_next_gen_mesh_object(
                    mesh_obj,
                    model,
                    render_id=render_id,
                    source_filepath=filepath,
                    model_root_obj=model_root_obj,
                    source_arm_obj=source_arm_obj,
                )

        arm_obj = source_arm_obj
        if arm_obj is not None:
            for mesh_entry in mesh_entries:
                ArmatureBuilder.bind_vertices(mesh_entry['object'], arm_obj, mesh_entry['bind_model'])
        elif model.segments:
            armature_name = f'{root_name}_NextGen_Armature'
            arm_obj = ArmatureBuilder(context, filepath, collection=collection, object_name=armature_name).build(model)
            self._parent_to_model_empty(arm_obj, model_root_obj)
            for mesh_entry in mesh_entries:
                ArmatureBuilder.bind_vertices(mesh_entry['object'], arm_obj, mesh_entry['bind_model'])
            if self.import_bounding_boxes:
                ArmatureBuilder.build_bounds(arm_obj, model)
        elif model_root_obj is not None:
            for mesh_obj in mesh_objects:
                self._parent_to_model_empty(mesh_obj, model_root_obj)

        built_face_count = sum(len(entry.indices) // 3 for entry in model.strips) or len(model.faces)
        return {
            'model': model,
            'mesh_obj': mesh_objects[0] if mesh_objects else None,
            'mesh_objects': mesh_objects,
            'arm_obj': arm_obj,
            'model_root_obj': model_root_obj,
            'faces': built_face_count,
            'next_gen': True,
            'cdc_render_data_id': render_id,
        }

    def _clone_model_subset(self, model, strips):
        from copy import copy, deepcopy

        subset = copy(model)
        subset.faces = []
        used_vertex_indices = sorted({int(index) for strip in strips for index in getattr(strip, 'indices', []) if int(index) >= 0})
        vertex_remap = {source_index: new_index for new_index, source_index in enumerate(used_vertex_indices)}

        subset.vertices = []
        for source_index in used_vertex_indices:
            vertex_copy = deepcopy(model.vertices[source_index])
            vertex_copy.index = vertex_remap[source_index]
            subset.vertices.append(vertex_copy)

        subset.vertex_colors = (
            [model.vertex_colors[source_index] for source_index in used_vertex_indices if source_index < len(model.vertex_colors)]
            if model.vertex_colors else model.vertex_colors
        )
        subset.env_mapped_face_indices = [vertex_remap[index] for index in getattr(model, 'env_mapped_face_indices', []) if index in vertex_remap]
        subset.eye_ref_env_mapped_face_indices = [vertex_remap[index] for index in getattr(model, 'eye_ref_env_mapped_face_indices', []) if index in vertex_remap]

        subset.strips = []
        for strip in strips:
            strip_copy = deepcopy(strip)
            strip_copy.indices = [vertex_remap[int(index)] for index in getattr(strip, 'indices', []) if int(index) in vertex_remap]
            subset.strips.append(strip_copy)

        subset.segments = deepcopy(model.segments)
        for segment in subset.segments:
            remapped = [vertex_remap[source_index] for source_index in used_vertex_indices if segment.first_vertex <= source_index <= segment.last_vertex]
            segment.first_vertex = min(remapped) if remapped else -1
            segment.last_vertex = max(remapped) if remapped else -1

        subset.virt_segments = []
        for virt_segment in model.virt_segments:
            remapped = [vertex_remap[source_index] for source_index in used_vertex_indices if virt_segment.first_vertex <= source_index <= virt_segment.last_vertex]
            if not remapped:
                continue
            virt_copy = deepcopy(virt_segment)
            virt_copy.first_vertex = min(remapped)
            virt_copy.last_vertex = max(remapped)
            subset.virt_segments.append(virt_copy)
        return subset


    @staticmethod
    def _strip_material_group(strip, default_index: int) -> int:
        try:
            group = int(getattr(strip, 'material_group', -1))
        except Exception:
            group = -1
        return group if group >= 0 else default_index

    @staticmethod
    def _material_texture_id_for_name(material) -> int:
        try:
            pc_nextgen_diffuse_texture_id = int(getattr(material, 'trlau_ui_pcng_diffuse_texture_id', material.get('trlau_pc_nextgen_diffuse_texture_id', -1)))
            if pc_nextgen_diffuse_texture_id >= 0:
                return pc_nextgen_diffuse_texture_id & 0xFFFFFFFF
        except Exception:
            pass
        try:
            tr8_diffuse_texture_id = int(material.get('trlau_tr8_diffuse_texture_id', -1))
            if tr8_diffuse_texture_id >= 0:
                return tr8_diffuse_texture_id & 0xFFFFFFFF
        except Exception:
            pass
        try:
            flags = decode_tpage_flags_for_platform(get_material_tpageid(material), get_material_platform_name(material))
            return int(flags.get('texture_id', 0)) & 0xFFFFFFFF
        except Exception:
            pass
        try:
            return int(get_material_tpageid(material)) & 0xFFFFFFFF
        except Exception:
            return 0

    @staticmethod
    def _rename_material_for_strip(material, material_index: int, source_strip_index: int | None = None) -> None:
        try:
            texture_id = ModelImporterMixin._material_texture_id_for_name(material)
            material_index = int(material_index)
            texture_id = int(texture_id) & 0xFFFFFFFF
            material.name = f'Material_{material_index}_{texture_id}'
        except Exception:
            pass
        for stale_key in ('trlau_texture_strip_index', 'trlau_material_group'):
            try:
                if stale_key in material:
                    del material[stale_key]
            except Exception:
                pass

    @staticmethod
    def _material_index_by_group(model) -> dict[int, int]:
        material_index_by_group: dict[int, int] = {}
        for strip_index, strip in enumerate(getattr(model, 'strips', []) or []):
            group = ModelImporterMixin._strip_material_group(strip, strip_index)
            if group not in material_index_by_group:
                material_index_by_group[group] = len(material_index_by_group)
        return material_index_by_group

    @staticmethod
    def _is_underworld_model(model) -> bool:
        uv_format = str(getattr(model, 'uv_format', '') or '').strip().lower()
        if uv_format in {'underworld', 'pc_nextgen'}:
            return True
        try:
            return int(getattr(model, 'version', 0) or 0) == 19
        except Exception:
            return False

    @staticmethod
    def _imports_underworld_materials_only(model) -> bool:
        uv_format = str(getattr(model, 'uv_format', '') or '').strip().lower()
        if uv_format == 'underworld':
            return True
        if uv_format == 'pc_nextgen':
            return False
        try:
            return int(getattr(model, 'version', 0) or 0) == 19
        except Exception:
            return False

    @staticmethod
    def _underworld_batch_key(strip, default_index: int) -> tuple[int, int, int] | None:
        try:
            batch_index = int(getattr(strip, 'tr8_batch_index', -1))
        except Exception:
            batch_index = -1
        try:
            vertex_format_offset = int(getattr(strip, 'tr8_vertex_format_offset', -1))
        except Exception:
            vertex_format_offset = -1
        try:
            geometry_source_index = int(getattr(strip, 'tr8_geometry_source_index', -1))
        except Exception:
            geometry_source_index = -1

        if batch_index < 0 and vertex_format_offset < 0 and geometry_source_index < 0:
            return None
        return (
            batch_index if batch_index >= 0 else int(default_index),
            vertex_format_offset,
            geometry_source_index,
        )

    @staticmethod
    def _store_underworld_batch_metadata(obj, strips) -> None:
        if obj is None or not strips:
            return
        first_strip = strips[0]
        metadata = {
            'trlau_tr8_batch_index': getattr(first_strip, 'tr8_batch_index', -1),
            'trlau_tr8_vertex_format_offset': getattr(first_strip, 'tr8_vertex_format_offset', -1),
            'trlau_tr8_palette_source_index': getattr(first_strip, 'tr8_palette_source_index', -1),
            'trlau_tr8_geometry_source_index': getattr(first_strip, 'tr8_geometry_source_index', -1),
            'trlau_tr8_batch_strip_count': len(strips),
        }
        for key, value in metadata.items():
            try:
                obj[key] = int(value)
            except Exception:
                try:
                    obj[key] = value
                except Exception:
                    pass

    def _build_mesh_objects(self, context, filepath: str, model, collection, mesh_name: str, *, force_single_mesh: bool = False):
        underworld_materials_only = self._imports_underworld_materials_only(model)
        settings = MeshBuildSettings(
            apply_segment_pivots=True,
            import_textures=bool(self.import_textures) and not underworld_materials_only,
        )

        if force_single_mesh or underworld_materials_only or str(getattr(model, 'uv_format', '') or '').strip().lower() == 'pc_nextgen':
            return [{
                'object': MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=mesh_name).build(model),
                'bind_model': model,
            }]

        if self._is_underworld_model(model) and model.strips:
            keyed_strips = []
            for strip_index, strip in enumerate(model.strips):
                batch_key = self._underworld_batch_key(strip, strip_index)
                if batch_key is None:
                    keyed_strips = []
                    break
                keyed_strips.append((batch_key, strip_index, strip))

            batch_keys = {batch_key for batch_key, _strip_index, _strip in keyed_strips}
            if len(batch_keys) > 1 and len(keyed_strips) == len(model.strips):
                results = []

                if self.separate_by_material:
                    grouped: dict[tuple[tuple[int, int, int], int], list] = {}
                    group_source_indices: dict[tuple[tuple[int, int, int], int], int] = {}
                    for batch_key, strip_index, strip in keyed_strips:
                        material_group = self._strip_material_group(strip, strip_index)
                        composite_key = (batch_key, material_group)
                        grouped.setdefault(composite_key, []).append(strip)
                        group_source_indices.setdefault(composite_key, strip_index)

                    material_index_by_group = self._material_index_by_group(model)
                    for group_index, ((_batch_key, material_group), strips) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
                        subset_model = self._clone_model_subset(model, strips)
                        source_strip_index = group_source_indices.get((_batch_key, material_group), group_index)
                        source_material_index = material_index_by_group.get(material_group, group_index)
                        obj = MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=f'{mesh_name}_{group_index}').build(subset_model)
                        self._store_underworld_batch_metadata(obj, strips)
                        for material in getattr(getattr(obj, 'data', None), 'materials', []):
                            self._rename_material_for_strip(material, source_material_index, source_strip_index)
                        results.append({'object': obj, 'bind_model': subset_model})
                    return results

                if self.separate_by_drawgroup:
                    grouped: dict[tuple[tuple[int, int, int], int], list] = {}
                    for batch_key, _strip_index, strip in keyed_strips:
                        grouped.setdefault((batch_key, int(strip.draw_group)), []).append(strip)

                    material_index_by_group = self._material_index_by_group(model)
                    source_strip_index_by_group: dict[int, int] = {}
                    for strip_index, strip in enumerate(model.strips):
                        material_group = self._strip_material_group(strip, strip_index)
                        source_strip_index_by_group.setdefault(material_group, strip_index)

                    for group_index, ((_batch_key, _drawgroup), strips) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
                        subset_model = self._clone_model_subset(model, strips)
                        obj = MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=f'{mesh_name}_{group_index}').build(subset_model)
                        self._store_underworld_batch_metadata(obj, strips)
                        for local_index, material in enumerate(getattr(getattr(obj, 'data', None), 'materials', [])):
                            material_group = material.get('trlau_material_group', None)
                            try:
                                material_group = int(material_group)
                            except Exception:
                                material_group = local_index
                            source_material_index = material_index_by_group.get(material_group, local_index)
                            source_strip_index = source_strip_index_by_group.get(material_group, local_index)
                            self._rename_material_for_strip(material, source_material_index, source_strip_index)
                        results.append({'object': obj, 'bind_model': subset_model})
                    return results

                grouped_by_batch: dict[tuple[int, int, int], list] = {}
                for batch_key, _strip_index, strip in keyed_strips:
                    grouped_by_batch.setdefault(batch_key, []).append(strip)

                for group_index, (_batch_key, strips) in enumerate(sorted(grouped_by_batch.items(), key=lambda item: item[0])):
                    subset_model = self._clone_model_subset(model, strips)
                    obj = MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=f'{mesh_name}_{group_index}').build(subset_model)
                    self._store_underworld_batch_metadata(obj, strips)
                    results.append({'object': obj, 'bind_model': subset_model})
                return results

        if self.separate_by_material and model.strips:
            grouped: dict[int, list] = {}
            group_source_indices: dict[int, int] = {}
            for strip_index, strip in enumerate(model.strips):
                material_group = self._strip_material_group(strip, strip_index)
                grouped.setdefault(material_group, []).append(strip)
                group_source_indices.setdefault(material_group, strip_index)

            material_index_by_group = self._material_index_by_group(model)
            results = []
            for group_index, (material_group, strips) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
                subset_model = self._clone_model_subset(model, strips)
                source_strip_index = group_source_indices.get(material_group, group_index)
                source_material_index = material_index_by_group.get(material_group, group_index)
                obj = MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=f'{mesh_name}_{group_index}').build(subset_model)
                for material in getattr(getattr(obj, 'data', None), 'materials', []):
                    self._rename_material_for_strip(material, source_material_index, source_strip_index)
                results.append({'object': obj, 'bind_model': subset_model})
            return results

        if self.separate_by_drawgroup and model.strips:
            grouped: dict[int, list] = {}
            for strip in model.strips:
                grouped.setdefault(int(strip.draw_group), []).append(strip)

            material_index_by_group = self._material_index_by_group(model)
            source_strip_index_by_group: dict[int, int] = {}
            for strip_index, strip in enumerate(model.strips):
                material_group = self._strip_material_group(strip, strip_index)
                source_strip_index_by_group.setdefault(material_group, strip_index)

            results = []
            for group_index, (_drawgroup, strips) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
                subset_model = self._clone_model_subset(model, strips)
                obj = MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=f'{mesh_name}_{group_index}').build(subset_model)
                for local_index, material in enumerate(getattr(getattr(obj, 'data', None), 'materials', [])):
                    material_group = material.get('trlau_material_group', None)
                    try:
                        material_group = int(material_group)
                    except Exception:
                        material_group = local_index
                    source_material_index = material_index_by_group.get(material_group, local_index)
                    source_strip_index = source_strip_index_by_group.get(material_group, local_index)
                    self._rename_material_for_strip(material, source_material_index, source_strip_index)
                results.append({'object': obj, 'bind_model': subset_model})
            return results

        return [{
            'object': MeshBuilder(context=context, filepath=filepath, settings=settings, collection=collection, object_name=mesh_name).build(model),
            'bind_model': model,
        }]


    @staticmethod
    def _is_psp_shadow_proxy_model(model) -> bool:
        try:
            vertices = list(getattr(model, 'vertices', []) or [])
            faces = list(getattr(model, 'faces', []) or [])
            strips = list(getattr(model, 'strips', []) or [])
            segments = list(getattr(model, 'segments', []) or [])
        except Exception:
            return False

        if len(segments) != 1 or len(strips) != 1:
            return False
        if not (8 <= len(vertices) <= 32 and 6 <= len(faces) <= 24):
            return False

        raw_components = [abs(int(component)) for vertex in vertices for component in getattr(vertex, 'position_raw', (0, 0, 0))]
        if not raw_components:
            return False
        high_component_ratio = sum(1 for value in raw_components if value >= 32000) / float(len(raw_components))
        if high_component_ratio < 0.75:
            return False

        try:
            scale = tuple(float(value) for value in getattr(model, 'model_scale', (0.0, 0.0, 0.0, 1.0))[:3])
        except Exception:
            scale = ()
        if scale and not all(0.0030 <= abs(value) <= 0.0045 for value in scale):
            return False

        try:
            max_rad = float(getattr(model, 'max_rad', 0.0) or 0.0)
            if max_rad and not (150.0 <= max_rad <= 260.0):
                return False
        except Exception:
            pass

        return True

    def _filter_psp_auxiliary_model_refs(self, model_refs):
        refs = list(model_refs or [])
        if str(getattr(self, 'platform', '')).upper() != 'PSP' or len(refs) <= 1:
            return refs

        filtered = []
        for model_ref in refs:
            if int(getattr(model_ref, 'index', 0)) <= 0:
                filtered.append(model_ref)
                continue
            try:
                model = TRPSPModelParser(
                    model_ref.target_filepath,
                    import_hinfo=False,
                    import_markups=False,
                ).parse()
            except Exception as exc:
                logger.debug('Could not inspect PSP auxiliary model %s: %s', model_ref.target_filepath, exc)
                filtered.append(model_ref)
                continue
            if self._is_psp_shadow_proxy_model(model):
                logger.debug('Skipping PSP shadow/proxy model reference %d: %s', model_ref.index, model_ref.target_filepath)
                continue
            filtered.append(model_ref)

        return filtered or refs[:1]

    def _import_from_object_refs(self, context, source_path: Path, object_filepath: str, collection_name: str | None = None, collection=None, object_endian: str = '<', extracted_paths=None, extracted_sections=None):
        collection = collection or self._get_or_create_collection(context, str(source_path), collection_name=collection_name)
        try:
            collection['trlau_source_file'] = str(source_path.resolve())
            collection['trlau_source_dir'] = str(source_path.resolve().parent)
            if source_path.suffix.lower() == '.drm':
                collection['trlau_source_drm'] = str(source_path.resolve())
        except Exception:
            pass
        object_parser = TRObjectParser(object_filepath, endian=object_endian)
        model_refs = object_parser.parse_model_references()
        cloth_setups = []
        if getattr(self, 'import_cloth', True):
            try:
                cloth_setups = object_parser.parse_cloth_setups()
            except Exception as exc:
                logger.warning('Failed to parse cloth setup list from %s: %s', Path(object_filepath).name, exc)
        if not model_refs:
            raise ValueError('No model references were found in the object file')
        if self.import_main_model_only:
            model_refs = model_refs[:1]
        elif str(getattr(self, 'platform', '')).upper() == 'PSP':
            model_refs = self._filter_psp_auxiliary_model_refs(model_refs)

        imported_results = []
        imported_next_gen_ids: set[int] = set()
        for model_ref in model_refs:
            logger.debug('Importing referenced model %d from %s', model_ref.index, model_ref.target_filepath)
            imported_result = self._import_mesh_file(context, model_ref.target_filepath, collection=collection, name_prefix=collection_name or source_path.parent.name or source_path.stem)
            imported_results.append(imported_result)

            if (
                bool(getattr(self, 'import_next_gen_model', False))
                and str(getattr(self, 'platform', '')).upper() == 'PC'
                and object_endian == '<'
            ):
                model = imported_result.get('model')
                render_id = int(getattr(model, 'cdc_render_data_id', 0) or 0)
                if render_id and render_id not in imported_next_gen_ids:
                    next_gen_path = self._find_next_gen_model_path(
                        render_id,
                        extracted_paths=extracted_paths,
                        extracted_sections=extracted_sections,
                        directory=Path(model_ref.target_filepath).parent,
                    )
                    if next_gen_path is not None:
                        logger.info(
                            'Importing next-gen render model cdcRenderDataID=0x%X from %s',
                            render_id,
                            next_gen_path.name,
                        )
                        imported_next_gen_ids.add(render_id)
                        imported_results.append(self._import_next_gen_mesh_file(
                            context,
                            str(next_gen_path),
                            imported_result,
                            collection=collection,
                            name_prefix=collection_name or source_path.parent.name or source_path.stem,
                        ))
                    else:
                        logger.warning(
                            'Import Next Gen Model is enabled, but no section type 7 was found for cdcRenderDataID=0x%X',
                            render_id,
                        )

        if cloth_setups:
            target_armature = next((result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None), None)
            self._import_cloth_setups(collection, target_armature, cloth_setups, source_name=collection_name or source_path.stem)
        return imported_results

    @staticmethod
    def _load_objectlist_mapping(directory: Path) -> dict[int, str]:
        objectlist_path = directory / 'objectlist.txt'
        if not objectlist_path.exists():
            return {}
        mapping: dict[int, str] = {}
        try:
            for raw_line in objectlist_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                line = raw_line.strip()
                if not line or ',' not in line:
                    continue
                object_id_text, object_name = line.split(',', 1)
                try:
                    mapping[int(object_id_text.strip())] = object_name.strip()
                except Exception:
                    continue
        except Exception:
            return {}
        return mapping

    @staticmethod
    def _find_named_drm(directory: Path, object_name: str) -> Path | None:
        exact = directory / f'{object_name}.drm'
        if exact.exists():
            return exact
        lowered = object_name.strip().lower()
        return next((candidate for candidate in directory.glob('*.drm') if candidate.stem.lower() == lowered), None)

    def _import_drm_first_model(self, context, drm_path: Path, collection, *, include_hinfo: bool = False, include_targets: bool = True):
        platform = str(getattr(self, 'platform', '') or '').upper()
        drm_parser = DRMContainerParser(str(drm_path), decompress_derickw=(platform == 'PSP'))
        with drm_parser.temporary_extract_sections() as (extract_dir, extracted_paths, _sections):
            logger.debug('Extracted DRM first-model import source %s into %s (%d sections)', drm_path.name, extract_dir, len(extracted_paths))
            if not extracted_paths:
                raise ValueError('No sections were found in the DRM container')

            object_path = Path(extracted_paths[0])
            finder = getattr(self, '_find_object_root_with_model_refs', None)
            if callable(finder):
                object_path = Path(finder(extracted_paths))

            model_refs = TRObjectParser(str(object_path)).parse_model_references()
            if not model_refs:
                raise ValueError('No model references were found in the object file')

            if platform == 'PSP':
                model_refs = self._filter_psp_auxiliary_model_refs(model_refs)

            previous_skip_targets = getattr(self, '_trlau_skip_targets', False)
            self._trlau_skip_targets = not include_targets
            try:
                with self._temporary_import_overrides(import_hinfo=include_hinfo):
                    return [self._import_mesh_file(context, model_refs[0].target_filepath, collection=collection, name_prefix=collection.name, create_model_root=False)]
            finally:
                self._trlau_skip_targets = previous_skip_targets

    def _import_gamecube_drm(self, context, filepath: str, collection_override=None):
        source_path = Path(filepath)
        collection = collection_override or self._get_or_create_collection(context, str(source_path), collection_name=source_path.stem)

        drm_parser = DRMContainerParser(filepath, endian='>')
        with drm_parser.temporary_extract_sections() as (extract_dir, extracted_paths, _sections):
            logger.debug('Extracted Gamecube DRM %s into %s (%d sections)', source_path.name, extract_dir, len(extracted_paths))
            if not extracted_paths:
                raise ValueError('No sections were found in the Gamecube DRM container')

            object_parser = TRObjectParser(str(extracted_paths[0]), endian='>')
            model_refs = object_parser.parse_model_references()
            if not model_refs:
                raise ValueError('No model references were found in the Gamecube object section')

            cloth_setups = []
            if getattr(self, 'import_cloth', True):
                try:
                    cloth_setups = object_parser.parse_cloth_setups()
                except Exception as exc:
                    logger.warning('Failed to parse Gamecube cloth setup list from %s: %s', Path(extracted_paths[0]).name, exc)

            results = []
            for model_ref in (model_refs[:1] if self.import_main_model_only else model_refs):
                model = TRModelParser(model_ref.target_filepath, import_hinfo=self.import_hinfo, import_markups=getattr(self, 'import_markups', True), endian='>', parse_segments_only=False).parse()
                model_file_stem = Path(model_ref.target_filepath).stem
                model_name_part = model_file_stem.split('_')[0] if '_' in model_file_stem else model_file_stem
                model_root_name = f'{source_path.stem}_{model_name_part}_Model'
                mesh_name = f'{model_root_name}_Mesh'
                armature_name = f'{source_path.stem}_{model_name_part}_Armature'
                model_root_obj = self._get_or_create_model_empty(collection, model_root_name, model)

                mesh_entries = []
                mesh_objects = []
                if not self.import_armature_only and model.vertices and model.faces:
                    mesh_entries = self._build_mesh_objects(context, model_ref.target_filepath, model, collection, mesh_name)
                    mesh_objects = [entry['object'] for entry in mesh_entries]

                arm_obj = None
                if model.segments:
                    arm_obj = ArmatureBuilder(context, model_ref.target_filepath, collection=collection, object_name=armature_name).build(model)
                    self._parent_to_model_empty(arm_obj, model_root_obj)
                    for mesh_entry in mesh_entries:
                        ArmatureBuilder.bind_vertices(mesh_entry['object'], arm_obj, mesh_entry['bind_model'])
                    if self.import_hinfo:
                        print('Importing HInfo...')
                        ArmatureBuilder.build_hmarkers(arm_obj, model)
                        ArmatureBuilder.build_hspheres(arm_obj, model)
                        ArmatureBuilder.build_hboxes(arm_obj, model)
                        ArmatureBuilder.build_hcapsules(arm_obj, model)
                    if self.import_hinfo:
                        ArmatureBuilder.build_targets(arm_obj, model)
                elif model_root_obj is not None:
                    for mesh_obj in mesh_objects:
                        self._parent_to_model_empty(mesh_obj, model_root_obj)

                self._import_model_markups(collection, model, arm_obj=arm_obj, source_name=armature_name if arm_obj is not None else model_root_name)

                results.append({
                    'model': model,
                    'mesh_obj': mesh_objects[0] if mesh_objects else None,
                    'mesh_objects': mesh_objects,
                    'arm_obj': arm_obj,
                    'model_root_obj': model_root_obj,
                    'faces': len(model.faces),
                    'source': str(source_path),
                })
            if cloth_setups:
                target_armature = next((result.get('arm_obj') for result in results if result.get('arm_obj') is not None), None)
                self._import_cloth_setups(collection, target_armature, cloth_setups, source_name=source_path.stem)

            if getattr(self, 'import_all_animations', False):
                results.extend(
                    self._import_all_animation_files(
                        context,
                        extracted_paths,
                        armatures=[result.get('arm_obj') for result in results if result.get('arm_obj') is not None],
                        source_name=source_path.stem,
                        default_endianness='>',
                    )
                )
            return results
