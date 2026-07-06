from __future__ import annotations

import os

import bpy
import mathutils
from mathutils import Euler, Matrix, Vector

from ..core.model_types import HBox, HCapsule, ModelData, Target
from ..core.hinfo_properties import (
    set_hbox_data_props,
    set_hcapsule_data_props,
    set_hmarker_data_props,
    set_hsphere_data_props,
)
from .transform_utils import accumulate_segment_pivot, get_vertex_weights


class ArmatureBuilder:
    def __init__(self, context, filepath: str, collection=None, object_name: str | None = None):
        self.context = context
        self.filepath = filepath
        self.collection = collection or context.collection
        self.object_name = object_name

    def build(self, model: ModelData):
        name = self.object_name or (os.path.splitext(os.path.basename(self.filepath))[0] + "_Armature")
        armature = bpy.data.armatures.new(name)
        arm_obj = bpy.data.objects.new(name, armature)
        self.collection.objects.link(arm_obj)
        armature.display_type = 'STICK'
        arm_obj.show_in_front = True

        previous_active = self.context.view_layer.objects.active
        self.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")

        bones = []
        for segment in model.segments:
            bone = armature.edit_bones.new(f"bone_{segment.index}")
            head = accumulate_segment_pivot(model.segments, segment.index)
            bone.head = head
            bone.tail = (head[0], head[1] + 0.05, head[2])
            bones.append(bone)

        for segment in model.segments:
            if 0 <= segment.parent < len(bones) and segment.parent != segment.index:
                bones[segment.index].parent = bones[segment.parent]

        bpy.ops.object.mode_set(mode="OBJECT")
        self.context.view_layer.objects.active = previous_active

        return arm_obj

    @staticmethod
    def bind_vertices(mesh_obj, arm_obj, model: ModelData):
        assignments: dict[int, list[tuple[int, float]]] = {}
        segment_count = len(model.segments)

        for local_vertex_index, vertex in enumerate(model.vertices):
            weights = get_vertex_weights(model, vertex.index)
            for segment_index, weight in weights.items():
                segment_index = int(segment_index)
                weight = float(weight)
                if segment_index < 0 or segment_index >= segment_count or weight <= 0.0:
                    continue
                assignments.setdefault(segment_index, []).append((local_vertex_index, weight))

        for vertex_group in list(mesh_obj.vertex_groups):
            mesh_obj.vertex_groups.remove(vertex_group)

        group_by_segment: dict[int, bpy.types.VertexGroup] = {}
        for segment_index in sorted(assignments):
            group_by_segment[segment_index] = mesh_obj.vertex_groups.new(name=f"bone_{segment_index}")

        for segment_index, vertex_weights in assignments.items():
            group = group_by_segment.get(segment_index)
            if group is None:
                continue
            for local_vertex_index, weight in vertex_weights:
                group.add([local_vertex_index], weight, "REPLACE")

        modifier = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
        modifier.object = arm_obj
        mesh_obj.parent = arm_obj


    @staticmethod
    def _get_base_name(arm_obj):
        return arm_obj.name[:-len('_Armature')] if arm_obj.name.endswith('_Armature') else arm_obj.name

    @staticmethod
    def _get_collection(arm_obj):
        return arm_obj.users_collection[0] if arm_obj.users_collection else bpy.context.scene.collection

    @staticmethod
    def _get_or_create_empty(collection, name: str, parent=None):
        obj = bpy.data.objects.get(name)
        if obj is None:
            obj = bpy.data.objects.new(name, None)
            obj.empty_display_type = 'PLAIN_AXES'
            obj.empty_display_size = 10.0
        if obj.name not in collection.objects:
            collection.objects.link(obj)
        if parent is not None:
            obj.parent = parent
            obj.matrix_parent_inverse.identity()
        return obj

    @staticmethod
    def _get_or_create_hinfo_objects(arm_obj):
        base_name = ArmatureBuilder._get_base_name(arm_obj)
        collection = ArmatureBuilder._get_collection(arm_obj)
        hinfo_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_HInfo', arm_obj)
        return base_name, collection, hinfo_obj

    @staticmethod
    def build_hmarkers(arm_obj, model: ModelData):
        if not model.hmarkers:
            return None

        base_name, collection, hinfo_obj = ArmatureBuilder._get_or_create_hinfo_objects(arm_obj)
        hmarkers_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_HMarkers', hinfo_obj)

        for marker in model.hmarkers:
            marker_obj = bpy.data.objects.new(f'{base_name}_HMarker_{marker.index}', None)
            marker_obj.empty_display_type = 'ARROWS'
            marker_obj.empty_display_size = 10.0
            collection.objects.link(marker_obj)
            marker_obj.parent = hmarkers_obj
            marker_obj.matrix_parent_inverse.identity()

            bone_index = marker.bone if 0 <= marker.bone < len(model.segments) else marker.owner_segment
            bone_name = f'bone_{bone_index}'
            bone = arm_obj.data.bones.get(bone_name)

            local_transform = Matrix.Translation(Vector(marker.position)) @ Euler(marker.rotation, 'ZYX').to_matrix().to_4x4()
            marker_obj.rotation_mode = 'ZYX'
            if bone is not None:
                marker_obj.matrix_local = bone.matrix_local @ local_transform

                child_of = marker_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLau HMarker Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
                child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ hmarkers_obj.matrix_world
            else:
                owner_head = Vector(accumulate_segment_pivot(model.segments, marker.owner_segment))
                marker_obj.location = owner_head + Vector(marker.position)
                marker_obj.rotation_euler = marker.rotation

            set_hmarker_data_props(marker_obj, marker)

        return hmarkers_obj

    @staticmethod
    def build_hspheres(arm_obj, model: ModelData):
        if not model.hspheres:
            return None

        base_name, collection, hinfo_obj = ArmatureBuilder._get_or_create_hinfo_objects(arm_obj)
        hspheres_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_HSpheres', hinfo_obj)

        for sphere in model.hspheres:
            sphere_obj = bpy.data.objects.new(f'{base_name}_HSphere_{sphere.global_index}', None)
            sphere_obj.empty_display_type = 'SPHERE'
            sphere_obj.empty_display_size = 1.0
            collection.objects.link(sphere_obj)
            sphere_obj.parent = hspheres_obj
            sphere_obj.matrix_parent_inverse.identity()
            sphere_obj.show_in_front = True
            sphere_obj.lock_rotation = (True, True, True)
            sphere_obj.lock_rotation_w = True
            sphere_obj.lock_rotations_4d = True

            bone_index = sphere.owner_segment if 0 <= sphere.owner_segment < len(model.segments) else 0
            bone_name = f'bone_{bone_index}'
            bone = arm_obj.data.bones.get(bone_name)
            local_position = Vector((sphere.x, sphere.y, sphere.z))

            if bone is not None:
                sphere_obj.matrix_local = bone.matrix_local @ Matrix.Translation(local_position)
                child_of = sphere_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLau HSphere Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
                child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ hspheres_obj.matrix_world
            else:
                owner_head = Vector(accumulate_segment_pivot(model.segments, sphere.owner_segment))
                sphere_obj.location = owner_head + local_position

            sphere_obj.scale = (float(sphere.radius), float(sphere.radius), float(sphere.radius))
            set_hsphere_data_props(sphere_obj, sphere)

        return hspheres_obj


    @staticmethod
    def build_hboxes(arm_obj, model: ModelData):
        if not getattr(model, "hboxes", None):
            return None

        base_name, collection, hinfo_obj = ArmatureBuilder._get_or_create_hinfo_objects(arm_obj)
        hboxes_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_HBoxes', hinfo_obj)

        for hbox in model.hboxes:
            hbox_obj = bpy.data.objects.new(f'{base_name}_HBox_{hbox.global_index}', None)
            hbox_obj.empty_display_type = 'CUBE'
            hbox_obj.empty_display_size = 1.0
            collection.objects.link(hbox_obj)
            hbox_obj.parent = hboxes_obj
            hbox_obj.matrix_parent_inverse.identity()
            hbox_obj.show_in_front = True
            hbox_obj.rotation_mode = 'QUATERNION'

            bone_index = hbox.owner_segment if 0 <= hbox.owner_segment < len(model.segments) else 0
            bone_name = f'bone_{bone_index}'
            bone = arm_obj.data.bones.get(bone_name)

            position = Vector(tuple(float(v) for v in hbox.position[:3]))
            quaternion = tuple(float(v) for v in hbox.quaternion)
            rotation = Matrix(
                mathutils.Quaternion((quaternion[3], quaternion[0], quaternion[1], quaternion[2])).to_matrix()
            ).to_4x4()
            local_matrix = Matrix.Translation(position) @ rotation

            if bone is not None:
                hbox_obj.matrix_local = bone.matrix_local @ local_matrix
                child_of = hbox_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLau HBox Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
                child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ hboxes_obj.matrix_world
            else:
                owner_head = Vector(accumulate_segment_pivot(model.segments, hbox.owner_segment))
                hbox_obj.matrix_local = Matrix.Translation(owner_head) @ local_matrix

            hbox_obj.scale = (
                max(float(hbox.dimensions[0]) * 0.5, 1e-6),
                max(float(hbox.dimensions[1]) * 0.5, 1e-6),
                max(float(hbox.dimensions[2]) * 0.5, 1e-6),
            )
            set_hbox_data_props(hbox_obj, hbox)

        return hboxes_obj




    @staticmethod
    def build_bounds(arm_obj, model: ModelData):
        if not getattr(model, "segments", None):
            return None

        base_name = ArmatureBuilder._get_base_name(arm_obj)
        collection = ArmatureBuilder._get_collection(arm_obj)
        bounds_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_Bounds', arm_obj)

        for segment in model.segments:
            bound_obj = bpy.data.objects.new(f'{base_name}_Bound_{segment.index}', None)
            bound_obj.empty_display_type = 'CUBE'
            bound_obj.empty_display_size = 1.0
            collection.objects.link(bound_obj)
            bound_obj.parent = bounds_obj
            bound_obj.matrix_parent_inverse.identity()
            bound_obj.show_in_front = True

            bone_name = f'bone_{segment.index}'
            bone = arm_obj.data.bones.get(bone_name)

            min_v = Vector(tuple(float(v) for v in segment.min_v[:3]))
            max_v = Vector(tuple(float(v) for v in segment.max_v[:3]))
            center = (min_v + max_v) * 0.5
            dimensions = max_v - min_v

            if bone is not None:
                bound_obj.matrix_local = bone.matrix_local @ Matrix.Translation(center)
                child_of = bound_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLau Bound Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
                child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ bounds_obj.matrix_world
            else:
                owner_head = Vector(accumulate_segment_pivot(model.segments, segment.index))
                bound_obj.matrix_local = Matrix.Translation(owner_head + center)

            bound_obj.scale = (
                max(float(dimensions[0]) * 0.5, 1e-6),
                max(float(dimensions[1]) * 0.5, 1e-6),
                max(float(dimensions[2]) * 0.5, 1e-6),
            )
            bound_obj['trlau_type'] = 'Bound'
            bound_obj['trlau_owner_segment'] = int(segment.index)
            bound_obj['trlau_bound_min'] = tuple(float(v) for v in segment.min_v[:3])
            bound_obj['trlau_bound_max'] = tuple(float(v) for v in segment.max_v[:3])

        return bounds_obj

    @staticmethod
    def build_targets(arm_obj, model: ModelData):
        if not getattr(model, "targets", None):
            return None

        base_name = ArmatureBuilder._get_base_name(arm_obj)
        collection = ArmatureBuilder._get_collection(arm_obj)
        targets_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_Targets', arm_obj)

        for target in model.targets:
            target_obj = bpy.data.objects.new(f'{base_name}_Target_{target.global_index}', None)
            target_obj.empty_display_type = 'CIRCLE'
            target_obj.empty_display_size = 100.0
            collection.objects.link(target_obj)
            target_obj.parent = targets_obj
            target_obj.matrix_parent_inverse.identity()
            target_obj.show_in_front = True

            bone_index = target.segment if 0 <= target.segment < len(model.segments) else 0
            bone_name = f'bone_{bone_index}'
            bone = arm_obj.data.bones.get(bone_name)

            local_transform = Matrix.Translation(Vector(target.position)) @ Euler(target.rotation, 'ZYX').to_matrix().to_4x4()
            target_obj.rotation_mode = 'ZYX'
            if bone is not None:
                target_obj.matrix_local = bone.matrix_local @ local_transform

                child_of = target_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLau Target Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
                child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ targets_obj.matrix_world
            else:
                owner_head = Vector(accumulate_segment_pivot(model.segments, bone_index))
                target_obj.matrix_local = Matrix.Translation(owner_head) @ local_transform

            target_obj["trlau_segment"] = int(target.segment)
            target_obj["trlau_target_flags"] = int(target.flags)
            target_obj["trlau_target_position"] = tuple(float(v) for v in target.position)
            target_obj["trlau_target_rotation"] = tuple(float(v) for v in target.rotation)
            target_obj["trlau_target_unique_id"] = int(target.unique_id)

        return targets_obj

    @staticmethod
    def build_hcapsules(arm_obj, model: ModelData):
        if not getattr(model, "hcapsules", None):
            return None

        base_name, collection, hinfo_obj = ArmatureBuilder._get_or_create_hinfo_objects(arm_obj)
        hcapsules_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_HCapsules', hinfo_obj)

        for capsule in model.hcapsules:
            capsule_obj = bpy.data.objects.new(f'{base_name}_HCapsule_{capsule.global_index}', None)
            capsule_obj.empty_display_type = 'SPHERE'
            capsule_obj.empty_display_size = 1.0
            collection.objects.link(capsule_obj)
            capsule_obj.parent = hcapsules_obj
            capsule_obj.matrix_parent_inverse.identity()
            capsule_obj.show_in_front = True
            capsule_obj.lock_rotation = (False, False, False)
            capsule_obj.lock_rotation_w = False
            capsule_obj.lock_rotations_4d = False

            bone_index = capsule.owner_segment if 0 <= capsule.owner_segment < len(model.segments) else 0
            bone_name = f'bone_{bone_index}'
            bone = arm_obj.data.bones.get(bone_name)

            start = Vector(tuple(float(v) for v in capsule.start))
            end = Vector(tuple(float(v) for v in capsule.end))
            length = max(float(capsule.length), (end - start).length)
            position = Vector(tuple(float(v) for v in capsule.position[:3]))
            quaternion = tuple(float(v) for v in capsule.quaternion)
            rotation = mathutils.Quaternion((quaternion[3], quaternion[0], quaternion[1], quaternion[2])).to_matrix().to_4x4()
            local_matrix = Matrix.Translation(position) @ rotation

            if bone is not None:
                capsule_obj.matrix_local = bone.matrix_local @ local_matrix
                child_of = capsule_obj.constraints.new(type='CHILD_OF')
                child_of.name = 'TRLau HCapsule Bone Follow'
                child_of.target = arm_obj
                child_of.subtarget = bone_name
                child_of.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ hcapsules_obj.matrix_world
            else:
                owner_head = Vector(accumulate_segment_pivot(model.segments, capsule.owner_segment))
                capsule_obj.matrix_local = Matrix.Translation(owner_head) @ local_matrix

            capsule_obj.scale = (float(capsule.radius), float(capsule.radius), max(length * 0.5, 1e-6))
            set_hcapsule_data_props(capsule_obj, capsule)

        return hcapsules_obj
