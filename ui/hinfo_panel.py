from __future__ import annotations

from ..core.object_utils import trlau_object_type
import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty
from bpy.types import OperatorFileListElement

from ..builders.armature_builder import ArmatureBuilder
from ..core.hinfo_ui import HINFO_TYPES
from ..core.hinfo_properties import HINFO_DATA_FIELD_DEFS, ensure_hinfo_data_props, get_hinfo_data
from ..core.import_options import sync_separation_flags
from .level_properties import MODEL_TARGET_FLAG_ITEMS
from .model_common import MODEL_IMPORT_PLATFORM_ITEMS, _find_model_armature, _iter_object_descendants, _parse_main_model_source, _selected_model_root
from .model_panel import _sync_bone_mirror_entries_from_model



HINFO_COMPONENT_TYPE_ITEMS = (
    ('HMarker', 'HMarker', 'Create an HMarker on the active bone'),
    ('HSphere', 'HSphere', 'Create an HSphere on the active bone'),
    ('HCapsule', 'HCapsule', 'Create an HCapsule on the active bone'),
    ('HBox', 'HBox', 'Create an HBox on the active bone'),
)

class TRLAU_OT_snap_object_to_hmarker(bpy.types.Operator):
    bl_idname = 'trlau.snap_object_to_hmarker'
    bl_label = 'Snap Object to HMarker'
    bl_description = 'Snap selected object(s) to the active HMarker and parent them to it, or import a model and snap it automatically'
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def _sync_separation_flags(self, changed: str):
        sync_separation_flags(self, changed)

    def _update_separate_by_drawgroup(self, _context):
        self._sync_separation_flags(self, 'drawgroup')

    def _update_separate_by_material(self, _context):
        self._sync_separation_flags(self, 'material')

    filename_ext = '.tr7aemesh'
    filter_glob: StringProperty(
        default='*.obj;*.tr7aemesh;*.drm',
        options={'HIDDEN'},
    )
    filepath: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    files: CollectionProperty(
        type=OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    platform: EnumProperty(
        name='Platform',
        items=MODEL_IMPORT_PLATFORM_ITEMS,
        default='PC',
    )
    show_advanced_settings: BoolProperty(
        name='Advanced Settings',
        default=False,
        options={'SKIP_SAVE'},
    )

    import_textures: BoolProperty(name='Import Textures', default=True, description='Enable or disable texture importing')
    import_hinfo: BoolProperty(name='Import HInfo', default=False, description='Import HInfo components such as HMarkers, HSpheres, HBoxes, and HCapsules')
    import_main_model_only: BoolProperty(name='Main Model Only', default=True, description='Only import the first referenced model (.drm, .obj)')
    import_all_textures: BoolProperty(name='Import All Textures', default=False, description='Import all textures found in the .drm or folder')
    import_all_animations: BoolProperty(name='Import All Animations', default=False, description='Import all animations found in the .drm or folder')
    import_armature_only: BoolProperty(name='Import Armature Only', default=False, description='Import only the armature')
    import_bounding_boxes: BoolProperty(name='Import Bounding Boxes', default=False, description="Import bones' bounding boxes")
    separate_by_drawgroup: BoolProperty(name='Separate by Drawgroup', default=False, description='Split the imported model into separate meshes by drawgroup', update=_update_separate_by_drawgroup)
    separate_by_material: BoolProperty(name='Separate by Material', default=False, description='Split the imported model into separate meshes by texture strip/material', update=_update_separate_by_material)
    debug: BoolProperty(name='Debug', default=False, description='Enable console debug log')

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, 'platform')
        layout.prop(self, 'import_textures')
        layout.prop(self, 'separate_by_material')

    @classmethod
    def poll(cls, context):
        marker = context.object
        return marker is not None and trlau_object_type(marker) == 'HMarker'

    def invoke(self, context, _event):
        marker = context.object
        targets = [obj for obj in context.selected_objects if obj != marker]
        if targets:
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    @staticmethod
    def _mark_targets_as_hmarker_attachments(marker, targets):
        marker_name = str(getattr(marker, 'name', '') or '')
        marked = 0
        for root in list(targets or []):
            if root is None or root == marker:
                continue
            objects = [root]
            try:
                objects.extend(list(getattr(root, 'children_recursive', []) or []))
            except Exception:
                pass
            for obj in objects:
                if obj is None or obj == marker:
                    continue
                try:
                    obj['trlau_export_ignore'] = True
                    obj['trlau_hmarker_attachment'] = True
                    if marker_name:
                        obj['trlau_attached_hmarker'] = marker_name
                    marked += 1
                except Exception:
                    continue
        return marked

    @staticmethod
    def _snap_targets_to_marker(marker, targets):
        marker_world = marker.matrix_world.copy()
        snapped = 0
        for obj in targets:
            if obj == marker:
                continue
            try:
                obj.parent = None
                obj.matrix_world = marker_world
                obj.parent = marker
                obj.matrix_parent_inverse = marker.matrix_world.inverted()
                snapped += 1
            except Exception:
                continue
        return snapped

    def execute(self, context):
        marker = bpy.data.objects.get(context.object.name) if context.object is not None else None
        if marker is None or trlau_object_type(marker) != 'HMarker':
            self.report({'ERROR'}, 'Active HMarker is no longer available')
            return {'CANCELLED'}

        targets = [obj for obj in context.selected_objects if obj != marker]
        if targets:
            self._mark_targets_as_hmarker_attachments(marker, targets)
            snapped = self._snap_targets_to_marker(marker, targets)
            if snapped == 0:
                self.report({'WARNING'}, 'No objects were snapped to the HMarker')
                return {'CANCELLED'}
            return {'FINISHED'}

        if not self.filepath:
            self.report({'WARNING'}, 'Select a .drm, .obj, or .tr7aemesh file to import and snap')
            return {'CANCELLED'}

        existing_object_names = {obj.name for obj in bpy.data.objects}
        target_collection = next(iter(getattr(marker, 'users_collection', []) or []), None)
        if target_collection is None:
            target_collection = getattr(context, 'collection', None) or getattr(context.scene, 'collection', None)
        result = bpy.ops.import_scene.trlau_model(
            filepath=self.filepath,
            platform=self.platform,
            import_textures=self.import_textures,
            import_hinfo=False,
            use_integrated_import_defaults=False,
            create_import_collection=False,
            import_collection_name=str(getattr(target_collection, 'name', '') or ''),
            import_main_model_only=self.import_main_model_only,
            import_all_textures=self.import_all_textures,
            import_all_animations=self.import_all_animations,
            import_armature_only=self.import_armature_only,
            import_bounding_boxes=self.import_bounding_boxes,
            separate_by_drawgroup=self.separate_by_drawgroup,
            separate_by_material=self.separate_by_material,
            debug=self.debug,
            force_unique_import_names=True,
        )
        if 'FINISHED' not in result:
            self.report({'ERROR'}, 'Import failed')
            return {'CANCELLED'}

        imported_objects = [
            obj for obj in bpy.data.objects
            if obj.name not in existing_object_names and obj.type in {'MESH', 'ARMATURE', 'EMPTY'}
        ]
        top_level_imports = [obj for obj in imported_objects if obj.parent is None]
        targets = top_level_imports or imported_objects
        if not targets:
            self.report({'WARNING'}, 'Import finished, but no new objects were found to snap')
            return {'CANCELLED'}

        self._mark_targets_as_hmarker_attachments(marker, targets)
        snapped = self._snap_targets_to_marker(marker, targets)
        if snapped == 0:
            self.report({'WARNING'}, 'Imported objects could not be snapped to the HMarker')
            return {'CANCELLED'}

        return {'FINISHED'}

def _remove_object_tree(root) -> int:
    if root is None:
        return 0
    objects = list(_iter_object_descendants(root))
    objects.append(root)
    removed = 0
    for obj in reversed(objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        except Exception:
            pass
    return removed


def _clear_existing_hinfo_objects(arm_obj) -> int:
    if arm_obj is None:
        return 0
    base_name = ArmatureBuilder._get_base_name(arm_obj)
    expected_name = f'{base_name}_HInfo'
    removed = 0
    for child in list(getattr(arm_obj, 'children', []) or []):
        if child.name == expected_name or child.name.endswith('_HInfo') or str(child.get('trlau_component_name', '') or '') == 'HInfo':
            removed += _remove_object_tree(child)
    return removed


def _import_hinfo_to_armature(arm_obj, model, *, replace_existing: bool = True) -> int:
    if arm_obj is None or getattr(arm_obj, 'type', None) != 'ARMATURE':
        raise ValueError('The selected model has no armature')
    if replace_existing:
        _clear_existing_hinfo_objects(arm_obj)
    before = len(bpy.data.objects)
    ArmatureBuilder.build_hmarkers(arm_obj, model)
    ArmatureBuilder.build_hspheres(arm_obj, model)
    ArmatureBuilder.build_hboxes(arm_obj, model)
    ArmatureBuilder.build_hcapsules(arm_obj, model)
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    return max(0, len(bpy.data.objects) - before)

def _parse_bone_index_from_name(name: str):
    text = str(name or '')
    if text.startswith('bone_') and text[5:].isdigit():
        return int(text[5:])
    return None


def _bone_index_for_name(arm_obj, name: str) -> int:
    parsed = _parse_bone_index_from_name(name)
    if parsed is not None:
        return parsed
    for index, bone in enumerate(getattr(arm_obj.data, 'bones', []) or []):
        if bone.name == name:
            return index
    return 0


def _active_armature_and_bone(context):
    arm_obj = getattr(context, 'object', None)
    if arm_obj is None or getattr(arm_obj, 'type', None) != 'ARMATURE':
        return None, None, 0

    mode = str(getattr(context, 'mode', '') or '')
    bone_name = None
    if mode == 'POSE':
        pose_bone = getattr(context, 'active_pose_bone', None)
        if pose_bone is not None:
            bone_name = pose_bone.name
    elif mode == 'EDIT_ARMATURE':
        edit_bone = getattr(getattr(arm_obj.data, 'edit_bones', None), 'active', None)
        if edit_bone is not None:
            bone_name = edit_bone.name
    if not bone_name:
        active_bone = getattr(arm_obj.data, 'bones', None).active if getattr(arm_obj.data, 'bones', None) is not None else None
        if active_bone is not None:
            bone_name = active_bone.name
    if not bone_name:
        return arm_obj, None, 0
    return arm_obj, bone_name, _bone_index_for_name(arm_obj, bone_name)


def _next_hinfo_global_index(arm_obj, component_type: str) -> int:
    max_index = -1
    for obj in _iter_object_descendants(arm_obj):
        if trlau_object_type(obj) != component_type:
            continue
        name = str(obj.name or '')
        suffix = name.rsplit('_', 1)[-1]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))
    return max_index + 1


def _create_child_of_constraint(obj, component_type: str, arm_obj, bone_name: str, group_obj):
    constraint = obj.constraints.new(type='CHILD_OF')
    constraint.name = {
        'HMarker': 'TRLau HMarker Bone Follow',
        'HSphere': 'TRLau HSphere Bone Follow',
        'HBox': 'TRLau HBox Bone Follow',
        'HCapsule': 'TRLau HCapsule Bone Follow',
    }.get(component_type, 'TRLau HInfo Bone Follow')
    constraint.target = arm_obj
    constraint.subtarget = bone_name
    bone = arm_obj.data.bones.get(bone_name) if arm_obj.data else None
    if bone is not None:
        try:
            constraint.inverse_matrix = (arm_obj.matrix_world @ bone.matrix_local).inverted() @ group_obj.matrix_world
        except Exception:
            pass
    return constraint


def _create_hinfo_component_object(context, component_type: str):
    arm_obj, bone_name, bone_index = _active_armature_and_bone(context)
    if arm_obj is None or bone_name is None:
        raise ValueError('Select an active armature bone in Pose Mode or Edit Mode')

    try:
        if str(getattr(context, 'mode', '') or '') != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    base_name, collection, hinfo_obj = ArmatureBuilder._get_or_create_hinfo_objects(arm_obj)
    group_names = {
        'HMarker': 'HMarkers',
        'HSphere': 'HSpheres',
        'HBox': 'HBoxes',
        'HCapsule': 'HCapsules',
    }
    group_obj = ArmatureBuilder._get_or_create_empty(collection, f'{base_name}_{group_names[component_type]}', hinfo_obj)

    global_index = _next_hinfo_global_index(arm_obj, component_type)
    obj = bpy.data.objects.new(f'{base_name}_{component_type}_{global_index}', None)
    collection.objects.link(obj)
    obj.parent = group_obj
    obj.matrix_parent_inverse.identity()
    obj.show_in_front = True

    bone = arm_obj.data.bones.get(bone_name)
    if bone is not None:
        obj.matrix_local = bone.matrix_local.copy()

    if component_type == 'HMarker':
        obj.empty_display_type = 'ARROWS'
        obj.empty_display_size = 10.0
        obj.rotation_mode = 'ZYX'
    elif component_type == 'HSphere':
        radius = 64
        obj.empty_display_type = 'SPHERE'
        obj.empty_display_size = 1.0
        obj.lock_rotation = (True, True, True)
        obj.lock_rotation_w = True
        obj.lock_rotations_4d = True
        obj.scale = (float(radius), float(radius), float(radius))
    elif component_type == 'HBox':
        obj.empty_display_type = 'CUBE'
        obj.empty_display_size = 1.0
        obj.rotation_mode = 'QUATERNION'
        obj.scale = (64.0, 64.0, 64.0)
    elif component_type == 'HCapsule':
        radius = 32
        length = 128
        obj.empty_display_type = 'SPHERE'
        obj.empty_display_size = 1.0
        obj.lock_rotation = (False, False, False)
        obj.lock_rotation_w = False
        obj.lock_rotations_4d = False
        obj.scale = (float(radius), float(radius), float(length) * 0.5)

    _create_child_of_constraint(obj, component_type, arm_obj, bone_name, group_obj)
    data_defaults = {
        f'trlau_{component_type.lower()}_id': int(global_index),
        'trlau_marker_index': int(global_index),
    }
    if component_type == 'HSphere':
        data_defaults['trlau_hsphere_radius_sq'] = int(radius) * int(radius)
    ensure_hinfo_data_props(obj, component_type, data_defaults)
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except Exception:
        pass
    obj.select_set(True)
    context.view_layer.objects.active = obj
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    return obj


class TRLAU_OT_import_model_hinfo(bpy.types.Operator):
    bl_idname = 'trlau.import_model_hinfo'
    bl_label = 'Import HInfo'
    bl_description = 'Import HInfo from a .tr7aemesh, .tr8mesh, or first referenced model of a .drm file'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.tr7aemesh'
    filter_glob: StringProperty(default='*.tr7aemesh;*.tr8mesh;*.drm', options={'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    platform: EnumProperty(name='Platform', items=MODEL_IMPORT_PLATFORM_ITEMS, default='PC')
    replace_existing: BoolProperty(name='Replace Existing HInfo', default=True)

    @classmethod
    def poll(cls, context):
        return _selected_model_root(context) is not None

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, _context):
        self.layout.prop(self, 'platform')
        self.layout.prop(self, 'replace_existing')

    def execute(self, context):
        model_root = _selected_model_root(context)
        if model_root is None:
            self.report({'WARNING'}, 'Select a Model empty')
            return {'CANCELLED'}
        arm_obj = _find_model_armature(model_root)
        if arm_obj is None:
            self.report({'WARNING'}, 'Selected model has no armature')
            return {'CANCELLED'}
        if not self.filepath:
            self.report({'WARNING'}, 'Select a .tr7aemesh, .tr8mesh, or .drm file')
            return {'CANCELLED'}
        try:
            model = _parse_main_model_source(self.filepath, self.platform, import_hinfo=True)
            hinfo_count = sum(len(getattr(model, attr, []) or []) for attr in ('hmarkers', 'hspheres', 'hboxes', 'hcapsules'))
            if hinfo_count <= 0:
                self.report({'WARNING'}, 'No HInfo entries found in source model')
                return {'CANCELLED'}
            created_count = _import_hinfo_to_armature(arm_obj, model, replace_existing=self.replace_existing)
        except Exception as exc:
            self.report({'ERROR'}, f'Import HInfo failed: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Imported {hinfo_count} HInfo entr{ "y" if hinfo_count == 1 else "ies" } ({created_count} object(s) created)')
        return {'FINISHED'}


class TRLAU_OT_import_model_bone_mirrors(bpy.types.Operator):
    bl_idname = 'trlau.import_model_bone_mirrors'
    bl_label = 'Import Bone Mirrors'
    bl_description = 'Import Bone Mirrors from a .tr7aemesh, .tr8mesh, or first referenced model of a .drm file'
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.tr7aemesh'
    filter_glob: StringProperty(default='*.tr7aemesh;*.tr8mesh;*.drm', options={'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE'})
    platform: EnumProperty(name='Platform', items=MODEL_IMPORT_PLATFORM_ITEMS, default='PC')

    @classmethod
    def poll(cls, context):
        return _selected_model_root(context) is not None

    def invoke(self, context, _event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, _context):
        self.layout.prop(self, 'platform')

    def execute(self, context):
        model_root = _selected_model_root(context)
        if model_root is None:
            self.report({'WARNING'}, 'Select a Model empty')
            return {'CANCELLED'}
        if not self.filepath:
            self.report({'WARNING'}, 'Select a .tr7aemesh, .tr8mesh, or .drm file')
            return {'CANCELLED'}
        try:
            model = _parse_main_model_source(self.filepath, self.platform, import_hinfo=False)
            count = len(getattr(model, 'bone_mirror_entries', []) or [])
            if count <= 0:
                self.report({'WARNING'}, 'No Bone Mirror entries found in source model')
                return {'CANCELLED'}
            _sync_bone_mirror_entries_from_model(model_root, model)
        except Exception as exc:
            self.report({'ERROR'}, f'Import Bone Mirrors failed: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Imported {count} Bone Mirror entr{ "y" if count == 1 else "ies" }')
        return {'FINISHED'}


class TRLAU_OT_add_hinfo_component(bpy.types.Operator):
    bl_idname = 'trlau.add_hinfo_component'
    bl_label = 'Add HInfo Component'
    bl_description = 'Create an HInfo component on the active armature bone'
    bl_options = {'REGISTER', 'UNDO'}

    component_type: EnumProperty(name='Type', items=HINFO_COMPONENT_TYPE_ITEMS, default='HMarker')

    @classmethod
    def poll(cls, context):
        arm_obj = getattr(context, 'object', None)
        return arm_obj is not None and getattr(arm_obj, 'type', None) == 'ARMATURE' and str(getattr(context, 'mode', '') or '') in {'POSE', 'EDIT_ARMATURE'}

    def execute(self, context):
        try:
            obj = _create_hinfo_component_object(context, self.component_type)
        except Exception as exc:
            self.report({'ERROR'}, f'Add HInfo failed: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Added {self.component_type} {obj.name}')
        return {'FINISHED'}


class VIEW3D_PT_trlau_hinfo_creation(bpy.types.Panel):
    bl_label = 'HInfo Creation'
    bl_idname = 'VIEW3D_PT_trlau_hinfo_creation'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'

    @classmethod
    def poll(cls, context):
        arm_obj = getattr(context, 'object', None)
        return arm_obj is not None and getattr(arm_obj, 'type', None) == 'ARMATURE' and str(getattr(context, 'mode', '') or '') in {'POSE', 'EDIT_ARMATURE'}

    def draw(self, context):
        _draw_hinfo_creation_content(self.layout, context)


class TRLAU_OT_toggle_component_visibility(bpy.types.Operator):
    bl_idname = 'trlau.toggle_component_visibility'
    bl_label = 'Toggle Component Visibility'
    bl_description = 'Hide or unhide high-level asset data'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        roots = []
        direct_component_objects = []

        root_names = {'HInfo', 'Targets', 'Bounds', 'AreaDBase', 'CameraData', 'Markup', 'Signals', 'TerrainLight'}
        root_suffixes = tuple(f'_{name}' for name in root_names)
        cloth_component_types = {
            'ClothCollision', 'ClothPoint', 'ClothCapsuleEndpoint',
            'ClothCollisionRules', 'ClothCollisionRule',
            'ClothPlaneRules', 'ClothPlaneRule', 'ClothPlaneRulePlane', 'ClothPlaneRuleSelector',
        }
        cloth_component_root_types = {'ClothCollisionRules', 'ClothPlaneRules', 'ClothPlaneRule'}
        signal_types = {'Signals', 'Signal', 'SignalMesh'}

        def _is_named_component_root(obj, name: str) -> bool:
            if name in root_names or name.endswith(root_suffixes):
                return True
            try:
                if bool(obj.get('trlau_component_empty')) and str(obj.get('trlau_component_name', '') or '') in root_names:
                    return True
            except Exception:
                pass
            return False

        def _is_terrain_collision_component(obj, name: str) -> bool:
            try:
                if (
                    bool(obj.get('trlau_terrain_collision'))
                    or bool(obj.get('trlau_terrain_collision_empty'))
                    or bool(obj.get('trlau_terrain_collision_kd'))
                    or bool(obj.get('trlau_terrain_collision_kd_node'))
                ):
                    return True
            except Exception:
                pass
            return 'TerrainGroup' in name and 'Collision' in name

        def _is_cloth_component(obj, name: str) -> bool:
            try:
                obj_type = trlau_object_type(obj)
                if obj_type in cloth_component_types:
                    return True
            except Exception:
                pass
            return (
                '_Cloth_CollisionRules' in name
                or '_Cloth_CollisionRule_' in name
                or '_Cloth_PlaneRules' in name
                or '_Cloth_PlaneRule_' in name
            )

        def _is_cloth_component_root(obj, name: str) -> bool:
            try:
                obj_type = trlau_object_type(obj)
                if obj_type in cloth_component_root_types:
                    return True
            except Exception:
                pass
            return '_Cloth_CollisionRules' in name or '_Cloth_PlaneRules' in name or '_Cloth_PlaneRule_' in name

        def _is_signal_component(obj, name: str) -> bool:
            try:
                obj_type = trlau_object_type(obj)
                if obj_type in signal_types:
                    return True
                if bool(obj.get('trlau_signal')) or bool(obj.get('trlau_signal_mesh')) or bool(obj.get('trlau_terrain_signals')):
                    return True
            except Exception:
                pass
            return name == 'Signals' or name.endswith('_Signals') or '_Signal_' in name or name.endswith('_Signal_Mesh')

        def _is_intro_model_helper(obj, name: str) -> bool:
            obj_kind = str(getattr(obj, 'type', '') or '')
            if obj_kind not in {'EMPTY', 'ARMATURE'}:
                return False
            try:
                if bool(obj.get('trlau_intro_model_helper')):
                    return True
            except Exception:
                pass

            parent = getattr(obj, 'parent', None)
            while parent is not None:
                parent_name = str(getattr(parent, 'name', '') or '')
                try:
                    parent_type = str(parent.get('trlau_type', '') or '')
                    if parent_type == 'IntroData' and (parent_name.startswith('Intro_') or '_Intro_' in parent_name or 'IntroData' in parent_name):
                        return True
                except Exception:
                    pass
                if parent_name.endswith('_IntroData') or parent_name == 'IntroData' or parent_name.startswith('Intro_') or '_Intro_' in parent_name:
                    return True
                parent = getattr(parent, 'parent', None)

            return False

        for obj in bpy.data.objects:
            name = str(getattr(obj, 'name', '') or '')
            try:
                if _is_named_component_root(obj, name):
                    roots.append(obj)
                if _is_cloth_component_root(obj, name):
                    roots.append(obj)
                elif _is_cloth_component(obj, name):
                    direct_component_objects.append(obj)
                if _is_signal_component(obj, name):
                    direct_component_objects.append(obj)
                if _is_intro_model_helper(obj, name):
                    direct_component_objects.append(obj)
                if _is_terrain_collision_component(obj, name):
                    roots.append(obj)
            except Exception:
                continue

        if not roots and not direct_component_objects:
            self.report({'WARNING'}, 'No imported TRLAU helper components found')
            return {'CANCELLED'}

        component_objects = []
        seen = set()

        def _add_component_object(obj):
            name = str(getattr(obj, 'name', '') or '')
            key = name or str(id(obj))
            if key in seen:
                return False
            seen.add(key)
            component_objects.append(obj)
            return True

        for root in roots:
            _add_component_object(root)
        for component_obj in direct_component_objects:
            _add_component_object(component_obj)

        stack = list(roots)
        while stack:
            obj = stack.pop()
            for child in obj.children:
                if not _add_component_object(child):
                    continue
                stack.append(child)

        if not component_objects:
            self.report({'WARNING'}, 'No TRLAU helper components found to toggle')
            return {'CANCELLED'}

        view_layer_objects = getattr(getattr(context, 'view_layer', None), 'objects', None)

        def _is_in_view_layer(obj) -> bool:
            if view_layer_objects is None:
                return True
            try:
                return view_layer_objects.get(obj.name) is obj
            except Exception:
                try:
                    return obj.name in view_layer_objects
                except Exception:
                    return True

        editable_objects = [obj for obj in component_objects if _is_in_view_layer(obj)]
        if not editable_objects:
            self.report({'WARNING'}, 'TRLAU components are in hidden or excluded collections in the active view layer')
            return {'CANCELLED'}

        def _is_component_hidden(obj):
            try:
                return bool(obj.hide_get(view_layer=context.view_layer))
            except TypeError:
                try:
                    return bool(obj.hide_get())
                except Exception:
                    return False
            except Exception:
                return False

        should_hide = any(not _is_component_hidden(obj) for obj in editable_objects)
        updated = 0
        skipped = len(component_objects) - len(editable_objects)
        for obj in editable_objects:
            try:
                obj.hide_viewport = False
            except Exception:
                pass
            try:
                obj.hide_set(should_hide, view_layer=context.view_layer)
                updated += 1
                continue
            except TypeError:
                pass
            except Exception:
                skipped += 1
                continue
            try:
                obj.hide_set(should_hide)
                updated += 1
            except Exception:
                skipped += 1

        if updated == 0:
            self.report({'WARNING'}, 'No TRLAU components could be toggled in the active view layer')
            return {'CANCELLED'}

        message = 'TRLAU components hidden' if should_hide else 'TRLAU components shown'
        if skipped:
            message += f' ({skipped} skipped in hidden/excluded collections)'
        self.report({'INFO'}, message)
        return {'FINISHED'}

def _draw_hinfo_data_props(layout, obj, component_type: str):
    ensure_hinfo_data_props(obj, component_type)
    fields = HINFO_DATA_FIELD_DEFS.get(component_type, ())
    if not fields:
        return
    data = get_hinfo_data(obj)
    data_box = layout.box()
    data_box.label(text='Panel Data')
    col = data_box.column(align=True)
    if data is None:
        for key, label, _default in fields:
            if key in obj:
                col.prop(obj, f'["{key}"]', text=label)
        return
    for key, label, _default in fields:
        if hasattr(data, key):
            col.prop(data, key, text=label)


class VIEW3D_PT_trlau_hinfo(bpy.types.Panel):
    bl_label = 'TRLAU HInfo'
    bl_idname = 'VIEW3D_PT_trlau_hinfo'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and trlau_object_type(obj) in HINFO_TYPES

    def draw_idprop(self, layout, obj, key: str, text: str):
        if key in obj:
            layout.prop(obj, f'["{key}"]', text=text)

    def draw_vector_prop(self, layout, data, prop_name: str, labels, title: str):
        box = layout.box()
        box.label(text=title)
        col = box.column(align=True)
        for index, axis in enumerate(labels):
            col.prop(data, prop_name, index=index, text=axis)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        component_type = trlau_object_type(obj)

        header = layout.box()
        keep_row = header.row()
        keep_row.prop(obj, 'trlau_keep_position')
        header.prop(obj, 'trlau_hinfo_bone')

        if component_type == 'HMarker':
            transform_box = layout.box()
            transform_box.label(text='Transform')
            self.draw_vector_prop(transform_box, obj, 'trlau_hinfo_position', ('X', 'Y', 'Z'), 'Position')
            self.draw_vector_prop(transform_box, obj, 'trlau_hinfo_rotation', ('X', 'Y', 'Z'), 'Rotation')
            transform_box.operator('trlau.snap_object_to_hmarker')
            _draw_hinfo_data_props(layout, obj, component_type)

        elif component_type == 'HSphere':
            transform_box = layout.box()
            transform_box.label(text='Transform')
            self.draw_vector_prop(transform_box, obj, 'trlau_hsphere_position_ui', ('X', 'Y', 'Z'), 'Position')
            transform_box.prop(obj, 'trlau_hsphere_radius_ui')
            _draw_hinfo_data_props(layout, obj, component_type)

        elif component_type == 'HBox':
            transform_box = layout.box()
            transform_box.label(text='Transform')
            self.draw_vector_prop(transform_box, obj, 'trlau_hbox_position_ui', ('X', 'Y', 'Z'), 'Position')
            self.draw_vector_prop(transform_box, obj, 'trlau_hbox_quaternion_ui', ('X', 'Y', 'Z', 'W'), 'Rotation')
            self.draw_vector_prop(transform_box, obj, 'trlau_hbox_dimensions_ui', ('X', 'Y', 'Z', 'W'), 'Scale')
            _draw_hinfo_data_props(layout, obj, component_type)

        elif component_type == 'HCapsule':
            transform_box = layout.box()
            transform_box.label(text='Transform')
            self.draw_vector_prop(transform_box, obj, 'trlau_hcapsule_position_ui', ('X', 'Y', 'Z'), 'Position')
            self.draw_vector_prop(transform_box, obj, 'trlau_hcapsule_quaternion_ui', ('X', 'Y', 'Z', 'W'), 'Rotation')
            self.draw_vector_prop(transform_box, obj, 'trlau_hcapsule_start_ui', ('X', 'Y', 'Z'), 'Start')
            self.draw_vector_prop(transform_box, obj, 'trlau_hcapsule_end_ui', ('X', 'Y', 'Z'), 'End')
            transform_box.prop(obj, 'trlau_hcapsule_radius_ui')
            transform_box.prop(obj, 'trlau_hcapsule_length_raw_ui')
            _draw_hinfo_data_props(layout, obj, component_type)

        elif component_type == 'Target':
            transform_box = layout.box()
            transform_box.label(text='Transform')
            self.draw_vector_prop(transform_box, obj, 'trlau_target_position_ui', ('X', 'Y', 'Z'), 'Position')
            self.draw_vector_prop(transform_box, obj, 'trlau_target_rotation_ui', ('X', 'Y', 'Z'), 'Rotation')

            data_box = layout.box()
            data_box.label(text='Target Properties')
            data_box.prop(obj, 'trlau_target_flags_ui', text='Flags')
            self.draw_idprop(data_box, obj, 'trlau_target_unique_id', 'Unique ID')





def _draw_hinfo_idprop(layout, obj, key: str, text: str):
    if key in obj:
        layout.prop(obj, f'["{key}"]', text=text)


def _draw_hinfo_vector_prop(layout, data, prop_name: str, labels, title: str):
    box = layout.box()
    box.label(text=title)
    col = box.column(align=True)
    for index, axis in enumerate(labels):
        col.prop(data, prop_name, index=index, text=axis)


def _draw_hinfo_creation_content(layout, context):
    grid = layout.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
    for component_type, label, _description in HINFO_COMPONENT_TYPE_ITEMS:
        op = grid.operator('trlau.add_hinfo_component', text=label)
        op.component_type = component_type


def _draw_hinfo_details_content(layout, context):
    obj = context.object
    if obj is None:
        return
    component_type = trlau_object_type(obj)

    header = layout.box()
    keep_row = header.row()
    keep_row.prop(obj, 'trlau_keep_position')
    header.prop(obj, 'trlau_hinfo_bone')

    if component_type == 'HMarker':
        transform_box = layout.box()
        transform_box.label(text='Transform')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hinfo_position', ('X', 'Y', 'Z'), 'Position')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hinfo_rotation', ('X', 'Y', 'Z'), 'Rotation')
        transform_box.operator('trlau.snap_object_to_hmarker')
        _draw_hinfo_data_props(layout, obj, component_type)

    elif component_type == 'HSphere':
        transform_box = layout.box()
        transform_box.label(text='Transform')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hsphere_position_ui', ('X', 'Y', 'Z'), 'Position')
        transform_box.prop(obj, 'trlau_hsphere_radius_ui')
        _draw_hinfo_data_props(layout, obj, component_type)

    elif component_type == 'HBox':
        transform_box = layout.box()
        transform_box.label(text='Transform')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hbox_position_ui', ('X', 'Y', 'Z'), 'Position')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hbox_quaternion_ui', ('X', 'Y', 'Z', 'W'), 'Rotation')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hbox_dimensions_ui', ('X', 'Y', 'Z', 'W'), 'Scale')
        _draw_hinfo_data_props(layout, obj, component_type)

    elif component_type == 'HCapsule':
        transform_box = layout.box()
        transform_box.label(text='Transform')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hcapsule_position_ui', ('X', 'Y', 'Z'), 'Position')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hcapsule_quaternion_ui', ('X', 'Y', 'Z', 'W'), 'Rotation')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hcapsule_start_ui', ('X', 'Y', 'Z'), 'Start')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_hcapsule_end_ui', ('X', 'Y', 'Z'), 'End')
        transform_box.prop(obj, 'trlau_hcapsule_radius_ui')
        transform_box.prop(obj, 'trlau_hcapsule_length_raw_ui')
        _draw_hinfo_data_props(layout, obj, component_type)

    elif component_type == 'Target':
        transform_box = layout.box()
        transform_box.label(text='Transform')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_target_position_ui', ('X', 'Y', 'Z'), 'Position')
        _draw_hinfo_vector_prop(transform_box, obj, 'trlau_target_rotation_ui', ('X', 'Y', 'Z'), 'Rotation')

        data_box = layout.box()
        data_box.label(text='Target Properties')
        data_box.prop(obj, 'trlau_target_flags_ui', text='Flags')
        _draw_hinfo_idprop(data_box, obj, 'trlau_target_unique_id', 'Unique ID')


def _draw_target_details_content(layout, context):
    obj = context.object
    if obj is None or trlau_object_type(obj) != 'Target':
        return

    header = layout.box()
    keep_row = header.row()
    keep_row.prop(obj, 'trlau_keep_position')
    header.prop(obj, 'trlau_hinfo_bone')

    transform_box = layout.box()
    transform_box.label(text='Transform')
    _draw_hinfo_vector_prop(transform_box, obj, 'trlau_target_position_ui', ('X', 'Y', 'Z'), 'Position')
    _draw_hinfo_vector_prop(transform_box, obj, 'trlau_target_rotation_ui', ('X', 'Y', 'Z'), 'Rotation')

    data_box = layout.box()
    data_box.label(text='Target Properties')
    data_box.prop(obj, 'trlau_target_flags_ui', text='Flags')
    _draw_hinfo_idprop(data_box, obj, 'trlau_target_unique_id', 'Unique ID')


def _has_hinfo_creation_context(context) -> bool:
    arm_obj = getattr(context, 'object', None)
    return arm_obj is not None and getattr(arm_obj, 'type', None) == 'ARMATURE' and str(getattr(context, 'mode', '') or '') in {'POSE', 'EDIT_ARMATURE'}



def _draw_hinfo_dropdown_content(layout, context):
    obj = getattr(context, 'object', None)
    has_details = obj is not None and trlau_object_type(obj) in HINFO_TYPES
    has_creation = _has_hinfo_creation_context(context)
    if not (has_details or has_creation):
        return

    scene = getattr(context, 'scene', None)
    expanded = True if scene is None else bool(getattr(scene, 'trlau_editor_hinfo_expanded', True))

    box = layout.box()
    header = box.row(align=True)
    icon = 'TRIA_DOWN' if expanded else 'TRIA_RIGHT'
    if scene is not None and hasattr(scene, 'trlau_editor_hinfo_expanded'):
        header.prop(scene, 'trlau_editor_hinfo_expanded', text='HInfo', icon=icon, emboss=False)
    else:
        header.label(text='HInfo', icon=icon)

    if not expanded:
        return

    body = box.column(align=True)
    if has_creation:
        _draw_hinfo_creation_content(body, context)
    if has_details:
        if has_creation:
            body.separator()
        _draw_hinfo_details_content(body, context)

class VIEW3D_PT_trlau_editor_hinfo(bpy.types.Panel):
    bl_label = 'HInfo'
    bl_idname = 'VIEW3D_PT_trlau_editor_hinfo'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_order = 10
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        has_details = obj is not None and trlau_object_type(obj) in HINFO_TYPES
        return has_details or _has_hinfo_creation_context(context)

    def draw(self, context):
        layout = self.layout
        obj = getattr(context, 'object', None)
        has_details = obj is not None and trlau_object_type(obj) in HINFO_TYPES
        has_creation = _has_hinfo_creation_context(context)

        col = layout.column(align=True)
        if has_creation:
            _draw_hinfo_creation_content(col, context)
        if has_details:
            if has_creation:
                col.separator()
            _draw_hinfo_details_content(col, context)

class VIEW3D_PT_trlau_editor_target(bpy.types.Panel):
    bl_label = 'Target'
    bl_idname = 'VIEW3D_PT_trlau_editor_target'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        return obj is not None and trlau_object_type(obj) == 'Target'

    def draw(self, context):
        _draw_target_details_content(self.layout, context)


classes = (
    TRLAU_OT_snap_object_to_hmarker,
    TRLAU_OT_import_model_hinfo,
    TRLAU_OT_import_model_bone_mirrors,
    TRLAU_OT_add_hinfo_component,
    TRLAU_OT_toggle_component_visibility,
    VIEW3D_PT_trlau_editor_hinfo,
    VIEW3D_PT_trlau_editor_target,
)
