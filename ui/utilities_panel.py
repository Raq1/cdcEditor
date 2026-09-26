from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import random
import re
import shutil
import struct

import bpy
import mathutils
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatVectorProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from ..builders.mesh_builder import MeshBuildSettings, MeshBuilder, _decode_tpage_flags
from ..core.blender_mesh_utils import get_or_create_color_attribute, set_active_color_attribute
from ..core.drawgroup_visibility import ensure_material_visibility, get_drawgroup_materials, get_tagged_scene_materials
from ..core.material_ui import (
    _assign_current_material_nodes,
    _get_mesh_vertex_color_name,
    apply_pc_nextgen_material_settings,
    apply_ps3_material_settings,
    apply_trlau_material_settings,
    convert_material_to_type,
    find_material_image,
    get_material_reflection_mode,
    get_material_platform_name,
    decode_tpage_flags,
    encode_tpage_flags,
    get_material_tpageid,
    get_material_draw_group,
    reload_textures_for_scene,
    rename_material_from_tpageid,
    set_material_panel_value,
    set_material_tpageid,
    sync_pc_nextgen_material_from_panel,
)
from ..core.naming_schemes import get_scheme_items, get_scheme_map
from ..core.wave_audio import (
    WaveAudioError,
    encode_pcm_i16_to_encoded_wave_audio,
    iter_wave_section_candidates,
    read_audio_as_mono_i16,
    replace_wave_section_file_with_encoded_audio,
)
from ..operators.mul_import import (
    MUL_BONE_FLAGS_OFFSET,
    MUL_BONE_LOCATION_OFFSET,
    MUL_BONE_ROTATION_OFFSET,
    MUL_BONE_SCALE_OFFSET,
    MUL_FRAME_HEADER_SIZE,
    MUL_ALIGNMENT,
    MUL_PACKET_TYPE_CINEMATIC,
    MUL_SKELETON_ROOT_CHANNEL_COUNT,
    MUL_BONE_CHANNEL_STRIDE,
    MUL_STREAM_START_OFFSET,
    MultiplexStreamImporterMixin,
)


class TRLAU_OT_batch_replace_wave_audio(bpy.types.Operator):
    bl_idname = 'trlau.batch_replace_wave_audio'
    bl_label = 'Batch Replace Wave Audio'
    bl_description = 'Replace the ADPCM sound payload of game Wave section files with one source audio file'
    bl_options = {'REGISTER'}

    source_audio_path: StringProperty(
        name='Source Audio',
        description='Audio file to encode into the selected game Wave section files. WAV is built in; MP3/OGG/FLAC/M4A and similar formats use FFmpeg if available',
        subtype='FILE_PATH',
        default='',
    )
    target_wave_directory: StringProperty(
        name='Wave Folder',
        description='Folder containing the original game Wave section files to replace',
        subtype='DIR_PATH',
        default='',
    )
    output_directory: StringProperty(
        name='Output Folder',
        description='Where replaced Wave files will be written when Overwrite Originals is disabled. Leave empty to create a wave_replaced folder inside the Wave folder',
        subtype='DIR_PATH',
        default='',
    )
    overwrite_originals: BoolProperty(
        name='Overwrite Originals',
        description='Write the replaced Wave data back over the original files. Disable this to write copies instead',
        default=False,
    )
    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, 'source_audio_path')
        layout.prop(self, 'target_wave_directory')
        layout.separator()
        layout.prop(self, 'overwrite_originals')
        if not self.overwrite_originals:
            layout.prop(self, 'output_directory')

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except Exception:
            return False

    def execute(self, context):
        source_audio = Path(bpy.path.abspath(self.source_audio_path or '')).expanduser()
        target_dir = Path(bpy.path.abspath(self.target_wave_directory or '')).expanduser()
        if not source_audio.is_file():
            self.report({'ERROR'}, 'Select a valid source audio file')
            return {'CANCELLED'}
        if not target_dir.is_dir():
            self.report({'ERROR'}, 'Select a valid folder containing game .wave files')
            return {'CANCELLED'}

        try:
            samples, sample_rate = read_audio_as_mono_i16(source_audio)
            encoded_audio = encode_pcm_i16_to_encoded_wave_audio(samples, sample_rate)
        except WaveAudioError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to load source audio: {exc}')
            return {'CANCELLED'}

        try:
            candidates = iter_wave_section_candidates(target_dir)
        except WaveAudioError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to scan Wave folder: {exc}')
            return {'CANCELLED'}

        overwrite = bool(self.overwrite_originals)
        output_root = target_dir
        if not overwrite:
            output_root_text = bpy.path.abspath(self.output_directory or '') if self.output_directory else ''
            output_root = Path(output_root_text).expanduser() if output_root_text else (target_dir / 'wave_replaced')
            if output_root.exists() and self._is_relative_to(output_root, target_dir):
                candidates = [path for path in candidates if not self._is_relative_to(path, output_root)]

        if not candidates:
            self.report({'ERROR'}, 'No Wave section files found in the selected folder')
            return {'CANCELLED'}

        replaced = 0
        failed: list[str] = []
        total_original_bytes = 0
        total_new_bytes = 0
        def _replace_one(input_path: Path):
            try:
                rel_path = input_path.relative_to(target_dir)
            except Exception:
                rel_path = Path(input_path.name)
            output_path = input_path if overwrite else (output_root / rel_path)
            tmp_path = input_path.with_name(f'{input_path.name}.trlau_tmp')
            try:
                if overwrite:
                    result = replace_wave_section_file_with_encoded_audio(
                        input_path,
                        tmp_path,
                        encoded_audio,
                    )
                    tmp_path.replace(input_path)
                    return input_path, result, None
                result = replace_wave_section_file_with_encoded_audio(
                    input_path,
                    output_path,
                    encoded_audio,
                )
                return input_path, result, None
            except Exception as exc:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                return input_path, None, exc

        worker_count = max(1, min(8, int(os.cpu_count() or 1), len(candidates)))
        context.window_manager.progress_begin(0, len(candidates))
        try:
            if worker_count <= 1 or len(candidates) <= 1:
                for index, input_path in enumerate(candidates, 1):
                    path, result, error = _replace_one(input_path)
                    context.window_manager.progress_update(index)
                    if error is not None:
                        failed.append(f'{path.name}: {error}')
                        continue
                    replaced += 1
                    total_original_bytes += int(result.original_size)
                    total_new_bytes += int(result.new_size)
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    futures = [executor.submit(_replace_one, input_path) for input_path in candidates]
                    for index, future in enumerate(as_completed(futures), 1):
                        path, result, error = future.result()
                        context.window_manager.progress_update(index)
                        if error is not None:
                            failed.append(f'{path.name}: {error}')
                            continue
                        replaced += 1
                        total_original_bytes += int(result.original_size)
                        total_new_bytes += int(result.new_size)
        finally:
            context.window_manager.progress_end()

        if replaced <= 0:
            message = 'No Wave files were replaced'
            if failed:
                message += f'; first error: {failed[0]}'
            self.report({'ERROR'}, message)
            return {'CANCELLED'}

        message = f'Replaced audio in {replaced} Wave file(s) from {source_audio.name} ({sample_rate} Hz, {len(samples)} samples; ADPCM encoded once)'
        if not overwrite:
            message += f'; output: {output_root}'
        if failed:
            message += f'; failed {len(failed)}'
        if total_original_bytes != total_new_bytes:
            message += f'; size {total_original_bytes} -> {total_new_bytes} bytes'
        self.report({'INFO'}, message)
        return {'FINISHED'}

class TRLAU_OT_toggle_drawgroup_visibility(bpy.types.Operator):
    bl_idname = 'trlau.toggle_drawgroup_visibility'
    bl_label = 'Toggle Drawgroup Visibility'
    bl_description = 'Hide or unhide materials based on drawgroup'
    bl_options = {'REGISTER', 'UNDO'}

    exclude_all_except: BoolProperty(
        name='Exclude All Except',
        default=False,
        description='Show only the selected drawgroup and hide every other tagged drawgroup in the scene',
    )
    drawgroup: IntProperty(name='Drawgroup', min=0, default=0)

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=240)

    def draw(self, _context):
        self.layout.prop(self, 'exclude_all_except')
        self.layout.prop(self, 'drawgroup')

    def execute(self, context):
        scene = context.scene
        drawgroup_materials = get_drawgroup_materials(scene, self.drawgroup)
        if not drawgroup_materials:
            self.report({'WARNING'}, f'No scene materials found for drawgroup {self.drawgroup}')
            return {'CANCELLED'}

        if self.exclude_all_except:
            tagged_materials = get_tagged_scene_materials(scene)
            other_materials = [material for material in tagged_materials if int(get_material_draw_group(material)) != int(self.drawgroup)]
            is_currently_isolated = bool(other_materials) and all(bool(material.get('trlau_drawgroup_hidden', False)) for material in other_materials) and all(not bool(material.get('trlau_drawgroup_hidden', False)) for material in drawgroup_materials)
            show_all = is_currently_isolated

            updated = 0
            for material in drawgroup_materials:
                if ensure_material_visibility(material, False):
                    updated += 1
            for material in other_materials:
                if ensure_material_visibility(material, False if show_all else True):
                    updated += 1

            if updated == 0:
                self.report({'WARNING'}, 'No drawgroup materials could be updated')
                return {'CANCELLED'}

            self.report({'INFO'}, 'All drawgroups shown' if show_all else f'Only drawgroup {self.drawgroup} is visible')
            return {'FINISHED'}

        should_hide = any(not bool(material.get('trlau_drawgroup_hidden', False)) for material in drawgroup_materials)
        updated = 0
        for material in drawgroup_materials:
            if ensure_material_visibility(material, should_hide):
                updated += 1

        if updated == 0:
            self.report({'WARNING'}, 'No drawgroup materials could be updated')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Drawgroup {self.drawgroup} hidden' if should_hide else f'Drawgroup {self.drawgroup} shown')
        return {'FINISHED'}



class TRLAU_OT_add_vertex_color(bpy.types.Operator):
    bl_idname = 'trlau.add_vertex_color'
    bl_label = 'Add Vertex Color'
    bl_description = 'Add a Color attribute to selected meshes'
    bl_options = {'REGISTER', 'UNDO'}

    DEFAULT_COLOR = (0.440, 0.440, 0.440, 1.0)

    vertex_color: FloatVectorProperty(
        name='Vertex Color',
        description='Color to write into the Color attribute on all selected mesh vertices',
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=DEFAULT_COLOR,
    )

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, _context):
        self.layout.prop(self, 'vertex_color')

    @staticmethod
    def _ensure_color_attribute(mesh, color):
        color_attributes = getattr(mesh, 'color_attributes', None)
        if color_attributes is None:
            return False

        color_attr = get_or_create_color_attribute(mesh, 'Color', domain='POINT')
        if color_attr is None:
            return False

        rgba = tuple(float(channel) for channel in color)
        if len(rgba) < 4:
            rgba = (rgba[0], rgba[1], rgba[2], 1.0)

        vertex_count = len(mesh.vertices)
        if vertex_count > 0:
            flat_colors = list(rgba) * vertex_count
            try:
                color_attr.data.foreach_set('color', flat_colors)
            except Exception:
                for item in color_attr.data:
                    item.color = rgba

        set_active_color_attribute(mesh, color_attr, 'Color')
        return True

    @staticmethod
    def _find_material_image(material):
        return find_material_image(material)

    @staticmethod
    def _rebuild_material_with_vertex_color(material):
        builder = MeshBuilder(
            context=bpy.context,
            filepath=str(Path(bpy.app.tempdir or '/tmp') / 'trlau_vertexcolor_add.tr7aemesh'),
            settings=MeshBuildSettings(apply_segment_pivots=True, import_textures=True),
            collection=bpy.context.scene.collection,
        )
        tpage_flags = _decode_tpage_flags(int(get_material_tpageid(material)))
        image = TRLAU_OT_add_vertex_color._find_material_image(material)
        reflection_mode = get_material_reflection_mode(material)
        reflective = reflection_mode != 'none'
        builder._setup_material_nodes(
            material,
            image,
            tpage_flags,
            use_vertex_colors=True,
            reflective=reflective,
            reflection_mode=reflection_mode,
        )

    def execute(self, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'WARNING'}, 'Select at least one mesh object')
            return {'CANCELLED'}

        updated_mesh_count = 0
        affected_materials = []
        for obj in selected_meshes:
            if self._ensure_color_attribute(obj.data, self.vertex_color):
                updated_mesh_count += 1
            for material in getattr(obj.data, 'materials', []):
                if material is not None:
                    affected_materials.append(material)

        unique_materials = []
        seen = set()
        for material in affected_materials:
            if material.name in seen:
                continue
            seen.add(material.name)
            unique_materials.append(material)

        for material in unique_materials:
            self._rebuild_material_with_vertex_color(material)

        if updated_mesh_count == 0 and not unique_materials:
            self.report({'WARNING'}, 'No selected meshes or connected materials could be updated')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Added vertex color usage to {len(unique_materials)} material(s) across {updated_mesh_count} mesh object(s)')
        return {'FINISHED'}


class TRLAU_OT_remove_vertex_color(bpy.types.Operator):
    bl_idname = 'trlau.remove_vertex_color'
    bl_label = 'Remove Vertex Color'
    bl_description = 'Remove the Color attribute from selected meshes'
    bl_options = {'REGISTER', 'UNDO'}

    @staticmethod
    def _remove_color_attribute(mesh):
        removed = False
        color_attributes = getattr(mesh, 'color_attributes', None)
        if color_attributes is not None:
            color_attr = color_attributes.get('Color')
            if color_attr is not None:
                color_attributes.remove(color_attr)
                removed = True

        legacy_vertex_colors = getattr(mesh, 'vertex_colors', None)
        if legacy_vertex_colors is not None:
            color_layer = legacy_vertex_colors.get('Color')
            if color_layer is not None:
                legacy_vertex_colors.remove(color_layer)
                removed = True

        return removed

    @staticmethod
    def _find_material_image(material):
        return find_material_image(material)

    @staticmethod
    def _rebuild_material_without_vertex_color(material):
        builder = MeshBuilder(
            context=bpy.context,
            filepath=str(Path(bpy.app.tempdir or '/tmp') / 'trlau_vertexcolor_remove.tr7aemesh'),
            settings=MeshBuildSettings(apply_segment_pivots=True, import_textures=True),
            collection=bpy.context.scene.collection,
        )
        tpage_flags = _decode_tpage_flags(int(get_material_tpageid(material)))
        image = TRLAU_OT_remove_vertex_color._find_material_image(material)
        reflection_mode = get_material_reflection_mode(material)
        reflective = reflection_mode != 'none'
        builder._setup_material_nodes(
            material,
            image,
            tpage_flags,
            use_vertex_colors=False,
            reflective=reflective,
            reflection_mode=reflection_mode,
        )

    def execute(self, context):
        selected_meshes = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if not selected_meshes:
            self.report({'WARNING'}, 'Select at least one mesh object')
            return {'CANCELLED'}

        removed_mesh_count = 0
        affected_materials = []
        for obj in selected_meshes:
            if self._remove_color_attribute(obj.data):
                removed_mesh_count += 1
            for material in getattr(obj.data, 'materials', []):
                if material is not None:
                    affected_materials.append(material)

        unique_materials = []
        seen = set()
        for material in affected_materials:
            if material.name in seen:
                continue
            seen.add(material.name)
            unique_materials.append(material)

        for material in unique_materials:
            self._rebuild_material_without_vertex_color(material)

        if removed_mesh_count == 0 and not unique_materials:
            self.report({'WARNING'}, 'No Color attribute or connected materials were found on the selected meshes')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Removed vertex color usage from {len(unique_materials)} material(s) across {len(selected_meshes)} mesh object(s)')
        return {'FINISHED'}


class TRLAU_OT_rename_bones(bpy.types.Operator):
    bl_idname = 'trlau.rename_bones'
    bl_label = 'Rename Bones'
    bl_description = 'Rename the bones of the selected armature using a predefined naming scheme'
    bl_options = {'REGISTER', 'UNDO'}

    naming_scheme: EnumProperty(
        name='Scheme',
        items=get_scheme_items(),
        default='lara',
    )

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=240)

    def draw(self, _context):
        self.layout.prop(self, 'naming_scheme')

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'ARMATURE'

    @staticmethod
    def _apply_rename_pairs(bones, rename_pairs):
        temp_names = {}
        for index, (source, _target) in enumerate(rename_pairs):
            bone = bones.get(source)
            if bone is None:
                continue
            temp_name = f'__trlau_tmp_{index}__'
            bone.name = temp_name
            temp_names[source] = temp_name

        renamed_count = 0
        for source, target in rename_pairs:
            temp_name = temp_names.get(source)
            bone = bones.get(temp_name) if temp_name is not None else None
            if bone is None:
                continue
            bone.name = target
            renamed_count += 1
        return renamed_count

    def execute(self, context):
        armature_obj = context.object
        if armature_obj is None or armature_obj.type != 'ARMATURE':
            self.report({'WARNING'}, 'Select an armature object')
            return {'CANCELLED'}

        forward_map = get_scheme_map(self.naming_scheme)
        if not forward_map:
            self.report({'WARNING'}, f'Unsupported naming scheme: {self.naming_scheme}')
            return {'CANCELLED'}

        reverse_map = {target: source for source, target in forward_map.items()}
        bones = armature_obj.data.bones

        forward_pairs = [(source, target) for source, target in forward_map.items() if bones.get(source) is not None]
        reverse_pairs = [(source, target) for source, target in reverse_map.items() if bones.get(source) is not None]

        if not forward_pairs and not reverse_pairs:
            self.report({'WARNING'}, 'No matching bones were found for the selected naming scheme')
            return {'CANCELLED'}

        use_reverse = len(reverse_pairs) > len(forward_pairs)
        rename_pairs = reverse_pairs if use_reverse else forward_pairs
        renamed_count = self._apply_rename_pairs(bones, rename_pairs)
        direction = 'to internal names' if use_reverse else f'using the {self.naming_scheme} scheme'
        self.report({'INFO'}, f'Renamed {renamed_count} bone(s) {direction}')
        return {'FINISHED'}


_BONE_NAME_RE = re.compile(r'^bone_(\d+)$')


class TRLAU_OT_make_bones_contiguous(bpy.types.Operator):
    bl_idname = 'trlau.make_bones_contiguous'
    bl_label = 'Make Bones Contiguous'
    bl_description = 'Rename non bone_X bones and add missing bone_# entries so bone indices are contiguous for model export'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and getattr(obj, 'type', None) == 'ARMATURE'

    @staticmethod
    def _bone_index(name: str):
        match = _BONE_NAME_RE.match(str(name or ''))
        if match is None:
            return None
        return int(match.group(1))

    @classmethod
    def _bone_indices(cls, armature_obj):
        indices = set()
        bones = getattr(getattr(armature_obj, 'data', None), 'bones', None)
        if bones is None:
            return indices
        for bone in bones:
            index = cls._bone_index(getattr(bone, 'name', ''))
            if index is not None:
                indices.add(index)
        return indices

    @classmethod
    def _non_scheme_bone_names(cls, armature_obj):
        names = []
        bones = getattr(getattr(armature_obj, 'data', None), 'bones', None)
        if bones is None:
            return names
        for bone in bones:
            name = str(getattr(bone, 'name', '') or '')
            if cls._bone_index(name) is None:
                names.append(name)
        return names

    @staticmethod
    def _set_active_armature(context, armature_obj):
        view_layer = getattr(context, 'view_layer', None)
        if view_layer is not None:
            try:
                view_layer.objects.active = armature_obj
            except Exception:
                pass
        try:
            armature_obj.select_set(True)
        except Exception:
            pass

    @staticmethod
    def _unique_temp_name(bones, base: str) -> str:
        candidate = base
        suffix = 0
        while bones.get(candidate) is not None:
            suffix += 1
            candidate = f'{base}_{suffix}'
        return candidate

    @classmethod
    def _build_non_scheme_rename_pairs(cls, armature_obj):
        bones = getattr(getattr(armature_obj, 'data', None), 'bones', None)
        if bones is None:
            return []

        non_scheme_names = cls._non_scheme_bone_names(armature_obj)
        if not non_scheme_names:
            return []

        indices = cls._bone_indices(armature_obj)
        if indices:
            max_index = max(indices)
            missing_slots = [index for index in range(max_index + 1) if index not in indices]
            next_index = max_index + 1
        else:
            missing_slots = []
            next_index = 0

        rename_pairs = []
        used_indices = set(indices)
        for old_name in non_scheme_names:
            while missing_slots:
                target_index = missing_slots.pop(0)
                if target_index not in used_indices:
                    break
            else:
                target_index = next_index
                next_index += 1
                while target_index in used_indices:
                    target_index = next_index
                    next_index += 1

            used_indices.add(target_index)
            rename_pairs.append((old_name, f'bone_{target_index}'))

        return rename_pairs

    @classmethod
    def _rename_non_scheme_bones(cls, armature_obj):
        bones = getattr(getattr(armature_obj, 'data', None), 'bones', None)
        if bones is None:
            return {}

        rename_pairs = cls._build_non_scheme_rename_pairs(armature_obj)
        if not rename_pairs:
            return {}

        temp_pairs = []
        for index, (old_name, target_name) in enumerate(rename_pairs):
            bone = bones.get(old_name)
            if bone is None:
                continue
            temp_name = cls._unique_temp_name(bones, f'__trlau_contiguous_tmp_{index}__')
            bone.name = temp_name
            temp_pairs.append((old_name, temp_name, target_name))

        rename_map = {}
        for old_name, temp_name, target_name in temp_pairs:
            bone = bones.get(temp_name)
            if bone is None:
                continue
            bone.name = target_name
            rename_map[old_name] = target_name

        return rename_map

    @staticmethod
    def _iter_meshes_using_armature(armature_obj):
        for obj in getattr(bpy.data, 'objects', []) or []:
            if obj is None or getattr(obj, 'type', None) != 'MESH':
                continue
            if getattr(obj, 'parent', None) == armature_obj:
                yield obj
                continue
            modifiers = getattr(obj, 'modifiers', []) or []
            for modifier in modifiers:
                if getattr(modifier, 'type', None) == 'ARMATURE' and getattr(modifier, 'object', None) == armature_obj:
                    yield obj
                    break

    @staticmethod
    def _merge_vertex_group(mesh_obj, old_name: str, target_name: str) -> bool:
        groups = getattr(mesh_obj, 'vertex_groups', None)
        mesh = getattr(mesh_obj, 'data', None)
        if groups is None or mesh is None:
            return False

        old_group = groups.get(old_name)
        if old_group is None:
            return False

        target_group = groups.get(target_name)
        if target_group is None:
            old_group.name = target_name
            return True

        old_index = old_group.index
        target_index = target_group.index
        for vertex in getattr(mesh, 'vertices', []) or []:
            old_weight = None
            target_weight = None
            for group_ref in getattr(vertex, 'groups', []) or []:
                group_index = getattr(group_ref, 'group', None)
                if group_index == old_index:
                    old_weight = getattr(group_ref, 'weight', 0.0)
                elif group_index == target_index:
                    target_weight = getattr(group_ref, 'weight', 0.0)
            if old_weight is not None:
                weight = max(float(old_weight), float(target_weight or 0.0))
                target_group.add([vertex.index], weight, 'REPLACE')

        try:
            groups.remove(old_group)
        except Exception:
            pass
        return True

    @classmethod
    def _retarget_bone_references(cls, armature_obj, rename_map):
        if not rename_map:
            return 0

        updated = 0
        for mesh_obj in cls._iter_meshes_using_armature(armature_obj):
            for old_name, target_name in rename_map.items():
                if cls._merge_vertex_group(mesh_obj, old_name, target_name):
                    updated += 1

        for obj in getattr(bpy.data, 'objects', []) or []:
            constraints = getattr(obj, 'constraints', []) or []
            for constraint in constraints:
                if getattr(constraint, 'target', None) == armature_obj:
                    subtarget = getattr(constraint, 'subtarget', '')
                    if subtarget in rename_map:
                        constraint.subtarget = rename_map[subtarget]
                        updated += 1

        pose = getattr(armature_obj, 'pose', None)
        pose_bones = getattr(pose, 'bones', []) if pose is not None else []
        for pose_bone in pose_bones:
            constraints = getattr(pose_bone, 'constraints', []) or []
            for constraint in constraints:
                if getattr(constraint, 'target', None) == armature_obj:
                    subtarget = getattr(constraint, 'subtarget', '')
                    if subtarget in rename_map:
                        constraint.subtarget = rename_map[subtarget]
                        updated += 1

        return updated

    @staticmethod
    def _format_created_message(created: int, missing):
        if created <= 0:
            return ''
        message = f'Added {created} missing bone(s): ' + ', '.join(f'bone_{index}' for index in missing[:8])
        if len(missing) > 8:
            message += f', and {len(missing) - 8} more'
        return message

    @staticmethod
    def _restore_mode(context, armature_obj, previous_active, previous_mode):
        try:
            context.view_layer.objects.active = previous_active
        except Exception:
            pass
        try:
            if previous_active is not None:
                previous_active.select_set(True)
        except Exception:
            pass
        if previous_active == armature_obj and previous_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode=previous_mode)
            except Exception:
                pass

    def execute(self, context):
        armature_obj = context.object
        if armature_obj is None or getattr(armature_obj, 'type', None) != 'ARMATURE':
            self.report({'WARNING'}, 'Select an armature object')
            return {'CANCELLED'}

        previous_active = getattr(context.view_layer.objects, 'active', None)
        previous_mode = getattr(armature_obj, 'mode', 'OBJECT')

        try:
            if previous_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass

        created = 0
        missing = []
        renamed_map = {}
        reference_updates = 0

        try:
            renamed_map = self._rename_non_scheme_bones(armature_obj)
            reference_updates = self._retarget_bone_references(armature_obj, renamed_map)

            indices = self._bone_indices(armature_obj)
            if not indices:
                self.report({'WARNING'}, f'Armature {armature_obj.name} does not contain any bones')
                return {'CANCELLED'}

            max_index = max(indices)
            missing = [index for index in range(max_index + 1) if index not in indices]

            if missing:
                self._set_active_armature(context, armature_obj)
                if getattr(armature_obj, 'mode', 'OBJECT') != 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')

                edit_bones = getattr(getattr(armature_obj, 'data', None), 'edit_bones', None)
                if edit_bones is None:
                    raise RuntimeError('Armature edit bones are unavailable')

                for index in missing:
                    name = f'bone_{index}'
                    if edit_bones.get(name) is not None:
                        continue
                    bone = edit_bones.new(name)
                    bone.head = (0.0, 0.0, 0.0)
                    bone.tail = (0.0, 0.0, 0.01)
                    bone.roll = 0.0
                    bone.parent = None
                    try:
                        bone.use_deform = False
                    except Exception:
                        pass
                    created += 1

                bpy.ops.object.mode_set(mode='OBJECT')
        except Exception as exc:
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
            self.report({'ERROR'}, f'Could not make bones contiguous: {exc}')
            return {'CANCELLED'}
        finally:
            self._restore_mode(context, armature_obj, previous_active, previous_mode)

        messages = []
        if renamed_map:
            messages.append(f'Renamed {len(renamed_map)} non-scheme bone(s)')
        created_message = self._format_created_message(created, missing)
        if created_message:
            messages.append(created_message)
        if reference_updates:
            messages.append(f'updated {reference_updates} related reference(s)')

        if messages:
            self.report({'INFO'}, '; '.join(messages))
        else:
            self.report({'INFO'}, f'Armature {armature_obj.name} already has contiguous bones')
        return {'FINISHED'}


class TRLAU_OT_replace_mul_skeleton(MultiplexStreamImporterMixin, bpy.types.Operator, ImportHelper):
    bl_idname = 'trlau.replace_mul_skeleton'
    bl_label = 'Replace MUL Skeleton'
    bl_description = 'Replace the MUL skeleton and retarget its animation channels for the chosen InstanceID with the selected armature proportions'
    bl_options = {'REGISTER', 'UNDO'}

    _ANSI_BLUE = '\033[94m'
    _ANSI_GREEN = '\033[92m'
    _ANSI_RED = '\033[91m'
    _ANSI_RESET = '\033[0m'

    filename_ext = '.mul'
    filter_glob: StringProperty(default='*.mul', options={'HIDDEN'})
    files: CollectionProperty(
        name='MUL Files',
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    instance_id: IntProperty(
        name='InstanceID',
        description='MUL skeleton InstanceID whose default matrices should be replaced',
        default=-1,
        min=-2147483648,
        max=2147483647,
    )
    create_backup: BoolProperty(
        name='Create .bak backup',
        description='Write a .bak copy next to each MUL before replacing it in place',
        default=False,
    )

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None) or getattr(context, 'active_object', None)
        return obj is not None and getattr(obj, 'type', None) == 'ARMATURE'

    @staticmethod
    def _active_armature(context):
        obj = getattr(context, 'object', None) or getattr(context, 'active_object', None)
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return obj
        return None

    @staticmethod
    def _iter_related_objects(root_obj):
        stack = [root_obj]
        seen = set()
        while stack:
            obj = stack.pop()
            if obj is None:
                continue
            key = getattr(obj, 'name', None) or id(obj)
            if key in seen:
                continue
            seen.add(key)
            yield obj
            parent = getattr(obj, 'parent', None)
            if parent is not None:
                stack.append(parent)
            for child in getattr(obj, 'children', []) or []:
                stack.append(child)

    @classmethod
    def _candidate_instance_id(cls, armature_obj):
        for obj in cls._iter_related_objects(armature_obj):
            if obj is None or 'trlau_bginstance_id' not in obj:
                continue
            try:
                return int(obj['trlau_bginstance_id'])
            except Exception:
                continue
        return None

    @staticmethod
    def _flatten_matrix(matrix) -> list[float]:
        try:
            return [float(matrix[row][column]) for row in range(4) for column in range(4)]
        except Exception:
            return [1.0 if row == column else 0.0 for row in range(4) for column in range(4)]

    @staticmethod
    def _matrix_from_values(values) -> mathutils.Matrix:
        return MultiplexStreamImporterMixin._mul_default_transform_to_matrix(values)

    @staticmethod
    def _wrap_angle_near_reference(value: float, reference: float) -> float:
        try:
            value = float(value)
            reference = float(reference)
            if not math.isfinite(value) or not math.isfinite(reference):
                return value
            return value + (round((reference - value) / math.tau) * math.tau)
        except Exception:
            return float(value)

    @staticmethod
    def _float_close(left: float, right: float, epsilon: float = 1.0e-5) -> bool:
        try:
            return abs(float(left) - float(right)) <= float(epsilon)
        except Exception:
            return False

    @staticmethod
    def _bone_depth(pose_bone: bpy.types.PoseBone) -> int:
        depth = 0
        parent = getattr(pose_bone, 'parent', None)
        while parent is not None:
            depth += 1
            parent = getattr(parent, 'parent', None)
        return depth

    @staticmethod
    def _pose_bone_mapping_by_index(armature_obj: bpy.types.Object, num_bones: int) -> dict[int, bpy.types.PoseBone]:
        pose_bones = list(getattr(getattr(armature_obj, 'pose', None), 'bones', []) or [])
        pose_bones_by_name = {pose_bone.name: pose_bone for pose_bone in pose_bones}
        mapped: dict[int, bpy.types.PoseBone] = {}
        used_names: set[str] = set()
        for bone_index in range(min(int(num_bones), len(pose_bones))):
            pose_bone = pose_bones_by_name.get(f'bone_{bone_index}')
            if pose_bone is None and bone_index < len(pose_bones):
                pose_bone = pose_bones[bone_index]
            if pose_bone is not None and pose_bone.name not in used_names:
                mapped[int(bone_index)] = pose_bone
                used_names.add(pose_bone.name)
        return mapped

    @staticmethod
    def _preserve_mul_location_for_bone(bone_index: int) -> bool:
        try:
            return int(bone_index) < 2
        except Exception:
            return False

    @classmethod
    def _mul_channel_conversion_context(cls, default_matrix: mathutils.Matrix, parent_default_matrix: mathutils.Matrix) -> dict:
        parent_default_orientation = cls._mul_orientation_only_matrix(parent_default_matrix)
        default_orientation = cls._mul_orientation_only_matrix(default_matrix)
        local_default_orientation = parent_default_orientation.inverted_safe() @ default_orientation
        return {
            'parent_default_orientation': parent_default_orientation,
            'parent_default_orientation_inv': parent_default_orientation.inverted_safe(),
            'parent_default_orientation_3x3': parent_default_orientation.to_3x3(),
            'parent_default_orientation_3x3_inv': parent_default_orientation.to_3x3().inverted_safe(),
            'local_default_orientation': local_default_orientation,
            'local_default_orientation_inv': local_default_orientation.inverted_safe(),
        }

    @classmethod
    def _mul_source_local_to_channel_values(
        cls,
        source_local_matrix: mathutils.Matrix,
        conversion_context: dict,
        reference_rotation: tuple[float, float, float] | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        parent_default_orientation = conversion_context.get('parent_default_orientation', mathutils.Matrix.Identity(4))
        parent_default_orientation_inv = conversion_context.get('parent_default_orientation_inv', mathutils.Matrix.Identity(4))
        parent_default_orientation_3x3 = conversion_context.get('parent_default_orientation_3x3')
        local_default_orientation_inv = conversion_context.get('local_default_orientation_inv', mathutils.Matrix.Identity(4))
        if parent_default_orientation_3x3 is None:
            parent_default_orientation_3x3 = parent_default_orientation.to_3x3()

        local_location, local_rotation, local_scale = source_local_matrix.decompose()
        source_rotation = local_rotation.to_matrix().to_4x4() @ local_default_orientation_inv
        channel_rotation_matrix = parent_default_orientation @ source_rotation @ parent_default_orientation_inv
        try:
            if reference_rotation is not None:
                euler_compat = mathutils.Euler(tuple(float(reference_rotation[axis]) for axis in range(3)), 'XYZ')
                channel_euler = channel_rotation_matrix.to_3x3().to_euler('XYZ', euler_compat)
            else:
                channel_euler = channel_rotation_matrix.to_3x3().to_euler('XYZ')
        except Exception:
            channel_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')

        raw_location = parent_default_orientation_3x3 @ local_location
        scale = (float(local_scale[0]), float(local_scale[1]), float(local_scale[2]))
        rotation = (float(channel_euler.x), float(channel_euler.y), float(channel_euler.z))
        if reference_rotation is not None:
            rotation = tuple(cls._wrap_angle_near_reference(rotation[axis], float(reference_rotation[axis])) for axis in range(3))
        location = (float(raw_location[0]), float(raw_location[1]), float(raw_location[2]))
        return scale, rotation, location

    @classmethod
    def _armature_default_transforms(cls, armature_obj, num_bones: int) -> list[list[float]]:
        pose = getattr(armature_obj, 'pose', None)
        pose_bones = list(getattr(pose, 'bones', []) or [])
        if len(pose_bones) < int(num_bones):
            raise ValueError(f'Armature {armature_obj.name} has {len(pose_bones)} bone(s), but the target MUL skeleton needs {num_bones}')

        pose_bones_by_name = {pose_bone.name: pose_bone for pose_bone in pose_bones}
        used_names: set[str] = set()
        transforms: list[list[float]] = []
        for bone_index in range(int(num_bones)):
            pose_bone = pose_bones_by_name.get(f'bone_{bone_index}')
            if pose_bone is None and bone_index < len(pose_bones):
                pose_bone = pose_bones[bone_index]
            if pose_bone is None or pose_bone.name in used_names:
                raise ValueError(f'Armature {armature_obj.name} does not have a usable bone for MUL bone_{bone_index}')
            used_names.add(pose_bone.name)
            matrix = getattr(getattr(pose_bone, 'bone', None), 'matrix_local', None)
            if matrix is None:
                matrix = mathutils.Matrix.Identity(4)

            transforms.append(cls._flatten_matrix(cls._mul_orientation_only_matrix(matrix)))
        return transforms

    @staticmethod
    def _read_i32(data: bytes | bytearray, offset: int, endian: str) -> int:
        return int(struct.unpack_from(str(endian) + 'i', data, int(offset))[0])

    @classmethod
    def _legacy_matrix_offsets_for_instance(cls, data: bytes | bytearray, header_offset: int, header: dict, instance_id: int, endian: str) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        pos = int(header_offset) + 0x50
        if pos + 4 > len(data):
            return offsets
        num_anchors = cls._read_i32(data, pos, endian)
        pos += 4
        if num_anchors < 0 or num_anchors > 256:
            return offsets
        pos += int(num_anchors) * 12
        if pos + 4 > len(data):
            return offsets
        num_skeletons = cls._read_i32(data, pos, endian)
        pos += 4
        if num_skeletons < 0 or num_skeletons > 256:
            return offsets

        for _skeleton_index in range(int(num_skeletons)):
            if pos + 12 > len(data):
                break
            stored_instance_id = cls._read_i32(data, pos, endian)
            num_bones = cls._read_i32(data, pos + 4, endian)
            matrix_start = pos + 12
            matrix_end = matrix_start + (max(0, int(num_bones)) * 0x40)
            if num_bones < 0 or matrix_end > len(data):
                break
            if int(stored_instance_id) == int(instance_id):
                offsets.append((matrix_start, int(num_bones)))
            pos = matrix_end
        return offsets

    @staticmethod
    def _tru_matrix_offsets_for_instance(header: dict, header_offset: int, instance_id: int) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        for skeleton in list(header.get('skeletons', []) or []):
            try:
                stored_instance_id = int(skeleton.get('instance_id', -1))
                num_bones = int(skeleton.get('num_bones', 0))
                descriptor_offset = int(skeleton.get('descriptor_offset'))
            except Exception:
                continue
            if stored_instance_id == int(instance_id) and num_bones > 0:
                offsets.append((int(header_offset) + descriptor_offset + 16, num_bones))
        return offsets

    def _find_target_skeleton_offsets(self, data: bytes | bytearray, instance_id: int) -> tuple[list[tuple[int, int]], str]:
        platform_info = self._mul_detect_platform(bytes(data))
        endian = str(platform_info.get('endian', '<') or '<')
        packet_header_size = int(platform_info.get('packet_header_size', 0x10) or 0x10)
        packet_alignment = int(platform_info.get('packet_alignment', 0x10) or 0x10)

        packet_offset = MUL_STREAM_START_OFFSET
        while packet_offset + packet_header_size <= len(data):
            try:
                packet_type, packet_size = struct.unpack_from(endian + 'ii', data, packet_offset)
            except Exception:
                break
            packet_data_start = packet_offset + packet_header_size
            packet_data_end = packet_data_start + max(0, int(packet_size))
            if int(packet_size) < 0 or packet_data_end > len(data):
                break

            if int(packet_type) == MUL_PACKET_TYPE_CINEMATIC and bytes(data[packet_data_start:packet_data_start + 4]) in {b'ENIC', b'CINE'}:
                header, _frame_offset = self._read_mul_cine_header(bytes(data), packet_data_start, endian=endian)
                layout = str(header.get('cine_layout', '') or '')
                if layout == 'legacy':
                    offsets = self._legacy_matrix_offsets_for_instance(data, packet_data_start, header, instance_id, endian)
                else:
                    offsets = self._tru_matrix_offsets_for_instance(header, packet_data_start, instance_id)
                return offsets, layout or 'unknown'

            packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)

        return [], 'unknown'

    @staticmethod
    def _write_default_transforms(data: bytearray, matrix_start: int, num_bones: int, transforms: list[list[float]], endian: str) -> None:
        if len(transforms) < int(num_bones):
            raise ValueError(f'Armature default transform count {len(transforms)} is lower than target MUL bone count {num_bones}')
        for bone_index in range(int(num_bones)):
            offset = int(matrix_start) + (bone_index * 0x40)
            if offset < 0 or offset + 0x40 > len(data):
                raise ValueError(f'MUL skeleton matrix offset 0x{offset:X} is outside the file')
            values = list(transforms[bone_index])
            if len(values) != 16:
                raise ValueError(f'Generated default transform for bone_{bone_index} is invalid')
            struct.pack_into(str(endian) + '16f', data, offset, *(float(value) for value in values))

    def _read_cine_rows_for_retarget(self, data: bytes | bytearray) -> dict:
        platform_info = self._mul_detect_platform(bytes(data))
        endian = str(platform_info.get('endian', '<') or '<')
        packet_header_size = int(platform_info.get('packet_header_size', 0x10) or 0x10)
        packet_alignment = int(platform_info.get('packet_alignment', 0x10) or 0x10)

        cine_header: dict | None = None
        channel_values: list[float] | None = None
        channel_run_lengths: list[int] | None = None
        channel_run_types: list[int] | None = None
        run_segments: list[list[dict]] = []
        frame_rows: list[dict] = []
        packet_records: list[dict] = []
        frame_size_mod_counts: dict[int, int] = {}

        packet_offset = MUL_STREAM_START_OFFSET
        while packet_offset + packet_header_size <= len(data):
            try:
                packet_type, packet_size = struct.unpack_from(endian + 'ii', data, packet_offset)
            except Exception:
                break

            packet_data_start = packet_offset + packet_header_size
            packet_data_end = packet_data_start + max(0, int(packet_size))
            if int(packet_size) < 0 or packet_data_end > len(data):
                break

            if int(packet_type) != MUL_PACKET_TYPE_CINEMATIC:
                packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)
                continue

            packet_record = {
                'packet_offset': int(packet_offset),
                'packet_data_start': int(packet_data_start),
                'packet_data_end': int(packet_data_end),
                'packet_header': bytes(data[packet_offset:packet_data_start]),
                'has_cine_header': False,
                'header_bytes': b'',
                'frames': [],
                'tail_bytes': b'',
            }

            if cine_header is None:
                magic = bytes(data[packet_data_start:packet_data_start + 4])
                if magic not in {b'ENIC', b'CINE'}:
                    packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)
                    continue
                cine_header, frame_offset = self._read_mul_cine_header(bytes(data), packet_data_start, endian=endian)
                channel_count = int(cine_header.get('channel_count', 0) or 0)
                if channel_count <= 0:
                    raise ValueError('The MUL cinematic packet has no animation channels')
                channel_values = [0.0] * channel_count
                channel_run_lengths = [0] * channel_count
                channel_run_types = [0] * channel_count
                run_segments = [[] for _ in range(channel_count)]
                packet_record['has_cine_header'] = True
                packet_record['header_bytes'] = bytes(data[packet_data_start:int(frame_offset)])
            else:
                frame_offset = packet_data_start

            assert channel_values is not None and channel_run_lengths is not None and channel_run_types is not None and cine_header is not None

            while frame_offset + MUL_FRAME_HEADER_SIZE <= packet_data_end:
                frame_size, frame_number = struct.unpack_from(endian + 'ii', data, frame_offset)
                frame_record_end = frame_offset + max(int(frame_size), 0)
                if int(frame_size) < MUL_FRAME_HEADER_SIZE or frame_record_end > packet_data_end:
                    break

                aligned_frame_end = min(packet_data_end, self._align_mul_offset(frame_record_end))
                frame_record = {
                    'frame_number': int(frame_number),
                    'row_index': None,
                    'raw': bytes(data[frame_offset:aligned_frame_end]),
                }

                if int(frame_number) < 0:
                    packet_record['frames'].append(frame_record)
                    frame_offset = aligned_frame_end
                    continue

                frame_size_mod = int(frame_size) % MUL_ALIGNMENT
                frame_size_mod_counts[frame_size_mod] = int(frame_size_mod_counts.get(frame_size_mod, 0)) + 1

                frame_index = len(frame_rows)
                value_offset = frame_offset + MUL_FRAME_HEADER_SIZE
                value_offsets: dict[int, int] = {}

                for channel_index in range(len(channel_values)):
                    read_channel_value = False
                    if channel_run_lengths[channel_index] == 0:
                        if value_offset + 4 > frame_record_end:
                            raise ValueError('Unexpected end of MUL channel run data')
                        channel_run = struct.unpack_from(endian + 'I', data, value_offset)[0]
                        value_offset += 4
                        run_length = int(channel_run & 0x0FFFFFFF)
                        run_type = int((channel_run >> 28) & 0xF)
                        if run_length <= 0:
                            raise ValueError(f'Invalid MUL channel run length {run_length} on channel {channel_index}')
                        channel_run_lengths[channel_index] = run_length
                        channel_run_types[channel_index] = run_type
                        run_segments[channel_index].append({
                            'start': int(frame_index),
                            'length': int(run_length),
                            'type': int(run_type),
                        })
                        read_channel_value = True
                    else:
                        read_channel_value = channel_run_types[channel_index] != 0

                    if read_channel_value:
                        if value_offset + 4 > frame_record_end:
                            raise ValueError('Unexpected end of MUL channel value data')
                        value_offsets[channel_index] = int(value_offset)
                        channel_values[channel_index] = struct.unpack_from(endian + 'f', data, value_offset)[0]
                        value_offset += 4

                    channel_run_lengths[channel_index] -= 1

                frame_rows.append({
                    'frame': int(frame_number),
                    'row': list(channel_values),
                    'value_offsets': value_offsets,
                })
                frame_record['row_index'] = int(frame_index)
                packet_record['frames'].append(frame_record)
                frame_offset = aligned_frame_end

            if frame_offset < packet_data_end:
                packet_record['tail_bytes'] = bytes(data[frame_offset:packet_data_end])
            packet_records.append(packet_record)
            packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)

        if cine_header is None:
            raise ValueError('No cinematic packet with a CINE header was found in the MUL file')

        return {
            'platform_info': platform_info,
            'endian': endian,
            'header': cine_header,
            'frame_rows': frame_rows,
            'run_segments': run_segments,
            'packet_records': packet_records,
            'frame_size_mod': max(frame_size_mod_counts.items(), key=lambda item: item[1])[0] if frame_size_mod_counts else (MUL_ALIGNMENT - 4),
        }

    @staticmethod
    def _skeleton_transform_channel_indices(skeleton: dict, channel_count: int) -> list[int]:
        channels: list[int] = []
        first_channel = int(skeleton.get('first_channel', -1))
        root_channel_count = int(skeleton.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT) or MUL_SKELETON_ROOT_CHANNEL_COUNT)
        bone_channel_stride = int(skeleton.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE) or MUL_BONE_CHANNEL_STRIDE)
        num_bones = int(skeleton.get('num_bones', 0) or 0)
        if first_channel < 0 or num_bones <= 0:
            return channels
        for bone_index in range(num_bones):
            channel_base = first_channel + root_channel_count + (bone_index * bone_channel_stride)
            for relative_index in range(MUL_BONE_FLAGS_OFFSET + 1):
                channel_index = channel_base + relative_index
                if 0 <= channel_index < int(channel_count):
                    channels.append(int(channel_index))
        return channels

    @staticmethod
    def _frame_info_from_row(row: list[float], skeleton: dict, bone_index: int) -> dict:
        first_channel = int(skeleton.get('first_channel', -1))
        root_channel_count = int(skeleton.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT) or MUL_SKELETON_ROOT_CHANNEL_COUNT)
        bone_channel_stride = int(skeleton.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE) or MUL_BONE_CHANNEL_STRIDE)
        channel_base = first_channel + root_channel_count + (int(bone_index) * bone_channel_stride)

        def value(relative_index: int, default: float = 0.0) -> float:
            channel_index = channel_base + int(relative_index)
            if 0 <= channel_index < len(row):
                try:
                    return float(row[channel_index])
                except Exception:
                    pass
            return float(default)

        return {
            'scale': (
                value(MUL_BONE_SCALE_OFFSET + 0, 1.0),
                value(MUL_BONE_SCALE_OFFSET + 1, 1.0),
                value(MUL_BONE_SCALE_OFFSET + 2, 1.0),
            ),
            'rotation': (
                value(MUL_BONE_ROTATION_OFFSET + 0, 0.0),
                value(MUL_BONE_ROTATION_OFFSET + 1, 0.0),
                value(MUL_BONE_ROTATION_OFFSET + 2, 0.0),
            ),
            'location': (
                value(MUL_BONE_LOCATION_OFFSET + 0, 0.0),
                value(MUL_BONE_LOCATION_OFFSET + 1, 0.0),
                value(MUL_BONE_LOCATION_OFFSET + 2, 0.0),
            ),
            'flags': value(MUL_BONE_FLAGS_OFFSET, 5.0),
        }

    @staticmethod
    def _write_frame_info_to_row(row: list[float], skeleton: dict, bone_index: int, frame_info: dict) -> None:
        first_channel = int(skeleton.get('first_channel', -1))
        root_channel_count = int(skeleton.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT) or MUL_SKELETON_ROOT_CHANNEL_COUNT)
        bone_channel_stride = int(skeleton.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE) or MUL_BONE_CHANNEL_STRIDE)
        channel_base = first_channel + root_channel_count + (int(bone_index) * bone_channel_stride)

        scale = tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0)))
        rotation = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
        location = tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0)))
        flags = float(frame_info.get('flags', 5.0))
        values = (
            scale[0], scale[1], scale[2],
            rotation[0], rotation[1], rotation[2],
            location[0], location[1], location[2],
            flags,
        )
        for relative_index, value in enumerate(values):
            channel_index = channel_base + relative_index
            if 0 <= channel_index < len(row):
                row[channel_index] = float(value)

    def _retarget_skeleton_channel_rows(
        self,
        frame_rows: list[dict],
        output_rows: list[list[float]],
        skeleton: dict,
        armature_obj: bpy.types.Object,
        target_default_transforms: list[list[float]],
    ) -> None:
        num_bones = int(skeleton.get('num_bones', 0) or 0)
        if num_bones <= 0:
            return

        mapped = self._pose_bone_mapping_by_index(armature_obj, num_bones)
        if len(mapped) < num_bones:
            raise ValueError(f'Armature {armature_obj.name} has {len(mapped)} mapped bone(s), but the MUL skeleton needs {num_bones}')

        pose_bone_index_by_name = {pose_bone.name: bone_index for bone_index, pose_bone in mapped.items()}
        bone_order = sorted(mapped.keys(), key=lambda index: self._bone_depth(mapped[index]))
        old_default_transforms = list(skeleton.get('default_bone_transforms', []) or [])

        identity = mathutils.Matrix.Identity(4)
        old_default_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        old_parent_default_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        target_default_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        target_parent_default_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        target_conversion_context_by_bone: dict[int, dict] = {}
        old_build_context_by_bone: dict[int, dict] = {}
        parent_index_by_bone: dict[int, int | None] = {}
        target_rest_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        target_rest_local_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        target_rest_inverse_by_bone: dict[int, mathutils.Matrix] = {}
        target_rest_local_inverse_by_bone: dict[int, mathutils.Matrix] = {}
        target_parent_rest_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        target_length_by_bone: dict[int, float] = {}
        source_length_by_bone: dict[int, float] = {}

        for bone_index in bone_order:
            pose_bone = mapped[bone_index]
            parent_index = pose_bone_index_by_name.get(pose_bone.parent.name) if pose_bone.parent is not None else None
            parent_index_by_bone[bone_index] = parent_index

            old_default = self._matrix_from_values(old_default_transforms[bone_index]) if bone_index < len(old_default_transforms) else identity.copy()
            old_parent_default = self._matrix_from_values(old_default_transforms[parent_index]) if parent_index is not None and parent_index < len(old_default_transforms) else identity.copy()
            target_default = self._matrix_from_values(target_default_transforms[bone_index]) if bone_index < len(target_default_transforms) else identity.copy()
            target_parent_default = self._matrix_from_values(target_default_transforms[parent_index]) if parent_index is not None and parent_index < len(target_default_transforms) else identity.copy()
            old_default_matrix_by_bone[bone_index] = old_default
            old_parent_default_matrix_by_bone[bone_index] = old_parent_default
            target_default_matrix_by_bone[bone_index] = target_default
            target_parent_default_matrix_by_bone[bone_index] = target_parent_default

            old_build_context_by_bone[bone_index] = self._mul_channel_conversion_context(old_default, old_parent_default)
            target_conversion_context_by_bone[bone_index] = self._mul_channel_conversion_context(target_default, target_parent_default)

            target_rest = pose_bone.bone.matrix_local.copy()
            target_rest_matrix_by_bone[bone_index] = target_rest
            if pose_bone.parent is None:
                target_rest_local = target_rest.copy()
                target_rest_inverse_by_bone[bone_index] = target_rest.inverted_safe()
                target_parent_rest_matrix_by_bone[bone_index] = identity.copy()
            else:
                target_parent_rest = pose_bone.parent.bone.matrix_local.copy()
                target_parent_rest_matrix_by_bone[bone_index] = target_parent_rest
                target_rest_local = target_parent_rest.inverted_safe() @ target_rest
                target_rest_local_inverse_by_bone[bone_index] = target_rest_local.inverted_safe()
            target_rest_local_matrix_by_bone[bone_index] = target_rest_local

            if parent_index is not None:
                try:
                    target_length_by_bone[bone_index] = float(target_rest_local.to_translation().length)
                except Exception:
                    target_length_by_bone[bone_index] = 0.0
                try:
                    source_length_by_bone[bone_index] = float((old_parent_default.inverted_safe() @ old_default).to_translation().length)
                except Exception:
                    source_length_by_bone[bone_index] = 0.0

        def build_old_source_local_matrix(bone_index: int, frame_info: dict) -> mathutils.Matrix:
            context = old_build_context_by_bone[bone_index]
            location = mathutils.Vector(tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0))))
            location = context['parent_default_orientation_3x3_inv'] @ location
            rotation_values = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
            scale_values = tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0)))
            channel_rotation = self._mul_euler_xyz_to_matrix(rotation_values)
            source_rotation = (
                context['parent_default_orientation_inv']
                @ channel_rotation
                @ context['parent_default_orientation']
            )
            return (
                mathutils.Matrix.Translation(location)
                @ source_rotation
                @ context['local_default_orientation']
                @ mathutils.Matrix.Diagonal((float(scale_values[0]), float(scale_values[1]), float(scale_values[2]), 1.0))
            )

        def source_to_target_pose_basis(bone_index: int, source_matrix: mathutils.Matrix, parent_source_matrix: mathutils.Matrix | None) -> mathutils.Matrix:
            if parent_index_by_bone.get(bone_index) is None:
                return target_rest_inverse_by_bone[bone_index] @ source_matrix
            parent_pose_matrix = parent_source_matrix if parent_source_matrix is not None else target_parent_rest_matrix_by_bone[bone_index]
            return target_rest_local_inverse_by_bone[bone_index] @ parent_pose_matrix.inverted_safe() @ source_matrix

        reference_location_by_bone: dict[int, mathutils.Vector] = {}
        reference_source_local_by_bone: dict[int, mathutils.Matrix] = {}
        retarget_location_scale_by_bone: dict[int, float] = {}
        previous_rotation_by_bone: dict[int, tuple[float, float, float]] = {}

        def retarget_location_scale(bone_index: int) -> float:
            cached = retarget_location_scale_by_bone.get(bone_index)
            if cached is not None:
                return float(cached)
            scale = 1.0
            if parent_index_by_bone.get(bone_index) is not None:
                target_length = float(target_length_by_bone.get(bone_index, 0.0) or 0.0)
                source_length = float(source_length_by_bone.get(bone_index, 0.0) or 0.0)
                if source_length <= 1.0e-6:
                    try:
                        source_length = float(reference_source_local_by_bone[bone_index].to_translation().length)
                    except Exception:
                        source_length = 0.0
                if source_length > 1.0e-6 and target_length > 1.0e-6:
                    scale = target_length / source_length
            retarget_location_scale_by_bone[bone_index] = float(scale)
            return float(scale)

        for frame_index, row_info in enumerate(frame_rows):
            original_row = list(row_info.get('row', []) or [])
            if frame_index >= len(output_rows):
                continue

            source_matrix_by_bone: dict[int, mathutils.Matrix] = {}
            target_pose_matrix_by_bone: dict[int, mathutils.Matrix] = {}
            for bone_index in bone_order:
                frame_info = self._frame_info_from_row(original_row, skeleton, bone_index)
                old_source_local = build_old_source_local_matrix(bone_index, frame_info)
                parent_index = parent_index_by_bone.get(bone_index)
                parent_source_matrix = source_matrix_by_bone.get(parent_index) if parent_index is not None else None
                source_matrix = parent_source_matrix @ old_source_local if parent_source_matrix is not None else old_source_local
                source_matrix_by_bone[bone_index] = source_matrix

                parent_target_pose_matrix = target_pose_matrix_by_bone.get(parent_index) if parent_index is not None else None
                pose_basis = source_to_target_pose_basis(bone_index, source_matrix, parent_target_pose_matrix)
                location, quaternion, _scale = pose_basis.decompose()

                if parent_index is None:
                    write_location = location
                else:
                    reference_location = reference_location_by_bone.get(bone_index)
                    if reference_location is None:
                        reference_location = location.copy()
                        reference_location_by_bone[bone_index] = reference_location
                        reference_source_local_by_bone[bone_index] = old_source_local.copy()
                    write_location = (location - reference_location) * retarget_location_scale(bone_index)

                adjusted_pose_basis = (
                    mathutils.Matrix.Translation(write_location)
                    @ quaternion.to_matrix().to_4x4()
                    @ mathutils.Matrix.Diagonal((1.0, 1.0, 1.0, 1.0))
                )
                target_source_local = target_rest_local_matrix_by_bone[bone_index] @ adjusted_pose_basis
                if parent_target_pose_matrix is not None:
                    target_pose_matrix_by_bone[bone_index] = parent_target_pose_matrix @ target_source_local
                else:
                    target_pose_matrix_by_bone[bone_index] = target_source_local.copy()

                raw_rotation = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
                reference_rotation = previous_rotation_by_bone.get(bone_index, raw_rotation)
                scale, rotation, channel_location = self._mul_source_local_to_channel_values(
                    target_source_local,
                    target_conversion_context_by_bone[bone_index],
                    reference_rotation,
                )
                previous_rotation_by_bone[bone_index] = rotation

                if self._preserve_mul_location_for_bone(bone_index):
                    channel_location = tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0)))

                self._write_frame_info_to_row(output_rows[frame_index], skeleton, bone_index, {
                    'scale': scale,
                    'rotation': rotation,
                    'location': channel_location,
                    'flags': frame_info.get('flags', 5.0),
                })

    @staticmethod
    def _float_bits(value: float, endian: str) -> bytes:
        try:
            return struct.pack(str(endian) + 'f', float(value))
        except Exception:
            return struct.pack(str(endian) + 'f', 0.0)

    @classmethod
    def _mul_values_equal_for_run(cls, left: float, right: float, endian: str) -> bool:
        return cls._float_bits(left, endian) == cls._float_bits(right, endian)

    @classmethod
    def _build_compressed_mul_run_plan(cls, output_rows: list[list[float]], endian: str) -> list[list[dict]]:
        frame_count = len(output_rows)
        channel_count = len(output_rows[0]) if output_rows else 0
        run_plan: list[list[dict]] = [[] for _ in range(channel_count)]
        if frame_count <= 0 or channel_count <= 0:
            return run_plan

        for channel_index in range(channel_count):
            frame_index = 0
            while frame_index < frame_count:
                static_length = 1
                current_value = output_rows[frame_index][channel_index]
                while (
                    frame_index + static_length < frame_count
                    and cls._mul_values_equal_for_run(current_value, output_rows[frame_index + static_length][channel_index], endian)
                ):
                    static_length += 1

                if static_length >= 2:
                    run_plan[channel_index].append({
                        'start': int(frame_index),
                        'length': int(static_length),
                        'type': 0,
                    })
                    frame_index += static_length
                    continue

                dynamic_start = frame_index
                frame_index += 1
                while frame_index < frame_count:
                    probe_static_length = 1
                    probe_value = output_rows[frame_index][channel_index]
                    while (
                        frame_index + probe_static_length < frame_count
                        and cls._mul_values_equal_for_run(probe_value, output_rows[frame_index + probe_static_length][channel_index], endian)
                    ):
                        probe_static_length += 1
                    if probe_static_length >= 2:
                        break
                    frame_index += 1

                run_plan[channel_index].append({
                    'start': int(dynamic_start),
                    'length': int(frame_index - dynamic_start),
                    'type': 1,
                })

        return run_plan

    @staticmethod
    def _run_segment_by_start(run_plan: list[list[dict]]) -> dict[tuple[int, int], dict]:
        segments_by_start: dict[tuple[int, int], dict] = {}
        for channel_index, channel_segments in enumerate(run_plan):
            for segment in channel_segments:
                try:
                    start = int(segment.get('start', -1))
                    if start >= 0:
                        segments_by_start[(int(start), int(channel_index))] = segment
                except Exception:
                    continue
        return segments_by_start

    @staticmethod
    def _encode_compressed_mul_frame(
        frame_number: int,
        frame_index: int,
        row: list[float],
        segments_by_start: dict[tuple[int, int], dict],
        active_run_remaining: list[int],
        active_run_types: list[int],
        endian: str,
        frame_size_mod: int = MUL_ALIGNMENT - 4,
    ) -> bytes:
        channel_count = len(row)
        payload = bytearray()
        for channel_index, value in enumerate(row):
            if active_run_remaining[channel_index] <= 0:
                segment = segments_by_start.get((int(frame_index), int(channel_index)))
                if segment is None:
                    # Fallback: a one-frame dynamic run keeps the stream valid
                    # even if the plan is somehow incomplete.
                    run_type = 1
                    run_length = 1
                else:
                    run_type = int(segment.get('type', 1) or 0)
                    run_length = max(1, int(segment.get('length', 1) or 1))
                channel_run = ((run_type & 0xF) << 28) | (run_length & 0x0FFFFFFF)
                payload.extend(struct.pack(str(endian) + 'I', int(channel_run)))
                active_run_remaining[channel_index] = int(run_length)
                active_run_types[channel_index] = int(run_type)
                payload.extend(struct.pack(str(endian) + 'f', float(value)))
            elif int(active_run_types[channel_index]) != 0:
                payload.extend(struct.pack(str(endian) + 'f', float(value)))

            active_run_remaining[channel_index] -= 1

        raw_frame_size = MUL_FRAME_HEADER_SIZE + len(payload)
        try:
            target_mod = int(frame_size_mod) % MUL_ALIGNMENT
        except Exception:
            target_mod = MUL_ALIGNMENT - 4
        inner_padding = (int(target_mod) - (int(raw_frame_size) % MUL_ALIGNMENT)) % MUL_ALIGNMENT
        frame_size = int(raw_frame_size) + int(inner_padding)
        frame_data = bytearray(frame_size)
        struct.pack_into(str(endian) + 'ii', frame_data, 0, int(frame_size), int(frame_number))
        frame_data[MUL_FRAME_HEADER_SIZE:MUL_FRAME_HEADER_SIZE + len(payload)] = payload

        aligned_size = MultiplexStreamImporterMixin._align_mul_offset(int(frame_size))
        if aligned_size > frame_size:
            frame_data.extend(b'\0' * (aligned_size - frame_size))
        return bytes(frame_data)

    def _rewrite_cine_packets_with_rows(self, data: bytearray, cine_info: dict, output_rows: list[list[float]]) -> None:
        platform_info = dict(cine_info.get('platform_info', {}) or {})
        endian = str(cine_info.get('endian', platform_info.get('endian', '<')) or '<')
        packet_header_size = int(platform_info.get('packet_header_size', 0x10) or 0x10)
        packet_alignment = int(platform_info.get('packet_alignment', 0x10) or 0x10)
        packet_records = {
            int(packet_record.get('packet_offset', -1)): packet_record
            for packet_record in list(cine_info.get('packet_records', []) or [])
        }
        channel_count = len(output_rows[0]) if output_rows else 0
        run_plan = self._build_compressed_mul_run_plan(output_rows, endian)
        segments_by_start = self._run_segment_by_start(run_plan)
        active_run_remaining = [0] * int(channel_count)
        active_run_types = [0] * int(channel_count)
        frame_size_mod = int(cine_info.get('frame_size_mod', MUL_ALIGNMENT - 4) or (MUL_ALIGNMENT - 4))

        rebuilt = bytearray(data[:MUL_STREAM_START_OFFSET])
        packet_offset = MUL_STREAM_START_OFFSET
        while packet_offset + packet_header_size <= len(data):
            try:
                packet_type, packet_size = struct.unpack_from(endian + 'ii', data, packet_offset)
            except Exception:
                break

            packet_data_start = packet_offset + packet_header_size
            packet_data_end = packet_data_start + max(0, int(packet_size))
            if int(packet_size) < 0 or packet_data_end > len(data):
                break

            packet_record = packet_records.get(int(packet_offset))
            if int(packet_type) == MUL_PACKET_TYPE_CINEMATIC and packet_record is not None:
                payload = bytearray()
                payload.extend(bytes(packet_record.get('header_bytes', b'') or b''))
                for frame_record in list(packet_record.get('frames', []) or []):
                    row_index = frame_record.get('row_index')
                    if row_index is not None and 0 <= int(row_index) < len(output_rows):
                        payload.extend(self._encode_compressed_mul_frame(
                            int(frame_record.get('frame_number', 0) or 0),
                            int(row_index),
                            list(output_rows[int(row_index)]),
                            segments_by_start,
                            active_run_remaining,
                            active_run_types,
                            endian,
                            frame_size_mod,
                        ))
                    else:
                        payload.extend(bytes(frame_record.get('raw', b'') or b''))
                payload.extend(bytes(packet_record.get('tail_bytes', b'') or b''))

                packet_header = bytearray(data[packet_offset:packet_data_start])
                struct.pack_into(str(endian) + 'i', packet_header, 4, len(payload))
                rebuilt.extend(packet_header)
                rebuilt.extend(payload)
            else:
                rebuilt.extend(data[packet_offset:packet_data_end])

            aligned_new_end = self._align_mul_offset_to(len(rebuilt), packet_alignment)
            if aligned_new_end > len(rebuilt):
                rebuilt.extend(b'\0' * (aligned_new_end - len(rebuilt)))
            packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)

        if packet_offset < len(data):
            rebuilt.extend(data[packet_offset:])
        data[:] = rebuilt


    @staticmethod
    def _collapse_retargeted_values_to_existing_mul_runs(
        output_rows: list[list[float]],
        run_segments: list[list[dict]],
        target_channels: set[int],
    ) -> int:
        collapsed_values = 0
        frame_count = len(output_rows)
        if frame_count <= 0:
            return 0

        for channel_index in sorted(int(channel) for channel in target_channels):
            if channel_index < 0 or channel_index >= len(run_segments):
                continue
            for segment in list(run_segments[channel_index] or []):
                try:
                    run_type = int(segment.get('type', 0) or 0)
                    start = int(segment.get('start', 0) or 0)
                    length = max(1, int(segment.get('length', 1) or 1))
                except Exception:
                    continue
                if run_type != 0:
                    continue
                if start < 0 or start >= frame_count:
                    continue
                end = min(frame_count, start + length)
                if end <= start:
                    continue
                if channel_index >= len(output_rows[start]):
                    continue

                stored_value = float(output_rows[start][channel_index])
                for frame_index in range(start + 1, end):
                    if channel_index >= len(output_rows[frame_index]):
                        continue
                    if not TRLAU_OT_replace_mul_skeleton._float_close(output_rows[frame_index][channel_index], stored_value):
                        collapsed_values += 1
                    output_rows[frame_index][channel_index] = stored_value

        return int(collapsed_values)

    def _validate_static_runs_for_in_place_patch(self, frame_rows: list[dict], output_rows: list[list[float]], run_segments: list[list[dict]], target_channels: set[int]) -> None:
        frame_count = len(frame_rows)
        for channel_index in sorted(target_channels):
            if channel_index < 0 or channel_index >= len(run_segments):
                continue
            for segment in run_segments[channel_index]:
                if int(segment.get('type', 0) or 0) != 0:
                    continue
                start = int(segment.get('start', 0) or 0)
                length = int(segment.get('length', 1) or 1)
                end = min(frame_count, start + max(1, length))
                if start < 0 or start >= frame_count or end <= start + 1:
                    continue
                first_value = float(output_rows[start][channel_index])
                for frame_index in range(start + 1, end):
                    if not self._float_close(output_rows[frame_index][channel_index], first_value):
                        raise ValueError(
                            f'Retargeting channel {channel_index} would require adding new key values inside an existing static MUL run. '
                            'This file needs a full MUL re-export for this skeleton change.'
                        )

    @staticmethod
    def _patch_channel_values(data: bytearray, frame_rows: list[dict], output_rows: list[list[float]], target_channels: set[int], endian: str) -> None:
        for frame_index, row_info in enumerate(frame_rows):
            if frame_index >= len(output_rows):
                continue
            value_offsets = dict(row_info.get('value_offsets', {}) or {})
            for channel_index in target_channels:
                value_offset = value_offsets.get(int(channel_index))
                if value_offset is None:
                    continue
                if int(channel_index) >= len(output_rows[frame_index]):
                    continue
                struct.pack_into(str(endian) + 'f', data, int(value_offset), float(output_rows[frame_index][int(channel_index)]))

    def _retarget_mul_keyframes_in_data(
        self,
        data: bytearray,
        armature_obj: bpy.types.Object,
        instance_id: int,
        target_default_transforms: list[list[float]],
    ) -> int:
        cine_info = self._read_cine_rows_for_retarget(data)
        header = dict(cine_info.get('header', {}) or {})
        frame_rows = list(cine_info.get('frame_rows', []) or [])
        run_segments = list(cine_info.get('run_segments', []) or [])
        endian = str(cine_info.get('endian', '<') or '<')

        matching_skeletons = [
            skeleton
            for skeleton in list(header.get('skeletons', []) or [])
            if int(skeleton.get('instance_id', -1)) == int(instance_id)
        ]
        if not matching_skeletons:
            raise ValueError(f'No MUL skeleton with InstanceID {instance_id} was found in the CINE header')
        if not frame_rows:
            return 0

        output_rows = [list(row_info.get('row', []) or []) for row_info in frame_rows]
        channel_count = len(output_rows[0]) if output_rows else 0
        target_channels: set[int] = set()
        for skeleton in matching_skeletons:
            self._retarget_skeleton_channel_rows(frame_rows, output_rows, skeleton, armature_obj, target_default_transforms)
            target_channels.update(self._skeleton_transform_channel_indices(skeleton, channel_count))

        collapsed_values = self._collapse_retargeted_values_to_existing_mul_runs(output_rows, run_segments, target_channels)
        try:
            self._last_mul_retarget_collapsed_values = int(getattr(self, '_last_mul_retarget_collapsed_values', 0) or 0) + int(collapsed_values)
        except Exception:
            self._last_mul_retarget_collapsed_values = int(collapsed_values)
        self._patch_channel_values(data, frame_rows, output_rows, target_channels, endian)
        return len(matching_skeletons)

    def _replace_skeleton_in_file(self, filepath: Path, armature_obj, instance_id: int) -> int:
        path = Path(filepath)
        data = bytearray(path.read_bytes())
        platform_info = self._mul_detect_platform(bytes(data))
        endian = str(platform_info.get('endian', '<') or '<')
        targets, _layout = self._find_target_skeleton_offsets(data, instance_id)
        if not targets:
            raise ValueError(f'No MUL skeleton with InstanceID {instance_id} was found')

        max_bones = max(num_bones for _matrix_start, num_bones in targets)
        transforms = self._armature_default_transforms(armature_obj, max_bones)
        retargeted_skeletons = self._retarget_mul_keyframes_in_data(data, armature_obj, int(instance_id), transforms)
        for matrix_start, num_bones in targets:
            self._write_default_transforms(data, matrix_start, num_bones, transforms, endian)

        if bool(self.create_backup):
            backup_path = path.with_name(f'{path.name}.bak')
            if not backup_path.exists():
                shutil.copy2(path, backup_path)
        path.write_bytes(bytes(data))
        return max(len(targets), int(retargeted_skeletons))

    def _selected_filepaths(self) -> list[Path]:
        paths: list[Path] = []
        if len(self.files) > 0:
            base_dir = Path(bpy.path.abspath(str(self.directory or '')))
            for file_item in self.files:
                name = str(getattr(file_item, 'name', '') or '')
                if not name:
                    continue
                candidate = Path(name)
                if not candidate.is_absolute():
                    candidate = base_dir / candidate
                paths.append(candidate)
        elif str(getattr(self, 'filepath', '') or ''):
            paths.append(Path(bpy.path.abspath(str(self.filepath))))
        return paths

    def invoke(self, context, event):
        armature_obj = self._active_armature(context)
        candidate_id = self._candidate_instance_id(armature_obj) if armature_obj is not None else None
        if candidate_id is not None:
            self.instance_id = int(candidate_id)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, 'instance_id')
        layout.prop(self, 'create_backup')
        armature_obj = self._active_armature(context)
        if armature_obj is not None:
            layout.label(text=f'Armature: {armature_obj.name}', icon='ARMATURE_DATA')

    @staticmethod
    def _get_windows_console_handle() -> int:
        try:
            import sys
            if sys.platform != 'win32':
                return 0
            import ctypes
            return int(ctypes.windll.kernel32.GetConsoleWindow() or 0)
        except Exception:
            return 0

    @classmethod
    def _is_windows_console_visible(cls) -> bool:
        try:
            hwnd = cls._get_windows_console_handle()
            if not hwnd:
                return False
            import ctypes
            return bool(ctypes.windll.user32.IsWindowVisible(hwnd))
        except Exception:
            return False

    @staticmethod
    def _focus_main_blender_window():
        try:
            import sys
            if sys.platform != 'win32':
                return
            import ctypes
            import os
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            current_pid = os.getpid()
            console_hwnd = int(kernel32.GetConsoleWindow() or 0)
            handles: list[int] = []

            EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def _enum_window(hwnd, _lparam):
                try:
                    if int(hwnd) == console_hwnd or not bool(user32.IsWindowVisible(hwnd)):
                        return True
                    pid = wintypes.DWORD()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                    if int(pid.value) != int(current_pid):
                        return True
                    title_length = int(user32.GetWindowTextLengthW(hwnd) or 0)
                    title_buffer = ctypes.create_unicode_buffer(title_length + 1)
                    if title_length > 0:
                        user32.GetWindowTextW(hwnd, title_buffer, title_length + 1)
                    title = str(title_buffer.value or '')
                    if title and 'Console' not in title:
                        handles.append(int(hwnd))
                except Exception:
                    pass
                return True

            user32.EnumWindows(EnumWindowsProc(_enum_window), 0)
            if not handles:
                return
            hwnd = handles[0]
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _show_system_console_for_mul_replace(self):
        self._mul_replace_console_opened_by_operator = False
        self._mul_replace_console_was_visible = self._is_windows_console_visible()

        try:
            if not bool(self._mul_replace_console_was_visible):
                console_toggle = getattr(getattr(bpy.ops, 'wm', None), 'console_toggle', None)
                if console_toggle is not None and console_toggle.poll():
                    console_toggle()
                    self._mul_replace_console_opened_by_operator = True
        except Exception:
            pass

    def _finish_system_console_for_mul_replace(self):
        try:
            if bool(getattr(self, '_mul_replace_console_opened_by_operator', False)) and self._is_windows_console_visible():
                console_toggle = getattr(getattr(bpy.ops, 'wm', None), 'console_toggle', None)
                if console_toggle is not None and console_toggle.poll():
                    console_toggle()
        except Exception:
            pass
        self._focus_main_blender_window()

    def _print_mul_replace_header(self):
        try:
            print(f'{self._ANSI_BLUE}__________________________________{self._ANSI_RESET}')
            print(f'{self._ANSI_BLUE}Tomb Raider Legend/Anniversary/Underworld Import/Export{self._ANSI_RESET}\n')
        except Exception:
            pass

    def _print_mul_replace_footer(self, succeeded: bool):
        try:
            if bool(succeeded):
                print(f'{self._ANSI_GREEN}__________________________________{self._ANSI_RESET}')
                print(f'{self._ANSI_GREEN}Tomb Raider LAU MUL skeleton replacement finished{self._ANSI_RESET}')
            else:
                print(f'{self._ANSI_RED}__________________________________{self._ANSI_RESET}')
                print(f'{self._ANSI_RED}Tomb Raider LAU MUL skeleton replacement finished with errors{self._ANSI_RESET}')
        except Exception:
            pass

    def _log_mul_replace_progress(self, path: Path):
        message = f'Replacing Instance ID {int(self.instance_id)} skeleton for {path.name}...'
        try:
            print(message, flush=True)
        except Exception:
            pass

    def execute(self, context):
        armature_obj = self._active_armature(context)
        if armature_obj is None:
            self.report({'ERROR'}, 'Select an armature object before replacing a MUL skeleton')
            return {'CANCELLED'}

        filepaths = self._selected_filepaths()
        if not filepaths:
            self.report({'ERROR'}, 'Select one or more .mul files')
            return {'CANCELLED'}

        self._show_system_console_for_mul_replace()

        replaced_files = 0
        replaced_skeletons = 0
        report_kind = {'INFO'}
        report_message = ''
        result = {'FINISHED'}
        failed: list[tuple[str, str]] = []

        try:
            self._print_mul_replace_header()
            self._last_mul_retarget_collapsed_values = 0
            for path in filepaths:
                try:
                    if path.suffix.lower() != '.mul':
                        raise ValueError('Not a .mul file')
                    self._log_mul_replace_progress(path)
                    count = self._replace_skeleton_in_file(path, armature_obj, int(self.instance_id))
                    replaced_files += 1
                    replaced_skeletons += int(count)
                except Exception as exc:
                    failed.append((path.name, str(exc)))

            if failed:
                first_name, first_error = failed[0]
                if replaced_files:
                    report_kind = {'WARNING'}
                    report_message = f'Replaced {replaced_skeletons} skeleton(s) in {replaced_files} file(s); failed {len(failed)} file(s). First failure: {first_name}: {first_error}'
                    result = {'FINISHED'}
                    self._print_mul_replace_footer(False)
                else:
                    report_kind = {'ERROR'}
                    report_message = f'Failed to replace MUL skeleton. {first_name}: {first_error}'
                    result = {'CANCELLED'}
                    self._print_mul_replace_footer(False)
            else:
                report_kind = {'INFO'}
                report_message = f'Replaced {replaced_skeletons} MUL skeleton(s) in {replaced_files} file(s)'
                result = {'FINISHED'}
                self._print_mul_replace_footer(True)
        finally:
            self._finish_system_console_for_mul_replace()

        self.report(report_kind, report_message)
        return result

class TRLAU_OT_reload_textures(bpy.types.Operator):
    bl_idname = 'trlau.reload_textures'
    bl_label = 'Reload Materials'
    bl_description = 'Refresh materials by Texture ID'
    bl_options = {'REGISTER', 'UNDO'}

    selected_only: BoolProperty(
        name='Selected Objects Only',
        default=False,
        description='Only reload textures for selected objects',
    )

    def execute(self, context):
        result = reload_textures_for_scene(context.scene, selected_only=self.selected_only)
        self.report({'INFO'}, f"Reloaded {result['materials']} material(s); refreshed textures on {result['updated']} material(s)")
        return {'FINISHED'}


_CONVERT_MATERIAL_TYPE_ITEMS = (
    ('pc_oldgen', 'PC Old-Gen', 'Legacy PC tpage material'),
    ('pc_nextgen', 'PC Next-Gen', 'PC next-generation PCMaterialData material'),
    ('ps2', 'PS2', 'PS2 / GS-state material layout'),
    ('psp', 'PSP', 'PSP material layout'),
    ('ps3', 'PS3', 'PS3 next-generation material layout'),
    ('xbox360', 'Xbox 360', 'Xbox 360 / Xenon material layout'),
)
_CONVERT_TYPES_WITH_NORMAL_SPECULAR = {'pc_nextgen', 'ps3', 'xbox360'}


def _iter_selected_mesh_materials(context):
    seen_materials: set[int] = set()
    for obj in getattr(context, 'selected_objects', []) or []:
        if obj is None or getattr(obj, 'type', None) != 'MESH':
            continue
        data = getattr(obj, 'data', None)
        materials = getattr(data, 'materials', None)
        if materials is None:
            continue
        for material in materials:
            if material is None:
                continue
            material_key = id(material)
            if material_key in seen_materials:
                continue
            seen_materials.add(material_key)
            yield material


def _iter_selected_mesh_material_entries(context):
    seen_materials: set[int] = set()
    for obj in getattr(context, 'selected_objects', []) or []:
        if obj is None or getattr(obj, 'type', None) != 'MESH':
            continue
        data = getattr(obj, 'data', None)
        materials = getattr(data, 'materials', None)
        if materials is None:
            continue
        for slot_index, material in enumerate(materials):
            if material is None:
                continue
            material_key = id(material)
            if material_key in seen_materials:
                continue
            seen_materials.add(material_key)
            yield material, obj, int(slot_index)


def _node_label_text(node) -> str:
    parts = []
    for attr in ('name', 'label'):
        try:
            value = str(getattr(node, attr, '') or '').strip()
        except Exception:
            value = ''
        if value:
            parts.append(value)
    try:
        image = getattr(node, 'image', None)
        if image is not None:
            parts.append(str(getattr(image, 'name', '') or ''))
            raw_path = str(getattr(image, 'filepath_raw', '') or getattr(image, 'filepath', '') or '')
            if raw_path:
                parts.append(Path(raw_path).name)
    except Exception:
        pass
    return ' '.join(parts).lower()


def _role_from_text(text: str) -> str | None:
    value = str(text or '').lower()
    if re.search(r'(^|[_\-.\s])(normal|norm|nrm|bump)([_\-.\s]|$)', value):
        return 'normal'
    if re.search(r'(^|[_\-.\s])(specular|spec|gloss|roughness|rough)([_\-.\s]|$)', value):
        return 'specular'
    if re.search(r'(^|[_\-.\s])(diffuse|diff|base|albedo|color|colour|col)([_\-.\s]|$)', value):
        return 'base'
    return None


def _upstream_images_from_socket(socket, depth: int = 0, seen_nodes: set[int] | None = None):
    if socket is None or depth > 8:
        return []
    if seen_nodes is None:
        seen_nodes = set()
    images = []
    try:
        links = list(getattr(socket, 'links', []) or [])
    except Exception:
        links = []
    for link in links:
        node = getattr(link, 'from_node', None)
        if node is None:
            continue
        node_key = id(node)
        if node_key in seen_nodes:
            continue
        seen_nodes.add(node_key)
        try:
            if getattr(node, 'bl_idname', '') == 'ShaderNodeTexImage':
                image = getattr(node, 'image', None)
                if image is not None:
                    images.append(image)
                continue
        except Exception:
            pass
        try:
            for input_socket in getattr(node, 'inputs', []) or []:
                images.extend(_upstream_images_from_socket(input_socket, depth + 1, seen_nodes))
        except Exception:
            pass
    return images


def _first_upstream_image(node, socket_names: tuple[str, ...]):
    if node is None:
        return None
    for socket_name in socket_names:
        try:
            socket = node.inputs.get(socket_name)
        except Exception:
            socket = None
        for image in _upstream_images_from_socket(socket):
            if image is not None:
                return image
    return None


def _find_blender_material_role_images(material) -> dict[str, object | None]:
    roles = {'base': None, 'normal': None, 'specular': None}
    node_tree = getattr(material, 'node_tree', None)
    if material is None or not getattr(material, 'use_nodes', False) or node_tree is None:
        return roles

    nodes = []
    try:
        nodes = list(node_tree.nodes)
    except Exception:
        nodes = []

    principled_nodes = [node for node in nodes if getattr(node, 'bl_idname', '') == 'ShaderNodeBsdfPrincipled']
    for bsdf in principled_nodes:
        if roles['base'] is None:
            roles['base'] = _first_upstream_image(bsdf, ('Base Color', 'Alpha'))
        if roles['normal'] is None:
            roles['normal'] = _first_upstream_image(bsdf, ('Normal',))
        if roles['specular'] is None:
            roles['specular'] = _first_upstream_image(bsdf, ('Specular IOR Level', 'Specular', 'Roughness', 'Metallic'))

    texture_nodes = []
    for node in nodes:
        if getattr(node, 'bl_idname', '') != 'ShaderNodeTexImage':
            continue
        image = getattr(node, 'image', None)
        if image is None:
            continue
        texture_nodes.append((node, image))
        role = _role_from_text(_node_label_text(node))
        if role is not None and roles.get(role) is None:
            roles[role] = image

    if roles['base'] is None:
        for node, image in texture_nodes:
            role = _role_from_text(_node_label_text(node))
            if role in {'normal', 'specular'}:
                continue
            roles['base'] = image
            break
        if roles['base'] is None and texture_nodes:
            roles['base'] = texture_nodes[0][1]

    return roles


def _image_assignment_key(image, fallback_key):
    if image is None:
        return fallback_key
    try:
        return ('image', str(getattr(image, 'name', '') or ''), id(image))
    except Exception:
        return ('image', id(image))


def _tag_image_texture_id(image, texture_id: int, role: str | None = None, target_material_type: str | None = None, layer_index: int | None = None) -> None:
    if image is None:
        return
    texture_id = int(texture_id) & 0x1FFF
    try:
        image['trlau_texture_id'] = texture_id
        if role:
            image['trlau_texture_role'] = str(role)
        if target_material_type in {'ps3', 'xbox360'} and role:
            image['trlau_ps3_texture_role'] = str(role)
    except Exception:
        pass


def _material_alpha_blend_hint(material) -> bool:
    try:
        if str(getattr(material, 'blend_method', '') or '').upper() in {'BLEND', 'HASHED'}:
            return True
    except Exception:
        pass
    try:
        if str(getattr(material, 'surface_render_method', '') or '').upper() in {'BLENDED', 'DITHERED'}:
            return True
    except Exception:
        pass
    try:
        flags = decode_tpage_flags(get_material_tpageid(material))
        return bool(int(flags.get('blend_value', 0)))
    except Exception:
        return False


def _apply_converted_texture_ids(material, target_material_type: str, base_id: int, normal_id: int = -1, specular_id: int = -1, owner=None) -> None:
    target_material_type = str(target_material_type or 'pc_oldgen').lower()
    base_id = int(base_id) & 0x1FFF
    normal_id = int(normal_id) if int(normal_id) >= 0 else -1
    specular_id = int(specular_id) if int(specular_id) >= 0 else -1

    convert_material_to_type(material, target_material_type, owner=owner)

    if target_material_type == 'pc_nextgen':
        layer_ids = [base_id, normal_id, specular_id] + [-1] * 5
        layer_enabled = [1, 1 if normal_id >= 0 else 0, 1 if specular_id >= 0 else 0] + [0] * 5
        set_material_panel_value(material, 'trlau_ui_pcng_diffuse_texture_id', int(base_id))
        set_material_panel_value(material, 'trlau_ui_pcng_normal_texture_id', int(normal_id))
        set_material_panel_value(material, 'trlau_ui_pcng_specular_texture_id', int(specular_id))
        if _material_alpha_blend_hint(material):
            set_material_panel_value(material, 'trlau_ui_pcng_blend_mode', '2')
        for index in range(8):
            set_material_panel_value(material, f'trlau_ui_pcng_layer{index}_enabled', bool(layer_enabled[index]))
            set_material_panel_value(material, f'trlau_ui_pcng_layer{index}_texture_id', int(layer_ids[index]))
        sync_pc_nextgen_material_from_panel(material)
        apply_pc_nextgen_material_settings(material, owner=owner)
        rename_material_from_tpageid(material)
        return

    if target_material_type in {'ps3', 'xbox360'}:
        set_material_panel_value(material, 'trlau_ui_ps3_diffuse_texture_id', int(base_id))
        set_material_panel_value(material, 'trlau_ui_ps3_normal_texture_id', int(normal_id))
        set_material_panel_value(material, 'trlau_ui_ps3_specular_texture_id', int(specular_id))
        set_material_panel_value(material, 'trlau_ui_ps3_external_render_stream', bool(normal_id >= 0 or specular_id >= 0))
        flags = decode_tpage_flags(get_material_tpageid(material))
        flags['texture_id'] = int(base_id) & 0x1FFF
        set_material_tpageid(material, encode_tpage_flags(flags))
        apply_ps3_material_settings(material, owner=owner)
        rename_material_from_tpageid(material)
        return

    flags = decode_tpage_flags(get_material_tpageid(material))
    flags['texture_id'] = int(base_id) & (0xFFFF if target_material_type == 'ps2' else 0x1FFF)
    set_material_panel_value(material, 'trlau_ui_texture_id', int(flags['texture_id']))
    set_material_tpageid(material, encode_tpage_flags(flags))
    if target_material_type == 'psp':
        try:
            material['trlau_psp_texture_id'] = int(base_id) & 0x1FFF
        except Exception:
            pass
    _assign_current_material_nodes(material, int(base_id))
    rename_material_from_tpageid(material)
    apply_trlau_material_settings(material, owner=owner)


class TRLAU_OT_assign_texture_ids(bpy.types.Operator):
    bl_idname = 'trlau.assign_texture_ids'
    bl_label = 'Convert Materials'
    bl_description = 'Convert selected Blender materials to a game material type and assign Texture IDs from a range'
    bl_options = {'REGISTER', 'UNDO'}

    target_material_type: EnumProperty(
        name='Target Type',
        description='Game material layout to author from the selected Blender materials',
        items=_CONVERT_MATERIAL_TYPE_ITEMS,
        default='pc_oldgen',
    )
    start_texture_id: IntProperty(
        name='Start Texture ID',
        description='First Texture ID in the assignment range',
        default=0,
        min=0,
        max=0x1FFF,
    )
    end_texture_id: IntProperty(
        name='End Texture ID',
        description='Last Texture ID in the assignment range',
        default=0x1FFF,
        min=0,
        max=0x1FFF,
    )
    enable_single_sided: BoolProperty(
        name='Enable Single Sided',
        description='Enable the Single Sided material flag on converted old-gen-compatible materials',
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return any(getattr(obj, 'type', None) == 'MESH' for obj in (getattr(context, 'selected_objects', []) or []))

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def _get_material_entries(self, context):
        return list(_iter_selected_mesh_material_entries(context))

    def _planned_assignments(self, context):
        entries = self._get_material_entries(context)
        supports_extra_maps = str(self.target_material_type) in _CONVERT_TYPES_WITH_NORMAL_SPECULAR
        assignments = []
        material_roles = []
        seen_keys = set()

        for material, owner, slot_index in entries:
            roles = _find_blender_material_role_images(material)
            material_roles.append((material, owner, roles))
            base_key = _image_assignment_key(roles.get('base'), ('material', id(material), 'base'))
            if base_key not in seen_keys:
                seen_keys.add(base_key)
                assignments.append((base_key, roles.get('base'), 'base', 0))
            if supports_extra_maps:
                normal = roles.get('normal')
                if normal is not None:
                    key = _image_assignment_key(normal, ('material', id(material), 'normal'))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        assignments.append((key, normal, 'normal', 1))
                specular = roles.get('specular')
                if specular is not None:
                    key = _image_assignment_key(specular, ('material', id(material), 'specular'))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        assignments.append((key, specular, 'specular', 2))
        return entries, material_roles, assignments

    def draw(self, context):
        layout = self.layout
        entries, material_roles, assignments = self._planned_assignments(context)
        material_count = len(entries)
        texture_count = len(assignments)
        range_start = min(int(self.start_texture_id), int(self.end_texture_id))
        range_end = max(int(self.start_texture_id), int(self.end_texture_id))
        available_ids = max(0, range_end - range_start + 1)
        end_used = range_start + texture_count - 1 if texture_count > 0 else range_start

        layout.prop(self, 'target_material_type')
        row = layout.row(align=True)
        row.prop(self, 'start_texture_id')
        row.prop(self, 'end_texture_id')
        if str(self.target_material_type) in {'pc_oldgen', 'ps2', 'psp'}:
            layout.prop(self, 'enable_single_sided')

        info_box = layout.box()
        info_box.label(text=f'Selected materials: {material_count}')
        info_box.label(text=f'Texture IDs needed: {texture_count}')
        info_box.label(text=f'Available IDs in range: {available_ids}')
        if texture_count > 0:
            info_box.label(text=f'Assignment range used: {range_start} to {min(end_used, range_end)}')
        if str(self.target_material_type) in _CONVERT_TYPES_WITH_NORMAL_SPECULAR:
            info_box.label(text='Maps: base color, normal, specular')
        else:
            info_box.label(text='Maps: base color only')

        if material_count == 0:
            warn = layout.box()
            warn.alert = True
            warn.label(text='No mesh materials found in the current selection.', icon='ERROR')
        elif texture_count > available_ids:
            warn = layout.box()
            warn.alert = True
            warn.label(text='Not enough Texture IDs available in the selected range.', icon='ERROR')
            warn.label(text=f'Need {texture_count}, but only {available_ids} available.')

    def execute(self, context):
        entries, material_roles, assignments = self._planned_assignments(context)
        material_count = len(entries)
        if material_count == 0:
            self.report({'WARNING'}, 'No materials found on the selected mesh objects')
            return {'CANCELLED'}

        range_start = min(int(self.start_texture_id), int(self.end_texture_id))
        range_end = max(int(self.start_texture_id), int(self.end_texture_id))
        available_ids = max(0, range_end - range_start + 1)
        if len(assignments) > available_ids:
            self.report({'ERROR'}, f'Not enough Texture IDs available: {len(assignments)} texture(s), {available_ids} ID(s) available from {range_start} to {range_end}')
            return {'CANCELLED'}

        assigned_by_key = {}
        for offset, (key, image, role, layer_index) in enumerate(assignments):
            texture_id = range_start + offset
            assigned_by_key[key] = texture_id
            _tag_image_texture_id(image, texture_id, role=role, target_material_type=str(self.target_material_type), layer_index=layer_index)

        converted = 0
        supports_extra_maps = str(self.target_material_type) in _CONVERT_TYPES_WITH_NORMAL_SPECULAR
        for material, owner, roles in material_roles:
            base_key = _image_assignment_key(roles.get('base'), ('material', id(material), 'base'))
            base_id = assigned_by_key.get(base_key, range_start)
            normal_id = -1
            specular_id = -1
            if supports_extra_maps and roles.get('normal') is not None:
                normal_key = _image_assignment_key(roles.get('normal'), ('material', id(material), 'normal'))
                normal_id = assigned_by_key.get(normal_key, -1)
            if supports_extra_maps and roles.get('specular') is not None:
                specular_key = _image_assignment_key(roles.get('specular'), ('material', id(material), 'specular'))
                specular_id = assigned_by_key.get(specular_key, -1)

            if str(self.target_material_type) in {'pc_oldgen', 'ps2', 'psp'} and self.enable_single_sided:
                flags = decode_tpage_flags(get_material_tpageid(material))
                flags['single_sided'] = 1
                set_material_tpageid(material, encode_tpage_flags(flags))
            _apply_converted_texture_ids(material, str(self.target_material_type), int(base_id), int(normal_id), int(specular_id), owner=owner)
            converted += 1

        texture_count = len(assignments)
        self.report({'INFO'}, f'Converted {converted} material(s) to {dict((key, label) for key, label, _ in _CONVERT_MATERIAL_TYPE_ITEMS).get(str(self.target_material_type), str(self.target_material_type))}; assigned {texture_count} Texture ID(s) from {range_start} to {range_start + max(texture_count - 1, 0)}')
        return {'FINISHED'}


class VIEW3D_PT_trlau_utilities(bpy.types.Panel):
    bl_label = 'Utilities'
    bl_idname = 'VIEW3D_PT_trlau_utilities'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_order = 9999

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'trlau_texture_id_assignment')
        layout.operator('trlau.toggle_component_visibility', icon='EMPTY_DATA')
        layout.operator('trlau.toggle_drawgroup_visibility', icon='HIDE_OFF')
        layout.operator('trlau.add_vertex_color', icon='ADD')
        layout.operator('trlau.remove_vertex_color', icon='REMOVE')
        layout.operator('trlau.reload_textures', icon='FILE_REFRESH')
        layout.operator('trlau.assign_texture_ids', icon='TEXTURE_DATA')
        layout.operator('trlau.rename_bones', icon='BONE_DATA')
        layout.operator('trlau.make_bones_contiguous', icon='ARMATURE_DATA')
        row = layout.row()
        row.enabled = TRLAU_OT_replace_mul_skeleton.poll(context)
        row.operator(TRLAU_OT_replace_mul_skeleton.bl_idname, text='Replace MUL Skeleton', icon='ARMATURE_DATA')
        layout.operator(TRLAU_OT_batch_replace_wave_audio.bl_idname, icon='SOUND')



def _ensure_wave_audio_operator_property_annotations() -> None:
    TRLAU_OT_batch_replace_wave_audio.__annotations__.update({
        'source_audio_path': StringProperty(
            name='Source Audio',
            description='Audio file to encode into the selected game Wave section files. WAV is built in; MP3/OGG/FLAC/M4A and similar formats use FFmpeg if available',
            subtype='FILE_PATH',
            default='',
        ),
        'target_wave_directory': StringProperty(
            name='Wave Folder',
            description='Folder containing the original game Wave section files to replace',
            subtype='DIR_PATH',
            default='',
        ),
        'output_directory': StringProperty(
            name='Output Folder',
            description='Where replaced Wave files will be written when Overwrite Originals is disabled. Leave empty to create a wave_replaced folder inside the Wave folder',
            subtype='DIR_PATH',
            default='',
        ),
        'overwrite_originals': BoolProperty(
            name='Overwrite Originals',
            description='Write the replaced Wave data back over the original files. Disable this to write copies instead',
            default=False,
        ),
    })


_ensure_wave_audio_operator_property_annotations()

classes = (
    TRLAU_OT_batch_replace_wave_audio,
    TRLAU_OT_toggle_drawgroup_visibility,
    TRLAU_OT_add_vertex_color,
    TRLAU_OT_remove_vertex_color,
    TRLAU_OT_rename_bones,
    TRLAU_OT_make_bones_contiguous,
    TRLAU_OT_replace_mul_skeleton,
    TRLAU_OT_reload_textures,
    TRLAU_OT_assign_texture_ids,
    VIEW3D_PT_trlau_utilities,
)
