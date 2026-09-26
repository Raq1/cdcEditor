from __future__ import annotations

import re

import bpy

from ..core.game_utils import normalize_game_value
from ..core.material_ui import (
    apply_pc_nextgen_material_settings,
    apply_ps3_material_settings,
    ensure_material_type_panel_prop,
    ensure_pc_nextgen_material_panel_props,
    get_material_platform_name,
    get_material_type_name,
    is_pc_nextgen_material,
)


_COLLISION_NAME_SUFFIX_RE = re.compile(r'\.\d{3}$')
_COLLISION_CLIENT_FLAG_LABELS = {
    0: 'Wall',
    32: 'Ground',
    4: 'Slope',
    1: 'Water',
    16: 'Water1',
    17: 'Water2',
    48: 'Snow',
}
_COLLISION_CLIENT_FLAG_COLORS = {
    0: (1.0, 0.0, 0.0),
    32: (0.0, 1.0, 0.0),
    4: (1.0, 1.0, 0.0),
    1: (0.0, 0.0, 1.0),
    48: (1.0, 1.0, 1.0),
}




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


def _draw_underworld_material_layout(layout, material) -> None:
    box = layout.box()
    box.prop(material, 'trlau_ui_tr8_double_sided', text='Double Sided')

def _base_object_name(obj) -> str:
    return _COLLISION_NAME_SUFFIX_RE.sub('', str(getattr(obj, 'name', '') or '').strip())


def _is_signal_object(obj) -> bool:
    if obj is None:
        return False
    if bool(obj.get('trlau_signal')) or bool(obj.get('trlau_signal_mesh')):
        return True
    if str(obj.get('trlau_type', '') or '') == 'Signal':
        return True
    return False


def _is_collision_mesh(obj) -> bool:
    if obj is None or getattr(obj, 'type', None) != 'MESH':
        return False
    if bool(obj.get('trlau_terrain_collision')):
        return True
    return _base_object_name(obj).endswith('_Collision')




def _is_ps3_material(material) -> bool:
    return material is not None and get_material_platform_name(material) in {'ps3', 'xbox360'}


def _is_pc_nextgen_material(material) -> bool:
    return is_pc_nextgen_material(material)

def _collision_material_name(client_flag: int) -> str:
    return _COLLISION_CLIENT_FLAG_LABELS.get(int(client_flag), str(int(client_flag)))


def _collision_material_color(client_flag: int) -> tuple[float, float, float, float]:
    r, g, b = _COLLISION_CLIENT_FLAG_COLORS.get(int(client_flag), (1.0, 0.0, 1.0))
    return (r, g, b, 0.05)


def _readonly_material_prop(box, material, label: str, prop_name: str):
    row = box.row(align=True)
    row.enabled = False
    row.prop(material, prop_name, text=label)


def _draw_ps3_xbox_material_layout(layout, material) -> None:
    texture_box = layout.box()
    texture_box.label(text='Texture Bindings')
    texture_box.prop(material, 'trlau_ui_ps3_diffuse_texture_id')
    texture_box.prop(material, 'trlau_ui_ps3_normal_texture_id')
    texture_box.prop(material, 'trlau_ui_ps3_specular_texture_id')
    texture_box.operator('trlau.rebuild_ps3_material_shader', icon='NODETREE')

    render_box = layout.box()
    render_box.label(text='Render Flags')
    render_box.prop(material, 'trlau_ui_blend_value')
    # PS3/Xbox 360 material words do not use the PC Cull Mode enum; the
    # observed console field is imported as Single Sided instead.
    render_box.prop(material, 'trlau_ui_single_sided')
    render_box.prop(material, 'trlau_ui_texture_wrap')
    render_box.prop(material, 'trlau_ui_alpha_ref')
    render_box.prop(material, 'trlau_ui_draw_group')


def _draw_pc_nextgen_material_layout(layout, material) -> None:
    ensure_pc_nextgen_material_panel_props(material)

    header_box = layout.box()
    header_box.label(text='PCMaterialData')
    derived = header_box.box()
    derived.label(text='Derived Texture Roles')
    _readonly_material_prop(derived, material, 'Diffuse', 'trlau_ui_pcng_diffuse_texture_id')
    _readonly_material_prop(derived, material, 'Normal', 'trlau_ui_pcng_normal_texture_id')
    _readonly_material_prop(derived, material, 'Specular', 'trlau_ui_pcng_specular_texture_id')
    header_box.operator('trlau.rebuild_pc_nextgen_material_shader', icon='NODETREE')

    render_box = layout.box()
    render_box.label(text='Render State')
    render_box.prop(material, 'trlau_ui_pcng_blend_mode')
    render_box.prop(material, 'trlau_ui_pcng_combiner_type')
    render_box.prop(material, 'trlau_ui_pcng_material_flags')
    render_box.prop(material, 'trlau_ui_pcng_poly_flags')
    render_box.prop(material, 'trlau_ui_pcng_double_sided')
    render_box.prop(material, 'trlau_ui_pcng_special_material_flag')
    render_box.prop(material, 'trlau_ui_pcng_opacity')
    render_box.prop(material, 'trlau_ui_draw_group')

    surface_box = layout.box()
    surface_box.label(text='Surface Parameters')
    surface_box.prop(material, 'trlau_ui_pcng_uv_auto_scroll_speed')
    surface_box.prop(material, 'trlau_ui_pcng_sort_bias')
    surface_box.prop(material, 'trlau_ui_pcng_detail_range_mul')
    surface_box.prop(material, 'trlau_ui_pcng_detail_scale')
    surface_box.prop(material, 'trlau_ui_pcng_parallax_scale')
    surface_box.prop(material, 'trlau_ui_pcng_parallax_offset')
    surface_box.prop(material, 'trlau_ui_pcng_specular_power')
    surface_box.prop(material, 'trlau_ui_pcng_specular_shift0')
    surface_box.prop(material, 'trlau_ui_pcng_specular_shift1')

    rim_box = layout.box()
    rim_box.label(text='Rim Light')
    rim_box.prop(material, 'trlau_ui_pcng_rim_light_color')
    rim_box.prop(material, 'trlau_ui_pcng_rim_light_intensity')

    water_box = layout.box()
    water_box.label(text='Water')
    water_box.prop(material, 'trlau_ui_pcng_water_blend_bias')
    water_box.prop(material, 'trlau_ui_pcng_water_blend_exponent')
    water_box.prop(material, 'trlau_ui_pcng_water_deep_color')

    shader_box = layout.box()
    shader_box.label(text='Shader Indices')
    for index, label in enumerate((
        'MultiPass Light VS 2.0',
        'MultiPass Light VS 3.0',
        'SinglePass Light VS 2.0',
        'SinglePass Light VS 3.0',
        'SinglePass Light PS 2.0',
        'SinglePass Light PS 3.0',
        'SinglePass Light FX PS 2.0',
        'SinglePass Light FX PS 3.0',
    )):
        shader_box.prop(material, f'trlau_ui_pcng_shader_index{index}', text=label)

    layers_box = layout.box()
    layers_box.label(text='PCMaterialDataLayer[8]')
    layer_roles = {
        0: 'Diffuse / Base',
        1: 'Normal',
        2: 'Specular / Gloss',
    }
    for index in range(8):
        role = layer_roles.get(index, f'Layer {index}')
        layer_box = layers_box.box()
        layer_box.label(text=f'Layer {index} - {role}')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_enabled')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_texture_id')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_texcoord_source')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_modifier')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_param_id')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_color')
        layer_box.prop(material, f'trlau_ui_pcng_layer{index}_constant')


def _ensure_collision_material(client_flag: int) -> bpy.types.Material:
    client_flag = int(client_flag)
    name = _collision_material_name(client_flag)
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.blend_method = 'BLEND'
    if hasattr(material, 'shadow_method'):
        material.shadow_method = 'NONE'
    color = _collision_material_color(client_flag)
    material.diffuse_color = color
    material['trlau_collision_client_flag'] = client_flag
    material['trlau_collision_client_flag_label'] = name
    principled = material.node_tree.nodes.get('Principled BSDF') if material.node_tree is not None else None
    if principled is not None:
        principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
        principled.inputs['Alpha'].default_value = color[3]
    return material


def _ensure_material_slot(obj, material: bpy.types.Material) -> int:
    mesh = getattr(obj, 'data', None)
    if mesh is None:
        return 0
    for index, existing in enumerate(mesh.materials):
        if existing == material or (existing is not None and existing.name == material.name):
            return index
    mesh.materials.append(material)
    return len(mesh.materials) - 1


class TRLAU_OT_set_collision_faces(bpy.types.Operator):
    bl_idname = 'trlau.set_collision_faces'
    bl_label = 'Set Collision Faces'
    bl_description = '\u00A0'
    bl_options = {'UNDO'}

    client_flag: bpy.props.IntProperty(default=0)

    @classmethod
    def poll(cls, context):
        return _is_collision_mesh(getattr(context, 'object', None))

    def execute(self, context):
        obj = context.object
        if not _is_collision_mesh(obj):
            self.report({'ERROR'}, 'Select a collision mesh')
            return {'CANCELLED'}
        mesh = getattr(obj, 'data', None)
        if mesh is None:
            self.report({'ERROR'}, 'Selected object has no mesh data')
            return {'CANCELLED'}

        material = _ensure_collision_material(int(self.client_flag))
        slot_index = _ensure_material_slot(obj, material)
        changed = 0

        if getattr(obj, 'mode', 'OBJECT') == 'EDIT':
            import bmesh
            bm = bmesh.from_edit_mesh(mesh)
            for face in bm.faces:
                if face.select:
                    face.material_index = slot_index
                    changed += 1
            bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        else:
            for poly in mesh.polygons:
                if poly.select:
                    poly.material_index = slot_index
                    changed += 1
            mesh.update()

        if changed <= 0:
            self.report({'WARNING'}, 'No selected faces to update')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Set {changed} collision face(s) to {_collision_material_name(int(self.client_flag))}')
        return {'FINISHED'}



class TRLAU_OT_rebuild_ps3_material_shader(bpy.types.Operator):
    bl_idname = 'trlau.rebuild_ps3_material_shader'
    bl_label = 'Rebuild PS3 Shader'
    bl_description = '\u00A0'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        material = getattr(obj, 'active_material', None) if obj is not None else None
        return _is_ps3_material(material)

    def execute(self, context):
        obj = context.object
        material = obj.active_material
        apply_ps3_material_settings(material, owner=obj)
        self.report({'INFO'}, 'Rebuilt PS3 material shader')
        return {'FINISHED'}

class TRLAU_OT_rebuild_pc_nextgen_material_shader(bpy.types.Operator):
    bl_idname = 'trlau.rebuild_pc_nextgen_material_shader'
    bl_label = 'Rebuild PC Next Gen Shader'
    bl_description = '\u00A0'
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        material = getattr(obj, 'active_material', None) if obj is not None else None
        return _is_pc_nextgen_material(material)

    def execute(self, context):
        obj = context.object
        material = obj.active_material
        ensure_pc_nextgen_material_panel_props(material)
        apply_pc_nextgen_material_settings(material, owner=obj)
        self.report({'INFO'}, 'Rebuilt PC Next Gen material shader')
        return {'FINISHED'}


class VIEW3D_PT_trlau_material(bpy.types.Panel):
    bl_label = 'Material'
    bl_idname = 'VIEW3D_PT_trlau_material'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (
            obj is not None
            and getattr(obj, 'active_material', None) is not None
            and not _is_collision_mesh(obj)
            and not _is_signal_object(obj)
        )

    def draw(self, context):
        layout = self.layout
        obj = context.object
        material = obj.active_material

        if _is_underworld_model_context(obj, context):
            _draw_underworld_material_layout(layout, material)
            return

        ensure_material_type_panel_prop(material)
        type_box = layout.box()
        type_box.prop(material, 'trlau_ui_material_type', text='Type')

        material_type = get_material_type_name(material)
        if material_type == 'pc_nextgen':
            _draw_pc_nextgen_material_layout(layout, material)
            return
        if material_type in {'ps3', 'xbox360'}:
            _draw_ps3_xbox_material_layout(layout, material)
            return

        flags_box = layout.box()
        material_platform = get_material_platform_name(material)
        is_ps2_material = material_platform == 'ps2'

        if is_ps2_material:
            flags_box.label(text='PS2 TPage / GS State')
            flags_box.prop(material, 'trlau_ui_texture_id', text='Texture Section ID')
            flags_box.prop(material, 'trlau_ui_blend_value', text='Alpha / Blend Mode')
            flags_box.prop(material, 'trlau_ui_cull_mode', text='Render Mode')
            flags_box.prop(material, 'trlau_ui_ps2_double_sided', text='Double Sided')
            flags_box.label(text='Double Sided: off=single-sided, on=double-sided')
            flags_box.prop(material, 'trlau_ui_texture_wrap', text='Texture Address Mode')
            flags_box.label(text='Address: 0=Default/Clamp, 1=Repeat, 2/3=Experimental')
            flags_box.prop(material, 'trlau_ui_unknown_1', text='State Flag 23')
            flags_box.prop(material, 'trlau_ui_unknown_2', text='State Flag 24')
            flags_box.prop(material, 'trlau_ui_unknown_3', text='Mip / LOD Flag 27')
            flags_box.prop(material, 'trlau_ui_stencil_func', text='Alpha Texture Flag 28')
            flags_box.prop(material, 'trlau_ui_flat_shading', text='Flat Shading')
            flags_box.prop(material, 'trlau_ui_sort_z', text='Depth Sort / Render Pass')
        else:
            flags_box.prop(material, 'trlau_ui_texture_id')
            flags_box.prop(material, 'trlau_ui_blend_value')
            flags_box.prop(material, 'trlau_ui_cull_mode')
            flags_box.prop(material, 'trlau_ui_unknown_1')
            flags_box.prop(material, 'trlau_ui_single_sided')
            flags_box.prop(material, 'trlau_ui_texture_wrap')
            flags_box.prop(material, 'trlau_ui_unknown_2')
            flags_box.prop(material, 'trlau_ui_unknown_3')
            flags_box.prop(material, 'trlau_ui_flat_shading')
            flags_box.prop(material, 'trlau_ui_sort_z')
            flags_box.prop(material, 'trlau_ui_stencil_pass')
            flags_box.prop(material, 'trlau_ui_stencil_func')
            flags_box.prop(material, 'trlau_ui_alpha_ref')
        flags_box.prop(material, 'trlau_ui_env_mapping')
        flags_box.prop(material, 'trlau_ui_eye_ref_env_mapping')
        # Always expose Drawgroup.  Ported/new materials may not have the
        # backing custom property until the user edits this value, but export
        # already treats a missing value as drawgroup 0.
        flags_box.prop(material, 'trlau_ui_draw_group')



class VIEW3D_PT_trlau_ps3_material(bpy.types.Panel):
    bl_label = 'PS3 / Xbox 360 Material'
    bl_idname = 'VIEW3D_PT_trlau_ps3_material'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        material = getattr(obj, 'active_material', None) if obj is not None else None
        return (
            obj is not None
            and material is not None
            and _is_ps3_material(material)
            and not _is_underworld_model_context(obj, context)
            and not _is_collision_mesh(obj)
            and not _is_signal_object(obj)
        )

    @staticmethod
    def _label_prop(box, material, label: str, key: str, default=''):
        row = box.row(align=True)
        row.label(text=label)
        row.label(text=str(material.get(key, default)))

    def draw(self, context):
        layout = self.layout
        obj = context.object
        material = obj.active_material

        texture_box = layout.box()
        texture_box.label(text='Texture Bindings')
        texture_box.prop(material, 'trlau_ui_ps3_diffuse_texture_id')
        texture_box.prop(material, 'trlau_ui_ps3_normal_texture_id')
        texture_box.prop(material, 'trlau_ui_ps3_specular_texture_id')
        texture_box.operator('trlau.rebuild_ps3_material_shader', icon='NODETREE')

        render_box = layout.box()
        render_box.label(text='Render Flags')
        render_box.prop(material, 'trlau_ui_blend_value')
        # PS3/Xbox 360 material words do not use the PC Cull Mode enum; the
        # observed console field is imported as Single Sided instead.
        render_box.prop(material, 'trlau_ui_single_sided')
        render_box.prop(material, 'trlau_ui_texture_wrap')
        render_box.prop(material, 'trlau_ui_alpha_ref')
        # Always expose Drawgroup for PS3/Xbox 360-derived materials too.
        # Some ported materials only get trlau_draw_group after this value is
        # explicitly authored.
        render_box.prop(material, 'trlau_ui_draw_group')



class VIEW3D_PT_trlau_pc_nextgen_material(bpy.types.Panel):
    bl_label = 'PC Next Gen Material'
    bl_idname = 'VIEW3D_PT_trlau_pc_nextgen_material'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = getattr(context, 'object', None)
        material = getattr(obj, 'active_material', None) if obj is not None else None
        return (
            obj is not None
            and material is not None
            and _is_pc_nextgen_material(material)
            and not _is_underworld_model_context(obj, context)
            and not _is_collision_mesh(obj)
            and not _is_signal_object(obj)
        )

    @staticmethod
    def _readonly_prop(box, material, label: str, prop_name: str):
        row = box.row(align=True)
        row.enabled = False
        row.prop(material, prop_name, text=label)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        material = obj.active_material
        ensure_pc_nextgen_material_panel_props(material)

        header_box = layout.box()
        header_box.label(text='PCMaterialData')
        derived = header_box.box()
        derived.label(text='Derived Texture Roles')
        self._readonly_prop(derived, material, 'Diffuse', 'trlau_ui_pcng_diffuse_texture_id')
        self._readonly_prop(derived, material, 'Normal', 'trlau_ui_pcng_normal_texture_id')
        self._readonly_prop(derived, material, 'Specular', 'trlau_ui_pcng_specular_texture_id')
        header_box.operator('trlau.rebuild_pc_nextgen_material_shader', icon='NODETREE')

        render_box = layout.box()
        render_box.label(text='Render State')
        render_box.prop(material, 'trlau_ui_pcng_blend_mode')
        render_box.prop(material, 'trlau_ui_pcng_combiner_type')
        render_box.prop(material, 'trlau_ui_pcng_material_flags')
        render_box.prop(material, 'trlau_ui_pcng_poly_flags')
        render_box.prop(material, 'trlau_ui_pcng_double_sided')
        render_box.prop(material, 'trlau_ui_pcng_special_material_flag')
        render_box.prop(material, 'trlau_ui_pcng_opacity')
        render_box.prop(material, 'trlau_ui_draw_group')

        surface_box = layout.box()
        surface_box.label(text='Surface Parameters')
        surface_box.prop(material, 'trlau_ui_pcng_uv_auto_scroll_speed')
        surface_box.prop(material, 'trlau_ui_pcng_sort_bias')
        surface_box.prop(material, 'trlau_ui_pcng_detail_range_mul')
        surface_box.prop(material, 'trlau_ui_pcng_detail_scale')
        surface_box.prop(material, 'trlau_ui_pcng_parallax_scale')
        surface_box.prop(material, 'trlau_ui_pcng_parallax_offset')
        surface_box.prop(material, 'trlau_ui_pcng_specular_power')
        surface_box.prop(material, 'trlau_ui_pcng_specular_shift0')
        surface_box.prop(material, 'trlau_ui_pcng_specular_shift1')

        rim_box = layout.box()
        rim_box.label(text='Rim Light')
        rim_box.prop(material, 'trlau_ui_pcng_rim_light_color')
        rim_box.prop(material, 'trlau_ui_pcng_rim_light_intensity')

        water_box = layout.box()
        water_box.label(text='Water')
        water_box.prop(material, 'trlau_ui_pcng_water_blend_bias')
        water_box.prop(material, 'trlau_ui_pcng_water_blend_exponent')
        water_box.prop(material, 'trlau_ui_pcng_water_deep_color')

        shader_box = layout.box()
        shader_box.label(text='Shader Indices')
        for index, label in enumerate((
            'MultiPass Light VS 2.0',
            'MultiPass Light VS 3.0',
            'SinglePass Light VS 2.0',
            'SinglePass Light VS 3.0',
            'SinglePass Light PS 2.0',
            'SinglePass Light PS 3.0',
            'SinglePass Light FX PS 2.0',
            'SinglePass Light FX PS 3.0',
        )):
            shader_box.prop(material, f'trlau_ui_pcng_shader_index{index}', text=label)

        layers_box = layout.box()
        layers_box.label(text='PCMaterialDataLayer[8]')
        layer_roles = {
            0: 'Diffuse / Base',
            1: 'Normal',
            2: 'Specular / Gloss',
        }
        for index in range(8):
            role = layer_roles.get(index, f'Layer {index}')
            layer_box = layers_box.box()
            layer_box.label(text=f'Layer {index} - {role}')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_enabled')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_texture_id')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_texcoord_source')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_modifier')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_param_id')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_color')
            layer_box.prop(material, f'trlau_ui_pcng_layer{index}_constant')


class VIEW3D_PT_trlau_texture_scrolling(bpy.types.Panel):
    bl_label = 'Texture Scrolling'
    bl_idname = 'VIEW3D_PT_trlau_texture_scrolling'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        if not VIEW3D_PT_trlau_material.poll(context):
            return False
        obj = getattr(context, 'object', None)
        material = getattr(obj, 'active_material', None) if obj is not None else None
        if _is_underworld_model_context(obj, context):
            return False
        return get_material_type_name(material) not in {'pc_nextgen', 'ps3', 'xbox360'}

    def draw(self, context):
        layout = self.layout
        obj = context.object
        material = obj.active_material

        col = layout.column(align=True)
        col.prop(material, 'trlau_ui_scroll_enabled')
        speed_row = col.row(align=True)
        speed_row.enabled = bool(material.trlau_ui_scroll_enabled)
        speed_row.prop(material, 'trlau_ui_scroll_speed')


class VIEW3D_PT_trlau_collision_editing(bpy.types.Panel):
    bl_label = 'Collision Editing'
    bl_idname = 'VIEW3D_PT_trlau_collision_editing'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _is_collision_mesh(getattr(context, 'object', None))

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        for label, flag in (
            ('Set as Ground', 32),
            ('Set as Wall', 0),
            ('Set as Slope', 4),
            ('Set as Water', 1),
            ('Set as Snow', 48),
        ):
            op = col.operator('trlau.set_collision_faces', text=label)
            op.client_flag = int(flag)


classes = (
    TRLAU_OT_set_collision_faces,
    TRLAU_OT_rebuild_ps3_material_shader,
    TRLAU_OT_rebuild_pc_nextgen_material_shader,
    VIEW3D_PT_trlau_material,
    VIEW3D_PT_trlau_texture_scrolling,
    VIEW3D_PT_trlau_collision_editing,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
