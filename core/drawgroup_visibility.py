from __future__ import annotations

from pathlib import Path

import bpy

from ..builders.mesh_builder import MeshBuildSettings, MeshBuilder, _decode_tpage_flags
from .material_ui import find_material_image, get_material_reflection_mode, get_material_tpageid, get_material_draw_group, material_uses_vertex_colors

HIDDEN_NODE_LABEL = 'TRLAU_DrawgroupHidden'
HIDDEN_PROP = 'trlau_drawgroup_hidden'
HIDDEN_NODE_PROP = 'trlau_drawgroup_hidden_node'


def iter_scene_materials(scene):
    seen = set()
    for obj in getattr(scene, 'objects', []):
        if obj.type != 'MESH':
            continue
        for material in getattr(obj.data, 'materials', []) or []:
            if material is None:
                continue
            key = material.name_full
            if key in seen:
                continue
            seen.add(key)
            yield material



def ensure_material_visibility(material, hidden: bool):
    if not material.use_nodes or material.node_tree is None:
        return False

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = next((node for node in nodes if node.bl_idname == 'ShaderNodeOutputMaterial'), None)
    if output is None:
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)

    for link in list(output.inputs['Surface'].links):
        links.remove(link)

    if hidden:
        transparent = next((node for node in nodes if node.bl_idname == 'ShaderNodeBsdfTransparent' and (node.get(HIDDEN_NODE_PROP) or node.label == HIDDEN_NODE_LABEL)), None)
        if transparent is None:
            transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        try:
            transparent[HIDDEN_NODE_PROP] = True
            transparent.label = ''
        except Exception:
            pass
        transparent.location = (80, 0)
        links.new(transparent.outputs['BSDF'], output.inputs['Surface'])
        material.blend_method = 'BLEND'
        material[HIDDEN_PROP] = True
        return True

    hidden_node = next((node for node in nodes if node.bl_idname == 'ShaderNodeBsdfTransparent' and (node.get(HIDDEN_NODE_PROP) or node.label == HIDDEN_NODE_LABEL)), None)
    if hidden_node is not None:
        nodes.remove(hidden_node)

    builder = MeshBuilder(
        context=bpy.context,
        filepath=str(Path(bpy.app.tempdir or '/tmp') / 'trlau_restore.tr7aemesh'),
        settings=MeshBuildSettings(apply_segment_pivots=True, import_textures=True),
        collection=bpy.context.scene.collection,
    )
    tpage_flags = _decode_tpage_flags(int(get_material_tpageid(material)))
    image = find_material_image(material)
    reflection_mode = get_material_reflection_mode(material)
    reflective = reflection_mode != 'none'
    builder._setup_material_nodes(
        material,
        image,
        tpage_flags,
        use_vertex_colors=material_uses_vertex_colors(material),
        reflective=reflective,
        reflection_mode=reflection_mode,
    )
    material[HIDDEN_PROP] = False
    return True


def get_drawgroup_materials(scene, drawgroup: int):
    return [material for material in iter_scene_materials(scene) if int(get_material_draw_group(material)) == int(drawgroup)]


def get_tagged_scene_materials(scene):
    return [material for material in iter_scene_materials(scene)]
