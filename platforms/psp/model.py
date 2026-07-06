from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import bpy

from ...core.log import logger
from ...core.model_types import Face, ModelData
from ..common.model import decode_normal


def decode_psp_uv(vertex) -> Tuple[float, float]:
    decoded_uv = getattr(vertex, 'uv_decoded', None)
    if decoded_uv is not None:
        return (float(decoded_uv[0]), 1.0 - float(decoded_uv[1]))

    # Very unsure about this
    uvx, uvy = vertex.uv_raw
    return (float(int(uvx) & 0xFF) / 254.0, 1.0 - (float(int(uvy) & 0xFF) / 254.0))


class PspModelBuilderMixin:
    @staticmethod
    def _vector_sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _vector_cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0]),
        )

    @staticmethod
    def _vector_dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])

    def _orient_psp_faces_to_vertex_normals(
        self,
        faces: List[Face],
        verts: List[Tuple[float, float, float]],
        model: ModelData,
    ) -> List[Face]:
        uv_format = str(getattr(model, "uv_format", "") or "").lower()
        if uv_format not in {'psp', 'underworld', 'pc_nextgen'} or not faces or not model.vertices:
            return faces

        decoded_normals = [decode_normal(vertex) for vertex in model.vertices]
        oriented: List[Face] = []
        flipped = 0
        for face in faces:
            v0, v1, v2 = face
            if v0 >= len(verts) or v1 >= len(verts) or v2 >= len(verts):
                oriented.append(face)
                continue

            edge_a = self._vector_sub(verts[v1], verts[v0])
            edge_b = self._vector_sub(verts[v2], verts[v0])
            face_normal = self._vector_cross(edge_a, edge_b)
            normal_sum = (
                decoded_normals[v0][0] + decoded_normals[v1][0] + decoded_normals[v2][0],
                decoded_normals[v0][1] + decoded_normals[v1][1] + decoded_normals[v2][1],
                decoded_normals[v0][2] + decoded_normals[v1][2] + decoded_normals[v2][2],
            )
            if self._vector_dot(face_normal, normal_sum) < 0.0:
                oriented.append((v0, v2, v1))
                flipped += 1
            else:
                oriented.append(face)

        try:
            if uv_format == 'pc_nextgen':
                model.pc_nextgen_vertex_normal_oriented_face_count = int(flipped)
        except Exception:
            pass
        if flipped:
            logger.debug('Oriented %d %s faces to imported vertex normals', flipped, uv_format.upper())
        return oriented

    def _apply_psp_material_state(self, material, tpage_flags: Dict[str, int]) -> None:
        material.use_backface_culling = bool(tpage_flags['single_sided'])
        try:
            psp_blend = int(material.get('trlau_psp_blend', tpage_flags.get('blend_value', 0))) & 0xFF
        except Exception:
            psp_blend = int(tpage_flags.get('blend_value', 0)) & 0xF
        try:
            psp_render_flags = int(material.get('trlau_psp_render_flags', 0)) & 0xFF
        except Exception:
            psp_render_flags = 0
        try:
            psp_vertex_format_flags = int(material.get('trlau_psp_vertex_format_flags', 0)) & 0xFFFFFFFF
        except Exception:
            psp_vertex_format_flags = 0

        alpha_blend_hint = bool(psp_blend != 0 or psp_render_flags != 0 or (psp_vertex_format_flags & 0x40))
        material['trlau_psp_alpha_blend_hint'] = alpha_blend_hint
        material.blend_method = 'BLEND' if alpha_blend_hint else 'OPAQUE'
        try:
            material.shadow_method = 'NONE' if alpha_blend_hint else 'OPAQUE'
        except Exception:
            pass
        try:
            material.show_transparent_back = bool(alpha_blend_hint and not tpage_flags['single_sided'])
        except Exception:
            pass

    def _setup_psp_reflective_material_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        reflection_mode: str = 'env',
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ) -> None:
        """Build a PSP-only environment-mapped preview material.

        PSP model sections mark reflective primitives in PrimitiveInfo flags.
        This deliberately avoids the PC reflective material setup: no PC
        envMappedVertices mask, no texture or vertex alpha linked to the shader,
        and no transparent/additive alpha branch.  PSP vertex color, when
        present, only modulates RGB. Its alpha/mask channel is kept in the
        Blender color attribute but is not linked into this shader.
        """
        material.use_nodes = True
        self._apply_psp_material_state(material, tpage_flags)

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (420, 0)

        if self._apply_culling_mode_settings(material, tpage_flags['cull_mode']):
            self._create_invisible_shader(nodes, links, output)
            return

        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (-900, 0)

        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (-660, 0)
        mapping.vector_type = 'TEXTURE'
        mapping.inputs['Location'].default_value[0] = -2.0
        mapping.inputs['Location'].default_value[1] = 0.0
        mapping.inputs['Location'].default_value[2] = -2.0
        mapping.inputs['Rotation'].default_value[0] = 0.0
        mapping.inputs['Rotation'].default_value[1] = 0.0
        mapping.inputs['Rotation'].default_value[2] = 0.0
        scale_value = 2.0 if reflection_mode == 'eye_ref' else 4.0
        mapping.inputs['Scale'].default_value[0] = scale_value
        mapping.inputs['Scale'].default_value[1] = scale_value
        mapping.inputs['Scale'].default_value[2] = scale_value
        links.new(tex_coord.outputs['Reflection'], mapping.inputs['Vector'])

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-400, 0)
        tex_image.image = image
        tex_image.projection = 'BOX'
        try:
            tex_image.projection_blend = 1.0
        except Exception:
            pass
        try:
            tex_image.extension = 'EXTEND'
        except Exception:
            pass
        links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])

        color_output = tex_image.outputs.get('Color')
        if use_vertex_colors or vertex_color_attribute_name:
            vertex_color = self._create_vertex_color_node(
                nodes,
                location=(-400, -190),
                attribute_name=vertex_color_attribute_name,
            )
            if vertex_color is not None and color_output is not None:
                multiply_color = nodes.new(type='ShaderNodeMixRGB')
                multiply_color.blend_type = 'MULTIPLY'
                multiply_color.inputs['Fac'].default_value = 1.0
                multiply_color.location = (-160, -60)
                links.new(color_output, multiply_color.inputs['Color1'])
                links.new(vertex_color.outputs['Color'], multiply_color.inputs['Color2'])
                color_output = multiply_color.outputs['Color']

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (-40, 0)
        base_color = bsdf.inputs.get('Base Color')
        if base_color is not None and color_output is not None:
            links.new(color_output, base_color)
        roughness_input = bsdf.inputs.get('Roughness')
        if roughness_input is not None:
            roughness_input.default_value = 0.18
        metallic_input = bsdf.inputs.get('Metallic')
        if metallic_input is not None:
            metallic_input.default_value = 0.0
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            specular_input.default_value = 0.0

        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    def _setup_psp_material_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ) -> None:
        """Build a PSP diffuse preview material without PC alpha behavior."""
        material.use_nodes = True
        self._apply_psp_material_state(material, tpage_flags)

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (520, 0)

        if self._apply_culling_mode_settings(material, tpage_flags['cull_mode']):
            self._create_invisible_shader(nodes, links, output)
            return

        color_output = None
        tex_image = None
        if image is not None:
            tex_image = nodes.new(type='ShaderNodeTexImage')
            tex_image.location = (-520, 0)
            tex_image.image = image
            if self._is_uv_scroll_material(material):
                self._ensure_uv_scroll_nodes(material, tex_image)
            color_output = tex_image.outputs.get('Color')

        if use_vertex_colors or vertex_color_attribute_name:
            vertex_color = self._create_vertex_color_node(
                nodes,
                location=(-520, -190),
                attribute_name=vertex_color_attribute_name,
            )
            if vertex_color is not None:
                if color_output is not None:
                    multiply_color = nodes.new(type='ShaderNodeMixRGB')
                    multiply_color.blend_type = 'MULTIPLY'
                    multiply_color.inputs['Fac'].default_value = 1.0
                    multiply_color.location = (-220, -30)
                    links.new(color_output, multiply_color.inputs['Color1'])
                    links.new(vertex_color.outputs['Color'], multiply_color.inputs['Color2'])
                    color_output = multiply_color.outputs['Color']
                else:
                    color_output = vertex_color.outputs.get('Color')

        texture_id = int(tpage_flags.get('texture_id', 0))
        if texture_id == 0 and image is None and not use_vertex_colors:
            color_output = None

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (120, 0)
        base_color = bsdf.inputs.get('Base Color')
        if base_color is not None:
            if color_output is not None:
                links.new(color_output, base_color)
            else:
                try:
                    base_color.default_value = (1.0, 1.0, 1.0, 1.0)
                except Exception:
                    pass

        roughness_input = bsdf.inputs.get('Roughness')
        if roughness_input is not None:
            roughness_input.default_value = 0.45
        metallic_input = bsdf.inputs.get('Metallic')
        if metallic_input is not None:
            metallic_input.default_value = 0.0
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            specular_input.default_value = 0.0

        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
