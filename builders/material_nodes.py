from __future__ import annotations

import json

from typing import Dict, Optional, Tuple

import bpy

from ..core.log import logger


def _set_material_panel_value(material, name: str, value) -> None:
    from ..core.material_ui import set_material_panel_value
    set_material_panel_value(material, name, value)


def _get_material_platform_name(material) -> str:
    from ..core.material_ui import get_material_platform_name
    return get_material_platform_name(material)


def _decode_tpage_flags(tpageid: int) -> Dict[str, int]:
    tpageid = int(tpageid) & 0xFFFFFFFF
    return {
        'texture_id': tpageid & 0x1FFF,
        'blend_value': (tpageid >> 13) & 0xF,
        'cull_mode': (tpageid >> 17) & 0x7,
        'unknown_1': (tpageid >> 20) & 0x1,
        'single_sided': (tpageid >> 21) & 0x1,
        'texture_wrap': (tpageid >> 22) & 0x3,
        'unknown_2': (tpageid >> 24) & 0x1,
        'unknown_3': (tpageid >> 25) & 0x1,
        'flat_shading': (tpageid >> 26) & 0x1,
        'sort_z': (tpageid >> 27) & 0x1,
        'stencil_pass': (tpageid >> 28) & 0x3,
        'stencil_func': (tpageid >> 30) & 0x1,
        'alpha_ref': (tpageid >> 31) & 0x1,
    }


class MaterialNodeBuilderMixin:
    @staticmethod
    def _uses_transparency_overlap_for_cull(culling_mode: int) -> bool:
        return int(culling_mode) in {0, 3, 5, 6, 7}

    @staticmethod
    def _set_socket_default(node, socket_name: str, value) -> None:
        socket = node.inputs.get(socket_name)
        if socket is None:
            return
        try:
            socket.default_value = value
        except Exception:
            pass

    @staticmethod
    def _set_vector_socket_default(node, socket_name: str, values: Tuple[float, float, float]) -> None:
        socket = node.inputs.get(socket_name)
        if socket is None:
            return
        try:
            for axis, value in enumerate(values):
                socket.default_value[axis] = value
        except Exception:
            pass

    @staticmethod
    def _apply_culling_mode_settings(material, culling_mode: int) -> bool:
        culling_mode = int(culling_mode)
        is_invisible = culling_mode == 1

        try:
            if hasattr(material, 'use_transparency_overlap'):
                material.use_transparency_overlap = MaterialNodeBuilderMixin._uses_transparency_overlap_for_cull(culling_mode)
        except Exception:
            pass

        if is_invisible:
            for attr_name, value in (
                ('blend_method', 'BLEND'),
                ('shadow_method', 'NONE'),
                ('use_backface_culling', False),
            ):
                try:
                    setattr(material, attr_name, value)
                except Exception:
                    pass

        return is_invisible

    def _prepare_material_node_tree(
        self,
        material,
        tpage_flags: Dict[str, int],
        *,
        blend_method: str = 'OPAQUE',
        output_location: Tuple[int, int] = (860, 0),
        transparent_back: bool | None = None,
        shadow_method: str | None = None,
        screen_refraction: bool | None = None,
    ):
        material.use_nodes = True
        material.use_backface_culling = bool(tpage_flags['single_sided'])
        try:
            if hasattr(material, 'use_transparency_overlap'):
                material.use_transparency_overlap = self._uses_transparency_overlap_for_cull(tpage_flags.get('cull_mode', 0))
        except Exception:
            pass
        material.blend_method = blend_method
        if transparent_back is not None:
            try:
                material.show_transparent_back = bool(transparent_back)
            except Exception:
                pass
        if shadow_method is not None:
            try:
                material.shadow_method = shadow_method
            except Exception:
                pass
        if screen_refraction is not None:
            try:
                material.use_screen_refraction = bool(screen_refraction)
            except Exception:
                pass

        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = output_location
        return nodes, links, output

    @staticmethod
    def _create_texture_node(nodes, image: Optional[bpy.types.Image], location=(-450, 0)):
        if image is None:
            return None
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = location
        tex_image.image = image
        return tex_image

    def _create_scrolling_texture_node(self, material, nodes, image: Optional[bpy.types.Image], location=(-450, 0)):
        tex_image = self._create_texture_node(nodes, image, location=location)
        if tex_image is not None and self._is_uv_scroll_material(material):
            self._ensure_uv_scroll_nodes(material, tex_image)
        return tex_image

    @staticmethod
    def _create_alpha_greater_than_node(nodes, links, alpha_output, location=(-20, -190)):
        if alpha_output is None:
            return None

        greater_than = nodes.new(type='ShaderNodeMath')
        greater_than.operation = 'GREATER_THAN'
        greater_than.location = location
        greater_than.inputs[1].default_value = 0.0
        links.new(alpha_output, greater_than.inputs[0])
        return greater_than.outputs['Value']

    @staticmethod
    def _create_multiply_rgb_node(nodes, links, first_output, second_output, location=(-150, 0)):
        if first_output is None or second_output is None:
            return first_output or second_output
        multiply = nodes.new(type='ShaderNodeMixRGB')
        multiply.blend_type = 'MULTIPLY'
        multiply.inputs['Fac'].default_value = 1.0
        multiply.location = location
        links.new(first_output, multiply.inputs['Color1'])
        links.new(second_output, multiply.inputs['Color2'])
        return multiply.outputs['Color']

    @staticmethod
    def _create_multiply_value_node(nodes, links, first_output, second_output=None, *, default_second=1.0, location=(-150, -190)):
        if first_output is None and second_output is None:
            return None
        if first_output is None:
            return second_output
        if second_output is None and default_second is None:
            return first_output

        multiply = nodes.new(type='ShaderNodeMath')
        multiply.operation = 'MULTIPLY'
        multiply.location = location
        links.new(first_output, multiply.inputs[0])
        if second_output is not None:
            links.new(second_output, multiply.inputs[1])
        else:
            multiply.inputs[1].default_value = float(default_second)
        return multiply.outputs['Value']

    @staticmethod
    def _set_image_non_color(image) -> None:
        if image is None:
            return
        try:
            image.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    def _attach_auxiliary_pbr_maps(self, nodes, links, bsdf, normal_image=None, specular_image=None) -> None:
        if bsdf is None:
            return

        if normal_image is not None:
            self._set_image_non_color(normal_image)
            normal_tex = self._create_texture_node(nodes, normal_image, location=(-450, -360))
            if normal_tex is not None:
                try:
                    normal_tex.label = 'Normal Map'
                    normal_tex.name = 'TRLAU_NormalMap'
                except Exception:
                    pass
                normal_map = nodes.new(type='ShaderNodeNormalMap')
                normal_map.location = (-150, -360)
                normal_input = bsdf.inputs.get('Normal')
                if normal_input is not None:
                    try:
                        normal_color = normal_tex.outputs['Color']
                        if self._should_invert_ps3_normal_green(None, normal_image):
                            normal_color = self._create_inverted_green_normal_color(nodes, links, normal_color, location=(-410, -360))
                        links.new(normal_color, normal_map.inputs['Color'])
                        links.new(normal_map.outputs['Normal'], normal_input)
                    except Exception:
                        pass

        if specular_image is not None:
            self._set_image_non_color(specular_image)
            specular_tex = self._create_texture_node(nodes, specular_image, location=(-450, -620))
            if specular_tex is not None:
                try:
                    specular_tex.label = 'Specular Map'
                    specular_tex.name = 'TRLAU_SpecularMap'
                except Exception:
                    pass
                rgb_to_bw = nodes.new(type='ShaderNodeRGBToBW')
                rgb_to_bw.location = (-190, -620)
                specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
                if specular_input is not None:
                    try:
                        links.new(specular_tex.outputs['Color'], rgb_to_bw.inputs['Color'])
                        links.new(rgb_to_bw.outputs['Val'], specular_input)
                    except Exception:
                        pass

    @staticmethod
    def _safe_material_get(material, name: str, default=None):
        if material is None:
            return default
        try:
            if name in material:
                return material.get(name, default)
        except Exception:
            pass
        pcng_map = {
            'trlau_pc_nextgen_blend_mode': 'trlau_ui_pcng_blend_mode',
            'trlau_pc_nextgen_combiner_type': 'trlau_ui_pcng_combiner_type',
            'trlau_pc_nextgen_material_flags': 'trlau_ui_pcng_material_flags',
            'trlau_pc_nextgen_double_sided': 'trlau_ui_pcng_double_sided',
            'trlau_pc_nextgen_opacity': 'trlau_ui_pcng_opacity',
            'trlau_pc_nextgen_specular_power': 'trlau_ui_pcng_specular_power',
            'trlau_pc_nextgen_rim_light_intensity': 'trlau_ui_pcng_rim_light_intensity',
            'trlau_pc_nextgen_layer_enabled': 'trlau_ui_pcng_layer_enabled',
            'trlau_pc_nextgen_layer_colors': 'trlau_ui_pcng_layer_colors',
            'trlau_pc_nextgen_layer_texcoord_sources': 'trlau_ui_pcng_layer_texcoord_sources',
            'trlau_pc_nextgen_rim_light_color': 'trlau_ui_pcng_rim_light_color',
        }
        if name in pcng_map:
            try:
                if name == 'trlau_pc_nextgen_blend_mode':
                    return int(getattr(material, 'trlau_ui_pcng_blend_mode', default or 0))
                if name == 'trlau_pc_nextgen_combiner_type':
                    return int(getattr(material, 'trlau_ui_pcng_combiner_type', default or 0))
                if name == 'trlau_pc_nextgen_material_flags':
                    text = str(getattr(material, 'trlau_ui_pcng_material_flags', default or '0') or '0')
                    return int(text, 16) if text.lower().startswith('0x') else int(text)
                if name == 'trlau_pc_nextgen_double_sided':
                    return bool(getattr(material, 'trlau_ui_pcng_double_sided', default or False))
                if name in {'trlau_pc_nextgen_opacity', 'trlau_pc_nextgen_specular_power', 'trlau_pc_nextgen_rim_light_intensity'}:
                    return float(getattr(material, pcng_map[name], default or 0.0))
                if name == 'trlau_pc_nextgen_layer_enabled':
                    return ','.join('1' if bool(getattr(material, f'trlau_ui_pcng_layer{index}_enabled', False)) else '0' for index in range(8))
                if name == 'trlau_pc_nextgen_layer_colors':
                    return json.dumps([list(getattr(material, f'trlau_ui_pcng_layer{index}_color', (1.0, 1.0, 1.0, 1.0)))[:4] for index in range(8)])
                if name == 'trlau_pc_nextgen_layer_texcoord_sources':
                    return ','.join(str(int(getattr(material, f'trlau_ui_pcng_layer{index}_texcoord_source', '0') or '0')) for index in range(8))
                if name == 'trlau_pc_nextgen_rim_light_color':
                    return json.dumps([list(getattr(material, 'trlau_ui_pcng_rim_light_color', (0.0, 0.0, 0.0, 0.0)))[:4]])
            except Exception:
                pass
        try:
            value = getattr(material, name)
            if value is not None:
                return value
        except Exception:
            pass
        return default

    @staticmethod
    def _try_set_node_label(node, name: str, label: str) -> None:
        if node is None:
            return
        try:
            node.name = name
            node.label = label
        except Exception:
            pass

    @staticmethod
    def _socket_by_names(sockets, *names):
        for name in names:
            try:
                socket = sockets.get(name)
            except Exception:
                socket = None
            if socket is not None:
                return socket
        return None

    def _create_inverted_green_normal_color(self, nodes, links, color_output, location=(-520, -360)):
        if color_output is None:
            return None

        try:
            separate = nodes.new(type='ShaderNodeSeparateRGB')
            combine = nodes.new(type='ShaderNodeCombineRGB')
            sep_input = self._socket_by_names(separate.inputs, 'Image', 'Color')
            sep_r = self._socket_by_names(separate.outputs, 'R', 'Red')
            sep_g = self._socket_by_names(separate.outputs, 'G', 'Green')
            sep_b = self._socket_by_names(separate.outputs, 'B', 'Blue')
            comb_r = self._socket_by_names(combine.inputs, 'R', 'Red')
            comb_g = self._socket_by_names(combine.inputs, 'G', 'Green')
            comb_b = self._socket_by_names(combine.inputs, 'B', 'Blue')
            comb_out = self._socket_by_names(combine.outputs, 'Image', 'Color')
        except Exception:
            try:
                separate = nodes.new(type='ShaderNodeSeparateColor')
                combine = nodes.new(type='ShaderNodeCombineColor')
                try:
                    separate.mode = 'RGB'
                    combine.mode = 'RGB'
                except Exception:
                    pass
                sep_input = self._socket_by_names(separate.inputs, 'Color', 'Image')
                sep_r = self._socket_by_names(separate.outputs, 'Red', 'R')
                sep_g = self._socket_by_names(separate.outputs, 'Green', 'G')
                sep_b = self._socket_by_names(separate.outputs, 'Blue', 'B')
                comb_r = self._socket_by_names(combine.inputs, 'Red', 'R')
                comb_g = self._socket_by_names(combine.inputs, 'Green', 'G')
                comb_b = self._socket_by_names(combine.inputs, 'Blue', 'B')
                comb_out = self._socket_by_names(combine.outputs, 'Color', 'Image')
            except Exception:
                return color_output

        invert_green = nodes.new(type='ShaderNodeMath')
        invert_green.operation = 'SUBTRACT'
        separate.location = (location[0], location[1])
        invert_green.location = (location[0] + 210, location[1] - 18)
        combine.location = (location[0] + 420, location[1])
        try:
            invert_green.inputs[0].default_value = 1.0
        except Exception:
            pass

        if any(socket is None for socket in (sep_input, sep_r, sep_g, sep_b, comb_r, comb_g, comb_b, comb_out)):
            return color_output

        try:
            links.new(color_output, sep_input)
            links.new(sep_r, comb_r)
            links.new(sep_g, invert_green.inputs[1])
            links.new(invert_green.outputs['Value'], comb_g)
            links.new(sep_b, comb_b)
            return comb_out
        except Exception:
            return color_output

    @staticmethod
    def _link_matching_texture_vector(links, source_tex, target_tex) -> None:
        if source_tex is None or target_tex is None:
            return
        try:
            source_vector = source_tex.inputs.get('Vector')
            target_vector = target_tex.inputs.get('Vector')
            if source_vector is not None and target_vector is not None and source_vector.is_linked:
                links.new(source_vector.links[0].from_socket, target_vector)
        except Exception:
            pass

    @staticmethod
    def _sample_image_pixels(image, max_samples: int = 2048):
        if image is None:
            return []
        try:
            width = int(image.size[0])
            height = int(image.size[1])
        except Exception:
            return []
        total_pixels = max(0, width * height)
        if total_pixels <= 0:
            return []
        try:
            pixels = image.pixels
        except Exception:
            return []
        stride = max(1, total_pixels // max(1, int(max_samples)))
        samples = []
        try:
            for pixel_index in range(0, total_pixels, stride):
                base = pixel_index * 4
                samples.append((
                    float(pixels[base]),
                    float(pixels[base + 1]),
                    float(pixels[base + 2]),
                    float(pixels[base + 3]),
                ))
                if len(samples) >= max_samples:
                    break
        except Exception:
            return []
        return samples

    @staticmethod
    def _image_custom_bool(image, key: str, default=None):
        if image is None:
            return default
        try:
            value = image.get(key, default)
        except Exception:
            return default
        if value is default:
            return default
        return bool(value)

    @staticmethod
    def _image_custom_string(image, key: str, default: str = '') -> str:
        if image is None:
            return default
        try:
            value = image.get(key, default)
        except Exception:
            return default
        return str(value or default)

    def _should_invert_ps3_normal_green(self, material, image=None) -> bool:
        try:
            material_platform = _get_material_platform_name(material)
        except Exception:
            material_platform = ''
        image_platform = self._image_custom_string(image, 'trlau_texture_platform', '').lower()
        return material_platform in {'ps3', 'xbox360'} or image_platform in {'ps3', 'xbox360'}

    def _image_is_probably_grayscale(self, image, *, allow_pixel_sampling: bool = True) -> bool:
        if not allow_pixel_sampling:
            return bool(self._image_custom_bool(image, 'trlau_ps3_treat_as_height_map', False))

        samples = self._sample_image_pixels(image, max_samples=1024)
        if not samples:
            return False
        try:
            channel_delta = 0.0
            blue_bias = 0.0
            for r, g, b, _a in samples:
                channel_delta += (abs(r - g) + abs(g - b) + abs(r - b)) / 3.0
                blue_bias += b - ((r + g) * 0.5)
            channel_delta /= float(len(samples))
            blue_bias /= float(len(samples))
            # Tangent normal maps are usually strongly blue/purple.  Several
            # PS3 character maps are grayscale bump/height maps instead.
            return channel_delta < 0.055 and blue_bias < 0.08
        except Exception:
            return False

    def _image_has_meaningful_alpha(self, image, *, allow_pixel_sampling: bool = True) -> bool:
        # Prefer loader metadata for PS3 DDS textures.  This avoids forcing a
        # full DDS pixel decode during model import.
        alpha_hint = self._image_custom_bool(image, 'trlau_ps3_texture_has_alpha', None)
        if alpha_hint is not None:
            return bool(alpha_hint)
        if not allow_pixel_sampling:
            return False

        samples = self._sample_image_pixels(image, max_samples=1024)
        if not samples:
            return False
        try:
            alphas = [float(sample[3]) for sample in samples]
            min_alpha = min(alphas)
            max_alpha = max(alphas)
            mean_alpha = sum(alphas) / float(len(alphas))
            return min_alpha < 0.985 and (max_alpha - min_alpha > 0.025 or mean_alpha < 0.985)
        except Exception:
            return False

    def _create_ps3_normal_nodes(self, material, nodes, links, bsdf, normal_image, diffuse_tex=None) -> None:
        normal_input = bsdf.inputs.get('Normal') if bsdf is not None else None
        if normal_image is None or normal_input is None:
            return

        self._set_image_non_color(normal_image)
        normal_tex = self._create_texture_node(nodes, normal_image, location=(-760, -360))
        self._try_set_node_label(normal_tex, 'TRLAU_PS3_NormalMap', 'PS3 Normal')
        self._link_matching_texture_vector(links, diffuse_tex, normal_tex)
        if normal_tex is None:
            return

        invert_green = self._should_invert_ps3_normal_green(material, normal_image)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-210, -360)
        try:
            normal_map.space = 'TANGENT'
        except Exception:
            pass
        try:
            normal_color = normal_tex.outputs['Color']
            if invert_green:
                normal_color = self._create_inverted_green_normal_color(nodes, links, normal_color, location=(-520, -360))
            links.new(normal_color, normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], normal_input)
        except Exception:
            pass

    def _create_ps3_specular_nodes(self, material, nodes, links, bsdf, specular_image, diffuse_tex=None) -> None:
        if bsdf is None:
            return
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_image is None:
            if specular_input is not None:
                try:
                    specular_input.default_value = 0.0
                except Exception:
                    pass
            return

        self._set_image_non_color(specular_image)
        specular_tex = self._create_texture_node(nodes, specular_image, location=(-760, -650))
        self._try_set_node_label(specular_tex, 'TRLAU_PS3_SpecularGlossMap', 'PS3 Specular / Gloss')
        self._link_matching_texture_vector(links, diffuse_tex, specular_tex)
        if specular_tex is None:
            return

        use_alpha = self._image_has_meaningful_alpha(specular_image, allow_pixel_sampling=False)
        if use_alpha:
            mask_output = specular_tex.outputs.get('Alpha')
        else:
            rgb_to_bw = nodes.new(type='ShaderNodeRGBToBW')
            rgb_to_bw.location = (-500, -650)
            mask_output = rgb_to_bw.outputs.get('Val')
            try:
                links.new(specular_tex.outputs['Color'], rgb_to_bw.inputs['Color'])
            except Exception:
                mask_output = None

        if mask_output is None or specular_input is None:
            return

        try:
            links.new(mask_output, specular_input)
        except Exception:
            pass

    def _setup_ps3_material_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
        reflective: bool = False,
        reflection_mode: str = 'env',
        normal_image: Optional[bpy.types.Image] = None,
        specular_image: Optional[bpy.types.Image] = None,
        ps2_stage_overlay_image: Optional[bpy.types.Image] = None,
    ) -> None:
        nodes, links, output = self._prepare_material_node_tree(material, tpage_flags)
        is_invisible = self._apply_culling_mode_settings(material, tpage_flags.get('cull_mode', 0))
        if is_invisible:
            self._create_invisible_shader(nodes, links, output)
            return

        material_platform = _get_material_platform_name(material)
        use_vertex_color_alpha = material_platform != 'ps3'

        vertex_color = self._create_vertex_color_node(nodes, attribute_name=vertex_color_attribute_name) if use_vertex_colors else None
        diffuse_tex = self._create_scrolling_texture_node(material, nodes, image, location=(-760, 40))
        self._try_set_node_label(diffuse_tex, 'TRLAU_PS3_DiffuseMap', 'PS3 Diffuse')

        color_output = diffuse_tex.outputs.get('Color') if diffuse_tex is not None else None
        alpha_output = diffuse_tex.outputs.get('Alpha') if diffuse_tex is not None else None
        vertex_alpha_output = None
        blend_value = int(tpage_flags.get('blend_value', 0))
        if vertex_color is not None:
            try:
                vertex_color.label = 'PS3 Vertex Color'
                vertex_color.name = 'TRLAU_PS3_VertexColor'
            except Exception:
                pass
            color_output = self._create_multiply_rgb_node(
                nodes,
                links,
                color_output,
                vertex_color.outputs.get('Color'),
                location=(-450, 20),
            )
            vertex_alpha_output = self._vertex_alpha_socket(vertex_color) if use_vertex_color_alpha else None
            if vertex_alpha_output is not None and blend_value == 1:
                # Xbox 360 vertex colors can still contribute alpha on Blend 1.
                # PS3 keeps the secondary-stream alpha as imported color data
                # only; routing it into material transparency makes the preview
                # visibly wrong on the supplied PS3 samples.
                alpha_output = self._create_multiply_value_node(
                    nodes,
                    links,
                    alpha_output,
                    vertex_alpha_output,
                    default_second=None,
                    location=(-450, -180),
                )
        elif image is None and int(tpage_flags.get('texture_id', 0)) != 0:
            color_output, alpha_output = self._apply_missing_texture_wireframe(nodes, links, vertex_color)

        clipped_alpha_output = None
        if (vertex_alpha_output is None or blend_value != 1) and blend_value == 0 and alpha_output is not None:
            clipped_alpha_output = self._create_alpha_greater_than_node(nodes, links, alpha_output, location=(-180, -180))

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (90, 0)
        self._set_socket_default(bsdf, 'Metallic', 0.0)
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            try:
                specular_input.default_value = 0.25
            except Exception:
                pass

        if color_output is not None:
            try:
                links.new(color_output, bsdf.inputs['Base Color'])
            except Exception:
                pass
        else:
            self._set_socket_default(bsdf, 'Base Color', (1.0, 1.0, 1.0, 1.0))

        alpha_input = bsdf.inputs.get('Alpha')
        if blend_value == 1 and vertex_alpha_output is not None and alpha_output is not None and alpha_input is not None:
            try:
                links.new(alpha_output, alpha_input)
                material.blend_method = 'BLEND'
                material.shadow_method = 'HASHED'
            except Exception:
                pass
        elif blend_value in {1, 8} and alpha_output is not None and alpha_input is not None:
            try:
                links.new(alpha_output, alpha_input)
                material.blend_method = 'BLEND'
                material.shadow_method = 'HASHED'
            except Exception:
                pass
        elif clipped_alpha_output is not None and alpha_input is not None:
            try:
                links.new(clipped_alpha_output, alpha_input)
                material.blend_method = 'CLIP'
            except Exception:
                pass

        self._create_ps3_normal_nodes(material, nodes, links, bsdf, normal_image, diffuse_tex=diffuse_tex)
        self._create_ps3_specular_nodes(material, nodes, links, bsdf, specular_image, diffuse_tex=diffuse_tex)

        shader_output = bsdf.outputs.get('BSDF')
        if reflective and shader_output is not None and image is not None:
            shader_output = self._apply_reflection_overlay(nodes, links, shader_output, image, reflection_mode)

        if shader_output is not None:
            try:
                links.new(shader_output, output.inputs['Surface'])
            except Exception:
                pass

    @staticmethod
    def _vertex_alpha_socket(vertex_color_node):
        if vertex_color_node is None:
            return None
        return vertex_color_node.outputs.get('Alpha') or vertex_color_node.outputs.get('Fac')

    def _create_additive_alpha_shader(self, nodes, links, color_output, alpha_output, location=(100, 0)):
        emission = nodes.new(type='ShaderNodeEmission')
        emission.location = location
        strength_input = emission.inputs.get('Strength')
        if strength_input is not None:
            strength_input.default_value = 2.0

        emission_color_output = color_output
        if color_output is not None and alpha_output is not None:
            scale_color = nodes.new(type='ShaderNodeMixRGB')
            scale_color.blend_type = 'MULTIPLY'
            scale_color.inputs['Fac'].default_value = 1.0
            scale_color.location = (location[0] - 270, location[1] - 30)
            links.new(color_output, scale_color.inputs['Color1'])
            links.new(alpha_output, scale_color.inputs['Color2'])
            emission_color_output = scale_color.outputs['Color']

        if emission_color_output is not None:
            links.new(emission_color_output, emission.inputs['Color'])
        else:
            try:
                emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
            except Exception:
                pass

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (location[0], location[1] - 200)

        add_shader = nodes.new(type='ShaderNodeAddShader')
        add_shader.location = (location[0] + 220, location[1] - 20)
        links.new(transparent.outputs['BSDF'], add_shader.inputs[0])
        links.new(emission.outputs['Emission'], add_shader.inputs[1])
        return add_shader.outputs['Shader']

    def _create_vertex_color_node(self, nodes, location=(-700, 0), force_attribute: bool = False, attribute_name: str | None = None):
        resolved_attribute_name = str(attribute_name or 'Color')
        if force_attribute:
            try:
                vertex_color = nodes.new(type='ShaderNodeAttribute')
                vertex_color.attribute_name = resolved_attribute_name
                vertex_color.location = location
                return vertex_color
            except Exception as exc:
                logger.warning('Failed to create attribute shader node: %s', exc)
                return None

        try:
            vertex_color = nodes.new(type='ShaderNodeVertexColor')
            vertex_color.layer_name = resolved_attribute_name
        except Exception:
            try:
                vertex_color = nodes.new(type='ShaderNodeAttribute')
                vertex_color.attribute_name = resolved_attribute_name
            except Exception as exc:
                logger.warning('Failed to create vertex color shader node: %s', exc)
                return None
        vertex_color.location = location
        return vertex_color

    @staticmethod
    def _is_uv_scroll_material(material) -> bool:
        try:
            return bool(getattr(material, 'trlau_ui_scroll_enabled', False))
        except Exception:
            return False

    @staticmethod
    def _clear_driver_variables(driver) -> None:
        try:
            while driver.variables:
                driver.variables.remove(driver.variables[0])
        except Exception:
            pass

    def _configure_uv_scroll_mapping_drivers(self, material, mapping_node) -> None:
        location_input = mapping_node.inputs.get('Location')
        if location_input is None:
            return

        scene = getattr(bpy.context, 'scene', None)
        axis_configs = (
            (0, 0.0),
            (1, 1.0),
        )
        for axis, axis_enabled in axis_configs:
            try:
                mapping_node.inputs['Location'].driver_remove('default_value', axis)
            except Exception:
                pass
            try:
                fcurve = mapping_node.inputs['Location'].driver_add('default_value', axis)
            except Exception:
                continue
            driver = fcurve.driver
            driver.type = 'SCRIPTED'
            self._clear_driver_variables(driver)

            for var_name, id_type, target_id, data_path in (
                ('speed', 'MATERIAL', material, 'trlau_ui_scroll_speed'),
                ('enabled', 'MATERIAL', material, 'trlau_ui_scroll_enabled'),
            ):
                var = driver.variables.new()
                var.name = var_name
                target = var.targets[0]
                target.id_type = id_type
                target.id = target_id
                target.data_path = data_path

            if scene is not None:
                for var_name, data_path in (
                    ('fps', 'render.fps'),
                    ('fps_base', 'render.fps_base'),
                ):
                    var = driver.variables.new()
                    var.name = var_name
                    target = var.targets[0]
                    target.id_type = 'SCENE'
                    target.id = scene
                    target.data_path = data_path

                driver.expression = f'enabled and ({axis_enabled}) and (((frame * fps_base / fps) * speed / 2048.0 * 60.0) % 1.0) or 0.0'
            else:
                driver.expression = f'enabled and ({axis_enabled}) and (((frame / 60.0) * speed / 2048.0 * 60.0) % 1.0) or 0.0'

        try:
            mapping_node.inputs['Location'].default_value[2] = 0.0
        except Exception:
            pass

    def _ensure_uv_scroll_nodes(self, material, tex_image):
        node_tree = getattr(material, 'node_tree', None)
        if node_tree is None or tex_image is None:
            return None
        nodes = node_tree.nodes
        links = node_tree.links

        mapping = None
        tex_coord = None
        for node in nodes:
            try:
                role = node.get('trlau_uv_scroll_role')
            except Exception:
                role = None
            if node.bl_idname == 'ShaderNodeMapping' and (role == 'mapping' or node.name == 'TRLAU_UVScroll_Mapping'):
                mapping = node
                try:
                    node['trlau_uv_scroll_role'] = 'mapping'
                    node.label = ''
                except Exception:
                    pass
            elif node.bl_idname == 'ShaderNodeTexCoord' and (role == 'texcoord' or node.name == 'TRLAU_UVScroll_TexCoord'):
                tex_coord = node
                try:
                    node['trlau_uv_scroll_role'] = 'texcoord'
                    node.label = ''
                except Exception:
                    pass

        if tex_coord is None:
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            try:
                tex_coord['trlau_uv_scroll_role'] = 'texcoord'
            except Exception:
                pass
        tex_coord.location = (tex_image.location[0] - 520, tex_image.location[1] - 40)

        if mapping is None:
            mapping = nodes.new(type='ShaderNodeMapping')
            try:
                mapping['trlau_uv_scroll_role'] = 'mapping'
            except Exception:
                pass
        mapping.location = (tex_image.location[0] - 260, tex_image.location[1] - 40)
        try:
            mapping.vector_type = 'TEXTURE'
        except Exception:
            pass
        self._set_vector_socket_default(mapping, 'Location', (0.0, 0.0, 0.0))

        upstream_socket = None
        vector_input = tex_image.inputs.get('Vector')
        mapping_vector_input = mapping.inputs.get('Vector')
        mapping_vector_output = mapping.outputs.get('Vector')
        tex_coord_uv_output = tex_coord.outputs.get('UV')

        try:
            if vector_input is not None and vector_input.is_linked:
                existing_link = vector_input.links[0]
                existing_from_node = getattr(existing_link.from_socket, 'node', None)
                if existing_from_node == mapping and mapping_vector_input is not None and mapping_vector_input.is_linked:
                    upstream_socket = mapping_vector_input.links[0].from_socket
                else:
                    upstream_socket = existing_link.from_socket
        except Exception:
            upstream_socket = None

        if getattr(upstream_socket, 'node', None) == mapping:
            upstream_socket = None

        for socket in (mapping_vector_input, vector_input):
            if socket is None:
                continue
            try:
                while socket.is_linked:
                    links.remove(socket.links[0])
            except Exception:
                pass

        source_socket = upstream_socket
        if source_socket is None or getattr(source_socket, 'node', None) == tex_coord:
            source_socket = tex_coord_uv_output

        if source_socket is not None and mapping_vector_input is not None:
            try:
                links.new(source_socket, mapping_vector_input)
            except Exception:
                pass
        if mapping_vector_output is not None and vector_input is not None:
            try:
                links.new(mapping_vector_output, vector_input)
            except Exception:
                pass

        self._configure_uv_scroll_mapping_drivers(material, mapping)
        return mapping

    def _apply_uv_scroll_animation_to_material(self, material) -> None:
        if not self._is_uv_scroll_material(material):
            return
        node_tree = getattr(material, 'node_tree', None)
        if node_tree is None:
            return

        tex_image = None
        for node in node_tree.nodes:
            if node.bl_idname == 'ShaderNodeTexImage' and getattr(node, 'image', None) is not None:
                tex_image = node
                break
        if tex_image is not None:
            self._ensure_uv_scroll_nodes(material, tex_image)

    def _create_reflection_texture_nodes(
        self,
        nodes,
        links,
        image: bpy.types.Image,
        *,
        reflection_mode: str = 'env',
        location=(-900, -320),
    ):
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_coord.location = (location[0], location[1])

        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.location = (location[0] + 220, location[1])
        mapping.vector_type = 'TEXTURE'
        self._set_vector_socket_default(mapping, 'Location', (-2.0, 0.0, -2.0))
        self._set_vector_socket_default(mapping, 'Rotation', (0.0, 0.0, 0.0))
        scale_value = 4.0 if reflection_mode != 'eye_ref' else 2.0
        self._set_vector_socket_default(mapping, 'Scale', (scale_value, scale_value, scale_value))
        links.new(tex_coord.outputs['Reflection'], mapping.inputs['Vector'])

        reflection_image = self._create_texture_node(nodes, image, location=(location[0] + 460, location[1]))
        reflection_image.projection = 'BOX'
        try:
            reflection_image.projection_blend = 1.0
        except Exception:
            pass
        try:
            reflection_image.extension = 'EXTEND'
        except Exception:
            pass
        links.new(mapping.outputs['Vector'], reflection_image.inputs['Vector'])
        return reflection_image

    def _setup_reflective_material_nodes_exact(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        reflection_mode: str = 'env',
    ) -> None:
        nodes, links, output = self._prepare_material_node_tree(
            material,
            tpage_flags,
            blend_method='BLEND',
            output_location=(300, -20),
        )

        tex_image = self._create_reflection_texture_nodes(
            nodes,
            links,
            image,
            reflection_mode=reflection_mode,
            location=(-1250, -20),
        )
        attribute = self._create_vertex_color_node(nodes, location=(-680, -330), force_attribute=True)

        color_multiply = nodes.new(type='ShaderNodeMixRGB')
        color_multiply.blend_type = 'MULTIPLY'
        color_multiply.inputs['Fac'].default_value = 1.0
        color_multiply.location = (-340, 40)

        alpha_multiply = nodes.new(type='ShaderNodeMixRGB')
        alpha_multiply.blend_type = 'MULTIPLY'
        alpha_multiply.inputs['Fac'].default_value = 1.0
        alpha_multiply.location = (-340, -220)

        links.new(tex_image.outputs['Color'], color_multiply.inputs['Color1'])
        if attribute is not None:
            links.new(attribute.outputs['Color'], color_multiply.inputs['Color2'])

        alpha_socket = self._vertex_alpha_socket(attribute)
        links.new(tex_image.outputs['Color'], alpha_multiply.inputs['Color1'])
        if alpha_socket is not None:
            links.new(alpha_socket, alpha_multiply.inputs['Color2'])

        if tpage_flags['blend_value'] == 2:
            shader_output = self._create_additive_alpha_shader(
                nodes,
                links,
                color_multiply.outputs['Color'],
                alpha_socket or tex_image.outputs.get('Alpha'),
                location=(10, -40),
            )
            links.new(shader_output, output.inputs['Surface'])
            return

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (10, -40)
        self._set_socket_default(bsdf, 'Roughness', 0.5)
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            specular_input.default_value = 0.0

        links.new(color_multiply.outputs['Color'], bsdf.inputs['Base Color'])
        if 'Alpha' in bsdf.inputs:
            links.new(alpha_multiply.outputs['Color'], bsdf.inputs['Alpha'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    @staticmethod
    def _create_invisible_shader(nodes, links, output):
        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (100, 0)
        links.new(transparent.outputs['BSDF'], output.inputs['Surface'])
        return transparent

    def _create_blend_preview_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        *,
        texture_location: Tuple[int, int],
        vertex_location: Tuple[int, int],
        output_location: Tuple[int, int] = (520, 0),
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ):
        nodes, links, output = self._prepare_material_node_tree(
            material,
            tpage_flags,
            blend_method='BLEND',
            output_location=output_location,
            transparent_back=not bool(tpage_flags['single_sided']),
            shadow_method='NONE',
            screen_refraction=False,
        )
        tex_image = self._create_scrolling_texture_node(material, nodes, image, location=texture_location)
        vertex_color = None
        if use_vertex_colors or vertex_color_attribute_name:
            vertex_color = self._create_vertex_color_node(
                nodes,
                location=vertex_location,
                attribute_name=vertex_color_attribute_name,
            )
        return nodes, links, output, tex_image, vertex_color

    def _setup_blend4_material_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ) -> None:
        nodes, links, output, tex_image, vertex_color = self._create_blend_preview_nodes(
            material,
            image,
            tpage_flags,
            texture_location=(-650, 80),
            vertex_location=(-650, -160),
            use_vertex_colors=use_vertex_colors,
            vertex_color_attribute_name=vertex_color_attribute_name,
        )

        color_output = tex_image.outputs.get('Color') if tex_image is not None else None
        alpha_output = tex_image.outputs.get('Alpha') if tex_image is not None else None
        if vertex_color is not None:
            color_output = self._create_multiply_rgb_node(
                nodes,
                links,
                color_output,
                vertex_color.outputs.get('Color'),
                location=(-360, 40),
            )
            alpha_output = self._create_multiply_value_node(
                nodes,
                links,
                alpha_output,
                self._vertex_alpha_socket(vertex_color),
                default_second=None,
                location=(-360, -160),
            )

        if color_output is None:
            self._create_invisible_shader(nodes, links, output)
            return

        shader_output = self._create_additive_alpha_shader(nodes, links, color_output, alpha_output, location=(40, 0))
        links.new(shader_output, output.inputs['Surface'])

    def _setup_blend9_material_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ) -> None:
        nodes, links, output, tex_image, vertex_color = self._create_blend_preview_nodes(
            material,
            image,
            tpage_flags,
            texture_location=(-620, 60),
            vertex_location=(-620, -160),
            use_vertex_colors=use_vertex_colors,
            vertex_color_attribute_name=vertex_color_attribute_name,
        )

        mask_output = self._create_multiply_value_node(
            nodes,
            links,
            tex_image.outputs.get('Alpha') if tex_image is not None else None,
            self._vertex_alpha_socket(vertex_color),
            default_second=None,
            location=(-360, -40),
        )
        if mask_output is not None:
            mask_output = self._create_multiply_value_node(
                nodes,
                links,
                mask_output,
                default_second=0.1,
                location=(-100, -40),
            )

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (40, 120)

        try:
            specular_shader = nodes.new(type='ShaderNodeBsdfGlossy')
        except Exception:
            specular_shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        specular_shader.location = (40, -80)
        color_input = specular_shader.inputs.get('Color') or specular_shader.inputs.get('Base Color')
        if color_input is not None:
            try:
                color_input.default_value = (1.0, 1.0, 1.0, 1.0)
            except Exception:
                pass
        self._set_socket_default(specular_shader, 'Metallic', 0.0)
        self._set_socket_default(specular_shader, 'Roughness', 0.1)
        specular_input = specular_shader.inputs.get('Specular IOR Level') or specular_shader.inputs.get('Specular')
        if specular_input is not None:
            specular_input.default_value = 1.0

        mix_shader = nodes.new(type='ShaderNodeMixShader')
        mix_shader.location = (300, 20)
        if mask_output is not None:
            links.new(mask_output, mix_shader.inputs['Fac'])
        else:
            mix_shader.inputs['Fac'].default_value = 0.0
        links.new(transparent.outputs['BSDF'], mix_shader.inputs[1])
        links.new(specular_shader.outputs['BSDF'], mix_shader.inputs[2])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    def _base_material_color_alpha_outputs(
        self,
        material,
        nodes,
        links,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        vertex_color,
    ):
        color_output = None
        alpha_output = None
        tex_image = self._create_scrolling_texture_node(material, nodes, image, location=(-450, 0))

        if tex_image is not None:
            color_output = tex_image.outputs.get('Color')
            alpha_output = tex_image.outputs.get('Alpha')

            if vertex_color is not None:
                color_output = self._create_multiply_rgb_node(
                    nodes,
                    links,
                    vertex_color.outputs.get('Color'),
                    tex_image.outputs.get('Color'),
                    location=(-150, 0),
                )
                vertex_alpha_output = self._vertex_alpha_socket(vertex_color)
                texture_alpha_output = tex_image.outputs.get('Alpha')
                if tpage_flags['blend_value'] == 8:
                    alpha_output = self._create_multiply_value_node(
                        nodes,
                        links,
                        tex_image.outputs.get('Color'),
                        vertex_alpha_output,
                        location=(-150, -190),
                    )
                elif vertex_alpha_output is not None:
                    alpha_output = self._create_multiply_value_node(
                        nodes,
                        links,
                        texture_alpha_output,
                        vertex_alpha_output,
                        default_second=None,
                        location=(-150, -190),
                    )
        elif vertex_color is not None:
            color_output = vertex_color.outputs.get('Color')
            alpha_output = self._vertex_alpha_socket(vertex_color)

        return color_output, alpha_output

    def _apply_missing_texture_wireframe(self, nodes, links, vertex_color):
        wireframe = nodes.new(type='ShaderNodeWireframe')
        wireframe.location = (-450, -40)
        if hasattr(wireframe, 'use_pixel_size'):
            wireframe.use_pixel_size = True
        wireframe.inputs['Size'].default_value = 1.0

        if vertex_color is not None:
            color_output = self._create_multiply_rgb_node(
                nodes,
                links,
                wireframe.outputs.get('Fac'),
                vertex_color.outputs.get('Color'),
                location=(-150, 0),
            )
        else:
            color_output = wireframe.outputs.get('Fac')
        return color_output, wireframe.outputs.get('Fac')

    def _create_flat_shader(self, nodes, links, color_output, alpha_output, clipped_alpha_output, tpage_flags, is_invisible: bool):
        shader = nodes.new(type='ShaderNodeEmission')
        shader.location = (100, 0)
        self._set_socket_default(shader, 'Strength', 3.0)
        if color_output is not None:
            links.new(color_output, shader.inputs['Color'])
        else:
            self._set_socket_default(shader, 'Color', (1.0, 1.0, 1.0, 1.0))

        should_mix_transparent = (tpage_flags['blend_value'] == 1 and alpha_output is not None) or clipped_alpha_output is not None or is_invisible
        if not should_mix_transparent:
            return shader.outputs['Emission'], None

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        mix_shader = nodes.new(type='ShaderNodeMixShader')
        transparent.location = (100, -180)
        mix_shader.location = (280, 0)
        fac_output = alpha_output if tpage_flags['blend_value'] == 1 else clipped_alpha_output
        if is_invisible:
            fac_output = None
        if fac_output is not None:
            links.new(fac_output, mix_shader.inputs['Fac'])
        else:
            mix_shader.inputs['Fac'].default_value = 1.0
        links.new(transparent.outputs['BSDF'], mix_shader.inputs[1])
        links.new(shader.outputs['Emission'], mix_shader.inputs[2])
        blend_method = 'BLEND' if tpage_flags['blend_value'] == 1 else 'CLIP'
        return mix_shader.outputs['Shader'], blend_method

    def _create_principled_shader(self, nodes, links, color_output, alpha_output, clipped_alpha_output, tpage_flags, is_invisible: bool, normal_image=None, specular_image=None):
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (100, 0)
        if color_output is not None:
            links.new(color_output, bsdf.inputs['Base Color'])
        else:
            self._set_socket_default(bsdf, 'Base Color', (1.0, 1.0, 1.0, 1.0))
        self._set_socket_default(bsdf, 'Roughness', 0.45)
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            specular_input.default_value = 0.0

        self._attach_auxiliary_pbr_maps(nodes, links, bsdf, normal_image=normal_image, specular_image=specular_image)

        blend_method = None
        alpha_input = bsdf.inputs.get('Alpha')
        if tpage_flags['blend_value'] in {1, 8} and alpha_output is not None and alpha_input is not None:
            links.new(alpha_output, alpha_input)
            blend_method = 'BLEND'
        elif clipped_alpha_output is not None and alpha_input is not None:
            links.new(clipped_alpha_output, alpha_input)
            blend_method = 'CLIP'
        elif is_invisible and alpha_input is not None:
            alpha_input.default_value = 0.0

        return bsdf.outputs['BSDF'], blend_method

    def _apply_reflection_overlay(self, nodes, links, shader_output, image: bpy.types.Image, reflection_mode: str):
        reflection_image = self._create_reflection_texture_nodes(nodes, links, image, reflection_mode=reflection_mode)
        reflection_bsdf = nodes.new(type='ShaderNodeBsdfGlossy')
        reflection_bsdf.location = (250, -280)
        reflection_bsdf.inputs['Roughness'].default_value = 0.02
        try:
            links.new(reflection_image.outputs['Color'], reflection_bsdf.inputs['Color'])
        except Exception:
            pass

        layer_weight = nodes.new(type='ShaderNodeLayerWeight')
        layer_weight.location = (250, -470)
        layer_weight.inputs['Blend'].default_value = 0.15

        reflection_mix = nodes.new(type='ShaderNodeMixShader')
        reflection_mix.location = (580, 0)
        links.new(layer_weight.outputs['Facing'], reflection_mix.inputs['Fac'])
        links.new(reflection_bsdf.outputs['BSDF'], reflection_mix.inputs[1])
        links.new(shader_output, reflection_mix.inputs[2])
        return reflection_mix.outputs['Shader']

    def _setup_tr8_ps2_stage_overlay_nodes(
        self,
        material,
        base_image: Optional[bpy.types.Image],
        overlay_image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ) -> None:
        nodes, links, output = self._prepare_material_node_tree(material, tpage_flags)
        is_invisible = self._apply_culling_mode_settings(material, tpage_flags.get('cull_mode', 0))
        if is_invisible:
            self._create_invisible_shader(nodes, links, output)
            return

        vertex_color = self._create_vertex_color_node(nodes, attribute_name=vertex_color_attribute_name) if use_vertex_colors else None
        base_tex = self._create_scrolling_texture_node(material, nodes, base_image, location=(-760, 80))
        overlay_tex = self._create_texture_node(nodes, overlay_image, location=(-760, -170))

        for tex in (base_tex, overlay_tex):
            try:
                tex.extension = 'EXTEND'
            except Exception:
                pass

        base_color = base_tex.outputs.get('Color') if base_tex is not None else None
        base_alpha = base_tex.outputs.get('Alpha') if base_tex is not None else None
        overlay_color = overlay_tex.outputs.get('Color') if overlay_tex is not None else None
        overlay_alpha = overlay_tex.outputs.get('Alpha') if overlay_tex is not None else None

        color_output = base_color or overlay_color
        alpha_output = base_alpha or overlay_alpha
        if base_color is not None and overlay_color is not None:
            mix = nodes.new(type='ShaderNodeMixRGB')
            mix.blend_type = 'MIX'
            mix.location = (-430, 20)
            try:
                links.new(overlay_alpha, mix.inputs['Fac'])
            except Exception:
                mix.inputs['Fac'].default_value = 1.0
            links.new(base_color, mix.inputs['Color1'])
            links.new(overlay_color, mix.inputs['Color2'])
            color_output = mix.outputs['Color']
            alpha_output = base_alpha

        if vertex_color is not None and color_output is not None:
            color_output = self._create_multiply_rgb_node(
                nodes,
                links,
                vertex_color.outputs.get('Color'),
                color_output,
                location=(-170, 20),
            )

        clipped_alpha_output = None
        if int(tpage_flags.get('blend_value', 0)) == 0 and alpha_output is not None:
            clipped_alpha_output = self._create_alpha_greater_than_node(nodes, links, alpha_output, location=(-170, -190))

        shader_output, blend_method = self._create_principled_shader(
            nodes,
            links,
            color_output,
            alpha_output,
            clipped_alpha_output,
            tpage_flags,
            is_invisible,
        )
        if blend_method is not None:
            material.blend_method = blend_method
        if shader_output is not None:
            links.new(shader_output, output.inputs['Surface'])

    @staticmethod
    def _pc_nextgen_blend_mode_name(blend_mode: int) -> str:
        names = {
            0: 'Opaque',
            1: 'AlphaTest',
            2: 'AlphaBlend',
            3: 'Additive',
            4: 'Subtract',
            5: 'DestAlpha',
            6: 'DestAdd',
            7: 'Modulate',
            8: 'Blend5050',
            9: 'DestAlphaSrcOnly',
            10: 'ColorModulate',
            13: 'MultipassAlpha',
            20: 'LightPassAdditive',
        }
        return names.get(int(blend_mode), f'Blend{int(blend_mode)}')

    @staticmethod
    def _pc_nextgen_combiner_name(combiner_type: int) -> str:
        names = {
            0: 'Default',
            1: 'Lightmap',
            2: 'Reflection',
            3: 'MaskedReflection',
            4: 'StencilReflection',
            5: 'Diffuse',
            6: 'MaskedDiffuse',
            7: 'ImmediateDraw',
            8: 'ImmediateDrawPredator',
            9: 'DepthOfField',
        }
        return names.get(int(combiner_type), f'Combiner{int(combiner_type)}')

    @staticmethod
    def _pc_nextgen_json_float_tuples(value, fallback=None):
        if fallback is None:
            fallback = []
        try:
            parsed = json.loads(str(value or '[]'))
            result = []
            for item in parsed:
                result.append(tuple(float(component) for component in item))
            return result
        except Exception:
            return list(fallback)

    @staticmethod
    def _pc_nextgen_csv_ints(value) -> list[int]:
        try:
            return [int(part.strip()) for part in str(value or '').split(',') if part.strip()]
        except Exception:
            return []

    @staticmethod
    def _create_value_node(nodes, value: float, location=(-450, -260)):
        node = nodes.new(type='ShaderNodeValue')
        node.location = location
        try:
            node.outputs['Value'].default_value = float(value)
        except Exception:
            pass
        return node

    @staticmethod
    def _create_rgb_node(nodes, color, location=(-450, 220)):
        node = nodes.new(type='ShaderNodeRGB')
        node.location = location
        try:
            rgba = tuple(float(color[index]) for index in range(min(4, len(color))))
            if len(rgba) == 3:
                rgba = rgba + (1.0,)
            node.outputs['Color'].default_value = rgba
        except Exception:
            try:
                node.outputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
            except Exception:
                pass
        return node

    @staticmethod
    def _pc_nextgen_color_is_not_white(color) -> bool:
        try:
            return any(abs(float(color[index]) - 1.0) > 0.003 for index in range(3))
        except Exception:
            return False

    def _pc_nextgen_tinted_color_output(self, nodes, links, color_output, color, location=(-220, 60)):
        if color_output is None or not self._pc_nextgen_color_is_not_white(color):
            return color_output
        color_node = self._create_rgb_node(nodes, color, location=(location[0] - 230, location[1] + 120))
        return self._create_multiply_rgb_node(nodes, links, color_output, color_node.outputs.get('Color'), location=location)

    def _pc_nextgen_alpha_output(self, nodes, links, alpha_output, *, opacity: float, layer_alpha: float, blend_mode: int):
        needs_constant_alpha = int(blend_mode) in {2, 5, 8, 9, 13} or float(opacity) < 0.999 or float(layer_alpha) < 0.999
        alpha = alpha_output
        factor = max(0.0, min(1.0, float(opacity) * float(layer_alpha)))
        if int(blend_mode) == 8:
            factor *= 0.5
        if alpha is None and needs_constant_alpha:
            alpha = self._create_value_node(nodes, factor, location=(-450, -250)).outputs.get('Value')
        elif alpha is not None and abs(factor - 1.0) > 0.003:
            alpha = self._create_multiply_value_node(nodes, links, alpha, default_second=factor, location=(-220, -250))
        return alpha

    def _pc_nextgen_alpha_test_output(self, nodes, links, alpha_output, *, threshold: float = 0.5, location=(-10, -250)):
        if alpha_output is None:
            return None
        greater_than = nodes.new(type='ShaderNodeMath')
        greater_than.operation = 'GREATER_THAN'
        greater_than.location = location
        try:
            greater_than.inputs[1].default_value = float(threshold)
        except Exception:
            pass
        links.new(alpha_output, greater_than.inputs[0])
        return greater_than.outputs.get('Value')

    def _pc_nextgen_attach_normal_map(self, nodes, links, bsdf, normal_image, location=(-560, -430)) -> None:
        if bsdf is None or normal_image is None:
            return
        self._set_image_non_color(normal_image)
        normal_tex = self._create_texture_node(nodes, normal_image, location=location)
        if normal_tex is None:
            return
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (location[0] + 410, location[1])
        normal_input = bsdf.inputs.get('Normal')
        if normal_input is None:
            return
        try:
            normal_color = self._create_inverted_green_normal_color(nodes, links, normal_tex.outputs['Color'], location=(location[0] + 180, location[1]))
            links.new(normal_color, normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], normal_input)
        except Exception:
            pass

    def _pc_nextgen_first_enabled_layer_index(self, layer_enabled, layer_images, start_index: int = 0) -> int | None:
        for index in range(max(0, int(start_index)), min(len(layer_enabled), len(layer_images))):
            try:
                enabled = int(layer_enabled[index]) != 0
            except Exception:
                enabled = False
            if enabled and layer_images[index] is not None:
                return int(index)
        return None

    @staticmethod
    def _pc_nextgen_layer_color(layer_colors, index: int, fallback=(1.0, 1.0, 1.0, 1.0)):
        try:
            color = layer_colors[int(index)]
            if len(color) >= 4:
                return (float(color[0]), float(color[1]), float(color[2]), float(color[3]))
            if len(color) >= 3:
                return (float(color[0]), float(color[1]), float(color[2]), 1.0)
        except Exception:
            pass
        return fallback

    def _pc_nextgen_create_layer_texture_node(self, material, nodes, image, layer_index: int, role: str, location=(-760, 80)):
        tex = self._create_scrolling_texture_node(material, nodes, image, location=location)
        if tex is not None:
            try:
                tex.name = f'TRLAU_PCNG_Layer_{int(layer_index)}_{str(role)}'
                tex.label = f'PCNG Layer {int(layer_index)} {str(role).title()}'
            except Exception:
                pass
        return tex

    def _pc_nextgen_tinted_texture_outputs(self, nodes, links, tex_node, tint_color, location=(-430, 80)):
        color_output = tex_node.outputs.get('Color') if tex_node is not None else None
        alpha_output = tex_node.outputs.get('Alpha') if tex_node is not None else None
        if color_output is not None:
            color_output = self._pc_nextgen_tinted_color_output(nodes, links, color_output, tint_color, location=location)
        return color_output, alpha_output

    @staticmethod
    def _pc_nextgen_specular_roughness_from_power(specular_power: float) -> float:
        try:
            value = max(0.0, float(specular_power))
        except Exception:
            value = 0.0
        if value <= 0.0:
            return 0.82
        # D3D-style specular power is a gloss exponent.  Principled uses
        # roughness, so map high powers to sharply lower roughness while keeping
        # values editable and visible in Blender preview.
        return max(0.025, min(0.82, (2.0 / (value + 2.0)) ** 0.5))

    @staticmethod
    def _pc_nextgen_specular_level_from_power(specular_power: float, has_specular_map: bool) -> float:
        try:
            value = max(0.0, float(specular_power))
        except Exception:
            value = 0.0
        if value <= 0.0 and not has_specular_map:
            return 0.0
        # Give the imported exponent an immediate viewport effect even when the
        # specular mask is weak or absent.  The texture still modulates the input
        # when present; this default only establishes the base response.
        return max(0.05, min(1.0, 0.25 + (value / (value + 32.0))))

    def _pc_nextgen_attach_specular_map(
        self,
        nodes,
        links,
        bsdf,
        specular_image,
        *,
        tint_color=(1.0, 1.0, 1.0, 1.0),
        specular_power: float = 0.0,
        vertex_tint_node=None,
        location=(-560, -650),
    ) -> None:
        if bsdf is None or specular_image is None:
            return
        self._set_image_non_color(specular_image)
        specular_tex = self._create_texture_node(nodes, specular_image, location=location)
        if specular_tex is None:
            return

        color_output = specular_tex.outputs.get('Color')
        if color_output is not None:
            color_output = self._pc_nextgen_tinted_color_output(nodes, links, color_output, tint_color, location=(location[0] + 210, location[1] + 60))

        tint_output = None
        if vertex_tint_node is not None:
            try:
                tint_output = vertex_tint_node.outputs.get('Color')
            except Exception:
                tint_output = None
            if tint_output is not None and self._pc_nextgen_color_is_not_white(tint_color):
                tint_color_node = self._create_rgb_node(nodes, tint_color, location=(location[0] + 210, location[1] - 135))
                tint_output = self._create_multiply_rgb_node(
                    nodes,
                    links,
                    tint_output,
                    tint_color_node.outputs.get('Color'),
                    location=(location[0] + 440, location[1] - 135),
                )
            if color_output is not None and tint_output is not None:
                color_output = self._create_multiply_rgb_node(
                    nodes,
                    links,
                    color_output,
                    tint_output,
                    location=(location[0] + 455, location[1] + 70),
                )

        rgb_to_bw = nodes.new(type='ShaderNodeRGBToBW')
        rgb_to_bw.location = (location[0] + 690, location[1] + 40)
        try:
            links.new(color_output or specular_tex.outputs['Color'], rgb_to_bw.inputs['Color'])
        except Exception:
            return

        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            try:
                links.new(rgb_to_bw.outputs['Val'], specular_input)
            except Exception:
                pass

        tint_input = bsdf.inputs.get('Specular Tint')
        if tint_input is not None and tint_output is not None:
            try:
                links.new(tint_output, tint_input)
            except Exception:
                pass

        # The next-gen PC specular layer is effectively a gloss/specular mask.
        # Feed an inverted luminance approximation into Roughness so high gloss
        # areas actually look sharper in Blender instead of staying flat.
        roughness_input = bsdf.inputs.get('Roughness')
        if roughness_input is not None:
            try:
                invert = nodes.new(type='ShaderNodeMath')
                invert.operation = 'SUBTRACT'
                invert.location = (location[0] + 690, location[1] - 60)
                invert.inputs[0].default_value = 1.0
                links.new(rgb_to_bw.outputs['Val'], invert.inputs[1])
                scale = nodes.new(type='ShaderNodeMath')
                scale.operation = 'MULTIPLY'
                scale.location = (location[0] + 900, location[1] - 60)
                base_roughness = self._pc_nextgen_specular_roughness_from_power(float(specular_power))
                scale.inputs[1].default_value = max(0.05, min(0.85, base_roughness + 0.18))
                links.new(invert.outputs['Value'], scale.inputs[0])
                links.new(scale.outputs['Value'], roughness_input)
            except Exception:
                pass

        tint_input = bsdf.inputs.get('Specular Tint')
        if tint_input is not None and tint_output is None:
            try:
                tint_input.default_value = (
                    max(0.0, min(1.0, float(tint_color[0]))),
                    max(0.0, min(1.0, float(tint_color[1]))),
                    max(0.0, min(1.0, float(tint_color[2]))),
                    1.0,
                )
            except Exception:
                pass

    @staticmethod
    def _pc_nextgen_shader_index(material, index: int, default: int = 0) -> int:
        try:
            return int(getattr(material, f'trlau_ui_pcng_shader_index{int(index)}'))
        except Exception:
            pass
        try:
            values = str(material.get('trlau_pc_nextgen_shader_indices', '') or '').split(',')
            if 0 <= int(index) < len(values):
                return int(values[int(index)], 0)
        except Exception:
            pass
        return int(default)

    @staticmethod
    def _pc_nextgen_shader_table_ids(material) -> list[int]:
        raw = ''
        try:
            raw = str(getattr(material, 'trlau_ui_pcng_shader_table_ids', '') or '')
        except Exception:
            raw = ''
        if not raw:
            try:
                raw = str(material.get('trlau_pc_nextgen_shader_table_ids', '') or '')
            except Exception:
                raw = ''

        table: list[int] = []
        for part in str(raw or '').split(','):
            part = part.strip()
            if not part:
                continue
            try:
                table.append(int(part, 0))
            except Exception:
                continue
        return table

    @staticmethod
    def _pc_nextgen_resolved_shader_id(material, shader_slot: int, default_index: int = 0) -> int | None:
        shader_index = MaterialNodeBuilderMixin._pc_nextgen_shader_index(material, shader_slot, default_index)
        shader_table = MaterialNodeBuilderMixin._pc_nextgen_shader_table_ids(material)
        if 0 <= int(shader_index) < len(shader_table):
            return int(shader_table[int(shader_index)])
        return None

    @staticmethod
    def _pc_nextgen_resolved_singlepass_light_ps30_id(material) -> int | None:
        return MaterialNodeBuilderMixin._pc_nextgen_resolved_shader_id(material, 5, 22)

    @staticmethod
    def _pc_nextgen_uses_vertex_color_lighting(material) -> bool:
        # PCMaterialData stores shader *table indices*, not global shader ids.
        # The same no-diffuse-vertex-color SinglePass Light PS 3.0 shader is
        # resource 682, but it appears as index 22 in Lara and as index 5 in
        # several smaller sample tables. Resolve through the imported shader
        # table first; only fall back to the old raw-index test when the table
        # is unavailable.
        resolved_ps30_shader_id = MaterialNodeBuilderMixin._pc_nextgen_resolved_singlepass_light_ps30_id(material)
        if resolved_ps30_shader_id is not None:
            return int(resolved_ps30_shader_id) != 682
        return MaterialNodeBuilderMixin._pc_nextgen_shader_index(material, 5, 22) != 22

    @staticmethod
    def _pc_nextgen_uses_vertex_color_specular_tint(material) -> bool:
        # Shader resource 682 does not multiply vertex RGB into diffuse/base
        # lighting, but Lara samples show the same vertex RGB is still consumed
        # by the specular path.  In material slots 12-15 the authored vertex RGB
        # is black, which suppresses/tints the specular layer without blackening
        # the diffuse texture.
        resolved_ps30_shader_id = MaterialNodeBuilderMixin._pc_nextgen_resolved_singlepass_light_ps30_id(material)
        if resolved_ps30_shader_id is not None:
            return int(resolved_ps30_shader_id) == 682
        return MaterialNodeBuilderMixin._pc_nextgen_shader_index(material, 5, 22) == 22

    def _setup_pc_nextgen_material_nodes(
        self,
        material,
        layer_images: list[Optional[bpy.types.Image]],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
    ) -> None:
        material.use_nodes = True
        blend_mode = int(self._safe_material_get(material, 'trlau_pc_nextgen_blend_mode', 0) or 0)
        combiner_type = int(self._safe_material_get(material, 'trlau_pc_nextgen_combiner_type', 0) or 0)
        material_flags = int(self._safe_material_get(material, 'trlau_pc_nextgen_material_flags', 0) or 0)
        double_sided = bool(self._safe_material_get(material, 'trlau_pc_nextgen_double_sided', False))
        opacity = max(0.0, min(1.0, float(self._safe_material_get(material, 'trlau_pc_nextgen_opacity', 1.0) or 1.0)))
        specular_power = max(0.0, float(self._safe_material_get(material, 'trlau_pc_nextgen_specular_power', 0.0) or 0.0))
        rim_intensity = max(0.0, float(self._safe_material_get(material, 'trlau_pc_nextgen_rim_light_intensity', 0.0) or 0.0))
        layer_enabled = self._pc_nextgen_csv_ints(self._safe_material_get(material, 'trlau_pc_nextgen_layer_enabled', ''))
        layer_colors = self._pc_nextgen_json_float_tuples(self._safe_material_get(material, 'trlau_pc_nextgen_layer_colors', '[]'))
        layer_texcoord_sources = self._pc_nextgen_csv_ints(self._safe_material_get(material, 'trlau_pc_nextgen_layer_texcoord_sources', ''))
        rim_colors = self._pc_nextgen_json_float_tuples(self._safe_material_get(material, 'trlau_pc_nextgen_rim_light_color', '[]'))
        rim_color = rim_colors[0] if rim_colors else (0.0, 0.0, 0.0, 0.0)

        local_tpage_flags = dict(tpage_flags or _decode_tpage_flags(0))
        # PCMaterialData has its own render state.  Keep the old tpage flags as
        # metadata only; do not let them impose legacy culling/alpha behaviour.
        # The authored PC Next-Gen Double Sided flag is the Blender viewport
        # culling source of truth: enabled means backface culling must be off.
        local_tpage_flags['single_sided'] = 0 if double_sided else 1
        local_tpage_flags['cull_mode'] = 0
        blend_method = 'OPAQUE'
        if blend_mode in {2, 3, 4, 5, 6, 8, 9, 13, 20} or opacity < 0.999:
            blend_method = 'BLEND'
        elif blend_mode == 1 or combiner_type == 6:
            blend_method = 'CLIP'

        nodes, links, output = self._prepare_material_node_tree(
            material,
            local_tpage_flags,
            blend_method=blend_method,
            output_location=(980, 0),
            shadow_method='HASHED' if blend_method == 'BLEND' else None,
        )
        # Keep PC next-gen material names on the same Material_<index>_<textureID>
        # scheme used by old-gen imports.  Do not append blend/combiner labels
        # to the Blender material name; those values are editable in the panel.

        # Make the complete layer table visible in the node tree.  Only the
        # documented primary roles are connected to Principled, but extra layers
        # are still created/labeled so the material is inspectable instead of
        # silently collapsing back to a single old-gen diffuse texture.
        layer_nodes = {}
        for layer_index, layer_image in enumerate(layer_images or []):
            if layer_image is None:
                continue
            try:
                enabled = int(layer_enabled[layer_index]) != 0 if layer_index < len(layer_enabled) else True
            except Exception:
                enabled = True
            if not enabled:
                continue
            role = 'Layer'
            if layer_index == 0:
                role = 'BaseColor'
            elif layer_index == 1:
                role = 'Normal'
            elif layer_index == 2:
                role = 'SpecularGloss'
            elif layer_index < len(layer_texcoord_sources) and int(layer_texcoord_sources[layer_index]) in {6, 9}:
                role = 'Reflection'
            layer_nodes[layer_index] = self._pc_nextgen_create_layer_texture_node(
                material,
                nodes,
                layer_image,
                layer_index,
                role,
                location=(-980, 140 - (layer_index * 170)),
            )

        diffuse_layer = self._pc_nextgen_first_enabled_layer_index(layer_enabled, layer_images, 0)
        normal_layer = self._pc_nextgen_first_enabled_layer_index(layer_enabled, layer_images, 1)
        specular_layer = self._pc_nextgen_first_enabled_layer_index(layer_enabled, layer_images, 2)
        if diffuse_layer is None:
            diffuse_layer = 0 if layer_images else None
        if normal_layer == diffuse_layer:
            normal_layer = None
        if specular_layer in {diffuse_layer, normal_layer}:
            specular_layer = None

        diffuse_tex = layer_nodes.get(diffuse_layer) if diffuse_layer is not None else None
        diffuse_color = self._pc_nextgen_layer_color(layer_colors, diffuse_layer or 0)
        color_output, alpha_output = self._pc_nextgen_tinted_texture_outputs(nodes, links, diffuse_tex, diffuse_color, location=(-540, 120))
        layer0_alpha = float(diffuse_color[3]) if len(diffuse_color) > 3 else 1.0

        vertex_color = None
        use_pcng_vertex_diffuse = self._pc_nextgen_uses_vertex_color_lighting(material)
        use_pcng_vertex_specular_tint = self._pc_nextgen_uses_vertex_color_specular_tint(material)
        if (use_vertex_colors or vertex_color_attribute_name) and (use_pcng_vertex_diffuse or use_pcng_vertex_specular_tint):
            vertex_color = self._create_vertex_color_node(
                nodes,
                location=(-540, -90),
                attribute_name=vertex_color_attribute_name,
            )
            if vertex_color is not None:
                try:
                    vertex_color.label = 'PC Next Gen Vertex Color'
                    vertex_color.name = 'TRLAU_PCNG_VertexColor'
                except Exception:
                    pass
                if use_pcng_vertex_diffuse:
                    color_output = self._create_multiply_rgb_node(
                        nodes,
                        links,
                        color_output,
                        vertex_color.outputs.get('Color'),
                        location=(-270, 105),
                    )

        if color_output is None:
            color_output, alpha_output = self._apply_missing_texture_wireframe(nodes, links, None)

        alpha_output = self._pc_nextgen_alpha_output(nodes, links, alpha_output, opacity=opacity, layer_alpha=layer0_alpha, blend_mode=blend_mode)

        normal_image = layer_images[normal_layer] if normal_layer is not None and normal_layer < len(layer_images) else None
        specular_image = layer_images[specular_layer] if specular_layer is not None and specular_layer < len(layer_images) else None
        specular_tint = self._pc_nextgen_layer_color(layer_colors, specular_layer or 2)

        # Additive next-gen passes are closer to emission/add shaders than to an
        # alpha-blended Principled surface.
        if blend_mode in {3, 6, 20}:
            shader_output = self._create_additive_alpha_shader(nodes, links, color_output, alpha_output, location=(380, 0))
            material.blend_method = 'BLEND'
            try:
                material.show_transparent_back = False
            except Exception:
                pass
            links.new(shader_output, output.inputs['Surface'])
            return

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (380, 0)
        if color_output is not None:
            links.new(color_output, bsdf.inputs['Base Color'])
        else:
            self._set_socket_default(bsdf, 'Base Color', (1.0, 1.0, 1.0, 1.0))

        roughness = self._pc_nextgen_specular_roughness_from_power(specular_power)
        self._set_socket_default(bsdf, 'Roughness', roughness)
        specular_input = bsdf.inputs.get('Specular IOR Level') or bsdf.inputs.get('Specular')
        if specular_input is not None:
            specular_input.default_value = self._pc_nextgen_specular_level_from_power(specular_power, specular_image is not None)

        if rim_intensity > 0.0 and len(rim_color) >= 3:
            emission_color_input = bsdf.inputs.get('Emission Color') or bsdf.inputs.get('Emission')
            emission_strength_input = bsdf.inputs.get('Emission Strength')
            if emission_color_input is not None:
                try:
                    emission_color_input.default_value = (
                        max(0.0, min(1.0, float(rim_color[0]))),
                        max(0.0, min(1.0, float(rim_color[1]))),
                        max(0.0, min(1.0, float(rim_color[2]))),
                        1.0,
                    )
                except Exception:
                    pass
            if emission_strength_input is not None:
                try:
                    emission_strength_input.default_value = min(2.0, float(rim_intensity) * 0.35)
                except Exception:
                    pass

        self._pc_nextgen_attach_normal_map(nodes, links, bsdf, normal_image)
        self._pc_nextgen_attach_specular_map(
            nodes,
            links,
            bsdf,
            specular_image,
            tint_color=specular_tint,
            specular_power=specular_power,
            vertex_tint_node=vertex_color if use_pcng_vertex_specular_tint else None,
        )

        alpha_input = bsdf.inputs.get('Alpha')
        shader_output = bsdf.outputs['BSDF']
        if blend_mode == 1 or combiner_type == 6:
            clipped_alpha_output = self._pc_nextgen_alpha_test_output(nodes, links, alpha_output, threshold=0.5)
            if clipped_alpha_output is not None and alpha_input is not None:
                links.new(clipped_alpha_output, alpha_input)
            material.blend_method = 'CLIP'
        elif blend_mode in {2, 4, 5, 8, 9, 13} or opacity < 0.999:
            if alpha_output is not None and alpha_input is not None:
                links.new(alpha_output, alpha_input)
            material.blend_method = 'BLEND'

        if combiner_type in {2, 3, 4}:
            # Reflection variants use reflection-vector texcoord sources in the
            # game.  Blender cannot directly reproduce that fixed-function setup
            # here, but using the diffuse/reflection layer as an overlay gives a
            # visibly distinct material instead of the old flat diffuse fallback.
            reflection_image = None
            for layer_index, texcoord_source in enumerate(layer_texcoord_sources):
                if int(texcoord_source) in {6, 9} and layer_index < len(layer_images):
                    reflection_image = layer_images[layer_index]
                    break
            if reflection_image is None and diffuse_layer is not None and diffuse_layer < len(layer_images):
                reflection_image = layer_images[diffuse_layer]
            if reflection_image is not None:
                shader_output = self._apply_reflection_overlay(nodes, links, shader_output, reflection_image, 'env')
                material.blend_method = 'BLEND'

        links.new(shader_output, output.inputs['Surface'])

    def _setup_material_nodes(
        self,
        material,
        image: Optional[bpy.types.Image],
        tpage_flags: Dict[str, int],
        use_vertex_colors: bool = False,
        vertex_color_attribute_name: Optional[str] = None,
        reflective: bool = False,
        reflection_mode: str = 'env',
        normal_image: Optional[bpy.types.Image] = None,
        specular_image: Optional[bpy.types.Image] = None,
        ps2_stage_overlay_image: Optional[bpy.types.Image] = None,
    ) -> None:
        try:
            _set_material_panel_value(material, 'trlau_ui_env_mapping', bool(reflective and reflection_mode in {'env', 'both'}))
            _set_material_panel_value(material, 'trlau_ui_eye_ref_env_mapping', bool(reflective and reflection_mode in {'eye_ref', 'both'}))
        except Exception:
            pass
        if _get_material_platform_name(material) == 'pc_nextgen':
            # Hard guard: if a shared material refresh accidentally reaches the
            # legacy node builder, preserve the PCMaterialData path instead of
            # rebuilding a tpage/diffuse-only old-gen shader.
            layer_images = [image, normal_image, specular_image]
            if len(layer_images) < 8:
                layer_images.extend([None] * (8 - len(layer_images)))
            self._setup_pc_nextgen_material_nodes(
                material,
                layer_images,
                tpage_flags,
                use_vertex_colors=use_vertex_colors,
                vertex_color_attribute_name=vertex_color_attribute_name,
            )
            return
        if _get_material_platform_name(material) in {'ps3', 'xbox360'}:
            self._setup_ps3_material_nodes(
                material,
                image,
                tpage_flags,
                use_vertex_colors=use_vertex_colors,
                vertex_color_attribute_name=vertex_color_attribute_name,
                reflective=reflective,
                reflection_mode=reflection_mode,
                normal_image=normal_image,
                specular_image=specular_image,
            )
            return
        if (
            _get_material_platform_name(material) == 'underworld'
            and ps2_stage_overlay_image is not None
            and str(self._safe_material_get(material, 'trlau_tr8_ps2_stage_blend_mode', '') or '') == 'secondary_base_primary_alpha'
        ):
            self._setup_tr8_ps2_stage_overlay_nodes(
                material,
                image,
                ps2_stage_overlay_image,
                tpage_flags,
                use_vertex_colors=use_vertex_colors,
                vertex_color_attribute_name=vertex_color_attribute_name,
            )
            return
        if reflective and image is not None:
            self._setup_reflective_material_nodes_exact(material, image, tpage_flags, reflection_mode=reflection_mode)
            return

        nodes, links, output = self._prepare_material_node_tree(material, tpage_flags)
        is_invisible = self._apply_culling_mode_settings(material, tpage_flags['cull_mode'])
        if is_invisible:
            self._create_invisible_shader(nodes, links, output)
            return

        blend_value = int(tpage_flags.get('blend_value', 0))
        if blend_value == 4:
            self._setup_blend4_material_nodes(
                material,
                image,
                tpage_flags,
                use_vertex_colors=use_vertex_colors,
                vertex_color_attribute_name=vertex_color_attribute_name,
            )
            return

        if blend_value == 9:
            self._setup_blend9_material_nodes(
                material,
                image,
                tpage_flags,
                use_vertex_colors=use_vertex_colors,
                vertex_color_attribute_name=vertex_color_attribute_name,
            )
            return

        vertex_color = self._create_vertex_color_node(nodes, attribute_name=vertex_color_attribute_name) if use_vertex_colors else None
        color_output, alpha_output = self._base_material_color_alpha_outputs(material, nodes, links, image, tpage_flags, vertex_color)

        texture_id = int(tpage_flags.get('texture_id', 0))
        if texture_id == 0:
            material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
            if vertex_color is None:
                color_output = None
                alpha_output = None
        elif image is None:
            color_output, alpha_output = self._apply_missing_texture_wireframe(nodes, links, vertex_color)

        clipped_alpha_output = None
        if blend_value == 0 and alpha_output is not None:
            clipped_alpha_output = self._create_alpha_greater_than_node(nodes, links, alpha_output)

        if blend_value == 2:
            shader_output = self._create_additive_alpha_shader(nodes, links, color_output, alpha_output)
            material.blend_method = 'BLEND'
        elif tpage_flags['flat_shading']:
            shader_output, blend_method = self._create_flat_shader(nodes, links, color_output, alpha_output, clipped_alpha_output, tpage_flags, is_invisible)
            if blend_method is not None:
                material.blend_method = blend_method
        else:
            shader_output, blend_method = self._create_principled_shader(
                nodes,
                links,
                color_output,
                alpha_output,
                clipped_alpha_output,
                tpage_flags,
                is_invisible,
                normal_image=normal_image,
                specular_image=specular_image,
            )
            if blend_method is not None:
                material.blend_method = blend_method

        if reflective and shader_output is not None and image is not None:
            shader_output = self._apply_reflection_overlay(nodes, links, shader_output, image, reflection_mode)

        if shader_output is not None:
            links.new(shader_output, output.inputs['Surface'])

    def _apply_reflective_material_to_material(self, material, source_material_index: int, reflection_mode: str = 'env'):
        try:
            _set_material_panel_value(material, 'trlau_ui_env_mapping', reflection_mode in {'env', 'both'})
            _set_material_panel_value(material, 'trlau_ui_eye_ref_env_mapping', reflection_mode in {'eye_ref', 'both'})
        except Exception:
            pass
        tpage_flags = {
            'texture_id': int(getattr(material, 'trlau_ui_texture_id', 0)) & 0x1FFF,
            'blend_value': int(getattr(material, 'trlau_ui_blend_value', 0)) & 0xF,
            'cull_mode': int(getattr(material, 'trlau_ui_cull_mode', 0)) & 0x7,
            'unknown_1': 1 if bool(getattr(material, 'trlau_ui_unknown_1', False)) else 0,
            'single_sided': 1 if bool(getattr(material, 'trlau_ui_single_sided', False)) else 0,
            'texture_wrap': int(getattr(material, 'trlau_ui_texture_wrap', 0)) & 0x3,
            'unknown_2': 1 if bool(getattr(material, 'trlau_ui_unknown_2', False)) else 0,
            'unknown_3': 1 if bool(getattr(material, 'trlau_ui_unknown_3', False)) else 0,
            'flat_shading': 1 if bool(getattr(material, 'trlau_ui_flat_shading', False)) else 0,
            'sort_z': 1 if bool(getattr(material, 'trlau_ui_sort_z', False)) else 0,
            'stencil_pass': int(getattr(material, 'trlau_ui_stencil_pass', 0)) & 0x3,
            'stencil_func': 1 if bool(getattr(material, 'trlau_ui_stencil_func', False)) else 0,
            'alpha_ref': 1 if bool(getattr(material, 'trlau_ui_alpha_ref', False)) else 0,
        }
        image = None
        try:
            if material.use_nodes and material.node_tree is not None:
                for node in material.node_tree.nodes:
                    if node.bl_idname == 'ShaderNodeTexImage' and getattr(node, 'image', None) is not None:
                        image = node.image
                        break
        except Exception:
            image = None
        use_vertex_colors = any(getattr(node, 'bl_idname', '') in {'ShaderNodeVertexColor', 'ShaderNodeAttribute'} for node in getattr(getattr(material, 'node_tree', None), 'nodes', []) or [])
        self._setup_material_nodes(material, image, tpage_flags, use_vertex_colors=use_vertex_colors, reflective=True, reflection_mode=reflection_mode)
        return material
