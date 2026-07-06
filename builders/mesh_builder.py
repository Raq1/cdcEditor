from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy

from ..core.blender_mesh_utils import get_or_create_color_attribute, set_active_color_attribute
from ..core.log import logger
from ..core.model_types import Face, ModelData, MVertex
from .transform_utils import get_vertex_translation
from ..platforms.common.model import decode_normal as _decode_normal
from ..platforms.common.texture_loader import PlatformTextureLoaderMixin
from ..platforms.pc.model import decode_pc_uv
from ..platforms.nintendo.model import decode_gamecube_uv
from ..platforms.ps2.model import decode_ps2_uv
from ..platforms.psp.model import PspModelBuilderMixin, decode_psp_uv
from ..platforms.ps3.model import decode_ps3_uv
from ..platforms.xbox360.model import decode_xbox360_uv
from .material_nodes import MaterialNodeBuilderMixin, _decode_tpage_flags
from ..platforms.xbox.model import decode_xbox_uv


def _decode_uv(
    vertex: MVertex,
    model: ModelData | None = None,
    texture_size: Tuple[int, int] | None = None,
) -> Tuple[float, float]:
    uv_format = getattr(model, "uv_format", "pc") if model is not None else "pc"
    if uv_format == 'gamecube':
        return decode_gamecube_uv(vertex)
    if uv_format == 'wii':
        return decode_gamecube_uv(vertex)
    if uv_format == 'xbox':
        return decode_xbox_uv(vertex)
    if uv_format == 'ps2':
        return decode_ps2_uv(vertex)
    if uv_format == 'psp':
        return decode_psp_uv(vertex)
    if uv_format == 'ps3':
        return decode_ps3_uv(vertex)
    if uv_format == 'xbox360':
        return decode_xbox360_uv(vertex)
    if uv_format in {'underworld', 'pc_nextgen'}:
        decoded = getattr(vertex, 'uv_decoded', None)
        if decoded is not None:
            return decoded
    return decode_pc_uv(vertex)



def _set_material_panel_value(material, name: str, value) -> None:
    try:
        from ..core.material_ui import set_material_panel_value
        set_material_panel_value(material, name, value)
        return
    except Exception:
        pass
    try:
        setattr(material, name, value)
    except Exception:
        pass


def _set_material_platform_value(material, platform: str | None) -> None:
    try:
        from ..core.material_ui import set_material_platform_name
        set_material_platform_name(material, platform)
    except Exception:
        pass


def _cleanup_material_metadata(material) -> None:
    try:
        from ..core.material_ui import cleanup_material_metadata
        cleanup_material_metadata(material)
    except Exception:
        pass




class MeshBuildSettings:
    def __init__(self, apply_segment_pivots: bool = True, import_textures: bool = True):
        self.apply_segment_pivots = apply_segment_pivots
        self.import_textures = import_textures


class MeshBuilder(PspModelBuilderMixin, PlatformTextureLoaderMixin, MaterialNodeBuilderMixin):
    def __init__(self, context, filepath: str, settings: MeshBuildSettings, collection=None, object_name: Optional[str] = None):
        self.context = context
        self.filepath = filepath
        self.settings = settings
        self.collection = collection or context.collection
        self.object_name = object_name
        self._loaded_texture_cache: Dict[str, bpy.types.Image] = {}
        self._texture_dimension_cache: Dict[str, Tuple[int, int] | None] = {}
        self._texture_path_lookup_cache: Dict[int, Path | None] = {}
        self._texture_path_index_cache: Dict[Path, Dict[int, Path]] = {}
        self._active_uv_format: str | None = None


    def _final_vertex_position(self, vertex: MVertex, model: ModelData) -> Tuple[float, float, float]:
        sx, sy, sz, _sw = model.model_scale
        x = float(vertex.position_raw[0]) * sx
        y = float(vertex.position_raw[1]) * sy
        z = float(vertex.position_raw[2]) * sz

        # PS3 and X360 are sane platforms and they don't do cursed stuff
        uses_pretransformed_stream_positions = str(getattr(model, 'uv_format', '') or '').lower() in {'ps3', 'xbox360', 'underworld', 'pc_nextgen'}
        if self.settings.apply_segment_pivots and not uses_pretransformed_stream_positions:
            tx, ty, tz = get_vertex_translation(model, vertex.index)
            x += tx
            y += ty
            z += tz

        return (x, y, z)


    @staticmethod
    def _material_group_key(strip, strip_index: int) -> int:
        try:
            group = int(getattr(strip, 'material_group', -1))
        except Exception:
            group = -1
        return group if group >= 0 else strip_index

    def _build_material_slot_map(self, model: ModelData) -> tuple[dict[int, int], list[int]]:
        slot_by_group: dict[int, int] = {}
        representative_strip_indices: list[int] = []
        strips = list(getattr(model, 'strips', []) or [])

        if str(getattr(model, 'uv_format', '') or '').lower() == 'pc_nextgen':
            representative_by_group: dict[int, int] = {}
            for strip_index, strip in enumerate(strips):
                key = self._material_group_key(strip, strip_index)
                if key < 0 or key in representative_by_group:
                    continue
                representative_by_group[int(key)] = int(strip_index)
            for key in sorted(representative_by_group):
                slot_by_group[int(key)] = len(representative_strip_indices)
                representative_strip_indices.append(representative_by_group[int(key)])
            return slot_by_group, representative_strip_indices

        for strip_index, strip in enumerate(strips):
            key = self._material_group_key(strip, strip_index)
            if key in slot_by_group:
                continue
            slot_by_group[key] = len(representative_strip_indices)
            representative_strip_indices.append(strip_index)
        return slot_by_group, representative_strip_indices

    def _faces_from_triangle_lists(self, model: ModelData) -> Tuple[List[Face], List[int]]:
        triangle_faces: List[Face] = []
        material_indices: List[int] = []
        slot_by_group, _representative_strip_indices = self._build_material_slot_map(model)
        for strip_index, entry in enumerate(model.strips):
            indices = entry.indices
            usable_count = len(indices) - (len(indices) % 3)
            material_slot = slot_by_group.get(self._material_group_key(entry, strip_index), 0)
            for i in range(0, usable_count, 3):
                face = (indices[i], indices[i + 1], indices[i + 2])
                triangle_faces.append(face)
                material_indices.append(material_slot)
        return triangle_faces, material_indices

    def _collect_faces(self, model: ModelData, vertex_count: int) -> Tuple[List[Face], List[int], List[int]]:
        triangle_faces, material_indices = self._faces_from_triangle_lists(model)
        if not triangle_faces:
            triangle_faces = [(face.v0, face.v1, face.v2) for face in model.faces]
            material_indices = [0] * len(triangle_faces)

        valid_faces: List[Face] = []
        valid_material_indices: List[int] = []
        source_face_indices: List[int] = []
        for source_face_index, (face, material_index) in enumerate(zip(triangle_faces, material_indices)):
            if len(face) != 3:
                continue
            if min(face) < 0 or max(face) >= vertex_count:
                continue
            if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
                continue
            valid_faces.append(face)
            valid_material_indices.append(material_index)
            source_face_indices.append(source_face_index)
        return valid_faces, valid_material_indices, source_face_indices

    @staticmethod
    def _material_image_size(material) -> Tuple[int, int] | None:
        try:
            if material is None or not material.use_nodes or material.node_tree is None:
                return None
            for node in material.node_tree.nodes:
                if node.bl_idname != 'ShaderNodeTexImage':
                    continue
                image = getattr(node, 'image', None)
                if image is None:
                    continue
                width, height = image.size
                width = int(width)
                height = int(height)
                if width > 0 and height > 0:
                    return (width, height)
        except Exception:
            return None
        return None

    def _apply_uvs(self, mesh, model: ModelData) -> None:
        if not model.vertices or not mesh.loops:
            return

        uv_layer = mesh.uv_layers.new(name="UVMap")

        if getattr(model, "uv_format", "") == 'psp':
            material_texture_sizes: Dict[int, Tuple[int, int] | None] = {}
            for polygon in mesh.polygons:
                material_index = int(polygon.material_index)
                if material_index not in material_texture_sizes:
                    material = mesh.materials[material_index] if 0 <= material_index < len(mesh.materials) else None
                    material_texture_sizes[material_index] = self._material_image_size(material)
                texture_size = material_texture_sizes.get(material_index)
                for loop_index in polygon.loop_indices:
                    loop = mesh.loops[loop_index]
                    uv_layer.data[loop.index].uv = _decode_uv(
                        model.vertices[loop.vertex_index],
                        model,
                        texture_size=texture_size,
                    )
            return

        decoded_uvs = [_decode_uv(vertex, model) for vertex in model.vertices]

        for loop in mesh.loops:
            uv_layer.data[loop.index].uv = decoded_uvs[loop.vertex_index]

    @staticmethod
    def _ps3_custom_normal_quality(mesh, decoded_normals: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        total = 0
        dot_sum = 0.0
        aligned = 0
        for polygon in mesh.polygons:
            polygon_normal = polygon.normal
            pn = (float(polygon_normal.x), float(polygon_normal.y), float(polygon_normal.z))
            for loop_index in polygon.loop_indices:
                loop = mesh.loops[loop_index]
                if loop.vertex_index < 0 or loop.vertex_index >= len(decoded_normals):
                    continue
                normal = decoded_normals[loop.vertex_index]
                dot = (normal[0] * pn[0]) + (normal[1] * pn[1]) + (normal[2] * pn[2])
                dot_sum += dot
                if dot > 0.15:
                    aligned += 1
                total += 1
        if total <= 0:
            return 0.0, 0.0
        return dot_sum / float(total), aligned / float(total)

    def _apply_normals(self, mesh, model: ModelData) -> None:
        if not model.vertices or not mesh.loops:
            return

        for polygon in mesh.polygons:
            polygon.use_smooth = True

        is_ps3 = str(getattr(model, 'uv_format', '') or '').lower() in {'ps3', 'xbox360'}
        if is_ps3:
            # No normals for whatever reason
            return

        decoded_normals = [_decode_normal(vertex) for vertex in model.vertices]
        loop_normals = [decoded_normals[loop.vertex_index] for loop in mesh.loops]

        if hasattr(mesh, "normals_split_custom_set"):
            mesh.normals_split_custom_set(loop_normals)


    def _apply_vertex_colors(self, mesh, model: ModelData) -> None:
        if not model.vertex_colors:
            return
        if len(model.vertex_colors) != len(model.vertices):
            logger.warning(
                'Vertex color count mismatch: colors=%d vertices=%d; skipping Color attribute creation',
                len(model.vertex_colors),
                len(model.vertices),
            )
            return

        render_uv_format = str(getattr(model, 'uv_format', '') or '').lower()
        is_ps3 = render_uv_format in {'ps3', 'xbox360'}
        uses_255_alpha_scale = render_uv_format in {'ps3', 'xbox360', 'pc_nextgen'} or any(
            getattr(vertex, 'psp_color_rgba', None) is not None or getattr(vertex, 'ps3_color_rgba', None) is not None
            for vertex in model.vertices
        )
        alpha_denominator = 255.0 if uses_255_alpha_scale else 128.0
        color_attribute_name = 'PS3_VertexColor' if is_ps3 else 'Color'

        flat_colors: List[float] = []
        flat_colors_extend = flat_colors.extend
        for r, g, b, a in model.vertex_colors:
            flat_colors_extend((r / 255.0, g / 255.0, b / 255.0, min(a / alpha_denominator, 1.0)))

        color_attr = get_or_create_color_attribute(mesh, color_attribute_name, domain='POINT')
        if color_attr is None and getattr(mesh, 'color_attributes', None) is not None:
            logger.warning('Failed to create POINT color attribute %s on mesh.color_attributes', color_attribute_name)

        if color_attr is not None:
            try:
                color_attr.data.foreach_set('color', flat_colors)
                set_active_color_attribute(mesh, color_attr, color_attribute_name)
                return
            except Exception as exc:
                logger.warning('Failed writing POINT color attribute %s: %s', color_attribute_name, exc)

        logger.warning('Unable to create any Blender color attribute/layer for parsed vertex colors')

    def _apply_ps3_vertex_color_raw_attributes(self, mesh, model: ModelData) -> None:
        if str(getattr(model, 'uv_format', '') or '').lower() not in {'ps3', 'xbox360'} or not model.vertices:
            return
        if not any(getattr(vertex, 'ps3_color_rgba', None) is not None for vertex in model.vertices):
            return
        channels = (
            ('TRLauPS3VertexColorR', 0),
            ('TRLauPS3VertexColorG', 1),
            ('TRLauPS3VertexColorB', 2),
            ('TRLauPS3VertexColorA', 3),
        )
        values_by_channel: dict[str, list[int]] = {name: [] for name, _index in channels}
        for vertex in model.vertices:
            color = getattr(vertex, 'ps3_color_rgba', None) or (255, 255, 255, 255)
            for name, channel_index in channels:
                values_by_channel[name].append(int(color[channel_index]) & 0xFF)
        for name, _channel_index in channels:
            attr = self._create_generic_attribute(mesh, name, 'INT', 'POINT')
            if attr is None:
                continue
            try:
                attr.data.foreach_set('value', values_by_channel[name])
            except Exception as exc:
                logger.warning('Failed writing PS3 raw vertex color attribute %s: %s', name, exc)



    def _create_generic_attribute(self, mesh, name: str, data_type: str, domain: str):
        attributes = getattr(mesh, 'attributes', None)
        if attributes is None:
            return None
        attribute = attributes.get(name)
        if attribute is not None:
            return attribute
        try:
            return attributes.new(name=name, type=data_type, domain=domain)
        except Exception as exc:
            logger.warning('Failed to create attribute %s (%s/%s): %s', name, data_type, domain, exc)
            return None

    @staticmethod
    def _mface_corner_count(face, corner: int) -> int:
        if corner == 0:
            value = int(getattr(face, 'same_vertex_count_0', 0) or 0)
        elif corner == 1:
            value = int(getattr(face, 'same_vertex_count_1', 0) or 0)
        else:
            value = int(getattr(face, 'same_vertex_count_2', 0) or 0)
        return max(1, min(31, value))

    def _collect_mface_source_ranges(self, model: ModelData) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
        ranges_by_start: Dict[int, int] = {}
        vertex_count = len(model.vertices or [])
        for face in model.faces or []:
            starts = (int(face.v0), int(face.v1), int(face.v2))
            for corner, start in enumerate(starts):
                if start < 0 or start >= vertex_count:
                    continue
                count = self._mface_corner_count(face, corner)
                count = max(1, min(count, vertex_count - start))
                previous = ranges_by_start.get(start)
                if previous is None or count > previous:
                    ranges_by_start[start] = count

        if not ranges_by_start:
            for index in range(vertex_count):
                ranges_by_start[index] = 1

        covered = [False] * vertex_count
        for start, count in list(ranges_by_start.items()):
            for index in range(start, min(vertex_count, start + count)):
                covered[index] = True
        for index, is_covered in enumerate(covered):
            if not is_covered:
                ranges_by_start[index] = 1

        ranges = sorted((int(start), int(count)) for start, count in ranges_by_start.items())
        source_by_game_vertex: Dict[int, int] = {}
        for source_index, (start, count) in enumerate(ranges):
            for game_vertex in range(start, min(vertex_count, start + count)):
                source_by_game_vertex.setdefault(game_vertex, source_index)
        for game_vertex in range(vertex_count):
            source_by_game_vertex.setdefault(game_vertex, game_vertex)
        return ranges, source_by_game_vertex

    def _clear_render_mface_metadata(self, mesh) -> None:
        try:
            attributes = getattr(mesh, 'attributes', None)
            attr = attributes.get('MFacePoint') if attributes is not None else None
            if attr is not None:
                attributes.remove(attr)
        except Exception:
            pass
        for key in ('trlau_mface_sort_top_to_bottom',):
            try:
                if key in mesh:
                    del mesh[key]
            except Exception:
                pass

    def _cleanup_import_debug_metadata(self, obj) -> None:
        """Remove importer/debug ID properties from newly built data blocks."""
        stale_object_keys = (
            'trlau_model_scale',
        )
        for key in stale_object_keys:
            try:
                if key in obj:
                    del obj[key]
            except Exception:
                pass
        stale_mesh_keys = (
            'trlau_uv_raw_attribute_count',
            'trlau_ps3_custom_normals_applied',
            'trlau_ps3_custom_normals_reason',
            'trlau_ps3_vertex_color_attribute',
            'trlau_ps3_vertex_color_count',
            'trlau_mface_metadata_version',
            'trlau_mface_storage_type',
            'trlau_mface_vertex_count',
            'trlau_mface_face_count',
            'trlau_mface_faces_flat',
            'trlau_mface_source_representatives',
            'trlau_mface_source_counts',
            'trlau_loop_source_vertices',
        )
        stale_material_keys = (
            'trlau_ps3_specular_shader_source',
            'trlau_reflective_source_material_index',
            'trlau_texture_strip_source_index',
            'trlau_source_strip_index',
            'trlau_source_material_slot',
            'trlau_psp_debug_mode_word',
            'trlau_psp_debug_vertex_mode',
            'trlau_psp_debug_render_flags',
            'trlau_psp_debug_vertex_format_flags',
        )
        mesh = getattr(obj, 'data', None)
        if mesh is not None:
            for key in stale_mesh_keys:
                try:
                    if key in mesh:
                        del mesh[key]
                except Exception:
                    pass
            try:
                attributes = getattr(mesh, 'attributes', None)
                if attributes is not None:
                    for attr_name in (
                        'TRLauUVRawU', 'TRLauUVRawV',
                        'TrlauUVRawU', 'TrlauUVRawV',
                        'TRLauPS2UVRawU', 'TRLauPS2UVRawV',
                        'TrlauPS2UVRawU', 'TrlauPS2UVRawV',
                        'trlau_uv_raw_u', 'trlau_uv_raw_v',
                        'trlau_ps2_uv_raw_u', 'trlau_ps2_uv_raw_v',
                    ):
                        attr = attributes.get(attr_name)
                        if attr is not None:
                            attributes.remove(attr)
            except Exception:
                pass
        for slot in getattr(obj, 'material_slots', []) or []:
            material = getattr(slot, 'material', None)
            if material is None:
                continue
            for key in stale_material_keys:
                try:
                    if key in material:
                        del material[key]
                except Exception:
                    pass

    def _apply_mface_metadata(self, mesh, model: ModelData) -> None:
        self._clear_render_mface_metadata(mesh)
        vertex_count = len(model.vertices or [])
        source_faces = list(model.faces or [])
        if vertex_count <= 0 or not source_faces:
            return
        if not any(int(getattr(face, 'same_vert_bits', 0) or 0) != 0 for face in source_faces):
            return

        _ranges, source_by_game_vertex = self._collect_mface_source_ranges(model)
        try:
            attributes = getattr(mesh, 'attributes', None)
            if attributes is None:
                return
            attr = attributes.new(name='MFacePoint', type='INT', domain='POINT')
            values = [int(source_by_game_vertex.get(index, index)) for index in range(len(mesh.vertices))]
            try:
                attr.data.foreach_set('value', values)
            except Exception:
                for item, value in zip(attr.data, values):
                    item.value = int(value)
            mesh['trlau_mface_sort_top_to_bottom'] = False
        except Exception as exc:
            logger.warning('Failed storing MFacePoint attribute: %s', exc)

    def _build_texture_path_index(self, model_dir: Path) -> Dict[int, Path]:
        cache_key = Path(model_dir).resolve()
        cached = self._texture_path_index_cache.get(cache_key)
        if cached is not None:
            return cached

        index: Dict[int, Path] = {}
        try:
            entries = sorted(cache_key.iterdir())
        except Exception:
            entries = []

        for entry in entries:
            if not entry.is_file() or entry.suffix.lower() != ".pcd":
                continue
            suffix = entry.stem.rsplit('_', 1)[-1]
            try:
                candidate_id = int(suffix, 16)
            except Exception:
                continue
            index.setdefault(int(candidate_id), entry)
            index.setdefault(int(candidate_id) & 0x1FFF, entry)

        self._texture_path_index_cache[cache_key] = index
        return index

    def _find_texture_path(self, texture_id: int) -> Optional[Path]:
        try:
            raw_id = int(texture_id)
        except Exception:
            return None
        if raw_id < 0:
            return None

        cache_key = int(raw_id) & 0xFFFFFFFF
        if cache_key in self._texture_path_lookup_cache:
            return self._texture_path_lookup_cache[cache_key]

        model_dir = Path(self.filepath).resolve().parent
        index = self._build_texture_path_index(model_dir)
        result = index.get(raw_id)
        if result is None:
            result = index.get(raw_id & 0x1FFF)
        self._texture_path_lookup_cache[cache_key] = result
        return result

    @staticmethod
    def _signed_u32(value: int) -> int:
        value = int(value) & 0xFFFFFFFF
        return value - 0x100000000 if value & 0x80000000 else value

    @staticmethod
    def _apply_console_cull_single_sided_rule(tpage_flags: Dict[str, int], material_platform: str) -> Dict[str, int]:
        flags = dict(tpage_flags)
        if str(material_platform or '').lower() in {'ps3', 'xbox360'}:
            console_single_sided_value = int(flags.get('cull_mode', 0)) & 0x7
            flags['single_sided'] = 1 if console_single_sided_value == 2 else 0
            flags['cull_mode'] = 0
        return flags

    @staticmethod
    def _set_tpage_material_properties(material, tpage_flags: Dict[str, int]) -> None:
        for flag_name in (
            'blend_value',
            'cull_mode',
            'unknown_1',
            'single_sided',
            'texture_wrap',
            'unknown_2',
            'unknown_3',
            'flat_shading',
            'sort_z',
            'stencil_pass',
            'stencil_func',
            'alpha_ref',
        ):
            material[f'trlau_{flag_name}'] = int(tpage_flags[flag_name])

    @staticmethod
    def _set_texture_dimension_properties(material, texture_dimensions: Tuple[int, int] | None) -> None:
        for key in ('trlau_texture_width', 'trlau_texture_height'):
            try:
                if key in material:
                    del material[key]
            except Exception:
                pass

    def _set_psp_material_properties(self, material, strip, texture_id: int) -> None:
        psp_mode_word = int(getattr(strip, 'psp_mode_word', 0) or 0)
        psp_vertex_mode = int(getattr(strip, 'psp_vertex_mode', 0) or 0)
        psp_render_flags = int(getattr(strip, 'psp_render_flags', 0) or 0)
        psp_blend = int(getattr(strip, 'psp_blend', 0) or 0)
        psp_vertex_format_flags = int(getattr(strip, 'psp_vertex_format_flags', 0) or 0)

        material['trlau_psp_texture_id'] = int(texture_id)
        material['trlau_psp_mode_word'] = psp_mode_word
        material['trlau_psp_vertex_mode'] = psp_vertex_mode
        material['trlau_psp_render_flags'] = psp_render_flags
        material['trlau_psp_blend'] = psp_blend
        material['trlau_psp_vertex_format_flags'] = psp_vertex_format_flags

    @staticmethod
    def _valid_texture_id(texture_id: int) -> int:
        try:
            value = int(texture_id)
        except Exception:
            return -1
        return value if value >= 0 else -1

    def _set_ps3_material_properties(
        self,
        material,
        strip,
        *,
        diffuse_texture_id: int,
        normal_texture_id: int,
        specular_texture_id: int,
        normal_texture_path: Path | None,
        specular_texture_path: Path | None,
        external_render_stream: bool = False,
    ) -> None:
        stage_ids = [int(value) for value in getattr(strip, 'ps3_texture_stage_ids', []) or []]
        stage_tpageids = [int(value) for value in getattr(strip, 'ps3_texture_stage_tpageids', []) or []]

        _set_material_panel_value(material, 'trlau_ui_ps3_diffuse_texture_id', int(diffuse_texture_id))
        _set_material_panel_value(material, 'trlau_ui_ps3_normal_texture_id', int(normal_texture_id))
        _set_material_panel_value(material, 'trlau_ui_ps3_normal_candidate_texture_id', int(getattr(strip, 'ps3_normal_candidate_texture_id', -1)))
        _set_material_panel_value(material, 'trlau_ui_ps3_specular_texture_id', int(specular_texture_id))
        _set_material_panel_value(material, 'trlau_ui_ps3_external_render_stream', bool(external_render_stream))
        _set_material_panel_value(material, 'trlau_ui_ps3_role_mapping_version', 4)

        for key in (
            'trlau_ps3_texture_stage_ids',
            'trlau_ps3_texture_stage_tpageids',
            'trlau_ps3_has_normal_map',
            'trlau_ps3_has_specular_map',
        ):
            try:
                if key in material:
                    del material[key]
            except Exception:
                pass

    def _set_tr8_material_properties(
        self,
        material,
        strip,
        *,
        diffuse_texture_id: int,
        normal_texture_id: int,
        specular_texture_id: int = -1,
    ) -> None:
        def _set_prop(name: str, value) -> None:
            try:
                material[name] = value
            except Exception:
                try:
                    material[name] = str(value)
                except Exception:
                    pass

        def _csv(values) -> str:
            try:
                return ','.join(str(int(value)) for value in values)
            except Exception:
                return ''

        def _csv_str(values) -> str:
            try:
                return ','.join(str(value) for value in values if str(value))
            except Exception:
                return ''

        _set_prop('trlau_tr8_material_id', int(getattr(strip, 'tr8_material_resource_id', -1)))
        _set_prop('trlau_tr8_material_file', str(getattr(strip, 'tr8_material_file', '') or ''))
        _set_prop('trlau_tr8_shader_ids', _csv(getattr(strip, 'tr8_shader_ids', []) or []))
        _set_prop('trlau_tr8_shader_files', _csv_str(getattr(strip, 'tr8_shader_files', []) or []))
        _set_prop('trlau_tr8_shader_strings', _csv_str(getattr(strip, 'tr8_shader_strings', []) or []))
        _set_prop('trlau_tr8_texture_stage_ids', _csv(getattr(strip, 'tr8_texture_stage_ids', []) or []))
        _set_prop('trlau_tr8_texture_stage_types', _csv(getattr(strip, 'tr8_texture_stage_types', []) or []))
        _set_prop('trlau_tr8_texture_stage_slots', _csv(getattr(strip, 'tr8_texture_stage_slots', []) or []))
        _set_prop('trlau_tr8_diffuse_texture_id', int(diffuse_texture_id))
        _set_prop('trlau_tr8_normal_texture_id', int(normal_texture_id))
        _set_prop('trlau_tr8_specular_texture_id', int(specular_texture_id))
        _set_prop('trlau_tr8_ao_texture_id', int(getattr(strip, 'tr8_ao_texture_id', -1)))
        _set_prop('trlau_tr8_detail_texture_id', int(getattr(strip, 'tr8_detail_texture_id', -1)))
        _set_prop('trlau_tr8_detail_ao_texture_id', int(getattr(strip, 'tr8_detail_ao_texture_id', -1)))
        _set_prop('trlau_tr8_mask_texture_id', int(getattr(strip, 'tr8_mask_texture_id', -1)))
        _set_prop('trlau_tr8_reflection_texture_id', int(getattr(strip, 'tr8_reflection_texture_id', -1)))
        double_sided = bool(getattr(strip, 'tr8_double_sided', False))
        double_wound_pair_count = int(getattr(strip, 'tr8_double_wound_pair_count', 0) or 0)
        _set_prop('trlau_tr8_double_sided', bool(double_sided))
        _set_prop('trlau_tr8_double_wound_pair_count', int(double_wound_pair_count))
        try:
            _set_material_panel_value(material, 'trlau_ui_tr8_double_sided', bool(double_sided))
            material.use_backface_culling = not bool(double_sided)
        except Exception:
            pass
        _set_prop('trlau_tr8_ps2_run_flags', int(getattr(strip, 'tr8_ps2_run_flags', 0)))
        _set_prop('trlau_tr8_ps2_alpha_blend', bool(getattr(strip, 'tr8_ps2_alpha_blend', False)))
        _set_prop('trlau_tr8_ps2_material_index', int(getattr(strip, 'tr8_ps2_material_index', -1)))
        _set_prop('trlau_tr8_ps2_stage_blend_mode', str(getattr(strip, 'tr8_ps2_stage_blend_mode', '') or ''))

    def _set_pc_nextgen_material_properties(
        self,
        material,
        strip,
        *,
        diffuse_texture_id: int,
        normal_texture_id: int,
        specular_texture_id: int = -1,
    ) -> None:
        def _set_prop(name: str, value) -> None:
            try:
                material[name] = value
            except Exception:
                try:
                    material[name] = str(value)
                except Exception:
                    pass

        def _csv_int(values) -> str:
            try:
                return ','.join(str(int(value)) for value in values)
            except Exception:
                return ''

        def _json_float_tuples(values) -> str:
            try:
                return json.dumps([[float(component) for component in value] for value in values])
            except Exception:
                return '[]'

        _set_prop('trlau_pc_nextgen_material_id', int(getattr(strip, 'pc_nextgen_material_id', -1)))
        _set_prop('trlau_pc_nextgen_material_record_offset', int(getattr(strip, 'pc_nextgen_material_record_offset', 0)))
        _set_prop('trlau_pc_nextgen_asset_id_hi', int(getattr(strip, 'pc_nextgen_asset_id_hi', 0)))
        _set_prop('trlau_pc_nextgen_asset_id_lo', int(getattr(strip, 'pc_nextgen_asset_id_lo', 0)))
        _set_prop('trlau_pc_nextgen_asset_id_padding_hex', str(getattr(strip, 'pc_nextgen_asset_id_padding_hex', '') or ''))
        _set_prop('trlau_pc_nextgen_material_record_hex', str(getattr(strip, 'pc_nextgen_material_record_hex', '') or ''))
        _set_prop('trlau_pc_nextgen_blend_mode', int(getattr(strip, 'pc_nextgen_blend_mode', 0)))
        _set_prop('trlau_pc_nextgen_combiner_type', int(getattr(strip, 'pc_nextgen_combiner_type', 0)))
        _set_prop('trlau_pc_nextgen_material_flags', int(getattr(strip, 'pc_nextgen_material_flags', 0)))
        _set_prop('trlau_pc_nextgen_opacity', float(getattr(strip, 'pc_nextgen_opacity', 1.0)))
        _set_prop('trlau_pc_nextgen_poly_flags', int(getattr(strip, 'pc_nextgen_poly_flags', 0)))
        _set_prop('trlau_pc_nextgen_uv_auto_scroll_speed', int(getattr(strip, 'pc_nextgen_uv_auto_scroll_speed', 0)))
        _set_prop('trlau_pc_nextgen_sort_bias', float(getattr(strip, 'pc_nextgen_sort_bias', 0.0)))
        _set_prop('trlau_pc_nextgen_detail_range_mul', float(getattr(strip, 'pc_nextgen_detail_range_mul', 0.0)))
        _set_prop('trlau_pc_nextgen_detail_scale', float(getattr(strip, 'pc_nextgen_detail_scale', 0.0)))
        _set_prop('trlau_pc_nextgen_parallax_scale', float(getattr(strip, 'pc_nextgen_parallax_scale', 0.0)))
        _set_prop('trlau_pc_nextgen_parallax_offset', float(getattr(strip, 'pc_nextgen_parallax_offset', 0.0)))
        _set_prop('trlau_pc_nextgen_specular_power', float(getattr(strip, 'pc_nextgen_specular_power', 0.0)))
        _set_prop('trlau_pc_nextgen_specular_shift0', float(getattr(strip, 'pc_nextgen_specular_shift0', 0.0)))
        _set_prop('trlau_pc_nextgen_specular_shift1', float(getattr(strip, 'pc_nextgen_specular_shift1', 0.0)))
        _set_prop('trlau_pc_nextgen_rim_light_color', _json_float_tuples([getattr(strip, 'pc_nextgen_rim_light_color', (0.0, 0.0, 0.0, 0.0))]))
        _set_prop('trlau_pc_nextgen_rim_light_intensity', float(getattr(strip, 'pc_nextgen_rim_light_intensity', 0.0)))
        _set_prop('trlau_pc_nextgen_water_blend_bias', float(getattr(strip, 'pc_nextgen_water_blend_bias', 0.0)))
        _set_prop('trlau_pc_nextgen_water_blend_exponent', float(getattr(strip, 'pc_nextgen_water_blend_exponent', 0.0)))
        _set_prop('trlau_pc_nextgen_water_deep_color', _json_float_tuples([getattr(strip, 'pc_nextgen_water_deep_color', (0.0, 0.0, 0.0, 0.0))]))
        _set_prop('trlau_pc_nextgen_local_num_pixmaps', int(getattr(strip, 'pc_nextgen_local_num_pixmaps', 0)))
        _set_prop('trlau_pc_nextgen_layer_texture_ids', _csv_int(getattr(strip, 'pc_nextgen_layer_texture_ids', []) or []))
        _set_prop('trlau_pc_nextgen_layer_texture_indices', _csv_int(getattr(strip, 'pc_nextgen_layer_texture_indices', []) or []))
        _set_prop('trlau_pc_nextgen_layer_enabled', _csv_int(getattr(strip, 'pc_nextgen_layer_enabled', []) or []))
        _set_prop('trlau_pc_nextgen_layer_colors', _json_float_tuples(getattr(strip, 'pc_nextgen_layer_colors', []) or []))
        _set_prop('trlau_pc_nextgen_layer_texcoord_sources', _csv_int(getattr(strip, 'pc_nextgen_layer_texcoord_sources', []) or []))
        _set_prop('trlau_pc_nextgen_layer_modifiers', _csv_int(getattr(strip, 'pc_nextgen_layer_modifiers', []) or []))
        _set_prop('trlau_pc_nextgen_layer_param_ids', _csv_int(getattr(strip, 'pc_nextgen_layer_param_ids', []) or []))
        _set_prop('trlau_pc_nextgen_layer_constants', _json_float_tuples(getattr(strip, 'pc_nextgen_layer_constants', []) or []))
        _set_prop('trlau_pc_nextgen_layer_num_textures', _csv_int(getattr(strip, 'pc_nextgen_layer_num_textures', []) or []))
        _set_prop('trlau_pc_nextgen_shader_indices', _csv_int(getattr(strip, 'pc_nextgen_shader_indices', []) or []))
        _set_prop('trlau_pc_nextgen_shader_table_ids', _csv_int(getattr(strip, 'pc_nextgen_shader_table_ids', []) or []))
        _set_prop('trlau_pc_nextgen_special_material_flag', bool(getattr(strip, 'pc_nextgen_special_material_flag', False)))
        _set_prop('trlau_pc_nextgen_double_sided', bool(getattr(strip, 'pc_nextgen_double_sided', False)))
        _set_prop('trlau_pc_nextgen_double_wound_pair_count', int(getattr(strip, 'pc_nextgen_double_wound_pair_count', 0)))
        _set_prop('trlau_pc_nextgen_diffuse_texture_id', int(diffuse_texture_id))
        _set_prop('trlau_pc_nextgen_normal_texture_id', int(normal_texture_id))
        _set_prop('trlau_pc_nextgen_specular_texture_id', int(specular_texture_id))
        _set_prop('trlau_pc_nextgen_reversed_winding_count', int(getattr(strip, 'pc_nextgen_reversed_winding_count', 0)))
        _set_prop('trlau_pc_nextgen_index_data_offset_extra', int(getattr(strip, 'pc_nextgen_index_data_offset_extra', 0)))
        _set_prop('trlau_pc_nextgen_index_data_validation_score', str(getattr(strip, 'pc_nextgen_index_data_validation_score', '') or ''))
        _set_prop('trlau_pc_nextgen_panel_sync_version', 0)
        try:
            from ..core.material_ui import ensure_pc_nextgen_material_panel_props
            ensure_pc_nextgen_material_panel_props(material)
        except Exception:
            pass

    def _set_strip_scroll_properties(self, material, strip) -> None:
        is_animated = bool(getattr(strip, 'has_scroll_animation', False))
        raw_scroll_speed = float(strip.scroll_speed) if getattr(strip, 'scroll_speed', None) is not None else 0.0

        try:
            _set_material_panel_value(material, 'trlau_ui_draw_group', int(strip.draw_group))
            _set_material_panel_value(material, 'trlau_ui_scroll_enabled', bool(is_animated))
            _set_material_panel_value(material, 'trlau_ui_scroll_speed', raw_scroll_speed if is_animated else 0.0)
        except Exception:
            pass

    def _set_strip_material_metadata(
        self,
        material,
        strip,
        *,
        source_strip_index: int,
        material_slot: int,
        material_platform: str,
        texture_id: int,
        texture_path: Path | None,
        use_vertex_colors: bool,
        normal_texture_id: int = -1,
        specular_texture_id: int = -1,
        normal_texture_path: Path | None = None,
        specular_texture_path: Path | None = None,
        ps3_external_render_stream: bool = False,
    ) -> int:
        material_group = self._material_group_key(strip, source_strip_index)
        env_mapping = int(getattr(strip, 'env_mapping', 0))

        for stale_key in (
            'trlau_texture_strip_index',
            'trlau_material_group',
            'trlau_texture_path',
            'trlau_texture_image_name',
            'trlau_source_dir',
        ):
            try:
                if stale_key in material:
                    del material[stale_key]
            except Exception:
                pass
        _set_material_platform_value(material, material_platform)
        tpage_flags = _decode_tpage_flags(self._signed_u32(strip.tpageid))
        tpage_flags = self._apply_console_cull_single_sided_rule(tpage_flags, material_platform)
        if material_platform == 'ps2':
            try:
                raw_ps2_tpageid = int(getattr(strip, 'ps2_tpageid_raw', -1))
                if raw_ps2_tpageid >= 0:
                    tpage_flags['texture_id'] = raw_ps2_tpageid & 0xFFFF
            except Exception:
                pass
        elif material_platform == 'underworld':
            tpage_flags = _decode_tpage_flags(0)
            try:
                tpage_flags['texture_id'] = max(0, int(texture_id))
            except Exception:
                pass
        elif material_platform == 'pc_nextgen':
            tpage_flags = _decode_tpage_flags(0)
            try:
                tpage_flags['texture_id'] = max(0, int(texture_id))
            except Exception:
                pass
        try:
            texture_mask = 0xFFFF if material_platform == 'ps2' else 0x1FFF
            _set_material_panel_value(material, 'trlau_ui_texture_id', int(tpage_flags.get('texture_id', 0)) & texture_mask)
            _set_material_panel_value(material, 'trlau_ui_blend_value', int(tpage_flags.get('blend_value', 0)) & 0xF)
            _set_material_panel_value(material, 'trlau_ui_cull_mode', int(tpage_flags.get('cull_mode', 0)) & 0x7)
            _set_material_panel_value(material, 'trlau_ui_unknown_1', bool(tpage_flags.get('unknown_1', 0)))
            _set_material_panel_value(material, 'trlau_ui_single_sided', bool(tpage_flags.get('single_sided', 0)))
            _set_material_panel_value(material, 'trlau_ui_texture_wrap', int(tpage_flags.get('texture_wrap', 0)) & 0x3)
            _set_material_panel_value(material, 'trlau_ui_unknown_2', bool(tpage_flags.get('unknown_2', 0)))
            _set_material_panel_value(material, 'trlau_ui_unknown_3', bool(tpage_flags.get('unknown_3', 0)))
            _set_material_panel_value(material, 'trlau_ui_flat_shading', bool(tpage_flags.get('flat_shading', 0)))
            _set_material_panel_value(material, 'trlau_ui_sort_z', bool(tpage_flags.get('sort_z', 0)))
            _set_material_panel_value(material, 'trlau_ui_stencil_pass', int(tpage_flags.get('stencil_pass', 0)) & 0x3)
            if material_platform == 'ps2':
                _set_material_panel_value(material, 'trlau_ui_ps2_double_sided', bool(int(tpage_flags.get('stencil_pass', 0)) & 0x1))
                _set_material_panel_value(material, 'trlau_ui_single_sided', not bool(int(tpage_flags.get('stencil_pass', 0)) & 0x1))
            _set_material_panel_value(material, 'trlau_ui_stencil_func', bool(tpage_flags.get('stencil_func', 0)))
            _set_material_panel_value(material, 'trlau_ui_alpha_ref', bool(tpage_flags.get('alpha_ref', 0)))
        except Exception:
            pass
        try:
            _set_material_panel_value(material, 'trlau_ui_env_mapping', env_mapping in {64, 128})
            _set_material_panel_value(material, 'trlau_ui_eye_ref_env_mapping', env_mapping == 128)
        except Exception:
            pass

        if material_platform == 'ps2':
            try:
                if 'trlau_ps2_tpageid_raw' in material:
                    del material['trlau_ps2_tpageid_raw']
            except Exception:
                pass
        elif material_platform == 'psp':
            self._set_psp_material_properties(material, strip, texture_id)
        elif material_platform in {'ps3', 'xbox360'}:
            self._set_ps3_material_properties(
                material,
                strip,
                diffuse_texture_id=texture_id,
                normal_texture_id=normal_texture_id,
                specular_texture_id=specular_texture_id,
                normal_texture_path=normal_texture_path,
                specular_texture_path=specular_texture_path,
                external_render_stream=ps3_external_render_stream,
            )
        elif material_platform == 'underworld':
            try:
                tpage_flags['texture_id'] = max(0, int(texture_id))
            except Exception:
                pass
            self._set_tr8_material_properties(
                material,
                strip,
                diffuse_texture_id=texture_id,
                normal_texture_id=normal_texture_id,
                specular_texture_id=specular_texture_id,
            )
        elif material_platform == 'pc_nextgen':
            try:
                tpage_flags['texture_id'] = max(0, int(texture_id))
            except Exception:
                pass
            self._set_pc_nextgen_material_properties(
                material,
                strip,
                diffuse_texture_id=texture_id,
                normal_texture_id=normal_texture_id,
                specular_texture_id=specular_texture_id,
            )
        else:
            try:
                if 'trlau_texture_id' in material:
                    del material['trlau_texture_id']
            except Exception:
                pass

        self._set_strip_scroll_properties(material, strip)
        _cleanup_material_metadata(material)
        return env_mapping

    def _ensure_strip_materials(self, obj, model: ModelData) -> None:
        material_platform = str(getattr(model, 'uv_format', '') or '').lower()
        import_texture_images = bool(self.settings.import_textures) and material_platform != 'underworld'
        use_vertex_colors = bool(model.vertex_colors and len(model.vertex_colors) == len(model.vertices))
        vertex_color_attribute_name = 'PS3_VertexColor' if material_platform in {'ps3', 'xbox360'} else None
        ps3_external_render_stream = bool(getattr(model, 'ps3_external_render_stream', False))

        if model.strips:
            _slot_by_group, representative_strip_indices = self._build_material_slot_map(model)
            for material_slot, strip_index in enumerate(representative_strip_indices):
                strip = model.strips[strip_index]
                tpage_flags = _decode_tpage_flags(strip.tpageid)
                tpage_flags = self._apply_console_cull_single_sided_rule(tpage_flags, material_platform)
                texture_id = int(tpage_flags['texture_id'])
                if material_platform == 'psp':
                    psp_texture_id = int(getattr(strip, 'psp_texture_id', -1))
                    if psp_texture_id >= 0:
                        texture_id = psp_texture_id
                elif material_platform in {'ps3', 'xbox360'}:
                    ps3_diffuse_texture_id = int(getattr(strip, 'ps3_diffuse_texture_id', -1))
                    if ps3_diffuse_texture_id >= 0:
                        texture_id = ps3_diffuse_texture_id
                elif material_platform == 'underworld':
                    tr8_diffuse_texture_id = self._valid_texture_id(int(getattr(strip, 'tr8_diffuse_texture_id', -1)))
                    if tr8_diffuse_texture_id >= 0:
                        texture_id = tr8_diffuse_texture_id
                    tpage_flags = _decode_tpage_flags(0)
                    tpage_flags['texture_id'] = max(0, int(texture_id))
                elif material_platform == 'pc_nextgen':
                    pc_nextgen_diffuse_texture_id = self._valid_texture_id(int(getattr(strip, 'pc_nextgen_diffuse_texture_id', -1)))
                    if pc_nextgen_diffuse_texture_id >= 0:
                        texture_id = pc_nextgen_diffuse_texture_id
                    tpage_flags = _decode_tpage_flags(0)
                    tpage_flags['texture_id'] = max(0, int(texture_id))

                material_name_index = int(material_slot)
                if material_platform == 'underworld':
                    try:
                        ps2_material_index = int(getattr(strip, 'tr8_ps2_material_index', -1))
                    except Exception:
                        ps2_material_index = -1
                    if ps2_material_index >= 0:
                        material_name_index = ps2_material_index
                material_name = f"Material_{int(material_name_index)}_{int(texture_id) & 0xFFFFFFFF}"
                material = bpy.data.materials.new(name=material_name)

                texture_path = None
                image = None
                if import_texture_images:
                    texture_path = self._find_texture_path(texture_id)
                    image = self._load_packed_image_from_pcd(texture_id, platform_hint=getattr(model, 'uv_format', None))

                normal_texture_id = -1
                specular_texture_id = -1
                normal_texture_path = None
                specular_texture_path = None
                normal_image = None
                specular_image = None
                ps2_stage_overlay_image = None
                pc_nextgen_layer_images = []
                if material_platform == 'pc_nextgen':
                    layer_texture_ids = [int(value) for value in (getattr(strip, 'pc_nextgen_layer_texture_ids', []) or [])]
                    layer_enabled = [int(value) for value in (getattr(strip, 'pc_nextgen_layer_enabled', []) or [])]
                    for layer_index, layer_texture_id in enumerate(layer_texture_ids):
                        layer_image = None
                        is_layer_enabled = layer_enabled[layer_index] if layer_index < len(layer_enabled) else 0
                        valid_layer_texture_id = self._valid_texture_id(layer_texture_id)
                        if is_layer_enabled and valid_layer_texture_id >= 0 and import_texture_images:
                            layer_image = self._load_packed_image_from_pcd(valid_layer_texture_id, platform_hint=getattr(model, 'uv_format', None))
                            if layer_image is not None:
                                try:
                                    layer_image['trlau_texture_id'] = int(valid_layer_texture_id) & 0x1FFF
                                except Exception:
                                    pass
                        pc_nextgen_layer_images.append(layer_image)
                    if pc_nextgen_layer_images:
                        if pc_nextgen_layer_images[0] is not None:
                            image = pc_nextgen_layer_images[0]
                        if len(pc_nextgen_layer_images) > 1 and pc_nextgen_layer_images[1] is not None:
                            normal_image = pc_nextgen_layer_images[1]
                            normal_texture_id = self._valid_texture_id(int(getattr(strip, 'pc_nextgen_normal_texture_id', -1)))
                        if len(pc_nextgen_layer_images) > 2 and pc_nextgen_layer_images[2] is not None:
                            specular_image = pc_nextgen_layer_images[2]
                            specular_texture_id = self._valid_texture_id(int(getattr(strip, 'pc_nextgen_specular_texture_id', -1)))
                if material_platform == 'underworld':
                    normal_texture_id = self._valid_texture_id(int(getattr(strip, 'tr8_normal_texture_id', -1)))
                    if normal_texture_id >= 0 and import_texture_images:
                        normal_texture_path = self._find_texture_path(normal_texture_id)
                        normal_image = self._load_packed_image_from_pcd(normal_texture_id, platform_hint=getattr(model, 'uv_format', None))
                        if normal_image is not None:
                            try:
                                normal_image['trlau_tr8_texture_role'] = 'normal'
                            except Exception:
                                pass
                    ps2_stage_blend_mode = str(getattr(strip, 'tr8_ps2_stage_blend_mode', '') or '')
                    if ps2_stage_blend_mode == 'secondary_base_primary_alpha' and import_texture_images:
                        try:
                            stage_ids = [int(value) for value in (getattr(strip, 'tr8_texture_stage_ids', []) or [])]
                        except Exception:
                            stage_ids = []
                        if len(stage_ids) >= 2:
                            overlay_texture_id = self._valid_texture_id(stage_ids[1])
                            if overlay_texture_id >= 0 and overlay_texture_id != texture_id:
                                ps2_stage_overlay_image = self._load_packed_image_from_pcd(overlay_texture_id, platform_hint=getattr(model, 'uv_format', None))
                                if ps2_stage_overlay_image is not None:
                                    try:
                                        ps2_stage_overlay_image['trlau_tr8_texture_role'] = 'ps2_alpha_overlay'
                                    except Exception:
                                        pass
                elif material_platform in {'ps3', 'xbox360'}:
                    if ps3_external_render_stream:
                        normal_texture_id = self._valid_texture_id(int(getattr(strip, 'ps3_normal_texture_id', -1)))
                        if normal_texture_id < 0:
                            normal_texture_id = self._valid_texture_id(int(getattr(strip, 'ps3_normal_candidate_texture_id', -1)))
                    else:
                        normal_texture_id = -1
                    specular_texture_id = self._valid_texture_id(int(getattr(strip, 'ps3_specular_texture_id', -1)))
                    if normal_texture_id >= 0 and import_texture_images:
                        normal_texture_path = self._find_texture_path(normal_texture_id)
                        normal_image = self._load_packed_image_from_pcd(normal_texture_id, platform_hint=getattr(model, 'uv_format', None))
                        if normal_image is not None:
                            try:
                                normal_image['trlau_ps3_texture_role'] = 'normal'
                                normal_image['trlau_ps3_treat_as_height_map'] = False
                            except Exception:
                                pass
                    if specular_texture_id >= 0 and import_texture_images:
                        specular_texture_path = self._find_texture_path(specular_texture_id)
                        specular_image = self._load_packed_image_from_pcd(specular_texture_id, platform_hint=getattr(model, 'uv_format', None))
                        if specular_image is not None:
                            try:
                                specular_image['trlau_ps3_texture_role'] = 'specular'
                            except Exception:
                                pass

                self._set_texture_dimension_properties(material, None)
                self._set_tpage_material_properties(material, tpage_flags)
                env_mapping = self._set_strip_material_metadata(
                    material,
                    strip,
                    source_strip_index=int(strip_index),
                    material_slot=int(material_slot),
                    material_platform=material_platform,
                    texture_id=texture_id,
                    texture_path=texture_path,
                    use_vertex_colors=use_vertex_colors,
                    normal_texture_id=normal_texture_id,
                    specular_texture_id=specular_texture_id,
                    normal_texture_path=normal_texture_path,
                    specular_texture_path=specular_texture_path,
                    ps3_external_render_stream=ps3_external_render_stream,
                )
                if material_platform == 'pc_nextgen':
                    self._setup_pc_nextgen_material_nodes(
                        material,
                        pc_nextgen_layer_images,
                        tpage_flags,
                        use_vertex_colors=use_vertex_colors,
                        vertex_color_attribute_name=vertex_color_attribute_name,
                    )
                    obj.data.materials.append(material)
                    continue
                self._setup_material_nodes(
                    material,
                    image,
                    tpage_flags,
                    use_vertex_colors=use_vertex_colors,
                    reflective=env_mapping in {64, 128},
                    reflection_mode='both' if env_mapping == 128 else 'env',
                    normal_image=normal_image,
                    specular_image=specular_image,
                    ps2_stage_overlay_image=ps2_stage_overlay_image,
                    vertex_color_attribute_name=vertex_color_attribute_name,
                )
                obj.data.materials.append(material)
            return

    def _apply_material_assignments(self, mesh, material_indices: List[int]) -> None:
        if not mesh.polygons:
            return

        for polygon, material_index in zip(mesh.polygons, material_indices):
            polygon.material_index = material_index

    def _resolve_reflective_strip_indices(self, model: ModelData, reflective_vertex_indices: List[int]) -> List[int]:
        reflective_vertex_set = {int(index) for index in reflective_vertex_indices if int(index) >= 0}
        if not reflective_vertex_set or not model.strips:
            return []

        slot_by_group, _representative_strip_indices = self._build_material_slot_map(model)
        reflective_strip_indices: List[int] = []
        for strip_index, strip in enumerate(model.strips):
            try:
                strip_indices = [int(index) for index in strip.indices]
            except Exception:
                strip_indices = []
            if any(index in reflective_vertex_set for index in strip_indices):
                reflective_strip_indices.append(slot_by_group.get(self._material_group_key(strip, strip_index), strip_index))

        reflective_strip_indices = sorted(set(reflective_strip_indices))
        logger.debug(
            'Resolved %d reflective material groups from %d global vertex indices',
            len(reflective_strip_indices),
            len(reflective_vertex_set),
        )
        return reflective_strip_indices

    def _apply_reflective_material_assignments(
        self,
        obj,
        model: ModelData,
        env_reflective_vertex_indices: List[int],
        eye_reflective_vertex_indices: List[int],
    ) -> None:
        if not obj.data.polygons or not obj.data.materials:
            return

        env_strip_indices = self._resolve_reflective_strip_indices(model, env_reflective_vertex_indices)
        eye_ref_strip_indices = self._resolve_reflective_strip_indices(model, eye_reflective_vertex_indices)
        if not env_strip_indices and not eye_ref_strip_indices:
            logger.debug('No reflective strips matched built materials')
            return

        mesh = obj.data
        env_strip_set = set(env_strip_indices)
        eye_ref_strip_set = set(eye_ref_strip_indices)
        reflective_strip_set = env_strip_set | eye_ref_strip_set

        for base_material_index in sorted(reflective_strip_set):
            if base_material_index < 0 or base_material_index >= len(mesh.materials):
                continue
            if base_material_index in reflective_strip_set and base_material_index in eye_ref_strip_set:
                reflection_mode = 'both'
            else:
                reflection_mode = 'eye_ref' if base_material_index in eye_ref_strip_set else 'env'
            self._apply_reflective_material_to_material(mesh.materials[base_material_index], base_material_index, reflection_mode=reflection_mode)

        reflective_polygon_count = 0
        for polygon in mesh.polygons:
            if int(polygon.material_index) in reflective_strip_set:
                reflective_polygon_count += 1


    def build(self, model: ModelData):
        object_name = self.object_name or os.path.splitext(os.path.basename(self.filepath))[0]
        previous_active_uv_format = self._active_uv_format
        self._active_uv_format = str(getattr(model, 'uv_format', '') or '').lower()
        mesh = bpy.data.meshes.new(object_name)
        obj = bpy.data.objects.new(object_name, mesh)
        self.collection.objects.link(obj)

        verts = [self._final_vertex_position(vertex, model) for vertex in model.vertices]
        valid_faces, material_indices, _source_face_indices = self._collect_faces(model, len(verts))
        valid_faces = self._orient_psp_faces_to_vertex_normals(valid_faces, verts, model)
        mesh.from_pydata(verts, [], valid_faces)
        mesh.update(calc_edges=True)

        self._ensure_strip_materials(obj, model)
        self._apply_material_assignments(mesh, material_indices)
        self._apply_reflective_material_assignments(
            obj,
            model,
            list(getattr(model, 'env_mapped_face_indices', [])),
            list(getattr(model, 'eye_ref_env_mapped_face_indices', [])),
        )
        self._apply_uvs(mesh, model)
        self._apply_normals(mesh, model)
        self._apply_vertex_colors(mesh, model)
        self._apply_ps3_vertex_color_raw_attributes(mesh, model)
        self._apply_mface_metadata(mesh, model)

        self._cleanup_import_debug_metadata(obj)
        self._active_uv_format = previous_active_uv_format
        return obj
