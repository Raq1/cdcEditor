from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Dict, List, Optional, Tuple


Vector4 = Tuple[float, float, float, float]
Vector3Short = Tuple[int, int, int]
Vector3Byte = Tuple[int, int, int]
Face = Tuple[int, int, int]


@dataclass(slots=True)
class RelocationEntry:
    type: int
    section_index_or_type: int
    type_specific: int
    offset: int


@dataclass(slots=True)
class SectionInfo:
    magic: bytes
    size: int
    section_type: int
    skip_flags: int
    version_id: int
    packed_data: int
    has_debug_info: int
    resource_type: int
    num_relocations: int
    section_id: int
    spec_mask: int
    info_size: int
    relocations: List[RelocationEntry] = field(default_factory=list)
    relocations_by_offset: Dict[int, RelocationEntry] = field(default_factory=dict)


@dataclass(slots=True)
class Segment:
    index: int
    min_v: Vector4
    max_v: Vector4
    pivot: Vector4
    flags: int
    first_vertex: int
    last_vertex: int
    parent: int
    hinfo: int


@dataclass(slots=True)
class VirtSegment:
    virt_index: int
    min_v: Vector4
    max_v: Vector4
    pivot: Vector4
    flags: int
    first_vertex: int
    last_vertex: int
    index: int
    weight_index: int
    weight: float


@dataclass(slots=True)
class MVertex:
    index: int
    position_raw: Vector3Short
    normal_raw: Vector3Byte
    segment: int
    uv_raw: tuple[int, int]
    # Optional already-decoded UVs. PSP models can have primitive-local
    # UV transforms, so raw 8-bit S/T is not always enough.
    uv_decoded: Optional[Tuple[float, float]] = None
    gc_transform_id: int = -1
    gc_bind_segment: int = -1
    gc_primary_segment: int = -1
    gc_secondary_segment: int = -1
    gc_secondary_weight: float = 0.0
    psp_color_rgba: Optional[Tuple[int, int, int, int]] = None
    ps3_color_rgba: Optional[Tuple[int, int, int, int]] = None
    # TR8/Underworld vertices can carry explicit four-bone skinning.  Older
    # LAU formats keep single/virtual segment skinning and leave this unset.
    skin_weights: Optional[List[Tuple[int, float]]] = None


@dataclass(slots=True)
class MFace:
    index: int
    v0: int
    v1: int
    v2: int
    same_vert_bits: int

    @property
    def same_vertex_count_0(self) -> int:
        return self.same_vert_bits & 0x1F

    @property
    def same_vertex_count_1(self) -> int:
        return (self.same_vert_bits >> 5) & 0x1F

    @property
    def same_vertex_count_2(self) -> int:
        return (self.same_vert_bits >> 10) & 0x1F

    @property
    def same_vertex_flag(self) -> int:
        return (self.same_vert_bits >> 15) & 0x1


@dataclass(slots=True)
class TextureStrip:
    offset: int
    vertex_count: int
    draw_group: int
    tpageid: int
    sort_push: float
    scroll_offset: float
    env_mapping: int = 0
    next_texture: int = -1
    indices: List[int] = field(default_factory=list)
    bone_ids: List[int] = field(default_factory=list)
    material_group: int = -1
    source_file: str = ''
    scroll_speeds: List[float] = field(default_factory=list)
    ps2_tpageid_raw: int = -1
    psp_mode_word: int = 0
    psp_vertex_mode: int = 0
    psp_render_flags: int = 0
    psp_blend: int = 0
    psp_vertex_format_flags: int = 0
    psp_texture_id: int = -1
    ps3_diffuse_tpageid_raw: int = -1
    ps3_normal_tpageid_raw: int = -1
    ps3_normal_candidate_tpageid_raw: int = -1
    ps3_specular_tpageid_raw: int = -1
    ps3_diffuse_texture_id: int = -1
    ps3_normal_texture_id: int = -1
    ps3_normal_candidate_texture_id: int = -1
    ps3_specular_texture_id: int = -1
    ps3_texture_stage_ids: List[int] = field(default_factory=list)
    ps3_texture_stage_tpageids: List[int] = field(default_factory=list)
    tr8_material_resource_id: int = -1
    tr8_material_file: str = ''
    tr8_shader_ids: List[int] = field(default_factory=list)
    tr8_shader_files: List[str] = field(default_factory=list)
    tr8_shader_strings: List[str] = field(default_factory=list)
    tr8_texture_stage_ids: List[int] = field(default_factory=list)
    tr8_texture_stage_types: List[int] = field(default_factory=list)
    tr8_texture_stage_slots: List[int] = field(default_factory=list)
    tr8_diffuse_texture_id: int = -1
    tr8_normal_texture_id: int = -1
    tr8_ao_texture_id: int = -1
    tr8_detail_texture_id: int = -1
    tr8_detail_ao_texture_id: int = -1
    tr8_mask_texture_id: int = -1
    tr8_reflection_texture_id: int = -1
    tr8_ps2_run_flags: int = 0
    tr8_ps2_alpha_blend: bool = False
    tr8_ps2_material_index: int = -1
    tr8_ps2_stage_blend_mode: str = ''
    tr8_batch_index: int = -1
    tr8_vertex_format_offset: int = -1
    tr8_palette_source_index: int = -1
    tr8_geometry_source_index: int = -1
    tr8_double_sided: bool = False
    tr8_double_wound_pair_count: int = 0
    # TR7 PC next-generation render-data material state. These fields mirror
    # the PCD9 PCMaterialData/PCMaterialDataLayer records instead of reusing
    # the old PC tpage bitfield material system.
    pc_nextgen_material_id: int = -1
    pc_nextgen_material_record_offset: int = 0
    pc_nextgen_asset_id_hi: int = 0
    pc_nextgen_asset_id_lo: int = 0
    pc_nextgen_asset_id_padding_hex: str = ''
    pc_nextgen_material_record_hex: str = ''
    pc_nextgen_blend_mode: int = 0
    pc_nextgen_combiner_type: int = 0
    pc_nextgen_material_flags: int = 0
    pc_nextgen_opacity: float = 1.0
    pc_nextgen_poly_flags: int = 0
    pc_nextgen_uv_auto_scroll_speed: int = 0
    pc_nextgen_sort_bias: float = 0.0
    pc_nextgen_detail_range_mul: float = 0.0
    pc_nextgen_detail_scale: float = 0.0
    pc_nextgen_parallax_scale: float = 0.0
    pc_nextgen_parallax_offset: float = 0.0
    pc_nextgen_specular_power: float = 0.0
    pc_nextgen_specular_shift0: float = 0.0
    pc_nextgen_specular_shift1: float = 0.0
    pc_nextgen_rim_light_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pc_nextgen_rim_light_intensity: float = 0.0
    pc_nextgen_water_blend_bias: float = 0.0
    pc_nextgen_water_blend_exponent: float = 0.0
    pc_nextgen_water_deep_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    pc_nextgen_local_num_pixmaps: int = 0
    pc_nextgen_layer_texture_ids: List[int] = field(default_factory=list)
    pc_nextgen_layer_texture_indices: List[int] = field(default_factory=list)
    pc_nextgen_layer_enabled: List[int] = field(default_factory=list)
    pc_nextgen_layer_colors: List[Tuple[float, float, float, float]] = field(default_factory=list)
    pc_nextgen_layer_texcoord_sources: List[int] = field(default_factory=list)
    pc_nextgen_layer_modifiers: List[int] = field(default_factory=list)
    pc_nextgen_layer_param_ids: List[int] = field(default_factory=list)
    pc_nextgen_layer_constants: List[Tuple[float, float, float, float]] = field(default_factory=list)
    pc_nextgen_layer_num_textures: List[int] = field(default_factory=list)
    pc_nextgen_shader_indices: List[int] = field(default_factory=list)
    # Render-data shader table section IDs referenced by relocation rows.
    # Material shader_indices are indices into this table, not raw section IDs.
    pc_nextgen_shader_table_ids: List[int] = field(default_factory=list)
    pc_nextgen_special_material_flag: bool = False
    pc_nextgen_diffuse_texture_id: int = -1
    pc_nextgen_normal_texture_id: int = -1
    pc_nextgen_specular_texture_id: int = -1
    # Number of triangles whose index data requested reversed winding during import.
    pc_nextgen_reversed_winding_count: int = 0
    pc_nextgen_double_sided: bool = False
    pc_nextgen_double_wound_pair_count: int = 0
    pc_nextgen_index_data_offset_extra: int = 0
    pc_nextgen_index_data_validation_score: str = ''
    scroll_num_tiles: int = 0
    scroll_tile: int = 0
    scroll_entry_count: int = 0

    @property
    def scroll_speed(self) -> Optional[float]:
        return self.scroll_speeds[0] if self.scroll_speeds else None

    @property
    def has_scroll_animation(self) -> bool:
        return bool(self.scroll_speeds)




@dataclass(slots=True)
class BoneMirrorEntry:
    bone1: int
    bone2: int
    count: int


@dataclass(slots=True)
class HMarker:
    global_index: int
    owner_segment: int
    bone: int
    index: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]


@dataclass(slots=True)
class HSphere:
    global_index: int
    owner_segment: int
    flags: int
    id: int
    rank: int
    radius: int
    x: int
    y: int
    z: int
    radius_sq: int
    mass: int
    buoyancy_factor: int
    explosion_factor: int
    material_type: int
    pad: int
    damage: int



@dataclass(slots=True)
class HBox:
    global_index: int
    owner_segment: int
    flags: int
    id: int
    rank: int
    mass: int
    buoyancy_factor: int
    explosion_factor: int
    material_type: int
    pad: int
    damage: int
    dimensions: Vector4
    position: Vector4
    quaternion: Vector4

@dataclass(slots=True)
class HCapsule:
    global_index: int
    owner_segment: int
    flags: int
    id: int
    rank: int
    radius: int
    length: int
    mass: int
    buoyancy_factor: int
    explosion_factor: int
    material_type: int
    pad: int
    damage: int
    position: Vector4
    quaternion: Vector4
    start: Tuple[float, float, float]
    end: Tuple[float, float, float]


class ModelTargetFlags(IntFlag):
    CombatTargetFlag = 0x1
    IncidentalTargetFlag = 0x2
    GrappleTugTargetFlag = 0x4
    GrappleSwingTargetFlag = 0x8
    InteractTargetFlag = 0x10
    MagGunTargetFlag = 0x20
    ForcedIncidentalTargetFlag = 0x40
    AimAssistTargetFlag = 0x80


@dataclass(slots=True)
class Target:
    global_index: int
    segment: int
    flags: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    unique_id: int

@dataclass(slots=True)
class ModelData:
    version: int
    model_scale: Vector4
    segments: List[Segment]
    virt_segments: List[VirtSegment]
    vertices: List[MVertex]
    faces: List[MFace]
    strips: List[TextureStrip]
    vertex_colors: Optional[List[Tuple[int, int, int, int]]] = None
    env_mapped_face_indices: List[int] = field(default_factory=list)
    eye_ref_env_mapped_face_indices: List[int] = field(default_factory=list)
    hmarkers: List[HMarker] = field(default_factory=list)
    hspheres: List[HSphere] = field(default_factory=list)
    hboxes: List[HBox] = field(default_factory=list)
    hcapsules: List[HCapsule] = field(default_factory=list)
    targets: List[Target] = field(default_factory=list)
    markups: List[object] = field(default_factory=list)
    max_rad: float = 0.0
    max_rad_sq: float = 0.0
    bone_mirror_entries: List[BoneMirrorEntry] = field(default_factory=list)
    cdc_render_data_id: int = 0
    uv_format: str = 'pc'
    # Import-side face-orientation correction count for PC next-gen/Underworld-
    # style streams whose face data can request the opposite normal direction.
    pc_nextgen_vertex_normal_oriented_face_count: int = 0
    ps3_external_render_stream: bool = False
