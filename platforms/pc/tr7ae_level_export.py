from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.game_utils import normalize_game_value

_LEVEL_VERSION = 0x04C204BB
_UV_SCALE = 0.00024414062
_RELOCATION_POINTER = 0
_ROOT_HEADER_SIZE = 0x130
_TERRAIN_HEADER_SIZE = 0x6C
_GROUP_ENTRY_SIZE = 176
_COLLISION_HEADER_SIZE = 78
_MATERIAL_ENTRY_SIZE = 20
_STRIP_HEADER_SIZE = 44
_OCTREE_NODE_HEADER_SIZE = 24
_OCTREE_NODE_SIZE = _OCTREE_NODE_HEADER_SIZE
_TERRAIN_VERTEX_SIZE = 20
_BGOBJECT_ENTRY_SIZE = 96
_BGINSTANCE_ENTRY_SIZE = 0xF0
_BGOBJECT_STRIP_HEADER_SIZE = 28
_BGOBJECT_VERTEX_SIZE = 12
_STREAM_UNIT_PORTAL_SIZE = 160
_SECTION_VERSION = 14
_GENERIC_INTRO_STRUCT = '<ffBBHHhfi'
_REWARD_INTRO_STRUCT = '<hhi'
_WATER_VOLUME_INTRO_STRUCT = '<fhhffhhfhH'
_ROPE_RENDER_PARAMS_STRUCT = '<fffIBBBB'
_ROPE_OBJ_INTRO_PREFIX_STRUCT = '<BBHBBHBBHH2xiIff'
_GENERIC_INTRO_SIZE = struct.calcsize(_GENERIC_INTRO_STRUCT)
_MARKUP_FLAG_PERCH = 262144
_MARKUP_FLAG_WATER = 2147483648
_MARKUP_BBOX_FLAGS = _MARKUP_FLAG_PERCH | _MARKUP_FLAG_WATER
_MULTI_SPLINE_SIZE = 0x64
_SPLINE_HEADER_SIZE = 0x08
_SPLINE_KEY_SIZE = 0x40
_RSPLINE_KEY_SIZE = 0x20
_MULTI_SPLINE_MAX_KEYS = 4096

_SFX_MARKER_SIZE = 260
_SFX_POINTER_ARRAY_CAPACITY = 16
_SFX_EVENT_SOUND_BASE_SIZE = 24
_SFX_PERIODIC_SOUND_BASE_SIZE = 28
_SFX_ONESHOT_SOUND_BASE_SIZE = 16
_SFX_STREAM_SOUND_SIZE = 76
_SFX_PERIMETER_SIZE = 28


_SPLINE_CAMERA_LEVEL_DATA_FIELDS = (
    ('CameraFollowSmooth', 'i16'),
    ('CameraFollowTilt', 'i16'),
    ('CameraFollowDistance', 'i16'),
    ('CameraFollowRotation', 'i16'),
    ('CameraZOffset', 'i16'),
    ('CameraCutAngle', 'i16'),
    ('CameraCombatMinDistance', 'i16'),
    ('CameraCombatDistance', 'i16'),
    ('CameraCombatZOffset', 'i16'),
    ('CameraCombatLockedOut', 'i8'),
    ('CameraDisableLookaroundFlag', 'i8'),
    ('CameraSpline0', 'i16'),
    ('CameraSpline1', 'i16'),
    ('CameraSpline2', 'i16'),
    ('CameraAlternateSpline0', 'i16'),
    ('CameraAlternateSpline1', 'i16'),
    ('CameraAlternateSpline2', 'i16'),
    ('CameraMode', 'i16'),
    ('CameraLeadAmount', 'i16'),
    ('CameraSpline0Width', 'i16'),
    ('CameraSpline1Width', 'i16'),
    ('CameraSpline0WidthZ', 'i16'),
    ('CameraSpline1WidthZ', 'i16'),
    ('CameraZoomDist', 'i16'),
    ('CameraVelocity', 'i16'),
    ('CameraDampening', 'i32'),
    ('CameraTargetVelocity', 'i16'),
    ('CameraCombatFraming', 'i16'),
    ('CameraTargetDampening', 'i32'),
    ('CameraCutFlag', 'i8'),
    ('CameraFollowCamOverrideEnabled', 'i8'),
    ('CameraCrossfade', 'i16'),
    ('CameraInterestInstId', 'i16'),
    ('CameraInterestTune', 'i16'),
    ('CameraSwitchToFollowDist', 'i16'),
    ('CameraFollowVerticalBias', 'i16'),
    ('CameraFollowHighTiltDistance', 'i16'),
    ('CameraFollowHighTiltAngle', 'i16'),
    ('CameraFollowMedTiltDistance', 'i16'),
    ('CameraFollowMedTiltAngle', 'i16'),
    ('CameraFollowZeroTiltDistance', 'i16'),
    ('CameraFollowZeroTiltAngle', 'i16'),
    ('CameraFollowLowTiltDistance', 'i16'),
    ('CameraFollowLowTiltAngle', 'i16'),
)
_SPLINE_CAMERA_LEVEL_DATA_SIZE = 88
_SIGNAL_STRUCT_SIZE = 208
_SIGNAL_MESH_HEADER_SIZE = 78
_SIGNAL_VERTEX_TYPE_FLOAT = 1
_SIGNAL_VERTEX_FLOAT_STRIDE = 16
_SIGNAL_FACE_SIZE = 10
_ATTACK_WAVE_RUNTIME_SIZE = 804
_ATTACK_WAVE_RUNTIME_NEXT_POINTER_OFFSETS = {
    'enableAttackWaves': 24,
    'disableAttackWaves': 28,
    'killAttackWaves': 32,
}
_SIGNAL_FLAG_BITS = (
    ('exitPortal', 6, 0x0001),
    ('entryPortal', 7, 0x0001),
    ('autoStream', 8, 0x0001),
    ('ResetFSFXToDefaultOut', 9, 0x0001),
    ('ResetFSFXToDefaultIn', 10, 0x0001),
    ('MaterialOnly', 11, 0x0001),
    ('SpectralOnly', 12, 0x0001),
    ('ObjectHitSignal', 13, 0x0001),
    ('ResetSlideAngle', 14, 0x0001),
    ('triggered', 15, 0x0001),
)


@dataclass(slots=True)
class ExportVertex:
    position: Tuple[float, float, float]
    uv: Tuple[float, float] = (0.0, 0.0)
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)


@dataclass(slots=True)
class ExportStrip:
    name: str
    tpageid: int = 0
    flags: int = 0
    vertices: List[ExportVertex] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    vertex_base_offset: int = 0
    scroll_enabled: bool = False
    scroll_speed: float = 0.0
    scroll_num_tiles: int = 0
    scroll_tile: int = 0
    sort_vertex: Tuple[int, int, int] = (0, 0, 0)
    sort_push: int = 0
    scroll_offset: float = 0.0
    raw_count: int = -1
    is_terminator: bool = False
    env_mapping: bool = False
    eye_ref_env_mapping: bool = False


@dataclass(slots=True)
class ExportCollisionFace:
    i0: int
    i1: int
    i2: int
    adjacency_flags: int = 0
    collision_flags: int = 3
    client_flags: int = 0
    material_type: int = 0


@dataclass(slots=True)
class ExportCollisionKDNode:
    neg_offset: float
    pos_offset: float
    index: int
    axis: int
    num_faces: int = 0


@dataclass(slots=True)
class ExportSignalFace:
    i0: int
    i1: int
    i2: int
    signal_index: int
    adjacency_flags: int = 0
    collision_flags: int = 255


@dataclass(slots=True)
class ExportSignalMesh:
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[ExportSignalFace] = field(default_factory=list)
    vertex_type: int = _SIGNAL_VERTEX_TYPE_FLOAT
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kd_nodes: List[ExportCollisionKDNode] = field(default_factory=list)
    kd_max_depth: int = 0


@dataclass(slots=True)
class ExportSignal:
    index: int
    signal_id: int
    data: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExportCollision:
    position: Tuple[float, float, float]
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[ExportCollisionFace] = field(default_factory=list)
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kd_nodes: List[ExportCollisionKDNode] = field(default_factory=list)
    kd_max_depth: int = 0


@dataclass(slots=True)
class ExportTerrainGroup:
    index: int
    position: Tuple[float, float, float]
    global_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    flags: int = 0
    terrain_id: int = 0
    unique_id: int = 0
    spline_id: int = 0
    texture_morph_value: float = 0.0
    texture_morph_step: float = 0.0
    group_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    strips: List[ExportStrip] = field(default_factory=list)
    collision: Optional[ExportCollision] = None
    sorted_material_indices: List[int] = field(default_factory=list)


@dataclass(slots=True)
class ExportBGObject:
    index: int
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    flags: int = 0
    stride: int = _BGOBJECT_ENTRY_SIZE
    cdc_render_data_id: int = 0
    vertices: List[ExportVertex] = field(default_factory=list)
    strips: List[ExportStrip] = field(default_factory=list)
    raw_vertex_data: bytes = b''
    raw_color_data: bytes = b''
    raw_color_slot_count: int = 1
    env_mapped_vertices: List[int] = field(default_factory=list)
    eye_ref_env_mapped_vertices: List[int] = field(default_factory=list)


@dataclass(slots=True)
class ExportBGInstance:
    index: int
    bg_object_index: int
    instance_id: int
    flags: int = 0
    multi_spline_data: Optional[Dict[str, object]] = None
    original_radius: float = 0.0
    radius: float = 0.0
    bg_flags: int = 0
    target_frame: int = 0
    clip_beg: int = 0
    clip_end: int = 0
    link_seg: int = 0
    color_data_index: int = 0
    lod: int = 0
    active_light_bitfield: int = 0
    matrix_rows: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


@dataclass(slots=True)
class ExportIntroData:
    index: int
    object_id: int
    intro_num: int
    unique_id: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, ...] = (0.0, 0.0, 0.0)
    dummy1: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    dummy2: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    scale: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
    start_frame: int = 0
    end_frame: int = 0
    intro_flags: int = 0
    attached_vmo: int = 0
    data: int = 0
    intro_data_type: int = 0
    generic_intro_data: Dict[str, object] = field(default_factory=dict)
    specific_intro_data: Dict[str, object] = field(default_factory=dict)
    multi_spline: int = 0
    multi_spline_data: Optional[Dict[str, object]] = None
    max_radius: float = 0.0


@dataclass(slots=True)
class ExportTerrainLightData:
    index: int
    position: Tuple[float, float, float]
    radius: int = 0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    type: int = 0
    multiplier: int = 0
    hotspot_angle: int = 0
    direction: Tuple[float, float, float] = (0.0, 0.0, -4096.0)
    falloff_angle: int = 0
    light_id: int = 0


@dataclass(slots=True)
class ExportMarkupData:
    index: int
    flags: int = 0
    animated_segment: int = 0
    intro_id: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    polyline: List[Tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class ExportSFXSound:
    index: int
    kind: str
    sound_ids: List[int] = field(default_factory=list)
    properties: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ExportSFXPerimeter:
    index: int
    radius: int = 0
    n_fired: int = 0
    invader_offset: int = 0
    action: int = 0
    breached: int = 0
    on_exit: int = 0
    always: int = 0
    event_index: int = -1
    periodic_index: int = -1
    varnum: int = 0
    value: int = 0


@dataclass(slots=True)
class ExportSFXMarker:
    index: int
    position: Tuple[float, float, float]
    unique_id: int
    plane: int = 0
    spline_id: int = 0
    sound_instance_offset: int = 0
    event_sounds: List[ExportSFXSound] = field(default_factory=list)
    periodic_sounds: List[ExportSFXSound] = field(default_factory=list)
    one_shot_sounds: List[ExportSFXSound] = field(default_factory=list)
    stream_sounds: List[ExportSFXSound] = field(default_factory=list)
    perimeter_actions: List[ExportSFXPerimeter] = field(default_factory=list)


@dataclass(slots=True)
class ExportPassthroughSection:
    name: str
    data: bytes = b''
    section_type: int = 0
    section_id: int = 0
    skip_flags: int = 0
    version_id: int = 0
    has_debug_info: int = 0
    resource_type: int = 0
    spec_mask: int = 0xFFFFFFFF
    relocations: List[Tuple[int, int, int]] = field(default_factory=list)
    pointer_relocations: List[Tuple[int, str, int]] = field(default_factory=list)


@dataclass(slots=True)
class _CollisionKDNode:
    neg_offset: float
    pos_offset: float
    index: int
    axis: int
    num_faces: int = 0


@dataclass(slots=True)
class ExportLevel:
    drm_name: str
    terrain_groups: List[ExportTerrainGroup] = field(default_factory=list)
    signals: List[ExportSignal] = field(default_factory=list)
    signal_mesh: Optional[ExportSignalMesh] = None
    bg_objects: List[ExportBGObject] = field(default_factory=list)
    bg_instances: List[ExportBGInstance] = field(default_factory=list)
    intro_data: List[ExportIntroData] = field(default_factory=list)
    terrain_lights: List[ExportTerrainLightData] = field(default_factory=list)
    markups: List[ExportMarkupData] = field(default_factory=list)
    sfx_markers: List[ExportSFXMarker] = field(default_factory=list)
    terrain_light_grid_cells: List[Tuple[int, Tuple[int, ...]]] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    unit_data: Optional[Dict[str, object]] = None
    admd_data: Optional[Dict[str, object]] = None
    scene_center_offset: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    passthrough_sections: List[ExportPassthroughSection] = field(default_factory=list)
    game: str = 'legend'


@dataclass(slots=True)
class _OctreeStripInfo:
    material_index: int
    strip: ExportStrip
    centroid: Tuple[float, float, float]
    points: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class _OctreeNode:
    center: Tuple[float, float, float]
    radius: float
    strip_infos: List[_OctreeStripInfo] = field(default_factory=list)
    children: List['_OctreeNode'] = field(default_factory=list)


@dataclass(slots=True)
class _PendingRelocation:
    source_offset: int
    target_section_name: str
    target_offset: int


class _SectionBuffer:
    def __init__(
        self,
        name: str,
        *,
        section_type: int = 0,
        section_id: int = 0,
        skip_flags: int = 0,
        version_id: int = 0,
        has_debug_info: int = 0,
        resource_type: int = 0,
        spec_mask: int = 0xFFFFFFFF,
        raw_relocations: Optional[List[Tuple[int, int, int]]] = None,
    ):
        self.name = name
        self.section_type = int(section_type)
        self.section_id = int(section_id)
        self.skip_flags = int(skip_flags)
        self.version_id = int(version_id)
        self.has_debug_info = int(has_debug_info)
        self.resource_type = int(resource_type)
        self.spec_mask = int(spec_mask)
        self.data = bytearray()
        self.relocations: List[_PendingRelocation] = []
        self.raw_relocations: Optional[List[Tuple[int, int, int]]] = list(raw_relocations) if raw_relocations is not None else None

    def align(self, alignment: int) -> int:
        alignment = max(1, int(alignment))
        padding = (-len(self.data)) % alignment
        if padding:
            self.data.extend(b'\x00' * padding)
        return len(self.data)

    def reserve(self, size: int, *, alignment: int = 1) -> int:
        self.align(alignment)
        offset = len(self.data)
        self.data.extend(b'\x00' * int(size))
        return offset

    def append(self, payload: bytes, *, alignment: int = 1) -> int:
        self.align(alignment)
        offset = len(self.data)
        self.data.extend(payload)
        return offset

    def pack_at(self, offset: int, fmt: str, *values) -> None:
        struct.pack_into(fmt, self.data, int(offset), *values)

    def write_pointer_at(self, offset: int, target_section_name: str, target_offset: int) -> None:
        self.pack_at(offset, '<I', int(target_offset))
        self.relocations.append(_PendingRelocation(int(offset), str(target_section_name), int(target_offset)))


class TRLevelDRMWriter:
    def write(self, level: ExportLevel, filepath: str) -> None:
        sections = self._build_sections(level)
        payload = self._build_drm(sections)
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def _build_drm(self, sections: Sequence[_SectionBuffer]) -> bytes:
        section_index_by_name = {section.name: index for index, section in enumerate(sections)}

        payload = bytearray()
        payload.extend(struct.pack('<iI', _SECTION_VERSION, len(sections)))
        for section in sections:
            raw_relocations = list(section.raw_relocations or [])
            if raw_relocations:
                raw_relocations.sort(key=lambda entry: int(entry[2]))
            else:
                section.relocations.sort(key=lambda relocation: int(relocation.source_offset))
            relocation_count = len(raw_relocations) if raw_relocations else len(section.relocations)
            packed_data = (int(section.has_debug_info) & 0x1) | ((int(section.resource_type) & 0x7F) << 1) | (relocation_count << 8)
            payload.extend(struct.pack(
                '<IBBHIII',
                len(section.data),
                section.section_type,
                int(section.skip_flags) & 0xFF,
                int(section.version_id) & 0xFFFF,
                packed_data,
                section.section_id,
                int(section.spec_mask) & 0xFFFFFFFF,
            ))

        for section in sections:
            raw_relocations = list(section.raw_relocations or [])
            if raw_relocations:
                raw_relocations.sort(key=lambda entry: int(entry[2]))
                for type_and_section_info, type_specific, offset in raw_relocations:
                    payload.extend(struct.pack('<HhI', int(type_and_section_info) & 0xFFFF, int(type_specific), int(offset)))
            else:
                for relocation in section.relocations:
                    target_index = section_index_by_name[relocation.target_section_name]
                    payload.extend(struct.pack('<HhI', (target_index << 3) | _RELOCATION_POINTER, 0, relocation.source_offset))
            payload.extend(section.data)

        return bytes(payload)

    def _build_sections(self, level: ExportLevel) -> List[_SectionBuffer]:
        level_game = self._normalize_level_game(getattr(level, 'game', 'legend'))
        include_psp_data = level_game == 'anniversary'
        omitted_object_id = 72 if include_psp_data else 171
        collision_support_section_type = 10 if include_psp_data else 0

        root = _SectionBuffer('root', section_type=0)
        terrain = _SectionBuffer('terrain', section_type=0)
        vertices = _SectionBuffer('vertices', section_type=0)
        terrain_nulls = _SectionBuffer('terrain_nulls', section_type=0)
        groups_section = _SectionBuffer('groups', section_type=0)
        bgobjects_section = _SectionBuffer('bgobjects', section_type=0)
        bginstances_section = _SectionBuffer('bginstances', section_type=0)
        collisions = _SectionBuffer('collisions', section_type=0)
        collision_roots = _SectionBuffer('collision_roots', section_type=collision_support_section_type)
        collision_vertices = _SectionBuffer('collision_vertices', section_type=collision_support_section_type)
        collision_faces = _SectionBuffer('collision_faces', section_type=collision_support_section_type)

        unit_meta = dict(level.unit_data or {}) if level.unit_data else None
        admd_meta = dict(level.admd_data or {}) if level.admd_data else None
        admd_section = _SectionBuffer('admd', section_type=7, section_id=int(admd_meta.get('id', 0) if admd_meta else 0)) if admd_meta else None
        unit_section = _SectionBuffer('unit', section_type=7, section_id=int(unit_meta.get('id', 0) if unit_meta else 0)) if unit_meta else None

        terrain_nulls.append(b'\x00\x00\x00\x00\x00\x00\x00\x00', alignment=4)

        root_offset = root.reserve(_ROOT_HEADER_SIZE, alignment=16)

        groups = sorted(level.terrain_groups, key=lambda item: int(item.index))
        signal_entries = self._prepare_signal_entries(getattr(level, 'signals', []) or [])
        signal_mesh = getattr(level, 'signal_mesh', None)

        terrain_vertex_count = 0
        for group in groups:
            for strip in group.strips:
                strip.vertex_base_offset = terrain_vertex_count
                terrain_vertex_count += len(strip.vertices)

        material_list_offsets: Dict[int, Optional[int]] = {}
        sorted_material_offsets: Dict[int, Optional[int]] = {}
        octree_offsets: Dict[int, Optional[int]] = {}
        scroll_info_offsets: Dict[int, Optional[int]] = {}
        animated_info_offsets: Dict[int, Optional[int]] = {}
        collision_offsets: Dict[int, Optional[int]] = {}
        prepared_octrees: Dict[int, object] = {}
        strip_entry_offsets_by_group: Dict[int, Dict[int, List[int]]] = {}
        ordered_strip_entries_by_group: Dict[int, List[Tuple[int, int]]] = {}

        for group in groups:
            prepared_octrees[int(group.index)] = self._prepare_group_octree_tree(terrain, group)
            (
                octree_offsets[int(group.index)],
                strip_entry_offsets_by_group[int(group.index)],
                ordered_strip_entries_by_group[int(group.index)],
            ) = self._finalize_group_octree_tree(
                terrain,
                terrain.name,
                prepared_octrees[int(group.index)],
            )
            sorted_material_offsets[int(group.index)] = self._write_sorted_material_indices(terrain, group)
            material_list_offsets[int(group.index)] = self._write_material_list(
                terrain,
                group,
                strip_entry_offsets_by_group[int(group.index)],
            )
            scroll_info_offsets[int(group.index)] = self._write_group_scroll_info(
                terrain,
                group,
                strip_entry_offsets_by_group[int(group.index)],
                ordered_strip_entries_by_group[int(group.index)],
            )
            animated_info_offsets[int(group.index)] = self._write_zero_count_block(terrain)

        terrain_offset = terrain.reserve(_TERRAIN_HEADER_SIZE, alignment=16)
        group_table_offset = groups_section.reserve(max(1, len(groups)) * _GROUP_ENTRY_SIZE, alignment=16)
        group_entry_offsets: Dict[int, int] = {}
        for group_order, group in enumerate(groups):
            group_entry_offsets[int(group.index)] = group_table_offset + (group_order * _GROUP_ENTRY_SIZE)

        signal_group_offset: Optional[int] = None
        signal_mesh_offset: Optional[int] = None
        if signal_entries and signal_mesh is not None and getattr(signal_mesh, 'vertices', None) and getattr(signal_mesh, 'faces', None):
            signal_group_offset = groups_section.reserve(_GROUP_ENTRY_SIZE, alignment=16)
            signal_mesh_offset = self._write_signal_mesh(groups_section, signal_mesh, group_entry_offset=signal_group_offset)
            self._write_signal_terrain_group_entry(
                groups_section,
                signal_group_offset,
                root_section=root.name,
                signal_mesh_offset=signal_mesh_offset,
                unique_id=max((int(getattr(group, 'unique_id', 0) or 0) for group in groups), default=0) + 1,
            )

        for group in groups:
            collision_offsets[int(group.index)] = self._write_collision(
                collisions,
                collision_roots,
                collision_vertices,
                collision_faces,
                group.collision,
                groups_section_name=groups_section.name,
                group_entry_offset=group_entry_offsets[int(group.index)],
            )

        vertex_buffer_offset = self._write_vertex_buffer(vertices, groups)

        bg_objects = sorted(list(getattr(level, 'bg_objects', []) or []), key=lambda item: int(getattr(item, 'index', 0)))
        bg_instances = sorted(list(getattr(level, 'bg_instances', []) or []), key=lambda item: int(getattr(item, 'index', 0)))
        bg_object_offsets = self._write_bg_objects(bgobjects_section, bg_objects, bg_instances) if bg_objects else {}
        bg_instance_list_offset = self._write_bg_instances(bginstances_section, bg_instances, bgobjects_section.name, bg_object_offsets, bg_objects) if bg_instances and bg_object_offsets else None
        bg_object_list_offset = min(bg_object_offsets.values()) if bg_object_offsets else None

        world_name = str(level.metadata.get('worldName', '') or level.drm_name or '')
        player_name = str(level.metadata.get('playerName', '') or '')
        player_name_offset = root.append(player_name.encode('utf-8', errors='ignore') + b'\x00', alignment=1) if player_name else None
        world_name_offset = root.append(world_name.encode('utf-8', errors='ignore') + b'\x00', alignment=1) if world_name else None

        signal_list_offset, signal_id_list_offset, signal_offsets, start_going_into_signal_offset, start_going_out_of_signal_offset = self._write_signals(root, signal_entries)
        stream_portals_offset, stream_portal_count = self._write_stream_unit_portals(root, (getattr(level, 'metadata', {}) or {}).get('streamUnitPortals', []))

        intro_entries = sorted(level.intro_data, key=lambda item: int(item.index))
        intro_data_section, intro_data_pointers = self._write_intro_data_section(intro_entries)
        if intro_entries:
            root.align(16)
        intro_list_offset = self._write_intro_list(root, intro_entries, intro_data_pointers) if intro_entries else None

        object_name_list_values = self._derive_object_name_list_from_intro_data(
            intro_entries,
            level.metadata.get('playerObjectID', -1),
            omitted_object_ids={omitted_object_id},
        )
        object_name_list_offset = self._write_object_name_list(root, object_name_list_values) if object_name_list_values else None

        terrain_light_entries = sorted(list(getattr(level, 'terrain_lights', []) or []), key=lambda item: int(getattr(item, 'index', 0)))
        terrain_lights_offset = self._write_terrain_lights(root, terrain_light_entries) if terrain_light_entries else None
        terrain_light_grids_offset = self._write_terrain_light_grids(root, terrain_light_entries) if terrain_light_entries else None
        markup_entries = sorted(list(getattr(level, 'markups', []) or []), key=lambda item: int(getattr(item, 'index', 0)))
        markups_offset = self._write_markups(root, markup_entries, level_game=level_game) if markup_entries else None
        sfx_marker_entries = sorted(list(getattr(level, 'sfx_markers', []) or []), key=lambda item: int(getattr(item, 'index', 0)))
        sfx_marker_list_offset = self._write_sfx_markers(root, sfx_marker_entries) if sfx_marker_entries else None

        unit_offset = self._write_unit_data_section(unit_section, unit_meta, include_psp_data=include_psp_data) if unit_section is not None and unit_meta is not None else None
        admd_offset = self._write_admd_section(admd_section, admd_meta) if admd_section is not None and admd_meta is not None else None

        self._write_root_header(
            root,
            root_offset,
            terrain_section=terrain.name,
            terrain_offset=terrain_offset,
            strings_section=root.name,
            world_name_offset=world_name_offset,
            player_name_offset=player_name_offset,
            intro_section=root.name,
            intro_list_offset=intro_list_offset,
            intro_count=len(intro_entries),
            object_name_list_section=root.name,
            object_name_list_offset=object_name_list_offset,
            terrain_lights_section=root.name,
            terrain_lights_offset=terrain_lights_offset,
            terrain_lights_count=len(terrain_light_entries),
            markups_section=root.name,
            markups_offset=markups_offset,
            markups_count=len(markup_entries),
            sfx_marker_section=root.name,
            sfx_marker_list_offset=sfx_marker_list_offset,
            sfx_marker_count=len(sfx_marker_entries),
            terrain_light_grids_section=root.name,
            terrain_light_grids_offset=terrain_light_grids_offset,
            unit_section=unit_section.name if unit_section is not None else None,
            unit_offset=unit_offset,
            admd_section=admd_section.name if admd_section is not None else None,
            admd_offset=admd_offset,
            signal_section=root.name,
            signal_list_offset=signal_list_offset,
            signal_id_list_offset=signal_id_list_offset,
            start_going_into_signal_offset=start_going_into_signal_offset,
            start_going_out_of_signal_offset=start_going_out_of_signal_offset,
            metadata=level.metadata,
            scene_center_offset=level.scene_center_offset,
        )
        self._write_terrain_header(
            terrain,
            terrain_offset,
            intro_section=root.name,
            intro_list_offset=intro_list_offset,
            intro_count=len(intro_entries),
            groups_section=groups_section.name,
            group_table_offset=group_table_offset,
            terrain_nulls_section=terrain_nulls.name,
            vertices_section=vertices.name,
            vertex_buffer_offset=vertex_buffer_offset,
            group_count=len(groups),
            stream_portals_section=root.name if stream_portals_offset is not None else None,
            stream_portals_offset=stream_portals_offset,
            stream_portal_count=stream_portal_count,
            signal_group_section=groups_section.name if signal_group_offset is not None else None,
            signal_group_offset=signal_group_offset,
            signal_section=root.name if signal_list_offset is not None else None,
            signal_list_offset=signal_list_offset,
            terrain_vertex_count=terrain_vertex_count,
            bg_objects_section=bgobjects_section.name,
            bg_object_list_offset=bg_object_list_offset,
            bg_object_count=len(bg_objects),
            bg_instances_section=bginstances_section.name,
            bg_instance_list_offset=bg_instance_list_offset,
            bg_instance_count=len(bg_instances) if bg_instance_list_offset is not None else 0,
        )

        for group_order, group in enumerate(groups):
            group_offset = group_entry_offsets[int(group.index)]
            self._write_group_entry(
                groups_section,
                group_offset,
                group=group,
                root_section=root.name,
                terrain_support_section=terrain.name,
                material_list_offset=material_list_offsets.get(int(group.index)),
                sorted_material_offset=sorted_material_offsets.get(int(group.index)),
                octree_offset=octree_offsets.get(int(group.index)),
                scroll_info_offset=scroll_info_offsets.get(int(group.index)),
                animated_info_offset=animated_info_offsets.get(int(group.index)),
                collisions_section=collisions.name,
                collision_offset=collision_offsets.get(int(group.index), 0),
            )

        sections: List[_SectionBuffer] = [root, terrain, terrain_nulls]
        if intro_data_section is not None:
            sections.append(intro_data_section)
        if unit_section is not None:
            sections.append(unit_section)
        if bg_object_offsets:
            sections.append(bgobjects_section)
        if bg_instance_list_offset is not None:
            sections.append(bginstances_section)
        sections.extend([vertices, groups_section])
        if any(value is not None for value in collision_offsets.values()):
            sections.extend([collision_roots, collision_faces, collision_vertices, collisions])
        if admd_section is not None:
            sections.append(admd_section)
        for passthrough in list(level.passthrough_sections or []):
            pointer_relocations = list(getattr(passthrough, 'pointer_relocations', []) or [])
            passthrough_section = _SectionBuffer(
                passthrough.name,
                section_type=int(passthrough.section_type),
                section_id=int(passthrough.section_id),
                skip_flags=int(passthrough.skip_flags),
                version_id=int(passthrough.version_id),
                has_debug_info=int(passthrough.has_debug_info),
                resource_type=int(passthrough.resource_type),
                spec_mask=int(passthrough.spec_mask),
                raw_relocations=None if pointer_relocations else list(passthrough.relocations or []),
            )
            passthrough_section.data = bytearray(bytes(passthrough.data or b''))
            for source_offset, target_section_name, target_offset in pointer_relocations:
                passthrough_section.write_pointer_at(
                    int(source_offset),
                    str(target_section_name or passthrough_section.name),
                    int(target_offset),
                )
            sections.append(passthrough_section)
        return sections

    @staticmethod
    def _prepare_signal_entries(signals: Sequence[ExportSignal]) -> List[ExportSignal]:
        entries: Dict[int, ExportSignal] = {}
        for signal in list(signals or []):
            try:
                index = int(getattr(signal, 'index', -1))
            except Exception:
                continue
            if index < 0 or index > 0xFFFF:
                continue
            entries[index] = signal
        return [entries[index] for index in sorted(entries)]

    @staticmethod
    def _dict_from_signal_data(signal: ExportSignal) -> Dict[str, object]:
        data = getattr(signal, 'data', None)
        return dict(data or {}) if isinstance(data, dict) else {}

    def _compose_signal_flags(self, data: Dict[str, object]) -> int:
        try:
            flags = int(data.get('raw_flags', 0) or 0) & 0xFFFF
        except Exception:
            flags = 0
        for field_name, bit_shift, bit_mask in _SIGNAL_FLAG_BITS:
            mask = (int(bit_mask) << int(bit_shift)) & 0xFFFF
            if bool(data.get(field_name, False)):
                flags |= mask
            else:
                flags &= (~mask) & 0xFFFF
        return flags & 0xFFFF

    def _write_fixed_c_string_at(self, section: _SectionBuffer, offset: int, value: object, size: int) -> None:
        raw = str(value or '').encode('utf-8', errors='ignore')[:max(0, int(size) - 1)]
        payload = raw + (b'\x00' * (int(size) - len(raw)))
        section.data[int(offset):int(offset) + int(size)] = payload[:int(size)]

    @staticmethod
    def _dict_from_maybe_json(value: object) -> Dict[str, object]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                decoded = json.loads(value)
            except Exception:
                return {}
            return dict(decoded) if isinstance(decoded, dict) else {}
        return {}

    @staticmethod
    def _float_list(value: object, size: int, default: Optional[Sequence[float]] = None) -> List[float]:
        source = list(default or [])[:int(size)]
        if len(source) < int(size):
            source.extend([0.0] * (int(size) - len(source)))
        if isinstance(value, (list, tuple)):
            for index in range(min(int(size), len(value))):
                try:
                    item = float(value[index])
                except Exception:
                    item = source[index]
                source[index] = item if math.isfinite(item) else 0.0
        return source[:int(size)]

    @staticmethod
    def _state_from_dict(value: object) -> Tuple[float, int, int]:
        data = dict(value or {}) if isinstance(value, dict) else {}
        try:
            t = float(data.get('t', 0.0) or 0.0)
        except Exception:
            t = 0.0
        if not math.isfinite(t):
            t = 0.0
        return (t, int(data.get('currKey', data.get('curr_key', 0)) or 0), int(data.get('pad', 0) or 0))

    @staticmethod
    def _pack_vec4(section: _SectionBuffer, offset: int, value: object, default_w: float = 0.0) -> None:
        vals = TRLevelDRMWriter._float_list(value, 4, (0.0, 0.0, 0.0, float(default_w)))
        section.pack_at(offset, '<4f', vals[0], vals[1], vals[2], vals[3])

    def _write_spline_data(self, section: _SectionBuffer, values: object) -> Optional[int]:
        data = self._dict_from_maybe_json(values)
        if not data:
            return None
        raw_keys = data.get('keys', [])
        keys = list(raw_keys or []) if isinstance(raw_keys, list) else []
        num_keys = max(0, min(_MULTI_SPLINE_MAX_KEYS, len(keys)))
        offset = section.reserve(_SPLINE_HEADER_SIZE + (num_keys * _SPLINE_KEY_SIZE), alignment=16)
        try:
            count = float(data.get('count', float(num_keys)) or 0.0)
        except Exception:
            count = float(num_keys)
        if not math.isfinite(count):
            count = 0.0
        section.pack_at(offset, '<HBBf', self._u16(num_keys), self._u8(data.get('type', 0)), self._u8(data.get('flags', 0)), count)
        cursor = offset + _SPLINE_HEADER_SIZE
        for key in keys[:num_keys]:
            key_data = dict(key or {}) if isinstance(key, dict) else {}
            self._pack_vec4(section, cursor + 0x00, key_data.get('point'), 0.0)
            self._pack_vec4(section, cursor + 0x10, key_data.get('d1'), 0.0)
            self._pack_vec4(section, cursor + 0x20, key_data.get('d2'), 0.0)
            section.pack_at(
                cursor + 0x30,
                '<4f',
                float(key_data.get('t0', 0.0) or 0.0),
                float(key_data.get('tf', 0.0) or 0.0),
                float(key_data.get('count', 0.0) or 0.0),
                float(key_data.get('invCount', key_data.get('inv_count', 0.0)) or 0.0),
            )
            cursor += _SPLINE_KEY_SIZE
        return offset

    def _write_rspline_data(self, section: _SectionBuffer, values: object) -> Optional[int]:
        data = self._dict_from_maybe_json(values)
        if not data:
            return None
        raw_keys = data.get('keys', [])
        keys = list(raw_keys or []) if isinstance(raw_keys, list) else []
        num_keys = max(0, min(_MULTI_SPLINE_MAX_KEYS, len(keys)))
        offset = section.reserve(_SPLINE_HEADER_SIZE + (num_keys * _RSPLINE_KEY_SIZE), alignment=16)
        try:
            count = float(data.get('count', float(num_keys)) or 0.0)
        except Exception:
            count = float(num_keys)
        if not math.isfinite(count):
            count = 0.0
        section.pack_at(offset, '<HBBf', self._u16(num_keys), self._u8(data.get('type', 0)), self._u8(data.get('flags', 0)), count)
        cursor = offset + _SPLINE_HEADER_SIZE
        for key in keys[:num_keys]:
            key_data = dict(key or {}) if isinstance(key, dict) else {}
            self._pack_vec4(section, cursor + 0x00, key_data.get('q'), 0.0)
            section.pack_at(
                cursor + 0x10,
                '<4f',
                float(key_data.get('t0', 0.0) or 0.0),
                float(key_data.get('tf', 0.0) or 0.0),
                float(key_data.get('count', 0.0) or 0.0),
                float(key_data.get('invCount', key_data.get('inv_count', 0.0)) or 0.0),
            )
            cursor += _RSPLINE_KEY_SIZE
        return offset

    def _write_multi_spline_data(self, section: _SectionBuffer, values: object) -> Optional[int]:
        data = self._dict_from_maybe_json(values)
        if not data:
            return None
        identity = (
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        offset = section.reserve(_MULTI_SPLINE_SIZE, alignment=16)
        matrix = self._float_list(data.get('curRotMatrix'), 16, identity)
        section.pack_at(offset + 0x00, '<16f', *matrix)

        positional_offset = self._write_spline_data(section, data.get('positional'))
        rotational_offset = self._write_rspline_data(section, data.get('rotational'))
        scaling_offset = self._write_spline_data(section, data.get('scaling'))

        if positional_offset is not None:
            section.write_pointer_at(offset + 0x40, section.name, positional_offset)
        else:
            section.pack_at(offset + 0x40, '<I', 0)
        if rotational_offset is not None:
            section.write_pointer_at(offset + 0x44, section.name, rotational_offset)
        else:
            section.pack_at(offset + 0x44, '<I', 0)
        if scaling_offset is not None:
            section.write_pointer_at(offset + 0x48, section.name, scaling_offset)
        else:
            section.pack_at(offset + 0x48, '<I', 0)

        section.pack_at(offset + 0x4C, '<fHH', *self._state_from_dict(data.get('curPositional')))
        section.pack_at(offset + 0x54, '<fHH', *self._state_from_dict(data.get('curRotational')))
        section.pack_at(offset + 0x5C, '<fHH', *self._state_from_dict(data.get('curScaling')))
        return offset

    def _write_spline_camera_level_data_at(self, section: _SectionBuffer, offset: int, values: object) -> None:
        values_dict = dict(values or {}) if isinstance(values, dict) else {}
        cursor = int(offset)
        for field_name, field_type in _SPLINE_CAMERA_LEVEL_DATA_FIELDS:
            value = values_dict.get(field_name, 0)
            if field_type == 'i8':
                section.pack_at(cursor, '<b', self._i8(value)); cursor += 1
            elif field_type == 'u8':
                section.pack_at(cursor, '<B', self._u8(value)); cursor += 1
            elif field_type == 'u16':
                section.pack_at(cursor, '<H', self._u16(value)); cursor += 2
            elif field_type == 'i32':
                section.pack_at(cursor, '<i', self._i32(value)); cursor += 4
            elif field_type == 'u32':
                section.pack_at(cursor, '<I', int(value or 0) & 0xFFFFFFFF); cursor += 4
            else:
                section.pack_at(cursor, '<h', self._i16(value)); cursor += 2
        consumed = cursor - int(offset)
        if consumed < _SPLINE_CAMERA_LEVEL_DATA_SIZE:
            section.data[cursor:int(offset) + _SPLINE_CAMERA_LEVEL_DATA_SIZE] = b'\x00' * (_SPLINE_CAMERA_LEVEL_DATA_SIZE - consumed)

    def _append_spline_camera_level_data(self, section: _SectionBuffer, values: object) -> Optional[int]:
        if not isinstance(values, dict) or not values:
            return None
        offset = section.reserve(_SPLINE_CAMERA_LEVEL_DATA_SIZE, alignment=4)
        self._write_spline_camera_level_data_at(section, offset, values)
        return offset

    @staticmethod
    def _signal_spline_params_from_data(data: Dict[str, object]) -> Dict[str, object]:
        value = data.get('splineCamParams')
        if isinstance(value, dict):
            return dict(value)
        return {field_name: data.get(f'splineCamParams_{field_name}', 0) for field_name, _field_type in _SPLINE_CAMERA_LEVEL_DATA_FIELDS}

    @staticmethod
    def _sanitize_signal_index(value: object, default: int = -1) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _write_signal_move_pos_limits_at(self, section: _SectionBuffer, offset: int, wave: Dict[str, object]) -> None:
        cursor = int(offset)
        limits = list(wave.get('limits', []) or []) if isinstance(wave.get('limits', []), list) else []
        section.pack_at(cursor, '<i', max(0, min(6, int(wave.get('numLimits', len(limits)) or 0)))); cursor += 4
        for index in range(6):
            item = limits[index] if index < len(limits) and isinstance(limits[index], dict) else {}
            section.pack_at(cursor, '<5f', float(item.get('x', 0.0) or 0.0), float(item.get('y', 0.0) or 0.0), float(item.get('z', 0.0) or 0.0), float(item.get('rad', 0.0) or 0.0), float(item.get('pursueRad', 0.0) or 0.0)); cursor += 20

        box_limits = list(wave.get('boxLimits', []) or []) if isinstance(wave.get('boxLimits', []), list) else []
        section.pack_at(cursor, '<i', max(0, min(6, int(wave.get('numBoxLimits', len(box_limits)) or 0)))); cursor += 4
        for index in range(6):
            item = box_limits[index] if index < len(box_limits) and isinstance(box_limits[index], dict) else {}
            section.pack_at(cursor, '<4f4h', float(item.get('x', 0.0) or 0.0), float(item.get('y', 0.0) or 0.0), float(item.get('z', 0.0) or 0.0), float(item.get('zrot', 0.0) or 0.0), self._i16(item.get('width', 0)), self._i16(item.get('length', 0)), self._i16(item.get('pursueWidth', 0)), self._i16(item.get('pursueLength', 0))); cursor += 24

        plane_limits = list(wave.get('planeLimits', []) or []) if isinstance(wave.get('planeLimits', []), list) else []
        section.pack_at(cursor, '<i', max(0, min(1, int(wave.get('numPlaneLimits', len(plane_limits)) or 0)))); cursor += 4
        for index in range(1):
            item = plane_limits[index] if index < len(plane_limits) and isinstance(plane_limits[index], dict) else {}
            section.pack_at(cursor, '<6f', float(item.get('px', 0.0) or 0.0), float(item.get('py', 0.0) or 0.0), float(item.get('pz', 0.0) or 0.0), float(item.get('nx', 0.0) or 0.0), float(item.get('ny', 0.0) or 0.0), float(item.get('nz', 0.0) or 0.0)); cursor += 24

        run_and_gun = list(wave.get('runAndGunPositions', []) or []) if isinstance(wave.get('runAndGunPositions', []), list) else []
        section.pack_at(cursor, '<i', max(0, min(6, int(wave.get('numRunAndGunPos', len(run_and_gun)) or 0)))); cursor += 4
        marker_pointer_fields: List[Tuple[int, str]] = []
        for index in range(6):
            item = run_and_gun[index] if index < len(run_and_gun) and isinstance(run_and_gun[index], dict) else {}
            section.pack_at(cursor, '<3f3hH', float(item.get('x', 0.0) or 0.0), float(item.get('y', 0.0) or 0.0), float(item.get('z', 0.0) or 0.0), self._i16(item.get('priority', 0)), self._i16(item.get('waittime', 0)), self._i16(item.get('failAction', 0)), 0); cursor += 20
            marker_name = str(item.get('markerName', '') or '')
            marker_pointer_fields.append((cursor, marker_name))
            section.pack_at(cursor, '<I', 0); cursor += 4

        for pointer_offset, marker_name in marker_pointer_fields:
            if not marker_name:
                continue
            marker_offset = section.append(marker_name.encode('utf-8', errors='ignore') + b'\x00', alignment=1)
            section.write_pointer_at(pointer_offset, section.name, marker_offset)

        patrol_positions = list(wave.get('patrolPositions', []) or []) if isinstance(wave.get('patrolPositions', []), list) else []
        section.pack_at(cursor, '<hh', max(0, min(8, int(wave.get('numPatrolPos', len(patrol_positions)) or 0))), self._i16(wave.get('patrolType', 0))); cursor += 4
        for index in range(8):
            item = patrol_positions[index] if index < len(patrol_positions) and isinstance(patrol_positions[index], dict) else {}
            section.pack_at(cursor, '<6f4h', float(item.get('x', 0.0) or 0.0), float(item.get('y', 0.0) or 0.0), float(item.get('z', 0.0) or 0.0), float(item.get('lookx', 0.0) or 0.0), float(item.get('looky', 0.0) or 0.0), float(item.get('lookz', 0.0) or 0.0), self._i16(item.get('waittime', 0)), self._i16(item.get('anim', 0)), self._i16(item.get('mode', 0)), self._i16(item.get('look', 0))); cursor += 32

    def _write_attack_wave_messages_at(self, section: _SectionBuffer, offset: int, wave: Dict[str, object]) -> None:
        messages = list(wave.get('messages', []) or []) if isinstance(wave.get('messages', []), list) else []
        section.pack_at(offset, '<i', max(0, min(6, int(wave.get('numMessages', len(messages)) or 0))))
        cursor = int(offset) + 4
        for index in range(6):
            item = messages[index] if index < len(messages) and isinstance(messages[index], dict) else {}
            section.pack_at(cursor, '<ii', self._i32(item.get('message', 0)), self._i32(item.get('data', 0))); cursor += 8

    def _write_attack_wave_runtime_chain(self, section: _SectionBuffer, chain_name: str, waves: object) -> Optional[int]:
        if not isinstance(waves, list) or not waves:
            return None
        normalized = [dict(wave) for wave in waves if isinstance(wave, dict)]
        if not normalized:
            return None
        offsets = [section.reserve(_ATTACK_WAVE_RUNTIME_SIZE, alignment=16) for _wave in normalized]
        for index, wave in enumerate(normalized):
            offset = offsets[index]
            section.pack_at(offset + 0, '<f', float(wave.get('radiusCheckTimer', 0.0) or 0.0))
            cursor = offset + 4
            for field_name in (
                'flags', 'rtFlags', 'probability', 'numAttackerSpawns', 'numSpawnsThisLoad',
                'currentAttacker', 'triggerRemaining', 'numAttackersPerWave', 'numLinkedAttackWaves',
                'numBlockedAttackWaves', 'numFinishedAttackWaves', 'numAttackers', 'musicRank',
            ):
                section.pack_at(cursor, '<B', self._u8(wave.get(field_name, 0))); cursor += 1
            section.pack_at(cursor, '<b', self._i8(wave.get('pad', 0))); cursor += 1
            section.pack_at(cursor, '<Hh', self._u16(wave.get('radiusCheckMarker', 0)), self._i16(wave.get('radiusCheckRadius', 0)))
            if index + 1 < len(offsets):
                next_field_offset = offset + int(_ATTACK_WAVE_RUNTIME_NEXT_POINTER_OFFSETS.get(chain_name, 24))
                section.write_pointer_at(next_field_offset, section.name, offsets[index + 1])
            self._write_signal_move_pos_limits_at(section, offset + 36, wave)
            self._write_attack_wave_messages_at(section, offset + 748, wave)
            section.pack_at(offset + 800, '<i', self._i32(wave.get('endMarker', 0)))
        return offsets[0]

    @staticmethod
    def _stream_portal_vec4(value: object, default: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> Tuple[float, float, float, float]:
        if isinstance(value, (list, tuple)):
            items = list(value)
        else:
            items = []
        result: List[float] = []
        for index in range(4):
            try:
                result.append(float(items[index]))
            except Exception:
                result.append(float(default[index]))
        return (result[0], result[1], result[2], result[3])

    def _write_stream_unit_portals(self, section: _SectionBuffer, portals: object) -> Tuple[Optional[int], int]:
        if not isinstance(portals, list) or not portals:
            return None, 0
        entries = [entry for entry in portals if isinstance(entry, dict)]
        if not entries:
            return None, 0
        count = len(entries)
        offset = section.reserve(count * _STREAM_UNIT_PORTAL_SIZE, alignment=16)
        for index, portal in enumerate(entries):
            entry_offset = offset + (index * _STREAM_UNIT_PORTAL_SIZE)
            name = str(portal.get('tolevelname', '') or '')[:29]
            encoded_name = name.encode('ascii', errors='ignore')[:29]
            section.data[entry_offset:entry_offset + 30] = encoded_name + (b'\x00' * (30 - len(encoded_name)))
            section.pack_at(entry_offset + 30, '<hhh',
                self._i16(portal.get('toSignalID', 0)),
                self._i16(portal.get('MSignalID', 0)),
                self._i16(portal.get('streamID', 0)),
            )
            section.pack_at(entry_offset + 36, '<I', 0)
            section.pack_at(entry_offset + 40, '<f', float(portal.get('activeDistance', 0.0) or 0.0))
            section.pack_at(entry_offset + 44, '<I', 0)
            cursor = entry_offset + 48
            for vec in (
                self._stream_portal_vec4(portal.get('min')),
                self._stream_portal_vec4(portal.get('max')),
            ):
                section.pack_at(cursor, '<4f', *vec); cursor += 16
            portalquad = portal.get('portalquad', [])
            if not isinstance(portalquad, list):
                portalquad = []
            for quad_index in range(4):
                vec = portalquad[quad_index] if quad_index < len(portalquad) else (0.0, 0.0, 0.0, 0.0)
                section.pack_at(cursor, '<4f', *self._stream_portal_vec4(vec)); cursor += 16
            section.pack_at(cursor, '<4f', *self._stream_portal_vec4(portal.get('normal')))
        return offset, count

    def _write_signals(self, section: _SectionBuffer, signals: Sequence[ExportSignal]) -> Tuple[Optional[int], Optional[int], Dict[int, int], Optional[int], Optional[int]]:
        entries = self._prepare_signal_entries(signals)
        if not entries:
            return None, None, {}, None, None
        max_index = max(int(signal.index) for signal in entries)
        count = max_index + 1
        signal_by_index = {int(signal.index): signal for signal in entries}
        signal_list_offset = section.reserve(count * _SIGNAL_STRUCT_SIZE, alignment=16)
        signal_id_list_offset = section.reserve(count * 2, alignment=2)
        signal_offsets = {index: signal_list_offset + (index * _SIGNAL_STRUCT_SIZE) for index in range(count)}
        start_going_into_signal_offset: Optional[int] = None
        start_going_out_of_signal_offset: Optional[int] = None

        for index in range(count):
            signal = signal_by_index.get(index)
            signal_id = int(getattr(signal, 'signal_id', index) if signal is not None else index)
            section.pack_at(signal_id_list_offset + (index * 2), '<h', self._i16(signal_id))

        for index in range(count):
            signal = signal_by_index.get(index)
            if signal is None:
                continue
            data = self._dict_from_signal_data(signal)
            signal_offset = signal_offsets[index]
            section.pack_at(signal_offset + 0, '<H', self._compose_signal_flags(data))
            spline_ptr_values = data.get('splineCamDataPtr', [])
            if isinstance(spline_ptr_values, list):
                for slot in range(min(3, len(spline_ptr_values))):
                    target_offset = self._append_spline_camera_level_data(section, spline_ptr_values[slot])
                    if target_offset is not None:
                        section.write_pointer_at(signal_offset + 4 + (slot * 4), section.name, target_offset)
            section.pack_at(signal_offset + 16, '<ii', self._i32(data.get('CameraLock', 0)), self._i32(data.get('CameraUnlock', 0)))
            section.pack_at(signal_offset + 24, '<hhHh', self._i16(data.get('CameraSave', 0)), self._i16(data.get('CameraRestore', 0)), self._u16(data.get('CameraShakeScale', 0)), self._i16(data.get('CameraShakeTime', 0)))
            self._write_spline_camera_level_data_at(section, signal_offset + 32, self._signal_spline_params_from_data(data))
            self._write_fixed_c_string_at(section, signal_offset + 120, data.get('streamName', ''), 20)
            section.pack_at(signal_offset + 140, '<7i',
                self._i32(data.get('streamPortalIndex', 0)),
                self._i32(data.get('SetSlideAngle', 0)),
                self._i32(data.get('ReverbType', 0)),
                self._i32(data.get('ReverbVolume', 0)),
                self._i32(data.get('setMusicVar', 0)),
                self._i32(data.get('setMusicVarValue', 0)),
                self._i32(data.get('fatalType', 0)),
            )
            for pointer_name, pointer_offset in (('enableAttackWaves', 168), ('disableAttackWaves', 172), ('killAttackWaves', 176)):
                target_offset = self._write_attack_wave_runtime_chain(section, pointer_name, data.get(pointer_name, []))
                if target_offset is not None:
                    section.write_pointer_at(signal_offset + pointer_offset, section.name, target_offset)
            stack_type = self._i8(data.get('CameraStackType', 0))
            stack_bits = ((self._u8(data.get('CameraStackCamInUse', 0)) & 0x0F) << 4) | (self._u8(data.get('CameraStackData', 0)) & 0x0F)
            section.pack_at(signal_offset + 180, '<bBBB', stack_type, stack_bits, self._u8(data.get('CameraStackSignalFlags0', 0)), self._u8(data.get('CameraStackSignalFlags1', 0)))
            section.pack_at(signal_offset + 184, '<ii', self._i32(data.get('DTPCameraID0', 0)), self._i32(data.get('DTPCameraID1', 0)))
            camera_link_index = self._sanitize_signal_index(data.get('cameraLinkSignalIndex', -1), -1)
            if 0 <= camera_link_index < count:
                section.write_pointer_at(signal_offset + 192, section.name, signal_offsets[camera_link_index])
            section.pack_at(signal_offset + 196, '<iiHH',
                self._i32(data.get('FSFXInDTPID', 0)),
                self._i32(data.get('FSFXExitDTPID', 0)),
                self._u16(data.get('FSFXInActiveDist', 0)),
                self._u16(data.get('FSFXExitActiveDist', 0)),
            )
            if bool(data.get('startGoingIntoWaterSignal', False)) and start_going_into_signal_offset is None:
                start_going_into_signal_offset = signal_offset
            if bool(data.get('startGoingOutOfWaterSignal', False)) and start_going_out_of_signal_offset is None:
                start_going_out_of_signal_offset = signal_offset
        return signal_list_offset, signal_id_list_offset, signal_offsets, start_going_into_signal_offset, start_going_out_of_signal_offset

    def _write_signal_mesh(self, section: _SectionBuffer, signal_mesh: ExportSignalMesh, *, group_entry_offset: int) -> Optional[int]:
        vertices = [tuple(float(component) for component in vertex[:3]) for vertex in list(getattr(signal_mesh, 'vertices', []) or [])]
        faces = [face for face in list(getattr(signal_mesh, 'faces', []) or []) if 0 <= int(face.i0) < len(vertices) and 0 <= int(face.i1) < len(vertices) and 0 <= int(face.i2) < len(vertices)]
        if not vertices or not faces:
            return None
        raw_vertices = [self._blender_to_raw_float_triplet(vertex) for vertex in vertices]
        raw_bbox_min = tuple(min(float(vertex[axis]) for vertex in raw_vertices) for axis in range(3))
        raw_bbox_max = tuple(max(float(vertex[axis]) for vertex in raw_vertices) for axis in range(3))
        mesh_offset = section.reserve(_SIGNAL_MESH_HEADER_SIZE, alignment=16)
        self._pack_raw_vec3f(section, mesh_offset + 0, raw_bbox_min)
        self._pack_raw_vec3f(section, mesh_offset + 16, raw_bbox_max)
        self._pack_padded_vec3f(section, mesh_offset + 32, getattr(signal_mesh, 'position', (0.0, 0.0, 0.0)))

        vertex_list_offset = section.reserve(len(raw_vertices) * _SIGNAL_VERTEX_FLOAT_STRIDE, alignment=16)
        for vertex_index, (raw_x, raw_y, raw_z) in enumerate(raw_vertices):
            section.pack_at(vertex_list_offset + (vertex_index * _SIGNAL_VERTEX_FLOAT_STRIDE), '<4f', float(raw_x), float(raw_y), float(raw_z), 0.0)
        section.write_pointer_at(mesh_offset + 48, section.name, vertex_list_offset)

        kd_nodes = list(getattr(signal_mesh, 'kd_nodes', []) or [])
        ordered_faces = faces
        kd_max_depth = int(getattr(signal_mesh, 'kd_max_depth', 0) or 0)
        if not kd_nodes:
            kd_nodes, ordered_faces, kd_max_depth = self._build_collision_kd_nodes([(int(round(x)), int(round(y)), int(round(z))) for x, y, z in raw_vertices], faces)
        face_list_offset = section.reserve(len(ordered_faces) * _SIGNAL_FACE_SIZE, alignment=16)
        for face_index, face in enumerate(ordered_faces):
            section.pack_at(
                face_list_offset + (face_index * _SIGNAL_FACE_SIZE),
                '<HHHBBH',
                self._u16(face.i0),
                self._u16(face.i1),
                self._u16(face.i2),
                self._u8(getattr(face, 'adjacency_flags', 0)),
                self._u8(getattr(face, 'collision_flags', 255)),
                self._u16(getattr(face, 'signal_index', 0)),
            )
        section.write_pointer_at(mesh_offset + 52, section.name, face_list_offset)

        if kd_nodes:
            root_offset = section.reserve(len(kd_nodes) * 12, alignment=16)
            for node_index, node in enumerate(kd_nodes):
                section.pack_at(
                    root_offset + (node_index * 12),
                    '<ffHBB',
                    float(node.neg_offset),
                    float(node.pos_offset),
                    self._u16(node.index),
                    self._u8(node.axis),
                    self._u8(node.num_faces),
                )
            section.write_pointer_at(mesh_offset + 56, section.name, root_offset)
        else:
            section.pack_at(mesh_offset + 56, '<I', 0)
        section.write_pointer_at(mesh_offset + 60, section.name, int(group_entry_offset))
        section.pack_at(mesh_offset + 64, '<HHHHHHH', _SIGNAL_VERTEX_TYPE_FLOAT, self._u16(len(kd_nodes)), self._u16(len(ordered_faces)), self._u16(len(raw_vertices)), self._u16(kd_max_depth), 0, 0)
        return mesh_offset

    def _write_signal_terrain_group_entry(self, section: _SectionBuffer, group_offset: int, *, root_section: str, signal_mesh_offset: Optional[int], unique_id: int = 0) -> None:
        self._pack_padded_vec3f(section, group_offset + 0, (0.0, 0.0, 0.0))
        self._pack_padded_vec3f(section, group_offset + 16, (0.0, 0.0, 0.0))
        section.pack_at(group_offset + 32, '<4i', 0, -1, int(unique_id), 0)
        section.write_pointer_at(group_offset + 52, root_section, 0)
        if signal_mesh_offset is not None:
            section.write_pointer_at(group_offset + 56, section.name, int(signal_mesh_offset))
        section.pack_at(group_offset + 60, '<ff', 0.0, 0.0)
        self._pack_identity_matrix(section, group_offset + 80)
        self._pack_padded_vec3f(section, group_offset + 160, (0.0, 0.0, 0.0))


    def _write_root_header(
        self,
        section: _SectionBuffer,
        root_offset: int,
        *,
        terrain_section: str,
        terrain_offset: int,
        strings_section: str,
        world_name_offset: int,
        player_name_offset: int,
        intro_section: str,
        intro_list_offset: Optional[int],
        intro_count: int,
        object_name_list_section: str,
        object_name_list_offset: Optional[int],
        terrain_lights_section: str,
        terrain_lights_offset: Optional[int],
        terrain_lights_count: int,
        markups_section: str,
        markups_offset: Optional[int],
        markups_count: int,
        sfx_marker_section: str,
        sfx_marker_list_offset: Optional[int],
        sfx_marker_count: int,
        terrain_light_grids_section: str,
        terrain_light_grids_offset: Optional[int],
        unit_section: Optional[str],
        unit_offset: Optional[int],
        admd_section: Optional[str],
        admd_offset: Optional[int],
        signal_section: Optional[str],
        signal_list_offset: Optional[int],
        signal_id_list_offset: Optional[int],
        start_going_into_signal_offset: Optional[int],
        start_going_out_of_signal_offset: Optional[int],
        metadata: Dict[str, object],
        scene_center_offset: Tuple[float, float, float, float],
    ) -> None:
        meta = dict(metadata or {})
        section.write_pointer_at(root_offset + 0, terrain_section, terrain_offset)
        section.pack_at(root_offset + 4, '<f', float(meta.get('waterZLevel', 0.0)))
        section.pack_at(root_offset + 8, '<BBB', self._u8(meta.get('backColorR', 0)), self._u8(meta.get('backColorG', 0)), self._u8(meta.get('backColorB', 0)))
        section.pack_at(root_offset + 12, '<BBBB', self._u8(meta.get('spectralColorR', 0)), self._u8(meta.get('spectralColorG', 0)), self._u8(meta.get('spectralColorB', 0)), self._u8(meta.get('spectralFXAlways', 0)))
        section.pack_at(root_offset + 16, '<BBBb', self._u8(meta.get('waterColorR', 0)), self._u8(meta.get('waterColorG', 0)), self._u8(meta.get('waterColorB', 0)), self._i8(meta.get('waterBlend', 0)))
        section.pack_at(root_offset + 20, '<f', float(meta.get('farPlane', 0.0)))
        section.pack_at(root_offset + 24, '<f', float(meta.get('fogFar', 0.0)))
        section.pack_at(root_offset + 28, '<f', float(meta.get('fogNear', 0.0)))
        section.pack_at(root_offset + 32, '<f', float(meta.get('spectralFarPlane', 0.0)))
        section.pack_at(root_offset + 36, '<f', float(meta.get('spectralFogFar', 0.0)))
        section.pack_at(root_offset + 40, '<f', float(meta.get('spectralFogNear', 0.0)))
        section.pack_at(root_offset + 44, '<f', float(meta.get('waterFarPlane', 0.0)))
        section.pack_at(root_offset + 48, '<f', float(meta.get('waterFogFar', 0.0)))
        section.pack_at(root_offset + 52, '<f', float(meta.get('waterFogNear', 0.0)))
        section.pack_at(root_offset + 56, '<i', int(meta.get('UnderwaterFXAlpha', 0)))
        section.pack_at(root_offset + 60, '<f', float(meta.get('UnderwaterFXMovement', 0.0)))
        section.pack_at(root_offset + 64, '<f', float(meta.get('UnderwaterFXSpeed', 0.0)))
        section.pack_at(root_offset + 68, '<i', int(meta.get('SpectralFXAlpha', 0)))
        section.pack_at(root_offset + 72, '<f', float(meta.get('SpectralFXIncrease', 0.0)))
        section.pack_at(root_offset + 76, '<f', float(meta.get('SpectralFXCenter', 0.0)))
        section.pack_at(root_offset + 80, '<i', int(meta.get('WaterFXAlpha', 0)))
        section.pack_at(root_offset + 84, '<f', float(meta.get('WaterFXMovement', 0.0)))
        section.pack_at(root_offset + 88, '<f', float(meta.get('WaterFXSpeed', 0.0)))
        section.pack_at(root_offset + 92, '<i', int(meta.get('WaterFSFX', 0)))
        section.pack_at(root_offset + 96, '<i', int(max(0, markups_count)))
        if markups_offset is not None and markups_count > 0:
            section.write_pointer_at(root_offset + 100, markups_section, markups_offset)
        else:
            section.pack_at(root_offset + 100, '<I', 0)
        section.pack_at(root_offset + 104, '<i', 0)
        section.pack_at(root_offset + 108, '<I', 0)
        section.pack_at(root_offset + 112, '<i', int(meta.get('flags', 0)))
        section.pack_at(root_offset + 116, '<i', int(intro_count))
        section.pack_at(root_offset + 140, '<i', int(meta.get('unitFlags', 0)))
        section.pack_at(root_offset + 160, '<I', int(max(0, sfx_marker_count)))
        if sfx_marker_list_offset is not None and sfx_marker_count > 0:
            section.write_pointer_at(root_offset + 164, sfx_marker_section, int(sfx_marker_list_offset))
        else:
            section.pack_at(root_offset + 164, '<I', 0)
        reloc_module_section = str(meta.get('relocModuleSectionName', '') or '')
        if reloc_module_section:
            try:
                reloc_module_offset = int(meta.get('relocModuleTargetOffset', 0) or 0)
            except Exception:
                reloc_module_offset = 0
            section.write_pointer_at(root_offset + 156, reloc_module_section, reloc_module_offset)
        else:
            section.pack_at(root_offset + 156, '<I', 0)
        section.pack_at(root_offset + 168, '<I', _LEVEL_VERSION)
        section.pack_at(root_offset + 172, '<I', int(meta.get('guiID', 0)))
        section.pack_at(root_offset + 180, '<i', int(meta.get('streamUnitID', 0)))
        section.pack_at(root_offset + 212, '<i', int(max(0, terrain_lights_count)))
        if terrain_lights_offset is not None and terrain_lights_count > 0:
            section.write_pointer_at(root_offset + 216, terrain_lights_section, terrain_lights_offset)
        else:
            section.pack_at(root_offset + 216, '<I', 0)
        if terrain_light_grids_offset is not None and terrain_lights_count > 0:
            section.write_pointer_at(root_offset + 220, terrain_light_grids_section, terrain_light_grids_offset)
        else:
            section.pack_at(root_offset + 220, '<I', 0)
        area_dbase_section = str(meta.get('areaDBaseSectionName', '') or '')
        if area_dbase_section:
            try:
                move_offset = int(meta.get('areaDBaseMoveDataOffset', -1))
            except Exception:
                move_offset = -1
            try:
                planner_offset = int(meta.get('areaDBasePlannerDataOffset', -1))
            except Exception:
                planner_offset = -1
            try:
                runtime_area_dbase_offset = int(meta.get('areaDBaseRuntimeObjectOffset', -1))
            except Exception:
                runtime_area_dbase_offset = -1
            section.pack_at(root_offset + 240, '<I', 0)
            if planner_offset < 0:
                try:
                    planner_offset = int(meta.get('areaDBaseHeaderContentOffset', -1))
                except Exception:
                    planner_offset = -1
            if planner_offset >= 0:
                section.write_pointer_at(root_offset + 244, area_dbase_section, planner_offset)
            else:
                section.pack_at(root_offset + 244, '<I', 0)
            if runtime_area_dbase_offset >= 0:
                section.write_pointer_at(root_offset + 248, area_dbase_section, runtime_area_dbase_offset)
            else:
                section.pack_at(root_offset + 248, '<I', 0)
        else:
            section.pack_at(root_offset + 240, '<I', 0)
            section.pack_at(root_offset + 244, '<I', 0)
            section.pack_at(root_offset + 248, '<I', 0)
        if unit_section is not None and unit_offset is not None:
            section.write_pointer_at(root_offset + 252, unit_section, unit_offset)
        else:
            section.pack_at(root_offset + 252, '<I', 0)
        if admd_section is not None and admd_offset is not None:
            section.write_pointer_at(root_offset + 256, admd_section, admd_offset)
        else:
            section.pack_at(root_offset + 256, '<I', 0)
        if intro_list_offset is not None and intro_count > 0:
            section.write_pointer_at(root_offset + 120, intro_section, intro_list_offset)
        if object_name_list_offset is not None:
            section.write_pointer_at(root_offset + 124, object_name_list_section, object_name_list_offset)
        if world_name_offset is not None:
            section.write_pointer_at(root_offset + 128, strings_section, world_name_offset)
        if player_name_offset is not None:
            section.write_pointer_at(root_offset + 232, strings_section, player_name_offset)
        if signal_section is not None and signal_list_offset is not None:
            section.write_pointer_at(root_offset + 144, signal_section, int(signal_list_offset))
        if signal_section is not None and signal_id_list_offset is not None:
            section.write_pointer_at(root_offset + 148, signal_section, int(signal_id_list_offset))
        if signal_section is not None and start_going_into_signal_offset is not None:
            section.write_pointer_at(root_offset + 132, signal_section, int(start_going_into_signal_offset))
        if signal_section is not None and start_going_out_of_signal_offset is not None:
            section.write_pointer_at(root_offset + 136, signal_section, int(start_going_out_of_signal_offset))

        scx, scy, scz = self._blender_to_raw_float_triplet(scene_center_offset[:3])
        section.pack_at(root_offset + 272, '<4f', float(scx), float(scy), float(scz), float(scene_center_offset[3] if len(scene_center_offset) > 3 else 0.0))
        section.pack_at(root_offset + 288, '<h', self._i16(meta.get('playerObjectID', -1)))

    def _write_terrain_header(
        self,
        section: _SectionBuffer,
        terrain_offset: int,
        *,
        intro_section: str,
        intro_list_offset: Optional[int],
        intro_count: int,
        groups_section: str,
        group_table_offset: int,
        terrain_nulls_section: str,
        vertices_section: str,
        vertex_buffer_offset: int,
        group_count: int,
        stream_portals_section: Optional[str],
        stream_portals_offset: Optional[int],
        stream_portal_count: int,
        signal_group_section: Optional[str],
        signal_group_offset: Optional[int],
        signal_section: Optional[str],
        signal_list_offset: Optional[int],
        terrain_vertex_count: int,
        bg_objects_section: str,
        bg_object_list_offset: Optional[int],
        bg_object_count: int,
        bg_instances_section: str,
        bg_instance_list_offset: Optional[int],
        bg_instance_count: int,
    ) -> None:
        section.pack_at(terrain_offset + 4, '<i', int(intro_count))
        if intro_list_offset is not None and intro_count > 0:
            section.write_pointer_at(terrain_offset + 8, intro_section, intro_list_offset)
        section.pack_at(terrain_offset + 12, '<i', int(max(0, stream_portal_count)))
        if stream_portals_section is not None and stream_portals_offset is not None and stream_portal_count > 0:
            section.write_pointer_at(terrain_offset + 16, stream_portals_section, int(stream_portals_offset))
        else:
            section.pack_at(terrain_offset + 16, '<I', 0)
        section.pack_at(terrain_offset + 20, '<i', int(group_count))
        section.write_pointer_at(terrain_offset + 24, groups_section, group_table_offset)
        section.write_pointer_at(terrain_offset + 36, terrain_nulls_section, 4)
        section.pack_at(terrain_offset + 40, '<i', int(max(0, bg_instance_count)))
        if bg_instance_list_offset is not None and bg_instance_count > 0:
            section.write_pointer_at(terrain_offset + 44, bg_instances_section, int(bg_instance_list_offset))
        else:
            section.pack_at(terrain_offset + 44, '<I', 0)
        section.pack_at(terrain_offset + 48, '<i', int(max(0, bg_object_count)))
        if bg_object_list_offset is not None and bg_object_count > 0:
            section.write_pointer_at(terrain_offset + 52, bg_objects_section, int(bg_object_list_offset))
        else:
            section.pack_at(terrain_offset + 52, '<I', 0)
        if signal_group_section is not None and signal_group_offset is not None:
            section.write_pointer_at(terrain_offset + 28, signal_group_section, int(signal_group_offset))
        if signal_section is not None and signal_list_offset is not None:
            section.write_pointer_at(terrain_offset + 32, signal_section, int(signal_list_offset))
        section.pack_at(terrain_offset + 56, '<I', 0)
        section.pack_at(terrain_offset + 60, '<I', 0)
        section.pack_at(terrain_offset + 64, '<I', 0)
        section.write_pointer_at(terrain_offset + 68, vertices_section, vertex_buffer_offset)
        section.write_pointer_at(terrain_offset + 72, terrain_nulls_section, 0)
        section.write_pointer_at(terrain_offset + 76, terrain_nulls_section, 0)
        section.pack_at(terrain_offset + 80, '<I', 0)
        section.pack_at(terrain_offset + 84, '<i', int(terrain_vertex_count))
        section.pack_at(terrain_offset + 88, '<I', 0)
        section.pack_at(terrain_offset + 92, '<i', 0)
        section.pack_at(terrain_offset + 96, '<I', 0)
        section.pack_at(terrain_offset + 100, '<I', 0)

    def _write_group_entry(
        self,
        section: _SectionBuffer,
        group_offset: int,
        *,
        group: ExportTerrainGroup,
        root_section: str,
        terrain_support_section: str,
        material_list_offset: Optional[int],
        sorted_material_offset: Optional[int],
        octree_offset: Optional[int],
        scroll_info_offset: Optional[int],
        animated_info_offset: Optional[int],
        collisions_section: str,
        collision_offset: Optional[int],
    ) -> None:
        self._pack_padded_vec3f(section, group_offset + 0, group.global_offset or group.position)
        self._pack_padded_vec3f(section, group_offset + 16, group.local_offset)
        section.pack_at(group_offset + 32, '<4i', int(group.flags), int(group.terrain_id), int(group.unique_id), int(group.spline_id))
        section.write_pointer_at(group_offset + 52, root_section, 0)
        if collision_offset is not None:
            section.write_pointer_at(group_offset + 56, collisions_section, collision_offset)
        section.pack_at(group_offset + 60, '<f', float(group.texture_morph_value))
        section.pack_at(group_offset + 64, '<f', float(group.texture_morph_step))
        if octree_offset is not None:
            section.write_pointer_at(group_offset + 68, terrain_support_section, octree_offset)
        if scroll_info_offset is not None:
            section.write_pointer_at(group_offset + 72, terrain_support_section, scroll_info_offset)
        if animated_info_offset is not None:
            section.write_pointer_at(group_offset + 76, terrain_support_section, animated_info_offset)
        self._pack_identity_matrix(section, group_offset + 80)
        if material_list_offset is not None:
            section.write_pointer_at(group_offset + 144, terrain_support_section, material_list_offset)
        if sorted_material_offset is not None:
            section.write_pointer_at(group_offset + 148, terrain_support_section, sorted_material_offset)
        self._pack_padded_vec3f(section, group_offset + 160, group.group_origin or group.position)

    def _get_sorted_material_order(self, group: ExportTerrainGroup) -> List[int]:
        """Derive the sorted material table from exported strip content.

        The original import stored this table as a custom property, but it is not
        authored data.  Rebuild a stable order from the strips that will be
        written: animated/scrolling materials first, then by texture page, flags,
        and material index.
        """
        count = max(0, len(group.strips))
        if count <= 0:
            return []
        indices = list(range(count))
        return sorted(
            indices,
            key=lambda material_index: (
                0 if bool(getattr(group.strips[material_index], 'scroll_enabled', False)) else 1,
                int(getattr(group.strips[material_index], 'tpageid', 0) or 0) & 0xFFFFFFFF,
                int(getattr(group.strips[material_index], 'flags', 0) or 0) & 0xFFFFFFFF,
                int(material_index),
            ),
        )

    def _write_sorted_material_indices(
        self,
        section: _SectionBuffer,
        group: ExportTerrainGroup,
    ) -> Optional[int]:
        order = self._get_sorted_material_order(group)
        count = len(order)
        if count <= 0:
            return None
        offset = section.reserve(count * 2, alignment=2)
        for order_index, material_index in enumerate(order):
            section.pack_at(offset + (order_index * 2), '<H', self._u16(material_index))
        return offset

    def _write_material_list(
        self,
        section: _SectionBuffer,
        group: ExportTerrainGroup,
        strip_entry_offsets: Dict[int, List[int]],
    ) -> int:
        count = max(0, len(group.strips))
        block_size = 4 + (count * _MATERIAL_ENTRY_SIZE)
        offset = section.reserve(block_size, alignment=4)
        section.pack_at(offset + 0, '<i', count)

        strip_fixup_offsets: Dict[int, int] = {}
        for material_index in range(count):
            strip_fixup_offsets[int(material_index)] = section.reserve(len(strip_entry_offsets.get(int(material_index), [])) * 4, alignment=4)

        for material_index, strip in enumerate(group.strips):
            entry_offset = offset + 4 + (material_index * _MATERIAL_ENTRY_SIZE)
            section.pack_at(entry_offset + 0, '<I', int(strip.tpageid) & 0xFFFFFFFF)
            section.pack_at(entry_offset + 4, '<I', int(strip.flags) & 0xFFFFFFFF)
            section.pack_at(entry_offset + 8, '<I', int(strip.vertex_base_offset) & 0xFFFFFFFF)
            section.pack_at(entry_offset + 12, '<i', 0)
            section.write_pointer_at(entry_offset + 16, section.name, strip_fixup_offsets[int(material_index)])
        return offset

    def _write_group_scroll_info(
        self,
        section: _SectionBuffer,
        group: ExportTerrainGroup,
        strip_entry_offsets: Dict[int, List[int]],
        ordered_strip_entries: Sequence[Tuple[int, int]],
    ) -> int:
        entries: List[tuple[int, float, int, int, int]] = []
        if ordered_strip_entries:
            for strip_offset, material_index in ordered_strip_entries:
                if material_index < 0 or material_index >= len(group.strips):
                    continue
                strip = group.strips[int(material_index)]
                if not bool(getattr(strip, 'scroll_enabled', False)):
                    continue
                raw_speed = float(getattr(strip, 'scroll_speed', 0.0))
                scroll_num_tiles = int(getattr(strip, 'scroll_num_tiles', 0))
                scroll_tile = int(getattr(strip, 'scroll_tile', 0))
                entries.append((int(strip_offset), raw_speed, scroll_num_tiles, scroll_tile, int(material_index)))
        else:
            for material_index, strip in enumerate(group.strips):
                if not bool(getattr(strip, 'scroll_enabled', False)):
                    continue
                raw_speed = float(getattr(strip, 'scroll_speed', 0.0))
                scroll_num_tiles = int(getattr(strip, 'scroll_num_tiles', 0))
                scroll_tile = int(getattr(strip, 'scroll_tile', 0))
                for strip_offset in strip_entry_offsets.get(int(material_index), []):
                    entries.append((int(strip_offset), raw_speed, scroll_num_tiles, scroll_tile, int(material_index)))

        offset = section.reserve(4 + (len(entries) * 20), alignment=4)
        section.pack_at(offset + 0, '<i', len(entries))
        for entry_index, (strip_offset, raw_speed, scroll_num_tiles, scroll_tile, _material_index) in enumerate(entries):
            entry_offset = offset + 4 + (entry_index * 20)
            next_entry_offset = entry_offset + 20
            section.write_pointer_at(entry_offset + 0, section.name, next_entry_offset)
            section.pack_at(entry_offset + 4, '<f', float(raw_speed))
            section.pack_at(entry_offset + 8, '<i', int(scroll_num_tiles))
            section.pack_at(entry_offset + 12, '<i', int(scroll_tile))
            section.write_pointer_at(entry_offset + 16, section.name, int(strip_offset) + 8)
        return offset


    def _write_zero_count_block(self, section: _SectionBuffer) -> int:
        offset = section.reserve(4, alignment=4)
        section.pack_at(offset + 0, '<i', 0)
        return offset

    def _write_group_strip_chain(self, section: _SectionBuffer, group: ExportTerrainGroup) -> Optional[int]:
        if not group.strips:
            return None

        first_strip_offset: Optional[int] = None
        previous_strip_offset: Optional[int] = None
        for material_index, strip in enumerate(group.strips):
            strip_offset = self._write_strip_chain_entry(section, material_index, strip)
            if first_strip_offset is None:
                first_strip_offset = strip_offset
            if previous_strip_offset is not None:
                section.write_pointer_at(previous_strip_offset + 40, section.name, strip_offset)
            previous_strip_offset = strip_offset
        if previous_strip_offset is not None:
            terminator_offset = self._write_zero_count_block(section)
            section.write_pointer_at(previous_strip_offset + 40, section.name, terminator_offset)
        return first_strip_offset

    @staticmethod
    def _compute_octree_sphere_from_points(points: Sequence[Tuple[float, float, float]], default_center: Sequence[float]) -> tuple[tuple[float, float, float], float]:
        if not points:
            center = (
                float(default_center[0]) if len(default_center) > 0 else 0.0,
                float(default_center[1]) if len(default_center) > 1 else 0.0,
                float(default_center[2]) if len(default_center) > 2 else 0.0,
            )
            return center, 0.0
        min_x = min(point[0] for point in points); max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points); max_y = max(point[1] for point in points)
        min_z = min(point[2] for point in points); max_z = max(point[2] for point in points)
        center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
        radius_sq = 0.0
        for point in points:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dz = point[2] - center[2]
            radius_sq = max(radius_sq, (dx * dx) + (dy * dy) + (dz * dz))
        return center, radius_sq ** 0.5

    @staticmethod
    def _build_octree_strip_infos(group: ExportTerrainGroup) -> List[_OctreeStripInfo]:
        infos: List[_OctreeStripInfo] = []
        for material_index, strip in enumerate(group.strips):
            points = [
                (float(vertex.position[0]), float(vertex.position[1]), float(vertex.position[2]))
                for vertex in strip.vertices
                if len(vertex.position) >= 3
            ]
            center, _radius = TRLevelDRMWriter._compute_octree_sphere_from_points(points, group.position)
            infos.append(_OctreeStripInfo(material_index=material_index, strip=strip, centroid=center, points=points))
        return infos

    def _build_octree_node(self, group: ExportTerrainGroup, strip_infos: List[_OctreeStripInfo], depth: int = 0) -> _OctreeNode:
        all_points: List[Tuple[float, float, float]] = []
        for info in strip_infos:
            all_points.extend(info.points)
        center, radius = self._compute_octree_sphere_from_points(all_points, group.position)
        if len(strip_infos) <= 1 or depth >= 8:
            return _OctreeNode(center=center, radius=radius, strip_infos=list(strip_infos), children=[])

        buckets: Dict[int, List[_OctreeStripInfo]] = {}
        for info in strip_infos:
            octant = 0
            if info.centroid[0] >= center[0]:
                octant |= 1
            if info.centroid[1] >= center[1]:
                octant |= 2
            if info.centroid[2] >= center[2]:
                octant |= 4
            buckets.setdefault(octant, []).append(info)

        if len(buckets) <= 1:
            return _OctreeNode(center=center, radius=radius, strip_infos=list(strip_infos), children=[])

        children = [self._build_octree_node(group, buckets[octant], depth + 1) for octant in sorted(buckets) if buckets[octant]]
        if not children:
            return _OctreeNode(center=center, radius=radius, strip_infos=list(strip_infos), children=[])
        return _OctreeNode(center=center, radius=radius, strip_infos=[], children=children)

    def _write_strip_chain_for_infos(
        self,
        section: _SectionBuffer,
        strip_infos: Sequence[_OctreeStripInfo],
        strip_entry_offsets: Dict[int, List[int]],
        ordered_strip_entries: List[Tuple[int, int]],
    ) -> Optional[int]:
        if not strip_infos:
            return None
        first_strip_offset: Optional[int] = None
        previous_strip_offset: Optional[int] = None
        for info in strip_infos:
            strip_offset = self._write_strip_chain_entry(section, info.material_index, info.strip)
            strip_entry_offsets.setdefault(int(info.material_index), []).append(int(strip_offset))
            ordered_strip_entries.append((int(strip_offset), int(info.material_index)))
            if first_strip_offset is None:
                first_strip_offset = strip_offset
            if previous_strip_offset is not None:
                section.write_pointer_at(previous_strip_offset + 40, section.name, strip_offset)
            previous_strip_offset = strip_offset
        if previous_strip_offset is not None:
            terminator_offset = self._write_zero_count_block(section)
            section.write_pointer_at(previous_strip_offset + 40, section.name, terminator_offset)
        return first_strip_offset

    def _reserve_octree_nodes_recursive(self, section: _SectionBuffer, node: _OctreeNode, node_offsets: Dict[int, int]) -> int:
        child_count = min(8, len(node.children))
        octree_size = _OCTREE_NODE_HEADER_SIZE + (child_count * 4)
        octree_offset = section.reserve(octree_size, alignment=16)
        node_offsets[id(node)] = octree_offset
        for child in node.children[:child_count]:
            self._reserve_octree_nodes_recursive(section, child, node_offsets)
        return octree_offset

    def _write_octree_leaf_strip_chains_recursive(
        self,
        section: _SectionBuffer,
        node: _OctreeNode,
        leaf_strip_offsets: Dict[int, Optional[int]],
        strip_entry_offsets: Dict[int, List[int]],
        ordered_strip_entries: List[Tuple[int, int]],
    ) -> None:
        if node.children:
            for child in node.children[:8]:
                self._write_octree_leaf_strip_chains_recursive(section, child, leaf_strip_offsets, strip_entry_offsets, ordered_strip_entries)
            return
        leaf_strip_offsets[id(node)] = self._write_strip_chain_for_infos(section, node.strip_infos, strip_entry_offsets, ordered_strip_entries)

    def _finalize_octree_nodes_recursive(
        self,
        section: _SectionBuffer,
        strip_section_name: str,
        node: _OctreeNode,
        node_offsets: Dict[int, int],
        leaf_strip_offsets: Dict[int, Optional[int]],
    ) -> None:
        octree_offset = node_offsets[id(node)]
        child_nodes = node.children[:8]
        child_count = len(child_nodes)
        raw_x, raw_y, raw_z = self._blender_to_raw_float_triplet(node.center)
        section.pack_at(octree_offset + 0, '<4f', float(raw_x), float(raw_y), float(raw_z), float(node.radius))
        first_strip_offset = leaf_strip_offsets.get(id(node))
        if first_strip_offset is not None:
            section.write_pointer_at(octree_offset + 16, strip_section_name, first_strip_offset)
        else:
            section.pack_at(octree_offset + 16, '<I', 0)
        section.pack_at(octree_offset + 20, '<i', child_count)
        for child_index, child in enumerate(child_nodes):
            child_offset = node_offsets[id(child)]
            section.write_pointer_at(octree_offset + 24 + (child_index * 4), section.name, child_offset)
            self._finalize_octree_nodes_recursive(section, strip_section_name, child, node_offsets, leaf_strip_offsets)

    def _prepare_group_octree_tree(self, section: _SectionBuffer, group: ExportTerrainGroup) -> Optional[tuple[_OctreeNode, Dict[int, int], int]]:
        strip_infos = self._build_octree_strip_infos(group)
        if not strip_infos:
            return None
        root_node = self._build_octree_node(group, strip_infos)
        node_offsets: Dict[int, int] = {}
        root_offset = self._reserve_octree_nodes_recursive(section, root_node, node_offsets)
        return root_node, node_offsets, root_offset

    def _finalize_group_octree_tree(
        self,
        section: _SectionBuffer,
        strip_section_name: str,
        prepared: Optional[tuple[_OctreeNode, Dict[int, int], int]],
    ) -> tuple[Optional[int], Dict[int, List[int]], List[Tuple[int, int]]]:
        if prepared is None:
            return None, {}, []
        root_node, node_offsets, root_offset = prepared
        leaf_strip_offsets: Dict[int, Optional[int]] = {}
        strip_entry_offsets: Dict[int, List[int]] = {}
        ordered_strip_entries: List[Tuple[int, int]] = []
        self._write_octree_leaf_strip_chains_recursive(section, root_node, leaf_strip_offsets, strip_entry_offsets, ordered_strip_entries)
        self._finalize_octree_nodes_recursive(section, strip_section_name, root_node, node_offsets, leaf_strip_offsets)
        return root_offset, strip_entry_offsets, ordered_strip_entries

    def _write_strip_chain_entry(self, section: _SectionBuffer, material_index: int, strip: ExportStrip) -> int:
        index_count = len(strip.indices)
        offset = section.reserve(_STRIP_HEADER_SIZE + (index_count * 2), alignment=16)
        section.pack_at(offset + 0, '<i', index_count)
        section.pack_at(offset + 4, '<i', -1)
        section.pack_at(offset + 16, '<I', int(strip.flags) & 0xFFFFFFFF)
        section.pack_at(offset + 20, '<i', int(material_index))
        for index_offset, index_value in enumerate(strip.indices):
            section.pack_at(offset + _STRIP_HEADER_SIZE + (index_offset * 2), '<h', self._i16(index_value))
        return offset

    def _write_collision(
        self,
        headers_section: _SectionBuffer,
        roots_section: _SectionBuffer,
        vertices_section: _SectionBuffer,
        faces_section: _SectionBuffer,
        collision: Optional[ExportCollision],
        *,
        groups_section_name: str,
        group_entry_offset: int,
    ) -> Optional[int]:
        if collision is None or not collision.vertices or not collision.faces:
            return None

        header_offset = headers_section.reserve(_COLLISION_HEADER_SIZE, alignment=16)
        raw_vertices = [self._blender_to_raw_int_triplet(vertex) for vertex in collision.vertices]
        if raw_vertices:
            raw_bbox_min = (
                float(min(vertex[0] for vertex in raw_vertices)),
                float(min(vertex[1] for vertex in raw_vertices)),
                float(min(vertex[2] for vertex in raw_vertices)),
            )
            raw_bbox_max = (
                float(max(vertex[0] for vertex in raw_vertices)),
                float(max(vertex[1] for vertex in raw_vertices)),
                float(max(vertex[2] for vertex in raw_vertices)),
            )
        else:
            raw_bbox_min = (0.0, 0.0, 0.0)
            raw_bbox_max = (0.0, 0.0, 0.0)
        self._pack_raw_vec3f(headers_section, header_offset + 0, raw_bbox_min)
        self._pack_raw_vec3f(headers_section, header_offset + 16, raw_bbox_max)
        self._pack_padded_vec3f(headers_section, header_offset + 32, collision.position)

        if getattr(collision, 'kd_nodes', None):
            kd_nodes = [
                _CollisionKDNode(
                    float(node.neg_offset),
                    float(node.pos_offset),
                    int(node.index),
                    int(node.axis),
                    int(node.num_faces),
                )
                for node in list(getattr(collision, 'kd_nodes', []) or [])
            ]
            ordered_faces = list(collision.faces)
            max_depth = int(getattr(collision, 'kd_max_depth', 0) or 0)
            if max_depth <= 0:
                max_depth = 8 if kd_nodes else 0
        else:
            kd_nodes, ordered_faces, max_depth = self._build_collision_kd_nodes(raw_vertices, collision.faces)

        vertex_list_offset = vertices_section.reserve(len(raw_vertices) * 6, alignment=16)
        for vertex_index, (raw_x, raw_y, raw_z) in enumerate(raw_vertices):
            vertices_section.pack_at(vertex_list_offset + (vertex_index * 6), '<hhh', raw_x, raw_y, raw_z)
        headers_section.write_pointer_at(header_offset + 48, vertices_section.name, vertex_list_offset)

        face_list_offset = faces_section.reserve(len(ordered_faces) * 10, alignment=16)
        for face_index, face in enumerate(ordered_faces):
            faces_section.pack_at(
                face_list_offset + (face_index * 10),
                '<HHHBBBB',
                self._u16(face.i0),
                self._u16(face.i1),
                self._u16(face.i2),
                self._u8(face.adjacency_flags),
                self._u8(face.collision_flags),
                self._u8(face.client_flags),
                self._u8(face.material_type),
            )
        headers_section.write_pointer_at(header_offset + 52, faces_section.name, face_list_offset)

        if kd_nodes:
            root_offset = roots_section.reserve(len(kd_nodes) * 12, alignment=16)
            for node_index, node in enumerate(kd_nodes):
                roots_section.pack_at(
                    root_offset + (node_index * 12),
                    '<ffHBB',
                    float(node.neg_offset),
                    float(node.pos_offset),
                    self._u16(node.index),
                    self._u8(node.axis),
                    self._u8(node.num_faces),
                )
            headers_section.write_pointer_at(header_offset + 56, roots_section.name, root_offset)
        else:
            headers_section.pack_at(header_offset + 56, '<I', 0)
        headers_section.write_pointer_at(header_offset + 60, groups_section_name, group_entry_offset)

        headers_section.pack_at(header_offset + 64, '<H', 0)
        headers_section.pack_at(header_offset + 66, '<H', self._u16(len(kd_nodes)))
        headers_section.pack_at(header_offset + 68, '<H', self._u16(len(ordered_faces)))
        headers_section.pack_at(header_offset + 70, '<H', self._u16(len(raw_vertices)))
        headers_section.pack_at(header_offset + 72, '<H', self._u16(max_depth))
        headers_section.pack_at(header_offset + 74, '<H', 0)
        headers_section.pack_at(header_offset + 76, '<H', 0)
        return header_offset

    def _build_collision_kd_nodes(
        self,
        raw_vertices: Sequence[Tuple[int, int, int]],
        faces: Sequence[ExportCollisionFace],
    ) -> tuple[List[_CollisionKDNode], List[ExportCollisionFace], int]:
        if not raw_vertices or not faces:
            return [], [], 0

        leaf_face_limit = 2
        max_leaf_face_count = 0xFF
        max_depth_limit = 63

        def make_face_record(face: ExportCollisionFace) -> tuple[ExportCollisionFace, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float]]:
            vertices = (
                raw_vertices[int(face.i0)],
                raw_vertices[int(face.i1)],
                raw_vertices[int(face.i2)],
            )
            bounds_min = (
                min(int(vertex[0]) for vertex in vertices),
                min(int(vertex[1]) for vertex in vertices),
                min(int(vertex[2]) for vertex in vertices),
            )
            bounds_max = (
                max(int(vertex[0]) for vertex in vertices),
                max(int(vertex[1]) for vertex in vertices),
                max(int(vertex[2]) for vertex in vertices),
            )
            centroid = (
                (float(vertices[0][0]) + float(vertices[1][0]) + float(vertices[2][0])) / 3.0,
                (float(vertices[0][1]) + float(vertices[1][1]) + float(vertices[2][1])) / 3.0,
                (float(vertices[0][2]) + float(vertices[1][2]) + float(vertices[2][2])) / 3.0,
            )
            return face, bounds_min, bounds_max, centroid

        face_records = [make_face_record(face) for face in faces]

        def compute_bounds(records: Sequence[tuple[ExportCollisionFace, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float]]]) -> tuple[Tuple[int, int, int], Tuple[int, int, int]]:
            return (
                (
                    min(int(record[1][0]) for record in records),
                    min(int(record[1][1]) for record in records),
                    min(int(record[1][2]) for record in records),
                ),
                (
                    max(int(record[2][0]) for record in records),
                    max(int(record[2][1]) for record in records),
                    max(int(record[2][2]) for record in records),
                ),
            )

        def sort_axes_by_extent(bounds_min: Tuple[int, int, int], bounds_max: Tuple[int, int, int]) -> List[int]:
            extents = [int(bounds_max[axis]) - int(bounds_min[axis]) for axis in range(3)]
            return sorted(range(3), key=lambda axis_index: (extents[axis_index], -axis_index), reverse=True)

        def choose_split(
            records: Sequence[tuple[ExportCollisionFace, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float]]],
            bounds_min: Tuple[int, int, int],
            bounds_max: Tuple[int, int, int],
        ) -> Optional[tuple[int, List[tuple[ExportCollisionFace, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float]]], List[tuple[ExportCollisionFace, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float]]]]]:
            axes = sort_axes_by_extent(bounds_min, bounds_max)
            for axis in axes:
                extent = int(bounds_max[axis]) - int(bounds_min[axis])
                if extent <= 0:
                    continue
                split_value = (float(bounds_min[axis]) + float(bounds_max[axis])) * 0.5
                negative_records = [record for record in records if float(record[3][axis]) <= split_value]
                positive_records = [record for record in records if float(record[3][axis]) > split_value]
                if negative_records and positive_records:
                    return int(axis), negative_records, positive_records

            if len(records) > max_leaf_face_count:
                axis = int(axes[0]) if axes else 0
                ordered_records = sorted(records, key=lambda record: float(record[3][axis]))
                split_index = len(ordered_records) // 2
                if 0 < split_index < len(ordered_records):
                    return axis, ordered_records[:split_index], ordered_records[split_index:]
            return None

        def build(
            records: Sequence[tuple[ExportCollisionFace, Tuple[int, int, int], Tuple[int, int, int], Tuple[float, float, float]]],
            depth: int,
        ) -> tuple[dict[str, object], int]:
            count = len(records)
            if count <= 0:
                return {
                    'leaf': True,
                    'faces': [],
                    'bounds_min': (0, 0, 0),
                    'bounds_max': (0, 0, 0),
                }, depth

            bounds_min, bounds_max = compute_bounds(records)
            if count <= leaf_face_limit or (depth >= max_depth_limit and count <= max_leaf_face_count):
                return {
                    'leaf': True,
                    'faces': list(records),
                    'bounds_min': bounds_min,
                    'bounds_max': bounds_max,
                }, depth

            split = choose_split(records, bounds_min, bounds_max)
            if split is None:
                return {
                    'leaf': True,
                    'faces': list(records),
                    'bounds_min': bounds_min,
                    'bounds_max': bounds_max,
                }, depth

            axis, negative_records, positive_records = split
            negative_node, negative_depth = build(negative_records, depth + 1)
            positive_node, positive_depth = build(positive_records, depth + 1)
            return {
                'leaf': False,
                'axis': int(axis),
                'negative': negative_node,
                'positive': positive_node,
                'bounds_min': bounds_min,
                'bounds_max': bounds_max,
            }, max(int(negative_depth), int(positive_depth))

        tree, max_depth = build(face_records, 1)
        nodes: List[_CollisionKDNode] = []
        ordered_faces: List[ExportCollisionFace] = []

        def flatten(node: dict[str, object]) -> int:
            node_index = len(nodes)
            if bool(node.get('leaf', False)):
                leaf_records = list(node.get('faces', []) or [])
                start_index = len(ordered_faces)
                ordered_faces.extend(record[0] for record in leaf_records)
                nodes.append(_CollisionKDNode(0.0, 0.0, start_index, 0, len(leaf_records)))
                return node_index

            nodes.append(_CollisionKDNode(0.0, 0.0, 0, 0, 0))
            flatten(node['negative'])
            positive_index = flatten(node['positive'])
            axis = int(node.get('axis', 0))
            negative_max = int(node['negative']['bounds_max'][axis])
            positive_min = int(node['positive']['bounds_min'][axis])
            nodes[node_index] = _CollisionKDNode(
                float(negative_max + 1),
                float(positive_min - 1),
                positive_index - node_index,
                axis,
                0,
            )
            return node_index

        flatten(tree)
        return nodes, ordered_faces, int(max_depth)

    @staticmethod
    def _normalize_level_game(value: object) -> str:
        return normalize_game_value(value)

    def _derive_object_name_list_from_intro_data(
        self,
        intro_entries: Sequence[ExportIntroData],
        player_object_id: object = -1,
        omitted_object_ids: Optional[set[int]] = None,
    ) -> tuple[int, ...]:
        result: list[int] = []
        seen: set[int] = set()
        omitted = {self._u16(value) for value in (omitted_object_ids or set()) if int(value) > 0}

        def add_object_id(value: object) -> None:
            try:
                object_id = int(value)
            except Exception:
                return
            if object_id <= 0:
                return
            object_id = self._u16(object_id)
            if object_id == 0 or object_id in omitted or object_id in seen:
                return
            seen.add(object_id)
            result.append(object_id)

        for intro in intro_entries:
            add_object_id(getattr(intro, 'object_id', -1))
        add_object_id(player_object_id)
        return tuple(result)

    def _write_object_name_list(self, section: _SectionBuffer, values: Sequence[int]) -> int:
        offset = section.append(b'', alignment=2)
        for value in values:
            section.append(struct.pack('<H', self._u16(value)), alignment=1)
        section.append(struct.pack('<H', 0), alignment=1)
        return offset

    def _write_intro_list(self, section: _SectionBuffer, intro_entries: Sequence[ExportIntroData], intro_data_pointers: Optional[Dict[int, Tuple[str, int]]] = None) -> int:
        offset = section.reserve(max(1, len(intro_entries)) * 112, alignment=16)
        for entry_index, intro in enumerate(intro_entries):
            entry_offset = offset + (entry_index * 112)
            rotation = tuple(float(v) for v in (intro.rotation or (0.0, 0.0, 0.0)))
            rx = float(rotation[0]) if len(rotation) > 0 else 0.0
            ry = float(rotation[1]) if len(rotation) > 1 else 0.0
            rz = float(rotation[2]) if len(rotation) > 2 else 0.0
            rw = float(rotation[3]) if len(rotation) > 3 else 0.0
            section.pack_at(entry_offset + 0, '<4f', rx, ry, rz, rw)

            px, py, pz = self._blender_to_raw_float_triplet(intro.position[:3])
            section.pack_at(entry_offset + 16, '<4f', float(px), float(py), float(pz), 0.0)
            dummy1 = (0.0, 0.0, 0.0, 0.0)
            dummy2 = (0.0, 0.0, 0.0, 0.0)
            scale = (1.0, 1.0, 1.0, 0.0)
            section.pack_at(entry_offset + 32, '<4f', *dummy1)
            section.pack_at(entry_offset + 48, '<4f', *dummy2)
            section.pack_at(entry_offset + 64, '<4f', *(scale[:4] + (0.0,) * max(0, 4 - len(scale[:4]))))
            section.pack_at(entry_offset + 80, '<hhifIiiihh', self._i16(intro.object_id), self._i16(intro.intro_num), int(intro.unique_id), float(intro.max_radius), int(intro.intro_flags) & 0xFFFFFFFF, int(intro.attached_vmo), 0, 0, self._i16(intro.start_frame), self._i16(intro.end_frame))
            pointer = (intro_data_pointers or {}).get(int(intro.index))
            if pointer is not None:
                section_name, payload_offset = pointer
                section.write_pointer_at(entry_offset + 100, section_name, int(payload_offset))
            multi_spline_offset = self._write_multi_spline_data(section, getattr(intro, 'multi_spline_data', None))
            if multi_spline_offset is not None:
                section.write_pointer_at(entry_offset + 104, section.name, int(multi_spline_offset))
        return offset

    def _write_intro_data_section(self, intro_entries: Sequence[ExportIntroData]) -> Tuple[Optional[_SectionBuffer], Dict[int, Tuple[str, int]]]:
        section_name = 'intro_data'
        intro_section = _SectionBuffer(section_name, section_type=0)
        pointers: Dict[int, Tuple[str, int]] = {}
        for intro in intro_entries:
            payload = self._pack_intro_data_payload(intro)
            if not payload:
                continue
            payload_offset = intro_section.append(payload, alignment=4)
            pointers[int(intro.index)] = (section_name, int(payload_offset))
        if not pointers:
            return None, {}
        return intro_section, pointers

    def _pack_intro_data_payload(self, intro: ExportIntroData) -> bytes:
        data_type = int(getattr(intro, 'intro_data_type', 0) or 0)
        generic = dict(getattr(intro, 'generic_intro_data', {}) or {})
        specific = dict(getattr(intro, 'specific_intro_data', {}) or {})
        if data_type <= 0 or not generic:
            return b''
        payload = bytearray()
        payload.extend(struct.pack('<i', int(data_type)))
        payload.extend(struct.pack(
            _GENERIC_INTRO_STRUCT,
            float(generic.get('in_view_remove_dist', 0.0) or 0.0),
            float(generic.get('out_of_view_remove_dist', 0.0) or 0.0),
            self._u8(generic.get('use_model', 0) or 0),
            self._u8(generic.get('pad0', 0) or 0),
            self._u16(generic.get('pad1', 0) or 0),
            self._u16(generic.get('flags', 0) or 0),
            self._i16(generic.get('attached_instance', 0) or 0),
            float(generic.get('swing_length', 0.0) or 0.0),
            self._i32(generic.get('dtp_camera_id', 0) or 0),
        ))
        if data_type == 17:
            payload.extend(struct.pack(_REWARD_INTRO_STRUCT, self._i16(specific.get('reward_type', 0) or 0), self._i16(specific.get('unique_id', 0) or 0), self._i32(specific.get('sound_id', 0) or 0)))
        elif data_type == 13:
            payload.extend(struct.pack(
                _WATER_VOLUME_INTRO_STRUCT,
                float(specific.get('water_depth', 0.0) or 0.0),
                self._i16(specific.get('water_inflow', 0) or 0),
                self._i16(specific.get('water_outflow', 0) or 0),
                float(specific.get('water_speed', 0.0) or 0.0),
                float(specific.get('flow_radius', 0.0) or 0.0),
                self._i16(specific.get('bob_height', 0) or 0),
                self._i16(specific.get('bob_frequency', 0) or 0),
                float(specific.get('bob_grid_size', 0.0) or 0.0),
                self._i16(specific.get('priority', 0) or 0),
                self._u16(specific.get('water_flags', 0) or 0),
            ))
        elif data_type == 12:
            payload.extend(struct.pack(
                _ROPE_OBJ_INTRO_PREFIX_STRUCT,
                self._u8(specific.get('top_connect_type', 0) or 0),
                self._u8(specific.get('bottom_connect_type', 0) or 0),
                self._u16(specific.get('top_connect_instance', 0) or 0),
                self._u8(specific.get('top_connect_model_index', 0) or 0),
                self._u8(specific.get('top_connect_model_marker_index', 0) or 0),
                self._u16(specific.get('bottom_connect_instance', 0) or 0),
                self._u8(specific.get('bottom_connect_model_index', 0) or 0),
                self._u8(specific.get('bottom_connect_model_marker_index', 0) or 0),
                self._u16(specific.get('collision_plane_instance0', 0) or 0),
                self._u16(specific.get('collision_plane_instance1', 0) or 0),
                self._i32(specific.get('rope_camera_dtpid', 0) or 0),
                self._u32(specific.get('rope_camera_overrides_movement', 0) or 0),
                float(specific.get('sound_input_min', 0.0) or 0.0),
                float(specific.get('sound_input_max', 0.0) or 0.0),
            ))
            payload.extend(struct.pack(
                _ROPE_RENDER_PARAMS_STRUCT,
                float(specific.get('render_width', 0.0) or 0.0),
                float(specific.get('render_length_per_v', 0.0) or 0.0),
                float(specific.get('render_u_width', 0.0) or 0.0),
                self._u32(specific.get('render_color', 0) or 0),
                self._u8(specific.get('render_model', 0) or 0),
                self._u8(specific.get('render_texture_main', 0) or 0),
                self._u8(specific.get('render_texture_top', 0) or 0),
                self._u8(specific.get('render_texture_bottom', 0) or 0),
            ))
        else:
            return b''
        return bytes(payload)

    def _write_terrain_light_grids(self, section: _SectionBuffer, lights: Sequence[ExportTerrainLightData]) -> Optional[int]:
        if not lights:
            return None

        grid_size = 32
        cell_size = 4096
        origin = -65536
        max_cell = grid_size - 1
        table_offset = section.reserve(grid_size * grid_size * 4, alignment=4)
        cells: dict[int, list[int]] = {}

        for normalized_index, light in enumerate(lights[:0x4000]):
            raw_x, raw_y, _raw_z = self._blender_to_raw_float_triplet(light.position[:3])
            radius = max(0, int(round(float(getattr(light, 'radius', 0) or 0))))
            col_min = max(0, min(max_cell, int((float(raw_x) - radius - origin) // cell_size)))
            col_max = max(0, min(max_cell, int((float(raw_x) + radius - origin) // cell_size)))
            row_min = max(0, min(max_cell, int((float(raw_y) - radius - origin) // cell_size)))
            row_max = max(0, min(max_cell, int((float(raw_y) + radius - origin) // cell_size)))
            if col_max < col_min:
                col_min, col_max = col_max, col_min
            if row_max < row_min:
                row_min, row_max = row_max, row_min

            encoded_index = int(normalized_index) & 0x3FFF
            if self._terrain_light_grid_high_bits(light):
                encoded_index |= self._terrain_light_grid_high_bits(light)

            for row in range(row_min, row_max + 1):
                for col in range(col_min, col_max + 1):
                    cells.setdefault((row * grid_size) + col, []).append(encoded_index & 0xFFFF)

        for cell_index in sorted(cells):
            values = cells[cell_index][:256]
            if not values:
                continue
            list_offset = section.reserve((len(values) + 1) * 2, alignment=2)
            for value_index, value in enumerate(values):
                section.pack_at(list_offset + (value_index * 2), '<H', int(value) & 0xFFFF)
            section.pack_at(list_offset + (len(values) * 2), '<H', 0xFFFF)
            section.write_pointer_at(table_offset + (cell_index * 4), section.name, list_offset)
        return table_offset

    @staticmethod
    def _terrain_light_grid_high_bits(light: ExportTerrainLightData) -> int:
        try:
            color = tuple(float(v) for v in (getattr(light, 'color', ()) or ()))
        except Exception:
            color = ()
        is_black = len(color) >= 3 and all(abs(float(v)) <= 1.0e-6 for v in color[:3])
        try:
            multiplier = int(getattr(light, 'multiplier', 0) or 0)
        except Exception:
            multiplier = 0
        try:
            light_type = int(getattr(light, 'type', 0) or 0)
        except Exception:
            light_type = 0
        if is_black and multiplier > 4096 and light_type == 24:
            return 0xC000
        return 0

    def _write_markup_polyline(self, section: _SectionBuffer, points: Sequence[Sequence[float]]) -> int:
        point_count = max(0, len(points))
        offset = section.reserve(16 + (point_count * 16), alignment=16)
        section.pack_at(offset + 0, '<i', point_count)
        for point_index, point in enumerate(points):
            px, py, pz = self._blender_to_raw_float_triplet(point[:3])
            pw = float(self._tuple_get(point, 3, 1.0))
            section.pack_at(offset + 16 + (point_index * 16), '<4f', float(px), float(py), float(pz), float(pw))
        return offset

    def _write_markups(self, section: _SectionBuffer, markups: Sequence[ExportMarkupData], *, level_game: str = 'legend') -> Optional[int]:
        count = len(markups)
        if count <= 0:
            return None
        game = self._normalize_level_game(level_game)
        entry_size = 76 if game == 'anniversary' else 48
        polyline_offsets = [
            None
            if (int(getattr(markup, 'flags', 0) or 0) & _MARKUP_BBOX_FLAGS)
            else self._write_markup_polyline(section, list(getattr(markup, 'polyline', []) or []))
            for markup in markups
        ]
        table_offset = section.reserve(count * entry_size, alignment=16)
        for entry_index, markup in enumerate(markups):
            entry_offset = table_offset + (entry_index * entry_size)
            section.pack_at(entry_offset + 0, '<i', 0)  # OverrideMovementCamera
            section.pack_at(entry_offset + 4, '<i', 0)  # DTPCameraDataID
            section.pack_at(entry_offset + 8, '<i', 0)  # DTPMarkupDataID
            cursor = entry_offset + 12
            if game == 'anniversary':
                section.pack_at(cursor, '<i', int(getattr(markup, 'animated_segment', 0) or 0))
                cursor += 4
                section.pack_at(cursor, '<i', 0)  # CameraAntic.UseAnticCamera
                cursor += 4
                for _ in range(5):
                    section.pack_at(cursor, '<i', 0)
                    cursor += 4
            section.pack_at(cursor, '<I', int(getattr(markup, 'flags', 0) or 0) & 0xFFFFFFFF)
            cursor += 4
            section.pack_at(cursor, '<hh', self._i16(getattr(markup, 'intro_id', 0) or 0), 0)  # introID, markupID
            cursor += 4
            px, py, pz = self._blender_to_raw_float_triplet(getattr(markup, 'position', (0.0, 0.0, 0.0))[:3])
            section.pack_at(cursor, '<3f', float(px), float(py), float(pz))
            cursor += 12
            bbox = tuple(getattr(markup, 'bbox', (0, 0, 0, 0, 0, 0)) or (0, 0, 0, 0, 0, 0))
            bbox_values = tuple(self._i16(self._tuple_get(bbox, bbox_index, 0)) for bbox_index in range(6))
            section.pack_at(cursor, '<6h', *bbox_values)
            cursor += 12
            polyline_offset = polyline_offsets[entry_index]
            if polyline_offset is None:
                section.pack_at(cursor, '<I', 0)
            else:
                section.write_pointer_at(cursor, section.name, polyline_offset)
        return table_offset

    @staticmethod
    def _safe_sfx_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _safe_sfx_float(value: object, default: float = 0.0) -> float:
        try:
            result = float(value)
        except Exception:
            return float(default)
        return result if math.isfinite(result) else float(default)

    @staticmethod
    def _sfx_props(sound: ExportSFXSound) -> Dict[str, object]:
        props = getattr(sound, 'properties', None)
        return dict(props or {}) if isinstance(props, dict) else {}

    def _write_sfx_sound_ids(self, section: _SectionBuffer, offset: int, base_size: int, sound_ids: Sequence[int]) -> None:
        cursor = int(offset) + int(base_size)
        for sound_id in list(sound_ids or [])[:255]:
            section.pack_at(cursor, '<I', self._u32(sound_id))
            cursor += 4

    def _write_sfx_sound_payload(self, section: _SectionBuffer, sound: ExportSFXSound) -> Optional[int]:
        kind = str(getattr(sound, 'kind', '') or '').strip().lower()
        props = self._sfx_props(sound)
        sound_ids = [self._u32(value) for value in list(getattr(sound, 'sound_ids', []) or []) if self._safe_sfx_int(value, 0) > 0]
        sound_ids = sound_ids[:255]
        num_sfx_ids = self._u8(len(sound_ids))

        if kind in {'periodic', 'periodicsound'}:
            offset = section.reserve(_SFX_PERIODIC_SOUND_BASE_SIZE + (len(sound_ids) * 4), alignment=4)
            section.pack_at(
                offset,
                '<BBHffBBHHHHHHxx',
                num_sfx_ids,
                self._u8(props.get('flags', 0)),
                self._u16(props.get('min_vol_distance', props.get('minVolDistance', 0))),
                self._safe_sfx_float(props.get('pitch', 0.0)),
                self._safe_sfx_float(props.get('pitch_variation', props.get('pitchVariation', 0.0))),
                self._u8(props.get('max_volume', props.get('maxVolume', 0))),
                self._u8(props.get('max_vol_variation', props.get('maxVolVariation', 0))),
                self._u16(props.get('initial_delay', props.get('initialDelay', 0))),
                self._u16(props.get('initial_delay_variation', props.get('initialDelayVariation', 0))),
                self._u16(props.get('on_time', props.get('onTime', 0))),
                self._u16(props.get('on_time_variation', props.get('onTimeVariation', 0))),
                self._u16(props.get('off_time', props.get('offTime', 0))),
                self._u16(props.get('off_time_variation', props.get('offTimeVariation', 0))),
            )
            self._write_sfx_sound_ids(section, offset, _SFX_PERIODIC_SOUND_BASE_SIZE, sound_ids)
            return offset

        if kind in {'event', 'eventsound'}:
            offset = section.reserve(_SFX_EVENT_SOUND_BASE_SIZE + (len(sound_ids) * 4), alignment=4)
            section.pack_at(
                offset,
                '<BBHffBBxxff',
                self._u8(props.get('sound_group', props.get('soundGroup', 0))),
                num_sfx_ids,
                self._u16(props.get('min_vol_distance', props.get('minVolDistance', 0))),
                self._safe_sfx_float(props.get('pitch', 0.0)),
                self._safe_sfx_float(props.get('pitch_variation', props.get('pitchVariation', 0.0))),
                self._u8(props.get('max_volume', props.get('maxVolume', 0))),
                self._u8(props.get('max_vol_variation', props.get('maxVolVariation', 0))),
                self._safe_sfx_float(props.get('delay', 0.0)),
                self._safe_sfx_float(props.get('delay_variation', props.get('delayVariation', 0.0))),
            )
            self._write_sfx_sound_ids(section, offset, _SFX_EVENT_SOUND_BASE_SIZE, sound_ids)
            return offset

        if kind in {'one_shot', 'oneshot', 'one-shot'}:
            offset = section.reserve(_SFX_ONESHOT_SOUND_BASE_SIZE + (len(sound_ids) * 4), alignment=4)
            section.pack_at(
                offset,
                '<BBHffBBxx',
                self._u8(props.get('sound_group', props.get('soundGroup', 0))),
                num_sfx_ids,
                self._u16(props.get('min_vol_distance', props.get('minVolDistance', 0))),
                self._safe_sfx_float(props.get('pitch', 0.0)),
                self._safe_sfx_float(props.get('pitch_variation', props.get('pitchVariation', 0.0))),
                self._u8(props.get('max_volume', props.get('maxVolume', 0))),
                self._u8(props.get('max_vol_variation', props.get('maxVolVariation', 0))),
            )
            self._write_sfx_sound_ids(section, offset, _SFX_ONESHOT_SOUND_BASE_SIZE, sound_ids)
            return offset

        if kind == 'stream':
            name = str(props.get('name', '') or '').encode('utf-8', errors='ignore')[:63]
            name_bytes = name + (b'\x00' * (64 - len(name)))
            music_vars_value = props.get('music_vars', props.get('musicVars', (0, 0, 0, 0)))
            if isinstance(music_vars_value, str):
                try:
                    decoded = json.loads(music_vars_value)
                    music_vars_value = decoded if isinstance(decoded, (list, tuple)) else (0, 0, 0, 0)
                except Exception:
                    music_vars_value = (0, 0, 0, 0)
            music_vars = list(music_vars_value or []) if isinstance(music_vars_value, (list, tuple)) else []
            while len(music_vars) < 4:
                music_vars.append(0)
            offset = section.reserve(_SFX_STREAM_SOUND_SIZE, alignment=4)
            section.pack_at(
                offset,
                '<hhhh4B64s',
                self._i16(props.get('choose_chance', props.get('chooseChance', 0))),
                self._i16(props.get('play_chance', props.get('playChance', 0))),
                self._i16(props.get('min_volume_dist', props.get('minVolumeDist', props.get('min_vol_distance', 0)))),
                self._i16(props.get('max_volume', props.get('maxVolume', 0))),
                self._u8(music_vars[0]),
                self._u8(music_vars[1]),
                self._u8(music_vars[2]),
                self._u8(music_vars[3]),
                name_bytes,
            )
            return offset

        return None

    def _write_sfx_pointer_list(self, section: _SectionBuffer, target_offsets: Sequence[int]) -> Optional[int]:
        offsets = [int(value) for value in list(target_offsets or []) if int(value) > 0]
        if not offsets:
            return None
        pointer_list_offset = section.reserve(len(offsets) * 4, alignment=4)
        for index, target_offset in enumerate(offsets):
            section.write_pointer_at(pointer_list_offset + (index * 4), section.name, int(target_offset))
        return pointer_list_offset

    def _write_sfx_perimeter_payload(self, section: _SectionBuffer, perimeter: ExportSFXPerimeter, event_offsets_by_index: Dict[int, int], periodic_offsets_by_index: Dict[int, int]) -> int:
        offset = section.reserve(_SFX_PERIMETER_SIZE, alignment=4)
        flags = (self._u8(getattr(perimeter, 'action', 0)) & 0xFF)
        flags |= (1 if self._safe_sfx_int(getattr(perimeter, 'breached', 0), 0) else 0) << 8
        flags |= (1 if self._safe_sfx_int(getattr(perimeter, 'on_exit', 0), 0) else 0) << 14
        flags |= (1 if self._safe_sfx_int(getattr(perimeter, 'always', 0), 0) else 0) << 15
        section.pack_at(
            offset,
            '<IIIHHII2Bxx',
            self._u32(getattr(perimeter, 'radius', 0)),
            self._u32(getattr(perimeter, 'n_fired', 0)),
            self._u32(getattr(perimeter, 'invader_offset', 0)),
            int(flags) & 0xFFFF,
            0,
            0,
            0,
            self._u8(getattr(perimeter, 'varnum', 0)),
            self._u8(getattr(perimeter, 'value', 0)),
        )
        try:
            event_index = int(getattr(perimeter, 'event_index', -1))
        except Exception:
            event_index = -1
        try:
            periodic_index = int(getattr(perimeter, 'periodic_index', -1))
        except Exception:
            periodic_index = -1
        event_offset = event_offsets_by_index.get(event_index)
        periodic_offset = periodic_offsets_by_index.get(periodic_index)
        if event_offset is not None:
            section.write_pointer_at(offset + 16, section.name, int(event_offset))
        if periodic_offset is not None:
            section.write_pointer_at(offset + 20, section.name, int(periodic_offset))
        return offset

    def _write_sfx_marker_sound_group(
        self,
        section: _SectionBuffer,
        marker_offset: int,
        data_count_offset: int,
        data_pointer_offset: int,
        inline_array_offset: Optional[int],
        sounds: Sequence[ExportSFXSound],
        *,
        use_inline_up_to_capacity: bool,
    ) -> tuple[List[int], Dict[int, int]]:
        sound_list = sorted(list(sounds or []), key=lambda item: int(getattr(item, 'index', 0)))
        target_offsets: List[int] = []
        offsets_by_index: Dict[int, int] = {}
        for default_index, sound in enumerate(sound_list):
            try:
                sound_index = int(getattr(sound, 'index', default_index))
            except Exception:
                sound_index = int(default_index)
            payload_offset = self._write_sfx_sound_payload(section, sound)
            if payload_offset is None:
                continue
            target_offsets.append(int(payload_offset))
            offsets_by_index[sound_index] = int(payload_offset)
        section.pack_at(marker_offset + int(data_count_offset), '<I', len(target_offsets))
        if use_inline_up_to_capacity and inline_array_offset is not None and len(target_offsets) <= _SFX_POINTER_ARRAY_CAPACITY:
            section.pack_at(marker_offset + int(data_pointer_offset), '<I', 0)
            for index, target_offset in enumerate(target_offsets[:_SFX_POINTER_ARRAY_CAPACITY]):
                section.write_pointer_at(marker_offset + int(inline_array_offset) + (index * 4), section.name, int(target_offset))
        else:
            pointer_list_offset = self._write_sfx_pointer_list(section, target_offsets)
            if pointer_list_offset is not None:
                section.write_pointer_at(marker_offset + int(data_pointer_offset), section.name, int(pointer_list_offset))
            else:
                section.pack_at(marker_offset + int(data_pointer_offset), '<I', 0)
        return target_offsets, offsets_by_index

    def _write_sfx_markers(self, section: _SectionBuffer, markers: Sequence[ExportSFXMarker]) -> Optional[int]:
        marker_entries = sorted(list(markers or []), key=lambda item: int(getattr(item, 'index', 0)))
        if not marker_entries:
            return None
        marker_table_offset = section.reserve(len(marker_entries) * _SFX_MARKER_SIZE, alignment=16)
        for marker_order, marker in enumerate(marker_entries):
            marker_offset = marker_table_offset + (marker_order * _SFX_MARKER_SIZE)
            self._pack_padded_vec3f(section, marker_offset + 0, getattr(marker, 'position', (0.0, 0.0, 0.0)))
            section.pack_at(marker_offset + 12, '<i', self._i32(getattr(marker, 'unique_id', 0)))
            section.pack_at(marker_offset + 16, '<I', 0)
            section.pack_at(marker_offset + 20, '<B3x', self._u8(getattr(marker, 'plane', 0)))
            section.pack_at(marker_offset + 24, '<I', self._u32(getattr(marker, 'spline_id', 0)))
            event_offsets, event_offsets_by_index = self._write_sfx_marker_sound_group(
                section,
                marker_offset,
                28 + 8,
                28 + 12,
                68,
                list(getattr(marker, 'event_sounds', []) or []),
                use_inline_up_to_capacity=True,
            )
            periodic_offsets, periodic_offsets_by_index = self._write_sfx_marker_sound_group(
                section,
                marker_offset,
                28 + 0,
                28 + 4,
                132,
                list(getattr(marker, 'periodic_sounds', []) or []),
                use_inline_up_to_capacity=True,
            )
            self._write_sfx_marker_sound_group(
                section,
                marker_offset,
                28 + 16,
                28 + 20,
                None,
                list(getattr(marker, 'one_shot_sounds', []) or []),
                use_inline_up_to_capacity=False,
            )
            self._write_sfx_marker_sound_group(
                section,
                marker_offset,
                28 + 24,
                28 + 28,
                None,
                list(getattr(marker, 'stream_sounds', []) or []),
                use_inline_up_to_capacity=False,
            )
            section.pack_at(marker_offset + 28 + 32, '<I', 0)
            section.pack_at(marker_offset + 28 + 36, '<I', 0)

            perimeter_entries = sorted(list(getattr(marker, 'perimeter_actions', []) or []), key=lambda item: int(getattr(item, 'index', 0)))[:_SFX_POINTER_ARRAY_CAPACITY]
            for perimeter_index, perimeter in enumerate(perimeter_entries):
                perimeter_offset = self._write_sfx_perimeter_payload(section, perimeter, event_offsets_by_index, periodic_offsets_by_index)
                section.write_pointer_at(marker_offset + 196 + (perimeter_index * 4), section.name, int(perimeter_offset))
        return marker_table_offset


    def _write_terrain_lights(self, section: _SectionBuffer, lights: Sequence[ExportTerrainLightData]) -> Optional[int]:
        count = len(lights)
        if count <= 0:
            return None
        offset = section.reserve(count * 36, alignment=4)
        for index, light in enumerate(lights):
            item_offset = offset + (index * 36)
            raw_x, raw_y, raw_z = self._blender_to_raw_int_triplet(light.position[:3])
            section.pack_at(item_offset + 0, '<iii', raw_x, raw_y, raw_z)
            section.pack_at(item_offset + 12, '<i', int(light.radius))
            red_value = float(self._tuple_get(light.color, 0, 1.0))
            green_value = float(self._tuple_get(light.color, 1, 1.0))
            blue_value = float(self._tuple_get(light.color, 2, 1.0))
            red = self._u8(round(red_value * 255.0) if 0.0 <= red_value <= 1.0 else red_value)
            green = self._u8(round(green_value * 255.0) if 0.0 <= green_value <= 1.0 else green_value)
            blue = self._u8(round(blue_value * 255.0) if 0.0 <= blue_value <= 1.0 else blue_value)
            section.pack_at(item_offset + 16, '<BBBB', red, green, blue, self._u8(light.type))
            section.pack_at(item_offset + 20, '<hh', self._i16(light.multiplier), self._i16(light.hotspot_angle))
            raw_nx, raw_ny, raw_nz = self._blender_direction_to_raw_i16_triplet(light.direction[:3])
            section.pack_at(item_offset + 24, '<hhh', raw_nx, raw_ny, raw_nz)
            section.pack_at(item_offset + 30, '<hh', self._i16(light.falloff_angle), self._i16(light.light_id))
            section.pack_at(item_offset + 34, '<H', 0)
        return offset

    @staticmethod
    def _blender_direction_to_raw_i16_triplet(value: Sequence[float]) -> tuple[int, int, int]:
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        return TRLevelDRMWriter._i16(round(-x)), TRLevelDRMWriter._i16(round(z)), TRLevelDRMWriter._i16(round(y))

    @staticmethod
    def _raw_bg_position_from_blender(value: Sequence[float]) -> tuple[float, float, float]:
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        return (-x, z, y)

    def _write_bg_objects(self, section: _SectionBuffer, bg_objects: Sequence[ExportBGObject], bg_instances: Sequence[ExportBGInstance] | None = None) -> Dict[int, int]:
        if not bg_objects:
            return {}

        count = len(bg_objects)
        table_offset = section.reserve(count * _BGOBJECT_ENTRY_SIZE, alignment=16)
        object_offsets: Dict[int, int] = {}
        for order_index, bg_object in enumerate(bg_objects):
            object_offsets[int(bg_object.index)] = table_offset + (order_index * _BGOBJECT_ENTRY_SIZE)

        color_list_count_by_object: Dict[int, int] = {int(bg.index): 1 for bg in bg_objects}
        for instance in list(bg_instances or []):
            try:
                bg_index = int(instance.bg_object_index)
                color_index = int(getattr(instance, 'color_data_index', 0) or 0)
            except Exception:
                continue
            if bg_index in color_list_count_by_object and color_index >= 0:
                color_list_count_by_object[bg_index] = max(color_list_count_by_object.get(bg_index, 1), color_index + 1)

        scroll_info_offsets: Dict[int, int] = {}
        animated_info_offsets: Dict[int, int] = {}
        env_vertex_offsets: Dict[int, int] = {}
        eye_ref_env_vertex_offsets: Dict[int, int] = {}
        for bg_object in bg_objects:
            bg_index = int(bg_object.index)
            scroll_info_offsets[bg_index] = section.reserve(4, alignment=4)
            section.pack_at(scroll_info_offsets[bg_index], '<i', 0)
            animated_info_offsets[bg_index] = section.reserve(4, alignment=4)
            section.pack_at(animated_info_offsets[bg_index], '<i', 0)

        strip_offsets_by_object: Dict[int, Optional[int]] = {}
        vertex_offsets_by_object: Dict[int, Optional[int]] = {}
        color_offsets_by_object: Dict[int, Optional[int]] = {}

        for bg_object in bg_objects:
            bg_index = int(bg_object.index)
            raw_vertex_data = bytes(getattr(bg_object, 'raw_vertex_data', b'') or b'')
            raw_color_data = bytes(getattr(bg_object, 'raw_color_data', b'') or b'')
            if raw_vertex_data and len(raw_vertex_data) >= len(bg_object.vertices) * _BGOBJECT_VERTEX_SIZE:
                vertex_offsets_by_object[bg_index] = self._write_bg_object_raw_block(section, raw_vertex_data, alignment=4)
            else:
                vertex_offsets_by_object[bg_index] = self._write_bg_object_vertices(section, bg_object.vertices)
            per_color_slot_size = int(len(bg_object.vertices)) * 4
            required_color_len = per_color_slot_size * max(1, int(color_list_count_by_object.get(bg_index, 1) or 1))
            if raw_color_data and per_color_slot_size > 0 and len(raw_color_data) >= per_color_slot_size:
                if len(raw_color_data) < required_color_len:
                    raw_color_data = raw_color_data + ((b'\xFF\xFF\xFF\x80' * ((required_color_len - len(raw_color_data) + 3) // 4))[:required_color_len - len(raw_color_data)])
                color_offsets_by_object[bg_index] = self._write_bg_object_raw_block(section, raw_color_data, alignment=16)
            else:
                color_offsets_by_object[bg_index] = self._write_bg_object_colors(
                    section,
                    bg_object.vertices,
                    color_list_count_by_object.get(bg_index, 1),
                )
            env_vertex_offsets[bg_index] = self._write_bg_object_empty_count_block(section)
            eye_ref_env_vertex_offsets[bg_index] = self._write_bg_object_empty_count_block(section)
            strip_offsets_by_object[bg_index] = self._write_bg_object_strip_chain(section, bg_object)

        for bg_object in bg_objects:
            entry_offset = object_offsets[int(bg_object.index)]
            sx = float(self._tuple_get(bg_object.scale, 0, 1.0))
            sy = float(self._tuple_get(bg_object.scale, 1, 1.0))
            sz = float(self._tuple_get(bg_object.scale, 2, 1.0))
            section.pack_at(entry_offset + 0, '<4f', sx, sy, sz, 0.0)

            raw_px, raw_py, raw_pz = self._raw_bg_position_from_blender(bg_object.position[:3])
            section.pack_at(entry_offset + 16, '<4f', float(raw_px), float(raw_py), float(raw_pz), 1.0)
            bg_index = int(bg_object.index)
            section.pack_at(entry_offset + 32, '<i', int(getattr(bg_object, 'flags', 0) or 0))
            section.pack_at(entry_offset + 36, '<I', 0)
            section.write_pointer_at(entry_offset + 40, section.name, int(animated_info_offsets.get(bg_index, 0)))
            section.write_pointer_at(entry_offset + 44, section.name, int(scroll_info_offsets.get(bg_index, 0)))
            strip_offset = strip_offsets_by_object.get(bg_index)
            if strip_offset is not None:
                section.write_pointer_at(entry_offset + 48, section.name, int(strip_offset))
            else:
                section.pack_at(entry_offset + 48, '<I', 0)
            section.pack_at(entry_offset + 52, '<I', 0)
            section.pack_at(entry_offset + 56, '<I', 0)
            section.pack_at(entry_offset + 60, '<2f', 0.00001, 0.00001)
            vertex_offset = vertex_offsets_by_object.get(int(bg_object.index))
            if vertex_offset is not None and bg_object.vertices:
                section.write_pointer_at(entry_offset + 68, section.name, int(vertex_offset))
            else:
                section.pack_at(entry_offset + 68, '<I', 0)
            section.pack_at(entry_offset + 72, '<i', int(max(0, len(bg_object.vertices))))
            color_offset = color_offsets_by_object.get(int(bg_object.index))
            if color_offset is not None and bg_object.vertices:
                section.write_pointer_at(entry_offset + 76, section.name, int(color_offset))
            else:
                section.pack_at(entry_offset + 76, '<I', 0)
            section.write_pointer_at(entry_offset + 80, section.name, int(env_vertex_offsets.get(int(bg_object.index), 0)))
            section.write_pointer_at(entry_offset + 84, section.name, int(eye_ref_env_vertex_offsets.get(int(bg_object.index), 0)))
            section.pack_at(entry_offset + 88, '<I', int(getattr(bg_object, 'cdc_render_data_id', 0) or 0) & 0xFFFFFFFF)
            section.pack_at(entry_offset + 92, '<I', 0)
        return object_offsets

    def _resolve_bg_object_reflection_indices(self, bg_object: ExportBGObject, list_attr: str, strip_attr: str) -> List[int]:
        explicit = [int(value) for value in list(getattr(bg_object, list_attr, []) or []) if int(value) >= 0]
        if explicit:
            max_index = max(0, len(getattr(bg_object, 'vertices', []) or []))
            return sorted({value for value in explicit if max_index <= 0 or value < max_index})
        indices: List[int] = []
        for strip in list(getattr(bg_object, 'strips', []) or []):
            if bool(getattr(strip, 'is_terminator', False)) or not bool(getattr(strip, strip_attr, False)):
                continue
            base = int(getattr(strip, 'vertex_base_offset', 0) or 0)
            for value in list(getattr(strip, 'indices', []) or []):
                index = base + int(value)
                if index >= 0:
                    indices.append(index)
        max_index = max(0, len(getattr(bg_object, 'vertices', []) or []))
        return sorted({value for value in indices if max_index <= 0 or value < max_index})

    def _write_bg_object_empty_count_block(self, section: _SectionBuffer) -> int:
        offset = section.reserve(4, alignment=4)
        section.pack_at(offset, '<i', 0)
        return offset

    @staticmethod
    def _normalized_vector(x: float, y: float, z: float, default: Tuple[float, float, float] = (0.0, 0.0, 1.0)) -> Tuple[float, float, float]:
        length = math.sqrt((float(x) * float(x)) + (float(y) * float(y)) + (float(z) * float(z)))
        if not math.isfinite(length) or length <= 0.000001:
            return default
        return (float(x) / length, float(y) / length, float(z) / length)

    def _bg_object_vertex_normals(self, bg_object: ExportBGObject) -> Dict[int, Tuple[float, float, float]]:
        vertices = list(getattr(bg_object, 'vertices', []) or [])
        accum: Dict[int, List[float]] = {index: [0.0, 0.0, 0.0] for index in range(len(vertices))}
        for strip in list(getattr(bg_object, 'strips', []) or []):
            if bool(getattr(strip, 'is_terminator', False)):
                continue
            base = int(getattr(strip, 'vertex_base_offset', 0) or 0)
            raw_indices = [base + int(value) for value in list(getattr(strip, 'indices', []) or [])]
            usable = raw_indices[: (len(raw_indices) // 3) * 3]
            for tri_start in range(0, len(usable), 3):
                i0, i1, i2 = usable[tri_start:tri_start + 3]
                if i0 == i1 or i1 == i2 or i2 == i0:
                    continue
                if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= len(vertices):
                    continue
                p0 = vertices[i0].position
                p1 = vertices[i1].position
                p2 = vertices[i2].position
                ax = float(p1[0]) - float(p0[0])
                ay = float(p1[1]) - float(p0[1])
                az = float(p1[2]) - float(p0[2])
                bx = float(p2[0]) - float(p0[0])
                by = float(p2[1]) - float(p0[1])
                bz = float(p2[2]) - float(p0[2])
                nx = (ay * bz) - (az * by)
                ny = (az * bx) - (ax * bz)
                nz = (ax * by) - (ay * bx)
                length = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
                if not math.isfinite(length) or length <= 0.000001:
                    continue
                for vertex_index in (i0, i1, i2):
                    accum[vertex_index][0] += nx
                    accum[vertex_index][1] += ny
                    accum[vertex_index][2] += nz
        normals: Dict[int, Tuple[float, float, float]] = {}
        for index, value in accum.items():
            normals[index] = self._normalized_vector(float(value[0]), float(value[1]), float(value[2]))
        return normals

    def _write_bg_object_env_patch_vertices(
        self,
        section: _SectionBuffer,
        bg_object: ExportBGObject,
        indices: Sequence[int],
        vertex_offset: int,
        *,
        eye_ref: bool,
    ) -> int:
        vertices = list(getattr(bg_object, 'vertices', []) or [])
        max_index = len(vertices)
        values = sorted({int(value) for value in list(indices or []) if int(value) >= 0 and int(value) < max_index})
        entry_size = 24 if eye_ref else 16
        offset = section.reserve(4 + (len(values) * entry_size), alignment=4)
        section.pack_at(offset, '<i', len(values))
        if not values:
            return offset

        normals = self._bg_object_vertex_normals(bg_object)
        for entry_index, vertex_index in enumerate(values):
            entry_offset = offset + 4 + (entry_index * entry_size)
            uv_offset = int(vertex_offset) + (int(vertex_index) * _BGOBJECT_VERTEX_SIZE) + 8
            section.write_pointer_at(entry_offset, section.name, uv_offset)
            nx, ny, nz = normals.get(int(vertex_index), (0.0, 0.0, 1.0))
            section.pack_at(entry_offset + 4, '<3f', float(nx), float(ny), float(nz))
            if eye_ref:
                pos = vertices[int(vertex_index)].position
                raw_x = self._i16(round(float(pos[0]) if len(pos) > 0 else 0.0))
                raw_y = self._i16(round(float(pos[1]) if len(pos) > 1 else 0.0))
                raw_z = self._i16(round(float(pos[2]) if len(pos) > 2 else 0.0))
                section.pack_at(entry_offset + 16, '<4h', raw_x, raw_y, raw_z, 0)
        return offset

    def _write_bg_object_raw_block(self, section: _SectionBuffer, payload: bytes, *, alignment: int = 4) -> Optional[int]:
        if not payload:
            return None
        return section.append(bytes(payload), alignment=alignment)

    def _write_bg_object_vertices(self, section: _SectionBuffer, vertices: Sequence[ExportVertex]) -> Optional[int]:
        if not vertices:
            return None
        offset = section.reserve(len(vertices) * _BGOBJECT_VERTEX_SIZE, alignment=4)
        for index, vertex in enumerate(vertices):
            item_offset = offset + (index * _BGOBJECT_VERTEX_SIZE)
            pos = vertex.position
            raw_x = self._i16(round(float(pos[0]) if len(pos) > 0 else 0.0))
            raw_y = self._i16(round(float(pos[1]) if len(pos) > 1 else 0.0))
            raw_z = self._i16(round(float(pos[2]) if len(pos) > 2 else 0.0))
            uv = vertex.uv
            raw_u = self._i16(round(float(uv[0] if len(uv) > 0 else 0.0) * 4096.0))
            raw_v = self._i16(round(float(uv[1] if len(uv) > 1 else 0.0) * 4096.0))
            section.pack_at(item_offset + 0, '<hhhhhh', raw_x, raw_y, raw_z, 0, raw_u, raw_v)
        return offset

    def _write_bg_object_colors(self, section: _SectionBuffer, vertices: Sequence[ExportVertex], color_list_count: int = 1) -> Optional[int]:
        if not vertices:
            return None
        slot_count = max(1, int(color_list_count or 1))
        offset = section.reserve(len(vertices) * 4 * slot_count, alignment=16)
        for slot_index in range(slot_count):
            slot_offset = offset + (slot_index * len(vertices) * 4)
            for index, vertex in enumerate(vertices):
                color = vertex.color
                r = self._u8(color[0] if len(color) > 0 else 255)
                g = self._u8(color[1] if len(color) > 1 else 255)
                b = self._u8(color[2] if len(color) > 2 else 255)
                a = self._u8(color[3] if len(color) > 3 else 255)
                section.pack_at(slot_offset + (index * 4), '<I', (a << 24) | (r << 16) | (g << 8) | b)
        return offset

    def _write_bg_object_strip_chain(self, section: _SectionBuffer, bg_object: ExportBGObject) -> Optional[int]:
        source_strips = list(getattr(bg_object, 'strips', []) or [])
        strips = [strip for strip in source_strips if bool(getattr(strip, 'is_terminator', False)) or len(getattr(strip, 'indices', []) or []) > 0]
        if not strips:
            return None
        if not bool(getattr(strips[-1], 'is_terminator', False)):
            strips = list(strips) + [ExportStrip(name=f'{getattr(bg_object, "index", 0)}_BGObjectStripTerminator', raw_count=0, is_terminator=True)]

        first_offset: Optional[int] = None
        previous_offset: Optional[int] = None
        for strip in strips:
            indices = [int(value) for value in list(getattr(strip, 'indices', []) or [])]
            is_terminator = bool(getattr(strip, 'is_terminator', False))
            try:
                stored_count = int(getattr(strip, 'raw_count', -1))
            except Exception:
                stored_count = -1
            if stored_count < 0:
                stored_count = 0 if is_terminator else len(indices)
            if stored_count <= 0:
                indices = []
                is_terminator = True
                stored_count = 0

            strip_offset = section.reserve(_BGOBJECT_STRIP_HEADER_SIZE + (len(indices) * 2), alignment=4)
            if first_offset is None:
                first_offset = strip_offset
            if previous_offset is not None:
                section.write_pointer_at(previous_offset + 24, section.name, strip_offset)

            section.pack_at(strip_offset + 0, '<i', int(stored_count))
            sort_vertex_prop = getattr(strip, 'sort_vertex', None)
            if sort_vertex_prop is not None:
                try:
                    sort_vertex = (
                        self._i16(int(sort_vertex_prop[0])),
                        self._i16(int(sort_vertex_prop[1])),
                        self._i16(int(sort_vertex_prop[2])),
                    )
                except Exception:
                    sort_vertex = (0, 0, 0)
            else:
                sort_vertex = (0, 0, 0)
            if sort_vertex == (0, 0, 0) and indices and bg_object.vertices:
                source_index = int(indices[0])
                if 0 <= source_index < len(bg_object.vertices):
                    pos = bg_object.vertices[source_index].position
                    sort_vertex = (
                        self._i16(round(float(pos[0]) if len(pos) > 0 else 0.0)),
                        self._i16(round(float(pos[1]) if len(pos) > 1 else 0.0)),
                        self._i16(round(float(pos[2]) if len(pos) > 2 else 0.0)),
                    )
            section.pack_at(strip_offset + 4, '<hhhH', sort_vertex[0], sort_vertex[1], sort_vertex[2], 0)
            section.pack_at(strip_offset + 12, '<I', int(getattr(strip, 'tpageid', 0)) & 0xFFFFFFFF)
            section.pack_at(strip_offset + 16, '<i', int(getattr(strip, 'sort_push', 0) or 0))
            section.pack_at(strip_offset + 20, '<f', float(getattr(strip, 'scroll_offset', 0.0) or 0.0))
            section.pack_at(strip_offset + 24, '<I', 0)
            cursor = strip_offset + _BGOBJECT_STRIP_HEADER_SIZE
            for index in indices:
                section.pack_at(cursor, '<H', self._u16(index))
                cursor += 2
            previous_offset = strip_offset
        return first_offset

    def _write_bg_instances(
        self,
        section: _SectionBuffer,
        bg_instances: Sequence[ExportBGInstance],
        bg_objects_section: str,
        bg_object_offsets: Dict[int, int],
        bg_objects: Sequence[ExportBGObject],
    ) -> Optional[int]:
        valid_instances = [instance for instance in bg_instances if int(instance.bg_object_index) in bg_object_offsets]
        if not valid_instances:
            return None

        scale_by_object = {int(bg.index): tuple(float(v) for v in bg.scale) for bg in bg_objects}
        offset = section.reserve(len(valid_instances) * _BGINSTANCE_ENTRY_SIZE, alignment=16)
        for order_index, instance in enumerate(valid_instances):
            item_offset = offset + (order_index * _BGINSTANCE_ENTRY_SIZE)
            rows = self._coerce_matrix_rows(instance.matrix_rows)
            inv_rows = self._build_bginstance_inverse_rows(rows, scale_by_object.get(int(instance.bg_object_index), (1.0, 1.0, 1.0)))
            self._pack_matrix_rows(section, item_offset + 0x00, rows)
            self._pack_matrix_rows(section, item_offset + 0x40, rows)
            self._pack_matrix_rows(section, item_offset + 0x80, inv_rows)
            section.write_pointer_at(item_offset + 0xC0, bg_objects_section, int(bg_object_offsets[int(instance.bg_object_index)]))
            section.pack_at(item_offset + 0xC4, '<I', int(getattr(instance, 'flags', 0) or 0) & 0xFFFFFFFF)
            section.pack_at(item_offset + 0xC8, '<I', 0)
            multi_spline_offset = self._write_multi_spline_data(section, getattr(instance, 'multi_spline_data', None))
            if multi_spline_offset is not None:
                section.write_pointer_at(item_offset + 0xC8, section.name, int(multi_spline_offset))
            section.pack_at(item_offset + 0xCC, '<f', float(getattr(instance, 'original_radius', 0.0) or 0.0))
            section.pack_at(item_offset + 0xD0, '<f', float(getattr(instance, 'radius', 0.0) or 0.0))
            section.pack_at(item_offset + 0xD4, '<H', self._u16(getattr(instance, 'instance_id', order_index)))
            section.pack_at(item_offset + 0xD6, '<H', self._u16(getattr(instance, 'bg_flags', 0)))
            section.pack_at(item_offset + 0xD8, '<h', self._i16(getattr(instance, 'target_frame', 0)))
            section.pack_at(item_offset + 0xDA, '<h', self._i16(getattr(instance, 'clip_beg', 0)))
            section.pack_at(item_offset + 0xDC, '<h', self._i16(getattr(instance, 'clip_end', 0)))
            section.pack_at(item_offset + 0xDE, '<h', self._i16(getattr(instance, 'link_seg', 0)))
            section.pack_at(item_offset + 0xE0, '<I', 0)
            section.pack_at(item_offset + 0xE4, '<h', self._i16(getattr(instance, 'color_data_index', 0)))
            section.pack_at(item_offset + 0xE6, '<h', self._i16(getattr(instance, 'lod', 0)))
            section.pack_at(item_offset + 0xE8, '<I', 0)
            section.pack_at(item_offset + 0xEC, '<I', int(getattr(instance, 'active_light_bitfield', 0) or 0) & 0xFFFFFFFF)
        return offset

    @staticmethod
    def _coerce_matrix_rows(rows: Sequence[Sequence[float]]) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        identity_rows = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        result = []
        try:
            for row_index in range(4):
                row = rows[row_index]
                result.append(tuple(float(row[col_index]) for col_index in range(4)))
            return tuple(result)
        except Exception:
            return default

    @staticmethod
    def _build_bginstance_inverse_rows(
        rows: Sequence[Sequence[float]],
        scale: Sequence[float],
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        rows = TRLevelDRMWriter._coerce_matrix_rows(rows)
        sx = float(scale[0]) if len(scale) > 0 else 1.0
        sy = float(scale[1]) if len(scale) > 1 else 1.0
        sz = float(scale[2]) if len(scale) > 2 else 1.0
        return (
            (float(rows[0][0]), float(rows[1][0]), float(rows[2][0]), 0.0),
            (float(rows[0][1]), float(rows[1][1]), float(rows[2][1]), 0.0),
            (float(rows[0][2]), float(rows[1][2]), float(rows[2][2]), 0.0),
            (sx, sy, sz, 1.0),
        )

    @staticmethod
    def _pack_matrix_rows(section: _SectionBuffer, offset: int, rows: Sequence[Sequence[float]]) -> None:
        rows = TRLevelDRMWriter._coerce_matrix_rows(rows)
        flat = []
        for row in rows:
            flat.extend(float(value) for value in row)
        section.pack_at(offset, '<16f', *flat)


    def _write_unit_data_section(self, section: _SectionBuffer, metadata: Dict[str, object], *, include_psp_data: bool = True) -> int:
        meta = dict(metadata or {})
        overlay_count = max(0, int(meta.get('overlay_num_effects', 0) or 0))
        fsfx_links = [entry for entry in list(meta.get('fsfx_links') or []) if isinstance(entry, dict)]
        cine_entries = [
            entry
            for entry in list(meta.get('cine_entries') or [])
            if isinstance(entry, dict) and str(entry.get('export_section_name', '') or entry.get('section_name', '') or '').strip()
        ]
        event_variable_entries = [entry for entry in list(meta.get('event_variable_storage_entries') or []) if isinstance(entry, dict)]
        if include_psp_data:
            base_camera_vehicle_offset = None
            base_camera_death_offset = 412
            script_offset = 416
            event_count_offset = 420
            event_pointer_offset = 424
            nextgen_base = 428
            psp_base = 460
            overlay_base = 472
        else:
            base_camera_vehicle_offset = 412
            base_camera_death_offset = 416
            script_offset = 420
            event_count_offset = 424
            event_pointer_offset = 428
            nextgen_base = 432
            psp_base = 464
            overlay_base = 464
        overlay_count_offset = overlay_base + 4
        overlay_pointer_offset = overlay_base + 8
        overlay_effect_offset = overlay_base + 12
        section_size = overlay_effect_offset + (overlay_count * 48)
        offset = section.reserve(section_size, alignment=16)
        num_fsfx = len(fsfx_links)
        num_cines = len(cine_entries)
        num_event_variable_storage = len(event_variable_entries)
        section.pack_at(offset + 0, '<I', num_fsfx)
        section.pack_at(offset + 8, '<I', num_cines)
        if num_cines <= 0:
            section.write_pointer_at(offset + 12, section.name, len(section.data))
        else:
            cines_offset = section.reserve(num_cines * 4, alignment=4)
            section.write_pointer_at(offset + 12, section.name, cines_offset)
            for index, entry in enumerate(cine_entries):
                target_section_name = str(entry.get('export_section_name', '') or entry.get('section_name', '') or '')
                try:
                    target_offset = int(entry.get('target_offset', 0) or 0)
                except Exception:
                    target_offset = 0
                if target_section_name:
                    section.write_pointer_at(cines_offset + (index * 4), target_section_name, target_offset)
                else:
                    section.pack_at(cines_offset + (index * 4), '<I', 0)

        section.pack_at(offset + 16, '<BBB', self._u8(self._tuple_get(meta.get('depth_backcolor'), 0)), self._u8(self._tuple_get(meta.get('depth_backcolor'), 1)), self._u8(self._tuple_get(meta.get('depth_backcolor'), 2)))
        section.pack_at(offset + 19, '<BBB', self._u8(self._tuple_get(meta.get('depth_watercolor'), 0)), self._u8(self._tuple_get(meta.get('depth_watercolor'), 1)), self._u8(self._tuple_get(meta.get('depth_watercolor'), 2)))
        section.pack_at(offset + 22, '<BBB', self._u8(self._tuple_get(meta.get('depth_ambientcolor'), 0)), self._u8(self._tuple_get(meta.get('depth_ambientcolor'), 1)), self._u8(self._tuple_get(meta.get('depth_ambientcolor'), 2)))
        section.pack_at(offset + 25, '<BBB', self._u8(self._tuple_get(meta.get('depth_objectambientcolor'), 0)), self._u8(self._tuple_get(meta.get('depth_objectambientcolor'), 1)), self._u8(self._tuple_get(meta.get('depth_objectambientcolor'), 2)))
        section.pack_at(offset + 28, '<B', self._u8(meta.get('depth_waterblend', 0)))
        section.pack_at(offset + 32, '<f', float(meta.get('depth_fogfar', 0.0) or 0.0))
        section.pack_at(offset + 36, '<f', float(meta.get('depth_fogmax', 0.0) or 0.0))
        section.pack_at(offset + 40, '<f', float(meta.get('depth_fognear', 0.0) or 0.0))
        section.pack_at(offset + 44, '<f', float(meta.get('depth_waterfogfar', 0.0) or 0.0))
        section.pack_at(offset + 48, '<f', float(meta.get('depth_waterfogmax', 0.0) or 0.0))
        section.pack_at(offset + 52, '<f', float(meta.get('depth_waterfognear', 0.0) or 0.0))
        section.pack_at(offset + 56, '<i', int(meta.get('depth_underwaterfxalpha', 0) or 0))
        section.pack_at(offset + 60, '<f', float(meta.get('depth_underwaterfxmovement', 0.0) or 0.0))
        section.pack_at(offset + 64, '<f', float(meta.get('depth_underwaterfxspeed', 0.0) or 0.0))
        section.pack_at(offset + 68, '<f', float(meta.get('depth_underwaterfxrandscale', 0.0) or 0.0))
        section.pack_at(offset + 72, '<f', float(meta.get('depth_underwaterfxscale', 0.0) or 0.0))
        section.pack_at(offset + 76, '<BBB', self._u8(self._tuple_get(meta.get('depth_underwaterfxcolor'), 0)), self._u8(self._tuple_get(meta.get('depth_underwaterfxcolor'), 1)), self._u8(self._tuple_get(meta.get('depth_underwaterfxcolor'), 2)))
        section.pack_at(offset + 79, '<B', self._pack_fx_flags(meta.get('depth_underwaterfxadditive', 0), meta.get('depth_underwaterfxbeforezsort', 0), meta.get('depth_underwaterfxvertalpha', 0)))
        section.pack_at(offset + 80, '<BBB', self._u8(self._tuple_get(meta.get('depth_underwaterplayercolor'), 0)), self._u8(self._tuple_get(meta.get('depth_underwaterplayercolor'), 1)), self._u8(self._tuple_get(meta.get('depth_underwaterplayercolor'), 2)))
        section.pack_at(offset + 84, '<i', int(meta.get('depth_waterfxalpha', 0) or 0))
        section.pack_at(offset + 88, '<f', float(meta.get('depth_waterfxmovement', 0.0) or 0.0))
        section.pack_at(offset + 92, '<f', float(meta.get('depth_waterfxspeed', 0.0) or 0.0))
        section.pack_at(offset + 96, '<f', float(meta.get('depth_waterfxrandscale', 0.0) or 0.0))
        section.pack_at(offset + 100, '<f', float(meta.get('depth_waterfxscale', 0.0) or 0.0))
        section.pack_at(offset + 104, '<BBB', self._u8(self._tuple_get(meta.get('depth_waterfxcolor'), 0)), self._u8(self._tuple_get(meta.get('depth_waterfxcolor'), 1)), self._u8(self._tuple_get(meta.get('depth_waterfxcolor'), 2)))
        section.pack_at(offset + 107, '<B', self._pack_fx_flags(meta.get('depth_waterfxadditive', 0), meta.get('depth_waterfxbeforezsort', 0), meta.get('depth_waterfxvertalpha', 0)))

        bend_names = ('black', 'blue', 'green', 'bluegreen', 'red', 'redblue', 'redgreen', 'redgreenblue')
        bend_offset = offset + 108
        for bend_index, bend_name in enumerate(bend_names):
            item_offset = bend_offset + (bend_index * 36)
            self._pack_vec3f_raw(section, item_offset + 0, meta.get(f'bend_{bend_name}_offset'))
            self._pack_vec3f_raw(section, item_offset + 12, meta.get(f'bend_{bend_name}_time_offset'))
            self._pack_vec3f_raw(section, item_offset + 24, meta.get(f'bend_{bend_name}_time_scale'))

        section.pack_at(offset + 396, '<i', int(meta.get('base_camera_use_camera_stack_system', 0) or 0))
        section.pack_at(offset + 400, '<I', int(meta.get('base_camera_basecam', 0) or 0))
        section.pack_at(offset + 404, '<I', int(meta.get('base_camera_ledgecam', 0) or 0))
        section.pack_at(offset + 408, '<I', int(meta.get('base_camera_crawlcam', 0) or 0))
        if base_camera_vehicle_offset is not None:
            section.pack_at(offset + base_camera_vehicle_offset, '<I', int(meta.get('base_camera_vehiclecam', 0) or 0))
        section.pack_at(offset + base_camera_death_offset, '<I', int(meta.get('base_camera_deathcam', 0) or 0))
        section.pack_at(offset + script_offset, '<I', int(meta.get('script', 0) or 0))
        section.pack_at(offset + event_count_offset, '<I', num_event_variable_storage)
        section.pack_at(offset + nextgen_base + 0, '<f', float(meta.get('nextgen_vertex_color_percent', 0.0) or 0.0))
        section.pack_at(offset + nextgen_base + 4, '<BBB', self._u8(self._tuple_get(meta.get('nextgen_global_ambient_color'), 0)), self._u8(self._tuple_get(meta.get('nextgen_global_ambient_color'), 1)), self._u8(self._tuple_get(meta.get('nextgen_global_ambient_color'), 2)))
        section.pack_at(offset + nextgen_base + 8, '<f', float(meta.get('nextgen_global_ambient_intensity', 0.0) or 0.0))
        section.pack_at(offset + nextgen_base + 12, '<bBBB', self._i8(meta.get('nextgen_override_fog_color', 0)), self._u8(self._tuple_get(meta.get('nextgen_fog_color'), 0)), self._u8(self._tuple_get(meta.get('nextgen_fog_color'), 1)), self._u8(self._tuple_get(meta.get('nextgen_fog_color'), 2)))
        section.pack_at(offset + nextgen_base + 16, '<f', float(meta.get('nextgen_fog_start', 0.0) or 0.0))
        section.pack_at(offset + nextgen_base + 20, '<f', float(meta.get('nextgen_fog_end', 0.0) or 0.0))
        section.pack_at(offset + nextgen_base + 24, '<bb', self._i8(meta.get('nextgen_enable_pls_spot', 0)), self._i8(meta.get('nextgen_enable_pls_spot_shadows', 0)))
        section.pack_at(offset + nextgen_base + 28, '<f', float(meta.get('nextgen_light_fade_time', 0.0) or 0.0))
        if include_psp_data:
            section.pack_at(offset + psp_base + 0, '<bBBB', self._i8(meta.get('psp_override_fog_color', 0)), self._u8(self._tuple_get(meta.get('psp_fog_color'), 0)), self._u8(self._tuple_get(meta.get('psp_fog_color'), 1)), self._u8(self._tuple_get(meta.get('psp_fog_color'), 2)))
            section.pack_at(offset + psp_base + 4, '<f', float(meta.get('psp_fog_start', 0.0) or 0.0))
            section.pack_at(offset + psp_base + 8, '<f', float(meta.get('psp_fog_end', 0.0) or 0.0))
        section.pack_at(offset + overlay_base, '<b', self._i8(meta.get('overlay_snow_level', 0)))
        section.pack_at(offset + overlay_count_offset, '<I', overlay_count)
        if overlay_count <= 0:
            section.write_pointer_at(offset + overlay_pointer_offset, section.name, len(section.data))
        else:
            section.write_pointer_at(offset + overlay_pointer_offset, section.name, offset + overlay_effect_offset)
            effect_offset = offset + overlay_effect_offset
            for overlay_index in range(overlay_count):
                item_offset = effect_offset + (overlay_index * 48)
                if overlay_index != 0:
                    continue
                section.pack_at(item_offset + 0, '<i', int(meta.get('overlay_effect_identification', 0) or 0))
                section.pack_at(item_offset + 4, '<h', self._i16(meta.get('overlay_effect_type', 0)))
                section.pack_at(item_offset + 8, '<f', float(meta.get('overlay_effect_feet', 0.0) or 0.0))
                section.pack_at(item_offset + 12, '<f', float(meta.get('overlay_effect_legs', 0.0) or 0.0))
                section.pack_at(item_offset + 16, '<f', float(meta.get('overlay_effect_torso', 0.0) or 0.0))
                section.pack_at(item_offset + 20, '<f', float(meta.get('overlay_effect_face', 0.0) or 0.0))
                section.pack_at(item_offset + 24, '<f', float(meta.get('overlay_effect_hair', 0.0) or 0.0))
                section.pack_at(item_offset + 28, '<f', float(meta.get('overlay_effect_feet_cap', 0.0) or 0.0))
                section.pack_at(item_offset + 32, '<f', float(meta.get('overlay_effect_legs_cap', 0.0) or 0.0))
                section.pack_at(item_offset + 36, '<f', float(meta.get('overlay_effect_torso_cap', 0.0) or 0.0))
                section.pack_at(item_offset + 40, '<f', float(meta.get('overlay_effect_face_cap', 0.0) or 0.0))
                section.pack_at(item_offset + 44, '<f', float(meta.get('overlay_effect_hair_cap', 0.0) or 0.0))

        if num_fsfx <= 0:
            section.write_pointer_at(offset + 4, section.name, len(section.data))
        else:
            fsfx_offset = section.reserve(num_fsfx * 12, alignment=4)
            section.write_pointer_at(offset + 4, section.name, fsfx_offset)
            for index, entry in enumerate(fsfx_links):
                item_offset = fsfx_offset + (index * 12)
                section.pack_at(
                    item_offset,
                    '<IfB3x',
                    int(entry.get('id', 0) or 0) & 0xFFFFFFFF,
                    float(entry.get('alpha', 0.0) or 0.0),
                    self._u8(1 if entry.get('enabled', 0) else 0),
                )

        if num_event_variable_storage <= 0:
            section.write_pointer_at(offset + event_pointer_offset, section.name, len(section.data))
        else:
            event_offset = section.reserve(num_event_variable_storage * 16, alignment=4)
            section.write_pointer_at(offset + event_pointer_offset, section.name, event_offset)
            for index, entry in enumerate(event_variable_entries):
                item_offset = event_offset + (index * 16)
                section.pack_at(
                    item_offset,
                    '<H2xiii',
                    self._u16(entry.get('variable', 0)),
                    int(entry.get('medium_value', 0) or 0),
                    int(entry.get('easy_value', 0) or 0),
                    int(entry.get('hard_value', 0) or 0),
                )
        return offset

    def _write_admd_section(self, section: _SectionBuffer, metadata: Dict[str, object]) -> int:
        meta = dict(metadata or {})
        count = max(0, int(meta.get('light_instance_count', 0) or 0))
        if count <= 0:
            offset = section.reserve(20, alignment=16)
            section.pack_at(offset + 0, '<I', 0)
            section.write_pointer_at(offset + 4, section.name, 16)
            section.pack_at(offset + 16, '<I', 0)
            return offset
        offset = section.reserve(16 + (count * 160), alignment=16)
        section.pack_at(offset + 0, '<I', count)
        section.write_pointer_at(offset + 4, section.name, 16)
        for index in range(count):
            base_key = f'light_{index:03d}_'
            item_offset = offset + 16 + (index * 160)
            self._pack_vec4f_raw(section, item_offset + 0, meta.get(f'{base_key}transform_row0'))
            self._pack_vec4f_raw(section, item_offset + 16, meta.get(f'{base_key}transform_row1'))
            self._pack_vec4f_raw(section, item_offset + 32, meta.get(f'{base_key}transform_row2'))
            self._pack_vec4f_raw(section, item_offset + 48, meta.get(f'{base_key}transform_row3'))
            section.pack_at(item_offset + 64, '<i', int(meta.get(f'{base_key}property_type', 0) or 0))
            section.pack_at(item_offset + 68, '<I', int(meta.get(f'{base_key}light_resource', 0) or 0))
            section.pack_at(item_offset + 72, '<f', float(meta.get(f'{base_key}intensity', 0.0) or 0.0))
            section.pack_at(item_offset + 76, '<f', float(meta.get(f'{base_key}rim_intensity', 0.0) or 0.0))
            section.pack_at(item_offset + 80, '<f', float(meta.get(f'{base_key}specular_intensity', 0.0) or 0.0))
            section.pack_at(item_offset + 84, '<B', self._u8(meta.get(f'{base_key}enable_per_instance_intensity', 0)))
            section.pack_at(item_offset + 85, '<BBB', self._u8(self._tuple_get(meta.get(f'{base_key}light_color'), 0)), self._u8(self._tuple_get(meta.get(f'{base_key}light_color'), 1)), self._u8(self._tuple_get(meta.get(f'{base_key}light_color'), 2)))
            section.pack_at(item_offset + 88, '<B', self._u8(meta.get(f'{base_key}enable_per_instance_color', 0)))
            section.pack_at(item_offset + 92, '<f', float(meta.get(f'{base_key}range', 0.0) or 0.0))
            section.pack_at(item_offset + 96, '<B', self._u8(meta.get(f'{base_key}enable_per_instance_range', 0)))
            section.pack_at(item_offset + 97, '<BBB', self._u8(self._tuple_get(meta.get(f'{base_key}ambient_color'), 0)), self._u8(self._tuple_get(meta.get(f'{base_key}ambient_color'), 1)), self._u8(self._tuple_get(meta.get(f'{base_key}ambient_color'), 2)))
            section.pack_at(item_offset + 100, '<f', float(meta.get(f'{base_key}ambient_percentage', 0.0) or 0.0))
            section.pack_at(item_offset + 104, '<B', self._u8(meta.get(f'{base_key}enable_per_instance_ambient', 0)))
            section.pack_at(item_offset + 105, '<B', self._u8(meta.get(f'{base_key}enable_per_instance_shadow_bias', 0)))
            section.pack_at(item_offset + 108, '<f', float(meta.get(f'{base_key}shadow_map_bias', 0.0) or 0.0))
            section.pack_at(item_offset + 112, '<f', float(meta.get(f'{base_key}shadow_map_slope_bias', 0.0) or 0.0))
            section.pack_at(item_offset + 116, '<f', float(meta.get(f'{base_key}cull_light_distance', 0.0) or 0.0))
            section.pack_at(item_offset + 120, '<B', self._u8(meta.get(f'{base_key}enable_light_culling', 0)))
            section.pack_at(item_offset + 124, '<f', float(meta.get(f'{base_key}cull_light_fade_distance', 0.0) or 0.0))
            section.pack_at(item_offset + 128, '<f', float(meta.get(f'{base_key}shadow_cull_distance', 0.0) or 0.0))
            section.pack_at(item_offset + 132, '<B', self._u8(meta.get(f'{base_key}enable_shadow_lod', 0)))
            section.pack_at(item_offset + 133, '<B', self._u8(meta.get(f'{base_key}active_in_gameplay', 0)))
            section.pack_at(item_offset + 134, '<B', self._u8(meta.get(f'{base_key}active_in_cinematics', 0)))
            section.pack_at(item_offset + 135, '<B', self._u8(meta.get(f'{base_key}affects_player', 0)))
            section.pack_at(item_offset + 136, '<B', self._u8(meta.get(f'{base_key}affects_intros', 0)))
            section.pack_at(item_offset + 137, '<B', self._u8(meta.get(f'{base_key}affects_bgobjects', 0)))
            section.pack_at(item_offset + 138, '<B', self._u8(meta.get(f'{base_key}affects_terrain', 0)))
            section.pack_at(item_offset + 139, '<B', self._u8(meta.get(f'{base_key}affects_neighboring_units', 0)))
            section.pack_at(item_offset + 140, '<B', self._u8(meta.get(f'{base_key}affects_water', 0)))
            cinematic_name = str(meta.get(f'{base_key}cinematic_name', '') or '')
            if cinematic_name:
                string_offset = section.append(cinematic_name.encode('utf-8', errors='ignore') + b'\x00', alignment=4)
                section.write_pointer_at(item_offset + 144, section.name, string_offset)
            section.pack_at(item_offset + 148, '<I', int(meta.get(f'{base_key}scene_light', 0) or 0))
        return offset

    def _write_vertex_buffer(self, section: _SectionBuffer, groups: Sequence[ExportTerrainGroup]) -> int:
        vertex_count = sum(len(strip.vertices) for group in groups for strip in group.strips)
        offset = section.reserve(max(1, vertex_count) * _TERRAIN_VERTEX_SIZE, alignment=16)
        cursor = offset
        for group in groups:
            for strip in group.strips:
                for vertex in strip.vertices:
                    raw_x, raw_y, raw_z = self._blender_to_raw_int_triplet(vertex.position)
                    u_raw = self._i16(round(float(vertex.uv[0]) / _UV_SCALE))
                    v_raw = self._i16(round(float(vertex.uv[1]) / _UV_SCALE))
                    red, green, blue, alpha = self._sanitize_color(vertex.color)
                    alpha = (alpha + 1) >> 1
                    color_bgra = (blue & 0xFF) | ((green & 0xFF) << 8) | ((red & 0xFF) << 16) | ((alpha & 0xFF) << 24)
                    section.pack_at(cursor + 0, '<hhh', raw_x, raw_y, raw_z)
                    section.pack_at(cursor + 6, '<h', 1)
                    section.pack_at(cursor + 8, '<I', color_bgra)
                    section.pack_at(cursor + 12, '<hh', u_raw, v_raw)
                    section.pack_at(cursor + 16, '<HH', 0, 0)
                    cursor += _TERRAIN_VERTEX_SIZE
        return offset

    def _write_c_string(self, section: _SectionBuffer, value: str) -> int:
        encoded = value.encode('utf-8', errors='ignore') + b'\x00'
        return section.append(encoded, alignment=4)

    @staticmethod
    def _pack_identity_matrix(section: _SectionBuffer, offset: int) -> None:
        section.pack_at(
            offset,
            '<16f',
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )

    @staticmethod
    def _pack_padded_vec3f(section: _SectionBuffer, offset: int, value: Sequence[float]) -> None:
        x, y, z = TRLevelDRMWriter._blender_to_raw_float_triplet(value[:3])
        section.pack_at(offset, '<3f', float(x), float(y), float(z))

    @staticmethod
    def _pack_raw_vec3f(section: _SectionBuffer, offset: int, value: Sequence[float]) -> None:
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        section.pack_at(offset, '<3f', x, y, z)

    @staticmethod
    def _blender_to_raw_float_triplet(value: Sequence[float]) -> tuple[float, float, float]:
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        return (-x, -y, z)

    @classmethod
    def _blender_to_raw_int_triplet(cls, value: Sequence[float]) -> tuple[int, int, int]:
        raw_x, raw_y, raw_z = cls._blender_to_raw_float_triplet(value)
        return cls._i16(round(raw_x)), cls._i16(round(raw_y)), cls._i16(round(raw_z))

    @staticmethod
    def _tuple_get(value: object, index: int, default: object = 0) -> object:
        if hasattr(value, 'to_list'):
            try:
                value = value.to_list()
            except Exception:
                pass
        if not isinstance(value, (str, bytes, bytearray, dict)) and not isinstance(value, Sequence) and hasattr(value, '__iter__'):
            try:
                value = tuple(value)
            except Exception:
                pass
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, dict)) and len(value) > index:
            return value[index]
        return default

    @staticmethod
    def _pack_fx_flags(additive: object, before_zsort: object, vert_alpha: object) -> int:
        return (int(additive) & 0x3) | ((int(before_zsort) & 0x3) << 2) | ((int(vert_alpha) & 0x3) << 4)

    @staticmethod
    def _pack_vec3f_raw(section: _SectionBuffer, offset: int, value: object) -> None:
        x = float(TRLevelDRMWriter._tuple_get(value, 0, 0.0))
        y = float(TRLevelDRMWriter._tuple_get(value, 1, 0.0))
        z = float(TRLevelDRMWriter._tuple_get(value, 2, 0.0))
        section.pack_at(offset, '<3f', x, y, z)

    @staticmethod
    def _pack_vec4f_raw(section: _SectionBuffer, offset: int, value: object) -> None:
        x = float(TRLevelDRMWriter._tuple_get(value, 0, 0.0))
        y = float(TRLevelDRMWriter._tuple_get(value, 1, 0.0))
        z = float(TRLevelDRMWriter._tuple_get(value, 2, 0.0))
        w = float(TRLevelDRMWriter._tuple_get(value, 3, 0.0))
        section.pack_at(offset, '<4f', x, y, z, w)

    @staticmethod
    def _sanitize_color(value: Sequence[int | float]) -> tuple[int, int, int, int]:
        result: List[int] = []
        for channel_index in range(4):
            channel = value[channel_index] if channel_index < len(value) else 255
            if isinstance(channel, float):
                channel = int(round(max(0.0, min(1.0, channel)) * 255.0)) if 0.0 <= channel <= 1.0 else int(round(channel))
            result.append(max(0, min(255, int(channel))))
        return result[0], result[1], result[2], result[3]

    @staticmethod
    def _u8(value: object) -> int:
        return max(0, min(0xFF, int(value)))

    @staticmethod
    def _u16(value: object) -> int:
        return max(0, min(0xFFFF, int(value)))

    @staticmethod
    def _i8(value: object) -> int:
        value = int(value)
        return max(-0x80, min(0x7F, value))

    @staticmethod
    def _i16(value: object) -> int:
        value = int(value)
        return max(-0x8000, min(0x7FFF, value))

    @staticmethod
    def _i32(value: object) -> int:
        value = int(value)
        return max(-0x80000000, min(0x7FFFFFFF, value))

    @staticmethod
    def _u32(value: object) -> int:
        return max(0, min(0xFFFFFFFF, int(value)))