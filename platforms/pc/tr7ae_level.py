from __future__ import annotations

import base64
import json
import math
import mathutils
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import bpy

from ...builders.mesh_builder import MeshBuildSettings, MeshBuilder, _decode_tpage_flags
from ...core.blender_mesh_utils import get_or_create_color_attribute, set_active_color_attribute
from .texture import (
    D3DFMT_A8R8G8B8,
    D3DFMT_X8R8G8B8,
    _convert_pcd_argb32_to_rgba,
    _create_dds_header,
)
from ...core.log import logger
from ...core.cine_format import cine_name_from_section, parse_cine_payload
from ...core.material_ui import make_material_name_from_tpageid, initialize_material_flag_properties, material_uses_vertex_colors, set_material_panel_value
from ...core.level_combat import ROOT_COMBAT_FIELDS, rebuild_combat_summary
from ..common.section import SectionContext, SectionContextCache, resolve_pointer

def _int_or(value, default=0):
    try:
        return int(default) if value is None else int(value)
    except Exception:
        return int(default)


_LEVEL_VERSION = 0x04C204BB
_UV_SCALE = 0.00024414062
RELOCATION_POINTER = 0
_BGOBJECT_MODEL_SIZE = 96
_BGOBJECT_STRIDE_CANDIDATES = (96, 80, 112, 128)
_BGOBJECT_POSITION_CANDIDATE_OFFSETS = (16,)
_BGOBJECT_BLOB_CHUNK_SIZE = 30000
_BGOBJECT_BLOB_PROP_PREFIX = 'trlau_bgobject_blob_'
_GENERIC_INTRO_STRUCT = '<ffBBHHhfi'
_GENERIC_INTRO_SIZE = struct.calcsize(_GENERIC_INTRO_STRUCT)
_INTRO_DATA_HEADER_SIZE = 4 + _GENERIC_INTRO_SIZE
_REWARD_INTRO_STRUCT = '<hhi'
_WATER_VOLUME_INTRO_STRUCT = '<fhhffhhfhH'
_ROPE_RENDER_PARAMS_STRUCT = '<fffIBBBB'
_ROPE_OBJ_INTRO_PREFIX_STRUCT = '<BBHBBHBBHH2xiIff'
_INTRO_SPECIFIC_SIZES = {
    12: struct.calcsize(_ROPE_OBJ_INTRO_PREFIX_STRUCT) + struct.calcsize(_ROPE_RENDER_PARAMS_STRUCT),
    13: struct.calcsize(_WATER_VOLUME_INTRO_STRUCT),
    17: struct.calcsize(_REWARD_INTRO_STRUCT),
}

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
_SIGNAL_MAX_IMPORT_COUNT = 4096
_SIGNAL_SPLINE_CAMERA_POINTER_OFFSETS = (4, 8, 12)
_SIGNAL_ATTACK_WAVE_POINTER_OFFSETS = {
    'enableAttackWaves': 168,
    'disableAttackWaves': 172,
    'killAttackWaves': 176,
}
_SIGNAL_CAMERA_LINK_POINTER_OFFSET = 192
_STREAM_UNIT_PORTAL_SIZE = 160
_STREAM_UNIT_PORTAL_MAX_IMPORT_COUNT = 4096
_ATTACK_WAVE_RUNTIME_SIZE = 804
_ATTACK_WAVE_RUNTIME_NEXT_POINTER_OFFSETS = {
    'enableAttackWaves': 24,
    'disableAttackWaves': 28,
    'killAttackWaves': 32,
}
_ATTACK_WAVE_CHAIN_MAX_IMPORT_COUNT = 64
_SIGNAL_FLAG_BITS = (
    ('pad', 0, 0x003F),
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

_MULTI_SPLINE_SIZE = 0x64
_SPLINE_HEADER_SIZE = 0x08
_SPLINE_KEY_SIZE = 0x40
_RSPLINE_KEY_SIZE = 0x20
_MULTI_SPLINE_MAX_KEYS = 4096

_SFX_MARKER_SIZE = 260
_SFX_MARKER_MAX_IMPORT_COUNT = 4096
_SFX_POINTER_ARRAY_CAPACITY = 16
_SFX_DATA_SIZE = 40
_SFX_EVENT_SOUND_BASE_SIZE = 24
_SFX_PERIODIC_SOUND_BASE_SIZE = 28
_SFX_ONESHOT_SOUND_BASE_SIZE = 16
_SFX_STREAM_SOUND_SIZE = 76
_SFX_PERIMETER_SIZE = 28



def _clear_bgobject_raw_blob_metadata(obj) -> None:
    try:
        for key in list(obj.keys()):
            if str(key).startswith(_BGOBJECT_BLOB_PROP_PREFIX):
                del obj[key]
        for key in ('trlau_bgobject_blob_encoding', 'trlau_bgobject_blob_chunk_count', 'trlau_bgobject_blob_size'):
            if key in obj:
                del obj[key]
    except Exception:
        pass


def _encode_bgobject_raw_blob(bg_object) -> bytes:
    raw_vertices = bytes(getattr(bg_object, 'raw_vertex_data', b'') or b'')
    raw_colors = bytes(getattr(bg_object, 'raw_color_data', b'') or b'')
    strips = list(getattr(bg_object, 'strips', []) or [])
    if not raw_vertices or not strips:
        return b''
    env_indices = []
    eye_indices = []
    blob = bytearray(b'BGO3')
    vertex_count = len(raw_vertices) // 12
    color_slot_count = max(1, int(getattr(bg_object, 'raw_color_slot_count', 1) or 1))
    blob.extend(struct.pack('<IIIIIII', vertex_count, len(raw_vertices), len(raw_colors), color_slot_count, len(strips), len(env_indices), len(eye_indices)))
    blob.extend(raw_vertices)
    blob.extend(raw_colors)
    for value in env_indices:
        blob.extend(struct.pack('<H', value))
    for value in eye_indices:
        blob.extend(struct.pack('<H', value))
    for strip in strips:
        indices = [int(value) & 0xFFFF for value in list(getattr(strip, 'indices', []) or [])]
        sort_vertex = tuple(getattr(strip, 'sort_vertex', (0, 0, 0)) or (0, 0, 0))
        sx = int(sort_vertex[0]) if len(sort_vertex) > 0 else 0
        sy = int(sort_vertex[1]) if len(sort_vertex) > 1 else 0
        sz = int(sort_vertex[2]) if len(sort_vertex) > 2 else 0
        try:
            stored_count = int(getattr(strip, 'raw_count', -1))
        except Exception:
            stored_count = -1
        if stored_count < 0:
            stored_count = 0 if bool(getattr(strip, 'is_terminator', False)) else len(indices)
        if stored_count <= 0:
            indices = []
        blob.extend(struct.pack('<i', stored_count))
        blob.extend(struct.pack('<hhhH', sx, sy, sz, 0))
        blob.extend(struct.pack('<IifI', int(getattr(strip, 'tpageid', 0) or 0) & 0xFFFFFFFF, int(getattr(strip, 'sort_push', 0) or 0), float(getattr(strip, 'scroll_offset', 0.0) or 0.0), len(indices)))
        for value in indices:
            blob.extend(struct.pack('<H', value))
    return bytes(blob)


def _store_bgobject_raw_blob_metadata(obj, bg_object) -> None:
    _clear_bgobject_raw_blob_metadata(obj)
    blob = _encode_bgobject_raw_blob(bg_object)
    if not blob:
        return
    encoded = base64.b64encode(blob).decode('ascii')
    chunk_count = 0
    for chunk_index, start in enumerate(range(0, len(encoded), _BGOBJECT_BLOB_CHUNK_SIZE)):
        obj[f'{_BGOBJECT_BLOB_PROP_PREFIX}{chunk_index:04d}'] = encoded[start:start + _BGOBJECT_BLOB_CHUNK_SIZE]
        chunk_count += 1
    obj['trlau_bgobject_blob_encoding'] = 'base64'
    obj['trlau_bgobject_blob_chunk_count'] = int(chunk_count)
    obj['trlau_bgobject_blob_size'] = int(len(blob))


def _rotate_level_geometry(position: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    return (x, -z, y)


def _convert_level_position(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return _rotate_level_geometry((-float(x), float(z), float(y)))


def _convert_raw_bounds_to_level(raw_min: Tuple[float, float, float], raw_max: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    corner_a = _convert_level_position(float(raw_min[0]), float(raw_min[1]), float(raw_min[2]))
    corner_b = _convert_level_position(float(raw_max[0]), float(raw_max[1]), float(raw_max[2]))
    return (
        tuple(min(corner_a[index], corner_b[index]) for index in range(3)),
        tuple(max(corner_a[index], corner_b[index]) for index in range(3)),
    )



def _populate_collision_kd_bounds_impl(collision) -> None:
    nodes = list(getattr(collision, 'kd_nodes', []) or [])
    if not nodes:
        return

    visited: set[int] = set()

    def walk(node_index: int, raw_min: Tuple[float, float, float], raw_max: Tuple[float, float, float], depth: int) -> None:
        if node_index in visited or node_index < 0 or node_index >= len(nodes):
            return
        visited.add(node_index)
        node = nodes[node_index]
        node.depth = int(depth)
        node.bounds_min_raw = tuple(float(v) for v in raw_min)
        node.bounds_max_raw = tuple(float(v) for v in raw_max)
        node.face_start = int(node.ref_index) if node.is_leaf else 0
        if node.is_leaf:
            return
        axis = int(node.axis)
        if axis < 0 or axis > 2:
            return
        split_value = float(node.split_value)
        negative_min = list(raw_min)
        negative_max = list(raw_max)
        positive_min = list(raw_min)
        positive_max = list(raw_max)
        negative_max[axis] = min(float(negative_max[axis]), split_value)
        positive_min[axis] = max(float(positive_min[axis]), split_value)
        walk(node_index + 1, tuple(float(v) for v in negative_min), tuple(float(v) for v in negative_max), depth + 1)
        walk(node_index + int(node.ref_index), tuple(float(v) for v in positive_min), tuple(float(v) for v in positive_max), depth + 1)

    walk(0, tuple(float(v) for v in collision.raw_bbox_min), tuple(float(v) for v in collision.raw_bbox_max), 1)
    collision.kd_max_depth = max((int(node.depth) for node in nodes), default=0)


def _apply_bgobject_reflection_flags(bg_object) -> None:
    return


def _bgobject_has_render_data(bg_object) -> bool:
    if not getattr(bg_object, 'vertices', None):
        return False
    for strip in list(getattr(bg_object, 'strips', []) or []):
        if bool(getattr(strip, 'is_terminator', False)):
            continue
        if len(list(getattr(strip, 'indices', []) or [])) >= 3:
            return True
    return False


def _convert_bgobject_position(
    raw_x: float,
    raw_y: float,
    raw_z: float,
    scale_x: float,
    scale_y: float,
    scale_z: float,
) -> Tuple[float, float, float]:
    return (
        -float(raw_x) * float(scale_x),
        float(raw_z) * float(scale_z),
        float(raw_y) * float(scale_y),
    )


def _convert_bgobject_translation(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (-float(x), float(z), float(y))


def _convert_bgobject_basis_vector(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (-float(x), float(z), float(y))


def _convert_bginstance_matrix(rows: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]):
    import mathutils

    source_rows = mathutils.Matrix((
        rows[0],
        rows[1],
        rows[2],
        rows[3],
    ))
    source_matrix = source_rows.transposed()
    basis_change = mathutils.Matrix((
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    instance_matrix = basis_change @ source_matrix @ basis_change
    level_rotation = mathutils.Matrix.Rotation(math.radians(90.0), 4, 'X')
    return level_rotation @ instance_matrix


def _make_bgobject_local_matrix(scale: Tuple[float, float, float]):
    import mathutils

    sx, sy, sz = (float(scale[0]), float(scale[1]), float(scale[2]))
    rotation_basis = mathutils.Matrix((
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    scale_matrix = mathutils.Matrix.Diagonal((sx, sy, sz, 1.0))
    return rotation_basis @ scale_matrix




@dataclass(slots=True)
class LevelVertex:
    position: Tuple[float, float, float]
    uv: Tuple[float, float]


@dataclass(slots=True)
class LevelStrip:
    material_index: int
    tpageid: int
    flags: int
    vertex_base_offset: int
    indices: List[int] = field(default_factory=list)
    scroll_speeds: List[float] = field(default_factory=list)
    scroll_num_tiles: int = 0
    scroll_tile: int = 0
    scroll_entry_count: int = 0
    sort_vertex: Tuple[int, int, int] = (0, 0, 0)
    sort_push: int = 0
    scroll_offset: float = 0.0
    raw_count: int = -1
    is_terminator: bool = False
    env_mapping: bool = False
    eye_ref_env_mapping: bool = False

    @property
    def uses_vmo_buffer(self) -> bool:
        return (int(self.flags) & 0x1C) != 0

    @property
    def scroll_speed(self) -> Optional[float]:
        return self.scroll_speeds[0] if self.scroll_speeds else None

    @property
    def has_scroll_animation(self) -> bool:
        return bool(self.scroll_speeds) or (int(self.flags) == 1)



def _resolve_group_strip_material_index(group: TerrainGroup, raw_value: int, resolved_abs: Optional[int] = None) -> Optional[int]:
    entry_offsets = list(getattr(group, 'material_entry_offsets', []) or [])
    if resolved_abs is not None:
        try:
            return entry_offsets.index(int(resolved_abs))
        except ValueError:
            pass

    raw_u32 = int(raw_value) & 0xFFFFFFFF
    try:
        return entry_offsets.index(int(raw_u32))
    except ValueError:
        pass

    if 0 <= raw_u32 < len(group.strips):
        return int(raw_u32)

    raw_i32 = raw_u32 - 0x100000000 if (raw_u32 & 0x80000000) else raw_u32
    if 0 <= raw_i32 < len(group.strips):
        return int(raw_i32)

    return None


@dataclass(slots=True)
class TerrainCollisionFace:
    i0: int
    i1: int
    i2: int
    adjacency_flags: int = 0
    collision_flags: int = 0
    client_flags: int = 0
    material_type: int = 0
    signal_id: int = -1


@dataclass(slots=True)
class TerrainCollisionKDNode:
    node_index: int
    neg_offset: float = 0.0
    pos_offset: float = 0.0
    ref_index: int = 0
    axis: int = 0
    num_faces: int = 0
    depth: int = 0
    face_start: int = 0
    bounds_min_raw: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max_raw: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_leaf(self) -> bool:
        return int(self.num_faces) > 0

    @property
    def split_value(self) -> float:
        return (float(self.neg_offset) + float(self.pos_offset)) * 0.5


@dataclass(slots=True)
class TerrainGroupCollision:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[TerrainCollisionFace] = field(default_factory=list)
    kd_nodes: List[TerrainCollisionKDNode] = field(default_factory=list)
    kd_max_depth: int = 0


@dataclass(slots=True)
class TerrainSignalMesh:
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_bbox_min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    raw_bbox_max: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[TerrainCollisionFace] = field(default_factory=list)
    vertex_type: int = 0


@dataclass(slots=True)
class TerrainSignal:
    index: int
    signal_id: int
    mesh: TerrainSignalMesh
    face_id: int = -1
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TerrainGroup:
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
    collision: Optional[TerrainGroupCollision] = None
    strips: List[LevelStrip] = field(default_factory=list)
    material_entry_offsets: List[int] = field(default_factory=list)
    sorted_material_indices: List[int] = field(default_factory=list)


@dataclass(slots=True)
class BGObject:
    index: int
    scale: Tuple[float, float, float]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    stride: int = _BGOBJECT_MODEL_SIZE
    flags: int = 0
    cdc_render_data_id: int = 0
    position_candidates: List[Tuple[int, Tuple[float, float, float]]] = field(default_factory=list)
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    vertex_colors: List[Tuple[int, int, int, int]] = field(default_factory=list)
    strips: List[LevelStrip] = field(default_factory=list)
    raw_vertex_data: bytes = b''
    raw_color_data: bytes = b''
    raw_color_slot_count: int = 1
    env_mapped_vertices: List[int] = field(default_factory=list)
    eye_ref_env_mapped_vertices: List[int] = field(default_factory=list)


@dataclass(slots=True)
class BGInstance:
    index: int
    bg_object_index: int
    bg_object_offset: int
    instance_id: int
    flags: int = 0
    multi_spline_offset: int = 0
    multi_spline_data: Optional[dict[str, object]] = None
    original_radius: float = 0.0
    radius: float = 0.0
    bg_flags: int = 0
    target_frame: int = 0
    clip_beg: int = 0
    clip_end: int = 0
    link_seg: int = 0
    link_instance_offset: int = 0
    color_data_index: int = 0
    lod: int = 0
    active_light_bitfield: int = 0
    matrix_rows: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]] = ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))


@dataclass(slots=True)
class TerrainLightData:
    index: int
    position: Tuple[float, float, float]
    radius: int
    color: Tuple[float, float, float]
    type: int
    multiplier: int
    hotspot_angle: int
    direction: Tuple[float, float, float]
    falloff_angle: int
    light_id: int


@dataclass(slots=True)
class CameraKeyData:
    index: int
    position: Tuple[float, float, float]
    camera_id: int
    rotation: Tuple[int, int, int]
    flags: int
    target: Tuple[int, int, int]


@dataclass(slots=True)
class CameraAnticData:
    use_antic_camera: int = 0
    new_dtp_camera_antic_data_id: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)


@dataclass(slots=True)
class MarkupData:
    index: int
    game: str = 'unknown'
    override_movement_camera: int = 0
    dtp_camera_data_id: int = 0
    dtp_markup_data_id: int = 0
    animated_segment: int = 0
    camera_antic: Optional[CameraAnticData] = None
    flags: int = 0
    intro_id: int = -1
    markup_id: int = -1
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    polyline_offset: int = 0
    polyline: List[Tuple[float, float, float, float]] = field(default_factory=list)




@dataclass(slots=True)
class GenericIntroData:
    in_view_remove_dist: float = 0.0
    out_of_view_remove_dist: float = 0.0
    use_model: int = 0
    pad0: int = 0
    pad1: int = 0
    flags: int = 0
    attached_instance: int = 0
    swing_length: float = 0.0
    dtp_camera_id: int = 0


@dataclass(slots=True)
class IntroDataBlock:
    data_type: int = 0
    generic: Optional[GenericIntroData] = None
    specific: dict[str, object] = field(default_factory=dict)
    raw_specific: bytes = b''
    source_offset: int = 0
    source_raw_value: int = 0

@dataclass(slots=True)
class IntroData:
    index: int
    object_id: int
    intro_num: int
    unique_id: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, ...]
    dummy1: Tuple[float, float, float, float]
    dummy2: Tuple[float, float, float, float]
    scale: Tuple[float, float, float, float]
    start_frame: int
    end_frame: int
    intro_flags: int
    attached_vmo: int
    data: int
    multi_spline: int
    max_radius: float
    intro_data_block: Optional[IntroDataBlock] = None
    multi_spline_data: Optional[dict[str, object]] = None


@dataclass(slots=True)
class RelocModuleData:
    target_offset: int = 0
    section_index: int = -1
    section_type: int = 0
    section_id: int = 0
    skip_flags: int = 0
    version_id: int = 0
    has_debug_info: int = 0
    resource_type: int = 0
    spec_mask: int = 0xFFFFFFFF
    file_name: str = ''
    section_blob: bytes = b''


@dataclass(slots=True)
class LevelSFXSound:
    index: int
    kind: str
    source_offset: int
    sound_ids: List[int] = field(default_factory=list)
    properties: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LevelSFXPerimeter:
    index: int
    source_offset: int
    radius: int = 0
    n_fired: int = 0
    invader_offset: int = 0
    action: int = 0
    breached: int = 0
    on_exit: int = 0
    always: int = 0
    event_offset: int = 0
    periodic_offset: int = 0
    varnum: int = 0
    value: int = 0


@dataclass(slots=True)
class LevelSFXMarker:
    index: int
    source_offset: int
    position: Tuple[float, float, float]
    raw_position: Tuple[float, float, float]
    unique_id: int
    plane: int = 0
    spline_id: int = 0
    sound_instance_offset: int = 0
    periodic_sounds: List[LevelSFXSound] = field(default_factory=list)
    event_sounds: List[LevelSFXSound] = field(default_factory=list)
    one_shot_sounds: List[LevelSFXSound] = field(default_factory=list)
    stream_sounds: List[LevelSFXSound] = field(default_factory=list)
    perimeter_actions: List[LevelSFXPerimeter] = field(default_factory=list)


@dataclass(slots=True)
class LevelData:
    filepath: str
    texture_images: dict[int, str] = field(default_factory=dict)
    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    scene_center_offset: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    level_metadata: dict[str, object] = field(default_factory=dict)
    reloc_module: Optional[RelocModuleData] = None
    unit_data: dict[str, object] = field(default_factory=dict)
    admd_data: dict[str, object] = field(default_factory=dict)
    level_flags: int = 0
    unit_flags: int = 0
    stream_unit_id: int = 0
    player_object_id: int = -1
    stream_unit_portals: List[dict[str, object]] = field(default_factory=list)
    terrain_lights: List[TerrainLightData] = field(default_factory=list)
    sfx_markers: List[LevelSFXMarker] = field(default_factory=list)
    terrain_light_grid_cells: List[Tuple[int, Tuple[int, ...]]] = field(default_factory=list)
    camera_data: List[CameraKeyData] = field(default_factory=list)
    intro_data: List[IntroData] = field(default_factory=list)
    markups: List[MarkupData] = field(default_factory=list)
    source_game: str = 'unknown'
    cdc_render_data_id: int = 0
    uvs: List[Tuple[float, float]] = field(default_factory=list)
    vertex_colors: List[Tuple[int, int, int, int]] = field(default_factory=list)
    vmo_vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    vmo_uvs: List[Tuple[float, float]] = field(default_factory=list)
    vmo_vertex_colors: List[Tuple[int, int, int, int]] = field(default_factory=list)
    terrain_groups: List[TerrainGroup] = field(default_factory=list)
    signal_mesh: Optional[TerrainSignalMesh] = None
    signals: List[TerrainSignal] = field(default_factory=list)
    terrain_signal_list_abs: int = 0
    terrain_signal_list_context: Optional[SectionContext] = None
    bg_objects: List[BGObject] = field(default_factory=list)
    bg_instances: List[BGInstance] = field(default_factory=list)
    combat_data: dict[str, object] = field(default_factory=dict)




def _is_finite_triplet(values: Tuple[float, float, float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _read_bgobject_position_candidate_from_bytes(data: bytes, absolute_offset: int) -> Optional[Tuple[float, float, float]]:
    if absolute_offset < 0 or (absolute_offset + 12) > len(data):
        return None
    values = struct.unpack_from('<3f', data, absolute_offset)
    if not _is_finite_triplet(values):
        return None
    return _convert_bgobject_translation(values[0], values[1], values[2])


def _pick_bgobject_position(
    scale: Tuple[float, float, float],
    candidates: List[Tuple[int, Tuple[float, float, float]]],
) -> Tuple[float, float, float]:
    if not candidates:
        return (0.0, 0.0, 0.0)

    scale_vec = tuple(float(v) for v in scale)

    def _score(item: Tuple[int, Tuple[float, float, float]]) -> Tuple[int, float, int]:
        offset, position = item
        pos = tuple(float(v) for v in position)
        magnitude = abs(pos[0]) + abs(pos[1]) + abs(pos[2])
        zeroish = magnitude < 1e-6
        looks_like_scale = all(abs(pos[i] - scale_vec[i]) < 1e-6 for i in range(3))
        preferred_offset = 0
        if offset == 16:
            preferred_offset = 3
        return (0 if zeroish else 1, 0.0 if looks_like_scale else magnitude, preferred_offset)

    best_offset, best_position = max(candidates, key=_score)
    if all(abs(float(v)) < 1e-6 for v in best_position):
        for offset, position in candidates:
            if offset == 16 and any(abs(float(v)) >= 1e-6 for v in position):
                return tuple(float(v) for v in position)
    return tuple(float(v) for v in best_position)


def _detect_bgobject_stride_from_bytes(data: bytes, list_offset: int, count: int) -> int:
    valid_count = max(1, min(int(count), 32))
    best_stride = _BGOBJECT_MODEL_SIZE
    best_score = -1
    data_len = len(data)
    for stride in _BGOBJECT_STRIDE_CANDIDATES:
        score = 0
        for bg_index in range(valid_count):
            entry_offset = list_offset + (bg_index * stride)
            if entry_offset < 0 or (entry_offset + _BGOBJECT_MODEL_SIZE) > data_len:
                break
            try:
                scale = struct.unpack_from('<3f', data, entry_offset)
                vertex_count = struct.unpack_from('<I', data, entry_offset + 72)[0]
                vertex_ptr = struct.unpack_from('<I', data, entry_offset + 68)[0]
                color_ptr = struct.unpack_from('<I', data, entry_offset + 76)[0]
            except struct.error:
                break
            if not _is_finite_triplet(scale):
                continue
            if vertex_count > 500000:
                continue
            score += 1
            if vertex_count == 0:
                continue
            vertex_end = vertex_ptr + (vertex_count * 12)
            color_end = color_ptr + (vertex_count * 4)
            if 0 < vertex_ptr < data_len and vertex_end <= data_len:
                score += 2
            if 0 < color_ptr < data_len and color_end <= data_len:
                score += 1
        if score > best_score:
            best_score = score
            best_stride = stride
    return best_stride

class _RawBinaryReader:
    def __init__(self, data: bytes):
        self.data = bytearray(data)
        self.offset = 0

    def seek(self, offset: int, whence: int = 0) -> None:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.data) + offset
        else:
            raise ValueError(f'Invalid whence: {whence}')

    def tell(self) -> int:
        return self.offset

    def read(self, size: int) -> bytes:
        size = max(0, int(size))
        if (self.offset + size) > len(self.data):
            raise EOFError(f'Expected {size} bytes at 0x{self.offset:X}, got {max(0, len(self.data) - self.offset)}')
        value = bytes(self.data[self.offset:self.offset + size])
        self.offset += size
        return value

    def read_i8(self) -> int:
        value = struct.unpack_from('<b', self.data, self.offset)[0]
        self.offset += 1
        return value

    def read_u8(self) -> int:
        value = self.data[self.offset]
        self.offset += 1
        return value

    def read_u16(self) -> int:
        value = struct.unpack_from('<H', self.data, self.offset)[0]
        self.offset += 2
        return value

    def read_i16(self) -> int:
        value = struct.unpack_from('<h', self.data, self.offset)[0]
        self.offset += 2
        return value

    def read_u32(self) -> int:
        value = struct.unpack_from('<I', self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_i32(self) -> int:
        value = struct.unpack_from('<i', self.data, self.offset)[0]
        self.offset += 4
        return value

    def read_f32(self) -> float:
        value = struct.unpack_from('<f', self.data, self.offset)[0]
        self.offset += 4
        return value

    def write_u32_at(self, offset: int, value: int) -> None:
        struct.pack_into('<I', self.data, offset, value)



def _reader_read_i8(reader) -> int:
    return int(reader.i8() if hasattr(reader, 'i8') else reader.read_i8())


def _reader_read_u8(reader) -> int:
    return int(reader.u8() if hasattr(reader, 'u8') else reader.read_u8())


def _reader_read_i16(reader) -> int:
    return int(reader.i16() if hasattr(reader, 'i16') else reader.read_i16())


def _reader_read_u16(reader) -> int:
    return int(reader.u16() if hasattr(reader, 'u16') else reader.read_u16())


def _reader_read_i32(reader) -> int:
    return int(reader.i32() if hasattr(reader, 'i32') else reader.read_i32())


def _reader_read_u32(reader) -> int:
    return int(reader.u32() if hasattr(reader, 'u32') else reader.read_u32())


def _reader_read_f32(reader) -> float:
    return float(reader.f32() if hasattr(reader, 'f32') else reader.read_f32())


def _reader_read_bytes(reader, size: int) -> bytes:
    return bytes(reader.read(int(size)))


def _reader_seek(reader, offset: int) -> None:
    reader.seek(int(offset))


def _safe_f32(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _read_raw_vec4_at(reader: _RawBinaryReader, absolute_offset: int) -> tuple[float, float, float, float]:
    if absolute_offset < 0 or absolute_offset + 16 > len(reader.data):
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(v) for v in struct.unpack_from('<4f', reader.data, int(absolute_offset)))


def _read_raw_spline_state_at(reader: _RawBinaryReader, absolute_offset: int) -> dict[str, object]:
    if absolute_offset < 0 or absolute_offset + 8 > len(reader.data):
        return {'t': 0.0, 'currKey': 0, 'pad': 0}
    t, curr_key, pad = struct.unpack_from('<fHH', reader.data, int(absolute_offset))
    return {'t': _safe_f32(t), 'currKey': int(curr_key), 'pad': int(pad)}


def _read_raw_spline_at(reader: _RawBinaryReader, absolute_offset: int) -> Optional[dict[str, object]]:
    absolute_offset = int(absolute_offset or 0)
    if absolute_offset <= 0 or absolute_offset + _SPLINE_HEADER_SIZE > len(reader.data):
        return None
    try:
        num_keys, spline_type, flags, count = struct.unpack_from('<HBBf', reader.data, absolute_offset)
    except Exception:
        return None
    if int(num_keys) > _MULTI_SPLINE_MAX_KEYS:
        return None
    end_offset = absolute_offset + _SPLINE_HEADER_SIZE + (int(num_keys) * _SPLINE_KEY_SIZE)
    if end_offset > len(reader.data):
        return None
    keys: list[dict[str, object]] = []
    cursor = absolute_offset + _SPLINE_HEADER_SIZE
    for _index in range(int(num_keys)):
        keys.append({
            'point': list(_read_raw_vec4_at(reader, cursor + 0x00)),
            'd1': list(_read_raw_vec4_at(reader, cursor + 0x10)),
            'd2': list(_read_raw_vec4_at(reader, cursor + 0x20)),
            't0': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x30)[0]),
            'tf': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x34)[0]),
            'count': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x38)[0]),
            'invCount': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x3C)[0]),
        })
        cursor += _SPLINE_KEY_SIZE
    return {
        'numKeys': int(num_keys),
        'type': int(spline_type),
        'flags': int(flags),
        'count': _safe_f32(count),
        'keys': keys,
        'sourceOffset': int(absolute_offset),
    }


def _read_raw_rspline_at(reader: _RawBinaryReader, absolute_offset: int) -> Optional[dict[str, object]]:
    absolute_offset = int(absolute_offset or 0)
    if absolute_offset <= 0 or absolute_offset + _SPLINE_HEADER_SIZE > len(reader.data):
        return None
    try:
        num_keys, spline_type, flags, count = struct.unpack_from('<HBBf', reader.data, absolute_offset)
    except Exception:
        return None
    if int(num_keys) > _MULTI_SPLINE_MAX_KEYS:
        return None
    end_offset = absolute_offset + _SPLINE_HEADER_SIZE + (int(num_keys) * _RSPLINE_KEY_SIZE)
    if end_offset > len(reader.data):
        return None
    keys: list[dict[str, object]] = []
    cursor = absolute_offset + _SPLINE_HEADER_SIZE
    for _index in range(int(num_keys)):
        keys.append({
            'q': list(_read_raw_vec4_at(reader, cursor + 0x00)),
            't0': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x10)[0]),
            'tf': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x14)[0]),
            'count': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x18)[0]),
            'invCount': _safe_f32(struct.unpack_from('<f', reader.data, cursor + 0x1C)[0]),
        })
        cursor += _RSPLINE_KEY_SIZE
    return {
        'numKeys': int(num_keys),
        'type': int(spline_type),
        'flags': int(flags),
        'count': _safe_f32(count),
        'keys': keys,
        'sourceOffset': int(absolute_offset),
    }


def _read_multi_spline_from_raw_reader(reader: _RawBinaryReader, absolute_offset: int, *, source_raw_value: int = 0) -> Optional[dict[str, object]]:
    absolute_offset = int(absolute_offset or 0)
    if absolute_offset <= 0 or absolute_offset + _MULTI_SPLINE_SIZE > len(reader.data):
        return None
    try:
        cur_rot_matrix = list(float(v) for v in struct.unpack_from('<16f', reader.data, absolute_offset))
        positional_ptr, rotational_ptr, scaling_ptr = struct.unpack_from('<III', reader.data, absolute_offset + 0x40)
    except Exception:
        return None
    positional = _read_raw_spline_at(reader, int(positional_ptr)) if positional_ptr else None
    rotational = _read_raw_rspline_at(reader, int(rotational_ptr)) if rotational_ptr else None
    scaling = _read_raw_spline_at(reader, int(scaling_ptr)) if scaling_ptr else None
    return {
        'format': 'TRLAU.MultiSpline.v1',
        'sourceOffset': int(absolute_offset),
        'sourceRawValue': int(source_raw_value or absolute_offset),
        'curRotMatrix': cur_rot_matrix,
        'positional': positional,
        'rotational': rotational,
        'scaling': scaling,
        'curPositional': _read_raw_spline_state_at(reader, absolute_offset + 0x4C),
        'curRotational': _read_raw_spline_state_at(reader, absolute_offset + 0x54),
        'curScaling': _read_raw_spline_state_at(reader, absolute_offset + 0x5C),
    }


def _read_fixed_c_string_from_reader(reader, size: int) -> str:
    try:
        data = _reader_read_bytes(reader, size)
    except Exception:
        return ''
    data = data.split(b'\x00', 1)[0]
    return data.decode('utf-8', errors='ignore')


def _read_stream_unit_portal_vec4(reader) -> tuple[float, float, float, float]:
    return (_reader_read_f32(reader), _reader_read_f32(reader), _reader_read_f32(reader), _reader_read_f32(reader))


def _read_stream_unit_portals_from_reader(reader, list_offset: int, count: int) -> list[dict[str, object]]:
    portals: list[dict[str, object]] = []
    list_offset = int(list_offset or 0)
    count = int(count or 0)
    if count <= 0 or count > _STREAM_UNIT_PORTAL_MAX_IMPORT_COUNT or list_offset <= 0:
        return portals
    data = getattr(reader, 'data', None)
    data_len = len(data) if data is not None else 0
    if data_len and (list_offset + (count * _STREAM_UNIT_PORTAL_SIZE)) > data_len:
        logger.warning('Skipping stream unit portals at 0x%X due to invalid count=%d', list_offset, count)
        return portals
    old_pos = int(reader.tell())
    try:
        for index in range(count):
            entry_offset = list_offset + (index * _STREAM_UNIT_PORTAL_SIZE)
            _reader_seek(reader, entry_offset)
            name = _read_fixed_c_string_from_reader(reader, 30)
            to_signal_id = _reader_read_i16(reader)
            m_signal_id = _reader_read_i16(reader)
            stream_id = _reader_read_i16(reader)
            _reader_read_u32(reader)  # closeVertList; no safe count is stored in StreamUnitPortal.
            active_distance = _reader_read_f32(reader)
            _reader_read_u32(reader)  # toStreamUnit runtime pointer; always exported null.
            min_vec = _read_stream_unit_portal_vec4(reader)
            max_vec = _read_stream_unit_portal_vec4(reader)
            portalquad = [_read_stream_unit_portal_vec4(reader) for _ in range(4)]
            normal = _read_stream_unit_portal_vec4(reader)
            portals.append({
                'tolevelname': str(name),
                'toSignalID': int(to_signal_id),
                'MSignalID': int(m_signal_id),
                'streamID': int(stream_id),
                'activeDistance': float(active_distance),
                'min': tuple(float(v) for v in min_vec),
                'max': tuple(float(v) for v in max_vec),
                'portalquad': [tuple(float(v) for v in vec) for vec in portalquad],
                'normal': tuple(float(v) for v in normal),
            })
    except Exception as exc:
        logger.warning('Failed reading stream unit portals at 0x%X: %s', list_offset, exc)
    finally:
        try:
            _reader_seek(reader, old_pos)
        except Exception:
            pass
    return portals



def _read_spline_camera_level_data_from_reader(reader) -> dict[str, int]:
    values: dict[str, int] = {}
    start = reader.tell()
    for field_name, field_type in _SPLINE_CAMERA_LEVEL_DATA_FIELDS:
        if field_type == 'i8':
            values[field_name] = _reader_read_i8(reader)
        elif field_type == 'u8':
            values[field_name] = _reader_read_u8(reader)
        elif field_type == 'u16':
            values[field_name] = _reader_read_u16(reader)
        elif field_type == 'i32':
            values[field_name] = _reader_read_i32(reader)
        elif field_type == 'u32':
            values[field_name] = _reader_read_u32(reader)
        else:
            values[field_name] = _reader_read_i16(reader)
    consumed = int(reader.tell()) - int(start)
    if consumed < _SPLINE_CAMERA_LEVEL_DATA_SIZE:
        reader.seek(int(reader.tell()) + (_SPLINE_CAMERA_LEVEL_DATA_SIZE - consumed))
    return values


def _reader_range_valid(reader, absolute_offset: int, size: int = 1) -> bool:
    data = getattr(reader, 'data', None)
    if data is None:
        return False
    absolute_offset = int(absolute_offset or 0)
    size = max(0, int(size or 0))
    return 0 <= absolute_offset and (absolute_offset + size) <= len(data)


def _context_range_valid(context: SectionContext, absolute_offset: int, size: int = 1) -> bool:
    absolute_offset = int(absolute_offset or 0)
    size = max(0, int(size or 0))
    return int(getattr(context, 'data_start', 0)) <= absolute_offset and (absolute_offset + size) <= int(getattr(context, 'data_end', 0))


def _read_c_string_from_reader_at(reader, absolute_offset: int, limit: int = 256) -> str:
    absolute_offset = int(absolute_offset or 0)
    if absolute_offset <= 0:
        return ''
    old_pos = int(reader.tell())
    try:
        _reader_seek(reader, absolute_offset)
        data = bytearray()
        for _ in range(max(0, int(limit))):
            value = _reader_read_u8(reader)
            if value == 0:
                break
            data.append(value)
        return bytes(data).decode('utf-8', errors='ignore')
    except Exception:
        return ''
    finally:
        try:
            _reader_seek(reader, old_pos)
        except Exception:
            pass


def _read_spline_camera_level_data_at(reader, absolute_offset: int) -> dict[str, int]:
    absolute_offset = int(absolute_offset or 0)
    old_pos = int(reader.tell())
    try:
        _reader_seek(reader, absolute_offset)
        return _read_spline_camera_level_data_from_reader(reader)
    finally:
        try:
            _reader_seek(reader, old_pos)
        except Exception:
            pass


def _read_signal_spline_pointer_values_raw(reader, signal_offset: int) -> list[Optional[dict[str, int]]]:
    values: list[Optional[dict[str, int]]] = []
    for pointer_offset in _SIGNAL_SPLINE_CAMERA_POINTER_OFFSETS:
        field_abs = int(signal_offset) + int(pointer_offset)
        if not _reader_range_valid(reader, field_abs, 4):
            values.append(None)
            continue
        raw_pointer = struct.unpack_from('<I', getattr(reader, 'data'), field_abs)[0]
        if raw_pointer > 0 and _reader_range_valid(reader, raw_pointer, _SPLINE_CAMERA_LEVEL_DATA_SIZE):
            values.append(_read_spline_camera_level_data_at(reader, raw_pointer))
        else:
            values.append(None)
    return values


def _resolve_section_pointer_at(cache: SectionContextCache, context: SectionContext, field_absolute_offset: int, raw_pointer: int) -> Tuple[SectionContext, int]:
    field_local_offset = int(field_absolute_offset) - int(context.data_start)
    return resolve_pointer(cache, context, field_local_offset, int(raw_pointer or 0))


def _read_signal_spline_pointer_values_section(cache: SectionContextCache, context: SectionContext, signal_abs: int) -> list[Optional[dict[str, int]]]:
    values: list[Optional[dict[str, int]]] = []
    reader = context.reader
    old_pos = int(reader.tell())
    try:
        for pointer_offset in _SIGNAL_SPLINE_CAMERA_POINTER_OFFSETS:
            field_abs = int(signal_abs) + int(pointer_offset)
            if not _context_range_valid(context, field_abs, 4):
                values.append(None)
                continue
            _reader_seek(reader, field_abs)
            raw_pointer = _reader_read_u32(reader)
            if raw_pointer <= 0:
                values.append(None)
                continue
            target_context, target_abs = _resolve_section_pointer_at(cache, context, field_abs, raw_pointer)
            if target_abs > 0 and _context_range_valid(target_context, target_abs, _SPLINE_CAMERA_LEVEL_DATA_SIZE):
                values.append(_read_spline_camera_level_data_at(target_context.reader, target_abs))
            else:
                values.append(None)
    finally:
        try:
            _reader_seek(reader, old_pos)
        except Exception:
            pass
    return values


def _read_attack_wave_runtime_entry(reader, absolute_offset: int, resolve_pointer_target) -> Tuple[dict[str, object], dict[str, Tuple[object, object, int]]]:
    entry: dict[str, object] = {}
    next_targets: dict[str, Tuple[object, object, int]] = {}
    base = int(absolute_offset or 0)
    old_pos = int(reader.tell())
    try:
        _reader_seek(reader, base)
        entry['radiusCheckTimer'] = _reader_read_f32(reader)
        for field_name in (
            'flags', 'rtFlags', 'probability', 'numAttackerSpawns', 'numSpawnsThisLoad',
            'currentAttacker', 'triggerRemaining', 'numAttackersPerWave', 'numLinkedAttackWaves',
            'numBlockedAttackWaves', 'numFinishedAttackWaves', 'numAttackers', 'musicRank',
        ):
            entry[field_name] = _reader_read_u8(reader)
        _reader_read_i8(reader)  # pad
        entry['radiusCheckMarker'] = _reader_read_u16(reader)
        entry['radiusCheckRadius'] = _reader_read_i16(reader)
        _reader_seek(reader, base + 24)
        for field_name in ('enableAttackWaves', 'disableAttackWaves', 'killAttackWaves'):
            field_abs = int(reader.tell())
            raw_pointer = _reader_read_u32(reader)
            if raw_pointer > 0:
                next_targets[field_name] = resolve_pointer_target(field_abs, raw_pointer)

        # AttackWaveRuntimeMovePosLimits starts at offset 36.
        _reader_seek(reader, base + 36)
        num_limits = _reader_read_i32(reader)
        all_limits = []
        for index in range(6):
            value = {
                'x': _reader_read_f32(reader),
                'y': _reader_read_f32(reader),
                'z': _reader_read_f32(reader),
                'rad': _reader_read_f32(reader),
                'pursueRad': _reader_read_f32(reader),
            }
            if index < max(0, min(6, int(num_limits))):
                all_limits.append(value)
        entry['numLimits'] = int(num_limits)
        entry['limits'] = all_limits

        num_box_limits = _reader_read_i32(reader)
        all_box_limits = []
        for index in range(6):
            value = {
                'x': _reader_read_f32(reader),
                'y': _reader_read_f32(reader),
                'z': _reader_read_f32(reader),
                'zrot': _reader_read_f32(reader),
                'width': _reader_read_i16(reader),
                'length': _reader_read_i16(reader),
                'pursueWidth': _reader_read_i16(reader),
                'pursueLength': _reader_read_i16(reader),
            }
            if index < max(0, min(6, int(num_box_limits))):
                all_box_limits.append(value)
        entry['numBoxLimits'] = int(num_box_limits)
        entry['boxLimits'] = all_box_limits

        num_plane_limits = _reader_read_i32(reader)
        all_plane_limits = []
        for index in range(1):
            value = {
                'px': _reader_read_f32(reader),
                'py': _reader_read_f32(reader),
                'pz': _reader_read_f32(reader),
                'nx': _reader_read_f32(reader),
                'ny': _reader_read_f32(reader),
                'nz': _reader_read_f32(reader),
            }
            if index < max(0, min(1, int(num_plane_limits))):
                all_plane_limits.append(value)
        entry['numPlaneLimits'] = int(num_plane_limits)
        entry['planeLimits'] = all_plane_limits

        num_run_and_gun = _reader_read_i32(reader)
        all_run_and_gun = []
        for index in range(6):
            value = {
                'x': _reader_read_f32(reader),
                'y': _reader_read_f32(reader),
                'z': _reader_read_f32(reader),
                'priority': _reader_read_i16(reader),
                'waittime': _reader_read_i16(reader),
                'failAction': _reader_read_i16(reader),
            }
            _reader_seek(reader, int(reader.tell()) + 2)
            marker_field_abs = int(reader.tell())
            marker_raw = _reader_read_u32(reader)
            marker_name = ''
            if marker_raw > 0:
                marker_reader, _marker_context, marker_abs = resolve_pointer_target(marker_field_abs, marker_raw)
                marker_name = _read_c_string_from_reader_at(marker_reader, marker_abs, 256)
            value['markerName'] = marker_name
            if index < max(0, min(6, int(num_run_and_gun))):
                all_run_and_gun.append(value)
        entry['numRunAndGunPos'] = int(num_run_and_gun)
        entry['runAndGunPositions'] = all_run_and_gun

        num_patrol_pos = _reader_read_i16(reader)
        patrol_type = _reader_read_i16(reader)
        all_patrol = []
        for index in range(8):
            value = {
                'x': _reader_read_f32(reader),
                'y': _reader_read_f32(reader),
                'z': _reader_read_f32(reader),
                'lookx': _reader_read_f32(reader),
                'looky': _reader_read_f32(reader),
                'lookz': _reader_read_f32(reader),
                'waittime': _reader_read_i16(reader),
                'anim': _reader_read_i16(reader),
                'mode': _reader_read_i16(reader),
                'look': _reader_read_i16(reader),
            }
            if index < max(0, min(8, int(num_patrol_pos))):
                all_patrol.append(value)
        entry['numPatrolPos'] = int(num_patrol_pos)
        entry['patrolType'] = int(patrol_type)
        entry['patrolPositions'] = all_patrol

        num_messages = _reader_read_i32(reader)
        all_messages = []
        for index in range(6):
            value = {'message': _reader_read_i32(reader), 'data': _reader_read_i32(reader)}
            if index < max(0, min(6, int(num_messages))):
                all_messages.append(value)
        entry['numMessages'] = int(num_messages)
        entry['messages'] = all_messages
        entry['endMarker'] = _reader_read_i32(reader)
        return entry, next_targets
    finally:
        try:
            _reader_seek(reader, old_pos)
        except Exception:
            pass


def _read_attack_wave_chain_raw(reader, start_pointer: int, chain_name: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    start_pointer = int(start_pointer or 0)
    if start_pointer <= 0 or not _reader_range_valid(reader, start_pointer, _ATTACK_WAVE_RUNTIME_SIZE):
        return result

    def resolve_raw(_field_abs: int, raw_pointer: int):
        return reader, None, int(raw_pointer or 0)

    current_reader = reader
    current_context = None
    current_abs = start_pointer
    visited: set[int] = set()
    for chain_index in range(_ATTACK_WAVE_CHAIN_MAX_IMPORT_COUNT):
        if current_abs <= 0 or current_abs in visited or not _reader_range_valid(current_reader, current_abs, _ATTACK_WAVE_RUNTIME_SIZE):
            break
        visited.add(current_abs)
        entry, next_targets = _read_attack_wave_runtime_entry(current_reader, current_abs, resolve_raw)
        entry['chain_index'] = int(chain_index)
        result.append(entry)
        target = next_targets.get(chain_name)
        if not target:
            break
        current_reader, current_context, current_abs = target
    return result


def _read_attack_wave_chain_section(cache: SectionContextCache, context: SectionContext, start_abs: int, chain_name: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    start_abs = int(start_abs or 0)
    if start_abs <= 0 or not _context_range_valid(context, start_abs, _ATTACK_WAVE_RUNTIME_SIZE):
        return result
    current_context = context
    current_abs = start_abs
    visited: set[Tuple[str, int]] = set()
    for chain_index in range(_ATTACK_WAVE_CHAIN_MAX_IMPORT_COUNT):
        key = (str(getattr(current_context, 'file_name', '')), int(current_abs))
        if current_abs <= 0 or key in visited or not _context_range_valid(current_context, current_abs, _ATTACK_WAVE_RUNTIME_SIZE):
            break
        visited.add(key)

        def resolve_section(field_abs: int, raw_pointer: int, *, source_context=current_context):
            target_context, target_abs = _resolve_section_pointer_at(cache, source_context, field_abs, raw_pointer)
            return target_context.reader, target_context, int(target_abs or 0)

        entry, next_targets = _read_attack_wave_runtime_entry(current_context.reader, current_abs, resolve_section)
        entry['chain_index'] = int(chain_index)
        result.append(entry)
        target = next_targets.get(chain_name)
        if not target:
            break
        _target_reader, target_context, target_abs = target
        if target_context is None:
            break
        current_context = target_context
        current_abs = int(target_abs or 0)
    return result


def _signal_flags_to_properties(raw_flags: int) -> dict[str, object]:
    raw_flags = int(raw_flags) & 0xFFFF
    values: dict[str, object] = {'raw_flags': raw_flags}
    for field_name, bit_shift, bit_mask in _SIGNAL_FLAG_BITS:
        if field_name == 'pad':
            continue
        values[field_name] = bool((raw_flags >> int(bit_shift)) & int(bit_mask))
    return values


def _signal_pointer_to_index(pointer_value: int, signal_list_start: int) -> int:
    pointer_value = int(pointer_value or 0)
    signal_list_start = int(signal_list_start or 0)
    if pointer_value <= 0 or signal_list_start <= 0 or pointer_value < signal_list_start:
        return -1
    delta = pointer_value - signal_list_start
    if delta % _SIGNAL_STRUCT_SIZE != 0:
        return -1
    index = delta // _SIGNAL_STRUCT_SIZE
    return int(index) if 0 <= index < _SIGNAL_MAX_IMPORT_COUNT else -1


def _read_signal_from_reader(
    reader,
    signal_offset: int,
    *,
    start_going_into_water_offset: int = 0,
    start_going_out_of_water_offset: int = 0,
    signal_list_start: int = 0,
    spline_pointer_values: Optional[list[Optional[dict[str, int]]]] = None,
    attack_wave_values: Optional[dict[str, list[dict[str, object]]]] = None,
    camera_link_signal_index: Optional[int] = None,
) -> dict[str, object]:
    props: dict[str, object] = {}
    signal_offset = int(signal_offset or 0)
    if signal_offset <= 0:
        return props
    _reader_seek(reader, signal_offset)
    raw_flags = _reader_read_u16(reader)
    props.update(_signal_flags_to_properties(raw_flags))
    if spline_pointer_values is None:
        spline_pointer_values = _read_signal_spline_pointer_values_raw(reader, signal_offset)
    props['splineCamDataPtr'] = list(spline_pointer_values or [None, None, None])[:3]
    _reader_seek(reader, signal_offset + 16)
    props['CameraLock'] = _reader_read_i32(reader)
    props['CameraUnlock'] = _reader_read_i32(reader)
    props['CameraSave'] = _reader_read_i16(reader)
    props['CameraRestore'] = _reader_read_i16(reader)
    props['CameraShakeScale'] = _reader_read_u16(reader)
    props['CameraShakeTime'] = _reader_read_i16(reader)
    spline_values = _read_spline_camera_level_data_from_reader(reader)
    for key, value in spline_values.items():
        props[f'splineCamParams_{key}'] = int(value)
    props['streamName'] = _read_fixed_c_string_from_reader(reader, 20)
    props['streamPortalIndex'] = _reader_read_i32(reader)
    props['SetSlideAngle'] = _reader_read_i32(reader)
    props['ReverbType'] = _reader_read_i32(reader)
    props['ReverbVolume'] = _reader_read_i32(reader)
    props['setMusicVar'] = _reader_read_i32(reader)
    props['setMusicVarValue'] = _reader_read_i32(reader)
    props['fatalType'] = _reader_read_i32(reader)
    raw_attack_wave_pointers = {
        'enableAttackWaves': _reader_read_u32(reader),
        'disableAttackWaves': _reader_read_u32(reader),
        'killAttackWaves': _reader_read_u32(reader),
    }
    if attack_wave_values is None:
        attack_wave_values = {
            name: _read_attack_wave_chain_raw(reader, pointer_value, name)
            for name, pointer_value in raw_attack_wave_pointers.items()
        }
    for name in ('enableAttackWaves', 'disableAttackWaves', 'killAttackWaves'):
        props[name] = list((attack_wave_values or {}).get(name, []) or [])
    props['CameraStackType'] = _reader_read_i8(reader)
    stack_bits = _reader_read_u8(reader)
    props['CameraStackData'] = int(stack_bits & 0x0F)
    props['CameraStackCamInUse'] = int((stack_bits >> 4) & 0x0F)
    props['CameraStackSignalFlags0'] = _reader_read_u8(reader)
    props['CameraStackSignalFlags1'] = _reader_read_u8(reader)
    props['DTPCameraID0'] = _reader_read_i32(reader)
    props['DTPCameraID1'] = _reader_read_i32(reader)
    raw_camera_link_signal = _reader_read_u32(reader)
    if camera_link_signal_index is None:
        camera_link_signal_index = _signal_pointer_to_index(raw_camera_link_signal, signal_list_start)
    props['cameraLinkSignalIndex'] = int(camera_link_signal_index)
    props['FSFXInDTPID'] = _reader_read_i32(reader)
    props['FSFXExitDTPID'] = _reader_read_i32(reader)
    props['FSFXInActiveDist'] = _reader_read_u16(reader)
    props['FSFXExitActiveDist'] = _reader_read_u16(reader)
    props['startGoingIntoWaterSignal'] = bool(int(start_going_into_water_offset or 0) == signal_offset)
    props['startGoingOutOfWaterSignal'] = bool(int(start_going_out_of_water_offset or 0) == signal_offset)
    return props


def _read_signal_from_section_context(
    cache: SectionContextCache,
    context: SectionContext,
    signal_abs: int,
    *,
    start_going_into_water_abs: int = 0,
    start_going_out_of_water_abs: int = 0,
    signal_list_start_abs: int = 0,
) -> dict[str, object]:
    reader = context.reader
    signal_abs = int(signal_abs or 0)
    spline_pointer_values = _read_signal_spline_pointer_values_section(cache, context, signal_abs)
    attack_wave_values: dict[str, list[dict[str, object]]] = {}
    old_pos = int(reader.tell())
    try:
        for name, pointer_offset in _SIGNAL_ATTACK_WAVE_POINTER_OFFSETS.items():
            field_abs = signal_abs + int(pointer_offset)
            attack_wave_values[name] = []
            if not _context_range_valid(context, field_abs, 4):
                continue
            _reader_seek(reader, field_abs)
            raw_pointer = _reader_read_u32(reader)
            if raw_pointer <= 0:
                continue
            target_context, target_abs = _resolve_section_pointer_at(cache, context, field_abs, raw_pointer)
            attack_wave_values[name] = _read_attack_wave_chain_section(cache, target_context, target_abs, name)

        camera_link_signal_index = -1
        field_abs = signal_abs + _SIGNAL_CAMERA_LINK_POINTER_OFFSET
        if _context_range_valid(context, field_abs, 4):
            _reader_seek(reader, field_abs)
            raw_pointer = _reader_read_u32(reader)
            if raw_pointer > 0:
                target_context, target_abs = _resolve_section_pointer_at(cache, context, field_abs, raw_pointer)
                camera_link_signal_index = _signal_pointer_to_index(target_abs, signal_list_start_abs)

        return _read_signal_from_reader(
            reader,
            signal_abs,
            start_going_into_water_offset=int(start_going_into_water_abs or 0),
            start_going_out_of_water_offset=int(start_going_out_of_water_abs or 0),
            signal_list_start=int(signal_list_start_abs or 0),
            spline_pointer_values=spline_pointer_values,
            attack_wave_values=attack_wave_values,
            camera_link_signal_index=camera_link_signal_index,
        )
    finally:
        try:
            _reader_seek(reader, old_pos)
        except Exception:
            pass


def _copy_signal_submesh(source_mesh: TerrainSignalMesh, signal_id: int, faces: List[Tuple[int, TerrainCollisionFace]]) -> TerrainSignalMesh:
    remap: dict[int, int] = {}
    vertices: List[Tuple[float, float, float]] = []
    copied_faces: List[TerrainCollisionFace] = []
    for face_index, face in faces:
        tri = (int(face.i0), int(face.i1), int(face.i2))
        if min(tri) < 0 or max(tri) >= len(source_mesh.vertices) or len(set(tri)) != 3:
            continue
        new_tri = []
        for vertex_index in tri:
            if vertex_index not in remap:
                remap[vertex_index] = len(vertices)
                vertices.append(tuple(float(v) for v in source_mesh.vertices[vertex_index]))
            new_tri.append(remap[vertex_index])
        copied_faces.append(TerrainCollisionFace(
            i0=int(new_tri[0]),
            i1=int(new_tri[1]),
            i2=int(new_tri[2]),
            adjacency_flags=int(face.adjacency_flags),
            collision_flags=int(face.collision_flags),
            client_flags=int(face.client_flags),
            material_type=int(face.material_type),
            signal_id=int(signal_id),
        ))
    return TerrainSignalMesh(
        position=tuple(float(v) for v in getattr(source_mesh, 'position', (0.0, 0.0, 0.0))),
        bbox_min=tuple(float(v) for v in getattr(source_mesh, 'bbox_min', (0.0, 0.0, 0.0))),
        bbox_max=tuple(float(v) for v in getattr(source_mesh, 'bbox_max', (0.0, 0.0, 0.0))),
        raw_bbox_min=tuple(float(v) for v in getattr(source_mesh, 'raw_bbox_min', (0.0, 0.0, 0.0))),
        raw_bbox_max=tuple(float(v) for v in getattr(source_mesh, 'raw_bbox_max', (0.0, 0.0, 0.0))),
        vertices=vertices,
        faces=copied_faces,
        vertex_type=int(getattr(source_mesh, 'vertex_type', 0)),
    )


def _split_signal_mesh_by_face_id(source_mesh: Optional[TerrainSignalMesh]) -> List[Tuple[int, TerrainSignalMesh]]:
    if source_mesh is None:
        return []
    faces_by_signal_id: dict[int, List[Tuple[int, TerrainCollisionFace]]] = {}
    for face_index, face in enumerate(list(getattr(source_mesh, 'faces', []) or [])):
        signal_id = int(getattr(face, 'signal_id', -1))
        if signal_id < 0:
            continue
        faces_by_signal_id.setdefault(signal_id, []).append((int(face_index), face))
    result: List[Tuple[int, TerrainSignalMesh]] = []
    for signal_id in sorted(faces_by_signal_id):
        submesh = _copy_signal_submesh(source_mesh, signal_id, faces_by_signal_id[signal_id])
        if submesh.vertices and submesh.faces:
            result.append((int(signal_id), submesh))
    return result


def _read_raw_signal_id_list_entry(reader: _RawBinaryReader, signal_id_list: int, signal_index: int) -> Optional[int]:
    signal_id_list = int(signal_id_list or 0)
    signal_index = int(signal_index)
    if signal_id_list <= 0 or signal_index < 0 or signal_index >= _SIGNAL_MAX_IMPORT_COUNT:
        return None
    pos = signal_id_list + (signal_index * 2)
    if pos < 0 or (pos + 2) > len(reader.data):
        return None
    return int(struct.unpack_from('<h', reader.data, pos)[0])


def _build_signals_from_raw_reader(reader: _RawBinaryReader, source_mesh: Optional[TerrainSignalMesh], signal_list_start: int, signal_id_list: int, start_going_into_water: int = 0, start_going_out_of_water: int = 0) -> List[TerrainSignal]:
    split_meshes = _split_signal_mesh_by_face_id(source_mesh)
    if not split_meshes:
        return []
    signals: List[TerrainSignal] = []
    for default_index, (face_id, mesh) in enumerate(split_meshes):
        signal_index = int(face_id) if 0 <= int(face_id) < _SIGNAL_MAX_IMPORT_COUNT else int(default_index)
        level_signal_id = _read_raw_signal_id_list_entry(reader, signal_id_list, signal_index)
        if level_signal_id is None:
            level_signal_id = int(face_id)
        signal_offset = int(signal_list_start or 0) + (signal_index * _SIGNAL_STRUCT_SIZE) if signal_list_start else 0
        props: dict[str, object] = {}
        if signal_offset > 0 and (signal_offset + _SIGNAL_STRUCT_SIZE) <= len(reader.data):
            try:
                props = _read_signal_from_reader(
                    reader,
                    signal_offset,
                    start_going_into_water_offset=int(start_going_into_water or 0),
                    start_going_out_of_water_offset=int(start_going_out_of_water or 0),
                    signal_list_start=int(signal_list_start or 0),
                )
            except Exception as exc:
                logger.warning('Failed reading Signal index=%d face_id=%d at 0x%X: %s', int(signal_index), int(face_id), signal_offset, exc)
        signals.append(TerrainSignal(index=int(signal_index), signal_id=int(level_signal_id), face_id=int(face_id), mesh=mesh, properties=props))
    return signals


@dataclass
class _RawRelocation:
    section: int
    offset: int
    type: int


@dataclass
class _RawSection:
    size: int
    type: int
    id: int
    num_relocations: int
    skip_flags: int = 0
    version_id: int = 0
    has_debug_info: int = 0
    resource_type: int = 0
    spec_mask: int = 0xFFFFFFFF
    packed_data: int = 0
    raw_relocations: List[Tuple[int, int, int]] = field(default_factory=list)
    relocations: List[_RawRelocation] = field(default_factory=list)
    offset: int = 0


class _RawSectionList:
    def __init__(self, reader: _RawBinaryReader, relocate: bool):
        self.reader = reader
        self.sections: List[_RawSection] = []
        self.read_header()
        if relocate:
            for section in self.sections:
                self.relocate(section)

    def read_header(self) -> None:
        version = self.reader.read_i32()
        if version != 14:
            raise ValueError(f'Invalid DRM version, expected 14 but got {version}')
        num_sections = self.reader.read_u32()
        for _ in range(num_sections):
            size = self.reader.read_u32()
            type_ = self.reader.read_u8()
            skip_flags = self.reader.read_u8()
            version_id = self.reader.read_u16()
            packed_data = self.reader.read_u32()
            id_ = self.reader.read_u32()
            spec_mask = self.reader.read_u32()
            self.sections.append(_RawSection(
                size=size,
                type=type_,
                id=id_,
                num_relocations=packed_data >> 8,
                skip_flags=skip_flags,
                version_id=version_id,
                has_debug_info=packed_data & 0x1,
                resource_type=(packed_data >> 1) & 0x7F,
                spec_mask=spec_mask,
                packed_data=packed_data,
            ))
        for section in self.sections:
            for _ in range(section.num_relocations):
                type_and_section_info = self.reader.read_u16()
                type_specific = self.reader.read_i16()
                offset = self.reader.read_u32()
                section.raw_relocations.append((int(type_and_section_info), int(type_specific), int(offset)))
                section.relocations.append(
                    _RawRelocation(
                        section=type_and_section_info >> 3,
                        offset=offset,
                        type=type_and_section_info & 7,
                    )
                )
            section.offset = self.reader.tell()
            self.reader.seek(section.size, 1)

    def relocate(self, section: _RawSection) -> None:
        for relocation in section.relocations:
            if relocation.type != RELOCATION_POINTER:
                continue
            position = section.offset + relocation.offset
            target = self.sections[relocation.section]
            stored_offset = struct.unpack_from('<I', self.reader.data, position)[0]
            self.reader.write_u32_at(position, target.offset + stored_offset)

    def get_section(self, index: int) -> _RawSection:
        return self.sections[index]


class _RawLevelReader:
    def __init__(self, reader: _RawBinaryReader, filepath: str, *, import_signals: bool = True):
        self.reader = reader
        self.filepath = filepath
        self.import_signals = bool(import_signals)
        self.level = LevelData(filepath=filepath)

    def parse(self) -> LevelData:
        level_offset = self.reader.tell()
        terrain = self.reader.read_u32()
        start_going_into_water = 0
        start_going_out_of_water = 0
        signal_list_start = 0
        signal_id_list = 0
        if level_offset >= 0 and (level_offset + 152) <= len(self.reader.data):
            start_going_into_water = struct.unpack_from('<I', self.reader.data, level_offset + 132)[0]
            start_going_out_of_water = struct.unpack_from('<I', self.reader.data, level_offset + 136)[0]
            signal_list_start = struct.unpack_from('<I', self.reader.data, level_offset + 144)[0]
            signal_id_list = struct.unpack_from('<I', self.reader.data, level_offset + 148)[0]
        self.reader.seek(terrain)
        self.reader.seek(12, 1)
        num_stream_unit_portals = self.reader.read_i32()
        stream_unit_portals = self.reader.read_u32()
        self.level.stream_unit_portals = _read_stream_unit_portals_from_reader(self.reader, stream_unit_portals, num_stream_unit_portals)

        num_groups = self.reader.read_i32()
        terrain_groups = self.reader.read_u32()
        signal_terrain_group = self.reader.read_u32()
        terrain_signal_list = self.reader.read_u32()
        self.reader.read_u32()  # terrainAnimTextures

        num_bg_instances = self.reader.read_u32()
        bg_instance_list = self.reader.read_u32()

        num_bg_objects = self.reader.read_u32()
        bg_object_list = self.reader.read_u32()

        self.reader.seek(12, 1)
        xboxpc_vertex_buffer = self.reader.read_u32()
        vmo_buffer = self.reader.read_u32()
        self.reader.seek(8, 1)
        num_vertices = self.reader.read_i32()
        vertex_buffer_alt = self.reader.read_u32()
        num_vmo_vertices = self.reader.read_i32()
        self.level.cdc_render_data_id = int(self.reader.read_u32())
        vertex_buffer = vertex_buffer_alt if vertex_buffer_alt != 0 else xboxpc_vertex_buffer

        for group_index in range(num_groups):
            self._read_terrain_group(group_index, terrain_groups + (group_index * 176))
        if self.import_signals and signal_terrain_group > 0:
            self.level.signal_mesh = self._read_signal_mesh_from_terrain_group(signal_terrain_group)
            self.level.signals = _build_signals_from_raw_reader(
                self.reader,
                self.level.signal_mesh,
                terrain_signal_list or signal_list_start,
                signal_id_list,
                start_going_into_water,
                start_going_out_of_water,
            )
        self._read_bg_objects(bg_object_list, num_bg_objects)
        self.level.bg_instances = self._read_bg_instances(bg_instance_list, num_bg_instances, bg_object_list)

        self.reader.seek(vertex_buffer)
        for _ in range(num_vertices):
            x = self.reader.read_i16()
            y = self.reader.read_i16()
            z = self.reader.read_i16()
            self.reader.seek(2, 1)
            color_bgra = self.reader.read_u32()
            u = self.reader.read_i16()
            v = self.reader.read_i16()
            self.reader.seek(4, 1)
            self.level.vertices.append(_convert_level_position(x, y, z))
            self.level.uvs.append((float(u) * _UV_SCALE, float(v) * _UV_SCALE))
            b = color_bgra & 0xFF
            g = (color_bgra >> 8) & 0xFF
            r = (color_bgra >> 16) & 0xFF
            a = (color_bgra >> 24) & 0xFF
            self.level.vertex_colors.append((r, g, b, a))

        self.reader.seek(vmo_buffer)
        for _ in range(num_vmo_vertices):
            x = self.reader.read_i16()
            y = self.reader.read_i16()
            z = self.reader.read_i16()
            self.reader.seek(2, 1)
            color_bgra = self.reader.read_u32()
            u = self.reader.read_i16()
            v = self.reader.read_i16()
            self.reader.seek(20, 1)
            self.level.vmo_vertices.append(_convert_level_position(x, y, z))
            self.level.vmo_uvs.append((float(u) * _UV_SCALE, float(v) * _UV_SCALE))
            b = color_bgra & 0xFF
            g = (color_bgra >> 8) & 0xFF
            r = (color_bgra >> 16) & 0xFF
            a = (color_bgra >> 24) & 0xFF
            self.level.vmo_vertex_colors.append((r, g, b, a))

        return self.level

    def _read_signal_mesh_from_terrain_group(self, terrain_group_offset: int) -> Optional[TerrainSignalMesh]:
        if terrain_group_offset <= 0 or (terrain_group_offset + 60) > len(self.reader.data):
            return None
        self.reader.seek(terrain_group_offset + 56)
        mesh_offset = self.reader.read_u32()
        return self._read_signal_mesh(mesh_offset)

    def _read_signal_mesh(self, mesh_offset: int) -> Optional[TerrainSignalMesh]:
        if mesh_offset <= 0 or (mesh_offset + 72) > len(self.reader.data):
            return None
        self.reader.seek(mesh_offset)
        bmin = (self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32())
        self.reader.seek(4, 1)
        bmax = (self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32())
        self.reader.seek(4, 1)
        position = (self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32())
        self.reader.seek(4, 1)
        vertex_list_offset = self.reader.read_u32()
        face_list_offset = self.reader.read_u32()
        self.reader.seek(8, 1)
        vertex_type = self.reader.read_u16()
        self.reader.seek(2, 1)
        num_faces = self.reader.read_u16()
        num_vertices = self.reader.read_u16()
        if num_vertices > 2_000_000 or num_faces > 2_000_000:
            logger.warning('Skipping signal mesh at 0x%X due to suspicious counts vertices=%d faces=%d', mesh_offset, num_vertices, num_faces)
            return None
        signal_mesh = TerrainSignalMesh(
            position=_convert_level_position(*position),
            bbox_min=_convert_level_position(*bmin),
            bbox_max=_convert_level_position(*bmax),
            raw_bbox_min=tuple(float(v) for v in bmin),
            raw_bbox_max=tuple(float(v) for v in bmax),
            vertex_type=int(vertex_type),
        )
        vertex_stride = 6 if vertex_type == 0 else 16
        if vertex_list_offset > 0 and num_vertices > 0 and (vertex_list_offset + (num_vertices * vertex_stride)) <= len(self.reader.data):
            self.reader.seek(vertex_list_offset)
            for _ in range(num_vertices):
                if vertex_type == 0:
                    signal_mesh.vertices.append(_convert_level_position(self.reader.read_i16(), self.reader.read_i16(), self.reader.read_i16()))
                else:
                    signal_mesh.vertices.append(_convert_level_position(self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32()))
                    self.reader.seek(4, 1)
        if face_list_offset > 0 and num_faces > 0 and (face_list_offset + (num_faces * 10)) <= len(self.reader.data):
            self.reader.seek(face_list_offset)
            for _ in range(num_faces):
                signal_mesh.faces.append(TerrainCollisionFace(
                    i0=self.reader.read_u16(),
                    i1=self.reader.read_u16(),
                    i2=self.reader.read_u16(),
                    adjacency_flags=self.reader.read_u8(),
                    collision_flags=self.reader.read_u8(),
                    signal_id=self.reader.read_u16(),
                ))
        if not signal_mesh.vertices or not signal_mesh.faces:
            return None
        return signal_mesh

    def _read_collision_kd_nodes(self, absolute_offset: int, node_count: int) -> List[TerrainCollisionKDNode]:
        if absolute_offset <= 0 or node_count <= 0:
            return []
        data_size = node_count * 12
        if (absolute_offset + data_size) > len(self.reader.data):
            return []

        self.reader.seek(absolute_offset)
        nodes: List[TerrainCollisionKDNode] = []
        for node_index in range(node_count):
            nodes.append(
                TerrainCollisionKDNode(
                    node_index=node_index,
                    neg_offset=self.reader.read_f32(),
                    pos_offset=self.reader.read_f32(),
                    ref_index=self.reader.read_u16(),
                    axis=self.reader.read_u8(),
                    num_faces=self.reader.read_u8(),
                )
            )
        return nodes

    def _populate_collision_kd_bounds(self, collision: TerrainGroupCollision) -> None:
        _populate_collision_kd_bounds_impl(collision)


    def _read_bg_instances(self, list_offset: int, count: int, bg_object_list: int) -> List[BGInstance]:
        instances: List[BGInstance] = []
        if list_offset == 0 or count <= 0:
            return instances
        if count < 0 or count > 100000:
            logger.warning('Skipping BGInstances due to suspicious count %d', count)
            return instances

        stride = 0xF0
        data_len = len(self.reader.data)
        for instance_index in range(count):
            entry_offset = list_offset + (instance_index * stride)
            if entry_offset < 0 or (entry_offset + stride) > data_len:
                logger.warning('Skipping BGInstance %d at invalid offset 0x%X', instance_index, entry_offset)
                continue
            rows = struct.unpack_from('<16f', self.reader.data, entry_offset)
            matrix_rows = (
                rows[0:4],
                rows[4:8],
                rows[8:12],
                rows[12:16],
            )
            bg_object_offset = struct.unpack_from('<I', self.reader.data, entry_offset + 0xC0)[0]
            flags = struct.unpack_from('<I', self.reader.data, entry_offset + 0xC4)[0]
            multi_spline_offset = struct.unpack_from('<I', self.reader.data, entry_offset + 0xC8)[0]
            original_radius = struct.unpack_from('<f', self.reader.data, entry_offset + 0xCC)[0]
            radius = struct.unpack_from('<f', self.reader.data, entry_offset + 0xD0)[0]
            instance_id = struct.unpack_from('<H', self.reader.data, entry_offset + 0xD4)[0]
            bg_flags = struct.unpack_from('<H', self.reader.data, entry_offset + 0xD6)[0]
            target_frame = struct.unpack_from('<h', self.reader.data, entry_offset + 0xD8)[0]
            clip_beg = struct.unpack_from('<h', self.reader.data, entry_offset + 0xDA)[0]
            clip_end = struct.unpack_from('<h', self.reader.data, entry_offset + 0xDC)[0]
            link_seg = struct.unpack_from('<h', self.reader.data, entry_offset + 0xDE)[0]
            link_instance_offset = struct.unpack_from('<I', self.reader.data, entry_offset + 0xE0)[0]
            color_data_index = struct.unpack_from('<h', self.reader.data, entry_offset + 0xE4)[0]
            lod = struct.unpack_from('<h', self.reader.data, entry_offset + 0xE6)[0]
            active_light_bitfield = struct.unpack_from('<I', self.reader.data, entry_offset + 0xEC)[0]
            if bg_object_offset >= bg_object_list:
                bg_object_index = (bg_object_offset - bg_object_list) // 0x60
            else:
                bg_object_index = -1
            instances.append(BGInstance(
                index=instance_index,
                bg_object_index=int(bg_object_index),
                bg_object_offset=int(bg_object_offset),
                instance_id=int(instance_id),
                flags=int(flags),
                multi_spline_offset=int(multi_spline_offset),
                multi_spline_data=_read_multi_spline_from_raw_reader(self.reader, int(multi_spline_offset), source_raw_value=int(multi_spline_offset)),
                original_radius=float(original_radius),
                radius=float(radius),
                bg_flags=int(bg_flags),
                target_frame=int(target_frame),
                clip_beg=int(clip_beg),
                clip_end=int(clip_end),
                link_seg=int(link_seg),
                link_instance_offset=int(link_instance_offset),
                color_data_index=int(color_data_index),
                lod=int(lod),
                active_light_bitfield=int(active_light_bitfield),
                matrix_rows=matrix_rows,
            ))
        return instances


    def _read_bg_objects(self, list_offset: int, num_bg_objects: int) -> None:
        if list_offset == 0 or num_bg_objects <= 0:
            return
        if num_bg_objects < 0 or num_bg_objects > 100000:
            logger.warning('Skipping BGObjects due to suspicious count %d', num_bg_objects)
            return

        stride = _detect_bgobject_stride_from_bytes(self.reader.data, list_offset, num_bg_objects)
        for bg_index in range(num_bg_objects):
            entry_offset = list_offset + (bg_index * stride)
            if entry_offset < 0 or (entry_offset + _BGOBJECT_MODEL_SIZE) > len(self.reader.data):
                logger.warning('Skipping BGObject %d at invalid offset 0x%X', bg_index, entry_offset)
                continue
            bg_object = self._read_bg_object(entry_offset, bg_index, stride)
            if bg_object is not None:
                self.level.bg_objects.append(bg_object)

    def _read_bg_object_index_list(self, absolute_offset: int, vertex_count: int, label: str) -> List[int]:
        if absolute_offset <= 0 or absolute_offset + 4 > len(self.reader.data):
            return []
        if absolute_offset < 0:
            return []
        previous = self.reader.tell()
        try:
            self.reader.seek(absolute_offset)
            count = self.reader.read_i32()
            if count <= 0:
                return []
            if count > max(500000, int(vertex_count) * 64):
                logger.warning('Skipping BGObject %s at 0x%X due to suspicious count %d', label, absolute_offset, count)
                return []

            max_vertex = max(0, int(vertex_count))

            data_start = self.reader.tell()
            values32: List[int] = []
            if data_start + (count * 4) <= len(self.reader.data):
                values32 = [int(struct.unpack_from('<i', self.reader.data, data_start + (index * 4))[0]) for index in range(count)]
                valid32 = [value for value in values32 if value >= 0 and (max_vertex <= 0 or value < max_vertex)]
                if len(valid32) == count:
                    return sorted(set(valid32))

            if data_start + (count * 2) > len(self.reader.data):
                logger.warning('Skipping BGObject %s at 0x%X due to truncated list count=%d', label, absolute_offset, count)
                return []
            values16 = [int(struct.unpack_from('<H', self.reader.data, data_start + (index * 2))[0]) for index in range(count)]
            return sorted({value for value in values16 if value >= 0 and (max_vertex <= 0 or value < max_vertex)})
        except Exception:
            logger.warning('Failed to parse BGObject %s at 0x%X', label, absolute_offset)
            return []
        finally:
            try:
                self.reader.seek(previous)
            except Exception:
                pass

    def _read_bg_object(self, absolute_offset: int, bg_index: int, stride: int) -> Optional[BGObject]:
        self.reader.seek(absolute_offset)
        scale_x = self.reader.read_f32()
        scale_y = self.reader.read_f32()
        scale_z = self.reader.read_f32()

        strip_ptr_offset = absolute_offset + 48
        vertex_ptr_offset = absolute_offset + 68
        vertex_count_offset = absolute_offset + 72
        color_ptr_offset = absolute_offset + 76
        if vertex_count_offset + 4 > len(self.reader.data):
            return None

        self.reader.seek(vertex_count_offset)
        vertex_count = self.reader.read_u32()
        if vertex_count > 500000:
            logger.warning('Skipping BGObject %d due to suspicious vertex count %d', bg_index, vertex_count)
            return None

        self.reader.seek(vertex_ptr_offset)
        vertex_list_offset = self.reader.read_u32()
        self.reader.seek(strip_ptr_offset)
        strip_offset = self.reader.read_u32()
        self.reader.seek(color_ptr_offset)
        color_list_offset = self.reader.read_u32()
        env_vertex_offset = 0
        eye_ref_env_vertex_offset = 0
        try:
            self.reader.seek(absolute_offset + 80)
            env_vertex_offset = self.reader.read_u32()
            eye_ref_env_vertex_offset = self.reader.read_u32()
        except Exception:
            env_vertex_offset = 0
            eye_ref_env_vertex_offset = 0

        position_candidates: List[Tuple[int, Tuple[float, float, float]]] = []
        for rel_offset in _BGOBJECT_POSITION_CANDIDATE_OFFSETS:
            if rel_offset == 80 and stride <= _BGOBJECT_MODEL_SIZE:
                continue
            candidate = _read_bgobject_position_candidate_from_bytes(self.reader.data, absolute_offset + rel_offset)
            if candidate is not None:
                position_candidates.append((rel_offset, candidate))

        flags = 0
        cdc_render_data_id = 0
        if (absolute_offset + 36) <= len(self.reader.data):
            flags = struct.unpack_from('<i', self.reader.data, absolute_offset + 32)[0]
        if (absolute_offset + 92) <= len(self.reader.data):
            cdc_render_data_id = struct.unpack_from('<I', self.reader.data, absolute_offset + 88)[0]

        bg_object = BGObject(
            index=bg_index,
            scale=(scale_x, scale_y, scale_z),
            position=_pick_bgobject_position((scale_x, scale_y, scale_z), position_candidates),
            stride=int(stride),
            flags=int(flags),
            cdc_render_data_id=int(cdc_render_data_id),
            position_candidates=position_candidates,
        )
        if vertex_count > 0 and vertex_list_offset > 0 and (vertex_list_offset + int(vertex_count) * 12) <= len(self.reader.data):
            bg_object.raw_vertex_data = bytes(self.reader.data[vertex_list_offset:vertex_list_offset + int(vertex_count) * 12])
        color_byte_count = int(vertex_count) * 4
        if color_byte_count > 0 and color_list_offset > 0:
            if env_vertex_offset > color_list_offset:
                candidate_size = int(env_vertex_offset - color_list_offset)
                if candidate_size >= color_byte_count:
                    color_byte_count = candidate_size - (candidate_size % 4)
            if color_byte_count > 0 and (color_list_offset + color_byte_count) <= len(self.reader.data):
                bg_object.raw_color_data = bytes(self.reader.data[color_list_offset:color_list_offset + color_byte_count])
                per_slot = max(1, int(vertex_count) * 4)
                bg_object.raw_color_slot_count = max(1, len(bg_object.raw_color_data) // per_slot)
        self._read_bg_object_vertices(bg_object, vertex_list_offset, vertex_count)
        self._read_bg_object_colors(bg_object, color_list_offset, vertex_count)
        self._read_bg_object_strip_chain(bg_object, strip_offset)
        if not _bgobject_has_render_data(bg_object):
            return None
        bg_object.env_mapped_vertices = []
        bg_object.eye_ref_env_mapped_vertices = []
        _apply_bgobject_reflection_flags(bg_object)
        return bg_object

    def _read_bg_object_vertices(self, bg_object: BGObject, absolute_offset: int, count: int) -> None:
        if absolute_offset == 0 or count <= 0:
            return
        stride = 12
        if absolute_offset < 0 or (absolute_offset + (count * stride)) > len(self.reader.data):
            logger.warning('Skipping BGObject %d vertices at invalid offset 0x%X count=%d', bg_object.index, absolute_offset, count)
            return

        self.reader.seek(absolute_offset)
        for _ in range(count):
            raw_x = self.reader.read_i16()
            raw_y = self.reader.read_i16()
            raw_z = self.reader.read_i16()
            self.reader.seek(2, 1)
            u = self.reader.read_i16()
            v = self.reader.read_i16()
            bg_object.vertices.append((float(raw_x), float(raw_y), float(raw_z)))
            bg_object.uvs.append((float(u) * _UV_SCALE, float(v) * _UV_SCALE))

    def _read_bg_object_colors(self, bg_object: BGObject, absolute_offset: int, count: int) -> None:
        if absolute_offset == 0 or count <= 0:
            return
        stride = 4
        if absolute_offset < 0 or (absolute_offset + (count * stride)) > len(self.reader.data):
            logger.warning('Skipping BGObject %d colors at invalid offset 0x%X count=%d', bg_object.index, absolute_offset, count)
            return

        self.reader.seek(absolute_offset)
        for _ in range(count):
            color_bgra = self.reader.read_u32()
            b = color_bgra & 0xFF
            g = (color_bgra >> 8) & 0xFF
            r = (color_bgra >> 16) & 0xFF
            a = (color_bgra >> 24) & 0xFF
            bg_object.vertex_colors.append((r, g, b, a))

    def _read_bg_object_strip_chain(self, bg_object: BGObject, absolute_offset: int) -> None:
        seen_offsets: set[int] = set()
        next_offset = absolute_offset
        material_index = 0
        while next_offset != 0:
            if next_offset in seen_offsets:
                logger.warning('Detected recursive BGObject strip list at 0x%X', next_offset)
                break
            seen_offsets.add(next_offset)
            next_offset = self._read_bg_object_strip(bg_object, next_offset, material_index)
            material_index += 1

    def _read_bg_object_strip(self, bg_object: BGObject, absolute_offset: int, material_index: int) -> int:
        if absolute_offset < 0 or (absolute_offset + 28) > len(self.reader.data):
            logger.warning('Skipping BGObject strip at invalid offset 0x%X', absolute_offset)
            return 0

        self.reader.seek(absolute_offset)
        vertex_count = self.reader.read_u32()
        if vertex_count <= 0:
            return 0
        if vertex_count > 2000000:
            logger.warning('Skipping BGObject strip at 0x%X due to suspicious vertex count %d', absolute_offset, vertex_count)
            return 0

        sort_vertex = (self.reader.read_i16(), self.reader.read_i16(), self.reader.read_i16())
        self.reader.read_u16()
        tpageid = self.reader.read_u32()
        sort_push = self.reader.read_i32()
        scroll_offset = self.reader.read_f32()
        next_offset = self.reader.read_u32()
        index_data_size = vertex_count * 2
        if self.reader.tell() + index_data_size > len(self.reader.data):
            logger.warning('Skipping BGObject strip indices at 0x%X due to truncated data', absolute_offset)
            return 0

        if (tpageid & 0x0001E000) == 0x00012000:
            self.reader.seek(index_data_size, 1)
            return next_offset

        strip = LevelStrip(material_index=material_index, tpageid=tpageid, flags=0, vertex_base_offset=0, sort_vertex=sort_vertex, sort_push=int(sort_push), scroll_offset=float(scroll_offset))
        for _ in range(vertex_count):
            strip.indices.append(self.reader.read_u16())
        bg_object.strips.append(strip)
        return next_offset

    def _read_terrain_collision(self, mesh_offset: int) -> Optional[TerrainGroupCollision]:
        if mesh_offset <= 0 or (mesh_offset + 80) > len(self.reader.data):
            return None

        self.reader.seek(mesh_offset)
        bmin = (self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32())
        self.reader.seek(4, 1)
        bmax = (self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32())
        self.reader.seek(4, 1)
        position = (self.reader.read_f32(), self.reader.read_f32(), self.reader.read_f32())
        self.reader.seek(4, 1)
        vertex_list_offset = self.reader.read_u32()
        face_list_offset = self.reader.read_u32()
        root_list_offset = self.reader.read_u32()
        self.reader.read_u32()
        self.reader.read_u16()
        node_count = self.reader.read_u16()
        num_faces = self.reader.read_u16()
        num_vertices = self.reader.read_u16()
        kd_max_depth = self.reader.read_u16()
        self.reader.seek(4, 1)
        self.reader.seek(2, 1)

        collision = TerrainGroupCollision(
            position=_convert_level_position(*position),
            bbox_min=_convert_level_position(*bmin),
            bbox_max=_convert_level_position(*bmax),
            raw_bbox_min=tuple(float(v) for v in bmin),
            raw_bbox_max=tuple(float(v) for v in bmax),
            kd_max_depth=int(kd_max_depth),
        )

        if vertex_list_offset > 0 and num_vertices > 0:
            vertex_data_size = num_vertices * 6
            if (vertex_list_offset + vertex_data_size) <= len(self.reader.data):
                self.reader.seek(vertex_list_offset)
                for _ in range(num_vertices):
                    collision.vertices.append(_convert_level_position(self.reader.read_i16(), self.reader.read_i16(), self.reader.read_i16()))

        if face_list_offset > 0 and num_faces > 0:
            face_data_size = num_faces * 10
            if (face_list_offset + face_data_size) <= len(self.reader.data):
                self.reader.seek(face_list_offset)
                for _ in range(num_faces):
                    collision.faces.append(TerrainCollisionFace(
                        i0=self.reader.read_u16(),
                        i1=self.reader.read_u16(),
                        i2=self.reader.read_u16(),
                        adjacency_flags=self.reader.read_u8(),
                        collision_flags=self.reader.read_u8(),
                        client_flags=self.reader.read_u8(),
                        material_type=self.reader.read_u8(),
                    ))

        if root_list_offset > 0 and node_count > 0:
            collision.kd_nodes = self._read_collision_kd_nodes(root_list_offset, node_count)
            if collision.kd_nodes:
                self._populate_collision_kd_bounds(collision)
                if collision.kd_max_depth <= 0:
                    collision.kd_max_depth = max((int(node.depth) for node in collision.kd_nodes), default=0)

        if not collision.vertices or not collision.faces:
            return None
        return collision

    def _read_terrain_group(self, index: int, offset: int) -> None:
        self.reader.seek(offset)
        global_offset = (
            self.reader.read_f32(),
            self.reader.read_f32(),
            self.reader.read_f32(),
        )
        self.reader.seek(4, 1)
        local_offset = (
            self.reader.read_f32(),
            self.reader.read_f32(),
            self.reader.read_f32(),
        )
        self.reader.seek(4, 1)
        flags = self.reader.read_i32()
        terrain_id = self.reader.read_i32()
        unique_id = self.reader.read_i32()
        spline_id = self.reader.read_i32()
        self.reader.seek(8, 1)
        mesh_offset = self.reader.read_u32()
        texture_morph_value = self.reader.read_f32()
        texture_morph_step = self.reader.read_f32()
        octree = self.reader.read_u32()
        octreeScrollInfo = self.reader.read_u32()
        octreeAnimatedInfo = self.reader.read_u32()
        self.reader.seek(64, 1)
        material_list = self.reader.read_u32()
        sorted_materials = self.reader.read_u32()
        self.reader.seek(8, 1)
        group_origin = (
            self.reader.read_f32(),
            self.reader.read_f32(),
            self.reader.read_f32(),
        )

        group = TerrainGroup(
            index=index,
            position=_convert_level_position(*global_offset),
            global_offset=_convert_level_position(*global_offset),
            local_offset=_convert_level_position(*local_offset),
            flags=flags,
            terrain_id=terrain_id,
            unique_id=unique_id,
            spline_id=spline_id,
            texture_morph_value=texture_morph_value,
            texture_morph_step=texture_morph_step,
            group_origin=_convert_level_position(*group_origin),
            collision=self._read_terrain_collision(mesh_offset),
        )
        self.level.terrain_groups.append(group)

        self.reader.seek(material_list)
        num_materials = self.reader.read_i32()
        for material_index in range(num_materials):
            entry_abs = material_list + 4 + (material_index * 20)
            self.reader.seek(entry_abs)
            tpageid = self.reader.read_u32()
            flags = self.reader.read_u32()
            vertex_base_offset = self.reader.read_u32()
            group.material_entry_offsets.append(int(entry_abs))
            group.strips.append(
                LevelStrip(
                    material_index=material_index,
                    tpageid=tpageid,
                    flags=flags,
                    vertex_base_offset=vertex_base_offset,
                )
            )
            self.reader.seek(8, 1)

        if sorted_materials > 0 and (sorted_materials + (num_materials * 2)) <= len(self.reader.data):
            self.reader.seek(sorted_materials)
            group.sorted_material_indices = [self.reader.read_u16() for _ in range(max(0, num_materials))]

        if octreeScrollInfo != 0:
            self._read_octree_scroll_info(octreeScrollInfo, group)

        if octree != 0:
            self._read_octree(octree, group)

    def _read_octree(self, offset: int, group: TerrainGroup) -> None:
        self.reader.seek(offset + 16)
        strip = self.reader.read_u32()
        num_spheres = self.reader.read_i32()
        if num_spheres > 8:
            raise ValueError(f'Octree sphere has more than 8 spheres: {num_spheres}')

        for _ in range(num_spheres):
            sphere = self.reader.read_u32()
            if sphere == 0:
                continue
            cursor = self.reader.tell()
            self._read_octree(sphere, group)
            self.reader.seek(cursor)

        while strip != 0:
            self.reader.seek(strip)
            vertex_count = self.reader.read_i32()
            if vertex_count == 0:
                break
            self.reader.seek(16, 1)
            material_ref = self.reader.read_u32()
            mat_index = _resolve_group_strip_material_index(group, material_ref, int(material_ref))
            self.reader.seek(16, 1)
            strip = self.reader.read_u32()
            if mat_index is None:
                raise IndexError(
                    f'Strip material reference 0x{int(material_ref) & 0xFFFFFFFF:08X} could not be resolved for group {group.index} '
                    f'({len(group.strips)} strips)'
                )
            for _ in range(vertex_count):
                group.strips[mat_index].indices.append(self.reader.read_i16())


    def _read_octree_scroll_info(self, offset: int, group: TerrainGroup) -> None:
        data_len = len(self.reader.data)
        if offset <= 0 or (offset + 4) > data_len:
            return

        self.reader.seek(offset)
        count = self.reader.read_i32()
        if count <= 0 or count > 100000:
            return

        unresolved = 0
        for entry_index in range(count):
            entry_offset = offset + 4 + (entry_index * 20)
            if (entry_offset + 20) > data_len:
                break
            self.reader.seek(entry_offset)
            self.reader.read_u32()
            scroll_speed = self.reader.read_f32()
            scroll_num_tiles = self.reader.read_i32()
            scroll_tile = self.reader.read_i32()
            fixup_value_offset = self.reader.read_u32()

            material_index = self._resolve_scroll_material_index(fixup_value_offset, group)
            if material_index is None:
                unresolved += 1
                continue

            strip = group.strips[material_index]
            strip.scroll_entry_count += 1
            if not strip.scroll_speeds or all(abs(float(existing) - float(scroll_speed)) > 1e-6 for existing in strip.scroll_speeds):
                strip.scroll_speeds.append(float(scroll_speed))
            if strip.scroll_num_tiles == 0:
                strip.scroll_num_tiles = int(scroll_num_tiles)
            if strip.scroll_tile == 0:
                strip.scroll_tile = int(scroll_tile)

        if unresolved:
            logger.debug('Terrain group %d raw scroll info unresolved entries: %d/%d', group.index, unresolved, count)

    def _resolve_scroll_material_index(self, fixup_value_offset: int, group: TerrainGroup) -> Optional[int]:
        if fixup_value_offset <= 0 or not group.strips:
            return None

        data_len = len(self.reader.data)
        preferred_deltas = (12, 8, 0, 4, 16, 20, -4)
        candidates: List[Tuple[int, int]] = []
        for delta_rank, delta in enumerate(preferred_deltas):
            probe = int(fixup_value_offset) + int(delta)
            if probe < 0 or (probe + 4) > data_len:
                continue
            try:
                value = struct.unpack_from('<i', self.reader.data, probe)[0]
            except struct.error:
                continue
            if 0 <= value < len(group.strips):
                score = 100 - (delta_rank * 10)
                if delta == 12:
                    score += 25
                candidates.append((score, int(value)))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]


class TRLevelParser:
    def __init__(self, filepath: str, *, import_textures: bool = True, import_bgobjects: bool = True, import_collisions: bool = True, import_markups: bool = True, import_signals: bool = True, game_hint: Optional[str] = None):
        self.filepath = filepath
        self.import_textures = import_textures
        self.import_bgobjects = import_bgobjects
        self.import_collisions = import_collisions
        self.import_markups = import_markups
        self.import_signals = import_signals
        self.game_hint = str(game_hint or '').strip().lower() or None
        self.level_cdc_render_data_id = 0
        self.bgobject_collection = None

    @staticmethod
    def is_level_file(filepath: str) -> bool:
        suffix = Path(filepath).suffix.lower()
        if suffix == '.drm':
            try:
                with open(filepath, 'rb') as f:
                    data = f.read()
                reader = _RawBinaryReader(data)
                sections = _RawSectionList(reader, relocate=False)
                first_section = sections.get_section(0)
                reader.seek(first_section.offset + 168)
                return reader.read_u32() == _LEVEL_VERSION
            except Exception:
                return False

        cache = SectionContextCache(filepath)
        try:
            context = cache.get_root_context()
            marker_offset = context.data_start + 168
            if marker_offset + 4 > context.file_size:
                return False
            context.reader.seek(marker_offset)
            return context.reader.u32() == _LEVEL_VERSION
        except Exception:
            return False
        finally:
            cache.close()

    def parse(self) -> LevelData:
        if Path(self.filepath).suffix.lower() == '.drm':
            return self._parse_drm()
        return self._parse_extracted_level()

    def _parse_intro_data_block_payload(self, payload: bytes | bytearray, *, source_offset: int = 0, source_raw_value: int = 0) -> Optional[IntroDataBlock]:
        if len(payload) < _INTRO_DATA_HEADER_SIZE:
            return None
        try:
            data_type = int(struct.unpack_from('<i', payload, 0)[0])
            (
                in_view_remove_dist,
                out_of_view_remove_dist,
                use_model,
                pad0,
                pad1,
                flags,
                attached_instance,
                swing_length,
                dtp_camera_id,
            ) = struct.unpack_from(_GENERIC_INTRO_STRUCT, payload, 4)
        except struct.error:
            return None

        generic = GenericIntroData(
            in_view_remove_dist=float(in_view_remove_dist),
            out_of_view_remove_dist=float(out_of_view_remove_dist),
            use_model=int(use_model),
            pad0=int(pad0),
            pad1=int(pad1),
            flags=int(flags),
            attached_instance=int(attached_instance),
            swing_length=float(swing_length),
            dtp_camera_id=int(dtp_camera_id),
        )
        specific_offset = _INTRO_DATA_HEADER_SIZE
        specific: dict[str, object] = {}
        raw_specific = bytes(payload[specific_offset:specific_offset + int(_INTRO_SPECIFIC_SIZES.get(data_type, 0))])

        try:
            if data_type == 17 and len(payload) >= specific_offset + _INTRO_SPECIFIC_SIZES[17]:
                reward_type, unique_id, sound_id = struct.unpack_from(_REWARD_INTRO_STRUCT, payload, specific_offset)
                specific = {
                    'reward_type': int(reward_type),
                    'unique_id': int(unique_id),
                    'sound_id': int(sound_id),
                }
            elif data_type == 13 and len(payload) >= specific_offset + _INTRO_SPECIFIC_SIZES[13]:
                (
                    water_depth,
                    water_inflow,
                    water_outflow,
                    water_speed,
                    flow_radius,
                    bob_height,
                    bob_frequency,
                    bob_grid_size,
                    priority,
                    water_flags,
                ) = struct.unpack_from(_WATER_VOLUME_INTRO_STRUCT, payload, specific_offset)
                specific = {
                    'water_depth': float(water_depth),
                    'water_inflow': int(water_inflow),
                    'water_outflow': int(water_outflow),
                    'water_speed': float(water_speed),
                    'flow_radius': float(flow_radius),
                    'bob_height': int(bob_height),
                    'bob_frequency': int(bob_frequency),
                    'bob_grid_size': float(bob_grid_size),
                    'priority': int(priority),
                    'water_flags': int(water_flags),
                }
            elif data_type == 12 and len(payload) >= specific_offset + _INTRO_SPECIFIC_SIZES[12]:
                prefix_size = struct.calcsize(_ROPE_OBJ_INTRO_PREFIX_STRUCT)
                (
                    top_connect_type,
                    bottom_connect_type,
                    top_connect_instance,
                    top_connect_model_index,
                    top_connect_model_marker_index,
                    bottom_connect_instance,
                    bottom_connect_model_index,
                    bottom_connect_model_marker_index,
                    collision_plane_instance0,
                    collision_plane_instance1,
                    rope_camera_dtpid,
                    rope_camera_overrides_movement,
                    sound_input_min,
                    sound_input_max,
                ) = struct.unpack_from(_ROPE_OBJ_INTRO_PREFIX_STRUCT, payload, specific_offset)
                (
                    render_width,
                    render_length_per_v,
                    render_u_width,
                    render_color,
                    render_model,
                    render_texture_main,
                    render_texture_top,
                    render_texture_bottom,
                ) = struct.unpack_from(_ROPE_RENDER_PARAMS_STRUCT, payload, specific_offset + prefix_size)
                specific = {
                    'top_connect_type': int(top_connect_type),
                    'bottom_connect_type': int(bottom_connect_type),
                    'top_connect_instance': int(top_connect_instance),
                    'top_connect_model_index': int(top_connect_model_index),
                    'top_connect_model_marker_index': int(top_connect_model_marker_index),
                    'bottom_connect_instance': int(bottom_connect_instance),
                    'bottom_connect_model_index': int(bottom_connect_model_index),
                    'bottom_connect_model_marker_index': int(bottom_connect_model_marker_index),
                    'collision_plane_instance0': int(collision_plane_instance0),
                    'collision_plane_instance1': int(collision_plane_instance1),
                    'rope_camera_dtpid': int(rope_camera_dtpid),
                    'rope_camera_overrides_movement': int(rope_camera_overrides_movement),
                    'sound_input_min': float(sound_input_min),
                    'sound_input_max': float(sound_input_max),
                    'render_width': float(render_width),
                    'render_length_per_v': float(render_length_per_v),
                    'render_u_width': float(render_u_width),
                    'render_color': int(render_color),
                    'render_model': int(render_model),
                    'render_texture_main': int(render_texture_main),
                    'render_texture_top': int(render_texture_top),
                    'render_texture_bottom': int(render_texture_bottom),
                }
        except struct.error:
            specific = {}

        return IntroDataBlock(
            data_type=int(data_type),
            generic=generic,
            specific=specific,
            raw_specific=raw_specific,
            source_offset=int(source_offset),
            source_raw_value=int(source_raw_value),
        )

    def _read_intro_data_block_from_context(self, cache: SectionContextCache, source_context: SectionContext, field_abs: int) -> Optional[IntroDataBlock]:
        raw_value, data_ctx, data_abs = self._resolve_pointer_at_absolute(cache, source_context, field_abs)
        if data_abs <= 0 or not self._is_valid_abs(data_ctx, data_abs, _INTRO_DATA_HEADER_SIZE):
            return None
        max_read = min(128, max(0, data_ctx.file_size - int(data_abs)))
        if max_read < _INTRO_DATA_HEADER_SIZE:
            return None
        br = data_ctx.reader
        previous = br.tell()
        try:
            br.seek(data_abs)
            payload = br.read(max_read)
        except Exception:
            return None
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass
        return self._parse_intro_data_block_payload(payload, source_offset=int(data_abs), source_raw_value=int(raw_value))

    def _read_vec4_from_context(self, context: SectionContext, absolute_offset: int) -> tuple[float, float, float, float]:
        if not self._is_valid_abs(context, absolute_offset, 16):
            return (0.0, 0.0, 0.0, 0.0)
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            return (br.f32(), br.f32(), br.f32(), br.f32())
        except Exception:
            return (0.0, 0.0, 0.0, 0.0)
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

    def _read_spline_state_from_context(self, context: SectionContext, absolute_offset: int) -> dict[str, object]:
        if not self._is_valid_abs(context, absolute_offset, 8):
            return {'t': 0.0, 'currKey': 0, 'pad': 0}
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            return {'t': _safe_f32(br.f32()), 'currKey': int(br.u16()), 'pad': int(br.u16())}
        except Exception:
            return {'t': 0.0, 'currKey': 0, 'pad': 0}
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

    def _read_spline_from_context(self, context: SectionContext, absolute_offset: int) -> Optional[dict[str, object]]:
        absolute_offset = int(absolute_offset or 0)
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, _SPLINE_HEADER_SIZE):
            return None
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            num_keys = br.u16(); spline_type = br.u8(); flags = br.u8(); count = br.f32()
            if int(num_keys) > _MULTI_SPLINE_MAX_KEYS:
                return None
            if not self._is_valid_abs(context, absolute_offset, _SPLINE_HEADER_SIZE + (int(num_keys) * _SPLINE_KEY_SIZE)):
                return None
            keys: list[dict[str, object]] = []
            cursor = absolute_offset + _SPLINE_HEADER_SIZE
            for _index in range(int(num_keys)):
                keys.append({
                    'point': list(self._read_vec4_from_context(context, cursor + 0x00)),
                    'd1': list(self._read_vec4_from_context(context, cursor + 0x10)),
                    'd2': list(self._read_vec4_from_context(context, cursor + 0x20)),
                    't0': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x30))[0]),
                    'tf': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x34))[0]),
                    'count': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x38))[0]),
                    'invCount': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x3C))[0]),
                })
                cursor += _SPLINE_KEY_SIZE
            return {'numKeys': int(num_keys), 'type': int(spline_type), 'flags': int(flags), 'count': _safe_f32(count), 'keys': keys, 'sourceOffset': int(absolute_offset)}
        except Exception:
            return None
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

    def _read_rspline_from_context(self, context: SectionContext, absolute_offset: int) -> Optional[dict[str, object]]:
        absolute_offset = int(absolute_offset or 0)
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, _SPLINE_HEADER_SIZE):
            return None
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            num_keys = br.u16(); spline_type = br.u8(); flags = br.u8(); count = br.f32()
            if int(num_keys) > _MULTI_SPLINE_MAX_KEYS:
                return None
            if not self._is_valid_abs(context, absolute_offset, _SPLINE_HEADER_SIZE + (int(num_keys) * _RSPLINE_KEY_SIZE)):
                return None
            keys: list[dict[str, object]] = []
            cursor = absolute_offset + _SPLINE_HEADER_SIZE
            for _index in range(int(num_keys)):
                keys.append({
                    'q': list(self._read_vec4_from_context(context, cursor + 0x00)),
                    't0': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x10))[0]),
                    'tf': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x14))[0]),
                    'count': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x18))[0]),
                    'invCount': _safe_f32(struct.unpack('<f', context.reader.peek(4, offset=cursor + 0x1C))[0]),
                })
                cursor += _RSPLINE_KEY_SIZE
            return {'numKeys': int(num_keys), 'type': int(spline_type), 'flags': int(flags), 'count': _safe_f32(count), 'keys': keys, 'sourceOffset': int(absolute_offset)}
        except Exception:
            return None
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

    def _read_multi_spline_from_context(self, cache: SectionContextCache, source_context: SectionContext, field_abs: int) -> Optional[dict[str, object]]:
        raw_value, multi_ctx, multi_abs = self._resolve_pointer_at_absolute(cache, source_context, field_abs)
        if multi_abs <= 0 or not self._is_valid_abs(multi_ctx, multi_abs, _MULTI_SPLINE_SIZE):
            return None
        br = multi_ctx.reader
        previous = br.tell()
        try:
            br.seek(multi_abs)
            cur_rot_matrix = [float(br.f32()) for _index in range(16)]
        except Exception:
            return None
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

        _pos_raw, pos_ctx, pos_abs = self._resolve_pointer_at_absolute(cache, multi_ctx, multi_abs + 0x40)
        _rot_raw, rot_ctx, rot_abs = self._resolve_pointer_at_absolute(cache, multi_ctx, multi_abs + 0x44)
        _scale_raw, scale_ctx, scale_abs = self._resolve_pointer_at_absolute(cache, multi_ctx, multi_abs + 0x48)
        return {
            'format': 'TRLAU.MultiSpline.v1',
            'sourceOffset': int(multi_abs),
            'sourceRawValue': int(raw_value),
            'curRotMatrix': cur_rot_matrix,
            'positional': self._read_spline_from_context(pos_ctx, pos_abs) if pos_abs > 0 else None,
            'rotational': self._read_rspline_from_context(rot_ctx, rot_abs) if rot_abs > 0 else None,
            'scaling': self._read_spline_from_context(scale_ctx, scale_abs) if scale_abs > 0 else None,
            'curPositional': self._read_spline_state_from_context(multi_ctx, multi_abs + 0x4C),
            'curRotational': self._read_spline_state_from_context(multi_ctx, multi_abs + 0x54),
            'curScaling': self._read_spline_state_from_context(multi_ctx, multi_abs + 0x5C),
        }

    def _read_intro_data_block_from_raw_reader(self, reader: _RawBinaryReader, absolute_offset: int) -> Optional[IntroDataBlock]:
        absolute_offset = int(absolute_offset or 0)
        if absolute_offset <= 0 or absolute_offset + _INTRO_DATA_HEADER_SIZE > len(reader.data):
            return None
        payload = bytes(reader.data[absolute_offset:min(len(reader.data), absolute_offset + 128)])
        return self._parse_intro_data_block_payload(payload, source_offset=absolute_offset, source_raw_value=absolute_offset)

    def _populate_raw_intro_data_blocks(self, reader: _RawBinaryReader, level: LevelData) -> None:
        for intro in list(getattr(level, 'intro_data', []) or []):
            try:
                data_offset = int(getattr(intro, 'data', 0) or 0)
            except Exception:
                data_offset = 0
            block = self._read_intro_data_block_from_raw_reader(reader, data_offset)
            if block is not None:
                intro.intro_data_block = block
            if getattr(intro, 'multi_spline_data', None) is None:
                try:
                    multi_spline_offset = int(getattr(intro, 'multi_spline', 0) or 0)
                except Exception:
                    multi_spline_offset = 0
                multi_spline = _read_multi_spline_from_raw_reader(reader, multi_spline_offset, source_raw_value=multi_spline_offset)
                if multi_spline is not None:
                    intro.multi_spline_data = multi_spline

    def _populate_raw_bginstance_multi_spline_data(self, reader: _RawBinaryReader, level: LevelData) -> None:
        for bg_instance in list(getattr(level, 'bg_instances', []) or []):
            if getattr(bg_instance, 'multi_spline_data', None) is not None:
                continue
            try:
                multi_spline_offset = int(getattr(bg_instance, 'multi_spline_offset', 0) or 0)
            except Exception:
                multi_spline_offset = 0
            multi_spline = _read_multi_spline_from_raw_reader(reader, multi_spline_offset, source_raw_value=multi_spline_offset)
            if multi_spline is not None:
                bg_instance.multi_spline_data = multi_spline

    @staticmethod
    def _pack_raw_section_blob(reader: _RawBinaryReader, section: _RawSection) -> bytes:
        payload_start = int(section.offset)
        payload_end = payload_start + max(0, int(section.size))
        payload = bytes(reader.data[payload_start:payload_end])
        packed_data = (int(section.has_debug_info) & 0x1) | ((int(section.resource_type) & 0x7F) << 1) | (len(section.raw_relocations or []) << 8)
        blob = bytearray()
        blob.extend(b'DRM\0')
        blob.extend(struct.pack(
            '<iBBHIII',
            int(section.size),
            int(section.type) & 0xFF,
            int(section.skip_flags) & 0xFF,
            int(section.version_id) & 0xFFFF,
            int(packed_data) & 0xFFFFFFFF,
            int(section.id),
            int(section.spec_mask) & 0xFFFFFFFF,
        ))
        for type_and_section_info, type_specific, offset in section.raw_relocations or []:
            blob.extend(struct.pack('<HhI', int(type_and_section_info) & 0xFFFF, int(type_specific), int(offset) & 0xFFFFFFFF))
        blob.extend(payload)
        return bytes(blob)

    @staticmethod
    def _read_raw_drm_reloc_module(raw_reader: _RawBinaryReader, raw_sections: _RawSectionList) -> Optional[RelocModuleData]:
        if not raw_sections.sections:
            return None
        root_section = raw_sections.get_section(0)
        reloc = next((entry for entry in root_section.relocations if int(entry.offset) == 0x9C and int(entry.type) == RELOCATION_POINTER), None)
        field_pos = int(root_section.offset) + 0x9C
        if field_pos < 0 or field_pos + 4 > len(raw_reader.data):
            return None
        try:
            raw_value = int(struct.unpack_from('<I', raw_reader.data, field_pos)[0])
        except struct.error:
            return None
        if reloc is None or raw_value == 0 and int(reloc.section) == 0:
            return None
        target_index = int(reloc.section)
        if target_index < 0 or target_index >= len(raw_sections.sections):
            return None
        target = raw_sections.get_section(target_index)
        return RelocModuleData(
            target_offset=int(raw_value),
            section_index=target_index,
            section_type=int(target.type),
            section_id=int(target.id),
            skip_flags=int(target.skip_flags),
            version_id=int(target.version_id),
            has_debug_info=int(target.has_debug_info),
            resource_type=int(target.resource_type),
            spec_mask=int(target.spec_mask),
            file_name=f'reloc_module_{target_index:03d}_{int(target.id)}',
            section_blob=TRLevelParser._pack_raw_section_blob(raw_reader, target),
        )

    @staticmethod
    def _capture_raw_combat_data(raw_reader: _RawBinaryReader, raw_sections: _RawSectionList) -> dict[str, object]:
        sections = list(getattr(raw_sections, 'sections', []) or [])
        if not sections:
            return {}
        root = sections[0]
        root_start = int(getattr(root, 'offset', 0) or 0)
        root_size = int(getattr(root, 'size', 0) or 0)
        root_data = bytes(raw_reader.data[root_start:root_start + root_size])
        if len(root_data) < 0xD0:
            return {}

        def relocation_record(section, raw_entry):
            type_and_section_info, type_specific, source_offset = raw_entry
            source_offset = int(source_offset)
            target_section = int(type_and_section_info) >> 3
            relocation_type = int(type_and_section_info) & 0x7
            section_start = int(getattr(section, 'offset', 0) or 0)
            section_size = int(getattr(section, 'size', 0) or 0)
            if source_offset < 0 or source_offset + 4 > section_size:
                raw_target = 0
            else:
                raw_target = int(struct.unpack_from('<I', raw_reader.data, section_start + source_offset)[0])
            return {
                'source_offset': source_offset,
                'target_section': target_section,
                'target_offset': raw_target,
                'relocation_type': relocation_type,
                'type_specific': int(type_specific),
            }

        root_relocations = [relocation_record(root, entry) for entry in list(getattr(root, 'raw_relocations', []) or [])]
        relocation_by_offset = {int(entry['source_offset']): entry for entry in root_relocations}
        root_fields: list[dict[str, object]] = []
        root_targets: set[int] = set()
        pending_external: set[int] = set()
        for field_name, field_offset in ROOT_COMBAT_FIELDS:
            relocation = relocation_by_offset.get(int(field_offset))
            if relocation is None:
                continue
            target_section = _int_or(relocation.get('target_section'), -1)
            target_offset = _int_or(relocation.get('target_offset'), 0)
            if target_section < 0 or target_section >= len(sections):
                continue
            root_fields.append({
                'name': str(field_name),
                'field_offset': int(field_offset),
                'target_section': target_section,
                'target_offset': target_offset,
                'relocation_type': _int_or(relocation.get('relocation_type'), 0),
                'type_specific': _int_or(relocation.get('type_specific'), 0),
            })
            if target_section == 0:
                root_targets.add(target_offset)
            else:
                pending_external.add(target_section)
        if not root_fields:
            return {}

        external_indices: set[int] = set()
        external_relocations: dict[int, list[dict[str, object]]] = {}

        def collect_external(section_index: int) -> None:
            section_index = int(section_index)
            if section_index == 0 or section_index in external_indices:
                return
            if section_index < 0 or section_index >= len(sections):
                return
            external_indices.add(section_index)
            section = sections[section_index]
            records = [relocation_record(section, entry) for entry in list(getattr(section, 'raw_relocations', []) or [])]
            external_relocations[section_index] = records
            for record in records:
                target_index = _int_or(record.get('target_section'), -1)
                target_offset = _int_or(record.get('target_offset'), 0)
                if target_index == 0:
                    root_targets.add(target_offset)
                elif 0 < target_index < len(sections):
                    collect_external(target_index)

        for section_index in sorted(pending_external):
            collect_external(section_index)

        processed_tail_start: int | None = None
        root_tail_relocations: list[dict[str, object]] = []
        while root_targets:
            valid_root_targets = [int(value) for value in root_targets if int(value) >= 0]
            if not valid_root_targets:
                break
            tail_start = min(valid_root_targets)
            if tail_start < 0 or tail_start >= root_size:
                break
            if processed_tail_start is not None and tail_start >= processed_tail_start:
                break
            processed_tail_start = tail_start
            root_tail_relocations = [record for record in root_relocations if _int_or(record.get('source_offset'), -1) >= tail_start]
            for record in root_tail_relocations:
                target_index = _int_or(record.get('target_section'), -1)
                target_offset = _int_or(record.get('target_offset'), 0)
                if target_index == 0:
                    root_targets.add(target_offset)
                elif 0 < target_index < len(sections):
                    collect_external(target_index)

        graph: dict[str, object] = {
            'version': 3,
            'root_fields': root_fields,
            'root_tail_start': int(processed_tail_start) if processed_tail_start is not None else -1,
            'root_tail_data': root_data[int(processed_tail_start):] if processed_tail_start is not None else b'',
            'root_tail_relocations': root_tail_relocations,
            'sections': [],
        }
        for section_index in sorted(external_indices):
            section = sections[section_index]
            section_start = int(getattr(section, 'offset', 0) or 0)
            section_size = int(getattr(section, 'size', 0) or 0)
            graph['sections'].append({
                'source_index': int(section_index),
                'base_offset': 0,
                'section_type': int(getattr(section, 'type', 0) or 0),
                'section_id': int(getattr(section, 'id', 0) or 0),
                'skip_flags': int(getattr(section, 'skip_flags', 0) or 0),
                'version_id': int(getattr(section, 'version_id', 0) or 0),
                'has_debug_info': int(getattr(section, 'has_debug_info', 0) or 0),
                'resource_type': int(getattr(section, 'resource_type', 0) or 0),
                'spec_mask': int(getattr(section, 'spec_mask', 0xFFFFFFFF) or 0xFFFFFFFF),
                'data': bytes(raw_reader.data[section_start:section_start + section_size]),
                'relocations': list(external_relocations.get(section_index, []) or []),
            })
        rebuild_combat_summary(graph)
        return graph

    @staticmethod
    def _read_extracted_reloc_module(cache: SectionContextCache, context: SectionContext) -> Optional[RelocModuleData]:
        try:
            context.reader.seek(context.data_start + 0x9C)
            raw_value = int(context.reader.u32())
        except Exception:
            return None
        relocation = context.section_info.relocations_by_offset.get(0x9C)
        if relocation is None:
            return None
        if raw_value == 0 and int(getattr(relocation, 'section_index_or_type', 0) or 0) == 0:
            return None
        try:
            target_context = cache.resolve_target_context(context, relocation.section_index_or_type)
        except Exception as exc:
            logger.warning('Failed to resolve Level.relocModule target for %s: %s', context.file_name, exc)
            return None
        try:
            section_blob = Path(target_context.filepath).read_bytes()
        except Exception as exc:
            logger.warning('Failed to read relocModule section %s: %s', target_context.file_name, exc)
            section_blob = b''
        info = target_context.section_info
        return RelocModuleData(
            target_offset=int(raw_value),
            section_index=int(getattr(relocation, 'section_index_or_type', -1)),
            section_type=int(info.section_type),
            section_id=int(info.section_id),
            skip_flags=int(info.skip_flags),
            version_id=int(info.version_id),
            has_debug_info=int(info.has_debug_info),
            resource_type=int(info.resource_type),
            spec_mask=int(info.spec_mask),
            file_name=str(target_context.file_name),
            section_blob=bytes(section_blob),
        )

    def _parse_drm(self) -> LevelData:
        with open(self.filepath, 'rb') as f:
            data = f.read()

        type_check_reader = _RawBinaryReader(data)
        type_sections = _RawSectionList(type_check_reader, relocate=False)
        first_section = type_sections.get_section(0)
        type_check_reader.seek(first_section.offset + 168)
        level_magic = type_check_reader.read_u32()
        if level_magic != _LEVEL_VERSION:
            raise ValueError(f'File does not appear to contain TRLAU level data (marker=0x{level_magic:08X})')

        reader = _RawBinaryReader(data)
        sections = _RawSectionList(reader, relocate=True)
        root_section = sections.get_section(0)
        reader.seek(root_section.offset)
        level = _RawLevelReader(reader, self.filepath, import_signals=self.import_signals).parse()
        self.level_cdc_render_data_id = int(getattr(level, 'cdc_render_data_id', 0))
        level.source_game = self._normalize_level_game(self.game_hint or self._detect_drm_game(sections))
        level.reloc_module = self._read_raw_drm_reloc_module(type_check_reader, type_sections)
        level.combat_data = self._capture_raw_combat_data(type_check_reader, type_sections)
        self._populate_drm_level_metadata(reader, sections, root_section, level)
        self._populate_raw_intro_data_blocks(reader, level)
        self._populate_raw_bginstance_multi_spline_data(reader, level)
        if self.import_textures:
            level.texture_images = self._load_drm_texture_images(reader, sections)
        return level

    def _read_c_string(self, reader: _RawBinaryReader, absolute_offset: int, limit: int = 512) -> str:
        if absolute_offset <= 0 or absolute_offset >= len(reader.data):
            return ''
        end = absolute_offset
        max_end = min(len(reader.data), absolute_offset + max(1, int(limit)))
        while end < max_end and reader.data[end] != 0:
            end += 1
        try:
            return bytes(reader.data[absolute_offset:end]).decode('utf-8', errors='ignore')
        except Exception:
            return ''


    def _parse_terrain_light_grid_cells_from_raw_reader(self, reader: _RawBinaryReader, grid_offset: int) -> List[Tuple[int, Tuple[int, ...]]]:
        cells: List[Tuple[int, Tuple[int, ...]]] = []
        grid_offset = int(grid_offset or 0)
        if grid_offset <= 0 or (grid_offset + 4096) > len(reader.data):
            return cells
        previous = reader.tell()
        try:
            for cell_index in range(1024):
                pointer_offset = grid_offset + (cell_index * 4)
                pointer_value = struct.unpack_from('<I', reader.data, pointer_offset)[0]
                if pointer_value <= 0 or pointer_value >= len(reader.data):
                    continue
                values: list[int] = []
                cursor = int(pointer_value)
                for _ in range(256):
                    if cursor + 2 > len(reader.data):
                        values = []
                        break
                    value = struct.unpack_from('<H', reader.data, cursor)[0]
                    cursor += 2
                    if value == 0xFFFF:
                        break
                    values.append(int(value))
                if values:
                    cells.append((int(cell_index), tuple(values)))
        except Exception as exc:
            logger.warning('Failed to parse TerrainLightGrids at 0x%X in %s: %s', grid_offset, getattr(self, 'filepath', '<memory>'), exc)
        finally:
            try:
                reader.seek(previous)
            except Exception:
                pass
        return cells

    def _parse_terrain_light_grid_cells_from_context(self, cache: SectionContextCache, context: SectionContext, grid_abs: int) -> List[Tuple[int, Tuple[int, ...]]]:
        cells: List[Tuple[int, Tuple[int, ...]]] = []
        grid_abs = int(grid_abs or 0)
        if grid_abs <= 0 or not self._is_valid_abs(context, grid_abs, 4096):
            return cells
        br = context.reader
        previous = br.tell()
        try:
            for cell_index in range(1024):
                pointer_field_abs = grid_abs + (cell_index * 4)
                raw_value, list_ctx, list_abs = self._resolve_pointer_at_absolute(cache, context, pointer_field_abs)
                pointer_value = list_abs if list_abs > 0 else raw_value
                if pointer_value <= 0:
                    continue
                if not self._is_valid_abs(list_ctx, pointer_value, 2):
                    continue
                values: list[int] = []
                list_reader = list_ctx.reader
                old_list_pos = list_reader.tell()
                try:
                    list_reader.seek(pointer_value)
                    for _ in range(256):
                        if not self._is_valid_abs(list_ctx, list_reader.tell(), 2):
                            values = []
                            break
                        value = int(list_reader.u16())
                        if value == 0xFFFF:
                            break
                        values.append(value)
                finally:
                    try:
                        list_reader.seek(old_list_pos)
                    except Exception:
                        pass
                if values:
                    cells.append((int(cell_index), tuple(values)))
        except Exception as exc:
            logger.warning('Failed to parse TerrainLightGrids at 0x%X in %s: %s', grid_abs, getattr(context, 'file_name', '<memory>'), exc)
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass
        return cells

    def _read_c_string_from_section_context(self, context: SectionContext, absolute_offset: int, limit: int = 512) -> str:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 1):
            return ''
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            byte_values = bytearray()
            max_end = min(context.file_size, absolute_offset + max(1, int(limit)))
            while br.tell() < max_end:
                value = br.u8()
                if value == 0:
                    break
                byte_values.append(value)
            return bytes(byte_values).decode('utf-8', errors='ignore')
        except Exception:
            return ''
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

    def _read_u16_list_until_zero(self, reader: _RawBinaryReader, absolute_offset: int, limit: int = 4096) -> tuple[int, ...]:
        if absolute_offset <= 0 or absolute_offset >= len(reader.data):
            return ()
        previous = reader.tell()
        values: list[int] = []
        try:
            reader.seek(absolute_offset)
            for _ in range(max(1, int(limit))):
                if (reader.tell() + 2) > len(reader.data):
                    break
                value = reader.read_u16()
                if value == 0:
                    break
                values.append(int(value))
        finally:
            try:
                reader.seek(previous)
            except Exception:
                pass
        return tuple(values)

    def _read_u16_list_until_zero_from_section_context(self, context: SectionContext, absolute_offset: int, limit: int = 4096) -> tuple[int, ...]:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 2):
            return ()
        br = context.reader
        previous = br.tell()
        values: list[int] = []
        try:
            br.seek(absolute_offset)
            for _ in range(max(1, int(limit))):
                if not self._is_valid_abs(context, br.tell(), 2):
                    break
                value = br.u16()
                if value == 0:
                    break
                values.append(int(value))
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass
        return tuple(values)

    def _normalize_level_game(self, game: Optional[str]) -> str:
        normalized = str(game or '').strip().lower()
        if normalized == 'legend':
            return 'legend'
        if normalized == 'anniversary':
            return 'anniversary'
        return 'unknown'

    def _detect_drm_game(self, sections: _RawSectionList) -> str:
        return 'legend' if int(getattr(self, 'level_cdc_render_data_id', 0)) != 0 else 'anniversary'

    def _read_markup_polyline_raw(self, reader: _RawBinaryReader, absolute_offset: int) -> List[Tuple[float, float, float, float]]:
        polyline: List[Tuple[float, float, float, float]] = []
        if absolute_offset <= 0 or (absolute_offset + 16) > len(reader.data):
            return polyline
        reader.seek(absolute_offset)
        num_points = reader.read_i32()
        if num_points <= 0 or num_points > 8192:
            return polyline
        data_start = absolute_offset + 16
        data_end = data_start + (num_points * 16)
        if data_end > len(reader.data):
            return polyline
        reader.seek(data_start)
        for _ in range(num_points):
            x = reader.read_f32()
            y = reader.read_f32()
            z = reader.read_f32()
            w = reader.read_f32()
            converted = _convert_level_position(x, y, z)
            polyline.append((float(converted[0]), float(converted[1]), float(converted[2]), float(w)))
        return polyline

    def _parse_raw_markups(self, reader: _RawBinaryReader, list_offset: int, count: int, game: str) -> List[MarkupData]:
        markups: List[MarkupData] = []
        if not self.import_markups or count <= 0 or list_offset <= 0 or list_offset >= len(reader.data):
            return markups
        normalized_game = self._normalize_level_game(game)
        if normalized_game == 'unknown':
            normalized_game = 'legend' if int(getattr(self, 'level_cdc_render_data_id', 0)) != 0 else 'anniversary'
        entry_size = 76 if normalized_game == 'anniversary' else 48
        requested_count = max(0, min(int(count), 65535))
        available_count = max(0, (len(reader.data) - int(list_offset)) // entry_size)
        parse_count = min(requested_count, available_count)
        if parse_count < requested_count:
            logger.warning('MarkUp list at 0x%X has count=%d but only %d complete entries fit; importing the complete entries', list_offset, count, parse_count)
        if parse_count <= 0:
            return markups
        for index in range(parse_count):
            entry_offset = list_offset + (index * entry_size)
            reader.seek(entry_offset)
            override_movement_camera = reader.read_i32()
            dtp_camera_data_id = reader.read_i32()
            dtp_markup_data_id = reader.read_i32()
            animated_segment = 0
            camera_antic = None
            if normalized_game == 'anniversary':
                animated_segment = reader.read_i32()
                use_antic_camera = reader.read_i32()
                antic_ids = tuple(int(reader.read_i32()) for _ in range(5))
                camera_antic = CameraAnticData(use_antic_camera=int(use_antic_camera), new_dtp_camera_antic_data_id=antic_ids)
            flags = reader.read_u32()
            intro_id = reader.read_i16()
            markup_id = reader.read_i16()
            px = reader.read_f32(); py = reader.read_f32(); pz = reader.read_f32()
            bbox = tuple(int(reader.read_i16()) for _ in range(6))
            polyline_offset = reader.read_u32()
            polyline = self._read_markup_polyline_raw(reader, polyline_offset)
            markups.append(MarkupData(
                index=index,
                game=normalized_game,
                override_movement_camera=int(override_movement_camera),
                dtp_camera_data_id=int(dtp_camera_data_id),
                dtp_markup_data_id=int(dtp_markup_data_id),
                animated_segment=int(animated_segment),
                camera_antic=camera_antic,
                flags=int(flags),
                intro_id=int(intro_id),
                markup_id=int(markup_id),
                position=_convert_level_position(px, py, pz),
                bbox=bbox,
                polyline_offset=int(polyline_offset),
                polyline=polyline,
            ))
        return markups

    @staticmethod
    def _apply_level_metadata(level: LevelData, metadata: dict[str, object]) -> None:
        level.level_metadata = dict(metadata)
        level.level_flags = int(metadata.get('flags', 0))
        level.unit_flags = int(metadata.get('unitFlags', 0))
        level.stream_unit_id = int(metadata.get('streamUnitID', 0))
        level.player_object_id = int(metadata.get('playerObjectID', -1))

    def _read_level_metadata_from_raw_reader(self, reader: _RawBinaryReader, base: int) -> dict[str, object]:
        reader.seek(base)
        reader.read_u32()
        metadata = {
            'waterZLevel': float(reader.read_f32()),
            'backColorR': int(reader.read_u8()),
            'backColorG': int(reader.read_u8()),
            'backColorB': int(reader.read_u8()),
        }
        reader.read_u8()
        metadata.update({
            'spectralColorR': int(reader.read_u8()),
            'spectralColorG': int(reader.read_u8()),
            'spectralColorB': int(reader.read_u8()),
            'spectralFXAlways': int(reader.read_u8()),
            'waterColorR': int(reader.read_u8()),
            'waterColorG': int(reader.read_u8()),
            'waterColorB': int(reader.read_u8()),
            'waterBlend': int(reader.read_i8()),
            'farPlane': float(reader.read_f32()),
            'fogFar': float(reader.read_f32()),
            'fogNear': float(reader.read_f32()),
            'spectralFarPlane': float(reader.read_f32()),
            'spectralFogFar': float(reader.read_f32()),
            'spectralFogNear': float(reader.read_f32()),
            'waterFarPlane': float(reader.read_f32()),
            'waterFogFar': float(reader.read_f32()),
            'waterFogNear': float(reader.read_f32()),
            'UnderwaterFXAlpha': int(reader.read_i32()),
            'UnderwaterFXMovement': float(reader.read_f32()),
            'UnderwaterFXSpeed': float(reader.read_f32()),
            'SpectralFXAlpha': int(reader.read_i32()),
            'SpectralFXIncrease': float(reader.read_f32()),
            'SpectralFXCenter': float(reader.read_f32()),
            'WaterFXAlpha': int(reader.read_i32()),
            'WaterFXMovement': float(reader.read_f32()),
            'WaterFXSpeed': float(reader.read_f32()),
            'WaterFSFX': int(reader.read_i32()),
        })
        return metadata

    def _read_level_metadata_from_section_context(self, context: SectionContext, base: int) -> dict[str, object]:
        br = context.reader
        br.seek(base)
        br.u32()
        metadata = {
            'waterZLevel': float(br.f32()),
            'backColorR': int(br.u8()),
            'backColorG': int(br.u8()),
            'backColorB': int(br.u8()),
        }
        br.u8()
        metadata.update({
            'spectralColorR': int(br.u8()),
            'spectralColorG': int(br.u8()),
            'spectralColorB': int(br.u8()),
            'spectralFXAlways': int(br.u8()),
            'waterColorR': int(br.u8()),
            'waterColorG': int(br.u8()),
            'waterColorB': int(br.u8()),
            'waterBlend': int(br.i8()),
            'farPlane': float(br.f32()),
            'fogFar': float(br.f32()),
            'fogNear': float(br.f32()),
            'spectralFarPlane': float(br.f32()),
            'spectralFogFar': float(br.f32()),
            'spectralFogNear': float(br.f32()),
            'waterFarPlane': float(br.f32()),
            'waterFogFar': float(br.f32()),
            'waterFogNear': float(br.f32()),
            'UnderwaterFXAlpha': int(br.i32()),
            'UnderwaterFXMovement': float(br.f32()),
            'UnderwaterFXSpeed': float(br.f32()),
            'SpectralFXAlpha': int(br.i32()),
            'SpectralFXIncrease': float(br.f32()),
            'SpectralFXCenter': float(br.f32()),
            'WaterFXAlpha': int(br.i32()),
            'WaterFXMovement': float(br.f32()),
            'WaterFXSpeed': float(br.f32()),
            'WaterFSFX': int(br.i32()),
        })
        return metadata


    @staticmethod
    def _decode_unitdata_fx_flags(value: int) -> Tuple[int, int, int]:
        raw = int(value) & 0xFF
        return (raw & 0x3, (raw >> 2) & 0x3, (raw >> 4) & 0x3)

    def _parse_unit_data_blob(self, data: bytes | bytearray, absolute_offset: int, game: str, section_id: Optional[int] = None, *, parse_pointer_arrays: bool = False, raw_sections=None) -> dict[str, object]:
        normalized_game = self._normalize_level_game(game)
        if normalized_game == 'unknown':
            normalized_game = 'legend' if int(getattr(self, 'level_cdc_render_data_id', 0)) != 0 else 'anniversary'
        if absolute_offset <= 0 or absolute_offset >= len(data):
            return {}

        reader = _RawBinaryReader(bytes(data))
        reader.seek(int(absolute_offset))
        metadata: dict[str, object] = {
            'id': int(section_id) if section_id is not None else 0,
            'absolute_offset': int(absolute_offset),
        }

        def _read_vec3f() -> Tuple[float, float, float]:
            return (float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32()))

        def _read_color3u8() -> Tuple[int, int, int]:
            return (int(reader.read_u8()), int(reader.read_u8()), int(reader.read_u8()))

        def _read_bend_item(prefix: str) -> None:
            metadata[f'bend_{prefix}_offset'] = _read_vec3f()
            metadata[f'bend_{prefix}_time_offset'] = _read_vec3f()
            metadata[f'bend_{prefix}_time_scale'] = _read_vec3f()

        try:
            metadata['num_fsfx'] = int(reader.read_u32())
            p_fsfx_field_abs = int(absolute_offset) + 4
            p_fsfx = int(reader.read_u32())
            if parse_pointer_arrays and raw_sections is not None:
                resolved_p_fsfx, _fsfx_section = self._resolve_raw_pointer_at_absolute_offset(data, raw_sections, p_fsfx_field_abs)
                if resolved_p_fsfx:
                    p_fsfx = int(resolved_p_fsfx)
            metadata['p_fsfx'] = int(p_fsfx)
            metadata['num_cines'] = int(reader.read_u32())
            p_cines_field_abs = int(absolute_offset) + 12
            p_cines = int(reader.read_u32())
            if parse_pointer_arrays and raw_sections is not None:
                resolved_p_cines, _cine_array_section = self._resolve_raw_pointer_at_absolute_offset(data, raw_sections, p_cines_field_abs)
                if resolved_p_cines:
                    p_cines = int(resolved_p_cines)
            metadata['p_cines'] = int(p_cines)

            metadata['depth_backcolor'] = _read_color3u8()
            metadata['depth_watercolor'] = _read_color3u8()
            metadata['depth_ambientcolor'] = _read_color3u8()
            metadata['depth_objectambientcolor'] = _read_color3u8()
            metadata['depth_waterblend'] = int(reader.read_u8())
            reader.seek(3, 1)
            metadata['depth_fogfar'] = float(reader.read_f32())
            metadata['depth_fogmax'] = float(reader.read_f32())
            metadata['depth_fognear'] = float(reader.read_f32())
            metadata['depth_waterfogfar'] = float(reader.read_f32())
            metadata['depth_waterfogmax'] = float(reader.read_f32())
            metadata['depth_waterfognear'] = float(reader.read_f32())
            metadata['depth_underwaterfxalpha'] = int(reader.read_i32())
            metadata['depth_underwaterfxmovement'] = float(reader.read_f32())
            metadata['depth_underwaterfxspeed'] = float(reader.read_f32())
            metadata['depth_underwaterfxrandscale'] = float(reader.read_f32())
            metadata['depth_underwaterfxscale'] = float(reader.read_f32())
            metadata['depth_underwaterfxcolor'] = _read_color3u8()
            underwater_flags = self._decode_unitdata_fx_flags(reader.read_i8())
            metadata['depth_underwaterfxadditive'] = int(underwater_flags[0])
            metadata['depth_underwaterfxbeforezsort'] = int(underwater_flags[1])
            metadata['depth_underwaterfxvertalpha'] = int(underwater_flags[2])
            metadata['depth_underwaterplayercolor'] = _read_color3u8()
            reader.seek(1, 1)
            metadata['depth_waterfxalpha'] = int(reader.read_i32())
            metadata['depth_waterfxmovement'] = float(reader.read_f32())
            metadata['depth_waterfxspeed'] = float(reader.read_f32())
            metadata['depth_waterfxrandscale'] = float(reader.read_f32())
            metadata['depth_waterfxscale'] = float(reader.read_f32())
            metadata['depth_waterfxcolor'] = _read_color3u8()
            water_flags = self._decode_unitdata_fx_flags(reader.read_i8())
            metadata['depth_waterfxadditive'] = int(water_flags[0])
            metadata['depth_waterfxbeforezsort'] = int(water_flags[1])
            metadata['depth_waterfxvertalpha'] = int(water_flags[2])

            for bend_name in ('black', 'blue', 'green', 'bluegreen', 'red', 'redblue', 'redgreen', 'redgreenblue'):
                _read_bend_item(bend_name)

            metadata['base_camera_use_camera_stack_system'] = int(reader.read_i32())
            metadata['base_camera_basecam'] = int(reader.read_u32())
            metadata['base_camera_ledgecam'] = int(reader.read_u32())
            metadata['base_camera_crawlcam'] = int(reader.read_u32())
            metadata['base_camera_vehiclecam'] = 0
            if normalized_game == 'legend':
                metadata['base_camera_vehiclecam'] = int(reader.read_u32())
            metadata['base_camera_deathcam'] = int(reader.read_u32())

            metadata['script'] = int(reader.read_u32())
            metadata['num_event_variable_storage'] = int(reader.read_u32())
            p_event_variable_storage_field_abs = int(reader.tell())
            p_event_variable_storage = int(reader.read_u32())
            if parse_pointer_arrays and raw_sections is not None:
                resolved_p_event_variable_storage, _event_section = self._resolve_raw_pointer_at_absolute_offset(data, raw_sections, p_event_variable_storage_field_abs)
                if resolved_p_event_variable_storage:
                    p_event_variable_storage = int(resolved_p_event_variable_storage)

            metadata['nextgen_vertex_color_percent'] = float(reader.read_f32())
            metadata['nextgen_global_ambient_color'] = _read_color3u8()
            reader.seek(1, 1)
            metadata['nextgen_global_ambient_intensity'] = float(reader.read_f32())
            metadata['nextgen_override_fog_color'] = int(reader.read_i8())
            metadata['nextgen_fog_color'] = _read_color3u8()
            metadata['nextgen_fog_start'] = float(reader.read_f32())
            metadata['nextgen_fog_end'] = float(reader.read_f32())
            metadata['nextgen_enable_pls_spot'] = int(reader.read_i8())
            metadata['nextgen_enable_pls_spot_shadows'] = int(reader.read_i8())
            reader.seek(2, 1)
            metadata['nextgen_light_fade_time'] = float(reader.read_f32())

            metadata['psp_override_fog_color'] = 0
            metadata['psp_fog_color'] = (0, 0, 0)
            metadata['psp_fog_start'] = 0.0
            metadata['psp_fog_end'] = 0.0
            if normalized_game == 'anniversary':
                metadata['psp_override_fog_color'] = int(reader.read_i8())
                metadata['psp_fog_color'] = _read_color3u8()
                metadata['psp_fog_start'] = float(reader.read_f32())
                metadata['psp_fog_end'] = float(reader.read_f32())

            metadata['overlay_snow_level'] = int(reader.read_i8())
            reader.seek(3, 1)
            metadata['overlay_num_effects'] = int(reader.read_u32())
            if int(metadata['overlay_num_effects']) <= 0:
                if parse_pointer_arrays:
                    fsfx_entries = self._read_unitdata_fsfx_entries_from_blob(data, p_fsfx, int(metadata.get('num_fsfx', 0) or 0))
                    if fsfx_entries:
                        metadata['fsfx_links'] = fsfx_entries
                    cine_entries = self._read_unitdata_cine_entries_from_blob(data, p_cines, int(metadata.get('num_cines', 0) or 0), raw_sections=raw_sections)
                    if cine_entries:
                        metadata['cine_entries'] = cine_entries
                    event_entries = self._read_unitdata_event_variable_entries_from_blob(data, p_event_variable_storage, int(metadata.get('num_event_variable_storage', 0) or 0))
                    if event_entries:
                        metadata['event_variable_storage_entries'] = event_entries
                return metadata
            metadata['overlay_p_effects_storage'] = int(reader.read_u32())
            metadata['overlay_effect_identification'] = int(reader.read_i32())
            metadata['overlay_effect_type'] = int(reader.read_i16())
            reader.seek(2, 1)
            metadata['overlay_effect_feet'] = float(reader.read_f32())
            metadata['overlay_effect_legs'] = float(reader.read_f32())
            metadata['overlay_effect_torso'] = float(reader.read_f32())
            metadata['overlay_effect_face'] = float(reader.read_f32())
            metadata['overlay_effect_hair'] = float(reader.read_f32())
            metadata['overlay_effect_feet_cap'] = float(reader.read_f32())
            metadata['overlay_effect_legs_cap'] = float(reader.read_f32())
            metadata['overlay_effect_torso_cap'] = float(reader.read_f32())
            metadata['overlay_effect_face_cap'] = float(reader.read_f32())
            metadata['overlay_effect_hair_cap'] = float(reader.read_f32())

            if parse_pointer_arrays:
                fsfx_entries = self._read_unitdata_fsfx_entries_from_blob(data, p_fsfx, int(metadata.get('num_fsfx', 0) or 0))
                if fsfx_entries:
                    metadata['fsfx_links'] = fsfx_entries
                cine_entries = self._read_unitdata_cine_entries_from_blob(data, p_cines, int(metadata.get('num_cines', 0) or 0), raw_sections=raw_sections)
                if cine_entries:
                    metadata['cine_entries'] = cine_entries
                event_entries = self._read_unitdata_event_variable_entries_from_blob(data, p_event_variable_storage, int(metadata.get('num_event_variable_storage', 0) or 0))
                if event_entries:
                    metadata['event_variable_storage_entries'] = event_entries
        except Exception as exc:
            logger.warning('Failed to parse UnitData at 0x%X in %s: %s', absolute_offset, getattr(self, 'filepath', '<memory>'), exc)
        return metadata

    @staticmethod
    def _read_unitdata_fsfx_entries_from_blob(data: bytes | bytearray, absolute_offset: int, count: int) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        count = max(0, min(int(count), 8192))
        absolute_offset = int(absolute_offset)
        if count <= 0 or absolute_offset <= 0 or absolute_offset >= len(data):
            return entries
        max_possible = max(0, (len(data) - absolute_offset) // 12)
        for index in range(min(count, max_possible)):
            item_offset = absolute_offset + (index * 12)
            try:
                fsfx_id, alpha, enabled = struct.unpack_from('<IfB', data, item_offset)
            except Exception:
                break
            entries.append({
                'index': int(index),
                'id': int(fsfx_id),
                'alpha': float(alpha),
                'enabled': int(1 if enabled else 0),
            })
        return entries

    @staticmethod
    def _read_unitdata_event_variable_entries_from_blob(data: bytes | bytearray, absolute_offset: int, count: int) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        count = max(0, min(int(count), 8192))
        absolute_offset = int(absolute_offset)
        if count <= 0 or absolute_offset <= 0 or absolute_offset >= len(data):
            return entries
        max_possible = max(0, (len(data) - absolute_offset) // 16)
        for index in range(min(count, max_possible)):
            item_offset = absolute_offset + (index * 16)
            try:
                variable, medium, easy, hard = struct.unpack_from('<H2xiii', data, item_offset)
            except Exception:
                break
            entries.append({
                'index': int(index),
                'variable': int(variable),
                'medium_value': int(medium),
                'easy_value': int(easy),
                'hard_value': int(hard),
            })
        return entries

    @staticmethod
    def _raw_sections_iter(raw_sections):
        return list(getattr(raw_sections, 'sections', raw_sections) or [])

    @staticmethod
    def _raw_section_content_offset(section) -> int:
        try:
            return int(getattr(section, 'offset', getattr(section, 'content_offset', 0)) or 0)
        except Exception:
            return 0

    @classmethod
    def _raw_section_index(cls, raw_sections, section) -> int:
        if section is None:
            return -1
        try:
            value = int(getattr(section, 'index'))
            if value >= 0:
                return value
        except Exception:
            pass
        try:
            for idx, candidate in enumerate(cls._raw_sections_iter(raw_sections)):
                if candidate is section:
                    return int(idx)
        except Exception:
            pass
        return -1

    @staticmethod
    def _raw_section_id(section) -> int:
        if section is None:
            return 0
        for attr in ('section_id', 'id'):
            try:
                return int(getattr(section, attr, 0) or 0)
            except Exception:
                pass
        return 0

    @staticmethod
    def _raw_relocation_type(relocation) -> int:
        try:
            return int(getattr(relocation, 'type'))
        except Exception:
            pass
        try:
            return int(getattr(relocation, 'type_and_section_info')) & 7
        except Exception:
            return -1

    @staticmethod
    def _raw_relocation_target_index(relocation) -> int:
        try:
            return int(getattr(relocation, 'section'))
        except Exception:
            pass
        try:
            return (int(getattr(relocation, 'type_and_section_info')) >> 3) & 0x1FFF
        except Exception:
            return -1

    @classmethod
    def _raw_relocation_at_local_offset(cls, section, local_offset: int):
        try:
            local_offset = int(local_offset)
        except Exception:
            return None
        for relocation in getattr(section, 'relocations', []) or []:
            try:
                if int(getattr(relocation, 'offset')) == local_offset:
                    return relocation
            except Exception:
                continue
        return None

    @classmethod
    def _find_raw_section_for_absolute_offset(cls, raw_sections, absolute_offset: int):
        if raw_sections is None:
            return None
        try:
            absolute_offset = int(absolute_offset)
        except Exception:
            return None
        for section in cls._raw_sections_iter(raw_sections):
            try:
                start = cls._raw_section_content_offset(section)
                end = start + int(getattr(section, 'size', 0))
            except Exception:
                continue
            if start <= absolute_offset < end:
                return section
        return None

    @classmethod
    def _resolve_raw_pointer_at_absolute_offset(cls, data: bytes | bytearray, raw_sections, field_absolute_offset: int) -> tuple[int, object | None]:
        try:
            field_absolute_offset = int(field_absolute_offset)
        except Exception:
            return 0, None
        if field_absolute_offset < 0 or field_absolute_offset + 4 > len(data):
            return 0, None
        try:
            raw_value = int(struct.unpack_from('<I', data, field_absolute_offset)[0])
        except Exception:
            return 0, None

        owner_section = cls._find_raw_section_for_absolute_offset(raw_sections, field_absolute_offset)
        if owner_section is None:
            return raw_value, cls._find_raw_section_for_absolute_offset(raw_sections, raw_value)
        owner_start = cls._raw_section_content_offset(owner_section)
        local_offset = int(field_absolute_offset) - int(owner_start)
        relocation = cls._raw_relocation_at_local_offset(owner_section, local_offset)
        if relocation is None or cls._raw_relocation_type(relocation) != RELOCATION_POINTER:
            return raw_value, cls._find_raw_section_for_absolute_offset(raw_sections, raw_value)

        target_index = cls._raw_relocation_target_index(relocation)
        sections = cls._raw_sections_iter(raw_sections)
        if target_index < 0 or target_index >= len(sections):
            return raw_value, cls._find_raw_section_for_absolute_offset(raw_sections, raw_value)
        target_section = sections[target_index]
        target_start = cls._raw_section_content_offset(target_section)
        target_size = int(getattr(target_section, 'size', 0) or 0)

        if target_start <= raw_value < target_start + max(0, target_size):
            return raw_value, target_section
        return int(target_start) + int(raw_value), target_section

    @staticmethod
    def _parse_cine_payload_name(data: bytes | bytearray, absolute_offset: int = 0) -> str:
        try:
            blob = bytes(data)
        except Exception:
            return ''
        try:
            absolute_offset = int(absolute_offset)
        except Exception:
            absolute_offset = 0
        if not blob or absolute_offset < 0 or absolute_offset >= len(blob):
            return ''
        payload = blob[absolute_offset:]
        parsed = parse_cine_payload(payload)
        return str(parsed.get('name', '') or '')

    @classmethod
    def _read_unitdata_cine_entries_from_blob(cls, data: bytes | bytearray, absolute_offset: int, count: int, raw_sections=None) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        count = max(0, min(int(count), 8192))
        absolute_offset = int(absolute_offset)
        if count <= 0 or absolute_offset <= 0 or absolute_offset >= len(data):
            return entries
        max_possible = max(0, (len(data) - absolute_offset) // 4)
        for index in range(min(count, max_possible)):
            item_offset = absolute_offset + (index * 4)
            try:
                stored_value = int(struct.unpack_from('<I', data, item_offset)[0])
            except Exception:
                break

            pointer_value, target_section = cls._resolve_raw_pointer_at_absolute_offset(data, raw_sections, item_offset)
            if int(pointer_value or 0) <= 0:
                pointer_value = int(stored_value)
                target_section = cls._find_raw_section_for_absolute_offset(raw_sections, pointer_value)

            entry: dict[str, object] = {
                'index': int(index),
                'pointer': int(pointer_value or 0),
                'stored_pointer': int(stored_value),
            }
            if target_section is not None:
                section_index = cls._raw_section_index(raw_sections, target_section)
                target_offset = cls._raw_section_content_offset(target_section)
                section_size = int(getattr(target_section, 'size', 0) or 0)
                entry['section_index'] = int(section_index)
                entry['section_id'] = int(cls._raw_section_id(target_section))
                entry['target_offset'] = int(pointer_value) - int(target_offset)
                try:
                    entry['name'] = cls._parse_cine_payload_name(data, int(pointer_value))
                except Exception:
                    pass
                try:
                    if 0 <= int(pointer_value) - int(target_offset) < max(1, section_size):
                        payload = bytes(data[int(pointer_value): int(target_offset) + section_size])
                        parsed = parse_cine_payload(payload)
                        entry['name'] = str(parsed.get('name', entry.get('name', '')) or entry.get('name', '') or '')
                        entry['cine_structured'] = 1
                        entry['cine_version_major'] = int(parsed.get('version_major', 0) or 0)
                        entry['cine_version_minor'] = int(parsed.get('version_minor', 0) or 0)
                        entry['cine_stream_unit_id'] = int(parsed.get('stream_unit_id', 0) or 0)
                        entry['cine_id'] = int(parsed.get('cine_id', 0) or 0)
                        entry['cine_end_time'] = float(parsed.get('end_time', 0.0) or 0.0)
                        entry['cine_spline_count'] = int(parsed.get('spline_count', 0) or 0)
                        entry['cine_command_count'] = int(parsed.get('command_count', 0) or 0)
                        entry['cine_command_types'] = str(parsed.get('command_types', '') or '')
                        entry['cine_rebuildable'] = int(1 if parsed.get('rebuildable') else 0)
                        entry['cine_parsed_size'] = int(parsed.get('parsed_size', 0) or 0)
                        for spline in list(parsed.get('splines') or []):
                            if str(spline.get('type_name', '')) != 'MasterCommandSpline':
                                continue
                            for node in list(spline.get('nodes') or []):
                                for command in list(node.get('commands') or []):
                                    if str(command.get('type', '')) == 'Cinematic':
                                        entry['cine_command_type'] = 'Cinematic'
                                        entry['cine_command_unit_id'] = int(command.get('unit_id', 0) or 0)
                                        entry['cine_command_load'] = int(command.get('load', 0) or 0)
                                        entry['cine_command_camera_control'] = int(command.get('camera_control', 0) or 0)
                                        entry['cine_command_channels'] = int(command.get('channels', 0) or 0)
                                        entry['cine_command_positions_after_playback'] = int(command.get('positions_after_playback', 0) or 0)
                                        entry['cine_command_end_trigger_id'] = int(command.get('end_trigger_id', 0) or 0)
                                        entry['cine_command_data_pointer'] = int(command.get('data_pointer', 0) or 0)
                                        entry['cine_command_data_size'] = int(command.get('data_size', 0) or 0)
                                        entry['cine_command_cinematic_name'] = str(command.get('cinematic_name', '') or '')
                                        raise StopIteration
                except StopIteration:
                    pass
                except Exception:
                    pass
            entries.append(entry)
        return entries

    def _parse_unitdata_arrays_from_context(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int, metadata: dict[str, object], game: str) -> None:
        normalized_game = self._normalize_level_game(game)
        if normalized_game == 'unknown':
            normalized_game = 'legend' if int(getattr(self, 'level_cdc_render_data_id', 0)) != 0 else 'anniversary'
        if absolute_offset <= 0 or absolute_offset >= context.file_size:
            return
        unit_local_base = int(absolute_offset) - int(context.data_start)
        if unit_local_base < 0:
            return

        def _resolve_pointer(field_rel_offset: int):
            field_abs = int(absolute_offset) + int(field_rel_offset)
            if field_abs + 4 > context.file_size:
                return context, 0
            context.reader.seek(field_abs)
            pointer_value = int(context.reader.u32())
            return resolve_pointer(cache, context, unit_local_base + int(field_rel_offset), pointer_value)

        try:
            fsfx_count = max(0, min(int(metadata.get('num_fsfx', 0) or 0), 8192))
            if fsfx_count > 0:
                fsfx_ctx, fsfx_abs = _resolve_pointer(4)
                if fsfx_abs > 0 and fsfx_abs < fsfx_ctx.data_end:
                    max_possible = max(0, (fsfx_ctx.data_end - fsfx_abs) // 12)
                    entries: list[dict[str, object]] = []
                    br = fsfx_ctx.reader
                    for index in range(min(fsfx_count, max_possible)):
                        br.seek(fsfx_abs + (index * 12))
                        entries.append({
                            'index': int(index),
                            'id': int(br.u32()),
                            'alpha': float(br.f32()),
                            'enabled': int(1 if br.u8() else 0),
                        })
                    if entries:
                        metadata['fsfx_links'] = entries

            cine_count = max(0, min(int(metadata.get('num_cines', 0) or 0), 8192))
            if cine_count > 0:
                cines_ctx, cines_abs = _resolve_pointer(12)
                if cines_abs > 0 and cines_abs < cines_ctx.data_end:
                    max_possible = max(0, (cines_ctx.data_end - cines_abs) // 4)
                    entries: list[dict[str, object]] = []
                    br = cines_ctx.reader
                    cines_local_base = int(cines_abs) - int(cines_ctx.data_start)
                    for index in range(min(cine_count, max_possible)):
                        item_abs = int(cines_abs) + (index * 4)
                        br.seek(item_abs)
                        raw_pointer = int(br.u32())
                        target_ctx, target_abs = resolve_pointer(cache, cines_ctx, cines_local_base + (index * 4), raw_pointer)
                        entry: dict[str, object] = {
                            'index': int(index),
                            'pointer': int(raw_pointer),
                            'target_offset': int(max(0, target_abs - target_ctx.data_start)),
                            'section_file': str(target_ctx.filepath.name),
                            'section_id': int(getattr(target_ctx.section_info, 'section_id', 0) or 0),
                            'name': '',
                        }
                        try:
                            m = __import__('re').match(r'^(\d+)_', target_ctx.filepath.name)
                            if m:
                                entry['section_index'] = int(m.group(1))
                        except Exception:
                            pass
                        try:
                            with open(target_ctx.filepath, 'rb') as cine_fh:
                                cine_blob = cine_fh.read()
                            entry['name'] = self._parse_cine_payload_name(cine_blob, int(target_ctx.data_start))
                            payload = bytes(cine_blob[int(target_ctx.data_start): int(getattr(target_ctx, 'data_end', len(cine_blob)) or len(cine_blob))])
                            parsed = parse_cine_payload(payload)
                            entry['name'] = str(parsed.get('name', entry.get('name', '')) or entry.get('name', '') or '')
                            entry['cine_structured'] = 1
                            entry['cine_version_major'] = int(parsed.get('version_major', 0) or 0)
                            entry['cine_version_minor'] = int(parsed.get('version_minor', 0) or 0)
                            entry['cine_stream_unit_id'] = int(parsed.get('stream_unit_id', 0) or 0)
                            entry['cine_id'] = int(parsed.get('cine_id', 0) or 0)
                            entry['cine_end_time'] = float(parsed.get('end_time', 0.0) or 0.0)
                            entry['cine_spline_count'] = int(parsed.get('spline_count', 0) or 0)
                            entry['cine_command_count'] = int(parsed.get('command_count', 0) or 0)
                            entry['cine_command_types'] = str(parsed.get('command_types', '') or '')
                            entry['cine_rebuildable'] = int(1 if parsed.get('rebuildable') else 0)
                            entry['cine_parsed_size'] = int(parsed.get('parsed_size', 0) or 0)
                            for command in list(parsed.get('commands') or []):
                                if str(command.get('type', '')) == 'Cinematic':
                                    entry['cine_command_type'] = 'Cinematic'
                                    entry['cine_command_unit_id'] = int(command.get('unit_id', 0) or 0)
                                    entry['cine_command_load'] = int(command.get('load', 0) or 0)
                                    entry['cine_command_camera_control'] = int(command.get('camera_control', 0) or 0)
                                    entry['cine_command_channels'] = int(command.get('channels', 0) or 0)
                                    entry['cine_command_positions_after_playback'] = int(command.get('positions_after_playback', 0) or 0)
                                    entry['cine_command_end_trigger_id'] = int(command.get('end_trigger_id', 0) or 0)
                                    entry['cine_command_data_pointer'] = int(command.get('data_pointer', 0) or 0)
                                    entry['cine_command_data_size'] = int(command.get('data_size', 0) or 0)
                                    entry['cine_command_cinematic_name'] = str(command.get('cinematic_name', '') or '')
                                    break
                        except Exception:
                            pass
                        entries.append(entry)
                    if entries:
                        metadata['cine_entries'] = entries

            event_count = max(0, min(int(metadata.get('num_event_variable_storage', 0) or 0), 8192))
            if event_count > 0:
                event_pointer_rel = 424 if normalized_game == 'anniversary' else 428
                event_ctx, event_abs = _resolve_pointer(event_pointer_rel)
                if event_abs > 0 and event_abs < event_ctx.data_end:
                    max_possible = max(0, (event_ctx.data_end - event_abs) // 16)
                    entries: list[dict[str, object]] = []
                    br = event_ctx.reader
                    for index in range(min(event_count, max_possible)):
                        br.seek(event_abs + (index * 16))
                        variable = int(br.u16())
                        br.seek(2, 1)
                        entries.append({
                            'index': int(index),
                            'variable': variable,
                            'medium_value': int(br.i32()),
                            'easy_value': int(br.i32()),
                            'hard_value': int(br.i32()),
                        })
                    if entries:
                        metadata['event_variable_storage_entries'] = entries
        except Exception as exc:
            logger.warning('Failed to parse UnitData pointed arrays at 0x%X in %s: %s', absolute_offset, getattr(context, 'file_name', '<memory>'), exc)

    def _parse_unit_data_from_context(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int, game: str) -> dict[str, object]:
        if absolute_offset <= 0 or absolute_offset >= context.file_size:
            return {}
        try:
            with open(context.filepath, 'rb') as fh:
                data = fh.read()
        except Exception as exc:
            logger.warning('Failed to read %s for UnitData import: %s', context.file_name, exc)
            return {}
        metadata = self._parse_unit_data_blob(data, absolute_offset, game, section_id=context.section_info.section_id, parse_pointer_arrays=False)
        self._parse_unitdata_arrays_from_context(cache, context, absolute_offset, metadata, game)
        return metadata


    def _parse_admd_blob(self, data: bytes | bytearray, absolute_offset: int, section_id: Optional[int] = None) -> dict[str, object]:
        if absolute_offset <= 0 or absolute_offset >= len(data):
            return {}
        reader = _RawBinaryReader(bytes(data))
        reader.seek(int(absolute_offset))
        metadata: dict[str, object] = {
            'id': int(section_id) if section_id is not None else 0,
            'absolute_offset': int(absolute_offset),
            'element_stride': 160,
        }

        def _read_bool() -> int:
            return int(1 if reader.read_u8() else 0)

        def _read_color3u8() -> Tuple[int, int, int]:
            return (int(reader.read_u8()), int(reader.read_u8()), int(reader.read_u8()))

        def _read_matrix_rows() -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float], Tuple[float, float, float, float]]:
            return (
                (float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32())),
                (float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32())),
                (float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32())),
                (float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32()), float(reader.read_f32())),
            )

        try:
            count = int(reader.read_u32())
            metadata['light_instance_count'] = count
            metadata['pp_light_instances'] = int(reader.read_u32())
            if count <= 0:
                return metadata
            max_possible = max(0, (len(data) - (absolute_offset + 8)) // 160)
            count = min(count, max_possible, 512)
            metadata['parsed_light_instance_count'] = count
            for index in range(count):
                base_key = f'light_{index:03d}_'
                rows = _read_matrix_rows()
                for row_index, row in enumerate(rows):
                    metadata[f'{base_key}transform_row{row_index}'] = row
                metadata[f'{base_key}property_type'] = int(reader.read_i32())
                metadata[f'{base_key}light_resource'] = int(reader.read_u32())
                metadata[f'{base_key}intensity'] = float(reader.read_f32())
                metadata[f'{base_key}rim_intensity'] = float(reader.read_f32())
                metadata[f'{base_key}specular_intensity'] = float(reader.read_f32())
                metadata[f'{base_key}enable_per_instance_intensity'] = _read_bool()
                metadata[f'{base_key}light_color'] = _read_color3u8()
                metadata[f'{base_key}enable_per_instance_color'] = _read_bool()
                reader.seek(3, 1)
                metadata[f'{base_key}range'] = float(reader.read_f32())
                metadata[f'{base_key}enable_per_instance_range'] = _read_bool()
                metadata[f'{base_key}ambient_color'] = _read_color3u8()
                metadata[f'{base_key}ambient_percentage'] = float(reader.read_f32())
                metadata[f'{base_key}enable_per_instance_ambient'] = _read_bool()
                metadata[f'{base_key}enable_per_instance_shadow_bias'] = _read_bool()
                reader.seek(2, 1)
                metadata[f'{base_key}shadow_map_bias'] = float(reader.read_f32())
                metadata[f'{base_key}shadow_map_slope_bias'] = float(reader.read_f32())
                metadata[f'{base_key}cull_light_distance'] = float(reader.read_f32())
                metadata[f'{base_key}enable_light_culling'] = _read_bool()
                reader.seek(3, 1)
                metadata[f'{base_key}cull_light_fade_distance'] = float(reader.read_f32())
                metadata[f'{base_key}shadow_cull_distance'] = float(reader.read_f32())
                metadata[f'{base_key}enable_shadow_lod'] = _read_bool()
                metadata[f'{base_key}active_in_gameplay'] = _read_bool()
                metadata[f'{base_key}active_in_cinematics'] = _read_bool()
                metadata[f'{base_key}affects_player'] = _read_bool()
                metadata[f'{base_key}affects_intros'] = _read_bool()
                metadata[f'{base_key}affects_bgobjects'] = _read_bool()
                metadata[f'{base_key}affects_terrain'] = _read_bool()
                metadata[f'{base_key}affects_neighboring_units'] = _read_bool()
                metadata[f'{base_key}affects_water'] = _read_bool()
                reader.seek(3, 1)
                cinematic_name_ptr = int(reader.read_u32())
                metadata[f'{base_key}cinematic_name_ptr'] = cinematic_name_ptr
                metadata[f'{base_key}cinematic_name'] = self._read_c_string(_RawBinaryReader(bytes(data)), cinematic_name_ptr) if 0 < cinematic_name_ptr < len(data) else ''
                metadata[f'{base_key}scene_light'] = int(reader.read_u32())
            return metadata
        except Exception as exc:
            logger.warning('Failed to parse ADMDData at 0x%X in %s: %s', absolute_offset, getattr(self, 'filepath', '<memory>'), exc)
            return metadata

    def _parse_admd_from_context(self, context: SectionContext, absolute_offset: int) -> dict[str, object]:
        if absolute_offset <= 0 or absolute_offset >= context.file_size:
            return {}
        try:
            with open(context.filepath, 'rb') as fh:
                data = fh.read()
        except Exception as exc:
            logger.warning('Failed to read %s for ADMDData import: %s', context.file_name, exc)
            return {}
        return self._parse_admd_blob(data, absolute_offset, section_id=context.section_info.section_id)


    @staticmethod
    def _raw_reader_has_range(reader: _RawBinaryReader, absolute_offset: int, size: int = 1) -> bool:
        try:
            absolute_offset = int(absolute_offset)
            size = int(size)
        except Exception:
            return False
        return absolute_offset >= 0 and size >= 0 and (absolute_offset + size) <= len(reader.data)

    @staticmethod
    def _read_sfx_sound_ids_from_raw(reader: _RawBinaryReader, absolute_offset: int, count: int, base_size: int) -> List[int]:
        sound_ids: List[int] = []
        try:
            absolute_offset = int(absolute_offset)
            count = max(0, min(int(count), 256))
            base_size = int(base_size)
        except Exception:
            return sound_ids
        id_offset = absolute_offset + base_size
        for index in range(count):
            field_offset = id_offset + (index * 4)
            if not TRLevelParser._raw_reader_has_range(reader, field_offset, 4):
                break
            sound_id = struct.unpack_from('<I', reader.data, field_offset)[0]
            if sound_id:
                sound_ids.append(int(sound_id))
        return sound_ids

    def _parse_sfx_periodic_sound_raw(self, reader: _RawBinaryReader, absolute_offset: int, index: int) -> Optional[LevelSFXSound]:
        if not self._raw_reader_has_range(reader, absolute_offset, _SFX_PERIODIC_SOUND_BASE_SIZE):
            return None
        try:
            num_sfx_ids = int(reader.data[absolute_offset])
            flags = int(reader.data[absolute_offset + 1])
            min_vol_distance = struct.unpack_from('<H', reader.data, absolute_offset + 2)[0]
            pitch = struct.unpack_from('<f', reader.data, absolute_offset + 4)[0]
            pitch_variation = struct.unpack_from('<f', reader.data, absolute_offset + 8)[0]
            max_volume = int(reader.data[absolute_offset + 12])
            max_vol_variation = int(reader.data[absolute_offset + 13])
            initial_delay = struct.unpack_from('<H', reader.data, absolute_offset + 14)[0]
            initial_delay_variation = struct.unpack_from('<H', reader.data, absolute_offset + 16)[0]
            on_time = struct.unpack_from('<H', reader.data, absolute_offset + 18)[0]
            on_time_variation = struct.unpack_from('<H', reader.data, absolute_offset + 20)[0]
            off_time = struct.unpack_from('<H', reader.data, absolute_offset + 22)[0]
            off_time_variation = struct.unpack_from('<H', reader.data, absolute_offset + 24)[0]
            return LevelSFXSound(
                index=int(index),
                kind='periodic',
                source_offset=int(absolute_offset),
                sound_ids=self._read_sfx_sound_ids_from_raw(reader, absolute_offset, num_sfx_ids, _SFX_PERIODIC_SOUND_BASE_SIZE),
                properties={
                    'num_sfx_ids': num_sfx_ids,
                    'flags': flags,
                    'min_vol_distance': int(min_vol_distance),
                    'pitch': float(pitch),
                    'pitch_variation': float(pitch_variation),
                    'max_volume': max_volume,
                    'max_vol_variation': max_vol_variation,
                    'initial_delay': int(initial_delay),
                    'initial_delay_variation': int(initial_delay_variation),
                    'on_time': int(on_time),
                    'on_time_variation': int(on_time_variation),
                    'off_time': int(off_time),
                    'off_time_variation': int(off_time_variation),
                },
            )
        except Exception as exc:
            logger.warning('Failed to parse SFX periodic sound at 0x%X in %s: %s', int(absolute_offset), getattr(self, 'filepath', '<memory>'), exc)
            return None

    def _parse_sfx_event_sound_raw(self, reader: _RawBinaryReader, absolute_offset: int, index: int) -> Optional[LevelSFXSound]:
        if not self._raw_reader_has_range(reader, absolute_offset, _SFX_EVENT_SOUND_BASE_SIZE):
            return None
        try:
            sound_group = int(reader.data[absolute_offset])
            num_sfx_ids = int(reader.data[absolute_offset + 1])
            min_vol_distance = struct.unpack_from('<H', reader.data, absolute_offset + 2)[0]
            pitch = struct.unpack_from('<f', reader.data, absolute_offset + 4)[0]
            pitch_variation = struct.unpack_from('<f', reader.data, absolute_offset + 8)[0]
            max_volume = int(reader.data[absolute_offset + 12])
            max_vol_variation = int(reader.data[absolute_offset + 13])
            delay = struct.unpack_from('<f', reader.data, absolute_offset + 16)[0]
            delay_variation = struct.unpack_from('<f', reader.data, absolute_offset + 20)[0]
            return LevelSFXSound(
                index=int(index),
                kind='event',
                source_offset=int(absolute_offset),
                sound_ids=self._read_sfx_sound_ids_from_raw(reader, absolute_offset, num_sfx_ids, _SFX_EVENT_SOUND_BASE_SIZE),
                properties={
                    'sound_group': sound_group,
                    'num_sfx_ids': num_sfx_ids,
                    'min_vol_distance': int(min_vol_distance),
                    'pitch': float(pitch),
                    'pitch_variation': float(pitch_variation),
                    'max_volume': max_volume,
                    'max_vol_variation': max_vol_variation,
                    'delay': float(delay),
                    'delay_variation': float(delay_variation),
                },
            )
        except Exception as exc:
            logger.warning('Failed to parse SFX event sound at 0x%X in %s: %s', int(absolute_offset), getattr(self, 'filepath', '<memory>'), exc)
            return None

    def _parse_sfx_one_shot_sound_raw(self, reader: _RawBinaryReader, absolute_offset: int, index: int) -> Optional[LevelSFXSound]:
        if not self._raw_reader_has_range(reader, absolute_offset, _SFX_ONESHOT_SOUND_BASE_SIZE):
            return None
        try:
            sound_group = int(reader.data[absolute_offset])
            num_sfx_ids = int(reader.data[absolute_offset + 1])
            min_vol_distance = struct.unpack_from('<H', reader.data, absolute_offset + 2)[0]
            pitch = struct.unpack_from('<f', reader.data, absolute_offset + 4)[0]
            pitch_variation = struct.unpack_from('<f', reader.data, absolute_offset + 8)[0]
            max_volume = int(reader.data[absolute_offset + 12])
            max_vol_variation = int(reader.data[absolute_offset + 13])
            return LevelSFXSound(
                index=int(index),
                kind='one_shot',
                source_offset=int(absolute_offset),
                sound_ids=self._read_sfx_sound_ids_from_raw(reader, absolute_offset, num_sfx_ids, _SFX_ONESHOT_SOUND_BASE_SIZE),
                properties={
                    'sound_group': sound_group,
                    'num_sfx_ids': num_sfx_ids,
                    'min_vol_distance': int(min_vol_distance),
                    'pitch': float(pitch),
                    'pitch_variation': float(pitch_variation),
                    'max_volume': max_volume,
                    'max_vol_variation': max_vol_variation,
                },
            )
        except Exception as exc:
            logger.warning('Failed to parse SFX one-shot sound at 0x%X in %s: %s', int(absolute_offset), getattr(self, 'filepath', '<memory>'), exc)
            return None

    def _parse_sfx_stream_sound_raw(self, reader: _RawBinaryReader, absolute_offset: int, index: int) -> Optional[LevelSFXSound]:
        if not self._raw_reader_has_range(reader, absolute_offset, _SFX_STREAM_SOUND_SIZE):
            return None
        try:
            choose_chance = struct.unpack_from('<h', reader.data, absolute_offset)[0]
            play_chance = struct.unpack_from('<h', reader.data, absolute_offset + 2)[0]
            min_vol_distance = struct.unpack_from('<h', reader.data, absolute_offset + 4)[0]
            max_volume = struct.unpack_from('<h', reader.data, absolute_offset + 6)[0]
            music_vars = tuple(int(reader.data[absolute_offset + 8 + i]) for i in range(4))
            name_bytes = bytes(reader.data[absolute_offset + 12:absolute_offset + 76])
            name = name_bytes.split(b'\x00', 1)[0].decode('utf-8', errors='replace')
            return LevelSFXSound(
                index=int(index),
                kind='stream',
                source_offset=int(absolute_offset),
                sound_ids=[],
                properties={
                    'choose_chance': int(choose_chance),
                    'play_chance': int(play_chance),
                    'min_vol_distance': int(min_vol_distance),
                    'max_volume': int(max_volume),
                    'music_vars': music_vars,
                    'name': name,
                },
            )
        except Exception as exc:
            logger.warning('Failed to parse SFX stream sound at 0x%X in %s: %s', int(absolute_offset), getattr(self, 'filepath', '<memory>'), exc)
            return None

    def _parse_sfx_pointer_targets_raw(
        self,
        reader: _RawBinaryReader,
        count: int,
        list_pointer: int,
        fixed_array_offset: int,
        fixed_capacity: int = _SFX_POINTER_ARRAY_CAPACITY,
    ) -> List[int]:
        try:
            count = max(0, min(int(count), 4096))
            list_pointer = int(list_pointer or 0)
            fixed_array_offset = int(fixed_array_offset or 0)
            fixed_capacity = max(0, min(int(fixed_capacity), 256))
        except Exception:
            return []
        targets: List[int] = []
        if list_pointer > 0 and count > 0 and self._raw_reader_has_range(reader, list_pointer, count * 4):
            for index in range(count):
                target = struct.unpack_from('<I', reader.data, list_pointer + (index * 4))[0]
                if target > 0 and self._raw_reader_has_range(reader, target, 1):
                    targets.append(int(target))
        if targets:
            return targets
        array_count = fixed_capacity if count <= 0 else min(count, fixed_capacity)
        if array_count <= 0 or not self._raw_reader_has_range(reader, fixed_array_offset, array_count * 4):
            return targets
        for index in range(array_count):
            target = struct.unpack_from('<I', reader.data, fixed_array_offset + (index * 4))[0]
            if target > 0 and self._raw_reader_has_range(reader, target, 1):
                targets.append(int(target))
        return targets

    def _parse_sfx_perimeter_raw(self, reader: _RawBinaryReader, absolute_offset: int, index: int) -> Optional[LevelSFXPerimeter]:
        if not self._raw_reader_has_range(reader, absolute_offset, _SFX_PERIMETER_SIZE):
            return None
        try:
            flags = struct.unpack_from('<H', reader.data, absolute_offset + 12)[0]
            return LevelSFXPerimeter(
                index=int(index),
                source_offset=int(absolute_offset),
                radius=int(struct.unpack_from('<I', reader.data, absolute_offset)[0]),
                n_fired=int(struct.unpack_from('<I', reader.data, absolute_offset + 4)[0]),
                invader_offset=int(struct.unpack_from('<I', reader.data, absolute_offset + 8)[0]),
                action=int(flags & 0xFF),
                breached=int((flags >> 8) & 0x01),
                on_exit=int((flags >> 14) & 0x01),
                always=int((flags >> 15) & 0x01),
                event_offset=int(struct.unpack_from('<I', reader.data, absolute_offset + 16)[0]),
                periodic_offset=int(struct.unpack_from('<I', reader.data, absolute_offset + 20)[0]),
                varnum=int(reader.data[absolute_offset + 24]),
                value=int(reader.data[absolute_offset + 25]),
            )
        except Exception as exc:
            logger.warning('Failed to parse SFX perimeter action at 0x%X in %s: %s', int(absolute_offset), getattr(self, 'filepath', '<memory>'), exc)
            return None

    def _parse_sfx_markers_raw(self, reader: _RawBinaryReader, marker_list: int, count: int) -> List[LevelSFXMarker]:
        markers: List[LevelSFXMarker] = []
        try:
            marker_list = int(marker_list or 0)
            count = max(0, min(int(count or 0), _SFX_MARKER_MAX_IMPORT_COUNT))
        except Exception:
            return markers
        if count <= 0 or not self._raw_reader_has_range(reader, marker_list, count * _SFX_MARKER_SIZE):
            return markers

        for marker_index in range(count):
            marker_offset = marker_list + (marker_index * _SFX_MARKER_SIZE)
            try:
                x, y, z = struct.unpack_from('<fff', reader.data, marker_offset)
                unique_id = struct.unpack_from('<i', reader.data, marker_offset + 12)[0]
                sound_instance = struct.unpack_from('<I', reader.data, marker_offset + 16)[0]
                plane = int(reader.data[marker_offset + 20])
                spline_id = struct.unpack_from('<I', reader.data, marker_offset + 24)[0]
                sfx_data_offset = marker_offset + 28
                num_periodic = struct.unpack_from('<I', reader.data, sfx_data_offset)[0]
                periodic_list = struct.unpack_from('<I', reader.data, sfx_data_offset + 4)[0]
                num_event = struct.unpack_from('<I', reader.data, sfx_data_offset + 8)[0]
                event_list = struct.unpack_from('<I', reader.data, sfx_data_offset + 12)[0]
                num_one_shot = struct.unpack_from('<I', reader.data, sfx_data_offset + 16)[0]
                one_shot_list = struct.unpack_from('<I', reader.data, sfx_data_offset + 20)[0]
                num_stream = struct.unpack_from('<I', reader.data, sfx_data_offset + 24)[0]
                stream_list = struct.unpack_from('<I', reader.data, sfx_data_offset + 28)[0]

                event_array_offset = marker_offset + 68
                periodic_array_offset = marker_offset + 132
                perimeter_array_offset = marker_offset + 196

                event_sounds: List[LevelSFXSound] = []
                for entry_index, target in enumerate(self._parse_sfx_pointer_targets_raw(reader, num_event, event_list, event_array_offset)):
                    parsed = self._parse_sfx_event_sound_raw(reader, target, entry_index)
                    if parsed is not None:
                        event_sounds.append(parsed)

                periodic_sounds: List[LevelSFXSound] = []
                for entry_index, target in enumerate(self._parse_sfx_pointer_targets_raw(reader, num_periodic, periodic_list, periodic_array_offset)):
                    parsed = self._parse_sfx_periodic_sound_raw(reader, target, entry_index)
                    if parsed is not None:
                        periodic_sounds.append(parsed)

                one_shot_sounds: List[LevelSFXSound] = []
                for entry_index, target in enumerate(self._parse_sfx_pointer_targets_raw(reader, num_one_shot, one_shot_list, 0, fixed_capacity=0)):
                    parsed = self._parse_sfx_one_shot_sound_raw(reader, target, entry_index)
                    if parsed is not None:
                        one_shot_sounds.append(parsed)

                stream_sounds: List[LevelSFXSound] = []
                for entry_index, target in enumerate(self._parse_sfx_pointer_targets_raw(reader, num_stream, stream_list, 0, fixed_capacity=0)):
                    parsed = self._parse_sfx_stream_sound_raw(reader, target, entry_index)
                    if parsed is not None:
                        stream_sounds.append(parsed)

                perimeters: List[LevelSFXPerimeter] = []
                for entry_index, target in enumerate(self._parse_sfx_pointer_targets_raw(reader, _SFX_POINTER_ARRAY_CAPACITY, 0, perimeter_array_offset)):
                    parsed = self._parse_sfx_perimeter_raw(reader, target, entry_index)
                    if parsed is not None:
                        perimeters.append(parsed)

                markers.append(LevelSFXMarker(
                    index=int(marker_index),
                    source_offset=int(marker_offset),
                    position=_convert_level_position(float(x), float(y), float(z)),
                    raw_position=(float(x), float(y), float(z)),
                    unique_id=int(unique_id),
                    plane=plane,
                    spline_id=int(spline_id),
                    sound_instance_offset=int(sound_instance),
                    periodic_sounds=periodic_sounds,
                    event_sounds=event_sounds,
                    one_shot_sounds=one_shot_sounds,
                    stream_sounds=stream_sounds,
                    perimeter_actions=perimeters,
                ))
            except Exception as exc:
                logger.warning('Failed to parse SFX marker index=%d at 0x%X in %s: %s', int(marker_index), marker_offset, getattr(self, 'filepath', '<memory>'), exc)
        return markers


    @staticmethod
    def _find_raw_section_by_absolute_offset(sections: _RawSectionList, absolute_offset: int) -> Optional[_RawSection]:
        absolute_offset = int(absolute_offset)
        if absolute_offset <= 0:
            return None
        for section in sections.sections:
            start = int(section.offset)
            end = start + int(section.size)
            if start <= absolute_offset < end:
                return section
        return None

    def _populate_drm_level_metadata(self, reader: _RawBinaryReader, sections: _RawSectionList, root_section: _RawSection, level: LevelData) -> None:
        base = int(root_section.offset)
        end = base + int(root_section.size)
        if base <= 0 or end > len(reader.data):
            return

        metadata = self._read_level_metadata_from_raw_reader(reader, base)
        num_markups = reader.read_i32()
        markup_list = reader.read_u32()
        num_cameras = reader.read_i32()
        camera_list = reader.read_u32()
        metadata['flags'] = int(reader.read_i32())
        num_intros = reader.read_i32()
        intro_list = reader.read_u32()
        reader.read_u32()  # objectNameList
        world_name_offset = reader.read_u32()
        metadata['worldName'] = self._read_c_string(reader, world_name_offset)
        reader.read_i32()  # startGoingIntoWaterSignal
        reader.read_i32()  # startGoingOutOfWaterSignal
        metadata['unitFlags'] = int(reader.read_i32())
        signal_list_start = reader.read_u32()
        signal_id_list = reader.read_u32()
        reader.read_u32()  # splineCameraData
        reader.read_u32()  # relocModule
        number_of_sfx_markers = reader.read_u32()
        sfx_marker_list = reader.read_u32()
        level.sfx_markers = self._parse_sfx_markers_raw(reader, sfx_marker_list, number_of_sfx_markers)
        metadata['NumberOfSFXMarkers'] = int(number_of_sfx_markers)
        version_number = reader.read_u32()
        metadata['guiID'] = int(reader.read_u32())
        reader.read_u32()  # dynamicMusicName
        metadata['streamUnitID'] = int(reader.read_i32())
        reader.read_u32()  # textureLoadList
        reader.read_u32()  # planData
        reader.read_u32()  # mapRegion
        reader.read_u32()  # WeatherHeightmapData
        reader.read_u32()  # attackWaveList
        reader.read_u32()  # attackWaveGroupList
        reader.read_u32()  # combatDoorsList
        num_terrain_lights = reader.read_i32()
        terrain_lights = reader.read_u32()
        terrain_light_grids = reader.read_u32()
        if terrain_light_grids > 0:
            level.terrain_light_grid_cells = self._parse_terrain_light_grid_cells_from_raw_reader(reader, terrain_light_grids)
        reader.read_u32()  # vmarkerList
        reader.read_u32()  # pmarkerList
        player_name_offset = reader.read_u32()
        metadata['playerName'] = self._read_c_string(reader, player_name_offset)
        reader.read_u32()  # levelCount
        move_data_offset = reader.read_u32()
        planner_data_offset = reader.read_u32()
        runtime_area_dbase_offset = reader.read_u32()

        area_pointer_values = (
            ('areaDBaseMoveDataOffset', int(move_data_offset)),
            ('areaDBasePlannerDataOffset', int(planner_data_offset)),
            ('areaDBaseRuntimeObjectOffset', int(runtime_area_dbase_offset)),
        )
        area_section = None
        for metadata_key, absolute_pointer in area_pointer_values:
            target_section = self._find_raw_section_by_absolute_offset(sections, absolute_pointer)
            if target_section is None:
                continue
            if area_section is None:
                area_section = target_section
            if target_section is area_section:
                metadata[metadata_key] = int(absolute_pointer) - int(target_section.offset)
        if area_section is not None:
            try:
                metadata['areaDBaseSourceSectionIndex'] = int(sections.sections.index(area_section))
            except ValueError:
                pass
            if 'areaDBasePlannerDataOffset' in metadata:
                metadata['areaDBaseHeaderContentOffset'] = int(metadata['areaDBasePlannerDataOffset'])

        unit_data_offset = reader.read_u32()
        unit_data_section = self._find_raw_section_by_absolute_offset(sections, unit_data_offset)
        level.unit_data = self._parse_unit_data_blob(reader.data, unit_data_offset, getattr(level, 'source_game', self.game_hint), section_id=getattr(unit_data_section, 'id', None), parse_pointer_arrays=True, raw_sections=sections)
        admd_data_offset = reader.read_u32()
        admd_data_section = self._find_raw_section_by_absolute_offset(sections, admd_data_offset)
        level.admd_data = self._parse_admd_blob(reader.data, admd_data_offset, section_id=getattr(admd_data_section, 'id', None))
        for _ in range(3):
            reader.read_u32()
        scx = reader.read_f32()
        scy = reader.read_f32()
        scz = reader.read_f32()
        scw = reader.read_f32()
        level.scene_center_offset = (_convert_level_position(scx, scy, scz) + (float(scw),))
        metadata['playerObjectID'] = int(reader.read_i16())
        reader.seek(14, 1)
        self._apply_level_metadata(level, metadata)

        if num_markups > 0 and 0 < markup_list < len(reader.data):
            level.markups = self._parse_raw_markups(reader, markup_list, num_markups, getattr(level, 'source_game', self.game_hint))

        if num_terrain_lights > 0 and 0 < terrain_lights < len(reader.data):
            reader.seek(terrain_lights)
            for index in range(num_terrain_lights):
                x = reader.read_i32(); y = reader.read_i32(); z = reader.read_i32()
                radius = reader.read_i32()
                r = reader.read_u8(); g = reader.read_u8(); b = reader.read_u8(); light_type = reader.read_u8()
                multiplier = reader.read_i16(); hotspot_angle = reader.read_i16()
                nx = reader.read_i16(); ny = reader.read_i16(); nz = reader.read_i16()
                falloff_angle = reader.read_i16(); light_id = reader.read_i16(); reader.seek(2, 1)
                level.terrain_lights.append(TerrainLightData(
                    index=index,
                    position=_convert_level_position(x, y, z),
                    radius=int(radius),
                    color=(max(0.0, min(1.0, r / 255.0)), max(0.0, min(1.0, g / 255.0)), max(0.0, min(1.0, b / 255.0))),
                    type=int(light_type),
                    multiplier=int(multiplier),
                    hotspot_angle=int(hotspot_angle),
                    direction=_convert_bgobject_basis_vector(nx, ny, nz),
                    falloff_angle=int(falloff_angle),
                    light_id=int(light_id),
                ))

        if num_cameras > 0 and 0 < camera_list < len(reader.data):
            reader.seek(camera_list)
            for index in range(num_cameras):
                x = reader.read_i16(); y = reader.read_i16(); z = reader.read_i16(); camera_id = reader.read_i16()
                rx = reader.read_i16(); ry = reader.read_i16(); rz = reader.read_i16(); flags = reader.read_i16()
                tx = reader.read_i16(); ty = reader.read_i16(); tz = reader.read_i16(); reader.read_i16()
                level.camera_data.append(CameraKeyData(
                    index=index,
                    position=_convert_level_position(x, y, z),
                    camera_id=int(camera_id),
                    rotation=(int(rx), int(ry), int(rz)),
                    flags=int(flags),
                    target=(int(tx), int(ty), int(tz)),
                ))

        if num_intros > 0 and 0 < intro_list < len(reader.data):
            reader.seek(intro_list)
            for index in range(num_intros):
                rx = reader.read_f32()
                ry = reader.read_f32()
                rz = reader.read_f32()
                rw = reader.read_f32()
                rotation = (rx, ry, rz, rw)
                px = reader.read_f32(); py = reader.read_f32(); pz = reader.read_f32(); _pw = reader.read_f32()
                dummy1 = (reader.read_f32(), reader.read_f32(), reader.read_f32(), reader.read_f32())
                dummy2 = (reader.read_f32(), reader.read_f32(), reader.read_f32(), reader.read_f32())
                reader.read_f32(); reader.read_f32(); reader.read_f32(); reader.read_f32()
                scale = (1.0, 1.0, 1.0, 0.0)
                object_id = reader.read_i16(); intro_num = reader.read_i16(); unique_id = reader.read_i32()
                max_rad = reader.read_f32(); intro_flags = reader.read_u32(); attached_vmo = reader.read_i32()
                data = reader.read_i32(); multi_spline = reader.read_i32(); start_frame = reader.read_i16(); end_frame = reader.read_i16()
                level.intro_data.append(IntroData(
                    index=index,
                    object_id=int(object_id),
                    intro_num=int(intro_num),
                    unique_id=int(unique_id),
                    position=_convert_level_position(px, py, pz),
                    rotation=tuple(float(v) for v in rotation),
                    dummy1=tuple(float(v) for v in dummy1),
                    dummy2=tuple(float(v) for v in dummy2),
                    scale=tuple(float(v) for v in scale),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    intro_flags=int(intro_flags),
                    attached_vmo=int(attached_vmo),
                    data=int(data),
                    multi_spline=int(multi_spline),
                    max_radius=float(max_rad),
                    multi_spline_data=_read_multi_spline_from_raw_reader(reader, int(multi_spline), source_raw_value=int(multi_spline)),
                ))

    def _load_drm_texture_images(self, reader: _RawBinaryReader, sections: _RawSectionList) -> dict[int, str]:
        texture_images: dict[int, str] = {}
        for section_index, section in enumerate(sections.sections):
            if int(section.type) != 5:
                continue
            try:
                image = self._create_image_from_texture_section(reader, section_index, section)
            except Exception as exc:
                logger.warning('Failed to decode DRM texture section %s (id=%s): %s', section.type, section.id, exc)
                continue
            if image is not None:
                texture_images[int(section.id)] = image.name
        logger.debug('Decoded %d DRM texture images from %s', len(texture_images), self.filepath)
        return texture_images

    def _create_image_from_texture_section(self, reader: _RawBinaryReader, section_index: int, section: _RawSection) -> Optional[bpy.types.Image]:
        base = int(section.offset)
        if base <= 0 or (base + 20) > len(reader.data):
            return None

        format_ = struct.unpack_from('<I', reader.data, base + 4)[0]
        size = struct.unpack_from('<I', reader.data, base + 8)[0]
        width = struct.unpack_from('<H', reader.data, base + 16)[0]
        height = struct.unpack_from('<H', reader.data, base + 18)[0]
        data_offset = base + 24
        data_end = data_offset + int(size)
        if width <= 0 or height <= 0 or data_end > len(reader.data):
            return None

        image_name = f'{int(section_index)}_{int(section.id):x}'
        existing = bpy.data.images.get(image_name)
        if existing is not None:
            return existing

        bitmap_data = bytes(reader.data[data_offset:data_end])
        if format_ in (D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8):
            image = bpy.data.images.new(
                image_name,
                width=width,
                height=height,
                alpha=(format_ == D3DFMT_A8R8G8B8),
            )
            image.pixels[:] = _convert_pcd_argb32_to_rgba(
                bitmap_data,
                width,
                height,
                'alpha' if format_ == D3DFMT_A8R8G8B8 else 'opaque',
            )
            image.pack()
            return image

        dds_bytes = _create_dds_header(width, height, 1, format_, len(bitmap_data)) + bitmap_data
        temp_dir = Path(tempfile.mkdtemp(prefix='trlau_editor_drm_import_'))
        try:
            temp_dds_path = temp_dir / f'{image_name}.dds'
            with open(temp_dds_path, 'wb') as fh:
                fh.write(dds_bytes)

            image = bpy.data.images.load(str(temp_dds_path), check_existing=True)
            image.name = image_name
            image.pack()
            return image
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _parse_extracted_level(self) -> LevelData:
        cache = SectionContextCache(self.filepath)
        try:
            context = cache.get_root_context()
            if not self._context_looks_like_level(context):
                raise ValueError('File does not appear to contain TRLAU level data')

            terrain_raw, terrain_ctx, terrain_abs = self._resolve_root_pointer(cache, context, 0)
            if terrain_raw == 0 or terrain_abs <= 0:
                raise ValueError('Level terrain header pointer is null')

            level = self._read_level_data(cache, terrain_ctx, terrain_abs)
            inferred_game = self._normalize_level_game(self.game_hint)
            if inferred_game == 'unknown':
                inferred_game = 'legend' if int(getattr(level, 'cdc_render_data_id', 0)) != 0 else 'anniversary'
            level.source_game = inferred_game
            self.level_cdc_render_data_id = int(getattr(level, 'cdc_render_data_id', 0))
            self._populate_extracted_level_metadata(cache, context, level)
            return level
        finally:
            cache.close()

    def _context_looks_like_level(self, context: SectionContext) -> bool:
        marker_offset = context.data_start + 168
        if marker_offset + 4 > context.file_size:
            return False
        context.reader.seek(marker_offset)
        marker = context.reader.u32()
        logger.debug('Level marker for %s at 0x%X = 0x%X', context.file_name, marker_offset, marker)
        return marker == _LEVEL_VERSION

    def _is_valid_abs(self, context: SectionContext, offset: int, size: int = 1) -> bool:
        return 0 <= int(offset) <= max(0, context.file_size - size)

    def _resolve_root_pointer(self, cache: SectionContextCache, context: SectionContext, local_offset: int):
        context.reader.seek(context.data_start + local_offset)
        raw_value = context.reader.u32()
        target_context, target_abs = resolve_pointer(cache, context, local_offset, raw_value)
        return raw_value, target_context, target_abs

    def _resolve_pointer_at_absolute(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int):
        if not self._is_valid_abs(context, absolute_offset, 4):
            return 0, context, 0
        context.reader.seek(absolute_offset)
        raw_value = context.reader.u32()
        field_local_offset = absolute_offset - context.data_start
        if field_local_offset < 0:
            return raw_value, context, raw_value
        target_context, target_abs = resolve_pointer(cache, context, field_local_offset, raw_value)
        return raw_value, target_context, target_abs

    def _read_section_signal_id_list_entry(self, signal_id_context: SectionContext, signal_id_list_abs: int, signal_index: int) -> Optional[int]:
        signal_id_list_abs = int(signal_id_list_abs or 0)
        signal_index = int(signal_index)
        if signal_id_list_abs <= 0 or signal_index < 0 or signal_index >= _SIGNAL_MAX_IMPORT_COUNT:
            return None
        entry_abs = signal_id_list_abs + (signal_index * 2)
        if not self._is_valid_abs(signal_id_context, entry_abs, 2):
            return None
        id_reader = signal_id_context.reader
        previous = id_reader.tell()
        try:
            id_reader.seek(entry_abs)
            return int(id_reader.i16())
        finally:
            try:
                id_reader.seek(previous)
            except Exception:
                pass

    def _build_signals_from_section_context(
        self,
        cache: SectionContextCache,
        source_mesh: Optional[TerrainSignalMesh],
        signal_context: SectionContext,
        signal_list_abs: int,
        signal_id_context: SectionContext,
        signal_id_list_abs: int,
        *,
        start_going_into_water_abs: int = 0,
        start_going_out_of_water_abs: int = 0,
    ) -> List[TerrainSignal]:
        split_meshes = _split_signal_mesh_by_face_id(source_mesh)
        if not split_meshes:
            return []
        signals: List[TerrainSignal] = []
        signal_reader = signal_context.reader
        previous_signal_pos = signal_reader.tell()
        try:
            for default_index, (face_id, mesh) in enumerate(split_meshes):
                signal_index = int(face_id) if 0 <= int(face_id) < _SIGNAL_MAX_IMPORT_COUNT else int(default_index)
                level_signal_id = self._read_section_signal_id_list_entry(signal_id_context, signal_id_list_abs, signal_index)
                if level_signal_id is None:
                    level_signal_id = int(face_id)
                signal_abs = int(signal_list_abs or 0) + (signal_index * _SIGNAL_STRUCT_SIZE) if signal_list_abs else 0
                props: dict[str, object] = {}
                if signal_abs > 0 and self._is_valid_abs(signal_context, signal_abs, _SIGNAL_STRUCT_SIZE):
                    try:
                        props = _read_signal_from_section_context(
                            cache,
                            signal_context,
                            signal_abs,
                            start_going_into_water_abs=int(start_going_into_water_abs or 0),
                            start_going_out_of_water_abs=int(start_going_out_of_water_abs or 0),
                            signal_list_start_abs=int(signal_list_abs or 0),
                        )
                    except Exception as exc:
                        logger.warning('Failed reading Signal index=%d face_id=%d at %s:0x%X: %s', int(signal_index), int(face_id), signal_context.file_name, signal_abs, exc)
                signals.append(TerrainSignal(
                    index=int(signal_index),
                    signal_id=int(level_signal_id),
                    face_id=int(face_id),
                    mesh=mesh,
                    properties=props,
                ))
        finally:
            try:
                signal_reader.seek(previous_signal_pos)
            except Exception:
                pass
        return signals

    def _read_markup_polyline_section(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int) -> List[Tuple[float, float, float, float]]:
        polyline: List[Tuple[float, float, float, float]] = []
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 16):
            return polyline
        br = context.reader
        br.seek(absolute_offset)
        num_points = br.i32()
        if num_points <= 0 or num_points > 8192:
            return polyline
        data_start = absolute_offset + 16
        if not self._is_valid_abs(context, data_start, num_points * 16):
            return polyline
        br.seek(data_start)
        for _ in range(num_points):
            x = br.f32()
            y = br.f32()
            z = br.f32()
            w = br.f32()
            converted = _convert_level_position(x, y, z)
            polyline.append((float(converted[0]), float(converted[1]), float(converted[2]), float(w)))
        return polyline

    def _parse_section_markups(self, cache: SectionContextCache, context: SectionContext, list_abs: int, count: int, game: str) -> List[MarkupData]:
        markups: List[MarkupData] = []
        if not self.import_markups or count <= 0 or list_abs <= 0:
            return markups
        normalized_game = self._normalize_level_game(game)
        if normalized_game == 'unknown':
            normalized_game = 'legend' if int(getattr(self, 'level_cdc_render_data_id', 0)) != 0 else 'anniversary'
        entry_size = 76 if normalized_game == 'anniversary' else 48
        requested_count = max(0, min(int(count), 65535))
        br = context.reader
        parse_count = 0
        for index in range(requested_count):
            if not self._is_valid_abs(context, list_abs + (index * entry_size), entry_size):
                break
            parse_count += 1
        if parse_count < requested_count:
            logger.warning('MarkUp section list at 0x%X in %s has count=%d but only %d complete entries fit; importing the complete entries', list_abs, context.file_name, count, parse_count)
        if parse_count <= 0:
            return markups
        for index in range(parse_count):
            entry_abs = list_abs + (index * entry_size)
            br.seek(entry_abs)
            override_movement_camera = br.i32()
            dtp_camera_data_id = br.i32()
            dtp_markup_data_id = br.i32()
            animated_segment = 0
            camera_antic = None
            if normalized_game == 'anniversary':
                animated_segment = br.i32()
                use_antic_camera = br.i32()
                antic_ids = tuple(int(br.i32()) for _ in range(5))
                camera_antic = CameraAnticData(use_antic_camera=int(use_antic_camera), new_dtp_camera_antic_data_id=antic_ids)
            flags = br.u32()
            intro_id = br.i16()
            markup_id = br.i16()
            px = br.f32(); py = br.f32(); pz = br.f32()
            bbox = tuple(int(br.i16()) for _ in range(6))
            polyline_field_abs = br.tell()
            polyline_raw, polyline_ctx, polyline_abs = self._resolve_pointer_at_absolute(cache, context, polyline_field_abs)
            polyline = self._read_markup_polyline_section(cache, polyline_ctx, polyline_abs if polyline_abs > 0 else polyline_raw)
            markups.append(MarkupData(
                index=index,
                game=normalized_game,
                override_movement_camera=int(override_movement_camera),
                dtp_camera_data_id=int(dtp_camera_data_id),
                dtp_markup_data_id=int(dtp_markup_data_id),
                animated_segment=int(animated_segment),
                camera_antic=camera_antic,
                flags=int(flags),
                intro_id=int(intro_id),
                markup_id=int(markup_id),
                position=_convert_level_position(px, py, pz),
                bbox=bbox,
                polyline_offset=int(polyline_abs if polyline_abs > 0 else polyline_raw),
                polyline=polyline,
            ))
        return markups

    def _populate_extracted_level_metadata(self, cache: SectionContextCache, context: SectionContext, level: LevelData) -> None:
        base = context.data_start
        if not self._is_valid_abs(context, base + 228, 2):
            return
        metadata = self._read_level_metadata_from_section_context(context, base)
        br = context.reader
        br.seek(base + 96)
        num_markups = br.i32()
        _markup_raw, markup_ctx, markup_abs = self._resolve_root_pointer(cache, context, 100)
        br.seek(base + 112)
        metadata['flags'] = int(br.i32())
        num_intros = br.i32()
        _intro_raw, intro_ctx, intro_abs = self._resolve_root_pointer(cache, context, 120)
        _world_name_raw, world_name_ctx, world_name_abs = self._resolve_root_pointer(cache, context, 128)
        metadata['worldName'] = self._read_c_string_from_section_context(world_name_ctx, world_name_abs if world_name_abs > 0 else _world_name_raw)
        start_water_in_raw, start_water_in_ctx, start_water_in_abs = self._resolve_root_pointer(cache, context, 132)
        start_water_out_raw, start_water_out_ctx, start_water_out_abs = self._resolve_root_pointer(cache, context, 136)
        br.seek(base + 140)
        metadata['unitFlags'] = int(br.i32())
        signal_list_raw, signal_list_ctx, signal_list_abs = self._resolve_root_pointer(cache, context, 144)
        signal_id_list_raw, signal_id_list_ctx, signal_id_list_abs = self._resolve_root_pointer(cache, context, 148)
        level.reloc_module = self._read_extracted_reloc_module(cache, context)
        br.seek(base + 172)
        metadata['guiID'] = int(br.u32())
        br.u32()  # dynamicMusicName
        metadata['streamUnitID'] = int(br.i32())
        _player_name_raw, player_name_ctx, player_name_abs = self._resolve_root_pointer(cache, context, 188)
        metadata['playerName'] = self._read_c_string_from_section_context(player_name_ctx, player_name_abs if player_name_abs > 0 else _player_name_raw)
        _unit_data_raw, unit_data_ctx, unit_data_abs = self._resolve_root_pointer(cache, context, 208)
        level.unit_data = self._parse_unit_data_from_context(cache, unit_data_ctx, unit_data_abs if unit_data_abs > 0 else _unit_data_raw, getattr(level, 'source_game', self.game_hint))
        _admd_raw, admd_ctx, admd_abs = self._resolve_root_pointer(cache, context, 212)
        level.admd_data = self._parse_admd_from_context(admd_ctx, admd_abs if admd_abs > 0 else _admd_raw)
        br.seek(base + 228)
        metadata['playerObjectID'] = int(br.i16())
        self._apply_level_metadata(level, metadata)
        if self.import_signals and getattr(level, 'signal_mesh', None) is not None:
            resolved_level_signal_list_abs = signal_list_abs if signal_list_abs > 0 else signal_list_raw
            resolved_signal_list_abs = resolved_level_signal_list_abs
            resolved_signal_list_ctx = signal_list_ctx

            # Prefer Terrain::signals when present.  The signal terrain mesh is owned
            # by Terrain::signalTerrainGroup, and Terrain::signals is the matching
            # Terrain-owned Signal* list.
            terrain_signal_list_abs = int(getattr(level, 'terrain_signal_list_abs', 0) or 0)
            terrain_signal_list_ctx = getattr(level, 'terrain_signal_list_context', None)
            if terrain_signal_list_abs > 0 and terrain_signal_list_ctx is not None and self._is_valid_abs(terrain_signal_list_ctx, terrain_signal_list_abs, _SIGNAL_STRUCT_SIZE):
                resolved_signal_list_abs = terrain_signal_list_abs
                resolved_signal_list_ctx = terrain_signal_list_ctx

            resolved_signal_id_list_abs = signal_id_list_abs if signal_id_list_abs > 0 else signal_id_list_raw
            resolved_start_water_in = start_water_in_abs if start_water_in_abs > 0 else start_water_in_raw
            resolved_start_water_out = start_water_out_abs if start_water_out_abs > 0 else start_water_out_raw
            if resolved_signal_list_abs > 0:
                level.signals = self._build_signals_from_section_context(
                    cache,
                    getattr(level, 'signal_mesh'),
                    resolved_signal_list_ctx,
                    resolved_signal_list_abs,
                    signal_id_list_ctx,
                    resolved_signal_id_list_abs,
                    start_going_into_water_abs=resolved_start_water_in,
                    start_going_out_of_water_abs=resolved_start_water_out,
                )
        if num_intros > 0 and intro_abs > 0 and self._is_valid_abs(intro_ctx, intro_abs, num_intros * 112):
            intro_reader = intro_ctx.reader
            for index in range(num_intros):
                entry_abs = intro_abs + (index * 112)
                intro_reader.seek(entry_abs)
                rx = intro_reader.f32()
                ry = intro_reader.f32()
                rz = intro_reader.f32()
                rw = intro_reader.f32()
                rotation = (rx, ry, rz, rw)
                px = intro_reader.f32(); py = intro_reader.f32(); pz = intro_reader.f32(); _pw = intro_reader.f32()
                dummy1 = (intro_reader.f32(), intro_reader.f32(), intro_reader.f32(), intro_reader.f32())
                dummy2 = (intro_reader.f32(), intro_reader.f32(), intro_reader.f32(), intro_reader.f32())
                intro_reader.f32(); intro_reader.f32(); intro_reader.f32(); intro_reader.f32()
                scale = (1.0, 1.0, 1.0, 0.0)
                object_id = intro_reader.i16(); intro_num = intro_reader.i16(); unique_id = intro_reader.i32()
                max_rad = intro_reader.f32(); intro_flags = intro_reader.u32(); attached_vmo = intro_reader.i32()
                data = intro_reader.i32(); multi_spline = intro_reader.i32(); start_frame = intro_reader.i16(); end_frame = intro_reader.i16()
                intro_data_block = self._read_intro_data_block_from_context(cache, intro_ctx, entry_abs + 100)
                level.intro_data.append(IntroData(
                    index=index,
                    object_id=int(object_id),
                    intro_num=int(intro_num),
                    unique_id=int(unique_id),
                    position=_convert_level_position(px, py, pz),
                    rotation=tuple(float(v) for v in rotation),
                    dummy1=tuple(float(v) for v in dummy1),
                    dummy2=tuple(float(v) for v in dummy2),
                    scale=tuple(float(v) for v in scale),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    intro_flags=int(intro_flags),
                    attached_vmo=int(attached_vmo),
                    data=int(data),
                    multi_spline=int(multi_spline),
                    max_radius=float(max_rad),
                    intro_data_block=intro_data_block,
                    multi_spline_data=self._read_multi_spline_from_context(cache, intro_ctx, entry_abs + 104),
                ))
        if self.import_markups and num_markups > 0 and markup_abs > 0:
            level.markups = self._parse_section_markups(cache, markup_ctx, markup_abs, num_markups, getattr(level, 'source_game', self.game_hint))

    def _read_level_data(self, cache: SectionContextCache, terrain_ctx: SectionContext, terrain_abs: int) -> LevelData:
        if not self._is_valid_abs(terrain_ctx, terrain_abs, 0x58):
            raise ValueError(f'Invalid terrain header offset 0x{terrain_abs:X} in {terrain_ctx.file_name}')

        br = terrain_ctx.reader
        br.seek(terrain_abs)

        br.skip(12)
        num_stream_unit_portals = br.i32()
        stream_unit_portals_field_abs = br.tell()
        _stream_portals_raw, stream_portals_ctx, stream_portals_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, stream_unit_portals_field_abs)
        num_groups = br.i32()
        terrain_groups_field_abs = br.tell()
        _groups_raw, groups_ctx, groups_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, terrain_groups_field_abs)
        signal_terrain_group_field_abs = br.tell()
        signal_tg_raw, signal_tg_ctx, signal_tg_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, signal_terrain_group_field_abs)
        terrain_signals_field_abs = br.tell()
        terrain_signals_raw, terrain_signals_ctx, terrain_signals_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, terrain_signals_field_abs)
        br.skip(4)  # terrainAnimTextures

        num_bg_instances = br.u32()
        bg_instances_field_abs = br.tell()
        _bginst_raw, bginst_ctx, bginst_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, bg_instances_field_abs)

        num_bg_objects = br.u32()
        bg_objects_field_abs = br.tell()
        _bg_raw, bg_ctx, bg_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, bg_objects_field_abs)

        br.skip(12)
        xboxpc_vertex_buffer_field_abs = br.tell()
        _xbox_vertex_raw, xbox_vertex_ctx, xbox_vertex_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, xboxpc_vertex_buffer_field_abs)
        vmo_buffer_field_abs = br.tell()
        _vmo_raw, vmo_ctx, vmo_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, vmo_buffer_field_abs)
        br.skip(8)
        num_vertices = br.i32()
        vertex_buffer_field_abs = br.tell()
        _vertex_raw, vertex_ctx, vertex_abs = self._resolve_pointer_at_absolute(cache, terrain_ctx, vertex_buffer_field_abs)
        num_vmo_vertices = br.i32()
        cdc_render_data_id = br.u32()
        if _vertex_raw == 0 or vertex_abs <= 0:
            _vertex_raw = _xbox_vertex_raw
            vertex_ctx = xbox_vertex_ctx
            vertex_abs = xbox_vertex_abs

        if num_groups < 0 or num_groups > 8192:
            raise ValueError(f'Suspicious terrain group count {num_groups}')
        if num_vertices < 0 or num_vertices > 20_000_000:
            raise ValueError(f'Suspicious terrain vertex count {num_vertices}')
        if num_vmo_vertices < 0 or num_vmo_vertices > 20_000_000:
            raise ValueError(f'Suspicious terrain VMO vertex count {num_vmo_vertices}')

        logger.debug(
            'Parsed level terrain header from %s: groups=%d bg_instances=%d bg_objects=%d vertices=%d vmo_vertices=%d groups_abs=0x%X signal_tg_abs=0x%X terrain_signals_abs=0x%X bginst_abs=0x%X bg_abs=0x%X vb_abs=0x%X vmo_abs=0x%X',
            terrain_ctx.file_name,
            num_groups,
            num_bg_instances,
            num_bg_objects,
            num_vertices,
            num_vmo_vertices,
            groups_abs,
            signal_tg_abs if signal_tg_abs > 0 else signal_tg_raw,
            terrain_signals_abs if terrain_signals_abs > 0 else terrain_signals_raw,
            bginst_abs,
            bg_abs,
            vertex_abs,
            vmo_abs,
        )

        level = LevelData(filepath=self.filepath)
        level.terrain_signal_list_abs = int(terrain_signals_abs if terrain_signals_abs > 0 else terrain_signals_raw)
        level.terrain_signal_list_context = terrain_signals_ctx if int(level.terrain_signal_list_abs) > 0 else None
        level.cdc_render_data_id = int(cdc_render_data_id)
        resolved_stream_portals_abs = stream_portals_abs if stream_portals_abs > 0 else _stream_portals_raw
        if resolved_stream_portals_abs > 0:
            level.stream_unit_portals = _read_stream_unit_portals_from_reader(stream_portals_ctx.reader, resolved_stream_portals_abs, num_stream_unit_portals)
        level.vertices, level.uvs, level.vertex_colors = self._read_vertex_buffer(vertex_ctx, vertex_abs, num_vertices, is_vmo=False)
        level.vmo_vertices, level.vmo_uvs, level.vmo_vertex_colors = self._read_vertex_buffer(vmo_ctx, vmo_abs, num_vmo_vertices, is_vmo=True)

        for group_index in range(num_groups):
            group_abs = groups_abs + (group_index * 176)
            if not self._is_valid_abs(groups_ctx, group_abs, 176):
                logger.warning('Skipping terrain group %d at invalid offset 0x%X in %s', group_index, group_abs, groups_ctx.file_name)
                continue
            level.terrain_groups.append(self._read_terrain_group(cache, groups_ctx, group_index, group_abs))

        if self.import_signals:
            signal_group_abs = signal_tg_abs if signal_tg_abs > 0 else signal_tg_raw
            if signal_group_abs > 0:
                level.signal_mesh = self._read_signal_mesh_from_terrain_group(cache, signal_tg_ctx, signal_group_abs)

        level.bg_objects = self._read_bg_objects(cache, bg_ctx, bg_abs, num_bg_objects)
        level.bg_instances = self._read_bg_instances(cache, bginst_ctx, bginst_abs, num_bg_instances, bg_abs)
        return level


    def _read_bg_instances(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
        count: int,
        bg_object_list_abs: int,
    ) -> List[BGInstance]:
        instances: List[BGInstance] = []
        if absolute_offset <= 0 or count <= 0:
            return instances
        if count < 0 or count > 100000:
            logger.warning('Skipping BGInstances due to suspicious count %d', count)
            return instances

        stride = 0xF0
        for instance_index in range(count):
            entry_abs = absolute_offset + (instance_index * stride)
            if not self._is_valid_abs(context, entry_abs, stride):
                logger.warning('Skipping BGInstance %d at invalid offset 0x%X in %s', instance_index, entry_abs, context.file_name)
                continue
            br = context.reader
            br.seek(entry_abs)
            matrix_rows = (
                (br.f32(), br.f32(), br.f32(), br.f32()),
                (br.f32(), br.f32(), br.f32(), br.f32()),
                (br.f32(), br.f32(), br.f32(), br.f32()),
                (br.f32(), br.f32(), br.f32(), br.f32()),
            )
            bg_object_ptr_abs = entry_abs + 0xC0
            if not self._is_valid_abs(context, bg_object_ptr_abs, 4):
                continue
            bg_object_raw, _bgobj_ctx, bg_object_abs = self._resolve_pointer_at_absolute(cache, context, bg_object_ptr_abs)
            metadata_abs = entry_abs + 0xC4
            if not self._is_valid_abs(context, metadata_abs, 0x2C):
                continue
            br.seek(metadata_abs)
            flags = br.u32()
            multi_spline_offset = br.u32()
            original_radius = br.f32()
            radius = br.f32()
            instance_id = br.u16()
            bg_flags = br.u16()
            target_frame = br.i16()
            clip_beg = br.i16()
            clip_end = br.i16()
            link_seg = br.i16()
            link_instance_offset = br.u32()
            color_data_index = br.i16()
            lod = br.i16()
            _entity_offset = br.u32()
            active_light_bitfield = br.u32()
            if bg_object_abs >= bg_object_list_abs:
                bg_object_index = (bg_object_abs - bg_object_list_abs) // 0x60
            elif bg_object_raw >= bg_object_list_abs:
                bg_object_index = (bg_object_raw - bg_object_list_abs) // 0x60
            else:
                bg_object_index = -1
            instances.append(BGInstance(
                index=instance_index,
                bg_object_index=int(bg_object_index),
                bg_object_offset=int(bg_object_abs if bg_object_abs > 0 else bg_object_raw),
                instance_id=int(instance_id),
                flags=int(flags),
                multi_spline_offset=int(multi_spline_offset),
                multi_spline_data=self._read_multi_spline_from_context(cache, context, metadata_abs + 4),
                original_radius=float(original_radius),
                radius=float(radius),
                bg_flags=int(bg_flags),
                target_frame=int(target_frame),
                clip_beg=int(clip_beg),
                clip_end=int(clip_end),
                link_seg=int(link_seg),
                link_instance_offset=int(link_instance_offset),
                color_data_index=int(color_data_index),
                lod=int(lod),
                active_light_bitfield=int(active_light_bitfield),
                matrix_rows=matrix_rows,
            ))
        return instances


    def _score_bgobject_entry(self, cache: SectionContextCache, context: SectionContext, entry_abs: int) -> int:
        if not self._is_valid_abs(context, entry_abs, _BGOBJECT_MODEL_SIZE):
            return -1000
        br = context.reader
        br.seek(entry_abs)
        scale = (br.f32(), br.f32(), br.f32())
        if not _is_finite_triplet(scale):
            return -100
        vertex_count_abs = entry_abs + 72
        if not self._is_valid_abs(context, vertex_count_abs, 4):
            return -100
        br.seek(vertex_count_abs)
        vertex_count = br.u32()
        if vertex_count > 500000:
            return -50
        score = 1
        if vertex_count == 0:
            return score
        _raw_v, vctx, vabs = self._resolve_pointer_at_absolute(cache, context, entry_abs + 68)
        _raw_c, cctx, cabs = self._resolve_pointer_at_absolute(cache, context, entry_abs + 76)
        if self._is_valid_abs(vctx, vabs, vertex_count * 12):
            score += 2
        if self._is_valid_abs(cctx, cabs, vertex_count * 4):
            score += 1
        return score

    def _detect_bgobject_stride(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int, count: int) -> int:
        valid_count = max(1, min(int(count), 32))
        best_stride = _BGOBJECT_MODEL_SIZE
        best_score = -100000
        for stride in _BGOBJECT_STRIDE_CANDIDATES:
            score = 0
            for bg_index in range(valid_count):
                entry_abs = absolute_offset + (bg_index * stride)
                if not self._is_valid_abs(context, entry_abs, _BGOBJECT_MODEL_SIZE):
                    break
                score += self._score_bgobject_entry(cache, context, entry_abs)
            if score > best_score:
                best_score = score
                best_stride = stride
        return best_stride

    def _read_bg_objects(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
        count: int,
    ) -> List[BGObject]:
        bg_objects: List[BGObject] = []
        if absolute_offset <= 0 or count <= 0:
            return bg_objects
        if count < 0 or count > 100000:
            logger.warning('Skipping BGObjects due to suspicious count %d', count)
            return bg_objects

        stride = self._detect_bgobject_stride(cache, context, absolute_offset, count)
        for bg_index in range(count):
            entry_abs = absolute_offset + (bg_index * stride)
            if not self._is_valid_abs(context, entry_abs, _BGOBJECT_MODEL_SIZE):
                logger.warning('Skipping BGObject %d at invalid offset 0x%X in %s', bg_index, entry_abs, context.file_name)
                continue
            bg_object = self._read_bg_object(cache, context, entry_abs, bg_index, stride)
            if bg_object is not None:
                bg_objects.append(bg_object)
        return bg_objects

    def _read_bg_object_index_list_from_context(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
        vertex_count: int,
        label: str,
        *,
        vertex_ctx: Optional[SectionContext] = None,
        vertex_abs: int = 0,
    ) -> List[int]:
        if context is None or absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 4):
            return []
        br = context.reader
        previous = br.tell()
        try:
            br.seek(absolute_offset)
            count = br.i32()
            if count <= 0:
                return []
            if count > max(500000, int(vertex_count) * 64):
                logger.warning('Skipping BGObject %s in %s at 0x%X due to suspicious count %d', label, context.file_name, absolute_offset, count)
                return []

            max_vertex = max(0, int(vertex_count))
            data_start = br.tell()

            def _indices_from_patch_records(entry_size: int) -> List[int]:
                if vertex_ctx is None or vertex_abs <= 0:
                    return []
                if not self._is_valid_abs(context, data_start, count * entry_size):
                    return []
                values: List[int] = []
                for entry_index in range(count):
                    pointer_abs = data_start + (entry_index * entry_size)
                    _raw, uv_ctx, uv_abs = self._resolve_pointer_at_absolute(cache, context, pointer_abs)
                    if uv_abs <= 0:
                        continue
                    same_file = getattr(uv_ctx, 'filepath', None) == getattr(vertex_ctx, 'filepath', None)
                    if not same_file:
                        continue
                    delta = int(uv_abs) - (int(vertex_abs) + 8)
                    if delta < 0 or (delta % 12) != 0:
                        continue
                    vertex_index = delta // 12
                    if 0 <= vertex_index < max_vertex:
                        values.append(int(vertex_index))
                return sorted(set(values)) if len(values) == count else []

            patch_values = _indices_from_patch_records(16)
            if patch_values:
                return patch_values
            patch_values = _indices_from_patch_records(24)
            if patch_values:
                return patch_values

            if self._is_valid_abs(context, data_start, count * 4):
                try:
                    payload32 = br.peek(count * 4, offset=data_start)
                    values32 = [int(struct.unpack_from('<i', payload32, index * 4)[0]) for index in range(count)]
                except Exception:
                    values32 = []
                valid32 = [value for value in values32 if value >= 0 and (max_vertex <= 0 or value < max_vertex)]
                if len(valid32) == count:
                    return sorted(set(valid32))

            if not self._is_valid_abs(context, data_start, count * 2):
                logger.warning('Skipping BGObject %s in %s at 0x%X due to truncated list count=%d', label, context.file_name, absolute_offset, count)
                return []
            payload16 = br.peek(count * 2, offset=data_start)
            values16 = [int(struct.unpack_from('<H', payload16, index * 2)[0]) for index in range(count)]
            return sorted({value for value in values16 if value >= 0 and (max_vertex <= 0 or value < max_vertex)})
        except Exception:
            logger.warning('Failed to parse BGObject %s in %s at 0x%X', label, context.file_name, absolute_offset)
            return []
        finally:
            try:
                br.seek(previous)
            except Exception:
                pass

    def _read_bg_object(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
        bg_index: int,
        stride: int,
    ) -> Optional[BGObject]:
        br = context.reader
        br.seek(absolute_offset)
        scale_x = br.f32()
        scale_y = br.f32()
        scale_z = br.f32()

        vertex_count_abs = absolute_offset + 72
        if not self._is_valid_abs(context, vertex_count_abs, 4):
            return None
        br.seek(vertex_count_abs)
        vertex_count = br.u32()
        if vertex_count > 500000:
            logger.warning('Skipping BGObject %d due to suspicious vertex count %d', bg_index, vertex_count)
            return None

        _vertex_raw, vertex_ctx, vertex_abs = self._resolve_pointer_at_absolute(cache, context, absolute_offset + 68)
        _strip_raw, strip_ctx, strip_abs = self._resolve_pointer_at_absolute(cache, context, absolute_offset + 48)
        _color_raw, color_ctx, color_abs = self._resolve_pointer_at_absolute(cache, context, absolute_offset + 76)
        _env_raw, env_ctx, env_abs = self._resolve_pointer_at_absolute(cache, context, absolute_offset + 80)
        _eye_raw, eye_ctx, eye_abs = self._resolve_pointer_at_absolute(cache, context, absolute_offset + 84)

        position_candidates: List[Tuple[int, Tuple[float, float, float]]] = []
        for rel_offset in _BGOBJECT_POSITION_CANDIDATE_OFFSETS:
            if rel_offset == 80 and stride <= _BGOBJECT_MODEL_SIZE:
                continue
            if self._is_valid_abs(context, absolute_offset + rel_offset, 12):
                br.seek(absolute_offset + rel_offset)
                candidate_raw = (br.f32(), br.f32(), br.f32())
                if _is_finite_triplet(candidate_raw):
                    position_candidates.append((rel_offset, _convert_bgobject_translation(*candidate_raw)))

        flags = 0
        cdc_render_data_id = 0
        if self._is_valid_abs(context, absolute_offset + 32, 4):
            br.seek(absolute_offset + 32)
            flags = br.i32()
        if self._is_valid_abs(context, absolute_offset + 88, 4):
            br.seek(absolute_offset + 88)
            cdc_render_data_id = br.u32()

        bg_object = BGObject(
            index=bg_index,
            scale=(scale_x, scale_y, scale_z),
            position=_pick_bgobject_position((scale_x, scale_y, scale_z), position_candidates),
            stride=int(stride),
            flags=int(flags),
            cdc_render_data_id=int(cdc_render_data_id),
            position_candidates=position_candidates,
        )
        if vertex_count > 0 and self._is_valid_abs(vertex_ctx, vertex_abs, int(vertex_count) * 12):
            try:
                bg_object.raw_vertex_data = vertex_ctx.reader.peek(int(vertex_count) * 12, offset=vertex_abs)
            except Exception:
                bg_object.raw_vertex_data = b''
        color_byte_count = int(vertex_count) * 4
        if color_byte_count > 0 and color_abs > 0:
            try:
                if env_abs > color_abs and getattr(env_ctx, 'filepath', None) == getattr(color_ctx, 'filepath', None):
                    candidate_size = int(env_abs - color_abs)
                    if candidate_size >= color_byte_count:
                        color_byte_count = candidate_size - (candidate_size % 4)
                if color_byte_count > 0 and self._is_valid_abs(color_ctx, color_abs, color_byte_count):
                    bg_object.raw_color_data = color_ctx.reader.peek(color_byte_count, offset=color_abs)
                    per_slot = max(1, int(vertex_count) * 4)
                    bg_object.raw_color_slot_count = max(1, len(bg_object.raw_color_data) // per_slot)
            except Exception:
                bg_object.raw_color_data = b''
                bg_object.raw_color_slot_count = 1
        bg_object.vertices, bg_object.uvs = self._read_bg_object_vertices(vertex_ctx, vertex_abs, vertex_count, bg_object.scale)
        bg_object.vertex_colors = self._read_bg_object_colors(color_ctx, color_abs, vertex_count)
        bg_object.strips = self._read_bg_object_strip_chain(cache, strip_ctx, strip_abs)
        if not _bgobject_has_render_data(bg_object):
            return None
        bg_object.env_mapped_vertices = []
        bg_object.eye_ref_env_mapped_vertices = []
        _apply_bgobject_reflection_flags(bg_object)
        return bg_object

    def _read_bg_object_vertices(
        self,
        context: SectionContext,
        absolute_offset: int,
        count: int,
        scale: Tuple[float, float, float],
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
        vertices: List[Tuple[float, float, float]] = []
        uvs: List[Tuple[float, float]] = []
        if absolute_offset <= 0 or count <= 0:
            return vertices, uvs
        stride = 12
        if not self._is_valid_abs(context, absolute_offset, count * stride):
            logger.warning('Skipping BGObject vertices in %s at 0x%X count=%d because they exceed file bounds', context.file_name, absolute_offset, count)
            return vertices, uvs

        br = context.reader
        br.seek(absolute_offset)
        for _ in range(count):
            raw_x = br.i16()
            raw_y = br.i16()
            raw_z = br.i16()
            br.skip(2)
            u = br.i16()
            v = br.i16()
            vertices.append((float(raw_x), float(raw_y), float(raw_z)))
            uvs.append((float(u) * _UV_SCALE, float(v) * _UV_SCALE))
        return vertices, uvs

    def _read_bg_object_colors(
        self,
        context: SectionContext,
        absolute_offset: int,
        count: int,
    ) -> List[Tuple[int, int, int, int]]:
        colors: List[Tuple[int, int, int, int]] = []
        if absolute_offset <= 0 or count <= 0:
            return colors
        stride = 4
        if not self._is_valid_abs(context, absolute_offset, count * stride):
            logger.warning('Skipping BGObject colors in %s at 0x%X count=%d because they exceed file bounds', context.file_name, absolute_offset, count)
            return colors

        br = context.reader
        br.seek(absolute_offset)
        for _ in range(count):
            color_bgra = br.u32()
            b = color_bgra & 0xFF
            g = (color_bgra >> 8) & 0xFF
            r = (color_bgra >> 16) & 0xFF
            a = (color_bgra >> 24) & 0xFF
            colors.append((r, g, b, a))
        return colors

    def _read_bg_object_strip_chain(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
    ) -> List[LevelStrip]:
        strips: List[LevelStrip] = []
        if absolute_offset <= 0:
            return strips

        seen_strip_offsets: set[Tuple[str, int]] = set()
        current_ctx = context
        current_abs = absolute_offset
        material_index = 0
        while current_abs > 0:
            strip_key = (current_ctx.file_name, current_abs)
            if strip_key in seen_strip_offsets:
                logger.warning('Detected recursive BGObject strip list at %s:0x%X', current_ctx.file_name, current_abs)
                break
            seen_strip_offsets.add(strip_key)

            strip, next_ctx, next_abs = self._read_bg_object_strip_entry(cache, current_ctx, current_abs, material_index)
            if strip is not None:
                strips.append(strip)
                material_index += 1
            if next_abs <= 0:
                break
            current_ctx = next_ctx
            current_abs = next_abs
        return strips

    def _read_bg_object_strip_entry(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
        material_index: int,
    ) -> Tuple[Optional[LevelStrip], SectionContext, int]:
        if not self._is_valid_abs(context, absolute_offset, 28):
            logger.warning('Skipping invalid BGObject strip offset %s:0x%X', context.file_name, absolute_offset)
            return None, context, 0

        br = context.reader
        br.seek(absolute_offset)
        vertex_count = br.i32()
        if vertex_count > 2_000_000:
            logger.warning('Skipping BGObject strip at %s:0x%X due to suspicious vertex_count=%d', context.file_name, absolute_offset, vertex_count)
            return None, context, 0

        sort_vertex = (br.i16(), br.i16(), br.i16())
        br.u16()
        tpageid = br.u32()
        sort_push = br.i32()
        scroll_offset = br.f32()

        strip = LevelStrip(
            material_index=material_index,
            tpageid=tpageid,
            flags=0,
            vertex_base_offset=0,
            sort_vertex=sort_vertex,
            sort_push=int(sort_push),
            scroll_offset=float(scroll_offset),
            raw_count=int(vertex_count),
            is_terminator=vertex_count <= 0,
        )
        if vertex_count <= 0:
            return strip, context, 0

        next_raw, next_ctx, next_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        index_data_size = vertex_count * 2
        if not self._is_valid_abs(context, br.tell(), index_data_size):
            logger.warning('Skipping BGObject strip indices at %s:0x%X due to truncated data', context.file_name, absolute_offset)
            return None, context, 0

        for _ in range(vertex_count):
            strip.indices.append(br.u16())
        return strip, next_ctx, (next_abs if next_raw != 0 else 0)

    def _read_vertex_buffer(self, context: SectionContext, absolute_offset: int, count: int, is_vmo: bool):
        vertices: List[Tuple[float, float, float]] = []
        uvs: List[Tuple[float, float]] = []
        vertex_colors: List[Tuple[int, int, int, int]] = []
        if absolute_offset <= 0 or count <= 0:
            return vertices, uvs, vertex_colors

        stride = 32 if is_vmo else 16
        if not self._is_valid_abs(context, absolute_offset, count * stride):
            logger.warning(
                'Skipping %s vertex buffer in %s at 0x%X count=%d because it exceeds file bounds',
                'VMO' if is_vmo else 'terrain',
                context.file_name,
                absolute_offset,
                count,
            )
            return vertices, uvs, vertex_colors

        br = context.reader
        br.seek(absolute_offset)
        for _ in range(count):
            x = br.i16()
            y = br.i16()
            z = br.i16()
            br.skip(2)
            color_bgra = br.u32()
            b = color_bgra & 0xFF
            g = (color_bgra >> 8) & 0xFF
            r = (color_bgra >> 16) & 0xFF
            a = (color_bgra >> 24) & 0xFF
            u = br.i16()
            v = br.i16()
            br.skip(20 if is_vmo else 4)
            vertices.append(_convert_level_position(x, y, z))
            uvs.append((float(u) * _UV_SCALE, float(v) * _UV_SCALE))
            vertex_colors.append((r, g, b, a))
        return vertices, uvs, vertex_colors

    def _read_signal_mesh_from_terrain_group(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int) -> Optional[TerrainSignalMesh]:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 60):
            return None
        _mesh_raw, mesh_ctx, mesh_abs = self._resolve_pointer_at_absolute(cache, context, absolute_offset + 56)
        return self._read_signal_mesh(cache, mesh_ctx, mesh_abs if mesh_abs > 0 else _mesh_raw)

    def _read_signal_mesh(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int) -> Optional[TerrainSignalMesh]:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 72):
            return None
        br = context.reader
        br.seek(absolute_offset)
        bmin = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        bmax = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        position = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        _vertex_raw, vertex_ctx, vertex_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        _face_raw, face_ctx, face_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        br.skip(8)
        vertex_type = br.u16()
        br.skip(2)
        num_faces = br.u16()
        num_vertices = br.u16()
        if num_vertices > 2_000_000 or num_faces > 2_000_000:
            logger.warning('Skipping signal mesh at %s:0x%X due to suspicious counts vertices=%d faces=%d', context.file_name, absolute_offset, num_vertices, num_faces)
            return None
        signal_mesh = TerrainSignalMesh(
            position=_convert_level_position(*position),
            bbox_min=_convert_level_position(*bmin),
            bbox_max=_convert_level_position(*bmax),
            raw_bbox_min=tuple(float(v) for v in bmin),
            raw_bbox_max=tuple(float(v) for v in bmax),
            vertex_type=int(vertex_type),
        )
        vertex_stride = 6 if vertex_type == 0 else 16
        if vertex_abs > 0 and num_vertices > 0 and self._is_valid_abs(vertex_ctx, vertex_abs, num_vertices * vertex_stride):
            vertex_reader = vertex_ctx.reader
            vertex_reader.seek(vertex_abs)
            for _ in range(num_vertices):
                if vertex_type == 0:
                    signal_mesh.vertices.append(_convert_level_position(vertex_reader.i16(), vertex_reader.i16(), vertex_reader.i16()))
                else:
                    signal_mesh.vertices.append(_convert_level_position(vertex_reader.f32(), vertex_reader.f32(), vertex_reader.f32()))
                    vertex_reader.skip(4)
        if face_abs > 0 and num_faces > 0 and self._is_valid_abs(face_ctx, face_abs, num_faces * 10):
            face_reader = face_ctx.reader
            face_reader.seek(face_abs)
            for _ in range(num_faces):
                signal_mesh.faces.append(TerrainCollisionFace(
                    i0=face_reader.u16(),
                    i1=face_reader.u16(),
                    i2=face_reader.u16(),
                    adjacency_flags=face_reader.u8(),
                    collision_flags=face_reader.u8(),
                    signal_id=face_reader.u16(),
                ))
        if not signal_mesh.vertices or not signal_mesh.faces:
            return None
        return signal_mesh

    def _read_collision_kd_nodes(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int, node_count: int) -> List[TerrainCollisionKDNode]:
        if absolute_offset <= 0 or node_count <= 0 or not self._is_valid_abs(context, absolute_offset, node_count * 12):
            return []

        br = context.reader
        br.seek(absolute_offset)
        nodes: List[TerrainCollisionKDNode] = []
        for node_index in range(node_count):
            nodes.append(
                TerrainCollisionKDNode(
                    node_index=node_index,
                    neg_offset=br.f32(),
                    pos_offset=br.f32(),
                    ref_index=br.u16(),
                    axis=br.u8(),
                    num_faces=br.u8(),
                )
            )
        return nodes

    def _populate_collision_kd_bounds(self, collision: TerrainGroupCollision) -> None:
        _populate_collision_kd_bounds_impl(collision)

    def _read_terrain_collision(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int) -> Optional[TerrainGroupCollision]:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 80):
            return None

        br = context.reader
        br.seek(absolute_offset)
        bmin = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        bmax = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        position = (br.f32(), br.f32(), br.f32())
        br.skip(4)

        _vertex_raw, vertex_ctx, vertex_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        _face_raw, face_ctx, face_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        _root_raw, root_ctx, root_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        br.u32()  # owner/group pointer
        br.u16()
        node_count = br.u16()
        num_faces = br.u16()
        num_vertices = br.u16()
        kd_max_depth = br.u16()
        br.skip(4)
        br.skip(2)

        collision = TerrainGroupCollision(
            position=_convert_level_position(*position),
            bbox_min=_convert_level_position(*bmin),
            bbox_max=_convert_level_position(*bmax),
            raw_bbox_min=tuple(float(v) for v in bmin),
            raw_bbox_max=tuple(float(v) for v in bmax),
            kd_max_depth=int(kd_max_depth),
        )

        if vertex_abs > 0 and num_vertices > 0 and self._is_valid_abs(vertex_ctx, vertex_abs, num_vertices * 6):
            vertex_reader = vertex_ctx.reader
            vertex_reader.seek(vertex_abs)
            for _ in range(num_vertices):
                collision.vertices.append(_convert_level_position(vertex_reader.i16(), vertex_reader.i16(), vertex_reader.i16()))

        if face_abs > 0 and num_faces > 0 and self._is_valid_abs(face_ctx, face_abs, num_faces * 10):
            face_reader = face_ctx.reader
            face_reader.seek(face_abs)
            for _ in range(num_faces):
                collision.faces.append(TerrainCollisionFace(
                    i0=face_reader.u16(),
                    i1=face_reader.u16(),
                    i2=face_reader.u16(),
                    adjacency_flags=face_reader.u8(),
                    collision_flags=face_reader.u8(),
                    client_flags=face_reader.u8(),
                    material_type=face_reader.u8(),
                ))

        if root_abs > 0 and node_count > 0:
            collision.kd_nodes = self._read_collision_kd_nodes(cache, root_ctx, root_abs, node_count)
            if collision.kd_nodes:
                self._populate_collision_kd_bounds(collision)
                if collision.kd_max_depth <= 0:
                    collision.kd_max_depth = max((int(node.depth) for node in collision.kd_nodes), default=0)

        if not collision.vertices or not collision.faces:
            return None
        return collision

    def _read_terrain_group(self, cache: SectionContextCache, context: SectionContext, index: int, absolute_offset: int) -> TerrainGroup:
        br = context.reader
        br.seek(absolute_offset)

        global_offset = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        local_offset = (br.f32(), br.f32(), br.f32())
        br.skip(4)
        flags = br.i32()
        terrain_id = br.i32()
        unique_id = br.i32()
        spline_id = br.i32()
        br.skip(8)  # instanceSpline, level
        _mesh_raw, mesh_ctx, mesh_abs = self._resolve_pointer_at_absolute(cache, context, br.tell())
        texture_morph_value = br.f32()
        texture_morph_step = br.f32()

        octree_field_abs = br.tell()
        _octree_raw, octree_ctx, octree_abs = self._resolve_pointer_at_absolute(cache, context, octree_field_abs)

        octree_scroll_field_abs = br.tell()
        _scroll_raw, scroll_ctx, scroll_abs = self._resolve_pointer_at_absolute(cache, context, octree_scroll_field_abs)
        br.skip(4)  # octreeAnimatedInfo
        br.skip(64)  # matrix
        material_list_field_abs = br.tell()
        _mat_raw, material_ctx, material_abs = self._resolve_pointer_at_absolute(cache, context, material_list_field_abs)
        sorted_materials_field_abs = br.tell()
        _sort_raw, sorted_ctx, sorted_abs = self._resolve_pointer_at_absolute(cache, context, sorted_materials_field_abs)

        br.skip(8)  # cdcRenderDetailID, cdcRenderTerrainData
        group_origin = (br.f32(), br.f32(), br.f32())
        position = _convert_level_position(*global_offset)

        group = TerrainGroup(
            index=index,
            position=position,
            global_offset=_convert_level_position(*global_offset),
            local_offset=_convert_level_position(*local_offset),
            flags=flags,
            terrain_id=terrain_id,
            unique_id=unique_id,
            spline_id=spline_id,
            texture_morph_value=texture_morph_value,
            texture_morph_step=texture_morph_step,
            group_origin=_convert_level_position(*group_origin),
            collision=self._read_terrain_collision(cache, mesh_ctx, mesh_abs),
        )
        self._read_group_materials(material_ctx, material_abs, group)
        if sorted_abs > 0:
            self._read_group_sorted_materials(sorted_ctx, sorted_abs, group)
        if scroll_abs > 0:
            self._read_group_scroll_info(scroll_ctx, scroll_abs, group)
        if octree_abs > 0:
            self._read_octree(cache, octree_ctx, octree_abs, group, set())
        return group

    def _read_group_materials(self, context: SectionContext, absolute_offset: int, group: TerrainGroup) -> None:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 4):
            return

        br = context.reader
        br.seek(absolute_offset)
        num_materials = br.i32()
        if num_materials < 0 or num_materials > 100000:
            logger.warning('Terrain group %d has suspicious material count %d', group.index, num_materials)
            return

        for material_index in range(num_materials):
            entry_abs = absolute_offset + 4 + (material_index * 20)
            if not self._is_valid_abs(context, entry_abs, 20):
                logger.warning('Terrain group %d material %d entry out of range at 0x%X in %s', group.index, material_index, entry_abs, context.file_name)
                break
            br.seek(entry_abs)
            tpageid = br.u32()
            flags = br.u32()
            vertex_base_offset = br.u32()
            group.material_entry_offsets.append(int(entry_abs))
            group.strips.append(
                LevelStrip(
                    material_index=material_index,
                    tpageid=tpageid,
                    flags=flags,
                    vertex_base_offset=vertex_base_offset,
                )
            )
            br.skip(8)

    def _read_group_sorted_materials(self, context: SectionContext, absolute_offset: int, group: TerrainGroup) -> None:
        if absolute_offset <= 0 or not group.strips:
            return
        entry_count = len(group.strips)
        if not self._is_valid_abs(context, absolute_offset, entry_count * 2):
            return

        br = context.reader
        br.seek(absolute_offset)
        group.sorted_material_indices = [br.u16() for _ in range(entry_count)]

    def _read_group_scroll_info(self, context: SectionContext, absolute_offset: int, group: TerrainGroup) -> None:
        if absolute_offset <= 0 or not self._is_valid_abs(context, absolute_offset, 4):
            return

        br = context.reader
        br.seek(absolute_offset)
        count = br.i32()
        if count <= 0 or count > 100000:
            logger.debug('Terrain group %d has suspicious scroll info count %d at 0x%X in %s', group.index, count, absolute_offset, context.file_name)
            return

        unresolved = 0
        for entry_index in range(count):
            entry_abs = absolute_offset + 4 + (entry_index * 20)
            if not self._is_valid_abs(context, entry_abs, 20):
                break
            br.seek(entry_abs)
            br.u32()  # next
            scroll_speed = br.f32()
            scroll_num_tiles = br.i32()
            scroll_tile = br.i32()
            fixup_value_offset = br.u32()

            material_index = self._resolve_scroll_material_index(context, fixup_value_offset, group)
            if material_index is None:
                unresolved += 1
                continue

            strip = group.strips[material_index]
            strip.scroll_entry_count += 1
            if not strip.scroll_speeds or all(abs(float(existing) - float(scroll_speed)) > 1e-6 for existing in strip.scroll_speeds):
                strip.scroll_speeds.append(float(scroll_speed))
            if strip.scroll_num_tiles == 0:
                strip.scroll_num_tiles = int(scroll_num_tiles)
            if strip.scroll_tile == 0:
                strip.scroll_tile = int(scroll_tile)

        if unresolved:
            logger.debug('Terrain group %d unresolved scroll entries: %d/%d', group.index, unresolved, count)

    def _resolve_scroll_material_index(self, context: SectionContext, fixup_value_offset: int, group: TerrainGroup) -> Optional[int]:
        if fixup_value_offset <= 0 or not group.strips:
            return None

        preferred_deltas = (12, 8, 0, 4, 16, 20, -4)
        base_offsets: List[int] = []
        local_base = context.data_start + int(fixup_value_offset)
        if self._is_valid_abs(context, local_base, 4):
            base_offsets.append(local_base)
        if self._is_valid_abs(context, int(fixup_value_offset), 4):
            base_offsets.append(int(fixup_value_offset))

        br = context.reader
        candidates: List[Tuple[int, int]] = []
        for base in base_offsets:
            for delta_rank, delta in enumerate(preferred_deltas):
                probe_abs = int(base) + int(delta)
                if not self._is_valid_abs(context, probe_abs, 4):
                    continue
                br.seek(probe_abs)
                value = br.i32()
                if 0 <= value < len(group.strips):
                    score = 100 - (delta_rank * 10)
                    if delta == 12:
                        score += 25
                    candidates.append((score, int(value)))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]


    def _read_octree(
        self,
        cache: SectionContextCache,
        context: SectionContext,
        absolute_offset: int,
        group: TerrainGroup,
        visited: set[Tuple[str, int]],
    ) -> None:
        if absolute_offset <= 0:
            return

        visit_key = (context.file_name, absolute_offset)
        if visit_key in visited:
            return
        visited.add(visit_key)

        header_offset = absolute_offset + 16
        if not self._is_valid_abs(context, header_offset, 8):
            logger.warning('Skipping octree at invalid offset 0x%X in %s', absolute_offset, context.file_name)
            return

        br = context.reader
        br.seek(header_offset)
        strip_field_abs = br.tell()
        strip_raw, strip_ctx, strip_abs = self._resolve_pointer_at_absolute(cache, context, strip_field_abs)
        num_spheres = br.i32()
        if num_spheres < 0 or num_spheres > 8:
            logger.warning('Octree at 0x%X in %s has suspicious sphere count %d', absolute_offset, context.file_name, num_spheres)
            return

        for sphere_index in range(num_spheres):
            sphere_field_abs = br.tell()
            sphere_raw, sphere_ctx, sphere_abs = self._resolve_pointer_at_absolute(cache, context, sphere_field_abs)
            if sphere_raw != 0 and sphere_abs > 0:
                self._read_octree(cache, sphere_ctx, sphere_abs, group, visited)
            br.seek(header_offset + 8 + ((sphere_index + 1) * 4))

        current_ctx = strip_ctx
        current_abs = strip_abs
        current_raw = strip_raw
        seen_strip_offsets: set[Tuple[str, int]] = set()
        chain_guard = 0
        while current_raw != 0 and current_abs > 0:
            if not self._is_valid_abs(current_ctx, current_abs, 4):
                logger.debug('Stopping strip chain at %s:0x%X because the next pointer resolves to EOF/truncated data', current_ctx.file_name, current_abs)
                break
            strip_key = (current_ctx.file_name, current_abs)
            if strip_key in seen_strip_offsets:
                logger.warning('Detected recursive strip list at %s:0x%X', current_ctx.file_name, current_abs)
                break
            seen_strip_offsets.add(strip_key)
            chain_guard += 1
            if chain_guard > 100000:
                logger.warning('Aborting strip chain due to excessive iterations')
                break
            next_raw, next_ctx, next_abs = self._read_strip_chain_entry(cache, current_ctx, current_abs, group)
            current_raw = next_raw
            current_ctx = next_ctx
            current_abs = next_abs

    def _read_strip_chain_entry(self, cache: SectionContextCache, context: SectionContext, absolute_offset: int, group: TerrainGroup):
        if not self._is_valid_abs(context, absolute_offset, 4):
            logger.warning('Skipping invalid strip offset %s:0x%X', context.file_name, absolute_offset)
            return 0, context, 0

        br = context.reader
        br.seek(absolute_offset)
        vertex_count = br.i32()
        if vertex_count == 0:
            return 0, context, 0
        if vertex_count < 0 or vertex_count > 2_000_000:
            logger.warning('Stopping strip chain at %s:0x%X due to suspicious vertex_count=%d', context.file_name, absolute_offset, vertex_count)
            return 0, context, 0
        if not self._is_valid_abs(context, br.tell(), 36):
            logger.warning('Skipping truncated strip header at %s:0x%X', context.file_name, absolute_offset)
            return 0, context, 0

        br.skip(16)
        material_field_abs = br.tell()
        material_raw, material_ctx, material_abs = self._resolve_pointer_at_absolute(cache, context, material_field_abs)
        mat_index = _resolve_group_strip_material_index(group, material_raw, material_abs if material_raw != 0 else None)
        br.skip(16)
        next_field_abs = br.tell()
        next_raw, next_ctx, next_abs = self._resolve_pointer_at_absolute(cache, context, next_field_abs)

        index_data_size = vertex_count * 2
        if not self._is_valid_abs(context, br.tell(), index_data_size):
            logger.warning('Skipping strip indices at %s:0x%X due to truncated data', context.file_name, absolute_offset)
            return 0, context, 0

        if mat_index is None:
            logger.warning(
                'Skipping strip indices for group %d at %s:0x%X because material_ref=0x%08X could not be resolved (%d materials)',
                group.index,
                context.file_name,
                absolute_offset,
                int(material_raw) & 0xFFFFFFFF,
                len(group.strips),
            )
            br.skip(index_data_size)
        else:
            target_strip = group.strips[mat_index]
            for _ in range(vertex_count):
                target_strip.indices.append(br.i16())

        return next_raw, next_ctx, next_abs


class TRLevelBuilder:
    def __init__(
        self,
        context,
        filepath: str,
        collection=None,
        *,
        import_textures: bool = True,
        import_bgobjects: bool = True,
        import_collisions: bool = True,
        import_signals: bool = True,
        import_kdnodes: bool = False,
        split_terrain_groups_by_strip: bool = False,
    ):
        self.context = context
        self.filepath = filepath
        self.collection = collection or context.collection
        self.import_textures = import_textures
        self.import_bgobjects = import_bgobjects
        self.import_collisions = import_collisions
        self.import_signals = import_signals
        self.import_kdnodes = import_kdnodes
        self.split_terrain_groups_by_strip = bool(split_terrain_groups_by_strip)
        self.bgobject_collection = None
        self._material_cache: dict[Tuple[int, int], bpy.types.Material] = {}
        self.drm_name = Path(filepath).stem or (collection.name if collection is not None else 'Level')
        self.level_root = None
        self._runtime_texture_images: dict[int, bpy.types.Image] = {}
        self._collision_material_cache: dict[int, bpy.types.Material] = {}
        self._signal_material: Optional[bpy.types.Material] = None
        self._mesh_builder = MeshBuilder(
            context=context,
            filepath=filepath,
            settings=MeshBuildSettings(apply_segment_pivots=False, import_textures=import_textures),
            collection=self.collection,
        )


    def _prefixed_name(self, name: str) -> str:
        prefix = str(getattr(self, 'drm_name', '') or '').strip()
        if not prefix:
            return name
        normalized = f'{prefix}_'
        return name if name.startswith(normalized) else f'{prefix}_{name}'

    def _ensure_level_root(self) -> bpy.types.Object:
        root_name = self._prefixed_name('Level')
        root = bpy.data.objects.get(root_name)
        if root is None:
            root = bpy.data.objects.new(root_name, None)
            self.collection.objects.link(root)
        elif all(coll != self.collection for coll in getattr(root, 'users_collection', ())):
            try:
                self.collection.objects.link(root)
            except Exception:
                pass
        root.empty_display_type = 'PLAIN_AXES'
        root.empty_display_size = 96.0
        root.parent = None
        scene_center = getattr(self, 'scene_center_offset', (0.0, 0.0, 0.0, 0.0))
        try:
            scene_center_tuple = tuple(float(v) for v in scene_center)
        except Exception:
            scene_center_tuple = (0.0, 0.0, 0.0, 0.0)
        if len(scene_center_tuple) < 4:
            scene_center_tuple = tuple(scene_center_tuple) + ((0.0,) * (4 - len(scene_center_tuple)))
        root.location = (scene_center_tuple[0], scene_center_tuple[1], scene_center_tuple[2])
        root.rotation_euler = (0.0, 0.0, 0.0)
        root.scale = (1.0, 1.0, 1.0)
        root['trlau_level'] = True
        root['trlau_level_root'] = True
        root['trlau_drm_name'] = str(getattr(self, 'drm_name', ''))
        for stale_key in ('trlau_scene_center_offset', 'trlau_scene_center_offset_w'):
            if stale_key in root:
                try:
                    del root[stale_key]
                except Exception:
                    pass
        self.level_root = root
        return root

    def _get_or_create_component_empty(self, component_name: str, *, display_size: float = 56.0) -> bpy.types.Object:
        empty_name = self._prefixed_name(component_name)
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
            self.collection.objects.link(empty)
        elif all(coll != self.collection for coll in getattr(empty, 'users_collection', ())):
            try:
                self.collection.objects.link(empty)
            except Exception:
                pass
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = float(display_size)
        empty.parent = self._ensure_level_root()
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_mode = 'XYZ'
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        empty['trlau_level'] = True
        empty['trlau_component_empty'] = True
        empty['trlau_component_name'] = component_name
        return empty

    def _get_or_create_terrain_empty(self) -> bpy.types.Object:
        empty = self._get_or_create_component_empty('Terrain', display_size=72.0)
        empty['trlau_type'] = 'Terrain'
        return empty

    def build(self, level: LevelData) -> List[bpy.types.Object]:
        self.scene_center_offset = tuple(float(v) for v in getattr(level, 'scene_center_offset', (0.0, 0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0, 0.0))
        if len(self.scene_center_offset) < 4:
            self.scene_center_offset = tuple(self.scene_center_offset) + ((0.0,) * (4 - len(self.scene_center_offset)))
        self._runtime_texture_images = {}
        for texture_id, image_name in getattr(level, 'texture_images', {}).items():
            image = bpy.data.images.get(image_name)
            if image is not None:
                self._runtime_texture_images[int(texture_id)] = image
        built_objects: List[bpy.types.Object] = []
        level_root = self._ensure_level_root()
        built_objects.append(level_root)
        for group in level.terrain_groups:
            group_empty = self._get_or_create_group_empty(group)
            if group_empty is not None:
                built_objects.append(group_empty)
            if self.import_collisions:
                collision_parent = group_empty
                if collision_parent is not None and getattr(group, 'collision', None) is not None:
                    collision_obj = self._build_terrain_group_collision_object(group, parent=collision_parent)
                    if collision_obj is not None:
                        built_objects.append(collision_obj)
                    if self.import_kdnodes:
                        for kd_obj in self._build_collision_kd_objects(group, parent=collision_parent):
                            if kd_obj is not None:
                                built_objects.append(kd_obj)
            if self.split_terrain_groups_by_strip:
                for strip in group.strips:
                    obj = self._build_strip_object(level, group, strip, parent=group_empty)
                    if obj is not None:
                        built_objects.append(obj)
            else:
                obj = self._build_terrain_group_mesh_object(level, group, parent=group_empty)
                if obj is not None:
                    built_objects.append(obj)
        if self.import_signals:
            signal_parent = self._get_or_create_signals_empty() if (getattr(level, 'signals', None) or getattr(level, 'signal_mesh', None) is not None) else None
            if getattr(level, 'signals', None):
                for signal in getattr(level, 'signals', []) or []:
                    signal_obj = self._build_signal_object(signal, parent=signal_parent)
                    if signal_obj is not None:
                        built_objects.append(signal_obj)
            elif getattr(level, 'signal_mesh', None) is not None:
                signal_obj = self._build_signal_mesh_object(getattr(level, 'signal_mesh'), parent=signal_parent)
                if signal_obj is not None:
                    built_objects.append(signal_obj)
        if self.import_bgobjects:
            self.bgobject_collection = self.collection
            if getattr(level, 'bg_objects', None):
                print('BGObjects found, importing...')
                self._get_or_create_component_empty('BGInstance')
            bg_instances = list(getattr(level, 'bg_instances', []))
            if bg_instances:
                instances_by_object: dict[int, List[BGInstance]] = {}
                for bg_instance in bg_instances:
                    instances_by_object.setdefault(int(bg_instance.bg_object_index), []).append(bg_instance)
                for bg_object in getattr(level, 'bg_objects', []):
                    if not _bgobject_has_render_data(bg_object):
                        continue

                    object_instances = instances_by_object.get(int(bg_object.index), [])
                    if object_instances:
                        for bg_instance in object_instances:
                            obj = self._build_bgobject_mesh_object(bg_object, bg_instance)
                            if obj is not None:
                                built_objects.append(obj)
                    else:
                        obj = self._build_bgobject_mesh_object(bg_object, None)
                        if obj is not None:
                            built_objects.append(obj)
            else:
                for bg_object in getattr(level, 'bg_objects', []):
                    if not _bgobject_has_render_data(bg_object):
                        continue
                    obj = self._build_bgobject_mesh_object(bg_object, None)
                    if obj is not None:
                        built_objects.append(obj)
        return built_objects

    def _get_or_create_group_empty(self, group: TerrainGroup) -> bpy.types.Object:
        empty_name = self._prefixed_name(f'TerrainGroup_{int(group.index):03d}')
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
            self.collection.objects.link(empty)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 64.0
        empty.location = tuple(float(v) for v in getattr(group, 'position', (0.0, 0.0, 0.0)))
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        empty.parent = self._get_or_create_terrain_empty()
        empty['trlau_terrain_group_flags'] = int(getattr(group, 'flags', 0))
        empty['trlau_terrain_group_id'] = int(getattr(group, 'terrain_id', 0))
        empty['trlau_terrain_group_unique_id'] = int(getattr(group, 'unique_id', 0))
        empty['trlau_terrain_group_spline_id'] = int(getattr(group, 'spline_id', 0))
        texture_morph_value = float(getattr(group, 'texture_morph_value', 0.0))
        texture_morph_step = float(getattr(group, 'texture_morph_step', 0.0))
        if abs(texture_morph_value) > 1e-9:
            empty['trlau_terrain_group_texture_morph_value'] = texture_morph_value
        if abs(texture_morph_step) > 1e-9:
            empty['trlau_terrain_group_texture_morph_step'] = texture_morph_step
        if 'trlau_terrain_group_sorted_material_indices' in empty:
            del empty['trlau_terrain_group_sorted_material_indices']
        return empty


    def _get_or_create_signals_empty(self) -> bpy.types.Object:
        empty = self._get_or_create_component_empty('Signals', display_size=56.0)
        empty.parent = self._get_or_create_terrain_empty()
        empty['trlau_type'] = 'Signals'
        empty['trlau_terrain_signals'] = True
        return empty

    def _get_or_create_signal_material(self) -> bpy.types.Material:
        if self._signal_material is not None and getattr(self._signal_material, 'name', None) in bpy.data.materials:
            return self._signal_material
        material_name = self._prefixed_name('Signal')
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(material_name)
        material.diffuse_color = (0.0, 0.75, 1.0, 0.18)
        material.use_nodes = True
        material.blend_method = 'BLEND'
        if hasattr(material, 'shadow_method'):
            material.shadow_method = 'NONE'
        principled = None
        if material.node_tree is not None:
            principled = material.node_tree.nodes.get('Principled BSDF')
        if principled is not None:
            principled.inputs['Base Color'].default_value = (0.0, 0.75, 1.0, 1.0)
            principled.inputs['Alpha'].default_value = 0.18
        material['trlau_signal_material'] = True
        self._signal_material = material
        return material

    def _apply_signal_face_attributes(self, mesh: bpy.types.Mesh, faces: List[TerrainCollisionFace]) -> None:
        if len(mesh.polygons) != len(faces):
            return
        attributes = getattr(mesh, 'attributes', None)
        if attributes is None:
            return
        values_by_name = {
            'trlau_signal_adjacency_flags': [int(face.adjacency_flags) for face in faces],
            'trlau_signal_collision_flags': [int(face.collision_flags) for face in faces],
        }
        for name, values in values_by_name.items():
            attr = attributes.get(name)
            if attr is None:
                try:
                    attr = attributes.new(name=name, type='INT', domain='FACE')
                except Exception as exc:
                    logger.warning('Failed creating signal FACE attribute %s: %s', name, exc)
                    continue
            try:
                attr.data.foreach_set('value', values)
            except Exception as exc:
                logger.warning('Failed writing signal FACE attribute %s: %s', name, exc)

    def _build_signal_mesh_object(self, signal_mesh: Optional[TerrainSignalMesh], parent: Optional[bpy.types.Object] = None, *, object_name: Optional[str] = None) -> Optional[bpy.types.Object]:
        if signal_mesh is None or not signal_mesh.vertices or not signal_mesh.faces:
            return None
        vertices = [(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in signal_mesh.vertices]
        faces: List[Tuple[int, int, int]] = []
        source_faces: List[TerrainCollisionFace] = []
        for face in signal_mesh.faces:
            tri = (int(face.i0), int(face.i1), int(face.i2))
            if min(tri) < 0 or max(tri) >= len(vertices) or len(set(tri)) != 3:
                continue
            faces.append(tri)
            source_faces.append(face)
        if not faces:
            return None
        mesh_name = self._prefixed_name(object_name or 'Signal_Mesh')
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        mesh.materials.append(self._get_or_create_signal_material())
        self._apply_signal_face_attributes(mesh, source_faces)
        obj = bpy.data.objects.new(mesh_name, mesh)
        self.collection.objects.link(obj)
        if parent is not None:
            obj.parent = parent
            try:
                obj.matrix_parent_inverse.identity()
            except Exception:
                pass
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)
        obj.display_type = 'SOLID'
        obj['trlau_level'] = True
        obj['trlau_type'] = 'SignalMesh'
        obj['trlau_signal_mesh'] = True
        return obj

    @staticmethod
    def _clear_legacy_signal_idprops(obj: bpy.types.Object) -> None:
        for key in list(getattr(obj, 'keys', lambda: [])()):
            key_text = str(key)
            if key_text.startswith('trlau_signal_') and key_text not in {'trlau_signal_mesh'}:
                try:
                    del obj[key]
                except Exception:
                    pass

    @staticmethod
    def _assign_signal_property(target, field_name: str, value) -> None:
        if not hasattr(target, field_name):
            return
        try:
            setattr(target, field_name, value)
        except Exception:
            try:
                setattr(target, field_name, int(value))
            except Exception:
                try:
                    setattr(target, field_name, str(value))
                except Exception:
                    pass

    def _assign_signal_spline_pointer(self, target, slot: int, values: Optional[dict[str, int]]) -> None:
        if target is None:
            return
        self._assign_signal_property(target, 'slot', int(slot))
        present = isinstance(values, dict) and bool(values)
        self._assign_signal_property(target, 'present', bool(present))
        data_target = getattr(target, 'data', None)
        if data_target is None or not present:
            return
        for field_name, value in values.items():
            self._assign_signal_property(data_target, str(field_name), value)

    def _assign_signal_dict_collection(self, collection, values: object) -> None:
        if collection is None:
            return
        try:
            collection.clear()
        except Exception:
            return
        if not isinstance(values, list):
            return
        for value in values:
            if not isinstance(value, dict):
                continue
            item = collection.add()
            for field_name, field_value in value.items():
                if isinstance(field_value, (list, dict)):
                    continue
                self._assign_signal_property(item, str(field_name), field_value)

    def _assign_signal_attack_wave_collection(self, collection, values: object) -> None:
        if collection is None:
            return
        try:
            collection.clear()
        except Exception:
            return
        if not isinstance(values, list):
            return
        nested_collections = {
            'limits',
            'boxLimits',
            'planeLimits',
            'runAndGunPositions',
            'patrolPositions',
            'messages',
        }
        for value in values:
            if not isinstance(value, dict):
                continue
            item = collection.add()
            for field_name, field_value in value.items():
                field_name = str(field_name)
                if field_name in nested_collections:
                    self._assign_signal_dict_collection(getattr(item, field_name, None), field_value)
                elif not isinstance(field_value, (list, dict)):
                    self._assign_signal_property(item, field_name, field_value)

    def _apply_signal_properties_to_object(self, obj: bpy.types.Object, signal: TerrainSignal) -> None:
        self._clear_legacy_signal_idprops(obj)
        obj['trlau_type'] = 'Signal'
        obj['trlau_signal'] = True
        obj['trlau_signal_mesh'] = True

        signal_data = getattr(obj, 'trlau_signal_data', None)
        if signal_data is None:
            # Fallback for unusual registration/order cases.  Keep this minimal rather
            # than writing the full Signal struct as unstructured ID properties.
            obj['trlau_signal_id'] = int(getattr(signal, 'signal_id', -1))
            obj['trlau_signal_list_index'] = int(getattr(signal, 'index', -1))
            obj['trlau_signal_face_id'] = int(getattr(signal, 'face_id', -1))
            return

        signal_data.signal_id = int(getattr(signal, 'signal_id', -1))
        signal_data.list_index = int(getattr(signal, 'index', -1))
        signal_data.face_id = int(getattr(signal, 'face_id', -1))
        spline_data = getattr(signal_data, 'splineCamParams', None)
        for slot in range(3):
            self._assign_signal_spline_pointer(getattr(signal_data, f'splineCamDataPtr{slot}', None), slot, None)
        for collection_name in ('enableAttackWaves', 'disableAttackWaves', 'killAttackWaves'):
            collection = getattr(signal_data, collection_name, None)
            try:
                if collection is not None:
                    collection.clear()
            except Exception:
                pass

        for field_name, value in (getattr(signal, 'properties', {}) or {}).items():
            field_name = str(field_name)
            if field_name.startswith('splineCamParams_'):
                if spline_data is not None:
                    self._assign_signal_property(spline_data, field_name.replace('splineCamParams_', '', 1), value)
                continue
            if field_name == 'splineCamDataPtr':
                pointer_values = list(value or []) if isinstance(value, list) else []
                for slot in range(3):
                    slot_value = pointer_values[slot] if slot < len(pointer_values) else None
                    self._assign_signal_spline_pointer(getattr(signal_data, f'splineCamDataPtr{slot}', None), slot, slot_value)
                continue
            if field_name in {'enableAttackWaves', 'disableAttackWaves', 'killAttackWaves'}:
                self._assign_signal_attack_wave_collection(getattr(signal_data, field_name, None), value)
                continue
            self._assign_signal_property(signal_data, field_name, value)

    def _build_signal_object(self, signal: TerrainSignal, parent: Optional[bpy.types.Object] = None) -> Optional[bpy.types.Object]:
        signal_id = int(getattr(signal, 'signal_id', -1))
        signal_index = int(getattr(signal, 'index', -1))
        obj = self._build_signal_mesh_object(
            getattr(signal, 'mesh', None),
            parent=parent,
            object_name=f'Signal_{signal_id:03d}' if signal_id >= 0 else f'Signal_{signal_index:03d}',
        )
        if obj is None:
            return None
        self._apply_signal_properties_to_object(obj, signal)
        return obj

    @staticmethod
    def _collision_client_flag_label(client_flag: int) -> str:
        client_flag = int(client_flag)
        labels = {
            0: 'Wall',
            32: 'Ground',
            4: 'Slope',
            1: 'Water',
            16: 'Water1',
            17: 'Water2',
            48: 'Snow',
        }
        return labels.get(client_flag, str(client_flag))

    def _terrain_collision_color(self, client_flag: int) -> Tuple[float, float, float, float]:
        client_flag = int(client_flag)
        colors = {
            0: (1.0, 0.0, 0.0),      # Wall
            32: (0.0, 1.0, 0.0),     # Ground
            4: (1.0, 1.0, 0.0),      # Slope
            1: (0.0, 0.0, 1.0),      # Water
            48: (1.0, 1.0, 1.0),     # Snow
        }
        red, green, blue = colors.get(client_flag, (1.0, 0.0, 1.0))
        return (red, green, blue, 0.05)

    def _get_or_create_terrain_collision_material(self, client_flag: int) -> bpy.types.Material:
        client_flag = int(client_flag)
        material = self._collision_material_cache.get(client_flag)
        if material is not None and getattr(material, 'name', None) in bpy.data.materials:
            return material

        label = self._collision_client_flag_label(client_flag)
        material_name = label
        material = bpy.data.materials.get(material_name)
        if material is None:
            material = bpy.data.materials.new(material_name)
        material.use_nodes = True
        material.blend_method = 'BLEND'
        if hasattr(material, 'shadow_method'):
            material.shadow_method = 'NONE'
        color = self._terrain_collision_color(client_flag)
        material.diffuse_color = color
        material['trlau_collision_client_flag'] = client_flag
        material['trlau_collision_client_flag_label'] = label
        principled = None
        if material.node_tree is not None:
            principled = material.node_tree.nodes.get('Principled BSDF')
        if principled is not None:
            principled.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
            principled.inputs['Alpha'].default_value = color[3]
        self._collision_material_cache[client_flag] = material
        return material

    def _get_or_create_group_collision_empty(self, group: TerrainGroup) -> Optional[bpy.types.Object]:
        collision = getattr(group, 'collision', None)
        if collision is None or not collision.vertices or not collision.faces:
            return None

        empty_name = self._prefixed_name(f'TerrainGroup_Collision_{int(group.index):03d}')
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
            self.collection.objects.link(empty)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 48.0
        empty.location = tuple(float(v) for v in getattr(collision, 'position', (0.0, 0.0, 0.0)))
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        empty.parent = self._get_or_create_terrain_empty()
        empty['trlau_level'] = True
        empty['trlau_terrain_collision'] = True
        empty['trlau_terrain_collision_empty'] = True
        empty['trlau_terrain_group_index'] = int(group.index)
        empty['trlau_terrain_collision_position'] = tuple(float(v) for v in getattr(collision, 'position', (0.0, 0.0, 0.0)))
        empty['trlau_terrain_collision_bbox_min'] = tuple(float(v) for v in getattr(collision, 'bbox_min', (0.0, 0.0, 0.0)))
        empty['trlau_terrain_collision_bbox_max'] = tuple(float(v) for v in getattr(collision, 'bbox_max', (0.0, 0.0, 0.0)))
        return empty

    def _get_or_create_group_collision_kd_empty(self, group: TerrainGroup, parent: Optional[bpy.types.Object] = None) -> Optional[bpy.types.Object]:
        collision = getattr(group, 'collision', None)
        if collision is None or not getattr(collision, 'kd_nodes', None):
            return None

        empty_name = self._prefixed_name(f'TerrainGroup_Collision_KD_{int(group.index):03d}')
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
            self.collection.objects.link(empty)
        elif all(coll != self.collection for coll in getattr(empty, 'users_collection', ())):
            try:
                self.collection.objects.link(empty)
            except Exception:
                pass
        if parent is not None:
            empty.parent = parent
            try:
                empty.matrix_parent_inverse.identity()
            except Exception:
                pass
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 32.0
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)
        empty['trlau_level'] = True
        empty['trlau_terrain_collision_kd'] = True
        empty['trlau_terrain_group_index'] = int(group.index)
        empty['trlau_collision_kd_depth'] = int(getattr(collision, 'kd_max_depth', 0))
        return empty

    def _build_collision_kd_objects(self, group: TerrainGroup, parent: Optional[bpy.types.Object] = None) -> List[bpy.types.Object]:
        collision = getattr(group, 'collision', None)
        if collision is None or not getattr(collision, 'kd_nodes', None):
            return []

        kd_root = self._get_or_create_group_collision_kd_empty(group, parent=parent)
        built: List[bpy.types.Object] = []
        if kd_root is not None:
            built.append(kd_root)

        for node in getattr(collision, 'kd_nodes', []):
            node_name = self._prefixed_name(f'TerrainGroup_Collision_KD_{int(group.index):03d}_Node_{int(node.node_index):03d}')
            obj = bpy.data.objects.get(node_name)
            if obj is None:
                obj = bpy.data.objects.new(node_name, None)
                self.collection.objects.link(obj)
            elif all(coll != self.collection for coll in getattr(obj, 'users_collection', ())):
                try:
                    self.collection.objects.link(obj)
                except Exception:
                    pass
            if kd_root is not None:
                obj.parent = kd_root
                try:
                    obj.matrix_parent_inverse.identity()
                except Exception:
                    pass
            display_min, display_max = _convert_raw_bounds_to_level(
                tuple(float(v) for v in getattr(node, 'bounds_min_raw', (0.0, 0.0, 0.0))),
                tuple(float(v) for v in getattr(node, 'bounds_max_raw', (0.0, 0.0, 0.0))),
            )
            center = tuple((float(display_min[index]) + float(display_max[index])) * 0.5 for index in range(3))
            half_extents = tuple(max(0.01, abs(float(display_max[index]) - float(display_min[index])) * 0.5) for index in range(3))
            obj.empty_display_type = 'CUBE'
            obj.empty_display_size = 1.0
            obj.location = center
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = half_extents
            obj['trlau_level'] = True
            obj['trlau_terrain_collision_kd_node'] = True
            obj['trlau_terrain_group_index'] = int(group.index)
            obj['trlau_collision_kd_node_index'] = int(getattr(node, 'node_index', 0))
            obj['trlau_collision_kd_axis'] = int(getattr(node, 'axis', 0))
            obj['trlau_collision_kd_num_faces'] = int(getattr(node, 'num_faces', 0))
            obj['trlau_collision_kd_ref_index'] = int(getattr(node, 'ref_index', 0))
            obj['trlau_collision_kd_depth'] = int(getattr(node, 'depth', 0))
            obj['trlau_collision_kd_face_start'] = int(getattr(node, 'face_start', 0))
            obj['trlau_collision_kd_neg_offset'] = float(getattr(node, 'neg_offset', 0.0))
            obj['trlau_collision_kd_pos_offset'] = float(getattr(node, 'pos_offset', 0.0))
            obj['trlau_collision_kd_bbox_min'] = tuple(float(v) for v in display_min)
            obj['trlau_collision_kd_bbox_max'] = tuple(float(v) for v in display_max)
            built.append(obj)

        return built

    def _build_terrain_group_collision_object(self, group: TerrainGroup, parent: Optional[bpy.types.Object] = None) -> Optional[bpy.types.Object]:
        collision = getattr(group, 'collision', None)
        if collision is None or not collision.vertices or not collision.faces:
            return None

        vertices = [
            (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            for vertex in collision.vertices
        ]
        faces = []
        source_faces = []
        for face in collision.faces:
            tri = (int(face.i0), int(face.i1), int(face.i2))
            if min(tri) < 0 or max(tri) >= len(vertices) or len(set(tri)) != 3:
                continue
            faces.append(tri)
            source_faces.append(face)
        if not faces:
            return None

        mesh_name = self._prefixed_name(f'TerrainGroup_{int(group.index)}_Collision')
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(mesh_name, mesh)
        self.collection.objects.link(obj)
        if parent is not None:
            obj.parent = parent
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)
        obj['trlau_level'] = True
        obj['trlau_terrain_collision'] = True
        obj['trlau_terrain_group_index'] = int(group.index)
        obj.display_type = 'SOLID'

        client_flag_to_slot: dict[int, int] = {}
        for face_index, face in enumerate(source_faces):
            client_flag = int(getattr(face, 'client_flags', 0) or 0)
            slot_index = client_flag_to_slot.get(client_flag)
            if slot_index is None:
                material = self._get_or_create_terrain_collision_material(client_flag)
                obj.data.materials.append(material)
                slot_index = len(obj.data.materials) - 1
                client_flag_to_slot[client_flag] = slot_index
            if face_index < len(obj.data.polygons):
                obj.data.polygons[face_index].material_index = slot_index

        return obj


    def _build_terrain_group_mesh_object(self, level: LevelData, group: TerrainGroup, parent: Optional[bpy.types.Object] = None) -> Optional[bpy.types.Object]:
        vertices: List[Tuple[float, float, float]] = []
        local_uvs: List[Tuple[float, float]] = []
        vertex_colors: List[Tuple[int, int, int, int]] = []
        faces: List[Tuple[int, int, int]] = []
        face_material_indices: List[int] = []
        materials: List[bpy.types.Material] = []
        slot_records: List[dict[str, object]] = []
        has_any_vertex_colors = False

        for strip in group.strips:
            if len(strip.indices) < 3:
                continue

            use_vmo = strip.uses_vmo_buffer
            source_vertices = level.vmo_vertices if use_vmo else level.vertices
            source_uvs = level.vmo_uvs if use_vmo else level.uvs
            source_vertex_colors = level.vmo_vertex_colors if use_vmo else level.vertex_colors
            if not source_vertices:
                continue

            adjusted = [int(raw_index) + int(strip.vertex_base_offset) for raw_index in strip.indices]
            if not adjusted or len(adjusted) < 3:
                continue
            max_index = len(source_vertices) - 1
            if min(adjusted) < 0 or max(adjusted) > max_index:
                logger.warning(
                    'Adjusted strip indices out of range for group %d strip %d: min=%d max=%d vertex_count=%d',
                    group.index,
                    strip.material_index,
                    min(adjusted),
                    max(adjusted),
                    len(source_vertices),
                )
                continue

            used_indices = adjusted[: (len(adjusted) // 3) * 3]
            if len(used_indices) < 3:
                continue

            unique_map: dict[int, int] = {}
            strip_faces: List[Tuple[int, int, int]] = []
            strip_ordered_indices: List[int] = []
            strip_base_vertex = len(vertices)
            for tri_start in range(0, len(used_indices), 3):
                tri_source = used_indices[tri_start:tri_start + 3]
                if len(tri_source) != 3 or len(set(tri_source)) < 3:
                    continue
                face: List[int] = []
                for src_index in tri_source:
                    local_index = unique_map.get(src_index)
                    if local_index is None:
                        local_index = len(vertices)
                        unique_map[src_index] = local_index
                        strip_ordered_indices.append(src_index)
                        source_vertex = source_vertices[src_index]
                        vertices.append((
                            float(source_vertex[0]),
                            float(source_vertex[1]),
                            float(source_vertex[2]),
                        ))
                        local_uv = source_uvs[src_index] if 0 <= src_index < len(source_uvs) else (0.0, 0.0)
                        local_uvs.append((float(local_uv[0]), 1.0 - float(local_uv[1])))
                        if source_vertex_colors and 0 <= src_index < len(source_vertex_colors):
                            color = source_vertex_colors[src_index]
                            vertex_colors.append((int(color[0]), int(color[1]), int(color[2]), int(color[3])))
                            has_any_vertex_colors = True
                        else:
                            vertex_colors.append((255, 255, 255, 128))
                    face.append(local_index)
                if len(set(face)) == 3:
                    strip_faces.append(tuple(face))

            if not strip_faces:
                del vertices[strip_base_vertex:]
                del local_uvs[strip_base_vertex:]
                del vertex_colors[strip_base_vertex:]
                continue

            use_vertex_colors = bool(source_vertex_colors) and all(0 <= index < len(source_vertex_colors) for index in strip_ordered_indices)
            material = self._get_or_create_material(strip, use_vertex_colors)
            slot_index = len(materials)
            materials.append(material)
            slot_records.append({
                'slot': slot_index,
                'strip_index': int(strip.material_index),
                'tpageid': int(strip.tpageid),
                'flags': int(strip.flags),
                'vertex_buffer': 'vmo' if use_vmo else 'terrain',
                'vertex_base_offset': int(strip.vertex_base_offset),
            })
            for face in strip_faces:
                faces.append(face)
                face_material_indices.append(slot_index)

        if not faces or not vertices:
            return None

        mesh_name = self._prefixed_name(f'TerrainGroup_{int(group.index)}_Mesh')
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        uv_layer = mesh.uv_layers.new(name='UVMap')
        if uv_layer is not None:
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    uv_layer.data[loop_index].uv = local_uvs[vertex_index]

        if has_any_vertex_colors:
            self._apply_vertex_colors_from_rgba(mesh, vertex_colors)

        for material in materials:
            mesh.materials.append(material)
        for poly_index, poly in enumerate(mesh.polygons):
            if poly_index < len(face_material_indices):
                poly.material_index = int(face_material_indices[poly_index])

        obj = bpy.data.objects.new(mesh_name, mesh)
        self.collection.objects.link(obj)
        if parent is not None:
            obj.parent = parent
            obj.matrix_parent_inverse.identity()
        # Terrain mesh identity and strip data are derived from parent/name, material slots,
        # mesh geometry and material properties during export.  Do not attach duplicate ID props.
        return obj

    def _build_strip_object(self, level: LevelData, group: TerrainGroup, strip: LevelStrip, parent: Optional[bpy.types.Object] = None) -> Optional[bpy.types.Object]:
        if len(strip.indices) < 3:
            return None

        use_vmo = strip.uses_vmo_buffer
        source_vertices = level.vmo_vertices if use_vmo else level.vertices
        source_uvs = level.vmo_uvs if use_vmo else level.uvs
        source_vertex_colors = level.vmo_vertex_colors if use_vmo else level.vertex_colors
        if not source_vertices:
            return None

        adjusted = [int(raw_index) + int(strip.vertex_base_offset) for raw_index in strip.indices]
        if not adjusted or len(adjusted) < 3:
            return None
        max_index = len(source_vertices) - 1
        if min(adjusted) < 0 or max(adjusted) > max_index:
            logger.warning(
                'Adjusted strip indices out of range for group %d strip %d: min=%d max=%d vertex_count=%d',
                group.index,
                strip.material_index,
                min(adjusted),
                max(adjusted),
                len(source_vertices),
            )
            return None

        used_indices = adjusted[: (len(adjusted) // 3) * 3]
        if len(used_indices) < 3:
            return None

        group_position = tuple(float(v) for v in group.position)

        unique_map: dict[int, int] = {}
        vertices: List[Tuple[float, float, float]] = []
        local_uvs: List[Tuple[float, float]] = []
        ordered_indices: List[int] = []
        faces: List[Tuple[int, int, int]] = []

        for tri_start in range(0, len(used_indices), 3):
            face: List[int] = []
            tri_source = used_indices[tri_start:tri_start + 3]
            for src_index in tri_source:
                local_index = unique_map.get(src_index)
                if local_index is None:
                    local_index = len(vertices)
                    unique_map[src_index] = local_index
                    source_vertex = source_vertices[src_index]
                    vertices.append((
                        float(source_vertex[0]),
                        float(source_vertex[1]),
                        float(source_vertex[2]),
                    ))
                    local_uv = source_uvs[src_index] if 0 <= src_index < len(source_uvs) else (0.0, 0.0)
                    local_uvs.append((float(local_uv[0]), 1.0 - float(local_uv[1])))
                    ordered_indices.append(src_index)
                face.append(local_index)
            if len(set(face)) == 3:
                faces.append(tuple(face))

        if not faces:
            return None

        mesh_name = self._prefixed_name(f'TerrainGroup_{int(group.index)}_Strip_{int(strip.material_index)}')
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        uv_layer = mesh.uv_layers.new(name='UVMap')
        if uv_layer is not None:
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    uv_layer.data[loop_index].uv = local_uvs[vertex_index]

        use_vertex_colors = self._apply_level_vertex_colors(mesh, source_vertex_colors, ordered_indices)

        material = self._get_or_create_material(strip, use_vertex_colors)
        mesh.materials.append(material)
        for poly in mesh.polygons:
            poly.material_index = 0

        obj = bpy.data.objects.new(mesh_name, mesh)
        self.collection.objects.link(obj)
        if parent is not None:
            obj.parent = parent
            obj.matrix_parent_inverse.identity()
        # Per-strip object metadata is represented by its material and geometry.
        return obj


    def _get_or_create_bgobject_empty(self, bg_object: BGObject) -> bpy.types.Object:
        import mathutils

        empty_name = self._prefixed_name(f'BGObject_{int(bg_object.index):03d}')
        target_collection = self.bgobject_collection or self.collection
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
            target_collection.objects.link(empty)
        elif all(coll != target_collection for coll in getattr(empty, 'users_collection', ())):
            try:
                target_collection.objects.link(empty)
            except Exception:
                pass
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 48.0
        empty.parent = self._get_or_create_component_empty('BGInstance')
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        bg_position = tuple(float(v) for v in getattr(bg_object, 'position', (0.0, 0.0, 0.0)))
        empty.matrix_local = mathutils.Matrix.Translation(bg_position) @ _make_bgobject_local_matrix(getattr(bg_object, 'scale', (1.0, 1.0, 1.0)))
        empty['trlau_level'] = True
        empty['trlau_bgobject'] = True
        empty['trlau_bgobject_empty'] = True
        empty['trlau_bgobject_index'] = int(bg_object.index)
        empty['trlau_bgobject_stride'] = int(getattr(bg_object, 'stride', _BGOBJECT_MODEL_SIZE))
        empty['trlau_bgobject_position'] = tuple(float(v) for v in getattr(bg_object, 'position', (0.0, 0.0, 0.0)))
        empty['trlau_bgobject_scale'] = tuple(float(v) for v in getattr(bg_object, 'scale', (1.0, 1.0, 1.0)))
        try:
            empty['trlau_bgobject_scale_x'] = float(bg_object.scale[0])
            empty['trlau_bgobject_scale_y'] = float(bg_object.scale[1])
            empty['trlau_bgobject_scale_z'] = float(bg_object.scale[2])
            empty['trlau_bgobject_position_x'] = float(bg_object.position[0])
            empty['trlau_bgobject_position_y'] = float(bg_object.position[1])
            empty['trlau_bgobject_position_z'] = float(bg_object.position[2])
        except Exception:
            pass
        try:
            empty['trlau_bgobject_scale_x'] = float(bg_object.scale[0])
            empty['trlau_bgobject_scale_y'] = float(bg_object.scale[1])
            empty['trlau_bgobject_scale_z'] = float(bg_object.scale[2])
            empty['trlau_bgobject_position_x'] = float(bg_object.position[0])
            empty['trlau_bgobject_position_y'] = float(bg_object.position[1])
            empty['trlau_bgobject_position_z'] = float(bg_object.position[2])
        except Exception:
            pass
        empty['trlau_bgobject_flags'] = int(getattr(bg_object, 'flags', 0))
        empty['trlau_bgobject_cdc_render_data_id'] = int(getattr(bg_object, 'cdc_render_data_id', 0))
        if getattr(bg_object, 'position_candidates', None):
            for candidate_offset, candidate_position in bg_object.position_candidates:
                suffix = f'{int(candidate_offset):02d}'
                empty[f'trlau_bgobject_position_candidate_{suffix}_x'] = float(candidate_position[0])
                empty[f'trlau_bgobject_position_candidate_{suffix}_y'] = float(candidate_position[1])
                empty[f'trlau_bgobject_position_candidate_{suffix}_z'] = float(candidate_position[2])
        _store_bgobject_raw_blob_metadata(empty, bg_object)
        return empty

    def _get_or_create_bginstance_empty(self, bg_object: BGObject, bg_instance: BGInstance) -> bpy.types.Object:
        import mathutils

        empty_name = self._prefixed_name(f'BGObject_{int(bg_object.index):03d}_Instance_{int(bg_instance.index):03d}')
        target_collection = self.bgobject_collection or self.collection
        empty = bpy.data.objects.get(empty_name)
        if empty is None:
            empty = bpy.data.objects.new(empty_name, None)
            target_collection.objects.link(empty)
        elif all(coll != target_collection for coll in getattr(empty, 'users_collection', ())):
            try:
                target_collection.objects.link(empty)
            except Exception:
                pass
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 48.0
        empty.parent = self._get_or_create_component_empty('BGInstance')
        empty.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        empty.matrix_local = _convert_bginstance_matrix(bg_instance.matrix_rows) @ _make_bgobject_local_matrix(getattr(bg_object, 'scale', (1.0, 1.0, 1.0)))
        empty['trlau_level'] = True
        empty['trlau_type'] = 'BGInstance'
        empty['trlau_bgobject'] = True
        empty['trlau_bginstance'] = True
        empty['trlau_bginstance_empty'] = True
        empty['trlau_bgobject_index'] = int(bg_object.index)
        empty['trlau_bginstance_index'] = int(bg_instance.index)
        empty['trlau_bginstance_id'] = int(bg_instance.instance_id)
        empty['trlau_bginstance_flags'] = int(getattr(bg_instance, 'flags', 0))
        empty['trlau_bginstance_multi_spline_offset'] = int(getattr(bg_instance, 'multi_spline_offset', 0))
        try:
            from ...ui.level_panel import trlau_load_multispline_property
            trlau_load_multispline_property(empty, getattr(bg_instance, 'multi_spline_data', None))
        except Exception:
            pass
        try:
            if 'trlau_bginstance_multi_spline_json' in empty:
                del empty['trlau_bginstance_multi_spline_json']
        except Exception:
            pass
        empty['trlau_bginstance_bgflags'] = int(getattr(bg_instance, 'bg_flags', 0))
        empty['trlau_bginstance_target_frame'] = int(getattr(bg_instance, 'target_frame', 0))
        empty['trlau_bginstance_clip_beg'] = int(getattr(bg_instance, 'clip_beg', 0))
        empty['trlau_bginstance_clip_end'] = int(getattr(bg_instance, 'clip_end', 0))
        empty['trlau_bginstance_link_seg'] = int(getattr(bg_instance, 'link_seg', 0))
        empty['trlau_bginstance_link_instance_offset'] = int(getattr(bg_instance, 'link_instance_offset', 0))
        empty['trlau_bginstance_color_data_index'] = int(getattr(bg_instance, 'color_data_index', 0))
        empty['trlau_bginstance_lod'] = int(getattr(bg_instance, 'lod', 0))
        empty['trlau_bginstance_active_light_bitfield'] = int(getattr(bg_instance, 'active_light_bitfield', 0))
        empty['trlau_bginstance_bgobject_index'] = int(bg_instance.bg_object_index)
        empty['trlau_bginstance_bgobject_offset'] = int(bg_instance.bg_object_offset)
        empty['trlau_bgobject_stride'] = int(getattr(bg_object, 'stride', _BGOBJECT_MODEL_SIZE))
        empty['trlau_bgobject_position'] = tuple(float(v) for v in getattr(bg_object, 'position', (0.0, 0.0, 0.0)))
        empty['trlau_bgobject_scale'] = tuple(float(v) for v in getattr(bg_object, 'scale', (1.0, 1.0, 1.0)))
        empty['trlau_bgobject_flags'] = int(getattr(bg_object, 'flags', 0))
        empty['trlau_bgobject_cdc_render_data_id'] = int(getattr(bg_object, 'cdc_render_data_id', 0))
        for row_index, row in enumerate(bg_instance.matrix_rows):
            for col_index, value in enumerate(row):
                empty[f'trlau_bginstance_m{row_index}{col_index}'] = float(value)
        return empty

    @staticmethod
    def _decode_bgobject_vertex_color_slot(bg_object: BGObject, color_data_index: int = 0) -> List[Tuple[int, int, int, int]]:
        vertex_count = len(getattr(bg_object, 'vertices', []) or [])
        if vertex_count <= 0:
            return []

        stored_vertex_colors = list(getattr(bg_object, 'vertex_colors', []) or [])
        raw_color_data = bytes(getattr(bg_object, 'raw_color_data', b'') or b'')
        per_slot_size = int(vertex_count) * 4
        if per_slot_size <= 0 or len(raw_color_data) < per_slot_size:
            return stored_vertex_colors

        slot_count = max(1, len(raw_color_data) // per_slot_size)
        try:
            slot_index = int(color_data_index)
        except Exception:
            slot_index = 0
        if slot_index < 0 or slot_index >= slot_count:
            logger.warning(
                'BGObject %d color_data_index %d is outside available color slots 0..%d; using slot 0',
                int(getattr(bg_object, 'index', -1)),
                slot_index,
                max(0, slot_count - 1),
            )
            slot_index = 0

        start = slot_index * per_slot_size
        end = start + per_slot_size
        if end > len(raw_color_data):
            return stored_vertex_colors

        colors: List[Tuple[int, int, int, int]] = []
        for offset in range(start, end, 4):
            color_bgra = struct.unpack_from('<I', raw_color_data, offset)[0]
            b = color_bgra & 0xFF
            g = (color_bgra >> 8) & 0xFF
            r = (color_bgra >> 16) & 0xFF
            a = (color_bgra >> 24) & 0xFF
            colors.append((r, g, b, a))
        return colors if len(colors) == vertex_count else stored_vertex_colors

    @classmethod
    def _bgobject_vertex_colors_for_instance(cls, bg_object: BGObject, bg_instance: Optional[BGInstance]) -> List[Tuple[int, int, int, int]]:
        color_data_index = int(getattr(bg_instance, 'color_data_index', 0) or 0) if bg_instance is not None else 0
        return cls._decode_bgobject_vertex_color_slot(bg_object, color_data_index)


    def _build_bgobject_mesh_object(self, bg_object: BGObject, bg_instance: Optional[BGInstance] = None) -> Optional[bpy.types.Object]:
        """Build one editable BGObject mesh for a BGInstance, using source vertex indices directly.

        Earlier imports compacted each mesh to only used vertices and stored a
        trlau_bgobject_source_indices property to map local vertices back to the
        BGObject vertex/color list.  That property is easy to delete during
        metadata cleanup, and then per-instance vertex-color export loses the
        original color-slot layout.  Keep the Blender mesh vertex index identical
        to the BGObject source vertex index instead; export can then calculate
        the mapping from the mesh itself.
        """
        if not bg_object.vertices:
            return None

        source_vertex_count = len(bg_object.vertices)
        vertices: List[Tuple[float, float, float]] = []
        local_uvs: List[Tuple[float, float]] = []
        source_vertex_colors = self._bgobject_vertex_colors_for_instance(bg_object, bg_instance)
        vertex_colors: List[Tuple[int, int, int, int]] = []
        has_any_vertex_colors = bool(source_vertex_colors)

        for src_index, source_vertex in enumerate(bg_object.vertices):
            vertices.append((float(source_vertex[0]), float(source_vertex[1]), float(source_vertex[2])))
            local_uv = bg_object.uvs[src_index] if 0 <= src_index < len(bg_object.uvs) else (0.0, 0.0)
            local_uvs.append((float(local_uv[0]), 1.0 - float(local_uv[1])))
            if source_vertex_colors and 0 <= src_index < len(source_vertex_colors):
                color = source_vertex_colors[src_index]
                vertex_colors.append((int(color[0]), int(color[1]), int(color[2]), int(color[3])))
            else:
                vertex_colors.append((255, 255, 255, 128))

        faces: List[Tuple[int, int, int]] = []
        face_material_indices: List[int] = []
        materials: List[bpy.types.Material] = []
        slot_records: List[dict[str, object]] = []

        for strip in bg_object.strips:
            if len(strip.indices) < 3:
                continue
            adjusted = [int(raw_index) + int(strip.vertex_base_offset) for raw_index in strip.indices]
            if not adjusted or len(adjusted) < 3:
                continue
            max_index = source_vertex_count - 1
            if min(adjusted) < 0 or max(adjusted) > max_index:
                logger.warning(
                    'Adjusted BGObject strip indices out of range for bgobject %d strip %d: min=%d max=%d vertex_count=%d',
                    bg_object.index,
                    strip.material_index,
                    min(adjusted),
                    max(adjusted),
                    source_vertex_count,
                )
                continue

            used_indices = adjusted[: (len(adjusted) // 3) * 3]
            strip_faces: List[Tuple[int, int, int]] = []
            for tri_start in range(0, len(used_indices), 3):
                tri_source = used_indices[tri_start:tri_start + 3]
                if len(tri_source) == 3 and len(set(tri_source)) == 3:
                    strip_faces.append((int(tri_source[0]), int(tri_source[1]), int(tri_source[2])))

            if not strip_faces:
                continue

            material = self._get_or_create_material(strip, has_any_vertex_colors)
            slot_index = len(materials)
            materials.append(material)
            slot_records.append({
                'slot': slot_index,
                'strip_index': int(strip.material_index),
                'tpageid': int(strip.tpageid),
                'flags': int(strip.flags),
                'vertex_base_offset': int(strip.vertex_base_offset),
                'sort_vertex': tuple(getattr(strip, 'sort_vertex', (0, 0, 0)) or (0, 0, 0)),
                'sort_push': int(getattr(strip, 'sort_push', 0) or 0),
                'scroll_offset': float(getattr(strip, 'scroll_offset', 0.0) or 0.0),
                'raw_count': int(getattr(strip, 'raw_count', -1) or -1),
            })
            for face in strip_faces:
                faces.append(face)
                face_material_indices.append(slot_index)

        if not faces or not vertices:
            return None

        if bg_instance is not None:
            mesh_name = self._prefixed_name(f'BGObject_{int(bg_object.index):03d}_Instance_{int(bg_instance.index):03d}_Mesh')
        else:
            mesh_name = self._prefixed_name(f'BGObject_{int(bg_object.index):03d}_Mesh')
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        uv_layer = mesh.uv_layers.new(name='UVMap')
        if uv_layer is not None:
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    uv_layer.data[loop_index].uv = local_uvs[vertex_index]

        if has_any_vertex_colors:
            self._apply_vertex_colors_from_rgba(mesh, vertex_colors)

        for material in materials:
            mesh.materials.append(material)
        for poly_index, poly in enumerate(mesh.polygons):
            if poly_index < len(face_material_indices):
                poly.material_index = int(face_material_indices[poly_index])

        obj = bpy.data.objects.new(mesh_name, mesh)
        target_collection = self.bgobject_collection or self.collection
        target_collection.objects.link(obj)
        if bg_instance is not None:
            instance_empty = self._get_or_create_bginstance_empty(bg_object, bg_instance)
            obj.parent = instance_empty
        else:
            bgobject_empty = self._get_or_create_bgobject_empty(bg_object)
            obj.parent = bgobject_empty
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        obj.location = (0.0, 0.0, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.scale = (1.0, 1.0, 1.0)

        bg_position = tuple(float(v) for v in getattr(bg_object, 'position', (0.0, 0.0, 0.0)))
        obj['trlau_level'] = True
        obj['trlau_type'] = 'BGObject'
        obj['trlau_bgobject'] = True
        obj['trlau_bgobject_mesh'] = True
        obj['trlau_bgobject_index'] = int(bg_object.index)
        obj['trlau_bgobject_scale_x'] = float(bg_object.scale[0])
        obj['trlau_bgobject_scale_y'] = float(bg_object.scale[1])
        obj['trlau_bgobject_scale_z'] = float(bg_object.scale[2])
        obj['trlau_bgobject_position_x'] = float(bg_position[0])
        obj['trlau_bgobject_position_y'] = float(bg_position[1])
        obj['trlau_bgobject_position_z'] = float(bg_position[2])
        obj['trlau_bgobject_stride'] = int(getattr(bg_object, 'stride', _BGOBJECT_MODEL_SIZE))
        obj['trlau_bgobject_flags'] = int(getattr(bg_object, 'flags', 0))
        obj['trlau_bgobject_cdc_render_data_id'] = int(getattr(bg_object, 'cdc_render_data_id', 0))
        obj['trlau_level_vertex_buffer'] = 'bgobject'
        obj['trlau_bgobject_vertex_count'] = int(source_vertex_count)
        obj['trlau_bgobject_color_data_index'] = int(getattr(bg_instance, 'color_data_index', 0) or 0) if bg_instance is not None else 0
        obj['trlau_bgobject_color_slot_count'] = int(getattr(bg_object, 'raw_color_slot_count', 1) or 1)
        obj['trlau_vertex_color_count'] = len(source_vertex_colors)
        if getattr(bg_object, 'position_candidates', None):
            for candidate_offset, candidate_position in bg_object.position_candidates:
                suffix = f'{int(candidate_offset):02d}'
                obj[f'trlau_bgobject_position_candidate_{suffix}_x'] = float(candidate_position[0])
                obj[f'trlau_bgobject_position_candidate_{suffix}_y'] = float(candidate_position[1])
                obj[f'trlau_bgobject_position_candidate_{suffix}_z'] = float(candidate_position[2])
        obj['trlau_bgobject_slot_count'] = int(len(slot_records))
        for record in slot_records:
            slot = int(record['slot'])
            prefix = f'trlau_bgobject_slot_{slot:03d}_'
            obj[prefix + 'strip_index'] = int(record['strip_index'])
            obj[prefix + 'tpageid'] = int(record['tpageid'])
            obj[prefix + 'flags'] = int(record['flags'])
            obj[prefix + 'vertex_base_offset'] = int(record['vertex_base_offset'])
            sort_vertex = tuple(record.get('sort_vertex', (0, 0, 0)) or (0, 0, 0))
            obj[prefix + 'sort_vertex_x'] = int(sort_vertex[0]) if len(sort_vertex) > 0 else 0
            obj[prefix + 'sort_vertex_y'] = int(sort_vertex[1]) if len(sort_vertex) > 1 else 0
            obj[prefix + 'sort_vertex_z'] = int(sort_vertex[2]) if len(sort_vertex) > 2 else 0
            obj[prefix + 'sort_push'] = int(record['sort_push'])
            obj[prefix + 'scroll_offset'] = float(record['scroll_offset'])
            obj[prefix + 'raw_count'] = int(record['raw_count'])
        return obj

    def _build_bgobject_strip_object(self, bg_object: BGObject, strip: LevelStrip, bg_instance: Optional[BGInstance] = None) -> Optional[bpy.types.Object]:
        if len(strip.indices) < 3 or not bg_object.vertices:
            return None

        adjusted = [int(raw_index) + int(strip.vertex_base_offset) for raw_index in strip.indices]
        if not adjusted or len(adjusted) < 3:
            return None
        max_index = len(bg_object.vertices) - 1
        if min(adjusted) < 0 or max(adjusted) > max_index:
            logger.warning(
                'Adjusted BGObject strip indices out of range for bgobject %d strip %d: min=%d max=%d vertex_count=%d',
                bg_object.index,
                strip.material_index,
                min(adjusted),
                max(adjusted),
                len(bg_object.vertices),
            )
            return None

        used_indices = adjusted[: (len(adjusted) // 3) * 3]
        if len(used_indices) < 3:
            return None

        bg_position = tuple(float(v) for v in getattr(bg_object, 'position', (0.0, 0.0, 0.0)))

        unique_map: dict[int, int] = {}
        vertices: List[Tuple[float, float, float]] = []
        local_uvs: List[Tuple[float, float]] = []
        ordered_indices: List[int] = []
        faces: List[Tuple[int, int, int]] = []

        for tri_start in range(0, len(used_indices), 3):
            tri_source = used_indices[tri_start:tri_start + 3]
            if len(tri_source) != 3 or len(set(tri_source)) < 3:
                continue
            face: List[int] = []
            for src_index in tri_source:
                local_index = unique_map.get(src_index)
                if local_index is None:
                    local_index = len(vertices)
                    unique_map[src_index] = local_index
                    source_vertex = bg_object.vertices[src_index]
                    vx = float(source_vertex[0])
                    vy = float(source_vertex[1])
                    vz = float(source_vertex[2])
                    vertices.append((vx, vy, vz))
                    local_uv = bg_object.uvs[src_index] if 0 <= src_index < len(bg_object.uvs) else (0.0, 0.0)
                    local_uvs.append((float(local_uv[0]), 1.0 - float(local_uv[1])))
                    ordered_indices.append(src_index)
                face.append(local_index)
            if len(set(face)) == 3:
                faces.append(tuple(face))

        if not faces:
            return None

        if bg_instance is not None:
            mesh_name = self._prefixed_name(f'BGObject_{bg_object.index:03d}_Instance_{bg_instance.index:03d}_Strip_{strip.material_index:03d}')
        else:
            mesh_name = self._prefixed_name(f'BGObject_{bg_object.index:03d}_Strip_{strip.material_index:03d}')
        mesh = bpy.data.meshes.new(mesh_name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        uv_layer = mesh.uv_layers.new(name='UVMap')
        if uv_layer is not None:
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    uv_layer.data[loop_index].uv = local_uvs[vertex_index]

        source_vertex_colors = self._bgobject_vertex_colors_for_instance(bg_object, bg_instance)
        use_vertex_colors = self._apply_level_vertex_colors(mesh, source_vertex_colors, ordered_indices)

        material = self._get_or_create_material(strip, use_vertex_colors)
        mesh.materials.append(material)
        for poly in mesh.polygons:
            poly.material_index = 0

        obj = bpy.data.objects.new(mesh_name, mesh)
        target_collection = self.bgobject_collection or self.collection
        target_collection.objects.link(obj)
        if bg_instance is not None:
            instance_empty = self._get_or_create_bginstance_empty(bg_object, bg_instance)
            obj.parent = instance_empty
            obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            obj.location = (0.0, 0.0, 0.0)
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = (1.0, 1.0, 1.0)
        else:
            bgobject_empty = self._get_or_create_bgobject_empty(bg_object)
            obj.parent = bgobject_empty
            obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            obj.location = (0.0, 0.0, 0.0)
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = (1.0, 1.0, 1.0)
        obj['trlau_level'] = True
        obj['trlau_type'] = 'BGObject'
        obj['trlau_bgobject'] = True
        obj['trlau_bgobject_index'] = int(bg_object.index)
        obj['trlau_bgobject_scale_x'] = float(bg_object.scale[0])
        obj['trlau_bgobject_scale_y'] = float(bg_object.scale[1])
        obj['trlau_bgobject_scale_z'] = float(bg_object.scale[2])
        obj['trlau_bgobject_position_x'] = float(bg_position[0])
        obj['trlau_bgobject_position_y'] = float(bg_position[1])
        obj['trlau_bgobject_position_z'] = float(bg_position[2])
        obj['trlau_bgobject_stride'] = int(getattr(bg_object, 'stride', _BGOBJECT_MODEL_SIZE))
        obj['trlau_bgobject_flags'] = int(getattr(bg_object, 'flags', 0))
        obj['trlau_bgobject_cdc_render_data_id'] = int(getattr(bg_object, 'cdc_render_data_id', 0))
        if getattr(bg_object, 'position_candidates', None):
            for candidate_offset, candidate_position in bg_object.position_candidates:
                suffix = f'{int(candidate_offset):02d}'
                obj[f'trlau_bgobject_position_candidate_{suffix}_x'] = float(candidate_position[0])
                obj[f'trlau_bgobject_position_candidate_{suffix}_y'] = float(candidate_position[1])
                obj[f'trlau_bgobject_position_candidate_{suffix}_z'] = float(candidate_position[2])
        obj['trlau_strip_flags'] = int(strip.flags)
        obj['trlau_level_vertex_buffer'] = 'bgobject'
        obj['trlau_vertex_base_offset'] = int(strip.vertex_base_offset)
        obj['trlau_bgobject_strip_index'] = int(strip.material_index)
        obj['trlau_bgobject_vertex_count'] = int(len(bg_object.vertices))
        obj['trlau_bgobject_color_data_index'] = int(getattr(bg_instance, 'color_data_index', 0) or 0) if bg_instance is not None else 0
        obj['trlau_bgobject_color_slot_count'] = int(getattr(bg_object, 'raw_color_slot_count', 1) or 1)
        obj['trlau_vertex_color_count'] = len(source_vertex_colors)
        try:
            sort_vertex = tuple(getattr(strip, 'sort_vertex', (0, 0, 0)) or (0, 0, 0))
            obj['trlau_bgobject_strip_sort_vertex_x'] = int(sort_vertex[0]) if len(sort_vertex) > 0 else 0
            obj['trlau_bgobject_strip_sort_vertex_y'] = int(sort_vertex[1]) if len(sort_vertex) > 1 else 0
            obj['trlau_bgobject_strip_sort_vertex_z'] = int(sort_vertex[2]) if len(sort_vertex) > 2 else 0
        except Exception:
            pass
        obj['trlau_bgobject_strip_sort_push'] = int(getattr(strip, 'sort_push', 0) or 0)
        obj['trlau_bgobject_strip_scroll_offset'] = float(getattr(strip, 'scroll_offset', 0.0) or 0.0)
        obj['trlau_bgobject_strip_raw_count'] = int(getattr(strip, 'raw_count', -1) or -1)
        try:
            obj['trlau_bgobject_source_indices'] = [int(value) for value in ordered_indices]
        except Exception:
            pass
        return obj

    def _apply_vertex_colors_from_rgba(
        self,
        mesh: bpy.types.Mesh,
        vertex_colors: List[Tuple[int, int, int, int]],
    ) -> bool:
        if not vertex_colors:
            return False
        vertex_count = len(getattr(mesh, 'vertices', []) or [])
        if vertex_count <= 0 or len(vertex_colors) < vertex_count:
            return False

        flat_colors: List[float] = []
        flat_colors_extend = flat_colors.extend
        for vertex_index in range(vertex_count):
            r, g, b, a = vertex_colors[vertex_index]
            flat_colors_extend((int(r) / 255.0, int(g) / 255.0, int(b) / 255.0, min(int(a) / 128.0, 1.0)))

        color_attr = get_or_create_color_attribute(mesh, 'Color', domain='POINT')
        if color_attr is not None:
            try:
                color_attr.data.foreach_set('color', flat_colors)
                set_active_color_attribute(mesh, color_attr, 'Color')
                return True
            except Exception:
                pass

        legacy_vertex_colors = getattr(mesh, 'vertex_colors', None)
        if legacy_vertex_colors is not None and len(mesh.loops) > 0:
            try:
                color_layer = legacy_vertex_colors.get('Color') or legacy_vertex_colors.new(name='Color')
                loop_colors: List[float] = []
                loop_colors_extend = loop_colors.extend
                for loop in mesh.loops:
                    r, g, b, a = vertex_colors[loop.vertex_index]
                    loop_colors_extend((int(r) / 255.0, int(g) / 255.0, int(b) / 255.0, min(int(a) / 128.0, 1.0)))
                color_layer.data.foreach_set('color', loop_colors)
                return True
            except Exception:
                pass

        return False

    def _apply_level_vertex_colors(
        self,
        mesh: bpy.types.Mesh,
        source_vertex_colors: List[Tuple[int, int, int, int]],
        ordered_indices: List[int],
    ) -> bool:
        if not source_vertex_colors:
            return False
        if any(index < 0 or index >= len(source_vertex_colors) for index in ordered_indices):
            return False
        vertex_colors = [
            tuple(int(v) for v in source_vertex_colors[index])
            for index in ordered_indices
        ]
        return self._apply_vertex_colors_from_rgba(mesh, vertex_colors)

    def _configure_level_material(
        self,
        material: bpy.types.Material,
        strip: LevelStrip,
        use_vertex_colors: bool,
    ) -> None:
        tpageid_u32 = int(strip.tpageid) & 0xFFFFFFFF
        tpage_flags = _decode_tpage_flags(tpageid_u32)
        texture_id = tpage_flags['texture_id']
        image = None
        if self.import_textures:
            image = self._runtime_texture_images.get(int(texture_id))
            if image is None:
                image = self._mesh_builder._load_packed_image_from_pcd(texture_id)

        initialize_material_flag_properties(material, (tpageid_u32 - 0x100000000) if (tpageid_u32 & 0x80000000) else tpageid_u32)
        env_mapping = bool(getattr(strip, 'env_mapping', False))
        eye_ref_env_mapping = bool(getattr(strip, 'eye_ref_env_mapping', False))
        reflective = bool(env_mapping or eye_ref_env_mapping)
        reflection_mode = 'both' if env_mapping and eye_ref_env_mapping else ('eye_ref' if eye_ref_env_mapping else 'env')
        try:
            set_material_panel_value(material, 'trlau_ui_env_mapping', env_mapping)
            set_material_panel_value(material, 'trlau_ui_eye_ref_env_mapping', eye_ref_env_mapping)
        except Exception:
            pass

        is_animated = bool(strip.has_scroll_animation)
        raw_scroll_speed = float(strip.scroll_speed) if strip.scroll_speed is not None else 0.0
        try:
            set_material_panel_value(material, 'trlau_ui_scroll_enabled', bool(is_animated))
            set_material_panel_value(material, 'trlau_ui_scroll_speed', raw_scroll_speed if is_animated else 0.0)
        except Exception:
            pass
        material_name = make_material_name_from_tpageid(tpageid_u32)
        try:
            material.name = material_name
        except Exception:
            pass

        self._mesh_builder._setup_material_nodes(material, image, tpage_flags, use_vertex_colors=use_vertex_colors, reflective=reflective, reflection_mode=reflection_mode)
        if bool(getattr(material, 'trlau_ui_scroll_enabled', False)):
            try:
                self._mesh_builder._apply_uv_scroll_animation_to_material(material)
            except Exception as exc:
                logger.warning('Failed to apply UV scroll animation to %s: %s', material.name, exc)
        try:
            material.use_backface_culling = bool(tpage_flags['single_sided'])
        except Exception:
            pass

    def _get_or_create_material(self, strip: LevelStrip, use_vertex_colors: bool) -> bpy.types.Material:
        key = (
            int(strip.tpageid),
            int(strip.flags),
            bool(use_vertex_colors),
            bool(getattr(strip, 'env_mapping', False)),
            bool(getattr(strip, 'eye_ref_env_mapping', False)),
            bool(strip.has_scroll_animation),
            None if strip.scroll_speed is None else round(float(strip.scroll_speed), 6),
            int(getattr(strip, 'scroll_num_tiles', 0)),
            int(getattr(strip, 'scroll_tile', 0)),
        )
        existing = self._material_cache.get(key)
        if existing is not None:
            return existing

        scroll_suffix = ''
        if strip.has_scroll_animation:
            speed_tag = 'NA' if strip.scroll_speed is None else str(int(round(float(strip.scroll_speed) * 1000.0)))
            scroll_suffix = f'_SC{speed_tag}_NT{int(getattr(strip, "scroll_num_tiles", 0))}_ST{int(getattr(strip, "scroll_tile", 0))}'
        reflection_suffix = ''
        if bool(getattr(strip, 'env_mapping', False)) and bool(getattr(strip, 'eye_ref_env_mapping', False)):
            reflection_suffix = '_REFB'
        elif bool(getattr(strip, 'env_mapping', False)):
            reflection_suffix = '_REFE'
        elif bool(getattr(strip, 'eye_ref_env_mapping', False)):
            reflection_suffix = '_REFR'
        name_suffix = f'L{int(strip.flags) & 0xFFFFFFFF:08X}_VC{1 if use_vertex_colors else 0}{reflection_suffix}{scroll_suffix}'
        material_name = make_material_name_from_tpageid(int(strip.tpageid) & 0xFFFFFFFF, name_suffix=name_suffix)
        material = bpy.data.materials.get(material_name)
        if material is not None and bool(material_uses_vertex_colors(material)) != bool(use_vertex_colors):
            material = None
        if material is None:
            material = bpy.data.materials.new(material_name)
        self._configure_level_material(material, strip, use_vertex_colors)

        self._material_cache[key] = material
        return material
