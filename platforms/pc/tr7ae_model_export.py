from __future__ import annotations

from ...core.object_utils import trlau_object_type, hmarker_index_from_name
from ...core.hinfo_properties import get_hinfo_int_prop
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bpy
import mathutils
from mathutils import Matrix, Vector

from ...core.game_utils import normalize_game_value
from ...core.log import logger
from ...core.object_utils import is_descendant_of
from ...core.material_ui import decode_tpage_flags, get_material_tpageid, get_material_draw_group, get_material_scroll_enabled, get_material_scroll_speed, get_material_env_mapping, get_material_eye_ref_env_mapping
from ...core.model_types import (
    BoneMirrorEntry,
    HBox,
    HCapsule,
    HMarker,
    HSphere,
    MFace,
    ModelData,
    MVertex,
    Segment,
    Target,
    TextureStrip,
    VirtSegment,
)
from ...core.texture_export import clear_texture_export_cache, image_to_pcd_bytes
from ..common.drm_container import DRMContainerParser, write_drm_from_standalone_directory
from .tr7ae_object import TRObjectParser, ObjectModelReference
from .tr7ae_nextgen_model_export import export_pc_nextgen_model_section, find_pc_nextgen_meshes, is_pc_nextgen_mesh_object
from .tr8_model_export import export_pc_underworld_model_section
from .tr7ae_cloth import (
    RD_SETUP_LIST_FIELD_OFFSET,
    ClothJointMap,
    ClothPoint,
    ClothSetup,
    FDCapsuleRule,
    FDDistanceRule,
    FDPlaneRule,
    FDPointMoveRule,
    FDSphereCollideRule,
    pack_cloth_section,
    read_standalone_section,
    write_standalone_section,
)
from ..common.section import SectionContextCache, SectionContext, resolve_pointer


_RELOCATION_POINTER = 0
_PC_MODEL_VERSION_MAGIC = 79823955
_MODEL_HEADER_SIZE_PC = 0x1A0
_SEGMENT_SIZE_PC = 64
_VIRT_SEGMENT_SIZE_PC = 64
_VERTEX_SIZE_PC = 16
_FACE_SIZE_PC = 8
_MODEL_ROOT_RE = re.compile(r'^(?P<prefix>.+?)_(?P<index>\d+)_Model(?:\.\d+)?$')
_BONE_RE = re.compile(r'^bone_(\d+)$')
_SECTION_NAME_RE = re.compile(r'^(?P<index>\d+)_(?P<id>[0-9a-fA-F]+)\.(?P<ext>[^.]+)$')
_EXPORT_COMPONENT_TYPES = {'HMarker', 'HSphere', 'HBox', 'HCapsule', 'Target', 'Bound', 'Collision', 'ClothCollision', 'ClothPoint', 'ClothCapsuleEndpoint', 'ClothPlaneRule', 'ClothPlaneRulePlane', 'ClothPlaneRuleSelector', 'ClothPlaneRules'}
_EXPORT_COMPONENT_ROOT_SUFFIXES = ('_HInfo', '_Targets', '_Bounds')
_EXPORT_COMPONENT_GROUP_SUFFIXES = ('_HMarkers', '_HSpheres', '_HBoxes', '_HCapsules')

_MARKUP_FLAG_PERCH = 262144
_MARKUP_FLAG_WATER = 2147483648
_MARKUP_BBOX_FLAGS = _MARKUP_FLAG_PERCH | _MARKUP_FLAG_WATER
_TEXTURE_STRIP_MAX_I16_COUNT = 32767
_TEXTURE_STRIP_TRIANGLE_CHUNK_COUNT = 32766
_MODEL_MAX_GAME_VERTICES = 21845
_MODEL_FLAT_SHADED_MAX_VIRT_SEGMENTS = 153
_MODEL_FLAT_SHADED_WEIGHT_STEPS = (0.01, 0.02, 0.05, 0.1, 0.25)
_MODEL_MAX_WEIGHTS_PER_VERTEX = 2


def _align_length(buffer: bytearray, alignment: int) -> int:
    alignment = max(1, int(alignment))
    padding = (-len(buffer)) % alignment
    if padding:
        buffer.extend(b'\x00' * padding)
    return len(buffer)


@dataclass(slots=True)
class _Relocation:
    target_section_index: int
    offset: int
    type_specific: int = 0
    relocation_type: int = _RELOCATION_POINTER


@dataclass
class _StandaloneSectionBuffer:
    section_index: int
    filename: str
    section_type: int = 0
    section_id: int = 0
    skip_flags: int = 0
    version_id: int = 0
    has_debug_info: int = 0
    resource_type: int = 0
    spec_mask: int = 0xFFFFFFFF
    data: bytearray = field(default_factory=bytearray)
    relocations: List[_Relocation] = field(default_factory=list)

    def align(self, alignment: int) -> int:
        return _align_length(self.data, alignment)

    def append(self, payload: bytes, alignment: int = 1) -> int:
        self.align(alignment)
        offset = len(self.data)
        self.data.extend(payload)
        return offset

    def reserve(self, size: int, alignment: int = 1) -> int:
        self.align(alignment)
        offset = len(self.data)
        self.data.extend(b'\x00' * int(size))
        return offset

    def pack_at(self, offset: int, fmt: str, *values) -> None:
        struct.pack_into(fmt, self.data, int(offset), *values)

    def write_pointer_at(self, offset: int, target_section_index: int, target_offset: int) -> None:
        self.pack_at(offset, '<I', int(target_offset) & 0xFFFFFFFF)
        self.relocations.append(_Relocation(int(target_section_index), int(offset)))


def _write_standalone_section_file(path: Path, section: _StandaloneSectionBuffer) -> None:
    relocs_by_offset: Dict[int, _Relocation] = {}
    for reloc in section.relocations:
        relocs_by_offset[int(reloc.offset)] = reloc
    relocs = [relocs_by_offset[offset] for offset in sorted(relocs_by_offset)]
    packed_data = (
        (int(section.has_debug_info) & 0x1)
        | ((int(section.resource_type) & 0x7F) << 1)
        | (len(relocs) << 8)
    )
    size = len(section.data)
    payload = bytearray()
    payload.extend(b'SECT')
    payload.extend(struct.pack('<iBBHIII', int(size), int(section.section_type) & 0xFF, int(section.skip_flags) & 0xFF, int(section.version_id) & 0xFFFF, packed_data, int(section.section_id) & 0xFFFFFFFF, int(section.spec_mask) & 0xFFFFFFFF))
    for reloc in relocs:
        type_and_section = ((int(reloc.target_section_index) & 0x1FFF) << 3) | (int(reloc.relocation_type) & 0x7)
        payload.extend(struct.pack('<HhI', type_and_section & 0xFFFF, int(reloc.type_specific), int(reloc.offset) & 0xFFFFFFFF))
    payload.extend(section.data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(payload))


@dataclass(slots=True)
class _ExportModelMarkupData:
    index: int
    game: str = 'legend'
    flags: int = 0
    animated_segment: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    polyline: List[Tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class _SectionList:
    path: Path
    entries: List[str]

    @classmethod
    def load(cls, directory: Path) -> '_SectionList':
        path = directory / 'sectionList.txt'
        if not path.exists():
            return cls(path=path, entries=[])
        entries = [line.strip() for line in path.read_text(encoding='utf-8', errors='ignore').splitlines()]
        return cls(path=path, entries=entries)

    def filename_for_index(self, section_index: int) -> Optional[str]:
        section_index = int(section_index)
        if 0 <= section_index < len(self.entries):
            value = str(self.entries[section_index]).strip()
            return value or None
        return None

    def ensure_index(self, section_index: int, filename: str) -> None:
        section_index = int(section_index)
        while len(self.entries) <= section_index:
            self.entries.append('')
        self.entries[section_index] = str(filename)

    def allocate(self, filename: str) -> int:
        index = len(self.entries)
        self.entries.append(str(filename))
        return index

    def used_indices(self) -> set[int]:
        used: set[int] = set()
        for idx, name in enumerate(self.entries):
            if name:
                used.add(idx)
        return used

    def write(self) -> None:
        if not self.entries:
            return
        self.path.write_text('\n'.join(self.entries) + '\n', encoding='utf-8')


@dataclass(slots=True)
class _OriginalModelLayout:
    model_section_index: int
    model_filename: str
    model_section_info: object
    pointer_targets: Dict[int, int]
    section_infos: Dict[int, object]
    section_filenames: Dict[int, str]
    hinfo_targets: Dict[int, int]
    pointer_values: Dict[int, int] = field(default_factory=dict)
    segment_list_offset: int = _MODEL_HEADER_SIZE_PC
    version: int = 8

    @classmethod
    def from_model_file(cls, filepath: Path, section_list: _SectionList) -> '_OriginalModelLayout':
        section_infos: Dict[int, object] = {}
        section_filenames: Dict[int, str] = {}
        for candidate in sorted(filepath.parent.iterdir()):
            if not candidate.is_file():
                continue
            match = _SECTION_NAME_RE.match(candidate.name)
            if match is None:
                continue
            index = int(match.group('index'))
            section_filenames[index] = candidate.name
            try:
                cache = SectionContextCache(str(candidate))
                ctx = cache.get_root_context()
                section_infos[index] = ctx.section_info
            except Exception:
                pass
            finally:
                try:
                    cache.close()
                except Exception:
                    pass

        root_index = _section_index_from_filename(filepath.name)
        section_filenames[root_index] = filepath.name
        cache = SectionContextCache(str(filepath))
        try:
            root_ctx = cache.get_root_context()
            root_info = root_ctx.section_info
            br = root_ctx.reader
            br.seek(root_ctx.data_start)
            try:
                version = br.i32()
            except Exception:
                version = 8
            try:
                br.seek(root_ctx.data_start + 12)
                segment_list_offset = int(br.u32())
            except Exception:
                segment_list_offset = _MODEL_HEADER_SIZE_PC
            pointer_targets: Dict[int, int] = {}
            pointer_values: Dict[int, int] = {}
            for field_offset in (12, 36, 44, 52, 56, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 116, 120, 128, 136):
                reloc = root_ctx.section_info.relocations_by_offset.get(field_offset)
                pointer_targets[field_offset] = int(reloc.section_index_or_type) if reloc is not None else root_index
                try:
                    br.seek(root_ctx.data_start + field_offset)
                    pointer_values[field_offset] = int(br.u32())
                except Exception:
                    pointer_values[field_offset] = 0

            hinfo_targets: Dict[int, int] = {}
            try:
                br.seek(root_ctx.data_start + 4)
                num_segments = br.i32()
                _num_virt = br.i32()
                raw = br.u32()
                seg_ctx, seg_abs = resolve_pointer(cache, root_ctx, 12, raw)
                seg_section_index = _section_index_from_filename(seg_ctx.file_name)
                pointer_targets.setdefault(12, seg_section_index)
                first_hinfo_target = None
                for segment_index in range(max(0, int(num_segments))):
                    field_local = (seg_abs - seg_ctx.data_start) + (segment_index * _SEGMENT_SIZE_PC) + (_SEGMENT_SIZE_PC - 4)
                    reloc = seg_ctx.section_info.relocations_by_offset.get(field_local)
                    target = int(reloc.section_index_or_type) if reloc is not None else seg_section_index
                    hinfo_targets[segment_index] = target
                    if first_hinfo_target is None:
                        first_hinfo_target = target
                if first_hinfo_target is not None:
                    hinfo_targets[-1] = first_hinfo_target
            except Exception as exc:
                logger.debug('Could not infer original HInfo relocation targets for %s: %s', filepath.name, exc)

            section_infos[root_index] = root_info
            return cls(
                model_section_index=root_index,
                model_filename=filepath.name,
                model_section_info=root_info,
                pointer_targets=pointer_targets,
                section_infos=section_infos,
                section_filenames=section_filenames,
                hinfo_targets=hinfo_targets,
                pointer_values=pointer_values,
                segment_list_offset=max(_MODEL_HEADER_SIZE_PC, int(segment_list_offset or 0)),
                version=int(version),
            )
        finally:
            cache.close()

    def target_for(self, field_offset: int) -> int:
        return int(self.pointer_targets.get(int(field_offset), self.model_section_index))

    def hinfo_target_for(self, segment_index: int) -> int:
        return int(self.hinfo_targets.get(int(segment_index), self.hinfo_targets.get(-1, self.target_for(12))))


def _section_index_from_filename(filename: str) -> int:
    match = _SECTION_NAME_RE.match(Path(filename).name)
    if match is not None:
        return int(match.group('index'))
    stem = Path(filename).stem
    try:
        return int(stem.split('_', 1)[0])
    except Exception:
        return 0


def _decode_signed_id(value) -> int:
    try:
        value = int(value)
    except Exception:
        return 0
    return value & 0xFFFFFFFF


def _decode_pc_uv_component(raw: int) -> float:
    try:
        return float(struct.unpack('<f', struct.pack('<I', (int(raw) & 0xFFFF) << 16))[0])
    except Exception:
        return 0.0


def _encode_pc_uv_component(value: float) -> int:
    try:
        bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        return (rounded >> 16) & 0xFFFF
    except Exception:
        return 0


def _encode_ps2_scaled_uv_component(value: float) -> int:
    try:
        scaled = int(round(float(value) * 4096.0))
    except Exception:
        scaled = 0
    return max(-32768, min(32767, int(scaled)))



def _float_color_to_byte(value: float) -> int:
    return max(0, min(255, int(round(float(value) * 255.0))))


def _float_alpha_to_trlau_byte(value: float) -> int:
    return max(0, min(255, int(round(float(value) * 128.0))))


def _normal_to_i8(value: float) -> int:
    return max(-127, min(127, int(round(float(value) * 127.0))))


def _int16(value: float) -> int:
    return max(-32768, min(32767, int(round(float(value)))))


def _iter_collection_objects(collection) -> Iterable[object]:
    return getattr(collection, 'all_objects', None) or collection.objects


def _is_descendant_of(obj, ancestor) -> bool:
    return is_descendant_of(obj, ancestor)


def _custom_prop_bool(obj, key: str) -> bool:
    if obj is None:
        return False
    try:
        return bool(obj.get(key, False))
    except Exception:
        return False


def _custom_prop_text(obj, key: str) -> str:
    if obj is None:
        return ''
    try:
        return str(obj.get(key, '') or '')
    except Exception:
        return ''


def _is_hmarker_attachment_object(obj) -> bool:
    current = getattr(obj, 'parent', None)
    while current is not None:
        if trlau_object_type(current) == 'HMarker':
            return True
        current = getattr(current, 'parent', None)
    return False


def _should_ignore_model_export_object(obj) -> bool:
    if obj is None:
        return False
    if _custom_prop_bool(obj, 'trlau_export_ignore') or _custom_prop_bool(obj, 'trlau_hmarker_attachment'):
        return True
    current = getattr(obj, 'parent', None)
    while current is not None:
        if _custom_prop_bool(current, 'trlau_export_ignore') or _custom_prop_bool(current, 'trlau_hmarker_attachment'):
            return True
        if trlau_object_type(current) == 'HMarker':
            return True
        current = getattr(current, 'parent', None)
    return False


def _is_inside_cloth_authoring_root(obj) -> bool:
    current = obj
    while current is not None:
        try:
            if trlau_object_type(current) == 'Cloth':
                return True
        except Exception:
            pass
        try:
            name = str(getattr(current, 'name', '') or '')
            if name.endswith('_Cloth'):
                return True
        except Exception:
            pass
        current = getattr(current, 'parent', None)
    return False


@dataclass(slots=True)
class _SyntheticClothRoot:
    """Transient cloth root used when bones are authored as cloth but no Cloth empty exists."""

    name: str
    parent: object
    props: Dict[str, object] = field(default_factory=dict)
    children: List[object] = field(default_factory=list)
    type: str = 'EMPTY'

    def get(self, key: str, default=None):
        return self.props.get(key, default)

class TRLAUModelExporter:
    """Export edited regular model sections back to an extracted TRLAU folder."""

    def __init__(self, *, debug: bool = False):
        self.debug = bool(debug)
        self.warnings: List[str] = []
        self._export_scene_matrix_overrides: Dict[int, Matrix] | None = None

    def _add_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        logger.warning(message)

    def _snapshot_export_component_world_matrices(self, collection) -> Dict[int, Matrix]:
        """Capture authoring helper transforms before export forces armatures to REST.

        Model geometry is exported from rest pose, but helper components are
        authored in the visible scene.  Cloth collision helpers and HInfo
        objects can have Child Of constraints; if we read their matrix_world
        after switching armatures to REST, their positions are silently
        recomputed.  Store their pre-export world matrices so export writes the
        positions the user actually has in the scene.
        """
        result: Dict[int, Matrix] = {}
        if collection is None:
            return result
        helper_types = {
            'HMarker', 'HSphere', 'HBox', 'HCapsule', 'Target',
            'ClothCollision', 'ClothPoint', 'ClothPinCollision',
            'ClothCapsuleEndpoint', 'ClothCollisionRule',
            'ClothPlaneRule', 'ClothPlaneRulePlane', 'ClothPlaneRuleSelector',
        }
        for obj in list(_iter_collection_objects(collection)):
            try:
                obj_type = trlau_object_type(obj)
            except Exception:
                obj_type = ''
            try:
                name = str(getattr(obj, 'name', '') or '')
            except Exception:
                name = ''
            if obj_type not in helper_types and '_Cloth_' not in name and '_HMarker_' not in name and '_HSphere_' not in name and '_HBox_' not in name and '_HCapsule_' not in name:
                continue
            try:
                result[id(obj)] = obj.matrix_world.copy()
            except Exception:
                pass
        return result

    def _export_object_world_matrix(self, obj) -> Matrix:
        overrides = getattr(self, '_export_scene_matrix_overrides', None)
        if overrides is not None:
            matrix = overrides.get(id(obj))
            if matrix is not None:
                return matrix.copy()
        try:
            return obj.matrix_world.copy()
        except Exception:
            return Matrix.Identity(4)

    def _export_object_world_translation(self, obj) -> Vector:
        try:
            return self._export_object_world_matrix(obj).translation.copy()
        except Exception:
            return Vector((0.0, 0.0, 0.0))

    @staticmethod
    def collection_has_models(collection) -> bool:
        if collection is None:
            return False
        for obj in _iter_collection_objects(collection):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) == 'EMPTY' and bool(getattr(obj, 'trlau_is_model_empty', False)):
                return True
        return False


    @staticmethod
    def _is_cloth_root(obj) -> bool:
        return obj is not None and getattr(obj, 'type', None) == 'EMPTY' and trlau_object_type(obj) == 'Cloth'

    @staticmethod
    def _bone_has_cloth_authoring(bone, pose_bone=None) -> bool:
        for candidate in (pose_bone, bone):
            if candidate is None:
                continue
            try:
                if 'trlau_cloth_pinned' in candidate:
                    return True
                if bool(candidate.get('trlau_cloth_enabled', False)):
                    return True
            except Exception:
                pass
        return False

    @classmethod
    def _armature_has_authored_cloth_bones(cls, arm_obj) -> bool:
        if getattr(arm_obj, 'type', None) != 'ARMATURE':
            return False
        data_bones = getattr(getattr(arm_obj, 'data', None), 'bones', {})
        pose_bones = getattr(getattr(arm_obj, 'pose', None), 'bones', {})
        for bone in data_bones:
            bone_name = str(getattr(bone, 'name', '') or '')
            pose_bone = pose_bones.get(bone_name) if pose_bones is not None else None
            if cls._bone_has_cloth_authoring(bone, pose_bone):
                return True
        return False

    @staticmethod
    def _auto_cloth_root_name_for_armature(arm_obj) -> str:
        name = str(getattr(arm_obj, 'name', '') or 'TRLAU')
        for suffix in ('_Armature', '_Skeleton'):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return f'{name or "TRLAU"}_Cloth'

    def _make_synthetic_cloth_root(self, arm_obj) -> _SyntheticClothRoot:
        return _SyntheticClothRoot(
            name=self._auto_cloth_root_name_for_armature(arm_obj),
            parent=arm_obj,
        )

    def _cloth_roots_for_collection(self, collection) -> List[object]:
        roots: List[object] = []
        if collection is None:
            return roots
        seen_names: set[str] = set()
        rooted_armatures: set[str] = set()
        objects = list(_iter_collection_objects(collection))
        for obj in objects:
            if _should_ignore_model_export_object(obj):
                continue
            if not self._is_cloth_root(obj):
                continue
            name = str(getattr(obj, 'name', '') or '')
            if name in seen_names:
                continue
            seen_names.add(name)
            roots.append(obj)
            parent = getattr(obj, 'parent', None)
            if getattr(parent, 'type', None) == 'ARMATURE':
                rooted_armatures.add(str(getattr(parent, 'name', '') or ''))

        for obj in objects:
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'ARMATURE':
                continue
            arm_name = str(getattr(obj, 'name', '') or '')
            if arm_name in rooted_armatures:
                continue
            if not self._armature_has_authored_cloth_bones(obj):
                continue
            root = self._make_synthetic_cloth_root(obj)
            root_name = str(getattr(root, 'name', '') or '')
            if root_name in seen_names:
                continue
            seen_names.add(root_name)
            rooted_armatures.add(arm_name)
            roots.append(root)

        roots.sort(key=lambda item: str(getattr(item, 'name', '') or ''))
        return roots

    @staticmethod
    def _get_idprop_int(obj, key: str, default: int = 0) -> int:
        try:
            value = obj.get(key, default)
            if isinstance(value, str):
                return int(value.strip(), 0)
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _distance_between_objects(a, b) -> float:
        try:
            return float((a.location - b.location).length)
        except Exception:
            return 0.0

    @staticmethod
    def _cloth_bone_prop(bone, pose_bone, key: str, default=None):
        for candidate in (pose_bone, bone):
            try:
                if candidate is not None and key in candidate:
                    return candidate.get(key)
            except Exception:
                pass
        return default

    @staticmethod
    def _cloth_point_flag_value(flags: int) -> int:
        return 1 if int(flags) in {1, 5} else 0

    @staticmethod
    def _cloth_distance_sq(a: Vector, b: Vector) -> float:
        try:
            return float((a - b).length_squared)
        except Exception:
            return 0.0

    def _build_cloth_setup_from_root(self, root) -> ClothSetup:
        """Build a ClothSetup from the scene authoring state.

        The authoring contract is intentionally small: bones are either pinned
        or unpinned, and collision points are sphere objects under the Cloth
        empty. Everything that is specific to the Forward Dynamics setup
        (point indices, joint maps, distance constraints, and move rules) is
        regenerated here from the armature graph and sphere placement.
        """
        setup = ClothSetup()
        setup.gravity = 10.0
        setup.drag = 0.1
        setup.windResponse = 0.1

        arm_obj = getattr(root, 'parent', None) if getattr(getattr(root, 'parent', None), 'type', None) == 'ARMATURE' else None
        point_entries: list[dict] = []
        key_to_index: dict[str, int] = {}
        indexed_paths: list[list[int]] = []
        path_axis_by_root_index: dict[int, int] = {}
        path_names_by_root_index: dict[int, list[str]] = {}

        # The exporter now generates ClothSetup tables from the visible scene
        # state instead of consulting imported setup JSON/offset metadata stored on
        # the Cloth empty.  Imported per-point collision helpers resolve by name
        # and by position, and pinned-only points are represented by pinned bones
        # with trlau_cloth_enabled=False.
        imported_points_hint: list[dict] = []
        imported_point_index_by_bone_segment: dict[int, int] = {}
        imported_collision_point_indices: set[int] = set()
        imported_point_only_pinned_segments: set[int] = set()

        def imported_point_index_for_bone_segment(segment: int) -> Optional[int]:
            try:
                value = imported_point_index_by_bone_segment.get(int(segment))
                return int(value) if value is not None else None
            except Exception:
                return None

        def is_imported_collision_point_index(index: Optional[int]) -> bool:
            try:
                return index is not None and int(index) in imported_collision_point_indices
            except Exception:
                return False

        def is_imported_point_only_pinned_segment(segment: int) -> bool:
            try:
                return int(segment) in imported_point_only_pinned_segments
            except Exception:
                return False

        def bone_segment_from_name(name: str, default: int = 0) -> int:
            match = _BONE_RE.match(str(name or ''))
            if match:
                return int(match.group(1))
            return int(default)

        def prop_is_set(bone, pose_bone, key: str) -> bool:
            for candidate in (pose_bone, bone):
                try:
                    if candidate is not None and key in candidate:
                        return True
                except Exception:
                    pass
            return False

        def cloth_flag_for_bone(bone, pose_bone) -> Optional[int]:
            if prop_is_set(bone, pose_bone, 'trlau_cloth_pinned'):
                return 5 if bool(self._cloth_bone_prop(bone, pose_bone, 'trlau_cloth_pinned', False)) else 4
            if bool(self._cloth_bone_prop(bone, pose_bone, 'trlau_cloth_enabled', False)):
                return 4
            return None

        def point_flag_value(index: int) -> int:
            try:
                return self._cloth_point_flag_value(setup.points[int(index)].flags)
            except Exception:
                return 0

        def dominant_axis_code(vector: Vector) -> int:
            try:
                x, y, z = abs(float(vector.x)), abs(float(vector.y)), abs(float(vector.z))
            except Exception:
                return 5
            if x >= y and x >= z:
                return 3
            if y >= x and y >= z:
                return 4
            return 5

        def distance_sq(a: Vector, b: Vector) -> float:
            return max(0.0, self._cloth_distance_sq(a, b))

        def data_bone_by_segment(segment: int):
            if arm_obj is None:
                return None
            return getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(f'bone_{int(segment)}')

        def bone_position_by_name(name: str) -> Vector:
            if arm_obj is None:
                return Vector((0.0, 0.0, 0.0))
            bone = getattr(getattr(arm_obj, 'data', None), 'bones', {}).get(name)
            if bone is None:
                return Vector((0.0, 0.0, 0.0))
            try:
                return Vector(tuple(float(v) for v in bone.head_local))
            except Exception:
                return Vector((0.0, 0.0, 0.0))

        def segment_is_descendant(segment: int, owner_segment: int) -> bool:
            if arm_obj is None:
                return False
            bone = data_bone_by_segment(segment)
            owner_name = f'bone_{int(owner_segment)}'
            seen: set[str] = set()
            while bone is not None:
                name = str(getattr(bone, 'name', '') or '')
                if name == owner_name:
                    return True
                if name in seen:
                    return False
                seen.add(name)
                try:
                    bone = getattr(bone, 'parent', None)
                except Exception:
                    return False
            return False

        cloth_by_name: dict[str, dict] = {}
        child_names: dict[str, list[str]] = defaultdict(list)
        chain_roots: list[str] = []
        chain_paths: list[list[str]] = []
        ordered_chain_paths: list[list[str]] = []
        primary_chain_root_name: str = ''

        if arm_obj is not None:
            data_bones = getattr(getattr(arm_obj, 'data', None), 'bones', {})
            pose_bones = getattr(getattr(arm_obj, 'pose', None), 'bones', {})
            for default_index, bone in enumerate(data_bones):
                bone_name = str(getattr(bone, 'name', '') or '')
                pose_bone = pose_bones.get(bone_name) if pose_bones is not None else None
                flag = cloth_flag_for_bone(bone, pose_bone)
                if flag is None:
                    continue
                segment = bone_segment_from_name(bone_name, default_index)
                try:
                    parent_bone = getattr(bone, 'parent', None)
                    armature_parent_name = str(getattr(parent_bone, 'name', '') or '')
                except Exception:
                    armature_parent_name = ''
                imported_point_index = -1
                imported_joint_order = 0
                imported_up_to = 0
                cloth_by_name[bone_name] = {
                    'key': f'bone:{segment}',
                    'source': 'bone',
                    'bone_name': bone_name,
                    'armature_parent_name': armature_parent_name,
                    'segment': int(segment),
                    'flags': int(flag),
                    'jointOrder': int(imported_joint_order),
                    'upTo': int(imported_up_to),
                    'imported_point_index': int(imported_point_index) if imported_point_index >= 0 else None,
                    'position': bone_position_by_name(bone_name),
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'children': [],
                    'point_only_pinned': bool(
                        int(flag) == 5
                        and (
                            is_imported_point_only_pinned_segment(int(segment))
                            or not bool(self._cloth_bone_prop(bone, pose_bone, 'trlau_cloth_enabled', True))
                        )
                    ),
                }

            for bone_name, entry in cloth_by_name.items():
                bone = data_bones.get(bone_name) if data_bones is not None else None
                parent_name = None
                try:
                    parent = getattr(bone, 'parent', None)
                    parent_candidate = str(getattr(parent, 'name', '') or '') if parent is not None else ''
                    parent_entry = cloth_by_name.get(parent_candidate)
                    if (
                        parent_candidate in cloth_by_name
                        and int(entry.get('flags', 4)) != 5
                        and not bool(entry.get('point_only_pinned', False))
                        and not bool((parent_entry or {}).get('point_only_pinned', False))
                    ):
                        parent_name = parent_candidate
                except Exception:
                    parent_name = None
                if parent_name:
                    child_names[parent_name].append(bone_name)
                else:
                    chain_roots.append(bone_name)

            def sort_key(name: str) -> tuple[int, str]:
                return (int(cloth_by_name.get(name, {}).get('segment', 0)), str(name))

            for names in child_names.values():
                names.sort(key=sort_key)
            chain_roots.sort(key=sort_key)

            def walk(name: str, prefix: list[str]) -> None:
                current = prefix + [name]
                children = child_names.get(name, [])
                if not children:
                    chain_paths.append(current)
                    return
                for child_name in children:
                    walk(child_name, current)

            for root_name in chain_roots:
                walk(root_name, [])

            def chain_has_movable(path_names: list[str]) -> bool:
                return any(int(cloth_by_name.get(name, {}).get('flags', 0)) == 4 for name in path_names)

            def chain_root_is_pinned(path_names: list[str]) -> bool:
                return bool(path_names) and int(cloth_by_name.get(path_names[0], {}).get('flags', 0)) == 5

            def chain_order_key(path_names: list[str]) -> tuple[int, int, int, float, int, str]:
                root_name = path_names[0] if path_names else ''
                root_entry = cloth_by_name.get(root_name, {})
                root_segment = int(root_entry.get('segment', 0))
                parent_name = str(root_entry.get('armature_parent_name', '') or '')
                parent_segment = bone_segment_from_name(parent_name, 0) if parent_name else 0
                try:
                    delta = bone_position_by_name(root_name) - (bone_position_by_name(parent_name) if parent_name else Vector((0.0, 0.0, 0.0)))
                    x = float(delta.x)
                    y = float(delta.y)
                except Exception:
                    x = 0.0
                    y = 0.0
                return (
                    int(parent_segment),
                    0 if y < 0.0 else 1,
                    0 if x >= 0.0 else 1,
                    -abs(x),
                    int(root_segment),
                    str(root_name),
                )

            movable_chain_paths = [list(path_names) for path_names in chain_paths if chain_has_movable(path_names)]
            ordered_chain_paths = sorted(movable_chain_paths, key=chain_order_key)
            primary_chain_root_name = str(ordered_chain_paths[0][0]) if ordered_chain_paths else ''

            ordered_names: list[str] = []
            seen_bones: set[str] = set()

            def append_bone_name(bone_name: str) -> None:
                if not bone_name or bone_name in seen_bones or bone_name not in cloth_by_name:
                    return
                seen_bones.add(bone_name)
                entry = dict(cloth_by_name[bone_name])
                imported_index = entry.get('imported_point_index', None)
                if imported_index is None:
                    imported_index = imported_point_index_for_bone_segment(int(entry.get('segment', -1)))
                if imported_index is not None:
                    entry['imported_point_index'] = int(imported_index)
                    entry['old_index'] = int(imported_index)
                else:
                    entry['imported_point_index'] = None
                    entry['old_index'] = len(point_entries)
                point_entries.append(entry)

            reversed_paths = list(reversed(ordered_chain_paths))
            pinned_chain_roots = {
                str(path_names[0])
                for path_names in ordered_chain_paths
                if chain_root_is_pinned(path_names)
            }

            for path_names in reversed_paths:
                movable_names = [
                    name for name in path_names
                    if int(cloth_by_name.get(name, {}).get('flags', 0)) == 4
                ]
                for bone_name in reversed(movable_names):
                    append_bone_name(bone_name)

            isolated_pins = [
                name for name, entry in cloth_by_name.items()
                if int(entry.get('flags', 0)) == 5 and (name not in pinned_chain_roots or bool(entry.get('point_only_pinned', False)))
            ]
            isolated_pins.sort(key=lambda name: (-int(cloth_by_name.get(name, {}).get('segment', 0)), str(name)))
            for bone_name in isolated_pins:
                append_bone_name(bone_name)

            for path_names in reversed_paths:
                root_name = str(path_names[0]) if path_names else ''
                if root_name == primary_chain_root_name:
                    continue
                if chain_root_is_pinned(path_names):
                    append_bone_name(root_name)

            for bone_name in sorted(cloth_by_name, key=sort_key):
                if bone_name != primary_chain_root_name:
                    append_bone_name(bone_name)

        def nearest_bone_segment(world_position: Vector) -> int:
            if arm_obj is None:
                return 0
            best_segment = 0
            best_dist = None
            data_bones = getattr(getattr(arm_obj, 'data', None), 'bones', {})
            for default_index, bone in enumerate(data_bones):
                try:
                    segment = bone_segment_from_name(getattr(bone, 'name', ''), default_index)
                    bone_world = arm_obj.matrix_world @ bone.head_local
                    dist = float((bone_world - world_position).length_squared)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_segment = int(segment)
                except Exception:
                    continue
            return int(best_segment)

        def collision_owner_segment(child) -> int:
            if arm_obj is not None:
                for constraint in list(getattr(child, 'constraints', []) or []):
                    try:
                        if getattr(constraint, 'type', '') == 'CHILD_OF' and getattr(constraint, 'target', None) == arm_obj:
                            subtarget = str(getattr(constraint, 'subtarget', '') or '')
                            match = _BONE_RE.match(subtarget)
                            if match:
                                return int(match.group(1))
                    except Exception:
                        pass
            try:
                return nearest_bone_segment(self._export_object_world_translation(child))
            except Exception:
                return 0

        def cloth_collision_name_index(child) -> Optional[int]:
            child_name = str(getattr(child, 'name', '') or '')
            root_name = str(getattr(root, 'name', '') or '')
            bases = [root_name]
            if root_name.endswith('_Cloth'):
                bases.append(root_name[:-len('_Cloth')])
            for base in bases:
                if not base:
                    continue
                prefix = f'{base}_Cloth_Collision_'
                if not child_name.startswith(prefix):
                    continue
                suffix = child_name[len(prefix):]
                index_text = suffix.split('.', 1)[0]
                if index_text.isdigit():
                    return int(index_text)
            return None


        def cloth_pin_collision_name_index(child) -> Optional[int]:
            child_name = str(getattr(child, 'name', '') or '')
            root_name = str(getattr(root, 'name', '') or '')
            bases = [root_name]
            if root_name.endswith('_Cloth'):
                bases.append(root_name[:-len('_Cloth')])
            for base in bases:
                if not base:
                    continue
                prefix = f'{base}_Cloth_PinCollision_'
                if not child_name.startswith(prefix):
                    continue
                suffix = child_name[len(prefix):]
                index_text = suffix.split('.', 1)[0]
                if index_text.isdigit():
                    return int(index_text)
            return None

        def is_cloth_pin_collision_child(child) -> bool:
            child_type = trlau_object_type(child)
            if child_type == 'ClothPinCollision':
                return True
            return cloth_pin_collision_name_index(child) is not None

        def cloth_capsule_endpoint_info(child) -> Optional[tuple[int, str]]:
            child_type = trlau_object_type(child)
            name = str(getattr(child, 'name', '') or '')
            root_name = str(getattr(root, 'name', '') or '')
            if '.' in name:
                base_name = name.split('.', 1)[0]
            else:
                base_name = name

            bases = [root_name]
            if root_name.endswith('_Cloth'):
                bases.append(root_name[:-len('_Cloth')])
            for base in bases:
                if not base:
                    continue
                prefix = f'{base}_Cloth_Capsule_'
                if not base_name.startswith(prefix):
                    continue
                suffix = base_name[len(prefix):]
                parts = suffix.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    endpoint = parts[1].upper()
                    if endpoint in {'A', 'B'}:
                        return int(parts[0]), endpoint
            return None

        def is_cloth_collision_child(child) -> bool:
            child_type = trlau_object_type(child)
            if child_type in {'ClothCollision', 'ClothPoint'}:
                return True
            return cloth_collision_name_index(child) is not None

        def is_cloth_collision_rule_child(child) -> bool:
            child_type = trlau_object_type(child)
            if child_type == 'ClothCollisionRule':
                return True
            name = str(getattr(child, 'name', '') or '')
            root_name = str(getattr(root, 'name', '') or '')
            bases = [root_name]
            if root_name.endswith('_Cloth'):
                bases.append(root_name[:-len('_Cloth')])
            for base in bases:
                if base and name.startswith(f'{base}_Cloth_CollisionRule_'):
                    return True
            return False

        def is_cloth_collision_rules_group(child) -> bool:
            try:
                if trlau_object_type(child) == 'ClothCollisionRules':
                    return True
            except Exception:
                pass
            try:
                return '_Cloth_CollisionRules' in str(getattr(child, 'name', '') or '')
            except Exception:
                return False

        def is_cloth_plane_rule_child(child) -> bool:
            try:
                child_type = trlau_object_type(child)
                if child_type == 'ClothPlaneRule':
                    return True
                if child_type in {'ClothPlaneRulePlane', 'ClothPlaneRuleSelector'}:
                    return False
            except Exception:
                child_type = ''
            name = str(getattr(child, 'name', '') or '')
            if name.endswith('_Selector') or '_Selector.' in name:
                return False
            root_name = str(getattr(root, 'name', '') or '')
            bases = [root_name]
            if root_name.endswith('_Cloth'):
                bases.append(root_name[:-len('_Cloth')])
            for base in bases:
                if base and name.startswith(f'{base}_Cloth_PlaneRule_'):
                    return True
            return False

        def is_cloth_plane_rules_group(child) -> bool:
            try:
                if trlau_object_type(child) == 'ClothPlaneRules':
                    return True
            except Exception:
                pass
            try:
                return '_Cloth_PlaneRules' in str(getattr(child, 'name', '') or '')
            except Exception:
                return False

        def iter_object_descendants(obj):
            for descendant in list(getattr(obj, 'children', []) or []):
                yield descendant
                yield from iter_object_descendants(descendant)

        def iter_cloth_collision_rule_children():
            for child in list(getattr(root, 'children', []) or []):
                if is_cloth_collision_rule_child(child):
                    yield child
                elif is_cloth_collision_rules_group(child):
                    for descendant in iter_object_descendants(child):
                        if is_cloth_collision_rule_child(descendant):
                            yield descendant

        def iter_cloth_plane_rule_children():
            for child in list(getattr(root, 'children', []) or []):
                if is_cloth_plane_rule_child(child):
                    yield child
                elif is_cloth_plane_rules_group(child):
                    for descendant in iter_object_descendants(child):
                        if is_cloth_plane_rule_child(descendant):
                            yield descendant

        def child_point_index_prop(child) -> Optional[int]:
            return cloth_collision_name_index(child)

        def collision_rule_name_info(child) -> tuple[int | None, int | None, int | None, int | None]:
            try:
                name = str(getattr(child, 'name', '') or '')
                marker = '_Cloth_CollisionRule_'
                if marker not in name:
                    return None, None, None, None
                suffix = name.split(marker, 1)[1].split('.', 1)[0]
                parts = suffix.split('_')
                order = int(parts[0]) if parts and parts[0].isdigit() else None
                target = None
                bone_segment = None
                collision = None
                for part in parts[1:]:
                    if len(part) >= 2 and part[0].upper() == 'P' and part[1:].isdigit():
                        target = int(part[1:])
                    elif len(part) >= 2 and part[0].upper() == 'B' and part[1:].isdigit():
                        bone_segment = int(part[1:])
                    elif len(part) >= 2 and part[0].upper() == 'C' and part[1:].isdigit():
                        collision = int(part[1:])
                return order, target, bone_segment, collision
            except Exception:
                return None, None, None, None

        def collision_rule_index(child, default: int = 0) -> int:
            order, _target, _bone_segment, _collision = collision_rule_name_info(child)
            if order is not None:
                return int(order)
            return int(default)

        def plane_rule_name_info(child) -> tuple[int | None, int | None, int | None]:
            try:
                name = str(getattr(child, 'name', '') or '')
                marker = '_Cloth_PlaneRule_'
                if marker not in name:
                    return None, None, None
                suffix = name.split(marker, 1)[1].split('.', 1)[0]
                parts = suffix.split('_')
                order = int(parts[0]) if parts and parts[0].isdigit() else None
                markers = []
                for part in parts[1:]:
                    if len(part) >= 2 and part[0].upper() == 'M' and part[1:].isdigit():
                        markers.append(int(part[1:]))
                marker_a = markers[0] if len(markers) > 0 else None
                marker_b = markers[1] if len(markers) > 1 else None
                return order, marker_a, marker_b
            except Exception:
                return None, None, None

        def plane_rule_index(child, default: int = 0) -> int:
            try:
                if 'trlau_cloth_plane_rule_index' in child:
                    return int(child.get('trlau_cloth_plane_rule_index'))
            except Exception:
                pass
            order, _marker_a, _marker_b = plane_rule_name_info(child)
            if order is not None:
                return int(order)
            return int(default)

        def _idprop_int_list(obj, key: str) -> list[int]:
            try:
                value = obj.get(key, [])
            except Exception:
                value = []
            if value is None:
                return []
            if isinstance(value, str):
                result = []
                for part in value.replace(';', ',').split(','):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        result.append(int(part))
                    except Exception:
                        pass
                return result
            try:
                return [int(item) for item in list(value)]
            except Exception:
                try:
                    return [int(value)]
                except Exception:
                    return []

        def collision_rule_radius(child, default: float = 0.0) -> float:
            try:
                values = tuple(float(v) for v in child.scale)
                visible_radius = max(abs(v) for v in values) if values else float(default)
                if visible_radius > 0.0:
                    # ClothCollisionRule transform scale is the functional FD radius.
                    # Empty Display Size is viewport-only and is intentionally ignored.
                    return float(visible_radius)
            except Exception:
                pass
            return float(default)

        def authored_collision_radius(child, default: float = 1.0) -> float:
            try:
                scale_values = tuple(float(v) for v in child.scale)
                radius = max(abs(v) for v in scale_values) if scale_values else float(default)
                if radius > 0.0:
                    return float(radius)
            except Exception:
                pass
            return float(default)


        capsule_endpoint_children: list[tuple[int, str, int, object]] = []
        for raw_child_index, child in enumerate(list(getattr(root, 'children', []) or [])):
            if getattr(child, 'type', None) not in {'MESH', 'EMPTY'}:
                continue
            try:
                if not bool(child.get('trlau_cloth_enabled', True)):
                    continue
            except Exception:
                pass
            endpoint_info = cloth_capsule_endpoint_info(child)
            if endpoint_info is None:
                continue
            capsule_index, endpoint_name = endpoint_info
            capsule_endpoint_children.append((int(capsule_index), str(endpoint_name), int(raw_child_index), child))

        def capsule_endpoint_sort_key(item: tuple[int, str, int, object]) -> tuple[int, int, int, str]:
            capsule_index, endpoint_name, raw_child_index, child = item
            return (int(capsule_index), 0 if endpoint_name == 'A' else 1, int(raw_child_index), str(getattr(child, 'name', '') or ''))

        capsule_endpoint_groups: dict[int, dict[str, object]] = defaultdict(dict)
        for capsule_index, endpoint_name, _raw_child_index, child in sorted(capsule_endpoint_children, key=capsule_endpoint_sort_key):
            capsule_endpoint_groups[int(capsule_index)].setdefault(str(endpoint_name), child)

        capsule_rule_refs: list[tuple[str, str, float]] = []
        for capsule_index in sorted(capsule_endpoint_groups):
            endpoints = capsule_endpoint_groups.get(capsule_index, {})
            endpoint_a = endpoints.get('A')
            endpoint_b = endpoints.get('B')
            if endpoint_a is None or endpoint_b is None:
                continue
            radius = max(authored_collision_radius(endpoint_a, 0.0), authored_collision_radius(endpoint_b, 0.0), 0.0)
            if radius <= 0.0:
                radius = 1.0
            endpoint_keys: dict[str, str] = {}
            for endpoint_name, endpoint_child in (('A', endpoint_a), ('B', endpoint_b)):
                segment = collision_owner_segment(endpoint_child)
                key = f'capsule:{int(capsule_index)}:{endpoint_name}:{segment}:{getattr(endpoint_child, "name", "")}'
                local_offset = Vector((0.0, 0.0, 0.0))
                position = Vector((0.0, 0.0, 0.0))
                if arm_obj is not None:
                    bone = data_bone_by_segment(segment)
                    if bone is not None:
                        try:
                            local_offset, position = self._point_relative_to_segment_origin(self._export_object_world_translation(endpoint_child), arm_obj, segment)
                        except Exception:
                            local_offset = Vector((0.0, 0.0, 0.0))
                    else:
                        try:
                            position = arm_obj.matrix_world.inverted() @ self._export_object_world_translation(endpoint_child)
                            local_offset = Vector(tuple(position))
                        except Exception:
                            pass
                else:
                    try:
                        position = self._export_object_world_translation(endpoint_child)
                        local_offset = Vector(tuple(position))
                    except Exception:
                        pass
                point_entries.append({
                    'key': key,
                    'source': 'capsule_endpoint',
                    'old_index': len(point_entries),
                    'segment': int(segment),
                    'flags': 1,
                    'jointOrder': 0,
                    'upTo': 3,
                    'position': position,
                    'x': float(local_offset.x),
                    'y': float(local_offset.y),
                    'z': float(local_offset.z),
                    'radius': float(radius),
                    'generate_rules': False,
                })
                endpoint_keys[endpoint_name] = key
            if 'A' in endpoint_keys and 'B' in endpoint_keys:
                capsule_rule_refs.append((endpoint_keys['A'], endpoint_keys['B'], float(radius)))

        collision_children: list[tuple[int, object]] = []
        for raw_child_index, child in enumerate(list(getattr(root, 'children', []) or [])):
            if not is_cloth_collision_child(child):
                continue
            if getattr(child, 'type', None) not in {'MESH', 'EMPTY'}:
                continue
            try:
                if not bool(child.get('trlau_cloth_enabled', True)):
                    continue
            except Exception:
                pass
            collision_children.append((int(raw_child_index), child))

        def collision_child_sort_key(item: tuple[int, object]) -> tuple[int, int, int, str]:
            raw_child_index, child = item
            name_index = cloth_collision_name_index(child)
            if name_index is not None:
                return (0, int(name_index), int(raw_child_index), str(getattr(child, 'name', '') or ''))
            return (1, int(raw_child_index), int(raw_child_index), str(getattr(child, 'name', '') or ''))

        collision_key_by_point_index: dict[int, str] = {}
        collision_key_by_name_index: dict[int, str] = {}
        collision_key_by_name: dict[str, str] = {}
        for child_index, (_raw_child_index, child) in enumerate(sorted(collision_children, key=collision_child_sort_key)):
            segment = collision_owner_segment(child)
            helper_name_index = cloth_collision_name_index(child)
            possible_imported_point_index = child_point_index_prop(child)
            imported_point_index = int(possible_imported_point_index) if is_imported_collision_point_index(possible_imported_point_index) else None
            if imported_point_index is not None:
                key = f'collision_point:{int(imported_point_index)}:{segment}:{getattr(child, "name", "")}'
            else:
                key_index = int(helper_name_index) if helper_name_index is not None else int(child_index)
                key = f'collision:{key_index}:{segment}:{getattr(child, "name", "")}'
            local_offset = Vector((0.0, 0.0, 0.0))
            position = Vector((0.0, 0.0, 0.0))
            if arm_obj is not None:
                bone = data_bone_by_segment(segment)
                if bone is not None:
                    try:
                        local_offset, position = self._point_relative_to_segment_origin(self._export_object_world_translation(child), arm_obj, segment)
                    except Exception:
                        local_offset = Vector((0.0, 0.0, 0.0))
                else:
                    try:
                        position = arm_obj.matrix_world.inverted() @ self._export_object_world_translation(child)
                        local_offset = Vector(tuple(position))
                    except Exception:
                        pass
            else:
                try:
                    position = self._export_object_world_translation(child)
                    local_offset = Vector(tuple(position))
                except Exception:
                    pass
            try:
                scale_values = tuple(float(v) for v in child.scale)
                radius = max(abs(v) for v in scale_values) if scale_values else 1.0
            except Exception:
                radius = 1.0
            point_entries.append({
                'key': key,
                'source': 'collision',
                'old_index': int(imported_point_index) if imported_point_index is not None else len(point_entries),
                'imported_point_index': int(imported_point_index) if imported_point_index is not None else None,
                'segment': int(segment),
                'flags': 1,
                'jointOrder': 0,
                'upTo': 3,
                'position': position,
                'x': float(local_offset.x),
                'y': float(local_offset.y),
                'z': float(local_offset.z),
                'radius': max(0.0, float(radius)),
                'generate_rules': True,
            })
            if imported_point_index is not None:
                collision_key_by_point_index[int(imported_point_index)] = str(key)
            if helper_name_index is not None:
                collision_key_by_name_index[int(helper_name_index)] = str(key)
            collision_key_by_name[str(getattr(child, 'name', '') or '')] = str(key)


        # Imported setups can use a pinned ClothPoint as a collision anchor
        # (flags=5, but used by min-distance rules like a fixed collision point).
        # Represent that with a visible ClothPinCollision helper and resolve it
        # back to the existing pinned bone point here, instead of creating a new
        # flags=1 collision point or relying on the removed setup JSON.
        for child in list(getattr(root, 'children', []) or []):
            if getattr(child, 'type', None) not in {'MESH', 'EMPTY'}:
                continue
            if not is_cloth_pin_collision_child(child):
                continue
            name_index = cloth_pin_collision_name_index(child)
            segment = collision_owner_segment(child)
            key = f'bone:{int(segment)}'
            if name_index is not None:
                collision_key_by_point_index[int(name_index)] = str(key)
                collision_key_by_name_index[int(name_index)] = str(key)
            collision_key_by_name[str(getattr(child, 'name', '') or '')] = str(key)

        collision_rule_children: list[tuple[int, int, object]] = []
        for raw_child_index, child in enumerate(list(iter_cloth_collision_rule_children())):
            if getattr(child, 'type', None) not in {'MESH', 'EMPTY'}:
                continue
            if not is_cloth_collision_rule_child(child):
                continue
            collision_rule_children.append((collision_rule_index(child, raw_child_index), int(raw_child_index), child))
        collision_rule_children.sort(key=lambda item: (int(item[0]), int(item[1]), str(getattr(item[2], 'name', '') or '')))

        plane_rule_children: list[tuple[int, int, object]] = []
        for raw_child_index, child in enumerate(list(iter_cloth_plane_rule_children())):
            if getattr(child, 'type', None) not in {'MESH', 'EMPTY'}:
                continue
            if not is_cloth_plane_rule_child(child):
                continue
            plane_rule_children.append((plane_rule_index(child, raw_child_index), int(raw_child_index), child))
        plane_rule_children.sort(key=lambda item: (int(item[0]), int(item[1]), str(getattr(item[2], 'name', '') or '')))

        if primary_chain_root_name and primary_chain_root_name in cloth_by_name:
            try:
                primary_entry = cloth_by_name.get(primary_chain_root_name, {})
                if int(primary_entry.get('flags', 0)) == 5:
                    append_bone_name(primary_chain_root_name)
            except Exception:
                pass

        def point_entry_sort_key(entry: dict) -> tuple[int, int, int, str]:
            try:
                imported_index = entry.get('imported_point_index', None)
                if imported_index is not None:
                    return (0, int(imported_index), 0, str(entry.get('key', '')))
            except Exception:
                pass
            try:
                old_index = int(entry.get('old_index', 0))
            except Exception:
                old_index = 0
            return (1, old_index, int(entry.get('segment', 0) or 0), str(entry.get('key', '')))

        point_entries.sort(key=point_entry_sort_key)

        for new_index, entry in enumerate(point_entries):
            key_to_index[str(entry['key'])] = int(new_index)
            setup.points.append(ClothPoint(
                segment=int(entry['segment']) & 0xFFFF,
                flags=int(entry['flags']) & 0xFFFF,
                jointOrder=int(entry.get('jointOrder', 0)) & 0xFFFF,
                upTo=int(entry.get('upTo', 0)) & 0xFFFF,
                x=float(entry.get('x', 0.0)),
                y=float(entry.get('y', 0.0)),
                z=float(entry.get('z', 0.0)),
            ))

        def index_for_bone_name(name: str) -> Optional[int]:
            entry = cloth_by_name.get(name)
            if not entry:
                return None
            return key_to_index.get(str(entry.get('key')))

        for key_a, key_b, radius in capsule_rule_refs:
            point_a = key_to_index.get(str(key_a))
            point_b = key_to_index.get(str(key_b))
            if point_a is None or point_b is None:
                continue
            setup.capsuleRules.append(FDCapsuleRule(
                point=(int(point_a), int(point_b)),
                flags=(point_flag_value(point_a), point_flag_value(point_b)),
                radius=max(0.0, float(radius)),
            ))

        def _signed_marker_value(value, default: int = 0) -> int:
            try:
                marker_value = int(value)
            except Exception:
                marker_value = int(default)
            if marker_value == 0:
                return 0
            # FDPlaneRule references use signed HMarker stored indices.  The
            # original PC data uses negative references for these cloth planes,
            # so authored positive indices are normalized to that convention.
            if marker_value > 0:
                marker_value = -marker_value
            return max(-32768, min(32767, int(marker_value)))

        def _iter_related_scene_objects():
            seen: set[int] = set()

            def yield_once(obj):
                if obj is None:
                    return
                try:
                    key = int(obj.as_pointer())
                except Exception:
                    key = id(obj)
                if key in seen:
                    return
                seen.add(key)
                yield obj

            try:
                for collection in list(getattr(root, 'users_collection', []) or []):
                    for obj in list(getattr(collection, 'all_objects', []) or []):
                        yield from yield_once(obj)
            except Exception:
                pass
            parent = getattr(root, 'parent', None)
            while parent is not None:
                yield from yield_once(parent)
                try:
                    for obj in list(getattr(parent, 'children_recursive', []) or getattr(parent, 'children', []) or []):
                        yield from yield_once(obj)
                except Exception:
                    pass
                parent = getattr(parent, 'parent', None)
            try:
                for obj in list(getattr(bpy.data, 'objects', []) or []):
                    yield from yield_once(obj)
            except Exception:
                pass

        def _hmarker_world_matrix(marker_obj):
            if marker_obj is None:
                return None
            try:
                return self._export_object_world_matrix(marker_obj)
            except Exception:
                return None

        def _collect_plane_rule_hmarkers():
            markers = []
            for obj in _iter_related_scene_objects():
                try:
                    if trlau_object_type(obj) != 'HMarker':
                        continue
                except Exception:
                    continue
                stored_index = hmarker_index_from_name(obj, 0)
                if int(stored_index) <= 0:
                    continue
                matrix = _hmarker_world_matrix(obj)
                if matrix is None:
                    continue
                markers.append((int(stored_index), obj, matrix))
            # Prefer the closest object instances when duplicate linked/imported
            # marker indices exist, but keep all candidates available for scoring.
            return markers

        def _point_entry_world_position(index: int) -> Vector:
            try:
                pos = Vector(tuple(float(v) for v in point_entries[int(index)].get('position', (0.0, 0.0, 0.0))))
            except Exception:
                pos = Vector((0.0, 0.0, 0.0))
            if arm_obj is not None:
                try:
                    return arm_obj.matrix_world @ pos
                except Exception:
                    pass
            return pos

        def _safe_vector_axis(vector, default=(1.0, 0.0, 0.0)) -> Vector:
            try:
                axis = Vector(tuple(float(v) for v in vector))
                if axis.length > 1.0e-6:
                    axis.normalize()
                    return axis
            except Exception:
                pass
            axis = Vector(tuple(float(v) for v in default))
            if axis.length <= 1.0e-6:
                axis = Vector((1.0, 0.0, 0.0))
            axis.normalize()
            return axis

        def _mesh_world_plane_faces(obj) -> list[dict]:
            faces: list[dict] = []

            def _is_plane_source(source_obj) -> bool:
                try:
                    source_type = str(trlau_object_type(source_obj) or '')
                    if source_type == 'ClothPlaneRuleSelector':
                        return False
                    if source_type == 'ClothPlaneRulePlane':
                        return True
                except Exception:
                    pass
                try:
                    if str(getattr(source_obj, 'name', '') or '').endswith('_Selector'):
                        return False
                except Exception:
                    pass
                return getattr(source_obj, 'type', None) == 'MESH'

            def _append_faces_from(source_obj):
                if not _is_plane_source(source_obj):
                    return
                eval_obj = source_obj
                try:
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = source_obj.evaluated_get(depsgraph)
                except Exception:
                    eval_obj = source_obj
                mesh = getattr(eval_obj, 'data', None) or getattr(source_obj, 'data', None)
                if mesh is None:
                    return
                try:
                    world = self._export_object_world_matrix(source_obj)
                except Exception:
                    world = Matrix.Identity(4)
                vertices = list(getattr(mesh, 'vertices', []) or [])
                for polygon_index, polygon in enumerate(list(getattr(mesh, 'polygons', []) or [])):
                    try:
                        indices = list(getattr(polygon, 'vertices', []) or [])
                    except Exception:
                        indices = []
                    if len(indices) < 3:
                        continue
                    try:
                        world_vertices = [world @ vertices[int(index)].co for index in indices if 0 <= int(index) < len(vertices)]
                    except Exception:
                        continue
                    if len(world_vertices) < 3:
                        continue

                    origin = world_vertices[0].copy()
                    x_axis = None
                    normal = None
                    for i in range(1, len(world_vertices)):
                        edge = world_vertices[i] - origin
                        if edge.length > 1.0e-6:
                            x_axis = edge.normalized()
                            break
                    if x_axis is None:
                        continue
                    for i in range(1, len(world_vertices) - 1):
                        a = world_vertices[i] - origin
                        b = world_vertices[i + 1] - origin
                        candidate = a.cross(b)
                        if candidate.length > 1.0e-6:
                            normal = candidate.normalized()
                            break
                    if normal is None:
                        continue
                    y_axis = normal.cross(x_axis)
                    if y_axis.length <= 1.0e-6:
                        continue
                    y_axis.normalize()
                    normal = x_axis.cross(y_axis)
                    if normal.length <= 1.0e-6:
                        continue
                    normal.normalize()

                    xs = [float((vertex - origin).dot(x_axis)) for vertex in world_vertices]
                    ys = [float((vertex - origin).dot(y_axis)) for vertex in world_vertices]
                    center = Vector((0.0, 0.0, 0.0))
                    for vertex in world_vertices:
                        center += vertex
                    center /= max(1, len(world_vertices))
                    x_min = min(xs)
                    x_max = max(xs)
                    y_min = min(ys)
                    y_max = max(ys)
                    extent = max(1.0e-5, float(x_max) - float(x_min), float(y_max) - float(y_min))
                    faces.append({
                        'source_name': str(getattr(source_obj, 'name', '') or ''),
                        'polygon_index': int(polygon_index),
                        'origin': origin,
                        'center': center,
                        'normal': normal,
                        'x_axis': x_axis,
                        'y_axis': y_axis,
                        'x_min': float(x_min),
                        'x_max': float(x_max),
                        'y_min': float(y_min),
                        'y_max': float(y_max),
                        'extent': float(extent),
                    })

            _append_faces_from(obj)
            for descendant in iter_object_descendants(obj):
                _append_faces_from(descendant)
            return faces

        def _marker_candidates_for_face(face: dict, markers: list[tuple[int, object, Matrix]]) -> list[tuple[float, int]]:
            scored: list[tuple[float, int]] = []
            for stored_index, _marker_obj, marker_matrix in markers:
                try:
                    marker_pos = marker_matrix.translation
                    marker_normal = _safe_vector_axis(marker_matrix.to_3x3().col[2], (0.0, 0.0, 1.0))
                    offset = marker_pos - face['origin']
                    local_x = float(offset.dot(face['x_axis']))
                    local_y = float(offset.dot(face['y_axis']))
                    plane_distance = abs(float(offset.dot(face['normal'])))
                    outside_x = max(0.0, float(face['x_min']) - local_x, local_x - float(face['x_max']))
                    outside_y = max(0.0, float(face['y_min']) - local_y, local_y - float(face['y_max']))
                    align_penalty = (1.0 - abs(float(face['normal'].dot(marker_normal)))) * max(1.0, float(face['extent']))
                    score = plane_distance * 6.0 + outside_x + outside_y + align_penalty
                    scored.append((float(score), int(stored_index)))
                except Exception:
                    continue
            scored.sort(key=lambda item: (item[0], item[1]))
            return scored

        def _plane_rule_explicit_marker_pair(child):
            """Resolve the authored marker pair from live plane-face constraints.

            This is still scene-derived authoring data, not a copied imported
            FDPlaneRule list: each ClothPlaneRulePlane follows an actual HMarker.
            Prefer that explicit relationship over nearest-marker scoring so
            user-created plane rules remain stable when several HMarkers are
            close together.
            """
            resolved: list[int] = []
            seen: set[int] = set()

            def _stored_index_from_marker_obj(marker_obj):
                if marker_obj is None:
                    return None
                try:
                    if str(trlau_object_type(marker_obj) or '') != 'HMarker':
                        return None
                except Exception:
                    return None
                value = hmarker_index_from_name(marker_obj, 0)
                return abs(int(value)) if int(value) != 0 else None

            def _append_index(value):
                try:
                    index = abs(int(value))
                except Exception:
                    return
                if index <= 0 or index in seen:
                    return
                seen.add(index)
                resolved.append(index)

            sources = [child]
            try:
                sources.extend(list(iter_object_descendants(child)))
            except Exception:
                pass
            for source_obj in sources:
                try:
                    if str(trlau_object_type(source_obj) or '') != 'ClothPlaneRulePlane':
                        continue
                except Exception:
                    continue
                from_constraint = None
                for constraint in list(getattr(source_obj, 'constraints', []) or []):
                    try:
                        target = getattr(constraint, 'target', None)
                        marker_index = _stored_index_from_marker_obj(target)
                        if marker_index is not None:
                            from_constraint = int(marker_index)
                            break
                    except Exception:
                        pass
                if from_constraint is not None:
                    _append_index(from_constraint)
                    continue
                try:
                    _append_index(source_obj.get('trlau_cloth_plane_marker_index'))
                except Exception:
                    pass
                if len(resolved) >= 2:
                    break
            if len(resolved) >= 2:
                return (_signed_marker_value(resolved[0]), _signed_marker_value(resolved[1]))
            return None

        def _plane_rule_marker_pair_from_faces(child, faces: list[dict]) -> tuple[int, int]:
            explicit_pair = _plane_rule_explicit_marker_pair(child)
            if explicit_pair is not None:
                return explicit_pair
            markers = _collect_plane_rule_hmarkers()
            if len(markers) >= 2 and len(faces) >= 2:
                chosen: list[int] = []
                used: set[int] = set()
                for face in faces[:2]:
                    selected = None
                    for _score, stored_index in _marker_candidates_for_face(face, markers):
                        if int(stored_index) in used:
                            continue
                        selected = int(stored_index)
                        break
                    if selected is None:
                        for _score, stored_index in _marker_candidates_for_face(face, markers):
                            selected = int(stored_index)
                            break
                    if selected is not None:
                        chosen.append(int(selected))
                        used.add(int(selected))
                if len(chosen) >= 2:
                    return (_signed_marker_value(chosen[0]), _signed_marker_value(chosen[1]))

            return 0, 0

        def _point_projects_inside_face(point_world: Vector, face: dict) -> bool:
            try:
                offset = point_world - face['origin']
                local_x = float(offset.dot(face['x_axis']))
                local_y = float(offset.dot(face['y_axis']))
                signed_distance = float(offset.dot(face['normal']))
                margin = max(0.1, float(face['extent']) * 0.035)
                normal_margin = max(0.25, float(face['extent']) * 0.02)
                return (
                    signed_distance >= -normal_margin
                    and (float(face['x_min']) - margin) <= local_x <= (float(face['x_max']) + margin)
                    and (float(face['y_min']) - margin) <= local_y <= (float(face['y_max']) + margin)
                )
            except Exception:
                return False

        def _mesh_world_selector_bounds(obj):
            def _loose_world_points(source_obj):
                mesh = getattr(source_obj, 'data', None)
                if mesh is None:
                    return []
                try:
                    world = self._export_object_world_matrix(source_obj)
                except Exception:
                    world = Matrix.Identity(4)
                vertices = list(getattr(mesh, 'vertices', []) or [])
                used_by_faces: set[int] = set()
                try:
                    for polygon in list(getattr(mesh, 'polygons', []) or []):
                        for index in list(getattr(polygon, 'vertices', []) or []):
                            used_by_faces.add(int(index))
                except Exception:
                    pass
                points = []
                for index, vertex in enumerate(vertices):
                    if int(index) in used_by_faces:
                        continue
                    try:
                        points.append(world @ vertex.co)
                    except Exception:
                        pass
                return points

            def _is_selector_object(source_obj) -> bool:
                try:
                    if str(trlau_object_type(source_obj) or '') == 'ClothPlaneRuleSelector':
                        return True
                except Exception:
                    pass
                try:
                    name = str(getattr(source_obj, 'name', '') or '')
                    return name.endswith('_Selector') or '_Selector.' in name
                except Exception:
                    return False

            loose_points = list(_loose_world_points(obj))
            if len(loose_points) < 2:
                for descendant in iter_object_descendants(obj):
                    if not _is_selector_object(descendant):
                        continue
                    loose_points.extend(_loose_world_points(descendant))

            if len(loose_points) < 2:
                return None
            xs = [float(point.x) for point in loose_points]
            ys = [float(point.y) for point in loose_points]
            zs = [float(point.z) for point in loose_points]
            extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0e-5)
            margin = max(0.1, float(extent) * 0.035)
            return (
                Vector((min(xs) - margin, min(ys) - margin, min(zs) - margin)),
                Vector((max(xs) + margin, max(ys) + margin, max(zs) + margin)),
            )

        def _point_inside_selector_bounds(point_world: Vector, bounds) -> bool:
            if bounds is None:
                return False
            min_v, max_v = bounds
            try:
                return (
                    float(min_v.x) <= float(point_world.x) <= float(max_v.x)
                    and float(min_v.y) <= float(point_world.y) <= float(max_v.y)
                    and float(min_v.z) <= float(point_world.z) <= float(max_v.z)
                )
            except Exception:
                return False

        def _plane_rule_target_points_from_faces(child, faces: list[dict]) -> list[int]:
            selector_bounds = _mesh_world_selector_bounds(child)
            active_faces = list(faces[:2])
            if selector_bounds is None and not active_faces:
                return []
            targets: list[int] = []
            for index, point in enumerate(setup.points):
                try:
                    if int(point.flags) != 4:
                        continue
                    point_world = _point_entry_world_position(int(index))
                    if selector_bounds is not None:
                        if _point_inside_selector_bounds(point_world, selector_bounds):
                            targets.append(int(index))
                    elif all(_point_projects_inside_face(point_world, face) for face in active_faces):
                        targets.append(int(index))
                except Exception:
                    continue
            return targets

        added_plane_rules: set[tuple[int, int, int]] = set()
        for _plane_sort_index, _raw_child_index, child in plane_rule_children:
            try:
                if not bool(child.get('trlau_cloth_enabled', True)):
                    continue
            except Exception:
                pass
            faces = _mesh_world_plane_faces(child)
            marker_a, marker_b = _plane_rule_marker_pair_from_faces(child, faces)
            if marker_a == 0 and marker_b == 0:
                self._add_warning(f'Cloth plane rule "{getattr(child, "name", "<unnamed>")}" could not resolve two HMarker-backed plane faces; skipping FDPlaneRule export for that plane.')
                continue
            target_indices = _plane_rule_target_points_from_faces(child, faces)
            if not target_indices:
                self._add_warning(f'Cloth plane rule "{getattr(child, "name", "<unnamed>")}" contains no movable cloth points projected inside its marker plane faces; skipping FDPlaneRule export for that plane.')
                continue
            for target_index in target_indices:
                dedupe_key = (int(target_index), int(marker_a), int(marker_b))
                if dedupe_key in added_plane_rules:
                    continue
                added_plane_rules.add(dedupe_key)
                setup.planeRules.append(FDPlaneRule(
                    point=(int(target_index),),
                    flags=(point_flag_value(int(target_index)) & 0xFFFF,),
                    planeSegment=(int(marker_a), int(marker_b)),
                ))

        def axis_for_path(path_names: list[str]) -> int:
            if len(path_names) >= 2:
                return dominant_axis_code(bone_position_by_name(path_names[1]) - bone_position_by_name(path_names[0]))
            if path_names:
                entry = cloth_by_name.get(path_names[0], {})
                parent_name = str(entry.get('armature_parent_name', '') or '')
                if parent_name:
                    return dominant_axis_code(bone_position_by_name(path_names[0]) - bone_position_by_name(parent_name))
            return 5

        for path_names in ordered_chain_paths:
            indices = [index_for_bone_name(name) for name in path_names]
            indices = [int(index) for index in indices if index is not None]
            if indices:
                indexed_paths.append(indices)
                root_index = int(indices[0])
                path_axis_by_root_index[root_index] = axis_for_path(path_names)
                path_names_by_root_index[root_index] = list(path_names)

        added_distance_keys: set[tuple[int, int, int]] = set()

        def add_distance(p0: int, p1: int, min_dist: float, max_dist: float, *, mode: int = 0) -> None:
            if p0 < 0 or p1 < 0 or p0 >= len(point_entries) or p1 >= len(point_entries):
                return
            dedupe = (min(int(p0), int(p1)), max(int(p0), int(p1)), int(mode))
            if dedupe in added_distance_keys:
                return
            added_distance_keys.add(dedupe)
            setup.distRules.append(FDDistanceRule(
                point=(int(p0), int(p1)),
                flags=(point_flag_value(p0), point_flag_value(p1)),
                minDist=float(min_dist),
                maxDist=float(max_dist),
            ))

        def nearest_point_entry_index(child, allowed_flags: set[int] | None = None, source: str | None = None) -> Optional[int]:
            try:
                child_pos = self._export_object_world_translation(child)
                local_pos = arm_obj.matrix_world.inverted() @ child_pos if arm_obj is not None else child_pos
            except Exception:
                return None
            best_index = None
            best_dist = None
            for entry_index, entry in enumerate(point_entries):
                try:
                    if allowed_flags is not None and int(entry.get('flags', 0)) not in allowed_flags:
                        continue
                    if source is not None and str(entry.get('source')) != source:
                        continue
                    entry_pos = Vector(tuple(entry.get('position')))
                    dist = float((entry_pos - local_pos).length_squared)
                except Exception:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_index = int(entry_index)
            return best_index

        exact_collision_rules: list[tuple[int, int, float, float, int]] = []
        exact_collision_targets_by_collision: dict[int, set[int]] = defaultdict(set)
        for rule_sort_index, _raw_child_index, child in collision_rule_children:
            _name_order, name_target_point_index, name_bone_segment, name_collision_point_index = collision_rule_name_info(child)
            target_segment = int(name_bone_segment) if name_bone_segment is not None else -1
            target_index = key_to_index.get(f'bone:{int(target_segment)}') if target_segment >= 0 else None
            if target_index is None and name_target_point_index is not None:
                original_target = int(name_target_point_index)
                for entry_index, entry in enumerate(point_entries):
                    entry_imported_index = entry.get('imported_point_index', None)
                    if entry_imported_index is None:
                        entry_imported_index = entry.get('old_index', -9999)
                    try:
                        if int(entry.get('flags', 0)) in {4, 5} and int(entry_imported_index) == int(original_target):
                            target_index = int(entry_index)
                            break
                    except Exception:
                        continue
            if target_index is None:
                target_index = nearest_point_entry_index(child, {4, 5}, None)

            collision_key = None
            has_stable_collision_reference = name_collision_point_index is not None
            if name_collision_point_index is not None:
                collision_key = collision_key_by_name_index.get(int(name_collision_point_index))
                if collision_key is None:
                    continue
            collision_index = key_to_index.get(str(collision_key)) if collision_key is not None else None
            if target_index is None or collision_index is None:
                if has_stable_collision_reference:
                    continue
                nearest_collision = nearest_point_entry_index(child, {1}, 'collision')
                if collision_index is None and nearest_collision is not None:
                    collision_index = int(nearest_collision)
            if target_index is None or collision_index is None:
                continue
            radius = collision_rule_radius(child, 0.0)
            min_dist = max(0.0, float(radius) * float(radius))
            if min_dist <= 0.0:
                continue
            max_dist = 1000000.0
            exact_collision_rules.append((int(target_index), int(collision_index), float(min_dist), float(max_dist), int(rule_sort_index)))
            exact_collision_targets_by_collision[int(collision_index)].add(int(target_index))

        mapped_indices: set[int] = set()
        for indices in indexed_paths:
            if indices and bool(point_entries[int(indices[0])].get('point_only_pinned', False)):
                continue
            has_movable = any(int(setup.points[index].flags) == 4 for index in indices)
            if not has_movable:
                continue
            axis_code = int(path_axis_by_root_index.get(int(indices[0]), 5))
            for order, current_index in enumerate(indices):
                if current_index in mapped_indices:
                    continue
                mapped_indices.add(current_index)
                if order + 1 < len(indices):
                    next_index = int(indices[order + 1])
                    map_points = (int(current_index), int(next_index), 0, 0) if axis_code == 4 else (int(next_index), int(current_index), 0, 0)
                elif order > 0:
                    # Terminal weighted bones still need a non-degenerate map.
                    # Original game data maps the last point back to its parent
                    # neighbor, e.g. center=tip and points=(tip,parent,0,0).
                    # Exporting (tip,tip,0,0) leaves the last weighted bone with
                    # no usable FD direction.
                    previous_index = int(indices[order - 1])
                    map_points = (int(current_index), int(previous_index), 0, 0)
                else:
                    map_points = (int(current_index), int(current_index), 0, 0)
                current_entry = point_entries[current_index]
                setup.maps.append(ClothJointMap(
                    segment=int(current_entry['segment']) & 0xFFFF,
                    flags=0,
                    axis=int(axis_code) & 0xFF,
                    jointOrder=0,
                    center=int(current_index),
                    points=map_points,
                    xMin=0.0,
                    xMax=0.0,
                    yMin=0.0,
                    yMax=0.0,
                    zMin=0.0,
                    zMax=0.0,
                ))

        deferred_anchor_rules: list[tuple[int, int, float, float, int]] = []

        def collect_anchor_rules(indices: list[int], *, exact_axis: bool) -> None:
            if not indices or int(setup.points[indices[0]].flags) != 5:
                return
            root_index = int(indices[0])
            start = 1 if exact_axis else 2
            for descendant_index in indices[start:]:
                deferred_anchor_rules.append((
                    root_index,
                    int(descendant_index),
                    0.0,
                    distance_sq(point_entries[root_index]['position'], point_entries[int(descendant_index)]['position']),
                    1,
                ))

        collision_indices = [
            index for index, entry in enumerate(point_entries)
            if str(entry.get('source')) == 'collision'
        ]

        def collision_rule_min_dist_sq(movable_index: int, collision_index: int, order: int, total: int) -> float:
            """Generate the normal/default collision lower bound.

            Per-point ClothCollisionRule helpers add exact authored lower bounds
            for specific point/collider pairs.  They should not replace the base
            ClothCollision radius.  The default collision rule therefore stays
            the editable collision empty radius squared, matching the v10
            behavior that was less stiff.
            """
            collision_entry = point_entries[int(collision_index)]
            radius = max(0.0, float(collision_entry.get('radius', 0.0) or 0.0))
            return float(radius * radius)

        deferred_pin_collision_rules: list[tuple[int, int, float, float]] = []

        def add_collision_rules_for_path(indices: list[int]) -> None:
            movable_indices = [
                int(index) for index in indices
                if int(setup.points[int(index)].flags) == 4
            ]
            if not movable_indices:
                return
            path_set = {int(index) for index in movable_indices}
            for target_index, collision_index, min_dist_sq, max_dist_sq, _rule_sort_index in exact_collision_rules:
                if int(target_index) not in path_set:
                    continue
                collision_entry = point_entries[int(collision_index)]
                # Some assets use a pinned point as a fixed collision anchor.  In
                # those files the pin-collision rules appear after the pin anchor
                # stretch rules, not with normal flags=1 collider rules.
                if int(collision_entry.get('flags', 0)) == 5:
                    deferred_pin_collision_rules.append((int(target_index), int(collision_index), float(min_dist_sq), float(max_dist_sq)))
                    continue
                if bool(collision_entry.get('generate_rules', True)):
                    default_min_dist_sq = collision_rule_min_dist_sq(int(target_index), int(collision_index), 0, 1)
                    min_dist_sq = max(float(min_dist_sq), float(default_min_dist_sq))
                add_distance(int(target_index), int(collision_index), float(min_dist_sq), float(max_dist_sq), mode=3)

            if not collision_indices:
                return
            # Keep root-to-tip order.  The game assets are order-sensitive here;
            # reversing these rules changes the sequential FD solve.
            ordered_movable = list(movable_indices)
            for collision_index in collision_indices:
                collision_entry = point_entries[int(collision_index)]
                if not bool(collision_entry.get('generate_rules', True)):
                    continue
                for order, movable_index in enumerate(ordered_movable):
                    min_dist_sq = collision_rule_min_dist_sq(int(movable_index), int(collision_index), int(order), len(ordered_movable))
                    add_distance(int(movable_index), int(collision_index), min_dist_sq, 1000000.0, mode=3)

        primary_path_root_index: Optional[int] = None
        try:
            primary_path_root_index = index_for_bone_name(primary_chain_root_name) if primary_chain_root_name else None
        except Exception:
            primary_path_root_index = None

        for indices in indexed_paths:
            if indices and bool(point_entries[int(indices[0])].get('point_only_pinned', False)):
                continue
            has_movable = any(int(setup.points[index].flags) == 4 for index in indices)
            if not has_movable:
                continue
            axis_code = int(path_axis_by_root_index.get(int(indices[0]), 5))
            exact_axis = axis_code == 4
            for order in range(max(0, len(indices) - 1)):
                p0, p1 = int(indices[order]), int(indices[order + 1])
                dist2 = distance_sq(point_entries[p0]['position'], point_entries[p1]['position'])
                add_distance(p0, p1, dist2 if exact_axis else 0.0, dist2, mode=0)
            if exact_axis and indices:
                last = int(indices[-1])
                add_distance(last, last, 0.0, 0.0, mode=0)

            add_collision_rules_for_path(indices)

            is_primary_path = primary_path_root_index is not None and int(indices[0]) == int(primary_path_root_index)
            if is_primary_path:
                collect_anchor_rules(indices, exact_axis=exact_axis)
                for p0, p1, min_dist, max_dist, mode in list(deferred_anchor_rules):
                    add_distance(p0, p1, min_dist, max_dist, mode=mode)
                deferred_anchor_rules.clear()
            else:
                collect_anchor_rules(indices, exact_axis=exact_axis)
            for p0, p1, min_dist, max_dist in list(deferred_pin_collision_rules):
                add_distance(p0, p1, min_dist, max_dist, mode=4)
            deferred_pin_collision_rules.clear()

        paths_by_parent: dict[str, list[list[int]]] = defaultdict(list)
        for path in indexed_paths:
            if not path or not any(int(setup.points[index].flags) == 4 for index in path):
                continue
            parent_name = str(point_entries[path[0]].get('armature_parent_name', '') or '')
            paths_by_parent[parent_name].append(path)

        for parent_name, paths in paths_by_parent.items():
            if len(paths) < 2:
                continue
            parent_pos = bone_position_by_name(parent_name) if parent_name else Vector((0.0, 0.0, 0.0))

            def lateral_group_key(path: list[int]) -> tuple[int, float]:
                try:
                    delta = point_entries[path[0]]['position'] - parent_pos
                    return (0 if float(delta.y) < 0.0 else 1, -float(delta.x))
                except Exception:
                    return (1, 0.0)

            grouped_paths: dict[int, list[list[int]]] = defaultdict(list)
            for path in paths:
                grouped_paths[lateral_group_key(path)[0]].append(path)

            for group_index in sorted(grouped_paths):
                ordered_paths = sorted(grouped_paths[group_index], key=lambda item: lateral_group_key(item)[1])
                if len(ordered_paths) < 2:
                    continue
                for left, right in zip(ordered_paths, ordered_paths[1:]):
                    max_depth = min(len(left), len(right))
                    for depth in range(max_depth - 1, -1, -1):
                        p0, p1 = int(left[depth]), int(right[depth])
                        add_distance(p0, p1, 0.0, distance_sq(point_entries[p0]['position'], point_entries[p1]['position']), mode=2)

        for p0, p1, min_dist, max_dist, mode in list(deferred_anchor_rules):
            add_distance(p0, p1, min_dist, max_dist, mode=mode)

        for index, point in enumerate(setup.points):
            if int(point.flags) == 4:
                setup.pointMoveRules.append(FDPointMoveRule(point=int(index), count=1))

        return setup

    def _allocate_new_section_index(self, directory: Path, section_list: _SectionList) -> int:
        used_indices = set(section_list.used_indices())
        try:
            for candidate in Path(directory).iterdir():
                if not candidate.is_file():
                    continue
                match = _SECTION_NAME_RE.match(candidate.name)
                if match is not None:
                    used_indices.add(int(match.group('index')))
        except Exception:
            logger.debug('Could not scan %s for existing section indices', directory, exc_info=True)

        return (max(used_indices) + 1) if used_indices else 0

    def _resolve_or_allocate_cloth_section(self, object_path: Path, section_list: _SectionList, roots: Sequence[object]) -> int:
        directory = Path(object_path).parent

        def existing_filename_for_index(section_index: int) -> Optional[str]:
            """Resolve a section filename even when sectionList.txt is absent.

            Direct .drm export extracts standalone files into a temporary folder
            but does not create sectionList.txt.  In that path the Object
            section still has the authoritative rdSetupList relocation target,
            so rejecting it just because sectionList has no entry creates a
            duplicate cloth section.
            """
            try:
                section_index = int(section_index)
            except Exception:
                return None
            listed = section_list.filename_for_index(section_index)
            if listed and (directory / listed).exists():
                return str(listed)
            try:
                existing = self._resolve_existing_section_file(directory, section_index)
            except Exception:
                existing = None
            if existing is not None and existing.exists():
                return existing.name
            return str(listed) if listed else None

        # Prefer the current object section's RDSetup relocation target.  Older
        # versions stored this as trlau_cloth_section_index on the Cloth empty,
        # but that made imported Cloth empties noisy.  The object section already
        # contains the authoritative pointer, so derive it during export.
        try:
            object_section = read_standalone_section(object_path)
            for reloc in list(getattr(object_section, 'relocations', []) or []):
                try:
                    target, _type_specific, offset, _relocation_type = reloc
                    if int(offset) != RD_SETUP_LIST_FIELD_OFFSET:
                        continue
                    section_index = int(target)
                    filename = existing_filename_for_index(section_index)
                    if filename:
                        section_list.ensure_index(section_index, filename)
                        return int(section_index)
                except Exception:
                    continue
        except Exception:
            logger.debug('Could not infer cloth section from %s', object_path, exc_info=True)

        next_index = self._allocate_new_section_index(directory, section_list)
        section_list.ensure_index(next_index, f'{next_index}_0.gnc')
        return int(next_index)

    def _write_object_rdsetup_pointer(self, object_path: Path, cloth_section_index: int) -> Path:
        section = read_standalone_section(object_path)
        data = bytearray(section.data)
        if len(data) < RD_SETUP_LIST_FIELD_OFFSET + 4:
            data.extend(b'\x00' * ((RD_SETUP_LIST_FIELD_OFFSET + 4) - len(data)))
        struct.pack_into('<I', data, RD_SETUP_LIST_FIELD_OFFSET, 0)
        relocs = [
            reloc for reloc in section.relocations
            if int(reloc[2]) != RD_SETUP_LIST_FIELD_OFFSET
        ]
        relocs.append((int(cloth_section_index), 0, RD_SETUP_LIST_FIELD_OFFSET, 0))
        write_standalone_section(
            object_path,
            data=bytes(data),
            relocations=relocs,
            section_type=section.section_type,
            skip_flags=section.skip_flags,
            version_id=section.version_id,
            has_debug_info=section.has_debug_info,
            resource_type=section.resource_type,
            section_id=section.section_id,
            spec_mask=section.spec_mask,
        )
        return object_path

    def _export_cloth_sections(self, collection, object_path: Path, section_list: _SectionList) -> List[Path]:
        roots = self._cloth_roots_for_collection(collection)
        if not roots:
            return []

        setups = [self._build_cloth_setup_from_root(root) for root in roots]
        cloth_section_index = self._resolve_or_allocate_cloth_section(object_path, section_list, roots)
        cloth_filename = section_list.filename_for_index(cloth_section_index) or f'{cloth_section_index}_0.gnc'
        if _section_index_from_filename(cloth_filename) != cloth_section_index:
            cloth_filename = f'{cloth_section_index}_0.gnc'
        section_list.ensure_index(cloth_section_index, cloth_filename)
        cloth_path = object_path.parent / cloth_filename

        data, relocs, metadata = pack_cloth_section(setups)
        fixed_relocs = [
            (cloth_section_index if int(target) < 0 else int(target), int(type_specific), int(offset), int(relocation_type))
            for target, type_specific, offset, relocation_type in relocs
        ]
        write_standalone_section(
            cloth_path,
            data=data,
            relocations=fixed_relocs,
            section_type=int(getattr(metadata, 'section_type', 0)),
            skip_flags=int(getattr(metadata, 'skip_flags', 0)),
            version_id=int(getattr(metadata, 'version_id', 0)),
            has_debug_info=int(getattr(metadata, 'has_debug_info', 0)),
            resource_type=int(getattr(metadata, 'resource_type', 0)),
            section_id=int(getattr(metadata, 'section_id', 0)),
            spec_mask=int(getattr(metadata, 'spec_mask', 0xFFFFFFFF)),
        )
        self._write_object_rdsetup_pointer(object_path, cloth_section_index)
        logger.info('Exported %d cloth setup(s) to section %s', len(setups), cloth_path.name)
        return [object_path, cloth_path]


    def export_collection(self, context, collection, object_filepath: str, *, export_textures: bool = True, export_cloth: bool = True) -> List[Path]:
        previous_overrides = getattr(self, '_export_scene_matrix_overrides', None)
        self._export_scene_matrix_overrides = self._snapshot_export_component_world_matrices(collection)
        rest_pose_state = self._temporarily_use_rest_position_for_export(context, collection)
        try:
            return self._export_collection_rest_position(context, collection, object_filepath, export_textures=export_textures, export_cloth=export_cloth)
        finally:
            self._restore_pose_position_after_export(context, rest_pose_state)
            self._export_scene_matrix_overrides = previous_overrides


    def _collection_has_underworld_model_roots(self, collection) -> bool:
        if collection is None:
            return False
        for obj in _iter_collection_objects(collection):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not bool(getattr(obj, 'trlau_is_model_empty', False)):
                continue
            if self._model_game_from_root(obj) == 'underworld':
                return True
        return False

    def _export_underworld_collection_rest_position(self, context, collection, object_filepath: str, *, export_textures: bool = True, export_cloth: bool = True) -> List[Path]:
        del export_textures
        obj_path = Path(object_filepath)
        if obj_path.suffix.lower() != '.obj':
            raise ValueError('Underworld model collection export requires selecting the extracted .obj/objectref section file or exporting to a source .drm template')
        if not obj_path.exists():
            raise FileNotFoundError(str(obj_path))

        directory = obj_path.parent
        section_list = _SectionList.load(directory)
        refs = TRObjectParser(str(obj_path)).parse_underworld_mesh_references()
        if not refs:
            raise ValueError(f'No Underworld cdcModelData references were found in {obj_path.name}')

        roots_by_index = self._model_roots_by_file_index(collection)
        if not roots_by_index:
            raise ValueError(f'Collection "{collection.name}" does not contain imported Underworld model empties')

        written: List[Path] = []
        missing_section_indices: List[int] = []
        for ref in refs:
            tr8mesh_path = Path(ref.tr8mesh_filepath)
            section_index = _section_index_from_filename(tr8mesh_path.name)
            model_root = roots_by_index.get(section_index)
            if model_root is None:
                missing_section_indices.append(int(section_index))
                continue
            if self._model_game_from_root(model_root) != 'underworld':
                missing_section_indices.append(int(section_index))
                continue

            target_mesh_path = self._resolve_existing_section_file(directory, section_index, preferred=tr8mesh_path.name)
            if target_mesh_path is None:
                raise FileNotFoundError(f'Could not find Underworld cdcModelData section {section_index} ({tr8mesh_path.name}) in the export template')

            component_visibility_state = self._temporarily_enable_component_visibility_for_export(context, model_root)
            try:
                arm_obj = self._find_armature(model_root)
                mesh_objects = self._find_mesh_objects(model_root, arm_obj)
                if not mesh_objects:
                    raise ValueError(f'Underworld model "{getattr(model_root, "name", "<model>")}" contains no mesh object to export')
                if len(mesh_objects) > 1:
                    self._add_warning(
                        f'Underworld model "{getattr(model_root, "name", "<model>")}" has multiple mesh objects; exporting the first one: {getattr(mesh_objects[0], "name", "<mesh>")}.'
                    )
                render_id = int(getattr(ref, 'cdc_modeldata_id', 0) or 0) & 0xFFFFFFFF
                if render_id == 0:
                    render_id = int(getattr(model_root, 'trlau_ui_cdc_render_data_id', model_root.get('trlau_cdc_render_data_id', 0)) or 0) & 0xFFFFFFFF
                result = export_pc_underworld_model_section(
                    context,
                    directory,
                    model_root,
                    arm_obj,
                    mesh_objects[0],
                    render_id,
                    section_list=section_list,
                    template_path=target_mesh_path,
                    warning_callback=self._add_warning,
                )
                if result is not None and result.path not in written:
                    written.append(result.path)
            finally:
                self._restore_component_visibility_after_export(context, component_visibility_state)

        if not written:
            if missing_section_indices:
                shown = ', '.join(str(index) for index in missing_section_indices[:16])
                if len(missing_section_indices) > 16:
                    shown += ', ...'
                self._add_warning(f'No matching Underworld model root was found for cdcModelData section(s): {shown}')
            raise ValueError('No Underworld mesh sections were exported; the selected collection did not match the source DRM references')

        if export_cloth:
            cloth_paths = self._export_cloth_sections(collection, obj_path, section_list)
            written.extend(path for path in cloth_paths if path not in written)
        section_list.write()
        return written

    def _export_collection_rest_position(self, context, collection, object_filepath: str, *, export_textures: bool = True, export_cloth: bool = True) -> List[Path]:
        obj_path = Path(object_filepath)
        if obj_path.suffix.lower() != '.obj':
            raise ValueError('Model collection export requires selecting the extracted .obj object section file')
        if not obj_path.exists():
            raise FileNotFoundError(str(obj_path))

        if self._collection_has_underworld_model_roots(collection):
            return self._export_underworld_collection_rest_position(context, collection, object_filepath, export_textures=export_textures, export_cloth=export_cloth)

        directory = obj_path.parent
        section_list = _SectionList.load(directory)
        model_refs = TRObjectParser(str(obj_path)).parse_model_references()
        if not model_refs:
            raise ValueError(f'No model references were found in {obj_path.name}')

        roots_by_index = self._model_roots_by_file_index(collection)
        if not roots_by_index:
            raise ValueError(f'Collection "{collection.name}" does not contain imported TRLAU model empties')

        written: List[Path] = []
        model_written: List[Path] = []
        missing_section_indices: List[int] = []
        for ref in model_refs:
            section_index = _section_index_from_filename(Path(ref.target_filepath).name)
            model_root = roots_by_index.get(section_index)
            if model_root is None:
                missing_section_indices.append(int(section_index))
                continue
            target_model_path = self._resolve_existing_section_file(directory, section_index, preferred=Path(ref.target_filepath).name)
            if target_model_path is None:
                target_model_path = directory / f'{section_index}_0.tr7aemesh'
            layout = _OriginalModelLayout.from_model_file(target_model_path, section_list) if target_model_path.exists() else self._default_layout(target_model_path, section_list)
            component_visibility_state = self._temporarily_enable_component_visibility_for_export(context, model_root)
            nextgen_written_path = None
            try:
                self._sync_hinfo_transforms(context, model_root)
                requested_render_id = int(getattr(model_root, 'trlau_ui_cdc_render_data_id', model_root.get('trlau_cdc_render_data_id', 0)) or 0) & 0xFFFFFFFF
                nextgen_meshes = find_pc_nextgen_meshes(model_root, requested_render_id) if requested_render_id else []
                model = self._build_model_data_from_scene(context, model_root)
                if requested_render_id:
                    if not nextgen_meshes:
                        self._add_warning(
                            f'Model "{getattr(model_root, "name", "<model>")}" has RenderID 0x{requested_render_id:X}, but no PC Next Gen mesh was found under the model root; exporting the ordinary model with RenderID 0.'
                        )
                        model.cdc_render_data_id = 0
                    else:
                        if len(nextgen_meshes) > 1:
                            self._add_warning(
                                f'Model "{getattr(model_root, "name", "<model>")}" has multiple PC Next Gen meshes for RenderID 0x{requested_render_id:X}; exporting the first one: {getattr(nextgen_meshes[0], "name", "<mesh>")}.'
                            )
                        arm_obj = self._find_armature(model_root)
                        try:
                            nextgen_result = export_pc_nextgen_model_section(
                                context,
                                directory,
                                model_root,
                                arm_obj,
                                nextgen_meshes[0],
                                requested_render_id,
                                section_list=section_list,
                                warning_callback=self._add_warning,
                            )
                        except Exception as exc:
                            self._add_warning(
                                f'PC Next Gen export for RenderID 0x{requested_render_id:X} failed: {exc}'
                            )
                            nextgen_result = None
                        if nextgen_result is None:
                            self._add_warning(
                                f'PC Next Gen export for RenderID 0x{requested_render_id:X} was not written; exporting the ordinary model with RenderID 0.'
                            )
                            model.cdc_render_data_id = 0
                        else:
                            nextgen_written_path = nextgen_result.path
                            model.cdc_render_data_id = requested_render_id
                section_paths = self._write_model_sections(directory, layout, model, section_list)
            finally:
                self._restore_component_visibility_after_export(context, component_visibility_state)
            model_written.extend(section_paths)
            written.extend(section_paths)
            if nextgen_written_path is not None and nextgen_written_path not in written:
                written.append(nextgen_written_path)

        if not model_written:
            if missing_section_indices:
                shown = ', '.join(str(index) for index in missing_section_indices[:16])
                if len(missing_section_indices) > 16:
                    shown += ', ...'
                message = f'No matching model section IDs were found in collection {collection.name}; missing referenced section(s): {shown}'
                self._add_warning(message)
            raise ValueError('No model sections were exported; the selected collection did not match the modelList section IDs')

        if export_cloth:
            cloth_paths = self._export_cloth_sections(collection, obj_path, section_list)
            written.extend(path for path in cloth_paths if path not in written)

        if export_textures:
            texture_paths = self._export_model_textures(collection, directory, section_list)
            written.extend(texture_paths)
        section_list.write()

        return written

    def export_collection_to_drm(self, context, collection, drm_filepath: str, *, source_drm_filepath: str | None = None, export_textures: bool = True, export_cloth: bool = True) -> List[Path]:
        """Export a regular model collection directly into a DRM container.

        The regular model writer still needs an object-section template. For a
        DRM target, use an existing/source DRM as that template, extract it into
        a scoped temporary directory, rewrite the edited model/texture sections,
        then repack every section into the requested DRM path.
        """
        target_path = Path(drm_filepath)
        if target_path.suffix.lower() != '.drm':
            raise ValueError('Direct model DRM export requires a .drm target path')

        source_path = Path(source_drm_filepath).expanduser() if source_drm_filepath else self.find_source_drm_for_collection(collection, target_path)
        if source_path is None:
            raise FileNotFoundError(
                'Could not find a source DRM template for this model collection. '
                'Export over the original DRM, keep the original DRM next to the imported collection source, or export to an extracted .obj section instead.'
            )
        source_path = source_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(str(source_path))

        drm_parser = DRMContainerParser(str(source_path))
        with drm_parser.temporary_extract_sections() as (extract_dir, extracted_paths, source_sections):
            object_path = next((path for path in extracted_paths if path.suffix.lower() == '.obj'), None)
            if object_path is None:
                object_path = extracted_paths[0] if extracted_paths else None
            if object_path is None or not object_path.exists():
                raise ValueError(f'No object section was found in {source_path.name}')

            section_written = self.export_collection(context, collection, str(object_path), export_textures=export_textures, export_cloth=export_cloth)
            packed_sections = write_drm_from_standalone_directory(extract_dir, target_path, source_sections=source_sections)
            logger.info('Packed %d DRM section(s) into %s', len(packed_sections), target_path)
            return [target_path, *section_written]

    @staticmethod
    def find_source_drm_for_collection(collection, target_path: str | Path | None = None) -> Optional[Path]:
        """Infer the original/source DRM for a model collection when possible."""
        if target_path is not None:
            candidate = Path(target_path)
            if candidate.suffix.lower() == '.drm' and candidate.exists():
                return candidate

        if collection is None:
            return None

        explicit_values = []
        for key in ('trlau_source_drm', 'trlau_source_file'):
            try:
                value = str(collection.get(key, '') or '').strip()
            except Exception:
                value = ''
            if value:
                explicit_values.append(value)

        objects = list(_iter_collection_objects(collection))
        for obj in objects:
            for key in ('trlau_source_drm', 'trlau_source_file'):
                try:
                    value = str(obj.get(key, '') or '').strip()
                except Exception:
                    value = ''
                if value:
                    explicit_values.append(value)

        for value in explicit_values:
            candidate = Path(value)
            if candidate.suffix.lower() == '.drm' and candidate.exists():
                return candidate.resolve()

        source_dirs: list[Path] = []
        for key in ('trlau_source_dir',):
            try:
                value = str(collection.get(key, '') or '').strip()
            except Exception:
                value = ''
            if value:
                source_dirs.append(Path(value))
        for obj in objects:
            try:
                value = str(obj.get('trlau_source_dir', '') or '').strip()
            except Exception:
                value = ''
            if value:
                source_dirs.append(Path(value))
        seen: set[Path] = set()
        unique_dirs: list[Path] = []
        for directory in source_dirs:
            try:
                resolved = directory.expanduser().resolve()
            except Exception:
                resolved = directory
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_dir():
                unique_dirs.append(resolved)

        collection_name = str(getattr(collection, 'name', '') or '').strip()
        possible_stems = {collection_name}
        if collection_name:
            possible_stems.add(collection_name.removesuffix('_Model'))
        for directory in unique_dirs:
            for stem in [stem for stem in possible_stems if stem]:
                candidate = directory / f'{stem}.drm'
                if candidate.exists():
                    return candidate.resolve()
            matches = [candidate for candidate in directory.glob('*.drm') if candidate.stem.lower() == collection_name.lower()]
            if matches:
                return matches[0].resolve()
        return None


    def _default_layout(self, target_model_path: Path, section_list: _SectionList) -> _OriginalModelLayout:
        root_index = _section_index_from_filename(target_model_path.name)
        filename = target_model_path.name
        section_list.ensure_index(root_index, filename)
        return _OriginalModelLayout(
            model_section_index=root_index,
            model_filename=filename,
            model_section_info=None,
            pointer_targets={field: root_index for field in (12, 36, 44, 52, 88, 92, 96, 100, 116, 128, 136)},
            section_infos={},
            section_filenames={root_index: filename},
            hinfo_targets={-1: root_index},
            pointer_values={},
            segment_list_offset=_MODEL_HEADER_SIZE_PC,
            version=8,
        )

    def _model_roots_by_file_index(self, collection) -> Dict[int, object]:
        result: Dict[int, object] = {}
        for obj in _iter_collection_objects(collection):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'EMPTY':
                continue
            if not bool(getattr(obj, 'trlau_is_model_empty', False)):
                continue
            file_index = self._file_index_from_model_root(obj)
            if file_index is not None:
                result[int(file_index)] = obj
        return result

    @staticmethod
    def _file_index_from_model_root(obj) -> Optional[int]:
        for key in ('trlau_model_file_index', 'trlau_file_index'):
            if key in obj:
                try:
                    return int(obj.get(key))
                except Exception:
                    pass
        match = _MODEL_ROOT_RE.match(obj.name)
        if match is not None:
            try:
                return int(match.group('index'))
            except Exception:
                return None
        parts = obj.name.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return None

    def _resolve_existing_section_file(self, directory: Path, section_index: int, preferred: str = '') -> Optional[Path]:
        if preferred:
            preferred_path = directory / preferred
            if preferred_path.exists():
                return preferred_path
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file() and _section_index_from_filename(candidate.name) == int(section_index):
                match = _SECTION_NAME_RE.match(candidate.name)
                if match is not None and int(match.group('index')) == int(section_index):
                    return candidate
        return None

    def _section_filename(self, directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, section_index: int, *, default_ext: str = 'gnc', section_id: int = 0) -> str:
        section_index = int(section_index)
        if section_index in layout.section_filenames:
            return layout.section_filenames[section_index]
        listed = section_list.filename_for_index(section_index)
        if listed:
            return listed
        for candidate in sorted(directory.iterdir()):
            match = _SECTION_NAME_RE.match(candidate.name)
            if match is not None and int(match.group('index')) == section_index:
                return candidate.name
        return f'{section_index}_{int(section_id):x}.{default_ext}'

    def _make_buffer(self, directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, section_index: int, *, root: bool = False, default_ext: str = 'gnc', section_id: int = 0) -> _StandaloneSectionBuffer:
        section_index = int(section_index)
        filename = layout.model_filename if root else self._section_filename(directory, layout, section_list, section_index, default_ext=default_ext, section_id=section_id)
        info = layout.model_section_info if root else layout.section_infos.get(section_index)
        buffer = _StandaloneSectionBuffer(
            section_index=section_index,
            filename=filename,
            section_type=int(getattr(info, 'section_type', 0) if info is not None else (0 if default_ext != 'pcd' else 5)),
            section_id=int(getattr(info, 'section_id', section_id) if info is not None else section_id),
            skip_flags=int(getattr(info, 'skip_flags', 0) if info is not None else 0),
            version_id=int(getattr(info, 'version_id', 0) if info is not None else 0),
            has_debug_info=int(getattr(info, 'has_debug_info', 0) if info is not None else 0),
            resource_type=int(getattr(info, 'resource_type', 0) if info is not None else 0),
            spec_mask=int(getattr(info, 'spec_mask', 0xFFFFFFFF) if info is not None else 0xFFFFFFFF),
        )
        section_list.ensure_index(section_index, filename)
        return buffer

    def _get_or_make_buffer(self, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, section_index: int) -> _StandaloneSectionBuffer:
        if int(section_index) not in buffers:
            buffers[int(section_index)] = self._make_buffer(directory, layout, section_list, int(section_index))
        return buffers[int(section_index)]

    def _append_to_target(self, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, target_section_index: int, payload: bytes, alignment: int = 4) -> tuple[int, _StandaloneSectionBuffer]:
        buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, int(target_section_index))
        offset = buffer.append(payload, alignment=alignment)
        return offset, buffer

    def _write_model_sections(self, directory: Path, layout: _OriginalModelLayout, model: ModelData, section_list: _SectionList) -> List[Path]:
        buffers: Dict[int, _StandaloneSectionBuffer] = {}
        root = self._make_buffer(directory, layout, section_list, layout.model_section_index, root=True, default_ext='tr7aemesh')
        root.reserve(0x90, alignment=16)
        buffers[root.section_index] = root

        struct.pack_into('<iii', root.data, 0, _PC_MODEL_VERSION_MAGIC, len(model.segments), len(model.virt_segments))
        self._write_bone_mirrors(root, buffers, directory, layout, section_list, model)
        self._write_segment_data(root, buffers, directory, layout, section_list, model)
        self._write_vertex_data(root, buffers, directory, layout, section_list, model)
        self._write_face_data(root, buffers, directory, layout, section_list, model)
        strip_entry_offsets = self._write_strip_data(root, buffers, directory, layout, section_list, model)
        self._write_model_scroll_info(root, buffers, directory, layout, section_list, model, strip_entry_offsets)
        self._write_index_list(root, buffers, directory, layout, section_list, 92, model.env_mapped_face_indices)
        self._write_index_list(root, buffers, directory, layout, section_list, 96, model.eye_ref_env_mapped_face_indices)
        self._write_missing_model_extra_zero_blocks(root, buffers, directory, layout, section_list)
        self._write_vertex_colors(root, buffers, directory, layout, section_list, model)
        self._write_markups(root, buffers, directory, layout, section_list, model)
        self._write_targets(root, buffers, directory, layout, section_list, model)

        max_rad = self._generate_max_radius(model)
        struct.pack_into('<4f', root.data, 16, *[float(v) for v in model.model_scale])
        struct.pack_into('<i', root.data, 40, 1)
        struct.pack_into('<ff', root.data, 60, max_rad, max_rad * max_rad)
        struct.pack_into('<hh', root.data, 108, 0, 0)
        struct.pack_into('<hh', root.data, 112, 0, 0)
        struct.pack_into('<I', root.data, 140, int(model.cdc_render_data_id) & 0xFFFFFFFF)

        written: List[Path] = []
        for section_index, buffer in sorted(buffers.items(), key=lambda item: item[0]):
            out_path = directory / buffer.filename
            _write_standalone_section_file(out_path, buffer)
            written.append(out_path)
            logger.info('Wrote TRLAU model export section %s', out_path.name)
        return written

    def _write_pointer(self, root: _StandaloneSectionBuffer, field_offset: int, target_section_index: int, target_offset: int) -> None:
        root.write_pointer_at(int(field_offset), int(target_section_index), int(target_offset))

    def _preferred_pointer_offset(self, layout: _OriginalModelLayout, field_offset: int) -> Optional[int]:
        value = int(getattr(layout, 'pointer_values', {}).get(int(field_offset), 0) or 0)
        return value if value > 0 else None

    def _write_payload_to_target(self, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, target_section_index: int, payload: bytes, *, alignment: int = 4, preferred_offset: Optional[int] = None) -> tuple[int, _StandaloneSectionBuffer]:
        buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, int(target_section_index))
        if preferred_offset is not None and preferred_offset >= 0:
            preferred_offset = int(preferred_offset)
            if preferred_offset >= len(buffer.data):
                if int(target_section_index) != int(layout.model_section_index) or (preferred_offset + len(payload)) <= int(getattr(layout, 'segment_list_offset', _MODEL_HEADER_SIZE_PC) or _MODEL_HEADER_SIZE_PC):
                    buffer.data.extend(b'\x00' * (preferred_offset - len(buffer.data)))
                    offset = preferred_offset
                    buffer.data.extend(payload)
                    return offset, buffer
        return self._append_to_target(buffers, directory, layout, section_list, int(target_section_index), payload, alignment=alignment)

    def _extra_section_target(self, layout: _OriginalModelLayout) -> int:
        root_index = int(layout.model_section_index)
        for field_offset in (100, 96, 92, 44, 76, 80, 84):
            target = int(layout.target_for(field_offset))
            if target != root_index:
                return target
        return root_index

    def _ensure_zero_dword(self, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, target_section_index: int) -> int:
        buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, int(target_section_index))
        existing = getattr(buffer, '_trlau_zero_dword_offset', None)
        if existing is not None:
            return int(existing)
        offset = buffer.append(b'\x00\x00\x00\x00', alignment=4)
        setattr(buffer, '_trlau_zero_dword_offset', int(offset))
        return int(offset)

    def _write_zero_pointer(self, root: _StandaloneSectionBuffer, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, field_offset: int, target_section_index: Optional[int] = None) -> None:
        target = self._extra_section_target(layout) if target_section_index is None else int(target_section_index)
        zero_offset = self._ensure_zero_dword(buffers, directory, layout, section_list, target)
        self._write_pointer(root, int(field_offset), target, zero_offset)

    @staticmethod
    def _has_pointer_relocation(root: _StandaloneSectionBuffer, field_offset: int) -> bool:
        return any(int(reloc.offset) == int(field_offset) for reloc in getattr(root, 'relocations', []) or [])

    def _append_dedicated_zero_block(self, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, target_section_index: int, size: int = 4) -> int:
        buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, int(target_section_index))
        return buffer.append(b'\x00' * max(4, int(size)), alignment=4)

    def _write_dedicated_zero_pointer(self, root: _StandaloneSectionBuffer, buffers: Dict[int, _StandaloneSectionBuffer], directory: Path, layout: _OriginalModelLayout, section_list: _SectionList, field_offset: int, target_section_index: Optional[int] = None, size: int = 4) -> None:
        target = self._extra_section_target(layout) if target_section_index is None else int(target_section_index)
        zero_offset = self._append_dedicated_zero_block(buffers, directory, layout, section_list, target, size=size)
        self._write_pointer(root, int(field_offset), target, zero_offset)

    def _write_missing_model_extra_zero_blocks(self, root, buffers, directory, layout, section_list) -> None:
        extra_target = self._extra_section_target(layout)
        for field_offset in (44, 76, 80, 84):
            if not self._has_pointer_relocation(root, field_offset):
                self._write_dedicated_zero_pointer(root, buffers, directory, layout, section_list, field_offset, extra_target)

    def _write_segment_data(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        target = layout.target_for(12)
        seg_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, target)

        if seg_buffer is root and int(target) == int(root.section_index):
            desired_offset = max(_MODEL_HEADER_SIZE_PC, int(getattr(layout, 'segment_list_offset', _MODEL_HEADER_SIZE_PC) or _MODEL_HEADER_SIZE_PC))
            if len(seg_buffer.data) < desired_offset:
                seg_buffer.data.extend(b'\x00' * (desired_offset - len(seg_buffer.data)))
        segment_list_offset = seg_buffer.align(16)

        segments = sorted(model.segments, key=lambda item: int(item.index))
        for ordinal, segment in enumerate(segments):
            expected_index = int(segment.index)
            if expected_index != ordinal:
                expected_index = ordinal
            offset = segment_list_offset + (expected_index * _SEGMENT_SIZE_PC)
            if len(seg_buffer.data) < offset + _SEGMENT_SIZE_PC:
                seg_buffer.data.extend(b'\x00' * ((offset + _SEGMENT_SIZE_PC) - len(seg_buffer.data)))
            struct.pack_into('<4f4f4fihhiI', seg_buffer.data, offset,
                *[float(v) for v in segment.min_v],
                *[float(v) for v in segment.max_v],
                *[float(v) for v in segment.pivot],
                int(segment.flags), int(segment.first_vertex), int(segment.last_vertex), int(segment.parent), 0)

        virt_base = segment_list_offset + (len(segments) * _SEGMENT_SIZE_PC)
        for virt_index, virt in enumerate(model.virt_segments):
            offset = virt_base + (virt_index * _VIRT_SEGMENT_SIZE_PC)
            if len(seg_buffer.data) < offset + _VIRT_SEGMENT_SIZE_PC:
                seg_buffer.data.extend(b'\x00' * ((offset + _VIRT_SEGMENT_SIZE_PC) - len(seg_buffer.data)))
            struct.pack_into('<4f4f4fihhhhf', seg_buffer.data, offset,
                *[float(v) for v in virt.min_v],
                *[float(v) for v in virt.max_v],
                *[float(v) for v in virt.pivot],
                int(virt.flags), int(virt.first_vertex), int(virt.last_vertex), int(virt.index), int(virt.weight_index), float(virt.weight))

        hinfo_payloads = self._build_hinfo_payloads(model)
        for segment in segments:
            hinfo_payload = hinfo_payloads.get(int(segment.index))
            if not hinfo_payload:
                continue
            hinfo_target = layout.hinfo_target_for(int(segment.index))
            hinfo_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, hinfo_target)
            hinfo_offset = hinfo_buffer.align(4)
            hinfo_data = bytearray(hinfo_payload)
            hinfo_pointer_fields: List[int] = []
            for pointer_field in (4, 12, 20, 28):
                value = struct.unpack_from('<I', hinfo_data, pointer_field)[0]
                if value:
                    struct.pack_into('<I', hinfo_data, pointer_field, int(hinfo_offset + value) & 0xFFFFFFFF)
                    hinfo_pointer_fields.append(pointer_field)
            hinfo_buffer.data.extend(hinfo_data)
            for pointer_field in hinfo_pointer_fields:
                hinfo_buffer.relocations.append(_Relocation(hinfo_target, int(hinfo_offset + pointer_field)))
            segment_record_offset = segment_list_offset + (int(segment.index) * _SEGMENT_SIZE_PC)
            field_offset = segment_record_offset + (_SEGMENT_SIZE_PC - 4)
            seg_buffer.pack_at(field_offset, '<I', int(hinfo_offset) & 0xFFFFFFFF)
            seg_buffer.relocations.append(_Relocation(hinfo_target, int(field_offset)))

        self._write_pointer(root, 12, target, segment_list_offset)

    def _write_vertex_data(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        target = layout.target_for(36)
        payload = bytearray()
        for vertex in model.vertices:
            payload.extend(struct.pack('<hhhbbbbhHH',
                int(vertex.position_raw[0]), int(vertex.position_raw[1]), int(vertex.position_raw[2]),
                int(vertex.normal_raw[0]), int(vertex.normal_raw[1]), int(vertex.normal_raw[2]), 0,
                int(vertex.segment), int(vertex.uv_raw[0]) & 0xFFFF, int(vertex.uv_raw[1]) & 0xFFFF))
        offset, _buffer = self._append_to_target(buffers, directory, layout, section_list, target, bytes(payload), alignment=16)
        struct.pack_into('<i', root.data, 32, len(model.vertices))
        self._write_pointer(root, 36, target, offset)

    def _write_face_data(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        if not model.faces:
            struct.pack_into('<iI', root.data, 48, 0, 0)
            return
        target = layout.target_for(52)
        payload = bytearray()
        for face in model.faces:
            payload.extend(struct.pack('<HHHH', int(face.v0), int(face.v1), int(face.v2), int(face.same_vert_bits) & 0xFFFF))
        offset, _buffer = self._append_to_target(buffers, directory, layout, section_list, target, bytes(payload), alignment=16)
        struct.pack_into('<i', root.data, 48, len(model.faces))
        self._write_pointer(root, 52, target, offset)

    @staticmethod
    def _split_strip_indices(indices: Sequence[int]) -> Iterable[List[int]]:
        clean = [int(index) & 0xFFFF for index in list(indices or [])]
        if not clean:
            return
        if len(clean) <= _TEXTURE_STRIP_MAX_I16_COUNT:
            yield clean
            return

        triangle_aligned = (len(clean) % 3) == 0
        chunk_size = _TEXTURE_STRIP_TRIANGLE_CHUNK_COUNT if triangle_aligned else _TEXTURE_STRIP_MAX_I16_COUNT
        start = 0
        while start < len(clean):
            remaining = len(clean) - start
            count = remaining if remaining <= _TEXTURE_STRIP_MAX_I16_COUNT else chunk_size
            if triangle_aligned:
                count -= count % 3
                if count <= 0:
                    count = min(remaining, _TEXTURE_STRIP_TRIANGLE_CHUNK_COUNT)
            yield clean[start:start + count]
            start += count

    def _write_strip_data(self, root, buffers, directory, layout, section_list, model: ModelData) -> Dict[int, List[Tuple[int, int]]]:
        strip_entry_offsets: Dict[int, List[Tuple[int, int]]] = {}
        if not model.strips:
            return strip_entry_offsets
        target = layout.target_for(88)
        strip_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, target)
        first_offset = None
        previous_next_field = None
        for strip_index, strip in enumerate(model.strips):
            indices = [int(index) & 0xFFFF for index in list(strip.indices or [])]
            if not indices:
                continue
            draw_group = int(getattr(strip, 'draw_group', 0) or 0)
            if draw_group < -32768 or draw_group > 32767:
                raise ValueError(f'Model texture strip {strip_index} has draw group {draw_group}, which is outside the signed 16-bit range -32768..32767')

            chunks = list(self._split_strip_indices(indices))
            if not chunks:
                continue
            if len(chunks) > 1:
                logger.info('Splitting model texture strip %d into %d TextureStripInfo entries (%d indices)', int(strip_index), len(chunks), len(indices))

            for chunk_indices in chunks:
                entry_offset = strip_buffer.append(b'', alignment=4)
                strip_entry_offsets.setdefault(int(strip_index), []).append((int(target), int(entry_offset)))
                if first_offset is None:
                    first_offset = entry_offset
                if previous_next_field is not None:
                    strip_buffer.pack_at(previous_next_field, '<I', entry_offset)
                    strip_buffer.relocations.append(_Relocation(target, previous_next_field))
                vertex_count = len(chunk_indices)
                header = struct.pack('<hhiffI', int(vertex_count), draw_group, int(strip.tpageid) & 0xFFFFFFFF, float(strip.sort_push), float(strip.scroll_offset), 0)
                strip_buffer.data.extend(header)
                previous_next_field = entry_offset + 16
                for index in chunk_indices:
                    strip_buffer.data.extend(struct.pack('<H', index))
                if (vertex_count & 1) != 0:
                    strip_buffer.data.extend(struct.pack('<h', -1))
        if previous_next_field is not None:
            sentinel_offset = strip_buffer.append(b'\x00\x00\x00\x00', alignment=4)
            strip_buffer.pack_at(previous_next_field, '<I', sentinel_offset)
            strip_buffer.relocations.append(_Relocation(target, previous_next_field))
        if first_offset is not None:
            self._write_pointer(root, 88, target, first_offset)
        return strip_entry_offsets

    def _write_model_scroll_info(self, root, buffers, directory, layout, section_list, model: ModelData, strip_entry_offsets: Dict[int, List[Tuple[int, int]]]) -> None:
        entries: List[Tuple[int, int, int, float, int, int]] = []
        for strip_index, strip in enumerate(model.strips or []):
            offsets = list(strip_entry_offsets.get(int(strip_index), []) or [])
            if not offsets:
                continue

            speeds = list(getattr(strip, 'scroll_speeds', []) or [])
            if not speeds and int(getattr(strip, 'scroll_entry_count', 0) or 0) > 0:
                speeds = [0.0]
            if not speeds:
                continue

            scroll_num_tiles = int(getattr(strip, 'scroll_num_tiles', 0) or 0)
            scroll_tile = int(getattr(strip, 'scroll_tile', 0) or 0)
            for strip_target, strip_offset in offsets:
                for speed in speeds:
                    entries.append((int(strip_index), int(strip_target), int(strip_offset), float(speed), scroll_num_tiles, scroll_tile))

        if not entries:
            return

        target = self._extra_section_target(layout)
        scroll_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, target)
        scroll_offset = scroll_buffer.reserve(4 + (len(entries) * 20) + 4, alignment=4)
        scroll_buffer.pack_at(scroll_offset + 0, '<i', len(entries))
        animated_zero_offset = scroll_offset + 4 + (len(entries) * 20)
        scroll_buffer.pack_at(animated_zero_offset, '<i', 0)

        for entry_index, (_strip_index, strip_target, strip_offset, raw_speed, scroll_num_tiles, scroll_tile) in enumerate(entries):
            entry_offset = scroll_offset + 4 + (entry_index * 20)
            next_entry_offset = entry_offset + 20
            scroll_buffer.write_pointer_at(entry_offset + 0, target, next_entry_offset)
            scroll_buffer.pack_at(entry_offset + 4, '<f', float(raw_speed))
            scroll_buffer.pack_at(entry_offset + 8, '<i', int(scroll_num_tiles))
            scroll_buffer.pack_at(entry_offset + 12, '<i', int(scroll_tile))
            scroll_buffer.write_pointer_at(entry_offset + 16, int(strip_target), int(strip_offset) + 12)

        self._write_pointer(root, 80, target, animated_zero_offset)
        self._write_pointer(root, 84, target, scroll_offset)

    def _write_index_list(self, root, buffers, directory, layout, section_list, field_offset: int, indices: Sequence[int]) -> None:
        clean = sorted({int(index) for index in indices if int(index) >= 0})
        target = layout.target_for(field_offset)
        if not clean:
            self._write_zero_pointer(root, buffers, directory, layout, section_list, field_offset, self._extra_section_target(layout))
            return
        payload = struct.pack('<i', len(clean)) + b''.join(struct.pack('<H', int(index) & 0xFFFF) for index in clean)
        offset, _buffer = self._append_to_target(buffers, directory, layout, section_list, target, payload, alignment=4)
        self._write_pointer(root, field_offset, target, offset)

    def _write_vertex_colors(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        colors = list(model.vertex_colors or [])
        if not colors or len(colors) != len(model.vertices):
            return
        payload = bytearray()
        for r, g, b, a in colors:
            payload.extend(bytes((int(b) & 0xFF, int(g) & 0xFF, int(r) & 0xFF, int(a) & 0xFF)))
        target = layout.target_for(100)
        offset, _buffer = self._append_to_target(buffers, directory, layout, section_list, target, bytes(payload), alignment=4)
        self._write_pointer(root, 100, target, offset)

    def _write_bone_mirrors(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        entries = list(model.bone_mirror_entries or [])
        if not entries:
            return
        payload = bytearray()
        for entry in entries:
            payload.extend(bytes((int(entry.bone1) & 0xFF, int(entry.bone2) & 0xFF, int(entry.count) & 0xFF)))
        payload.extend(b'\x00\x00')
        target = layout.target_for(116)
        preferred = self._preferred_pointer_offset(layout, 116) if int(target) == int(layout.model_section_index) else None
        if int(target) == int(layout.model_section_index):
            if preferred is not None:
                preferred = int(preferred) + 4
            else:
                preferred = 0x98
        offset, _buffer = self._write_payload_to_target(buffers, directory, layout, section_list, target, bytes(payload), alignment=4, preferred_offset=preferred)
        self._write_pointer(root, 116, target, offset)

    @staticmethod
    def _normalize_model_game(value: object) -> str:
        return normalize_game_value(value)

    @classmethod
    def _model_game_from_root(cls, model_root) -> str:
        if model_root is None:
            return 'legend'
        for key in ('trlau_model_game_id',):
            try:
                if key in model_root:
                    return cls._normalize_model_game(model_root.get(key))
            except Exception:
                pass
        try:
            cdc_id = int(getattr(model_root, 'trlau_ui_cdc_render_data_id', model_root.get('trlau_cdc_render_data_id', 0)) or 0)
            return 'legend' if cdc_id != 0 else 'anniversary'
        except Exception:
            return 'legend'

    @staticmethod
    def _identity_float_triplet(value: Sequence[float]) -> Tuple[float, float, float]:
        x = float(value[0]) if len(value) > 0 else 0.0
        y = float(value[1]) if len(value) > 1 else 0.0
        z = float(value[2]) if len(value) > 2 else 0.0
        return (x, y, z)

    @staticmethod
    def _tuple_get(value: object, index: int, default: object = 0) -> object:
        if hasattr(value, 'to_list'):
            try:
                value = value.to_list()
            except Exception:
                pass
        try:
            return value[index]
        except Exception:
            return default

    @staticmethod
    def _i16(value: object) -> int:
        try:
            value = int(round(float(value)))
        except Exception:
            value = 0
        return max(-0x8000, min(0x7FFF, value))

    def _write_markup_polyline(self, section: _StandaloneSectionBuffer, points: Sequence[Sequence[float]]) -> int:
        point_count = max(0, len(points))
        offset = section.reserve(16 + (point_count * 16), alignment=16)
        section.pack_at(offset + 0, '<i', point_count)
        for point_index, point in enumerate(points):
            px, py, pz = self._identity_float_triplet(point[:3])
            pw = float(self._tuple_get(point, 3, 1.0))
            section.pack_at(offset + 16 + (point_index * 16), '<4f', float(px), float(py), float(pz), float(pw))
        return offset

    def _write_markups(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        markups = sorted(list(getattr(model, 'markups', []) or []), key=lambda item: int(getattr(item, 'index', 0)))
        if not markups:
            struct.pack_into('<iI', root.data, 124, 0, 0)
            return

        game = self._normalize_model_game(getattr(markups[0], 'game', 'legend'))
        entry_size = 76 if game == 'anniversary' else 48
        target_section = layout.target_for(128)
        markup_buffer = self._get_or_make_buffer(buffers, directory, layout, section_list, target_section)
        polyline_offsets = [
            None
            if (int(getattr(markup, 'flags', 0) or 0) & _MARKUP_BBOX_FLAGS)
            else self._write_markup_polyline(markup_buffer, list(getattr(markup, 'polyline', []) or []))
            for markup in markups
        ]
        table_offset = markup_buffer.reserve(len(markups) * entry_size, alignment=16)
        for entry_index, markup in enumerate(markups):
            entry_offset = table_offset + (entry_index * entry_size)
            markup_buffer.pack_at(entry_offset + 0, '<i', 0)  # OverrideMovementCamera
            markup_buffer.pack_at(entry_offset + 4, '<i', 0)  # DTPCameraDataID
            markup_buffer.pack_at(entry_offset + 8, '<i', 0)  # DTPMarkupDataID
            cursor = entry_offset + 12
            if game == 'anniversary':
                markup_buffer.pack_at(cursor, '<i', int(getattr(markup, 'animated_segment', 0) or 0))
                cursor += 4
                markup_buffer.pack_at(cursor, '<i', 0)  # CameraAntic.UseAnticCamera
                cursor += 4
                for _ in range(5):
                    markup_buffer.pack_at(cursor, '<i', 0)
                    cursor += 4
            markup_buffer.pack_at(cursor, '<I', int(getattr(markup, 'flags', 0) or 0) & 0xFFFFFFFF)
            cursor += 4
            markup_buffer.pack_at(cursor, '<hh', 0, 0)  # introID, markupID
            cursor += 4
            px, py, pz = self._identity_float_triplet(getattr(markup, 'position', (0.0, 0.0, 0.0))[:3])
            markup_buffer.pack_at(cursor, '<3f', float(px), float(py), float(pz))
            cursor += 12
            bbox = tuple(getattr(markup, 'bbox', (0, 0, 0, 0, 0, 0)) or (0, 0, 0, 0, 0, 0))
            bbox_values = tuple(self._i16(self._tuple_get(bbox, bbox_index, 0)) for bbox_index in range(6))
            markup_buffer.pack_at(cursor, '<6h', *bbox_values)
            cursor += 12
            polyline_offset = polyline_offsets[entry_index]
            if polyline_offset is None:
                markup_buffer.pack_at(cursor, '<I', 0)
            else:
                markup_buffer.write_pointer_at(cursor, target_section, polyline_offset)

        struct.pack_into('<i', root.data, 124, len(markups))
        self._write_pointer(root, 128, target_section, table_offset)

    def _write_targets(self, root, buffers, directory, layout, section_list, model: ModelData) -> None:
        targets = list(model.targets or [])
        if not targets:
            return
        payload = bytearray()
        for target in targets:
            payload.extend(struct.pack('<HHffffffI', int(target.segment) & 0xFFFF, int(target.flags) & 0xFFFF, *[float(v) for v in target.position], *[float(v) for v in target.rotation], int(target.unique_id) & 0xFFFFFFFF))
        target_section = layout.target_for(136)
        offset, _buffer = self._append_to_target(buffers, directory, layout, section_list, target_section, bytes(payload), alignment=4)
        struct.pack_into('<i', root.data, 132, len(targets))
        self._write_pointer(root, 136, target_section, offset)

    def _build_hinfo_payloads(self, model: ModelData) -> Dict[int, bytes]:
        spheres_by_segment: Dict[int, List[HSphere]] = defaultdict(list)
        boxes_by_segment: Dict[int, List[HBox]] = defaultdict(list)
        markers_by_segment: Dict[int, List[HMarker]] = defaultdict(list)
        capsules_by_segment: Dict[int, List[HCapsule]] = defaultdict(list)
        for item in model.hspheres:
            spheres_by_segment[int(item.owner_segment)].append(item)
        for item in model.hboxes:
            boxes_by_segment[int(item.owner_segment)].append(item)
        for item in model.hmarkers:
            markers_by_segment[int(item.owner_segment)].append(item)
        for item in model.hcapsules:
            capsules_by_segment[int(item.owner_segment)].append(item)

        payloads: Dict[int, bytes] = {}
        for segment in model.segments:
            segment_index = int(segment.index)
            spheres = spheres_by_segment.get(segment_index, [])
            boxes = boxes_by_segment.get(segment_index, [])
            markers = markers_by_segment.get(segment_index, [])
            capsules = capsules_by_segment.get(segment_index, [])
            if not (spheres or boxes or markers or capsules):
                continue
            data = bytearray(b'\x00' * 32)
            cursor = 32

            def append_block(payload: bytes) -> int:
                nonlocal cursor
                _align_length(data, 4)
                cursor = len(data)
                data.extend(payload)
                return cursor

            sphere_payload = bytearray()
            for sphere in spheres:
                sphere_payload.extend(struct.pack('<HBBHhhhIHBBBBh', int(sphere.flags) & 0xFFFF, int(sphere.id) & 0xFF, int(sphere.rank) & 0xFF, int(sphere.radius) & 0xFFFF, int(sphere.x), int(sphere.y), int(sphere.z), int(sphere.radius_sq) & 0xFFFFFFFF, int(sphere.mass) & 0xFFFF, int(sphere.buoyancy_factor) & 0xFF, int(sphere.explosion_factor) & 0xFF, int(sphere.material_type) & 0xFF, int(sphere.pad) & 0xFF, int(sphere.damage)))
            sphere_offset = append_block(bytes(sphere_payload)) if spheres else 0

            box_payload = bytearray()
            for box in boxes:
                box_payload.extend(struct.pack('<4f4f4fHBBHBBBBhI', *[float(v) for v in box.dimensions], *[float(v) for v in box.position], *[float(v) for v in box.quaternion], int(box.flags) & 0xFFFF, int(box.id) & 0xFF, int(box.rank) & 0xFF, int(box.mass) & 0xFFFF, int(box.buoyancy_factor) & 0xFF, int(box.explosion_factor) & 0xFF, int(box.material_type) & 0xFF, int(box.pad) & 0xFF, int(box.damage), 0))
            box_offset = append_block(bytes(box_payload)) if boxes else 0

            marker_payload = bytearray()
            for marker in markers:
                # HMarker records store bone and marker index as 32-bit signed
                # integers, followed by position and ZYX rotation floats.
                marker_payload.extend(struct.pack('<iiffffff', int(marker.bone), int(marker.index), *[float(v) for v in marker.position], *[float(v) for v in marker.rotation]))
            marker_offset = append_block(bytes(marker_payload)) if markers else 0

            capsule_payload = bytearray()
            for capsule in capsules:
                capsule_payload.extend(struct.pack('<4f4fHBBHHHBBBBh', *[float(v) for v in capsule.position], *[float(v) for v in capsule.quaternion], int(capsule.flags) & 0xFFFF, int(capsule.id) & 0xFF, int(capsule.rank) & 0xFF, int(capsule.radius) & 0xFFFF, int(capsule.length) & 0xFFFF, int(capsule.mass) & 0xFFFF, int(capsule.buoyancy_factor) & 0xFF, int(capsule.explosion_factor) & 0xFF, int(capsule.material_type) & 0xFF, int(capsule.pad) & 0xFF, int(capsule.damage)))
            capsule_offset = append_block(bytes(capsule_payload)) if capsules else 0

            struct.pack_into('<iIiIiIiI', data, 0, len(spheres), sphere_offset, len(boxes), box_offset, len(markers), marker_offset, len(capsules), capsule_offset)
            payloads[segment_index] = bytes(data)
        return payloads

    @staticmethod
    def _is_export_component_visibility_object(obj) -> bool:
        if obj is None:
            return False
        try:
            if trlau_object_type(obj) in _EXPORT_COMPONENT_TYPES:
                return True
        except Exception:
            pass
        name = str(getattr(obj, 'name', '') or '')
        if name.endswith(_EXPORT_COMPONENT_ROOT_SUFFIXES) or name.endswith(_EXPORT_COMPONENT_GROUP_SUFFIXES):
            return True
        try:
            component_name = str(obj.get('trlau_component_name', '') or '')
        except Exception:
            component_name = ''
        return component_name in {'HInfo', 'Targets', 'Bounds'}

    def _component_visibility_objects_for_export(self, model_root) -> List[object]:
        result: List[object] = []
        seen: set[str] = set()
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if not self._is_export_component_visibility_object(obj):
                continue
            name = str(getattr(obj, 'name', '') or '')
            key = name or str(id(obj))
            if key in seen:
                continue
            seen.add(key)
            result.append(obj)
        return result

    def _temporarily_enable_component_visibility_for_export(self, context, model_root) -> List[tuple]:
        states: List[tuple] = []
        for obj in self._component_visibility_objects_for_export(model_root):
            try:
                hide_viewport = bool(getattr(obj, 'hide_viewport', False))
            except Exception:
                hide_viewport = False
            try:
                hide_set = bool(obj.hide_get())
            except Exception:
                hide_set = None
            if not hide_viewport and not hide_set:
                continue
            states.append((obj, hide_viewport, hide_set))
            try:
                obj.hide_viewport = False
            except Exception:
                pass
            if hide_set is not None:
                try:
                    obj.hide_set(False)
                except Exception:
                    pass
        if states:
            try:
                context.view_layer.update()
            except Exception:
                pass
        return states

    @staticmethod
    def _restore_component_visibility_after_export(context, states: Sequence[tuple]) -> None:
        if not states:
            return
        for obj, hide_viewport, hide_set in reversed(list(states)):
            try:
                obj.hide_viewport = bool(hide_viewport)
            except Exception:
                pass
            if hide_set is not None:
                try:
                    obj.hide_set(bool(hide_set))
                except Exception:
                    pass
        try:
            context.view_layer.update()
        except Exception:
            pass

    @staticmethod
    def _temporarily_use_rest_position_for_export(context, collection) -> List[tuple]:
        states: List[tuple] = []
        seen_data: set[int] = set()
        if collection is None:
            return states
        for obj in list(_iter_collection_objects(collection)):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'ARMATURE':
                continue
            arm_data = getattr(obj, 'data', None)
            if arm_data is None or not hasattr(arm_data, 'pose_position'):
                continue
            data_key = id(arm_data)
            if data_key in seen_data:
                continue
            seen_data.add(data_key)
            try:
                original = str(getattr(arm_data, 'pose_position', 'POSE') or 'POSE')
            except Exception:
                continue
            states.append((arm_data, original))
            if original != 'REST':
                try:
                    arm_data.pose_position = 'REST'
                except Exception:
                    pass
        if states:
            try:
                context.view_layer.update()
            except Exception:
                pass
        return states

    @staticmethod
    def _restore_pose_position_after_export(context, states: Sequence[tuple]) -> None:
        if not states:
            return
        for arm_data, original in reversed(list(states)):
            try:
                arm_data.pose_position = original
            except Exception:
                pass
        try:
            context.view_layer.update()
        except Exception:
            pass

    def _build_model_data_from_scene(self, context, model_root) -> ModelData:
        arm_obj = self._find_armature(model_root)
        mesh_objects = self._find_mesh_objects(model_root, arm_obj)
        if not mesh_objects:
            raise ValueError(f'Model {model_root.name} contains no mesh objects to export')
        # HInfo transforms are collected directly from object matrices below.
        bones = self._collect_bones(arm_obj)
        segments, pivot_by_segment = self._build_segments_from_bones(bones, mesh_objects)
        model_scale = self._compute_model_scale(mesh_objects, arm_obj, segments, pivot_by_segment)
        vertices, faces, strips, vertex_colors, virt_segments, env_indices, eye_ref_indices, vertex_weights = self._collect_mesh_geometry(context, model_root, arm_obj, mesh_objects, segments, pivot_by_segment, model_scale)
        self._validate_model_vertex_limit(model_root, vertices)
        segments = self._finalize_segment_bounds(segments, vertices, vertex_weights, pivot_by_segment, model_scale)
        return ModelData(
            version=_PC_MODEL_VERSION_MAGIC,
            model_scale=model_scale,
            segments=segments,
            virt_segments=virt_segments,
            vertices=vertices,
            faces=faces,
            strips=strips,
            vertex_colors=vertex_colors,
            env_mapped_face_indices=env_indices,
            eye_ref_env_mapped_face_indices=eye_ref_indices,
            hmarkers=self._collect_hmarkers(model_root, arm_obj),
            hspheres=self._collect_hspheres(model_root, arm_obj),
            hboxes=self._collect_hboxes(model_root, arm_obj),
            hcapsules=self._collect_hcapsules(model_root, arm_obj),
            targets=self._collect_targets(model_root, arm_obj),
            markups=self._collect_markups(model_root, self._model_game_from_root(model_root)),
            max_rad=0.0,
            max_rad_sq=0.0,
            bone_mirror_entries=self._collect_bone_mirrors(model_root),
            cdc_render_data_id=int(getattr(model_root, 'trlau_ui_cdc_render_data_id', model_root.get('trlau_cdc_render_data_id', 0)) or 0),
            uv_format='pc',
        )

    def _validate_model_vertex_limit(self, model_root, vertices: Sequence[MVertex]) -> None:
        vertex_count = len(vertices or [])
        if vertex_count <= _MODEL_MAX_GAME_VERTICES:
            return
        model_name = getattr(model_root, 'name', '<unnamed model>')
        raise ValueError(
            f'Model "{model_name}" exports {vertex_count} vertices, but one TRLAU model mesh cannot exceed '
            f'{_MODEL_MAX_GAME_VERTICES} vertices or the game can crash. Split the geometry into another Model empty '
            'or reduce the mesh before exporting. UV, normal, and color seams can make the exported vertex count higher '
            'than Blender\'s visible vertex count.'
        )

    def _sync_hinfo_transforms(self, context, model_root) -> None:
        try:
            context.view_layer.update()
        except Exception:
            pass
        try:
            from ...core.hinfo_ui import sync_hinfo_object
        except Exception:
            sync_hinfo_object = None
        if sync_hinfo_object is None:
            return
        try:
            depsgraph = context.evaluated_depsgraph_get()
        except Exception:
            depsgraph = None
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if trlau_object_type(obj) in {'HMarker', 'HSphere', 'HBox', 'HCapsule', 'Target'}:
                try:
                    sync_hinfo_object(obj, depsgraph)
                except Exception as exc:
                    logger.debug('Could not sync HInfo transform for %s before export: %s', getattr(obj, 'name', '<unknown>'), exc)

    def _find_armature(self, model_root):
        for obj in _iter_collection_objects(model_root.users_collection[0] if model_root.users_collection else bpy.context.scene.collection):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) == 'ARMATURE' and _is_descendant_of(obj, model_root):
                return obj
        for child in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(child):
                continue
            if getattr(child, 'type', None) == 'ARMATURE':
                return child
        return None

    def _find_mesh_objects(self, model_root, arm_obj) -> List[object]:
        objects = list(getattr(model_root, 'children_recursive', []) or [])
        result = []
        for obj in objects:
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'MESH':
                continue
            if is_pc_nextgen_mesh_object(obj):
                continue
            if trlau_object_type(obj) in {'Bound', 'Collision'}:
                continue
            if _is_inside_cloth_authoring_root(obj):
                continue
            result.append(obj)
        if result:
            return result
        if arm_obj is not None:
            for obj in getattr(arm_obj, 'children', []) or []:
                if _should_ignore_model_export_object(obj):
                    continue
                if _is_inside_cloth_authoring_root(obj):
                    continue
                if getattr(obj, 'type', None) == 'MESH' and not is_pc_nextgen_mesh_object(obj):
                    result.append(obj)
        return result

    def _collect_bones(self, arm_obj) -> Dict[int, object]:
        bones: Dict[int, object] = {}
        if arm_obj is None or getattr(arm_obj, 'data', None) is None:
            bones[0] = None
            return bones
        for bone in arm_obj.data.bones:
            match = _BONE_RE.match(bone.name)
            if match is None:
                continue
            bones[int(match.group(1))] = bone
        if not bones:
            raise ValueError(f'Armature {arm_obj.name} does not contain bone_# bones')
        max_index = max(bones)
        exported_bone_count = int(max_index) + 1
        if exported_bone_count > 180:
            self._add_warning(
                f'Armature "{arm_obj.name}" exports {exported_bone_count} bones; the model bone limit is 180.'
            )
        missing = [index for index in range(max_index + 1) if index not in bones]
        if missing:
            raise ValueError(
                'Model armature bones must be contiguous; missing '
                + ', '.join(f'bone_{index}' for index in missing[:8])
                + '. Use Utilities > Make Bones Contiguous before export.'
            )
        return bones

    def _build_segments_from_bones(self, bones: Dict[int, object], mesh_objects: Sequence[object]) -> tuple[List[Segment], Dict[int, Vector]]:
        segments: List[Segment] = []
        pivot_by_segment: Dict[int, Vector] = {}
        if not bones or bones == {0: None}:
            pivot_by_segment[0] = Vector((0.0, 0.0, 0.0))
            return [Segment(0, (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), 0, 0, -1, -1, 0)], pivot_by_segment
        for index in range(max(bones) + 1):
            bone = bones[index]
            head = Vector(getattr(bone, 'head_local', (0.0, 0.0, 0.0)))
            parent_index = -1
            parent_head = Vector((0.0, 0.0, 0.0))
            parent = getattr(bone, 'parent', None)
            if parent is not None:
                match = _BONE_RE.match(getattr(parent, 'name', ''))
                if match is not None:
                    parent_index = int(match.group(1))
                    parent_head = Vector(getattr(parent, 'head_local', (0.0, 0.0, 0.0)))
            pivot = head - parent_head if parent_index >= 0 else head
            pivot_by_segment[index] = head
            segments.append(Segment(index=index, min_v=(0.0, 0.0, 0.0, 0.0), max_v=(0.0, 0.0, 0.0, 0.0), pivot=(float(pivot.x), float(pivot.y), float(pivot.z), 1.0), flags=0, first_vertex=0, last_vertex=-1, parent=parent_index, hinfo=0))
        return segments, pivot_by_segment

    def _compute_model_scale(self, mesh_objects: Sequence[object], arm_obj, segments: Sequence[Segment], pivot_by_segment: Dict[int, Vector]) -> tuple[float, float, float, float]:
        max_abs = [0.0, 0.0, 0.0]
        arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else (mesh_objects[0].matrix_world.inverted_safe() if mesh_objects else Matrix.Identity(4))
        for mesh_obj in mesh_objects:
            mesh = getattr(mesh_obj, 'data', None)
            if mesh is None:
                continue
            for src_vertex in getattr(mesh, 'vertices', []) or []:
                primary, _secondary, _secondary_weight = self._vertex_segment_weights(mesh_obj, int(src_vertex.index), arm_obj)
                primary = max(0, min(primary, max(0, len(segments) - 1)))
                world = mesh_obj.matrix_world @ src_vertex.co
                arm_local = arm_inv @ world
                pivot = pivot_by_segment.get(primary, Vector((0.0, 0.0, 0.0)))
                rel = arm_local - pivot
                max_abs[0] = max(max_abs[0], abs(float(rel.x)))
                max_abs[1] = max(max_abs[1], abs(float(rel.y)))
                max_abs[2] = max(max_abs[2], abs(float(rel.z)))
        # Use the modelScale fields as quantization factors. Choosing them from
        # the scene-space per-axis extents keeps signed 16-bit vertex positions
        # in range while using as much precision as the format allows.
        result = []
        for axis in range(3):
            if max_abs[axis] <= 1e-9:
                result.append(1.0)
            else:
                result.append(max(float(max_abs[axis]) / 32760.0, 1e-9))
        return (float(result[0]), float(result[1]), float(result[2]), 1.0)

    def _is_ps2_export_context(self) -> bool:
        return str(getattr(self, 'platform_name', '') or '').lower() == 'ps2'

    @staticmethod
    def _mesh_int_attribute_value(mesh, name: str, *, local_vertex_index: int, loop_index: int) -> int | None:
        try:
            attributes = getattr(mesh, 'attributes', None)
            attr = attributes.get(name) if attributes is not None else None
            if attr is None:
                return None
            domain = str(getattr(attr, 'domain', '') or '').upper()
            data = getattr(attr, 'data', None)
            if data is None:
                return None
            attr_index = int(loop_index) if domain in {'CORNER', 'FACE_CORNER'} else int(local_vertex_index)
            if attr_index < 0 or attr_index >= len(data):
                return None
            item = data[attr_index]
            if hasattr(item, 'value'):
                return int(getattr(item, 'value'))
            return int(item)
        except Exception:
            return None

    def _ps2_uv_from_loop(self, mesh, local_vertex_index: int, loop_index: int, uv) -> tuple[int, int] | None:
        if not self._is_ps2_export_context():
            return None

        # PS2 model ST values are signed 4096-scale coordinates.  Always encode
        # from the authored Blender UV layer; do not use hidden/raw UV mesh
        # attributes.  Direct float -> PS2 ST encoding keeps edited UVs usable
        # and avoids the PC half-float quantization path.
        if uv is not None:
            return (_encode_ps2_scaled_uv_component(float(uv.x)), _encode_ps2_scaled_uv_component(1.0 - float(uv.y)))
        return (0, 0)


    def _collect_mesh_geometry(self, context, model_root, arm_obj, mesh_objects, segments, pivot_by_segment, model_scale):
        sx, sy, sz, _sw = model_scale
        sx = sx if abs(float(sx)) > 1e-9 else 1.0
        sy = sy if abs(float(sy)) > 1e-9 else 1.0
        sz = sz if abs(float(sz)) > 1e-9 else 1.0
        arm_inv = arm_obj.matrix_world.inverted_safe() if arm_obj is not None else model_root.matrix_world.inverted_safe()
        vertices: List[MVertex] = []
        faces: List[MFace] = []
        strips_by_material: Dict[tuple, TextureStrip] = {}
        vertex_colors: List[Tuple[int, int, int, int]] = []
        vertex_weights: List[tuple[int, int, float]] = []
        env_indices: List[int] = []
        eye_ref_indices: List[int] = []
        any_vertex_color = False
        has_flat_shaded_material = False
        overweight_stats: Dict[str, int] = {}
        mface_group_by_model_vertex: Dict[int, tuple] = {}

        for mesh_obj in mesh_objects:
            mesh = mesh_obj.data
            try:
                mesh.calc_loop_triangles()
                mesh.calc_normals_split()
            except Exception:
                pass
            uv_layer = mesh.uv_layers.active.data if getattr(mesh.uv_layers, 'active', None) is not None else (mesh.uv_layers[0].data if mesh.uv_layers else None)
            color_attr = self._active_color_attribute(mesh)
            has_color_attr = color_attr is not None
            any_vertex_color = any_vertex_color or has_color_attr
            material_lookup = list(getattr(mesh, 'materials', []) or [])
            has_flat_shaded_material = has_flat_shaded_material or any(self._material_flat_shading(material) for material in material_lookup)
            mface_group_by_local_vertex, mface_triangles = self._mface_embedded_data(mesh_obj, mesh)
            mface_sort_top_to_bottom = bool(mesh.get('trlau_mface_sort_top_to_bottom', False))
            local_vertex_order = list(range(len(mesh.vertices)))
            mface_height_by_local_vertex: Dict[int, float] = {}
            mface_point_sort_rank: Dict[int, int] = {}
            mesh_export_start = len(vertices)

            mface_group_key_by_local_vertex: Dict[int, tuple] = {}
            vertex_base_data: Dict[int, tuple[tuple[int, int, int], tuple[int, int, float]]] = {}
            for src_vertex in mesh.vertices:
                local_vertex_index = int(src_vertex.index)
                primary, secondary, secondary_weight = self._vertex_segment_weights(mesh_obj, local_vertex_index, arm_obj, overweight_stats)
                primary = max(0, min(primary, max(0, len(segments) - 1)))
                if secondary >= 0:
                    secondary = max(0, min(secondary, max(0, len(segments) - 1)))
                    if secondary == primary or secondary_weight <= 1e-6:
                        secondary = -1
                        secondary_weight = 0.0
                world = mesh_obj.matrix_world @ src_vertex.co
                arm_local = arm_inv @ world
                if mface_group_by_local_vertex:
                    mface_height_by_local_vertex[int(local_vertex_index)] = float(getattr(arm_local, 'z', 0.0))
                pivot = pivot_by_segment.get(primary, Vector((0.0, 0.0, 0.0)))
                raw = Vector(((arm_local.x - pivot.x) / sx, (arm_local.y - pivot.y) / sy, (arm_local.z - pivot.z) / sz))
                vertex_base_data[local_vertex_index] = ((_int16(raw.x), _int16(raw.y), _int16(raw.z)), (primary, secondary, secondary_weight))

            used_loops_by_vertex: Dict[int, List[int]] = defaultdict(list)
            for tri in getattr(mesh, 'loop_triangles', []) or []:
                for loop_index in tri.loops:
                    try:
                        local_vertex_index = int(mesh.loops[loop_index].vertex_index)
                    except Exception:
                        continue
                    used_loops_by_vertex[local_vertex_index].append(int(loop_index))

            def estimate_mface_variant_count(local_vertex_index: int) -> int:
                local_vertex_index = int(local_vertex_index)
                loop_indices = used_loops_by_vertex.get(local_vertex_index) or [-1]
                variants = set()
                for loop_index in loop_indices:
                    normal = Vector((0.0, 0.0, 1.0))
                    if int(loop_index) >= 0:
                        try:
                            normal = Vector(mesh.loops[int(loop_index)].normal)
                        except Exception:
                            normal = Vector((0.0, 0.0, 1.0))
                    else:
                        try:
                            normal = Vector(mesh.vertices[local_vertex_index].normal)
                        except Exception:
                            normal = Vector((0.0, 0.0, 1.0))
                    if normal.length <= 1e-8:
                        normal = Vector((0.0, 0.0, 1.0))
                    else:
                        normal = normal.normalized()
                    normal_raw = (_normal_to_i8(normal.x), _normal_to_i8(normal.y), _normal_to_i8(normal.z))

                    uvx = uvy = 0
                    if uv_layer is not None and int(loop_index) >= 0:
                        uv = uv_layer[int(loop_index)].uv
                        ps2_uv = self._ps2_uv_from_loop(mesh, local_vertex_index, int(loop_index), uv)
                        if ps2_uv is not None:
                            uvx, uvy = ps2_uv
                        else:
                            uvx = _encode_pc_uv_component(float(uv.x))
                            uvy = _encode_pc_uv_component(1.0 - float(uv.y))

                    color = self._sample_color(color_attr, int(loop_index), local_vertex_index) if has_color_attr and int(loop_index) >= 0 else ((255, 255, 255, 128) if has_color_attr else None)
                    variants.add((int(uvx), int(uvy), normal_raw, color if has_color_attr else None))
                return max(1, len(variants))

            # An authored MFacePoint can span vertices that must be ordered in
            # different export buckets because of bone/weight data. SameVertBits
            # ranges must remain consecutive and can store at most 31 vertices,
            # so split the authored point automatically by the normalized export
            # weight bucket and then into 31-safe chunks when needed. This keeps
            # the user-facing grouping broad while emitting encodable runtime
            # ranges.
            if mface_group_by_local_vertex:
                owner_name = str(getattr(mesh_obj, 'name', '') or getattr(mesh, 'name', 'Mesh'))
                point_heights: Dict[int, List[float]] = defaultdict(list)
                for local_vertex_index, point_index in mface_group_by_local_vertex.items():
                    point_heights[int(point_index)].append(float(mface_height_by_local_vertex.get(int(local_vertex_index), 0.0)))
                if mface_sort_top_to_bottom:
                    ordered_points = sorted(
                        point_heights.keys(),
                        key=lambda point: (
                            -sum(point_heights[point]) / max(1, len(point_heights[point])),
                            int(point),
                        ),
                    )
                    mface_point_sort_rank = {int(point): int(rank) for rank, point in enumerate(ordered_points)}
                else:
                    mface_point_sort_rank = {int(point): int(point) for point in point_heights.keys()}

                segment_count_for_mface = max(1, len(segments or []))
                base_group_vertices: Dict[tuple, List[int]] = defaultdict(list)
                for local_vertex_index, point_index in mface_group_by_local_vertex.items():
                    _position_raw, weight_tuple = vertex_base_data.get(int(local_vertex_index), ((0, 0, 0), (0, -1, 0.0)))
                    primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count_for_mface)
                    base_key = (
                        owner_name,
                        int(mface_point_sort_rank.get(int(point_index), int(point_index))),
                        int(point_index),
                        int(primary),
                        int(secondary),
                        round(float(weight), 6),
                    )
                    base_group_vertices[base_key].append(int(local_vertex_index))

                auto_chunked_groups = 0
                for base_key, local_indices in base_group_vertices.items():
                    ordered_local_indices = sorted(
                        (int(index) for index in local_indices),
                        key=lambda index: (-float(mface_height_by_local_vertex.get(int(index), 0.0)), int(index)) if mface_sort_top_to_bottom else (int(index),),
                    )
                    chunk_index = 0
                    chunk_count = 0
                    for local_vertex_index in ordered_local_indices:
                        variant_count = estimate_mface_variant_count(int(local_vertex_index))
                        if variant_count > 31:
                            raise ValueError(
                                f'MFacePoint {base_key!r} on "{getattr(mesh_obj, "name", "Mesh")}" has one render vertex that exports to {variant_count} variants; SameVertBits can store at most 31.'
                            )
                        if chunk_count > 0 and chunk_count + int(variant_count) > 31:
                            chunk_index += 1
                            chunk_count = 0
                            auto_chunked_groups += 1
                        mface_group_key_by_local_vertex[int(local_vertex_index)] = (*base_key, int(chunk_index))
                        chunk_count += int(variant_count)
                # Oversized MFace runtime ranges are silently chunked to fit the
                # 5-bit SameVertBits field. This is an internal export fix, not
                # something that should show up as an export warning/report.

                local_vertex_order.sort(key=lambda vertex_index: (mface_group_key_by_local_vertex.get(int(vertex_index), (owner_name, 0, 0, 0, -1, 0.0, 0)), int(vertex_index)))

            loop_to_export_index: Dict[int, int] = {}
            variant_by_key: Dict[tuple, int] = {}
            mface_group_by_export_index: Dict[int, tuple] = {}

            def make_variant(local_vertex_index: int, loop_index: int = -1) -> int:
                local_vertex_index = int(local_vertex_index)
                if local_vertex_index not in vertex_base_data:
                    return 0
                position_raw, weight_tuple = vertex_base_data[local_vertex_index]
                primary, secondary, secondary_weight = weight_tuple
                normal = Vector((0.0, 0.0, 1.0))
                if loop_index >= 0:
                    try:
                        normal = Vector(mesh.loops[loop_index].normal)
                    except Exception:
                        normal = Vector((0.0, 0.0, 1.0))
                else:
                    try:
                        normal = Vector(mesh.vertices[local_vertex_index].normal)
                    except Exception:
                        normal = Vector((0.0, 0.0, 1.0))
                if normal.length <= 1e-8:
                    normal = Vector((0.0, 0.0, 1.0))
                else:
                    normal = normal.normalized()
                normal_raw = (_normal_to_i8(normal.x), _normal_to_i8(normal.y), _normal_to_i8(normal.z))

                uvx = uvy = 0
                if uv_layer is not None and loop_index >= 0:
                    uv = uv_layer[loop_index].uv
                    ps2_uv = self._ps2_uv_from_loop(mesh, local_vertex_index, int(loop_index), uv)
                    if ps2_uv is not None:
                        uvx, uvy = ps2_uv
                    else:
                        uvx = _encode_pc_uv_component(float(uv.x))
                        uvy = _encode_pc_uv_component(1.0 - float(uv.y))

                color = self._sample_color(color_attr, loop_index, local_vertex_index) if has_color_attr else (255, 255, 255, 128)
                key = (local_vertex_index, int(uvx), int(uvy), normal_raw, color if has_color_attr else None)
                existing = variant_by_key.get(key)
                if existing is not None:
                    return existing
                export_index = len(vertices)
                vertices.append(MVertex(index=export_index, position_raw=position_raw, normal_raw=normal_raw, segment=int(primary), uv_raw=(int(uvx), int(uvy))))
                vertex_colors.append(color)
                vertex_weights.append((int(primary), int(secondary), float(secondary_weight)))
                variant_by_key[key] = export_index
                if mface_group_key_by_local_vertex:
                    group_key = mface_group_key_by_local_vertex.get(int(local_vertex_index))
                    if group_key is not None:
                        mface_group_by_export_index[int(export_index)] = group_key
                return export_index

            for local_vertex_index in local_vertex_order:
                loop_indices = used_loops_by_vertex.get(local_vertex_index)
                if loop_indices:
                    for loop_index in loop_indices:
                        loop_to_export_index[int(loop_index)] = make_variant(local_vertex_index, int(loop_index))
                else:
                    make_variant(local_vertex_index, -1)

            for tri in mesh.loop_triangles:
                material_index = int(getattr(tri, 'material_index', 0) or 0)
                material = material_lookup[material_index] if 0 <= material_index < len(material_lookup) else None
                strip_key = self._strip_key(material, material_index, mesh_obj)
                strip = strips_by_material.get(strip_key)
                if strip is None:
                    ps2_raw_tpageid = -1
                    try:
                        ps2_raw_getter = getattr(self, '_material_ps2_raw_tpageid', None)
                        if ps2_raw_getter is not None:
                            ps2_raw_tpageid = int(ps2_raw_getter(material))
                    except Exception:
                        ps2_raw_tpageid = -1
                    strip = TextureStrip(offset=0, vertex_count=0, draw_group=get_material_draw_group(material), tpageid=self._material_tpageid(material), sort_push=0.0, scroll_offset=0.0, env_mapping=self._material_env_mapping(material), indices=[], ps2_tpageid_raw=ps2_raw_tpageid)
                    if material is not None and bool(get_material_scroll_enabled(material)):
                        raw_speed = get_material_scroll_speed(material)
                        strip.scroll_speeds.append(float(raw_speed))
                        strip.scroll_entry_count = 1
                        strip.scroll_num_tiles = 0
                        strip.scroll_tile = 0
                    strips_by_material[strip_key] = strip
                is_env = bool(material and get_material_env_mapping(material))
                is_eye = bool(material and get_material_eye_ref_env_mapping(material))
                for loop_index in tri.loops:
                    try:
                        vertex_index = loop_to_export_index[int(loop_index)]
                    except Exception:
                        local_vertex_index = int(mesh.loops[loop_index].vertex_index)
                        vertex_index = make_variant(local_vertex_index, int(loop_index))
                        loop_to_export_index[int(loop_index)] = vertex_index
                    strip.indices.append(vertex_index)
                    if is_env:
                        env_indices.append(vertex_index)
                    if is_eye:
                        eye_ref_indices.append(vertex_index)

            if mface_triangles:
                embedded_faces = self._mface_faces_from_embedded(
                    mesh_obj,
                    mface_triangles,
                    mface_group_by_export_index,
                    mface_group_key_by_local_vertex,
                    mface_height_by_local_vertex,
                    mface_sort_top_to_bottom,
                )
                for embedded_face in embedded_faces:
                    embedded_face.index = len(faces)
                    faces.append(embedded_face)
            if mface_group_by_export_index:
                mface_group_by_model_vertex.update(mface_group_by_export_index)

        if overweight_stats:
            total_overweight = sum(int(count) for count in overweight_stats.values())
            shown_items = sorted(overweight_stats.items(), key=lambda item: (-int(item[1]), str(item[0])))[:5]
            detail = ', '.join(f'{name}: {int(count)}' for name, count in shown_items)
            if len(overweight_stats) > len(shown_items):
                detail += ', ...'
            model_name = str(getattr(model_root, 'name', '') or 'Model')
            message = (
                f'Model "{model_name}" had {total_overweight} vertex/vertices with more than '
                f'{_MODEL_MAX_WEIGHTS_PER_VERTEX} bone weights; export kept the two strongest weights and renormalized them'
            )
            if detail:
                message += f' ({detail})'
            logger.warning(message)
            self.warnings.append(message)

        strips = list(strips_by_material.values())
        for strip in strips:
            strip.vertex_count = len(strip.indices)

        if has_flat_shaded_material:
            pre_optimization_weights = list(vertex_weights)
            vertex_weights, optimization_message = self._optimize_flat_shaded_virt_segment_weights(
                model_root, vertex_weights, segments, _MODEL_FLAT_SHADED_MAX_VIRT_SEGMENTS
            )
            self._rebase_vertex_positions_for_primary_changes(
                vertices, pre_optimization_weights, vertex_weights, pivot_by_segment, model_scale
            )
            if optimization_message:
                logger.warning(optimization_message)
                self.warnings.append(optimization_message)

        vertices, vertex_colors, vertex_weights, faces, strips, env_indices, eye_ref_indices = self._reorder_model_vertices_by_bone_order(
            vertices, vertex_colors, vertex_weights, faces, strips, env_indices, eye_ref_indices, segments, mface_group_by_model_vertex
        )
        virt_segments = self._build_virt_segments(vertices, vertex_weights, segments, pivot_by_segment, model_scale)
        return vertices, faces, strips, vertex_colors if any_vertex_color else None, virt_segments, env_indices, eye_ref_indices, vertex_weights

    def _normalize_export_weight(self, weight_tuple: tuple[int, int, float], segment_count: int) -> tuple[int, int, float]:
        try:
            primary, secondary, weight = weight_tuple
        except Exception:
            primary, secondary, weight = 0, -1, 0.0
        primary = max(0, min(int(primary), max(0, int(segment_count) - 1)))
        secondary = int(secondary)
        weight = max(0.0, min(1.0, float(weight)))
        if secondary >= 0:
            secondary = max(0, min(secondary, max(0, int(segment_count) - 1)))
        if secondary < 0 or secondary == primary or weight <= 1e-6:
            return int(primary), -1, 0.0
        return int(primary), int(secondary), float(weight)

    def _estimate_virt_segment_count_from_weights(self, vertex_weights: Sequence[tuple[int, int, float]], segments: Sequence[Segment]) -> int:
        segment_count = max(1, len(segments or []))
        keys = set()
        for weight_tuple in vertex_weights or []:
            primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count)
            if secondary >= 0 and weight > 1e-6:
                keys.add((int(primary), int(secondary), round(float(weight), 6)))
        return len(keys)

    def _rebase_vertex_positions_for_primary_changes(
        self,
        vertices: Sequence[MVertex],
        old_vertex_weights: Sequence[tuple[int, int, float]],
        new_vertex_weights: Sequence[tuple[int, int, float]],
        pivot_by_segment: Dict[int, Vector],
        model_scale: Sequence[float],
    ) -> None:
        if not vertices:
            return
        sx = float(model_scale[0] or 1.0) if len(model_scale) > 0 else 1.0
        sy = float(model_scale[1] or 1.0) if len(model_scale) > 1 else 1.0
        sz = float(model_scale[2] or 1.0) if len(model_scale) > 2 else 1.0
        if abs(sx) <= 1e-9:
            sx = 1.0
        if abs(sy) <= 1e-9:
            sy = 1.0
        if abs(sz) <= 1e-9:
            sz = 1.0
        segment_count = max(1, (max(pivot_by_segment.keys()) + 1) if pivot_by_segment else 1)

        for vertex_index, vertex in enumerate(vertices):
            if vertex_index >= len(old_vertex_weights or []) or vertex_index >= len(new_vertex_weights or []):
                continue
            old_primary, _old_secondary, _old_weight = self._normalize_export_weight(old_vertex_weights[vertex_index], segment_count)
            new_primary, _new_secondary, _new_weight = self._normalize_export_weight(new_vertex_weights[vertex_index], segment_count)
            if int(old_primary) == int(new_primary):
                continue

            old_pivot = pivot_by_segment.get(int(old_primary), Vector((0.0, 0.0, 0.0)))
            new_pivot = pivot_by_segment.get(int(new_primary), Vector((0.0, 0.0, 0.0)))
            x, y, z = vertex.position_raw
            arm_local = Vector((
                old_pivot.x + (float(x) * sx),
                old_pivot.y + (float(y) * sy),
                old_pivot.z + (float(z) * sz),
            ))
            rebased = Vector((
                (arm_local.x - new_pivot.x) / sx,
                (arm_local.y - new_pivot.y) / sy,
                (arm_local.z - new_pivot.z) / sz,
            ))
            vertex.position_raw = (_int16(rebased.x), _int16(rebased.y), _int16(rebased.z))
            vertex.segment = int(new_primary)

    @staticmethod
    def _snap_weight_to_step(weight: float, step: float) -> float:
        step = max(1e-6, float(step or 0.01))
        value = max(0.0, min(1.0, float(weight)))
        snapped = math.floor((value / step) + 0.5) * step
        if snapped <= step * 0.5:
            snapped = 0.0
        elif snapped >= 1.0 - (step * 0.5):
            snapped = 1.0
        return round(max(0.0, min(1.0, snapped)), 6)

    @staticmethod
    def _collapse_weight_to_nearest_segment(primary: int, secondary: int, weight: float) -> tuple[int, int, float]:
        if int(secondary) >= 0 and float(weight) >= 0.5:
            return int(secondary), -1, 0.0
        return int(primary), -1, 0.0

    def _quantize_flat_shaded_weights(
        self,
        vertex_weights: Sequence[tuple[int, int, float]],
        segment_count: int,
        step: float,
    ) -> List[tuple[int, int, float]]:
        quantized: List[tuple[int, int, float]] = []
        for weight_tuple in vertex_weights or []:
            primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count)
            if secondary < 0 or weight <= 1e-6:
                quantized.append((primary, -1, 0.0))
                continue
            snapped = self._snap_weight_to_step(weight, step)
            if snapped <= 0.0 or snapped >= 1.0:
                quantized.append(self._collapse_weight_to_nearest_segment(primary, secondary, snapped))
            else:
                quantized.append((primary, secondary, snapped))
        return quantized

    def _weights_changed(
        self,
        before: Sequence[tuple[int, int, float]],
        after: Sequence[tuple[int, int, float]],
        segment_count: int,
    ) -> bool:
        if len(before or []) != len(after or []):
            return True
        for old, new in zip(before or [], after or []):
            old_primary, old_secondary, old_weight = self._normalize_export_weight(old, segment_count)
            new_primary, new_secondary, new_weight = self._normalize_export_weight(new, segment_count)
            if old_primary != new_primary or old_secondary != new_secondary:
                return True
            if abs(float(old_weight) - float(new_weight)) > 1e-6:
                return True
        return False

    def _optimize_flat_shaded_virt_segment_weights(
        self,
        model_root,
        vertex_weights: Sequence[tuple[int, int, float]],
        segments: Sequence[Segment],
        max_virt_segments: int,
    ) -> tuple[List[tuple[int, int, float]], Optional[str]]:
        max_virt_segments = max(0, int(max_virt_segments))
        segment_count = max(1, len(segments or []))
        normalized = [self._normalize_export_weight(weight_tuple, segment_count) for weight_tuple in (vertex_weights or [])]
        original_count = self._estimate_virt_segment_count_from_weights(normalized, segments)

        if max_virt_segments <= 0:
            optimized = [self._collapse_weight_to_nearest_segment(primary, secondary, weight) for primary, secondary, weight in normalized]
            final_count = self._estimate_virt_segment_count_from_weights(optimized, segments)
            return optimized, self._flat_shaded_optimization_message(model_root, original_count, final_count, max_virt_segments, 'all weighted vertices were baked to single bones')

        chosen_step = float(_MODEL_FLAT_SHADED_WEIGHT_STEPS[0])
        optimized = normalized
        final_count = original_count
        strategy_notes: List[str] = []

        # Always remove precision noise for Flat Shading. This intentionally
        # turns imported values like 0.05098 into 0.05, even when the current
        # VirtSegment count is already under the hard limit.
        for step in _MODEL_FLAT_SHADED_WEIGHT_STEPS:
            candidate = self._quantize_flat_shaded_weights(normalized, segment_count, float(step))
            candidate_count = self._estimate_virt_segment_count_from_weights(candidate, segments)
            optimized = candidate
            final_count = candidate_count
            chosen_step = float(step)
            if candidate_count <= max_virt_segments:
                break

        if chosen_step <= float(_MODEL_FLAT_SHADED_WEIGHT_STEPS[0]) + 1e-9:
            strategy_notes.append(f'weights snapped to {chosen_step:.2f} increments')
        else:
            strategy_notes.append(f'weights snapped to coarser {chosen_step:.2f} increments')

        if final_count > max_virt_segments:
            before_reduce = final_count
            optimized = self._reduce_weight_keys_to_limit(optimized, max_virt_segments, segment_count, chosen_step)
            final_count = self._estimate_virt_segment_count_from_weights(optimized, segments)
            if final_count < before_reduce:
                strategy_notes.append('low-impact bone-pair buckets were merged or baked')

        if final_count > max_virt_segments:
            before_collapse = final_count
            optimized = self._collapse_lowest_impact_weight_keys(optimized, segments, max_virt_segments)
            final_count = self._estimate_virt_segment_count_from_weights(optimized, segments)
            if final_count < before_collapse:
                strategy_notes.append('remaining low-impact weights were collapsed to the nearest single bone')

        if not self._weights_changed(normalized, optimized, segment_count):
            return list(normalized), None

        detail = '; '.join(dict.fromkeys(strategy_notes)) or 'vertex weights were optimized'
        message = self._flat_shaded_optimization_message(model_root, original_count, final_count, max_virt_segments, detail)
        return optimized, message

    @staticmethod
    def _flat_shaded_optimization_message(model_root, original_count: int, final_count: int, max_virt_segments: int, detail: str) -> str:
        model_name = getattr(model_root, 'name', '<unnamed model>')
        return (
            f'Model "{model_name}" uses Flat Shading, so export automatically optimized vertex weights '
            f'({detail}). VirtSegments: {original_count} -> {final_count} (limit {max_virt_segments}).'
        )

    def _reduce_weight_keys_to_limit(
        self,
        vertex_weights: Sequence[tuple[int, int, float]],
        max_keys: int,
        segment_count: int,
        snap_step: float = 0.01,
    ) -> List[tuple[int, int, float]]:
        if max_keys <= 0:
            return [self._collapse_weight_to_nearest_segment(primary, secondary, weight) for primary, secondary, weight in vertex_weights]

        pair_stats: Dict[tuple[int, int], Dict[str, object]] = {}
        for index, weight_tuple in enumerate(vertex_weights):
            primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count)
            if secondary < 0 or weight <= 1e-6:
                continue
            pair = (int(primary), int(secondary))
            stats = pair_stats.setdefault(pair, {'indices': [], 'impact': 0.0, 'weights': defaultdict(int)})
            stats['indices'].append(int(index))
            stats['impact'] = float(stats['impact']) + min(float(weight), 1.0 - float(weight))
            stats['weights'][round(float(weight), 6)] += 1

        if not pair_stats:
            return [self._normalize_export_weight(weight_tuple, segment_count) for weight_tuple in vertex_weights]

        # If there are already more bone pairs than the game allows as virtual
        # segments, keep the most visually significant pairs. A weight near
        # 0.0 or 1.0 is cheap to bake to a single bone, while a 0.5 blend is
        # more expensive to lose.
        sorted_pairs = sorted(
            pair_stats,
            key=lambda pair: (float(pair_stats[pair]['impact']), len(pair_stats[pair]['indices'])),
            reverse=True,
        )
        kept_pairs = set(sorted_pairs[:max_keys])

        working: List[tuple[int, int, float]] = []
        for weight_tuple in vertex_weights:
            primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count)
            if secondary >= 0 and (primary, secondary) not in kept_pairs:
                working.append(self._collapse_weight_to_nearest_segment(primary, secondary, weight))
            else:
                working.append((primary, secondary, weight))

        # Rebuild stats after dropping low-impact pairs.
        pair_stats = {}
        for index, weight_tuple in enumerate(working):
            primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count)
            if secondary < 0 or weight <= 1e-6:
                continue
            pair = (int(primary), int(secondary))
            stats = pair_stats.setdefault(pair, {'indices': [], 'impact': 0.0, 'weights': defaultdict(int)})
            stats['indices'].append(int(index))
            stats['impact'] = float(stats['impact']) + min(float(weight), 1.0 - float(weight))
            stats['weights'][round(float(weight), 6)] += 1

        if not pair_stats:
            return working

        allocations: Dict[tuple[int, int], int] = {pair: 1 for pair in pair_stats}
        remaining = max(0, int(max_keys) - len(pair_stats))
        while remaining > 0:
            candidates = [
                pair for pair, stats in pair_stats.items()
                if int(allocations[pair]) < len(stats['weights'])
            ]
            if not candidates:
                break
            candidates.sort(
                key=lambda pair: (len(pair_stats[pair]['weights']) - int(allocations[pair]), len(pair_stats[pair]['indices']), float(pair_stats[pair]['impact'])),
                reverse=True,
            )
            allocations[candidates[0]] += 1
            remaining -= 1

        replacements: Dict[tuple[int, int], Dict[float, float]] = {}
        for pair, stats in pair_stats.items():
            unique_weights = sorted((float(weight), int(count)) for weight, count in stats['weights'].items())
            bins = max(1, min(int(allocations.get(pair, 1)), len(unique_weights)))
            if len(unique_weights) <= bins:
                replacements[pair] = {round(weight, 6): self._snap_weight_to_step(weight, snap_step) for weight, _count in unique_weights}
                continue
            replacements[pair] = self._quantize_weight_values(unique_weights, bins, snap_step)

        optimized: List[tuple[int, int, float]] = []
        for weight_tuple in working:
            primary, secondary, weight = self._normalize_export_weight(weight_tuple, segment_count)
            if secondary < 0 or weight <= 1e-6:
                optimized.append((primary, -1, 0.0))
                continue
            pair = (int(primary), int(secondary))
            rounded_weight = round(float(weight), 6)
            mapped = replacements.get(pair, {}).get(rounded_weight, rounded_weight)
            mapped = self._snap_weight_to_step(mapped, snap_step)
            if mapped <= 0.0 or mapped >= 1.0:
                optimized.append(self._collapse_weight_to_nearest_segment(primary, secondary, mapped))
            else:
                optimized.append((primary, secondary, max(0.0, min(1.0, float(mapped)))))
        return optimized

    @staticmethod
    def _quantize_weight_values(unique_weights: Sequence[tuple[float, int]], bins: int, snap_step: float = 0.01) -> Dict[float, float]:
        bins = max(1, int(bins))
        weighted_total = sum(max(1, int(count)) for _weight, count in unique_weights)
        target_per_bin = max(1.0, float(weighted_total) / float(bins))
        groups: List[List[tuple[float, int]]] = []
        current: List[tuple[float, int]] = []
        current_count = 0
        remaining_bins = bins
        remaining_values = len(unique_weights)
        for weight, count in unique_weights:
            current.append((float(weight), int(count)))
            current_count += max(1, int(count))
            remaining_values -= 1
            if remaining_bins > 1 and current_count >= target_per_bin and remaining_values >= remaining_bins - 1:
                groups.append(current)
                current = []
                current_count = 0
                remaining_bins -= 1
        if current:
            groups.append(current)

        mapping: Dict[float, float] = {}
        step = max(1e-6, float(snap_step or 0.01))
        for group in groups:
            expanded_count = sum(max(1, int(count)) for _weight, count in group)
            midpoint = (expanded_count + 1) / 2.0
            cursor = 0
            representative = group[-1][0]
            for weight, count in group:
                cursor += max(1, int(count))
                if cursor >= midpoint:
                    representative = float(weight)
                    break
            representative = math.floor((max(0.0, min(1.0, float(representative))) / step) + 0.5) * step
            representative = round(max(0.0, min(1.0, representative)), 6)
            for weight, _count in group:
                mapping[round(float(weight), 6)] = representative
        return mapping

    def _collapse_lowest_impact_weight_keys(self, vertex_weights: Sequence[tuple[int, int, float]], segments: Sequence[Segment], max_keys: int) -> List[tuple[int, int, float]]:
        segment_count = max(1, len(segments or []))
        optimized = [self._normalize_export_weight(weight_tuple, segment_count) for weight_tuple in vertex_weights]
        while self._estimate_virt_segment_count_from_weights(optimized, segments) > max_keys:
            key_stats: Dict[tuple[int, int, float], Dict[str, object]] = {}
            for index, (primary, secondary, weight) in enumerate(optimized):
                if secondary < 0 or weight <= 1e-6:
                    continue
                key = (int(primary), int(secondary), round(float(weight), 6))
                stats = key_stats.setdefault(key, {'indices': [], 'impact': 0.0})
                stats['indices'].append(int(index))
                stats['impact'] = float(stats['impact']) + min(float(weight), 1.0 - float(weight))
            if not key_stats:
                break
            drop_key = min(key_stats, key=lambda key: (float(key_stats[key]['impact']), len(key_stats[key]['indices'])))
            for index in key_stats[drop_key]['indices']:
                primary, secondary, weight = optimized[index]
                optimized[index] = self._collapse_weight_to_nearest_segment(primary, secondary, weight)
        return optimized

    def _reorder_model_vertices_by_bone_order(
        self,
        vertices: List[MVertex],
        vertex_colors: List[Tuple[int, int, int, int]],
        vertex_weights: List[tuple[int, int, float]],
        faces: List[MFace],
        strips: List[TextureStrip],
        env_indices: List[int],
        eye_ref_indices: List[int],
        segments: Sequence[Segment],
        mface_group_by_vertex_index: Optional[Dict[int, tuple[str, int]]] = None,
    ) -> tuple[List[MVertex], List[Tuple[int, int, int, int]], List[tuple[int, int, float]], List[MFace], List[TextureStrip], List[int], List[int]]:
        if not vertices:
            return vertices, vertex_colors, vertex_weights, faces, strips, env_indices, eye_ref_indices

        segment_count = max(1, len(segments or []))

        def normalized_weight(old_index: int) -> tuple[int, int, float]:
            if 0 <= old_index < len(vertex_weights):
                return self._normalize_export_weight(vertex_weights[old_index], segment_count)
            vertex = vertices[old_index]
            return self._normalize_export_weight((int(getattr(vertex, 'segment', 0) or 0), -1, 0.0), segment_count)

        def mface_sort_key(old_index: int):
            if not mface_group_by_vertex_index:
                return (1, '', -1, -1, -1, 0.0, 0)
            group = mface_group_by_vertex_index.get(int(old_index))
            if group is None:
                return (1, '', int(old_index), -1, -1, 0.0, 0)
            try:
                if isinstance(group, tuple) and len(group) >= 7:
                    owner, point_rank, point_index, primary, secondary, weight, chunk_index = group[:7]
                    return (0, str(owner), int(point_rank), int(point_index), int(primary), int(secondary), round(float(weight), 6), int(chunk_index))
                if isinstance(group, tuple) and len(group) >= 6:
                    owner, point_rank, point_index, primary, secondary, weight = group[:6]
                    return (0, str(owner), int(point_rank), int(point_index), int(primary), int(secondary), round(float(weight), 6), 0)
                if isinstance(group, tuple) and len(group) >= 5:
                    owner, point_index, primary, secondary, weight = group[:5]
                    return (0, str(owner), int(point_index), int(point_index), int(primary), int(secondary), round(float(weight), 6), 0)
                if isinstance(group, tuple) and len(group) >= 2:
                    owner, group_index = group[:2]
                    return (0, str(owner), int(group_index), int(group_index), -1, -1, 0.0, 0)
            except Exception:
                pass
            return (0, str(group), 0, -1, -1, 0.0, 0)

        def sort_key(old_index: int) -> tuple:
            primary, secondary, weight = normalized_weight(old_index)
            is_weighted = 1 if secondary >= 0 and weight > 1e-6 else 0
            return (primary, is_weighted, secondary if is_weighted else -1, round(weight, 6) if is_weighted else 0.0, mface_sort_key(old_index), old_index)

        ordered_old_indices = sorted(range(len(vertices)), key=sort_key)
        old_to_new = {old_index: new_index for new_index, old_index in enumerate(ordered_old_indices)}

        if mface_group_by_vertex_index:
            new_indices_by_group: Dict[tuple[str, int], List[int]] = defaultdict(list)
            for old_index, group in mface_group_by_vertex_index.items():
                if int(old_index) not in old_to_new:
                    continue
                new_indices_by_group[group].append(int(old_to_new[int(old_index)]))
            for group, new_indices in new_indices_by_group.items():
                if not new_indices:
                    continue
                ordered = sorted(new_indices)
                if ordered[-1] - ordered[0] + 1 != len(ordered):
                    raise ValueError(
                        f'MFace point group {group!r} does not remain contiguous after bone/weight export ordering. '
                        'Split that MFace point by segment/weight, or keep all render vertices assigned to it on the same segment/weight.'
                    )

        if all(old_index == new_index for new_index, old_index in enumerate(ordered_old_indices)):
            for index, vertex in enumerate(vertices):
                vertex.index = index
            return vertices, vertex_colors, [normalized_weight(index) for index in range(len(vertices))], faces, strips, env_indices, eye_ref_indices

        reordered_vertices: List[MVertex] = []
        reordered_colors: List[Tuple[int, int, int, int]] = []
        reordered_weights: List[tuple[int, int, float]] = []
        for new_index, old_index in enumerate(ordered_old_indices):
            vertex = vertices[old_index]
            vertex.index = new_index
            reordered_vertices.append(vertex)
            if old_index < len(vertex_colors):
                reordered_colors.append(vertex_colors[old_index])
            reordered_weights.append(normalized_weight(old_index))

        remapped_faces: List[MFace] = []
        for face_index, face in enumerate(faces or []):
            try:
                v0 = old_to_new[int(face.v0)]
                v1 = old_to_new[int(face.v1)]
                v2 = old_to_new[int(face.v2)]
            except Exception:
                continue
            remapped_faces.append(MFace(index=face_index, v0=v0, v1=v1, v2=v2, same_vert_bits=int(face.same_vert_bits) & 0xFFFF))

        for strip in strips or []:
            strip.indices = [old_to_new[int(index)] for index in list(getattr(strip, 'indices', []) or []) if int(index) in old_to_new]
            strip.vertex_count = len(strip.indices)

        remapped_env = [old_to_new[int(index)] for index in list(env_indices or []) if int(index) in old_to_new]
        remapped_eye = [old_to_new[int(index)] for index in list(eye_ref_indices or []) if int(index) in old_to_new]
        return reordered_vertices, reordered_colors, reordered_weights, remapped_faces, strips, remapped_env, remapped_eye

    def _build_virt_segments(self, vertices: Sequence[MVertex], vertex_weights: Sequence[tuple[int, int, float]], segments: Sequence[Segment], pivot_by_segment: Dict[int, Vector], model_scale: Sequence[float]) -> List[VirtSegment]:
        virt_segments: List[VirtSegment] = []
        sx, sy, sz = (float(model_scale[0] or 1.0), float(model_scale[1] or 1.0), float(model_scale[2] or 1.0))
        run_start = None
        run_key = None

        def same_key(a, b) -> bool:
            if a is None or b is None:
                return False
            return int(a[0]) == int(b[0]) and int(a[1]) == int(b[1]) and abs(float(a[2]) - float(b[2])) <= 1e-5

        def flush_run(end_index: int) -> None:
            nonlocal run_start, run_key
            if run_start is None or run_key is None:
                return
            primary, secondary, weight = run_key
            virt_index = len(virt_segments)
            scaled_positions = []
            for vertex_index in range(run_start, end_index + 1):
                vertex = vertices[vertex_index]
                vertex.segment = len(segments) + virt_index
                x, y, z = vertex.position_raw
                scaled_positions.append((float(x) * sx, float(y) * sy, float(z) * sz))
            min_v, max_v = self._bounds_from_scaled_positions(scaled_positions)
            primary_pivot = pivot_by_segment.get(int(primary), Vector((0.0, 0.0, 0.0)))
            secondary_pivot = pivot_by_segment.get(int(secondary), Vector((0.0, 0.0, 0.0)))
            pivot = primary_pivot - secondary_pivot
            virt_segments.append(VirtSegment(virt_index=virt_index, min_v=min_v, max_v=max_v, pivot=(float(pivot.x), float(pivot.y), float(pivot.z), 1.0), flags=8, first_vertex=int(run_start), last_vertex=int(end_index), index=int(primary), weight_index=int(secondary), weight=max(0.0, min(1.0, float(weight)))))
            run_start = None
            run_key = None

        for vertex_index, (primary, secondary, weight) in enumerate(vertex_weights):
            if secondary < 0 or secondary == primary or weight <= 1e-6:
                vertices[vertex_index].segment = int(primary)
                flush_run(vertex_index - 1)
                continue
            key = (int(primary), int(secondary), round(float(weight), 6))
            if run_start is None:
                run_start = vertex_index
                run_key = key
            elif not same_key(key, run_key):
                flush_run(vertex_index - 1)
                run_start = vertex_index
                run_key = key
        flush_run(len(vertices) - 1)
        return virt_segments

    @staticmethod
    def _bone_is_ancestor(arm_obj, ancestor_index: int, child_index: int) -> bool:
        if arm_obj is None or getattr(arm_obj, 'data', None) is None:
            return False
        ancestor = arm_obj.data.bones.get(f'bone_{int(ancestor_index)}')
        child = arm_obj.data.bones.get(f'bone_{int(child_index)}')
        if ancestor is None or child is None:
            return False
        current = child.parent
        while current is not None:
            if current == ancestor:
                return True
            current = current.parent
        return False

    def _choose_primary_secondary(self, weighted_segments: Sequence[tuple[int, float]], arm_obj=None) -> tuple[int, int, float]:
        clean = [(int(index), float(weight)) for index, weight in weighted_segments if float(weight) > 1e-6]
        if not clean:
            return (0, -1, 0.0)
        # The PC model format supports exactly two effective weights per
        # vertex: the vertex segment plus one secondary VirtSegment weight.
        # Keep the two strongest Blender weights and discard the rest before
        # choosing which of the pair should be the primary/bind segment.
        clean.sort(key=lambda item: (-item[1], item[0]))
        clean = clean[:_MODEL_MAX_WEIGHTS_PER_VERTEX]
        if len(clean) == 1:
            return (clean[0][0], -1, 0.0)
        # TR7/TRA PC model vertices can reference only one base segment plus
        # one secondary weighted segment through a VirtSegment. Keep the two
        # strongest weights, but choose the bind/primary segment by skeleton
        # ancestry rather than by largest weight. Imported files often have a
        # secondary weight greater than 0.5; using the largest group as primary
        # flips the original VirtSegment.index/weightIndex pair.
        first, second = clean[0], clean[1]
        a, aw = first
        b, bw = second
        if self._bone_is_ancestor(arm_obj, a, b):
            primary, primary_weight, secondary, secondary_raw_weight = a, aw, b, bw
        elif self._bone_is_ancestor(arm_obj, b, a):
            primary, primary_weight, secondary, secondary_raw_weight = b, bw, a, aw
        elif a <= b:
            primary, primary_weight, secondary, secondary_raw_weight = a, aw, b, bw
        else:
            primary, primary_weight, secondary, secondary_raw_weight = b, bw, a, aw
        total = max(primary_weight + secondary_raw_weight, 1e-9)
        secondary_weight = secondary_raw_weight / total
        if secondary_weight <= 1e-6:
            return (primary, -1, 0.0)
        return (primary, secondary, max(0.0, min(1.0, float(secondary_weight))))

    def _vertex_segment_weights(self, mesh_obj, vertex_index: int, arm_obj=None, overweight_stats: Optional[Dict[str, int]] = None) -> tuple[int, int, float]:
        vertex = mesh_obj.data.vertices[vertex_index]
        weights: List[tuple[int, float]] = []
        for group_ref in getattr(vertex, 'groups', []) or []:
            try:
                group = mesh_obj.vertex_groups[group_ref.group]
            except Exception:
                continue
            match = _BONE_RE.match(group.name)
            if match is None:
                continue
            weight = float(group_ref.weight)
            if weight > 0.0:
                weights.append((int(match.group(1)), weight))
        if overweight_stats is not None and len(weights) > _MODEL_MAX_WEIGHTS_PER_VERTEX:
            try:
                mesh_name = str(getattr(mesh_obj, 'name', '') or getattr(getattr(mesh_obj, 'data', None), 'name', '') or 'Mesh')
            except Exception:
                mesh_name = 'Mesh'
            overweight_stats[mesh_name] = int(overweight_stats.get(mesh_name, 0)) + 1
        return self._choose_primary_secondary(weights, arm_obj)

    @staticmethod
    def _bounds_from_scaled_positions(positions: Sequence[tuple[float, float, float]]) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        if not positions:
            return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)
        xs = [float(value[0]) for value in positions]
        ys = [float(value[1]) for value in positions]
        zs = [float(value[2]) for value in positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        diagonal = math.sqrt(((max_x - min_x) * (max_x - min_x)) + ((max_y - min_y) * (max_y - min_y)) + ((max_z - min_z) * (max_z - min_z)))
        return (min_x, min_y, min_z, diagonal), (max_x, max_y, max_z, 0.0)

    def _finalize_segment_bounds(self, segments: List[Segment], vertices: Sequence[MVertex], vertex_weights: Sequence[tuple[int, int, float]], pivot_by_segment: Dict[int, Vector], model_scale: Sequence[float]) -> List[Segment]:
        sx, sy, sz = (float(model_scale[0] or 1.0), float(model_scale[1] or 1.0), float(model_scale[2] or 1.0))
        by_segment: Dict[int, List[MVertex]] = defaultdict(list)
        for vertex_index, vertex in enumerate(vertices):
            if 0 <= vertex_index < len(vertex_weights):
                segment_index = int(vertex_weights[vertex_index][0])
            else:
                segment_index = int(vertex.segment)
            if 0 <= segment_index < len(segments):
                by_segment[segment_index].append(vertex)
        result: List[Segment] = []
        for segment in segments:
            verts = by_segment.get(int(segment.index), [])
            if verts:
                scaled_positions = []
                for vertex in verts:
                    x, y, z = vertex.position_raw
                    scaled_positions.append((float(x) * sx, float(y) * sy, float(z) * sz))
                min_v, max_v = self._bounds_from_scaled_positions(scaled_positions)
                first = min(v.index for v in verts)
                last = max(v.index for v in verts)
            else:
                min_v = (0.0, 0.0, 0.0, 0.0)
                max_v = (0.0, 0.0, 0.0, 0.0)
                first = 0
                last = -1
            result.append(Segment(index=segment.index, min_v=min_v, max_v=max_v, pivot=segment.pivot, flags=segment.flags, first_vertex=first, last_vertex=last, parent=segment.parent, hinfo=0))
        return result

    def _mface_embedded_data(self, mesh_obj, mesh) -> tuple[Dict[int, int], List[Tuple[int, int, int]]]:
        """Read embedded MFace authoring data from the render mesh.

        MFacePoint is a POINT-domain integer attribute.  Each render vertex
        belongs to exactly one authored wet/dirty point.  Export may split an
        authored point into multiple runtime ranges when the final vertex order
        requires it, for example when the point spans different bone/weight
        buckets.
        """
        try:
            bind_attr = getattr(mesh, 'attributes', None).get('MFacePoint') if getattr(mesh, 'attributes', None) is not None else None
        except Exception:
            bind_attr = None
        if bind_attr is None:
            return {}, []
        if getattr(bind_attr, 'domain', '') != 'POINT' or len(bind_attr.data) != len(mesh.vertices):
            raise ValueError(
                f'Mesh "{getattr(mesh_obj, "name", "Mesh")}" has an invalid MFacePoint attribute; run Create MFace before export.'
            )

        group_by_local_vertex: Dict[int, int] = {}
        invalid = 0
        for local_vertex_index, item in enumerate(bind_attr.data):
            try:
                point = int(item.value)
            except Exception:
                point = -1
            if point < 0:
                invalid += 1
                continue
            group_by_local_vertex[int(local_vertex_index)] = int(point)
        if invalid:
            raise ValueError(
                f'Mesh "{getattr(mesh_obj, "name", "Mesh")}" has {invalid} invalid MFacePoint value(s); run Create MFace before export.'
            )
        if len(group_by_local_vertex) != len(mesh.vertices):
            raise ValueError(
                f'Mesh "{getattr(mesh_obj, "name", "Mesh")}" is missing MFacePoint values on some vertices.'
            )

        try:
            mesh.calc_loop_triangles()
        except Exception:
            pass

        # Store triangle corners as render local vertex indices, not only as the
        # authored MFacePoint ids.  The final export step needs the source corner
        # vertex so it can choose the automatically split runtime point when one
        # authored point spans several bone/weight export buckets.
        triangles: List[Tuple[int, int, int]] = []
        for tri in getattr(mesh, 'loop_triangles', []) or []:
            local_vertices: List[int] = []
            for loop_index in tri.loops:
                try:
                    local_vertex_index = int(mesh.loops[int(loop_index)].vertex_index)
                    if local_vertex_index not in group_by_local_vertex:
                        local_vertex_index = -1
                    local_vertices.append(int(local_vertex_index))
                except Exception:
                    local_vertices.append(-1)
            if len(local_vertices) != 3 or min(local_vertices) < 0:
                continue
            triangles.append((int(local_vertices[0]), int(local_vertices[1]), int(local_vertices[2])))
        return group_by_local_vertex, triangles

    def _mface_faces_from_embedded(
        self,
        mesh_obj,
        triangles: Sequence[Tuple[int, int, int]],
        mface_group_by_export_index: Dict[int, tuple],
        mface_group_key_by_local_vertex: Dict[int, tuple],
        mface_height_by_local_vertex: Optional[Dict[int, float]] = None,
        mface_sort_top_to_bottom: bool = False,
    ) -> List[MFace]:
        if not triangles:
            return []
        if not mface_group_by_export_index:
            raise ValueError(
                f'Mesh "{getattr(mesh_obj, "name", "Mesh")}" has MFacePoint data but no exported vertices could be assigned to it.'
            )

        def authored_point_from_group(group) -> int:
            try:
                if isinstance(group, tuple) and len(group) >= 6:
                    return int(group[2])
                if isinstance(group, tuple) and len(group) >= 2:
                    return int(group[1])
            except Exception:
                pass
            return 0

        indices_by_group: Dict[tuple, List[int]] = defaultdict(list)
        for export_index, group in mface_group_by_export_index.items():
            indices_by_group[group].append(int(export_index))

        authored_point_to_runtime_groups: Dict[int, set] = defaultdict(set)
        for group in indices_by_group.keys():
            try:
                authored_point_to_runtime_groups[authored_point_from_group(group)].add(group)
            except Exception:
                pass

        range_by_group: Dict[tuple, Tuple[int, int]] = {}
        near_limit_groups: List[Tuple[tuple, int]] = []
        for group, export_indices in indices_by_group.items():
            if not export_indices:
                continue
            ordered = sorted(set(int(index) for index in export_indices))
            if not ordered:
                continue
            start = ordered[0]
            count = len(ordered)
            if ordered[-1] - start + 1 != count:
                raise ValueError(
                    f'MFacePoint {group!r} on "{getattr(mesh_obj, "name", "Mesh")}" maps to non-consecutive exported vertices after automatic split. '
                    'This usually means another export ordering pass is separating vertices inside one runtime MFace range.'
                )
            if count > 31:
                raise ValueError(
                    f'MFacePoint {group!r} on "{getattr(mesh_obj, "name", "Mesh")}" exports to {count} vertices after automatic split; SameVertBits can store at most 31.'
                )
            if count >= 24:
                near_limit_groups.append((group, int(count)))
            range_by_group[group] = (int(start), int(count))

        result: List[MFace] = []
        skipped_degenerate = 0
        missing_range_count = 0
        height_lookup = mface_height_by_local_vertex or {}

        def triangle_top_down_key(item):
            original_index, tri = item
            heights = [float(height_lookup.get(int(local_vertex), 0.0)) for local_vertex in tri]
            average_height = sum(heights) / max(1, len(heights))
            # Lower sort key comes first; negate so the MFace table is written
            # from top to bottom for the drying iterator.
            return (-float(average_height), int(original_index))

        triangle_items = sorted(enumerate(triangles), key=triangle_top_down_key) if bool(mface_sort_top_to_bottom) else enumerate(triangles)
        for _tri_original_index, tri in triangle_items:
            try:
                groups = [mface_group_key_by_local_vertex[int(local_vertex)] for local_vertex in tri]
            except KeyError as exc:
                raise ValueError(
                    f'Mesh "{getattr(mesh_obj, "name", "Mesh")}" contains an MFace triangle using vertex {int(exc.args[0])}, but that vertex has no MFacePoint export group.'
                ) from exc
            # If two corners resolve to the same runtime point, the MFace entry
            # would be degenerate.  This can happen after the user deliberately
            # merges adjacent render vertices into one MFacePoint, so skip it.
            if groups[0] == groups[1] or groups[1] == groups[2] or groups[0] == groups[2]:
                skipped_degenerate += 1
                continue
            try:
                ranges = [range_by_group[group] for group in groups]
            except KeyError as exc:
                missing_range_count += 1
                raise ValueError(
                    f'Mesh "{getattr(mesh_obj, "name", "Mesh")}" contains an MFace triangle using point {exc.args[0]!r}, but no render vertices export from that point.'
                ) from exc
            same_bits = (int(ranges[0][1]) & 0x1F) | ((int(ranges[1][1]) & 0x1F) << 5) | ((int(ranges[2][1]) & 0x1F) << 10)
            result.append(MFace(index=len(result), v0=int(ranges[0][0]), v1=int(ranges[1][0]), v2=int(ranges[2][0]), same_vert_bits=int(same_bits) & 0xFFFF))

        # Export-time MFace adaptation is intentionally quiet. The exporter may
        # split authored points into runtime ranges, skip degenerate generated
        # triangles, or chunk ranges to fit SameVertBits. Those are expected
        # internal conversions, not user-facing export warnings. Fatal structural
        # problems above still raise ValueError and stop export.
        return result

    @staticmethod
    def _point_int_attribute_values(mesh, names: Sequence[str], expected_count: int) -> Optional[List[int]]:
        attributes = getattr(mesh, 'attributes', None)
        if attributes is None:
            return None
        attr = None
        for name in names:
            attr = attributes.get(name)
            if attr is not None:
                break
        if attr is None:
            return None
        try:
            if str(getattr(attr, 'domain', '')).upper() != 'POINT':
                return None
            values: List[int] = []
            for item in attr.data:
                values.append(int(getattr(item, 'value', 0)) & 0xFFFF)
            if len(values) < int(expected_count):
                values.extend([0] * (int(expected_count) - len(values)))
            return values
        except Exception:
            return None

    @staticmethod
    def _active_color_attribute(mesh, preferred_name: str | None = None):
        color_attributes = getattr(mesh, 'color_attributes', None)
        if color_attributes is not None:
            if preferred_name:
                attr = color_attributes.get(preferred_name)
                if attr is not None:
                    return attr
                return None
            active = getattr(color_attributes, 'active_color', None) or getattr(color_attributes, 'active', None)
            if active is not None:
                return active
            for attr in color_attributes:
                return attr
        vertex_colors = getattr(mesh, 'vertex_colors', None)
        if vertex_colors:
            if preferred_name:
                return vertex_colors.get(preferred_name)
            active = vertex_colors.active
            if active is not None:
                return active
            for attr in vertex_colors:
                return attr
        return None

    @staticmethod
    def _sample_color(color_attr, loop_index: int, vertex_index: int) -> Tuple[int, int, int, int]:
        if color_attr is None:
            return (255, 255, 255, 128)
        try:
            domain = str(getattr(color_attr, 'domain', '')).upper()
            if domain == 'POINT':
                data_index = vertex_index
            elif loop_index >= 0:
                data_index = loop_index
            else:
                return (255, 255, 255, 128)
            color = color_attr.data[data_index].color
            return (_float_color_to_byte(color[0]), _float_color_to_byte(color[1]), _float_color_to_byte(color[2]), _float_alpha_to_trlau_byte(color[3] if len(color) > 3 else 1.0))
        except Exception:
            return (255, 255, 255, 128)

    @staticmethod
    def _material_flat_shading(material) -> bool:
        if material is None:
            return False
        try:
            if bool(getattr(material, 'trlau_ui_flat_shading')):
                return True
        except Exception:
            pass
        try:
            return bool(decode_tpage_flags(TRLAUModelExporter._material_tpageid(material)).get('flat_shading', 0))
        except Exception:
            return False

    @staticmethod
    def _material_tpageid(material) -> int:
        if material is None:
            return 0
        try:
            return _decode_signed_id(get_material_tpageid(material))
        except Exception:
            return 0

    @staticmethod
    def _material_int(material, key: str, default: int = 0) -> int:
        if material is None:
            return int(default)
        try:
            return int(material.get(key, default) or default)
        except Exception:
            return int(default)

    @staticmethod
    def _material_float(material, key: str, default: float = 0.0) -> float:
        if material is None:
            return float(default)
        try:
            return float(material.get(key, default) or default)
        except Exception:
            return float(default)

    @staticmethod
    def _material_env_mapping(material) -> int:
        if material is None:
            return 0
        try:
            if bool(get_material_eye_ref_env_mapping(material)):
                return 128
            if bool(get_material_env_mapping(material)):
                return 64
            return 0
        except Exception:
            return 0

    def _strip_key(self, material, material_index: int, mesh_obj=None) -> tuple:
        # TextureStripInfo is a draw list for a material, not for a face.
        # Imported models may carry stale per-strip metadata such as
        # trlau_texture_strip_index/trlau_material_group, so those fields must
        # not split the export back into one TextureStripInfo per source face or
        # source strip.  Group by the actual Blender material datablock first;
        # all triangles assigned to that material are appended to one index list.
        if material is not None:
            try:
                pointer = int(material.as_pointer())
                if pointer:
                    return ('material_ptr', pointer)
            except Exception:
                pass
            name = str(getattr(material, 'name_full', None) or getattr(material, 'name', '') or '')
            if name:
                return ('material_name', name)
        mesh_name = str(getattr(mesh_obj, 'name_full', None) or getattr(mesh_obj, 'name', '') or '')
        return ('slot', mesh_name, int(material_index))

    @staticmethod
    def _bone_for_index(arm_obj, bone_index: int):
        try:
            if arm_obj is None or arm_obj.data is None:
                return None
            return arm_obj.data.bones.get(f'bone_{int(bone_index)}')
        except Exception:
            return None

    def _segment_origin_local(self, arm_obj, bone_index: int) -> Vector:
        """Return the game segment origin in armature-local space.

        TRLAU HInfo and ClothSetup records are stored relative to the segment
        pivot/origin. Blender bones also carry an orientation, but their tail
        direction is an editor/display detail for this exporter and must not
        rotate authored helper offsets. Earlier code used bone.matrix_local,
        which silently assumed every bone shaft pointed along local +Y. If a
        user adjusted bone tails, HInfo and cloth collision data exported at a
        rotated/rebased position. Use the bone head only so exported data
        depends on the segment pivot, not on the visual bone tail axis.
        """
        bone = self._bone_for_index(arm_obj, bone_index)
        if bone is not None:
            try:
                return Vector(tuple(float(v) for v in bone.head_local))
            except Exception:
                pass
        return Vector((0.0, 0.0, 0.0))

    def _segment_origin_world_matrix(self, arm_obj, bone_index: int) -> Matrix:
        origin_local = self._segment_origin_local(arm_obj, bone_index)
        arm_world = Matrix.Identity(4)
        if arm_obj is not None:
            try:
                arm_world = arm_obj.matrix_world.copy()
            except Exception:
                arm_world = Matrix.Identity(4)
        return arm_world @ Matrix.Translation(origin_local)

    def _matrix_relative_to_segment_origin(self, world_matrix: Matrix, arm_obj, bone_index: int) -> Matrix:
        segment_world = self._segment_origin_world_matrix(arm_obj, bone_index)
        try:
            return segment_world.inverted_safe() @ world_matrix
        except Exception:
            try:
                return segment_world.inverted() @ world_matrix
            except Exception:
                return world_matrix.copy()

    def _point_relative_to_segment_origin(self, world_point: Vector, arm_obj, bone_index: int) -> Tuple[Vector, Vector]:
        """Return (segment-local offset, armature-local absolute position)."""
        if arm_obj is not None:
            try:
                arm_local = arm_obj.matrix_world.inverted_safe() @ world_point
            except Exception:
                try:
                    arm_local = arm_obj.matrix_world.inverted() @ world_point
                except Exception:
                    arm_local = Vector(tuple(world_point))
            origin_local = self._segment_origin_local(arm_obj, bone_index)
            local_offset = arm_local - origin_local
            return local_offset, origin_local + local_offset
        point = Vector(tuple(world_point))
        return point, point

    def _hinfo_local_matrix(self, obj, arm_obj, bone_index: int) -> Matrix:
        overrides = getattr(self, '_export_scene_matrix_overrides', None)
        if overrides is not None and id(obj) in overrides:
            world_matrix = self._export_object_world_matrix(obj)
            if arm_obj is not None:
                return self._matrix_relative_to_segment_origin(world_matrix, arm_obj, bone_index)
            try:
                parent = getattr(obj, 'parent', None)
                parent_world = self._export_object_world_matrix(parent) if parent is not None else Matrix.Identity(4)
                return parent_world.inverted_safe() @ world_matrix
            except Exception:
                return world_matrix

        # Fallback for non-standard export calls that did not snapshot helpers.
        # Still derive from world space and the segment origin so bone tail/roll
        # edits do not affect exported HInfo/Target transforms.
        try:
            world_matrix = self._export_object_world_matrix(obj)
        except Exception:
            return Matrix.Identity(4)
        if arm_obj is not None:
            return self._matrix_relative_to_segment_origin(world_matrix, arm_obj, bone_index)
        try:
            parent = getattr(obj, 'parent', None)
            parent_world = self._export_object_world_matrix(parent) if parent is not None else Matrix.Identity(4)
            return parent_world.inverted_safe() @ world_matrix
        except Exception:
            return world_matrix

    @staticmethod
    def _quat_to_xyzw(quat) -> Tuple[float, float, float, float]:
        try:
            q = quat.normalized()
            return (float(q.x), float(q.y), float(q.z), float(q.w))
        except Exception:
            return (0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def _matrix_close(a: Matrix, b: Matrix, tolerance: float = 1e-4) -> bool:
        try:
            for row in range(4):
                for col in range(4):
                    if abs(float(a[row][col]) - float(b[row][col])) > tolerance:
                        return False
            return True
        except Exception:
            return False

    def _hinfo_bone_index(self, obj, default: int = 0) -> int:
        try:
            for constraint in list(getattr(obj, 'constraints', []) or []):
                if getattr(constraint, 'type', '') != 'CHILD_OF':
                    continue
                subtarget = str(getattr(constraint, 'subtarget', '') or '')
                match = _BONE_RE.match(subtarget)
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        try:
            name = str(getattr(obj, 'name', '') or '')
            if '_B' in name:
                suffix = name.rsplit('_B', 1)[1].split('_', 1)[0].split('.', 1)[0]
                if suffix.isdigit():
                    return int(suffix)
        except Exception:
            pass
        return int(default)

    @staticmethod
    def _hinfo_index_from_name(obj, marker: str, default: int = 0) -> int:
        try:
            name = str(getattr(obj, 'name', '') or '')
            if marker in name:
                suffix = name.split(marker, 1)[1].split('.', 1)[0]
                if suffix.isdigit():
                    return int(suffix)
        except Exception:
            pass
        return int(default)

    def _collect_hmarkers(self, model_root, arm_obj) -> List[HMarker]:
        result: List[HMarker] = []
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if trlau_object_type(obj) != 'HMarker':
                continue
            bone_index = self._hinfo_bone_index(obj, 0)
            owner_segment = bone_index
            local_matrix = self._hinfo_local_matrix(obj, arm_obj, bone_index)
            loc, rot, _scale = local_matrix.decompose()
            pos = (float(loc.x), float(loc.y), float(loc.z))
            euler = rot.to_euler('ZYX')
            rot_tuple = (float(euler.x), float(euler.y), float(euler.z))
            marker_index = get_hinfo_int_prop(obj, 'trlau_marker_index', hmarker_index_from_name(obj, len(result)), min_value=0)
            result.append(HMarker(
                global_index=len(result),
                owner_segment=int(owner_segment),
                bone=bone_index,
                index=int(marker_index),
                position=pos,
                rotation=rot_tuple,
            ))
        return result

    def _collect_hspheres(self, model_root, arm_obj) -> List[HSphere]:
        result: List[HSphere] = []
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if trlau_object_type(obj) != 'HSphere':
                continue
            owner_segment = self._hinfo_bone_index(obj, 0)
            local_matrix = self._hinfo_local_matrix(obj, arm_obj, owner_segment)
            loc = local_matrix.to_translation()
            scale = local_matrix.to_scale()
            radius = max(int(round(max(abs(float(scale.x)), abs(float(scale.y)), abs(float(scale.z))))), 0)
            x = int(round(float(loc.x)))
            y = int(round(float(loc.y)))
            z = int(round(float(loc.z)))
            default_id = self._hinfo_index_from_name(obj, '_HSphere_', len(result))
            result.append(HSphere(
                global_index=len(result),
                owner_segment=int(owner_segment),
                flags=get_hinfo_int_prop(obj, 'trlau_hsphere_flags', 4352, min_value=0, max_value=0xFFFF),
                id=get_hinfo_int_prop(obj, 'trlau_hsphere_id', default_id, min_value=0, max_value=0xFF),
                rank=get_hinfo_int_prop(obj, 'trlau_hsphere_rank', 0, min_value=0, max_value=0xFF),
                radius=radius,
                x=x,
                y=y,
                z=z,
                radius_sq=get_hinfo_int_prop(obj, 'trlau_hsphere_radius_sq', radius * radius, min_value=0, max_value=0xFFFFFFFF),
                mass=get_hinfo_int_prop(obj, 'trlau_hsphere_mass', 100, min_value=0, max_value=0xFFFF),
                buoyancy_factor=get_hinfo_int_prop(obj, 'trlau_hsphere_buoyancy_factor', 100, min_value=0, max_value=0xFF),
                explosion_factor=get_hinfo_int_prop(obj, 'trlau_hsphere_explosion_factor', 100, min_value=0, max_value=0xFF),
                material_type=get_hinfo_int_prop(obj, 'trlau_hsphere_material_type', 0, min_value=0, max_value=0xFF),
                pad=get_hinfo_int_prop(obj, 'trlau_hsphere_pad', 0, min_value=0, max_value=0xFF),
                damage=get_hinfo_int_prop(obj, 'trlau_hsphere_damage', 0, min_value=-0x8000, max_value=0x7FFF),
            ))
        return result

    def _collect_hboxes(self, model_root, arm_obj) -> List[HBox]:
        result: List[HBox] = []
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if trlau_object_type(obj) != 'HBox':
                continue
            owner_segment = self._hinfo_bone_index(obj, 0)
            local_matrix = self._hinfo_local_matrix(obj, arm_obj, owner_segment)
            loc, rot, scale = local_matrix.decompose()
            dims = (
                abs(float(scale.x)) * 2.0,
                abs(float(scale.y)) * 2.0,
                abs(float(scale.z)) * 2.0,
                1.0,
            )
            position = (float(loc.x), float(loc.y), float(loc.z), 1.0)
            quaternion = self._quat_to_xyzw(rot)
            default_id = self._hinfo_index_from_name(obj, '_HBox_', len(result))
            result.append(HBox(
                global_index=len(result),
                owner_segment=int(owner_segment),
                flags=get_hinfo_int_prop(obj, 'trlau_hbox_flags', 4352, min_value=0, max_value=0xFFFF),
                id=get_hinfo_int_prop(obj, 'trlau_hbox_id', default_id, min_value=0, max_value=0xFF),
                rank=get_hinfo_int_prop(obj, 'trlau_hbox_rank', 0, min_value=0, max_value=0xFF),
                mass=get_hinfo_int_prop(obj, 'trlau_hbox_mass', 100, min_value=0, max_value=0xFFFF),
                buoyancy_factor=get_hinfo_int_prop(obj, 'trlau_hbox_buoyancy_factor', 100, min_value=0, max_value=0xFF),
                explosion_factor=get_hinfo_int_prop(obj, 'trlau_hbox_explosion_factor', 100, min_value=0, max_value=0xFF),
                material_type=get_hinfo_int_prop(obj, 'trlau_hbox_material_type', 0, min_value=0, max_value=0xFF),
                pad=get_hinfo_int_prop(obj, 'trlau_hbox_pad', 0, min_value=0, max_value=0xFF),
                damage=get_hinfo_int_prop(obj, 'trlau_hbox_damage', 0, min_value=-0x8000, max_value=0x7FFF),
                dimensions=dims,
                position=position,
                quaternion=quaternion,
            ))
        return result

    def _collect_hcapsules(self, model_root, arm_obj) -> List[HCapsule]:
        result: List[HCapsule] = []
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if trlau_object_type(obj) != 'HCapsule':
                continue
            owner_segment = self._hinfo_bone_index(obj, 0)
            local_matrix = self._hinfo_local_matrix(obj, arm_obj, owner_segment)
            loc, rot, scale = local_matrix.decompose()
            radius = max(int(round(max(abs(float(scale.x)), abs(float(scale.y))))), 0)
            length = max(int(round(abs(float(scale.z)) * 2.0)), 0)
            quat = self._quat_to_xyzw(rot)
            axis = rot @ Vector((0.0, 0.0, 1.0))
            if axis.length <= 1e-8:
                axis = Vector((0.0, 0.0, 1.0))
            else:
                axis.normalize()
            half = axis * (float(length) * 0.5)
            start_v = Vector((float(loc.x), float(loc.y), float(loc.z))) - half
            end_v = Vector((float(loc.x), float(loc.y), float(loc.z))) + half
            start = (float(start_v.x), float(start_v.y), float(start_v.z))
            end = (float(end_v.x), float(end_v.y), float(end_v.z))
            pos = (float(loc.x), float(loc.y), float(loc.z), 1.0)

            default_id = self._hinfo_index_from_name(obj, '_HCapsule_', len(result))
            result.append(HCapsule(
                global_index=len(result),
                owner_segment=owner_segment,
                flags=get_hinfo_int_prop(obj, 'trlau_hcapsule_flags', 4352, min_value=0, max_value=0xFFFF),
                id=get_hinfo_int_prop(obj, 'trlau_hcapsule_id', default_id, min_value=0, max_value=0xFF),
                rank=get_hinfo_int_prop(obj, 'trlau_hcapsule_rank', 0, min_value=0, max_value=0xFF),
                radius=radius,
                length=length,
                mass=get_hinfo_int_prop(obj, 'trlau_hcapsule_mass', 100, min_value=0, max_value=0xFFFF),
                buoyancy_factor=get_hinfo_int_prop(obj, 'trlau_hcapsule_buoyancy_factor', 100, min_value=0, max_value=0xFF),
                explosion_factor=get_hinfo_int_prop(obj, 'trlau_hcapsule_explosion_factor', 100, min_value=0, max_value=0xFF),
                material_type=get_hinfo_int_prop(obj, 'trlau_hcapsule_material_type', 0, min_value=0, max_value=0xFF),
                pad=get_hinfo_int_prop(obj, 'trlau_hcapsule_pad', 0, min_value=0, max_value=0xFF),
                damage=get_hinfo_int_prop(obj, 'trlau_hcapsule_damage', 0, min_value=-0x8000, max_value=0x7FFF),
                position=pos,
                quaternion=quat,
                start=start,
                end=end,
            ))
        return result

    @staticmethod
    def _markup_curve_points(obj, root_matrix_inv=None) -> List[Tuple[float, float, float, float]]:
        points: List[Tuple[float, float, float, float]] = []
        data = getattr(obj, 'data', None)
        if getattr(obj, 'type', None) != 'CURVE' or data is None:
            return points
        for spline in getattr(data, 'splines', []) or []:
            if getattr(spline, 'type', '') == 'BEZIER':
                source_points = getattr(spline, 'bezier_points', []) or []
                for point in source_points:
                    co = getattr(point, 'co', (0.0, 0.0, 0.0))
                    local = Vector((float(co[0]), float(co[1]), float(co[2])))
                    world = obj.matrix_world @ local
                    if root_matrix_inv is not None:
                        world = root_matrix_inv @ world
                    points.append((float(world.x), float(world.y), float(world.z), 1.0))
            else:
                source_points = getattr(spline, 'points', []) or []
                for point in source_points:
                    co = getattr(point, 'co', (0.0, 0.0, 0.0, 1.0))
                    local = Vector((float(co[0]), float(co[1]), float(co[2])))
                    world = obj.matrix_world @ local
                    if root_matrix_inv is not None:
                        world = root_matrix_inv @ world
                    pw = float(co[3]) if len(co) > 3 else 1.0
                    points.append((float(world.x), float(world.y), float(world.z), pw))
        return points

    @staticmethod
    def _markup_object_bbox_points(obj, root_matrix_inv=None) -> List[Tuple[float, float, float, float]]:
        points: List[Tuple[float, float, float, float]] = []
        bound_box = getattr(obj, 'bound_box', None)
        if not bound_box:
            return points
        try:
            raw_corners = [tuple(float(coord) for coord in corner[:3]) for corner in bound_box]
        except Exception:
            return points
        if not raw_corners:
            return points
        first = raw_corners[0]
        if all(all(abs(corner[index] - first[index]) <= 1e-6 for index in range(3)) for corner in raw_corners):
            return points
        for corner in raw_corners:
            local = Vector(corner)
            world = obj.matrix_world @ local
            if root_matrix_inv is not None:
                world = root_matrix_inv @ world
            points.append((float(world.x), float(world.y), float(world.z), 1.0))
        return points

    @classmethod
    def _markup_bbox_from_points(cls, points: Sequence[Sequence[float]], default_position: Sequence[float], *, relative_to_position: bool = False) -> Tuple[int, int, int, int, int, int]:
        sample_points = list(points or [])
        if not sample_points:
            sample_points = [default_position]
        if relative_to_position:
            base = (
                float(default_position[0]) if len(default_position) > 0 else 0.0,
                float(default_position[1]) if len(default_position) > 1 else 0.0,
                float(default_position[2]) if len(default_position) > 2 else 0.0,
            )
            raw_points = [
                cls._identity_float_triplet((
                    float(point[0]) - base[0],
                    float(point[1]) - base[1],
                    float(point[2]) - base[2],
                ))
                for point in sample_points
            ]
        else:
            raw_points = [cls._identity_float_triplet(point[:3]) for point in sample_points]
        min_x = min(float(point[0]) for point in raw_points)
        min_y = min(float(point[1]) for point in raw_points)
        min_z = min(float(point[2]) for point in raw_points)
        max_x = max(float(point[0]) for point in raw_points)
        max_y = max(float(point[1]) for point in raw_points)
        max_z = max(float(point[2]) for point in raw_points)
        return (
            cls._i16(min_x),
            cls._i16(min_y),
            cls._i16(min_z),
            cls._i16(max_x),
            cls._i16(max_y),
            cls._i16(max_z),
        )

    def _collect_markups(self, model_root, model_game: str) -> List[_ExportModelMarkupData]:
        if model_root is None:
            return []
        try:
            root_matrix_inv = model_root.matrix_world.inverted_safe()
        except Exception:
            root_matrix_inv = None
        game = self._normalize_model_game(model_game)
        markup_objects = [
            obj for obj in getattr(model_root, 'children_recursive', []) or []
            if not _should_ignore_model_export_object(obj) and trlau_object_type(obj) == 'Markup'
        ]
        markup_objects.sort(key=lambda obj: (int(obj.get('trlau_markup_index', 0) or 0), getattr(obj, 'name', '')))

        result: List[_ExportModelMarkupData] = []
        for default_index, obj in enumerate(markup_objects):
            position = self._relative_to_model_translation(obj, root_matrix_inv)
            flags = int(obj.get('trlau_markup_flags', 0) or 0)
            points = self._markup_curve_points(obj, root_matrix_inv)
            shape_points = points
            bbox_relative_to_position = False
            if flags & _MARKUP_BBOX_FLAGS:
                shape_points = points or self._markup_object_bbox_points(obj, root_matrix_inv)
                bbox_relative_to_position = True
            bbox = self._markup_bbox_from_points(
                shape_points,
                position,
                relative_to_position=bbox_relative_to_position,
            )
            result.append(_ExportModelMarkupData(
                index=int(obj.get('trlau_markup_index', default_index) or default_index),
                game=game,
                flags=flags,
                animated_segment=int(obj.get('trlau_markup_animated_segment', 0) or 0) if game == 'anniversary' else 0,
                position=position,
                bbox=bbox,
                polyline=[] if (flags & _MARKUP_BBOX_FLAGS) else points,
            ))
        result.sort(key=lambda item: int(item.index))
        for normalized_index, item in enumerate(result):
            item.index = normalized_index
        return result

    @staticmethod
    def _relative_to_model_translation(obj, root_matrix_inv=None) -> Tuple[float, float, float]:
        vector = obj.matrix_world.translation
        if root_matrix_inv is not None:
            vector = root_matrix_inv @ vector
        return (float(vector.x), float(vector.y), float(vector.z))

    def _collect_targets(self, model_root, arm_obj) -> List[Target]:
        result: List[Target] = []
        for obj in getattr(model_root, 'children_recursive', []) or []:
            if _should_ignore_model_export_object(obj):
                continue
            if trlau_object_type(obj) != 'Target':
                continue
            segment = int(obj.get('trlau_segment', self._hinfo_bone_index(obj, 0)) or 0)
            local_matrix = self._hinfo_local_matrix(obj, arm_obj, segment)
            loc, rot, _scale = local_matrix.decompose()
            euler = rot.to_euler('ZYX')
            result.append(Target(
                global_index=len(result),
                segment=segment,
                flags=int(obj.get('trlau_target_flags', 0) or 0),
                position=(float(loc.x), float(loc.y), float(loc.z)),
                rotation=(float(euler.x), float(euler.y), float(euler.z)),
                unique_id=int(obj.get('trlau_target_unique_id', 0) or 0),
            ))
        return result

    @staticmethod
    def _tuple_prop(obj, key: str, size: int, default=None):
        if default is None:
            default = tuple(0.0 for _ in range(size))
        value = obj.get(key, default)
        if not isinstance(value, Sequence):
            return tuple(float(v) for v in default)
        result = list(value[:size])
        while len(result) < size:
            result.append(default[len(result)] if len(default) > len(result) else 0.0)
        return tuple(float(v) for v in result)

    def _collect_bone_mirrors(self, model_root) -> List[BoneMirrorEntry]:
        entries = []
        collection = getattr(model_root, 'trlau_bone_mirror_entries', None)
        if collection is None:
            return entries
        for entry in collection:
            entries.append(BoneMirrorEntry(bone1=int(entry.bone1), bone2=int(entry.bone2), count=int(entry.count)))
        return entries

    @staticmethod
    def _generate_max_radius(model: ModelData) -> float:
        def accumulated_pivot(segment_index: int) -> tuple[float, float, float]:
            result = [0.0, 0.0, 0.0]
            visited: set[int] = set()
            current = int(segment_index)
            while 0 <= current < len(model.segments) and current not in visited:
                visited.add(current)
                pivot = model.segments[current].pivot
                result[0] += float(pivot[0])
                result[1] += float(pivot[1])
                result[2] += float(pivot[2])
                parent = int(model.segments[current].parent)
                if parent < 0 or parent == current:
                    break
                current = parent
            return (result[0], result[1], result[2])

        max_sq = 0.0
        sx, sy, sz, _sw = model.model_scale
        for vertex in model.vertices:
            segment_index = int(vertex.segment)
            bind_segment = segment_index
            if segment_index >= len(model.segments):
                virt_index = segment_index - len(model.segments)
                if 0 <= virt_index < len(model.virt_segments):
                    bind_segment = int(model.virt_segments[virt_index].index)
            tx, ty, tz = accumulated_pivot(bind_segment)
            x, y, z = vertex.position_raw
            px = (float(x) * float(sx)) + tx
            py = (float(y) * float(sy)) + ty
            pz = (float(z) * float(sz)) + tz
            radius_sq = (px * px) + (py * py) + (pz * pz)
            if radius_sq > max_sq:
                max_sq = radius_sq
        return math.sqrt(max_sq) if max_sq > 0.0 else 0.0

    def _export_model_textures(self, collection, directory: Path, section_list: _SectionList) -> List[Path]:
        clear_texture_export_cache()
        written: List[Path] = []
        seen: set[int] = set()
        for obj in _iter_collection_objects(collection):
            if _should_ignore_model_export_object(obj):
                continue
            if getattr(obj, 'type', None) != 'MESH':
                continue
            mesh = getattr(obj, 'data', None)
            if mesh is None:
                continue
            for material in getattr(mesh, 'materials', []) or []:
                if material is None:
                    continue
                tpageid = self._material_tpageid(material)
                texture_id = int(decode_tpage_flags(tpageid).get('texture_id', 0) or 0)
                if texture_id <= 0 or texture_id in seen:
                    continue
                image = self._find_material_image_texture(material)
                if image is None:
                    continue
                pcd_data = image_to_pcd_bytes(image, texture_id=texture_id, material_name=getattr(material, 'name', ''))
                section_index, filename = self._texture_section_for_id(directory, section_list, texture_id)
                buffer = _StandaloneSectionBuffer(section_index=section_index, filename=filename, section_type=5, section_id=texture_id, spec_mask=0xFFFFFFFF)
                buffer.data.extend(pcd_data)
                out_path = directory / filename
                _write_standalone_section_file(out_path, buffer)
                section_list.ensure_index(section_index, filename)
                written.append(out_path)
                seen.add(texture_id)
        return written

    def _texture_section_for_id(self, directory: Path, section_list: _SectionList, texture_id: int) -> tuple[int, str]:
        suffix = f'_{int(texture_id):x}.pcd'.lower()
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file() and candidate.name.lower().endswith(suffix):
                return _section_index_from_filename(candidate.name), candidate.name

        # Direct DRM export works from a temporary extraction that usually has no
        # sectionList.txt.  Allocating from len(section_list.entries) can then
        # collide with untouched sections that already exist in the extracted DRM
        # folder, producing duplicate names such as 30_0.gnc and 30_<id>.pcd and
        # making the repacker fail.  Allocate after every section index already
        # present on disk as well as every index recorded in sectionList.txt.
        used_indices = set(section_list.used_indices())
        for candidate in directory.iterdir():
            if not candidate.is_file():
                continue
            match = _SECTION_NAME_RE.match(candidate.name)
            if match is not None:
                used_indices.add(int(match.group('index')))
        index = (max(used_indices) + 1) if used_indices else 0
        filename = f'{index}_{int(texture_id):x}.pcd'
        section_list.ensure_index(index, filename)
        return index, filename

    @staticmethod
    def _find_material_image_texture(material):
        try:
            if material.use_nodes and material.node_tree is not None:
                for node in material.node_tree.nodes:
                    if getattr(node, 'bl_idname', '') == 'ShaderNodeTexImage' and getattr(node, 'image', None) is not None:
                        return node.image
        except Exception:
            pass
        return None
