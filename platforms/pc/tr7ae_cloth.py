from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...core.binary import BinaryReader
from ...core.log import logger
from ..common.section import SectionContextCache, SectionContext, resolve_pointer


RD_SETUP_LIST_FIELD_OFFSET = 0x7C
_CLOTH_SETUP_SIZE = 76
_CLOTH_POINT_SIZE = 20
_CLOTH_JOINT_MAP_SIZE = 40
_FD_DISTANCE_RULE_SIZE = 16
_FD_CAPSULE_RULE_SIZE = 12
_FD_PLANE_RULE_SIZE = 8
_FD_HAND_RULE_SIZE = 8
_FD_COLLIDE_RULE_SIZE = 8
_FD_SPHERE_COLLIDE_RULE_SIZE = 8
_FD_SIDE_COLLIDE_RULE_SIZE = 4
_FD_POINT_MOVE_RULE_SIZE = 4

_CLOTH_POINTER_NAMES = (
    'points',
    'maps',
    'distRules',
    'capsuleRules',
    'planeRules',
    'handRules',
    'collideRules',
    'sphereCollideRules',
    'sideCollideRules',
    'pointMoveRules',
)


@dataclass(slots=True)
class ClothPoint:
    segment: int
    flags: int
    jointOrder: int
    upTo: int
    x: float
    y: float
    z: float


@dataclass(slots=True)
class ClothJointMap:
    segment: int
    flags: int
    axis: int
    jointOrder: int
    center: int
    points: Tuple[int, int, int, int]
    xMin: float
    xMax: float
    yMin: float
    yMax: float
    zMin: float
    zMax: float


@dataclass(slots=True)
class FDDistanceRule:
    point: Tuple[int, int]
    flags: Tuple[int, int]
    minDist: float
    maxDist: float


# Once again, wow this works
@dataclass(slots=True)
class FDCapsuleRule:
    point: Tuple[int, int]
    flags: Tuple[int, int]
    radius: float


@dataclass(slots=True)
class FDPlaneRule:
    point: Tuple[int]
    flags: Tuple[int]
    planeSegment: Tuple[int, int]


@dataclass(slots=True)
class FDHandednessRule:
    movePt: int
    point: Tuple[int, int, int]


@dataclass(slots=True)
class FDCollideRule:
    point: int
    radius: float


@dataclass(slots=True)
class FDSphereCollideRule:
    point: int
    radius: float


@dataclass(slots=True)
class FDSideCollideRule:
    point: Tuple[int, int]


@dataclass(slots=True)
class FDPointMoveRule:
    point: int
    count: int


@dataclass(slots=True)
class ClothSetup:
    index: int = 0
    gravity: float = 10.0
    drag: float = 0.1
    windResponse: float = 0.1
    flags: int = 0
    positionPoint: int = 0
    points: List[ClothPoint] = field(default_factory=list)
    maps: List[ClothJointMap] = field(default_factory=list)
    distRules: List[FDDistanceRule] = field(default_factory=list)
    capsuleRules: List[FDCapsuleRule] = field(default_factory=list)
    planeRules: List[FDPlaneRule] = field(default_factory=list)
    handRules: List[FDHandednessRule] = field(default_factory=list)
    collideRules: List[FDCollideRule] = field(default_factory=list)
    sphereCollideRules: List[FDSphereCollideRule] = field(default_factory=list)
    sideCollideRules: List[FDSideCollideRule] = field(default_factory=list)
    pointMoveRules: List[FDPointMoveRule] = field(default_factory=list)
    source_object_path: str = ''
    list_section_index: int = -1
    list_local_offset: int = 0
    setup_section_index: int = -1
    setup_local_offset: int = 0
    section_type: int = 0
    section_id: int = 0
    skip_flags: int = 0
    version_id: int = 0
    has_debug_info: int = 0
    resource_type: int = 0
    spec_mask: int = 0xFFFFFFFF
    pointer_type_specifics: Dict[str, int] = field(default_factory=dict)

    @property
    def numPoints(self) -> int:
        return len(self.points)

    @property
    def numMaps(self) -> int:
        return len(self.maps)

    @property
    def numDistRules(self) -> int:
        return len(self.distRules)

    @property
    def numCapsuleRules(self) -> int:
        return len(self.capsuleRules)

    @property
    def numPlaneRules(self) -> int:
        return len(self.planeRules)

    @property
    def numHandRules(self) -> int:
        return len(self.handRules)

    @property
    def numCollides(self) -> int:
        return len(self.collideRules)

    @property
    def numSphereCollides(self) -> int:
        return len(self.sphereCollideRules)

    @property
    def numSideCollides(self) -> int:
        return len(self.sideCollideRules)

    @property
    def numPointMoveRules(self) -> int:
        return len(self.pointMoveRules)


def _section_index_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    try:
        return int(stem.split('_', 1)[0])
    except Exception:
        return -1


def _safe_read_array(ctx: SectionContext, absolute_offset: int, count: int, item_size: int) -> bool:
    if count <= 0:
        return True
    return ctx.data_start <= absolute_offset and absolute_offset + (count * item_size) <= ctx.data_end


def _read_pointer(ctx: SectionContext, local_offset: int) -> int:
    ctx.reader.seek(ctx.data_start + int(local_offset))
    return int(ctx.reader.u32())


def _resolve_optional_pointer(cache: SectionContextCache, ctx: SectionContext, field_local_offset: int, pointer_value: int) -> tuple[SectionContext, int]:
    relocation = ctx.section_info.relocations_by_offset.get(int(field_local_offset))
    if relocation is None and int(pointer_value) == 0:
        return ctx, 0
    return resolve_pointer(cache, ctx, int(field_local_offset), int(pointer_value))


def _signed_i16(value: int) -> int:
    value = int(value)
    if value > 32767:
        value -= 65536
    return max(-32768, min(32767, value))


def _reloc_type_specific(ctx: SectionContext, field_local_offset: int, default: int = 0) -> int:
    relocation = ctx.section_info.relocations_by_offset.get(int(field_local_offset))
    if relocation is None:
        return int(default)
    try:
        return _signed_i16(int(relocation.type_specific))
    except Exception:
        return int(default)


def _section_info_metadata(ctx: SectionContext) -> dict:
    info = ctx.section_info
    return {
        'section_type': int(getattr(info, 'section_type', 0)),
        'section_id': int(getattr(info, 'section_id', 0)),
        'skip_flags': int(getattr(info, 'skip_flags', 0)),
        'version_id': int(getattr(info, 'version_id', 0)),
        'has_debug_info': int(getattr(info, 'has_debug_info', 0)),
        'resource_type': int(getattr(info, 'resource_type', 0)),
        'spec_mask': int(getattr(info, 'spec_mask', 0xFFFFFFFF)),
    }


class TRClothParser:

    def __init__(self, object_filepath: str, endian: str = '<'):
        self.object_filepath = str(object_filepath)
        self.endian = endian

    def parse(self, max_setups: int = 64) -> List[ClothSetup]:
        cache = SectionContextCache(self.object_filepath, endian=self.endian)
        try:
            object_ctx = cache.get_root_context()
            if object_ctx.data_size < RD_SETUP_LIST_FIELD_OFFSET + 4:
                return []
            raw_list_pointer = _read_pointer(object_ctx, RD_SETUP_LIST_FIELD_OFFSET)
            if raw_list_pointer == 0 and RD_SETUP_LIST_FIELD_OFFSET not in object_ctx.section_info.relocations_by_offset:
                return []

            list_ctx, list_abs = resolve_pointer(cache, object_ctx, RD_SETUP_LIST_FIELD_OFFSET, raw_list_pointer)
            if list_abs <= 0 or not (list_ctx.data_start <= list_abs < list_ctx.data_end):
                return []

            setups: List[ClothSetup] = []
            list_local_base = int(list_abs - list_ctx.data_start)
            for setup_index in range(max(0, int(max_setups))):
                entry_local = list_local_base + (setup_index * 4)
                if entry_local + 4 > list_ctx.data_size:
                    break
                raw_setup_pointer = _read_pointer(list_ctx, entry_local)
                has_relocation = entry_local in list_ctx.section_info.relocations_by_offset
                if raw_setup_pointer == 0 and not has_relocation:
                    break
                setup_ctx, setup_abs = _resolve_optional_pointer(cache, list_ctx, entry_local, raw_setup_pointer)
                if setup_abs <= 0 or not (setup_ctx.data_start <= setup_abs + _CLOTH_SETUP_SIZE <= setup_ctx.data_end):
                    logger.warning('Skipping invalid ClothSetup pointer %d in %s', setup_index, Path(self.object_filepath).name)
                    continue
                setup = self._parse_setup(cache, setup_ctx, setup_abs, setup_index)
                setup.source_object_path = str(Path(self.object_filepath))
                setup.list_section_index = _section_index_from_filename(list_ctx.file_name)
                setup.list_local_offset = int(entry_local)
                setup.setup_section_index = _section_index_from_filename(setup_ctx.file_name)
                setup.setup_local_offset = int(setup_abs - setup_ctx.data_start)
                setups.append(setup)
            return setups
        finally:
            cache.close()

    def _parse_setup(self, cache: SectionContextCache, ctx: SectionContext, setup_abs: int, setup_index: int) -> ClothSetup:
        br = ctx.reader
        br.seek(setup_abs)
        gravity = float(br.f32())
        drag = float(br.f32())
        wind = float(br.f32())
        flags = int(br.u16())
        position_point = int(br.u16())
        counts = [int(br.u16()) for _ in range(10)]
        pointer_field_base = int(setup_abs - ctx.data_start) + 36
        raw_pointers = [int(br.u32()) for _ in range(10)]

        resolved: Dict[str, tuple[SectionContext, int]] = {}
        pointer_type_specifics: Dict[str, int] = {}
        for i, name in enumerate(_CLOTH_POINTER_NAMES):
            field_local = pointer_field_base + (i * 4)
            pointer_type_specifics[name] = _reloc_type_specific(ctx, field_local, 0)
            try:
                resolved[name] = _resolve_optional_pointer(cache, ctx, field_local, raw_pointers[i])
            except Exception as exc:
                logger.warning('Could not resolve ClothSetup.%s pointer in %s: %s', name, ctx.file_name, exc)
                resolved[name] = (ctx, 0)

        setup = ClothSetup(
            index=int(setup_index),
            gravity=gravity,
            drag=drag,
            windResponse=wind,
            flags=flags,
            positionPoint=position_point,
            pointer_type_specifics=pointer_type_specifics,
            **_section_info_metadata(ctx),
        )

        num_points, num_maps, num_dist, num_capsule, num_plane, num_hand, num_collide, num_sphere, num_side, num_move = counts

        points_ctx, points_abs = resolved['points']
        if _safe_read_array(points_ctx, points_abs, num_points, _CLOTH_POINT_SIZE):
            brp = points_ctx.reader
            brp.seek(points_abs)
            for _ in range(num_points):
                setup.points.append(ClothPoint(
                    segment=int(brp.u16()),
                    flags=int(brp.u16()),
                    jointOrder=int(brp.u16()),
                    upTo=int(brp.u16()),
                    x=float(brp.f32()),
                    y=float(brp.f32()),
                    z=float(brp.f32()),
                ))

        maps_ctx, maps_abs = resolved['maps']
        if _safe_read_array(maps_ctx, maps_abs, num_maps, _CLOTH_JOINT_MAP_SIZE):
            brm = maps_ctx.reader
            brm.seek(maps_abs)
            for _ in range(num_maps):
                segment = int(brm.u16())
                map_flags = int(brm.u16())
                axis = int(brm.u8())
                joint_order = int(brm.u8())
                center = int(brm.u16())
                pts = tuple(int(brm.u16()) for _ in range(4))
                setup.maps.append(ClothJointMap(
                    segment=segment,
                    flags=map_flags,
                    axis=axis,
                    jointOrder=joint_order,
                    center=center,
                    points=pts,
                    xMin=float(brm.f32()),
                    xMax=float(brm.f32()),
                    yMin=float(brm.f32()),
                    yMax=float(brm.f32()),
                    zMin=float(brm.f32()),
                    zMax=float(brm.f32()),
                ))

        dist_ctx, dist_abs = resolved['distRules']
        if _safe_read_array(dist_ctx, dist_abs, num_dist, _FD_DISTANCE_RULE_SIZE):
            brd = dist_ctx.reader
            brd.seek(dist_abs)
            for _ in range(num_dist):
                p0, p1, f0, f1 = int(brd.u16()), int(brd.u16()), int(brd.u16()), int(brd.u16())
                setup.distRules.append(FDDistanceRule(
                    point=(p0, p1),
                    flags=(f0, f1),
                    minDist=float(brd.f32()),
                    maxDist=float(brd.f32()),
                ))

        capsule_ctx, capsule_abs = resolved['capsuleRules']
        if _safe_read_array(capsule_ctx, capsule_abs, num_capsule, _FD_CAPSULE_RULE_SIZE):
            brc = capsule_ctx.reader
            brc.seek(capsule_abs)
            for _ in range(num_capsule):
                p0, p1, f0, f1 = int(brc.u16()), int(brc.u16()), int(brc.u16()), int(brc.u16())
                setup.capsuleRules.append(FDCapsuleRule(point=(p0, p1), flags=(f0, f1), radius=float(brc.f32())))

        plane_ctx, plane_abs = resolved['planeRules']
        if _safe_read_array(plane_ctx, plane_abs, num_plane, _FD_PLANE_RULE_SIZE):
            brpl = plane_ctx.reader
            brpl.seek(plane_abs)
            for _ in range(num_plane):
                setup.planeRules.append(FDPlaneRule(
                    point=(int(brpl.u16()),),
                    flags=(int(brpl.u16()),),
                    planeSegment=(int(brpl.i16()), int(brpl.i16())),
                ))

        hand_ctx, hand_abs = resolved['handRules']
        if _safe_read_array(hand_ctx, hand_abs, num_hand, _FD_HAND_RULE_SIZE):
            brh = hand_ctx.reader
            brh.seek(hand_abs)
            for _ in range(num_hand):
                setup.handRules.append(FDHandednessRule(
                    movePt=int(brh.u16()),
                    point=(int(brh.u16()), int(brh.u16()), int(brh.u16())),
                ))

        collide_ctx, collide_abs = resolved['collideRules']
        if _safe_read_array(collide_ctx, collide_abs, num_collide, _FD_COLLIDE_RULE_SIZE):
            brcol = collide_ctx.reader
            brcol.seek(collide_abs)
            for _ in range(num_collide):
                setup.collideRules.append(FDCollideRule(point=int(brcol.u32()), radius=float(brcol.f32())))

        sphere_ctx, sphere_abs = resolved['sphereCollideRules']
        if _safe_read_array(sphere_ctx, sphere_abs, num_sphere, _FD_SPHERE_COLLIDE_RULE_SIZE):
            brs = sphere_ctx.reader
            brs.seek(sphere_abs)
            for _ in range(num_sphere):
                setup.sphereCollideRules.append(FDSphereCollideRule(point=int(brs.u32()), radius=float(brs.f32())))

        side_ctx, side_abs = resolved['sideCollideRules']
        if _safe_read_array(side_ctx, side_abs, num_side, _FD_SIDE_COLLIDE_RULE_SIZE):
            brside = side_ctx.reader
            brside.seek(side_abs)
            for _ in range(num_side):
                setup.sideCollideRules.append(FDSideCollideRule(point=(int(brside.u16()), int(brside.u16()))))

        move_ctx, move_abs = resolved['pointMoveRules']
        if _safe_read_array(move_ctx, move_abs, num_move, _FD_POINT_MOVE_RULE_SIZE):
            brmove = move_ctx.reader
            brmove.seek(move_abs)
            for _ in range(num_move):
                setup.pointMoveRules.append(FDPointMoveRule(point=int(brmove.u16()), count=int(brmove.u16())))

        return setup


def _point_to_dict(point: ClothPoint) -> dict:
    return {
        'segment': int(point.segment),
        'flags': int(point.flags),
        'jointOrder': int(point.jointOrder),
        'upTo': int(point.upTo),
        'x': float(point.x),
        'y': float(point.y),
        'z': float(point.z),
    }


def _setup_to_dict(setup: ClothSetup, *, include_points: bool = True) -> dict:
    data = {
        'index': int(setup.index),
        'gravity': float(setup.gravity),
        'drag': float(setup.drag),
        'windResponse': float(setup.windResponse),
        'flags': int(setup.flags),
        'positionPoint': int(setup.positionPoint),
        'source_object_path': setup.source_object_path,
        'list_section_index': int(setup.list_section_index),
        'list_local_offset': int(setup.list_local_offset),
        'setup_section_index': int(setup.setup_section_index),
        'setup_local_offset': int(setup.setup_local_offset),
        'section_type': int(setup.section_type),
        'section_id': int(setup.section_id),
        'skip_flags': int(setup.skip_flags),
        'version_id': int(setup.version_id),
        'has_debug_info': int(setup.has_debug_info),
        'resource_type': int(setup.resource_type),
        'spec_mask': int(setup.spec_mask),
        'pointer_type_specifics': {str(k): int(v) for k, v in (setup.pointer_type_specifics or {}).items()},
        'points': [_point_to_dict(point) for point in setup.points] if include_points else [],
        'maps': [
            {
                'segment': int(item.segment),
                'flags': int(item.flags),
                'axis': int(item.axis),
                'jointOrder': int(item.jointOrder),
                'center': int(item.center),
                'points': [int(v) for v in item.points],
                'xMin': float(item.xMin),
                'xMax': float(item.xMax),
                'yMin': float(item.yMin),
                'yMax': float(item.yMax),
                'zMin': float(item.zMin),
                'zMax': float(item.zMax),
            }
            for item in setup.maps
        ],
        'distRules': [
            {
                'point': [int(v) for v in item.point],
                'flags': [int(v) for v in item.flags],
                'minDist': float(item.minDist),
                'maxDist': float(item.maxDist),
            }
            for item in setup.distRules
        ],
        'capsuleRules': [
            {'point': [int(v) for v in item.point], 'flags': [int(v) for v in item.flags], 'radius': float(item.radius)}
            for item in setup.capsuleRules
        ],
        'planeRules': [
            {'point': [int(v) for v in item.point], 'flags': [int(v) for v in item.flags], 'planeSegment': [int(v) for v in item.planeSegment]}
            for item in setup.planeRules
        ],
        'handRules': [
            {'movePt': int(item.movePt), 'point': [int(v) for v in item.point]}
            for item in setup.handRules
        ],
        'collideRules': [
            {'point': int(item.point), 'radius': float(item.radius)}
            for item in setup.collideRules
        ],
        'sphereCollideRules': [
            {'point': int(item.point), 'radius': float(item.radius)}
            for item in setup.sphereCollideRules
        ],
        'sideCollideRules': [
            {'point': [int(v) for v in item.point]}
            for item in setup.sideCollideRules
        ],
        'pointMoveRules': [
            {'point': int(item.point), 'count': int(item.count)}
            for item in setup.pointMoveRules
        ],
    }
    return data


def _setup_from_dict(data: dict) -> ClothSetup:
    setup = ClothSetup(
        index=int(data.get('index', 0) or 0),
        gravity=float(data.get('gravity', 10.0) or 0.0),
        drag=float(data.get('drag', 0.1) or 0.0),
        windResponse=float(data.get('windResponse', 0.1) or 0.0),
        flags=int(data.get('flags', 0) or 0),
        positionPoint=int(data.get('positionPoint', 0) or 0),
        source_object_path=str(data.get('source_object_path', '') or ''),
        list_section_index=int(data.get('list_section_index', -1) or -1),
        list_local_offset=int(data.get('list_local_offset', 0) or 0),
        setup_section_index=int(data.get('setup_section_index', -1) or -1),
        setup_local_offset=int(data.get('setup_local_offset', 0) or 0),
        section_type=int(data.get('section_type', 0) or 0),
        section_id=int(data.get('section_id', 0) or 0),
        skip_flags=int(data.get('skip_flags', 0) or 0),
        version_id=int(data.get('version_id', 0) or 0),
        has_debug_info=int(data.get('has_debug_info', 0) or 0),
        resource_type=int(data.get('resource_type', 0) or 0),
        spec_mask=int(data.get('spec_mask', 0xFFFFFFFF) or 0xFFFFFFFF),
        pointer_type_specifics={str(k): int(v) for k, v in dict(data.get('pointer_type_specifics', {}) or {}).items()},
    )
    for item in data.get('points', []) or []:
        setup.points.append(ClothPoint(
            segment=int(item.get('segment', 0) or 0),
            flags=int(item.get('flags', 4) or 0),
            jointOrder=int(item.get('jointOrder', 0) or 0),
            upTo=int(item.get('upTo', 0) or 0),
            x=float(item.get('x', 0.0) or 0.0),
            y=float(item.get('y', 0.0) or 0.0),
            z=float(item.get('z', 0.0) or 0.0),
        ))
    for item in data.get('maps', []) or []:
        pts = list(item.get('points', [0, 0, 0, 0]) or [0, 0, 0, 0])[:4]
        while len(pts) < 4:
            pts.append(0)
        setup.maps.append(ClothJointMap(
            segment=int(item.get('segment', 0) or 0),
            flags=int(item.get('flags', 0) or 0),
            axis=int(item.get('axis', 0) or 0),
            jointOrder=int(item.get('jointOrder', 0) or 0),
            center=int(item.get('center', 0) or 0),
            points=tuple(int(v) for v in pts),
            xMin=float(item.get('xMin', 0.0) or 0.0),
            xMax=float(item.get('xMax', 0.0) or 0.0),
            yMin=float(item.get('yMin', 0.0) or 0.0),
            yMax=float(item.get('yMax', 0.0) or 0.0),
            zMin=float(item.get('zMin', 0.0) or 0.0),
            zMax=float(item.get('zMax', 0.0) or 0.0),
        ))
    for item in data.get('distRules', []) or []:
        p = list(item.get('point', [0, 0]) or [0, 0])[:2]
        f = list(item.get('flags', [0, 0]) or [0, 0])[:2]
        while len(p) < 2:
            p.append(0)
        while len(f) < 2:
            f.append(0)
        setup.distRules.append(FDDistanceRule(point=(int(p[0]), int(p[1])), flags=(int(f[0]), int(f[1])), minDist=float(item.get('minDist', 0.0) or 0.0), maxDist=float(item.get('maxDist', 0.0) or 0.0)))
    for item in data.get('capsuleRules', []) or []:
        p = list(item.get('point', [0, 0]) or [0, 0])[:2]
        f = list(item.get('flags', [0, 0]) or [0, 0])[:2]
        while len(p) < 2:
            p.append(0)
        while len(f) < 2:
            f.append(0)
        setup.capsuleRules.append(FDCapsuleRule(point=(int(p[0]), int(p[1])), flags=(int(f[0]), int(f[1])), radius=float(item.get('radius', 0.0) or 0.0)))
    for item in data.get('planeRules', []) or []:
        p = list(item.get('point', [0]) or [0])[:1]
        f = list(item.get('flags', [0]) or [0])[:1]
        seg = list(item.get('planeSegment', [0, 0]) or [0, 0])[:2]
        while len(seg) < 2:
            seg.append(0)
        setup.planeRules.append(FDPlaneRule(point=(int(p[0]) if p else 0,), flags=(int(f[0]) if f else 0,), planeSegment=(int(seg[0]), int(seg[1]))))
    for item in data.get('handRules', []) or []:
        pts = list(item.get('point', [0, 0, 0]) or [0, 0, 0])[:3]
        while len(pts) < 3:
            pts.append(0)
        setup.handRules.append(FDHandednessRule(movePt=int(item.get('movePt', 0) or 0), point=tuple(int(v) for v in pts)))
    for item in data.get('collideRules', []) or []:
        setup.collideRules.append(FDCollideRule(point=int(item.get('point', 0) or 0), radius=float(item.get('radius', 0.0) or 0.0)))
    for item in data.get('sphereCollideRules', []) or []:
        setup.sphereCollideRules.append(FDSphereCollideRule(point=int(item.get('point', 0) or 0), radius=float(item.get('radius', 0.0) or 0.0)))
    for item in data.get('sideCollideRules', []) or []:
        p = list(item.get('point', [0, 0]) or [0, 0])[:2]
        while len(p) < 2:
            p.append(0)
        setup.sideCollideRules.append(FDSideCollideRule(point=(int(p[0]), int(p[1]))))
    for item in data.get('pointMoveRules', []) or []:
        setup.pointMoveRules.append(FDPointMoveRule(point=int(item.get('point', 0) or 0), count=int(item.get('count', 0) or 0)))
    return setup


def setup_counts(setup: ClothSetup) -> dict:
    return {
        'points': len(setup.points),
        'maps': len(setup.maps),
        'distRules': len(setup.distRules),
        'capsuleRules': len(setup.capsuleRules),
        'planeRules': len(setup.planeRules),
        'handRules': len(setup.handRules),
        'collideRules': len(setup.collideRules),
        'sphereCollideRules': len(setup.sphereCollideRules),
        'sideCollideRules': len(setup.sideCollideRules),
        'pointMoveRules': len(setup.pointMoveRules),
    }


def pack_cloth_section(setups: Sequence[ClothSetup]) -> tuple[bytes, List[tuple[int, int, int, int]], ClothSetup]:
    clean_setups = [setup for setup in list(setups or []) if setup is not None]
    if not clean_setups:
        clean_setups = [ClothSetup()]
    metadata = clean_setups[0]

    data = bytearray()
    relocs: List[tuple[int, int, int, int]] = []
    pointer_table_size = (len(clean_setups) + 1) * 4
    data.extend(b'\x00' * pointer_table_size)

    for setup_index, setup in enumerate(clean_setups):
        while len(data) % 4:
            data.append(0)
        setup_offset = len(data)
        struct.pack_into('<I', data, setup_index * 4, setup_offset)
        relocs.append((-1, 0, setup_index * 4, 0))

        setup_header_offset = len(data)
        data.extend(b'\x00' * _CLOTH_SETUP_SIZE)

        arrays: list[tuple[str, int, bytes]] = []

        point_payload = bytearray()
        for point in setup.points:
            point_payload.extend(struct.pack(
                '<HHHHfff',
                int(point.segment) & 0xFFFF,
                int(point.flags) & 0xFFFF,
                int(point.jointOrder) & 0xFFFF,
                int(point.upTo) & 0xFFFF,
                float(point.x),
                float(point.y),
                float(point.z),
            ))
        arrays.append(('points', len(setup.points), bytes(point_payload)))

        map_payload = bytearray()
        for item in setup.maps:
            pts = list(item.points)[:4]
            while len(pts) < 4:
                pts.append(0)
            map_payload.extend(struct.pack(
                '<HHBBHHHHHffffff',
                int(item.segment) & 0xFFFF,
                int(item.flags) & 0xFFFF,
                int(item.axis) & 0xFF,
                int(item.jointOrder) & 0xFF,
                int(item.center) & 0xFFFF,
                int(pts[0]) & 0xFFFF,
                int(pts[1]) & 0xFFFF,
                int(pts[2]) & 0xFFFF,
                int(pts[3]) & 0xFFFF,
                float(item.xMin),
                float(item.xMax),
                float(item.yMin),
                float(item.yMax),
                float(item.zMin),
                float(item.zMax),
            ))
        arrays.append(('maps', len(setup.maps), bytes(map_payload)))

        dist_payload = bytearray()
        for item in setup.distRules:
            dist_payload.extend(struct.pack(
                '<HHHHff',
                int(item.point[0]) & 0xFFFF,
                int(item.point[1]) & 0xFFFF,
                int(item.flags[0]) & 0xFFFF,
                int(item.flags[1]) & 0xFFFF,
                float(item.minDist),
                float(item.maxDist),
            ))
        arrays.append(('distRules', len(setup.distRules), bytes(dist_payload)))

        capsule_payload = bytearray()
        for item in setup.capsuleRules:
            capsule_payload.extend(struct.pack(
                '<HHHHf',
                int(item.point[0]) & 0xFFFF,
                int(item.point[1]) & 0xFFFF,
                int(item.flags[0]) & 0xFFFF,
                int(item.flags[1]) & 0xFFFF,
                float(item.radius),
            ))
        arrays.append(('capsuleRules', len(setup.capsuleRules), bytes(capsule_payload)))

        plane_payload = bytearray()
        for item in setup.planeRules:
            plane_payload.extend(struct.pack(
                '<HHhh',
                int(item.point[0]) & 0xFFFF,
                int(item.flags[0]) & 0xFFFF,
                max(-32768, min(32767, int(item.planeSegment[0]))),
                max(-32768, min(32767, int(item.planeSegment[1]))),
            ))
        arrays.append(('planeRules', len(setup.planeRules), bytes(plane_payload)))

        hand_payload = bytearray()
        for item in setup.handRules:
            hand_payload.extend(struct.pack(
                '<HHHH',
                int(item.movePt) & 0xFFFF,
                int(item.point[0]) & 0xFFFF,
                int(item.point[1]) & 0xFFFF,
                int(item.point[2]) & 0xFFFF,
            ))
        arrays.append(('handRules', len(setup.handRules), bytes(hand_payload)))

        collide_payload = bytearray()
        for item in setup.collideRules:
            collide_payload.extend(struct.pack('<If', int(item.point) & 0xFFFFFFFF, float(item.radius)))
        arrays.append(('collideRules', len(setup.collideRules), bytes(collide_payload)))

        sphere_payload = bytearray()
        for item in setup.sphereCollideRules:
            sphere_payload.extend(struct.pack('<If', int(item.point) & 0xFFFFFFFF, float(item.radius)))
        arrays.append(('sphereCollideRules', len(setup.sphereCollideRules), bytes(sphere_payload)))

        side_payload = bytearray()
        for item in setup.sideCollideRules:
            side_payload.extend(struct.pack('<HH', int(item.point[0]) & 0xFFFF, int(item.point[1]) & 0xFFFF))
        arrays.append(('sideCollideRules', len(setup.sideCollideRules), bytes(side_payload)))

        move_payload = bytearray()
        for item in setup.pointMoveRules:
            move_payload.extend(struct.pack('<HH', int(item.point) & 0xFFFF, int(item.count) & 0xFFFF))
        arrays.append(('pointMoveRules', len(setup.pointMoveRules), bytes(move_payload)))

        pointer_values: List[int] = []
        for _name, _count, payload in arrays:
            while len(data) % 4:
                data.append(0)
            pointer_values.append(len(data))
            data.extend(payload)

        counts = [
            len(setup.points),
            len(setup.maps),
            len(setup.distRules),
            len(setup.capsuleRules),
            len(setup.planeRules),
            len(setup.handRules),
            len(setup.collideRules),
            len(setup.sphereCollideRules),
            len(setup.sideCollideRules),
            len(setup.pointMoveRules),
        ]
        struct.pack_into(
            '<fffHH10H10I',
            data,
            setup_header_offset,
            float(setup.gravity),
            float(setup.drag),
            float(setup.windResponse),
            int(setup.flags) & 0xFFFF,
            int(setup.positionPoint) & 0xFFFF,
            *[int(value) & 0xFFFF for value in counts],
            *[int(value) & 0xFFFFFFFF for value in pointer_values],
        )

        type_specifics = dict(setup.pointer_type_specifics or {})
        for i, name in enumerate(_CLOTH_POINTER_NAMES):
            relocs.append((-1, int(type_specifics.get(name, 0)), setup_header_offset + 36 + (i * 4), 0))

    return bytes(data), relocs, metadata


@dataclass(slots=True)
class StandaloneSection:
    section_type: int
    skip_flags: int
    version_id: int
    has_debug_info: int
    resource_type: int
    section_id: int
    spec_mask: int
    data: bytes
    relocations: List[tuple[int, int, int, int]]


def read_standalone_section(path: str | Path) -> StandaloneSection:
    path = Path(path)
    blob = path.read_bytes()
    if len(blob) < 24 or blob[:4] not in {b'SECT', b'TCES'}:
        raise ValueError(f'{path.name} is not a standalone TR section')
    size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from('<iBBHIII', blob, 4)
    num_relocations = (int(packed_data) >> 8) & 0x00FFFFFF
    has_debug_info = int(packed_data) & 0x1
    resource_type = (int(packed_data) >> 1) & 0x7F
    relocs: List[tuple[int, int, int, int]] = []
    cursor = 24
    for _ in range(num_relocations):
        type_and_section, type_specific, offset = struct.unpack_from('<HhI', blob, cursor)
        relocs.append((((int(type_and_section) >> 3) & 0x1FFF), int(type_specific), int(offset), int(type_and_section) & 0x7))
        cursor += 8
    data = blob[cursor:]
    if len(data) != max(0, int(size)):
        logger.warning('Standalone section %s stored size=%d but payload=%d', path.name, size, len(data))
    return StandaloneSection(
        section_type=int(section_type),
        skip_flags=int(skip_flags),
        version_id=int(version_id),
        has_debug_info=int(has_debug_info),
        resource_type=int(resource_type),
        section_id=int(section_id),
        spec_mask=int(spec_mask),
        data=bytes(data),
        relocations=relocs,
    )


def write_standalone_section(
    path: str | Path,
    *,
    data: bytes,
    relocations: Sequence[tuple[int, int, int, int]],
    section_type: int = 0,
    skip_flags: int = 0,
    version_id: int = 0,
    has_debug_info: int = 0,
    resource_type: int = 0,
    section_id: int = 0,
    spec_mask: int = 0xFFFFFFFF,
) -> None:
    relocs_by_offset: Dict[int, tuple[int, int, int, int]] = {}
    for target, type_specific, offset, relocation_type in relocations:
        relocs_by_offset[int(offset)] = (int(target), int(type_specific), int(offset), int(relocation_type))
    relocs = [relocs_by_offset[offset] for offset in sorted(relocs_by_offset)]
    packed_data = (
        (int(has_debug_info) & 0x1)
        | ((int(resource_type) & 0x7F) << 1)
        | (len(relocs) << 8)
    )
    payload = bytearray()
    payload.extend(b'SECT')
    payload.extend(struct.pack('<iBBHIII', len(data), int(section_type) & 0xFF, int(skip_flags) & 0xFF, int(version_id) & 0xFFFF, packed_data & 0xFFFFFFFF, int(section_id) & 0xFFFFFFFF, int(spec_mask) & 0xFFFFFFFF))
    for target, type_specific, offset, relocation_type in relocs:
        payload.extend(struct.pack('<HhI', (((int(target) & 0x1FFF) << 3) | (int(relocation_type) & 0x7)) & 0xFFFF, _signed_i16(int(type_specific)), int(offset) & 0xFFFFFFFF))
    Path(path).write_bytes(bytes(payload) + bytes(data))
