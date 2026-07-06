from __future__ import annotations

import math
from io import BytesIO
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import bpy
import mathutils
from ..core.log import logger
from .animation_import import AnimationImporterMixin
from .animation_types import TRLAUTransformType

_BONE_PATH_RE = re.compile(r'pose\.bones\["(?P<name>(?:[^"\\]|\\.)+)"\]\."?(?P<prop>location|scale|rotation_quaternion|rotation_axis_angle|rotation_euler)"?')
_BONE_NAME_RE = re.compile(r'^bone_(\d+)$')
_FILENAME_INT_RE = re.compile(r'(\d+)')

_GAME_TICKS_PER_SECOND = 3000.0
_EPSILON = 1.0e-6
_ROTATION_LINEAR_TOLERANCE = 1.0e-5
_KEYED_CONSTANT_TOLERANCE = 1.0e-8



@dataclass(slots=True)
class _ExistingAnimationHeader:
    section_type: int = 2
    skip_flags: int = 0
    version_id: int = 0
    packed_data: int = 0
    section_id: int = 0
    spec_mask: int = 0xFFFFFFFF
    prefix: bytes = bytes(0x18)
    anim_id: int = 0
    frame_count: int = 0
    time_per_key: int = 100
    bone_count: int = 0
    section_count: int = 1
    transform_flags: int = 0
    endianness: str = '<'
    format: str = 'TR7AE'
    underworld_unk_38: int = 0
    underworld_unk_40: int = 0


@dataclass(slots=True)
class _CurveSet:
    curves: dict[int, bpy.types.FCurve] = field(default_factory=dict)

    def has_any(self) -> bool:
        return bool(self.curves)

    def evaluate(self, index: int, frame: float, default: float) -> float:
        curve = self.curves.get(index)
        if curve is None:
            return float(default)
        return float(curve.evaluate(frame))


@dataclass(slots=True)
class _BoneCurveData:
    location: _CurveSet = field(default_factory=_CurveSet)
    scale: _CurveSet = field(default_factory=_CurveSet)
    rotation_quaternion: _CurveSet = field(default_factory=_CurveSet)
    rotation_axis_angle: _CurveSet = field(default_factory=_CurveSet)
    rotation_euler: _CurveSet = field(default_factory=_CurveSet)

    def has_transform(self, transform_type: int) -> bool:
        if transform_type == TRLAUTransformType.LOCATION:
            return self.location.has_any()
        if transform_type == TRLAUTransformType.SCALE:
            return self.scale.has_any()
        if transform_type == TRLAUTransformType.ROTATION:
            return self.rotation_quaternion.has_any() or self.rotation_axis_angle.has_any() or self.rotation_euler.has_any()
        return False


@dataclass(slots=True)
class _PreparedAnimation:
    anim_id: int
    frame_count: int
    time_per_key: int
    bone_count: int
    transform_flags: int
    axis_flags: list[list[int]]
    tracks: list[list[list[list[float]]]]
    linear_track_keys: list[list[list[list[tuple[int, float]] | None]]]
    track_modes: list[list[list[int | None]]]


class TRLAUAnimationExporter:
    def __init__(self, *, debug: bool = False):
        self.debug = bool(debug)
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(message)

    @staticmethod
    def _get_action_fcurves(action: bpy.types.Action):
        yield from AnimationImporterMixin()._get_action_fcurves(action)

    @staticmethod
    def _decode_bone_name(name: str) -> str:
        try:
            return bytes(str(name), 'utf-8').decode('unicode_escape')
        except Exception:
            return str(name)

    @classmethod
    def _bone_index_from_name(cls, name: str) -> int | None:
        match = _BONE_NAME_RE.match(str(name or ''))
        if match is None:
            return None
        return int(match.group(1))

    @classmethod
    def _collect_action_bone_curves(cls, action: bpy.types.Action) -> dict[int, _BoneCurveData]:
        bone_curves: dict[int, _BoneCurveData] = {}
        for fcurve in cls._get_action_fcurves(action):
            data_path = str(getattr(fcurve, 'data_path', '') or '')
            match = _BONE_PATH_RE.search(data_path)
            if match is None:
                continue
            bone_name = cls._decode_bone_name(match.group('name'))
            bone_index = cls._bone_index_from_name(bone_name)
            if bone_index is None:
                continue
            prop = match.group('prop')
            array_index = int(getattr(fcurve, 'array_index', 0) or 0)
            curves = bone_curves.setdefault(bone_index, _BoneCurveData())
            curve_set = getattr(curves, prop, None)
            if curve_set is not None:
                curve_set.curves[array_index] = fcurve
        return bone_curves

    @staticmethod
    def _action_frame_bounds(action: bpy.types.Action) -> tuple[int, int]:
        frames: list[float] = []
        for fcurve in TRLAUAnimationExporter._get_action_fcurves(action):
            try:
                frames.extend(float(point.co[0]) for point in fcurve.keyframe_points)
            except Exception:
                pass
        if not frames:
            try:
                start, end = action.frame_range
                frames.extend([float(start), float(end)])
            except Exception:
                frames.extend([0.0, 0.0])

        start = int(math.floor(min(frames)))
        end = int(math.ceil(max(frames)))
        if end < start:
            end = start
        return start, end

    @staticmethod
    def _scene_fps(context) -> float:
        render = getattr(getattr(context, 'scene', None), 'render', None)
        fps = float(getattr(render, 'fps', 30.0) or 30.0)
        fps_base = float(getattr(render, 'fps_base', 1.0) or 1.0)
        if abs(fps_base) < _EPSILON:
            fps_base = 1.0
        return max(fps / fps_base, 1.0)

    @classmethod
    def time_per_key_from_fps(cls, fps: float) -> int:
        return max(1, int(round(_GAME_TICKS_PER_SECOND / max(float(fps), 1.0))))

    @staticmethod
    def _animation_header_is_plausible(data: bytes, anim_base: int, *, endianness: str, frame_offset: int, bone_offset: int, offset_offset: int, flags_offset: int) -> bool:
        try:
            if anim_base + flags_offset >= len(data):
                return False
            frame_count = struct.unpack_from(f'{endianness}h', data, anim_base + frame_offset)[0]
            bone_count = int(data[anim_base + bone_offset])
            transform_flags = int(data[anim_base + flags_offset])
            track_values_offset = struct.unpack_from(f'{endianness}i', data, anim_base + offset_offset)[0]
            track_data_start = anim_base + offset_offset + int(track_values_offset)
            return (
                0 < int(frame_count) <= 65535
                and 0 < int(bone_count) <= 255
                and 0 < int(transform_flags) <= 0b111
                and track_data_start > anim_base + flags_offset
                and track_data_start < len(data)
            )
        except Exception:
            return False

    @staticmethod
    def _parse_standalone_header(path: Path) -> _ExistingAnimationHeader | None:
        try:
            data = path.read_bytes()
        except Exception:
            return None
        if len(data) < 0x3D or data[:4] not in {b'SECT', b'TCES'}:
            return None
        endianness = '<' if data[:4] == b'SECT' else '>'
        try:
            _stored_size, section_type, skip_flags, version_id, packed_data, section_id, spec_mask = struct.unpack_from(f'{endianness}iBBHIII', data, 4)
            num_relocations = (int(packed_data) >> 8) & 0x00FFFFFF
            payload_base = 0x18 + (num_relocations * 0x08)
            if payload_base + 0x25 >= len(data):
                return None

            looks_tr7ae = TRLAUAnimationExporter._animation_header_is_plausible(
                data,
                payload_base,
                endianness=endianness,
                frame_offset=0x1A,
                bone_offset=0x1E,
                offset_offset=0x20,
                flags_offset=0x24,
            )
            looks_underworld = TRLAUAnimationExporter._animation_header_is_plausible(
                data,
                payload_base,
                endianness=endianness,
                frame_offset=0x32,
                bone_offset=0x36,
                offset_offset=0x44,
                flags_offset=0x48,
            )

            if looks_underworld and not looks_tr7ae:
                anim_id = struct.unpack_from(f'{endianness}h', data, payload_base + 0x30)[0]
                frame_count = struct.unpack_from(f'{endianness}h', data, payload_base + 0x32)[0]
                time_per_key = struct.unpack_from(f'{endianness}h', data, payload_base + 0x34)[0]
                bone_count = int(data[payload_base + 0x36])
                section_count = struct.unpack_from(f'{endianness}b', data, payload_base + 0x37)[0]
                transform_flags = int(data[payload_base + 0x48])
                underworld_unk_38 = struct.unpack_from(f'{endianness}i', data, payload_base + 0x38)[0]
                underworld_unk_40 = struct.unpack_from(f'{endianness}i', data, payload_base + 0x40)[0]
                if anim_id <= 0 and section_id:
                    anim_id = int(section_id)
                return _ExistingAnimationHeader(
                    section_type=int(section_type),
                    skip_flags=int(skip_flags),
                    version_id=int(version_id),
                    packed_data=int(packed_data) & 0xFF,
                    section_id=int(section_id) if int(section_id) else int(anim_id),
                    spec_mask=int(spec_mask),
                    prefix=bytes(data[payload_base:payload_base + 0x30]).ljust(0x30, b'\x00')[:0x30],
                    anim_id=int(anim_id),
                    frame_count=int(frame_count),
                    time_per_key=int(time_per_key),
                    bone_count=int(bone_count),
                    section_count=int(section_count) if int(section_count) > 0 else 1,
                    transform_flags=int(transform_flags),
                    endianness=endianness,
                    format='UNDERWORLD',
                    underworld_unk_38=int(underworld_unk_38),
                    underworld_unk_40=int(underworld_unk_40),
                )

            if not looks_tr7ae:
                return None

            anim_id = struct.unpack_from(f'{endianness}h', data, 0x10)[0]
            payload_anim_id = struct.unpack_from(f'{endianness}h', data, payload_base + 0x18)[0]
            frame_count = struct.unpack_from(f'{endianness}h', data, payload_base + 0x1A)[0]
            time_per_key = struct.unpack_from(f'{endianness}h', data, payload_base + 0x1C)[0]
            bone_count = int(data[payload_base + 0x1E])
            section_count = int(data[payload_base + 0x1F])
            transform_flags = int(data[payload_base + 0x24])
            if payload_anim_id:
                anim_id = int(payload_anim_id)
            if anim_id <= 0 and section_id:
                anim_id = int(section_id)
            return _ExistingAnimationHeader(
                section_type=int(section_type),
                skip_flags=int(skip_flags),
                version_id=int(version_id),
                packed_data=int(packed_data) & 0xFF,
                section_id=int(section_id) if int(section_id) else int(anim_id),
                spec_mask=int(spec_mask),
                prefix=bytes(data[payload_base:payload_base + 0x18]).ljust(0x18, b'\x00')[:0x18],
                anim_id=int(anim_id),
                frame_count=int(frame_count),
                time_per_key=int(time_per_key),
                bone_count=int(bone_count),
                section_count=max(1, int(section_count)),
                transform_flags=int(transform_flags),
                endianness=endianness,
                format='TR7AE',
            )
        except Exception:
            return None

    @staticmethod
    def _anim_id_from_filename(path: Path) -> int | None:
        stem = path.stem
        matches = _FILENAME_INT_RE.findall(stem)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except Exception:
            return None

    @classmethod
    def _resolve_anim_id(cls, action: bpy.types.Action, path: Path, existing: _ExistingAnimationHeader | None) -> int:
        filename_id = cls._anim_id_from_filename(path)
        if filename_id is not None:
            return int(filename_id)
        if existing is not None and int(existing.anim_id) > 0:
            return int(existing.anim_id)
        raise ValueError('Could not determine animation ID. Use a .ani filename containing the animation ID.')

    @staticmethod
    def _bone_index_range_from_armature(armature: bpy.types.Object) -> int:
        max_index = -1
        for bone in getattr(getattr(armature, 'data', None), 'bones', []) or []:
            match = _BONE_NAME_RE.match(str(getattr(bone, 'name', '') or ''))
            if match is not None:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    @classmethod
    def _resolve_bone_count(cls, armature: bpy.types.Object, action: bpy.types.Action, bone_curves: dict[int, _BoneCurveData], existing: _ExistingAnimationHeader | None) -> int:
        if existing is not None and int(existing.bone_count) > 0:
            return int(existing.bone_count)
        max_keyed_index = max(bone_curves.keys(), default=-1) + 1
        max_armature_index = cls._bone_index_range_from_armature(armature)
        if max_keyed_index > 0:
            return int(max_keyed_index)
        if max_armature_index > 0:
            return int(max_armature_index)
        raise ValueError('No bone_X bones or pose-bone animation curves were found.')

    @staticmethod
    def _evaluate_location(curves: _BoneCurveData, frame: float) -> mathutils.Vector:
        return mathutils.Vector((
            curves.location.evaluate(0, frame, 0.0),
            curves.location.evaluate(1, frame, 0.0),
            curves.location.evaluate(2, frame, 0.0),
        ))

    @staticmethod
    def _evaluate_scale(curves: _BoneCurveData, frame: float) -> mathutils.Vector:
        return mathutils.Vector((
            curves.scale.evaluate(0, frame, 1.0),
            curves.scale.evaluate(1, frame, 1.0),
            curves.scale.evaluate(2, frame, 1.0),
        ))

    @staticmethod
    def _evaluate_rotation(curves: _BoneCurveData, pose_bone: bpy.types.PoseBone, frame: float) -> mathutils.Quaternion:
        if curves.rotation_quaternion.has_any():
            quat = mathutils.Quaternion((
                curves.rotation_quaternion.evaluate(0, frame, 1.0),
                curves.rotation_quaternion.evaluate(1, frame, 0.0),
                curves.rotation_quaternion.evaluate(2, frame, 0.0),
                curves.rotation_quaternion.evaluate(3, frame, 0.0),
            ))
            try:
                quat.normalize()
            except Exception:
                pass
            return quat

        if curves.rotation_axis_angle.has_any():
            angle = curves.rotation_axis_angle.evaluate(0, frame, 0.0)
            axis = mathutils.Vector((
                curves.rotation_axis_angle.evaluate(1, frame, 0.0),
                curves.rotation_axis_angle.evaluate(2, frame, 0.0),
                curves.rotation_axis_angle.evaluate(3, frame, 1.0),
            ))
            if axis.length < _EPSILON:
                return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
            return mathutils.Quaternion(axis.normalized(), angle)

        if curves.rotation_euler.has_any():
            mode = str(getattr(pose_bone, 'rotation_mode', 'XYZ') or 'XYZ')
            if mode in {'QUATERNION', 'AXIS_ANGLE'}:
                mode = 'XYZ'
            euler = mathutils.Euler((
                curves.rotation_euler.evaluate(0, frame, 0.0),
                curves.rotation_euler.evaluate(1, frame, 0.0),
                curves.rotation_euler.evaluate(2, frame, 0.0),
            ), mode)
            return euler.to_quaternion()

        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))

    @staticmethod
    def _rotation_curve_sets(curves: _BoneCurveData) -> tuple[_CurveSet, _CurveSet, _CurveSet]:
        return (curves.rotation_quaternion, curves.rotation_axis_angle, curves.rotation_euler)

    @classmethod
    def _rotation_fcurves(cls, curves: _BoneCurveData):
        for curve_set in cls._rotation_curve_sets(curves):
            for fcurve in curve_set.curves.values():
                if fcurve is not None:
                    yield fcurve

    @classmethod
    def _all_rotation_keyframes_are_linear(cls, curves: _BoneCurveData) -> bool:
        saw_keyframe = False
        for fcurve in cls._rotation_fcurves(curves):
            try:
                points = list(getattr(fcurve, 'keyframe_points', []) or [])
            except Exception:
                points = []
            for point in points:
                saw_keyframe = True
                if str(getattr(point, 'interpolation', '') or '') != 'LINEAR':
                    return False
        return saw_keyframe

    @classmethod
    def _integer_rotation_keyframe_frames(cls, curves: _BoneCurveData, start: int, end: int) -> list[int] | None:
        frames: set[int] = set()
        for fcurve in cls._rotation_fcurves(curves):
            try:
                points = list(getattr(fcurve, 'keyframe_points', []) or [])
            except Exception:
                points = []
            for point in points:
                try:
                    frame = float(point.co[0])
                except Exception:
                    continue
                if frame < float(start) - _EPSILON or frame > float(end) + _EPSILON:
                    continue
                rounded = int(round(frame))
                if abs(frame - float(rounded)) > _EPSILON:
                    return None
                frames.add(rounded)

        if not frames:
            return None
        frames.add(int(start))
        frames.add(int(end))
        return sorted(frame for frame in frames if int(start) <= int(frame) <= int(end))

    @classmethod
    def _game_rotation_tracks_from_linear_keyframes(
        cls,
        curves: _BoneCurveData,
        pose_bone: bpy.types.PoseBone,
        start: int,
        end: int,
    ) -> tuple[list[list[float]], list[list[tuple[int, float]]]] | None:
        """Recover game-space rotation-vector tracks from sparse linear keys.

        Imported rotations are stored as three independent game-space
        rotation-vector axis tracks, but Blender displays them as quaternion
        curves. If we compress dense quaternion evaluation directly, tiny
        quaternion/log-map roundoff can create many fake breakpoints on axes
        that were originally simple Linear tracks. For linear integer-frame
        rotation curves, convert only the real keyed frames back to game-space,
        simplify each axis independently, then expand those simplified axis keys
        back to per-frame values for Constant/Linear/Raw selection.
        """
        if not cls._all_rotation_keyframes_are_linear(curves):
            return None
        frames = cls._integer_rotation_keyframe_frames(curves, start, end)
        if frames is None or not frames:
            return None

        keyed_vectors: list[tuple[int, mathutils.Vector]] = []
        for scene_frame in frames:
            quaternion = cls._evaluate_rotation(curves, pose_bone, float(scene_frame))
            vector = cls._game_rotation_vector(pose_bone, quaternion)
            keyed_vectors.append((int(scene_frame) - int(start), vector))

        frame_count = int(end) - int(start) + 1
        if frame_count <= 0:
            return None

        axis_keyframes: list[list[tuple[int, float]]] = []
        axis_values: list[list[float]] = []
        for axis in range(3):
            keys = cls._simplify_linear_keyframes(
                [(local_frame, float(vector[axis])) for local_frame, vector in keyed_vectors],
                tolerance=_ROTATION_LINEAR_TOLERANCE,
            )
            axis_keyframes.append(keys)
            axis_values.append([cls._interpolated_value_from_keyframes(keys, local_frame) for local_frame in range(frame_count)])

        return axis_values, axis_keyframes

    @staticmethod
    def _rotation_basis(pose_bone: bpy.types.PoseBone) -> mathutils.Matrix:
        if pose_bone.parent:
            return (pose_bone.parent.bone.matrix_local @ pose_bone.bone.matrix_local.inverted()).to_3x3()
        return pose_bone.bone.matrix_local.inverted().to_3x3()

    @staticmethod
    def _location_basis(pose_bone: bpy.types.PoseBone) -> mathutils.Matrix:
        if pose_bone.parent:
            return pose_bone.parent.bone.matrix_local @ pose_bone.bone.matrix_local.inverted()
        return pose_bone.bone.matrix_local.inverted()

    @classmethod
    def _game_rotation_vector(cls, pose_bone: bpy.types.PoseBone, blender_quaternion: mathutils.Quaternion) -> mathutils.Vector:
        basis = cls._rotation_basis(pose_bone)
        try:
            source_matrix = basis.inverted() @ blender_quaternion.to_matrix() @ basis
            source_quaternion = source_matrix.to_quaternion()
            source_quaternion.normalize()
        except Exception:
            source_quaternion = blender_quaternion.copy()
        angle = float(source_quaternion.angle)
        if angle > math.pi:
            angle -= math.tau
        axis = mathutils.Vector(source_quaternion.axis)
        if axis.length < _EPSILON or abs(angle) < _EPSILON:
            return mathutils.Vector((0.0, 0.0, 0.0))
        return axis.normalized() * angle

    @classmethod
    def _game_location_vector(cls, pose_bone: bpy.types.PoseBone, blender_location: mathutils.Vector) -> mathutils.Vector:
        basis = cls._location_basis(pose_bone)
        basis_translation = basis.to_translation()
        basis_3x3 = basis.to_3x3()
        result = mathutils.Vector((0.0, 0.0, 0.0))
        for axis in range(3):
            denom = float(basis_3x3[axis][axis])
            if abs(denom) < _EPSILON:
                result[axis] = float(blender_location[axis])
            else:
                result[axis] = (float(blender_location[axis]) - float(basis_translation[axis])) / denom
        return result

    @classmethod
    def _game_location_axis_value(cls, pose_bone: bpy.types.PoseBone, axis: int, blender_axis_value: float) -> float:
        basis = cls._location_basis(pose_bone)
        basis_translation = basis.to_translation()
        basis_3x3 = basis.to_3x3()
        denom = float(basis_3x3[int(axis)][int(axis)])
        if abs(denom) < _EPSILON:
            return float(blender_axis_value)
        return (float(blender_axis_value) - float(basis_translation[int(axis)])) / denom

    @staticmethod
    def _curve_has_only_linear_integer_keys(fcurve: bpy.types.FCurve, start: int, end: int) -> bool:
        try:
            points = list(getattr(fcurve, 'keyframe_points', []) or [])
        except Exception:
            points = []
        if not points:
            return False
        for point in points:
            try:
                frame = float(point.co[0])
            except Exception:
                return False
            if frame < float(start) - _EPSILON or frame > float(end) + _EPSILON:
                continue
            if abs(frame - float(int(round(frame)))) > _EPSILON:
                return False
            if str(getattr(point, 'interpolation', '') or '') != 'LINEAR':
                return False
        return True

    @classmethod
    def _linear_keyframes_from_fcurve(
        cls,
        fcurve: bpy.types.FCurve | None,
        start: int,
        end: int,
        convert_value,
    ) -> list[tuple[int, float]] | None:
        if fcurve is None or not cls._curve_has_only_linear_integer_keys(fcurve, start, end):
            return None
        keyed_frames: set[int] = {int(start), int(end)}
        try:
            for point in getattr(fcurve, 'keyframe_points', []) or []:
                frame = float(point.co[0])
                if float(start) - _EPSILON <= frame <= float(end) + _EPSILON:
                    keyed_frames.add(int(round(frame)))
        except Exception:
            return None

        keyframes: list[tuple[int, float]] = []
        for scene_frame in sorted(keyed_frames):
            try:
                value = float(fcurve.evaluate(float(scene_frame)))
            except Exception:
                return None
            keyframes.append((int(scene_frame) - int(start), float(convert_value(value))))
        return cls._simplify_linear_keyframes(keyframes)

    @staticmethod
    def _track_has_meaningful_values(values: Iterable[float], default: float) -> bool:
        return any(abs(float(value) - float(default)) > _EPSILON for value in values)

    @staticmethod
    def _axis_flags_for_transform(axis_values: list[list[list[float]]], default: float) -> list[int]:
        flags: list[int] = []
        for bone_axes in axis_values:
            bone_flags = 0
            for axis in range(3):
                if TRLAUAnimationExporter._track_has_meaningful_values(bone_axes[axis], default):
                    bone_flags |= (1 << axis)
            flags.append(bone_flags)
        return flags

    @staticmethod
    def _values_are_constant(values: list[float], tolerance: float = _EPSILON) -> bool:
        if not values:
            return True
        first = float(values[0])
        return all(abs(float(value) - first) <= float(tolerance) for value in values)

    @staticmethod
    def _raw_track_size(frame_count: int) -> int:
        return 4 + max(0, int(frame_count)) * 4

    @staticmethod
    def _linear_track_size(keyframe_count: int) -> int:
        keyframe_count = max(1, int(keyframe_count))
        time_delta_size = max(0, keyframe_count - 1)
        padded_time_delta_size = (time_delta_size + 3) & ~3
        return 4 + padded_time_delta_size + keyframe_count * 4

    @classmethod
    def _linear_keyframes_from_values(cls, values: list[float], tolerance: float = 1.0e-5) -> list[tuple[int, float]]:
        """Build exact-enough piecewise-linear keys using TRLAU's byte-delta frame indices."""
        if not values:
            return [(0, 0.0)]
        if len(values) == 1:
            return [(0, float(values[0]))]

        selected: set[int] = {0, len(values) - 1}
        stack: list[tuple[int, int]] = [(0, len(values) - 1)]
        while stack:
            start, end = stack.pop()
            if end <= start + 1:
                continue
            start_value = float(values[start])
            end_value = float(values[end])
            span = float(end - start)
            worst_index = -1
            worst_error = 0.0
            for frame in range(start + 1, end):
                alpha = float(frame - start) / span
                expected = start_value + (end_value - start_value) * alpha
                error = abs(float(values[frame]) - expected)
                if error > worst_error:
                    worst_error = error
                    worst_index = frame
            if worst_index >= 0 and worst_error > tolerance:
                selected.add(worst_index)
                stack.append((start, worst_index))
                stack.append((worst_index, end))

        ordered = sorted(selected)
        return cls._expand_linear_keyframes([(int(frame), float(values[frame])) for frame in ordered])

    @staticmethod
    def _interpolated_value_from_keyframes(keyframes: list[tuple[int, float]], frame: int) -> float:
        if not keyframes:
            return 0.0
        frame = int(frame)
        if frame <= int(keyframes[0][0]):
            return float(keyframes[0][1])
        if frame >= int(keyframes[-1][0]):
            return float(keyframes[-1][1])
        for left, right in zip(keyframes, keyframes[1:]):
            left_frame, left_value = int(left[0]), float(left[1])
            right_frame, right_value = int(right[0]), float(right[1])
            if left_frame <= frame <= right_frame:
                span = right_frame - left_frame
                alpha = float(frame - left_frame) / float(span) if span else 0.0
                return left_value + ((right_value - left_value) * alpha)
        return 0.0

    @classmethod
    def _simplify_linear_keyframes(cls, keyframes: list[tuple[int, float]], tolerance: float = 1.0e-5) -> list[tuple[int, float]]:
        if not keyframes:
            return [(0, 0.0)]
        ordered_map: dict[int, float] = {}
        for frame, value in keyframes:
            ordered_map[int(frame)] = float(value)
        ordered = sorted(ordered_map.items())
        if len(ordered) <= 2:
            return cls._expand_linear_keyframes(ordered)

        selected: set[int] = {ordered[0][0], ordered[-1][0]}
        stack: list[tuple[int, int]] = [(0, len(ordered) - 1)]
        while stack:
            left_index, right_index = stack.pop()
            if right_index <= left_index + 1:
                continue
            left_frame, left_value = ordered[left_index]
            right_frame, right_value = ordered[right_index]
            span = float(right_frame - left_frame)
            if abs(span) < _EPSILON:
                continue
            worst_index = -1
            worst_error = 0.0
            for index in range(left_index + 1, right_index):
                frame, value = ordered[index]
                alpha = float(frame - left_frame) / span
                expected = left_value + ((right_value - left_value) * alpha)
                error = abs(float(value) - expected)
                if error > worst_error:
                    worst_error = error
                    worst_index = index
            if worst_index >= 0 and worst_error > tolerance:
                selected.add(ordered[worst_index][0])
                stack.append((left_index, worst_index))
                stack.append((worst_index, right_index))

        return cls._expand_linear_keyframes([(frame, value) for frame, value in ordered if frame in selected])

    @classmethod
    def _expand_linear_keyframes(cls, keyframes: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if not keyframes:
            return [(0, 0.0)]
        ordered = sorted((int(frame), float(value)) for frame, value in keyframes)
        expanded: list[tuple[int, float]] = []
        for frame, value in ordered:
            if not expanded:
                expanded.append((frame, value))
                continue
            previous_frame, _previous_value = expanded[-1]
            while frame - previous_frame > 255:
                previous_frame += 255
                interpolated = cls._interpolated_value_from_keyframes(ordered, previous_frame)
                expanded.append((previous_frame, interpolated))
            if frame != expanded[-1][0]:
                expanded.append((frame, value))
        return expanded

    @classmethod
    def _linear_keyframes_for_track(cls, values: list[float], keyframes: list[tuple[int, float]] | None = None) -> list[tuple[int, float]]:
        if keyframes:
            return cls._simplify_linear_keyframes(keyframes)
        return cls._linear_keyframes_from_values(values)

    @classmethod
    def _choose_track_mode(cls, values: list[float], keyframes: list[tuple[int, float]] | None = None) -> int:
        """Choose Constant(1), Linear(2), or Raw(0) from the exported values.

        Constant is used for unchanged tracks. Linear is used when the data has
        sparse integer-frame linear keys that pack smaller than raw per-frame
        storage. Raw is used when dense storage is smaller or more accurate.
        
        When a track came from actual integer-frame linear keys, use a tighter
        constant tolerance. Original .ani files sometimes store tiny but real
        two-key linear spans, and collapsing those to Constant changes the
        packed structure and size.
        """
        keyed_frames = {int(frame) for frame, _value in (keyframes or [])}
        constant_tolerance = _KEYED_CONSTANT_TOLERANCE if len(keyed_frames) >= 2 else _EPSILON
        if cls._values_are_constant(values, tolerance=constant_tolerance):
            return 1
        linear_key_count = len(cls._linear_keyframes_for_track(values, keyframes))
        return 2 if cls._linear_track_size(linear_key_count) < cls._raw_track_size(len(values)) else 0

    def prepare_animation(
        self,
        context,
        armature: bpy.types.Object,
        action: bpy.types.Action,
        filepath: str,
        *,
        frame_start: int | None = None,
        frame_end: int | None = None,
        existing: _ExistingAnimationHeader | None = None,
    ) -> _PreparedAnimation:
        if armature is None or getattr(armature, 'type', None) != 'ARMATURE':
            raise ValueError('Select an armature object before exporting animation.')
        if action is None:
            raise ValueError('Choose an action to export.')

        path = Path(filepath)
        bone_curves = self._collect_action_bone_curves(action)
        action_start, action_end = self._action_frame_bounds(action)
        start = int(action_start if frame_start is None else frame_start)
        end = int(action_end if frame_end is None else frame_end)
        if end < start:
            raise ValueError('Animation end frame must be greater than or equal to start frame.')
        frame_count = int(end - start + 1)
        if frame_count <= 0 or frame_count > 32767:
            raise ValueError(f'Unsupported animation frame count: {frame_count}')

        time_per_key = self.time_per_key_from_fps(self._scene_fps(context))
        anim_id = self._resolve_anim_id(action, path, existing)
        bone_count = self._resolve_bone_count(armature, action, bone_curves, existing)
        if bone_count <= 0 or bone_count > 255:
            raise ValueError(f'Unsupported animation bone count: {bone_count}. The format stores this as one byte.')

        keyed_beyond_range = [index for index in bone_curves if int(index) >= int(bone_count)]
        if keyed_beyond_range:
            raise ValueError(f'Action contains keyed bone(s) beyond export bone count {bone_count}: {keyed_beyond_range[:8]}')

        rotation_values = [[[[] for _axis in range(3)] for _bone in range(bone_count)]][0]
        scale_values = [[[[] for _axis in range(3)] for _bone in range(bone_count)]][0]
        location_values = [[[[] for _axis in range(3)] for _bone in range(bone_count)]][0]
        linear_track_keys: list[list[list[list[tuple[int, float]] | None]]] = [
            [[None for _axis in range(3)] for _bone in range(bone_count)]
            for _transform in range(3)
        ]

        pose_bones = getattr(getattr(armature, 'pose', None), 'bones', None)
        if pose_bones is None:
            raise ValueError('Armature pose bones are unavailable.')

        missing_pose_bones: list[int] = []
        for bone_index in range(bone_count):
            pose_bone = pose_bones.get(f'bone_{bone_index}')
            curves = bone_curves.get(bone_index, _BoneCurveData())
            if pose_bone is None:
                if curves.location.has_any() or curves.rotation_quaternion.has_any() or curves.rotation_axis_angle.has_any() or curves.rotation_euler.has_any() or curves.scale.has_any():
                    missing_pose_bones.append(bone_index)
                for _frame in range(frame_count):
                    for axis in range(3):
                        rotation_values[bone_index][axis].append(0.0)
                        scale_values[bone_index][axis].append(1.0)
                        location_values[bone_index][axis].append(0.0)
                continue

            rotation_axis_values = None
            rotation_axis_keyframes = None
            if curves.has_transform(TRLAUTransformType.ROTATION):
                rotation_track_data = self._game_rotation_tracks_from_linear_keyframes(curves, pose_bone, start, end)
                if rotation_track_data is not None:
                    rotation_axis_values, rotation_axis_keyframes = rotation_track_data
                    for axis in range(3):
                        linear_track_keys[TRLAUTransformType.ROTATION][bone_index][axis] = rotation_axis_keyframes[axis]

            for axis in range(3):
                scale_curve = curves.scale.curves.get(axis)
                linear_track_keys[TRLAUTransformType.SCALE][bone_index][axis] = self._linear_keyframes_from_fcurve(
                    scale_curve,
                    start,
                    end,
                    lambda value: float(value),
                )

                location_curve = curves.location.curves.get(axis)
                linear_track_keys[TRLAUTransformType.LOCATION][bone_index][axis] = self._linear_keyframes_from_fcurve(
                    location_curve,
                    start,
                    end,
                    lambda value, axis=axis, pose_bone=pose_bone: self._game_location_axis_value(pose_bone, axis, value),
                )

            for local_frame, scene_frame in enumerate(range(start, end + 1)):
                sample_frame = float(scene_frame)
                if rotation_axis_values is not None:
                    rotation_vector = mathutils.Vector((
                        rotation_axis_values[0][local_frame],
                        rotation_axis_values[1][local_frame],
                        rotation_axis_values[2][local_frame],
                    ))
                else:
                    rotation_vector = self._game_rotation_vector(pose_bone, self._evaluate_rotation(curves, pose_bone, sample_frame)) if curves.has_transform(TRLAUTransformType.ROTATION) else mathutils.Vector((0.0, 0.0, 0.0))
                scale_vector = self._evaluate_scale(curves, sample_frame) if curves.has_transform(TRLAUTransformType.SCALE) else mathutils.Vector((1.0, 1.0, 1.0))
                location_vector = self._game_location_vector(pose_bone, self._evaluate_location(curves, sample_frame)) if curves.has_transform(TRLAUTransformType.LOCATION) else mathutils.Vector((0.0, 0.0, 0.0))
                for axis in range(3):
                    rotation_values[bone_index][axis].append(float(rotation_vector[axis]))
                    scale_values[bone_index][axis].append(float(scale_vector[axis]))
                    location_values[bone_index][axis].append(float(location_vector[axis]))

        if missing_pose_bones:
            raise ValueError(f'Action keys bone_X entries that do not exist on the armature: {missing_pose_bones[:8]}')

        rotation_flags = self._axis_flags_for_transform(rotation_values, 0.0)
        scale_flags = self._axis_flags_for_transform(scale_values, 1.0)
        location_flags = self._axis_flags_for_transform(location_values, 0.0)
        axis_flags = [rotation_flags, scale_flags, location_flags]

        transform_flags = 0
        if any(rotation_flags):
            transform_flags |= (1 << TRLAUTransformType.ROTATION)
        if any(scale_flags):
            transform_flags |= (1 << TRLAUTransformType.SCALE)
        if any(location_flags):
            transform_flags |= (1 << TRLAUTransformType.LOCATION)
        if transform_flags == 0:
            raise ValueError('The selected action does not contain non-default bone transform animation for bone_X bones.')

        tracks = [rotation_values, scale_values, location_values]
        track_modes: list[list[list[int | None]]] = [[[None for _axis in range(3)] for _bone in range(bone_count)] for _transform in range(3)]
        for transform_type in range(3):
            if transform_flags & (1 << transform_type) == 0:
                continue
            for bone_index in range(bone_count):
                flags = axis_flags[transform_type][bone_index]
                if flags == 0:
                    continue
                for axis in range(3):
                    if flags & (1 << axis):
                        track_modes[transform_type][bone_index][axis] = self._choose_track_mode(
                            tracks[transform_type][bone_index][axis],
                            linear_track_keys[transform_type][bone_index][axis],
                        )

        return _PreparedAnimation(
            anim_id=int(anim_id),
            frame_count=int(frame_count),
            time_per_key=int(time_per_key),
            bone_count=int(bone_count),
            transform_flags=int(transform_flags),
            axis_flags=axis_flags,
            tracks=tracks,
            linear_track_keys=linear_track_keys,
            track_modes=track_modes,
        )

    @staticmethod
    def _pack_axis_flags(prepared: _PreparedAnimation) -> bytes:
        bit_pos = 0
        packed = 0
        for transform_type in range(3):
            if prepared.transform_flags & (1 << transform_type) == 0:
                continue
            flags = prepared.axis_flags[transform_type]
            for bone_index in range(prepared.bone_count):
                packed |= (int(flags[bone_index]) & 0b111) << (bit_pos + (bone_index * 3))
            bit_pos += prepared.bone_count * 3
            if bit_pos % 8:
                bit_pos += 8 - (bit_pos % 8)
        byte_count = (bit_pos + 7) // 8
        return int(packed).to_bytes(byte_count, 'little')

    @classmethod
    def _pack_linear_track(cls, values: list[float], endianness: str, keyframes: list[tuple[int, float]] | None = None) -> bytes:
        keyframes = cls._linear_keyframes_for_track(values, keyframes)
        if not keyframes:
            keyframes = [(0, 0.0)]
        if len(keyframes) > 65535:
            raise ValueError(f'Linear animation track has too many keys: {len(keyframes)}')

        payload = bytearray()
        payload.extend(struct.pack(f'{endianness}HH', 2, len(keyframes)))
        previous_frame = int(keyframes[0][0])
        if previous_frame != 0:
            raise ValueError('Linear animation tracks must begin at local frame 0')
        for frame, _value in keyframes[1:]:
            delta = int(frame) - previous_frame
            if delta < 0 or delta > 255:
                raise ValueError(f'Linear animation track frame delta is out of range: {delta}')
            payload.append(delta)
            previous_frame = int(frame)
        if len(payload) % 4:
            payload.extend(b'\x00' * (4 - (len(payload) % 4)))

        previous_value = 0.0
        for _frame, value in keyframes:
            value = float(value)
            payload.extend(struct.pack(f'{endianness}f', value - previous_value))
            previous_value = value
        return bytes(payload)

    @classmethod
    def _pack_track(cls, values: list[float], endianness: str, mode: int | None = None, keyframes: list[tuple[int, float]] | None = None) -> bytes:
        values = [float(value) for value in (values or [0.0])]
        mode = cls._choose_track_mode(values, keyframes) if mode is None else int(mode)
        if mode == 1:
            return struct.pack(f'{endianness}HHf', 1, 0, float(values[0]))
        if mode == 2:
            return cls._pack_linear_track(values, endianness, keyframes)

        # Raw stores one float for every frame
        payload = bytearray()
        payload.extend(struct.pack(f'{endianness}HH', 0, 0))
        payload.extend(struct.pack(f'{endianness}{len(values)}f', *values))
        return bytes(payload)

    def build_section_payload(self, prepared: _PreparedAnimation, existing: _ExistingAnimationHeader | None = None, *, endianness: str = '<') -> bytes:
        prefix = bytes(existing.prefix if existing is not None else bytes(0x18)).ljust(0x18, b'\x00')[:0x18]
        axis_blob = self._pack_axis_flags(prepared)
        flags_start = 0x25
        track_data_start = (flags_start + len(axis_blob) + 3) & ~3
        track_values_offset = track_data_start - 0x20

        payload = bytearray(prefix)
        payload.extend(struct.pack(
            f'{endianness}hhhBBiB',
            int(prepared.anim_id),
            int(prepared.frame_count),
            int(prepared.time_per_key),
            int(prepared.bone_count),
            int(existing.section_count if existing is not None and existing.section_count > 0 else 1),
            int(track_values_offset),
            int(prepared.transform_flags) & 0xFF,
        ))
        payload.extend(axis_blob)
        if len(payload) < track_data_start:
            payload.extend(b'\x00' * (track_data_start - len(payload)))

        for bone_index in range(prepared.bone_count):
            for transform_type in range(3):
                if prepared.transform_flags & (1 << transform_type) == 0:
                    continue
                flags = prepared.axis_flags[transform_type][bone_index]
                if flags == 0:
                    continue
                for axis in range(3):
                    if flags & (1 << axis):
                        payload.extend(self._pack_track(
                            prepared.tracks[transform_type][bone_index][axis],
                            endianness,
                            prepared.track_modes[transform_type][bone_index][axis],
                            prepared.linear_track_keys[transform_type][bone_index][axis],
                        ))
        return bytes(payload)

    @staticmethod
    def _default_underworld_prefix(endianness: str) -> bytes:
        # TR8 has nine floats before AnimFragment
        return struct.pack(f'{endianness}fffffffff', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0) + bytes(0x0C)

    @classmethod
    def _underworld_prefix_for_endianness(cls, existing: _ExistingAnimationHeader | None, endianness: str) -> bytes:
        if existing is None or str(getattr(existing, 'format', '')).upper() != 'UNDERWORLD' or len(existing.prefix) < 0x30:
            return cls._default_underworld_prefix(endianness)
        prefix = bytes(existing.prefix).ljust(0x30, b'\x00')[:0x30]
        source_endianness = existing.endianness if existing.endianness in {'<', '>'} else endianness
        try:
            values = struct.unpack(f'{source_endianness}fffffffff', prefix[:0x24])
            return struct.pack(f'{endianness}fffffffff', *values) + prefix[0x24:0x30]
        except Exception:
            return prefix

    def build_underworld_section_payload(self, prepared: _PreparedAnimation, existing: _ExistingAnimationHeader | None = None, *, endianness: str = '<') -> bytes:
        prefix = self._underworld_prefix_for_endianness(existing, endianness)
        axis_blob = self._pack_axis_flags(prepared)
        flags_start = 0x49
        track_data_start = (flags_start + len(axis_blob) + 3) & ~3
        section_data_offset = track_data_start - 0x44
        section_count = int(existing.section_count if existing is not None and existing.section_count > 0 else 1)
        section_count = max(-128, min(127, section_count))
        underworld_unk_38 = int(existing.underworld_unk_38 if existing is not None else 0)
        underworld_unk_40 = int(existing.underworld_unk_40 if existing is not None else 0)

        payload = bytearray(prefix)
        payload.extend(struct.pack(
            f'{endianness}hhhBbiiiiB',
            int(prepared.anim_id),
            int(prepared.frame_count),
            int(prepared.time_per_key),
            int(prepared.bone_count),
            int(section_count),
            int(underworld_unk_38),
            0,
            int(underworld_unk_40),
            int(section_data_offset),
            int(prepared.transform_flags) & 0xFF,
        ))
        payload.extend(axis_blob)
        if len(payload) < track_data_start:
            payload.extend(b'\x00' * (track_data_start - len(payload)))

        for bone_index in range(prepared.bone_count):
            for transform_type in range(3):
                if prepared.transform_flags & (1 << transform_type) == 0:
                    continue
                flags = prepared.axis_flags[transform_type][bone_index]
                if flags == 0:
                    continue
                for axis in range(3):
                    if flags & (1 << axis):
                        payload.extend(self._pack_track(
                            prepared.tracks[transform_type][bone_index][axis],
                            endianness,
                            prepared.track_modes[transform_type][bone_index][axis],
                            prepared.linear_track_keys[transform_type][bone_index][axis],
                        ))

        file_size_from_current_position = max(0, len(payload) - 0x3C)
        struct.pack_into(f'{endianness}i', payload, 0x3C, int(file_size_from_current_position))
        return bytes(payload)

    def export_animation(
        self,
        context,
        armature: bpy.types.Object,
        action: bpy.types.Action,
        filepath: str,
        *,
        frame_start: int | None = None,
        frame_end: int | None = None,
        big_endian: bool = False,
        underworld: bool = False,
    ) -> _PreparedAnimation:
        path = Path(filepath)
        existing = self._parse_standalone_header(path) if path.exists() else None
        prepared = self.prepare_animation(
            context,
            armature,
            action,
            str(path),
            frame_start=frame_start,
            frame_end=frame_end,
            existing=existing,
        )
        endianness = '>' if bool(big_endian) else (existing.endianness if existing is not None else '<')
        export_underworld = bool(underworld) or (existing is not None and str(getattr(existing, 'format', '')).upper() == 'UNDERWORLD')
        if export_underworld:
            payload = self.build_underworld_section_payload(prepared, existing, endianness=endianness)
        else:
            payload = self.build_section_payload(prepared, existing, endianness=endianness)
        section_type = int(existing.section_type if existing is not None else 2)
        skip_flags = int(existing.skip_flags if existing is not None else 0)
        version_id = int(existing.version_id if existing is not None else 0)
        packed_data = int(existing.packed_data if existing is not None else 0) & 0xFF  # no relocations in anims, thank god
        section_id = int(existing.section_id if existing is not None and existing.section_id > 0 else prepared.anim_id)
        spec_mask = int(existing.spec_mask if existing is not None else 0xFFFFFFFF)
        magic = b'SECT' if endianness == '<' else b'TCES'

        blob = bytearray()
        blob.extend(magic)
        blob.extend(struct.pack(f'{endianness}iBBHIII', len(payload), section_type & 0xFF, skip_flags & 0xFF, version_id & 0xFFFF, packed_data & 0xFFFFFFFF, section_id & 0xFFFFFFFF, spec_mask & 0xFFFFFFFF))
        blob.extend(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(blob))
        logger.info('Exported TRLAU animation %s: format=%s anim_id=%d frames=%d time_per_key=%d bones=%d flags=0x%X endian=%s', path.name, 'UNDERWORLD' if export_underworld else 'TR7AE', prepared.anim_id, prepared.frame_count, prepared.time_per_key, prepared.bone_count, prepared.transform_flags, 'big' if endianness == '>' else 'little')
        return prepared


def register():
    pass


def unregister():
    pass
