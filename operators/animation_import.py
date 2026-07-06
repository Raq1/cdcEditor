from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import bpy
import mathutils
from mathutils import Vector

from ..core.blender_mesh_utils import armature_bone_count
from ..core.log import logger
from .animation_types import (
    TRLAUBoneAnimation,
    TRLAUKeyframe,
    TRLAUTrack,
    TRLAUTransformAnimation,
    TRLAUTransformType,
)


class AnimationImporterMixin:

    def _ensure_action_fcurve(self, action: bpy.types.Action, datablock: bpy.types.ID, data_path: str, index: int):
        if hasattr(action, 'fcurve_ensure_for_datablock'):
            try:
                return action.fcurve_ensure_for_datablock(datablock, data_path, index=index)
            except RuntimeError:
                pass

        curve = action.fcurves.find(data_path, index=index)
        return curve if curve is not None else action.fcurves.new(data_path=data_path, index=index)

    @staticmethod
    def _reset_pose_bones(armature: bpy.types.Object) -> None:
        for bone in armature.pose.bones:
            bone.location = (0.0, 0.0, 0.0)
            bone.rotation_mode = 'AXIS_ANGLE'
            bone.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
            bone.scale = (1.0, 1.0, 1.0)

    def _get_unique_action_name(self, base_name: str) -> str:
        action_name = base_name
        suffix = 1
        while action_name in bpy.data.actions:
            action_name = f'{base_name}_{suffix}'
            suffix += 1
        return action_name

    @staticmethod
    def _read_animation_axis_flags(
        data: bytes,
        offset: int,
        transform_type_flags: int,
        bone_count: int,
        *,
        align_transform_groups: bool = True,
    ) -> list[list[int] | None]:
        axis_flag_bits = int.from_bytes(data[offset:offset + 512], 'little')
        bit_pos = 0
        axis_flags_per_transform_type: list[list[int] | None] = [None, None, None]

        for transform_type in range(3):
            if transform_type_flags & (1 << transform_type) == 0:
                continue

            axis_flags = [(axis_flag_bits >> (bit_pos + (bone_index * 3))) & 0b111 for bone_index in range(bone_count)]
            bit_pos += bone_count * 3
            if align_transform_groups and bit_pos % 8:
                bit_pos += 8 - (bit_pos % 8)
            axis_flags_per_transform_type[transform_type] = axis_flags

        return axis_flags_per_transform_type

    @staticmethod
    def _read_animation_track_data(stream: BytesIO, endianness: str, key_count: int) -> TRLAUTrack:
        header = stream.read(4)
        if len(header) < 4:
            return TRLAUTrack()

        mode, num_keyframes = struct.unpack(endianness + 'HH', header)
        if mode == 1:
            return TRLAUTrack([TRLAUKeyframe(0, struct.unpack(endianness + 'f', stream.read(4))[0])], mode=1)
        if mode == 0:
            return TRLAUTrack([TRLAUKeyframe(frame, struct.unpack(endianness + 'f', stream.read(4))[0]) for frame in range(key_count)], mode=0)
        if mode != 2 or num_keyframes <= 0:
            return TRLAUTrack(mode=int(mode))

        absolute_frames = [0]
        absolute_frame = 0
        for _ in range(num_keyframes - 1):
            absolute_frame += struct.unpack(endianness + 'B', stream.read(1))[0]
            absolute_frames.append(absolute_frame)

        if stream.tell() % 4:
            stream.seek((stream.tell() + 3) & ~3)

        absolute_value = 0.0
        keyframes = []
        for frame in absolute_frames:
            absolute_value += struct.unpack(endianness + 'f', stream.read(4))[0]
            keyframes.append(TRLAUKeyframe(frame, absolute_value))
        return TRLAUTrack(keyframes, mode=2)

    @staticmethod
    def _read_psp_half_float(stream: BytesIO) -> float:
        value = stream.read(2)
        if len(value) < 2:
            raise EOFError('Unexpected end of PSP animation track while reading hfloat')
        return float(struct.unpack('<e', value)[0])

    @classmethod
    def _read_psp_animation_track_data(cls, stream: BytesIO, key_count: int) -> TRLAUTrack:
        mode_data = stream.read(2)
        if len(mode_data) < 2:
            return TRLAUTrack()

        mode = struct.unpack('<h', mode_data)[0]
        if mode == 1:  # Constant
            return TRLAUTrack([TRLAUKeyframe(0, cls._read_psp_half_float(stream))], mode=1)

        if mode == 0:  # Raw
            return TRLAUTrack([TRLAUKeyframe(frame, cls._read_psp_half_float(stream)) for frame in range(max(int(key_count), 0))], mode=0)

        if mode != 2:  # Unknown
            return TRLAUTrack(mode=int(mode))

        keyframe_count_data = stream.read(2)
        if len(keyframe_count_data) < 2:
            return TRLAUTrack(mode=2)
        keyframe_count = struct.unpack('<H', keyframe_count_data)[0]
        if keyframe_count <= 0:
            return TRLAUTrack(mode=2)

        time_indices_start = stream.tell()
        absolute_frames = [0]
        absolute_frame = 0
        for _ in range(keyframe_count - 1):
            time_delta_data = stream.read(1)
            if len(time_delta_data) < 1:
                return TRLAUTrack(mode=2)
            absolute_frame += time_delta_data[0]
            absolute_frames.append(absolute_frame)

        padded_time_indices_size = ((keyframe_count + 2) >> 2) * 4
        stream.seek(time_indices_start + padded_time_indices_size)

        absolute_value = 0.0
        keyframes: list[TRLAUKeyframe] = []
        for frame in absolute_frames:
            absolute_value += cls._read_psp_half_float(stream)
            keyframes.append(TRLAUKeyframe(frame, absolute_value))
        return TRLAUTrack(keyframes, mode=2)

    @staticmethod
    def _psp_section_data_size(data: bytes, path_name: str) -> int:
        if len(data) < 0x18:
            raise ValueError(f'PSP animation file is too small: {path_name}')
        if data[:4] != b'SECT':
            raise ValueError(f'Invalid PSP animation signature in {path_name}: {data[:4]!r}')
        packed_data = struct.unpack_from('<I', data, 0x0C)[0]
        num_relocations = packed_data >> 8
        section_data_size = 0x18 + (int(num_relocations) * 0x08)
        if section_data_size + 0x25 > len(data):
            raise ValueError(f'Invalid PSP animation section header in {path_name}')
        return section_data_size

    @classmethod
    def _parse_psp_animation_file_data(cls, data: bytes, path: Path) -> dict:
        section_data_size = cls._psp_section_data_size(data, path.name)
        anim_base = section_data_size

        anim_id = struct.unpack_from('<h', data, anim_base + 0x18)[0]
        frame_count = struct.unpack_from('<h', data, anim_base + 0x1A)[0]
        time_per_frame = struct.unpack_from('<h', data, anim_base + 0x1C)[0]
        bone_count = struct.unpack_from('<B', data, anim_base + 0x1E)[0]
        section_count = struct.unpack_from('<B', data, anim_base + 0x1F)[0]
        section_data_offset = struct.unpack_from('<i', data, anim_base + 0x20)[0]
        transform_type_flags = data[anim_base + 0x24]
        final_frame = max(frame_count - 1, 0)

        if frame_count <= 0 or frame_count > 65535:
            raise ValueError(f'Invalid PSP animation frame count {frame_count} in {path.name}')
        if bone_count <= 0 or bone_count > 255:
            raise ValueError(f'Invalid PSP animation bone count {bone_count} in {path.name}')
        if transform_type_flags <= 0 or (transform_type_flags & ~0b111):
            raise ValueError(f'Invalid PSP animation transform flags 0x{transform_type_flags:X} in {path.name}')

        track_data_start = anim_base + 0x20 + section_data_offset
        if track_data_start <= anim_base + 0x24 or track_data_start >= len(data):
            raise ValueError(f'Invalid PSP animation track data offset 0x{section_data_offset:X} in {path.name}')

        axis_flags_per_transform = cls._read_animation_axis_flags(data, anim_base + 0x25, transform_type_flags, bone_count)
        has_any_transform_axis = any(
            any(int(flags) != 0 for flags in axis_flags)
            for axis_flags in axis_flags_per_transform
            if axis_flags is not None
        )
        if not has_any_transform_axis:
            raise ValueError(f'PSP animation contains no transform tracks: {path.name}')

        stream = BytesIO(data)
        stream.seek(track_data_start)
        bone_anims: list[TRLAUBoneAnimation] = []
        for bone_index in range(bone_count):
            bone_anim = TRLAUBoneAnimation()

            scale_flags = axis_flags_per_transform[TRLAUTransformType.SCALE]
            if scale_flags is not None and scale_flags[bone_index]:
                scale_anim = TRLAUTransformAnimation()
                for axis in range(3):
                    if scale_flags[bone_index] & (1 << axis):
                        scale_anim.axis_tracks[axis] = cls._read_psp_animation_track_data(stream, frame_count)
                bone_anim.transforms[TRLAUTransformType.SCALE] = scale_anim

            rotation_flags = axis_flags_per_transform[TRLAUTransformType.ROTATION]
            if rotation_flags is not None and rotation_flags[bone_index]:
                rotation_anim = TRLAUTransformAnimation()
                for axis in range(3):
                    if rotation_flags[bone_index] & (1 << axis):
                        rotation_anim.axis_tracks[axis] = cls._read_psp_animation_track_data(stream, frame_count)
                bone_anim.transforms[TRLAUTransformType.ROTATION] = rotation_anim

            location_flags = axis_flags_per_transform[TRLAUTransformType.LOCATION]
            if location_flags is not None and location_flags[bone_index]:
                location_anim = TRLAUTransformAnimation()
                for axis in range(3):
                    if location_flags[bone_index] & (1 << axis):
                        location_anim.axis_tracks[axis] = cls._read_psp_animation_track_data(stream, frame_count)
                bone_anim.transforms[TRLAUTransformType.LOCATION] = location_anim

            bone_anims.append(bone_anim)

        return {
            'path': path,
            'endianness': '<',
            'platform': 'PSP',
            'anim_id': anim_id,
            'frame_count': frame_count,
            'time_per_frame': time_per_frame,
            'bone_count': bone_count,
            'section_count': section_count,
            'section_data_offset': section_data_offset,
            'track_values_offset': section_data_offset,
            'transform_type_flags': transform_type_flags,
            'final_frame': final_frame,
            'bone_anims': bone_anims,
        }

    def _create_bone_location_curves(
        self,
        action: bpy.types.Action,
        armature_obj: bpy.types.Object,
        pose_bone: bpy.types.PoseBone,
        location_anim: TRLAUTransformAnimation,
    ) -> None:
        data_path = f'pose.bones["{pose_bone.name}"].location'
        if pose_bone.parent:
            basis = pose_bone.parent.bone.matrix_local @ pose_bone.bone.matrix_local.inverted()
        else:
            basis = pose_bone.bone.matrix_local.inverted()

        for axis, axis_track in enumerate(location_anim.axis_tracks):
            if axis_track is None:
                continue
            curve = self._ensure_action_fcurve(action, armature_obj, data_path, axis)
            for keyframe in axis_track.keyframes:
                translation = Vector((0.0, 0.0, 0.0))
                translation[axis] = keyframe.value
                local_value = (basis @ mathutils.Matrix.Translation(translation)).to_translation()[axis]
                point = curve.keyframe_points.insert(frame=keyframe.frame, value=local_value)
                point.interpolation = 'LINEAR'

    def _create_bone_rotation_curves(
        self,
        action: bpy.types.Action,
        armature_obj: bpy.types.Object,
        pose_bone: bpy.types.PoseBone,
        rotation_anim: TRLAUTransformAnimation,
    ) -> None:
        pose_bone.rotation_mode = 'QUATERNION'
        data_path = f'pose.bones["{pose_bone.name}"].rotation_quaternion'
        curves = [self._ensure_action_fcurve(action, armature_obj, data_path, index) for index in range(4)]

        if pose_bone.parent:
            basis = (pose_bone.parent.bone.matrix_local @ pose_bone.bone.matrix_local.inverted()).to_3x3()
        else:
            basis = pose_bone.bone.matrix_local.inverted().to_3x3()

        previous_quaternion = None
        for frame in rotation_anim.get_keyframe_frames():
            rotation_vector = rotation_anim.get_vector_at_frame(frame, 0.0)
            angle = rotation_vector.length
            if angle < 1e-8:
                quaternion = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
            else:
                source_quaternion = mathutils.Quaternion(rotation_vector.normalized(), angle)
                quaternion = (basis @ source_quaternion.to_matrix() @ basis.inverted()).to_quaternion()
            if previous_quaternion is not None and previous_quaternion.dot(quaternion) < 0.0:
                quaternion.negate()
            previous_quaternion = quaternion.copy()
            for curve_index, value in enumerate((quaternion.w, quaternion.x, quaternion.y, quaternion.z)):
                point = curves[curve_index].keyframe_points.insert(frame=frame, value=value)
                point.interpolation = 'LINEAR'

    def _create_bone_scale_curves(
        self,
        action: bpy.types.Action,
        armature_obj: bpy.types.Object,
        pose_bone: bpy.types.PoseBone,
        scale_anim: TRLAUTransformAnimation,
    ) -> None:
        data_path = f'pose.bones["{pose_bone.name}"].scale'
        for axis, axis_track in enumerate(scale_anim.axis_tracks):
            if axis_track is None:
                continue
            curve = self._ensure_action_fcurve(action, armature_obj, data_path, axis)
            for keyframe in axis_track.keyframes:
                point = curve.keyframe_points.insert(frame=keyframe.frame, value=keyframe.value)
                point.interpolation = 'LINEAR'

    def _get_action_fcurves(self, action: bpy.types.Action):
        try:
            for fcurve in getattr(action, 'fcurves', []) or []:
                yield fcurve
        except Exception:
            pass

        try:
            for layer in getattr(action, 'layers', []) or []:
                for layer_strip in getattr(layer, 'strips', []) or []:
                    for channelbag in getattr(layer_strip, 'channelbags', []) or []:
                        for fcurve in getattr(channelbag, 'fcurves', []) or []:
                            yield fcurve
        except Exception:
            pass

    def _action_frame_bounds(self, action: bpy.types.Action, final_frame: int) -> tuple[float, float]:
        starts: list[float] = []
        ends: list[float] = []

        for fcurve in self._get_action_fcurves(action):
            try:
                for point in fcurve.keyframe_points:
                    frame = float(point.co[0])
                    starts.append(frame)
                    ends.append(frame)
            except Exception:
                pass

        if not starts or not ends:
            try:
                start, end = action.frame_range
                starts.append(float(start))
                ends.append(float(end))
            except Exception:
                starts.append(0.0)
                ends.append(max(float(final_frame), 1.0))

        action_start = min(starts) if starts else 0.0
        action_end = max(ends) if ends else max(float(final_frame), 1.0)
        if action_end <= action_start:
            action_end = action_start + max(float(final_frame), 1.0)
        if action_end <= action_start:
            action_end = action_start + 1.0
        return action_start, action_end

    @staticmethod
    def _slot_belongs_to_action(action: bpy.types.Action, slot) -> bool:
        if action is None or slot is None:
            return False
        try:
            return any(candidate == slot for candidate in getattr(action, 'slots', []) or [])
        except Exception:
            return False

    def _assign_action_and_slot(self, anim_data, action: bpy.types.Action):
        anim_data.action = action
        selected_slot = None
        if hasattr(anim_data, 'action_slot'):
            try:
                current_slot = getattr(anim_data, 'action_slot', None)
                if self._slot_belongs_to_action(action, current_slot):
                    selected_slot = current_slot
            except Exception:
                selected_slot = None

            if selected_slot is None:
                try:
                    slots = list(getattr(action, 'slots', []) or [])
                    if slots:
                        selected_slot = slots[0]
                        anim_data.action_slot = selected_slot
                except Exception:
                    selected_slot = None
        return selected_slot

    @staticmethod
    def _assign_strip_slot(strip, slot) -> None:
        if strip is None or slot is None:
            return
        try:
            strip.action_slot = slot
            return
        except Exception:
            pass
        try:
            strip.action_slot_handle = slot.handle
        except Exception:
            pass

    def _create_looping_nla_strip(
        self,
        context,
        armature: bpy.types.Object,
        action: bpy.types.Action,
        final_frame: int,
        *,
        track_name: str = 'IntroData Animation',
        strip_name: str | None = None,
        repeat_count: float = 100.0,
    ):
        if armature is None or getattr(armature, 'type', None) != 'ARMATURE' or action is None:
            return None

        anim_data = armature.animation_data_create()
        action.use_fake_user = True
        repeat_count = max(float(repeat_count), 1.0)
        strip_label = strip_name or action.name

        action_start, action_end = self._action_frame_bounds(action, final_frame)
        strip_start = int(action_start)
        action_length = max(float(action_end) - float(action_start), 1.0)

        created_track = None
        created_strip = None
        selected_slot = None

        try:
            try:
                anim_data.use_nla = True
            except Exception:
                pass

            selected_slot = self._assign_action_and_slot(anim_data, action)

            created_track = anim_data.nla_tracks.new()
            created_track.name = track_name
            created_strip = created_track.strips.new(strip_label, strip_start, action)
            self._assign_strip_slot(created_strip, selected_slot)

            try:
                created_strip.name = strip_label
            except Exception:
                pass
            try:
                created_strip.action_frame_start = float(action_start)
                created_strip.action_frame_end = float(action_end)
            except Exception:
                pass
            try:
                created_strip.frame_start = float(strip_start)
                created_strip.frame_end = float(strip_start) + action_length * repeat_count
            except Exception:
                pass
            try:
                created_strip.scale = 1.0
            except Exception:
                pass
            try:
                created_strip.repeat = repeat_count
            except Exception:
                pass
            try:
                created_strip.blend_type = 'REPLACE'
            except Exception:
                pass
            try:
                created_strip.extrapolation = 'NOTHING'
            except Exception:
                pass
            try:
                for nla_track in anim_data.nla_tracks:
                    nla_track.select = nla_track == created_track
                    for candidate in nla_track.strips:
                        candidate.select = candidate == created_strip
                anim_data.nla_tracks.active = created_track
                created_strip.active = True
            except Exception:
                pass

            # *shrug*
            anim_data.action = None
            try:
                if hasattr(anim_data, 'action_slot'):
                    anim_data.action_slot = None
            except Exception:
                pass

            try:
                armature.update_tag(refresh={'OBJECT', 'DATA', 'TIME'})
            except Exception:
                pass
            try:
                if context is not None and getattr(context, 'scene', None) is not None:
                    context.scene.frame_set(context.scene.frame_current)
            except Exception:
                pass

            return created_strip
        except Exception as exc:
            logger.warning('Failed to push down action %s onto %s: %s', getattr(action, 'name', '<unnamed>'), getattr(armature, 'name', '<unnamed>'), exc)
            try:
                self._assign_action_and_slot(anim_data, action)
            except Exception:
                pass
            return None

    @staticmethod
    def _pose_bone_for_animation_index(armature: bpy.types.Object, bone_index: int):
        bone_name = f'bone_{int(bone_index)}'
        try:
            pose_bone = armature.pose.bones.get(bone_name)
        except Exception:
            pose_bone = None
        if pose_bone is not None:
            return pose_bone

        try:
            if 0 <= int(bone_index) < len(armature.pose.bones):
                return armature.pose.bones[int(bone_index)]
        except Exception:
            pass
        return None

    @staticmethod
    def _armature_has_animation_index_bones(armature: bpy.types.Object, bone_count: int) -> bool:
        try:
            pose_bones = armature.pose.bones
        except Exception:
            return False
        for bone_index in range(max(0, int(bone_count))):
            try:
                if pose_bones.get(f'bone_{bone_index}') is None:
                    return False
            except Exception:
                return False
        return True

    def _build_action_from_bone_anims(
        self,
        context,
        armature: bpy.types.Object,
        base_name: str,
        bone_anims: list[TRLAUBoneAnimation],
        final_frame: int,
        time_per_frame: int,
        extra_properties: dict | None = None,
        assign_to_armature: bool = True,
    ) -> bpy.types.Action:
        action = bpy.data.actions.new(name=self._get_unique_action_name(base_name))
        for key, value in (extra_properties or {}).items():
            action[key] = value
        if assign_to_armature:
            self._assign_action_and_slot(armature.animation_data_create(), action)

        self._reset_pose_bones(armature)
        skipped_bone_indices: list[int] = []
        for bone_index, bone_anim in enumerate(bone_anims):
            pose_bone = self._pose_bone_for_animation_index(armature, bone_index)
            if pose_bone is None:
                skipped_bone_indices.append(int(bone_index))
                continue
            location_anim = bone_anim.transforms[TRLAUTransformType.LOCATION]
            rotation_anim = bone_anim.transforms[TRLAUTransformType.ROTATION]
            scale_anim = bone_anim.transforms[TRLAUTransformType.SCALE]
            if location_anim is not None:
                self._create_bone_location_curves(action, armature, pose_bone, location_anim)
            if rotation_anim is not None:
                self._create_bone_rotation_curves(action, armature, pose_bone, rotation_anim)
            if scale_anim is not None:
                self._create_bone_scale_curves(action, armature, pose_bone, scale_anim)

        if skipped_bone_indices:
            logger.warning(
                'Animation %s skipped %d bone track(s) because matching bone_X names were not found: %s',
                action.name,
                len(skipped_bone_indices),
                ', '.join(str(index) for index in skipped_bone_indices[:12]),
            )

        if assign_to_armature:
            self._assign_action_and_slot(armature.animation_data_create(), action)

        context.scene.frame_start = 0
        context.scene.frame_end = max(int(context.scene.frame_end), int(final_frame))
        context.scene.frame_current = 0
        return action

    @staticmethod
    def _animation_endianness_for_data(data: bytes, default_endianness: str | None = None) -> str:
        magic = data[:4]
        if magic == b'SECT':
            return '<'
        if magic == b'TCES':
            return '>'
        if magic == b'DRM\x00':
            return default_endianness if default_endianness in {'<', '>'} else '<'
        raise ValueError(f'Unknown animation file signature: {magic!r}')

    @staticmethod
    def _standalone_animation_payload_base(data: bytes, endianness: str, path_name: str) -> tuple[int, int | None]:
        magic = data[:4]
        if magic in {b'SECT', b'TCES'}:
            if len(data) < 0x18:
                raise ValueError(f'Animation section is too small: {path_name}')
            packed_data = struct.unpack_from(endianness + 'I', data, 0x0C)[0]
            num_relocations = (packed_data >> 8) & 0x00FFFFFF
            anim_base = 0x18 + (int(num_relocations) * 0x08)
            section_id = struct.unpack_from(endianness + 'I', data, 0x10)[0]
            return anim_base, int(section_id)
        return 0, None

    @staticmethod
    def _validate_animation_header(path_name: str, frame_count: int, bone_count: int, transform_type_flags: int) -> None:
        if frame_count <= 0 or frame_count > 65535:
            raise ValueError(f'Invalid animation frame count {frame_count} in {path_name}')
        if bone_count <= 0 or bone_count > 255:
            raise ValueError(f'Invalid animation bone count {bone_count} in {path_name}')
        if transform_type_flags <= 0 or (transform_type_flags & ~0b111):
            raise ValueError(f'Invalid animation transform flags 0x{transform_type_flags:X} in {path_name}')

    def _read_bone_animation_tracks(
        self,
        data: bytes,
        *,
        path_name: str,
        stream_offset: int,
        endianness: str,
        frame_count: int,
        bone_count: int,
        transform_type_flags: int,
        axis_flags_offset: int,
        align_transform_groups: bool = True,
    ) -> list[TRLAUBoneAnimation]:
        if stream_offset < 0 or stream_offset >= len(data):
            raise ValueError(f'Invalid animation track data offset 0x{stream_offset:X} in {path_name}')

        stream = BytesIO(data)
        stream.seek(stream_offset)
        axis_flags_per_transform = self._read_animation_axis_flags(
            data,
            axis_flags_offset,
            transform_type_flags,
            bone_count,
            align_transform_groups=align_transform_groups,
        )
        has_any_transform_axis = any(
            any(int(flags) != 0 for flags in axis_flags)
            for axis_flags in axis_flags_per_transform
            if axis_flags is not None
        )
        if not has_any_transform_axis:
            raise ValueError(f'Animation contains no transform tracks: {path_name}')

        bone_anims: list[TRLAUBoneAnimation] = []
        for bone_index in range(bone_count):
            bone_anim = TRLAUBoneAnimation()
            for transform_type, axis_flags in enumerate(axis_flags_per_transform):
                if axis_flags is None:
                    continue
                transform_anim = TRLAUTransformAnimation()
                has_any_axis = False
                for axis in range(3):
                    if axis_flags[bone_index] & (1 << axis):
                        transform_anim.axis_tracks[axis] = self._read_animation_track_data(stream, endianness, frame_count)
                        has_any_axis = True
                if has_any_axis:
                    bone_anim.transforms[transform_type] = transform_anim
            bone_anims.append(bone_anim)
        return bone_anims

    @staticmethod
    def _platform_requests_underworld_animation(platform: str | None) -> bool:
        normalized = str(platform or '').strip().upper()
        return normalized in {'UNDERWORLD', 'TR8', 'TRU', 'TOMB RAIDER UNDERWORLD'}

    @classmethod
    def _looks_like_tr7ae_animation_header(cls, data: bytes, endianness: str, path_name: str) -> bool:
        try:
            anim_base, _section_id = cls._standalone_animation_payload_base(data, endianness, path_name)
            if anim_base + 0x25 > len(data):
                return False
            frame_count = struct.unpack_from(endianness + 'h', data, anim_base + 0x1A)[0]
            bone_count = struct.unpack_from(endianness + 'B', data, anim_base + 0x1E)[0]
            track_values_offset = struct.unpack_from(endianness + 'I', data, anim_base + 0x20)[0]
            transform_type_flags = data[anim_base + 0x24]
            track_data_start = anim_base + 0x20 + int(track_values_offset)
            return (
                0 < int(frame_count) <= 65535
                and 0 < int(bone_count) <= 255
                and 0 < int(transform_type_flags) <= 0b111
                and track_data_start > anim_base + 0x24
                and track_data_start < len(data)
            )
        except Exception:
            return False

    @classmethod
    def _looks_like_underworld_animation_header(cls, data: bytes, endianness: str, path_name: str) -> bool:
        try:
            anim_base, _section_id = cls._standalone_animation_payload_base(data, endianness, path_name)
            if anim_base + 0x49 > len(data):
                return False
            frame_count = struct.unpack_from(endianness + 'h', data, anim_base + 0x32)[0]
            bone_count = struct.unpack_from(endianness + 'B', data, anim_base + 0x36)[0]
            section_data_offset = struct.unpack_from(endianness + 'i', data, anim_base + 0x44)[0]
            transform_type_flags = data[anim_base + 0x48]
            track_data_start = anim_base + 0x44 + int(section_data_offset)
            return (
                0 < int(frame_count) <= 65535
                and 0 < int(bone_count) <= 255
                and 0 < int(transform_type_flags) <= 0b111
                and track_data_start > anim_base + 0x48
                and track_data_start < len(data)
            )
        except Exception:
            return False

    def _parse_underworld_animation_file_data(self, data: bytes, path: Path, default_endianness: str | None = None, platform: str | None = None) -> dict:
        endianness = self._animation_endianness_for_data(data, default_endianness)
        if len(data) < 0x61:
            raise ValueError(f'Underworld animation file is too small: {path.name}')

        anim_base, section_id = self._standalone_animation_payload_base(data, endianness, path.name)
        if anim_base + 0x49 >= len(data):
            raise ValueError(f'Underworld animation section payload is too small: {path.name}')

        fragment_anim_id = struct.unpack_from(endianness + 'h', data, anim_base + 0x30)[0]
        anim_id = int(fragment_anim_id if fragment_anim_id >= 0 else (section_id if section_id is not None else fragment_anim_id))
        frame_count = struct.unpack_from(endianness + 'h', data, anim_base + 0x32)[0]
        time_per_frame = struct.unpack_from(endianness + 'h', data, anim_base + 0x34)[0]
        bone_count = struct.unpack_from(endianness + 'B', data, anim_base + 0x36)[0]
        section_count = struct.unpack_from(endianness + 'b', data, anim_base + 0x37)[0]
        file_size_from_current_position = struct.unpack_from(endianness + 'i', data, anim_base + 0x3C)[0]
        section_data_offset = struct.unpack_from(endianness + 'i', data, anim_base + 0x44)[0]
        transform_type_flags = data[anim_base + 0x48]
        final_frame = max(frame_count - 1, 0)

        self._validate_animation_header(path.name, frame_count, bone_count, transform_type_flags)

        track_data_start = anim_base + 0x44 + int(section_data_offset)
        if track_data_start <= anim_base + 0x48 or track_data_start >= len(data):
            raise ValueError(f'Invalid Underworld animation track data offset 0x{section_data_offset:X} in {path.name}')

        bone_anims = self._read_bone_animation_tracks(
            data,
            path_name=path.name,
            stream_offset=track_data_start,
            endianness=endianness,
            frame_count=frame_count,
            bone_count=bone_count,
            transform_type_flags=transform_type_flags,
            axis_flags_offset=anim_base + 0x49,
            align_transform_groups=True,
        )

        return {
            'path': path,
            'endianness': endianness,
            'platform': 'UNDERWORLD',
            'anim_id': anim_id,
            'frame_count': frame_count,
            'time_per_frame': time_per_frame,
            'bone_count': bone_count,
            'section_count': section_count,
            'file_size_from_current_position': file_size_from_current_position,
            'section_data_offset': section_data_offset,
            'track_values_offset': section_data_offset,
            'transform_type_flags': transform_type_flags,
            'final_frame': final_frame,
            'bone_anims': bone_anims,
        }

    def _parse_animation_file(self, filepath: str, default_endianness: str | None = None, platform: str | None = None) -> dict:
        path = Path(filepath)
        data = path.read_bytes()
        if str(platform or '').upper() == 'PSP':
            return self._parse_psp_animation_file_data(data, path)

        endianness = self._animation_endianness_for_data(data, default_endianness)
        if self._platform_requests_underworld_animation(platform):
            return self._parse_underworld_animation_file_data(data, path, default_endianness=default_endianness, platform=platform)
        if (
            not self._looks_like_tr7ae_animation_header(data, endianness, path.name)
            and self._looks_like_underworld_animation_header(data, endianness, path.name)
        ):
            return self._parse_underworld_animation_file_data(data, path, default_endianness=default_endianness, platform=platform)

        if len(data) < 0x3D:
            raise ValueError(f'Animation file is too small: {path.name}')

        anim_base, section_id = self._standalone_animation_payload_base(data, endianness, path.name)
        if anim_base + 0x25 >= len(data):
            raise ValueError(f'Animation section payload is too small: {path.name}')

        anim_id_offset = 0x10 if section_id is not None else 0x10
        anim_id = struct.unpack_from(endianness + 'h', data, anim_id_offset)[0]
        frame_count = struct.unpack_from(endianness + 'h', data, anim_base + 0x1A)[0]
        time_per_frame = struct.unpack_from(endianness + 'h', data, anim_base + 0x1C)[0]
        bone_count = struct.unpack_from(endianness + 'B', data, anim_base + 0x1E)[0]
        track_values_offset = struct.unpack_from(endianness + 'I', data, anim_base + 0x20)[0]
        transform_type_flags = data[anim_base + 0x24]
        final_frame = max(frame_count - 1, 0)

        self._validate_animation_header(path.name, frame_count, bone_count, transform_type_flags)

        track_data_start = anim_base + 0x20 + track_values_offset
        if track_data_start <= anim_base + 0x24 or track_data_start >= len(data):
            raise ValueError(f'Invalid animation track data offset 0x{track_values_offset:X} in {path.name}')

        bone_anims = self._read_bone_animation_tracks(
            data,
            path_name=path.name,
            stream_offset=track_data_start,
            endianness=endianness,
            frame_count=frame_count,
            bone_count=bone_count,
            transform_type_flags=transform_type_flags,
            axis_flags_offset=anim_base + 0x25,
            align_transform_groups=True,
        )

        return {
            'path': path,
            'endianness': endianness,
            'platform': str(platform or 'AUTO'),
            'anim_id': anim_id,
            'frame_count': frame_count,
            'time_per_frame': time_per_frame,
            'bone_count': bone_count,
            'track_values_offset': track_values_offset,
            'transform_type_flags': transform_type_flags,
            'final_frame': final_frame,
            'bone_anims': bone_anims,
        }

    def _probe_animation_file(self, filepath: str, default_endianness: str | None = None, platform: str | None = None) -> dict | None:
        try:
            parsed = self._parse_animation_file(filepath, default_endianness=default_endianness, platform=platform)
        except Exception:
            return None
        return parsed

    @staticmethod
    def _animation_armature_bone_count(armature: bpy.types.Object) -> int:
        return armature_bone_count(armature)

    def _match_animation_armature(self, armatures: list[bpy.types.Object], bone_count: int) -> bpy.types.Object | None:
        for armature in armatures:
            if self._animation_armature_bone_count(armature) == bone_count and self._armature_has_animation_index_bones(armature, bone_count):
                return armature
        for armature in armatures:
            if self._animation_armature_bone_count(armature) == bone_count:
                return armature
        for armature in armatures:
            if self._armature_has_animation_index_bones(armature, bone_count):
                return armature
        if len(armatures) == 1:
            return armatures[0]
        return None

    @staticmethod
    def _unique_animation_armatures(context, armatures: list[bpy.types.Object] | None = None) -> list[bpy.types.Object]:
        candidates: list[bpy.types.Object] = []
        seen_names: set[str] = set()

        def add_candidate(obj):
            if obj is None or getattr(obj, 'type', None) != 'ARMATURE':
                return
            if obj.name in seen_names:
                return
            seen_names.add(obj.name)
            candidates.append(obj)

        for obj in armatures or []:
            add_candidate(obj)
        add_candidate(getattr(context, 'active_object', None))
        for obj in getattr(context, 'selected_objects', []) or []:
            add_candidate(obj)
        return candidates

    def _find_animation_files(
        self,
        filepaths: list[Path],
        default_endianness: str | None = None,
        platform: str | None = None,
        allowed_anim_ids: list[int] | None = None,
    ) -> list[tuple[Path, dict]]:
        candidates: list[tuple[Path, dict]] = []
        allowed_order = {int(anim_id): index for index, anim_id in enumerate(allowed_anim_ids or [])}
        for path in sorted((Path(filepath) for filepath in filepaths), key=lambda item: item.name):
            if not path.is_file():
                continue
            if path.suffix.lower() in {'.pcd', '.level', '.obj'}:
                continue
            parsed = self._probe_animation_file(str(path), default_endianness=default_endianness, platform=platform)
            if parsed is None:
                continue
            if allowed_order and int(parsed.get('anim_id', -1)) not in allowed_order:
                continue
            candidates.append((path, parsed))

        if allowed_order:
            candidates.sort(key=lambda item: (allowed_order.get(int(item[1].get('anim_id', -1)), 1_000_000), item[0].name))
        return candidates

    def _import_all_animation_files(
        self,
        context,
        filepaths: list[Path],
        *,
        armatures: list[bpy.types.Object] | None = None,
        source_name: str | None = None,
        default_endianness: str | None = None,
        platform: str | None = None,
        allowed_anim_ids: list[int] | None = None,
    ) -> list[dict]:
        animation_files = self._find_animation_files(filepaths, default_endianness=default_endianness, platform=platform, allowed_anim_ids=allowed_anim_ids)
        if not animation_files:
            return []

        armature_candidates = self._unique_animation_armatures(context, armatures)
        if not armature_candidates:
            logger.warning('Import All Animations found %d animation file(s), but no armature is available.', len(animation_files))
            return []

        imported_results: list[dict] = []
        for animation_path, parsed in animation_files:
            bone_count = int(parsed.get('bone_count', 0) or 0)
            armature = self._match_animation_armature(armature_candidates, bone_count)
            if armature is None:
                logger.warning(
                    'Skipping animation %s: no compatible armature with %d bone(s).',
                    animation_path.name,
                    bone_count,
                )
                continue

            action_name = f'{source_name}_{animation_path.stem}' if source_name else animation_path.stem
            try:
                imported_results.append(
                    self._import_animation_file(
                        context,
                        str(animation_path),
                        armature=armature,
                        default_endianness=default_endianness,
                        action_name=action_name,
                        platform=platform,
                    )
                )
            except Exception as exc:
                logger.warning('Failed to import animation %s: %s', animation_path.name, exc)
        return imported_results

    def _import_animation_file(
        self,
        context,
        filepath: str,
        *,
        armature: bpy.types.Object | None = None,
        default_endianness: str | None = None,
        action_name: str | None = None,
        platform: str | None = None,
    ) -> dict:
        armature = armature or context.active_object
        if not armature or armature.type != 'ARMATURE':
            raise ValueError('Select an Armature to import animation onto.')

        parsed = self._parse_animation_file(filepath, default_endianness=default_endianness, platform=platform)
        path = parsed['path']
        anim_id = parsed['anim_id']
        final_frame = parsed['final_frame']
        time_per_frame = parsed['time_per_frame']
        bone_anims = parsed['bone_anims']

        action = self._build_action_from_bone_anims(
            context,
            armature,
            action_name or path.stem,
            bone_anims,
            final_frame,
            time_per_frame,
        )
        context.scene.render.fps = 30
        logger.info('Imported animation %s onto %s', path.name, armature.name)
        return {
            'model': None,
            'mesh_obj': None,
            'mesh_objects': [],
            'arm_obj': armature,
            'faces': 0,
            'animation_action': action,
            'animation_final_frame': final_frame,
            'animation_anim_id': anim_id,
            'animation_time_per_frame': time_per_frame,
            'source': str(path),
        }
