from __future__ import annotations

from pathlib import Path
import math
import struct
import wave

import blf
import bpy
import mathutils
from bpy.props import BoolProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper

from ..core.blender_mesh_utils import armature_bone_count
from ..core.log import logger
from ..core.mul_subtitles import MUL_SUBTITLE_TEXT_PROP, decode_subtitle_set, normalize_subtitle_frames, parse_subtitle_text, subtitle_blob_has_data
from ..operators.animation_import import AnimationImporterMixin
from ..operators.mul_import import MultiplexStreamImporterMixin, MUL_MULTIPLEX_OBJECT_CHUNK_COUNT, MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX, MUL_BONE_FLAGS_PROP


MUL_EXPORT_PACKET_HEADER_SIZE = 0x10
MUL_EXPORT_STREAM_START_OFFSET = 0x800
MUL_EXPORT_ALIGNMENT = 0x10
MUL_EXPORT_PACKET_TYPE_SOUND = 0
MUL_EXPORT_PACKET_TYPE_CINEMATIC = 1
MUL_EXPORT_CAMERA_VISUAL_SCALE = 1.0
MUL_EXPORT_DEFAULT_VERSION = 5
MUL_EXPORT_DEFAULT_SKELETON_FLAGS = 5.0
MUL_EXPORT_ROOT_CHANNEL_0 = -1.0
MUL_EXPORT_ROOT_MATRIX_INDEX = 1.0

_MUL_PAYLOAD_CACHE: dict[tuple, dict] = {}
_MUL_METADATA_CACHE: dict[tuple, dict] = {}
_MUL_CACHE_MAX_ITEMS = 16


def _align_mul_export_offset(offset: int, alignment: int = MUL_EXPORT_ALIGNMENT) -> int:
    return (int(offset) + (int(alignment) - 1)) & ~(int(alignment) - 1)


def _pad_mul_export_alignment(buffer: bytearray, alignment: int = MUL_EXPORT_ALIGNMENT) -> None:
    aligned = _align_mul_export_offset(len(buffer), alignment)
    if aligned > len(buffer):
        buffer.extend(b'\x00' * (aligned - len(buffer)))


def _flatten_matrix_rows(matrix) -> list[float]:
    try:
        return [float(matrix[row][column]) for row in range(4) for column in range(4)]
    except Exception:
        return _flatten_matrix_rows(mathutils.Matrix.Identity(4))


def _matrix_from_stored_default(values) -> mathutils.Matrix:
    return MultiplexStreamImporterMixin._mul_default_transform_to_matrix(values)


def _iter_action_fcurves(action):
    if action is None:
        return
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


def _action_has_fcurves(action) -> bool:
    try:
        return any(True for _ in _iter_action_fcurves(action))
    except Exception:
        return False


def _build_action_fcurve_lookup(action) -> dict[tuple[str, int], object]:
    lookup: dict[tuple[str, int], object] = {}
    if action is None:
        return lookup
    for curve in _iter_action_fcurves(action):
        try:
            key = (str(getattr(curve, 'data_path', '') or ''), int(getattr(curve, 'array_index', 0)))
        except Exception:
            continue
        if key[0] and key not in lookup:
            lookup[key] = curve
    return lookup


def _find_action_fcurve(action, data_path: str, array_index: int, fcurve_lookup: dict | None = None):
    if fcurve_lookup is not None:
        try:
            return fcurve_lookup.get((str(data_path), int(array_index)))
        except Exception:
            return None
    if action is None:
        return None
    try:
        curve = action.fcurves.find(data_path, index=int(array_index))
        if curve is not None:
            return curve
    except Exception:
        pass
    for curve in _iter_action_fcurves(action):
        try:
            if getattr(curve, 'data_path', None) == data_path and int(getattr(curve, 'array_index', -1)) == int(array_index):
                return curve
        except Exception:
            pass
    return None


def _evaluate_action_value(action, data_path: str, array_index: int, frame: int, default: float, fcurve_lookup: dict | None = None) -> float:
    curve = _find_action_fcurve(action, data_path, array_index, fcurve_lookup)
    if curve is None:
        return float(default)
    try:
        return float(curve.evaluate(float(frame)))
    except Exception:
        return float(default)


def _pose_bone_custom_prop_path(pose_bone, prop_name: str) -> str:
    return f'pose.bones["{pose_bone.name}"]["{prop_name}"]'


def _evaluate_pose_bone_flags(action, pose_bone, frame: int, fcurve_lookup: dict | None = None) -> float | None:
    data_path = _pose_bone_custom_prop_path(pose_bone, MUL_BONE_FLAGS_PROP)
    curve = _find_action_fcurve(action, data_path, 0, fcurve_lookup)
    if curve is None:
        return None
    try:
        return float(int(round(curve.evaluate(float(frame)))))
    except Exception:
        try:
            return float(int(round(float(pose_bone.get(MUL_BONE_FLAGS_PROP, MUL_EXPORT_DEFAULT_SKELETON_FLAGS)))))
        except Exception:
            return float(MUL_EXPORT_DEFAULT_SKELETON_FLAGS)


def _action_frame_range(action) -> tuple[int, int] | None:
    if action is None:
        return None
    try:
        start, end = action.frame_range
        return int(math.floor(float(start))), int(math.ceil(float(end)))
    except Exception:
        return None


def _armature_has_non_rest_pose(armature: bpy.types.Object, tolerance: float = 1.0e-6) -> bool:
    """Return True when an armature carries an explicit static pose edit.

    A MUL row may have an assigned armature only as a binding target. If the
    armature has no action and no non-rest pose, sampling it would export bind
    pose over the stored MUL animation. That is the T-pose failure mode.
    """
    if armature is None or getattr(armature, 'type', None) != 'ARMATURE':
        return False
    try:
        pose_bones = list(getattr(getattr(armature, 'pose', None), 'bones', []) or [])
    except Exception:
        pose_bones = []
    for pose_bone in pose_bones:
        try:
            basis = pose_bone.matrix_basis
            for row in range(4):
                for column in range(4):
                    expected = 1.0 if row == column else 0.0
                    if abs(float(basis[row][column]) - expected) > float(tolerance):
                        return True
        except Exception:
            try:
                if any(abs(float(value)) > float(tolerance) for value in pose_bone.location):
                    return True
                if any(abs(float(value) - 1.0) > float(tolerance) for value in pose_bone.scale):
                    return True
                if str(getattr(pose_bone, 'rotation_mode', '') or '') == 'QUATERNION':
                    quat = pose_bone.rotation_quaternion
                    if abs(float(quat[0]) - 1.0) > float(tolerance) or any(abs(float(quat[index])) > float(tolerance) for index in range(1, 4)):
                        return True
                else:
                    if any(abs(float(value)) > float(tolerance) for value in pose_bone.rotation_euler):
                        return True
            except Exception:
                continue
    return False


def _armature_has_active_nla(armature: bpy.types.Object) -> bool:
    try:
        anim_data = getattr(armature, 'animation_data', None)
        for nla_track in getattr(anim_data, 'nla_tracks', []) or []:
            if not bool(getattr(nla_track, 'mute', False)):
                return True
    except Exception:
        pass
    return False


def _armature_has_pose_constraints_or_drivers(armature: bpy.types.Object) -> bool:
    try:
        for constraint in getattr(armature, 'constraints', []) or []:
            if not bool(getattr(constraint, 'mute', False)):
                return True
    except Exception:
        pass
    try:
        for pose_bone in getattr(getattr(armature, 'pose', None), 'bones', []) or []:
            for constraint in getattr(pose_bone, 'constraints', []) or []:
                if not bool(getattr(constraint, 'mute', False)):
                    return True
    except Exception:
        pass
    try:
        anim_data = getattr(armature, 'animation_data', None)
        if len(getattr(anim_data, 'drivers', []) or []) > 0:
            return True
    except Exception:
        pass
    return False


def _armature_can_use_fast_action_sampling(armature: bpy.types.Object, action) -> bool:
    return (
        armature is not None
        and getattr(armature, 'type', None) == 'ARMATURE'
        and _action_has_fcurves(action)
        and not _armature_has_active_nla(armature)
        and not _armature_has_pose_constraints_or_drivers(armature)
    )


def _should_sample_armature_for_mul_export(armature: bpy.types.Object, action) -> bool:
    if armature is None or getattr(armature, 'type', None) != 'ARMATURE':
        return False
    if _action_has_fcurves(action):
        return True
    if _armature_has_active_nla(armature):
        return True
    return _armature_has_non_rest_pose(armature)


def _sample_pose_basis_from_action(action, pose_bone, frame: int, fcurve_lookup: dict | None = None) -> mathutils.Matrix:
    bone_name = pose_bone.name
    location_path = f'pose.bones["{bone_name}"].location'
    scale_path = f'pose.bones["{bone_name}"].scale'
    quat_path = f'pose.bones["{bone_name}"].rotation_quaternion'
    euler_path = f'pose.bones["{bone_name}"].rotation_euler'

    location = mathutils.Vector((
        _evaluate_action_value(action, location_path, 0, frame, float(pose_bone.location[0]), fcurve_lookup),
        _evaluate_action_value(action, location_path, 1, frame, float(pose_bone.location[1]), fcurve_lookup),
        _evaluate_action_value(action, location_path, 2, frame, float(pose_bone.location[2]), fcurve_lookup),
    ))
    scale = mathutils.Vector((
        _evaluate_action_value(action, scale_path, 0, frame, float(pose_bone.scale[0]), fcurve_lookup),
        _evaluate_action_value(action, scale_path, 1, frame, float(pose_bone.scale[1]), fcurve_lookup),
        _evaluate_action_value(action, scale_path, 2, frame, float(pose_bone.scale[2]), fcurve_lookup),
    ))

    has_quat = any(_find_action_fcurve(action, quat_path, index, fcurve_lookup) is not None for index in range(4))
    if has_quat:
        quat = mathutils.Quaternion((
            _evaluate_action_value(action, quat_path, 0, frame, float(pose_bone.rotation_quaternion[0]), fcurve_lookup),
            _evaluate_action_value(action, quat_path, 1, frame, float(pose_bone.rotation_quaternion[1]), fcurve_lookup),
            _evaluate_action_value(action, quat_path, 2, frame, float(pose_bone.rotation_quaternion[2]), fcurve_lookup),
            _evaluate_action_value(action, quat_path, 3, frame, float(pose_bone.rotation_quaternion[3]), fcurve_lookup),
        ))
        try:
            quat.normalize()
        except Exception:
            quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
        rotation_matrix = quat.to_matrix().to_4x4()
    else:
        euler = mathutils.Euler((
            _evaluate_action_value(action, euler_path, 0, frame, float(pose_bone.rotation_euler[0]), fcurve_lookup),
            _evaluate_action_value(action, euler_path, 1, frame, float(pose_bone.rotation_euler[1]), fcurve_lookup),
            _evaluate_action_value(action, euler_path, 2, frame, float(pose_bone.rotation_euler[2]), fcurve_lookup),
        ), 'XYZ')
        rotation_matrix = euler.to_matrix().to_4x4()

    return (
        mathutils.Matrix.Translation(location)
        @ rotation_matrix
        @ mathutils.Matrix.Diagonal((float(scale[0]), float(scale[1]), float(scale[2]), 1.0))
    )


def _pose_bone_mapping_by_index(armature: bpy.types.Object, num_bones: int) -> dict[int, bpy.types.PoseBone]:
    pose_bones_by_name = {pose_bone.name: pose_bone for pose_bone in armature.pose.bones}
    mapped: dict[int, bpy.types.PoseBone] = {}
    used_names: set[str] = set()
    for bone_index in range(min(int(num_bones), len(armature.pose.bones))):
        pose_bone = pose_bones_by_name.get(f'bone_{bone_index}')
        if pose_bone is None and bone_index < len(armature.pose.bones):
            pose_bone = armature.pose.bones[bone_index]
        if pose_bone is not None and pose_bone.name not in used_names:
            mapped[bone_index] = pose_bone
            used_names.add(pose_bone.name)
    return mapped


def _bone_depth(pose_bone: bpy.types.PoseBone) -> int:
    depth = 0
    parent = pose_bone.parent
    while parent is not None:
        depth += 1
        parent = parent.parent
    return depth


def _mul_channel_conversion_context(default_matrix: mathutils.Matrix, parent_default_matrix: mathutils.Matrix) -> dict:
    parent_default_orientation = MultiplexStreamImporterMixin._mul_orientation_only_matrix(parent_default_matrix)
    default_orientation = MultiplexStreamImporterMixin._mul_orientation_only_matrix(default_matrix)
    local_default_orientation = parent_default_orientation.inverted_safe() @ default_orientation
    return {
        'parent_default_orientation': parent_default_orientation,
        'parent_default_orientation_inv': parent_default_orientation.inverted_safe(),
        'parent_default_orientation_3x3': parent_default_orientation.to_3x3(),
        'local_default_orientation_inv': local_default_orientation.inverted_safe(),
    }


def _mul_source_local_to_channel_values_with_context(
    source_local_matrix: mathutils.Matrix,
    conversion_context: dict,
    reference_rotation: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    parent_default_orientation = conversion_context.get('parent_default_orientation', mathutils.Matrix.Identity(4))
    parent_default_orientation_inv = conversion_context.get('parent_default_orientation_inv', mathutils.Matrix.Identity(4))
    parent_default_orientation_3x3 = conversion_context.get('parent_default_orientation_3x3')
    local_default_orientation_inv = conversion_context.get('local_default_orientation_inv', mathutils.Matrix.Identity(4))
    if parent_default_orientation_3x3 is None:
        parent_default_orientation_3x3 = parent_default_orientation.to_3x3()

    local_location, local_rotation, local_scale = source_local_matrix.decompose()
    source_rotation = local_rotation.to_matrix().to_4x4() @ local_default_orientation_inv
    channel_rotation_matrix = parent_default_orientation @ source_rotation @ parent_default_orientation_inv
    try:
        if reference_rotation is not None:
            euler_compat = mathutils.Euler(tuple(float(reference_rotation[axis]) for axis in range(3)), 'XYZ')
            channel_euler = channel_rotation_matrix.to_3x3().to_euler('XYZ', euler_compat)
        else:
            channel_euler = channel_rotation_matrix.to_3x3().to_euler('XYZ')
    except Exception:
        channel_euler = mathutils.Euler((0.0, 0.0, 0.0), 'XYZ')

    raw_location = parent_default_orientation_3x3 @ local_location
    scale = (float(local_scale[0]), float(local_scale[1]), float(local_scale[2]))
    rotation = (float(channel_euler.x), float(channel_euler.y), float(channel_euler.z))
    if reference_rotation is not None:
        rotation = tuple(_wrap_angle_near_reference(rotation[axis], float(reference_rotation[axis])) for axis in range(3))
    location = (float(raw_location[0]), float(raw_location[1]), float(raw_location[2]))
    return scale, rotation, location


def _mul_source_local_to_channel_values(
    source_local_matrix: mathutils.Matrix,
    default_matrix: mathutils.Matrix,
    parent_default_matrix: mathutils.Matrix,
    reference_rotation: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return _mul_source_local_to_channel_values_with_context(
        source_local_matrix,
        _mul_channel_conversion_context(default_matrix, parent_default_matrix),
        reference_rotation,
    )


def _reference_rotation_for_bone_frame(reference_bone_frames, bone_index: int, frame_index: int, frame: int, fallback=None):
    try:
        bone_frames = reference_bone_frames[int(bone_index)] if reference_bone_frames is not None and int(bone_index) < len(reference_bone_frames) else []
    except Exception:
        bone_frames = []

    candidates = []
    try:
        if 0 <= int(frame_index) < len(bone_frames):
            candidates.append(bone_frames[int(frame_index)])
    except Exception:
        pass
    try:
        for item in bone_frames:
            if int(item.get('frame', -999999)) == int(frame):
                candidates.append(item)
                break
    except Exception:
        pass

    for item in candidates:
        try:
            rotation = item.get('rotation')
            if rotation is not None and len(rotation) >= 3:
                return tuple(float(rotation[axis]) for axis in range(3))
        except Exception:
            continue
    if fallback is not None:
        try:
            return tuple(float(fallback[axis]) for axis in range(3))
        except Exception:
            pass
    return None


def _default_transform_for_bone(default_transforms, bone_index: int, pose_bone=None) -> mathutils.Matrix:
    if default_transforms is not None and int(bone_index) < len(default_transforms):
        return _matrix_from_stored_default(default_transforms[int(bone_index)])
    if pose_bone is not None:
        try:
            return pose_bone.bone.matrix_local.copy()
        except Exception:
            pass
    return mathutils.Matrix.Identity(4)


def _mul_export_bone_specs(armature: bpy.types.Object, default_transforms, num_bones: int):
    mapped = _pose_bone_mapping_by_index(armature, int(num_bones))
    if len(mapped) < int(num_bones):
        raise ValueError(f'Armature {armature.name} has {len(mapped)} mapped bones, MUL skeleton needs {num_bones}')

    pose_bone_index_by_name = {pose_bone.name: bone_index for bone_index, pose_bone in mapped.items()}
    bone_order = sorted(mapped.keys(), key=lambda index: _bone_depth(mapped[index]))
    specs: list[dict] = []
    for bone_index in bone_order:
        pose_bone = mapped[bone_index]
        if pose_bone.parent is None:
            rest_local_matrix = pose_bone.bone.matrix_local.copy()
            parent_index = None
            parent_default_matrix = mathutils.Matrix.Identity(4)
        else:
            parent_rest_matrix = pose_bone.parent.bone.matrix_local.copy()
            rest_local_matrix = parent_rest_matrix.inverted_safe() @ pose_bone.bone.matrix_local.copy()
            parent_index = pose_bone_index_by_name.get(pose_bone.parent.name)
            parent_default_matrix = _default_transform_for_bone(default_transforms, parent_index, pose_bone.parent) if parent_index is not None else mathutils.Matrix.Identity(4)
        default_matrix = _default_transform_for_bone(default_transforms, bone_index, pose_bone)
        specs.append({
            'bone_index': int(bone_index),
            'pose_bone': pose_bone,
            'rest_local_matrix': rest_local_matrix,
            'conversion_context': _mul_channel_conversion_context(default_matrix, parent_default_matrix),
        })
    return mapped, bone_order, specs


def _sample_mul_skeleton_from_armature(action, armature: bpy.types.Object, default_transforms, frames: list[int], num_bones: int, reference_bone_frames=None) -> list[list[dict]]:
    _mapped, _bone_order, bone_specs = _mul_export_bone_specs(armature, default_transforms, int(num_bones))
    output = [[] for _ in range(int(num_bones))]
    previous_rotations: list[tuple[float, float, float] | None] = [None] * int(num_bones)
    fcurve_lookup = _build_action_fcurve_lookup(action)

    for frame_index, frame in enumerate(frames):
        for spec in bone_specs:
            bone_index = int(spec['bone_index'])
            pose_bone = spec['pose_bone']
            pose_basis = _sample_pose_basis_from_action(action, pose_bone, int(frame), fcurve_lookup)
            source_local_matrix = spec['rest_local_matrix'] @ pose_basis

            reference_rotation = _reference_rotation_for_bone_frame(reference_bone_frames, bone_index, frame_index, frame, previous_rotations[bone_index])
            scale, rotation, location = _mul_source_local_to_channel_values_with_context(source_local_matrix, spec['conversion_context'], reference_rotation)
            previous_rotations[bone_index] = rotation
            frame_data = {
                'frame': int(frame),
                'scale': scale,
                'rotation': rotation,
                'location': location,
            }
            flags = _evaluate_pose_bone_flags(action, pose_bone, int(frame), fcurve_lookup)
            if flags is not None:
                frame_data['flags'] = flags
            output[bone_index].append(frame_data)
    return output


def _source_local_matrix_from_evaluated_pose(eval_pose_bone, parent_inverse_cache: dict | None = None) -> mathutils.Matrix:
    try:
        pose_matrix = eval_pose_bone.matrix.copy()
        parent = getattr(eval_pose_bone, 'parent', None)
        if parent is not None:
            cache_key = getattr(parent, 'name', None) or id(parent)
            parent_inverse = None
            if parent_inverse_cache is not None:
                parent_inverse = parent_inverse_cache.get(cache_key)
            if parent_inverse is None:
                parent_inverse = parent.matrix.inverted_safe()
                if parent_inverse_cache is not None:
                    parent_inverse_cache[cache_key] = parent_inverse
            return parent_inverse @ pose_matrix
        return pose_matrix
    except Exception:
        return mathutils.Matrix.Identity(4)


def _sample_mul_skeleton_from_evaluated_armature(context, action, armature: bpy.types.Object, default_transforms, frames: list[int], num_bones: int, reference_bone_frames=None) -> list[list[dict]]:
    scene = getattr(context, 'scene', None) if context is not None else None
    if scene is None:
        return _sample_mul_skeleton_from_armature(action, armature, default_transforms, frames, num_bones, reference_bone_frames)

    _mapped, _bone_order, bone_specs = _mul_export_bone_specs(armature, default_transforms, int(num_bones))
    output = [[] for _ in range(int(num_bones))]
    previous_rotations: list[tuple[float, float, float] | None] = [None] * int(num_bones)
    fcurve_lookup = _build_action_fcurve_lookup(action)

    try:
        original_frame = int(scene.frame_current)
    except Exception:
        original_frame = None

    try:
        depsgraph = context.evaluated_depsgraph_get()
    except Exception:
        depsgraph = None

    try:
        for frame_index, frame in enumerate(frames):
            try:
                scene.frame_set(int(frame))
            except Exception:
                pass
            try:
                if depsgraph is not None:
                    eval_armature = armature.evaluated_get(depsgraph)
                else:
                    eval_armature = armature
            except Exception:
                eval_armature = armature

            try:
                eval_pose_bones = eval_armature.pose.bones
            except Exception:
                eval_pose_bones = None
            parent_inverse_cache: dict = {}

            for spec in bone_specs:
                bone_index = int(spec['bone_index'])
                pose_bone = spec['pose_bone']
                eval_pose_bone = pose_bone
                if eval_pose_bones is not None:
                    try:
                        eval_pose_bone = eval_pose_bones.get(pose_bone.name) or pose_bone
                    except Exception:
                        eval_pose_bone = pose_bone
                source_local_matrix = _source_local_matrix_from_evaluated_pose(eval_pose_bone, parent_inverse_cache)

                reference_rotation = _reference_rotation_for_bone_frame(reference_bone_frames, bone_index, frame_index, frame, previous_rotations[bone_index])
                scale, rotation, location = _mul_source_local_to_channel_values_with_context(source_local_matrix, spec['conversion_context'], reference_rotation)
                previous_rotations[bone_index] = rotation
                frame_data = {
                    'frame': int(frame),
                    'scale': scale,
                    'rotation': rotation,
                    'location': location,
                }
                flags = _evaluate_pose_bone_flags(action, pose_bone, int(frame), fcurve_lookup)
                if flags is not None:
                    frame_data['flags'] = flags
                output[bone_index].append(frame_data)
    finally:
        if original_frame is not None:
            try:
                scene.frame_set(int(original_frame))
            except Exception:
                pass

    return output


def _skeleton_frames_have_transform_variation(bone_frames: list[list[dict]], tolerance: float = 1.0e-5) -> bool:
    for frames in bone_frames or []:
        if len(frames) < 2:
            continue
        first = frames[0]
        for frame in frames[1:]:
            for kind in ('scale', 'rotation', 'location'):
                first_values = first.get(kind, ())
                values = frame.get(kind, ())
                for axis in range(min(len(first_values), len(values), 3)):
                    try:
                        if abs(float(values[axis]) - float(first_values[axis])) > float(tolerance):
                            return True
                    except Exception:
                        pass
    return False


def _wrap_angle_near_reference(value: float, reference: float) -> float:
    try:
        value = float(value)
        reference = float(reference)
        if not math.isfinite(value) or not math.isfinite(reference):
            return value
        period = math.tau
        return value + (round((reference - value) / period) * period)
    except Exception:
        return float(value)


def _merge_scene_bone_frames_with_raw(scene_bone_frames: list[list[dict]], raw_bone_frames: list[list[dict]], export_scene_locations: bool = False) -> list[list[dict]]:

    def _finite_float(value, fallback: float = 0.0) -> float:
        try:
            result = float(value)
            if math.isfinite(result):
                return result
        except Exception:
            pass
        return float(fallback)

    def _component_range(values: list[float]) -> float:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return 0.0
        return max(finite) - min(finite)

    static_location_epsilon = 5.0e-2

    merged: list[list[dict]] = []
    bone_count = max(len(scene_bone_frames or []), len(raw_bone_frames or []))
    for bone_index in range(bone_count):
        scene_frames = scene_bone_frames[bone_index] if bone_index < len(scene_bone_frames or []) else []
        raw_frames = raw_bone_frames[bone_index] if bone_index < len(raw_bone_frames or []) else []
        frame_count = max(len(scene_frames), len(raw_frames))

        raw_locations: list[tuple[float, float, float]] = []
        scene_locations: list[tuple[float, float, float]] = []
        for frame_index in range(frame_count):
            scene_info = scene_frames[frame_index] if frame_index < len(scene_frames) else {}
            raw_info = raw_frames[frame_index] if frame_index < len(raw_frames) else {}
            raw_location = tuple(_finite_float(v) for v in raw_info.get('location', scene_info.get('location', (0.0, 0.0, 0.0))))
            scene_location = tuple(_finite_float(v, raw_location[i] if i < len(raw_location) else 0.0) for i, v in enumerate(scene_info.get('location', raw_location)))
            raw_locations.append(raw_location[:3])
            scene_locations.append(scene_location[:3])

        exported_locations: list[tuple[float, float, float]] = list(raw_locations)
        if bool(export_scene_locations) and frame_count > 0:
            per_axis_values: list[list[float]] = []
            for axis in range(3):
                raw_axis = [location[axis] for location in raw_locations]
                scene_axis = [location[axis] for location in scene_locations]
                raw_range = _component_range(raw_axis)
                scene_range = _component_range(scene_axis)
                if raw_range <= static_location_epsilon and scene_range <= static_location_epsilon:
                    # Retargeted rest offset: write it once as a static run.
                    per_axis_values.append([scene_axis[0]] * frame_count)
                else:
                    # Real animated translation: keep scene-sampled values.
                    per_axis_values.append(scene_axis)
            exported_locations = [
                (per_axis_values[0][frame_index], per_axis_values[1][frame_index], per_axis_values[2][frame_index])
                for frame_index in range(frame_count)
            ]

        raw_scales: list[tuple[float, float, float]] = []
        scene_scales: list[tuple[float, float, float]] = []
        scene_scale_available = False
        for frame_index in range(frame_count):
            scene_info = scene_frames[frame_index] if frame_index < len(scene_frames) else {}
            raw_info = raw_frames[frame_index] if frame_index < len(raw_frames) else {}
            raw_scale = tuple(_finite_float(v, 1.0) for v in raw_info.get('scale', scene_info.get('scale', (1.0, 1.0, 1.0))))
            if 'scale' in scene_info:
                scene_scale_available = True
            scene_scale = tuple(_finite_float(v, raw_scale[i] if i < len(raw_scale) else 1.0) for i, v in enumerate(scene_info.get('scale', raw_scale)))
            raw_scales.append(raw_scale[:3])
            scene_scales.append(scene_scale[:3])

        exported_scales: list[tuple[float, float, float]] = list(raw_scales)
        if scene_scale_available and frame_count > 0:
            per_axis_values: list[list[float]] = []
            static_scale_epsilon = 1.0e-5
            for axis in range(3):
                scene_axis = [scale[axis] for scale in scene_scales]
                scene_range = _component_range(scene_axis)
                if scene_range <= static_scale_epsilon:
                    per_axis_values.append([scene_axis[0]] * frame_count)
                else:
                    per_axis_values.append(scene_axis)
            exported_scales = [
                (per_axis_values[0][frame_index], per_axis_values[1][frame_index], per_axis_values[2][frame_index])
                for frame_index in range(frame_count)
            ]

        merged_frames: list[dict] = []
        for frame_index in range(frame_count):
            scene_info = scene_frames[frame_index] if frame_index < len(scene_frames) else {}
            raw_info = raw_frames[frame_index] if frame_index < len(raw_frames) else {}
            raw_rotation = tuple(_finite_float(v) for v in raw_info.get('rotation', (0.0, 0.0, 0.0)))
            scene_rotation = tuple(_finite_float(v, raw_rotation[i] if i < len(raw_rotation) else 0.0) for i, v in enumerate(scene_info.get('rotation', raw_rotation)))
            rotation = tuple(_wrap_angle_near_reference(scene_rotation[axis], raw_rotation[axis]) for axis in range(3))
            merged_frames.append({
                'frame': int(raw_info.get('frame', scene_info.get('frame', frame_index))),
                'scale': exported_scales[frame_index] if frame_index < len(exported_scales) else (1.0, 1.0, 1.0),
                'rotation': rotation,
                'location': exported_locations[frame_index] if frame_index < len(exported_locations) else (0.0, 0.0, 0.0),
                'flags': _finite_float(scene_info.get('flags', raw_info.get('flags', MUL_EXPORT_DEFAULT_SKELETON_FLAGS)), MUL_EXPORT_DEFAULT_SKELETON_FLAGS),
            })
        merged.append(merged_frames)
    return merged


def _raw_stored_frames_by_frame(stored_bone_frames: list[list[dict]], frames: list[int], num_bones: int) -> list[list[dict]]:
    output = [[] for _ in range(int(num_bones))]
    for bone_index in range(int(num_bones)):
        source_frames = list(stored_bone_frames[bone_index]) if bone_index < len(stored_bone_frames) else []
        lookup = {int(item.get('frame', 0)): item for item in source_frames}
        last = None
        sorted_keys = sorted(lookup.keys())
        key_cursor = 0
        for frame in frames:
            while key_cursor < len(sorted_keys) and sorted_keys[key_cursor] <= int(frame):
                last = lookup[sorted_keys[key_cursor]]
                key_cursor += 1
            item = lookup.get(int(frame), last)
            if item is None:
                item = {'frame': int(frame), 'scale': (1.0, 1.0, 1.0), 'rotation': (0.0, 0.0, 0.0), 'location': (0.0, 0.0, 0.0)}
            output[bone_index].append({
                'frame': int(frame),
                'scale': tuple(float(v) for v in item.get('scale', (1.0, 1.0, 1.0))),
                'rotation': tuple(float(v) for v in item.get('rotation', (0.0, 0.0, 0.0))),
                'location': tuple(float(v) for v in item.get('location', (0.0, 0.0, 0.0))),
                'flags': float(item.get('flags', MUL_EXPORT_DEFAULT_SKELETON_FLAGS)),
            })
    return output



def _raw_stored_root_values_by_frame(root_frames: list, frames: list[int]) -> list[dict]:
    lookup = {}
    for item in root_frames or []:
        if isinstance(item, dict):
            frame = int(item.get('frame', 0))
            lookup[frame] = {
                'frame': frame,
                'channel_0': float(item.get('channel_0', MUL_EXPORT_ROOT_CHANNEL_0)),
                'root_matrix_index': float(item.get('root_matrix_index', MUL_EXPORT_ROOT_MATRIX_INDEX)),
            }
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            frame = int(item[0])
            lookup[frame] = {
                'frame': frame,
                'channel_0': float(item[1]),
                'root_matrix_index': float(item[2]),
            }
    output = []
    last = None
    sorted_keys = sorted(lookup.keys())
    key_cursor = 0
    for frame in frames:
        while key_cursor < len(sorted_keys) and sorted_keys[key_cursor] <= int(frame):
            last = lookup[sorted_keys[key_cursor]]
            key_cursor += 1
        item = lookup.get(int(frame), last)
        if item is None:
            item = {
                'frame': int(frame),
                'channel_0': MUL_EXPORT_ROOT_CHANNEL_0,
                'root_matrix_index': MUL_EXPORT_ROOT_MATRIX_INDEX,
            }
        output.append({
            'frame': int(frame),
            'channel_0': float(item.get('channel_0', MUL_EXPORT_ROOT_CHANNEL_0)),
            'root_matrix_index': float(item.get('root_matrix_index', MUL_EXPORT_ROOT_MATRIX_INDEX)),
        })
    return output


def _raw_stored_camera_frames_by_frame(stored_camera_frames: list[dict], frames: list[int]) -> list[dict]:
    lookup = {int(item.get('frame', 0)): item for item in stored_camera_frames or [] if isinstance(item, dict)}
    last = None
    sorted_keys = sorted(lookup.keys())
    key_cursor = 0
    output = []
    for frame in frames:
        while key_cursor < len(sorted_keys) and sorted_keys[key_cursor] <= int(frame):
            last = lookup[sorted_keys[key_cursor]]
            key_cursor += 1
        item = lookup.get(int(frame), last)
        if item is None:
            item = {
                'frame': int(frame),
                'scale': (1.0, 1.0, 1.0),
                'rotation': (0.0, 0.0, 0.0),
                'location': (0.0, 0.0, 0.0),
                'fov': math.radians(50.0),
                'flags': 0,
            }
        output.append({
            'frame': int(frame),
            'scale': tuple(float(v) for v in item.get('scale', (1.0, 1.0, 1.0))),
            'rotation': tuple(float(v) for v in item.get('rotation', (0.0, 0.0, 0.0))),
            'location': tuple(float(v) for v in item.get('location', (0.0, 0.0, 0.0))),
            'fov': float(item.get('fov', math.radians(50.0))),
            'flags': int(item.get('flags', 0)),
        })
    return output


def _expand_payload_camera_frames(camera_entry: dict) -> list[dict]:
    expanded = []
    for frame_info in list(camera_entry.get('frames', []) or []):
        if isinstance(frame_info, dict):
            expanded.append({
                'frame': int(frame_info.get('frame', 0)),
                'scale': tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0))),
                'rotation': tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0))),
                'location': tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0))),
                'fov': float(frame_info.get('fov', math.radians(50.0))),
                'flags': int(frame_info.get('flags', 0)),
            })
            continue
        if not isinstance(frame_info, (list, tuple)) or len(frame_info) < 6:
            continue
        frame, scale, rotation, location, fov, flags = frame_info[:6]
        expanded.append({
            'frame': int(frame),
            'scale': tuple(float(v) for v in scale),
            'rotation': tuple(float(v) for v in rotation),
            'location': tuple(float(v) for v in location),
            'fov': float(fov),
            'flags': int(flags),
        })
    return expanded

def _camera_fov_from_lens(camera_data) -> float:
    if camera_data is None:
        return math.radians(50.0)
    try:
        if str(getattr(camera_data, 'sensor_fit', 'AUTO')) == 'VERTICAL':
            sensor_size = float(getattr(camera_data, 'sensor_height', 24.0) or 24.0)
        else:
            sensor_size = float(getattr(camera_data, 'sensor_width', 36.0) or 36.0)
        lens = float(getattr(camera_data, 'lens', 50.0) or 50.0)
        if lens <= 0.0:
            return math.radians(50.0)
        return 2.0 * math.atan(sensor_size / (2.0 * lens))
    except Exception:
        return math.radians(50.0)


def _sample_camera_frames(camera_obj: bpy.types.Object | None, frames: list[int]) -> list[dict]:
    if camera_obj is None:
        return [
            {
                'frame': int(frame),
                'scale': (1.0, 1.0, 1.0),
                'rotation': (0.0, 0.0, 0.0),
                'location': (0.0, 0.0, 0.0),
                'fov': math.radians(50.0),
                'flags': 0,
            }
            for frame in frames
        ]

    object_action = getattr(getattr(camera_obj, 'animation_data', None), 'action', None)
    camera_data = getattr(camera_obj, 'data', None)
    camera_action = getattr(getattr(camera_data, 'animation_data', None), 'action', None) if camera_data is not None else None
    camera_obj.rotation_mode = 'XYZ'

    result = []
    for frame in frames:
        location = tuple(_evaluate_action_value(object_action, 'location', axis, frame, float(camera_obj.location[axis])) for axis in range(3))
        rotation = tuple(_evaluate_action_value(object_action, 'rotation_euler', axis, frame, float(camera_obj.rotation_euler[axis])) for axis in range(3))
        raw_scale = tuple(_evaluate_action_value(object_action, 'scale', axis, frame, float(camera_obj.scale[axis])) / float(MUL_EXPORT_CAMERA_VISUAL_SCALE) for axis in range(3))
        flags = int(round(_evaluate_action_value(object_action, '["trlau_mul_camera_flags"]', 0, frame, float(camera_obj.get('trlau_mul_camera_flags', 0) or 0))))
        lens_default = float(getattr(camera_data, 'lens', 50.0) or 50.0) if camera_data is not None else 50.0
        lens = _evaluate_action_value(camera_action, 'lens', 0, frame, lens_default)
        if camera_data is not None:
            original_lens = float(getattr(camera_data, 'lens', 50.0) or 50.0)
            try:
                camera_data.lens = float(lens)
                fov = _camera_fov_from_lens(camera_data)
            finally:
                try:
                    camera_data.lens = original_lens
                except Exception:
                    pass
        else:
            fov = math.radians(50.0)
        result.append({
            'frame': int(frame),
            'scale': tuple(float(v) for v in raw_scale),
            'rotation': tuple(float(v) for v in rotation),
            'location': tuple(float(v) for v in location),
            'fov': float(fov),
            'flags': int(flags),
        })
    return result


def _sample_camera_frames_from_scene(context, camera_obj: bpy.types.Object | None, frames: list[int]) -> list[dict]:
    if camera_obj is None:
        return _sample_camera_frames(None, frames)

    scene = getattr(context, 'scene', None) if context is not None else None
    if scene is None:
        return _sample_camera_frames(camera_obj, frames)

    try:
        original_frame = int(scene.frame_current)
    except Exception:
        original_frame = None

    camera_obj.rotation_mode = 'XYZ'
    result: list[dict] = []
    try:
        for frame in frames:
            scene.frame_set(int(frame))
            try:
                depsgraph = context.evaluated_depsgraph_get()
                eval_obj = camera_obj.evaluated_get(depsgraph)
            except Exception:
                eval_obj = camera_obj

            try:
                location = tuple(float(v) for v in eval_obj.location[:3])
            except Exception:
                location = tuple(float(v) for v in camera_obj.location[:3])

            try:
                rotation = tuple(float(v) for v in eval_obj.rotation_euler[:3])
            except Exception:
                rotation = tuple(float(v) for v in camera_obj.rotation_euler[:3])

            try:
                eval_scale = tuple(float(v) for v in eval_obj.scale[:3])
            except Exception:
                eval_scale = tuple(float(v) for v in camera_obj.scale[:3])
            raw_scale = tuple(float(v) / float(MUL_EXPORT_CAMERA_VISUAL_SCALE) for v in eval_scale)

            try:
                flags = int(round(float(camera_obj.get('trlau_mul_camera_flags', 0) or 0)))
            except Exception:
                flags = 0

            eval_data = getattr(eval_obj, 'data', None) or getattr(camera_obj, 'data', None)
            fov = _camera_fov_from_lens(eval_data)

            result.append({
                'frame': int(frame),
                'scale': tuple(float(v) for v in raw_scale),
                'rotation': tuple(float(v) for v in rotation),
                'location': tuple(float(v) for v in location),
                'fov': float(fov),
                'flags': int(flags),
            })
    finally:
        if original_frame is not None:
            try:
                scene.frame_set(int(original_frame))
            except Exception:
                pass
    return result

def _camera_index_from_name(name: str, base_candidates) -> int | None:
    import re

    clean_name = str(name or '').strip()
    if not clean_name:
        return None
    lower_name = clean_name.lower()
    # Remove Blender duplicate suffix only after the numeric camera index.
    lower_name = re.sub(r'(\d+)\.\d+$', r'\1', lower_name)

    for candidate in base_candidates:
        base = str(candidate or '').strip().lower()
        if not base:
            continue
        base = re.escape(base)
        match = re.match(rf'^{base}_mul_cam_(\d+)$', lower_name)
        if match:
            return int(match.group(1))
    return None


def _find_multiplex_camera_objects(multiplex_obj, metadata: dict) -> dict[int, bpy.types.Object]:
    result: dict[int, bpy.types.Object] = {}
    if multiplex_obj is None:
        return result

    mul_name = str(metadata.get('name', multiplex_obj.get('trlau_mul_name', '')) or '')
    source_path = str(metadata.get('source', multiplex_obj.get('trlau_mul_source', '')) or '')
    source_stem = Path(source_path).stem if source_path else ''
    object_name = str(multiplex_obj.name or '')
    object_stem = object_name[:-10] if object_name.endswith('_Multiplex') else object_name
    collection_names = []
    try:
        collection_names = [str(collection.name or '') for collection in getattr(multiplex_obj, 'users_collection', [])]
    except Exception:
        collection_names = []

    base_candidates = []
    for candidate in (source_stem, mul_name, object_stem, object_name, *collection_names):
        candidate = str(candidate or '').strip()
        if candidate and candidate not in base_candidates:
            base_candidates.append(candidate)

    for obj in bpy.data.objects:
        if getattr(obj, 'type', None) != 'CAMERA':
            continue
        camera_index = _camera_index_from_name(getattr(obj, 'name', ''), base_candidates)
        if camera_index is None:
            camera_data = getattr(obj, 'data', None)
            camera_index = _camera_index_from_name(getattr(camera_data, 'name', ''), base_candidates)
        if camera_index is not None:
            result[int(camera_index)] = obj

    return result

def _write_c_string_fixed(text: str, length: int) -> bytes:
    raw = str(text or '').encode('ascii', errors='ignore')[:max(0, int(length) - 1)]
    return raw + (b'\x00' * (int(length) - len(raw)))


def _pack_stream_header(final_frame: int, audio_info: dict | None = None, fps: float = 30.0, has_subtitles: bool = False) -> bytes:
    audio_info = audio_info or {}
    sample_rate = int(audio_info.get('sample_rate', 44100) or 44100)
    sample_count = int(audio_info.get('sample_count', 0) or 0)
    channel_count = int(audio_info.get('channel_count', 0) or 0)
    if channel_count <= 0 or sample_count <= 0:
        channel_count = 0
        sample_count = 0

    try:
        fps_value = float(fps) if float(fps) > 0.0 else 30.0
    except Exception:
        fps_value = 30.0
    audio_media_frame = int(round((float(sample_count) / float(sample_rate)) * fps_value)) if sample_rate > 0 and sample_count > 0 else 0
    media_length = float(max(0, int(final_frame), int(audio_media_frame)))

    header = bytearray()
    header.extend(struct.pack(
        '<10i',
        int(sample_rate),
        -1,
        int(sample_count),
        int(channel_count),
        0,
        0,
        0,
        0,
        1,
        1 if bool(has_subtitles) else 0,
    ))
    header.extend(struct.pack('<iiif', 0, 0, 65536, media_length))

    left_volumes = [0.0] * 12
    right_volumes = [0.0] * 12
    if channel_count == 1:
        left_volumes[0] = 1.0
        right_volumes[0] = 1.0
    elif channel_count >= 2:
        left_volumes[0] = 1.0
        right_volumes[1] = 1.0

    header.extend(struct.pack('<12f', *left_volumes))
    header.extend(struct.pack('<12f', *right_volumes))
    header.extend(struct.pack('<12I', *([0] * 12)))
    return bytes(header)


def _pack_cine_header(name: str, main_unit_id: int, anchors: list[dict], skeleton_headers: list[dict], cameras: list[dict], skeleton_frame_data: list[list[list[dict]]], final_frame: int, num_subtitles: int = 0) -> bytes:
    header = bytearray()
    header.extend(b'ENIC')
    header.extend(struct.pack('<i', MUL_EXPORT_DEFAULT_VERSION))
    header.extend(struct.pack('<i', 0))
    header.extend(_write_c_string_fixed(name, 0x40))
    header.extend(struct.pack('<i', int(main_unit_id)))

    header.extend(struct.pack('<i', len(anchors)))
    current_channel = 0
    anchor_layouts = []
    for anchor in anchors:
        anchor_layouts.append(current_channel)
        header.extend(struct.pack('<iii', int(anchor.get('area_id', -1)), int(anchor.get('marker_id', -1)), int(current_channel)))
        current_channel += 9

    header.extend(struct.pack('<i', len(skeleton_headers)))
    for skeleton_index, skeleton in enumerate(skeleton_headers):
        num_bones = int(skeleton.get('num_bones', 0) or 0)
        skeleton['first_channel'] = int(current_channel)
        header.extend(struct.pack('<iii', int(skeleton.get('instance_id', -1)), num_bones, int(current_channel)))
        default_transforms = list(skeleton.get('default_bone_transforms', []) or [])
        for bone_index in range(num_bones):
            if bone_index < len(default_transforms) and default_transforms[bone_index] is not None:
                values = list(default_transforms[bone_index])
            else:
                values = _flatten_matrix_rows(mathutils.Matrix.Identity(4))
            if len(values) != 16:
                values = _flatten_matrix_rows(mathutils.Matrix.Identity(4))
            header.extend(struct.pack('<16f', *(float(v) for v in values[:16])))
        current_channel += 2 + (num_bones * 10)

    header.extend(struct.pack('<i', len(cameras)))
    for camera in cameras:
        camera['first_channel'] = int(current_channel)
        header.extend(struct.pack('<i', int(current_channel)))
        current_channel += 11

    header.extend(struct.pack('<ii', int(main_unit_id), 0))  # triggerUnitId, numTriggers
    header.extend(struct.pack('<i', max(0, int(num_subtitles))))  # numSubtitles

    for skeleton_index, skeleton in enumerate(skeleton_headers):
        frames_by_bone = skeleton_frame_data[skeleton_index] if skeleton_index < len(skeleton_frame_data) else []
        root_frames = frames_by_bone[0] if frames_by_bone else []
        final_info = root_frames[-1] if root_frames else {}
        end_position = tuple(float(v) for v in final_info.get('location', (0.0, 0.0, 0.0)))
        end_rotation = tuple(float(v) for v in final_info.get('rotation', (0.0, 0.0, 0.0)))
        header.extend(struct.pack('<3f3fi', end_position[0], end_position[1], end_position[2], end_rotation[0], end_rotation[1], end_rotation[2], int(main_unit_id)))

    header_size = len(header) - 8
    struct.pack_into('<i', header, 8, int(header_size))
    return bytes(header)


def _build_channel_value_rows(frames: list[int], anchors: list[dict], skeleton_headers: list[dict], skeleton_frame_data: list[list[list[dict]]], camera_layouts: list[dict], camera_frame_data: list[list[dict]], channel_count: int) -> list[list[float]]:
    rows = []
    frame_count = len(frames)
    for frame_index, frame in enumerate(frames):
        values = [0.0] * int(channel_count)

        # Anchor channels are exported as a static identity transform.
        for anchor in anchors:
            first = int(anchor.get('first_channel', 0) or 0)
            if first + 8 < len(values):
                values[first + 0] = 1.0
                values[first + 1] = 1.0
                values[first + 2] = 1.0
                values[first + 3] = 0.0
                values[first + 4] = 0.0
                values[first + 5] = 0.0
                values[first + 6] = 0.0
                values[first + 7] = 0.0
                values[first + 8] = 0.0

        for skeleton_index, skeleton in enumerate(skeleton_headers):
            first = int(skeleton.get('first_channel', 0) or 0)
            num_bones = int(skeleton.get('num_bones', 0) or 0)
            frames_by_bone = skeleton_frame_data[skeleton_index] if skeleton_index < len(skeleton_frame_data) else []
            root_values = list(skeleton.get('root_values', []) or [])
            root_info = root_values[frame_index] if frame_index < len(root_values) else {}
            if first + 1 < len(values):
                values[first + 0] = float(root_info.get('channel_0', MUL_EXPORT_ROOT_CHANNEL_0))
                values[first + 1] = float(root_info.get('root_matrix_index', MUL_EXPORT_ROOT_MATRIX_INDEX))
            for bone_index in range(num_bones):
                channel_base = first + 2 + (bone_index * 10)
                if channel_base + 9 >= len(values):
                    continue
                bone_frames = frames_by_bone[bone_index] if bone_index < len(frames_by_bone) else []
                frame_info = bone_frames[frame_index] if frame_index < len(bone_frames) else {}
                scale = tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0)))
                rotation = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
                location = tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0)))
                values[channel_base + 0] = scale[0]
                values[channel_base + 1] = scale[1]
                values[channel_base + 2] = scale[2]
                values[channel_base + 3] = rotation[0]
                values[channel_base + 4] = rotation[1]
                values[channel_base + 5] = rotation[2]
                values[channel_base + 6] = location[0]
                values[channel_base + 7] = location[1]
                values[channel_base + 8] = location[2]
                values[channel_base + 9] = float(frame_info.get('flags', MUL_EXPORT_DEFAULT_SKELETON_FLAGS))

        for camera_index, camera in enumerate(camera_layouts):
            first = int(camera.get('first_channel', 0) or 0)
            if first + 10 >= len(values):
                continue
            frames_for_camera = camera_frame_data[camera_index] if camera_index < len(camera_frame_data) else []
            frame_info = frames_for_camera[frame_index] if frame_index < len(frames_for_camera) else {}
            scale = tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0)))
            rotation = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
            location = tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0)))
            values[first + 0] = scale[0]
            values[first + 1] = scale[1]
            values[first + 2] = scale[2]
            values[first + 3] = rotation[0]
            values[first + 4] = rotation[1]
            values[first + 5] = rotation[2]
            values[first + 6] = location[0]
            values[first + 7] = location[1]
            values[first + 8] = location[2]
            values[first + 9] = float(frame_info.get('fov', math.radians(50.0)))
            values[first + 10] = float(int(frame_info.get('flags', 0)))

        rows.append(values)
    return rows



def _float_values_equal(left: float, right: float, epsilon: float = 1.0e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= float(epsilon)
    except Exception:
        return False


def _build_runs_for_channel(values: list[float]) -> list[tuple[int, int]]:
    count = len(values)
    if count <= 0:
        return []

    runs: list[tuple[int, int]] = []
    index = 0
    while index < count:
        # Prefer static spans whenever the current value persists for at least
        # two frames. A one-frame run at EOF is also static.
        static_end = index + 1
        while static_end < count and _float_values_equal(values[static_end], values[index]):
            static_end += 1
        static_length = static_end - index
        if static_length >= 2 or index == count - 1:
            runs.append((int(static_length), 0))
            index = static_end
            continue

        animated_end = index + 1
        while animated_end < count:
            next_static_end = animated_end + 1
            while next_static_end < count and _float_values_equal(values[next_static_end], values[animated_end]):
                next_static_end += 1
            if next_static_end - animated_end >= 2:
                break
            animated_end += 1
        animated_length = max(1, animated_end - index)
        runs.append((int(animated_length), 1))
        index += animated_length

    return runs


def _build_cine_channel_run_plans(channel_rows: list[list[float]], channel_count: int) -> list[list[tuple[int, int]]]:
    frame_count = len(channel_rows)
    plans: list[list[tuple[int, int]]] = []
    for channel_index in range(int(channel_count)):
        values = []
        for frame_index in range(frame_count):
            try:
                values.append(float(channel_rows[frame_index][channel_index]))
            except Exception:
                values.append(0.0)
        plans.append(_build_runs_for_channel(values))
    return plans


def _new_cine_run_state(channel_count: int, run_plans: list[list[tuple[int, int]]]) -> dict:
    return {
        'remaining': [0] * int(channel_count),
        'types': [0] * int(channel_count),
        'indices': [0] * int(channel_count),
        'plans': run_plans,
    }


def _pack_cine_frame(frame: int, values: list[float], channel_count: int, frame_index: int, total_frames: int, absolute_start_offset: int, run_state: dict | None = None, subtitle_data: bytes | None = None) -> bytes:
    frame_data = bytearray()
    total_frames = max(1, int(total_frames))

    if run_state is None:
        # Fallback path: a valid but less compact single-frame run for every
        # channel. Normal export passes a persistent run_state.
        for channel_index in range(int(channel_count)):
            frame_data.extend(struct.pack('<If', 1, float(values[channel_index])))
    else:
        remaining = run_state.get('remaining')
        run_types = run_state.get('types')
        indices = run_state.get('indices')
        plans = run_state.get('plans')
        for channel_index in range(int(channel_count)):
            read_value = False
            if int(remaining[channel_index]) <= 0:
                channel_plan = plans[channel_index] if channel_index < len(plans) else []
                run_index = int(indices[channel_index])
                if run_index < len(channel_plan):
                    run_length, run_type = channel_plan[run_index]
                else:
                    run_length, run_type = max(1, total_frames - int(frame_index)), 0
                run_length = max(1, int(run_length))
                run_type = 1 if int(run_type) else 0
                indices[channel_index] = run_index + 1
                remaining[channel_index] = run_length
                run_types[channel_index] = run_type
                frame_data.extend(struct.pack('<I', (int(run_type) << 28) | int(run_length)))
                read_value = True
            else:
                read_value = int(run_types[channel_index]) != 0

            if read_value:
                frame_data.extend(struct.pack('<f', float(values[channel_index])))
            remaining[channel_index] = int(remaining[channel_index]) - 1

    if subtitle_data is None or len(subtitle_data) == 0:
        frame_data.extend(struct.pack('<i', 0))  # empty SubtitleSet
    else:
        frame_data.extend(bytes(subtitle_data))

    payload = bytearray()
    payload.extend(struct.pack('<ii', 0, int(frame)))
    payload.extend(frame_data)

    absolute_end = int(absolute_start_offset) + len(payload)
    aligned_end = _align_mul_export_offset(absolute_end, MUL_EXPORT_ALIGNMENT)
    if aligned_end > absolute_end:
        payload.extend(b'\x00' * (aligned_end - absolute_end))

    struct.pack_into('<i', payload, 0, max(0, len(payload) - 4))
    return bytes(payload)




def _pack_cine_terminal_frame(absolute_start_offset: int) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack('<ii', 0x0C, -1))
    payload.extend(struct.pack('<i', 0))
    absolute_end = int(absolute_start_offset) + len(payload)
    aligned_end = _align_mul_export_offset(absolute_end, MUL_EXPORT_ALIGNMENT)
    if aligned_end > absolute_end:
        payload.extend(b'\x00' * (aligned_end - absolute_end))
    return bytes(payload)


def _read_source_packet_type_sequence(source_path: str) -> list[int]:
    if not source_path:
        return []
    try:
        data = Path(source_path).read_bytes()
    except Exception:
        return []
    packet_offset = MUL_EXPORT_STREAM_START_OFFSET
    packet_types: list[int] = []
    while packet_offset + MUL_EXPORT_PACKET_HEADER_SIZE <= len(data):
        try:
            packet_type, packet_size = struct.unpack_from('<ii', data, packet_offset)
        except Exception:
            break
        if packet_size < 0:
            break
        packet_data_end = packet_offset + MUL_EXPORT_PACKET_HEADER_SIZE + int(packet_size)
        if packet_data_end > len(data):
            break
        if int(packet_type) in {MUL_EXPORT_PACKET_TYPE_SOUND, MUL_EXPORT_PACKET_TYPE_CINEMATIC}:
            packet_types.append(int(packet_type))
        packet_offset = _align_mul_export_offset(packet_data_end, MUL_EXPORT_ALIGNMENT)
    return packet_types


def _read_source_stream_header_block(source_path: str) -> bytes | None:
    if not source_path:
        return None
    try:
        data = Path(source_path).read_bytes()
    except Exception:
        return None
    if len(data) < MUL_EXPORT_STREAM_START_OFFSET:
        return None
    return bytes(data[:MUL_EXPORT_STREAM_START_OFFSET])


def _read_source_sound_packet_payloads(source_path: str) -> list[bytes]:
    if not source_path:
        return []
    try:
        data = Path(source_path).read_bytes()
    except Exception:
        return []
    packet_offset = MUL_EXPORT_STREAM_START_OFFSET
    payloads: list[bytes] = []
    while packet_offset + MUL_EXPORT_PACKET_HEADER_SIZE <= len(data):
        try:
            packet_type, packet_size = struct.unpack_from('<ii', data, packet_offset)
        except Exception:
            break
        if packet_size < 0:
            break
        payload_start = packet_offset + MUL_EXPORT_PACKET_HEADER_SIZE
        payload_end = payload_start + int(packet_size)
        if payload_end > len(data):
            break
        if int(packet_type) == MUL_EXPORT_PACKET_TYPE_SOUND:
            payloads.append(bytes(data[payload_start:payload_end]))
        packet_offset = _align_mul_export_offset(payload_end, MUL_EXPORT_ALIGNMENT)
    return payloads


def _find_multiplex_audio_wav_path(context, multiplex_obj) -> str:
    if multiplex_obj is None:
        return ''

    candidates: list[str] = []
    try:
        candidates.append(str(multiplex_obj.get('trlau_mul_audio_wav_path', '') or ''))
    except Exception:
        pass

    strip = _find_multiplex_audio_strip(context, multiplex_obj)
    if strip is not None:
        for key in ('trlau_mul_temp_wav_path', 'trlau_mul_audio_wav_path'):
            try:
                candidates.append(str(strip.get(key, '') or ''))
            except Exception:
                pass
        sound = getattr(strip, 'sound', None)
        if sound is not None:
            try:
                candidates.append(bpy.path.abspath(str(getattr(sound, 'filepath', '') or '')))
            except Exception:
                pass

    for candidate in candidates:
        if not candidate:
            continue
        try:
            candidate_path = Path(bpy.path.abspath(candidate))
        except Exception:
            candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path)
    return ''


def _decode_wav_pcm_samples(wav_path: str) -> tuple[int, list[list[int]]]:
    with wave.open(str(wav_path), 'rb') as wav_file:
        sample_rate = int(wav_file.getframerate())
        channel_count = int(wav_file.getnchannels())
        sample_width = int(wav_file.getsampwidth())
        frame_count = int(wav_file.getnframes())
        compression = str(wav_file.getcomptype() or 'NONE')
        if compression not in {'NONE', 'not compressed'}:
            raise ValueError(f'Unsupported WAV compression: {compression}')
        raw = wav_file.readframes(frame_count)

    if sample_rate <= 0 or channel_count <= 0:
        raise ValueError('WAV has no valid sample rate/channel count')
    if channel_count > 12:
        raise ValueError(f'WAV has too many channels for MUL audio: {channel_count}')
    if sample_width not in {1, 2, 3, 4}:
        raise ValueError(f'Unsupported WAV sample width: {sample_width} bytes')

    channel_samples: list[list[int]] = [[] for _ in range(channel_count)]
    frame_stride = channel_count * sample_width
    if frame_stride <= 0:
        return sample_rate, channel_samples

    frame_count = len(raw) // frame_stride
    for frame_index in range(frame_count):
        frame_base = frame_index * frame_stride
        for channel_index in range(channel_count):
            sample_base = frame_base + (channel_index * sample_width)
            sample_bytes = raw[sample_base:sample_base + sample_width]
            if sample_width == 1:
                value = (int(sample_bytes[0]) - 128) << 8
            elif sample_width == 2:
                value = struct.unpack('<h', sample_bytes)[0]
            elif sample_width == 3:
                raw_int = int.from_bytes(sample_bytes + (b'\xff' if sample_bytes[2] & 0x80 else b'\x00'), 'little', signed=True)
                value = int(raw_int >> 8)
            else:
                raw_int = struct.unpack('<i', sample_bytes)[0]
                value = int(raw_int >> 16)
            channel_samples[channel_index].append(max(-32768, min(32767, int(value))))
    return sample_rate, channel_samples


_MUL_AUDIO_VALUE_TABLE = (
    0x0800, 0x1800, 0x2800, 0x3800, 0x4800, 0x5800, 0x6800, 0x7800,
    -0x0800, -0x1800, -0x2800, -0x3800, -0x4800, -0x5800, -0x6800, -0x7800,
)
_MUL_AUDIO_MULTIPLIER_TABLE = (
    28, 32, 36, 40, 44, 48, 52, 56,
    64, 68, 76, 84, 92, 100, 112, 124,
    136, 148, 164, 180, 200, 220, 240, 264,
    292, 320, 352, 388, 428, 472, 520, 572,
    628, 692, 760, 836, 920, 1012, 1116, 1228,
    1348, 1484, 1632, 1796, 1976, 2176, 2392, 2632,
    2896, 3184, 3504, 3852, 4240, 4664, 5128, 5644,
    6208, 6828, 7512, 8264, 9088, 9996, 10996, 12096,
    13308, 14640, 16104, 17712, 19484, 21432, 23576, 25936,
    28528, 31380, 32764, 32764, 32764, 32764, 32764, 32764,
    32764, 32764, 32764, 32764, 32764, 32764, 32764, 32764,
    32764,
)
_MUL_AUDIO_INDEX_CHANGE_TABLE = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)
_MUL_AUDIO_SAMPLES_PER_BLOCK = 64
_MUL_AUDIO_BYTES_PER_BLOCK = 36


def _mul_audio_get_multiplier_index(multiplier: int) -> int:
    multiplier = int(multiplier)
    left = 0
    right = len(_MUL_AUDIO_MULTIPLIER_TABLE)
    while left < right - 1:
        pivot = (left + right) // 2
        if multiplier < int(_MUL_AUDIO_MULTIPLIER_TABLE[pivot]):
            right = pivot
        else:
            left = pivot
    return int(left)


def _mul_audio_encode_sample(prev_sample: int, new_sample: int, index: int) -> int:
    prev_sample = max(-32768, min(32767, int(prev_sample)))
    new_sample = max(-32768, min(32767, int(new_sample)))
    index = max(0, min(len(_MUL_AUDIO_MULTIPLIER_TABLE) - 1, int(index)))
    delta = int(new_sample) - int(prev_sample)
    value = int((int(delta) << 16) / int(_MUL_AUDIO_MULTIPLIER_TABLE[index]))
    if value >= 0:
        code = min((int(value) * 8) // 0x8000, 7)
    else:
        code = 8 + min((-int(value) * 8) // 0x8000, 7)
    return int(code) & 0xF


def _mul_audio_decode_sample(prev_sample: int, index: int, code: int) -> int:
    index = max(0, min(len(_MUL_AUDIO_MULTIPLIER_TABLE) - 1, int(index)))
    code = int(code) & 0xF
    delta = (int(_MUL_AUDIO_VALUE_TABLE[code]) * int(_MUL_AUDIO_MULTIPLIER_TABLE[index])) >> 16
    return max(-32768, min(32767, int(prev_sample) + int(delta)))


def _mul_audio_encode_block(samples: list[int], start_sample: int) -> bytes:
    def read_sample(offset: int) -> int:
        index = int(start_sample) + int(offset)
        if 0 <= index < len(samples):
            return max(-32768, min(32767, int(samples[index])))
        return 0

    input_offset = 0
    sample = read_sample(input_offset)
    input_offset += 1
    output = bytearray(_MUL_AUDIO_BYTES_PER_BLOCK)
    struct.pack_into('<h', output, 0, int(sample))

    prev_sample = int(sample)
    sample = read_sample(input_offset)
    input_offset += 1
    predictor_index = _mul_audio_get_multiplier_index(abs(int(sample) - int(prev_sample)) // 2)
    struct.pack_into('<h', output, 2, int(predictor_index))

    output_offset = 4
    upper_nibble = True
    for _ in range(1, _MUL_AUDIO_SAMPLES_PER_BLOCK):
        code = _mul_audio_encode_sample(prev_sample, sample, predictor_index)
        if upper_nibble:
            if output_offset < _MUL_AUDIO_BYTES_PER_BLOCK:
                output[output_offset] |= (int(code) & 0xF) << 4
            output_offset += 1
        else:
            if output_offset < _MUL_AUDIO_BYTES_PER_BLOCK:
                output[output_offset] = int(code) & 0xF

        prev_sample = _mul_audio_decode_sample(prev_sample, predictor_index, code)
        predictor_index = max(0, min(len(_MUL_AUDIO_MULTIPLIER_TABLE) - 1, int(predictor_index) + int(_MUL_AUDIO_INDEX_CHANGE_TABLE[code])))
        sample = read_sample(input_offset)
        input_offset += 1
        upper_nibble = not upper_nibble

    return bytes(output)



def _read_source_mul_sound_packet_block_plan(source_path: str, channel_count: int) -> list[int]:
    try:
        path = Path(bpy.path.abspath(str(source_path or '')))
    except Exception:
        path = Path(str(source_path or ''))
    if not str(path) or not path.exists() or not path.is_file():
        return []

    try:
        data = path.read_bytes()
    except Exception:
        return []

    channel_count = max(1, int(channel_count or 1))
    plan: list[int] = []
    offset = MUL_EXPORT_STREAM_START_OFFSET
    data_len = len(data)
    while offset + MUL_EXPORT_PACKET_HEADER_SIZE <= data_len:
        try:
            packet_type, packet_size = struct.unpack_from('<ii', data, offset)
        except Exception:
            break
        payload_start = offset + MUL_EXPORT_PACKET_HEADER_SIZE
        payload_end = payload_start + int(packet_size)
        if packet_type not in {MUL_EXPORT_PACKET_TYPE_SOUND, MUL_EXPORT_PACKET_TYPE_CINEMATIC, 2}:
            break
        if packet_size < 0 or payload_end > data_len:
            break

        if int(packet_type) == MUL_EXPORT_PACKET_TYPE_SOUND:
            entry_sizes: list[int] = []
            entry_offset = payload_start
            while entry_offset + 16 <= payload_end:
                try:
                    entry_size, _entry_channel = struct.unpack_from('<ii', data, entry_offset)
                except Exception:
                    break
                entry_data_start = entry_offset + 16
                entry_data_end = entry_data_start + int(entry_size)
                if entry_size < 0 or entry_data_end > payload_end:
                    break
                entry_sizes.append(int(entry_size))
                entry_offset = entry_data_end

            packet_blocks = 0
            if len(entry_sizes) == 1 and channel_count > 1:
                # Retail MULs commonly store all channels in one channel-data
                # entry, split channel-major inside the blob.
                denom = _MUL_AUDIO_BYTES_PER_BLOCK * channel_count
                if denom > 0 and entry_sizes[0] % denom == 0:
                    packet_blocks = entry_sizes[0] // denom
            if packet_blocks <= 0 and entry_sizes:
                # Fallback for one entry per output channel.
                per_entry_blocks = [size // _MUL_AUDIO_BYTES_PER_BLOCK for size in entry_sizes if size % _MUL_AUDIO_BYTES_PER_BLOCK == 0]
                if per_entry_blocks:
                    packet_blocks = max(per_entry_blocks)
            if packet_blocks > 0:
                plan.append(int(packet_blocks))

        offset = _align_mul_export_offset(payload_end, MUL_EXPORT_ALIGNMENT)

    return plan

def _mul_audio_packet_block_plan(sample_count: int, sample_rate: int, fps: float) -> list[int]:
    if sample_count <= 0 or sample_rate <= 0:
        return []
    try:
        fps_value = float(fps) if float(fps) > 0.0 else 30.0
    except Exception:
        fps_value = 30.0

    plan: list[int] = []
    cumulative_blocks = 0
    packet_index = 0
    while cumulative_blocks * _MUL_AUDIO_SAMPLES_PER_BLOCK < int(sample_count):
        target_samples = float(packet_index + 1) * (float(sample_rate) / fps_value)
        candidates = (20, 24)
        block_count = min(candidates, key=lambda value: abs(((cumulative_blocks + value) * _MUL_AUDIO_SAMPLES_PER_BLOCK) - target_samples))
        plan.append(int(block_count))
        cumulative_blocks += int(block_count)
        packet_index += 1
    return plan


def _build_mul_sound_packet_payloads_from_wav(wav_path: str, fps: float, source_mul_path: str = '') -> tuple[list[bytes], dict]:
    sample_rate, channel_samples = _decode_wav_pcm_samples(wav_path)
    channel_count = len(channel_samples)
    if channel_count <= 0:
        return [], {}

    sample_count = max((len(samples) for samples in channel_samples), default=0)
    if sample_count <= 0:
        return [], {}

    for channel_index, samples in enumerate(channel_samples):
        if len(samples) < sample_count:
            pad_value = int(samples[-1]) if samples else 0
            samples.extend([pad_value] * (sample_count - len(samples)))

    source_block_plan = _read_source_mul_sound_packet_block_plan(source_mul_path, channel_count)
    if source_block_plan and (sum(source_block_plan) * _MUL_AUDIO_SAMPLES_PER_BLOCK) >= sample_count:
        block_plan = source_block_plan
    else:
        block_plan = _mul_audio_packet_block_plan(sample_count, sample_rate, fps)
    if not block_plan:
        return [], {}

    sample_cursor = 0
    packet_payloads: list[bytes] = []
    for blocks_in_packet in block_plan:
        audio_data = bytearray()
        for channel_index in range(channel_count):
            channel = channel_samples[channel_index]
            for block_index in range(int(blocks_in_packet)):
                block_start = sample_cursor + (block_index * _MUL_AUDIO_SAMPLES_PER_BLOCK)
                block_bytes = _mul_audio_encode_block(channel, block_start)
                audio_data.extend(block_bytes)

        # The retail MUL sound packets seen so far store one channel-data entry
        # whose data blob is split channel-major by the decoder/header channelCount.
        payload = bytearray()
        payload.extend(struct.pack('<ii8x', len(audio_data), 0))
        payload.extend(audio_data)
        packet_payloads.append(bytes(payload))
        sample_cursor += int(blocks_in_packet) * _MUL_AUDIO_SAMPLES_PER_BLOCK

    return packet_payloads, {
        'wav_path': str(wav_path),
        'sample_rate': int(sample_rate),
        'sample_count': int(sample_count),
        'channel_count': int(channel_count),
        'encoded_packet_count': int(len(packet_payloads)),
        'encoded_block_count': int(sum(block_plan)),
    }


def _write_mul_stream_packet(output: bytearray, packet_type: int, payload: bytes) -> None:
    output.extend(struct.pack('<ii8x', int(packet_type), int(len(payload))))
    output.extend(payload)
    _pad_mul_export_alignment(output, MUL_EXPORT_ALIGNMENT)


def _default_export_packet_sequence(cine_packet_count: int, sound_packet_count: int) -> list[int]:
    sequence: list[int] = []
    cine_remaining = int(cine_packet_count)
    sound_remaining = int(sound_packet_count)
    if cine_remaining <= 0:
        return [MUL_EXPORT_PACKET_TYPE_SOUND] * max(0, sound_remaining)

    # Keep a small cinematic lead-in so the header and early channel runs are
    # available before sound packets begin, then distribute the rest.
    lead_cine = min(cine_remaining, 142 if sound_remaining > 0 else cine_remaining)
    sequence.extend([MUL_EXPORT_PACKET_TYPE_CINEMATIC] * lead_cine)
    cine_remaining -= lead_cine

    while cine_remaining > 0 or sound_remaining > 0:
        if sound_remaining > 0:
            sequence.append(MUL_EXPORT_PACKET_TYPE_SOUND)
            sound_remaining -= 1
        if cine_remaining > 0:
            sequence.append(MUL_EXPORT_PACKET_TYPE_CINEMATIC)
            cine_remaining -= 1
    return sequence


def _trim_mul_cache(cache: dict) -> None:
    while len(cache) > _MUL_CACHE_MAX_ITEMS:
        try:
            cache.pop(next(iter(cache)))
        except Exception:
            break


def _multiplex_object_pointer(obj) -> int:
    try:
        return int(obj.as_pointer())
    except Exception:
        return id(obj)


def _multiplex_storage_signature(obj) -> tuple:
    if obj is None:
        return (0, '', 0)
    try:
        count = int(obj.get(MUL_MULTIPLEX_OBJECT_CHUNK_COUNT, 0) or 0)
    except Exception:
        count = 0
    if count <= 0:
        legacy_values = []
        for key in (
            'trlau_mul_name',
            'trlau_mul_sz_name',
            'trlau_mul_main_unit_id',
            'trlau_mul_source',
            'trlau_mul_anchor_count',
            'trlau_mul_skeleton_count',
            'trlau_mul_camera_count',
        ):
            try:
                legacy_values.append((key, obj.get(key)))
            except Exception:
                legacy_values.append((key, None))
        return (_multiplex_object_pointer(obj), getattr(obj, 'name', ''), 0, tuple(legacy_values))

    first_key = f'{MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX}{0:04d}'
    last_key = f'{MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX}{count - 1:04d}'
    try:
        first = str(obj.get(first_key, '') or '')
    except Exception:
        first = ''
    try:
        last = str(obj.get(last_key, '') or '')
    except Exception:
        last = ''
    return (
        _multiplex_object_pointer(obj),
        getattr(obj, 'name', ''),
        count,
        len(first),
        len(last),
        first[:96],
        first[-96:],
        last[:96],
        last[-96:],
    )


def _multiplex_anchor_signature(multiplex_obj) -> tuple:
    if multiplex_obj is None or not hasattr(multiplex_obj, 'trlau_multiplex_anchors'):
        return ()
    try:
        return tuple(
            (
                int(getattr(anchor, 'area_id', -1)),
                int(getattr(anchor, 'marker_id', -1)),
                int(getattr(anchor, 'first_channel', -1)),
            )
            for anchor in multiplex_obj.trlau_multiplex_anchors
        )
    except Exception:
        return ()


def _multiplex_metadata_cache_key(multiplex_obj) -> tuple:
    return (_multiplex_storage_signature(multiplex_obj), _multiplex_anchor_signature(multiplex_obj))


def _clear_multiplex_runtime_cache(multiplex_obj=None) -> None:
    if multiplex_obj is None:
        _MUL_PAYLOAD_CACHE.clear()
        _MUL_METADATA_CACHE.clear()
        return
    pointer = _multiplex_object_pointer(multiplex_obj)
    for cache in (_MUL_PAYLOAD_CACHE, _MUL_METADATA_CACHE):
        for key in list(cache.keys()):
            try:
                signature = key if cache is _MUL_PAYLOAD_CACHE else key[0]
                if isinstance(signature, tuple) and signature and signature[0] == pointer:
                    del cache[key]
            except Exception:
                pass


def _has_multiplex_storage_payload(obj) -> bool:
    if obj is None:
        return False
    try:
        return int(obj.get(MUL_MULTIPLEX_OBJECT_CHUNK_COUNT, 0) or 0) > 0
    except Exception:
        return False


def _is_multiplex_empty(obj) -> bool:
    return obj is not None and (
        _has_multiplex_storage_payload(obj)
        or bool(obj.get('trlau_multiplex_empty'))
        or str(obj.get('trlau_type', '') or '') == 'Multiplex'
    )


def _write_multiplex_payload(multiplex_obj, payload: dict) -> None:
    _clear_multiplex_runtime_cache(multiplex_obj)
    encoded_payload = MultiplexStreamImporterMixin._encode_multiplex_storage_payload(payload)
    MultiplexStreamImporterMixin._write_multiplex_storage_payload_to_object(multiplex_obj, encoded_payload)
    try:
        signature = _multiplex_storage_signature(multiplex_obj)
        _MUL_PAYLOAD_CACHE[signature] = payload
        _trim_mul_cache(_MUL_PAYLOAD_CACHE)
    except Exception:
        pass


def _get_multiplex_sz_name(self) -> str:
    try:
        payload = _read_multiplex_payload(self)
        return str(payload.get('name', self.get('trlau_mul_name', self.get('trlau_mul_sz_name', ''))) or '')
    except Exception:
        return str(self.get('trlau_mul_name', self.get('trlau_mul_sz_name', '')) or '')


def _set_multiplex_sz_name(self, value: str):
    value = str(value or '')
    try:
        payload = _read_multiplex_payload(self)
        if str(payload.get('name', '') or '') == value:
            return
        payload['name'] = value
        _write_multiplex_payload(self, payload)
    except Exception:
        if str(self.get('trlau_mul_name', '') or '') != value:
            self['trlau_mul_name'] = value
        _clear_multiplex_runtime_cache(self)


def _get_multiplex_main_unit_id(self) -> int:
    try:
        payload = _read_multiplex_payload(self)
        return int(payload.get('main_unit_id', self.get('trlau_mul_main_unit_id', -1)) or -1)
    except Exception:
        try:
            return int(self.get('trlau_mul_main_unit_id', -1) or -1)
        except Exception:
            return -1


def _set_multiplex_main_unit_id(self, value: int):
    value = int(value)
    try:
        payload = _read_multiplex_payload(self)
        if int(payload.get('main_unit_id', -1) or -1) == value:
            return
        payload['main_unit_id'] = value
        _write_multiplex_payload(self, payload)
    except Exception:
        try:
            current = int(self.get('trlau_mul_main_unit_id', -1) or -1)
        except Exception:
            current = -1
        if current != value:
            self['trlau_mul_main_unit_id'] = value
        _clear_multiplex_runtime_cache(self)

def _active_or_selected_armature(context):
    active = getattr(context, 'active_object', None)
    if active is not None and getattr(active, 'type', None) == 'ARMATURE':
        return active
    for obj in getattr(context, 'selected_objects', []) or []:
        if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
            return obj
    return None


def _poll_armature_object(self, obj):
    return obj is not None and getattr(obj, 'type', None) == 'ARMATURE'


def _poll_camera_object(self, obj):
    return obj is not None and getattr(obj, 'type', None) == 'CAMERA'




class TRLAU_PG_multiplex_anchor(bpy.types.PropertyGroup):
    area_id: IntProperty(name='areaID', default=-1)
    marker_id: IntProperty(name='markerID', default=-1)
    first_channel: IntProperty(name='firstChannel', default=-1)

class TRLAU_PG_multiplex_skeleton_binding(bpy.types.PropertyGroup):
    skeleton_index: IntProperty(name='Skeleton Index', default=0, min=0)
    target_armature: PointerProperty(
        name='Armature',
        description='Armature that will receive this MUL skeleton animation',
        type=bpy.types.Object,
        poll=_poll_armature_object,
    )



def _get_multiplex_binding(multiplex_obj, skeleton_index: int, create: bool = True):
    if multiplex_obj is None or not hasattr(multiplex_obj, 'trlau_multiplex_skeleton_bindings'):
        return None
    skeleton_index = int(skeleton_index)
    bindings = multiplex_obj.trlau_multiplex_skeleton_bindings
    for binding in bindings:
        if int(getattr(binding, 'skeleton_index', -1)) == skeleton_index:
            return binding
    if not create:
        return None
    binding = bindings.add()
    binding.skeleton_index = skeleton_index
    return binding


def _ensure_multiplex_bindings(multiplex_obj, skeletons) -> int:
    if multiplex_obj is None or not hasattr(multiplex_obj, 'trlau_multiplex_skeleton_bindings'):
        return 0
    created = 0
    for fallback_index, skeleton in enumerate(list(skeletons or [])):
        skeleton_index = int(skeleton.get('index', fallback_index) or fallback_index)
        if _get_multiplex_binding(multiplex_obj, skeleton_index, create=False) is None:
            _get_multiplex_binding(multiplex_obj, skeleton_index, create=True)
            created += 1
    return created


def _multiplex_bindings_ready(multiplex_obj, skeletons) -> bool:
    if multiplex_obj is None or not hasattr(multiplex_obj, 'trlau_multiplex_skeleton_bindings'):
        return False
    for fallback_index, skeleton in enumerate(list(skeletons or [])):
        skeleton_index = int(skeleton.get('index', fallback_index) or fallback_index)
        if _get_multiplex_binding(multiplex_obj, skeleton_index, create=False) is None:
            return False
    return True



def _resolve_armature_by_name(name: str):
    obj = bpy.data.objects.get(str(name or ''))
    if obj is not None and getattr(obj, 'type', None) == 'ARMATURE':
        return obj
    return None


def _find_context_multiplex_object(context, preferred_name: str = ''):
    if preferred_name:
        obj = bpy.data.objects.get(preferred_name)
        if _is_multiplex_empty(obj):
            return obj

    active = getattr(context, 'active_object', None)
    if _is_multiplex_empty(active):
        return active

    for obj in getattr(context, 'selected_objects', []) or []:
        if _is_multiplex_empty(obj):
            return obj

    return None


# -----------------------------------------------------------------------------
# MUL subtitle viewport preview
# -----------------------------------------------------------------------------

_MUL_SUBTITLE_DRAW_HANDLE = None
_MUL_SUBTITLE_PREVIEW_CACHE: dict[str, tuple[str, list[tuple[int, list[dict[str, str]]]], str]] = {}
_MUL_SUBTITLE_TEXT_REDRAW_SIGNATURES: dict[str, tuple[int, int]] = {}
_MUL_SUBTITLE_PREVIEW_STAGE_FRAMES = 90


def _subtitle_preview_timeline(multiplex_obj) -> tuple[list[tuple[int, list[dict[str, str]]]], str]:
    if multiplex_obj is None:
        return [], ''
    try:
        text_name = str(multiplex_obj.get(MUL_SUBTITLE_TEXT_PROP, '') or '')
    except Exception:
        text_name = ''
    if not text_name:
        return [], ''

    text_block = bpy.data.texts.get(text_name)
    if text_block is None:
        return [], f'Subtitle Text datablock not found: {text_name}'

    try:
        source = text_block.as_string()
    except Exception as exc:
        return [], f'Could not read subtitle Text: {exc}'

    cached = _MUL_SUBTITLE_PREVIEW_CACHE.get(text_block.name)
    if cached is not None and cached[0] == source:
        return cached[1], cached[2]

    endian = '<'
    try:
        payload = _read_multiplex_payload(multiplex_obj)
        endian = str(payload.get('endian', '<') or '<')
    except Exception:
        pass

    timeline: list[tuple[int, list[dict[str, str]]]] = []
    error = ''
    try:
        _count, frame_map = parse_subtitle_text(source, endian=endian)
        for frame, blob in sorted(frame_map.items()):
            decoded = decode_subtitle_set(blob, endian=endian)
            if decoded is None:
                continue
            entries, _suffix = decoded
            clean_entries = []
            for entry in entries:
                language = str(entry.get('language', '') or '')
                dialogue = str(entry.get('text', '') or '')
                if dialogue:
                    clean_entries.append({'language': language, 'text': dialogue})
            if clean_entries:
                timeline.append((int(frame), clean_entries))
    except Exception as exc:
        error = str(exc)

    # Discard stale cache entries if a Text was renamed/deleted/recreated.
    if len(_MUL_SUBTITLE_PREVIEW_CACHE) > 32:
        _MUL_SUBTITLE_PREVIEW_CACHE.clear()
    _MUL_SUBTITLE_PREVIEW_CACHE[text_block.name] = (source, timeline, error)
    return timeline, error


def _subtitle_preview_entries_for_selected_language(multiplex_obj, entries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return only entries that count as a trigger for the current preview language.

    A MUL subtitle frame that does not contain the selected language must not
    advance or replace the visible stack for that language.
    """
    if not entries:
        return []
    try:
        language = str(getattr(multiplex_obj, 'trlau_mul_subtitle_preview_language', '') or '').strip()
    except Exception:
        language = ''

    if language == '*':
        return [entry for entry in entries if str(entry.get('text', '') or '')]

    if language == 'AUTO':
        language = ''

    if language:
        return [
            entry for entry in entries
            if str(entry.get('language', '') or '') == language
            and str(entry.get('text', '') or '')
        ]

    # Automatic mode: any non-empty subtitle frame is a valid trigger; the
    # renderer will display the first language present on that frame.
    return [entry for entry in entries if str(entry.get('text', '') or '')]


def _active_subtitle_preview_entries(multiplex_obj, current_frame: int):
    timeline, error = _subtitle_preview_timeline(multiplex_obj)
    if error or not timeline:
        return [], error

    current_frame = int(current_frame)
    stage_frames = _MUL_SUBTITLE_PREVIEW_STAGE_FRAMES

    lower = None  # (start_frame, promotion_frame, entries)
    upper = None  # (promotion_frame, entries)

    for trigger_frame, entries in timeline:
        trigger_frame = int(trigger_frame)
        if trigger_frame > current_frame:
            break

        effective_entries = _subtitle_preview_entries_for_selected_language(multiplex_obj, entries)
        if not effective_entries:
            # Events that do not contain the selected language do not alter
            # that language's preview stack.
            continue

        # If the current lower subtitle naturally reached its 90-frame limit
        # before this trigger, promote it at that point first.
        if lower is not None and lower[1] <= trigger_frame:
            upper = (lower[1], lower[2])
            lower = None

        # A new subtitle must appear at its authored frame.  If a previous
        # subtitle is still occupying the lower slot, promote it immediately
        # rather than delaying the new trigger.
        if lower is not None:
            upper = (trigger_frame, lower[2])

        lower = (trigger_frame, trigger_frame + stage_frames, effective_entries)

    # After processing all authored triggers up to the current frame, apply a
    # pending natural 90-frame promotion if one has elapsed.
    if lower is not None and lower[1] <= current_frame:
        upper = (lower[1], lower[2])
        lower = None

    stack = []
    if upper is not None:
        stack.append(('upper', upper[0], upper[1]))
    if lower is not None:
        stack.append(('lower', lower[0], lower[2]))
    return stack, ''


def _select_subtitle_preview_text(multiplex_obj, entries: list[dict[str, str]]) -> list[str]:
    if not entries:
        return []
    try:
        language = str(getattr(multiplex_obj, 'trlau_mul_subtitle_preview_language', '') or '').strip()
    except Exception:
        language = ''

    if language == '*':
        return [
            f'[{entry.get("language", "")}] {entry.get("text", "")}'
            for entry in entries
            if str(entry.get('text', '') or '')
        ]

    if language == 'AUTO':
        language = ''

    if language:
        for entry in entries:
            if str(entry.get('language', '') or '') == language:
                text = str(entry.get('text', '') or '')
                return [text] if text else []
        # A specific language was requested but this trigger has no dialogue
        # for it. Do not fall back to another language.
        return []

    # Auto mode (blank Language ID): use the first language present.
    text = str(entries[0].get('text', '') or '')
    return [text] if text else []


def _wrap_subtitle_preview_line(font_id: int, text: str, max_width: float) -> list[str]:
    paragraphs = str(text or '').replace('\r', '').split('\n')
    wrapped: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            wrapped.append('')
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f'{line} {word}'
            try:
                width = float(blf.dimensions(font_id, candidate)[0])
            except Exception:
                width = float(len(candidate) * 12)
            if width <= max_width:
                line = candidate
            else:
                wrapped.append(line)
                line = word
        wrapped.append(line)
    return wrapped


def _draw_multiplex_subtitle_preview():
    context = bpy.context
    area = getattr(context, 'area', None)
    region = getattr(context, 'region', None)
    if area is None or getattr(area, 'type', '') != 'VIEW_3D':
        return
    if region is None or getattr(region, 'type', '') != 'WINDOW':
        return

    multiplex_obj = _find_context_multiplex_object(context)
    if multiplex_obj is None:
        return
    try:
        if not bool(getattr(multiplex_obj, 'trlau_mul_subtitle_preview_enabled', True)):
            return
    except Exception:
        pass

    active_stack, _error = _active_subtitle_preview_entries(multiplex_obj, int(context.scene.frame_current))
    if not active_stack:
        return

    # Keep the lower and upper slots separate. If the selected language is not
    # present for an event, that slot remains blank instead of falling back to
    # a different language.
    slot_texts: dict[str, list[str]] = {'upper': [], 'lower': []}
    for slot, _trigger_frame, entries in active_stack:
        slot_texts[slot] = _select_subtitle_preview_text(multiplex_obj, entries)

    if not slot_texts['upper'] and not slot_texts['lower']:
        return

    font_id = 0
    width = max(1, int(getattr(region, 'width', 1)))
    height = max(1, int(getattr(region, 'height', 1)))
    try:
        font_size = int(getattr(multiplex_obj, 'trlau_mul_subtitle_preview_font_size', 24))
    except Exception:
        font_size = 24
    font_size = max(8, min(128, font_size))
    try:
        blf.size(font_id, font_size)
    except TypeError:
        # Compatibility with older Blender signatures.
        blf.size(font_id, font_size, 72)

    try:
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 5, 0.0, 0.0, 0.0, 1.0)
        blf.shadow_offset(font_id, 2, -2)
    except Exception:
        pass
    try:
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    except Exception:
        pass

    max_text_width = width * 0.78
    wrapped_slots: dict[str, list[str]] = {'upper': [], 'lower': []}
    for slot in ('upper', 'lower'):
        for text in slot_texts[slot]:
            wrapped_slots[slot].extend(_wrap_subtitle_preview_line(font_id, text, max_text_width))

    if not wrapped_slots['upper'] and not wrapped_slots['lower']:
        return

    line_height = float(font_size) * 1.28
    # Bottom-centred, approximately where in-game cinematic subtitles appear.
    base_y = max(36.0, height * 0.095)
    lower_lines = wrapped_slots['lower']
    upper_lines = wrapped_slots['upper']
    slot_gap = line_height * 0.22

    def draw_block(lines: list[str], bottom_y: float) -> bool:
        if not lines:
            return True
        y = bottom_y + line_height * (len(lines) - 1)
        for line in lines:
            try:
                text_width = float(blf.dimensions(font_id, line)[0])
            except Exception:
                text_width = float(len(line) * font_size * 0.5)
            x = max(10.0, (width - text_width) * 0.5)
            try:
                blf.position(font_id, x, y, 0.0)
                blf.draw(font_id, line)
            except Exception:
                return False
            y -= line_height
        return True

    if lower_lines and not draw_block(lower_lines, base_y):
        return

    # Even when the lower slot is blank, keep an aged subtitle visibly shifted
    # upward rather than letting it fall back to the bottom position.
    lower_height = line_height * max(1, len(lower_lines))
    upper_bottom_y = base_y + lower_height + slot_gap
    if upper_lines and not draw_block(upper_lines, upper_bottom_y):
        return

    try:
        blf.disable(font_id, blf.SHADOW)
    except Exception:
        pass


def _tag_all_view3d_redraws(*_args):
    try:
        wm = bpy.context.window_manager
        for window in getattr(wm, 'windows', []) or []:
            screen = getattr(window, 'screen', None)
            for area in getattr(screen, 'areas', []) or []:
                if getattr(area, 'type', '') == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _poll_mul_subtitle_text_changes():
    active_names: set[str] = set()
    changed = False
    try:
        for obj in bpy.data.objects:
            if not _is_multiplex_empty(obj):
                continue
            if not bool(getattr(obj, 'trlau_mul_subtitle_preview_enabled', True)):
                continue
            text_name = str(obj.get(MUL_SUBTITLE_TEXT_PROP, '') or '')
            if not text_name:
                continue
            text_block = bpy.data.texts.get(text_name)
            if text_block is None:
                continue
            source = text_block.as_string()
            signature = (len(source), hash(source))
            active_names.add(text_name)
            if _MUL_SUBTITLE_TEXT_REDRAW_SIGNATURES.get(text_name) != signature:
                _MUL_SUBTITLE_TEXT_REDRAW_SIGNATURES[text_name] = signature
                changed = True
    except Exception:
        pass

    for stale_name in list(_MUL_SUBTITLE_TEXT_REDRAW_SIGNATURES):
        if stale_name not in active_names:
            _MUL_SUBTITLE_TEXT_REDRAW_SIGNATURES.pop(stale_name, None)

    if changed:
        _tag_all_view3d_redraws()
    return 0.25


def register_subtitle_viewport_preview():
    global _MUL_SUBTITLE_DRAW_HANDLE
    if _MUL_SUBTITLE_DRAW_HANDLE is None:
        try:
            _MUL_SUBTITLE_DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
                _draw_multiplex_subtitle_preview, (), 'WINDOW', 'POST_PIXEL'
            )
        except Exception:
            _MUL_SUBTITLE_DRAW_HANDLE = None
    if _tag_all_view3d_redraws not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_tag_all_view3d_redraws)
    if not bpy.app.timers.is_registered(_poll_mul_subtitle_text_changes):
        bpy.app.timers.register(_poll_mul_subtitle_text_changes, first_interval=0.25, persistent=True)


def unregister_subtitle_viewport_preview():
    global _MUL_SUBTITLE_DRAW_HANDLE
    if _tag_all_view3d_redraws in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_tag_all_view3d_redraws)
    if bpy.app.timers.is_registered(_poll_mul_subtitle_text_changes):
        bpy.app.timers.unregister(_poll_mul_subtitle_text_changes)
    if _MUL_SUBTITLE_DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_MUL_SUBTITLE_DRAW_HANDLE, 'WINDOW')
        except Exception:
            pass
        _MUL_SUBTITLE_DRAW_HANDLE = None
    _MUL_SUBTITLE_PREVIEW_CACHE.clear()
    _MUL_SUBTITLE_TEXT_REDRAW_SIGNATURES.clear()




def _iter_scene_sequences(scene):
    sequence_editor = getattr(scene, 'sequence_editor', None)
    if sequence_editor is None:
        return []
    for attr_name in ('sequences_all', 'sequences', 'strips'):
        collection = getattr(sequence_editor, attr_name, None)
        if collection is not None:
            try:
                return list(collection)
            except Exception:
                pass
    return []


def _multiplex_source_path(multiplex_obj) -> str:
    if multiplex_obj is None:
        return ''
    try:
        payload = _read_multiplex_payload(multiplex_obj)
        source = str(payload.get('source', '') or '')
        if source:
            return source
    except Exception:
        pass
    try:
        return str(multiplex_obj.get('trlau_mul_source', '') or '')
    except Exception:
        return ''


def _find_multiplex_audio_strip(context, multiplex_obj):
    if multiplex_obj is None:
        return None

    strip_name = str(multiplex_obj.get('trlau_mul_audio_strip_name', '') or '')
    source_path = _multiplex_source_path(multiplex_obj)

    candidates = _iter_scene_sequences(context.scene)
    if strip_name:
        for strip in candidates:
            if getattr(strip, 'name', '') == strip_name:
                return strip

    for strip in candidates:
        try:
            if str(strip.get('trlau_mul_multiplex_object', '') or '') == multiplex_obj.name:
                return strip
        except Exception:
            pass

    if source_path:
        for strip in candidates:
            try:
                if str(strip.get('trlau_type', '') or '') == 'MULAudio' and str(strip.get('trlau_mul_source', '') or '') == source_path:
                    return strip
            except Exception:
                pass

    return None


def _multiplex_audio_is_playing(context, multiplex_obj) -> bool:
    strip = _find_multiplex_audio_strip(context, multiplex_obj)
    if strip is None:
        return False
    try:
        return not bool(getattr(strip, 'mute', False))
    except Exception:
        return bool(multiplex_obj.get('trlau_mul_audio_enabled', False))


def _clamp_multiplex_audio_volume(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 1.0
    if not math.isfinite(value):
        return 1.0
    return max(0.0, min(1.0, value))


def _get_multiplex_audio_volume(owner) -> float:
    strip = None
    try:
        strip = _find_multiplex_audio_strip(bpy.context, owner)
    except Exception:
        strip = None
    if strip is not None:
        return _clamp_multiplex_audio_volume(getattr(strip, 'volume', 1.0))
    try:
        return _clamp_multiplex_audio_volume(owner.get('trlau_mul_audio_volume_preview', 1.0))
    except Exception:
        return 1.0


def _set_multiplex_audio_volume(owner, value) -> None:
    volume = _clamp_multiplex_audio_volume(value)
    strip = None
    try:
        strip = _find_multiplex_audio_strip(bpy.context, owner)
    except Exception:
        strip = None
    if strip is not None:
        try:
            strip.volume = volume
        except Exception:
            pass
    try:
        owner['trlau_mul_audio_volume_preview'] = volume
    except Exception:
        pass

def _read_multiplex_payload(multiplex_obj) -> dict:
    if multiplex_obj is None:
        raise ValueError('Select a Multiplex empty')

    signature = _multiplex_storage_signature(multiplex_obj)
    cached = _MUL_PAYLOAD_CACHE.get(signature)
    if isinstance(cached, dict):
        return cached

    raw = MultiplexStreamImporterMixin._read_multiplex_storage_payload_from_object(multiplex_obj)
    if not raw.strip():
        raise ValueError('Multiplex payload not found on object')

    payload = MultiplexStreamImporterMixin._decode_multiplex_storage_payload(raw)
    if int(payload.get('schema', 0) or 0) not in {2}:
        raise ValueError('Unsupported Multiplex data schema')
    _MUL_PAYLOAD_CACHE[signature] = payload
    _trim_mul_cache(_MUL_PAYLOAD_CACHE)
    return payload


def _read_multiplex_metadata(multiplex_obj, payload: dict | None = None) -> dict:
    if multiplex_obj is None:
        raise ValueError('Select a Multiplex empty')

    cache_key = _multiplex_metadata_cache_key(multiplex_obj)
    cached_metadata = _MUL_METADATA_CACHE.get(cache_key)
    if isinstance(cached_metadata, dict):
        return cached_metadata

    payload_available = True
    if payload is None:
        try:
            payload = _read_multiplex_payload(multiplex_obj)
        except Exception:
            payload = {}
            payload_available = False
    elif not isinstance(payload, dict):
        payload = {}
        payload_available = False

    def _object_int(key: str, default: int) -> int:
        try:
            return int(multiplex_obj.get(key, default) or default)
        except Exception:
            return int(default)

    def _payload_camera_final_frame(camera: dict) -> int:
        final_frame = 0
        for frame_info in list(camera.get('frames', []) or []):
            try:
                if isinstance(frame_info, dict):
                    final_frame = max(final_frame, int(frame_info.get('frame', 0) or 0))
                elif isinstance(frame_info, (list, tuple)) and frame_info:
                    final_frame = max(final_frame, int(frame_info[0]))
            except Exception:
                pass
        return final_frame

    anchors: list[dict] = []
    payload_anchors = list(payload.get('anchors', []) or [])
    if hasattr(multiplex_obj, 'trlau_multiplex_anchors') and len(multiplex_obj.trlau_multiplex_anchors) > 0:
        for fallback_index, anchor in enumerate(multiplex_obj.trlau_multiplex_anchors):
            first_channel = int(payload_anchors[fallback_index].get('first_channel', -1)) if fallback_index < len(payload_anchors) else -1
            anchors.append({
                'area_id': int(getattr(anchor, 'area_id', -1)),
                'marker_id': int(getattr(anchor, 'marker_id', -1)),
                'first_channel': first_channel,
            })
    elif payload_available:
        for anchor in payload_anchors:
            anchors.append({
                'area_id': int(anchor.get('area_id', -1)),
                'marker_id': int(anchor.get('marker_id', -1)),
                'first_channel': int(anchor.get('first_channel', -1)),
            })
    else:
        anchor_count = _object_int('trlau_mul_anchor_count', 0)
        for fallback_index in range(anchor_count):
            prefix = f'trlau_mul_anchor_{fallback_index:03d}'
            anchors.append({
                'area_id': _object_int(f'{prefix}_area_id', -1),
                'marker_id': _object_int(f'{prefix}_marker_id', -1),
                'first_channel': _object_int(f'{prefix}_first_channel', -1),
            })

    skeletons: list[dict] = []
    if int(payload.get('schema', 0) or 0) == 2:
        skeleton_headers = list(payload.get('skeleton_headers', []) or [])
        skeleton_frames = list(payload.get('skeleton_frames', []) or [])
        skeleton_total = max(len(skeleton_headers), len(skeleton_frames))
        for fallback_index in range(skeleton_total):
            header = skeleton_headers[fallback_index] if fallback_index < len(skeleton_headers) else {}
            frames_entry = skeleton_frames[fallback_index] if fallback_index < len(skeleton_frames) else {}
            index = int(frames_entry.get('index', header.get('index', fallback_index)) if isinstance(frames_entry, dict) else header.get('index', fallback_index))
            skeletons.append({
                'index': index,
                'instance_id': int(frames_entry.get('instance_id', header.get('instance_id', -1)) if isinstance(frames_entry, dict) else header.get('instance_id', -1)),
                'num_bones': int(frames_entry.get('num_bones', header.get('num_bones', 0)) if isinstance(frames_entry, dict) else header.get('num_bones', 0)),
                'first_channel': int(header.get('first_channel', -1)),
                'final_frame': MultiplexStreamImporterMixin._mul_skeleton_final_frame(frames_entry if isinstance(frames_entry, dict) else {}),
            })
    elif payload_available:
        for fallback_index, skeleton in enumerate(list(payload.get('skeletons', []) or [])):
            skeletons.append({
                'index': int(skeleton.get('index', fallback_index)),
                'instance_id': int(skeleton.get('instance_id', -1)),
                'num_bones': int(skeleton.get('num_bones', 0)),
                'first_channel': int(skeleton.get('first_channel', -1)),
                'final_frame': MultiplexStreamImporterMixin._mul_skeleton_final_frame(skeleton),
            })
    else:
        skeleton_count = _object_int('trlau_mul_skeleton_count', 0)
        for fallback_index in range(skeleton_count):
            prefix = f'trlau_mul_skeleton_{fallback_index:03d}'
            skeletons.append({
                'index': _object_int(f'{prefix}_index', fallback_index),
                'instance_id': _object_int(f'{prefix}_instance_id', -1),
                'num_bones': _object_int(f'{prefix}_num_bones', 0),
                'first_channel': _object_int(f'{prefix}_first_channel', -1),
                'final_frame': _object_int(f'{prefix}_final_frame', 0),
            })

    cameras: list[dict] = []
    if payload_available:
        for fallback_index, camera in enumerate(list(payload.get('camera_frames', payload.get('cameras', [])) or [])):
            frames = list(camera.get('frames', []) or [])
            cameras.append({
                'index': int(camera.get('index', fallback_index)),
                'first_channel': int(camera.get('first_channel', -1)),
                'frame_count': len(frames),
                'final_frame': _payload_camera_final_frame(camera),
            })
    else:
        camera_count = _object_int('trlau_mul_camera_count', 0)
        for fallback_index in range(camera_count):
            prefix = f'trlau_mul_camera_{fallback_index:03d}'
            cameras.append({
                'index': _object_int(f'{prefix}_index', fallback_index),
                'first_channel': _object_int(f'{prefix}_first_channel', -1),
                'frame_count': _object_int(f'{prefix}_frame_count', 0),
                'final_frame': _object_int(f'{prefix}_final_frame', 0),
            })

    name = str(payload.get('name', '') or '')
    if not name:
        name = str(multiplex_obj.get('trlau_mul_name', multiplex_obj.get('trlau_mul_sz_name', multiplex_obj.name)) or multiplex_obj.name)
    elif 'trlau_mul_name' in multiplex_obj or 'trlau_mul_sz_name' in multiplex_obj:
        # Backwards compatibility for scenes imported before MUL metadata was kept only in the storage payload.
        name = str(multiplex_obj.get('trlau_mul_name', multiplex_obj.get('trlau_mul_sz_name', name)) or name)

    try:
        main_unit_id = int(payload.get('main_unit_id', -1) or -1)
    except Exception:
        main_unit_id = -1
    if 'trlau_mul_main_unit_id' in multiplex_obj:
        try:
            main_unit_id = int(multiplex_obj.get('trlau_mul_main_unit_id', main_unit_id) or main_unit_id)
        except Exception:
            pass

    metadata = {
        'schema': int(payload.get('schema', 2) or 2),
        'source': str(payload.get('source', '') or multiplex_obj.get('trlau_mul_source', '') or multiplex_obj.name),
        'name': name,
        'main_unit_id': int(main_unit_id),
        'anchors': anchors,
        'skeletons': skeletons,
        'cameras': cameras,
    }
    _MUL_METADATA_CACHE[cache_key] = metadata
    _trim_mul_cache(_MUL_METADATA_CACHE)
    return metadata


def _read_multiplex_subtitles(multiplex_obj, payload: dict) -> tuple[int, dict[int, bytes]]:
    try:
        stored_count = max(0, int(payload.get('num_subtitles', 0) or 0))
    except Exception:
        stored_count = 0
    stored_frames = normalize_subtitle_frames(payload.get('subtitle_frames', []))

    text_name = ''
    try:
        text_name = str(multiplex_obj.get(MUL_SUBTITLE_TEXT_PROP, '') or '')
    except Exception:
        text_name = ''
    if not text_name:
        return stored_count, stored_frames

    text_block = bpy.data.texts.get(text_name)
    if text_block is None:
        raise ValueError(f'MUL subtitle Text datablock not found: {text_name}')
    try:
        edited_count, edited_frames = parse_subtitle_text(
            text_block.as_string(),
            endian=str(payload.get('endian', '<') or '<'),
        )
        if int(edited_count) < 0:
            edited_count = stored_count
        return int(edited_count), edited_frames
    except Exception as exc:
        raise ValueError(f'Invalid MUL subtitle Text "{text_block.name}": {exc}') from exc


def _patch_stream_header_subtitle_flag(header_block: bytes, has_subtitles: bool) -> bytes:
    block = bytearray(header_block or b'')
    if len(block) >= 40:
        struct.pack_into('<i', block, 9 * 4, 1 if bool(has_subtitles) else 0)
    return bytes(block)

def _skeleton_summary_label(skeleton: dict) -> str:
    instance_id = int(skeleton.get('instance_id', -1))
    num_bones = int(skeleton.get('num_bones', 0))
    return f'InstanceID {instance_id} | Bones {num_bones}'


def _load_multiplex_skeleton_to_armature(
    operator,
    context,
    payload: dict,
    multiplex_obj: bpy.types.Object,
    skeleton_index: int,
    armature: bpy.types.Object,
    preserve_bone_positions: bool = False,
    keep_root_motion: bool = True,
) -> bpy.types.Action:
    if armature is None or getattr(armature, 'type', None) != 'ARMATURE':
        raise ValueError('Choose an armature for this Multiplex skeleton')

    skeleton_index = int(skeleton_index)
    if int(payload.get('schema', 0) or 0) == 2:
        skeleton_frames = list(payload.get('skeleton_frames', []) or [])
        skeleton_headers = list(payload.get('skeleton_headers', []) or [])
        if skeleton_index < 0 or skeleton_index >= len(skeleton_frames):
            raise ValueError(f'Multiplex skeleton index out of range: {skeleton_index}')
        skeleton = skeleton_frames[skeleton_index]
        header = skeleton_headers[skeleton_index] if skeleton_index < len(skeleton_headers) else {}
        frames = operator._expand_multiplex_bone_frames(list(skeleton.get('frames', []) or []))
        default_bone_transforms = list(header.get('default_bone_transforms', []) or [])
        first_channel = int(header.get('first_channel', -1))
        instance_id = int(skeleton.get('instance_id', header.get('instance_id', -1)))
        num_bones = int(skeleton.get('num_bones', header.get('num_bones', 0)))
    else:
        skeletons = list(payload.get('skeletons', []) or [])
        if skeleton_index < 0 or skeleton_index >= len(skeletons):
            raise ValueError(f'Multiplex skeleton index out of range: {skeleton_index}')
        skeleton = skeletons[skeleton_index]
        frames = operator._expand_multiplex_bone_frames(list(skeleton.get('frames', []) or []))
        default_bone_transforms = list(skeleton.get('default_bone_transforms', []) or [])
        first_channel = int(skeleton.get('first_channel', -1))
        instance_id = int(skeleton.get('instance_id', -1))
        num_bones = int(skeleton.get('num_bones', 0))

    target_bone_count = int(armature_bone_count(armature))
    if target_bone_count != num_bones:
        logger.warning(
            'Loading MUL skeleton %d onto %s with bone-count mismatch: armature=%d MUL=%d; overlapping bone indices will be used and extra tracks/bones ignored',
            skeleton_index,
            armature.name,
            target_bone_count,
            num_bones,
        )

    skeleton_frame_info = {
        'index': skeleton_index,
        'instance_id': instance_id,
        'num_bones': num_bones,
        'frames': frames,
    }
    final_frame = operator._mul_skeleton_final_frame(skeleton_frame_info)

    source_stem = Path(str(payload.get('source', '') or multiplex_obj.name)).stem or multiplex_obj.name
    action_base_name = f'{source_stem}_skel{skeleton_index:02d}_inst{instance_id}_{armature.name}'
    header_skeleton_info = {
        'index': skeleton_index,
        'instance_id': instance_id,
        'num_bones': num_bones,
        'first_channel': first_channel,
        'default_bone_transforms': default_bone_transforms,
    }

    return operator._build_action_from_mul_skeleton_frames(
        context,
        armature,
        action_base_name,
        skeleton_frame_info,
        header_skeleton_info,
        final_frame,
        time_per_frame=1,
        extra_properties=None,
        assign_to_armature=True,
        preserve_bone_positions=bool(preserve_bone_positions),
        keep_root_motion=True,
    )




class TRLAU_OT_export_multiplex_mul(MultiplexStreamImporterMixin, bpy.types.Operator, ExportHelper):
    bl_idname = 'trlau.export_multiplex_mul'
    bl_label = 'Export MUL'
    bl_description = 'Export the selected Multiplex object as a .mul file'
    bl_options = {'UNDO'}

    filename_ext = '.mul'
    filter_glob: StringProperty(default='*.mul', options={'HIDDEN'})
    multiplex_object_name: StringProperty(name='Multiplex Object', default='')
    use_scene_edits: BoolProperty(
        name='Export Scene Edits',
        description='Always sample assigned armatures and linked MUL camera objects during export',
        default=True,
        options={'HIDDEN'},
    )
    export_retargeted_bone_locations: BoolProperty(
        name='Export Retargeted Bone Locations',
        description='Always export sampled armature bone location channels during scene-edit export',
        default=True,
        options={'HIDDEN'},
    )
    reencode_audio_from_wav: BoolProperty(
        name='Re-encode Audio from WAV',
        description='Encode the linked WAV bridge into MUL audio packets. Leave off to preserve original compressed sound packets when the source MUL is available',
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _find_context_multiplex_object(context) is not None

    def invoke(self, context, event):
        multiplex_obj = _find_context_multiplex_object(context, self.multiplex_object_name)
        if multiplex_obj is not None:
            self.multiplex_object_name = multiplex_obj.name
            try:
                metadata = _read_multiplex_metadata(multiplex_obj)
                stem = str(metadata.get('name', multiplex_obj.name) or multiplex_obj.name)
            except Exception:
                stem = str(multiplex_obj.get('trlau_mul_name', multiplex_obj.get('trlau_mul_sz_name', multiplex_obj.name)) or multiplex_obj.name)
            safe_stem = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in stem).strip('_') or 'multiplex'
            if not self.filepath:
                self.filepath = bpy.path.ensure_ext(str(Path('//') / f'{safe_stem}.mul'), '.mul')
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.prop(self, 'reencode_audio_from_wav')

    def execute(self, context):
        multiplex_obj = _find_context_multiplex_object(context, self.multiplex_object_name)
        if multiplex_obj is None:
            self.report({'ERROR'}, 'Select a Multiplex empty')
            return {'CANCELLED'}

        try:
            payload = _read_multiplex_payload(multiplex_obj)
            metadata = _read_multiplex_metadata(multiplex_obj, payload)
            self._last_export_messages = []
            self._export_multiplex_mul(context, multiplex_obj, metadata, payload, self.filepath)
        except Exception as exc:
            logger.exception('MUL export failed')
            self.report({'ERROR'}, f'MUL export failed: {exc}')
            return {'CANCELLED'}

        notes = list(getattr(self, '_last_export_messages', []) or [])
        if notes:
            shown = '; '.join(str(note) for note in notes[:3])
            if len(notes) > 3:
                shown += f'; +{len(notes) - 3} more'
            self.report({'INFO'}, f'Exported MUL: {Path(self.filepath).name} ({shown})')
        else:
            self.report({'INFO'}, f'Exported MUL: {Path(self.filepath).name}')
        return {'FINISHED'}

    def _payload_skeleton_entries(self, payload: dict) -> tuple[list[dict], list[dict]]:
        if int(payload.get('schema', 0) or 0) == 2:
            return list(payload.get('skeleton_headers', []) or []), list(payload.get('skeleton_frames', []) or [])
        skeletons = list(payload.get('skeletons', []) or [])
        return skeletons, skeletons

    def _determine_export_final_frame(self, multiplex_obj, metadata: dict, skeleton_entries: list[dict], camera_objects: dict[int, bpy.types.Object], use_scene_edits: bool = False) -> int:
        final_frame = 0
        for skeleton in metadata.get('skeletons', []) or []:
            final_frame = max(final_frame, int(skeleton.get('final_frame', 0) or 0))
        for camera in metadata.get('cameras', []) or []:
            final_frame = max(final_frame, int(camera.get('final_frame', 0) or 0))

        if use_scene_edits:
            for fallback_index, _skeleton in enumerate(skeleton_entries):
                binding = _get_multiplex_binding(multiplex_obj, fallback_index, create=False)
                armature = getattr(binding, 'target_armature', None) if binding is not None else None
                action = getattr(getattr(armature, 'animation_data', None), 'action', None) if armature is not None else None
                action_range = _action_frame_range(action)
                if action_range is not None:
                    final_frame = max(final_frame, int(action_range[1]))

            for camera_obj in camera_objects.values():
                object_action = getattr(getattr(camera_obj, 'animation_data', None), 'action', None)
                camera_action = getattr(getattr(getattr(camera_obj, 'data', None), 'animation_data', None), 'action', None)
                for action in (object_action, camera_action):
                    action_range = _action_frame_range(action)
                    if action_range is not None:
                        final_frame = max(final_frame, int(action_range[1]))

        try:
            scene_end = int(getattr(bpy.context.scene, 'frame_end', 0) or 0)
        except Exception:
            scene_end = 0
        if final_frame <= 0:
            final_frame = max(final_frame, scene_end)
        return max(0, int(final_frame))

    def _build_export_skeleton_frames(self, context, multiplex_obj, metadata: dict, payload: dict, frames: list[int], use_scene_edits: bool = False, export_scene_locations: bool = False) -> tuple[list[dict], list[list[list[dict]]], list[str]]:
        payload_headers, payload_frames = self._payload_skeleton_entries(payload)
        metadata_skeletons = list(metadata.get('skeletons', []) or [])
        skeleton_headers: list[dict] = []
        skeleton_frame_data: list[list[list[dict]]] = []
        warnings: list[str] = []

        for fallback_index, metadata_skeleton in enumerate(metadata_skeletons):
            skeleton_index = int(metadata_skeleton.get('index', fallback_index) or fallback_index)
            payload_header = payload_headers[skeleton_index] if 0 <= skeleton_index < len(payload_headers) else {}
            payload_frame_entry = payload_frames[skeleton_index] if 0 <= skeleton_index < len(payload_frames) else {}
            num_bones = int(metadata_skeleton.get('num_bones', payload_header.get('num_bones', 0)) or 0)
            default_transforms = list(payload_header.get('default_bone_transforms', []) or [])

            binding = _get_multiplex_binding(multiplex_obj, skeleton_index, create=False)
            armature = getattr(binding, 'target_armature', None) if binding is not None else None
            action = getattr(getattr(armature, 'animation_data', None), 'action', None) if armature is not None else None

            compact = list(payload_frame_entry.get('frames', []) or [])
            expanded_raw_frames = self._expand_multiplex_bone_frames(compact)
            raw_root_values = _raw_stored_root_values_by_frame(list(payload_frame_entry.get('root_frames', []) or []), frames)
            raw_bone_frames = _raw_stored_frames_by_frame(expanded_raw_frames, frames, num_bones)

            if use_scene_edits and _should_sample_armature_for_mul_export(armature, action):
                target_bone_count = int(armature_bone_count(armature))
                sampled_bone_count = min(int(num_bones), max(0, target_bone_count))
                if target_bone_count != int(num_bones):
                    warnings.append(
                        f'Skeleton {skeleton_index} sampled from {armature.name} with bone-count mismatch; '
                        f'{sampled_bone_count}/{num_bones} MUL bone track(s) were scene-sampled and the rest stayed raw'
                    )
                    logger.warning(
                        'MUL export skeleton %d bone-count mismatch: %s has %d, MUL skeleton has %d; sampling overlapping tracks and preserving raw missing tracks',
                        skeleton_index,
                        armature.name,
                        target_bone_count,
                        num_bones,
                    )
                if len(default_transforms) < num_bones:
                    mapped = _pose_bone_mapping_by_index(armature, num_bones)
                    generated_default_transforms = [
                        _flatten_matrix_rows(MultiplexStreamImporterMixin._mul_orientation_only_matrix(mapped[i].bone.matrix_local)) if i in mapped else _flatten_matrix_rows(mathutils.Matrix.Identity(4))
                        for i in range(num_bones)
                    ]
                    default_transforms = generated_default_transforms
                used_fast_action_sampling = _armature_can_use_fast_action_sampling(armature, action)
                if used_fast_action_sampling:
                    scene_bone_frames = _sample_mul_skeleton_from_armature(action, armature, default_transforms, frames, sampled_bone_count, raw_bone_frames)
                    sample_source = 'action f-curves (fast)'
                else:
                    scene_bone_frames = _sample_mul_skeleton_from_evaluated_armature(context, action, armature, default_transforms, frames, sampled_bone_count, raw_bone_frames)
                    sample_source = 'evaluated pose'

                if (
                    _action_has_fcurves(action)
                    and not _skeleton_frames_have_transform_variation(scene_bone_frames)
                    and not _armature_has_non_rest_pose(armature)
                    and _skeleton_frames_have_transform_variation(raw_bone_frames)
                ):
                    if used_fast_action_sampling:
                        alternate_bone_frames = _sample_mul_skeleton_from_evaluated_armature(context, action, armature, default_transforms, frames, sampled_bone_count, raw_bone_frames)
                        alternate_source = 'evaluated pose'
                    else:
                        alternate_bone_frames = _sample_mul_skeleton_from_armature(action, armature, default_transforms, frames, sampled_bone_count, raw_bone_frames)
                        alternate_source = 'action f-curves'
                    if _skeleton_frames_have_transform_variation(alternate_bone_frames):
                        scene_bone_frames = alternate_bone_frames
                        sample_source = alternate_source
                    else:
                        bone_frames = raw_bone_frames
                        warnings.append(
                            f'Skeleton {skeleton_index} kept stored raw MUL frames because scene sampling of {armature.name} produced only bind-pose transforms'
                        )
                        skeleton_headers.append({
                            'index': skeleton_index,
                            'instance_id': int(metadata_skeleton.get('instance_id', payload_header.get('instance_id', -1)) or -1),
                            'num_bones': int(num_bones),
                            'first_channel': -1,
                            'default_bone_transforms': default_transforms,
                            'root_values': raw_root_values,
                        })
                        skeleton_frame_data.append(bone_frames)
                        continue

                if len(scene_bone_frames) < int(num_bones):
                    scene_bone_frames.extend([[] for _ in range(int(num_bones) - len(scene_bone_frames))])
                bone_frames = _merge_scene_bone_frames_with_raw(
                    scene_bone_frames,
                    raw_bone_frames,
                    export_scene_locations=bool(export_scene_locations),
                )
                if bool(export_scene_locations):
                    warnings.append(f'Skeleton {skeleton_index} sampled from scene armature {armature.name} ({sample_source}) with retargeted bone locations/scale; static transform noise collapsed')
                else:
                    warnings.append(f'Skeleton {skeleton_index} sampled from scene armature {armature.name} ({sample_source}); original MUL bone locations preserved')
            else:
                bone_frames = raw_bone_frames
                warnings.append(f'Skeleton {skeleton_index} exported from stored raw MUL frames because no usable armature action or static pose edit was found')

            skeleton_headers.append({
                'index': skeleton_index,
                'instance_id': int(metadata_skeleton.get('instance_id', payload_header.get('instance_id', -1)) or -1),
                'num_bones': int(num_bones),
                'first_channel': -1,
                'default_bone_transforms': default_transforms,
                'root_values': raw_root_values,
            })
            skeleton_frame_data.append(bone_frames)

        return skeleton_headers, skeleton_frame_data, warnings

    def _build_export_camera_frames(self, context, multiplex_obj, metadata: dict, payload: dict, frames: list[int], use_scene_edits: bool = False) -> tuple[list[dict], list[list[dict]], list[str]]:
        cameras = []
        camera_frame_data = []
        warnings = []
        camera_objects = _find_multiplex_camera_objects(multiplex_obj, metadata)
        payload_cameras = list(payload.get('camera_frames', payload.get('cameras', [])) or [])
        metadata_cameras = list(metadata.get('cameras', []) or [])
        metadata_camera_by_index = {}
        for fallback_index, camera in enumerate(metadata_cameras):
            try:
                metadata_camera_by_index[int(camera.get('index', fallback_index) or fallback_index)] = camera
            except Exception:
                metadata_camera_by_index[int(fallback_index)] = camera
        detected_camera_indices = sorted(set(metadata_camera_by_index.keys()) | set(int(index) for index in camera_objects.keys()))
        if detected_camera_indices:
            camera_indices = list(range(0, max(detected_camera_indices) + 1))
        else:
            camera_indices = []
        for fallback_index, camera_index in enumerate(camera_indices):
            camera = metadata_camera_by_index.get(int(camera_index), {})
            camera_obj = camera_objects.get(int(camera_index))
            payload_camera = None
            for candidate in payload_cameras:
                try:
                    if int(candidate.get('index', -999999)) == camera_index:
                        payload_camera = candidate
                        break
                except Exception:
                    continue
            if payload_camera is None and fallback_index < len(payload_cameras):
                payload_camera = payload_cameras[fallback_index]

            cameras.append({
                'index': camera_index,
                'first_channel': -1,
            })

            if use_scene_edits:
                if camera_obj is not None:
                    camera_frame_data.append(_sample_camera_frames_from_scene(context, camera_obj, frames))
                    message = f'Camera {camera_index} sampled from scene object {camera_obj.name}'
                    warnings.append(message)
                    logger.info('MUL export %s', message)
                elif payload_camera is not None:
                    raw_frames = _expand_payload_camera_frames(payload_camera)
                    camera_frame_data.append(_raw_stored_camera_frames_by_frame(raw_frames, frames))
                    warnings.append(f'Camera {camera_index} exported from stored raw MUL frames because no linked MUL camera object was found')
                else:
                    camera_frame_data.append(_sample_camera_frames(None, frames))
                    warnings.append(f'Camera {camera_index} exported as a static default camera because no stored or linked MUL camera data was found')
            elif payload_camera is not None:
                raw_frames = _expand_payload_camera_frames(payload_camera)
                camera_frame_data.append(_raw_stored_camera_frames_by_frame(raw_frames, frames))
            else:
                camera_frame_data.append(_sample_camera_frames(None, frames))
                warnings.append(f'Camera {camera_index} exported as a static default camera because no stored or linked MUL camera data was found')
        return cameras, camera_frame_data, warnings

    def _export_multiplex_mul(self, context, multiplex_obj, metadata: dict, payload: dict, filepath: str) -> None:
        camera_objects = _find_multiplex_camera_objects(multiplex_obj, metadata)
        payload_headers, _payload_frames = self._payload_skeleton_entries(payload)
        final_frame = self._determine_export_final_frame(multiplex_obj, metadata, payload_headers, camera_objects, use_scene_edits=True)
        try:
            fps_value = float(getattr(context.scene.render, 'fps', 30) or 30)
        except Exception:
            fps_value = 30.0

        source_mul_path = str(metadata.get('source', '') or '')
        audio_packet_payloads: list[bytes] = []
        audio_info: dict = {}
        preserved_source_header_block: bytes | None = None
        if not bool(self.reencode_audio_from_wav):
            audio_packet_payloads = _read_source_sound_packet_payloads(source_mul_path)
            if audio_packet_payloads:
                preserved_source_header_block = _read_source_stream_header_block(source_mul_path)
                if preserved_source_header_block is not None:
                    audio_info = {'preserved_source_packets': True, 'preserved_source_header': True}
                else:
                    audio_info = {'preserved_source_packets': True, 'sample_count': 0, 'channel_count': 0}
                logger.info('MUL export preserving %d original compressed sound packet(s)', len(audio_packet_payloads))

        if not audio_packet_payloads:
            audio_wav_path = _find_multiplex_audio_wav_path(context, multiplex_obj)
            if audio_wav_path and bool(self.reencode_audio_from_wav):
                try:
                    audio_packet_payloads, audio_info = _build_mul_sound_packet_payloads_from_wav(audio_wav_path, fps_value, source_mul_path=source_mul_path)
                except Exception as exc:
                    logger.warning('MUL audio export skipped; could not encode WAV %s: %s', audio_wav_path, exc)
                    audio_packet_payloads = []
                    audio_info = {}
            elif audio_wav_path:
                logger.info('MUL export found linked WAV audio, but Re-encode Audio from WAV is disabled; writing cinematic-only audio unless original sound packets were available')
            else:
                logger.info('MUL export found no linked WAV audio and no source sound packets; writing cinematic-only stream')

        num_subtitles, subtitle_frames = _read_multiplex_subtitles(multiplex_obj, payload)
        if subtitle_frames:
            final_frame = max(int(final_frame), max((int(frame) for frame in subtitle_frames.keys() if int(frame) >= 0), default=0))
        frames = list(range(0, int(final_frame) + 1)) or [0]
        has_subtitles = bool(num_subtitles > 0 or any(subtitle_blob_has_data(blob) for blob in subtitle_frames.values()))

        anchors = []
        for anchor in list(metadata.get('anchors', []) or []):
            anchors.append({
                'area_id': int(anchor.get('area_id', -1)),
                'marker_id': int(anchor.get('marker_id', -1)),
                'first_channel': -1,
            })

        skeleton_headers, skeleton_frame_data, skeleton_warnings = self._build_export_skeleton_frames(
            context,
            multiplex_obj,
            metadata,
            payload,
            frames,
            use_scene_edits=True,
            export_scene_locations=True,
        )
        cameras, camera_frame_data, camera_warnings = self._build_export_camera_frames(context, multiplex_obj, metadata, payload, frames, use_scene_edits=True)

        # Assign generated channel layout. The header packer writes the same firstChannel values.
        current_channel = 0
        for anchor in anchors:
            anchor['first_channel'] = current_channel
            current_channel += 9
        for skeleton in skeleton_headers:
            skeleton['first_channel'] = current_channel
            current_channel += 2 + (int(skeleton.get('num_bones', 0) or 0) * 10)
        for camera in cameras:
            camera['first_channel'] = current_channel
            current_channel += 11
        channel_count = int(current_channel)

        name = str(metadata.get('name', multiplex_obj.name) or multiplex_obj.name)
        main_unit_id = int(metadata.get('main_unit_id', -1) or -1)
        cine_header = _pack_cine_header(
            name,
            main_unit_id,
            anchors,
            skeleton_headers,
            cameras,
            skeleton_frame_data,
            final_frame,
            num_subtitles=num_subtitles,
        )
        channel_rows = _build_channel_value_rows(frames, anchors, skeleton_headers, skeleton_frame_data, cameras, camera_frame_data, channel_count)
        run_plans = _build_cine_channel_run_plans(channel_rows, channel_count)
        run_state = _new_cine_run_state(channel_count, run_plans)
        static_channel_count = 0
        animated_channel_count = 0
        for plan in run_plans:
            if len(plan) == 1 and int(plan[0][1]) == 0:
                static_channel_count += 1
            else:
                animated_channel_count += 1

        output = bytearray()
        if preserved_source_header_block is not None and audio_packet_payloads and not bool(self.reencode_audio_from_wav):
            preserved_header = _patch_stream_header_subtitle_flag(
                preserved_source_header_block[:MUL_EXPORT_STREAM_START_OFFSET],
                has_subtitles,
            )
            output.extend(preserved_header)
            if len(output) < MUL_EXPORT_STREAM_START_OFFSET:
                output.extend(b'\x00' * (MUL_EXPORT_STREAM_START_OFFSET - len(output)))
        else:
            output.extend(_pack_stream_header(final_frame, audio_info=audio_info, fps=fps_value, has_subtitles=has_subtitles))
            if len(output) > MUL_EXPORT_STREAM_START_OFFSET:
                raise ValueError('Generated stream header exceeds 0x800 bytes')
            output.extend(b'\x00' * (MUL_EXPORT_STREAM_START_OFFSET - len(output)))

        total_frames = len(frames)
        total_cine_packets = total_frames + 1  # one packet per frame plus terminal frameNr=-1
        source_packet_sequence = _read_source_packet_type_sequence(source_mul_path)
        if not source_packet_sequence:
            source_packet_sequence = _default_export_packet_sequence(total_cine_packets, len(audio_packet_payloads))

        cine_packet_index = 0
        sound_packet_index = 0

        def write_next_cine_packet() -> bool:
            nonlocal cine_packet_index
            if cine_packet_index >= total_cine_packets:
                return False
            if cine_packet_index < total_frames:
                frame_index = cine_packet_index
                frame = frames[frame_index]
                packet_prefix = cine_header if frame_index == 0 else b''
                frame_absolute_start = len(output) + MUL_EXPORT_PACKET_HEADER_SIZE + len(packet_prefix)
                frame_payload = _pack_cine_frame(
                    int(frame),
                    channel_rows[frame_index],
                    channel_count,
                    frame_index,
                    total_frames,
                    frame_absolute_start,
                    run_state,
                    subtitle_data=subtitle_frames.get(int(frame)),
                )
                cine_payload = bytearray(packet_prefix)
                cine_payload.extend(frame_payload)
                _write_mul_stream_packet(output, MUL_EXPORT_PACKET_TYPE_CINEMATIC, bytes(cine_payload))
            else:
                terminal_payload = _pack_cine_terminal_frame(len(output) + MUL_EXPORT_PACKET_HEADER_SIZE)
                _write_mul_stream_packet(output, MUL_EXPORT_PACKET_TYPE_CINEMATIC, terminal_payload)
            cine_packet_index += 1
            return True

        def write_next_sound_packet() -> bool:
            nonlocal sound_packet_index
            if sound_packet_index >= len(audio_packet_payloads):
                return False
            _write_mul_stream_packet(output, MUL_EXPORT_PACKET_TYPE_SOUND, audio_packet_payloads[sound_packet_index])
            sound_packet_index += 1
            return True

        for packet_type in source_packet_sequence:
            if int(packet_type) == MUL_EXPORT_PACKET_TYPE_CINEMATIC:
                write_next_cine_packet()
            elif int(packet_type) == MUL_EXPORT_PACKET_TYPE_SOUND:
                write_next_sound_packet()
            if cine_packet_index >= total_cine_packets and sound_packet_index >= len(audio_packet_payloads):
                break

        while cine_packet_index < total_cine_packets or sound_packet_index < len(audio_packet_payloads):
            if cine_packet_index < total_cine_packets:
                write_next_cine_packet()
            if sound_packet_index < len(audio_packet_payloads):
                write_next_sound_packet()

        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(bytes(output))

        export_messages = []
        for message in skeleton_warnings + camera_warnings:
            logger.warning(message)
            if str(message).startswith('Camera '):
                export_messages.append(str(message))
        try:
            self._last_export_messages = export_messages
        except Exception:
            pass
        logger.info(
            'Exported MUL %s: frames=%d channels=%d animated_channels=%d static_channels=%d anchors=%d skeletons=%d cameras=%d subtitles=%d subtitle_frames=%d sound_packets=%d audio_samples=%d bytes=%d',
            target_path.name,
            len(frames),
            channel_count,
            animated_channel_count,
            static_channel_count,
            len(anchors),
            len(skeleton_headers),
            len(cameras),
            int(num_subtitles),
            sum(1 for blob in subtitle_frames.values() if subtitle_blob_has_data(blob)),
            len(audio_packet_payloads),
            int(audio_info.get('sample_count', 0) or 0),
            len(output),
        )


class TRLAU_OT_toggle_multiplex_audio(bpy.types.Operator):
    bl_idname = 'trlau.toggle_multiplex_audio'
    bl_label = 'Toggle MUL Audio'
    bl_description = 'Mute or unmute the audio strip linked to this Multiplex object'
    bl_options = {'UNDO'}

    multiplex_object_name: StringProperty(name='Multiplex Object', default='')
    play_audio: BoolProperty(name='Play Audio', default=True)

    @classmethod
    def poll(cls, context):
        return _find_context_multiplex_object(context) is not None

    def execute(self, context):
        multiplex_obj = _find_context_multiplex_object(context, self.multiplex_object_name)
        if multiplex_obj is None:
            self.report({'ERROR'}, 'Select a Multiplex empty')
            return {'CANCELLED'}

        strip = _find_multiplex_audio_strip(context, multiplex_obj)
        if strip is None:
            self.report({'ERROR'}, 'No linked MUL audio strip was found for this Multiplex object')
            return {'CANCELLED'}

        enabled = bool(self.play_audio)
        try:
            strip.mute = not enabled
        except Exception as exc:
            self.report({'ERROR'}, f'Could not change audio mute state: {exc}')
            return {'CANCELLED'}

        try:
            strip['trlau_mul_audio_enabled'] = enabled
        except Exception:
            pass
        multiplex_obj['trlau_mul_audio_enabled'] = enabled
        multiplex_obj['trlau_mul_audio_strip_name'] = str(getattr(strip, 'name', '') or '')

        self.report({'INFO'}, 'MUL audio enabled' if enabled else 'MUL audio muted')
        return {'FINISHED'}

class TRLAU_OT_apply_multiplex_skeleton_to_armature(AnimationImporterMixin, MultiplexStreamImporterMixin, bpy.types.Operator):
    bl_idname = 'trlau.apply_multiplex_skeleton_to_armature'
    bl_label = 'Load Animation'
    bl_description = 'Bake this stored MUL skeleton track to its chosen armature using that armature rest pose'
    bl_options = {'UNDO'}

    multiplex_object_name: StringProperty(name='Multiplex Object', default='')
    skeleton_index: IntProperty(name='Skeleton Index', default=0, min=0)
    target_armature_name: StringProperty(name='Target Armature', default='')
    preserve_bone_positions: BoolProperty(
        name='Retarget Bone Positions',
        description='Import MUL location keys as retargeted deltas relative to the target armature, instead of copying source child bone placement literally',
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _find_context_multiplex_object(context) is not None

    def execute(self, context):
        multiplex_obj = _find_context_multiplex_object(context, self.multiplex_object_name)
        if multiplex_obj is None:
            self.report({'ERROR'}, 'Select a Multiplex empty')
            return {'CANCELLED'}

        armature = _resolve_armature_by_name(self.target_armature_name)
        if armature is None:
            binding = _get_multiplex_binding(multiplex_obj, int(self.skeleton_index), create=False)
            armature = getattr(binding, 'target_armature', None) if binding is not None else None
        if armature is None or getattr(armature, 'type', None) != 'ARMATURE':
            self.report({'ERROR'}, 'Choose an armature in this Multiplex row')
            return {'CANCELLED'}

        try:
            payload = _read_multiplex_payload(multiplex_obj)
            action = _load_multiplex_skeleton_to_armature(
                self,
                context,
                payload,
                multiplex_obj,
                int(self.skeleton_index),
                armature,
                preserve_bone_positions=bool(self.preserve_bone_positions),
                keep_root_motion=True,
            )
        except Exception as exc:
            logger.exception('Failed to load Multiplex skeleton %s to %s', self.skeleton_index, armature.name if armature else '<none>')
            self.report({'ERROR'}, f'Failed to load animation: {exc}')
            return {'CANCELLED'}

        self.report({'INFO'}, f'Loaded animation to {armature.name}: {action.name}')
        return {'FINISHED'}


class TRLAU_OT_load_all_multiplex_animations(AnimationImporterMixin, MultiplexStreamImporterMixin, bpy.types.Operator):
    bl_idname = 'trlau.load_all_multiplex_animations'
    bl_label = 'Load All Animations'
    bl_description = 'Bake every stored MUL skeleton track that has an armature assigned in the Multiplex list'
    bl_options = {'UNDO'}

    multiplex_object_name: StringProperty(name='Multiplex Object', default='')
    preserve_bone_positions: BoolProperty(
        name='Retarget Bone Positions',
        description='Import MUL location keys as retargeted deltas relative to each target armature, instead of copying source child bone placement literally',
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return _find_context_multiplex_object(context) is not None

    def execute(self, context):
        multiplex_obj = _find_context_multiplex_object(context, self.multiplex_object_name)
        if multiplex_obj is None:
            self.report({'ERROR'}, 'Select a Multiplex empty')
            return {'CANCELLED'}

        try:
            payload = _read_multiplex_payload(multiplex_obj)
            metadata = _read_multiplex_metadata(multiplex_obj, payload)
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to read Multiplex data: {exc}')
            return {'CANCELLED'}

        loaded = []
        skipped = 0
        failed = []
        for fallback_index, skeleton in enumerate(list(metadata.get('skeletons', []) or [])):
            skeleton_index = int(skeleton.get('index', fallback_index) or fallback_index)
            binding = _get_multiplex_binding(multiplex_obj, skeleton_index, create=False)
            armature = getattr(binding, 'target_armature', None) if binding is not None else None
            if armature is None or getattr(armature, 'type', None) != 'ARMATURE':
                skipped += 1
                continue
            try:
                action = _load_multiplex_skeleton_to_armature(
                    self,
                    context,
                    payload,
                    multiplex_obj,
                    skeleton_index,
                    armature,
                    preserve_bone_positions=bool(self.preserve_bone_positions),
                    keep_root_motion=True,
                )
                loaded.append((skeleton_index, armature.name, action.name))
            except Exception as exc:
                logger.exception('Failed to load Multiplex skeleton %s to %s', skeleton_index, armature.name if armature else '<none>')
                failed.append((skeleton_index, armature.name if armature else '<none>', str(exc)))

        if not loaded and not failed:
            self.report({'ERROR'}, 'No Multiplex rows have an armature assigned')
            return {'CANCELLED'}

        if failed:
            first_failure = failed[0]
            self.report({'WARNING'}, f'Loaded {len(loaded)} animation(s), failed {len(failed)}. First failure: skeleton {first_failure[0]} -> {first_failure[1]}: {first_failure[2]}')
        else:
            suffix = f', skipped {skipped} unassigned row(s)' if skipped else ''
            self.report({'INFO'}, f'Loaded {len(loaded)} animation(s){suffix}')
        return {'FINISHED'}




class VIEW3D_PT_trlau_multiplex_header(bpy.types.Panel):
    bl_label = 'Header'
    bl_idname = 'VIEW3D_PT_trlau_multiplex_header'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_order = 0

    @classmethod
    def poll(cls, context):
        return _find_context_multiplex_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        multiplex_obj = _find_context_multiplex_object(context)
        if multiplex_obj is None:
            layout.label(text='Select a Multiplex empty', icon='INFO')
            return

        header_box = layout.box()
        header_box.prop(multiplex_obj, 'trlau_mul_sz_name_ui', text='szName')
        header_box.prop(multiplex_obj, 'trlau_mul_main_unit_id_ui', text='mainUnitID')
        export_row = header_box.row(align=True)
        export_op = export_row.operator(TRLAU_OT_export_multiplex_mul.bl_idname, text='Export MUL', icon='EXPORT')
        export_op.multiplex_object_name = multiplex_obj.name

        subtitle_text_name = str(multiplex_obj.get(MUL_SUBTITLE_TEXT_PROP, '') or '')
        if subtitle_text_name:
            subtitle_row = header_box.row(align=True)
            if bpy.data.texts.get(subtitle_text_name) is not None:
                subtitle_row.label(text=f'Subtitles: {subtitle_text_name}', icon='TEXT')
            else:
                subtitle_row.label(text=f'Subtitles text missing: {subtitle_text_name}', icon='ERROR')

            preview_box = header_box.box()
            preview_box.label(text='Viewport Subtitle Preview', icon='HIDE_OFF')
            preview_box.prop(multiplex_obj, 'trlau_mul_subtitle_preview_enabled', text='Show in Viewport')
            preview_col = preview_box.column(align=True)
            preview_col.enabled = bool(getattr(multiplex_obj, 'trlau_mul_subtitle_preview_enabled', True))
            preview_col.prop(multiplex_obj, 'trlau_mul_subtitle_preview_language', text='Language')
            preview_col.prop(multiplex_obj, 'trlau_mul_subtitle_preview_font_size', text='Font Size', slider=True)
            _timeline, preview_error = _subtitle_preview_timeline(multiplex_obj)
            if preview_error:
                error_row = preview_box.row()
                error_row.alert = True
                error_row.label(text=f'Preview: {preview_error}', icon='ERROR')

        anchor_root = layout.box()
        anchor_root.label(text='Anchor', icon='EMPTY_AXIS')
        anchors = list(getattr(multiplex_obj, 'trlau_multiplex_anchors', []) or [])
        if anchors:
            for fallback_index, anchor in enumerate(anchors):
                anchor_box = anchor_root.box() if len(anchors) > 1 else anchor_root
                if len(anchors) > 1:
                    anchor_box.label(text=f'Anchor {fallback_index + 1}', icon='EMPTY_AXIS')
                data_col = anchor_box.column(align=True)
                data_col.prop(anchor, 'area_id', text='areaID')
                data_col.prop(anchor, 'marker_id', text='markerID')
                data_col.prop(anchor, 'first_channel', text='firstChannel')
        else:
            anchor_count = int(multiplex_obj.get('trlau_mul_anchor_count', 0) or 0)
            if anchor_count <= 0:
                anchor_root.label(text='No anchors stored in this MUL', icon='INFO')
            else:
                for fallback_index in range(anchor_count):
                    prefix = f'trlau_mul_anchor_{fallback_index:03d}'
                    anchor_box = anchor_root.box() if anchor_count > 1 else anchor_root
                    if anchor_count > 1:
                        anchor_box.label(text=f'Anchor {fallback_index + 1}', icon='EMPTY_AXIS')
                    data_col = anchor_box.column(align=True)
                    if f'{prefix}_area_id' in multiplex_obj:
                        data_col.prop(multiplex_obj, f'["{prefix}_area_id"]', text='areaID')
                    if f'{prefix}_marker_id' in multiplex_obj:
                        data_col.prop(multiplex_obj, f'["{prefix}_marker_id"]', text='markerID')
                    if f'{prefix}_first_channel' in multiplex_obj:
                        data_col.prop(multiplex_obj, f'["{prefix}_first_channel"]', text='firstChannel')

class VIEW3D_PT_trlau_multiplex(bpy.types.Panel):
    bl_label = 'Multiplex'
    bl_idname = 'VIEW3D_PT_trlau_multiplex'
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'
    bl_order = 1

    @classmethod
    def poll(cls, context):
        return _find_context_multiplex_object(context) is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False

        multiplex_obj = _find_context_multiplex_object(context)
        if multiplex_obj is None:
            layout.label(text='Select a Multiplex empty', icon='INFO')
            return

        try:
            payload = _read_multiplex_metadata(multiplex_obj)
        except Exception as exc:
            box = layout.box()
            box.label(text='Stored Multiplex metadata could not be read', icon='ERROR')
            box.label(text=str(exc))
            return

        header = layout.box()
        header.label(text=str(payload.get('name', multiplex_obj.name) or multiplex_obj.name), icon='EMPTY_DATA')
        header.label(text=f'Source: {Path(str(payload.get("source", "") or multiplex_obj.name)).name}')
        header.label(text=f'Skeletons: {len(payload.get("skeletons", []) or [])}')

        audio_strip = _find_multiplex_audio_strip(context, multiplex_obj)
        if audio_strip is not None:
            audio_playing = _multiplex_audio_is_playing(context, multiplex_obj)
            audio_col = header.column(align=True)
            audio_row = audio_col.row(align=True)
            audio_op = audio_row.operator(
                TRLAU_OT_toggle_multiplex_audio.bl_idname,
                text='Stop Audio' if audio_playing else 'Play Audio',
                icon='MUTE_IPO_OFF' if audio_playing else 'PLAY',
            )
            audio_op.multiplex_object_name = multiplex_obj.name
            audio_op.play_audio = not audio_playing
            if hasattr(multiplex_obj, 'trlau_mul_audio_volume_ui'):
                audio_col.prop(multiplex_obj, 'trlau_mul_audio_volume_ui', text='Volume', slider=True)
            elif hasattr(audio_strip, 'volume'):
                audio_col.prop(audio_strip, 'volume', text='Volume', slider=True)
        elif str(multiplex_obj.get('trlau_mul_audio_wav_path', '') or '') or str(multiplex_obj.get('trlau_mul_audio_strip_name', '') or ''):
            header.label(text='Linked audio strip not found', icon='ERROR')
        else:
            header.label(text='No MUL audio track linked', icon='INFO')

        skeletons = list(payload.get('skeletons', []) or [])
        if not skeletons:
            empty_box = layout.box()
            empty_box.label(text='No skeleton tracks stored in this MUL', icon='INFO')
            return

        list_box = layout.box()
        top_row = list_box.row(align=True)
        any_assigned = False
        for fallback_index, skeleton in enumerate(skeletons):
            skeleton_index = int(skeleton.get('index', fallback_index) or fallback_index)
            binding = _get_multiplex_binding(multiplex_obj, skeleton_index, create=False)
            armature_ref = getattr(binding, 'target_armature', None) if binding is not None else None
            if armature_ref is not None and getattr(armature_ref, 'type', None) == 'ARMATURE':
                any_assigned = True
                break
        top_row.enabled = any_assigned
        all_op = top_row.operator(TRLAU_OT_load_all_multiplex_animations.bl_idname, text='Load All Animations', icon='ARMATURE_DATA')
        all_op.multiplex_object_name = multiplex_obj.name
        all_op.preserve_bone_positions = bool(getattr(multiplex_obj, 'trlau_mul_preserve_bone_positions', False))

        retarget_row = list_box.row(align=True)
        retarget_row.prop(multiplex_obj, 'trlau_mul_preserve_bone_positions', text='Retarget Bone Positions')

        list_col = list_box.column(align=True)
        for fallback_index, skeleton in enumerate(skeletons):
            skeleton_index = int(skeleton.get('index', fallback_index) or fallback_index)
            binding = _get_multiplex_binding(multiplex_obj, skeleton_index, create=False)
            row = list_col.row(align=True)
            row.label(text=_skeleton_summary_label(skeleton), icon='ANIM_DATA')
            if binding is not None:
                row.prop(binding, 'target_armature', text='')
                armature_ref = getattr(binding, 'target_armature', None)
            else:
                row.label(text='Binding missing', icon='ERROR')
                armature_ref = None
            button_col = row.column(align=True)
            button_col.enabled = armature_ref is not None and getattr(armature_ref, 'type', None) == 'ARMATURE'
            op = button_col.operator(TRLAU_OT_apply_multiplex_skeleton_to_armature.bl_idname, text='Load Animation')
            op.multiplex_object_name = multiplex_obj.name
            op.skeleton_index = skeleton_index
            op.target_armature_name = armature_ref.name if armature_ref is not None else ''
            op.preserve_bone_positions = bool(getattr(multiplex_obj, 'trlau_mul_preserve_bone_positions', False))
