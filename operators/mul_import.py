from __future__ import annotations

import base64
import math
import pickle
import struct
import zlib
import tempfile
import wave
from io import BytesIO
from pathlib import Path

import bpy
import mathutils

from ..core.blender_mesh_utils import armature_bone_count
from ..core.log import logger
from .animation_types import TRLAUBoneAnimation, TRLAUKeyframe, TRLAUTrack, TRLAUTransformAnimation, TRLAUTransformType


MUL_PACKET_HEADER_SIZE = 0x10
MUL_NINTENDO_PACKET_HEADER_SIZE = 0x20
MUL_STREAM_START_OFFSET = 0x800
MUL_PACKET_TYPE_SOUND = 0
MUL_PACKET_TYPE_CINEMATIC = 1
MUL_FRAME_HEADER_SIZE = 0x8
MUL_ALIGNMENT = 0x10
MUL_CAMERA_CHANNEL_COUNT = 16
MUL_CAMERA_SCALE_OFFSET = 0
MUL_CAMERA_ROTATION_OFFSET = 3
MUL_CAMERA_LOCATION_OFFSET = 6
MUL_CAMERA_FOV_OFFSET = 9
MUL_CAMERA_FLAGS_OFFSET = 14
MUL_CAMERA_LEGACY_FLAGS_OFFSET = 10
MUL_SKELETON_ROOT_CHANNEL_COUNT = 3
MUL_BONE_CHANNEL_STRIDE = 14
MUL_BONE_SCALE_OFFSET = 0
MUL_BONE_ROTATION_OFFSET = 3
MUL_BONE_LOCATION_OFFSET = 6
MUL_BONE_FLAGS_OFFSET = 9
MUL_FOV_MIN_RADIANS = math.radians(1.0)
MUL_FOV_MAX_RADIANS = math.radians(179.0)
MUL_AUDIO_BYTES_PER_BLOCK = 36
MUL_AUDIO_SAMPLES_PER_BLOCK = 64
MUL_DSP_AUDIO_BYTES_PER_FRAME = 8
MUL_DSP_AUDIO_SAMPLES_PER_FRAME = 14
MUL_DSP_CHANNEL_HEADER_SIZE = 0x2E
MUL_DSP_CHANNEL_HEADERS_OFFSET = 0xCC
MUL_AUDIO_MAX_CHANNELS = 12
MUL_CAMERA_VISUAL_SCALE = 1.0
MUL_CAMERA_DISPLAY_SIZE = 100.0
MUL_MULTIPLEX_STORAGE_MAGIC = 'TRLAU_MUL_MULTIPLEX_STORAGE_V2'
MUL_MULTIPLEX_STORAGE_LINE_LENGTH = 120
MUL_MULTIPLEX_OBJECT_CHUNK_LENGTH = 60000
MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX = 'trlau_mul_data_chunk_'
MUL_MULTIPLEX_OBJECT_CHUNK_COUNT = 'trlau_mul_data_chunk_count'
MUL_BONE_FLAGS_PROP = 'trlau_mul_bone_flags'


class MultiplexStreamImporterMixin:

    @staticmethod
    def _align_mul_offset(offset: int) -> int:
        return (offset + (MUL_ALIGNMENT - 1)) & ~(MUL_ALIGNMENT - 1)

    @staticmethod
    def _align_mul_offset_to(offset: int, alignment: int) -> int:
        if alignment <= 1:
            return offset
        return (offset + (alignment - 1)) & ~(alignment - 1)

    @staticmethod
    def _mul_detect_platform(data: bytes) -> dict:
        def packet_is_valid(endian: str, packet_header_size: int, packet_alignment: int) -> bool:
            if len(data) < MUL_STREAM_START_OFFSET + packet_header_size:
                return False
            try:
                packet_type, packet_size = struct.unpack_from(endian + 'ii', data, MUL_STREAM_START_OFFSET)
            except Exception:
                return False
            if packet_type not in (MUL_PACKET_TYPE_SOUND, MUL_PACKET_TYPE_CINEMATIC):
                return False
            if packet_size < 0 or MUL_STREAM_START_OFFSET + packet_header_size + packet_size > len(data):
                return False
            if packet_type == MUL_PACKET_TYPE_CINEMATIC:
                magic_offset = MUL_STREAM_START_OFFSET + packet_header_size
                magic = data[magic_offset:magic_offset + 4]
                if endian == '<' and magic != b'ENIC':
                    return False
                if endian == '>' and magic != b'CINE':
                    return False
            return True

        try:
            be_sample_rate = struct.unpack_from('>i', data, 0)[0]
        except Exception:
            be_sample_rate = 0
        try:
            le_sample_rate = struct.unpack_from('<i', data, 0)[0]
        except Exception:
            le_sample_rate = 0

        if packet_is_valid('>', MUL_NINTENDO_PACKET_HEADER_SIZE, 0x20) and 8000 <= be_sample_rate <= 96000:
            return {
                'platform': 'nintendo',
                'endian': '>',
                'packet_header_size': MUL_NINTENDO_PACKET_HEADER_SIZE,
                'packet_alignment': 0x20,
                'sound_header_size': 0x20,
                'audio_codec': 'gc_dsp_adpcm',
            }

        if packet_is_valid('>', MUL_PACKET_HEADER_SIZE, MUL_ALIGNMENT) and 8000 <= be_sample_rate <= 96000:
            return {
                'platform': 'xbox',
                'endian': '>',
                'packet_header_size': MUL_PACKET_HEADER_SIZE,
                'packet_alignment': MUL_ALIGNMENT,
                'sound_header_size': 0x10,
                'audio_codec': 'xbox_lau_adpcm',
            }

        if packet_is_valid('<', MUL_PACKET_HEADER_SIZE, MUL_ALIGNMENT) and 8000 <= le_sample_rate <= 96000:
            return {
                'platform': 'pc',
                'endian': '<',
                'packet_header_size': MUL_PACKET_HEADER_SIZE,
                'packet_alignment': MUL_ALIGNMENT,
                'sound_header_size': 0x10,
                'audio_codec': 'lau_adpcm',
            }

        # Default to the original little-endian PC/PS2-style path so older files fail in
        # the same place they used to, rather than being misdetected as Nintendo data.
        return {
            'platform': 'pc',
            'endian': '<',
            'packet_header_size': MUL_PACKET_HEADER_SIZE,
            'packet_alignment': MUL_ALIGNMENT,
            'sound_header_size': 0x10,
            'audio_codec': 'lau_adpcm',
        }

    def _read_mul_cine_header_legacy(self, data: bytes, offset: int, endian: str = '<') -> tuple[dict, int]:
        """Read the older/guessed MUL CINE header layout.

        TRU PC CINE packets use a different object descriptor layout; the public
        method below detects that format first and falls back here only when the
        descriptor scanner cannot find a usable TRU-style header.
        """
        stream = BytesIO(data)
        stream.seek(offset)

        def read_i32() -> int:
            raw = stream.read(4)
            if len(raw) < 4:
                raise ValueError('Unexpected end of file while reading .mul cinematic header')
            return struct.unpack(endian + 'i', raw)[0]

        def read_floats(count: int) -> tuple[float, ...]:
            raw = stream.read(count * 4)
            if len(raw) < count * 4:
                raise ValueError('Unexpected end of file while reading .mul cinematic header')
            return struct.unpack(endian + ('f' * count), raw)

        magic = stream.read(4)
        version = read_i32()
        header_size = read_i32()
        name = stream.read(0x40).split(b'\x00', 1)[0].decode('ascii', errors='ignore')
        main_unit_id = read_i32()

        skeletons = []
        cameras = []
        channel_count = 0

        anchors = []
        num_anchors = read_i32()
        for anchor_index in range(num_anchors):
            area_id = read_i32()
            marker_id = read_i32()
            first_channel = read_i32()
            anchors.append({
                'area_id': area_id,
                'marker_id': marker_id,
                'first_channel': first_channel,
            })
            channel_count += 9

        num_skeletons = read_i32()
        for skeleton_index in range(num_skeletons):
            instance_id = read_i32()
            num_bones = read_i32()
            first_channel = read_i32()
            default_bone_transforms = [read_floats(16) for _ in range(num_bones)]
            skeletons.append({
                'index': skeleton_index,
                'instance_id': instance_id,
                'num_bones': num_bones,
                'first_channel': first_channel,
                'default_bone_transforms': default_bone_transforms,
                'root_channel_count': 2,
                'bone_channel_stride': 10,
            })
            channel_count += 2 + (num_bones * 10)

        num_cameras = read_i32()
        for camera_index in range(num_cameras):
            first_channel = read_i32()
            cameras.append({
                'index': camera_index,
                'first_channel': first_channel,
                'channel_count': 11,
            })
            channel_count += 11

        trigger_unit_id = read_i32()
        num_triggers = read_i32()
        for _ in range(num_triggers):
            stream.read(8)
            channel_count += 1

        num_subtitles = read_i32()

        skeleton_end_transforms = []
        for _ in range(num_skeletons):
            end_position = read_floats(3)
            end_rotation = read_floats(3)
            final_unit_id = read_i32()
            skeleton_end_transforms.append({
                'end_position': end_position,
                'end_rotation': end_rotation,
                'final_unit_id': final_unit_id,
            })

        # Some original Xbox Legend MUL CINE packets use the older header layout but
        # store one more camera channel than the original PC/PS2 estimate.  Frame 0
        # initializes every channel, but frame_size can also include internal zero
        # padding.  Do not treat padding as another channel; validate inferred extra
        # channels by walking the frame-0 run/value pairs and requiring positive run
        # lengths.
        try:
            frame_offset = stream.tell()
            if frame_offset + MUL_FRAME_HEADER_SIZE <= len(data):
                first_frame_size, first_frame_number = struct.unpack_from(endian + 'ii', data, frame_offset)
                frame_record_end = frame_offset + int(first_frame_size)
                if first_frame_number >= 0 and first_frame_size >= MUL_FRAME_HEADER_SIZE and frame_record_end <= len(data):
                    max_initial_channel_count = (int(first_frame_size) - MUL_FRAME_HEADER_SIZE) // 8

                    def frame0_candidate_is_valid(candidate_count: int) -> bool:
                        value_offset = frame_offset + MUL_FRAME_HEADER_SIZE
                        if candidate_count < 0:
                            return False
                        for _channel_index in range(int(candidate_count)):
                            if value_offset + 8 > frame_record_end:
                                return False
                            channel_run = struct.unpack_from(endian + 'I', data, value_offset)[0]
                            run_length = int(channel_run & 0x0FFFFFFF)
                            if run_length <= 0:
                                return False
                            value_offset += 8
                        remaining = frame_record_end - value_offset
                        if remaining < 0 or remaining >= MUL_ALIGNMENT:
                            return False
                        return bytes(data[value_offset:frame_record_end]).strip(b'\x00') == b''

                    initial_channel_count = int(channel_count)
                    for candidate_count in range(int(channel_count) + 1, int(max_initial_channel_count) + 1):
                        if candidate_count - int(channel_count) > 32:
                            break
                        if frame0_candidate_is_valid(candidate_count):
                            initial_channel_count = int(candidate_count)
                        else:
                            break

                    missing_camera_channels = int(initial_channel_count) - int(channel_count)
                    if cameras and 0 < missing_camera_channels <= 32:
                        cameras[-1]['channel_count'] = int(cameras[-1].get('channel_count', 11)) + missing_camera_channels
                        channel_count = int(initial_channel_count)
        except Exception:
            pass

        header = {
            'magic': magic,
            'endian': endian,
            'version': version,
            'header_size': header_size,
            'name': name,
            'main_unit_id': main_unit_id,
            'anchors': anchors,
            'num_subtitles': num_subtitles,
            'trigger_unit_id': trigger_unit_id,
            'skeletons': skeletons,
            'cameras': cameras,
            'skeleton_end_transforms': skeleton_end_transforms,
            'channel_count': channel_count,
            'cine_layout': 'legacy',
        }
        return header, stream.tell()

    def _read_mul_cine_header(self, data: bytes, offset: int, endian: str = '<') -> tuple[dict, int]:
        """Read a MUL CINE header and return the first CineFrame offset.

        Tomb Raider: Underworld PC stores CINE object descriptors as:

            i32 instance_id, i32 parent_id, i32 bone_count, i32 first_channel,
            followed by bone_count 4x4 default matrices.

        Each object uses three root channels and fourteen channels per bone.  The
        first frame initializes every channel, so its payload size is a reliable
        channel-count source even when the header contains unknown channel blocks
        between objects.
        """

        def read_i32_at(pos: int) -> int:
            return struct.unpack_from(endian + 'i', data, pos)[0]

        def read_u32_at(pos: int) -> int:
            return struct.unpack_from(endian + 'I', data, pos)[0]

        def read_floats_at(pos: int, count: int) -> tuple[float, ...]:
            return struct.unpack_from(endian + ('f' * int(count)), data, pos)

        if offset + 0x4C > len(data):
            raise ValueError('Unexpected end of file while reading .mul cinematic header')

        magic = data[offset:offset + 4]
        version = read_i32_at(offset + 4)
        header_size = read_i32_at(offset + 8)
        name = data[offset + 0x0C:offset + 0x0C + 0x40].split(b'\x00', 1)[0].decode('ascii', errors='ignore')

        # In TRU PC files the stored CINE header size excludes the first 8 bytes
        # (magic + version).  Older layouts may use a different convention, so keep
        # two conservative candidates and choose the first one that looks like a
        # CineFrame header.
        frame_offset_candidates = [offset + int(header_size) + 8, offset + int(header_size)]
        frame_offset = 0
        first_frame_size = 0
        first_frame_number = 0
        for candidate in frame_offset_candidates:
            if candidate + MUL_FRAME_HEADER_SIZE > len(data):
                continue
            try:
                candidate_size, candidate_number = struct.unpack_from(endian + 'ii', data, candidate)
            except Exception:
                continue
            if candidate_size >= MUL_FRAME_HEADER_SIZE and candidate + candidate_size <= len(data):
                frame_offset = int(candidate)
                first_frame_size = int(candidate_size)
                first_frame_number = int(candidate_number)
                break

        if frame_offset <= 0:
            return self._read_mul_cine_header_legacy(data, offset, endian=endian)

        first_frame_payload_size = max(0, int(first_frame_size) - MUL_FRAME_HEADER_SIZE)
        # Frame 0 contains an initial run word and an initial float value for every
        # channel.  A few TRU files keep a 4-byte subtitle/pad word inside the frame;
        # integer division intentionally ignores that pad.
        initial_channel_count = int(first_frame_payload_size // 8) if first_frame_number >= 0 else 0

        # Legend/Anniversary PC MULs use the older version-5 CINE header layout even
        # though they still have a normal ENIC packet and a valid frame offset.  The
        # TRU descriptor scanner below can accept false descriptors inside default
        # matrices; the usual bad symptom is instance id 0x3F800000 (float 1.0) and
        # no camera descriptor.  Prefer the explicit legacy layout for version 5 and
        # only fall back to the scanner if that parse fails.
        if int(version) <= 5:
            try:
                return self._read_mul_cine_header_legacy(data, offset, endian=endian)
            except Exception as exc:
                logger.debug('Structured legacy MUL CINE parse failed, trying descriptor scanner: %s', exc)

        scan_start = offset + 0x0C + 0x40
        header_scan_end = max(scan_start, frame_offset)

        main_unit_id = -1
        anchors = []
        try:
            pos = scan_start
            main_unit_id = read_i32_at(pos)
            pos += 4
            num_anchors = read_i32_at(pos)
            pos += 4
            if 0 <= num_anchors <= 32:
                for _ in range(num_anchors):
                    if pos + 16 > header_scan_end:
                        anchors = []
                        break
                    anchor_unit_id, marker_id, first_channel, channel_count_hint = struct.unpack_from(endian + '4i', data, pos)
                    pos += 16
                    anchors.append({
                        'area_id': int(anchor_unit_id),
                        'marker_id': int(marker_id),
                        'first_channel': int(first_channel),
                        'channel_count_hint': int(channel_count_hint),
                    })
        except Exception:
            main_unit_id = -1
            anchors = []

        descriptor_candidates: list[tuple[int, int, int, int, int, int, int]] = []
        for descriptor_offset in range(scan_start, max(scan_start, header_scan_end - 16) + 1, 4):
            try:
                instance_id, parent_id, num_bones, first_channel = struct.unpack_from(endian + '4i', data, descriptor_offset)
            except Exception:
                continue
            if int(parent_id) != -1:
                continue
            if int(num_bones) <= 0 or int(num_bones) > 512:
                continue
            if int(first_channel) < 0:
                continue
            if initial_channel_count > 0 and int(first_channel) >= initial_channel_count:
                continue

            matrix_start = descriptor_offset + 16
            matrix_end = matrix_start + (int(num_bones) * 0x40)
            next_channel = int(first_channel) + MUL_SKELETON_ROOT_CHANNEL_COUNT + (int(num_bones) * MUL_BONE_CHANNEL_STRIDE)
            if matrix_end > header_scan_end:
                continue
            if initial_channel_count > 0 and next_channel > initial_channel_count + 128:
                continue

            try:
                probe_matrix = read_floats_at(matrix_start, 16)
                if not all(math.isfinite(float(value)) for value in probe_matrix):
                    continue
            except Exception:
                continue

            descriptor_candidates.append((
                int(descriptor_offset),
                int(instance_id),
                int(parent_id),
                int(num_bones),
                int(first_channel),
                int(matrix_end),
                int(next_channel),
            ))

        skeletons = []
        last_matrix_end = scan_start
        for skeleton_index, candidate in enumerate(descriptor_candidates):
            descriptor_offset, instance_id, parent_id, num_bones, first_channel, matrix_end, next_channel = candidate
            if descriptor_offset < last_matrix_end:
                continue
            default_bone_transforms = [
                read_floats_at(descriptor_offset + 16 + (bone_index * 0x40), 16)
                for bone_index in range(num_bones)
            ]
            skeletons.append({
                'index': int(len(skeletons)),
                'instance_id': int(instance_id),
                'parent_id': int(parent_id),
                'num_bones': int(num_bones),
                'first_channel': int(first_channel),
                'next_channel': int(next_channel),
                'descriptor_offset': int(descriptor_offset - offset),
                'default_bone_transforms': default_bone_transforms,
                'root_channel_count': MUL_SKELETON_ROOT_CHANNEL_COUNT,
                'bone_channel_stride': MUL_BONE_CHANNEL_STRIDE,
            })
            last_matrix_end = int(matrix_end)

        if not skeletons:
            return self._read_mul_cine_header_legacy(data, offset, endian=endian)

        cursor = last_matrix_end
        try:
            if cursor + 16 <= header_scan_end:
                zero_0, next_0, zero_1, next_1 = struct.unpack_from(endian + '4i', data, cursor)
                if zero_0 == 0 and zero_1 == 0 and next_0 == next_1 and 0 <= next_0 <= max(initial_channel_count, next_0):
                    cursor += 16
        except Exception:
            pass

        cameras = []
        try:
            if cursor + 4 <= header_scan_end:
                num_cameras = read_i32_at(cursor)
                if 0 <= num_cameras <= 16 and cursor + 4 + (num_cameras * 4) <= header_scan_end:
                    camera_first_channels = [read_i32_at(cursor + 4 + (camera_index * 4)) for camera_index in range(num_cameras)]
                    valid_camera_channels = all(0 <= int(first_channel) < max(initial_channel_count, 1) for first_channel in camera_first_channels)
                    if valid_camera_channels:
                        for camera_index, first_channel in enumerate(camera_first_channels):
                            if camera_index + 1 < len(camera_first_channels):
                                camera_channel_count = max(0, int(camera_first_channels[camera_index + 1]) - int(first_channel))
                            elif initial_channel_count > 0:
                                camera_channel_count = max(0, int(initial_channel_count) - int(first_channel))
                            else:
                                camera_channel_count = MUL_CAMERA_CHANNEL_COUNT
                            cameras.append({
                                'index': int(camera_index),
                                'first_channel': int(first_channel),
                                'channel_count': int(camera_channel_count),
                            })
                        cursor += 4 + (num_cameras * 4)
        except Exception:
            cameras = []

        skeleton_end_transforms = []
        end_transform_cursor = cursor
        for _ in skeletons:
            if end_transform_cursor + 28 > header_scan_end:
                break
            try:
                final_unit_id = read_i32_at(end_transform_cursor)
                end_position = read_floats_at(end_transform_cursor + 4, 3)
                end_rotation = read_floats_at(end_transform_cursor + 16, 3)
            except Exception:
                break
            skeleton_end_transforms.append({
                'end_position': tuple(float(value) for value in end_position),
                'end_rotation': tuple(float(value) for value in end_rotation),
                'final_unit_id': int(final_unit_id),
            })
            end_transform_cursor += 28

        channel_count = int(initial_channel_count)
        if channel_count <= 0:
            channel_count = 0
            for anchor in anchors:
                channel_count = max(channel_count, int(anchor.get('first_channel', 0)) + 9)
            for skeleton in skeletons:
                channel_count = max(channel_count, int(skeleton.get('next_channel', 0)))
            for camera in cameras:
                channel_count = max(channel_count, int(camera.get('first_channel', 0)) + int(camera.get('channel_count', MUL_CAMERA_CHANNEL_COUNT)))

        header = {
            'magic': magic,
            'endian': endian,
            'version': int(version),
            'header_size': int(header_size),
            'name': name,
            'main_unit_id': int(main_unit_id),
            'anchors': anchors,
            'num_subtitles': 0,
            'trigger_unit_id': int(main_unit_id),
            'skeletons': skeletons,
            'cameras': cameras,
            'skeleton_end_transforms': skeleton_end_transforms,
            'channel_count': int(channel_count),
            'cine_layout': 'tru_pc',
        }
        return header, frame_offset

    def _parse_multiplexstream_file(self, filepath: str, data: bytes | None = None) -> tuple[dict, list[dict], list[dict]]:
        path = Path(filepath)
        if data is None:
            data = path.read_bytes()
        platform_info = self._mul_detect_platform(data)
        endian = str(platform_info.get('endian', '<'))
        packet_header_size = int(platform_info.get('packet_header_size', MUL_PACKET_HEADER_SIZE))
        packet_alignment = int(platform_info.get('packet_alignment', MUL_ALIGNMENT))
        if len(data) < MUL_STREAM_START_OFFSET + packet_header_size:
            raise ValueError('The .mul file is too small to contain a valid cinematic stream')

        logger.info(
            'Starting multiplexstream parse: %s (%d bytes, platform=%s endian=%s)',
            path.name,
            len(data),
            platform_info.get('platform', 'pc'),
            endian,
        )

        packet_offset = MUL_STREAM_START_OFFSET
        channel_values: list[float] | None = None
        channel_run_lengths: list[int] | None = None
        channel_run_types: list[int] | None = None
        cine_header: dict | None = None
        skeleton_frames: list[dict] | None = None
        camera_frames: list[dict] | None = None
        packet_index = 0
        total_frames_parsed = 0

        while packet_offset + packet_header_size <= len(data):
            packet_type, packet_size = struct.unpack_from(endian + 'ii', data, packet_offset)
            logger.debug('MUL packet %d at 0x%X: type=%d size=%d', packet_index, packet_offset, packet_type, packet_size)
            packet_data_start = packet_offset + packet_header_size
            packet_data_end = packet_data_start + max(packet_size, 0)
            if packet_data_end > len(data):
                break

            if packet_type != MUL_PACKET_TYPE_CINEMATIC:
                logger.debug(
                    'Skipping non-cinematic MUL packet %d at 0x%X (type=%d, size=%d)',
                    packet_index,
                    packet_offset,
                    packet_type,
                    packet_size,
                )
                packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)
                packet_index += 1
                continue

            frame_offset = packet_data_start
            if cine_header is None:
                cine_header, frame_offset = self._read_mul_cine_header(data, packet_data_start, endian=endian)
                cine_header['platform'] = platform_info.get('platform', 'pc')
                cine_header['packet_header_size'] = packet_header_size
                cine_header['packet_alignment'] = packet_alignment
                cine_header['sound_header_size'] = int(platform_info.get('sound_header_size', 0x10))
                cine_header['audio_codec'] = platform_info.get('audio_codec', 'lau_adpcm')
                logger.info(
                    'MUL cine header: name=%s version=%s anchors=%d skeletons=%d cameras=%d channels=%d main_unit_id=%d',
                    cine_header.get('name', ''),
                    cine_header.get('version', -1),
                    len(cine_header.get('anchors', [])),
                    len(cine_header.get('skeletons', [])),
                    len(cine_header.get('cameras', [])),
                    int(cine_header.get('channel_count', 0)),
                    int(cine_header.get('main_unit_id', -1)),
                )
                for anchor_log_index, anchor_info in enumerate(cine_header.get('anchors', [])):
                    logger.info(
                        'MUL anchor %d: area_id=%d marker_id=%d first_channel=%d',
                        int(anchor_log_index),
                        int(anchor_info.get('area_id', -1)),
                        int(anchor_info.get('marker_id', -1)),
                        int(anchor_info.get('first_channel', -1)),
                    )
                for skeleton_info in cine_header.get('skeletons', []):
                    logger.info(
                        'MUL skeleton %d: instance_id=%d bones=%d first_channel=%d',
                        int(skeleton_info.get('index', -1)),
                        int(skeleton_info.get('instance_id', -1)),
                        int(skeleton_info.get('num_bones', 0)),
                        int(skeleton_info.get('first_channel', -1)),
                    )
                for camera_info in cine_header.get('cameras', []):
                    logger.info(
                        'MUL camera %d: first_channel=%d',
                        int(camera_info.get('index', -1)),
                        int(camera_info.get('first_channel', -1)),
                    )
                channel_values = [0.0] * int(cine_header['channel_count'])
                channel_run_lengths = [0] * int(cine_header['channel_count'])
                channel_run_types = [0] * int(cine_header['channel_count'])
                skeleton_frames = [
                    {
                        'index': int(skeleton.get('index', skeleton_index)),
                        'instance_id': int(skeleton['instance_id']),
                        'num_bones': int(skeleton['num_bones']),
                        'first_channel': int(skeleton.get('first_channel', -1)),
                        'next_channel': int(skeleton.get('next_channel', -1)),
                        'root_channel_count': int(skeleton.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT)),
                        'bone_channel_stride': int(skeleton.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE)),
                        'root_frames': [],
                        'frames': [[] for _ in range(int(skeleton['num_bones']))],
                    }
                    for skeleton_index, skeleton in enumerate(cine_header['skeletons'])
                ]
                camera_frames = [
                    {
                        'index': int(camera['index']),
                        'first_channel': int(camera['first_channel']),
                        'channel_count': int(camera.get('channel_count', MUL_CAMERA_CHANNEL_COUNT)),
                        'frames': [],
                    }
                    for camera in cine_header['cameras']
                ]

            while frame_offset + MUL_FRAME_HEADER_SIZE <= packet_data_end:
                frame_size, frame_number = struct.unpack_from(endian + 'ii', data, frame_offset)
                frame_record_end = frame_offset + max(int(frame_size), 0)
                if frame_size < MUL_FRAME_HEADER_SIZE or frame_record_end > packet_data_end:
                    logger.debug(
                        'Stopping MUL frame parse at packet %d offset 0x%X: invalid frame_size=%d packet_end=0x%X',
                        packet_index,
                        frame_offset,
                        frame_size,
                        packet_data_end,
                    )
                    break

                if frame_number < 0:
                    logger.info(
                        'MUL non-animation frame encountered (frame=%d, frame_size=%d, packet=%d, offset=0x%X)',
                        frame_number,
                        frame_size,
                        packet_index,
                        frame_offset,
                    )
                    frame_offset = self._align_mul_offset(frame_record_end)
                    continue

                value_offset = frame_offset + MUL_FRAME_HEADER_SIZE
                assert channel_values is not None and channel_run_lengths is not None and channel_run_types is not None and skeleton_frames is not None and camera_frames is not None and cine_header is not None

                for channel_index in range(len(channel_values)):
                    read_channel_value = False
                    if channel_run_lengths[channel_index] == 0:
                        if value_offset + 4 > frame_record_end:
                            raise ValueError('Unexpected end of .mul channel run data')
                        channel_run = struct.unpack_from(endian + 'I', data, value_offset)[0]
                        value_offset += 4
                        channel_run_lengths[channel_index] = int(channel_run & 0x0FFFFFFF)
                        channel_run_types[channel_index] = int((channel_run >> 28) & 0xF)
                        read_channel_value = True
                    else:
                        read_channel_value = channel_run_types[channel_index] != 0

                    if read_channel_value:
                        if value_offset + 4 > frame_record_end:
                            raise ValueError('Unexpected end of .mul channel value data')
                        channel_values[channel_index] = struct.unpack_from(endian + 'f', data, value_offset)[0]
                        value_offset += 4

                    channel_run_lengths[channel_index] -= 1

                for skeleton_index, skeleton in enumerate(cine_header['skeletons']):
                    skeleton_first_channel = int(skeleton['first_channel'])
                    root_channel_count = int(skeleton.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT) or MUL_SKELETON_ROOT_CHANNEL_COUNT)
                    bone_channel_stride = int(skeleton.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE) or MUL_BONE_CHANNEL_STRIDE)
                    first_channel = skeleton_first_channel + root_channel_count
                    num_bones = int(skeleton['num_bones'])
                    root_values = [
                        float(channel_values[skeleton_first_channel + root_axis])
                        if 0 <= skeleton_first_channel + root_axis < len(channel_values) else 0.0
                        for root_axis in range(root_channel_count)
                    ]
                    skeleton_frames[skeleton_index].setdefault('root_frames', []).append((int(frame_number), *root_values))
                    for bone_index in range(num_bones):
                        channel_base = first_channel + (bone_index * bone_channel_stride)
                        if channel_base + MUL_BONE_FLAGS_OFFSET >= len(channel_values):
                            continue
                        # TRU PC stores 14 channels per bone. The first ten are the
                        # useful transform channels: scale XYZ, Euler rotation XYZ,
                        # location XYZ, and a flags/value channel. The remaining four
                        # channels are currently unknown and are preserved only by the
                        # original stream, not by the editable transform payload.
                        skeleton_frames[skeleton_index]['frames'][bone_index].append((
                            int(frame_number),
                            (
                                float(channel_values[channel_base + MUL_BONE_SCALE_OFFSET + 0]),
                                float(channel_values[channel_base + MUL_BONE_SCALE_OFFSET + 1]),
                                float(channel_values[channel_base + MUL_BONE_SCALE_OFFSET + 2]),
                            ),
                            (
                                float(channel_values[channel_base + MUL_BONE_ROTATION_OFFSET + 0]),
                                float(channel_values[channel_base + MUL_BONE_ROTATION_OFFSET + 1]),
                                float(channel_values[channel_base + MUL_BONE_ROTATION_OFFSET + 2]),
                            ),
                            (
                                float(channel_values[channel_base + MUL_BONE_LOCATION_OFFSET + 0]),
                                float(channel_values[channel_base + MUL_BONE_LOCATION_OFFSET + 1]),
                                float(channel_values[channel_base + MUL_BONE_LOCATION_OFFSET + 2]),
                            ),
                            float(channel_values[channel_base + MUL_BONE_FLAGS_OFFSET]),
                        ))

                for camera_index, camera in enumerate(cine_header.get('cameras', [])):
                    channel_base = int(camera['first_channel'])
                    camera_channel_count = int(camera.get('channel_count', MUL_CAMERA_CHANNEL_COUNT) or MUL_CAMERA_CHANNEL_COUNT)
                    if channel_base + MUL_CAMERA_FOV_OFFSET >= len(channel_values):
                        continue
                    scale = tuple(float(channel_values[channel_base + MUL_CAMERA_SCALE_OFFSET + axis]) for axis in range(3))
                    rotation = tuple(float(channel_values[channel_base + MUL_CAMERA_ROTATION_OFFSET + axis]) for axis in range(3))
                    location = tuple(float(channel_values[channel_base + MUL_CAMERA_LOCATION_OFFSET + axis]) for axis in range(3))
                    fov = float(channel_values[channel_base + MUL_CAMERA_FOV_OFFSET])
                    flags_offset = MUL_CAMERA_FLAGS_OFFSET if camera_channel_count > MUL_CAMERA_FLAGS_OFFSET and channel_base + MUL_CAMERA_FLAGS_OFFSET < len(channel_values) else MUL_CAMERA_LEGACY_FLAGS_OFFSET
                    flags = int(round(channel_values[channel_base + flags_offset])) if channel_base + flags_offset < len(channel_values) else 0
                    camera_frames[camera_index]['frames'].append({
                        'frame': int(frame_number),
                        'scale': scale,
                        'rotation': rotation,
                        'location': location,
                        'fov': fov,
                        'flags': flags,
                    })

                total_frames_parsed += 1
                if total_frames_parsed <= 5 or total_frames_parsed % 50 == 0:
                    logger.info(
                        'MUL frame %d parsed (global_frame_index=%d, frame_size=%d, packet=%d)',
                        frame_number,
                        total_frames_parsed,
                        frame_size,
                        packet_index,
                    )

                frame_offset = self._align_mul_offset(frame_record_end)

            packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)
            packet_index += 1

        if cine_header is None or skeleton_frames is None or camera_frames is None:
            raise ValueError('No cinematic packet with header was found in the .mul file')

        logger.info('Finished multiplexstream parse: packets=%d frames=%d', packet_index, total_frames_parsed)
        return cine_header, skeleton_frames, camera_frames


    @staticmethod
    def _mul_skeleton_final_frame(skeleton_frame_info: dict) -> int:
        final_frame = 0
        for bone_frames in skeleton_frame_info.get('frames', []) or []:
            if bone_frames:
                try:
                    last = bone_frames[-1]
                    if isinstance(last, dict):
                        frame = int(last.get('frame', 0))
                    elif isinstance(last, (list, tuple)) and last:
                        frame = int(last[0])
                    else:
                        frame = 0
                    final_frame = max(final_frame, frame)
                except Exception:
                    pass
        return final_frame

    @staticmethod
    def _compact_multiplex_bone_frames(bone_frames: list[list[dict]]) -> list[list[tuple]]:
        compact_bones = []
        for frames in bone_frames or []:
            compact_frames = []
            for frame_info in frames or []:
                if isinstance(frame_info, (list, tuple)) and len(frame_info) >= 4:
                    frame, scale, rotation, location = frame_info[:4]
                    flags = float(frame_info[4]) if len(frame_info) >= 5 else 5.0
                    compact_frames.append((
                        int(frame),
                        tuple(float(v) for v in scale),
                        tuple(float(v) for v in rotation),
                        tuple(float(v) for v in location),
                        flags,
                    ))
                else:
                    compact_frames.append((
                        int(frame_info.get('frame', 0)),
                        tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0))),
                        tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0))),
                        tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0))),
                        float(frame_info.get('flags', 5.0)),
                    ))
            compact_bones.append(compact_frames)
        return compact_bones

    @staticmethod
    def _expand_multiplex_bone_frames(stored_bone_frames: list) -> list[list[dict]]:
        expanded_bones = []
        for frames in stored_bone_frames or []:
            expanded_frames = []
            for frame_info in frames or []:
                if isinstance(frame_info, dict):
                    expanded_frames.append(frame_info)
                    continue
                if not isinstance(frame_info, (list, tuple)) or len(frame_info) < 4:
                    continue
                frame, scale, rotation, location = frame_info[:4]
                flags = float(frame_info[4]) if len(frame_info) >= 5 else 5.0
                expanded_frames.append({
                    'frame': int(frame),
                    'scale': tuple(float(v) for v in scale),
                    'rotation': tuple(float(v) for v in rotation),
                    'location': tuple(float(v) for v in location),
                    'flags': flags,
                })
            expanded_bones.append(expanded_frames)
        return expanded_bones

    def _build_multiplex_metadata_payload(
        self,
        filepath: str,
        cine_header: dict,
        skeleton_frames: list[dict],
        camera_frames: list[dict],
    ) -> dict:
        header_skeletons = list(cine_header.get('skeletons', []) or [])
        stored_skeletons = []
        for skeleton_index, skeleton in enumerate(skeleton_frames or []):
            header_info = header_skeletons[skeleton_index] if skeleton_index < len(header_skeletons) else {}
            stored_skeletons.append({
                'index': int(skeleton.get('index', skeleton_index)),
                'instance_id': int(skeleton.get('instance_id', header_info.get('instance_id', -1))),
                'num_bones': int(skeleton.get('num_bones', header_info.get('num_bones', 0))),
                'first_channel': int(header_info.get('first_channel', -1)),
                'next_channel': int(header_info.get('next_channel', -1)),
                'root_channel_count': int(header_info.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT)),
                'bone_channel_stride': int(header_info.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE)),
                'final_frame': self._mul_skeleton_final_frame(skeleton),
            })

        stored_cameras = []
        header_cameras = list(cine_header.get('cameras', []) or [])
        for camera_index, camera in enumerate(camera_frames or []):
            header_info = header_cameras[camera_index] if camera_index < len(header_cameras) else {}
            frames = list(camera.get('frames', []) or [])
            stored_cameras.append({
                'index': int(camera.get('index', header_info.get('index', camera_index))),
                'first_channel': int(camera.get('first_channel', header_info.get('first_channel', -1))),
                'channel_count': int(header_info.get('channel_count', camera.get('channel_count', MUL_CAMERA_CHANNEL_COUNT))),
                'frame_count': len(frames),
                'final_frame': int(frames[-1].get('frame', 0)) if frames else 0,
            })

        stored_anchors = []
        for anchor in cine_header.get('anchors', []) or []:
            stored_anchors.append({
                'area_id': int(anchor.get('area_id', -1)),
                'marker_id': int(anchor.get('marker_id', -1)),
                'first_channel': int(anchor.get('first_channel', -1)),
            })

        return {
            'schema': 2,
            'source': str(filepath),
            'platform': str(cine_header.get('platform', 'pc') or 'pc'),
            'endian': str(cine_header.get('endian', '<') or '<'),
            'packet_header_size': int(cine_header.get('packet_header_size', MUL_PACKET_HEADER_SIZE) or MUL_PACKET_HEADER_SIZE),
            'packet_alignment': int(cine_header.get('packet_alignment', MUL_ALIGNMENT) or MUL_ALIGNMENT),
            'sound_header_size': int(cine_header.get('sound_header_size', 0x10) or 0x10),
            'audio_codec': str(cine_header.get('audio_codec', 'lau_adpcm') or 'lau_adpcm'),
            'platform': str(cine_header.get('platform', 'pc') or 'pc'),
            'endian': str(cine_header.get('endian', '<') or '<'),
            'packet_header_size': int(cine_header.get('packet_header_size', MUL_PACKET_HEADER_SIZE) or MUL_PACKET_HEADER_SIZE),
            'packet_alignment': int(cine_header.get('packet_alignment', MUL_ALIGNMENT) or MUL_ALIGNMENT),
            'sound_header_size': int(cine_header.get('sound_header_size', 0x10) or 0x10),
            'audio_codec': str(cine_header.get('audio_codec', 'lau_adpcm') or 'lau_adpcm'),
            'name': str(cine_header.get('name', Path(filepath).stem) or Path(filepath).stem),
            'main_unit_id': int(cine_header.get('main_unit_id', -1)),
            'anchors': stored_anchors,
            'trigger_unit_id': int(cine_header.get('trigger_unit_id', -1)),
            'num_subtitles': int(cine_header.get('num_subtitles', 0)),
            'skeletons': stored_skeletons,
            'cameras': stored_cameras,
        }

    def _build_multiplex_storage_payload(
        self,
        filepath: str,
        cine_header: dict,
        skeleton_frames: list[dict],
        camera_frames: list[dict],
    ) -> dict:
        # Keep this payload in Python-native form and serialize it with pickle.
        # The previous implementation expanded every frame into JSON, which can
        # become very large and makes Blender appear to hang immediately after
        # the packet parser finishes. The panel reads lightweight metadata from
        # the Empty custom properties; this heavy payload is decoded only when a
        # skeleton is actually applied to an armature.
        header_skeletons = []
        for skeleton in cine_header.get('skeletons', []) or []:
            header_skeletons.append({
                'index': int(skeleton.get('index', len(header_skeletons))),
                'instance_id': int(skeleton.get('instance_id', -1)),
                'num_bones': int(skeleton.get('num_bones', 0)),
                'first_channel': int(skeleton.get('first_channel', -1)),
                'next_channel': int(skeleton.get('next_channel', -1)),
                'root_channel_count': int(skeleton.get('root_channel_count', MUL_SKELETON_ROOT_CHANNEL_COUNT)),
                'bone_channel_stride': int(skeleton.get('bone_channel_stride', MUL_BONE_CHANNEL_STRIDE)),
                'default_bone_transforms': skeleton.get('default_bone_transforms', []),
            })

        compact_skeleton_frames = []
        for skeleton_index, skeleton in enumerate(skeleton_frames or []):
            compact_skeleton_frames.append({
                'index': int(skeleton.get('index', skeleton_index)),
                'instance_id': int(skeleton.get('instance_id', -1)),
                'num_bones': int(skeleton.get('num_bones', 0)),
                'root_frames': [
                    (
                        int(frame_info[0]),
                        *(float(value) for value in frame_info[1:]),
                    ) if isinstance(frame_info, (list, tuple)) and len(frame_info) >= 2 else (
                        int(frame_info.get('frame', 0)),
                        float(frame_info.get('channel_0', -1.0)),
                        float(frame_info.get('root_matrix_index', 1.0)),
                        float(frame_info.get('root_channel_2', 0.0)),
                    )
                    for frame_info in list(skeleton.get('root_frames', []) or [])
                ],
                'frames': self._compact_multiplex_bone_frames(skeleton.get('frames', []) or []),
            })

        return {
            'schema': 2,
            'source': str(filepath),
            'name': str(cine_header.get('name', Path(filepath).stem) or Path(filepath).stem),
            'main_unit_id': int(cine_header.get('main_unit_id', -1)),
            'anchors': [
                {
                    'area_id': int(anchor.get('area_id', -1)),
                    'marker_id': int(anchor.get('marker_id', -1)),
                    'first_channel': int(anchor.get('first_channel', -1)),
                }
                for anchor in cine_header.get('anchors', []) or []
            ],
            'trigger_unit_id': int(cine_header.get('trigger_unit_id', -1)),
            'num_subtitles': int(cine_header.get('num_subtitles', 0)),
            'skeleton_headers': header_skeletons,
            'skeleton_frames': compact_skeleton_frames,
            'camera_frames': [
                {
                    'index': int(camera.get('index', camera_index)),
                    'first_channel': int(camera.get('first_channel', -1)),
                    'channel_count': int(camera.get('channel_count', MUL_CAMERA_CHANNEL_COUNT)),
                    'frames': [
                        (
                            int(frame_info.get('frame', 0)),
                            tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0))),
                            tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0))),
                            tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0))),
                            float(frame_info.get('fov', 0.0)),
                            int(frame_info.get('flags', 0)),
                        )
                        for frame_info in list(camera.get('frames', []) or [])
                    ],
                }
                for camera_index, camera in enumerate(camera_frames or [])
            ],
        }

    @staticmethod
    def _encode_multiplex_storage_payload(payload: dict) -> str:
        # Keep import responsive. The payload is already hidden on the Multiplex
        # Empty, so a lower zlib level and no cosmetic line wrapping are better
        # tradeoffs than maximum compression during import.
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(raw, level=1)
        encoded = base64.b64encode(compressed).decode('ascii')
        return MUL_MULTIPLEX_STORAGE_MAGIC + '\n' + encoded

    @staticmethod
    def _decode_multiplex_storage_payload(raw: str) -> dict:
        if not raw.startswith(MUL_MULTIPLEX_STORAGE_MAGIC):
            raise ValueError('Unsupported Multiplex storage payload format')
        encoded = ''.join(raw.splitlines()[1:]).strip()
        if not encoded:
            raise ValueError('Multiplex storage payload is empty')
        compressed = base64.b64decode(encoded.encode('ascii'))
        payload = pickle.loads(zlib.decompress(compressed))
        if not isinstance(payload, dict):
            raise ValueError('Multiplex storage payload is not a dictionary')
        return payload

    @staticmethod
    def _clear_multiplex_storage_chunks(obj: bpy.types.Object) -> None:
        if obj is None:
            return
        try:
            existing_count = int(obj.get(MUL_MULTIPLEX_OBJECT_CHUNK_COUNT, 0) or 0)
        except Exception:
            existing_count = 0
        keys_to_delete = []
        try:
            for key in list(obj.keys()):
                if str(key).startswith(MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX):
                    keys_to_delete.append(str(key))
        except Exception:
            keys_to_delete = []
        for index in range(existing_count):
            keys_to_delete.append(f'{MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX}{index:04d}')
        for key in sorted(set(keys_to_delete)):
            try:
                del obj[key]
            except Exception:
                pass
        try:
            if MUL_MULTIPLEX_OBJECT_CHUNK_COUNT in obj:
                del obj[MUL_MULTIPLEX_OBJECT_CHUNK_COUNT]
        except Exception:
            pass

    @staticmethod
    def _write_multiplex_storage_payload_to_object(obj: bpy.types.Object, encoded_payload: str) -> None:
        if obj is None:
            raise ValueError('Cannot store Multiplex payload without an object')
        encoded_payload = str(encoded_payload or '')
        MultiplexStreamImporterMixin._clear_multiplex_storage_chunks(obj)
        chunks = [
            encoded_payload[index:index + MUL_MULTIPLEX_OBJECT_CHUNK_LENGTH]
            for index in range(0, len(encoded_payload), MUL_MULTIPLEX_OBJECT_CHUNK_LENGTH)
        ]
        obj[MUL_MULTIPLEX_OBJECT_CHUNK_COUNT] = int(len(chunks))
        for index, chunk in enumerate(chunks):
            obj[f'{MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX}{index:04d}'] = chunk

    @staticmethod
    def _read_multiplex_storage_payload_from_object(obj: bpy.types.Object) -> str:
        if obj is None:
            return ''
        try:
            count = int(obj.get(MUL_MULTIPLEX_OBJECT_CHUNK_COUNT, 0) or 0)
        except Exception:
            count = 0
        if count <= 0:
            return ''
        chunks = []
        for index in range(count):
            key = f'{MUL_MULTIPLEX_OBJECT_CHUNK_PREFIX}{index:04d}'
            value = str(obj.get(key, '') or '')
            if not value:
                raise ValueError(f'Multiplex payload chunk is missing: {key}')
            chunks.append(value)
        return ''.join(chunks)

    @staticmethod
    def _unique_object_name(base_name: str) -> str:
        name = base_name
        suffix = 1
        while name in bpy.data.objects:
            name = f'{base_name}.{suffix:03d}'
            suffix += 1
        return name

    def _create_multiplex_data_empty(
        self,
        context,
        filepath: str,
        cine_header: dict,
        skeleton_frames: list[dict],
        camera_frames: list[dict],
        audio_result: dict | None = None,
        collection: bpy.types.Collection | None = None,
    ) -> bpy.types.Object:
        path = Path(filepath)
        metadata = self._build_multiplex_metadata_payload(filepath, cine_header, skeleton_frames, camera_frames)
        storage_payload = self._build_multiplex_storage_payload(filepath, cine_header, skeleton_frames, camera_frames)

        logger.info(
            'Encoding MUL Multiplex payload: skeletons=%d cameras=%d',
            len(skeleton_frames or []),
            len(camera_frames or []),
        )
        encoded_payload = self._encode_multiplex_storage_payload(storage_payload)

        empty_name = self._unique_object_name(f'{path.stem}_Multiplex')
        empty = bpy.data.objects.new(empty_name, None)
        empty.empty_display_type = 'CUBE'
        empty.empty_display_size = 1.0
        # Multiplex identity and editable header fields are inferred from the stored MUL payload.
        # Do not add visible duplicate custom properties on newly imported empties.

        if hasattr(empty, 'trlau_multiplex_anchors'):
            anchors = empty.trlau_multiplex_anchors
            try:
                anchors.clear()
            except Exception:
                pass
            for index, anchor in enumerate(metadata.get('anchors', []) or []):
                anchor_row = anchors.add()
                anchor_row.area_id = int(anchor.get('area_id', -1))
                anchor_row.marker_id = int(anchor.get('marker_id', -1))
                anchor_row.first_channel = int(anchor.get('first_channel', -1))

        if audio_result is not None:
            strip = audio_result.get('audio_strip')
            if strip is not None:
                try:
                    strip.mute = True
                    strip['trlau_mul_audio_enabled'] = False
                    strip['trlau_mul_multiplex_object'] = empty.name
                except Exception:
                    pass
                empty['trlau_mul_audio_strip_name'] = str(getattr(strip, 'name', '') or '')
                empty['trlau_mul_audio_enabled'] = False
            empty['trlau_mul_audio_wav_path'] = str(audio_result.get('audio_wav_path', '') or '')

        # Pre-create the per-skeleton UI binding rows while importing. Do not
        # create them lazily from the panel draw callback: Blender forbids
        # writing to ID data while UI panels are drawing.
        if hasattr(empty, 'trlau_multiplex_skeleton_bindings'):
            bindings = empty.trlau_multiplex_skeleton_bindings
            try:
                bindings.clear()
            except Exception:
                pass
            for index, skeleton in enumerate(metadata.get('skeletons', []) or []):
                binding = bindings.add()
                binding.skeleton_index = int(skeleton.get('index', index))

        logger.info('Writing MUL Multiplex payload object chunks: chars=%d', len(encoded_payload))
        self._write_multiplex_storage_payload_to_object(empty, encoded_payload)

        target_collection = collection or context.collection or context.scene.collection
        target_collection.objects.link(empty)
        try:
            empty.select_set(True)
            context.view_layer.objects.active = empty
        except Exception:
            pass

        logger.info(
            'Created MUL Multiplex data empty %s with %d skeleton(s), %d camera(s), final_frame=%d',
            empty.name,
            len(metadata.get('skeletons', []) or []),
            len(metadata.get('cameras', []) or []),
            max([0] + [int(item.get('final_frame', 0) or 0) for item in list(metadata.get('skeletons', []) or []) + list(metadata.get('cameras', []) or [])]),
        )
        return empty

    def _frames_to_bone_animation(self, bone_frames: list[dict]) -> TRLAUBoneAnimation:
        bone_anim = TRLAUBoneAnimation()
        transform_names = ('rotation', 'scale', 'location')
        for transform_type, key in enumerate(transform_names):
            transform_anim = TRLAUTransformAnimation()
            has_any_axis = False
            for axis in range(3):
                keyframes = [TRLAUKeyframe(int(frame_info['frame']), float(frame_info[key][axis])) for frame_info in bone_frames]
                if keyframes:
                    transform_anim.axis_tracks[axis] = TRLAUTrack(keyframes)
                    has_any_axis = True
            if has_any_axis:
                bone_anim.transforms[transform_type] = transform_anim
        return bone_anim


    @staticmethod
    def _mul_default_transform_to_matrix(transform_values) -> mathutils.Matrix:
        if not transform_values or len(transform_values) != 16:
            return mathutils.Matrix.Identity(4)
        try:
            rows = [tuple(float(transform_values[(row_index * 4) + column_index]) for column_index in range(4)) for row_index in range(4)]
            return mathutils.Matrix(rows)
        except Exception:
            return mathutils.Matrix.Identity(4)

    @staticmethod
    def _mul_orientation_only_matrix(transform_values: mathutils.Matrix | None) -> mathutils.Matrix:
        if transform_values is None:
            return mathutils.Matrix.Identity(4)
        result = transform_values.copy()
        result[0][3] = 0.0
        result[1][3] = 0.0
        result[2][3] = 0.0
        return result

    @staticmethod
    def _mul_rotation_vector_to_matrix(rotation_vector: mathutils.Vector) -> mathutils.Matrix:
        angle = float(rotation_vector.length)
        if angle < 1e-8:
            return mathutils.Matrix.Identity(4)
        try:
            return mathutils.Quaternion(rotation_vector.normalized(), angle).to_matrix().to_4x4()
        except Exception:
            return mathutils.Matrix.Identity(4)


    @staticmethod
    def _mul_euler_xyz_to_matrix(rotation_values) -> mathutils.Matrix:
        try:
            values = tuple(float(v) for v in rotation_values)
            return mathutils.Euler(values, 'XYZ').to_matrix().to_4x4()
        except Exception:
            return mathutils.Matrix.Identity(4)

    @staticmethod
    def _mul_build_source_matrix(
        frame_info: dict,
        default_matrix: mathutils.Matrix | None = None,
        parent_default_matrix: mathutils.Matrix | None = None,
    ) -> mathutils.Matrix:
        location = mathutils.Vector(tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0))))
        rotation_values = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
        scale_values = tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0)))

        # MUL default transforms are stored as default-pose/global bone
        # orientations, not local parent-relative bone deltas. Convert them to a
        # local orientation before composing the source bone matrix:
        #
        #   local_default = inverse(parent_default_global) @ bone_default_global
        #
        # The channel translation is stored in the parent's default/global basis,
        # so convert it back to Blender/rest local space with inverse(parent D).
        #
        # The channel rotation is still best applied before the local default
        # orientation, as in v8. The previous v8 build interpreted the XYZ triple
        # as a single exponential-map vector. That is identical to Euler XYZ for
        # one-axis bones, but it introduces small unwanted cross-axis rotations
        # when two or three axes are keyed. Treat the triple as Euler XYZ, then
        # conjugate the resulting rotation matrix out of the parent's default
        # basis. This keeps the v8 arm/default behavior but removes the residual
        # multi-axis drift.
        parent_default_orientation = MultiplexStreamImporterMixin._mul_orientation_only_matrix(parent_default_matrix)
        parent_default_orientation_3x3 = parent_default_orientation.to_3x3()
        inverse_parent_default_orientation_3x3 = parent_default_orientation_3x3.inverted_safe()

        location = inverse_parent_default_orientation_3x3 @ location

        default_orientation = MultiplexStreamImporterMixin._mul_orientation_only_matrix(default_matrix)
        local_default_orientation = parent_default_orientation.inverted_safe() @ default_orientation

        channel_rotation = MultiplexStreamImporterMixin._mul_euler_xyz_to_matrix(rotation_values)
        source_rotation = (
            parent_default_orientation.inverted_safe()
            @ channel_rotation
            @ parent_default_orientation
        )

        return (
            mathutils.Matrix.Translation(location)
            @ source_rotation
            @ local_default_orientation
            @ mathutils.Matrix.Diagonal((float(scale_values[0]), float(scale_values[1]), float(scale_values[2]), 1.0))
        )

    @staticmethod
    def _mul_source_to_pose_basis(
        pose_bone: bpy.types.PoseBone,
        source_matrix: mathutils.Matrix,
        parent_source_matrix: mathutils.Matrix | None = None,
    ) -> mathutils.Matrix:
        rest_matrix = pose_bone.bone.matrix_local.copy()
        if pose_bone.parent is None:
            return rest_matrix.inverted_safe() @ source_matrix

        parent_rest_matrix = pose_bone.parent.bone.matrix_local.copy()
        rest_local_matrix = parent_rest_matrix.inverted_safe() @ rest_matrix
        parent_pose_matrix = parent_source_matrix if parent_source_matrix is not None else parent_rest_matrix
        return rest_local_matrix.inverted_safe() @ parent_pose_matrix.inverted_safe() @ source_matrix

    @staticmethod
    def _mul_iter_related_objects(root_obj: bpy.types.Object | None):
        if root_obj is None:
            return
        stack = [root_obj]
        seen = set()
        while stack:
            obj = stack.pop()
            if obj is None or obj.name in seen:
                continue
            seen.add(obj.name)
            yield obj
            parent = getattr(obj, 'parent', None)
            if parent is not None and parent.name not in seen:
                stack.append(parent)
            for child in getattr(obj, 'children', ()):
                if child.name not in seen:
                    stack.append(child)

    def _get_mul_candidate_instance_id(self, armature: bpy.types.Object) -> int | None:
        for obj in self._mul_iter_related_objects(armature):
            if 'trlau_bginstance_id' in obj:
                try:
                    return int(obj['trlau_bginstance_id'])
                except Exception:
                    continue
        return None

    @staticmethod
    def _armature_bone_count(armature: bpy.types.Object) -> int:
        return armature_bone_count(armature)


    @staticmethod
    def _clear_fcurve_keyframes(curve) -> None:
        points = curve.keyframe_points
        try:
            points.clear()
            return
        except Exception:
            pass
        try:
            while points:
                points.remove(points[-1], fast=True)
        except Exception:
            while len(points):
                points.remove(points[-1])

    @staticmethod
    def _bulk_write_fcurve_keyframes(curve, frames, values, tolerance: float = 1e-6) -> None:
        if not frames or not values:
            return

        frame_values = [(float(frame), float(value)) for frame, value in zip(frames, values)]
        if not frame_values:
            return

        # Reduce truly static curves to one key. This keeps imported actions much
        # lighter without changing evaluated playback for constant channels.
        first_value = frame_values[0][1]
        if len(frame_values) > 1 and all(abs(value - first_value) <= tolerance for _, value in frame_values[1:]):
            frame_values = [frame_values[0]]

        MultiplexStreamImporterMixin._clear_fcurve_keyframes(curve)
        points = curve.keyframe_points
        count = len(frame_values)
        try:
            points.add(count)
            flat = []
            for frame, value in frame_values:
                flat.extend((frame, value))
            points.foreach_set('co', flat)
        except Exception:
            # Fallback for older Blender builds/API edge cases. This path is
            # slower, but it keeps the importer functional if bulk F-curve writes
            # are unavailable.
            MultiplexStreamImporterMixin._clear_fcurve_keyframes(curve)
            for frame, value in frame_values:
                point = points.insert(frame=frame, value=value)
                point.interpolation = 'LINEAR'
        else:
            for point in points:
                point.interpolation = 'LINEAR'
        try:
            curve.update()
        except Exception:
            pass

    def _create_mul_bone_rotation_curves(
        self,
        action: bpy.types.Action,
        armature_obj: bpy.types.Object,
        pose_bone: bpy.types.PoseBone,
        bone_frames: list[dict],
    ) -> None:
        pose_bone.rotation_mode = 'QUATERNION'
        data_path = f'pose.bones["{pose_bone.name}"].rotation_quaternion'
        curves = [self._ensure_action_fcurve(action, armature_obj, data_path, index) for index in range(4)]

        if pose_bone.parent:
            basis = (pose_bone.parent.bone.matrix_local @ pose_bone.bone.matrix_local.inverted()).to_3x3()
        else:
            basis = pose_bone.bone.matrix_local.inverted().to_3x3()

        frames: list[int] = []
        values = [[], [], [], []]
        previous_quaternion = None
        for frame_info in bone_frames:
            frame = int(frame_info.get('frame', 0))
            rotation_values = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
            source_rotation = mathutils.Euler(rotation_values, 'XYZ').to_matrix()
            quaternion = (basis @ source_rotation @ basis.inverted()).to_quaternion()
            if previous_quaternion is not None and previous_quaternion.dot(quaternion) < 0.0:
                quaternion.negate()
            previous_quaternion = quaternion.copy()
            frames.append(frame)
            for curve_index, value in enumerate((quaternion.w, quaternion.x, quaternion.y, quaternion.z)):
                values[curve_index].append(float(value))

        for curve_index, curve in enumerate(curves):
            self._bulk_write_fcurve_keyframes(curve, frames, values[curve_index])

    def _create_mul_baked_bone_curves(
        self,
        action: bpy.types.Action,
        armature_obj: bpy.types.Object,
        bone_frames: list[list[dict]],
        header_skeleton_info: dict | None,
        preserve_bone_positions: bool = False,
        keep_root_motion: bool = True,
    ) -> None:
        default_transforms = list((header_skeleton_info or {}).get('default_bone_transforms', []) or [])
        pose_bones_by_name = {pose_bone.name: pose_bone for pose_bone in armature_obj.pose.bones}
        pose_bones_by_index: dict[int, bpy.types.PoseBone] = {}
        mapped_pose_bone_names: set[str] = set()

        for bone_index in range(min(len(bone_frames), len(armature_obj.pose.bones))):
            pose_bone = pose_bones_by_name.get(f'bone_{bone_index}')
            if pose_bone is None and bone_index < len(armature_obj.pose.bones):
                pose_bone = armature_obj.pose.bones[bone_index]
            if pose_bone is not None and pose_bone.name not in mapped_pose_bone_names:
                pose_bones_by_index[bone_index] = pose_bone
                mapped_pose_bone_names.add(pose_bone.name)

        if not pose_bones_by_index:
            return

        pose_bone_index_by_name = {pose_bone.name: bone_index for bone_index, pose_bone in pose_bones_by_index.items()}

        def bone_depth(pose_bone: bpy.types.PoseBone) -> int:
            depth = 0
            parent = pose_bone.parent
            while parent is not None:
                depth += 1
                parent = parent.parent
            return depth

        bone_order = sorted(pose_bones_by_index.keys(), key=lambda index: bone_depth(pose_bones_by_index[index]))
        frame_lookup_by_bone: list[dict[int, dict]] = []
        all_frames: set[int] = set()

        for frames in bone_frames:
            frame_lookup: dict[int, dict] = {}
            for frame_info in frames:
                frame = int(frame_info.get('frame', 0))
                frame_lookup[frame] = frame_info
                all_frames.add(frame)
            frame_lookup_by_bone.append(frame_lookup)

        if not all_frames:
            return

        identity_matrix = mathutils.Matrix.Identity(4)
        default_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        parent_default_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        parent_index_by_bone: dict[int, int | None] = {}
        parent_rest_matrix_by_bone: dict[int, mathutils.Matrix] = {}
        rest_inverse_by_bone: dict[int, mathutils.Matrix] = {}
        rest_local_inverse_by_bone: dict[int, mathutils.Matrix] = {}
        build_source_cache_by_bone: dict[int, dict] = {}

        for bone_index in bone_order:
            pose_bone = pose_bones_by_index[bone_index]
            parent_index = pose_bone_index_by_name.get(pose_bone.parent.name) if pose_bone.parent is not None else None
            parent_index_by_bone[bone_index] = parent_index

            if bone_index < len(default_transforms):
                default_matrix = self._mul_default_transform_to_matrix(default_transforms[bone_index])
            else:
                default_matrix = identity_matrix.copy()
            default_matrix_by_bone[bone_index] = default_matrix

            if parent_index is not None and parent_index < len(default_transforms):
                parent_default_matrix = self._mul_default_transform_to_matrix(default_transforms[parent_index])
            else:
                parent_default_matrix = identity_matrix.copy()
            parent_default_matrix_by_bone[bone_index] = parent_default_matrix

            rest_matrix = pose_bone.bone.matrix_local.copy()
            if pose_bone.parent is None:
                rest_inverse_by_bone[bone_index] = rest_matrix.inverted_safe()
            else:
                parent_rest_matrix = pose_bone.parent.bone.matrix_local.copy()
                parent_rest_matrix_by_bone[bone_index] = parent_rest_matrix
                rest_local_matrix = parent_rest_matrix.inverted_safe() @ rest_matrix
                rest_local_inverse_by_bone[bone_index] = rest_local_matrix.inverted_safe()

            parent_default_orientation = self._mul_orientation_only_matrix(parent_default_matrix)
            parent_default_orientation_inv = parent_default_orientation.inverted_safe()
            default_orientation = self._mul_orientation_only_matrix(default_matrix)
            build_source_cache_by_bone[bone_index] = {
                'parent_default_orientation': parent_default_orientation,
                'parent_default_orientation_inv': parent_default_orientation_inv,
                'parent_default_orientation_3x3_inv': parent_default_orientation.to_3x3().inverted_safe(),
                'local_default_orientation': parent_default_orientation_inv @ default_orientation,
            }

        def build_source_matrix_fast(bone_index: int, frame_info: dict) -> mathutils.Matrix:
            cache = build_source_cache_by_bone[bone_index]
            location = mathutils.Vector(tuple(float(v) for v in frame_info.get('location', (0.0, 0.0, 0.0))))
            location = cache['parent_default_orientation_3x3_inv'] @ location
            rotation_values = tuple(float(v) for v in frame_info.get('rotation', (0.0, 0.0, 0.0)))
            scale_values = tuple(float(v) for v in frame_info.get('scale', (1.0, 1.0, 1.0)))
            channel_rotation = self._mul_euler_xyz_to_matrix(rotation_values)
            source_rotation = (
                cache['parent_default_orientation_inv']
                @ channel_rotation
                @ cache['parent_default_orientation']
            )
            return (
                mathutils.Matrix.Translation(location)
                @ source_rotation
                @ cache['local_default_orientation']
                @ mathutils.Matrix.Diagonal((float(scale_values[0]), float(scale_values[1]), float(scale_values[2]), 1.0))
            )

        def source_to_pose_basis_fast(
            bone_index: int,
            source_matrix: mathutils.Matrix,
            parent_source_matrix: mathutils.Matrix | None,
        ) -> mathutils.Matrix:
            if parent_index_by_bone.get(bone_index) is None:
                return rest_inverse_by_bone[bone_index] @ source_matrix
            parent_pose_matrix = parent_source_matrix if parent_source_matrix is not None else parent_rest_matrix_by_bone[bone_index]
            return rest_local_inverse_by_bone[bone_index] @ parent_pose_matrix.inverted_safe() @ source_matrix

        previous_quaternion_by_bone: dict[int, mathutils.Quaternion] = {}
        retarget_reference_location_by_bone: dict[int, mathutils.Vector] = {}
        retarget_location_scale_by_bone: dict[int, float] = {}
        sampled_values_by_bone: dict[int, dict] = {
            bone_index: {
                'frames': [],
                'location': [[], [], []],
                'rotation': [[], [], [], []],
                'scale': [[], [], []],
                'flags': [],
            }
            for bone_index in bone_order
        }

        def get_retarget_location_scale(
            bone_index: int,
            pose_bone: bpy.types.PoseBone,
            default_matrix: mathutils.Matrix,
            parent_default_matrix: mathutils.Matrix,
        ) -> float:
            cached = retarget_location_scale_by_bone.get(bone_index)
            if cached is not None:
                return cached

            scale = 1.0
            if pose_bone.parent is not None:
                try:
                    target_parent_rest = pose_bone.parent.bone.matrix_local.copy()
                    target_rest = pose_bone.bone.matrix_local.copy()
                    target_local_offset = (target_parent_rest.inverted_safe() @ target_rest).to_translation()
                    target_length = float(target_local_offset.length)

                    source_local_offset = (parent_default_matrix.inverted_safe() @ default_matrix).to_translation()
                    source_length = float(source_local_offset.length)

                    if source_length > 1e-6 and target_length > 1e-6:
                        scale = target_length / source_length
                except Exception:
                    scale = 1.0

            retarget_location_scale_by_bone[bone_index] = float(scale)
            return float(scale)

        for frame in sorted(all_frames):
            source_matrix_by_bone: dict[int, mathutils.Matrix] = {}

            for bone_index in bone_order:
                if bone_index >= len(frame_lookup_by_bone):
                    continue

                frame_info = frame_lookup_by_bone[bone_index].get(frame)
                if frame_info is None:
                    continue

                pose_bone = pose_bones_by_index.get(bone_index)
                if pose_bone is None:
                    continue

                parent_index = parent_index_by_bone.get(bone_index)
                source_local_matrix = build_source_matrix_fast(bone_index, frame_info)
                parent_source_matrix = source_matrix_by_bone.get(parent_index) if parent_index is not None else None

                if parent_source_matrix is not None:
                    source_matrix = parent_source_matrix @ source_local_matrix
                else:
                    source_matrix = source_local_matrix

                source_matrix_by_bone[bone_index] = source_matrix
                pose_basis = source_to_pose_basis_fast(bone_index, source_matrix, parent_source_matrix)
                location, quaternion, scale = pose_basis.decompose()

                previous_quaternion = previous_quaternion_by_bone.get(bone_index)
                if previous_quaternion is not None and previous_quaternion.dot(quaternion) < 0.0:
                    quaternion.negate()
                previous_quaternion_by_bone[bone_index] = quaternion.copy()

                if preserve_bone_positions:
                    # Retarget/proportions mode: keep root motion and import
                    # child locations as deltas around the target armature's
                    # own rest layout.  This preserves useful translation keys
                    # without forcing the source skeleton's bone lengths onto
                    # the selected armature.
                    if pose_bone.parent is None:
                        write_location = location
                    else:
                        reference_location = retarget_reference_location_by_bone.get(bone_index)
                        if reference_location is None:
                            reference_location = location.copy()
                            retarget_reference_location_by_bone[bone_index] = reference_location
                        location_scale = get_retarget_location_scale(
                            bone_index,
                            pose_bone,
                            default_matrix_by_bone[bone_index],
                            parent_default_matrix_by_bone[bone_index],
                        )
                        write_location = (location - reference_location) * location_scale
                    write_scale = (1.0, 1.0, 1.0)
                else:
                    write_location = location
                    write_scale = scale

                samples = sampled_values_by_bone[bone_index]
                samples['frames'].append(frame)
                for axis, value in enumerate(write_location):
                    samples['location'][axis].append(float(value))
                for axis, value in enumerate((quaternion.w, quaternion.x, quaternion.y, quaternion.z)):
                    samples['rotation'][axis].append(float(value))
                for axis, value in enumerate(write_scale):
                    samples['scale'][axis].append(float(value))
                samples['flags'].append(float(frame_info.get('flags', 5.0)))

        for bone_index in bone_order:
            samples = sampled_values_by_bone.get(bone_index)
            if not samples or not samples['frames']:
                continue
            pose_bone = pose_bones_by_index[bone_index]
            pose_bone.rotation_mode = 'QUATERNION'
            location_path = f'pose.bones["{pose_bone.name}"].location'
            rotation_path = f'pose.bones["{pose_bone.name}"].rotation_quaternion'
            scale_path = f'pose.bones["{pose_bone.name}"].scale'

            for axis in range(3):
                curve = self._ensure_action_fcurve(action, armature_obj, location_path, axis)
                self._bulk_write_fcurve_keyframes(curve, samples['frames'], samples['location'][axis])
            for axis in range(4):
                curve = self._ensure_action_fcurve(action, armature_obj, rotation_path, axis)
                self._bulk_write_fcurve_keyframes(curve, samples['frames'], samples['rotation'][axis])
            for axis in range(3):
                curve = self._ensure_action_fcurve(action, armature_obj, scale_path, axis)
                self._bulk_write_fcurve_keyframes(curve, samples['frames'], samples['scale'][axis])

            flags = [float(int(round(value))) for value in samples.get('flags', [])]
            if flags:
                try:
                    pose_bone[MUL_BONE_FLAGS_PROP] = int(round(flags[0]))
                except Exception:
                    pose_bone[MUL_BONE_FLAGS_PROP] = 5

                flags_path = f'pose.bones["{pose_bone.name}"]["{MUL_BONE_FLAGS_PROP}"]'
                curve = self._ensure_action_fcurve(action, armature_obj, flags_path, 0)
                self._bulk_write_fcurve_keyframes(curve, samples['frames'], flags, tolerance=0.0)
                try:
                    for point in curve.keyframe_points:
                        point.interpolation = 'CONSTANT'
                    curve.update()
                except Exception:
                    pass


    def _build_action_from_mul_skeleton_frames(
        self,
        context,
        armature: bpy.types.Object,
        base_name: str,
        skeleton_frame_info: dict,
        header_skeleton_info: dict | None,
        final_frame: int,
        time_per_frame: int,
        extra_properties: dict | None = None,
        assign_to_armature: bool = True,
        preserve_bone_positions: bool = False,
        keep_root_motion: bool = True,
    ) -> bpy.types.Action:
        action = bpy.data.actions.new(name=self._get_unique_action_name(base_name))
        for key, value in (extra_properties or {}).items():
            action[key] = value

        if assign_to_armature:
            armature.animation_data_create()
            armature.animation_data.action = action
            if hasattr(action, 'slots') and action.slots and hasattr(armature.animation_data, 'action_slot'):
                armature.animation_data.action_slot = action.slots[0]

        self._reset_pose_bones(armature)

        bone_frames = list(skeleton_frame_info.get('frames', []) or [])
        if not bone_frames:
            return action

        self._create_mul_baked_bone_curves(
            action,
            armature,
            bone_frames,
            header_skeleton_info,
            preserve_bone_positions=bool(preserve_bone_positions),
            keep_root_motion=bool(keep_root_motion),
        )

        context.scene.frame_start = 0
        context.scene.frame_end = max(int(context.scene.frame_end), int(final_frame))
        context.scene.frame_current = 0
        return action

    def _collect_candidate_armatures(self, context) -> list[bpy.types.Object]:
        candidates = []
        seen_names = set()

        def add_candidate(obj):
            if obj is None or getattr(obj, 'type', None) != 'ARMATURE':
                return
            if obj.name in seen_names:
                return
            seen_names.add(obj.name)
            candidates.append(obj)

        add_candidate(context.active_object)
        for obj in context.selected_objects:
            add_candidate(obj)
        for obj in bpy.data.objects:
            add_candidate(obj)
        return candidates

    def _match_mul_skeletons_to_armatures(self, context, skeleton_frames: list[dict]) -> dict[int, bpy.types.Object | None]:
        candidates = self._collect_candidate_armatures(context)
        logger.info('Matching %d MUL skeleton(s) against %d armature candidate(s)', len(skeleton_frames), len(candidates))
        for armature in candidates:
            logger.info(
                'Armature candidate: %s bginstance_id=%s bones=%s',
                armature.name,
                self._get_mul_candidate_instance_id(armature),
                self._armature_bone_count(armature),
            )
        assignments: dict[int, bpy.types.Object | None] = {}
        used_armatures: set[str] = set()

        logger.info('Building actions for %d MUL skeleton(s)', len(skeleton_frames))
        for skeleton_index, skeleton in enumerate(skeleton_frames):
            num_bones = int(skeleton['num_bones'])
            instance_id = int(skeleton.get('instance_id', -1))
            matched = None

            for armature in candidates:
                if armature.name in used_armatures:
                    continue
                if self._get_mul_candidate_instance_id(armature) == instance_id and self._armature_bone_count(armature) == num_bones:
                    matched = armature
                    break

            if matched is None:
                for armature in candidates:
                    if armature.name in used_armatures:
                        continue
                    if self._armature_bone_count(armature) == num_bones:
                        matched = armature
                        break

            if matched is None and skeleton_index < len(candidates):
                candidate = candidates[skeleton_index]
                if candidate.name not in used_armatures:
                    matched = candidate

            if matched is None and len(candidates) == 1:
                matched = candidates[0]

            assignments[skeleton_index] = matched
            logger.info(
                'Skeleton %d assignment: instance_id=%d bones=%d -> %s',
                skeleton_index,
                instance_id,
                num_bones,
                matched.name if matched is not None else '<unassigned>',
            )
            if matched is not None:
                used_armatures.add(matched.name)

        return assignments

    @staticmethod
    def _mul_fov_to_blender_angle(raw_fov: float) -> float:
        if not math.isfinite(raw_fov) or raw_fov <= 0.0:
            return math.radians(50.0)

        return min(max(raw_fov, MUL_FOV_MIN_RADIANS), MUL_FOV_MAX_RADIANS)

    def _mul_fov_to_blender_lens(self, camera_data: bpy.types.Camera, raw_fov: float) -> float:
        angle = self._mul_fov_to_blender_angle(raw_fov)
        sensor_fit = getattr(camera_data, 'sensor_fit', 'AUTO')
        if sensor_fit == 'VERTICAL':
            sensor_size = float(getattr(camera_data, 'sensor_height', 24.0))
        else:
            sensor_size = float(getattr(camera_data, 'sensor_width', 36.0))

        tan_half_angle = math.tan(angle * 0.5)
        if not math.isfinite(tan_half_angle) or tan_half_angle <= 0.0:
            return float(getattr(camera_data, 'lens', 50.0))

        return sensor_size / (2.0 * tan_half_angle)

    def _create_object_vector_curves(
        self,
        action: bpy.types.Action,
        obj: bpy.types.ID,
        data_path: str,
        frames: list[dict],
        key: str,
        value_scale: float = 1.0,
    ) -> None:
        frame_numbers = [int(frame_info['frame']) for frame_info in frames]
        for axis in range(3):
            curve = self._ensure_action_fcurve(action, obj, data_path, axis)
            values = [float(frame_info[key][axis]) * float(value_scale) for frame_info in frames]
            self._bulk_write_fcurve_keyframes(curve, frame_numbers, values)

    def _create_camera_fov_curves(self, action: bpy.types.Action, camera_obj: bpy.types.Object, frames: list[dict]) -> None:
        camera_data = camera_obj.data
        curve = self._ensure_action_fcurve(action, camera_data, 'lens', 0)
        frame_numbers = [int(frame_info['frame']) for frame_info in frames]
        values = [self._mul_fov_to_blender_lens(camera_data, float(frame_info.get('fov', 0.0))) for frame_info in frames]
        self._bulk_write_fcurve_keyframes(curve, frame_numbers, values)

    def _create_camera_flags_curves(self, action: bpy.types.Action, camera_obj: bpy.types.Object, frames: list[dict]) -> None:
        curve = self._ensure_action_fcurve(action, camera_obj, '["trlau_mul_camera_flags"]', 0)
        frame_numbers = [int(frame_info['frame']) for frame_info in frames]
        values = [float(int(frame_info.get('flags', 0))) for frame_info in frames]
        self._bulk_write_fcurve_keyframes(curve, frame_numbers, values)
        try:
            for point in curve.keyframe_points:
                point.interpolation = 'CONSTANT'
            curve.update()
        except Exception:
            pass

    def _build_action_from_camera_frames(
        self,
        context,
        camera_obj: bpy.types.Object,
        base_name: str,
        camera_frame_info: dict,
        time_per_frame: int,
        extra_properties: dict | None = None,
    ) -> tuple[bpy.types.Action | None, bpy.types.Action | None, int]:
        frames = list(camera_frame_info.get('frames', []))
        if not frames:
            return None, None, 0

        final_frame = int(frames[-1]['frame'])
        object_action = bpy.data.actions.new(name=self._get_unique_action_name(base_name))
        if extra_properties:
            for key, value in extra_properties.items():
                object_action[key] = value

        camera_obj.animation_data_create()
        camera_obj.animation_data.action = object_action
        if hasattr(object_action, 'slots') and len(object_action.slots) > 0 and hasattr(camera_obj.animation_data, 'action_slot'):
            slot = next((slot for slot in object_action.slots if getattr(slot, 'id_root', None) == 'OBJECT'), None)
            if slot is not None:
                camera_obj.animation_data.action_slot = slot

        camera_data = getattr(camera_obj, 'data', None)
        camera_action = None
        if camera_data is not None:
            camera_action = bpy.data.actions.new(name=self._get_unique_action_name(f'{base_name}_data'))
            if extra_properties:
                for key, value in extra_properties.items():
                    camera_action[key] = value

            camera_data.animation_data_create()
            camera_data.animation_data.action = camera_action
            if hasattr(camera_action, 'slots') and len(camera_action.slots) > 0 and hasattr(camera_data.animation_data, 'action_slot'):
                slot = next((slot for slot in camera_action.slots if getattr(slot, 'id_root', None) == 'CAMERA'), None)
                if slot is not None:
                    camera_data.animation_data.action_slot = slot

        camera_obj.rotation_mode = 'XYZ'
        camera_obj.location = tuple(float(v) for v in frames[0].get('location', (0.0, 0.0, 0.0)))
        camera_obj.rotation_euler = tuple(float(v) for v in frames[0].get('rotation', (0.0, 0.0, 0.0)))
        camera_obj.scale = tuple(float(v) * MUL_CAMERA_VISUAL_SCALE for v in frames[0].get('scale', (1.0, 1.0, 1.0)))
        camera_obj['trlau_mul_camera_flags'] = int(frames[0].get('flags', 0))
        if camera_data is not None:
            camera_data.lens_unit = 'MILLIMETERS'
            camera_data.sensor_fit = 'VERTICAL'
            camera_data.clip_end = 100000.0
            camera_data.lens = self._mul_fov_to_blender_lens(camera_data, float(frames[0].get('fov', 0.0)))

        self._create_object_vector_curves(object_action, camera_obj, 'location', frames, 'location')
        self._create_object_vector_curves(object_action, camera_obj, 'rotation_euler', frames, 'rotation')
        self._create_object_vector_curves(object_action, camera_obj, 'scale', frames, 'scale', value_scale=MUL_CAMERA_VISUAL_SCALE)
        self._create_camera_flags_curves(object_action, camera_obj, frames)
        if camera_data is not None and camera_action is not None:
            self._create_camera_fov_curves(camera_action, camera_obj, frames)

        context.scene.frame_start = 0
        context.scene.frame_end = max(int(context.scene.frame_end), final_frame)
        context.scene.frame_current = 0
        return object_action, camera_action, final_frame

    def _import_mul_cameras(
        self,
        context,
        filepath: str,
        cine_header: dict,
        camera_frames: list[dict],
        collection: bpy.types.Collection | None = None,
    ) -> list[dict]:
        if not camera_frames:
            return []

        path = Path(filepath)
        if collection is None:
            base_collection = context.collection or context.scene.collection
            if getattr(base_collection, 'name', '') == path.stem:
                collection = base_collection
            else:
                collection = self._get_or_create_child_collection(base_collection, path.stem)
        camera_collection = collection
        imported_results = []

        for camera_info in camera_frames:
            camera_index = int(camera_info.get('index', len(imported_results)))
            camera_data = bpy.data.cameras.new(f'{path.stem}_mul_cam_{camera_index:02d}')
            camera_data.display_size = MUL_CAMERA_DISPLAY_SIZE
            camera_data.clip_end = 100000.0
            camera_obj = bpy.data.objects.new(camera_data.name, camera_data)
            camera_collection.objects.link(camera_obj)

            object_action, camera_action, final_frame = self._build_action_from_camera_frames(
                context,
                camera_obj,
                f'{path.stem}_cam{camera_index:02d}',
                camera_info,
                time_per_frame=1,
                extra_properties=None,
            )
            if context.scene.camera is None:
                context.scene.camera = camera_obj

            imported_results.append({
                'model': None,
                'mesh_obj': None,
                'mesh_objects': [],
                'arm_obj': None,
                'camera_obj': camera_obj,
                'faces': 0,
                'animation_action': object_action,
                'animation_camera_action': camera_action,
                'animation_final_frame': final_frame,
                'animation_anim_id': None,
                'animation_time_per_frame': 1,
                'source': str(path),
                'mul_header_name': cine_header.get('name', path.stem),
                'mul_camera_index': camera_index,
                'mul_first_channel': int(camera_info.get('first_channel', -1)),
            })

            logger.info(
                'Created MUL camera %d (%s) with %d frames',
                camera_index,
                camera_obj.name,
                len(camera_info.get('frames', [])),
            )

        return imported_results


    @staticmethod
    def _mul_read_audio_header(data: bytes) -> dict | None:
        if len(data) < 0xC8:
            return None

        platform_info = MultiplexStreamImporterMixin._mul_detect_platform(data)
        endian = str(platform_info.get('endian', '<'))
        audio_codec = str(platform_info.get('audio_codec', 'lau_adpcm'))

        try:
            (
                sample_rate,
                start_loop,
                end_loop,
                audio_channel_count,
                reverb_vol,
                start_size_to_load,
                partial_loop,
                loop_area_size,
                has_animation,
                has_subtitles,
            ) = struct.unpack_from(endian + '10i', data, 0)
        except Exception:
            return None

        if sample_rate <= 0 or audio_channel_count <= 0 or audio_channel_count > MUL_AUDIO_MAX_CHANNELS:
            return None

        header_part2_offset = 0x28
        if platform_info.get('platform') == 'xbox':
            try:
                loop_start_file_offset, loop_start_bundle_offset, max_ee_bytes_per_read, media_length = struct.unpack_from(endian + 'iiif', data, header_part2_offset)
            except Exception:
                return None
            # Xbox MUL sound packets identify channels explicitly.  The shipped
            # volume table is not laid out like the TRU PC table, so use the normal
            # stereo routing for the first two channels.
            left_volumes = tuple(1.0 if channel_index == 0 else 0.0 for channel_index in range(12))
            right_volumes = tuple(1.0 if channel_index == 1 else 0.0 for channel_index in range(12))
        else:
            try:
                # TRU PC keeps one extra 32-bit field before mediaLength.  Older code
                # treated mediaLength as part of the left-volume table, which made the
                # first decoded audio channel far too loud in files whose mediaLength is
                # a large frame count such as 338.0 or 346.0.
                (
                    loop_start_file_offset,
                    loop_start_bundle_offset,
                    max_ee_bytes_per_read,
                    _stream_header_unknown_34,
                    media_length,
                ) = struct.unpack_from(endian + 'iiiif', data, header_part2_offset)
                volume_offset = header_part2_offset + 0x14
                left_volumes = struct.unpack_from(endian + '12f', data, volume_offset)
                right_volumes = struct.unpack_from(endian + '12f', data, volume_offset + (12 * 4))
            except Exception:
                try:
                    loop_start_file_offset, loop_start_bundle_offset, max_ee_bytes_per_read, media_length = struct.unpack_from(endian + 'iiif', data, header_part2_offset)
                    volume_offset = header_part2_offset + 0x10
                    left_volumes = struct.unpack_from(endian + '12f', data, volume_offset)
                    right_volumes = struct.unpack_from(endian + '12f', data, volume_offset + (12 * 4))
                except Exception:
                    return None

        dsp_coefficients = []
        if audio_codec == 'gc_dsp_adpcm':
            # Nintendo GameCube/Wii MUL sound packets carry Nintendo DSP ADPCM frames.
            # The stream header stores a compact per-channel DSP setup block. The first
            # channel's 16 signed predictor coefficients start at 0xCC; subsequent
            # coefficient tables are spaced by 0x2E bytes. The previous implementation
            # used a 0x30 stride, which shifted channel 1's coefficients by two bytes
            # and produced severe clipping/crackle.
            for channel_index in range(int(audio_channel_count)):
                coeff_offset = MUL_DSP_CHANNEL_HEADERS_OFFSET + (channel_index * MUL_DSP_CHANNEL_HEADER_SIZE)
                if coeff_offset + 32 > len(data):
                    break
                try:
                    dsp_coefficients.append(tuple(int(v) for v in struct.unpack_from(endian + '16h', data, coeff_offset)))
                except Exception:
                    break

        return {
            'sample_rate': int(sample_rate),
            'start_loop': int(start_loop),
            'end_loop': int(end_loop),
            'audio_channel_count': int(audio_channel_count),
            'reverb_vol': int(reverb_vol),
            'start_size_to_load': int(start_size_to_load),
            'partial_loop': int(partial_loop),
            'loop_area_size': int(loop_area_size),
            'has_animation': int(has_animation),
            'has_subtitles': int(has_subtitles),
            'loop_start_file_offset': int(loop_start_file_offset),
            'loop_start_bundle_offset': int(loop_start_bundle_offset),
            'max_ee_bytes_per_read': int(max_ee_bytes_per_read),
            'media_length': float(media_length),
            'left_volumes': tuple(float(left_volumes[i]) for i in range(int(audio_channel_count))),
            'right_volumes': tuple(float(right_volumes[i]) for i in range(int(audio_channel_count))),
            'platform': platform_info.get('platform', 'pc'),
            'endian': endian,
            'packet_header_size': int(platform_info.get('packet_header_size', MUL_PACKET_HEADER_SIZE)),
            'packet_alignment': int(platform_info.get('packet_alignment', MUL_ALIGNMENT)),
            'sound_header_size': int(platform_info.get('sound_header_size', 0x10)),
            'audio_codec': audio_codec,
            'dsp_coefficients': dsp_coefficients,
        }

    @staticmethod
    def _mul_decode_adpcm_block(input_data: bytes, input_offset: int, endian: str = '<') -> list[int]:
        if input_offset + MUL_AUDIO_BYTES_PER_BLOCK > len(input_data):
            return []
        endian = '>' if endian == '>' else '<'

        value_table = (
            0x0800, 0x1800, 0x2800, 0x3800, 0x4800, 0x5800, 0x6800, 0x7800,
            -0x0800, -0x1800, -0x2800, -0x3800, -0x4800, -0x5800, -0x6800, -0x7800,
        )
        multiplier_table = (
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
        index_change_table = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)

        sample = struct.unpack_from(endian + 'h', input_data, input_offset)[0]
        index = int(struct.unpack_from(endian + 'h', input_data, input_offset + 2)[0])
        index = min(max(index, 0), len(multiplier_table) - 1)
        output = [int(sample)]
        nibble_offset = input_offset + 4
        upper_nibble = True

        for _ in range(1, MUL_AUDIO_SAMPLES_PER_BLOCK):
            if nibble_offset >= input_offset + MUL_AUDIO_BYTES_PER_BLOCK:
                break

            packed = input_data[nibble_offset]
            if upper_nibble:
                code = (packed >> 4) & 0xF
                nibble_offset += 1
            else:
                code = packed & 0xF

            delta = (value_table[code] * multiplier_table[index]) >> 16
            sample = min(max(int(sample) + int(delta), -32768), 32767)
            output.append(int(sample))
            index = min(max(index + index_change_table[code], 0), len(multiplier_table) - 1)
            upper_nibble = not upper_nibble

        if len(output) < MUL_AUDIO_SAMPLES_PER_BLOCK:
            output.extend([output[-1] if output else 0] * (MUL_AUDIO_SAMPLES_PER_BLOCK - len(output)))
        return output

    @staticmethod
    def _mul_decode_dsp_adpcm_frame(input_data: bytes, input_offset: int, coefficients: tuple[int, ...], history: tuple[int, int]) -> tuple[list[int], tuple[int, int]]:
        if input_offset + MUL_DSP_AUDIO_BYTES_PER_FRAME > len(input_data) or len(coefficients) < 16:
            return [], history

        header = input_data[input_offset]
        predictor_index = (header >> 4) & 0xF
        scale = 1 << (header & 0xF)
        if predictor_index > 7:
            predictor_index = 7
        coef1 = int(coefficients[predictor_index * 2])
        coef2 = int(coefficients[(predictor_index * 2) + 1])
        hist1, hist2 = int(history[0]), int(history[1])
        output: list[int] = []

        for packed in input_data[input_offset + 1:input_offset + MUL_DSP_AUDIO_BYTES_PER_FRAME]:
            for code in ((packed >> 4) & 0xF, packed & 0xF):
                if code >= 8:
                    code -= 16
                sample = (((int(code) * scale) << 11) + (coef1 * hist1) + (coef2 * hist2) + 1024) >> 11
                sample = min(max(int(sample), -32768), 32767)
                output.append(sample)
                hist2 = hist1
                hist1 = sample

        return output, (hist1, hist2)

    @staticmethod
    def _mul_decode_dsp_channel_data(channel_data: bytes, coefficients: tuple[int, ...], history: tuple[int, int]) -> tuple[list[int], tuple[int, int]]:
        output: list[int] = []
        usable_size = len(channel_data) - (len(channel_data) % MUL_DSP_AUDIO_BYTES_PER_FRAME)
        for frame_offset in range(0, usable_size, MUL_DSP_AUDIO_BYTES_PER_FRAME):
            frame_samples, history = MultiplexStreamImporterMixin._mul_decode_dsp_adpcm_frame(channel_data, frame_offset, coefficients, history)
            output.extend(frame_samples)
        return output, history

    def _mul_decode_audio_channels(self, data: bytes, audio_header: dict) -> list[list[int]]:
        num_channels = int(audio_header.get('audio_channel_count', 0))
        if num_channels <= 0:
            return []

        endian = str(audio_header.get('endian', '<'))
        packet_header_size = int(audio_header.get('packet_header_size', MUL_PACKET_HEADER_SIZE))
        packet_alignment = int(audio_header.get('packet_alignment', MUL_ALIGNMENT))
        sound_header_size = int(audio_header.get('sound_header_size', 0x10))
        audio_codec = str(audio_header.get('audio_codec', 'lau_adpcm'))
        dsp_coefficients = list(audio_header.get('dsp_coefficients', []) or [])

        channel_samples: list[list[int]] = [[] for _ in range(num_channels)]
        dsp_histories: list[tuple[int, int]] = [(0, 0) for _ in range(num_channels)]
        packet_offset = MUL_STREAM_START_OFFSET
        sound_packet_count = 0

        while packet_offset + packet_header_size <= len(data):
            packet_type, packet_size = struct.unpack_from(endian + 'ii', data, packet_offset)
            packet_data_start = packet_offset + packet_header_size
            packet_data_end = packet_data_start + max(int(packet_size), 0)
            if packet_data_end > len(data):
                break

            if packet_type == MUL_PACKET_TYPE_SOUND:
                if packet_data_start + sound_header_size <= packet_data_end:
                    audio_data_size = int(struct.unpack_from(endian + 'i', data, packet_data_start)[0])
                    audio_data_start = packet_data_start + sound_header_size
                    audio_data_end = min(audio_data_start + max(audio_data_size, 0), packet_data_end)
                    if audio_data_size > 0 and audio_data_end > audio_data_start:
                        if audio_codec == 'xbox_lau_adpcm':
                            # Xbox packets are split into one or more 0x10-byte channel
                            # subheaders followed by that channel's ADPCM payload.  The
                            # second field is the channel index.
                            decoded_any_channel = False
                            channel_block_offset = packet_data_start
                            while channel_block_offset + sound_header_size <= packet_data_end:
                                try:
                                    channel_data_size, channel_index = struct.unpack_from(endian + 'ii', data, channel_block_offset)
                                except Exception:
                                    break
                                channel_data_start = channel_block_offset + sound_header_size
                                channel_data_end = channel_data_start + max(int(channel_data_size), 0)
                                if channel_data_size <= 0 or channel_data_end > packet_data_end:
                                    break
                                if 0 <= int(channel_index) < num_channels:
                                    channel_data = data[channel_data_start:channel_data_end]
                                    for block_offset in range(0, len(channel_data) - (len(channel_data) % MUL_AUDIO_BYTES_PER_BLOCK), MUL_AUDIO_BYTES_PER_BLOCK):
                                        channel_samples[int(channel_index)].extend(self._mul_decode_adpcm_block(channel_data, block_offset, endian=endian))
                                    decoded_any_channel = True
                                channel_block_offset = self._align_mul_offset_to(channel_data_end, MUL_ALIGNMENT)
                            if decoded_any_channel:
                                sound_packet_count += 1
                        else:
                            bytes_per_channel = audio_data_size // num_channels
                            for channel_index in range(num_channels):
                                channel_start = audio_data_start + (channel_index * bytes_per_channel)
                                channel_end = min(channel_start + bytes_per_channel, audio_data_end)
                                channel_data = data[channel_start:channel_end]
                                if audio_codec == 'gc_dsp_adpcm':
                                    if channel_index >= len(dsp_coefficients):
                                        continue
                                    decoded, dsp_histories[channel_index] = self._mul_decode_dsp_channel_data(
                                        channel_data,
                                        tuple(dsp_coefficients[channel_index]),
                                        dsp_histories[channel_index],
                                    )
                                    channel_samples[channel_index].extend(decoded)
                                else:
                                    for block_offset in range(0, len(channel_data) - (len(channel_data) % MUL_AUDIO_BYTES_PER_BLOCK), MUL_AUDIO_BYTES_PER_BLOCK):
                                        channel_samples[channel_index].extend(self._mul_decode_adpcm_block(channel_data, block_offset, endian=endian))
                            sound_packet_count += 1

            packet_offset = self._align_mul_offset_to(packet_data_end, packet_alignment)

        if sound_packet_count == 0:
            return []

        end_loop = int(audio_header.get('end_loop', 0) or 0)
        if end_loop > 0:
            for channel_index, samples in enumerate(channel_samples):
                if len(samples) > end_loop:
                    channel_samples[channel_index] = samples[:end_loop]

        logger.info(
            'Decoded MUL audio: packets=%d codec=%s channels=%d samples=%d sample_rate=%d',
            sound_packet_count,
            audio_codec,
            num_channels,
            max((len(samples) for samples in channel_samples), default=0),
            int(audio_header.get('sample_rate', 0)),
        )
        return channel_samples

    @staticmethod
    def _mul_default_stereo_volumes(num_channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        num_channels = max(0, int(num_channels))
        if num_channels == 1:
            return (1.0,), (1.0,)
        return (
            tuple(1.0 if channel_index == 0 else 0.0 for channel_index in range(num_channels)),
            tuple(1.0 if channel_index == 1 else 0.0 for channel_index in range(num_channels)),
        )

    @staticmethod
    def _mul_sanitized_volume_table(values, num_channels: int) -> tuple[float, ...] | None:
        if values is None or len(values) < int(num_channels):
            return None

        sanitized: list[float] = []
        for channel_index in range(int(num_channels)):
            try:
                value = float(values[channel_index])
            except Exception:
                return None
            if not math.isfinite(value):
                return None

            # MUL volume tables should contain small gain values. Very large values
            # usually mean the header variant was parsed at the wrong offset.
            if abs(value) > 16.0:
                return None
            sanitized.append(value)

        return tuple(sanitized)

    @staticmethod
    def _mul_source_channel_peaks(channel_samples: list[list[int]]) -> list[int]:
        peaks: list[int] = []
        for samples in channel_samples:
            peak = 0
            for sample in samples:
                abs_sample = abs(int(sample))
                if abs_sample > peak:
                    peak = abs_sample
            peaks.append(int(peak))
        return peaks

    @staticmethod
    def _mul_centered_mono_from_channels(channel_samples: list[list[int]], max_sample_count: int) -> list[tuple[int, int]]:
        source_peaks = MultiplexStreamImporterMixin._mul_source_channel_peaks(channel_samples)
        active_channels = [index for index, peak in enumerate(source_peaks) if peak > 16]
        if not active_channels:
            active_channels = list(range(len(channel_samples)))

        centered_samples: list[tuple[int, int]] = []
        for sample_index in range(max_sample_count):
            mixed_value = 0.0
            active_count = 0
            for channel_index in active_channels:
                samples = channel_samples[channel_index]
                if sample_index < len(samples):
                    mixed_value += float(samples[sample_index])
                    active_count += 1
            if active_count > 1:
                mixed_value /= float(active_count)

            value = int(round(min(max(mixed_value, -32768.0), 32767.0)))
            centered_samples.append((value, value))
        return centered_samples

    @staticmethod
    def _mul_mix_audio_to_stereo(channel_samples: list[list[int]], audio_header: dict) -> list[tuple[int, int]]:
        if not channel_samples:
            return []

        num_channels = len(channel_samples)
        max_sample_count = max((len(samples) for samples in channel_samples), default=0)
        if max_sample_count <= 0:
            return []

        left_volumes = MultiplexStreamImporterMixin._mul_sanitized_volume_table(
            tuple(audio_header.get('left_volumes', ()) or ()),
            num_channels,
        )
        right_volumes = MultiplexStreamImporterMixin._mul_sanitized_volume_table(
            tuple(audio_header.get('right_volumes', ()) or ()),
            num_channels,
        )

        default_left_volumes, default_right_volumes = MultiplexStreamImporterMixin._mul_default_stereo_volumes(num_channels)
        if left_volumes is None or right_volumes is None:
            left_volumes, right_volumes = default_left_volumes, default_right_volumes

        left_has_gain = any(abs(float(v)) > 1e-8 for v in left_volumes)
        right_has_gain = any(abs(float(v)) > 1e-8 for v in right_volumes)

        if num_channels == 1:
            # Retail MULs frequently store mono dialogue/music.  Some headers expose
            # only one side in the volume table, but the decoded WAV bridge should be
            # centered so Blender does not play the whole track from one speaker.
            mono_gain = max(abs(float(left_volumes[0])), abs(float(right_volumes[0])), 1.0)
            left_volumes = (mono_gain,)
            right_volumes = (mono_gain,)
        elif not left_has_gain and not right_has_gain:
            left_volumes, right_volumes = default_left_volumes, default_right_volumes
        elif left_has_gain != right_has_gain:
            # A whole decoded stream collapsing to one output side is more likely a
            # misread routing table than intended panning for these cinematic MULs.
            left_volumes, right_volumes = default_left_volumes, default_right_volumes

        stereo_samples: list[tuple[int, int]] = []
        left_peak = 0
        right_peak = 0
        for sample_index in range(max_sample_count):
            left_value = 0.0
            right_value = 0.0
            for channel_index, samples in enumerate(channel_samples):
                sample = float(samples[sample_index]) if sample_index < len(samples) else 0.0
                left_value += sample * float(left_volumes[channel_index])
                right_value += sample * float(right_volumes[channel_index])

            left_int = int(round(min(max(left_value, -32768.0), 32767.0)))
            right_int = int(round(min(max(right_value, -32768.0), 32767.0)))
            left_peak = max(left_peak, abs(left_int))
            right_peak = max(right_peak, abs(right_int))
            stereo_samples.append((left_int, right_int))

        if num_channels > 1 and ((left_peak <= 16 and right_peak > 512) or (right_peak <= 16 and left_peak > 512)):
            logger.info('MUL audio routing produced one silent stereo side; centering decoded source channels')
            return MultiplexStreamImporterMixin._mul_centered_mono_from_channels(channel_samples, max_sample_count)

        return stereo_samples

    @staticmethod
    def _mul_write_stereo_wav(wav_path: Path, stereo_samples: list[tuple[int, int]], sample_rate: int) -> None:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            frame_data = bytearray()
            for left_value, right_value in stereo_samples:
                frame_data.extend(struct.pack('<hh', int(left_value), int(right_value)))
            wav_file.writeframes(bytes(frame_data))

    @staticmethod
    def _mul_audio_temp_path(source_path: Path) -> Path:
        temp_dir = Path(getattr(bpy.app, 'tempdir', '') or tempfile.gettempdir())
        temp_dir.mkdir(parents=True, exist_ok=True)

        safe_stem = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in source_path.stem)
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f'{safe_stem}_mul_audio_',
            suffix='.wav',
            dir=str(temp_dir),
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        return temp_path

    @staticmethod
    def _mul_pack_sound(sound) -> bool:
        if sound is None:
            return False

        pack_method = getattr(sound, 'pack', None)
        if not callable(pack_method):
            return False

        try:
            pack_method()
        except Exception as exc:
            logger.warning('Could not pack MUL audio sound datablock: %s', exc)
            return False

        try:
            return getattr(sound, 'packed_file', None) is not None
        except Exception:
            return True

    @staticmethod
    def _mul_sound_is_packed(sound) -> bool:
        if sound is None:
            return False
        try:
            return getattr(sound, 'packed_file', None) is not None
        except Exception:
            return False

    @staticmethod
    def _mul_delete_temp_file_later(temp_path: Path) -> None:
        # Blender may lazily initialize sound strips after new_sound() returns.
        # Deferring deletion avoids invalidating a just-created sound strip while
        # still keeping the decoded WAV out of the user's project folder.
        def delete_temp_file():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning('Could not delete temporary MUL audio file %s: %s', temp_path, exc)
            return None

        try:
            bpy.app.timers.register(delete_temp_file, first_interval=5.0)
        except Exception:
            delete_temp_file()

    def _extract_mul_audio_to_wav(self, filepath: str, data: bytes | None = None) -> dict | None:
        path = Path(filepath)
        if data is None:
            data = path.read_bytes()
        audio_header = self._mul_read_audio_header(data)
        if audio_header is None:
            logger.info('MUL audio skipped: no valid audio header in %s', path.name)
            return None

        channel_samples = self._mul_decode_audio_channels(data, audio_header)
        if not channel_samples:
            logger.info('MUL audio skipped: no sound packets in %s', path.name)
            return None

        stereo_samples = self._mul_mix_audio_to_stereo(channel_samples, audio_header)
        if not stereo_samples:
            logger.info('MUL audio skipped: decoded audio was empty in %s', path.name)
            return None

        wav_path = self._mul_audio_temp_path(path)
        self._mul_write_stereo_wav(wav_path, stereo_samples, int(audio_header['sample_rate']))
        duration_seconds = float(len(stereo_samples)) / float(audio_header['sample_rate']) if int(audio_header['sample_rate']) > 0 else 0.0
        return {
            'wav_path': str(wav_path),
            'sample_rate': int(audio_header['sample_rate']),
            'sample_count': int(len(stereo_samples)),
            'duration_seconds': duration_seconds,
            'audio_channel_count': int(audio_header.get('audio_channel_count', len(channel_samples))),
        }

    def _find_free_sequence_channel(self, context) -> int:
        sequence_editor = context.scene.sequence_editor_create()
        used_channels = [int(sequence.channel) for sequence in getattr(sequence_editor, 'sequences_all', [])]
        if not used_channels:
            return 1
        return max(used_channels) + 1

    @staticmethod
    def _mul_set_sound_cache(sound) -> None:
        if sound is None:
            return
        try:
            if hasattr(sound, 'use_memory_cache'):
                sound.use_memory_cache = True
        except Exception:
            pass

    def _import_mul_audio_track(self, context, filepath: str, data: bytes | None = None) -> dict | None:
        source_name = Path(filepath).stem
        diagnostics: list[str] = [
            f'MUL audio import report for: {Path(filepath).name}',
            '',
            'This importer does not write a sidecar WAV next to the .mul file.',
            'Blender still needs a WAV file as a decode bridge, so this build keeps that WAV in Blender\'s temp/cache folder.',
            'The sound datablock is packed when Blender exposes sound packing, but the temp WAV is retained for reliable playback in the current session.',
            '',
        ]

        try:
            audio_info = self._extract_mul_audio_to_wav(filepath, data=data)
        except Exception as exc:
            logger.warning('MUL audio import failed for %s: %s', Path(filepath).name, exc)
            diagnostics.append(f'Decode failed: {exc}')
            return None

        if audio_info is None:
            diagnostics.append('No decoded audio was produced. The .mul may not contain sound packets, or the audio header was not recognized.')
            return None

        temp_wav_path = Path(audio_info['wav_path'])
        diagnostics.extend([
            f'Decoded WAV bridge path: {temp_wav_path}',
            f'Temp WAV exists after decode: {temp_wav_path.exists()}',
            f'Sample rate: {int(audio_info["sample_rate"])} Hz',
            f'Sample count: {int(audio_info["sample_count"])}',
            f'Duration: {float(audio_info["duration_seconds"]):.3f} seconds',
            f'Source audio channels: {int(audio_info.get("audio_channel_count", 0))}',
            '',
        ])

        try:
            sequence_editor = context.scene.sequence_editor_create()
            diagnostics.append('Sequence editor: created/available')
        except Exception as exc:
            logger.warning('Could not create Blender sequence editor for MUL audio: %s', exc)
            diagnostics.append(f'Sequence editor creation failed: {exc}')
            return None

        channel = self._find_free_sequence_channel(context)
        strip = None
        sound = None
        packed_audio = False

        try:
            try:
                sound = bpy.data.sounds.load(str(temp_wav_path), check_existing=False)
                sound.name = f'{source_name}_MUL_Audio'
                self._mul_set_sound_cache(sound)
                diagnostics.append(f'Sound datablock loaded: {sound.name}')

                packed_audio = self._mul_pack_sound(sound)
                diagnostics.append(f'Sound datablock packed: {packed_audio}')
                diagnostics.append(f'Sound filepath: {getattr(sound, "filepath", "") or "<empty>"}')
            except Exception as exc:
                logger.warning('Could not create MUL audio sound datablock: %s', exc)
                diagnostics.append(f'Sound datablock load failed: {exc}')
                sound = None
                packed_audio = False

            try:
                strip_collection = getattr(sequence_editor, 'sequences', None)
                strip_collection_name = 'sequences'
                if strip_collection is None or not hasattr(strip_collection, 'new_sound'):
                    strip_collection = getattr(sequence_editor, 'strips', None)
                    strip_collection_name = 'strips'

                if strip_collection is None or not hasattr(strip_collection, 'new_sound'):
                    available_attrs = ', '.join(
                        sorted(name for name in dir(sequence_editor) if 'sequence' in name.lower() or 'strip' in name.lower())
                    )
                    raise AttributeError(
                        'SequenceEditor exposes neither sequences.new_sound nor strips.new_sound. '
                        f'Available sequence/strip attributes: {available_attrs}'
                    )

                strip = strip_collection.new_sound(
                    name=f'{source_name}_MUL_Audio',
                    filepath=str(temp_wav_path),
                    channel=channel,
                    frame_start=0,
                )
                diagnostics.append(f'Video Sequencer sound strip created via SequenceEditor.{strip_collection_name}.new_sound: {strip.name}')
                diagnostics.append(f'Video Sequencer channel: {int(strip.channel)}')
                diagnostics.append(f'Video Sequencer frame_start: {int(strip.frame_start)}')
            except Exception as exc:
                logger.warning('Could not add MUL audio strip to the Video Sequencer: %s', exc)
                diagnostics.append(f'Video Sequencer sound strip creation failed: {exc}')
                strip = None

            strip_sound = getattr(strip, 'sound', None) if strip is not None else None
            if strip_sound is not None:
                try:
                    strip_sound.name = f'{source_name}_MUL_Audio'
                except Exception:
                    pass
                self._mul_set_sound_cache(strip_sound)

                # Prefer the explicitly loaded sound datablock when Blender allows assignment.
                # Some Blender builds do not allow this on sound strips; the strip remains usable
                # with its own file-backed Sound datablock in that case.
                if sound is not None and sound is not strip_sound:
                    try:
                        strip.sound = sound
                        strip_sound = sound
                        diagnostics.append('Video Sequencer strip was assigned to the explicitly loaded Sound datablock')
                    except Exception as exc:
                        diagnostics.append(f'Video Sequencer strip kept its own Sound datablock: {exc}')

                if not self._mul_sound_is_packed(strip_sound):
                    strip_packed = self._mul_pack_sound(strip_sound)
                    packed_audio = bool(packed_audio or strip_packed)
                    diagnostics.append(f'Video Sequencer strip sound packed: {strip_packed}')
                else:
                    packed_audio = True
                    diagnostics.append('Video Sequencer strip sound was already packed')

                try:
                    strip.show_waveform = True
                except Exception:
                    pass
                try:
                    strip.name = f'{source_name}_MUL_Audio'
                except Exception:
                    pass
                try:
                    strip.mute = True
                    diagnostics.append('Video Sequencer sound strip muted by default')
                except Exception as exc:
                    diagnostics.append(f'Could not mute Video Sequencer sound strip by default: {exc}')
                try:
                    strip['trlau_type'] = 'MULAudio'
                    strip['trlau_mul_audio_enabled'] = False
                    strip['trlau_mul_source'] = str(filepath)
                except Exception:
                    pass
            else:
                diagnostics.append('Video Sequencer strip has no accessible Sound datablock')

            diagnostics.append('Speaker object creation: skipped; Video Sequencer strip is used for playback')

            try:
                context.scene.sync_mode = 'AUDIO_SYNC'
                diagnostics.append('Scene sync mode set to AUDIO_SYNC')
            except Exception as exc:
                diagnostics.append(f'Could not set scene sync mode to AUDIO_SYNC: {exc}')
            try:
                context.scene.use_audio_scrub = True
                diagnostics.append('Scene audio scrubbing enabled')
            except Exception as exc:
                diagnostics.append(f'Could not enable scene audio scrubbing: {exc}')

            # Do not delete the temporary WAV in this build. Blender sound strips are file-backed,
            # and packed Sound datablocks are not always enough for immediate timeline playback.
            diagnostics.append('Temp WAV retained for playback reliability: yes')
            diagnostics.append(f'Temp WAV exists after Blender import: {temp_wav_path.exists()}')

        except Exception as exc:
            logger.warning('Unexpected MUL audio import error for %s: %s', Path(filepath).name, exc)
            diagnostics.append(f'Unexpected import error: {exc}')

        if strip is None:
            diagnostics.append('Result: failed. Decoded audio exists, but Blender did not create a usable Video Sequencer strip.')
            logger.warning('MUL audio decoded, but Blender did not create a sequencer strip')
            return None

        if not packed_audio:
            diagnostics.append('Packed Sound datablock: no. Playback still uses the retained temp WAV bridge.')
            logger.warning('MUL audio was imported from a temporary WAV because Blender did not expose sound packing')
        else:
            diagnostics.append('Packed Sound datablock: yes')

        final_frame = int(math.ceil(float(audio_info.get('duration_seconds', 0.0)) * float(context.scene.render.fps or 30)))
        context.scene.frame_start = 0
        context.scene.frame_end = max(int(context.scene.frame_end), final_frame)
        diagnostics.extend([
            f'Scene frame_start: {int(context.scene.frame_start)}',
            f'Scene frame_end: {int(context.scene.frame_end)}',
            '',
            'Where to look in Blender:',
            f'- Video Sequencer: {source_name}_MUL_Audio',
        ])

        logger.info(
            'Imported MUL audio track %s (%d samples @ %d Hz, %.3fs, strip=%s, packed=%s, temp_wav=%s)',
            f'{source_name}_MUL_Audio',
            int(audio_info['sample_count']),
            int(audio_info['sample_rate']),
            float(audio_info['duration_seconds']),
            strip.name if strip is not None else 'none',
            packed_audio,
            str(temp_wav_path),
        )

        return {
            'model': None,
            'mesh_obj': None,
            'mesh_objects': [],
            'arm_obj': None,
            'camera_obj': None,
            'audio_strip': strip,
            'audio_speaker_obj': None,
            'audio_wav_path': str(temp_wav_path),
            'audio_embedded': bool(packed_audio),
            'audio_sample_rate': int(audio_info['sample_rate']),
            'audio_sample_count': int(audio_info['sample_count']),
            'audio_duration_seconds': float(audio_info['duration_seconds']),
            'audio_channel_count': int(audio_info.get('audio_channel_count', 0)),
            'faces': 0,
            'animation_action': None,
            'animation_final_frame': final_frame,
            'animation_anim_id': None,
            'animation_time_per_frame': 1,
            'source': str(filepath),
        }

    def _import_multiplexstream_file(self, context, filepath: str) -> list[dict]:
        logger.info('Beginning MUL import for %s', Path(filepath).name)
        path = Path(filepath)
        file_data = path.read_bytes()
        imported_results = []
        max_final_frame = 0
        context.scene.render.fps = 30
        base_collection = context.collection or context.scene.collection
        if getattr(base_collection, 'name', '') == path.stem:
            mul_collection = base_collection
        else:
            mul_collection = self._get_or_create_child_collection(base_collection, path.stem)

        audio_result = self._import_mul_audio_track(context, filepath, data=file_data)
        if audio_result is not None:
            imported_results.append(audio_result)
            max_final_frame = max(max_final_frame, int(audio_result.get('animation_final_frame', 0) or 0))

        try:
            cine_header, skeleton_frames, camera_frames = self._parse_multiplexstream_file(filepath, data=file_data)
        except Exception:
            if audio_result is None:
                raise
            logger.warning('MUL animation/camera parse failed for %s, but audio was imported', path.name)
            return imported_results

        if not skeleton_frames and not camera_frames and audio_result is None:
            raise ValueError('The .mul file does not contain any cinematic skeleton, camera, or audio tracks')


        multiplex_empty = self._create_multiplex_data_empty(
            context,
            filepath,
            cine_header,
            skeleton_frames,
            camera_frames,
            audio_result=audio_result,
            collection=mul_collection,
        )
        multiplex_metadata = self._build_multiplex_metadata_payload(filepath, cine_header, skeleton_frames, camera_frames)
        multiplex_final_frame = max([0] + [int(item.get('final_frame', 0) or 0) for item in list(multiplex_metadata.get('skeletons', []) or []) + list(multiplex_metadata.get('cameras', []) or [])])
        max_final_frame = max(max_final_frame, int(multiplex_final_frame))
        imported_results.append({
            'model': None,
            'mesh_obj': multiplex_empty,
            'mesh_objects': [],
            'arm_obj': None,
            'multiplex_obj': multiplex_empty,
            'faces': 0,
            'animation_action': None,
            'animation_final_frame': int(multiplex_final_frame),
            'animation_anim_id': None,
            'animation_time_per_frame': 1,
            'source': str(path),
            'mul_header_name': cine_header.get('name', path.stem),
            'mul_skeleton_count': len(skeleton_frames or []),
        })

        camera_results = self._import_mul_cameras(context, filepath, cine_header, camera_frames, collection=mul_collection)
        for result in camera_results:
            max_final_frame = max(max_final_frame, int(result.get('animation_final_frame', 0) or 0))
        imported_results.extend(camera_results)

        context.scene.render.fps = 30
        context.scene.frame_start = 0
        context.scene.frame_end = max(int(context.scene.frame_end), int(max_final_frame))
        logger.info(
            'Imported %d multiplexstream item(s) from %s (%d stored skeleton tracks, %d cameras, %d audio tracks)',
            len(imported_results),
            path.name,
            len(skeleton_frames or []),
            len(camera_results),
            1 if audio_result is not None else 0,
        )
        return imported_results
