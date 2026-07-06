from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .binary import BinaryReader
from .log import logger
from ..platforms.common.section import TRSectionParser


WAVE_SECTION_TYPE = 6
WAVE_PAYLOAD_HEADER_SIZE = 12
ADPCM_BLOCK_SIZE = 36
ADPCM_SAMPLES_PER_BLOCK = 64

AUDIO_SOURCE_EXTENSIONS = {
    '.wav', '.waveaudio', '.mp3', '.ogg', '.oga', '.flac', '.m4a', '.aac', '.aiff', '.aif',
    '.wma', '.opus', '.webm', '.mp4', '.m4v', '.mov', '.mka', '.caf', '.amr', '.ac3',
}
_NON_SECTION_SCAN_SUFFIXES = {
    '.wav', '.mp3', '.ogg', '.oga', '.flac', '.m4a', '.aac', '.aiff', '.aif', '.wma',
    '.opus', '.webm', '.mp4', '.m4v', '.mov', '.mka', '.caf', '.amr', '.ac3', '.txt',
    '.json', '.py', '.pyc', '.md', '.png', '.jpg', '.jpeg', '.bmp', '.tga', '.dds',
    '.zip', '.7z', '.rar', '.blend', '.bak', '.tmp',
}

_STEP_TABLE = (
    7, 8, 9, 10, 11, 12, 13, 14,
    16, 17, 19, 21, 23, 25, 28, 31,
    34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143,
    157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658,
    724, 796, 876, 963, 1060, 1166, 1282, 1411,
    1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024,
    3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484,
    7132, 7845, 8191, 8191, 8191, 8191, 8191, 8191,
    8191, 8191, 8191, 8191, 8191, 8191, 8191, 8191,
    8191,
)
_INDEX_TABLE = (-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8)


@dataclass(frozen=True)
class WaveSectionReplaceResult:
    input_path: Path
    output_path: Path
    sample_rate: int
    source_sample_count: int
    adpcm_block_count: int
    loop_start: int
    loop_end: int
    original_loop_start: int
    original_loop_end: int
    original_size: int
    new_size: int




@dataclass(frozen=True)
class EncodedWaveAudio:
    sample_rate: int
    source_sample_count: int
    adpcm_data: bytes
    adpcm_block_count: int

    def payload_for_loop(self, loop_start: int = 0, loop_end: int | None = None) -> bytes:
        sample_count = int(self.source_sample_count)
        start = max(0, min(int(loop_start), sample_count))
        if loop_end is None:
            loop_end = sample_count
        end = max(start, min(int(loop_end), sample_count))
        return struct.pack('<III', int(self.sample_rate), int(start), int(end)) + self.adpcm_data

class WaveAudioError(Exception):
    pass


def _clamp_i16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def _decode_sample(code: int, predicted: int, step: int) -> int:
    delta = int(step) >> 3
    if int(code) & 1:
        delta += int(step) >> 2
    if int(code) & 2:
        delta += int(step) >> 1
    if int(code) & 4:
        delta += int(step)
    if int(code) & 8:
        delta = -delta
    return _clamp_i16(int(predicted) + int(delta))


def _next_step_index(step_index: int, code: int) -> int:
    return max(0, min(88, int(step_index) + int(_INDEX_TABLE[int(code) & 0x0F])))


def _encode_nibble(sample: int, predicted: int, step_index: int) -> tuple[int, int, int]:
    step_index = max(0, min(88, int(step_index)))
    step = int(_STEP_TABLE[step_index])
    diff = int(sample) - int(predicted)
    code = 0
    if diff < 0:
        code |= 8
        diff = -diff

    temp_step = step
    if diff >= temp_step:
        code |= 4
        diff -= temp_step
    temp_step >>= 1
    if diff >= temp_step:
        code |= 2
        diff -= temp_step
    temp_step >>= 1
    if diff >= temp_step:
        code |= 1

    predicted = _decode_sample(code, predicted, step)
    step_index = _next_step_index(step_index, code)
    return int(code) & 0x0F, int(predicted), int(step_index)


def _initial_step_index_for_block(samples: list[int]) -> int:
    if len(samples) < 2:
        return 0
    deltas = [abs(int(samples[i]) - int(samples[i - 1])) for i in range(1, min(len(samples), ADPCM_SAMPLES_PER_BLOCK))]
    deltas = [value for value in deltas if value > 0]
    if not deltas:
        return 0
    deltas.sort()
    target = deltas[min(len(deltas) - 1, max(0, int(len(deltas) * 0.75)))]
    target = max(7, int(target / 1.5))
    for index, step in enumerate(_STEP_TABLE):
        if int(step) >= target:
            return int(index)
    return 88


def encode_cd_adpcm_block(input_samples: Iterable[int]) -> bytes:
    samples = [_clamp_i16(value) for value in input_samples]
    if not samples:
        samples = [0]
    if len(samples) < ADPCM_SAMPLES_PER_BLOCK:
        samples.extend([samples[-1]] * (ADPCM_SAMPLES_PER_BLOCK - len(samples)))
    elif len(samples) > ADPCM_SAMPLES_PER_BLOCK:
        samples = samples[:ADPCM_SAMPLES_PER_BLOCK]

    predicted = int(samples[0])
    step_index = _initial_step_index_for_block(samples)
    block = bytearray(ADPCM_BLOCK_SIZE)
    struct.pack_into('<h', block, 0, predicted)
    block[2] = int(step_index) & 0xFF
    block[3] = 0

    code, predicted, step_index = _encode_nibble(samples[1], predicted, step_index)
    block[4] = (int(code) & 0x0F) << 4

    sample_index = 2
    for byte_index in range(5, ADPCM_BLOCK_SIZE):
        low, predicted, step_index = _encode_nibble(samples[sample_index], predicted, step_index)
        sample_index += 1
        high, predicted, step_index = _encode_nibble(samples[sample_index], predicted, step_index)
        sample_index += 1
        block[byte_index] = (int(low) & 0x0F) | ((int(high) & 0x0F) << 4)

    return bytes(block)


def encode_pcm_i16_to_encoded_wave_audio(samples: list[int], sample_rate: int) -> EncodedWaveAudio:
    if not samples:
        raise WaveAudioError('The source audio does not contain any samples')
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise WaveAudioError(f'Invalid source audio sample rate: {sample_rate}')

    source_sample_count = len(samples)
    adpcm_data = bytearray()
    block_count = 0
    for offset in range(0, source_sample_count, ADPCM_SAMPLES_PER_BLOCK):
        adpcm_data.extend(encode_cd_adpcm_block(samples[offset:offset + ADPCM_SAMPLES_PER_BLOCK]))
        block_count += 1
    return EncodedWaveAudio(
        sample_rate=int(sample_rate),
        source_sample_count=int(source_sample_count),
        adpcm_data=bytes(adpcm_data),
        adpcm_block_count=int(block_count),
    )


def encode_pcm_i16_to_wave_payload(samples: list[int], sample_rate: int, loop_start: int = 0, loop_end: int | None = None) -> tuple[bytes, int]:
    encoded_audio = encode_pcm_i16_to_encoded_wave_audio(samples, sample_rate)
    return encoded_audio.payload_for_loop(loop_start, loop_end), int(encoded_audio.adpcm_block_count)


def _convert_pcm_frames_to_mono_i16(raw: bytes, sample_width: int, channels: int) -> list[int]:
    channels = max(1, int(channels))
    sample_width = int(sample_width)
    if sample_width not in {1, 2, 3, 4}:
        raise WaveAudioError(f'Unsupported source audio sample width: {sample_width} byte(s)')
    frame_size = sample_width * channels
    if frame_size <= 0:
        return []

    samples: list[int] = []
    for frame_offset in range(0, len(raw) - frame_size + 1, frame_size):
        total = 0
        for channel_index in range(channels):
            offset = frame_offset + (channel_index * sample_width)
            if sample_width == 1:
                value = (int(raw[offset]) - 128) << 8
            elif sample_width == 2:
                value = struct.unpack_from('<h', raw, offset)[0]
            elif sample_width == 3:
                chunk = raw[offset:offset + 3]
                sign = b'\xff' if chunk[2] & 0x80 else b'\x00'
                value = int.from_bytes(chunk + sign, 'little', signed=True) >> 8
            else:
                value = struct.unpack_from('<i', raw, offset)[0] >> 16
            total += int(value)
        samples.append(_clamp_i16(int(round(total / channels))))
    return samples


def _read_wav_stream_as_mono_i16(wav_source, label: str) -> tuple[list[int], int]:
    try:
        with wave.open(wav_source, 'rb') as wav_file:
            comp_type = wav_file.getcomptype()
            if comp_type not in {'NONE', 'not compressed'}:
                raise WaveAudioError(f'Unsupported compressed WAV type: {comp_type}')
            channels = int(wav_file.getnchannels())
            sample_width = int(wav_file.getsampwidth())
            sample_rate = int(wav_file.getframerate())
            frame_count = int(wav_file.getnframes())
            raw = wav_file.readframes(frame_count)
    except WaveAudioError:
        raise
    except Exception as exc:
        raise WaveAudioError(f'Failed to read source WAV {label}: {exc}') from exc

    samples = _convert_pcm_frames_to_mono_i16(raw, sample_width, channels)
    if not samples:
        raise WaveAudioError(f'Source audio contains no samples: {label}')
    return samples, sample_rate


def read_wav_as_mono_i16(filepath: str | Path) -> tuple[list[int], int]:
    return _read_wav_stream_as_mono_i16(str(Path(filepath)), str(Path(filepath)))


def _candidate_ffmpeg_paths() -> list[str]:
    candidates: list[str] = []
    for env_name in ('TRLAU_FFMPEG', 'FFMPEG_BINARY', 'FFMPEG_PATH'):
        value = os.environ.get(env_name, '')
        if value:
            candidates.append(value)

    found = shutil.which('ffmpeg')
    if found:
        candidates.append(found)
    found_exe = shutil.which('ffmpeg.exe')
    if found_exe:
        candidates.append(found_exe)

    # Some users keep ffmpeg next to Blender or in a bundled bin directory.
    try:
        import bpy  # type: ignore
        blender_dir = Path(getattr(bpy.app, 'binary_path', '') or '').resolve().parent
        for name in ('ffmpeg', 'ffmpeg.exe'):
            candidates.append(str(blender_dir / name))
            candidates.append(str(blender_dir / 'bin' / name))
            candidates.append(str(blender_dir.parent / 'bin' / name))
    except Exception:
        pass

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate or '').strip().strip('"')
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _find_ffmpeg_executable() -> str:
    for candidate in _candidate_ffmpeg_paths():
        path = Path(candidate)
        if path.is_file():
            return str(path)
        if os.path.sep not in candidate and (os.path.altsep is None or os.path.altsep not in candidate):
            found = shutil.which(candidate)
            if found:
                return found
    return ''


def _decode_audio_with_ffmpeg(filepath: str | Path) -> tuple[list[int], int]:
    path = Path(filepath)
    ffmpeg = _find_ffmpeg_executable()
    if not ffmpeg:
        raise WaveAudioError(
            'This source audio format requires FFmpeg. Install ffmpeg and make it available on PATH, '
            'or set TRLAU_FFMPEG to the ffmpeg executable path.'
        )

    cmd = [
        ffmpeg,
        '-hide_banner',
        '-nostdin',
        '-v',
        'error',
        '-i',
        str(path),
        '-map',
        '0:a:0',
        '-vn',
        '-ac',
        '1',
        '-f',
        'wav',
        '-acodec',
        'pcm_s16le',
        'pipe:1',
    ]
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception as exc:
        raise WaveAudioError(f'Failed to launch FFmpeg for source audio {path}: {exc}') from exc

    if completed.returncode != 0 or not completed.stdout:
        stderr = completed.stderr.decode('utf-8', errors='replace').strip()
        if len(stderr) > 500:
            stderr = stderr[:500] + '...'
        detail = f': {stderr}' if stderr else ''
        raise WaveAudioError(f'FFmpeg could not decode source audio {path}{detail}')

    return _read_wav_stream_as_mono_i16(io.BytesIO(completed.stdout), str(path))


def read_audio_as_mono_i16(filepath: str | Path) -> tuple[list[int], int]:
    path = Path(filepath)
    if not path.is_file():
        raise WaveAudioError(f'Source audio file does not exist: {path}')

    direct_wav_error: WaveAudioError | None = None
    if path.suffix.lower() == '.wav':
        try:
            return read_wav_as_mono_i16(path)
        except WaveAudioError as exc:
            direct_wav_error = exc

    try:
        return _decode_audio_with_ffmpeg(path)
    except WaveAudioError as exc:
        if direct_wav_error is not None:
            raise WaveAudioError(f'{direct_wav_error}; FFmpeg fallback failed: {exc}') from exc
        raise


def parse_wave_section_info(raw_data: bytes):
    if len(raw_data) < 24:
        raise WaveAudioError('File is too small to contain a TRLAU section header')
    try:
        return TRSectionParser.parse(BinaryReader(io.BytesIO(raw_data), endian='<'))
    except Exception as exc:
        raise WaveAudioError(f'Failed to parse TRLAU section header: {exc}') from exc


def is_wave_section_file(filepath: str | Path) -> bool:
    path = Path(filepath)
    try:
        with path.open('rb') as fh:
            header = fh.read(24)
        if len(header) < 24:
            return False
        if header[:4] != b'SECT':
            return False
        return int(header[8]) == WAVE_SECTION_TYPE
    except Exception:
        return False


def replace_wave_section_audio_bytes_with_encoded(raw_data: bytes, encoded_audio: EncodedWaveAudio) -> tuple[bytes, dict[str, int]]:
    info = parse_wave_section_info(raw_data)
    if int(getattr(info, 'section_type', -1)) != WAVE_SECTION_TYPE:
        raise WaveAudioError(f'Section type is {int(getattr(info, "section_type", -1))}, expected Wave section type {WAVE_SECTION_TYPE}')

    info_size = int(getattr(info, 'info_size', 24) or 24)
    section_size = int(getattr(info, 'size', 0) or 0)

    payload_end = int(info_size) + max(0, section_size)
    if payload_end > len(raw_data):
        fallback_total_end = max(info_size, section_size)
        if fallback_total_end <= len(raw_data):
            payload_end = fallback_total_end
        else:
            payload_end = len(raw_data)
    old_payload = bytes(raw_data[info_size:payload_end])
    original_size = int(info_size) + len(old_payload)
    original_loop_start = 0
    original_loop_end = 0
    if len(old_payload) >= WAVE_PAYLOAD_HEADER_SIZE:
        try:
            _old_rate, original_loop_start, original_loop_end = struct.unpack_from('<III', old_payload, 0)
        except Exception:
            pass

    source_sample_count = int(encoded_audio.source_sample_count)
    loop_start = 0
    loop_end = source_sample_count

    new_payload = encoded_audio.payload_for_loop(loop_start=loop_start, loop_end=loop_end)
    new_payload_size = len(new_payload)
    new_size = int(info_size) + int(new_payload_size)
    new_data = bytearray(raw_data[:info_size])
    struct.pack_into('<i', new_data, 4, int(new_payload_size))
    new_data.extend(new_payload)

    metadata = {
        'sample_rate': int(encoded_audio.sample_rate),
        'source_sample_count': int(source_sample_count),
        'adpcm_block_count': int(encoded_audio.adpcm_block_count),
        'loop_start': int(loop_start),
        'loop_end': int(loop_end),
        'original_loop_start': int(original_loop_start),
        'original_loop_end': int(original_loop_end),
        'original_size': int(original_size),
        'new_size': int(new_size),
    }
    return bytes(new_data), metadata


def _result_from_meta(input_path: Path, output_path: Path, meta: dict[str, int]) -> WaveSectionReplaceResult:
    return WaveSectionReplaceResult(
        input_path=input_path,
        output_path=output_path,
        sample_rate=int(meta['sample_rate']),
        source_sample_count=int(meta['source_sample_count']),
        adpcm_block_count=int(meta['adpcm_block_count']),
        loop_start=int(meta['loop_start']),
        loop_end=int(meta['loop_end']),
        original_loop_start=int(meta['original_loop_start']),
        original_loop_end=int(meta['original_loop_end']),
        original_size=int(meta['original_size']),
        new_size=int(meta['new_size']),
    )


def replace_wave_section_audio_bytes(raw_data: bytes, source_samples: list[int], sample_rate: int) -> tuple[bytes, dict[str, int]]:
    encoded_audio = encode_pcm_i16_to_encoded_wave_audio(source_samples, sample_rate)
    return replace_wave_section_audio_bytes_with_encoded(raw_data, encoded_audio)


def replace_wave_section_file_with_encoded_audio(input_path: str | Path, output_path: str | Path, encoded_audio: EncodedWaveAudio) -> WaveSectionReplaceResult:
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw = input_path.read_bytes()
    new_data, meta = replace_wave_section_audio_bytes_with_encoded(raw, encoded_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(new_data)
    return _result_from_meta(input_path, output_path, meta)


def replace_wave_section_file(input_path: str | Path, output_path: str | Path, source_samples: list[int], sample_rate: int) -> WaveSectionReplaceResult:
    encoded_audio = encode_pcm_i16_to_encoded_wave_audio(source_samples, sample_rate)
    return replace_wave_section_file_with_encoded_audio(input_path, output_path, encoded_audio)


def iter_wave_section_candidates(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        raise WaveAudioError(f'Target Wave folder does not exist: {directory}')

    candidates: list[Path] = []
    for path in directory.rglob('*'):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == '.wave':
            candidates.append(path)
            continue
        if suffix in _NON_SECTION_SCAN_SUFFIXES:
            continue
        try:
            if is_wave_section_file(path):
                candidates.append(path)
        except Exception as exc:
            logger.debug('Skipped non-Wave file during Wave batch scan %s: %s', path, exc) 
    candidates.sort(key=lambda value: str(value).lower())
    return candidates
