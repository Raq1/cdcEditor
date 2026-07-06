from __future__ import annotations

from struct import pack, unpack
from typing import Tuple

from ...core.model_types import MVertex


def ushort_to_float(value: int) -> float:
    return float(unpack('<f', pack('<I', int(value) << 16))[0])


def decode_pc_uv(vertex: MVertex) -> Tuple[float, float]:
    uvx, uvy = vertex.uv_raw
    return (ushort_to_float(uvx), 1.0 - ushort_to_float(uvy))
