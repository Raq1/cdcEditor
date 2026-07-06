from __future__ import annotations

from typing import Tuple

from ...core.model_types import MVertex
from ..common.model import decode_scaled_uv


def decode_ps2_uv(vertex: MVertex) -> Tuple[float, float]:
    return decode_scaled_uv(vertex)
