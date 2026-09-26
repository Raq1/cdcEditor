from __future__ import annotations

from typing import Tuple

from ...core.model_types import MVertex
from ..ps3.model import decode_ps3_uv


def decode_xbox360_uv(vertex: MVertex) -> Tuple[float, float]:
    return decode_ps3_uv(vertex)
