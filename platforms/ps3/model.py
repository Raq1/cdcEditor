from __future__ import annotations

from typing import Tuple

from ...core.model_types import MVertex


def decode_ps3_uv(vertex: MVertex) -> Tuple[float, float]:
    if vertex.uv_decoded is not None:
        return vertex.uv_decoded
    uvx, uvy = vertex.uv_raw
    return (float(uvx) / 4096.0, 1.0 - (float(uvy) / 4096.0))
