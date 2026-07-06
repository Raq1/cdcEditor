from __future__ import annotations

from typing import Tuple

from ...core.model_types import MVertex
from ..common.model import decode_scaled_uv


def decode_gamecube_uv(vertex: MVertex) -> Tuple[float, float]:
    return decode_scaled_uv(vertex)


# I'm not sure this is even necessary but ok
decode_wii_uv = decode_gamecube_uv
