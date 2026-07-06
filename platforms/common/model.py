from __future__ import annotations

from math import sqrt
from typing import Tuple

from ...core.model_types import MVertex, ModelData, Segment, VirtSegment

NORMAL_SCALE = 127.0


def decode_normal(vertex: MVertex) -> Tuple[float, float, float]:
    nx = max(-1.0, min(1.0, float(vertex.normal_raw[0]) / NORMAL_SCALE))
    ny = max(-1.0, min(1.0, float(vertex.normal_raw[1]) / NORMAL_SCALE))
    nz = max(-1.0, min(1.0, float(vertex.normal_raw[2]) / NORMAL_SCALE))
    length = sqrt((nx * nx) + (ny * ny) + (nz * nz))
    if length <= 1e-8:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def parse_standard_segment(context, index: int) -> Segment:
    br = context.reader
    return Segment(
        index=index,
        min_v=br.vec4(),
        max_v=br.vec4(),
        pivot=br.vec4(),
        flags=br.i32(),
        first_vertex=br.i16(),
        last_vertex=br.i16(),
        parent=br.i32(),
        hinfo=br.u32(),
    )


def parse_compact_segment(context, index: int) -> Segment:
    br = context.reader
    return Segment(
        index=index,
        min_v=(0.0, 0.0, 0.0, 0.0),
        max_v=(0.0, 0.0, 0.0, 0.0),
        pivot=br.vec4(),
        flags=br.i32(),
        first_vertex=br.i16(),
        last_vertex=br.i16(),
        parent=br.i32(),
        hinfo=br.u32(),
    )


def parse_standard_virt_segment(context, virt_index: int) -> VirtSegment:
    br = context.reader
    return VirtSegment(
        virt_index=virt_index,
        min_v=br.vec4(),
        max_v=br.vec4(),
        pivot=br.vec4(),
        flags=br.i32(),
        first_vertex=br.i16(),
        last_vertex=br.i16(),
        index=br.i16(),
        weight_index=br.i16(),
        weight=br.f32(),
    )


def parse_compact_virt_segment(context, virt_index: int) -> VirtSegment:
    br = context.reader
    return VirtSegment(
        virt_index=virt_index,
        min_v=(0.0, 0.0, 0.0, 0.0),
        max_v=(0.0, 0.0, 0.0, 0.0),
        pivot=br.vec4(),
        flags=br.i32(),
        first_vertex=br.i16(),
        last_vertex=br.i16(),
        index=br.i16(),
        weight_index=br.i16(),
        weight=br.f32(),
    )


def resolve_weighted_transform(
    transform_id: int,
    segments: list[Segment],
    virt_segments: list[VirtSegment],
) -> tuple[int, int, int, int, float]:
    bind_segment = -1
    primary_segment = -1
    secondary_segment = -1
    secondary_weight = 0.0
    total_transforms = len(segments) + len(virt_segments)

    if 0 <= transform_id < len(segments):
        bind_segment = transform_id
        primary_segment = transform_id
    elif len(segments) <= transform_id < total_transforms:
        virt_segment = virt_segments[transform_id - len(segments)]
        if 0 <= int(virt_segment.index) < len(segments):
            bind_segment = int(virt_segment.index)
            primary_segment = int(virt_segment.index)
        if 0 <= int(virt_segment.weight_index) < len(segments):
            secondary_segment = int(virt_segment.weight_index)
            secondary_weight = max(0.0, min(1.0, float(virt_segment.weight)))

    return transform_id, bind_segment, primary_segment, secondary_segment, secondary_weight


def make_weighted_virt_segment(
    vertex_index: int,
    primary_segment: int,
    secondary_segment: int,
    secondary_weight: float,
    virt_index: int,
) -> VirtSegment:
    return VirtSegment(
        virt_index=virt_index,
        min_v=(0.0, 0.0, 0.0, 0.0),
        max_v=(0.0, 0.0, 0.0, 0.0),
        pivot=(0.0, 0.0, 0.0, 0.0),
        flags=0,
        first_vertex=vertex_index,
        last_vertex=vertex_index,
        index=int(primary_segment),
        weight_index=int(secondary_segment),
        weight=float(secondary_weight),
    )


def decode_scaled_uv(vertex: MVertex, scale: float = 4096.0) -> Tuple[float, float]:
    uvx, uvy = vertex.uv_raw
    return (float(uvx) / float(scale), 1.0 - (float(uvy) / float(scale)))


def make_segments_only_model_data(
    *,
    version: int,
    segments: list[Segment],
    virt_segments: list[VirtSegment],
    uv_format: str,
    strips=None,
    hmarkers=None,
    hspheres=None,
    hboxes=None,
    hcapsules=None,
) -> ModelData:
    return ModelData(
        version=version,
        model_scale=(1.0, 1.0, 1.0, 1.0),
        segments=segments,
        virt_segments=virt_segments,
        vertices=[],
        faces=[],
        strips=list(strips or []),
        hmarkers=list(hmarkers or []),
        hspheres=list(hspheres or []),
        hboxes=list(hboxes or []),
        hcapsules=list(hcapsules or []),
        targets=[],
        markups=[],
        uv_format=uv_format,
    )
