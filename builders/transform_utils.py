from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.model_types import ModelData, Segment, VirtSegment


Vector3 = Tuple[float, float, float]
VertexWeights = Dict[int, float]


def accumulate_segment_pivot(segments: List[Segment], segment_index: int) -> Vector3:
    if segment_index < 0 or segment_index >= len(segments):
        return (0.0, 0.0, 0.0)

    result = [0.0, 0.0, 0.0]
    visited: set[int] = set()
    current = segment_index

    while 0 <= current < len(segments) and current not in visited:
        visited.add(current)
        pivot = segments[current].pivot
        result[0] += float(pivot[0])
        result[1] += float(pivot[1])
        result[2] += float(pivot[2])

        parent = segments[current].parent
        if parent < 0 or parent == current:
            break
        current = parent

    return (result[0], result[1], result[2])


def _clamp_weight(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def find_virt_segment(model: ModelData, vertex_index: int) -> Optional[VirtSegment]:
    for virt_segment in model.virt_segments:
        if virt_segment.first_vertex <= vertex_index <= virt_segment.last_vertex:
            return virt_segment
    return None


def get_vertex_weights(model: ModelData, vertex_index: int) -> VertexWeights:
    if 0 <= int(vertex_index) < len(model.vertices):
        vertex = model.vertices[int(vertex_index)]
        explicit_weights = getattr(vertex, 'skin_weights', None)
        if explicit_weights:
            weights: VertexWeights = {}
            for bone_index, weight in explicit_weights:
                bone_index = int(bone_index)
                weight = _clamp_weight(float(weight))
                if 0 <= bone_index < len(model.segments) and weight > 0.0:
                    weights[bone_index] = weights.get(bone_index, 0.0) + weight
            total = sum(weights.values())
            if total > 0.0:
                return {segment_index: weight / total for segment_index, weight in weights.items()}

        primary_segment = int(getattr(vertex, 'gc_primary_segment', -1))
        secondary_segment = int(getattr(vertex, 'gc_secondary_segment', -1))
        secondary_weight = _clamp_weight(float(getattr(vertex, 'gc_secondary_weight', 0.0)))
        if 0 <= primary_segment < len(model.segments):
            if 0 <= secondary_segment < len(model.segments) and secondary_weight > 0.0:
                primary_weight = 1.0 - secondary_weight
                weights: VertexWeights = {}
                if primary_weight > 0.0:
                    weights[primary_segment] = primary_weight
                weights[secondary_segment] = weights.get(secondary_segment, 0.0) + secondary_weight
                total = sum(weights.values())
                if total > 0.0:
                    return {segment_index: weight / total for segment_index, weight in weights.items()}
            return {primary_segment: 1.0}

    virt_segment = find_virt_segment(model, vertex_index)
    if virt_segment is not None:
        primary_weight = 1.0 - _clamp_weight(virt_segment.weight)
        secondary_weight = _clamp_weight(virt_segment.weight)
        weights: VertexWeights = {}

        if 0 <= virt_segment.index < len(model.segments) and primary_weight > 0.0:
            weights[virt_segment.index] = primary_weight

        if 0 <= virt_segment.weight_index < len(model.segments) and secondary_weight > 0.0:
            weights[virt_segment.weight_index] = secondary_weight

        if weights:
            total = sum(weights.values())
            if total > 0.0:
                return {segment_index: weight / total for segment_index, weight in weights.items()}

    base_segment = model.vertices[vertex_index].segment
    if 0 <= base_segment < len(model.segments):
        return {base_segment: 1.0}
    return {}


def get_vertex_bind_segment_index(model: ModelData, vertex_index: int) -> int:
    if 0 <= int(vertex_index) < len(model.vertices):
        vertex = model.vertices[int(vertex_index)]
        explicit_weights = getattr(vertex, 'skin_weights', None)
        if explicit_weights:
            for bone_index, weight in explicit_weights:
                bone_index = int(bone_index)
                if 0 <= bone_index < len(model.segments) and float(weight) > 0.0:
                    return bone_index

        bind_segment = int(getattr(vertex, 'gc_bind_segment', -1))
        if 0 <= bind_segment < len(model.segments):
            return bind_segment
        primary_segment = int(getattr(vertex, 'gc_primary_segment', -1))
        if 0 <= primary_segment < len(model.segments):
            return primary_segment

    virt_segment = find_virt_segment(model, vertex_index)
    if virt_segment is not None and 0 <= virt_segment.index < len(model.segments):
        return virt_segment.index

    base_segment = model.vertices[vertex_index].segment
    if 0 <= base_segment < len(model.segments):
        return base_segment

    return -1


def get_vertex_translation(model: ModelData, vertex_index: int) -> Vector3:
    bind_segment_index = get_vertex_bind_segment_index(model, vertex_index)
    if bind_segment_index < 0:
        return (0.0, 0.0, 0.0)
    return accumulate_segment_pivot(model.segments, bind_segment_index)
