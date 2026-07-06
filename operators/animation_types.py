from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from mathutils import Vector


@dataclass(slots=True)
class TRLAUKeyframe:
    frame: int
    value: float


@dataclass(slots=True)
class TRLAUTrack:
    keyframes: list[TRLAUKeyframe] = field(default_factory=list)
    mode: int | None = None

    def get_value_at_frame(self, frame: int) -> float:
        if not self.keyframes:
            return 0.0
        if frame <= self.keyframes[0].frame:
            return self.keyframes[0].value
        if frame >= self.keyframes[-1].frame:
            return self.keyframes[-1].value

        for left_keyframe, right_keyframe in zip(self.keyframes, self.keyframes[1:]):
            if left_keyframe.frame <= frame <= right_keyframe.frame:
                span = right_keyframe.frame - left_keyframe.frame
                subframe = (frame - left_keyframe.frame) / span if span else 0.0
                return left_keyframe.value + (right_keyframe.value - left_keyframe.value) * subframe
        return 0.0


@dataclass(slots=True)
class TRLAUTransformAnimation:
    axis_tracks: list[TRLAUTrack | None] = field(default_factory=lambda: [None, None, None])

    def get_keyframe_frames(self) -> list[int]:
        return sorted({keyframe.frame for axis_track in self.axis_tracks if axis_track is not None for keyframe in axis_track.keyframes})

    def get_vector_at_frame(self, frame: int, default_axis_value: float) -> Vector:
        result = Vector([default_axis_value] * len(self.axis_tracks))
        for axis, axis_track in enumerate(self.axis_tracks):
            if axis_track is not None:
                result[axis] = axis_track.get_value_at_frame(frame)
        return result


class TRLAUTransformType(IntEnum):
    ROTATION = 0
    SCALE = 1
    LOCATION = 2


@dataclass(slots=True)
class TRLAUBoneAnimation:
    transforms: list[TRLAUTransformAnimation | None] = field(default_factory=lambda: [None, None, None])
