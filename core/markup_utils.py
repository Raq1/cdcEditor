from __future__ import annotations


def markup_bbox_is_valid(bbox) -> bool:
    try:
        values = tuple(int(v) for v in bbox[:6])
    except Exception:
        return False
    if len(values) != 6:
        return False
    return any(values[index] != values[index + 3] for index in range(3))


def markup_bbox_wire_edges(bbox, *, level_space: bool = False):
    if not markup_bbox_is_valid(bbox):
        return []
    min_x, min_y, min_z, max_x, max_y, max_z = tuple(int(v) for v in bbox[:6])

    def convert(raw_point):
        x, y, z = raw_point
        if level_space:
            return (float(-x), float(-y), float(z))
        return (float(x), float(y), float(z))

    corners = [
        convert((min_x, min_y, min_z)),
        convert((max_x, min_y, min_z)),
        convert((max_x, max_y, min_z)),
        convert((min_x, max_y, min_z)),
        convert((min_x, min_y, max_z)),
        convert((max_x, min_y, max_z)),
        convert((max_x, max_y, max_z)),
        convert((min_x, max_y, max_z)),
    ]
    edge_indices = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return [(corners[a], corners[b]) for a, b in edge_indices]


def add_markup_bbox_wire(curve_data, bbox, markup_position, *, level_space: bool = False, bbox_is_local: bool = False) -> bool:
    edges = markup_bbox_wire_edges(bbox, level_space=level_space)
    if not edges:
        return False
    if bbox_is_local:
        base_x = base_y = base_z = 0.0
    else:
        base_x, base_y, base_z = (float(markup_position[0]), float(markup_position[1]), float(markup_position[2]))
    for edge in edges:
        spline = curve_data.splines.new('POLY')
        spline.points.add(1)
        for point_index, point in enumerate(edge):
            px, py, pz = point
            spline.points[point_index].co = (float(px) - base_x, float(py) - base_y, float(pz) - base_z, 1.0)
    return True
