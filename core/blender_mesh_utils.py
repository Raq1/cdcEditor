from __future__ import annotations


def get_or_create_color_attribute(mesh, name: str = 'Color', *, domain: str = 'POINT'):
    color_attributes = getattr(mesh, 'color_attributes', None)
    if color_attributes is None:
        return None
    color_attr = color_attributes.get(name)
    if color_attr is not None:
        return color_attr
    for attr_type in ('BYTE_COLOR', 'FLOAT_COLOR'):
        try:
            return color_attributes.new(name=name, type=attr_type, domain=domain)
        except Exception:
            continue
    return None


def set_active_color_attribute(mesh, color_attr, name: str = 'Color') -> None:
    if color_attr is None:
        return
    try:
        mesh.color_attributes.active_color = color_attr
    except Exception:
        pass
    try:
        render_index = list(mesh.color_attributes.keys()).index(name)
        mesh.color_attributes.render_color_index = render_index
        mesh.color_attributes.active_color_index = render_index
    except Exception:
        pass


def lock_object_rotation(obj) -> None:
    if obj is None:
        return
    try:
        obj.rotation_mode = 'XYZ'
    except Exception:
        pass
    try:
        obj.rotation_euler = (0.0, 0.0, 0.0)
    except Exception:
        pass
    try:
        obj.lock_rotation = (True, True, True)
    except Exception:
        pass
    try:
        obj.lock_rotation_w = True
    except Exception:
        pass


def armature_bone_count(armature) -> int:
    pose_bones = getattr(getattr(armature, 'pose', None), 'bones', None)
    if pose_bones is not None:
        try:
            return len(pose_bones)
        except Exception:
            pass
    data_bones = getattr(getattr(armature, 'data', None), 'bones', None)
    try:
        return len(data_bones or [])
    except Exception:
        return 0
