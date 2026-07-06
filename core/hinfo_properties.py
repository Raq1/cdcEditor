from typing import Any, Mapping

import bpy


HINFO_DATA_FIELD_DEFS: dict[str, tuple[tuple[str, str, int], ...]] = {
    'HMarker': (
        ('trlau_marker_index', 'Index', 0),
    ),
    'HSphere': (
        ('trlau_hsphere_flags', 'Flags', 4352),
        ('trlau_hsphere_id', 'ID', 0),
        ('trlau_hsphere_rank', 'Rank', 0),
        ('trlau_hsphere_radius_sq', 'Radius Squared', 0),
        ('trlau_hsphere_mass', 'Mass', 100),
        ('trlau_hsphere_buoyancy_factor', 'Buoyancy Factor', 100),
        ('trlau_hsphere_explosion_factor', 'Explosion Factor', 100),
        ('trlau_hsphere_material_type', 'iHitMaterialType', 0),
        ('trlau_hsphere_pad', 'Pad', 0),
        ('trlau_hsphere_damage', 'Damage', 0),
    ),
    'HBox': (
        ('trlau_hbox_flags', 'Flags', 4352),
        ('trlau_hbox_id', 'ID', 0),
        ('trlau_hbox_rank', 'Rank', 0),
        ('trlau_hbox_mass', 'Mass', 100),
        ('trlau_hbox_buoyancy_factor', 'Buoyancy Factor', 100),
        ('trlau_hbox_explosion_factor', 'Explosion Factor', 100),
        ('trlau_hbox_material_type', 'iHitMaterialType', 0),
        ('trlau_hbox_pad', 'Pad', 0),
        ('trlau_hbox_damage', 'Damage', 0),
    ),
    'HCapsule': (
        ('trlau_hcapsule_flags', 'Flags', 4352),
        ('trlau_hcapsule_id', 'ID', 0),
        ('trlau_hcapsule_rank', 'Rank', 0),
        ('trlau_hcapsule_mass', 'Mass', 100),
        ('trlau_hcapsule_buoyancy_factor', 'Buoyancy Factor', 100),
        ('trlau_hcapsule_explosion_factor', 'Explosion Factor', 100),
        ('trlau_hcapsule_material_type', 'iHitMaterialType', 0),
        ('trlau_hcapsule_pad', 'Pad', 0),
        ('trlau_hcapsule_damage', 'Damage', 0),
    ),
}

HINFO_DATA_KEYS: frozenset[str] = frozenset(
    key for fields in HINFO_DATA_FIELD_DEFS.values() for key, _label, _default in fields
)

HINFO_MIGRATED_CUSTOM_PROP_KEYS: frozenset[str] = frozenset({
    *HINFO_DATA_KEYS,
    'trlau_owner_segment',
    'trlau_marker_bone',
})


def _int_prop(name: str, default: int = 0, **kwargs):
    return bpy.props.IntProperty(name=name, default=int(default), **kwargs)


class TRLAU_HInfoDataProperties(bpy.types.PropertyGroup):
    trlau_hinfo_component_type: bpy.props.StringProperty(name='HInfo Component Type', default='')
    trlau_owner_segment: _int_prop('Owner Segment', 0, min=0)
    trlau_marker_bone: _int_prop('Marker Bone', 0, min=0)

    trlau_marker_index: _int_prop('Index', 0, min=0)

    trlau_hsphere_flags: _int_prop('Flags', 4352, min=0, max=0xFFFF)
    trlau_hsphere_id: _int_prop('ID', 0, min=0, max=0xFF)
    trlau_hsphere_rank: _int_prop('Rank', 0, min=0, max=0xFF)
    trlau_hsphere_radius_sq: _int_prop('Radius Squared', 0, min=0)
    trlau_hsphere_mass: _int_prop('Mass', 100, min=0, max=0xFFFF)
    trlau_hsphere_buoyancy_factor: _int_prop('Buoyancy Factor', 100, min=0, max=0xFF)
    trlau_hsphere_explosion_factor: _int_prop('Explosion Factor', 100, min=0, max=0xFF)
    trlau_hsphere_material_type: _int_prop('iHitMaterialType', 0, min=0, max=0xFF)
    trlau_hsphere_pad: _int_prop('Pad', 0, min=0, max=0xFF)
    trlau_hsphere_damage: _int_prop('Damage', 0, min=-0x8000, max=0x7FFF)

    trlau_hbox_flags: _int_prop('Flags', 4352, min=0, max=0xFFFF)
    trlau_hbox_id: _int_prop('ID', 0, min=0, max=0xFF)
    trlau_hbox_rank: _int_prop('Rank', 0, min=0, max=0xFF)
    trlau_hbox_mass: _int_prop('Mass', 100, min=0, max=0xFFFF)
    trlau_hbox_buoyancy_factor: _int_prop('Buoyancy Factor', 100, min=0, max=0xFF)
    trlau_hbox_explosion_factor: _int_prop('Explosion Factor', 100, min=0, max=0xFF)
    trlau_hbox_material_type: _int_prop('iHitMaterialType', 0, min=0, max=0xFF)
    trlau_hbox_pad: _int_prop('Pad', 0, min=0, max=0xFF)
    trlau_hbox_damage: _int_prop('Damage', 0, min=-0x8000, max=0x7FFF)

    trlau_hcapsule_flags: _int_prop('Flags', 4352, min=0, max=0xFFFF)
    trlau_hcapsule_id: _int_prop('ID', 0, min=0, max=0xFF)
    trlau_hcapsule_rank: _int_prop('Rank', 0, min=0, max=0xFF)
    trlau_hcapsule_mass: _int_prop('Mass', 100, min=0, max=0xFFFF)
    trlau_hcapsule_buoyancy_factor: _int_prop('Buoyancy Factor', 100, min=0, max=0xFF)
    trlau_hcapsule_explosion_factor: _int_prop('Explosion Factor', 100, min=0, max=0xFF)
    trlau_hcapsule_material_type: _int_prop('iHitMaterialType', 0, min=0, max=0xFF)
    trlau_hcapsule_pad: _int_prop('Pad', 0, min=0, max=0xFF)
    trlau_hcapsule_damage: _int_prop('Damage', 0, min=-0x8000, max=0x7FFF)


def get_hinfo_data(obj: Any):
    try:
        return getattr(obj, 'trlau_hinfo_data', None)
    except Exception:
        return None


def _data_has_key(data: Any, key: str) -> bool:
    return data is not None and hasattr(data, key)


def _raw_prop_exists(obj: Any, key: str) -> bool:
    try:
        return key in obj
    except Exception:
        return False


def _delete_raw_prop(obj: Any, key: str) -> None:
    try:
        if key in obj:
            del obj[key]
    except Exception:
        pass


def _raw_prop_value(obj: Any, key: str, default: Any = None) -> Any:
    try:
        return obj.get(key, default)
    except Exception:
        return default


def cleanup_hinfo_custom_props(obj: Any, extra_keys: tuple[str, ...] = ()) -> None:
    """Remove old HInfo panel-data ID properties after typed data is available."""
    data = get_hinfo_data(obj)
    if data is None:
        return
    for key in tuple(HINFO_MIGRATED_CUSTOM_PROP_KEYS) + tuple(extra_keys):
        _delete_raw_prop(obj, key)


def set_hinfo_component_type(obj: Any, component_type: str) -> None:
    data = get_hinfo_data(obj)
    if data is not None:
        try:
            data.trlau_hinfo_component_type = str(component_type or '')
        except Exception:
            pass


def set_hinfo_int_prop(obj: Any, key: str, value: Any) -> None:
    data = get_hinfo_data(obj)
    if _data_has_key(data, key):
        try:
            setattr(data, key, int(value))
        except Exception:
            try:
                setattr(data, key, value)
            except Exception:
                pass
        _delete_raw_prop(obj, key)
        return
    try:
        obj[key] = int(value)
    except Exception:
        try:
            obj[key] = value
        except Exception:
            pass


def _set_prop(obj: Any, key: str, value: Any) -> None:
    set_hinfo_int_prop(obj, key, value)


def _get_attr_int(source: Any, name: str, default: int = 0) -> int:
    try:
        value = getattr(source, name)
    except Exception:
        value = default
    try:
        return int(value)
    except Exception:
        return int(default)


def set_hmarker_data_props(obj: Any, marker: Any) -> None:
    set_hinfo_component_type(obj, 'HMarker')
    _set_prop(obj, 'trlau_marker_index', _get_attr_int(marker, 'index', 0))
    _set_prop(obj, 'trlau_owner_segment', _get_attr_int(marker, 'owner_segment', 0))
    _set_prop(obj, 'trlau_marker_bone', _get_attr_int(marker, 'bone', _get_attr_int(marker, 'owner_segment', 0)))
    cleanup_hinfo_custom_props(obj)


def set_hsphere_data_props(obj: Any, sphere: Any) -> None:
    set_hinfo_component_type(obj, 'HSphere')
    _set_prop(obj, 'trlau_hsphere_flags', _get_attr_int(sphere, 'flags', 4352))
    _set_prop(obj, 'trlau_hsphere_id', _get_attr_int(sphere, 'id', 0))
    _set_prop(obj, 'trlau_hsphere_rank', _get_attr_int(sphere, 'rank', 0))
    _set_prop(obj, 'trlau_hsphere_radius_sq', _get_attr_int(sphere, 'radius_sq', _get_attr_int(sphere, 'radius', 0) ** 2))
    _set_prop(obj, 'trlau_hsphere_mass', _get_attr_int(sphere, 'mass', 100))
    _set_prop(obj, 'trlau_hsphere_buoyancy_factor', _get_attr_int(sphere, 'buoyancy_factor', 100))
    _set_prop(obj, 'trlau_hsphere_explosion_factor', _get_attr_int(sphere, 'explosion_factor', 100))
    _set_prop(obj, 'trlau_hsphere_material_type', _get_attr_int(sphere, 'material_type', 0))
    _set_prop(obj, 'trlau_hsphere_pad', _get_attr_int(sphere, 'pad', 0))
    _set_prop(obj, 'trlau_hsphere_damage', _get_attr_int(sphere, 'damage', 0))
    _set_prop(obj, 'trlau_owner_segment', _get_attr_int(sphere, 'owner_segment', 0))
    cleanup_hinfo_custom_props(obj)


def set_hbox_data_props(obj: Any, box: Any) -> None:
    set_hinfo_component_type(obj, 'HBox')
    _set_prop(obj, 'trlau_hbox_flags', _get_attr_int(box, 'flags', 4352))
    _set_prop(obj, 'trlau_hbox_id', _get_attr_int(box, 'id', 0))
    _set_prop(obj, 'trlau_hbox_rank', _get_attr_int(box, 'rank', 0))
    _set_prop(obj, 'trlau_hbox_mass', _get_attr_int(box, 'mass', 100))
    _set_prop(obj, 'trlau_hbox_buoyancy_factor', _get_attr_int(box, 'buoyancy_factor', 100))
    _set_prop(obj, 'trlau_hbox_explosion_factor', _get_attr_int(box, 'explosion_factor', 100))
    _set_prop(obj, 'trlau_hbox_material_type', _get_attr_int(box, 'material_type', 0))
    _set_prop(obj, 'trlau_hbox_pad', _get_attr_int(box, 'pad', 0))
    _set_prop(obj, 'trlau_hbox_damage', _get_attr_int(box, 'damage', 0))
    _set_prop(obj, 'trlau_owner_segment', _get_attr_int(box, 'owner_segment', 0))
    cleanup_hinfo_custom_props(obj)


def set_hcapsule_data_props(obj: Any, capsule: Any) -> None:
    set_hinfo_component_type(obj, 'HCapsule')
    _set_prop(obj, 'trlau_hcapsule_flags', _get_attr_int(capsule, 'flags', 4352))
    _set_prop(obj, 'trlau_hcapsule_id', _get_attr_int(capsule, 'id', 0))
    _set_prop(obj, 'trlau_hcapsule_rank', _get_attr_int(capsule, 'rank', 0))
    _set_prop(obj, 'trlau_hcapsule_mass', _get_attr_int(capsule, 'mass', 100))
    _set_prop(obj, 'trlau_hcapsule_buoyancy_factor', _get_attr_int(capsule, 'buoyancy_factor', 100))
    _set_prop(obj, 'trlau_hcapsule_explosion_factor', _get_attr_int(capsule, 'explosion_factor', 100))
    _set_prop(obj, 'trlau_hcapsule_material_type', _get_attr_int(capsule, 'material_type', 0))
    _set_prop(obj, 'trlau_hcapsule_pad', _get_attr_int(capsule, 'pad', 0))
    _set_prop(obj, 'trlau_hcapsule_damage', _get_attr_int(capsule, 'damage', 0))
    _set_prop(obj, 'trlau_owner_segment', _get_attr_int(capsule, 'owner_segment', 0))
    cleanup_hinfo_custom_props(obj)


def ensure_hinfo_data_props(obj: Any, component_type: str, defaults: Mapping[str, int] | None = None) -> None:
    component_type = str(component_type or '')
    fields = HINFO_DATA_FIELD_DEFS.get(component_type, ())
    supplied = defaults or {}
    data = get_hinfo_data(obj)

    if data is None:
        for key, _label, default in fields:
            if not _raw_prop_exists(obj, key):
                _set_prop(obj, key, supplied.get(key, default))
        return

    raw_values = {key: _raw_prop_value(obj, key) for key, _label, _default in fields if _raw_prop_exists(obj, key)}
    current_component = str(getattr(data, 'trlau_hinfo_component_type', '') or '')
    initialize = current_component != component_type

    for key, _label, default in fields:
        if not _data_has_key(data, key):
            continue
        if key in raw_values:
            value = raw_values[key]
        elif initialize:
            value = supplied.get(key, default)
        else:
            continue
        try:
            setattr(data, key, int(value))
        except Exception:
            try:
                setattr(data, key, value)
            except Exception:
                pass

    if initialize:
        set_hinfo_component_type(obj, component_type)
    cleanup_hinfo_custom_props(obj)


def get_hinfo_int_prop(obj: Any, key: str, default: int = 0, *, min_value: int | None = None, max_value: int | None = None) -> int:
    data = get_hinfo_data(obj)
    if _data_has_key(data, key):
        if _raw_prop_exists(obj, key):
            try:
                setattr(data, key, int(_raw_prop_value(obj, key, default)))
            except Exception:
                pass
            _delete_raw_prop(obj, key)
        try:
            value = getattr(data, key)
        except Exception:
            value = default
    else:
        value = _raw_prop_value(obj, key, default)
    try:
        result = int(value)
    except Exception:
        result = int(default)
    if min_value is not None:
        result = max(int(min_value), result)
    if max_value is not None:
        result = min(int(max_value), result)
    return result
