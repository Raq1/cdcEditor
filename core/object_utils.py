from __future__ import annotations

import re


_HINFO_NAME_PATTERNS = (
    ('HMarker', re.compile(r'(^|.*_)HMarker_(?:Missing_)?\d+(?:\.\d+)?$')),
    ('HSphere', re.compile(r'(^|.*_)HSphere_\d+(?:\.\d+)?$')),
    ('HBox', re.compile(r'(^|.*_)HBox_\d+(?:\.\d+)?$')),
    ('HCapsule', re.compile(r'(^|.*_)HCapsule_\d+(?:\.\d+)?$')),
    ('Target', re.compile(r'(^|.*_)Target_\d+(?:\.\d+)?$')),
)

_CLOTH_RULES_RE = re.compile(r'(^|.*_)Cloth_CollisionRules(?:_|$|\.\d+$)')
_CLOTH_RULE_RE = re.compile(r'(^|.*_)Cloth_CollisionRule_\d+_B\d+_P\d+_C\d+(?:\.\d+)?$')
_CLOTH_COLLISION_RE = re.compile(r'(^|.*_)Cloth_Collision_\d+(?:\.\d+)?$')
_CLOTH_PIN_COLLISION_RE = re.compile(r'(^|.*_)Cloth_PinCollision_\d+(?:\.\d+)?$')
_CLOTH_CAPSULE_ENDPOINT_RE = re.compile(r'(^|.*_)Cloth_Capsule_\d+_[AB](?:\.\d+)?$')
_CLOTH_PLANE_RULE_RE = re.compile(r'(^|.*_)Cloth_PlaneRule_\d+(?:_|$|\.\d+$)')
_CLOTH_PLANE_RULE_PLANE_RE = re.compile(r'(^|.*_)Cloth_PlaneRule_\d+_.*_M\d+_[AB](?:\.\d+)?$')
_CLOTH_PLANE_RULE_SELECTOR_RE = re.compile(r'(^|.*_)Cloth_PlaneRule_\d+_.*_Selector(?:\.\d+)?$')

_BLENDER_NUMERIC_SUFFIX_RE = re.compile(r'\.\d{3}$')


def _strip_blender_numeric_suffix(name: str) -> str:
    return _BLENDER_NUMERIC_SUFFIX_RE.sub('', str(name or ''))


_HMARKER_INDEX_RE = re.compile(r'(?:^|_)HMarker_(?:Missing_)?(\d+)(?:\D|$)')


def hmarker_index_from_name(obj_or_name, default: int = 0) -> int:
    try:
        name = str(getattr(obj_or_name, 'name', obj_or_name) or '')
        match = _HMARKER_INDEX_RE.search(_strip_blender_numeric_suffix(name))
        if match is not None:
            return int(match.group(1))
    except Exception:
        pass
    try:
        return int(default)
    except Exception:
        return 0


def is_descendant_of(obj, ancestor) -> bool:
    current = getattr(obj, 'parent', None)
    while current is not None:
        if current == ancestor:
            return True
        current = getattr(current, 'parent', None)
    return False



def _idprop_text(obj, key: str) -> str:
    try:
        return str(obj.get(key, '') or '')
    except Exception:
        return ''


def trlau_object_type(obj) -> str:
    """Return the TRLAU helper/component type without requiring stored ID props.

    Runtime helper objects are identified from their canonical generated names and
    parentage.  The final ID-property read remains for non-helper level/component
    empties that still use trlau_type as their authored component discriminator.
    """
    if obj is None:
        return ''

    name = str(getattr(obj, 'name', '') or '')
    base_name = _strip_blender_numeric_suffix(name)
    parent = getattr(obj, 'parent', None)
    parent_name = str(getattr(parent, 'name', '') or '') if parent is not None else ''

    for component_type, pattern in _HINFO_NAME_PATTERNS:
        if pattern.match(name):
            return component_type

    if base_name.endswith('_Cloth') and getattr(parent, 'type', None) == 'ARMATURE':
        return 'Cloth'
    if _CLOTH_RULE_RE.match(name):
        return 'ClothCollisionRule'
    if _CLOTH_RULES_RE.match(name):
        return 'ClothCollisionRules'
    if _CLOTH_COLLISION_RE.match(name):
        return 'ClothCollision'
    if _CLOTH_PIN_COLLISION_RE.match(name):
        return 'ClothPinCollision'
    if _CLOTH_CAPSULE_ENDPOINT_RE.match(name):
        return 'ClothCapsuleEndpoint'
    if _CLOTH_PLANE_RULE_SELECTOR_RE.match(name):
        return 'ClothPlaneRuleSelector'
    if _CLOTH_PLANE_RULE_PLANE_RE.match(name):
        return 'ClothPlaneRulePlane'
    if _CLOTH_PLANE_RULE_RE.match(name):
        return 'ClothPlaneRule'

    if parent_name.endswith('_HMarkers') and '_HMarker_' in name:
        return 'HMarker'
    if parent_name.endswith('_HSpheres') and '_HSphere_' in name:
        return 'HSphere'
    if parent_name.endswith('_HBoxes') and '_HBox_' in name:
        return 'HBox'
    if parent_name.endswith('_HCapsules') and '_HCapsule_' in name:
        return 'HCapsule'
    if parent_name.endswith('_Targets') and '_Target_' in name:
        return 'Target'

    return _idprop_text(obj, 'trlau_type')
