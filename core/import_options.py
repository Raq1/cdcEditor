from __future__ import annotations


def sync_separation_flags(operator, changed: str) -> None:
    if changed == 'drawgroup' and getattr(operator, 'separate_by_drawgroup', False):
        operator.separate_by_material = False
    elif changed == 'material' and getattr(operator, 'separate_by_material', False):
        operator.separate_by_drawgroup = False
