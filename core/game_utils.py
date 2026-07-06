from __future__ import annotations


def normalize_game_value(value: object) -> str:
    if isinstance(value, int):
        return 'anniversary' if int(value) == 1 else 'legend'
    normalized = str(value or '').strip().lower()
    if normalized in {'anniversary', 'tra', 'tr8', '1', 'tombraideranniversary'}:
        return 'anniversary'
    if normalized in {'legend', 'trl', 'tr7', '0', 'tombraiderlegend'}:
        return 'legend'
    if normalized in {'underworld', 'tru', 'tombraiderunderworld'}:
        return 'underworld'
    return 'legend'
