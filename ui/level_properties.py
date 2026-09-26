from __future__ import annotations

from ..core.game_utils import normalize_game_value


INTRO_GENERIC_FIELDS = (
    ('in_view_remove_dist', 'InViewRemoveDist'),
    ('out_of_view_remove_dist', 'OutOfViewRemoveDist'),
    ('use_model', 'UseModel'),
    ('pad0', 'pad0'),
    ('pad1', 'pad1'),
    ('flags', 'Flags'),
    ('attached_instance', 'AttachedInstance'),
    ('swing_length', 'SwingLength'),
    ('dtp_camera_id', 'DTPCameraID'),
)
INTRO_SPECIFIC_FIELDS = {
    'RewardIntroData': (('reward_type', 'rewardType'), ('unique_id', 'uniqueID'), ('sound_id', 'soundID')),
    'RopeObjIntroData': (
        ('top_connect_type', 'topConnectType'), ('bottom_connect_type', 'bottomConnectType'),
        ('top_connect_instance', 'topConnectInstance'), ('top_connect_model_index', 'topConnectModelIndex'),
        ('top_connect_model_marker_index', 'topConnectModelMarkerIndex'), ('bottom_connect_instance', 'bottomConnectInstance'),
        ('bottom_connect_model_index', 'bottomConnectModelIndex'), ('bottom_connect_model_marker_index', 'bottomConnectModelMarkerIndex'),
        ('collision_plane_instance0', 'collisionPlaneInstance0'), ('collision_plane_instance1', 'collisionPlaneInstance1'),
        ('rope_camera_dtpid', 'ropeCameraDTPID'), ('rope_camera_overrides_movement', 'ropeCameraOverridesMovement'),
        ('sound_input_min', 'soundInputMin'), ('sound_input_max', 'soundInputMax'),
        ('render_width', 'renderWidth'), ('render_length_per_v', 'renderLengthPerV'), ('render_u_width', 'renderUWidth'),
        ('render_color', 'renderColor'), ('render_model', 'renderModel'), ('render_texture_main', 'renderTextureMain'),
        ('render_texture_top', 'renderTextureTop'), ('render_texture_bottom', 'renderTextureBottom'),
    ),
    'WaterVolumeIntroData': (
        ('water_depth', 'waterDepth'), ('water_inflow', 'waterInflow'), ('water_outflow', 'waterOutflow'),
        ('water_speed', 'waterSpeed'), ('flow_radius', 'flowRadius'), ('bob_height', 'bobHeight'),
        ('bob_frequency', 'bobFrequency'), ('bob_grid_size', 'bobGridSize'), ('priority', 'priority'), ('water_flags', 'waterFlags'),
    ),
}

INTRO_SOUND_SFX_FIELDS = (
    ('calltype', 'calltype'),
    ('filename', 'filename'),
    ('priority', 'priority'),
    ('volume', 'volume'),
    ('volume_variation', 'volumeVariation'),
    ('mode', 'mode'),
    ('sound_group', 'soundGroup'),
    ('delay', 'delay'),
    ('loops', 'loops'),
    ('cur_voices', 'curVoices'),
    ('note', 'note'),
    ('armode', 'armode'),
    ('ar', 'ar'),
    ('dr', 'dr'),
    ('sl', 'sl'),
    ('srmode', 'srmode'),
    ('srsign', 'srsign'),
    ('srpad', 'srpad'),
    ('rr', 'rr'),
    ('sr', 'sr'),
    ('rrmode', 'rrmode'),
    ('pitch_variation', 'pitchVariation'),
    ('initial_delay', 'initialDelay'),
    ('initial_delay_variation', 'initialDelayVariation'),
    ('max_distance', 'maxDistance'),
    ('subtitlemode', 'subtitlemode'),
)


SFX_MARKER_FIELDS = (
    ('marker_index', 'index'),
    ('marker_unique_id', 'uniqueID'),
    ('marker_plane', 'plane'),
    ('marker_spline_id', 'splineID'),
    ('marker_source_offset', 'sourceOffset'),
    ('marker_raw_position', 'rawPosition'),
)

SFX_SOUND_BASE_FIELDS = (
    ('kind', 'kind'),
    ('entry_index', 'entryIndex'),
    ('source_offset', 'sourceOffset'),
    ('ids', 'sfxIDs'),
    ('speaker_count', 'speakerCount'),
)

SFX_SOUND_PERIODIC_FIELDS = (
    ('num_sfx_ids', 'numSfxIDs'),
    ('flags', 'flags'),
    ('min_vol_distance', 'minVolDistance'),
    ('pitch', 'pitch'),
    ('pitch_variation', 'pitchVariation'),
    ('max_volume', 'maxVolume'),
    ('max_vol_variation', 'maxVolVariation'),
    ('initial_delay', 'initialDelay'),
    ('initial_delay_variation', 'initialDelayVariation'),
    ('on_time', 'onTime'),
    ('on_time_variation', 'onTimeVariation'),
    ('off_time', 'offTime'),
    ('off_time_variation', 'offTimeVariation'),
)

SFX_SOUND_EVENT_FIELDS = (
    ('sound_group', 'soundGroup'),
    ('num_sfx_ids', 'numSfxIDs'),
    ('min_vol_distance', 'minVolDistance'),
    ('pitch', 'pitch'),
    ('pitch_variation', 'pitchVariation'),
    ('max_volume', 'maxVolume'),
    ('max_vol_variation', 'maxVolVariation'),
    ('delay', 'delay'),
    ('delay_variation', 'delayVariation'),
)

SFX_SOUND_STREAM_FIELDS = (
    ('choose_chance', 'chooseChance'),
    ('play_chance', 'playChance'),
    ('min_vol_distance', 'minVolDistance'),
    ('max_volume', 'maxVolume'),
    ('music_vars', 'musicVars'),
    ('name', 'name'),
)

SFX_SPEAKER_FIELDS = (
    ('wave_id', 'waveID'),
    ('id', 'sfxID'),
)

SFX_REFERENCE_FIELDS = (
    ('sfx_id', 'sfxID'),
    ('section_file', 'sectionFile'),
    ('section_type', 'sectionType'),
    ('calltype', 'calltype'),
    ('wave_id', 'waveID'),
    ('filename', 'filename'),
    ('priority', 'priority'),
    ('volume', 'volume'),
    ('volume_variation', 'volumeVariation'),
    ('mode', 'mode'),
    ('sound_group', 'soundGroup'),
    ('delay', 'delay'),
    ('max_voices', 'maxVoices'),
    ('loops', 'loops'),
    ('cur_voices', 'curVoices'),
    ('note', 'note'),
    ('armode', 'armode'),
    ('ar', 'ar'),
    ('dr', 'dr'),
    ('sl', 'sl'),
    ('srmode', 'srmode'),
    ('srsign', 'srsign'),
    ('srpad', 'srpad'),
    ('rr', 'rr'),
    ('sr', 'sr'),
    ('rrmode', 'rrmode'),
    ('pitch_variation', 'pitchVariation'),
    ('initial_delay', 'initialDelay'),
    ('initial_delay_variation', 'initialDelayVariation'),
    ('max_distance', 'maxDistance'),
    ('pan_position', 'panPosition'),
    ('subtitlemode', 'subtitlemode'),
    ('group_sfx_id', 'groupSfxID'),
    ('group_entry_index', 'groupEntryIndex'),
    ('group_weight', 'groupWeight'),
)

WAVE_METADATA_FIELDS = (
    ('wave_section_id', 'waveSectionID'),
    ('wave_section_file', 'waveSectionFile'),
    ('decoded_wav_path', 'decodedWavPath'),
    ('sample_rate', 'sampleRate'),
    ('loop_start', 'loopStart'),
    ('loop_end', 'loopEnd'),
    ('adpcm_block_count', 'adpcmBlockCount'),
    ('sample_count', 'sampleCount'),
)

REWARD_TYPE_ITEMS = (
    ('kBronzeReward', 'kBronzeReward', '0: Bronze reward'),
    ('kSilverReward', 'kSilverReward', '1: Silver reward'),
    ('kGoldReward', 'kGoldReward', '2: Gold reward'),
    ('kCommentaryMarker', 'kCommentaryMarker', '3: Commentary marker'),
)

MARKUP_TYPE_DEFS = (
    ('LEDGE', 'Ledge', '1073741824', 1073741824),
    ('LADDER', 'Ladder', '4194304', 4194304),
    ('VERTICAL_POLE', 'Vertical Pole', '67108864', 67108864),
    ('WALL_VERTICAL_POLE', 'Wall Vertical Pole', '134217728', 134217728),
    ('HORIZONTAL_POLE', 'Horizontal Pole', '33554432', 33554432),
    ('ZIPLINE', 'Zipline', '16777216', 16777216),
    ('PERCH', 'Perch', '262144', 262144),
    ('WATER', 'Water', '2147483648', 2147483648),
)
MARKUP_TYPE_ITEMS = tuple((identifier, label, description, 0, index) for index, (identifier, label, description, _value) in enumerate(MARKUP_TYPE_DEFS))
MARKUP_TYPE_VALUES = tuple(int(value) for _identifier, _label, _description, value in MARKUP_TYPE_DEFS)
MARKUP_FLAG_LEDGE = 1073741824
MARKUP_FLAG_PERCH = 262144
MARKUP_FLAG_WATER = 2147483648
MARKUP_BBOX_FLAGS = MARKUP_FLAG_PERCH | MARKUP_FLAG_WATER

TERRAIN_GROUP_FLAG_DEFS = (
    ('PLAYER_COLLISION', 'Player Collision', '0', 0),
    ('ENEMY_COLLISION', 'Enemy collision', '1610829856', 1610829856),
    ('SKYBOX', 'Skybox', '19922944', 19922944),
    ('WATER', 'Water', '64', 64),
)
TERRAIN_GROUP_FLAG_ITEMS = tuple((identifier, label, description, 0, index) for index, (identifier, label, description, _value) in enumerate(TERRAIN_GROUP_FLAG_DEFS))
TERRAIN_GROUP_FLAG_VALUES = tuple(int(value) for _identifier, _label, _description, value in TERRAIN_GROUP_FLAG_DEFS)


MODEL_TARGET_FLAG_DEFS = (
    ('COMBAT_TARGET', 'Combat Target', '0x1', 0x1),
    ('INCIDENTAL_TARGET', 'Incidental Target', '0x2', 0x2),
    ('GRAPPLE_TUG_TARGET', 'Grapple Tug Target', '0x4', 0x4),
    ('GRAPPLE_SWING_TARGET', 'Grapple Swing Target', '0x8', 0x8),
    ('INTERACT_TARGET', 'Interact Target', '0x10', 0x10),
    ('MAG_GUN_TARGET', 'MagGun Target', '0x20', 0x20),
    ('FORCED_INCIDENTAL_TARGET', 'Forced Incidental Target', '0x40', 0x40),
    ('AIM_ASSIST_TARGET', 'Aim Assist Target', '0x80', 0x80),
)
MODEL_TARGET_FLAG_ITEMS = tuple((identifier, label, description, 0, int(value)) for identifier, label, description, value in MODEL_TARGET_FLAG_DEFS)
MODEL_TARGET_FLAG_MASK = 0
for _identifier, _label, _description, _value in MODEL_TARGET_FLAG_DEFS:
    MODEL_TARGET_FLAG_MASK |= int(_value)


def _get_intro_reward_type_enum(obj):
    try:
        data = getattr(obj, 'trlau_intro_specific_data', None)
        if data is not None and bool(getattr(data, 'enabled', False)):
            value = int(getattr(data, 'reward_type', 0) or 0)
            return value if 0 <= value < len(REWARD_TYPE_ITEMS) else 0
    except Exception:
        pass
    return 0


def _set_intro_reward_type_enum(obj, value):
    try:
        value = int(value)
    except Exception:
        value = 0
    value = value if 0 <= value < len(REWARD_TYPE_ITEMS) else 0
    try:
        data = getattr(obj, 'trlau_intro_specific_data', None)
        if data is not None:
            data.enabled = True
            data.data_type = 17
            data.struct_name = 'RewardIntroData'
            data.reward_type = value
            return
    except Exception:
        pass


def _read_int_prop(obj, key: str, default: int = 0) -> int:
    try:
        return int(obj.get(key, default) or default)
    except Exception:
        return int(default)


def _write_int_prop(obj, key: str, value: int) -> None:
    value = int(value)
    # Blender IDProperties are signed 32-bit for integer storage.  Use a string
    # for high-bit TRLAU flags such as 0x80000000 and coerce back on export.
    if -(2 ** 31) <= value <= (2 ** 31 - 1):
        obj[key] = value
    else:
        obj[key] = str(value)


def _get_enum_index_from_value(value: int, values) -> int:
    try:
        value = int(value)
    except Exception:
        value = 0
    for index, item_value in enumerate(values):
        if int(item_value) == value:
            return index
    return 0


def _get_markup_type_enum(obj):
    return _get_enum_index_from_value(_read_int_prop(obj, 'trlau_markup_flags', 0), MARKUP_TYPE_VALUES)


def _set_markup_type_enum(obj, value):
    try:
        index = int(value)
    except Exception:
        index = 0
    if index < 0 or index >= len(MARKUP_TYPE_VALUES):
        index = 0
    _write_int_prop(obj, 'trlau_markup_flags', MARKUP_TYPE_VALUES[index])


def _get_terrain_group_flags_enum(obj):
    return _get_enum_index_from_value(_read_int_prop(obj, 'trlau_terrain_group_flags', 0), TERRAIN_GROUP_FLAG_VALUES)


def _set_terrain_group_flags_enum(obj, value):
    try:
        index = int(value)
    except Exception:
        index = 0
    if index < 0 or index >= len(TERRAIN_GROUP_FLAG_VALUES):
        index = 0
    _write_int_prop(obj, 'trlau_terrain_group_flags', TERRAIN_GROUP_FLAG_VALUES[index])



def _get_target_flags_enum(obj):
    return _read_int_prop(obj, 'trlau_target_flags', 0) & MODEL_TARGET_FLAG_MASK


def _set_target_flags_enum(obj, value):
    _write_int_prop(obj, 'trlau_target_flags', int(value) & MODEL_TARGET_FLAG_MASK)


LEVEL_METADATA_PROP_PREFIX = 'trlau_level_'
UNIT_DATA_PROP_PREFIX = 'trlau_unitdata_'
ADMD_DATA_PROP_PREFIX = 'trlau_admd_'
LEVEL_METADATA_FIELDS = (
    'waterZLevel',
    'backColorR',
    'backColorG',
    'backColorB',
    'spectralColorR',
    'spectralColorG',
    'spectralColorB',
    'spectralFXAlways',
    'waterColorR',
    'waterColorG',
    'waterColorB',
    'waterBlend',
    'farPlane',
    'fogFar',
    'fogNear',
    'spectralFarPlane',
    'spectralFogFar',
    'spectralFogNear',
    'waterFarPlane',
    'waterFogFar',
    'waterFogNear',
    'UnderwaterFXAlpha',
    'UnderwaterFXMovement',
    'UnderwaterFXSpeed',
    'SpectralFXAlpha',
    'SpectralFXIncrease',
    'SpectralFXCenter',
    'WaterFXAlpha',
    'WaterFXMovement',
    'WaterFXSpeed',
    'WaterFSFX',
    'flags',
    'worldName',
    'unitFlags',
    'guiID',
    'streamUnitID',
    'playerName',
    'playerObjectID',
)

LEVEL_GAME_ITEMS = (
    ('legend', 'Legend', 'Export as Tomb Raider: Legend'),
    ('anniversary', 'Anniversary', 'Export as Tomb Raider: Anniversary'),
)


def _normalize_level_game_value(value) -> str:
    return normalize_game_value(value)


def _get_level_game_enum(obj) -> int:
    game = _normalize_level_game_value(obj.get('trlau_level_game_id', 'legend'))
    for index, item in enumerate(LEVEL_GAME_ITEMS):
        if item[0] == game:
            return index
    return 0


def _set_level_game_enum(obj, value: int) -> None:
    index = int(value)
    if index < 0 or index >= len(LEVEL_GAME_ITEMS):
        index = 0
    obj['trlau_level_game_id'] = LEVEL_GAME_ITEMS[index][0]


def _get_model_game_enum(obj) -> int:
    try:
        game = _normalize_level_game_value(getattr(obj, 'trlau_model_game'))
    except Exception:
        game = _normalize_level_game_value(obj.get('trlau_model_game_id', 'legend'))
    for index, item in enumerate(LEVEL_GAME_ITEMS):
        if item[0] == game:
            return index
    return 0


def _set_model_game_enum(obj, value: int) -> None:
    index = int(value)
    if index < 0 or index >= len(LEVEL_GAME_ITEMS):
        index = 0
    try:
        obj.trlau_model_game = LEVEL_GAME_ITEMS[index][0]
    except Exception:
        obj['trlau_model_game_id'] = LEVEL_GAME_ITEMS[index][0]


def _find_related_level_root(obj, context=None):
    current = obj
    while current is not None:
        if bool(current.get('trlau_level_root')):
            return current
        current = getattr(current, 'parent', None)
    collections = list(getattr(obj, 'users_collection', []) or [])
    if context is not None and getattr(context, 'collection', None) is not None:
        collections.append(context.collection)
    for collection in collections:
        for candidate in getattr(collection, 'all_objects', []) or getattr(collection, 'objects', []):
            if bool(candidate.get('trlau_level_root')):
                return candidate
    return None


def _find_related_model_root(obj, context=None):
    current = obj
    while current is not None:
        if bool(getattr(current, 'trlau_is_model_empty', False)):
            return current
        current = getattr(current, 'parent', None)
    collections = list(getattr(obj, 'users_collection', []) or [])
    if context is not None and getattr(context, 'collection', None) is not None:
        collections.append(context.collection)
    for collection in collections:
        for candidate in getattr(collection, 'all_objects', []) or getattr(collection, 'objects', []):
            if bool(getattr(candidate, 'trlau_is_model_empty', False)):
                return candidate
    return None


def _level_game_for_object(obj, context=None) -> str:
    root = _find_related_level_root(obj, context)
    if root is not None:
        return _normalize_level_game_value(root.get('trlau_level_game_id', 'legend'))
    model_root = _find_related_model_root(obj, context)
    if model_root is not None:
        return _normalize_level_game_value(model_root.get('trlau_model_game_id', 'legend'))
    return _normalize_level_game_value(obj.get('trlau_markup_game', obj.get('trlau_level_game_id', obj.get('trlau_model_game_id', 'legend'))))


def _unitdata_field_visible_for_game(group_label: str, field_name: str, game: str) -> bool:
    game = _normalize_level_game_value(game)
    if group_label == 'PSPData' and game != 'anniversary':
        return False
    if field_name == 'base_camera_vehiclecam' and game == 'anniversary':
        return False
    if field_name == 'base_camera_use_camera_stack_system' and game == 'legend':
        return False
    return True


LEVEL_METADATA_GROUPS = (
    ('ENVIRONMENT', 'Env', (
        'waterZLevel',
        'backColorR',
        'backColorG',
        'backColorB',
        'spectralColorR',
        'spectralColorG',
        'spectralColorB',
        'spectralFXAlways',
        'waterColorR',
        'waterColorG',
        'waterColorB',
        'waterBlend',
    )),
    ('PLANES_FOG', 'Fog', (
        'farPlane',
        'fogFar',
        'fogNear',
        'spectralFarPlane',
        'spectralFogFar',
        'spectralFogNear',
        'waterFarPlane',
        'waterFogFar',
        'waterFogNear',
    )),
    ('FX', 'FX', (
        'UnderwaterFXAlpha',
        'UnderwaterFXMovement',
        'UnderwaterFXSpeed',
        'SpectralFXAlpha',
        'SpectralFXIncrease',
        'SpectralFXCenter',
        'WaterFXAlpha',
        'WaterFXMovement',
        'WaterFXSpeed',
        'WaterFSFX',
    )),
    ('ID_FLAGS', 'Data', (
        'flags',
        'worldName',
        'unitFlags',
        'guiID',
        'streamUnitID',
        'playerName',
        'playerObjectID',
    )),
)




def _next_index_for_prop(objects, prop_name: str) -> int:
    max_index = -1
    for obj in objects:
        if prop_name not in obj:
            continue
        try:
            max_index = max(max_index, int(obj.get(prop_name)))
        except Exception:
            continue
    return max_index + 1


def _next_unique_int_for_prop(objects, prop_name: str) -> int:
    used = set()
    for obj in objects:
        if prop_name not in obj:
            continue
        try:
            used.add(int(obj.get(prop_name)))
        except Exception:
            continue
    value = 0
    while value in used:
        value += 1
    return value
