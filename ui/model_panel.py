from __future__ import annotations

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty
from ..core.game_utils import normalize_game_value

def _sync_bone_mirror_entries_from_model(model_root, model) -> int:
    if model_root is None:
        raise ValueError('Select a Model empty')
    entries = list(getattr(model, 'bone_mirror_entries', []) or [])
    collection = getattr(model_root, 'trlau_bone_mirror_entries', None)
    if collection is not None:
        collection.clear()
    for entry in entries:
        bone1 = int(getattr(entry, 'bone1', 0))
        bone2 = int(getattr(entry, 'bone2', 0))
        count = int(getattr(entry, 'count', 0))
        if collection is not None:
            item = collection.add()
            item.bone1 = bone1
            item.bone2 = bone2
            item.count = count
    return len(entries)

def _is_model_empty(obj) -> bool:
    return obj is not None and bool(getattr(obj, 'trlau_is_model_empty', False))


MODEL_GAME_ITEMS = (
    ('legend', 'Legend', 'Tomb Raider: Legend'),
    ('anniversary', 'Anniversary', 'Tomb Raider: Anniversary'),
    ('underworld', 'Underworld', 'Tomb Raider: Underworld'),
)


MODEL_PLATFORM_ITEMS = (
    ('pc', 'PC', 'Use the standard PC model exporter'),
    ('ps2', 'PS2', 'Use the PS2 model exporter'),
    ('psp', 'PSP', 'Imported PSP model data'),
    ('ps3', 'PS3', 'Imported PS3 model data'),
    ('xbox', 'Xbox', 'Imported Xbox model data'),
    ('xbox360', 'Xbox 360', 'Imported Xbox 360 model data'),
    ('gamecube', 'Nintendo / GameCube', 'Imported Nintendo/GameCube or Wii model data'),
)


def _model_game_items_by_id() -> dict[str, int]:
    return {item[0]: index for index, item in enumerate(MODEL_GAME_ITEMS)}


def _model_platform_items_by_id() -> dict[str, int]:
    return {item[0]: index for index, item in enumerate(MODEL_PLATFORM_ITEMS)}


def _normalize_model_game_value(value) -> str:
    return normalize_game_value(value)


def _get_model_game_enum(obj) -> int:
    game = _normalize_model_game_value(obj.get('trlau_model_game_id', 'legend'))
    return _model_game_items_by_id().get(game, 0)


def _set_model_game_enum(obj, value: int) -> None:
    try:
        index = int(value)
    except Exception:
        index = 0
    if index < 0 or index >= len(MODEL_GAME_ITEMS):
        index = 0
    obj['trlau_model_game_id'] = MODEL_GAME_ITEMS[index][0]


def _normalize_model_platform_value(value) -> str:
    platform = str(value or 'pc').strip().lower()
    aliases = {
        '': 'pc',
        'default': 'pc',
        'game cube': 'gamecube',
        'gc': 'gamecube',
        'nintendo': 'gamecube',
        'xbox 360': 'xbox360',
        'xb360': 'xbox360',
        'xenon': 'xbox360',
    }
    platform = aliases.get(platform, platform)
    valid = {item[0] for item in MODEL_PLATFORM_ITEMS}
    return platform if platform in valid else 'pc'


def _get_model_platform_enum(obj) -> int:
    platform = _normalize_model_platform_value(obj.get('trlau_model_platform_id', 'pc'))
    return _model_platform_items_by_id().get(platform, 0)


def _set_model_platform_enum(obj, value: int) -> None:
    try:
        index = int(value)
    except Exception:
        index = 0
    if index < 0 or index >= len(MODEL_PLATFORM_ITEMS):
        index = 0
    obj['trlau_model_platform_id'] = MODEL_PLATFORM_ITEMS[index][0]


class TRLAU_PG_bone_mirror_entry(bpy.types.PropertyGroup):
    bone1: IntProperty(name='Bone 1', min=0, max=255)
    bone2: IntProperty(name='Bone 2', min=0, max=255)
    count: IntProperty(name='Count', min=0, max=255)


def draw_model_properties_content(layout, context):
    obj = context.object
    box = layout.box()
    box.label(text='Model Properties')
    box.prop(obj, 'trlau_model_game', text='Game')
    box.prop(obj, 'trlau_model_platform', text='Platform')
    box.prop(obj, 'trlau_ui_cdc_render_data_id')
    box.prop(obj, 'trlau_ui_bone_mirror_data_count')

    import_box = layout.box()
    import_box.operator('trlau.import_model_hinfo', text='Import HInfo')
    import_box.operator('trlau.import_model_bone_mirrors', text='Import Bone Mirrors')


class VIEW3D_PT_trlau_model(bpy.types.Panel):
    bl_label = 'Model'
    bl_idname = 'VIEW3D_PT_trlau_model'
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'

    @classmethod
    def poll(cls, context):
        return _is_model_empty(context.object)

    def draw(self, context):
        draw_model_properties_content(self.layout, context)

class VIEW3D_PT_trlau_model_bone_mirroring(bpy.types.Panel):
    bl_label = 'Bone Mirroring'
    bl_idname = 'VIEW3D_PT_trlau_model_bone_mirroring'
    bl_options = {'DEFAULT_CLOSED'}
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_parent_id = 'VIEW3D_PT_trlau_level_editor'

    @classmethod
    def poll(cls, context):
        return _is_model_empty(context.object)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False
        layout.use_property_decorate = False

        obj = context.object
        entries = getattr(obj, 'trlau_bone_mirror_entries', [])

        if not entries:
            empty = layout.box()
            empty.label(text='No bone mirroring entries', icon='INFO')
            return

        list_box = layout.box()
        list_col = list_box.column(align=True)
        list_col.scale_y = 0.82
        for index, entry in enumerate(entries):
            row = list_col.row(align=True)
            index_col = row.column(align=True)
            index_col.ui_units_x = 2.0
            index_col.label(text=str(index))

            value_row = row.row(align=True)
            value_row.scale_x = 0.9
            value_row.prop(entry, 'bone1', text='')
            value_row.prop(entry, 'bone2', text='')
            value_row.prop(entry, 'count', text='')


def _set_bone_mirror_data_count(self, value):
    target_count = max(0, int(value))
    entries = getattr(self, 'trlau_bone_mirror_entries', None)
    if entries is None:
        return

    current_count = len(entries)
    if target_count > current_count:
        for _ in range(target_count - current_count):
            item = entries.add()
            item.bone1 = 0
            item.bone2 = 0
            item.count = 0
    elif target_count < current_count:
        for _ in range(current_count - target_count):
            entries.remove(len(entries) - 1)



def register_model_properties():
    bpy.types.Object.trlau_ui_bone_mirror_data_count = IntProperty(
        name='Bone Mirrors',
        min=0,
        get=lambda self: len(getattr(self, 'trlau_bone_mirror_entries', [])),
        set=_set_bone_mirror_data_count,
    )
    bpy.types.Object.trlau_ui_cdc_render_data_id = IntProperty(
        name='Render ID',
        default=0,
    )
    bpy.types.Object.trlau_model_platform = EnumProperty(
        name='Platform',
        description='Target hardware/platform format for TRLAU model export',
        items=MODEL_PLATFORM_ITEMS,
        get=_get_model_platform_enum,
        set=_set_model_platform_enum,
    )
    bpy.types.Object.trlau_is_model_empty = BoolProperty(
        name='TRLAU Model Empty',
        default=False,
        options={'HIDDEN'},
    )
    bpy.types.Object.trlau_pc_nextgen_mesh = BoolProperty(
        name='PC Next Gen Mesh', default=False, options={'HIDDEN'},
    )
    bpy.types.Mesh.trlau_pc_nextgen_mesh = BoolProperty(
        name='PC Next Gen Mesh', default=False, options={'HIDDEN'},
    )
    bpy.types.Object.trlau_pc_nextgen_source_section_file = StringProperty(
        name='PC Next Gen Source Section File', default='', options={'HIDDEN'},
    )
    bpy.types.Object.trlau_pc_nextgen_material_count = IntProperty(
        name='PC Next Gen Material Count', default=0, options={'HIDDEN'},
    )
    bpy.types.Object.trlau_pc_nextgen_model_root = StringProperty(
        name='PC Next Gen Model Root', default='', options={'HIDDEN'},
    )
    bpy.types.Object.trlau_pc_nextgen_armature = StringProperty(
        name='PC Next Gen Armature', default='', options={'HIDDEN'},
    )
    bpy.types.Object.trlau_bone_mirror_entries = CollectionProperty(type=TRLAU_PG_bone_mirror_entry)


def unregister_model_properties():
    for name in ('trlau_ui_bone_mirror_data_count', 'trlau_ui_cdc_render_data_id', 'trlau_model_platform', 'trlau_is_model_empty', 'trlau_pc_nextgen_mesh', 'trlau_pc_nextgen_source_section_file', 'trlau_pc_nextgen_material_count', 'trlau_pc_nextgen_model_root', 'trlau_pc_nextgen_armature', 'trlau_bone_mirror_entries'):
        if hasattr(bpy.types.Object, name):
            delattr(bpy.types.Object, name)
    if hasattr(bpy.types.Mesh, 'trlau_pc_nextgen_mesh'):
        delattr(bpy.types.Mesh, 'trlau_pc_nextgen_mesh')


def unregister_armature_properties():
    unregister_model_properties()

classes = (
    TRLAU_PG_bone_mirror_entry,
    VIEW3D_PT_trlau_model,
    VIEW3D_PT_trlau_model_bone_mirroring,
)
