bl_info = {
    'name': 'cdcEditor',
    'author': 'Raq',
    'version': (0, 1, 1),
    'blender': (5, 0, 0),
    'location': 'File > Import/Export > Tomb Raider LAU Import / Tomb Raider LAU Export',
    'description': 'Toolset for Tomb Raider Legend/Anniversary/Underworld model, level, animation editing and more',
    'collaborators': 'TheIndra, DKDave, Joschka, Che, arc, Edness',
    'category': 'Import-Export',
}

import bpy
from bpy.props import BoolProperty, IntProperty

from . import updater as addon_updater


class CDCEDITOR_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    auto_check_update: BoolProperty(
        name='Automatically check for updates',
        description='Check GitHub Releases for new cdcEditor versions when Blender starts',
        default=False,
    )
    updater_interval_days: IntProperty(
        name='Update check interval',
        description='Minimum number of days between automatic update checks',
        default=7,
        min=1,
        max=365,
    )

    def draw(self, context):
        donate_box = self.layout.box()
        donate_box.label(text='Support cdcEditor')
        donate_button = donate_box.operator(
            'wm.url_open',
            text='Donate on Ko-Fi',
            icon='HEART',
        )
        donate_button.url = 'https://ko-fi.com/raq'

        addon_updater.draw_preferences(self.layout, context)


from .core.hinfo_ui import register_hinfo_properties, sync_hinfo_scene, unregister_hinfo_properties, _timer_sync_hinfo_scene
from .core.material_ui import register_material_properties, unregister_material_properties
from .core.log import configure_logging
from .core.cloth_simulation import register_cloth_simulation, unregister_cloth_simulation
from .operators.import_model import register as register_import_operator
from .operators.import_model import unregister as unregister_import_operator
from .operators.export_level import register as register_export_operator
from .operators.export_level import unregister as unregister_export_operator
from .ui.editor_ui import register as register_editor_ui
from .ui.editor_ui import unregister as unregister_editor_ui
from .ui.material_panel import register as register_material_ui
from .ui.material_panel import unregister as unregister_material_ui

configure_logging()


def register():
    addon_updater.register(bl_info['version'])
    bpy.utils.register_class(CDCEDITOR_AddonPreferences)
    register_hinfo_properties()
    register_material_properties()
    register_import_operator()
    register_export_operator()
    register_editor_ui()
    register_material_ui()
    register_cloth_simulation()
    if sync_hinfo_scene not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(sync_hinfo_scene)
    if not bpy.app.timers.is_registered(_timer_sync_hinfo_scene):
        bpy.app.timers.register(_timer_sync_hinfo_scene, first_interval=0.1, persistent=True)


def unregister():
    if sync_hinfo_scene in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(sync_hinfo_scene)
    if bpy.app.timers.is_registered(_timer_sync_hinfo_scene):
        bpy.app.timers.unregister(_timer_sync_hinfo_scene)
    unregister_cloth_simulation()
    unregister_material_ui()
    unregister_editor_ui()
    unregister_export_operator()
    unregister_import_operator()
    unregister_material_properties()
    unregister_hinfo_properties()
    bpy.utils.unregister_class(CDCEDITOR_AddonPreferences)
    addon_updater.unregister()


__all__ = ['register', 'unregister', 'bl_info']
