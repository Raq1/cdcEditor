from __future__ import annotations

import bpy

from . import archive_panel, cloth_panel, hinfo_panel, level_panel, mface_panel, model_panel, multiplex_panel, utilities_panel
from .level_properties import (
    LEVEL_GAME_ITEMS,
    LEVEL_METADATA_GROUPS,
    MARKUP_TYPE_ITEMS,
    MODEL_TARGET_FLAG_ITEMS,
    REWARD_TYPE_ITEMS,
    TERRAIN_GROUP_FLAG_ITEMS,
    _get_intro_reward_type_enum,
    _get_level_game_enum,
    _get_markup_type_enum,
    _get_target_flags_enum,
    _get_terrain_group_flags_enum,
    _set_intro_reward_type_enum,
    _set_level_game_enum,
    _set_markup_type_enum,
    _set_target_flags_enum,
    _set_terrain_group_flags_enum,
)
from .level_panel import (
    LEVEL_METADATA_TAB_ITEMS,
    _get_terrain_light_multiplier_ui,
    _get_terrain_light_radius_ui,
    _set_terrain_light_multiplier_ui,
    _set_terrain_light_radius_ui,
)


classes = (
    model_panel.TRLAU_PG_bone_mirror_entry,
    multiplex_panel.TRLAU_PG_multiplex_anchor,
    level_panel.TRLAU_PG_multispline_key,
    level_panel.TRLAU_PG_multispline_spline,
    level_panel.TRLAU_PG_multispline_state,
    level_panel.TRLAU_PG_multispline,
    level_panel.TRLAU_PG_markup_data,
    level_panel.TRLAU_PG_intro_data,
    level_panel.TRLAU_PG_intro_generic_data,
    level_panel.TRLAU_PG_intro_specific_data,
    level_panel.TRLAU_PG_intro_sound_data,
    level_panel.TRLAU_OT_multispline_add_key,
    level_panel.TRLAU_OT_multispline_remove_key,
    multiplex_panel.TRLAU_PG_multiplex_skeleton_binding,
    level_panel.TRLAU_PG_signal_spline_camera_level_data,
    level_panel.TRLAU_PG_signal_spline_camera_pointer,
    level_panel.TRLAU_PG_signal_attack_wave_limit,
    level_panel.TRLAU_PG_signal_attack_wave_box_limit,
    level_panel.TRLAU_PG_signal_attack_wave_plane_limit,
    level_panel.TRLAU_PG_signal_attack_wave_run_and_gun_pos,
    level_panel.TRLAU_PG_signal_attack_wave_patrol_pos,
    level_panel.TRLAU_PG_signal_attack_wave_message,
    level_panel.TRLAU_PG_signal_attack_wave_runtime,
    level_panel.TRLAU_PG_level_attack_wave_definition,
    level_panel.TRLAU_PG_level_attack_wave_entry,
    level_panel.TRLAU_PG_level_pmarker_entry,
    level_panel.TRLAU_PG_level_combat_data,
    level_panel.TRLAU_PG_signal_data,
    hinfo_panel.TRLAU_OT_snap_object_to_hmarker,
    hinfo_panel.TRLAU_OT_import_model_hinfo,
    hinfo_panel.TRLAU_OT_import_model_bone_mirrors,
    cloth_panel.TRLAU_OT_set_cloth_bone_state,
    cloth_panel.TRLAU_OT_add_cloth_point,
    cloth_panel.TRLAU_OT_add_cloth_capsule,
    cloth_panel.TRLAU_OT_align_cloth_chain_to_axis,
    cloth_panel.TRLAU_OT_apply_cloth_pose_as_rest,
    cloth_panel.TRLAU_OT_create_cloth_from_active_bone,
    cloth_panel.TRLAU_OT_toggle_cloth_preview,
    cloth_panel.TRLAU_OT_add_cloth_plane_rule_from_hmarkers,
    cloth_panel.TRLAU_OT_fit_cloth_plane_rule_selector,
    cloth_panel.TRLAU_OT_rebuild_cloth_plane_rule_visuals,
    cloth_panel.TRLAU_OT_validate_cloth_plane_rules,
    mface_panel.TRLAU_OT_create_mface,
    mface_panel.TRLAU_OT_create_mface_top_to_bottom,
    mface_panel.TRLAU_OT_delete_mface,
    archive_panel.TRLAU_PG_bigfile_record_entry,
    archive_panel.TRLAU_PG_archive_tools,
    archive_panel.TRLAU_UL_bigfile_records,
    archive_panel.TRLAU_OT_select_bigfile_path,
    archive_panel.TRLAU_OT_select_bigfile_output_path,
    archive_panel.TRLAU_OT_select_bigfile_filelist_path,
    archive_panel.TRLAU_OT_set_bigfile_filelist,
    archive_panel.TRLAU_MT_bigfile_filelists,
    archive_panel.TRLAU_OT_toggle_bigfile_folder,
    archive_panel.TRLAU_OT_open_bigfile_browser,
    archive_panel.TRLAU_OT_select_all_bigfile_records,
    archive_panel.TRLAU_OT_unpack_selected_bigfile_records,
    archive_panel.TRLAU_OT_import_selected_bigfile_records,
    archive_panel.TRLAU_OT_export_selected_bigfile_drms,
    archive_panel.TRLAU_OT_create_bigfile_backup,
    archive_panel.TRLAU_OT_unpack_bigfile,
    archive_panel.TRLAU_OT_repack_bigfile,
    multiplex_panel.TRLAU_OT_export_multiplex_mul,
    multiplex_panel.TRLAU_OT_toggle_multiplex_audio,
    multiplex_panel.TRLAU_OT_apply_multiplex_skeleton_to_armature,
    multiplex_panel.TRLAU_OT_load_all_multiplex_animations,
    hinfo_panel.TRLAU_OT_add_hinfo_component,
    hinfo_panel.TRLAU_OT_toggle_component_visibility,
    utilities_panel.TRLAU_OT_batch_replace_wave_audio,
    utilities_panel.TRLAU_OT_toggle_drawgroup_visibility,
    utilities_panel.TRLAU_OT_add_vertex_color,
    utilities_panel.TRLAU_OT_remove_vertex_color,
    utilities_panel.TRLAU_OT_rename_bones,
    utilities_panel.TRLAU_OT_make_bones_contiguous,
    utilities_panel.TRLAU_OT_replace_mul_skeleton,
    utilities_panel.TRLAU_OT_reload_textures,
    utilities_panel.TRLAU_OT_assign_texture_ids,
    level_panel.TRLAU_OT_fsfx_decode_stored_blob,
    level_panel.TRLAU_OT_reverse_markup_direction,
    level_panel.TRLAU_OT_add_level_terrain_group,
    level_panel.TRLAU_OT_add_level_markup,
    level_panel.TRLAU_OT_add_level_light,
    level_panel.TRLAU_OT_add_level_intro_data,
    level_panel.VIEW3D_PT_trlau_level_editor,
    model_panel.VIEW3D_PT_trlau_model,
    model_panel.VIEW3D_PT_trlau_model_bone_mirroring,
    multiplex_panel.VIEW3D_PT_trlau_multiplex_header,
    multiplex_panel.VIEW3D_PT_trlau_multiplex,
    cloth_panel.VIEW3D_PT_trlau_cloth,
    mface_panel.VIEW3D_PT_trlau_mface,
    hinfo_panel.VIEW3D_PT_trlau_editor_hinfo,
    hinfo_panel.VIEW3D_PT_trlau_editor_target,
    level_panel.VIEW3D_PT_trlau_level_editing,
    level_panel.VIEW3D_PT_trlau_level_metadata,
    archive_panel.VIEW3D_PT_trlau_archive_tools,
    utilities_panel.VIEW3D_PT_trlau_utilities,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    cloth_panel._hide_cloth_collision_viewport_names()
    bpy.types.Scene.trlau_level_metadata_tab = bpy.props.EnumProperty(
        name='Level Metadata Tab',
        items=LEVEL_METADATA_TAB_ITEMS,
        default=LEVEL_METADATA_GROUPS[0][0],
        options={'SKIP_SAVE'},
    )
    bpy.types.Object.trlau_level_game = bpy.props.EnumProperty(
        name='Game',
        description='Target game format for TRLAU level export',
        items=LEVEL_GAME_ITEMS,
        get=_get_level_game_enum,
        set=_set_level_game_enum,
    )
    bpy.types.Object.trlau_model_game = bpy.props.EnumProperty(
        name='Game',
        description='Target game format for TRLAU model import/export metadata',
        items=model_panel.MODEL_GAME_ITEMS,
        get=model_panel._get_model_game_enum,
        set=model_panel._set_model_game_enum,
    )
    bpy.types.Object.trlau_intro_reward_type_ui = bpy.props.EnumProperty(
        name='rewardType',
        description='RewardIntroData rewardType enum',
        items=REWARD_TYPE_ITEMS,
        get=_get_intro_reward_type_enum,
        set=_set_intro_reward_type_enum,
    )
    bpy.types.Object.trlau_markup_type_ui = bpy.props.EnumProperty(
        name='Markup Type',
        description='Editable MarkUp movement type',
        items=MARKUP_TYPE_ITEMS,
        get=_get_markup_type_enum,
        set=_set_markup_type_enum,
    )
    bpy.types.Object.trlau_markup_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_markup_data)
    bpy.types.Object.trlau_intro_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_intro_data)
    bpy.types.Object.trlau_intro_generic_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_intro_generic_data)
    bpy.types.Object.trlau_intro_specific_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_intro_specific_data)
    bpy.types.Object.trlau_intro_sound_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_intro_sound_data)
    bpy.types.Object.trlau_terrain_group_flags_ui = bpy.props.EnumProperty(
        name='TerrainGroup Flags',
        description='Editable TerrainGroup flags',
        items=TERRAIN_GROUP_FLAG_ITEMS,
        get=_get_terrain_group_flags_enum,
        set=_set_terrain_group_flags_enum,
    )
    bpy.types.Object.trlau_target_flags_ui = bpy.props.EnumProperty(
        name='Target Flags',
        description='Editable ModelTarget flags',
        items=MODEL_TARGET_FLAG_ITEMS,
        options={'ENUM_FLAG'},
        get=_get_target_flags_enum,
        set=_set_target_flags_enum,
    )
    bpy.types.Object.trlau_terrain_light_radius_ui = bpy.props.IntProperty(
        name='Radius',
        description='TerrainLight radius, synchronized with Light Shadow Soft Size using radius = shadow_soft_size * 10',
        min=0,
        get=_get_terrain_light_radius_ui,
        set=_set_terrain_light_radius_ui,
    )
    bpy.types.Object.trlau_terrain_light_multiplier_ui = bpy.props.IntProperty(
        name='Multiplier',
        description='TerrainLight multiplier, synchronized with Light Power using multiplier = power / 100',
        get=_get_terrain_light_multiplier_ui,
        set=_set_terrain_light_multiplier_ui,
    )
    bpy.types.Object.trlau_signal_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_signal_data)
    bpy.types.Object.trlau_combat_data = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_level_combat_data)
    bpy.types.Object.trlau_multi_spline = bpy.props.PointerProperty(type=level_panel.TRLAU_PG_multispline)
    bpy.types.Object.trlau_mul_sz_name_ui = bpy.props.StringProperty(
        name='szName',
        description='MUL header szName value stored for export',
        get=multiplex_panel._get_multiplex_sz_name,
        set=multiplex_panel._set_multiplex_sz_name,
    )
    bpy.types.Object.trlau_mul_main_unit_id_ui = bpy.props.IntProperty(
        name='mainUnitID',
        description='MUL header mainUnitID value stored for export',
        get=multiplex_panel._get_multiplex_main_unit_id,
        set=multiplex_panel._set_multiplex_main_unit_id,
    )
    bpy.types.Object.trlau_mul_preserve_bone_positions = bpy.props.BoolProperty(
        name='Retarget Bone Positions',
        description='When loading MUL skeleton animation, import location keys as retargeted deltas relative to the selected armature proportions, instead of copying source bone placement literally',
        default=False,
    )
    bpy.types.Object.trlau_mul_audio_volume_ui = bpy.props.FloatProperty(
        name='Volume',
        description='Preview volume for the linked MUL audio strip',
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
        default=1.0,
        get=multiplex_panel._get_multiplex_audio_volume,
        set=multiplex_panel._set_multiplex_audio_volume,
    )
    bpy.types.Object.trlau_mul_subtitle_preview_enabled = bpy.props.BoolProperty(
        name='Show Subtitles in Viewport',
        description='Preview the editable MUL subtitle dialogue in the 3D Viewport while scrubbing or playing the timeline',
        default=True,
        update=lambda self, context: multiplex_panel._tag_all_view3d_redraws(),
    )
    bpy.types.Object.trlau_mul_subtitle_preview_font_size = bpy.props.IntProperty(
        name='Font Size',
        description='Font size in pixels for MUL subtitle preview text in the 3D Viewport',
        min=8,
        max=128,
        soft_min=12,
        soft_max=64,
        default=24,
        update=lambda self, context: multiplex_panel._tag_all_view3d_redraws(),
    )
    bpy.types.Object.trlau_mul_subtitle_preview_language = bpy.props.EnumProperty(
        name='Subtitle Language',
        description='Language to preview in the 3D Viewport',
        items=(
            ('0', 'English', 'Preview English subtitles'),
            ('1', 'French', 'Preview French subtitles'),
            ('2', 'German', 'Preview German subtitles'),
            ('3', 'Italian', 'Preview Italian subtitles'),
            ('4', 'Spanish', 'Preview Spanish subtitles'),
            ('7', 'Polish', 'Preview Polish subtitles'),
            ('9', 'Russian', 'Preview Russian subtitles'),
            ('10', 'Czech', 'Preview Czech subtitles'),
            ('*', 'All Languages', 'Preview all subtitle languages present on the current subtitle frame'),
            ('AUTO', 'Automatic', 'Use the first subtitle language present on each subtitle frame'),
        ),
        default='0',
        update=lambda self, context: multiplex_panel._tag_all_view3d_redraws(),
    )
    bpy.types.Object.trlau_multiplex_anchors = bpy.props.CollectionProperty(
        type=multiplex_panel.TRLAU_PG_multiplex_anchor
    )
    bpy.types.Object.trlau_multiplex_skeleton_bindings = bpy.props.CollectionProperty(
        type=multiplex_panel.TRLAU_PG_multiplex_skeleton_binding
    )
    level_panel.register_fsfx_panel_properties()
    level_panel.register_sfx_panel_properties()
    bpy.types.Scene.trlau_archive_tools = bpy.props.PointerProperty(type=archive_panel.TRLAU_PG_archive_tools)
    bpy.types.Scene.trlau_texture_id_assignment = bpy.props.BoolProperty(
        name='Texture ID Assignment',
        description='Keep the current material node assignments and bind the Texture ID to them instead of auto-reloading by ID',
        default=False,
    )
    model_panel.register_model_properties()
    cloth_panel.register_cloth_panel_properties()
    multiplex_panel.register_subtitle_viewport_preview()


def unregister():
    multiplex_panel.unregister_subtitle_viewport_preview()
    cloth_panel.unregister_cloth_panel_properties()
    model_panel.unregister_model_properties()
    if hasattr(bpy.types.Scene, 'trlau_texture_id_assignment'):
        del bpy.types.Scene.trlau_texture_id_assignment
    if hasattr(bpy.types.Scene, 'trlau_editor_hinfo_expanded'):
        del bpy.types.Scene.trlau_editor_hinfo_expanded
    if hasattr(bpy.types.Scene, 'trlau_level_metadata_tab'):
        del bpy.types.Scene.trlau_level_metadata_tab
    if hasattr(bpy.types.Object, 'trlau_level_game'):
        del bpy.types.Object.trlau_level_game
    if hasattr(bpy.types.Object, 'trlau_model_game'):
        del bpy.types.Object.trlau_model_game
    if hasattr(bpy.types.Object, 'trlau_intro_reward_type_ui'):
        del bpy.types.Object.trlau_intro_reward_type_ui
    if hasattr(bpy.types.Object, 'trlau_intro_sound_data'):
        del bpy.types.Object.trlau_intro_sound_data
    if hasattr(bpy.types.Object, 'trlau_intro_specific_data'):
        del bpy.types.Object.trlau_intro_specific_data
    if hasattr(bpy.types.Object, 'trlau_intro_generic_data'):
        del bpy.types.Object.trlau_intro_generic_data
    if hasattr(bpy.types.Object, 'trlau_intro_data'):
        del bpy.types.Object.trlau_intro_data
    if hasattr(bpy.types.Object, 'trlau_markup_data'):
        del bpy.types.Object.trlau_markup_data
    if hasattr(bpy.types.Object, 'trlau_markup_type_ui'):
        del bpy.types.Object.trlau_markup_type_ui
    if hasattr(bpy.types.Object, 'trlau_target_flags_ui'):
        del bpy.types.Object.trlau_target_flags_ui
    if hasattr(bpy.types.Object, 'trlau_terrain_group_flags_ui'):
        del bpy.types.Object.trlau_terrain_group_flags_ui
    if hasattr(bpy.types.Object, 'trlau_terrain_group_type_ui'):
        del bpy.types.Object.trlau_terrain_group_type_ui
    if hasattr(bpy.types.Object, 'trlau_terrain_light_radius_ui'):
        del bpy.types.Object.trlau_terrain_light_radius_ui
    if hasattr(bpy.types.Object, 'trlau_terrain_light_multiplier_ui'):
        del bpy.types.Object.trlau_terrain_light_multiplier_ui
    level_panel.unregister_sfx_panel_properties()
    level_panel.unregister_fsfx_panel_properties()
    for attr in (
        'trlau_archive_tools',
    ):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    if hasattr(bpy.types.Object, 'trlau_multiplex_skeleton_bindings'):
        del bpy.types.Object.trlau_multiplex_skeleton_bindings
    if hasattr(bpy.types.Object, 'trlau_multiplex_anchors'):
        del bpy.types.Object.trlau_multiplex_anchors
    if hasattr(bpy.types.Object, 'trlau_mul_subtitle_preview_font_size'):
        del bpy.types.Object.trlau_mul_subtitle_preview_font_size
    if hasattr(bpy.types.Object, 'trlau_mul_subtitle_preview_language'):
        del bpy.types.Object.trlau_mul_subtitle_preview_language
    if hasattr(bpy.types.Object, 'trlau_mul_subtitle_preview_enabled'):
        del bpy.types.Object.trlau_mul_subtitle_preview_enabled
    if hasattr(bpy.types.Object, 'trlau_mul_audio_volume_ui'):
        del bpy.types.Object.trlau_mul_audio_volume_ui
    if hasattr(bpy.types.Object, 'trlau_mul_preserve_bone_positions'):
        del bpy.types.Object.trlau_mul_preserve_bone_positions
    if hasattr(bpy.types.Object, 'trlau_mul_main_unit_id_ui'):
        del bpy.types.Object.trlau_mul_main_unit_id_ui
    if hasattr(bpy.types.Object, 'trlau_mul_sz_name_ui'):
        del bpy.types.Object.trlau_mul_sz_name_ui
    if hasattr(bpy.types.Object, 'trlau_multi_spline'):
        del bpy.types.Object.trlau_multi_spline
    if hasattr(bpy.types.Object, 'trlau_combat_data'):
        del bpy.types.Object.trlau_combat_data
    if hasattr(bpy.types.Object, 'trlau_signal_data'):
        del bpy.types.Object.trlau_signal_data
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
