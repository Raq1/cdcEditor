from __future__ import annotations

import traceback
from pathlib import Path

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, StringProperty
from bpy.types import Operator, OperatorFileListElement
from bpy_extras.io_utils import ImportHelper

from ..core.import_options import sync_separation_flags
from ..core.log import configure_logging, logger
from ..platforms.common.drm_container import DRMContainerParser
from ..platforms.pc.area_dbase import AreaDBaseParser
from ..platforms.pc.tr7ae_level import TRLevelParser
from ..platforms.psp.tr7ae_level import TRPSPLevelParser
from ..platforms.pc.tr7ae_object import TRObjectParser
from .animation_import import AnimationImporterMixin
from .import_utils import ImportUtilityMixin
from .level_import import LevelImporterMixin, TRLAU_OT_play_sfx_sound, TRLAU_OT_stop_sfx_sound
from .model_importer import ModelImporterMixin
from .mul_import import MultiplexStreamImporterMixin


class IMPORT_SCENE_OT_trlau_model(
    ImportUtilityMixin,
    ModelImporterMixin,
    LevelImporterMixin,
    AnimationImporterMixin,
    MultiplexStreamImporterMixin,
    Operator,
    ImportHelper,
):

    bl_idname = 'import_scene.trlau_model'
    bl_label = 'Tomb Raider LAU Import'
    bl_description = '\u00A0'
    bl_options = {'UNDO'}

    filename_ext = '.tr7aemesh'
    filter_glob: StringProperty(default='*.obj;*.tr7aemesh;*.tr8mesh;*.drm;*.ani;*.mul;*.gnc', options={'HIDDEN'})
    files: CollectionProperty(type=OperatorFileListElement, options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    show_model_settings: BoolProperty(name='Model Settings', default=True, options={'SKIP_SAVE'})
    show_advanced_settings: BoolProperty(name='Advanced Settings', default=False, options={'SKIP_SAVE'})
    show_level_settings: BoolProperty(name='Level Settings', default=True, options={'SKIP_SAVE'})

    platform: EnumProperty(
        name='Platform',
        items=(
            ('PC', 'PC', ''),
            ('PS2', 'PS2', ''),
            ('PS3', 'PS3', ''),
            ('PSP', 'PSP', ''),
            ('XBOX', 'Xbox', ''),
            ('XBOX360', 'Xbox 360', ''),
            ('GAMECUBE', 'Nintendo', ''),
        ),
        default='PC',
    )
    import_textures: BoolProperty(name='Import Textures', default=True, description='Enable or disable texture importing')
    use_integrated_import_defaults: BoolProperty(name='Use Integrated Import Defaults', default=True, description='Enable the standard full import pass for regular TRLAU imports', options={'HIDDEN', 'SKIP_SAVE'})
    create_import_collection: BoolProperty(name='Create Import Collection', default=True, description='Create or reuse a source-named collection for imported content', options={'HIDDEN', 'SKIP_SAVE'})
    force_unique_import_names: BoolProperty(name='Force Unique Import Names', default=False, description='Always create fresh imported model roots instead of reusing source-named objects', options={'HIDDEN', 'SKIP_SAVE'})
    import_collection_name: StringProperty(name='Import Collection Name', default='', description='Existing collection to receive imported objects when collection creation is disabled', options={'HIDDEN', 'SKIP_SAVE'})
    import_hinfo: BoolProperty(name='Import HInfo', default=True, description='Import HInfo components such as HMarkers, HSpheres, HBoxes, and HCapsules', options={'HIDDEN', 'SKIP_SAVE'})
    import_cloth: BoolProperty(name='Import Cloth', default=True, description='Import ClothSetup data and create cloth bone/collision authoring objects when available', options={'HIDDEN', 'SKIP_SAVE'})
    import_bgobjects: BoolProperty(name='Import BGObjects', default=True, description='Import BGObjects when importing level DRMs', options={'HIDDEN', 'SKIP_SAVE'})
    import_intro_data: BoolProperty(name='Import IntroData', default=True, description='Import IntroData references from level DRMs', options={'HIDDEN', 'SKIP_SAVE'})
    import_camera_data: BoolProperty(name='Import Camera Data', default=True, description='Import CameraData from level DRMs', options={'HIDDEN', 'SKIP_SAVE'})
    import_terrain_lights: BoolProperty(name='Import Terrain Lights', default=True, description='Import TerrainLights from level DRMs', options={'HIDDEN', 'SKIP_SAVE'})
    import_audio: BoolProperty(name='Import Audio', default=True, description='Import PC level SFX markers and decode referenced Wave sections as Blender speakers', options={'HIDDEN', 'SKIP_SAVE'})
    import_collisions: BoolProperty(name='Import Collisions', default=True, description='Import TerrainGroup collision meshes when importing levels', options={'HIDDEN', 'SKIP_SAVE'})
    import_signals: BoolProperty(name='Import Signals', default=True, description='Import the Terrain signal mesh from Terrain.signalTerrainGroup', options={'HIDDEN', 'SKIP_SAVE'})
    import_markups: BoolProperty(name='Import Markups', default=True, description='Import MarkUp polylines as curves for levels and regular models when present', options={'HIDDEN', 'SKIP_SAVE'})
    import_kdnodes: BoolProperty(name='Import KDNodes', default=False, description='Import TerrainGroup collision KDNode empties when importing levels')
    import_area_dbase: BoolProperty(name='Import AreaDBase', default=True, description='Import cdc::AreaDBase movement/navigation areas from level DRMs or standalone .gnc sections', options={'HIDDEN', 'SKIP_SAVE'})
    split_terrain_groups_by_strip: BoolProperty(name='Split TerrainGroups by Strip', default=False, description='Import TerrainGroups as invidual strip meshes')
    import_main_model_only: BoolProperty(name='Main Model Only', default=True, description='Only import the first referenced model (.drm, .obj)')
    import_next_gen_model: BoolProperty(name='Import Next Gen Model', default=False, description='Import the next-generation render model referenced by cdcRenderDataID when present')
    import_all_textures: BoolProperty(name='Import All Textures', default=False, description='Import all textures found in the .drm or folder')
    import_all_animations: BoolProperty(name='Import All Animations', default=False, description='Import all animations found in the .drm or folder')
    import_armature_only: BoolProperty(name='Import Armature Only', default=False, description='Import only the armature')
    import_bounding_boxes: BoolProperty(name='Import Bounding Boxes', default=False, description="Import bones' bounding boxes")
    debug: BoolProperty(name='Debug', default=False, description='Enable console debug log')

    @staticmethod
    def _sync_separation_flags(self, changed: str):
        sync_separation_flags(self, changed)

    def _update_separate_by_drawgroup(self, _context):
        self._sync_separation_flags(self, 'drawgroup')

    def _update_separate_by_material(self, _context):
        self._sync_separation_flags(self, 'material')

    separate_by_drawgroup: BoolProperty(
        name='Separate by Drawgroup',
        default=False,
        description='Split the imported model into separate meshes by drawgroup',
        update=_update_separate_by_drawgroup,
    )
    separate_by_material: BoolProperty(
        name='Separate by Material',
        default=False,
        description='Split the imported model into separate meshes by texture strip/material',
        update=_update_separate_by_material,
    )

    def _enable_integrated_import_defaults(self) -> None:
        for prop_name in (
            'import_hinfo',
            'import_cloth',
            'import_bgobjects',
            'import_intro_data',
            'import_camera_data',
            'import_terrain_lights',
            'import_audio',
            'import_collisions',
            'import_signals',
            'import_markups',
            'import_area_dbase',
        ):
            setattr(self, prop_name, True)

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, 'platform')
        layout.prop(self, 'import_textures')

        model_props = ['import_main_model_only', 'separate_by_drawgroup', 'separate_by_material']
        if self.platform == 'PC':
            model_props.insert(1, 'import_next_gen_model')
        sections = [
            ('show_model_settings', 'Model Settings', tuple(model_props)),
        ]
        if self.platform in {'PC', 'PSP'}:
            props = ('import_kdnodes', 'split_terrain_groups_by_strip') if self.platform == 'PC' else ('split_terrain_groups_by_strip',)
            sections.append(
                ('show_level_settings', 'Level Settings', props)
            )
        sections.append(('show_advanced_settings', 'Advanced Settings', ('import_all_textures', 'import_all_animations', 'import_armature_only')))
        for flag_name, label, props in sections:
            box = layout.box()
            header = box.row()
            expanded = getattr(self, flag_name)
            header.prop(self, flag_name, text=label, icon='DOWNARROW_HLT' if expanded else 'RIGHTARROW_THIN', emboss=True)
            if not expanded:
                continue
            column = box.column(align=True)
            for prop_name in props:
                column.prop(self, prop_name)
            if flag_name == 'show_advanced_settings' and self.platform == 'PC':
                column.prop(self, 'import_bounding_boxes')

        layout.prop(self, 'debug')

    def _animation_ids_from_object_root(self, object_filepath: str) -> list[int]:
        try:
            refs = TRObjectParser(object_filepath).parse_animation_references()
        except Exception as exc:
            logger.debug('Could not read object animation list from %s: %s', object_filepath, exc)
            return []
        return [int(ref.anim_id) for ref in refs if int(ref.anim_id) >= 0]

    def _find_object_root_with_model_refs(self, extracted_paths, *, endian: str = '<') -> Path:
        paths = [Path(path) for path in extracted_paths]
        if not paths:
            raise ValueError('No sections were found in the DRM container')

        ordered: list[Path] = []
        for candidate in paths:
            if candidate.suffix.lower() == '.obj':
                ordered.append(candidate)
        for candidate in paths:
            if candidate not in ordered:
                ordered.append(candidate)

        first_existing = ordered[0]
        for candidate in ordered:
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                refs = TRObjectParser(str(candidate), endian=endian).parse_model_references()
            except Exception as exc:
                logger.debug('Skipping object-root candidate %s: %s', candidate.name, exc)
                continue
            if refs:
                if candidate != first_existing:
                    logger.warning(
                        'Using %s as DRM object root instead of %s because it has %d model reference(s)',
                        candidate.name,
                        first_existing.name,
                        len(refs),
                    )
                return candidate

        raise ValueError('Could not find an extracted object section with model references')

    def _import_underworld_drm_probe(self, context, source_path: Path, object_root_path: Path, collection=None) -> list[dict]:
        refs = TRObjectParser(str(object_root_path)).parse_underworld_mesh_references()
        if not refs:
            raise ValueError(f'No TR8 mesh reference chain was found in {object_root_path.name}')

        collection = collection or self._get_or_create_collection(context, str(source_path), collection_name=source_path.stem)
        selected_refs = refs[:1] if self.import_main_model_only else refs
        results = []
        seen_mesh_paths: set[str] = set()
        for ref in selected_refs:
            tr8mesh_path = Path(ref.tr8mesh_filepath)
            resolved_key = str(tr8mesh_path.resolve())
            if resolved_key in seen_mesh_paths:
                continue
            seen_mesh_paths.add(resolved_key)
            logger.info(
                'Underworld DRM importing tr8mesh: objectref=%s object_id=0x%X object=%s model[%d]=0x%X tr8model=%s cdcModelData=0x%X tr8mesh=%s',
                Path(ref.objectref_filepath).name,
                ref.object_id,
                Path(ref.object_filepath).name,
                ref.model_index,
                ref.tr8model_id,
                Path(ref.tr8model_filepath).name,
                ref.cdc_modeldata_id,
                tr8mesh_path.name,
            )
            imported = self._import_mesh_file(
                context,
                str(tr8mesh_path),
                collection=collection,
                name_prefix=source_path.stem,
                create_model_root=True,
                underworld_model_filepath=str(ref.tr8model_filepath),
            )
            imported['source'] = str(source_path)
            imported['underworld_objectref_path'] = str(ref.objectref_filepath)
            imported['underworld_object_id'] = int(ref.object_id)
            imported['underworld_object_path'] = str(ref.object_filepath)
            imported['underworld_model_index'] = int(ref.model_index)
            imported['underworld_tr8model_id'] = int(ref.tr8model_id)
            imported['underworld_tr8model_path'] = str(ref.tr8model_filepath)
            imported['underworld_cdc_modeldata_id'] = int(ref.cdc_modeldata_id)
            imported['underworld_tr8mesh_path'] = str(tr8mesh_path)
            results.append(imported)
        if not results:
            raise ValueError(f'No importable TR8 mesh was found in {source_path.name}')
        return results

    def _import_single_path(self, context, filepath: str, collection_override=None):
        source_path = Path(filepath)
        suffix = source_path.suffix.lower()

        if suffix == '.drm':
            if self.platform in {'PS3', 'XBOX360'}:
                drm_parser = DRMContainerParser(filepath, endian='>', decompress_derickw=True)
                with drm_parser.temporary_extract_sections() as (extract_dir, extracted_paths, _sections):
                    logger.debug('Extracted %s DRM %s into %s (%d sections)', self.platform, source_path.name, extract_dir, len(extracted_paths))
                    if not extracted_paths:
                        raise ValueError(f'No sections were found in the {self.platform} DRM container')

                    if self.import_all_textures:
                        self._import_all_texture_files(context, extract_dir, platform_hint='xbox360' if self.platform == 'XBOX360' else 'ps3')

                    object_root_path = self._find_object_root_with_model_refs(extracted_paths, endian='>')
                    imported_results = self._import_from_object_refs(context, source_path, str(object_root_path), collection_name=source_path.stem, collection=collection_override, object_endian='>', extracted_paths=extracted_paths, extracted_sections=_sections)
                    if self.import_all_animations:
                        imported_results.extend(
                            self._import_all_animation_files(
                                context,
                                extracted_paths,
                                armatures=[result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None],
                                source_name=source_path.stem,
                                default_endianness='>',
                                platform=self.platform,
                            )
                        )
                    return imported_results

            if self.platform == 'PSP':
                if TRPSPLevelParser.is_level_file(filepath):
                    collection = collection_override or self._get_or_create_collection(context, str(source_path), collection_name=source_path.stem)
                    return self._import_level_file(context, filepath, collection=collection, drm_game='anniversary')

                drm_parser = DRMContainerParser(filepath, decompress_derickw=True)
                with drm_parser.temporary_extract_sections() as (extract_dir, extracted_paths, sections):
                    logger.debug('Extracted PSP DRM %s into %s (%d sections)', source_path.name, extract_dir, len(extracted_paths))
                    if not extracted_paths:
                        raise ValueError('No sections were found in the PSP DRM container')

                    if self.import_all_textures:
                        self._import_all_texture_files(context, extract_dir)

                    imported_results = self._import_from_object_refs(context, source_path, str(extracted_paths[0]), collection_name=source_path.stem, collection=collection_override, extracted_paths=extracted_paths, extracted_sections=sections)
                    if self.import_all_animations:
                        allowed_anim_ids = self._animation_ids_from_object_root(str(extracted_paths[0]))
                        imported_results.extend(
                            self._import_all_animation_files(
                                context,
                                extracted_paths,
                                armatures=[result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None],
                                source_name=source_path.stem,
                                default_endianness='<',
                                platform=self.platform,
                                allowed_anim_ids=allowed_anim_ids or None,
                            )
                        )
                    return imported_results

            if self.platform == 'GAMECUBE':
                return self._import_gamecube_drm(context, filepath, collection_override=collection_override)

            if TRLevelParser.is_level_file(filepath):
                collection = collection_override or self._get_or_create_collection(context, str(source_path), collection_name=source_path.stem)
                return self._import_level_file(context, filepath, collection=collection)

            drm_parser = DRMContainerParser(filepath)
            with drm_parser.temporary_extract_sections() as (extract_dir, extracted_paths, sections):
                drm_game = drm_parser.detect_game(sections)
                logger.debug('Extracted DRM %s into %s (%d sections) game=%s', source_path.name, extract_dir, len(extracted_paths), drm_game)
                if not extracted_paths:
                    raise ValueError('No sections were found in the DRM container')

                if drm_game == 'underworld':
                    object_root_path = next((Path(path) for path in extracted_paths if Path(path).suffix.lower() == '.obj'), Path(extracted_paths[0]))
                    collection = collection_override or self._get_or_create_collection(context, str(source_path), collection_name=source_path.stem)
                    imported_results = self._import_underworld_drm_probe(context, source_path, object_root_path, collection=collection)
                    if self.import_all_animations:
                        imported_results.extend(
                            self._import_all_animation_files(
                                context,
                                extracted_paths,
                                armatures=[result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None],
                                source_name=source_path.stem,
                                default_endianness='<',
                                platform='UNDERWORLD',
                            )
                        )
                    return imported_results

                if self.import_all_textures:
                    self._import_all_texture_files(context, extract_dir)

                level_path = next((path for path in extracted_paths if path.suffix.lower() == '.level'), None)
                if level_path is not None:
                    try:
                        collection = collection_override or self._get_or_create_collection(context, str(source_path), collection_name=source_path.stem)
                        return self._import_level_file(context, str(level_path), collection=collection, drm_game=drm_game)
                    except Exception as exc:
                        logger.warning('Level auto-import probe failed for %s: %s. Falling back to object import.', source_path.name, exc)

                imported_results = self._import_from_object_refs(context, source_path, str(extracted_paths[0]), collection_name=source_path.stem, collection=collection_override, extracted_paths=extracted_paths, extracted_sections=sections)
                if self.import_all_animations:
                    allowed_anim_ids = self._animation_ids_from_object_root(str(extracted_paths[0]))
                    imported_results.extend(
                        self._import_all_animation_files(
                            context,
                            extracted_paths,
                            armatures=[result.get('arm_obj') for result in imported_results if result.get('arm_obj') is not None],
                            source_name=source_path.stem,
                            default_endianness='<',
                            platform=self.platform,
                            allowed_anim_ids=allowed_anim_ids or None,
                        )
                    )
                return imported_results

        if suffix == '.obj':
            if self.import_all_textures:
                self._import_all_texture_files(context, source_path.parent)
            return self._import_from_object_refs(context, source_path, filepath, collection_name=source_path.parent.name or source_path.stem, collection=collection_override)


        if suffix == '.gnc':
            if AreaDBaseParser.looks_like_area_dbase(filepath):
                return [self._import_area_dbase_file(context, filepath, collection=collection_override)]
            raise ValueError(f'Unsupported .gnc section type: {source_path.name}')

        if suffix == '.ani':
            return [self._import_animation_file(context, filepath, platform=self.platform)]

        if suffix == '.mul':
            return self._import_multiplexstream_file(context, filepath)

        if self.import_all_textures and suffix in {'.tr7aemesh'}:
            self._import_all_texture_files(context, source_path.parent)

        if suffix == '.tr7aemesh':
            return [self._import_mesh_file(context, filepath, collection=collection_override)]
        if self.platform == 'PSP' and TRPSPLevelParser.is_level_file(filepath):
            return self._import_level_file(context, filepath, collection=collection_override, drm_game='anniversary')
        if TRLevelParser.is_level_file(filepath):
            return self._import_level_file(context, filepath, collection=collection_override)
        return [self._import_mesh_file(context, filepath, collection=collection_override)]

    def execute(self, context):
        configure_logging(self.debug)
        if bool(getattr(self, 'use_integrated_import_defaults', True)):
            self._enable_integrated_import_defaults()
        selected_filepaths = self._get_selected_filepaths()
        imported_results = []
        failed_files: list[str] = []

        self._print_import_header()
        for filepath in selected_filepaths:
            self._print_import_line(filepath)
            try:
                logger.debug('Import requested for %s', filepath)
                path_results = self._import_single_path(context, filepath)
                source_dir = Path(filepath).resolve().parent
                texture_reload_announced = False
                for imported_result in path_results:
                    if imported_result.get('diagnostic_only'):
                        continue
                    if (
                        self.import_textures
                        and imported_result.get('animation_action') is None
                        and not self._imports_underworld_materials_only(imported_result.get('model'))
                    ):
                        try:
                            self._reload_imported_result_textures(
                                imported_result,
                                source_dir,
                                announce=not texture_reload_announced,
                            )
                            texture_reload_announced = True
                        except Exception as exc:
                            logger.warning('Texture reload post-pass failed for %s: %s', Path(filepath).name, exc)
                imported_results.extend(path_results)
            except Exception as exc:
                failed_files.append(Path(filepath).name)
                if self.debug:
                    logger.error('TRLAU import failed for %s', filepath)
                    traceback.print_exc()
                self.report({'ERROR'}, f'TRLAU import failed for {Path(filepath).name}: {exc}')

        self._disable_relationship_lines(context)
        self._print_import_footer()
        if not imported_results:
            return {'CANCELLED'}

        imported_animation_count = sum(1 for result in imported_results if result.get('animation_action') is not None)
        imported_audio_count = sum(1 for result in imported_results if result.get('audio_strip') is not None or result.get('audio_embedded') or result.get('audio_wav_path'))
        imported_model_count = len(imported_results) - imported_animation_count - imported_audio_count
        message_parts = []
        if imported_model_count:
            message_parts.append(f'{imported_model_count} TRLAU model item(s)')
        if imported_animation_count:
            message_parts.append(f'{imported_animation_count} animation(s)')
        if imported_audio_count:
            message_parts.append(f'{imported_audio_count} audio track(s)')
        message = 'Imported ' + ', '.join(message_parts) if message_parts else 'Imported TR LAU Model(s)'
        if failed_files:
            message += f' | Failed files: {", ".join(failed_files)}'
        self.report({'INFO'}, message)
        return {'FINISHED'}



classes = (IMPORT_SCENE_OT_trlau_model, TRLAU_OT_play_sfx_sound, TRLAU_OT_stop_sfx_sound)


def menu_func_import(self, _context):
    self.layout.operator(IMPORT_SCENE_OT_trlau_model.bl_idname, text='Tomb Raider LAU Import')


def register():
    configure_logging(False)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
