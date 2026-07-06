from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import bpy

from ..builders.mesh_builder import MeshBuildSettings, MeshBuilder
from ..core.material_ui import reload_textures_for_objects


class ImportUtilityMixin:
    _ANSI_BLUE = '\033[94m'
    _ANSI_GREEN = '\033[92m'
    _ANSI_RESET = '\033[0m'

    @contextmanager
    def _temporary_import_overrides(self, **overrides):
        sentinel = object()
        original_values = {}
        try:
            for name, value in overrides.items():
                original_values[name] = getattr(self, name, sentinel)
                setattr(self, name, value)
            yield
        finally:
            for name, original_value in original_values.items():
                if original_value is sentinel:
                    try:
                        delattr(self, name)
                    except AttributeError:
                        pass
                else:
                    setattr(self, name, original_value)

    def _get_selected_filepaths(self) -> list[str]:
        if getattr(self, 'files', None):
            base_directory = Path(self.directory) if self.directory else Path(self.filepath).parent
            return [str((base_directory / file_elem.name).resolve()) for file_elem in self.files]
        return [self.filepath]

    def _print_import_header(self):
        print(f'{self._ANSI_BLUE}__________________________________{self._ANSI_RESET}')
        print(f'{self._ANSI_BLUE}Tomb Raider Legend/Anniversary/Underworld Model Import{self._ANSI_RESET}\n')

    @staticmethod
    def _print_import_line(filepath: str):
        print(f'Importing {Path(filepath).name}...')

    def _print_import_footer(self):
        print(f'{self._ANSI_GREEN}__________________________________{self._ANSI_RESET}')
        print(f'{self._ANSI_GREEN}Tomb Raider LAU Model finished importing{self._ANSI_RESET}')

    def _clear_scene(self, context):
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in list(bpy.data.objects):
                obj.select_set(True)
            bpy.ops.object.delete()
        except Exception:
            for obj in list(bpy.data.objects):
                try:
                    bpy.data.objects.remove(obj, do_unlink=True)
                except Exception:
                    pass

        for collection in list(bpy.data.collections):
            if collection != context.scene.collection:
                try:
                    bpy.data.collections.remove(collection)
                except Exception:
                    pass

        for datablock_group in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.curves):
            for datablock in list(datablock_group):
                if getattr(datablock, 'users', 0) == 0:
                    try:
                        datablock_group.remove(datablock)
                    except Exception:
                        pass

    def _import_all_texture_files(self, context, texture_directory: Path, collection=None, platform_hint: str | None = None):
        if not self.import_textures or not texture_directory.is_dir():
            return []

        if platform_hint is None:
            platform = str(getattr(self, 'platform', '') or '').upper()
            platform_hint = {
                'PS2': 'ps2',
                'PSP': 'psp',
                'PS3': 'ps3',
                'XBOX360': 'xbox360',
                'GAMECUBE': 'gamecube',
                'XBOX': 'xbox',
            }.get(platform)

        print('Importing Textures...')

        collection = collection or context.scene.collection
        loader = MeshBuilder(
            context=context,
            filepath=str(texture_directory / 'dummy.tr7aemesh'),
            settings=MeshBuildSettings(apply_segment_pivots=True, import_textures=True),
            collection=collection,
        )
        images = []
        for texture_path in sorted(texture_directory.glob('*.pcd')):
            try:
                image = loader._load_packed_image_from_pcd(int(texture_path.stem.split('_')[-1], 16), platform_hint=platform_hint)
            except Exception:
                image = None
            if image is not None:
                images.append(image)
        return images

    def _get_or_create_collection(self, context, filepath: str, collection_name: str | None = None):
        if not bool(getattr(self, 'create_import_collection', True)):
            target_name = str(getattr(self, 'import_collection_name', '') or '').strip()
            if target_name:
                collection = bpy.data.collections.get(target_name)
                if collection is not None:
                    return collection
            collection = getattr(context, 'collection', None) if context is not None else None
            if collection is not None:
                return collection
            scene = getattr(context, 'scene', None) if context is not None else None
            return getattr(scene, 'collection', None) or bpy.context.scene.collection

        filepath_obj = Path(filepath)
        name = collection_name or filepath_obj.parent.name or filepath_obj.stem
        collection = bpy.data.collections.get(name)
        if collection is None:
            collection = bpy.data.collections.new(name)
            context.scene.collection.children.link(collection)
        return collection

    @staticmethod
    def _get_or_create_child_collection(parent_collection, collection_name: str):
        collection = bpy.data.collections.get(collection_name)
        if collection is None:
            collection = bpy.data.collections.new(collection_name)
        if all(child != collection for child in parent_collection.children):
            parent_collection.children.link(collection)
        return collection

    def _get_or_create_level_root(self, collection, scene_center, level=None) -> bpy.types.Object | None:
        return None

    def _iter_collection_objects_recursive(self, collection):
        seen_collections = set()
        seen_objects = set()

        def walk(current_collection):
            collection_key = id(current_collection)
            if collection_key in seen_collections:
                return
            seen_collections.add(collection_key)
            for obj in current_collection.objects:
                object_key = id(obj)
                if object_key in seen_objects:
                    continue
                seen_objects.add(object_key)
                yield obj
            for child in current_collection.children:
                yield from walk(child)

        yield from walk(collection)

    def _apply_scene_center_root(self, collection, level) -> bpy.types.Object | None:
        return None

    @staticmethod
    def _tag_imported_result_source(imported_result: dict, source_dir: Path) -> None:
        return None

    def _reload_imported_result_textures(self, imported_result: dict, source_dir: Path, *, announce: bool = True) -> dict[str, int]:
        if announce:
            print('Importing Textures...')
        objects = [obj for obj in imported_result.get('mesh_objects', []) or [] if obj is not None]
        mesh_obj = imported_result.get('mesh_obj')
        if mesh_obj is not None and mesh_obj not in objects:
            objects.append(mesh_obj)
        return reload_textures_for_objects(objects, extra_dirs=[source_dir])

    @staticmethod
    def _disable_relationship_lines(context):
        try:
            screen = getattr(context, 'screen', None)
            if screen is None:
                return
            for area in screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                for space in area.spaces:
                    overlay = getattr(space, 'overlay', None)
                    if overlay is not None and hasattr(overlay, 'show_relationship_lines'):
                        overlay.show_relationship_lines = False
        except Exception:
            pass
