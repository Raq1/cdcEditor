from __future__ import annotations

import bpy
import json
import tempfile
import uuid
from pathlib import Path
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty

from ..core.bigfile import (
    BigfileError,
    available_bigfile_filelists,
    bundled_filelist_path,
    create_legend_bigfile_backup,
    format_bigfile_specialization,
    is_legend_primary_part,
    legend_bigfile_backup_dir,
    legend_bigfile_backup_exists,
    inspect_legend_bigfile,
    load_bigfile_filename_map,
    load_bigfile_filename_map_for_hashes,
    read_legend_header,
    repack_legend_bigfile,
    repack_legend_bigfile_with_replacements,
    specialize_bigfile_record_path,
    unpack_legend_bigfile,
    unpack_legend_bigfile_selection,
)

from ..core.import_options import sync_separation_flags
from ..operators.export_level import _enum_collection_items


_CUSTOM_FILELIST_ID = '__CUSTOM__'
_AUTO_COLLECTION_ID = '__TRLAU_AUTO_MATCH_COLLECTION__'


def _enum_bigfile_export_collection_items(_self=None, context=None):
    items = [(_AUTO_COLLECTION_ID, 'Auto-match selected DRM(s)', 'Find matching collection from each selected DRM path')]
    try:
        export_items = [item for item in _enum_collection_items(_self, context) if item[0] != '__TRLAU_NO_EXPORT_COLLECTION__']
    except Exception:
        export_items = []
    items.extend(export_items)
    return items


def _filelist_display_label(name: str) -> str:
    if not name:
        return 'None'
    if name == _CUSTOM_FILELIST_ID:
        return 'Custom...'
    return name[:-4] if name.lower().endswith('.txt') else name


def _selected_filelist_path(props) -> str | None:
    if props is None:
        return None
    selected = getattr(props, 'filelist_name', '') or ''
    if selected == _CUSTOM_FILELIST_ID:
        path = getattr(props, 'custom_filelist_path', '') or ''
        return bpy.path.abspath(path) if path else None
    bundled = bundled_filelist_path(selected)
    return str(bundled) if bundled is not None else None


def _load_selected_filename_map(props) -> dict[int, str]:
    path = _selected_filelist_path(props)
    return load_bigfile_filename_map(path) if path else {}


def _load_selected_filename_map_for_hashes(props, hash_values) -> dict[int, str]:
    path = _selected_filelist_path(props)
    return load_bigfile_filename_map_for_hashes(path, hash_values) if path else {}

def _guess_bigfile_import_platform(props) -> str:
    text = ' '.join((
        getattr(props, 'filelist_name', '') or '',
        getattr(props, 'custom_filelist_path', '') or '',
        getattr(props, 'bigfile_path', '') or '',
    )).lower()
    if 'xenon' in text or 'xbox 360' in text or 'xbox360' in text:
        return 'XBOX360'
    if 'gamecube' in text or 'wii' in text or 'nintendo' in text:
        return 'GAMECUBE'
    if 'ps3' in text or 'playstation 3' in text:
        return 'PS3'
    if 'psp' in text:
        return 'PSP'
    if 'ps2' in text or 'playstation 2' in text:
        return 'PS2'
    if 'xbox' in text:
        return 'XBOX'
    return 'PC'


_IMPORTABLE_BIGFILE_SUFFIXES = {'.drm', '.mul', '.ani', '.obj', '.tr7aemesh', '.gnc'}
_IMPORT_SUFFIX_PRIORITY = {
    '.drm': 0,
    '.tr7aemesh': 1,
    '.obj': 1,
    '.gnc': 2,
    '.mul': 3,
    '.ani': 4,
}


def _bigfile_import_temp_dir(props) -> Path:
    base = getattr(props, 'unpack_output_directory', '') or ''
    if base:
        root = Path(bpy.path.abspath(base)) / '_bigfile_import_cache'
    else:
        temp_root = bpy.app.tempdir or tempfile.gettempdir()
        root = Path(temp_root) / 'trlau_bigfile_import'
    stem = Path(getattr(props, 'bigfile_path', '') or 'bigfile').stem or 'bigfile'
    path = root / f'{stem}_{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _manifest_importable_files(manifest_path: Path) -> tuple[list[Path], int]:
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    base = Path(manifest_path).parent
    paths: list[Path] = []
    skipped = 0
    for record in manifest.get('records') or []:
        rel = record.get('file') or ''
        if not rel:
            skipped += 1
            continue
        path = (base / rel).resolve()
        suffix = path.suffix.lower()
        if suffix in _IMPORTABLE_BIGFILE_SUFFIXES:
            paths.append(path)
        else:
            skipped += 1
    paths.sort(key=lambda path: (_IMPORT_SUFFIX_PRIORITY.get(path.suffix.lower(), 99), str(path).lower()))
    return paths, skipped


class TRLAU_PG_bigfile_record_entry(bpy.types.PropertyGroup):
    selected: BoolProperty(
        name='Extract',
        description='Mark this record for Extract Selected',
        default=False,
    )
    index: IntProperty(name='Index', default=0)
    hash_value: StringProperty(name='Hash', default='00000000')
    size: IntProperty(name='Size', default=0)
    stored_size: IntProperty(name='Stored', default=0)
    offset: IntProperty(name='Offset', default=0)
    size_text: StringProperty(name='Size', default='0 B')
    stored_size_text: StringProperty(name='Stored', default='0 B')
    offset_text: StringProperty(name='Offset', default='0x00000000')
    spec_mask: StringProperty(name='Specialization', default='')
    compressed: BoolProperty(name='Compressed', default=False)
    filename: StringProperty(name='Filename', default='')
    file_type: StringProperty(name='Type', default='unknown')
    matched: BoolProperty(name='Named', default=False)
    entry_kind: StringProperty(name='Entry Kind', default='file')
    depth: IntProperty(name='Depth', default=0)
    folder_path: StringProperty(name='Folder Path', default='')
    full_path: StringProperty(name='Full Path', default='')
    search_text: StringProperty(name='Search Text', default='')
    expanded: BoolProperty(name='Expanded', default=True)


class TRLAU_PG_archive_tools(bpy.types.PropertyGroup):
    bigfile_path: StringProperty(
        name='Bigfile',
        description='Tomb Raider Legend .000 or .dat Bigfile archive to inspect or open',
        default='',
    )
    filelist_name: StringProperty(
        name='File List',
        description='Optional filename list used to resolve Bigfile hashes into paths',
        default='',
    )
    custom_filelist_path: StringProperty(
        name='Custom File List',
        description='Custom text file containing one archive path per line',
        subtype='FILE_PATH',
        default='',
    )
    unpack_output_directory: StringProperty(
        name='Output Folder',
        description='Folder where manifest.json and unpacked files will be written. Leave empty to create one next to the selected bigfile',
        subtype='DIR_PATH',
        default='',
    )
    repack_manifest_path: StringProperty(
        name='Manifest',
        description='manifest.json produced by Unpack Bigfile',
        subtype='FILE_PATH',
        default='',
    )
    repack_output_path: StringProperty(
        name='Output Bigfile',
        description='Output .000 or .dat archive path. Leave empty to write bigfile_repacked.000 next to the manifest',
        default='',
    )
    compress_marked_records: BoolProperty(
        name='Compress Marked Records',
        description='Recompress records that were marked as compressed in the manifest',
        default=True,
    )
    bigfile_records: CollectionProperty(type=TRLAU_PG_bigfile_record_entry)
    bigfile_record_index: IntProperty(name='Record Index', default=0)
    bigfile_browser_summary: StringProperty(name='Summary', default='')
    bigfile_search: StringProperty(
        name='Search',
        description='Filter the opened Bigfile file list',
        default='',
    )
    bigfile_export_collection_name: StringProperty(
        name='Mesh Collection',
        description='Collection to export into the selected Bigfile DRM. Leave empty to auto-match by selected DRM path.',
        default='',
    )
    bigfile_export_textures: BoolProperty(
        name='Export Textures',
        description='Export modified texture sections while writing selected DRM records back into the Bigfile',
        default=True,
    )
    bigfile_export_cloth: BoolProperty(
        name='Export Cloth',
        description='Export ClothSetup sections while writing selected DRM records back into the Bigfile',
        default=True,
    )


def _archive_props(context) -> TRLAU_PG_archive_tools | None:
    scene = getattr(context, 'scene', None)
    return getattr(scene, 'trlau_archive_tools', None) if scene is not None else None


def _record_filename(index: int, hash_value: int) -> str:
    return f'{index:05d}_{hash_value:08X}.bin'


def _infer_bigfile_file_type(filename: str) -> str:
    ext = (filename or '').rsplit('/', 1)[-1].rsplit('\\', 1)[-1].rsplit('.', 1)[-1].lower() if '.' in (filename or '') else ''
    mapping = {
        'drm': 'DRM',
        'mul': 'MUL',
        'raw': 'RAW',
        'mus': 'MUS',
        'tfb': 'TFB',
        'sam': 'SAM',
        'ids': 'IDS',
        'dat': 'DAT',
        'txt': 'TXT',
    }
    return mapping.get(ext, 'unknown')


def _format_bytes(value: int) -> str:
    return f'{int(value):,} B'


def _format_offset(value: int) -> str:
    return f'0x{int(value):X}'


def _normalize_browser_path(path_text: str) -> str:
    text = str(path_text or '').replace('\\', '/').strip().strip('/')
    parts = [part.strip() for part in text.split('/') if part.strip() and part.strip() not in {'.', '..'}]
    return '/'.join(parts)


def _path_sort_key(path_text: str):
    return tuple(part.lower() for part in _normalize_browser_path(path_text).split('/') if part)


def _add_bigfile_browser_folder(props: TRLAU_PG_archive_tools, folder_path: str, depth: int):
    folder_path = _normalize_browser_path(folder_path)
    item = props.bigfile_records.add()
    item.selected = False
    item.index = -1
    item.hash_value = ''
    item.size = 0
    item.stored_size = 0
    item.offset = 0
    item.size_text = ''
    item.stored_size_text = ''
    item.offset_text = ''
    item.spec_mask = ''
    item.compressed = False
    item.filename = folder_path.rsplit('/', 1)[-1] if folder_path else ''
    item.file_type = 'Folder'
    item.matched = True
    item.entry_kind = 'folder'
    item.depth = int(depth)
    item.folder_path = folder_path
    item.full_path = folder_path
    item.search_text = _make_bigfile_entry_search_text(item.filename, item.full_path, item.folder_path, item.file_type, item.spec_mask, item.hash_value)
    item.expanded = False
    return item


def _add_bigfile_browser_file(props: TRLAU_PG_archive_tools, info: dict, depth: int):
    item = props.bigfile_records.add()
    item.selected = False
    item.index = int(info['index'])
    item.hash_value = f"{int(info['hash']):08X}"
    size_value = int(info['size'])
    stored_value = int(info['stored'])
    absolute_offset = int(info['offset']) << 11
    item.size = min(size_value, 2147483646)
    item.stored_size = min(stored_value, 2147483646)
    item.offset = 0
    item.size_text = _format_bytes(size_value)
    item.stored_size_text = _format_bytes(stored_value)
    item.offset_text = _format_offset(absolute_offset)
    item.spec_mask = format_bigfile_specialization(int(info['spec_mask']))
    item.compressed = bool(info['compressed'])
    full_path = _normalize_browser_path(info['path']) or _record_filename(int(info['index']), int(info['hash']))
    full_path = _normalize_browser_path(specialize_bigfile_record_path(full_path, int(info['spec_mask'])))
    item.filename = full_path.rsplit('/', 1)[-1]
    item.file_type = _infer_bigfile_file_type(full_path)
    item.matched = bool(info['matched'])
    item.entry_kind = 'file'
    item.depth = int(depth)
    item.folder_path = full_path.rsplit('/', 1)[0] if '/' in full_path else ''
    item.full_path = full_path
    item.search_text = _make_bigfile_entry_search_text(item.filename, item.full_path, item.folder_path, item.file_type, item.spec_mask, item.hash_value)
    item.expanded = True
    return item


def _build_bigfile_browser_tree(props: TRLAU_PG_archive_tools, file_infos: list[dict]) -> None:
    tree = {'folders': {}, 'files': []}
    for info in file_infos:
        full_path = _normalize_browser_path(info.get('path') or '')
        if not full_path:
            full_path = _record_filename(int(info['index']), int(info['hash']))
        info['path'] = full_path
        parts = [part for part in full_path.split('/') if part]
        if not parts:
            tree['files'].append(info)
            continue
        node = tree
        for folder in parts[:-1]:
            node = node['folders'].setdefault(folder, {'folders': {}, 'files': []})
        node['files'].append(info)

    def emit_node(node: dict, parent_path: str, depth: int) -> None:
        for folder_name in sorted(node['folders'], key=lambda value: value.lower()):
            folder_path = f'{parent_path}/{folder_name}' if parent_path else folder_name
            _add_bigfile_browser_folder(props, folder_path, depth)
            emit_node(node['folders'][folder_name], folder_path, depth + 1)
        for info in sorted(node['files'], key=lambda item: _path_sort_key(item.get('path', ''))):
            _add_bigfile_browser_file(props, info, depth)

    emit_node(tree, '', 0)


def _bigfile_folder_states(props: TRLAU_PG_archive_tools) -> dict[str, bool]:
    return {item.folder_path: bool(item.expanded) for item in props.bigfile_records if item.entry_kind == 'folder'}


def _bigfile_search_terms(props: TRLAU_PG_archive_tools) -> list[str]:
    text = str(getattr(props, 'bigfile_search', '') or '').strip().lower()
    return [term for term in text.split() if term]


def _make_bigfile_entry_search_text(*values) -> str:
    return ' '.join(str(value or '') for value in values).lower()


def _bigfile_entry_search_text(item: TRLAU_PG_bigfile_record_entry) -> str:
    cached = getattr(item, 'search_text', '') or ''
    if cached:
        return cached
    return _make_bigfile_entry_search_text(
        getattr(item, 'filename', '') or '',
        getattr(item, 'full_path', '') or '',
        getattr(item, 'folder_path', '') or '',
        getattr(item, 'file_type', '') or '',
        getattr(item, 'spec_mask', '') or '',
        getattr(item, 'hash_value', '') or '',
    )


def _bigfile_entry_matches_search(item: TRLAU_PG_bigfile_record_entry, terms: list[str]) -> bool:
    if not terms:
        return True
    text = _bigfile_entry_search_text(item)
    return all(term in text for term in terms)


def _bigfile_folder_has_matching_descendant(props: TRLAU_PG_archive_tools, folder_path: str, terms: list[str]) -> bool:
    if not terms:
        return True
    prefix = _normalize_browser_path(folder_path)
    if not prefix:
        return False
    for item in props.bigfile_records:
        if item.entry_kind != 'file':
            continue
        full_path = _normalize_browser_path(item.full_path)
        if full_path.startswith(prefix + '/') and _bigfile_entry_matches_search(item, terms):
            return True
    return False


def _is_bigfile_entry_visible(item: TRLAU_PG_bigfile_record_entry, folder_states: dict[str, bool], props: TRLAU_PG_archive_tools | None = None) -> bool:
    terms = _bigfile_search_terms(props) if props is not None else []
    if terms:
        if item.entry_kind == 'folder':
            return _bigfile_entry_matches_search(item, terms) or (props is not None and _bigfile_folder_has_matching_descendant(props, item.folder_path, terms))
        return _bigfile_entry_matches_search(item, terms)

    path = item.folder_path if item.entry_kind == 'folder' else item.full_path
    path = _normalize_browser_path(path)
    if not path or item.depth <= 0:
        return True
    parts = path.split('/')
    if item.entry_kind == 'file':
        parts = parts[:-1]
    else:
        parts = parts[:-1]
    current = ''
    for part in parts:
        current = f'{current}/{part}' if current else part
        if not folder_states.get(current, True):
            return False
    return True


def _selected_bigfile_record_indices(props: TRLAU_PG_archive_tools) -> list[int]:
    selected_indices: set[int] = set()
    selected_folders = [item.folder_path for item in props.bigfile_records if item.entry_kind == 'folder' and item.selected]
    for item in props.bigfile_records:
        if item.entry_kind != 'file':
            continue
        if item.selected:
            selected_indices.add(int(item.index))
            continue
        full_path = _normalize_browser_path(item.full_path)
        for folder_path in selected_folders:
            prefix = _normalize_browser_path(folder_path)
            if prefix and full_path.startswith(prefix + '/'):
                selected_indices.add(int(item.index))
                break
    return sorted(selected_indices)


def _load_bigfile_records_into_props(props: TRLAU_PG_archive_tools, path: str) -> dict:
    records = read_legend_header(path, allow_incomplete=True)
    filename_map = _load_selected_filename_map_for_hashes(props, (record.hash_value for record in records))
    props.bigfile_records.clear()
    compressed_count = 0
    total_size = 0
    total_stored = 0
    matched_count = 0
    file_infos: list[dict] = []
    for record in records:
        if record.is_compressed:
            compressed_count += 1
        total_size += int(record.size)
        total_stored += int(record.stored_size)
        matched_name = filename_map.get(int(record.hash_value))
        if matched_name:
            matched_count += 1
        file_infos.append({
            'index': int(record.index),
            'hash': int(record.hash_value),
            'size': int(record.size),
            'stored': int(record.stored_size),
            'offset': int(record.offset),
            'spec_mask': int(record.spec_mask),
            'compressed': bool(record.is_compressed),
            'matched': bool(matched_name),
            'path': matched_name or _record_filename(record.index, record.hash_value),
        })
    _build_bigfile_browser_tree(props, file_infos)
    props.bigfile_record_index = 0
    filelist_note = f', {matched_count} named' if filename_map else ''
    expected_virtual_size = max(((int(record.offset) << 11) + int(record.stored_size)) for record in records) if records else 0
    try:
        available_size = Path(path).stat().st_size
    except OSError:
        available_size = 0
    incomplete_note = ', incomplete archive fragment' if available_size and expected_virtual_size > available_size else ''
    props.bigfile_browser_summary = (
        f'{len(records)} record(s), {compressed_count} compressed{filelist_note}, '
        f'{total_size:,} bytes unpacked, {total_stored:,} bytes stored{incomplete_note}'
    )
    return {
        'recordCount': len(records),
        'compressedCount': compressed_count,
        'totalSize': total_size,
        'totalStoredSize': total_stored,
        'namedCount': matched_count,
    }


class TRLAU_OT_select_bigfile_path(bpy.types.Operator):
    bl_idname = 'trlau.select_bigfile_path'
    bl_label = 'Select Bigfile'
    bl_description = 'Select the first Tomb Raider Legend Bigfile segment (.000) or a .dat archive'
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default='*.000;*.dat', options={'HIDDEN'})

    def invoke(self, context, event):
        self.filepath = ''
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = _archive_props(context)
        if props is None:
            return {'CANCELLED'}
        if not self.filepath:
            return {'CANCELLED'}
        if not is_legend_primary_part(self.filepath):
            self.report({'ERROR'}, 'Select the first Bigfile segment (.000) or a .dat Bigfile archive')
            return {'CANCELLED'}
        props.bigfile_path = self.filepath
        return {'FINISHED'}


class TRLAU_OT_select_bigfile_output_path(bpy.types.Operator):
    bl_idname = 'trlau.select_bigfile_output_path'
    bl_label = 'Select Output Bigfile'
    bl_description = 'Select a .000 or .dat output archive path'
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default='*.000;*.dat', options={'HIDDEN'})

    def invoke(self, context, event):
        props = _archive_props(context)
        self.filepath = getattr(props, 'repack_output_path', '') if props is not None else ''
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = _archive_props(context)
        if props is None:
            return {'CANCELLED'}
        if not self.filepath:
            return {'CANCELLED'}
        if not is_legend_primary_part(self.filepath):
            self.report({'ERROR'}, 'Output Bigfile path must end in .000 or .dat')
            return {'CANCELLED'}
        props.repack_output_path = self.filepath
        return {'FINISHED'}




class TRLAU_OT_select_bigfile_filelist_path(bpy.types.Operator):
    bl_idname = 'trlau.select_bigfile_filelist_path'
    bl_label = 'Select File List'
    bl_description = 'Select a custom text file containing archive paths'
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default='*.txt', options={'HIDDEN'})

    def invoke(self, context, event):
        props = _archive_props(context)
        self.filepath = getattr(props, 'custom_filelist_path', '') if props is not None else ''
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = _archive_props(context)
        if props is None:
            return {'CANCELLED'}
        if not self.filepath:
            return {'CANCELLED'}
        props.filelist_name = _CUSTOM_FILELIST_ID
        props.custom_filelist_path = self.filepath
        if props.bigfile_path and len(props.bigfile_records) > 0:
            try:
                _load_bigfile_records_into_props(props, bpy.path.abspath(props.bigfile_path))
            except Exception:
                pass
        return {'FINISHED'}



class TRLAU_OT_set_bigfile_filelist(bpy.types.Operator):
    bl_idname = 'trlau.set_bigfile_filelist'
    bl_label = 'Set File List'
    bl_description = 'Select the file list used to resolve Bigfile hashes'
    bl_options = {'REGISTER'}

    filelist_name: StringProperty(default='')

    def execute(self, context):
        props = _archive_props(context)
        if props is None:
            return {'CANCELLED'}
        props.filelist_name = self.filelist_name or ''
        if props.bigfile_path and len(props.bigfile_records) > 0:
            try:
                _load_bigfile_records_into_props(props, bpy.path.abspath(props.bigfile_path))
            except Exception:
                pass
        return {'FINISHED'}


class TRLAU_MT_bigfile_filelists(bpy.types.Menu):
    bl_idname = 'TRLAU_MT_bigfile_filelists'
    bl_label = 'File List'

    def draw(self, context):
        layout = self.layout
        op = layout.operator('trlau.set_bigfile_filelist', text='None')
        op.filelist_name = ''
        names = available_bigfile_filelists()
        if names:
            layout.separator()
            for name in names:
                op = layout.operator('trlau.set_bigfile_filelist', text=_filelist_display_label(name))
                op.filelist_name = name
        else:
            row = layout.row()
            row.enabled = False
            row.label(text='No bundled file lists found')
        layout.separator()
        op = layout.operator('trlau.set_bigfile_filelist', text='Custom...')
        op.filelist_name = _CUSTOM_FILELIST_ID


class TRLAU_UL_bigfile_records(bpy.types.UIList):
    bl_idname = 'TRLAU_UL_bigfile_records'

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        terms = _bigfile_search_terms(data)
        if terms:
            flags = [0] * len(items)
            direct_matches = [False] * len(items)
            folder_paths_with_matches: set[str] = set()

            for index, item in enumerate(items):
                if not _bigfile_entry_matches_search(item, terms):
                    continue
                direct_matches[index] = True
                if item.entry_kind == 'file':
                    folder_path = _normalize_browser_path(item.folder_path)
                    while folder_path:
                        folder_paths_with_matches.add(folder_path)
                        folder_path = folder_path.rsplit('/', 1)[0] if '/' in folder_path else ''

            for index, item in enumerate(items):
                visible = direct_matches[index]
                if item.entry_kind == 'folder' and not visible:
                    visible = _normalize_browser_path(item.folder_path) in folder_paths_with_matches
                flags[index] = self.bitflag_filter_item if visible else 0
            return flags, []

        folder_states = _bigfile_folder_states(data)
        flags = [
            self.bitflag_filter_item if _is_bigfile_entry_visible(item, folder_states, None) else 0
            for item in items
        ]
        return flags, []

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.prop(item, 'selected', text='')
            name_split = row.split(factor=0.56, align=True)
            name_row = name_split.row(align=True)
            for _ in range(min(int(item.depth), 8)):
                name_row.label(text='', icon='BLANK1')
            if item.entry_kind == 'folder':
                toggle = name_row.operator(
                    'trlau.toggle_bigfile_folder',
                    text='',
                    icon='TRIA_DOWN' if item.expanded else 'TRIA_RIGHT',
                    emboss=False,
                )
                toggle.folder_path = item.folder_path
                name_row.label(text=item.filename, icon='FILE_FOLDER')
                detail_row = name_split.row(align=True)
                type_split = detail_row.split(factor=0.22, align=True)
                type_split.label(text='Folder')
                size_split = type_split.split(factor=0.48, align=True)
                size_split.label(text='')
                size_split.label(text='')
            else:
                name_row.label(text='', icon='BLANK1')
                name_row.label(text=item.filename, icon='FILE_TEXT' if item.matched else 'FILE')
                detail_row = name_split.row(align=True)
                type_split = detail_row.split(factor=0.22, align=True)
                type_split.label(text=item.file_type or 'unknown')
                size_split = type_split.split(factor=0.48, align=True)
                size_split.label(text=item.size_text or _format_bytes(item.size))
                size_split.label(text=item.spec_mask)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.filename if item.entry_kind == 'folder' else f'{item.index:05d}')


class TRLAU_OT_toggle_bigfile_folder(bpy.types.Operator):
    bl_idname = 'trlau.toggle_bigfile_folder'
    bl_label = 'Toggle Folder'
    bl_description = 'Expand or collapse this folder'
    bl_options = {'REGISTER'}

    folder_path: StringProperty(default='')

    def execute(self, context):
        props = _archive_props(context)
        if props is None:
            return {'CANCELLED'}
        target = _normalize_browser_path(self.folder_path)
        for item in props.bigfile_records:
            if item.entry_kind == 'folder' and item.folder_path == target:
                item.expanded = not item.expanded
                return {'FINISHED'}
        return {'CANCELLED'}


class TRLAU_OT_open_bigfile_browser(bpy.types.Operator):
    bl_idname = 'trlau.open_bigfile_browser'
    bl_label = 'Open Bigfile'
    bl_description = 'Open a dialog showing the records inside a Tomb Raider Legend Bigfile'
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        try:
            source_path = bpy.path.abspath(props.bigfile_path)
            _load_bigfile_records_into_props(props, source_path)
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to open bigfile: {exc}')
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=1040)

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        props = _archive_props(context)
        if props is None:
            layout.label(text='Archive settings unavailable', icon='ERROR')
            return

        search_row = layout.row(align=True)
        search_row.prop(props, 'bigfile_search', text='', icon='VIEWZOOM')

        row = layout.row(align=True)
        row.operator('trlau.unpack_bigfile', text='Unpack All', icon='PACKAGE')
        row.operator('trlau.unpack_selected_bigfile_records', text='Extract Selected', icon='EXPORT')
        row.operator('trlau.import_selected_bigfile_records', text='Import Selected', icon='IMPORT')
        row.operator('trlau.export_selected_bigfile_drms', text='Export DRM', icon='EXPORT')
        row.operator('trlau.create_bigfile_backup', text='Create Backup', icon='DUPLICATE')
        row.operator('trlau.select_all_bigfile_records', text='Select All', icon='CHECKBOX_HLT').selected = True
        row.operator('trlau.select_all_bigfile_records', text='Select None', icon='CHECKBOX_DEHLT').selected = False

        header = layout.row(align=True)
        header.label(text='')
        name_split = header.split(factor=0.56, align=True)
        name_split.label(text='Filename')
        detail_row = name_split.row(align=True)
        type_split = detail_row.split(factor=0.22, align=True)
        type_split.label(text='Type')
        size_split = type_split.split(factor=0.48, align=True)
        size_split.label(text='Size')
        size_split.label(text='Specialization')

        layout.template_list(
            'TRLAU_UL_bigfile_records',
            '',
            props,
            'bigfile_records',
            props,
            'bigfile_record_index',
            rows=18,
        )


class TRLAU_OT_select_all_bigfile_records(bpy.types.Operator):
    bl_idname = 'trlau.select_all_bigfile_records'
    bl_label = 'Select Bigfile Records'
    bl_description = 'Select or deselect all visible Bigfile records'
    bl_options = {'REGISTER'}

    selected: BoolProperty(default=True)

    def execute(self, context):
        props = _archive_props(context)
        if props is None:
            return {'CANCELLED'}
        value = bool(self.selected)
        folder_states = _bigfile_folder_states(props)
        for item in props.bigfile_records:
            if _is_bigfile_entry_visible(item, folder_states, props):
                item.selected = value
        return {'FINISHED'}


class TRLAU_OT_unpack_selected_bigfile_records(bpy.types.Operator):
    bl_idname = 'trlau.unpack_selected_bigfile_records'
    bl_label = 'Extract Selected'
    bl_description = 'Extract the selected Bigfile records to the output folder'
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        selected = _selected_bigfile_record_indices(props)
        if not selected:
            self.report({'ERROR'}, 'No Bigfile records selected')
            return {'CANCELLED'}
        try:
            source_path = bpy.path.abspath(props.bigfile_path)
            output_dir = bpy.path.abspath(props.unpack_output_directory) if props.unpack_output_directory else ''
            manifest_path = unpack_legend_bigfile_selection(
                source_path,
                output_dir,
                selected,
                filename_map=_load_selected_filename_map(props),
                allow_incomplete=True,
            )
            props.repack_manifest_path = str(manifest_path)
            if not props.unpack_output_directory:
                props.unpack_output_directory = str(manifest_path.parent)
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to extract selected records: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Extracted {len(selected)} record(s) to {manifest_path.parent}')
        return {'FINISHED'}


class TRLAU_OT_import_selected_bigfile_records(bpy.types.Operator):
    bl_idname = 'trlau.import_selected_bigfile_records'
    bl_label = 'Import Selected'
    bl_description = 'Extract the selected Bigfile records to a cache folder and import supported TRLAU files'
    bl_options = {'REGISTER', 'UNDO'}

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
    import_main_model_only: BoolProperty(name='Main Model Only', default=True, description='Only import the first referenced model (.drm, .obj)')
    import_next_gen_model: BoolProperty(name='Import Next Gen Model', default=False, description='Import the next-generation render model referenced by cdcRenderDataID when present')
    import_all_textures: BoolProperty(name='Import All Textures', default=False, description='Import all textures found in the .drm or folder')
    import_all_animations: BoolProperty(name='Import All Animations', default=False, description='Import all animations found in the .drm or folder')
    import_armature_only: BoolProperty(name='Import Armature Only', default=False, description='Import only the armature')
    import_bounding_boxes: BoolProperty(name='Import Bounding Boxes', default=False, description="Import bones' bounding boxes")
    import_kdnodes: BoolProperty(name='Import KDNodes', default=False, description='Import TerrainGroup collision KDNode empties when importing levels')
    import_audio: BoolProperty(name='Import Audio', default=True, description='Import PC level SFX markers and decode referenced Wave sections as Blender speakers', options={'HIDDEN', 'SKIP_SAVE'})
    split_terrain_groups_by_strip: BoolProperty(name='Split TerrainGroups by Strip', default=False, description='Import TerrainGroups as invidual strip meshes')
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

    def invoke(self, context, event):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        selected = _selected_bigfile_record_indices(props)
        if not selected:
            self.report({'ERROR'}, 'No Bigfile records selected')
            return {'CANCELLED'}
        try:
            self.platform = _guess_bigfile_import_platform(props)
        except Exception:
            pass
        return context.window_manager.invoke_props_dialog(self, width=540)

    def draw(self, _context):
        layout = self.layout
        layout.label(text='Bigfile Import Options')
        layout.prop(self, 'platform')
        layout.prop(self, 'import_textures')

        model_props = ['import_main_model_only', 'separate_by_drawgroup', 'separate_by_material']
        if self.platform == 'PC':
            model_props.insert(1, 'import_next_gen_model')
        sections = [
            ('show_model_settings', 'Model Settings', tuple(model_props)),
        ]
        if self.platform == 'PC':
            sections.append(('show_level_settings', 'Level Settings', ('import_kdnodes', 'split_terrain_groups_by_strip')))
        sections.append(('show_advanced_settings', 'Advanced Settings', ('import_all_textures', 'import_all_animations', 'import_armature_only')))
        for flag_name, label, prop_names in sections:
            box = layout.box()
            header = box.row()
            expanded = getattr(self, flag_name)
            header.prop(self, flag_name, text=label, icon='DOWNARROW_HLT' if expanded else 'RIGHTARROW_THIN', emboss=True)
            if not expanded:
                continue
            column = box.column(align=True)
            for prop_name in prop_names:
                column.prop(self, prop_name)
            if flag_name == 'show_advanced_settings' and self.platform == 'PC':
                column.prop(self, 'import_bounding_boxes')

        layout.prop(self, 'debug')

    def execute(self, context):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        selected = _selected_bigfile_record_indices(props)
        if not selected:
            self.report({'ERROR'}, 'No Bigfile records selected')
            return {'CANCELLED'}

        try:
            source_path = bpy.path.abspath(props.bigfile_path)
            extract_dir = _bigfile_import_temp_dir(props)
            manifest_path = unpack_legend_bigfile_selection(
                source_path,
                extract_dir,
                selected,
                filename_map=_load_selected_filename_map(props),
                allow_incomplete=True,
            )
            import_paths, skipped = _manifest_importable_files(Path(manifest_path))
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to extract selected records for import: {exc}')
            return {'CANCELLED'}

        if not import_paths:
            self.report({'ERROR'}, f'No supported importable files found in {len(selected)} selected record(s)')
            return {'CANCELLED'}

        imported = 0
        failed: list[str] = []
        for path in import_paths:
            try:
                result = bpy.ops.import_scene.trlau_model(
                    'EXEC_DEFAULT',
                    filepath=str(path),
                    platform=self.platform,
                    import_textures=bool(self.import_textures),
                    import_main_model_only=bool(self.import_main_model_only),
                    import_next_gen_model=bool(self.import_next_gen_model),
                    separate_by_drawgroup=bool(self.separate_by_drawgroup),
                    separate_by_material=bool(self.separate_by_material),
                    import_all_textures=bool(self.import_all_textures),
                    import_all_animations=bool(self.import_all_animations),
                    import_armature_only=bool(self.import_armature_only),
                    import_bounding_boxes=bool(self.import_bounding_boxes),
                    import_kdnodes=bool(self.import_kdnodes),
                    split_terrain_groups_by_strip=bool(self.split_terrain_groups_by_strip),
                    debug=bool(self.debug),
                )
                if 'FINISHED' in result:
                    imported += 1
                else:
                    failed.append(path.name)
            except Exception as exc:
                failed.append(f'{path.name}: {exc}')

        if imported <= 0:
            self.report({'ERROR'}, 'No selected Bigfile files were imported')
            return {'CANCELLED'}

        message = f'Imported {imported} file(s) from Bigfile'
        if skipped:
            message += f'; skipped {skipped} unsupported record(s)'
        if failed:
            message += f'; failed {len(failed)}'
        self.report({'INFO'}, message)
        return {'FINISHED'}


_LANGUAGE_SUFFIXES = ('_en', '_fr', '_de', '_it', '_es', '_nl', '_nextgen')


def _drm_record_items_for_indices(props: TRLAU_PG_archive_tools, indices: set[int]) -> list[TRLAU_PG_bigfile_record_entry]:
    items = []
    for item in props.bigfile_records:
        if item.entry_kind != 'file' or int(item.index) not in indices:
            continue
        if _normalize_browser_path(item.full_path).lower().endswith('.drm'):
            items.append(item)
    return items


def _collection_matches_drm_path(collection: bpy.types.Collection, full_path: str) -> bool:
    path = _normalize_browser_path(full_path)
    filename = path.rsplit('/', 1)[-1]
    stem = filename.rsplit('.', 1)[0]
    candidates = {stem.lower(), filename.lower(), path.lower()}
    for suffix in _LANGUAGE_SUFFIXES:
        if stem.lower().endswith(suffix):
            candidates.add(stem[: -len(suffix)].lower())
    collection_names = {collection.name.lower()}
    for key in ('trlau_source_drm', 'trlau_drm_name'):
        value = collection.get(key)
        if value:
            text = _normalize_browser_path(str(value)).lower()
            collection_names.add(text)
            collection_names.add(text.rsplit('/', 1)[-1])
            collection_names.add(text.rsplit('/', 1)[-1].rsplit('.', 1)[0])
    for obj in getattr(collection, 'all_objects', []) or []:
        for key in ('trlau_source_drm', 'trlau_drm_name'):
            value = obj.get(key) if hasattr(obj, 'get') else None
            if value:
                text = _normalize_browser_path(str(value)).lower()
                collection_names.add(text)
                collection_names.add(text.rsplit('/', 1)[-1])
                collection_names.add(text.rsplit('/', 1)[-1].rsplit('.', 1)[0])
    return bool(candidates & collection_names)


def _find_drm_export_collection(full_path: str) -> bpy.types.Collection | None:
    path = _normalize_browser_path(full_path)
    filename = path.rsplit('/', 1)[-1]
    stem = filename.rsplit('.', 1)[0]
    direct_names = [stem, filename]
    lowered = stem.lower()
    for suffix in _LANGUAGE_SUFFIXES:
        if lowered.endswith(suffix):
            direct_names.append(stem[: -len(suffix)])
            break
    for name in direct_names:
        collection = bpy.data.collections.get(name)
        if collection is not None:
            return collection
    matches = [collection for collection in bpy.data.collections if _collection_matches_drm_path(collection, full_path)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        matches.sort(key=lambda col: len(col.name))
        return matches[0]
    return None


def _safe_export_filename_from_record(item: TRLAU_PG_bigfile_record_entry) -> str:
    filename = _normalize_browser_path(item.full_path).rsplit('/', 1)[-1]
    if not filename.lower().endswith('.drm'):
        filename += '.drm'
    for char in '<>:"|?*':
        filename = filename.replace(char, '_')
    return filename or f'{int(item.index):05d}_{item.hash_value}.drm'


class TRLAU_OT_create_bigfile_backup(bpy.types.Operator):
    bl_idname = 'trlau.create_bigfile_backup'
    bl_label = 'Create Backup'
    bl_description = 'Create a backup folder containing the original Bigfile parts'
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        layout.label(text='This will create a backup folder containing a copy of the original bigfiles')
        layout.label(text="in case you'd want to revert back to vanilla game.")
        layout.separator()
        layout.label(text='Are you sure you want to proceed?')

    def execute(self, context):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        try:
            backup_dir = create_legend_bigfile_backup(bpy.path.abspath(props.bigfile_path))
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to create Bigfile backup: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Created Bigfile backup: {backup_dir}')
        return {'FINISHED'}


class TRLAU_OT_export_selected_bigfile_drms(bpy.types.Operator):
    bl_idname = 'trlau.export_selected_bigfile_drms'
    bl_label = 'Export DRM'
    bl_description = 'Export selected DRM records from matching Blender collections and write them into a repacked Bigfile'
    bl_options = {'REGISTER'}

    collection_name: StringProperty(
        name='Mesh Collection',
        description='Collection to export into the selected Bigfile DRM. Leave empty to auto-match by selected DRM path.',
        default='',
    )

    export_textures: BoolProperty(
        name='Export Textures',
        default=True,
        description='Export modified texture sections while writing the selected DRM records back into the Bigfile',
    )
    export_cloth: BoolProperty(
        name='Export Cloth',
        default=True,
        description='Export ClothSetup sections while writing the selected DRM records back into the Bigfile',
    )

    def invoke(self, context, event):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        selected = set(_selected_bigfile_record_indices(props))
        drm_items = _drm_record_items_for_indices(props, selected)
        target_collection_name = _AUTO_COLLECTION_ID
        if len(drm_items) == 1:
            collection = _find_drm_export_collection(drm_items[0].full_path)
            if collection is not None:
                target_collection_name = collection.name

        try:
            props.bigfile_export_collection_name = '' if target_collection_name == _AUTO_COLLECTION_ID else target_collection_name
            self.collection_name = props.bigfile_export_collection_name
        except Exception:
            pass
        return context.window_manager.invoke_props_dialog(self, width=560)

    def draw(self, context):
        layout = self.layout
        props = _archive_props(context)
        scene = getattr(context, 'scene', None)
        has_backup = False
        if props is not None and props.bigfile_path:
            try:
                has_backup = legend_bigfile_backup_exists(bpy.path.abspath(props.bigfile_path))
            except Exception:
                has_backup = False

        layout.label(text='DRM Bigfile Export Options')
        if props is not None:
            try:
                layout.prop_search(props, 'bigfile_export_collection_name', bpy.data, 'collections', text='Mesh Collection')
            except Exception:
                layout.prop(props, 'bigfile_export_collection_name', text='Mesh Collection')
            layout.label(text='Leave empty to auto-match selected DRM(s).')
            layout.prop(props, 'bigfile_export_textures')
            layout.prop(props, 'bigfile_export_cloth')
        else:
            layout.label(text='Mesh Collection: Auto-match selected DRM(s)')
        if not has_backup:
            layout.separator()
            layout.label(text="WARNING! You're attempting to export a DRM to an archive without a backup.", icon='ERROR')
            layout.label(text='Are you sure you want to proceed?')

    def execute(self, context):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        selected = set(_selected_bigfile_record_indices(props))
        if not selected:
            self.report({'ERROR'}, 'No Bigfile records selected')
            return {'CANCELLED'}
        drm_items = _drm_record_items_for_indices(props, selected)
        if not drm_items:
            self.report({'ERROR'}, 'No selected records are DRM files')
            return {'CANCELLED'}

        output_path = bpy.path.abspath(props.repack_output_path) if props.repack_output_path else ''
        if not output_path:
            output_path = bpy.path.abspath(props.bigfile_path)

        export_dir = Path(bpy.app.tempdir or tempfile.gettempdir()) / f'trlau_bigfile_drm_export_{uuid.uuid4().hex[:8]}'
        export_dir.mkdir(parents=True, exist_ok=True)

        replacements: dict[int, bytes] = {}
        exported = 0
        missing: list[str] = []
        failed: list[str] = []
        selected_collection_name = str(getattr(props, 'bigfile_export_collection_name', '') or getattr(self, 'collection_name', '') or '').strip()
        export_textures = bool(getattr(props, 'bigfile_export_textures', getattr(self, 'export_textures', True)))
        export_cloth = bool(getattr(props, 'bigfile_export_cloth', getattr(self, 'export_cloth', True)))
        explicit_collection = None
        if selected_collection_name and selected_collection_name != _AUTO_COLLECTION_ID:
            explicit_collection = bpy.data.collections.get(selected_collection_name)
            if explicit_collection is None:
                self.report({'ERROR'}, 'Choose a valid collection to export')
                return {'CANCELLED'}
            if len(drm_items) != 1:
                self.report({'ERROR'}, 'Choose one DRM record when exporting a specific collection, or use Auto-match for multiple records')
                return {'CANCELLED'}

        for item in drm_items:
            collection = explicit_collection or _find_drm_export_collection(item.full_path)
            if collection is None:
                missing.append(item.full_path or item.filename)
                continue
            export_path = export_dir / _safe_export_filename_from_record(item)
            try:
                result = bpy.ops.export_scene.trlau_level(
                    'EXEC_DEFAULT',
                    filepath=str(export_path),
                    collection_name=collection.name,
                    export_textures=export_textures,
                    export_cloth=export_cloth,
                    debug=False,
                )
                if 'FINISHED' not in result or not export_path.exists():
                    failed.append(f'{item.full_path}: export did not finish')
                    continue
                replacements[int(item.index)] = export_path.read_bytes()
                exported += 1
            except Exception as exc:
                failed.append(f'{item.full_path}: {exc}')

        if not replacements:
            detail = ''
            if missing:
                detail = f' No matching collection for: {missing[0]}'
            elif failed:
                detail = f' {failed[0]}'
            self.report({'ERROR'}, f'No DRM records were exported.{detail}')
            return {'CANCELLED'}

        try:
            output = repack_legend_bigfile_with_replacements(
                bpy.path.abspath(props.bigfile_path),
                replacements,
                output_path,
                compress_replaced_records=True,
            )
            props.repack_output_path = str(output)
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to write Bigfile replacements: {exc}')
            return {'CANCELLED'}

        message = f'Exported {exported} DRM file(s) into {Path(output).name}'
        if missing:
            message += f'; missing collection for {len(missing)}'
        if failed:
            message += f'; failed {len(failed)}'
        self.report({'INFO'}, message)
        return {'FINISHED'}


class TRLAU_OT_unpack_bigfile(bpy.types.Operator):
    bl_idname = 'trlau.unpack_bigfile'
    bl_label = 'Unpack Bigfile'
    bl_description = 'Unpack all records from the opened Tomb Raider Legend Bigfile'
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = _archive_props(context)
        if props is None or not props.bigfile_path:
            self.report({'ERROR'}, 'Select a Bigfile path first')
            return {'CANCELLED'}
        try:
            source_path = bpy.path.abspath(props.bigfile_path)
            output_dir = bpy.path.abspath(props.unpack_output_directory) if props.unpack_output_directory else ''
            info = inspect_legend_bigfile(source_path)
            allow_incomplete = not bool(info.get('archiveComplete', True))
            manifest_path = unpack_legend_bigfile(
                source_path,
                output_dir,
                filename_map=_load_selected_filename_map(props),
                allow_incomplete=allow_incomplete,
            )
            props.repack_manifest_path = str(manifest_path)
            if not props.unpack_output_directory:
                props.unpack_output_directory = str(manifest_path.parent)
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to unpack bigfile: {exc}')
            return {'CANCELLED'}
        extracted_count = int(info.get('recordCount', 0))
        skipped_count = 0
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            extracted_count = int(manifest.get('recordCount', extracted_count))
            skipped_count = int(manifest.get('skippedUnavailableCount', 0))
        except Exception:
            pass
        if skipped_count:
            self.report(
                {'WARNING'},
                f'Archive is incomplete; unpacked {extracted_count} available record(s), skipped {skipped_count} unavailable record(s) to {manifest_path.parent}',
            )
        else:
            self.report({'INFO'}, f'Unpacked {extracted_count} record(s) to {manifest_path.parent}')
        return {'FINISHED'}


class TRLAU_OT_repack_bigfile(bpy.types.Operator):
    bl_idname = 'trlau.repack_bigfile'
    bl_label = 'Repack Bigfile'
    bl_description = 'Repack a Tomb Raider Legend bigfile archive from an unpack manifest'
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = _archive_props(context)
        if props is None or not props.repack_manifest_path:
            self.report({'ERROR'}, 'Select a manifest.json produced by Unpack Bigfile')
            return {'CANCELLED'}
        try:
            manifest_path = bpy.path.abspath(props.repack_manifest_path)
            output_path = bpy.path.abspath(props.repack_output_path) if props.repack_output_path else ''
            output = repack_legend_bigfile(
                manifest_path,
                output_path,
                compress_marked_records=props.compress_marked_records,
            )
            info = inspect_legend_bigfile(output)
            props.repack_output_path = str(output)
        except BigfileError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, f'Failed to repack bigfile: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'}, f'Repacked {info["recordCount"]} record(s) to {output}')
        return {'FINISHED'}


class VIEW3D_PT_trlau_archive_tools(bpy.types.Panel):
    bl_label = 'Archive Tools'
    bl_idname = 'VIEW3D_PT_trlau_archive_tools'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'cdcEditor'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = _archive_props(context)
        if props is None:
            layout.label(text='Archive settings unavailable', icon='ERROR')
            return

        box = layout.box()
        box.label(text='Bigfile')
        row = box.row(align=True)
        row.prop(props, 'bigfile_path')
        row.operator('trlau.select_bigfile_path', text='', icon='FILE_FOLDER')
        row = box.row(align=True)
        row.label(text='File List')
        row.menu('TRLAU_MT_bigfile_filelists', text=_filelist_display_label(props.filelist_name))
        if props.filelist_name == _CUSTOM_FILELIST_ID:
            row = box.row(align=True)
            row.prop(props, 'custom_filelist_path')
            row.operator('trlau.select_bigfile_filelist_path', text='', icon='FILE_FOLDER')
        box.prop(props, 'unpack_output_directory')
        row = box.row(align=True)
        row.operator('trlau.open_bigfile_browser', icon='FILE_FOLDER')

        box = layout.box()
        box.label(text='Repack')
        box.prop(props, 'repack_manifest_path')
        row = box.row(align=True)
        row.prop(props, 'repack_output_path')
        row.operator('trlau.select_bigfile_output_path', text='', icon='FILE_FOLDER')
        box.prop(props, 'compress_marked_records')
        box.operator('trlau.repack_bigfile', icon='FILE_TICK')



def _ensure_bigfile_operator_property_annotations() -> None:
    TRLAU_PG_archive_tools.__annotations__.update({
        'bigfile_export_collection_name': StringProperty(
            name='Mesh Collection',
            description='Collection to export into the selected Bigfile DRM. Leave empty to auto-match by selected DRM path.',
            default='',
        ),
        'bigfile_export_textures': BoolProperty(
            name='Export Textures',
            description='Export modified texture sections while writing selected DRM records back into the Bigfile',
            default=True,
        ),
        'bigfile_export_cloth': BoolProperty(
            name='Export Cloth',
            description='Export ClothSetup sections while writing selected DRM records back into the Bigfile',
            default=True,
        ),
    })

    TRLAU_OT_import_selected_bigfile_records.__annotations__.update({
        'show_model_settings': BoolProperty(name='Model Settings', default=True, options={'SKIP_SAVE'}),
        'show_advanced_settings': BoolProperty(name='Advanced Settings', default=False, options={'SKIP_SAVE'}),
        'show_level_settings': BoolProperty(name='Level Settings', default=True, options={'SKIP_SAVE'}),
        'platform': EnumProperty(
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
        ),
        'import_textures': BoolProperty(name='Import Textures', default=True, description='Enable or disable texture importing'),
        'import_main_model_only': BoolProperty(name='Main Model Only', default=True, description='Only import the first referenced model (.drm, .obj)'),
        'import_next_gen_model': BoolProperty(name='Import Next Gen Model', default=False, description='Import the next-generation render model referenced by cdcRenderDataID when present'),
        'import_all_textures': BoolProperty(name='Import All Textures', default=False, description='Import all textures found in the .drm or folder'),
        'import_all_animations': BoolProperty(name='Import All Animations', default=False, description='Import all animations found in the .drm or folder'),
        'import_armature_only': BoolProperty(name='Import Armature Only', default=False, description='Import only the armature'),
        'import_bounding_boxes': BoolProperty(name='Import Bounding Boxes', default=False, description="Import bones' bounding boxes"),
        'import_kdnodes': BoolProperty(name='Import KDNodes', default=False, description='Import TerrainGroup collision KDNode empties when importing levels'),
        'import_audio': BoolProperty(name='Import Audio', default=True, description='Import PC level SFX markers and decode referenced Wave sections as Blender speakers', options={'HIDDEN', 'SKIP_SAVE'}),
        'split_terrain_groups_by_strip': BoolProperty(name='Split TerrainGroups by Strip', default=False, description='Import TerrainGroups as invidual strip meshes'),
        'debug': BoolProperty(name='Debug', default=False, description='Enable console debug log'),
        'separate_by_drawgroup': BoolProperty(
            name='Separate by Drawgroup',
            default=False,
            description='Split the imported model into separate meshes by drawgroup',
            update=TRLAU_OT_import_selected_bigfile_records._update_separate_by_drawgroup,
        ),
        'separate_by_material': BoolProperty(
            name='Separate by Material',
            default=False,
            description='Split the imported model into separate meshes by texture strip/material',
            update=TRLAU_OT_import_selected_bigfile_records._update_separate_by_material,
        ),
    })

    TRLAU_OT_export_selected_bigfile_drms.__annotations__.update({
        'collection_name': StringProperty(
            name='Mesh Collection',
            description='Collection to export into the selected Bigfile DRM. Leave empty to auto-match by selected DRM path.',
            default='',
        ),
        'export_textures': BoolProperty(
            name='Export Textures',
            default=True,
            description='Export modified texture sections while writing the selected DRM records back into the Bigfile',
        ),
        'export_cloth': BoolProperty(
            name='Export Cloth',
            default=True,
            description='Export ClothSetup sections while writing the selected DRM records back into the Bigfile',
        ),
    })


_ensure_bigfile_operator_property_annotations()


__all__ = [
    'TRLAU_PG_bigfile_record_entry',
    'TRLAU_PG_archive_tools',
    'TRLAU_UL_bigfile_records',
    'TRLAU_OT_select_bigfile_path',
    'TRLAU_OT_select_bigfile_output_path',
    'TRLAU_OT_select_bigfile_filelist_path',
    'TRLAU_OT_set_bigfile_filelist',
    'TRLAU_MT_bigfile_filelists',
    'TRLAU_OT_toggle_bigfile_folder',
    'TRLAU_OT_open_bigfile_browser',
    'TRLAU_OT_select_all_bigfile_records',
    'TRLAU_OT_unpack_selected_bigfile_records',
    'TRLAU_OT_import_selected_bigfile_records',
    'TRLAU_OT_export_selected_bigfile_drms',
    'TRLAU_OT_create_bigfile_backup',
    'TRLAU_OT_unpack_bigfile',
    'TRLAU_OT_repack_bigfile',
    'VIEW3D_PT_trlau_archive_tools',
]
