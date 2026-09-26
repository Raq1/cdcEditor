from __future__ import annotations

from pathlib import Path

from ..platforms.common.drm_container import DRMContainerParser
from ..platforms.pc.tr7ae_model import TRModelParser
from ..platforms.pc.tr8_model import TR8ModelParser
from ..platforms.xbox.tr7ae_model import TRXboxModelParser
from ..platforms.ps2.tr7ae_model import TRPS2ModelParser
from ..platforms.psp.tr7ae_model import TRPSPModelParser
from ..platforms.ps3.tr7ae_model import TRPS3ModelParser
from ..platforms.xbox360.tr7ae_model import TRXbox360ModelParser
from ..platforms.pc.tr7ae_object import TRObjectParser
from .level_properties import _find_related_model_root

MODEL_IMPORT_PLATFORM_ITEMS = (
    ('PC', 'PC', ''),
    ('UNDERWORLD', 'Underworld', ''),
    ('PS2', 'PS2', ''),
    ('PS3', 'PS3', ''),
    ('PSP', 'PSP', ''),
    ('XBOX', 'Xbox', ''),
    ('XBOX360', 'Xbox 360', ''),
    ('GAMECUBE', 'Nintendo', ''),
)


def _iter_object_descendants(root):
    stack = list(getattr(root, 'children', []) or [])
    while stack:
        obj = stack.pop(0)
        yield obj
        stack[0:0] = list(getattr(obj, 'children', []) or [])


def _find_model_armature(model_root):
    if model_root is None:
        return None
    if getattr(model_root, 'type', None) == 'ARMATURE':
        return model_root
    for obj in _iter_object_descendants(model_root):
        if getattr(obj, 'type', None) == 'ARMATURE':
            return obj
    return None


def _selected_model_root(context):
    obj = getattr(context, 'object', None)
    if obj is None:
        return None
    if bool(getattr(obj, 'trlau_is_model_empty', False)):
        return obj
    return _find_related_model_root(obj, context)


def _mesh_parser_for_platform(filepath: str, platform: str, *, import_hinfo: bool = True):
    platform = str(platform or 'PC').upper()
    if platform in {'UNDERWORLD', 'TR8', 'TRU'} or TR8ModelParser.looks_like_tr8mesh(filepath):
        return TR8ModelParser(filepath, import_hinfo=import_hinfo, import_markups=False)
    if platform == 'XBOX':
        return TRXboxModelParser(filepath, import_hinfo=import_hinfo, import_markups=False)
    if platform == 'PS2':
        return TRPS2ModelParser(filepath, import_hinfo=import_hinfo, import_markups=False)
    if platform == 'PSP':
        return TRPSPModelParser(filepath, import_hinfo=import_hinfo, import_markups=False)
    if platform == 'PS3':
        return TRPS3ModelParser(filepath, import_hinfo=import_hinfo, import_markups=False)
    if platform == 'XBOX360':
        return TRXbox360ModelParser(filepath, import_hinfo=import_hinfo, import_markups=False)
    endian = '>' if platform == 'GAMECUBE' else '<'
    return TRModelParser(filepath, import_hinfo=import_hinfo, import_markups=False, endian=endian)


def _parse_main_model_source(filepath: str, platform: str, *, import_hinfo: bool = True):
    source_path = Path(filepath)
    suffix = source_path.suffix.lower()
    platform = str(platform or 'PC').upper()
    if suffix in {'.tr7aemesh', '.tr8mesh'} or TR8ModelParser.looks_like_tr8mesh(str(source_path)):
        return _mesh_parser_for_platform(str(source_path), platform, import_hinfo=import_hinfo).parse()
    if suffix != '.drm':
        raise ValueError('Select a .tr7aemesh, .tr8mesh, or .drm file')

    endian = '>' if platform in {'GAMECUBE', 'PS3', 'XBOX360'} else '<'
    drm_parser = DRMContainerParser(str(source_path), endian=endian, decompress_derickw=(platform in {'PSP', 'PS3', 'XBOX360'}))
    with drm_parser.temporary_extract_sections() as (_extract_dir, extracted_paths, _sections):
        if not extracted_paths:
            raise ValueError('No sections were found in the DRM container')

        drm_game = drm_parser.detect_game(_sections)
        if drm_game == 'underworld' or platform in {'UNDERWORLD', 'TR8', 'TRU'}:
            object_path = next((str(path) for path in extracted_paths if Path(path).suffix.lower() == '.obj'), str(extracted_paths[0]))
            refs = TRObjectParser(object_path, endian=endian).parse_underworld_mesh_references()
            if not refs:
                raise ValueError('No Underworld mesh references were found in the main DRM object section')
            parser = TR8ModelParser(refs[0].tr8mesh_filepath, import_hinfo=import_hinfo, import_markups=False)
            model = parser.parse()
            return parser.attach_underworld_model_hinfo(model, refs[0].tr8model_filepath)

        object_path = str(extracted_paths[0])
        model_refs = TRObjectParser(object_path, endian=endian).parse_model_references()
        if not model_refs:
            raise ValueError('No model references were found in the main DRM object section')
        return _mesh_parser_for_platform(model_refs[0].target_filepath, platform, import_hinfo=import_hinfo).parse()
