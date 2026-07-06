from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ...core.log import logger
from ..common.section import SectionContextCache, resolve_pointer
from .tr7ae_cloth import TRClothParser


@dataclass(slots=True)
class ObjectModelReference:
    index: int
    raw_value: int
    target_filepath: str


@dataclass(slots=True)
class ObjectAnimationReference:
    index: int
    anim_id: int
    pad: int
    raw_list_value: int


@dataclass(slots=True)
class UnderworldMeshReference:
    objectref_filepath: str
    object_id: int
    object_filepath: str
    model_index: int
    tr8model_id: int
    tr8model_filepath: str
    cdc_modeldata_id: int
    tr8mesh_filepath: str

    @property
    def mesh_meta_filepath(self) -> str:
        return self.tr8model_filepath

    @property
    def cdc_render_model_id(self) -> int:
        return self.cdc_modeldata_id


OBJECT_STRUCT_DOCUMENTATION = """\
TR7/TRA object root layout used by this parser:

struct Object
{
  int oflags;
  int oflags2;
  int uniqueID;
  unsigned int guiID;
  int functionTableID;
  void *obsoleteSoundBank;
  __int16 numModels;
  __int16 numAnims;
  __int16 numAnimPatterns;
  Model **modelList;
  AnimListEntry *animList;
  AnimFxHeader **animFXList;
  AnimScriptObject **animPatternList;
  int introDist;
  int vvIntroDist;
  int removeDist;
  int vvRemoveDist;
  ObjectBase *baseData;
  void *data;
  char *name;
  SFXData *soundData;
  __int16 sectionA;
  __int16 sectionB;
  __int16 sectionC;
  __int16 numberOfEffects;
  ObjectEffectFXA *effectList;
  _GenericFXObject *effectData;
  ObjectDTPData *objectDTPData;
  struct VramLink *textureLoadList;
  unsigned __int16 *childObjectList;
  int lod1Dist;
  int lod2Dist;
  char lod1Model;
  char lod2Model;
  char shadowModel;
  char lightingOverride;
  float maxCheckeeDistance;
  ClothSetup **rdSetupList;
};
"""


class TRObjectParser:

    def __init__(self, filepath: str, endian: str = "<"):
        self.filepath = filepath
        self.endian = endian

    def _ps3_fallback_model_references(self, cache: SectionContextCache, context) -> List[ObjectModelReference]:
        """Recover PS3 model references from the relocation table.

        Most PS3 object roots still resemble the PC object header, but some
        extracted roots can fail the generic model-list validation even though
        the model list is visible in the relocation table.  In the supplied PS3
        actor DRMs the object header stores numModels at local +0x18 and the
        modelList pointer field at local +0x20; the list itself is a contiguous
        run of relocated 32-bit model pointers.  This fallback reconstructs the
        references from those relocated pointer fields rather than returning an
        empty import.
        """
        if self.endian != ">":
            return []

        def _read_i16(local_offset: int) -> int:
            previous = context.reader.tell()
            try:
                if local_offset < 0 or context.data_start + local_offset + 2 > context.data_end:
                    return 0
                context.reader.seek(context.data_start + local_offset)
                return int(context.reader.i16())
            except Exception:
                return 0
            finally:
                try:
                    context.reader.seek(previous)
                except Exception:
                    pass

        def _read_u32(ctx, abs_offset: int) -> int:
            previous = ctx.reader.tell()
            try:
                if abs_offset < 0 or abs_offset + 4 > ctx.file_size:
                    return 0
                ctx.reader.seek(abs_offset)
                return int(ctx.reader.u32())
            except Exception:
                return 0
            finally:
                try:
                    ctx.reader.seek(previous)
                except Exception:
                    pass

        def _target_exists(section_index: int):
            try:
                return cache.resolve_target_context(context, int(section_index))
            except Exception:
                return None

        def _make_refs_from_list(list_ctx, list_abs: int, count: int) -> List[ObjectModelReference]:
            refs: List[ObjectModelReference] = []
            if count <= 0 or count > 256:
                return refs
            for index in range(int(count)):
                entry_abs = int(list_abs) + (index * 4)
                field_local_offset = entry_abs - list_ctx.data_start
                relocation = list_ctx.section_info.relocations_by_offset.get(field_local_offset)
                if relocation is None:
                    return []
                try:
                    target_ctx = cache.resolve_target_context(list_ctx, relocation.section_index_or_type)
                except Exception:
                    return []
                raw_value = _read_u32(list_ctx, entry_abs)
                refs.append(ObjectModelReference(index=index, raw_value=raw_value, target_filepath=str(Path(target_ctx.filepath))))
            return refs

        num_models = _read_i16(0x18)
        if not (0 < num_models <= 64):
            num_models = 0

        # Preferred PS3 object-root layout: local +0x20 points to a relocated
        # model pointer list, and local +0x18 is the model count.
        if num_models:
            for list_field_local_offset in (0x20, 0x24, 0x2C, 0x30):
                try:
                    context.reader.seek(context.data_start + list_field_local_offset)
                    model_list_raw = context.reader.u32()
                    model_list_ctx, model_list_abs = resolve_pointer(
                        cache,
                        context,
                        list_field_local_offset,
                        model_list_raw,
                    )
                except Exception:
                    continue
                refs = _make_refs_from_list(model_list_ctx, model_list_abs, int(num_models))
                if refs:
                    logger.warning(
                        'Recovered %d PS3 model references from relocation-backed object list in %s at local offset 0x%X',
                        len(refs),
                        context.file_name,
                        int(list_field_local_offset),
                    )
                    return refs

        # Last-resort scan: find a short contiguous run of relocated pointer
        # fields near the object header.  This avoids using the much larger
        # texture/object dependency tables later in the root section.
        reloc_by_offset = context.section_info.relocations_by_offset
        candidate_counts = [num_models] if num_models else []
        candidate_counts.extend([count for count in range(1, 17) if count not in candidate_counts])
        for count in candidate_counts:
            if count <= 0:
                continue
            max_start = 0x400
            for start_offset in sorted(offset for offset in reloc_by_offset.keys() if 0 <= int(offset) < max_start):
                offsets = [int(start_offset) + (i * 4) for i in range(int(count))]
                if not all(offset in reloc_by_offset for offset in offsets):
                    continue
                refs: List[ObjectModelReference] = []
                for index, offset in enumerate(offsets):
                    relocation = reloc_by_offset[offset]
                    target_ctx = _target_exists(relocation.section_index_or_type)
                    if target_ctx is None:
                        refs = []
                        break
                    raw_value = _read_u32(context, context.data_start + offset)
                    refs.append(ObjectModelReference(index=index, raw_value=raw_value, target_filepath=str(Path(target_ctx.filepath))))
                if refs:
                    logger.warning(
                        'Recovered %d PS3 model references by scanning contiguous relocations in %s at local offset 0x%X',
                        len(refs),
                        context.file_name,
                        int(start_offset),
                    )
                    return refs

        return []


    @staticmethod
    def _normalise_hex_id(value: int) -> str:
        return f'{int(value) & 0xFFFFFFFF:x}'

    @staticmethod
    def _section_index_from_filename(path: Path) -> int:
        try:
            return int(path.stem.split('_', 1)[0], 10)
        except Exception:
            return 0x7FFFFFFF

    @staticmethod
    def _find_file_by_id(directory: Path, file_id: int) -> Path | None:
        # TR8 section IDs are not globally unique in every DRM.  For example,
        # lara.drm contains multiple extracted sections named *_352.*.  The object
        # root points to the earliest section-table occurrence, while a plain
        # lexicographic sort can pick 1616_352 before 3_352.  Collect all matches
        # and prefer the lowest numeric section index.
        id_hex = TRObjectParser._normalise_hex_id(file_id).lower()
        id_hex_no_zero = id_hex.lstrip('0') or '0'
        matches: list[Path] = []
        for candidate in Path(directory).iterdir():
            if not candidate.is_file():
                continue
            stem = candidate.stem.lower()
            parts = stem.split('_')[1:]
            matched = False
            for part in parts:
                part_no_zero = part.lstrip('0') or '0'
                if part == id_hex or part_no_zero == id_hex_no_zero:
                    matched = True
                    break
            if matched or f'_{id_hex}' in stem:
                matches.append(candidate)
        if not matches:
            return None
        return sorted(matches, key=lambda path: (TRObjectParser._section_index_from_filename(path), path.name.lower()))[0]

    @staticmethod
    def _read_context_u32(context, local_offset: int) -> int:
        if local_offset < 0 or context.data_start + local_offset + 4 > context.data_end:
            raise ValueError(f'u32 local offset 0x{local_offset:X} is outside {context.file_name}')
        previous = context.reader.tell()
        try:
            context.reader.seek(context.data_start + int(local_offset))
            return int(context.reader.u32())
        finally:
            try:
                context.reader.seek(previous)
            except Exception:
                pass

    @staticmethod
    def _read_context_u32_abs(context, absolute_offset: int) -> int:
        if absolute_offset < 0 or absolute_offset + 4 > context.file_size:
            raise ValueError(f'u32 absolute offset 0x{absolute_offset:X} is outside {context.file_name}')
        previous = context.reader.tell()
        try:
            context.reader.seek(int(absolute_offset))
            return int(context.reader.u32())
        finally:
            try:
                context.reader.seek(previous)
            except Exception:
                pass

    @staticmethod
    def _section_file_id(context) -> int:
        try:
            return int(context.section_info.section_id) & 0xFFFFFFFF
        except Exception:
            return 0

    def _read_context_u32_safe(self, context, local_offset: int) -> int | None:
        try:
            return self._read_context_u32(context, local_offset)
        except Exception:
            return None

    def _read_context_u32_abs_safe(self, context, absolute_offset: int) -> int | None:
        try:
            return self._read_context_u32_abs(context, absolute_offset)
        except Exception:
            return None

    def _parse_underworld_objectref_ids(self, cache: SectionContextCache, objectref_context) -> list[int]:
        """Read the TR8 object-reference root section.

        TR9 objectref files are documented as a single RefDefinitions block plus
        one Ref.  TR8/Underworld PC DRMs seen so far use a simpler root section:
        a small contiguous list of resource/file IDs at the start of the payload.
        Those IDs may reference several object-ish files, not just one object.
        """
        ids: list[int] = []
        root_id = self._section_file_id(objectref_context)
        max_words = min(64, max(0, objectref_context.data_size // 4))

        for word_index in range(max_words):
            value = self._read_context_u32_safe(objectref_context, word_index * 4)
            if value is None:
                break
            value = int(value) & 0xFFFFFFFF
            if value == 0:
                # The sample root uses a zero terminator after the leading object
                # ID list.  Stop here once at least one plausible reference was read.
                if ids:
                    break
                continue
            if value == root_id:
                continue
            target_path = self._find_file_by_id(Path(objectref_context.filepath).parent, value)
            if target_path is None or target_path.resolve() == Path(objectref_context.filepath).resolve():
                if ids:
                    break
                continue
            if value not in ids:
                ids.append(value)

        if not ids:
            # Fallback for slightly different TR8 roots: scan the whole small root
            # payload for values that resolve to extracted sibling sections.
            seen: set[int] = set()
            for word_index in range(max_words):
                value = self._read_context_u32_safe(objectref_context, word_index * 4)
                if value is None:
                    continue
                value = int(value) & 0xFFFFFFFF
                if value == 0 or value == root_id or value in seen:
                    continue
                target_path = self._find_file_by_id(Path(objectref_context.filepath).parent, value)
                if target_path is None or target_path.resolve() == Path(objectref_context.filepath).resolve():
                    continue
                seen.add(value)
                ids.append(value)

        logger.info(
            'TR8 objectref %s references %d object candidate(s): %s',
            objectref_context.file_name,
            len(ids),
            ', '.join(f'0x{value:X}' for value in ids) if ids else 'none',
        )
        return ids

    def _parse_underworld_object_model_ids(self, cache: SectionContextCache, object_context) -> list[int]:
        """Read model resource IDs from a TR8 object section.

        In the supplied TR8 Lara DRM, the object's render/model list is stored as:
          local +0xA0 = model count
          local +0xA4 = pointer to a u32 model-resource-ID array
        Keep a relocation-backed validation pass so unrelated files referenced by
        the objectref section are ignored instead of treated as objects.
        """
        directory = Path(object_context.filepath).parent
        candidates: list[tuple[int, int]] = [(0xA0, 0xA4)]

        # Conservative fallback: look for a small count immediately followed by a
        # relocated pointer.  This covers minor TR8 layout drift without scanning
        # unrelated large tables deep in the object file.
        for count_offset in range(0x40, min(0x140, object_context.data_size - 8), 4):
            pointer_offset = count_offset + 4
            if pointer_offset not in object_context.section_info.relocations_by_offset:
                continue
            if (count_offset, pointer_offset) not in candidates:
                candidates.append((count_offset, pointer_offset))

        best_ids: list[int] = []
        best_offsets: tuple[int, int] | None = None
        for count_offset, pointer_offset in candidates:
            count = self._read_context_u32_safe(object_context, count_offset)
            pointer_raw = self._read_context_u32_safe(object_context, pointer_offset)
            if count is None or pointer_raw is None:
                continue
            count = int(count)
            if count <= 0 or count > 64:
                continue
            if pointer_offset not in object_context.section_info.relocations_by_offset:
                continue
            try:
                list_ctx, list_abs = resolve_pointer(cache, object_context, pointer_offset, int(pointer_raw))
            except Exception:
                continue
            if list_abs <= 0 or list_abs + (count * 4) > list_ctx.file_size:
                continue

            ids: list[int] = []
            valid_ids = 0
            for index in range(count):
                value = self._read_context_u32_abs_safe(list_ctx, list_abs + (index * 4))
                if value is None:
                    ids = []
                    break
                value = int(value) & 0xFFFFFFFF
                ids.append(value)
                if value != 0 and self._find_file_by_id(directory, value) is not None:
                    valid_ids += 1
            if not ids:
                continue
            if valid_ids <= 0:
                continue
            if valid_ids > len(best_ids):
                best_ids = ids
                best_offsets = (count_offset, pointer_offset)
            if (count_offset, pointer_offset) == (0xA0, 0xA4):
                break

        if best_ids:
            assert best_offsets is not None
            logger.info(
                'TR8 object %s model list: count=%d count_offset=0x%X list_offset=0x%X ids=%s',
                object_context.file_name,
                len(best_ids),
                best_offsets[0],
                best_offsets[1],
                ', '.join(f'0x{value:X}' for value in best_ids),
            )
        else:
            logger.debug('TR8 object candidate %s has no validated model list', object_context.file_name)
        return best_ids

    def _parse_underworld_modeldata_id(self, model_context) -> int | None:
        """Read the cdcModelData/tr8mesh resource ID from a TR8 model section.

        TR8 differs from the TR9 template's direct RefDefinitions-style model data
        ref.  The Lara sample stores the cdcModelData resource ID at local +0x64.
        The surrounding model header begins with the same 0x04C20453 signature seen
        in multiple model entries, so validate that before accepting the ID.
        """
        magic = self._read_context_u32_safe(model_context, 0x00)
        if magic != 0x04C20453:
            logger.debug('TR8 model candidate %s rejected: unexpected magic 0x%X', model_context.file_name, int(magic or 0))
            return None

        value = self._read_context_u32_safe(model_context, 0x64)
        if value is None:
            return None
        value = int(value) & 0xFFFFFFFF
        if value == 0 or value == 0xFFFFFFFF:
            logger.debug('TR8 model %s has no cdcModelData resource ID at local 0x64', model_context.file_name)
            return None
        return value

    def parse_underworld_mesh_references(self) -> List[UnderworldMeshReference]:
        """Follow the TR8 objectref -> object -> model -> cdcModelData chain.

        This intentionally stops before decoding mesh geometry.  It resolves the
        final TR8 mesh/data section so the importer can log the file that will
        become the actual mesh source in the next implementation step.
        """
        cache = SectionContextCache(self.filepath, endian=self.endian)
        try:
            objectref_context = cache.get_root_context()
            directory = Path(objectref_context.filepath).parent
            object_ids = self._parse_underworld_objectref_ids(cache, objectref_context)
            if not object_ids:
                # Backward-compatible fallback for early TR8 probe builds.
                first_file_id = self._read_context_u32(objectref_context, 0)
                object_ids = [int(first_file_id)] if int(first_file_id) else []

            refs: List[UnderworldMeshReference] = []
            seen_meshes: set[tuple[int, int, int]] = set()
            for object_id in object_ids:
                object_path = self._find_file_by_id(directory, object_id)
                if object_path is None:
                    logger.debug('TR8 objectref %s object id=0x%X did not resolve to an extracted section', objectref_context.file_name, object_id)
                    continue
                try:
                    object_context = cache.get_context(object_path)
                except Exception as exc:
                    logger.debug('Could not open TR8 object candidate id=0x%X at %s: %s', object_id, object_path.name, exc)
                    continue

                model_ids = self._parse_underworld_object_model_ids(cache, object_context)
                if not model_ids:
                    continue

                for model_index, tr8model_id in enumerate(model_ids):
                    if int(tr8model_id) == 0:
                        continue
                    tr8model_path = self._find_file_by_id(directory, tr8model_id)
                    if tr8model_path is None:
                        logger.debug('TR8 object %s model[%d] id=0x%X did not resolve to an extracted section', object_context.file_name, model_index, tr8model_id)
                        continue
                    try:
                        tr8model_context = cache.get_context(tr8model_path)
                    except Exception as exc:
                        logger.debug('Could not open TR8 model id=0x%X at %s: %s', tr8model_id, tr8model_path.name, exc)
                        continue

                    cdc_modeldata_id = self._parse_underworld_modeldata_id(tr8model_context)
                    if cdc_modeldata_id is None:
                        continue
                    tr8mesh_path = self._find_file_by_id(directory, cdc_modeldata_id)
                    if tr8mesh_path is None:
                        logger.debug('TR8 model %s cdcModelData id=0x%X did not resolve to an extracted section', tr8model_context.file_name, cdc_modeldata_id)
                        continue

                    key = (int(object_id), int(tr8model_id), int(cdc_modeldata_id))
                    if key in seen_meshes:
                        continue
                    seen_meshes.add(key)
                    logger.info(
                        'TR8 mesh chain: objectref=%s object_id=0x%X object=%s model[%d]=0x%X (%s) cdcModelData=0x%X -> %s',
                        objectref_context.file_name,
                        int(object_id),
                        object_context.file_name,
                        int(model_index),
                        int(tr8model_id),
                        tr8model_context.file_name,
                        int(cdc_modeldata_id),
                        Path(tr8mesh_path).name,
                    )
                    refs.append(
                        UnderworldMeshReference(
                            objectref_filepath=str(Path(objectref_context.filepath)),
                            object_id=int(object_id),
                            object_filepath=str(Path(object_path)),
                            model_index=int(model_index),
                            tr8model_id=int(tr8model_id),
                            tr8model_filepath=str(Path(tr8model_path)),
                            cdc_modeldata_id=int(cdc_modeldata_id),
                            tr8mesh_filepath=str(Path(tr8mesh_path)),
                        )
                    )

            if not refs:
                raise ValueError(f'No TR8 mesh reference chain was found in {objectref_context.file_name}')
            return refs
        finally:
            cache.close()

    def parse_cloth_setups(self):
        """Parse ClothSetup records referenced by Object.rdSetupList."""
        return TRClothParser(self.filepath, endian=self.endian).parse()

    def parse_model_references(self) -> List[ObjectModelReference]:
        cache = SectionContextCache(self.filepath, endian=self.endian)
        try:
            context = cache.get_root_context()
            br = context.reader
            br.seek(context.data_start)

            br.skip(4)   # oflags
            br.skip(4)   # oflags2
            br.skip(4)   # uniqueID
            br.skip(4)   # guiID
            br.skip(4)   # functionTableID
            br.skip(4)   # obsoleteSoundBank
            num_models = br.i16()
            br.skip(2)   # numAnims
            br.skip(2)   # numAnimPatterns
            br.skip(2)   # 4-byte alignment padding before modelList

            model_list_field_local_offset = br.tell() - context.data_start
            model_list_raw = br.u32()
            model_list_ctx, model_list_abs = resolve_pointer(
                cache,
                context,
                model_list_field_local_offset,
                model_list_raw,
            )

            logger.debug(
                'Parsed object header from %s: num_models=%d model_list=0x%X -> %s:0x%X',
                context.file_name,
                num_models,
                model_list_raw,
                model_list_ctx.file_name,
                model_list_abs,
            )

            if num_models <= 0 or model_list_abs <= 0:
                fallback_refs = self._ps3_fallback_model_references(cache, context)
                return fallback_refs

            entry_size = 4
            list_end = model_list_abs + (num_models * entry_size)
            if list_end > model_list_ctx.file_size:
                logger.warning(
                    'Skipping modelList read in %s: abs=0x%X count=%d exceeds %s file_size=0x%X',
                    context.file_name,
                    model_list_abs,
                    num_models,
                    model_list_ctx.file_name,
                    model_list_ctx.file_size,
                )
                fallback_refs = self._ps3_fallback_model_references(cache, context)
                return fallback_refs

            refs: List[ObjectModelReference] = []
            model_list_ctx.reader.seek(model_list_abs)
            for index in range(num_models):
                entry_start = model_list_ctx.reader.tell()
                field_local_offset = entry_start - model_list_ctx.data_start
                raw_value = model_list_ctx.reader.u32()
                target_ctx, _absolute_offset = resolve_pointer(
                    cache,
                    model_list_ctx,
                    field_local_offset,
                    raw_value,
                )
                refs.append(
                    ObjectModelReference(
                        index=index,
                        raw_value=raw_value,
                        target_filepath=str(Path(target_ctx.filepath)),
                    )
                )
                logger.debug(
                    'Object model[%d]: raw=0x%X -> file=%s',
                    index,
                    raw_value,
                    target_ctx.file_name,
                )

            if not refs:
                return self._ps3_fallback_model_references(cache, context)
            return refs
        finally:
            cache.close()

    def parse_animation_references(self) -> List[ObjectAnimationReference]:
        cache = SectionContextCache(self.filepath, endian=self.endian)
        try:
            context = cache.get_root_context()
            br = context.reader
            br.seek(context.data_start)

            br.skip(4)   # oflags
            br.skip(4)   # oflags2
            br.skip(4)   # uniqueID
            br.skip(4)   # guiID
            br.skip(4)   # functionTableID
            br.skip(4)   # obsoleteSoundBank
            br.skip(2)   # numModels
            num_anims = br.i16()
            br.skip(2)   # numAnimPatterns
            br.skip(2)   # 4-byte alignment padding before modelList
            br.skip(4)   # modelList

            anim_list_field_local_offset = br.tell() - context.data_start
            anim_list_raw = br.u32()
            if num_anims <= 0 or anim_list_raw == 0:
                return []

            anim_list_ctx, anim_list_abs = resolve_pointer(
                cache,
                context,
                anim_list_field_local_offset,
                anim_list_raw,
            )

            logger.debug(
                'Parsed object animation list from %s: num_anims=%d anim_list=0x%X -> %s:0x%X',
                context.file_name,
                num_anims,
                anim_list_raw,
                anim_list_ctx.file_name,
                anim_list_abs,
            )

            if anim_list_abs <= 0:
                return []

            entry_size = 4
            list_end = anim_list_abs + (num_anims * entry_size)
            if list_end > anim_list_ctx.file_size:
                logger.warning(
                    'Skipping animList read in %s: abs=0x%X count=%d exceeds %s file_size=0x%X',
                    context.file_name,
                    anim_list_abs,
                    num_anims,
                    anim_list_ctx.file_name,
                    anim_list_ctx.file_size,
                )
                return []

            refs: List[ObjectAnimationReference] = []
            anim_list_ctx.reader.seek(anim_list_abs)
            for index in range(num_anims):
                anim_id = anim_list_ctx.reader.i16()
                pad = anim_list_ctx.reader.i16()
                refs.append(
                    ObjectAnimationReference(
                        index=index,
                        anim_id=int(anim_id),
                        pad=int(pad),
                        raw_list_value=int(anim_list_raw),
                    )
                )
                logger.debug('Object animList[%d]: animID=%d pad=%d', index, anim_id, pad)

            return refs
        finally:
            cache.close()

    def parse_first_animation_id(self) -> int | None:
        refs = self.parse_animation_references()
        return refs[0].anim_id if refs else None
