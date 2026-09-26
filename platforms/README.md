# Platform modules

This package contains platform-specific importer code that used to be spread across
`builders/` and `formats/`.

- `common/` contains shared platform dispatch and utilities.
- `pc/` contains PC texture and mesh helpers.
- `nintendo/` contains GameCube/Wii mesh, parser, and texture helpers.
- `ps2/` contains PS2 mesh, parser, and texture helpers.
- `psp/` contains PSP mesh/material, parser, and texture helpers.
- `xbox/` contains Xbox mesh, parser, and texture helpers.

Compatibility wrappers remain in `builders/` and `formats/` so older imports keep
working while new code can import from `trlau_editor.platforms` directly.
