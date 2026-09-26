from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import re
import shutil
import ssl
import tempfile
import threading
import urllib.error
import urllib.request
import uuid
import zipfile

import bpy


GITHUB_OWNER = "Raq1"
GITHUB_REPOSITORY = "cdcEditor"
RELEASES_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases"
LATEST_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPOSITORY}/releases/latest"
)
PREFERRED_ASSET_NAME = "cdcEditor.zip"
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024

_CURRENT_VERSION = (0, 0, 0)
_REGISTERED = False
_MAIN_THREAD_QUEUE: queue.Queue = queue.Queue()


class _UpdaterState:
    def __init__(self) -> None:
        self.checking = False
        self.installing = False
        self.update_available = False
        self.latest_version: tuple[int, ...] | None = None
        self.latest_tag = ""
        self.download_url = ""
        self.release_url = RELEASES_URL
        self.status = "Update status has not been checked."
        self.error = ""
        self.restart_required = False


STATE = _UpdaterState()


def _version_from_text(value: str) -> tuple[int, ...]:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)+)(?!\d)", value or "")
    if not match:
        raise ValueError(f"Release tag does not contain a version number: {value!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def _normalized_version(value: tuple[int, ...], length: int = 4) -> tuple[int, ...]:
    return tuple(value[:length]) + (0,) * max(0, length - len(value))


def _format_version(value: tuple[int, ...] | None) -> str:
    if not value:
        return "unknown"
    return ".".join(str(part) for part in value)


def _request_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "cdcEditor-Blender-Addon-Updater",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=20, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _select_download_url(release: dict) -> str:
    assets = release.get("assets") or []
    for asset in assets:
        if str(asset.get("name", "")).casefold() == PREFERRED_ASSET_NAME.casefold():
            url = asset.get("browser_download_url")
            if url:
                return str(url)

    # GitHub's generated source archive is a safe fallback because the repository
    # root is also the cdcEditor add-on root.
    fallback = release.get("zipball_url")
    if fallback:
        return str(fallback)
    raise RuntimeError("The release has no downloadable ZIP file.")


def _check_latest_release() -> dict:
    release = _request_json(LATEST_RELEASE_API)
    tag = str(release.get("tag_name") or "")
    latest_version = _version_from_text(tag)
    release_url = str(release.get("html_url") or RELEASES_URL)
    return {
        "latest_version": latest_version,
        "latest_tag": tag,
        "download_url": _select_download_url(release),
        "release_url": release_url,
    }


def _friendly_network_error(error: BaseException) -> str:
    if isinstance(error, urllib.error.HTTPError):
        if error.code == 403:
            return "GitHub rejected the request or the API rate limit was reached."
        if error.code == 404:
            return "No published GitHub release was found."
        return f"GitHub returned HTTP {error.code}."
    if isinstance(error, urllib.error.URLError):
        return f"Could not contact GitHub: {error.reason}"
    return str(error) or error.__class__.__name__


def _queue_on_main_thread(callback) -> None:
    # Worker threads never call Blender's API directly. A timer registered on
    # Blender's main thread drains this queue.
    _MAIN_THREAD_QUEUE.put(callback)


def _result_timer():
    if not _REGISTERED:
        return None
    for _ in range(20):
        try:
            callback = _MAIN_THREAD_QUEUE.get_nowait()
        except queue.Empty:
            break
        try:
            callback()
        except Exception as error:
            STATE.error = str(error) or error.__class__.__name__
            STATE.status = "The updater encountered an internal error."
    return 0.2


def _apply_check_result(result: dict | None, error: BaseException | None) -> None:
    STATE.checking = False
    if error is not None:
        STATE.error = _friendly_network_error(error)
        STATE.status = "Update check failed."
        return

    assert result is not None
    STATE.error = ""
    STATE.latest_version = result["latest_version"]
    STATE.latest_tag = result["latest_tag"]
    STATE.download_url = result["download_url"]
    STATE.release_url = result["release_url"]
    STATE.update_available = _normalized_version(STATE.latest_version) > _normalized_version(
        _CURRENT_VERSION
    )

    if STATE.update_available:
        STATE.status = f"Version {_format_version(STATE.latest_version)} is available."
    elif _normalized_version(STATE.latest_version) < _normalized_version(_CURRENT_VERSION):
        STATE.status = (
            f"Installed development version {_format_version(_CURRENT_VERSION)} is newer "
            f"than release {_format_version(STATE.latest_version)}."
        )
    else:
        STATE.status = f"cdcEditor {_format_version(_CURRENT_VERSION)} is up to date."

    _write_metadata(
        {
            "last_check": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "latest_tag": STATE.latest_tag,
        }
    )


def check_for_updates_async() -> bool:
    if STATE.checking or STATE.installing or STATE.restart_required:
        return False

    STATE.checking = True
    STATE.error = ""
    STATE.status = "Checking GitHub Releases..."

    def worker() -> None:
        result = None
        error = None
        try:
            result = _check_latest_release()
        except BaseException as exc:  # Preserve the error for Blender's main thread.
            error = exc
        _queue_on_main_thread(lambda: _apply_check_result(result, error))

    threading.Thread(target=worker, name="cdcEditor update check", daemon=True).start()
    return True


def _storage_directory() -> Path:
    try:
        path = bpy.utils.user_resource(
            "CONFIG", path="cdcEditor_updater", create=True
        )
        if path:
            return Path(path)
    except Exception:
        pass

    fallback = Path(tempfile.gettempdir()) / "cdcEditor_updater"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _metadata_path() -> Path:
    return _storage_directory() / "state.json"


def _read_metadata() -> dict:
    path = _metadata_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_metadata(updates: dict) -> None:
    try:
        data = _read_metadata()
        data.update(updates)
        path = _metadata_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "cdcEditor-Blender-Addon-Updater",
        },
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("The update archive is unexpectedly large.")

        written = 0
        with destination.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("The update archive exceeded the size limit.")
                output.write(block)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0o170000
    return unix_mode == 0o120000


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for info in zip_file.infolist():
            if _is_zip_symlink(info):
                raise RuntimeError("The update archive contains an unsupported symbolic link.")
            target = (destination / info.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise RuntimeError("The update archive contains an unsafe path.") from exc
        zip_file.extractall(destination)


def _looks_like_cdceditor_init(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:12000]
    except Exception:
        return False
    return "bl_info" in text and "cdcEditor" in text


def _locate_addon_root(extracted: Path) -> Path:
    candidates: list[Path] = []
    for init_file in extracted.rglob("__init__.py"):
        try:
            relative = init_file.relative_to(extracted)
        except ValueError:
            continue
        if len(relative.parts) > 4:
            continue
        if _looks_like_cdceditor_init(init_file):
            candidates.append(init_file.parent)

    if not candidates:
        raise RuntimeError("Could not find the cdcEditor add-on inside the update ZIP.")

    candidates.sort(
        key=lambda path: (
            path.name.casefold() != "cdceditor",
            len(path.relative_to(extracted).parts),
        )
    )
    return candidates[0]


def _create_backup(addon_root: Path, storage: Path) -> Path:
    storage.mkdir(parents=True, exist_ok=True)
    version_text = _format_version(_CURRENT_VERSION).replace(".", "_")
    backup = storage / f"cdcEditor_backup_{version_text}.zip"

    temporary = backup.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in addon_root.rglob("*"):
            if path.is_file():
                archive_name = Path("cdcEditor") / path.relative_to(addon_root)
                zip_file.write(path, archive_name.as_posix())
    temporary.replace(backup)
    return backup


def _replace_addon_directory(source_root: Path, addon_root: Path) -> None:
    parent = addon_root.parent
    token = uuid.uuid4().hex[:10]
    staged_root = parent / f".{addon_root.name}_new_{token}"
    old_root = parent / f".{addon_root.name}_old_{token}"

    shutil.rmtree(staged_root, ignore_errors=True)
    shutil.rmtree(old_root, ignore_errors=True)
    shutil.copytree(source_root, staged_root)

    try:
        addon_root.rename(old_root)
        staged_root.rename(addon_root)
    except Exception:
        if not addon_root.exists() and old_root.exists():
            old_root.rename(addon_root)
        shutil.rmtree(staged_root, ignore_errors=True)
        raise

    # The previous directory is no longer needed because a ZIP backup already exists.
    # Deletion can fail transiently on Windows; leaving a hidden sibling is harmless.
    shutil.rmtree(old_root, ignore_errors=True)


def _perform_install(download_url: str, addon_root: Path, storage: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="cdcEditor_update_") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        archive = temp_dir / "update.zip"
        extracted = temp_dir / "extracted"
        extracted.mkdir()

        _download_file(download_url, archive)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError("GitHub did not return a valid ZIP archive.")
        _safe_extract(archive, extracted)
        source_root = _locate_addon_root(extracted)
        backup = _create_backup(addon_root, storage)
        _replace_addon_directory(source_root, addon_root)
        return {"backup": str(backup)}


def _apply_install_result(result: dict | None, error: BaseException | None) -> None:
    STATE.installing = False
    if error is not None:
        STATE.error = _friendly_network_error(error)
        STATE.status = "Update installation failed. The installed add-on was not changed."
        return

    STATE.error = ""
    STATE.restart_required = True
    STATE.update_available = False
    STATE.status = "Update installed. Restart Blender to load the new version."
    if result:
        _write_metadata({"latest_backup": result.get("backup", "")})


def install_update_async() -> bool:
    if (
        STATE.installing
        or STATE.checking
        or STATE.restart_required
        or not STATE.download_url
    ):
        return False

    STATE.installing = True
    STATE.error = ""
    STATE.status = "Downloading and installing the update..."
    addon_root = Path(__file__).resolve().parent
    download_url = STATE.download_url
    storage = _storage_directory()

    def worker() -> None:
        result = None
        error = None
        try:
            result = _perform_install(download_url, addon_root, storage)
        except BaseException as exc:
            error = exc
        _queue_on_main_thread(lambda: _apply_install_result(result, error))

    threading.Thread(target=worker, name="cdcEditor update install", daemon=True).start()
    return True


def _restore_backup(backup: Path, addon_root: Path) -> None:
    if not backup.is_file() or not zipfile.is_zipfile(backup):
        raise RuntimeError("The updater backup is missing or invalid.")

    with tempfile.TemporaryDirectory(prefix="cdcEditor_restore_") as temp_dir_text:
        extracted = Path(temp_dir_text) / "extracted"
        extracted.mkdir()
        _safe_extract(backup, extracted)
        source_root = _locate_addon_root(extracted)
        _replace_addon_directory(source_root, addon_root)


def _apply_restore_result(error: BaseException | None) -> None:
    STATE.installing = False
    if error is not None:
        STATE.error = str(error) or error.__class__.__name__
        STATE.status = "Backup restoration failed."
        return
    STATE.error = ""
    STATE.restart_required = True
    STATE.status = "Backup restored. Restart Blender to load it."


def restore_backup_async() -> bool:
    if STATE.installing or STATE.checking or STATE.restart_required:
        return False
    backup_text = str(_read_metadata().get("latest_backup") or "")
    backup = Path(backup_text) if backup_text else Path()
    if not backup_text or not backup.is_file():
        STATE.error = "No updater backup is available."
        STATE.status = "Backup restoration failed."
        return False

    STATE.installing = True
    STATE.error = ""
    STATE.status = "Restoring the previous cdcEditor version..."
    addon_root = Path(__file__).resolve().parent

    def worker() -> None:
        error = None
        try:
            _restore_backup(backup, addon_root)
        except BaseException as exc:
            error = exc
        _queue_on_main_thread(lambda: _apply_restore_result(error))

    threading.Thread(target=worker, name="cdcEditor update restore", daemon=True).start()
    return True


class CDCEDITOR_OT_UpdaterCheck(bpy.types.Operator):
    bl_idname = "cdc_editor.updater_check"
    bl_label = "Check for Updates"
    bl_description = "Check GitHub Releases for a newer cdcEditor version"
    bl_options = {"INTERNAL"}

    def execute(self, _context):
        if not check_for_updates_async():
            self.report({"INFO"}, "An updater operation is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class CDCEDITOR_OT_UpdaterInstall(bpy.types.Operator):
    bl_idname = "cdc_editor.updater_install"
    bl_label = "Install cdcEditor Update"
    bl_description = "Back up the installed add-on and install the latest GitHub release"
    bl_options = {"INTERNAL"}

    def invoke(self, context, _event):
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, _context):
        layout = self.layout
        layout.label(text=f"Install cdcEditor {STATE.latest_tag}?", icon="IMPORT")
        layout.label(text="The current add-on will be backed up first.")
        layout.label(text="Blender must be restarted after installation.", icon="INFO")

    def execute(self, _context):
        if not install_update_async():
            self.report({"ERROR"}, "No downloadable update is ready")
            return {"CANCELLED"}
        return {"FINISHED"}


class CDCEDITOR_OT_UpdaterRestore(bpy.types.Operator):
    bl_idname = "cdc_editor.updater_restore"
    bl_label = "Restore Previous Version"
    bl_description = "Restore the most recent backup made by the cdcEditor updater"
    bl_options = {"INTERNAL"}

    def invoke(self, context, _event):
        return context.window_manager.invoke_confirm(self, _event)

    def execute(self, _context):
        if not restore_backup_async():
            self.report({"ERROR"}, STATE.error or "No backup is available")
            return {"CANCELLED"}
        return {"FINISHED"}


_CLASSES = (
    CDCEDITOR_OT_UpdaterCheck,
    CDCEDITOR_OT_UpdaterInstall,
    CDCEDITOR_OT_UpdaterRestore,
)


def _backup_available() -> bool:
    backup_text = str(_read_metadata().get("latest_backup") or "")
    return bool(backup_text and Path(backup_text).is_file())


def draw_preferences(layout, context) -> None:
    box = layout.box()
    box.label(text="Updates", icon="FILE_REFRESH")

    addon_entry = context.preferences.addons.get(__package__)
    preferences = addon_entry.preferences if addon_entry else None
    if preferences is not None:
        row = box.row(align=True)
        row.prop(preferences, "auto_check_update")
        interval = row.row(align=True)
        interval.enabled = preferences.auto_check_update
        interval.prop(preferences, "updater_interval_days", text="Every (days)")

    box.label(text=f"Installed version: {_format_version(_CURRENT_VERSION)}")

    if STATE.error:
        error_box = box.box()
        error_box.alert = True
        error_box.label(text=STATE.status, icon="ERROR")
        error_box.label(text=STATE.error)
    else:
        icon = "INFO"
        if STATE.update_available:
            icon = "IMPORT"
        elif STATE.restart_required:
            icon = "ERROR"
        box.label(text=STATE.status, icon=icon)

    row = box.row(align=True)
    row.enabled = (
        not STATE.checking and not STATE.installing and not STATE.restart_required
    )
    row.operator(
        CDCEDITOR_OT_UpdaterCheck.bl_idname,
        text="Checking..." if STATE.checking else "Check for Updates",
        icon="FILE_REFRESH",
    )
    releases = row.operator("wm.url_open", text="Release Page", icon="URL")
    releases.url = STATE.release_url or RELEASES_URL

    if STATE.update_available and not STATE.restart_required:
        install_row = box.row()
        install_row.scale_y = 1.4
        install_row.enabled = not STATE.installing
        install_row.operator(
            CDCEDITOR_OT_UpdaterInstall.bl_idname,
            text=(
                "Installing..."
                if STATE.installing
                else f"Install Update {STATE.latest_tag}"
            ),
            icon="IMPORT",
        )

    if _backup_available():
        restore_row = box.row()
        restore_row.enabled = (
            not STATE.checking and not STATE.installing and not STATE.restart_required
        )
        restore_row.operator(CDCEDITOR_OT_UpdaterRestore.bl_idname, icon="RECOVER_LAST")

    if STATE.restart_required:
        restart_box = box.box()
        restart_box.alert = True
        restart_box.label(text="Restart Blender to finish the update.", icon="ERROR")


def _auto_check_timer():
    if not _REGISTERED:
        return None

    try:
        addon_entry = bpy.context.preferences.addons.get(__package__)
        preferences = addon_entry.preferences if addon_entry else None
        if preferences is None or not preferences.auto_check_update:
            return 60.0

        metadata = _read_metadata()
        last_check_text = str(metadata.get("last_check") or "")
        should_check = True
        if last_check_text:
            from datetime import datetime, timedelta

            try:
                last_check = datetime.fromisoformat(last_check_text)
                should_check = datetime.now() - last_check >= timedelta(
                    days=max(1, int(preferences.updater_interval_days))
                )
            except ValueError:
                should_check = True

        if should_check:
            check_for_updates_async()
    except Exception:
        pass
    return 3600.0


def register(current_version: tuple[int, ...]) -> None:
    global _CURRENT_VERSION, _REGISTERED
    _CURRENT_VERSION = tuple(int(part) for part in current_version)
    _REGISTERED = True
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    if not bpy.app.timers.is_registered(_result_timer):
        bpy.app.timers.register(_result_timer, first_interval=0.2)
    if not bpy.app.timers.is_registered(_auto_check_timer):
        bpy.app.timers.register(_auto_check_timer, first_interval=2.0)


def unregister() -> None:
    global _REGISTERED
    _REGISTERED = False
    if bpy.app.timers.is_registered(_auto_check_timer):
        bpy.app.timers.unregister(_auto_check_timer)
    if bpy.app.timers.is_registered(_result_timer):
        bpy.app.timers.unregister(_result_timer)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
