#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1111 Inpaint Batch Queue

AUTOMATIC1111 Stable Diffusion WebUI API 用の、inpaintジョブをローカルで準備して
まとめて実行するためのGUIツール。

Required libraries:
    PySide6, numpy, opencv-python, Pillow
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

APP_TITLE = "A1111 Inpaint Batch Queue"
APP_REV = "v36"
SETTINGS_NAME = "a1111-inpaint-batch-queue-settings.json"
PROJECT_FILE_NAME = "project.json"
PROJECT_SETTINGS_NAME = "settings.json"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MASK_FILE_NAME = "mask.png"
JOB_FILE_NAME = "job.json"
RESULT_DIR_NAME = "result"
DEBUG_DIR_NAME = "debug"
REQUEST_FILE_NAME = "request.json"
MAX_API_SIZE_OPTIONS = {
    "unlimited": None,
    "1920x1080": (1920, 1080),
    "1280x720": (1280, 720),
}

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    print("numpy is required.", exc)
    raise

try:
    import cv2
except Exception as exc:  # pragma: no cover
    print("opencv-python is required.", exc)
    raise

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    print("Pillow is required.", exc)
    raise

try:
    from PySide6.QtCore import QByteArray, QFile, QMimeData, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl, Signal
    from PySide6.QtGui import QAction, QActionGroup, QColor, QDragEnterEvent, QDropEvent, QImage, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QShortcut, QWheelEvent, QTextCursor
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QDoubleSpinBox,
        QSplitter,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QSizePolicy,
    )
except Exception as exc:  # pragma: no cover
    print("PySide6 is required to run this GUI tool.")
    print("Install example: pip install PySide6 numpy opencv-python Pillow")
    raise


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def settings_path() -> Path:
    return app_dir() / SETTINGS_NAME


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def safe_exception_text(exc: BaseException) -> str:
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:
        return type(exc).__name__


def read_json_utf8(path: Path, default=None):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"JSON read error: {safe_exception_text(exc)}", file=sys.stderr)
    return default


def write_json_utf8(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cv2_read_rgba_unicode(path: Path) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像として読み込めません: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


def cv2_read_mask_unicode(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"マスクとして読み込めません: {path}")
    if img.ndim == 3:
        if img.shape[2] == 4:
            gray = img[:, :, 3]
        else:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = np.ascontiguousarray(gray.astype(np.uint8))
    if size is not None:
        w, h = size
        if gray.shape[1] != w or gray.shape[0] != h:
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_NEAREST)
    return gray


def cv2_write_png_unicode(path: Path, rgba_or_gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rgba_or_gray
    if arr.ndim == 2:
        enc_src = arr.astype(np.uint8)
    elif arr.shape[2] == 4:
        enc_src = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    else:
        enc_src = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", enc_src)
    if not ok:
        raise ValueError(f"PNGエンコードに失敗しました: {path}")
    path.write_bytes(buf.tobytes())


def ndarray_rgba_to_qimage(rgba: np.ndarray) -> QImage:
    if rgba is None:
        return QImage()
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGBA)
    if rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_RGB2RGBA)
    if not rgba.flags["C_CONTIGUOUS"]:
        rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    fmt = getattr(QImage, "Format_RGBA8888", None)
    if fmt is None:
        fmt = QImage.Format.Format_RGBA8888
    return QImage(rgba.data, w, h, rgba.strides[0], fmt).copy()


def ndarray_rgba_to_pixmap(rgba: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(ndarray_rgba_to_qimage(rgba))


def mask_to_rgba(mask: np.ndarray) -> np.ndarray:
    gray = mask.astype(np.uint8)
    rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = gray
    rgba[:, :, 1] = gray
    rgba[:, :, 2] = gray
    rgba[:, :, 3] = 255
    return rgba


def overlay_rgba(base_crop: np.ndarray, mask_crop: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = base_crop.copy()
    if out.ndim != 3 or out.shape[2] != 4:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2RGBA)
    if mask_crop.size == 0:
        return out
    m = mask_crop.astype(np.float32) / 255.0
    strength = m * float(alpha)
    # 白マスク部分を黄色寄りにして、元画像の明度を残す。
    color = np.array([255.0, 215.0, 40.0], dtype=np.float32)
    rgb = out[:, :, :3].astype(np.float32)
    rgb = rgb * (1.0 - strength[:, :, None]) + color.reshape(1, 1, 3) * strength[:, :, None]
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return out


def pil_image_to_png_base64(path: Path) -> str:
    # API送信用。Pillowで読み直してPNGへ統一する。日本語パスはPath.read_bytes経由。
    from io import BytesIO
    img = Image.open(path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bio = BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")


def numpy_mask_to_png_base64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8))
    if not ok:
        raise ValueError("マスクPNGエンコードに失敗しました。")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def numpy_rgba_to_png_base64(rgba: np.ndarray) -> str:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("RGBA画像ではありません。")
    bgra = cv2.cvtColor(rgba.astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError("画像PNGエンコードに失敗しました。")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def rgba_to_opaque_rgb(rgba: np.ndarray, background: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Flatten RGBA onto a solid background for Stable Diffusion API input.

    AUTOMATIC1111 accepts PNG input, but alpha-bearing PNGs can be interpreted
    inconsistently by model/preprocess paths.  The inpaint mask is sent
    separately, so the init image should be an ordinary opaque RGB image.
    """
    if rgba.ndim != 3:
        raise ValueError("画像配列ではありません。")
    arr = rgba.astype(np.uint8)
    if arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, :, :3])
    if arr.shape[2] != 4:
        raise ValueError("RGB/RGBA画像ではありません。")
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.array(background, dtype=np.float32).reshape(1, 1, 3)
    rgb = arr[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


def numpy_rgba_to_api_png_base64(rgba: np.ndarray) -> str:
    rgb = rgba_to_opaque_rgb(rgba)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("API送信画像PNGエンコードに失敗しました。")
    return base64.b64encode(buf.tobytes()).decode("ascii")


MIN_CROP_SIZE = 8
API_DIM_MULTIPLE = 8
MIN_CROP_API_SIDE = 512


def ceil_to_multiple(value: int, multiple: int = API_DIM_MULTIPLE) -> int:
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    return ((value + multiple - 1) // multiple) * multiple


def floor_to_multiple(value: int, multiple: int = API_DIM_MULTIPLE) -> int:
    value = max(1, int(value))
    multiple = max(1, int(multiple))
    return max(multiple, (value // multiple) * multiple)


def normalize_crop_rect_data(rect: object, image_size: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    if rect is None:
        return None
    x = y = w = h = None
    if isinstance(rect, dict):
        try:
            x = int(rect.get("x", 0)); y = int(rect.get("y", 0)); w = int(rect.get("w", 0)); h = int(rect.get("h", 0))
        except Exception:
            return None
    elif isinstance(rect, (list, tuple)) and len(rect) >= 4:
        try:
            x = int(rect[0]); y = int(rect[1]); w = int(rect[2]); h = int(rect[3])
        except Exception:
            return None
    if x is None or y is None or w is None or h is None:
        return None
    if image_size is not None:
        max_w, max_h = int(image_size[0]), int(image_size[1])
        x = max(0, min(x, max_w))
        y = max(0, min(y, max_h))
        w = max(0, min(w, max_w - x))
        h = max(0, min(h, max_h - y))
    if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
        return None
    return int(x), int(y), int(w), int(h)


def crop_rect_from_drag_points_keep_anchor(
    start_x: float,
    start_y: float,
    current_x: float,
    current_y: float,
    image_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """Build a manual crop rectangle from drag points.

    The mouse-down point stays inside the selected crop.  Small selections are
    expanded immediately to an API-safe minimum size.  When the preferred drag
    direction hits an image edge, the rectangle expands back into the image
    instead of collapsing into a thin strip.
    """
    img_w, img_h = int(image_size[0]), int(image_size[1])
    if img_w <= 0 or img_h <= 0:
        return None

    def build_axis(start: float, current: float, limit: int) -> Tuple[int, int]:
        s = max(0.0, min(float(start), float(limit)))
        c = max(0.0, min(float(current), float(limit)))
        raw_len = max(MIN_CROP_SIZE, int(math.ceil(abs(c - s))))
        target = min(limit, max(ceil_to_multiple(raw_len), min(limit, MIN_CROP_API_SIDE)))

        if c < s:
            # Prefer expanding toward the drag direction, but if the image edge
            # prevents the minimum size, keep the selection valid by growing
            # back into the image.
            high = int(math.ceil(s))
            low = high - target
            if low < 0:
                low = 0
                high = min(limit, target)
        else:
            low = int(math.floor(s))
            high = low + target
            if high > limit:
                high = limit
                low = max(0, high - target)

        low = max(0, min(int(low), limit))
        high = max(low, min(int(high), limit))
        return low, max(0, high - low)

    x, target_w = build_axis(start_x, current_x, img_w)
    y, target_h = build_axis(start_y, current_y, img_h)
    return normalize_crop_rect_data((x, y, target_w, target_h), (img_w, img_h))


def pad_pair_to_multiple(rgba: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    h, w = rgba.shape[:2]
    target_w = ceil_to_multiple(w)
    target_h = ceil_to_multiple(h)
    pad_right = max(0, target_w - w)
    pad_bottom = max(0, target_h - h)
    if pad_right == 0 and pad_bottom == 0:
        return np.ascontiguousarray(rgba), np.ascontiguousarray(mask), (w, h)
    rgba_pad = cv2.copyMakeBorder(
        rgba.astype(np.uint8), 0, pad_bottom, 0, pad_right,
        cv2.BORDER_REPLICATE
    )
    mask_pad = cv2.copyMakeBorder(
        mask.astype(np.uint8), 0, pad_bottom, 0, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )
    return np.ascontiguousarray(rgba_pad), np.ascontiguousarray(mask_pad), (target_w, target_h)


def resized_for_api(rgba: np.ndarray, mask: np.ndarray, max_size_key: str, force_multiple: bool = False) -> Tuple[np.ndarray, np.ndarray, bool]:
    limit = MAX_API_SIZE_OPTIONS.get(max_size_key)
    if limit is None:
        return rgba, mask, False
    max_w, max_h = limit
    h, w = rgba.shape[:2]
    if w <= 0 or h <= 0:
        return rgba, mask, False
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999999:
        return rgba, mask, False
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if force_multiple:
        new_w = min(max_w, floor_to_multiple(new_w))
        new_h = min(max_h, floor_to_multiple(new_h))
    rgba_out = cv2.resize(rgba.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    mask_out = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(rgba_out), np.ascontiguousarray(mask_out), True


def resized_crop_for_api(rgba: np.ndarray, mask: np.ndarray, max_size_key: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Downscale manual-crop inputs only when the API max-size setting requires it.

    Do not upscale tiny crops here. Small crop areas are handled by expanding the
    visible/saved crop rectangle at selection time.
    """
    return resized_for_api(rgba, mask, max_size_key, force_multiple=True)


@dataclass(frozen=True)
class ApiImagePrep:
    base_api: np.ndarray
    mask_api: np.ndarray
    paste_rect: Optional[Tuple[int, int, int, int]]
    paste_canvas_size: Optional[Tuple[int, int]]


def prepare_image_pair_for_api(rgba: np.ndarray, mask: np.ndarray, crop_enabled: bool, crop_rect: object, max_size_key: str) -> ApiImagePrep:
    image_size = (int(rgba.shape[1]), int(rgba.shape[0]))
    rect = normalize_crop_rect_data(crop_rect if crop_enabled else None, image_size)
    if rect is None:
        base_api, mask_api, _resized = resized_for_api(rgba, mask, max_size_key, force_multiple=False)
        return ApiImagePrep(base_api, mask_api, None, None)
    api_rect = rect
    x, y, w, h = api_rect
    base_crop = np.ascontiguousarray(rgba[y:y+h, x:x+w])
    mask_crop = np.ascontiguousarray(mask[y:y+h, x:x+w])
    base_pad, mask_pad, padded_size = pad_pair_to_multiple(base_crop, mask_crop)
    base_api, mask_api, _resized = resized_crop_for_api(base_pad, mask_pad, max_size_key)
    return ApiImagePrep(base_api, mask_api, api_rect, padded_size)


def composite_result_on_base(
    base_rgba: np.ndarray,
    result_rgba: np.ndarray,
    crop_rect: Optional[Tuple[int, int, int, int]],
    paste_canvas_size: Optional[Tuple[int, int]] = None,
    paste_mask: Optional[np.ndarray] = None,
    mask_blur: int = 0,
) -> np.ndarray:
    if crop_rect is None:
        return np.ascontiguousarray(result_rgba)
    x, y, w, h = crop_rect
    out = np.ascontiguousarray(base_rgba.copy())
    patch = result_rgba
    target_w, target_h = paste_canvas_size or (w, h)
    if patch.shape[:2] != (target_h, target_w):
        patch = cv2.resize(patch.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    patch = np.ascontiguousarray(patch[:h, :w, :4].astype(np.uint8))

    if paste_mask is not None:
        mask_crop = np.ascontiguousarray(paste_mask[y:y+h, x:x+w].astype(np.uint8))
        if mask_crop.shape[:2] != (h, w):
            mask_crop = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        if int(mask_crop.max()) <= 0:
            return out
        blur = max(0, int(mask_blur))
        if blur > 0:
            kernel = blur * 2 + 1
            mask_crop = cv2.GaussianBlur(mask_crop, (kernel, kernel), 0)
        alpha = (mask_crop.astype(np.float32) / 255.0)[:, :, None]
        base_crop = out[y:y+h, x:x+w, :4].astype(np.float32)
        patch_f = patch.astype(np.float32)
        mixed = patch_f * alpha + base_crop * (1.0 - alpha)
        out[y:y+h, x:x+w, :4] = np.clip(mixed, 0, 255).astype(np.uint8)
    else:
        out[y:y+h, x:x+w] = patch
    return out


def prompt_text_for_api(text: str) -> str:
    """Remove prompt comment lines before sending to the API.

    Lines whose first character is '#' are user comments and are kept in the
    job file/UI, but are not sent to AUTOMATIC1111.
    """
    lines = []
    for line in str(text or "").splitlines():
        if line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def unique_name(base: str, existing: Iterable[str]) -> str:
    base = str(base or "").strip() or "新規プリセット"
    existing_set = set(str(x) for x in existing)
    if base not in existing_set:
        return base
    index = 2
    while f"{base} {index}" in existing_set:
        index += 1
    return f"{base} {index}"


def sanitize_param_presets(data: object) -> Dict[str, Dict[str, object]]:
    cleaned: Dict[str, Dict[str, object]] = {}
    if isinstance(data, dict):
        for raw_name, raw_values in data.items():
            name = str(raw_name).strip()
            if not name or not isinstance(raw_values, dict):
                continue
            params = InpaintParams.from_dict(raw_values)
            cleaned[name] = asdict(params)
    return cleaned


def open_path_in_explorer(path: Path, select_file: bool = False) -> None:
    path = Path(path)
    if sys.platform.startswith("win"):
        try:
            if select_file and path.is_file():
                subprocess.Popen(["explorer", f"/select,{str(path)}"])
            else:
                target = path if path.is_dir() else path.parent
                os.startfile(str(target))  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    try:
        target = path.parent if select_file and path.is_file() else path
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception:
        pass


def move_path_to_trash(path: Path) -> bool:
    """Move a file to the OS trash/recycle bin without adding extra dependencies."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        result = QFile.moveToTrash(str(path))
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            FO_DELETE = 0x0003
            FOF_ALLOWUNDO = 0x0040
            FOF_NOCONFIRMATION = 0x0010
            FOF_NOERRORUI = 0x0400
            FOF_SILENT = 0x0004

            class SHFILEOPSTRUCTW(ctypes.Structure):
                _fields_ = [
                    ("hwnd", wintypes.HWND),
                    ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR),
                    ("fFlags", wintypes.USHORT),
                    ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", wintypes.LPVOID),
                    ("lpszProgressTitle", wintypes.LPCWSTR),
                ]

            op = SHFILEOPSTRUCTW()
            op.wFunc = FO_DELETE
            op.pFrom = str(path) + "\0\0"
            op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT
            ret = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
            return ret == 0 and not bool(op.fAnyOperationsAborted) and not path.exists()
        except Exception:
            return False
    return False


def decode_api_image_to_rgba(text_b64: str) -> np.ndarray:
    if "," in text_b64 and text_b64.strip().startswith("data:"):
        text_b64 = text_b64.split(",", 1)[1]
    raw = base64.b64decode(text_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("APIレスポンス画像のデコードに失敗しました。")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)


def file_urls_to_paths(mime_data) -> List[Path]:
    paths: List[Path] = []
    if not mime_data.hasUrls():
        return paths
    for url in mime_data.urls():
        if url.isLocalFile():
            p = Path(url.toLocalFile())
            paths.append(p)
    return paths


@dataclass
class InpaintParams:
    prompt: str = "best quality, natural hand, correct fingers, five fingers, relaxed natural hand, natural wrist angle, same characters, same pose, same composition, same soft anime style"
    negative_prompt: str = "bad hand, extra fingers, missing fingers, fused fingers, malformed hand, deformed fingers, bad anatomy, extra limb, broken wrist, ugly hand"
    sampler_name: str = "Euler a"
    steps: int = 28
    cfg_scale: float = 5.0
    denoising_strength: float = 0.55
    mask_blur: int = 8
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 96
    inpainting_fill: int = 1
    inpainting_mask_invert: int = 0
    batch_size: int = 1
    n_iter: int = 4
    width: int = 0
    height: int = 0
    seed: int = -1
    restore_faces: bool = False

    @classmethod
    def from_dict(cls, data: object) -> "InpaintParams":
        base = cls()
        if not isinstance(data, dict):
            return base
        for key, value in data.items():
            if not hasattr(base, key):
                continue
            try:
                current = getattr(base, key)
                if isinstance(current, bool):
                    setattr(base, key, bool(value))
                elif isinstance(current, int):
                    setattr(base, key, int(value))
                elif isinstance(current, float):
                    setattr(base, key, float(value))
                else:
                    setattr(base, key, str(value))
            except Exception:
                pass
        base.clamp()
        return base

    def clamp(self) -> None:
        self.steps = clamp_int(self.steps, 1, 150)
        self.cfg_scale = clamp_float(self.cfg_scale, 0.0, 30.0)
        self.denoising_strength = clamp_float(self.denoising_strength, 0.0, 1.0)
        self.mask_blur = clamp_int(self.mask_blur, 0, 128)
        self.inpaint_full_res_padding = clamp_int(self.inpaint_full_res_padding, 0, 512)
        self.inpainting_fill = clamp_int(self.inpainting_fill, 0, 3)
        self.inpainting_mask_invert = clamp_int(self.inpainting_mask_invert, 0, 1)
        self.batch_size = clamp_int(self.batch_size, 1, 16)
        self.n_iter = clamp_int(self.n_iter, 1, 100)
        self.width = max(0, int(self.width))
        self.height = max(0, int(self.height))
        self.seed = int(self.seed)


@dataclass
class ApiSettings:
    base_url: str = "http://localhost:7860"
    username: str = ""
    password: str = ""
    save_password: bool = False
    timeout: int = 1800
    verify_ssl: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "ApiSettings":
        base = cls()
        if not isinstance(data, dict):
            return base
        base.base_url = str(data.get("base_url", base.base_url)).strip() or base.base_url
        base.username = str(data.get("username", ""))
        base.save_password = bool(data.get("save_password", False))
        base.password = str(data.get("password", "")) if base.save_password else ""
        try:
            base.timeout = clamp_int(int(data.get("timeout", base.timeout)), 5, 86400)
        except Exception:
            pass
        base.verify_ssl = bool(data.get("verify_ssl", True))
        return base

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_url": self.base_url,
            "username": self.username,
            "password": self.password if self.save_password else "",
            "save_password": bool(self.save_password),
            "timeout": int(self.timeout),
            "verify_ssl": bool(self.verify_ssl),
        }


@dataclass
class JobData:
    job_id: str
    name: str
    input_path: str
    mask_path: str = MASK_FILE_NAME
    status: str = "未実行"
    checked: bool = True
    params: InpaintParams = field(default_factory=InpaintParams)
    latest_result: str = ""
    preset_name: str = ""
    crop_enabled: bool = False
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: object) -> "JobData":
        if not isinstance(data, dict):
            raise ValueError("job.jsonが不正です。")
        crop_enabled = bool(data.get("crop_enabled", False))
        crop_rect = normalize_crop_rect_data(data.get("crop_rect"))
        if crop_rect is None:
            crop_enabled = False
        job = cls(
            job_id=str(data.get("job_id", "job_0000")),
            name=str(data.get("name", "Job")),
            input_path=str(data.get("input_path", "input.png")),
            mask_path=str(data.get("mask_path", MASK_FILE_NAME)),
            status=str(data.get("status", "未実行")),
            checked=bool(data.get("checked", True)),
            params=InpaintParams.from_dict(data.get("params")),
            latest_result=str(data.get("latest_result", "")),
            preset_name=str(data.get("preset_name", "")),
            crop_enabled=crop_enabled,
            crop_rect=crop_rect,
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
        )
        return job

    def to_dict(self) -> Dict[str, object]:
        rect = self.crop_rect if self.crop_enabled else None
        rect_dict = None if rect is None else {"x": int(rect[0]), "y": int(rect[1]), "w": int(rect[2]), "h": int(rect[3])}
        return {
            "job_id": self.job_id,
            "name": self.name,
            "input_path": self.input_path,
            "mask_path": self.mask_path,
            "status": self.status,
            "checked": self.checked,
            "params": asdict(self.params),
            "latest_result": self.latest_result,
            "preset_name": str(self.preset_name or ""),
            "crop_enabled": bool(self.crop_enabled and rect_dict is not None),
            "crop_rect": rect_dict,
            "created_at": self.created_at,
            "updated_at": time.time(),
        }


class MaskCanvas(QWidget):
    maskChanged = Signal()
    cropChanged = Signal()
    toolChanged = Signal(str)
    zoomChanged = Signal(float)
    fileDropped = Signal(list)

    MIN_ZOOM_SCALE = 0.05
    MAX_ZOOM_SCALE = 40.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 440)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.base_rgba: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.result_rgba: Optional[np.ndarray] = None
        self.mode = "overlay"
        self.tool = "brush"
        self.brush_size = 48
        self.zoom_scale = 1.0
        self.view_center: Optional[QPointF] = None
        self.panning = False
        self.drawing = False
        self.space_down = False
        self.pan_last_pos: Optional[QPointF] = None
        self.last_img_pos: Optional[QPointF] = None
        self.cursor_img_pos: Optional[QPointF] = None
        self._qimage_cache_key = None
        self._qimage_cache: Optional[QImage] = None
        self._pixmap_cache: Optional[QPixmap] = None
        self._mask_stamp_cache: Dict[Tuple[int, bool], np.ndarray] = {}
        self._mask_version = 0
        self._live_draw_points: List[QPointF] = []
        self._live_draw_tool = "brush"
        self._live_draw_brush_size = self.brush_size
        self.crop_enabled = False
        self.crop_rect: Optional[Tuple[int, int, int, int]] = None
        self.crop_drawing = False
        self.crop_start_img: Optional[QPointF] = None
        self.crop_current_img: Optional[QPointF] = None

    # ---------- data ----------
    def set_images(self, base_rgba: Optional[np.ndarray], mask: Optional[np.ndarray], result_rgba: Optional[np.ndarray] = None, fit: bool = True) -> None:
        self.base_rgba = np.ascontiguousarray(base_rgba) if base_rgba is not None else None
        if self.base_rgba is not None:
            h, w = self.base_rgba.shape[:2]
            if mask is None:
                self.mask = np.zeros((h, w), dtype=np.uint8)
            else:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                self.mask = np.ascontiguousarray(mask.astype(np.uint8))
            if result_rgba is not None and result_rgba.shape[:2] != (h, w):
                result_rgba = cv2.resize(result_rgba, (w, h), interpolation=cv2.INTER_LINEAR)
            self.result_rgba = np.ascontiguousarray(result_rgba) if result_rgba is not None else None
            self.crop_rect = normalize_crop_rect_data(self.crop_rect, (w, h))
            self.crop_enabled = self.crop_rect is not None and bool(self.crop_enabled)
            self._mask_version += 1
            if fit or self.view_center is None:
                QTimer.singleShot(0, self.fit_to_window)
            else:
                self.clamp_view_center()
                self.update()
        else:
            self.mask = None
            self.result_rgba = None
            self._mask_version += 1
            self.view_center = None
            self.crop_enabled = False
            self.crop_rect = None
            self.crop_drawing = False
            self.crop_start_img = None
            self.crop_current_img = None
            self.update()

    def set_result(self, result_rgba: Optional[np.ndarray]) -> None:
        if self.base_rgba is None:
            self.result_rgba = None
        else:
            h, w = self.base_rgba.shape[:2]
            if result_rgba is not None and result_rgba.shape[:2] != (h, w):
                result_rgba = cv2.resize(result_rgba, (w, h), interpolation=cv2.INTER_LINEAR)
            self.result_rgba = np.ascontiguousarray(result_rgba) if result_rgba is not None else None
        self.invalidate_cache()
        self.update()

    def image_shape(self) -> Optional[Tuple[int, int]]:
        if self.base_rgba is None:
            return None
        return self.base_rgba.shape[:2]

    def invalidate_cache(self) -> None:
        self._qimage_cache_key = None
        self._qimage_cache = None
        self._pixmap_cache = None

    def get_crop_rect(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.crop_enabled or self.base_rgba is None:
            return None
        return normalize_crop_rect_data(self.crop_rect, (int(self.base_rgba.shape[1]), int(self.base_rgba.shape[0])))

    def set_crop_rect(self, rect: object, emit_signal: bool = True) -> None:
        size = (int(self.base_rgba.shape[1]), int(self.base_rgba.shape[0])) if self.base_rgba is not None else None
        norm = normalize_crop_rect_data(rect, size)
        self.crop_rect = norm
        self.crop_enabled = norm is not None
        self.crop_drawing = False
        self.crop_start_img = None
        self.crop_current_img = None
        self.update()
        if emit_signal:
            self.cropChanged.emit()

    def clear_crop_rect(self, emit_signal: bool = True) -> None:
        changed = self.crop_enabled or self.crop_rect is not None or self.crop_drawing
        self.crop_enabled = False
        self.crop_rect = None
        self.crop_drawing = False
        self.crop_start_img = None
        self.crop_current_img = None
        self.update()
        if changed and emit_signal:
            self.cropChanged.emit()

    def _crop_rect_from_points(self, p0: Optional[QPointF], p1: Optional[QPointF]) -> Optional[Tuple[int, int, int, int]]:
        if self.base_rgba is None or p0 is None or p1 is None:
            return None
        h, w = self.base_rgba.shape[:2]
        return crop_rect_from_drag_points_keep_anchor(p0.x(), p0.y(), p1.x(), p1.y(), (w, h))

    def begin_crop(self, img_pos: QPointF) -> None:
        self.crop_drawing = True
        self.crop_start_img = QPointF(img_pos.x(), img_pos.y())
        self.crop_current_img = QPointF(img_pos.x(), img_pos.y())
        self.update()

    def continue_crop(self, img_pos: QPointF) -> None:
        if not self.crop_drawing:
            return
        self.crop_current_img = QPointF(img_pos.x(), img_pos.y())
        self.update()

    def end_crop(self) -> None:
        if not self.crop_drawing:
            return
        rect = self._crop_rect_from_points(self.crop_start_img, self.crop_current_img)
        self.crop_drawing = False
        self.crop_start_img = None
        self.crop_current_img = None
        self.set_crop_rect(rect, emit_signal=True)

    # ---------- view math ----------
    def get_fit_scale(self) -> float:
        shape = self.image_shape()
        if shape is None:
            return 1.0
        h, w = shape
        area_w = max(1, self.width() - 12)
        area_h = max(1, self.height() - 12)
        return max(0.0001, min(area_w / max(1, w), area_h / max(1, h), 1.0))

    def get_current_scale(self) -> float:
        return max(0.0001, self.get_fit_scale() * self.zoom_scale)

    def ensure_view_center(self) -> None:
        shape = self.image_shape()
        if shape is None:
            self.view_center = None
            return
        h, w = shape
        if self.view_center is None:
            self.view_center = QPointF(w / 2.0, h / 2.0)
        self.clamp_view_center()

    def clamp_view_center(self) -> None:
        shape = self.image_shape()
        if shape is None or self.view_center is None:
            return
        h, w = shape
        scale = self.get_current_scale()
        view_w = self.width() / max(scale, 1e-6)
        view_h = self.height() / max(scale, 1e-6)
        if view_w >= w:
            cx = w / 2.0
        else:
            half = view_w / 2.0
            cx = min(max(self.view_center.x(), half), w - half)
        if view_h >= h:
            cy = h / 2.0
        else:
            half = view_h / 2.0
            cy = min(max(self.view_center.y(), half), h - half)
        self.view_center = QPointF(cx, cy)

    def fit_to_window(self) -> None:
        if self.base_rgba is None:
            return
        self.zoom_scale = 1.0
        self.view_center = None
        self.ensure_view_center()
        self.invalidate_cache()
        self.zoomChanged.emit(self.get_current_scale())
        self.update()

    def set_actual_size(self) -> None:
        if self.base_rgba is None:
            return
        pos = QPointF(self.width() / 2.0, self.height() / 2.0)
        before = self.view_to_image(pos)
        fit = self.get_fit_scale()
        self.zoom_scale = clamp_float(1.0 / max(fit, 1e-6), self.MIN_ZOOM_SCALE, self.MAX_ZOOM_SCALE)
        if before is not None:
            scale = self.get_current_scale()
            self.view_center = QPointF(
                before.x() - (pos.x() - self.width() / 2.0) / max(scale, 1e-6),
                before.y() - (pos.y() - self.height() / 2.0) / max(scale, 1e-6),
            )
        self.clamp_view_center()
        self.invalidate_cache()
        self.zoomChanged.emit(self.get_current_scale())
        self.update()

    def set_zoom(self, new_zoom_scale: float, anchor_pos: Optional[QPointF] = None) -> None:
        if self.base_rgba is None:
            return
        self.ensure_view_center()
        if self.view_center is None:
            return
        old_scale = self.get_current_scale()
        anchor_img = None
        if anchor_pos is not None:
            anchor_img = QPointF(
                self.view_center.x() + (anchor_pos.x() - self.width() / 2.0) / max(old_scale, 1e-6),
                self.view_center.y() + (anchor_pos.y() - self.height() / 2.0) / max(old_scale, 1e-6),
            )
        self.zoom_scale = clamp_float(new_zoom_scale, self.MIN_ZOOM_SCALE, self.MAX_ZOOM_SCALE)
        new_scale = self.get_current_scale()
        if anchor_img is not None and anchor_pos is not None:
            self.view_center = QPointF(
                anchor_img.x() - (anchor_pos.x() - self.width() / 2.0) / max(new_scale, 1e-6),
                anchor_img.y() - (anchor_pos.y() - self.height() / 2.0) / max(new_scale, 1e-6),
            )
        self.clamp_view_center()
        self.invalidate_cache()
        self.zoomChanged.emit(self.get_current_scale())
        self.update()

    def image_to_view(self, p: QPointF) -> Optional[QPointF]:
        if self.base_rgba is None:
            return None
        self.ensure_view_center()
        if self.view_center is None:
            return None
        scale = self.get_current_scale()
        return QPointF(self.width() / 2.0 + (p.x() - self.view_center.x()) * scale, self.height() / 2.0 + (p.y() - self.view_center.y()) * scale)

    def view_to_image(self, p: QPointF) -> Optional[QPointF]:
        if self.base_rgba is None:
            return None
        self.ensure_view_center()
        if self.view_center is None:
            return None
        h, w = self.base_rgba.shape[:2]
        scale = self.get_current_scale()
        x = self.view_center.x() + (p.x() - self.width() / 2.0) / max(scale, 1e-6)
        y = self.view_center.y() + (p.y() - self.height() / 2.0) / max(scale, 1e-6)
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return QPointF(x, y)

    # ---------- render ----------
    def current_source_image(self) -> Optional[np.ndarray]:
        if self.base_rgba is None:
            return None
        if self.mode == "result" and self.result_rgba is not None:
            return self.result_rgba
        if self.mode == "mask" and self.mask is not None:
            return mask_to_rgba(self.mask)
        return self.base_rgba

    def build_display_crop(self, x0: int, y0: int, x1: int, y1: int, target_w: int, target_h: int) -> np.ndarray:
        if self.base_rgba is None:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.base_rgba.shape[1], x1)
        y1 = min(self.base_rgba.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        if self.mode == "mask" and self.mask is not None:
            crop = mask_to_rgba(self.mask[y0:y1, x0:x1])
        elif self.mode == "result" and self.result_rgba is not None:
            crop = self.result_rgba[y0:y1, x0:x1]
        elif self.mode == "overlay" and self.mask is not None:
            crop = overlay_rgba(self.base_rgba[y0:y1, x0:x1], self.mask[y0:y1, x0:x1])
        else:
            crop = self.base_rgba[y0:y1, x0:x1]
        target_w = max(1, int(round(target_w)))
        target_h = max(1, int(round(target_h)))
        if crop.shape[1] != target_w or crop.shape[0] != target_h:
            interp = cv2.INTER_NEAREST if self.mode == "mask" else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (target_w, target_h), interpolation=interp)
        return np.ascontiguousarray(crop)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 32, 34))
        if self.base_rgba is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "画像をD&Dしてジョブ追加")
            return
        self.ensure_view_center()
        if self.view_center is None:
            return
        scale = self.get_current_scale()
        h, w = self.base_rgba.shape[:2]
        view_w_img = self.width() / max(scale, 1e-6)
        view_h_img = self.height() / max(scale, 1e-6)
        x0_f = self.view_center.x() - view_w_img / 2.0
        y0_f = self.view_center.y() - view_h_img / 2.0
        x1_f = self.view_center.x() + view_w_img / 2.0
        y1_f = self.view_center.y() + view_h_img / 2.0
        x0 = max(0, int(math.floor(x0_f)))
        y0 = max(0, int(math.floor(y0_f)))
        x1 = min(w, int(math.ceil(x1_f)))
        y1 = min(h, int(math.ceil(y1_f)))
        if x1 > x0 and y1 > y0:
            target_x = self.width() / 2.0 - (self.view_center.x() - x0) * scale
            target_y = self.height() / 2.0 - (self.view_center.y() - y0) * scale
            target_w = (x1 - x0) * scale
            target_h = (y1 - y0) * scale
            cache_key = (id(self.base_rgba), id(self.mask), id(self.result_rgba), self.mode, self._mask_version, x0, y0, x1, y1, int(round(target_w)), int(round(target_h)))
            if self._qimage_cache_key != cache_key or self._qimage_cache is None or self._pixmap_cache is None:
                crop = self.build_display_crop(x0, y0, x1, y1, int(round(target_w)), int(round(target_h)))
                self._qimage_cache = ndarray_rgba_to_qimage(crop)
                self._pixmap_cache = QPixmap.fromImage(self._qimage_cache)
                self._qimage_cache_key = cache_key
            pix = self._pixmap_cache
            painter.drawPixmap(QRectF(target_x, target_y, target_w, target_h), pix, QRectF(0, 0, pix.width(), pix.height()))
            painter.setPen(QPen(QColor(105, 105, 110), 1))
            painter.drawRect(QRectF(
                self.width() / 2.0 - self.view_center.x() * scale,
                self.height() / 2.0 - self.view_center.y() * scale,
                w * scale,
                h * scale,
            ))
        if self.mode == "result" and self.result_rgba is None:
            painter.setPen(QColor(255, 230, 120))
            painter.drawText(12, 24, "最新結果なし")
        active_crop = self._crop_rect_from_points(self.crop_start_img, self.crop_current_img) if self.crop_drawing else self.get_crop_rect()
        if active_crop is not None:
            x, y, cw, ch = active_crop
            tl = self.image_to_view(QPointF(float(x), float(y)))
            br = self.image_to_view(QPointF(float(x + cw), float(y + ch)))
            if tl is not None and br is not None:
                crop_rect_view = QRectF(tl, br).normalized()
                pen_color = QColor(80, 220, 255, 220) if self.crop_drawing else QColor(255, 225, 70, 230)
                painter.setPen(QPen(pen_color, 2.0, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(crop_rect_view)
                label_rect = QRectF(crop_rect_view.left() + 4, crop_rect_view.top() + 4, 72, 20)
                painter.fillRect(label_rect, QColor(0, 0, 0, 140))
                painter.setPen(QColor(255, 245, 160))
                painter.drawText(label_rect.adjusted(4, 0, -4, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "CROP")
        if self._live_draw_points:
            painter.save()
            img_rect = QRectF(
                self.width() / 2.0 - self.view_center.x() * scale,
                self.height() / 2.0 - self.view_center.y() * scale,
                w * scale,
                h * scale,
            )
            painter.setClipRect(img_rect)
            live_color = QColor(100, 210, 255, 150) if self._live_draw_tool == "eraser" else QColor(255, 230, 80, 145)
            live_width = max(1.0, self._live_draw_brush_size * scale)
            if len(self._live_draw_points) == 1:
                vp0 = self.image_to_view(self._live_draw_points[0])
                if vp0 is not None:
                    # 単発クリックのプレビューは塗り円だけ描く。
                    # 太いペンを付けたまま drawEllipse すると、外周線の分だけブラシより大きく見える。
                    no_pen = QPen()
                    no_pen.setStyle(Qt.PenStyle.NoPen)
                    painter.setPen(no_pen)
                    painter.setBrush(live_color)
                    painter.drawEllipse(vp0, live_width / 2.0, live_width / 2.0)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
            else:
                painter.setPen(QPen(live_color, live_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                start = self.image_to_view(self._live_draw_points[0])
                if start is not None:
                    path = QPainterPath(start)
                    for point in self._live_draw_points[1:]:
                        vp = self.image_to_view(point)
                        if vp is not None:
                            path.lineTo(vp)
                    painter.drawPath(path)
            painter.restore()
        if self.cursor_img_pos is not None and self.tool in {"brush", "eraser"}:
            vp = self.image_to_view(self.cursor_img_pos)
            if vp is not None:
                r = max(1.0, self.brush_size * scale / 2.0)
                color = QColor(255, 230, 80) if self.tool == "brush" else QColor(100, 210, 255)
                painter.setPen(QPen(color, 1.5))
                painter.drawEllipse(vp, r, r)
                painter.setPen(QPen(QColor(0, 0, 0, 160), 1))
                painter.drawEllipse(vp, max(1.0, r - 1.5), max(1.0, r - 1.5))

    # ---------- brush ----------
    def set_tool(self, tool: str) -> None:
        if tool not in {"brush", "eraser", "crop"}:
            tool = "brush"
        if self.tool != tool:
            self.tool = tool
            self.toolChanged.emit(tool)
            self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in {"image", "mask", "overlay", "result"}:
            mode = "overlay"
        if self.mode != mode:
            self.mode = mode
            self.invalidate_cache()
            self.update()

    def set_brush_size(self, size: int) -> None:
        self.brush_size = clamp_int(size, 1, 500)
        self._mask_stamp_cache.clear()
        self.update()

    def _apply_brush_line(self, p0: QPointF, p1: QPointF) -> None:
        if self.mask is None:
            return
        radius = max(1, int(round(self.brush_size / 2.0)))
        color = 0 if self.tool == "eraser" else 255
        pt0 = (int(round(p0.x())), int(round(p0.y())))
        pt1 = (int(round(p1.x())), int(round(p1.y())))
        cv2.line(self.mask, pt0, pt1, int(color), thickness=radius * 2, lineType=cv2.LINE_AA)
        # lineだけだとクリック点が細くなることがあるので端点を丸める。
        cv2.circle(self.mask, pt1, radius, int(color), thickness=-1, lineType=cv2.LINE_AA)

    def begin_draw(self, img_pos: QPointF) -> None:
        if self.mask is None:
            return
        self.drawing = True
        self.last_img_pos = img_pos
        self._live_draw_points = [QPointF(img_pos.x(), img_pos.y())]
        self._live_draw_tool = self.tool
        self._live_draw_brush_size = self.brush_size
        self._apply_brush_line(img_pos, img_pos)
        # 描画中は表示用の合成キャッシュを作り直さない。
        # Canvas上に軽いライブ線を重ね、正式なマスク合成はマウスを離した時だけ更新する。
        self.update()

    def continue_draw(self, img_pos: QPointF) -> None:
        if not self.drawing or self.last_img_pos is None:
            return
        if math.hypot(img_pos.x() - self.last_img_pos.x(), img_pos.y() - self.last_img_pos.y()) < 0.35:
            return
        self._apply_brush_line(self.last_img_pos, img_pos)
        self.last_img_pos = img_pos
        self._live_draw_points.append(QPointF(img_pos.x(), img_pos.y()))
        # ここで _mask_version を増やしたり cache を消したりしない。
        # それをやるとドラッグ中に毎回 overlay crop/resize/QPixmap 再生成が走って重くなる。
        self.update()

    def end_draw(self) -> None:
        if not self.drawing:
            return
        self.drawing = False
        self.last_img_pos = None
        self._live_draw_points = []
        self._mask_version += 1
        self.invalidate_cache()
        self.maskChanged.emit()
        self.update()

    def clear_mask(self) -> None:
        if self.mask is not None:
            self.mask[:, :] = 0
            self._mask_version += 1
            self.invalidate_cache()
            self.maskChanged.emit()
            self.update()

    def invert_mask(self) -> None:
        if self.mask is not None:
            self.mask[:, :] = 255 - self.mask
            self._mask_version += 1
            self.invalidate_cache()
            self.maskChanged.emit()
            self.update()

    # ---------- events ----------
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        paths = file_urls_to_paths(event.mimeData())
        if any(p.suffix.lower() in SUPPORTED_IMAGE_EXTS for p in paths):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = [p for p in file_urls_to_paths(event.mimeData()) if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        if paths:
            self.fileDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self.base_rgba is None:
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            event.ignore()
            return
        factor = 1.25 ** steps
        self.set_zoom(self.zoom_scale * factor, anchor_pos=QPointF(event.position()))
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = QPointF(event.position())
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or (event.button() == Qt.MouseButton.LeftButton and self.space_down):
            self.panning = True
            self.pan_last_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            img_pos = self.view_to_image(pos)
            if img_pos is not None:
                if self.tool == "crop":
                    self.begin_crop(img_pos)
                else:
                    self.begin_draw(img_pos)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        pos = QPointF(event.position())
        img_pos = self.view_to_image(pos)
        cursor_changed = (
            (img_pos is None) != (self.cursor_img_pos is None)
            or (img_pos is not None and self.cursor_img_pos is not None and (abs(img_pos.x() - self.cursor_img_pos.x()) > 0.5 or abs(img_pos.y() - self.cursor_img_pos.y()) > 0.5))
        )
        self.cursor_img_pos = img_pos
        if self.panning and self.pan_last_pos is not None and self.view_center is not None:
            scale = self.get_current_scale()
            delta = pos - self.pan_last_pos
            self.view_center = QPointF(self.view_center.x() - delta.x() / max(scale, 1e-6), self.view_center.y() - delta.y() / max(scale, 1e-6))
            self.pan_last_pos = pos
            self.clamp_view_center()
            self.invalidate_cache()
            self.update()
            event.accept()
            return
        if self.crop_drawing and img_pos is not None:
            self.continue_crop(img_pos)
            event.accept()
            return
        if self.drawing and img_pos is not None:
            self.continue_draw(img_pos)
            event.accept()
            return
        if cursor_changed:
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self.crop_drawing and event.button() == Qt.MouseButton.LeftButton:
            self.end_crop()
            event.accept()
            return
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.end_draw()
            event.accept()
            return
        if self.panning and event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self.panning = False
            self.pan_last_pos = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self.cursor_img_pos = None
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        key = event.key()
        if key == Qt.Key.Key_1:
            self.set_tool("brush")
            event.accept()
            return
        if key == Qt.Key.Key_2:
            self.set_tool("eraser")
            event.accept()
            return
        if key == Qt.Key.Key_3:
            self.set_tool("crop")
            event.accept()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.crop_enabled or self.crop_rect is not None:
                self.clear_crop_rect(emit_signal=True)
                event.accept()
                return
        if key == Qt.Key.Key_Space:
            self.space_down = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self.space_down = False
            if not self.panning:
                self.unsetCursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.clamp_view_center()
        self.invalidate_cache()


class ApiSettingsDialog(QDialog):
    def __init__(self, parent, settings: ApiSettings, log_func=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("API設定")
        self.settings = ApiSettings.from_dict(settings.to_dict())
        self.log_func = log_func
        self.resize(560, 260)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.base_url_edit = QLineEdit(self.settings.base_url)
        self.user_edit = QLineEdit(self.settings.username)
        self.password_edit = QLineEdit(self.settings.password)
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.save_password_check = QCheckBox("パスワードを保存")
        self.save_password_check.setChecked(self.settings.save_password)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 86400)
        self.timeout_spin.setValue(self.settings.timeout)
        self.verify_ssl_check = QCheckBox("SSL証明書を検証")
        self.verify_ssl_check.setChecked(self.settings.verify_ssl)
        form.addRow("Base URL", self.base_url_edit)
        form.addRow("User", self.user_edit)
        form.addRow("Password", self.password_edit)
        form.addRow("", self.save_password_check)
        form.addRow("Client Timeout 秒", self.timeout_spin)
        form.addRow("", self.verify_ssl_check)
        layout.addLayout(form)
        hint = QLabel("例: http://localhost:7860 / https://xxxx.container.sakurausercontent.com\nHTTP 504 はサーバー/プロキシ側の upstream timeout です。実行は1枚ずつ分割して長時間リクエストを避けます。")
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        self.test_btn = QPushButton("Test API")
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("キャンセル")
        self.test_btn.clicked.connect(self.test_api)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.test_btn)
        buttons.addStretch(1)
        # Windows風: [OK][キャンセル]
        buttons.addWidget(self.ok_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

    def collect(self) -> ApiSettings:
        self.settings.base_url = self.base_url_edit.text().strip().rstrip("/") or "http://localhost:7860"
        self.settings.username = self.user_edit.text().strip()
        self.settings.password = self.password_edit.text()
        self.settings.save_password = bool(self.save_password_check.isChecked())
        self.settings.timeout = int(self.timeout_spin.value())
        self.settings.verify_ssl = bool(self.verify_ssl_check.isChecked())
        return self.settings

    def test_api(self) -> None:
        settings = self.collect()
        try:
            models = api_get(settings, "/sdapi/v1/sd-models")
            samplers = api_get(settings, "/sdapi/v1/samplers")
            options = api_get(settings, "/sdapi/v1/options")
            model_name = options.get("sd_model_checkpoint", "") if isinstance(options, dict) else ""
            msg = f"API接続OK / models={len(models) if isinstance(models, list) else '?'} / samplers={len(samplers) if isinstance(samplers, list) else '?'} / checkpoint={model_name}"
            QMessageBox.information(self, "Test API", msg)
            if self.log_func:
                self.log_func(msg)
        except Exception as exc:
            QMessageBox.critical(self, "Test API", f"API接続失敗:\n{exc}")
            if self.log_func:
                self.log_func(f"API接続失敗: {exc}")


def api_request(settings: ApiSettings, method: str, path: str, payload: Optional[dict] = None) -> object:
    base = settings.base_url.rstrip("/")
    url = base + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    if settings.username or settings.password:
        token = base64.b64encode(f"{settings.username}:{settings.password}".encode("utf-8")).decode("ascii")
        req.add_header("Authorization", f"Basic {token}")
    context = None
    if url.lower().startswith("https://") and not settings.verify_ssl:
        import ssl
        context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=settings.timeout, context=context) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 504:
            raise RuntimeError(
                f"HTTP 504: upstream request timeout / サーバーまたはプロキシ側のタイムアウトです。"
                f" Client Timeoutを伸ばしても、サーバー側の上限が短いと回避できません。detail={detail[:800]}"
            )
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}")


def api_get(settings: ApiSettings, path: str) -> object:
    return api_request(settings, "GET", path)


def api_post(settings: ApiSettings, path: str, payload: dict) -> object:
    return api_request(settings, "POST", path, payload)


def set_widget_can_shrink(widget: QWidget) -> None:
    try:
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Policy.Ignored, widget.sizePolicy().verticalPolicy())
    except Exception:
        pass


def make_bold_label(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label




def set_button_keep_visible(button: QPushButton) -> None:
    try:
        button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(max(24, button.minimumSizeHint().width()))
    except Exception:
        pass


class ResultCanvas(MaskCanvas):
    """出力結果専用ビュー。MaskCanvasの高速描画・ズーム・パンだけを使い、編集操作は持たせない。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(False)
        self.tool = "view"
        self.mode = "image"

    def set_result_image(self, rgba: Optional[np.ndarray], fit: bool = True) -> None:
        if rgba is None:
            self.set_images(None, None, None, fit=fit)
            return
        h, w = rgba.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        self.crop_enabled = False
        self.crop_rect = None
        self.crop_drawing = False
        self.set_images(rgba, mask, None, fit=fit)
        self.mode = "image"
        self.tool = "view"

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self.base_rgba is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(32, 32, 34))
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "出力結果なし")
            return
        super().paintEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        event.ignore()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = QPointF(event.position())
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or (event.button() == Qt.MouseButton.LeftButton and self.space_down):
            self.panning = True
            self.pan_last_pos = pos
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        pos = QPointF(event.position())
        if self.panning and self.pan_last_pos is not None and self.view_center is not None:
            scale = self.get_current_scale()
            delta = pos - self.pan_last_pos
            self.view_center = QPointF(self.view_center.x() - delta.x() / max(scale, 1e-6), self.view_center.y() - delta.y() / max(scale, 1e-6))
            self.pan_last_pos = pos
            self.clamp_view_center()
            self.invalidate_cache()
            self.update()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self.panning and event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self.panning = False
            self.pan_last_pos = None
            self.unsetCursor()
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self.space_down = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        event.ignore()

    def keyReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space:
            self.space_down = False
            if not self.panning:
                self.unsetCursor()
            event.accept()
            return
        event.ignore()


class MainWindow(QMainWindow):
    logSignal = Signal(str)
    progressSignal = Signal(str)
    refreshJobsSignal = Signal()
    reloadResultSignal = Signal(str)
    runningSignal = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} {APP_REV}")
        self.setAcceptDrops(True)
        self.project_dir: Path = app_dir() / "a1111_inpaint_project"
        self.jobs: List[JobData] = []
        self.current_job_index: Optional[int] = None
        self.api_settings = ApiSettings()
        self.api_max_size = "unlimited"
        self.ui_font_point_size = self._initial_ui_font_point_size()
        self.param_presets: Dict[str, Dict[str, object]] = sanitize_param_presets(None)
        self.loading_ui = True
        self._refreshing_job_list = False
        self._selecting_job = False
        self._closing_app = False
        self.param_dirty = False
        self.image_dirty = False
        self.job_meta_dirty = False
        self.preset_mismatch = False
        self._mask_save_pending = False
        self._mask_save_timer = QTimer(self)
        self._mask_save_timer.setSingleShot(True)
        self._mask_save_timer.timeout.connect(self.flush_mask_save)
        self._run_thread: Optional[threading.Thread] = None
        self._cancel_requested = False
        self._last_normal_geometry = QRect(100, 100, 1380, 900)
        self._build_ui()
        self._build_actions()
        self.logSignal.connect(self.log)
        self.progressSignal.connect(self.progress_label.setText)
        self.refreshJobsSignal.connect(self.refresh_job_list)
        self.reloadResultSignal.connect(self.reload_current_result_by_id)
        self.runningSignal.connect(self.set_running_ui)
        self._load_app_settings()
        self._ensure_project()
        self.load_project(self.project_dir)
        self.loading_ui = False
        self.log(f"起動: {APP_TITLE} {APP_REV}")

    def _initial_ui_font_point_size(self) -> int:
        try:
            size = QApplication.instance().font().pointSize()
        except Exception:
            size = 10
        if size <= 0:
            size = 10
        return clamp_int(size, 9, 25)

    def _sync_text_area_heights(self) -> None:
        try:
            if hasattr(self, "prompt_edit"):
                prompt_h = self.prompt_edit.fontMetrics().lineSpacing() * 4 + 24
                self.prompt_edit.setMinimumHeight(prompt_h)
                self.prompt_edit.setMaximumHeight(prompt_h + 16)
            if hasattr(self, "negative_edit"):
                neg_h = self.negative_edit.fontMetrics().lineSpacing() * 2 + 18
                self.negative_edit.setMinimumHeight(neg_h)
                self.negative_edit.setMaximumHeight(neg_h + 8)
        except Exception:
            pass

    def apply_ui_font_size(self, point_size: int, save: bool = True, log_change: bool = True) -> None:
        point_size = clamp_int(point_size, 9, 25)
        self.ui_font_point_size = point_size
        try:
            app = QApplication.instance()
            if app is not None:
                font = app.font()
                font.setPointSize(point_size)
                app.setFont(font)
                self.setFont(font)
        except Exception:
            pass
        self._sync_text_area_heights()
        self.update_button_minimum_widths()
        self.sync_font_size_actions()
        if save:
            self._save_app_settings()
        if log_change:
            self.log(f"文字サイズ: {point_size}pt")

    def update_button_minimum_widths(self) -> None:
        try:
            for button in self.findChildren(QPushButton):
                set_button_keep_visible(button)
        except Exception:
            pass

    # ---------- UI ----------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        self.setMinimumSize(720, 520)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root.addWidget(self.splitter, 1)
        self.setCentralWidget(central)

        # left jobs
        left = QWidget()
        left.setMinimumWidth(150)
        left.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        job_list_title = QLabel("ジョブ一覧（画像D&Dで追加）")
        set_widget_can_shrink(job_list_title)
        left_layout.addWidget(job_list_title)
        self.job_list = QListWidget()
        self.job_list.setMinimumWidth(0)
        self.job_list.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.job_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.job_list.currentRowChanged.connect(self.on_job_selected)
        left_layout.addWidget(self.job_list, 1)
        job_btns1 = QHBoxLayout()
        self.add_btn = QPushButton("画像追加")
        self.remove_btn = QPushButton("削除")
        self.up_btn = QPushButton("↑")
        self.down_btn = QPushButton("↓")
        self.check_btn = QPushButton("チェック切替")
        self.add_btn.clicked.connect(self.add_images_dialog)
        self.remove_btn.clicked.connect(self.remove_current_job)
        self.up_btn.clicked.connect(lambda: self.move_job(-1))
        self.down_btn.clicked.connect(lambda: self.move_job(1))
        self.check_btn.clicked.connect(self.toggle_current_checked)
        for b in [self.add_btn, self.remove_btn, self.up_btn, self.down_btn, self.check_btn]:
            set_button_keep_visible(b)
            job_btns1.addWidget(b, 0)
        left_layout.addLayout(job_btns1)
        self.splitter.addWidget(left)

        # right tabs
        right = QWidget()
        right.setMinimumWidth(300)
        right.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(0)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.tabs, 1)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([330, 1050])

        self._build_image_tab()
        self._build_params_tab()
        self._build_run_tab()
        self._build_result_tab()
        self.update_button_minimum_widths()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.update_result_shortcuts_enabled()

    def _build_image_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        image_header = QHBoxLayout()
        image_header.addStretch(1)
        self.save_job_btn_image = QPushButton("現在のジョブを保存")
        self.save_job_btn_image.setToolTip("マスク、クロップ、プロンプト、パラメータを含めて現在のジョブを保存します。")
        set_button_keep_visible(self.save_job_btn_image)
        self.save_job_btn_image.clicked.connect(self.save_current_job_from_ui)
        image_header.addWidget(self.save_job_btn_image, 0)
        layout.addLayout(image_header)
        self.canvas = MaskCanvas()
        self.canvas.maskChanged.connect(self.on_mask_changed)
        self.canvas.cropChanged.connect(self.on_crop_changed)
        self.canvas.toolChanged.connect(self.on_canvas_tool_changed)
        self.canvas.fileDropped.connect(self.add_image_paths)
        layout.addWidget(self.canvas, 1)

        row1 = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(110)
        self.mode_combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.mode_combo.currentIndexChanged.connect(self.on_display_combo_changed)
        row1.addWidget(self.mode_combo, 0)
        self.refresh_display_combo(None, preferred_mode="overlay")
        row1.addSpacing(10)
        self.brush_btn = QPushButton("ブラシ [1]")
        self.brush_btn.setCheckable(True)
        self.eraser_btn = QPushButton("消しゴム [2]")
        self.eraser_btn.setCheckable(True)
        self.crop_btn = QPushButton("クロップ [3]")
        self.crop_btn.setCheckable(True)
        self.brush_btn.clicked.connect(lambda: self.canvas.set_tool("brush"))
        self.eraser_btn.clicked.connect(lambda: self.canvas.set_tool("eraser"))
        self.crop_btn.clicked.connect(lambda: self.canvas.set_tool("crop"))
        for b in [self.brush_btn, self.eraser_btn, self.crop_btn]:
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        row1.addWidget(self.brush_btn, 0)
        row1.addWidget(self.eraser_btn, 0)
        row1.addWidget(self.crop_btn, 0)
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 500)
        self.brush_size_spin.setValue(48)
        self.brush_size_spin.setMinimumWidth(60)
        self.brush_size_spin.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.brush_size_spin.valueChanged.connect(self.canvas.set_brush_size)
        row1.addWidget(QLabel("サイズ"))
        row1.addWidget(self.brush_size_spin, 0)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.fit_btn = QPushButton("表示合わせ")
        self.actual_btn = QPushButton("100%")
        self.clear_mask_btn = QPushButton("マスク全消去")
        self.invert_mask_btn = QPushButton("反転")
        self.load_mask_btn = QPushButton("マスク読込")
        self.clear_crop_btn = QPushButton("クロップ解除")
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.actual_btn.clicked.connect(self.canvas.set_actual_size)
        self.clear_mask_btn.clicked.connect(self.canvas.clear_mask)
        self.invert_mask_btn.clicked.connect(self.canvas.invert_mask)
        self.load_mask_btn.clicked.connect(self.load_mask_dialog)
        self.clear_crop_btn.clicked.connect(lambda: self.canvas.clear_crop_rect())
        for b in [self.fit_btn, self.actual_btn, self.clear_mask_btn, self.invert_mask_btn, self.load_mask_btn, self.clear_crop_btn]:
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            row2.addWidget(b, 0)
        row2.addStretch(1)
        layout.addLayout(row2)
        hint = QLabel("左ドラッグ: ブラシ/消しゴム/クロップ / 右ドラッグ: 表示移動 / ホイール: 拡大縮小 / Space+左ドラッグ: 表示移動 / Delete: クロップ解除")
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)
        self.crop_info_label = QLabel("クロップ: 未設定")
        self.crop_info_label.setStyleSheet("color: #666;")
        layout.addWidget(self.crop_info_label)
        self.tabs.addTab(tab, "画像編集")
        self.on_canvas_tool_changed("brush")
        self.update_crop_info_label()

    def _build_params_tab(self) -> None:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(0, 0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)
        param_header = QHBoxLayout()
        param_header.addStretch(1)
        self.save_job_btn_params = QPushButton("現在のジョブを保存")
        self.save_job_btn_params.setToolTip("マスク、クロップ、プロンプト、パラメータを含めて現在のジョブを保存します。")
        set_button_keep_visible(self.save_job_btn_params)
        self.save_job_btn_params.clicked.connect(self.save_current_job_from_ui)
        param_header.addWidget(self.save_job_btn_params, 0)
        root.addLayout(param_header)

        name_row = QHBoxLayout()
        name_row.addWidget(make_bold_label("ジョブ名"))
        self.job_name_edit = QLineEdit()
        set_widget_can_shrink(self.job_name_edit)
        self.job_name_edit.textEdited.connect(lambda _text: self.mark_current_job_dirty(meta=True))
        name_row.addWidget(self.job_name_edit, 1)
        self.checked_box = QCheckBox("一括実行対象")
        self.checked_box.toggled.connect(lambda _c: self.mark_current_job_dirty(meta=True))
        name_row.addWidget(self.checked_box)
        root.addLayout(name_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(make_bold_label("プリセット"))
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        no_insert = getattr(getattr(QComboBox, "InsertPolicy", QComboBox), "NoInsert")
        self.preset_combo.setInsertPolicy(no_insert)
        self.preset_combo.setMinimumWidth(120)
        set_widget_can_shrink(self.preset_combo)
        self.preset_combo.activated.connect(self.on_preset_activated)
        self.preset_combo.editTextChanged.connect(lambda _text: self.update_preset_buttons_state())
        self.preset_status_label = QLabel("")
        self.preset_status_label.setStyleSheet("color: #d08000;")
        self.preset_new_btn = QPushButton("新規")
        self.preset_update_btn = QPushButton("変更")
        self.preset_del_btn = QPushButton("削除")
        set_button_keep_visible(self.preset_new_btn)
        set_button_keep_visible(self.preset_update_btn)
        set_button_keep_visible(self.preset_del_btn)
        self.preset_new_btn.clicked.connect(self.new_param_preset_from_current)
        self.preset_update_btn.clicked.connect(self.update_current_param_preset)
        self.preset_del_btn.clicked.connect(self.delete_current_param_preset)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.preset_status_label)
        preset_row.addWidget(self.preset_new_btn)
        preset_row.addWidget(self.preset_update_btn)
        preset_row.addWidget(self.preset_del_btn)
        root.addLayout(preset_row)
        self.refresh_preset_combo()

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Prompt")
        set_widget_can_shrink(self.prompt_edit)
        self.negative_edit = QTextEdit()
        self.negative_edit.setPlaceholderText("Negative Prompt")
        set_widget_can_shrink(self.negative_edit)
        self._sync_text_area_heights()
        root.addWidget(make_bold_label("Prompt"))
        root.addWidget(self.prompt_edit)
        root.addWidget(make_bold_label("Negative Prompt"))
        root.addWidget(self.negative_edit)

        self.sampler_edit = QLineEdit("Euler a")
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 150); self.steps_spin.setValue(28)
        self.cfg_spin = QDoubleSpinBox(); self.cfg_spin.setRange(0.0, 30.0); self.cfg_spin.setSingleStep(0.5); self.cfg_spin.setValue(5.0)
        self.denoise_spin = QDoubleSpinBox(); self.denoise_spin.setRange(0.0, 1.0); self.denoise_spin.setSingleStep(0.05); self.denoise_spin.setDecimals(2); self.denoise_spin.setValue(0.55)
        self.mask_blur_spin = QSpinBox(); self.mask_blur_spin.setRange(0, 128); self.mask_blur_spin.setValue(8)
        self.padding_spin = QSpinBox(); self.padding_spin.setRange(0, 512); self.padding_spin.setValue(96)
        self.fill_combo = QComboBox(); self.fill_combo.addItem("fill", 0); self.fill_combo.addItem("original", 1); self.fill_combo.addItem("latent noise", 2); self.fill_combo.addItem("latent nothing", 3); self.fill_combo.setCurrentIndex(1)
        self.full_res_check = QCheckBox("inpaint_full_res"); self.full_res_check.setChecked(True)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 16); self.batch_spin.setValue(1)
        self.niter_spin = QSpinBox(); self.niter_spin.setRange(1, 100); self.niter_spin.setValue(4)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 2147483647); self.seed_spin.setValue(-1)
        for widget in [
            self.sampler_edit, self.steps_spin, self.cfg_spin, self.denoise_spin,
            self.mask_blur_spin, self.padding_spin, self.fill_combo, self.full_res_check,
            self.batch_spin, self.niter_spin, self.seed_spin, self.checked_box,
        ]:
            set_widget_can_shrink(widget)
        self._connect_preset_dirty_signals()

        param_column = QVBoxLayout()
        param_column.setSpacing(6)
        param_descriptions = [
            ("Sampler", self.sampler_edit, "画像生成の計算方式。Euler a は速くて変化が出やすい。"),
            ("Steps", self.steps_spin, "生成の反復回数。多いほど丁寧だが遅い。28なら普通〜やや多め。"),
            ("CFG", self.cfg_spin, "プロンプトへの従わせ具合。高いほど指示に強く寄るが破綻もしやすい。5.0は控えめ。"),
            ("Denoise", self.denoise_spin, "元画像をどれだけ作り変えるか。0.55 はそこそこ変える。低いほど元画像維持。"),
            ("Mask blur", self.mask_blur_spin, "マスク境界のぼかし量。大きいほど修正部分の境目がなじむが、広がりやすい。"),
            ("Padding", self.padding_spin, "マスク周辺をどれだけ含めて再生成するか。大きいほど周囲との整合性を見やすい。"),
            ("Fill", self.fill_combo, "マスク部分の初期内容。original は元画像をベースに修正する。"),
            ("inpaint_full_res", self.full_res_check, "マスク部分を高解像度で切り抜いて処理する。細部修正向き。"),
            ("Batch size", self.batch_spin, "同時に生成する枚数。VRAM使用量に直結。"),
            ("n_iter", self.niter_spin, "生成を何回繰り返すか。最終的な出力枚数は Batch size × n_iter。"),
            ("Seed", self.seed_spin, "乱数の固定値。-1 は毎回ランダム。"),
        ]
        for label_text, widget, description in param_descriptions:
            field_row = QHBoxLayout()
            label = make_bold_label(label_text)
            label.setMinimumWidth(120)
            field_row.addWidget(label)
            field_row.addWidget(widget, 1)
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666;")
            param_column.addLayout(field_row)
            param_column.addWidget(desc_label)
        root.addLayout(param_column)
        root.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        self.tabs.addTab(tab, "パラメータ")

    def _build_run_tab(self) -> None:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(8, 8, 8, 8)

        self.api_btn = QPushButton("API設定")
        self.test_btn = QPushButton("Test API")
        self.dry_btn = QPushButton("DryRun request.json")
        self.run_current_btn = QPushButton("選択ジョブ実行")
        self.run_checked_btn = QPushButton("チェック済み一括実行")
        self.run_failed_btn = QPushButton("失敗だけ再実行")
        self.stop_btn = QPushButton("中断")

        self.api_btn.clicked.connect(self.show_api_settings)
        self.test_btn.clicked.connect(self.test_api)
        self.dry_btn.clicked.connect(self.dry_run_current)
        self.run_current_btn.clicked.connect(self.run_current_job)
        self.run_checked_btn.clicked.connect(self.run_checked_jobs)
        self.run_failed_btn.clicked.connect(self.run_failed_jobs)
        self.stop_btn.clicked.connect(self.request_stop)

        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        for b in [self.api_btn, self.test_btn, self.dry_btn, self.run_current_btn]:
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            row1.addWidget(b, 0)
        row1.addStretch(1)
        for b in [self.run_checked_btn, self.run_failed_btn, self.stop_btn]:
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            row2.addWidget(b, 0)
        row2.addStretch(1)
        root.addLayout(row1)
        root.addLayout(row2)

        self.progress_label = QLabel("待機中")
        root.addWidget(self.progress_label)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        root.addWidget(self.log_edit, 1)
        self.tabs.addTab(tab, "実行")

    def _build_result_tab(self) -> None:
        tab = QWidget()
        self.result_tab = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        self.result_canvas = ResultCanvas()
        layout.addWidget(self.result_canvas, 1)

        row = QHBoxLayout()
        self.result_combo = QComboBox()
        self.result_combo.setMinimumWidth(150)
        self.result_combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.result_combo.currentIndexChanged.connect(self.on_result_combo_changed)
        self.result_open_btn = QPushButton("開く")
        self.result_copy_btn = QPushButton("コピー")
        self.result_delete_btn = QPushButton("削除")
        for b in [self.result_open_btn, self.result_copy_btn, self.result_delete_btn]:
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.result_open_btn.clicked.connect(self.open_selected_result_path)
        self.result_copy_btn.clicked.connect(self.copy_selected_result_path)
        self.result_delete_btn.clicked.connect(self.delete_selected_result_path)
        row.addWidget(self.result_combo, 0)
        row.addWidget(self.result_open_btn, 0)
        row.addWidget(self.result_copy_btn, 0)
        row.addWidget(self.result_delete_btn, 0)
        row.addStretch(1)
        layout.addLayout(row)

        hint = QLabel("ホイール: 拡大縮小 / 右ドラッグ: 表示移動 / Space+左ドラッグ: 表示移動 / 次: Alt+Right, Right, Down / 前: Alt+Left, Left, Up")
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        self.result_shortcuts = []

        def add_result_shortcut(seq: str, delta: int) -> None:
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.setEnabled(False)
            shortcut.activated.connect(lambda d=delta: self.navigate_result_selection(d))
            self.result_shortcuts.append(shortcut)

        for seq in ["Right", "Down", "Alt+Right"]:
            add_result_shortcut(seq, 1)
        for seq in ["Left", "Up", "Alt+Left"]:
            add_result_shortcut(seq, -1)

        self.refresh_result_combo(None)
        self.tabs.addTab(tab, "出力結果")

    def update_result_shortcuts_enabled(self) -> None:
        enabled = getattr(self, "result_tab", None) is not None and self.tabs.currentWidget() is self.result_tab
        for shortcut in getattr(self, "result_shortcuts", []):
            try:
                shortcut.setEnabled(enabled)
            except Exception:
                pass

    def on_tab_changed(self, _index: int) -> None:
        self.update_result_shortcuts_enabled()
        if getattr(self, "result_tab", None) is not None and self.tabs.currentWidget() is self.result_tab:
            try:
                self.result_canvas.setFocus(Qt.FocusReason.TabFocusReason)
            except Exception:
                pass

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")
        new_project = QAction("新規プロジェクト", self)
        new_project.setShortcut(QKeySequence("Ctrl+N"))
        new_project.triggered.connect(self.new_project_dialog)
        open_project = QAction("プロジェクトを開く", self)
        open_project.setShortcut(QKeySequence("Ctrl+O"))
        open_project.triggered.connect(self.open_project_dialog)
        save_project = QAction("保存", self)
        save_project.setShortcut(QKeySequence("Ctrl+S"))
        save_project.triggered.connect(self.save_project_all)
        add_img = QAction("画像追加", self)
        add_img.triggered.connect(self.add_images_dialog)
        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        for a in [new_project, open_project, save_project, add_img, exit_action]:
            file_menu.addAction(a)
        edit_menu = self.menuBar().addMenu("編集")
        brush_action = QAction("ブラシ", self); brush_action.triggered.connect(lambda: self.canvas.set_tool("brush"))
        eraser_action = QAction("消しゴム", self); eraser_action.triggered.connect(lambda: self.canvas.set_tool("eraser"))
        crop_action = QAction("クロップ", self); crop_action.triggered.connect(lambda: self.canvas.set_tool("crop"))
        clear_crop_action = QAction("クロップ解除", self); clear_crop_action.triggered.connect(lambda: self.canvas.clear_crop_rect())
        edit_menu.addAction(brush_action); edit_menu.addAction(eraser_action); edit_menu.addAction(crop_action); edit_menu.addAction(clear_crop_action)
        view_menu = self.menuBar().addMenu("表示")
        fit_action = QAction("表示合わせ", self); fit_action.setShortcut("F"); fit_action.triggered.connect(self.canvas.fit_to_window)
        actual_action = QAction("100%", self); actual_action.setShortcut("Ctrl+1"); actual_action.triggered.connect(self.canvas.set_actual_size)
        view_menu.addAction(fit_action); view_menu.addAction(actual_action)
        settings_menu = self.menuBar().addMenu("設定")
        max_menu = settings_menu.addMenu("最大サイズ")
        self.max_size_action_group = QActionGroup(self)
        self.max_size_action_group.setExclusive(True)
        self.max_size_actions: Dict[str, QAction] = {}
        for label, key in [("無制限", "unlimited"), ("1920x1080", "1920x1080"), ("1280x720", "1280x720")]:
            action = QAction(label, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, k=key: self.set_api_max_size(k))
            self.max_size_action_group.addAction(action)
            self.max_size_actions[key] = action
            max_menu.addAction(action)
        self.sync_max_size_actions()

        font_menu = settings_menu.addMenu("文字サイズ")
        self.font_size_action_group = QActionGroup(self)
        self.font_size_action_group.setExclusive(True)
        self.font_size_actions: Dict[int, QAction] = {}
        for point_size in range(9, 26):
            action = QAction(f"{point_size}pt", self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, s=point_size: self.apply_ui_font_size(s))
            self.font_size_action_group.addAction(action)
            self.font_size_actions[point_size] = action
            font_menu.addAction(action)
        self.sync_font_size_actions()

    def set_api_max_size(self, key: str) -> None:
        if key not in MAX_API_SIZE_OPTIONS:
            key = "unlimited"
        if self.api_max_size != key:
            self.api_max_size = key
            self.sync_max_size_actions()
            self._save_app_settings()
            label = "無制限" if key == "unlimited" else key
            self.log(f"API最大サイズ: {label}")

    def sync_max_size_actions(self) -> None:
        actions = getattr(self, "max_size_actions", {})
        key = self.api_max_size if self.api_max_size in MAX_API_SIZE_OPTIONS else "unlimited"
        for action_key, action in actions.items():
            action.blockSignals(True)
            action.setChecked(action_key == key)
            action.blockSignals(False)

    def sync_font_size_actions(self) -> None:
        actions = getattr(self, "font_size_actions", {})
        size = clamp_int(getattr(self, "ui_font_point_size", self._initial_ui_font_point_size()), 9, 25)
        for point_size, action in actions.items():
            action.blockSignals(True)
            action.setChecked(point_size == size)
            action.blockSignals(False)

    # ---------- dirty state / parameter presets ----------
    def current_preset_text(self) -> str:
        combo = getattr(self, "preset_combo", None)
        return combo.currentText().strip() if combo is not None else ""

    def current_valid_preset_name(self) -> str:
        name = self.current_preset_text()
        return name if name in self.param_presets else ""

    def has_unsaved_current_job(self) -> bool:
        return bool(self.param_dirty or self.image_dirty or self.job_meta_dirty or self.preset_mismatch)

    def clear_dirty_flags(self) -> None:
        self.param_dirty = False
        self.image_dirty = False
        self.job_meta_dirty = False
        self.preset_mismatch = False
        self.update_dirty_ui()

    def mark_current_job_dirty(self, meta: bool = False, param: bool = False, image: bool = False, preset_mismatch: bool = False) -> None:
        if getattr(self, "loading_ui", False):
            return
        if self.current_job() is None:
            return
        if meta:
            self.job_meta_dirty = True
        if param:
            self.param_dirty = True
        if image:
            self.image_dirty = True
        if preset_mismatch and self.current_valid_preset_name():
            self.preset_mismatch = True
        job = self.current_job()
        if job is not None:
            job.status = "編集中"
            self.update_current_job_list_item()
        self.update_dirty_ui()

    def update_dirty_ui(self) -> None:
        dirty = self.has_unsaved_current_job()
        title = f"{APP_TITLE} {APP_REV}" + (" *" if dirty else "")
        try:
            self.setWindowTitle(title)
        except Exception:
            pass
        if hasattr(self, "preset_status_label"):
            self.preset_status_label.setText("*" if self.preset_mismatch and self.current_valid_preset_name() else "")
        self.update_preset_buttons_state()

    def update_current_job_list_item(self) -> None:
        idx = self.current_job_index
        if idx is None or not (0 <= idx < len(self.jobs)):
            return
        item = self.job_list.item(idx) if hasattr(self, "job_list") else None
        if item is None:
            return
        job = self.jobs[idx]
        if hasattr(self, "checked_box") and idx == self.current_job_index:
            mark = "☑" if self.checked_box.isChecked() else "☐"
        else:
            mark = "☑" if job.checked else "☐"
        display_name = self.job_name_edit.text().strip() if hasattr(self, "job_name_edit") and idx == self.current_job_index else job.name
        display_name = display_name or job.name or job.job_id
        item.setText(f"{mark} {job.job_id} / {job.status} / {display_name}")
        item.setData(Qt.ItemDataRole.UserRole, job.job_id)

    def refresh_preset_combo(self, select_name: Optional[str] = None) -> None:
        combo = getattr(self, "preset_combo", None)
        if combo is None:
            return
        target = str(select_name or "").strip()
        if target and target not in self.param_presets:
            target = ""
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem("")
            for name in self.param_presets.keys():
                combo.addItem(name)
            if target:
                idx = combo.findText(target)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                combo.setCurrentIndex(0)
                if combo.lineEdit() is not None:
                    combo.lineEdit().setText("")
        finally:
            combo.blockSignals(False)
        self.update_dirty_ui()

    def set_preset_combo_text(self, name: str) -> None:
        combo = getattr(self, "preset_combo", None)
        if combo is None:
            return
        name = str(name or "").strip()
        if name and name not in self.param_presets:
            name = ""
        combo.blockSignals(True)
        try:
            idx = combo.findText(name) if name else 0
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            if combo.lineEdit() is not None:
                combo.lineEdit().setText(name)
        finally:
            combo.blockSignals(False)
        self.update_dirty_ui()

    def update_preset_buttons_state(self) -> None:
        combo = getattr(self, "preset_combo", None)
        text = combo.currentText().strip() if combo is not None else ""
        exists = bool(text and text in self.param_presets)
        if hasattr(self, "preset_update_btn"):
            self.preset_update_btn.setEnabled(exists)
        if hasattr(self, "preset_del_btn"):
            self.preset_del_btn.setEnabled(exists)

    def _connect_preset_dirty_signals(self) -> None:
        dirty_param = lambda: self.mark_current_job_dirty(param=True, preset_mismatch=True)
        self.prompt_edit.textChanged.connect(dirty_param)
        self.negative_edit.textChanged.connect(dirty_param)
        self.sampler_edit.textChanged.connect(dirty_param)
        self.steps_spin.valueChanged.connect(lambda _v: dirty_param())
        self.cfg_spin.valueChanged.connect(lambda _v: dirty_param())
        self.denoise_spin.valueChanged.connect(lambda _v: dirty_param())
        self.mask_blur_spin.valueChanged.connect(lambda _v: dirty_param())
        self.padding_spin.valueChanged.connect(lambda _v: dirty_param())
        self.fill_combo.currentIndexChanged.connect(lambda _v: dirty_param())
        self.full_res_check.toggled.connect(lambda _v: dirty_param())
        self.batch_spin.valueChanged.connect(lambda _v: dirty_param())
        self.niter_spin.valueChanged.connect(lambda _v: dirty_param())
        self.seed_spin.valueChanged.connect(lambda _v: dirty_param())

    def set_params_to_ui(self, params: InpaintParams) -> None:
        prev_loading = self.loading_ui
        self.loading_ui = True
        try:
            self.prompt_edit.setPlainText(params.prompt)
            self.negative_edit.setPlainText(params.negative_prompt)
            self.sampler_edit.setText(params.sampler_name)
            self.steps_spin.setValue(params.steps)
            self.cfg_spin.setValue(params.cfg_scale)
            self.denoise_spin.setValue(params.denoising_strength)
            self.mask_blur_spin.setValue(params.mask_blur)
            self.padding_spin.setValue(params.inpaint_full_res_padding)
            idx = self.fill_combo.findData(params.inpainting_fill)
            self.fill_combo.setCurrentIndex(max(0, idx))
            self.full_res_check.setChecked(params.inpaint_full_res)
            self.batch_spin.setValue(params.batch_size)
            self.niter_spin.setValue(params.n_iter)
            self.seed_spin.setValue(params.seed)
        finally:
            self.loading_ui = prev_loading

    def on_preset_activated(self, index: int) -> None:
        if getattr(self, "loading_ui", False):
            return
        name = self.preset_combo.itemText(index).strip() if index >= 0 else self.preset_combo.currentText().strip()
        self.apply_param_preset(name)

    def apply_param_preset(self, name: str) -> None:
        name = str(name or "").strip()
        values = self.param_presets.get(name)
        if not values:
            return
        prev_applying = getattr(self, "_applying_preset", False)
        self._applying_preset = True
        try:
            params = InpaintParams.from_dict(values)
            self.set_params_to_ui(params)
            job = self.current_job()
            if job is not None:
                job.preset_name = name
            self.set_preset_combo_text(name)
            self.preset_mismatch = False
            self.mark_current_job_dirty(param=True)
        finally:
            self._applying_preset = prev_applying
        self.log(f"プリセット適用: {name}")

    def new_param_preset_from_current(self) -> None:
        base = self.preset_combo.currentText().strip() if hasattr(self, "preset_combo") else ""
        name = unique_name(base or "新規プリセット", self.param_presets.keys())
        self.param_presets[name] = asdict(self.params_from_ui())
        self.refresh_preset_combo(name)
        job = self.current_job()
        if job is not None:
            job.preset_name = name
            self.preset_mismatch = False
            self.mark_current_job_dirty(param=True)
        self._save_app_settings()
        self.log(f"プリセット保存: {name}")

    def update_current_param_preset(self) -> None:
        if not hasattr(self, "preset_combo"):
            return
        name = self.preset_combo.currentText().strip()
        if not name or name not in self.param_presets:
            return
        affected = [j for j in self.jobs if getattr(j, "preset_name", "") == name]
        current = self.current_job()
        if current is not None and current not in affected and self.current_valid_preset_name() == name:
            affected.append(current)
        msg = QMessageBox(self)
        msg.setWindowTitle("プリセット変更")
        msg.setText(f"プリセット「{name}」を現在の設定で変更します。\n同じ名前のプリセットを使用している {len(affected)} 件のジョブにも反映します。")
        ok_btn = msg.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(ok_btn)
        msg.exec()
        if msg.clickedButton() is not ok_btn:
            return
        values = asdict(self.params_from_ui())
        params = InpaintParams.from_dict(values)
        self.param_presets[name] = asdict(params)
        for job in affected:
            job.params = InpaintParams.from_dict(values)
            job.preset_name = name
            self.save_job(job)
        if current is not None and current in affected:
            self.param_dirty = False
            self.preset_mismatch = False
            self.update_dirty_ui()
            self.refresh_job_list(select_row=self.current_job_index if self.current_job_index is not None else -1)
            self.update_current_job_list_item()
        self._save_app_settings()
        self.log(f"プリセット変更: {name} / 反映 {len(affected)}件")

    def delete_current_param_preset(self) -> None:
        if not hasattr(self, "preset_combo"):
            return
        name = self.preset_combo.currentText().strip()
        if not name or name not in self.param_presets:
            return
        msg = QMessageBox(self)
        msg.setWindowTitle("プリセット削除")
        msg.setText(f"プリセット「{name}」を削除します。\nこのプリセットを使用しているジョブのプリセット名は空欄になります。")
        ok_btn = msg.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("キャンセル", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        if msg.clickedButton() is not ok_btn:
            return
        del self.param_presets[name]
        changed = 0
        for job in self.jobs:
            if getattr(job, "preset_name", "") == name:
                job.preset_name = ""
                self.save_job(job)
                changed += 1
        current = self.current_job()
        if current is not None and getattr(current, "preset_name", "") == name:
            current.preset_name = ""
        self.refresh_preset_combo("")
        self.preset_mismatch = False
        self.update_dirty_ui()
        self._save_app_settings()
        self.log(f"プリセット削除: {name} / 解除 {changed}件")

    def confirm_unsaved_current_job(self) -> bool:
        if not self.has_unsaved_current_job():
            return True
        msg = QMessageBox(self)
        msg.setWindowTitle("未保存の変更")
        msg.setText("現在のジョブに未保存の変更があります。保存しますか？")
        save_btn = msg.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = msg.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(save_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is save_btn:
            return bool(self.save_current_job_from_ui())
        if clicked is discard_btn:
            self.discard_current_job_changes()
            return True
        return False

    def discard_current_job_changes(self) -> None:
        idx = self.current_job_index
        if idx is None or not (0 <= idx < len(self.jobs)):
            self.clear_dirty_flags()
            return
        old_job = self.jobs[idx]
        job_file = self.job_dir(old_job.job_id) / JOB_FILE_NAME
        try:
            if job_file.exists():
                fresh = JobData.from_dict(read_json_utf8(job_file, {}))
                self.jobs[idx] = fresh
                self.load_job_to_ui(fresh)
            else:
                self.load_job_to_ui(old_job)
            self.clear_dirty_flags()
            self.refresh_job_list(select_row=idx)
            self.log("未保存の変更を破棄")
        except Exception as exc:
            self.log(f"破棄失敗: {safe_exception_text(exc)}")

    # ---------- settings ----------
    def _load_app_settings(self) -> None:
        data = read_json_utf8(settings_path(), {})
        if not isinstance(data, dict):
            data = {}
        self.api_settings = ApiSettings.from_dict(data.get("api", {}))
        self.param_presets = sanitize_param_presets(data.get("param_presets"))
        self.refresh_preset_combo()
        max_size = str(data.get("max_api_size", "unlimited"))
        self.api_max_size = max_size if max_size in MAX_API_SIZE_OPTIONS else "unlimited"
        self.sync_max_size_actions()
        try:
            self.apply_ui_font_size(int(data.get("ui_font_point_size", self.ui_font_point_size)), save=False, log_change=False)
        except Exception:
            self.apply_ui_font_size(self.ui_font_point_size, save=False, log_change=False)
        proj = str(data.get("last_project", "")).strip()
        if proj:
            self.project_dir = Path(proj)
        window = data.get("window", {}) if isinstance(data.get("window"), dict) else {}
        try:
            geom = window.get("normal_geometry")
            if isinstance(geom, dict):
                x = int(geom.get("x", 100)); y = int(geom.get("y", 100)); w = int(geom.get("width", 1380)); h = int(geom.get("height", 900))
                w = clamp_int(w, 640, 5000); h = clamp_int(h, 480, 4000)
                self.resize(w, h)
                self.move(x, y)
                self._last_normal_geometry = QRect(x, y, w, h)
            else:
                self.resize(1380, 900)
            if window.get("maximized"):
                QTimer.singleShot(0, self.showMaximized)
        except Exception:
            self.resize(1380, 900)
        sizes = data.get("splitter_sizes")
        if isinstance(sizes, list) and len(sizes) >= 2:
            try:
                self.splitter.setSizes([int(sizes[0]), int(sizes[1])])
            except Exception:
                pass

    def _save_app_settings(self) -> None:
        if getattr(self, "loading_ui", False):
            return
        try:
            geom = self._last_normal_geometry
            if not self.isMaximized() and not self.isMinimized():
                geom = self.geometry()
                self._last_normal_geometry = QRect(geom)
            data = {
                "app": APP_TITLE,
                "app_rev": APP_REV,
                "last_project": str(self.project_dir),
                "api": self.api_settings.to_dict(),
                "max_api_size": self.api_max_size,
                "ui_font_point_size": self.ui_font_point_size,
                "param_presets": self.param_presets,
                "window": {
                    "normal_geometry": {"x": geom.x(), "y": geom.y(), "width": geom.width(), "height": geom.height()},
                    "maximized": bool(self.isMaximized()),
                },
                "splitter_sizes": self.splitter.sizes(),
            }
            write_json_utf8(settings_path(), data)
        except Exception as exc:
            print(f"settings save error: {exc}")

    # ---------- project/jobs ----------
    def _ensure_project(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "jobs").mkdir(parents=True, exist_ok=True)
        if not (self.project_dir / PROJECT_FILE_NAME).exists():
            write_json_utf8(self.project_dir / PROJECT_FILE_NAME, {"app": APP_TITLE, "app_rev": APP_REV, "jobs": []})

    def new_project_dialog(self) -> None:
        if not self.confirm_unsaved_current_job():
            return
        path = QFileDialog.getExistingDirectory(self, "新規/既存プロジェクトフォルダを選択", str(self.project_dir.parent))
        if not path:
            return
        self.project_dir = Path(path)
        self._ensure_project()
        self.jobs = []
        self.current_job_index = None
        self.write_project_index()
        self.load_project(self.project_dir)
        self._save_app_settings()

    def open_project_dialog(self) -> None:
        if not self.confirm_unsaved_current_job():
            return
        path = QFileDialog.getExistingDirectory(self, "プロジェクトフォルダを開く", str(self.project_dir))
        if path:
            self.load_project(Path(path))
            self._save_app_settings()

    def load_project(self, project_dir: Path) -> None:
        prev_loading = self.loading_ui
        self.loading_ui = True
        try:
            self.project_dir = project_dir
            self._ensure_project()
            self.jobs = []
            self.current_job_index = None
            proj = read_json_utf8(self.project_dir / PROJECT_FILE_NAME, {})
            raw_jobs = proj.get("jobs", []) if isinstance(proj, dict) else []
            if isinstance(raw_jobs, list):
                for item in raw_jobs:
                    try:
                        jid = str(item.get("job_id") if isinstance(item, dict) else item)
                        job_path = self.job_dir(jid) / JOB_FILE_NAME
                        if job_path.exists():
                            self.jobs.append(JobData.from_dict(read_json_utf8(job_path, {})))
                    except Exception as exc:
                        print(f"Job metadata read error: {safe_exception_text(exc)}", file=sys.stderr)
            # フォルダにあるがproject.jsonにないジョブも拾う。
            jobs_root = self.project_dir / "jobs"
            for folder in sorted(jobs_root.glob("job_*")):
                if not folder.is_dir():
                    continue
                if any(j.job_id == folder.name for j in self.jobs):
                    continue
                job_file = folder / JOB_FILE_NAME
                if job_file.exists():
                    try:
                        self.jobs.append(JobData.from_dict(read_json_utf8(job_file, {})))
                    except Exception as exc:
                        print(f"Job folder read error: {safe_exception_text(exc)}", file=sys.stderr)
            if self.jobs:
                self.current_job_index = 0
                self.refresh_job_list(select_row=0)
                self.load_job_to_ui(self.jobs[0])
            else:
                self.refresh_job_list(select_row=-1)
                self.load_job_to_ui(None)
            self.log(f"プロジェクト読込: {self.project_dir}")
        finally:
            self.loading_ui = prev_loading

    def write_project_index(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        write_json_utf8(self.project_dir / PROJECT_FILE_NAME, {"app": APP_TITLE, "app_rev": APP_REV, "jobs": [{"job_id": j.job_id} for j in self.jobs]})
        self._save_app_settings()

    def save_project_all(self) -> None:
        if not self.save_current_job_from_ui():
            return
        self.project_dir.mkdir(parents=True, exist_ok=True)
        for job in self.jobs:
            self.save_job(job)
        self.write_project_index()
        self.log("保存完了")

    def next_job_id(self) -> str:
        root = self.project_dir / "jobs"
        root.mkdir(parents=True, exist_ok=True)
        n = 1
        existing = {j.job_id for j in self.jobs}
        while True:
            jid = f"job_{n:04d}"
            if jid not in existing and not (root / jid).exists():
                return jid
            n += 1

    def job_dir(self, job_id: str) -> Path:
        return self.project_dir / "jobs" / job_id

    def job_input_abs(self, job: JobData) -> Path:
        return self.job_dir(job.job_id) / job.input_path

    def job_mask_abs(self, job: JobData) -> Path:
        return self.job_dir(job.job_id) / job.mask_path

    def job_latest_result_abs(self, job: JobData) -> Optional[Path]:
        if not job.latest_result:
            return None
        p = self.job_dir(job.job_id) / job.latest_result
        return p if p.exists() else None

    def save_job(self, job: JobData) -> None:
        d = self.job_dir(job.job_id)
        d.mkdir(parents=True, exist_ok=True)
        write_json_utf8(d / JOB_FILE_NAME, job.to_dict())

    def refresh_job_list(self, select_row: Optional[int] = None) -> None:
        old = self.job_list.currentRow() if select_row is None else int(select_row)
        old = old if 0 <= old < len(self.jobs) else -1
        self._refreshing_job_list = True
        self.job_list.blockSignals(True)
        try:
            self.job_list.clear()
            for job in self.jobs:
                mark = "☑" if job.checked else "☐"
                item = QListWidgetItem(f"{mark} {job.job_id} / {job.status} / {job.name}")
                item.setData(Qt.ItemDataRole.UserRole, job.job_id)
                self.job_list.addItem(item)
            if old >= 0:
                self.job_list.setCurrentRow(old)
            else:
                self.job_list.clearSelection()
        finally:
            self.job_list.blockSignals(False)
            self._refreshing_job_list = False

    def add_images_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "画像を追加", str(Path.home()), "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if paths:
            self.add_image_paths([Path(p) for p in paths])

    def add_image_paths(self, paths: Iterable[Path]) -> None:
        if not self.confirm_unsaved_current_job():
            return
        added = 0
        for src in paths:
            if not src.exists() or src.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                continue
            try:
                rgba = cv2_read_rgba_unicode(src)
                jid = self.next_job_id()
                d = self.job_dir(jid)
                d.mkdir(parents=True, exist_ok=True)
                ext = src.suffix.lower()
                input_name = f"input{ext if ext in SUPPORTED_IMAGE_EXTS else '.png'}"
                # 日本語パスでも確実にコピーできるようPath/shutilを使う。
                shutil.copy2(src, d / input_name)
                h, w = rgba.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2_write_png_unicode(d / MASK_FILE_NAME, mask)
                job = JobData(job_id=jid, name=src.stem, input_path=input_name)
                self.jobs.append(job)
                self.save_job(job)
                added += 1
            except Exception as exc:
                self.log(f"画像追加失敗: {src} / {exc}")
        if added:
            self.write_project_index()
            select_row = max(0, len(self.jobs) - added)
            self.current_job_index = select_row
            self.refresh_job_list(select_row=select_row)
            self.load_job_to_ui(self.jobs[select_row])
            self.log(f"画像追加: {added}件")

    def remove_current_job(self) -> None:
        idx = self.current_job_index
        if idx is None or not (0 <= idx < len(self.jobs)):
            return
        job = self.jobs[idx]
        if QMessageBox.question(self, "削除", f"{job.name} を削除しますか？") != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(self.job_dir(job.job_id), ignore_errors=True)
        except Exception as exc:
            self.log(f"ジョブ削除中のファイル削除失敗: {safe_exception_text(exc)}")
        del self.jobs[idx]
        self.current_job_index = None
        self.clear_dirty_flags()
        self.write_project_index()
        if self.jobs:
            select_row = min(idx, len(self.jobs) - 1)
            self.current_job_index = select_row
            self.refresh_job_list(select_row=select_row)
            self.load_job_to_ui(self.jobs[select_row])
        else:
            self.current_job_index = None
            self.refresh_job_list(select_row=-1)
            self.load_job_to_ui(None)

    def move_job(self, delta: int) -> None:
        idx = self.current_job_index
        if idx is None:
            return
        ni = idx + delta
        if ni < 0 or ni >= len(self.jobs):
            return
        self.jobs[idx], self.jobs[ni] = self.jobs[ni], self.jobs[idx]
        self.current_job_index = ni
        self.write_project_index()
        self.refresh_job_list(select_row=ni)

    def toggle_current_checked(self) -> None:
        job = self.current_job()
        if job is None:
            return
        job.checked = not job.checked
        self.checked_box.blockSignals(True)
        self.checked_box.setChecked(job.checked)
        self.checked_box.blockSignals(False)
        self.mark_current_job_dirty(meta=True)
        self.update_current_job_list_item()

    def current_job(self) -> Optional[JobData]:
        if self.current_job_index is None or not (0 <= self.current_job_index < len(self.jobs)):
            return None
        return self.jobs[self.current_job_index]

    def on_job_selected(self, row: int) -> None:
        if self.loading_ui or self._refreshing_job_list or self._selecting_job:
            return
        old_idx = self.current_job_index
        if old_idx is not None and row != old_idx and not self.confirm_unsaved_current_job():
            self.job_list.blockSignals(True)
            try:
                self.job_list.setCurrentRow(old_idx)
            finally:
                self.job_list.blockSignals(False)
            return
        self._selecting_job = True
        try:
            if row < 0 or row >= len(self.jobs):
                self.current_job_index = None
                self.load_job_to_ui(None)
                return
            self.current_job_index = row
            self.load_job_to_ui(self.jobs[row])
        finally:
            self._selecting_job = False

    def load_job_to_ui(self, job: Optional[JobData]) -> None:
        prev_loading = self.loading_ui
        self.loading_ui = True
        try:
            if job is None:
                self.job_name_edit.setText("")
                self.checked_box.setChecked(False)
                self.prompt_edit.setPlainText("")
                self.negative_edit.setPlainText("")
                self.set_preset_combo_text("")
                self.canvas.set_images(None, None, None)
                self.canvas.clear_crop_rect(emit_signal=False)
                self.result_canvas.set_result_image(None)
                self.update_crop_info_label()
                self.refresh_display_combo(None, preferred_mode="overlay")
                self.refresh_result_combo(None)
                return
            self.job_name_edit.setText(job.name)
            self.checked_box.setChecked(job.checked)
            self.set_params_to_ui(job.params)
            self.set_preset_combo_text(job.preset_name if job.preset_name in self.param_presets else "")
            base = cv2_read_rgba_unicode(self.job_input_abs(job))
            mask_path = self.job_mask_abs(job)
            mask = cv2_read_mask_unicode(mask_path, (base.shape[1], base.shape[0])) if mask_path.exists() else np.zeros(base.shape[:2], dtype=np.uint8)
            preferred_mode = self.current_display_mode()
            self.refresh_display_combo(job, preferred_mode=preferred_mode)
            self.canvas.set_images(base, mask, None, fit=True)
            self.canvas.set_crop_rect(job.crop_rect if job.crop_enabled else None, emit_signal=False)
            self.update_crop_info_label()
            self.canvas.set_mode(self.current_display_mode())
            self.refresh_result_combo(job, select_rel=job.latest_result)
        except Exception as exc:
            self.log(f"ジョブ読込失敗: {safe_exception_text(exc)}")
            print(f"Job load error: {safe_exception_text(exc)}", file=sys.stderr)
        finally:
            self.loading_ui = prev_loading
            self.clear_dirty_flags()

    def params_from_ui(self) -> InpaintParams:
        p = InpaintParams()
        p.prompt = self.prompt_edit.toPlainText()
        p.negative_prompt = self.negative_edit.toPlainText()
        p.sampler_name = self.sampler_edit.text().strip() or "Euler a"
        p.steps = int(self.steps_spin.value())
        p.cfg_scale = float(self.cfg_spin.value())
        p.denoising_strength = float(self.denoise_spin.value())
        p.mask_blur = int(self.mask_blur_spin.value())
        p.inpaint_full_res_padding = int(self.padding_spin.value())
        p.inpainting_fill = int(self.fill_combo.currentData())
        p.inpaint_full_res = bool(self.full_res_check.isChecked())
        p.batch_size = int(self.batch_spin.value())
        p.n_iter = int(self.niter_spin.value())
        p.seed = int(self.seed_spin.value())
        p.clamp()
        return p

    def save_current_job_from_ui(self, refresh_list: bool = True) -> bool:
        if getattr(self, "loading_ui", False):
            return True
        job = self.current_job()
        if job is None:
            self.clear_dirty_flags()
            return True
        try:
            job.name = self.job_name_edit.text().strip() or job.job_id
            job.checked = bool(self.checked_box.isChecked())
            job.params = self.params_from_ui()
            if self.preset_mismatch:
                job.preset_name = ""
                self.set_preset_combo_text("")
            else:
                job.preset_name = self.current_valid_preset_name()
            crop_rect = self.canvas.get_crop_rect()
            job.crop_enabled = crop_rect is not None
            job.crop_rect = crop_rect
            if self.canvas.mask is not None:
                cv2_write_png_unicode(self.job_mask_abs(job), self.canvas.mask)
            self.save_job(job)
            self.clear_dirty_flags()
            if refresh_list:
                self.refresh_job_list(select_row=self.current_job_index if self.current_job_index is not None else -1)
            self.log("現在のジョブを保存")
            return True
        except Exception as exc:
            self.log(f"保存失敗: {safe_exception_text(exc)}")
            return False

    def on_mask_changed(self) -> None:
        if self.current_job() is None or self.canvas.mask is None:
            return
        self._mask_save_pending = False
        self._mask_save_timer.stop()
        self.mark_current_job_dirty(image=True)

    def flush_mask_save(self) -> None:
        self._mask_save_pending = False

    def on_canvas_tool_changed(self, tool: str) -> None:
        self.brush_btn.blockSignals(True); self.eraser_btn.blockSignals(True); self.crop_btn.blockSignals(True)
        self.brush_btn.setChecked(tool == "brush")
        self.eraser_btn.setChecked(tool == "eraser")
        self.crop_btn.setChecked(tool == "crop")
        self.brush_btn.blockSignals(False); self.eraser_btn.blockSignals(False); self.crop_btn.blockSignals(False)

    def update_crop_info_label(self) -> None:
        rect = self.canvas.get_crop_rect()
        if rect is None:
            self.crop_info_label.setText("クロップ: 未設定")
            self.clear_crop_btn.setEnabled(False)
        else:
            x, y, w, h = rect
            self.crop_info_label.setText(f"クロップ: x={x} y={y} w={w} h={h}")
            self.clear_crop_btn.setEnabled(True)

    def on_crop_changed(self) -> None:
        self.update_crop_info_label()
        if self.current_job() is not None:
            self.mark_current_job_dirty(image=True)

    def load_mask_dialog(self) -> None:
        job = self.current_job()
        if job is None or self.canvas.base_rgba is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "マスクを読み込み", str(Path.home()), "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if not path:
            return
        try:
            h, w = self.canvas.base_rgba.shape[:2]
            mask = cv2_read_mask_unicode(Path(path), (w, h))
            self.canvas.mask = mask
            self.canvas._mask_version += 1
            self.canvas.invalidate_cache()
            self.canvas.maskChanged.emit()
            self.canvas.update()
            self.log(f"マスク読込: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "マスク読込失敗", str(exc))

    # ---------- API execution ----------
    def show_api_settings(self) -> None:
        dlg = ApiSettingsDialog(self, self.api_settings, self.log)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.api_settings = dlg.collect()
            self._save_app_settings()
            self.log("API設定を保存")

    def test_api(self) -> None:
        dlg = ApiSettingsDialog(self, self.api_settings, self.log)
        dlg.test_api()
        self.api_settings = dlg.collect()
        self._save_app_settings()

    def build_request_for_job(self, job: JobData) -> Tuple[dict, ApiImagePrep, np.ndarray, np.ndarray]:
        input_path = self.job_input_abs(job)
        mask_path = self.job_mask_abs(job)
        base = cv2_read_rgba_unicode(input_path)
        mask = cv2_read_mask_unicode(mask_path, (base.shape[1], base.shape[0])) if mask_path.exists() else np.zeros(base.shape[:2], dtype=np.uint8)
        prep = prepare_image_pair_for_api(base, mask, job.crop_enabled, job.crop_rect, self.api_max_size)
        base_api = prep.base_api
        mask_api = prep.mask_api
        p = job.params
        crop_active = prep.paste_rect is not None
        # Manual crop sends an already-cropped image. Force request dimensions to the
        # actual API image size.  Do not override inpaint_full_res here; A1111's
        # full-res inpaint is still useful for tiny masked areas inside the crop.
        w = int(base_api.shape[1]) if crop_active else (p.width or int(base_api.shape[1]))
        h = int(base_api.shape[0]) if crop_active else (p.height or int(base_api.shape[0]))
        req = {
            "init_images": [numpy_rgba_to_api_png_base64(base_api)],
            "mask": numpy_mask_to_png_base64(mask_api),
            "prompt": prompt_text_for_api(p.prompt),
            "negative_prompt": prompt_text_for_api(p.negative_prompt),
            "sampler_name": p.sampler_name,
            "steps": p.steps,
            "cfg_scale": p.cfg_scale,
            "denoising_strength": p.denoising_strength,
            "resize_mode": 0,
            "mask_blur": p.mask_blur,
            "inpaint_full_res": p.inpaint_full_res,
            "inpaint_full_res_padding": p.inpaint_full_res_padding,
            "inpainting_fill": p.inpainting_fill,
            "inpainting_mask_invert": p.inpainting_mask_invert,
            "batch_size": p.batch_size,
            "n_iter": p.n_iter,
            "width": w,
            "height": h,
            "seed": p.seed,
            "restore_faces": p.restore_faces,
            "include_init_images": False,
        }
        return req, prep, base, mask

    def save_api_debug_artifacts(self, job: JobData, req: dict, prep: ApiImagePrep, base_full: np.ndarray, mask_full: np.ndarray) -> Path:
        debug_dir = self.job_dir(job.job_id) / DEBUG_DIR_NAME
        debug_dir.mkdir(parents=True, exist_ok=True)
        base_api = np.ascontiguousarray(prep.base_api)
        mask_api = np.ascontiguousarray(prep.mask_api.astype(np.uint8))
        overlay_base = base_api
        if overlay_base.ndim == 3 and overlay_base.shape[2] == 3:
            overlay_base = cv2.cvtColor(overlay_base, cv2.COLOR_RGB2RGBA)
        elif overlay_base.ndim == 2:
            overlay_base = cv2.cvtColor(overlay_base, cv2.COLOR_GRAY2RGBA)
        overlay_api = overlay_rgba(overlay_base, mask_api)
        cv2_write_png_unicode(debug_dir / 'debug_api_input.png', base_api)
        cv2_write_png_unicode(debug_dir / 'debug_api_mask.png', mask_api)
        cv2_write_png_unicode(debug_dir / 'debug_api_overlay.png', overlay_api)
        rect = prep.paste_rect
        saved_rect = normalize_crop_rect_data(job.crop_rect, (int(base_full.shape[1]), int(base_full.shape[0]))) if job.crop_enabled else None
        meta = {
            'job_id': job.job_id,
            'job_name': job.name,
            'input_image_size': {'w': int(base_full.shape[1]), 'h': int(base_full.shape[0])},
            'crop_enabled': bool(job.crop_enabled),
            'saved_crop_rect': {'x': saved_rect[0], 'y': saved_rect[1], 'w': saved_rect[2], 'h': saved_rect[3]} if saved_rect else None,
            'api_crop_rect': {'x': rect[0], 'y': rect[1], 'w': rect[2], 'h': rect[3]} if rect else None,
            'api_input_size': {'w': int(base_api.shape[1]), 'h': int(base_api.shape[0])},
            'api_mask_nonzero': int(np.count_nonzero(mask_api)),
            'paste_canvas_size': {'w': int(prep.paste_canvas_size[0]), 'h': int(prep.paste_canvas_size[1])} if prep.paste_canvas_size else None,
            'request_width': int(req.get('width', 0)),
            'request_height': int(req.get('height', 0)),
            'request_batch_size': int(req.get('batch_size', 1)),
            'request_n_iter': int(req.get('n_iter', 1)),
            'split_execution': self.request_output_count(req) > 1,
            'split_total_outputs': self.request_output_count(req),
            'resize_mode': req.get('resize_mode'),
            'inpaint_full_res': req.get('inpaint_full_res'),
            'inpaint_full_res_padding': req.get('inpaint_full_res_padding'),
            'mask_blur': req.get('mask_blur'),
            'api_max_size': self.api_max_size,
        }
        write_json_utf8(debug_dir / 'debug_request_meta.json', meta)
        return debug_dir

    def split_request_for_single_image(self, req: dict, output_index: int) -> dict:
        single = dict(req)
        single['batch_size'] = 1
        single['n_iter'] = 1
        try:
            seed = int(req.get('seed', -1))
        except Exception:
            seed = -1
        if seed >= 0:
            single['seed'] = seed + max(0, int(output_index))
        return single

    def request_output_count(self, req: dict) -> int:
        try:
            batch_size = max(1, int(req.get('batch_size', 1)))
            n_iter = max(1, int(req.get('n_iter', 1)))
            return max(1, batch_size * n_iter)
        except Exception:
            return 1

    def dry_run_current(self) -> None:
        if not self.confirm_unsaved_current_job():
            return
        job = self.current_job()
        if job is None:
            return
        try:
            req, prep, base_full, mask_full = self.build_request_for_job(job)
            write_json_utf8(self.job_dir(job.job_id) / REQUEST_FILE_NAME, req)
            debug_dir = self.save_api_debug_artifacts(job, req, prep, base_full, mask_full)
            self.log(f"DryRun保存: {self.job_dir(job.job_id) / REQUEST_FILE_NAME}")
            self.log(f"API debug保存: {debug_dir}")
        except Exception as exc:
            self.log(f"DryRun失敗: {exc}")

    def run_current_job(self) -> None:
        if not self.confirm_unsaved_current_job():
            return
        job = self.current_job()
        if job is not None:
            self.start_run_jobs([job])

    def run_checked_jobs(self) -> None:
        if not self.confirm_unsaved_current_job():
            return
        self.start_run_jobs([j for j in self.jobs if j.checked])

    def run_failed_jobs(self) -> None:
        if not self.confirm_unsaved_current_job():
            return
        self.start_run_jobs([j for j in self.jobs if "失敗" in j.status])

    def request_stop(self) -> None:
        self._cancel_requested = True
        try:
            api_post(self.api_settings, "/sdapi/v1/interrupt", {})
            self.log("interrupt送信")
        except Exception as exc:
            self.log(f"interrupt失敗: {exc}")

    def start_run_jobs(self, jobs: List[JobData]) -> None:
        if not jobs:
            self.log("実行対象なし")
            return
        if self._run_thread is not None and self._run_thread.is_alive():
            self.log("実行中です")
            return
        self._cancel_requested = False
        self.set_running_ui(True)
        self._run_thread = threading.Thread(target=self._run_jobs_worker, args=(list(jobs),), daemon=True)
        self._run_thread.start()

    def set_running_ui(self, running: bool) -> None:
        for b in [self.run_current_btn, self.run_checked_btn, self.run_failed_btn, self.dry_btn, self.test_btn, self.api_btn]:
            b.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _run_jobs_worker(self, jobs: List[JobData]) -> None:
        try:
            for index, job in enumerate(jobs, 1):
                if self._cancel_requested:
                    self.ui_log("中断しました")
                    break
                self.ui_progress(f"実行中 {index}/{len(jobs)}: {job.name}")
                self.ui_log(f"実行開始: {job.job_id} / {job.name}")
                try:
                    job.status = "実行中"
                    self.ui_refresh_jobs()
                    req, prep, base_full, mask_full = self.build_request_for_job(job)
                    write_json_utf8(self.job_dir(job.job_id) / REQUEST_FILE_NAME, req)
                    debug_dir = self.save_api_debug_artifacts(job, req, prep, base_full, mask_full)
                    self.ui_log(f"API debug保存: {debug_dir}")
                    crop_rect = prep.paste_rect
                    paste_canvas_size = prep.paste_canvas_size
                    total_outputs = self.request_output_count(req)
                    out_dir = self.job_dir(job.job_id) / RESULT_DIR_NAME
                    out_dir.mkdir(parents=True, exist_ok=True)
                    latest_rel = ""
                    saved_count = 0
                    if total_outputs > 1:
                        self.ui_log(f"分割実行: {total_outputs}枚を1枚ずつAPI送信")
                    for request_index in range(total_outputs):
                        if self._cancel_requested:
                            self.ui_log("中断しました")
                            break
                        single_req = self.split_request_for_single_image(req, request_index)
                        write_json_utf8(debug_dir / f"debug_actual_request_{request_index + 1:03d}.json", single_req)
                        if total_outputs > 1:
                            self.ui_progress(f"実行中 {index}/{len(jobs)}: {job.name} / {request_index + 1}/{total_outputs}")
                        res = api_post(self.api_settings, "/sdapi/v1/img2img", single_req)
                        images = res.get("images", []) if isinstance(res, dict) else []
                        if not images:
                            raise RuntimeError("APIレスポンスにimagesがありません。")
                        for img_b64 in images:
                            saved_count += 1
                            rgba_raw = decode_api_image_to_rgba(img_b64)
                            cv2_write_png_unicode(debug_dir / f"debug_raw_result_{saved_count:03d}.png", rgba_raw)
                            rgba = rgba_raw
                            if crop_rect is not None:
                                rgba = composite_result_on_base(
                                    base_full, rgba, crop_rect, paste_canvas_size,
                                    paste_mask=mask_full, mask_blur=job.params.mask_blur,
                                )
                            out_path = out_dir / f"result_{int(time.time())}_{saved_count:03d}.png"
                            cv2_write_png_unicode(out_path, rgba)
                            latest_rel = str(out_path.relative_to(self.job_dir(job.job_id))).replace("\\", "/")
                    if saved_count <= 0:
                        raise RuntimeError("出力画像が保存されませんでした。")
                    job.latest_result = latest_rel
                    job.status = "完了"
                    self.save_job(job)
                    self.ui_log(f"完了: {job.name} / {saved_count}枚")
                    self.ui_refresh_jobs()
                    if self.current_job() is not None and self.current_job().job_id == job.job_id:
                        self.ui_reload_current_result(job.job_id)
                except Exception as exc:
                    job.status = f"失敗: {exc}"
                    self.save_job(job)
                    self.ui_log(f"失敗: {job.name} / {exc}")
                    self.ui_refresh_jobs()
            self.ui_progress("待機中")
        finally:
            self.runningSignal.emit(False)

    def ui_log(self, text: str) -> None:
        self.logSignal.emit(text)

    def ui_progress(self, text: str) -> None:
        self.progressSignal.emit(text)

    def ui_refresh_jobs(self) -> None:
        self.refreshJobsSignal.emit()

    def ui_reload_current_result(self, job_id: str) -> None:
        self.reloadResultSignal.emit(job_id)

    def reload_current_result_by_id(self, job_id: str) -> None:
        job = self.current_job()
        if job is None or job.job_id != job_id:
            return
        self.refresh_result_combo(job, select_rel=job.latest_result)

    # ---------- image/result display ----------
    def display_item_data(self, index: Optional[int] = None) -> Dict[str, str]:
        combo_index = self.mode_combo.currentIndex() if index is None else int(index)
        data = self.mode_combo.itemData(combo_index) if combo_index >= 0 else None
        if isinstance(data, dict):
            return {"mode": str(data.get("mode", "overlay")), "rel": ""}
        if isinstance(data, str):
            return {"mode": data, "rel": ""}
        return {"mode": "overlay", "rel": ""}

    def current_display_mode(self) -> str:
        mode = self.display_item_data().get("mode", "overlay")
        return mode if mode in {"image", "mask", "overlay"} else "overlay"

    def result_dir(self, job: JobData) -> Path:
        return self.job_dir(job.job_id) / RESULT_DIR_NAME

    def debug_dir(self, job: JobData) -> Path:
        return self.job_dir(job.job_id) / DEBUG_DIR_NAME

    def result_abs(self, job: JobData, rel_path: str) -> Path:
        return self.job_dir(job.job_id) / rel_path

    def result_entries(self, job: Optional[JobData]) -> List[Tuple[str, Path]]:
        if job is None:
            return []
        out_dir = self.result_dir(job)
        if not out_dir.exists():
            return []
        paths = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        paths.sort(key=lambda p: (p.stat().st_mtime, p.name.lower()))
        entries: List[Tuple[str, Path]] = []
        for p in paths:
            rel = str(p.relative_to(self.job_dir(job.job_id))).replace("\\", "/")
            entries.append((rel, p))
        return entries

    def refresh_display_combo(self, job: Optional[JobData], preferred_mode: str = "overlay", select_rel: str = "") -> None:
        del job, select_rel
        preferred_mode = preferred_mode if preferred_mode in {"image", "mask", "overlay"} else "overlay"
        self.mode_combo.blockSignals(True)
        try:
            self.mode_combo.clear()
            static_items = [("画像", "image"), ("マスク", "mask"), ("合成", "overlay")]
            for label, mode in static_items:
                self.mode_combo.addItem(label, {"mode": mode, "rel": ""})
            selected_index = {"image": 0, "mask": 1, "overlay": 2}.get(preferred_mode, 2)
            self.mode_combo.setCurrentIndex(selected_index)
        finally:
            self.mode_combo.blockSignals(False)
        self.on_display_combo_changed(self.mode_combo.currentIndex())

    def on_display_combo_changed(self, _index: int) -> None:
        self.canvas.set_mode(self.current_display_mode())

    def result_item_data(self, index: Optional[int] = None) -> Dict[str, str]:
        combo_index = self.result_combo.currentIndex() if index is None else int(index)
        data = self.result_combo.itemData(combo_index) if combo_index >= 0 else None
        if isinstance(data, dict):
            return {"rel": str(data.get("rel", ""))}
        if isinstance(data, str):
            return {"rel": data}
        return {"rel": ""}

    def refresh_result_combo(self, job: Optional[JobData], select_rel: str = "") -> None:
        entries = self.result_entries(job)
        self.result_combo.blockSignals(True)
        try:
            self.result_combo.clear()
            if not entries:
                self.result_combo.addItem("出力結果なし", {"rel": ""})
                self.result_combo.setCurrentIndex(0)
            else:
                selected_index = len(entries) - 1
                for i, (rel, _path) in enumerate(entries, 1):
                    self.result_combo.addItem(f"出力結果{i:03d}", {"rel": rel})
                    if select_rel and rel == select_rel:
                        selected_index = i - 1
                self.result_combo.setCurrentIndex(selected_index)
        finally:
            self.result_combo.blockSignals(False)
        self.on_result_combo_changed(self.result_combo.currentIndex())

    def on_result_combo_changed(self, _index: int) -> None:
        path = self.selected_result_path()
        if path is None:
            self.result_canvas.set_result_image(None)
            return
        try:
            self.result_canvas.set_result_image(cv2_read_rgba_unicode(path), fit=True)
        except Exception as exc:
            self.result_canvas.set_result_image(None)
            self.log(f"出力結果読込失敗: {safe_exception_text(exc)}")

    def selected_result_path(self) -> Optional[Path]:
        job = self.current_job()
        if job is None:
            return None
        rel = self.result_item_data().get("rel", "")
        if not rel:
            return None
        path = self.result_abs(job, rel)
        return path if path.exists() else None

    def open_selected_result_path(self) -> None:
        job = self.current_job()
        if job is None:
            return
        result_path = self.selected_result_path()
        if result_path is not None:
            open_path_in_explorer(result_path, select_file=True)
            return
        folder = self.result_dir(job)
        folder.mkdir(parents=True, exist_ok=True)
        open_path_in_explorer(folder, select_file=False)

    def copy_selected_result_path(self) -> None:
        path = self.selected_result_path()
        if path is None:
            QMessageBox.information(self, "コピー", "コピーする出力結果がありません。")
            return
        try:
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(path))])
            mime.setText(str(path))
            # Windows Explorer uses this to treat the clipboard item as a file copy,
            # not a move.  QMimeData URLs alone work in many apps, but this makes
            # Explorer paste behavior explicit.
            mime.setData('application/x-qt-windows-mime;value="Preferred DropEffect"', QByteArray(b"\x01\x00\x00\x00"))
            QApplication.clipboard().setMimeData(mime)
            self.log(f"出力結果コピー: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "コピー失敗", safe_exception_text(exc))

    def delete_selected_result_path(self) -> None:
        job = self.current_job()
        path = self.selected_result_path()
        if job is None or path is None:
            QMessageBox.information(self, "削除", "削除する出力結果がありません。")
            return
        rel = self.result_item_data().get("rel", "")
        reply = QMessageBox.question(
            self,
            "削除確認",
            f"この出力結果をゴミ箱へ移動しますか？\n\n{path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        old_index = self.result_combo.currentIndex()
        try:
            if not move_path_to_trash(path):
                raise RuntimeError("ゴミ箱への移動に失敗しました。")
            entries_after = self.result_entries(job)
            select_rel = ""
            if entries_after:
                select_index = clamp_int(old_index, 0, len(entries_after) - 1)
                select_rel = entries_after[select_index][0]
            latest_path = self.job_latest_result_abs(job)
            if job.latest_result == rel or latest_path is None or not latest_path.exists():
                job.latest_result = entries_after[-1][0] if entries_after else ""
                self.save_job(job)
            self.refresh_result_combo(job, select_rel=select_rel or job.latest_result)
            self.log(f"出力結果削除: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "削除失敗", safe_exception_text(exc))

    def result_combo_indexes(self) -> List[int]:
        indexes: List[int] = []
        for i in range(self.result_combo.count()):
            if self.result_item_data(i).get("rel"):
                indexes.append(i)
        return indexes

    def navigate_result_selection(self, delta: int) -> None:
        if getattr(self, "result_tab", None) is not None and self.tabs.currentWidget() is not self.result_tab:
            return
        indexes = self.result_combo_indexes()
        if not indexes:
            return
        current = self.result_combo.currentIndex()
        if current in indexes:
            pos = indexes.index(current)
            next_pos = clamp_int(pos + int(delta), 0, len(indexes) - 1)
        else:
            next_pos = 0 if delta > 0 else len(indexes) - 1
        self.result_combo.setCurrentIndex(indexes[next_pos])

    # ---------- window/system events ----------
    def log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_edit.append(f"[{ts}] {text}")
        self.log_edit.moveCursor(QTextCursor.MoveOperation.End)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        paths = file_urls_to_paths(event.mimeData())
        if any(p.suffix.lower() in SUPPORTED_IMAGE_EXTS for p in paths):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = [p for p in file_urls_to_paths(event.mimeData()) if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        if paths:
            self.add_image_paths(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if not self.isMaximized() and not self.isMinimized():
            self._last_normal_geometry = QRect(self.geometry())
        if not getattr(self, "loading_ui", False):
            self._save_app_settings()

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        if not self.isMaximized() and not self.isMinimized():
            self._last_normal_geometry = QRect(self.geometry())
        if not getattr(self, "loading_ui", False):
            self._save_app_settings()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._closing_app = True
        try:
            if not self.confirm_unsaved_current_job():
                event.ignore()
                self._closing_app = False
                return
            self.write_project_index()
            self._save_app_settings()
        except Exception as exc:
            print(f"Close error: {safe_exception_text(exc)}", file=sys.stderr)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
