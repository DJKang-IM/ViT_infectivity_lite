# -*- coding: utf-8 -*-
"""DICOM preprocessing for Infectivity_ViT v1 (256x256 + CLAHE)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def pixel_array_to_hu(ds: pydicom.Dataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def window_lo_hi(ds: pydicom.Dataset, hu: np.ndarray) -> tuple[float, float]:
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
    else:
        lo, hi = [float(x) for x in np.percentile(hu, [0.5, 99.5])]
    return lo, hi


def hu_to_float01(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.clip(hu.astype(np.float32, copy=False), lo, hi)
    arr = (arr - lo) / max(hi - lo, 1e-6)
    return arr.astype(np.float32, copy=False)


def apply_clahe_float01(
    arr: np.ndarray,
    *,
    clip_limit: float = 0.03,
    kernel_size: int | tuple[int, int] | None = None,
) -> np.ndarray:
    from skimage import exposure

    if arr.ndim != 2:
        raise ValueError(f"CLAHE expects HxW grayscale, got shape {arr.shape}")
    a = np.clip(arr.astype(np.float64, copy=False), 0.0, 1.0)
    ks = kernel_size
    if isinstance(ks, int):
        ks = (ks, ks)
    out = exposure.equalize_adapthist(a, kernel_size=ks, clip_limit=float(clip_limit), nbins=256)
    return out.astype(np.float32, copy=False)


def dicom_to_float01(
    dcm_path: Path,
    *,
    clahe: bool = True,
    clahe_clip_limit: float = 0.03,
) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    hu = pixel_array_to_hu(ds)
    lo, hi = window_lo_hi(ds, hu)
    arr = hu_to_float01(hu, lo, hi)
    if clahe:
        arr = apply_clahe_float01(arr, clip_limit=clahe_clip_limit)
    return arr


def _resize_gray01(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape
    if max(h, w) <= max_side:
        return arr
    scale = max_side / float(max(h, w))
    t = torch.from_numpy(arr)[None]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return TF.resize(t, [nh, nw], antialias=True)[0].numpy().astype(np.float32)


def jpeg_to_float01(
    jpeg_path: Path,
    *,
    clahe: bool = True,
    clahe_clip_limit: float = 0.03,
    brightness_norm: bool = True,
    resize_max_side: int | None = None,
) -> np.ndarray:
    """Load a JPEG CXR -> grayscale float [0,1] HxW, optional CLAHE.

    brightness_norm rescales per-image to full [0,1] range (min-max) to
    harmonise scanner/export brightness. When resize_max_side is set the image
    is downscaled BEFORE CLAHE (equalize_adapthist cost scales with pixels).
    """
    from PIL import Image

    with Image.open(jpeg_path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    if brightness_norm:
        lo = float(arr.min())
        hi = float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1e-6)
    if resize_max_side is not None:
        arr = _resize_gray01(arr, int(resize_max_side))
    if clahe:
        arr = apply_clahe_float01(arr, clip_limit=clahe_clip_limit)
    return arr.astype(np.float32, copy=False)


def load_tensor_from_jpeg(
    jpeg_path: Path,
    transform: T.Compose,
    *,
    clahe: bool = True,
    clahe_clip_limit: float = 0.03,
    brightness_norm: bool = True,
) -> torch.Tensor:
    arr = jpeg_to_float01(
        jpeg_path,
        clahe=clahe,
        clahe_clip_limit=clahe_clip_limit,
        brightness_norm=brightness_norm,
    )
    return transform(arr)


def letterbox_chw(x: torch.Tensor, *, target: int, pad_value: float = 0.0) -> torch.Tensor:
    c, h, w = x.shape
    t = int(target)
    scale = t / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    x2 = TF.resize(x, [nh, nw], antialias=True)
    pad_h = t - nh
    pad_w = t - nw
    pl, pt = pad_w // 2, pad_h // 2
    x3 = TF.pad(x2, [pl, pt, pad_w - pl, pad_h - pt], fill=float(pad_value))
    if x3.shape[-2:] != (t, t):
        x3 = TF.resize(x3, [t, t], antialias=True)
    return x3


class _Letterbox:
    """Picklable letterbox resize (Windows DataLoader workers need this)."""

    def __init__(self, target: int) -> None:
        self.target = int(target)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return letterbox_chw(x, target=self.target)


class _ToRGB:
    """Picklable grayscale->3ch repeat."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1)


def build_transform(
    image_size: int,
    *,
    resize_mode: str = "stretch",
    use_imagenet_norm: bool = True,
) -> T.Compose:
    steps: list = [T.ToTensor()]
    if resize_mode == "stretch":
        steps.append(T.Resize((image_size, image_size), antialias=True))
    elif resize_mode == "letterbox":
        steps.append(_Letterbox(image_size))
    else:
        raise ValueError(f"Unknown resize_mode: {resize_mode}")
    steps.append(_ToRGB())
    if use_imagenet_norm:
        steps.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(steps)


def load_tensor_from_dicom(
    dcm_path: Path,
    transform: T.Compose,
    *,
    clahe: bool = True,
    clahe_clip_limit: float = 0.03,
) -> torch.Tensor:
    arr = dicom_to_float01(dcm_path, clahe=clahe, clahe_clip_limit=clahe_clip_limit)
    return transform(arr)
