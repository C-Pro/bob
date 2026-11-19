"""Strict inline base64 image URI decoder and sanitizer."""
from __future__ import annotations

import base64
import binascii
from io import BytesIO
import re

from PIL import Image, UnidentifiedImageError

ALLOWED_IMAGE_FORMATS = frozenset({"png", "jpeg", "webp", "gif"})
_DATA_URI_RE = re.compile(r"^data:image/(png|jpeg|webp|gif);base64,([A-Za-z0-9+/=]+)$")


class ImageValidationError(ValueError):
    """Raised when an inline image URI fails format, size, or safety checks."""

    def __init__(self, message: str, *, code: str = "invalid_image", param: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param


def configure_pillow_limits(max_pixels: int) -> None:
    """Configure process-wide Pillow decompression-bomb limit once at startup."""
    if max_pixels <= 0:
        raise ValueError(f"max_pixels must be positive, got {max_pixels}")
    Image.MAX_IMAGE_PIXELS = int(max_pixels)


def _decode_one_with_pixels(
    url: str,
    *,
    max_bytes: int,
    max_pixels: int,
    remaining_total_pixels: int | None = None,
    max_total_pixels: int | None = None,
    param: str | None = None,
) -> tuple[Image.Image, int]:
    if not isinstance(url, str) or not url:
        raise ImageValidationError("Image URL must be a non-empty string", param=param)

    match = _DATA_URI_RE.match(url)
    if match is None:
        raise ImageValidationError(
            "Only inline data:image/(png|jpeg|webp|gif);base64,... URIs are accepted",
            param=param,
        )

    declared_subtype = match.group(1)
    b64_payload = match.group(2)

    estimated_bytes = (len(b64_payload) * 3) // 4
    if estimated_bytes > max_bytes + 3:
        raise ImageValidationError(
            f"Encoded image exceeds max_image_bytes ({max_bytes})",
            code="image_too_large",
            param=param,
        )

    try:
        raw_bytes = base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageValidationError(f"Invalid base64 image payload: {exc}", param=param) from exc

    if not raw_bytes:
        raise ImageValidationError("Decoded image payload is empty", param=param)
    if len(raw_bytes) > max_bytes:
        raise ImageValidationError(
            f"Decoded image size {len(raw_bytes)} exceeds max_image_bytes ({max_bytes})",
            code="image_too_large",
            param=param,
        )

    # First pass: inspect header dimensions & CRC before decoding full pixel buffers.
    try:
        with Image.open(BytesIO(raw_bytes)) as probe:
            fmt = (probe.format or "").lower()
            if fmt not in ALLOWED_IMAGE_FORMATS:
                raise ImageValidationError(
                    f"Unsupported decoded image format {fmt!r}; expected {sorted(ALLOWED_IMAGE_FORMATS)}",
                    param=param,
                )
            if fmt != declared_subtype:
                raise ImageValidationError(
                    f"Declared MIME image/{declared_subtype} does not match decoded format {fmt!r}",
                    param=param,
                )
            if getattr(probe, "is_animated", False) or getattr(probe, "n_frames", 1) != 1:
                raise ImageValidationError("Animated or multi-frame images are not supported", param=param)
            width, height = probe.size
            if width <= 0 or height <= 0:
                raise ImageValidationError(f"Invalid image dimensions {width}x{height}", param=param)
            pixel_count = width * height
            if pixel_count > max_pixels:
                raise ImageValidationError(
                    f"Image pixel count {pixel_count} exceeds max_image_pixels ({max_pixels})",
                    code="image_too_large",
                    param=param,
                )
            if remaining_total_pixels is not None and pixel_count > remaining_total_pixels:
                limit = max_total_pixels if max_total_pixels is not None else remaining_total_pixels
                raise ImageValidationError(
                    f"Aggregate decoded image pixels exceed request budget ({limit})",
                    code="image_too_large",
                    param=param,
                )
            probe.verify()

        with Image.open(BytesIO(raw_bytes)) as img:
            img.load()
            return img.convert("RGB"), pixel_count
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(
            f"Image exceeds max_image_pixels ({max_pixels}) safety limit: {exc}",
            code="image_too_large",
            param=param,
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"Failed to decode image payload: {exc}", param=param) from exc


def decode_data_uri_image(
    url: str,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    max_pixels: int = 4096 * 4096,
    param: str | None = None,
) -> Image.Image:
    """Decode and validate an inline data:image/...;base64 URI into an RGB PIL Image."""
    img, _ = _decode_one_with_pixels(
        url,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        param=param,
    )
    return img


def decode_data_uri_images_batch(
    items: list[tuple[str, str]],
    *,
    max_bytes: int,
    max_pixels: int,
    max_total_pixels: int,
) -> list[Image.Image]:
    """Decode a list of (url, param_path) pairs while enforcing an aggregate pixel budget."""
    remaining = max_total_pixels
    decoded: list[Image.Image] = []
    for url, param in items:
        img, used = _decode_one_with_pixels(
            url,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
            remaining_total_pixels=remaining,
            max_total_pixels=max_total_pixels,
            param=param,
        )
        remaining -= used
        decoded.append(img)
    return decoded
