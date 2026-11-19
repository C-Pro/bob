"""Phase 2 tests: config parsing, strict inline image decoding, and model loader validation."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import pytest
from PIL import Image
import torch

from bob_s1.config import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MODEL_ID,
    MODEL_MAX_POSITION_EMBEDDINGS,
    SidecarConfig,
)
from bob_s1.images import ImageValidationError, configure_pillow_limits, decode_data_uri_image
from bob_s1.loader import (
    CheckpointOverlayLoader,
    ModelLoadError,
    import_jev_model_module,
    sha256_file,
    validate_checkpoint_payload,
    verify_model_runtime,
)


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (8, 8), fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format=fmt)
    return buf.getvalue()


def _png_data_uri(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (8, 8), fmt: str = "PNG") -> str:
    subtype = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{subtype};base64," + base64.b64encode(_png_bytes(color=color, size=size, fmt=fmt)).decode("ascii")


def test_config_elastic_max_seq_len_and_validation():
    cfg = SidecarConfig.from_env({})
    assert cfg.host == "127.0.0.1"
    assert cfg.max_seq_len == DEFAULT_MAX_SEQ_LEN
    assert cfg.attn_implementation == "eager"

    cfg_8k = SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": "8192", "BOB_S1_ATTN_IMPLEMENTATION": "sdpa"})
    assert cfg_8k.max_seq_len == 8192
    assert cfg_8k.attn_implementation == "sdpa"

    cfg_max = SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": str(MODEL_MAX_POSITION_EMBEDDINGS)})
    assert cfg_max.max_seq_len == MODEL_MAX_POSITION_EMBEDDINGS

    with pytest.raises(ValueError, match="BOB_S1_MAX_SEQ_LEN"):
        SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": "128"})
    with pytest.raises(ValueError, match="BOB_S1_MAX_SEQ_LEN"):
        SidecarConfig.from_env({"BOB_S1_MAX_SEQ_LEN": str(MODEL_MAX_POSITION_EMBEDDINGS + 1)})
    with pytest.raises(ValueError, match="BOB_S1_ATTN_IMPLEMENTATION"):
        SidecarConfig.from_env({"BOB_S1_ATTN_IMPLEMENTATION": "flash_attention_99"})


def test_image_decoder_accepts_valid_formats_and_is_thread_safe():
    configure_pillow_limits(4096 * 4096)
    for fmt in ("PNG", "JPEG", "WEBP"):
        uri = _png_data_uri(fmt=fmt)
        img = decode_data_uri_image(uri)
        assert img.mode == "RGB"
        assert img.size == (8, 8)

    # Concurrent decode calls do not mutate global Pillow limits
    before = Image.MAX_IMAGE_PIXELS
    uris = [_png_data_uri(color=(i * 10, 0, 255 - i * 10), fmt="PNG") for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda u: decode_data_uri_image(u, max_pixels=1024), uris))
    assert len(results) == 8
    assert all(img.size == (8, 8) for img in results)
    assert Image.MAX_IMAGE_PIXELS == before


def test_image_decoder_rejects_unsafe_uris_and_corrupt_crc(tmp_path):
    local_img = tmp_path / "secret.png"
    Image.new("RGB", (4, 4)).save(local_img)

    bad_urls = [
        "https://example.com/test.png",
        "http://127.0.0.1:8080/test.png",
        f"file://{local_img}",
        str(local_img),
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "data:image/jpg;base64,AAAA",
        "data:image/png;base64,not-valid-base64!!!",
    ]
    for url in bad_urls:
        with pytest.raises(ImageValidationError):
            decode_data_uri_image(url)

    # Valid PNG header with corrupted IDAT CRC (Pillow verify() raises SyntaxError)
    raw = bytearray(_png_bytes(size=(4, 4), fmt="PNG"))
    idat_idx = raw.find(b"IDAT")
    assert idat_idx > 0
    raw[idat_idx + 5] ^= 0xFF
    corrupt_uri = "data:image/png;base64," + base64.b64encode(bytes(raw)).decode("ascii")
    with pytest.raises(ImageValidationError, match="Failed to decode image payload"):
        decode_data_uri_image(corrupt_uri)


def test_image_decoder_rejects_mime_mismatch_oversize_and_animated():
    # PNG bytes wrapped with JPEG header
    png_b64 = _png_data_uri(fmt="PNG").split(",", 1)[1]
    with pytest.raises(ImageValidationError, match="does not match decoded format"):
        decode_data_uri_image(f"data:image/jpeg;base64,{png_b64}")

    # Byte-size limit
    with pytest.raises(ImageValidationError, match="max_image_bytes"):
        decode_data_uri_image(_png_data_uri(fmt="PNG"), max_bytes=16)

    # Pixel-count limit (without mutating global MAX_IMAGE_PIXELS)
    with pytest.raises(ImageValidationError, match="max_image_pixels"):
        decode_data_uri_image(_png_data_uri(size=(16, 16), fmt="PNG"), max_pixels=100)

    # Animated GIF rejection
    buf = BytesIO()
    frames = [Image.new("RGB", (4, 4), (255, 0, 0)), Image.new("RGB", (4, 4), (0, 0, 255))]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], loop=0)
    gif_uri = "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    with pytest.raises(ImageValidationError, match="Animated"):
        decode_data_uri_image(gif_uri)


def test_loader_validation_and_preprocessing_guards():
    repo_root = Path(__file__).resolve().parents[3]
    jev_model = import_jev_model_module(repo_root)
    model = jev_model.synthetic_model(seed=42, with_lora=True, include_vision=True)

    meta = verify_model_runtime(model, device="cpu", max_seq_len=512)
    assert meta["min_pixels"] == jev_model.DEFAULT_IMAGE_MIN_PIXELS
    assert meta["max_pixels"] == jev_model.DEFAULT_IMAGE_MAX_PIXELS
    assert model.pointer.q.weight.dtype == torch.float32
    assert model.W_emb.weight.dtype == torch.float32

    state = model.state_dict()
    valid_tensors = {
        k: v.clone()
        for k, v in state.items()
        if ("lora_" in k or k.startswith(("pointer.", "W_emb.")))
    }
    valid_vis = {
        "min_pixels": jev_model.DEFAULT_IMAGE_MIN_PIXELS,
        "max_pixels": jev_model.DEFAULT_IMAGE_MAX_PIXELS,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    payload = {
        "special_ids": dict(model.special_ids),
        "tensors": valid_tensors,
        "metadata": {"include_vision": True, "vision_preprocess": valid_vis},
    }
    synth_lora_count = sum(1 for k in valid_tensors if "lora_" in k)

    with pytest.raises(ModelLoadError, match="LoRA tensor count mismatch"):
        validate_checkpoint_payload(payload, model)

    validate_checkpoint_payload(payload, model, expected_lora_count=synth_lora_count)

    # Missing vision_preprocess metadata
    no_vis_prep = {**payload, "metadata": {"include_vision": True}}
    with pytest.raises(ModelLoadError, match="vision_preprocess"):
        validate_checkpoint_payload(no_vis_prep, model, expected_lora_count=synth_lora_count)

    # Invalid std <= 0 and non-finite mean
    neg_std = {**payload, "metadata": {"include_vision": True, "vision_preprocess": {**valid_vis, "std": [0.5, -0.5, 0.5]}}}
    with pytest.raises(ModelLoadError, match="must be > 0"):
        validate_checkpoint_payload(neg_std, model, expected_lora_count=synth_lora_count)

    nan_mean = {**payload, "metadata": {"include_vision": True, "vision_preprocess": {**valid_vis, "mean": [0.5, float("inf"), 0.5]}}}
    with pytest.raises(ModelLoadError, match="Non-finite"):
        validate_checkpoint_payload(nan_mean, model, expected_lora_count=synth_lora_count)

    # Wrong pixel bounds
    bad_px = {**payload, "metadata": {"include_vision": True, "vision_preprocess": {**valid_vis, "max_pixels": 65536}}}
    with pytest.raises(ModelLoadError, match="pixel bounds"):
        validate_checkpoint_payload(bad_px, model, expected_lora_count=synth_lora_count)

    # Missing W_emb
    no_emb = {**payload, "tensors": {k: v for k, v in valid_tensors.items() if k != "W_emb.weight"}}
    with pytest.raises(ModelLoadError, match="missing required head tensor"):
        validate_checkpoint_payload(no_emb, model, expected_lora_count=synth_lora_count)

    # Non-finite tensor
    nan_tensors = dict(valid_tensors)
    nan_tensors["W_emb.weight"] = torch.full_like(valid_tensors["W_emb.weight"], float("nan"))
    with pytest.raises(ModelLoadError, match="Non-finite"):
        validate_checkpoint_payload({**payload, "tensors": nan_tensors}, model, expected_lora_count=synth_lora_count)

    # Wrong special_ids
    bad_special = {**payload, "special_ids": {"broken": 0}}
    with pytest.raises(ModelLoadError, match="boundary token IDs"):
        validate_checkpoint_payload(bad_special, model, expected_lora_count=synth_lora_count)


def test_checkpoint_overlay_loader_integration_with_stubbed_pretrained(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    jev_model = import_jev_model_module(repo_root)

    # Build a synthetic checkpoint saved to disk via save_checkpoint()
    source_model = jev_model.synthetic_model(seed=101, with_lora=True, include_vision=True)
    with torch.no_grad():
        source_model.W_emb.weight.fill_(0.25)
    ckpt_dir = tmp_path / "ckpt"
    source_model.save_checkpoint(ckpt_dir, {"source": "unit_test"})
    ckpt_path = ckpt_dir / "model.pt"
    actual_sha = sha256_file(ckpt_path)

    # Stub DualHeadDecisionEmbedder.from_pretrained to return a fresh synthetic model with different weights
    def fake_from_pretrained(**kwargs):
        assert kwargs["include_vision"] is True
        assert kwargs["attn_implementation"] == "sdpa"
        fresh = jev_model.synthetic_model(seed=7, with_lora=True, include_vision=True)
        with torch.no_grad():
            fresh.W_emb.weight.zero_()
        return fresh

    monkeypatch.setattr(jev_model.DualHeadDecisionEmbedder, "from_pretrained", staticmethod(fake_from_pretrained))
    synth_lora_count = sum(1 for k in source_model.state_dict() if "lora_" in k)

    # Custom checkpoint SHA produces explicit custom checkpoint_id (not release "bob-s1-0.8b")
    cfg = SidecarConfig.from_env(
        {
            "BOB_S1_CHECKPOINT": str(ckpt_path),
            "BOB_S1_EXPECTED_SHA256": actual_sha,
            "BOB_S1_DEVICE": "cpu",
            "BOB_S1_ATTN_IMPLEMENTATION": "sdpa",
            "BOB_S1_MAX_SEQ_LEN": "512",
        }
    )
    bundle = CheckpointOverlayLoader(cfg, expected_lora_count=synth_lora_count).load()
    assert bundle.warmup_ok is True
    assert bundle.checkpoint_sha256 == actual_sha
    assert bundle.checkpoint_id == f"custom:{actual_sha[:12]}"
    assert bundle.checkpoint_id != DEFAULT_MODEL_ID
    assert torch.allclose(bundle.model.W_emb.weight, torch.full_like(bundle.model.W_emb.weight, 0.25))

    # When SHA matches release digest, checkpoint_id resolves to DEFAULT_MODEL_ID
    monkeypatch.setattr("bob_s1.loader.DEFAULT_CHECKPOINT_SHA256", actual_sha)
    bundle_release = CheckpointOverlayLoader(cfg, expected_lora_count=synth_lora_count).load()
    assert bundle_release.checkpoint_id == DEFAULT_MODEL_ID


def test_checkpoint_overlay_loader_fails_on_missing_or_sha_mismatch(tmp_path):
    missing = tmp_path / "missing.pt"
    cfg = SidecarConfig.from_env({"BOB_S1_CHECKPOINT": str(missing), "BOB_S1_DEVICE": "cpu"})
    with pytest.raises(ModelLoadError, match="does not exist"):
        CheckpointOverlayLoader(cfg).load()

    bad_ckpt = tmp_path / "fake.pt"
    bad_ckpt.write_bytes(b"not-a-real-checkpoint")
    cfg_sha = SidecarConfig.from_env(
        {
            "BOB_S1_CHECKPOINT": str(bad_ckpt),
            "BOB_S1_EXPECTED_SHA256": "deadbeef" * 8,
            "BOB_S1_DEVICE": "cpu",
        }
    )
    with pytest.raises(ModelLoadError, match="SHA-256 mismatch"):
        CheckpointOverlayLoader(cfg_sha).load()
