"""Model loading, checkpoint verification, and startup warmup for bob-s1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys
from typing import Any, Protocol

from PIL import Image
import torch

from .config import DEFAULT_CHECKPOINT_SHA256, DEFAULT_MODEL_ID, SidecarConfig

EXPECTED_LORA_TENSOR_COUNT = 372
EXPECTED_MIN_PIXELS = 16384
EXPECTED_MAX_PIXELS = 98304
REQUIRED_HEAD_TENSORS = (
    "W_emb.weight",
    "pointer.q.weight",
    "pointer.q.bias",
    "pointer.k.weight",
    "pointer.k.bias",
)


class ModelLoadError(RuntimeError):
    """Raised when model artifacts, checkpoints, or warmup checks fail."""


@dataclass(frozen=True)
class LoadedModelBundle:
    model: Any
    model_id: str
    checkpoint_id: str
    checkpoint_sha256: str | None
    device: str
    dtype: str
    attn_implementation: str
    max_seq_len: int
    preprocess_metadata: dict[str, Any]
    warmup_ok: bool


class ModelLoader(Protocol):
    """Loader boundary allowing checkpoint_overlay today and merged_hf later."""

    def load(self) -> LoadedModelBundle:
        ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_jev_model_module(repo_root: Path):
    """Import eval.jev.model deterministically relative to repo_root without training imports."""
    root_str = str(repo_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("eval.jev.model")


def resolve_device_and_dtype(cfg: SidecarConfig) -> tuple[str, torch.dtype, str]:
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg.device

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelLoadError(f"Configured device {device!r} is unavailable (torch.cuda.is_available() is False)")

    if cfg.dtype == "auto":
        dtype_name = "bf16" if device.startswith("cuda") else "fp32"
    else:
        dtype_name = cfg.dtype

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return device, dtype_map[dtype_name], dtype_name


def _validate_preprocess_metadata(vis_prep: Any) -> dict[str, Any]:
    if not isinstance(vis_prep, dict):
        raise ModelLoadError("Checkpoint metadata missing required vision_preprocess mapping")

    raw_min = vis_prep.get("min_pixels")
    raw_max = vis_prep.get("max_pixels")
    if isinstance(raw_min, bool) or not isinstance(raw_min, int) or isinstance(raw_max, bool) or not isinstance(raw_max, int):
        raise ModelLoadError("vision_preprocess min_pixels and max_pixels must be integers")
    if raw_min != EXPECTED_MIN_PIXELS or raw_max != EXPECTED_MAX_PIXELS:
        raise ModelLoadError(
            f"Unsupported vision_preprocess pixel bounds ({raw_min}, {raw_max}); "
            f"expected ({EXPECTED_MIN_PIXELS}, {EXPECTED_MAX_PIXELS})"
        )

    validated: dict[str, Any] = {"min_pixels": raw_min, "max_pixels": raw_max}
    for key, require_positive in (("mean", False), ("std", True)):
        vals = vis_prep.get(key)
        if not isinstance(vals, (list, tuple)) or len(vals) != 3:
            raise ModelLoadError(f"Invalid vision_preprocess {key}: expected 3-element sequence")
        parsed: list[float] = []
        for idx, item in enumerate(vals):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ModelLoadError(f"Invalid vision_preprocess {key}[{idx}]: expected finite number")
            fval = float(item)
            if not math.isfinite(fval):
                raise ModelLoadError(f"Non-finite vision_preprocess {key}[{idx}]: {item!r}")
            if require_positive and fval <= 0.0:
                raise ModelLoadError(f"vision_preprocess {key}[{idx}] must be > 0, got {fval}")
            parsed.append(fval)
        validated[key] = parsed
    return validated


def validate_checkpoint_payload(
    payload: dict[str, Any],
    model: Any,
    *,
    expected_lora_count: int = EXPECTED_LORA_TENSOR_COUNT,
) -> dict[str, Any]:
    """Verify checkpoint special tokens, required tensors, shapes, finiteness, and vision metadata."""
    if not isinstance(payload, dict):
        raise ModelLoadError("Checkpoint payload must be a dict")
    if payload.get("special_ids") != model.special_ids:
        raise ModelLoadError("Checkpoint boundary token IDs do not match tokenizer")

    tensors = payload.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ModelLoadError("Checkpoint payload contains no tensors")

    for req_key in REQUIRED_HEAD_TENSORS:
        if req_key not in tensors:
            raise ModelLoadError(f"Checkpoint missing required head tensor: {req_key}")

    lora_keys = [key for key in tensors if "lora_" in key]
    if len(lora_keys) != expected_lora_count:
        raise ModelLoadError(
            f"Checkpoint LoRA tensor count mismatch: expected {expected_lora_count}, got {len(lora_keys)}"
        )

    model_state = model.state_dict()
    for name, tensor in tensors.items():
        if name not in model_state:
            raise ModelLoadError(f"Unexpected checkpoint tensor: {name}")
        if tuple(tensor.shape) != tuple(model_state[name].shape):
            raise ModelLoadError(
                f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} vs model {tuple(model_state[name].shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ModelLoadError(f"Non-finite values detected in checkpoint tensor: {name}")

    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        raise ModelLoadError("Checkpoint metadata must be a mapping")
    if meta.get("include_vision") is not True:
        raise ModelLoadError("Checkpoint metadata must declare include_vision=True for multimodal serving")
    _validate_preprocess_metadata(meta.get("vision_preprocess"))
    return meta


def verify_model_runtime(model: Any, *, device: str, max_seq_len: int) -> dict[str, Any]:
    """Ensure eval/freeze invariants, FP32 heads, finite parameters, and run startup warmup."""
    model.eval()
    if model.visual is None:
        raise ModelLoadError("Loaded model has no attached vision encoder")
    model.visual.eval()
    for param in model.visual.parameters():
        param.requires_grad_(False)

    model.pointer.to(device=device, dtype=torch.float32)
    model.W_emb.to(device=device, dtype=torch.float32)

    for head_name in ("pointer", "W_emb"):
        head = getattr(model, head_name)
        for p_name, param in head.named_parameters():
            if param.dtype != torch.float32:
                raise ModelLoadError(f"{head_name}.{p_name} must remain float32, got {param.dtype}")
            if not torch.isfinite(param).all():
                raise ModelLoadError(f"{head_name}.{p_name} contains non-finite values")

    prep = _validate_preprocess_metadata(
        {
            "min_pixels": int(model.image_min_pixels),
            "max_pixels": int(model.image_max_pixels),
            "mean": list(model.image_mean),
            "std": list(model.image_std),
        }
    )

    warmup_budget = min(256, max_seq_len)
    warmup_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    warmup_opts = [("no", "No"), ("yes", "Yes")]
    with torch.inference_mode():
        prefix = model.encode_prefix("warmup", max_seq_len=warmup_budget, suffix_reserve=64, images=[warmup_img])
        for dim, vec in prefix.embeddings.items():
            if not torch.isfinite(vec).all():
                raise ModelLoadError(f"Warmup embedding slice {dim}d produced non-finite values")
        out = model.question_from_prefix(prefix, "Ready?", warmup_opts, isolate=True)
        if not torch.isfinite(out["logits"]).all():
            raise ModelLoadError("Warmup decision logits produced non-finite values")

    return {
        "min_pixels": prep["min_pixels"],
        "max_pixels": prep["max_pixels"],
        "image_mean": prep["mean"],
        "image_std": prep["std"],
    }


class CheckpointOverlayLoader:
    """Loads pinned Qwen3.5-0.8B-Base + kev-0.8b donor and overlays bob-s1 model.pt."""

    def __init__(self, cfg: SidecarConfig, *, expected_lora_count: int = EXPECTED_LORA_TENSOR_COUNT):
        self.cfg = cfg
        self.expected_lora_count = expected_lora_count

    def load(self) -> LoadedModelBundle:
        ckpt_path = self.cfg.checkpoint_path
        if not ckpt_path.is_file():
            raise ModelLoadError(f"Checkpoint file does not exist: {ckpt_path}")

        actual_sha = sha256_file(ckpt_path)
        if self.cfg.expected_checkpoint_sha256 and actual_sha != self.cfg.expected_checkpoint_sha256:
            raise ModelLoadError(
                f"Checkpoint SHA-256 mismatch: expected {self.cfg.expected_checkpoint_sha256}, got {actual_sha}"
            )

        if self.cfg.cpu_threads is not None:
            torch.set_num_threads(self.cfg.cpu_threads)

        device, torch_dtype, dtype_name = resolve_device_and_dtype(self.cfg)
        jev_model = import_jev_model_module(self.cfg.repo_root)

        cache_dir = str(self.cfg.hf_cache_dir) if self.cfg.hf_cache_dir is not None else None
        model = jev_model.DualHeadDecisionEmbedder.from_pretrained(
            device="cpu",
            dtype=torch_dtype,
            trainable_adapter=False,
            include_vision=True,
            cache_dir=cache_dir,
            attn_implementation=self.cfg.attn_implementation,
        )

        # Checkpoint SHA-256 was verified above and weights_only=True blocks arbitrary pickle code execution.
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        validate_checkpoint_payload(payload, model, expected_lora_count=self.expected_lora_count)
        model.load_checkpoint(ckpt_path)
        if self.cfg.merge_lora and hasattr(model.backbone, "merge_and_unload"):
            model.backbone = model.backbone.merge_and_unload()
        model.to(device=device)

        preprocess_meta = verify_model_runtime(model, device=device, max_seq_len=self.cfg.max_seq_len)
        checkpoint_id = DEFAULT_MODEL_ID if actual_sha == DEFAULT_CHECKPOINT_SHA256 else f"custom:{actual_sha[:12]}"
        return LoadedModelBundle(
            model=model,
            model_id=self.cfg.model_id,
            checkpoint_id=checkpoint_id,
            checkpoint_sha256=actual_sha,
            device=device,
            dtype=dtype_name,
            attn_implementation=self.cfg.attn_implementation,
            max_seq_len=self.cfg.max_seq_len,
            preprocess_metadata=preprocess_meta,
            warmup_ok=True,
        )
