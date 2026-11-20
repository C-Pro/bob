"""Configuration parsing and validation for the bob-s1 inference sidecar."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

MODEL_MAX_POSITION_EMBEDDINGS = 262144
MIN_CONFIGURABLE_SEQ_LEN = 256
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MODEL_ID = "bob-s1-0.8b"
DEFAULT_CHECKPOINT_SHA256 = "a0cfc5bb5fd214f76acdb2774cc99e2be35f4432c5cc85547ae9b410b731397c"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_int(name: str, raw: str | None, default: int, minimum: int, maximum: int | None = None) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be in {bound}, got {value}")
    return value


@dataclass(frozen=True)
class SidecarConfig:
    """Immutable runtime configuration for the bob-s1 sidecar."""

    host: str = "127.0.0.1"
    port: int = 8090
    api_key: str | None = None
    model_id: str = DEFAULT_MODEL_ID
    loader_kind: str = "checkpoint_overlay"
    repo_root: Path = _default_repo_root()
    checkpoint_path: Path = _default_repo_root() / "eval/jev/checkpoints/bob-s1-0.8b/model.pt"
    expected_checkpoint_sha256: str | None = DEFAULT_CHECKPOINT_SHA256
    hf_cache_dir: Path | None = None
    device: str = "auto"
    dtype: str = "auto"
    attn_implementation: str = "eager"
    merge_lora: bool = False
    cpu_threads: int | None = None
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN
    max_queue_depth: int = 16
    max_batch_size: int = 16
    max_decisions: int = 16
    max_options: int = 255
    max_images: int = 8
    max_request_images: int = 16
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 4096 * 4096
    max_request_image_pixels: int = 16 * 1024 * 1024
    max_body_bytes: int = 16 * 1024 * 1024

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SidecarConfig":
        src = os.environ if env is None else env
        repo_root = Path(src["BOB_S1_REPO_ROOT"]).resolve() if src.get("BOB_S1_REPO_ROOT") else _default_repo_root()
        default_ckpt = repo_root / "eval/jev/checkpoints/bob-s1-0.8b/model.pt"
        checkpoint_path = Path(src["BOB_S1_CHECKPOINT"]).expanduser() if src.get("BOB_S1_CHECKPOINT") else default_ckpt
        if not checkpoint_path.is_absolute():
            checkpoint_path = (repo_root / checkpoint_path).resolve()

        raw_sha = src.get("BOB_S1_EXPECTED_SHA256")
        if raw_sha is None:
            expected_sha: str | None = DEFAULT_CHECKPOINT_SHA256
        elif raw_sha.strip() == "":
            expected_sha = None
        else:
            expected_sha = raw_sha.strip().lower()

        hf_cache_raw = src.get("BOB_S1_HF_CACHE_DIR")
        hf_cache_dir = Path(hf_cache_raw).expanduser().resolve() if hf_cache_raw else None

        loader_kind = (src.get("BOB_S1_LOADER") or "checkpoint_overlay").strip()
        if loader_kind not in ("checkpoint_overlay",):
            raise ValueError(f"Unsupported BOB_S1_LOADER: {loader_kind!r}")

        device = (src.get("BOB_S1_DEVICE") or "auto").strip()
        if device not in ("auto", "cpu", "cuda") and not device.startswith("cuda:"):
            raise ValueError(f"Unsupported BOB_S1_DEVICE: {device!r}")

        dtype = (src.get("BOB_S1_DTYPE") or "auto").strip().lower()
        if dtype not in ("auto", "bf16", "fp32", "fp16"):
            raise ValueError(f"Unsupported BOB_S1_DTYPE: {dtype!r}")

        attn = (src.get("BOB_S1_ATTN_IMPLEMENTATION") or "eager").strip().lower()
        if attn not in ("eager", "sdpa"):
            raise ValueError(f"Unsupported BOB_S1_ATTN_IMPLEMENTATION: {attn!r}")

        api_key = src.get("BOB_S1_API_KEY")
        if api_key is not None:
            api_key = api_key.strip() or None

        merge_lora_raw = (src.get("BOB_S1_MERGE_LORA") or "").strip().lower()
        merge_lora = merge_lora_raw in ("1", "true", "yes", "on")
        cpu_threads_raw = src.get("BOB_S1_CPU_THREADS")
        cpu_threads = (
            _parse_int("BOB_S1_CPU_THREADS", cpu_threads_raw, 1, 1, 512)
            if cpu_threads_raw is not None and cpu_threads_raw.strip() != ""
            else None
        )

        return cls(
            host=(src.get("BOB_S1_HOST") or "127.0.0.1").strip(),
            port=_parse_int("BOB_S1_PORT", src.get("BOB_S1_PORT"), 8090, 1, 65535),
            api_key=api_key,
            model_id=(src.get("BOB_S1_MODEL_ID") or DEFAULT_MODEL_ID).strip(),
            loader_kind=loader_kind,
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_sha,
            hf_cache_dir=hf_cache_dir,
            device=device,
            dtype=dtype,
            attn_implementation=attn,
            merge_lora=merge_lora,
            cpu_threads=cpu_threads,
            max_seq_len=_parse_int(
                "BOB_S1_MAX_SEQ_LEN",
                src.get("BOB_S1_MAX_SEQ_LEN"),
                DEFAULT_MAX_SEQ_LEN,
                MIN_CONFIGURABLE_SEQ_LEN,
                MODEL_MAX_POSITION_EMBEDDINGS,
            ),
            max_queue_depth=_parse_int("BOB_S1_MAX_QUEUE_DEPTH", src.get("BOB_S1_MAX_QUEUE_DEPTH"), 16, 1, 1024),
            max_batch_size=_parse_int("BOB_S1_MAX_BATCH_SIZE", src.get("BOB_S1_MAX_BATCH_SIZE"), 16, 1, 256),
            max_decisions=_parse_int("BOB_S1_MAX_DECISIONS", src.get("BOB_S1_MAX_DECISIONS"), 16, 1, 128),
            max_options=_parse_int("BOB_S1_MAX_OPTIONS", src.get("BOB_S1_MAX_OPTIONS"), 255, 2, 255),
            max_images=_parse_int("BOB_S1_MAX_IMAGES", src.get("BOB_S1_MAX_IMAGES"), 8, 1, 64),
            max_request_images=_parse_int("BOB_S1_MAX_REQUEST_IMAGES", src.get("BOB_S1_MAX_REQUEST_IMAGES"), 16, 1, 256),
            max_image_bytes=_parse_int("BOB_S1_MAX_IMAGE_BYTES", src.get("BOB_S1_MAX_IMAGE_BYTES"), 10 * 1024 * 1024, 1024),
            max_image_pixels=_parse_int("BOB_S1_MAX_IMAGE_PIXELS", src.get("BOB_S1_MAX_IMAGE_PIXELS"), 4096 * 4096, 64),
            max_request_image_pixels=_parse_int(
                "BOB_S1_MAX_REQUEST_IMAGE_PIXELS",
                src.get("BOB_S1_MAX_REQUEST_IMAGE_PIXELS"),
                16 * 1024 * 1024,
                64,
            ),
            max_body_bytes=_parse_int("BOB_S1_MAX_BODY_BYTES", src.get("BOB_S1_MAX_BODY_BYTES"), 16 * 1024 * 1024, 1024),
        )
