"""Request/response schemas and strict semantic validation for bob-s1 OpenAI endpoints."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .calibration import BUILTIN_QUESTIONS
from .config import MODEL_MAX_POSITION_EMBEDDINGS, SidecarConfig
from .images import _DATA_URI_RE, decode_data_uri_images_batch

SUPPORTED_DIMENSIONS = frozenset({128, 256, 384})
SUPPORTED_ENCODINGS = frozenset({"float", "base64"})
CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"
DECISION_RESPONSE_SCHEMA_PATH = CONTRACTS_DIR / "decision.response.schema.json"

FORBIDDEN_CHAT_PARAMS = frozenset(
    {
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "temperature",
        "top_p",
        "n",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "top_logprobs",
        "audio",
        "modalities",
        "prediction",
        "reasoning_effort",
    }
)


class APIError(Exception):
    """Structured OpenAI-compatible HTTP error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_explicit_nulls(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in data.items():
                if value is None:
                    raise ValueError(f"Field '{key}' must be omitted rather than null")
        return data


class OptionSpec(_StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    value: float | None = None


class DecisionSpec(_StrictModel):
    id: str = Field(min_length=1)
    task: Literal["compaction", "outcome_prefix", "loop_guard"] | None = None
    question: str | None = Field(default=None, min_length=1)
    options: list[OptionSpec] | None = None
    scoring_mode: Literal["joint", "equivariant"] = "joint"
    apply_calibration: bool = True
    require_calibrated: bool = False


class JointEmbeddingSpec(_StrictModel):
    dimensions: int = 384
    encoding_format: Literal["float", "base64"] = "float"


class DecisionProtocolRequest(_StrictModel):
    protocol: Literal["bob-s1.decision.v1"]
    state: str
    decisions: list[DecisionSpec] = Field(min_length=1)
    embedding: JointEmbeddingSpec | None = None
    prefix_mode: Literal["per_question", "shared_prefix"] = "per_question"
    truncation: Literal["head_tail", "error"] = "head_tail"
    max_seq_len: int | None = None


@dataclass(frozen=True)
class RawEmbeddingItem:
    text: str
    image_refs: list[tuple[str, str]]  # (data_uri, param_path)


@dataclass(frozen=True)
class RawEmbeddingsRequest:
    model: str
    items: list[RawEmbeddingItem]
    dimensions: int
    encoding_format: str
    truncation: str
    effective_max_seq_len: int


@dataclass(frozen=True)
class ParsedEmbeddingItem:
    text: str
    images: list[Image.Image]


@dataclass(frozen=True)
class ParsedEmbeddingsRequest:
    model: str
    items: list[ParsedEmbeddingItem]
    dimensions: int
    encoding_format: str
    truncation: str
    effective_max_seq_len: int


@dataclass(frozen=True)
class ResolvedDecision:
    id: str
    task: str | None
    question: str
    options: list[tuple[str, str]]
    option_ids: list[str]
    option_values: list[float] | None
    scoring_mode: str
    apply_calibration: bool
    require_calibrated: bool


@dataclass(frozen=True)
class RawChatDecisionsRequest:
    model: str
    state: str
    image_refs: list[tuple[str, str]]  # (data_uri, param_path)
    decisions: list[ResolvedDecision]
    embedding_dimensions: int | None
    embedding_encoding_format: str | None
    prefix_mode: str
    truncation: str
    effective_max_seq_len: int


@dataclass(frozen=True)
class ParsedChatDecisionsRequest:
    model: str
    state: str
    images: list[Image.Image]
    decisions: list[ResolvedDecision]
    embedding_dimensions: int | None
    embedding_encoding_format: str | None
    prefix_mode: str
    truncation: str
    effective_max_seq_len: int


def _reject_bool_int(val: Any, *, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise APIError(400, f"{name} must be an integer", param=name)
    return val


def validate_requested_model(model_name: Any, cfg: SidecarConfig) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        raise APIError(400, "model must be a non-empty string", param="model")
    if model_name != cfg.model_id:
        raise APIError(
            400,
            f"Unknown model {model_name!r}; expected {cfg.model_id!r}",
            param="model",
            code="model_not_found",
        )
    return model_name


def validate_max_seq_len(raw: Any, cfg: SidecarConfig, *, key_present: bool) -> int:
    if not key_present:
        return cfg.max_seq_len
    if raw is None:
        raise APIError(400, "max_seq_len must be omitted rather than null", param="max_seq_len")
    val = _reject_bool_int(raw, name="max_seq_len")
    if val < 1 or val > MODEL_MAX_POSITION_EMBEDDINGS or val > cfg.max_seq_len:
        raise APIError(
            400,
            f"max_seq_len={val} exceeds configured server limit {cfg.max_seq_len} (must be in 1..{cfg.max_seq_len})",
            param="max_seq_len",
            code="context_length_exceeded",
        )
    return val


def _parse_content_parts_refs(
    parts: Any,
    *,
    cfg: SidecarConfig,
    param_prefix: str,
    require_text: bool,
) -> tuple[list[tuple[str, str]], str | None]:
    if not isinstance(parts, list) or not parts:
        raise APIError(400, f"{param_prefix} must be a non-empty list of content parts", param=param_prefix)

    image_refs: list[tuple[str, str]] = []
    text_val: str | None = None
    seen_text = False

    for idx, part in enumerate(parts):
        part_param = f"{param_prefix}[{idx}]"
        if not isinstance(part, dict):
            raise APIError(400, "Content part must be an object", param=part_param)
        part_type = part.get("type")
        if part_type == "image_url":
            if seen_text:
                raise APIError(
                    400,
                    "Arbitrary text-image interleaving is unsupported; all image_url parts must precede text",
                    param=part_param,
                )
            if set(part.keys()) != {"type", "image_url"}:
                raise APIError(400, "Unexpected fields in image_url content part", param=part_param)
            img_obj = part.get("image_url")
            if not isinstance(img_obj, dict) or set(img_obj.keys()) != {"url"}:
                raise APIError(400, "image_url must be an object containing only url", param=f"{part_param}.image_url")
            url = img_obj["url"]
            if not isinstance(url, str) or _DATA_URI_RE.match(url) is None:
                raise APIError(
                    400,
                    "Only inline data:image/(png|jpeg|webp|gif);base64,... URIs are accepted",
                    param=f"{part_param}.image_url.url",
                    code="invalid_image",
                )
            if len(image_refs) >= cfg.max_images:
                raise APIError(
                    400,
                    f"Image count exceeds BOB_S1_MAX_IMAGES ({cfg.max_images})",
                    param=part_param,
                    code="too_many_images",
                )
            image_refs.append((url, f"{part_param}.image_url.url"))
        elif part_type == "text":
            if seen_text:
                raise APIError(400, "At most one text part is allowed per item", param=part_param)
            if set(part.keys()) != {"type", "text"}:
                raise APIError(400, "Unexpected fields in text content part", param=part_param)
            txt = part.get("text")
            if not isinstance(txt, str):
                raise APIError(400, "text part must contain a string", param=f"{part_param}.text")
            text_val = txt
            seen_text = True
        else:
            raise APIError(400, f"Unsupported content part type {part_type!r}", param=part_param)

    if require_text and text_val is None:
        raise APIError(400, f"{param_prefix} requires a final text part", param=param_prefix)
    return image_refs, text_val


def parse_embeddings_structure(payload: dict[str, Any], cfg: SidecarConfig) -> RawEmbeddingsRequest:
    if not isinstance(payload, dict):
        raise APIError(400, "Request body must be a JSON object")

    allowed_keys = {"model", "input", "dimensions", "encoding_format", "user", "truncation", "max_seq_len"}
    unknown = sorted(set(payload.keys()) - allowed_keys)
    if unknown:
        raise APIError(400, f"Unsupported request parameters: {', '.join(unknown)}", param=unknown[0])

    model = validate_requested_model(payload.get("model"), cfg)

    if "dimensions" in payload:
        dims = _reject_bool_int(payload["dimensions"], name="dimensions")
    else:
        dims = 384
    if dims not in SUPPORTED_DIMENSIONS:
        raise APIError(400, f"dimensions must be one of {sorted(SUPPORTED_DIMENSIONS)}, got {dims}", param="dimensions")

    if "encoding_format" in payload:
        enc = payload["encoding_format"]
        if not isinstance(enc, str) or enc not in SUPPORTED_ENCODINGS:
            raise APIError(400, "encoding_format must be 'float' or 'base64'", param="encoding_format")
    else:
        enc = "float"

    if "truncation" in payload:
        truncation = payload["truncation"]
        if not isinstance(truncation, str) or truncation not in ("error", "head_tail"):
            raise APIError(400, "truncation must be 'error' or 'head_tail'", param="truncation")
    else:
        truncation = "error"

    if "user" in payload and not isinstance(payload["user"], str):
        raise APIError(400, "user must be a string", param="user")

    effective_max_seq_len = validate_max_seq_len(
        payload.get("max_seq_len"),
        cfg,
        key_present=("max_seq_len" in payload),
    )

    if "input" not in payload:
        raise APIError(400, "Missing required field 'input'", param="input")
    raw_input = payload["input"]
    items: list[RawEmbeddingItem] = []

    if isinstance(raw_input, str):
        items.append(RawEmbeddingItem(text=raw_input, image_refs=[]))
    elif isinstance(raw_input, list):
        if not raw_input:
            raise APIError(400, "input list must not be empty", param="input")
        if len(raw_input) > cfg.max_batch_size:
            raise APIError(
                400,
                f"Batch size {len(raw_input)} exceeds BOB_S1_MAX_BATCH_SIZE ({cfg.max_batch_size})",
                param="input",
                code="batch_too_large",
            )
        if all(isinstance(el, str) for el in raw_input):
            items = [RawEmbeddingItem(text=el, image_refs=[]) for el in raw_input]
        elif all(isinstance(el, dict) for el in raw_input):
            total_images = 0
            for idx, el in enumerate(raw_input):
                if set(el.keys()) != {"content"}:
                    raise APIError(
                        400,
                        "Multimodal input item must contain only 'content'",
                        param=f"input[{idx}]",
                    )
                refs, txt = _parse_content_parts_refs(
                    el["content"],
                    cfg=cfg,
                    param_prefix=f"input[{idx}].content",
                    require_text=False,
                )
                total_images += len(refs)
                if total_images > cfg.max_request_images:
                    raise APIError(
                        400,
                        f"Total request image count exceeds BOB_S1_MAX_REQUEST_IMAGES ({cfg.max_request_images})",
                        param="input",
                        code="too_many_images",
                    )
                items.append(RawEmbeddingItem(text=txt or "", image_refs=refs))
        else:
            raise APIError(
                400,
                "input must be a string, list of strings, or list of multimodal {content: [...]} objects; token arrays are unsupported",
                param="input",
            )
    else:
        raise APIError(
            400,
            "input must be a string, list of strings, or list of multimodal {content: [...]} objects",
            param="input",
        )

    return RawEmbeddingsRequest(
        model=model,
        items=items,
        dimensions=dims,
        encoding_format=enc,
        truncation=truncation,
        effective_max_seq_len=effective_max_seq_len,
    )


def materialize_embeddings_request(raw: RawEmbeddingsRequest, cfg: SidecarConfig) -> ParsedEmbeddingsRequest:
    flat_refs: list[tuple[str, str]] = []
    counts: list[int] = []
    for item in raw.items:
        counts.append(len(item.image_refs))
        flat_refs.extend(item.image_refs)

    decoded = (
        decode_data_uri_images_batch(
            flat_refs,
            max_bytes=cfg.max_image_bytes,
            max_pixels=cfg.max_image_pixels,
            max_total_pixels=cfg.max_request_image_pixels,
        )
        if flat_refs
        else []
    )

    items: list[ParsedEmbeddingItem] = []
    offset = 0
    for item, cnt in zip(raw.items, counts):
        items.append(ParsedEmbeddingItem(text=item.text, images=decoded[offset : offset + cnt]))
        offset += cnt

    return ParsedEmbeddingsRequest(
        model=raw.model,
        items=items,
        dimensions=raw.dimensions,
        encoding_format=raw.encoding_format,
        truncation=raw.truncation,
        effective_max_seq_len=raw.effective_max_seq_len,
    )


def parse_embeddings_request(payload: dict[str, Any], cfg: SidecarConfig) -> ParsedEmbeddingsRequest:
    return materialize_embeddings_request(parse_embeddings_structure(payload, cfg), cfg)


def _strip_schema_metadata(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_schema_metadata(v)
            for k, v in node.items()
            if k not in ("$schema", "$id", "title", "description")
        }
    if isinstance(node, list):
        return [_strip_schema_metadata(item) for item in node]
    return node


_CANONICAL_DECISION_RESPONSE_SCHEMA = _strip_schema_metadata(
    json.loads(DECISION_RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))
)


def _validate_response_format(raw_rf: Any, *, key_present: bool) -> None:
    if not key_present:
        return
    if not isinstance(raw_rf, dict):
        raise APIError(400, "response_format must be an object", param="response_format")
    rf_type = raw_rf.get("type")
    if rf_type in ("text", "json_object"):
        if set(raw_rf.keys()) != {"type"}:
            raise APIError(400, "Unexpected fields in response_format", param="response_format")
        return
    if rf_type == "json_schema":
        if set(raw_rf.keys()) != {"type", "json_schema"}:
            raise APIError(400, "Unexpected fields in json_schema response_format", param="response_format")
        js = raw_rf.get("json_schema")
        if not isinstance(js, dict):
            raise APIError(400, "response_format.json_schema must be an object", param="response_format.json_schema")
        allowed = {"name", "schema", "strict"}
        if not {"name", "schema"}.issubset(js.keys()) or not set(js.keys()).issubset(allowed):
            raise APIError(400, "Invalid keys in response_format.json_schema", param="response_format.json_schema")
        if js.get("name") != "bob_s1_decision_v1":
            raise APIError(
                400,
                "Only json_schema name 'bob_s1_decision_v1' is supported",
                param="response_format.json_schema.name",
            )
        if "strict" in js and not isinstance(js["strict"], bool):
            raise APIError(
                400,
                "response_format.json_schema.strict must be a boolean",
                param="response_format.json_schema.strict",
            )
        if _strip_schema_metadata(js.get("schema")) != _CANONICAL_DECISION_RESPONSE_SCHEMA:
            raise APIError(
                400,
                "Arbitrary custom json_schema is unsupported; schema must match decision.response.schema.json",
                param="response_format.json_schema.schema",
            )
        return
    raise APIError(400, f"Unsupported response_format type {rf_type!r}", param="response_format")


def _resolve_decision(spec: DecisionSpec, idx: int, cfg: SidecarConfig) -> ResolvedDecision:
    param = f"decisions[{idx}]"
    if spec.task is not None:
        canonical_q, canonical_opts = BUILTIN_QUESTIONS[spec.task]
        if spec.question is not None and spec.question != canonical_q:
            raise APIError(
                400,
                f"Decision {spec.id!r} contradicts canonical question for task {spec.task!r}",
                param=f"{param}.question",
            )
        if spec.options is not None:
            if len(spec.options) != len(canonical_opts):
                raise APIError(
                    400,
                    f"Decision {spec.id!r} options count contradicts canonical task {spec.task!r}",
                    param=f"{param}.options",
                )
            for opt, (c_id, c_desc) in zip(spec.options, canonical_opts):
                if opt.id != c_id or opt.description != c_desc:
                    raise APIError(
                        400,
                        f"Decision {spec.id!r} option {opt.id!r} contradicts canonical task {spec.task!r}",
                        param=f"{param}.options",
                    )
            raw_opts = spec.options
        else:
            raw_opts = [OptionSpec(id=c_id, description=c_desc) for c_id, c_desc in canonical_opts]
        question = canonical_q
    else:
        if spec.question is None or spec.options is None:
            raise APIError(
                400,
                f"Decision {spec.id!r} requires either 'task' or both 'question' and 'options'",
                param=param,
            )
        question = spec.question
        raw_opts = spec.options

    if len(raw_opts) < 2 or len(raw_opts) > cfg.max_options:
        raise APIError(
            400,
            f"Decision {spec.id!r} option count {len(raw_opts)} must be in 2..{cfg.max_options}",
            param=f"{param}.options",
        )

    option_ids = [o.id for o in raw_opts]
    if len(set(option_ids)) != len(option_ids):
        raise APIError(400, f"Decision {spec.id!r} contains duplicate option IDs", param=f"{param}.options")

    has_values = [o.value is not None for o in raw_opts]
    if any(has_values) and not all(has_values):
        raise APIError(
            400,
            f"Decision {spec.id!r} options must either all provide finite 'value' or none",
            param=f"{param}.options",
        )
    option_values: list[float] | None = None
    if all(has_values):
        option_values = []
        for o in raw_opts:
            assert o.value is not None
            if not math.isfinite(o.value):
                raise APIError(400, f"Option {o.id!r} value must be finite", param=f"{param}.options")
            option_values.append(float(o.value))

    return ResolvedDecision(
        id=spec.id,
        task=spec.task,
        question=question,
        options=[(o.id, o.description) for o in raw_opts],
        option_ids=option_ids,
        option_values=option_values,
        scoring_mode=spec.scoring_mode,
        apply_calibration=spec.apply_calibration,
        require_calibrated=spec.require_calibrated,
    )


def parse_chat_completions_structure(payload: dict[str, Any], cfg: SidecarConfig) -> RawChatDecisionsRequest:
    if not isinstance(payload, dict):
        raise APIError(400, "Request body must be a JSON object")

    for forbidden in sorted(FORBIDDEN_CHAT_PARAMS & set(payload.keys())):
        raise APIError(
            400,
            f"Parameter {forbidden!r} is unsupported by bob-s1 decision protocol",
            param=forbidden,
        )

    allowed_outer = {"model", "messages", "response_format", "stream", "logprobs", "user"}
    unknown_outer = sorted(set(payload.keys()) - allowed_outer)
    if unknown_outer:
        raise APIError(400, f"Unsupported request parameters: {', '.join(unknown_outer)}", param=unknown_outer[0])

    model = validate_requested_model(payload.get("model"), cfg)

    if "stream" in payload and payload["stream"] is not False:
        raise APIError(400, "Streaming (stream=true or non-boolean) is unsupported", param="stream")

    if "logprobs" in payload and payload["logprobs"] is not False:
        raise APIError(400, "Token logprobs (logprobs=true or non-boolean) are unsupported", param="logprobs")

    if "user" in payload and not isinstance(payload["user"], str):
        raise APIError(400, "user must be a string", param="user")

    _validate_response_format(payload.get("response_format"), key_present=("response_format" in payload))

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise APIError(400, "messages must contain exactly one user message", param="messages")

    msg = messages[0]
    if not isinstance(msg, dict) or msg.get("role") != "user":
        raise APIError(400, "messages[0] must be an object with role='user'", param="messages[0]")
    if set(msg.keys()) != {"role", "content"}:
        raise APIError(400, "messages[0] must contain only 'role' and 'content'", param="messages[0]")

    raw_content = msg.get("content")
    image_refs: list[tuple[str, str]] = []
    if isinstance(raw_content, str):
        inner_text = raw_content
    elif isinstance(raw_content, list):
        image_refs, parsed_txt = _parse_content_parts_refs(
            raw_content,
            cfg=cfg,
            param_prefix="messages[0].content",
            require_text=True,
        )
        if len(image_refs) > cfg.max_request_images:
            raise APIError(
                400,
                f"Total request image count exceeds BOB_S1_MAX_REQUEST_IMAGES ({cfg.max_request_images})",
                param="messages[0].content",
                code="too_many_images",
            )
        assert parsed_txt is not None
        inner_text = parsed_txt
    else:
        raise APIError(400, "messages[0].content must be a JSON string or content parts array", param="messages[0].content")

    try:
        inner_obj = json.loads(inner_text)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise APIError(
            400,
            f"messages[0] text must be valid JSON for bob-s1.decision.v1: {exc}",
            param="messages[0].content",
        ) from exc

    if not isinstance(inner_obj, dict):
        raise APIError(400, "Decision protocol payload must be a JSON object", param="messages[0].content")

    effective_max_seq_len = validate_max_seq_len(
        inner_obj.get("max_seq_len"),
        cfg,
        key_present=("max_seq_len" in inner_obj),
    )

    try:
        proto = DecisionProtocolRequest.model_validate(inner_obj)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first["loc"])
        raise APIError(400, f"Invalid decision payload ({loc}): {first['msg']}", param=loc or "messages[0].content") from exc

    if len(proto.decisions) > cfg.max_decisions:
        raise APIError(
            400,
            f"Decision count {len(proto.decisions)} exceeds BOB_S1_MAX_DECISIONS ({cfg.max_decisions})",
            param="decisions",
        )

    decision_ids = [d.id for d in proto.decisions]
    if len(set(decision_ids)) != len(decision_ids):
        raise APIError(400, "Duplicate decision IDs are not allowed", param="decisions")

    if proto.embedding is not None:
        if proto.embedding.dimensions not in SUPPORTED_DIMENSIONS:
            raise APIError(
                400,
                f"embedding.dimensions must be one of {sorted(SUPPORTED_DIMENSIONS)}, got {proto.embedding.dimensions}",
                param="embedding.dimensions",
            )
        if len(proto.decisions) > 1 and proto.prefix_mode != "shared_prefix":
            raise APIError(
                400,
                "Requesting a joint embedding across multiple decisions requires prefix_mode='shared_prefix'",
                param="prefix_mode",
            )

    resolved = [_resolve_decision(d, idx, cfg) for idx, d in enumerate(proto.decisions)]

    return RawChatDecisionsRequest(
        model=model,
        state=proto.state,
        image_refs=image_refs,
        decisions=resolved,
        embedding_dimensions=proto.embedding.dimensions if proto.embedding is not None else None,
        embedding_encoding_format=proto.embedding.encoding_format if proto.embedding is not None else None,
        prefix_mode=proto.prefix_mode,
        truncation=proto.truncation,
        effective_max_seq_len=effective_max_seq_len,
    )


def materialize_chat_completions_request(raw: RawChatDecisionsRequest, cfg: SidecarConfig) -> ParsedChatDecisionsRequest:
    images = (
        decode_data_uri_images_batch(
            raw.image_refs,
            max_bytes=cfg.max_image_bytes,
            max_pixels=cfg.max_image_pixels,
            max_total_pixels=cfg.max_request_image_pixels,
        )
        if raw.image_refs
        else []
    )
    return ParsedChatDecisionsRequest(
        model=raw.model,
        state=raw.state,
        images=images,
        decisions=raw.decisions,
        embedding_dimensions=raw.embedding_dimensions,
        embedding_encoding_format=raw.embedding_encoding_format,
        prefix_mode=raw.prefix_mode,
        truncation=raw.truncation,
        effective_max_seq_len=raw.effective_max_seq_len,
    )


def parse_chat_completions_request(payload: dict[str, Any], cfg: SidecarConfig) -> ParsedChatDecisionsRequest:
    return materialize_chat_completions_request(parse_chat_completions_structure(payload, cfg), cfg)


def encode_embedding_vector(values: list[float], encoding_format: str) -> list[float] | str:
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Embedding vector contains non-finite values")
    if encoding_format == "float":
        return values
    if encoding_format == "base64":
        packed = struct.pack(f"<{len(values)}f", *values)
        return base64.b64encode(packed).decode("ascii")
    raise ValueError(f"Unsupported encoding_format: {encoding_format!r}")
