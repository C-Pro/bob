"""Single device-owning worker queue and DeltaNet-safe inference execution."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
import queue
import threading
from typing import Any, Callable

from PIL import Image
import torch

from .calibration import CalibrationRegistry, score_decision
from .config import SidecarConfig
from .loader import LoadedModelBundle
from .schemas import (
    APIError,
    ParsedChatDecisionsRequest,
    ParsedEmbeddingsRequest,
    ResolvedDecision,
    encode_embedding_vector,
)


@dataclass
class QueuedJob:
    """Represents a unit of device work with serialized queue-only cancellation/failure."""

    fn: Callable[[], Any]
    future: Future[Any] = field(default_factory=Future)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _started: bool = False
    _cancelled: bool = False

    def try_cancel_if_queued(self) -> bool:
        with self._lock:
            if self._started:
                return False
            if not self._cancelled and not self.future.done():
                self._cancelled = True
                self.future.cancel()
            return True

    def fail_if_queued(self, exc: BaseException) -> bool:
        with self._lock:
            if self._started or self._cancelled or self.future.done():
                return False
            self._cancelled = True
            self.future.set_exception(exc)
            return True

    def mark_started_unless_cancelled(self) -> bool:
        with self._lock:
            if self._cancelled or self.future.cancelled():
                return False
            self._started = True
            return True


class AdmissionLease:
    """Exactly-once admission reservation released on decode abort or promoted to queue."""

    def __init__(self, worker: "InferenceWorker"):
        self._worker = worker
        self._lock = threading.Lock()
        self._consumed = False

    def release(self) -> None:
        with self._lock:
            if self._consumed:
                return
            self._consumed = True
        self._worker._release_admission_slot()

    def promote_to_queue(self, fn: Callable[[], Any]) -> QueuedJob:
        with self._lock:
            if self._consumed:
                raise RuntimeError("AdmissionLease has already been consumed")
            self._consumed = True
        return self._worker._enqueue_admitted(fn)


@dataclass
class _PreparedPrefix:
    prefix_ids: list[int]
    truncated: bool
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    image_token_count: int


def _prepare_prefix_checked(
    model: Any,
    state_text: str,
    images: list[Image.Image],
    *,
    effective_max_seq_len: int,
    suffix_reserve: int,
    truncation_policy: str,
) -> _PreparedPrefix:
    """Reproduce DualHeadDecisionEmbedder.prepare_state with explicit overflow/budget checks."""
    pixel_values, image_grid_thw, token_counts = model.preprocess_images(images or None)
    image_token_count = sum(2 + c for c in token_counts)
    budget = effective_max_seq_len - suffix_reserve
    text_budget = budget - image_token_count
    if text_budget < 1:
        raise APIError(
            400,
            f"Images ({image_token_count} tok) and suffix ({suffix_reserve} tok) leave no room for prefix delimiter within max_seq_len={effective_max_seq_len}",
            param="max_seq_len",
            code="context_length_exceeded",
        )

    raw_text_ids = model._text_ids(state_text)
    keep = text_budget - 1
    if len(raw_text_ids) > keep and truncation_policy == "error":
        raise APIError(
            400,
            f"Input requires {1 + image_token_count + len(raw_text_ids) + suffix_reserve} tokens, exceeding max_seq_len={effective_max_seq_len} under truncation='error'",
            param="input",
            code="context_length_exceeded",
        )

    ids, truncated, pv, grid = model.prepare_state(state_text, budget, images=images or None)
    return _PreparedPrefix(
        prefix_ids=ids,
        truncated=truncated,
        pixel_values=pv,
        image_grid_thw=grid,
        image_token_count=image_token_count,
    )


def _encode_prefix_prepared(model: Any, prep: _PreparedPrefix, *, effective_max_seq_len: int) -> Any:
    """Compute immutable prefix cache and embeddings once from already prepared tokens."""
    from eval.jev.model import Prefix

    input_ids_tensor = torch.tensor([prep.prefix_ids], dtype=torch.long, device=model.device)
    position_ids, next_pos = model._compute_position_ids(input_ids_tensor, prep.image_grid_thw)
    output = model._run(
        prep.prefix_ids,
        use_cache=True,
        pixel_values=prep.pixel_values,
        image_grid_thw=prep.image_grid_thw,
        position_ids=position_ids,
    )
    return Prefix(
        output.past_key_values,
        prep.prefix_ids,
        model.embeddings(output.last_hidden_state[0, -1]),
        prep.truncated,
        next_pos=next_pos,
        max_seq_len=effective_max_seq_len,
    )


class InferenceWorker:
    """Owns the model/device on a dedicated worker thread with bounded admission + FIFO queue."""

    def __init__(self, bundle: LoadedModelBundle, cfg: SidecarConfig, registry: CalibrationRegistry):
        self.bundle = bundle
        self.cfg = cfg
        self.registry = registry
        self._state_lock = threading.Lock()
        self._closing = False
        self._pending_admission = 0  # decoding + queued (not yet running on GPU)
        self._queue: queue.Queue[QueuedJob | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="bob-s1-worker", daemon=True)
        self._thread.start()

    def reserve_slot(self) -> AdmissionLease:
        """Reserve a bounded admission slot before decoding large payloads."""
        with self._state_lock:
            if self._closing:
                raise APIError(
                    503,
                    "Worker is shutting down",
                    error_type="service_unavailable_error",
                    code="model_not_ready",
                )
            if self._pending_admission >= self.cfg.max_queue_depth:
                raise APIError(
                    429,
                    f"Inference queue is full (max_queue_depth={self.cfg.max_queue_depth})",
                    error_type="rate_limit_error",
                    code="queue_full",
                )
            self._pending_admission += 1
        return AdmissionLease(self)

    def _release_admission_slot(self) -> None:
        with self._state_lock:
            if self._pending_admission > 0:
                self._pending_admission -= 1

    def _enqueue_admitted(self, fn: Callable[[], Any]) -> QueuedJob:
        """Promote a reserved admission slot into the execution queue atomically."""
        job = QueuedJob(fn=fn)
        with self._state_lock:
            if self._closing:
                if self._pending_admission > 0:
                    self._pending_admission -= 1
                raise APIError(
                    503,
                    "Worker is shutting down",
                    error_type="service_unavailable_error",
                    code="model_not_ready",
                )
            self._queue.put_nowait(job)
        return job

    def close(self, timeout: float = 10.0) -> None:
        """Reject new work, fail queued jobs cleanly, and wait for the active forward to finish."""
        with self._state_lock:
            if self._closing:
                already_closing = True
            else:
                self._closing = True
                already_closing = False

            drained: list[QueuedJob] = []
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    drained.append(item)
                    if self._pending_admission > 0:
                        self._pending_admission -= 1
                self._queue.task_done()

        shutdown_err = APIError(
            503,
            "Worker is shutting down",
            error_type="service_unavailable_error",
            code="model_not_ready",
        )
        for job in drained:
            job.fail_if_queued(shutdown_err)

        if not already_closing:
            self._stop_event.set()
            self._queue.put_nowait(None)
        self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if job is None:
                self._queue.task_done()
                break

            with self._state_lock:
                if self._pending_admission > 0:
                    self._pending_admission -= 1
                closing_now = self._closing

            try:
                if closing_now:
                    job.fail_if_queued(
                        APIError(
                            503,
                            "Worker is shutting down",
                            error_type="service_unavailable_error",
                            code="model_not_ready",
                        )
                    )
                    continue
                if not job.mark_started_unless_cancelled():
                    continue
                with torch.inference_mode():
                    result = job.fn()
                if not job.future.done():
                    job.future.set_result(result)
            except BaseException as exc:  # noqa: BLE001 - propagate any worker exception to caller future
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def submit(self, fn: Callable[[], Any]) -> Any:
        lease = self.reserve_slot()
        try:
            job = lease.promote_to_queue(fn)
        except BaseException:
            lease.release()
            raise
        wrapped = asyncio.wrap_future(job.future)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            job.try_cancel_if_queued()
            raise

    def run_embeddings(self, req: ParsedEmbeddingsRequest) -> dict[str, Any]:
        model = self.bundle.model
        model.eval()

        prepared = [
            _prepare_prefix_checked(
                model,
                item.text,
                item.images,
                effective_max_seq_len=req.effective_max_seq_len,
                suffix_reserve=0,
                truncation_policy=req.truncation,
            )
            for item in req.items
        ]

        data: list[dict[str, Any]] = []
        total_tokens = 0
        for idx, prep in enumerate(prepared):
            out = model._run(
                prep.prefix_ids,
                pixel_values=prep.pixel_values,
                image_grid_thw=prep.image_grid_thw,
            )
            hidden = out.last_hidden_state[0, -1]
            emb_slices = model.embeddings(hidden)
            vec = emb_slices[req.dimensions].detach().cpu().float().tolist()
            data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": encode_embedding_vector(vec, req.encoding_format),
                }
            )
            total_tokens += len(prep.prefix_ids)

        return {
            "object": "list",
            "data": data,
            "model": req.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    def run_chat_decisions(self, req: ParsedChatDecisionsRequest) -> tuple[dict[str, Any], int]:
        model = self.bundle.model
        model.eval()

        suffix_info: list[tuple[ResolvedDecision, int]] = []
        for dec in req.decisions:
            if dec.scoring_mode == "joint":
                s_ids, _, _ = model.question_ids(dec.question, dec.options)
                reserve = len(s_ids)
            else:
                opt_runs = [model.question_ids(dec.question, [opt]) for opt in dec.options]
                reserve = max(len(item[0]) for item in opt_runs)
            suffix_info.append((dec, reserve))

        decision_results: dict[str, dict[str, Any]] = {}
        embedding_payload: dict[str, Any] | None = None
        prompt_tokens = 0
        shared_prefix_tokens: int | None = None
        shared_suffix_reserve: int | None = None
        any_truncated = False

        if req.prefix_mode == "shared_prefix":
            shared_reserve = max(reserve for _, reserve in suffix_info)
            prep = _prepare_prefix_checked(
                model,
                req.state,
                req.images,
                effective_max_seq_len=req.effective_max_seq_len,
                suffix_reserve=shared_reserve,
                truncation_policy=req.truncation,
            )
            shared_prefix_tokens = len(prep.prefix_ids)
            shared_suffix_reserve = shared_reserve
            any_truncated = prep.truncated

            prefix = _encode_prefix_prepared(model, prep, effective_max_seq_len=req.effective_max_seq_len)
            prompt_tokens += len(prep.prefix_ids)

            if req.embedding_dimensions is not None:
                vec = prefix.embeddings[req.embedding_dimensions].detach().cpu().float().tolist()
                embedding_payload = {
                    "dimensions": req.embedding_dimensions,
                    "encoding_format": req.embedding_encoding_format or "float",
                    "vector": encode_embedding_vector(vec, req.embedding_encoding_format or "float"),
                }

            for dec, reserve in suffix_info:
                logits_list, branch_tokens = self._score_from_prefix(model, prefix, dec)
                prompt_tokens += branch_tokens
                decision_results[dec.id] = self._build_decision_entry(
                    dec=dec,
                    logits=logits_list,
                    req=req,
                    prefix_tokens=len(prep.prefix_ids),
                    suffix_tokens=reserve,
                    truncated=prep.truncated,
                )
        else:
            prepared_groups: OrderedDict[tuple[tuple[int, ...], int], list[tuple[ResolvedDecision, int, _PreparedPrefix]]] = OrderedDict()
            prepared_rows: list[tuple[ResolvedDecision, int, _PreparedPrefix]] = []
            for dec, reserve in suffix_info:
                prep = _prepare_prefix_checked(
                    model,
                    req.state,
                    req.images,
                    effective_max_seq_len=req.effective_max_seq_len,
                    suffix_reserve=reserve,
                    truncation_policy=req.truncation,
                )
                any_truncated = any_truncated or prep.truncated
                prepared_rows.append((dec, reserve, prep))
                key = (tuple(prep.prefix_ids), prep.image_token_count)
                prepared_groups.setdefault(key, []).append((dec, reserve, prep))

            for group in prepared_groups.values():
                if len(group) == 1 and req.embedding_dimensions is None and group[0][0].scoring_mode == "joint":
                    dec, reserve, prep = group[0]
                    s_ids, ends, q_end = model.question_ids(dec.question, dec.options)
                    from eval.jev.model import Prompt

                    prompt = Prompt(
                        prep.prefix_ids,
                        s_ids,
                        ends,
                        q_end,
                        prep.truncated,
                        prep.pixel_values,
                        prep.image_grid_thw,
                    )
                    out = model.forward_prompt(prompt)
                    logits_list = out["logits"].detach().cpu().float().tolist()
                    prompt_tokens += len(prep.prefix_ids) + len(s_ids)
                    decision_results[dec.id] = self._build_decision_entry(
                        dec=dec,
                        logits=logits_list,
                        req=req,
                        prefix_tokens=len(prep.prefix_ids),
                        suffix_tokens=reserve,
                        truncated=prep.truncated,
                    )
                else:
                    rep_prep = group[0][2]
                    prefix = _encode_prefix_prepared(model, rep_prep, effective_max_seq_len=req.effective_max_seq_len)
                    prompt_tokens += len(rep_prep.prefix_ids)

                    if req.embedding_dimensions is not None and embedding_payload is None:
                        vec = prefix.embeddings[req.embedding_dimensions].detach().cpu().float().tolist()
                        embedding_payload = {
                            "dimensions": req.embedding_dimensions,
                            "encoding_format": req.embedding_encoding_format or "float",
                            "vector": encode_embedding_vector(vec, req.embedding_encoding_format or "float"),
                        }

                    for dec, reserve, prep in group:
                        logits_list, branch_tokens = self._score_from_prefix(model, prefix, dec)
                        prompt_tokens += branch_tokens
                        decision_results[dec.id] = self._build_decision_entry(
                            dec=dec,
                            logits=logits_list,
                            req=req,
                            prefix_tokens=len(prep.prefix_ids),
                            suffix_tokens=reserve,
                            truncated=prep.truncated,
                        )

            unique_prefixes = {len(prep.prefix_ids) for _, _, prep in prepared_rows}
            unique_reserves = {reserve for _, reserve, _ in prepared_rows}
            if len(unique_prefixes) == 1 and len(unique_reserves) == 1:
                shared_prefix_tokens = next(iter(unique_prefixes))
                shared_suffix_reserve = next(iter(unique_reserves))
            else:
                shared_prefix_tokens = None
                shared_suffix_reserve = None

        ordered_decisions = [decision_results[dec.id] for dec in req.decisions]
        body = {
            "protocol": "bob-s1.decision.v1",
            "decisions": ordered_decisions,
            "embedding": embedding_payload,
            "preparation": {
                "prefix_mode": req.prefix_mode,
                "truncation_policy": req.truncation,
                "effective_max_seq_len": req.effective_max_seq_len,
                "suffix_reserve": shared_suffix_reserve,
                "prefix_tokens": shared_prefix_tokens,
                "truncated": any_truncated,
            },
        }
        return body, prompt_tokens

    def _score_from_prefix(self, model: Any, prefix: Any, dec: ResolvedDecision) -> tuple[list[float], int]:
        if dec.scoring_mode == "joint":
            suffix_ids, _, _ = model.question_ids(dec.question, dec.options)
            out = model.question_from_prefix(prefix, dec.question, dec.options, isolate=True)
            logits_list = out["logits"].detach().cpu().float().tolist()
            return logits_list, len(suffix_ids)

        logits_list: list[float] = []
        branch_tokens = 0
        for opt in dec.options:
            opt_suffix_ids, _, _ = model.question_ids(dec.question, [opt])
            out = model.question_from_prefix(prefix, dec.question, [opt], isolate=True)
            logits_list.append(float(out["logits"][0].detach().cpu().item()))
            branch_tokens += len(opt_suffix_ids)
        return logits_list, branch_tokens

    def _build_decision_entry(
        self,
        *,
        dec: ResolvedDecision,
        logits: list[float],
        req: ParsedChatDecisionsRequest,
        prefix_tokens: int,
        suffix_tokens: int,
        truncated: bool,
    ) -> dict[str, Any]:
        scored = score_decision(
            logits=logits,
            option_ids=dec.option_ids,
            option_values=dec.option_values,
            task=dec.task,
            scoring_mode=dec.scoring_mode,
            prefix_mode=req.prefix_mode,
            effective_max_seq_len=req.effective_max_seq_len,
            checkpoint_id=self.bundle.checkpoint_id,
            checkpoint_sha256=self.bundle.checkpoint_sha256,
            apply_calibration=dec.apply_calibration,
            require_calibrated=dec.require_calibrated,
            registry=self.registry,
            decision_param=f"decisions[{dec.id}]",
        )
        return {
            "id": dec.id,
            "task": dec.task,
            "scoring_mode": dec.scoring_mode,
            "choice": scored["choice"],
            "choice_basis": scored["choice_basis"],
            "scores": scored["scores"],
            "probability_kind": scored["probability_kind"],
            "expected_score": scored["expected_score"],
            "calibration": scored["calibration"],
            "prefix_tokens": prefix_tokens,
            "suffix_tokens": suffix_tokens,
            "truncated": truncated,
        }
