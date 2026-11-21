"""Self-contained serving calibration registry and PointerHead probability scoring."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"
DEFAULT_CALIBRATION_PATH = CONTRACTS_DIR / "calibration_profiles.json"

# Canonical built-in task definitions matching eval/jev/model.py QUESTIONS.
BUILTIN_QUESTIONS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "compaction": (
        "Should the observed task state be compacted now?",
        (
            ("skip", "Keep raw context; no coherent verified checkpoint yet."),
            ("compact", "Summarize a coherent verified checkpoint into task state."),
        ),
    ),
    "outcome_prefix": (
        "Does the observed progress suggest eventual task success?",
        (
            ("failed", "The task will fail its clean terminal evaluation."),
            ("passed", "The task will pass its clean terminal evaluation."),
        ),
    ),
    "loop_guard": (
        "Is this proposed tool execution stuck in a repeating loop?",
        (
            ("none", "Allow productive progress or a constructive error."),
            ("intercept", "Block repeated calls with unchanged observations."),
            ("force_synthesis", "A blocked call is repeated; stop the tool loop."),
        ),
    ),
}


class CalibrationError(ValueError):
    """Raised when calibration policy conflicts or required calibration is unavailable."""

    def __init__(self, message: str, *, code: str, param: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.param = param


@dataclass(frozen=True)
class CalibrationProfile:
    task: str
    profile_id: str
    positive_option_id: str
    negative_option_id: str
    question: str
    options: tuple[tuple[str, str], ...]
    temperature: float
    bias: float
    balanced_threshold: float | None
    risk_threshold_5pct: float | None
    selective_policy: bool = False


@dataclass(frozen=True)
class CalibrationRegistry:
    checkpoint_id: str
    checkpoint_sha256: str
    scoring_mode: str
    prefix_mode: str
    effective_max_seq_len: int
    profiles: dict[str, CalibrationProfile]

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CALIBRATION_PATH) -> "CalibrationRegistry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        req = raw["binding_requirements"]
        profiles: dict[str, CalibrationProfile] = {}
        for task_name, entry in raw["profiles"].items():
            opts = tuple((o["id"], o["description"]) for o in entry["options"])
            canonical = BUILTIN_QUESTIONS.get(task_name)
            if canonical is None:
                raise ValueError(f"Calibration registry references unknown built-in task {task_name!r}")
            if entry["question"] != canonical[0] or opts != canonical[1]:
                raise ValueError(f"Calibration profile {task_name!r} diverges from canonical BUILTIN_QUESTIONS")
            risk = entry.get("risk_policy") or {}
            profiles[task_name] = CalibrationProfile(
                task=task_name,
                profile_id=entry["profile_id"],
                positive_option_id=entry["positive_option_id"],
                negative_option_id=entry["negative_option_id"],
                question=entry["question"],
                options=opts,
                temperature=float(entry["temperature"]),
                bias=float(entry["bias"]),
                balanced_threshold=float(entry["balanced_threshold"]) if entry.get("balanced_threshold") is not None else None,
                risk_threshold_5pct=float(risk["threshold"]) if risk.get("threshold") is not None else None,
                selective_policy=bool(entry.get("selective_policy", False)),
            )
        return cls(
            checkpoint_id=str(req["checkpoint_id"]),
            checkpoint_sha256=str(req["checkpoint_model_pt_sha256"]),
            scoring_mode=str(req["scoring_mode"]),
            prefix_mode=str(req["prefix_mode"]),
            effective_max_seq_len=int(req["effective_max_seq_len"]),
            profiles=profiles,
        )


def _stable_softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(x - peak) for x in logits]
    denom = sum(exps)
    return [val / denom for val in exps]


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _argmax_earliest(values: list[float]) -> int:
    best_idx = 0
    best_val = values[0]
    for idx in range(1, len(values)):
        if values[idx] > best_val:
            best_val = values[idx]
            best_idx = idx
    return best_idx


def score_decision(
    *,
    logits: list[float],
    option_ids: list[str],
    option_values: list[float] | None,
    task: str | None,
    scoring_mode: str,
    prefix_mode: str,
    effective_max_seq_len: int,
    checkpoint_id: str,
    checkpoint_sha256: str | None,
    apply_calibration: bool,
    require_calibrated: bool,
    registry: CalibrationRegistry,
    decision_param: str = "decisions",
) -> dict[str, Any]:
    """Apply either bound affine calibration or uncalibrated softmax to raw PointerHead logits."""
    if len(logits) != len(option_ids) or len(logits) < 2:
        raise ValueError("Logits and option_ids must have identical length >= 2")
    if any(not math.isfinite(x) for x in logits):
        raise ValueError("Decision logits contain non-finite values")

    if not apply_calibration and require_calibrated:
        raise CalibrationError(
            "apply_calibration=false conflicts with require_calibrated=true",
            code="calibration_conflict",
            param=decision_param,
        )

    mismatch_reason: str | None = None
    profile: CalibrationProfile | None = None

    if not apply_calibration:
        mismatch_reason = "calibration_disabled_by_request"
    elif task is None:
        mismatch_reason = "no_calibration_profile_for_custom_task"
    elif task not in registry.profiles:
        mismatch_reason = "no_calibration_profile_for_task"
    elif scoring_mode != registry.scoring_mode:
        mismatch_reason = "scoring_mode_not_calibrated"
    elif prefix_mode != registry.prefix_mode:
        mismatch_reason = "prefix_mode_not_calibrated"
    elif effective_max_seq_len != registry.effective_max_seq_len:
        mismatch_reason = "max_seq_len_not_calibrated"
    elif checkpoint_id != registry.checkpoint_id or checkpoint_sha256 != registry.checkpoint_sha256:
        mismatch_reason = "checkpoint_digest_mismatch"
    else:
        candidate = registry.profiles[task]
        expected_ids = [candidate.negative_option_id, candidate.positive_option_id]
        if len(option_ids) != 2 or option_ids != expected_ids:
            mismatch_reason = "option_layout_mismatch"
        else:
            profile = candidate

    if profile is None:
        if require_calibrated:
            raise CalibrationError(
                f"Calibrated probabilities requested but unavailable: {mismatch_reason}",
                code="calibration_unavailable",
                param=decision_param,
            )
        probs = _stable_softmax(logits)
        probability_kind = "uncalibrated_softmax"
        choice_basis = "raw_argmax"
        calibration_meta: dict[str, Any] = {
            "profile": None,
            "applied": False,
            "mismatch_reason": mismatch_reason,
            "temperature": None,
            "bias": None,
            "balanced_threshold": None,
            "risk_threshold_5pct": None,
        }
    else:
        margin = (logits[1] - logits[0] + profile.bias) / profile.temperature
        p1 = _stable_sigmoid(margin)
        p0 = 1.0 - p1
        probs = [p0, p1]
        probability_kind = "calibrated_affine"
        choice_basis = "calibrated_argmax"
        calibration_meta = {
            "profile": profile.profile_id,
            "applied": True,
            "mismatch_reason": None,
            "temperature": profile.temperature,
            "bias": profile.bias,
            "balanced_threshold": profile.balanced_threshold,
            "risk_threshold_5pct": profile.risk_threshold_5pct,
            "selective_policy": profile.selective_policy,
        }

    choice_idx = _argmax_earliest(probs)
    expected_score: float | None = None
    if option_values is not None:
        expected_score = float(sum(p * v for p, v in zip(probs, option_values)))

    scores = [
        {"id": opt_id, "logit": float(logit), "probability": float(prob)}
        for opt_id, logit, prob in zip(option_ids, logits, probs)
    ]
    return {
        "choice": option_ids[choice_idx],
        "choice_basis": choice_basis,
        "scores": scores,
        "probability_kind": probability_kind,
        "expected_score": expected_score,
        "calibration": calibration_meta,
    }


def compaction_action(probability: float, *, balanced_threshold: float, skip_threshold: float,
                      selective_policy: bool = True) -> str:
    """Map calibrated P(compact) into compact, skip, or System-2 uncertainty zones."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and in [0, 1]")
    if not 0 <= skip_threshold <= balanced_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= skip <= balanced <= 1")
    if probability >= balanced_threshold:
        return "compact"
    if selective_policy and probability < skip_threshold:
        return "system1_skip"
    return "defer_to_system2"


def calibration_metrics(probabilities: list[float], labels: list[int], *, threshold: float,
                        skip_threshold: float | None = None) -> dict[str, float | int | None]:
    """Train-only binary metrics for a named calibration/acceptance partition."""
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must be nonempty and have equal length")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary")
    eps = 1e-15
    tp = sum(p >= threshold and y == 1 for p, y in zip(probabilities, labels))
    fp = sum(p >= threshold and y == 0 for p, y in zip(probabilities, labels))
    fn = sum(p < threshold and y == 1 for p, y in zip(probabilities, labels))
    predicted = sum(p >= threshold for p in probabilities)
    return {
        "count": len(labels), "skip_rate": sum(p < skip_threshold for p in probabilities) / len(labels) if skip_threshold is not None else None,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "brier": sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels),
        "nll": -sum(y * math.log(max(eps, p)) + (1-y) * math.log(max(eps, 1-p)) for p, y in zip(probabilities, labels)) / len(labels),
        "predicted_compact_rate": predicted / len(labels),
    }


def evaluate_train_partitions(partition_rows: dict[str, list[dict[str, Any]]], *,
                              balanced_threshold: float, skip_threshold: float) -> dict[str, Any]:
    """Evaluate only explicitly supplied train_calibration/train_acceptance scored rows."""
    allowed = {"train_calibration", "train_acceptance"}
    unexpected = set(partition_rows) - allowed
    if unexpected:
        raise ValueError(f"evaluation accepts train partitions only: {sorted(unexpected)}")
    result: dict[str, Any] = {}
    for partition, rows in partition_rows.items():
        if any(row.get("partition") != partition for row in rows):
            raise ValueError(f"row partition mismatch in {partition}")
        result[partition] = calibration_metrics(
            [float(row["probability"]) for row in rows],
            [int(row["label"]) for row in rows],
            threshold=balanced_threshold,
            skip_threshold=skip_threshold,
        )
    return {"source": "certified_train_only", "partitions": result}
