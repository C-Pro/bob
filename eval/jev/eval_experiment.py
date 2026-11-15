"""Calibrate on cal only; evaluate frozen experiments once on dev."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

try:
    from .model import QUESTIONS, SLICE_DIMS
    from .train_experiment import DATASET, HERE, make_model, read_rows, smoke_rows
except ImportError:
    from model import QUESTIONS, SLICE_DIMS
    from train_experiment import DATASET, HERE, make_model, read_rows, smoke_rows


def probabilities(logits, temperature=1.0, bias=0.0):
    values = np.asarray(logits, dtype=np.float64).copy()
    if values.shape[-1] == 2 and bias:
        values[..., 1] += bias
    values = values / temperature
    values -= values.max(axis=-1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=-1, keepdims=True)


def temperature_scale(logits, labels):
    if len(labels) == 0:
        raise ValueError("Temperature scaling needs calibration examples")
    labels = np.asarray(labels, dtype=int)
    candidates = np.exp(np.linspace(math.log(.05), math.log(20), 161))
    nll = [-np.log(probabilities(logits, t)[np.arange(len(labels)), labels].clip(1e-12)).mean() for t in candidates]
    return float(candidates[int(np.argmin(nll))])


def affine_scale(logits, labels):
    """Fit (temperature, binary option bias) strictly on calibration NLL."""
    if len(labels) == 0:
        raise ValueError("Calibration scaling needs calibration examples")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if logits.shape[-1] != 2:
        return temperature_scale(logits, labels), 0.0
    temps = np.exp(np.linspace(math.log(.05), math.log(20), 81))
    biases = np.linspace(-4.0, 4.0, 81)
    best_nll, best_t, best_b = math.inf, 1.0, 0.0
    for bias in biases:
        shifted = logits.copy()
        shifted[:, 1] += bias
        for temp in temps:
            nll = -np.log(probabilities(shifted, temp)[np.arange(len(labels)), labels].clip(1e-12)).mean()
            if nll < best_nll:
                best_nll, best_t, best_b = float(nll), float(temp), float(bias)
    # Local refinement around best grid cell.
    fine_t = np.exp(np.linspace(math.log(max(0.05, best_t / 1.5)), math.log(min(20.0, best_t * 1.5)), 41))
    fine_b = np.linspace(best_b - 0.25, best_b + 0.25, 41)
    for bias in fine_b:
        shifted = logits.copy()
        shifted[:, 1] += bias
        for temp in fine_t:
            nll = -np.log(probabilities(shifted, temp)[np.arange(len(labels)), labels].clip(1e-12)).mean()
            if nll < best_nll:
                best_nll, best_t, best_b = float(nll), float(temp), float(bias)
    return best_t, best_b


def balanced_decision_threshold(probs, labels):
    """Choose positive-class cutoff maximizing balanced accuracy on cal only."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if probs.shape[-1] != 2 or not (labels == 0).any() or not (labels == 1).any():
        return 0.5
    pos_scores = probs[:, 1]
    sorted_scores = np.sort(np.unique(pos_scores))
    mids = (sorted_scores[:-1] + sorted_scores[1:]) * 0.5 if len(sorted_scores) > 1 else sorted_scores
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], mids)))
    best_thresh, best_bal = 0.5, -1.0
    pos_mask, neg_mask = labels == 1, labels == 0
    for thresh in candidates:
        pred_pos = pos_scores >= thresh
        bal = 0.5 * (float(pred_pos[pos_mask].mean()) + float((~pred_pos)[neg_mask].mean()))
        if bal > best_bal + 1e-12 or (abs(bal - best_bal) <= 1e-12 and abs(thresh - 0.5) < abs(best_thresh - 0.5)):
            best_bal, best_thresh = bal, float(thresh)
    return best_thresh


def risk_threshold(probs, labels, max_error=.05):
    """Empirical calibration risk with whole confidence ties; no dev fitting."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidence = probs.max(-1)
    correct = probs.argmax(-1) == labels
    best_threshold, best_count, best_risk = 1.000001, 0, None
    for threshold in np.unique(confidence):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        risk = float(1 - correct[accepted].mean())
        if risk <= max_error + 1e-12 and count > best_count:
            best_threshold, best_count, best_risk = float(threshold), count, risk
    return {"threshold": best_threshold, "cal_coverage": best_count / len(labels),
            "cal_error": best_risk, "target_error": max_error,
            "guarantee": "empirical_calibration_only_not_a_population_risk_bound"}


def binary_ranking_metrics(pos_scores, labels):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    pos = pos_scores[labels == 1]
    neg = pos_scores[labels == 0]
    if not len(pos) or not len(neg):
        return None, None
    roc_auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    order = np.argsort(-pos_scores, kind="mergesort")
    sorted_scores = pos_scores[order]
    sorted_labels = labels[order]
    # Group tied scores so PR-AUC is order-invariant even on constant predictions.
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tps = np.cumsum(sorted_labels == 1)[distinct]
    total_counts = distinct + 1
    precisions = tps / total_counts
    recalls = tps / len(pos)
    prev_recalls = np.r_[0.0, recalls[:-1]]
    pr_auc = float(np.sum((recalls - prev_recalls) * precisions))
    return roc_auc, pr_auc


def paired_group_bootstrap_ci(pos_scores, labels, groups, n_boot=1000, seed=42):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    by_group = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(g, []).append(idx)
    group_ids = sorted(by_group)
    if len(group_ids) < 2:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(group_ids), size=len(group_ids))
        idx = np.concatenate([by_group[group_ids[g]] for g in picked])
        auc, _ = binary_ranking_metrics(pos_scores[idx], labels[idx])
        if auc is not None:
            aucs.append(auc)
    if len(aucs) < 10:
        return None
    low, high = np.percentile(aucs, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "valid_draws": len(aucs)}


def group_distributed_rows(rows, limit):
    if limit is None or limit <= 0 or len(rows) <= limit:
        return list(rows)
    by_group = {}
    for row in rows:
        by_group.setdefault(row["task_group"], []).append(row)
    selected = []
    ordered = [by_group[key] for key in sorted(by_group)]
    offset = 0
    while len(selected) < limit:
        progressed = False
        for bucket in ordered:
            if offset < len(bucket):
                selected.append(bucket[offset])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        offset += 1
    return selected


def classification_metrics(probs, labels, threshold=None, decision_cutoff=None):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if not len(labels):
        return {"count": 0}
    predicted = (probs[:, 1] >= decision_cutoff).astype(int) if (decision_cutoff is not None and probs.shape[1] == 2) else probs.argmax(-1)
    confidence = probs.max(-1)
    correct = predicted == labels
    per_class = {}
    f1s, recalls = [], []
    for label in range(probs.shape[1]):
        tp = int(((predicted == label) & (labels == label)).sum())
        support = int((labels == label).sum())
        precision = tp / max(1, int((predicted == label).sum()))
        recall = tp / max(1, support)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        if support:
            recalls.append(recall)
    onehot = np.eye(probs.shape[1])[labels]
    ece = 0.0
    bins = []
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
        count = int(mask.sum())
        accuracy = float((probs.argmax(-1) == labels)[mask].mean()) if count else None
        mean_conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - mean_conf)
        bins.append({"count": count, "accuracy": accuracy, "confidence": mean_conf})
    roc_auc, pr_auc = binary_ranking_metrics(probs[:, 1], labels) if probs.shape[1] == 2 else (None, None)
    result = {"count": len(labels), "accuracy": float(correct.mean()),
              "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
              "roc_auc": roc_auc, "pr_auc": pr_auc,
              "per_class": per_class, "brier": float(np.square(probs - onehot).sum(-1).mean()),
              "nll": float(-np.log(probs[np.arange(len(labels)), labels].clip(1e-12)).mean()),
              "ece_10bin": float(ece), "calibration_bins": bins}
    if threshold is not None:
        accepted = confidence >= threshold
        result["coverage_at_calibrated_5pct_error"] = float(accepted.mean())
        result["selective_error"] = float(1 - (probs.argmax(-1) == labels)[accepted].mean()) if accepted.any() else None
    return result


def decision_logits(model, rows, family, max_seq_len):
    question, options = QUESTIONS[family]
    with torch.no_grad():
        return np.asarray([model(r["state_text"], question, options, max_seq_len)["logits"].float().cpu().tolist() for r in rows])


def calibrated_evaluation(cal_logits, cal_rows, dev_logits, dev_rows):
    cal_labels = [row["label"] for row in cal_rows]
    temperature, bias = affine_scale(cal_logits, cal_labels)
    cal_probs = probabilities(cal_logits, temperature, bias)
    threshold = risk_threshold(cal_probs, cal_labels)
    bal_cutoff = balanced_decision_threshold(cal_probs, cal_labels)
    probs = probabilities(dev_logits, temperature, bias)
    labels = [row["label"] for row in dev_rows]
    bal_metrics = classification_metrics(probs, labels, threshold["threshold"], decision_cutoff=bal_cutoff)
    result = {"temperature": temperature, "bias": bias, "risk_policy": threshold,
              "balanced_threshold": bal_cutoff,
              "metrics": classification_metrics(probs, labels, threshold["threshold"]),
              "balanced_operating_metrics": bal_metrics,
              "length_buckets": {}}
    for name, low, high in (("<=512", 0, 512), ("513..1024", 513, 1024), (">1024", 1025, math.inf)):
        indices = [i for i, row in enumerate(dev_rows) if low <= row["token_count"] <= high]
        result["length_buckets"][name] = classification_metrics(probs[indices], [labels[i] for i in indices], threshold["threshold"])
    return result


def retrieval_metrics(scores, anchor_groups, gallery_groups):
    ranks = []
    for score, group in zip(scores, anchor_groups):
        order = np.argsort(-score, kind="stable")
        positives = [rank + 1 for rank, index in enumerate(order) if gallery_groups[index] == group]
        if not positives:
            raise ValueError("Retrieval anchor lacks a positive in the gallery")
        ranks.append(min(positives))
    ranks = np.asarray(ranks)
    return {"recall_at_1": float((ranks <= 1).mean()), "recall_at_5": float((ranks <= 5).mean()),
            "mrr": float((1 / ranks).mean())}


def words(text):
    return set(re.findall(r"\w+", text.casefold()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def embedding_evaluation(model, rows, max_seq_len):
    if not rows:
        return {"count": 0}
    gallery = list(dict.fromkeys([(r["positive_text"], r["task_group"]) for r in rows] +
                                 [(r["hard_negative_text"], r["hard_negative_task_group"]) for r in rows]))
    anchor_groups = [r["task_group"] for r in rows]
    gallery_groups = [group for _, group in gallery]
    anchor_words = [words(r["anchor_text"]) for r in rows]
    gallery_words = [words(text) for text, _ in gallery]
    lexical = np.asarray([[jaccard(a, g) for g in gallery_words] for a in anchor_words])
    with torch.no_grad():
        anchors = [model.embed_state(r["anchor_text"], max_seq_len) for r in rows]
        candidates = [model.embed_state(text, max_seq_len) for text, _ in gallery]
    result = {"count": len(rows), "gallery_size": len(gallery),
              "positive_rule": "any_same_task_group_gallery_item",
              "lexical_jaccard": retrieval_metrics(lexical, anchor_groups, gallery_groups), "slices": {}}
    positive_indices = [gallery.index((r["positive_text"], r["task_group"])) for r in rows]
    for dim in SLICE_DIMS:
        a = torch.stack([v[dim] for v in anchors]).cpu().numpy()
        g = torch.stack([v[dim] for v in candidates]).cpu().numpy()
        scores = a @ g.T
        metrics = retrieval_metrics(scores, anchor_groups, gallery_groups)
        metrics["alignment"] = float(np.square(a - g[positive_indices]).sum(-1).mean())
        pairwise = np.square(a[:, None, :] - a[None, :, :]).sum(-1)
        distances = pairwise[np.triu_indices(len(a), k=1)]
        metrics["uniformity"] = float(np.log(np.exp(-2 * distances).mean())) if len(distances) else None
        metrics["lexical_jaccard_pearson"] = float(np.corrcoef(scores.ravel(), lexical.ravel())[0, 1]) if scores.std() > 1e-12 and lexical.std() > 1e-12 else None
        result["slices"][str(dim)] = metrics
    return result


def latency_metrics(model, rows, max_seq_len, count=8):
    question, options = QUESTIONS["compaction"]
    reserve = len(model.question_ids(question, options)[0])
    full_times, state_times, question_times = [], [], []

    def sync():
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        # Warm up kernels without including startup in latency measurements.
        if rows:
            model(rows[0]["state_text"], question, options, max_seq_len)
        for row in rows[:count]:
            sync()
            start = time.perf_counter()
            model(row["state_text"], question, options, max_seq_len)
            sync()
            full_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            prefix = model.encode_prefix(row["state_text"], max_seq_len, reserve)
            sync()
            state_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            model.question_from_prefix(prefix, question, options)
            sync()
            question_times.append((time.perf_counter() - start) * 1000)
    return {"sample_count": len(full_times), "single_pass_ms_per_decision": float(np.mean(full_times)),
            "ms_per_state": float(np.mean(state_times)), "ms_per_question": float(np.mean(question_times)),
            "question_timing_includes_cache_clone": True,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if model.device.type == "cuda" else 0.0}


def baseline_logits(rows, name):
    if name == "always_compact":
        return np.tile([-12., 12.], (len(rows), 1))
    return np.asarray([[-12., 12.] if r["recorded_verdict"] == "probed_compact" else [12., -12.] for r in rows])


def write_report(report, directory, prefix="experiment_report"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{prefix}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    has_cv = any("raw_metrics" in d for v in report["models"].values() for d in v.get("decisions", {}).values())
    lines = ["# Jev experiment report", "", report["interpretation"], "",
             "Temperature, class bias, balanced cutoff, and empirical deferral thresholds were fitted on cal only. Checkpoints used a fixed training budget.", ""]
    if has_cv:
        lines += ["| Model | Task | Raw ROC-AUC | Within-Fold AUC | Cal-OOF AUC (diag) | Bal Acc (Cal Thresh) | Bal Acc (Nat) | Macro-F1 | Brier | NLL | ECE | Coverage |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        lines += ["| Model | Task | ROC-AUC | Bal Acc (Cal Thresh) | Bal Acc (Nat) | Macro-F1 | Brier | NLL | ECE | Coverage |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in report["models"].items():
        for family, decision in value.get("decisions", {}).items():
            m = decision["metrics"]
            bm = decision.get("balanced_operating_metrics", m)
            auc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "n/a"
            if has_cv:
                rm = decision.get("raw_metrics", {})
                w_auc = f"{rm['within_fold_roc_mean']:.4f}" if rm.get("within_fold_roc_mean") is not None else "n/a"
                c_auc = f"{m['pooled_calibrated_roc_auc_diag']:.4f}" if m.get("pooled_calibrated_roc_auc_diag") is not None else "n/a"
                lines.append(f"| {name} | {family} | {auc} | {w_auc} | {c_auc} | {bm['balanced_accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | {m['brier']:.4f} | {m['nll']:.4f} | {m['ece_10bin']:.4f} | {m['coverage_at_calibrated_5pct_error']:.4f} |")
            else:
                lines.append(f"| {name} | {family} | {auc} | {bm['balanced_accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} | {m['brier']:.4f} | {m['nll']:.4f} | {m['ece_10bin']:.4f} | {m['coverage_at_calibrated_5pct_error']:.4f} |")
    lines += ["", "Current-Policy reuses the teacher verdict that defines compaction labels; its score is circular, not independent policy validation.",
              "", "Coverage is measured on held-out folds/dev using the cal threshold; <=5% error is not guaranteed.",
              "", "All loop examples are none; the false-positive check cannot measure recall for actual loops.",
              "", f"Retrieval, per-class metrics, length buckets, latency, and memory details are in {prefix}.json."]
    (directory / f"{prefix}.md").write_text("\n".join(lines) + "\n")


def _classification_with_predictions(probs, labels, predicted, accepted, length_rows=None):
    base = classification_metrics(probs, labels)
    # Recompute thresholded balanced_operating_metrics from out-of-fold predictions cleanly.
    per_class = {}
    f1s, recalls = [], []
    for label in range(probs.shape[1]):
        tp = int(((predicted == label) & (labels == label)).sum())
        support = int((labels == label).sum())
        precision = tp / max(1, int((predicted == label).sum()))
        recall = tp / max(1, support)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[str(label)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        if support:
            recalls.append(recall)
    base["coverage_at_calibrated_5pct_error"] = float(accepted.mean()) if len(labels) else 0.0
    base["selective_error"] = float(1 - (probs.argmax(-1) == labels)[accepted].mean()) if accepted.any() else None
    bal_metrics = {
        "count": len(labels),
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "roc_auc": base["roc_auc"],
        "pr_auc": base["pr_auc"],
        "per_class": per_class,
        "coverage_at_calibrated_5pct_error": base["coverage_at_calibrated_5pct_error"],
        "selective_error": base["selective_error"],
    }
    buckets = {}
    if length_rows is not None:
        for name, low, high in (("<=512", 0, 512), ("513..1024", 513, 1024), (">1024", 1025, math.inf)):
            indices = [i for i, row in enumerate(length_rows) if low <= row["token_count"] <= high]
            m = classification_metrics(probs[indices], [labels[i] for i in indices])
            if indices:
                acc_i = accepted[indices]
                m["coverage_at_calibrated_5pct_error"] = float(acc_i.mean())
                m["selective_error"] = float(1 - (probs[indices].argmax(-1) == labels[indices])[acc_i].mean()) if acc_i.any() else None
            buckets[name] = m
    return base, bal_metrics, buckets


def cal_cv_evaluation(logits, rows, n_folds=5):
    """5-fold task-group-disjoint out-of-fold calibration & evaluation strictly on cal."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    groups = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row["task_group"], []).append(idx)
    ordered_groups = sorted(groups, key=lambda g: (sum(labels[i] for i in groups[g]) / len(groups[g]), g))
    n_folds = min(n_folds, len(ordered_groups))
    folds = [[] for _ in range(n_folds)]
    for pos, group in enumerate(ordered_groups):
        folds[pos % n_folds].extend(groups[group])

    oof_probs = np.zeros_like(logits)
    oof_bal_pred = np.zeros(len(rows), dtype=int)
    oof_accepted = np.zeros(len(rows), dtype=bool)
    temps, biases, cutoffs = [], [], []
    raw_margins = logits[:, 1] - logits[:, 0] if logits.shape[-1] == 2 else logits.max(-1)
    within_fold_aucs, fold_supports = [], []
    for val_idx in folds:
        val_set = set(val_idx)
        tr_idx = [i for i in range(len(rows)) if i not in val_set]
        tr_labels = labels[tr_idx]
        t, b = affine_scale(logits[tr_idx], tr_labels)
        tr_probs = probabilities(logits[tr_idx], t, b)
        thresh = risk_threshold(tr_probs, tr_labels)
        bal_cut = balanced_decision_threshold(tr_probs, tr_labels)
        fold_probs = probabilities(logits[val_idx], t, b)
        oof_probs[val_idx] = fold_probs
        oof_bal_pred[val_idx] = (fold_probs[:, 1] >= bal_cut).astype(int) if fold_probs.shape[1] == 2 else fold_probs.argmax(-1)
        oof_accepted[val_idx] = fold_probs.max(-1) >= thresh["threshold"]
        temps.append(t)
        biases.append(b)
        cutoffs.append(bal_cut)
        f_auc, _ = binary_ranking_metrics(raw_margins[val_idx], labels[val_idx])
        within_fold_aucs.append(f_auc)
        fold_supports.append({"count": len(val_idx), "neg": int((labels[val_idx] == 0).sum()), "pos": int((labels[val_idx] == 1).sum())})

    base, bal_metrics, buckets = _classification_with_predictions(oof_probs, labels, oof_bal_pred, oof_accepted, rows)
    raw_roc, raw_pr = binary_ranking_metrics(raw_margins, labels)
    base["pooled_calibrated_roc_auc_diag"] = base["roc_auc"]
    base["pooled_calibrated_pr_auc_diag"] = base["pr_auc"]
    base["roc_auc"] = raw_roc
    base["pr_auc"] = raw_pr
    bal_metrics["roc_auc"] = raw_roc
    bal_metrics["pr_auc"] = raw_pr
    for b_metrics in buckets.values():
        if b_metrics.get("count"):
            idx = [i for i, row in enumerate(rows) if b_metrics is buckets["<=512"] and row["token_count"] <= 512
                   or b_metrics is buckets["513..1024"] and 513 <= row["token_count"] <= 1024
                   or b_metrics is buckets[">1024"] and row["token_count"] > 1024]
            b_roc, b_pr = binary_ranking_metrics(raw_margins[idx], labels[idx])
            b_metrics["pooled_calibrated_roc_auc_diag"] = b_metrics["roc_auc"]
            b_metrics["roc_auc"] = b_roc
            b_metrics["pr_auc"] = b_pr
    per_benchmark = {}
    for bench in sorted({row.get("benchmark", "unknown") for row in rows}):
        b_idx = [i for i, row in enumerate(rows) if row.get("benchmark", "unknown") == bench]
        b_labels = labels[b_idx]
        b_margins = raw_margins[b_idx]
        b_roc, b_pr = binary_ranking_metrics(b_margins, b_labels)
        per_benchmark[bench] = {
            "count": len(b_idx),
            "neg": int((b_labels == 0).sum()),
            "pos": int((b_labels == 1).sum()),
            "roc_auc": b_roc,
            "pr_auc": b_pr,
            "mean_margin_0": float(b_margins[b_labels == 0].mean()) if (b_labels == 0).any() else None,
            "mean_margin_1": float(b_margins[b_labels == 1].mean()) if (b_labels == 1).any() else None,
        }
    valid_fold_aucs = [a for a in within_fold_aucs if a is not None]
    raw_metrics = {
        "roc_auc": raw_roc,
        "pr_auc": raw_pr,
        "within_fold_roc_aucs": within_fold_aucs,
        "within_fold_roc_mean": float(np.mean(valid_fold_aucs)) if valid_fold_aucs else None,
        "fold_supports": fold_supports,
        "per_benchmark": per_benchmark,
        "bootstrap_ci_95": paired_group_bootstrap_ci(raw_margins, labels, [row["task_group"] for row in rows]),
    }
    return {
        "cv_folds": n_folds,
        "temperature_mean": float(np.mean(temps)),
        "bias_mean": float(np.mean(biases)),
        "balanced_threshold_mean": float(np.mean(cutoffs)),
        "raw_metrics": raw_metrics,
        "metrics": base,
        "balanced_operating_metrics": bal_metrics,
        "length_buckets": buckets,
    }


def run_cal_cv_evaluation(args):
    """Evaluate checkpoints strictly using 5-fold CV on cal; dev is never read."""
    output = Path(args.out_dir)
    if args.smoke:
        torch.set_num_threads(1)
        args.max_seq_len = min(args.max_seq_len, 512)
    families = ("compaction", "outcome_prefix", "emb_pairs", "loop_guard")
    cal_data = smoke_rows("cal") if args.smoke else {
        family: read_rows(args.dataset, family, "cal", args.max_eval or None) for family in families
    }
    if any(not cal_data[family] for family in families):
        raise ValueError("Calibration CV requires nonempty cal splits")
    report = {
        "smoke": args.smoke,
        "models": {},
        "calibration_split": "cal",
        "evaluation_split": "cal_5fold_oof",
        "interpretation": "Out-of-fold 5-fold grouped CV strictly on cal; dev remains untouched.",
        "max_seq_len": args.max_seq_len,
        "max_eval": args.max_eval,
        "device": args.device,
        "data_sha256": {"cal": hashlib.sha256(json.dumps(cal_data, sort_keys=True).encode()).hexdigest()},
    }
    for name in ("always_compact", "current_policy"):
        cal = cal_data["compaction"]
        report["models"][name] = {"decisions": {"compaction": cal_cv_evaluation(baseline_logits(cal, name), cal)}}
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[args.dtype]
    model_names = getattr(args, "models", None) or ("zero_shot_kev", "exp_a", "exp_b")
    for name in model_names:
        checkpoint = Path(args.checkpoint_root) / name / "model.pt"
        if name != "zero_shot_kev" and not checkpoint.exists():
            report["models"][name] = {"status": "checkpoint_missing", "checkpoint": str(checkpoint)}
            continue
        model = make_model(args.smoke, args.device, dtype, args.seed)
        checkpoint_metadata = model.load_checkpoint(checkpoint) if name != "zero_shot_kev" else None
        if checkpoint_metadata and checkpoint_metadata["smoke"] != args.smoke:
            raise ValueError("Cannot mix synthetic and real checkpoints")
        model.eval()
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = {
            "decisions": {},
            "checkpoint_metadata": checkpoint_metadata,
            "kev_original_temperature": model.temperature,
            "embedding_head": "untrained_random_projection" if name == "zero_shot_kev" else "trained",
        }
        for family in ("compaction", "outcome_prefix"):
            cal = cal_data[family]
            logits = decision_logits(model, cal, family, args.max_seq_len)
            result["decisions"][family] = cal_cv_evaluation(logits, cal)
        with torch.no_grad():
            result["embeddings"] = embedding_evaluation(model, cal_data["emb_pairs"], args.max_seq_len)
            loop_rows = group_distributed_rows(cal_data["loop_guard"], args.loop_limit)
            loop_logits = decision_logits(model, loop_rows, "loop_guard", args.max_seq_len)
            normal = np.asarray([r["label"] == "none" for r in loop_rows])
            result["loop_guard"] = {
                "count": len(loop_rows),
                "group_count": len({r["task_group"] for r in loop_rows}),
                "negative_count": int(normal.sum()),
                "false_positive_rate": float((loop_logits.argmax(-1)[normal] != 0).mean()) if normal.any() else None,
                "positive_recall": None,
                "limitation": "all-none dataset cannot validate true-loop detection",
            }
            result["latency"] = latency_metrics(model, cal_data["compaction"], args.max_seq_len)
        report["models"][name] = result
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_report(report, output, prefix="cal_cv_report")
    return report


def run_evaluation(args):
    output = Path(args.out_dir)
    if (output / "experiment_report.json").exists() or (output / "dev_evaluation_started.json").exists():
        raise FileExistsError("Dev evaluation already started or completed in this output directory; use a fresh experiment directory")
    if args.smoke:
        torch.set_num_threads(1)
        args.max_seq_len = min(args.max_seq_len, 512)
    families = ("compaction", "outcome_prefix", "emb_pairs", "loop_guard")
    data = {split: smoke_rows(split) if args.smoke else {
        family: read_rows(args.dataset, family, split, args.max_eval or None) for family in families}
        for split in ("cal", "dev")}
    if any(not data[split][family] for split in data for family in families):
        raise ValueError("Evaluation requires nonempty calibration and dev splits")
    cal_groups = {r["task_group"] for rows in data["cal"].values() for r in rows}
    dev_groups = {r["task_group"] for rows in data["dev"].values() for r in rows}
    if cal_groups & dev_groups:
        raise ValueError("Calibration and dev task groups overlap")
    report = {"smoke": args.smoke, "models": {}, "calibration_split": "cal", "evaluation_split": "dev",
              "interpretation": "Synthetic smoke verification only; not benchmark performance." if args.smoke else "Fixed-checkpoint held-out dev evaluation.",
              "max_seq_len": args.max_seq_len, "max_eval": args.max_eval,
              "device": args.device, "inference": "single_pass_uniform_delimiters_canonical_option_order",
              "data_sha256": {split: hashlib.sha256(json.dumps(data[split], sort_keys=True).encode()).hexdigest() for split in data}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "dev_evaluation_started.json").write_text(json.dumps({"data_sha256": report["data_sha256"]}, indent=2) + "\n")
    for name in ("always_compact", "current_policy"):
        cal, dev = data["cal"]["compaction"], data["dev"]["compaction"]
        report["models"][name] = {"decisions": {"compaction": calibrated_evaluation(
            baseline_logits(cal, name), cal, baseline_logits(dev, name), dev)}}
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[args.dtype]
    model_names = getattr(args, "models", None) or ("zero_shot_kev", "exp_a", "exp_b")
    for name in model_names:
        checkpoint = Path(args.checkpoint_root) / name / "model.pt"
        if name != "zero_shot_kev" and not checkpoint.exists():
            report["models"][name] = {"status": "checkpoint_missing", "checkpoint": str(checkpoint)}
            continue
        model = make_model(args.smoke, args.device, dtype, args.seed)
        checkpoint_metadata = model.load_checkpoint(checkpoint) if name != "zero_shot_kev" else None
        if checkpoint_metadata and checkpoint_metadata["smoke"] != args.smoke:
            raise ValueError("Cannot mix synthetic and real checkpoints")
        model.eval()
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        result = {"decisions": {}, "checkpoint_metadata": checkpoint_metadata,
                  "kev_original_temperature": model.temperature,
                  "embedding_head": "untrained_random_projection" if name == "zero_shot_kev" else "trained"}
        # Freeze every calibration policy before computing dev predictions.
        policies = {}
        for family in ("compaction", "outcome_prefix"):
            cal = data["cal"][family]
            logits = decision_logits(model, cal, family, args.max_seq_len)
            labels = [r["label"] for r in cal]
            temp, bias = affine_scale(logits, labels)
            cal_probs = probabilities(logits, temp, bias)
            policies[family] = (logits, temp, bias, risk_threshold(cal_probs, labels), balanced_decision_threshold(cal_probs, labels))
        for family in ("compaction", "outcome_prefix"):
            dev = data["dev"][family]
            dev_logits = decision_logits(model, dev, family, args.max_seq_len)
            logits, temp, bias, threshold, bal_cutoff = policies[family]
            decision = calibrated_evaluation(logits, data["cal"][family], dev_logits, dev)
            assert (decision["temperature"] == temp and decision["bias"] == bias
                    and decision["risk_policy"] == threshold and decision["balanced_threshold"] == bal_cutoff)
            result["decisions"][family] = decision
        with torch.no_grad():
            result["embeddings"] = embedding_evaluation(model, data["dev"]["emb_pairs"], args.max_seq_len)
            loop_rows = data["dev"]["loop_guard"][:args.loop_limit]
            loop_logits = decision_logits(model, loop_rows, "loop_guard", args.max_seq_len)
            none_index = 0
            normal = np.asarray([r["label"] == "none" for r in loop_rows])
            result["loop_guard"] = {"count": len(loop_rows), "negative_count": int(normal.sum()),
                "false_positive_rate": float((loop_logits.argmax(-1)[normal] != none_index).mean()) if normal.any() else None,
                "positive_recall": None, "limitation": "all-none dataset cannot validate true-loop detection"}
            result["latency"] = latency_metrics(model, data["dev"]["compaction"], args.max_seq_len)
        report["models"][name] = result
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_report(report, output)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--out-dir", type=Path, default=DATASET)
    p.add_argument("--checkpoint-root", type=Path, default=HERE / "checkpoints")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16", "bfloat16"), default="bf16")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--max-eval", type=int, default=0)
    p.add_argument("--loop-limit", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--cal-cv", action="store_true", help="Evaluate strictly via 5-fold grouped CV on cal without reading dev")
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if min(args.max_seq_len, args.loop_limit) <= 0 or args.max_eval < 0:
        p.error("Sequence length and loop limit must be positive; max-eval must be nonnegative")
    if args.cal_cv:
        report = run_cal_cv_evaluation(args)
        print(json.dumps({"report": str(Path(args.out_dir) / "cal_cv_report.json"),
                          "models": list(report["models"]), "smoke": report["smoke"]}, indent=2))
    else:
        report = run_evaluation(args)
        print(json.dumps({"report": str(Path(args.out_dir) / "experiment_report.json"),
                          "models": list(report["models"]), "smoke": report["smoke"]}, indent=2))
