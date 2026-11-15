"""Fixed-epoch Jev experiments. Training never reads cal or dev data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

try:
    from .model import DualHeadDecisionEmbedder, QUESTIONS, SLICE_DIMS, synthetic_model
except ImportError:
    from model import DualHeadDecisionEmbedder, QUESTIONS, SLICE_DIMS, synthetic_model

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "results/jev_dataset"


def read_rows(directory, family, split, limit=None):
    rows = []
    with (Path(directory) / f"{family}_{split}.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError(f"Unexpected split in {family}_{split}.jsonl")
            if family == "emb_pairs":
                if row["task_group"] == row["hard_negative_task_group"] or row["hard_negative_split"] != split:
                    raise ValueError("Unsafe embedding negative: shared group or different split")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def smoke_rows(split):
    result = {family: [] for family in (*QUESTIONS, "emb_pairs")}
    for i in range(4):
        group = f"{split}_{i}"
        common = {"task_group": group, "task_id": group, "event_id": f"event_{i}", "split": split,
                  "token_count": 20, "task_balanced_weight": 1.0}
        state = ("Tests pass; verified checkpoint." if i % 2 else "Tests fail; investigation continues.")
        for family in ("compaction", "outcome_prefix"):
            result[family].append({**common, "state_text": state, "label": i % 2,
                                   "recorded_verdict": "probed_compact" if i % 2 else "probed_skip"})
        result["loop_guard"].append({**common, "state_text": state, "label": "none"})
        result["emb_pairs"].append({**common, "anchor_text": state,
                                    "positive_text": state + " Continue from here.",
                                    "hard_negative_text": "Inspect an unrelated binary file.",
                                    "positive_task_group": group,
                                    "hard_negative_task_group": f"{split}_negative_{i}",
                                    "hard_negative_split": split})
    return result


def class_balanced_weights(rows):
    mass = [sum(float(r.get("task_balanced_weight", 1)) for r in rows if r["label"] == label) for label in (0, 1)]
    if not all(mass):
        raise ValueError("Decision training requires both classes")
    return [float(r.get("task_balanced_weight", 1)) * len(rows) / (2 * mass[r["label"]]) for r in rows]


def per_class_unit_weights(rows):
    """Normalize task_balanced_weight within each class so balanced sampling keeps mean weight 1.0."""
    by_label = {label: [float(r.get("task_balanced_weight", 1)) for r in rows if r["label"] == label] for label in (0, 1)}
    if not all(by_label.values()):
        raise ValueError("Decision training requires both classes")
    means = {label: sum(vals) / len(vals) for label, vals in by_label.items()}
    return [float(r.get("task_balanced_weight", 1)) / means[r["label"]] for r in rows]


class _GroupCycle:
    """Cycles task groups in shuffled order without repeating a group before visiting others."""

    def __init__(self, group_to_indices, rng):
        self.group_to_indices = {g: list(idxs) for g, idxs in group_to_indices.items()}
        self.rng = rng
        self.queues = {}
        self.group_order = []
        self.pos = 0

    def next(self):
        if self.pos >= len(self.group_order):
            self.group_order = sorted(self.group_to_indices)
            self.rng.shuffle(self.group_order)
            self.pos = 0
        group = self.group_order[self.pos]
        self.pos += 1
        if not self.queues.get(group):
            pool = list(self.group_to_indices[group])
            self.rng.shuffle(pool)
            self.queues[group] = pool
        return self.queues[group].pop()


def _group_buckets(rows, indices=None):
    buckets = {}
    iterable = range(len(rows)) if indices is None else indices
    for idx in iterable:
        buckets.setdefault(rows[idx]["task_group"], []).append(idx)
    return buckets


def balanced_group_indices(rows, rng, total_steps):
    c0 = _GroupCycle(_group_buckets(rows, [i for i, r in enumerate(rows) if r["label"] == 0]), rng)
    c1 = _GroupCycle(_group_buckets(rows, [i for i, r in enumerate(rows) if r["label"] == 1]), rng)
    if not c0.group_to_indices or not c1.group_to_indices:
        raise ValueError("Balanced group sampling requires both classes")
    result = []
    for i in range(total_steps):
        result.append(c0.next() if i % 2 == 0 else c1.next())
    return result


def compaction_ratio_indices(rows, rng, total_steps):
    """Benchmark-balanced, label-balanced, group-cycling compaction sequence (75% SWE / 25% CTF)."""
    pairs = {}
    for idx, row in enumerate(rows):
        bench = row.get("benchmark", "unknown")
        pairs.setdefault((bench, row["label"]), {}).setdefault(row["task_group"], []).append(idx)
    paired_benches = [b for b in sorted({r.get("benchmark", "unknown") for r in rows})
                      if (b, 0) in pairs and (b, 1) in pairs]
    if "swe-bench-lite" in paired_benches and "intercode-ctf" in paired_benches:
        pattern = [
            ("swe-bench-lite", 0), ("swe-bench-lite", 1), ("swe-bench-lite", 0), ("intercode-ctf", 1),
            ("swe-bench-lite", 1), ("swe-bench-lite", 0), ("swe-bench-lite", 1), ("intercode-ctf", 0),
        ]
    elif paired_benches:
        pattern = [(b, label) for b in paired_benches for label in (0, 1)]
    else:
        return balanced_group_indices(rows, rng, total_steps)
    cycles = {key: _GroupCycle(pairs[key], rng) for key in set(pattern)}
    return [cycles[pattern[i % len(pattern)]].next() for i in range(total_steps)]


def distinct_group_emb_batch(rows, cycle, emb_batch_size):
    target = min(emb_batch_size, len(cycle.group_to_indices))
    batch, seen = [], set()
    attempts = 0
    while len(batch) < target and attempts < max(len(cycle.group_to_indices) * 2, target * 4):
        idx = cycle.next()
        attempts += 1
        group = rows[idx]["task_group"]
        if group not in seen:
            seen.add(group)
            batch.append(rows[idx])
    while len(batch) < min(emb_batch_size, len(rows)):
        batch.append(rows[cycle.next()])
    return batch


def option_order(count, seed, epoch, row_index):
    order = list(range(count))
    random.Random(f"{seed}:{epoch}:{row_index}").shuffle(order)
    return order


def decision_key(family, row, order, max_seq_len, teacher_tag=None):
    content = [family, row["state_text"], order, max_seq_len, QUESTIONS[family]]
    if teacher_tag:
        content.append(teacher_tag)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def decision_forward(model, family, row, order, max_seq_len):
    question, options = QUESTIONS[family]
    shuffled = [options[i] for i in order]
    return model(row["state_text"], question, shuffled, max_seq_len)["logits"]


def iter_scheduled_steps(data, args):
    """Deterministic step schedule shared by reference cache generation and run_training."""
    schedule_mode = getattr(args, "schedule_mode", "legacy")
    kl_loop_weight = float(getattr(args, "kl_loop_weight", 0) or 0)
    loop_ce_weight = float(getattr(args, "loop_ce_weight", 0) or 0)
    include_loop = bool((kl_loop_weight or loop_ce_weight) and data.get("loop_guard"))
    emb_batch_size = getattr(args, "emb_batch_size", 1)
    balanced = getattr(args, "balanced_sampling", False)
    seed = getattr(args, "seed", 42)
    epochs = getattr(args, "epochs", 1)
    max_steps = getattr(args, "max_steps", 0)

    if schedule_mode == "ratio_4_2_1_1":
        if not all(data.get(f) for f in ("compaction", "outcome_prefix", "emb_pairs", "loop_guard")):
            raise ValueError("ratio_4_2_1_1 schedule requires all four task families")
        total_slots = max_steps or 1600
        block = ["compaction", "compaction", "outcome_prefix", "compaction", "emb_pairs", "compaction", "outcome_prefix", "loop_guard"]
        rng = random.Random(seed)
        c_need = sum(1 for i in range(total_slots) if block[i % len(block)] == "compaction")
        o_need = sum(1 for i in range(total_slots) if block[i % len(block)] == "outcome_prefix")
        c_list = compaction_ratio_indices(data["compaction"], rng, c_need)
        o_list = balanced_group_indices(data["outcome_prefix"], rng, o_need)
        e_cycle = _GroupCycle(_group_buckets(data["emb_pairs"]), rng)
        l_cycle = _GroupCycle(_group_buckets(data["loop_guard"]), rng)
        cursors = {"compaction": 0, "outcome_prefix": 0}
        for step_idx in range(1, total_slots + 1):
            family = block[(step_idx - 1) % len(block)]
            if family == "compaction":
                index = c_list[cursors["compaction"]]
                cursors["compaction"] += 1
                row = data[family][index]
                batch, order = None, option_order(len(QUESTIONS[family][1]), seed, 0, step_idx)
            elif family == "outcome_prefix":
                index = o_list[cursors["outcome_prefix"]]
                cursors["outcome_prefix"] += 1
                row = data[family][index]
                batch, order = None, option_order(len(QUESTIONS[family][1]), seed, 0, step_idx)
            elif family == "emb_pairs":
                batch = distinct_group_emb_batch(data["emb_pairs"], e_cycle, emb_batch_size)
                index, row, order = 0, batch[0], None
            else:
                index = l_cycle.next()
                row = data[family][index]
                batch, order = None, option_order(len(QUESTIONS[family][1]), seed, 0, step_idx)
            yield {
                "step": step_idx,
                "epoch": 0,
                "family": family,
                "index": index,
                "row": row,
                "order": order,
                "batch": batch,
            }
        return

    families = ("compaction", "outcome_prefix", "emb_pairs", "loop_guard") if include_loop else ("compaction", "outcome_prefix", "emb_pairs")
    if not all(data.get(f) for f in ("compaction", "outcome_prefix", "emb_pairs")):
        raise ValueError("All three training tasks need examples")
    total_steps = 0
    for epoch in range(epochs):
        rng = random.Random(seed + epoch)
        indices = {}
        for family in families:
            if balanced and family in ("compaction", "outcome_prefix"):
                c0 = [i for i, r in enumerate(data[family]) if r["label"] == 0]
                c1 = [i for i, r in enumerate(data[family]) if r["label"] == 1]
                if not c0 or not c1:
                    raise ValueError(f"Balanced sampling requires both classes in {family}")
                rng.shuffle(c0)
                rng.shuffle(c1)
                pairs = max(len(c0), len(c1))
                interleaved = []
                for k in range(pairs):
                    interleaved.append(c0[k % len(c0)])
                    interleaved.append(c1[k % len(c1)])
                indices[family] = interleaved
            else:
                values = list(range(len(data[family])))
                rng.shuffle(values)
                indices[family] = values
        core_rounds = max(len(indices[f]) for f in ("compaction", "outcome_prefix", "emb_pairs"))
        for round_index in range(core_rounds):
            for family in families:
                index = indices[family][round_index % len(indices[family])]
                row = data[family][index]
                if family == "emb_pairs":
                    batch = [data[family][indices[family][(round_index * emb_batch_size + k) % len(indices[family])]]
                             for k in range(min(emb_batch_size, len(indices[family])))]
                    order = None
                else:
                    batch = None
                    opt_count = len(QUESTIONS[family][1])
                    order = option_order(opt_count, seed, epoch, round_index * len(data[family]) + index if balanced else index)
                total_steps += 1
                yield {
                    "step": total_steps,
                    "epoch": epoch,
                    "family": family,
                    "index": index,
                    "row": row,
                    "order": order,
                    "batch": batch,
                }
                if max_steps and total_steps >= max_steps:
                    return


def teacher_signature(checkpoint_path):
    if checkpoint_path is None:
        return "pristine_kev:prompt-v1"
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    return f"ckpt:{path}:{stat.st_size}:{stat.st_mtime_ns}:prompt-v1"


def frozen_reference_cache(model, data, args, path, family_teachers=None):
    """Cache only CPU decision logits, fingerprinted by teacher checkpoint and prompt format."""
    path = Path(path)
    existing_cache = {}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") in ("jev-kev-reference-v1", "jev-kev-reference-v2") and saved.get("smoke") == getattr(args, "smoke", False):
            existing_cache = saved.get("logits", {})
    family_teachers = family_teachers or {}
    kl_comp = getattr(args, "kl_compaction_weight", None)
    kl_out = getattr(args, "kl_outcome_weight", None)
    kl_ref_raw = getattr(args, "kl_ref_weight", 0.1)
    kl_ref = 0.0 if kl_ref_raw is None else float(kl_ref_raw)
    kl_loop = float(getattr(args, "kl_loop_weight", 0) or 0)
    need_comp = (kl_comp if kl_comp is not None else kl_ref) > 0
    need_out = (kl_out if kl_out is not None else kl_ref) > 0
    target_families = tuple(
        f for f, active in (("compaction", need_comp), ("outcome_prefix", need_out), ("loop_guard", kl_loop > 0)) if active
    )
    if not target_families:
        return {}
    cache = {}
    by_teacher = {}
    for step in iter_scheduled_steps(data, args):
        family = step["family"]
        if family not in target_families:
            continue
        ckpt_path = family_teachers.get(family)
        tag = teacher_signature(ckpt_path)
        row, order = step["row"], step["order"]
        orders = [order]
        if family != "loop_guard" and getattr(args, "perm_kl", 0):
            orders.append(list(reversed(order)))
        for ord_ in orders:
            cache_key = decision_key(family, row, ord_, args.max_seq_len, tag if ckpt_path is not None else None)
            if cache_key in cache:
                continue
            if cache_key in existing_cache:
                cache[cache_key] = existing_cache[cache_key]
            else:
                by_teacher.setdefault(ckpt_path, []).append((cache_key, family, row, ord_))

    for ckpt_path, pending_items in by_teacher.items():
        if ckpt_path is None:
            teacher_model = model
        else:
            dtype = next(model.parameters()).dtype
            teacher_model = make_model(getattr(args, "smoke", False), str(model.device), dtype, getattr(args, "seed", 42))
            teacher_model.load_checkpoint(ckpt_path)
        teacher_model.eval()
        with torch.no_grad():
            for cache_key, family, row, ord_ in pending_items:
                cache[cache_key] = decision_forward(teacher_model, family, row, ord_, args.max_seq_len).float().cpu()
        if ckpt_path is not None:
            del teacher_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "jev-kev-reference-v2" if any(v is not None for v in family_teachers.values()) else "jev-kev-reference-v1"
    torch.save({"format": fmt, "smoke": getattr(args, "smoke", False), "logits": cache}, path)
    return cache


def infonce_from_embeddings(anchors, positives, negatives, rows, device, temperature=.07):
    """Matryoshka InfoNCE loss over precomputed per-slice normalized embeddings."""
    if not rows:
        raise ValueError("Embedding batch must not be empty")
    groups = [r["task_group"] for r in rows]
    candidate_groups = groups + [r["hard_negative_task_group"] for r in rows]
    if any(r["task_group"] == r["hard_negative_task_group"] for r in rows):
        raise ValueError("Hard negative shares anchor task group")
    weights = torch.tensor([r.get("task_balanced_weight", 1.0) for r in rows], device=device)
    losses = []
    for dim in SLICE_DIMS:
        a = torch.stack([v[dim] for v in anchors])
        candidates = torch.stack([v[dim] for v in positives + negatives])
        logits = a @ candidates.T / temperature
        invalid = torch.tensor([[g == h and j != i for j, h in enumerate(candidate_groups)]
                                for i, g in enumerate(groups)], device=device)
        logits = logits.masked_fill(invalid, float("-inf"))
        loss = F.cross_entropy(logits, torch.arange(len(rows), device=device), reduction="none")
        losses.append((loss * weights).mean())
    return torch.stack(losses).mean()


def matryoshka_infonce(model, rows, max_seq_len, temperature=.07, detach_backbone=False):
    """Explicit hard negatives plus in-batch positives/negatives with group masks."""
    if not rows:
        raise ValueError("Embedding batch must not be empty")
    anchors = [model.embed_state(r["anchor_text"], max_seq_len, detach_backbone=detach_backbone) for r in rows]
    positives = [model.embed_state(r["positive_text"], max_seq_len, detach_backbone=detach_backbone) for r in rows]
    negatives = [model.embed_state(r["hard_negative_text"], max_seq_len, detach_backbone=detach_backbone) for r in rows]
    return infonce_from_embeddings(anchors, positives, negatives, rows, model.device, temperature)


def matryoshka_infonce_grad_cache(model, rows, max_seq_len, loss_scale=1.0, temperature=.07, detach_backbone=False):
    """Two-pass gradient-cached Matryoshka InfoNCE: batch-B negatives at batch-1 VRAM."""
    if not rows:
        raise ValueError("Embedding batch must not be empty")
    texts = ([r["anchor_text"] for r in rows] +
             [r["positive_text"] for r in rows] +
             [r["hard_negative_text"] for r in rows])
    with torch.no_grad():
        cached = [model.embed_state(text, max_seq_len) for text in texts]
    leaves = [{dim: vec[dim].detach().requires_grad_(True) for dim in SLICE_DIMS} for vec in cached]
    n = len(rows)
    loss = loss_scale * infonce_from_embeddings(leaves[:n], leaves[n:2 * n], leaves[2 * n:], rows, model.device, temperature)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite emb_pairs loss")
    loss.backward()
    for text, leaf in zip(texts, leaves):
        terms = [dim for dim in SLICE_DIMS if leaf[dim].grad is not None]
        if not terms:
            continue
        emb = model.embed_state(text, max_seq_len, detach_backbone=detach_backbone)
        surrogate = sum((emb[dim] * leaf[dim].grad).sum() for dim in terms)
        surrogate.backward()
    return float(loss.detach() / loss_scale) if loss_scale else float(loss.detach())


def symmetric_kl(logits, other):
    p, q = logits.log_softmax(-1), other.log_softmax(-1)
    return .5 * (F.kl_div(p, q.exp(), reduction="sum") + F.kl_div(q, p.exp(), reduction="sum"))


def make_lr_scheduler(optimizer, total_updates, cosine_lr=False, warmup_steps=0):
    if not cosine_lr and not warmup_steps:
        return None
    total_updates = max(1, total_updates)
    warmup = min(warmup_steps, total_updates)

    def lr_lambda(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if not cosine_lr:
            return 1.0
        denom = max(1, total_updates - warmup)
        progress = min(1.0, max(0.0, (step - warmup) / denom))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_model(smoke, device, dtype, seed):
    torch.manual_seed(seed)
    if smoke:
        return synthetic_model(seed=seed, with_lora=True).to(device=device)
    return DualHeadDecisionEmbedder.from_pretrained(device=device, dtype=dtype, trainable_adapter=True)


def _grad_norm(parameters):
    norms = [p.grad.detach().float().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)))


def _run_emb_only(model, data, args, run_dir, checkpoint_dir, started):
    run_name = getattr(args, "run_name", None) or args.mode
    if getattr(args, "init_checkpoint", None):
        model.load_checkpoint(args.init_checkpoint)
    model.configure_training("emb_only", gradient_checkpointing=False)
    model.eval()
    rows = data["emb_pairs"]
    unique_texts = list(dict.fromkeys(
        [r["anchor_text"] for r in rows] + [r["positive_text"] for r in rows] + [r["hard_negative_text"] for r in rows]
    ))
    hidden_cache = {}
    with torch.no_grad():
        for text in unique_texts:
            hidden_cache[text] = model.state_hidden(text, args.max_seq_len).detach().float().cpu()
    if model.device.type == "cuda":
        torch.cuda.empty_cache()

    emb_params = list(model.W_emb.parameters())
    emb_lr = args.emb_lr if getattr(args, "emb_lr", None) is not None else args.head_lr
    emb_optimizer = torch.optim.AdamW(emb_params, lr=emb_lr, weight_decay=0.01)
    total_updates = args.max_steps or 400
    scheduler = make_lr_scheduler(emb_optimizer, total_updates, getattr(args, "cosine_lr", False), getattr(args, "warmup_steps", 0))
    initial_head = model.pointer.q.weight.detach().clone()
    initial_emb = model.W_emb.weight.detach().clone()
    lora_params = [p for n, p in model.backbone.named_parameters() if "lora_" in n]
    initial_lora_sums = [float(p.detach().float().abs().sum()) for p in lora_params]

    rng = random.Random(args.seed)
    cycle = _GroupCycle(_group_buckets(rows), rng)
    log_path = run_dir / "metrics.jsonl"
    model.W_emb.train()
    with log_path.open("w") as log:
        for step in range(1, total_updates + 1):
            batch = distinct_group_emb_batch(rows, cycle, args.emb_batch_size)
            texts = ([r["anchor_text"] for r in batch] +
                     [r["positive_text"] for r in batch] +
                     [r["hard_negative_text"] for r in batch])
            feats = torch.stack([hidden_cache[t] for t in texts]).to(model.device)
            embs = model.embeddings(feats)
            n = len(batch)
            anchors = [{d: embs[d][i] for d in SLICE_DIMS} for i in range(n)]
            positives = [{d: embs[d][n + i] for d in SLICE_DIMS} for i in range(n)]
            negatives = [{d: embs[d][2 * n + i] for d in SLICE_DIMS} for i in range(n)]
            emb_optimizer.zero_grad(set_to_none=True)
            loss = infonce_from_embeddings(anchors, positives, negatives, batch, model.device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite emb_pairs loss")
            loss.backward()
            grad_norm = _grad_norm(emb_params)
            torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
            emb_optimizer.step()
            if scheduler is not None:
                scheduler.step()
            log.write(json.dumps({"step": step, "epoch": 0, "family": "emb_pairs",
                                  "loss": float(loss.detach()), "grad_norm": grad_norm,
                                  "optimizer_updates": step}) + "\n")
    metrics = {
        "mode": "emb_only", "run_name": run_name, "smoke": args.smoke, "seed": args.seed,
        "init_checkpoint": str(args.init_checkpoint) if getattr(args, "init_checkpoint", None) else None,
        "steps": total_updates, "optimizer_updates": total_updates,
        "task_steps": {"emb_pairs": total_updates}, "cached_unique_texts": len(unique_texts),
        "emb_lr": emb_lr, "emb_batch_size": args.emb_batch_size,
        "cosine_lr": getattr(args, "cosine_lr", False), "warmup_steps": getattr(args, "warmup_steps", 0),
        "selection": "fixed_training_budget_no_cal_or_dev_access", "elapsed_seconds": time.perf_counter() - started,
        "head_changed": not torch.equal(initial_head, model.pointer.q.weight),
        "embedding_changed": not torch.equal(initial_emb, model.W_emb.weight),
        "lora_changed": (any(abs(old - float(new.detach().float().abs().sum())) > 1e-6
                             for old, new in zip(initial_lora_sums, lora_params)) if lora_params else False),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "max_seq_len": args.max_seq_len, "device": args.device, "dtype": args.dtype,
        "adapter_tensor_count": model.adapter_tensor_count,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.0,
    }
    model.save_checkpoint(checkpoint_dir, metrics)
    (run_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def run_training(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is not available in this process")
    if args.smoke:
        torch.set_num_threads(1)
        args.max_seq_len = min(args.max_seq_len, 512)
        args.max_steps = min(args.max_steps or 12, 12)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[args.dtype]
    run_name = getattr(args, "run_name", None) or args.mode
    run_dir = (Path(args.out_dir) / "runs" / run_name) if args.out_dir else HERE / "runs" / run_name
    checkpoint_dir = (Path(args.out_dir) / "checkpoints" / run_name) if args.out_dir else HERE / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    kl_loop_weight = float(getattr(args, "kl_loop_weight", 0.0) or 0.0)
    loop_ce_weight = float(getattr(args, "loop_ce_weight", 0.0) or 0.0)
    include_loop = bool(kl_loop_weight or loop_ce_weight)
    families = ("compaction", "outcome_prefix", "emb_pairs", "loop_guard") if include_loop else ("compaction", "outcome_prefix", "emb_pairs")
    if args.mode == "emb_only":
        data = smoke_rows("train") if args.smoke else {"emb_pairs": read_rows(args.dataset, "emb_pairs", "train")}
        emb_scale = len(data["emb_pairs"]) / sum(r.get("task_balanced_weight", 1) for r in data["emb_pairs"])
        for row in data["emb_pairs"]:
            row["task_balanced_weight"] = row.get("task_balanced_weight", 1) * emb_scale
        model = make_model(args.smoke, args.device, dtype, args.seed)
        return _run_emb_only(model, data, args, run_dir, checkpoint_dir, time.perf_counter())

    data = smoke_rows("train") if args.smoke else {family: read_rows(args.dataset, family, "train")
                                                     for family in families}
    if not all(data[family] for family in ("compaction", "outcome_prefix", "emb_pairs")):
        raise ValueError("All three training tasks need examples")
    schedule_mode = getattr(args, "schedule_mode", "legacy")
    if schedule_mode == "ratio_4_2_1_1":
        weights = {family: [1.0] * len(data[family]) for family in ("compaction", "outcome_prefix")}
    else:
        weight_fn = per_class_unit_weights if getattr(args, "balanced_sampling", False) else class_balanced_weights
        weights = {family: weight_fn(data[family]) for family in ("compaction", "outcome_prefix")}
    # Normalize embedding weights once over the dataset, never within a batch=1 row.
    emb_scale = len(data["emb_pairs"]) / sum(r.get("task_balanced_weight", 1) for r in data["emb_pairs"])
    for row in data["emb_pairs"]:
        row["task_balanced_weight"] = row.get("task_balanced_weight", 1) * emb_scale

    model = make_model(args.smoke, args.device, dtype, args.seed)
    kl_comp_weight = float(args.kl_compaction_weight if getattr(args, "kl_compaction_weight", None) is not None else args.kl_ref_weight)
    kl_out_weight = float(args.kl_outcome_weight if getattr(args, "kl_outcome_weight", None) is not None else args.kl_ref_weight)
    family_kl = {"compaction": kl_comp_weight, "outcome_prefix": kl_out_weight}
    init_checkpoint = getattr(args, "init_checkpoint", None)
    comp_ref_ckpt = getattr(args, "compaction_ref_checkpoint", None)
    out_ref_ckpt = getattr(args, "outcome_ref_checkpoint", None)
    family_teachers = {}
    if comp_ref_ckpt:
        family_teachers["compaction"] = comp_ref_ckpt
    if out_ref_ckpt:
        family_teachers["outcome_prefix"] = out_ref_ckpt
    need_ref = args.mode == "exp_b" and (kl_comp_weight > 0 or kl_out_weight > 0 or kl_loop_weight > 0)
    reference = {}
    if need_ref:
        # Collect pristine Kev reference logits first while `model` is still untuned.
        pristine_args = argparse.Namespace(**vars(args))
        pristine_teachers = {f: family_teachers[f] for f in family_teachers}
        reference = frozen_reference_cache(
            model, data, pristine_args, run_dir / "reference_logits.pt",
            family_teachers=pristine_teachers or None,
        )
    if init_checkpoint:
        model.load_checkpoint(init_checkpoint)
    model.configure_training(args.mode, gradient_checkpointing=args.gradient_checkpointing)
    pointer_params = list(model.pointer.parameters())
    emb_params = list(model.W_emb.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if args.mode == "exp_b" and not lora_params:
        raise AssertionError("Experiment B has no trainable LoRA tensors")
    pointer_lr = args.pointer_lr if getattr(args, "pointer_lr", None) is not None else args.head_lr
    emb_lr = args.emb_lr if getattr(args, "emb_lr", None) is not None else args.head_lr
    emb_detach = bool(getattr(args, "emb_detach_backbone", False))
    if emb_detach:
        optimizer = torch.optim.AdamW([
            {"params": pointer_params, "lr": pointer_lr},
            {"params": lora_params, "lr": args.lr},
        ], weight_decay=.01)
        emb_optimizer = torch.optim.AdamW(emb_params, lr=emb_lr, weight_decay=.01)
    else:
        optimizer = torch.optim.AdamW([
            {"params": pointer_params, "lr": pointer_lr},
            {"params": emb_params, "lr": emb_lr},
            {"params": lora_params, "lr": args.lr},
        ], weight_decay=.01)
        emb_optimizer = None
    optimizer.zero_grad(set_to_none=True)
    if emb_optimizer is not None:
        emb_optimizer.zero_grad(set_to_none=True)
    scheduled_steps = list(iter_scheduled_steps(data, args))
    planned_updates = math.ceil(len(scheduled_steps) / args.grad_accum) if scheduled_steps else 0
    scheduler = make_lr_scheduler(optimizer, planned_updates, getattr(args, "cosine_lr", False), getattr(args, "warmup_steps", 0))
    emb_planned_updates = sum(1 for s in scheduled_steps if s["family"] == "emb_pairs") if emb_detach else 0
    emb_scheduler = make_lr_scheduler(emb_optimizer, emb_planned_updates, getattr(args, "cosine_lr", False), getattr(args, "warmup_steps", 0)) if emb_detach else None
    initial_head = model.pointer.q.weight.detach().clone()
    initial_emb = model.W_emb.weight.detach().clone()
    initial_lora_sums = [float(p.detach().float().abs().sum()) for p in lora_params]
    decision_params = pointer_params + lora_params if emb_detach else pointer_params + emb_params + lora_params
    task_steps = {family: 0 for family in families}
    total_steps, updates, pending = 0, 0, 0
    emb_grad_cache = bool(getattr(args, "emb_grad_cache", False))
    save_every = int(getattr(args, "save_every", 0) or 0)
    started = time.perf_counter()
    log_path = run_dir / "metrics.jsonl"
    with log_path.open("w") as log:
        for step_info in scheduled_steps:
            step = step_info["step"]
            epoch = step_info["epoch"]
            family = step_info["family"]
            index = step_info["index"]
            row = step_info["row"]
            order = step_info["order"]
            batch = step_info["batch"]
            ce_val, kl_val, w_val, step_grad_norm = None, None, None, None
            if family == "emb_pairs":
                emb_scale_arg = 1.0 if emb_detach else (args.emb_weight / args.grad_accum)
                if emb_grad_cache and not emb_detach:
                    loss_val = matryoshka_infonce_grad_cache(
                        model, batch, args.max_seq_len, loss_scale=emb_scale_arg, detach_backbone=emb_detach
                    )
                    loss_scalar = args.emb_weight * loss_val
                else:
                    loss = matryoshka_infonce(model, batch, args.max_seq_len, detach_backbone=emb_detach)
                    if not emb_detach:
                        loss = args.emb_weight * loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {family} loss")
                    (loss if emb_detach else loss / args.grad_accum).backward()
                    loss_scalar = float(loss.detach())
                if emb_detach:
                    step_grad_norm = _grad_norm(emb_params)
                    torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
                    emb_optimizer.step()
                    if emb_scheduler is not None:
                        emb_scheduler.step()
                    emb_optimizer.zero_grad(set_to_none=True)
            elif family == "loop_guard":
                logits = decision_forward(model, family, row, order, args.max_seq_len)
                loss = logits.new_zeros(())
                if loop_ce_weight:
                    none_label = torch.tensor([order.index(0)], device=model.device)
                    ce = F.cross_entropy(logits[None, :], none_label)
                    ce_val = float(ce.detach())
                    loss = loss + loop_ce_weight * ce
                if kl_loop_weight:
                    if not reference:
                        raise ValueError("loop_guard KL regularization requires frozen reference cache")
                    ckpt_path = family_teachers.get(family)
                    ref_key = decision_key(family, row, order, args.max_seq_len, teacher_signature(ckpt_path) if ckpt_path else None)
                    ref = reference[ref_key].to(model.device)
                    kl = F.kl_div(logits.log_softmax(-1), ref.softmax(-1), reduction="sum")
                    kl_val = float(kl.detach())
                    loss = loss + kl_loop_weight * kl
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {family} loss")
                (loss / args.grad_accum).backward()
                loss_scalar = float(loss.detach())
            else:
                logits = decision_forward(model, family, row, order, args.max_seq_len)
                label = torch.tensor([order.index(row["label"])], device=model.device)
                w_val = float(weights[family][index])
                ce = F.cross_entropy(logits[None, :], label)
                ce_val = float(ce.detach())
                loss = w_val * ce
                kl_w = family_kl[family]
                if reference and kl_w:
                    ckpt_path = family_teachers.get(family)
                    ref_key = decision_key(family, row, order, args.max_seq_len, teacher_signature(ckpt_path) if ckpt_path else None)
                    ref = reference[ref_key].to(model.device)
                    kl = F.kl_div(logits.log_softmax(-1), ref.softmax(-1), reduction="sum")
                    kl_val = float(kl.detach())
                    loss = loss + kl_w * kl
                if args.perm_kl:
                    other = decision_forward(model, family, row, list(reversed(order)), args.max_seq_len).flip(0)
                    loss = loss + args.perm_kl * symmetric_kl(logits, other)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {family} loss")
                (loss / args.grad_accum).backward()
                loss_scalar = float(loss.detach())
            pending += 1
            total_steps = step
            task_steps[family] += 1
            if pending == args.grad_accum:
                step_grad_norm = _grad_norm(decision_params)
                torch.nn.utils.clip_grad_norm_(decision_params, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if model.device.type == "cuda":
                    torch.cuda.synchronize()
                updates += 1
                pending = 0
                if save_every and updates % save_every == 0:
                    model.save_checkpoint(checkpoint_dir / "latest", {
                        "mode": args.mode, "run_name": run_name, "smoke": args.smoke,
                        "steps": total_steps, "optimizer_updates": updates,
                    })
            log.write(json.dumps({
                "step": total_steps, "epoch": epoch, "family": family,
                "benchmark": row.get("benchmark"), "task_group": row.get("task_group"),
                "label": row.get("label"), "ce_loss": ce_val, "kl_loss": kl_val, "weight": w_val,
                "loss": loss_scalar, "grad_norm": step_grad_norm, "optimizer_updates": updates,
            }) + "\n")
            log.flush()
    if pending:
        for parameter in decision_params:
            if parameter.grad is not None:
                parameter.grad.mul_(args.grad_accum / pending)
        torch.nn.utils.clip_grad_norm_(decision_params, 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
    metrics = {"mode": args.mode, "run_name": run_name, "smoke": args.smoke, "seed": args.seed,
               "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
               "schedule_mode": schedule_mode,
               "steps": total_steps, "optimizer_updates": updates, "task_steps": task_steps,
               "balanced_sampling": getattr(args, "balanced_sampling", False),
               "pointer_lr": pointer_lr, "emb_lr": emb_lr, "lora_lr": args.lr,
               "kl_compaction_weight": kl_comp_weight, "kl_outcome_weight": kl_out_weight,
               "kl_loop_weight": kl_loop_weight, "loop_ce_weight": loop_ce_weight,
               "perm_kl": args.perm_kl, "emb_batch_size": args.emb_batch_size,
               "emb_grad_cache": emb_grad_cache, "emb_detach_backbone": emb_detach,
               "cosine_lr": getattr(args, "cosine_lr", False),
               "warmup_steps": getattr(args, "warmup_steps", 0),
               "selection": "fixed_training_budget_no_cal_or_dev_access", "elapsed_seconds": time.perf_counter() - started,
               "head_changed": not torch.equal(initial_head, model.pointer.q.weight),
               "embedding_changed": not torch.equal(initial_emb, model.W_emb.weight),
               "lora_changed": (any(abs(old - float(new.detach().float().abs().sum())) > 1e-6
                                    for old, new in zip(initial_lora_sums, lora_params)) if lora_params else False),
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "reference_logit_count": len(reference), "max_seq_len": args.max_seq_len,
               "gradient_checkpointing": args.mode == "exp_b" and args.gradient_checkpointing,
               "device": args.device, "dtype": args.dtype, "adapter_tensor_count": model.adapter_tensor_count,
               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.0}
    model.save_checkpoint(checkpoint_dir, metrics)
    (run_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("exp_a", "exp_b", "emb_only"), required=True)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16", "bfloat16"), default="bf16")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="Training examples (or emb_only updates); 0 means full epochs")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--emb-batch-size", type=int, default=1)
    p.add_argument("--emb-grad-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--emb-detach-backbone", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--schedule-mode", choices=("legacy", "ratio_4_2_1_1"), default="legacy")
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--compaction-ref-checkpoint", type=Path, default=None)
    p.add_argument("--outcome-ref-checkpoint", type=Path, default=None)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3, help="Default LR for W_emb and fallback for pointer if --pointer-lr omitted")
    p.add_argument("--pointer-lr", type=float, default=None)
    p.add_argument("--emb-lr", type=float, default=None)
    p.add_argument("--cosine-lr", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--emb-weight", type=float, default=1.0)
    p.add_argument("--kl-ref-weight", type=float, default=.1)
    p.add_argument("--kl-compaction-weight", type=float, default=None)
    p.add_argument("--kl-outcome-weight", type=float, default=None)
    p.add_argument("--kl-loop-weight", type=float, default=0.0)
    p.add_argument("--loop-ce-weight", type=float, default=0.0)
    p.add_argument("--perm-kl", type=float, default=0)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, help="Root containing runs/<mode> and checkpoints/<mode>")
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    if min(args.epochs, args.grad_accum, args.emb_batch_size, args.max_seq_len) < 1 or args.max_steps < 0:
        p.error("Epochs, accumulation, batch size and sequence length must be positive")
    print(json.dumps(run_training(args), indent=2))
