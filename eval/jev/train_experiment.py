"""Fixed-epoch Jev experiments. Training never reads cal or dev data."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def option_order(count, seed, epoch, row_index):
    order = list(range(count))
    random.Random(f"{seed}:{epoch}:{row_index}").shuffle(order)
    return order


def decision_key(family, row, order, max_seq_len):
    content = [family, row["state_text"], order, max_seq_len, QUESTIONS[family]]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def decision_forward(model, family, row, order, max_seq_len):
    question, options = QUESTIONS[family]
    shuffled = [options[i] for i in order]
    return model(row["state_text"], question, shuffled, max_seq_len)["logits"]


def iter_scheduled_steps(data, args):
    """Deterministic step schedule shared by frozen_reference_cache and run_training."""
    families = ("compaction", "outcome_prefix", "emb_pairs")
    if not all(data.get(f) for f in families):
        raise ValueError("All three training tasks need examples")
    emb_batch_size = getattr(args, "emb_batch_size", 1)
    seed = getattr(args, "seed", 42)
    epochs = getattr(args, "epochs", 1)
    max_steps = getattr(args, "max_steps", 0)

    total_steps = 0
    for epoch in range(epochs):
        indices = {family: list(range(len(data[family]))) for family in families}
        rng = random.Random(seed + epoch)
        for values in indices.values():
            rng.shuffle(values)
        rounds = max(len(v) for v in indices.values())
        for round_index in range(rounds):
            for family in families:
                index = indices[family][round_index % len(indices[family])]
                row = data[family][index]
                if family == "emb_pairs":
                    batch = [data[family][indices[family][(round_index * emb_batch_size + k) % len(indices[family])]]
                             for k in range(min(emb_batch_size, len(indices[family])))]
                    order = None
                else:
                    batch = None
                    order = option_order(2, seed, epoch, index)
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


def frozen_reference_cache(model, data, args, path):
    """Cache only CPU decision logits, using a pristine frozen Kev before updates."""
    path = Path(path)
    existing_cache = {}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") == "jev-kev-reference-v1" and saved.get("smoke") == getattr(args, "smoke", False):
            existing_cache = saved.get("logits", {})
    cache = {}
    model.eval()
    with torch.no_grad():
        for step in iter_scheduled_steps(data, args):
            family = step["family"]
            if family not in ("compaction", "outcome_prefix"):
                continue
            row = step["row"]
            order = step["order"]
            orders = [order]
            if getattr(args, "perm_kl", 0):
                orders.append(list(reversed(order)))
            for ord_ in orders:
                key = decision_key(family, row, ord_, args.max_seq_len)
                if key in cache:
                    continue
                if key in existing_cache:
                    cache[key] = existing_cache[key]
                else:
                    cache[key] = decision_forward(model, family, row, ord_, args.max_seq_len).float().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "jev-kev-reference-v1", "smoke": getattr(args, "smoke", False), "logits": cache}, path)
    return cache


def matryoshka_infonce(model, rows, max_seq_len, temperature=.07):
    """Explicit hard negatives plus in-batch positives/negatives with group masks."""
    if not rows:
        raise ValueError("Embedding batch must not be empty")
    anchors = [model.embed_state(r["anchor_text"], max_seq_len) for r in rows]
    positives = [model.embed_state(r["positive_text"], max_seq_len) for r in rows]
    negatives = [model.embed_state(r["hard_negative_text"], max_seq_len) for r in rows]
    groups = [r["task_group"] for r in rows]
    candidate_groups = groups + [r["hard_negative_task_group"] for r in rows]
    if any(r["task_group"] == r["hard_negative_task_group"] for r in rows):
        raise ValueError("Hard negative shares anchor task group")
    weights = torch.tensor([r.get("task_balanced_weight", 1.0) for r in rows], device=model.device)
    losses = []
    for dim in SLICE_DIMS:
        a = torch.stack([v[dim] for v in anchors])
        candidates = torch.stack([v[dim] for v in positives + negatives])
        logits = a @ candidates.T / temperature
        invalid = torch.tensor([[g == h and j != i for j, h in enumerate(candidate_groups)]
                                for i, g in enumerate(groups)], device=model.device)
        logits = logits.masked_fill(invalid, float("-inf"))
        loss = F.cross_entropy(logits, torch.arange(len(rows), device=model.device), reduction="none")
        losses.append((loss * weights).mean())
    return torch.stack(losses).mean()


def symmetric_kl(logits, other):
    p, q = logits.log_softmax(-1), other.log_softmax(-1)
    return .5 * (F.kl_div(p, q.exp(), reduction="sum") + F.kl_div(q, p.exp(), reduction="sum"))


def make_model(smoke, device, dtype, seed):
    torch.manual_seed(seed)
    if smoke:
        return synthetic_model(seed=seed, with_lora=True).to(device=device)
    return DualHeadDecisionEmbedder.from_pretrained(device=device, dtype=dtype, trainable_adapter=True)


def run_training(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm is not available in this process")
    if args.smoke:
        torch.set_num_threads(1)
        args.max_seq_len = min(args.max_seq_len, 512)
        args.max_steps = min(args.max_steps or 12, 12)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[args.dtype]
    run_dir = (Path(args.out_dir) / "runs" / args.mode) if args.out_dir else HERE / "runs" / args.mode
    checkpoint_dir = (Path(args.out_dir) / "checkpoints" / args.mode) if args.out_dir else HERE / "checkpoints" / args.mode
    run_dir.mkdir(parents=True, exist_ok=True)
    data = smoke_rows("train") if args.smoke else {family: read_rows(args.dataset, family, "train")
                                                     for family in ("compaction", "outcome_prefix", "emb_pairs")}
    if not all(data[family] for family in ("compaction", "outcome_prefix", "emb_pairs")):
        raise ValueError("All three training tasks need examples")
    weights = {family: class_balanced_weights(data[family]) for family in ("compaction", "outcome_prefix")}
    # Normalize embedding weights once over the dataset, never within a batch=1 row.
    emb_scale = len(data["emb_pairs"]) / sum(r.get("task_balanced_weight", 1) for r in data["emb_pairs"])
    for row in data["emb_pairs"]:
        row["task_balanced_weight"] = row.get("task_balanced_weight", 1) * emb_scale
    model = make_model(args.smoke, args.device, dtype, args.seed)
    reference = frozen_reference_cache(model, data, args, run_dir / "reference_logits.pt") if args.mode == "exp_b" and args.kl_ref_weight else {}
    model.configure_training(args.mode, gradient_checkpointing=args.gradient_checkpointing)
    head_params = list(model.pointer.parameters()) + list(model.W_emb.parameters())
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if args.mode == "exp_b" and not lora_params:
        raise AssertionError("Experiment B has no trainable LoRA tensors")
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": args.head_lr},
                                  {"params": lora_params, "lr": args.lr}], weight_decay=.01)
    optimizer.zero_grad(set_to_none=True)
    initial_head = model.pointer.q.weight.detach().clone()
    initial_emb = model.W_emb.weight.detach().clone()
    initial_lora = [p.detach().clone() for p in lora_params] if args.smoke else []
    task_steps = {family: 0 for family in ("compaction", "outcome_prefix", "emb_pairs")}
    total_steps, updates, pending = 0, 0, 0
    started = time.perf_counter()
    log_path = run_dir / "metrics.jsonl"
    with log_path.open("w") as log:
        for step_info in iter_scheduled_steps(data, args):
            step = step_info["step"]
            epoch = step_info["epoch"]
            family = step_info["family"]
            index = step_info["index"]
            row = step_info["row"]
            order = step_info["order"]
            batch = step_info["batch"]
            if family == "emb_pairs":
                loss = args.emb_weight * matryoshka_infonce(model, batch, args.max_seq_len)
            else:
                logits = decision_forward(model, family, row, order, args.max_seq_len)
                label = torch.tensor([order.index(row["label"])], device=model.device)
                loss = weights[family][index] * F.cross_entropy(logits[None, :], label)
                if reference:
                    ref = reference[decision_key(family, row, order, args.max_seq_len)].to(model.device)
                    loss = loss + args.kl_ref_weight * F.kl_div(logits.log_softmax(-1), ref.softmax(-1), reduction="sum")
                if args.perm_kl:
                    other = decision_forward(model, family, row, list(reversed(order)), args.max_seq_len).flip(0)
                    loss = loss + args.perm_kl * symmetric_kl(logits, other)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {family} loss")
            (loss / args.grad_accum).backward()
            pending += 1
            total_steps = step
            task_steps[family] += 1
            if pending == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(head_params + lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                pending = 0
            log.write(json.dumps({"step": total_steps, "epoch": epoch, "family": family,
                                  "loss": float(loss.detach()), "optimizer_updates": updates}) + "\n")
    if pending:
        for parameter in head_params + lora_params:
            if parameter.grad is not None:
                parameter.grad.mul_(args.grad_accum / pending)
        torch.nn.utils.clip_grad_norm_(head_params + lora_params, 1.0)
        optimizer.step()
        updates += 1
    metrics = {"mode": args.mode, "smoke": args.smoke, "seed": args.seed,
               "steps": total_steps, "optimizer_updates": updates, "task_steps": task_steps,
               "selection": "fixed_training_budget_no_cal_or_dev_access", "elapsed_seconds": time.perf_counter() - started,
               "head_changed": not torch.equal(initial_head, model.pointer.q.weight),
               "embedding_changed": not torch.equal(initial_emb, model.W_emb.weight),
               "lora_changed": any(not torch.equal(old, new) for old, new in zip(initial_lora, lora_params)) if args.smoke else None,
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
    p.add_argument("--mode", choices=("exp_a", "exp_b"), required=True)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16", "bfloat16"), default="bf16")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="Training examples, not optimizer updates; 0 means full epochs")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--emb-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--emb-weight", type=float, default=1.0)
    p.add_argument("--kl-ref-weight", type=float, default=.1)
    p.add_argument("--perm-kl", type=float, default=0)
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
