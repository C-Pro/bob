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
RETRIEVAL_FAMILIES = ("emb_pairs", "emb_pairs_bilingual", "multimodal_emb_pairs")
DECISION_FAMILIES = ("compaction", "outcome_prefix", "loop_guard", "generic_decisions", "multimodal_decisions")
DEFAULT_V2_MIX = (
    "compaction:3,outcome_prefix:2,loop_guard:3,generic_decisions:2,"
    "multimodal_decisions:2,emb_pairs:2,emb_pairs_bilingual:1,multimodal_emb_pairs:1"
)


def parse_task_mix(spec: str) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, count_str = part.split(":", 1)
        count = int(count_str)
        if count < 0:
            raise ValueError(f"Negative mix weight for {name}: {count}")
        if count > 0:
            items.append((name.strip(), count))
    if not items:
        raise ValueError("Task mix must include at least one positive family weight")
    return items


def resolve_decision_spec(family: str, row: dict):
    if "question" in row and "options" in row:
        return row["question"], [tuple(opt) for opt in row["options"]]
    if family not in QUESTIONS:
        raise KeyError(f"Missing question/options for decision family {family!r}")
    return QUESTIONS[family]


def resolve_label_index(family: str, row: dict) -> int:
    _, options = resolve_decision_spec(family, row)
    label = row["label"]
    if isinstance(label, int):
        if not 0 <= label < len(options):
            raise ValueError(f"Integer label {label} out of range for {family}")
        return label
    names = [name for name, _ in options]
    if label not in names:
        raise ValueError(f"String label {label!r} not in option names {names} for {family}")
    return names.index(label)


def resolve_images(row: dict, dataset_root: Path | None = None, key: str = "state_images"):
    raw = row.get(key)
    if not raw:
        return None
    resolved = []
    root = Path(dataset_root) if dataset_root is not None else None
    for item in raw:
        if isinstance(item, dict):
            path = Path(item["path"])
            resolved.append((root / path) if (root is not None and not path.is_absolute()) else path)
        else:
            resolved.append(item)
    return resolved


def image_fingerprints(row: dict, key: str = "state_images", dataset_root: Path | None = None) -> tuple[str, ...]:
    raw = row.get(key) or []
    fps = []
    root = Path(dataset_root) if dataset_root is not None else None
    for item in raw:
        if isinstance(item, dict):
            if item.get("sha256"):
                dims = f"{item.get('width', '')}x{item.get('height', '')}"
                fps.append(f"sha256:{item['sha256']}:{dims}")
            else:
                p = Path(item["path"])
                full = (root / p) if (root is not None and not p.is_absolute()) else p
                fps.append(f"file:{hashlib.sha256(full.read_bytes()).hexdigest()}" if full.exists() else f"path:{p}")
        elif hasattr(item, "tobytes") and hasattr(item, "size") and hasattr(item, "mode"):
            header = f"{item.mode}:{item.size[0]}x{item.size[1]}:".encode("utf-8")
            fps.append("pil:" + hashlib.sha256(header + item.tobytes()).hexdigest())
        elif isinstance(item, (str, Path)):
            p = Path(item)
            full = (root / p) if (root is not None and not p.is_absolute()) else p
            fps.append(f"file:{hashlib.sha256(full.read_bytes()).hexdigest()}" if full.exists() else f"path:{p}")
        else:
            fps.append(str(item))
    return tuple(fps)


def read_rows(directory, family, split, limit=None):
    rows = []
    with (Path(directory) / f"{family}_{split}.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError(f"Unexpected split in {family}_{split}.jsonl")
            if family in RETRIEVAL_FAMILIES:
                if row["task_group"] == row["hard_negative_task_group"] or row["hard_negative_split"] != split:
                    raise ValueError("Unsafe embedding negative: shared group or different split")
            if "options" in row:
                names = [opt[0] for opt in row["options"]]
                if len(names) != len(set(names)):
                    raise ValueError(f"Duplicate option names in {family}_{split}.jsonl")
                if row.get("label") not in names and not isinstance(row.get("label"), int):
                    raise ValueError(f"Invalid label {row.get('label')!r} in {family}_{split}.jsonl")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _smoke_image(seed_val: int):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (64, 64), ((seed_val * 53) % 200 + 20, 40, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([8, 8, 44, 44], fill=(220, (seed_val * 97) % 255, 30))
    return img


def smoke_rows(split):
    families = (*QUESTIONS, "emb_pairs", "generic_decisions", "emb_pairs_bilingual",
                "multimodal_emb_pairs", "multimodal_decisions")
    result = {family: [] for family in families}
    loop_labels = ("none", "intercept", "force_synthesis", "none")
    for i in range(4):
        group = f"{split}_{i}"
        common = {"task_group": group, "task_id": group,
                  "event_id": f"event_{i}", "split": split, "token_count": 20, "task_balanced_weight": 1.0}
        common_v2 = {**common, "split_group": f"sg_{group}"}
        state = ("Tests pass; verified checkpoint." if i % 2 else "Tests fail; investigation continues.")
        for family in ("compaction", "outcome_prefix"):
            result[family].append({**common, "state_text": state, "label": i % 2,
                                   "recorded_verdict": "probed_compact" if i % 2 else "probed_skip"})
        result["loop_guard"].append({
            **common,
            "schema_version": "jev-extension-v2" if i in (1, 2, 3) else "jev-dataset-v1",
            "state_text": state,
            "label": loop_labels[i],
        })
        result["emb_pairs"].append({**common, "anchor_text": state,
                                    "positive_text": state + " Continue from here.",
                                    "hard_negative_text": "Inspect an unrelated binary file.",
                                    "positive_task_group": group,
                                    "hard_negative_task_group": f"{split}_negative_{i}",
                                    "hard_negative_split": split})
        result["generic_decisions"].append({
            **common_v2,
            "subfamily": "policy_rules" if i % 2 == 0 else "nli",
            "language": "ru" if i % 2 else "en",
            "state_text": f"Rule threshold 100. Observed metric={120 if i % 2 else 80}.",
            "state_images": [],
            "question": "Which action applies?",
            "options": [["low", "Low tier."], ["high", "High tier."], ["none_of_the_above", "None matches."]],
            "label": "high" if i % 2 else "low",
        })
        ru_state = "Тесты прошли успешно." if i % 2 else "Сборка завершилась с ошибкой."
        result["emb_pairs_bilingual"].append({
            **common_v2,
            "language": "en_ru" if i % 2 else "ru",
            "subfamily": "episode_chat_to_summary",
            "anchor_text": state if i % 2 else ru_state,
            "positive_text": ru_state + " Продолжаем.",
            "hard_negative_text": "Несвязанный эпизод о конфигурации сети.",
            "positive_task_group": group,
            "hard_negative_task_group": f"{split}_bi_neg_{i}",
            "hard_negative_split": split,
        })
        img_pos = _smoke_image(i + 1)
        img_neg = _smoke_image(i + 101)
        result["multimodal_decisions"].append({
            **common_v2,
            "subfamily": "terminal_outcome",
            "language": "en",
            "state_text": f"Check terminal snapshot {i}.",
            "state_images": [img_pos],
            "question": "Did the command pass?",
            "options": [["passed", "Command exited 0."], ["failed", "Command failed."]],
            "label": "passed" if i % 2 else "failed",
        })
        result["multimodal_emb_pairs"].append({
            **common_v2,
            "subfamily": "image_to_text_trace",
            "language": "en",
            "anchor_text": f"Terminal capture {i}",
            "anchor_images": [img_pos],
            "positive_text": f"Trace {i} matching visual output.",
            "positive_images": [],
            "hard_negative_text": f"Counterfactual trace {i} with opposite status.",
            "hard_negative_images": [img_neg],
            "positive_task_group": group,
            "hard_negative_task_group": f"{split}_mm_neg_{i}",
            "hard_negative_split": split,
        })
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


def _group_buckets(rows, indices=None, group_key="task_group"):
    buckets = {}
    iterable = range(len(rows)) if indices is None else indices
    for idx in iterable:
        key = rows[idx].get(group_key) or rows[idx]["task_group"]
        buckets.setdefault(key, []).append(idx)
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


def balanced_loop_indices(rows, rng, total_steps):
    """3-class balanced group cycling over none (mixing v1 + synthetic), intercept, force_synthesis."""
    by_class = {lbl: [i for i, r in enumerate(rows) if r["label"] == lbl]
                for lbl in ("none", "intercept", "force_synthesis")}
    active = [lbl for lbl in ("none", "intercept", "force_synthesis") if by_class[lbl]]
    if len(active) == 1:
        cyc = _GroupCycle(_group_buckets(rows, by_class[active[0]], group_key="split_group"), rng)
        return [cyc.next() for _ in range(total_steps)]

    # Split "none" evenly between synthetic hard negatives and original v1 rows when both exist.
    none_v1 = [i for i in by_class["none"] if rows[i].get("schema_version") != "jev-extension-v2"]
    none_v2 = [i for i in by_class["none"] if rows[i].get("schema_version") == "jev-extension-v2"]
    c_none_v1 = _GroupCycle(_group_buckets(rows, none_v1, group_key="split_group"), rng) if none_v1 else None
    c_none_v2 = _GroupCycle(_group_buckets(rows, none_v2, group_key="split_group"), rng) if none_v2 else None
    c_int = _GroupCycle(_group_buckets(rows, by_class["intercept"], group_key="split_group"), rng) if by_class["intercept"] else None
    c_fs = _GroupCycle(_group_buckets(rows, by_class["force_synthesis"], group_key="split_group"), rng) if by_class["force_synthesis"] else None

    result = []
    none_toggle = 0
    for i in range(total_steps):
        cls = active[i % len(active)]
        if cls == "none":
            if c_none_v1 is not None and c_none_v2 is not None:
                result.append(c_none_v2.next() if none_toggle % 3 != 0 else c_none_v1.next())
                none_toggle += 1
            elif c_none_v2 is not None:
                result.append(c_none_v2.next())
            else:
                result.append(c_none_v1.next())
        elif cls == "intercept":
            result.append(c_int.next())
        else:
            result.append(c_fs.next())
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


def decision_key(family, row, order, max_seq_len, teacher_tag=None, preprocess_sig=None):
    question, options = resolve_decision_spec(family, row)
    effective_tag = teacher_tag if teacher_tag is not None else teacher_signature(None)
    content = [family, row["state_text"], order, max_seq_len, [question, options], effective_tag]
    imgs = image_fingerprints(row, "state_images")
    if imgs:
        content.append(list(imgs))
        if preprocess_sig is not None:
            content.append(preprocess_sig)
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def decision_forward(model, family, row, order, max_seq_len, dataset_root=None):
    question, options = resolve_decision_spec(family, row)
    shuffled = [options[i] for i in order]
    images = resolve_images(row, dataset_root, "state_images")
    return model(row["state_text"], question, shuffled, max_seq_len, images=images)["logits"]


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

    if schedule_mode == "multitask_v2":
        mix_spec = getattr(args, "v2_mix", None) or DEFAULT_V2_MIX
        mix_items = parse_task_mix(mix_spec)
        block = []
        for fam, count in mix_items:
            if not data.get(fam):
                raise ValueError(f"multitask_v2 schedule requires nonempty split for {fam!r}")
            block.extend([fam] * count)
        rng = random.Random(seed)
        rng.shuffle(block)
        total_slots = max_steps or 640
        counts_needed = {fam: sum(1 for i in range(total_slots) if block[i % len(block)] == fam)
                         for fam, _ in mix_items}
        lists = {}
        cycles = {}
        cursors = {}
        for fam, _ in mix_items:
            need = counts_needed[fam]
            if fam == "compaction":
                lists[fam] = compaction_ratio_indices(data[fam], rng, need)
                cursors[fam] = 0
            elif fam == "outcome_prefix":
                lists[fam] = balanced_group_indices(data[fam], rng, need)
                cursors[fam] = 0
            elif fam == "loop_guard":
                lists[fam] = balanced_loop_indices(data[fam], rng, need)
                cursors[fam] = 0
            elif fam in RETRIEVAL_FAMILIES:
                cycles[fam] = _GroupCycle(_group_buckets(data[fam]), rng)
            else:
                cycles[fam] = _GroupCycle(_group_buckets(data[fam], group_key="split_group"), rng)

        for step_idx in range(1, total_slots + 1):
            family = block[(step_idx - 1) % len(block)]
            if family in RETRIEVAL_FAMILIES:
                batch = distinct_group_emb_batch(data[family], cycles[family], emb_batch_size)
                index, row, order = 0, batch[0], None
            else:
                if family in lists:
                    index = lists[family][cursors[family]]
                    cursors[family] += 1
                else:
                    index = cycles[family].next()
                row = data[family][index]
                _, options = resolve_decision_spec(family, row)
                batch, order = None, option_order(len(options), seed, 0, step_idx)
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


def teacher_signature(checkpoint_path, smoke=False, seed=None):
    if checkpoint_path is None:
        base = "smoke_synthetic" if smoke else "pristine_kev"
        return f"{base}:seed={seed}:prompt-v2" if smoke and seed is not None else f"{base}:prompt-v2"
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    return f"ckpt:{path}:{stat.st_size}:{stat.st_mtime_ns}:prompt-v2"


def model_preprocess_signature(model):
    if getattr(model, "visual", None) is None:
        return None
    return [model.image_min_pixels, model.image_max_pixels, list(model.image_mean), list(model.image_std)]


def _apply_checkpoint_preprocess(model, checkpoint_path):
    """Apply saved vision_preprocess metadata to a model without overwriting its weights."""
    if checkpoint_path is None or getattr(model, "visual", None) is None:
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vis_prep = (payload.get("metadata") or {}).get("vision_preprocess")
    if isinstance(vis_prep, dict):
        model.image_min_pixels = int(vis_prep.get("min_pixels", model.image_min_pixels))
        model.image_max_pixels = int(vis_prep.get("max_pixels", model.image_max_pixels))
        if "mean" in vis_prep:
            model.image_mean = tuple(float(x) for x in vis_prep["mean"])
        if "std" in vis_prep:
            model.image_std = tuple(float(x) for x in vis_prep["std"])
        model._image_processor = None


def frozen_reference_cache(model, data, args, path, family_teachers=None):
    """Cache only CPU decision logits, fingerprinted by teacher checkpoint, images, and prompt format."""
    smoke_flag = bool(getattr(args, "smoke", False))
    seed_val = getattr(args, "seed", 42)
    path = Path(path)
    existing_cache = {}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved.get("format") in ("jev-kev-reference-v1", "jev-kev-reference-v2", "jev-kev-reference-v3")
                and saved.get("smoke") == smoke_flag
                and (not smoke_flag or saved.get("smoke_seed", seed_val) == seed_val)):
            existing_cache = saved.get("logits", {})
    family_teachers = family_teachers or {}
    kl_comp = getattr(args, "kl_compaction_weight", None)
    kl_out = getattr(args, "kl_outcome_weight", None)
    kl_ref_raw = getattr(args, "kl_ref_weight", 0.1)
    kl_ref = 0.0 if kl_ref_raw is None else float(kl_ref_raw)
    kl_loop = float(getattr(args, "kl_loop_weight", 0) or 0)
    kl_gen = float(getattr(args, "kl_generic_weight", 0) or 0)
    kl_mm = float(getattr(args, "kl_mm_weight", 0) or 0)
    need_comp = (kl_comp if kl_comp is not None else kl_ref) > 0
    need_out = (kl_out if kl_out is not None else kl_ref) > 0
    active_map = {
        "compaction": need_comp,
        "outcome_prefix": need_out,
        "loop_guard": kl_loop > 0,
        "generic_decisions": kl_gen > 0,
        "multimodal_decisions": kl_mm > 0,
    }
    target_families = tuple(f for f, active in active_map.items() if active)
    if not target_families:
        return {}
    cache = {}
    by_teacher = {}
    student_prep_sig = model_preprocess_signature(model)
    dataset_root = None if smoke_flag else getattr(args, "dataset", None)
    for step in iter_scheduled_steps(data, args):
        family = step["family"]
        if family not in target_families:
            continue
        ckpt_path = family_teachers.get(family)
        tag = teacher_signature(ckpt_path)
        row, order = step["row"], step["order"]
        orders = [order]
        if family in ("compaction", "outcome_prefix") and getattr(args, "perm_kl", 0):
            orders.append(list(reversed(order)))
        for ord_ in orders:
            cache_key = decision_key(
                family, row, ord_, args.max_seq_len,
                tag if ckpt_path is not None else None,
                preprocess_sig=student_prep_sig,
            )
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
            teacher_model = make_model(
                smoke_flag, str(model.device), dtype, seed_val,
                include_vision=(getattr(model, "visual", None) is not None),
            )
            teacher_model.load_checkpoint(ckpt_path)
        teacher_model.eval()
        with torch.no_grad():
            for cache_key, family, row, ord_ in pending_items:
                cache[cache_key] = decision_forward(
                    teacher_model, family, row, ord_, args.max_seq_len, dataset_root=dataset_root
                ).float().cpu()
        if ckpt_path is not None:
            del teacher_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    path.parent.mkdir(parents=True, exist_ok=True)
    has_v2_targets = any(f in ("generic_decisions", "multimodal_decisions") for f in target_families)
    if has_v2_targets:
        fmt = "jev-kev-reference-v3"
    elif any(v is not None for v in family_teachers.values()):
        fmt = "jev-kev-reference-v2"
    else:
        fmt = "jev-kev-reference-v1"
    torch.save({"format": fmt, "smoke": smoke_flag, "smoke_seed": seed_val if smoke_flag else None, "logits": cache}, path)
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


def matryoshka_infonce(model, rows, max_seq_len, temperature=.07, detach_backbone=False, dataset_root=None):
    """Explicit hard negatives plus in-batch positives/negatives with group masks."""
    if not rows:
        raise ValueError("Embedding batch must not be empty")
    anchors = [model.embed_state(r["anchor_text"], max_seq_len, detach_backbone=detach_backbone,
                                 images=resolve_images(r, dataset_root, "anchor_images")) for r in rows]
    positives = [model.embed_state(r["positive_text"], max_seq_len, detach_backbone=detach_backbone,
                                   images=resolve_images(r, dataset_root, "positive_images")) for r in rows]
    negatives = [model.embed_state(r["hard_negative_text"], max_seq_len, detach_backbone=detach_backbone,
                                   images=resolve_images(r, dataset_root, "hard_negative_images")) for r in rows]
    return infonce_from_embeddings(anchors, positives, negatives, rows, model.device, temperature)


def matryoshka_infonce_grad_cache(model, rows, max_seq_len, loss_scale=1.0, temperature=.07, detach_backbone=False, dataset_root=None):
    """Two-pass gradient-cached Matryoshka InfoNCE: batch-B negatives at batch-1 VRAM."""
    if not rows:
        raise ValueError("Embedding batch must not be empty")
    items = (
        [(r["anchor_text"], resolve_images(r, dataset_root, "anchor_images")) for r in rows] +
        [(r["positive_text"], resolve_images(r, dataset_root, "positive_images")) for r in rows] +
        [(r["hard_negative_text"], resolve_images(r, dataset_root, "hard_negative_images")) for r in rows]
    )
    with torch.no_grad():
        cached = [model.embed_state(text, max_seq_len, images=imgs) for text, imgs in items]
    leaves = [{dim: vec[dim].detach().requires_grad_(True) for dim in SLICE_DIMS} for vec in cached]
    n = len(rows)
    loss = loss_scale * infonce_from_embeddings(leaves[:n], leaves[n:2 * n], leaves[2 * n:], rows, model.device, temperature)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite emb_pairs loss")
    loss.backward()
    for (text, imgs), leaf in zip(items, leaves):
        terms = [dim for dim in SLICE_DIMS if leaf[dim].grad is not None]
        if not terms:
            continue
        emb = model.embed_state(text, max_seq_len, detach_backbone=detach_backbone, images=imgs)
        surrogate = sum((emb[dim] * leaf[dim].grad).sum() for dim in terms)
        surrogate.backward()
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
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


def make_model(smoke, device, dtype, seed, include_vision=False):
    torch.manual_seed(seed)
    if smoke:
        return synthetic_model(seed=seed, with_lora=True, include_vision=include_vision).to(device=device)
    return DualHeadDecisionEmbedder.from_pretrained(
        device=device, dtype=dtype, trainable_adapter=True, include_vision=include_vision
    )


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
    dataset_root = None if args.smoke else args.dataset
    emb_families = [f for f in RETRIEVAL_FAMILIES if data.get(f)]
    hidden_cache = {}
    with torch.no_grad():
        for fam in emb_families:
            for r in data[fam]:
                for text_k, img_k in (("anchor_text", "anchor_images"), ("positive_text", "positive_images"), ("hard_negative_text", "hard_negative_images")):
                    key = (r[text_k], image_fingerprints(r, img_k))
                    if key not in hidden_cache:
                        hidden_cache[key] = model.state_hidden(
                            r[text_k], args.max_seq_len, images=resolve_images(r, dataset_root, img_k)
                        ).detach().float().cpu()
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
    cycles = {fam: _GroupCycle(_group_buckets(data[fam]), rng) for fam in emb_families}
    task_steps = {fam: 0 for fam in emb_families}
    log_path = run_dir / "metrics.jsonl"
    model.W_emb.train()
    with log_path.open("w") as log:
        for step in range(1, total_updates + 1):
            fam = emb_families[(step - 1) % len(emb_families)]
            batch = distinct_group_emb_batch(data[fam], cycles[fam], args.emb_batch_size)
            keys = (
                [(r["anchor_text"], image_fingerprints(r, "anchor_images")) for r in batch] +
                [(r["positive_text"], image_fingerprints(r, "positive_images")) for r in batch] +
                [(r["hard_negative_text"], image_fingerprints(r, "hard_negative_images")) for r in batch]
            )
            feats = torch.stack([hidden_cache[k] for k in keys]).to(model.device)
            embs = model.embeddings(feats)
            n = len(batch)
            anchors = [{d: embs[d][i] for d in SLICE_DIMS} for i in range(n)]
            positives = [{d: embs[d][n + i] for d in SLICE_DIMS} for i in range(n)]
            negatives = [{d: embs[d][2 * n + i] for d in SLICE_DIMS} for i in range(n)]
            emb_optimizer.zero_grad(set_to_none=True)
            loss = infonce_from_embeddings(anchors, positives, negatives, batch, model.device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {fam} loss")
            loss.backward()
            grad_norm = _grad_norm(emb_params)
            torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
            emb_optimizer.step()
            if scheduler is not None:
                scheduler.step()
            task_steps[fam] += 1
            log.write(json.dumps({"step": step, "epoch": 0, "family": fam,
                                  "loss": float(loss.detach()), "grad_norm": grad_norm,
                                  "optimizer_updates": step}) + "\n")
    metrics = {
        "mode": "emb_only", "run_name": run_name, "smoke": args.smoke, "seed": args.seed,
        "init_checkpoint": str(args.init_checkpoint) if getattr(args, "init_checkpoint", None) else None,
        "steps": total_updates, "optimizer_updates": total_updates,
        "task_steps": task_steps, "cached_unique_states": len(hidden_cache),
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
        args.max_steps = min(args.max_steps or 12, 16)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[args.dtype]
    run_name = getattr(args, "run_name", None) or args.mode
    run_dir = (Path(args.out_dir) / "runs" / run_name) if args.out_dir else HERE / "runs" / run_name
    checkpoint_dir = (Path(args.out_dir) / "checkpoints" / run_name) if args.out_dir else HERE / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    schedule_mode = getattr(args, "schedule_mode", "legacy")
    is_v2 = schedule_mode == "multitask_v2"

    def _opt_float(attr, default_v2):
        val = getattr(args, attr, None)
        if val is None:
            return float(default_v2) if is_v2 else 0.0
        return float(val)

    kl_loop_weight = _opt_float("kl_loop_weight", 0.05)
    loop_ce_weight = _opt_float("loop_ce_weight", 0.5)
    kl_generic_weight = _opt_float("kl_generic_weight", 1.0)
    generic_ce_weight = _opt_float("generic_ce_weight", 0.25)
    kl_mm_weight = _opt_float("kl_mm_weight", 0.1)
    mm_ce_weight = _opt_float("mm_ce_weight", 1.0)

    if args.mode == "emb_only":
        emb_fams = tuple(f.strip() for f in (getattr(args, "emb_families", None) or "emb_pairs").split(",") if f.strip())
        if args.smoke:
            all_smoke = smoke_rows("train")
            data = {f: all_smoke[f] for f in emb_fams}
        else:
            data = {f: read_rows(args.dataset, f, "train") for f in emb_fams}
        for fam in emb_fams:
            emb_scale = len(data[fam]) / sum(r.get("task_balanced_weight", 1) for r in data[fam])
            for row in data[fam]:
                row["task_balanced_weight"] = row.get("task_balanced_weight", 1) * emb_scale
        need_vision = bool(getattr(args, "include_vision", False) or any(f.startswith("multimodal_") for f in emb_fams))
        model = make_model(args.smoke, args.device, dtype, args.seed, include_vision=need_vision)
        return _run_emb_only(model, data, args, run_dir, checkpoint_dir, time.perf_counter())

    if schedule_mode == "multitask_v2":
        mix_spec = getattr(args, "v2_mix", None) or DEFAULT_V2_MIX
        families = tuple(dict.fromkeys(fam for fam, _ in parse_task_mix(mix_spec)))
    else:
        include_loop = bool(kl_loop_weight or loop_ce_weight)
        families = ("compaction", "outcome_prefix", "emb_pairs", "loop_guard") if include_loop else ("compaction", "outcome_prefix", "emb_pairs")

    if args.smoke:
        all_smoke = smoke_rows("train")
        data = {family: all_smoke[family] for family in families}
    else:
        data = {family: read_rows(args.dataset, family, "train") for family in families}

    if not all(data.get(f) for f in families):
        raise ValueError("All scheduled training tasks need examples")

    if schedule_mode in ("ratio_4_2_1_1", "multitask_v2"):
        weights = {family: [1.0] * len(data[family]) for family in ("compaction", "outcome_prefix") if family in data}
    else:
        weight_fn = per_class_unit_weights if getattr(args, "balanced_sampling", False) else class_balanced_weights
        weights = {family: weight_fn(data[family]) for family in ("compaction", "outcome_prefix") if family in data}

    for fam in families:
        if fam in RETRIEVAL_FAMILIES:
            emb_scale = len(data[fam]) / sum(r.get("task_balanced_weight", 1) for r in data[fam])
            for row in data[fam]:
                row["task_balanced_weight"] = row.get("task_balanced_weight", 1) * emb_scale

    need_vision = bool(getattr(args, "include_vision", False) or any(f.startswith("multimodal_") for f in families))
    model = make_model(args.smoke, args.device, dtype, args.seed, include_vision=need_vision)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint:
        _apply_checkpoint_preprocess(model, init_checkpoint)

    if args.mode == "exp_b":
        kl_comp_weight = float(args.kl_compaction_weight if getattr(args, "kl_compaction_weight", None) is not None else args.kl_ref_weight)
        kl_out_weight = float(args.kl_outcome_weight if getattr(args, "kl_outcome_weight", None) is not None else args.kl_ref_weight)
        family_kl = {
            "compaction": kl_comp_weight,
            "outcome_prefix": kl_out_weight,
            "loop_guard": kl_loop_weight,
            "generic_decisions": kl_generic_weight,
            "multimodal_decisions": kl_mm_weight,
        }
    else:
        kl_comp_weight, kl_out_weight = 0.0, 0.0
        kl_loop_weight = kl_generic_weight = kl_mm_weight = 0.0
        family_kl = {f: 0.0 for f in DECISION_FAMILIES}

    # Verify every scheduled decision family has an active objective before running.
    ce_weights = {"loop_guard": loop_ce_weight, "generic_decisions": generic_ce_weight, "multimodal_decisions": mm_ce_weight}
    for fam in families:
        if fam in ("loop_guard", "generic_decisions", "multimodal_decisions"):
            if ce_weights[fam] == 0.0 and family_kl.get(fam, 0.0) == 0.0:
                raise ValueError(f"Scheduled decision family {fam!r} has zero CE and zero KL weight")

    comp_ref_ckpt = getattr(args, "compaction_ref_checkpoint", None)
    out_ref_ckpt = getattr(args, "outcome_ref_checkpoint", None)
    family_teachers = {}
    if comp_ref_ckpt:
        family_teachers["compaction"] = comp_ref_ckpt
    if out_ref_ckpt:
        family_teachers["outcome_prefix"] = out_ref_ckpt
    need_ref = args.mode == "exp_b" and any(family_kl[f] > 0 and f in families for f in family_kl)
    reference = {}
    if need_ref:
        pristine_args = argparse.Namespace(**vars(args))
        pristine_args.kl_loop_weight = kl_loop_weight
        pristine_args.loop_ce_weight = loop_ce_weight
        pristine_args.kl_generic_weight = kl_generic_weight
        pristine_args.generic_ce_weight = generic_ce_weight
        pristine_args.kl_mm_weight = kl_mm_weight
        pristine_args.mm_ce_weight = mm_ce_weight
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
    emb_planned_updates = sum(1 for s in scheduled_steps if s["family"] in RETRIEVAL_FAMILIES) if emb_detach else 0
    emb_scheduler = make_lr_scheduler(emb_optimizer, emb_planned_updates, getattr(args, "cosine_lr", False), getattr(args, "warmup_steps", 0)) if emb_detach else None
    initial_head = model.pointer.q.weight.detach().clone()
    initial_emb = model.W_emb.weight.detach().clone()
    initial_lora_sums = [float(p.detach().float().abs().sum()) for p in lora_params]
    decision_params = pointer_params + lora_params if emb_detach else pointer_params + emb_params + lora_params
    task_steps = {family: 0 for family in families}
    total_steps, updates, pending = 0, 0, 0
    emb_grad_cache = bool(getattr(args, "emb_grad_cache", False))
    save_every = int(getattr(args, "save_every", 0) or 0)
    dataset_root = None if args.smoke else args.dataset
    prep_sig = model_preprocess_signature(model)
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
            if family in RETRIEVAL_FAMILIES:
                emb_scale_arg = 1.0 if emb_detach else (args.emb_weight / args.grad_accum)
                if emb_grad_cache and not emb_detach:
                    loss_val = matryoshka_infonce_grad_cache(
                        model, batch, args.max_seq_len, loss_scale=emb_scale_arg,
                        detach_backbone=emb_detach, dataset_root=dataset_root,
                    )
                    loss_scalar = args.emb_weight * loss_val
                else:
                    loss = matryoshka_infonce(
                        model, batch, args.max_seq_len, detach_backbone=emb_detach, dataset_root=dataset_root
                    )
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
            else:
                logits = decision_forward(model, family, row, order, args.max_seq_len, dataset_root=dataset_root)
                loss = logits.new_zeros(())
                target_idx = resolve_label_index(family, row)
                label_t = torch.tensor([order.index(target_idx)], device=model.device)
                if family == "loop_guard":
                    ce_weight = loop_ce_weight
                elif family == "generic_decisions":
                    ce_weight = generic_ce_weight
                elif family == "multimodal_decisions":
                    ce_weight = mm_ce_weight
                else:
                    ce_weight = float(weights[family][index])
                if ce_weight:
                    w_val = float(ce_weight)
                    ce = F.cross_entropy(logits[None, :], label_t)
                    ce_val = float(ce.detach())
                    loss = loss + ce_weight * ce
                kl_w = family_kl.get(family, 0.0)
                if kl_w:
                    if not reference:
                        raise ValueError(f"{family} KL regularization requires frozen reference cache")
                    ckpt_path = family_teachers.get(family)
                    ref_key = decision_key(
                        family, row, order, args.max_seq_len,
                        teacher_signature(ckpt_path) if ckpt_path is not None else None,
                        preprocess_sig=prep_sig,
                    )
                    ref = reference[ref_key].to(model.device)
                    kl = F.kl_div(logits.log_softmax(-1), ref.softmax(-1), reduction="sum")
                    kl_val = float(kl.detach())
                    loss = loss + kl_w * kl
                if family in ("compaction", "outcome_prefix") and args.perm_kl:
                    other = decision_forward(
                        model, family, row, list(reversed(order)), args.max_seq_len, dataset_root=dataset_root
                    ).flip(0)
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
                    torch.cuda.empty_cache()
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
               "include_vision": need_vision,
               "steps": total_steps, "optimizer_updates": updates, "task_steps": task_steps,
               "balanced_sampling": getattr(args, "balanced_sampling", False),
               "pointer_lr": pointer_lr, "emb_lr": emb_lr, "lora_lr": args.lr,
               "kl_compaction_weight": kl_comp_weight, "kl_outcome_weight": kl_out_weight,
               "kl_loop_weight": kl_loop_weight, "loop_ce_weight": loop_ce_weight,
               "kl_generic_weight": kl_generic_weight, "generic_ce_weight": generic_ce_weight,
               "kl_mm_weight": kl_mm_weight, "mm_ce_weight": mm_ce_weight,
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
    p.add_argument("--include-vision", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="Training examples (or emb_only updates); 0 means full epochs")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--emb-batch-size", type=int, default=1)
    p.add_argument("--emb-grad-cache", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--emb-detach-backbone", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--emb-families", default="emb_pairs", help="Comma-separated retrieval families for emb_only mode")
    p.add_argument("--schedule-mode", choices=("legacy", "ratio_4_2_1_1", "multitask_v2"), default="legacy")
    p.add_argument("--v2-mix", default=DEFAULT_V2_MIX, help="Task mix for multitask_v2 schedule")
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
    p.add_argument("--kl-loop-weight", type=float, default=None)
    p.add_argument("--loop-ce-weight", type=float, default=None)
    p.add_argument("--kl-generic-weight", type=float, default=None)
    p.add_argument("--generic-ce-weight", type=float, default=None)
    p.add_argument("--kl-mm-weight", type=float, default=None)
    p.add_argument("--mm-ce-weight", type=float, default=None)
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
