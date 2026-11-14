#!/usr/bin/env python3
"""Build isolated, time-local Jev supervision from recorded FSM trajectories.

State text follows the Go probe's byte budgets, with provenance identifiers and
label fields removed. Skip events store the *previous* cutoff, not the attempted
boundary: reconstruct the latter using the recorded run's tail setting (default
1). This inference is explicitly marked in metadata. Loop labels are conservative
weak supervision; raw DetectLoop actions are retained separately because the Go
signature threshold alone does not establish unchanged output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.5-0.8B-Base"
SCHEMA_VERSION = "jev-dataset-v1"
RENDERER_VERSION = "go-probe-sanitized-v1"
SPLITS = ("train", "dev", "cal")
FAMILIES = ("compaction", "outcome_prefix", "loop_guard", "emb_pairs")
MAX_BYTES = 32768
STATE_FIELDS = (
    "original_objective", "immutable_constraints", "verified_findings",
    "decisions_and_rejected", "current_subgoal", "next_intended_actions",
    "unresolved_blockers", "produced_artifacts", "active_skill_versions",
    "active_memory_versions",
)


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def decode_json(value):
    """Decode bounded nested serialization; leave legacy non-JSON text intact."""
    for _ in range(8):
        if not isinstance(value, str):
            break
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            break
        if decoded == value:
            break
        value = decoded
    return value


def as_text(value):
    return value if isinstance(value, str) else compact_json(value)


def stored_result(value):
    decoded = decode_json(value)
    if isinstance(decoded, dict) and decoded.get("bob_tool_result_version", 0) > 0:
        result = decode_json(decoded.get("result", {}))
        if isinstance(result, dict):
            return as_text(result.get("content", ""))
    # Preserve raw legacy JSON spelling, including whitespace, like Go.
    return as_text(decoded) if not isinstance(value, str) else (
        decoded if isinstance(decoded, str) else value
    )


def normalize_instruction(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def task_identity(record):
    task_id = str(record.get("task_id", "")).strip()
    instruction = normalize_instruction(record.get("instruction", ""))
    if not task_id or not instruction:
        raise ValueError("Every task requires a task_id and nonempty instruction")
    return task_id, instruction


def identity_keys(record):
    """Comparable identities; missing/empty gold fields never imply a match."""
    task_id, instruction = task_identity(record)
    keys = {("task_id", task_id), ("instruction", instruction)}
    metadata = decode_json(record.get("metadata", {}))
    if not isinstance(metadata, dict):
        metadata = {}
    values = {**metadata, **record}
    for field in ("gold_command", "gold_flag", "gold_answer", "instance_id"):
        value = values.get(field)
        if value is None or isinstance(value, (dict, list, bool)):
            continue
        value = unicodedata.normalize("NFKC", str(value)).strip()
        if value:
            keys.add((field, normalize_instruction(value) if field == "gold_answer" else value))
    repo, commit = values.get("repo"), values.get("base_commit")
    if isinstance(repo, str) and isinstance(commit, str) and repo.strip() and commit.strip():
        keys.add(("repo+base_commit", (repo.strip(), commit.strip())))
    return keys


def heldout_collisions(records, heldout):
    forbidden = defaultdict(set)
    for task in heldout:
        for key in identity_keys(task):
            forbidden[key].add(task["task_id"])
    return [{"heldout_task_ids": sorted({task_id for key in identity_keys(record)
                                        for task_id in forbidden.get(key, ())}),
             "identity_fields": sorted({key[0] for key in identity_keys(record) if key in forbidden})}
            for record in records]


def assert_no_heldout_overlap(records, heldout):
    for record, collision in zip(records, heldout_collisions(records, heldout)):
        if collision["identity_fields"]:
            fields = ", ".join(collision["identity_fields"])
            raise ValueError(f"Held-out overlap: {record['task_id']} ({fields})")


def group_tasks(records, heldout):
    """Connected components under any shared task, instruction, or gold identity."""
    assert_no_heldout_overlap(records, heldout)
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        for key in sorted(identity_keys(record)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    members = defaultdict(list)
    for i, record in enumerate(records):
        members[find(i)].append(record)
    for component in members.values():
        identities = sorted({task_identity(r) for r in component})
        group = "tg_" + hashlib.sha256(compact_json(identities).encode()).hexdigest()[:24]
        for record in component:
            record["task_group"] = group


def outcome_class(record):
    if record.get("benchmark") == "intercode-bash" and record.get("grading_status") == "oracle_error":
        return "excluded_oracle_error"
    if record.get("benchmark") == "swe-bench-lite" and "Docker eval failed to produce report.json" in as_text(record.get("eval_details", "")):
        return "excluded_swe_missing_report"
    if record.get("grading_status") != "graded" or record.get("benchmark_reward") not in (0, 1):
        return "excluded_ungraded"
    return "passed" if record["benchmark_reward"] == 1 else "failed"


def split_tasks(records, seed=42, heldout=()):
    assert_no_heldout_overlap(records, heldout)
    identity_groups = {}
    for record in records:
        for key in identity_keys(record):
            if key in identity_groups and identity_groups[key] != record["task_group"]:
                raise ValueError(f"Shared {key[0]} identity assigned to different task groups")
            identity_groups[key] = record["task_group"]
    groups = defaultdict(list)
    for record in records:
        groups[record["task_group"]].append(record)
    strata = defaultdict(list)
    for group, members in groups.items():
        # Conflicting replicate outcomes stay together in an explicit mixed stratum.
        key = (tuple(sorted({r["benchmark"] for r in members})),
               tuple(sorted({outcome_class(r) for r in members})))
        strata[key].append(group)
    rng = random.Random(seed)
    assignment = {}
    for key in sorted(strata):
        group_ids = sorted(strata[key])
        rng.shuffle(group_ids)
        exact = [len(group_ids) * f for f in (.70, .15, .15)]
        sizes = [math.floor(x) for x in exact]
        for i in sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))[:len(group_ids) - sum(sizes)]:
            sizes[i] += 1
        start = 0
        for split, size in zip(SPLITS, sizes):
            for group in group_ids[start:start + size]:
                assignment[group] = split
            start += size
    for record in records:
        record["split"] = assignment[record["task_group"]]
    return assignment


def ordered_steps(record):
    return sorted(record["trajectory"].get("steps", []),
                  key=lambda s: (int(s["iteration"]), int(s.get("step_index", 0)), s["id"]))


def compaction_events(record):
    """Events supply execution time; never invent it from snapshot cutoffs."""
    events, seen = [], set()
    for index, event in enumerate(record["trajectory"].get("events", [])):
        if event.get("event_type") != "compaction":
            continue
        payload = decode_json(event.get("payload", {}))
        if not isinstance(payload, dict):
            raise ValueError("Invalid compaction event payload")
        key = (event.get("run_id"), event["iteration"], payload.get("version", 0),
               payload.get("rubric_verdict"))
        if key in seen:
            continue
        seen.add(key)
        events.append({**event, "payload": payload, "event_order": index})
    return sorted(events, key=lambda e: (e["iteration"], e["event_order"]))


def original_goal(record):
    # context_json also contains future assistant messages: only the initial user
    # objective is eligible. The manifest instruction is the fallback.
    context = decode_json(record["trajectory"].get("run", {}).get("context_json", []))
    if isinstance(context, list):
        for message in context:
            if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
                return as_text(message["content"]).strip()
    return record["instruction"]


def prior_state(events, iteration, version=None, event_order=None):
    eligible = []
    for event in events:
        p = event["payload"]
        if not p.get("state_json") or not p.get("version"):
            continue
        if event["iteration"] > iteration:
            continue
        if event["iteration"] == iteration and (event_order is None or event["event_order"] >= event_order):
            continue
        if version is not None and p["version"] >= version:
            continue
        eligible.append(event)
    return max(eligible, key=lambda e: e["payload"]["version"], default=None)


def clip_utf8(text, budget, reserve, suffix):
    if len(text.encode()) <= budget:
        return text, False
    return text.encode()[:budget - reserve].decode("utf-8", errors="ignore") + suffix, True


def render_state(record, steps, cutoff_iter, execution_iter, previous=None, tail=None, candidates=None):
    """Go-shaped probe, stripping run/step identifiers and previous label fields."""
    truncated = False

    def clip(text, budget, reserve=23, suffix="...[truncated for probe]"):
        nonlocal truncated
        result, did_clip = clip_utf8(text, budget, reserve, suffix)
        truncated |= did_clip
        return result

    prev_payload = previous["payload"] if previous else {}
    prev = decode_json(prev_payload.get("state_json", {}))
    goal = prev.get("original_objective") or original_goal(record)
    chunks = [f"Candidate Compaction Boundary: Iterations 1..{cutoff_iter} (Current Run Iteration: {execution_iter})\n",
              f"Original Objective: {clip(goal, 2048, 15, '...[truncated]')}\n\n"]
    if previous:
        safe = {key: prev[key] for key in STATE_FIELDS if key in prev}
        # Findings keep their content but not source run/step identifiers.
        if "verified_findings" in safe:
            safe["verified_findings"] = [
                {"summary": f.get("summary", "")} if isinstance(f, dict) else f
                for f in safe["verified_findings"]
            ]
        text = clip(compact_json(safe), 8192, 15, "...[truncated]")
        chunks.append(f"Previous Compacted State (v{prev_payload['version']}, up to iter {prev_payload['compacted_up_to_iter']}):\n{text}\n\n")
    chunks.append("Candidate Steps to Compact:\n")
    lower = prev_payload.get("compacted_up_to_iter", 0)
    selected = [s for s in steps if lower < s["iteration"] <= cutoff_iter]
    for s in selected:
        args = clip(as_text(decode_json(s.get("args_json", ""))).strip(), 512)
        output = stored_result(s.get("result_json", "")).strip() or s.get("error_text", "")
        output = clip(output, 1024)
        chunks.append(f"- Iter={s['iteration']} Tool={s['tool_name']} Status={s.get('status', '')} Args={args}\n  Output: {output}\n")
        if sum(len(c.encode()) for c in chunks) >= MAX_BYTES:
            chunks.append("...[additional candidate steps truncated for probe budget]\n")
            truncated = True
            break
    if tail and sum(len(c.encode()) for c in chunks) < MAX_BYTES:
        chunks.append("\nRetained Raw Tail Steps (not compacted in this pass):\n")
        for s in tail:
            if cutoff_iter < s["iteration"] <= execution_iter:
                chunks.append(f"- Iter={s['iteration']} Tool={s['tool_name']} Status={s.get('status', '')}\n")
    if candidates:
        chunks.append("\nCandidate Tool Calls (not yet executed):\n")
        for c in candidates:
            chunks.append(f"- Tool={c['tool_name']} Args={clip(as_text(c['args_json']), 512)}\n")
    text = clip("".join(chunks), MAX_BYTES, 16, "\n...[truncated]")
    return text, truncated


class TokenCounter:
    def __init__(self, char_fallback=False, cache_dir=None):
        self.char_fallback = char_fallback
        self.tokenizer = None
        if not char_fallback:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=cache_dir)

    def __call__(self, text):
        if self.char_fallback:
            return len(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def row_metadata(record, event_id, phase, cutoff_iter, cutoff_step, previous, source, quality, text, truncated, counter):
    return {
        "schema_version": SCHEMA_VERSION, "task_id": record["task_id"],
        "task_group": record["task_group"], "benchmark": record["benchmark"],
        "run_id": record["trajectory"].get("run", {}).get("id", ""),
        "event_id": event_id, "decision_phase": phase, "cutoff_iter": cutoff_iter,
        "cutoff_step": cutoff_step,
        "prev_compaction_version": previous["payload"]["version"] if previous else 0,
        "label_source": source, "label_quality": quality,
        "token_count": counter(text), "rendered_bytes": len(text.encode()),
        "truncated": truncated, "source_sha256": record["source_sha256"],
        "renderer_version": RENDERER_VERSION, "split": record["split"],
    }


def probe_error(payload):
    reason = payload.get("reason", "")
    unescaped = reason.replace('\\"', '"')
    return reason.lstrip().lower().startswith("probe_error:") or bool(
        re.search(r'"should_compact"\s*:\s*true', unescaped, re.I)
    )


def recall_metadata(steps, event, events):
    linked = []
    version = event["payload"].get("version", 0)
    for step in steps:
        if step["tool_name"] != "recall_task_history" or step["iteration"] <= event["iteration"]:
            continue
        result = decode_json(stored_result(step.get("result_json", "")))
        args = decode_json(step.get("args_json", {}))
        if not isinstance(result, dict) or not isinstance(args, dict):
            continue
        prior = [e for e in events if e["iteration"] < step["iteration"] and e["payload"].get("version", 0) > 0]
        if not prior:
            continue
        refs = result.get("steps", [])
        requested = {args.get("step_id")} | {r.get("id") for r in refs if isinstance(r, dict)}
        owners = set()
        # The first compaction containing a raw step owns its recall attribution.
        for ref in requested - {None}:
            for e in prior:
                state = decode_json(e["payload"].get("state_json", {}))
                if ref in state.get("compacted_step_ids", []):
                    owners.add(e["payload"]["version"])
                    break
        linkage = "returned_step_owner"
        if not owners:
            owners = {prior[-1]["payload"]["version"]}
            linkage = "latest_prior_context"
        if version not in owners:
            continue
        paginated = bool(result.get("has_more_steps") or result.get("has_more") or
                         result.get("next_step_offset") or args.get("step_offset") or args.get("offset_bytes") or
                         any(r.get("has_more") for r in refs if isinstance(r, dict)))
        empty = result.get("returned_count") == 0 or ("steps" in result and not refs)
        linked.append({"step_id": step["id"], "iteration": step["iteration"],
                       "version": version, "linkage": linkage,
                       "paginated": paginated, "empty": empty})
    return {"recall_linked_count": len(linked), "recall_paginated": any(x["paginated"] for x in linked),
            "recall_empty": any(x["empty"] for x in linked), "recall_links": linked,
            "recall_metadata_is_post_decision": True}


def compaction_rows(record, counter, excluded, tail_iterations=1):
    steps, events = ordered_steps(record), compaction_events(record)
    rows = []
    snapshot_versions = {c["version"] for c in record["trajectory"].get("compactions", [])}
    event_versions = {e["payload"].get("version", 0) for e in events}
    for version in sorted(snapshot_versions - event_versions):
        excluded.append(exclusion(record, "compaction", "snapshot_without_execution_event", f"snapshot_v{version}", counter))
    for event in events:
        p = event["payload"]
        verdict = p.get("rubric_verdict")
        if verdict not in ("probed_compact", "probed_skip"):
            excluded.append(exclusion(record, "compaction", "non_semantic_verdict", event["id"], counter))
            continue
        if verdict == "probed_skip" and probe_error(p):
            item = exclusion(record, "compaction", "probe_error", event["id"], counter)
            item["error_example"] = p.get("reason", "")[:1024]
            excluded.append(item)
            continue
        if verdict == "probed_skip" and p.get("reason", "") in (
            "compaction_did_not_shrink_context", "probe_oversized_state",
        ):
            excluded.append(exclusion(record, "compaction", "validation_rejection", event["id"], counter))
            continue
        version = p.get("version", 0) or p.get("parent_version", 0) + 1
        previous = prior_state(events, event["iteration"], version, event["event_order"])
        cutoff = p["compacted_up_to_iter"] if verdict == "probed_compact" else event["iteration"] - tail_iterations
        if cutoff < 0 or cutoff > event["iteration"]:
            raise ValueError("Invalid compaction cutoff")
        text, truncated = render_state(record, steps, cutoff, event["iteration"], previous, steps)
        row = row_metadata(record, event["id"], "before_compaction_probe", cutoff,
                           sum(s["iteration"] <= cutoff for s in steps), previous,
                           "semantic_compaction_probe", "teacher", text, truncated, counter)
        row.update(row_type="compaction", state_text=text, label=int(verdict == "probed_compact"),
                   execution_iter=event["iteration"], decision_version=version,
                   recorded_compacted_up_to_iter=p["compacted_up_to_iter"],
                   cutoff_source="event" if verdict == "probed_compact" else "inferred_default_tail_iterations",
                   recorded_verdict=verdict, **recall_metadata(steps, event, events))
        rows.append(row)
    return rows


def exclusion(record, family, reason, event_id, counter):
    row = row_metadata(record, event_id, "excluded", 0, 0, None,
                       reason, "excluded", "", False, counter)
    row.update(row_type="excluded", excluded_family=family, exclusion_reason=reason)
    return row


def prefix_rows(record, counter):
    steps, events = ordered_steps(record), compaction_events(record)
    rows, seen = [], set()
    for fraction in (.25, .50, .75):
        limit = math.floor(len(steps) * fraction)
        if limit == 0:
            continue
        # Keep whole tool batches: no prefix contains a partially completed batch.
        if limit < len(steps) and steps[limit - 1]["iteration"] == steps[limit]["iteration"]:
            iteration = steps[limit]["iteration"]
            while limit and steps[limit - 1]["iteration"] == iteration:
                limit -= 1
        if not limit or limit in seen:
            continue
        seen.add(limit)
        cutoff = steps[limit - 1]["iteration"]
        previous = prior_state(events, cutoff)
        text, truncated = render_state(record, steps[:limit], cutoff, cutoff, previous)
        row = row_metadata(record, f"prefix_{limit}", "after_tool_batch", cutoff, limit, previous,
                           "clean_terminal_outcome", "trajectory_weak", text, truncated, counter)
        row.update(row_type="outcome_prefix", state_text=text,
                   label=int(outcome_class(record) == "passed"), prefix_fraction=fraction,
                   observed_step_fraction=limit / len(steps))
        rows.append(row)
    return rows


def signature(call):
    return call["tool_name"].strip(), canonical_args(call.get("args_json", ""))


def canonical_args(args):
    decoded = decode_json(args)
    return compact_json(decoded) if not isinstance(decoded, str) else decoded.strip()


def blocked(step):
    return step.get("error_text") == "loop_guard_blocked" or stored_result(step.get("result_json", "")).startswith("Loop guard notice:")


def detect_loop(history, candidates):
    """Port default DetectLoop signature logic, including batch and cycle checks."""
    if not candidates:
        return "none", []
    hs, cs = [signature(s) for s in history], [signature(c) for c in candidates]
    strikes = [s for s in history if blocked(s)]
    if any(signature(s) in cs for s in strikes):
        return "force_synthesis", [signature(s) for s in strikes if signature(s) in cs]
    signatures = hs + cs
    consecutive = 0
    for sig in reversed(signatures):
        if sig != signatures[-1]:
            break
        consecutive += 1
    detected = consecutive >= 3
    if not detected and consecutive >= 2 and len(history) >= 2:
        matching_tail = hs[len(hs) - max(0, consecutive - len(cs)):]
        if matching_tail.count(signatures[-1]) >= 2:
            outputs = [as_text(s.get("result_json", "")) for s in history if signature(s) == signatures[-1]]
            detected = len(set(outputs)) == 1
    if detected:
        return ("force_synthesis" if strikes else "intercept"), [signatures[-1]]
    for period in range(2, min(5, len(signatures) // 2) + 1):
        if signatures[-period:] == signatures[-period * 2:-period]:
            return ("force_synthesis" if strikes else "intercept"), list(dict.fromkeys(signatures[-period:]))
    return "none", []


def observation(step):
    output = stored_result(step.get("result_json", ""))
    match = re.search(r"(?m)^Exit Code:\s*(-?\d+)", output)
    exit_code = int(match.group(1)) if match else None
    stable = re.sub(r"(?m)^Duration:.*\n?", "", output)
    return stable, exit_code


def loop_features(history, candidates):
    features = []
    for c in candidates:
        matching = [s for s in history if signature(s) == signature(c) and not blocked(s)]
        outputs = [observation(s) for s in matching[-2:]]
        features.append({"tool_name": c["tool_name"], "raw_args": c.get("args_json", ""),
                         "canonical_args": canonical_args(c.get("args_json", "")),
                         "prior_matching_count": len(matching),
                         "prior_output_sha256": [hashlib.sha256(o.encode()).hexdigest() for o, _ in outputs],
                         "output_changed": outputs[-1][0] != outputs[-2][0] if len(outputs) == 2 else None,
                         "exit_codes": [code for _, code in outputs],
                         "exit_code_delta": outputs[-1][1] - outputs[-2][1] if len(outputs) == 2 and all(code is not None for _, code in outputs) else None})
    return features


def loop_rows(record, counter):
    steps, events = ordered_steps(record), compaction_events(record)
    batches = []
    seen = set()
    for event in record["trajectory"].get("events", []):
        if event.get("event_type") != "llm_response" or event.get("phase", "loop") != "loop":
            continue
        payload = decode_json(event.get("payload", {}))
        calls = payload.get("tool_calls", [])
        if not calls:
            continue
        candidates = [{"tool_name": c["function"]["name"], "args_json": c["function"]["arguments"]} for c in calls]
        iteration = event["iteration"] + 1
        key = (iteration, tuple(signature(c) for c in candidates))
        if key not in seen:
            batches.append((iteration, event["id"], candidates, "llm_response"))
            seen.add(key)
    # Older exports may lack response events; reconstruct only from candidate args.
    known_iterations = {b[0] for b in batches}
    for iteration in sorted({s["iteration"] for s in steps} - known_iterations):
        candidates = [{"tool_name": s["tool_name"], "args_json": s.get("args_json", "")} for s in steps if s["iteration"] == iteration]
        batches.append((iteration, f"candidate_i{iteration}", candidates, "step_args_fallback"))
    rows = []
    for iteration, event_id, candidates, origin in sorted(batches, key=lambda b: (b[0], b[1])):
        history = [s for s in steps if s["iteration"] < iteration]
        raw_action, suspect = detect_loop(history, candidates)
        features = loop_features(history, candidates)
        strike_count = sum(blocked(s) for s in history)
        progress = any(f["output_changed"] is True for f in features)
        supported = bool(suspect)
        for sig in suspect:
            matching = [s for s in history if signature(s) == sig]
            if any(blocked(s) for s in matching):
                continue
            outputs = [observation(s) for s in matching[-2:]]
            if len(outputs) < 2 or outputs[-1] != outputs[-2]:
                supported = False
        action = raw_action if supported and not progress else "none"
        quality = "uncertain" if raw_action != "none" and not supported and not progress else "heuristic"
        cutoff = max((s["iteration"] for s in history), default=0)
        # Windows are fixed length and use only preceding completed batches.
        window = history[-12:]
        previous = prior_state(events, iteration)
        text, truncated = render_state(record, window, cutoff, iteration, None, candidates=candidates)
        row = row_metadata(record, event_id, "before_tool_execution", cutoff, len(history), previous,
                           "detect_loop_with_observed_progress_filter", quality, text, truncated, counter)
        row.update(row_type="loop_guard", state_text=text, label=action, raw_detect_loop_action=raw_action,
                   execution_iter=iteration, candidate_source=origin, candidates=features,
                   strike_count=strike_count, window_step_count=len(window),
                   productive_progress=progress,
                   constructive_nonzero=action == "none" and any(observation(s)[1] not in (None, 0) for s in history[-2:]))
        rows.append(row)
    return rows


def balance_weights(rows):
    counts = Counter(r["task_group"] for r in rows)
    for row in rows:
        row["task_balanced_weight"] = 1 / counts[row["task_group"]]


def embedding_rows(prefixes, records, counter):
    by_run = {r["trajectory"]["run"]["id"]: r for r in records}
    buckets = defaultdict(list)
    for row in prefixes:
        buckets[(row["split"], row["benchmark"])].append(row)
    result = []
    for bucket in buckets.values():
        token_sets = [set(re.findall(r"\w+", row["state_text"].casefold())) for row in bucket]
        for index, anchor in enumerate(bucket):
            options = [j for j, row in enumerate(bucket) if row["task_group"] != anchor["task_group"]]
            if not options:
                continue
            words = token_sets[index]
            negative_index = max(options, key=lambda j: (len(words & token_sets[j]) / max(1, len(words | token_sets[j])), -j))
            negative = bucket[negative_index]
            record = by_run[anchor["run_id"]]
            steps = ordered_steps(record)[:anchor["cutoff_step"]]
            # Adjacent prefix within the same neighborhood, never after anchor time.
            positive_steps = steps[:-1] if len(steps) > 1 else steps
            positive_cutoff = positive_steps[-1]["iteration"]
            positive, truncated = render_state(record, positive_steps, positive_cutoff, positive_cutoff)
            if len(steps) == 1:
                positive = positive.replace("Candidate Steps to Compact:", "Observed tool history:")
            row = {k: v for k, v in anchor.items() if k not in ("state_text", "label", "task_balanced_weight")}
            row.update(row_type="emb_pairs", decision_phase="prefix_pair", label_source="same_task_adjacent_prefix",
                       label_quality="self_supervised", anchor_text=anchor["state_text"], positive_text=positive,
                       positive_task_group=anchor["task_group"], positive_cutoff_step=len(positive_steps),
                       positive_token_count=counter(positive), positive_rendered_bytes=len(positive.encode()),
                       positive_truncated=truncated, hard_negative_text=negative["state_text"],
                       hard_negative_task_group=negative["task_group"], hard_negative_task_id=negative["task_id"],
                       hard_negative_run_id=negative["run_id"], hard_negative_event_id=negative["event_id"],
                       hard_negative_source_sha256=negative["source_sha256"],
                       hard_negative_split=negative["split"], hard_negative_benchmark=negative["benchmark"],
                       hard_negative_token_count=negative["token_count"],
                       hard_negative_rendered_bytes=negative["rendered_bytes"],
                       hard_negative_truncated=negative["truncated"], negative_selection="same_benchmark_lexical_jaccard")
            result.append(row)
    return result


def implicit_signal(record):
    trajectory = record["trajectory"].get("trajectory", {})
    payload = decode_json(trajectory.get("payload", trajectory.get("payload_json", {})))
    if isinstance(payload, dict):
        return payload.get("eval_breakdown", {}).get("implicit_signal", 0)
    return 0


def build_dataset(rollout_dir, heldout_path, output_dir, counter, seed=42, tail_iterations=1,
                  quarantine_heldout: bool = False):
    records = []
    paths = sorted(p for p in Path(rollout_dir).glob("*.json") if p.name != "summary.json")
    if not paths:
        raise ValueError("No rollout JSON files found")
    for path in paths:
        raw = path.read_bytes()
        record = json.loads(raw)
        record["source_sha256"] = hashlib.sha256(raw).hexdigest()
        records.append(record)
    heldout_bytes = Path(heldout_path).read_bytes()
    heldout = json.loads(heldout_bytes)
    heldout = heldout["tasks"] if isinstance(heldout, dict) else heldout
    quarantined, excluded = [], []
    if quarantine_heldout:
        group_tasks(records, [])
        collisions = heldout_collisions(records, heldout)
        contaminated = {r["task_group"] for r, c in zip(records, collisions) if c["identity_fields"]}
        component_matches = defaultdict(list)
        for record, collision in zip(records, collisions):
            if collision["identity_fields"]:
                component_matches[record["task_group"]].append({"task_id": record["task_id"], **collision})
        retained = []
        for record, collision in zip(records, collisions):
            if record["task_group"] not in contaminated:
                retained.append(record)
                continue
            record["split"] = "quarantine"
            quarantined.append(record["task_id"])
            item = exclusion(record, "all", "heldout_overlap", "heldout_identity_guard", counter)
            item.update(direct_collision=bool(collision["identity_fields"]),
                        heldout_matches=component_matches[record["task_group"]])
            excluded.append(item)
        records = retained
    group_tasks(records, heldout)
    assignment = split_tasks(records, seed, heldout=heldout)
    rows = {family: [] for family in FAMILIES}
    prefixes = []
    for record in records:
        rows["compaction"].extend(compaction_rows(record, counter, excluded, tail_iterations))
        views = prefix_rows(record, counter)
        prefixes.extend(views)
        if outcome_class(record) in ("passed", "failed"):
            rows["outcome_prefix"].extend(views)
        else:
            excluded.append(exclusion(record, "outcome_prefix", outcome_class(record), "terminal_outcome", counter))
        rows["loop_guard"].extend(loop_rows(record, counter))
    rows["emb_pairs"] = embedding_rows(prefixes, records, counter)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "schema_version": SCHEMA_VERSION, "renderer_version": RENDERER_VERSION, "seed": seed,
        "tokenizer": "character_count_TEST_ONLY" if counter.char_fallback else MODEL,
        "tokenizer_add_special_tokens": False, "tail_iterations_for_skip_reconstruction": tail_iterations,
        "input_rollout_count": len(paths), "rollout_count": len(records), "task_group_count": len(assignment),
        "quarantine_heldout": quarantine_heldout, "quarantined_count": len(quarantined),
        "quarantined_task_ids": sorted(set(quarantined)),
        "heldout_source_sha256": hashlib.sha256(heldout_bytes).hexdigest(),
        "clean_outcome_counts": dict(sorted(Counter(outcome_class(r) for r in records).items())),
        "implicit_signal_nonzero_count": sum(implicit_signal(r) != 0 for r in records),
        "source_step_count": sum(len(ordered_steps(r)) for r in records),
        "compaction_event_count": sum(len(compaction_events(r)) for r in records),
        "snapshot_count": sum(len(r["trajectory"].get("compactions", [])) for r in records),
        "excluded_counts": dict(sorted(Counter(r["exclusion_reason"] for r in excluded).items())),
        "excluded_examples": {reason: [{"task_id": r["task_id"], "event_id": r["event_id"]} for r in excluded if r["exclusion_reason"] == reason][:5] for reason in sorted({r["exclusion_reason"] for r in excluded})},
        "task_group_overlap_count": 0, "heldout_overlap_count": 0,
        "splits": {}, "families": {},
        "limitations": [
            "Probe text omits provenance IDs and label fields; it is not byte-identical to the Go prompt.",
            "Skip-event attempted cutoffs are inferred: Go persists the previous cutoff and version=0.",
            "Loop actions are weak labels filtered by prior output progress, with unfiltered Go actions preserved.",
            "Recall linkage is post-decision auxiliary metadata; never include it in decision model inputs.",
            "Embedding negatives are lexical candidates, not verified semantic negatives.",
        ],
    }
    for split in SPLITS:
        members = [r for r in records if r["split"] == split]
        stats["splits"][split] = {"tasks": len(members), "task_groups": len({r["task_group"] for r in members}),
                                  "strata": dict(sorted(Counter(f"{r['benchmark']}|{outcome_class(r)}" for r in members).items()))}
    for family, values in rows.items():
        balance_weights(values)
        stats["families"][family] = {"rows": len(values), "labels": dict(sorted(Counter(str(r.get("label", "pair")) for r in values).items())),
                                      "truncated": sum(r["truncated"] for r in values),
                                      "max_rendered_bytes": max((r["rendered_bytes"] for r in values), default=0),
                                      "splits": {s: sum(r["split"] == s for r in values) for s in SPLITS}}
        for split in SPLITS:
            with (output_dir / f"{family}_{split}.jsonl").open("w") as handle:
                for row in values:
                    if row["split"] == split:
                        handle.write(compact_json(row) + "\n")
    stats["recall"] = {"source_events": sum(s["tool_name"] == "recall_task_history" for r in records for s in ordered_steps(r)),
                       "linked_count": sum(r["recall_linked_count"] for r in rows["compaction"]),
                       "compactions_paginated": sum(r["recall_paginated"] for r in rows["compaction"]),
                       "compactions_empty": sum(r["recall_empty"] for r in rows["compaction"])}
    stats["loop_raw_actions"] = dict(Counter(r["raw_detect_loop_action"] for r in rows["loop_guard"]))
    stats["loop_progress_filtered_count"] = sum(r["label"] != r["raw_detect_loop_action"] for r in rows["loop_guard"])
    with (output_dir / "excluded.jsonl").open("w") as handle:
        for row in excluded:
            handle.write(compact_json(row) + "\n")
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=ROOT / "eval/results/train/batch_train_pool_1794581821")
    parser.add_argument("--heldout", type=Path, default=ROOT / "eval/tasks/heldout/heldout_100.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval/results/jev_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tail-iterations", type=int, default=1)
    parser.add_argument("--char-fallback", action="store_true", help="Synthetic tests only: count characters instead of real tokens")
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--quarantine-heldout", action="store_true",
                        help="Exclude entire task groups colliding with held-out identities; audit in excluded.jsonl")
    args = parser.parse_args()
    if args.tail_iterations < 0:
        parser.error("--tail-iterations must be nonnegative")
    counter = TokenCounter(args.char_fallback, args.tokenizer_cache)
    stats = build_dataset(args.rollout_dir, args.heldout, args.output_dir, counter, args.seed, args.tail_iterations,
                          quarantine_heldout=args.quarantine_heldout)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
