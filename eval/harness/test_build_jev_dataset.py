"""Regression tests for temporal reconstruction and task-level data isolation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("build_jev_dataset", Path(__file__).with_name("build_jev_dataset.py"))
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)

REQUIRED = {
    "schema_version", "task_id", "task_group", "run_id", "event_id",
    "decision_phase", "cutoff_iter", "cutoff_step", "prev_compaction_version",
    "label_source", "label_quality", "token_count", "rendered_bytes",
    "source_sha256", "renderer_version",
}


def step(i, command=None, output=None, tool="execute", **extra):
    return {"id": f"run_secret_i{i}", "iteration": i, "step_index": 0,
            "tool_name": tool, "args_json": {"command": command or f"command_{i}"},
            "result_json": {"bob_tool_result_version": 1, "result": {
                "content": output if output is not None else f"Exit Code: 0\nDuration: 1ms\nOUTPUT_{i}"}},
            "status": "COMPLETED", **extra}


def event(version, iteration, cutoff, verdict="probed_compact", **extra):
    payload = {"version": version, "parent_version": max(0, version - 1),
               "compacted_up_to_iter": cutoff, "rubric_verdict": verdict,
               "reason": "SECRET_REASON", "should_compact": verdict == "probed_compact"}
    if version:
        payload["state_json"] = {"original_objective": "objective", "current_subgoal": f"STATE_V{version}",
                                 "rubric_verdict": "SECRET_VERDICT", "model": "SECRET_MODEL",
                                 "run_id": "SECRET_RUN", "compacted_step_ids": [f"run_secret_i{i}" for i in range(1, cutoff + 1)]}
    payload.update(extra)
    return {"id": f"event_{version}_{iteration}", "event_type": "compaction",
            "iteration": iteration, "run_id": "run_secret", "payload": payload}


def record(task_id="task", instruction="objective", count=8, **extra):
    value = {"task_id": task_id, "instruction": instruction, "benchmark": "frames",
             "benchmark_reward": 1, "grading_status": "graded", "eval_details": "SECRET_EVAL",
             "source_sha256": "a" * 64, "task_group": f"group_{task_id}", "split": "train",
             "trajectory": {"run": {"id": f"run_{task_id}", "status": "SECRET_TERMINAL",
                                    "context_json": json.dumps([{"role": "user", "content": instruction},
                                                                 {"role": "assistant", "content": "SECRET_FINAL"}])},
                            "steps": [step(i) for i in range(1, count + 1)],
                            "events": [], "compactions": [], "trajectory": {}}}
    value.update(extra)
    return value


@pytest.fixture
def counter():
    return b.TokenCounter(char_fallback=True)


def test_future_and_label_data_never_rendered(counter):
    r = record()
    r["trajectory"]["events"] = [event(1, 3, 2), event(2, 6, 5), event(3, 8, 7)]
    rows = b.compaction_rows(r, counter, [])
    text = rows[1]["state_text"]
    assert "STATE_V1" in text
    for forbidden in ("STATE_V2", "STATE_V3", "SECRET_REASON", "SECRET_VERDICT", "SECRET_MODEL",
                      "SECRET_RUN", "SECRET_EVAL", "SECRET_TERMINAL", "SECRET_FINAL",
                      "benchmark_reward", "run_secret", "OUTPUT_6", "OUTPUT_7", "OUTPUT_8", "command_6"):
        assert forbidden not in text
    assert "OUTPUT_3" in text and "OUTPUT_5" in text
    assert "OUTPUT_1" not in text
    assert "Retained Raw Tail Steps" in text
    assert rows[1]["execution_iter"] == 6
    assert rows[1]["cutoff_iter"] == 5
    assert rows[1]["prev_compaction_version"] == 1


def test_future_mutations_do_not_change_compaction_input(counter):
    r = record()
    r["trajectory"]["events"] = [event(1, 3, 2)]
    original = b.compaction_rows(r, counter, [])[0]["state_text"]
    r["benchmark_reward"] = 0
    r["eval_details"] = "OTHER_EVAL"
    r["trajectory"]["run"]["status"] = "FAILED"
    r["trajectory"]["steps"][3]["args_json"] = {"command": "future mutation"}
    r["trajectory"]["events"][0]["payload"]["state_json"]["current_subgoal"] = "CURRENT_OUTPUT"
    r["trajectory"]["events"][0]["payload"]["reason"] = "CURRENT_REASON"
    assert b.compaction_rows(r, counter, [])[0]["state_text"] == original


def test_dedup_authoritative_events_and_orphan_snapshots(counter):
    r = record()
    e = event(1, 3, 2)
    r["trajectory"]["events"] = [e, copy.deepcopy(e)]
    r["trajectory"]["compactions"] = [dict(e["payload"], state_json={"current_subgoal": "WRONG_SNAPSHOT"}),
                                       {"version": 99, "compacted_up_to_iter": 8}]
    excluded = []
    rows = b.compaction_rows(r, counter, excluded)
    assert len(rows) == 1
    assert rows[0]["event_id"] == e["id"]
    assert rows[0]["execution_iter"] == 3
    assert len(excluded) == 1
    assert excluded[0]["exclusion_reason"] == "snapshot_without_execution_event"


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_nested_result_and_args_and_context_parsing(counter, depth):
    r = record(count=3)
    r["trajectory"]["events"] = [event(1, 3, 2)]
    s = r["trajectory"]["steps"][0]
    for _ in range(depth):
        s["result_json"] = json.dumps(s["result_json"])
        s["args_json"] = json.dumps(s["args_json"])
        r["trajectory"]["run"]["context_json"] = json.dumps(r["trajectory"]["run"]["context_json"])
    text = b.compaction_rows(r, counter, [])[0]["state_text"]
    assert "OUTPUT_1" in text
    assert '"command":"command_1"' in text
    assert "Original Objective: objective" in text
    assert "SECRET_FINAL" not in text


@pytest.mark.parametrize("value,expected", [
    ("legacy plain output", "legacy plain output"),
    ('{ "answer": 42 }', '{ "answer": 42 }'),
    ({"bob_tool_result_version": 2, "result": {"content": "decoded"}}, "decoded"),
    ("", ""),
])
def test_decode_stored_result_legacy(value, expected):
    assert b.stored_result(value) == expected


@pytest.mark.parametrize("reason", [
    "probe_error: invalid JSON", 'parse error raw_preview={"should_compact": true}',
    r'failed to parse raw_preview={\"should_compact\": true}',
])
def test_probe_error_skip_excluded(counter, reason):
    r = record()
    r["trajectory"]["events"] = [event(0, 3, 0, "probed_skip", reason=reason)]
    excluded = []
    assert b.compaction_rows(r, counter, excluded) == []
    assert excluded[0]["exclusion_reason"] == "probe_error"
    assert REQUIRED <= excluded[0].keys()


def test_skip_uses_attempted_boundary_and_prior_version(counter):
    r = record()
    r["trajectory"]["events"] = [event(1, 3, 2), event(0, 6, 2, "probed_skip", parent_version=1)]
    skip = b.compaction_rows(r, counter, [])[1]
    assert skip["label"] == 0
    assert skip["recorded_compacted_up_to_iter"] == 2
    assert skip["cutoff_iter"] == 5
    assert skip["prev_compaction_version"] == 1
    assert skip["decision_version"] == 2
    assert "STATE_V1" in skip["state_text"]
    assert "OUTPUT_5" in skip["state_text"]
    assert "OUTPUT_6" not in skip["state_text"]


def test_recall_pagination_empty_and_owner_links_preserve_label(counter):
    r = record(count=7)
    r["trajectory"]["events"] = [event(1, 3, 2), event(2, 5, 4)]
    r["trajectory"]["steps"][5] = step(6, tool="recall_task_history", args_json={"step_offset": 2},
        output=json.dumps({"returned_count": 1, "has_more_steps": True,
                           "steps": [{"id": "run_secret_i1", "has_more": True}]}))
    r["trajectory"]["steps"][6] = step(7, tool="recall_task_history", args_json={"query": "missing"},
        output=json.dumps({"returned_count": 0, "steps": []}))
    first, second = b.compaction_rows(r, counter, [])
    assert first["recall_linked_count"] == second["recall_linked_count"] == 1
    assert first["recall_paginated"] and not first["recall_empty"]
    assert second["recall_empty"] and not second["recall_paginated"]
    assert first["label"] == second["label"] == 1
    assert "recall_linked_count" not in first["state_text"]


@pytest.mark.parametrize("extra,expected", [
    ({"benchmark": "swe-bench-lite", "eval_details": "Patch generated -> Docker eval failed to produce report.json"}, "excluded_swe_missing_report"),
    ({"benchmark": "intercode-bash", "grading_status": "oracle_error"}, "excluded_oracle_error"),
    ({"benchmark_reward": 0}, "failed"), ({"benchmark_reward": 1}, "passed"),
    ({"benchmark_reward": None}, "excluded_ungraded"),
])
def test_clean_outcome_classification(extra, expected):
    assert b.outcome_class(record(**extra)) == expected


@pytest.mark.parametrize("collision", [
    {"task_id": "held", "instruction": "different"},
    {"task_id": "different", "instruction": "  SAME\n\t Instruction "},
    {"task_id": "different", "instruction": "ＳＡＭＥ instruction"},
])
def test_heldout_id_or_normalized_instruction_raises(collision):
    with pytest.raises(ValueError, match="Held-out overlap"):
        b.group_tasks([record(**collision)], [{"task_id": "held", "instruction": "same instruction"}])


def test_transitive_groups_and_deterministic_stratification():
    records = [record(f"id_{i}", f"instruction {i}") for i in range(40)]
    records += [record("id_0", "alias instruction"), record("another_id", "ALIAS\nINSTRUCTION")]
    b.group_tasks(records, [])
    assert records[0]["task_group"] == records[-1]["task_group"] == records[-2]["task_group"]
    assignment = b.split_tasks(records)
    again = copy.deepcopy(list(reversed(records)))
    b.group_tasks(again, [])
    assert b.split_tasks(again) == assignment
    groups = {s: {r["task_group"] for r in records if r["split"] == s} for s in b.SPLITS}
    assert [len(groups[s]) for s in b.SPLITS] == [28, 6, 6]
    for i, split in enumerate(b.SPLITS):
        for other in b.SPLITS[i + 1:]:
            assert groups[split].isdisjoint(groups[other])
            assert {r["task_id"] for r in records if r["split"] == split}.isdisjoint(
                {r["task_id"] for r in records if r["split"] == other})


def test_utf8_go_probe_budgets_and_truncation(counter):
    r = record(instruction="界" * 3000, count=100)
    for s in r["trajectory"]["steps"]:
        s["args_json"] = {"command": "界" * 1000}
        s["result_json"] = "😀" * 2000
    text, truncated = b.render_state(r, b.ordered_steps(r), 100, 100)
    assert truncated
    assert len(text.encode()) <= 32768
    assert "...[truncated for probe]" in text
    assert text.endswith("\n...[truncated]")
    assert "�" not in text


def test_prefix_caps_and_batch_boundaries(counter):
    r = record(count=8)
    rows = b.prefix_rows(r, counter)
    assert [row["cutoff_step"] for row in rows] == [2, 4, 6]
    for row in rows:
        assert row["observed_step_fraction"] <= row["prefix_fraction"]
        assert f"OUTPUT_{row['cutoff_step'] + 1}" not in row["state_text"]
    r["trajectory"]["steps"][2]["iteration"] = 2
    assert b.prefix_rows(r, counter)[0]["cutoff_step"] == 1
    assert b.prefix_rows(record(count=1), counter) == []


def test_loop_guard_canonicalization_cycle_and_strikes():
    a = step(1, "same")
    a["args_json"] = '{ "z": 1, "a": 2 }'
    candidate = {"tool_name": "execute", "args_json": {"a": 2, "z": 1}}
    assert b.detect_loop([a, copy.deepcopy(a)], [candidate])[0] == "intercept"
    a["error_text"] = "loop_guard_blocked"
    assert b.detect_loop([a], [candidate])[0] == "force_synthesis"
    x, y = step(1, "a"), step(2, "b")
    assert b.detect_loop([x, y, x], [y])[0] == "intercept"
    assert b.detect_loop([x], [y])[0] == "none"


def test_loop_decision_uses_no_candidate_results_and_filters_progress(counter):
    r = record(count=3)
    r["trajectory"]["steps"] = [step(1, "same", "Exit Code: 1\nDuration: 1ms\nfirst failure"),
                                    step(2, "same", "Exit Code: 0\nDuration: 9ms\nprogress"),
                                    step(3, "same", "SECRET_CANDIDATE_OUTPUT")]
    row = b.loop_rows(r, counter)[-1]
    assert row["raw_detect_loop_action"] == "intercept"
    assert row["label"] == "none"
    assert row["productive_progress"]
    assert row["candidates"][0]["exit_code_delta"] == -1
    assert "SECRET_CANDIDATE_OUTPUT" not in row["state_text"]
    r["trajectory"]["steps"][1]["result_json"] = "Exit Code: 1\nDuration: 9ms\nfirst failure"
    row = b.loop_rows(r, counter)[-1]
    assert row["label"] == "intercept"
    assert row["candidates"][0]["output_changed"] is False


def test_constructive_nonzero_and_unexecuted_force_synthesis(counter):
    r = record(count=2)
    r["trajectory"]["steps"] = [step(1, "test", "Exit Code: 1\nfailing test"), step(2, "fix")]
    rows = b.loop_rows(r, counter)
    assert rows[1]["label"] == "none"
    assert rows[1]["constructive_nonzero"]
    r["trajectory"]["steps"][1]["error_text"] = "loop_guard_blocked"
    r["trajectory"]["events"] = [{"id": "force_event", "event_type": "llm_response", "iteration": 2,
                                    "phase": "loop", "payload": {"tool_calls": [{"function": {
                                        "name": "execute", "arguments": '{"command":"fix"}'}}]}}]
    row = b.loop_rows(r, counter)[-1]
    assert row["label"] == "force_synthesis"
    assert row["strike_count"] == 1
    assert row["event_id"] == "force_event"
    assert row["cutoff_step"] == 2


def test_same_iteration_candidates_do_not_observe_one_another(counter):
    r = record(count=2)
    r["trajectory"]["steps"][1]["iteration"] = 1
    rows = b.loop_rows(r, counter)
    assert len(rows) == 1
    assert rows[0]["cutoff_step"] == 0
    assert len(rows[0]["candidates"]) == 2
    assert "OUTPUT_1" not in rows[0]["state_text"]
    assert "OUTPUT_2" not in rows[0]["state_text"]


def test_exact_tokenizer_interface(monkeypatch):
    import transformers
    seen = []

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return [11, 22, 33]

    def from_pretrained(name, **kwargs):
        seen.append(name)
        return FakeTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    counter = b.TokenCounter()
    assert counter("text whose character count is different") == 3
    assert seen == ["Qwen/Qwen3.5-0.8B-Base"]


def test_end_to_end_files_weights_exclusions_and_pair_isolation(tmp_path, counter):
    source = tmp_path / "rollouts"
    source.mkdir()
    for i in range(30):
        r = record(f"task_{i}", f"instruction {i}", benchmark_reward=i % 2)
        r["trajectory"]["events"] = [event(1, 3, 2)]
        if i == 0:
            r.update(benchmark="swe-bench-lite", eval_details="Docker eval failed to produce report.json")
        if i == 1:
            r.update(benchmark="intercode-bash", grading_status="oracle_error")
        if i == 2:
            r["trajectory"]["events"].append(event(0, 6, 2, "probed_skip", reason="probe_error: parse", parent_version=1))
        (source / f"{i:02}.json").write_text(json.dumps(r))
    (source / "summary.json").write_text('{"not_a_task":true}')
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"tasks": [{"task_id": "heldout", "instruction": "never seen"}]}))
    output = tmp_path / "out"
    stats = b.build_dataset(source, heldout, output, counter)
    assert json.loads((output / "stats.json").read_text()) == stats
    assert stats["rollout_count"] == 30
    assert stats["excluded_counts"] == {"excluded_oracle_error": 1, "excluded_swe_missing_report": 1, "probe_error": 1}
    assert sum(stats["clean_outcome_counts"][k] for k in ("passed", "failed")) == 28
    groups = {s: set() for s in b.SPLITS}
    for family in b.FAMILIES:
        totals = {}
        for split in b.SPLITS:
            rows = [json.loads(line) for line in (output / f"{family}_{split}.jsonl").read_text().splitlines()]
            for row in rows:
                assert REQUIRED <= row.keys()
                assert row["row_type"] == family
                assert row["split"] == split
                assert row["rendered_bytes"] == len(row.get("state_text", row.get("anchor_text")).encode())
                assert row["rendered_bytes"] <= b.MAX_BYTES
                groups[split].add(row["task_group"])
                totals[row["task_group"]] = totals.get(row["task_group"], 0) + row["task_balanced_weight"]
                if family == "outcome_prefix":
                    assert row["task_id"] not in ("task_0", "task_1")
                if family == "emb_pairs":
                    assert row["task_group"] == row["positive_task_group"]
                    assert row["task_group"] != row["hard_negative_task_group"]
                    assert row["hard_negative_split"] == split
                    assert row["hard_negative_benchmark"] == row["benchmark"]
                    assert row["positive_cutoff_step"] <= row["cutoff_step"]
        assert all(weight == pytest.approx(1) for weight in totals.values())
    for i, split in enumerate(b.SPLITS):
        for other in b.SPLITS[i + 1:]:
            assert groups[split].isdisjoint(groups[other])
    for item in map(json.loads, (output / "excluded.jsonl").read_text().splitlines()):
        assert REQUIRED <= item.keys()
    first_bytes = {p.name: p.read_bytes() for p in output.iterdir()}
    b.build_dataset(source, heldout, output, counter)
    assert first_bytes == {p.name: p.read_bytes() for p in output.iterdir()}


def test_holdout_failure_writes_no_outputs(tmp_path, counter):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rollout.json").write_text(json.dumps(record("duplicate")))
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"tasks": [{"task_id": "duplicate", "instruction": "other"}]}))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Held-out overlap"):
        b.build_dataset(source, heldout, output, counter)
    assert not output.exists()


@pytest.mark.parametrize("key", ["payload", "payload_json"])
def test_implicit_signal_is_read_from_trajectory_payload(key):
    r = record()
    payload = {"eval_breakdown": {"implicit_signal": 0.25}}
    r["trajectory"]["trajectory"][key] = payload if key == "payload" else json.dumps(payload)
    assert b.implicit_signal(r) == 0.25


def test_previous_state_must_precede_event_time_and_version(counter):
    r = record()
    r["trajectory"]["events"] = [event(2, 2, 1), event(1, 4, 3)]
    rows = b.compaction_rows(r, counter, [])
    assert rows[1]["prev_compaction_version"] == 0
    assert "STATE_V2" not in rows[1]["state_text"]


@pytest.mark.parametrize("reason", ["probe_oversized_state", "compaction_did_not_shrink_context"])
def test_nonsemantic_validation_skip_exclusion(counter, reason):
    r = record()
    r["trajectory"]["events"] = [event(0, 3, 0, "probed_skip", reason=reason)]
    excluded = []
    assert not b.compaction_rows(r, counter, excluded)
    assert excluded[0]["exclusion_reason"] == "validation_rejection"


@pytest.mark.parametrize("training,heldout,field", [
    ({"gold_command": "cat /flag"}, {"gold_command": "cat /flag"}, "gold_command"),
    ({"gold_flag": "picoCTF{same}"}, {"gold_flag": "picoCTF{same}"}, "gold_flag"),
    ({"gold_answer": " New\n York "}, {"gold_answer": "new york"}, "gold_answer"),
    ({"gold_answer": 0}, {"gold_answer": "0"}, "gold_answer"),
    ({"instance_id": "django__django-123"}, {"instance_id": "django__django-123"}, "instance_id"),
    ({"repo": "org/repo", "base_commit": "abc"}, {"repo": "org/repo", "base_commit": "abc"}, "repo+base_commit"),
    ({"metadata": {"gold_flag": "picoCTF{same}"}}, {"gold_flag": "picoCTF{same}"}, "gold_flag"),
    ({"instance_id": "instance"}, {"metadata": {"instance_id": "instance"}}, "instance_id"),
])
def test_metadata_gold_identity_overlap_raises(training, heldout, field):
    r = record("train", "training instruction", **training)
    h = {"task_id": "heldout", "instruction": "held-out instruction", **heldout}
    with pytest.raises(ValueError, match="Held-out overlap"):
        b.group_tasks([r], [h])
    collision = b.heldout_collisions([r], [h])[0]
    assert collision == {"heldout_task_ids": ["heldout"], "identity_fields": [field]}


@pytest.mark.parametrize("training,heldout", [
    ({"gold_command": ""}, {"gold_command": ""}),
    ({"gold_flag": None}, {"gold_flag": None}),
    ({"gold_answer": " "}, {"gold_answer": "\t"}),
    ({"gold_answer": "answer"}, {}),
    ({}, {"gold_answer": "answer"}),
    ({"repo": "repo"}, {"repo": "repo"}),
    ({"repo": "repo", "base_commit": "a"}, {"repo": "repo", "base_commit": "b"}),
    ({"repo": "repo1", "base_commit": "a"}, {"repo": "repo2", "base_commit": "a"}),
    ({"gold_flag": "picoCTF{UPPER}"}, {"gold_flag": "picoCTF{upper}"}),
    ({"gold_command": "cat X"}, {"gold_command": "cat x"}),
])
def test_missing_or_distinct_gold_identities_do_not_collide(training, heldout):
    r = record("train", "training instruction", **training)
    h = {"task_id": "heldout", "instruction": "held-out instruction", **heldout}
    b.group_tasks([r], [h])
    assert b.heldout_collisions([r], [h])[0]["identity_fields"] == []


def test_split_rechecks_holdout_and_prevents_identity_groups_diverging():
    r = record("train", "training instruction", gold_flag="picoCTF{same}")
    h = {"task_id": "heldout", "instruction": "held-out instruction", "gold_flag": "picoCTF{same}"}
    with pytest.raises(ValueError, match="Held-out overlap"):
        b.split_tasks([r], heldout=[h])
    sibling = record("sibling", "different instruction", gold_flag="picoCTF{same}")
    with pytest.raises(ValueError, match="different task groups"):
        b.split_tasks([r, sibling])
    b.group_tasks([r, sibling], [])
    b.split_tasks([r, sibling])
    assert r["task_group"] == sibling["task_group"]
    assert r["split"] == sibling["split"]


def quarantine_fixture(tmp_path):
    source = tmp_path / "rollouts"
    source.mkdir()
    # A direct metadata collision pulls in two siblings through ID and normalized
    # instruction edges, plus another via a gold identity edge.
    records = [record("direct", "first instruction", gold_flag="picoCTF{heldout}"),
               record("direct", "second instruction"),
               record("sibling", " SECOND\nINSTRUCTION ", instance_id="shared_instance"),
               record("metadata_sibling", "third instruction", instance_id="shared_instance"),
               record("instruction_collision", " HELD-OUT\nINSTRUCTION "),
               record("heldout_id", "different heldout id instruction")]
    records.extend(record(f"clean_{i}", f"clean instruction {i}", benchmark_reward=i % 2) for i in range(20))
    for i, r in enumerate(records):
        r["trajectory"]["run"]["id"] = f"run_{i}"
        r["trajectory"]["events"] = [event(1, 3, 2)]
        (source / f"{i:02}.json").write_text(json.dumps(r))
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"tasks": [{"task_id": "heldout_id", "instruction": "held-out instruction",
                                              "gold_flag": "picoCTF{heldout}"}]}))
    return source, heldout, tmp_path / "out"


def test_quarantine_entire_connected_components_with_audit(tmp_path, counter):
    source, heldout, output = quarantine_fixture(tmp_path)
    stats = b.build_dataset(source, heldout, output, counter, quarantine_heldout=True)
    assert json.loads((output / "stats.json").read_text()) == stats
    assert stats["quarantine_heldout"] is True
    assert stats["input_rollout_count"] == 26
    assert stats["rollout_count"] == 20
    assert stats["quarantined_count"] == 6
    assert stats["quarantined_task_ids"] == ["direct", "heldout_id", "instruction_collision", "metadata_sibling", "sibling"]
    assert stats["excluded_counts"]["heldout_overlap"] == 6
    assert stats["heldout_overlap_count"] == stats["task_group_overlap_count"] == 0
    excluded = [json.loads(line) for line in (output / "excluded.jsonl").read_text().splitlines()]
    assert len(excluded) == 6
    assert all(r["exclusion_reason"] == "heldout_overlap" and r["split"] == "quarantine" for r in excluded)
    assert sum(r["direct_collision"] for r in excluded) == 3
    sibling = next(r for r in excluded if r["task_id"] == "metadata_sibling")
    assert sibling["heldout_matches"] == [{"task_id": "direct", "heldout_task_ids": ["heldout_id"], "identity_fields": ["gold_flag"]}]
    assert all(REQUIRED <= r.keys() for r in excluded)
    quarantined_groups = {r["task_group"] for r in excluded}
    groups = {s: set() for s in b.SPLITS}
    for family in b.FAMILIES:
        for split in b.SPLITS:
            rows = [json.loads(line) for line in (output / f"{family}_{split}.jsonl").read_text().splitlines()]
            assert rows
            for row in rows:
                assert row["task_id"].startswith("clean_")
                assert row["task_group"] not in quarantined_groups
                groups[split].add(row["task_group"])
                if family == "emb_pairs":
                    assert row["hard_negative_task_id"].startswith("clean_")
                    assert row["hard_negative_task_group"] not in quarantined_groups
    for i, split in enumerate(b.SPLITS):
        for other in b.SPLITS[i + 1:]:
            assert groups[split].isdisjoint(groups[other])
    first_bytes = {p.name: p.read_bytes() for p in output.iterdir()}
    b.build_dataset(source, heldout, output, counter, quarantine_heldout=True)
    assert first_bytes == {p.name: p.read_bytes() for p in output.iterdir()}


def test_strict_metadata_guard_preserves_existing_output(tmp_path, counter):
    source, heldout, output = quarantine_fixture(tmp_path)
    output.mkdir()
    sentinel = output / "stats.json"
    sentinel.write_text("existing data")
    with pytest.raises(ValueError, match="Held-out overlap"):
        b.build_dataset(source, heldout, output, counter)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_text() == "existing data"


def test_quarantine_all_records_produces_audited_empty_splits(tmp_path, counter):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rollout.json").write_text(json.dumps(record("duplicate")))
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"tasks": [{"task_id": "duplicate", "instruction": "other"}]}))
    output = tmp_path / "out"
    stats = b.build_dataset(source, heldout, output, counter, quarantine_heldout=True)
    assert stats["quarantined_count"] == 1
    assert stats["rollout_count"] == 0
    for family in b.FAMILIES:
        for split in b.SPLITS:
            assert (output / f"{family}_{split}.jsonl").read_text() == ""


def test_implicit_signal_prefers_payload_and_defaults_missing():
    r = record()
    assert b.implicit_signal(r) == 0
    r["trajectory"]["trajectory"] = {"payload": {"eval_breakdown": {"implicit_signal": .5}},
                                         "payload_json": json.dumps({"eval_breakdown": {"implicit_signal": -.5}})}
    assert b.implicit_signal(r) == .5
