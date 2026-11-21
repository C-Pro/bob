"""Integrity, zero-leakage, idempotency, and visual-asset tests for Jev v2 extension dataset."""
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
from eval.harness import train_provenance

SPEC = importlib.util.spec_from_file_location(
    "build_jev_generic_and_multimodal",
    Path(__file__).with_name("build_jev_generic_and_multimodal.py"),
)
ext = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ext)


def test_certified_train_control_mining_preserves_partitions(tmp_path):
    task = {"task_id": "train_task", "instruction": "inspect repository"}
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"tasks": [task]}))
    certificate = train_provenance.generate_certificate(manifest)
    entry = certificate["assignments"]["train_task"]
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    steps = [{"id": f"s{i}", "iteration": i, "step_index": 0, "tool_name": "read_file",
              "args_json": {"path": f"file{i}.go"}, "result_json": {"content": "short"},
              "status": "COMPLETED"} for i in range(1, 13)]
    rollout = {"task_id": "train_task", "instruction": "inspect repository", "benchmark": "frames",
               "benchmark_reward": 1, "grading_status": "graded",
               "trajectory": {"run": {"id": "run_train_task", "context_json": "[]"},
                              "steps": steps, "events": [], "compactions": [], "trajectory": {}}}
    (rollout_dir / "train_task.json").write_text(json.dumps(rollout))
    rows = ext.mine_train_control_rows(rollout_dir, certificate, ext.TokenCounter(char_fallback=True))
    compaction = next(row for row in rows if row["row_type"] == "compaction")
    assert compaction["decision_reason"] == "low_byte_growth_or_utility"
    assert compaction["label"] == "skip"
    assert compaction["partition"] == entry["partition"]
    assert compaction["group_id"] == entry["group_id"]
    assert compaction["rollout_sha256"]


def test_certified_train_group_assignment_is_inherited_by_extension_rows():
    partition = train_provenance.PARTITIONS[2]
    group_id = "certified_group"
    train_row = {"task_group": group_id, "split_group": group_id, "split": partition}
    generated_row = {"task_group": "derived_extension_row", "split_group": group_id, "split": "train"}
    assignments = ext.assign_split_groups(
        [train_row, generated_row], {group_id: partition}, certified=True
    )
    assert assignments[group_id] == partition
    assert train_row["split"] == generated_row["split"] == partition


def _write_v1_fixture(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "cal", "dev"):
        comp = {
            "task_group": f"tg_v1_{split}",
            "split": split,
            "label": 1,
            "state_text": f"v1 compaction {split}",
        }
        (directory / f"compaction_{split}.jsonl").write_text(json.dumps(comp) + "\n")
        loop = {
            "schema_version": "jev-dataset-v1",
            "task_group": f"tg_v1_{split}",
            "split": split,
            "event_id": f"v1_loop_{split}",
            "label": "none",
            "raw_detect_loop_action": "none",
            "state_text": f"v1 loop {split}",
        }
        (directory / f"loop_guard_{split}.jsonl").write_text(json.dumps(loop) + "\n")
    heldout = directory / "heldout.json"
    heldout.write_text(json.dumps([{"task_id": "heldout_1", "instruction": "Solve heldout task 1"}]))
    return heldout


def test_build_extension_dataset_zero_leakage_and_idempotent(tmp_path):
    heldout = _write_v1_fixture(tmp_path)
    counter = ext.TokenCounter(char_fallback=True)

    stats1 = ext.build_extension_dataset(tmp_path, heldout, counter, seed=42)
    assert stats1["split_group_overlap_count"] == 0
    assert stats1["heldout_overlap_count"] == 0

    manifest1 = json.loads((tmp_path / "extension_manifest_v2.json").read_text())

    # Second run must be 100% idempotent (no duplicate loop rows, identical file hashes).
    ext.build_extension_dataset(tmp_path, heldout, counter, seed=42)
    manifest2 = json.loads((tmp_path / "extension_manifest_v2.json").read_text())
    assert manifest1["v2_output_hashes"] == manifest2["v2_output_hashes"]

    # Check v1 compaction untouched and loop_guard preserved original rows + added 3 classes.
    for split in ("train", "cal", "dev"):
        comp_rows = [json.loads(line) for line in (tmp_path / f"compaction_{split}.jsonl").read_text().splitlines() if line]
        assert len(comp_rows) == 1 and comp_rows[0]["task_group"] == f"tg_v1_{split}"

        loop_rows = [json.loads(line) for line in (tmp_path / f"loop_guard_{split}.jsonl").read_text().splitlines() if line]
        v1_loops = [r for r in loop_rows if r.get("schema_version") == "jev-dataset-v1"]
        v2_loops = [r for r in loop_rows if r.get("schema_version") == ext.SCHEMA_VERSION]
        assert len(v1_loops) == 1
        assert {r["label"] for r in v2_loops} == {"none", "intercept", "force_synthesis"}

    # Check 0% split_group and task_group overlap across splits for all new families.
    by_split_sg = {s: set() for s in ("train", "cal", "dev")}
    by_split_tg = {s: set() for s in ("train", "cal", "dev")}
    for fam in ("generic_decisions", "emb_pairs_bilingual", "multimodal_emb_pairs", "multimodal_decisions"):
        for split in ("train", "cal", "dev"):
            rows = [json.loads(line) for line in (tmp_path / f"{fam}_{split}.jsonl").read_text().splitlines() if line]
            assert rows, f"Empty {fam}_{split}"
            for r in rows:
                assert r["split"] == split
                by_split_sg[split].add(r["split_group"])
                by_split_tg[split].add(r["task_group"])
                if "hard_negative_task_group" in r:
                    assert r["hard_negative_split"] == split
                    assert r["hard_negative_task_group"] != r["task_group"]
                    by_split_tg[split].add(r["hard_negative_task_group"])
                if "options" in r:
                    opt_names = [name for name, _ in r["options"]]
                    assert len(opt_names) == len(set(opt_names))
                    assert r["label"] in opt_names

    for a in ("train", "cal", "dev"):
        for b in ("train", "cal", "dev"):
            if a < b:
                assert not (by_split_sg[a] & by_split_sg[b])
                assert not (by_split_tg[a] & by_split_tg[b])


def test_multimodal_assets_integrity_and_counterfactuals(tmp_path):
    heldout = _write_v1_fixture(tmp_path)
    counter = ext.TokenCounter(char_fallback=True)
    ext.build_extension_dataset(tmp_path, heldout, counter, seed=42)

    all_mm_dec = []
    for split in ("train", "cal", "dev"):
        for fam in ("multimodal_decisions", "multimodal_emb_pairs"):
            rows = [json.loads(line) for line in (tmp_path / f"{fam}_{split}.jsonl").read_text().splitlines() if line]
            if fam == "multimodal_decisions":
                all_mm_dec.extend(rows)
            for r in rows:
                imgs = r.get("state_images") or r.get("anchor_images") or []
                assert imgs
                for img_ref in imgs:
                    p = tmp_path / img_ref["path"]
                    assert p.exists()
                    # Filenames must be opaque hashes, never containing labels.
                    assert str(r.get("label", "")) not in p.name or str(r.get("label", "")) == ""
                    raw = p.read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == img_ref["sha256"]
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        assert im.size == (img_ref["width"], img_ref["height"])
                        assert im.width * im.height <= 384 * 16 * 16

    # Verify paired counterfactuals exist: identical (state_text, question), different images and labels.
    by_prompt = {}
    for r in all_mm_dec:
        by_prompt.setdefault((r["split_group"], r["state_text"], r["question"]), []).append(r)
    paired = [grp for grp in by_prompt.values() if len(grp) >= 2]
    assert len(paired) >= 12
    for grp in paired:
        labels = {r["label"] for r in grp}
        img_hashes = {tuple(im["sha256"] for im in r["state_images"]) for r in grp}
        assert len(labels) >= 2
        assert len(img_hashes) >= 2


def test_bilingual_and_generic_coverage_and_heldout_guard(tmp_path):
    heldout = _write_v1_fixture(tmp_path)
    counter = ext.TokenCounter(char_fallback=True)
    stats = ext.build_extension_dataset(tmp_path, heldout, counter, seed=42)

    gen_langs = set(stats["families"]["generic_decisions"]["languages"])
    assert {"en", "ru"} <= gen_langs
    bi_langs = set(stats["families"]["emb_pairs_bilingual"]["languages"])
    assert {"en", "ru", "en_ru", "ru_en"} <= bi_langs
    gen_subs = set(stats["families"]["generic_decisions"]["subfamilies"])
    assert {"policy_rules", "policy_abstention", "nli", "boolean_qa", "chat_urgency",
            "confirmation_gate", "prompt_injection", "memory_relevance", "verification_readiness"} <= gen_subs

    # Conflicting v1 assignments fail loudly.
    (tmp_path / "compaction_cal.jsonl").write_text(json.dumps({"task_group": "tg_v1_train", "split": "cal", "label": 0, "state_text": "x"}) + "\n")
    with pytest.raises(ValueError, match="Conflicting split"):
        ext.build_extension_dataset(tmp_path, heldout, counter, seed=42)
