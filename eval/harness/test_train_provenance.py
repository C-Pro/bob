"""Certified train-only provenance and four-way isolation tests."""
import json

import pytest

from eval.harness import build_jev_dataset as dataset
from eval.harness import build_jev_generic_and_multimodal as extension
from eval.harness import train_provenance as provenance
from eval.harness.test_build_jev_dataset import record


def _manifest(path, tasks):
    path.write_text(json.dumps({"tasks": tasks}))
    return path


def test_transitive_identity_groups_and_stable_four_way_partitions(tmp_path):
    tasks = [{"task_id": f"task_{i}", "instruction": f"instruction {i}"} for i in range(80)]
    tasks[0].update(instruction=" Same  Instruction ")
    tasks[1].update(instruction="same instruction", gold_answer=" Shared Gold ")
    tasks[2].update(gold_flag="shared gold", repo="Owner/Repo", base_commit="ABC123")
    tasks[3].update(repo="owner/repo", base_commit="abc123")
    tasks.append({"task_id": "ic_ctf_005", "instruction": "quarantined"})
    path = _manifest(tmp_path / "train.json", tasks)
    first = provenance.generate_certificate(path)
    entries = first["assignments"]
    assert "ic_ctf_005" not in entries
    assert len({entries[f"task_{i}"]["group_id"] for i in range(4)}) == 1
    assert len({entries[f"task_{i}"]["partition"] for i in range(4)}) == 1
    by_partition = {partition: {value["group_id"] for value in entries.values()
                                if value["partition"] == partition} for partition in provenance.PARTITIONS}
    assert all(by_partition.values())
    assert all(not by_partition[a] & by_partition[b] for a in provenance.PARTITIONS
               for b in provenance.PARTITIONS if a != b)
    _manifest(path, list(reversed(tasks)))
    assert provenance.generate_certificate(path)["assignments"] == entries


def test_certificate_only_build_never_reads_heldout_and_rejects_uncertified_rollouts(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path / "train.json", [
        {"task_id": f"task_{i}", "instruction": f"instruction {i}"} for i in range(32)
    ])
    certificate = provenance.generate_certificate(manifest)
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text(json.dumps(certificate))
    manifest.unlink()
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    for i in range(32):
        (rollouts / f"{i:02}.json").write_text(json.dumps(record(f"task_{i}", f"instruction {i}", count=4)))
    heldout = tmp_path / "unmounted" / "heldout.json"
    original_read_bytes = type(heldout).read_bytes

    def guarded_read_bytes(path):
        if "heldout" in str(path):
            raise AssertionError("Held-out data was read")
        return original_read_bytes(path)

    monkeypatch.setattr(type(heldout), "read_bytes", guarded_read_bytes)
    output = tmp_path / "output"
    stats = dataset.build_dataset(rollouts, heldout, output, dataset.TokenCounter(char_fallback=True),
                                  certificate_path=certificate_path)
    assert stats["rollout_count"] == 32
    assert stats["heldout_source_sha256"] is None
    assert stats["train_certificate_sha256"] == certificate["certificate_sha256"]
    seen = set()
    for partition in provenance.PARTITIONS:
        path = output / f"outcome_prefix_{partition}.jsonl"
        assert path.exists()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            entry = certificate["assignments"][row["source_task_id"]]
            assert row["partition"] == row["split"] == partition == entry["partition"]
            assert row["group_id"] == row["task_group"] == entry["group_id"]
            assert len(row["rollout_sha256"]) == 64
            assert row["renderer_version"] and row["label_policy"]
            seen.add(row["source_task_id"])
    assert len(seen) == 32

    bad = rollouts / "bad.json"
    for task_id, instruction, message in (
        ("heldout_task", "unseen", "Uncertified"),
        ("ic_ctf_005", "quarantined", "Uncertified"),
        ("task_0", "held-out instruction substituted", "instruction mismatch"),
    ):
        bad.write_text(json.dumps(record(task_id, instruction)))
        with pytest.raises(ValueError, match=message):
            dataset.build_dataset(rollouts, heldout, tmp_path / "rejected", dataset.TokenCounter(char_fallback=True),
                                  certificate_path=certificate_path)
    bad.unlink()
    assert not (tmp_path / "rejected").exists()

    altered = json.loads(certificate_path.read_text())
    altered["assignments"]["heldout_task"] = altered["assignments"]["task_0"]
    certificate_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="digest mismatch"):
        provenance.load_certificate(certificate_path=certificate_path)


def test_certified_extension_uses_four_partitions_and_validates_v1_rows(tmp_path):
    manifest = _manifest(tmp_path / "train.json", [{"task_id": "task_0", "instruction": "instruction 0"}])
    certificate = provenance.generate_certificate(manifest)
    partition = certificate["assignments"]["task_0"]["partition"]
    output = tmp_path / "output"
    output.mkdir()
    v1 = {"schema_version": dataset.SCHEMA_VERSION, "task_id": "task_0", "source_task_id": "task_0",
          "task_group": certificate["assignments"]["task_0"]["group_id"], "split": partition,
          "source_sha256": "a" * 64, "label": "none", "event_id": "v1_loop"}
    (output / f"loop_guard_{partition}.jsonl").write_text(json.dumps(v1) + "\n")
    counter = extension.TokenCounter(char_fallback=True)
    stats = extension.build_extension_dataset(output, tmp_path / "unmounted" / "heldout.json", counter,
                                               train_manifest_path=manifest)
    assert stats["train_certificate_sha256"] == certificate["certificate_sha256"]
    for split in provenance.PARTITIONS:
        rows = [json.loads(line) for line in (output / f"generic_decisions_{split}.jsonl").read_text().splitlines()]
        assert rows
        assert all(row["partition"] == split and row["group_id"] and row["label_policy"]
                   and row["source_task_id"] is None and row["rollout_sha256"] is None for row in rows)
    loop_rows = [json.loads(line) for line in (output / f"loop_guard_{partition}.jsonl").read_text().splitlines()]
    original = next(row for row in loop_rows if row["event_id"] == "v1_loop")
    assert original["source_task_id"] == "task_0"
    assert original["group_id"] == v1["task_group"]
    assert original["rollout_sha256"] == "a" * 64

    bad = dict(v1, task_id="heldout_task", source_task_id="heldout_task")
    with (output / f"compaction_{partition}.jsonl").open("w") as handle:
        handle.write(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="Uncertified"):
        extension.build_extension_dataset(output, None, counter, train_manifest_path=manifest)
