"""Train-manifest provenance and stable, transitive four-way task partitioning."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

PARTITIONS = ("train_fit", "train_dev", "train_calibration", "train_acceptance")
QUARANTINED_TASK_IDS = frozenset({"ic_ctf_005", "ic_ctf_068"})
CERTIFICATE_VERSION = "train-allowlist-v1"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def identity_keys(task):
    task_id = str(task.get("task_id", "")).strip()
    instruction = _normalize(task.get("instruction", ""))
    if not task_id or not instruction:
        raise ValueError("Train tasks require a task_id and nonempty instruction")
    keys = {("task_id", task_id), ("instruction_sha256", _hash(instruction))}
    metadata = task.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}
    values = {**(metadata if isinstance(metadata, dict) else {}), **task}
    for field in ("gold_answer", "gold_flag", "gold_command"):
        value = values.get(field)
        if value is not None and not isinstance(value, (dict, list, bool)) and _normalize(value):
            keys.add(("gold_target_sha256", _hash(_normalize(value))))
    repo, commit = values.get("repo"), values.get("base_commit")
    if isinstance(repo, str) and isinstance(commit, str) and repo.strip() and commit.strip():
        keys.add(("repo_base_commit", _hash(_canonical([_normalize(repo), commit.strip().lower()]))))
    return keys


def generate_certificate(manifest_path, quarantine_ids=QUARANTINED_TASK_IDS):
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    tasks = manifest["tasks"] if isinstance(manifest, dict) else manifest
    if not isinstance(tasks, list):
        raise ValueError("Train manifest must contain a tasks list")
    quarantine = set(quarantine_ids) | set(QUARANTINED_TASK_IDS)
    tasks = [task for task in tasks if str(task.get("task_id", "")).strip() not in quarantine]
    instruction_hashes = {}
    for task in tasks:
        task_id = str(task["task_id"]).strip()
        digest = dict(identity_keys(task))["instruction_sha256"]
        if task_id in instruction_hashes and instruction_hashes[task_id] != digest:
            raise ValueError(f"Conflicting train manifest instruction for {task_id}")
        instruction_hashes[task_id] = digest
    parent = list(range(len(tasks)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, task in enumerate(tasks):
        for key in sorted(identity_keys(task)):
            if key in seen:
                parent[find(index)] = find(seen[key])
            seen[key] = index

    components = defaultdict(list)
    for index, task in enumerate(tasks):
        components[find(index)].append(task)
    assignments = {}
    for members in components.values():
        task_ids = sorted({str(task["task_id"]).strip() for task in members})
        group_id = "tg_" + _hash(_canonical(task_ids))[:24]
        bucket = int(_hash(group_id)[:16], 16) % 100
        partition = PARTITIONS[0 if bucket < 70 else 1 if bucket < 80 else 2 if bucket < 90 else 3]
        for task_id in task_ids:
            assignments[task_id] = {"group_id": group_id, "partition": partition,
                                    "instruction_sha256": instruction_hashes[task_id]}
    certificate = {
        "certificate_version": CERTIFICATE_VERSION,
        "source": "train_pool",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "quarantined_task_ids": sorted(quarantine),
        "assignments": dict(sorted(assignments.items())),
    }
    certificate["certificate_sha256"] = _hash(_canonical(certificate))
    return certificate


def load_certificate(*, certificate_path=None, train_manifest_path=None):
    if certificate_path is None and train_manifest_path is None:
        raise ValueError("A train allowlist certificate or train manifest is required")
    certificate = (json.loads(Path(certificate_path).read_text()) if certificate_path is not None
                   else generate_certificate(train_manifest_path))
    expected = certificate.get("certificate_sha256")
    payload = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    if certificate.get("certificate_version") != CERTIFICATE_VERSION or certificate.get("source") != "train_pool":
        raise ValueError("Invalid train allowlist certificate")
    if expected != _hash(_canonical(payload)):
        raise ValueError("Train allowlist certificate digest mismatch")
    assignments = certificate.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Empty train allowlist certificate")
    if QUARANTINED_TASK_IDS & assignments.keys():
        raise ValueError("Quarantined task in train allowlist certificate")
    for task_id, entry in assignments.items():
        if (not task_id or not isinstance(entry, dict) or not entry.get("group_id")
                or entry.get("partition") not in PARTITIONS or not isinstance(entry.get("instruction_sha256"), str)
                or len(entry["instruction_sha256"]) != 64):
            raise ValueError("Invalid train allowlist assignment")
    if train_manifest_path is not None and certificate_path is not None:
        if certificate != generate_certificate(train_manifest_path):
            raise ValueError("Train allowlist certificate does not match train manifest")
    return certificate


def certify_task(task_id, certificate, record=None):
    entry = certificate["assignments"].get(str(task_id).strip())
    if entry is None:
        raise ValueError(f"Uncertified train task_id: {task_id}")
    if record is not None and dict(identity_keys(record))["instruction_sha256"] != entry["instruction_sha256"]:
        raise ValueError(f"Train instruction mismatch for {task_id}")
    return entry
