#!/usr/bin/env python3
"""
build_manifest.py
Fetches SWE-bench Lite and Google DeepMind FRAMES tasks, merges them with InterCode & Terminal-Bench,
and generates:
1. tasks/heldout_100.json: Immutable held-out test suite (30 Bash, 10 Terminal-Bench, 30 FRAMES Research, 20 SWE-bench, 10 CTF)
2. tasks/train_pool.json: The remaining tasks for training, Jev data generation, and prompt optimization.
"""

import os
import sys
import json
import random
import hashlib
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
TASKS_DIR = BASE_DIR / "tasks"

HEADERS = {"User-Agent": "Bob-Eval-Harness/1.0"}

def fetch_json(url: str):
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Invalid URL scheme: {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        return json.loads(resp.read().decode())

def fetch_swe_bench_lite(limit=100):
    print("[-] Fetching SWE-bench Lite tasks...")
    url = f"https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Lite&config=default&split=test&offset=0&limit={limit}"
    data = fetch_json(url)
    tasks = []
    for item in data.get("rows", []):
        row = item["row"]
        tasks.append({
            "task_id": f"swe_{row['instance_id']}",
            "benchmark": "swe-bench-lite",
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "instruction": f"Fix the following issue in {row['repo']}:\n\n{row['problem_statement']}",
            "test_patch": row["test_patch"],
            "fail_to_pass": row.get("FAIL_TO_PASS", []),
            "pass_to_pass": row.get("PASS_TO_PASS", []),
            "eval_type": "pytest_patch"
        })
    print(f"    Loaded {len(tasks)} SWE-bench Lite tasks.")
    return tasks

def fetch_frames_benchmark(limit=150):
    print("[-] Fetching Google DeepMind FRAMES multi-hop research tasks...")
    url = f"https://datasets-server.huggingface.co/rows?dataset=google%2Fframes-benchmark&config=default&split=test&offset=0&limit={limit}"
    data = fetch_json(url)
    tasks = []
    for item in data.get("rows", []):
        row = item["row"]
        idx = row.get("Unnamed: 0", len(tasks))
        prompt = row["Prompt"].strip()
        ans = str(row["Answer"]).strip()
        tasks.append({
            "task_id": f"frames_{idx:03d}",
            "benchmark": "frames",
            "instruction": f"Answer the following question using web search and research tools:\n\n{prompt}",
            "gold_answer": ans,
            "reasoning_types": row.get("reasoning_types", ""),
            "wiki_links": row.get("wiki_links", ""),
            "eval_type": "exact_match_normalized"
        })
    print(f"    Loaded {len(tasks)} FRAMES research tasks.")
    return tasks

def main():
    # 1. Load previously sourced tasks (bash, ctf, terminal-bench)
    sourced_path = DATA_DIR / "all_sourced_tasks.json"
    if not sourced_path.exists():
        print("Run fetch_tasks.py first!")
        sys.exit(1)
        
    with open(sourced_path) as f:
        existing_tasks = json.load(f)
        
    swe_tasks = fetch_swe_bench_lite(100)
    frames_tasks = fetch_frames_benchmark(150)
    
    all_tasks = existing_tasks + swe_tasks + frames_tasks
    print(f"\n[+] Total tasks in master pool: {len(all_tasks)}")
    
    # 2. Group by benchmark
    by_bench = {}
    for t in all_tasks:
        b = t["benchmark"]
        by_bench.setdefault(b, []).append(t)
        
    # Print breakdown
    for b, ts in by_bench.items():
        print(f"    - {b}: {len(ts)} tasks")
        
    # 3. Deterministic selection for 100 held-out tasks (Seed 42)
    rng = random.Random(42)
    
    # Quotas for Non-Saturated 100-Task Holdout:
    # 25 SWE-bench Lite (Real repository bug fixes on Django)
    # 35 Google DeepMind FRAMES (Hardest multi-constraint & numerical quartile)
    # 20 InterCode-Bash (Advanced pipelines on /workspace/testbed fixtures)
    # 20 InterCode-CTF (Reverse engineering, binary analysis & decryption)
    quotas = {
        "swe-bench-lite": 25,
        "frames": 35,
        "intercode-bash": 20,
        "intercode-ctf": 20
    }
    
    heldout = []
    train_pool = []
    
    for bench, quota in quotas.items():
        candidates = by_bench.get(bench, [])
        if bench == "swe-bench-lite":
            django_candidates = [t for t in candidates if t.get("repo") == "django/django"]
            other_candidates = [t for t in candidates if t.get("repo") != "django/django"]
            rng.shuffle(django_candidates)
            bench_heldout = django_candidates[:quota]
            bench_train = django_candidates[quota:] + other_candidates
            heldout.extend(bench_heldout)
            train_pool.extend(bench_train)
        elif bench == "frames":
            sorted_frames = sorted(
                candidates,
                key=lambda t: (
                    "Numerical" in t.get("reasoning_types", ""),
                    "Tabular" in t.get("reasoning_types", ""),
                    "Temporal" in t.get("reasoning_types", ""),
                    len(t.get("reasoning_types", "").split("|"))
                ),
                reverse=True
            )
            heldout.extend(sorted_frames[:quota])
            train_pool.extend(sorted_frames[quota:])
        else:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            heldout.extend(shuffled[:quota])
            train_pool.extend(shuffled[quota:])
            
    assert len(heldout) == 100, f"Expected 100 held-out tasks, got {len(heldout)}"
    
    # Sort by task_id for canonical stability
    heldout.sort(key=lambda x: x["task_id"])
    train_pool.sort(key=lambda x: x["task_id"])
    
    # Compute manifest hash
    manifest_bytes = json.dumps([t["task_id"] for t in heldout], sort_keys=True).encode()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()[:16]
    
    heldout_manifest = {
        "manifest_version": "1.0",
        "manifest_hash": manifest_hash,
        "total_tasks": len(heldout),
        "quotas": quotas,
        "tasks": heldout
    }
    
    train_manifest = {
        "manifest_version": "1.0",
        "total_tasks": len(train_pool),
        "tasks": train_pool
    }
    
    heldout_file = TASKS_DIR / "heldout_100.json"
    train_file = TASKS_DIR / "train_pool.json"
    
    with open(heldout_file, "w") as f:
        json.dump(heldout_manifest, f, indent=2)
        
    with open(train_file, "w") as f:
        json.dump(train_manifest, f, indent=2)
        
    print(f"\n[+] Successfully created immutable Held-Out Test Set ({len(heldout)} tasks) -> {heldout_file}")
    print(f"[+] Manifest SHA256: {manifest_hash}")
    print(f"[+] Successfully created Training / Distillation Set ({len(train_pool)} tasks) -> {train_file}")

if __name__ == "__main__":
    main()
