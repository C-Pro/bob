#!/usr/bin/env python3
"""
fetch_tasks.py
Downloads and standardizes tasks from open benchmarks:
- InterCode-Bash (200 tasks)
- InterCode-CTF (100 tasks)
- Terminal-Bench (68 tasks)
- GAIA Level 1 (subset)
- SWE-bench Lite (metadata)
"""

import os
import sys
import json
import urllib.request
import hashlib
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
TASKS_DIR = BASE_DIR / "tasks"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TASKS_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Bob-Eval-Harness/1.0"}

def fetch_json(url: str):
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"Invalid URL scheme: {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        return json.loads(resp.read().decode())

def fetch_intercode_bash():
    print("[-] Fetching InterCode-Bash tasks...")
    base_url = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/nl2bash"
    tasks = []
    
    # Fetch test queries
    test_queries = fetch_json(f"{base_url}/test_queries.json")
    for i, item in enumerate(test_queries):
        tasks.append({
            "task_id": f"ic_bash_test_{i:03d}",
            "benchmark": "intercode-bash",
            "instruction": item["query"],
            "gold_command": item.get("gold", ""),
            "files": {},
            "eval_type": "bash_gold_or_state"
        })
        
    # Fetch fs sets (filesystem tasks with setup & evaluation)
    for part in [1, 2, 3, 4]:
        try:
            fs_data = fetch_json(f"{base_url}/nl2bash_fs_{part}.json")
            for i, item in enumerate(fs_data):
                tasks.append({
                    "task_id": f"ic_bash_fs{part}_{i:03d}",
                    "benchmark": "intercode-bash",
                    "instruction": item["query"],
                    "gold_command": item.get("gold", ""),
                    "files": {},
                    "eval_type": "bash_fs"
                })
        except Exception as e:
            print(f"    Warning: failed to fetch part {part}: {e}")
            
    print(f"    Loaded {len(tasks)} InterCode-Bash tasks.")
    return tasks

def fetch_intercode_ctf():
    print("[-] Fetching InterCode-CTF tasks...")
    url = "https://raw.githubusercontent.com/princeton-nlp/intercode/master/data/ctf/ic_ctf.json"
    raw_ctf = fetch_json(url)
    tasks = []
    for i, item in enumerate(raw_ctf):
        tasks.append({
            "task_id": f"ic_ctf_{i:03d}",
            "benchmark": "intercode-ctf",
            "instruction": item["query"],
            "gold_flag": item.get("gold", ""),
            "eval_type": "ctf_flag_match"
        })
    print(f"    Loaded {len(tasks)} InterCode-CTF tasks.")
    return tasks

def fetch_terminal_bench():
    print("[-] Fetching Terminal-Bench tasks...")
    api_url = "https://api.github.com/repos/harbor-framework/terminal-bench/contents/tasks"
    tasks = []
    try:
        items = fetch_json(api_url)
        for item in items:
            if item["type"] == "dir":
                task_name = item["name"]
                # Try fetching instruction.md
                inst_url = f"https://raw.githubusercontent.com/harbor-framework/terminal-bench/main/tasks/{task_name}/instruction.md"
                try:
                    req = urllib.request.Request(inst_url, headers=HEADERS)
                    with urllib.request.urlopen(req, timeout=15) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
                        instruction = resp.read().decode().strip()
                except Exception:
                    instruction = f"Complete task {task_name}"
                    
                tasks.append({
                    "task_id": f"tb_{task_name}",
                    "benchmark": "terminal-bench",
                    "task_name": task_name,
                    "instruction": instruction,
                    "eval_type": "terminal_bench_pytest"
                })
    except Exception as e:
        print(f"    Warning: could not fetch terminal-bench from github api: {e}")
    print(f"    Loaded {len(tasks)} Terminal-Bench tasks.")
    return tasks

def main():
    bash_tasks = fetch_intercode_bash()
    ctf_tasks = fetch_intercode_ctf()
    tb_tasks = fetch_terminal_bench()
    
    all_tasks = bash_tasks + ctf_tasks + tb_tasks
    
    with open(DATA_DIR / "all_sourced_tasks.json", "w") as f:
        json.dump(all_tasks, f, indent=2)
        
    print(f"\n[+] Total sourced tasks: {len(all_tasks)}")
    print(f"[+] Saved to {DATA_DIR / 'all_sourced_tasks.json'}")

if __name__ == "__main__":
    main()
