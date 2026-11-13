#!/usr/bin/env python3
"""
runner.py
Batch execution runner for Bob using local Qwen3.6-35B-A3B-MTP-GGUF (lemond).
Provisions clean, isolated bwrap workspaces and pristine per-task SQLite databases.
Guarantees zero database pollution, authentic task fixtures, boundary-aware grading,
rigorous state and output verification, and detailed trajectory export for Phase 4 Jev decision model training.
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import subprocess
import signal
import re
import string
import unicodedata
import hashlib
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parent.parent
BOB_BIN = REPO_ROOT / "bin" / "bob"
TEMPLATE_CONFIG = BASE_DIR / "config_template.yaml"
RESULTS_DIR = BASE_DIR.parent / "results"
CTF_ASSETS_DIR = BASE_DIR / "ctf_assets"
BASH_FIXTURES_DIR = BASE_DIR / "bash_fixtures"
REPOS_DIR = BASE_DIR / "repos"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ASCII", "ignore").decode("utf-8")
    translator = str.maketrans("", "", string.punctuation)
    return " ".join(ascii_text.lower().translate(translator).split())

def hash_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def backup_sqlite_db(src_path: Path, dst_path: Path) -> bool:
    if not src_path.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        src_conn = sqlite3.connect(src_path)
        src_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        dst_conn = sqlite3.connect(dst_path)
        with dst_conn:
            src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        return True
    except Exception as e:
        print(f"    [!] SQLite backup failed: {e}")
        return False

def snapshot_dir_state(directory: Path) -> dict:
    files = {}
    dirs = set()
    for root, dirnames, filenames in os.walk(directory):
        for d in dirnames:
            full_d = Path(root) / d
            dirs.add(str(full_d.relative_to(directory)))
        for f in filenames:
            full_p = Path(root) / f
            rel_p = str(full_p.relative_to(directory))
            try:
                files[rel_p] = hash_file(full_p)
            except Exception:
                pass
    return {"files": files, "dirs": list(dirs)}

def grade_ctf(task: dict, workspace: Path, stdout_output: str) -> dict:
    gold_flag = task.get("gold_flag", "")
    passed = gold_flag.lower() in stdout_output.lower()
    return {
        "benchmark_reward": 1.0 if passed else 0.0,
        "eval_details": f"Flag {'found' if passed else 'missing'} in agent stdout",
        "grading_status": "graded"
    }

def grade_frames(task: dict, workspace: Path, stdout_output: str) -> dict:
    gold = normalize_text(task.get("gold_answer", ""))
    norm_stdout = normalize_text(stdout_output)
    
    core_gold = gold
    for prefix in ["this was ", "the answer is ", "it was ", "answer: ", "the "]:
        if core_gold.startswith(prefix):
            core_gold = core_gold[len(prefix):].strip()
            
    gold_clean_num = core_gold.replace(",", "").strip()
    if gold_clean_num.isdigit():
        tokens = [t.strip(".,;:()\"'\n") for t in norm_stdout.replace(",", "").split()]
        passed = gold_clean_num in tokens
        return {
            "benchmark_reward": 1.0 if passed else 0.0,
            "eval_details": f"Numeric token '{gold_clean_num}' {'found' if passed else 'missing'}",
            "grading_status": "graded"
        }
        
    passed = (gold in norm_stdout) or (len(core_gold) > 1 and core_gold in norm_stdout)
    return {
        "benchmark_reward": 1.0 if passed else 0.0,
        "eval_details": f"Target '{core_gold}' {'found' if passed else 'missing'}",
        "grading_status": "graded"
    }

def evaluate_swe_patch_docker(task_id: str, patch_content: str, timeout_sec: int = 600) -> tuple:
    inst_id = task_id.replace("swe_", "")
    pred = [{
        "instance_id": inst_id,
        "model_patch": patch_content,
        "model_name_or_path": "bob"
    }]
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pred, f)
        pred_path = Path(f.name)
        
    run_id = f"eval_{inst_id}_{int(time.time())}"
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "-d", "SWE-bench/SWE-bench_Lite",
        "-p", str(pred_path),
        "-i", inst_id,
        "-id", run_id,
        "--max_workers", "1",
        "--timeout", str(timeout_sec)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec + 300)
        report_dir = Path("logs/run_evaluation") / run_id / "bob" / inst_id
        report_json = report_dir / "report.json"
        if report_json.exists():
            with open(report_json) as rf:
                rep = json.load(rf)
                inst_rep = rep.get(inst_id, {})
                resolved = inst_rep.get("resolved", False)
                tests_status = inst_rep.get("tests_status", {})
                f2p_fail = tests_status.get("FAIL_TO_PASS", {}).get("failure", [])
                f2p_succ = tests_status.get("FAIL_TO_PASS", {}).get("success", [])
                p2p_fail = tests_status.get("PASS_TO_PASS", {}).get("failure", [])
                p2p_succ = tests_status.get("PASS_TO_PASS", {}).get("success", [])
                if resolved:
                    return True, f"Resolved: All FAIL_TO_PASS ({len(f2p_succ)}) and PASS_TO_PASS ({len(p2p_succ)}) tests passed in Docker testbed"
                else:
                    return False, f"Unresolved in Docker: {len(f2p_fail)} fail_to_pass failure(s) (passed {len(f2p_succ)}), {len(p2p_fail)} pass_to_pass regression(s)"
        return False, f"Docker eval failed to produce report.json (Exit code {proc.returncode})"
    except Exception as e:
        return False, f"Docker eval error: {e}"
    finally:
        pred_path.unlink(missing_ok=True)

def grade_swe(task: dict, workspace: Path, base_sha: str = "HEAD", run_docker: bool = True) -> dict:
    patch = ""
    modified_files = []
    try:
        subprocess.run(["git", "add", "-A"], cwd=workspace, check=False, capture_output=True)
        patch = subprocess.check_output(
            ["git", "diff", "--binary", "--cached", base_sha],
            cwd=workspace, text=True, errors="ignore"
        )
        stat_out = subprocess.check_output(
            ["git", "diff", "--stat", "--cached", base_sha],
            cwd=workspace, text=True, errors="ignore"
        )
        modified_files = [line.split("|")[0].strip() for line in stat_out.splitlines() if "|" in line]
    except Exception as e:
        patch = ""
        
    has_patch = len(patch.strip()) > 0 and "diff --git" in patch
    if not has_patch:
        return {
            "patch_generated": False,
            "grading_status": "graded",
            "resolved": False,
            "benchmark_reward": 0.0,
            "patch": "",
            "patch_lines": 0,
            "modified_files": [],
            "eval_details": "No patch generated (iteration budget exhausted)"
        }
        
    resolved = False
    details = f"Patch generated ({len(patch.splitlines())} lines, {len(modified_files)} files)"
    if run_docker:
        print(f"    [*] Running Docker validation via swebench for {task['task_id']}...", flush=True)
        resolved, docker_details = evaluate_swe_patch_docker(task["task_id"], patch)
        print(f"    [*] Docker validation result: {docker_details}", flush=True)
        details = f"{details} -> {docker_details}"
        
    return {
        "patch_generated": True,
        "grading_status": "graded" if run_docker else "pending",
        "resolved": resolved,
        "benchmark_reward": 1.0 if resolved else 0.0,
        "patch": patch,
        "patch_lines": len(patch.splitlines()),
        "modified_files": modified_files,
        "eval_details": details
    }

def grade_bash(task: dict, workspace: Path, stdout_output: str, oracle_info: dict = None, initial_snapshot: dict = None) -> dict:
    if not oracle_info:
        return {
            "benchmark_reward": 0.0,
            "grading_status": "oracle_error",
            "eval_details": "Oracle information unavailable"
        }
        
    if oracle_info.get("oracle_exit_code") != 0:
        err_msg = oracle_info.get("oracle_stderr", "").strip()[:100]
        return {
            "benchmark_reward": 0.0,
            "grading_status": "oracle_error",
            "eval_details": f"Oracle command failed (exit code {oracle_info.get('oracle_exit_code')}): {err_msg}"
        }
        
    # 1. State change verification
    if oracle_info.get("state_changed"):
        if initial_snapshot is None:
            initial_snapshot = {"files": {}, "dirs": []}
            
        workspace_after = snapshot_dir_state(workspace)
        files_before = initial_snapshot.get("files", {})
        dirs_before = set(initial_snapshot.get("dirs", []))
        
        ws_removed_files = set(files_before.keys()) - set(workspace_after["files"].keys())
        expected_removed_files = set(oracle_info.get("removed_files", []))
        if ws_removed_files != expected_removed_files:
            return {
                "benchmark_reward": 0.0,
                "grading_status": "graded",
                "eval_details": f"File removal mismatch (expected: {list(expected_removed_files)[:3]}, actual: {list(ws_removed_files)[:3]})"
            }
            
        ws_removed_dirs = dirs_before - set(workspace_after["dirs"])
        expected_removed_dirs = set(oracle_info.get("removed_dirs", []))
        if ws_removed_dirs != expected_removed_dirs:
            return {
                "benchmark_reward": 0.0,
                "grading_status": "graded",
                "eval_details": f"Directory removal mismatch (expected: {list(expected_removed_dirs)[:3]}, actual: {list(ws_removed_dirs)[:3]})"
            }
            
        ws_added_files = set(workspace_after["files"].keys()) - set(files_before.keys())
        expected_added_files = set(oracle_info.get("added_files", []))
        if ws_added_files != expected_added_files:
            return {
                "benchmark_reward": 0.0,
                "grading_status": "graded",
                "eval_details": f"File addition mismatch (expected: {list(expected_added_files)[:3]}, actual: {list(ws_added_files)[:3]})"
            }
            
        ws_added_dirs = set(workspace_after["dirs"]) - dirs_before
        expected_added_dirs = set(oracle_info.get("added_dirs", []))
        if ws_added_dirs != expected_added_dirs:
            return {
                "benchmark_reward": 0.0,
                "grading_status": "graded",
                "eval_details": f"Directory addition mismatch (expected: {list(expected_added_dirs)[:3]}, actual: {list(ws_added_dirs)[:3]})"
            }
            
        for f in expected_added_files:
            expected_hash = oracle_info.get("file_hashes_after", {}).get(f)
            actual_hash = workspace_after["files"].get(f)
            if expected_hash and actual_hash != expected_hash:
                return {
                    "benchmark_reward": 0.0,
                    "grading_status": "graded",
                    "eval_details": f"Created file '{f}' content hash mismatch"
                }
                
        common = set(files_before.keys()) & set(workspace_after["files"].keys())
        ws_modified = {f for f in common if files_before[f] != workspace_after["files"][f]}
        expected_modified = set(oracle_info.get("modified_files", []))
        if ws_modified != expected_modified:
            return {
                "benchmark_reward": 0.0,
                "grading_status": "graded",
                "eval_details": f"File modification mismatch (expected: {list(expected_modified)[:3]}, actual: {list(ws_modified)[:3]})"
            }
            
        for f in expected_modified:
            expected_hash = oracle_info.get("file_hashes_after", {}).get(f)
            actual_hash = workspace_after["files"].get(f)
            if expected_hash and actual_hash != expected_hash:
                return {
                    "benchmark_reward": 0.0,
                    "grading_status": "graded",
                    "eval_details": f"Modified file '{f}' content hash mismatch"
                }
                
        return {
            "benchmark_reward": 1.0,
            "grading_status": "graded",
            "eval_details": "Complete filesystem state (files, directories, hashes) matched oracle"
        }
        
    # 2. Output verification (information query)
    oracle_out = oracle_info.get("oracle_stdout", "").strip()
    if not oracle_out:
        return {
            "benchmark_reward": 0.0,
            "grading_status": "oracle_error",
            "eval_details": "Oracle produced empty stdout for a query task"
        }
        
    def extract_path_tokens(text: str) -> set:
        tokens = re.findall(r'[\w\./\-]+\.\w+', text)
        cleaned = set()
        for t in tokens:
            c = re.sub(r'^(?:/workspace)?/(?:testbed|system|backup)/|^\./', '', t).strip('/')
            if len(c) > 3 and not c.startswith('http'):
                cleaned.add(c)
        return cleaned

    o_paths = extract_path_tokens(oracle_out)
    if o_paths:
        # For path-query tasks, enforce exact path set equality
        b_paths = extract_path_tokens(stdout_output)
        if o_paths == b_paths:
            return {
                "benchmark_reward": 1.0,
                "grading_status": "graded",
                "eval_details": f"Exact path target set matched ({len(o_paths)} targets)"
            }
        extra = b_paths - o_paths
        missing = o_paths - b_paths
        return {
            "benchmark_reward": 0.0,
            "grading_status": "graded",
            "eval_details": f"Path targets mismatch (missing: {list(missing)[:3]}, extra: {list(extra)[:3]})"
        }

    norm_oracle = normalize_text(oracle_out)
    norm_stdout = normalize_text(stdout_output)
    
    clean_num = norm_oracle.replace(",", "").strip()
    if clean_num.isdigit() or re.match(r'^\d+\.\d+$', clean_num):
        tokens = [t.strip(".,;:()\"'\n") for t in norm_stdout.replace(",", "").split()]
        passed = clean_num in tokens
        return {
            "benchmark_reward": 1.0 if passed else 0.0,
            "grading_status": "graded",
            "eval_details": f"Oracle number '{clean_num}' {'found' if passed else 'missing'} in stdout"
        }
        
    if norm_oracle in norm_stdout:
        return {
            "benchmark_reward": 1.0,
            "grading_status": "graded",
            "eval_details": "Oracle answer matched in stdout"
        }
        
    oracle_lines = [normalize_text(line) for line in oracle_out.splitlines() if len(line.strip()) > 2]
    if oracle_lines:
        all_matched = all(line in norm_stdout for line in oracle_lines)
        if all_matched:
            return {
                "benchmark_reward": 1.0,
                "grading_status": "graded",
                "eval_details": f"All {len(oracle_lines)} oracle output lines matched in stdout"
            }
            
    return {
        "benchmark_reward": 0.0,
        "grading_status": "graded",
        "eval_details": f"Oracle answer '{norm_oracle[:60]}...' missing in stdout"
    }

def grade_task(task: dict, workspace: Path, stdout_output: str, base_sha: str = "HEAD", oracle_info: dict = None, initial_snapshot: dict = None) -> dict:
    bmark = task.get("benchmark")
    if bmark == "intercode-ctf":
        return grade_ctf(task, workspace, stdout_output)
    elif bmark == "frames":
        return grade_frames(task, workspace, stdout_output)
    elif bmark == "swe-bench-lite":
        return grade_swe(task, workspace, base_sha=base_sha)
    elif bmark == "intercode-bash":
        return grade_bash(task, workspace, stdout_output, oracle_info=oracle_info, initial_snapshot=initial_snapshot)
    else:
        return {"benchmark_reward": 0.0, "grading_status": "unknown_benchmark", "eval_details": "Unknown benchmark"}

def setup_bash_fixtures(target_dir: Path, fs_id: str) -> tuple:
    if fs_id == "test":
        setup_file = BASH_FIXTURES_DIR / "setup_fs.sh"
    else:
        setup_file = BASH_FIXTURES_DIR / f"setup_nl2b_{fs_id}.sh"
        
    if not setup_file.exists():
        return False, f"Setup fixture not found: {setup_file}"
        
    txt = setup_file.read_text()
    
    if fs_id in ("fs_1", "test"):
        testbed_dir = target_dir / "testbed"
        testbed_dir.mkdir(parents=True, exist_ok=True)
        adapted = re.sub(r'/testbed\b', str(testbed_dir), txt)
        adapted = re.sub(r'/index\.html\b', str(target_dir / "index.html"), adapted)
    elif fs_id == "fs_2":
        sys_dir = target_dir / "system"
        sys_dir.mkdir(parents=True, exist_ok=True)
        adapted = re.sub(r'/system\b', str(sys_dir), txt)
    elif fs_id == "fs_3":
        backup_dir = target_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        adapted = re.sub(r'/workspace\b', str(target_dir), txt)
        adapted = re.sub(r'/backup\b', str(backup_dir), adapted)
    else:
        adapted = txt
        
    run_sh = target_dir / "setup_tmp.sh"
    run_sh.write_text(adapted)
    p = subprocess.run(["bash", str(run_sh)], cwd=target_dir, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    run_sh.unlink(missing_ok=True)
    if p.returncode != 0:
        return False, f"Fixture setup failed: {p.stderr}"
    return True, ""

def run_bash_oracle(task: dict, run_root: Path, fs_id: str) -> dict:
    oracle_dir = run_root / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    ok, err = setup_bash_fixtures(oracle_dir, fs_id)
    if not ok:
        return {"oracle_exit_code": -1, "oracle_stderr": err, "state_changed": False}
        
    gold_cmd = task.get("gold_command", "")
    if not gold_cmd:
        return {"oracle_exit_code": -1, "oracle_stderr": "No gold command specified", "state_changed": False}
        
    adapted_cmd = gold_cmd
    if fs_id in ("fs_1", "test"):
        adapted_cmd = re.sub(r'/testbed\b', str(oracle_dir / "testbed"), adapted_cmd)
    elif fs_id == "fs_2":
        adapted_cmd = re.sub(r'/system\b', str(oracle_dir / "system"), adapted_cmd)
    elif fs_id == "fs_3":
        adapted_cmd = re.sub(r'/workspace\b', str(oracle_dir), adapted_cmd)
        adapted_cmd = re.sub(r'/backup\b', str(oracle_dir / "backup"), adapted_cmd)
        
    before = snapshot_dir_state(oracle_dir)
    files_before = before["files"]
    dirs_before = set(before["dirs"])
    
    try:
        p = subprocess.run(
            ["bash", "-c", f"set -o pipefail\n{adapted_cmd}"],
            cwd=oracle_dir,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=30
        )
        returncode = p.returncode
        stdout = p.stdout
        stderr = p.stderr
    except subprocess.TimeoutExpired:
        return {
            "oracle_exit_code": -1,
            "oracle_stderr": "Oracle execution timed out after 30s",
            "state_changed": False
        }
        
    clean_stdout = stdout.replace(str(oracle_dir), "/workspace")
    clean_stderr = stderr.replace(str(oracle_dir), "/workspace")
    
    after = snapshot_dir_state(oracle_dir)
    files_after = after["files"]
    dirs_after = set(after["dirs"])
    
    removed_files = list(set(files_before.keys()) - set(files_after.keys()))
    removed_dirs = list(dirs_before - dirs_after)
    added_files = list(set(files_after.keys()) - set(files_before.keys()))
    added_dirs = list(dirs_after - dirs_before)
    common_files = set(files_before.keys()) & set(files_after.keys())
    modified_files = [f for f in common_files if files_before[f] != files_after[f]]
    
    state_changed = len(removed_files) > 0 or len(removed_dirs) > 0 or len(added_files) > 0 or len(added_dirs) > 0 or len(modified_files) > 0
    
    return {
        "oracle_stdout": clean_stdout,
        "oracle_stderr": clean_stderr,
        "oracle_exit_code": returncode,
        "state_changed": state_changed,
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "added_files": added_files,
        "added_dirs": added_dirs,
        "modified_files": modified_files,
        "file_hashes_after": files_after
    }

def extract_trajectory_from_db(db_path: Path) -> dict:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute("SELECT * FROM fsm_runs ORDER BY created_at DESC LIMIT 1")
    run_row = cur.fetchone()
    if not run_row:
        conn.close()
        return {}
        
    run_id = run_row["id"]
    cur.execute("SELECT * FROM fsm_steps WHERE run_id = ? ORDER BY iteration, step_index", (run_id,))
    steps = [dict(r) for r in cur.fetchall()]
    for s in steps:
        for col in ("args_json", "result_json"):
            if s.get(col) and isinstance(s[col], str):
                try:
                    s[col] = json.loads(s[col])
                except Exception:
                    pass
    
    cur.execute("SELECT * FROM fsm_compactions WHERE run_id = ? ORDER BY version", (run_id,))
    compactions = [dict(r) for r in cur.fetchall()]
    for c in compactions:
        if c.get("state_json") and isinstance(c["state_json"], str):
            try:
                c["state_json"] = json.loads(c["state_json"])
            except Exception:
                pass
    
    cur.execute("SELECT * FROM agent_trajectories WHERE run_id = ?", (run_id,))
    traj_row = cur.fetchone()
    traj_dict = {}
    if traj_row:
        traj_dict = dict(traj_row)
        if "payload" in traj_dict and isinstance(traj_dict["payload"], (bytes, bytearray)):
            try:
                traj_dict["payload"] = json.loads(traj_dict["payload"].decode("utf-8"))
            except Exception:
                traj_dict["payload"] = traj_dict["payload"].decode("utf-8", errors="replace")
    
    events = []
    try:
        cur.execute(
            "SELECT id, trajectory_id, run_id, generation, iteration, phase, step_id, attempt, event_type, payload, created_at FROM agent_trajectory_events WHERE run_id = ? ORDER BY created_at",
            (run_id,)
        )
        for r in cur.fetchall():
            ed = dict(r)
            if "payload" in ed and isinstance(ed["payload"], (bytes, bytearray)):
                try:
                    ed["payload"] = json.loads(ed["payload"].decode("utf-8"))
                except Exception:
                    ed["payload"] = ed["payload"].decode("utf-8", errors="replace")
            events.append(ed)
    except Exception as e:
        print(f"    Warning reading events: {e}")
        
    conn.close()
    return {
        "run": dict(run_row),
        "trajectory": traj_dict,
        "steps": steps,
        "compactions": compactions,
        "events": events
    }

def run_single_task(task: dict, max_iterations: int = 500, timeout_sec: int = 900, batch_dir: Path = None) -> dict:
    if batch_dir is None:
        batch_dir = ROLLOUTS_DIR
    batch_dir.mkdir(parents=True, exist_ok=True)

    task_id = task["task_id"]
    bmark = task.get("benchmark")
    print(f"\n========================================================")
    print(f"[*] Running task: {task_id} ({bmark})")
    print(f"    Instruction: {task['instruction'][:100]}...")
    
    run_id = f"eval_{task_id}_{int(time.time())}"
    run_root = Path(f"/tmp/bob-eval-runs/{run_id}")
    workspace = run_root / "workspace"
    host_dir = run_root / "host"
    data_dir = host_dir / "data"
    
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_file = host_dir / "config.yaml"
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if not tavily_key:
        for p in [TEMPLATE_CONFIG, Path.home() / ".config" / "bob" / "config.yaml", REPO_ROOT / "scratch" / "config_lemond.yaml"]:
            if p.exists():
                try:
                    import yaml
                    with open(p) as f:
                        cfg_data = yaml.safe_load(f)
                        k = cfg_data.get("web", {}).get("tavily_api_key", "")
                        if k and not k.startswith("${"):
                            tavily_key = k
                            break
                except Exception:
                    pass
    cfg_content = f"""llm:
  api_key: "dummy-key"
  base_url: "http://127.0.0.1:15432/v1"
  model: "Qwen3.6-35B-A3B-MTP-GGUF"
  first_byte_timeout: "5m"
  request_timeout: "10m"

web:
  tavily_api_key: "{tavily_key}"
  tavily_base_url: "https://api.tavily.com"

data_dir: "{data_dir}"

cli:
  max_iterations: {max_iterations}

sandbox:
  default_driver: "bwrap"
  allowed_network_modes: ["none", "restricted", "full"]
  default_exec_timeout: "2m"
  max_exec_timeout: "1h"
  max_lifetime: "2h"

fsm:
  compaction_interval_iterations: 10
"""
    cfg_file.write_text(cfg_content)
    cfg_file.chmod(0o600)
    
    instruction = task["instruction"]
    base_sha = "HEAD"
    oracle_info = None
    initial_snapshot = None
    
    try:
        # A. CTF Provisioning
        if bmark == "intercode-ctf":
            idx = int(task_id.split("_")[-1])
            src_assets = CTF_ASSETS_DIR / str(idx)
            if src_assets.exists():
                for item in src_assets.iterdir():
                    if item.name == ".placeholder":
                        continue
                    dst = workspace / item.name
                    if item.is_file():
                        shutil.copy2(item, dst)
                        if dst.suffix in (".sh", ".py") or dst.name == "static":
                            dst.chmod(dst.stat().st_mode | 0o755)
                    elif item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
            instruction = f"{task['instruction']}\n\nExamine the challenge files in this directory and solve the challenge. Print the final flag (e.g. picoCTF{{...}}) clearly in your response."
            
        # B. SWE Provisioning (Worktree -> Standalone local repo)
        elif bmark == "swe-bench-lite":
            repo_dir = REPOS_DIR / "django.git"
            if not repo_dir.exists():
                scratch_repo = REPO_ROOT / "scratch" / "eval_harness" / "repos" / "django.git"
                if scratch_repo.exists():
                    repo_dir = scratch_repo
                else:
                    print(f"[-] Cloned bare mirror of django not found at {repo_dir}. Cloning...")
                    REPOS_DIR.mkdir(parents=True, exist_ok=True)
                    subprocess.run(
                        ["git", "clone", "--bare", "https://github.com/django/django.git", str(repo_dir)],
                        check=True
                    )
                
            subprocess.run(
                ["git", f"--git-dir={repo_dir}", "worktree", "add", str(workspace), task["base_commit"]],
                check=True, capture_output=True
            )
            (workspace / ".git").unlink(missing_ok=True)
            subprocess.run(["git", "-C", str(workspace), "init"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.name", "eval"], check=True)
            subprocess.run(["git", "-C", str(workspace), "config", "user.email", "eval@example.com"], check=True)
            subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(workspace), "commit", "-m", "base_commit", "--no-gpg-sign"], check=True, capture_output=True)
            base_sha = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
            
            instruction = f"In this repository, fix the following issue:\n\n{task['instruction']}\n\nMake the necessary code changes in-place in the repository and ensure git diff shows your patch."
            
        # C. InterCode-Bash Provisioning
        elif bmark == "intercode-bash":
            fs_id = "test"
            if "fs1" in task_id:
                fs_id = "fs_1"
                if "/workspace/testbed" not in instruction:
                    instruction = re.sub(r'/testbed\b', '/workspace/testbed', instruction)
            elif "fs2" in task_id:
                fs_id = "fs_2"
                if "/workspace/system" not in instruction:
                    instruction = re.sub(r'/system\b', '/workspace/system', instruction)
            elif "fs3" in task_id:
                fs_id = "fs_3"
                if "/workspace/backup" not in instruction:
                    instruction = re.sub(r'/backup\b', '/workspace/backup', instruction)
            elif "test" in task_id:
                fs_id = "test"
                if "/workspace/testbed" not in instruction:
                    instruction = re.sub(r'/testbed\b', '/workspace/testbed', instruction)
                    
            setup_ok, setup_err = setup_bash_fixtures(workspace, fs_id)
            if not setup_ok:
                print(f"    [!] Fixture setup error: {setup_err}")
                return {
                    "task_id": task_id,
                    "benchmark": bmark,
                    "instruction": task["instruction"],
                    "exit_code": -1,
                    "duration_seconds": 0.0,
                    "benchmark_reward": 0.0,
                    "eval_details": f"Infrastructure error: {setup_err}",
                    "grading_status": "infrastructure_error",
                    "stdout": "",
                    "stderr": setup_err,
                    "trajectory": {}
                }
            # Snapshot Bob's initial workspace state
            initial_snapshot = snapshot_dir_state(workspace)
            # Run oracle in isolated oracle environment
            oracle_info = run_bash_oracle(task, run_root, fs_id)
            
        start_time = time.time()
        net_mode = "full" if bmark in ("frames", "hotpot-qa") else "none"
        cmd = [
            str(BOB_BIN),
            "--config", str(cfg_file),
            "-p", instruction,
            "--mount", f"{workspace}:rw",
            "--net", net_mode,
            "--max-iterations", str(max_iterations),
            "--timeout", f"{timeout_sec}s",
            "-q"
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        
        outer_timeout = timeout_sec + 30
        try:
            stdout_output, stderr_output = proc.communicate(timeout=outer_timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"    [!] Watchdog expired after {outer_timeout}s. Terminating process group...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                stdout_output, stderr_output = proc.communicate(timeout=5)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                stdout_output, stderr_output = proc.communicate()
            exit_code = -9
            
        duration = time.time() - start_time
        print(f"    Finished in {duration:.1f}s (Exit code: {exit_code})")
        
        grade_res = grade_task(task, workspace, stdout_output, base_sha=base_sha, oracle_info=oracle_info, initial_snapshot=initial_snapshot)
        print(f"    Outcome: {grade_res['eval_details']}")
        
        if "patch" in grade_res and grade_res["patch"]:
            patch_file = batch_dir / f"{task_id}.patch"
            patch_file.write_text(grade_res["patch"])
            print(f"    Saved candidate patch to {patch_file}")
            
        db_path = data_dir / "dm_cli.db"
        traj = {}
        if db_path.exists():
            try:
                traj = extract_trajectory_from_db(db_path)
            except Exception as e:
                print(f"    [!] Trajectory extraction error: {e}")
                
            backup_ok = backup_sqlite_db(db_path, batch_dir / f"{task_id}.db")
            if not backup_ok:
                print(f"    [!] Database backup failed for task {task_id}")
            
        result = {
            "task_id": task_id,
            "benchmark": bmark,
            "instruction": task["instruction"],
            "exit_code": exit_code,
            "duration_seconds": duration,
            "benchmark_reward": grade_res.get("benchmark_reward"),
            "eval_details": grade_res.get("eval_details"),
            "patch_generated": grade_res.get("patch_generated"),
            "grading_status": grade_res.get("grading_status", "graded"),
            "resolved": grade_res.get("resolved"),
            "stdout": stdout_output,
            "stderr": stderr_output,
            "trajectory": traj
        }
        
        out_file = batch_dir / f"{task_id}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    Saved rollout to {out_file}")
        return result
        
    finally:
        if bmark == "swe-bench-lite":
            repo_dir = REPOS_DIR / "django.git"
            if not repo_dir.exists():
                scratch_repo = REPO_ROOT / "scratch" / "eval_harness" / "repos" / "django.git"
                if scratch_repo.exists():
                    repo_dir = scratch_repo
            if repo_dir.exists():
                subprocess.run(["git", f"--git-dir={repo_dir}", "worktree", "remove", "--force", str(workspace)], check=False, capture_output=True)
                subprocess.run(["git", f"--git-dir={repo_dir}", "worktree", "prune"], check=False, capture_output=True)
        shutil.rmtree(run_root, ignore_errors=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: runner.py <manifest.json> [--limit N] [--timeout SEC]")
        sys.exit(1)
        
    manifest_file = Path(sys.argv[1])
    if not manifest_file.exists():
        print(f"Manifest not found: {manifest_file}")
        sys.exit(1)
        
    with open(manifest_file) as f:
        manifest = json.load(f)
        
    tasks = manifest.get("tasks", []) if isinstance(manifest, dict) else manifest
    
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])
        tasks = tasks[:limit]
        
    timeout_sec = 900
    if "--timeout" in sys.argv:
        idx = sys.argv.index("--timeout")
        timeout_sec = int(sys.argv[idx + 1])
        
    max_iterations = 500
    if "--max-iterations" in sys.argv:
        idx = sys.argv.index("--max-iterations")
        max_iterations = int(sys.argv[idx + 1])
    elif "-m" in sys.argv:
        idx = sys.argv.index("-m")
        max_iterations = int(sys.argv[idx + 1])
        
    batch_timestamp = int(time.time())
    batch_dir = RESULTS_DIR / f"batch_{manifest_file.stem}_{batch_timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Loaded {len(tasks)} tasks from {manifest_file} (Timeout: {timeout_sec}s, Max iterations: {max_iterations})")
    print(f"[+] Batch output directory: {batch_dir}")
    
    results = []
    graded_count = 0
    passed_count = 0
    swe_patch_count = 0
    swe_resolved_count = 0
    
    for i, t in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}]")
        r = run_single_task(t, max_iterations=max_iterations, timeout_sec=timeout_sec, batch_dir=batch_dir)
        results.append(r)
        
        if r.get("grading_status") == "graded":
            graded_count += 1
            if r.get("benchmark_reward") == 1.0:
                passed_count += 1
        if r.get("benchmark") == "swe-bench-lite":
            if r.get("patch_generated"):
                swe_patch_count += 1
            if r.get("resolved"):
                swe_resolved_count += 1
                
    print(f"\n========================================================")
    print(f"[+] Batch execution complete.")
    if graded_count > 0:
        print(f"    Graded Tasks Passed: {passed_count}/{graded_count} ({passed_count/graded_count*100:.1f}%)")
    swe_total = len([t for t in tasks if t.get("benchmark") == "swe-bench-lite"])
    if swe_total > 0:
        print(f"    SWE-bench Patch Generation Rate: {swe_patch_count}/{swe_total} ({swe_patch_count/swe_total*100:.1f}%)")
        print(f"    SWE-bench Resolved Rate: {swe_resolved_count}/{swe_total} ({swe_resolved_count/swe_total*100:.1f}%) [Docker Validated]")
    
    summary_file = batch_dir / f"summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "manifest": str(manifest_file),
            "batch_timestamp": batch_timestamp,
            "total_tasks": len(tasks),
            "graded_tasks": graded_count,
            "passed_graded": passed_count,
            "graded_pass_rate": passed_count / graded_count if graded_count else 0,
            "swe_total": swe_total,
            "swe_patches_generated": swe_patch_count,
            "swe_patch_rate": swe_patch_count / swe_total if swe_total else 0,
            "swe_resolved": swe_resolved_count,
            "swe_resolved_rate": swe_resolved_count / swe_total if swe_total else 0,
            "results": [{
                "task_id": r["task_id"],
                "benchmark": r["benchmark"],
                "reward": r["benchmark_reward"],
                "patch_generated": r.get("patch_generated"),
                "resolved": r.get("resolved"),
                "grading_status": r.get("grading_status"),
                "duration": r["duration_seconds"],
                "exit_code": r["exit_code"],
                "eval_details": r["eval_details"]
            } for r in results]
        }, f, indent=2)
    print(f"    Saved batch summary to {summary_file}")

if __name__ == "__main__":
    main()
