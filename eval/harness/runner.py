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
import importlib.util
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = BASE_DIR.parent.parent
BOB_BIN = REPO_ROOT / "bin" / "bob"
TEMPLATE_CONFIG = BASE_DIR / "config_template.yaml"
RESULTS_DIR = BASE_DIR.parent / "results"
ROLLOUTS_DIR = RESULTS_DIR / "rollouts"
CTF_ASSETS_DIR = BASE_DIR / "ctf_assets"
BASH_FIXTURES_DIR = BASE_DIR / "bash_fixtures"
REPOS_DIR = BASE_DIR / "repos"

VALID_LOOP_GUARD_MODES = ("off", "shadow", "warn", "hybrid", "enforce")

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

def resolve_swebench_python() -> str:
    override = os.environ.get("BOB_EVAL_SWEBENCH_PYTHON", "").strip()
    if override:
        return override
    if importlib.util.find_spec("swebench") is not None:
        return sys.executable
    candidates = sorted(
        (Path.home() / ".local" / "share" / "mise" / "installs" / "python").glob("*/bin/python"),
        reverse=True,
    )
    for cand in candidates:
        try:
            res = subprocess.run(
                [str(cand), "-c", "import swebench"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0:
                return str(cand)
        except Exception:
            continue
    return sys.executable


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
        resolve_swebench_python(), "-m", "swebench.harness.run_evaluation",
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

def resolve_tavily_key() -> str:
    k = os.environ.get("TAVILY_API_KEY", "")
    if k:
        return k
    user_cfg = Path.home() / ".config" / "bob" / "config.yaml"
    if user_cfg.exists():
        try:
            for line in user_cfg.read_text().splitlines():
                if line.strip().startswith("tavily_api_key:"):
                    cand = line.split(":", 1)[1].strip().strip('\"').strip('\'')
                    if cand and not cand.startswith("${"):
                        return cand
        except Exception:
            pass
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                if line.startswith("TAVILY_API_KEY="):
                    cand = line.split("=", 1)[1].strip().strip('\"').strip('\'')
                    if cand:
                        return cand
        except Exception:
            pass
    return ""

def resolve_llm_config(model: str = None, base_url: str = None, api_key: str = None):
    if not model:
        model = os.environ.get("BOB_EVAL_MODEL") or os.environ.get("GEMINI_MODEL") or os.environ.get("OPENAI_MODEL") or "Qwen3.6-35B-A3B-MTP-GGUF"
    if model in ("3.5-flash-lite", "gemini-3.5-flash-lite"):
        model = "gemini-3.5-flash-lite"
    elif model in ("3.7-flash", "gemini-3.7-flash"):
        model = "gemini-3.7-flash"
    elif model in ("3.8-flash", "gemini-3.8-flash"):
        model = "gemini-3.8-flash"

    if not base_url:
        base_url = os.environ.get("BOB_EVAL_BASE_URL") or os.environ.get("GEMINI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            if "gemini" in model.lower():
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            else:
                base_url = "http://127.0.0.1:15432/v1"

    if not api_key:
        api_key = os.environ.get("BOB_EVAL_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            user_cfg = Path.home() / ".config" / "bob" / "config.yaml"
            if user_cfg.exists():
                try:
                    for line in user_cfg.read_text().splitlines():
                        if line.strip().startswith("api_key:"):
                            k = line.split(":", 1)[1].strip().strip('\"').strip('\'')
                            if k and not k.startswith("${"):
                                api_key = k
                                break
                except Exception:
                    pass
        if not api_key:
            env_file = REPO_ROOT / ".env"
            if env_file.exists():
                try:
                    for line in env_file.read_text().splitlines():
                        if line.startswith("GEMINI_API_KEY=") or line.startswith("OPENAI_API_KEY="):
                            k = line.split("=", 1)[1].strip().strip('\"').strip('\'')
                            if k:
                                api_key = k
                                break
                except Exception:
                    pass
        if not api_key:
            api_key = "dummy-key"

    return model, base_url, api_key


def parse_bool_flag(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean flag value: {value!r}")


def resolve_system1_config(
    system1_url: str = None,
    system1_model: str = None,
    system1_loop_guard_mode: str = None,
    system1_fallback_on_error=None,
    system1_timeout: str = None,
) -> dict:
    url = (
        system1_url
        if system1_url is not None
        else (os.environ.get("BOB_EVAL_SYSTEM1_URL") or os.environ.get("SYSTEM1_BASE_URL") or "")
    ).strip()

    model = (
        system1_model
        if system1_model is not None
        else (os.environ.get("BOB_EVAL_SYSTEM1_MODEL") or os.environ.get("SYSTEM1_MODEL") or "bob-s1-0.8b")
    ).strip() or "bob-s1-0.8b"

    loop_guard_mode = (
        system1_loop_guard_mode
        if system1_loop_guard_mode is not None
        else (os.environ.get("BOB_EVAL_SYSTEM1_LOOP_GUARD_MODE") or os.environ.get("SYSTEM1_LOOP_GUARD_MODE") or "hybrid")
    ).strip().lower() or "hybrid"
    if loop_guard_mode not in VALID_LOOP_GUARD_MODES:
        raise ValueError(
            f"Invalid --system1-loop-guard-mode {loop_guard_mode!r}; expected one of {', '.join(VALID_LOOP_GUARD_MODES)}"
        )

    env_fallback = os.environ.get("BOB_EVAL_SYSTEM1_FALLBACK_ON_ERROR")
    if env_fallback is None:
        env_fallback = os.environ.get("SYSTEM1_FALLBACK_ON_ERROR")
    fallback_on_error = (
        parse_bool_flag(system1_fallback_on_error, default=True)
        if system1_fallback_on_error is not None
        else parse_bool_flag(env_fallback, default=True)
    )

    timeout = (
        system1_timeout
        if system1_timeout is not None
        else (os.environ.get("BOB_EVAL_SYSTEM1_TIMEOUT") or os.environ.get("SYSTEM1_TIMEOUT") or "60s")
    ).strip() or "60s"

    return {
        "enabled": bool(url),
        "url": url,
        "model": model,
        "loop_guard_mode": loop_guard_mode,
        "fallback_on_error": fallback_on_error,
        "timeout": timeout,
    }


def derive_system1_health_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raise ValueError(f"Invalid System-1 URL scheme: {base_url!r}")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    return f"{raw}/healthz"


def check_system1_health(base_url: str, timeout_sec: float = 5.0) -> dict:
    health_url = derive_system1_health_url(base_url)
    req = urllib.request.Request(health_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        status_code = getattr(resp, "status", 200)
        body = resp.read().decode("utf-8", errors="replace")
    if status_code != 200:
        raise RuntimeError(f"System-1 healthz returned HTTP {status_code}: {body[:200]}")
    payload = json.loads(body) if body.strip() else {}
    status_val = str(payload.get("status", "")).strip().lower()
    ready_val = payload.get("ready")
    if ready_val is False or (status_val and status_val not in ("ok", "ready", "healthy")):
        raise RuntimeError(f"System-1 sidecar not ready at {health_url}: {payload}")
    return payload


def _yaml_quote(val: str) -> str:
    return json.dumps(str(val))


def build_config_yaml(
    resolved_api_key: str,
    resolved_base_url: str,
    resolved_model: str,
    tavily_key: str,
    data_dir: Path,
    max_iterations: int,
    system1_cfg: dict = None,
) -> str:
    cfg = f"""llm:
  api_key: {_yaml_quote(resolved_api_key)}
  base_url: {_yaml_quote(resolved_base_url)}
  model: {_yaml_quote(resolved_model)}
  first_byte_timeout: "5m"
  request_timeout: "10m"

web:
  tavily_api_key: {_yaml_quote(tavily_key)}
  tavily_base_url: "https://api.tavily.com"

data_dir: {_yaml_quote(str(data_dir))}

cli:
  max_iterations: {int(max_iterations)}

sandbox:
  default_driver: "bwrap"
  allowed_network_modes: ["none", "restricted", "full"]
  default_exec_timeout: "2m"
  max_exec_timeout: "1h"
  max_lifetime: "2h"

fsm:
  compaction_interval_iterations: 10
"""
    if system1_cfg and system1_cfg.get("url"):
        fallback_literal = "true" if system1_cfg.get("fallback_on_error", True) else "false"
        timeout_val = system1_cfg.get("timeout", "60s")
        cfg += f"""
system1:
  base_url: {_yaml_quote(system1_cfg["url"])}
  model: {_yaml_quote(system1_cfg.get("model", "bob-s1-0.8b"))}
  timeout: {_yaml_quote(timeout_val)}
  loop_guard_mode: {_yaml_quote(system1_cfg.get("loop_guard_mode", "hybrid"))}
  fallback_on_error: {fallback_literal}
"""
    return cfg


def extract_task_system1_metrics(traj: dict) -> dict:
    metrics = {
        "decision_events": 0,
        "compaction_gate_calls": 0,
        "compaction_calibration_applied": 0,
        "compaction_calibration_unapplied": 0,
        "compaction_skipped_probes": 0,
        "compaction_system1_skipped_probes": 0,
        "compaction_deferred_to_system2": 0,
        "compaction_gate_latency_ms": 0,
        "compaction_summarizer_latency_ms": 0,
        "compaction_total_latency_ms": 0,
        "loop_guard_calls": 0,
        "loop_guard_interventions": 0,
        "fallback_count": 0,
        "error_count": 0,
    }
    if not isinstance(traj, dict):
        return metrics

    for ev in traj.get("events", []) or []:
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("event_type")
        payload = ev.get("payload")
        if not isinstance(payload, dict):
            continue

        if ev_type == "system1_decision":
            metrics["decision_events"] += 1
            task_name = payload.get("task", "")
            action = str(payload.get("applied_action", "")).lower()
            latency_ms = int(payload.get("latency_ms", 0) or 0)

            if payload.get("fallback_used"):
                metrics["fallback_count"] += 1
            if payload.get("error") or action in ("error", "probe_error"):
                metrics["error_count"] += 1

            if task_name == "compaction":
                metrics["compaction_gate_calls"] += 1
                metrics["compaction_gate_latency_ms"] += latency_ms
                if payload.get("calibration_applied"):
                    metrics["compaction_calibration_applied"] += 1
                else:
                    metrics["compaction_calibration_unapplied"] += 1
                if action == "defer_to_system2":
                    metrics["compaction_deferred_to_system2"] += 1
            elif task_name == "loop_guard":
                metrics["loop_guard_calls"] += 1
                if action in ("warn", "block", "intervene"):
                    metrics["loop_guard_interventions"] += 1

        elif ev_type == "compaction":
            decision_source = str(payload.get("decision_source", "")).lower()
            should_compact = payload.get("should_compact")
            metrics["compaction_summarizer_latency_ms"] += int(payload.get("summarizer_latency_ms", 0) or 0)
            if decision_source == "system1_skip":
                metrics["compaction_system1_skipped_probes"] += 1
                metrics["compaction_skipped_probes"] += 1
            elif should_compact is False and decision_source:
                metrics["compaction_skipped_probes"] += 1

    metrics["compaction_total_latency_ms"] = (
        metrics["compaction_gate_latency_ms"] + metrics["compaction_summarizer_latency_ms"]
    )
    return metrics


def summarize_batch_metrics(results: list) -> dict:
    per_benchmark = {}
    s1_totals = extract_task_system1_metrics({})

    for r in results:
        bmark = r.get("benchmark") or "unknown"
        bucket = per_benchmark.setdefault(bmark, {
            "total": 0,
            "graded": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "swe_patched": 0,
            "swe_resolved": 0,
        })
        bucket["total"] += 1
        if r.get("grading_status") == "graded":
            bucket["graded"] += 1
            if r.get("benchmark_reward") == 1.0:
                bucket["passed"] += 1
        if bmark == "swe-bench-lite":
            if r.get("patch_generated"):
                bucket["swe_patched"] += 1
            if r.get("resolved"):
                bucket["swe_resolved"] += 1

        task_s1 = r.get("system1_metrics")
        if not isinstance(task_s1, dict):
            task_s1 = extract_task_system1_metrics(r.get("trajectory", {}))
        for k in s1_totals:
            s1_totals[k] += int(task_s1.get(k, 0) or 0)

    for bucket in per_benchmark.values():
        denom = bucket["graded"] or bucket["total"]
        bucket["pass_rate"] = (bucket["passed"] / denom) if denom else 0.0

    return {
        "per_benchmark": per_benchmark,
        "system1_metrics": s1_totals,
    }

def resolve_swe_repo(task: dict) -> Path:
    repo = task.get("repo", "django/django")
    repo_slug = repo.split("/")[-1]
    candidates = [
        REPOS_DIR / f"{repo_slug}.git",
        REPO_ROOT / "scratch" / "eval_harness" / "repos" / f"{repo_slug}.git",
        REPOS_DIR / f"{repo.replace('/', '__')}.git",
        REPO_ROOT / "scratch" / "eval_harness" / "repos" / f"{repo.replace('/', '__')}.git",
    ]
    for c in candidates:
        if c.exists():
            return c

    target_dir = REPO_ROOT / "scratch" / "eval_harness" / "repos" / f"{repo_slug}.git"
    print(f"[-] Cloned bare mirror of {repo} not found. Cloning to {target_dir}...")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(target_dir)],
        check=True
    )
    return target_dir

def run_single_task(task: dict, max_iterations: int = 500, timeout_sec: int = 900, batch_dir: Path = None,
                    model: str = None, base_url: str = None, api_key: str = None,
                    system1_cfg: dict = None) -> dict:
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
    tavily_key = resolve_tavily_key()
    resolved_model, resolved_base_url, resolved_api_key = resolve_llm_config(model, base_url, api_key)
    resolved_s1 = system1_cfg if system1_cfg is not None else resolve_system1_config()
    cfg_content = build_config_yaml(
        resolved_api_key=resolved_api_key,
        resolved_base_url=resolved_base_url,
        resolved_model=resolved_model,
        tavily_key=tavily_key,
        data_dir=data_dir,
        max_iterations=max_iterations,
        system1_cfg=resolved_s1,
    )
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
            repo_dir = resolve_swe_repo(task)
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
            "system1_metrics": extract_task_system1_metrics(traj),
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
            try:
                repo_dir = resolve_swe_repo(task)
                if repo_dir.exists():
                    subprocess.run(["git", f"--git-dir={repo_dir}", "worktree", "remove", "--force", str(workspace)], check=False, capture_output=True)
                    subprocess.run(["git", f"--git-dir={repo_dir}", "worktree", "prune"], check=False, capture_output=True)
            except Exception:
                pass
        shutil.rmtree(run_root, ignore_errors=True)

def _extract_cli_value(argv: list, flag: str, default=None):
    prefix = f"{flag}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def _extract_cli_bool(argv: list, flag: str, default=None):
    prefix = f"{flag}="
    for i, arg in enumerate(argv):
        if arg == flag:
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                return argv[i + 1]
            return "true"
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return default


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: runner.py <manifest.json> [--limit N] [--timeout SEC] "
            "[--max-iterations N] [--system1-url URL] [--system1-model NAME] "
            "[--system1-loop-guard-mode MODE] [--system1-fallback-on-error BOOL]"
        )
        sys.exit(1)
        
    manifest_file = Path(sys.argv[1])
    if not manifest_file.exists():
        print(f"Manifest not found: {manifest_file}")
        sys.exit(1)
        
    with open(manifest_file) as f:
        manifest = json.load(f)
        
    tasks = manifest.get("tasks", []) if isinstance(manifest, dict) else manifest
    
    limit_val = _extract_cli_value(sys.argv, "--limit")
    if limit_val is not None:
        tasks = tasks[:int(limit_val)]
        
    timeout_sec = int(_extract_cli_value(sys.argv, "--timeout", 900))
    max_iter_val = _extract_cli_value(sys.argv, "--max-iterations")
    if max_iter_val is None:
        max_iter_val = _extract_cli_value(sys.argv, "-m", 30)
    max_iterations = int(max_iter_val)

    model = _extract_cli_value(sys.argv, "--model")
    base_url = _extract_cli_value(sys.argv, "--base-url")
    api_key = _extract_cli_value(sys.argv, "--api-key")

    system1_url = _extract_cli_value(sys.argv, "--system1-url")
    system1_model = _extract_cli_value(sys.argv, "--system1-model")
    system1_loop_guard_mode = _extract_cli_value(sys.argv, "--system1-loop-guard-mode")
    system1_fallback_on_error = _extract_cli_bool(sys.argv, "--system1-fallback-on-error")
    system1_timeout = _extract_cli_value(sys.argv, "--system1-timeout")
    system1_cfg = resolve_system1_config(
        system1_url=system1_url,
        system1_model=system1_model,
        system1_loop_guard_mode=system1_loop_guard_mode,
        system1_fallback_on_error=system1_fallback_on_error,
        system1_timeout=system1_timeout,
    )

    split = _extract_cli_value(sys.argv, "--split")
    if split is None:
        if "train" in manifest_file.parts:
            split = "train"
        elif "heldout" in manifest_file.parts:
            split = "heldout"

    batch_timestamp = int(time.time())
    output_dir_val = _extract_cli_value(sys.argv, "--output-dir")
    if output_dir_val:
        batch_dir = Path(output_dir_val)
    elif split:
        batch_dir = RESULTS_DIR / split / f"batch_{manifest_file.stem}_{batch_timestamp}"
    else:
        batch_dir = RESULTS_DIR / f"batch_{manifest_file.stem}_{batch_timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    resolved_model, resolved_base_url, _ = resolve_llm_config(model, base_url, api_key)
    print(f"[+] Loaded {len(tasks)} tasks from {manifest_file} (Timeout: {timeout_sec}s, Max iterations: {max_iterations})")
    print(f"[+] Model: {resolved_model} (Base URL: {resolved_base_url})")
    healthz_info = None
    if system1_cfg["enabled"]:
        print(
            f"[+] System-1: {system1_cfg['url']} (model={system1_cfg['model']}, "
            f"loop_guard_mode={system1_cfg['loop_guard_mode']}, "
            f"fallback_on_error={system1_cfg['fallback_on_error']})"
        )
        try:
            healthz_info = check_system1_health(system1_cfg["url"])
            print(f"[+] System-1 /healthz OK: {healthz_info}")
        except Exception as e:
            if not system1_cfg["fallback_on_error"]:
                print(f"[!] System-1 /healthz failed with fallback_on_error=false: {e}")
                sys.exit(2)
            print(f"[-] Warning: System-1 /healthz check failed (fallback_on_error=true): {e}")
    print(f"[+] Batch output directory: {batch_dir}")
    
    results = []
    graded_count = 0
    passed_count = 0
    swe_patch_count = 0
    swe_resolved_count = 0
    resume = "--resume" in sys.argv
    
    for i, t in enumerate(tasks):
        t_id = t.get("task_id") or t.get("instance_id") or f"{t.get('benchmark')}_{i}"
        out_file = batch_dir / f"{t_id}.json"
        if resume and out_file.exists():
            try:
                with open(out_file) as rf:
                    r = json.load(rf)
                print(f"\n[{i+1}/{len(tasks)}] Resuming {t_id} (already completed)")
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
                continue
            except Exception as e:
                print(f"[-] Could not read {out_file}: {e}. Re-running task...")

        print(f"\n[{i+1}/{len(tasks)}]")
        try:
            r = run_single_task(t, max_iterations=max_iterations, timeout_sec=timeout_sec, batch_dir=batch_dir,
                                model=resolved_model, base_url=resolved_base_url, api_key=api_key,
                                system1_cfg=system1_cfg)
        except Exception as e:
            print(f"[-] Task {t_id} failed with unhandled exception: {e}")
            r = {
                "task_id": t_id,
                "benchmark": t.get("benchmark"),
                "status": "error",
                "error": str(e),
                "duration_seconds": 0.0,
                "exit_code": -1,
                "grading_status": "graded",
                "benchmark_reward": 0.0,
                "resolved": False,
                "eval_details": f"Execution error: {e}",
            }
            with open(out_file, "w") as f:
                json.dump(r, f, indent=2)
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
                
    batch_agg = summarize_batch_metrics(results)
    s1_agg = batch_agg["system1_metrics"]
    per_bench = batch_agg["per_benchmark"]

    print(f"\n========================================================")
    print(f"[+] Batch execution complete.")
    if graded_count > 0:
        print(f"    Graded Tasks Passed: {passed_count}/{graded_count} ({passed_count/graded_count*100:.1f}%)")
    for b_name, b_stats in sorted(per_bench.items()):
        print(f"    - {b_name}: {b_stats['passed']}/{b_stats['graded']} ({b_stats['pass_rate']*100:.1f}%)")
    swe_total = len([t for t in tasks if t.get("benchmark") == "swe-bench-lite"])
    if swe_total > 0:
        print(f"    SWE-bench Patch Generation Rate: {swe_patch_count}/{swe_total} ({swe_patch_count/swe_total*100:.1f}%)")
        print(f"    SWE-bench Resolved Rate: {swe_resolved_count}/{swe_total} ({swe_resolved_count/swe_total*100:.1f}%) [Docker Validated]")
    if system1_cfg["enabled"] or s1_agg["decision_events"] > 0:
        print(
            f"    System-1 Telemetry: decisions={s1_agg['decision_events']}, "
            f"compaction_probes={s1_agg['compaction_gate_calls']} "
            f"(calibrated={s1_agg['compaction_calibration_applied']}, "
            f"uncalibrated={s1_agg['compaction_calibration_unapplied']}, "
            f"skipped={s1_agg['compaction_system1_skipped_probes']}), "
            f"compaction_ms={s1_agg['compaction_total_latency_ms']} "
            f"(gate={s1_agg['compaction_gate_latency_ms']}, summarizer={s1_agg['compaction_summarizer_latency_ms']}), "
            f"loop_guard_interventions={s1_agg['loop_guard_interventions']}/{s1_agg['loop_guard_calls']}, "
            f"fallbacks={s1_agg['fallback_count']}, errors={s1_agg['error_count']}"
        )
    
    summary_file = batch_dir / f"summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "manifest": str(manifest_file),
            "model": resolved_model,
            "base_url": resolved_base_url,
            "system1": {
                **system1_cfg,
                "healthz": healthz_info,
            },
            "batch_timestamp": batch_timestamp,
            "total_tasks": len(tasks),
            "graded_tasks": graded_count,
            "passed_graded": passed_count,
            "graded_pass_rate": passed_count / graded_count if graded_count else 0,
            "per_benchmark": per_bench,
            "system1_metrics": s1_agg,
            "swe_total": swe_total,
            "swe_patches_generated": swe_patch_count,
            "swe_patch_rate": swe_patch_count / swe_total if swe_total else 0,
            "swe_resolved": swe_resolved_count,
            "swe_resolved_rate": swe_resolved_count / swe_total if swe_total else 0,
            "results": [{
                "task_id": r["task_id"],
                "benchmark": r["benchmark"],
                "reward": r.get("benchmark_reward"),
                "patch_generated": r.get("patch_generated"),
                "resolved": r.get("resolved"),
                "grading_status": r.get("grading_status"),
                "duration": r.get("duration_seconds", 0.0),
                "exit_code": r.get("exit_code", -1),
                "system1_metrics": r.get("system1_metrics"),
                "eval_details": r.get("eval_details"),
            } for r in results]
        }, f, indent=2)
    print(f"    Saved batch summary to {summary_file}")

if __name__ == "__main__":
    main()
