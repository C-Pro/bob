#!/usr/bin/env python3
"""
test_grader_controls.py
Verification harness ensuring grading functions have zero false positives (negative controls)
and reliably detect correct solutions (positive controls) across all 4 benchmarks:
InterCode-CTF, DeepMind FRAMES, SWE-bench Lite, and InterCode-Bash.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import hashlib

from runner import (
    grade_task,
    grade_ctf,
    grade_frames,
    grade_bash,
    grade_swe,
    normalize_text,
    hash_file,
    snapshot_dir_state
)

def test_ctf_controls():
    print("[*] Testing CTF controls...")
    task = {
        "task_id": "test_ctf",
        "benchmark": "intercode-ctf",
        "eval_type": "ctf_flag_match",
        "gold_flag": "picoCTF{test_flag_12345}"
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        
        # 1. Negative: Untouched challenge files in workspace, flag NOT in stdout
        (ws / "challenge.txt").write_text("Binary data or challenge hints without flag\n")
        res_neg1 = grade_ctf(task, ws, stdout_output="I could not solve this challenge.")
        assert res_neg1["benchmark_reward"] == 0.0, f"Negative control 1 failed: {res_neg1}"
        
        # 2. Negative: Flag inside workspace file, but agent stdout does not contain it
        (ws / "secret.txt").write_text("picoCTF{test_flag_12345}")
        res_neg2 = grade_ctf(task, ws, stdout_output="I examined secret.txt but gave up.")
        assert res_neg2["benchmark_reward"] == 0.0, f"Negative control 2 failed: {res_neg2}"
        
        # 3. Positive: Agent outputs the flag in stdout
        res_pos = grade_ctf(task, ws, stdout_output="The flag is picoCTF{test_flag_12345}!")
        assert res_pos["benchmark_reward"] == 1.0, f"Positive control failed: {res_pos}"
        assert res_pos["grading_status"] == "graded"
        
    print("    CTF controls passed.")

def test_frames_controls():
    print("[*] Testing FRAMES controls...")
    task_num = {
        "task_id": "test_frames_num",
        "benchmark": "frames",
        "eval_type": "exact_match_normalized",
        "gold_answer": "1960"
    }
    task_text = {
        "task_id": "test_frames_text",
        "benchmark": "frames",
        "eval_type": "exact_match_normalized",
        "gold_answer": "Belgian Congo"
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        # 1. Negative numeric: Substring false positive (e.g. 196 or 19600)
        res_neg_num = grade_frames(task_num, ws, stdout_output="In the year 196 BC, things happened.")
        assert res_neg_num["benchmark_reward"] == 0.0, f"Negative control (numeric substring) failed: {res_neg_num}"
        
        res_neg_num2 = grade_frames(task_num, ws, stdout_output="The population was 196000.")
        assert res_neg_num2["benchmark_reward"] == 0.0, f"Negative control (numeric prefix) failed: {res_neg_num2}"
        
        # 2. Positive numeric: Exact boundary match with punctuation
        res_pos_num = grade_frames(task_num, ws, stdout_output="The independence was declared in 1960.")
        assert res_pos_num["benchmark_reward"] == 1.0, f"Positive control (numeric) failed: {res_pos_num}"
        
        # 3. Negative text: Wrong country
        res_neg_text = grade_frames(task_text, ws, stdout_output="The country was French Congo.")
        assert res_neg_text["benchmark_reward"] == 0.0, f"Negative control (text) failed: {res_neg_text}"
        
        # 4. Positive text: With conversational formatting and punctuation
        res_pos_text = grade_frames(task_text, ws, stdout_output="Based on the search results, it was Belgian Congo.")
        assert res_pos_text["benchmark_reward"] == 1.0, f"Positive control (text) failed: {res_pos_text}"
        
    print("    FRAMES controls passed.")

def test_swe_controls():
    print("[*] Testing SWE-bench controls...")
    task = {
        "task_id": "test_swe",
        "benchmark": "swe-bench-lite",
        "eval_type": "pytest_patch"
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        import subprocess
        subprocess.run(["git", "init"], cwd=ws, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "eval"], cwd=ws, check=True)
        subprocess.run(["git", "config", "user.email", "eval@example.com"], cwd=ws, check=True)
        (ws / "file.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "file.py"], cwd=ws, check=True)
        subprocess.run(["git", "commit", "-m", "base", "--no-gpg-sign"], cwd=ws, check=True)
        base_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ws, text=True).strip()
        
        # 1. Negative: No changes made
        res_neg = grade_swe(task, ws, base_sha=base_sha, run_docker=False)
        assert res_neg["patch_generated"] is False, f"Negative control (no edits) failed: {res_neg}"
        assert res_neg["resolved"] is False
        assert res_neg["benchmark_reward"] == 0.0
        
        # 2. Positive: File modified (binary diff check)
        (ws / "file.py").write_text("x = 2\n")
        res_pos = grade_swe(task, ws, base_sha=base_sha, run_docker=False)
        assert res_pos["patch_generated"] is True, f"Positive control (patch exists) failed: {res_pos}"
        assert res_pos["grading_status"] == "pending"
        assert res_pos["resolved"] is False
        assert "diff --git" in res_pos["patch"]
        assert "file.py" in res_pos["modified_files"]
        
        # 3. Positive untracked addition: Newly created file captured in patch
        (ws / "new_module.py").write_text("y = 42\n")
        res_untracked = grade_swe(task, ws, base_sha=base_sha, run_docker=False)
        assert res_untracked["patch_generated"] is True
        assert "new_module.py" in res_untracked["modified_files"]
        assert "y = 42" in res_untracked["patch"]
        
    print("    SWE-bench controls passed.")

def test_bash_controls():
    print("[*] Testing Bash controls...")
    task_query = {
        "task_id": "test_bash_query",
        "benchmark": "intercode-bash",
        "eval_type": "bash_fs",
        "instruction": "Count files",
        "gold_command": "echo 42"
    }
    task_state = {
        "task_id": "test_bash_state",
        "benchmark": "intercode-bash",
        "eval_type": "bash_fs",
        "instruction": "Remove temp file",
        "gold_command": "rm -f to_remove.txt"
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        
        # 1. Negative: Oracle unavailable -> infrastructure/oracle error, reward 0.0
        res_no_oracle = grade_bash(task_query, ws, stdout_output="42", oracle_info=None)
        assert res_no_oracle["benchmark_reward"] == 0.0
        assert res_no_oracle["grading_status"] == "oracle_error"
        
        # 2. Negative: Oracle command failed (exit code != 0) -> oracle error, reward 0.0
        oracle_fail = {"oracle_exit_code": 127, "oracle_stderr": "command not found", "state_changed": False}
        res_fail_oracle = grade_bash(task_query, ws, stdout_output="42", oracle_info=oracle_fail)
        assert res_fail_oracle["benchmark_reward"] == 0.0
        assert res_fail_oracle["grading_status"] == "oracle_error"
        
        # 3. Negative: Oracle produced empty output on query -> oracle error, reward 0.0
        oracle_empty = {"oracle_exit_code": 0, "oracle_stdout": "", "state_changed": False}
        res_empty_oracle = grade_bash(task_query, ws, stdout_output="Arbitrary text", oracle_info=oracle_empty)
        assert res_empty_oracle["benchmark_reward"] == 0.0
        assert res_empty_oracle["grading_status"] == "oracle_error"
        
        # 4. Negative: Arbitrary non-empty prose without answer -> reward 0.0
        oracle_valid = {"oracle_exit_code": 0, "oracle_stdout": "42\n", "state_changed": False}
        res_neg = grade_bash(task_query, ws, stdout_output="I checked the files and completed the task successfully.", oracle_info=oracle_valid)
        assert res_neg["benchmark_reward"] == 0.0, f"Negative control (prose) failed: {res_neg}"
        
        # 5. Positive: Numeric token present
        res_pos = grade_bash(task_query, ws, stdout_output="There are exactly 42 files found.", oracle_info=oracle_valid)
        assert res_pos["benchmark_reward"] == 1.0, f"Positive control (query) failed: {res_pos}"
        
        # 6. Negative: Incomplete path set (4 of 5 target paths) -> MUST FAIL (no 80% bypass!)
        oracle_paths = {
            "oracle_exit_code": 0,
            "oracle_stdout": "testbed/dir1/a.txt\ntestbed/dir1/b.txt\ntestbed/dir1/c.txt\ntestbed/dir1/d.txt\ntestbed/dir1/e.txt\n",
            "state_changed": False
        }
        res_partial_paths = grade_bash(
            task_query, ws,
            stdout_output="Files: testbed/dir1/a.txt, testbed/dir1/b.txt, testbed/dir1/c.txt, testbed/dir1/d.txt",
            oracle_info=oracle_paths
        )
        assert res_partial_paths["benchmark_reward"] == 0.0, f"Incomplete path set should score 0.0: {res_partial_paths}"
        
        # 7. Negative: Extra hallucinated path target -> MUST FAIL (strict exact equality!)
        res_extra_paths = grade_bash(
            task_query, ws,
            stdout_output="Files: testbed/dir1/a.txt, testbed/dir1/b.txt, testbed/dir1/c.txt, testbed/dir1/d.txt, testbed/dir1/e.txt, testbed/dir1/extra.txt",
            oracle_info=oracle_paths
        )
        assert res_extra_paths["benchmark_reward"] == 0.0, f"Extra path targets should score 0.0: {res_extra_paths}"
        
        # 8. Positive: Exact 5 target paths present
        res_full_paths = grade_bash(
            task_query, ws,
            stdout_output="Files: testbed/dir1/a.txt, testbed/dir1/b.txt, testbed/dir1/c.txt, testbed/dir1/d.txt, testbed/dir1/e.txt",
            oracle_info=oracle_paths
        )
        assert res_full_paths["benchmark_reward"] == 1.0, f"Exact complete path set should score 1.0: {res_full_paths}"
        
        # 9. State change negative: File was NOT removed
        (ws / "to_remove.txt").write_text("data")
        init_snapshot_file = {"files": {"to_remove.txt": hash_file(ws / "to_remove.txt")}, "dirs": []}
        oracle_info_file_del = {
            "oracle_exit_code": 0,
            "oracle_stdout": "",
            "state_changed": True,
            "removed_files": ["to_remove.txt"],
            "removed_dirs": [],
            "added_files": [],
            "added_dirs": [],
            "modified_files": []
        }
        res_state_neg = grade_bash(task_state, ws, stdout_output="Done", oracle_info=oracle_info_file_del, initial_snapshot=init_snapshot_file)
        assert res_state_neg["benchmark_reward"] == 0.0, f"Negative control (file not removed) failed: {res_state_neg}"
        
        # 10. State change positive: File was removed
        (ws / "to_remove.txt").unlink()
        res_state_pos = grade_bash(task_state, ws, stdout_output="Done", oracle_info=oracle_info_file_del, initial_snapshot=init_snapshot_file)
        assert res_state_pos["benchmark_reward"] == 1.0, f"Positive control (file removed) failed: {res_state_pos}"
        
        # 11. State change directory negative: Empty directory NOT removed
        (ws / "empty_dir").mkdir(parents=True, exist_ok=True)
        init_snapshot_dir = {"files": {}, "dirs": ["empty_dir"]}
        oracle_info_dir_del = {
            "oracle_exit_code": 0,
            "oracle_stdout": "",
            "state_changed": True,
            "removed_files": [],
            "removed_dirs": ["empty_dir"],
            "added_files": [],
            "added_dirs": [],
            "modified_files": []
        }
        res_dir_neg = grade_bash(task_state, ws, stdout_output="Done", oracle_info=oracle_info_dir_del, initial_snapshot=init_snapshot_dir)
        assert res_dir_neg["benchmark_reward"] == 0.0, f"Negative control (dir not removed) failed: {res_dir_neg}"
        
        # 12. State change directory positive: Empty directory removed
        (ws / "empty_dir").rmdir()
        res_dir_pos = grade_bash(task_state, ws, stdout_output="Done", oracle_info=oracle_info_dir_del, initial_snapshot=init_snapshot_dir)
        assert res_dir_pos["benchmark_reward"] == 1.0, f"Positive control (dir removed) failed: {res_dir_pos}"
        
        # 13. State change negative: Created file content hash mismatch
        dummy_hash = hashlib.sha256(b"correct content").hexdigest()
        (ws / "created.txt").write_text("WRONG content")
        init_snapshot_empty = {"files": {}, "dirs": []}
        oracle_create_state = {
            "oracle_exit_code": 0,
            "oracle_stdout": "",
            "state_changed": True,
            "removed_files": [],
            "removed_dirs": [],
            "added_files": ["created.txt"],
            "added_dirs": [],
            "modified_files": [],
            "file_hashes_after": {"created.txt": dummy_hash}
        }
        res_create_neg = grade_bash(task_state, ws, stdout_output="Done", oracle_info=oracle_create_state, initial_snapshot=init_snapshot_empty)
        assert res_create_neg["benchmark_reward"] == 0.0, f"Created file content mismatch should score 0.0: {res_create_neg}"
        
        # 14. State change positive: Created file content matches expected hash
        (ws / "created.txt").write_text("correct content")
        res_create_pos = grade_bash(task_state, ws, stdout_output="Done", oracle_info=oracle_create_state, initial_snapshot=init_snapshot_empty)
        assert res_create_pos["benchmark_reward"] == 1.0, f"Created file with correct content should score 1.0: {res_create_pos}"
        
    print("    Bash controls passed.")

if __name__ == "__main__":
    test_ctf_controls()
    test_frames_controls()
    test_swe_controls()
    test_bash_controls()
    print("\n[+] All 14 negative and positive controls verified successfully!")
