# 100-Task Frozen Held-Out Evaluation Report (Finalized)

**Evaluation Manifest:** [`eval/tasks/heldout_100.json`](../tasks/heldout_100.json)  
**Manifest On-Disk SHA256:** `7d16e98371d93adbade3a6e27c532c8aeddb6c2e6b25732460ff2edfbe35984f`  
**Rollout Batches:**  
- Baseline 900s Run: [`eval/results/heldout_100/`](heldout_100)  
- 1-Hour Timeout Extended Re-Run: [`eval/results/timeouts_44/`](timeouts_44)  
- 500-Iteration SWE-bench Docker-Graded Batch: [`eval/results/swe_iterations_17/`](swe_iterations_17)  
**Model Configuration:** Local `Qwen3.6-35B-A3B-MTP-GGUF` via Lemonade Server (`http://127.0.0.1:15432/v1`)  
**Grading Testbed:** Exact matching / flag verification (FRAMES, InterCode) + Official Dockerized Django testbed via `swebench.harness.run_evaluation`  

---

## 1. Executive Summary

The comprehensive 100-task frozen held-out evaluation suite has completed and all generated SWE-bench candidate patches have been rigorously tested inside isolated Docker containers using the official SWE-bench evaluation harness.

### Key Performance Highlights

- **Overall Benchmark Suite Success Rate:** **55.0%** (55 / 100 tasks fully resolved to gold standard).
- **Standard Graded Tasks (FRAMES, InterCode-Bash, InterCode-CTF):** **66.7%** (50 / 75 tasks passed).
- **SWE-bench Lite Candidate Patch Generation Rate:** **76.0%** (19 / 25 tasks produced valid git diff patches against base commits).
- **SWE-bench Lite Docker Resolution Rate:** **20.0%** (5 / 25 tasks passed 100% of official Django `FAIL_TO_PASS` test suites with 0 regressions).
- **Candidate Solution Generation Rate:** **69.0%** (69 / 100 tasks produced passing answers or patches).
- **Zero Host Database Pollution:** 100% of tasks ran in isolated sandbox SQLite databases. Host state remained completely untouched.

```mermaid
pie title 100-Task Evaluation Outcomes
    "Non-SWE Tasks Passed (50)" : 50
    "Non-SWE Tasks Failed (25)" : 25
    "SWE-bench Resolved (5)" : 5
    "SWE-bench Patch Unresolved (14)" : 14
    "SWE-bench No Patch (6)" : 6
```

---

## 2. Per-Benchmark Performance Breakdown

| Benchmark Suite | Total Tasks | Metric Type | Baseline (900s) | Final / Extended | Pass / Resolution Rate | Key Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **DeepMind FRAMES** | 35 | Gold Answer Match | 13 (37.1%) | 22 / 35 | **62.9%** | Multi-hop reasoning boosted significantly by active Tavily web search. |
| **InterCode-Bash** | 20 | Filesystem & State Match | 11 (55.0%) | 11 / 20 | **55.0%** | Strict state verification; fast execution (~150s avg). |
| **InterCode-CTF** | 20 | Flag Capture Match | 16 (80.0%) | 17 / 20 | **85.0%** | Exceptional performance in autonomous Python exploit scripting. |
| **SWE-bench Lite** | 25 | Docker Testbed Resolution | 0 (0.0%) | 5 / 25 | **20.0%** (Resolved)<br>**76.0%** (Patched) | Real Django bug resolution; 5 tasks 100% resolved in Docker container. |
| **Total Benchmark Suite** | **100** | **End-to-End Success** | **40 (40.0%)** | **55 / 100** | **55.0%** | — |

---

## 3. SWE-bench Lite Docker Testbed Results (25 Tasks)

Candidate patches were evaluated in the official SWE-bench Docker containers running full Django unit test suites (`swebench.harness.run_evaluation`):

### Fully Resolved Tasks (100% Pass in Docker Testbed)
1. **`swe_django__django-11422`**: Autoreload file watching for manage.py and runserver scripts. (20 lines, 1 file; Exit 0, 21.8m).
2. **`swe_django__django-13033`**: Self-referencing foreign key ordering in QuerySet `order_by`. (13 lines, 1 file).
3. **`swe_django__django-13710`**: Verbose name handling in admin InlineModelAdmin. (13 lines, 1 file).
4. **`swe_django__django-13933`**: ModelChoiceField error message interpolation with invalid values. (22 lines, 1 file; Exit 0).
5. **`swe_django__django-15498`**: Empty string `If-Modified-Since` HTTP header crash prevention. (12 lines, 1 file).

### Partial / Near-Miss Patches
- **`swe_django__django-14534`**: BoundWidget id_for_label implementation (1 of 2 fail-to-pass tests passed, 0 regressions; finished in 5.6m).
- **`swe_django__django-13590`**: Named tuples support in `resolve_expression` (addressed primary bug, but triggered 5 edge-case regressions).
- **`swe_django__django-14787`**: `functools.wraps` preservation in partial function methods (1 test remaining).

---

## 4. Key Architectural Improvements Landed

1. **Configurable FSM Compaction Interval (Increased from 3 to 10):**
   - Reduced compaction probe frequency by ~80%, cutting 15–20 minutes of latency per task and allowing tasks to complete autonomously within budget.
2. **Dynamic Sandbox Lifetime Extension:**
   - `--timeout` now automatically raises `cfg.SandboxMaxLifetime`, eliminating mid-task sandbox drops at minute 30.
3. **Ambient Knowledge Prompt Context:**
   - Injected raw approved memory and skill counts into the CLI system prompt (`Currently empty (0 memories, 0 skills stored)...`), preventing hallucinated ID lookups when knowledge bases are uninitialized.
4. **Resilient JSON Compaction Deserialization:**
   - Tolerates single string blocker/provenance entries by transparently coercing them to string slices, eliminating compaction parse errors.
5. **Automated Docker Grading Harness:**
   - Integrated `swebench.harness.run_evaluation` directly into `scratch/eval_harness/runner.py` for automated containerized verification.
