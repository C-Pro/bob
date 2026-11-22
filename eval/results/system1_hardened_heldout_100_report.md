# System-1 Hardened Control Plane: Held-Out 100-Task Evaluation

**Run:** [`batch_heldout_100_system1_hardened/summary.json`](heldout/batch_heldout_100_system1_hardened/summary.json) (100 task JSON trajectories, 100 SQLite snapshots, 21 generated SWE patches) on `HEAD` `cabd362` (`feat/bob-s1-go-integration`).
**Controller:** CPU `bob-s1-0.8b` (`bf16`, `sdpa`, `BOB_S1_MERGE_LORA=1`, `BOB_S1_CPU_THREADS=8`, `hybrid` loop guard, `fallback_on_error=false`, `timeout=60s`) with Phases 0–4 control-plane hardening (`eval/harness/runner.py --timeout 3600 --max-iterations 500`); task-solving LLM `Qwen3.6-35B-A3B-MTP-GGUF` via `lemond`.
**Comparators:** Pre-hardening [`batch_heldout_100_system1/summary.json`](heldout/batch_heldout_100_system1/summary.json) ([prior analysis](system1_heldout_100_report.md)) and the Docker-regraded merged baseline ([baseline report](baseline_100_report.md) + [8-patch regrade](heldout/baseline_pending_8_docker_regrade.json)).

## Executive summary

The hardened control plane resolved **59/98 gradable tasks (60.2%)**, improving by **+3 tasks (+3.1 pp)** over the pre-hardening System-1 run (**56/98, 57.1%**) and **+1 task (+1.0 pp)** over the Docker-regraded merged baseline (**58/98, 59.2%**; published baseline was **55/100**). Aggregate wall time across the 100 tasks dropped by **9,050.0 s (2.51 hours, −12.5%)**, from **72,376.4 s (20.10 h)** to **63,326.4 s (17.59 h)**.

| Benchmark | Tasks | Merged Baseline | Prior System-1 | Hardened System-1 | Delta vs. Prior S1 / Merged Base |
|---|---:|---:|---:|---:|---:|
| FRAMES | 35 | 22 (62.9%) | 20 (57.1%) | **21 (60.0%)** | +1 / −1 |
| InterCode-Bash | 20 (18 gradable) | 11 (61.1%) | 10 (55.6%) | **11 (61.1%)** | +1 / 0 |
| InterCode-CTF | 20 | 17 (85.0%) | 18 (90.0%) | **19 (95.0%)** | +1 / +2 |
| SWE-bench Lite | 25 | 8 (32.0%) | 8 (32.0%) | **8 (32.0%)** *(21/25 patched)* | 0 / 0 *(+2 patches vs. 19)* |
| **Total (98 gradable)** | **100** | **58 (59.2%)** | **56 (57.1%)** | **59 (60.2%)** | **+3 / +1** |

Two InterCode-Bash query tasks (`ic_bash_fs2_029`, `ic_bash_fs2_050`) remain ungradable (`oracle_error: Oracle produced empty stdout for a query task`) across every batch. Note that separate runs with stochastic LLM sampling remain subject to run-to-run search variance in addition to control-plane changes.

## What improved in the hardened control plane

1. **False-positive loop termination eliminated (`ic_ctf_003` +3pp CTF / non-destructive downgrades):**
   - `loop_guard_interventions` dropped from `1` fatal termination in the prior run to **`0` across 1,806 probes**.
   - On [`ic_ctf_003`](heldout/batch_heldout_100_system1_hardened/ic_ctf_003.json) — previously aborted after an unsupported `force_synthesis` strike at 812.5 s — Bob cleanly decoded the whitespace flag and solved the task in **113.6 s (7 steps)**.
   - On [`swe_django__django-13033`](heldout/batch_heldout_100_system1_hardened/swe_django__django-13033.json), `bob-s1` emitted `force_synthesis` **5 times** without deterministic repetition; Phase 1 downgraded all 5 to non-destructive `advisory` nudges.

2. **Tri-zone compaction gate & uncertainty deferrals actively filtered probes:**
   - Across **195** calibrated compaction probes (`calibration_applied=195`, `unapplied=0`, `P(compact)` range `0.6279–0.9978`, median `0.9717`, mean `0.9518`), the gate no longer approved 100% blindly (`226/226` previously):
     - **`system1_skip` (`P < 0.8394`):** **9** probes skipped immediately by System-1 (`0 -> 9`), avoiding unnecessary Stage-2 calls (`ic_ctf_070`, `10924`, `11422` ×2, `12284`, `12470`, `14016`, `14667`, `14915`).
     - **`system2_defer`:** **13** compaction events deferred to Stage-2 (`3 -> 13`; 9 counted as deferred gate decisions in batch summary), where Stage-2 declined compaction (`should_compact=false`) on **10** events (`9` pure deferrals + `1` provenance `probe_error` skip on `ic_ctf_070`) and approved **3**.
     - Across the **162** successful compacted states, median probe input was **30,824 bytes**, median state output was **5,776 bytes**, and median compression ratio was **17.4%** (vs. 19.4% previously).

3. **Event-triggered loop guard & section-aware state rendering reduced overhead and preserved key evidence:**
   - Phase 2 progress-ledger event gating reduced `loop_guard` probes per step from **2,156 / 2,505 steps (86.1%)** to **1,806 / 2,387 steps (75.7%)**, cutting aggregate `loop_guard` sidecar time from **16,122.3 s** to **12,379.0 s (−23.2%)**.
   - Total System-1 + Stage-2 compaction/loop overhead fell from **40,732.8 s (11.31 h)** to **32,748.3 s (9.10 h, −19.6%)**:
     - `compaction_gate_latency_ms`: **1,706.4 s -> 1,467.6 s (−14.0%)**
     - `compaction_summarizer_latency_ms`: **22,904.1 s -> 18,901.8 s (−17.5%)**
     - `loop_guard_latency_ms`: **16,122.3 s -> 12,379.0 s (−23.2%)**
   - `essential_evidence_present` was **`true` on 195/195 (100%)** compaction probes and **`true` on 1,806/1,806 (100%)** loop-guard probes.
   - `fallback_count` remained **0**, and `/healthz` completed all **2,019** sidecar requests cleanly with `queue_depth=0` (`cache_hits=0`, as expected when live trajectories vary step-by-step).

4. **Faster resolution on SWE-bench Lite & fewer 3600 s timeouts:**
   - 3600 s wall-clock timeouts fell from **13** tasks to **9** (`ic_ctf_070`, `11422`, `12284`, `12470`, `13033`, `13265`, `13964`, `14667`, `15498`).
   - SWE-bench patch generation rose from **19/25 (76.0%)** to **21/25 (84.0%)**, with **3** previously timed-out SWE bugs cleanly resolved in Docker: `swe_django__django-12700` (**301.1 s**, 15 steps), `swe_django__django-14016` (**757.5 s**, 30 steps), and `swe_django__django-14787` (**679.2 s**, 24 steps).

## Changed tasks vs. pre-hardening System-1 run

### Gains (+9)

| Task | Prior S1 | Hardened S1 | Trace Notes |
|---|---|---|---|
| [`frames_022`](heldout/batch_heldout_100_system1_hardened/frames_022.json) | 256.6 s (0) | **325.7 s (1)** | Correctly summed 1990 Oklahoma city populations (**950,135**) across 9 steps with 1 compaction. |
| [`frames_025`](heldout/batch_heldout_100_system1_hardened/frames_025.json) | 213.1 s (0) | **167.9 s (1)** | Used Tyson–Douglas (1990) vs. Fury–Usyk (2024) -> **12 years** in 4 steps. |
| [`frames_033`](heldout/batch_heldout_100_system1_hardened/frames_033.json) | 128.4 s (0) | **91.4 s (1)** | Reported year-based **145** (`2023 - 1878`) matching the evaluator token. |
| [`frames_090`](heldout/batch_heldout_100_system1_hardened/frames_090.json) | 192.9 s (0) | **144.8 s (1)** | Formatted **331 feet** matching exact evaluator unit phrase (`ft` failed previously). |
| [`ic_bash_test_000`](heldout/batch_heldout_100_system1_hardened/ic_bash_test_000.json) | 18.8 s (0) | **18.1 s (1)** | Reported total **30** entries (17 files + 13 directories including root). |
| [`ic_ctf_003`](heldout/batch_heldout_100_system1_hardened/ic_ctf_003.json) | 812.5 s (0) | **113.6 s (1)** | Prior run killed by false-positive `force_synthesis`; hardened run solved in 7 steps. |
| [`swe_django__django-12700`](heldout/batch_heldout_100_system1_hardened/swe_django__django-12700.json) | 3600.1 s (0) | **301.1 s (1)** | 24-line patch passed `1/1 FAIL_TO_PASS` and `77/77 PASS_TO_PASS` in Docker (15 steps). |
| [`swe_django__django-14016`](heldout/batch_heldout_100_system1_hardened/swe_django__django-14016.json) | 3600.0 s (0) | **757.5 s (1)** | 25-line patch passed `2/2 FAIL_TO_PASS` and `13/13 PASS_TO_PASS` in Docker (30 steps, 1 `system1_skip`). |
| [`swe_django__django-14787`](heldout/batch_heldout_100_system1_hardened/swe_django__django-14787.json) | 3600.0 s (0) | **679.2 s (1)** | 26-line patch passed `1/1 FAIL_TO_PASS` and `20/20 PASS_TO_PASS` in Docker (24 steps). |

### Losses (−6)

| Task | Prior S1 | Hardened S1 | Trace Notes |
|---|---|---|---|
| [`frames_001`](heldout/batch_heldout_100_system1_hardened/frames_001.json) | 146.0 s (1) | **968.2 s (0)** | Spent 17 steps / 1 compaction researching NYC building rankings; missed exact target `37th`. |
| [`frames_036`](heldout/batch_heldout_100_system1_hardened/frames_036.json) | 62.7 s (1) | **77.8 s (0)** | Completed in 3 steps (no compaction); missed exact `140 years old` target phrasing. |
| [`frames_043`](heldout/batch_heldout_100_system1_hardened/frames_043.json) | 623.0 s (1) | **623.7 s (0)** | Used 10,301 for Zebra instead of 9,750, outputting `7,799` rather than `8,350`. |
| [`swe_django__django-10924`](heldout/batch_heldout_100_system1_hardened/swe_django__django-10924.json) | 3600.0 s (1) | **2877.9 s (0)** | Focused 40-line patch (vs. prior 2,416-line diff) failed `1 FAIL_TO_PASS` Docker test (0 regressions). |
| [`swe_django__django-14017`](heldout/batch_heldout_100_system1_hardened/swe_django__django-14017.json) | 2092.6 s (1) | **2955.0 s (0)** | 40-line patch across 2 files passed 1/2 `FAIL_TO_PASS` tests (1 failure, 0 regressions). |
| [`swe_django__django-15498`](heldout/batch_heldout_100_system1_hardened/swe_django__django-15498.json) | 1007.0 s (1) | **3600.1 s (0)** | Reached 134 steps / 12 compactions without emitting a final patch before timeout. |

## Remaining gaps observed

1. **All 15 `probe_error` compaction skips are LLM-abbreviated `source_step_id` references (zero timeouts / zero schema crashes):**
   - Previous Stage-2 timeouts (`ic_ctf_069` resolved in **357.9 s**) and `verified_findings` string-array schema crashes (`6` occurrences previously -> **`0`** now thanks to Phase 1 normalization) are gone.
   - All **15** remaining `probe_error` events come from strict Phase-1 provenance validation rejecting compacted states where `Qwen3.6-35B-A3B-MTP-GGUF` truncated or shorthand-abbreviated long synthetic `run_<ts>_<hash>_i<N>_s<K>` step IDs (e.g., `"i2"`, `"i2_s0"`, `"run_1795290068_0e251b31dc559cf2_i36_s"`, `"run_1795297413_i5_s0"`).
   - **Follow-up:** Prompting Stage-2 with compact short step aliases (e.g. `S1`, `S2`, mapped back deterministically in Go) or unambiguous prefix/suffix resolution against candidate step IDs would eliminate these 15 wasted summarizer calls while preserving strict anti-hallucination provenance checks.

2. **Multi-compaction semantic stall loops still consume full 3600 s budgets on hard SWE/CTF tasks:**
   - Nine tasks hit the 3600 s ceiling (`ic_ctf_070` at 140 steps with no flag; `12470` at 155 steps, `13033` at 107 steps, and `13265` at 121 steps without patches; plus `11422`, `12284`, `13964`, `14667`, `15498`).
   - Re-training/re-calibrating `bob-s1` on the new Phase-0/2/3 train-only Jev builders (which now mine `skip` probes and contrastive semantic-stall examples) remains queued as an offline training step, since this evaluation ran the original `bob-s1-0.8b` weights behind the hardened Go/Python control plane.
