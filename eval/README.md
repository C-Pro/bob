# Bob Evaluation Suite & Benchmark Harness

The `eval/` subsystem implements **Phase 3: Evaluation Dataset and Deterministic CLI Gatekeeper Harness** from Bob's architecture roadmap. It provides a reproducible, hermetic measurement environment designed to evaluate Bob's autonomous agent capabilities across coding, system navigation, CTF challenge solving, and multi-hop research without human supervision, as well as a large-scale trajectory generation pool for Phase 4 Jev decision model distillation.

For complete specifications of the dataset partitioning, schema, and trajectory statistics, see **[DATASET.md](DATASET.md)**.

---

## Architecture & Principles

1. **Deterministic Sandboxing (`bwrap`):** Every task executes in a temporary, fresh Bubblewrap container with `/workspace` mounted read-write, resource limits, and network isolation (`--net none` for coding/CTF tasks; `--net full` with Tavily search for web research tasks).
2. **Zero Host Database Pollution:** Each evaluation run provisions a dedicated ephemeral SQLite database directory (`/tmp/bob-eval-runs/eval_<task_id>_<timestamp>/host/data`). The host database (`~/.local/share/bob/dm_cli.db`) remains completely untouched.
3. **Containerized SWE-bench Verification:** For software engineering benchmarks, generated `git diff` candidate patches are tested inside isolated Docker containers running the official Django test suite (`swebench.harness.run_evaluation`), measuring exact `FAIL_TO_PASS` and `PASS_TO_PASS` outcomes.
4. **Comprehensive Trajectory Logging:** Full turn-by-turn prompts, reasoning traces, tool inputs/outputs, and compaction checkpoints are serialized into replayable `.json` logs, providing offline dataset generation for Phase 4 decision head training.

---

## Benchmark Domains & Tasks

The evaluation suite draws from standard open benchmarks:

| Benchmark | Tasks in Suite | Domain / Focus | Metric & Grading |
| :--- | :---: | :--- | :--- |
| **DeepMind FRAMES** | 35 | Multi-hop reasoning, live web retrieval, tabular analysis, and temporal QA. | Exact normalized match against gold answers with punctuation/casing tolerance. |
| **InterCode-Bash** | 20 | Linux shell command execution, file tree transformations, and text processing. | Dual evaluation: exact filesystem hash tree snapshot match or gold command equivalence. |
| **InterCode-CTF** | 20 | Binary analysis, reverse engineering, cryptography, and Python exploit scripts. | Regex capture of standard CTF flag format (`picoCTF{...}`). |
| **SWE-bench Lite** | 25 | Real-world bug fixes in the `django/django` repository. | Docker containerized pytest/unittest run via `swebench.harness.run_evaluation`. |
| **Total Frozen Suite** | **100** | Full spectrum evaluation (`heldout_100.json`). | **55.0% End-to-End Success Rate**. |

---

## Directory Structure

```
eval/
├── README.md                      # Evaluation guide, benchmark specs, and reproduction steps
├── DATASET.md                     # Comprehensive dataset documentation, schema, and trajectory stats
├── harness/                       # Evaluation harness execution engine and testbed fixtures
│   ├── runner.py                  # Primary evaluation runner (bwrap sandbox + Docker grading)
│   ├── build_manifest.py          # Manifest generator (splits train/held-out task sets)
│   ├── fetch_tasks.py             # Sourcing script for open benchmark task sets
│   ├── test_grader_controls.py    # Positive & negative control tests for all grading oracles
│   ├── test_oracle_match.py       # Oracle matching validation tests
│   ├── config_template.yaml       # Template configuration for local LLM inference
│   ├── bash_fixtures/             # Setup fixtures and shell scripts for InterCode-Bash
│   └── ctf_assets/                # Challenge binaries, scripts, and disk images for InterCode-CTF
├── tasks/                         # Versioned JSON task manifests (partitioned; excluded from git)
│   ├── heldout/                   # 100-task frozen evaluation suite (SHA256 frozen)
│   │   ├── heldout_100.json       # Primary 100-task held-out evaluation manifest
│   │   ├── timeouts_44.json       # 44-task extended timeout subset
│   │   └── swe_failed_iterations_17.json # 17-task SWE-bench high-iteration subset
│   ├── train/                     # Clean training task pool (0% overlap with heldout)
│   │   ├── train_pool.json        # 417 clean training tasks across all 4 domains
│   │   ├── train_20.json          # 20-task training baseline slice
│   │   └── train_sample_5.json    # 5-task smoke validation set
│   └── smoke_4.json               # Quick 4-task smoke test across all 4 benchmark suites
└── results/                       # Archived evaluation summaries and rollout traces (excluded from git)
    ├── baseline_100_report.md     # Full comprehensive markdown report of the 100-task benchmark
    ├── heldout/                   # Held-out evaluation rollouts
    │   ├── heldout_100/           # Initial baseline rollout (local Qwen3.6-35B-A3B-MTP-GGUF)
    │   ├── timeouts_44/           # 1-hour timeout re-run
    │   └── swe_iterations_17/     # 500-iteration Docker-graded SWE-bench rollout traces
    └── train/                     # Training trajectory rollout batches
        └── batch_train_pool_1794581821/ # 417 rollouts, 19,706 steps, 2,004 compactions (gemini-3.5-flash-lite)
```

---

## Dataset Splits & Trajectory Pool

The dataset provides strict separation between evaluation and training:
- **Held-Out Evaluation Suite (`eval/tasks/heldout/heldout_100.json`):** 100 frozen tasks (35 FRAMES, 20 Bash, 20 CTF, 25 SWE-bench Lite) for zero-leakage evaluation.
- **Training Trajectory Dataset (`eval/tasks/train/train_pool.json`):** 417 clean tasks (65 FRAMES, 198 Bash, 79 CTF, 75 SWE-bench Lite) with **0% overlap** with the held-out set.
- **Recorded Training Rollouts (`eval/results/train/batch_train_pool_1794581821/`):** Full execution traces generated with `gemini-3.5-flash-lite` under `--timeout 3600` and `--max-iterations 500`. Contains **19,706 interaction steps**, **2,004 SelfCompact compaction events**, 417 SQLite database snapshots, and 48 candidate SWE patches.

See **[DATASET.md](DATASET.md)** for complete schema definitions, compaction payloads, and Python analysis examples.

---

## Running Evaluations

### Prerequisites

- **Python 3.10+**
- **Bubblewrap (`bwrap`)** installed on the host:
  ```sh
  sudo apt-get install bubblewrap   # Debian/Ubuntu
  sudo pacman -S bubblewrap         # Arch Linux
  ```
- **Bob binary built:**
  ```sh
  make build-bob
  ```
- **Local LLM endpoint:** By default, `config_template.yaml` connects to an OpenAI-compatible endpoint at `http://127.0.0.1:15432/v1` (e.g. Lemonade or vLLM running `Qwen3.6-35B-A3B-MTP-GGUF`).
- *(Optional for SWE-bench Docker grading)* **Docker** installed and running with permissions to pull SWE-bench evaluation images.

### Quick Smoke Test (4 Tasks)

Run a fast smoke test that exercises one task from each benchmark category:

```bash
python3 eval/harness/runner.py eval/tasks/smoke_4.json --timeout 300
```

### Full 100-Task Frozen Suite

To execute the complete 100-task evaluation manifest:

```bash
python3 eval/harness/runner.py eval/tasks/heldout_100.json --timeout 3600 --max-iterations 500
```

Batch results, summaries, git diff patches, and turn-by-turn trajectory traces are saved to:
`eval/results/batch_<manifest_name>_<timestamp>/`.

### Verifying Grader Oracles

To run the negative and positive control verification suite for all 4 grading mechanisms:

```bash
python3 eval/harness/test_grader_controls.py
```

---

## Latest Evaluation Results

Evaluating local `Qwen3.6-35B-A3B-MTP-GGUF` on the frozen 100-task suite:

| Benchmark | Tasks | Metric | Rate | Notes |
| :--- | :---: | :--- | :---: | :--- |
| **DeepMind FRAMES** | 35 | Gold Match | **62.9%** (22/35) | Tavily search integration enabled fast multi-hop answers (<5m avg). |
| **InterCode-Bash** | 20 | Exact State Match | **55.0%** (11/20) | Filesystem directory and file hash parity. |
| **InterCode-CTF** | 20 | Flag Match | **85.0%** (17/20) | High autonomous reverse-engineering and exploit scripting rate. |
| **SWE-bench Lite** | 25 | Patch Generation | **76.0%** (19/25) | Valid git diff patches against base commit. |
| **SWE-bench Lite** | 25 | Docker Resolution | **20.0%** (5/25) | 100% of official Django `FAIL_TO_PASS` test suites passed with 0 regressions. |
| **Total Suite** | **100** | **Gold Standard** | **55.0%** (55/100) | **69.0%** solution output rate across all domains. |

See [baseline_100_report.md](results/baseline_100_report.md) for the complete task-by-task breakdown, timing distributions, and patch analysis.
