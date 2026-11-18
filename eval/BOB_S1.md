---
model_name: bob-s1-0.8b
development_codename: jev
version: "1.0"
model_license: apache-2.0
code_license: mit
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_revision: dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
adapter_donor: jaredpalmer/kev-0.8b
adapter_donor_revision: bf75a6a8848ea6960ff2ed108d9ed44c2941174f
languages:
  - en
  - ru
pipeline_tags:
  - text-classification
  - feature-extraction
  - multimodal
library_name: transformers
hardware_trained_on: AMD Radeon 780M (gfx1103, ROCm 6.4)
artifacts_path: eval/jev/checkpoints/bob-s1-0.8b/
---

# `bob-s1-0.8b` Model Card & Reproducible Technical Report

**Unified 0.8B Multimodal System-1 Decision & Matryoshka Embedding Model for Autonomous Agent Control**

*(Release name: `bob-s1-0.8b`; training codename: `Jev`; final checkpoint: `exp_b_v2_4k_emb_onlymm`)*

---

## Abstract

`bob-s1-0.8b` is a compact (`853.90M` merged inference parameters; `504.8 MiB` `Q4_K_M` text GGUF + `197.7 MiB` `mmproj` BF16) System-1 control-plane and state-representation model designed for the [Bob autonomous agent](../README.md). Operating alongside a larger System-2 reasoning LLM (such as `Qwen3.6-35B-A3B`), `bob-s1-0.8b` is designed to replace repeated generative control-plane LLM calls and separate embedding APIs with a **single causal prefill pass** that produces two outputs from a shared state prefix:

1. **Matryoshka state embeddings** (`128d`, `256d`, `384d` independently L2-normalized vectors via `W_emb`) for episodic RAG and semantic deduplication across English, Russian, cross-lingual EN↔RU, and image+text states.
2. **Typed probabilistic decisions** (`PointerHead`) over variable-cardinality candidate options for context compaction gating, prefix outcome estimation, 3-class tool-loop guarding, and general agent/policy/visual classification—with `(KV, conv, recurrent)` prefix cache cloning across sibling questions (`158.0 ms` per additional question on GPU after a `2,467.6 ms` 4k state prefill).

Selected strictly via 5-fold grouped cross-validation and held-out suites on `cal` (`cal_cv_v2_4k_u48_emb`) and then evaluated once on the locked `dev` split (`dev_final_v2_4k`; prior V1 `1024`-token `dev` runs are disclosed in §5.1), `bob-s1-0.8b` preserves donor retention on held-out `generic_decisions` groups (`0.9091`, KL = `0.0309`), improves **`loop_guard` 3-class macro-F1 from `0.4758` → `1.0000`** (`27` synthetic `dev` rows across `3` split groups) while reducing V1 `"none"` false-positive interceptions from `28.91%` → `0.39%` (`1/256`), lifts **`compaction` ROC-AUC (`0.7415` → `0.7714`)** and pooled **`outcome_prefix` ROC-AUC (`0.5777` → `0.7941`)**, raises **in-domain text retrieval R@1 from `0.6535` → `0.9685` (`128d`)** vs the untrained donor projection, raises **multimodal retrieval R@1 from `0.4167` → `0.9167` / `1.0000` (`128d` / `384d`)**, and raises **multimodal visual decision accuracy (`0.8333` → `0.9333`)**. End-to-end integration into Go runtime and evaluation on `eval/tasks/heldout/heldout_100.json` remain follow-on work (§8).

---

## 1. Model Identity, Lineage & Artifacts

### 1.1 Naming & Compatibility Scope
During Phases 4 and 4b, this architecture and dataset pipeline used the working codename **`Jev`** (*Joint Embedding & Decision*) inside [`eval/jev/`](jev/model.py) and generated data splits under `eval/results/jev_dataset/`. For release and System-1 integration alongside Bob's System-2 models, the model is named **`bob-s1`** (`bob-s1-0.8b`). Existing source files (`eval/jev/*.py`, `eval/harness/build_jev_*.py`) and gitignored dataset paths (`eval/results/jev_dataset/`) keep their historical names so existing run logs and JSON reports remain valid without broken references.

### 1.2 Checkpoint Lineage
1. **Base Model:** [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (`Qwen3_5ForConditionalGeneration`, pinned snapshot revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`).
2. **Donor Adapter:** [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev), pinned local snapshot revision `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`; `372`-tensor PEFT LoRA, $r=16, \alpha=32$, + `PointerHead(1024, 256)`, Apache-2.0). Note that downstream `bob-s1-0.8b` adds a freshly trained `W_emb` (`Linear(1024, 384, bias=False)`) whereas `zero_shot_kev` evaluates `PointerHead` with an **untrained random `W_emb` projection**.
3. **V1/V2 Pilot Progression (`max_seq_len = 1536`):** `exp_b7_final` → `exp_b_v2_iter1` → `exp_b_v2_iter2` snapshot `u0005` (`exp_b_v2_iter2_s5`, `40` micro-steps / `5` updates) → `exp_b_v2_iter2_s5_emb` (`240` `emb_only` updates on `W_emb`).
4. **Stage 1 Joint 4k LoRA (`exp_b_v2_4k`):** `max_seq_len = 4096`, `80` planned updates (`640` micro-steps, `grad_accum=8`) across all 8 V1+V2 task families with frozen-donor KL anchoring; snapshot **`u0048`** locked via `cal` 5-fold CV.
5. **Stage 2 4k Multimodal `W_emb` Refit (`exp_b_v2_4k_emb_onlymm` → `bob-s1-0.8b`):** Frozen backbone/LoRA/`PointerHead`, `240` updates training `W_emb` over `1,160` cached states encoded up to `4096` tokens.

### 1.3 Exported Release Artifacts
Exported under `eval/jev/checkpoints/exp_b_v2_4k_emb_onlymm/` with relative symlink `eval/jev/checkpoints/bob-s1-0.8b -> exp_b_v2_4k_emb_onlymm` (both gitignored; recreate the symlink in a fresh clone via `ln -sfn exp_b_v2_4k_emb_onlymm eval/jev/checkpoints/bob-s1-0.8b`):

| File | Exact Size (Bytes / Binary) | SHA-256 | Contents |
| :--- | ---: | :--- | :--- |
| `model.pt` | `47,072,335` (`44.9 MiB`) | `a0cfc5bb5fd214f76acdb2774cc99e2be35f4432c5cc85547ae9b410b731397c` | PEFT LoRA (`372` tensors), `PointerHead`, `W_emb`, temperature & vision preprocess metadata |
| `bob-s1-0.8b-bf16.gguf` | `1,516,744,096` (`1,446.5 MiB`) | `8b3b059f49ad518af373088545a84fa367a9d6b4f05ea45cf21f60cd7806e8f4` | Merged LoRA + Qwen3.5-0.8B text backbone (`BF16`) |
| `bob-s1-0.8b-q4_k_m.gguf` | `529,297,056` (`504.8 MiB`) | `9e61c92e281e658f0d730d79ace41647821655c28ee1a539f57bf283ad442fc0` | `Q4_K_M` text backbone quantized with `train`-split `imatrix` (`16` chunks × `512` tokens) |
| `bob-s1-0.8b-mmproj-bf16.gguf` | `207,345,856` (`197.7 MiB`) | `a06b03d32f9acea089088ae5c08c43dd4cc3f685c95e783ae4330db10e3782b0` | Frozen Qwen3.5 vision tower + projector (`BF16`; carries `mean`/`std` — pair with `bob-s1-0.8b-preprocess.json` for `min_pixels`/`max_pixels`) |
| `bob-s1-0.8b-preprocess.json` | sidecar JSON | *(generated by `export_gguf.py`)* | Explicit vision resize limits (`min_pixels=16384`, `max_pixels=98304`, `mean`/`std=(0.5, 0.5, 0.5)`), since `llama.cpp` `mmproj` omits HF min/max pixel bounds |
| `bob-s1-0.8b-heads.pt` | `3,674,863` (`3.5 MiB`) | `3a0e2bff36d16281c0b4c9b55c2c97b33e1738317de8293616ef502f2d80ca70` | Standalone FP32 `W_emb` + `PointerHead` state dicts for decoupled `llama-server` scoring *(note: original exploratory export used internal PyTorch archive prefix `jev_heads/`, so re-exporting via `export_gguf.py` yields identical tensors with a new zip SHA-256)* |

### 1.4 Licensing
Repository source code under `eval/jev/` and `eval/harness/` is licensed under MIT ([`../LICENSE`](../LICENSE)). Derived model weights (`bob-s1-0.8b`), LoRA adapters, and projection heads (`model.pt`, `*.gguf`, `*-heads.pt`) inherit **Apache-2.0** from `Qwen/Qwen3.5-0.8B-Base` and `jaredpalmer/kev-0.8b`; distribute exported checkpoints with upstream Apache-2.0 attribution and notices.

---

## 2. Architecture

Implemented in [`eval/jev/model.py`](jev/model.py) (`DualHeadDecisionEmbedder`).

### 2.1 Hybrid Gated-DeltaNet + Vision Backbone
* **Text Backbone (`qwen3_5_text`):** `24` transformer layers (`hidden_size = 1024`, vocabulary `248,320`, `max_position_embeddings = 262,144`, `full_attention_interval = 4`):
  * **18 Gated DeltaNet linear-attention layers:** `16` value heads / key-value groups, inner size `2048`, head state dim `128`, causal `conv1d` kernel width `4`.
  * **6 Full-Attention layers:** `8` query heads, `2` KV heads (GQA), head dim `256`, RoPE $\theta = 10^7$, `partial_rotary_factor = 0.25`, HF `mrope_section: [11, 11, 10]` *(serialized as `[11, 11, 10, 0]` in GGUF metadata padding)*.
* **Frozen Vision Tower (`qwen3_5_vision`):** `12` ViT blocks (`hidden_size = 768`, spatial merge `2×2`, output hidden size `1024`, `image_token_id = 248056`). Preprocessed deterministically via `Qwen2VLImageProcessorPil` (`min_pixels = 64 × 16 × 16 = 16,384`, `max_pixels = 384 × 16 × 16 = 98,304`, `mean = std = (0.5, 0.5, 0.5)`). `model.visual` remains `requires_grad = False` and `.eval()` throughout all training stages.

### 2.2 Parameter Ledger (Merged Inference vs Training Instance)

| Component | Merged Params | Share | BF16 Memory (`MiB`) | Stage 1 Trainable | Stage 2 Trainable |
| :--- | ---: | ---: | ---: | :---: | :---: |
| `ffn_mlp` | `264.24M` | `30.9%` | `504.0 MiB` | LoRA (`r=16, α=32`) | Frozen |
| `token_embd` (`248,320 × 1024`) | `254.28M` | `29.8%` | `485.0 MiB` | Frozen | Frozen |
| `deltanet_ssm` | `189.78M` | `22.2%` | `362.0 MiB` | LoRA | Frozen |
| `visual` (`qwen3_5_vision`) | `100.59M` | `11.8%` | `191.9 MiB` | Frozen (`eval`) | Frozen (`eval`) |
| `full_attn` (6 layers) | `44.04M` | `5.2%` | `84.0 MiB` | LoRA | Frozen |
| `bob-s1` Heads (`W_emb` + `PointerHead`) | `0.92M` | `0.1%` | `1.8 MiB` | Yes (`918,016` params, FP32) | `W_emb` only (`393,216` params) |
| `norms_other` | `0.05M` | `0.0%` | `0.1 MiB` | Frozen | Frozen |
| **Merged Inference Total** | **`853.90M`** | **`100%`** | **`1,628.7 MiB`** | — | — |

In the unmerged PEFT training instance, the `372` trainable LoRA tensors add ~`10.823M` adapter parameters alongside `918,016` head parameters (`11,740,672` trainable parameters in Stage 1, or `393,216` in Stage 2).

### 2.3 Sequence Layout, State Invariance & DeltaNet Cache Cloning

Because Gated DeltaNet layers mutate recurrent and 1D-convolutional state along the token stream, block-diagonal attention masks inside a single packed sequence cannot isolate sibling question branches. `bob-s1` processes each `(state, question)` pair as an independent row (or explicit prefix cache clone):

```
┌─────────────────────────── Prefix [0 .. S-1] ───────────────────────────┐┌──────── Suffix [S .. L-1] ────────┐
│ <|fim_prefix|> [<|vision_start|> <|image_pad|>×N <|vision_end|>]* STATE ││ <|fim_middle|> Q (<|box_start|> opt_k: desc <|box_end|>)+ <|fim_suffix|> │
└───────────────────────────────────────────────────────────────────────┬─┘└───────────────────────────┬─────────┴──────────────┬────────────────────────┘
                                                                        │                              │                        │
                                                             H_{S-1} -> W_emb             K_k = W_k H_{end_k} + b_k    Q = W_q H_{decide} + b_q
                                                          [128d, 256d, 384d] L2                         logits_k = (Q · K_k) / sqrt(256)
```

* **Delimiter Sanitization:** Any raw occurrence of `<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`, `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, or `<|video_pad|>` inside untrusted state or question text is rewritten (`<|` → `‹`, `|>` → `›`) in `_text_ids`.
* **Head+Tail State Truncation:** When `len(state_ids) > text_budget - 1`, `prepare_state` retains equal head and tail spans separated by `\n[... K tokens omitted ...]\n` with an approximate omitted-token count `K` (determined via up to `4` token-count iterations stopping when marker token length stabilizes; omitted when `keep < len(marker) + 2`). Notice that `format_prompt` reserves `len(suffix)` before truncating `state`, whereas `embed_state(state, budget)` uses the full `budget`; thus, for **fixed prepared prefix tokens** (or whenever inputs fit without truncation), suffix content cannot influence `W_emb` (`max_diff < 1e-5` in [`test_jev_model.py`](jev/test_jev_model.py)). For long truncated states shared across multiple questions, encode once with a shared `suffix_reserve` via `encode_prefix` and branch using `question_from_prefix`.
* **Matryoshka State Embedding Head (`W_emb`):** Projects `H[offset - 1]` (last token of `prefix_ids`) via `Linear(1024, 384, bias=False)` in FP32 and normalizes each nested slice independently via `F.normalize` (`eps=1e-12`):
  $$e_d(x) = \frac{(W_{\text{emb}} h)_{1:d}}{\max\!\big(\|(W_{\text{emb}} h)_{1:d}\|_2,\; 10^{-12}\big)}, \quad d \in \{128, 256, 384\}$$
* **Pointer Decision Head (`PointerHead`):** Two bias-enabled linear projections (`Linear(1024, 256, bias=True)`) without LayerNorm (matching `kev-0.8b`):
  $$z_k = \frac{(W_q h_{\text{decide}} + b_q)^\top (W_k h_{\text{opt\_end}_k} + b_k)}{\sqrt{256}}$$
* **Cache-Cloned Multi-Question Inference:** `encode_prefix` computes the prefix once, extracts `W_emb`, captures `(kv_cache, conv_state, recurrent_state)` plus MRoPE `next_pos`, and `question_from_prefix(..., isolate=True)` deep-clones the mutable hybrid cache per question (`158.0 ms` per additional question vs `2,467.6 ms` full 4k prefix forward). Conceptually, questions follow `noul` (boolean), `choice` (multi-option), and `score` conventions; the raw PyTorch/C++ heads emit logits and embeddings (`calibration` applies task-specific `(T, b)` scaling externally).

---

## 3. Training & Evaluation Datasets (`eval/results/jev_dataset/`)

All splits enforce **transitive `task_group` / `split_group` isolation** (`0%` group overlap across `train`/`cal`/`dev` and `0%` overlap with [`eval/tasks/heldout/heldout_100.json`](tasks/heldout/heldout_100.json), `heldout_source_sha256 = 7d16e98371d93adbade3a6e27c532c8aeddb6c2e6b25732460ff2edfbe35984f`, verified in `test_build_jev_dataset.py` and `test_build_jev_generic_and_multimodal.py`).

### 3.1 V1 Agent Trajectory Dataset ([`build_jev_dataset.py`](harness/build_jev_dataset.py))
Extracted from `eval/results/train/batch_train_pool_1794581821/` (`417` source rollouts generated by `gemini-3.5-flash-lite` at `--max-iterations 500 --timeout 3600` across DeepMind FRAMES, InterCode-Bash, InterCode-CTF, and SWE-bench Lite). Running with `--quarantine-heldout` quarantines `2` residual held-out overlapping tasks (`ic_ctf_005`, `ic_ctf_068`), retaining **`415` clean rollouts (`19,687` source steps; `289` train / `61` cal / `65` dev tasks)**:
* **`compaction` (`2,317` rows: `1,563` train / `359` cal / `395` dev):** Time-$t$ boundary-reconstructed context states matching [`internal/fsm/compaction.go`](../internal/fsm/compaction.go) (omitting future steps, post-decision `state_json`, and terminal grades). Labels: `2,003` compact (`1`) vs `314` skip (`0`), excluding `10` `probe_error` timeouts/errors. *Benchmark skew:* SWE-bench contributes `1,669 / 2,003` clean positive compacts and `276` skips; CTF contributes `317` pos / `38` neg; FRAMES (`9` pos / `0` neg) and Bash (`8` pos / `0` neg) have zero skips. `Always-Compact` scores `84.30%` raw accuracy on `dev`.
* **`outcome_prefix` (`551` rows: `373` train / `83` cal / `95` dev):** Prefix snapshots from `321` cleanly graded clean tasks (`119` passed / `202` failed), excluding `46` Bash `oracle_error` and `48` SWE missing-`report.json` infra failures (`103` positive / `448` negative prefix rows).
* **`emb_pairs` (`769` rows: `527` train / `115` cal / `127` dev):** Contrastive agent-state prefix triplets (lexically mined negatives across distinct `task_group`s).
* **`loop_guard` V1 (`19,688` rows: `13,227` train / `3,169` cal / `3,292` dev):** Observed agent tool-loop steps (all labeled `"none"` in the original rollouts).

### 3.2 V2 Extension Dataset ([`build_jev_generic_and_multimodal.py`](harness/build_jev_generic_and_multimodal.py))
Adds `252` isolated `split_group`s (`0` overlap) across `en` and `ru`:
* **`generic_decisions` (`300` rows: `197` train / `59` cal / `44` dev; `240` EN / `60` RU):** Nine subfamilies: `policy_abstention` (`112`, `"none_of_the_above"` / `"insufficient_evidence"` minimal pairs), `policy_rules` (`56`), `nli` (`60`), `boolean_qa` (`24`), `chat_urgency` (`12`), `confirmation_gate` (`9`), `memory_relevance` (`9`), `prompt_injection` (`9`), `verification_readiness` (`9`).
* **`loop_guard_synthetic` (`216` rows: `153` train / `36` cal / `27` dev):** Balanced 3-class trajectories (`72` `none`, `72` `intercept`, `72` `force_synthesis`) across `9` canonical patterns, merged with V1 into `loop_guard_combined` (`19,904` rows: `13,380` train / `3,205` cal / `3,319` dev).
* **`emb_pairs_bilingual` (`60` rows: `36` train / `16` cal / `8` dev):** EN (`18`), RU (`18`), EN→RU (`12`), RU→EN (`12`) chat-to-summary and summary-to-chat episodic triplets.
* **`multimodal_emb_pairs` (`76` rows: `52` train / `12` cal / `12` dev; `62` EN / `14` RU):** Deterministically rendered PIL assets (terminal logs, metric charts, architecture flowcharts, UI mockups using Noto Sans/Mono fonts) paired with captions and hard visual negatives.
* **`multimodal_decisions` (`192` rows: `132` train / `30` cal / `30` dev; `164` EN / `28` RU):** Visual/interleaved `PointerHead` tasks across `architecture_flow`, `chart_sla`, `chart_trend`, `cross_modal_consistency`, `terminal_outcome`, `ui_modal_check`, `visual_loop`.

Serialized split data fingerprints (`experiment_report.json` `data_sha256`; these fingerprint loaded files rather than independently verifying isolation, which is tested separately by `test_build_jev_*.py`):
* `cal`: `81f741c4f0195f6da1af11db8f0f87ce4dc58c53ac89bd27efc8a8a044f612a3`
* `dev`: `d08b1fded815befbb5598ee3b3823abcec7a09907c1cd6f3010cf2a77c599c95`

---

## 4. Training Pipeline & Optimization

Training executes in [`eval/jev/train_experiment.py`](jev/train_experiment.py) on an **AMD Radeon 780M iGPU (`gfx1103`, ROCm 6.4)** in `bfloat16` (`PointerHead` and `W_emb` in FP32) with `gradient_checkpointing` and **exact unpadded `batch=1` causal forwards**. Training explicitly sets `module.p = 0.0` on all `nn.Dropout` layers (disabling the donor adapter's `lora_dropout = 0.05`) so two-pass `GradCache` gradients and permutation comparisons are deterministic.

### 4.1 Objectives
1. **Decision CE + Frozen-Donor KL Anchoring:**
   $$\mathcal{L}_{\text{dec}}(f, x, y) = w_{\text{ce}}^{(f)}\, \mathrm{CE}\!\big(z(x), y\big) + w_{\text{kl}}^{(f)}\, D_{\mathrm{KL}}\!\big(\sigma(z_{\text{ref}}(x)) \parallel \sigma(z(x))\big)$$
   with `reference_logits.pt` precomputed from the frozen zero-shot donor (`zero_shot_kev`; `compaction_ref_checkpoint=None`, `outcome_ref_checkpoint=None`).
2. **Matryoshka InfoNCE (`temperature = 0.07`) with `GradCache`:**
   For each retrieval batch of $B$ triplets `(a_i, p_i, n_i)`, candidate matrix $C_d = [e_d(p_1), \dots, e_d(p_B), e_d(n_1), \dots, e_d(n_B)]^\top \in \mathbb{R}^{2B \times d}$ includes in-batch positives and explicit hard negatives, with same-`task_group` entries masked (`-inf`) except the designated positive at index $i$. Per-row `task_balanced_weight` values are mean-normalized within each retrieval family (`mean = 1.0`):
   $$\mathcal{L}_{\text{emb}} = \frac{1}{3} \sum_{d \in \{128, 256, 384\}} \frac{1}{B} \sum_{i=1}^{B} \tilde{w}_i \cdot \mathrm{CE}\!\left(\frac{e_d(a_i)^\top C_d^\top}{0.07},\; i\right)$$
   In Stage 1, two-pass **`GradCache`** (`emb_batch_size = 3`) produces full-batch contrastive gradients at batch-1 activation memory (`peak VRAM = 4,370.8 MiB` @ `4096` tokens).

### 4.2 Two-Stage Training Configuration (`bob-s1-0.8b`)

| Hyperparameter | Stage 1 (`exp_b_v2_4k` → `u0048`) | Stage 2 (`exp_b_v2_4k_emb_onlymm`) |
| :--- | :--- | :--- |
| Init checkpoint | `eval/jev/checkpoints/exp_b_v2_iter2_s5_emb/model.pt` | `eval/jev/checkpoints/exp_b_v2_4k/snapshots/u0048/model.pt` |
| Mode & `max_seq_len` | `exp_b` (LoRA + `PointerHead` + `W_emb`), `4096` | `emb_only` (`W_emb` over `1,160` cached states ≤ `4096` tok), `4096` |
| Budget / Selection | `80` updates planned (`640` micro-steps, `grad_accum=8`); **`u0048` selected on `cal` CV** | `240` updates (`80` `emb_pairs`, `80` `bilingual`, `80` `mm_emb_pairs`) |
| Optimizer & LR schedule | AdamW (`wd=0.01`), cosine + `6`-step warmup; `lora=1e-5`, `pointer=8e-6`, `emb=2e-4` | AdamW (`wd=0.01`), cosine + `15`-step warmup; `emb=4e-4` |
| Task sampling mix (`4:2:4:2:2:1:1:2`) | `compaction` `142`, `outcome` `70`, `loop_guard` `143`, `generic` `72`, `mm_dec` `71`, `emb` `35`, `bi_emb` `35`, `mm_emb` `72` (`balanced_sampling`: 75% SWE / 25% CTF, label-balanced cycling) | Uniform `80` / `80` / `80` (`emb_batch_size = 6`) |
| Family loss weights `(CE, KL)` | `compaction`: `(1.0, 0.50)`, `outcome`: `(1.0, 0.25)`, `loop`: `(1.25, 0.05)`, `generic`: `(0.25, 1.00)`, `mm_dec`: `(1.00, 0.10)` | Pure `W_emb` Matryoshka InfoNCE (`τ = 0.07`) |
| Trainable params / Peak VRAM | `11,740,672` (`1.37%`) / `4,370.8 MiB` | `393,216` (`0.046%`) / `3,150.8 MiB` |
| Wall time (`summary.json`) | `4,703.5 s` (excludes prior ref-cache build) | `1,797.7 s` (includes `1,160`-state hidden-cache extraction) |

---

## 5. Evaluation Pipeline & Quantitative Results

Implemented in [`eval/jev/eval_experiment.py`](jev/eval_experiment.py).

### 5.1 Split Discipline & Prior `dev` Evaluations
* **V2 4k model selection:** Candidates (`exp_b_v2_4k` snapshots and Stage 2 `W_emb` refits) were compared exclusively on `cal` (`cal_cv_v2_*`) against 4 pre-registered Go/No-Go gates; `bob-s1-0.8b` (`exp_b_v2_4k_emb_onlymm`) was locked on `cal` (`cal_cv_v2_4k_u48_emb`) and then evaluated once in `dev_final_v2_4k`.
* **Disclosure of earlier V1 `dev` runs:** Earlier `1024`-token V1 baselines (`exp_a`, `exp_b` in `eval/results/jev_dataset/experiment_report.json`; `exp_a2`, `exp_b2` in `eval/results/jev_dataset/exp2/experiment_report.json`) ran on the V1 `dev` decision subsets (`395` compaction / `95` outcome rows) before the V2 `4k` extension track was created. Because `dev` decision rows share that underlying `395`/`95` V1 decision pool, [`eval/tasks/heldout/heldout_100.json`](tasks/heldout/heldout_100.json) is excluded from `bob-s1` training and reserved for the forthcoming end-to-end controller evaluation.

### 5.2 Calibration & Metric Definitions
* **Two-Parameter Affine Calibration (fit on `cal` only):** For binary logits $(z_0, z_1)$, $p_1 = \sigma\big((z_1 - z_0 + b)/T\big)$, minimizing NLL over `cal` (`bob-s1-0.8b`: `compaction` $T=1.8959, b=4.25$, balanced threshold `0.8965`, `5%`-risk threshold `0.8394`; `outcome_prefix` $T=3.1828, b=-3.8375$, balanced threshold `0.1154`, `5%`-risk threshold `0.8491`).
* **Protocol difference on `cal`:** Binary decision metrics (`compaction`, `outcome_prefix`) use 5 group-disjoint CV folds on `cal`, where `Raw ROC-AUC` is the **pooled raw-margin ROC-AUC** (`bootstrap_ci_95`: 1,000 paired task-group bootstrap draws), alongside within-fold mean AUC (`Within-Fold`) and calibrated OOF AUC (`Cal-OOF`). Retrieval, `generic_decisions`, `loop_guard`, and multimodal suites are evaluated directly on `cal` (not 5-fold CV).
* **Metric conventions:** `Nat` denotes calibrated `argmax`; `Brier` sums squared errors across all classes ($\frac{1}{N}\sum_{i,c}(p_{ic}-y_{ic})^2$, `2×` positive-class-only binary Brier); `PR-AUC` uses step-integrated average precision; `Macro-F1` averages F1 across classes; retrieval uses deduplicated gallery candidates where any candidate matching the anchor's `task_group` is a valid positive.

### 5.3 Calibration CV Progression (`cal`, `max_seq_len = 4096` vs pre-V2 `1536`)

| Model | `comp` AUC (`95% CI`) | `out` AUC (`95% CI`) | `gen` Acc (`KL vs ZS`) | `loop` Orig FPR ↓ | `loop` 3-Class F1 | `emb` 128/384 R@1 | `bi-emb` R@1 | `mm-emb` 128/384 R@1 | `mm-dec` Acc |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_shot_kev` (`4k`, random `W_emb`) | `0.6784` | `0.5644` | `0.8983` (`n/a`) | `0.3438` | `0.4886` | `0.8174 / 0.8174` | `1.000 / 1.000` | `0.4167 / 0.3333` | `0.8000` |
| `exp_b7_final` (`1536`, pre-V2) | `0.7737` | `0.8088` | `0.8983` (`0.0177`) | `0.0000` | `0.1818` | `0.9739 / 0.9913` | `1.000 / 1.000` | `0.5833 / 0.5000` | `0.8333` |
| **`bob-s1-0.8b` (`4k`)** | **`0.7739`** (`[0.6297, 0.8437]`) | **`0.8144`** (`[0.6509, 0.9317]`) | **`0.8983` (`0.0391`)** | **`0.0000`** | **`1.0000`** | **`0.9826 / 0.9913`** | **`1.000 / 1.000`** | **`0.9167 / 0.9167`** | **`0.9333`** |

**Within-Benchmark Diagnostic (`cal`):** `compaction` performs consistently within-domain (`swe-bench-lite`: `0.7843` on `277` rows; `intercode-ctf`: `0.7793` on `80` rows). By contrast, `outcome_prefix` `cal` pooled AUC (`0.8144`) is strongly aided by between-benchmark pass-rate priors (`intercode-ctf` `n=30` and `swe-bench-lite` `n=12` have `0` positive `cal` rows and negative mean margins `-4.71` and `-6.13`), while **within-domain `outcome_prefix` `cal` AUC is `0.4327` (`frames`, `n=21`) and `0.6042` (`intercode-bash`, `n=20`)**.

### 5.4 Locked V2 `dev` Evaluation (`max_seq_len = 4096`)

#### Agent Control Decisions (`compaction`, `n=395`; `outcome_prefix`, `n=95`)
| Model | Task | ROC-AUC | PR-AUC | Bal Acc (`Cal Thresh`) | Macro-F1 (`Nat`) | Brier ↓ | NLL ↓ | ECE ↓ | `5%`-Risk Cov ↑ (Sel Err) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `always_compact` | `compaction` | `0.5000` | — | `0.5000` | `0.4574` | `0.2739` | `0.4582` | `0.0679` | `0.0000` (—) |
| `zero_shot_kev` | `compaction` | `0.7415` | `0.9403` | `0.6726` | `0.4574` | `0.2554` | `0.4104` | `0.0585` | `0.3747` (`0.0473`) |
| **`bob-s1-0.8b`** | **`compaction`** | **`0.7714`** | **`0.9421`** | **`0.7028`** | **`0.5632`** | **`0.2274`** | **`0.3742`** | **`0.0310`** | **`0.7165`** (`0.0742`) |
| `zero_shot_kev` | `outcome_prefix` | `0.5777` | `0.2190` | `0.5087` | `0.4806` | `0.3248` | `0.4979` | **`0.0466`** | `0.0632` (`0.0000`) |
| **`bob-s1-0.8b`** | **`outcome_prefix`** | **`0.7941`** | **`0.4310`** | **`0.7413`** | **`0.5231`** | **`0.2601`** | **`0.3983`** | `0.0738` | **`0.6632`** (`0.0794`) |

#### Generic Decisions, Loop Guard, Retrieval & Multimodal Suites (`dev`)
| Suite (`n`) | Metric | Lexical / Zero-Shot Donor | **`bob-s1-0.8b`** | Interpretation & Caveats |
| :--- | :--- | ---: | ---: | :--- |
| `generic_decisions` (`44`) | Accuracy / KL vs ZS | `0.9091` / `n/a` | **`0.9091`** / `0.0309` | Retention on held-out generic groups (`197` train generic rows exist); `memory_relevance` & `prompt_injection` are `1/2` (`n=2` each) |
| `loop_guard` V1 (`256` `none`) | False Positive Rate ↓ | `0.2891` | **`0.0039`** (`1/256`) | Reduces false loop interceptions by `74×` |
| `loop_guard` V2 (`27` syn) | 3-Class Macro-F1 / Acc | `0.4758` / `0.5556` | **`1.0000` / `1.0000`** | `F1 = 1.0` across `3` split groups (`none`, `intercept`, `force_synthesis`); Brier `0.0257` |
| `emb_pairs` (`127`, gal `174`) | R@1 (`128d / 256d / 384d`) | ZS: `0.6535 / 0.6614 / 0.6614` (`lex`: `0.9134`) | **`0.9685 / 0.9764 / 0.9685`** | Outperforms lexical baseline (`0.9134`) and untrained random `W_emb`; R@5 `1.000` (`256d`/`384d`); `384d` uniformity **`-3.4263`** (vs `-0.2461`) |
| `emb_pairs_bilingual` (`8`, gal `12`) | R@1 (`128d / 384d`) | ZS: `1.0000` (`lex`: `1.0000`) | **`1.0000 / 1.0000`** | Note lexical Jaccard already scores `1.0` on this small split; `bob-s1` improves `384d` uniformity (`-0.8141` → `-2.5812`), while mean squared positive-pair distance (`alignment`) increases (`0.3870` → `0.5518`) |
| `multimodal_emb_pairs` (`12`, gal `20`) | R@1 (`128d / 256d / 384d`) | ZS: `0.4167` (`lex`: `0.5000`) | **`0.9167 / 1.0000 / 1.0000`** | `MRR = 1.0000` @ `256d`/`384d`; `384d` uniformity **`-3.2932`** (vs `-0.7607`) |
| `multimodal_decisions` (`30`) | Overall Accuracy | ZS: `0.8333` | **`0.9333`** | `architecture_flow`: `0.3333` → **`0.8333`**; `1.0` on `chart_trend`, `cross_modal_consistency`, `terminal_outcome`, `ui_modal_check`, `visual_loop` |

#### Latency & Memory (`ROCm bf16`, Radeon 780M, `max_seq_len = 4096`)
* Single-pass decision + embedding: `2,519.5 ms` (`peak VRAM = 3,151.4 MiB`)
* Prefix `encode_prefix`: `2,467.6 ms`
* Additional sibling question via `question_from_prefix(..., isolate=True)`: **`158.0 ms`** (`15.6×` speedup)

---

## 6. Quantization & Runtime Deployment

### 6.1 Exploratory GGUF Quantization Sweep (`cal` Text Subsets via `vulkan/llama-server`)
To characterize quantization sensitivity on `0.8B` hybrid Gated-DeltaNet weights, we ran an exploratory sweep converting `bob-s1-0.8b-bf16.gguf` with `train`-only `imatrix` calibration (`16` chunks × `512` tokens via `rocm-preview/llama-imatrix -ngl 0`; converter checkout `llama.cpp` commit `2943210c1eaa3fc9cc4f0ac6f0ae5f2ce2350f98`). GGUF hidden states were extracted via `vulkan/llama-server` (`0.3.0-dev` / `35999d1`, `-c 2560 -b 2560 -ub 2560 -ngl 99`, `--pooling none` for decisions and `--pooling last` for embeddings; note `2560` server context & `max_seq_len=2048` vs `4096` in PyTorch §5) and scored externally by `bob-s1-0.8b-heads.pt` on stratified `cal` subsets (`compaction`: `16` neg + `24` pos; `outcome_prefix`: `48`; `generic_decisions`: `59`; `loop_guard` V1: `20` distributed rows; `loop_guard` V2: `36`; `emb_pairs`: `115`; `emb_pairs_bilingual`: `16`). Because both context limit (`2048` vs `4096`) and subset sampling differ from §5.3 (`bf16` GGUF `comp` AUC `0.7266` vs full `cal` `0.7739`), treat these numbers as exploratory tradeoffs rather than full-suite benchmarks (`eval/jev/export_gguf.py` codifies the `UD_*` recipes below for reproducible export).

Because `token_embd` (`248,320 × 1024 = 254.28M` params) occupies `485.0 MiB` (`33.5%` of the BF16 text GGUF), `llama-quantize` keeps it at `Q5_K`/`Q6_K` (`166.7–198.9 MiB`), creating a floor near `336–403 MiB`. Unsloth-style mixed-precision `ud_*` recipes apply custom `--tensor-type` protection to `attn_q/k/v`, `attn_output`, `ssm_out`, `attn_gate`, `ssm_alpha/beta`, and `attn_qkv` while compressing `ffn_mlp`.

| Variant | Size (`MiB`) | `token_embd` (`MiB`) | `comp` AUC | `out` AUC | `gen` Acc | `lg` v1 FPR ↓ | `lg` 3-class F1 | `emb` R@1 (`128d`/`384d`) | `bi` R@1 |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bf16` | `1,446.5 MiB` | `BF16` (`485.0`) | `0.7266` | `0.8875` | `0.8814` | `0.00` | `1.0000` | `0.9826 / 0.9913` | `1.0` |
| `q8_0` | `774.2 MiB` | `Q8_0` (`257.7`) | `0.7344` | `0.8875` | `0.8814` | `0.00` | `1.0000` | `0.9826 / 0.9913` | `1.0` |
| `q6_k` | `600.6 MiB` | `Q6_K` (`198.9`) | `0.7109` | `0.8875` | `0.8814` | `0.00` | `1.0000` | `0.9913 / 0.9913` | `1.0` |
| `ud_iq4_xl` | `520.8 MiB` | `Q6_K` (`198.9`) | `0.7266` | `0.8750` | `0.8814` | `0.00` | **`1.0000`** | `0.9826 / 0.9913` | `1.0` |
| **`q4_k_m`** | **`504.8 MiB`** | `Q6_K` (`198.9`) | **`0.7344`** | **`0.8875`** | **`0.8983`** | **`0.00`** | **`0.9444`** | `0.9739 / 0.9739` | `1.0` |
| `iq4_xs` | `480.7 MiB` | `Q6_K` (`198.9`) | `0.7109` | `0.8938` | `0.8983` | `0.00` | `0.9153` | `0.9739 / 0.9739` | `1.0` |
| `q3_k_m` | `444.6 MiB` | `Q6_K` (`198.9`) | `0.7031` | `0.8281` | `0.8305` | `0.00` | `0.7454` | `0.9913 / 0.9913` | `1.0` |
| `ud_iq3_xxs` | `388.9 MiB` | `Q4_K` (`136.4`) | `0.6042` | `0.8187` | `0.8814` | `0.00` | `0.7811` | `0.9739 / 0.9913` | `1.0` |
| `iq3_xxs` | `379.6 MiB` | `Q5_K` (`166.7`) | `0.5990` | `0.8000` | `0.8475` | `0.05` | `0.3977` | `0.9652 / 0.9826` | `1.0` |
| `q2_k` | `402.8 MiB` | `Q6_K` (`198.9`) | `0.7031` | `0.8313` | `0.7966` | `0.00` | `0.4487` | `0.9739 / 0.9826` | `1.0` |
| `iq2_xxs` | `336.0 MiB` | `Q5_K` (`166.7`) | `0.5833` | `0.8031` | `0.4915` | `1.00` | `0.2955` | `0.9304 / 0.9478` | `1.0` |
| `ud_iq2_tiny` | `308.1 MiB` | `Q3_K` (`104.2`) | `0.6536` | `0.8438` | `0.6780` | `0.00` | `0.3923` | `0.9739 / 0.9652` | `1.0` |

* **Takeaway:** `Q4_K_M` (`504.8 MiB`) / `ud_iq4_xl` (`520.8 MiB`) / `IQ4_XS` (`480.7 MiB`) deliver ~3× compression with near-BF16 fidelity. Decisions break down below ~`480 MiB` (`<4-bit`), whereas text retrieval holds `>= 0.9652` R@1 down to `ud_iq2_tiny` (`308.1 MiB`) and drops notably only on unprotected `iq2_xxs` (`0.9304 / 0.9478`).

### 6.2 Runtime Serving Compatibility (`hidden_states` + external `bob-s1-0.8b-heads.pt`)

| Runtime | Text Embeddings (`W_emb`) | 1-Pass Decisions (`PointerHead`) | Multimodal `/embeddings` (`mmproj`) | Observed Constraint |
| :--- | :---: | :---: | :---: | :--- |
| **PyTorch (`eval/jev/model.py`)** | Yes | Yes (`encode_prefix` + clone) | Yes (`qwen3_5_vision`) | Only verified all-in-one multimodal path |
| **Direct `llama-server` (`POST /embeddings`, `0.3.0-dev` / `35999d1`)** | Yes (`cos = 0.99996`) | Yes (`--pooling none --embd-normalize -1`, `max_diff = 0.00663`) | No (`500` on `mtmd_tokenize`) | Requires `-b N -ub N` ≥ prompt length and `-c > max_len` |
| **`lemond` (`v11.6.0`)** | Yes (`--pooling last --embd-normalize -1`) | No (`400`: `none` not OAI-compatible) | No (`500`) | OpenAI `/v1/embeddings` rejects `--pooling none` |
| **Ollama (`0.40.16`)** | Yes *only if* `qwen35.pooling_type = 3` patched in GGUF | No (`400`: `none` not OAI-compatible) | Silently ignored (`cos(red, blue) = 1.0`) | Unpatched GGUF returns `501`; wraps `llama-server` |

---

## 7. Reproducibility Guide

### 7.1 Environment & Pinned Upstream Snapshots
* **Hardware:** `AMD Radeon 780M` (`gfx1103`, `ROCm 6.4`, unified memory)
* **Python environment:** `Python 3.12.12`, `torch 2.9.1+rocm6.4`, `transformers 5.13.0`, `peft 0.19.1`, `numpy 2.5.1`, `Pillow 12.3.0`
* **System fonts for deterministic PIL multimodal asset rendering:** `/usr/share/fonts/noto/NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`, `NotoSansMono-Regular.ttf` (missing Noto fonts falls back to PIL default font and changes rendered image SHA-256s).
* **Pinned Hugging Face snapshots (`local_files_only=True` in `model.py`):**
  * `Qwen/Qwen3.5-0.8B-Base`: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
  * `jaredpalmer/kev-0.8b`: `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`

### 7.2 Exact Commands (`pytest`, Dataset Build, Training, `cal` Selection, `dev` Eval, Export)
*(Scoped to reproduction from the locally retained `batch_train_pool_1794581821` rollouts, `heldout_100.json`, and prerequisite Stage-0/pilot checkpoint `eval/jev/checkpoints/exp_b_v2_iter2_s5_emb/model.pt`, which are gitignored research artifacts.)* Run from the repository root with `python` pointing to a virtualenv containing the stack above (and `HSA_OVERRIDE_GFX_VERSION=11.0.2` on `gfx1103` ROCm):

```bash
# 0. Pre-cache pinned upstream snapshots (model.py / build_jev_*.py / export_gguf.py load by revision)
python -c '
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B-Base", revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
snapshot_download("jaredpalmer/kev-0.8b", revision="bf75a6a8848ea6960ff2ed108d9ed44c2941174f")
'

# 1. Unit, invariance, exporter, and dataset-isolation tests
pytest -q \
  eval/jev/test_jev_model.py \
  eval/jev/test_export_gguf.py \
  eval/harness/test_build_jev_dataset.py \
  eval/harness/test_build_jev_generic_and_multimodal.py

# 2. Rebuild V1 + V2 datasets from eval/results/train/batch_train_pool_1794581821/
python eval/harness/build_jev_dataset.py --quarantine-heldout
python eval/harness/build_jev_generic_and_multimodal.py

# 3. Stage 1: 4k multi-task LoRA + dual-head training (starting from exp_b_v2_iter2_s5_emb -> snapshot u0048)
python eval/jev/train_experiment.py \
  --mode exp_b --run-name exp_b_v2_4k \
  --init-checkpoint eval/jev/checkpoints/exp_b_v2_iter2_s5_emb/model.pt \
  --schedule-mode multitask_v2 --include-vision --max-seq-len 4096 --max-steps 640 \
  --grad-accum 8 --balanced-sampling --emb-batch-size 3 --emb-grad-cache \
  --lr 0.00001 --pointer-lr 0.000008 --emb-lr 0.0002 \
  --kl-compaction-weight 0.5 --kl-outcome-weight 0.25 \
  --kl-loop-weight 0.05 --loop-ce-weight 1.25 \
  --kl-generic-weight 1.0 --generic-ce-weight 0.25 \
  --kl-mm-weight 0.1 --mm-ce-weight 1.0 \
  --v2-mix 'compaction:4,outcome_prefix:2,loop_guard:4,generic_decisions:2,multimodal_decisions:2,emb_pairs:1,emb_pairs_bilingual:1,multimodal_emb_pairs:2' \
  --cosine-lr --warmup-steps 6 --save-every 1 --device cuda --dtype bf16 --seed 45

# 4. Stage 2: Cached 4k text + bilingual + multimodal W_emb refit (final bob-s1-0.8b weights)
python eval/jev/train_experiment.py \
  --mode emb_only --run-name exp_b_v2_4k_emb_onlymm \
  --init-checkpoint eval/jev/checkpoints/exp_b_v2_4k/snapshots/u0048/model.pt \
  --include-vision --max-seq-len 4096 --max-steps 240 --emb-batch-size 6 \
  --emb-families emb_pairs,emb_pairs_bilingual,multimodal_emb_pairs \
  --emb-lr 0.0004 --cosine-lr --warmup-steps 15 --device cuda --dtype bf16 --seed 44

# 5. Model selection on cal 5-fold CV (run before dev)
python eval/jev/eval_experiment.py \
  --cal-cv --models zero_shot_kev exp_b_v2_4k_emb_onlymm \
  --checkpoint-root eval/jev/checkpoints \
  --max-seq-len 4096 --include-vision --loop-limit 256 \
  --device cuda --dtype bf16 --seed 42 \
  --out-dir eval/results/jev_dataset/cal_cv_v2_4k_u48_emb

# 6. Final dev evaluation (writes into a fresh --out-dir)
python eval/jev/eval_experiment.py \
  --models zero_shot_kev exp_b_v2_4k_emb_onlymm \
  --checkpoint-root eval/jev/checkpoints \
  --max-seq-len 4096 --include-vision --loop-limit 256 \
  --device cuda --dtype bf16 --seed 42 \
  --out-dir eval/results/jev_dataset/dev_reproduction_v2_4k

# 7. Formalized GGUF Export recipe (BF16, mmproj-BF16, heads, preprocess sidecar, manifest, train-imatrix Q4_K_M)
python eval/jev/export_gguf.py \
  --llama-cpp-dir /path/to/llama.cpp \
  --llama-imatrix /path/to/llama-imatrix \
  --llama-quantize /path/to/llama-quantize \
  --quants Q4_K_M
ln -sfn exp_b_v2_4k_emb_onlymm eval/jev/checkpoints/bob-s1-0.8b
```

### 7.3 Minimal Scoring & Calibration Snippets
```python
import json, urllib.request
import torch
from torch import nn
from transformers import AutoTokenizer
from eval.jev.model import (
    BASE_MODEL, BASE_MODEL_REVISION, DualHeadDecisionEmbedder,
    PointerHead, QUESTIONS, SPECIAL_TOKENS, resolve_cached_snapshot,
)

# Option A: Native PyTorch all-in-one (text + images + prefix cache cloning)
m = DualHeadDecisionEmbedder.from_pretrained(device="cuda", dtype=torch.bfloat16, include_vision=True)
m.load_checkpoint("eval/jev/checkpoints/bob-s1-0.8b/model.pt")
m.eval()
with torch.no_grad():
    prefix = m.encode_prefix("Agent tool trace...", max_seq_len=4096, suffix_reserve=256)
    emb_128 = prefix.embeddings[128]
    logits = m.question_from_prefix(prefix, *QUESTIONS["compaction"])["logits"]
p_compact = torch.sigmoid((logits[1] - logits[0] + 4.25) / 1.8959).item()  # cal affine fit

# Option B: Standalone CPU heads + tokenizer scoring llama-server (--embeddings --pooling none --embd-normalize -1 -c 4096 -b 4096 -ub 4096)
snap = resolve_cached_snapshot(BASE_MODEL, BASE_MODEL_REVISION, local_files_only=True)
tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
heads = torch.load("eval/jev/checkpoints/bob-s1-0.8b/bob-s1-0.8b-heads.pt", map_location="cpu", weights_only=True)
w_emb = nn.Linear(1024, 384, bias=False, dtype=torch.float32); w_emb.load_state_dict(heads["W_emb"])
pointer = PointerHead(1024, head_dim=256); pointer.load_state_dict(heads["pointer"])

state_ids = [tok.convert_tokens_to_ids(SPECIAL_TOKENS[0])] + tok.encode("Agent tool trace...", add_special_tokens=False)
q, opts = QUESTIONS["compaction"]
suffix_ids, ends = [tok.convert_tokens_to_ids(SPECIAL_TOKENS[1])] + tok.encode(q, add_special_tokens=False), []
for name, desc in opts:
    suffix_ids.append(tok.convert_tokens_to_ids(SPECIAL_TOKENS[2]))
    suffix_ids.extend(tok.encode(f"{name}: {desc}", add_special_tokens=False))
    suffix_ids.append(tok.convert_tokens_to_ids(SPECIAL_TOKENS[3]))
    ends.append(len(suffix_ids) - 1)
suffix_ids.append(tok.convert_tokens_to_ids(SPECIAL_TOKENS[4]))
query_end = len(suffix_ids) - 1

req = urllib.request.Request(
    "http://127.0.0.1:8080/embeddings",
    data=json.dumps({"content": state_ids + suffix_ids}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
tok_h = torch.tensor(json.loads(urllib.request.urlopen(req).read().decode())[0]["embedding"], dtype=torch.float32)
off = len(state_ids)
with torch.no_grad():
    emb_128 = torch.nn.functional.normalize(w_emb(tok_h[off - 1])[:128], dim=-1)
    logits = pointer(tok_h[off + query_end], tok_h[[off + i for i in ends]])  # shape [2]
    p_compact_gguf = torch.sigmoid((logits[1] - logits[0] + 4.25) / 1.8959).item()
```

---

## 8. Limitations & Remaining Integration Work
1. **Go Agent Integration & Held-Out End-to-End Test Pending:** `bob-s1-0.8b` is validated offline on `cal` and `dev`; wiring it into Go (`internal/config`, `internal/llm`, `internal/fsm`, `internal/memory`) and measuring task pass rate & latency on [`eval/tasks/heldout/heldout_100.json`](README.md) remain tracked under [`conductor/tracks/jev-multimodal-decision-embedder_20261114/`](../conductor/tracks/jev-multimodal-decision-embedder_20261114/plan.md) Steps 3–6.
2. **`outcome_prefix` Within-Domain Weak Signal:** Pooled `outcome_prefix` AUC (`0.8144` `cal`, `0.7941` `dev`) reflects benchmark-level pass-rate differences as much as intra-benchmark discrimination (`cal` within-domain AUC is `0.4327` on FRAMES and `0.6042` on Bash).
3. **Small Extension Suites & Empirical Calibration:** `generic_decisions` (`n=44`), `emb_pairs_bilingual` (`n=8`, where lexical Jaccard already hits `1.0`), `multimodal_emb_pairs` (`n=12`), `multimodal_decisions` (`n=30`), and `loop_guard` V2 (`n=27` across `3` groups) have small `dev` sample sizes; calibrated `5%`-risk deferral thresholds yielded `0.0742` (`compaction`) and `0.0794` (`outcome_prefix`) selective error on `dev`, confirming they are empirical heuristics rather than distribution-free bounds.
4. **C++ Multimodal Embedding Path:** GGUF deployment via `llama-server` supports text decisions and text embeddings (`--pooling none`), whereas multimodal (`mmproj`) hidden-state extraction currently requires PyTorch or custom `libmtmd`/`libllama` integration.

---

## Appendix A: System-2 Inference Runtime Comparison (`Qwen3.6-35B-A3B-MTP-GGUF`)
To check whether `Ollama 0.40.16` resolves System-2 generation slowdowns relative to `lemond v11.6.0` on the same machine (`AMD Radeon 780M`, Vulkan, `flash_attn` on), we observed single-run throughput on identical cached weights (`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, `temperature=0.0`, `seed=42`, thinking disabled) under each runtime's configured defaults (note differences in context window, `mmproj` attachment, MTP speculative decoding, and `240` vs `256` stopping on the medium prompt):

| Workload | Metric | Ollama `0.40.16` (`8k` ctx, no MTP) | `lemond v11.6.0` (`128k` ctx + `mmproj`, `--spec-draft-n-max 5`) | `lemond` Ratio |
| :--- | :--- | ---: | ---: | ---: |
| Cold Load | Model load | `10.67 s` | `9.75 s` | `1.09×` |
| Short (`25` in → `256` out) | Prefill / Decode | `64.22` / `13.33 tok/s` (`19.61 s` wall) | `64.23` / **`33.77 tok/s`** (`8.00 s` wall) | **`2.53×` decode** |
| Medium (`2,367` in → `256`/`240` out) | Prefill / Decode | `101.28` / `12.76 tok/s` (`43.44 s` wall) | **`318.15`** / **`26.05 tok/s`** (`16.69 s` wall) | **`3.14×` prefill / `2.04×` decode** |

Observed runtime logs show `lemond` activates the GGUF's Multi-Token Prediction speculative head (`97.1%` / `96.6%` draft acceptance) while Ollama logs `spec common_specu: no implementations specified for speculative decoding`. On long prefill, Ollama also logs `slot create_check` snapshots (`62.8 MiB` each); we hypothesize these DeltaNet checkpoint writes (and differing batching defaults) contribute to the prefill gap, though this exploratory comparison did not isolate checkpoints from context-length differences.

---

## 9. References

1. **Base Model & Vision-Language Architecture**
   * Qwen Team. [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`), Hugging Face.
   * Yang, A., et al. (2025). *Qwen3 Technical Report*. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388).
   * Wang, P., et al. (2024). *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution* (M-RoPE & dynamic resolution vision patching). [arXiv:2409.12191](https://arxiv.org/abs/2409.12191).
   * Bai, S., et al. (2025). *Qwen2.5-VL Technical Report*. [arXiv:2502.13923](https://arxiv.org/abs/2502.13923).
2. **Gated DeltaNet Hybrid Sequence Mixing**
   * Yang, S., Kautz, J., & Hatamizadeh, A. (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule*. ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464).
3. **Donor System-1 Decision Model**
   * Palmer, J. (2026). [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) (local snapshot `bf75a6a8848ea6960ff2ed108d9ed44c2941174f`, Apache-2.0) & [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev).
4. **Adaptation, Matryoshka & Contrastive Learning**
   * Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
   * Kusupati, A., et al. (2022). *Matryoshka Representation Learning*. NeurIPS 2022. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147).
   * van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding* (InfoNCE). [arXiv:1807.03748](https://arxiv.org/abs/1807.03748).
   * Wang, T., & Isola, P. (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. ICML 2020. [arXiv:2005.10242](https://arxiv.org/abs/2005.10242).
   * Gao, L., Zhang, Y., Han, J., & Callan, J. (2021). *Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup* (`GradCache`). [arXiv:2101.06983](https://arxiv.org/abs/2101.06983).
5. **Calibration & Selective Classification**
   * Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
   * Geifman, Y., & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. NeurIPS 2017. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500).
6. **Agent Benchmarks & Quantization Tooling**
   * Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770).
   * Yang, J., Prabhakar, A., Narasimhan, K., & Yao, S. (2023). *InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback*. NeurIPS 2023. [arXiv:2306.14898](https://arxiv.org/abs/2306.14898).
   * Krishna, S., et al. (2024). *Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation* (FRAMES). [arXiv:2409.12941](https://arxiv.org/abs/2409.12941).
   * Gerganov, G., et al. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (`convert_hf_to_gguf.py`, `llama-imatrix`, `llama-quantize`, `llama-server`); Unsloth AI dynamic per-tensor GGUF methodology ([docs.unsloth.ai/basics/unsloth-dynamic-ggufs](https://docs.unsloth.ai/basics/unsloth-dynamic-ggufs)).
