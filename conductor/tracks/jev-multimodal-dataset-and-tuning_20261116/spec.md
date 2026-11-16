# Specification: Comprehensive Multimodal Dataset, Zero-Shot Retention & Iterative Tuning Loop (`Jev`)

## 1. Objective & Current Baseline (`exp_b7_final`)

Build a comprehensive, leak-free multimodal and cross-domain dataset + evaluation harness for Bob's unified `0.8B` decision and embedding model (`Qwen/Qwen3.5-0.8B-Base` + `jaredpalmer/kev-0.8b` donor), wire the frozen native vision tower (`qwen3_5_vision`), and run a disciplined small-scale training / `cal` 5-fold CV improvement loop until all in-domain, out-of-domain, multilingual, and multimodal gates pass prior to full-scale multi-epoch training.

### Current Verified Baseline (`exp_b7_final`, `seq=1536`, `cal` 5-fold CV)
- **In-domain `compaction` (`n=359`):** Raw ROC-AUC `0.7722` (`swe-bench-lite`: `0.7720` vs `0.7105` zero-shot; `intercode-ctf`: `0.7950` vs `0.6541` zero-shot).
- **In-domain `outcome_prefix` (`n=83`):** Raw ROC-AUC `0.8097`, within-fold `0.8454`, calibrated OOF `0.7696`.
- **In-domain `loop_guard` (`n=256`, `61` groups):** FPR `0.0000` (all-`none` negatives; positive loop recall currently untested).
- **In-domain text `embeddings` (`n=115`, `159` gallery):** R@1 `0.9739` (`128d`) / `0.9826` (`256d`) / `0.9913` (`384d`), R@5 `1.0000`.

### Gaps Addressed by This Track
1. **Zero-Shot Generic Decision Retention:** Fine-tuning on narrow agent trajectories without generic replay/validation risks degrading `kev-0.8b`'s general zero-shot decision accuracy across arbitrary `choice`, `noul`, and `score` questions (and future Bob control-plane questions).
2. **True Positive `loop_guard` Supervision:** Existing `loop_guard` splits contain only `"none"` examples; `"intercept"` and `"force_synthesis"` classes lack positive training/evaluation support.
3. **EN + RU Besedka Chat & Episodic Coverage:** Bob operates in bilingual (English + Russian) Besedka chats and DMs, requiring EN+RU retrieval pairs and conversational/memory decision probes.
4. **Multimodal (`qwen3_5_vision`) Support & Supervision:** `eval/jev/model.py` currently loads `AutoModelForCausalLM` (`Qwen3_5TextModel` only), omitting the pretrained `model.visual` tower already present in `Qwen/Qwen3.5-0.8B-Base` (`Qwen3_5ForConditionalGeneration`, `image_token_id=248056`). Neither image embeddings nor image-conditioned decisions are tested or evaluated yet.

---

## 2. Comprehensive Dataset Architecture (`eval/results/jev_dataset_v2/`)

All splits (`train`, `cal`, `dev`) MUST enforce transitive `task_group` / source-document isolation (`0%` group overlap across splits and `0%` overlap with `eval/tasks/heldout/heldout_100.json` and `generic_decisions_dev`). `dev` remains strictly untouched until final model selection.

### Pillar A: Out-of-Domain Generic Decision Suite (`generic_decisions_{train,cal,dev}.jsonl`)
Designed both as a **validation benchmark** for zero-shot retention and a **replay + teacher-KL stream** (`10–20%` step budget) during training:
1. **Programmatic Policy, Rule & Abstention Suite (inspired by `kev` `decision-v7` / `hard-v1`):**
   - Multi-hop rule evaluation, numerical/date threshold policies, trade-off matrix decisions, and explicit missing-fact abstention / `"none_of_the_above"` minimal pairs (same question once with the true option present and once replaced by `"none_of_the_above"` / `"insufficient_evidence"`).
2. **General NLI, QA & Classification (EN + RU):**
   - Entailment / contradiction / neutral checks, boolean reading comprehension, topic/intent classification, and bilingual (EN/RU) conversational tone / urgency classification drawn from local cached corpora (`russian-telegram-chat-logs`, `gazeta`, public NLI/classification templates) with deterministic or teacher-verified labels.
3. **Extensible Agent & Developer-Tooling Questions (Zero-Shot / Future Bob Controls):**
   - Candidate control questions Bob may ask in future versions:
     - `confirmation_gate`: `"proceed"` vs `"ask_user"` vs `"refuse"` given a proposed mutating action and user policy.
     - `prompt_injection`: `"benign"` vs `"injection_attempt"` on untrusted `web_fetch` / tool output or chat messages.
     - `memory_relevance`: `"irrelevant"` vs `"helpful_context"` vs `"authoritative_fact"` for retrieved RAG chunks.
     - `verification_readiness`: `"ready_to_finish"` vs `"needs_test_execution"` vs `"needs_bug_fix"` on candidate diffs/outputs.

### Pillar B: Balanced 3-Class `loop_guard` Suite (`loop_guard_{train,cal,dev}.jsonl`)
Augment existing real trajectory `"none"` windows (including constructive non-zero `execute` test failures) with group-isolated positive loop trajectories matching `internal/fsm/loop_guard.go`:
- **`"none"`:** Productive progress, novel arguments, or constructive error output that changes state/diagnostics.
- **`"intercept"`:** Repeated identical or canonicalized-equivalent tool call with unchanged observation/error (`strike == 1..2`).
- **`"force_synthesis"`:** Repeated blocked call or cyclic tool-call loop (`A -> B -> A -> B`) exceeding the strike threshold (`strike >= 3`).

### Pillar C: EN + RU Conversational & Episodic Retrieval (`emb_pairs_bilingual_{train,cal,dev}.jsonl`)
- Bilingual (English, Russian, and cross-lingual EN<->RU) anchor / positive / hard-negative state triplets built from Besedka-style multi-turn chat windows, technical Q&A, and summary-to-raw-episode pairs (`internal/memory`).

### Pillar D: Multimodal Image + Text Embeddings & Visual Decisions (`multimodal_{emb_pairs,decisions}_{train,cal,dev}.jsonl`)
Generated deterministically via PIL/matplotlib/HTML-canvas rendering + trajectory artifacts so every image has exact ground-truth metadata alongside natural/complex layouts:
1. **`multimodal_emb_pairs` (Image-to-Text, Image-to-Image, Interleaved State-to-State):**
   - Terminal/log screenshots <-> matching text trace / error summary (vs hard negative trace from same benchmark/language).
   - System/metric charts & status dashboards <-> structured metric summary.
   - Architecture/flowchart cards & UI mockups <-> component specification text.
2. **`multimodal_decisions` (Visual + Interleaved `PointerHead` Questions):**
   - **UI / Status Verification:** Given a rendered UI card, dialog, or build badge (`[State: image + optional text]`), answer `choice`/`noul` questions (e.g., *"What status does the build panel display?"*, *"Is a blocking error modal visible?"*).
   - **Chart / Trend Readout:** Given a plot (`loss curve`, `latency histogram`, `bar chart`), identify trend direction, peak bucket, or threshold violation.
   - **Visual Loop / Stuck State Detection:** Given consecutive visual state snapshots or screenshot + tool call, decide `"progress"` vs `"stuck_ui"`.
   - **Cross-Modal Consistency:** Given text claim + image attachment, decide `"consistent"` vs `"contradictory"` vs `"unreadable"`.

---

## 3. Multimodal Model Architecture (`eval/jev/model.py`)

1. **Full Conditional Generation Loader (`Qwen3_5ForConditionalGeneration`):**
   - Load `Qwen3_5ForConditionalGeneration` from `Qwen/Qwen3.5-0.8B-Base`, retaining `model.visual` (`qwen3_5_vision`, `depth=12`, `hidden_size=768`, `out_hidden_size=1024`, `image_token_id=248056`) **100% frozen** while attaching the 372-tensor `kev-0.8b` LoRA adapter to `model.language_model` (`Qwen3_5TextModel`).
   - Support `include_vision=True` (default when multimodal inputs are used, or lazy/configurable so text-only tests remain fast).
2. **Causal Multimodal State Prefix Construction:**
   - All visual tokens (`<|vision_start|>`, `<|image_pad|>` * `N_patches`, `<|vision_end|>`) MUST be placed strictly inside `[State 1..S]` before `<|state_end|>` (`<|fim_prefix|>`).
   - Construct combined `inputs_embeds` for `[State 1..S]` by replacing `input_ids == image_token_id` positions with `model.visual(pixel_values, grid_thw)` embeddings (and computing 3D mRoPE `position_ids` when required by `Qwen3_5`), while `[Question & Options]` in `S+1..L` remain pure text tokens.
3. **Preserved Invariances:**
   - **State-Prefix Causal Invariance:** `W_emb(H_{<|state_end|>})` for an image+text state is invariant (`max_diff < 1e-5`) to the presence, wording, or option ordering of any suffix question.
   - **Option-Permutation Equivariance & DeltaNet Cache Cloning:** `encode_prefix(state, images=...)` caches `(kv, recurrent_state, conv_state)` at `<|state_end|>` and `question_from_prefix(...)` clones the cache per option/question without cross-branch leakage.

---

## 4. Iterative Small-Scale Training & Evaluation Loop (`train` + `cal` 5-fold CV)

Before running a multi-hour full-scale training job, run fast, bounded small-scale experiments (`30–60` optimizer updates per iteration on `AMD Radeon 780M Graphics`, `--save-every 5`, `--emb-batch-size 3`) evaluated strictly via 5-fold grouped CV on `cal`:

### Multi-Suite Go / No-Go Gates on `cal` (Required Before Full-Scale Training)
1. **Generic Decision Retention (`generic_decisions_cal`):**
   - Raw ROC-AUC / multi-class accuracy within `<= 1.0%` (0.01) of `zero_shot_kev` (or higher), and mean KL divergence to `zero_shot_kev` bounded (`<= 0.15`).
2. **In-Domain Bob Decisions (`compaction_cal`, `outcome_prefix_cal`, `loop_guard_cal`):**
   - `compaction` raw ROC-AUC `>= 0.75` overall AND `>= 0.74` on `swe-bench-lite` (zero regression vs `zero_shot_kev` `0.6968 / 0.7105`).
   - `outcome_prefix` raw ROC-AUC `>= 0.80` and `Cal-OOF >= 0.76`.
   - `loop_guard` 3-class macro-F1 `>= 0.85` on positive+negative `cal` while maintaining `false_positive_rate <= 0.02` on true `"none"` rows.
3. **Text & Bilingual Retrieval (`emb_pairs_cal` + `emb_pairs_bilingual_cal`):**
   - In-domain `emb_pairs` R@1 `>= 0.95` across `128d / 256d / 384d`.
   - Bilingual EN+RU R@1 `>= 0.85` at `128d` and `384d`.
4. **Multimodal Embeddings & Visual Decisions (`multimodal_emb_pairs_cal`, `multimodal_decisions_cal`):**
   - Multimodal retrieval R@1 `>= 0.85` (`128d` and `384d`) and significantly above text-only / random-projection baselines.
   - Visual decisions raw ROC-AUC / accuracy `>= 0.80` and `>= zero_shot_kev` (with frozen `qwen3_5_vision`).
