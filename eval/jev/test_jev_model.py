"""Tiny hybrid Qwen invariances plus an offline real-adapter integration check."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (BASE_MODEL, KEV_ADAPTER, SPECIAL_TOKENS, SLICE_DIMS,
                   DualHeadDecisionEmbedder, clone_prefix_cache, synthetic_model)
from train_experiment import (class_balanced_weights, iter_scheduled_steps,
                              matryoshka_infonce, option_order,
                              per_class_unit_weights, smoke_rows)
from eval_experiment import (affine_scale, balanced_decision_threshold,
                             calibrated_evaluation, classification_metrics,
                             probabilities, retrieval_metrics, risk_threshold,
                             temperature_scale)


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    return synthetic_model(seed=17)


STATE = "Inspect source. Tests pass. Verified result. Next fix is known."
QUESTION = "What action follows?"
OPTIONS = [("skip", "Keep raw evidence."), ("compact", "Summarize verified work."), ("wait", "Await more evidence.")]


def test_option_permutation_equivariance(model):
    with torch.no_grad():
        reference, _ = model.equivariant_scores(STATE * 10, QUESTION, OPTIONS, max_seq_len=160)
        for order in itertools.permutations(range(3)):
            scores, _ = model.equivariant_scores(STATE * 10, QUESTION, [OPTIONS[i] for i in order], max_seq_len=160)
            torch.testing.assert_close(scores, reference[list(order)], rtol=0, atol=1e-6)


def test_causal_embedding_invariance(model):
    # No truncation: changing suffix content and length cannot change the boundary.
    with torch.no_grad():
        first = model(STATE, QUESTION, OPTIONS)
        second = model(STATE, "A completely different question?", [("yes", "Proceed."), ("no", "Stay.")])
    for dim in SLICE_DIMS:
        torch.testing.assert_close(first["embeddings"][dim], second["embeddings"][dim], rtol=0, atol=1e-6)


def test_mult_question_deltanet_isolation_and_negative_control(model):
    questions = [(QUESTION, OPTIONS), ("What should happen later?", [("finish", "Conclude the task."), ("retry", "Repeat investigation.")])]
    reserve = max(len(model.question_ids(q, opts)[0]) for q, opts in questions)
    with torch.no_grad():
        prefix = model.encode_prefix(STATE, 512, reserve)
        saved = clone_prefix_cache(prefix.cache)
        for question, options in questions:
            isolated = model.question_from_prefix(prefix, question, options)
            fresh = model(STATE, question, options, max_seq_len=512)
            torch.testing.assert_close(isolated["logits"], fresh["logits"], atol=2e-5, rtol=2e-5)
        for layer, original in zip(prefix.cache.layers, saved.layers):
            for name, value in vars(original).items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(getattr(layer, name), value, atol=0, rtol=0)
        model.question_from_prefix(prefix, *questions[0], isolate=False)
        contaminated = model.question_from_prefix(prefix, *questions[1], isolate=False)
        clean = model(STATE, *questions[1], max_seq_len=512)
        assert not torch.allclose(contaminated["logits"], clean["logits"], atol=1e-5, rtol=1e-5)


def test_prefix_cache_equivalence(model):
    with torch.no_grad():
        prompt = model.format_prompt(STATE * 4, QUESTION, OPTIONS, max_seq_len=200)
        full = model.forward_prompt(prompt)
        prefix = model.encode_prefix(STATE * 4, 200, len(prompt.suffix_ids))
        cached = model.question_from_prefix(prefix, QUESTION, OPTIONS)
    torch.testing.assert_close(full["logits"], cached["logits"], atol=2e-5, rtol=2e-5)
    for dim in SLICE_DIMS:
        torch.testing.assert_close(full["embeddings"][dim], cached["embeddings"][dim], atol=1e-6, rtol=0)


def test_clone_copies_kv_conv_recurrent_tensors_and_flags(model):
    with torch.no_grad():
        prefix = model.encode_prefix(STATE)
    cloned = clone_prefix_cache(prefix.cache)
    tensor_names = set()
    flags = set()
    for original, fork in zip(prefix.cache.layers, cloned.layers):
        assert fork is not original
        for name, value in vars(original).items():
            if isinstance(value, torch.Tensor):
                tensor_names.add(name)
                assert getattr(fork, name).data_ptr() != value.data_ptr()
                torch.testing.assert_close(getattr(fork, name), value)
                getattr(fork, name).zero_()
                assert torch.count_nonzero(value) > 0
            elif isinstance(value, bool):
                flags.add(name)
                assert getattr(fork, name) == value
                setattr(fork, name, not value)
                assert getattr(original, name) == value
    assert {"keys", "values", "conv_states", "recurrent_states"} <= tensor_names
    assert {"is_initialized", "is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state"} <= flags


def test_middle_truncation_preserves_boundary_and_all_options(model):
    suffix, ends, query = model.question_ids(QUESTION, OPTIONS)
    prompt = model.format_prompt("HEAD" + "middle" * 100 + "TAIL", QUESTION, OPTIONS, max_seq_len=len(suffix) + 20)
    assert prompt.truncated
    assert len(prompt.input_ids) == len(suffix) + 20
    assert prompt.suffix_ids == suffix
    assert prompt.prefix_ids[0] == model.state_prefix_id
    assert prompt.prefix_ids[1:5] == model.tokenizer.encode("HEAD")
    assert prompt.prefix_ids[-4:] == model.tokenizer.encode("TAIL")
    assert prompt.input_ids.count(model.state_prefix_id) == 1
    assert all(suffix[i] == model.option_end_id for i in ends)
    assert suffix[query] == model.decide_id

    marked = model.format_prompt("HEAD" + "middle" * 100 + "TAIL", QUESTION, OPTIONS, max_seq_len=len(suffix) + 80)
    assert marked.truncated
    decoded = bytes(t - 1 for t in marked.prefix_ids[1:] if 1 <= t <= 256).decode("utf-8", errors="replace")
    assert "tokens omitted" in decoded

    with pytest.raises(ValueError, match="exceed"):
        model.format_prompt(STATE, QUESTION, OPTIONS, max_seq_len=len(suffix))


def test_boundary_reserved_rows_eos_init_and_injection_escape(model):
    weights = model.backbone.get_input_embeddings().weight
    assert weights.shape == (272, 32)
    assert model.state_prefix_id == 257
    for token_id in model.special_ids.values():
        assert 0 <= token_id < weights.shape[0]
    injected = "Injected " + " ".join(SPECIAL_TOKENS) + " marker"
    prompt = model.format_prompt(injected, QUESTION, OPTIONS)
    assert prompt.input_ids.count(model.state_prefix_id) == 1
    assert prompt.input_ids.count(model.question_id) == 1
    assert prompt.input_ids.count(model.decide_id) == 1
    assert prompt.input_ids.count(model.option_start_id) == len(OPTIONS)
    assert prompt.input_ids.count(model.option_end_id) == len(OPTIONS)


def test_matryoshka_per_slice_norm_and_fp32_heads(model):
    with torch.no_grad():
        embeddings = model.embed_state(STATE)
    for dim, value in embeddings.items():
        assert value.shape == (dim,)
        assert value.dtype == torch.float32
        torch.testing.assert_close(value.norm(), torch.tensor(1.), atol=1e-6, rtol=0)
    assert all(p.dtype == torch.float32 for p in model.pointer.parameters())
    assert model.W_emb.weight.dtype == torch.float32
    assert not torch.allclose(embeddings[384][:128].norm(), torch.tensor(1.))


def test_uncached_equivariant_helper_matches_forked(model):
    with torch.no_grad():
        cached, _ = model.equivariant_scores(STATE, QUESTION, OPTIONS, use_cache=True)
        uncached, _ = model.equivariant_scores(STATE, QUESTION, OPTIONS, use_cache=False)
    torch.testing.assert_close(cached, uncached, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", ["exp_a", "exp_b"])
def test_training_freezes_expected_parameters_and_checkpoint_roundtrip(mode, tmp_path):
    model = synthetic_model(seed=5, with_lora=True)
    model.configure_training(mode)
    for name, param in model.backbone.named_parameters():
        assert param.requires_grad == (mode == "exp_b" and "lora_" in name)
    result = model(STATE, QUESTION, OPTIONS)
    loss = result["logits"].square().mean() + result["embeddings"][384].sum()
    loss.backward()
    assert model.pointer.q.weight.grad is not None
    assert model.W_emb.weight.grad is not None
    if mode == "exp_b":
        assert any(p.grad is not None for n, p in model.backbone.named_parameters() if "lora_" in n)
    model.save_checkpoint(tmp_path, {"smoke": True, "mode": mode})
    restored = synthetic_model(seed=5, with_lora=True)
    assert restored.load_checkpoint(tmp_path / "model.pt")["mode"] == mode
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(STATE, QUESTION, OPTIONS)["logits"], restored(STATE, QUESTION, OPTIONS)["logits"])


def test_infonce_inbatch_negatives_and_false_negative_protection(model):
    rows = smoke_rows("train")["emb_pairs"][:2]
    loss = matryoshka_infonce(model, rows, 128)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.W_emb.weight.grad is not None
    rows[0]["hard_negative_task_group"] = rows[0]["task_group"]
    with pytest.raises(ValueError, match="shares anchor"):
        matryoshka_infonce(model, rows, 128)


def test_class_balance_and_deterministic_option_shuffling():
    rows = [{"label": 0, "task_balanced_weight": 1.}] + [{"label": 1, "task_balanced_weight": .5}] * 6
    weights = class_balanced_weights(rows)
    assert weights[0] == pytest.approx(sum(weights[1:]))
    assert option_order(3, 42, 1, 7) == option_order(3, 42, 1, 7)
    assert len({tuple(option_order(3, 42, epoch, 7)) for epoch in range(10)}) > 1


def test_calibration_never_uses_dev_labels():
    cal = [{"label": 0}, {"label": 1}]
    logits = [[3., -1.], [-1., 2.]]
    dev = [{"label": 0, "token_count": 200}, {"label": 0, "token_count": 2000}]
    first = calibrated_evaluation(logits, cal, logits, dev)
    dev[0]["label"], dev[1]["label"] = 1, 1
    second = calibrated_evaluation(logits, cal, logits, dev)
    assert first["temperature"] == second["temperature"]
    assert first["risk_policy"] == second["risk_policy"]
    assert temperature_scale(logits, [0, 1]) > 0


def test_metrics_and_tied_risk_threshold():
    metrics = classification_metrics([[1., 0.], [0., 1.]], [0, 1])
    assert metrics["balanced_accuracy"] == metrics["macro_f1"] == 1
    assert metrics["brier"] == metrics["nll"] == metrics["ece_10bin"] == 0
    policy = risk_threshold(np.asarray([[.9, .1], [.1, .9]]), [0, 0])
    assert policy["cal_coverage"] == 0
    assert policy["threshold"] > 1
    retrieval = retrieval_metrics(np.asarray([[.9, .1], [.1, .9]]), ["a", "b"], ["a", "b"])
    assert retrieval == {"recall_at_1": 1., "recall_at_5": 1., "mrr": 1.}


def test_cached_kev_adapter_all_372_tensors_loaded():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        base = Path(snapshot_download(BASE_MODEL, local_files_only=True))
        adapter = Path(snapshot_download(KEV_ADAPTER, local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip("Qwen/Kev snapshots are not in the local HF cache")
    if not list(base.glob("*.safetensors")) or not (adapter / "adapter_model.safetensors").exists() or not (adapter / "head.pt").exists():
        pytest.skip("Qwen/Kev weight files are not fully cached")
    model = DualHeadDecisionEmbedder.from_pretrained(device="cpu", dtype=torch.bfloat16)
    assert model.adapter_tensor_count == 372
    params = [p for name, p in model.backbone.named_parameters() if "lora_" in name]
    assert len(params) == 372
    assert all(torch.count_nonzero(p) > 0 for p in params)
    assert model.special_ids == {
        "<|fim_prefix|>": 248060,
        "<|fim_middle|>": 248061,
        "<|box_start|>": 248049,
        "<|box_end|>": 248050,
        "<|fim_suffix|>": 248062,
    }
    assert model.state_prefix_id == 248060
    assert model.backbone.get_input_embeddings().num_embeddings == 248320
    assert model.pointer.q.weight.shape == model.pointer.k.weight.shape == (256, 1024)
    assert model.temperature == pytest.approx(2.3510958125672174)


def test_balanced_sampling_and_affine_calibration_without_dev_leak():
    import argparse

    rows = [{"label": 0, "task_balanced_weight": 2.0}, {"label": 0, "task_balanced_weight": 1.0}] + [
        {"label": 1, "task_balanced_weight": 0.5} for _ in range(8)
    ]
    unit_weights = per_class_unit_weights(rows)
    assert np.mean([w for w, r in zip(unit_weights, rows) if r["label"] == 0]) == pytest.approx(1.0)
    assert np.mean([w for w, r in zip(unit_weights, rows) if r["label"] == 1]) == pytest.approx(1.0)

    data = smoke_rows("train")
    data["compaction"] = data["compaction"] + [dict(data["compaction"][1], event_id=f"extra_{i}") for i in range(6)]
    args = argparse.Namespace(epochs=1, max_steps=18, seed=42, emb_batch_size=1, balanced_sampling=True)
    comp_labels = [s["row"]["label"] for s in iter_scheduled_steps(data, args) if s["family"] == "compaction"]
    assert comp_labels[:6] == [0, 1, 0, 1, 0, 1]

    cal_logits = np.array([[-0.5, -1.5], [-0.5, -1.6], [-0.5, -0.7], [-0.5, -0.6]])
    cal_rows = [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]
    t, b = affine_scale(cal_logits, [r["label"] for r in cal_rows])
    assert b > 0.5
    cal_probs = probabilities(cal_logits, t, b)
    cutoff = balanced_decision_threshold(cal_probs, [r["label"] for r in cal_rows])
    dev_rows = [{"label": 0, "token_count": 100}, {"label": 1, "token_count": 200}]
    res1 = calibrated_evaluation(cal_logits, cal_rows, np.array([[-0.5, -1.55], [-0.5, -0.65]]), dev_rows)
    dev_rows[0]["label"], dev_rows[1]["label"] = 1, 0
    res2 = calibrated_evaluation(cal_logits, cal_rows, np.array([[-0.5, -1.55], [-0.5, -0.65]]), dev_rows)
    assert res1["temperature"] == res2["temperature"]
    assert res1["bias"] == res2["bias"]
    assert res1["balanced_threshold"] == cutoff == res2["balanced_threshold"]
    assert res1["balanced_operating_metrics"]["balanced_accuracy"] == pytest.approx(1.0)
    assert res1["metrics"]["roc_auc"] == pytest.approx(1.0)


def test_frozen_reference_cache_only_scheduled_keys_and_matches_untuned(tmp_path):
    import argparse
    from train_experiment import (decision_forward, decision_key,
                                  frozen_reference_cache, iter_scheduled_steps)

    model = synthetic_model(seed=42, with_lora=True)
    data = smoke_rows("train")
    args = argparse.Namespace(
        epochs=1,
        max_steps=5,
        seed=42,
        perm_kl=0.0,
        max_seq_len=512,
        smoke=True,
        emb_batch_size=1,
    )

    cache_file = tmp_path / "reference_logits.pt"
    cache = frozen_reference_cache(model, data, args, cache_file)

    scheduled = list(iter_scheduled_steps(data, args))
    assert len(scheduled) == 5

    expected_keys = {}
    for step in scheduled:
        if step["family"] in ("compaction", "outcome_prefix"):
            k = decision_key(step["family"], step["row"], step["order"], args.max_seq_len)
            expected_keys[k] = (step["family"], step["row"], step["order"])

    assert len(expected_keys) == 4
    assert set(cache.keys()) == set(expected_keys.keys())

    saved = torch.load(cache_file, map_location="cpu", weights_only=True)
    assert saved["format"] == "jev-kev-reference-v1"
    assert saved["smoke"] is True
    assert set(saved["logits"].keys()) == set(expected_keys.keys())

    untuned_model = synthetic_model(seed=42, with_lora=True)
    untuned_model.eval()
    with torch.no_grad():
        for k, (family, row, order) in expected_keys.items():
            expected_logits = decision_forward(untuned_model, family, row, order, args.max_seq_len).float().cpu()
            torch.testing.assert_close(cache[k], expected_logits, rtol=0, atol=0)

    all_possible_keys = {
        decision_key(family, row, ord_, args.max_seq_len)
        for family in ("compaction", "outcome_prefix")
        for row in data[family]
        for ord_ in ([0, 1], [1, 0])
    }
    unscheduled_keys = all_possible_keys - set(expected_keys.keys())
    assert len(unscheduled_keys) > 0
    for k in unscheduled_keys:
        assert k not in cache

    # Also verify reversed order inclusion when perm_kl is nonzero
    args_perm = argparse.Namespace(
        epochs=1,
        max_steps=5,
        seed=42,
        perm_kl=0.1,
        max_seq_len=512,
        smoke=True,
        emb_batch_size=1,
    )
    cache_file_perm = tmp_path / "reference_logits_perm.pt"
    cache_perm = frozen_reference_cache(model, data, args_perm, cache_file_perm)

    expected_perm_keys = {}
    for step in scheduled:
        if step["family"] in ("compaction", "outcome_prefix"):
            for ord_ in (step["order"], list(reversed(step["order"]))):
                k = decision_key(step["family"], step["row"], ord_, args_perm.max_seq_len)
                expected_perm_keys[k] = (step["family"], step["row"], ord_)

    assert set(cache_perm.keys()) == set(expected_perm_keys.keys())
    for k in all_possible_keys - set(expected_perm_keys.keys()):
        assert k not in cache_perm


def test_grad_cache_matches_full_infonce_gradients():
    from train_experiment import matryoshka_infonce, matryoshka_infonce_grad_cache

    rows = smoke_rows("train")["emb_pairs"][:3]
    m1 = synthetic_model(seed=42, with_lora=True)
    m2 = synthetic_model(seed=42, with_lora=True)
    m1.configure_training("exp_b", gradient_checkpointing=False)
    m2.configure_training("exp_b", gradient_checkpointing=False)

    loss1 = 0.25 * matryoshka_infonce(m1, rows, max_seq_len=256)
    loss1.backward()
    loss2 = 0.25 * matryoshka_infonce_grad_cache(m2, rows, max_seq_len=256, loss_scale=0.25)

    assert loss2 == pytest.approx(float(loss1.detach()), rel=1e-5, abs=1e-6)
    compared = 0
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if not p1.requires_grad:
            continue
        assert n1 == n2
        if p1.grad is None:
            assert p2.grad is None
            continue
        assert p2.grad is not None
        torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5)
        compared += 1
    assert compared > 0


def test_loop_kl_and_cosine_lr_smoke(tmp_path):
    import argparse
    from train_experiment import run_training

    args = argparse.Namespace(
        mode="exp_b", run_name="smoke_loop_kl", dataset=tmp_path, device="cpu", dtype="fp32",
        max_seq_len=512, epochs=1, max_steps=8, grad_accum=2, emb_batch_size=2,
        emb_grad_cache=True, lr=1e-4, head_lr=1e-3, pointer_lr=2e-5, emb_lr=5e-4,
        cosine_lr=True, warmup_steps=1, balanced_sampling=True, emb_weight=1.0,
        kl_ref_weight=0.05, kl_loop_weight=0.05, perm_kl=0.0, save_every=2,
        gradient_checkpointing=False, seed=42, out_dir=tmp_path, smoke=True,
    )
    summary = run_training(args)
    assert summary["task_steps"]["loop_guard"] == 2
    assert summary["emb_grad_cache"] is True
    assert summary["cosine_lr"] is True
    assert (tmp_path / "checkpoints" / "smoke_loop_kl" / "latest" / "model.pt").exists()
    assert (tmp_path / "checkpoints" / "smoke_loop_kl" / "model.pt").exists()


def test_cal_cv_never_reads_dev(tmp_path, monkeypatch):
    import argparse
    import json
    import eval_experiment
    from eval_experiment import run_cal_cv_evaluation

    monkeypatch.setattr(eval_experiment, "make_model", lambda smoke, device, dtype, seed: synthetic_model(seed=seed, with_lora=True))
    cal = smoke_rows("cal")
    # Duplicate cal rows across distinct task_groups so 5-fold CV has both classes per fold.
    for family in ("compaction", "outcome_prefix", "emb_pairs", "loop_guard"):
        expanded = []
        for rep in range(3):
            for idx, r in enumerate(cal[family]):
                row = dict(r)
                row["task_group"] = f"cal_g_{rep}_{idx}"
                if family == "emb_pairs":
                    row["positive_task_group"] = row["task_group"]
                    row["hard_negative_task_group"] = f"neg_{rep}_{idx}"
                expanded.append(row)
        (tmp_path / f"{family}_cal.jsonl").write_text("\n".join(json.dumps(x) for x in expanded) + "\n")
        (tmp_path / f"{family}_dev.jsonl").write_text("CORRUPT DEV SHOULD NEVER BE READ\n")

    out = tmp_path / "cal_cv_out"
    args = argparse.Namespace(
        dataset=tmp_path, out_dir=out, checkpoint_root=tmp_path / "ckpts",
        device="cpu", dtype="fp32", max_seq_len=512, max_eval=0, loop_limit=8,
        seed=42, models=["zero_shot_kev"], cal_cv=True, smoke=False,
    )
    report = run_cal_cv_evaluation(args)
    assert report["evaluation_split"] == "cal_5fold_oof"
    assert (out / "cal_cv_report.json").exists()
    assert not (out / "dev_evaluation_started.json").exists()
    comp = report["models"]["zero_shot_kev"]["decisions"]["compaction"]
    assert comp["cv_folds"] == 5
    assert 0.0 <= comp["metrics"]["roc_auc"] <= 1.0
    assert "raw_metrics" in comp
    assert "per_benchmark" in comp["raw_metrics"]


def test_save_checkpoint_preserves_frozen_lora_in_emb_only(tmp_path):
    m = synthetic_model(seed=42, with_lora=True)
    for name, p in m.backbone.named_parameters():
        if "lora_" in name:
            p.data.fill_(0.123)
    m.configure_training("emb_only", gradient_checkpointing=False)
    m.save_checkpoint(tmp_path / "frozen_emb_only", {"mode": "emb_only"})

    reloaded = synthetic_model(seed=42, with_lora=True)
    reloaded.load_checkpoint(tmp_path / "frozen_emb_only" / "model.pt")
    lora_tensors = [p for name, p in reloaded.backbone.named_parameters() if "lora_" in name]
    assert lora_tensors
    for p in lora_tensors:
        torch.testing.assert_close(p, torch.full_like(p, 0.123))


def test_tie_safe_ap_and_cal_cv_raw_metrics():
    from eval_experiment import binary_ranking_metrics, cal_cv_evaluation, group_distributed_rows

    roc1, ap1 = binary_ranking_metrics([0.0, 0.0, 0.0, 0.0], [1, 1, 0, 0])
    roc2, ap2 = binary_ranking_metrics([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 1])
    assert roc1 == pytest.approx(0.5)
    assert roc2 == pytest.approx(0.5)
    assert ap1 == pytest.approx(0.5)
    assert ap2 == pytest.approx(0.5)

    rows = [{"task_group": f"g{i}", "benchmark": "swe" if i < 6 else "ctf", "label": i % 2, "token_count": 200}
            for i in range(10)]
    const_logits = np.tile([-12.0, 12.0], (10, 1))
    res = cal_cv_evaluation(const_logits, rows, n_folds=5)
    assert res["raw_metrics"]["roc_auc"] == pytest.approx(0.5)
    assert res["metrics"]["roc_auc"] == pytest.approx(0.5)
    assert set(res["raw_metrics"]["per_benchmark"]) == {"ctf", "swe"}

    dist = group_distributed_rows(rows, 5)
    assert len({r["task_group"] for r in dist}) == 5


def test_ratio_schedule_dual_teacher_emb_only_and_loop_ce_smoke(tmp_path):
    import argparse
    from eval_experiment import decision_logits
    from train_experiment import run_training

    init_model = synthetic_model(seed=42, with_lora=True)
    for name, p in init_model.backbone.named_parameters():
        if "lora_" in name:
            p.data.add_(0.01)
    init_dir = tmp_path / "checkpoints" / "init_b4"
    init_model.save_checkpoint(init_dir, {"smoke": True})
    init_ckpt = init_dir / "model.pt"

    args = argparse.Namespace(
        mode="exp_b", run_name="pilot_a_smoke", dataset=tmp_path, device="cpu", dtype="fp32",
        max_seq_len=512, epochs=1, max_steps=8, grad_accum=8, emb_batch_size=2,
        emb_grad_cache=False, emb_detach_backbone=True, schedule_mode="ratio_4_2_1_1",
        init_checkpoint=init_ckpt, compaction_ref_checkpoint=None, outcome_ref_checkpoint=init_ckpt,
        lr=2e-5, head_lr=1e-3, pointer_lr=1e-5, emb_lr=5e-4,
        cosine_lr=True, warmup_steps=1, balanced_sampling=True, emb_weight=0.25,
        kl_ref_weight=0.1, kl_compaction_weight=0.5, kl_outcome_weight=0.15, kl_loop_weight=0.08,
        loop_ce_weight=0.5, perm_kl=0.0, save_every=0,
        gradient_checkpointing=False, seed=42, out_dir=tmp_path, smoke=True,
    )
    summary_a = run_training(args)
    assert summary_a["task_steps"] == {"compaction": 4, "outcome_prefix": 2, "emb_pairs": 1, "loop_guard": 1}
    assert summary_a["optimizer_updates"] == 1
    assert summary_a["emb_detach_backbone"] is True
    assert summary_a["lora_changed"] is True

    pilot_a_ckpt = tmp_path / "checkpoints" / "pilot_a_smoke" / "model.pt"
    before_logits = decision_logits(
        synthetic_model(seed=42, with_lora=True), smoke_rows("cal")["compaction"], "compaction", 512
    )
    m_before = synthetic_model(seed=42, with_lora=True)
    m_before.load_checkpoint(pilot_a_ckpt)
    m_before.eval()
    logits_before = decision_logits(m_before, smoke_rows("cal")["compaction"], "compaction", 512)

    args_b = argparse.Namespace(
        mode="emb_only", run_name="pilot_b_smoke", dataset=tmp_path, device="cpu", dtype="fp32",
        max_seq_len=512, epochs=1, max_steps=6, grad_accum=1, emb_batch_size=4,
        emb_grad_cache=False, emb_detach_backbone=False, schedule_mode="legacy",
        init_checkpoint=pilot_a_ckpt, compaction_ref_checkpoint=None, outcome_ref_checkpoint=None,
        lr=0.0, head_lr=5e-4, pointer_lr=0.0, emb_lr=5e-4,
        cosine_lr=True, warmup_steps=1, balanced_sampling=True, emb_weight=1.0,
        kl_ref_weight=0.0, kl_compaction_weight=None, kl_outcome_weight=None, kl_loop_weight=0.0,
        loop_ce_weight=0.0, perm_kl=0.0, save_every=0,
        gradient_checkpointing=False, seed=42, out_dir=tmp_path, smoke=True,
    )
    summary_b = run_training(args_b)
    assert summary_b["mode"] == "emb_only"
    assert summary_b["head_changed"] is False
    assert summary_b["lora_changed"] is False
    assert summary_b["embedding_changed"] is True

    m_after = synthetic_model(seed=42, with_lora=True)
    m_after.load_checkpoint(tmp_path / "checkpoints" / "pilot_b_smoke" / "model.pt")
    m_after.eval()
    logits_after = decision_logits(m_after, smoke_rows("cal")["compaction"], "compaction", 512)
    np.testing.assert_allclose(logits_before, logits_after, atol=1e-6)
    assert before_logits.shape == logits_after.shape


def _make_pattern_image(width: int, height: int, seed_val: int = 0):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (20 + (seed_val * 37) % 180, 30, 50))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, max(6, width // 2), max(6, height // 2)], fill=(230, (seed_val * 71) % 255, 40))
    draw.line([(0, height - 1), (width - 1, 0)], fill=(255, 255, 255), width=2)
    return img


def test_multimodal_causal_invariance_and_equivariance():
    model = synthetic_model(seed=19, include_vision=True)
    img_a = _make_pattern_image(96, 64, seed_val=1)
    img_b = _make_pattern_image(64, 96, seed_val=2)
    with torch.no_grad():
        first = model(STATE, QUESTION, OPTIONS, max_seq_len=256, images=[img_a, img_b])
        second = model(
            STATE,
            "Another visual question?",
            [("yes", "Proceed."), ("no", "Stop.")],
            max_seq_len=256,
            images=[img_a, img_b],
        )
        for dim in SLICE_DIMS:
            torch.testing.assert_close(first["embeddings"][dim], second["embeddings"][dim], rtol=0, atol=1e-5)

        ref_cached, ref_emb = model.equivariant_scores(STATE, QUESTION, OPTIONS, max_seq_len=256, use_cache=True, images=img_a)
        ref_uncached, _ = model.equivariant_scores(STATE, QUESTION, OPTIONS, max_seq_len=256, use_cache=False, images=img_a)
        torch.testing.assert_close(ref_cached, ref_uncached, rtol=2e-5, atol=2e-5)
        for order in itertools.permutations(range(3)):
            scores, emb = model.equivariant_scores(
                STATE, QUESTION, [OPTIONS[i] for i in order], max_seq_len=256, use_cache=True, images=img_a
            )
            torch.testing.assert_close(scores, ref_cached[list(order)], rtol=0, atol=1e-5)
            for dim in SLICE_DIMS:
                torch.testing.assert_close(emb[dim], ref_emb[dim], rtol=0, atol=1e-6)


def test_multimodal_prefix_cache_isolation_and_interleaved_prefixes():
    model = synthetic_model(seed=23, include_vision=True)
    img_a = _make_pattern_image(96, 64, seed_val=3)
    img_b = _make_pattern_image(128, 96, seed_val=4)
    questions = [
        (QUESTION, OPTIONS),
        ("Is a blocking modal visible?", [("clear", "No error modal."), ("blocked", "Blocking error modal visible.")]),
    ]
    reserve = max(len(model.question_ids(q, opts)[0]) for q, opts in questions)
    with torch.no_grad():
        prefix_a = model.encode_prefix(STATE, max_seq_len=320, suffix_reserve=reserve, images=[img_a])
        saved_a = clone_prefix_cache(prefix_a.cache)
        # Interleave a differently shaped image prefix before querying prefix_a.
        prefix_b = model.encode_prefix("Different state", max_seq_len=320, suffix_reserve=reserve, images=[img_b])
        _ = model.question_from_prefix(prefix_b, *questions[0])

        for q, opts in questions:
            cached = model.question_from_prefix(prefix_a, q, opts)
            fresh = model(STATE, q, opts, max_seq_len=320, images=[img_a])
            torch.testing.assert_close(cached["logits"], fresh["logits"], atol=2e-5, rtol=2e-5)
            for dim in SLICE_DIMS:
                torch.testing.assert_close(cached["embeddings"][dim], fresh["embeddings"][dim], atol=1e-5, rtol=0)

        for layer, original in zip(prefix_a.cache.layers, saved_a.layers):
            for name, value in vars(original).items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(getattr(layer, name), value, atol=0, rtol=0)
                elif isinstance(value, bool):
                    assert getattr(layer, name) == value

        model.question_from_prefix(prefix_a, *questions[0], isolate=False)
        contaminated = model.question_from_prefix(prefix_a, *questions[1], isolate=False)
        clean = model(STATE, *questions[1], max_seq_len=320, images=[img_a])
        assert not torch.allclose(contaminated["logits"], clean["logits"], atol=1e-5, rtol=1e-5)

        with pytest.raises(ValueError, match="exceed"):
            too_long = [("a", "x " * 200), ("b", "y")]
            model.question_from_prefix(prefix_a, "Long?", too_long)


def test_multimodal_truncation_preserves_image_spans_and_escapes_markers():
    from model import VISION_TOKENS

    model = synthetic_model(seed=29, include_vision=True)
    img1 = _make_pattern_image(96, 64, seed_val=5)
    img2 = _make_pattern_image(64, 64, seed_val=6)
    _, _, counts = model.preprocess_images([img1, img2])
    suffix, _, _ = model.question_ids(QUESTION, OPTIONS)
    vision_tokens_total = sum(c + 2 for c in counts)
    exact_fit = len(suffix) + vision_tokens_total + 1  # +1 for state_prefix_id
    prompt = model.format_prompt("HEAD" + "middle" * 50 + "TAIL", QUESTION, OPTIONS, max_seq_len=exact_fit, images=[img1, img2])
    assert prompt.truncated
    assert prompt.prefix_ids[0] == model.state_prefix_id
    assert prompt.prefix_ids.count(model.vision_start_id) == 2
    assert prompt.prefix_ids.count(model.vision_end_id) == 2
    assert prompt.prefix_ids.count(model.image_token_id) == sum(counts)

    with pytest.raises(ValueError, match="exceed"):
        model.format_prompt("text", QUESTION, OPTIONS, max_seq_len=exact_fit - 1, images=[img1, img2])

    injected = "Text with " + " ".join(VISION_TOKENS) + " tokens"
    escaped = model.format_prompt(injected, QUESTION, OPTIONS, max_seq_len=320, images=[img1])
    assert escaped.prefix_ids.count(model.vision_start_id) == 1
    assert escaped.prefix_ids.count(model.vision_end_id) == 1
    assert escaped.prefix_ids.count(model.image_token_id) == counts[0]


def test_multimodal_image_sensitivity_frozen_vision_and_text_compat(tmp_path):
    from PIL import Image

    model_vis = synthetic_model(seed=31, with_lora=True, include_vision=True)
    model_txt = synthetic_model(seed=31, with_lora=True, include_vision=False)
    with torch.no_grad():
        out_vis_no_img = model_vis(STATE, QUESTION, OPTIONS, images=[])
        out_txt = model_txt(STATE, QUESTION, OPTIONS)
        torch.testing.assert_close(out_vis_no_img["logits"], out_txt["logits"], atol=1e-6, rtol=0)

    img1 = _make_pattern_image(96, 64, seed_val=10)
    img2 = _make_pattern_image(96, 64, seed_val=25)
    with torch.no_grad():
        r1 = model_vis(STATE, QUESTION, OPTIONS, max_seq_len=256, images=img1)
        r2 = model_vis(STATE, QUESTION, OPTIONS, max_seq_len=256, images=img2)
    assert not torch.allclose(r1["logits"], r2["logits"], atol=1e-5)
    for dim in SLICE_DIMS:
        assert not torch.allclose(r1["embeddings"][dim], r2["embeddings"][dim], atol=1e-5)

    # Grayscale, RGBA, narrow CHW, bfloat16, and non-finite validation.
    gray = Image.new("L", (64, 64), 120)
    rgba = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
    flt = torch.rand(3, 64, 64)
    bf16 = torch.rand(3, 64, 64, dtype=torch.bfloat16)
    pv, grid, counts = model_vis.preprocess_images([gray, rgba, flt, bf16])
    assert len(counts) == 4 and grid.shape == (4, 3) and pv.ndim == 2

    narrow_tensor = torch.randint(0, 255, (3, 64, 4), dtype=torch.uint8)
    narrow_pil = Image.fromarray(narrow_tensor.permute(1, 2, 0).numpy(), mode="RGB")
    pv_t, grid_t, _ = model_vis.preprocess_images([narrow_tensor])
    pv_p, grid_p, _ = model_vis.preprocess_images([narrow_pil])
    assert torch.equal(grid_t, grid_p)
    torch.testing.assert_close(pv_t, pv_p, atol=1e-6, rtol=0)

    bad_nan = torch.full((3, 32, 32), float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        model_vis.preprocess_images([bad_nan])

    # Training keeps visual frozen and in eval mode even after .train().
    for use_gc in (False, True):
        model_vis.configure_training("exp_b", gradient_checkpointing=use_gc)
        model_vis.train()
        assert model_vis.visual.training is False
        assert all(not p.requires_grad for p in model_vis.visual.parameters())
        model_vis.zero_grad(set_to_none=True)
        out = model_vis(STATE, QUESTION, OPTIONS, max_seq_len=256, images=img1)
        loss = out["logits"].square().mean() + out["embeddings"][384].sum()
        loss.backward()
        assert all(p.grad is None for p in model_vis.visual.parameters())
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for n, p in model_vis.backbone.named_parameters() if "lora_" in n)
        assert model_vis.pointer.q.weight.grad is not None and torch.isfinite(model_vis.pointer.q.weight.grad).all()

    # Save checkpoint with custom preprocessing, modify temperature, and verify roundtrip + text->vision compat.
    model_vis.eval()
    model_vis.image_max_pixels = 128 * 16 * 16
    model_vis.image_mean = (0.4, 0.45, 0.5)
    model_vis.image_std = (0.25, 0.25, 0.25)
    model_vis._image_processor = None
    model_vis.temperature = 3.25
    with torch.no_grad():
        expected_custom = model_vis(STATE, QUESTION, OPTIONS, max_seq_len=256, images=img1)
    model_vis.save_checkpoint(tmp_path / "vis_ckpt", {"smoke": True, "include_vision": False})
    saved = torch.load(tmp_path / "vis_ckpt" / "model.pt", map_location="cpu", weights_only=True)
    assert not any(k.startswith("visual.") for k in saved["tensors"])
    assert saved["metadata"]["include_vision"] is True

    restored = synthetic_model(seed=31, with_lora=True, include_vision=True)
    restored.load_checkpoint(tmp_path / "vis_ckpt" / "model.pt")
    assert restored.temperature == pytest.approx(3.25)
    assert restored.image_max_pixels == 128 * 16 * 16
    assert restored.image_mean == pytest.approx((0.4, 0.45, 0.5))
    with torch.no_grad():
        restored_custom = restored(STATE, QUESTION, OPTIONS, max_seq_len=256, images=img1)
        torch.testing.assert_close(restored_custom["logits"], expected_custom["logits"], atol=1e-6, rtol=0)
        torch.testing.assert_close(restored_custom["embeddings"][384], expected_custom["embeddings"][384], atol=1e-6, rtol=0)

    model_txt.save_checkpoint(tmp_path / "txt_ckpt", {"smoke": True})
    restored.load_checkpoint(tmp_path / "txt_ckpt" / "model.pt")


def test_position_ids_match_qwen3_5_get_rope_index_oracle():
    from transformers import Qwen3_5Config, Qwen3_5Model

    model = synthetic_model(seed=37, include_vision=True)
    img1 = _make_pattern_image(96, 64, seed_val=7)
    img2 = _make_pattern_image(64, 128, seed_val=8)
    prompt = model.format_prompt(STATE, QUESTION, OPTIONS, max_seq_len=320, images=[img1, img2])
    input_ids = torch.tensor([prompt.input_ids], dtype=torch.long)
    pos_4d, next_pos = model._compute_position_ids(input_ids, prompt.image_grid_thw)
    assert torch.equal(pos_4d[0], torch.arange(len(prompt.input_ids)).unsqueeze(0))

    oracle_cfg = Qwen3_5Config(
        text_config={
            "vocab_size": 272, "hidden_size": 32, "intermediate_size": 48,
            "num_hidden_layers": 1, "layer_types": ["full_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                                "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
        },
        vision_config={"depth": 1, "hidden_size": 24, "intermediate_size": 32, "num_heads": 2,
                       "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2, "out_hidden_size": 32},
    )
    oracle = Qwen3_5Model(oracle_cfg)
    mm_types = (input_ids == model.image_token_id).to(torch.int32)
    oracle_pos, oracle_deltas = oracle.get_rope_index(input_ids, mm_types, image_grid_thw=prompt.image_grid_thw)
    assert torch.equal(pos_4d[1:], oracle_pos)
    assert next_pos == int(oracle_pos.max().item()) + 1
    assert next_pos - len(prompt.input_ids) == int(oracle_deltas.item())


def test_multitask_v2_training_and_v2_cal_cv_smoke(tmp_path):
    import argparse
    from eval_experiment import run_cal_cv_evaluation
    from train_experiment import run_training

    args = argparse.Namespace(
        mode="exp_b", run_name="v2_smoke", dataset=tmp_path, device="cpu", dtype="fp32",
        include_vision=True, max_seq_len=256, epochs=1, max_steps=16, grad_accum=4,
        emb_batch_size=2, emb_grad_cache=True, emb_detach_backbone=False,
        schedule_mode="multitask_v2",
        v2_mix="compaction:2,outcome_prefix:2,loop_guard:3,generic_decisions:2,multimodal_decisions:2,emb_pairs:2,emb_pairs_bilingual:2,multimodal_emb_pairs:1",
        init_checkpoint=None, compaction_ref_checkpoint=None, outcome_ref_checkpoint=None,
        lr=1e-4, head_lr=1e-3, pointer_lr=2e-5, emb_lr=5e-4,
        cosine_lr=True, warmup_steps=1, balanced_sampling=True, emb_weight=0.5,
        kl_ref_weight=0.1, kl_compaction_weight=0.2, kl_outcome_weight=0.1,
        kl_loop_weight=0.05, loop_ce_weight=0.5,
        kl_generic_weight=0.5, generic_ce_weight=0.25,
        kl_mm_weight=0.1, mm_ce_weight=1.0,
        perm_kl=0.0, save_every=0, gradient_checkpointing=False,
        seed=42, out_dir=tmp_path, smoke=True,
    )
    summary = run_training(args)
    assert summary["include_vision"] is True
    assert summary["optimizer_updates"] == 4
    assert summary["task_steps"]["generic_decisions"] == 2
    assert summary["task_steps"]["multimodal_decisions"] == 2
    assert summary["task_steps"]["multimodal_emb_pairs"] == 1
    assert summary["reference_logit_count"] > 0
    v2_ckpt = tmp_path / "checkpoints" / "v2_smoke" / "model.pt"
    assert v2_ckpt.exists()

    # Stage-B multimodal emb_only refit across all 3 embedding families
    args_emb = argparse.Namespace(
        mode="emb_only", run_name="v2_emb_only", dataset=tmp_path, device="cpu", dtype="fp32",
        include_vision=True, max_seq_len=256, epochs=1, max_steps=6, grad_accum=1,
        emb_batch_size=2, emb_families="emb_pairs,emb_pairs_bilingual,multimodal_emb_pairs",
        init_checkpoint=v2_ckpt, lr=0.0, head_lr=5e-4, emb_lr=5e-4,
        cosine_lr=True, warmup_steps=1, seed=42, out_dir=tmp_path, smoke=True,
    )
    emb_summary = run_training(args_emb)
    assert set(emb_summary["task_steps"]) == {"emb_pairs", "emb_pairs_bilingual", "multimodal_emb_pairs"}
    assert emb_summary["embedding_changed"] is True
    assert emb_summary["head_changed"] is False

    # Smoke cal-CV evaluation across all v1 + v2 suites
    eval_args = argparse.Namespace(
        dataset=tmp_path, out_dir=tmp_path / "cv_out", checkpoint_root=tmp_path / "checkpoints",
        device="cpu", dtype="fp32", include_vision=True, include_v2=True,
        max_seq_len=256, max_eval=0, loop_limit=8, seed=42,
        models=["zero_shot_kev", "v2_smoke"], cal_cv=True, smoke=True,
    )
    report = run_cal_cv_evaluation(eval_args)
    m = report["models"]["v2_smoke"]
    assert "generic_decisions" in m and "mean_kl_vs_zero_shot" in m["generic_decisions"]
    assert "synthetic_3class" in m["loop_guard"]
    assert "emb_pairs_bilingual" in m and "384" in m["emb_pairs_bilingual"]["slices"]
    assert "multimodal_emb_pairs" in m and "128" in m["multimodal_emb_pairs"]["slices"]
    assert "multimodal_decisions" in m and "mean_kl_vs_zero_shot" in m["multimodal_decisions"]
    assert (tmp_path / "cv_out" / "cal_cv_report.md").exists()


def test_exp_a_default_kl_multitask_v2_defaults_and_image_fingerprints(tmp_path):
    import argparse
    from PIL import Image
    from eval_experiment import evaluate_variable_option_decisions
    from train_experiment import image_fingerprints, run_training

    # 1. exp_a with default kl_ref_weight=0.1 should not raise.
    args_a = argparse.Namespace(
        mode="exp_a", run_name="exp_a_default_kl", dataset=tmp_path, device="cpu", dtype="fp32",
        max_seq_len=256, epochs=1, max_steps=6, grad_accum=2, emb_batch_size=1,
        lr=1e-4, head_lr=1e-3, balanced_sampling=True, emb_weight=1.0, kl_ref_weight=0.1,
        perm_kl=0.0, gradient_checkpointing=False, seed=42, out_dir=tmp_path, smoke=True,
    )
    summary_a = run_training(args_a)
    assert summary_a["kl_compaction_weight"] == 0.0

    # 2. multitask_v2 with omitted/None loss weights uses safe non-zero defaults.
    args_v2 = argparse.Namespace(
        mode="exp_b", run_name="v2_defaults", dataset=tmp_path, device="cpu", dtype="fp32",
        max_seq_len=256, epochs=1, max_steps=16, grad_accum=4, emb_batch_size=2,
        schedule_mode="multitask_v2", lr=1e-4, head_lr=1e-3, balanced_sampling=True,
        emb_weight=0.5, kl_ref_weight=0.1, perm_kl=0.0, gradient_checkpointing=False,
        seed=42, out_dir=tmp_path, smoke=True,
    )
    summary_v2 = run_training(args_v2)
    assert summary_v2["loop_ce_weight"] > 0 and summary_v2["kl_generic_weight"] > 0

    # 3. Equal-bytes different-shape PIL images produce distinct fingerprints.
    img_tall = Image.new("RGB", (32, 64), (10, 20, 30))
    img_wide = Image.new("RGB", (64, 32), (10, 20, 30))
    assert img_tall.tobytes() == img_wide.tobytes()
    fp1 = image_fingerprints({"state_images": [img_tall]})
    fp2 = image_fingerprints({"state_images": [img_wide]})
    assert fp1 != fp2

    # 4. Incomplete or non-finite logits fail validation cleanly.
    rows = smoke_rows("cal")["generic_decisions"][:2]
    with pytest.raises(ValueError, match="logit count"):
        evaluate_variable_option_decisions([[0.1, 0.2, 0.3]], rows, "generic_decisions")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_variable_option_decisions([[0.1, float("nan"), 0.3], [0.1, 0.2, 0.3]], rows, "generic_decisions")


def test_cached_kev_multimodal_smoke():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        base = Path(snapshot_download(BASE_MODEL, local_files_only=True))
        adapter = Path(snapshot_download(KEV_ADAPTER, local_files_only=True))
    except LocalEntryNotFoundError:
        pytest.skip("Qwen/Kev snapshots are not in the local HF cache")
    if not list(base.glob("*.safetensors")) or not (adapter / "adapter_model.safetensors").exists() or not (adapter / "head.pt").exists():
        pytest.skip("Qwen/Kev weight files are not fully cached")

    model = DualHeadDecisionEmbedder.from_pretrained(device="cpu", dtype=torch.bfloat16, include_vision=True)
    model.eval()
    assert model.adapter_tensor_count == 372
    assert model.visual is not None
    assert model.visual.training is False
    assert all(not p.requires_grad for p in model.visual.parameters())

    img_a = _make_pattern_image(96, 64, seed_val=1)
    img_b = _make_pattern_image(96, 64, seed_val=2)
    q = "What color is the highlighted block?"
    opts = [("red", "Red block."), ("green", "Green block.")]
    with torch.no_grad():
        full = model("Inspect screenshot.", q, opts, max_seq_len=256, images=[img_a])
        prefix = model.encode_prefix("Inspect screenshot.", max_seq_len=256, suffix_reserve=64, images=[img_a])
        cached = model.question_from_prefix(prefix, q, opts)
        other = model("Inspect screenshot.", q, opts, max_seq_len=256, images=[img_b])
    torch.testing.assert_close(full["logits"], cached["logits"], atol=1e-2, rtol=1e-2)
    assert not torch.allclose(full["logits"], other["logits"], atol=1e-4)
    assert not torch.allclose(full["embeddings"][384], other["embeddings"][384], atol=1e-4)

