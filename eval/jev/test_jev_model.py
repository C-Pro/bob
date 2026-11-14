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
