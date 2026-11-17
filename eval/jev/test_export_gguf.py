"""Unit tests for pinned snapshot loading, GGUF export helpers, and standalone heads."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_gguf import (  # noqa: E402
    LOCAL_QWEN35_PRETOKENIZER_HASH,
    UPSTREAM_QWEN35_HASH,
    build_quantize_command,
    build_train_imatrix_corpus,
    build_vision_preprocess_sidecar,
    prepare_converter,
)
from model import (  # noqa: E402
    BASE_MODEL_REVISION,
    KEV_ADAPTER_REVISION,
    PointerHead,
    resolve_cached_snapshot,
    synthetic_model,
)


def test_resolve_cached_snapshot_ignores_missing_or_stale_main_ref(tmp_path):
    repo_id = "org/fake-model"
    pinned_rev = "1111111111111111111111111111111111111111"
    other_rev = "2222222222222222222222222222222222222222"
    repo_cache = tmp_path / "models--org--fake-model"
    (repo_cache / "snapshots" / pinned_rev).mkdir(parents=True)
    (repo_cache / "snapshots" / other_rev).mkdir(parents=True)
    (repo_cache / "snapshots" / pinned_rev / "config.json").write_text('{"pinned": true}')
    (repo_cache / "snapshots" / other_rev / "config.json").write_text('{"pinned": false}')
    (repo_cache / "refs").mkdir()
    (repo_cache / "refs" / "main").write_text(other_rev)

    resolved = resolve_cached_snapshot(repo_id, pinned_rev, cache_dir=tmp_path, local_files_only=True)
    assert resolved == repo_cache / "snapshots" / pinned_rev
    assert '"pinned": true' in (resolved / "config.json").read_text()

    (repo_cache / "refs" / "main").unlink()
    resolved_no_ref = resolve_cached_snapshot(repo_id, pinned_rev, cache_dir=tmp_path, local_files_only=True)
    assert resolved_no_ref == repo_cache / "snapshots" / pinned_rev
    assert len(BASE_MODEL_REVISION) == 40
    assert len(KEV_ADAPTER_REVISION) == 40


def test_build_quantize_command_standard_and_ud_overrides(tmp_path):
    qbin = tmp_path / "llama-quantize"
    im = tmp_path / "imatrix.dat"
    bf16 = tmp_path / "model-bf16.gguf"
    out = tmp_path / "out.gguf"

    std = build_quantize_command(qbin, im, bf16, out, "Q4_K_M", 8)
    assert std == [str(qbin), "--imatrix", str(im), str(bf16), str(out), "Q4_K_M", "8"]

    ud = build_quantize_command(qbin, im, bf16, out, "ud_iq4_xl", 8)
    assert "--token-embedding-type" in ud
    assert ud[ud.index("--token-embedding-type") + 1] == "q6_k"
    assert ud[-2:] == ["IQ4_XS", "8"]
    tensor_args = [ud[i + 1] for i, tok in enumerate(ud) if tok == "--tensor-type"]
    assert "attn_q\\.=q8_0" in tensor_args
    assert "ssm_alpha=q8_0" in tensor_args
    assert "attn_qkv=q5_k" in tensor_args


def test_prepare_converter_patches_copies_package_and_validates(tmp_path):
    llama_dir = tmp_path / "llama_up"
    (llama_dir / "conversion").mkdir(parents=True)
    (llama_dir / "conversion" / "__init__.py").write_text("# pkg\n")
    (llama_dir / "convert_hf_to_gguf.py").write_text(f'HASH = "{UPSTREAM_QWEN35_HASH}"\n')

    scratch = tmp_path / "scratch1"
    scratch.mkdir()
    out_py, status = prepare_converter(llama_dir, scratch)
    assert status == "patched_upstream_qwen35_chkhsh"
    assert LOCAL_QWEN35_PRETOKENIZER_HASH in out_py.read_text()
    assert (scratch / "conversion" / "__init__.py").is_file()

    llama_bad = tmp_path / "llama_bad"
    llama_bad.mkdir()
    (llama_bad / "convert_hf_to_gguf.py").write_text("HASH = 'unknown'\n")
    with pytest.raises(ValueError, match="Unsupported convert_hf_to_gguf.py"):
        prepare_converter(llama_bad, tmp_path / "scratch2")


def test_vision_preprocess_sidecar_and_standalone_heads():
    m = synthetic_model(seed=7, with_lora=False, include_vision=True)
    sidecar = build_vision_preprocess_sidecar(m)
    assert sidecar["min_pixels"] == 64 * 16 * 16
    assert sidecar["max_pixels"] == 384 * 16 * 16
    assert sidecar["image_mean"] == [0.5, 0.5, 0.5]
    assert sidecar["image_std"] == [0.5, 0.5, 0.5]

    hidden_size = m.backbone.config.hidden_size
    pointer = PointerHead(hidden_size, head_dim=16)
    pointer.load_state_dict(m.pointer.state_dict())
    w_emb = nn.Linear(hidden_size, 384, bias=False, dtype=torch.float32)
    w_emb.load_state_dict(m.W_emb.state_dict())

    tok_h = torch.randn(12, hidden_size, dtype=torch.float32)
    with torch.no_grad():
        logits = pointer(tok_h[11], tok_h[[4, 7, 10]])
        emb_raw = w_emb(tok_h[2])
        emb_128 = torch.nn.functional.normalize(emb_raw[:128], dim=-1)
    assert logits.shape == (3,)
    assert emb_128.shape == (128,)
    assert torch.isclose(emb_128.norm(), torch.tensor(1.0), atol=1e-5)


def test_build_train_imatrix_corpus_creates_provenance(tmp_path):
    m = synthetic_model(seed=3, with_lora=False, include_vision=False)
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "compaction_train.jsonl").write_text(
        "\n".join(f'{{"split":"train","task_id":"t{i}","event_id":"e{i}","state_text":"trace {i}"}}' for i in range(10))
    )
    (ds / "loop_guard_train.jsonl").write_text(
        "\n".join(f'{{"split":"train","task_id":"l{i}","event_id":"e{i}","state_text":"loop {i}"}}' for i in range(10))
    )
    (ds / "generic_decisions_train.jsonl").write_text(
        "\n".join(
            f'{{"split":"train","task_id":"g{i}","state_text":"doc {i}","question":"q?","options":[["a","yes"],["b","no"]],"label":"a"}}'
            for i in range(20)
        )
    )
    (ds / "emb_pairs_train.jsonl").write_text(
        "\n".join(
            f'{{"split":"train","task_id":"p{i}","task_group":"g{i}","hard_negative_task_group":"neg{i}","hard_negative_split":"train","anchor_text":"anchor {i}"}}'
            for i in range(10)
        )
    )

    out = tmp_path / "calib.txt"
    selected = build_train_imatrix_corpus(m, ds, out, seed=42)
    assert len(selected) == 40
    assert out.read_text().strip()
