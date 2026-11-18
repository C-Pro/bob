#!/usr/bin/env python3
"""Export a trained bob-s1 / Jev checkpoint to BF16/quantized GGUFs, mmproj, and standalone heads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (  # noqa: E402
    BASE_MODEL,
    BASE_MODEL_REVISION,
    KEV_ADAPTER,
    KEV_ADAPTER_REVISION,
    QUESTIONS,
    DualHeadDecisionEmbedder,
    resolve_cached_snapshot,
)
from train_experiment import read_rows  # noqa: E402

# SHA-256 of tokenized probe string IDs checked by chkhsh() in convert_hf_to_gguf.py.
LOCAL_QWEN35_PRETOKENIZER_HASH = "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f"
UPSTREAM_QWEN35_HASH = "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4"

UD_RECIPES = {
    "ud_iq4_xl": {
        "base": "IQ4_XS",
        "token_embd": "q6_k",
        "tensors": {
            "attn_q\\.": "q8_0", "attn_k\\.": "q8_0", "attn_v\\.": "q8_0",
            "attn_output": "q6_k", "ssm_out": "q6_k", "attn_gate": "q6_k",
            "ssm_alpha": "q8_0", "ssm_beta": "q8_0", "attn_qkv": "q5_k",
        },
    },
    "ud_iq3_xxs": {
        "base": "IQ3_XXS",
        "token_embd": "q4_k",
        "tensors": {
            "attn_q\\.": "q6_k", "attn_k\\.": "q6_k", "attn_v\\.": "q6_k",
            "attn_output": "q5_k", "ssm_out": "q5_k", "attn_gate": "q5_k",
            "ssm_alpha": "q8_0", "ssm_beta": "q8_0", "attn_qkv": "q4_k",
        },
    },
    "ud_iq2_tiny": {
        "base": "IQ2_XXS",
        "token_embd": "q3_k",
        "tensors": {
            "attn_q\\.": "q5_k", "attn_k\\.": "q5_k", "attn_v\\.": "q5_k",
            "attn_output": "q4_k", "ssm_out": "q4_k", "attn_gate": "q4_k",
            "ssm_alpha": "q8_0", "ssm_beta": "q8_0", "attn_qkv": "iq4_xs",
        },
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_dir: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return res.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _tool_version(bin_path: Path) -> str | None:
    try:
        res = subprocess.run(
            [str(bin_path), "--version"],
            env={"LD_LIBRARY_PATH": str(bin_path.parent), "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, check=False,
        )
        out = (res.stdout or res.stderr or "").strip().splitlines()
        return out[0] if out else None
    except OSError:
        return None


def prepare_converter(llama_cpp_dir: Path, scratch: Path) -> tuple[Path, str]:
    src = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not src.is_file():
        raise FileNotFoundError(f"Missing convert_hf_to_gguf.py in {llama_cpp_dir}")
    # Support both monolithic convert_hf_to_gguf.py and multi-file conversion/ packages.
    for pkg_name in ("conversion",):
        pkg_dir = llama_cpp_dir / pkg_name
        if pkg_dir.is_dir():
            shutil.copytree(pkg_dir, scratch / pkg_name, dirs_exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst = scratch / "convert_hf_to_gguf.py"
    if UPSTREAM_QWEN35_HASH in text:
        dst.write_text(text.replace(UPSTREAM_QWEN35_HASH, LOCAL_QWEN35_PRETOKENIZER_HASH), encoding="utf-8")
        return dst, "patched_upstream_qwen35_chkhsh"
    if LOCAL_QWEN35_PRETOKENIZER_HASH in text:
        dst.write_text(text, encoding="utf-8")
        return dst, "already_recognized"
    raise ValueError(
        "Unsupported convert_hf_to_gguf.py: neither upstream nor local Qwen3.5 chkhsh probe hash was found"
    )


def build_quantize_command(quant_bin: Path, imatrix_dat: Path, bf16_path: Path, out_gguf: Path,
                           quant: str, threads: int) -> list[str]:
    q_lower = quant.lower()
    cmd = [str(quant_bin), "--imatrix", str(imatrix_dat)]
    if q_lower in UD_RECIPES:
        spec = UD_RECIPES[q_lower]
        cmd.extend(["--token-embedding-type", spec["token_embd"]])
        for pat, typ in spec["tensors"].items():
            cmd.extend(["--tensor-type", f"{pat}={typ}"])
        base_typ = spec["base"]
    else:
        base_typ = quant.upper()
    cmd.extend([str(bf16_path), str(out_gguf), base_typ, str(threads)])
    return cmd


def build_train_imatrix_corpus(jev: DualHeadDecisionEmbedder, dataset_dir: Path, out_path: Path, seed: int = 42):
    rng = random.Random(seed)
    chunks = []
    selected = []
    for r in rng.sample(read_rows(dataset_dir, "compaction", "train"), 8):
        p = jev.format_prompt(r["state_text"], *QUESTIONS["compaction"], max_seq_len=512)
        chunks.append(jev.tokenizer.decode(p.input_ids, skip_special_tokens=False))
        selected.append({"family": "compaction", "task_id": r.get("task_id"), "event_id": r.get("event_id")})
    for r in rng.sample(read_rows(dataset_dir, "loop_guard", "train"), 8):
        p = jev.format_prompt(r["state_text"], *QUESTIONS["loop_guard"], max_seq_len=512)
        chunks.append(jev.tokenizer.decode(p.input_ids, skip_special_tokens=False))
        selected.append({"family": "loop_guard", "task_id": r.get("task_id"), "event_id": r.get("event_id")})
    for r in rng.sample(read_rows(dataset_dir, "generic_decisions", "train"), 16):
        opts = [(o[0], o[1]) for o in r["options"]]
        p = jev.format_prompt(r["state_text"], r["question"], opts, max_seq_len=512)
        chunks.append(jev.tokenizer.decode(p.input_ids, skip_special_tokens=False))
        selected.append({"family": "generic_decisions", "task_id": r.get("task_id")})
    for r in rng.sample(read_rows(dataset_dir, "emb_pairs", "train"), 8):
        ids, _ = jev.state_ids(r["anchor_text"], 512)
        chunks.append(jev.tokenizer.decode(ids, skip_special_tokens=False))
        selected.append({"family": "emb_pairs", "task_id": r.get("task_id")})
    out_path.write_text("\n\n".join(chunks), encoding="utf-8")
    return selected


def build_vision_preprocess_sidecar(jev: DualHeadDecisionEmbedder) -> dict:
    patch_size = getattr(jev.vision_config, "patch_size", 16) if jev.vision_config is not None else 16
    merge_size = getattr(jev.vision_config, "spatial_merge_size", 2) if jev.vision_config is not None else 2
    temporal_patch_size = getattr(jev.vision_config, "temporal_patch_size", 2) if jev.vision_config is not None else 2
    return {
        "schema_version": "bob-s1-vision-preprocess-v1",
        "min_pixels": jev.image_min_pixels,
        "max_pixels": jev.image_max_pixels,
        "image_mean": list(jev.image_mean),
        "image_std": list(jev.image_std),
        "patch_size": int(patch_size),
        "spatial_merge_size": int(merge_size),
        "temporal_patch_size": int(temporal_patch_size),
        "note": "llama.cpp Qwen3VLVisionModel mmproj stores mean/std but does not export HF min_pixels/max_pixels; runtimes must apply these bounds explicitly.",
    }


def export_checkpoint(args):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    ckpt_dir = Path(args.checkpoint_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    llama_cpp_dir = Path(args.llama_cpp_dir).resolve()
    quant_bin = Path(args.llama_quantize).resolve()
    imatrix_bin = Path(args.llama_imatrix).resolve()
    model_pt = ckpt_dir / "model.pt"

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        scratch = Path(tmp)
        hf_dir = scratch / "hf_merged"
        hf_dir.mkdir(parents=True, exist_ok=True)

        base_snap = resolve_cached_snapshot(BASE_MODEL, args.base_revision, local_files_only=True)
        base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(base_snap), dtype=torch.bfloat16, local_files_only=True
        )
        jev = DualHeadDecisionEmbedder.from_pretrained(
            device="cpu", dtype=torch.bfloat16, include_vision=True,
            base_revision=args.base_revision, adapter_revision=args.adapter_revision,
        )
        jev.load_checkpoint(model_pt)
        merged_lm = jev.backbone.merge_and_unload()
        base_model.model.language_model.load_state_dict(merged_lm.state_dict(), strict=True)
        base_model.save_pretrained(hf_dir)
        Qwen3_5Config.from_pretrained(str(base_snap), local_files_only=True).to_json_file(hf_dir / "config.json")
        jev._get_image_processor().save_pretrained(hf_dir)
        for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            src = base_snap / fname
            if src.is_file():
                shutil.copy2(src, hf_dir / fname)

        heads_path = out_dir / f"{args.name}-heads.pt"
        torch.save({"W_emb": jev.W_emb.state_dict(), "pointer": jev.pointer.state_dict()}, heads_path)
        preprocess_path = out_dir / f"{args.name}-preprocess.json"
        preprocess_data = build_vision_preprocess_sidecar(jev)
        preprocess_path.write_text(json.dumps(preprocess_data, indent=2) + "\n", encoding="utf-8")

        convert_py, converter_patch_status = prepare_converter(llama_cpp_dir, scratch)
        bf16_path = out_dir / f"{args.name}-bf16.gguf"
        mmproj_path = out_dir / f"{args.name}-mmproj-bf16.gguf"
        conv_env = {"PYTHONPATH": str(llama_cpp_dir / "gguf-py"), "PATH": "/usr/bin:/bin"}
        subprocess.run(
            [sys.executable, str(convert_py), str(hf_dir), "--model-name", args.name,
             "--outfile", str(bf16_path), "--outtype", "bf16"],
            env=conv_env, check=True,
        )
        subprocess.run(
            [sys.executable, str(convert_py), str(hf_dir), "--mmproj", "--model-name", args.name,
             "--outfile", str(mmproj_path), "--outtype", "bf16"],
            env=conv_env, check=True,
        )

        calib_txt = scratch / "calib_train.txt"
        imatrix_dat = scratch / "imatrix.dat"
        selected_calib = build_train_imatrix_corpus(jev, Path(args.dataset_dir), calib_txt, seed=args.seed)
        im_env = {"LD_LIBRARY_PATH": str(imatrix_bin.parent), "PATH": "/usr/bin:/bin"}
        subprocess.run(
            [str(imatrix_bin), "-m", str(bf16_path), "-f", str(calib_txt), "-c", "512", "--chunks", "16",
             "-ngl", "0", "-o", str(imatrix_dat), "--parse-special"],
            env=im_env, check=True,
        )

        q_env = {"LD_LIBRARY_PATH": str(quant_bin.parent), "PATH": "/usr/bin:/bin"}
        outputs = [heads_path, preprocess_path, bf16_path, mmproj_path]
        for q in args.quants:
            out_gguf = out_dir / f"{args.name}-{q.lower()}.gguf"
            cmd = build_quantize_command(quant_bin, imatrix_dat, bf16_path, out_gguf, q, args.threads)
            subprocess.run(cmd, env=q_env, check=True)
            outputs.append(out_gguf)

        manifest = {
            "model_name": args.name,
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_sha256": sha256_file(model_pt),
            "base_model": BASE_MODEL,
            "base_revision": args.base_revision,
            "adapter_donor": KEV_ADAPTER,
            "adapter_revision": args.adapter_revision,
            "bob_git_commit": _git_head(Path(__file__).resolve().parents[2]),
            "llama_cpp_dir": str(llama_cpp_dir),
            "llama_cpp_commit": _git_head(llama_cpp_dir),
            "converter_patch_status": converter_patch_status,
            "tool_versions": {
                "llama_imatrix": _tool_version(imatrix_bin),
                "llama_quantize": _tool_version(quant_bin),
            },
            "imatrix": {
                "seed": args.seed,
                "ctx_len": 512,
                "chunks": 16,
                "calib_txt_sha256": sha256_file(calib_txt),
                "imatrix_dat_sha256": sha256_file(imatrix_dat),
                "selected_rows": selected_calib,
            },
            "quants": list(args.quants),
            "ud_recipes_used": {q.lower(): UD_RECIPES[q.lower()] for q in args.quants if q.lower() in UD_RECIPES},
            "outputs": {
                p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in outputs
            },
        }
        (out_dir / f"{args.name}-export-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", default="eval/jev/checkpoints/exp_b_v2_4k_emb_onlymm")
    p.add_argument("--out-dir", default="eval/jev/checkpoints/exp_b_v2_4k_emb_onlymm")
    p.add_argument("--dataset-dir", default="eval/results/jev_dataset")
    p.add_argument("--name", default="bob-s1-0.8b")
    p.add_argument("--base-revision", default=BASE_MODEL_REVISION)
    p.add_argument("--adapter-revision", default=KEV_ADAPTER_REVISION)
    p.add_argument("--quants", nargs="+", default=["Q4_K_M"])
    p.add_argument("--llama-cpp-dir", required=True)
    p.add_argument("--llama-imatrix", required=True)
    p.add_argument("--llama-quantize", required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    export_checkpoint(p.parse_args())


if __name__ == "__main__":
    main()
