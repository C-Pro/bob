"""Causal state embeddings and isolated, position-free option decisions."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
KEV_ADAPTER = "jaredpalmer/kev-0.8b"
# Reuse Qwen's native special delimiters aligned with jaredpalmer/kev-0.8b LoRA weights:
# [state_prefix] -> [question] [opt_start ... opt_end]+ [decide].
SPECIAL_TOKENS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
SLICE_DIMS = (128, 256, 384)

QUESTIONS = {
    "compaction": ("Should the observed task state be compacted now?", (
        ("skip", "Keep raw context; no coherent verified checkpoint yet."),
        ("compact", "Summarize a coherent verified checkpoint into task state."))),
    "outcome_prefix": ("Does the observed progress suggest eventual task success?", (
        ("failed", "The task will fail its clean terminal evaluation."),
        ("passed", "The task will pass its clean terminal evaluation."))),
    "loop_guard": ("Is this proposed tool execution stuck in a repeating loop?", (
        ("none", "Allow productive progress or a constructive error."),
        ("intercept", "Block repeated calls with unchanged observations."),
        ("force_synthesis", "A blocked call is repeated; stop the tool loop."))),
}


def clone_prefix_cache(cache):
    """Clone mutable hybrid attention state, including DeltaNet flags and tensors."""
    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return copy.copy(value)

    result = copy.copy(cache)
    for name, value in vars(cache).items():
        if name == "layers":
            layers = []
            for layer in value:
                new_layer = copy.copy(layer)
                for attr, item in vars(layer).items():
                    setattr(new_layer, attr, clone(item))
                layers.append(new_layer)
            result.layers = layers
        else:
            setattr(result, name, clone(value))
    return result


class PointerHead(nn.Module):
    def __init__(self, hidden_size, head_dim=256):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.k = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.head_dim = head_dim

    def forward(self, query_hidden, option_hidden):
        return (self.q(query_hidden.float()) * self.k(option_hidden.float())).sum(-1) / math.sqrt(self.head_dim)


@dataclass
class Prompt:
    prefix_ids: list[int]
    suffix_ids: list[int]
    option_ends: list[int]
    query_end: int
    truncated: bool

    @property
    def input_ids(self):
        return self.prefix_ids + self.suffix_ids


@dataclass
class Prefix:
    cache: object
    input_ids: list[int]
    embeddings: dict[int, torch.Tensor]
    truncated: bool


class DualHeadDecisionEmbedder(nn.Module):
    def __init__(self, backbone, tokenizer, head_dim=256, head_state=None, temperature=1.0):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.pointer = PointerHead(backbone.config.hidden_size, head_dim)
        self.W_emb = nn.Linear(backbone.config.hidden_size, 384, bias=False, dtype=torch.float32)
        self.temperature = float(temperature)
        self.adapter_tensor_count = 0
        if head_state is not None:
            self.pointer.load_state_dict(head_state, strict=True)
        self._register_boundary_tokens()
        self.pointer.to(self.device)
        self.W_emb.to(self.device)

    @property
    def device(self):
        return self.backbone.get_input_embeddings().weight.device

    def _register_boundary_tokens(self):
        embedding = self.backbone.get_input_embeddings()
        vocab = self.tokenizer.get_vocab()
        missing = [token for token in SPECIAL_TOKENS if token not in vocab]
        if missing:
            raise ValueError(f"Tokenizer missing required native delimiter tokens: {missing}")
        self.special_ids = {token: self.tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}
        for token, token_id in self.special_ids.items():
            if not 0 <= token_id < embedding.num_embeddings:
                raise ValueError(f"Delimiter token {token} ID {token_id} out of embedding bounds {embedding.num_embeddings}")
        self.state_prefix_id = self.special_ids[SPECIAL_TOKENS[0]]
        self.question_id = self.special_ids[SPECIAL_TOKENS[1]]
        self.option_start_id = self.special_ids[SPECIAL_TOKENS[2]]
        self.option_end_id = self.special_ids[SPECIAL_TOKENS[3]]
        self.decide_id = self.special_ids[SPECIAL_TOKENS[4]]

    @classmethod
    def from_pretrained(cls, device="cpu", dtype=torch.float32, trainable_adapter=False):
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_path = snapshot_download(BASE_MODEL, local_files_only=True)
        adapter_path = Path(snapshot_download(KEV_ADAPTER, local_files_only=True))
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        causal_model = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=dtype, local_files_only=True, attn_implementation="eager")
        # Kev keys are base_model.model.layers.*, not base_model.model.model.layers.*.
        backbone = PeftModel.from_pretrained(causal_model.model, str(adapter_path),
                                             is_trainable=trainable_adapter, local_files_only=True)
        del causal_model
        expected = load_file(str(adapter_path / "adapter_model.safetensors"))
        loaded = {name.replace(".default.", "."): tensor for name, tensor in backbone.named_parameters()
                  if "lora_" in name}
        expected = {name: tensor for name, tensor in expected.items() if "lora_" in name}
        if len(expected) != 372 or set(loaded) != set(expected):
            raise AssertionError(f"Kev adapter key mismatch: expected 372, file={len(expected)}, loaded={len(loaded)}")
        for name, tensor in loaded.items():
            reference = expected[name].to(dtype=tensor.dtype, device=tensor.device)
            if not torch.isfinite(tensor).all() or torch.count_nonzero(tensor) == 0 or not torch.equal(tensor, reference):
                raise AssertionError(f"Kev tensor was not loaded correctly: {name}")
        head = torch.load(adapter_path / "head.pt", map_location="cpu", weights_only=True)
        model = cls(backbone, tokenizer, head_dim=head.get("head_dim", 256),
                    head_state=head["head"], temperature=head["temperature"])
        model.adapter_tensor_count = len(loaded)
        model.to(device=device)
        return model

    def _text_ids(self, text):
        # Treat user-supplied marker spellings as text, never structural tokens.
        for token in SPECIAL_TOKENS:
            text = text.replace(token, token.replace("<|", "‹").replace("|>", "›"))
        return self.tokenizer.encode(text, add_special_tokens=False)

    def question_ids(self, question, options):
        if not options or len({name for name, _ in options}) != len(options):
            raise ValueError("Options require nonempty, unique names")
        ids = [self.question_id] + self._text_ids(question)
        ends = []
        for name, description in options:
            ids.append(self.option_start_id)
            ids.extend(self._text_ids(f"{name}: {description}"))
            ids.append(self.option_end_id)
            ends.append(len(ids) - 1)
        ids.append(self.decide_id)
        return ids, ends, len(ids) - 1

    def state_ids(self, state, budget):
        if budget < 1:
            raise ValueError("Question and options exceed sequence budget")
        ids = self._text_ids(state)
        keep = budget - 1
        truncated = len(ids) > keep
        if truncated:
            omitted = len(ids) - keep
            marker = self._text_ids(f"\n[... {omitted} tokens omitted ...]\n")
            for _ in range(4):
                if len(marker) >= keep - 1:
                    break
                recalculated = len(ids) - (keep - len(marker))
                new_marker = self._text_ids(f"\n[... {recalculated} tokens omitted ...]\n")
                if len(new_marker) == len(marker):
                    break
                marker = new_marker
            if len(marker) + 2 <= keep:
                content_keep = keep - len(marker)
                head = (content_keep + 1) // 2
                tail = content_keep - head
                ids = ids[:head] + marker + (ids[-tail:] if tail else [])
            else:
                head = (keep + 1) // 2
                tail = keep - head
                ids = ids[:head] + (ids[-tail:] if tail else [])
        return [self.state_prefix_id] + ids, truncated

    def format_prompt(self, state, question, options, max_seq_len=1024):
        suffix, ends, query_end = self.question_ids(question, options)
        prefix, truncated = self.state_ids(state, max_seq_len - len(suffix))
        return Prompt(prefix, suffix, ends, query_end, truncated)

    def embeddings(self, state_hidden):
        full = self.W_emb(state_hidden.float())
        return {dim: F.normalize(full[..., :dim], dim=-1) for dim in SLICE_DIMS}

    def _run(self, ids, cache=None, use_cache=False):
        inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
        if torch.is_grad_enabled() and not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                return self.backbone(input_ids=inputs, past_key_values=cache, use_cache=use_cache, return_dict=True)
        return self.backbone(input_ids=inputs, past_key_values=cache, use_cache=use_cache, return_dict=True)

    def forward_prompt(self, prompt):
        output = self._run(prompt.input_ids)
        hidden = output.last_hidden_state[0]
        offset = len(prompt.prefix_ids)
        return {
            "logits": self.pointer(hidden[offset + prompt.query_end], hidden[[offset + i for i in prompt.option_ends]]),
            "embeddings": self.embeddings(hidden[offset - 1]),
        }

    def forward(self, state, question, options, max_seq_len=1024):
        return self.forward_prompt(self.format_prompt(state, question, options, max_seq_len))

    def encode_prefix(self, state, max_seq_len=1024, suffix_reserve=0):
        ids, truncated = self.state_ids(state, max_seq_len - suffix_reserve)
        output = self._run(ids, use_cache=True)
        return Prefix(output.past_key_values, ids, self.embeddings(output.last_hidden_state[0, -1]), truncated)

    def question_from_prefix(self, prefix, question, options, isolate=True):
        suffix, ends, query_end = self.question_ids(question, options)
        cache = clone_prefix_cache(prefix.cache) if isolate else prefix.cache
        output = self._run(suffix, cache=cache, use_cache=True)
        hidden = output.last_hidden_state[0]
        return {"logits": self.pointer(hidden[query_end], hidden[ends]),
                "embeddings": prefix.embeddings, "cache": output.past_key_values}

    def equivariant_scores(self, state, question, options, max_seq_len=1024, use_cache=True):
        """Score each option in its own fork; reordering simply reorders scores.

        Without caching, independent full forwards also support checkpointed LoRA
        training. A shared maximum suffix budget makes state truncation symmetric.
        """
        suffixes = [self.question_ids(question, [option]) for option in options]
        reserve = max(len(item[0]) for item in suffixes)
        if use_cache:
            prefix = self.encode_prefix(state, max_seq_len, reserve)
            scores = [self.question_from_prefix(prefix, question, [option])["logits"][0] for option in options]
            return torch.stack(scores), prefix.embeddings
        ids, truncated = self.state_ids(state, max_seq_len - reserve)
        outputs = [self.forward_prompt(Prompt(ids, suffix, ends, end, truncated)) for suffix, ends, end in suffixes]
        return torch.stack([out["logits"][0] for out in outputs]), outputs[0]["embeddings"]

    def embed_state(self, state, max_seq_len=1024):
        ids, _ = self.state_ids(state, max_seq_len)
        return self.embeddings(self._run(ids).last_hidden_state[0, -1])

    def configure_training(self, mode, gradient_checkpointing=True):
        if mode not in ("exp_a", "exp_b"):
            raise ValueError("mode must be exp_a or exp_b")
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(mode == "exp_b" and "lora_" in name)
        if mode == "exp_b" and gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        self.pointer.requires_grad_(True)
        self.W_emb.requires_grad_(True)
        self.train()
        # Dropout would otherwise make option permutation comparisons stochastic.
        self.backbone.eval() if mode == "exp_a" else self.backbone.train()
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def save_checkpoint(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        tensors = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()
                   if name in trainable or name.startswith(("pointer.", "W_emb."))}
        torch.save({"tensors": tensors, "metadata": metadata, "temperature": self.temperature,
                    "special_ids": self.special_ids}, directory / "model.pt")

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["special_ids"] != self.special_ids:
            raise ValueError("Checkpoint boundary token IDs do not match tokenizer")
        current = self.state_dict()
        unexpected = set(payload["tensors"]) - set(current)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint tensors: {sorted(unexpected)}")
        current.update(payload["tensors"])
        self.load_state_dict(current, strict=True)
        return payload["metadata"]


class ByteTokenizer:
    """Deterministic synthetic tokenizer reserving Qwen-compatible special tokens."""
    eos_token_id = 0

    def __init__(self):
        self.vocab = {f"byte_{i}": i + 1 for i in range(256)} | {"<eos>": 0}
        for token in SPECIAL_TOKENS:
            self.vocab[token] = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, mapping):
        for token in mapping["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, text, add_special_tokens=False):
        return [value + 1 for value in text.encode("utf-8")]


def synthetic_model(seed=42, with_lora=False):
    from transformers import Qwen3_5Config, Qwen3_5TextModel

    torch.manual_seed(seed)
    config = Qwen3_5Config(text_config={
        "vocab_size": 272, "hidden_size": 32, "intermediate_size": 48,
        "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 2,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4, "eos_token_id": 0,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5},
    }).text_config
    config._attn_implementation = "eager"
    backbone = Qwen3_5TextModel(config)
    if with_lora:
        from peft import LoraConfig, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                  target_modules=["in_proj_qkv", "q_proj", "v_proj"], task_type="FEATURE_EXTRACTION"))
    return DualHeadDecisionEmbedder(backbone, ByteTokenizer(), head_dim=16).eval()
