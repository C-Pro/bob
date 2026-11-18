"""Causal state embeddings and isolated, position-free option decisions."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import itertools
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
BASE_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
KEV_ADAPTER = "jaredpalmer/kev-0.8b"
KEV_ADAPTER_REVISION = "bf75a6a8848ea6960ff2ed108d9ed44c2941174f"


def resolve_cached_snapshot(repo_id: str, revision: str, cache_dir: str | Path | None = None, local_files_only: bool = True) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only))
# Reuse Qwen's native special delimiters aligned with jaredpalmer/kev-0.8b LoRA weights:
# [state_prefix] -> [question] [opt_start ... opt_end]+ [decide].
SPECIAL_TOKENS = ("<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>")
VISION_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")
SLICE_DIMS = (128, 256, 384)

DEFAULT_IMAGE_MIN_PIXELS = 64 * 16 * 16
DEFAULT_IMAGE_MAX_PIXELS = 384 * 16 * 16
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)

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
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None

    @property
    def input_ids(self):
        return self.prefix_ids + self.suffix_ids


@dataclass
class Prefix:
    cache: object
    input_ids: list[int]
    embeddings: dict[int, torch.Tensor]
    truncated: bool
    next_pos: int | None = None
    max_seq_len: int | None = None


def _to_pil_rgb(item) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    if isinstance(item, (str, Path)):
        with Image.open(item) as img:
            return img.convert("RGB")
    if isinstance(item, torch.Tensor):
        t = item.detach().cpu()
        if t.is_floating_point():
            t = t.float()
        arr = t.numpy()
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Expected PyTorch image tensor (C, H, W) with C in {{1, 3, 4}} or (H, W), got {tuple(t.shape)}")
    elif isinstance(item, np.ndarray):
        arr = item
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected NumPy image array (H, W, C) with C in {{1, 3, 4}} or (H, W), got {arr.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(item).__name__}")

    if not np.isfinite(arr).all():
        raise ValueError("Image contains non-finite pixel values")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max(initial=0.0) <= 1.0 and arr.min(initial=0.0) >= 0.0:
            arr = np.round(arr * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _vision_position_ids(start_position: int, grid_thw: torch.Tensor, spatial_merge_size: int, device) -> torch.Tensor:
    llm_grid_t = int(grid_thw[0].item())
    llm_grid_h = int(grid_thw[1].item()) // spatial_merge_size
    llm_grid_w = int(grid_thw[2].item()) // spatial_merge_size
    pos_t = torch.arange(llm_grid_t, device=device)
    pos_h = torch.arange(llm_grid_h, device=device) + start_position
    pos_w = torch.arange(llm_grid_w, device=device) + start_position
    t_grid, h_grid, w_grid = torch.meshgrid(pos_t, pos_h, pos_w, indexing="ij")
    vision_pos = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(3, -1)
    vision_pos[0] += start_position
    return vision_pos


class DualHeadDecisionEmbedder(nn.Module):
    def __init__(
        self,
        backbone,
        tokenizer,
        head_dim=256,
        head_state=None,
        temperature=1.0,
        visual=None,
        vision_config=None,
        image_min_pixels=DEFAULT_IMAGE_MIN_PIXELS,
        image_max_pixels=DEFAULT_IMAGE_MAX_PIXELS,
        image_mean=DEFAULT_IMAGE_MEAN,
        image_std=DEFAULT_IMAGE_STD,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.visual = visual
        self.vision_config = vision_config
        self.image_min_pixels = int(image_min_pixels)
        self.image_max_pixels = int(image_max_pixels)
        self.image_mean = tuple(float(x) for x in image_mean)
        self.image_std = tuple(float(x) for x in image_std)
        self._image_processor = None
        if self.visual is not None:
            for param in self.visual.parameters():
                param.requires_grad_(False)
            self.visual.eval()
        self.pointer = PointerHead(backbone.config.hidden_size, head_dim)
        self.W_emb = nn.Linear(backbone.config.hidden_size, 384, bias=False, dtype=torch.float32)
        self.temperature = float(temperature)
        self.adapter_tensor_count = 0
        if head_state is not None:
            self.pointer.load_state_dict(head_state, strict=True)
        self._register_boundary_tokens()
        self.pointer.to(self.device)
        self.W_emb.to(self.device)
        if self.visual is not None:
            self.visual.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.visual is not None:
            self.visual.eval()
        return self

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

        self.vision_ids = {}
        if self.visual is not None or all(token in vocab for token in VISION_TOKENS):
            missing_vision = [token for token in VISION_TOKENS if token not in vocab]
            if missing_vision:
                raise ValueError(f"Tokenizer missing required vision tokens: {missing_vision}")
            self.vision_ids = {token: self.tokenizer.convert_tokens_to_ids(token) for token in VISION_TOKENS}
            for token, token_id in self.vision_ids.items():
                if not 0 <= token_id < embedding.num_embeddings:
                    raise ValueError(f"Vision token {token} ID {token_id} out of embedding bounds {embedding.num_embeddings}")
        self.vision_start_id = self.vision_ids.get("<|vision_start|>")
        self.vision_end_id = self.vision_ids.get("<|vision_end|>")
        self.image_token_id = self.vision_ids.get("<|image_pad|>")
        self.video_token_id = self.vision_ids.get("<|video_pad|>")

    def _get_image_processor(self):
        if self._image_processor is None:
            from transformers import Qwen2VLImageProcessorPil

            patch_size = getattr(self.vision_config, "patch_size", 16) if self.vision_config is not None else 16
            merge_size = getattr(self.vision_config, "spatial_merge_size", 2) if self.vision_config is not None else 2
            temporal_patch_size = getattr(self.vision_config, "temporal_patch_size", 2) if self.vision_config is not None else 2
            self._image_processor = Qwen2VLImageProcessorPil(
                do_resize=True,
                size={"shortest_edge": self.image_min_pixels, "longest_edge": self.image_max_pixels},
                do_rescale=True,
                rescale_factor=1.0 / 255.0,
                do_normalize=True,
                image_mean=list(self.image_mean),
                image_std=list(self.image_std),
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
            )
        return self._image_processor

    def preprocess_images(self, images):
        if images is None:
            return None, None, []
        if isinstance(images, (Image.Image, str, Path, torch.Tensor, np.ndarray)):
            items = [images]
        else:
            items = list(images)
        if not items:
            return None, None, []
        if self.visual is None or self.image_token_id is None:
            raise ValueError("Images provided but vision encoder is not attached")
        pil_images = [_to_pil_rgb(item) for item in items]
        processor = self._get_image_processor()
        processed = processor(images=pil_images, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]
        merge_size = getattr(self.vision_config, "spatial_merge_size", 2) if self.vision_config is not None else 2
        token_counts = [int(t * h * w) // (merge_size ** 2) for t, h, w in image_grid_thw.tolist()]
        return pixel_values, image_grid_thw, token_counts

    @classmethod
    def from_pretrained(cls, device="cpu", dtype=torch.float32, trainable_adapter=False, include_vision=False,
                        base_revision=BASE_MODEL_REVISION, adapter_revision=KEV_ADAPTER_REVISION, cache_dir=None):
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration

        base_path = str(resolve_cached_snapshot(BASE_MODEL, base_revision, cache_dir=cache_dir, local_files_only=True))
        adapter_path = resolve_cached_snapshot(KEV_ADAPTER, adapter_revision, cache_dir=cache_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        visual = None
        vision_config = None
        if include_vision:
            cond_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                base_path, dtype=dtype, local_files_only=True, attn_implementation="eager")
            visual = cond_model.model.visual
            vision_config = cond_model.config.vision_config
            text_model = cond_model.model.language_model
            del cond_model
        else:
            causal_model = AutoModelForCausalLM.from_pretrained(
                base_path, dtype=dtype, local_files_only=True, attn_implementation="eager")
            text_model = causal_model.model
            del causal_model
        # Kev keys are base_model.model.layers.*, not base_model.model.model.layers.*.
        backbone = PeftModel.from_pretrained(text_model, str(adapter_path),
                                             is_trainable=trainable_adapter, local_files_only=True)
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
                    head_state=head["head"], temperature=head["temperature"],
                    visual=visual, vision_config=vision_config)
        model.adapter_tensor_count = len(loaded)
        model.to(device=device)
        if model.visual is not None:
            model.visual.eval()
        return model

    def _text_ids(self, text):
        # Treat user-supplied marker spellings as text, never structural or vision tokens.
        for token in SPECIAL_TOKENS + VISION_TOKENS:
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

    def prepare_state(self, state, budget, images=None):
        pixel_values, image_grid_thw, token_counts = self.preprocess_images(images)
        image_tokens = []
        for count in token_counts:
            image_tokens.append(self.vision_start_id)
            image_tokens.extend([self.image_token_id] * count)
            image_tokens.append(self.vision_end_id)
        text_budget = budget - len(image_tokens)
        if text_budget < 1:
            raise ValueError("Images and suffix exceed sequence budget")
        ids = self._text_ids(state)
        keep = text_budget - 1
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
        return [self.state_prefix_id] + image_tokens + ids, truncated, pixel_values, image_grid_thw

    def state_ids(self, state, budget):
        ids, truncated, _, _ = self.prepare_state(state, budget)
        return ids, truncated

    def format_prompt(self, state, question, options, max_seq_len=1024, images=None):
        suffix, ends, query_end = self.question_ids(question, options)
        prefix, truncated, pixel_values, image_grid_thw = self.prepare_state(
            state, max_seq_len - len(suffix), images=images
        )
        return Prompt(prefix, suffix, ends, query_end, truncated, pixel_values, image_grid_thw)

    def embeddings(self, state_hidden):
        full = self.W_emb(state_hidden.float())
        return {dim: F.normalize(full[..., :dim], dim=-1) for dim in SLICE_DIMS}

    def _compute_position_ids(self, input_ids_tensor: torch.Tensor, image_grid_thw: torch.Tensor | None = None):
        seq_len = input_ids_tensor.shape[1]
        device = input_ids_tensor.device
        text_pos = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len)
        if image_grid_thw is None or self.image_token_id is None:
            mrope = text_pos.expand(3, 1, seq_len)
            return torch.cat([text_pos, mrope], dim=0), seq_len

        spatial_merge_size = getattr(self.vision_config, "spatial_merge_size", 2) if self.vision_config is not None else 2
        is_image = (input_ids_tensor[0] == self.image_token_id).tolist()
        grid_iter = iter(image_grid_thw)
        current_pos = 0
        chunks = []
        for modality, group in itertools.groupby(enumerate(is_image), key=lambda pair: int(pair[1])):
            items = list(group)
            span_len = len(items)
            if modality == 0:
                chunks.append(torch.arange(span_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + current_pos)
                current_pos += span_len
            else:
                grid_thw = next(grid_iter)
                vis_pos = _vision_position_ids(current_pos, grid_thw, spatial_merge_size, device=device)
                if vis_pos.shape[1] != span_len:
                    raise ValueError(f"Image token span length {span_len} != expected merged grid length {vis_pos.shape[1]}")
                chunks.append(vis_pos)
                current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // spatial_merge_size
        mrope = torch.cat(chunks, dim=1).view(3, 1, seq_len)
        return torch.cat([text_pos, mrope], dim=0), int(mrope.max().item()) + 1

    def _build_inputs_embeds(self, input_ids_tensor: torch.Tensor, pixel_values: torch.Tensor | None, image_grid_thw: torch.Tensor | None):
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids_tensor)
        if pixel_values is None:
            return inputs_embeds
        if self.visual is None or self.image_token_id is None:
            raise ValueError("pixel_values provided without an attached vision encoder")
        vis_param = next(self.visual.parameters(), None)
        vis_device = vis_param.device if vis_param is not None else self.device
        vis_dtype = vis_param.dtype if vis_param is not None else inputs_embeds.dtype
        with torch.no_grad():
            vision_out = self.visual(
                pixel_values.to(device=vis_device, dtype=vis_dtype),
                grid_thw=image_grid_thw.to(device=vis_device),
                return_dict=True,
            )
            image_embeds = vision_out.pooler_output
        image_embeds = image_embeds.detach().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_mask = input_ids_tensor == self.image_token_id
        expected_tokens = int(image_mask.sum().item())
        if image_embeds.ndim != 2 or image_embeds.shape[0] != expected_tokens or image_embeds.shape[1] != inputs_embeds.shape[-1]:
            raise ValueError(
                f"Vision feature shape {tuple(image_embeds.shape)} does not match "
                f"({expected_tokens}, {inputs_embeds.shape[-1]})"
            )
        return inputs_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds)

    def _run(self, ids, cache=None, use_cache=False, pixel_values=None, image_grid_thw=None, position_ids=None):
        input_ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        kwargs = {"past_key_values": cache, "use_cache": use_cache, "return_dict": True}
        if pixel_values is not None:
            if position_ids is None:
                position_ids, _ = self._compute_position_ids(input_ids_tensor, image_grid_thw)
            kwargs["inputs_embeds"] = self._build_inputs_embeds(input_ids_tensor, pixel_values, image_grid_thw)
            kwargs["position_ids"] = position_ids
        else:
            kwargs["input_ids"] = input_ids_tensor
            if position_ids is not None:
                kwargs["position_ids"] = position_ids

        if torch.is_grad_enabled() and not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                return self.backbone(**kwargs)
        return self.backbone(**kwargs)

    def forward_prompt(self, prompt):
        output = self._run(prompt.input_ids, pixel_values=prompt.pixel_values, image_grid_thw=prompt.image_grid_thw)
        hidden = output.last_hidden_state[0]
        offset = len(prompt.prefix_ids)
        return {
            "logits": self.pointer(hidden[offset + prompt.query_end], hidden[[offset + i for i in prompt.option_ends]]),
            "embeddings": self.embeddings(hidden[offset - 1]),
        }

    def forward(self, state, question, options, max_seq_len=1024, images=None):
        return self.forward_prompt(self.format_prompt(state, question, options, max_seq_len, images=images))

    def encode_prefix(self, state, max_seq_len=1024, suffix_reserve=0, images=None):
        ids, truncated, pixel_values, image_grid_thw = self.prepare_state(
            state, max_seq_len - suffix_reserve, images=images
        )
        input_ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        position_ids, next_pos = self._compute_position_ids(input_ids_tensor, image_grid_thw)
        output = self._run(
            ids,
            use_cache=True,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )
        return Prefix(
            output.past_key_values,
            ids,
            self.embeddings(output.last_hidden_state[0, -1]),
            truncated,
            next_pos=next_pos,
            max_seq_len=max_seq_len,
        )

    def question_from_prefix(self, prefix, question, options, isolate=True):
        suffix, ends, query_end = self.question_ids(question, options)
        if prefix.max_seq_len is not None and len(prefix.input_ids) + len(suffix) > prefix.max_seq_len:
            raise ValueError("Question and options exceed prefix sequence budget")
        cache = clone_prefix_cache(prefix.cache) if isolate else prefix.cache
        phys_start = len(prefix.input_ids)
        mrope_start = prefix.next_pos if prefix.next_pos is not None else phys_start
        text_pos = torch.arange(phys_start, phys_start + len(suffix), device=self.device, dtype=torch.long).view(1, 1, -1)
        mrope_pos = torch.arange(mrope_start, mrope_start + len(suffix), device=self.device, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
        position_ids = torch.cat([text_pos, mrope_pos], dim=0)
        output = self._run(suffix, cache=cache, use_cache=True, position_ids=position_ids)
        hidden = output.last_hidden_state[0]
        return {"logits": self.pointer(hidden[query_end], hidden[ends]),
                "embeddings": prefix.embeddings, "cache": output.past_key_values}

    def equivariant_scores(self, state, question, options, max_seq_len=1024, use_cache=True, images=None):
        """Score each option in its own fork; reordering simply reorders scores.

        Without caching, independent full forwards also support checkpointed LoRA
        training. A shared maximum suffix budget makes state truncation symmetric.
        """
        suffixes = [self.question_ids(question, [option]) for option in options]
        reserve = max(len(item[0]) for item in suffixes)
        if use_cache:
            prefix = self.encode_prefix(state, max_seq_len, reserve, images=images)
            scores = [self.question_from_prefix(prefix, question, [option])["logits"][0] for option in options]
            return torch.stack(scores), prefix.embeddings
        ids, truncated, pixel_values, image_grid_thw = self.prepare_state(
            state, max_seq_len - reserve, images=images
        )
        outputs = [
            self.forward_prompt(Prompt(ids, suffix, ends, end, truncated, pixel_values, image_grid_thw))
            for suffix, ends, end in suffixes
        ]
        return torch.stack([out["logits"][0] for out in outputs]), outputs[0]["embeddings"]

    def state_hidden(self, state, max_seq_len=1024, images=None):
        ids, _, pixel_values, image_grid_thw = self.prepare_state(state, max_seq_len, images=images)
        return self._run(ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw).last_hidden_state[0, -1]

    def embed_state(self, state, max_seq_len=1024, detach_backbone=False, images=None):
        if detach_backbone:
            with torch.no_grad():
                hidden = self.state_hidden(state, max_seq_len, images=images)
            return self.embeddings(hidden.detach())
        return self.embeddings(self.state_hidden(state, max_seq_len, images=images))

    def configure_training(self, mode, gradient_checkpointing=True):
        if mode not in ("exp_a", "exp_b", "emb_only"):
            raise ValueError("mode must be exp_a, exp_b, or emb_only")
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(mode == "exp_b" and "lora_" in name)
        if self.visual is not None:
            for parameter in self.visual.parameters():
                parameter.requires_grad_(False)
        if mode == "exp_b" and gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        self.pointer.requires_grad_(mode != "emb_only")
        self.W_emb.requires_grad_(True)
        self.train()
        # Dropout would otherwise make option permutation comparisons stochastic.
        self.backbone.eval() if mode in ("exp_a", "emb_only") else self.backbone.train()
        if self.visual is not None:
            self.visual.eval()
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def save_checkpoint(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        trainable = {name for name, param in self.named_parameters() if param.requires_grad}
        tensors = {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("visual.")
            and (name in trainable or "lora_" in name or name.startswith(("pointer.", "W_emb.")))
        }
        enriched_meta = dict(metadata)
        enriched_meta["include_vision"] = self.visual is not None
        if self.visual is not None:
            enriched_meta["vision_preprocess"] = {
                "min_pixels": self.image_min_pixels,
                "max_pixels": self.image_max_pixels,
                "mean": list(self.image_mean),
                "std": list(self.image_std),
            }
        torch.save({"tensors": tensors, "metadata": enriched_meta, "temperature": self.temperature,
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
        if "temperature" in payload:
            self.temperature = float(payload["temperature"])
        meta = payload.get("metadata") or {}
        vis_prep = meta.get("vision_preprocess")
        if isinstance(vis_prep, dict):
            self.image_min_pixels = int(vis_prep.get("min_pixels", self.image_min_pixels))
            self.image_max_pixels = int(vis_prep.get("max_pixels", self.image_max_pixels))
            if "mean" in vis_prep:
                self.image_mean = tuple(float(x) for x in vis_prep["mean"])
            if "std" in vis_prep:
                self.image_std = tuple(float(x) for x in vis_prep["std"])
            self._image_processor = None
        return meta


class ByteTokenizer:
    """Deterministic synthetic tokenizer reserving Qwen-compatible special and vision tokens."""
    eos_token_id = 0

    def __init__(self):
        self.vocab = {f"byte_{i}": i + 1 for i in range(256)} | {"<eos>": 0}
        for token in SPECIAL_TOKENS + VISION_TOKENS:
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

    def decode(self, ids, skip_special_tokens=False):
        inv = {idx: tok for tok, idx in self.vocab.items()}
        out = []
        raw_bytes = bytearray()
        for idx in ids:
            tok = inv.get(int(idx), "")
            if tok.startswith("byte_"):
                raw_bytes.append(int(tok[5:]))
                continue
            if raw_bytes:
                out.append(raw_bytes.decode("utf-8", errors="replace"))
                raw_bytes.clear()
            if not skip_special_tokens and tok:
                out.append(tok)
        if raw_bytes:
            out.append(raw_bytes.decode("utf-8", errors="replace"))
        return "".join(out)


class SyntheticVisionModel(nn.Module):
    """Tiny deterministic spatial vision encoder stub matching Qwen3_5VisionModel output contract."""

    def __init__(self, patch_dim: int, out_hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.proj = nn.Linear(patch_dim * (spatial_merge_size ** 2), out_hidden_size, bias=True)
        self.spatial_proj = nn.Linear(2, out_hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, return_dict: bool = True):
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        m = self.spatial_merge_size
        merged_chunks = []
        offset = 0
        for t, h, w in grid_thw.tolist():
            count = t * h * w
            patches = hidden_states[offset:offset + count]
            offset += count
            # Qwen2VLImageProcessor orders patches in (t, h//m, w//m, m, m) blocks.
            grouped = patches.reshape(t * (h // m) * (w // m), (m * m) * patches.shape[-1])
            coords_h = torch.linspace(-1.0, 1.0, h // m, device=patches.device, dtype=patches.dtype)
            coords_w = torch.linspace(-1.0, 1.0, w // m, device=patches.device, dtype=patches.dtype)
            gh, gw = torch.meshgrid(coords_h, coords_w, indexing="ij")
            coords = torch.stack([gh, gw], dim=-1).reshape((h // m) * (w // m), 2).repeat(t, 1)
            merged = torch.tanh(self.proj(grouped.to(self.proj.weight.dtype))) + 0.1 * self.spatial_proj(coords.to(self.proj.weight.dtype))
            merged_chunks.append(merged)
        pooler_output = torch.cat(merged_chunks, dim=0)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooler_output)


def synthetic_model(seed=42, with_lora=False, include_vision=True):
    from transformers import Qwen3_5Config, Qwen3_5TextModel

    torch.manual_seed(seed)
    full_config = Qwen3_5Config(
        text_config={
            "vocab_size": 272, "hidden_size": 32, "intermediate_size": 48,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "eos_token_id": 0,
            "rope_parameters": {
                "rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                "mrope_interleaved": True, "mrope_section": [2, 1, 1],
            },
        },
        vision_config={
            "depth": 1, "hidden_size": 24, "intermediate_size": 32, "num_heads": 2,
            "in_channels": 3, "patch_size": 16, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "out_hidden_size": 32,
        },
    )
    config = full_config.text_config
    config._attn_implementation = "eager"
    backbone = Qwen3_5TextModel(config)
    if with_lora:
        from peft import LoraConfig, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                  target_modules=["in_proj_qkv", "q_proj", "v_proj"], task_type="FEATURE_EXTRACTION"))
    visual = None
    vision_config = None
    if include_vision:
        vision_config = full_config.vision_config
        patch_dim = vision_config.in_channels * vision_config.temporal_patch_size * (vision_config.patch_size ** 2)
        with torch.random.fork_rng():
            torch.manual_seed(seed + 1000)
            visual = SyntheticVisionModel(patch_dim, config.hidden_size, vision_config.spatial_merge_size)
    return DualHeadDecisionEmbedder(
        backbone, ByteTokenizer(), head_dim=16, visual=visual, vision_config=vision_config
    ).eval()
