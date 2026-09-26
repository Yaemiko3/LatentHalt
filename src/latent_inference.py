from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache


@dataclass(frozen=True)
class LatentGenerationBatch:
    answer_token_ids: list[list[int]]
    latent_blocks: list[int]
    halted_by_region: list[bool]
    halt_score_trajectories: list[list[float]]


def _normalize_eos_token_ids(token_ids: int | Iterable[int] | None) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    return {int(token_id) for token_id in token_ids}


def _as_dynamic_cache(cache: Any) -> Cache:
    if isinstance(cache, Cache):
        return cache
    return DynamicCache.from_legacy_cache(cache)


def _validated_backbone(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise TypeError("Latent inference currently requires a Llama-style .model backbone")
    if model.get_input_embeddings() is None or model.get_output_embeddings() is None:
        raise TypeError("Latent inference requires input and output embedding modules")
    return backbone


@torch.inference_mode()
def generate_latent_batch(
    model: nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    think_end_ids: Sequence[int],
    think_region_center: torch.Tensor,
    halt_cosine_threshold: float,
    c_thought: int,
    min_latent_blocks: int,
    max_latent_blocks: int,
    max_new_tokens: int,
    eos_token_ids: int | Iterable[int],
    pad_token_id: int,
) -> LatentGenerationBatch:
    """Generate continuous latent blocks, force ``</think>``, then decode an answer.

    ``input_ids`` must end in the same ``<think>\n`` prefix used during training.
    A latent input is always the preceding position's final-layer hidden state. The
    halt score is evaluated only after a complete ``c_thought`` block.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must be equal rank-2 tensors")
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("Latent inference requires a non-empty batch")
    if not bool(attention_mask[:, -1].bool().all()):
        raise ValueError("Every prompt must end in an active <think> prefix token")
    if c_thought <= 0:
        raise ValueError("c_thought must be positive")
    if min_latent_blocks <= 0 or min_latent_blocks > max_latent_blocks:
        raise ValueError("min_latent_blocks must lie in [1, max_latent_blocks]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not -1.0 <= halt_cosine_threshold <= 1.0:
        raise ValueError("halt_cosine_threshold must lie in [-1, 1]")
    if not think_end_ids:
        raise ValueError("think_end_ids cannot be empty")

    eos_ids = _normalize_eos_token_ids(eos_token_ids)
    if not eos_ids:
        raise ValueError("At least one EOS token ID is required")

    backbone = _validated_backbone(model)
    device = input_ids.device
    batch_size = input_ids.shape[0]
    center = think_region_center.to(device=device, dtype=torch.float32).flatten()
    hidden_size = int(model.config.hidden_size)
    if center.numel() != hidden_size or not bool(torch.isfinite(center).all()):
        raise ValueError("The think-region center is invalid for this model")
    if not bool(center.norm() > 0):
        raise ValueError("The think-region center has zero norm")
    center = F.normalize(center, dim=0)

    attention_mask = attention_mask.to(device=device, dtype=torch.long)
    input_ids = input_ids.to(device=device, dtype=torch.long)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        return_dict=True,
    )
    cache = _as_dynamic_cache(outputs.past_key_values)
    previous_hidden = outputs.last_hidden_state[:, -1, :]

    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    latent_blocks = torch.zeros(batch_size, dtype=torch.long, device=device)
    halted_by_region = torch.zeros(batch_size, dtype=torch.bool, device=device)
    score_trajectories: list[list[float]] = [[] for _ in range(batch_size)]

    for block_index in range(1, max_latent_blocks + 1):
        for _ in range(c_thought):
            active_before_step = active.clone()
            next_attention = active_before_step.long().unsqueeze(1)
            next_position_ids = attention_mask.sum(dim=1, keepdim=True)
            attention_mask = torch.cat((attention_mask, next_attention), dim=1)
            latent_inputs = torch.where(
                active_before_step.unsqueeze(1),
                previous_hidden,
                torch.zeros_like(previous_hidden),
            ).unsqueeze(1)
            outputs = backbone(
                inputs_embeds=latent_inputs,
                attention_mask=attention_mask,
                position_ids=next_position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = _as_dynamic_cache(outputs.past_key_values)
            candidate_hidden = outputs.last_hidden_state[:, -1, :]
            previous_hidden = torch.where(
                active_before_step.unsqueeze(1), candidate_hidden, previous_hidden
            )

        scores = F.normalize(previous_hidden.float(), dim=-1) @ center
        active_rows = active.nonzero(as_tuple=True)[0].tolist()
        scores_cpu = scores.detach().cpu().tolist()
        for row in active_rows:
            score_trajectories[row].append(float(scores_cpu[row]))

        newly_halted = (
            active
            & (block_index >= min_latent_blocks)
            & scores.ge(halt_cosine_threshold)
        )
        latent_blocks[newly_halted] = block_index
        halted_by_region |= newly_halted
        active &= ~newly_halted
        if not bool(active.any()):
            break

    latent_blocks[active] = max_latent_blocks

    # Rows that halted early have masked latent cache slots after their terminal
    # block. They remain invisible to the forced boundary and answer tokens.
    boundary_ids = torch.tensor(
        tuple(int(token_id) for token_id in think_end_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    boundary_length = boundary_ids.shape[1]
    boundary_position_ids = attention_mask.sum(dim=1, keepdim=True) + torch.arange(
        boundary_length, device=device
    ).unsqueeze(0)
    attention_mask = torch.cat(
        (
            attention_mask,
            torch.ones(
                (batch_size, boundary_length), dtype=torch.long, device=device
            ),
        ),
        dim=1,
    )
    outputs = backbone(
        input_ids=boundary_ids,
        attention_mask=attention_mask,
        position_ids=boundary_position_ids,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    cache = _as_dynamic_cache(outputs.past_key_values)
    output_layer = model.get_output_embeddings()
    next_tokens = output_layer(outputs.last_hidden_state[:, -1, :]).argmax(dim=-1)

    generated: list[list[int]] = [[] for _ in range(batch_size)]
    answer_active = torch.ones(batch_size, dtype=torch.bool, device=device)
    eos_tensor = torch.tensor(sorted(eos_ids), dtype=torch.long, device=device)

    for generation_index in range(max_new_tokens):
        active_rows = answer_active.nonzero(as_tuple=True)[0].tolist()
        next_tokens_cpu = next_tokens.detach().cpu().tolist()
        for row in active_rows:
            generated[row].append(int(next_tokens_cpu[row]))

        selected_eos = torch.isin(next_tokens, eos_tensor)
        answer_active &= ~selected_eos
        if generation_index + 1 >= max_new_tokens or not bool(answer_active.any()):
            break

        next_attention = answer_active.long().unsqueeze(1)
        next_position_ids = attention_mask.sum(dim=1, keepdim=True)
        attention_mask = torch.cat((attention_mask, next_attention), dim=1)
        decode_ids = next_tokens.masked_fill(~answer_active, int(pad_token_id)).unsqueeze(1)
        outputs = backbone(
            input_ids=decode_ids,
            attention_mask=attention_mask,
            position_ids=next_position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = _as_dynamic_cache(outputs.past_key_values)
        next_tokens = output_layer(outputs.last_hidden_state[:, -1, :]).argmax(dim=-1)

    return LatentGenerationBatch(
        answer_token_ids=generated,
        latent_blocks=[int(value) for value in latent_blocks.cpu().tolist()],
        halted_by_region=[bool(value) for value in halted_by_region.cpu().tolist()],
        halt_score_trajectories=score_trajectories,
    )
