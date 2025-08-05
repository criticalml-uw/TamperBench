"""Soft Prompt Optimization for LLM Embedding Attack.

Code heavily refactored from:
https://github.com/samuelsimko/crl-llm-defense/blob/main/evaluation/embedding_attack/embedding_attack.py
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import tqdm
import transformers
from transformers import set_seed

logger = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class SoftOptConfig:
    """Configuration for the soft prompt optimization process.

    Attributes:
        num_steps: Number of optimization steps allowed.
        optim_str_init: The string intialization for optimization if randomly initialized.
        num_tokens: Number of tokens for the part of the prompt that is being optimized if randomly intialized.
        lr: Learning rate of optimizer.
        early_stop_loss: The loss threshold to trigger early stopping (i.e. if lower than this, it is stopped).
        add_space_before_target: Boolean flag to add a space before target section of prompt [query] [optimized part] [target].
        device: The CUDA device recognize-able by torch - generally in the form of 'cuda:0', etc.
        verbose: Boolean flag to enable outputting of losses in the soft prompt optimization process.
        extra_field: Allows capturing of extra keys not mentioned in above attributes.
    """

    num_steps: int = 100
    optim_str_init: str = "x " * 10
    rand_init: bool = False
    num_tokens: int = 10
    lr: float = 0.01
    early_stop_loss: float = None
    add_space_before_target: bool = False
    device: str = "cuda:0"
    seed: int = None
    verbose: bool = False
    extra_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, int | str | float | bool | dict | None]) -> None:
        """Populate a SoftOptConfig dataclass object from a dictionary.

        Args:
            data: A dictionary containing key value mappings corresponding to dataclass attributes.
        """
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        init_data = {k: v for k, v in data.items() if k in known_keys}
        extra_data = {k: v for k, v in data.items() if k not in known_keys}

        if "str_length" in extra_data and "optim_str_init" not in init_data:
            init_data["optim_str_init"] = "x " * extra_data["str_length"]
            logger.info(
                f"Converted str_length={extra_data['str_length']} to optim_str_init.",
            )

        init_data["extra_fields"] = data

        return cls(**init_data)


@dataclass
class SoftOptResult:
    """Result of the soft prompt optimization.

    Attributes:
        losses: The losses logged during the soft prompt optimization process.
        optim_embeds: The optimized embeddings post-attack.
        input_embeds: The embeddings to be input to the model which includes query, optimized embedding and target.
    """

    losses: list[float]
    optim_embeds: torch.Tensor
    input_embeds: torch.Tensor


def run_soft_opt(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: str | list[dict],
    target: str,
    config: SoftOptConfig,
) -> SoftOptResult:
    """Generate a single optimized string using soft-prompt optimization.

    Args:
        model: The model to optimize on.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The configuration to use.

    Returns:
        A SoftOptResult object that contains losses and the optimized strings.
    """
    model.enable_input_require_grads()

    if config.seed is not None:
        set_seed(config.seed)
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if not any("{optim_str}" in d["content"] for d in messages):
        messages[-1]["content"] = (
            messages[-1]["content"] + "{optim_str}"
        )  # only this one? why not all of them

    # model = model.cuda()

    template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(
            tokenizer.bos_token,
            "",
        )  # Remove the BOS token -- this will get added when tokenizing, if necessary

    before_str, after_str = template.split("{optim_str}")
    target = " " + target if config.add_space_before_target else target

    before_ids = tokenizer([before_str], padding=False)["input_ids"]
    after_ids, target_ids = [
        tokenizer([_str], add_special_tokens=False)["input_ids"]
        for _str in (after_str, target)
    ]
    before_ids, after_ids, target_ids = [
        torch.tensor(ids, device=config.device)
        for ids in (before_ids, after_ids, target_ids)
    ]

    embedding_layer = model.get_input_embeddings()
    before_embeds, after_embeds, target_embeds = [
        embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
    ]

    # How important is KV-Caching to this algorithm?
    # with torch.no_grad():
    #     output = model(inputs_embeds=before_embeds, use_cache=True)
    #     prefix_cache = output.past_key_values

    optim_embeds = torch.randn(
        (1, config.num_tokens, model.config.hidden_size),
        device=config.device,
    ).requires_grad_()

    if not config.rand_init:
        optim_ids = tokenizer(
            config.optim_str_init,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].cuda()
        optim_embeds = embedding_layer(optim_ids).detach().clone().requires_grad_()

    optimizer = torch.optim.Adam([optim_embeds], lr=config.lr)

    losses = []
    for i in tqdm.tqdm(range(config.num_steps)):
        optimizer.zero_grad()
        input_embeds = torch.cat(
            [optim_embeds, after_embeds.detach(), target_embeds.detach()],
            dim=1,
        )

        output = model(inputs_embeds=input_embeds, output_hidden_states=True)
        # output = model(inputs_embeds=input_embeds, past_key_values=prefix_cache, output_hidden_states=True)

        shift = (
            input_embeds.shape[1] - target_ids.shape[1]
        )  # shift logits so token n-1 predicts token n
        shift_logits = output.logits[
            ...,
            shift - 1 : -1,
            :,
        ].contiguous()  # (1, num_target_ids, vocab_size)

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_ids.view(-1),
        )
        loss_float = loss.item()
        losses.append(loss_float)

        if config.verbose:
            logger.info(f"Iter: {i} Loss: {loss_float}")

        if config.early_stop_loss and loss_float < config.early_stop_loss:
            break

        loss.backward()
        # loss.backward(retain_graph=True) # <-- is there a way to bypass this while KV-Caching
        optimizer.step()

    return SoftOptResult(
        losses=losses,
        optim_embeds=optim_embeds.cpu(),
        input_embeds=torch.cat(
            [before_embeds, optim_embeds, after_embeds],
            dim=1,
        ).cpu(),
    )
