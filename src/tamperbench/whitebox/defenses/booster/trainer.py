"""Booster Trainer implementing Algorithm 1.

Reference:
    Huang et al., "Booster: Tackling Harmful Fine-tuning for Large Language Models
    via Attenuating Harmful Perturbation", arXiv:2409.01586v4, 2025.
    https://github.com/git-disl/Booster
"""

from collections.abc import Iterator
from typing import Any, cast

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from typing_extensions import override


class BoosterTrainer(Trainer):
    """Custom trainer implementing Booster Algorithm 1.

    Algorithm 1 from the paper (Section 4, page 5):
    For each step t:
        1. Sample alignment batch (x_t, y_t)
        2. Sample harmful batch (x'_t, y'_t)
        3. Evaluate gradient grad_f(w_t) on alignment data
        4. Evaluate gradient grad_h(w_t) on harmful data
        5. Evaluate gradient grad_h(w_t - alpha * grad_h(w_t) / ||grad_h(w_t)||) on harmful data
        6. g(w_t) = grad_f(w_t) + lambda * (grad_h(w_t) - grad_h(w_t - alpha * grad_h(w_t) / ||grad_h(w_t)||))
        7. w_{t+1} = w_t - eta * g(w_t)

    The optimizer handles step 7, so we only need to set up the combined gradient
    in param.grad for each parameter.
    """

    harmful_dataset: datasets.Dataset
    lambda_: float
    alpha: float

    def __init__(
        self,
        harmful_dataset: datasets.Dataset,
        lambda_: float,
        alpha: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize BoosterTrainer.

        Args:
            harmful_dataset: Dataset of harmful prompt-answer pairs.
            lambda_: Regularizer intensity (lambda in paper, default 5).
            alpha: Inner step size for perturbation (default 0.1).
            *args: Additional arguments for Trainer.
            **kwargs: Additional keyword arguments for Trainer.
        """
        super().__init__(*args, **kwargs)
        self.harmful_dataset = harmful_dataset
        self.lambda_ = lambda_
        self.alpha = alpha
        self._harmful_dataloader: DataLoader[dict[str, torch.Tensor]] | None = None
        self._harmful_iter: Iterator[dict[str, torch.Tensor]] | None = None
        self._booster_step_count: int = 0

    def _get_harmful_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Create dataloader for harmful dataset.

        Returns:
            DataLoader for harmful data with random sampling.
        """
        if self._harmful_dataloader is not None:
            return self._harmful_dataloader

        sampler = RandomSampler(self.harmful_dataset)

        # Construct DataLoader with explicit keyword arguments.
        # datasets.Dataset is not generic over torch's Dataset[_T_co], and
        # seed_worker's signature doesn't match _worker_init_fn_t exactly.
        dataloader = DataLoader(
            self.harmful_dataset,  # pyright: ignore[reportArgumentType]  # datasets.Dataset vs torch Dataset
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            sampler=sampler,
            drop_last=self.args.dataloader_drop_last,
            worker_init_fn=seed_worker,  # pyright: ignore[reportArgumentType]  # transformers typing mismatch
        )

        prepared = self.accelerator.prepare(dataloader)
        # accelerator.prepare on a DataLoader always returns a DataLoader
        self._harmful_dataloader = cast(DataLoader[dict[str, torch.Tensor]], prepared)
        return self._harmful_dataloader

    def _sample_harmful_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch from the harmful dataset.

        Returns:
            Dict with input_ids, labels, attention_mask tensors.
        """
        if self._harmful_iter is None:
            self._harmful_iter = iter(self._get_harmful_dataloader())

        try:
            batch = next(self._harmful_iter)
        except StopIteration:
            # Reset iterator when exhausted
            self._harmful_iter = iter(self._get_harmful_dataloader())
            batch = next(self._harmful_iter)

        return batch

    def _compute_grad_norm(self, grads: dict[str, torch.Tensor], eps: float = 1e-7) -> float:
        """Compute L2 norm of gradients across all parameters.

        This matches the _grad_norm method from the paper's implementation:
        ||grad|| = sqrt(sum(||grad_i||^2))

        Returns a Python float so the result is device-agnostic (gradients may
        live on different GPUs when using device_map="auto").

        Args:
            grads: Dictionary mapping parameter names to gradient tensors.
            eps: Small constant for numerical stability.

        Returns:
            Gradient norm as a Python float.
        """
        norms = [grads[name].norm(p=2) for name in grads if grads[name] is not None]  # pyright: ignore[reportUnnecessaryComparison]  # defensive guard
        if not norms:
            return eps
        # Move all norms to the same device before stacking, since with
        # device_map="auto" parameters (and their gradients) may reside on
        # different GPUs.
        device = norms[0].device
        total_norm = torch.norm(torch.stack([n.to(device) for n in norms]), p=2)
        return total_norm.item() + eps

    @override
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute one training step of Booster Algorithm 1.

        This is a direct port of the reference implementation from:
        https://github.com/git-disl/Booster/blob/main/trainer.py

        The algorithm:
        1. Compute grad_h(w) on harmful data
        2. Perturb weights: w' = w - alpha * grad_h(w) / ||grad_h(w)||
        3. Compute grad_h(w') on harmful data at perturbed weights
        4. Restore weights
        5. Compute grad_f(w) on alignment data
        6. Combine: param.grad = grad_f + lambda * (grad_h - grad_h')

        Args:
            model: The model being trained.
            inputs: Batch from alignment dataset.
            num_items_in_batch: Number of items (unused, for compatibility).

        Returns:
            Detached loss tensor.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self._prepare_inputs(self._sample_harmful_batch())

        # Step 1: Compute gradient on harmful data - grad_h(w)
        # compute_loss with default return_outputs=False returns a scalar Tensor.
        with self.compute_loss_context_manager():
            loss = cast(torch.Tensor, self.compute_loss(model, harmful_inputs))
        self.accelerator.backward(loss)
        stored_grads = {
            name: param.grad.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        }
        model.zero_grad()

        # Step 2: Perturb weights - w' = w - alpha * grad_h(w) / ||grad_h(w)||
        with torch.no_grad():
            grad_norm = self._compute_grad_norm(stored_grads)
        for name, param in model.named_parameters():
            if param.requires_grad and name in stored_grads:
                param.data -= self.alpha * stored_grads[name] / grad_norm

        # Step 3: Compute gradient at perturbed weights - grad_h(w')
        with self.compute_loss_context_manager():
            loss2 = cast(torch.Tensor, self.compute_loss(model, harmful_inputs))
        self.accelerator.backward(loss2)
        perturb_grads = {
            name: param.grad.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        }
        model.zero_grad()

        # Step 4: Restore weights
        for name, param in model.named_parameters():
            if param.requires_grad and name in stored_grads:
                param.data += self.alpha * stored_grads[name] / grad_norm

        # Step 5: Compute gradient on alignment data - grad_f(w)
        with self.compute_loss_context_manager():
            loss3 = cast(torch.Tensor, self.compute_loss(model, inputs))
        self.accelerator.backward(loss3)

        # Step 6: Combine gradients - g = grad_f + lambda * (grad_h - grad_h')
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            has_harmful = name in stored_grads and name in perturb_grads
            if param.grad is not None and has_harmful:
                param.grad.data += self.lambda_ * (stored_grads[name] - perturb_grads[name])
            elif param.grad is None and has_harmful:
                param.grad = self.lambda_ * (stored_grads[name] - perturb_grads[name])

        # Log periodically
        self._booster_step_count += 1
        if self._booster_step_count % 100 == 0:
            print(
                f"Step {self._booster_step_count}: grad_norm={grad_norm:.4f}, "
                f"loss_change={loss - loss2:.4f}, harmful_loss={loss:.4f}"
            )

        # NOTE: gradient_accumulation_steps > 1 is not supported by the Booster
        # training_step. The manually composed gradients in step 6 would be
        # overwritten each micro-step rather than accumulated.
        return loss3.detach() / self.args.gradient_accumulation_steps
