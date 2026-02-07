"""Ported from https://github.com/aengusl/latent-adversarial-training/."""

import torch
from typing_extensions import override


class LowRankAdversary(torch.nn.Module):
    """Adversary parameterized by a low-rank MLP (LoRA-style).

    This adversary applies a low-rank perturbation to the input activations
    by projecting down to a low-dimensional space (via `lora_A`) and then
    projecting back up (via `lora_B`), before adding the perturbation to
    the original input.
    """

    def __init__(
        self,
        dim: int,
        rank: int,
        device: str,
        bias: bool = False,
        zero_init: bool = True,
    ):
        """Initialize LowRankAdversary.

        Args:
        dim: Dimensionality of the input activations.
        rank: Low-rank dimension for the intermediate projection.
        device: Device on which to place the model.
        bias: Whether to use a bias term in `lora_B`.
        zero_init: Whether to initialize `lora_B` weights to zero.
        """
        super().__init__()
        self.dim: int = dim
        self.rank: int = rank
        self.device: str = device
        self.lora_a: torch.nn.Linear = torch.nn.Linear(dim, rank, bias=False).to(device)
        self.lora_b: torch.nn.Linear = torch.nn.Linear(rank, dim, bias=bias).to(device)
        if zero_init:
            self.lora_b.weight.data.zero_()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-rank adversarial perturbation to the input."""
        return self.lora_b(self.lora_a(x)) + x


class FullRankAdversary(torch.nn.Module):
    """Adversary parameterized by a full-rank MLP.

    This adversary learns a full linear transformation `m` that generates
    perturbations to add to the input activations.
    """

    def __init__(self, dim: int, device: str, bias: bool = False):
        """Initialize FullRankAdversary.

        Args:
        dim: Dimensionality of the input activations.
        device: Device on which to place the model.
        bias: Whether to use a bias term in the linear layer.
        """
        super().__init__()
        self.dim: int = dim
        self.device: str = device
        self.m: torch.nn.Linear = torch.nn.Linear(dim, dim, bias=bias).to(device)

        # Start with no perturbation (weights initialized to zero).
        self.m.weight.data.zero_()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full-rank adversarial perturbation to the input."""
        return self.m(x) + x


class GDAdversary(torch.nn.Module):
    """Standard projected gradient descent (PGD)-style adversary.

    Perturbations are stored as trainable parameters and projected back
    into the epsilon-ball to constrain their norm.
    """

    def __init__(
        self,
        dim: int,
        epsilon: float,
        attack_mask: torch.Tensor,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize GD adversary.

        Args:
        dim: Dimensionality of the input activations.
        epsilon: Maximum perturbation norm.
        attack_mask: Boolean mask indicating which positions to perturb.
        device: Device for tensors.
        dtype: Optional data type for perturbations.
        """
        super().__init__()
        self.device: str | torch.device | None = device
        self.epsilon: float = epsilon

        if dtype:
            self.attack: torch.nn.Parameter = torch.nn.Parameter(
                torch.zeros(
                    attack_mask.shape[0],
                    attack_mask.shape[1],
                    dim,
                    device=self.device,
                    dtype=dtype,
                )
            )
        else:
            self.attack = torch.nn.Parameter(
                torch.zeros(attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device)
            )
        self.clip_attack()
        self.attack_mask: torch.Tensor = attack_mask

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add adversarial perturbations to the input tensor.

        Handles both generation mode (no perturbation applied if sequence
        length == 1) and training mode (perturb according to mask).
        """
        if x.shape[1] == 1 and self.attack.shape[1] != 1:
            # Generation mode: perturbation already applied
            return x
        else:
            # Ensure perturbation tensor is on the same device
            if self.device is None or self.device != x.device:
                with torch.no_grad():
                    self.device = x.device
                    self.attack.data = self.attack.data.to(self.device)
                    self.attack_mask = self.attack_mask.to(self.device)

            # Apply perturbations at masked positions
            perturbed_acts = x[self.attack_mask[:, : x.shape[1]]] + self.attack[self.attack_mask[:, : x.shape[1]]].to(
                x.dtype
            )
            x[self.attack_mask[:, : x.shape[1]]] = perturbed_acts

            return x

    def clip_attack(self) -> None:
        """Project the adversarial perturbations back into the epsilon-ball.

        Ensures that perturbation norms do not exceed `epsilon`.
        """
        with torch.no_grad():
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)


class WhitenedGDAdversary(torch.nn.Module):
    """Whitened variant of GD adversary.

    Perturbations are trained in a whitened space (e.g., via PCA).
    A projection matrix is used to map between whitened and original spaces.
    """

    def __init__(
        self,
        dim: int,
        device: str,
        epsilon: float,
        attack_mask: torch.Tensor,
        proj: torch.Tensor | None = None,
        inv_proj: torch.Tensor | None = None,
    ):
        """Initialize WhitenedPDG adversary.

        Args:
        dim: Dimensionality of the input activations.
        device: Device for tensors.
        epsilon: Maximum perturbation norm.
        attack_mask: Boolean mask indicating which positions to perturb.
        proj: Whitening projection matrix (e.g., PCA basis).
        inv_proj: Optional precomputed inverse of the projection matrix.
        """
        super().__init__()
        self.device: str = device
        self.epsilon: float = epsilon
        self.proj: torch.Tensor | None = proj

        if inv_proj is None and proj is not None:
            self.inv_proj: torch.Tensor = torch.inverse(proj)
        elif inv_proj is not None:
            self.inv_proj = inv_proj
        else:
            raise ValueError("At least one of `proj` or `inv_proj` must be provided")

        # Initialize attack perturbations in whitened space
        self.attack: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(attack_mask.shape[0], attack_mask.shape[1], dim, device=self.device)
        )
        self.clip_attack()
        self.attack_mask: torch.Tensor = attack_mask

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adversarial perturbation by projecting from whitened space.

        The perturbation is first unprojected via `inv_proj` before being
        added to the original activations.
        """
        # einsum notation: n = whitened dim, d = original hidden size
        unprojected_attack = torch.einsum("nd,bsn->bsd", self.inv_proj, self.attack)
        x[self.attack_mask[:, : x.shape[1]]] = (x + unprojected_attack.to(x.dtype))[self.attack_mask[:, : x.shape[1]]]
        return x

    def clip_attack(self) -> None:
        """Project perturbations back into the epsilon-ball in whitened space."""
        with torch.no_grad():
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)
