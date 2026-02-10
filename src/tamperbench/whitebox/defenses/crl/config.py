"""Configuration for CRL (Contrastive Representation Learning) defense."""

from dataclasses import dataclass, field

from tamperbench.whitebox.defenses.defense import AlignmentDefenseConfig


@dataclass
class CRLConfig(AlignmentDefenseConfig):
    """Configuration for CRL defense.

    Hyperparameters from the paper (Simko et al., 2025) Appendix A.3.

    NOTE ON EXPECTED BEHAVIOR:
    The CRL defense works by pushing harmful representations away from benign ones,
    disrupting the model's ability to generate coherent harmful content. This means
    responses to harmful prompts may appear as:
    - Clean refusals ("I can't help with that") at lower training steps (~500-700)
    - Degenerate/looping text ("T T T T T..." or "in this in this...") at higher steps

    Both behaviors are valid outcomes of the defense. Degenerate outputs indicate the
    model's representation space for harmful content has been sufficiently disrupted
    that it cannot generate coherent responses - arguably a stronger defense than
    polite refusals which could potentially be jailbroken. The KL divergence loss
    ensures benign behavior is preserved regardless.

    IMPLEMENTATION NOTES:
    1. TOKEN POSITION: Representations are extracted from ALL tokens in the
       assistant response (matching the paper's approach, also used by RepBend
       and Circuit Breaking). Per the author: "the last token position is
       probably never very harmful" and this significantly impacts performance.

    2. DISTANCE METRIC: Pure cosine similarity (hard-coded in trainer).
       Per the author, L2 or mixed distance made GCG losses converge slower.

    3. MARGINS: We use mb=0.3, mh=0.5 for cosine distance (bounded to [0, 2]).
       The paper's original margins (500/1500) were for unnormalized L2 distance
       which has a different scale, so the 1:3 ratio does not directly transfer.
       These values were chosen empirically for the normalized cosine setting.

    4. LORA TARGETS: The paper's reference code defaults to all 7 standard modules
       (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

    5. ADVERSARIAL TRAINING: The paper describes adversarial hard negative mining
       (Section 4.2) which we have NOT implemented. Per the author, this gives
       only a modest improvement and is complex to implement.

    Attributes:
        input_checkpoint_path: Path to the base model (before hardening).
        output_checkpoint_path: Path where the hardened model will be saved.
        benign_dataset_name: HuggingFace dataset for benign samples.
        benign_dataset_split: Dataset split for benign samples.
        harmful_dataset_name: HuggingFace dataset for harmful samples.
        harmful_dataset_config: Dataset config name for harmful dataset.
        harmful_dataset_split: Dataset split for harmful samples.
        num_samples: Number of samples per class (default: 10000 per paper).
        alpha: Weight for benign triplet loss (default: 0.5).
        beta: Weight for harmful triplet loss (default: 0.4).
        gamma: Weight for KL divergence loss (default: 0.9).
        margin_benign: Margin for benign triplet loss (paper: 500, ours: 0.3 normalized).
        margin_harmful: Margin for harmful triplet loss (paper: 1500, ours: 0.5 normalized).
        representation_layers: Layers to collect harmful representations from.
        lora_r: LoRA rank (default: 16).
        lora_alpha: LoRA alpha (default: 16).
        lora_dropout: LoRA dropout (default: 0.05).
        learning_rate: Learning rate (default: 1e-5).
        batch_size: Batch size (default: 16).
        num_steps: Number of training steps (default: 1100).
        max_length: Maximum sequence length (default: 512).
        checkpoint_interval: Save checkpoint every N steps (default: 100).
        shuffle_seed: Seed for DataLoader shuffle reproducibility (default: 42).
    """

    # Dataset configuration
    benign_dataset_name: str = "stingning/ultrachat"
    benign_dataset_split: str = "train"
    harmful_dataset_name: str = "allenai/wildguardmix"
    harmful_dataset_config: str = "wildguardtrain"
    harmful_dataset_split: str = "train"
    num_samples: int = 10000

    # Loss weights (from paper)
    alpha: float = 0.5
    beta: float = 0.4
    gamma: float = 0.9

    # Margins (scaled for normalized representations)
    # Paper (Appendix A.3): mb=500, mh=1500 for unnormalized representations.
    # Our implementation: mb=0.3, mh=0.5 for L2-normalized representations
    # where max distance is ~2 (for unit vectors pointing opposite directions).
    margin_benign: float = 0.3
    margin_harmful: float = 0.5

    # Representation extraction
    representation_layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(20, 32)))

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Training configuration
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_steps: int = 1100
    max_length: int = 512
    checkpoint_interval: int = 100
    shuffle_seed: int = 42
