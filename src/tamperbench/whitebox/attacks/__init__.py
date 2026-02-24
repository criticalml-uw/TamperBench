"""Tampering whitebox attacks supported by tamperbench.

Import modules for side effects so they register via the attacks registry.
"""

# pyright: reportDuplicateImport=false

# Side-effect imports to populate ATTACKS_REGISTRY
from tamperbench.whitebox.attacks.embedding_attack import (
    embedding_attack as _embedding,
)

# Re-export embedding attack classes for convenience
from tamperbench.whitebox.attacks.embedding_attack.embedding_attack import (
    EmbeddingAttack,
    EmbeddingAttackConfig,
)
from tamperbench.whitebox.attacks.full_parameter_finetune import (
    dataset_registry as _dataset_registry,
)
from tamperbench.whitebox.attacks.full_parameter_finetune import (
    full_parameter_finetune as _full_param,
)
from tamperbench.whitebox.attacks.jailbreak_finetune import (
    jailbreak_finetune as _jailbreak,
)
from tamperbench.whitebox.attacks.lora_finetune import lora_finetune as _lora
from tamperbench.whitebox.attacks.multilingual_finetune import (
    multilingual_finetune as _multilingual,
)

del _dataset_registry, _embedding, _full_param, _jailbreak, _lora, _multilingual

__all__ = [
    "EmbeddingAttack",
    "EmbeddingAttackConfig",
]
