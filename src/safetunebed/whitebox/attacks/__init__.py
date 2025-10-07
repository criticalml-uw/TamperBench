"""Tampering whitebox attacks supported by safetunebed.

Import modules for side effects so they register via the attacks registry.
"""

# Side-effect imports to populate ATTACKS_REGISTRY
from safetunebed.whitebox.attacks.embedding_attack import (
    embedding_attack as _,
)

# Re-export embedding attack classes for convenience
from safetunebed.whitebox.attacks.embedding_attack.embedding_attack import (
    EmbeddingAttack,
    EmbeddingAttackConfig,
)
from safetunebed.whitebox.attacks.full_parameter_finetune import (
    full_parameter_finetune as _,
)
from safetunebed.whitebox.attacks.jailbreak_finetune import (
    jailbreak_finetune as _,
)
from safetunebed.whitebox.attacks.lora_finetune import lora_finetune as _
from safetunebed.whitebox.attacks.multilingual_finetune import (
    multilingual_finetune as _,  # noqa: F401
)

__all__ = [
    "EmbeddingAttack",
    "EmbeddingAttackConfig",
]
