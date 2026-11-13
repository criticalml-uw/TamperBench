"""The vendored LoFiT Llama must match HF Llama's RoPE (rope_theta and rope_scaling)."""

import pytest
import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from tamperbench.whitebox.attacks.lofit.vendor.models.modeling_llama import LlamaForCausalLM

LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def tiny_config(rope_theta: float, rope_scaling: dict[str, float | str] | None) -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=131072,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize(
    ("rope_theta", "rope_scaling"),
    [(10000.0, None), (500000.0, None), (500000.0, LLAMA3_SCALING)],
    ids=["llama2", "llama3", "llama3.2"],
)
def test_vendored_llama_matches_hf_with_zero_lofit_params(
    rope_theta: float, rope_scaling: dict[str, float | str] | None
) -> None:
    config = tiny_config(rope_theta, rope_scaling)
    torch.manual_seed(0)
    hf_model = HFLlamaForCausalLM(config).eval()
    vendored = LlamaForCausalLM(config).eval()
    vendored.load_state_dict(hf_model.state_dict(), strict=False)
    vendored.model.set_applied_modules_to_layers("attention")

    input_ids = torch.randint(0, config.vocab_size, (2, 256))
    with torch.no_grad():
        expected = hf_model(input_ids).logits
        actual = vendored(input_ids).logits

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_custom_from_pretrained_keeps_rope_frequencies_fp32(tmp_path) -> None:
    config = tiny_config(500000.0, LLAMA3_SCALING)
    HFLlamaForCausalLM(config).save_pretrained(tmp_path)

    hf_model = HFLlamaForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
    vendored = LlamaForCausalLM.custom_from_pretrained(tmp_path, cache_dir=None, torch_dtype=torch.bfloat16)

    expected = hf_model.model.rotary_emb.inv_freq
    for layer in vendored.model.layers:
        inv_freq = layer.self_attn.rotary_emb.inv_freq
        assert inv_freq.dtype == torch.float32
        torch.testing.assert_close(inv_freq, expected)
