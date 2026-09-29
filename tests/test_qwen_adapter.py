import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM

from merge_critic.adapters.base import FrozenBackboneAdapter
from merge_critic.adapters.qwen3 import Qwen3Adapter
from merge_critic.merge_ops import counterfactual_batch


class TinyTokenizer:
    all_special_ids = [0]
    eos_token_id = 63

    def decode(self, values, skip_special_tokens=True):
        return " ".join(str(value) for value in values)


def tiny_adapter() -> Qwen3Adapter:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention", "full_attention"],
        attention_dropout=0.0,
    )
    adapter = Qwen3Adapter.__new__(Qwen3Adapter)
    adapter.tokenizer = TinyTokenizer()
    adapter.model = Qwen3ForCausalLM(config).eval()
    adapter.device = next(adapter.model.parameters()).device
    adapter.proportional_attention = True
    FrozenBackboneAdapter.__init__(adapter)
    return adapter


@torch.no_grad()
def native_greedy(adapter: Qwen3Adapter, input_ids: torch.Tensor, count: int) -> list[int]:
    outputs = adapter.model(input_ids=input_ids, use_cache=True, return_dict=True)
    cache = outputs.past_key_values
    logits = outputs.logits[0, -1]
    generated = []
    original_length = input_ids.shape[1]
    for step in range(count):
        token = logits.argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        position_ids = torch.tensor([[original_length + step]], dtype=torch.long)
        cache_position = torch.tensor([original_length + step], dtype=torch.long)
        outputs = adapter.model(
            input_ids=token.reshape(1, 1),
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]
    return generated


def test_qwen_zero_merge_matches_native_logits_prefill_and_generation():
    torch.manual_seed(5)
    adapter = tiny_adapter()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    with torch.no_grad():
        native = adapter.model(**batch).logits[0, -1]
        reference = adapter.reference(batch)
        state = adapter.to_layer(batch, 0)
        prefill, _ = adapter.compressed_prefill(state)
    assert torch.allclose(reference, native, atol=1e-5, rtol=1e-5)
    assert torch.allclose(prefill, native, atol=1e-5, rtol=1e-5)
    assert adapter.generate_token_ids_from_layer0_state(state, 4) == native_greedy(adapter, input_ids, 4)


def test_qwen_batched_counterfactual_matches_individual_outputs():
    torch.manual_seed(7)
    adapter = tiny_adapter()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    state = adapter.to_layer({"input_ids": input_ids}, 1)
    pairs = adapter.candidates(state)[:3]
    states = counterfactual_batch(state, pairs)
    individual = torch.stack([adapter.from_layer(value, 1) for value in states])
    batched = adapter.from_layer_batch(states, 1)
    assert torch.allclose(batched, individual, atol=1e-5, rtol=1e-5)
    layer_zero = adapter.to_layer({"input_ids": input_ids}, 0)
    merged_layer_zero = counterfactual_batch(layer_zero, adapter.candidates(layer_zero)[:1])[0]
    merged_logits, _ = adapter.compressed_prefill(merged_layer_zero)
    assert torch.isfinite(merged_logits).all()
    with pytest.raises(ValueError, match="layer-0"):
        adapter.compressed_prefill(states[0])
