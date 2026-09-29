import torch

from merge_critic.adapters import Qwen3Adapter
from merge_critic.metrics import timed_cuda
from merge_critic.policy import PolicyConfig, TransformerCostModel
from merge_critic.runtime import merge_round
from merge_critic.training import load_module


MODEL_NAME = "Qwen/Qwen3-8B"
CHECKPOINT = "artifacts/qwen_rate_distortion_policy_v2.pt"
TEXT_FILE = "long_prompt.txt"
ROUNDS = 3
EPSILON = 2e-3
TARGET_FRACTION = 0.5
MAX_NEW_TOKENS = 64
DEVICE_MAP = "auto"
DTYPE = torch.bfloat16


def main():
    adapter = Qwen3Adapter(MODEL_NAME, DEVICE_MAP, DTYPE, proportional_attention=False)
    module = load_module(CHECKPOINT, adapter.device).eval()
    text = open(TEXT_FILE, "r", encoding="utf-8").read()
    batch = adapter.prepare(text)
    state = adapter.to_layer(batch, 0)
    original_tokens = state.n_tokens
    policy = PolicyConfig(target_tokens=max(1, round(original_tokens * TARGET_FRACTION)), distortion_budget=EPSILON, maximum_rounds_per_site=ROUNDS)
    cost_model = TransformerCostModel.from_adapter(adapter)
    for round_idx in range(ROUNDS):
        pairs = adapter.candidates(state)
        result = merge_round(state, pairs, module, 0, round_idx, original_tokens, policy, cost_model)
        if result.selected_pairs.shape[0] == 0:
            break
        state = result.state
    compressed_tokens = state.n_tokens
    baseline = timed_cuda(lambda: adapter.model(**batch, use_cache=False), warmup=2, repeats=8)
    compressed = timed_cuda(lambda: adapter.compressed_prefill(state), warmup=2, repeats=8)
    generated = adapter.generate_from_layer0_state(state, MAX_NEW_TOKENS)
    print({"tokens": (original_tokens, compressed_tokens), "baseline": baseline, "compressed_prefill": compressed})
    print(generated)


if __name__ == "__main__":
    main()
