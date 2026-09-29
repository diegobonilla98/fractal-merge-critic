from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_name: str
    hidden_size: int
    n_layers: int
    modality: str
    default_merge_sites: tuple[int, ...]
    max_refine_ratio: float
    default_epsilon: float


MODEL_SPECS = {
    "clip_b16": ModelSpec(
        key="clip_b16",
        model_name="openai/clip-vit-base-patch16",
        hidden_size=768,
        n_layers=12,
        modality="vision",
        default_merge_sites=(1, 5),
        max_refine_ratio=0.10,
        default_epsilon=1e-3,
    ),
    "dinov3_vits16": ModelSpec(
        key="dinov3_vits16",
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        hidden_size=384,
        n_layers=12,
        modality="vision",
        default_merge_sites=(1, 5),
        max_refine_ratio=0.10,
        default_epsilon=1e-3,
    ),
    "qwen3_8b": ModelSpec(
        key="qwen3_8b",
        model_name="Qwen/Qwen3-8B",
        hidden_size=4096,
        n_layers=36,
        modality="text",
        default_merge_sites=(0,),
        max_refine_ratio=0.05,
        default_epsilon=2e-3,
    ),
    "qwen3_14b": ModelSpec(
        key="qwen3_14b",
        model_name="Qwen/Qwen3-14B",
        hidden_size=5120,
        n_layers=40,
        modality="text",
        default_merge_sites=(0,),
        max_refine_ratio=0.05,
        default_epsilon=2e-3,
    ),
}
