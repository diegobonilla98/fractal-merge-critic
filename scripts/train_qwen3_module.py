from merge_critic.training import train_module


MODEL_SIZE = "8B"
DATA_PATH = "artifacts/qwen_policy_calibration_v2.pt"
OUTPUT_PATH = "artifacts/qwen_rate_distortion_policy_v2.pt"
DEVICE = "cuda"


if __name__ == "__main__":
    sizes = {
        "8B": (4096, 36),
        "14B": (5120, 40),
    }
    hidden_size, n_layers = sizes[MODEL_SIZE]
    train_module(DATA_PATH, OUTPUT_PATH, hidden_size=hidden_size, n_layers=n_layers, critic_dim=64, max_refine_ratio=0.05, device=DEVICE)
