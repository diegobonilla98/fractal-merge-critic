from merge_critic.training import train_module


DATA_PATH = "artifacts/clip_policy_calibration_v2.pt"
OUTPUT_PATH = "artifacts/clip_rate_distortion_policy_v2.pt"
DEVICE = "cuda"


if __name__ == "__main__":
    train_module(DATA_PATH, OUTPUT_PATH, hidden_size=768, n_layers=12, critic_dim=64, max_refine_ratio=0.10, device=DEVICE)
