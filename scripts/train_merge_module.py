from merge_critic.training import train_module


DATA_PATH = "artifacts/policy_calibration_v2.pt"
OUTPUT_PATH = "artifacts/rate_distortion_policy_v2.pt"
HIDDEN_SIZE = 768
N_LAYERS = 12
CRITIC_DIM = 64
MAX_REFINE_RATIO = 0.10
BATCH_SIZE = 64
REFINER_EPOCHS = 8
CRITIC_EPOCHS = 8
LEARNING_RATE = 3e-4
VALUE_TARGET = "damage_terminal"
RANKING_WEIGHT = 0.5
DEVICE = "cuda"


if __name__ == "__main__":
    train_module(
        DATA_PATH,
        OUTPUT_PATH,
        HIDDEN_SIZE,
        N_LAYERS,
        CRITIC_DIM,
        MAX_REFINE_RATIO,
        BATCH_SIZE,
        REFINER_EPOCHS,
        CRITIC_EPOCHS,
        LEARNING_RATE,
        critic_target=VALUE_TARGET,
        ranking_weight=RANKING_WEIGHT,
        device=DEVICE,
    )
