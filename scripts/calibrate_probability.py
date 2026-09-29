import math

import torch


PREDICTIONS_PATH = "artifacts/validation_predictions.pt"
EPSILON = 1e-3
TARGET_FALSE_SAFE_RATE = 0.01


def main():
    data = torch.load(PREDICTIONS_PATH, weights_only=False)
    probabilities = data["safe_probability"].float()
    damage = data["damage"].float()
    thresholds = torch.linspace(0.5, 0.9999, 2000)
    best = 1.0
    for threshold in thresholds:
        predicted_safe = probabilities >= threshold
        if predicted_safe.sum() == 0:
            continue
        false_safe = (damage[predicted_safe] >= EPSILON).float().mean()
        if false_safe <= TARGET_FALSE_SAFE_RATE:
            best = float(threshold)
            break
    print({"probability_threshold": best, "epsilon": EPSILON, "target_false_safe_rate": TARGET_FALSE_SAFE_RATE})


if __name__ == "__main__":
    main()
