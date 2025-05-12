import os
import matplotlib.pyplot as plt
import pandas as pd

# Directory containing CSV logs
log_dir = "logs"  # or "loss_logs" if you've renamed it

# Fetch files for each model separately
baseline_files = sorted(
    [f for f in os.listdir(log_dir) if f.startswith("kaushikunet") and f.endswith(".csv")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

advanced_files = sorted(
    [f for f in os.listdir(log_dir) if f.startswith("advanced_unet") and f.endswith(".csv")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])
)

plt.figure(figsize=(12, 6))

# Plot baseline model
for file in baseline_files:
    path = os.path.join(log_dir, file)
    df = pd.read_csv(path)
    epoch = file.split("_")[-1].split(".")[0]
    plt.plot(df["Step"], df["Loss"], linestyle='--', label=f"Baseline - Epoch {epoch}")

# Plot advanced model
for file in advanced_files:
    path = os.path.join(log_dir, file)
    df = pd.read_csv(path)
    epoch = file.split("_")[-1].split(".")[0]
    plt.plot(df["Step"], df["Loss"], linestyle='-', label=f"Advanced - Epoch {epoch}")

plt.xlabel("Training Step")
plt.ylabel("L1 Loss")
plt.title("Step-wise Loss per Epoch (Baseline vs Advanced U-Net)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_loss_curve.png")
plt.show()
# This code will generate a plot comparing the loss curves of the two models.
# You can save the plot as "combined_loss_curve.png" and view it to analyze the performance of both models.