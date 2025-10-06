import torch
import torch.optim as optim
import dumpling
import time
import json
import polars as pl
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug','-d', action='store_true', help='Debug Mode Flag')
parser.add_argument('--epochs','-e', type=int, default=250, help='Number of epochs for training')
parser.add_argument('--learning_rate','-lr', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('--load', action='store_true', help='Load previous parameters from filename-based JSON file')
args = parser.parse_args()

epochs = args.epochs
learning_rate = args.learning_rate
debug = args.debug
load_previous = args.load

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

file = "LC0-6.csv"
df = pl.read_csv(file, separator=",")

df = df.filter(
    (pl.col("SA") >= -0.25) & (pl.col("SA") <= 0.25)
)

input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC", "IA"]
target_columns_lateral = "NFY"
target_columns_longitudinal = "NFX"

trainable_magic_parameters = {
    "B":
    "loadA": 0,
    "loadB": 1,
    "loadC": 0
}

constant_magic_parameters = {
}

# Combine both dictionaries for compatibility
all_magic_parameters = {**trainable_magic_parameters, **constant_magic_parameters}

# Load previous parameters if requested
if load_previous:
    previous_params_file = file[:-4] + "trained_magic_parameters_final.json"
    try:
        with open(previous_params_file, "r") as f:
            loaded_params = json.load(f)
        print(f"Loading previous parameters from: {previous_params_file}")

        # Update trainable_magic_parameters with loaded values
        for key, value in loaded_params.items():
            if key in trainable_magic_parameters:
                trainable_magic_parameters[key] = value
        print(f"Loaded {len(loaded_params)} parameters from previous training")

    except FileNotFoundError:
        print(f"No previous parameters file found at: {previous_params_file}")
        print("Starting with default parameters")
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        print("Starting with default parameters")

# Create trainable parameters for trainable_magic_parameters
trainable_scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32).to(device)) for key, value in trainable_magic_parameters.items()}

# Create constant tensors for constant_magic_parameters
constant_scalars = {key: torch.tensor(value, dtype=torch.float32).to(device) for key, value in constant_magic_parameters.items()}

# Combine both for compatibility with existing code
scalars = {**trainable_scalars, **constant_scalars}

# Only optimize trainable parameters
optimizer = optim.Adam(trainable_scalars.values(), lr=learning_rate)

inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32).to(device)
targets_lateral = torch.tensor(df[target_columns_lateral].to_numpy(), dtype=torch.float32).to(device)
targets_longitudinal = torch.tensor(df[target_columns_longitudinal].to_numpy(), dtype=torch.float32).to(device)

Constants = {
    "g": 9.8067,
    "e": 2.7183
}

MechanicalParameters = {
    "friction-coeff-lat": 1.7333,
    "friction-coeff-long": 1.7333,
    "unloaded-radius": 1.7333,
    "p_0": 82000,
    "load_0": 300
}

magic = scalars

def getPredictionsBatch(inputs):
    FZ, SR, SA, V, P, TSTC, IA = inputs.T

    tire = dumpling.Tire(FZ, SR, SA, V, P, TSTC, IA, MechanicalParameters, magic)

    lateral_force = tire.getLateralForce()
    longitudinal_force = tire.getLongForce()

    return lateral_force, longitudinal_force

def debugPredictions():
    lat_pred, long_pred = getPredictionsBatch(inputs)

    lat_pred_np = lat_pred.cpu().detach().numpy()
    long_pred_np = long_pred.cpu().detach().numpy()

    sa_values = df["SA"].to_numpy()
    sr_values = df["SR"].to_numpy()
    nfy_targets = df["NFY"].to_numpy()
    nfx_targets = df["NFX"].to_numpy()

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(sa_values, nfy_targets, alpha=0.5, label='NFY (Data)', s=1)
    plt.scatter(sa_values, lat_pred_np, alpha=0.5, label='NFY (Model)', s=1)
    plt.grid(True)
    plt.legend()
    plt.title("SA vs NFY")
    plt.xlabel("Slip Angle")
    plt.ylabel("Normalized Lateral Force")

    plt.subplot(1, 2, 2)
    plt.scatter(sr_values, nfx_targets, alpha=0.5, label='NFX (Data)', s=1)
    plt.scatter(sr_values, long_pred_np, alpha=0.5, label='NFX (Model)', s=1)
    plt.grid(True)
    plt.legend()
    plt.title("SR vs NFX")
    plt.xlabel("Slip Ratio")
    plt.ylabel("Normalized Longitudinal Force")

    plt.tight_layout()
    plt.savefig("train_debug_combined.png", dpi=150, bbox_inches='tight')
    plt.close()

print("BEGIN TRAINING")
start_time = time.time()

# Track best loss and parameters
best_loss = float('inf')
best_parameters = None

for epoch in range(epochs):
    optimizer.zero_grad()

    lateral_predictions, longitudinal_predictions = getPredictionsBatch(inputs)

    lateral_loss = torch.mean((lateral_predictions - targets_lateral).pow(2))
    longitudinal_loss = 0 #torch.mean((longitudinal_predictions - targets_longitudinal).pow(2))

    total_loss = lateral_loss #+ longitudinal_loss

    total_loss.backward()

    # Gradient clipping to prevent gradient explosion (only for trainable parameters)
    torch.nn.utils.clip_grad_norm_(trainable_scalars.values(), max_norm=1.0)

    optimizer.step()

    # Check if this is the best loss so far
    current_loss = total_loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        # Only save trainable parameters
        best_parameters = {key: value.item() for key, value in trainable_scalars.items()}

        # Save best parameters immediately
        json_filename = file[:-4] + "trained_magic_parameters_final.json"
        with open(json_filename, "w") as json_file:
            json.dump(best_parameters, json_file, indent=4)

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Total Loss: {current_loss:.6f}, Lateral Loss: {lateral_loss.item():.6f}") #", Longitudinal Loss: {longitudinal_loss.item():.6f}")

        if debug:
            debugPredictions()

    if debug and (epoch % 100 == 0):
        current_parameters = {key: value.item() for key, value in trainable_scalars.items()}
        json_filename = f"training_checkpoint_epoch_{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)

# Final check - save final parameters if no improvement was found
if best_parameters is not None:
    print(f"Best parameters achieved at loss: {best_loss:.6f}")
else:
    print("No improvement found during training, saving final parameters")
    optimized_parameters = {key: value.item() for key, value in trainable_scalars.items()}
    json_filename = file[:-4] + "trained_magic_parameters_final.json"
    with open(json_filename, "w") as json_file:
        json.dump(optimized_parameters, json_file, indent=4)

print("Training complete.")
print(f"Time elapsed: {time.time() - start_time} seconds")
json_filename = file[:-4] + "trained_magic_parameters_final.json"
print(f"Parameters saved to: {json_filename}")
