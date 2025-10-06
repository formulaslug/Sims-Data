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

file = "R20-7.csv"
df = pl.read_csv(file, separator=",")

input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC", "IA"]
target_columns_lateral = "NFY"
target_columns_longitudinal = "NFX"

trainable_magic_parameters = {
    "q_v2": 0.1,
    "Omega": 0.1,
    "q_Fcx": 0,
    "q_Fcy": 0,
    "q_Fz1": 1.7333,
    "q_Fz2": 0.1,
    "P_pFz1": 0,
    "shape-factor": 0.696268618106842,
    "curvature-scaling-factor": 1.7177082300186157,
    "horizontal-shift-factor": 1.0,
    "vertical-shift-factor": 0.9601083397865295,
    "zeta_1": 0.6045444011688232,
    "p_px1": 0.013196180574595928,
    "p_px2": 0.8705743551254272,
    "p_px3": -0.9542407989501953,
    "p_px4": -0.9245550036430359,
    "p_dx1": 0.642947793006897,
    "p_dx2": 0.9363532066345215,
    "p_dx3": 0.0,
    "p_ex1": 0.5635436177253723,
    "p_ex2": -0.5660403966903687,
    "p_ex3": 0.8500840663909912,
    "p_ex4": -0.005640119314193726,
    "p_kx1": 21.55539894104004,
    "p_kx2": 13.510042190551758,
    "p_kx3": -0.5750095248222351,
    "p_hx1": 0.0021615000441670418,
    "p_hx2": 0.0011597999837249517,
    "p_vx1": 0.41343268752098083,
    "p_vx2": -0.28164142370224,
    "lambda_loadscalarlong": 1.1716148853302002,
    "lambda_pressurescalarlong": 0.501448392868042,
    "tempXAPure": -0.01220773346722126,
    "tempXBPure": 0.9834595322608948,
    "tempXCPure": 3.0494847297668457,
    "r_bx1": 13.225804328918457,
    "r_bx2": 9.537531852722168,
    "r_bx3": 0.0,
    "r_cx1": 1.148189902305603,
    "r_ex1": -0.23479725420475006,
    "r_ex2": -0.27553272247314453,
    "r_hx1": -0.11146939545869827,
    "lambda_alphastar": 1.14445960521698,
    "lambda_xalpha": 1.1735954284667969,
    "lambda_combinedslipcoeff": 0.8948060870170593,
    "combined_long_offset": 0.8635340929031372,
    "tempXA": -0.001096199150197208,
    "tempXB": -0.004315061029046774,
    "tempXC": 0.9925193786621094,
    "By_pure": 0.2724847197532654,
    "p_cy1": 1.2571338415145874,
    "p_dy1": 0.46917635202407837,
    "p_dy2": -0.6315958499908447,
    "p_dy3": 0.0,
    "p_ey1": -0.9565475583076477,
    "p_ey2": -0.8617380857467651,
    "p_ey3": -0.0877305343747139,
    "p_ey4": -6.697000026702881,
    "p_ey5": 0.0,
    "p_ky1": -15.324000358581543,
    "p_ky2": 1.715000033378601,
    "p_ky3": 0.3695000112056732,
    "p_ky4": -6.697000026702881,
    "p_ky5": 0.0,
    "p_py1": -0.6255000233650208,
    "p_py2": -0.06522999703884125,
    "p_py3": -0.6809952855110168,
    "p_py4": -0.34313657879829407,
    "Svy": -0.8437159657478333,
    "zeta3": 1.0,
    "epsilon_y": 0.009999999776482582,
    "lambda_alphastarypure": 0.8513097167015076,
    "lambda_ey": -0.25506287813186646,
    "lambda_kyalpha": 1.0,
    "lambda_nominalload": 1.0,
    "lambda_coeffscalary": 1,
    "lambda_loadscalarlat": 1,
    "lambda_pressurescalarlat": 1,
    "tempYAPure": 0.014011678285896778,
    "tempYBPure": 0.11150991171598434,
    "tempYCPure": 9.20419979095459,
    "Byk": 0.7075067758560181,
    "Cyk": 1.7348066568374634,
    "Eyk": 0.9768351912498474,
    "Shyk": 0.6505736708641052,
    "Svyk": -6.833913803100586,
    "r_by1": 10.753209114074707,
    "r_by2": 7.820000171661377,
    "r_by3": 0.00203699991106987,
    "r_by4": 0.0,
    "r_cy1": 0.9608714580535889,
    "r_ey1": 0.18440714478492737,
    "r_ey2": -0.12552592158317566,
    "r_hy1": 0.00958038866519928,
    "r_hy2": 0.009862390346825123,
    "r_vy1": 0.534748911857605,
    "r_vy2": 0.48336413502693176,
    "r_vy3": 0.0,
    "r_vy4": 93.80660247802734,
    "r_vy5": 1.9759562015533447,
    "r_vy6": 23.584060668945312,
    "lambda_yk": 1.0,
    "lambda_vyk": 1.7570732831954956,
    "zeta_2": 1.7570732831954956,
    "pressureYA": -3.086921788053587e-05,
    "pressureYB": -9.812999633140862e-05,
    "pressureYC": 0.9998338222503662,
    "gysign": -0.10000000149011612,
    "loadA": -0.001,
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

    lateral_force = tire.getLateralForce() / FZ
    longitudinal_force = tire.getLongForce() / FZ

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

    # Constrain loadA to be negative (quadratic opens downward)
    with torch.no_grad():
        if scalars["loadA"].data > 0:
            scalars["loadA"].data = -abs(scalars["loadA"].data)

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
