# Big optimizations
# 1. Use vectorized operations instead of loops and lists (tensors and their operations instead of lists and for loops)
# 2. Close plots after saving to reduce memory usage and increase speed

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
args = parser.parse_args()

# Debug Flag
epochs = args.epochs
learning_rate = args.learning_rate
debug = args.debug


# Cuda requires cuda development kit and in theory cuda cores on your gpu. No idea if the second case is actually true.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pl.read_csv("CombinedLateralData.csv")

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC"]
target_column = "NFY"

# Define magic parameters
magic_parameters = {
    "r_by1": 10.622,
    "r_by2": 7.82,
    "r_by3": 0.002037,
    "r_by4": 0.0,
    "r_cy1": 1.0587,
    "r_ey1": 0.3148,
    "r_ey2": 0.004867,
    "r_hy1": 0.009472,
    "r_hy2": 0.009754,
    "r_vy1": 0.05187,
    "r_vy2": 0.0004853,
    "r_vy3": 0.0,
    "r_vy4": 94.63,
    "r_vy5": 1.8914,
    "r_vy6": 23.8,
    "lambda_yk": 1,
    "lambda_vyk": 1,
    "zeta_2": 1,
    "pressureYA": -3.086921788053587e-05,
    "pressureYB": -9.812999633140862e-05,
    "pressureYC": 0.9998338222503662,
    "gysign" : -0.1
}

# Move magic parameters to the GPU
scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32).to(device)) for key, value in magic_parameters.items()}

# Optimizer
optimizer = optim.Adam(scalars.values(), lr=learning_rate)

# Dummy dataset (replace with actual data)
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32).to(device)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32).to(device)  # Training targets
# Define constants and parameters for the Tire model
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
magicExisting = {
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
    "lambda_coeffscalary": 0.6159840226173401,
    "lambda_loadscalarlat": 0.48696190118789673,
    "lambda_pressurescalarlat": 0.5687988996505737,
    "tempYAPure": 0.014011678285896778,
    "tempYBPure": 0.11150991171598434,
    "tempYCPure": 9.20419979095459,
    "lambda_alphastar": 1.1797136068344116,
}

# Combine scalars and magicExisting
magic = {**magicExisting, **scalars}

# Define the getSlip function
def getSlipBatch(inputs):
    FZ, SR, SA, V, P, TSTC = inputs.T  # Unpack columns

    # batch_tires = [dumpling.Tire(fz, sr, sa, v, p, tstc, Constants, MechanicalParameters, magic) for fz, sr, sa, v, p, tstc in zip(FZ, SR, SA, V, P, TSTC)]
    # return torch.stack([tire.getLateralForce() / tire.normalForce for tire in batch_tires])
    tire = dumpling.Tire(FZ, SR, SA, V, P, TSTC, Constants, MechanicalParameters, magic, lat=True)
    return tire.getLateralForce() / FZ

def debugPredictions(scalars):
    # Vectorized predictions for debugging
    predictions = getSlipBatch(inputs).cpu().detach().numpy()
    sr_values = df["SR"].to_numpy()
    nfy_targets = df["NFY"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(sr_values, nfy_targets, alpha=0.5, label='NFY SR (Data)')
    plt.scatter(sr_values, predictions, alpha=0.5, label='NFY SR (Model)')
    plt.grid(True)
    plt.legend()
    plt.title("SR vs NFY (Train Script)")
    plt.savefig("train_sr_nfy_debug_Nathaniel.png")
    plt.close() # Optimization to reduce memory usage and increase speed

# Training loop
print("BEGIN TRAINING")
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()

    # Get predictions (pytorch tensor operations)
    predictions = getSlipBatch(inputs)

    # Compute loss (mse)
    loss = torch.mean((predictions - targets).pow(2))

    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        lastTime = time.time()
        if debug:
            debugPredictions(magic_parameters)
            print("Time elapsed:", time.time() - lastTime)

    # Save Progress
    if debug and (epoch % 100 == 0):
        optimized_parameters = {key: value.item() for key, value in scalars.items()}
        with open(f"CombinedLatCache1mil2/combined_lateral_better{epoch}.json", "w") as f:
            json.dump(optimized_parameters, f, indent=4)

# Save optimized parameters
optimized_parameters = {key: value.item() for key, value in scalars.items()}
with open("CombinedLateralParameters.json", "w") as f:
    json.dump(optimized_parameters, f, indent=4)

print("Training complete.")
print(f"Time elapsed: {time.time() - start_time} seconds")
