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


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pl.read_csv("pureLongSlipDataNoEnds.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC"]
target_column = "NFX"

# Define magic parameters
magic_parameters = {
        "r_bx1": 13.046,
        "r_bx2": 9.718,
        "r_bx3": 0.0,
        "r_cx1": 0.99995,
        "r_ex1": -0.04403,
        "r_ex2": -0.4663,
        "r_hx1": -0.00009968,
        "lambda_alphastar": 1,
        "lambda_xalpha": 1,
        "lambda_combinedslipcoeff": 1,
        "combined_long_offset": 1,
        "tempXAPure": 0,
        "tempXBPure": 0,
        "tempXCPure": 1,
        "tempXA": 0,
        "tempXB": 0,
        "tempXC": 0
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
        "tempXAPure": -100,
        "tempXBPure": 0.9834595322608948,
        "tempXCPure": 3.0494847297668457,
        "tempXAPure": -100,
        "tempXBPure": 0,
        "tempXCPure": 0
}

magic = {**magicExisting, **scalars}

def getSlipBatch(inputs):
    FZ, SR, SA, V, P, TSTC = inputs.T

    tire = dumpling.Tire(FZ, SR, SA, V, P, TSTC, Constants, MechanicalParameters, magic, long=True)
    return tire.getLongForce() / FZ


def debugPredictions(inputParams):

    predictions = getSlipBatch(inputs).cpu().detach().numpy()
    sa_values = df["SA"].to_numpy()
    nfx_targets = df["NFX"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.scatter(sa_values, nfx_targets, alpha=0.5, label='NFX SR (Data)')
    plt.scatter(sa_values, predictions, alpha=0.5, label='NFX SR (Model)')
    plt.grid(True)
    plt.legend()
    plt.title("SR vs NFX (Train Script)")
    plt.savefig("train_sr_nfx_debug_Nathaniel.png")
    plt.close()

# Training loop
print("BEGIN TRAINING")
start_time = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = getSlipBatch(inputs)

    # Compute loss
    loss = torch.mean((predictions - targets).pow(2))

    # Backpropagation
    loss.backward()

    optimizer.step()

    # if epoch % 100 == 0:
    #     for name, param in scalars.items():
    #         print(f"Gradient for {name}: {param.grad}")

    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        lastTime = time.time()
        if debug:
            print("Time elapsed:", time.time() - lastTime)
            debugPredictions(magic_parameters)

    if debug and (epoch % 100 == 0):
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"CombinedLongCache1/combined_longitudinal{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = f"CombinedLongitudinalParameters.json{epoch}.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete.")
print(f"Time elapsed: {time.time() - start_time} seconds")
