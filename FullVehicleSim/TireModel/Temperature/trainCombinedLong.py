import torch
import torch.optim as optim
import polars as pl
import dumpling
import time
import json
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Set device to MPS if available, otherwise default to CPU
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Timer for performance monitoring
lastTime = time.time()
initTime = time.time()

# Load dataset
df = pl.read_csv("combinedLongSlipData.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC"]
target_column = "NFX"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
        "r_bx1": 13.045999526977539,
        "r_bx2": 9.718000411987305,
        "r_bx3": 0.0,
        "r_cx1": 0.9999499917030334,
        "r_ex1": -0.044029999524354935,
        "r_ex2": -0.46630001068115234,
        "r_hx1": -9.968000085791573e-05,
        "lambda_alphastar": 1.0,
        "lambda_xalpha": 1.0,
        "lambda_combinedslipcoeff": 0.0074412147514522076,
        "combined_long_offset": 0.11826133728027344,
        "tempXA": -0.0050585162825882435,
        "tempXB": 0.36763498187065125,
        "tempXC": 1.8092873096466064,
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.01)


def getSlip(input_row, scalars):
    FZ, SR, SA, V, P, TSTC = input_row
    
    pureLongSlipParams = {
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
    }

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

    magic = pureLongSlipParams | scalars

    curr = dumpling.Tire(FZ, SR, SA, V, P, TSTC, Constants, MechanicalParameters, magic)
    return curr.getLongForce() / FZ


def debugPredictions(inputParams):
    pureLongSlipParams = {
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
    }

    scalars = inputParams | pureLongSlipParams

    sa_values = df["SA"].to_numpy()
    nfx_targets = df["NFX"].to_numpy()
    
    predictions = []
    for i in range(len(inputs)):
        pred = getSlip(inputs[i], scalars)
        predictions.append(pred.item())
    
    predictions = np.array(predictions)
    
    mse = np.mean((predictions - nfx_targets)**2)
    print(f"Train script MSE (sum of squared errors): {mse}")
    
    mean_mse = np.mean((predictions - nfx_targets)**2)
    print(f"Train script mean squared error: {mean_mse}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sa_values, nfx_targets, alpha=0.5, label='NFX SA (Data)')
    plt.scatter(sa_values, predictions, alpha=0.5, label='NFX SA (Model)')
    plt.grid(True)
    plt.legend()
    plt.title("SA vs NFX (Train Script)")
    plt.savefig("train_sa_nfx_debug.png")
    


print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 3000
for epoch in range(cycles):
    
    optimizer.zero_grad()  

    predictions = torch.stack([getSlip(row, scalars) for row in inputs])

    loss = torch.mean((predictions - targets).pow(2))

    loss.backward()

    if epoch % 100 == 0:
        for name, param in scalars.items():
            print(f"Gradient for {name}: {param.grad}")

    optimizer.step()

    print(epoch, loss.item())

    if epoch % 10 == 0 or epoch == cycles - 1:
        debugPredictions(magic_parameters)
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print("Time elapsed:", time.time() - lastTime)
        lastTime = time.time()

    if epoch % 10 == 0:
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"trainCache/combinedslip-out=parameters-epoch-{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "combinedslip-out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)