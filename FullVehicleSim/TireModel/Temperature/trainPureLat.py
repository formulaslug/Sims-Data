import torch
import torch.optim as optim
import polars as pl
import dumpling
import time
import json
import numpy as np
import matplotlib.pyplot as plt

# Timer for performance monitoring
lastTime = time.time()
initTime = time.time()

# Load dataset
df = pl.read_csv("pureLateralData.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC"]
target_column = "NFY"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
        "By_pure": 1.4088222980499268,
        "p_cy1": 1.7481794357299805,
        "p_dy1": 0.48669466376304626,
        "p_dy2": -0.6184473633766174,
        "p_dy3": 0.0,
        "p_ey1": -0.8394709229469299,
        "p_ey2": -0.7492206692695618,
        "p_ey3": 0.13710202276706696,
        "p_ey4": -6.697000026702881,
        "p_ey5": 0.0,
        "p_ky1": -15.324000358581543,
        "p_ky2": 1.715000033378601,
        "p_ky3": 0.3695000112056732,
        "p_ky4": -6.697000026702881,
        "p_ky5": 0.0,
        "p_py1": -0.6255000233650208,
        "p_py2": -0.06522999703884125,
        "p_py3": -0.6956681609153748,
        "p_py4": -0.3617802858352661,
        "Svy": 0.009003682062029839,
        "zeta3": 1.0,
        "epsilon_y": 0.009999999776482582,
        "lambda_alphastarypure": 1.0087872743606567,
        "lambda_ey": -0.1359310746192932,
        "lambda_kyalpha": 1.0,
        "lambda_nominalload": 1.0,
        "lambda_coeffscalary": 0.6242331266403198,
        "lambda_loadscalarlat": 0.4951983690261841,
        "lambda_pressurescalarlat": 0.5508548021316528,
        "tempYAPure": -0.001996215432882309,
        "tempYBPure": -0.007417622953653336,
        "tempYCPure": 8.987092018127441,
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.01)


def getSlip(input_row, scalars):
    FZ, SR, SA, V, P, TSTC = input_row
    
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


    curr = dumpling.Tire(FZ, SR, SA, V, P, TSTC, Constants, MechanicalParameters, scalars)
    return curr.getLateralForcePure() / FZ


def debugPredictions(scalars):
    # Extract SR values and NFX targets from the dataset
    sr_values = df["SA"].to_numpy()
    nfy_targets = df["NFY"].to_numpy()
    
    # Get model predictions for each SR value
    predictions = []
    for i in range(len(inputs)):
        pred = getSlip(inputs[i], scalars)
        predictions.append(pred.item())
    
    # Convert to numpy array
    predictions = np.array(predictions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sr_values, nfy_targets, alpha=0.5, label='NFY SA (Data)')
    plt.scatter(sr_values, predictions, alpha=0.5, label='NFY SA (Model)')
    plt.grid(True)
    plt.legend()
    plt.title("SA vs NFY (Train Script)")
    plt.savefig("train_sa_nfy_debug.png")
    
print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 10000
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
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print("Time elapsed:", time.time() - lastTime)
        lastTime = time.time()

    if epoch % 10 == 0 or epoch == cycles - 1:
        print("\n--- DEBUG SR/NFY PREDICTIONS ---")
        debugPredictions(scalars)
        print("--- END DEBUG ---\n")

    if epoch % 10 == 0:
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"trainCache/pureLatslip-out=parameters-epoch-{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "lateralpure-out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)