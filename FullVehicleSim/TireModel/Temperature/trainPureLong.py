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
df = pl.read_csv("pureLongSlipDataSpline.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC"]
target_column = "NFX"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
    "shape-factor": 0.6626899242401123,
    "curvature-scaling-factor": 1.7131842374801636,
    "horizontal-shift-factor": 1.0,
    "vertical-shift-factor": 0.9612321257591248,
    "zeta_1": 0.5975074768066406,
    "p_px1": 0.019048789516091347,
    "p_px2": 0.8727559447288513,
    "p_px3": -0.9590923190116882,
    "p_px4": -0.9251981973648071,
    "p_dx1": 0.6401030421257019,
    "p_dx2": 0.9295424818992615,
    "p_dx3": 0.0,
    "p_ex1": 0.5636742115020752,
    "p_ex2": -0.5553469657897949,
    "p_ex3": 0.8287177085876465,
    "p_ex4": -0.006091138813644648,
    "p_kx1": 21.55919075012207,
    "p_kx2": 13.501811027526855,
    "p_kx3": -0.582970380783081,
    "p_hx1": 0.0021615000441670418,
    "p_hx2": 0.0011597999837249517,
    "p_vx1": 0.417795330286026,
    "p_vx2": -0.2774057984352112,
    "lambda_loadscalarlong": 1.1716409921646118,
    "lambda_pressurescalarlong": 0.5004448890686035,
    "r_bx1": 13.045999526977539,
    "r_bx2": 9.718000411987305,
    "r_bx3": 0.0,
    "r_cx1": 0.9999499917030334,
    "r_ex1": -0.044029999524354935,
    "r_ex2": -0.46630001068115234,
    "r_hx1": -9.968000085791573e-05,
    "lambda_alphastar": 1.0,
    "lambda_xalpha": 1.0,
    "lambda_combinedslipcoeff": 1.0,
    "combined_long_offset": 1.0,
    "tempXAPure": -0.013552775606513023,
    "tempXBPure": 1.0498987436294556,
    "tempXCPure": 3.2294461727142334,
    "tempXA": 0.0,
    "tempXB": 0.0,
    "tempXC": 1.0
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
    return curr.getLongForcePureSlip() / FZ


def debugPredictions(scalars):
    # Extract SR values and NFX targets from the dataset
    sr_values = df["SR"].to_numpy()
    nfx_targets = df["NFX"].to_numpy()
    
    # Get model predictions for each SR value
    predictions = []
    for i in range(len(inputs)):
        pred = getSlip(inputs[i], scalars)
        predictions.append(pred.item())
    
    # Convert to numpy array
    predictions = np.array(predictions)
    
    # Calculate MSE using the same method as displayModel.py
    mse = np.mean((predictions - nfx_targets)**2)
    print(f"Train script MSE (sum of squared errors): {mse}")
    
    # For detailed comparison, also calculate mean squared error
    mean_mse = np.mean((predictions - nfx_targets)**2)
    print(f"Train script mean squared error: {mean_mse}")
    
    # Plot SR vs NFX (targets and predictions)
    plt.figure(figsize=(10, 6))
    plt.scatter(sr_values, nfx_targets, alpha=0.5, label='NFX SR (Data)')
    plt.scatter(sr_values, predictions, alpha=0.5, label='NFX SR (Model)')
    plt.grid(True)
    plt.legend()
    plt.title("SR vs NFX (Train Script)")
    plt.savefig("train_sr_nfx_debug.png")
    
    # Save predictions to CSV for comparison
    debug_data = np.column_stack((sr_values, nfx_targets, predictions))
    np.savetxt("train_predictions_debug.csv", debug_data, delimiter=",", 
               header="SR,NFX_Target,NFX_Prediction", comments="")
    
    # Print first 10 samples for quick inspection
    print("\nFirst 10 samples (SR, NFX_Target, NFX_Prediction):")
    for i in range(min(10, len(sr_values))):
        print(f"SR: {sr_values[i]:.6f}, Target: {nfx_targets[i]:.6f}, Prediction: {predictions[i]:.6f}, Diff: {abs(predictions[i] - nfx_targets[i]):.6f}")
    
    # Print the sum of squared errors by individual point
    individual_errors = (predictions - nfx_targets)**2
    sorted_errors = np.sort(individual_errors)[::-1]  # Sort in descending order
    print("\nTop 5 largest squared errors:")
    for i in range(min(5, len(sorted_errors))):
        idx = np.where(individual_errors == sorted_errors[i])[0][0]
        print(f"SR: {sr_values[idx]:.6f}, Target: {nfx_targets[idx]:.6f}, Prediction: {predictions[idx]:.6f}, Squared Error: {sorted_errors[i]:.6f}")



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
        print("\n--- DEBUG SR/NFX PREDICTIONS ---")
        debugPredictions(scalars)
        print("--- END DEBUG ---\n")

    if epoch % 10 == 0:
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"trainCache/pureLongslip-out=parameters-epoch-{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "combinedslip-out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)