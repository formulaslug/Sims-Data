import torch
import torch.optim as optim
import polars as pl
import dumpling
import time
import json
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

lastTime = time.time()
initTime = time.time()

# Load dataset
df = pl.read_csv("isolatedCombinedLateralData.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V", "P", "TSTC"]
target_column = "NFY"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

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
        "pressureYC": 0.9998338222503662
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.01)


def getSlip(input_row, scalars):
    FZ, SR, SA, V, P, TSTC = input_row
    
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

    magic = scalars | magicExisting

    curr = dumpling.Tire(FZ, SR, SA, V, P, TSTC, Constants, MechanicalParameters, magic)
    return curr.getLateralForce() / FZ
    

def debugPredictions(scalars):
    sr_values = df["SR"].to_numpy()
    nfy_targets = df["NFY"].to_numpy()
    
    predictions = []
    for i in range(len(inputs)):
        pred = getSlip(inputs[i], scalars)
        predictions.append(pred.item())
    
    predictions = np.array(predictions)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sr_values, nfy_targets, alpha=0.5, label='NFY SR (Data)')
    plt.scatter(sr_values, predictions, alpha=0.5, label='NFY SR (Model)')
    plt.grid(True)
    plt.legend()
    plt.title("SR vs NFY (Train Script)")
    plt.savefig("train_sr_nfy_debug.png")
    


print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 30000
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
        debugPredictions(scalars)

    if epoch % 10 == 0:
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"trainCache/LateralCombined-out=parameters-epoch-{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "LateralCombined-out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)