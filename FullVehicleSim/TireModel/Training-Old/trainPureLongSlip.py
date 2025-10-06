import torch
import torch.optim as optim
import polars as pl
import dumpling
import time
import json

# Set device to MPS if available, otherwise default to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Timer for performance monitoring
lastTime = time.time()
initTime = time.time()

# Load dataset
df = pl.read_csv("train.csv")
df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "V"]
target_column = "FX"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
  "shape-factor": 1.9,
  "stiffness-factor": 10,
  "curvature-scaling-factor": 1.29448978539636,
  "horizontal-shift-factor": 1,
  "vertical-shift-factor": 1,
  "zeta_1": 1,
  "p_px1": -0.3485,
  "p_px2": 0.37835,
  "p_dx1": 1.042,
  "p_dx2": -0.08285,
  "p_dx3": 0,
  "p_dx4": 0,
  "p_ex1": 0.11113,
  "p_ex2": 0.3143,
  "p_ex3": 0,
  "p_ex4": 0.001719,
  "p_kx1": 21.687,
  "p_kx2": 13.728,
  "p_kx3": -0.4098,
  "p_hx1": 0.00021615,
  "p_hx2": 0.0011598,
  "p_vx1": 0.000020283,
  "p_vx2": 0.00010568
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.01)


def getSlip(input_row, scalars):
    FZ, SR, V = input_row
    
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


    curr = dumpling.Tire(FZ, SR, 0, V, Constants, MechanicalParameters, scalars)
    return curr.getLongForcePureSlip()


print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 5000  
for epoch in range(cycles):
    optimizer.zero_grad()  

    predictions = torch.stack([getSlip(row, scalars) for row in inputs])

    loss = torch.mean((predictions - targets).pow(2))

    loss.backward()

    if epoch % 100 == 0:
        for name, param in scalars.items():
            print(f"Gradient for {name}: {param.grad}")

    optimizer.step()

    if epoch % 25 == 0 or epoch == cycles - 1:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print("Time elapsed:", time.time() - lastTime)
        lastTime = time.time()

    if epoch % 100 == 0:
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"trainCache/parameters_epoch_{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)