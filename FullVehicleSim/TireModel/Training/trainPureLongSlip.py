import torch
import torch.optim as optim
import polars as pl
import dumpling
import time
import json

# Set device to MPS if available, otherwise default to CPU
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Timer for performance monitoring
lastTime = time.time()
initTime = time.time()

# Load dataset
df = pl.read_csv("Data/combinedDataSet.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "V"]
target_column = "NFX"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
        "r_bx1": 13.187339782714844,
        "r_bx2": 9.575895309448242,
        "r_bx3": 0.0,
        "r_cx1": 1.1033161878585815,
        "r_ex1": -0.5640894174575806,
        "r_ex2": -0.46630001068115234,
        "r_hx1": 0.10596346855163574,
        "lambda_xalpha": 1.1376594305038452,
        "lambda_alphastar": 1.1374865770339966,
        "lambda_combinedslipcoeff": -0.00019185709243174642,
        "combined_long_offset": -0.040454503148794174
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.02)


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
    return curr.getLongForcePureSlip() / FZ


print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 3000
for epoch in range(cycles):
    
    optimizer.zero_grad()  

    predictions = torch.stack([getSlip(row, scalars) for row in inputs])

    loss = torch.sum((predictions - targets).pow(2))

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