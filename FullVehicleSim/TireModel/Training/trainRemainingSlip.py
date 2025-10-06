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
input_columns = ["FZ", "SR", "SA", "V"]
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
        "combined_long_offset": -0.040454503148794174,
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.01)


def getSlip(input_row, magic):
    FZ, SR, SA, V = input_row
    
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
    
    existing = {
        "shape-factor": 1.954084038734436,
        "stiffness-factor": 10.066422462463379,
        "curvature-scaling-factor": 1.213955283164978,
        "horizontal-shift-factor": 1.0,
        "vertical-shift-factor": 0.6640418767929077,
        "zeta_1": 0.7150991559028625,
        "p_px1": -0.34850001335144043,
        "p_px2": 0.37834998965263367,
        "p_dx1": 3.2977943420410156,
        "p_dx2": -0.0179620161652565,
        "p_dx3": 0.0,
        "p_dx4": 0.0,
        "p_ex1": 0.06139938533306122,
        "p_ex2": 0.34025561809539795,
        "p_ex3": 0.15006783604621887,
        "p_ex4": -0.07350758463144302,
        "p_kx1": 21.687000274658203,
        "p_kx2": 13.727999687194824,
        "p_kx3": -0.4097999930381775,
        "p_hx1": 0.00021615000150632113,
        "p_hx2": 0.0011597999837249517,
        "p_vx1": -0.0005141909932717681,
        "p_vx2": 0.03788552060723305,
        "lambda_loadscalar": 1.0094040632247925,
        "lambda_pressurescalar": 1.0
    }

    scalars = magic | existing


    curr = dumpling.Tire(FZ, SR, SA, V, Constants, MechanicalParameters, scalars)
    return curr.getLongForce() / FZ


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
        json_filename = f"trainCache/pureslip-out=parameters-epoch-{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "pureslip-out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)