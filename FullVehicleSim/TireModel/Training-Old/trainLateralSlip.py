import torch
import torch.optim as optim
import polars as pl
import dumpling
import time
import json

# Timer for performance monitoring
lastTime = time.time()
initTime = time.time()

# Load dataset
df = pl.read_csv("slipAngleData.dat", separator="	", skip_rows=1, skip_rows_after_header=1)
df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V"]
target_column = "FY"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
    "By1": 10,
    "Cy1": 1.9,
    "Dy1": 1,
    "Ey1": 0.97,
    "e_one": 0,
    "e_two": 0,
    "e_three": 0
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.01)


def getSlip(input_row, scalars):
    FZ, SR, SA, V = input_row
    
    Constants = {
        "g": 9.8067,
        "e": 2.7183,
        "pi": 3.1415
    }

    MechanicalParameters = {
        "friction-coeff-lat": 1.7333,
        "friction-coeff-long": 1.7333,
        "unloaded-radius": 1.7333,
        "p_0": 82000,
        "load_0": 300
    }

    constants = {
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
        "p_vx2": 0.00010568,
        "r_bx1": 13.045999526977539,
        "r_bx2": 9.718000411987305,
        "r_bx3": 0.0,
        "r_cx1": 0.9994999766349792,
        "r_ex1": -0.44029998779296875,
        "r_ex2": -0.46630001068115234,
        "r_hx1": -9.968000085791573e-05,
        "lambda_xalpha": 0.9999904036521912,
        "lambda_alphastar": 1.0,
        "lambda_combinedslipcoeff": -0.00010492587898625061
    }

    combinedParams = constants | scalars
    curr = dumpling.Tire(FZ, SR, SA, V, Constants, MechanicalParameters, combinedParams)
    return curr.getLateralForce()


print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 1000
for epoch in range(cycles):
    optimizer.zero_grad()  

    predictions = torch.stack([getSlip(row, scalars) for row in inputs])

    loss = torch.mean((predictions - targets).pow(2))

    loss.backward()

    if epoch % 50 == 0:
        for name, param in scalars.items():
            print(f"Gradient for {name}: {param.grad}")

    optimizer.step()

    if epoch % 25 == 0 or epoch == cycles - 1:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print("Time elapsed:", time.time() - lastTime)
        lastTime = time.time()

    if epoch % 100 == 0:
        current_parameters = {key: value.item() for key, value in scalars.items()}
        json_filename = f"trainCache/latslip_parameters_epoch_{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "lat_out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)