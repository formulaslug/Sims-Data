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
df = pl.read_csv("Data/lateralDataSetAvg.csv")
#df = df.with_columns(((-1 * pl.col("FX")) / pl.col("FZ")).alias("FX"))

# Define input and target columns
input_columns = ["FZ", "SR", "SA", "V"]
target_column = "NFY"

# Convert data to PyTorch tensors
inputs = torch.tensor(df[input_columns].to_numpy(), dtype=torch.float32)  # Training inputs
targets = torch.tensor(df[target_column].to_numpy(), dtype=torch.float32)  # Training targets

# Define magic parameters
magic_parameters = {
        "By1": 8.990806579589844,
        "Cy1": 0.9859746694564819,
        "Dy1": 1.9570221900939941,
        "Ey1": 0.977676272392273,
        "e_one": 0.28891706466674805,
        "e_two": 0.8587047457695007,
        "e_three": -0.7307533621788025,
        "r_by1": 11.30118465423584,
        "r_by2": 7.039249897003174,
        "r_by3": 0.09719188511371613,
        "r_by4": 0.0,
        "r_cy1": 1.2744159698486328,
        "r_ey1": -0.5013149976730347,
        "r_ey2": 0.759115993976593,
        "r_hy1": 0.03325833007693291,
        "r_hy2": -0.008811925537884235,
        "r_vy1": 0.062361788004636765,
        "r_vy2": 0.013297787867486477,
        "r_vy3": 0.0,
        "r_vy4": 94.67942810058594,
        "r_vy5": 2.32427716255188,
        "r_vy6": 24.60861587524414,
        "lambda_yk": 1.649388313293457,
        "lambda_vyk": 0.9428434371948242,
        "zeta_2": 0.9428434371948242,
        "lambda_loadscalarlat": 1.3173058032989502,
        "lambda_pressurescalarlat": 1.0
}

scalars = {key: torch.nn.Parameter(torch.tensor(value, dtype=torch.float32)) for key, value in magic_parameters.items()}

optimizer = optim.Adam(scalars.values(), lr=0.02)


def getSlip(input_row, scalars):
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


    curr = dumpling.Tire(FZ, SR, SA, V, Constants, MechanicalParameters, scalars)
    return curr.getLateralForce() / FZ


print("BEGIN TRAINING", time.time() - lastTime)
lastTime = time.time()

cycles = 10000
for epoch in range(cycles):
    
    optimizer.zero_grad()  

    predictions = torch.stack([getSlip(row, scalars) for row in inputs])

    loss = torch.sum((predictions - targets).pow(2))

    loss.backward()

    if epoch % 10 == 0:
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
        json_filename = f"trainCache/latslip-out-parameters-epoch-{epoch}.json"
        with open(json_filename, "w") as json_file:
            json.dump(current_parameters, json_file, indent=4)
        print(f"Saved parameters to {json_filename}")

optimized_parameters = {key: value.item() for key, value in scalars.items()}
json_filename = "latslip-out.json"
with open(json_filename, "w") as json_file:
    json.dump(optimized_parameters, json_file, indent=4)

print("Training complete. Optimized scalars:", optimized_parameters)
print(time.time() - initTime)