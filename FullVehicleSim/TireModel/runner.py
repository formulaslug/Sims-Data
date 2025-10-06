import dumpling
import json

# Parse data
with open("params.json", "r") as file:
    params = json.load(file)

tire = dumpling.Tire(750, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])

tire.updateParams(normalForce=700, slipRatio=.8, velocityX=1)
print(tire.getLongFoxxcePureSlip())
