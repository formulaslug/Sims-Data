import json

cones = []
for x in range(0, 310, 10):
    cones.append({"x": x, "y":  2, "type": "left"})
    cones.append({"x": x, "y": -2, "type": "right"})

with open("cones.json", "w") as f:
    json.dump(cones, f, indent=2)