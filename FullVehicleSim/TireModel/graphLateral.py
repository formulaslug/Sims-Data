import dumpling
import json

# Parse data
with open("params.json", "r") as file:
    params = json.load(file)

tire = dumpling.Tire(750, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])


def latTest():
    increments = []
    values = []

    x = 0
    while x <= 10:
        increments.append(x)
        tire = dumpling.Tire(750, params["Constants"], params["Mechanical-Parameters"], params["Magic-Parameters"])
        value = tire.getLateralForce()
        values.append(value)
        x += 0.01
        x = round(x, 2)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(increments, values)
    plt.title('Increment vs Calculated Value')
    plt.xlabel('Increment')
    plt.ylabel('Calculated Value')
    plt.grid(True)
    plt.show()

latTest()
