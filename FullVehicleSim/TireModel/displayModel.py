import dumpling as tire
import matplotlib.pyplot as plt
import matplotlib.axis
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lateral','-lat', action='store_true', help='Lateral Graphing Option')
parser.add_argument('--longitudinal','-long', action='store_true', help='Longitudinal Graphing Option')
parser.add_argument('--comparison','-comp', action='store_true', help='Show model vs data comparison')
parser.add_argument('--precision','-p', type=float, default=0.5, help='Precision for the model')
args = parser.parse_args()
lateral = args.lateral
longitudinal = args.longitudinal
comparison = args.comparison
if not lateral and not longitudinal:
    print("Please specify either --lateral or --longitudinal")
    exit(1)


precision = args.precision  # Controls step size in slip ratio and slip angle

with open('params.json', 'r') as file:
    params = json.load(file)

arr = []

def plotPureLongSRModel(velocity=40, NF=500):
    for SR in np.arange(-1, 1 + precision/10, precision/10):
        longForce = pureLongSRModel(SR)
        arr.append((SR, longForce))
    x, y = zip(*arr)
    plt.scatter(x, y, color='blue', label='NFX SR (Model)', s=0.1)
def pureLongSRModel(SR, velocity=40, NF=500):
    # Calculate the predicted force for a given slip ratio using the Tire model
    runTire = tire.Tire(
        NF, SR, 0, velocity, 85, 30, 0,  # Added camber parameter
        params["Mechanical-Parameters"],
        params["Magic-Parameters"])
    return runTire.getLongForcePureSlip() / NF
def plotPureLongSRData():
    file_path = 'Temperature/pureLongSlipData.csv'
    data = pd.read_csv(file_path)
    x_column = 'SR'
    y_column = 'NFX'
    plt.scatter(data[x_column], data[y_column], alpha=0.5, label='NFX SR (Data)', s=0.1)
    return data

def plotCombinedLatModel(ax, velocity, NF, plotOnlyData=False, fig=None):
    if not plotOnlyData:
        print(NF)
        points = []
        for SA in np.arange(0, 11.5 + precision, precision):
            for SR in np.arange(-0.2, 0.2 + precision/10, precision/10):
                runTire = tire.Tire(420, SR, SA, velocity, 96, 60, 0,  # Added camber parameter
                                    params["Mechanical-Parameters"],
                                    params["Magic-Parameters"])
                lateralForce = max(min(4,runTire.getLateralForce() / NF), -4)
                points.append((SR, SA, lateralForce))
        ltx, lty, ltz  = zip(*points)
        # Enhanced trisurf plotting without color mapping
        ax.plot_trisurf(ltx, lty, ltz, alpha=0.7, linewidth=0.1, antialiased=True)

        x_column = 'SR'
        y_column = 'SA'
        z_column = 'NFY'
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title('Lateral Force vs Slip Ratio and Slip Angle (Model)')

def plotCombinedLatData(ax):
    file_path = 'Cleaner/LC0-6.csv'
    data = pd.read_csv(file_path)

    x_column = 'SR'
    y_column = 'SA'
    z_column = 'NFY'
    sample_indices = np.random.choice(data.index, size=len(data) // 30, replace=False)
    sampled_data = data.loc[sample_indices]

    ax.scatter(sampled_data[x_column], sampled_data[y_column], sampled_data[z_column], alpha=0.5, label='Tire Data', s = 0.1)

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_zlabel(z_column)

def plotCombinedLongModel(ax, velocity, NF, plotOnlyData=False, fig=None):
    maxValue = 10000
    slipRatio = -5
    if not plotOnlyData:
        points = []
        for SA in np.arange(0, 8 + precision, precision):
            for SR in np.arange(-1, 1 + precision/10, precision/10):
                runTire = tire.Tire(420, SR, SA, velocity, 96, 60, 0, params["Mechanical-Parameters"], params["Magic-Parameters"])  # Added camber parameter

                longForce = runTire.getLongForceCombinedSlip() / NF
                if longForce < maxValue:
                    maxValue = longForce
                    slipRatio = SR
                points.append((SR, SA, longForce))
        lgx, lgy, lgz  = zip(*points)
        # Enhanced trisurf plotting without color mapping
        ax.plot_trisurf(lgx, lgy, lgz, alpha=0.7, linewidth=0.1, antialiased=True)

        x_column = 'SR'
        y_column = 'SA'
        z_column = 'NFX'
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title('Longitudinal Force vs Slip Ratio and Slip Angle (Model)')
    print(maxValue, slipRatio)

def combinedLongModel(SR, velocity, NF):
    pass

def plotModelComparison(ax, velocity, NF, mode='longitudinal', fig=None):
    """
    Plot both model and data on the same 3D surface for comparison
    mode: 'longitudinal' or 'lateral'
    """
    if mode == 'longitudinal':
        # Plot data first (as scatter)
        plotCombinedLongData(ax)

        # Plot model as surface
        points = []
        for SA in np.arange(0, 8 + precision, precision):
            for SR in np.arange(-1, 1 + precision/10, precision/10):
                runTire = tire.Tire(420, SR, SA, velocity, 96, 60, 0,
                                  params["Mechanical-Parameters"],
                                  params["Magic-Parameters"])
                longForce = -1 * runTire.getLongForceCombinedSlip() / NF
                points.append((SR, SA, longForce))

        lgx, lgy, lgz = zip(*points)
        ax.plot_trisurf(lgx, lgy, lgz, alpha=0.5, linewidth=0.1)
        ax.set_title('Longitudinal Force Comparison: Model vs Data')

    elif mode == 'lateral':
        # Plot data first (as scatter)
        plotCombinedLatData(ax)

        # Plot model as surface
        points = []
        for SA in np.arange(0, 11.5 + precision, precision):
            for SR in np.arange(-0.2, 0.2 + precision/10, precision/10):
                runTire = tire.Tire(420, SR, SA, velocity, 96, 60, 0,
                                  params["Mechanical-Parameters"],
                                  params["Magic-Parameters"])
                lateralForce = runTire.getLateralForce() / NF
                points.append((SR, SA, lateralForce))

        ltx, lty, ltz = zip(*points)
        ax.plot_trisurf(ltx, lty, ltz, alpha=0.5, linewidth=0.1)
        ax.set_title('Lateral Force Comparison: Model vs Data')
def plotCombinedLongData(ax):
    file_path = 'Temperature/combinedLongSlipData.csv'
    data = pd.read_csv(file_path)

    x_column = 'SR'
    y_column = 'SA'
    z_column = 'NFX'
    sample_indices = np.random.choice(data.index, size=len(data) // 30, replace=False)
    sampled_data = data.loc[sample_indices]

    ax.scatter(sampled_data[x_column], sampled_data[y_column], sampled_data[z_column], alpha=0.5, label='Tire Data', s = 0.1)

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_zlabel(z_column)

def plotPressure(ax, velocity):
    if True:
        points = []
        for SA in np.arange(0, 15 + precision, precision):
            runTire = tire.Tire(1000, 0.1, SA, velocity, 70, 60, 0, params["Mechanical-Parameters"], params["Magic-Parameters"])  # Added camber parameter
            longForce = runTire.getLateralForce() / 1000

            runTire2 = tire.Tire(1000, 0.1, SA + precision, velocity, 70, 60, 0, params["Mechanical-Parameters"], params["Magic-Parameters"])  # Added camber parameter
            longForce2 = runTire2.getLateralForce() / 1000
            points.append(( SA,(longForce2-longForce)/precision) )
        #print(len(lgx), len(lgy))
        x = []
        y = []
        for i in points:
            x.append(float(i[0]))
            y.append(float(i[1]))
        #@print("DONE")
        #lgx, lgy  = zip(*points)
        print(len(x))
        print(len(y))

        plt.xlabel('Slip Angle')
        plt.ylabel('Normalized Cornering Stiffness')
        plt.title('Normalized Cornering Stiffness Response to Slip Angle')

        plt.scatter(x, y, s=len(x))
        plt.show()


if __name__ == "__main__":
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Enable different plotting modes based on arguments
    if comparison:
        # Show model vs data comparison
        if longitudinal:
            plotModelComparison(ax, 40, 500, mode='longitudinal', fig=fig)
        elif lateral:
            plotModelComparison(ax, 40, 500, mode='lateral', fig=fig)
    else:
        # Show model and/or data separately
        if longitudinal:
            plotCombinedLongData(ax)
            plotCombinedLongModel(ax, 40, 500, fig=fig)
        elif lateral:
            plotCombinedLatData(ax)
            plotCombinedLatModel(ax, 40, 500, fig=fig)
        else:
            # Default: show longitudinal force model
            plotCombinedLongModel(ax, 40, 500, fig=fig)

    plt.legend()
    plt.show()
