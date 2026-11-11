# Full Vehicle Simulation

Documentation explaining the different software bits of the full vehicle simulation and how to run it. You can get all the required libraries by following instructions in the ```Sims-Data``` readme.

This folder contains the entirety of the full vehicle simulation for Formula Slug. It is run from [main.py](main.py) where you provide the necessary parameters to begin the simulation. There are a few main folders: ```Powertrain``` (Powertrain/Battery System), ```Mech```  (Mechanical System), ```SimulationInputs``` and ```TireModel``` (Tire Model Training). ```Powertrain``` and ```Mech``` are model bits used while running the simulation. ```SimulationInputs``` contains various simulation starting points: Initial parameters like timeStep, and control inputs for the duration of a simulation.

If you want more information on the basis of some of these simulation parts, you can find them in [GeneralModelingNotes.md](GeneralModelingNotes.md)

## Table of Contents

1. [Simulation Inputs](#siminputs)
    1. [Simulation Parameters](#pars)
    1. [Simulation Controls](#ctrls)
1. [Simulation Interfaces](#interfaces)
    1. [Main Interface](#minter)
    1. [Step By Step Interface](#sbsinter)
1. [Main Simulation Components](#msc)
    1. [Powertrain](#ptr)
    1. [Mechanical System](#mech)
1. [Model Training](#mt)
    1. [Tire Modeling](#tm)
    1. [Voltage Modeling](#vm)

<h2 id="siminputs"> Simulation Inputs </h2>

The simulations are broken out into inputs and parameters. The parameters define constants (both physical and machine learning based) and initial state of the car. The controls are throttle, steering, brake, and time to line them up.

<h3 id="pars"> Simulation Parameters </h3>

This is a json input file with a bunch of inputs for the simulation. json5 supports comments so stuff like units and other notes like options for certain fields live inside it! The file is [params.json5](params.json5). Tweak this for all the indepth constants and higher level stuff like step size and simulation duration.

<h3 id="ctrls"> Simulation Controls </h3>

Simulation controls is a ```.csv``` or ```.parquet``` file that contains ```time (sec)```, ```throttle (-1 to 1)```, ```brakePressureFront```, ```brakePressureRear```, ```brakePedalTravel```, and ```steering angle``` inputs. Time is in seconds, throttle is -1 to 1 (includes regen), both brake pressures are in ```PSI```, brake pedal travel is 0-1, and steering angle is in radians. The full vehicle simulation will do interpolation automatically to align the input/simulation time steps as needed. You can select from CS for cubic spline for continuous data from a run or LI for linear interpolation for a simpler input.

A key note, the simulation has its own parameter for simulation duration. It will pre-emptively end the simulation if it is shorter than the control inputs or continue running using the inputs from the last step if the simulation duration is longer than the control inputs. 

An example of a basic straight line acceleration:
```
time,throttle,brakePressureFront,brakePressureRear,brakePedalTravel,steerAngle
0.0,1.0,0.0,0.0,0.0,0.0
30.0,1.0,0.0,0.0,0.0,0.0
```

An example of some acceleration followed by some braking:
```
time,throttle,brakePressureFront,brakePressureRear,brakePedalTravel,steerAngle
0.0,0.0,0.0,0.0,0.0,0.0
10.0,1.0,0.0,0.0,0.0,0.0
20.0,0.0,0.0,0.0,0.0,0.0
30.0,0.0,300.0,300.0,0.0,0.0
40.0,0.0,0.0,0.0,0.0,0.0
```

<h2 id="interfaces"> Simulation Interfaces </h2>

There are two primary simulation interfaces (for which there is an additional variation of the second for ease of use).

<h3 id="minter"> Main Interface </h3>

The main interface is used by navigating to ```Sims-Data/FullVehicleSim/``` and running the sim as ```python main.py -c <control_inputs_file>``` where the control input files are typically stored in ```SimulationControlInputs/```. Doing so, will run the simulation (which should take less than 30 sec to run <30 min of car simulation plus however long it takes the libraries to load). The main interface handles repeated calls to ```stepState``` in ```engine.py``` which updates a world array 1 step at a time via the 

This will output a file called ```simulation_output.parquet``` which can be analyzed for how the car performed in the simulation.

![excalidraw_stepStateModel.png](<refs/excalidraw_stepStateModel.png>)

<h3 id="sbsinter"> Step by Step Interface </h3>

The second interface is via the function ```dynamicStepState``` which can be imported from ```engine.py```. This function takes a 1d array of data that is the simulation inputs concatenated with the previous state of the vehicle and returns a new state. The value of each column can be found in [paramLoader.py](paramLoader.py) as each of the variables that begin with var. To use this, first choose a starting state for each of the variables following ```varSteerAngle```.

![excalidraw_dynamicStepStateModel.png](<refs/excalidraw_dynamicStepStateModel.png>)

A critical part of setting this up is importing the full vehicle sim as a library properly. The following steps should be followed:

1. Ensure your working directory is adjacent to a clone of ```Sims-Data``` and that you have the ```main-sim-maintenence``` branch checked out from git. This would look like:
```
Documents/
    Sims-Data/
    Your Project Folder/
```
2. Next, run ```python -m pip install -e ../Sims-Data``` from inside the new working directory. Ideally you would do this in a vm like conda. This will install the necessary libraries and package the full vehicle sim as a library for your python installation. Importantly, this will currently statically define the parameters in ```params.json5``` so if you want to choose specific parameters like step size, you should change them before you install the library. Installing the library is also a short process so you can change the parameters and reinstall fairly quickly.
3. Import dynamicStepState from ```FullVehicleSim.engine``` and get to work! An example is demonstrated below.

If you want a farily default starting state, you can use this and replace the first 6 values with your desired control values for future steps. This hasn't been set up for time super well so you should probably update the time yourself to match the step size.

```
arr = np.array(
    [0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 1, 0, 
     0, 0, 25, 25, 1, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 80000, 0, 120, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 180, 0, 0]
)
```

Here is an example program I ran from ```Driverless/temp/test.py```
```python
import matplotlib.pyplot as plt
import numpy as np
from FullVehicleSim.engine import dynamicStepState
from FullVehicleSim.paramLoader import *

arr1 = np.array(
    [0, 1, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 1, 0, 
     0, 0, 25, 25, 1, 
     0, 0, 0, 0, 0, 
     0, 0, 0, 0, 0, 
     0, 80000, 0, 120, 0, 
     0, 0, 0, 0, 0, 
     0, 0, 180, 0, 0]
)

data = np.zeros((1000, arr1.shape[0]))
data[0, :] = arr1

for step in range(1, data.shape[0]):
    data[step] = dynamicStepState(data[step-1])

t = np.arange(data.shape[0]) / 100

plt.plot(t, data[:, varSpeed])
plt.show()
```

<h2 id="msc"> Main Simulation Systems </h2>

The simulation is divided into subsystems. The Powertrain folder contains the entire powertrain model, namely the tractive battery electrical/thermal systems. The mechanical system includes/will include suspension modeling, steering modeling, aero modeling, brake modeling, and tire modeling. 

<h3 id="ptr"> Powertrain </h3>


<h3 id="mech"> Mechanical System </h3>
