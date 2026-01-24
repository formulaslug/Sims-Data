# Full Vehicle Simulation

Documentation explaining the different software bits of the full vehicle simulation and how to run it.

## Table of Contents

1. [Introduction](#intro)
1. [Simulation Inputs](#siminputs)
    1. [Simulation Parameters](#pars)
    1. [Simulation Controls](#ctrls)
1. [Main Simulation Components](#msc)
    1. [Powertrain](#ptr)
    1. [Mechanical System](#mech)
1. [Model Training](#mt)
    1. [Tire Modeling](#tm)
    1. [Voltage Modeling](#vm)

<h2 id="intro"> Introduction </h2>

This folder contains the entirety of the full vehicle simulation for Formula Slug. It is run from [main.py](main.py) where you provide the necessary parameters to begin the simulation. There are a few main folders: ```Powertrain``` (Powertrain/Battery System), ```Mech```  (Mechanical System), ```SimulationInputs``` and ```TireModel``` (Tire Model Training). ```Powertrain``` and ```Mech``` are model bits used while running the simulation. ```SimulationInputs``` contains various simulation starting points: Initial parameters like timeStep, and control inputs for the duration of a simulation.

If you want more information on the basis of some of these simulation parts, you can find them in [GeneralModelingNotes.md](GeneralModelingNotes.md)

<h2 id="siminputs"> Simulation Inputs </h2>

<h3 id="pars"> Simulation Parameters </h3>

This is a json input file with a bunch of inputs for the simulation. Unfortunately JSON doesn't support comments so the unit explanations have to go here. It looks something like this:
```json
{
    "stepsPerSecond": 100,
    "simulationDuration": 20, // Seconds 
    "maxCurrentLimit": 300, //Amps
    "VehicleSOC": 1, // Out of 1
    "AmbientTemperature": 23, // Deg C
    "initialBatteryTemperature": 23, // Deg C
    "initialBrakeTemperature": 23 // Deg C
}
```

<h3 id="ctrls"> Simulation Controls </h3>

Simulation controls is a ```.csv``` or ```.parquet``` file that contains ```time (sec)```, ```throttle (-1 to 1)```, ```brake(Pressure)```, and ```steering angle``` inputs. Time is in seconds, throttle is -1 to 1 (includes regen),  



<h2 id="msc"> Main Simulation Systems </h2>

The simulation is divided into subsystems. The Powertrain contains the entire powertrain model, namely the tractive battery electrical/thermal systems. The mechanical system includes/will include suspension modeling, steering modeling, aero modeling, brake modeling, and tire modeling. 

<h3 id="ptr"> Powertrain </h3>
<h3 id="mech"> Mechanical System </h3>
