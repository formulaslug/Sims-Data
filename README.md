# Sims-Data

Vehicle Dynamics Simulations and Data Analysis for Formula Slug written from the ground up in python.

You can install dependencies either from requirements.txt or environment.yml

## Docs
- [Full Vehicle Simulation Documentation](FullVehicleSim/README.md)
- [General Guide to fs-data (Accompanying data repository)](<Docs/General Guide.md>)

## Drag Calculations
- edited blueMaxAnalysis -- added physics of the Drag Equation and used rolling resistance tests. (https://github.com/formulaslug/Sims-Data/blob/43-drag-and-downforce-calculations/Data/blueMaxDragAnalysis.py)
- calculated drag by pulling RR tests and initially creates a speed array, then looks at residuals with the difference of predicted vs actual, then changes our estimates to lower error, and repeats until the values all line up
-  Speed vs Time and SpeedxDrag vs Time graphs
<img width="1206" height="670" alt="image" src="https://github.com/user-attachments/assets/b1cf47bf-f9d7-474a-87d9-29674864c157" />

## Drag Testing Ideas
- accelerate to high speed, shift to neutral, measure deacceleration, and use for better drag and RR calculations
  (https://github.com/formulaslug/Sims-Data/blob/43-drag-and-downforce-calculations/testing-plan/drag_test.md)
