## Drag Calculations
- edited blueMaxAnalysis -- added physics of the Drag Equation and used rolling resistance tests. [blueMaxDragAnalysis.py](../Data/blueMaxDragAnalysis.py)
- calculated drag by pulling RR tests and initially creates a speed array, then looks at residuals with the difference of predicted vs actual, then changes our estimates to lower error, and repeats until the values all line up
-  Speed vs Time and SpeedxDrag vs Time graphs
<img width="1206" height="670" alt="image" src="https://github.com/user-attachments/assets/b1cf47bf-f9d7-474a-87d9-29674864c157" />

## Drag Testing Ideas
- accelerate to high speed, shift to neutral, measure deacceleration, and use for better drag and RR calculations. [drag_testing.md](<Testing Plan Documentation/drag_test.md>)

## Downforce Calculation
- built downforce calculations using damping forces - validation plot from time 30s-45s
<img width="2100" height="1500" alt="image" src="https://github.com/user-attachments/assets/6611cfcc-1e4a-46f4-9afe-c831c23390a1" />
