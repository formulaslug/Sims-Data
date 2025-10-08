# fs-data
Storage for all data gathered of all cars for modeling/simulating purposes

If you see anything wrong with these docs, please fix it as you see it! If you aren't sure, then ask questions!

# Docs

1. [General Guide](Docs/General%20Guide.md)
1. [Library Info](Docs/Library%20Info.md)
1. [Using Polars](Docs/Using%20Polars.md)
1. [.fsdaq Format](Docs/2025%20Custom%20Binary.md)
1. [Saving Data from the Car](Docs/Saving%20Data.md)
1. [Ideas](ideas.md)
1. [IMU Calibration](<IMU Calibration.md>)



## Description of Different Runs
1. September 6th -> Relatively Slow driving around a parking lot. Torque is inverted and capped at 1/3.
2. September 10th -> Similar Conditions to Sep 6th 
3. September 11th -> Similar Conditions to Sep 6th but torque gets switched back to positive (probably)
4. September 18th -> Added IMU, relatively normal slow driving in parking lot.
5. Sep 19th same as 18th
6. Sep 27th parking lot again
7. Dec 2nd -> Part 1 is autocross and acceleration style, Part 2 is endurance.

## Dev setup

### Prerequisites
- Python 3.13
- pip (Python package installer)
- git

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd fs-data
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up git hooks:
```bash
chmod +x setup-hooks.sh
./setup-hooks.sh
```
