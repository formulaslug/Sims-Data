# Data
Storage for all scripts and related work on car data analysis. Please refrain from leaving data in this folder and Sims-Data as a whole.

If you see anything wrong with these docs, please fix it as you see it! If you aren't sure, then ask questions!

# Docs

1. [General Guide](../Docs/General%20Guide.md)
1. [Library Info](../Docs/Library%20Info.md)
1. [Using Polars](../Docs/Using%20Polars.md)
1. [.fsdaq Format](../Docs/2025%20Custom%20Binary.md)
1. [Saving Data from the Car](../Docs/Saving%20Data.md)
1. [Ideas](../Docs/ideas.md)

## Dev setup

### Prerequisites
- Python
- Git
- Libraries:
    1. polars
    1. matplotlib
    1. numpy
    1. scipy
    1. json5

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd Sims-Data
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
pip install -e .
```

4. Set up git hooks:
```bash
chmod +x setup-hooks.sh
./setup-hooks.sh
```
