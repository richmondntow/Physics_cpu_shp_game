# 2D SPH Fluid Simulation (CPU) with PyGame Frontend

This project is a Smoothed Particle Hydrodynamics (SPH) fluid simulation running **without GPU**, using NumPy + Numba for performance and **PyGame** for real-time rendering and interactivity.
Works well on macOS (MacBook Pro 2019) and other desktop platforms.

## Features
- CPU SPH simulation (NumPy + Numba)
- Real-time rendering with PyGame
- Interactive controls:
  - Pause / Resume (Space)
  - Add particles with mouse click (left button)
  - Reset (R)
  - Increase / decrease particle count reasonably (I / K)
  - Adjusted simulation parameters in code (top of main.py)

## Requirements
- Python 3.8+
- Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Notes
- This uses a brute-force O(N^2) neighbor search in the physics update; keep N reasonably small (e.g., 200-800) for smooth performance on a laptop CPU.
- To scale, implement spatial hashing/grid neighbor search (I can help add that next).
- Controls are handled in the PyGame window. Close the window or press ESC to exit.
