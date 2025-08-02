# rastrigin-optimization
# Optimization of the 2D Rastrigin Function Using GA, DE, and PSO

## Overview
This project implements three population-based metaheuristic algorithms:
- **Genetic Algorithm (GA)**
- **Differential Evolution (DE)**
- **Particle Swarm Optimization (PSO)**

These algorithms minimize the 2D Rastrigin function, a widely used benchmark for testing optimization algorithms due to its multimodal landscape.

## Rastrigin Function
The function is defined as:
f(x, y) = 20 + x² + y² – 10[cos(2πx) + cos(2πy)]

Global minimum is at `(0, 0)` with `f(0, 0) = 0`.

## Repository Structure
rastrigin-optimization/
rastrigin-optimization/
├── ga.py        # Genetic Algorithm implementation
├── de.py        # Differential Evolution implementation
├── pso.py       # Particle Swarm Optimization implementation
├── data/        # CSV fitness results
├── plots/       # Convergence plots
└── report/      # Project report (PDF)


## Requirements
- Python 3.10+
- `numpy`
- `matplotlib`

Install dependencies with:
pip install numpy matplotlib

## Run algorithms individually
python ga.py   # Runs GA
python de.py   # Runs DE
python pso.py  # Runs PSO

Outputs:
Convergence plots saved in plots/
Fitness data saved in data/

## Results
All three algorithms successfully converged to the global minimum (0,0) during testing:
GA achieved the lowest mean fitness and highest robustness.
PSO demonstrated the fastest initial convergence but showed moderate variability.
DE achieved moderate performance and required careful parameter tuning.
For detailed analysis and comparative discussion, see the full report in the report/ directory.
