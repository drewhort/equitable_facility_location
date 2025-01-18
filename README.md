# A Scalable Approach to Equitable Facility Location

This repository contains code associated with our paper proposing a computationally tractable approach to optimize the Kolm-Pollak Equally-Distributed Equivalent (EDE) in facility location problems. The approach enables optimization of large-scale facility location problems while explicitly considering equity.

## Paper Status

This paper is currently under review at Transportation Research Part B. In the meantime, please cite:

Horton, D., Logan, T., Murrell, J., Skipper, D., & Speakman, E. (2024). A scalable approach to equitable facility location. *arXiv preprint arXiv:2401.15452*.

## Overview

The code in this repository implements our novel framework for equitable facility location, introducing a linearized proxy for the Kolm-Pollak EDE metric to balance efficiency and fairness. The approach enables:

- Optimization of facility locations considering both average access and equity
- Handling of real-world considerations like capacity constraints and location-specific penalties
- Solution of large-scale problems (tested on instances with over 200 million binary variables)
- Analysis of demographic subgroups to evaluate environmental justice implications

## Repository Structure

The `src` directory contains the core implementation files:
- `compute_ede.py`: Functions for computing the Kolm-Pollak EDE
- `compute_kappa.py`: Functions for calculating and scaling the inequality aversion parameter
- `equitablefacilitylocation.py`: Core facility location model implementation
- `kpcon_main.py`: Main execution script with constraints
- `main.py`: Basic execution script

