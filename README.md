# Minimal Models of Biological Tissue Development

This repository contains code to run simulations and reproduce figures from:

**"Incorporating stochastic gene expression, signaling-mediated intercellular interactions, and regulated cell proliferation in models of coordinated tissue development"**  
*Casey O. Barkan and Tom Chou*  
https://arxiv.org/abs/2501.11271

# Requirements

Python 3.9+, numpy, scipy, matplotlib, shapely

# Usage

## Example 1
```bash
cd example_1/
python run_simulation.py        # Run simulation and plot populations
python figure_2B.py             # Generate Figure 2B
python figure_3A.py             # Generate Figure 3A
python deterministic_model.py   # Generate Figure 3B (the deterministic limit)
python estimate_transition_rates.py   # Estimates cell type transition rates for the deterministic limit
```

## Example 2
```bash
cd example_2/
python run_simulation.py  # Run spatial simulation and plot populations
python figure_4B.py       # Generate Figure 4B
python figure_4C.py       # Generate Figure 4C
```
