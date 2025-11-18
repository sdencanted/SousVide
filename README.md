# SOUS VIDE

## Installation

### Prerequisites: Install FiGS First
SousVide depends on FiGS. Install FiGS in a conda environment first:

1) Clone and set up FiGS:
```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
cd FiGS-Standalone
conda env create -f environment.yml
conda activate FiGS
git submodule update --init --recursive
pip install -e .
```

### Install SousVide
2) Clone SousVide repository (no submodules needed):
```bash
# From parent directory
cd ..
git clone https://github.com/StanfordMSL/SousVide.git
cd SousVide
```

3) Update the FiGS conda environment with SousVide dependencies:
```bash
conda activate FiGS
conda env update -f environment_sousvide.yml
```

4) Install SousVide:
```bash
pip install -e .
```

5) Download Example GSplats (optional):
```bash
# Navigate to gsplats parent folder
cd <repository-path>/SousVide/

# Download the zip-ed file below and unpack the contents (capture and workspace) into the gsplats folder
https://drive.google.com/file/d/1kW5dzsfD3rbRA3RIQDyJPG6_UJaO9ALP/view
```

## Run SOUS VIDE Examples
Check out the notebook examples in the notebooks folder:
  1. <b>figs_examples</b>: Example code for generating GSplats and executing trajectories within them (using FiGS).
  2. <b>sous_vide_examples</b>: Use this notebook to try out two of the policies generated in the paper.


## [COMING SOON: Oct 2025] Deploy SOUS VIDE in the Real World
Deploy SOUS VIDE policies on an [MSL Drone](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware). Tutorial and code coming soon!