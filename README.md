# SOUS VIDE

## Installation
1) Clone repository and load the submodules.
```
git clone https://github.com/StanfordMSL/SousVide.git
git submodule update --recursive --init
```
2) Build ACADOS locally.
```
# Navigate to acados folder
cd <repository-path>/SousVide/FiGS/acados/

# Compile
mkdir -p build
cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4

# Add acados paths to bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
export ACADOS_SOURCE_DIR="<acados_root>"
```
3) Set up conda environment (in the main directory)
```
# Navigate to environment config location
cd <repository-path>/SousVide/

# Create and activate
conda env create -f environment_x86.yml
conda activate kitchen
```
5) Download Example GSplats
```
# Navigate to gsplats parent folder
cd <repository-path>/SousVide/

# Download the zip-ed file below and unpack the contents (capture and workspace) into the gsplats folder
https://drive.google.com/file/d/1kW5dzsfD3rbRA3RIQDyJPG6_UJaO9ALP/view
```

## Run SOUS VIDE Examples
Check out the notebook examples in the notebooks folder:
  1. <b>figs_examples</b>: Example code for generating GSplats and executing trajectories within them (using FiGS).
  2. <b>sous_vide_examples</b>: Use this notebook to try out two of the policies generated in the paper.

## Train with grayscale images

Use the `grayscale` image modality when generating CommNet observations and
training. Grayscale observations are derived from the existing lossless RGB
rollouts, so the rollout data does not need to be regenerated.

```python
import sousvide.synthesize.observation_generator as og
import sousvide.instruct.train_policy as tp

og.generate_observation_data(
    cohort, roster, networks=["commNet"], image_modality="grayscale")
tp.train_roster(
    cohort, roster, "commNet", 300, image_modality="grayscale")
```

Deployment uses the same option: `deploy_roster(...,
image_modality="grayscale")`. Grayscale observations and CommNet weights are
stored separately from their RGB counterparts.


## [COMING SOON: Oct 2025] Deploy SOUS VIDE in the Real World
Deploy SOUS VIDE policies on an [MSL Drone](https://github.com/StanfordMSL/TrajBridge/wiki/3.-Drone-Hardware). Tutorial and code coming soon!
