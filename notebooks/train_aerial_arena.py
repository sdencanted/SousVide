import torch
torch.set_float32_matmul_precision('high')

import sousvide.synthesize.rollout_generator as rg
import sousvide.synthesize.observation_generator as og
import sousvide.instruct.train_policy as tp
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.plot_learning as pl
import sousvide.flight.deploy_figs as df


# =========================================
# Aerial Arena Test
# =========================================

# cohort = "aerial_arena"               # Cohort name (parent folder) for the robustness test
# scene = "aerialarena_square_16x/splatfacto/2026-08-06_013130_clean"                  # Scene name for the robustness test
# courses = ["aerialarena_test"]              # Courses to be used in the robustness test

cohort = "aerial_arena"               # Cohort name (parent folder) for the robustness test
scene = "aerialarena_20260907/splatfacto/2026-09-11_133728_clean"                  # Scene name for the robustness test
courses = ["aerialarena_rect"]              # Courses to be used in the robustness test

# =========================================
# Robustness Test
# =========================================

# cohort = "robustness"               # Cohort name (parent folder) for the robustness test
# scene = "mid_gate"                  # Scene name for the robustness test
# courses = ["traverse"]              # Courses to be used in the robustness test

# # =========================================
# # Cluttered Test
# # =========================================

# cohort = "cluttered"               # Cohort name (parent folder) for the cluttered test
# scene = "backroom"                  # Scene name for the cluttered test
# courses = ["circuit"]              # Courses to be used in the cluttered test


# Pilot roster
roster = [
    "Maverick",
    # "Iceman"
    ]

# Data synthesis method.
# data_method = "data_alpha"          # Small data set for initial testing (use only to get a feel for the system).
data_method = "data_beta"           # Medium data set for training
# data_method = "data_gamma"          # Large data set for training

# Evaluation methods
# eval_method = "eval_single"         # Evaluate over a single trajectory, ideal frame and no noise.
eval_method = "eval_nominal"        # Evaluate over 10 trajectories, non-ideal frame and noise.
# eval_method = "eval_challenged"     # Evaluate over 10 trajectories, non-ideal frame and some noise.
# # eval_method = "eval_extreme"        # Evaluate over 10 trajectories after putting the drone and pilot through a washing machine.

# Event representations generated together from the same v2e stream.
event_surface_modalities = ("event_bin", "event_eros", "event_tos")
event_voxel_modalities = ("event_voxel_grid", "event_voxel_grid_polarity")
all_event_modalities = event_surface_modalities + event_voxel_modalities

# Surface defaults. Replace either assignment with a preset below as needed.
# event_modalities_to_generate = event_surface_modalities
# event_modalities_to_train = event_surface_modalities
# event_modalities_to_generate = event_voxel_modalities  # Generate 5/10-channel voxels.
# event_modalities_to_train = event_voxel_modalities     # Train both voxel models.
# event_modalities_to_generate = all_event_modalities    # Generate every representation.
# event_modalities_to_train = all_event_modalities       # Train every representation.
# Voxel inputs require an SVNet pilot such as Maverick or Iceman; DINO is RGB-only.
event_modalities_to_generate = ("event_cloud","event_eros")
event_modalities_to_train = ("event_cloud","event_eros")
# deployment_modality = event_modalities_to_train[0]
event_surface_options = {
    "event_eros": {"kernel_size": 7, "decay": 0.3},
    # "event_tos": {"kernel_size": 5, "parameter": 2.0},
}
cloud_options = {"num_points": 8192, "seed": 0}
deployment_modality = event_modalities_to_train[1]



tp.train_roster(
    cohort,roster,"commNet",300,
    image_modality="rgb",
    regen=False,
    deployment=(courses[0],scene,eval_method),numerical_mode="original",batch_size=8,lr=0.0008,dataloader_num_workers=0)
pl.plot_losses(
    cohort,roster,"commNet",use_log=True,
    image_modality="rgb")

