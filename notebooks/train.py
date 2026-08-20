import torch
torch.set_float32_matmul_precision('high')

import sousvide.synthesize.rollout_generator as rg
import sousvide.synthesize.observation_generator as og
import sousvide.instruct.train_policy as tp
import sousvide.visualize.plot_synthesize as ps
import sousvide.visualize.plot_learning as pl
import sousvide.flight.deploy_figs as df


# =========================================
# Robustness Test
# =========================================

# cohort = "robustness"               # Cohort name (parent folder) for the robustness test
# scene = "mid_gate"                  # Scene name for the robustness test
# courses = ["traverse"]              # Courses to be used in the robustness test

# # =========================================
# # Cluttered Test
# # =========================================

cohort = "cluttered"               # Cohort name (parent folder) for the cluttered test
scene = "backroom"                  # Scene name for the cluttered test
courses = ["circuit"]              # Courses to be used in the cluttered test

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

# # Train the Policy
tp.train_roster(
    cohort,roster,"histNet",200,
    dataloader_num_workers=8,cuda_prefetch=True,
    numerical_mode="original")

# # Plot the histNet losses
# pl.plot_losses(cohort,roster,"histNet",use_log=True)

# Refresh CommNet inputs after HistNet training. Keep generation separate so
# original numerical mode starts from fixed observation files and RNG state.
og.generate_observation_data(
    cohort,roster,networks=["commNet"],image_modality="rgb")

tp.train_roster(
    cohort,
    roster,
    "commNet",
    300,
    image_modality="rgb",
    batch_size=64,
    dataloader_num_workers=8,
    cuda_prefetch=True,
    numerical_mode="original",
    deployment=(courses[0], scene, eval_method),
)

og.generate_observation_data(
    cohort,roster,networks=["commNet"],
    image_modality="kronecker_delta")

tp.train_roster(
    cohort,
    roster,
    "commNet",
    300,
    image_modality="kronecker_delta",
    batch_size=64,
    dataloader_num_workers=8,
    cuda_prefetch=True,
    numerical_mode="original",
    deployment=(courses[0], scene, eval_method),
)
