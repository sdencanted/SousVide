import numpy as np
import torch
import os
import sousvide.visualize.plot_3D as p3
import sousvide.visualize.plot_time as pt
import sousvide.visualize.rich_utilities as ru

from rich.text import Text

from typing import List

def plot_rollout_data(cohort:str,Nsamples:int=50,
                      show_3D:bool=True,show_time:bool=True):
    """"
    Plot the rollout data for a cohort.
    """

    # Generate some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort)
    rollout_path = os.path.join(cohort_path,"rollout_data")

    # Initialize the console variable
    console = ru.get_console()
    subunits = "dpts"

    # Load Flight Data
    Ntot = 0
    dSets_desc = []
    for dSet in os.listdir(rollout_path):
        dSet_path = os.path.join(rollout_path, dSet)
        
        # Extract all files in child folder "trajectories"
        traj_folder_path = os.path.join(dSet_path, "trajectories")
        traj_files = os.listdir(traj_folder_path)
        traj_file_paths = [os.path.join(traj_folder_path, f) for f in traj_files]

        # Extract diagnostics data
        Ndsets = len(traj_file_paths)
        Ntot += Ndsets

        Ndata = 0
        for traj_file_path in traj_file_paths:
            trajectories = torch.load(traj_file_path,weights_only=False)

            for trajectory in trajectories:
                Ndata += trajectory["Ndata"]

        dSet_desc = ru.get_data_description(dSet, Ndata, subunits=subunits) + f" [{Ndsets} datasets]"
        dSets_desc.append(dSet_desc)

        # Plot example trajectory
        if show_3D == True or show_time == True:
            # Load a random trajectory file
            traj_file = np.random.choice(traj_files)
            traj_file_path = os.path.join(traj_folder_path, traj_file)
            trajectories = torch.load(traj_file_path,weights_only=False)
            
            # Trim the number of samples
            if Nsamples > len(trajectories):
                Nsamples = len(trajectories)
                console.print(f"Only {Nsamples} samples available in \[{dSet}]>{traj_file}. Showing all samples.")
            else:
                console.print(f"Showing {Nsamples} samples from \[{dSet}]>{traj_file}")
            trajectories = trajectories[0:Nsamples]

        # Plot the data
        if show_3D == True:
            p3.RO_to_3D(trajectories,scale=0.5)
        if show_time == True:
            pt.RO_to_time(trajectories)

    # Collate diagnostics
    dSets_desc = [Text.from_markup(dSet_desc).plain for dSet_desc in dSets_desc]  # Strip rich tags
    dSets_desc = "\n".join(dSets_desc)  # Join descriptions into a single string
    console.print(
        f"Rollout produced {Ntot} datasets with the following courses: \n"
        f"{dSets_desc}", style="white")