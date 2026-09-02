import numpy as np
import matplotlib.pyplot as plt
import os
import torch

import sousvide.visualize.plot_3D as p3
import sousvide.visualize.rich_utilities as ru
import sousvide.flight.flight_helper as fh
from sousvide.synthesize.image_modality import validate_visual_modality

from typing import List

def plot_losses(cohort_name:str, roster:List[str], network_name:str,
                Nln:int=70,use_log:bool=True,image_modality:str|None=None):
    """
    Plot the losses for each student in the roster.
    """

    if image_modality is not None:
        image_modality = validate_visual_modality(image_modality)

    # Initialize the rich variables
    console = ru.get_console()

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path, "cohorts", cohort_name)

    # Compile the learning summary header
    learning_summary = (
        f"{'=' * Nln}\n"
        f"Cohort : [bold cyan]{cohort_name}[/]\t\t"
        f"Network: [bold cyan]{network_name}[/]\n"
        f"{'=' * Nln}\n")

    # Plot the losses for each student
    student_data = {}
    for student in roster:
        # Load the losses for each student
        student_path = os.path.join(cohort_path, "roster", student)
        losses_path = os.path.join(student_path, f"losses_{network_name}.pt")
        if os.path.exists(losses_path):
            losses: dict = torch.load(losses_path,weights_only=False)
        else:
            student_summary = (
                f"{'-' * Nln}\n"
                f"Student [bold cyan]{student}[/] does not have a [bold cyan]{network_name}[/].\n"
            )
            learning_summary += student_summary

            continue

        # Gather plot data
        Loss_tn, Loss_tt, Eval_tte = np.zeros((2,0)), np.zeros((2,0)), np.zeros((2,0))
        Nd_tn, Nd_tt, T_tn = [], [], []
        Neps_tot = 0
        for loss_data in losses.values():
            if (image_modality is not None and
                    loss_data.get("image_modality","rgb") != image_modality):
                continue
            # Extract the loss data
            loss_tn,loss_tt = loss_data["Loss_tn"],loss_data["Loss_tt"]
            if np.any(loss_data["Eval_tte"]):
                eval_tte = loss_data["Eval_tte"]
            else:
                eval_tte = np.zeros((2,0))

            # Update their epoch counts
            loss_tn[0,:] += Neps_tot
            loss_tt[0,:] += Neps_tot
            eval_tte[0,:] += Neps_tot

            # Append the loss data
            Loss_tn = np.hstack((Loss_tn,loss_tn))
            Loss_tt = np.hstack((Loss_tt,loss_tt))
            Eval_tte = np.hstack((Eval_tte,eval_tte))

            # Update the total number of epochs
            Neps_tot += loss_data["N_eps"]

            # Append the number of data points for training and testing
            Nd_tn.append(loss_data["Nd_tn"])
            Nd_tt.append(loss_data["Nd_tt"])

            # Accumulate the training time
            T_tn.append(loss_data["t_tn"])

        # Compile the student plot data
        if Neps_tot == 0:
            learning_summary += (
                f"Student [bold cyan]{student}[/] has no {network_name} "
                f"losses for [bold cyan]{image_modality}[/].\n")
            continue

        student_data[student] = {
            "Train": Loss_tn,
            "Test": Loss_tt,
            "TTE": Eval_tte if Eval_tte.size > 0 else None
        }

        # Generate student summary
        Nd_mean = (np.mean(Nd_tn), np.mean(Nd_tt))
        T_tn_tot = np.sum(T_tn)
        loss_tn_f, loss_tt_f = Loss_tn[1,-1], Loss_tt[1,-1]
        eval_tte_f = Eval_tte[1,-1] if Eval_tte.size > 0 else None

        student_summary = ru.get_student_summary(
            student, Neps_tot, Nd_mean, T_tn_tot,
            loss_tn_f, loss_tt_f, eval_tte_f)

        learning_summary += student_summary

    # Compile the learning summary footer
    learning_summary += (f"{'=' * Nln}")

    # Print the learning summary
    console.print(learning_summary)

    # Plot the losses
    titles = ["Training", "Testing"]
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    
    # Plot the losses
    for student_name,data in student_data.items():
        axs[0].plot(data["Train"][0,:],data["Train"][1,:], label=student_name)
        axs[1].plot(data["Test"][0,:],data["Test"][1,:], label=student_name)

    axs[0].set_ylabel('Loss')
    for idx,ax in enumerate(axs):
        ax.set_title(titles[idx])
        ax.set_xlabel('Epochs')
        ax.legend(loc='upper right', fontsize='small')

        if use_log:
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Plot the TTE (if available)
    if any([data["TTE"] is not None for data in student_data.values()]):
        fig, ax = plt.subplots(figsize=(6, 3))
        for student_name,data in student_data.items():
            if data["TTE"] is not None:
                ax.plot(data["TTE"][0,:],data["TTE"][1,:], label=student_name)

        ax.set_ylabel('TTE (s)')
        ax.set_xlabel('Epochs')
        ax.set_yscale('log')
        ax.set_xlim(0, data["Train"][0,-1])
        ax.legend(loc='upper right', fontsize='small')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        plt.show(block=False)

def plot_deployments(cohort_name: str, course_name: str, roster: List[str], plot_show: bool = False):
    """
    Plot the simulations for each student in the roster.
    """

    # Initialize the rich variables
    console = ru.get_console()
    table = ru.get_deployment_table()

    # Add Expert to Roster
    roster = ["expert"] + roster

    # Some useful paths
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    deployment_folder = os.path.join(workspace_path, "cohorts", cohort_name, "deployment_data")

    for pilot in roster:
        # Load the deployment data
        try:
            deployment_path = os.path.join(deployment_folder,"sim_"+course_name+"_"+pilot+".pt")
            Trajectories = torch.load(deployment_path,weights_only=False)
            
            # Get pilot metrics
            metrics = fh.compute_flight_metrics(Trajectories)
            table = ru.update_deployment_table(table,pilot,metrics)
        except:
            console.print(f"Pilot [bold cyan]{pilot}[/] does not have a deployment.")
            continue

        # Plot the trajectories for each pilot
        if plot_show:
            console.print(f"Plotting trajectories for [bold cyan]{pilot}[/]...")
            p3.RO_to_3D(Trajectories,n=40,plot_last=True)

    # Print the summary table
    console.print(table)
