import numpy as np
import os
import time
import shutil
import contextlib
import io
import torch
import torch.nn as nn
import torch.optim as optim
import sousvide.synthesize.observation_generator as og
import sousvide.visualize.rich_utilities as ru

from torch.utils.data import DataLoader
from rich.progress import Progress
from sousvide.control.pilot import Pilot
from sousvide.control.policy import Policy
from sousvide.control.networks.base_net import BaseNet
from sousvide.instruct.losses import *
from sousvide.instruct.synthesized_data import *
import sousvide.flight.deploy_figs as df

def train_roster(cohort_name:str,roster:list[str],network_name:str,Neps:int,
                 regen:bool=False,deployment:None|tuple[str,str,str]=None,
                 lim_sv:int=50,lr:float=1e-4,batch_size:int=64):
    
    # Initialize the console variable
    console = ru.get_console()
    progress = ru.get_training_progress()

    # Re-generate observation data
    if regen:
        console.print("Regenerating observation data...")
        og.generate_observation_data(cohort_name,roster)
    else:
        console.print("Using existing observation data...")

    with progress:
        # Train each student
        for student in roster:
            # Initialize student progress bar
            student_desc = f"[bold green3]{student:>8} > {network_name}[/]"
            student_task = progress.add_task(student_desc,total=Neps,loss=0.0,units='epochs')
            student_bar = (progress,student_task)

            # Train the student
            train_student(cohort_name,student,network_name,Neps,
                          deployment,lim_sv,lr,batch_size,
                          student_bar)
            progress.refresh()


def train_student(cohort_name:str,student_name:str,network_name:str,Neps:int,
                  deployment:None|tuple[str,str,str]=None,
                  lim_sv:int=10,lr:float=1e-4,batch_size:int=64,
                  progress_bar:tuple[Progress,int]=None
                  ) -> None:

    # Record the start time
    start_time = time.time()

    # Pytorch Config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    criterion = LossFn()

    # Load the student
    student = Pilot(cohort_name,student_name)

    # Extract network if it exists
    if network_name not in student.policy.networks:
        # Update the progress bar if it exists
        if progress_bar is not None:
            progress,student_task = progress_bar
            progress.update(student_task,description=f"{student_name} does not use {network_name}.")

        return
    else:
        network:BaseNet = get_network(student.policy,network_name)    

    # Set parameters to optimize over
    opt = optim.Adam(network.parameters(),lr=lr)

    # Some Useful Paths
    student_path = student.path
    losses_path  = os.path.join(student_path,"losses_"+network_name+".pt")

    # Load loss log if it exists
    if os.path.exists(losses_path):
        prev_losses_log = torch.load(losses_path,weights_only=False)
    else:
        prev_losses_log = {}

    # Initialize the loss entry
    loss_entry = {
        "network": network_name,
        "N_eps": None, "Nd_tn": None, "Nd_tt": None, "t_tn": None,
        "Loss_tn": [], "Loss_tt": [], "Eval_tte": [],
    }

    # Run initial evaluation (if applicable)
    if deployment is not None:
        # Unpack the deployment
        course,scene,eval_method = deployment

        # Check if the checkpoint path exists
        ckpts_path  = os.path.join(student_path,"ckpts")
        if not os.path.exists(ckpts_path):
            os.makedirs(ckpts_path)

        # Evaluate using a deployment
        with contextlib.redirect_stdout(io.StringIO()):
            metric = df.deploy_roster(cohort_name,course,scene,eval_method,[student_name],mode="evaluate")

        # Initial checkpoint
        ckpt_name = network_name+"_ckpt"+str(0).zfill(3)
        ckpt_path = os.path.join(ckpts_path,ckpt_name+".pt")
        torch.save(network,ckpt_path)

        # Extract the metrics
        eval0 = (0,metric[student_name]["TTE"]["mean"])
    else:
        eval0 = []
        
    # Training + Testing Loop
    Loss_tn,Loss_tt = [],[]
    Eval_tte = [eval0]
    for ep in range(Neps):
        # Get Observation Data Files (Paths)
        od_train_files,od_test_files = get_data_paths(cohort_name,student.name,network_name)

        # Training
        epLosses_tn,Ndata_tn = [],0
        for od_train_file in od_train_files:
            # Load Datasets
            dataset = generate_dataset(od_train_file,device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

            # Training
            for xnn,ylb in dataloader:
                # Forward Pass
                ypd = network(xnn)
                loss = criterion(ypd,ylb)

                # Backward Pass
                loss.backward()
                opt.step()
                opt.zero_grad()

                # Save loss logs
                epLosses_tn.append(batch_size*loss.item())
                Ndata_tn += batch_size

        # Testing
        epLosses_tt,Ndata_tt = [],0
        for od_test_file in od_test_files:
            # Load Datasets
            dataset = generate_dataset(od_test_file,device)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = False)

            # Testing
            for xnn,ylb in dataloader:
                # Forward Pass
                ypd = network(xnn)
                loss = criterion(ypd,ylb)

                # Save loss logs
                epLosses_tt.append(batch_size*loss.item())
                Ndata_tt += batch_size

        # Loss Diagnostics
        epLoss_tn = sum(epLosses_tn)/Ndata_tn
        epLoss_tt = sum(epLosses_tt)/Ndata_tt
        Loss_tn.append((ep+1,epLoss_tn))
        Loss_tt.append((ep+1,epLoss_tt))
        
        # Update the progress bar
        if progress_bar is not None:
            progress,student_task = progress_bar
            progress.update(student_task,loss=epLoss_tn,advance=1)
            progress.refresh()

        # Save at intermediate steps and at the end
        if ((ep+1) % lim_sv == 0) or (ep+1==Neps):
            # Record the end time
            end_time = time.time()
            t_train = end_time - start_time

            # Save the latest network
            netw_path = os.path.join(student_path,network_name+".pt")
            torch.save(network,netw_path)            
            
            # Evaluation (optional)
            if deployment is not None:
                # Unpack the deployment
                course,scene,eval_method = deployment
                
                # Save as a checkpoint
                ckpt_name = network_name+"_ckpt"+str(ep+1).zfill(3)
                ckpt_path = os.path.join(ckpts_path,ckpt_name+".pt")
                torch.save(network,ckpt_path)

                # Evaluate using a deployment
                with contextlib.redirect_stdout(io.StringIO()):
                    metric = df.deploy_roster(cohort_name,course,scene,eval_method,[student_name],mode="evaluate")

                # Extract the metrics
                Eval_tte.append((ep+1,metric[student_name]["TTE"]["mean"]))
            else:
                Eval_tte.append([])

            # Update the loss entry
            loss_entry["Loss_tn"] = np.array(Loss_tn).T
            loss_entry["Loss_tt"] = np.array(Loss_tt).T
            loss_entry["Eval_tte"] = np.array(Eval_tte).T
            loss_entry["Nd_tn"],loss_entry["Nd_tt"] = Ndata_tn,Ndata_tt
            loss_entry["t_tn"],loss_entry["N_eps"] = t_train,ep+1

            # Save Loss
            timestamp = time.strftime("%y%m%d_%H%M%S")
            log_name = f"log_{timestamp}"

            curr_losses_log = prev_losses_log.copy()
            curr_losses_log[log_name] = loss_entry
            
            torch.save(curr_losses_log,losses_path)

    # Pick the checkpoint with the best evaluation metric
    if deployment is not None:
        # Get the best checkpoint
        best_ckpt = min(Eval_tte,key=lambda x: x[1])[0]
        best_ckpt = network_name+"_ckpt"+str(best_ckpt).zfill(3)
        ckpt_path = os.path.join(ckpts_path,best_ckpt+".pt")

        # Load the best checkpoint
        best_network = torch.load(ckpt_path,weights_only=False)

        # Save the best network
        network_path = os.path.join(student_path,network_name+".pt")
        torch.save(best_network,network_path)

        # Delete the checkpoints
        shutil.rmtree(ckpts_path)

        # Print the best checkpoint
        ru.console.print(f"[bold green3]{student_name} > {network_name}[/] : Best checkpoint is {best_ckpt}.")

    # Cap off progress update
    progress.refresh()

def get_network(policy:Policy,net_name:str|list[str]) -> BaseNet:
    """
    Get the networks based on the list of network names.

    Args:
        policy:     The policy object.
        net_name:   The list of network names or a single network name.

    Returns:
        network:    The target network.
    """

    # Get the network
    network = policy.networks[net_name]
    
    # Lock/Unlock the network
    network.train()
    for param in network.parameters():
        param.requires_grad = True
    
    return network