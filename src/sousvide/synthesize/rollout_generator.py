import numpy as np
import os
import torch

import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th

import sousvide.synthesize.synthesize_helper as sh
import sousvide.synthesize.data_compress_helper as dch
import sousvide.visualize.rich_utilities as ru
import sousvide.utilities.sousvide_utilities as svu

from figs.simulator import Simulator
from figs.dynamics.external_forces import ExternalForces
from figs.tsplines.min_time_snap import MinTimeSnap
from figs.control.vehicle_rate_mpc import VehicleRateMPC

def generate_rollout_data(cohort_name:str,course_names:list[str],
                          gsplat_name:str,method_name:str,
                          expert_name:str="Viper",bframe_name:str="carl",
                          Nro_ds:int=50,use_compress:bool=False) -> None:
    
    """
    Generates rollout data for a given cohort. The rollout data comprises a set of courses
    flown within a given scene on variations of a specific drone frame by a specific pilot
    using a user-defined data generation method. The rollout data is saved to as pairs of
    .pt files, one for trajectory data and one for image data, in a directory corresponding
    to a combination of the course, scene and method names.

    Args:
        cohort_name:    Directory to store the rollout data (and later the roster of pilots).
        course_names:   List of trajectory courses to be flown.
        gsplat_name:    3D reconstruction of the scene contained as a Gaussian Splat.
        method_name:    Data generation method detailing the sampling and simulation configs.
        expert_name:    Expert pilot (default is a vrmpc_fr) to fly the trajectories.
        bframe_name:    Base frame for flying the trajectories (default is carl).
        Nro_sv:         Number of rollouts per save.
        use_compress:   Compress the image data.

    Returns:
        None:           (flight data saved to cohort directory)
    """

    # Initialize the progress variables
    progress = ru.get_generation_progress()
    subunits = "dpts"
    sample_desc1 = "[bold dark_green]Generating rollouts...[/]"
    sample_desc2 = "[bold dark_green]Saving dataset...[/]"

    # Load configs
    gsplat = ch.get_gsplat(gsplat_name)
    method = ch.get_config(method_name,"methods")
    expert = ch.get_config(expert_name,"pilots")
    bframe = ch.get_config(bframe_name,"frames")

    # Unpack some stuff
    m_bs,kt_bs = bframe["mass"],bframe["motor_thrust_coeff"]
    kT,use_l2_time = expert["plan"]["kT"],expert["plan"]["use_l2_time"]
    hz = expert["track"]["hz"]

    # Generate rollouts for each course
    with progress:
        # Initialize sample progress bar
        sample_task = progress.add_task(sample_desc1,total=None,units='samples')

        # Initialize the simulator
        simulator = Simulator(gsplat,method)

        # Cycle through the courses
        for course_name in course_names:
            # Load and name the course_config
            course = ch.get_config(course_name,"courses")
            
            # Compute the desired variables
            mts = MinTimeSnap(course["waypoints"],hz,kT,use_l2_time)
            fex = ExternalForces(course["forces"])

            Tsd,FOd = mts.get_desired_trajectory()
            tXUd = th.TsFO_to_tXU(Tsd,FOd,m_bs,kt_bs,fex)

            # Print Desired Time Steps
            Tp = np.hstack([0.0,np.cumsum(mts.dTd)])
            ru.console.print(
                f"[bold bright_green] {course_name} - Ideal Time Steps: {np.around(Tp,3)}[/]")
            
            # Update simulation variables
            simulator.update_forces(course["forces"])
            controller = VehicleRateMPC(expert,course)

            # Get the batches of sample start times
            t0,tf = Tsd[0],Tsd[-1]
            dt_ro = method["duration"] or tf-t0
            rate = method["rate"] or 1/dt_ro
            reps = method["reps"] or 1

            Tsp_bts = sh.compute_Tsp_batches(t0,tf,dt_ro,rate,reps,Nro_ds)

            # Initialize course progress bar
            Ndata = 0
            course_desc = ru.get_data_description(course_name,Ndata,subunits=subunits)
            course_task = progress.add_task(course_desc,
                total=len(Tsp_bts), units='datasets')

            # Generate Sample Set Batches
            for idx_bt,Tsp_bt in enumerate(Tsp_bts):
                # Generate sample frames and perturbations
                Frames = sh.generate_frames(
                    Tsp_bt, bframe, method["randomization"]["parameters"]
                )
                Perturbations = sh.generate_perturbations(
                    Tsp_bt, tXUd, method["randomization"]["initial"]
                )

                # Update the samples progress bar config
                progress.reset(sample_task,description=sample_desc1,total=len(Frames))
                sample_bar = (progress,sample_task)

                # Generate rollout data
                Trajectories,Images = generate_rollouts(
                    simulator,controller,tXUd,bframe,
                    Frames,Perturbations,
                    dt_ro,method["tol_select"],
                    idx_bt,sample_bar)

                # Update the observations progress bar
                progress.update(sample_task,description=sample_desc2)
                progress.refresh()

                # Save the rollout data
                save_rollouts(cohort_name,course_name,
                            Trajectories,Images,
                            idx_bt,use_compress)

                # Update the data count
                Ndata += sum([trajectory["Ndata"] for trajectory in Trajectories])

                # Update the progress bar
                course_desc = ru.get_data_description(course_name,Ndata,subunits=subunits)
                progress.update(course_task,description=course_desc,advance=1)
                progress.refresh()

            # Ensure progress catches last update
            progress.refresh()

def generate_rollouts(
        simulator:Simulator,controller:VehicleRateMPC,tXUd:np.ndarray,bframe:dict[str,np.ndarray,str|int|float],
        Frames:list[dict[str,np.ndarray,str|int|float]],Perturbations:list[dict[str,float|np.ndarray]],
        dt_ro:float,tol_select:float,
        idx_set:int,progress_bar:tuple[ru.Progress,int]=None,
        debug:bool=False
        ) -> tuple[list[dict[str,np.ndarray]],list[torch.Tensor]]:
    """
    Generates rollout data for the quadcopter given a list of drones and initial states (perturbations).
    The rollout comprises trajectory data and image data. The trajectory data is generated by running
    the MPC controller on the quadcopter for a fixed number of steps. The trajectory data consists of
    time, states [p,v,q], body rate inputs [fn,w], objective state, data count, solver timings, advisor
    data, rollout id, and course name. The image data is generated by rendering the quadcopter at each
    state in the trajectory data. The image data consists of the image data and the data count.

    Args:
        simulator:      Simulator object.
        controller:     Controller object.
        tXUd:           Trajectory rollout.
        bframe:         Base frame for the quadcopter.
        Frames:         List of drone frame configurations.
        Perturbations:  List of perturbed initial states.
        dt_ro:          Rollout duration.
        tol_select:     Error tolerance.
        idx_set:        Index of the rollout set.
        progress_bar:   Progress bar (if available).
        debug:          Debug flag.

    Returns:
        Trajectories:   List of trajectory rollouts.
        Images:         List of image rollouts.
    """
    
    # Get console
    console = ru.get_console()
    
    # Initialize rollout variables
    Trajectories,Images = [],[]

    # Set the tolerance if undefined
    tol_select = tol_select or np.inf

    # Rollout the trajectories
    Ndata = len(Perturbations)
    for idx,(frame,perturbation) in enumerate(zip(Frames,Perturbations)):
        # Unpack rollout variables
        t0,x0 = perturbation["t0"],perturbation["x0"]
        tf = t0 + dt_ro

        # Update the simulation variables
        simulator.update_frame(frame)
        controller.update_frame(frame)    

        # Simulate the flight
        Tro,Xro,Uro,Wro,Rgb,Dpt,Tsol = simulator.simulate(controller,t0,tf,x0)

        # Check if the rollout data is useful
        err = np.min(np.linalg.norm(tXUd[:,1:4]-Xro[-1,0:3],axis=1))
        if err < tol_select:
            # Compute Additional Variables
            prms = svu.compute_prms(frame)
            Wrs = svu.compute_Wrs(Xro,Uro,Wro,frame,bframe)
            FOro = svu.compute_FOro(Tro,Xro,Uro,Wro,frame)

            # Package the rollout data
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,"Wro":Wro,
                "params":prms,"Wrs":Wrs,"FOro":FOro,
                "tXUd":tXUd,"Ndata":Uro.shape[0],"Tsol":Tsol,
                "rollout_id":str(idx_set+1).zfill(3)+str(idx).zfill(3),
                "frame":frame}

            images = {
                "rgb":Rgb,"depth":Dpt,
                "rollout_id":str(idx_set+1).zfill(3)+str(idx).zfill(3)
            }

            # Store rollout data
            Trajectories.append(trajectory)
            Images.append(images)

            # Update the progress bar
            if progress_bar is not None:
                progress,sample_task = progress_bar
                progress.update(sample_task,advance=1)
                progress.refresh()
        else:
            if debug:
                console.print(
                    f"[bold red]Rollout failed to meet tolerance. Skipping...[/]\n"
                    f"Euclidean Distance: {err:.3f} > {tol_select:.3f}")
            
            Ndata -= 1
            if progress_bar is not None:
                progress,sample_task = progress_bar
                progress.update(sample_task,total=Ndata)
                progress.refresh()

    return Trajectories,Images

def save_rollouts(cohort_name:str,course_name:str,
                  Trajectories:list[tuple[np.ndarray,np.ndarray,np.ndarray]],
                  Images:list[torch.Tensor],
                  stack_id:str|int,
                  use_compress:bool=False) -> None:
    """
    Saves the rollout data to a .pt file in folders corresponding to coursename within the cohort 
    directory. The rollout data is stored as a list of rollout dictionaries of size stack_size for
    ease of comprehension and loading (at a cost of storage space).
    
    Args:
        cohort_path:    Cohort path.
        course_name:    Name of the course.
        method_name:    Name of the method used to generate the data.
        Trajectories:   Rollout data.
        Images:         Image data.
        stack_id:       Stack id.
        use_compress:   Compress the image data.

    Returns:
        None:           (rollout data saved to cohort directory)
    """
    # Create rollout course directory (if it does not exist)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    cohort_path = os.path.join(workspace_path,"cohorts",cohort_name)
    
    dset_path = os.path.join(cohort_path,"rollout_data",course_name)
    traj_course_path = os.path.join(dset_path,"trajectories")
    imgs_course_path = os.path.join(dset_path,"images")

    if not os.path.exists(traj_course_path):
        os.makedirs(traj_course_path)
    
    if not os.path.exists(imgs_course_path):
        os.makedirs(imgs_course_path)

    # Save the stacks
    dset_name = str(stack_id+1).zfill(3) if type(stack_id) == int else str(stack_id)
    traj_path = os.path.join(traj_course_path,"trajectories"+dset_name+".pt")
    imgs_path = os.path.join(imgs_course_path,"images"+dset_name+".pt")

    assert all(isinstance(x["rgb"], np.ndarray) for x in Images)
    assert all(isinstance(x["depth"], np.ndarray) for x in Images)
    if use_compress:
        Images = dch.compress_data(Images)
    # Protocol 4 represents binary image buffers as bytes rather than text.
    # This avoids UTF-8 decoding failures for pixel values above 0x7f.
    torch.save(Trajectories, traj_path, pickle_protocol=4)
    torch.save(Images, imgs_path, pickle_protocol=4)

