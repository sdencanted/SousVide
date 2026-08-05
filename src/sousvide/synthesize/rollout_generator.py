import numpy as np
import os
import re
import tempfile
import torch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th

import sousvide.synthesize.synthesize_helper as sh
import sousvide.synthesize.data_compress_helper as dch
from sousvide.synthesize.alignment import validate_aligned_rollouts
from sousvide.synthesize.event_generator import V2ERolloutRecorder
from sousvide.synthesize.event_simulator import EventSimulator
from sousvide.synthesize.parallel_event_generator import (
    EventFrameBuffer,
    process_buffered_rollout,
)
import sousvide.visualize.rich_utilities as ru
import sousvide.utilities.sousvide_utilities as svu

from figs.simulator import Simulator
from figs.dynamics.external_forces import ExternalForces
from figs.tsplines.min_time_snap import MinTimeSnap
from figs.control.vehicle_rate_mpc import VehicleRateMPC

def generate_rollout_data(cohort_name:str,course_names:list[str],
                          gsplat_name:str,method_name:str,
                          expert_name:str="Viper",bframe_name:str="carl",
                          Nro_ds:int=50,use_compress:bool=False,
                          generate_events:bool=False,
                          event_workers:int=1) -> None:
    
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
        generate_events: Generate v2e H5 files and aligned Kronecker images.
        event_workers:  Number of CPU processes used for v2e (default 1).

    Returns:
        None:           (flight data saved to cohort directory)
    """

    # Initialize the progress variables
    if event_workers < 1:
        raise ValueError("event_workers must be at least 1.")

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
        simulator = EventSimulator(gsplat,method) if generate_events else Simulator(gsplat,method)

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

                # Stage H5 files until all aligned stack files save successfully.
                with tempfile.TemporaryDirectory(
                    prefix=f"sousvide-events-{course_name}-{idx_bt + 1:03d}-"
                ) as event_staging_dir:
                    rollout_data = generate_rollouts(
                        simulator,controller,tXUd,bframe,
                        Frames,Perturbations,
                        dt_ro,method["tol_select"],
                        idx_bt,sample_bar,
                        generate_events=generate_events,
                        event_staging_dir=event_staging_dir,
                        event_workers=event_workers)

                    if generate_events:
                        Trajectories,Images,KroneckerImages,EventPaths = rollout_data
                    else:
                        Trajectories,Images = rollout_data
                        KroneckerImages,EventPaths = None,None

                    progress.update(sample_task,description=sample_desc2)
                    progress.refresh()

                    save_rollouts(cohort_name,course_name,
                                Trajectories,Images,
                                idx_bt,use_compress,
                                KroneckerImages,EventPaths)

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
        debug:bool=False,generate_events:bool=False,
        event_staging_dir:str|None=None,
        event_workers:int=1
        ):
    """Generate rollouts, optionally processing accepted event streams in parallel."""
    if event_workers < 1:
        raise ValueError("event_workers must be at least 1.")

    kwargs = dict(
        simulator=simulator,controller=controller,tXUd=tXUd,bframe=bframe,
        Frames=Frames,Perturbations=Perturbations,dt_ro=dt_ro,
        tol_select=tol_select,idx_set=idx_set,progress_bar=progress_bar,
        debug=debug,generate_events=generate_events,
        event_staging_dir=event_staging_dir,event_workers=event_workers,
    )
    if generate_events and event_workers > 1:
        # Spawn avoids inheriting the already initialized CUDA/GSplat state.
        with ProcessPoolExecutor(
            max_workers=event_workers,mp_context=get_context("spawn")
        ) as event_pool:
            return _generate_rollouts_impl(**kwargs,event_pool=event_pool)
    return _generate_rollouts_impl(**kwargs,event_pool=None)


def _generate_rollouts_impl(
        simulator:Simulator,controller:VehicleRateMPC,tXUd:np.ndarray,bframe:dict[str,np.ndarray,str|int|float],
        Frames:list[dict[str,np.ndarray,str|int|float]],Perturbations:list[dict[str,float|np.ndarray]],
        dt_ro:float,tol_select:float,
        idx_set:int,progress_bar:tuple[ru.Progress,int]=None,
        debug:bool=False,generate_events:bool=False,
        event_staging_dir:str|None=None,
        event_workers:int=1,
        event_pool:ProcessPoolExecutor|None=None,
        ):
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
    KroneckerImages,EventPaths = [],[]

    if generate_events and event_staging_dir is None:
        raise ValueError("event_staging_dir is required when generate_events=True.")
    if event_workers < 1:
        raise ValueError("event_workers must be at least 1.")

    parallel_events = event_pool is not None
    event_jobs = []

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

        rollout_id = str(idx_set+1).zfill(3)+str(idx).zfill(3)

        # Event rollouts begin one RGB interval early and render every
        # simulation step through the final saved RGB frame.
        recorder = None
        event_buffer = None
        if generate_events:
            hz_sim = simulator.conFiG["rollout"]["frequency"]
            warmup_steps = int(hz_sim/controller.hz)
            expected_windows = int(np.round(dt_ro*controller.hz))
            staged_h5 = os.path.join(event_staging_dir,rollout_id+".h5")
            if parallel_events:
                event_buffer = EventFrameBuffer()
                event_callback = event_buffer.process_frame
            else:
                recorder = V2ERolloutRecorder(staged_h5,expected_windows)
                event_callback = recorder.process_frame
            try:
                Tro,Xro,Uro,Wro,Rgb,Dpt,Tsol = simulator.simulate_with_events(
                    controller,t0,tf,x0,event_callback,warmup_steps)
                if recorder is not None:
                    Kronecker = recorder.close()
            except Exception:
                if recorder is not None:
                    recorder.abort()
                raise
        else:
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
                "rollout_id":rollout_id,
                "frame":frame}

            images = {
                "rgb":Rgb,"depth":Dpt,
                "rollout_id":rollout_id
            }

            # Store rollout data
            Trajectories.append(trajectory)
            Images.append(images)
            if generate_events and parallel_events:
                frame_path = os.path.join(
                    event_staging_dir,rollout_id+".frames.npy")
                kronecker_path = os.path.join(
                    event_staging_dir,rollout_id+".kronecker.npy")
                timestamps,close_windows = event_buffer.save(frame_path)
                future = event_pool.submit(
                    process_buffered_rollout,
                    frame_path,timestamps,close_windows,staged_h5,
                    kronecker_path,expected_windows)
                event_jobs.append((
                    rollout_id,future,frame_path,kronecker_path,staged_h5))
            elif generate_events:
                KroneckerImages.append({
                    "kronecker_delta":Kronecker,"rollout_id":rollout_id})
                EventPaths.append(staged_h5)

            # Update the progress bar
            if progress_bar is not None:
                progress,sample_task = progress_bar
                progress.update(sample_task,advance=1)
                progress.refresh()
        else:
            if recorder is not None:
                recorder.abort()
            if debug:
                console.print(
                    f"[bold red]Rollout failed to meet tolerance. Skipping...[/]\n"
                    f"Euclidean Distance: {err:.3f} > {tol_select:.3f}")
            
            Ndata -= 1
            if progress_bar is not None:
                progress,sample_task = progress_bar
                progress.update(sample_task,total=Ndata)
                progress.refresh()

    if parallel_events:
        worker_error = None
        completed_jobs = []
        for rollout_id,future,frame_path,kronecker_path,staged_h5 in event_jobs:
            try:
                completed_kronecker,completed_h5 = future.result()
                completed_jobs.append((
                    rollout_id,
                    np.load(completed_kronecker,allow_pickle=False),
                    completed_h5))
            except Exception as error:
                if worker_error is None:
                    worker_error = error

        for _,_,frame_path,kronecker_path,_ in event_jobs:
            if os.path.isfile(frame_path):
                os.unlink(frame_path)
            if os.path.isfile(kronecker_path):
                os.unlink(kronecker_path)

        if worker_error is not None:
            for *_,staged_h5 in event_jobs:
                if os.path.isfile(staged_h5):
                    os.unlink(staged_h5)
            raise worker_error

        for rollout_id,kronecker,staged_h5 in completed_jobs:
            KroneckerImages.append({
                "kronecker_delta":kronecker,"rollout_id":rollout_id})
            EventPaths.append(staged_h5)

    if generate_events:
        return Trajectories,Images,KroneckerImages,EventPaths
    return Trajectories,Images

def save_rollouts(cohort_name:str,course_name:str,
                  Trajectories:list[tuple[np.ndarray,np.ndarray,np.ndarray]],
                  Images:list[torch.Tensor],
                  stack_id:str|int,
                  use_compress:bool=False,
                  KroneckerImages:list[dict]|None=None,
                  EventPaths:list[str]|None=None) -> None:
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
    kron_course_path = os.path.join(dset_path,"kronecker")
    events_course_path = os.path.join(dset_path,"events")

    if not os.path.exists(traj_course_path):
        os.makedirs(traj_course_path)
    
    if not os.path.exists(imgs_course_path):
        os.makedirs(imgs_course_path)
    if KroneckerImages is not None:
        os.makedirs(kron_course_path,exist_ok=True)
        os.makedirs(events_course_path,exist_ok=True)

    # Save the stacks
    dset_name = str(stack_id+1).zfill(3) if type(stack_id) == int else str(stack_id)
    traj_path = os.path.join(traj_course_path,"trajectories"+dset_name+".pt")
    imgs_path = os.path.join(imgs_course_path,"images"+dset_name+".pt")
    kron_path = os.path.join(kron_course_path,"kronecker"+dset_name+".pt")

    assert all(isinstance(x["rgb"], np.ndarray) for x in Images)
    assert all(isinstance(x["depth"], np.ndarray) for x in Images)
    if KroneckerImages is not None:
        validate_aligned_rollouts(Trajectories,Images,KroneckerImages)
        if EventPaths is None or len(EventPaths) != len(KroneckerImages):
            raise ValueError("Each accepted Kronecker rollout must have one staged H5 file.")
    if use_compress:
        Images = dch.compress_data(Images,key="rgb")
        if KroneckerImages is not None:
            KroneckerImages = dch.compress_data(KroneckerImages,key="kronecker_delta")
    # Protocol 4 represents binary image buffers as bytes rather than text.
    # This avoids UTF-8 decoding failures for pixel values above 0x7f.
    torch.save(Trajectories, traj_path, pickle_protocol=4)
    torch.save(Images, imgs_path, pickle_protocol=4)
    if KroneckerImages is not None:
        torch.save(KroneckerImages,kron_path,pickle_protocol=4)

        accepted_ids = {rollout["rollout_id"] for rollout in KroneckerImages}
        stack_pattern = re.compile(rf"^{re.escape(dset_name)}\d{{3}}\.h5$")
        for filename in os.listdir(events_course_path):
            if stack_pattern.fullmatch(filename) and filename[:-3] not in accepted_ids:
                os.unlink(os.path.join(events_course_path,filename))
        for staged_path,rollout in zip(EventPaths,KroneckerImages):
            final_path = os.path.join(events_course_path,rollout["rollout_id"]+".h5")
            os.replace(staged_path,final_path)

