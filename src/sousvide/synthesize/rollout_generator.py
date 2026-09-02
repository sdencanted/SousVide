import errno
import numpy as np
import os
import re
import shutil
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
from sousvide.synthesize.event_surfaces import (
    resolve_event_surface_options,validate_event_modalities,
    validate_event_representations,
)
from sousvide.synthesize.event_cloud import (
    event_cloud_metadata,resolve_event_cloud_options)
from sousvide.synthesize.image_modality import (
    is_event_cloud_modality,is_voxel_grid_modality,modality_storage)
from sousvide.synthesize.parallel_event_generator import (
    EventFrameBuffer,
    process_buffered_rollout,
    process_event_stream_rollout,
)
import sousvide.visualize.rich_utilities as ru
import sousvide.utilities.sousvide_utilities as svu

from figs.simulator import Simulator
from figs.dynamics.external_forces import ExternalForces
from figs.tsplines.min_time_snap import MinTimeSnap
from figs.control.vehicle_rate_mpc import VehicleRateMPC


def _replace_staged_file(staged_path:str,final_path:str) -> None:
    """Replace final_path, including when the paths are on different filesystems."""
    try:
        os.replace(staged_path,final_path)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise

    destination_dir = os.path.dirname(final_path)
    descriptor,temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(final_path)}.",
        suffix=".tmp",
        dir=destination_dir,
    )
    os.close(descriptor)
    try:
        shutil.copy2(staged_path,temporary_path)
        os.replace(temporary_path,final_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    os.unlink(staged_path)


def _resolve_requested_event_modalities(
        generate_events:bool,event_modalities,event_surface_options):
    if event_modalities is None:
        if not generate_events:
            return (),{}
        event_modalities = ("kronecker_delta",)
    elif not isinstance(event_modalities,str):
        event_modalities = tuple(event_modalities)
        if not event_modalities:
            if event_surface_options:
                raise ValueError(
                    "event_surface_options requires at least one event modality.")
            return (),{}
    modalities = validate_event_modalities(event_modalities)
    return modalities,resolve_event_surface_options(
        modalities,event_surface_options)

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
        Nro_ds:         Number of rollouts per saved dataset stack.
        use_compress:   Compress the image data.
        generate_events: Generate and retain raw v2e H5 event streams.
        event_workers:  Number of CPU processes used for v2e (default 1).

    Returns:
        None:           (flight data saved to cohort directory)
    """

    # Initialize the progress variables
    if event_workers < 1:
        raise ValueError("event_workers must be at least 1.")
    event_enabled = generate_events

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
        simulator = EventSimulator(gsplat,method) if event_enabled else Simulator(gsplat,method)

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

            event_staging_parent = None
            if event_enabled:
                workspace_path = os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.dirname(__file__))))
                event_staging_parent = os.path.join(
                    workspace_path,"cohorts",cohort_name,"rollout_data",
                    course_name)
                os.makedirs(event_staging_parent,exist_ok=True)

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
                    prefix=f".sousvide-events-{idx_bt + 1:03d}-",
                    dir=event_staging_parent,
                ) as event_staging_dir:
                    rollout_data = generate_rollouts(
                        simulator,controller,tXUd,bframe,
                        Frames,Perturbations,
                        dt_ro,method["tol_select"],
                        idx_bt,sample_bar,
                        generate_events=event_enabled,
                        event_modalities=(),
                        event_staging_dir=event_staging_dir,
                        event_workers=event_workers)

                    if event_enabled:
                        Trajectories,Images,_,EventPaths = rollout_data
                    else:
                        Trajectories,Images = rollout_data
                        EventPaths = None

                    progress.update(sample_task,description=sample_desc2)
                    progress.refresh()

                    save_rollouts(cohort_name,course_name,
                                Trajectories,Images,
                                idx_bt,use_compress,
                                EventPaths=EventPaths)

                # Update the data count
                Ndata += sum([trajectory["Ndata"] for trajectory in Trajectories])

                # Update the progress bar
                course_desc = ru.get_data_description(course_name,Ndata,subunits=subunits)
                progress.update(course_task,description=course_desc,advance=1)
                progress.refresh()

            # Ensure progress catches last update
            progress.refresh()


def _save_event_representation_stack(
        dset_path:str,dset_name:str,Trajectories:list[dict],Images:list[dict],
        EventImagesByModality:dict[str,list[dict]],
        use_compress:bool=False) -> None:
    """Validate and save one stack of derived event representations."""
    if not EventImagesByModality:
        raise ValueError("EventImagesByModality must not be empty.")
    lengths = set()
    for modality,event_images in EventImagesByModality.items():
        validate_aligned_rollouts(
            Trajectories,Images,event_images,image_modality=modality)
        lengths.add(len(event_images))
    if len(lengths) != 1:
        raise ValueError("Event modality rollout counts do not match.")

    for modality,event_images in EventImagesByModality.items():
        folder,prefix = modality_storage(modality)
        output_folder = os.path.join(dset_path,folder)
        os.makedirs(output_folder,exist_ok=True)
        if (use_compress and not is_voxel_grid_modality(modality)
                and not is_event_cloud_modality(modality)):
            dch.compress_data(event_images,key=modality)
        if (is_voxel_grid_modality(modality)
                or is_event_cloud_modality(modality)):
            # Tensor storage keeps large memory-mapped stacks disk-backed
            # while torch.save streams them to the final artifact.
            serialized_event_images = []
            for event_image in event_images:
                serialized_event_image = dict(event_image)
                serialized_event_image[modality] = torch.from_numpy(
                    event_image[modality])
                serialized_event_images.append(serialized_event_image)
        else:
            serialized_event_images = event_images
        torch.save(
            serialized_event_images,
            os.path.join(output_folder,prefix+dset_name+".pt"),
            pickle_protocol=4)


def generate_event_representations(
        cohort_name:str,course_names:list[str],event_modalities,
        event_workers:int=1,
        event_surface_options:dict[str,dict]|None=None,
        event_cloud_options:dict|None=None,
        use_compress:bool=False) -> None:
    """Derive aligned event-image stacks from saved rollout H5 streams.

    Run this after :func:`generate_rollout_data` with
    ``generate_events=True``. The saved RGB and trajectory stacks provide the
    image dimensions and control-window boundaries; the raw H5 streams provide
    the events, so flight simulation and v2e are not rerun.
    """
    if event_workers < 1:
        raise ValueError("event_workers must be at least 1.")
    modalities = validate_event_representations(event_modalities)
    raster_modalities = tuple(
        modality for modality in modalities if modality != "event_cloud")
    resolved_options = (
        resolve_event_surface_options(
            raster_modalities,event_surface_options)
        if raster_modalities else {})
    if not raster_modalities and event_surface_options:
        raise ValueError(
            "event_surface_options requires at least one raster event modality.")
    resolved_cloud_options = (
        resolve_event_cloud_options(event_cloud_options)
        if "event_cloud" in modalities else None)
    if "event_cloud" not in modalities and event_cloud_options:
        raise ValueError(
            "event_cloud_options requires the event_cloud modality.")
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    event_pool = None
    if event_workers > 1:
        event_pool = ProcessPoolExecutor(
            max_workers=event_workers,mp_context=get_context("spawn"))
    try:
        for course_name in course_names:
            dset_path = os.path.join(
                workspace_path,"cohorts",cohort_name,"rollout_data",
                course_name)
            images_path = os.path.join(dset_path,"images")
            trajectories_path = os.path.join(dset_path,"trajectories")
            events_path = os.path.join(dset_path,"events")
            if not os.path.isdir(images_path):
                raise FileNotFoundError(
                    f"Rollout image directory does not exist: {images_path}")
            if not os.path.isdir(events_path):
                raise FileNotFoundError(
                    "Raw event streams do not exist; run generate_rollout_data "
                    f"with generate_events=True first: {events_path}")

            stack_pattern = re.compile(r"^images(.+)\.pt$")
            stacks = []
            for filename in os.listdir(images_path):
                match = stack_pattern.fullmatch(filename)
                if match:
                    stacks.append((match.group(1),filename))
            for dset_name,images_filename in sorted(stacks):
                trajectory_path = os.path.join(
                    trajectories_path,"trajectories"+dset_name+".pt")
                if not os.path.isfile(trajectory_path):
                    raise FileNotFoundError(
                        f"Matching trajectory stack does not exist: {trajectory_path}")
                Trajectories = torch.load(
                    trajectory_path,weights_only=False)
                Images = torch.load(
                    os.path.join(images_path,images_filename),
                    weights_only=False)
                if len(Trajectories) != len(Images):
                    raise ValueError(
                        f"Trajectory and RGB rollout counts do not match in stack {dset_name}.")
                for image_data in Images:
                    dch.decompress_data(image_data,key="rgb")

                with tempfile.TemporaryDirectory(
                        prefix=f".sousvide-representations-{dset_name}-",
                        dir=dset_path) as staging_path:
                    jobs = []
                    for trajectory,image_data in zip(Trajectories,Images):
                        rollout_id = trajectory["rollout_id"]
                        if image_data["rollout_id"] != rollout_id:
                            raise ValueError(
                                f"Trajectory and RGB rollout IDs do not match: {rollout_id}")
                        rgb = image_data["rgb"]
                        if not isinstance(rgb,np.ndarray) or rgb.ndim != 4:
                            raise ValueError(
                                f"RGB rollout {rollout_id} must have shape (N,H,W,C).")
                        times = np.asarray(trajectory["Tro"],dtype=np.float64)
                        if len(times) != len(rgb)+1:
                            raise ValueError(
                                f"Trajectory and RGB frame counts do not match: {rollout_id}")
                        window_end_times = tuple(times[1:]-times[0])
                        h5_path = os.path.join(events_path,rollout_id+".h5")
                        if not os.path.isfile(h5_path):
                            raise FileNotFoundError(
                                f"Raw event stream does not exist: {h5_path}")
                        output_paths = {
                            modality:os.path.join(
                                staging_path,rollout_id+f".{modality}.npy")
                            for modality in modalities}
                        args = (
                            h5_path,window_end_times,rgb.shape[1],rgb.shape[2],
                            modalities,resolved_options,output_paths,
                            resolved_cloud_options,rollout_id)
                        if event_pool is None:
                            jobs.append(process_event_stream_rollout(*args))
                        else:
                            jobs.append(event_pool.submit(
                                process_event_stream_rollout,*args))

                    completed_outputs = (
                        jobs if event_pool is None
                        else [future.result() for future in jobs])
                    EventImages = {modality:[] for modality in modalities}
                    for trajectory,completed_output in zip(
                            Trajectories,completed_outputs):
                        rollout_id = trajectory["rollout_id"]
                        output_paths,raw_event_counts = completed_output
                        for modality,path in output_paths.items():
                            representation_config = (
                                event_cloud_metadata(resolved_cloud_options)
                                if modality == "event_cloud"
                                else resolved_options[modality])
                            EventImages[modality].append({
                                modality:np.load(
                                    path,mmap_mode="r+",allow_pickle=False),
                                "rollout_id":rollout_id,
                                "event_surface_config":(
                                    {} if modality == "event_cloud"
                                    else representation_config),
                                "event_cloud_config":(
                                    representation_config
                                    if modality == "event_cloud" else {}),
                                "raw_event_counts":(
                                    raw_event_counts
                                    if modality == "event_cloud" else None),
                            })
                    _save_event_representation_stack(
                        dset_path,dset_name,Trajectories,Images,EventImages,
                        use_compress=use_compress)
    finally:
        if event_pool is not None:
            event_pool.shutdown()


def generate_rollouts(
        simulator:Simulator,controller:VehicleRateMPC,tXUd:np.ndarray,bframe:dict[str,np.ndarray,str|int|float],
        Frames:list[dict[str,np.ndarray,str|int|float]],Perturbations:list[dict[str,float|np.ndarray]],
        dt_ro:float,tol_select:float,
        idx_set:int,progress_bar:tuple[ru.Progress,int]=None,
        debug:bool=False,generate_events:bool=False,
        event_staging_dir:str|None=None,
        event_workers:int=1,
        event_modalities=None,
        event_surface_options:dict[str,dict]|None=None,
        ):
    """Generate rollouts, optionally processing accepted event streams in parallel."""
    if event_workers < 1:
        raise ValueError("event_workers must be at least 1.")

    legacy_event_return = generate_events and event_modalities is None
    active_event_modalities,resolved_surface_options = (
        _resolve_requested_event_modalities(
            generate_events,event_modalities,event_surface_options))
    event_enabled = generate_events or bool(active_event_modalities)
    kwargs = dict(
        simulator=simulator,controller=controller,tXUd=tXUd,bframe=bframe,
        Frames=Frames,Perturbations=Perturbations,dt_ro=dt_ro,
        tol_select=tol_select,idx_set=idx_set,progress_bar=progress_bar,
        debug=debug,generate_events=event_enabled,
        event_modalities=active_event_modalities,
        event_surface_options=resolved_surface_options,
        event_staging_dir=event_staging_dir,event_workers=event_workers,
    )
    if event_enabled and event_workers > 1:
        # Spawn avoids inheriting the already initialized CUDA/GSplat state.
        with ProcessPoolExecutor(
            max_workers=event_workers,mp_context=get_context("spawn")
        ) as event_pool:
            result = _generate_rollouts_impl(**kwargs,event_pool=event_pool)
    else:
        result = _generate_rollouts_impl(**kwargs,event_pool=None)
    if legacy_event_return:
        trajectories,images,event_images,event_paths = result
        return trajectories,images,event_images["kronecker_delta"],event_paths
    return result


def _generate_rollouts_impl(
        simulator:Simulator,controller:VehicleRateMPC,tXUd:np.ndarray,bframe:dict[str,np.ndarray,str|int|float],
        Frames:list[dict[str,np.ndarray,str|int|float]],Perturbations:list[dict[str,float|np.ndarray]],
        dt_ro:float,tol_select:float,
        idx_set:int,progress_bar:tuple[ru.Progress,int]=None,
        debug:bool=False,generate_events:bool=False,
        event_staging_dir:str|None=None,
        event_workers:int=1,
        event_pool:ProcessPoolExecutor|None=None,
        event_modalities=(),
        event_surface_options:dict[str,dict]|None=None,
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
    EventImages = {modality:[] for modality in event_modalities}
    EventPaths = []

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
            output_paths = {
                modality:os.path.join(
                    event_staging_dir,rollout_id+f".{modality}.npy")
                for modality in event_modalities}
            if parallel_events:
                event_buffer = EventFrameBuffer()
                event_callback = event_buffer.process_frame
            else:
                recorder = V2ERolloutRecorder(
                    staged_h5,expected_windows,
                    retain_images=bool(event_modalities),
                    event_modalities=event_modalities,
                    event_surface_options=event_surface_options,
                    image_output_paths=output_paths)
                event_callback = recorder.process_frame
            try:
                Tro,Xro,Uro,Wro,Rgb,Dpt,Tsol = simulator.simulate_with_events(
                    controller,t0,tf,x0,event_callback,warmup_steps)
                if recorder is not None:
                    generated_event_images = recorder.close_all()
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
                timestamps,close_windows = event_buffer.save(frame_path)
                future = event_pool.submit(
                    process_buffered_rollout,
                    frame_path,timestamps,close_windows,staged_h5,
                    next(iter(output_paths.values()),None),expected_windows,
                    event_modalities,event_surface_options,output_paths,
                    bool(event_modalities))
                event_jobs.append((
                    rollout_id,future,frame_path,output_paths,staged_h5))
            elif generate_events:
                if event_modalities:
                    for modality,event_stack in generated_event_images.items():
                        EventImages[modality].append({
                            modality:event_stack,"rollout_id":rollout_id,
                            "event_surface_config":event_surface_options[modality]})
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
        for rollout_id,future,frame_path,output_paths,staged_h5 in event_jobs:
            try:
                completed_outputs,completed_h5 = future.result()
                completed_jobs.append((
                    rollout_id,
                    {
                        modality:np.load(
                            path,mmap_mode="r+",allow_pickle=False)
                        for modality,path in completed_outputs.items()
                    },
                    completed_h5))
            except Exception as error:
                if worker_error is None:
                    worker_error = error

        for _,_,frame_path,_,_ in event_jobs:
            if os.path.isfile(frame_path):
                os.unlink(frame_path)

        if worker_error is not None:
            for *_,output_paths,staged_h5 in event_jobs:
                for output_path in output_paths.values():
                    if os.path.isfile(output_path):
                        os.unlink(output_path)
                if os.path.isfile(staged_h5):
                    os.unlink(staged_h5)
            raise worker_error

        for rollout_id,generated_event_images,staged_h5 in completed_jobs:
            for modality,event_stack in generated_event_images.items():
                EventImages[modality].append({
                    modality:event_stack,"rollout_id":rollout_id,
                    "event_surface_config":event_surface_options[modality]})
            EventPaths.append(staged_h5)

    if generate_events:
        return Trajectories,Images,EventImages,EventPaths
    return Trajectories,Images

def save_rollouts(cohort_name:str,course_name:str,
                  Trajectories:list[tuple[np.ndarray,np.ndarray,np.ndarray]],
                  Images:list[torch.Tensor],
                  stack_id:str|int,
                  use_compress:bool=False,
                  KroneckerImages:list[dict]|None=None,
                  EventPaths:list[str]|None=None,
                  EventImagesByModality:dict[str,list[dict]]|None=None) -> None:
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
    events_course_path = os.path.join(dset_path,"events")

    if EventImagesByModality is not None and KroneckerImages is not None:
        raise ValueError(
            "Pass either KroneckerImages or EventImagesByModality, not both.")
    if EventImagesByModality is None and KroneckerImages is not None:
        EventImagesByModality = {"kronecker_delta":KroneckerImages}

    if not os.path.exists(traj_course_path):
        os.makedirs(traj_course_path)
    
    if not os.path.exists(imgs_course_path):
        os.makedirs(imgs_course_path)
    if EventImagesByModality is not None:
        for modality in EventImagesByModality:
            folder,_ = modality_storage(modality)
            os.makedirs(os.path.join(dset_path,folder),exist_ok=True)
    if EventPaths is not None:
        os.makedirs(events_course_path,exist_ok=True)

    # Save the stacks
    dset_name = str(stack_id+1).zfill(3) if type(stack_id) == int else str(stack_id)
    traj_path = os.path.join(traj_course_path,"trajectories"+dset_name+".pt")
    imgs_path = os.path.join(imgs_course_path,"images"+dset_name+".pt")

    assert all(isinstance(x["rgb"], np.ndarray) for x in Images)
    assert all(isinstance(x["depth"], np.ndarray) for x in Images)
    if EventImagesByModality is not None:
        if not EventImagesByModality:
            raise ValueError("EventImagesByModality must not be empty.")
        lengths = set()
        for modality,event_images in EventImagesByModality.items():
            validate_aligned_rollouts(
                Trajectories,Images,event_images,image_modality=modality)
            lengths.add(len(event_images))
        if len(lengths) != 1:
            raise ValueError("Event modality rollout counts do not match.")
        event_count = lengths.pop()
        if EventPaths is not None and len(EventPaths) != event_count:
            raise ValueError(
                "Event image and raw event rollout counts do not match.")
    if EventPaths is not None and len(EventPaths) != len(Trajectories):
        raise ValueError(
            "Each accepted rollout must have one staged H5 event stream.")
    if EventImagesByModality is not None:
        _save_event_representation_stack(
            dset_path,dset_name,Trajectories,Images,
            EventImagesByModality,use_compress=use_compress)
    if use_compress:
        Images = dch.compress_data(Images,key="rgb")
    # Protocol 4 represents binary image buffers as bytes rather than text.
    # This avoids UTF-8 decoding failures for pixel values above 0x7f.
    torch.save(Trajectories, traj_path, pickle_protocol=4)
    torch.save(Images, imgs_path, pickle_protocol=4)

    if EventPaths is not None:
        accepted_ids = {rollout["rollout_id"] for rollout in Trajectories}
        stack_pattern = re.compile(rf"^{re.escape(dset_name)}\d{{3}}\.h5$")
        for filename in os.listdir(events_course_path):
            if stack_pattern.fullmatch(filename) and filename[:-3] not in accepted_ids:
                os.unlink(os.path.join(events_course_path,filename))
        for staged_path,rollout in zip(EventPaths,Trajectories):
            final_path = os.path.join(events_course_path,rollout["rollout_id"]+".h5")
            _replace_staged_file(staged_path,final_path)
