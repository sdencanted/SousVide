import torch
import os
import numpy as np
import figs.utilities.config_helper as ch
import figs.utilities.transform_helper as th
import figs.visualize.generate_videos as gv
from tqdm.auto import tqdm
import sousvide.synthesize.synthesize_helper as sh
import sousvide.utilities.sousvide_utilities as svu
import sousvide.visualize.record_flight as rf
import sousvide.visualize.rich_utilities as ru
import sousvide.flight.flight_helper as fh

from typing import List,Literal,Union
from figs.tsplines.min_time_snap import MinTimeSnap
from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
from figs.dynamics.external_forces import ExternalForces
from sousvide.control.pilot import Pilot
from sousvide.synthesize.event_generator import (
    OnlineEventCloudGenerator,OnlineEventImageGenerator)
from sousvide.synthesize.event_simulator import EventSimulator
from sousvide.synthesize.image_modality import (
    VisualModality,image_modality_channels,is_event_cloud_modality,
    is_event_modality,is_grayscale_modality,is_voxel_grid_modality,
    rgb_to_grayscale,validate_visual_modality)
from sousvide.synthesize.event_surfaces import resolve_event_surface_options
from sousvide.synthesize.event_cloud import resolve_event_cloud_options


def _event_cloud_debug_image(
        event_cloud:np.ndarray,sensor_shape:tuple[int,int],
        max_side:int=224) -> np.ndarray:
    """Rasterize a normalized cloud without changing the sensor aspect ratio."""
    event_cloud = np.asarray(event_cloud)
    if event_cloud.ndim != 2 or event_cloud.shape[1] != 4:
        raise ValueError(
            f"Expected an event cloud with shape (N,4), got {event_cloud.shape}.")
    sensor_height,sensor_width = sensor_shape
    if sensor_height <= 0 or sensor_width <= 0 or max_side <= 0:
        raise ValueError("Event-cloud debug dimensions must be positive.")
    scale = max_side/max(sensor_height,sensor_width)
    display_height = max(1,int(round(sensor_height*scale)))
    display_width = max(1,int(round(sensor_width*scale)))
    display_image = np.zeros(
        (display_height,display_width),dtype=np.float32)
    xs = np.clip(
        (event_cloud[:,1]*display_width).astype(np.int64),
        0,display_width-1)
    ys = np.clip(
        (event_cloud[:,2]*display_height).astype(np.int64),
        0,display_height-1)
    np.add.at(display_image,(ys,xs),event_cloud[:,3])
    return display_image


class _NotebookPolicyDebugView:
    """Live notebook view of the exact policy image and control command."""

    def __init__(self,event_sensor_shape:tuple[int,int]=(224,224),
                 show_event_rgb:bool=False):
        from IPython import get_ipython
        from IPython.display import display
        import matplotlib.pyplot as plt

        if get_ipython() is None:
            raise RuntimeError(
                "Deployment debug mode requires an active IPython/Jupyter session.")

        self._display = display
        self._plt = plt
        self._handle = None
        self._figure = None
        self._image_axis = None
        self._rgb_axis = None
        self._thrust_axis = None
        self._rate_axis = None
        self._image_artist = None
        self._rgb_artist = None
        self._thrust_bar = None
        self._thrust_label = None
        self._bars = None
        self._rate_labels = None
        self._event_sensor_shape = event_sensor_shape
        self._show_event_rgb = show_event_rgb

    def update(self,pilot_name:str,image_modality:VisualModality,
               timestamp:float,image:np.ndarray,command:np.ndarray,
               source_rgb:np.ndarray|None=None) -> None:
        image = np.asarray(image)
        command = np.asarray(command).reshape(-1)
        if command.shape != (4,):
            raise ValueError(
                f"Expected [collective, wx, wy, wz], got command shape {command.shape}.")
        if is_event_cloud_modality(image_modality):
            display_image = _event_cloud_debug_image(
                image,self._event_sensor_shape)
        else:
            expected_channels = image_modality_channels(image_modality)
            if image.ndim != 3 or image.shape[-1] != expected_channels:
                raise ValueError(
                    "Expected the policy image to have shape "
                    f"(H, W, {expected_channels}), got {image.shape}.")

        if is_event_cloud_modality(image_modality):
            pass
        elif image_modality == "event_voxel_grid":
            display_image = image.sum(axis=-1)
        elif image_modality == "event_voxel_grid_polarity":
            display_image = (
                image[...,:5].sum(axis=-1)-image[...,5:].sum(axis=-1))
        else:
            # Grayscale images are repeated to three model channels.
            display_image = (
                image[...,0]
                if (is_event_modality(image_modality)
                    or is_grayscale_modality(image_modality))
                else image)
        voxel_limit = max(1e-6,float(np.max(np.abs(display_image))))
        thrust = command[0]
        rates = command[1:4]
        if self._show_event_rgb:
            if source_rgb is None:
                source_rgb = np.zeros(
                    (*self._event_sensor_shape,3),dtype=np.uint8)
            source_rgb = np.asarray(source_rgb)
            if source_rgb.ndim != 3 or source_rgb.shape[-1] != 3:
                raise ValueError(
                    "Event debug RGB source must have shape (H,W,3).")

        if self._figure is None:
            if self._show_event_rgb:
                self._figure = self._plt.figure(figsize=(13,7))
                grid = self._figure.add_gridspec(
                    2,3,width_ratios=[2,0.65,1])
                self._image_axis = self._figure.add_subplot(grid[0,0])
                self._rgb_axis = self._figure.add_subplot(grid[1,0])
                self._thrust_axis = self._figure.add_subplot(grid[:,1])
                self._rate_axis = self._figure.add_subplot(grid[:,2])
            else:
                (self._figure,
                 (self._image_axis,self._thrust_axis,self._rate_axis)) = (
                    self._plt.subplots(
                        1,3,figsize=(13,4),
                        gridspec_kw={"width_ratios":[2,0.65,1]}))
            if (is_voxel_grid_modality(image_modality)
                    or is_event_cloud_modality(image_modality)):
                self._image_artist = self._image_axis.imshow(
                    display_image,cmap="coolwarm",
                    vmin=-voxel_limit,vmax=voxel_limit)
            elif (is_event_modality(image_modality)
                  or is_grayscale_modality(image_modality)):
                self._image_artist = self._image_axis.imshow(
                    display_image,cmap="gray",vmin=0,vmax=255)
            else:
                self._image_artist = self._image_axis.imshow(display_image)
            self._image_axis.axis("off")
            if self._show_event_rgb:
                self._rgb_artist = self._rgb_axis.imshow(source_rgb)
                self._rgb_axis.set_title("RGB source frame")
                self._rgb_axis.axis("off")

            self._thrust_bar = self._thrust_axis.bar(
                [r"$f_{th}$"],[thrust],color="tab:blue")[0]
            self._thrust_axis.axhline(0,color="black",linewidth=0.8)
            self._thrust_axis.set_ylabel("Thrust command")
            self._thrust_label = self._thrust_axis.text(
                self._thrust_bar.get_x()+self._thrust_bar.get_width()/2,
                0,"",ha="center",va="bottom")

            self._bars = self._rate_axis.bar(
                [r"$\omega_x$",r"$\omega_y$",r"$\omega_z$"],rates)
            self._rate_axis.axhline(0,color="black",linewidth=0.8)
            self._rate_axis.set_ylabel("Body-rate command [rad/s]")
            self._rate_labels = [
                self._rate_axis.text(
                    bar.get_x()+bar.get_width()/2,0,"",ha="center",va="bottom")
                for bar in self._bars
            ]
            self._figure.tight_layout()

        self._image_artist.set_data(display_image)
        if self._rgb_artist is not None:
            self._rgb_artist.set_data(source_rgb)
        if (is_voxel_grid_modality(image_modality)
                or is_event_cloud_modality(image_modality)):
            self._image_artist.set_cmap("coolwarm")
            self._image_artist.set_clim(-voxel_limit,voxel_limit)
        elif (is_event_modality(image_modality)
              or is_grayscale_modality(image_modality)):
            self._image_artist.set_cmap("gray")
            self._image_artist.set_clim(0,255)
        self._image_axis.set_title(
            f"{pilot_name}: {image_modality} policy input at t={timestamp:.3f} s")
        thrust_limit = max(1.0,float(abs(thrust))*1.25)
        self._thrust_axis.set_ylim(-thrust_limit,thrust_limit)
        self._thrust_axis.set_title("Thrust output")
        self._thrust_bar.set_height(thrust)
        self._thrust_label.set_position((
            self._thrust_bar.get_x()+self._thrust_bar.get_width()/2,thrust))
        self._thrust_label.set_text(f"{thrust:.4f}")
        self._thrust_label.set_va("bottom" if thrust >= 0 else "top")

        rate_limit = max(1.0,float(np.max(np.abs(rates)))*1.25)
        self._rate_axis.set_ylim(-rate_limit,rate_limit)
        self._rate_axis.set_title("Body-rate outputs")
        for bar,label,value in zip(self._bars,self._rate_labels,rates):
            bar.set_height(value)
            label.set_position((bar.get_x()+bar.get_width()/2,value))
            label.set_text(f"{value:.3f}")
            label.set_va("bottom" if value >= 0 else "top")

        self._figure.canvas.draw_idle()
        if self._handle is None:
            self._handle = self._display(self._figure,display_id=True)
        else:
            self._handle.update(self._figure)


class _DebugPolicyController:
    """Controller proxy that reports the exact input/output of each command."""

    def __init__(self,controller,pilot_name:str,image_modality:VisualModality,debug_view):
        self._controller = controller
        self._pilot_name = pilot_name
        self._image_modality = image_modality
        self._debug_view = debug_view
        self._source_rgb = None

    def __getattr__(self,name):
        return getattr(self._controller,name)

    def set_debug_rgb(self,rgb:np.ndarray) -> None:
        """Receive the rendered RGB frame paired with an event policy input."""
        self._source_rgb = np.asarray(rgb).copy()

    def control(self,t_cr,x_cr,u_pr,rgb_cr,dpt_cr,fts_cr):
        command,timing = self._controller.control(
            t_cr,x_cr,u_pr,rgb_cr,dpt_cr,fts_cr)
        debug_image = (
            rgb_to_grayscale(rgb_cr)
            if is_grayscale_modality(self._image_modality)
            else rgb_cr)
        source_rgb = (
            self._source_rgb
            if is_event_modality(self._image_modality) else rgb_cr)
        self._debug_view.update(
            self._pilot_name,self._image_modality,t_cr,debug_image,command,
            source_rgb)
        return command,timing

def deploy_roster(cohort_name:str,
                  course_name:str,gsplat_name:str,method_name:str,
                  roster:List[str],
                  expert_name:str="Viper",expert_cname:str=None,
                  bframe_name:str="carl",
                  mode:Literal["evaluate","visualize","generate","debug"]="evaluate",
                  show_table:bool=False,
                  image_modality:VisualModality="rgb",
                  event_device:Literal["auto","cpu","cuda"]="auto",
                  event_surface_options:dict[str,dict]|None=None,
                  event_cloud_options:dict|None=None,
                  debug:bool=False,
                  require_commnet_weights:bool=True) -> Union[None,dict]:
    """"
    Simulate a roster of pilots on a given course within a given scene on
    variations of a specific drone frame using a specified method. This is
    a close mirror to generate_rollout_data with a few key differences; it
    computes flight performance metrics (Trajectory Tracking Error [TTE] 
    and Proximity Percentile [PP]) across multiple rollouts and it produces
    video output for the last trajectory for each pilot. 
    
    Args:
        cohort_name:    Name of the cohort to be used for the simulation.
        deploy:         (course_name,gsplat_name,method_name) tuple.
        roster:         List of pilot names to simulate.
        expert_name:    Name of the expert pilot to be used for the simulation (default is Viper).
        expert_cname:   Name of the course to be used for the expert pilot (default is None).
        bframe_name:    Base frame for flying the trajectories (default is carl).
        mode:           Deployment output mode. "debug" shows policy I/O without saving files.
        show_table:     Boolean to print the summary table of flight metrics.
        image_modality: Visual input used by student pilots.
        event_device:   Device used for synchronous online v2e processing.
        event_surface_options: Per-modality event-surface parameter overrides.
        event_cloud_options: SECNet cloud point-count and sampling-seed overrides.
        debug:          Show each policy image, thrust, and body-rate output live in a notebook.
        require_commnet_weights: Reject student deployment when selected CommNet weights are missing.

    Returns:
        None:           The function saves the simulation data and video to disk.
    """

    image_modality = validate_visual_modality(image_modality)
    resolved_surface_options = (
        resolve_event_surface_options(
            (image_modality,),event_surface_options)
        if (is_event_modality(image_modality)
            and not is_event_cloud_modality(image_modality)) else {})
    resolved_cloud_options = (
        resolve_event_cloud_options(event_cloud_options)
        if is_event_cloud_modality(image_modality) else None)
    if not is_event_cloud_modality(image_modality) and event_cloud_options:
        raise ValueError(
            "event_cloud_options requires image_modality='event_cloud'.")
    if is_event_cloud_modality(image_modality) and event_surface_options:
        raise ValueError(
            "event_surface_options cannot be used with image_modality='event_cloud'.")
    if event_device not in ("auto","cpu","cuda"):
        raise ValueError("event_device must be 'auto', 'cpu', or 'cuda'.")
    if event_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("event_device='cuda' requires CUDA to be available.")
    resolved_event_device = None if event_device == "auto" else event_device

    # Extract configs
    course = ch.get_config(course_name,"courses")
    gsplat = ch.get_gsplat(gsplat_name)
    method = ch.get_config(method_name,"methods")
    expert = ch.get_config(expert_name,"pilots")
    bframe = ch.get_config(bframe_name,"frames")
    
    if expert_cname is not None:
        expert_course = ch.get_config(expert_cname,"courses")
    else:
        expert_course = course

    # Unpack some stuff
    m_bs,kt_bs = bframe["mass"],bframe["motor_thrust_coeff"]
    kT,use_l2_time = expert["plan"]["kT"],expert["plan"]["use_l2_time"]
    hz = expert["track"]["hz"]
    
    # Compute the desired variables
    mts = MinTimeSnap(course["waypoints"],hz,kT,use_l2_time)
    fex = ExternalForces(course["forces"])

    Tsd,FOd = mts.get_desired_trajectory()
    tXUd = th.TsFO_to_tXU(Tsd,FOd,m_bs,kt_bs,fex)

    # Get the batch of sample start times
    t0,tf = Tsd[0],Tsd[-1]
    dt_ro = method["duration"] or tf-t0
    rate = method["rate"] or 1/dt_ro
    reps = method["reps"] or 1

    Tsp_bt = sh.compute_Tsp_batches(t0,tf,dt_ro,rate,reps)[0]

    # Generate sample frames and perturbations
    Frames = sh.generate_frames(
        Tsp_bt, bframe, method["randomization"]["parameters"]
    )
    Perturbations = sh.generate_perturbations(
        Tsp_bt, tXUd, method["randomization"]["initial"]
    )

    # Initialize the rich variables
    console = ru.get_console()
    table = ru.get_deployment_table()

    # Simulate samples across expert+roster
    crew = ["expert"]+roster
    # crew = roster
    
    # Initialize the simulator
    simulator = (
        EventSimulator(gsplat,method)
        if is_event_modality(image_modality)
        else Simulator(gsplat,method)
    )
    simulator.update_forces(course["forces"])
    warmup_controller = (
        VehicleRateMPC(expert,expert_course)
        if is_event_modality(image_modality)
        else None
    )
    camera_config = bframe.get("camera",{})
    event_sensor_shape = (
        int(camera_config.get("height",224)),
        int(camera_config.get("width",224)))
    debug_view = (
        _NotebookPolicyDebugView(
            event_sensor_shape,
            show_event_rgb=is_event_modality(image_modality))
        if debug or mode == "debug" else None)

    Metrics = {}
    for pilot in crew:
        # Load Pilot
        if pilot == "expert":
            controller = warmup_controller or VehicleRateMPC(expert,expert_course)
        else:
            controller = Pilot(
                cohort_name,pilot,image_modality=image_modality,
                event_cloud_options=resolved_cloud_options,
                require_commnet_weights=require_commnet_weights)
            controller.set_mode('deploy')
        base_controller = controller
        controller_modality = "rgb" if pilot == "expert" else image_modality
        if debug_view is not None and pilot != "expert":
            controller = _DebugPolicyController(
                base_controller,pilot,controller_modality,debug_view)

        # Simulate trajectory across samples
        trajectories = []
        for idx,(frame,perturbation) in tqdm(enumerate(zip(Frames,Perturbations)), total=len(Frames), desc=f"Simulating {pilot}"):
            # Unpack rollout variables
            t0,x0 = perturbation["t0"],perturbation["x0"]
            tf = t0 + dt_ro

            # Update the simulation variables
            simulator.update_frame(frame)

            # Update pilot
            controller.reset_memory(x0)
            controller.update_frame(frame)
            if warmup_controller is not None and warmup_controller is not base_controller:
                warmup_controller.reset_memory(x0)
                warmup_controller.update_frame(frame)

            # Simulate Trajectory
            if is_event_modality(image_modality):
                online_events = None
                callback = None
                if pilot != "expert":
                    expected_windows = int(round(dt_ro*controller.hz))
                    online_events = (
                        OnlineEventCloudGenerator(
                            expected_windows,device=resolved_event_device,
                            event_cloud_options=resolved_cloud_options,
                            stream_id=f"{pilot}:{idx}")
                        if is_event_cloud_modality(image_modality)
                        else OnlineEventImageGenerator(
                            expected_windows,device=resolved_event_device,
                            image_modality=image_modality,
                            event_surface_options=resolved_surface_options))
                    callback = online_events.process_frame

                try:
                    Tro,Xro,Uro,Wro,Rgb,Dpt,Tsol = simulator.simulate_with_events(
                        controller,t0,tf,x0,callback,
                        warmup_steps=int(
                            simulator.conFiG["rollout"]["frequency"]/controller.hz),
                        warmup_policy=warmup_controller,
                        image_modality=controller_modality)
                    if online_events is not None:
                        online_events.close()
                except Exception as e:
                    print(f"An error occurred during simulation for pilot '{pilot}' on sample {idx}: {e}")
                    if online_events is not None:
                        online_events.abort()
                    raise
            else:
                Tro,Xro,Uro,Wro,Rgb,Dpt,Tsol = simulator.simulate(
                    controller,t0,tf,x0)

            # Compute Additional Variables
            prms = svu.compute_prms(frame)
            Wrs = svu.compute_Wrs(Xro,Uro,Wro,frame,bframe)
            FOro = svu.compute_FOro(Tro,Xro,Uro,Wro,frame)

            # Save Trajectory
            trajectory = {
                "Tro":Tro,"Xro":Xro,"Uro":Uro,"Wro":Wro,
                "params":prms,"Wrs":Wrs,"FOro":FOro,
                "tXUd":tXUd,"Ndata":Uro.shape[0],"Tsol":Tsol,
                "rollout_id":"sim"+str(0).zfill(3)+str(idx).zfill(3),
                "frame":frame}
            
            trajectories.append(trajectory)

        # Compile deployment data
        deployment_data = {
            "trajectories":trajectories,
            "video": {"hz":controller.hz,"rgb":Rgb,"depth":Dpt},
        }

        # Update the metrics table
        metrics = fh.compute_flight_metrics(trajectories)
        table = ru.update_deployment_table(table,pilot,metrics)
        
        # Update the metrics dictionary
        Metrics[pilot] = metrics

        # Save last trajectory as video/flight recorder
        if mode == "visualize":
            save_deployments(cohort_name,course_name,pilot,deployment_data,
                             is_generate=False)
        elif mode == "generate":
            save_deployments(cohort_name,course_name,pilot,deployment_data,
                             is_generate=True)
        elif mode in ("evaluate","debug"):
            pass  # No action needed for evaluate mode
            
    # Print the summary table
    if show_table:
        console.print(table)
    
    if mode in ("evaluate","debug"):
        return Metrics
    else:
        return None

def save_deployments(cohort_name:str,course_name:str,pilot_name:str,deployment_data:dict,
                     is_generate:bool=False) -> None:
    
    # Some useful path(s)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    deployment_path = os.path.join(workspace_path,"cohorts",cohort_name,"deployment_data")
    
    # Create the deployment directory if it doesn't exist
    if not os.path.exists(deployment_path):
        os.makedirs(deployment_path)

    # Save the Data
    if is_generate:
        # TODO: Implement the generation of flight recorder data
        pass
        # for trajectory in deployment_data["trajectories"]:
        #     Tro:np.ndarray = trajectory["Tro"]
        #     Xro:np.ndarray = trajectory["Xro"]
        #     Uro:np.ndarray = trajectory["Uro"]
        #     Tsol:np.ndarray = trajectory["Tsol"]
        #     tXUd,obj = trajectory["tXUd"],trajectory["obj"]
        #     Adv = None

        #     hz = int(1/(Tro[1]-Tro[0]))
        #     flight_record = rf.FlightRecorder(
        #         Xro.shape[0],Uro.shape[0],
        #         hz,tXUd[0,-1],[360,640,3],obj,cohort_name,course_name,pilot_name)
        #     flight_record.simulation_import(images,Tro,Xro,Uro,tXUd,Tsol,Adv)
        #     flight_record.save()        
    else:
        data_name = "sim_"+course_name+"_"+pilot_name
        trajectories = deployment_data["trajectories"]
        rgbs = deployment_data["video"]["rgb"]
        dpts = deployment_data["video"]["depth"]
        hz = deployment_data["video"]["hz"]
        
        trajectories_path = os.path.join(deployment_path,data_name+".pt")

        rgb_path = os.path.join(deployment_path,data_name+"_rgb.mp4")
        dpt_path = os.path.join(deployment_path,data_name+"_dpt.mp4")

        torch.save(trajectories,trajectories_path)
        gv.images_to_mp4(rgbs,rgb_path, hz)
        gv.images_to_mp4(dpts,dpt_path, hz)
