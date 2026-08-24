import contextlib
import io
import os
import shutil
import time

import numpy as np
import torch
import torch.optim as optim

import sousvide.flight.deploy_figs as df
import sousvide.synthesize.observation_generator as og
import sousvide.visualize.rich_utilities as ru

from rich.progress import Progress
from torch.utils.data import DataLoader
from typing import Literal

from sousvide.control.networks.base_net import BaseNet
from sousvide.control.pilot import Pilot
from sousvide.control.policy import Policy
from sousvide.instruct.losses import LossFn
from sousvide.instruct.synthesized_data import (
    generate_dataset,generate_grouped_dataset,get_data_paths)
from sousvide.synthesize.image_modality import (
    ImageModality,is_event_modality,validate_image_modality)
from sousvide.synthesize.event_surfaces import resolve_event_surface_options


CompileMode = Literal["none","default","reduce-overhead"]
Precision = Literal["float32","bfloat16"]
RegenMode = bool|Literal["missing"]
NumericalMode = Literal["modern","original"]


def _validate_dataloader_num_workers(dataloader_num_workers:int) -> None:
    if (isinstance(dataloader_num_workers,bool)
            or not isinstance(dataloader_num_workers,int)
            or dataloader_num_workers < 0):
        raise ValueError("dataloader_num_workers must be a non-negative integer.")


def _validate_runtime_options(
        compile_mode:str,precision:str,regen:bool|str) -> None:
    if compile_mode not in ("none","default","reduce-overhead"):
        raise ValueError(
            "compile_mode must be 'none', 'default', or 'reduce-overhead'.")
    if precision not in ("float32","bfloat16"):
        raise ValueError("precision must be 'float32' or 'bfloat16'.")
    if regen not in (True,False,"missing"):
        raise ValueError("regen must be True, False, or 'missing'.")


def _validate_numerical_mode_options(
        network_name:str,numerical_mode:str,regen:bool|str,
        deployment:None|tuple[str,str,str],image_modality:str,
        compile_mode:str,precision:str,persistent_dataloader:bool,
        seed:int) -> None:
    if numerical_mode not in ("modern","original"):
        raise ValueError("numerical_mode must be 'modern' or 'original'.")
    if numerical_mode == "modern":
        return

    conflicts = []
    if network_name not in ("histNet","commNet"):
        conflicts.append("network_name must be 'histNet' or 'commNet'")
    if regen is not False:
        conflicts.append("regen must be False")
    if network_name == "histNet" and deployment is not None:
        conflicts.append("deployment must be None for 'histNet'")
    if network_name == "histNet" and image_modality != "rgb":
        conflicts.append("image_modality must be 'rgb'")
    if compile_mode != "none":
        conflicts.append("compile_mode must be 'none'")
    if precision != "float32":
        conflicts.append("precision must be 'float32'")
    if persistent_dataloader:
        conflicts.append("persistent_dataloader must be False")
    if seed != 0:
        conflicts.append(
            "seed must remain 0; seed the ambient torch RNG before training")
    if conflicts:
        raise ValueError(
            "original numerical mode requires: "+"; ".join(conflicts)+".")


def _move_to_device(data,device:torch.device,non_blocking:bool=False):
    if isinstance(data,torch.Tensor):
        return data.to(device,non_blocking=non_blocking)
    if isinstance(data,dict):
        return {
            key:_move_to_device(value,device,non_blocking)
            for key,value in data.items()
        }
    if isinstance(data,list):
        return [_move_to_device(value,device,non_blocking) for value in data]
    if isinstance(data,tuple):
        return tuple(
            _move_to_device(value,device,non_blocking) for value in data)
    return data


def _create_dataloader(dataset,batch_size:int|None,shuffle:bool,
                       dataloader_num_workers:int,use_cuda:bool,
                       batch_sampler=None,
                       persistent_workers:bool=False) -> DataLoader:
    common = {
        "dataset":dataset,
        "num_workers":dataloader_num_workers,
        "pin_memory":use_cuda,
        "persistent_workers":(
            persistent_workers and dataloader_num_workers > 0),
    }
    if batch_sampler is not None:
        return DataLoader(batch_sampler=batch_sampler,**common)
    return DataLoader(
        batch_size=batch_size,shuffle=shuffle,drop_last=False,**common)


def _observation_data_available(
        cohort_name:str,student_name:str,network_name:str,
        image_modality:ImageModality) -> bool:
    try:
        train_paths,test_paths = get_data_paths(
            cohort_name,student_name,network_name,
            image_modality=image_modality)
    except (FileNotFoundError,ValueError):
        return False
    return bool(train_paths and test_paths)


def _batch_length(data) -> int:
    if isinstance(data,torch.Tensor):
        return len(data)
    if isinstance(data,dict):
        if not data:
            return 0
        return _batch_length(next(iter(data.values())))
    if isinstance(data,(list,tuple)):
        if not data:
            return 0
        return _batch_length(data[0])
    raise TypeError(f"Cannot determine batch length from {type(data).__name__}.")


def _float_outputs(outputs:dict[str,torch.Tensor]) -> dict[str,torch.Tensor]:
    return {key:value.float() for key,value in outputs.items()}


def _autocast_context(device:torch.device,precision:str):
    if device.type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda",dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _record_stream(data,stream:torch.cuda.Stream) -> None:
    if isinstance(data,torch.Tensor):
        data.record_stream(stream)
    elif isinstance(data,dict):
        for value in data.values():
            _record_stream(value,stream)
    elif isinstance(data,(list,tuple)):
        for value in data:
            _record_stream(value,stream)


class _CUDAPrefetcher:
    def __init__(self,dataloader:DataLoader,device:torch.device):
        self.iterator = iter(dataloader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self.loader_wait_seconds = 0.0
        self.transfer_events:list[tuple[torch.cuda.Event,torch.cuda.Event]] = []
        self._preload()

    def _preload(self) -> None:
        wait_start = time.perf_counter()
        try:
            cpu_batch = next(self.iterator)
        except StopIteration:
            self.loader_wait_seconds += time.perf_counter()-wait_start
            self.next_batch = None
            return
        self.loader_wait_seconds += time.perf_counter()-wait_start

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start.record(self.stream)
            self.next_batch = _move_to_device(
                cpu_batch,self.device,non_blocking=True)
            end.record(self.stream)
        self.transfer_events.append((start,end))

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.stream)
        batch = self.next_batch
        _record_stream(batch,current_stream)
        self._preload()
        return batch

    def transfer_seconds(self) -> float:
        torch.cuda.synchronize(self.device)
        return sum(
            start.elapsed_time(end) for start,end in self.transfer_events)/1000


def _device_batches(
        dataloader:DataLoader,device:torch.device,use_cuda:bool,
        cuda_prefetch:bool,timings:dict[str,float]):
    if cuda_prefetch and use_cuda:
        prefetcher = _CUDAPrefetcher(dataloader,device)
        for batch in prefetcher:
            yield batch
        timings["loader_wait_seconds"] += prefetcher.loader_wait_seconds
        timings["h2d_seconds"] += prefetcher.transfer_seconds()
        return

    transfer_events = []
    iterator_start = time.perf_counter()
    iterator = iter(dataloader)
    timings["loader_wait_seconds"] += time.perf_counter()-iterator_start
    while True:
        wait_start = time.perf_counter()
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            timings["loader_wait_seconds"] += time.perf_counter()-wait_start
            break
        timings["loader_wait_seconds"] += time.perf_counter()-wait_start

        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            batch = _move_to_device(cpu_batch,device,non_blocking=True)
            end.record()
            transfer_events.append((start,end))
        else:
            transfer_start = time.perf_counter()
            batch = _move_to_device(cpu_batch,device)
            timings["h2d_seconds"] += time.perf_counter()-transfer_start
        yield batch

    if transfer_events:
        torch.cuda.synchronize(device)
        timings["h2d_seconds"] += sum(
            start.elapsed_time(end) for start,end in transfer_events)/1000


def _compile_network(network:BaseNet,compile_mode:str):
    if compile_mode == "none":
        return network
    if compile_mode == "default":
        return torch.compile(network)
    return torch.compile(network,mode=compile_mode)


def _new_phase_timings() -> dict[str,float]:
    return {"loader_wait_seconds":0.0,"h2d_seconds":0.0}


def _compute_seconds(
        events:list[tuple[torch.cuda.Event,torch.cuda.Event]],
        device:torch.device) -> float:
    if not events:
        return 0.0
    torch.cuda.synchronize(device)
    return sum(start.elapsed_time(end) for start,end in events)/1000


def _estimate_epoch_completion(
        epoch_durations:list[float],remaining_epochs:int,
        now:float|None=None) -> tuple[str,str]:
    if not epoch_durations:
        return "calculating...","--"
    if remaining_epochs <= 0:
        return "complete","0s"

    remaining_seconds = int(round(
        float(np.mean(epoch_durations))*remaining_epochs))
    completion_time = (time.time() if now is None else now)+remaining_seconds
    eta = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(completion_time))
    days,remainder = divmod(remaining_seconds,24*60*60)
    hours,remainder = divmod(remainder,60*60)
    minutes,seconds = divmod(remainder,60)
    if days:
        remaining = f"{days}d {hours:02d}h {minutes:02d}m"
    elif hours:
        remaining = f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes:
        remaining = f"{minutes}m {seconds:02d}s"
    else:
        remaining = f"{seconds}s"
    return eta,remaining


def train_roster(cohort_name:str,roster:list[str],network_name:str,Neps:int,
                 regen:RegenMode=False,
                 deployment:None|tuple[str,str,str]=None,
                 lim_sv:int=50,lr:float=1e-4,batch_size:int=64,
                 image_modality:ImageModality="rgb",
                 dataloader_num_workers:int=4,
                 compile_mode:CompileMode="none",
                 precision:Precision="float32",
                 persistent_dataloader:bool=False,
                 cuda_prefetch:bool=False,seed:int=0,
                 numerical_mode:NumericalMode="modern",
                 event_surface_options:dict[str,dict]|None=None):
    image_modality = validate_image_modality(image_modality)
    _validate_dataloader_num_workers(dataloader_num_workers)
    _validate_runtime_options(compile_mode,precision,regen)
    _validate_numerical_mode_options(
        network_name,numerical_mode,regen,deployment,image_modality,
        compile_mode,precision,persistent_dataloader,seed)

    console = ru.get_console()
    progress = ru.get_training_progress()
    generation_start = time.perf_counter()
    generation_roster = []
    if regen is True:
        generation_roster = roster
    elif regen == "missing":
        generation_roster = [
            student for student in roster
            if not _observation_data_available(
                cohort_name,student,network_name,image_modality)
        ]
    if generation_roster:
        console.print("Regenerating observation data...")
        og.generate_observation_data(
            cohort_name,generation_roster,networks=[network_name],
            image_modality=image_modality)
    else:
        console.print("Using existing observation data...")
    observation_generation_seconds = time.perf_counter()-generation_start

    with progress:
        for student in roster:
            student_desc = f"[bold green3]{student:>8} > {network_name}[/]"
            student_task = progress.add_task(
                student_desc,total=Neps,loss=0.0,units="epochs",
                eta="calculating...",remaining="--")
            train_student(
                cohort_name,student,network_name,Neps,
                deployment=deployment,lim_sv=lim_sv,lr=lr,
                batch_size=batch_size,progress_bar=(progress,student_task),
                image_modality=image_modality,
                dataloader_num_workers=dataloader_num_workers,
                compile_mode=compile_mode,precision=precision,
                persistent_dataloader=persistent_dataloader,
                cuda_prefetch=cuda_prefetch,seed=seed,
                numerical_mode=numerical_mode,
                event_surface_options=event_surface_options,
                observation_generation_seconds=observation_generation_seconds)
            progress.refresh()


def train_student(cohort_name:str,student_name:str,network_name:str,Neps:int,
                  deployment:None|tuple[str,str,str]=None,
                  lim_sv:int=10,lr:float=1e-4,batch_size:int=64,
                  progress_bar:tuple[Progress,int]|None=None,
                  image_modality:ImageModality="rgb",
                  dataloader_num_workers:int=4,
                  compile_mode:CompileMode="none",
                  precision:Precision="float32",
                  persistent_dataloader:bool=False,
                  cuda_prefetch:bool=False,seed:int=0,
                  numerical_mode:NumericalMode="modern",
                  event_surface_options:dict[str,dict]|None=None,
                  observation_generation_seconds:float=0.0) -> None:
    image_modality = validate_image_modality(image_modality)
    resolved_event_surface_options = (
        resolve_event_surface_options(
            (image_modality,),event_surface_options)
        if is_event_modality(image_modality) else {})
    _validate_dataloader_num_workers(dataloader_num_workers)
    _validate_runtime_options(compile_mode,precision,False)
    _validate_numerical_mode_options(
        network_name,numerical_mode,False,deployment,image_modality,
        compile_mode,precision,persistent_dataloader,seed)

    start_time = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if precision == "bfloat16" and not use_cuda:
        raise RuntimeError("bfloat16 training requires a CUDA device.")
    criterion = LossFn()
    student = Pilot(
        cohort_name,student_name,image_modality=image_modality)

    if network_name not in student.policy.networks:
        if progress_bar is not None:
            progress,student_task = progress_bar
            progress.update(
                student_task,
                description=f"{student_name} does not use {network_name}.")
        return

    network:BaseNet = get_network(student.policy,network_name)
    opt = optim.Adam(network.parameters(),lr=lr)
    training_network = _compile_network(network,compile_mode)
    student_path = student.path
    network_path = student.policy.network_paths[network_name]
    losses_path = os.path.join(student_path,"losses_"+network_name+".pt")
    artifact_name = (
        f"{network_name}_{image_modality}"
        if network_name == "commNet" else network_name)
    prev_losses_log = (
        torch.load(losses_path,weights_only=False)
        if os.path.exists(losses_path) else {})

    setup_timings = {
        "observation_generation_seconds":observation_generation_seconds,
        "dataset_initialization_seconds":0.0,
        "initial_checkpoint_seconds":0.0,
        "initial_deployment_seconds":0.0,
    }
    Eval_tte = []
    if deployment is not None:
        course,scene,eval_method = deployment
        ckpts_path = os.path.join(student_path,"ckpts",artifact_name)
        os.makedirs(ckpts_path,exist_ok=True)
        ckpt_path = os.path.join(
            ckpts_path,artifact_name+"_ckpt"+str(0).zfill(3)+".pt")

        if numerical_mode != "original":
            # Modern mode makes a newly initialized network available to the
            # deployment loader before evaluating it.
            checkpoint_start = time.perf_counter()
            torch.save(network,network_path)
            torch.save(network,ckpt_path)
            setup_timings["initial_checkpoint_seconds"] = (
                time.perf_counter()-checkpoint_start)

        deployment_start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            metric = df.deploy_roster(
                cohort_name,course,scene,eval_method,[student_name],
                mode="evaluate",image_modality=image_modality,
                event_surface_options=resolved_event_surface_options,
                require_commnet_weights=numerical_mode != "original")
        setup_timings["initial_deployment_seconds"] = (
            time.perf_counter()-deployment_start)
        Eval_tte.append((0,metric[student_name]["TTE"]["mean"]))

        # The legacy trainer records checkpoint zero after its initial
        # deployment. Keep modality-specific names to prevent RGB/event
        # checkpoints from colliding in the current repository.
        if numerical_mode == "original":
            checkpoint_start = time.perf_counter()
            torch.save(network,ckpt_path)
            setup_timings["initial_checkpoint_seconds"] = (
                time.perf_counter()-checkpoint_start)
    if numerical_mode == "original" and deployment is None:
        # Match the legacy no-deployment loss-log layout.
        Eval_tte = [[]]

    dataset_start = time.perf_counter()
    od_train_files,od_test_files = get_data_paths(
        cohort_name,student.name,network_name,
        image_modality=image_modality)
    if numerical_mode == "original":
        # Preserve the original file-local DataLoader iteration pattern. Each
        # iterator consumes the same global RNG draws as the legacy loop, while
        # the immutable datasets are mapped only once instead of every epoch.
        train_loaders = [
            _create_dataloader(
                generate_dataset(path,mmap=True),batch_size,True,
                dataloader_num_workers,use_cuda)
            for path in od_train_files
        ]
        test_loaders = [
            _create_dataloader(
                generate_dataset(path,mmap=True),batch_size,True,
                dataloader_num_workers,use_cuda)
            for path in od_test_files
        ]
        train_sampler = None
        train_loader = test_loader = None
    else:
        train_dataset,train_sampler = generate_grouped_dataset(
            od_train_files,batch_size,True,seed=seed,mmap=True)
        test_dataset,test_sampler = generate_grouped_dataset(
            od_test_files,batch_size,False,seed=seed,mmap=True)
        train_loader = _create_dataloader(
            train_dataset,None,False,dataloader_num_workers,use_cuda,
            batch_sampler=train_sampler,
            persistent_workers=persistent_dataloader)
        test_loader = _create_dataloader(
            test_dataset,None,False,dataloader_num_workers,use_cuda,
            batch_sampler=test_sampler,
            persistent_workers=persistent_dataloader)
        train_loaders = test_loaders = None
    setup_timings["dataset_initialization_seconds"] = (
        time.perf_counter()-dataset_start)

    loss_entry = {
        "network":network_name,"image_modality":image_modality,
        "event_surface_config":resolved_event_surface_options.get(
            image_modality,{}),
        "numerical_mode":numerical_mode,
        "compile_mode":compile_mode,"precision":precision,
        "batch_size":batch_size,
        "persistent_dataloader":persistent_dataloader,
        "cuda_prefetch":cuda_prefetch,
        "N_eps":None,"Nd_tn":None,"Nd_tt":None,"t_tn":None,
        "Loss_tn":[],"Loss_tt":[],"Eval_tte":[],
        "timings":{"setup":setup_timings,"epochs":[]},
    }

    Loss_tn,Loss_tt = [],[]
    epoch_durations = []
    for ep in range(Neps):
        epoch_start = time.perf_counter()
        if numerical_mode == "modern":
            train_sampler.set_epoch(ep)
        network.train()
        train_transfer = _new_phase_timings()
        train_compute_events = []
        train_compute_seconds = 0.0
        train_loss_sum = torch.zeros((),device=device)
        original_train_losses = []
        Ndata_tn = 0
        train_phase_start = time.perf_counter()

        active_train_loaders = (
            train_loaders if numerical_mode == "original" else (train_loader,))
        for active_loader in active_train_loaders:
            for xnn,ylb in _device_batches(
                    active_loader,device,use_cuda,cuda_prefetch,train_transfer):
                current_batch_size = _batch_length(ylb)
                if use_cuda:
                    compute_start = torch.cuda.Event(enable_timing=True)
                    compute_end = torch.cuda.Event(enable_timing=True)
                    compute_start.record()
                else:
                    compute_start_time = time.perf_counter()

                if numerical_mode == "modern":
                    opt.zero_grad(set_to_none=True)
                with _autocast_context(device,precision):
                    ypd = training_network(xnn)
                loss = criterion(
                    ypd if numerical_mode == "original"
                    else _float_outputs(ypd),ylb)
                loss.backward()
                opt.step()
                if numerical_mode == "original":
                    opt.zero_grad()

                if use_cuda:
                    compute_end.record()
                    train_compute_events.append((compute_start,compute_end))
                else:
                    train_compute_seconds += time.perf_counter()-compute_start_time
                if numerical_mode == "original":
                    original_train_losses.append(batch_size*loss.item())
                    Ndata_tn += batch_size
                else:
                    train_loss_sum += loss.detach()*current_batch_size
                    Ndata_tn += current_batch_size

        if use_cuda:
            train_compute_seconds = _compute_seconds(
                train_compute_events,device)
        train_total_seconds = time.perf_counter()-train_phase_start
        epLoss_tn = (
            sum(original_train_losses)/Ndata_tn
            if numerical_mode == "original"
            else (train_loss_sum/Ndata_tn).item())

        if numerical_mode == "modern":
            network.eval()
        test_transfer = _new_phase_timings()
        test_compute_events = []
        test_compute_seconds = 0.0
        test_loss_sum = torch.zeros((),device=device)
        original_test_losses = []
        Ndata_tt = 0
        test_phase_start = time.perf_counter()
        with torch.inference_mode():
            active_test_loaders = (
                test_loaders if numerical_mode == "original" else (test_loader,))
            for active_loader in active_test_loaders:
                for xnn,ylb in _device_batches(
                        active_loader,device,use_cuda,cuda_prefetch,test_transfer):
                    current_batch_size = _batch_length(ylb)
                    if use_cuda:
                        compute_start = torch.cuda.Event(enable_timing=True)
                        compute_end = torch.cuda.Event(enable_timing=True)
                        compute_start.record()
                    else:
                        compute_start_time = time.perf_counter()
                    with _autocast_context(device,precision):
                        ypd = training_network(xnn)
                    loss = criterion(
                        ypd if numerical_mode == "original"
                        else _float_outputs(ypd),ylb)
                    if use_cuda:
                        compute_end.record()
                        test_compute_events.append((compute_start,compute_end))
                    else:
                        test_compute_seconds += time.perf_counter()-compute_start_time
                    if numerical_mode == "original":
                        original_test_losses.append(batch_size*loss.item())
                        Ndata_tt += batch_size
                    else:
                        test_loss_sum += loss.detach()*current_batch_size
                        Ndata_tt += current_batch_size

        if use_cuda:
            test_compute_seconds = _compute_seconds(test_compute_events,device)
        test_total_seconds = time.perf_counter()-test_phase_start
        epLoss_tt = (
            sum(original_test_losses)/Ndata_tt
            if numerical_mode == "original"
            else (test_loss_sum/Ndata_tt).item())
        network.train()

        Loss_tn.append((ep+1,epLoss_tn))
        Loss_tt.append((ep+1,epLoss_tt))
        checkpoint_seconds = 0.0
        deployment_seconds = 0.0

        should_save = ((ep+1)%lim_sv == 0) or (ep+1 == Neps)
        if should_save:
            if use_cuda:
                torch.cuda.synchronize(device)
            checkpoint_start = time.perf_counter()
            torch.save(network,network_path)
            if deployment is not None:
                ckpt_path = os.path.join(
                    ckpts_path,
                    artifact_name+"_ckpt"+str(ep+1).zfill(3)+".pt")
                torch.save(network,ckpt_path)
            checkpoint_seconds = time.perf_counter()-checkpoint_start

            if deployment is not None:
                deployment_start = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    metric = df.deploy_roster(
                        cohort_name,course,scene,eval_method,[student_name],
                        mode="evaluate",image_modality=image_modality,
                        event_surface_options=resolved_event_surface_options,
                        require_commnet_weights=numerical_mode != "original")
                deployment_seconds = time.perf_counter()-deployment_start
                Eval_tte.append(
                    (ep+1,metric[student_name]["TTE"]["mean"]))
            elif numerical_mode == "original":
                Eval_tte.append([])

        epoch_seconds = time.perf_counter()-epoch_start
        epoch_timing = {
            "epoch":ep+1,
            "train_total_seconds":train_total_seconds,
            "train_loader_wait_seconds":train_transfer["loader_wait_seconds"],
            "train_h2d_seconds":train_transfer["h2d_seconds"],
            "train_compute_seconds":train_compute_seconds,
            "validation_total_seconds":test_total_seconds,
            "validation_loader_wait_seconds":test_transfer["loader_wait_seconds"],
            "validation_h2d_seconds":test_transfer["h2d_seconds"],
            "validation_compute_seconds":test_compute_seconds,
            "checkpoint_seconds":checkpoint_seconds,
            "deployment_seconds":deployment_seconds,
            "epoch_seconds":epoch_seconds,
        }
        loss_entry["timings"]["epochs"].append(epoch_timing)

        if should_save:
            loss_entry["Loss_tn"] = np.array(Loss_tn).T
            loss_entry["Loss_tt"] = np.array(Loss_tt).T
            loss_entry["Eval_tte"] = np.array(Eval_tte).T
            loss_entry["Nd_tn"],loss_entry["Nd_tt"] = Ndata_tn,Ndata_tt
            loss_entry["t_tn"] = time.time()-start_time
            loss_entry["N_eps"] = ep+1
            timestamp = time.strftime("%y%m%d_%H%M%S")
            curr_losses_log = prev_losses_log.copy()
            curr_losses_log["log_"+timestamp] = loss_entry
            torch.save(curr_losses_log,losses_path)

        epoch_durations.append(epoch_seconds)
        if progress_bar is not None:
            progress,student_task = progress_bar
            eta,remaining = _estimate_epoch_completion(
                epoch_durations,Neps-(ep+1))
            progress.update(
                student_task,loss=epLoss_tn,advance=1,
                eta=eta,remaining=remaining)
            progress.refresh()

    if deployment is not None:
        best_ckpt = min(Eval_tte,key=lambda value:value[1])[0]
        best_name = artifact_name+"_ckpt"+str(best_ckpt).zfill(3)
        ckpt_path = os.path.join(ckpts_path,best_name+".pt")
        best_network = torch.load(ckpt_path,weights_only=False)
        torch.save(best_network,network_path)
        shutil.rmtree(ckpts_path)
        ru.console.print(
            f"[bold green3]{student_name} > {network_name}[/] : "
            f"Best checkpoint is {best_name}.")

    if progress_bar is not None:
        progress.refresh()


def get_network(policy:Policy,net_name:str|list[str]) -> BaseNet:
    network = policy.networks[net_name]
    network.train()
    for param in network.parameters():
        param.requires_grad = True
    return network
