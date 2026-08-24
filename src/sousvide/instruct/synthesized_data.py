import os
import torch

from collections.abc import Iterator,Sequence
from typing import Any
from torch.utils.data import ConcatDataset,Dataset,Sampler
from sousvide.synthesize.image_modality import (
    ImageModality,is_event_modality,validate_image_modality)


OBSERVATION_FORMAT_VERSION = 2


def squeeze_data(data:Any):
    """Remove the singleton sample batch dimension from a nested tensor tree."""
    if isinstance(data,torch.Tensor):
        return data.squeeze(0)
    if isinstance(data,list):
        return [squeeze_data(item) for item in data]
    if isinstance(data,dict):
        return {key:squeeze_data(value) for key,value in data.items()}
    if isinstance(data,tuple):
        return tuple(squeeze_data(value) for value in data)
    return data


def stack_samples(samples:list[Any]) -> Any:
    """Pack legacy per-sample tensor trees into contiguous batched tensors."""
    if not samples:
        return []

    first = samples[0]
    if isinstance(first,torch.Tensor):
        first_sample = first.squeeze(0)
        packed = torch.empty(
            (len(samples),*first_sample.shape),dtype=first.dtype,
            device="cpu")
        for index,sample in enumerate(samples):
            packed[index].copy_(sample.squeeze(0),non_blocking=False)
        return packed.contiguous()
    if isinstance(first,dict):
        return {
            key:stack_samples([sample[key] for sample in samples])
            for key in first
        }
    if isinstance(first,tuple):
        return tuple(
            stack_samples([sample[index] for sample in samples])
            for index in range(len(first))
        )
    raise TypeError(
        "Observation samples must contain tensors, dictionaries, or tuples; "
        f"received {type(first).__name__}.")


def _tree_length(data:Any) -> int:
    if isinstance(data,torch.Tensor):
        return len(data)
    if isinstance(data,(list,tuple)):
        return len(data)
    if isinstance(data,dict):
        if not data:
            return 0
        lengths = {_tree_length(value) for value in data.values()}
        if len(lengths) != 1:
            raise ValueError("Observation tensor-tree fields have different lengths.")
        return lengths.pop()
    raise TypeError(f"Unsupported observation container: {type(data).__name__}")


def _tree_index(data:Any,index:int,legacy:bool) -> Any:
    if legacy and isinstance(data,list):
        return squeeze_data(data[index])
    if isinstance(data,torch.Tensor):
        return data[index]
    if isinstance(data,dict):
        return {
            key:_tree_index(value,index,legacy)
            for key,value in data.items()
        }
    if isinstance(data,tuple):
        return tuple(_tree_index(value,index,legacy) for value in data)
    if isinstance(data,list):
        return data[index]
    raise TypeError(f"Unsupported observation container: {type(data).__name__}")


class ObservationData(Dataset):
    """Dataset backed by either legacy samples or versioned batched tensors."""

    def __init__(self,Xnn:Any,Ynn:Any,format_version:int=1):
        self.Xnn = Xnn
        self.Ynn = Ynn
        self.format_version = format_version
        self.legacy = format_version < OBSERVATION_FORMAT_VERSION
        self._length = _tree_length(Xnn)
        if _tree_length(Ynn) != self._length:
            raise ValueError("Observation inputs and labels have different lengths.")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self,index:int) -> tuple[Any,Any]:
        return (
            _tree_index(self.Xnn,index,self.legacy),
            _tree_index(self.Ynn,index,self.legacy),
        )


def generate_dataset(data_path:str,mmap:bool=True) -> ObservationData:
    """Load an observation dataset on CPU, memory-mapping tensor storage when possible."""
    load_kwargs = {
        "map_location":"cpu",
        "weights_only":False,
    }
    if mmap:
        load_kwargs["mmap"] = True
    try:
        topic_data = torch.load(data_path,**load_kwargs)
    except (RuntimeError,TypeError,ValueError):
        # Older torch serialization formats cannot always be memory-mapped.
        load_kwargs.pop("mmap",None)
        topic_data = torch.load(data_path,**load_kwargs)

    return ObservationData(
        topic_data["Xnn"],topic_data["Ynn"],
        format_version=int(topic_data.get("format_version",1)))


class FileGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle within each source file while yielding file-local mini-batches."""

    def __init__(self,datasets:Sequence[Dataset],batch_size:int,
                 shuffle:bool,seed:int=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        self.lengths = [len(dataset) for dataset in datasets]
        self.offsets = []
        offset = 0
        for length in self.lengths:
            self.offsets.append(offset)
            offset += length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self,epoch:int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        for offset,length in zip(self.offsets,self.lengths):
            indices = (
                torch.randperm(length,generator=generator).tolist()
                if self.shuffle
                else list(range(length))
            )
            for start in range(0,length,self.batch_size):
                yield [offset+index for index in indices[start:start+self.batch_size]]

    def __len__(self) -> int:
        return sum(
            (length+self.batch_size-1)//self.batch_size
            for length in self.lengths)


def generate_grouped_dataset(
        data_paths:Sequence[str],batch_size:int,shuffle:bool,
        seed:int=0,mmap:bool=True
        ) -> tuple[ConcatDataset,FileGroupedBatchSampler]:
    """Map observation files once and return a locality-preserving batch sampler."""
    datasets = [generate_dataset(path,mmap=mmap) for path in data_paths]
    if not datasets:
        raise ValueError("At least one observation dataset is required.")
    return (
        ConcatDataset(datasets),
        FileGroupedBatchSampler(datasets,batch_size,shuffle,seed=seed),
    )


def get_data_paths(cohort_name:str,
                   student_name:str,
                   topic_name:str,
                   course_name:str|None=None,
                   image_modality:ImageModality="rgb"
                   ) -> tuple[list[str],list[str]]:
    """Return the file-local train/test split for an observation topic."""
    image_modality = validate_image_modality(image_modality)
    workspace_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    topic_parts = [workspace_path,"cohorts",cohort_name,"observation_data",student_name]
    if is_event_modality(image_modality):
        topic_parts.append(image_modality)
    topic_parts.append(topic_name)
    topic_data_path = os.path.join(*topic_parts)

    course_paths = (
        [course.path for course in os.scandir(topic_data_path) if course.is_dir()]
        if course_name is None
        else [os.path.join(topic_data_path,course_name)]
    )

    train_data,test_data = [],[]
    for course_path in course_paths:
        data_paths = sorted(file.path for file in os.scandir(course_path))
        if len(data_paths) == 1:
            train_data.append(data_paths[0])
            test_data.append(data_paths[0])
        elif len(data_paths) > 1:
            train_data.extend(data_paths[:-1])
            test_data.append(data_paths[-1])
        else:
            raise ValueError("No data found.")

    return train_data,test_data
