import os
import torch
from torch.utils.data import Dataset
from typing import Any
from sousvide.synthesize.image_modality import (
    ImageModality,validate_image_modality)

class ObservationData(Dataset):
    def __init__(self,
                 Xnn:list[list[torch.Tensor|dict[str,torch.Tensor]]],
                 Ynn:list[torch.Tensor|dict[str,torch.Tensor]]):
        """
        Initialize the Observation Data.

        Args:
            Xnn:  The input observation data.
            Ynn:  The output observation data.
        """

        # Squeeze the data
        Xnn = squeeze_data(Xnn)
        Ynn = squeeze_data(Ynn)

        # Store the data
        self.Xnn = Xnn
        self.Ynn = Ynn

    def __len__(self):
        return len(self.Xnn)

    def __getitem__(self,idx):
        xnn = self.Xnn[idx]
        ynn = self.Ynn[idx]

        return xnn,ynn
    
def squeeze_data(data:Any):
    if isinstance(data, torch.Tensor):
        return data.squeeze(0)  # Squeeze tensor
    elif isinstance(data, list):
        return [squeeze_data(item) for item in data]  # Recursively process lists
    elif isinstance(data, dict):
        return {key: squeeze_data(value) for key, value in data.items()}  # Process dictionaries
    else:
        return data  # Keep non-tensor values unchanged
    
def generate_dataset(data_path:str,device:torch.device) -> Dataset:
    """
    Generate a Pytorch Dataset from the given list of observation data path.

    Args:
        data_path:  Observation data path.
        device:  The device to use.

    Returns:
        dset:  The Pytorch Dataset object.
    """

    # Load the topic data
    topic_data = torch.load(data_path, map_location=device)

    # Extract the observation data
    Xnn_ds = topic_data["Xnn"]
    Ynn_ds = topic_data["Ynn"]
    
    return ObservationData(Xnn_ds,Ynn_ds)
    
def get_data_paths(cohort_name:str,
                   student_name:str,
                   topic_name:str,
                   course_name:str|None=None,
                   image_modality:ImageModality="rgb"
                   ) -> tuple[list[str],str]:
    """
    Get the paths to the observation data files for training or testing. If mode is 'train',
    the paths are shuffled. This way, we can mix the course data a little better. However, we
    need to keep the order constant across epochs and so we use a rng_seed to lock the randomness.
    If mode is 'test', the first file is selected.

    Args:
        cohort_name:  The name of the cohort.
        student_name: The name of the student.
        course_name:  The name of the course.

    Returns:
        train_data:  The list of training data paths.
        test_data:   The list of testing data paths.
    """

    # Some useful path(s)
    image_modality = validate_image_modality(image_modality)
    workspace_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    topic_parts = [workspace_path,"cohorts",cohort_name,"observation_data",student_name]
    if image_modality == "kronecker_delta":
        topic_parts.append(image_modality)
    topic_parts.append(topic_name)
    topic_data_path = os.path.join(*topic_parts)

    # Get course paths
    if course_name is None:
        course_paths = [course.path for course in os.scandir(topic_data_path) if course.is_dir()]
    else:
        course_paths = [os.path.join(topic_data_path,course_name)]

    # Split into training and testing data
    train_data,test_data = [],[]
    for course_path in course_paths:
        # Get data files for the course
        data_paths = []
        for file in os.scandir(course_path):
            data_paths.append(file.path)

        # Sort the data files
        data_paths.sort()

        # Split into training and testing data (re-use if only one file)
        if len(data_paths) == 1:
            train_data.append(data_paths[0])
            test_data.append(data_paths[0])
        elif len(data_paths) > 1:
            train_data.extend(data_paths[:-1])
            test_data.append(data_paths[-1])
        else:
            raise ValueError("No data found.")

    return train_data,test_data

