"""SECNet event-cloud feature encoder.

Adapted with permission from the SECNet_ICML reference implementation by
Hongwei Ren, Fei Ma, Yuetong Fang, Hongxiang Huang, Yue Zhou, Yulong Huang,
Haotian Fu, Ziyi Yang, Youxin Jiang, Xiangqian Wu, et al. at
commit 0638bb18776f5cdb44912770f8f117c96e59b684:
https://github.com/rhwxmx/SECNet_ICML

SECNet: Scalable Event Cloud Network for Event-based Classification,
Hongwei Ren et al., ICML 2026.  The classifier and research training code are
intentionally omitted.  This control-oriented adaptation parameterizes the
published dimensions and makes farthest-point sampling deterministic in eval
mode.
"""

# SPDX-License-Identifier: GPL-3.0-only

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    raw_size = indices.size()
    flat = indices.reshape(raw_size[0], -1)
    gathered = torch.gather(
        points, 1, flat[...,None].expand(-1,-1,points.size(-1)))
    return gathered.reshape(*raw_size,points.size(-1))


def _square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distance = -2*torch.matmul(source,target.transpose(1,2))
    distance += torch.sum(source*source,dim=-1).unsqueeze(-1)
    distance += torch.sum(target*target,dim=-1).unsqueeze(1)
    return distance


def _farthest_point_sample(
        coordinates: torch.Tensor,npoint: int,deterministic: bool) -> torch.Tensor:
    batch_size,num_points,_ = coordinates.shape
    if npoint > num_points:
        raise ValueError(
            f"Cannot sample {npoint} centroids from {num_points} events.")
    centroids = torch.zeros(
        batch_size,npoint,dtype=torch.long,device=coordinates.device)
    distance = torch.full(
        (batch_size,num_points),float("inf"),device=coordinates.device)
    if deterministic:
        center = coordinates.mean(dim=1,keepdim=True)
        farthest = torch.sum((coordinates-center)**2,dim=-1).argmax(dim=1)
    else:
        farthest = torch.randint(
            0,num_points,(batch_size,),device=coordinates.device)
    batch_indices = torch.arange(batch_size,device=coordinates.device)
    for index in range(npoint):
        centroids[:,index] = farthest
        centroid = coordinates[batch_indices,farthest].unsqueeze(1)
        candidate = torch.sum((coordinates-centroid)**2,dim=-1)
        distance = torch.minimum(distance,candidate)
        farthest = distance.argmax(dim=-1)
    return centroids


class _Linear1Layer(nn.Module):
    def __init__(self,in_channels: int,out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Linear2Layer(nn.Module):
    def __init__(self,channels: int) -> None:
        super().__init__()
        hidden = channels//2
        self.net1 = nn.Sequential(
            nn.Conv1d(channels,hidden,kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(hidden,channels,kernel_size=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net2(self.net1(x))+x)


class _FFTLayer(nn.Module):
    def __init__(self,frequency_shape: tuple[int,int]) -> None:
        super().__init__()
        weight = torch.zeros(*frequency_shape,2,dtype=torch.float32)
        weight[...,0] = 1.0
        self.complex_weight = nn.Parameter(weight)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x,dim=1,norm="ortho")
        spectrum = spectrum*torch.view_as_complex(self.complex_weight)
        return torch.fft.irfft(spectrum,dim=1,norm="ortho")


class _Attention(nn.Module):
    def __init__(self,hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size,1)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x).squeeze(-1),dim=1)


class _LocalGrouper(nn.Module):
    def __init__(self,channel: int,groups: int,kneighbors: int) -> None:
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.fps_scale = nn.Parameter(torch.ones(1,1,4))
        self.affine_alpha = nn.Parameter(torch.ones(1,1,1,channel+4))
        self.affine_beta = nn.Parameter(torch.zeros(1,1,1,channel+4))

    def forward(
            self,event_cloud: torch.Tensor,points: torch.Tensor,
            deterministic: bool) -> tuple[torch.Tensor,torch.Tensor]:
        batch_size,num_points,_ = event_cloud.shape
        if self.kneighbors > num_points:
            raise ValueError(
                f"SECNet requires at least {self.kneighbors} events per group input.")
        scaled_cloud = (self.fps_scale*event_cloud).contiguous()
        # Preserve the released implementation's FPS over its first three
        # [t,x,y,p] channels; polarity still participates in grouped features.
        fps_indices = _farthest_point_sample(
            scaled_cloud[:,:,:3],self.groups,deterministic)
        fps_indices = torch.sort(fps_indices,dim=1).values
        centroid_cloud = _index_points(scaled_cloud,fps_indices)
        centroid_points = _index_points(points,fps_indices)

        distances = _square_distance(centroid_points,points)
        neighbor_indices = torch.argsort(distances,dim=-1)[
            :,:,:self.kneighbors]
        neighbor_indices = torch.sort(neighbor_indices,dim=-1).values
        grouped_cloud = _index_points(scaled_cloud,neighbor_indices)
        evolved_cloud = grouped_cloud.mean(dim=2)
        grouped_points = _index_points(points,neighbor_indices)
        grouped_points = torch.cat((grouped_points,grouped_cloud),dim=-1)

        mean = grouped_points.mean(dim=2,keepdim=True)
        std = torch.std(
            (grouped_points-mean).reshape(batch_size,-1),dim=-1,
            keepdim=True).reshape(batch_size,1,1,1)
        grouped_points = (grouped_points-mean)/(std+1e-5)
        # The released model registers these affine parameters but does not
        # apply them in its center-normalized forward path.  Keep that exact
        # computation so adapted checkpoints retain upstream semantics.
        repeated_centroids = centroid_points.reshape(
            batch_size,self.groups,1,-1).expand(
                -1,-1,self.kneighbors,-1)
        return evolved_cloud,torch.cat(
            (grouped_points,repeated_centroids),dim=-1)


class SECNetEncoder(nn.Module):
    """SECNet hierarchy with its task-specific prediction head removed."""

    def __init__(
            self,
            num_points: int = 4096,
            # feature_dims: tuple[int,...] | list[int] = (32,68,140,284),
            feature_dims: tuple[int,...] | list[int] = (64,132,268,540),
            group_counts: tuple[int,...] | list[int] = (2048,1024,512),
            neighbors: int = 24,
            stages: int = 3,
    ) -> None:
        super().__init__()
        feature_dims = tuple(feature_dims)
        group_counts = tuple(group_counts)
        if isinstance(stages,bool) or not isinstance(stages,int) or stages <= 0:
            raise ValueError("SECNet stages must be a positive integer.")
        if (len(group_counts) != stages
                or len(feature_dims) != stages+1):
            raise ValueError(
                "SECNet expects one group count and one feature transition per stage.")
        if num_points < group_counts[0]:
            raise ValueError("SECNet num_points must cover the first group count.")
        previous_count = num_points
        for group_count in group_counts:
            if group_count <= 0 or group_count > previous_count:
                raise ValueError(
                    "SECNet group_counts must be positive and non-increasing from num_points.")
            if group_count % 2:
                raise ValueError(
                    "SECNet group_counts must be even for temporal FFT reconstruction.")
            if neighbors <= 0 or neighbors > previous_count:
                raise ValueError(
                    "SECNet neighbors must fit every hierarchy input.")
            previous_count = group_count
        for index in range(len(group_counts)):
            expected = 2*feature_dims[index]+4
            if feature_dims[index+1] != expected:
                raise ValueError(
                    "SECNet feature dimensions must satisfy D[i+1] = 2*D[i]+4.")

        self.num_points = num_points
        self.out_features = 2*feature_dims[-1]
        self.embedding = _Linear1Layer(4,feature_dims[0])
        self.groupers = nn.ModuleList()
        self.spatial_fft = nn.ModuleList()
        self.aggregations = nn.ModuleList()
        self.temporal_fft = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for index,group_count in enumerate(group_counts):
            next_dim = feature_dims[index+1]
            self.groupers.append(_LocalGrouper(
                feature_dims[index],group_count,neighbors))
            self.spatial_fft.append(_FFTLayer(
                (next_dim//2+1,neighbors)))
            self.aggregations.append(_Attention(next_dim))
            self.temporal_fft.append(_FFTLayer(
                (group_count//2+1,next_dim)))
            self.residuals.append(_Linear2Layer(next_dim))

    def forward(self,event_cloud: torch.Tensor) -> torch.Tensor:
        if (event_cloud.ndim != 3 or event_cloud.shape[1] != self.num_points
                or event_cloud.shape[2] != 4):
            raise ValueError(
                f"SECNet input must have shape (B,{self.num_points},4).")
        coordinates = event_cloud
        features = self.embedding(event_cloud.transpose(1,2)).transpose(1,2)
        deterministic = not self.training
        for grouper,spatial,aggregation,temporal,residual in zip(
                self.groupers,self.spatial_fft,self.aggregations,
                self.temporal_fft,self.residuals):
            coordinates,features = grouper(
                coordinates,features,deterministic)
            batch_size,groups,neighbors,channels = features.shape
            features = features.reshape(-1,channels,neighbors)
            features = F.gelu(spatial(features)).transpose(1,2)
            attention = aggregation(features)
            features = torch.bmm(
                attention.unsqueeze(1),features).squeeze(1)
            features = features.reshape(batch_size,groups,-1)
            features = F.gelu(temporal(features)).transpose(1,2)
            features = residual(features).transpose(1,2)

        feature_max = F.adaptive_max_pool1d(
            features.transpose(1,2),1).flatten(1)
        feature_mean = features.mean(dim=1)
        return torch.cat((feature_mean,feature_max),dim=1)
