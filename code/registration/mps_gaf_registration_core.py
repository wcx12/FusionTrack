"""Core implementation of MPS-GAF point-set registration.

This file extracts the method-specific registration code from the local
experimental project and leaves out datasets, training loops, baselines,
figures, debug scripts, and result files.  It contains the modules described
in the manuscript:

* point-wise PPF/DXYZ/XYZ feature encoding;
* intra-set self-graph correlation;
* cross-set message passing;
* reliability-aware gated adaptive fusion;
* Sinkhorn soft assignment and weighted SVD rigid transform estimation.

The model expects a mini-batch grouped as
``groups_per_batch * num_sources`` rows, where rows belonging to the same
group share the same target point set and contain different source variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-5


@dataclass
class MPSGAFConfig:
    """Minimal configuration needed by the core registration model."""

    features: Sequence[str] = ("ppf", "dxyz", "xyz")
    feat_dim: int = 96
    radius: float = 0.3
    num_neighbors: int = 64
    num_sources: int = 10
    num_sk_iter: int = 5
    no_slack: bool = False


def _cfg(config: Optional[Any], name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def transform_se3(
    transform: torch.Tensor,
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Apply a batched SE(3) transform stored as a B x 3 x 4 matrix."""

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]

    if len(transform.size()) != len(points.size()):
        raise ValueError("transform and points must both be batched")

    transformed_points = torch.matmul(points, rotation.transpose(-1, -2)) + translation[..., None, :]

    if normals is None:
        return transformed_points

    transformed_normals = normals @ rotation.transpose(-1, -2)
    return transformed_points, transformed_normals


def angle_difference(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise angular distance between normalized features."""

    cosine = torch.matmul(src, dst.permute(0, 2, 1))
    cosine = cosine.clamp(min=-1.0 + EPS, max=1.0 - EPS)
    return torch.acos(cosine)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Calculate pairwise squared Euclidean distance."""

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points using batched point indices."""

    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(batch_size, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest-point sampling."""

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    itself_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Find local neighborhoods inside a ball of the requested radius."""

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, num_centers, _ = new_xyz.shape
    group_idx = (
        torch.arange(num_points, dtype=torch.long, device=device)
        .view(1, 1, num_points)
        .repeat([batch_size, num_centers, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)

    if itself_indices is not None:
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[:, None].repeat(1, num_centers)
        row_indices = torch.arange(num_centers, dtype=torch.long, device=device)[None, :].repeat(batch_size, 1)
        group_idx[batch_indices, row_indices, itself_indices] = num_points

    group_idx[sqrdists > radius ** 2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(batch_size, num_centers, 1).repeat([1, 1, nsample])

    mask = group_idx == num_points
    group_idx[mask] = group_first[mask]
    return group_idx


def vector_angle(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Compute a stable angle between two vector tensors."""

    cross_prod = torch.stack(
        [
            v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
            v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
            v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
        ],
        dim=-1,
    )
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)
    return torch.atan2(cross_prod_norm, dot_prod)


def sample_and_group_multi(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    normals: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Sample local neighborhoods and compute XYZ, DXYZ, and PPF features."""

    batch_size, num_points, channels = xyz.shape

    if npoint > 0:
        num_centers = npoint
        fps_idx = farthest_point_sample(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        reference_normals = index_points(normals, fps_idx)[:, :, None, :]
    else:
        num_centers = num_points
        fps_idx = torch.arange(0, num_points, device=xyz.device)[None, ...].repeat(batch_size, 1)
        new_xyz = xyz
        reference_normals = normals[:, :, None, :]

    idx = query_ball_point(radius, nsample, xyz, new_xyz, fps_idx)
    grouped_xyz = index_points(xyz, idx)
    dxyz = grouped_xyz - new_xyz.view(batch_size, num_centers, 1, channels)
    neighbor_normals = index_points(normals, idx)

    nr_d = vector_angle(reference_normals, dxyz)
    ni_d = vector_angle(neighbor_normals, dxyz)
    nr_ni = vector_angle(reference_normals, neighbor_normals)
    d_norm = torch.norm(dxyz, dim=-1)
    ppf = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)

    return {"xyz": new_xyz, "dxyz": dxyz, "ppf": ppf}


RAW_FEATURE_SIZES = {"xyz": 3, "dxyz": 3, "ppf": 4}
RAW_FEATURE_ORDER = {"xyz": 0, "dxyz": 1, "ppf": 2}


def match_features(
    feat_src: torch.Tensor,
    feat_ref: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
    """Compute pairwise feature distance between source and target points."""

    if feat_src.shape[-1] != feat_ref.shape[-1]:
        raise ValueError("source and target feature dimensions must match")

    if metric == "l2":
        return square_distance(feat_src, feat_ref)
    if metric == "angle":
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + EPS)
        return angle_difference(feat_src_norm, feat_ref_norm)

    raise NotImplementedError(f"Unknown feature metric: {metric}")


class ParameterPredictionNet(nn.Module):
    """RPMNet-style network that predicts affinity scale and offset."""

    def __init__(self) -> None:
        super().__init__()
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        src_padded = F.pad(x[0], (0, 1), mode="constant", value=0)
        ref_padded = F.pad(x[1], (0, 1), mode="constant", value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha


def get_prepool(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )


def get_postpool(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )


class PointFeatureEncoder(nn.Module):
    """Point-wise encoder for local geometry and PPF features."""

    def __init__(
        self,
        features: Sequence[str],
        feature_dim: int,
        radius: float,
        num_neighbors: int,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.n_sample = num_neighbors
        self.features = sorted(features, key=lambda f: RAW_FEATURE_ORDER[f])

        raw_dim = int(np.sum([RAW_FEATURE_SIZES[f] for f in self.features]))
        self.prepool = get_prepool(raw_dim, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features["xyz"] = features["xyz"][:, :, None, :]

        concat = []
        for feature_name in self.features:
            expanded = features[feature_name].expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        new_feat = fused_input_feat.permute(0, 3, 2, 1)
        new_feat = self.prepool(new_feat)
        pooled_feat = torch.max(new_feat, 2)[0]
        post_feat = self.postpool(pooled_feat)
        cluster_feat = post_feat.permute(0, 2, 1)
        return cluster_feat / (torch.norm(cluster_feat, dim=-1, keepdim=True) + EPS)


def sinkhorn(
    log_alpha: torch.Tensor,
    n_iters: int = 5,
    slack: bool = True,
    eps: float = -1,
) -> torch.Tensor:
    """Run log-domain Sinkhorn normalization for soft correspondence."""

    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :]).squeeze(dim=1)

        for _ in range(n_iters):
            log_alpha_padded = torch.cat(
                (
                    log_alpha_padded[:, :-1, :]
                    - torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True),
                    log_alpha_padded[:, -1, None, :],
                ),
                dim=1,
            )
            log_alpha_padded = torch.cat(
                (
                    log_alpha_padded[:, :, :-1]
                    - torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True),
                    log_alpha_padded[:, :, -1, None],
                ),
                dim=2,
            )

            if eps > 0:
                cur_alpha = torch.exp(log_alpha_padded[:, :-1, :-1])
                if prev_alpha is not None:
                    abs_dev = torch.abs(cur_alpha - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = cur_alpha.clone()

        return log_alpha_padded[:, :-1, :-1]

    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

        if eps > 0:
            cur_alpha = torch.exp(log_alpha)
            if prev_alpha is not None:
                abs_dev = torch.abs(cur_alpha - prev_alpha)
                if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                    break
            prev_alpha = cur_alpha.clone()

    return log_alpha


def compute_rigid_transform(
    src_points: torch.Tensor,
    target_points: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Estimate a weighted rigid transform from source to target points."""

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + EPS)
    centroid_src = torch.sum(src_points * weights_normalized, dim=1)
    centroid_tgt = torch.sum(target_points * weights_normalized, dim=1)
    src_centered = src_points - centroid_src[:, None, :]
    tgt_centered = target_points - centroid_tgt[:, None, :]
    covariance = src_centered.transpose(-2, -1) @ (tgt_centered * weights_normalized)

    u, _, v = torch.svd(covariance, some=False, compute_uv=True)
    rot_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_neg = v_neg @ u.transpose(-1, -2)
    rotation = torch.where(torch.det(rot_pos)[:, None, None] > 0, rot_pos, rot_neg)
    translation = -rotation @ centroid_src[:, :, None] + centroid_tgt[:, :, None]
    return torch.cat((rotation, translation), dim=2)


class SelfGraphCorrelation(nn.Module):
    """Intra-set graph correlation from the manuscript."""

    def __init__(
        self,
        c_in: int,
        c_out: Optional[int] = None,
        proj_dim: Optional[int] = None,
        tau: float = 1.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        c_out = c_out or c_in
        proj_dim = proj_dim or c_in
        self.tau = tau
        self.proj = nn.Linear(c_in, proj_dim, bias=True)
        self.f_adj = nn.Linear(c_in, c_out, bias=True)
        self.f_self = nn.Linear(c_in, c_out, bias=True)
        self.use_residual = use_residual and (c_in == c_out)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        projected = self.proj(features)
        similarity = torch.matmul(projected, projected.transpose(-1, -2))
        if self.tau != 1.0:
            similarity = similarity / self.tau

        if mask is not None:
            mask_col = mask.unsqueeze(-2).bool()
            similarity = similarity.masked_fill(~mask_col, float("-inf"))

        adjacency = F.softmax(similarity, dim=-1)
        out = torch.matmul(adjacency, self.f_adj(features)) + self.f_self(features)

        if self.use_residual:
            out = out + features

        return self.norm(out)


class CrossGraphCorrelation(nn.Module):
    """Cross-set message passing between two point sets."""

    def __init__(
        self,
        c_in: int,
        c_out: Optional[int] = None,
        proj_dim: Optional[int] = None,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        c_out = c_out or c_in
        proj_dim = proj_dim or c_in
        self.tau = tau
        self.source_proj = nn.Linear(c_in, proj_dim, bias=True)
        self.target_proj = nn.Linear(c_in, proj_dim, bias=True)
        self.f_adj = nn.Linear(c_in, c_out, bias=True)
        self.f_self = nn.Linear(c_in, c_out, bias=True)
        self.norm = nn.LayerNorm(c_out)

    def forward(
        self,
        query_features: torch.Tensor,
        context_features: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_context = context_features.shape[1]
        query_proj = self.source_proj(query_features)
        context_proj = self.target_proj(context_features)
        similarity = query_proj @ context_proj.transpose(-1, -2)
        if self.tau != 1.0:
            similarity = similarity / self.tau

        if k is not None:
            k = min(k, num_context)
            _, top_indices = torch.topk(similarity, k, dim=-1)
            keep = torch.zeros_like(similarity, dtype=torch.bool).scatter_(-1, top_indices, True)
            similarity = similarity.masked_fill(~keep, float("-inf"))

        weights = F.softmax(similarity, dim=-1)
        message = weights @ self.f_adj(context_features)
        out = self.norm(message + self.f_self(query_features))
        return out, weights


def row_entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num_context = weights.size(-1)
    log_weights = torch.log(torch.clamp(weights, min=eps))
    entropy = -(weights * log_weights).sum(dim=-1)
    return entropy / (torch.log(torch.tensor(float(num_context), device=weights.device)) + eps)


def bidirectional_consistency(
    forward_weights: torch.Tensor,
    backward_weights: torch.Tensor,
) -> torch.Tensor:
    return (forward_weights * backward_weights.transpose(1, 2)).sum(dim=-1)


class GateMLP(nn.Module):
    """Point-wise reliability gate for one incoming message path."""

    def __init__(self, in_dim: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GatedAdaptiveFusion(nn.Module):
    """Reliability-aware multi-source fusion for sources and shared target."""

    def __init__(self, c_in: int, k: Optional[int] = None, tau: float = 1.0) -> None:
        super().__init__()
        self.k = k
        self.cross = CrossGraphCorrelation(c_in, c_in, proj_dim=c_in, tau=tau)
        self.gate_mlp = GateMLP(in_dim=4)
        self.post = nn.Linear(c_in, c_in)

    def _pair_stats(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        forward_weights: torch.Tensor,
        backward_weights: torch.Tensor,
    ) -> torch.Tensor:
        sharpness = forward_weights.amax(dim=-1)
        confidence = 1.0 - row_entropy(forward_weights)
        consistency = bidirectional_consistency(forward_weights, backward_weights)

        query_mean = F.normalize(query.mean(dim=1), dim=-1)
        context_mean = F.normalize(context.mean(dim=1), dim=-1)
        cosine = (query_mean * context_mean).sum(dim=-1, keepdim=True)
        cosine = cosine.unsqueeze(1).expand(-1, query.shape[1], -1)

        return torch.cat(
            [torch.stack([sharpness, confidence, consistency], dim=-1), cosine],
            dim=-1,
        )

    @staticmethod
    def _competitive_fuse(
        messages: Sequence[torch.Tensor],
        logits: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked_logits = torch.stack(list(logits), dim=1)
        weights = F.softmax(stacked_logits, dim=1)
        stacked_messages = torch.stack(list(messages), dim=1)
        fused = (weights * stacked_messages).sum(dim=1)
        return fused, weights.squeeze(-1)

    def fuse_group(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse one target and a group of source feature sets.

        Args:
            source_features: B_group x N_sources x N_src x C.
            target_features: B_group x N_ref x C.

        Returns:
            Enhanced source features and enhanced target features.
        """

        _, num_sources, _, _ = source_features.shape

        target_messages = []
        target_logits = []
        for source_idx in range(num_sources):
            msg_ts, weights_ts = self.cross(target_features, source_features[:, source_idx], k=self.k)
            _, weights_st = self.cross(source_features[:, source_idx], target_features, k=self.k)
            stats = self._pair_stats(target_features, source_features[:, source_idx], weights_ts, weights_st)
            target_messages.append(msg_ts)
            target_logits.append(self.gate_mlp(stats))

        target_agg, _ = self._competitive_fuse(target_messages, target_logits)
        target_enhanced = self.post(target_agg + target_features)

        enhanced_sources = []
        for source_idx in range(num_sources):
            messages = []
            logits = []

            msg_st, weights_st = self.cross(source_features[:, source_idx], target_enhanced, k=self.k)
            _, weights_ts = self.cross(target_enhanced, source_features[:, source_idx], k=self.k)
            stats = self._pair_stats(source_features[:, source_idx], target_enhanced, weights_st, weights_ts)
            messages.append(msg_st)
            logits.append(self.gate_mlp(stats))

            for other_idx in range(num_sources):
                if other_idx == source_idx:
                    continue
                msg_ss, weights_ss = self.cross(
                    source_features[:, source_idx],
                    source_features[:, other_idx],
                    k=self.k,
                )
                _, weights_ss_back = self.cross(
                    source_features[:, other_idx],
                    source_features[:, source_idx],
                    k=self.k,
                )
                stats = self._pair_stats(
                    source_features[:, source_idx],
                    source_features[:, other_idx],
                    weights_ss,
                    weights_ss_back,
                )
                messages.append(msg_ss)
                logits.append(self.gate_mlp(stats))

            source_agg, _ = self._competitive_fuse(messages, logits)
            enhanced_sources.append(self.post(source_agg + source_features[:, source_idx]))

        return torch.stack(enhanced_sources, dim=1), target_enhanced


class MPSGAFRegistration(nn.Module):
    """Multi-Point-Set Gated Adaptive Fusion registration model."""

    def __init__(self, config: Optional[Any] = None) -> None:
        super().__init__()

        default_config = MPSGAFConfig()
        features = tuple(_cfg(config, "features", default_config.features))
        feat_dim = int(_cfg(config, "feat_dim", default_config.feat_dim))
        radius = float(_cfg(config, "radius", default_config.radius))
        num_neighbors = int(_cfg(config, "num_neighbors", default_config.num_neighbors))
        num_sk_iter = int(_cfg(config, "num_sk_iter", default_config.num_sk_iter))

        fallback_sources = _cfg(config, "num_sources_per_ref", default_config.num_sources)
        num_sources = int(_cfg(config, "num_sources", fallback_sources))
        if num_sources < 1:
            raise ValueError("num_sources must be positive")

        self.num_sources = num_sources
        self.add_slack = not bool(_cfg(config, "no_slack", default_config.no_slack))
        self.num_sk_iter = num_sk_iter

        self.weights_net = ParameterPredictionNet()
        self.feat_extractor = PointFeatureEncoder(
            features=features,
            feature_dim=feat_dim,
            radius=radius,
            num_neighbors=num_neighbors,
        )
        self.self_corr = SelfGraphCorrelation(
            c_in=feat_dim,
            c_out=feat_dim,
            proj_dim=feat_dim,
            tau=1.0,
            use_residual=True,
        )
        self.group_fusion = GatedAdaptiveFusion(c_in=feat_dim)
        self.feature_fuse = nn.Linear(in_features=2 * feat_dim, out_features=feat_dim)

    @staticmethod
    def compute_affinity(
        beta: torch.Tensor,
        feat_distance: torch.Tensor,
        alpha: torch.Tensor | float = 0.5,
    ) -> torch.Tensor:
        if isinstance(alpha, float):
            return -beta[:, None, None] * (feat_distance - alpha)
        return -beta[:, None, None] * (feat_distance - alpha[:, None, None])

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        num_iter: int = 1,
    ) -> Tuple[Sequence[torch.Tensor], Dict[str, Any]]:
        """Estimate per-source rigid transforms against the shared target."""

        xyz_ref, norm_ref = data["points_ref"][:, :, :3], data["points_ref"][:, :, 3:6]
        xyz_src, norm_src = data["points_src"][:, :, :3], data["points_src"][:, :, 3:6]
        xyz_src_t, norm_src_t = xyz_src, norm_src

        transforms = []
        all_gamma = []
        all_perm_matrices = []
        all_weighted_ref = []
        all_beta = []
        all_alpha = []

        for _ in range(num_iter):
            beta, alpha = self.weights_net([xyz_src_t, xyz_ref])

            feat_src = self.feat_extractor(xyz_src_t, norm_src_t)
            feat_ref = self.feat_extractor(xyz_ref, norm_ref)

            feat_src_self = self.self_corr(feat_src)
            feat_ref_self = self.self_corr(feat_ref)

            batch_size = feat_src_self.shape[0]
            if batch_size % self.num_sources != 0:
                raise ValueError(
                    "Batch size must be a multiple of num_sources. "
                    f"Got batch_size={batch_size}, num_sources={self.num_sources}."
                )

            groups = batch_size // self.num_sources
            feat_src_grouped = feat_src_self.reshape(groups, self.num_sources, *feat_src_self.shape[1:])
            feat_ref_grouped = feat_ref_self.reshape(groups, self.num_sources, *feat_ref_self.shape[1:])
            shared_ref = feat_ref_grouped[:, 0]

            src_cross_grouped, ref_cross = self.group_fusion.fuse_group(feat_src_grouped, shared_ref)
            ref_cross_repeated = ref_cross.unsqueeze(1).expand(-1, self.num_sources, -1, -1)
            feat_src_cross = src_cross_grouped.reshape(batch_size, *src_cross_grouped.shape[2:])
            feat_ref_cross = ref_cross_repeated.reshape(batch_size, *ref_cross_repeated.shape[2:])

            feat_src_final = self.feature_fuse(torch.cat((feat_src_self, feat_src_cross), dim=-1))
            feat_ref_final = self.feature_fuse(torch.cat((feat_ref_self, feat_ref_cross), dim=-1))

            feat_distance = match_features(feat_src_final, feat_ref_final)
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            perm_matrix = torch.exp(log_perm_matrix)
            weighted_ref = perm_matrix @ xyz_ref / (torch.sum(perm_matrix, dim=2, keepdim=True) + EPS)

            transform = compute_rigid_transform(xyz_src, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            xyz_src_t, norm_src_t = transform_se3(transform.detach(), xyz_src, norm_src)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            all_beta.append(_to_numpy(beta))
            all_alpha.append(_to_numpy(alpha))

        endpoints = {
            "perm_matrices_init": all_gamma,
            "perm_matrices": all_perm_matrices,
            "weighted_ref": all_weighted_ref,
            "beta": np.stack(all_beta, axis=0),
            "alpha": np.stack(all_alpha, axis=0),
        }
        return transforms, endpoints


# Backward-compatible alias for scripts that still import the old class name.
MPR_model = MPSGAFRegistration


__all__ = [
    "MPSGAFConfig",
    "MPSGAFRegistration",
    "MPR_model",
    "PointFeatureEncoder",
    "SelfGraphCorrelation",
    "CrossGraphCorrelation",
    "GatedAdaptiveFusion",
    "sinkhorn",
    "compute_rigid_transform",
    "match_features",
]
