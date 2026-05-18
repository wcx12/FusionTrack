"""Dataset and augmentation pipeline for MPS-GAF registration.

This is a cleaned extraction of the ModelNet40 loading path used by the
registration experiments.  It keeps only the pieces required to build grouped
multi-source batches:

* load ModelNet40 HDF5 point clouds with normals;
* repeat each reference shape into ``num_sources_per_ref`` source variants;
* apply deterministic reference augmentation and source-specific augmentation;
* batch complete reference groups together so MPS-GAF can fuse sources safely.

Datasets are intentionally not bundled in this repository.  Put the processed
``modelnet40_ply_hdf5_2048`` directory outside the repository and pass its path
through ``MPSGAFDataConfig.dataset_path``.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass
class MPSGAFDataConfig:
    dataset_path: str
    num_points: int = 1024
    noise_type: str = "crop"
    rot_mag: float = 45.0
    trans_mag: float = 0.5
    partial: Sequence[float] = (0.7, 0.7)
    num_sources_per_ref: int = 10
    train_category_file: Optional[str] = None
    val_category_file: Optional[str] = None
    test_category_file: Optional[str] = None
    seed: int = 0


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Dict], Dict]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def _read_categories(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        categories = [line.strip() for line in handle if line.strip()]
    return sorted(categories)


def _seed_from(sample: Dict, key: str, salt: int = 0) -> Optional[int]:
    if key in sample:
        base = int(sample[key])
    elif sample.get("deterministic", False):
        base = int(sample["idx"])
    else:
        return None
    return (base * 1_000_003 + salt) % (2**32 - 1)


def _rng(sample: Dict, key: str, salt: int = 0) -> np.random.RandomState:
    seed = _seed_from(sample, key, salt=salt)
    if seed is None:
        return np.random.RandomState()
    return np.random.RandomState(seed)


def _uniform_sphere(rng: np.random.RandomState) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * np.pi)
    cos_theta = rng.uniform(-1.0, 1.0)
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
    return np.array([sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta], dtype=np.float32)


def _se3_inverse(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.concatenate([rotation.T, rotation.T @ -translation[:, None]], axis=1).astype(np.float32)


def _apply_se3(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    if points.shape[1] == 6:
        normals = points[:, 3:6] @ transform[:3, :3].T
        return np.concatenate([xyz, normals], axis=-1).astype(np.float32)
    return xyz.astype(np.float32)


class SplitSourceRef:
    """Clone one raw point cloud into source and reference point clouds."""

    def __call__(self, sample: Dict) -> Dict:
        sample["points_raw"] = sample.pop("points")
        sample["points_src"] = sample["points_raw"].copy()
        sample["points_ref"] = sample["points_raw"].copy()
        return sample


class Resampler:
    def __init__(self, num: int) -> None:
        self.num = int(num)

    def __call__(self, sample: Dict) -> Dict:
        if "points" in sample:
            sample["points"] = self._resample(sample["points"], self.num, _rng(sample, "seed_src", salt=11))
            return sample

        if "crop_proportion" not in sample:
            src_size, ref_size = self.num, self.num
        elif len(sample["crop_proportion"]) == 1:
            src_size = math.ceil(float(sample["crop_proportion"][0]) * self.num)
            ref_size = self.num
        elif len(sample["crop_proportion"]) == 2:
            src_size = math.ceil(float(sample["crop_proportion"][0]) * self.num)
            ref_size = math.ceil(float(sample["crop_proportion"][1]) * self.num)
        else:
            raise ValueError("crop_proportion must have one or two elements")

        sample["points_src"] = self._resample(sample["points_src"], src_size, _rng(sample, "seed_src", salt=13))
        sample["points_ref"] = self._resample(sample["points_ref"], ref_size, _rng(sample, "seed_ref", salt=17))
        return sample

    @staticmethod
    def _resample(points: np.ndarray, num_points: int, rng: np.random.RandomState) -> np.ndarray:
        if num_points <= points.shape[0]:
            indices = rng.choice(points.shape[0], num_points, replace=False)
        else:
            indices = np.concatenate(
                [
                    rng.choice(points.shape[0], points.shape[0], replace=False),
                    rng.choice(points.shape[0], num_points - points.shape[0], replace=True),
                ],
                axis=0,
            )
        return points[indices, :]


class FixedResampler(Resampler):
    @staticmethod
    def _resample(points: np.ndarray, num_points: int, rng: np.random.RandomState) -> np.ndarray:
        del rng
        multiple = num_points // points.shape[0]
        remainder = num_points % points.shape[0]
        return np.concatenate((np.tile(points, (multiple, 1)), points[:remainder, :]), axis=0)


class RandomCrop:
    def __init__(self, p_keep: Sequence[float] = (0.7, 0.7)) -> None:
        self.p_keep = np.array(p_keep, dtype=np.float32)

    def __call__(self, sample: Dict) -> Dict:
        sample["crop_proportion"] = self.p_keep
        if np.all(self.p_keep == 1.0):
            return sample

        if len(self.p_keep) == 1:
            sample["points_src"] = self._crop(sample["points_src"], float(self.p_keep[0]), _rng(sample, "seed_src", 23))
            return sample

        sample["points_ref"] = self._crop(sample["points_ref"], float(self.p_keep[1]), _rng(sample, "seed_ref", 29))
        sample["points_src"] = self._crop(sample["points_src"], float(self.p_keep[0]), _rng(sample, "seed_src", 31))
        return sample

    @staticmethod
    def _crop(points: np.ndarray, p_keep: float, rng: np.random.RandomState) -> np.ndarray:
        direction = _uniform_sphere(rng)
        centered = points[:, :3] - np.mean(points[:, :3], axis=0)
        distance = np.dot(centered, direction)
        if p_keep == 0.5:
            mask = distance > 0
        else:
            mask = distance > np.percentile(distance, (1.0 - p_keep) * 100.0)
        return points[mask, :]


class RandomTransformSE3Euler:
    """Apply a source-specific SE(3) perturbation and store source-to-ref GT."""

    def __init__(self, rot_mag: float = 45.0, trans_mag: float = 0.5, random_mag: bool = False) -> None:
        self.rot_mag = float(rot_mag)
        self.trans_mag = float(trans_mag)
        self.random_mag = bool(random_mag)

    def __call__(self, sample: Dict) -> Dict:
        if "points" in sample:
            transform = self._generate(_rng(sample, "seed_src", 37))
            sample["points"] = _apply_se3(transform, sample["points"])
            return sample

        transform = self._generate(_rng(sample, "seed_src", 41))
        sample["points_src"] = _apply_se3(transform, sample["points_src"])
        sample["transform_gt"] = _se3_inverse(transform)
        return sample

    def _generate(self, rng: np.random.RandomState) -> np.ndarray:
        if self.random_mag:
            attenuation = rng.random_sample()
            rot_mag = attenuation * self.rot_mag
            trans_mag = attenuation * self.trans_mag
        else:
            rot_mag = self.rot_mag
            trans_mag = self.trans_mag

        angles = rng.uniform(0.0, np.pi * rot_mag / 180.0, size=3)
        rotation = Rotation.from_euler("xyz", angles).as_matrix()
        translation = rng.uniform(-trans_mag, trans_mag, size=3)
        return np.concatenate([rotation, translation[:, None]], axis=1).astype(np.float32)


class RandomJitter:
    def __init__(self, scale: float = 0.01, clip: float = 0.05) -> None:
        self.scale = float(scale)
        self.clip = float(clip)

    def __call__(self, sample: Dict) -> Dict:
        if "points" in sample:
            sample["points"] = self._jitter(sample["points"], _rng(sample, "seed_src", 43))
            return sample

        sample["points_src"] = self._jitter(sample["points_src"], _rng(sample, "seed_src", 47))
        sample["points_ref"] = self._jitter(sample["points_ref"], _rng(sample, "seed_ref", 53))
        return sample

    def _jitter(self, points: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        out = points.copy()
        noise = rng.normal(0.0, self.scale, size=(out.shape[0], 3))
        out[:, :3] += np.clip(noise, a_min=-self.clip, a_max=self.clip)
        return out


class ShufflePoints:
    def __call__(self, sample: Dict) -> Dict:
        if "points" in sample:
            sample["points"] = self._shuffle(sample["points"], _rng(sample, "seed_src", 59))
            return sample

        sample["points_src"] = self._shuffle(sample["points_src"], _rng(sample, "seed_src", 61))
        sample["points_ref"] = self._shuffle(sample["points_ref"], _rng(sample, "seed_ref", 67))
        return sample

    @staticmethod
    def _shuffle(points: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        return points[rng.permutation(points.shape[0]), :]


class SetDeterministic:
    def __call__(self, sample: Dict) -> Dict:
        sample["deterministic"] = True
        return sample


class RepeatPerReferenceDataset(Dataset):
    """Expand each raw reference into several source variants."""

    def __init__(
        self,
        base_dataset: Dataset,
        transforms: Optional[Callable[[Dict], Dict]],
        num_sources: int,
        dynamic_epoch: bool = False,
        base_seed: int = 0,
    ) -> None:
        self.base = base_dataset
        self.transforms = transforms
        self.num_sources = int(num_sources)
        self.dynamic_epoch = bool(dynamic_epoch)
        self.base_seed = int(base_seed)
        self.epoch = 0
        if self.num_sources < 1:
            raise ValueError("num_sources must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.base) * self.num_sources

    def __getitem__(self, item: int) -> Dict:
        ref_idx = item // self.num_sources
        source_idx = item % self.num_sources

        sample = self.base.get_raw(ref_idx) if hasattr(self.base, "get_raw") else self.base[ref_idx]
        sample = copy.deepcopy(sample)
        base_idx = int(sample["idx"])

        epoch_value = self.epoch if self.dynamic_epoch else 0
        epoch_seed = self.base_seed + epoch_value * 1_000_003

        sample["seed_ref"] = epoch_seed + ref_idx
        sample["seed_src"] = epoch_seed + len(self.base) + ref_idx * self.num_sources + source_idx
        sample["group_ref_idx"] = ref_idx
        sample["multi_src_k"] = source_idx
        sample["idx"] = base_idx * self.num_sources + source_idx

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class ModelNetHdf(Dataset):
    def __init__(
        self,
        dataset_path: str,
        subset: str = "train",
        categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = dataset_path
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"ModelNet40 directory not found: {self.root}. "
                "Download modelnet40_ply_hdf5_2048 separately and pass dataset_path."
            )

        with open(os.path.join(self.root, "shape_names.txt"), "r", encoding="utf-8") as handle:
            self.classes = [line.strip() for line in handle if line.strip()]

        category_to_idx = {category: idx for idx, category in enumerate(self.classes)}
        category_indices = None
        if categories is not None:
            category_indices = [category_to_idx[category] for category in categories]
            self.classes = list(categories)

        h5_files = self._metadata_files(subset)
        self.data, self.labels = self._read_h5_files(h5_files, category_indices)

    def _metadata_files(self, subset: str) -> List[str]:
        metadata_path = os.path.join(self.root, f"{subset}_files.txt")
        with open(metadata_path, "r", encoding="utf-8") as handle:
            h5_files = [line.strip() for line in handle if line.strip()]
        h5_files = [path.replace("data/modelnet40_ply_hdf5_2048/", "") for path in h5_files]
        return [os.path.join(self.root, path) for path in h5_files]

    @staticmethod
    def _read_h5_files(filenames: Iterable[str], categories: Optional[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        all_data = []
        all_labels = []
        for filename in filenames:
            with h5py.File(filename, mode="r") as handle:
                data = np.concatenate([handle["data"][:], handle["normal"][:]], axis=-1)
                labels = handle["label"][:].flatten().astype(np.int64)

            if categories is not None:
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data.astype(np.float32))
            all_labels.append(labels)

        return np.concatenate(all_data, axis=0), np.concatenate(all_labels, axis=0)

    def get_raw(self, item: int) -> Dict:
        return {
            "points": self.data[item, :, :].copy(),
            "label": self.labels[item],
            "idx": int(item),
        }

    def __getitem__(self, item: int) -> Dict:
        return self.get_raw(item)

    def __len__(self) -> int:
        return self.data.shape[0]


class GroupedBatchSampler(Sampler[List[int]]):
    """Yield full multi-source groups and avoid mixing reference groups."""

    def __init__(
        self,
        dataset: RepeatPerReferenceDataset,
        groups_per_batch: int,
        shuffle_groups: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.groups_per_batch = int(groups_per_batch)
        self.shuffle_groups = bool(shuffle_groups)
        self.seed = seed
        self.epoch = 0
        if self.groups_per_batch < 1:
            raise ValueError("groups_per_batch must be positive")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        group_indices = np.arange(len(self.dataset.base))
        if self.shuffle_groups:
            seed = None if self.seed is None else int(self.seed) + self.epoch
            rng = np.random.RandomState(seed)
            rng.shuffle(group_indices)

        for start in range(0, len(group_indices), self.groups_per_batch):
            groups = group_indices[start : start + self.groups_per_batch]
            batch = []
            for group_idx in groups:
                base = int(group_idx) * self.dataset.num_sources
                batch.extend(range(base, base + self.dataset.num_sources))
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.dataset.base) / self.groups_per_batch)


def get_transforms(
    noise_type: str,
    rot_mag: float = 45.0,
    trans_mag: float = 0.5,
    num_points: int = 1024,
    partial_p_keep: Sequence[float] = (0.7, 0.7),
) -> tuple[Compose, Compose]:
    if noise_type == "clean":
        train_transforms = [
            SplitSourceRef(),
            RandomTransformSE3Euler(rot_mag=rot_mag, trans_mag=trans_mag),
            Resampler(num_points),
            ShufflePoints(),
        ]
        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(),
            RandomTransformSE3Euler(rot_mag=rot_mag, trans_mag=trans_mag),
            FixedResampler(num_points),
            ShufflePoints(),
        ]
    elif noise_type == "jitter":
        train_transforms = [
            SplitSourceRef(),
            RandomTransformSE3Euler(rot_mag=rot_mag, trans_mag=trans_mag),
            Resampler(num_points),
            RandomJitter(),
            ShufflePoints(),
        ]
        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(),
            RandomTransformSE3Euler(rot_mag=rot_mag, trans_mag=trans_mag),
            Resampler(num_points),
            RandomJitter(),
            ShufflePoints(),
        ]
    elif noise_type == "crop":
        train_transforms = [
            SplitSourceRef(),
            RandomCrop(partial_p_keep),
            RandomTransformSE3Euler(rot_mag=rot_mag, trans_mag=trans_mag),
            Resampler(num_points),
            RandomJitter(),
            ShufflePoints(),
        ]
        test_transforms = [
            SetDeterministic(),
            SplitSourceRef(),
            RandomCrop(partial_p_keep),
            RandomTransformSE3Euler(rot_mag=rot_mag, trans_mag=trans_mag),
            Resampler(num_points),
            RandomJitter(),
            ShufflePoints(),
        ]
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    return Compose(train_transforms), Compose(test_transforms)


def get_train_datasets(config: MPSGAFDataConfig) -> tuple[RepeatPerReferenceDataset, RepeatPerReferenceDataset]:
    train_transforms, val_transforms = get_transforms(
        config.noise_type,
        config.rot_mag,
        config.trans_mag,
        config.num_points,
        config.partial,
    )
    train_base = ModelNetHdf(
        config.dataset_path,
        subset="train",
        categories=_read_categories(config.train_category_file),
    )
    val_base = ModelNetHdf(
        config.dataset_path,
        subset="test",
        categories=_read_categories(config.val_category_file),
    )
    return (
        RepeatPerReferenceDataset(
            train_base,
            train_transforms,
            config.num_sources_per_ref,
            dynamic_epoch=True,
            base_seed=config.seed,
        ),
        RepeatPerReferenceDataset(
            val_base,
            val_transforms,
            config.num_sources_per_ref,
            dynamic_epoch=False,
            base_seed=config.seed,
        ),
    )


def get_test_dataset(config: MPSGAFDataConfig) -> RepeatPerReferenceDataset:
    _, test_transforms = get_transforms(
        config.noise_type,
        config.rot_mag,
        config.trans_mag,
        config.num_points,
        config.partial,
    )
    test_base = ModelNetHdf(
        config.dataset_path,
        subset="test",
        categories=_read_categories(config.test_category_file),
    )
    return RepeatPerReferenceDataset(
        test_base,
        test_transforms,
        config.num_sources_per_ref,
        dynamic_epoch=False,
        base_seed=config.seed,
    )


def make_grouped_dataloader(
    dataset: RepeatPerReferenceDataset,
    groups_per_batch: int,
    shuffle_groups: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    sampler = GroupedBatchSampler(
        dataset,
        groups_per_batch=groups_per_batch,
        shuffle_groups=shuffle_groups,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


__all__ = [
    "MPSGAFDataConfig",
    "ModelNetHdf",
    "RepeatPerReferenceDataset",
    "GroupedBatchSampler",
    "get_train_datasets",
    "get_test_dataset",
    "make_grouped_dataloader",
]
