from argparse import Namespace
import importlib.util
from pathlib import Path
import math
import sys
import types

import pytest


class _SubscriptableStub:
    @classmethod
    def __class_getitem__(cls, _item):
        return cls


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_dependency_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_data_stub = types.ModuleType("torch.utils.data")
    torch_f_stub = types.ModuleType("torch.nn.functional")
    torch_stub.Tensor = object
    torch_stub.int64 = object()
    torch_stub.zeros = lambda *_args, **_kwargs: None
    torch_stub.nn = torch_nn_stub
    torch_stub.utils = torch_utils_stub
    torch_utils_stub.data = torch_utils_data_stub
    torch_nn_stub.Module = object
    torch_utils_data_stub.DataLoader = _SubscriptableStub
    torch_utils_data_stub.Dataset = _SubscriptableStub
    torch_utils_data_stub.Sampler = _SubscriptableStub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_f_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_utils_data_stub
    sys.modules["mps_gaf_data_pipeline"] = types.SimpleNamespace(
        MPSGAFDataConfig=object,
        get_test_dataset=lambda *_args, **_kwargs: None,
        get_train_datasets=lambda *_args, **_kwargs: (None, None),
        make_grouped_dataloader=lambda *_args, **_kwargs: None,
    )
    sys.modules["mps_gaf_registration_core"] = types.SimpleNamespace(
        transform_se3=lambda *_args, **_kwargs: None,
    )


def _load_run_registration_benchmark():
    _install_dependency_stubs()
    module_path = Path(__file__).resolve().parents[1] / "run_registration_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_registration_benchmark_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_registration_benchmark = _load_run_registration_benchmark()


def test_validate_relative_paths_rejects_absolute_external_binaries() -> None:
    args = Namespace(
        dataset_path="datasets/modelnet40_ply_hdf5_2048",
        output_dir="runs/test",
        super4pcs_binary=str((Path("external_src") / "Super4PCS").resolve()),
        goicp_binary=None,
        train_category_file=None,
        val_category_file=None,
        test_category_file=None,
    )

    with pytest.raises(ValueError, match="super4pcs_binary"):
        run_registration_benchmark.validate_relative_paths(args)


def test_validate_relative_paths_rejects_absolute_category_file() -> None:
    args = Namespace(
        dataset_path="datasets/modelnet40_ply_hdf5_2048",
        output_dir="runs/test",
        super4pcs_binary=None,
        goicp_binary=None,
        train_category_file=None,
        val_category_file=None,
        test_category_file=str((Path("splits") / "test.txt").resolve()),
    )

    with pytest.raises(ValueError, match="test_category_file"):
        run_registration_benchmark.validate_relative_paths(args)


def test_json_safe_replaces_non_finite_values_for_strict_json() -> None:
    payload = {
        "ok": 1.25,
        "bad": float("inf"),
        "nested": [float("-inf"), float("nan"), {"keep": "value"}],
    }

    cleaned = run_registration_benchmark._json_safe(payload)

    assert cleaned == {
        "ok": 1.25,
        "bad": None,
        "nested": [None, None, {"keep": "value"}],
    }
    assert not any(
        isinstance(value, float) and not math.isfinite(value)
        for value in [cleaned["bad"], cleaned["nested"][0], cleaned["nested"][1]]
    )
