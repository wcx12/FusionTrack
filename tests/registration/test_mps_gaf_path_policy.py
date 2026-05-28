from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
import types

import pytest


def _install_dependency_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_f_stub = types.ModuleType("torch.nn.functional")
    torch_stub.nn = torch_nn_stub
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.no_grad = lambda: (lambda func: func)
    torch_nn_stub.Module = object
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_f_stub
    sys.modules["mps_gaf_data_pipeline"] = types.SimpleNamespace(
        MPSGAFDataConfig=object,
        get_test_dataset=lambda *_args, **_kwargs: None,
        get_train_datasets=lambda *_args, **_kwargs: (None, None),
        make_grouped_dataloader=lambda *_args, **_kwargs: None,
    )
    sys.modules["mps_gaf_registration_core"] = types.SimpleNamespace(
        MPSGAFConfig=object,
        MPSGAFRegistration=object,
        compute_rigid_transform=lambda *_args, **_kwargs: None,
        transform_se3=lambda *_args, **_kwargs: None,
    )


def _load_mps_gaf_run():
    _install_dependency_stubs()
    module_path = Path(__file__).resolve().parents[2] / "code" / "registration" / "mps_gaf_run.py"
    spec = importlib.util.spec_from_file_location("mps_gaf_run_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mps_gaf_run = _load_mps_gaf_run()


def _args(**overrides):
    values = dict(
        dataset_path="datasets/modelnet40_ply_hdf5_2048",
        split_root=None,
        pair_list=None,
        output_dir="runs/mps_gaf",
        checkpoint=None,
        train_category_file=None,
        val_category_file=None,
        test_category_file=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_validate_relative_paths_rejects_absolute_dataset_path() -> None:
    with pytest.raises(ValueError, match="dataset_path"):
        mps_gaf_run.validate_relative_paths(_args(dataset_path="/tmp/modelnet"))


def test_validate_relative_paths_rejects_absolute_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        mps_gaf_run.validate_relative_paths(_args(checkpoint="/tmp/best.pt"))
