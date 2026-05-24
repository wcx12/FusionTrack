from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
import types

import pytest


class _SubscriptableStub:
    @classmethod
    def __class_getitem__(cls, _item):
        return cls


def _install_dependency_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_optim_stub = types.ModuleType("torch.optim")
    torch_utils_stub = types.ModuleType("torch.utils")
    torch_utils_data_stub = types.ModuleType("torch.utils.data")
    torch_f_stub = types.ModuleType("torch.nn.functional")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.no_grad = lambda: (lambda func: func)
    torch_stub.nn = torch_nn_stub
    torch_stub.optim = torch_optim_stub
    torch_stub.utils = torch_utils_stub
    torch_utils_stub.data = torch_utils_data_stub
    torch_nn_stub.Module = object
    torch_optim_stub.Optimizer = object
    torch_utils_data_stub.DataLoader = _SubscriptableStub
    torch_utils_data_stub.Dataset = _SubscriptableStub
    torch_utils_data_stub.Sampler = _SubscriptableStub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.optim"] = torch_optim_stub
    sys.modules["torch.utils"] = torch_utils_stub
    sys.modules["torch.utils.data"] = torch_utils_data_stub
    sys.modules["torch.nn.functional"] = torch_f_stub
    sys.modules["mps_gaf_data_pipeline"] = types.SimpleNamespace(
        MPSGAFDataConfig=object,
        get_test_dataset=lambda *_args, **_kwargs: None,
        get_train_datasets=lambda *_args, **_kwargs: (None, None),
        make_grouped_dataloader=lambda *_args, **_kwargs: None,
    )
    sys.modules["mps_gaf_run"] = types.SimpleNamespace(
        chamfer_distance=lambda *_args, **_kwargs: None,
        rotation_error_deg=lambda *_args, **_kwargs: None,
    )


def _load_run_dcp_baseline():
    _install_dependency_stubs()
    module_path = Path(__file__).resolve().parents[1] / "run_dcp_baseline.py"
    spec = importlib.util.spec_from_file_location("run_dcp_baseline_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_dcp_baseline = _load_run_dcp_baseline()


def make_args(**overrides) -> Namespace:
    values = dict(
        dataset_path="datasets/modelnet40_ply_hdf5_2048",
        output_dir="runs/test",
        checkpoint=None,
        external_repo="external_src/learned_baselines/DCP",
        dcp_repo="external_src/learned_baselines/DCP",
        prnet_repo="external_src/learned_baselines/PRNet",
        idam_repo="external_src/learned_baselines/IDAM",
        rpmnet_repo="external_src/learned_baselines/RPMNet",
        pointnetlk_repo="external_src/learned_baselines/PointNetLK",
    )
    values.update(overrides)
    return Namespace(**values)


def test_validate_relative_paths_rejects_absolute_external_repo() -> None:
    args = make_args(external_repo=str(Path("external") / "DCP"))
    args.external_repo = str(Path(args.external_repo).resolve())

    with pytest.raises(ValueError, match="external_repo"):
        run_dcp_baseline.validate_learned_runner_paths(args)


def test_validate_relative_paths_rejects_absolute_pointnetlk_repo() -> None:
    args = make_args(external_repo=None, pointnetlk_repo=str(Path("external") / "PointNetLK"))
    args.pointnetlk_repo = str(Path(args.pointnetlk_repo).resolve())

    with pytest.raises(ValueError, match="pointnetlk_repo"):
        run_dcp_baseline.validate_learned_runner_paths(args)


def test_pose_metric_uses_mps_gaf_translation_weight() -> None:
    metrics = {
        "rotation_error_deg_mean": 10.0,
        "translation_error_mean": 0.2,
    }

    assert run_dcp_baseline.pose_metric(metrics, pose_trans_weight=50.0) == 20.0


def test_external_model_module_loading_does_not_pollute_global_modules(tmp_path, monkeypatch) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "model.py").write_text("MARKER = 'a'\n", encoding="utf-8")
    (repo_b / "model.py").write_text("MARKER = 'b'\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.delitem(sys.modules, "util", raising=False)

    module_a = run_dcp_baseline.load_external_model_module("repo_a")
    module_b = run_dcp_baseline.load_external_model_module("repo_b")

    assert module_a.MARKER == "a"
    assert module_b.MARKER == "b"
    assert "model" not in sys.modules
    assert "util" not in sys.modules


def test_schema_method_key_uses_real_family_name_for_non_dcp() -> None:
    assert run_dcp_baseline.schema_method_key(Namespace(model_family="pointnetlk")) == "pointnetlk"
