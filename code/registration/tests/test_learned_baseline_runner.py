from argparse import Namespace
import importlib.util
import json
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
    torch_stub.load = lambda *_args, **_kwargs: {}
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
        dataset_name="modelnet40",
        split_root=None,
        pair_list=None,
        no_estimate_normals=False,
        external_repo="external_src/learned_baselines/DCP",
        dcp_repo="external_src/learned_baselines/DCP",
        prnet_repo="external_src/learned_baselines/PRNet",
        idam_repo="external_src/learned_baselines/IDAM",
        rpmnet_repo="external_src/learned_baselines/RPMNet",
        pointnetlk_repo="external_src/learned_baselines/PointNetLK",
        omnet_repo="external_src/new_baselines/OMNet_Pytorch",
        train_category_file=None,
        val_category_file=None,
        test_category_file=None,
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


def test_validate_relative_paths_rejects_absolute_omnet_repo() -> None:
    args = make_args(external_repo=None, omnet_repo=str(Path("external") / "OMNet_Pytorch"))
    args.omnet_repo = str(Path(args.omnet_repo).resolve())

    with pytest.raises(ValueError, match="omnet_repo"):
        run_dcp_baseline.validate_learned_runner_paths(args)


def test_validate_relative_paths_rejects_absolute_category_file() -> None:
    args = make_args(test_category_file=str(Path("splits") / "test.txt"))
    args.test_category_file = str(Path(args.test_category_file).resolve())

    with pytest.raises(ValueError, match="test_category_file"):
        run_dcp_baseline.validate_learned_runner_paths(args)


def test_pose_metric_uses_mps_gaf_translation_weight() -> None:
    metrics = {
        "rotation_error_deg_mean": 10.0,
        "translation_error_mean": 0.2,
    }

    assert run_dcp_baseline.pose_metric(metrics, pose_trans_weight=50.0) == 20.0


def test_resume_selection_state_preserves_previous_best(tmp_path) -> None:
    summary = {
        "best_selection_metric": 12.5,
        "best_validation": {
            "rotation_error_deg_mean": 10.0,
            "translation_error_mean": 0.05,
        },
        "epochs_since_best": 7,
    }
    (tmp_path / "last_train_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    best_metric, best_metrics, epochs_since_best = run_dcp_baseline.load_previous_selection_state(tmp_path)

    assert best_metric == 12.5
    assert best_metrics == summary["best_validation"]
    assert epochs_since_best == 7


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


def test_apply_checkpoint_model_args_uses_stored_architecture(monkeypatch) -> None:
    args = make_args(
        checkpoint="runs/model/best.pt",
        no_checkpoint_model_args=False,
        model_family="dcp",
        emb_nn="pointnet",
        emb_dims=512,
        n_iters=3,
    )
    monkeypatch.setattr(
        run_dcp_baseline.torch,
        "load",
        lambda *_args, **_kwargs: {
            "args": {
                "model_family": "idam",
                "emb_nn": "GNN",
                "emb_dims": 64,
                "n_iters": 1,
            }
        },
    )

    updated = run_dcp_baseline.apply_checkpoint_model_args(args)

    assert updated.model_family == "idam"
    assert updated.emb_nn == "GNN"
    assert updated.emb_dims == 64
    assert updated.n_iters == 1


def test_apply_checkpoint_model_args_can_be_disabled(monkeypatch) -> None:
    args = make_args(
        checkpoint="runs/model/best.pt",
        no_checkpoint_model_args=True,
        model_family="dcp",
        emb_nn="pointnet",
        emb_dims=512,
        n_iters=3,
    )
    monkeypatch.setattr(
        run_dcp_baseline.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("checkpoint should not be loaded"),
    )

    updated = run_dcp_baseline.apply_checkpoint_model_args(args)

    assert updated.model_family == "dcp"
    assert updated.emb_nn == "pointnet"
    assert updated.emb_dims == 512
    assert updated.n_iters == 3


def test_build_data_config_passes_category_files(monkeypatch) -> None:
    captured = {}

    def fake_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(run_dcp_baseline, "MPSGAFDataConfig", fake_config)
    args = make_args(
        num_points=1024,
        noise_type="crop",
        rot_mag=45.0,
        trans_mag=0.5,
        partial=[0.7, 0.7],
        num_sources_per_ref=2,
        train_category_file="splits/train.txt",
        val_category_file="splits/val.txt",
        test_category_file="splits/test.txt",
        seed=0,
    )

    run_dcp_baseline.build_data_config(args)

    assert captured["train_category_file"] == "splits/train.txt"
    assert captured["val_category_file"] == "splits/val.txt"
    assert captured["test_category_file"] == "splits/test.txt"


def test_build_data_config_passes_kitti_metadata_options(monkeypatch) -> None:
    captured = {}

    def fake_config(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(run_dcp_baseline, "MPSGAFDataConfig", fake_config)
    args = make_args(
        dataset_name="kitti",
        dataset_path="datasets/kitti",
        split_root="external_src/new_baselines/GeoTransformer/data/Kitti/metadata",
        pair_list="splits/kitti/test.pkl",
        no_estimate_normals=True,
        num_points=2048,
        noise_type="crop",
        rot_mag=45.0,
        trans_mag=0.5,
        partial=[0.7, 0.7],
        num_sources_per_ref=1,
        seed=7,
    )

    run_dcp_baseline.build_data_config(args)

    assert captured["dataset_name"] == "kitti"
    assert captured["split_root"] == "external_src/new_baselines/GeoTransformer/data/Kitti/metadata"
    assert captured["pair_list"] == "splits/kitti/test.pkl"
    assert captured["estimate_normals"] is False


class _FakeTensor:
    def __getitem__(self, _item):
        return self

    def __matmul__(self, _other):
        return self

    def __add__(self, _other):
        return self

    def to(self, _device):
        return self

    def transpose(self, *_args):
        return self

    def contiguous(self):
        return self

    def unsqueeze(self, *_args):
        return self


class _FakeOmnetModel:
    training = True
    _mps_gaf_omnet_params = {}

    def __call__(self, _batch):
        transform = _FakeTensor()
        return {"transform_pair": [transform, transform]}


def test_omnet_predict_transform_skips_aux_loss_import_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        run_dcp_baseline,
        "_resolve_relative_dir",
        lambda _repo: pytest.fail("auxiliary loss repo should not be resolved"),
    )
    monkeypatch.setattr(
        run_dcp_baseline,
        "rotation_matrix_to_quaternion_wxyz",
        lambda _rotation: _FakeTensor(),
    )
    monkeypatch.setattr(
        run_dcp_baseline.torch,
        "cat",
        lambda _items, dim=0: _FakeTensor(),
        raising=False,
    )
    args = make_args(model_family="omnet", wt_aux=0.0)
    batch = {
        "points_src": _FakeTensor(),
        "points_ref": _FakeTensor(),
        "transform_gt": _FakeTensor(),
    }

    _transform, _pred_points, aux_loss = run_dcp_baseline.predict_transform(
        _FakeOmnetModel(),
        batch,
        "cpu",
        args,
    )

    assert aux_loss is None
