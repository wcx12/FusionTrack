from argparse import Namespace
import importlib.util
from pathlib import Path
import math
import sys
import types

import numpy as np
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


def _load_run_registration_benchmark_suite():
    module_path = Path(__file__).resolve().parents[1] / "run_registration_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_registration_benchmark_suite_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_registration_benchmark = _load_run_registration_benchmark()
run_registration_benchmark_suite = _load_run_registration_benchmark_suite()
non_learning_baselines = sys.modules["non_learning_baselines"]


class _MatrixLike:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=float)

    def __getitem__(self, item):
        out = self.value[item]
        if np.isscalar(out):
            return float(out)
        return _MatrixLike(out)

    def __matmul__(self, other):
        return _MatrixLike(self.value @ other.value)

    def t(self):
        return _MatrixLike(self.value.T)


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


def test_suite_case_args_include_external_dataset_options() -> None:
    args = Namespace(
        dataset_path="datasets/predator_3dmatch/data",
        dataset_name="3dmatch",
        dataset_split="test",
        methods="identity",
        rot_mag=45.0,
        trans_mag=0.5,
        num_sources_per_ref=2,
        groups_per_batch=1,
        num_workers=0,
        max_eval_batches=20,
        icp_iterations=20,
        icp_trim_fraction=0.7,
        success_rotation_deg=15.0,
        success_translation=0.5,
        icp_point_max_angle_deg=10.0,
        icp_point_max_translation=0.2,
        fpfh_voxel_size=0.05,
        fpfh_normal_radius=0.1,
        fpfh_feature_radius=0.25,
        fpfh_normal_max_nn=30,
        fpfh_feature_max_nn=100,
        fpfh_max_correspondence_distance=0.075,
        fpfh_ransac_n=4,
        fpfh_ransac_max_iterations=100000,
        split_root="external_src/new_baselines/GeoTransformer/data/3DMatch/metadata",
        pair_list="external_src/new_baselines/GeoTransformer/data/3DMatch/metadata/3DMatch.pkl",
        no_estimate_normals=True,
    )
    case = {
        "name": "standard_pair",
        "noise_type": "clean",
        "num_points": 2048,
        "partial": [1.0, 1.0],
    }

    case_args = run_registration_benchmark_suite._build_case_args(args, case)

    assert "--split_root" in case_args
    assert "--pair_list" in case_args
    assert "--no_estimate_normals" in case_args


def test_turboreg_is_registered_as_optional_baseline() -> None:
    assert "turboreg" in run_registration_benchmark.baseline_method_names()
    assert run_registration_benchmark.parse_baseline_methods("turboreg") == ["turboreg"]


def test_recent_spatial_filter_baselines_are_registered() -> None:
    assert "mac" in run_registration_benchmark.baseline_method_names()
    assert "mac_fpfh" in run_registration_benchmark.baseline_method_names()
    assert "sc2_pcr" in run_registration_benchmark.baseline_method_names()
    assert run_registration_benchmark.parse_baseline_methods("mac,sc2_pcr") == ["mac", "sc2_pcr"]


def test_kiss_matcher_is_registered_as_optional_baseline() -> None:
    assert "kiss_matcher" in run_registration_benchmark.baseline_method_names()
    assert run_registration_benchmark.parse_baseline_methods("kiss_matcher,kiss") == [
        "kiss_matcher",
        "kiss",
    ]


def test_turboreg_case_passes_configured_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_run(method, src, ref, **kwargs):
        calls["method"] = method
        calls["kwargs"] = kwargs
        return run_registration_benchmark.BaselineResult(
            transform=np.eye(3, 4, dtype=np.float32),
            runtime_sec=0.0,
            meta={},
        )

    monkeypatch.setattr(run_registration_benchmark, "run_non_learning_baseline", fake_run)
    args = Namespace(
        turboreg_voxel_size=0.08,
        turboreg_normal_radius=0.16,
        turboreg_feature_radius=0.40,
        turboreg_normal_max_nn=20,
        turboreg_feature_max_nn=80,
        turboreg_max_correspondences=256,
        turboreg_max_n=512,
        turboreg_tau_length_consis=0.02,
        turboreg_num_pivot=128,
        turboreg_radiu_nms=0.12,
        turboreg_tau_inlier=0.09,
        turboreg_metric="MAE",
        turboreg_device="cpu",
    )
    src = np.zeros((8, 3), dtype=np.float32)
    ref = np.ones((8, 3), dtype=np.float32)

    result, kwargs, error, valid = run_registration_benchmark._benchmark_methods_case(
        "turboreg",
        args,
        src,
        ref,
        src,
        ref,
    )

    assert valid is True
    assert error == ""
    assert result.runtime_sec == 0.0
    assert calls["method"] == "turboreg"
    assert kwargs == calls["kwargs"]
    assert kwargs["voxel_size"] == 0.08
    assert kwargs["max_correspondences"] == 256
    assert kwargs["metric"] == "MAE"
    assert kwargs["device"] == "cpu"


def test_mac_case_passes_configured_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_run(method, src, ref, **kwargs):
        calls["method"] = method
        calls["kwargs"] = kwargs
        return run_registration_benchmark.BaselineResult(
            transform=np.eye(3, 4, dtype=np.float32),
            runtime_sec=0.0,
            meta={},
        )

    monkeypatch.setattr(run_registration_benchmark, "run_non_learning_baseline", fake_run)
    args = Namespace(
        mac_voxel_size=0.07,
        mac_normal_radius=0.14,
        mac_feature_radius=0.35,
        mac_normal_max_nn=24,
        mac_feature_max_nn=96,
        mac_max_correspondences=384,
        mac_compatibility_distance=0.03,
        mac_max_seeds=32,
        mac_refine_iterations=6,
        mac_refine_trim_fraction=0.6,
    )
    src = np.zeros((8, 3), dtype=np.float32)
    ref = np.ones((8, 3), dtype=np.float32)

    result, kwargs, error, valid = run_registration_benchmark._benchmark_methods_case(
        "mac",
        args,
        src,
        ref,
        src,
        ref,
    )

    assert valid is True
    assert error == ""
    assert result.runtime_sec == 0.0
    assert calls["method"] == "mac"
    assert kwargs == calls["kwargs"]
    assert kwargs["voxel_size"] == 0.07
    assert kwargs["compatibility_distance"] == 0.03
    assert kwargs["max_seeds"] == 32


def test_sc2_case_passes_configured_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_run(method, src, ref, **kwargs):
        calls["method"] = method
        calls["kwargs"] = kwargs
        return run_registration_benchmark.BaselineResult(
            transform=np.eye(3, 4, dtype=np.float32),
            runtime_sec=0.0,
            meta={},
        )

    monkeypatch.setattr(run_registration_benchmark, "run_non_learning_baseline", fake_run)
    args = Namespace(
        sc2_voxel_size=0.06,
        sc2_normal_radius=0.12,
        sc2_feature_radius=0.30,
        sc2_normal_max_nn=22,
        sc2_feature_max_nn=88,
        sc2_max_correspondences=320,
        sc2_compatibility_distance=0.04,
        sc2_max_selected_correspondences=48,
        sc2_power_iterations=7,
        sc2_refine_iterations=5,
        sc2_refine_trim_fraction=0.65,
    )
    src = np.zeros((8, 3), dtype=np.float32)
    ref = np.ones((8, 3), dtype=np.float32)

    result, kwargs, error, valid = run_registration_benchmark._benchmark_methods_case(
        "sc2_pcr",
        args,
        src,
        ref,
        src,
        ref,
    )

    assert valid is True
    assert error == ""
    assert result.runtime_sec == 0.0
    assert calls["method"] == "sc2_pcr"
    assert kwargs == calls["kwargs"]
    assert kwargs["voxel_size"] == 0.06
    assert kwargs["max_selected_correspondences"] == 48
    assert kwargs["power_iterations"] == 7


def test_kiss_matcher_case_passes_configured_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_run(method, src, ref, **kwargs):
        calls["method"] = method
        calls["kwargs"] = kwargs
        return run_registration_benchmark.BaselineResult(
            transform=np.eye(3, 4, dtype=np.float32),
            runtime_sec=0.0,
            meta={},
        )

    monkeypatch.setattr(run_registration_benchmark, "run_non_learning_baseline", fake_run)
    args = Namespace(kiss_voxel_size=0.11)
    src = np.zeros((8, 3), dtype=np.float32)
    ref = np.ones((8, 3), dtype=np.float32)

    result, kwargs, error, valid = run_registration_benchmark._benchmark_methods_case(
        "kiss_matcher",
        args,
        src,
        ref,
        src,
        ref,
    )

    assert valid is True
    assert error == ""
    assert result.runtime_sec == 0.0
    assert calls["method"] == "kiss_matcher"
    assert kwargs == calls["kwargs"]
    assert kwargs["voxel_size"] == 0.11


def test_rotation_error_is_zero_for_identical_rotations() -> None:
    identity = _MatrixLike(np.eye(4))

    assert run_registration_benchmark.rotation_error_deg(identity, identity) == 0.0


def test_rotation_error_projects_slightly_non_orthogonal_inputs() -> None:
    transform = np.eye(4)
    transform[:3, :3] = np.diag([0.9995, 0.9997, 0.9996])
    matrix = _MatrixLike(transform)

    assert run_registration_benchmark.rotation_error_deg(matrix, matrix) == pytest.approx(0.0)


def test_failed_baseline_metrics_are_still_counted() -> None:
    acc = run_registration_benchmark.build_metric_accumulator()

    run_registration_benchmark.update_metrics(
        acc,
        rot=30.0,
        trans=0.4,
        chamfer=0.2,
        runtime=1.5,
        success=0,
        valid=False,
    )
    metrics = run_registration_benchmark.finalize_metrics(acc)

    assert metrics["num_pairs"] == 1
    assert metrics["num_successful_pairs"] == 0
    assert metrics["num_failed_pairs"] == 1
    assert metrics["rotation_error_deg_mean"] == 30.0
    assert metrics["translation_error_mean"] == 0.4
    assert metrics["chamfer_distance_mean"] == 0.2
    assert metrics["skip_rate"] == 1.0


def test_cpd_rigid_disables_similarity_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRigidRegistration:
        kwargs = {}

        def __init__(self, **kwargs):
            FakeRigidRegistration.kwargs = kwargs

        def register(self):
            return None, (2.5, np.eye(3, dtype=np.float32), np.array([0.1, 0.2, 0.3], dtype=np.float32))

    monkeypatch.setattr(non_learning_baselines, "_load_pycpd", lambda: FakeRigidRegistration)
    src = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ref = src.copy()

    result = non_learning_baselines.cpd_rigid(src, ref)

    assert FakeRigidRegistration.kwargs["scale"] is False
    assert result.transform[:, :3] == pytest.approx(np.eye(3, dtype=np.float32))
    assert result.meta["scale"] == pytest.approx(2.5)
    assert result.meta["scale_disabled"] is True


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
