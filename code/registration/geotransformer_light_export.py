"""Lightweight GeoTransformer inference export for schema conversion.

Run this file from a GeoTransformer experiment directory so it can import that
directory's ``config.py``, ``dataset.py``, ``model.py``, and ``loss.py``.  It
uses the official tester stack but saves only the fields required by
``convert_geotransformer_schema.py``.
"""

from __future__ import annotations

import argparse
import os.path as osp
import time

import numpy as np
import torch
from tqdm import tqdm

from geotransformer.engine import SingleTester
from geotransformer.utils.common import ensure_dir, get_log_string
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.torch import release_cuda, to_cuda

from config import make_cfg
from dataset import test_data_loader
from loss import Evaluator
from model import create_model


_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_compat


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--dataset_root",
        default=None,
        help="Optional GeoTransformer-format dataset root. Defaults to the experiment config dataset root.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--output_name", default="features_light")
    return parser


class LightExportTester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())
        if self.args.dataset_root is not None:
            cfg.data.dataset_root = self.args.dataset_root

        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
        loading_time = time.time() - start_time
        self.logger.info(f"Data loader created: {loading_time:.3f}s collapsed.")
        self.logger.info(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)

        model = create_model(cfg).cuda()
        self.register_model(model)

        self.evaluator = Evaluator(cfg).cuda()
        self.output_dir = osp.join(cfg.output_dir, self.args.output_name, self.args.benchmark)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        return self.model(data_dict)

    def eval_step(self, iteration, data_dict, output_dict):
        return self.evaluator(output_dict, data_dict)

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        ref_id = data_dict["ref_frame"]
        src_id = data_dict["src_frame"]

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f"{ref_id}_{src_id}.npz")
        np.savez_compressed(
            file_name,
            ref_points_f=release_cuda(output_dict["ref_points_f"]),
            src_points_f=release_cuda(output_dict["src_points_f"]),
            estimated_transform=release_cuda(output_dict["estimated_transform"]),
            transform=release_cuda(data_dict["transform"]),
        )

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        ref_frame = data_dict["ref_frame"]
        src_frame = data_dict["src_frame"]
        message = f"{scene_name}, id0: {ref_frame}, id1: {src_frame}"
        message += ", " + get_log_string(result_dict=result_dict)
        message += ", nCorr: {}".format(output_dict["corr_scores"].shape[0])
        return message

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        start_index = max(0, int(self.args.start_index))
        end_index = len(self.test_loader)
        if self.args.max_pairs is not None:
            end_index = min(end_index, start_index + int(self.args.max_pairs))
        total_iterations = max(0, end_index - start_index)
        pbar = tqdm(total=total_iterations)
        for iteration, data_dict in enumerate(self.test_loader):
            if iteration < start_index:
                continue
            if iteration >= end_index:
                break
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f", {timer.tostring()}"
            pbar.set_description(message)
            pbar.update(1)
            del data_dict, output_dict, result_dict
            torch.cuda.empty_cache()
        pbar.close()
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        self.logger.critical(get_log_string(result_dict=summary_dict, timer=timer))


def main() -> None:
    cfg = make_cfg()
    tester = LightExportTester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
