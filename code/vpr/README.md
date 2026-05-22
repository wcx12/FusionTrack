# TF-VPR 代码

这个目录包含 FusionTrack 使用的 Visual Place Recognition 实现。

- 来源仓库：https://github.com/ddfs430/TF-VPR
- 导入 commit：`c2c9b0d06842f0d8c89e91f8a451a2d5214739b5`
- 许可证：MIT，见 `LICENSE`

本目录只包含代码和文本元数据。数据集、预训练权重、运行日志、缓存文件和上游展示图片都不提交到仓库。

## 目录说明

- `eval.py`：主要评价入口。
- `parser.py`：命令行参数。
- `datasets_ws.py`：按 database/query 图片目录读取数据集。
- `test.py`：特征提取、最近邻检索、recall 计算和 per-query JSON 输出。
- `model/`：TF-VPR backbone 和 aggregation 模块。
- `UPSTREAM_README.md`：上游 README 的中文整理版，保留原始使用说明和链接。
- `requirements.txt`：上游 Python 环境依赖。

## 数据集格式

数据读取器期望使用 VisualGeoLocalization 风格的数据集目录：

```text
<eval_datasets_folder>/<dataset_name>/images/test/database/*.jpg
<eval_datasets_folder>/<dataset_name>/images/test/queries/*.jpg
```

图片文件名必须包含上游格式的 UTM 坐标：

```text
.../@<utm_easting>@<utm_northing>@...@.jpg
```

默认数据集根目录是 `/data/datasets/vpr`，可以通过 `--eval_datasets_folder` 覆盖。

## 预训练权重

默认 DINOv2 checkpoint 路径是：

```text
/data/users/model_weight/DINO_V2/dinov2_vitb14_pretrain.pth
```

可以通过 `--foundation_model_path` 覆盖。预训练权重应保存在 Git 仓库外部。

## 运行示例

请从 `code/vpr/` 目录运行，保证本地 import 能正确解析：

```bash
python eval.py \
  --eval_dataset_name pitts30k \
  --eval_datasets_folder /data/datasets/vpr \
  --foundation_model_path /data/users/model_weight/DINO_V2/dinov2_vitb14_pretrain.pth \
  --backbone dinov2 \
  --mode TF_VPR \
  --num_clusters 17
```

输出会写入：

```text
test/<save_dir>/<timestamp>/
```
