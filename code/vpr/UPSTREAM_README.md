# TF-VPR 上游说明中文整理

本文档整理自 TF-VPR 上游仓库 README，用于在 FusionTrack 仓库中保留可读的中文使用说明。

上游论文：`TF-VPR: A Novel Benchmark for Training-Free Visual Place Recognition`

上游仓库：https://github.com/ddfs430/TF-VPR

## 快速开始

上游代码按照 Visual Geo-localization Benchmark 的评价方式组织数据和实验：

- Visual Geo-localization Benchmark：https://github.com/gmberton/deep-visual-geo-localization-benchmark
- VPR 数据集下载器：https://github.com/gmberton/VPR-datasets-downloader

测试数据集需要整理成下面的目录结构：

```text
datasets/
├── pitts30k/
│   └── images/
│       ├── train/
│       │   ├── database/
│       │   └── queries/
│       ├── val/
│       │   ├── database/
│       │   └── queries/
│       └── test/
│           ├── database/
│           └── queries/
└── msls/
    └── images/
        ├── train/
        │   ├── database/
        │   └── queries/
        ├── val/
        │   ├── database/
        │   └── queries/
        └── test/
            ├── database/
            └── queries/
```

测试前需要下载 DINOv2 ViT-B/14 预训练 foundation model：

```text
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
```

## TF-VPR

```bash
python3 eval.py --eval_dataset_name=pitts30k --backbone=dinov2 --mode TF_VPR --num_clusters=17
```

## GeM + CLS

```bash
python3 eval.py --eval_dataset_name=pitts30k --backbone=dinov2 --mode GeM_CLS --num_clusters=14
```

## GeM + Mean

```bash
python3 eval.py --eval_dataset_name=pitts30k --backbone=dinov2 --mode GeM_Mean --num_clusters=14
```

## CLS + Mean

```bash
python3 eval.py --eval_dataset_name=pitts30k --backbone=dinov2 --mode CLS_Mean --num_clusters=2
```

## CLS Token

```bash
python3 eval.py --eval_dataset_name=pitts30k --backbone=dinov2 --mode CLS --num_clusters=1
```

## Mean Token

```bash
python3 eval.py --eval_dataset_name=pitts30k --backbone=dinov2 --mode Mean --num_clusters=1
```

## 主要结果

上游仓库 README 中包含结果图 `result.jpg`。FusionTrack 仓库没有提交上游展示图片，避免把无关二进制资源放进本仓库；如需查看图像，请到上游仓库查看。

## 致谢

上游 TF-VPR 项目参考了以下仓库：

- Visual Geo-localization Benchmark：https://github.com/gmberton/deep-visual-geo-localization-benchmark
- DINOv2：https://github.com/facebookresearch/dinov2
