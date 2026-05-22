# 异常检测数据集目录

数据集不提交到本仓库。这个目录只作为本地运行异常检测流水线时的数据放置约定。

运行个体异常检测或 benchmark 协议生成脚本时，建议把 VT-Tiny-MOT 数据集放在：

```text
code/anomaly_detection/datasets/VT-Tiny-MOT/
├── annotations/
│   ├── instances_00_train2017.json
│   ├── instances_01_train2017.json
│   ├── instances_00_test2017.json
│   └── instances_01_test2017.json
├── train2017/
└── test2017/
```

从 `code/anomaly_detection/individual/` 目录运行早期个体异常检测脚本时，默认数据根目录是：

```text
../datasets/VT-Tiny-MOT
```

如果数据集放在其它位置，请在命令中显式传入 `--data-root`。不要把原始数据、导出的中间数据、模型权重或实验输出提交到 GitHub。
