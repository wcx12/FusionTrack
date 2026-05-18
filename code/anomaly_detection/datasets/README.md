# Anomaly Detection Datasets

Datasets are intentionally not committed to this repository.

Place the VT-Tiny-MOT dataset here when running the individual anomaly detection pipeline:

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

From `code/anomaly_detection/individual/`, the default data root used by the uploaded scripts is:

```text
../datasets/VT-Tiny-MOT
```
