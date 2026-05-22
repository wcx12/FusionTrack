# 论文源码复现规则

本规则适用于本项目后续所有论文型 baseline、论文方法对比、论文实验复现。

## 强制规则

后续所有涉及论文方法的实验，必须优先以论文官方源码或论文明确指向的源码为准进行复现。只要结果要使用原论文方法名进入论文主表或正式对比表，就不能用本地近似实现、重新手写版本或 proxy 结果替代官方源码结果。

如果官方源码不可用、不可运行、license 不允许复用，或 adapter 无法在不改变数据语义的前提下达到完整覆盖，则该方法只能标注为 `paper-inspired proxy`、`internal/proxy/ablation` 或 `coverage-failed diagnostic`，不能写成已复现的原论文方法。

## 核心原则

只要方法要在论文主表中使用原论文方法名，就必须优先使用论文官方源码，或论文正文、附录、项目页明确指向的源码。每个结果必须记录源码 URL、commit、license、环境、adapter、数据 split、评价指标和配置差异。

如果没有可用源码，或者源码许可不允许直接复用，则本地实现不能写成“复现了该论文方法”。只能标注为：

- paper-inspired proxy
- lightweight proxy
- reimplementation following the high-level idea
- ablation / internal baseline

## 方法命名

论文主表禁止把本地近似实现直接命名为官方论文方法。例如：

- 本地 `individual_complementary:cetrajad_style` 只能写作 `CETrajAD-inspired proxy`，除非结果来自 CETrajAD 官方源码。
- 本地 `individual_trajectory_lm:ngram` 只能写作 `Trajectory-LM n-gram proxy`，除非结果来自 LM-TAD 官方源码。
- 本地 `individual_physics:kinematic_prior` 只能写作 `physics-informed kinematic prior`，除非结果来自 Pi-DPM 官方源码。

## 当前官方源码状态

- CETrajAD: 使用 [ShuruiCao/comp-ensemble-ad](https://github.com/ShuruiCao/comp-ensemble-ad) 作为官方/论文项目源码。实验可以直接外部 checkout 使用；提交到本仓库时，不复制其源码，除非同时记录明确的 license 或论文授权文本。当前已提供输入 bundle adapter 和 score 转换器。
- LM-TAD: 使用 [jonathankabala/LMTAD](https://github.com/jonathankabala/LMTAD) 作为官方源码。该仓库页面显示为 SIGSPATIAL 2024 PyTorch implementation，并显示 MIT license。当前已提供中间 sequence/vocab/manifest adapter 和 score 转换器，但官方 loader 仍需在外部 checkout 中接入自定义数据读取。
- Pi-DPM: 使用 [arunshar/Physics-Informed-Diffusion-Probabilistic-Model](https://github.com/arunshar/Physics-Informed-Diffusion-Probabilistic-Model) 作为官方源码。当前已完成输入/输出 adapter 脚手架，但还没有在服务器完成官方训练和打分。

## 审计记录字段

每个论文源码 baseline 至少记录：

```text
paper_name:
paper_url:
official_code_url:
commit:
license:
original_dataset:
original_metrics:
adapted_dataset:
adapted_metrics:
adapter_entrypoint:
environment:
status:
notes:
```

## 后续执行顺序

1. 外部 checkout 官方仓库，并锁定 commit。
2. 记录 license 和环境。
3. 写 adapter，把本项目数据转成官方仓库输入格式。
4. 写 adapter，把官方输出转回统一 score JSONL。
5. 在同一 train/val/test split 和同一评价指标下运行。
6. 只有完成上述步骤后，论文主表才允许使用原论文方法名。
