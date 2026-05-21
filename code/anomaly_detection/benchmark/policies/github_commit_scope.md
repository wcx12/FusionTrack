# GitHub 提交范围规则

本文件定义当前 benchmark 工作中哪些内容可以提交到我们的 GitHub 仓库，哪些内容必须留在本地或服务器实验目录。

## 可以提交

- 本项目自己写的 benchmark 源码、runner、adapter、evaluation、tests、example config。
- `code/anomaly_detection/benchmark/configs/official_sources.example.json` 这类不含密码、不含私有路径、不含数据的 manifest 模板。
- 论文官方源码的 wrapper/adapter，例如把本项目 JSONL 转成官方仓库输入格式，或把官方输出转回统一 score JSONL。
- 复现实验规则、提交范围规则、运行说明、结果表生成脚本。
- 小型 synthetic fixture、单元测试输入、不会泄露原始数据的最小测试样例。
- 本地 proxy baseline 的源码，但必须在命名和文档中标明是 proxy/internal/ablation，不能冒充官方论文复现。

## 不应提交

- 服务器 IP、端口、密码、token、私钥、cookie、账号信息。
- VT-Tiny-MOT 原始数据、转换后的完整数据集、含私有路径的数据 manifest。
- 训练输出、`outputs/`、`runs/`、`checkpoints/`、`logs/`、模型权重、缓存、`__pycache__`。
- 大体积结果文件、完整分数文件、临时 smoke test 输出。
- 第三方论文源码的完整复制，除非 license 明确允许，并且随源码一起保留 license 文本。

## 第三方论文源码处理

- CETrajAD: 可以作为外部 checkout 直接用于实验；暂不把其源码复制进我们的仓库。主表结果可以叫 `CETrajAD` 的前提是结果确实由该官方仓库产生，并记录 commit、配置和 adapter。
- LM-TAD: 可以使用 `jonathankabala/LMTAD`。因为仓库显示 MIT license，可以选择外部 checkout、git submodule 或在保留 MIT license 的前提下 vendoring；建议优先外部 checkout 或 submodule，避免把第三方训练代码混进本项目核心代码。
- Pi-DPM: 当前只提交我们自己的 adapter。官方仓库可以外部 checkout 或在确认 license 后作为 submodule。模型权重和官方仓库运行输出不提交。

## 推荐仓库结构

```text
code/anomaly_detection/benchmark/
  baselines/              # 我们自己的 classical/proxy/internal baseline
  external_sources/       # 官方仓库输入/输出 adapter，不放第三方完整源码
  configs/                # 不含秘密和私有路径的 example config
  policies/               # 可提交的复现规则和提交范围规则
  runners/                # 统一运行入口
  tests/                  # 单元测试
```

如果以后确实需要把某个第三方仓库纳入版本控制，优先使用 pinned submodule，并在 manifest 中记录 URL、commit、license 和用途。

## 提交前审计命令

提交 benchmark 相关文件前，先运行：

```powershell
python code/anomaly_detection/benchmark/runners/audit_commit_scope.py
```

该脚本会基于 Git 当前可提交候选文件检查：

- 是否误包含 `outputs/`、`runs/`、`checkpoints/`、`logs/`、`__pycache__/`。
- 是否误包含 `.pth`、`.pt`、`.pkl`、`.npy`、`.npz` 等实验产物。
- 是否出现明显 SSH、账号、token、secret 样式文本。

脚本返回非零退出码时，不应继续提交，先清理对应文件或修改 `.gitignore`。
