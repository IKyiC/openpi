# AGENTS.md

当前仓库是 openpi。QVLA 仓库位于同级目录 ../QVLA，且该仓库也是用户自己的 fork，可以读取，也可以在用户明确要求时修改，但必须经过用户同意。但当前优先把适配代码写在 openpi 仓库中。

## 工作方式

本地 Windows 只用于：
- 浏览代码
- 修改代码
- 用 Codex 生成实验脚本
- git commit / git push

服务器 Linux 才用于：
- 安装 openpi 环境
- 加载 checkpoint
- 跑推理和实验

不要假设本地 Windows 上存在模型 checkpoint。
不要在代码中硬编码 Windows 路径，例如 C:\ 或 E:\。
所有实验脚本都应该能在 Linux 服务器上运行。

## 当前目标

把 QVLA 的 training-free fake weight quantization 方法适配到 openpi 的 π0.5-LIBERO PyTorch 模型上。

使用模型：
- config: pi05_libero
- checkpoint: ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
- 使用微调后的 pi05_libero，不是 pi05_base。

## 限制

- 不要训练。
- 不要创建 conda 环境。
- 不要运行 scripts/train.py 或 scripts/train_pytorch.py。
- 不要修改 checkpoint 文件。
- 不要硬编码 Windows 路径。
- 不要直接套用 QVLA 的 OpenVLA 加载代码。
- QVLA 只作为量化方法参考。
- openpi 模型加载必须走 openpi 的 PyTorch policy 接口。

## 分支规则

当前项目使用多个方法分支隔离实验。

公共基础分支：
- pi05-quant-base

不要把不同方法的实现混在同一个分支里。
新增一个量化方法时，必须从 pi05-quant-base 新建分支。

本地 Windows 只用于改代码和 git push。
服务器 Linux 才用于运行实验。
不要硬编码 Windows 路径。