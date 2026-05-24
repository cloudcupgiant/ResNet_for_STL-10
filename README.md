# STL-10 Small ResNet 图像分类实验

本项目实现了基于 PyTorch 的 STL-10 图像分类流程：Small ResNet-18 from scratch、训练/验证曲线、四类优化实验、最终测试集评估、混淆矩阵和 Grad-CAM 可解释性分析。

## 数据目录

默认数据结构为：

```text
STL10/
  train/<class_name>/*.png
  test/<class_name>/*.png
```

训练和调参只读取 `STL10/train`，并按 `train:valid = 8:2` 做分层划分；`STL10/test` 只在最终评估和 Grad-CAM 阶段读取。

## 环境

```bash
python -m pip install -r requirements.txt
```

当前工作区已配置一个 Linux GPU 版虚拟环境：

```bash
source .venv_linux/bin/activate
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

本机检测到 `NVIDIA GeForce RTX 4090`，驱动版本为 `590.48.01`。项目环境已安装 CUDA 版 `torch==2.11.0+cu128` 和 `torchvision==0.26.0+cu128`。`scripts/*.sh` 会优先使用 `.venv_linux/bin/python`，通常不需要手动激活环境。

如需重装 GPU 版 PyTorch：

```bash
python -m pip install --force-reinstall -r requirements-gpu-cu128.txt
```

注意：在 Codex 普通沙箱内可能看不到 `/dev/nvidia*`，此时 `torch.cuda.is_available()` 会是 `False`；在本机正常终端中运行或沙箱外运行时可以访问 RTX 4090。

## 运行全部实验

```bash
bash scripts/run_all_experiments.sh
```
每个实验输出到 `outputs/<experiment>/`：

```text
best_model.pth
last_model.pth
train_log.csv
loss_curve.png
accuracy_curve.png
config.json
class_to_idx.json
split_indices.json
```

实验完成后会生成 `outputs/experiment_summary.csv`、`outputs/experiment_summary.md`、`outputs/best_experiment.json`，并把验证集准确率最高的模型保存为 `outputs/best_model.pth`。

## 单独训练

```bash
python -m src.train \
  --data-dir STL10 \
  --output-dir outputs/exp3_cosine \
  --experiment-name exp3_cosine \
  --epochs 150 \
  --batch-size 64 \
  --augmentation strong \
  --optimizer sgd \
  --lr 0.05 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler cosine \
  --dropout 0.2 \
  --label-smoothing 0.1 \
  --device cuda
```

## 最终测试集评估

```bash
bash scripts/evaluate_best.sh
```

输出目录 `outputs/final_test/` 包含：

```text
classification_report.txt
classification_report.json
confusion_matrix.png
confusion_matrix.csv
predictions.csv
metrics.json
```

## Grad-CAM

```bash
bash scripts/generate_gradcam.sh
```

Grad-CAM 默认选择 `layer4[-1].conv2`，每类至少尝试选取一个样本，并额外保存若干错误分类样本到 `outputs/gradcam/`。

## 报告

`report/report.md` 提供了完整报告结构。训练完成后，将 `outputs/experiment_summary.md`、`outputs/final_test/classification_report.txt`、`outputs/final_test/confusion_matrix.png` 和 `outputs/gradcam/` 中的可视化结果补入对应章节。
