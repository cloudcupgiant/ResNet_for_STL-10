# ResNet_for_STL-10
This project implements a complete STL-10 image classification pipeline based on PyTorch, using a Small ResNet-18 model trained from scratch. It covers the full experimental workflow, including stratified train/validation splitting, training and validation curve visualization, multiple optimization experiments, final test-set evaluation, confusion matrix analysis, and Grad-CAM-based model interpretability. The training process automatically saves the best and last model checkpoints, logs training metrics, generates loss and accuracy curves, and summarizes the validation performance of different experimental settings. After training, the best model is evaluated on the independent test set, producing a classification report, confusion matrix, prediction results, and evaluation metrics. Grad-CAM visualizations are further generated to analyze which image regions the model focuses on during classification, providing an end-to-end framework for model training, performance evaluation, and visual explanation on the STL-10 dataset.
## Dataset Directory

The default dataset structure is:

STL10/
  train/<class_name>/*.png
  test/<class_name>/*.png

Training and hyperparameter tuning only use STL10/train. The training set is split into training and validation subsets with a stratified ratio of train:valid = 8:2. The STL10/test directory is only used for final evaluation and Grad-CAM analysis.

Environment
python -m pip install -r requirements.txt

The current workspace has been configured with a Linux GPU virtual environment:

source .venv_linux/bin/activate
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"

The local machine detects an NVIDIA GeForce RTX 4090 with driver version 590.48.01. The project environment has installed the CUDA versions of torch==2.11.0+cu128 and torchvision==0.26.0+cu128.

The scripts under scripts/*.sh will prioritize using .venv_linux/bin/python, so manually activating the virtual environment is usually not required.

To reinstall the GPU version of PyTorch, run:

python -m pip install --force-reinstall -r requirements-gpu-cu128.txt

Note: In a standard Codex sandbox, /dev/nvidia* may not be visible, so torch.cuda.is_available() may return False. When running in a normal local terminal or outside the sandbox, the RTX 4090 should be accessible.

Run All Experiments
bash scripts/run_all_experiments.sh

Each experiment will save its outputs to outputs/<experiment>/:

best_model.pth
last_model.pth
train_log.csv
loss_curve.png
accuracy_curve.png
config.json
class_to_idx.json
split_indices.json

After all experiments are completed, the following summary files will be generated:

outputs/experiment_summary.csv
outputs/experiment_summary.md
outputs/best_experiment.json

The model with the highest validation accuracy will also be saved as:

outputs/best_model.pth
Train a Single Experiment
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
Final Test Evaluation
bash scripts/evaluate_best.sh

The output directory outputs/final_test/ contains:

classification_report.txt
classification_report.json
confusion_matrix.png
confusion_matrix.csv
predictions.csv
metrics.json
Grad-CAM
bash scripts/generate_gradcam.sh

Grad-CAM uses layer4[-1].conv2 by default. The script attempts to select at least one sample from each class and additionally saves several misclassified examples to:

outputs/gradcam/
Report

report/report.md provides the full report structure. After training is completed, the following results can be added to the corresponding sections of the report:

outputs/experiment_summary.md
outputs/final_test/classification_report.txt
outputs/final_test/confusion_matrix.png
outputs/gradcam/
