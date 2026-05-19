# CS60003-HW2: Flowers102 Image Classification with PyTorch

本项目完成 102 Category Flower Dataset 的花卉图像分类实验，支持 ImageNet 预训练 ResNet 微调、超参数分析、预训练消融、SE/CBAM 注意力模块，以及 ViT-Tiny 和 Swin-T 对比实验。所有任务都可以单独通过命令行运行，输出结果会保存到独立实验目录，方便后续写实验报告。

## 项目结构

```text
hw2/
├── data/
├── configs/
├── models/
│   ├── resnet_attention.py
│   └── model_factory.py
├── scripts/
│   ├── prepare_data.py
│   ├── run_all_resnet_experiments.sh
│   ├── run_task2_hparam_sweep.sh
│   ├── train_task1_baseline.py
│   ├── train_task2_hparam.py
│   ├── train_task3_ablation.py
│   └── train_task4_attention.py
├── utils/
│   ├── argparse_utils.py
│   ├── dataset.py
│   ├── logger.py
│   ├── metrics.py
│   └── train_utils.py
├── outputs/
├── environment.yml
├── requirements.txt
└── README.md
```

## 环境安装

建议先进入 `hw2` 目录：

```bash
cd hw2
```

### 方法一：使用 conda 创建虚拟环境

推荐使用 conda 创建独立环境：

```bash
conda create -n flowers102 python=3.10 -y
conda activate flowers102
pip install -r requirements.txt
```

也可以直接通过 `environment.yml` 创建：

```bash
conda env create -f environment.yml
conda activate flowers102
```

### 方法二：使用已有 Python 环境

```bash
pip install -r requirements.txt
```

如果使用 CUDA，请根据自己的 CUDA 版本优先安装对应的 PyTorch 官方版本。例如可以先安装 CUDA 版 PyTorch，再安装其余依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib tqdm scikit-learn timm scipy
```

## Task 0：数据准备

```bash
python scripts/prepare_data.py --data_dir ./data
```

该命令会检查并下载 `torchvision.datasets.Flowers102`，并输出 train/val/test 的样本数量和类别数。

## Mac M 系列运行建议

本项目在 Mac Apple Silicon 上建议使用：

```bash
--device mps
--num_workers 0
```

如果遇到 MPS 暂不支持的算子，可以在运行训练前启用 fallback：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Task 1：ImageNet 预训练 ResNet Baseline

ResNet-18：

```bash
python scripts/train_task1_baseline.py \
  --model resnet18 \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps \
  --output_dir outputs/task1_resnet18
```

ResNet-34：

```bash
python scripts/train_task1_baseline.py \
  --model resnet34 \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps \
  --output_dir outputs/task1_resnet34
```

## Task 2：超参数分析

每次运行对应一个超参数组合，默认输出目录会包含 epoch、backbone lr、classifier lr 和 batch size。

```bash
python scripts/train_task2_hparam.py \
  --model resnet18 \
  --data_dir ./data \
  --epochs 20 \
  --batch_size 32 \
  --backbone_lr 1e-5 \
  --classifier_lr 1e-4 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

```bash
python scripts/train_task2_hparam.py \
  --model resnet18 \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

```bash
python scripts/train_task2_hparam.py \
  --model resnet34 \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

也可以使用批量脚本一次跑多个超参数组合：

```bash
bash scripts/run_task2_hparam_sweep.sh
```

需要增加实验时，编辑 `scripts/run_task2_hparam_sweep.sh` 中的 `CONFIGS` 数组：

```bash
CONFIGS=(
  # Baseline: moderate lr, standard setting
  "30 1e-3 32"

  # Learning rate analysis
  "30 1e-4 32"
  "30 5e-4 32"
  "30 2e-3 32"

  # Epoch analysis
  "10 1e-3 32"
  "50 1e-3 32"

  # Batch size analysis
  "30 1e-3 16"
)
```

每一行格式为 `"epochs lr batch_size"`，其中 `lr` 会作为 `classifier_lr`，`backbone_lr` 默认自动设为 `lr * 0.1`。如果要跑 ResNet-34：

```bash
MODEL=resnet34 bash scripts/run_task2_hparam_sweep.sh
```

## Task 3：预训练消融实验

ImageNet 预训练：

```bash
python scripts/train_task3_ablation.py \
  --model resnet18 \
  --pretrained true \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

随机初始化：

```bash
python scripts/train_task3_ablation.py \
  --model resnet18 \
  --pretrained false \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-3 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

ResNet-34 随机初始化：

```bash
python scripts/train_task3_ablation.py \
  --model resnet34 \
  --pretrained false \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-3 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

## Task 4：注意力机制和 Transformer

SE-ResNet-18：

```bash
python scripts/train_task4_attention.py \
  --model resnet18 \
  --attention se \
  --pretrained true \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

CBAM-ResNet-18：

```bash
python scripts/train_task4_attention.py \
  --model resnet18 \
  --attention cbam \
  --pretrained true \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --backbone_lr 1e-4 \
  --classifier_lr 1e-3 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

SE-ResNet-34 / CBAM-ResNet-34 只需把 `--model` 改为 `resnet34`。

ViT-Tiny：

```bash
python scripts/train_task4_attention.py \
  --model vit_tiny \
  --pretrained true \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

Swin-T：

```bash
python scripts/train_task4_attention.py \
  --model swin_t \
  --pretrained true \
  --data_dir ./data \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --optimizer adamw \
  --num_workers 0 \
  --device mps
```

## 一键运行全部实验

如果想让电脑自动按顺序跑完 ResNet-18、ResNet-34 以及 Task 4 Transformer 对比实验，可以使用：

```bash
bash scripts/run_all_resnet_experiments.sh
```

执行顺序为：

```text
prepare_data
resnet18: task1 -> task2 -> task3 pretrained/scratch -> task4 se/cbam
resnet34: task1 -> task2 -> task3 pretrained/scratch -> task4 se/cbam
task4 transformers: vit_tiny -> swin_t
```

默认使用 Mac 本地参数：

```text
--device mps
--num_workers 0
```

可以在运行时覆盖部分参数：

```bash
EPOCHS=20 BATCH_SIZE=16 bash scripts/run_all_resnet_experiments.sh
```

Task 2 的批量超参数默认写在脚本变量 `TASK2_CONFIGS` 中，格式是：

```text
epochs:lr:batch_size,epochs:lr:batch_size
```

例如：

```bash
TASK2_CONFIGS="30:1e-3:32,30:1e-4:32,30:5e-4:32,30:2e-3:32,10:1e-3:32,50:1e-3:32,30:1e-3:16" bash scripts/run_all_resnet_experiments.sh
```

如果不手动覆盖，Task 2 默认会对 ResNet-18 和 ResNet-34 各跑下面七组：

| 目的 | epochs | classifier_lr | backbone_lr | batch_size |
| --- | --- | --- | --- | --- |
| baseline | 30 | 1e-3 | 1e-4 | 32 |
| lr analysis | 30 | 1e-4 | 1e-5 | 32 |
| lr analysis | 30 | 5e-4 | 5e-5 | 32 |
| lr analysis | 30 | 2e-3 | 2e-4 | 32 |
| epoch analysis | 10 | 1e-3 | 1e-4 | 32 |
| epoch analysis | 50 | 1e-3 | 1e-4 | 32 |
| batch size analysis | 30 | 1e-3 | 1e-4 | 16 |

其中 `backbone_lr = classifier_lr * 0.1`。

ViT-Tiny 和 Swin-T 默认使用：

```text
epochs = EPOCHS
batch_size = 8
lr = 1e-4
```

可以这样调整 Transformer batch size：

```bash
TRANSFORMER_BATCH_SIZE=16 bash scripts/run_all_resnet_experiments.sh
```

如果只想跑 ResNet，不跑 ViT-Tiny 和 Swin-T：

```bash
INCLUDE_TRANSFORMERS=false bash scripts/run_all_resnet_experiments.sh
```

### 后台运行方式

推荐使用 `nohup`，退出终端后任务也会继续运行：

```bash
mkdir -p outputs/logs
nohup bash scripts/run_all_resnet_experiments.sh > outputs/logs/run_all_resnet.log 2>&1 &
```

查看后台日志：

```bash
tail -f outputs/logs/run_all_resnet.log
```

查看任务是否还在运行：

```bash
ps aux | grep run_all_resnet_experiments
```

如果需要停止任务，先找到进程号 PID，然后执行：

```bash
kill PID
```

## 输出文件说明

每个实验目录至少包含：

- `best_model.pth`：验证集 accuracy 最好的模型权重和训练状态
- `last_model.pth`：最后一个 epoch 的模型权重和训练状态
- `train_log.csv`：每个 epoch 的 train/val loss、accuracy、learning rate
- `config.json`：本次实验的参数配置
- `test_result.json`：最佳验证模型在 test split 上的最终结果
- `accuracy_loss_curve.png`：训练/验证 loss 和 accuracy 曲线

如果输出目录已存在且非空，脚本默认会在目录名后追加时间戳，避免覆盖已有结果。若确实要写入同一目录，可添加 `--overwrite`。

## 查看实验结果

训练过程中终端会显示 `tqdm` 进度条和每个 epoch 的指标。训练结束后，可以直接查看：

```bash
cat outputs/task1_resnet18/test_result.json
```

也可以用表格工具打开 `train_log.csv`，或查看 `accuracy_loss_curve.png` 对比不同实验的收敛曲线。
