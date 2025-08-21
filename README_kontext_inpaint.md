# Kontext-inpaint: 伪 Mask-free 多轮编辑模型

## 🎯 项目目标

基于现有的 **FLUX.1-Fill-dev** 完整权重，结合 **Kontext 架构思想**（Flow-Matching + RoPE），实现"伪 mask-free"原型模型：

- **输入**: 原图RGB(3) + 纯白RGB(3) → 6通道
- **处理**: 经过串行2D VAE编码 → 32通道latent → Kontext Transformer
- **输出**: 3通道RGB编辑结果
- **特点**: 保留Fill的局部修复能力 + Kontext的多轮一致性 + 为SeedVR2风格扩展预留接口

---

## 🏗️ 架构设计

### 数据流设计
```
原图RGB(3) → VAE → 16 latent channels
纯白RGB(3) → VAE → 16 latent channels  
                ↓
        concat → 32 latent channels → Kontext Transformer → 16 output latent → VAE decode → RGB(3)
```

### 投影层初始化
- **32→hidden 投影层**: 前16通道复制原始16→hidden权重，后16通道置零
- **训练策略**: 两阶段fine-tune（非LoRA）

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 确保在 kontext_inpainting 环境中
cd /cloud/cloud-ssd1/kontext_inpainting
source venv_fill/bin/activate  # 或你的环境
```

### 2. 数据准备

推荐使用 `prepare_new_dataset.py` 生成规范的新数据集（保留与 `/cloud/cloud-ssd1/new_dataset` 相关的流程）：

```bash
python /cloud/cloud-ssd1/kontext_inpainting/prepare_new_dataset.py
```

生成后的目录结构：
```
/cloud/cloud-ssd1/new_dataset/
├── source_images/
│   ├── xxx.png         # 源图
│   ├── xxx.txt         # 对应prompt（英文）
└── target_images/
    └── xxx.png         # 目标图
```

### 3. 训练
```bash
# 推荐入口（本项目的训练流程入口）
# jobs/process/TrainFineTuneProcess.py 会被触发用于两阶段训练
python train_kontext_inpaint.py /cloud/cloud-ssd1/kontext_inpainting/config/train_kontext_inpaint.yaml

# 亦可直接使用 ai-toolkit 入口
python run.py /cloud/cloud-ssd1/kontext_inpainting/config/train_kontext_inpaint.yaml
```

快速复制使用（常用一条）：
```bash
python /cloud/cloud-ssd1/kontext_inpainting/run.py /cloud/cloud-ssd1/kontext_inpainting/config/train_kontext_inpaint.yaml
```

训练配置与流程说明：

- 训练流程由 `jobs/process/TrainFineTuneProcess.py` 实现并在运行时由 `train_kontext_inpaint.py` 调用。
- 配置文件 `config/train_kontext_inpaint.yaml` 中 `process: - type: finetune` 会对应到该进程，实现两阶段训练：
  - 阶段1：仅训练 `x_embedder`（32→hidden 投影层），步数由 `stage1_steps` 控制；
  - 阶段2：解冻全模型继续 fine-tune，学习率切换为 `stage2_lr`。

### 4. 推理

#### 单次编辑
```bash
python inference_kontext_inpaint.py \
    --model_path "/path/to/trained/model" \
    --source_image "test_image.jpg" \
    --prompt "make the person smile" \
    --output "result.png"
```

使用 checkpoint 目录（无 model_index.json）的完整推理命令（与训练保持一致的数据流；本项目数据集统一提示词为“add furniture”）：
```bash
python /cloud/cloud-ssd1/kontext_inpainting/inference_kontext_inpaint.py \
  --model_path "/cloud/cloud-ssd1/training_output_YYYYMMDD_HHMMSS/checkpoints/step_00XXXX" \
  --base_model_path "/cloud/cloud-ssd1/FLUX.1-Fill-dev" \
  --source_image "/cloud/cloud-ssd1/test.png" \
  --prompt "add furniture" \
  --output "/cloud/cloud-ssd1/training_output_YYYYMMDD_HHMMSS/infer_add_furniture.png" \
  --steps 30 --guidance 6.0 --seed 42 --size 512
```

#### 多轮编辑
```bash
python inference_kontext_inpaint.py \
    --model_path "/path/to/trained/model" \
    --source_image "test_image.jpg" \
    --multi_round \
    --prompts_file "example_multi_round_prompts.txt" \
    --output_dir "multi_round_results/"
```

---

## 📋 核心组件

### 1. **WhiteMaskDataset** (`toolkit/data_loader.py`)
- 自动生成纯白控制图像 RGB(255,255,255)
- 与现有 ai-toolkit 数据流完全兼容
- 支持三文件夹模式：source + target + (自动生成白色mask)

### 2. **FluxFillInpaintModel** (`extensions_built_in/diffusion_models/flux_fill_inpaint/`)
- 基于 FLUX.1-Fill-dev 完整模型，扩展32通道输入支持
- 智能投影层初始化（前16通道复制Fill权重+后16通道置零）
- 结合 Kontext 的 Flow-Matching + RoPE 实现多轮一致性
- 两阶段训练：第一阶段只训练projection，第二阶段全模型微调

### 3. **两阶段训练配置** (`config/train_kontext_inpaint.yaml`)
```yaml
model:
  arch: "flux_fill_inpaint"
  name_or_path: "/cloud/cloud-ssd1/FLUX.1-Fill-dev"  # 本地路径
  model_kwargs:
    kontext_inpaint_mode: true
    two_stage_training: true
    stage1_steps: 1000  # 第一阶段步数
    stage1_lr: 1e-4     # 第一阶段学习率
    stage2_lr: 5e-5     # 第二阶段学习率
```

配置文件关键路径（已对齐 new_dataset）：

```yaml
process:
  - type: finetune
    source_image_dir: "/cloud/cloud-ssd1/new_dataset/source_images"
    target_image_dir: "/cloud/cloud-ssd1/new_dataset/target_images"
    two_stage_training: true
    stage1_steps: 2000
    stage1_lr: 1e-4
    stage2_lr: 5e-5
```

### 4. **多轮编辑推理** (`inference_kontext_inpaint.py`)
- 支持单次和多轮编辑模式
- 保持Kontext的多轮一致性
- 自动生成纯白控制图像

---

## 🔧 训练流程详解

### 阶段1: Projection层预训练 (前1000步)
- **目标**: 让模型学会利用纯白mask信号
- **冻结**: 整个Transformer主干
- **训练**: 仅32→hidden投影层
- **学习率**: 1e-4

### 阶段2: 全模型微调 (后2000步)  
- **目标**: 优化整体inpainting性能
- **解冻**: 所有模型参数
- **训练**: 投影层 + Transformer主干
- **学习率**: 5e-5

---

## 📊 与其他方案对比

| 特性 | Kontext-inpaint | 标准LoRA微调 | 传统Inpainting |
|------|-----------------|--------------|----------------|
| 输入方式 | 原图+纯白 | 原图+控制图 | 原图+二值mask |
| 用户体验 | 伪mask-free | 需要控制图 | 需要精确mask |
| 多轮一致性 | ✅ | ❌ | ❌ |
| 训练方式 | 全模型微调 | 低秩适配器 | 全模型训练 |
| 扩展能力 | 高(为SeedVR2预留) | 中 | 低 |

---

## 🎨 使用示例

### 多轮编辑工作流
```python
# 示例：人像修饰工作流
prompts = [
    "improve the lighting",           # 第1轮：改善光照
    "make the person smile",          # 第2轮：调整表情  
    "add professional background",    # 第3轮：更换背景
    "enhance skin texture",           # 第4轮：优化细节
    "final color grading"            # 第5轮：最终调色
]

for i, prompt in enumerate(prompts):
    result = kontext_inpaint(current_image, white_mask, prompt)
    current_image = result  # 用于下一轮
```

---

## 🧭 代码与配置路径总览

- 训练入口与作业系统
  - `train_kontext_inpaint.py`：两阶段训练入口（调用标准 job 系统）
  - `run.py`：通用入口（同样读取 YAML 并运行 `TrainJob`）
  - `jobs/TrainJob.py`：加载并调度训练流程（`process: - type: finetune` → `TrainFineTuneProcess`）
  - `jobs/process/BaseTrainProcess.py`：训练过程基类（输出根目录、步数等通用字段）

- 训练流程（核心逻辑）
  - `jobs/process/TrainFineTuneProcess.py`
    - 加载 VAE/Transformer/TextEncoder/T5（路径来自 YAML）
    - 数据加载：使用 `WhiteMaskDataset`（见下）并做 latent 预缓存
    - 输入构建：`in_channels=64`，将“原图latent(16)+纯白latent(16)”拼接并 pad 到 64
    - 文本侧：冻结编码器，`proj_hidden` 投影至 `joint_attention_dim=4096`
    - 两阶段训练：阶段1仅训 `x_embedder`；阶段2解冻全模型
    - 采样与保存：写入 `training_folder/samples` 与 `training_folder/checkpoints/step_xxxxxx`

- 模型实现与架构
  - `extensions_built_in/diffusion_models/flux_fill_inpaint/flux_fill_inpaint.py`：Flux Fill Inpaint 模型扩展（Kontext 化的 Transformer）
  - 关键超参：`joint_attention_dim=4096`、`axes_dims_rope=[64,64]`

- 数据集与数据流
  - `toolkit/data_loader.py`
    - `WhiteMaskDataset`：自动生成纯白控制图（RGB 255），三文件夹模式 source/target
  - `prepare_new_dataset.py`：从 `/cloud/cloud-ssd1/raw_data_test` 生成新数据集到 `/cloud/cloud-ssd1/new_dataset/{source_images,target_images}`

- 配置文件（YAML）
  - `config/train_kontext_inpaint.yaml`
    - 模型路径：`vae_path`、`model_path`、`text_encoder_path`、`text_encoder_2_path`
    - 数据路径：`source_image_dir`、`target_image_dir`（通常为 `/cloud/cloud-ssd1/new_dataset/...`）
    - 训练：`resolution`、`batch_size`、`steps`、`two_stage_training`、`stage1_steps`、`stage1_lr`、`stage2_lr`、`gradient_accumulation_steps`、`device`
    - 输出：`training_folder`（写 `/cloud/cloud-ssd1/training_output` 会被重写为带时间戳）

- 输出目录时间戳逻辑
  - `toolkit/config.py`：在读取 YAML 时，若 `training_folder` 为 `/cloud/cloud-ssd1/training_output`，自动重写为 `/cloud/cloud-ssd1/training_output_YYYYMMDD_HHMMSS`

- 推理与验证
  - `inference_kontext_inpaint.py`：独立推理脚本（单次/多轮）
  - 训练过程内置采样：在 `TrainFineTuneProcess.sample_images` 中，每 1000 步与 step=0 生成对比图

> 修改指引：
> - 想改输入通道或拼接方式：看 `jobs/process/TrainFineTuneProcess.py` 的“构造 model_input”部分（`in_channels=64`、latent 拼接与 pad）
> - 想改文本维度或投影：看同文件中 `proj_hidden`（输出到 `joint_attention_dim`）
> - 想改数据或配对规则：看 `toolkit/data_loader.py` 的 `WhiteMaskDataset` 与 `prepare_new_dataset.py`
> - 想改保存/采样频率或输出路径：看 `TrainFineTuneProcess.save_model/sample_images` 与 YAML 的 `training_folder`

---

## 🔮 未来扩展方向

### SeedVR2 风格集成
- **多通道输入**: RGB + Depth + Normal + Semantic
- **少通道输出**: 只输出需要的编辑通道
- **模块化设计**: 可插拔的通道处理模块

### 高级功能
- **区域感知编辑**: 基于attention map的智能区域定位
- **语义理解增强**: 结合大语言模型的编辑指令理解
- **实时预览**: 基于轻量级预览网络的快速反馈

---

## 📄 许可证

本项目基于原 ai-toolkit 许可证，扩展部分遵循相同协议。

---

## 🤝 贡献

欢迎提交Issue和Pull Request来改进Kontext-inpaint！

**核心开发者**: Kontext-inpaint Team  
**基于**: ai-toolkit + FLUX.1 + Diffusers