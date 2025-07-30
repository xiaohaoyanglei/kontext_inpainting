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
```
my_dataset/
├── source_images/    # 原图
│   ├── image1.jpg
│   └── image2.jpg
└── target_images/    # 编辑后的目标图
    ├── image1.jpg
    └── image2.jpg
```

### 3. 训练
```bash
# 使用标准 ai-toolkit 训练脚本
python run.py config/train_kontext_inpaint.yaml

# 或使用专用的两阶段训练脚本
python train_kontext_inpaint.py config/train_kontext_inpaint.yaml
```

### 4. 推理

#### 单次编辑
```bash
python inference_kontext_inpaint.py \
    --model_path "/path/to/trained/model" \
    --source_image "test_image.jpg" \
    --prompt "make the person smile" \
    --output "result.png"
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